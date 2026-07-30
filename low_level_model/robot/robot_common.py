"""Shared robot-control utilities for System 1 (Generative Executor) deployment.

This module collects the pieces reused by both the real-time inference script
(:mod:`robot_inference`) and the DAgger data-collection script
(:mod:`collect_dagger_dataset`):

- :func:`setup_torch_compile_cache` : persistent ``torch.compile`` cache setup.
- :func:`validate_robot_cameras`    : check robot cameras match the policy.
- :func:`load_processor_pipelines`  : load saved pre/post-processor pipelines.
- :func:`build_dataset_features`    : robot hardware features -> dataset features.
- :func:`warmup_policy`             : trigger ``torch.compile`` before the loop.
- :class:`RobotAsyncExecutor`       : System-1 async action-chunk executor that
  prefetches the next chunk at the overlap boundary (future-state aware) and
  emits per-step robot action dicts.  This is the robot-facing counterpart of
  :class:`inference_system1.System1AsyncStreamer`.

Heavy / hardware-only dependencies (``lerobot.robots``, etc.) are imported at
module import time as in the reference deployment code; the policy-side imports
stay local so this module can be imported wherever lerobot is installed.
"""

from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import Robot
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.utils import get_safe_torch_device

logger = logging.getLogger(__name__)

__all__ = [
    "setup_torch_compile_cache",
    "validate_robot_cameras",
    "load_processor_pipelines",
    "build_dataset_features",
    "warmup_policy",
    "warmup_policy_rtc",
    "prompt_to_start",
    "RobotAsyncExecutor",
]


# --------------------------------------------------------------------- compile cache
def setup_torch_compile_cache(cache_dir: str = "./torch_compile_cache") -> None:
    """Configure a persistent on-disk cache for ``torch.compile``.

    Caching the compiled inductor graphs lets subsequent runs skip the expensive
    compilation step. Safe to call even when the policy is not compiled.
    """
    import torch._dynamo
    import torch._inductor

    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    # Parallel kernel compilation. The old hard-coded 4 throttled the one-time
    # max-autotune compile to 4 workers regardless of the box; on a many-core host
    # that dominates warmup time. Use most cores but cap so we stay polite on a
    # shared machine. Set env *and* the live config (config reads env only at import,
    # and setup runs after torch is already imported).
    compile_threads = max(8, min(32, (os.cpu_count() or 8)))
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(compile_threads)
    torch._inductor.config.compile_threads = compile_threads

    torch._inductor.config.fx_graph_cache = True
    # Persist autotune benchmark results under TORCHINDUCTOR_CACHE_DIR so the slow
    # max-autotune kernel search runs once *ever*, not once per process start.
    torch._inductor.config.autotune_local_cache = True
    torch._inductor.config.triton.cudagraphs = False
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.epilogue_fusion = True
    torch._inductor.config.fallback_random = True
    torch._inductor.config.max_autotune_gemm_backends = "ATEN,TRITON"
    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.automatic_dynamic_shapes = True
    torch._dynamo.config.suppress_errors = False

    logger.info("torch.compile cache enabled at %s", os.path.abspath(cache_dir))


# --------------------------------------------------------------------- robot helpers
def validate_robot_cameras(robot: Robot, policy_config: PreTrainedConfig) -> None:
    """Validate that the robot's cameras exactly match the policy image features."""
    robot_image_features = {f"{OBS_IMAGES}.{name}" for name in robot.cameras.keys()}
    policy_image_features = policy_config.image_features
    if not isinstance(policy_image_features, dict):
        raise ValueError(f"Policy image_features must be a dict, got {type(policy_image_features)}")
    if robot_image_features != set(policy_image_features.keys()):
        raise ValueError(
            "Robot camera names must match policy image features!\n"
            f"Robot cameras: {sorted(robot_image_features)}\n"
            f"Policy features: {sorted(policy_image_features.keys())}"
        )


def load_processor_pipelines(pretrained_path: str, device: str):
    """Load the saved pre/post-processor pipelines from a policy checkpoint."""
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.processor.pipeline import DataProcessorPipeline

    preprocessor = DataProcessorPipeline.from_pretrained(
        pretrained_path,
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": device}},
    )
    postprocessor = DataProcessorPipeline.from_pretrained(
        pretrained_path,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def build_dataset_features(robot: Robot) -> dict[str, dict]:
    """Build dataset-style feature definitions from a robot's hardware features."""
    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    return {**action_features, **obs_features}


@torch.inference_mode()
def warmup_policy(policy: PreTrainedPolicy, task: str | None, preprocessor=None, warmup_steps: int = 3) -> None:
    """Warm up a compiled policy by running a few dummy inferences."""
    device = get_safe_torch_device(policy.config.device)
    task = task or ""
    for _ in range(warmup_steps):
        dummy: dict[str, Any] = {}
        tdev = None if preprocessor is not None else device
        for img_key, feat in policy.config.image_features.items():
            c, h, w = feat.shape
            dummy[img_key] = torch.zeros((1, c, h, w), dtype=torch.float32, device=tdev)
        if "observation.state" in policy.config.input_features:
            sd = policy.config.input_features["observation.state"].shape[0]
            dummy["observation.state"] = torch.zeros((1, sd), dtype=torch.float32, device=tdev)
        dummy["task"] = task
        batch = preprocessor(dummy) if preprocessor is not None else dummy
        _ = policy.predict_action_chunk(batch)
    logger.info("Policy warmup complete (%d steps)", warmup_steps)


def _build_dummy_batch(policy: PreTrainedPolicy, task: str | None, preprocessor, device):
    """Build a single dummy, preprocessed inference batch (shared by the warmups)."""
    dummy: dict[str, Any] = {}
    tdev = None if preprocessor is not None else device
    for img_key, feat in policy.config.image_features.items():
        c, h, w = feat.shape
        dummy[img_key] = torch.zeros((1, c, h, w), dtype=torch.float32, device=tdev)
    if "observation.state" in policy.config.input_features:
        sd = policy.config.input_features["observation.state"].shape[0]
        dummy["observation.state"] = torch.zeros((1, sd), dtype=torch.float32, device=tdev)
    dummy["task"] = task or ""
    return preprocessor(dummy) if preprocessor is not None else dummy


def warmup_policy_rtc(policy: PreTrainedPolicy, task: str | None, preprocessor,
                      execution_horizon: int, inference_delay: int, warmup_steps: int = 2) -> None:
    """Warm up the RTC *guided* path so its grad+backward graph compiles here (before
    the control loop), not on the first prefetch mid-episode where it would stall the
    robot. Uses ``torch.no_grad`` (NOT ``inference_mode``) so the RTC processor can
    re-enable grad — mirroring :meth:`RobotAsyncExecutor._infer_rtc`. The dummy prefix
    only needs to make guidance active; its exact length/dim don't affect the compiled
    ``denoise_step`` guard (which keys on ``x_t``/timestep, not the prefix)."""
    device = get_safe_torch_device(policy.config.device)
    n = policy.config.n_action_steps
    adim = getattr(policy.config, "max_action_dim", None) or policy.config.action_feature.shape[0]
    s = max(1, min(execution_horizon, n))
    prev = torch.zeros((1, s, adim), dtype=torch.float32, device=device)
    # NOTE: no_grad, not inference_mode — the runtime RTC path (_infer_rtc) uses no_grad
    # so the RTC processor can re-enable grad; warming under inference_mode would compile
    # a variant with a different guard that the loop then recompiles on its critical path.
    with torch.no_grad():
        for _ in range(warmup_steps):
            batch = _build_dummy_batch(policy, task, preprocessor, device)
            # Unguided variant (matches the RTC bootstrap / prev=None first chunk).
            _ = policy.predict_action_chunk(batch)
            # Guided variant (compiles the grad+backward graph for the ΠGDM guidance).
            _ = policy.predict_action_chunk(
                batch, prev_chunk_left_over=prev,
                inference_delay=inference_delay, execution_horizon=s,
            )
    logger.info("RTC warmup complete (unguided + guided, %d steps)", warmup_steps)


# --------------------------------------------------------------------- start gate
def prompt_to_start(message: str = "Press ENTER to start") -> None:
    """Block until the operator confirms, so the robot never starts moving the
    instant the script launches — giving time to clear the workspace, position
    the arm, and arm the e-stop.

    Falls back to a no-op when stdin is not an interactive terminal (e.g. when
    launched from a non-TTY context), so unattended runs are not deadlocked.
    """
    import sys
    if not sys.stdin or not sys.stdin.isatty():
        logger.info("stdin is not a TTY; skipping start confirmation, beginning immediately.")
        return
    try:
        input(f"\n>>> {message} (Ctrl-C to abort) <<<\n")
    except EOFError:
        # No interactive input available — proceed rather than hang.
        pass

class RobotAsyncExecutor:
    """System-1 asynchronous action-chunk executor for a real robot.

    Mirrors the paper's Algorithm 1: while the current chunk is being executed,
    the next chunk is prefetched in a background thread at the overlap boundary,
    conditioned on the *future* state (the last action of the current chunk) so
    that inference latency is hidden.  The current task string is pulled fresh on
    every inference via ``task_provider`` so System-2 skill switches take effect
    seamlessly at the next chunk boundary.

    Args:
        policy: Trained PI0 policy.
        robot: Robot instance (provides ``action_features`` key order).
        task_provider: Callable returning the current task/skill instruction.
        overlap_steps: Steps before chunk end at which to prefetch the next chunk.
        preprocessor: Pre-processing pipeline (tokenise task, normalise state).
        postprocessor: Post-processing pipeline (unnormalise actions).
        future_state_aware: Whether to condition the next chunk on the predicted
            future state (last action of the current chunk).
        blender: Optional :class:`~low_level_model.runtime.action_smoothing.TemporalChunkBlender`.
            When set, the executor cross-fades each incoming chunk into the residual
            tail of the current one (χ₀ temporal chunk-wise smoothing) to remove the
            inter-chunk discontinuity. ``None`` (default) keeps the original
            hard-switch behaviour unchanged.
        rtc_enabled: Enable Real-Time Chunking (RTC) mode. Instead of blending chunks
            on the output, RTC feeds the *normalised* leftover of the current chunk
            back into the flow-matching sampler as a frozen prefix, so the next chunk
            is *generated* continuous with it (ΠGDM inpainting). Mutually exclusive
            with ``blender``.
        latency_tracker: ``lerobot.policies.rtc.latency_tracker.LatencyTracker`` used
            (in RTC mode) to estimate the inference delay ``d = ceil(latency * fps)``.
        fps: Control-loop rate, needed to convert measured latency (s) to steps.
        rtc_execution_horizon: RTC execution horizon ``s`` (overlap region the sampler
            regenerates freshly); clamped to the available leftover length.
        rtc_min_delay / rtc_max_delay: Bounds on the auto-measured inference delay ``d``
            (steps). ``rtc_max_delay=None`` caps at ``s-1``.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        robot: Robot,
        task_provider: Callable[[], Optional[str]],
        overlap_steps: int = 0,
        preprocessor=None,
        postprocessor=None,
        future_state_aware: bool = True,
        subtask_provider: Optional[Callable[[], Optional[str]]] = None,
        blender=None,
        rate_limiter=None,
        rtc_enabled: bool = False,
        latency_tracker=None,
        fps: Optional[float] = None,
        rtc_execution_horizon: int = 10,
        rtc_min_delay: int = 1,
        rtc_max_delay: Optional[int] = None,
    ):
        self.policy = policy
        self.robot = robot
        self.task_provider = task_provider
        self.subtask_provider = subtask_provider
        self.overlap_steps = overlap_steps
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.future_state_aware = future_state_aware and not rtc_enabled
        self.blender = blender
        # Pure output-side governor; applied to every emitted command in _emit. Decoupled
        # from the blender/future_state so the smoothing stages don't feed back into the model.
        self.rate_limiter = rate_limiter

        self.n_action_steps = policy.config.n_action_steps
        self.device = get_safe_torch_device(policy.config.device)
        self._action_keys = list(robot.action_features.keys())

        if rtc_enabled and blender is not None:
            raise ValueError("RTC and the output-side blender are mutually exclusive; enable only one.")
        if self.n_action_steps < self.overlap_steps:
            raise ValueError(f"n_action_steps ({self.n_action_steps}) must be >= overlap_steps ({self.overlap_steps})")
        if self.overlap_steps < 0:
            raise ValueError(f"overlap_steps ({self.overlap_steps}) must be non-negative")
        if rtc_enabled and (latency_tracker is None or fps is None):
            raise ValueError("RTC mode requires both latency_tracker and fps.")

        # RTC state / config
        self.rtc_enabled = rtc_enabled
        self._latency_tracker = latency_tracker
        self._fps = fps
        self._rtc_execution_horizon = rtc_execution_horizon
        self._rtc_min_delay = rtc_min_delay
        self._rtc_max_delay = rtc_max_delay
        self._model_space_chunk: torch.Tensor | None = None
        self._steps_since_launch = 0  # control steps elapsed since the in-flight infer launched
        self._rtc_underruns = 0       # count of buffer underruns (arm pauses) for tuning

        self.current_chunk: np.ndarray | None = None
        self._inference_future: Future | None = None
        self.chunk_index = 0
        self._last_action: np.ndarray | None = None  # last published action (blender seed)
        self._executor = ThreadPoolExecutor(max_workers=1)

    # --- polling predicates ------------------------------------------------
    def is_running(self) -> bool:
        return (self.current_chunk is not None) or (self._inference_future is not None)

    def should_switch_chunk(self) -> bool:
        return self.chunk_index == 0

    def should_launch_next_inference(self) -> bool:
        return self.chunk_index == self.n_action_steps - self.overlap_steps

    def should_fetch_observation(self) -> bool:
        return (not self.is_running()) or self.should_launch_next_inference()

    # --- inference ---------------------------------------------------------
    def _future_state(self) -> Optional[np.ndarray]:
        """Snapshot the future state on the calling thread (avoids the chunk
        being cleared underneath the background worker). Uses the last action of the
        current buffer; the buffer may be variable-length when a blender is active."""
        if self.future_state_aware and self.current_chunk is not None and len(self.current_chunk) > 0:
            idx = min(self.n_action_steps - 1, len(self.current_chunk) - 1)
            return np.asarray(self.current_chunk[idx]).copy()
        return None

    def _build_batch(self, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        """Build a preprocessed, batched policy input from a raw observation frame."""
        task = self.task_provider()
        subtask = self.subtask_provider() if self.subtask_provider is not None else None
        batch: dict[str, Any] = {}
        for key, val in observation.items():
            val_t = torch.from_numpy(val) if isinstance(val, np.ndarray) else val
            if "image" in key:
                val_t = val_t.float() / 255.0
                val_t = val_t.permute(2, 0, 1)  # HWC -> CHW
            val_t = val_t.unsqueeze(0)
            batch[key] = val_t
        batch["task"] = task
        if subtask:
            batch["subtask"] = subtask
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        return batch

    def _infer(self, observation: dict[str, np.ndarray], future_state: Optional[np.ndarray]) -> np.ndarray:
        if future_state is not None:
            observation = dict(observation)
            prev_state = observation.get("observation.state")
            future_state = np.asarray(future_state)
            if isinstance(prev_state, np.ndarray):
                future_state = future_state.astype(prev_state.dtype, copy=False)
            observation["observation.state"] = future_state

        with torch.inference_mode():
            batch = self._build_batch(observation)
            action_chunk = self.policy.predict_action_chunk(batch)
            if self.postprocessor is not None:
                action_chunk = self.postprocessor(action_chunk)
        return action_chunk.squeeze(0).detach().cpu().numpy()

    def _infer_rtc(
        self, observation: dict[str, np.ndarray], rtc_kwargs: Optional[dict]
    ) -> tuple[np.ndarray, torch.Tensor]:
        """RTC inference: return (denormalised chunk [T, A] numpy, normalised chunk
        [T, A] torch on device). Runs under ``torch.no_grad`` (NOT ``inference_mode``)
        so the RTC processor can locally re-enable grad for its ΠGDM guidance."""
        started = time.perf_counter()
        # no_grad (overridable by RTCProcessor's enable_grad); inference_mode is NOT.
        with torch.no_grad():
            batch = self._build_batch(observation)
            action_chunk = self.policy.predict_action_chunk(batch, **(rtc_kwargs or {}))
            normalised = action_chunk.detach().clone()  # keep model-space copy for next leftover
            if self.postprocessor is not None:
                action_chunk = self.postprocessor(action_chunk)
        # LatencyTracker is thread-safe enough for our purpose (single-writer here).
        self._latency_tracker.add(time.perf_counter() - started)
        denorm = action_chunk.squeeze(0).detach().cpu().numpy()
        return denorm, normalised.squeeze(0)

    def _rtc_delay_steps(self, execution_horizon: int) -> int:
        """Auto-measured inference delay d (steps), clamped to [min, max] and <= s."""
        latency = self._latency_tracker.max() or 0.0
        d = math.ceil(latency * self._fps) if latency > 0 else self._rtc_min_delay
        d = max(self._rtc_min_delay, d)
        cap = self._rtc_max_delay if self._rtc_max_delay is not None else max(1, execution_horizon - 1)
        return int(min(d, cap, execution_horizon))

    def _make_rtc_kwargs(self) -> Optional[dict]:
        """Snapshot the normalised leftover of the current chunk as the RTC prefix,
        with the auto-measured delay d and execution horizon s. None when there is no
        usable leftover (first chunk / exhausted buffer) -> unguided sample."""
        if self._model_space_chunk is None:
            return None
        leftover = self._model_space_chunk[self.chunk_index:]
        t_prev = int(leftover.shape[0])
        if t_prev == 0:
            return None
        s = min(self._rtc_execution_horizon, t_prev)
        d = self._rtc_delay_steps(s)
        return {
            "prev_chunk_left_over": leftover.detach().to(torch.float32),
            "inference_delay": d,
            "execution_horizon": s,
        }

    def _merge_new_chunk_rtc(self, denorm: np.ndarray, normalised: torch.Tensor, drop: int) -> None:
        """Adopt a freshly generated RTC chunk, dropping the ``drop`` leading steps that
        were already executed during inference, and reset the cursor."""
        drop = int(min(max(0, drop), len(denorm) - 1))
        self.current_chunk = denorm[drop:]
        self._model_space_chunk = normalised[drop:]
        self.chunk_index = 0

    def _run(self, observation: dict[str, np.ndarray], future_state: Optional[np.ndarray]) -> np.ndarray:
        """Run one inference on the dedicated worker thread (blocking).

        All policy inference — bootstrap, sync fallback, and prefetch — goes through
        the single ``_executor`` worker so the ``torch.compile``d model is only ever
        invoked from the one thread that warmed/compiled it. Calling a compiled CUDA
        model from a different thread than the one that compiled it can deadlock.
        """
        return self._executor.submit(self._infer, observation, future_state).result()

    def _run_rtc(
        self, observation: dict[str, np.ndarray], rtc_kwargs: Optional[dict]
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Run one RTC inference on the dedicated worker thread (blocking). Like
        :meth:`_run`, all policy calls go through the single worker so a compiled CUDA
        model is only ever invoked from the thread that warmed/compiled it."""
        return self._executor.submit(self._infer_rtc, observation, rtc_kwargs).result()

    def warmup(self, task: str | None, preprocessor=None, warmup_steps: int = 3) -> None:
        """Warm up (and compile) the policy ON the worker thread that will run every
        inference, so the compiled graph is owned by that thread (see :meth:`_run`).
        In RTC mode, warms the unguided *and* guided paths under no_grad (matching the
        runtime RTC context) so both graphs — including the guided grad+backward — compile
        here, not on the control loop's critical path (which would freeze the robot)."""
        if self.rtc_enabled:
            self._executor.submit(
                warmup_policy_rtc, self.policy, task, preprocessor,
                self._rtc_execution_horizon, self._rtc_min_delay,
            ).result()
        else:
            self._executor.submit(warmup_policy, self.policy, task, preprocessor, warmup_steps).result()

    def _emit(self, values) -> dict[str, float]:
        """Apply the optional publish-time rate limiter and pack into a robot action dict.

        The rate limiter is a pure output governor: it clamps the per-step joint delta of
        the *commanded* stream (bounding velocity so Piper MIT mode cannot reproduce a
        chunk-boundary jump as a jerk). It is intentionally the last stage and does not
        feed back into ``_last_action`` (blender seed) or ``_future_state``."""
        values = np.asarray(values)
        if self.rate_limiter is not None:
            values = self.rate_limiter(values)
        return dict(zip(self._action_keys, values.tolist()))

    def get_action(self, observation_frame: dict) -> dict[str, float]:
        """Return the next per-step robot action, managing chunk transitions."""
        if observation_frame is None:
            raise ValueError("observation_frame cannot be None")
        if self.rtc_enabled:
            return self._get_action_rtc(observation_frame)
        if self.blender is not None:
            return self._get_action_blended(observation_frame)

        if not self.is_running():
            self.current_chunk = self._run(observation_frame, self._future_state())
        elif self.should_switch_chunk():
            if self._inference_future is not None:
                try:
                    self.current_chunk = self._inference_future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error("Async inference failed: %s", exc, exc_info=True)
                    self.current_chunk = None
                self._inference_future = None
            if self.current_chunk is None:
                self.current_chunk = self._run(observation_frame, self._future_state())

        # Prefetch the next chunk at the overlap boundary.
        if self.should_launch_next_inference() and self._inference_future is None:
            future_state = self._future_state()
            self._inference_future = self._executor.submit(self._infer, observation_frame, future_state)

        values = self.current_chunk[self.chunk_index]
        action = self._emit(values)

        self.chunk_index = (self.chunk_index + 1) % self.n_action_steps
        if self.chunk_index == 0:
            self.current_chunk = None
        return action

    def _merge_new_chunk(self, new_chunk: np.ndarray) -> None:
        """Cross-fade ``new_chunk`` into the residual tail and reset the cursor."""
        old_remaining = (self.current_chunk[self.chunk_index:]
                         if self.current_chunk is not None else None)
        self.current_chunk = self.blender.merge(
            old_remaining, new_chunk, consumed=self.overlap_steps, last_action=self._last_action)
        self.chunk_index = 0

    def _get_action_blended(self, observation_frame: dict) -> dict[str, float]:
        """Blender path: keep a continuous residual buffer and cross-fade each new
        chunk into its tail (χ₀ temporal chunk-wise smoothing). The buffer is
        variable-length, so progress is tracked by ``chunk_index`` vs ``len``."""
        # Bootstrap: first chunk runs synchronously (no predecessor to blend).
        if self.current_chunk is None and self._inference_future is None:
            self.current_chunk = self._run(observation_frame, self._future_state())
            self.chunk_index = 0

        # Integrate the prefetched chunk into the residual tail. Eagerly once the
        # future is ready (so the cross-fade overlaps the old tail), and *always*
        # before the buffer runs dry — blocking on the future if necessary. We must
        # never launch a second inference while one is in flight: two concurrent
        # forward passes through a torch.compiled CUDA policy deadlock.
        exhausted = self.current_chunk is None or self.chunk_index >= len(self.current_chunk)
        if self._inference_future is not None and (self._inference_future.done() or exhausted):
            try:
                new_chunk = self._inference_future.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("Async inference failed: %s", exc, exc_info=True)
                new_chunk = None
            self._inference_future = None
            if new_chunk is not None:
                self._merge_new_chunk(new_chunk)

        # Still exhausted with no prefetch pending -> synchronous fallback (the only
        # place a blocking inference runs, and only when no future is in flight).
        if self.current_chunk is None or self.chunk_index >= len(self.current_chunk):
            self._merge_new_chunk(self._run(observation_frame, self._future_state()))

        # Prefetch the next chunk once the residual tail is down to the overlap window.
        remaining = len(self.current_chunk) - self.chunk_index
        if self._inference_future is None and remaining <= max(1, self.overlap_steps):
            future_state = self._future_state()
            self._inference_future = self._executor.submit(self._infer, observation_frame, future_state)

        values = self.current_chunk[self.chunk_index]
        self._last_action = np.asarray(values).copy()  # preserve dtype (no float64 upcast)
        self.chunk_index += 1
        return self._emit(values)

    def _get_action_rtc(self, observation_frame: dict) -> dict[str, float]:
        """RTC path: keep a continuous buffer whose leftover (in normalised model
        space) seeds the next chunk's flow-matching sampler as a frozen prefix, so the
        new chunk is *generated* continuous with what has already been committed. On
        arrival we drop the leading steps that were executed during inference. The
        buffer is variable-length, so progress is tracked by ``chunk_index`` vs ``len``.
        """
        # Bootstrap: first chunk runs synchronously, unguided (no predecessor).
        if self.current_chunk is None and self._inference_future is None:
            denorm, normalised = self._run_rtc(observation_frame, None)
            self.current_chunk = denorm
            self._model_space_chunk = normalised
            self.chunk_index = 0

        exhausted = self.current_chunk is None or self.chunk_index >= len(self.current_chunk)
        if self._inference_future is not None and (self._inference_future.done() or exhausted):
            blocked = exhausted and not self._inference_future.done()
            try:
                result = self._inference_future.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("Async RTC inference failed: %s", exc, exc_info=True)
                result = None
            self._inference_future = None
            if result is not None:
                denorm, normalised = result
                # Drop the steps executed since this inference launched (frozen prefix).
                self._merge_new_chunk_rtc(denorm, normalised, drop=self._steps_since_launch)
            if blocked:
                self._note_rtc_underrun()

        # Buffer empty with no prefetch even pending -> synchronous, unguided inference
        # (also a stall). Only fires when overlap_steps is too small / inference too slow.
        if self.current_chunk is None or self.chunk_index >= len(self.current_chunk):
            denorm, normalised = self._run_rtc(observation_frame, self._make_rtc_kwargs())
            self._merge_new_chunk_rtc(denorm, normalised, drop=0)
            self._note_rtc_underrun()

        # Prefetch the next chunk once the buffer is down to the overlap window, seeding
        # the sampler with the current (normalised) leftover as the RTC prefix.
        remaining = len(self.current_chunk) - self.chunk_index
        if self._inference_future is None and remaining <= max(1, self.overlap_steps):
            rtc_kwargs = self._make_rtc_kwargs()
            self._steps_since_launch = 0
            self._inference_future = self._executor.submit(self._infer_rtc, observation_frame, rtc_kwargs)

        values = self.current_chunk[self.chunk_index]
        self._last_action = np.asarray(values).copy()
        self.chunk_index += 1
        self._steps_since_launch += 1
        return self._emit(values)

    def _note_rtc_underrun(self) -> None:
        """Record + throttled-warn on an RTC buffer underrun (the arm paused waiting for
        a chunk). The message reports the measured inference latency and a concrete
        ``overlap_steps`` target so the run can be tuned from real numbers."""
        self._rtc_underruns += 1
        if self._rtc_underruns == 1 or self._rtc_underruns % 20 == 0:
            lat = self._latency_tracker.max() if self._latency_tracker is not None else None
            d = math.ceil(lat * self._fps) if (lat and self._fps) else None
            logger.warning(
                "RTC buffer underrun x%d: inference (~%s ms ≈ %s steps) outran the prefetch "
                "runway (overlap_steps=%d, n_action_steps=%d). Raise overlap_steps toward ~%s "
                "and/or lower the policy num_inference_steps to stop the arm pausing.",
                self._rtc_underruns,
                f"{lat * 1000:.0f}" if lat else "?", d if d is not None else "?",
                self.overlap_steps, self.n_action_steps,
                (d + 2) if d is not None else "d+2",
            )

    def reset(self) -> None:
        self.current_chunk = None
        self._inference_future = None
        self.chunk_index = 0
        self._last_action = None
        if self.rate_limiter is not None:
            self.rate_limiter.reset()
        self._model_space_chunk = None
        self._steps_since_launch = 0
        self._rtc_underruns = 0
        if self._latency_tracker is not None:
            self._latency_tracker.reset()

    def shutdown(self) -> None:
        # Do NOT block on a possibly-wedged inference: cancel what we can and return
        # so the caller's cleanup (robot.disconnect) still runs and the arm is safed.
        self._executor.shutdown(wait=False, cancel_futures=True)
