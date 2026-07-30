"""System 1 (Generative Executor) inference runtime.

Provides a clean, robot-agnostic inference interface for the PI0 policy:

- ``load_system1_policy``  : load a trained PI0 policy + its pre/post processors.
- ``predict_chunk``        : run one synchronous action-chunk inference.
- ``System1AsyncStreamer`` : the System-1 side of the paper's async inference
  (Algorithm 1).  It prefetches the next action chunk before the current one is
  exhausted (overlap boundary) and falls back to a synchronous call to preserve
  control continuity, with optional future-state awareness.
- ``resolve_policy_ref``   : map a ``Skill.system1_policy_ref`` string (from the
  System-2 skill library) to a checkpoint directory, wiring the two systems.

The streamer is decoupled from any concrete robot or System-2 plumbing: it takes
observations as dicts and emits action chunks/steps as plain numpy arrays, so it
can be driven by a 20 Hz control loop that also polls System 2.
"""

from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "load_system1_policy",
    "predict_chunk",
    "System1AsyncStreamer",
    "resolve_policy_ref",
]


# ----------------------------------------------------------------------- loading
def load_system1_policy(
    pretrained_path: str,
    device: str = "cuda",
    compile_model: Optional[bool] = None,
    compile_mode: Optional[str] = None,
    compile_cache_dir: Optional[str] = None,
):
    """Load a trained System-1 policy together with its pre/post processors.

    The policy type is read from the checkpoint's ``config.json`` and dispatched
    through the factory, so this works for any registered System-1 policy
    (``pi0_system1``, ``act``, ``smolvla``, ``qwen3vl_vla``).

    Args:
        pretrained_path: Directory containing ``model.safetensors`` (+ config and,
            optionally, saved processor pipelines).
        device: Target device string.
        compile_model: Override the checkpoint's saved ``compile_model`` flag
            (PI0 only — ``torch.compile`` of ``sample_actions``). ``None``
            (default) leaves the checkpoint's own setting untouched; ignored by
            policies without these config fields.
        compile_mode: Override the checkpoint's saved ``torch.compile`` mode
            (e.g. ``"max-autotune"``, ``"reduce-overhead"``). ``None`` keeps it.
        compile_cache_dir: Override the checkpoint's saved Inductor cache dir.
            ``None`` keeps it.

    Returns:
        Tuple ``(policy, preprocessor, postprocessor)``.  The processors are
        loaded from the checkpoint if present; otherwise they are rebuilt from the
        policy config (without dataset stats — provide stats for real deployment).
    """
    from low_level_model.models.factory import (
        get_policy_class,
        load_config_local,
        make_pre_post_processors,
    )

    config = load_config_local(pretrained_path)
    # Compile overrides are PI0-specific config fields; apply only where present.
    if compile_model is not None and hasattr(config, "compile_model"):
        config.compile_model = compile_model
    if compile_mode is not None and hasattr(config, "compile_mode"):
        config.compile_mode = compile_mode
    if compile_cache_dir is not None and hasattr(config, "compile_cache_dir"):
        config.compile_cache_dir = compile_cache_dir

    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(pretrained_path, config=config)
    policy.config.device = device
    policy.to(device)
    policy.eval()

    preprocessor = postprocessor = None
    try:
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            policy_action_to_transition,
            transition_to_policy_action,
        )

        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_path,
            config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": device}},
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_path,
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load saved processors (%s); rebuilding from config.", exc)
        preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=None)

    return policy, preprocessor, postprocessor


# ----------------------------------------------------------------------- sync inference
def predict_chunk(
    policy,
    preprocessor,
    postprocessor,
    observation: dict[str, np.ndarray],
    task: str,
) -> np.ndarray:
    """Run one synchronous action-chunk inference.

    Args:
        policy: Loaded PI0 policy.
        preprocessor: Preprocessor pipeline (tokenise task, normalise state).
        postprocessor: Postprocessor pipeline (unnormalise actions).
        observation: Dict of raw observations. Image entries (keys containing
            ``"image"``) are expected as HWC uint8/float arrays; they are
            converted to CHW float in [0, 1].
        task: Natural-language instruction for this chunk (System-2 atomic skill).

    Returns:
        Action chunk as a numpy array ``[n_action_steps, action_dim]``.
    """
    import torch

    # No copy needed: we only read `observation` to build a fresh `batch` dict.
    with torch.inference_mode():
        batch: dict[str, Any] = {}
        for key, val in observation.items():
            val_t = torch.from_numpy(val) if isinstance(val, np.ndarray) else val
            if "image" in key:
                val_t = val_t.float() / 255.0
                val_t = val_t.permute(2, 0, 1)  # HWC -> CHW
            # Batch here (like lerobot's prepare_observation_for_inference) rather
            # than relying on AddBatchDimensionProcessorStep, whose state branch
            # does unsqueeze(1) on a 1-D state ([D] -> [D, 1], batch=D) and breaks
            # the downstream concat.
            val_t = val_t.unsqueeze(0)
            batch[key] = val_t
        batch["task"] = task

        if preprocessor is not None:
            batch = preprocessor(batch)

        action_chunk = policy.predict_action_chunk(batch)

        if postprocessor is not None:
            action_chunk = postprocessor(action_chunk)

    return action_chunk.squeeze(0).detach().cpu().numpy()


# ----------------------------------------------------------------------- async streaming
class System1AsyncStreamer:
    """Asynchronous action-chunk streamer (System-1 side of Algorithm 1).

    Maintains a current action chunk and an in-flight inference future for the
    next chunk.  When the chunk-step index reaches ``n_action_steps - overlap``
    a new chunk is prefetched in a background thread; when the current chunk is
    exhausted the prefetched result is consumed (or a synchronous fallback runs).

    The streamer is robot-agnostic: ``infer_fn`` is an injected callable
    ``(observation, task, future_state) -> np.ndarray`` returning a chunk
    ``[n_action_steps, action_dim]``.  In practice this wraps :func:`predict_chunk`.

    Args:
        infer_fn: Callable producing an action chunk for an observation/task.
        n_action_steps: Number of steps per chunk.
        overlap_steps: Steps before chunk end at which to launch the next inference.
        future_state_aware: If True, the last action of the current chunk is passed
            as ``future_state`` so the next chunk is predicted from where the robot
            *will* be (hides inference latency), matching the paper's design.
        blender: Optional :class:`action_smoothing.TemporalChunkBlender`. When set,
            each incoming chunk is cross-faded into the residual tail of the current
            one (χ₀ temporal chunk-wise smoothing) to remove the inter-chunk
            discontinuity. ``None`` (default) keeps the original hard-switch behaviour.
        rtc_infer_fn: Optional Real-Time Chunking (RTC) inference callable
            ``(observation, task, rtc_kwargs) -> (denorm_chunk, model_space_chunk)``,
            both ``[T, action_dim]`` numpy. ``rtc_kwargs`` carries the normalised
            leftover (``prev_chunk_left_over``), the auto-measured ``inference_delay``
            and the ``execution_horizon``. When set, the streamer runs RTC mode: the
            next chunk is *generated* continuous with the current chunk's leftover.
            Mutually exclusive with ``blender``. Keeps the streamer torch-free (the
            caller's ``rtc_infer_fn`` owns the tensor conversion).
        latency_tracker / fps: required in RTC mode to estimate ``d = ceil(latency*fps)``.
        rtc_execution_horizon / rtc_min_delay / rtc_max_delay: RTC horizon ``s`` and
            bounds on the auto-measured delay ``d`` (``rtc_max_delay=None`` caps at ``s-1``).
    """

    def __init__(
        self,
        infer_fn: Callable[[dict, str, Optional[np.ndarray]], np.ndarray],
        n_action_steps: int,
        overlap_steps: int = 0,
        future_state_aware: bool = True,
        blender=None,
        rtc_infer_fn: Optional[Callable[[dict, str, Optional[dict]], tuple]] = None,
        latency_tracker=None,
        fps: Optional[float] = None,
        rtc_execution_horizon: int = 10,
        rtc_min_delay: int = 1,
        rtc_max_delay: Optional[int] = None,
    ):
        if n_action_steps < overlap_steps:
            raise ValueError(f"n_action_steps ({n_action_steps}) must be >= overlap_steps ({overlap_steps})")
        if overlap_steps < 0:
            raise ValueError(f"overlap_steps ({overlap_steps}) must be non-negative")

        self.rtc_enabled = rtc_infer_fn is not None
        if self.rtc_enabled and blender is not None:
            raise ValueError("RTC and the output-side blender are mutually exclusive; enable only one.")
        if self.rtc_enabled and (latency_tracker is None or fps is None):
            raise ValueError("RTC mode requires both latency_tracker and fps.")

        self.infer_fn = infer_fn
        self.n_action_steps = n_action_steps
        self.overlap_steps = overlap_steps
        # RTC handles latency via the frozen prefix; future-state override would double it.
        self.future_state_aware = future_state_aware and not self.rtc_enabled
        self.blender = blender

        # RTC state / config
        self.rtc_infer_fn = rtc_infer_fn
        self._latency_tracker = latency_tracker
        self._fps = fps
        self._rtc_execution_horizon = rtc_execution_horizon
        self._rtc_min_delay = rtc_min_delay
        self._rtc_max_delay = rtc_max_delay
        self._model_space_chunk: np.ndarray | None = None
        self._steps_since_launch = 0

        self.current_chunk: np.ndarray | None = None
        self._inference_future: Future | None = None
        self.chunk_index = 0
        self._last_action: np.ndarray | None = None  # last returned action (blender seed)
        self._executor = ThreadPoolExecutor(max_workers=1)

    # --- predicates (mirror the paper's polling conditions) ----------------
    def is_running(self) -> bool:
        return (self.current_chunk is not None) or (self._inference_future is not None)

    def should_switch_chunk(self) -> bool:
        return self.chunk_index == 0

    def should_launch_next_inference(self) -> bool:
        return self.chunk_index == self.n_action_steps - self.overlap_steps

    def should_fetch_observation(self) -> bool:
        return (not self.is_running()) or self.should_launch_next_inference()

    # --- core inference launch --------------------------------------------
    def _current_future_state(self) -> Optional[np.ndarray]:
        """Snapshot the future state (last action of the current chunk) to seed
        the next inference. Captured on the *calling* (main) thread so it is not
        affected by ``current_chunk`` being cleared once the chunk is exhausted."""
        if self.future_state_aware and self.current_chunk is not None and len(self.current_chunk) > 0:
            idx = min(self.n_action_steps - 1, len(self.current_chunk) - 1)
            return np.asarray(self.current_chunk[idx]).copy()
        return None

    def _launch(self, observation: dict, task: str, future_state: Optional[np.ndarray]) -> np.ndarray:
        return self.infer_fn(observation, task, future_state)

    # --- main control-loop entry point ------------------------------------
    def get_action(self, observation: dict, task: str) -> np.ndarray:
        """Return the action to execute this step, managing chunk transitions.

        Args:
            observation: Current observation (may be None on steps where
                ``should_fetch_observation`` is False — the streamer reuses the
                in-flight future in that case).
            task: Current atomic instruction from System 2.

        Returns:
            Action vector for this step as a numpy array ``[action_dim]``.
        """
        if self.rtc_enabled:
            return self._get_action_rtc(observation, task)
        if self.blender is not None:
            return self._get_action_blended(observation, task)

        # Bootstrap: first chunk runs synchronously.
        if not self.is_running():
            self.current_chunk = self._launch(observation, task, self._current_future_state())
        # Chunk transition: consume the prefetched next chunk.
        elif self.should_switch_chunk():
            if self._inference_future is not None:
                self.current_chunk = self._inference_future.result()
                self._inference_future = None
            else:
                # No prefetch available -> synchronous fallback for continuity.
                self.current_chunk = self._launch(observation, task, self._current_future_state())

        # Prefetch the next chunk asynchronously at the overlap boundary. The
        # future state is snapshotted now (main thread) before the chunk is cleared.
        if self.should_launch_next_inference() and self._inference_future is None:
            future_state = self._current_future_state()
            self._inference_future = self._executor.submit(self._launch, observation, task, future_state)

        action = self.current_chunk[self.chunk_index]

        # Advance index; clear the chunk when it is exhausted.
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

    def _get_action_blended(self, observation: dict, task: str) -> np.ndarray:
        """Blender path: continuous residual buffer cross-faded with each new chunk
        (χ₀ temporal chunk-wise smoothing). The buffer is variable-length, so
        progress is tracked by ``chunk_index`` vs ``len``."""
        if self.current_chunk is None and self._inference_future is None:
            self.current_chunk = self._launch(observation, task, self._current_future_state())
            self.chunk_index = 0

        # Integrate the prefetch eagerly when ready, and *always* before the buffer
        # runs dry — blocking on the future if necessary. Never launch a second
        # inference while one is in flight (two concurrent forward passes through a
        # torch.compiled CUDA policy deadlock).
        exhausted = self.current_chunk is None or self.chunk_index >= len(self.current_chunk)
        if self._inference_future is not None and (self._inference_future.done() or exhausted):
            new_chunk = self._inference_future.result()
            self._inference_future = None
            self._merge_new_chunk(new_chunk)

        if self.current_chunk is None or self.chunk_index >= len(self.current_chunk):
            self._merge_new_chunk(self._launch(observation, task, self._current_future_state()))

        remaining = len(self.current_chunk) - self.chunk_index
        if self._inference_future is None and remaining <= max(1, self.overlap_steps):
            future_state = self._current_future_state()
            self._inference_future = self._executor.submit(self._launch, observation, task, future_state)

        action = self.current_chunk[self.chunk_index]
        self._last_action = np.asarray(action).copy()  # preserve dtype (no float64 upcast)
        self.chunk_index += 1
        return action

    # --- RTC path ----------------------------------------------------------
    def _launch_rtc(self, observation: dict, task: str, rtc_kwargs: Optional[dict]) -> tuple:
        """Run one RTC inference, measuring its wall-clock latency for the delay
        estimator. Returns ``(denorm_chunk, model_space_chunk)``."""
        started = time.perf_counter()
        result = self.rtc_infer_fn(observation, task, rtc_kwargs)
        self._latency_tracker.add(time.perf_counter() - started)
        return result

    def _rtc_delay_steps(self, execution_horizon: int) -> int:
        """Auto-measured inference delay d (steps), clamped to [min, max] and <= s."""
        latency = self._latency_tracker.max() or 0.0
        d = math.ceil(latency * self._fps) if latency > 0 else self._rtc_min_delay
        d = max(self._rtc_min_delay, d)
        cap = self._rtc_max_delay if self._rtc_max_delay is not None else max(1, execution_horizon - 1)
        return int(min(d, cap, execution_horizon))

    def _make_rtc_kwargs(self) -> Optional[dict]:
        """Snapshot the normalised leftover as the RTC prefix, with the measured d and
        horizon s. None when there is no usable leftover -> unguided sample."""
        if self._model_space_chunk is None:
            return None
        leftover = self._model_space_chunk[self.chunk_index:]
        t_prev = int(leftover.shape[0])
        if t_prev == 0:
            return None
        s = min(self._rtc_execution_horizon, t_prev)
        d = self._rtc_delay_steps(s)
        return {
            "prev_chunk_left_over": np.asarray(leftover).copy(),
            "inference_delay": d,
            "execution_horizon": s,
        }

    def _merge_new_chunk_rtc(self, denorm: np.ndarray, model_space: np.ndarray, drop: int) -> None:
        """Adopt a freshly generated RTC chunk, dropping the ``drop`` leading steps that
        were already executed during inference, and reset the cursor."""
        drop = int(min(max(0, drop), len(denorm) - 1))
        self.current_chunk = denorm[drop:]
        self._model_space_chunk = model_space[drop:]
        self.chunk_index = 0

    def _get_action_rtc(self, observation: dict, task: str) -> np.ndarray:
        """RTC path: keep a continuous buffer whose normalised leftover seeds the next
        chunk's sampler as a frozen prefix, so the new chunk is *generated* continuous.
        On arrival, drop the leading steps executed during inference. Variable-length
        buffer, so progress is tracked by ``chunk_index`` vs ``len``."""
        # Bootstrap: first chunk runs synchronously, unguided (no predecessor).
        if self.current_chunk is None and self._inference_future is None:
            denorm, model_space = self._launch_rtc(observation, task, None)
            self.current_chunk = denorm
            self._model_space_chunk = model_space
            self.chunk_index = 0

        # Integrate the prefetch eagerly when ready, and always before the buffer runs
        # dry. Never launch a second inference while one is in flight.
        exhausted = self.current_chunk is None or self.chunk_index >= len(self.current_chunk)
        if self._inference_future is not None and (self._inference_future.done() or exhausted):
            denorm, model_space = self._inference_future.result()
            self._inference_future = None
            self._merge_new_chunk_rtc(denorm, model_space, drop=self._steps_since_launch)

        # Synchronous fallback (buffer dry, no prefetch pending). Leftover is empty ->
        # unguided sample, drop=0 (the loop was stalled, nothing executed meanwhile).
        if self.current_chunk is None or self.chunk_index >= len(self.current_chunk):
            denorm, model_space = self._launch_rtc(observation, task, self._make_rtc_kwargs())
            self._merge_new_chunk_rtc(denorm, model_space, drop=0)

        # Prefetch once the buffer is down to the overlap window, seeding the sampler
        # with the current normalised leftover as the RTC prefix.
        remaining = len(self.current_chunk) - self.chunk_index
        if self._inference_future is None and remaining <= max(1, self.overlap_steps):
            rtc_kwargs = self._make_rtc_kwargs()
            self._steps_since_launch = 0
            self._inference_future = self._executor.submit(self._launch_rtc, observation, task, rtc_kwargs)

        action = self.current_chunk[self.chunk_index]
        self._last_action = np.asarray(action).copy()
        self.chunk_index += 1
        self._steps_since_launch += 1
        return action

    def reset(self) -> None:
        """Drop the current chunk and any pending inference (e.g. on skill switch)."""
        self.current_chunk = None
        self._inference_future = None
        self.chunk_index = 0
        self._last_action = None
        self._model_space_chunk = None
        self._steps_since_launch = 0
        if self._latency_tracker is not None:
            self._latency_tracker.reset()

    def shutdown(self) -> None:
        """Shut down the background inference thread pool."""
        self._executor.shutdown(wait=True)


# ----------------------------------------------------------------------- System-2 seam
def resolve_policy_ref(ref: str, registry: dict[str, str] | None = None, root: str | None = None) -> str:
    """Resolve a ``Skill.system1_policy_ref`` handle to a checkpoint directory.

    The System-2 skill library stores an opaque ``system1_policy_ref`` per skill
    (e.g. ``"bloodgas/insert_tube"``).  This maps such handles to concrete
    System-1 checkpoint directories.

    Args:
        ref: The opaque policy-reference string from a Skill.
        registry: Optional explicit ``{ref: checkpoint_dir}`` mapping.
        root: Optional root directory; if given and ``registry`` lacks ``ref``,
            ``os.path.join(root, ref)`` is used as the checkpoint directory.

    Returns:
        Path to the checkpoint directory for this skill's System-1 policy.

    Raises:
        KeyError: If the reference cannot be resolved.
    """
    if registry and ref in registry:
        return registry[ref]
    if root is not None:
        candidate = os.path.join(root, ref)
        if os.path.isdir(candidate):
            return candidate
        return candidate  # return path even if not yet present (caller validates)
    raise KeyError(f"Cannot resolve system1_policy_ref '{ref}' (no registry/root provided).")
