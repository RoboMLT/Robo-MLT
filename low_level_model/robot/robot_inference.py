"""Real-time robot control: System 1 (Executor) + System 2 (Reasoner).

This is the deployment entry point for the Robo-MLT dual-system framework on a
real robot.  System 1 (the PI0 flow-matching policy) generates high-frequency
action chunks for the *current* atomic skill, while System 2 (the planner +
completion gate + pipeline controller from ``high_level_model/``) watches the
camera stream at a lower rate and decides when to advance to the next skill or
trigger a recovery/replan.

Architecture::

    robot ──obs(20 Hz)──> RobotAsyncExecutor ──action chunk──> robot
                               │ task = System2Controller.current_task()
    robot ──obs(sampled)──> System2Controller (planner + gate + pipeline)
                               └─ updates the current skill in the background

The System-2 inference runs in a background thread (sampled every N frames) so
it never blocks the 20 Hz control loop; the new skill instruction is picked up
by the executor at the next chunk boundary, giving a seamless hand-off.

Keyboard (requires pynput / a display):
    1-9     manually override System 2: jump to the N-th skill of the library
    0       print the current plan / pointer status
    q, ESC  stop the control loop

Configuration (same style as ``train_system1.py``): all parameters live in a
YAML file; individual fields can be overridden with ``key=value`` (dot-notation
for nested keys)::

    python -m low_level_model.robot.robot_inference \
        configs/system1/inference/inference.yaml

    # override fields
    python -m low_level_model.robot.robot_inference \
        configs/system1/inference/inference.yaml \
        robot.port=/dev/ttyACM1 use_llm=true overlap_steps=6

The plain draccus dotlist CLI also still works::

    python -m low_level_model.robot.robot_inference \
        --robot.type=so101_follower --robot.port=/dev/ttyACM0 \
        --robot.cameras="{cam_top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --policy_path=outputs/system1/model_best \
        --task="Pick up the green tube, analyze it, and return it to the rack." \
        --skill_library=configs/skill_library/bloodgas.yaml \
        --gate_ckpt=high_level_model/outputs/completion_gate/completion_gate_best.pth \
        --camera_names='[observation.images.cam_top]' \
        --use_llm=true --llm_model=qwen-plus \
        --fps=20 --control_time_s=600
"""

from __future__ import annotations

import glob
import logging
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
import yaml

from lerobot.robots import RobotConfig

# Importing the robot subpackages registers their RobotConfig subclasses with
# draccus' ChoiceRegistry so `--robot.type=...` resolves (lerobot pattern).
from lerobot.robots import (  # noqa: F401
    bi_so100_follower,
    dual_piper,
    koch_follower,
    lekiwi,
    piper,
    so100_follower,
    so101_follower,
)

from low_level_model.robot.keyboard_control import drain_keys, start_keyboard_listener
from low_level_model.robot.robot_common import (
    RobotAsyncExecutor,
    build_dataset_features,
    prompt_to_start,
)
from low_level_model.robot.runtime_builder import (
    RateLimitConfig,
    SmoothingConfig,
    build_action_blender,
    build_rate_limiter,
    load_robot_policy_runtime,
)
from low_level_model.robot.status_view import StatusView
from low_level_model.robot.task_manager import SkillTaskManager, digit_of
from low_level_model.utils.config_loader import parse_with_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- config
@dataclass
class PipelineConfig:
    """System-2 pointer-controller hyperparameters.

    Forwarded verbatim (``**pipeline_kwargs``) to
    :class:`high_level_model.planning.system2_pipeline.System2Pipeline`. See that
    class for the exact semantics; the defaults here mirror its own defaults.
    Every field MUST be a valid ``System2Pipeline.__init__`` argument, since the
    whole dataclass is splatted in via ``**asdict(cfg.pipeline)``.
    """

    # --- advance (completion head): advance after k_a consecutive ticks with
    # filtered completion_prob >= tau_done, but never within `cooldown` ticks of a
    # switch. This is the signal that actually drives subtask completion.
    tau_done: float = 0.6
    k_a: int = 2
    cooldown: int = 10
    # --- recover (back head): recover after k_b consecutive ticks with filtered
    # back_prob >= tau_back.
    tau_back: float = 0.8
    k_b: int = 3
    # replan_budget caps the number of LLM replans per episode.
    replan_budget: int = 3
    # --- output-side de-jitter of the raw gate signals
    # (high_level_model.planning.signal_filters.ProgressSignalFilter), applied before
    # the k-of-k debounce above. Both must be able to drop (no monotone clamp), so
    # they are EMA-smoothed only. Set ema_alpha=1.0 and median_window=1 to disable and
    # recover the exact paper Eq. 4-5 behaviour.
    back_ema_alpha: float = 0.5
    completion_median_window: int = 3
    completion_ema_alpha: float = 0.5


@dataclass
class RTCConfig:
    """Real-Time Chunking (RTC) inference smoothing (Black 2025, arXiv 2506.07339).

    **Extension beyond the Robo-MLT paper.** The paper's asynchronous runtime (Algorithm 1)
    uses only the output-side cross-fade described by :class:`SmoothingConfig`
    (:class:`~low_level_model.runtime.action_smoothing.TemporalChunkBlender`, blend window
    ``w``). RTC is an optional, disabled-by-default alternative kept for experimentation; the
    reported results do not use it.

    An *inference-time* alternative to :class:`SmoothingConfig`. Instead of blending
    chunks on the output, RTC feeds the normalised leftover of the current chunk back
    into the flow-matching sampler as a frozen prefix and generates the next chunk to
    be continuous with it (ΠGDM inpainting with a soft prefix mask). Training-free;
    supported for flow-matching policies (PI0, SmolVLA). Reuses lerobot's
    ``RTCProcessor``. **Mutually exclusive with** ``smoothing.enabled``.
    """

    enabled: bool = False
    # Soft-mask schedule over the overlap region: "exp" (paper-recommended),
    # "linear", "zeros" (hard mask), or "ones".
    prefix_attention_schedule: str = "exp"
    # Guidance-weight clip β (stabilises ΠGDM under few denoising steps).
    max_guidance_weight: float = 10.0
    # Execution horizon s: overlap region the sampler regenerates freshly (clamped to
    # the available leftover length at runtime).
    execution_horizon: int = 10
    # Sliding window (samples) for the latency estimator (d = ceil(max_latency * fps)).
    delay_window: int = 50
    # Bounds on the auto-measured inference delay d (steps). max=None caps at s-1.
    min_inference_delay: int = 1
    max_inference_delay: Optional[int] = None
    # Record per-step denoising traces (lerobot RTC debug tracker).
    debug: bool = False


@dataclass
class InferenceConfig:
    """Run config. Loaded from a YAML file (+ ``key=value`` overrides) or a
    draccus dotlist CLI — see the module docstring and
    ``configs/system1/inference/inference.yaml``."""

    robot: RobotConfig
    policy_path: str          # System-1 (PI0) checkpoint directory
    task: str                 # High-level task description
    device: str = "cuda:1"
    fps: int = 20
    control_time_s: float = 600.0
    # Chunk overlap for async prefetch; 0 disables prefetching entirely (every
    # chunk boundary becomes a synchronous inference). In RTC mode this is also the
    # prefetch *runway*: set it >= the RTC inference delay in steps (ceil(latency*fps))
    # to keep the action buffer from underrunning (arm pausing).
    overlap_steps: int = 4
    # Override the flow-matching denoising steps at inference (None = keep the
    # checkpoint's value). Fewer steps = lower latency (RTC cost scales ~linearly with
    # this); the main knob to fit the real-time budget. Typical: 10 -> 5.
    num_inference_steps: Optional[int] = None
    compile_model: Optional[bool] = None
    compile_mode: Optional[str] = None          # e.g. "max-autotune", "reduce-overhead"
    compile_cache_dir: Optional[str] = None
    # System 2 (optional; both gate_ckpt and skill_library enable it). No dataset is needed to
    # build the System-2 backbone (a frozen SigLIP2 encoder) — only the trained gate checkpoint.
    skill_library: Optional[str] = None   # Skill-library YAML
    gate_ckpt: Optional[str] = None       # Completion-gate checkpoint
    camera_names: list[str] = field(default_factory=lambda: ["observation.images.cam_top"])
    history_len: int = None
    history_skip_frame: int = None
    sampling_interval: int = 15
    cooldown_s: float = 1.0
    # System-2 pointer-controller hyperparameters (tau_done, k_a, signal filters, …).
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    # System-1 output-side action-chunk smoothing (anti-jitter; opt-in).
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    # Publish-time per-joint rate limiter (anti-jerk governor; opt-in). Composes with
    # smoothing/RTC — it is a pure output-side clamp on the commanded joint velocity.
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    # System-1 Real-Time Chunking (inference-time smoothing; opt-in, flow policies
    # only). Mutually exclusive with smoothing.
    rtc: RTCConfig = field(default_factory=RTCConfig)
    # LLM replanning (optional; requires DASHSCOPE_API_KEY)
    use_llm: bool = False
    llm_model: str = "qwen-plus"
    # Planner prompt template file (configs/skill_library/prompts/*.yaml). Empty = built-in default.
    planner_prompt: str = ""
    # Live status window (OpenCV): shows observation images + System-2 plan/stage overlay.
    enable_view: bool = False
    view_bgr: bool = True               # convert RGB->BGR for cv2 display (RealSense frames are RGB)
    view_max_width: int = 1280          # max width of the tiled camera strip (downscaled if larger)


GATE_SIDECAR_NAME = "completion_gate_config.yaml"


def _resolve_gate_ckpt(gate_ckpt: str) -> tuple[str, dict]:
    """Resolve ``gate_ckpt`` (a run folder *or* a ``.pth``) to ``(pth_path, gate_cfg)``.

    ``gate_cfg`` is the flat reconstruction config written by training
    (:data:`GATE_SIDECAR_NAME`): ``camera_names``, ``history_len``, ``history_skip_frame``, and a
    ``model`` sub-dict of gate-arch flags.
    Resolution order, all backward compatible:

      - folder  → pick ``completion_gate_best.pth`` (else newest ``*.pth``) + the sidecar YAML.
      - ``.pth`` → sibling sidecar YAML if present; else the config embedded in the checkpoint
        (``ck["config"]`` → flattened); else ``{}`` (old checkpoint → all defaults).
    """
    if os.path.isdir(gate_ckpt):
        pth = os.path.join(gate_ckpt, "completion_gate_best.pth")
        if not os.path.isfile(pth):
            cands = sorted(glob.glob(os.path.join(gate_ckpt, "*.pth")), key=os.path.getmtime)
            if not cands:
                raise FileNotFoundError(f"No .pth checkpoint found in folder {gate_ckpt!r}")
            pth = cands[-1]
        sidecar = os.path.join(gate_ckpt, GATE_SIDECAR_NAME)
    else:
        pth = gate_ckpt
        sidecar = os.path.join(os.path.dirname(pth), GATE_SIDECAR_NAME)

    if os.path.isfile(sidecar):
        with open(sidecar) as f:
            return pth, (yaml.safe_load(f) or {})

    # Fall back to the config embedded in the checkpoint (train_competion_gate saves asdict(cfg)).
    try:
        ck = torch.load(pth, map_location="cpu", weights_only=False)
        cfg = ck.get("config") if isinstance(ck, dict) else None
    except Exception:
        cfg = None
    if isinstance(cfg, dict):
        ds, mc = cfg.get("dataset", {}), cfg.get("model", {})
        return pth, {
            "camera_names": ds.get("camera_names"),
            "history_len": ds.get("history_len"),
            "history_skip_frame": ds.get("history_skip_frame"),
            "model": {
                "temporal_layers": mc.get("temporal_layers", 2),
                "temporal_heads": mc.get("temporal_heads", 8),
                "use_cross_attention": mc.get("use_cross_attention", False),
                "cross_attn_heads": mc.get("cross_attn_heads", 8),
            },
        }
    return pth, {}


# --------------------------------------------------------------------- System 2 adapter
class System2Controller:
    """Robo-MLT System 2: planner + completion gate + pipeline controller.

    Wraps ``high_level_model``'s :class:`System2Pipeline` behind a small,
    robot-loop-friendly interface.  Camera frames are accumulated into a short
    history; once ``history_len`` frames are available a background inference
    runs the completion gate and advances/recovers the plan pointer.  The
    current atomic skill instruction is exposed via :meth:`current_task` for
    System 1 to execute.

    Args:
        gate_ckpt: Path to the trained completion-gate checkpoint.
        skill_library: Path to the skill-library YAML.
        task_instruction: The high-level task description for planning.
        camera_names: Observation image keys fed to the gate, e.g.
            ``["observation.images.cam_top"]``.
        device: Torch device for System-2 inference.
        history_len / history_skip_frame: Temporal history for the gate.
        sampling_interval: Run System-2 inference every N control frames.
        cooldown_s: Minimum seconds between skill switches.
        llm_fn: Optional LLM callable for planning/replanning (offline falls back
            to declaration-order plans).
        pipeline_kwargs: Extra hyperparameters forwarded to ``System2Pipeline``.
    """

    def __init__(
        self,
        gate_ckpt: str,
        skill_library: str,
        task_instruction: str,
        camera_names: list[str],
        device: str = "cuda",
        history_len: int = 5,
        history_skip_frame: int = 1,
        sampling_interval: int = 15,
        cooldown_s: float = 1.0,
        llm_fn=None,
        prompt_path: str = "",
        **pipeline_kwargs,
    ):
        from high_level_model.models.completion_gate import CompletionGate
        from high_level_model.models.siglip_encoder import build_siglip_encoder
        from high_level_model.planning.planner import Planner
        from high_level_model.planning.prompt_templates import load_prompt_set
        from high_level_model.planning.skill_library import SkillLibrary
        from high_level_model.planning.system2_pipeline import System2Pipeline

        # Resolve gate_ckpt (folder or .pth) → the .pth + the gate-reconstruction sidecar. The sidecar
        # is the source of truth for the gate architecture (it must match how the gate was trained);
        # the matching constructor args become optional overrides. Mismatches are warned.
        pth_path, gate_cfg = _resolve_gate_ckpt(gate_ckpt)

        def _pick(name, passed):
            val = gate_cfg.get(name)
            if val is None:
                return passed
            if passed is not None and val != passed:
                logger.warning("System-2: %s from sidecar (%r) overrides inference config (%r) — "
                               "using the trained value.", name, val, passed)
            return val

        history_len = _pick("history_len", history_len)
        history_skip_frame = _pick("history_skip_frame", history_skip_frame)
        camera_names = _pick("camera_names", camera_names)
        gate_model_cfg = gate_cfg.get("model", {})

        self.device = device
        self.camera_names = camera_names
        self.history_len = int(history_len or 1)
        self.history_skip_frame = int(history_skip_frame or 1)
        self.sampling_interval = sampling_interval
        self.cooldown_s = cooldown_s

        library = SkillLibrary.from_file(skill_library)
        prompt_set = load_prompt_set(prompt_path)
        planner = Planner(library, llm_fn=llm_fn, prompt_set=prompt_set)

        backbone = build_siglip_encoder(device, self.history_len, freeze_siglip=True).to(device)
        gate = self._load_gate(CompletionGate, pth_path, backbone, device, gate_model_cfg)

        # The gate conditions on a frozen SigLIP embedding of the skill string, so a reworded
        # canonical_instruction lands elsewhere in text space and degrades the heads silently.
        # Warn, don't fail: running an unseen skill is a legitimate open-set test.
        trained_texts = set(gate_cfg.get("skill_texts") or [])
        if trained_texts:
            unseen = sorted(set(library.instructions()) - trained_texts)
            if unseen:
                logger.warning("Skill instructions not seen during gate training: %s — the gate is "
                               "extrapolating on these (check for typos/rewording in %s).",
                               unseen, skill_library)
        self.pipeline = System2Pipeline(library, planner, completion_gate=gate, **pipeline_kwargs)
        self.pipeline.reset(task_instruction)

        from torchvision.transforms import v2 as T

        self._norm = T.Compose([
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Raw rolling buffer at the control cadence (one frame per observe() call). It spans exactly
        # the trained window, so slicing every ``history_skip_frame``-th frame reproduces the training
        # window-frame spacing regardless of how often we actually launch inference.
        self._window_span = (self.history_len - 1) * self.history_skip_frame + 1
        self._raw_buffer: deque = deque(maxlen=self._window_span)
        self._frame_counter = 0
        self._last_switch_time: float | None = None
        self._last_info: dict | None = None   # most recent pipeline decision (for the status view)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None

    @staticmethod
    def _load_gate(CompletionGate, ckpt_path, backbone, device, gate_model_cfg=None):
        mc = gate_model_cfg or {}
        gate = CompletionGate(
            backbone, freeze_siglip=True,
            temporal_layers=mc.get("temporal_layers", 2),
            temporal_heads=mc.get("temporal_heads", 8),
            use_cross_attention=mc.get("use_cross_attention", False),
            cross_attn_heads=mc.get("cross_attn_heads", 8),
        ).to(device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd_ck = ck["state_dict"] if "state_dict" in ck else ck
        sd_model = gate.state_dict()
        filtered = {k: v for k, v in sd_ck.items() if k in sd_model and sd_model[k].shape == v.shape}
        gate.load_state_dict(filtered, strict=False)
        gate.eval()
        return gate

    # --- public API --------------------------------------------------------
    def current_task(self) -> str:
        with self._lock:
            return self.pipeline.current_skill_text()

    def is_done(self) -> bool:
        with self._lock:
            return self.pipeline.is_done()

    def status(self) -> dict:
        """Snapshot of the current System-2 execution stage (thread-safe), for the status view.

        Returns the high-level task, the full ordered plan with the current pointer, the current
        skill id/text, the latest (filtered) gate signals + action, and the replanning budget.
        """
        with self._lock:
            p = self.pipeline
            info = self._last_info or {}
            return {
                "task": p.task_instruction,
                "plan": list(p.plan),
                "pointer": p.pointer,
                "skill_id": p.current_skill_id() if p.plan else None,
                "skill_text": p.current_skill_text() if p.plan else None,
                "back_prob": info.get("back_prob"),
                "completion_prob": info.get("completion_prob"),
                "action": info.get("action"),
                "replan_budget": p.replan_budget,
                "is_done": p.is_done(),
            }

    def reset(self, task_instruction: str) -> None:
        with self._lock:
            self.pipeline.reset(task_instruction)
        self._raw_buffer.clear()
        self._frame_counter = 0
        self._last_switch_time = None

    def observe(self, raw_observation: dict) -> None:
        """Sample a frame and, when enough history is gathered, run System 2 in
        the background.  Non-blocking; safe to call every control step."""
        # Apply the result of any finished background inference first.
        self._collect_result()

        # 1) Advance the raw history buffer at the control cadence — ALWAYS, even while an inference
        #    is in flight or during cooldown — so the frames stay uniformly spaced in time and the
        #    window-frame stride below reflects true time.
        frame = self._extract_cameras(raw_observation)
        if frame is not None:
            self._raw_buffer.append(frame)

        # 2) Launch inference on its own cadence (don't stack jobs; respect the switch cooldown).
        if self._future is not None:
            return
        if self._last_switch_time is not None and (time.perf_counter() - self._last_switch_time) < self.cooldown_s:
            return
        self._frame_counter += 1
        if self._frame_counter % self.sampling_interval != 0:
            return
        if len(self._raw_buffer) < self._window_span:
            return

        # 3) Downsample the raw buffer by history_skip_frame → history_len frames ending at the current
        #    frame, spaced exactly as in training. buf[::stride] yields indices 0, S, …, (H-1)·S where
        #    (H-1)·S == span-1 == the latest frame.
        buf = list(self._raw_buffer)                       # len == self._window_span
        window = buf[::self.history_skip_frame]            # H frames, oldest→current, stride = skip
        image_sequence = torch.stack(window, dim=0).unsqueeze(0).to(self.device)
        self._future = self._executor.submit(self._infer, image_sequence)

    # --- internals ---------------------------------------------------------
    def _extract_cameras(self, raw_observation: dict) -> Optional[torch.Tensor]:
        cam_tensors = []
        for cam in self.camera_names:
            key = cam if cam in raw_observation else cam.removeprefix("observation.images.")
            if key not in raw_observation:
                logger.warning("Camera '%s' missing from observation; skipping System-2 sample", cam)
                return None
            img = raw_observation[key]
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(np.asarray(img))
            if img.dim() == 3 and img.shape[2] == 3:  # HWC -> CHW
                img = img.permute(2, 0, 1)
            cam_tensors.append(self._norm(img))
        return torch.stack(cam_tensors, dim=0)  # [Cams, C, H, W]

    def _infer(self, image_sequence: torch.Tensor) -> dict:
        with self._lock:
            skill_text = self.pipeline.current_skill_text()
        with torch.inference_mode():
            comp_t, back_t = self.pipeline.gate.predict(image_sequence, [skill_text])
        completion_prob = float(comp_t.reshape(-1)[0].item())
        back_prob = float(back_t.reshape(-1)[0].item())
        logger.debug("System-2 gate: back=%.3f completion=%.3f skill=%s",
                     back_prob, completion_prob, skill_text)
        with self._lock:
            return self.pipeline.step(back_prob=back_prob, completion_prob=completion_prob)

    def manual_override(self, digit: int, manager: SkillTaskManager) -> None:
        """Keyboard override: jump the System-2 pointer to a declaration-order skill.

        Digit 1-9 jumps to the corresponding skill (``pipeline.jump_to_skill``);
        digit 0 just logs the current plan/pointer (non-destructive status).
        The frame buffer is cleared and the cooldown restarted so the gate
        re-judges the new skill from fresh frames.
        """
        if digit == 0:
            with self._lock:
                logger.info("System 2 status: pointer=%d/%d plan=%s",
                            self.pipeline.pointer, len(self.pipeline.plan), self.pipeline.plan)
            return
        skill_id = manager.skill_for_digit(digit)
        if skill_id is None:
            logger.warning("Keyboard override: digit %d has no skill; ignored.", digit)
            return
        with self._lock:
            self.pipeline.jump_to_skill(skill_id, source="keyboard")
        self._last_switch_time = time.perf_counter()
        self._raw_buffer.clear()
        self._frame_counter = 0
        logger.info("Keyboard override -> skill '%s' (%s)", skill_id, self.current_task())

    def _collect_result(self) -> None:
        if self._future is None or not self._future.done():
            return
        try:
            info = self._future.result()
            with self._lock:
                self._last_info = info
            if info.get("action") in ("advance", "recover", "done"):
                self._last_switch_time = time.perf_counter()
                logger.info("System 2: %s -> skill '%s'", info["action"], self.current_task())
        except Exception as exc:  # noqa: BLE001
            logger.error("System-2 inference failed: %s", exc, exc_info=True)
        finally:
            self._future = None
            self._raw_buffer.clear()
            self._frame_counter = 0

    def shutdown(self) -> None:
        # Non-blocking: a wedged gate inference must not stall the cleanup path.
        self._executor.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------------- control loop
@torch.inference_mode()
def run_loop(robot, executor: RobotAsyncExecutor, controller: Optional[System2Controller],
             dataset_features: dict, fps: int, control_time_s: float,
             event_queue: Optional["queue.Queue"] = None,
             manager: Optional[SkillTaskManager] = None,
             view: Optional["StatusView"] = None) -> None:
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.datasets.utils import build_dataset_frame

    start = time.perf_counter()
    step = 0
    stop = False
    meas_fps = float(fps)  # EMA of the true (full-period) loop rate, shown in the status view
    prev_loop_start: Optional[float] = None
    while not stop and time.perf_counter() - start < control_time_s:
        loop_start = time.perf_counter()
        if prev_loop_start is not None:
            period = loop_start - prev_loop_start
            if period > 0:
                meas_fps = 0.9 * meas_fps + 0.1 * (1.0 / period)
        prev_loop_start = loop_start
        if event_queue is not None:
            for key in drain_keys(event_queue):
                digit = digit_of(key)
                if digit is not None and manager is not None:
                    if controller is not None:
                        controller.manual_override(digit, manager)
                    else:
                        manager.switch_to_digit(digit)
                    current = controller.current_task() if controller is not None else manager.current_text()
                    print(f"[subtask] -> {current}")
                elif key in ("q", "esc"):
                    logger.info("Keyboard: stop requested.")
                    stop = True
            if stop:
                break

        try:
            raw_observation = robot.get_observation()
        except TimeoutError as exc:
            logger.warning("[cam] frame read timed out at step %d (%s); skipping control step.", step, exc)
            precise_sleep(max(0.0, 1.0 / fps - (time.perf_counter() - loop_start)))
            step += 1
            continue

        if controller is not None:
            controller.observe(raw_observation)
            if controller.is_done():
                logger.info("System 2 reports task complete; stopping.")
                break

        observation_frame = build_dataset_frame(dataset_features, raw_observation, prefix="observation")
        try:
            action = executor.get_action(observation_frame)
        except Exception as exc:  # noqa: BLE001
            logger.error("System-1 inference failed at step %d: %s", step, exc)
            action = {key: 0.0 for key in robot.action_features}
        robot.send_action(action)

        if step % 100 == 0:
            subtask = (controller.current_task() if controller
                       else (executor.subtask_provider() if executor.subtask_provider else executor.task_provider()))
            logger.info("step %d, elapsed %.1fs, subtask: %s", step, time.perf_counter() - start, subtask)

        # Push the latest frame + execution stage to the live status window (non-blocking).
        if view is not None:
            elapsed = time.perf_counter() - start
            if controller is not None:
                status = controller.status()
            else:
                subtask = (executor.subtask_provider() if executor.subtask_provider
                           else executor.task_provider())
                status = {"task": executor.task_provider(), "skill_text": subtask}
            status.update(step=step, elapsed=elapsed, fps=meas_fps)
            view.update(raw_observation, status)

        loop_dt = time.perf_counter() - loop_start
        precise_sleep(max(0.0, 1.0 / fps - loop_dt))
        step += 1


def main():
    # If anything wedges, `kill -ABRT <pid>` (or SIGUSR1 where available) dumps every
    # thread's Python stack so the hang location is visible — invaluable for diagnosing
    # a frozen control loop / stuck background inference.
    import faulthandler
    faulthandler.enable()
    try:
        import signal
        faulthandler.register(signal.SIGUSR1)  # `kill -USR1 <pid>` -> dump all stacks
    except (AttributeError, ValueError, OSError):
        pass

    # YAML-first (train_system1 style: `<config.yaml> [key=value ...]`), with the
    # draccus dotlist CLI (`--robot.type=...`) kept as a fallback.
    cfg = parse_with_yaml(InferenceConfig)
    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(cfg.robot)
    runtime = load_robot_policy_runtime(
        cfg.policy_path,
        device=cfg.device,
        robot=robot,
        compile_model=cfg.compile_model,
        compile_mode=cfg.compile_mode,
        compile_cache_dir=cfg.compile_cache_dir,
    )
    policy_config = runtime.config
    policy = runtime.policy
    preprocessor = runtime.preprocessor
    postprocessor = runtime.postprocessor

    # Optional inference-time override of the denoising steps (latency knob, esp. for
    # RTC). Set on both the policy and its inner model config, whichever carries it.
    if cfg.num_inference_steps is not None:
        for c in (getattr(policy, "config", None), getattr(getattr(policy, "model", None), "config", None)):
            if c is not None and hasattr(c, "num_inference_steps"):
                c.num_inference_steps = cfg.num_inference_steps
        logger.info("System-1 num_inference_steps overridden to %d.", cfg.num_inference_steps)

    # Optional LLM backend for planning/replanning (falls back to declaration order).
    llm_fn = None
    if cfg.use_llm:
        from high_level_model.planning.llm_backends import QwenError, qwen_llm_fn
        try:
            llm_fn = qwen_llm_fn(model=cfg.llm_model)
            logger.info("LLM replanning enabled (model=%s).", cfg.llm_model)
        except QwenError as exc:
            logger.warning("LLM unavailable (%s); falling back to declaration-order planning.", exc)

    # Keyboard skill menu (digits 1-9 -> declaration-order skills).
    manager = None
    if cfg.skill_library:
        from high_level_model.planning.skill_library import SkillLibrary
        manager = SkillTaskManager(SkillLibrary.from_file(cfg.skill_library), cfg.task)
        manager.print_menu()

    controller = None
    if cfg.gate_ckpt and cfg.skill_library:
        controller = System2Controller(
            gate_ckpt=cfg.gate_ckpt,
            skill_library=cfg.skill_library,
            task_instruction=cfg.task,
            camera_names=cfg.camera_names,
            device=cfg.device,
            history_len=cfg.history_len,
            history_skip_frame=cfg.history_skip_frame,
            sampling_interval=cfg.sampling_interval,
            cooldown_s=cfg.cooldown_s,
            llm_fn=llm_fn,
            prompt_path=cfg.planner_prompt,
            **asdict(cfg.pipeline),
        )
    task_provider = lambda: cfg.task  # noqa: E731
    if controller is not None:
        subtask_provider = controller.current_task
    elif manager is not None:
        manager.switch_to_digit(1)
        logger.info("No System-2 checkpoint given; keyboard digits switch the System-1 subtask.")
        subtask_provider = manager.current_text
    else:
        logger.info("No skill library given; running System 1 with a fixed task (no subtask).")
        subtask_provider = None

    # Optional output-side anti-jitter: cross-fade each chunk into the previous one.
    blender = build_action_blender(cfg.smoothing, cfg.overlap_steps)
    # Optional publish-time per-joint rate limiter (anti-jerk governor).
    rate_limiter = build_rate_limiter(cfg.rate_limit)


    model_is_ttrtc = bool(getattr(policy_config, "ttrtc", False))
    use_prev_chunk = cfg.rtc.enabled or model_is_ttrtc

    rtc_enabled = False
    latency_tracker = None
    if use_prev_chunk and cfg.smoothing.enabled:
        raise ValueError(
            "The prev-chunk async path (rtc.enabled or a TTRTC policy) and smoothing.enabled "
            "are mutually exclusive; enable only one."
        )

    if model_is_ttrtc:
        from lerobot.policies.rtc.latency_tracker import LatencyTracker

        if cfg.rtc.enabled:
            logger.warning(
                "Policy was trained with TTRTC; ignoring rtc.enabled — TTRTC hard-clamps the "
                "action prefix itself and needs no VJP guidance processor."
            )
        latency_tracker = LatencyTracker(maxlen=cfg.rtc.delay_window)
        rtc_enabled = True  # executor prev-chunk async path
        logger.info(
            "System-1 TTRTC enabled (hard action-prefix clamp; s=%d, delay=auto[min=%d,max=%s]).",
            cfg.rtc.execution_horizon, cfg.rtc.min_inference_delay, cfg.rtc.max_inference_delay,
        )
    elif cfg.rtc.enabled:
        from lerobot.configs.types import RTCAttentionSchedule
        from lerobot.policies.rtc.configuration_rtc import RTCConfig as LeRobotRTCConfig
        from lerobot.policies.rtc.latency_tracker import LatencyTracker
        from lerobot.policies.rtc.modeling_rtc import RTCProcessor

        try:
            schedule = RTCAttentionSchedule[cfg.rtc.prefix_attention_schedule.upper()]
        except KeyError as exc:
            valid = [m.name.lower() for m in RTCAttentionSchedule]
            raise ValueError(
                f"Unknown rtc.prefix_attention_schedule {cfg.rtc.prefix_attention_schedule!r}; "
                f"expected one of {valid}."
            ) from exc

        rtc_cfg = LeRobotRTCConfig(
            enabled=True,
            prefix_attention_schedule=schedule,
            max_guidance_weight=cfg.rtc.max_guidance_weight,
            execution_horizon=cfg.rtc.execution_horizon,
            debug=cfg.rtc.debug,
        )

        policy_type = getattr(policy_config, "type", None)
        compiled = getattr(policy_config, "compile_model", False)
        if policy_type in ("pi0", "pi0_system1"):
            policy.model.rtc_processor = RTCProcessor(rtc_cfg)
            # RTC injects autograd into every denoising step, which graph-breaks a
            # fully-compiled sampler. Re-target compilation onto the per-step
            # denoise_step (RTC-compatible) now that the processor is attached, so the
            # heavy transformer forward stays fused while the guided loop runs in eager.
            if compiled:
                policy.model._apply_compile()
                logger.info("RTC + torch.compile: switched PI0 compile target to per-step denoise_step.")
        elif policy_type == "smolvla":
            policy.config.rtc_config = rtc_cfg
            policy.init_rtc_processor()
            if compiled:
                logger.warning(
                    "RTC needs autograd through each denoising step; SmolVLA's compile "
                    "path may graph-break or slow down. Consider compile_model=false."
                )
        else:
            raise ValueError(
                f"RTC is only supported for flow-matching policies (pi0, smolvla); "
                f"got policy.type={policy_type!r}."
            )

        latency_tracker = LatencyTracker(maxlen=cfg.rtc.delay_window)
        rtc_enabled = True
        logger.info(
            "System-1 RTC enabled (policy=%s, schedule=%s, beta=%.1f, s=%d, delay=auto[min=%d,max=%s]).",
            policy_type, cfg.rtc.prefix_attention_schedule, cfg.rtc.max_guidance_weight,
            cfg.rtc.execution_horizon, cfg.rtc.min_inference_delay, cfg.rtc.max_inference_delay,
        )

    executor = RobotAsyncExecutor(
        policy, robot, task_provider,
        overlap_steps=cfg.overlap_steps,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        subtask_provider=subtask_provider,
        blender=blender,
        rate_limiter=rate_limiter,
        rtc_enabled=rtc_enabled,
        latency_tracker=latency_tracker,
        fps=cfg.fps,
        rtc_execution_horizon=cfg.rtc.execution_horizon,
        rtc_min_delay=cfg.rtc.min_inference_delay,
        rtc_max_delay=cfg.rtc.max_inference_delay,
    )
    dataset_features = build_dataset_features(robot)

    if getattr(policy_config, "compile_model", False):
        executor.warmup(task_provider(), preprocessor=preprocessor)

    prompt_to_start("Press ENTER to connect the robot and start control (%s)"
                    % ("System-1 + System-2" if controller is not None else "System-1"))

    event_queue: "queue.Queue" = queue.Queue()
    listener = start_keyboard_listener(event_queue)

    view = None
    if cfg.enable_view:
        view = StatusView(camera_names=cfg.camera_names, bgr=cfg.view_bgr,
                          max_width=cfg.view_max_width).start()

    robot.connect()
    try:
        run_loop(robot, executor, controller, dataset_features, cfg.fps, cfg.control_time_s,
                 event_queue=event_queue, manager=manager, view=view)
    finally:
        for name, cleanup in (
            ("robot", robot.disconnect),
            ("executor", executor.shutdown),
            ("controller", controller.shutdown if controller is not None else None),
            ("view", view.stop if view is not None else None),
            ("listener", listener.stop if listener is not None else None),
        ):
            if cleanup is None:
                continue
            try:
                cleanup()
            except Exception as exc:  # noqa: BLE001
                logger.error("Cleanup step %r failed: %s", name, exc, exc_info=True)


if __name__ == "__main__":
    main()
