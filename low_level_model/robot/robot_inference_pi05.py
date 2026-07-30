"""Real-time robot control for the standalone π0.5 baseline (System 1 only).

π0.5 is a self-contained dual-system VLA: it autoregressively generates its own subtask
(high-level reasoning) *inside* ``predict_action_chunk``, so at deployment it needs only
the high-level task instruction — there is no external System-2 gate, planner, or skill
library. This is the point of the baseline: contrast it against Robo-MLT's explicit
System-2 (``robot_inference.py``), which drives the subtask from a completion gate.

This entry point is a slim wrapper over the shared deployment machinery
(``RobotAsyncExecutor``, ``run_loop``, processor pipelines) with System-2 removed:

    robot ──obs(fps)──> RobotAsyncExecutor ──action chunk──> robot
                             task = cfg.task (fixed)
                             subtask = π0.5 self-generated (logged, not injected as control)

Usage (YAML-first, same style as train_system1 / robot_inference):

    python -m low_level_model.robot.robot_inference_pi05 \
        configs/system1/inference/inference_pi05.yaml

    # override fields
    python -m low_level_model.robot.robot_inference_pi05 \
        configs/system1/inference/inference_pi05.yaml \
        task="Put the blood gas test tube in the blue plate." fps=25 overlap_steps=10
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass, field
from typing import Optional

from lerobot.robots import RobotConfig

# Importing the robot subpackages registers their RobotConfig subclasses with draccus'
# ChoiceRegistry so `--robot.type=...` / `robot.type=...` resolves (lerobot pattern).
from lerobot.robots import (  # noqa: F401
    bi_so100_follower,
    dual_piper,
    koch_follower,
    lekiwi,
    piper,
    so100_follower,
    so101_follower,
)

from low_level_model.robot.keyboard_control import start_keyboard_listener
from low_level_model.robot.robot_common import (
    RobotAsyncExecutor,
    build_dataset_features,
    prompt_to_start,
)
# Reuse the exact control loop from the dual-system entry point with controller=None.
from low_level_model.robot.robot_inference import run_loop
from low_level_model.robot.runtime_builder import (
    RateLimitConfig,
    SmoothingConfig,
    build_action_blender,
    build_rate_limiter,
    load_robot_policy_runtime,
)
from low_level_model.robot.status_view import StatusView
from low_level_model.utils.config_loader import parse_with_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Pi05InferenceConfig:
    """Run config for the standalone π0.5 baseline. Loaded from YAML (+ ``key=value``
    overrides) or a draccus dotlist — see ``configs/system1/inference/inference_pi05.yaml``."""

    robot: RobotConfig
    policy_path: str          # π0.5 checkpoint directory (config.json type == pi05_full)
    task: str                 # High-level task description (the only control input)
    device: str = "cuda"
    fps: int = 25
    control_time_s: float = 600.0
    # Chunk overlap for async prefetch; 0 disables prefetching (each boundary is a
    # synchronous inference). π0.5 pays an extra subtask AR-decode (up to
    # max_decoding_steps) whenever the subtask cache expires, so keep some runway.
    overlap_steps: int = 10
    # Override the flow-matching denoising steps at inference (None = keep checkpoint's).
    num_inference_steps: Optional[int] = None
    # Override how often π0.5 regenerates its subtask (seconds; None = keep checkpoint's).
    subtask_regeneration_interval: Optional[float] = None
    compile_model: Optional[bool] = None
    compile_mode: Optional[str] = None
    compile_cache_dir: Optional[str] = None
    # System-1 output-side action-chunk smoothing (anti-jitter; opt-in). RTC is intentionally
    # NOT wired here — the RTC branch upstream is gated to pi0/smolvla only.
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    # Publish-time per-joint rate limiter (anti-jerk governor; opt-in). Especially useful
    # for Piper MIT mode, which reproduces any chunk-boundary jump as a physical jerk.
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    # Camera keys for the optional live status view.
    camera_names: list[str] = field(default_factory=lambda: ["observation.images.cam_top"])
    enable_view: bool = False
    view_bgr: bool = True
    view_max_width: int = 1280


def _make_subtask_display(policy) -> "callable":
    """Return a callable that decodes π0.5's most recently self-generated subtask text.

    Used purely for logging / the status view so the operator can see what high-level
    step the model believes it is executing. The returned text is also injected into the
    batch as ``subtask`` by the executor, but π0.5's ``predict_action_chunk`` ignores any
    provided subtask and regenerates its own — so this never steers control.
    """
    def _display() -> str:
        tokens = getattr(policy, "_cached_subtask_tokens", None)
        masks = getattr(policy, "_cached_subtask_masks", None)
        if tokens is None:
            return ""
        # Reuse the model's decoder so the status view and the per-regeneration log
        # (predict_action_chunk) trim the subtask identically.
        return policy.model.decode_subtask_text(tokens, masks)

    return _display


def main():
    import faulthandler
    faulthandler.enable()
    try:
        import signal
        faulthandler.register(signal.SIGUSR1)
    except (AttributeError, ValueError, OSError):
        pass

    cfg = parse_with_yaml(Pi05InferenceConfig)
    from lerobot.robots import make_robot_from_config
    robot = make_robot_from_config(cfg.robot)
    runtime = load_robot_policy_runtime(
        cfg.policy_path,
        device=cfg.device,
        robot=robot,
        compile_model=cfg.compile_model,
        compile_mode=cfg.compile_mode,
        compile_cache_dir=cfg.compile_cache_dir,
        expected_policy_types=("pi05_full", "pi05"),
    )
    policy_config = runtime.config
    policy = runtime.policy
    preprocessor = runtime.preprocessor
    postprocessor = runtime.postprocessor

    # Inference-time overrides on the model config (latency / reasoning-cadence knobs).
    for c in (getattr(policy, "config", None), getattr(getattr(policy, "model", None), "config", None)):
        if c is None:
            continue
        if cfg.num_inference_steps is not None and hasattr(c, "num_inference_steps"):
            c.num_inference_steps = cfg.num_inference_steps
        if cfg.subtask_regeneration_interval is not None and hasattr(c, "subtask_regeneration_interval"):
            c.subtask_regeneration_interval = cfg.subtask_regeneration_interval
    if cfg.num_inference_steps is not None:
        logger.info("π0.5 num_inference_steps overridden to %d.", cfg.num_inference_steps)
    if cfg.subtask_regeneration_interval is not None:
        logger.info("π0.5 subtask_regeneration_interval overridden to %.2fs.",
                    cfg.subtask_regeneration_interval)

    task_provider = lambda: cfg.task  # noqa: E731
    # Display-only: surfaces π0.5's self-generated subtask in logs/status (see docstring).
    subtask_provider = _make_subtask_display(policy)

    blender = build_action_blender(cfg.smoothing, cfg.overlap_steps)
    rate_limiter = build_rate_limiter(cfg.rate_limit)

    executor = RobotAsyncExecutor(
        policy, robot, task_provider,
        overlap_steps=cfg.overlap_steps,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        subtask_provider=subtask_provider,
        blender=blender,
        rate_limiter=rate_limiter,
    )
    dataset_features = build_dataset_features(robot)

    if getattr(policy_config, "compile_model", False):
        executor.warmup(task_provider(), preprocessor=preprocessor)

    prompt_to_start("Press ENTER to connect the robot and start control (π0.5 standalone)")

    event_queue: "queue.Queue" = queue.Queue()
    listener = start_keyboard_listener(event_queue)

    view = None
    if cfg.enable_view:
        view = StatusView(camera_names=cfg.camera_names, bgr=cfg.view_bgr,
                          max_width=cfg.view_max_width).start()

    robot.connect()
    try:
        run_loop(robot, executor, None, dataset_features, cfg.fps, cfg.control_time_s,
                 event_queue=event_queue, manager=None, view=view)
    finally:
        for name, cleanup in (
            ("robot", robot.disconnect),
            ("executor", executor.shutdown),
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
