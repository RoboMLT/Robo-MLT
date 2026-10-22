"""Real-robot deployment of Robo-MLT: System 1 + System 2 under the asynchronous runtime.

Implements the control loop of paper Algorithm 1 at f_ctrl = 20 Hz:

    robot ──obs──> System2Controller.observe   (every step; samples a frame every K steps,
                                                monitor job J2 in the background)
          ──obs──> RobotAsyncExecutor.get_action (action queue A of N-step chunks, prefetch job
                                                J1 once <= r actions remain, w-step blend)
          <─action─

On an advance / recover the loop discards the queued chunk and the pending System-1 job and the
next chunk is conditioned on the new skill; while the replanner runs, the arm holds its last
command. System 1 is the Qwen3-VL flow-matching policy (``configs/system1/train/*/qwen3vl_vla``);
a baseline checkpoint runs through the same loop with System 2 disabled (no ``gate_ckpt``).

Keyboard (requires pynput / a display), operator affordances outside the method:
    1-9     jump the plan pointer to the N-th skill of the library
    0       print the current plan / pointer
    q, ESC  stop the control loop

Configuration: one YAML file plus ``key=value`` overrides (dot-notation for nested keys)::

    python -m low_level_model.robot.robot_inference configs/inference/blood_gas_analysis.yaml

    # ablation variants of Sec. IV-A-4
    python -m low_level_model.robot.robot_inference configs/inference/blood_gas_analysis.yaml \
        variant=timeout_only
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

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
from low_level_model.robot.system2_controller import System2Controller
from low_level_model.robot.task_manager import SkillTaskManager, digit_of
from low_level_model.utils.config_loader import parse_with_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- config
@dataclass
class PipelineConfig:
    """System-2 monitor hyperparameters (paper Table I), forwarded to
    :class:`high_level_model.planning.system2_pipeline.System2Pipeline`.

    Every field must be a valid ``System2Pipeline.__init__`` argument. The variant switches
    (``enable_recovery_head`` / ``enable_stall_timeout``) come from ``InferenceConfig.variant``.
    """

    tau_done: float = 0.6          # advance threshold
    k_a: int = 2                   # advance debounce (consecutive updates)
    tau_rec: float = 0.8           # recover threshold
    k_b: int = 3                   # recover debounce (consecutive updates)
    t_max_s: float = 30.0          # per-skill stall timeout T_max
    replan_budget: int = 3         # B_replan: replanner calls per episode
    max_plan_attempts: int = 3     # planner queries for an admissible initial plan
    # Optional pre-threshold filters; 1.0 / 1 / 1.0 disables them (paper behaviour).
    rec_ema_alpha: float = 1.0
    completion_median_window: int = 1
    completion_ema_alpha: float = 1.0


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
    ``configs/inference/<task>.yaml``."""

    robot: RobotConfig
    policy_path: str          # System-1 checkpoint directory (Qwen3-VL VLA or a baseline)
    task: str                 # High-level task instruction given to the planner
    device: str = "cuda:0"
    fps: int = 20
    control_time_s: float = 600.0
    # Chunk overlap for async prefetch; 0 disables prefetching entirely (every
    # chunk boundary becomes a synchronous inference). In RTC mode this is also the
    # prefetch *runway*: set it >= the RTC inference delay in steps (ceil(latency*fps))
    # to keep the action buffer from underrunning (arm pausing).
    overlap_steps: int = 4
    # Override the flow-matching integration steps at inference (None = keep the checkpoint's
    # value). Applied to whichever of num_inference_steps / num_inference_timesteps the policy has.
    num_inference_steps: Optional[int] = None
    compile_model: Optional[bool] = None
    compile_mode: Optional[str] = None          # e.g. "max-autotune", "reduce-overhead"
    compile_cache_dir: Optional[str] = None
    # System 2 (enabled when both gate_ckpt and skill_library are set). M, the camera order and
    # the trunk architecture are read from the checkpoint's sidecar.
    skill_library: Optional[str] = None   # Skill-library YAML (skills + protocol constraints)
    gate_ckpt: Optional[str] = None       # Monitor checkpoint (run folder or .pth)
    camera_names: list[str] = field(default_factory=lambda: ["observation.images.cam_top"])
    planner_camera: str = "observation.images.cam_top"   # global view sent to the planner
    sampling_interval: int = 15           # K: control steps between System-2 frames
    cooldown_steps: int = 10              # Delta_cd: control steps without sampling after a switch
    # Evaluated variant (paper Sec. IV-A-4): ours | single_frame | no_recovery | timeout_only.
    # single_frame additionally needs a monitor checkpoint trained with M = 1.
    variant: str = "ours"
    allow_inadmissible_plan: bool = False
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    # System-1 output-side action-chunk smoothing (anti-jitter; opt-in).
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    # Publish-time per-joint rate limiter (anti-jerk governor; opt-in). Composes with
    # smoothing/RTC — it is a pure output-side clamp on the commanded joint velocity.
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    # System-1 Real-Time Chunking (inference-time smoothing; opt-in, flow policies
    # only). Mutually exclusive with smoothing.
    rtc: RTCConfig = field(default_factory=RTCConfig)
    # Planner / replanner (requires DASHSCOPE_API_KEY). Without it the planner falls back to the
    # library declaration order, which is only meant for offline testing.
    use_llm: bool = True
    llm_model: str = "qwen3.7-max"
    # Planner prompt template file (configs/skill_library/prompts/*.yaml). Empty = built-in default.
    planner_prompt: str = ""
    # Live status window (OpenCV): shows observation images + System-2 plan/stage overlay.
    enable_view: bool = False
    view_bgr: bool = True               # convert RGB->BGR for cv2 display (RealSense frames are RGB)
    view_max_width: int = 1280          # max width of the tiled camera strip (downscaled if larger)


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

        holding = False
        if controller is not None:
            controller.observe(raw_observation)
            if controller.is_done():
                logger.info("System 2 reports task complete; stopping.")
                break
            if controller.consume_switch():
                # Algorithm 1: a new skill invalidates the queued chunk and the pending job.
                executor.discard_pending()
            holding = controller.is_replanning()

        if holding:
            action = executor.last_action()       # hold position while the replanner runs
        else:
            observation_frame = build_dataset_frame(dataset_features, raw_observation,
                                                    prefix="observation")
            try:
                action = executor.get_action(observation_frame)
            except Exception as exc:  # noqa: BLE001
                logger.error("System-1 inference failed at step %d: %s — holding position.",
                             step, exc)
                action = executor.last_action()
        if action is not None:
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
            for name in ("num_inference_steps", "num_inference_timesteps"):
                if c is not None and hasattr(c, name):
                    setattr(c, name, cfg.num_inference_steps)
        logger.info("System-1 integration steps overridden to %d.", cfg.num_inference_steps)

    # Planner / replanner backend (falls back to declaration order without an API key).
    llm_fn = None
    if cfg.use_llm:
        from high_level_model.planning.llm_backends import QwenError, qwen_llm_fn
        try:
            llm_fn = qwen_llm_fn(model=cfg.llm_model)
            logger.info("Planner / replanner: %s.", cfg.llm_model)
        except QwenError as exc:
            logger.warning("Planner unavailable (%s); falling back to declaration-order planning.", exc)

    # Keyboard skill menu (digits 1-9 -> declaration-order skills).
    manager = None
    if cfg.skill_library:
        from high_level_model.planning.skill_library import SkillLibrary
        manager = SkillTaskManager(SkillLibrary.from_file(cfg.skill_library), cfg.task)
        manager.print_menu()

    controller = None
    if cfg.gate_ckpt and cfg.skill_library:
        from high_level_model.planning.system2_pipeline import variant_flags

        logger.info("Variant: %s.", cfg.variant)
        controller = System2Controller.from_checkpoint(
            gate_ckpt=cfg.gate_ckpt,
            skill_library=cfg.skill_library,
            task_instruction=cfg.task,
            camera_names=cfg.camera_names,
            device=cfg.device,
            llm_fn=llm_fn,
            prompt_path=cfg.planner_prompt,
            sampling_interval=cfg.sampling_interval,
            cooldown_steps=cfg.cooldown_steps,
            fps=cfg.fps,
            planner_camera=cfg.planner_camera,
            allow_inadmissible_plan=cfg.allow_inadmissible_plan,
            **asdict(cfg.pipeline),
            **variant_flags(cfg.variant),
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
