"""Shared assembly helpers for robot-side System-1 policy runtimes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from low_level_model.models.factory import load_config_local
from low_level_model.robot.robot_common import (
    setup_torch_compile_cache,
    validate_robot_cameras,
)
from low_level_model.runtime.inference_system1 import load_system1_policy

logger = logging.getLogger(__name__)

__all__ = [
    "LoadedPolicyRuntime",
    "SmoothingConfig",
    "RateLimitConfig",
    "build_action_blender",
    "build_rate_limiter",
    "load_robot_policy_runtime",
]


@dataclass
class SmoothingConfig:
    """Output-side action-chunk cross-fade configuration."""

    enabled: bool = False
    latency_k: Optional[int] = None
    min_smooth_steps: int = 8
    weight_profile: str = "linear"


@dataclass
class RateLimitConfig:
    """Publish-time per-joint velocity (rate) limiter configuration.

    Caps the per-step change of each commanded joint so a chunk-boundary jump becomes
    a bounded ramp instead of a jerk — the software stand-in for the accel-limited
    trajectory smoothing that Piper MIT mode does not do internally. Composes with the
    blender (they are independent output stages); enable either or both.
    """

    enabled: bool = False
    # Max |Δ| per control step, in action units (radians for the Piper arm joints).
    # Bounds joint velocity to max_step * fps. Broadcast to every joint. Tune: lower
    # until the MIT jerk disappears, raise if normal motion feels laggy/clamped.
    max_step: float = 0.08
    # Optional per-joint override (length == action dim). When set, takes precedence
    # over `max_step`; give the gripper a large value so grasps are not throttled.
    max_step_per_joint: Optional[list[float]] = None


@dataclass(frozen=True)
class LoadedPolicyRuntime:
    """Policy and processors prepared for a robot entry point."""

    config: Any
    policy: Any
    preprocessor: Any
    postprocessor: Any


def _apply_compile_overrides(
    config,
    *,
    compile_model: Optional[bool],
    compile_mode: Optional[str],
    compile_cache_dir: Optional[str],
) -> None:
    for field_name, value in (
        ("compile_model", compile_model),
        ("compile_mode", compile_mode),
        ("compile_cache_dir", compile_cache_dir),
    ):
        if value is not None and hasattr(config, field_name):
            setattr(config, field_name, value)


def load_robot_policy_runtime(
    policy_path: str,
    *,
    device: str,
    robot,
    compile_model: Optional[bool] = None,
    compile_mode: Optional[str] = None,
    compile_cache_dir: Optional[str] = None,
    expected_policy_types: Optional[tuple[str, ...]] = None,
) -> LoadedPolicyRuntime:
    """Load and validate a policy runtime with consistent compile overrides."""
    config = load_config_local(policy_path)
    policy_type = getattr(config, "type", None)
    if expected_policy_types is not None and policy_type not in expected_policy_types:
        expected = ", ".join(expected_policy_types)
        raise ValueError(
            f"Expected checkpoint policy.type in [{expected}], got {policy_type!r}"
        )

    _apply_compile_overrides(
        config,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_cache_dir=compile_cache_dir,
    )
    if getattr(config, "compile_model", False):
        setup_torch_compile_cache(
            getattr(config, "compile_cache_dir", "./torch_compile_cache")
        )

    policy, preprocessor, postprocessor = load_system1_policy(
        policy_path,
        device=device,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_cache_dir=compile_cache_dir,
    )
    validate_robot_cameras(robot, config)
    return LoadedPolicyRuntime(config, policy, preprocessor, postprocessor)


def build_action_blender(config: SmoothingConfig, overlap_steps: int):
    """Build the optional chunk blender shared by deployment and collection."""
    if not config.enabled:
        return None

    from low_level_model.runtime.action_smoothing import TemporalChunkBlender

    latency_k = config.latency_k if config.latency_k is not None else overlap_steps
    blender = TemporalChunkBlender(
        latency_k=latency_k,
        min_smooth_steps=config.min_smooth_steps,
        weight_profile=config.weight_profile,
    )
    logger.info(
        "System-1 chunk smoothing enabled (latency_k=%d, min_smooth_steps=%d, profile=%s).",
        latency_k,
        config.min_smooth_steps,
        config.weight_profile,
    )
    return blender


def build_rate_limiter(config: RateLimitConfig):
    """Build the optional publish-time per-joint rate limiter (anti-jerk governor)."""
    if not config.enabled:
        return None

    from low_level_model.runtime.action_smoothing import JointRateLimiter

    limit = config.max_step_per_joint if config.max_step_per_joint is not None else config.max_step
    limiter = JointRateLimiter(limit)
    logger.info("System-1 publish-time joint rate limiter enabled (max_step=%s).", limit)
    return limiter
