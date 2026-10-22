"""System 1 Policy Factory.

Provides factory functions for creating and loading System-1 policy instances
(System 1 — Generative Executor) in the Robo-MLT framework.  Four policy types
are supported, selected by the ``type`` string (from YAML ``policy.type`` or a
checkpoint's ``config.json``):

- ``pi0_system1`` (alias ``pi0``) — native PI0 flow-matching policy.
- ``act``                        — lerobot ACT (re-exported wrapper).
- ``smolvla``                    — lerobot SmolVLA (re-exported wrapper).
- ``qwen3vl_vla``                — Qwen3-VL/Qwen2.5-VL layer-wise flow-matching VLA.
- ``pi05_full`` (alias ``pi05``) — π0.5 full port (subtask decode + FAST CE + knowledge insulation).

Usage::

    policy_cls = get_policy_class("pi0")
    policy = make_policy(cfg, ds_meta)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.pretrained import PreTrainedPolicy
logger = logging.getLogger(__name__)

__all__ = [
    "get_policy_class",
    "get_config_class",
    "load_config_local",
    "make_pre_post_processors",
    "make_policy",
    "load_policy",
]


@dataclass(frozen=True)
class _PolicySpec:
    canonical_name: str
    aliases: tuple[str, ...]
    config_target: str
    policy_target: str
    processor_target: str


_POLICY_SPECS = (
    _PolicySpec(
        "pi0_system1",
        ("pi0",),
        "low_level_model.models.pi0.configuration_pi0:PI0Config",
        "low_level_model.models.pi0.modeling_pi0:PI0Policy",
        "low_level_model.models.pi0.processor_pi0:make_pi0_pre_post_processors",
    ),
    _PolicySpec(
        "act",
        (),
        "low_level_model.models.act:ACTConfig",
        "low_level_model.models.act:ACTPolicy",
        "low_level_model.models.act:make_act_pre_post_processors",
    ),
    _PolicySpec(
        "smolvla",
        (),
        "low_level_model.models.smolvla:SmolVLAConfig",
        "low_level_model.models.smolvla:SmolVLAPolicy",
        "low_level_model.models.smolvla:make_smolvla_pre_post_processors",
    ),
    _PolicySpec(
        "qwen3vl_vla",
        (),
        "low_level_model.models.qwen3vl_vla:Qwen3VLVLAConfig",
        "low_level_model.models.qwen3vl_vla:Qwen3VLVLAPolicy",
        "low_level_model.models.qwen3vl_vla:make_qwen3vl_vla_pre_post_processors",
    ),
    _PolicySpec(
        "pi05_full",
        ("pi05",),
        "low_level_model.models.pi05:PI05FullConfig",
        "low_level_model.models.pi05:PI05FullPolicy",
        "low_level_model.models.pi05:make_pi05_full_pre_post_processors",
    ),
)
_POLICY_BY_NAME = {
    name: spec
    for spec in _POLICY_SPECS
    for name in (spec.canonical_name, *spec.aliases)
}


def _resolve_spec(name: str) -> _PolicySpec:
    try:
        return _POLICY_BY_NAME[name]
    except KeyError as exc:
        supported = ", ".join(spec.canonical_name for spec in _POLICY_SPECS)
        raise NotImplementedError(
            f"Unknown policy {name!r}. Supported policies: [{supported}]"
        ) from exc


def _load_target(target: str):
    module_name, attribute = target.split(":", 1)
    return getattr(import_module(module_name), attribute)


def make_pre_post_processors(policy_cfg: PreTrainedConfig, dataset_stats: Any = None):
    """Build the (pre, post) processor pipelines matching ``policy_cfg.type``.

    Single source of truth for processor construction, shared by the trainer and
    the inference runtime.  Each policy ships its own builder (PI0 has a
    PaliGemma tokenizer step, SmolVLA a SmolVLM tokenizer step, ACT none, and
    Qwen3VL-VLA tokenises inside the model so its pipeline has no tokenizer).
    """
    builder: Callable = _load_target(_resolve_spec(policy_cfg.type).processor_target)
    return builder(policy_cfg, dataset_stats=dataset_stats)


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """Return the policy class for the given policy type name.

    Args:
        name: Policy type name (``pi0_system1``/``pi0``, ``act``, ``smolvla``,
            ``qwen3vl_vla``).
    Returns:
        Policy class (not instance).
    Raises:
        NotImplementedError: If the policy name is unrecognised.
    """
    return _load_target(_resolve_spec(name).policy_target)


def get_config_class(name: str) -> type[PreTrainedConfig]:
    """Return the config class for the given policy type name.

    Importing the class also registers it with lerobot's ``PreTrainedConfig``
    choice registry (``@PreTrainedConfig.register_subclass(name)``).
    """
    return _load_target(_resolve_spec(name).config_target)


def load_config_local(pretrained_path: str | os.PathLike) -> PreTrainedConfig:
    """Build a policy config from a checkpoint's ``config.json``, dispatched by type.

    Reads the ``type`` field to pick the config class.  PI0 and Qwen3VL-VLA use
    their draccus-free ``from_pretrained_local`` (PI0 nests HF sub-configs that
    choke draccus); ACT/SmolVLA are plain dataclasses loaded via lerobot's
    standard ``PreTrainedConfig.from_pretrained``.
    """
    config_file = os.path.join(str(pretrained_path), "config.json")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"No 'config.json' found in: {pretrained_path}")
    with open(config_file) as f:
        policy_type = json.load(f).get("type", "pi0_system1")

    # Import (and thereby register with draccus' ChoiceRegistry) the matching subclass.
    config_cls = get_config_class(policy_type)
    if hasattr(config_cls, "from_pretrained_local"):
        return config_cls.from_pretrained_local(pretrained_path)
    # ACT/SmolVLA go through lerobot's draccus-based loader, which must be called on the
    # BASE PreTrainedConfig so draccus uses the `type` discriminator to pick the subclass.
    # Calling it on the concrete subclass (e.g. SmolVLAConfig.from_pretrained) makes
    # draccus treat `type` as an unknown dataclass field and raise a DecodingError.
    return PreTrainedConfig.from_pretrained(pretrained_path)


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata,
) -> PreTrainedPolicy:
    """Create a System-1 policy instance from config and dataset metadata.

    Args:
        cfg: Policy configuration with type, device, pretrained_path, etc.
        ds_meta: Dataset metadata with feature definitions and normalisation stats.

    Returns:
        Initialised policy ready for training or inference.
    """
    policy_cls = get_policy_class(cfg.type)
    features = dataset_to_policy_features(ds_meta.features)
    if not cfg.output_features:
        cfg.output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    if not cfg.input_features:
        cfg.input_features = {k: ft for k, ft in features.items() if k not in cfg.output_features}
    kwargs: dict[str, Any] = {
        "config": cfg,
        "dataset_stats": ds_meta.stats,
    }
    if getattr(cfg, "pretrained_path", None):
        policy = policy_cls.from_pretrained(pretrained_name_or_path=cfg.pretrained_path, **kwargs)
    else:
        policy = policy_cls(**kwargs)
    policy.to(cfg.device)
    policy.eval()
    assert isinstance(policy, nn.Module)
    logger.info("Created %s policy on device=%s", cfg.type, cfg.device)
    return policy


def load_policy(pretrained_path: str, device: str = "cuda") -> PreTrainedPolicy:
    """Load a trained System-1 policy from a checkpoint directory for inference.

    Unlike :func:`make_policy` (which needs dataset metadata and is used for
    training), this builds the policy purely from the saved checkpoint: the
    config is read locally (:func:`load_config_local`, dispatched by ``type``)
    and the matching policy class via :func:`get_policy_class`.  This is the
    robot-side entry point — no lerobot policy factory involved.

    Args:
        pretrained_path: Directory containing ``model.safetensors`` and ``config.json``.
        device: Target device string (e.g. ``"cuda"``, ``"cpu"``).

    Returns:
        An initialised, eval-mode policy on ``device``.
    """
    config = load_config_local(pretrained_path)
    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=pretrained_path, config=config)
    policy.config.device = device
    policy.to(device)
    policy.eval()
    assert isinstance(policy, nn.Module)
    logger.info("Loaded %s policy from %s on device=%s", config.type, pretrained_path, device)
    return policy
