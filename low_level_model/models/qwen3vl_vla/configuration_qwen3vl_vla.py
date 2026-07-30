"""Qwen3VL-VLA Policy Configuration.

A Vision-Language-Action policy whose backbone is a Qwen3-VL (or, where Qwen3-VL
is unavailable in the installed ``transformers``, Qwen2.5-VL) image-text model
and whose action head is a layer-wise cross-attention flow-matching DiT.  This is
a faithful port of starVLA's ``Qwen_PI`` (``framework/QwenPI.py`` +
``modules/action_model/LayerwiseFM_ActionHeader.py``) onto the lerobot
``PreTrainedConfig`` / ``PreTrainedPolicy`` contract used by Robo-MLT System 1.

Unlike :class:`~low_level_model.models.pi0.configuration_pi0.PI0Config`, this
config holds **only plain scalar fields** — the heavy HuggingFace VLM is loaded
inside the model's ``__init__`` from ``vlm_model_id`` rather than being nested as
a sub-config — so the standard lerobot/draccus loader round-trips it cleanly.  A
``from_pretrained_local`` classmethod is provided anyway for parity with the
robot-side loader.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("qwen3vl_vla")
@dataclass
class Qwen3VLVLAConfig(PreTrainedConfig):
    """Configuration for the Qwen3VL-VLA policy (System 1 of Robo-MLT)."""

    # === VLM backbone ===
    # Any Qwen3-VL or Qwen2.5-VL instruct checkpoint id (or local path).  The
    # backbone auto-selects the matching transformers class; Qwen3-VL needs
    # transformers >= 4.57, otherwise pass a Qwen2.5-VL id.
    vlm_model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    attn_implementation: str = "sdpa"  # or "flash_attention_2"
    dtype: str = "bfloat16"            # drives Accelerate mixed-precision + VLM weight dtype
    freeze_vlm: bool = False

    # === Action prediction ===
    n_obs_steps: int = 1
    chunk_size: int = 16        # actions predicted per chunk (= action horizon)
    n_action_steps: int = 16    # actions executed per chunk

    # Padding dimensions (state/action are zero-padded to these, sliced back at output)
    max_state_dim: int = 32
    max_action_dim: int = 32

    # === Flow-matching action head (layer-wise cross-attention DiT) ===
    num_inference_timesteps: int = 4
    num_target_vision_tokens: int = 32   # learned "future" query tokens prepended to the action seq
    attention_head_dim: int = 64
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    add_pos_embed: bool = True
    max_seq_len: int = 1024
    dit_dropout: float = 0.2
    repeated_diffusion_steps: int = 2    # batch-repeat factor inside the FM loss (starVLA forces 2)

    # === Image processing ===
    image_resolution: tuple[int, int] = (224, 224)
    empty_cameras: int = 0

    # === Normalization ===
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # === Training / runtime ===
    gradient_checkpointing: bool = False
    compile_model: bool = False
    device: str | None = None

    # === Optimizer / scheduler ===
    optimizer_lr: float = 1e-4            # action-head learning rate (peak)
    vlm_lr: float = 1e-5                  # VLM backbone learning rate (peak)
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        """Validate and set up input/output features (mirrors PI0Config)."""
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            self.input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
        if "observation.state" not in self.input_features:
            self.input_features["observation.state"] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
        if "action" not in self.output_features:
            self.output_features["action"] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    # === Local checkpoint loading =========================================
    _TUPLE_FIELDS = ("image_resolution", "optimizer_betas")

    @classmethod
    def from_pretrained_local(cls, pretrained_path: str | os.PathLike) -> "Qwen3VLVLAConfig":
        """Build a :class:`Qwen3VLVLAConfig` directly from a checkpoint's ``config.json``.

        Plain-field analogue of :meth:`PI0Config.from_pretrained_local`; used by
        the robot/runtime loaders for parity (this config has no nested HF
        sub-configs, so it would also round-trip through draccus).
        """
        config_file = os.path.join(str(pretrained_path), "config.json")
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"No 'config.json' found in: {pretrained_path}")
        with open(config_file) as f:
            raw = json.load(f)

        raw.pop("type", None)
        kwargs: dict = {}

        def _decode_features(serialized: dict | None) -> dict:
            if not serialized:
                return {}
            return {
                key: PolicyFeature(type=FeatureType(feat["type"]), shape=tuple(feat["shape"]))
                for key, feat in serialized.items()
            }

        if "input_features" in raw:
            kwargs["input_features"] = _decode_features(raw.pop("input_features"))
        if "output_features" in raw:
            kwargs["output_features"] = _decode_features(raw.pop("output_features"))
        if raw.get("normalization_mapping"):
            kwargs["normalization_mapping"] = {
                k: NormalizationMode(v) for k, v in raw.pop("normalization_mapping").items()
            }
        else:
            raw.pop("normalization_mapping", None)

        field_names = {f.name for f in fields(cls)}
        for key, value in raw.items():
            if key not in field_names:
                continue
            if key in cls._TUPLE_FIELDS and isinstance(value, list):
                value = tuple(value)
            kwargs[key] = value

        return cls(**kwargs)


__all__ = ["Qwen3VLVLAConfig"]
