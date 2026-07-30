"""PI0 Policy Configuration.

This module defines the configuration for the PI0 (π0) Vision-Language-Action
model. PI0 is the base VLA model: PaliGemma (SigLIP + Gemma-2B) backbone
coupled with a Gemma-300M action expert and a flow-matching action head.

This is the System 1 (Generative Executor) policy used in Robo-MLT.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.paligemma.configuration_paligemma import PaliGemmaConfig


@dataclass
class PI0VLMConfig(PaliGemmaConfig):
    """Configuration for the PaliGemma vision-language backbone."""

    def __init__(self):
        super().__init__()
        self._vocab_size = 257152
        self.image_token_index = 257152

        # Text encoder (Gemma) configuration
        self.text_config.hidden_size = 2048
        self.text_config.intermediate_size = 16_384
        self.text_config.num_attention_heads = 8
        self.text_config.head_dim = 256
        self.text_config.num_hidden_layers = 18
        self.text_config.num_key_value_heads = 1
        self.text_config.hidden_activation = "gelu_pytorch_tanh"
        self.text_config.torch_dtype = "float32"
        self.text_config.vocab_size = 257152
        self.text_config.use_adarms = False
        self.text_config.adarms_cond_dim = None

        # Vision encoder (SigLIP) configuration
        self.vision_config.intermediate_size = 4304
        self.vision_config.projection_dim = 2048
        self.vision_config.projector_hidden_act = "gelu_fast"
        self.vision_config.torch_dtype = "float32"


@dataclass
class PI0ActionExpertConfig(GemmaConfig):
    """Configuration for the Gemma-300M action expert network."""

    def __init__(self):
        super().__init__(
            head_dim=256,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=8,
            num_hidden_layers=18,
            num_key_value_heads=1,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=False,
            adarms_cond_dim=None,
        )


@PreTrainedConfig.register_subclass("pi0_system1")
@dataclass
class PI0Config(PreTrainedConfig):
    """Main configuration for the PI0 policy (System 1 of Robo-MLT).

    PI0 uses flow matching to generate action chunks conditioned on
    multi-view images and language instructions.
    """

    # === Model Architecture ===
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "bfloat16"

    # === Action Prediction ===
    n_obs_steps: int = 1
    chunk_size: int = 50       # Number of actions predicted per chunk
    n_action_steps: int = 50   # Number of actions executed per chunk

    # Padding dimensions (must be >= actual state/action dims)
    max_state_dim: int = 32
    max_action_dim: int = 32

    # === Flow Matching Parameters ===
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # === Image Processing ===
    image_resolution: tuple[int, int] = (224, 224)
    empty_cameras: int = 0

    # === Tokenization ===
    tokenizer_max_length: int = 200


    ttrtc: bool = False
    ttrtc_max_delay: int = 10

    # === Normalization ===
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # === Training Settings ===
    gradient_checkpointing: bool = True
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    compile_cache_dir: str = "./torch_compile_cache"
    device: str | None = None

    # Attention/MLP fusion (disable for LoRA fine-tuning)
    fuse_qkv: bool = True
    fuse_gate_up: bool = True

    # === Optimizer Settings ===
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # === Scheduler Settings ===
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # === Sub-model Configurations ===
    vlm_config: PI0VLMConfig = field(default_factory=PI0VLMConfig)
    action_expert_config: PI0ActionExpertConfig = field(default_factory=PI0ActionExpertConfig)

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )
        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")
        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
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
        """Return optimizer configuration."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        """Return learning rate scheduler configuration."""
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
    # Tuple-typed scalar fields: ``json`` round-trips tuples as lists, so we
    # restore them to tuples to match how draccus would have decoded them.
    _TUPLE_FIELDS = ("image_resolution", "optimizer_betas")

    @classmethod
    def from_pretrained_local(cls, pretrained_path: str | os.PathLike) -> "PI0Config":
        """Build a :class:`PI0Config` directly from a checkpoint's ``config.json``.

        This is a drop-in, lerobot-free replacement for
        ``PreTrainedConfig.from_pretrained``.  We read the JSON ourselves instead
        of routing it through draccus, because draccus tries to decode the nested
        HuggingFace sub-configs (:class:`PI0VLMConfig` / :class:`PI0ActionExpertConfig`)
        as dataclasses and chokes on their inherited type annotations
        (``NameError: name 'torch' is not defined``).  The sub-configs hardcode
        every field in their ``__init__``, so their serialized values carry no
        information and are simply rebuilt via the ``default_factory``.

        Args:
            pretrained_path: Directory containing ``config.json``.

        Returns:
            A fully-initialised :class:`PI0Config`.
        """
        config_file = os.path.join(str(pretrained_path), "config.json")
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"No 'config.json' found in: {pretrained_path}")
        with open(config_file) as f:
            raw = json.load(f)

        # ``type`` is a derived property (registry choice name), and the nested
        # HF sub-configs are rebuilt from their default_factory — drop all three.
        for key in ("type", "vlm_config", "action_expert_config"):
            raw.pop(key, None)

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


__all__ = ["PI0Config", "PI0VLMConfig", "PI0ActionExpertConfig"]
