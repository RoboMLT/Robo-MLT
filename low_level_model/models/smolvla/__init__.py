"""SmolVLA policy (System 1 — Generative Executor) — thin adapter over lerobot.

SmolVLA (a compact VLA built on the SmolVLM2 backbone + a flow-matching action
expert) is fully implemented in lerobot, which is already a dependency of this
repo.  Rather than duplicate the model code, this package re-exports lerobot's
classes so SmolVLA plugs into the Robo-MLT System-1 machinery exactly like the
native PI0 policy:

- importing :class:`SmolVLAConfig` registers
  ``@PreTrainedConfig.register_subclass("smolvla")`` so ``policy.type: smolvla``
  is selectable from YAML;
- :class:`SmolVLAPolicy` follows the same ``PreTrainedPolicy`` contract the async
  executors rely on.  Its SmolVLM2 backbone is loaded inside the model's
  ``__init__`` via ``transformers`` (only ``model.safetensors`` lives in the
  checkpoint);
- :func:`make_smolvla_pre_post_processors` builds the pipelines, including the
  SmolVLM tokenizer step that produces ``observation.language.*`` tensors.

Wired into ``low_level_model.models.factory`` and ``train_system1`` via the
``"smolvla"`` type string.
"""

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from low_level_model.models.smolvla.processor_smolvla import make_smolvla_pre_post_processors

__all__ = ["SmolVLAConfig", "SmolVLAPolicy", "make_smolvla_pre_post_processors"]
