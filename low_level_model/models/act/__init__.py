"""ACT policy (System 1 — Generative Executor) — thin adapter over lerobot.

ACT (Action Chunking Transformer) is fully implemented in lerobot, which is
already a dependency of this repo.  Rather than duplicate the model code, this
package re-exports lerobot's classes so ACT plugs into the Robo-MLT System-1
machinery exactly like the native PI0 policy:

- importing :class:`ACTConfig` registers ``@PreTrainedConfig.register_subclass("act")``
  so ``policy.type: act`` is selectable from YAML;
- :class:`ACTPolicy` follows the same ``PreTrainedPolicy`` contract
  (``forward``/``predict_action_chunk``/``select_action``/``reset``/``get_optim_params``)
  the async executors rely on;
- :func:`make_act_pre_post_processors` builds the normalise/device pipelines
  (ACT has no language tokenizer).

Wired into ``low_level_model.models.factory`` and ``train_system1`` via the
``"act"`` type string.
"""

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

__all__ = ["ACTConfig", "ACTPolicy", "make_act_pre_post_processors"]
