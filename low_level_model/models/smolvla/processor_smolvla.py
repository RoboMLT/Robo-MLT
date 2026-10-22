"""SmolVLA pre/post-processor pipelines with Robo-MLT atomic-skill conditioning.

lerobot's stock SmolVLA preprocessor only newline-pads and tokenises the global
``task`` field — it has **no** step that folds a per-frame atomic-skill instruction
(the ``subtask`` field) into the prompt.  In the Robo-MLT dual-system framework that
field is the whole point:

- training injects it via :class:`low_level_model.data.SubtaskInjectingDataset`
  (``dataset.skill_library`` in the YAML), and
- deployment feeds it via ``RobotAsyncExecutor`` (``batch["subtask"]`` =
  System-2's current skill instruction).

With the stock pipeline both are silently dropped, so SmolVLA can never be steered
by System 2 and ``dataset.skill_library`` becomes a no-op.  This wrapper rebuilds the
lerobot pipeline and inserts :class:`Pi0TaskSubtaskConcatProcessor` (a generic
"use the subtask as the prompt when present" step, shared with the PI0 path) just
before the newline/tokenizer steps — exactly mirroring how PI0 wires it in.

The inserted step is serialised into ``policy_preprocessor.json`` by its import path,
so it reconstructs automatically at inference (``DataProcessorPipeline.from_pretrained``)
with no registry entry required.
"""

from typing import Any

import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import (
    make_smolvla_pre_post_processors as _lerobot_make_smolvla_pre_post_processors,
)

from low_level_model.models.pi0.processor_pi0 import Pi0TaskSubtaskConcatProcessor

# Steps before which the subtask must already be merged into `task` (it has to happen
# before the prompt is newline-padded and tokenised).
_TEXT_STEP_NAMES = ("SmolVLANewLineProcessor", "TokenizerProcessorStep")


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
):
    """Build SmolVLA (pre, post) pipelines with atomic-skill (``subtask``) conditioning.

    Identical to lerobot's builder except a :class:`Pi0TaskSubtaskConcatProcessor`
    is inserted before the first text/tokenizer step so the current atomic-skill
    instruction is used as the prompt when present (falling back to the high-level
    ``task`` otherwise) — matching the PI0 System-1 behaviour.
    """
    pre, post = _lerobot_make_smolvla_pre_post_processors(config, dataset_stats=dataset_stats)

    # Idempotent: don't double-insert if a future lerobot version adds its own step.
    if not any(type(s).__name__ == "Pi0TaskSubtaskConcatProcessor" for s in pre.steps):
        insert_at = next(
            (i for i, s in enumerate(pre.steps) if type(s).__name__ in _TEXT_STEP_NAMES),
            len(pre.steps),
        )
        pre.steps.insert(insert_at, Pi0TaskSubtaskConcatProcessor())

    return pre, post


__all__ = ["make_smolvla_pre_post_processors"]
