"""External, swappable prompt templates for the System 2 planner and replanner.

The same frozen vision--language model serves as the planner Phi_plan (paper Eq. 2) and the
replanner Phi_replan (Eq. 5). Per-task prompt text lives in YAML files under
``configs/skill_library/prompts/``; a file is loaded into a :class:`PromptSet` and handed to
:class:`~high_level_model.planning.planner.Planner`.

File format (all fields optional; missing fields fall back to the built-in defaults)::

    plan: |          # Eq. 2 inputs: {task} {protocol} {skills} {image}
      ...
    replan: |        # Eq. 5 inputs: {task} {protocol} {skills} {completed} {memory} {image}
      ...            #               plus {failure_type} {facts}
    multimodal: true # forward the scene image (I_0 for plan, I_t for replan) to the backend

``{image}`` renders a short note that a scene image is attached; the image itself travels as a
separate message part (see ``llm_backends.qwen_llm_fn``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "PromptSet",
    "load_prompt_set",
    "DEFAULT_PLAN_PROMPT",
    "DEFAULT_REPLAN_PROMPT",
]

DEFAULT_PLAN_PROMPT = """You are the high-level planner of a bimanual laboratory robot.
Given a TASK, the instrument PROTOCOL, a library of atomic SKILLS and the initial scene image,
output the ordered list of skill_ids that accomplishes the task. Use only skill_ids from the
library, follow the protocol order, and repeat skills as often as the scene requires.

TASK:
{task}

PROTOCOL:
{protocol}

SKILLS (skill_id: instruction | pre -> post):
{skills}

SCENE: {image}

Respond with ONLY a JSON array of skill_id strings, in execution order. Example:
["skill_a", "skill_b", "skill_c"]
"""

DEFAULT_REPLAN_PROMPT = """You are the high-level planner of a bimanual laboratory robot.
The active skill can no longer be completed. Re-plan the REMAINING steps from the CURRENT scene.

TASK:
{task}

PROTOCOL:
{protocol}

SKILLS (skill_id: instruction | pre -> post):
{skills}

ALREADY COMPLETED (in order): {completed}
FAILURE TYPE: {failure_type}
CURRENT STATE FACTS:
{facts}
PREVIOUSLY ATTEMPTED (skill -> outcome): {memory}
CURRENT SCENE: {image}

Plan only what can still be done in the scene as it now stands, and never violate the protocol
order. Output ONLY a JSON array of skill_id strings for the REMAINING steps, in execution order.
"""


@dataclass
class PromptSet:
    """Plan/replan prompt templates plus the flag that forwards scene images to the backend."""

    plan: str = DEFAULT_PLAN_PROMPT
    replan: str = DEFAULT_REPLAN_PROMPT
    multimodal: bool = True

    @classmethod
    def default(cls) -> PromptSet:
        return cls()


def load_prompt_set(path: str | None) -> PromptSet:
    """Load a :class:`PromptSet` from a YAML file (``None``/empty -> built-in defaults)."""
    if not path:
        return PromptSet.default()

    import yaml  # local import keeps planning yaml-free at import time

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        logger.warning("Prompt file %s is not a mapping; using defaults.", path)
        return PromptSet.default()
    if "perceive" in data:
        logger.warning("Prompt file %s has a 'perceive' template; it is no longer used — scene "
                       "perception is part of the plan prompt.", path)

    return PromptSet(
        plan=data.get("plan") or DEFAULT_PLAN_PROMPT,
        replan=data.get("replan") or DEFAULT_REPLAN_PROMPT,
        multimodal=bool(data.get("multimodal", True)),
    )
