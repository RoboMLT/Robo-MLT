"""External, swappable prompt templates for the System 2 planner.

The planner's LLM prompts used to be hardcoded module constants in ``planner.py``. To run the
same planner on different tasks (different protocols, wording, constraints) without editing
code, the prompt text now lives in YAML files alongside the skill libraries, under
``configs/skill_library/prompts/``. A file is loaded into a :class:`PromptSet` and handed to
:class:`~high_level_model.planning.planner.Planner`.

File format (all fields optional; missing fields fall back to the built-in defaults)::

    plan: |          # initial-plan template; placeholders {task} {skills} (optional {image})
      ...
    replan: |        # recovery template; placeholders {task} {skills} {completed}
      ...            #                     {failure_type} {facts} {memory}
    multimodal: false  # reserved: when true, the planner may pass images to the LLM backend

The ``{image}`` placeholder and the ``multimodal`` flag are a reserved, forward-looking
interface: the text path is unchanged today, but a vision-language backend can later inject
image context. ``{image}`` is optional in the template — if absent it is simply not rendered.
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
    "DEFAULT_PERCEIVE_PROMPT",
]

# The built-in defaults (previously planner._PLAN_PROMPT / _REPLAN_PROMPT). Kept here so both
# the planner fallback and PromptSet.default() share one source of truth.
DEFAULT_PLAN_PROMPT = """You are the high-level planner of a bimanual laboratory robot.
Given a TASK and a library of atomic SKILLS, output the ordered list of skill_ids that
accomplishes the task. Use only skill_ids from the library. Respect preconditions: a skill
may only follow skills that establish its precondition.

TASK:
{task}

SKILLS (skill_id: instruction | pre -> post):
{skills}

Respond with ONLY a JSON array of skill_id strings, in execution order. Example:
["skill_a", "skill_b", "skill_c"]
"""

DEFAULT_REPLAN_PROMPT = """You are the high-level planner of a bimanual laboratory robot.
The current plan failed or was interrupted. Re-plan the REMAINING steps from the CURRENT STATE.

TASK:
{task}

SKILLS (skill_id: instruction | pre -> post):
{skills}

ALREADY COMPLETED (in order): {completed}
FAILURE TYPE: {failure_type}
CURRENT STATE FACTS:
{facts}
PREVIOUSLY ATTEMPTED (skill -> outcome): {memory}

To recover, you may re-insert a previously completed skill if its effect was undone (this is how
"go back" is expressed). Respect preconditions.
Output ONLY a JSON array of skill_id strings for the REMAINING steps, in execution order.
"""

# Vision perception template (multimodal). Unlike plan/replan (which emit skill_ids), this asks
# the VLM to *read the scene* and return a structured layout that downstream Python turns into a
# plan. Used by Planner.perceive_rack for the tube-sorting task. Placeholder: {task} (optional
# {image}). The default is a generic rack reader; per-task files override it.
DEFAULT_PERCEIVE_PROMPT = """You are the perception module of a laboratory robot. Look at the
top-down camera image of a sample-tube rack and read its layout.

TASK:
{task}

The rack is a fixed grid of slots; a slot may hold a tube or be EMPTY. Rows are NOT always full.
Scan the rack row by row, top to bottom; within each row go left to right over EVERY slot
position. For each slot report exactly one of:
  - "red"   : a tube whose CAP is red (blood-routine tube)
  - "blue"  : a tube whose CAP is blue (sodium-citrate tube)
  - "empty" : no tube in that slot

HOW TO CLASSIFY — read carefully:
  - Judge ONLY by the colour of the plastic screw CAP at the very TOP of the tube.
  - IGNORE everything else: the rubber stopper / inner plug (which often has a faint bluish or
    grey tint), the blood or liquid inside, the glass/plastic tube body, labels, shadows and
    reflections. None of these decide the class.
  - A tube is "blue" ONLY if its outer top cap is clearly blue. A faintly bluish rubber head on
    an otherwise red-capped tube is still "red".
  - If the cap looks dark red / maroon / pink, classify it as "red".

Keep every row the same length (the rack's number of columns) by padding missing tubes with
"empty", so the column positions stay aligned.

Respond with ONLY a JSON object mapping the rack to a grid, one inner list per row, e.g. a 3x4
rack whose 2nd row is half-empty:
{{"rows": [["red", "blue", "red", "red"], ["blue", "red", "empty", "empty"], ["red", "blue", "red", "blue"]]}}
Do not add any text outside the JSON object.
"""


@dataclass
class PromptSet:
    """Plan/replan/perceive prompt templates plus a multimodal flag.

    ``perceive`` is the vision-perception template consumed by
    :meth:`~high_level_model.planning.planner.Planner.perceive_rack`; ``plan``/``replan`` are the
    skill-id planning templates. ``multimodal=True`` lets the planner forward scene images to a
    vision LLM backend (e.g. ``qwen-vl-max``).
    """

    plan: str = DEFAULT_PLAN_PROMPT
    replan: str = DEFAULT_REPLAN_PROMPT
    perceive: str = DEFAULT_PERCEIVE_PROMPT
    multimodal: bool = False

    @classmethod
    def default(cls) -> "PromptSet":
        return cls()


def load_prompt_set(path: str | None) -> PromptSet:
    """Load a :class:`PromptSet` from a YAML file.

    ``path`` may be ``None`` or empty, in which case the built-in defaults are returned so the
    planner behaves exactly as before. Missing keys in the file fall back to the defaults too.
    """
    if not path:
        return PromptSet.default()

    import yaml  # local import keeps planning torch/yaml-free at import time

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        logger.warning("Prompt file %s is not a mapping; using defaults.", path)
        return PromptSet.default()

    return PromptSet(
        plan=data.get("plan") or DEFAULT_PLAN_PROMPT,
        replan=data.get("replan") or DEFAULT_REPLAN_PROMPT,
        perceive=data.get("perceive") or DEFAULT_PERCEIVE_PROMPT,
        multimodal=bool(data.get("multimodal", False)),
    )
