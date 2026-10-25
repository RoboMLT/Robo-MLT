"""Frozen vision--language planner and replanner for System 2 (paper Eq. 2 and Eq. 5).

Given the task, the instrument protocol, the :class:`~skill_library.SkillLibrary` and a scene
image (I_0 when planning, I_t when replanning), the planner emits an ordered skill sequence that
may repeat skills. It is queried zero-shot and never fine-tuned; the paper instantiates it with
the Qwen3.7-Max API (see ``llm_backends``).

The model is injected as ``llm_fn(prompt, images=None) -> str``. Without one, the planner falls
back to one protocol cycle plus the terminal skill, which keeps the control logic testable offline.
Candidate plans are *not* checked here: the pipeline validates them against the protocol
constraints and re-queries within the replan budget (``system2_pipeline``).
"""

import json
import logging
import re
from typing import Callable, List, Optional

from high_level_model.planning.prompt_templates import (
    DEFAULT_PLAN_PROMPT,
    DEFAULT_REPLAN_PROMPT,
    PromptSet,
)
from high_level_model.planning.skill_library import SkillLibrary

logger = logging.getLogger(__name__)

__all__ = ["Planner"]

# ``fn(prompt)`` for a text-only backend; a multimodal backend additionally accepts an ``images``
# keyword (``fn(prompt, images=...)``), used when the PromptSet enables it. See _call_llm.
LLMFn = Callable[..., str]

# Backwards-compatible aliases; the canonical defaults now live in prompt_templates.
_PLAN_PROMPT = DEFAULT_PLAN_PROMPT
_REPLAN_PROMPT = DEFAULT_REPLAN_PROMPT


class Planner:
    """Turns a task (+ protocol + scene image) into an ordered skill plan."""
    def __init__(
        self,
        skill_library: SkillLibrary,
        llm_fn: Optional[LLMFn] = None,
        prompt_set: Optional[PromptSet] = None,
    ) -> None:
        self.library = skill_library
        self.llm_fn = llm_fn
        self.prompt_set = prompt_set or PromptSet.default()

    # ------------------------------------------------------------------ planning
    def plan(self, task_text: str, images=None) -> List[str]:
        """Return an ordered list of skill_ids accomplishing ``task_text`` (Eq. 2).

        ``images`` is the initial workspace observation I_0 (a list of HWC RGB uint8 arrays). It
        is forwarded to the backend when the PromptSet is multimodal, so the plan can be sized to
        the scene (one assay cycle per detected tube; one pick-and-place per sorted tube).
        """
        if self.llm_fn is None:
            logger.warning("No llm_fn provided; falling back to one protocol cycle + terminal.")
            return self.cycle_ids() + self.terminal_ids()
        prompt = self.prompt_set.plan.format(
            task=task_text, protocol=self._render_protocol(), skills=self._render_skills(),
            image=self._render_image(images),
        )
        raw = self._call_llm(prompt, images)
        plan = self._parse_plan(raw)
        if not plan:
            logger.warning("Planner returned no valid skills; falling back to one protocol cycle.")
            return self.cycle_ids() + self.terminal_ids()
        return plan

    def replan(
        self,
        task_text: str,
        facts: Optional[dict] = None,
        failure_type: Optional[str] = None,
        completed_ids: Optional[List[str]] = None,
        memory: Optional[List] = None,
        state_text: str = "",
        images=None,
        repeat_cycles: int = 1,
    ) -> List[str]:
        """Re-plan the remaining steps from the current state (Eq. 5).

        Args:
            task_text: the high-level task.
            facts: structured pre/postcondition facts of the skill that can no longer be completed.
            failure_type: "recovery_head" (beta_t debounce fired) or "stall_timeout" (T_max).
            completed_ids: the completed plan prefix C_t (frozen).
            memory: the short-term record R_t of attempted skills, as (skill_id, outcome) pairs.
            state_text: optional free-text fallback if no structured facts are available.
            images: the current observation I_t (see :meth:`plan`).
            repeat_cycles: how many more times the protocol's repeatable body still has to run —
                i.e. how many work items are left in the scene (samples on the bench, tubes in the
                rack); 0 leaves only the terminal skill. The offline fallback turns this straight
                into ``cycle * n + terminal``; the LLM path passes it as a fact so the model can
                size its own answer. Defaults to 1.
        """
        completed_ids = completed_ids or []
        n = max(0, int(repeat_cycles))   # 0: nothing left to process, only the terminal skill
        if self.llm_fn is None:
            # Offline fallback. The old behaviour ("declaration order minus whatever was already
            # done") cannot express a second pass over the bench: after one full cycle every skill
            # is in ``completed_ids``, so the remaining plan collapsed to just the terminal skill
            # and the pointer finished the episode while the robot was still working. Repeat the
            # cycle body instead, once per work item still in the scene.
            plan = self.cycle_ids() * n + self.terminal_ids()
            logger.warning("No llm_fn provided; replan falls back to %d x protocol cycle "
                           "+ terminal (%d steps).", n, len(plan))
            return plan
        prompt = self.prompt_set.replan.format(
            task=task_text,
            protocol=self._render_protocol(),
            skills=self._render_skills(),
            completed=json.dumps(completed_ids),
            failure_type=failure_type or "unknown",
            facts=self._render_facts({**(facts or {}), "work_items_remaining": n}, state_text),
            memory=self._render_memory(memory),
            image=self._render_image(images),
        )
        raw = self._call_llm(prompt, images)
        plan = self._parse_plan(raw)
        if not plan:
            plan = self.cycle_ids() * n + self.terminal_ids()
        return plan

    # ------------------------------------------------- protocol shape (offline fallback)
    def cycle_ids(self) -> List[str]:
        """The repeatable body of the protocol: ``protocol.cycle`` when the library declares it,
        else declaration order minus the terminal skill.

        Both shipped libraries end with ``return_home`` — a skill that runs once, after all work
        items are done — so "everything before the last entry" is the per-item cycle. The
        tube-sorting declaration order (grasp, grasp, place, place) is not admissible, so that
        library declares its cycle explicitly.
        """
        if self.library.protocol.cycle:
            return list(self.library.protocol.cycle)
        ids = self.library.ids()
        return ids[:-1] if len(ids) > 1 else list(ids)

    def terminal_ids(self) -> List[str]:
        """The once-per-episode tail of the protocol (``[]`` for a single-skill library)."""
        ids = self.library.ids()
        return ids[-1:] if len(ids) > 1 else []

    # ------------------------------------------------------------------ llm dispatch
    def _call_llm(self, prompt: str, images=None) -> str:
        """Invoke ``llm_fn``, forwarding images only when multimodal is enabled and supported.

        Falls back to the text-only ``llm_fn(prompt)`` signature for text-only backends.
        """
        if self.prompt_set.multimodal and images is not None:
            try:
                return self.llm_fn(prompt, images=images)
            except TypeError:
                logger.warning(
                    "llm_fn does not accept images; falling back to text-only prompt."
                )
        return self.llm_fn(prompt)

    @staticmethod
    def _render_image(images) -> str:
        """Textual stand-in for the ``{image}`` placeholder (the pixels travel separately)."""
        if images is None:
            return "(no image available)"
        try:
            n = len(images)
        except TypeError:
            n = 1
        return f"[{n} scene image(s) attached]"

    @staticmethod
    def _render_facts(facts: Optional[dict], state_text: str = "") -> str:
        if facts:
            return "\n".join(f"- {k}: {v}" for k, v in facts.items())
        return state_text or "(not provided)"

    @staticmethod
    def _render_memory(memory: Optional[List]) -> str:
        if not memory:
            return "(none)"
        parts = []
        for item in memory:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                parts.append(f"{item[0]} -> {item[1]}")
            else:
                parts.append(str(item))
        return "; ".join(parts)

    # ------------------------------------------------------------------ helpers
    def _render_protocol(self) -> str:
        return self.library.protocol.text or "(follow the skill preconditions)"

    def _render_skills(self) -> str:
        lines = []
        for s in self.library:
            pre = s.precondition_text or "-"
            post = s.postcondition_text or "-"
            lines.append(f"- {s.skill_id}: {s.canonical_instruction} | {pre} -> {post}")
        return "\n".join(lines)

    def _parse_plan(self, raw: str) -> List[str]:
        """Extract a JSON array of skill_ids from the LLM response and validate against the library.

        Repeats are **kept**: a lab protocol that processes four samples is literally
        ``[grasp, uncap, ..., discard, grasp, uncap, ..., discard, return_home]``, and a global
        de-duplication (which this used to do) silently collapses it to a single cycle — the plan
        then runs out long before the episode does. Only *consecutive* duplicates are folded,
        which is the failure mode a set was really guarding against (an LLM stuttering the same id).
        """
        if not raw:
            return []
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            logger.warning("No JSON array found in planner output: %r", raw[:200])
            return []
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Failed to parse planner JSON: %r", match.group(0)[:200])
            return []

        plan: List[str] = []
        for item in items:
            sid = str(item).strip()
            if sid not in self.library:
                logger.warning("Planner produced unknown skill_id '%s'; dropping.", sid)
                continue
            if plan and plan[-1] == sid:
                continue                      # stutter, not a second pass over the same skill
            plan.append(sid)
        return plan
