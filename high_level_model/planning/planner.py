"""Frozen LLM planner for System 2 (Module A).

Given a high-level task and the :class:`~skill_library.SkillLibrary`, the planner emits an
ordered subsequence of ``skill_id`` s. It is *frozen* (zero-shot prompting) — extending the
skill set never retrains anything; the new skill simply appears in the prompt.

The LLM is injected as ``llm_fn: Callable[[str], str]`` so this module reuses whatever Qwen
client the deployment already has, without adding a dependency. When no ``llm_fn`` is given
the planner falls back to the library's declaration order, which keeps the rest of the
pipeline testable offline.
"""

import json
import logging
import re
from typing import Callable, Dict, List, Optional

from high_level_model.planning.prompt_templates import (
    DEFAULT_PLAN_PROMPT,
    DEFAULT_REPLAN_PROMPT,
    PromptSet,
)
from high_level_model.planning.skill_library import SkillLibrary

logger = logging.getLogger(__name__)

__all__ = ["Planner"]

# Default cap-colour -> place-skill mapping for the two-colour tube-sorting task. Used as the
# last-resort fallback when neither an explicit mapping nor one parsed from the task text is
# available. Red caps = blood-routine tubes, blue caps = sodium-citrate tubes.
DEFAULT_COLOR_TO_BOX: Dict[str, str] = {
    "red": "place_to_orange_box",
    "blue": "place_to_blue_box",
}

# The llm_fn is text-first: ``fn(prompt)``. A multimodal backend may additionally accept an
# ``images`` keyword (``fn(prompt, images=...)``); the planner detects this and only passes
# images when the active PromptSet enables it. See _call_llm.
LLMFn = Callable[..., str]

# Backwards-compatible aliases; the canonical defaults now live in prompt_templates.
_PLAN_PROMPT = DEFAULT_PLAN_PROMPT
_REPLAN_PROMPT = DEFAULT_REPLAN_PROMPT


class Planner:
    """Turns a task into an ordered skill plan (Module A)."""
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
        """Return an ordered list of skill_ids accomplishing ``task_text``.

        ``images`` is a reserved hook: when the active PromptSet has ``multimodal=True`` and a
        vision-capable ``llm_fn`` is provided, these images are forwarded to the backend. The
        default text path ignores it.
        """
        if self.llm_fn is None:
            logger.warning("No llm_fn provided; falling back to skill declaration order.")
            return self.library.ids()
        prompt = self.prompt_set.plan.format(
            task=task_text, skills=self._render_skills(), image=self._render_image(images)
        )
        raw = self._call_llm(prompt, images)
        plan = self._parse_plan(raw)
        if not plan:
            logger.warning("Planner returned no valid skills; falling back to declaration order.")
            return self.library.ids()
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
        """Re-plan the remaining steps from the current state (Stage 2 recovery loop).

        Args:
            task_text: the high-level task.
            facts: structured pre/postcondition facts, e.g.
                {"violated_skill": "insert_tube", "precondition": "...", "tube_in_gripper": False}.
            failure_type: "precondition_violation" | "stall" | "postcondition_mismatch" | "ood".
            completed_ids: already-executed skills (frozen prefix).
            memory: list of (skill_id, outcome) the controller already attempted.
            state_text: optional free-text fallback if no structured facts are available.
            images: reserved multimodal hook (see :meth:`plan`).
            repeat_cycles: how many more times the protocol's repeatable body still has to run —
                i.e. how many work items are left in the scene (samples on the bench, tubes in the
                rack). The offline fallback turns this straight into ``cycle * n + terminal``; the
                LLM path passes it as a fact so the model can size its own answer. Defaults to 1.
        """
        completed_ids = completed_ids or []
        n = max(1, int(repeat_cycles))
        if self.llm_fn is None:
            # Offline fallback. The old behaviour ("declaration order minus whatever was already
            # done") cannot express a second pass over the bench: after one full cycle every skill
            # is in ``completed_ids``, so the remaining plan collapsed to just the terminal skill
            # and the pointer finished the episode while the robot was still working. Repeat the
            # cycle body instead, once per work item still in the scene.
            plan = self.cycle_ids() * n + self.terminal_ids()
            logger.warning("No llm_fn provided; replan falls back to %d x declaration-order cycle "
                           "+ terminal (%d steps).", n, len(plan))
            return plan
        prompt = self.prompt_set.replan.format(
            task=task_text,
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
        """The repeatable body of the protocol: declaration order minus the terminal skill.

        Both shipped libraries end with ``return_home`` — a skill that runs once, after all work
        items are done — so "everything before the last entry" is the per-item cycle. Libraries
        with a different shape should override the plan through the LLM path.
        """
        ids = self.library.ids()
        return ids[:-1] if len(ids) > 1 else list(ids)

    def terminal_ids(self) -> List[str]:
        """The once-per-episode tail of the protocol (``[]`` for a single-skill library)."""
        ids = self.library.ids()
        return ids[-1:] if len(ids) > 1 else []

    # --------------------------------------------------------- tube-sorting (vision)
    def plan_tube_sorting(
        self,
        task_text: str,
        images,
        roi=None,
        color_to_box: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Build a pick-and-place plan for the two-colour tube-sorting task.

        A vision LLM reads the rack layout from ``images`` (the top-down camera frame, optionally
        cropped to ``roi``), then Python deterministically turns the layout into a flat, repeated
        skill-id plan honouring the cap-colour -> box rule, serving tubes in top-to-bottom,
        right-to-left order::

            [pick_next_tube, place_to_orange_box, pick_next_tube, place_to_blue_box, ...,
             return_home]

        Args:
            task_text: the high-level task (also used to parse the colour->box rule).
            images: one or more HWC scene images (numpy arrays or image-url strings).
            roi: optional ``[x, y, w, h]`` crop applied to every image before perception.
            color_to_box: optional explicit ``{"red": skill_id, "blue": skill_id}`` mapping; when
                omitted it is parsed from ``task_text`` and finally falls back to
                :data:`DEFAULT_COLOR_TO_BOX`.

        Returns the ordered skill-id plan; ``[]`` if perception yields nothing.
        """
        mapping = color_to_box or self.parse_color_to_box(task_text) or DEFAULT_COLOR_TO_BOX
        layout = self.perceive_rack(task_text, images, roi)
        return self._build_sorting_plan(layout, mapping)

    def perceive_rack(self, task_text: str, images, roi=None) -> List[List[str]]:
        """Read the rack layout via the vision LLM: a grid of ``"red"``/``"blue"`` cap colours.

        Rows are top-to-bottom, cells left-to-right. Returns ``[]`` when no ``llm_fn`` is set or
        the response cannot be parsed (callers then fall back / surface an empty plan).
        """
        if self.llm_fn is None:
            logger.warning("No llm_fn provided; cannot perceive rack layout.")
            return []
        cropped = self._crop_to_roi(images, roi)
        prompt = self.prompt_set.perceive.format(
            task=task_text, image=self._render_image(cropped)
        )
        raw = self._call_llm(prompt, cropped)
        return self._parse_layout(raw)

    @staticmethod
    def parse_color_to_box(task_text: str) -> Dict[str, str]:
        """Keyword-parse the cap-colour -> place-skill mapping from the user's task text.

        Recognises which destination box (orange / blue) each tube type (blood-routine/red cap vs
        sodium-citrate/blue cap) should go to, in either Chinese or English. Returns an empty dict
        when the text is ambiguous, so the caller can fall back to a default mapping.
        """
        if not task_text:
            return {}
        text = task_text.lower()

        def _box_after(*keywords: str) -> Optional[str]:
            # Find the box keyword that appears nearest *after* any of the tube/colour keywords.
            best: Optional[str] = None
            best_pos = len(text) + 1
            for kw in keywords:
                start = text.find(kw)
                if start < 0:
                    continue
                window = text[start:]
                orange = window.find("orange")
                if orange < 0:
                    orange = window.find("橙")
                blue = window.find("blue box")
                if blue < 0:
                    blue = window.find("蓝")
                for box_name, pos in (("place_to_orange_box", orange), ("place_to_blue_box", blue)):
                    if pos >= 0 and start + pos < best_pos:
                        best, best_pos = box_name, start + pos
            return best

        red_box = _box_after("blood", "血常规", "红", "red cap", "red tube")
        blue_box = _box_after("citrate", "柠檬酸钠", "蓝帽", "blue cap", "blue tube")
        mapping: Dict[str, str] = {}
        if red_box:
            mapping["red"] = red_box
        if blue_box:
            mapping["blue"] = blue_box
        # Require both ends resolved and distinct boxes; otherwise treat as ambiguous.
        if len(mapping) == 2 and mapping["red"] != mapping["blue"]:
            return mapping
        return {}

    def _build_sorting_plan(
        self, layout: List[List[str]], color_to_box: Dict[str, str]
    ) -> List[str]:
        """Flatten the rack grid (scan order) into a repeated pick/place plan + return_home."""
        # Pick/place order: rows top-to-bottom, and within each row RIGHT-TO-LEFT (the rack is
        # served from the rightmost tube of the top row). Perception reports the grid in natural
        # reading order (left-to-right); we reverse each row here so ordering stays deterministic
        # and independent of how the VLM lists cells.
        plan: List[str] = []
        for row in layout:
            for color in reversed(row):
                if color == "empty":
                    continue  # empty slot: no tube to pick here
                place = color_to_box.get(color)
                if place is None:
                    logger.warning("No box mapping for cap colour '%s'; skipping tube.", color)
                    continue
                if place not in self.library:
                    logger.warning("Place skill '%s' not in library; skipping tube.", place)
                    continue
                plan.append("pick_next_tube")
                plan.append(place)
        if plan and "return_home" in self.library:
            plan.append("return_home")
        return plan

    @staticmethod
    def _crop_to_roi(images, roi):
        """Crop every HWC image to ``roi = [x, y, w, h]``; pass images through when roi is None."""
        if roi is None or images is None:
            return images
        try:
            import numpy as np

            x, y, w, h = (int(v) for v in roi)
        except (TypeError, ValueError):
            logger.warning("Invalid roi %r; skipping crop.", roi)
            return images

        def _crop_one(img):
            if isinstance(img, str):
                return img  # already an image-url; cannot crop here
            arr = np.asarray(img)
            if arr.ndim < 2:
                return img
            return arr[y : y + h, x : x + w]

        if isinstance(images, str):
            return images
        try:
            return [_crop_one(im) for im in images]
        except TypeError:
            return _crop_one(images)

    def _parse_layout(self, raw: str) -> List[List[str]]:
        """Parse a ``{"rows": [[...], ...]}`` JSON object into a grid of ``"red"``/``"blue"``.

        Distinct from :meth:`_parse_plan` (which dedupes skill-ids): the layout keeps every cell
        and its order. Unknown cells are normalised to ``"red"``/``"blue"`` where possible and
        dropped (with a warning) otherwise.
        """
        if not raw:
            return []
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            logger.warning("No JSON object found in perception output: %r", raw[:200])
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Failed to parse perception JSON: %r", match.group(0)[:200])
            return []
        rows = data.get("rows") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            logger.warning("Perception JSON has no 'rows' list: %r", str(data)[:200])
            return []

        layout: List[List[str]] = []
        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            parsed_row: List[str] = []
            for cell in row:
                color = self._normalize_color(cell)
                if color is None:
                    logger.warning("Unrecognised cap value '%s'; dropping.", cell)
                    continue
                parsed_row.append(color)
            layout.append(parsed_row)
        return layout

    @staticmethod
    def _normalize_color(cell) -> Optional[str]:
        """Normalise a cell to ``"red"``, ``"blue"`` or ``"empty"`` (tolerant of synonyms).

        Empty slots (the rack is not always full) are kept as ``"empty"`` so the grid stays
        column-aligned; only truly unrecognised values return ``None`` (dropped by the caller).
        """
        s = str(cell).strip().lower()
        if s in ("", "-", "n/a", "na") or any(
            k in s for k in ("empty", "none", "null", "vacant", "no tube", "空", "无")
        ):
            return "empty"
        if any(k in s for k in ("red", "blood", "血常规", "红")):
            return "red"
        if any(k in s for k in ("blue", "citrate", "柠檬酸钠", "蓝")):
            return "blue"
        return None

    # ------------------------------------------------------------------ llm dispatch
    def _call_llm(self, prompt: str, images=None) -> str:
        """Invoke ``llm_fn``, forwarding images only when multimodal is enabled and supported.

        Falls back to the text-only ``llm_fn(prompt)`` signature so existing backends keep
        working unchanged.
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
        """Textual stand-in for the optional ``{image}`` placeholder in a template.

        Returns an empty string when no images are supplied. Templates that don't reference
        ``{image}`` simply ignore this (str.format drops unused kwargs)."""
        if images is None:
            return ""
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
