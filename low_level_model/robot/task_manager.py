"""Thread-safe skill/subtask switching shared by the robot scripts.

Maps keyboard digit keys onto the skills declared in a skill-library YAML
(`configs/skill_library/*.yaml`):

    digit 0     -> reset to the default (high-level) task text
    digits 1-9  -> the 1st..9th skill in declaration order

Used by both ``collect_dagger_dataset.py`` (the current skill text is recorded
as the per-frame ``subtask`` label) and ``robot_inference.py`` (manual override
of the System-2 decision).  The class is torch/lerobot-free so it can be unit
tested offline; the keyboard listener itself lives in ``keyboard_control.py``.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from high_level_model.planning.skill_library import SkillLibrary

logger = logging.getLogger(__name__)

__all__ = ["SkillTaskManager", "digit_of"]


def digit_of(key: str) -> Optional[int]:
    """Return the digit for a single-character key event, else None."""
    if isinstance(key, str) and len(key) == 1 and key.isdigit():
        return int(key)
    return None


class SkillTaskManager:
    """Digit-key -> skill switching over a :class:`SkillLibrary` (thread-safe).

    The *current* selection is either a skill_id from the library or ``None``,
    meaning the default task text (the high-level instruction passed on the
    CLI).  :meth:`current_text` is simultaneously the System-1 task string and
    the per-frame ``subtask`` label during DAgger collection.
    """

    def __init__(self, skill_library: SkillLibrary, default_task: str) -> None:
        self.library = skill_library
        self.default_task = default_task
        self._ordered_ids: list[str] = skill_library.ids()  # declaration order
        self._current_id: Optional[str] = None
        self._lock = threading.Lock()

    # --- switching -----------------------------------------------------------
    def skill_for_digit(self, digit: int) -> Optional[str]:
        """skill_id for digits 1-9 (declaration order), else None."""
        if 1 <= digit <= len(self._ordered_ids) and digit <= 9:
            return self._ordered_ids[digit - 1]
        return None

    def switch_to_digit(self, digit: int) -> Optional[str]:
        """Switch the current skill from a digit key.

        ``0`` resets to the default task (returns None); ``1-9`` switches to the
        corresponding skill and returns its skill_id; out-of-range digits are
        ignored (logged) and return None.
        """
        if digit == 0:
            self.reset_to_default()
            return None
        skill_id = self.skill_for_digit(digit)
        if skill_id is None:
            logger.warning("[task] digit %d has no skill (library has %d skills); ignored.",
                           digit, len(self._ordered_ids))
            return None
        with self._lock:
            self._current_id = skill_id
        logger.info("[task] switched to skill '%s': %s", skill_id, self.library.instruction_of(skill_id))
        return skill_id

    def reset_to_default(self) -> None:
        with self._lock:
            self._current_id = None
        logger.info("[task] reset to default task: %s", self.default_task)

    # --- queries --------------------------------------------------------------
    def current_skill_id(self) -> Optional[str]:
        with self._lock:
            return self._current_id

    def current_text(self) -> str:
        """Instruction of the current skill, or the default task text."""
        with self._lock:
            current = self._current_id
        return self.library.instruction_of(current) if current is not None else self.default_task

    def order_index(self, skill_id: str) -> int:
        """Position of *skill_id* in declaration order (raises ValueError if unknown)."""
        return self._ordered_ids.index(skill_id)

    def is_backward_switch(self, old_id: Optional[str], new_id: Optional[str]) -> bool:
        """True iff both ids are skills and *new* precedes *old* in declaration order.

        Used during DAgger collection: switching back to an earlier skill marks
        the operator's correction (back) window automatically.
        """
        if old_id is None or new_id is None:
            return False
        return self.order_index(new_id) < self.order_index(old_id)

    # --- UX --------------------------------------------------------------------
    def print_menu(self) -> None:
        """Log the digit -> skill instruction table (shown at startup)."""
        lines = ["[task] keyboard skill menu:", "  0: <default> %s" % self.default_task]
        for i, skill_id in enumerate(self._ordered_ids[:9], start=1):
            lines.append(f"  {i}: [{skill_id}] {self.library.instruction_of(skill_id)}")
        if len(self._ordered_ids) > 9:
            lines.append(f"  (… {len(self._ordered_ids) - 9} more skills not reachable by digit keys)")
        logger.info("\n".join(lines))
