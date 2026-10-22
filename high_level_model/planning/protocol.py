"""Hard protocol-ordering constraints checked against every candidate plan (paper Sec. III-B).

The replanner is a frozen VLM, so nothing guarantees that a candidate plan respects the order the
instrument protocol prescribes. :class:`ProtocolConstraints` is the deterministic check that sits
between the VLM and the plan pointer: a candidate that violates an ordering rule is *rejected*
(never repaired), and the pipeline re-queries the replanner while the replan budget lasts.

Rules are declared next to the skills in the skill-library YAML::

    protocol:
      text: |                       # free-text protocol, also shown to the planner
        ...
      terminal: [return_home]       # may only appear as the last step
      cycle: [...]                  # optional: per-item body for the offline fallback planner
      rules:
        - skill: discard_sample     # every occurrence of this skill ...
          after: [withdraw_tube]    # ... needs one of these earlier ...
          since: [discard_sample]   # ... after the latest of these (default: [skill])

Read a rule as: "a tube cannot be discarded before it is withdrawn from the analyzer". The
``since`` reset makes the rule per-cycle, so plans that repeat the assay for several samples are
checked cycle by cycle. The check runs over the *whole* plan (completed prefix + candidate), which
is what lets a rule look back across the point where execution was interrupted.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = ["OrderingRule", "ProtocolConstraints"]


@dataclass(frozen=True)
class OrderingRule:
    """``skill`` requires one of ``after`` since the latest occurrence of any of ``since``."""

    skill: str
    after: Tuple[str, ...]
    since: Tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping) -> "OrderingRule":
        if "skill" not in data or not data.get("after"):
            raise ValueError(f"Ordering rule needs 'skill' and a non-empty 'after': {dict(data)}")
        skill = str(data["skill"])
        since = tuple(str(s) for s in (data.get("since") or [skill]))
        return cls(skill=skill, after=tuple(str(s) for s in data["after"]), since=since)

    def violations(self, plan: Sequence[str]) -> List[str]:
        out: List[str] = []
        for i, sid in enumerate(plan):
            if sid != self.skill:
                continue
            # Only look back to the latest reset point; the reset skill itself at index i
            # (``since`` defaults to [skill]) must not count, hence the strict ``j < i``.
            start = max((j for j in range(i) if plan[j] in self.since), default=-1)
            if not any(plan[j] in self.after for j in range(start + 1, i)):
                out.append(f"step {i} '{sid}' requires one of {list(self.after)} "
                           f"after the latest of {list(self.since)}")
        return out


@dataclass(frozen=True)
class ProtocolConstraints:
    """Protocol text plus the ordering rules that every executed plan must satisfy."""

    text: str = ""
    rules: Tuple[OrderingRule, ...] = field(default_factory=tuple)
    terminal: Tuple[str, ...] = field(default_factory=tuple)
    # Admissible per-item body used by the planner's offline fallback (no LLM). Empty means
    # "declaration order minus the terminal skill", which only works when that order is itself
    # admissible; the declaration order is pinned to the dataset's subtask_index order.
    cycle: Tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: Optional[Mapping]) -> "ProtocolConstraints":
        data = data or {}
        rules = tuple(OrderingRule.from_dict(r) for r in data.get("rules") or [])
        return cls(text=str(data.get("text") or "").strip(), rules=rules,
                   terminal=tuple(str(s) for s in data.get("terminal") or []),
                   cycle=tuple(str(s) for s in data.get("cycle") or []))

    def referenced_skills(self) -> List[str]:
        ids = set(self.terminal) | set(self.cycle)
        for r in self.rules:
            ids.update((r.skill, *r.after, *r.since))
        return sorted(ids)

    def violations(self, plan: Sequence[str]) -> List[str]:
        """Human-readable list of every violated constraint (empty = the plan is admissible)."""
        out: List[str] = []
        for rule in self.rules:
            out.extend(rule.violations(plan))
        for i, sid in enumerate(plan[:-1]):
            if sid in self.terminal:
                out.append(f"step {i} '{sid}' is terminal but is followed by further steps")
        return out

    def is_valid(self, plan: Sequence[str]) -> bool:
        return not self.violations(plan)

    def as_dict(self) -> Dict:
        return {
            "text": self.text,
            "terminal": list(self.terminal),
            "cycle": list(self.cycle),
            "rules": [{"skill": r.skill, "after": list(r.after), "since": list(r.since)}
                      for r in self.rules],
        }
