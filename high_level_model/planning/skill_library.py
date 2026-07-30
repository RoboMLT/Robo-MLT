"""Declarative atomic-skill registry for System 2.

The library decouples *what skills exist* from the trained execution monitor. Adding a
skill is a data edit (one entry in a YAML/JSON file) plus attaching its System-1 policy;
it requires **no System-2 retraining**, because both the planner and the completion gate
condition on the skill's frozen SigLIP text embedding rather than a closed-set label.

Each :class:`Skill` carries the natural-language fields the rest of System 2 consumes:
    - ``canonical_instruction`` : text fed to the Temporal Selector / System 1.
    - ``precondition_text`` / ``postcondition_text`` : used by the completion gate
      (Stage 2) and by the LLM planner for ordering / replanning.
    - ``system1_policy_ref`` : opaque handle resolving to the low-level policy.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)

__all__ = ["Skill", "SkillLibrary", "load_skill_library"]


@dataclass(frozen=True)
class Skill:
    """A single atomic skill entry (immutable)."""

    skill_id: str
    canonical_instruction: str
    precondition_text: str = ""
    postcondition_text: str = ""
    system1_policy_ref: str = ""
    aliases: tuple = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: Dict) -> "Skill":
        """Build a Skill from a plain dict, ignoring unknown keys."""
        if "skill_id" not in data or "canonical_instruction" not in data:
            raise ValueError(
                f"Skill entry must define 'skill_id' and 'canonical_instruction': {data}"
            )
        return cls(
            skill_id=str(data["skill_id"]),
            canonical_instruction=str(data["canonical_instruction"]),
            precondition_text=str(data.get("precondition_text", "")),
            postcondition_text=str(data.get("postcondition_text", "")),
            system1_policy_ref=str(data.get("system1_policy_ref", "")),
            aliases=tuple(data.get("aliases", ()) or ()),
        )


class SkillLibrary:
    """Ordered collection of :class:`Skill`, keyed by ``skill_id``.

    The library is the open, extensible vocabulary that replaces the closed-set
    ``observation.subtask`` list hard-coded in the original System 2.
    """

    def __init__(self, skills: Optional[List[Skill]] = None) -> None:
        self._skills: Dict[str, Skill] = {}
        for skill in skills or []:
            self.add(skill)

    def add(self, skill: Skill) -> None:
        """Register a skill. Raises on duplicate id to avoid silent overwrites."""
        if skill.skill_id in self._skills:
            raise ValueError(f"Duplicate skill_id in library: '{skill.skill_id}'")
        self._skills[skill.skill_id] = skill

    def get(self, skill_id: str) -> Skill:
        if skill_id not in self._skills:
            raise KeyError(f"Unknown skill_id: '{skill_id}'")
        return self._skills[skill_id]

    def ids(self) -> List[str]:
        """Skill ids in insertion (declaration) order."""
        return list(self._skills.keys())

    def instructions(self) -> List[str]:
        """Canonical instructions, aligned with :meth:`ids`."""
        return [s.canonical_instruction for s in self._skills.values()]

    def instruction_of(self, skill_id: str) -> str:
        return self.get(skill_id).canonical_instruction

    def __contains__(self, skill_id: object) -> bool:
        return skill_id in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self) -> Iterator[Skill]:
        return iter(self._skills.values())

    @classmethod
    def from_dicts(cls, entries: List[Dict]) -> "SkillLibrary":
        return cls([Skill.from_dict(e) for e in entries])

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SkillLibrary":
        """Load a library from a JSON or YAML file.

        Accepted top-level shapes:
            - a list of skill dicts, or
            - a dict with a ``skills`` key holding that list.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Skill library file not found: {path}")

        raw = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required to load .yaml/.yml skill libraries; "
                    "install pyyaml or use a .json file."
                ) from exc
            data = yaml.safe_load(raw)
        elif suffix == ".json":
            data = json.loads(raw)
        else:
            raise ValueError(f"Unsupported skill library format '{suffix}': {path}")

        entries = data["skills"] if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise ValueError(
                f"Skill library must be a list of entries (or a dict with 'skills'): {path}"
            )
        lib = cls.from_dicts(entries)
        logger.info("Loaded %d skills from %s", len(lib), path)
        return lib


def load_skill_library(path: Union[str, Path]) -> SkillLibrary:
    """Convenience wrapper around :meth:`SkillLibrary.from_file`."""
    return SkillLibrary.from_file(path)
