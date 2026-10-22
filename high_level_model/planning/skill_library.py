"""Declarative atomic-skill library S = {s_j} for System 2 (paper Sec. III-A, Eq. 1).

The frozen planner re-orders and repeats the skills declared here without any gradient update.
A genuinely new laboratory operation is different: it needs System-1 demonstrations and further
System-2 training (paper Sec. IV-C), because the monitor has only seen the skills it was trained on.

Each :class:`Skill` carries the natural-language fields the rest of System 2 consumes:
    - ``canonical_instruction`` : the instruction l_j fed to the monitor and to System 1. It must
      match ``meta/subtasks.parquet`` of the training data byte for byte.
    - ``precondition_text`` / ``postcondition_text`` : facts given to the replanner.
    - ``system1_policy_ref`` : informational tag; one skill-conditioned System 1 serves all skills.

The optional top-level ``protocol`` block (protocol text + ordering rules) is parsed into
:class:`~high_level_model.planning.protocol.ProtocolConstraints`.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from high_level_model.planning.protocol import ProtocolConstraints

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

    ``protocol`` holds the instrument protocol and its hard ordering constraints (empty when the
    library file declares none).
    """

    def __init__(self, skills: Optional[List[Skill]] = None,
                 protocol: Optional[ProtocolConstraints] = None) -> None:
        self._skills: Dict[str, Skill] = {}
        for skill in skills or []:
            self.add(skill)
        self.protocol = protocol or ProtocolConstraints()
        unknown = [sid for sid in self.protocol.referenced_skills() if sid not in self._skills]
        if unknown:
            raise ValueError(f"Protocol rules reference unknown skill_ids: {unknown}")

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
    def from_dicts(cls, entries: List[Dict], protocol: Optional[Dict] = None) -> "SkillLibrary":
        return cls([Skill.from_dict(e) for e in entries], ProtocolConstraints.from_dict(protocol))

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SkillLibrary":
        """Load a library from a JSON or YAML file.

        Accepted top-level shapes:
            - a list of skill dicts, or
            - a dict with a ``skills`` key holding that list (and an optional ``protocol`` block).
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
        protocol = data.get("protocol") if isinstance(data, dict) else None
        lib = cls.from_dicts(entries, protocol)
        logger.info("Loaded %d skills (%d protocol rules) from %s",
                    len(lib), len(lib.protocol.rules), path)
        return lib


def load_skill_library(path: Union[str, Path]) -> SkillLibrary:
    """Convenience wrapper around :meth:`SkillLibrary.from_file`."""
    return SkillLibrary.from_file(path)
