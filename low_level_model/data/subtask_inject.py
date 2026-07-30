"""Inject per-frame atomic-skill language into System-1 training samples.

System 1 (PI0) is deployed *under* System 2, which feeds a **per-skill** instruction
(the skill's ``canonical_instruction``) as the language prompt — see
``robot_common.RobotAsyncExecutor`` setting ``batch["subtask"]`` and
``Pi0TaskSubtaskConcatProcessor`` building ``task: <high> subtask: <atomic>\\n``.

If training only ever conditions on the global task string (the dataset has a single
``task`` and no ``subtask`` text), the policy never learns to be steered by the atomic
instruction, and the train/deploy prompts diverge (train: ``<high>\\n`` vs deploy:
``task:  subtask: <atomic>\\n``). This module maps each frame's ``subtask_index`` to
its skill instruction and adds it under the ``subtask`` key, so the **same** processor
produces matching prompts at train and deploy time.

The index is the skill's position in the library's ``skills`` list, which matches the
recorded ``subtask_index`` column (and the keyboard shortcuts in the skill library).
"""

import logging
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

__all__ = ["build_subtask_map", "SubtaskInjectingDataset"]


def _coerce_int(x) -> int:
    """Robustly coerce a hf_dataset cell (scalar or shape-(1,) array/tensor) to int."""
    try:
        return int(x)
    except (TypeError, ValueError):
        return int(np.asarray(x).reshape(-1)[0])


def build_subtask_map(skill_library_path: str) -> dict[int, str]:
    """Map ``subtask_index`` → atomic-skill ``canonical_instruction``.

    Args:
        skill_library_path: path to a ``configs/skill_library/*.yaml`` file with a
            ``skills:`` list, each entry carrying a ``canonical_instruction``.

    Returns:
        ``{position_index: canonical_instruction}`` for every skill with a non-empty
        instruction. Position order must match the dataset's recorded ``subtask_index``.
    """
    import yaml

    with open(skill_library_path) as f:
        lib = yaml.safe_load(f) or {}
    skills = lib.get("skills", []) or []
    mapping: dict[int, str] = {}
    for i, sk in enumerate(skills):
        instr = (sk.get("canonical_instruction") or "").strip()
        if instr:
            mapping[i] = instr
    if not mapping:
        raise ValueError(
            f"No skills with a 'canonical_instruction' found in {skill_library_path}"
        )
    return mapping


class SubtaskInjectingDataset(Dataset):
    """Wrap a ``LeRobotDataset``(-like) dataset to add a per-frame ``subtask`` string.

    The wrapper transparently proxies every other dataset attribute (``meta``,
    ``hf_dataset``, ``num_frames``, ``num_episodes`` …) so it is a drop-in replacement
    that still works with back-event filtering, ``SubsetRandomSampler`` and the
    shared-observation collate. The ``subtask_index`` is read from ``hf_dataset`` at the
    *observation* frame, so it is correct under delay augmentation (where the action
    chunk — but not the observation — is shifted) and shared-observation mode (where the
    per-frame index column is otherwise dropped from the item).
    """

    def __init__(self, base, subtask_map: dict[int, str], column: str = "subtask_index"):
        self.base = base
        self.subtask_map = subtask_map
        self.column = column

    def __len__(self) -> int:
        return len(self.base)

    def _subtask_index(self, idx: int) -> Optional[int]:
        try:
            raw = self.base.hf_dataset[int(idx)]
        except Exception:  # noqa: BLE001 - missing column / out-of-range -> no injection
            return None
        if self.column not in raw:
            return None
        return _coerce_int(raw[self.column])

    def __getitem__(self, idx):
        item = self.base[idx]
        si = self._subtask_index(idx)
        if si is not None:
            item["subtask"] = self.subtask_map.get(si, "")
        return item

    def __getattr__(self, name):
        # Only invoked when normal attribute lookup fails; proxy to the wrapped dataset.
        # Guard dunders so pickling (num_workers > 0) does not recurse before ``base`` is set.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "base"), name)