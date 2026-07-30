"""Offline closed-loop test: DAgger collection -> completion-gate training data.

Run:  python -m low_level_model.tests.test_dagger_recorder   (from the repo root)

No robot / no GPU needed.  Drives ``LeRobotV3Recorder`` + ``DaggerSession`` +
``SkillTaskManager`` with synthetic frames and a scripted key sequence, then
verifies the full consumer chain:

    - ``meta/subtasks.parquet`` readable via lerobot's ``load_subtasks`` with
      row order == index value order;
    - ``back_annotations.json`` windows match the simulated manual ``e`` windows
      (resampled, episode-local frame indices);
    - a re-loaded ``LeRobotDataset`` resolves per-frame ``item["subtask"]``;
    - ``CompletionGateDataset`` builds from the collected data and yields the
      expected skill texts and back targets (the training-side contract);
    - resuming the dataset preserves the subtask map and appends new labels.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import tempfile

import numpy as np

from high_level_model.planning.skill_library import SkillLibrary
from low_level_model.robot.collect_dagger_dataset import (
    DaggerSession,
    LeRobotV3Recorder,
    handle_keys,
)
from low_level_model.robot.task_manager import SkillTaskManager

HERE = os.path.dirname(os.path.abspath(__file__))
# low_level_model/tests -> repo root -> top-level configs/
LIB_PATH = os.path.join(HERE, "..", "..", "configs", "skill_library", "bloodgas.yaml")
TASK = "Pick up the green tube, analyze it, and return it to the rack."
CAM_KEY = "observation.images.cam_top"
FPS = 20.0
STATE_DIM = 6


class _StubExecutor:
    """Minimal stand-in for RobotAsyncExecutor (DaggerSession only calls reset)."""

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


def _record_episode(recorder: LeRobotV3Recorder, session: DaggerSession,
                    keys_at_frame: dict[int, list[str]], n_frames: int, t0: float) -> None:
    """Simulate one recorded episode driven through the real handle_keys path."""
    rng = np.random.default_rng(0)
    eq: "queue.Queue" = queue.Queue()
    eq.put("c")  # start recording
    handle_keys(eq, recorder, session)
    for i in range(n_frames):
        for key in keys_at_frame.get(i, []):
            eq.put(key)
        handle_keys(eq, recorder, session)
        recorder.record(
            state=rng.standard_normal(STATE_DIM).astype(np.float32),
            action=rng.standard_normal(STATE_DIM).astype(np.float32),
            images={CAM_KEY: rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)},
            task=TASK, subtask=session.current_subtask(), back=session.back_active,
        )
        # Overwrite the wall-clock timestamp for deterministic resampling
        # (exact 1/FPS spacing -> nearest-neighbour resample maps 1:1).
        recorder._buf[-1]["timestamp"] = t0 + i / FPS
    eq.put("s")  # save the episode
    handle_keys(eq, recorder, session)


def main() -> None:
    root = os.path.join(tempfile.mkdtemp(prefix="dagger_test_"), "dataset")
    library = SkillLibrary.from_file(LIB_PATH)
    instr = library.instructions()  # declaration order: digit i -> instr[i-1]

    try:
        # ---------------------------------------------------------------- collect
        manager = SkillTaskManager(library, TASK)
        executor = _StubExecutor()
        session = DaggerSession(manager, teleop=None, executor=executor, default_task=TASK)
        recorder = LeRobotV3Recorder(repo_id="test/dagger", root=root, target_fps=int(FPS),
                                     robot_type="test_robot", use_videos=False, resume=True)

        # Episode 0: skill1 -> skill2 -> skill1, with an explicit manual back window.
        # Subtask segments: [0-9]=s1, [10-14]=s2, [15-19]=s1, [20-...]=s2.
        # Back frames: explicit 'e' toggle on at 15 and off at 25 -> [15, 24].
        _record_episode(recorder, session, n_frames=30, t0=1000.0, keys_at_frame={
            0: ["1"], 10: ["2"], 15: ["1", "e"], 20: ["2"], 25: ["e"],
        })
        assert not session.back_active, "back windows must be cleared after save"

        # Episode 1: single skill (digit 3), no corrections.
        _record_episode(recorder, session, n_frames=30, t0=2000.0, keys_at_frame={0: ["3"]})

        recorder.finalize()

        # ---------------------------------------------------------------- subtasks.parquet
        from lerobot.datasets.utils import load_subtasks

        df = load_subtasks(__import__("pathlib").Path(root))
        assert df is not None, "meta/subtasks.parquet must exist"
        assert list(df["subtask_index"]) == list(range(len(df))), \
            "row order must equal subtask_index order (lerobot resolves text positionally)"
        assert list(df.index[:3]) == [instr[0], instr[1], instr[2]], \
            f"first-seen order expected, got {list(df.index)}"
        print(f"   subtasks.parquet OK ({len(df)} labels, ordered)")

        # ---------------------------------------------------------------- back_annotations.json
        ann_path = os.path.join(root, "back_annotations.json")
        entries = json.loads(open(ann_path, encoding="utf-8").read())
        assert entries == [{"episode": 0, "frame_start": 15, "frame_end": 24}], \
            f"unexpected back windows: {entries}"
        print("   back_annotations.json OK (one merged window [15, 24] in episode 0)")

        # ---------------------------------------------------------------- reload as LeRobotDataset
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dset = LeRobotDataset("test/dagger", root=root, tolerance_s=1e-4,
                              revision=LeRobotV3Recorder._local_version(root))
        ep0_start = int(dset.meta.episodes[0]["dataset_from_index"])
        assert dset[ep0_start + 5]["subtask"] == instr[0], "frame 5 must be skill 1"
        assert dset[ep0_start + 12]["subtask"] == instr[1], "frame 12 must be skill 2"
        assert dset[ep0_start + 16]["subtask"] == instr[0], "frame 16 must be skill 1 (back)"
        print("   LeRobotDataset reload OK (per-frame item['subtask'] resolves)")

        # ---------------------------------------------------------------- CompletionGateDataset
        from high_level_model.data.completion_gate_dataset import CompletionGateDataset

        gate_ds = CompletionGateDataset(
            dset, camera_names=[CAM_KEY], history_len=2, prediction_offset=0,
            history_skip_frame=1, use_command_in_meta=False,
            back_annotations_path=ann_path, annotation_is_local=True,
        )
        by_frame = {frame: pos for pos, (_, frame) in enumerate(gate_ds.samples)}

        def _item(frame_abs):
            _, _, skill_text, progress, back, _, _ = gate_ds[by_frame[ep0_start + frame_abs]]
            return skill_text, progress, back

        skill_text, _, back = _item(5)
        assert skill_text == instr[0] and back == 0.0, f"frame 5: ({skill_text!r}, back={back})"
        skill_text, _, back = _item(20)
        assert skill_text == instr[1] and back == 1.0, f"frame 20: ({skill_text!r}, back={back})"
        skill_text, _, back = _item(16)
        assert skill_text == instr[0] and back == 1.0, f"frame 16: ({skill_text!r}, back={back})"
        print(f"   CompletionGateDataset OK ({len(gate_ds)} samples; skill text + back targets match)")

        # ---------------------------------------------------------------- resume
        manager2 = SkillTaskManager(library, TASK)
        session2 = DaggerSession(manager2, teleop=None, executor=_StubExecutor(), default_task=TASK)
        recorder2 = LeRobotV3Recorder(repo_id="test/dagger", root=root, target_fps=int(FPS),
                                      robot_type="test_robot", use_videos=False, resume=True)
        _record_episode(recorder2, session2, n_frames=30, t0=3000.0, keys_at_frame={0: ["4"]})
        assert recorder2._subtask_to_index[instr[0]] == 0, "resume must restore the old map"
        assert recorder2._subtask_to_index[instr[3]] == len(df), "new label must append"
        recorder2.finalize()
        df2 = load_subtasks(__import__("pathlib").Path(root))
        assert len(df2) == len(df) + 1 and list(df2["subtask_index"]) == list(range(len(df2)))
        entries2 = json.loads(open(ann_path, encoding="utf-8").read())
        assert entries2 == entries, "no new back windows expected from episode 2"
        print("   resume OK (map restored, new label appended)")

        # ---------------------------------------------------------------- teleop session logic
        class _StubTeleop:
            pass

        ex = _StubExecutor()
        s = DaggerSession(manager, teleop=_StubTeleop(), executor=ex, default_task=TASK)
        assert s.toggle_teleop() is True and s.back_active and ex.resets == 1, \
            "takeover must flag back_active and reset the executor"
        assert s.toggle_teleop() is False and not s.back_active and ex.resets == 2, \
            "handing control back must reset the executor again"
        print("   teleop takeover session logic OK")

        print("\nDAgger recorder closed-loop test OK.")
    finally:
        shutil.rmtree(os.path.dirname(root), ignore_errors=True)


if __name__ == "__main__":
    main()
