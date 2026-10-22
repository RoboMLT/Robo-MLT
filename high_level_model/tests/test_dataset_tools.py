"""Tests for the dataset bookkeeping tools on a synthetic LeRobot v3 tree (pandas + pyarrow only).

Run:  python -m high_level_model.tests.test_dataset_tools   (from the repo root)
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(HERE, "..", "..", "configs", "skill_library", "bloodgas.yaml")
SKILLS = ["grasp the routine blood tube.", "remove the sample tube cap.",
          "discard the tube cap into the waste bin."]


def make_tree(root: Path, episodes, subtask_texts=SKILLS, task="blood gas sample analysis",
              recovery_tags=None) -> Path:
    """``episodes``: list of (subtask_index sequence, back_event sequence or None, seed)."""
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    rows = []
    for ep, (subtasks, back, seed) in enumerate(episodes):
        rng = np.random.default_rng(seed)
        for t, s in enumerate(subtasks):
            act = rng.normal(size=14).astype(np.float32)
            rows.append({"episode_index": ep, "frame_index": t, "action": act,
                         "observation.state": act + np.float32(0.01),
                         "subtask_index": int(s), "back_event": int(back[t]) if back else 0})
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"subtask_index": range(len(subtask_texts))},
                 index=pd.Index(subtask_texts)).to_parquet(root / "meta" / "subtasks.parquet")
    pd.DataFrame({"task_index": [0]}, index=pd.Index([task])).to_parquet(root / "meta" / "tasks.parquet")
    info = {"robot_type": "dual_piper", "fps": 25, "total_episodes": len(episodes),
            "total_frames": len(rows),
            "features": {"observation.images.cam_top": {"dtype": "video"},
                         "observation.images.cam_left": {"dtype": "video"},
                         "action": {"dtype": "float32", "shape": [14]}}}
    (root / "meta" / "info.json").write_text(json.dumps(info))
    if recovery_tags:
        (root / "meta" / "recovery_episodes.json").write_text(json.dumps(recovery_tags))
    return root


def test_dataset_stats() -> bool:
    from high_level_model.data.dataset_stats import corpus_stats, library_check, recovery_stats, to_markdown

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nominal = make_tree(tmp / "nominal", [
            ([0] * 5 + [1] * 5 + [2] * 5, None, 1),     # complete workflow (3 skills)
            ([1] * 8, None, 2),                         # atomic-skill episode
            ([1] * 8, None, 2),                         # byte-identical duplicate of the above
        ])
        recovery = make_tree(tmp / "recovery", [
            ([0] * 10, [0] * 7 + [1] * 3, 3),
            ([0] * 6 + [1] * 6, [0] * 8 + [1] * 4, 4),
        ], recovery_tags={"0": {"type": "intervention"}, "1": {"type": "failure"}})

        n = corpus_stats(nominal, num_skills=3)
        hashes = n.pop("_hashes")
        assert (n["episodes"], n["distinct_episodes"], n["duplicate_episodes"]) == (3, 2, 1), n
        assert (n["complete_workflows"], n["atomic_skill_episodes"]) == (1, 2), n
        assert n["workflows_covering_all_skills"] == 1 and n["fps"] == 25
        assert n["tasks"] == ["blood gas sample analysis"]
        assert abs(n["mean_abs_action_minus_state"] - 0.01) < 1e-4

        r = recovery_stats(recovery, hashes)
        assert (r["episodes"], r["episodes_with_positives"]) == (2, 2), r
        assert (r["k_rec_min"], r["k_rec_max"]) == (3, 4) and r["positives_are_final_block"]
        assert r["episode_types"] == {"intervention": 1, "failure": 1}
        assert r["overlap_with_nominal"] == 0

        lib = library_check(nominal, LIB_PATH)
        assert not lib["match"] and set(lib["mismatches"]) == {3, 4, 5, 6, 7}, lib["mismatches"]
        md = to_markdown({"nominal": n, "recovery": r, "library": lib})
        assert "complete workflows | 1" in md and "K_rec range | 3–4" in md
    print("   distinct / duplicate / workflow / atomic / recovery counts OK")
    print("   subtask text checked byte-for-byte against the skill library")
    return True


def test_concat_gate_dataset() -> bool:
    """Nominal + recovery parts share global frame numbers; the concat must keep them apart."""
    try:
        from high_level_model.data.completion_gate_dataset import ConcatGateDataset, Segment, SegmentBatchSampler
    except Exception as exc:  # noqa: BLE001 - needs torch + lerobot importable
        print(f"   [skip] {exc}")
        return None

    class _Part:
        def __init__(self, tag, n, back):
            seg = Segment(0, n, f"skill-{tag}", 0, False)
            self.samples = [(0, f) for f in range(n)]
            self.frame_to_seg = {f: seg for f in range(n)}
            self.back_frames, self.completion_frames = set(back), {n - 1}
            self.segments, self.tag = [seg], tag

        def __getitem__(self, i):
            return self.tag, i

    cat = ConcatGateDataset([_Part("nominal", 10, []), _Part("recovery", 6, [4, 5])])
    assert len(cat) == 16 and cat[3] == ("nominal", 3) and cat[12] == ("recovery", 2)
    assert len(cat.back_frames) == 2 and len(cat.completion_frames) == 2, "labels must not collide"
    sampler = SegmentBatchSampler(cat, segs_per_batch=2, frames_per_seg=4)
    assert len(sampler.seg_ids) == 2, "the two sources' segments must stay distinct"
    batch = next(iter(sampler))
    assert {cat[i][0] for i in batch} == {"nominal", "recovery"}
    print("   nominal + recovery concatenation keeps frames, labels and segments apart")
    return True


def test_release_recovery_meta() -> bool:
    """Episode tags and the nested K_rec record are re-keyed to the aggregated indices."""
    try:
        from high_level_model.data.build_release_dataset import _carry_recovery_meta
    except ImportError as exc:
        print(f"   [skip] build_release_dataset needs lerobot: {exc}")
        return None
    with tempfile.TemporaryDirectory() as tmp:
        srcs = [Path(tmp) / "a", Path(tmp) / "b"]
        for i, src in enumerate(srcs):
            (src / "meta").mkdir(parents=True)
            (src / "meta" / "recovery_episodes.json").write_text(json.dumps(
                {str(e): {"type": f"t{i}{e}"} for e in range(3)}))
            (src / "meta" / "recovery_labels.json").write_text(json.dumps(
                {"k_min": 20, "k_max": 40, "seed": 0,
                 "k_rec": {str(e): 20 + 10 * i + e for e in range(3)}}))
        out = Path(tmp) / "out"
        (out / "meta").mkdir(parents=True)
        # source a keeps episodes 0 and 2, source b keeps 1 -> new indices 0, 1, 2
        _carry_recovery_meta(srcs, {srcs[0]: [0, 2], srcs[1]: [1]}, out)
        labels = json.loads((out / "meta" / "recovery_labels.json").read_text())
        assert labels == {"k_min": 20, "k_max": 40, "seed": 0,
                          "k_rec": {"0": 20, "1": 22, "2": 31}}, labels
        tags = json.loads((out / "meta" / "recovery_episodes.json").read_text())
        assert [tags[k]["type"] for k in ("0", "1", "2")] == ["t00", "t02", "t11"], tags
    print("   recovery tags and nested K_rec record re-keyed")
    return True


def main():
    results = [test_dataset_stats(), test_concat_gate_dataset(), test_release_recovery_meta()]
    failed = sum(r is False for r in results)
    print(f"\n==== summary: {sum(r is True for r in results)} passed, "
          f"{sum(r is None for r in results)} skipped, {failed} failed ====")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
