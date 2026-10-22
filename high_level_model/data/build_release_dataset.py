"""Build the public release of the Robo-MLT laboratory datasets (one subset per corpus).

Capture sessions overlap (a later session is often an earlier one plus new episodes) and a take
can be saved more than once, so raw ``total_episodes`` counters overstate how much was recorded.
This script turns the raw session roots into one release tree per subset that contains every
distinct take exactly once:

  1. **Deduplicate.** Episodes are keyed by the md5 of their ``action`` array; the first source
     listed wins. Episodes whose video segment and frame count disagree are excluded.
  2. **Normalize.** Sources are reconciled onto one schema (``robot_type``, joint names, a
     missing ``back_event`` column, subtask wording).
  3. **Aggregate.** ``lerobot.datasets.aggregate.aggregate_datasets`` merges the sources.

Each subset gets a ``release_stats.json``; ``dataset_stats.py`` reports the counts quoted in the
paper (complete workflows, atomic-skill episodes, recovery episodes) from the finished tree.

Subsets: ``blood_gas_analysis`` and ``tube_sorting`` (nominal corpora, used by System 1 and
System 2) and ``<task>_recovery`` (intervention / failure episodes, used by System 2 only). For a
recovery subset the episode tags in ``meta/recovery_episodes.json`` and the K_rec record in
``meta/recovery_labels.json`` are re-indexed along with the episodes.

Example::

    python -m high_level_model.data.build_release_dataset blood_gas_analysis \\
        --sources /data/piper/BloodGasAnalysis_<session1> /data/piper/BloodGasAnalysis_<session2> \\
        --out <path/to/release>/RoboMLT-Lab
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_info

from high_level_model.data.merge_lerobot_datasets import (
    _carry_subtasks_parquet,
    _delete_episodes_same_codec,
    _repo_id_for,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The 7th and 14th DoF are the parallel-gripper openings in metres, not revolute joints; the
# raw sessions name them inconsistently, so the release names them for what they are.
_DUAL_ARM_NAMES = (
    [f"left_joint_{i}.pos" for i in range(1, 7)]
    + ["left_gripper.pos"]
    + [f"right_joint_{i}.pos" for i in range(1, 7)]
    + ["right_gripper.pos"]
)
RECOVERY_META_FILES = ("recovery_episodes.json", "recovery_labels.json")


@dataclass
class Recipe:
    """Target schema of one released subset (the sources come from the command line)."""

    robot_type: str
    feature_names: dict[str, list[str]] = field(default_factory=dict)
    #: Scalar int64 columns to synthesize (all-zero) where a source predates them.
    add_zero_int_features: list[str] = field(default_factory=list)
    #: Rewrite ``meta/subtasks.parquet`` text so sources agree on wording.
    subtask_text_map: dict[str, str] = field(default_factory=dict)
    description: str = ""


_BLOOD_GAS = dict(
    robot_type="dual_piper",
    feature_names={"observation.state": _DUAL_ARM_NAMES, "action": _DUAL_ARM_NAMES},
    add_zero_int_features=["back_event"],
)
_TUBE_SORTING = dict(
    robot_type="dual_piper",
    feature_names={"observation.state": _DUAL_ARM_NAMES, "action": _DUAL_ARM_NAMES},
    add_zero_int_features=["back_event"],
    subtask_text_map={
        "grasp the sodium Citrate tube.": "grasp the sodium citrate tube.",
        "place the tube onto the blue plate.": "place to the blue plate.",
    },
)
RECIPES: dict[str, Recipe] = {
    "blood_gas_analysis": Recipe(**_BLOOD_GAS, description=(
        "Blood gas analysis: complete workflows and atomic-skill episodes.")),
    "blood_gas_analysis_recovery": Recipe(**_BLOOD_GAS, description=(
        "Blood gas analysis: operator-intervention and teleoperation-failure episodes.")),
    "tube_sorting": Recipe(**_TUBE_SORTING, description=(
        "Blood sample tube sorting: complete workflows and atomic-skill episodes.")),
    "tube_sorting_recovery": Recipe(**_TUBE_SORTING, description=(
        "Blood sample tube sorting: operator-intervention and teleoperation-failure episodes.")),
}


# --------------------------------------------------------------------------------------
# 1. Deduplication
# --------------------------------------------------------------------------------------
def _episode_action_hashes(root: Path) -> dict[int, str]:
    """md5 of each episode's float32 action matrix, keyed by episode_index.

    The action stream is the identity of a take: two episodes with byte-identical
    actions are the same recording saved twice, whatever their episode_index or the
    session they came from.
    """
    frames = [
        pd.read_parquet(f, columns=["episode_index", "action"])
        for f in sorted(root.glob("data/**/*.parquet"))
    ]
    df = pd.concat(frames, ignore_index=True)
    out: dict[int, str] = {}
    for ep, group in df.groupby("episode_index"):
        arr = np.stack(group["action"].to_numpy()).astype(np.float32)
        out[int(ep)] = hashlib.md5(arr.tobytes()).hexdigest()
    return out


def _repaired_source_view(src: Path, workdir: Path) -> Path:
    """Non-destructive view of ``src`` whose episode-metadata indices match reality.

    One session's ``meta/episodes`` rows carry ``meta/episodes/{chunk,file}_index`` values
    pointing at parquet files that a previous merge consolidated away, so LeRobot's
    per-episode stats loader goes looking for a file that no longer exists. The repair
    belongs in the release copy rather than in the recorded data, so ``data/`` and
    ``videos/`` (correct, and large) are symlinked while ``meta/`` is copied and fixed.
    """
    view = workdir / src.name
    if view.exists():
        return view
    view.mkdir(parents=True)
    for sub in ("data", "videos"):
        if (src / sub).exists():
            (view / sub).symlink_to((src / sub).resolve(), target_is_directory=True)
    shutil.copytree(src / "meta", view / "meta")

    for path in sorted((view / "meta" / "episodes").glob("**/*.parquet")):
        match = re.search(r"chunk-(\d+)/file-(\d+)", path.as_posix())
        if match is None:
            continue
        chunk, file_idx = int(match.group(1)), int(match.group(2))
        df = pd.read_parquet(path)
        declared = set(zip(df["meta/episodes/chunk_index"], df["meta/episodes/file_index"]))
        if declared == {(chunk, file_idx)}:
            continue
        logger.warning("  %s: episode metadata claims files %s but lives in (%d, %d) -- repairing",
                       src.name, sorted(declared), chunk, file_idx)
        df["meta/episodes/chunk_index"] = chunk
        df["meta/episodes/file_index"] = file_idx
        df.to_parquet(path, index=False)
    return view


def _misaligned_episodes(root: Path) -> dict[int, str]:
    """Episodes whose video segment is shorter/longer than their recorded frame count.

    A handful of takes lost a camera frame at record time, so for those episodes
    ``round(to_timestamp * fps) - round(from_timestamp * fps)`` disagrees with ``length``
    by one. LeRobot tolerates this on the plain-copy path but asserts on the re-encode
    path, and more importantly it means the released state/video streams would be
    misaligned by a frame at the end of the episode. They are excluded from the release.
    """
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info["fps"]
    eps = pd.concat([pd.read_parquet(f) for f in sorted(root.glob("meta/episodes/**/*.parquet"))],
                    ignore_index=True)
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    bad: dict[int, str] = {}
    for key in video_keys:
        span = (np.round(eps[f"videos/{key}/to_timestamp"].to_numpy() * fps)
                - np.round(eps[f"videos/{key}/from_timestamp"].to_numpy() * fps)).astype(int)
        for i in np.flatnonzero(span - eps["length"].to_numpy()):
            ep = int(eps["episode_index"].iloc[i])
            bad[ep] = f"{key} spans {span[i]} frames for a {int(eps['length'].iloc[i])}-frame episode"
    return bad


def _dedupe_plan(sources: list[Path]) -> tuple[dict[Path, list[int]], dict[Path, list[int]]]:
    """Split every source's episodes into (keep, drop) by first-seen action hash."""
    seen: set[str] = set()
    keep: dict[Path, list[int]] = {}
    drop: dict[Path, list[int]] = {}
    for root in sources:
        hashes = _episode_action_hashes(root)
        keep[root], drop[root] = [], []
        for ep in sorted(hashes):
            if hashes[ep] in seen:
                drop[root].append(ep)
            else:
                seen.add(hashes[ep])
                keep[root].append(ep)
        logger.info(
            "%s: %d episodes -> keep %d distinct, drop %d duplicate",
            root.name, len(hashes), len(keep[root]), len(drop[root]),
        )
    return keep, drop


# --------------------------------------------------------------------------------------
# 2. Schema normalization
# --------------------------------------------------------------------------------------
_SCALAR_STAT_NAMES = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


def _normalize_staged(root: Path, recipe: Recipe) -> None:
    """Reconcile one staged source onto the recipe's schema, in place."""
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())

    if info["robot_type"] != recipe.robot_type:
        logger.info("  robot_type %r -> %r", info["robot_type"], recipe.robot_type)
        info["robot_type"] = recipe.robot_type

    for key, names in recipe.feature_names.items():
        if key in info["features"] and info["features"][key].get("names") != names:
            if len(names) != info["features"][key]["shape"][0]:
                raise ValueError(f"{root.name}: {key} has shape {info['features'][key]['shape']} "
                                 f"but recipe supplies {len(names)} names")
            logger.info("  renaming %s dims", key)
            info["features"][key]["names"] = names

    missing = [f for f in recipe.add_zero_int_features if f not in info["features"]]
    for feature in missing:
        logger.info("  synthesizing all-zero %r column (absent in this session)", feature)
        # Data files: append the column. Insert after subtask_index when present so the
        # column order matches the sessions that already carry it.
        for path in sorted(root.glob("data/**/*.parquet")):
            table = pq.read_table(path)
            zeros = pa.array(np.zeros(table.num_rows, dtype=np.int64), type=pa.int64())
            pos = table.schema.get_field_index("subtask_index")
            pos = table.num_columns if pos < 0 else pos + 1
            pq.write_table(table.add_column(pos, feature, zeros), path,
                           compression="snappy", use_dictionary=True)
        # Episode metadata: add the matching stats/ columns. Everything is zero except
        # count, which is the episode's frame count.
        for path in sorted(root.glob("meta/episodes/**/*.parquet")):
            df = pd.read_parquet(path)
            for stat in _SCALAR_STAT_NAMES:
                df[f"stats/{feature}/{stat}"] = [np.zeros(1, dtype=np.float64) for _ in range(len(df))]
            df[f"stats/{feature}/count"] = [
                np.array([n], dtype=np.int64) for n in df["length"].to_numpy()
            ]
            df.to_parquet(path, index=False)
        info["features"][feature] = {"dtype": "int64", "shape": [1], "names": None}

    if missing:
        stats_path = root / "meta" / "stats.json"
        stats = json.loads(stats_path.read_text())
        for feature in missing:
            stats[feature] = {k: [0.0] for k in _SCALAR_STAT_NAMES}
            stats[feature]["count"] = [info["total_frames"]]
        stats_path.write_text(json.dumps(stats, indent=4))

    write_info(info, root)

    subtasks_path = root / "meta" / "subtasks.parquet"
    if recipe.subtask_text_map and subtasks_path.exists():
        df = pd.read_parquet(subtasks_path)
        renamed = df.rename(index=recipe.subtask_text_map)
        if not renamed.index.equals(df.index):
            logger.info("  normalizing subtask wording")
            renamed.to_parquet(subtasks_path)


# --------------------------------------------------------------------------------------
# 4. Recovery metadata (episode tags and K_rec record, keyed by episode index)
# --------------------------------------------------------------------------------------
def _carry_recovery_meta(sources: list[Path], keep: dict[Path, list[int]], out_root: Path) -> None:
    """Re-key per-episode recovery metadata to the aggregated episode indices.

    ``delete_episodes`` re-indexes each staged source to 0..n-1 in the order of its kept
    episodes, and aggregation concatenates the sources in order, so the new index of a kept
    episode is the source's offset plus its rank among the kept episodes.

    ``recovery_episodes.json`` is the per-episode map itself; ``recovery_labels.json`` nests it
    under ``k_rec`` next to the sampling parameters (``k_min``, ``k_max``, ``seed``).
    """
    for fname in RECOVERY_META_FILES:
        nested = fname == "recovery_labels.json"
        merged: dict[str, object] = {}
        header: dict[str, object] = {}
        offset = 0
        for src in sources:
            kept = sorted(keep[src])
            path = src / "meta" / fname
            if path.exists():
                records = json.loads(path.read_text(encoding="utf-8"))
                if nested:
                    params = {k: v for k, v in records.items() if k != "k_rec"}
                    if header and params != header:
                        logger.warning("  %s: sources labelled with different parameters (%s vs %s)",
                                       fname, header, params)
                    header = header or params
                    records = records.get("k_rec") or {}
                for rank, ep in enumerate(kept):
                    if str(ep) in records:
                        merged[str(offset + rank)] = records[str(ep)]
            offset += len(kept)
        if merged:
            out = {**header, "k_rec": merged} if nested else merged
            (out_root / "meta" / fname).write_text(json.dumps(out, indent=2), encoding="utf-8")
            logger.info("  carried %d %s record(s)", len(merged), fname)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def build(name: str, recipe: Recipe, sources: list[Path], out_root: Path, repo_id: str,
          tolerance_s: float) -> dict:
    if out_root.exists():
        raise FileExistsError(f"{out_root} already exists -- remove it or pick another --out")

    staging = out_root.parent / f".{out_root.name}_staging"
    if staging.exists():
        shutil.rmtree(staging)

    views = {src: _repaired_source_view(src, staging / "_sources") for src in sources}

    logger.info("[%s] deduplicating %d source(s)", name, len(sources))
    keep, drop = _dedupe_plan(list(views.values()))
    keep = {src: keep[views[src]] for src in sources}
    drop = {src: drop[views[src]] for src in sources}
    n_raw = sum(len(keep[s]) + len(drop[s]) for s in sources)
    n_duplicates = sum(len(drop[s]) for s in sources)

    excluded: list[str] = []
    for src in sources:
        bad = _misaligned_episodes(views[src])
        hits = [ep for ep in keep[src] if ep in bad]
        for ep in hits:
            logger.warning("[%s] excluding %s ep %d: %s", name, src.name, ep, bad[ep])
            excluded.append(f"{src.name}:{ep} ({bad[ep]})")
        if hits:
            keep[src] = [ep for ep in keep[src] if ep not in bad]
            drop[src] = sorted(drop[src] + hits)

    staged: list[LeRobotDataset] = []
    staged_roots: list[Path] = []
    contributing: list[Path] = []
    for src in sources:
        if not keep[src]:
            logger.info("[%s] %s contributes nothing distinct -- skipped", name, src.name)
            continue
        ds = LeRobotDataset(repo_id=_repo_id_for(src), root=views[src], tolerance_s=tolerance_s)
        dst = staging / src.name
        logger.info("[%s] staging %s (%d distinct episodes)", name, src.name, len(keep[src]))
        staged_ds = _delete_episodes_same_codec(
            ds, episode_indices=drop[src], output_dir=dst, repo_id=f"{_repo_id_for(src)}_distinct")
        # delete_episodes drops the project-specific subtask table; the vocabulary is
        # unchanged by dropping episodes, so carry it over before normalizing wording.
        src_subtasks = src / "meta" / "subtasks.parquet"
        if src_subtasks.exists():
            shutil.copy(src_subtasks, dst / "meta" / "subtasks.parquet")
        _normalize_staged(dst, recipe)
        staged.append(LeRobotDataset(repo_id=staged_ds.repo_id, root=dst, tolerance_s=tolerance_s))
        staged_roots.append(dst)
        contributing.append(src)

    logger.info("[%s] aggregating %d staged source(s) -> %s", name, len(staged), out_root)
    if len(staged) == 1:
        shutil.copytree(staged_roots[0], out_root)
    else:
        aggregate_datasets(
            repo_ids=[ds.repo_id for ds in staged],
            aggr_repo_id=repo_id,
            roots=[ds.root for ds in staged],
            aggr_root=out_root,
            video_files_size_in_mb=1.0,
        )
        _carry_subtasks_parquet(staged_roots, out_root)
    _carry_recovery_meta(contributing, keep, out_root)

    info = json.loads((out_root / "meta" / "info.json").read_text())
    summary = {
        "subset": name, "repo_id": repo_id, "description": recipe.description,
        "sources": [str(s) for s in sources],
        "raw_episodes": n_raw,
        "duplicate_episodes_dropped": n_duplicates,
        "excluded_misaligned": excluded,
        "released_episodes": int(info["total_episodes"]),
        "released_frames": int(info["total_frames"]),
    }
    (out_root / "release_stats.json").write_text(json.dumps(summary, indent=2))

    if staging.exists():
        shutil.rmtree(staging)
    logger.info("[%s] done: %s", name, json.dumps(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subset", choices=list(RECIPES), help="Which subset to build")
    parser.add_argument("--sources", nargs="+", required=True, type=Path,
                        help="Raw LeRobot v3 session roots, in dedupe priority order")
    parser.add_argument("--out", required=True, help="Release tree root; the subset becomes a subdir")
    parser.add_argument("--hf-repo-id", default="RoboMLT/RoboMLT-Lab",
                        help="Repo id prefix the release will be published under")
    parser.add_argument("--tolerance_s", type=float, default=1e-2)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    build(args.subset, RECIPES[args.subset], list(args.sources), out_root / args.subset,
          f"{args.hf_repo_id}/{args.subset}", args.tolerance_s)


if __name__ == "__main__":
    main()
