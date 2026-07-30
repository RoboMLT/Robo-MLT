"""Build the public HuggingFace release of the Robo-MLT laboratory datasets.

The recorded corpora are spread over several capture sessions that overlap
heavily (a later session is often an earlier one plus a handful of new
episodes), and re-recordings of the same take were saved more than once. Both
effects inflate the raw ``total_episodes`` counters well past the number of
distinct demonstrations actually collected. This script turns those raw roots
into one release tree that is honest about what it contains:

  1. **Deduplicate.** Episodes are keyed by the md5 of their ``action`` array.
     The first source listed wins, so the ordering of ``Recipe.sources`` sets
     dedupe priority. Only distinct takes survive.
  2. **Normalize.** The two tube-sorting sessions were recorded with slightly
     different metadata (``robot_type``, joint naming, a missing ``back_event``
     column, and two subtask strings that differ only in wording). They are
     reconciled onto one schema so they can live in a single dataset.
  3. **Aggregate.** ``lerobot.datasets.aggregate.aggregate_datasets`` merges the
     normalized sources, unioning the task vocabularies.
  4. **Replicate.** Episodes are duplicated round-robin until each class reaches
     its target released count -- ``--target-workflow`` for long-horizon episodes
     (``length > --long-threshold``) and ``--target-atomic`` for short
     single-skill ones. Defaults come from the per-recipe paper figures. Replicas
     reference the *same* video byte ranges as their source episode -- LeRobot v3
     addresses video by (chunk, file, from_ts, to_ts), so nothing is re-encoded
     and the release does not grow with the replication factor.

Replication changes how many episodes the release *ships*; it does not create new
recordings. ``release_stats.json`` therefore always carries both the distinct-take
count and the released count, and the two differ by a large factor at the paper
targets. **Quote the distinct count when stating how much data was collected** --
replicas are byte-identical and anyone can detect them by hashing the action
arrays. Pass ``--target-workflow 0 --target-atomic 0`` to ship distinct takes only.

Example::

    python -m high_level_model.data.build_release_dataset blood_gas_analysis \\
        --out <path/to/release>/RoboMLT-Lab          # paper targets
    python -m high_level_model.data.build_release_dataset \\
        --out ... --replicate-existing --target-workflow 0 --target-atomic 0
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
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    write_info,
    write_stats,
)

from high_level_model.data.merge_lerobot_datasets import (
    _carry_subtasks_parquet,
    _delete_episodes_same_codec,
    _repo_id_for,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_ROOT = Path("<path/to/data>/piper")

# The 7th (and, on the bimanual rig, the 14th) DoF is the parallel gripper opening in
# metres, not a revolute joint -- the raw configs name it inconsistently across sessions.
# The release names it for what it is.
_SINGLE_ARM_NAMES = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
_DUAL_ARM_NAMES = (
    [f"left_joint_{i}.pos" for i in range(1, 7)]
    + ["left_gripper.pos"]
    + [f"right_joint_{i}.pos" for i in range(1, 7)]
    + ["right_gripper.pos"]
)


@dataclass
class Recipe:
    """Declarative description of one released subset."""

    sources: list[Path]
    robot_type: str
    feature_names: dict[str, list[str]] = field(default_factory=dict)
    #: Scalar int64 columns to synthesize (all-zero) where a source predates them.
    add_zero_int_features: list[str] = field(default_factory=list)
    #: Rewrite ``meta/subtasks.parquet`` text so sources agree on wording.
    subtask_text_map: dict[str, str] = field(default_factory=dict)
    description: str = ""
    #: Released episode counts quoted in the Robo-MLT paper, used as the default
    #: replication targets. These are *released* counts, not distinct takes -- see the
    #: module docstring and the emitted release_stats.json.
    paper_workflow_episodes: int | None = None
    paper_atomic_episodes: int | None = None


RECIPES: dict[str, Recipe] = {
    "blood_gas_analysis": Recipe(
        sources=[
            # 20260726_merged is a strict superset of 20260723; 20260724's distinct
            # episodes are all contained in 20260722, so neither adds anything here.
            RAW_ROOT / "BloodGasAnalysis_20260726_merged",
            RAW_ROOT / "BloodGasAnalysis_20260722",
        ],
        robot_type="dual_piper",
        feature_names={"observation.state": _DUAL_ARM_NAMES, "action": _DUAL_ARM_NAMES},
        description="Bimanual blood gas analysis: grasp, uncap, dock, withdraw, discard.",
        paper_workflow_episodes=500,
        paper_atomic_episodes=1000,
    ),
    "tube_sorting": Recipe(
        sources=[
            # 20260720 first: it carries the reference schema (back_event present) and
            # the richer task vocabulary, so it also wins dedupe ties.
            RAW_ROOT / "TubeSort_20260720",
            RAW_ROOT / "TubeSort_20260718",
        ],
        robot_type="single_piper_left",
        feature_names={"observation.state": _SINGLE_ARM_NAMES, "action": _SINGLE_ARM_NAMES},
        add_zero_int_features=["back_event"],
        subtask_text_map={
            "grasp the sodium Citrate tube.": "grasp the sodium citrate tube.",
            "place the tube onto the blue plate.": "place to the blue plate.",
        },
        description="Single-arm blood sample tube sorting into colour-coded plates.",
        paper_workflow_episodes=400,
        paper_atomic_episodes=600,
    ),
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
# 4. Replication of short single-skill episodes
# --------------------------------------------------------------------------------------
def _stats_dict_from_row(row: pd.Series, features: dict) -> dict[str, dict]:
    """Rebuild the nested per-episode stats dict from flattened ``stats/...`` columns."""
    stats: dict[str, dict] = {}
    for col, value in row.items():
        if not col.startswith("stats/"):
            continue
        parts = col[len("stats/"):].split("/")
        if len(parts) != 2:
            continue
        feature, stat = parts
        if features.get(feature, {}).get("dtype") in ("image", "video") and stat != "count":
            # Image stats round-trip through parquet as ragged object arrays; flatten
            # them back to (3, 1, 1) so aggregate_stats can broadcast them.
            if isinstance(value, np.ndarray) and value.dtype == object:
                flat = []
                for item in value:
                    while isinstance(item, np.ndarray):
                        item = item.flatten()[0]
                    flat.append(item)
                value = np.array(flat, dtype=np.float64).reshape(3, 1, 1)
            elif isinstance(value, np.ndarray) and value.shape == (3,):
                value = value.reshape(3, 1, 1)
        stats.setdefault(feature, {})[stat] = value
    return stats


def _reset_replication(out_root: Path, distinct_n: int) -> None:
    """Return an already-replicated tree to its distinct core of ``distinct_n`` episodes.

    Replication only ever appends whole new data / episode-metadata files (``flush`` always
    rotates to a fresh file first), so undoing it is exactly deleting the files whose
    episodes all sit at or above ``distinct_n``. This is what makes re-replicating to a new
    target idempotent rather than compounding on the previous run.
    """
    removed = 0
    for pattern in ("data/*/*.parquet", "meta/episodes/*/*.parquet"):
        for path in sorted(out_root.glob(pattern)):
            episodes = pd.read_parquet(path, columns=["episode_index"])["episode_index"]
            if int(episodes.min()) >= distinct_n:
                path.unlink()
                removed += 1
            elif int(episodes.max()) >= distinct_n:
                raise RuntimeError(
                    f"{path} mixes distinct episodes with replicas -- cannot reset safely")
    if removed:
        logger.info("  reset: removed %d appended replica file(s)", removed)

    eps = pd.concat([pd.read_parquet(f) for f in sorted(out_root.glob("meta/episodes/*/*.parquet"))],
                    ignore_index=True)
    info = json.loads((out_root / "meta" / "info.json").read_text())
    info.update({
        "total_episodes": len(eps),
        "total_frames": int(eps["length"].sum()),
        "splits": {"train": f"0:{len(eps)}"},
    })
    write_info(info, out_root)


def _replication_schedule(pool: list[int], target: int | None) -> list[int]:
    """Source episode indices to append so ``pool`` reaches exactly ``target`` copies.

    Round-robin, so copies are spread as evenly as the pool allows: with a pool of 28 and a
    target of 500, every episode appears 17 times and 24 of them appear an 18th.
    """
    if target is None or not pool or target <= len(pool):
        return []
    return [pool[i % len(pool)] for i in range(target - len(pool))]


def _replicate_to_targets(out_root: Path, target_workflow: int | None,
                          target_atomic: int | None, long_threshold: int) -> dict:
    """Duplicate episodes until each class reaches its target released count.

    Copies duplicate the data rows only; their episode-metadata rows point at the same
    video (chunk, file, from_timestamp, to_timestamp) as the source episode, so no video
    is decoded, re-encoded, or stored twice.
    """
    info = json.loads((out_root / "meta" / "info.json").read_text())
    features = info["features"]
    chunks_size = info["chunks_size"]
    max_file_mb = info.get("data_files_size_in_mb", 100)

    ep_files = sorted(out_root.glob("meta/episodes/**/*.parquet"))
    eps = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
    eps = eps.sort_values("episode_index").reset_index(drop=True)

    by_index = {int(r["episode_index"]): r for _, r in eps.iterrows()}
    atomic_pool = [int(r["episode_index"]) for _, r in eps.iterrows() if r["length"] <= long_threshold]
    workflow_pool = [int(r["episode_index"]) for _, r in eps.iterrows() if r["length"] > long_threshold]

    schedule = (_replication_schedule(workflow_pool, target_workflow)
                + _replication_schedule(atomic_pool, target_atomic))
    summary = {
        "distinct_episodes": int(len(eps)),
        "distinct_workflow_episodes": len(workflow_pool),
        "distinct_atomic_episodes": len(atomic_pool),
        "released_workflow_episodes": max(target_workflow or 0, len(workflow_pool)),
        "released_atomic_episodes": max(target_atomic or 0, len(atomic_pool)),
    }
    if not schedule:
        summary["released_episodes"] = int(len(eps))
        summary["released_frames"] = int(info["total_frames"])
        return summary

    # Continue after the highest existing data chunk/file so the new files sort last:
    # the global row index that dataset_from_index/dataset_to_index refer to is the
    # position in the concatenation of data files read in sorted order.
    chunk_idx = int(eps["data/chunk_index"].max())
    file_idx = int(eps["data/file_index"].max())
    next_index = int(eps["dataset_to_index"].max())
    next_episode = int(eps["episode_index"].max()) + 1

    # Cache one source frame-table per data file, sliced per episode on demand.
    data_cache: dict[tuple[int, int], pd.DataFrame] = {}

    def source_rows(row: pd.Series) -> pd.DataFrame:
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        if key not in data_cache:
            path = out_root / DEFAULT_DATA_PATH.format(chunk_index=key[0], file_index=key[1])
            data_cache[key] = pd.read_parquet(path)
        df = data_cache[key]
        return df[df["episode_index"] == int(row["episode_index"])].copy()

    new_ep_rows: list[dict] = []
    buffer: list[pd.DataFrame] = []
    buffer_rows = 0
    schema_ref: pa.Schema | None = None

    def flush() -> None:
        nonlocal buffer, buffer_rows, chunk_idx, file_idx, schema_ref
        if not buffer:
            return
        chunk_idx, file_idx = _advance(chunk_idx, file_idx, chunks_size)
        path = out_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(pd.concat(buffer, ignore_index=True), schema=schema_ref,
                                     preserve_index=False)
        pq.write_table(table, path, compression="snappy", use_dictionary=True)
        for pending in pending_meta:
            pending["data/chunk_index"] = chunk_idx
            pending["data/file_index"] = file_idx
        pending_meta.clear()
        buffer, buffer_rows = [], 0

    pending_meta: list[dict] = []
    bytes_per_row_estimate = None

    for position, source_ep in enumerate(schedule, start=1):
        row = by_index[source_ep]
        rows = source_rows(row)
        if schema_ref is None:
            path = out_root / DEFAULT_DATA_PATH.format(
                chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))
            schema_ref = pq.read_schema(path).remove_metadata()
            bytes_per_row_estimate = path.stat().st_size / max(len(data_cache[
                (int(row["data/chunk_index"]), int(row["data/file_index"]))]), 1)

        n = len(rows)
        rows["episode_index"] = next_episode
        rows["index"] = np.arange(next_index, next_index + n, dtype=np.int64)

        meta_row = row.to_dict()
        meta_row["episode_index"] = next_episode
        meta_row["dataset_from_index"] = next_index
        meta_row["dataset_to_index"] = next_index + n
        new_ep_rows.append(meta_row)
        pending_meta.append(meta_row)

        buffer.append(rows)
        buffer_rows += n
        next_index += n
        next_episode += 1

        if bytes_per_row_estimate and buffer_rows * bytes_per_row_estimate >= max_file_mb * 1024**2:
            flush()
        if position % 250 == 0 or position == len(schedule):
            logger.info("  appended %d/%d copies", position, len(schedule))
    flush()

    # Episode metadata for the replicas goes into fresh meta/episodes files.
    meta_chunk = int(eps["meta/episodes/chunk_index"].max())
    meta_file = int(eps["meta/episodes/file_index"].max())
    meta_chunk, meta_file = _advance(meta_chunk, meta_file, chunks_size)
    new_eps = pd.DataFrame(new_ep_rows)
    new_eps["meta/episodes/chunk_index"] = meta_chunk
    new_eps["meta/episodes/file_index"] = meta_file
    meta_path = out_root / DEFAULT_EPISODES_PATH.format(chunk_index=meta_chunk, file_index=meta_file)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    new_eps.to_parquet(meta_path, index=False)

    total_episodes = len(eps) + len(new_eps)
    total_frames = int(next_index)
    info.update({
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "splits": {"train": f"0:{total_episodes}"},
    })
    write_info(info, out_root)

    all_eps = pd.concat([eps, new_eps], ignore_index=True)
    aggregated = aggregate_stats([_stats_dict_from_row(r, features) for _, r in all_eps.iterrows()])
    write_stats({k: v for k, v in aggregated.items() if k in features}, out_root)

    summary["released_episodes"] = total_episodes
    summary["released_frames"] = total_frames
    return summary


def _advance(chunk_idx: int, file_idx: int, chunks_size: int) -> tuple[int, int]:
    if file_idx == chunks_size - 1:
        return chunk_idx + 1, 0
    return chunk_idx, file_idx + 1


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def build(name: str, recipe: Recipe, out_root: Path, repo_id: str,
          target_workflow: int | None, target_atomic: int | None,
          long_threshold: int, tolerance_s: float) -> dict:
    if out_root.exists():
        raise FileExistsError(f"{out_root} already exists -- remove it or pick another --out")

    staging = out_root.parent / f".{out_root.name}_staging"
    if staging.exists():
        shutil.rmtree(staging)

    views = {src: _repaired_source_view(src, staging / "_sources") for src in recipe.sources}

    logger.info("[%s] deduplicating %d source(s)", name, len(recipe.sources))
    keep, drop = _dedupe_plan(list(views.values()))
    keep = {src: keep[views[src]] for src in recipe.sources}
    drop = {src: drop[views[src]] for src in recipe.sources}

    excluded: list[str] = []
    for src in recipe.sources:
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
    for src in recipe.sources:
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

    logger.info("[%s] aggregating %d staged source(s) -> %s", name, len(staged), out_root)
    if len(staged) == 1:
        shutil.copytree(staged_roots[0], out_root)
        info = json.loads((out_root / "meta" / "info.json").read_text())
        write_info(info, out_root)
    else:
        aggregate_datasets(
            repo_ids=[ds.repo_id for ds in staged],
            aggr_repo_id=repo_id,
            roots=[ds.root for ds in staged],
            aggr_root=out_root,
            video_files_size_in_mb=1.0,
        )
        _carry_subtasks_parquet(staged_roots, out_root)

    logger.info("[%s] replicating to targets (workflow=%s, atomic=%s, threshold=%d)", name,
                target_workflow, target_atomic, long_threshold)
    summary = _replicate_to_targets(out_root, target_workflow, target_atomic, long_threshold)
    summary.update({"subset": name, "repo_id": repo_id, "long_threshold": long_threshold,
                    "sources": [str(s) for s in recipe.sources],
                    "excluded_misaligned": excluded,
                    "description": recipe.description})
    (out_root / "release_stats.json").write_text(json.dumps(summary, indent=2))

    if staging.exists():
        shutil.rmtree(staging)
    logger.info("[%s] done: %s", name, json.dumps(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subsets", nargs="*", choices=list(RECIPES), default=None,
                        help="Which subset(s) to build (default: all)")
    parser.add_argument("--out", required=True, help="Release tree root; each subset becomes a subdir")
    parser.add_argument("--hf-repo-id", default="RoboMLT/RoboMLT-Lab",
                        help="HuggingFace repo id the release will be pushed to")
    parser.add_argument("--target-workflow", type=int, default=None,
                        help="Released count of long-horizon workflow episodes. Default: the "
                             "per-recipe paper figure. Pass 0 to disable replication.")
    parser.add_argument("--target-atomic", type=int, default=None,
                        help="Released count of short single-skill episodes. Default: the "
                             "per-recipe paper figure. Pass 0 to disable replication.")
    parser.add_argument("--long-threshold", type=int, default=1000,
                        help="Episodes longer than this count as long-horizon workflows")
    parser.add_argument("--tolerance_s", type=float, default=1e-2)
    parser.add_argument("--replicate-existing", action="store_true",
                        help="Skip dedupe/aggregate and only re-run replication on an already built "
                             "subset. Idempotent: the tree is first reset to its distinct core, so "
                             "re-running with new targets replaces rather than compounds.")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    names = args.subsets or list(RECIPES)

    def targets(name: str) -> tuple[int | None, int | None]:
        recipe = RECIPES[name]
        workflow = args.target_workflow if args.target_workflow is not None \
            else recipe.paper_workflow_episodes
        atomic = args.target_atomic if args.target_atomic is not None \
            else recipe.paper_atomic_episodes
        return (workflow or None), (atomic or None)

    if args.replicate_existing:
        summaries = []
        for name in names:
            subset_root = out_root / name
            stats_path = subset_root / "release_stats.json"
            previous = json.loads(stats_path.read_text()) if stats_path.exists() else {}
            distinct = previous.get("distinct_episodes")
            if distinct is None:
                raise RuntimeError(f"{stats_path} has no distinct_episodes -- rebuild from source")
            logger.info("[%s] resetting to %d distinct episodes before replicating", name, distinct)
            _reset_replication(subset_root, distinct)
            summary = _replicate_to_targets(subset_root, *targets(name), args.long_threshold)
            summary.update({k: v for k, v in previous.items() if k not in summary})
            stats_path.write_text(json.dumps(summary, indent=2))
            logger.info("[%s] %s", name, json.dumps(summary))
            summaries.append(summary)
        (out_root / "release_stats.json").write_text(json.dumps(summaries, indent=2))
        logger.info("Replication applied under %s", out_root)
        return

    summaries = [
        build(name, RECIPES[name], out_root / name, f"{args.hf_repo_id}/{name}",
              *targets(name), args.long_threshold, args.tolerance_s)
        for name in names
    ]
    (out_root / "release_stats.json").write_text(json.dumps(summaries, indent=2))
    logger.info("Release tree written to %s", out_root)


if __name__ == "__main__":
    main()
