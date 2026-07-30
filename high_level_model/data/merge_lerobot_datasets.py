"""Merge two or more LeRobot v3 datasets into one, with optional per-source
minimum-episode-length filtering.

Thin wrapper around lerobot's own dataset tools (kept in sync with the LeRobot
codebase so schema handling — data/video re-chunking, task reindexing, stats
aggregation — is never reimplemented here):
  - Episode-length filtering reimplements ``lerobot.datasets.dataset_tools
    .delete_episodes`` (data/video copy + reindex helpers, unchanged) but picks
    the video encoder from the source's own declared codec instead of that
    function's hardcoded ``vcodec="libsvtav1"`` default -- AV1 output was
    silently undecodable via this env's torchcodec build ("Could not push
    packet to decoder: Function not implemented"), even though the exact same
    file decoded fine with PyAV. See ``_delete_episodes_same_codec``.
  - ``lerobot.datasets.aggregate.aggregate_datasets`` concatenates the
    filtered/full datasets into the final output, unioning the task
    vocabularies and recomputing dataset-level stats. Called directly (not via
    the ``merge_datasets`` convenience wrapper) so ``video_files_size_in_mb``
    can be forced small -- see the flag's help text for why.
  - ``meta/subtasks.parquet`` (this project's subtask_index -> text table, not
    part of vanilla LeRobot) isn't touched by either of the above, so it's
    carried over separately after aggregation; see ``_carry_subtasks_parquet``.

Each ``--sources`` entry is ``path`` (keep every episode) or ``path:min_len``
(keep only episodes with more than ``min_len`` frames — matches the
``min_episode_len`` semantics already used by
``high_level_model.training.train_competion_gate.DatasetConfig``).

Example::

    python -m high_level_model.data.merge_lerobot_datasets \\
        --sources <path/to/data>/BloodGasAnalysis_20260722:1200 \\
                  <path/to/data>/BloodGasAnalysis_20260723 \\
        --out <path/to/data>/BloodGasAnalysis_20260724 \\
        --repo_id piper/BloodGasAnalysis_20260724
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_tools import (
    _copy_and_reindex_data,
    _copy_and_reindex_episodes_metadata,
    _copy_and_reindex_videos,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_source(spec: str) -> tuple[Path, int]:
    """Parse ``path`` or ``path:min_episode_len`` (plain paths never contain ':')."""
    if ":" in spec:
        path_str, min_len_str = spec.rsplit(":", 1)
        return Path(path_str), int(min_len_str)
    return Path(spec), 0


def _repo_id_for(root: Path) -> str:
    return "/".join(root.parts[-2:])


# lerobot's declared "video.codec" name -> a portable *software* PyAV/ffmpeg encoder
# name (never nvenc: that requires a GPU + nvenc build present, which we can't assume
# for a batch data-processing script). Falls back to av1 (this codebase's other
# common source codec) for anything unrecognized.
_CODEC_TO_ENCODER = {"h264": "libx264", "hevc": "libx265", "av1": "libsvtav1"}


def _source_vcodec(dataset: LeRobotDataset) -> str:
    """Software encoder name matching this dataset's own declared video codec."""
    for feat in dataset.meta.info.get("features", {}).values():
        codec = feat.get("info", {}).get("video.codec")
        if codec:
            return _CODEC_TO_ENCODER.get(codec, "libsvtav1")
    return "libsvtav1"


def _delete_episodes_same_codec(dataset: LeRobotDataset, episode_indices: list[int],
                                 output_dir: Path, repo_id: str) -> LeRobotDataset:
    """Same behavior as ``lerobot.datasets.dataset_tools.delete_episodes``, except
    video re-encoding (only triggered for source files that mix kept/dropped
    episodes) uses the source's own codec instead of that function's hardcoded
    ``vcodec="libsvtav1"`` default."""
    vcodec = _source_vcodec(dataset)
    episodes_to_keep = [i for i in range(dataset.meta.total_episodes) if i not in episode_indices]
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id, fps=dataset.meta.fps, features=dataset.meta.features,
        robot_type=dataset.meta.robot_type, root=output_dir,
        use_videos=len(dataset.meta.video_keys) > 0,
    )
    episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(episodes_to_keep)}
    video_metadata = None
    if dataset.meta.video_keys:
        logger.info("  re-encoding split video segments with vcodec=%s (source codec-matched)", vcodec)
        video_metadata = _copy_and_reindex_videos(dataset, new_meta, episode_mapping, vcodec=vcodec)
    data_metadata = _copy_and_reindex_data(dataset, new_meta, episode_mapping)
    _copy_and_reindex_episodes_metadata(dataset, new_meta, episode_mapping, data_metadata, video_metadata)
    return LeRobotDataset(repo_id=repo_id, root=output_dir, image_transforms=dataset.image_transforms,
                           delta_timestamps=dataset.delta_timestamps, tolerance_s=dataset.tolerance_s)


def _filter_by_length(ds: LeRobotDataset, min_len: int, staging_dir: Path,
                       tolerance_s: float) -> LeRobotDataset:
    filtered_repo_id = f"{ds.repo_id}_filtered"
    if (staging_dir / "meta" / "info.json").exists():
        logger.info("  reusing already-filtered dataset at %s", staging_dir)
        return LeRobotDataset(repo_id=filtered_repo_id, root=staging_dir, tolerance_s=tolerance_s)

    lengths = {int(row["episode_index"]): int(row["length"]) for row in ds.meta.episodes}
    drop = sorted(idx for idx, length in lengths.items() if length <= min_len)
    keep_n = ds.meta.total_episodes - len(drop)
    logger.info("  %s: keep %d/%d episodes with length > %d", ds.repo_id, keep_n,
                ds.meta.total_episodes, min_len)
    if not drop:
        return ds
    if keep_n == 0:
        raise ValueError(f"min_episode_len={min_len} would drop every episode in {ds.repo_id}")
    return _delete_episodes_same_codec(ds, episode_indices=drop, output_dir=staging_dir,
                                        repo_id=filtered_repo_id)


def _carry_subtasks_parquet(source_roots: list[Path], out_root: Path) -> None:
    """Copy ``meta/subtasks.parquet`` through to the merged dataset.

    This is a project-specific subtask_index -> text lookup table (written by
    ``data_to_lerobot3_node.py``), not part of vanilla LeRobot -- unlike
    ``tasks.parquet``, ``aggregate_datasets``/``delete_episodes`` don't know
    about it and silently drop it. Filtering by episode length doesn't change
    a dataset's subtask vocabulary, so we source it from the original
    (pre-filter) roots. If sources disagree, refuse to guess: silently picking
    one table could remap another source's subtask_index values to the wrong
    text.
    """
    found = [(root, root / "meta" / "subtasks.parquet") for root in source_roots]
    found = [(root, path) for root, path in found if path.exists()]
    if not found:
        return
    first_root, first_path = found[0]
    first_df = pd.read_parquet(first_path)
    for root, path in found[1:]:
        if not pd.read_parquet(path).equals(first_df):
            raise ValueError(
                f"meta/subtasks.parquet differs between {first_root} and {root} -- merging would "
                "silently remap subtask_index meaning for one of them. Resolve the vocabulary "
                "mismatch manually before merging."
            )
    dst = out_root / "meta" / "subtasks.parquet"
    shutil.copy(first_path, dst)
    logger.info("Copied meta/subtasks.parquet from %s -> %s (%d source(s) agreed)",
                first_root, dst, len(found))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", required=True,
                         help="One or more 'path' or 'path:min_episode_len' dataset roots to merge, "
                              "in order")
    parser.add_argument("--out", required=True, help="Output dataset root directory (must not exist)")
    parser.add_argument("--repo_id", default=None,
                         help="repo_id for the merged dataset (default: derived from --out)")
    parser.add_argument("--tolerance_s", type=float, default=1e-2,
                         help="Timestamp tolerance for loading source LeRobotDatasets")
    parser.add_argument("--video_files_size_in_mb", type=float, default=1.0,
                         help="Max MB per merged video file before rotating to a new one. Kept small "
                              "(default 1MB, well under lerobot's usual 200MB) so aggregate_datasets "
                              "always starts a fresh file per source video instead of muxer-concatenating "
                              "into an existing one -- concatenation via PyAV has hit "
                              "'non monotonically increasing dts' errors across independently-encoded "
                              "source videos.")
    args = parser.parse_args()

    out_root = Path(args.out)
    if out_root.exists():
        raise FileExistsError(f"Output dataset already exists at {out_root} -- remove it first or "
                               f"pick another --out")
    repo_id = args.repo_id or _repo_id_for(out_root)

    staging_root = out_root.parent / f".{out_root.name}_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)

    merge_inputs = []
    orig_source_roots = []
    for spec in args.sources:
        src_root, min_len = _parse_source(spec)
        orig_source_roots.append(src_root)
        src_repo_id = _repo_id_for(src_root)
        logger.info("Loading %s from %s (min_episode_len=%d) ...", src_repo_id, src_root, min_len)
        ds = LeRobotDataset(repo_id=src_repo_id, root=src_root, tolerance_s=args.tolerance_s)
        logger.info("  %d episodes, %d frames", ds.meta.total_episodes, ds.meta.total_frames)
        if min_len > 0:
            ds = _filter_by_length(ds, min_len, staging_root / src_root.name, args.tolerance_s)
        merge_inputs.append(ds)

    logger.info("Merging %d dataset(s) -> %s @ %s", len(merge_inputs), repo_id, out_root)
    aggregate_datasets(
        repo_ids=[ds.repo_id for ds in merge_inputs],
        aggr_repo_id=repo_id,
        roots=[ds.root for ds in merge_inputs],
        aggr_root=out_root,
        video_files_size_in_mb=args.video_files_size_in_mb,
    )
    _carry_subtasks_parquet(orig_source_roots, out_root)
    merged = LeRobotDataset(repo_id=repo_id, root=out_root, tolerance_s=args.tolerance_s)
    logger.info("Done: %d episodes, %d frames -> %s", merged.meta.total_episodes,
                merged.meta.total_frames, out_root)

    if staging_root.exists():
        shutil.rmtree(staging_root)
        logger.info("Cleaned up staging dir %s", staging_root)


if __name__ == "__main__":
    main()
