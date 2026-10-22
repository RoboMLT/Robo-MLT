#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for the dual_piper package.

Provides:
- Image conversion (ROS ↔ OpenCV)
- Point-cloud sampling
- LeRobot v3 dataset crash-recovery
- Piper SDK unit constants
"""
import os
import json
import time
import shutil
from pathlib import Path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2

# ═══════════════════════════════ Constants ═══════════════════════════════
# Piper SDK uses milli-degrees internally; convert to radians.
DEG_TO_RAD = 0.01745329251994  # π / 180
JOINT_SCALE = 1000.0           # SDK joint unit → milli-degrees
GRIPPER_SCALE = 1_000_000.0    # SDK gripper unit → angle
EFFORT_SCALE = 1000.0          # SDK effort unit
RAD_TO_DEG_X1000 = 180.0 / np.pi * 1000.0  # rad → SDK milli-degrees
ENABLE_ALL_JOINTS = 7          # EnableArm(7) enables all 6 joints + gripper
PUBLISH_RATE_HZ = 200          # Default joint-state publish rate


# ═════════════════════════ Image Conversion ══════════════════════════════
def imgmsg_to_cv2(img_msg):
    """
    Pure-Python ROS Image → OpenCV (NumPy) conversion.

    Supported encodings: ``rgb8``, ``bgr8``.

    Returns:
        numpy.ndarray: ``(H, W, 3)`` BGR uint8 image.
    """
    if img_msg.encoding not in ('rgb8', 'bgr8'):
        raise ValueError(f"Unsupported encoding: {img_msg.encoding}")

    image = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
        img_msg.height, img_msg.width, 3
    )
    if img_msg.encoding == 'rgb8':
        image = image[:, :, ::-1]  # RGB → BGR
    return image


# ═════════════════════════ Point Cloud ═══════════════════════════════════
def sample_pointcloud(pc_msg, target_points=5000):
    """
    Convert ``PointCloud2`` → ``(target_points, 6)`` float32 ``[x,y,z,r,g,b]``.

    If the cloud has only XYZ, the RGB columns are zero-filled.
    Points are randomly down-/up-sampled to exactly *target_points*.
    """
    pts = [list(p[:6]) for p in pc2.read_points(pc_msg, skip_nans=True)]
    if not pts:
        return np.zeros((target_points, 6), dtype=np.float32)

    arr = np.asarray(pts, dtype=np.float32)
    # Pad to 6 columns if fewer
    if arr.shape[1] < 6:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 6 - arr.shape[1]), dtype=np.float32)])
    else:
        arr = arr[:, :6]

    mask = np.isfinite(arr).all(axis=1)
    arr = arr[mask]
    if arr.shape[0] == 0:
        return np.zeros((target_points, 6), dtype=np.float32)

    n = arr.shape[0]
    idx = np.random.choice(n, target_points, replace=(n < target_points))
    return arr[idx]


# ═══════════════════ LeRobot v3 Dataset Crash Recovery ═══════════════════
def _parse_video_path(rel_parts, stem):
    """('observation.images.cam_top', 'chunk-000', 'file-001.mp4') → (key, 0, 1)."""
    key = rel_parts[0]
    chunk_idx = int(rel_parts[1].split('-')[1])
    file_idx = int(stem.split('-')[1])
    return key, chunk_idx, file_idx


def _truncate_parquet_by_episode(paths, keep_before, quarantine_dir):
    """Keep only rows with episode_index < keep_before; drop files left empty.

    Rewrites at the pyarrow level so the original schema (and its metadata) is
    preserved exactly. Emptied files are moved to *quarantine_dir*, never deleted.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    for p in paths:
        table = pq.read_table(p)
        if 'episode_index' not in table.schema.names:
            continue
        mask = pc.less(table.column('episode_index'), keep_before)
        kept = table.filter(mask)
        if kept.num_rows == table.num_rows:
            continue                        # nothing to drop
        if kept.num_rows == 0:
            _quarantine(p, quarantine_dir)
            continue
        tmp = str(p) + '.tmp'
        pq.write_table(kept, tmp, compression='snappy', use_dictionary=True)
        os.replace(tmp, p)


def _quarantine(path, quarantine_dir):
    """Move *path* under *quarantine_dir*, preserving its relative layout."""
    path = Path(path)
    dst = Path(quarantine_dir) / path.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Keep a hint of the original location so the backup stays interpretable.
    dst = dst.with_name(f'{path.parent.name}__{path.name}')
    shutil.move(str(path), str(dst))
    return dst


def repair_dataset_consistency(target_dir):
    """Make a crash-interrupted LeRobot v3 dataset self-consistent again.

    A hard kill during recording leaves the dataset in a state LeRobotDataset
    cannot load, because data and metadata have *different durability*:

    1. Episode **data** goes through a persistent ``pq.ParquetWriter`` whose
       footer is only emitted by ``close()`` (i.e. on ``finalize()``). A killed
       process leaves a footerless, unreadable parquet file.
    2. Episode **metadata** is flushed after every episode, so it survives — and
       then references frames whose data never became durable. The mirror case
       also occurs: data written after the last metadata flush is orphaned.

    Either way ``LeRobotDataset.__init__`` concludes the local cache is
    incomplete and falls back to ``get_safe_version()`` → a HuggingFace Hub
    request. For a local-only / non-existent repo_id that request **hangs**,
    which surfaces to the operator as a frozen resume.

    Recording only ever appends, so damage is confined to the tail. This keeps
    the longest valid *prefix* of episodes (data and metadata agreeing on every
    episode length) and rewrites the counters to match. Everything removed is
    moved into a ``_quarantine_*`` folder inside the dataset, never deleted.

    Returns the number of episodes kept.
    """
    import pyarrow.parquet as pq
    import pandas as pd

    root = Path(target_dir)
    info_path = root / 'meta' / 'info.json'
    if not info_path.is_file():
        return 0

    qdir = root / f'_quarantine_{time.strftime("%Y%m%d_%H%M%S")}'

    # ── 1. Remove parquet files left footerless by the crash ──────────────
    corrupt = []
    for p in sorted(root.rglob('*.parquet')):
        try:
            pq.read_metadata(p)
        except Exception:
            corrupt.append(str(p.relative_to(root)))
            _quarantine(p, qdir)
    if corrupt:
        rospy.logwarn(
            f'Quarantined {len(corrupt)} unreadable parquet file(s) from a previous '
            f'crash:\n  ' + '\n  '.join(corrupt)
        )

    ep_files = sorted((root / 'meta' / 'episodes').rglob('*.parquet'))
    data_files = sorted((root / 'data').rglob('*.parquet'))
    if not ep_files or not data_files:
        rospy.logwarn('No episode metadata or data files found — nothing to repair.')
        return 0

    # ── 2. Compare declared episodes against rows actually on disk ────────
    eps = pd.concat([pd.read_parquet(f) for f in ep_files])
    eps = eps.sort_values('episode_index').reset_index(drop=True)
    actual = pd.concat(
        [pd.read_parquet(f, columns=['episode_index']) for f in data_files])
    counts = actual['episode_index'].value_counts()

    # ── 3. Longest prefix whose data is fully present ─────────────────────
    keep = 0
    for i, row in eps.iterrows():
        idx = int(row['episode_index'])
        if idx != i:                                   # gap in episode numbering
            break
        if int(counts.get(idx, 0)) != int(row['length']):
            break
        keep = i + 1

    declared_frames = int(eps['length'].sum())
    orphan_rows = int((actual['episode_index'] >= keep).sum())
    if keep == len(eps) and orphan_rows == 0 and int(counts.sum()) == declared_frames:
        rospy.loginfo(
            f'Dataset consistency OK: {keep} episodes / {declared_frames} frames.')
        return keep

    kept_frames = int(eps.iloc[:keep]['length'].sum()) if keep else 0
    rospy.logwarn(
        f'Inconsistent dataset detected — truncating to the last complete episode.\n'
        f'  episodes: {len(eps)} declared → {keep} kept ({len(eps) - keep} dropped)\n'
        f'  frames:   {declared_frames} declared → {kept_frames} kept\n'
        f'  orphan data rows (no metadata): {orphan_rows}'
    )

    # ── 4. Videos still referenced by a surviving episode ─────────────────
    # A video file shared by kept and dropped episodes is KEPT: the trailing
    # frames are unreferenced but harmless, since episodes index video by
    # timestamp range.
    referenced = set()
    kept_rows = eps.iloc[:keep]
    for col in [c for c in eps.columns
                if c.startswith('videos/') and c.endswith('/chunk_index')]:
        key = col[len('videos/'):-len('/chunk_index')]
        fcol = f'videos/{key}/file_index'
        if fcol not in eps.columns:
            continue
        for ci, fi in zip(kept_rows[col], kept_rows[fcol]):
            referenced.add((key, int(ci), int(fi)))

    # ── 5. Truncate metadata + data to the surviving prefix ───────────────
    _truncate_parquet_by_episode(ep_files, keep, qdir)
    _truncate_parquet_by_episode(data_files, keep, qdir)

    # ── 6. Quarantine video files holding no surviving episode ────────────
    videos_root = root / 'videos'
    if videos_root.is_dir():
        for vf in sorted(videos_root.rglob('*.mp4')):
            rel = vf.relative_to(videos_root)
            try:
                ident = _parse_video_path(rel.parts, vf.stem)
            except (IndexError, ValueError):
                continue
            if ident not in referenced:
                rospy.logwarn(f'  quarantining unreferenced video {rel}')
                _quarantine(vf, qdir)

    # ── 7. Leftover streaming-encoder temp dirs ───────────────────────────
    for tmp in sorted(root.glob('tmp*')):
        if tmp.is_dir():
            shutil.move(str(tmp), str(qdir / tmp.name))

    # ── 8. Rewrite the counters ───────────────────────────────────────────
    with open(info_path, 'r') as f:
        info = json.load(f)
    info['total_episodes'] = keep
    info['total_frames'] = kept_frames
    info['splits'] = {'train': f'0:{keep}'}
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    rospy.logwarn(
        f'Repaired dataset: total_episodes={keep}, total_frames={kept_frames}. '
        f'Removed files kept in {qdir.name}/ (delete once you are happy).\n'
        f'  NOTE: meta/stats.json still includes the dropped episodes. It is only '
        f'used for normalisation, so this is a small skew, not a load error.'
    )
    return keep
