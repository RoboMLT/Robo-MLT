#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-Piper → Single-Piper LeRobot v3 Dataset Converter
======================================================
Splits a 14-DoF dual-arm Piper LeRobot v3 dataset into a 7-DoF single-arm
dataset for one chosen arm.

state / action layout in the source dataset:
    dims [0:7]  → left  arm  (6 joints + gripper)
    dims [7:14] → right arm  (6 joints + gripper)

Camera handling (to avoid the static/other arm leaking into the observation):
    --arm left   keeps  cam_top + cam_left   (drops cam_right)
    --arm right  keeps  cam_top + cam_right   (drops cam_left)
The overhead camera (cam_top) is kept by default; pass --no_top to drop it.
Pass --unify_wrist to rename the kept wrist camera to a common 'cam_wrist'
so that the left and right datasets share an identical feature schema.

Per-frame ``subtask_index`` and ``back_event`` columns and the recovery metadata files
(``meta/recovery_episodes.json``, ``meta/recovery_labels.json``) are carried over unchanged; the
output ``robot_type`` is ``single_piper_<arm>`` so the replay tool drives the right arm.

The Robo-MLT paper datasets are dual-arm and do not need this tool.

Usage::

    python tools/dualpiper_to_singlepiper.py --arm left \\
        --src <data_root>/<dual_arm_dataset> --out_dir <data_root>
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_SUBTASKS_PATH


# ── Arm → (state/action slice, kept wrist camera) ───────────────────────────
ARM_SPEC = {
    'left':  {'slice': slice(0, 7),   'wrist': 'cam_left'},
    'right': {'slice': slice(7, 14),  'wrist': 'cam_right'},
}

# Clean single-arm joint names (6 joints + gripper)
SINGLE_ARM_JOINT_NAMES = [f'joint_{i}.pos' for i in range(1, 7)] + ['gripper.pos']


# ═════════════════════════════════════════════════════════════════════════════
#  Source-dataset readers
# ═════════════════════════════════════════════════════════════════════════════

def load_info(src: str) -> dict:
    with open(os.path.join(src, 'meta', 'info.json')) as f:
        return json.load(f)


def load_tasks(src: str) -> dict:
    """Return {task_index (int): task_str}."""
    df = pq.read_table(os.path.join(src, 'meta', 'tasks.parquet')).to_pandas()
    # Index label is the task string, column 'task_index' holds the int.
    return {int(row['task_index']): idx for idx, row in df.iterrows()}


def load_episodes_meta(src: str) -> pd.DataFrame:
    ep_dir = os.path.join(src, 'meta', 'episodes')
    parts = []
    for chunk in sorted(os.listdir(ep_dir)):
        for fname in sorted(os.listdir(os.path.join(ep_dir, chunk))):
            parts.append(pq.read_table(os.path.join(ep_dir, chunk, fname)).to_pandas())
    return pd.concat(parts, ignore_index=True)


def load_episode_rows(src: str, ep_meta: pd.Series) -> pd.DataFrame:
    chunk_idx = int(ep_meta['data/chunk_index'])
    file_idx = int(ep_meta['data/file_index'])
    path = os.path.join(src, 'data', f'chunk-{chunk_idx:03d}', f'file-{file_idx:03d}.parquet')
    df = pq.read_table(path).to_pandas()
    ep_idx = int(ep_meta['episode_index'])
    return df[df['episode_index'] == ep_idx].sort_values('frame_index').reset_index(drop=True)


def decode_episode_camera(src: str, ep_meta: pd.Series, camera: str, fps: float) -> list:
    """Decode all RGB frames of *camera* for this episode (PyAV, handles av1/h264)."""
    key = f'videos/observation.images.{camera}'
    chunk_idx = int(ep_meta[f'{key}/chunk_index'])
    file_idx = int(ep_meta[f'{key}/file_index'])
    from_ts = float(ep_meta[f'{key}/from_timestamp'])
    to_ts = float(ep_meta[f'{key}/to_timestamp'])

    video_path = os.path.join(src, 'videos', f'observation.images.{camera}',
                              f'chunk-{chunk_idx:03d}', f'file-{file_idx:03d}.mp4')
    n_frames = int(round((to_ts - from_ts) * fps))

    container = av.open(video_path)
    stream = container.streams.video[0]
    container.seek(int(from_ts * 1e6))

    frames = []
    for pkt in container.decode(stream):
        if float(pkt.pts * stream.time_base) < from_ts - 1e-3:
            continue
        frames.append(pkt.to_ndarray(format='rgb24'))  # HWC uint8 RGB
        if len(frames) >= n_frames:
            break
    container.close()
    return frames


# ═════════════════════════════════════════════════════════════════════════════
#  Conversion
# ═════════════════════════════════════════════════════════════════════════════

def convert(args):
    src = args.src.rstrip('/')
    arm = args.arm
    spec = ARM_SPEC[arm]
    sl = spec['slice']

    info = load_info(src)
    fps = float(info['fps'])
    has_subtask = 'subtask_index' in info['features']
    has_back = 'back_event' in info['features']

    # ── Determine kept cameras and their output names ───────────────────────
    src_wrist = spec['wrist']
    out_wrist = 'cam_wrist' if args.unify_wrist else src_wrist
    cam_map = {}  # output_name → source_name
    if not args.no_top:
        cam_map['cam_top'] = 'cam_top'
    cam_map[out_wrist] = src_wrist

    # ── Output location ─────────────────────────────────────────────────────
    out_name = args.out_name or f'{Path(src).name}_{arm}'
    target = os.path.join(args.out_dir, out_name)
    if os.path.exists(target):
        if not args.overwrite:
            raise FileExistsError(f'{target} already exists. Use --overwrite to replace.')
        print(f'[warn] overwriting existing {target}')
        shutil.rmtree(target)

    print('=' * 70)
    print('  Dual → Single Piper conversion')
    print(f'  src      : {src}')
    print(f'  arm      : {arm}  (state/action dims {sl.start}:{sl.stop})')
    print('  cameras  : ' + ', '.join(f'{o}<-{s}' for o, s in cam_map.items()))
    print(f'  out      : {target}')
    print(f'  codec    : {args.video_codec}  streaming={not args.no_streaming}')
    print('=' * 70)

    # ── Build single-arm feature schema ─────────────────────────────────────
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (7,), 'names': SINGLE_ARM_JOINT_NAMES},
        'action':            {'dtype': 'float32', 'shape': (7,), 'names': SINGLE_ARM_JOINT_NAMES},
    }
    for out_cam, src_cam in cam_map.items():
        sfeat = info['features'][f'observation.images.{src_cam}']
        features[f'observation.images.{out_cam}'] = {
            'dtype': 'video',
            'shape': tuple(sfeat['shape']),
            'names': ['height', 'width', 'channel'],
        }
    if has_subtask:
        features['subtask_index'] = {'dtype': 'int64', 'shape': (1,), 'names': None}
    if has_back:
        features['back_event'] = {'dtype': 'int64', 'shape': (1,), 'names': None}

    # ── Create the output dataset ───────────────────────────────────────────
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id or out_name,
        root=target,
        fps=int(fps),
        robot_type=args.robot_type or f'single_piper_{arm}',
        features=features,
        use_videos=True,
        tolerance_s=1e-4,
        image_writer_processes=0,
        image_writer_threads=4,
        vcodec=args.video_codec,
        streaming_encoding=not args.no_streaming,
        encoder_threads=4,
    )

    tasks = load_tasks(src)
    episodes = load_episodes_meta(src)

    # ── Iterate episodes ────────────────────────────────────────────────────
    total_frames = 0
    for _, ep_meta in tqdm(episodes.iterrows(), total=len(episodes), desc='Episodes'):
        rows = load_episode_rows(src, ep_meta)
        n = len(rows)

        # Decode kept cameras for this episode
        cam_frames = {
            out_cam: decode_episode_camera(src, ep_meta, src_cam, fps)
            for out_cam, src_cam in cam_map.items()
        }
        # Guard against decode/parquet length mismatch
        n = min([n] + [len(v) for v in cam_frames.values()])

        for i in range(n):
            row = rows.iloc[i]
            state = np.asarray(row['observation.state'], dtype=np.float32)[sl]
            action = np.asarray(row['action'], dtype=np.float32)[sl]

            task_str = tasks.get(int(row['task_index']), '') if 'task_index' in rows.columns else ''
            frame = {
                'observation.state': state,
                'action': action,
                'task': task_str,
            }
            for out_cam in cam_map:
                frame[f'observation.images.{out_cam}'] = cam_frames[out_cam][i]
            if has_subtask:
                frame['subtask_index'] = np.array([int(row['subtask_index'])], dtype=np.int64)
            if has_back:
                frame['back_event'] = np.array(
                    [int(np.asarray(row['back_event']).reshape(-1)[0])], dtype=np.int64)

            dataset.add_frame(frame)

        dataset.save_episode()
        dataset.meta._flush_metadata_buffer()
        total_frames += n

    # ── Finalize ────────────────────────────────────────────────────────────
    if dataset.image_writer is not None:
        dataset.stop_image_writer()
    dataset.finalize()

    # ── Copy subtask name→index mapping so subtask_index stays meaningful ────
    if has_subtask:
        src_sub = os.path.join(src, 'meta', 'subtasks.parquet')
        if os.path.isfile(src_sub):
            dst_sub = Path(target) / DEFAULT_SUBTASKS_PATH
            dst_sub.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_sub, dst_sub)
    # Episodes are converted one-to-one and in order, so per-episode metadata keeps its keys.
    for name in ('recovery_episodes.json', 'recovery_labels.json'):
        src_meta = os.path.join(src, 'meta', name)
        if os.path.isfile(src_meta):
            shutil.copy(src_meta, os.path.join(target, 'meta', name))

    print(f'\n✅ Done. {dataset.meta.total_episodes} episodes, '
          f'{dataset.meta.total_frames} frames → {target}')


def _parse_args():
    p = argparse.ArgumentParser(description='Convert a dual-arm Piper LeRobot v3 dataset to single-arm.')
    p.add_argument('--src', type=str, required=True, help='Source dual-arm dataset root.')
    p.add_argument('--arm', type=str, required=True, choices=['left', 'right'],
                   help='Which arm to extract (the hyper-parameter).')
    p.add_argument('--out_dir', type=str, required=True,
                   help='Directory to write the single-arm dataset into.')
    p.add_argument('--out_name', type=str, default=None,
                   help='Output dataset folder name. Default: <src_name>_<arm>.')
    p.add_argument('--repo_id', type=str, default=None,
                   help='repo_id for the new dataset. Default: out_name.')
    p.add_argument('--robot_type', type=str, default=None,
                   help='robot_type written into the new dataset. Default: single_piper_<arm>.')
    p.add_argument('--no_top', action='store_true',
                   help='Drop the overhead cam_top, keep only the wrist camera.')
    p.add_argument('--unify_wrist', action='store_true',
                   help="Rename the kept wrist camera to 'cam_wrist' so left/right "
                        "datasets share an identical schema.")
    p.add_argument('--video_codec', type=str, default='h264_nvenc',
                   help='Output video codec (h264_nvenc/hevc_nvenc/libsvtav1).')
    p.add_argument('--no_streaming', action='store_true',
                   help='Disable real-time streaming encoding.')
    p.add_argument('--overwrite', action='store_true',
                   help='Overwrite the output dataset if it already exists.')
    return p.parse_args()


if __name__ == '__main__':
    convert(_parse_args())
