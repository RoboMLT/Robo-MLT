#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot v3 Data Replay Node

Loads a LeRobot v3 dataset episode (Parquet + MP4), plots the state / action curves, writes a
stitched camera video, and optionally replays the action trajectory on the real Piper arms.

Robot replay moves the arm(s) slowly to the first recorded pose, then streams the actions in
position mode at ``--speed`` percent. Keep a hand on the emergency stop.

Usage (standalone)::

    python data_replay_node.py --dataset_dir /path/to/lerobot_v3 --episode_idx 0

Usage (via rosrun)::

    rosrun dual_piper data_replay_node.py --dataset_dir /path/to/lerobot_v3 --episode_idx 0
"""

import argparse
import json
import os

import av
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# ── Optional ROS ──────────────────────────────────────────────────────────────
try:
    import rospy
    _HAS_ROS = True
except ImportError:
    _HAS_ROS = False


def _log(msg: str, level: str = 'info'):
    if _HAS_ROS and rospy.core.is_initialized():
        fn = {'info': 'loginfo', 'warn': 'logwarn', 'error': 'logerr'}.get(level, 'loginfo')
        getattr(rospy, fn)(msg)
    else:
        prefix = {'info': '[INFO]', 'warn': '[WARN]', 'error': '[ERROR]'}.get(level, '[INFO]')
        print(f'{prefix} {msg}', flush=True)


# ── Constants ─────────────────────────────────────────────────────────────────
# Piper SDK unit conversions (for robot replay)
RAD_TO_DEG_X1000 = 180.0 / np.pi * 1000.0
GRIPPER_SCALE = 1_000_000.0

# Camera stitch order (left → right in the output frame)
_STITCH_ORDER = ('cam_top', 'cam_left', 'cam_right')

# robot_type → ordered list of active arms. Mirrors the collection node so the
# state/action column layout (left arm first, then right) is interpreted the
# same way it was written.
_ROBOT_TYPE_ARMS = {
    'dual_piper':         ['left', 'right'],
    'single_piper_left':  ['left'],
    'single_piper_right': ['right'],
}


def _resolve_arms(info: dict, action_dim: int):
    """Determine which physical arm(s) an episode's action vector drives.

    Prefers the authoritative ``robot_type`` recorded in meta/info.json; falls
    back to inferring the arm count from the action width (assuming 7 DOF per
    arm) when robot_type is missing/unknown.

    Returns ``(arms, num_joints_per_arm)`` where *arms* is an ordered list of
    'left'/'right'.
    """
    rt = info.get('robot_type')
    if rt in _ROBOT_TYPE_ARMS:
        arms = list(_ROBOT_TYPE_ARMS[rt])
        njp = action_dim // len(arms)
        _log(f'  robot_type     : {rt} → arms={arms}, {njp} DOF/arm')
        return arms, njp

    # Fallback: infer arm count from the action dimension (7 DOF per arm).
    n = max(1, round(action_dim / 7))
    arms = ['left', 'right'][:n]
    njp = action_dim // n
    _log(f"  robot_type '{rt}' unknown — inferred {n} arm(s) from action_dim="
         f'{action_dim} → arms={arms} (assuming left first). Pass a correct '
         f'robot_type in info.json if this is wrong.', 'warn')
    return arms, njp


# ═════════════════════════════════════════════════════════════════════════════
#  Dataset helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_info(dataset_dir: str) -> dict:
    path = os.path.join(dataset_dir, 'meta', 'info.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'meta/info.json not found: {path}')
    with open(path) as f:
        return json.load(f)


def _load_subtasks(dataset_dir: str) -> dict:
    """Return {subtask_index (int): subtask_name (str)}, empty dict if file absent."""
    path = os.path.join(dataset_dir, 'meta', 'subtasks.parquet')
    if not os.path.isfile(path):
        return {}
    df = pq.read_table(path).to_pandas()
    # index column = subtask_index, 'subtask' is the row index label
    return {int(row['subtask_index']): idx for idx, row in df.iterrows()}


def _load_episode_meta(dataset_dir: str, episode_idx: int) -> pd.Series:
    """Return the episode metadata row for *episode_idx*."""
    ep_dir = os.path.join(dataset_dir, 'meta', 'episodes')
    rows = []
    for chunk_dir in sorted(os.listdir(ep_dir)):
        for fname in sorted(os.listdir(os.path.join(ep_dir, chunk_dir))):
            tbl = pq.read_table(os.path.join(ep_dir, chunk_dir, fname)).to_pandas()
            rows.append(tbl)
    all_eps = pd.concat(rows, ignore_index=True)
    match = all_eps[all_eps['episode_index'] == episode_idx]
    if match.empty:
        raise ValueError(f'Episode {episode_idx} not found (total: {len(all_eps)})')
    return match.iloc[0]


def _load_episode_data(dataset_dir: str, ep_meta: pd.Series) -> pd.DataFrame:
    """Load the data parquet rows that belong to this episode."""
    chunk_idx = int(ep_meta['data/chunk_index'])
    file_idx = int(ep_meta['data/file_index'])
    parquet_path = os.path.join(
        dataset_dir, 'data',
        f'chunk-{chunk_idx:03d}', f'file-{file_idx:03d}.parquet'
    )
    df = pq.read_table(parquet_path).to_pandas()
    ep_idx = int(ep_meta['episode_index'])
    return df[df['episode_index'] == ep_idx].reset_index(drop=True)


def _load_camera_frames(dataset_dir: str, ep_meta: pd.Series,
                        camera_name: str, fps: float) -> list:
    """
    Read all frames for *camera_name* in this episode from its MP4 file.

    Uses PyAV (libdav1d) so AV1-encoded videos are decoded correctly, unlike
    the default OpenCV build which lacks a software AV1 decoder.

    Returns a list of BGR uint8 ndarrays (H, W, 3).
    """
    key = f'videos/observation.images.{camera_name}'
    if f'{key}/chunk_index' not in ep_meta.index:
        _log(f'Camera {camera_name} not found in episode metadata — skipping.', 'warn')
        return []

    chunk_idx = int(ep_meta[f'{key}/chunk_index'])
    file_idx = int(ep_meta[f'{key}/file_index'])
    from_ts = float(ep_meta[f'{key}/from_timestamp'])
    to_ts = float(ep_meta[f'{key}/to_timestamp'])

    video_path = os.path.join(
        dataset_dir, 'videos', f'observation.images.{camera_name}',
        f'chunk-{chunk_idx:03d}', f'file-{file_idx:03d}.mp4'
    )
    if not os.path.isfile(video_path):
        _log(f'Video file not found: {video_path}', 'warn')
        return []

    n_frames = int(round((to_ts - from_ts) * fps))

    container = av.open(video_path)
    stream = container.streams.video[0]

    # Seek to approximately the right position (AV_TIME_BASE = microseconds)
    seek_pts = int(from_ts * 1e6)
    container.seek(seek_pts)

    frames = []
    for pkt in container.decode(stream):
        frame_ts = float(pkt.pts * stream.time_base)
        if frame_ts < from_ts - 1e-3:
            continue
        # PyAV returns RGB; convert to BGR for consistency with cv2
        frames.append(pkt.to_ndarray(format='bgr24'))
        if len(frames) >= n_frames:
            break

    container.close()
    return frames


# ═════════════════════════════════════════════════════════════════════════════
#  Visualisation
# ═════════════════════════════════════════════════════════════════════════════

def _joint_label(info: dict, idx: int, prefix: str) -> str:
    """Return a human-readable label for joint *idx* from info.json feature names."""
    try:
        return info['features'][prefix]['names'][idx]
    except (KeyError, IndexError, TypeError):
        return f'j{idx}'


def plot_qpos_action(qpos: np.ndarray, action: np.ndarray,
                     info: dict, output_path: str) -> None:
    """Plot joint-position (qpos) and action curves; save to *output_path*."""
    T = qpos.shape[0]
    t = np.arange(T) / float(info.get('fps', 20))

    dim = qpos.shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i in range(dim):
        lbl = _joint_label(info, i, 'observation.state')
        axes[0].plot(t, qpos[:, i], label=lbl)
    axes[0].set_title('Observation State (qpos) over Time')
    axes[0].set_ylabel('Position (rad)')
    axes[0].legend(fontsize=6, ncol=4)
    axes[0].grid(True, alpha=0.4)

    for i in range(action.shape[1]):
        lbl = _joint_label(info, i, 'action')
        axes[1].plot(t, action[:, i], label=lbl)
    axes[1].set_title('Action over Time')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Position (rad)')
    axes[1].legend(fontsize=6, ncol=4)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    _log(f'Saved joint curve plot → {output_path}')


def stitch_and_save_video(image_dict: dict, output_path: str, fps: float) -> None:
    """
    Horizontally stitch multi-camera frames and write an MP4 video.

    Args:
        image_dict: {camera_name: list[BGR ndarray]}
        output_path: output .mp4 path
        fps: output video frame rate
    """
    cam_names = [c for c in _STITCH_ORDER if c in image_dict and image_dict[c]]
    if not cam_names:
        _log('No camera frames to stitch.', 'warn')
        return

    frame_lists = [image_dict[c] for c in cam_names]
    n_frames = min(len(fl) for fl in frame_lists)
    if n_frames == 0:
        _log('All camera frame lists are empty — skipping video.', 'warn')
        return

    h = max(fl[0].shape[0] for fl in frame_lists)
    w = sum(fl[0].shape[1] for fl in frame_lists)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i in tqdm(range(n_frames), desc='Stitching video', ncols=80):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        x = 0
        for fl in frame_lists:
            frame = fl[i] if i < len(fl) else np.zeros_like(fl[0])
            fh, fw = frame.shape[:2]
            canvas[:fh, x:x + fw] = frame
            x += fw
        writer.write(canvas)

    writer.release()
    _log(f'Saved stitched video → {output_path}')


# ═════════════════════════════════════════════════════════════════════════════
#  Real-robot replay (optional)
# ═════════════════════════════════════════════════════════════════════════════

def replay_on_robot(actions: np.ndarray, fps: float, arms: list,
                    can_ports: dict, njp: int, speed: int = 30, mit: bool = False) -> None:
    """Send an action sequence to the active Piper arm(s) via the SDK.

    Only the arms in *arms* (derived from the dataset's robot_type) are
    connected and driven. The action vector is split into ``njp``-DOF segments,
    one per arm, in the same left→right order used during collection:
        action[i*njp : (i+1)*njp]  →  arms[i]
    Each segment is 6 joints + 1 gripper.
    """
    try:
        from piper_sdk import C_PiperInterface
    except ImportError:
        _log('piper_sdk not installed — cannot replay on robot.', 'error')
        return

    import time

    # Guard: the action width must match the resolved arm layout.
    expected = len(arms) * njp
    if actions.shape[1] != expected:
        _log(f'Action width {actions.shape[1]} != arms×njp ({expected}). '
             f'Refusing to replay — robot_type / dataset mismatch.', 'error')
        return

    # ── Connect only the active arm(s) ───────────────────────────────────
    pipers = {}
    for arm in arms:
        port = can_ports[arm]
        _log(f'Connecting {arm} arm on CAN port {port}')
        p = C_PiperInterface(can_name=port)
        p.ConnectPort()
        p.EnableArm(7)
        pipers[arm] = p
    time.sleep(1.0)

    def send(action):
        for i, arm in enumerate(arms):
            seg = action[i * njp:(i + 1) * njp]
            joints = [int(v * RAD_TO_DEG_X1000) for v in seg[:6]]
            grip = int(abs(seg[6] * GRIPPER_SCALE))
            pipers[arm].JointCtrl(*joints)
            pipers[arm].GripperCtrl(grip, 1000, 0x01, 0)

    # Approach the first recorded pose slowly (position mode, 20 % speed).
    _log('Moving to the first recorded pose …')
    for p in pipers.values():
        p.MotionCtrl_2(0x01, 0x01, 20)
    for _ in range(int(3.0 * fps)):
        send(actions[0])
        time.sleep(1.0 / fps)

    dt = 1.0 / fps
    speed = int(max(1, min(100, speed)))
    _log(f'Replaying {len(actions)} frames at {fps} Hz on arm(s) {arms} '
         f'({"MIT" if mit else "position"} mode, speed {speed} %) …')
    for p in pipers.values():
        if mit:
            p.MotionCtrl_2(0x01, 0x01, speed, 0xAD)
        else:
            p.MotionCtrl_2(0x01, 0x01, speed)

    for action in tqdm(actions, desc='Replaying', ncols=80):
        send(action)
        time.sleep(dt)

    _log('Trajectory replay finished.')


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Replay / visualize a LeRobot v3 dataset episode.')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Root directory of the LeRobot v3 dataset.')
    parser.add_argument('--episode_idx', type=int, default=0,
                        help='Episode index to replay / visualize.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots and videos. '
                             'Defaults to <dataset_dir>/replay_output.')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip joint-curve plot generation.')
    parser.add_argument('--no_video', action='store_true',
                        help='Skip stitched video generation.')
    parser.add_argument('--replay_on_robot', action='store_true',
                        help='Send actions to real Piper arms via CAN bus.')
    parser.add_argument('--can_left', type=str, default='can_left',
                        help='CAN port for the left arm (robot replay only).')
    parser.add_argument('--can_right', type=str, default='can_right',
                        help='CAN port for the right arm (robot replay only).')
    parser.add_argument('--speed', type=int, default=30,
                        help='Replay speed percentage sent to MotionCtrl_2 (robot replay only).')
    parser.add_argument('--mit', action='store_true',
                        help='Use MIT mode (0xAD) instead of position mode for the replay.')
    return parser.parse_args()


def main():
    if _HAS_ROS:
        rospy.init_node('data_replay_node', anonymous=True)

    args = _parse_args()

    dataset_dir = args.dataset_dir
    episode_idx = args.episode_idx
    output_dir = args.output_dir or os.path.join(dataset_dir, 'replay_output')

    # ── Load dataset metadata ──────────────────────────────────────────────
    _log(f'Loading dataset: {dataset_dir}')
    info = _load_info(dataset_dir)
    fps = float(info.get('fps', 25))
    camera_names = [
        k.replace('observation.images.', '')
        for k in info.get('features', {})
        if k.startswith('observation.images.')
    ]
    subtasks = _load_subtasks(dataset_dir)

    _log(f'  total_episodes : {info.get("total_episodes")}')
    _log(f'  total_frames   : {info.get("total_frames")}')
    _log(f'  fps            : {fps}')
    _log(f'  cameras        : {camera_names}')
    _log(f'  subtasks       : {subtasks}')

    # ── Load episode ───────────────────────────────────────────────────────
    _log(f'Loading episode {episode_idx} …')
    ep_meta = _load_episode_meta(dataset_dir, episode_idx)
    ep_df = _load_episode_data(dataset_dir, ep_meta)

    n_frames = len(ep_df)
    _log(f'  frames: {n_frames}  length: {ep_meta["length"]}')

    # ── Extract arrays ─────────────────────────────────────────────────────
    qpos = np.stack(ep_df['observation.state'].values).astype(np.float32)
    action = np.stack(ep_df['action'].values).astype(np.float32)

    # ── Auto-detect robot type → active arm(s) for replay ──────────────────
    arms, njp = _resolve_arms(info, action.shape[1])

    if subtasks and 'subtask_index' in ep_df.columns:
        unique_st = ep_df['subtask_index'].dropna().unique()
        _log(f'  subtasks in episode: {[subtasks.get(int(i), i) for i in sorted(unique_st)]}')

    # ── Plot joint curves ──────────────────────────────────────────────────
    if not args.no_plot:
        plot_path = os.path.join(output_dir, f'qpos_action_ep{episode_idx:04d}.png')
        plot_qpos_action(qpos, action, info, plot_path)

    # ── Load and stitch camera videos ──────────────────────────────────────
    if not args.no_video:
        image_dict = {}
        for cam in camera_names:
            _log(f'  loading camera: {cam}')
            frames = _load_camera_frames(dataset_dir, ep_meta, cam, fps)
            _log(f'    → {len(frames)} frames')
            image_dict[cam] = frames

        video_path = os.path.join(output_dir, f'stitched_ep{episode_idx:04d}.mp4')
        stitch_and_save_video(image_dict, video_path, fps)

    # ── Optional robot replay ──────────────────────────────────────────────
    if args.replay_on_robot:
        can_ports = {'left': args.can_left, 'right': args.can_right}
        replay_on_robot(action, fps, arms, can_ports, njp, speed=args.speed, mit=args.mit)

    _log(f'Done. Output dir: {output_dir}')


if __name__ == '__main__':
    main()
