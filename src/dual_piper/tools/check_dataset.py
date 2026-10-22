#!/usr/bin/env python3
"""Sanity-check a recorded LeRobot v3 dataset before training or release.

Reports, from parquet / JSON only (no ROS, no LeRobot):

    - episodes, frames, fps, robot_type, cameras, task text;
    - timestamp regularity (largest deviation of consecutive timestamps from 1/fps);
    - how ``action`` relates to ``observation.state`` (identical under
      ``action_source: follower_feedback``, see state_get_node.py);
    - subtask annotation: segments per episode, subtask indices never used;
    - recovery labels: episodes with ``back_event`` positives, whether those form the final block
      of each episode, and the episode tags of a recovery dataset.

Usage::

    python tools/check_dataset.py --root <data_root>/blood_gas_analysis
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_COLUMNS = ('episode_index', 'frame_index', 'timestamp', 'action', 'observation.state',
            'subtask_index', 'back_event')


def _scalar(values):
    return np.asarray([np.asarray(v).reshape(-1)[0] for v in values], dtype=np.int64)


def check(root: Path) -> dict:
    info = json.loads((root / 'meta' / 'info.json').read_text(encoding='utf-8'))
    files = sorted(root.glob('data/**/*.parquet'))
    names = set(pq.read_schema(files[0]).names)
    df = pd.concat([pd.read_parquet(f, columns=[c for c in _COLUMNS if c in names]) for f in files],
                   ignore_index=True)
    fps = float(info.get('fps', 0)) or None
    report = {
        'robot_type': info.get('robot_type'), 'fps': fps,
        'cameras': [k for k, v in info['features'].items() if v.get('dtype') in ('video', 'image')],
        'episodes': int(df['episode_index'].nunique()), 'frames': int(len(df)),
    }
    tasks = root / 'meta' / 'tasks.parquet'
    if tasks.exists():
        report['tasks'] = [str(t) for t in pd.read_parquet(tasks).index]

    sub = _scalar(df['subtask_index']) if 'subtask_index' in df else None
    back = _scalar(df['back_event']) if 'back_event' in df else None
    gaps, seg_counts, tails, n_pos_eps, eq_frames, abs_diff = [], [], [], 0, 0, []
    for _, idx in df.groupby('episode_index').indices.items():
        idx = idx[np.argsort(df['frame_index'].to_numpy()[idx])]
        if fps and 'timestamp' in df:
            ts = df['timestamp'].to_numpy()[idx]
            if len(ts) > 1:
                gaps.append(float(np.abs(np.diff(ts) - 1.0 / fps).max()))
        if 'action' in df and 'observation.state' in df:
            a = np.stack(df['action'].to_numpy()[idx])
            s = np.stack(df['observation.state'].to_numpy()[idx])
            abs_diff.append(float(np.abs(a - s).mean()))
            eq_frames += int(np.all(np.isclose(a, s), axis=1).sum())
        if sub is not None:
            sq = sub[idx]
            seg_counts.append(1 + int((sq[1:] != sq[:-1]).sum()))
        if back is not None:
            b = back[idx] > 0
            if b.any():
                n_pos_eps += 1
                tails.append(bool(b[-int(b.sum()):].all()))

    if gaps:
        report['max_timestamp_jitter_s'] = max(gaps)
    if abs_diff:
        report['mean_abs_action_minus_state'] = float(np.mean(abs_diff))
        report['frames_with_action_equal_state'] = f'{eq_frames}/{len(df)}'
    if seg_counts:
        n_sub = len(pd.read_parquet(root / 'meta' / 'subtasks.parquet')) \
            if (root / 'meta' / 'subtasks.parquet').exists() else None
        report['episodes_with_one_segment'] = int(sum(c == 1 for c in seg_counts))
        report['episodes_with_several_segments'] = int(sum(c > 1 for c in seg_counts))
        if n_sub is not None:
            report['unused_subtask_indices'] = sorted(set(range(n_sub)) - set(sub.tolist()))
    if back is not None:
        report['episodes_with_recovery_positives'] = n_pos_eps
        report['positives_are_final_block'] = all(tails) if tails else None
    tags = root / 'meta' / 'recovery_episodes.json'
    if tags.exists():
        kinds = {}
        for t in json.loads(tags.read_text(encoding='utf-8')).values():
            kinds[t.get('type', '?')] = kinds.get(t.get('type', '?'), 0) + 1
        report['recovery_episode_types'] = kinds
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, type=Path)
    args = ap.parse_args()
    for key, value in check(args.root).items():
        print(f'{key:34s} {value}')


if __name__ == '__main__':
    main()
