#!/usr/bin/env python3
"""Offline tests of the dataset tools on a synthetic LeRobot v3 tree (pandas + pyarrow only).

Run:  python src/dual_piper/tools/test_tools.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_dataset import check  # noqa: E402
from label_recovery_tail import draw_k_rec, label_dataset  # noqa: E402


def make_recovery_tree(root: Path, lengths, scalar_column=True) -> Path:
    (root / 'meta' / 'episodes' / 'chunk-000').mkdir(parents=True)
    (root / 'data' / 'chunk-000').mkdir(parents=True)
    rows, ep_rows = [], []
    for ep, n in enumerate(lengths):
        for t in range(n):
            act = np.full(14, ep + t / 100.0, dtype=np.float32)
            rows.append({'episode_index': ep, 'frame_index': t, 'timestamp': t / 25.0,
                         'action': act, 'observation.state': act.copy(), 'subtask_index': 0,
                         'back_event': 0 if scalar_column else [0]})
        ep_rows.append({'episode_index': ep, 'length': n,
                        'stats/back_event/mean': np.zeros(1), 'stats/back_event/max': np.zeros(1),
                        'stats/back_event/count': np.array([n])})
    half = sum(lengths[:1])
    df = pd.DataFrame(rows)
    df.iloc[:half].to_parquet(root / 'data' / 'chunk-000' / 'file-000.parquet')
    df.iloc[half:].to_parquet(root / 'data' / 'chunk-000' / 'file-001.parquet')
    pd.DataFrame(ep_rows).to_parquet(root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet')
    (root / 'meta' / 'info.json').write_text(json.dumps({
        'robot_type': 'dual_piper', 'fps': 25,
        'features': {'observation.images.cam_top': {'dtype': 'video'}}}))
    (root / 'meta' / 'stats.json').write_text(json.dumps({'back_event': {'mean': [0.0]}}))
    pd.DataFrame({'subtask_index': [0]}, index=pd.Index(['grasp.'])).to_parquet(
        root / 'meta' / 'subtasks.parquet')
    (root / 'meta' / 'recovery_episodes.json').write_text(json.dumps(
        {str(i): {'type': 'intervention' if i % 2 == 0 else 'failure'} for i in range(len(lengths))}))
    return root


def test_label_recovery_tail():
    with tempfile.TemporaryDirectory() as tmp:
        lengths = [120, 80, 60]
        root = make_recovery_tree(Path(tmp) / 'rec', lengths)
        k = label_dataset(root, 20, 40, seed=0)
        assert k == draw_k_rec(range(3), 20, 40, 0), 'K_rec must be reproducible from the seed'
        assert all(20 <= v <= 40 for v in k.values())
        data = pd.concat([pd.read_parquet(f) for f in sorted(root.glob('data/**/*.parquet'))])
        for ep, n in enumerate(lengths):
            b = data[data.episode_index == ep].sort_values('frame_index')['back_event'].to_numpy()
            assert b.sum() == k[ep] and b[-k[ep]:].all() and not b[:-k[ep]].any(), ep
        eps = pd.read_parquet(root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet')
        assert np.isclose(eps.loc[0, 'stats/back_event/mean'][0], k[0] / lengths[0])
        labels = json.loads((root / 'meta' / 'recovery_labels.json').read_text())
        assert labels['k_rec'] == {str(e): v for e, v in k.items()} and labels['seed'] == 0
        assert label_dataset(root, 20, 40, seed=0) == k, 'idempotent'

        rep = check(root)
        assert rep['episodes_with_recovery_positives'] == 3 and rep['positives_are_final_block']
        assert rep['recovery_episode_types'] == {'intervention': 2, 'failure': 1}
        assert rep['frames_with_action_equal_state'] == f'{sum(lengths)}/{sum(lengths)}'
        assert rep['max_timestamp_jitter_s'] < 1e-9
    with tempfile.TemporaryDirectory() as tmp:
        root = make_recovery_tree(Path(tmp) / 'rec', [50, 45], scalar_column=False)
        k = label_dataset(root, 20, 40, seed=1)
        data = pd.read_parquet(root / 'data' / 'chunk-000' / 'file-000.parquet')
        assert int(np.asarray(data['back_event'].iloc[-1]).reshape(-1)[0]) == 1, 'list column kept'
    print('label_recovery_tail + check_dataset: OK')


def test_trim_carries_recovery_meta():
    try:
        from trim_lerobotv3_episodes import carry_project_meta
    except SystemExit:  # the tool exits when lerobot is missing
        print('trim recovery meta: SKIPPED (lerobot unavailable)')
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = make_recovery_tree(Path(tmp) / 'rec', [60, 50, 45])
        k = label_dataset(root, 20, 40, seed=0)
        out = Path(tmp) / 'trimmed'
        (out / 'meta').mkdir(parents=True)
        carry_project_meta(root, out, drop_first=1)
        labels = json.loads((out / 'meta' / 'recovery_labels.json').read_text())
        assert labels['k_rec'] == {'0': k[1], '1': k[2]} and labels['seed'] == 0, labels
        tags = json.loads((out / 'meta' / 'recovery_episodes.json').read_text())
        assert tags == {'0': {'type': 'failure'}, '1': {'type': 'intervention'}}, tags
    print('trim recovery meta: OK')


if __name__ == '__main__':
    test_label_recovery_tail()
    test_trim_carries_recovery_meta()
