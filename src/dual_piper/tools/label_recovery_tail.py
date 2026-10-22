#!/usr/bin/env python3
"""Label the recovery-head positives of a recovery dataset (Robo-MLT paper, Sec. IV-A-3).

Every episode of a recovery dataset ends with the event that made the active skill impossible to
complete: an operator removed the sample from the gripper, or the teleoperated grasp failed (e.g. a
dropped tube) and recording was stopped right away. For each episode this tool draws
K_rec ~ U[k_min, k_max] (inclusive, independently per episode, from ``--seed``) and sets
``back_event = 1`` on the final K_rec frames and 0 elsewhere, so the recovery head is not tuned to
one fixed offset from the terminating event.

It rewrites the data parquet files in place (atomically, file by file), updates the
``back_event`` statistics in ``meta/episodes`` and ``meta/stats.json``, and records the K_rec of
every episode in ``meta/recovery_labels.json``. Re-running with the same seed is idempotent.

Usage::

    python tools/label_recovery_tail.py --root <data_root>/blood_gas_analysis_recovery \\
        --k-min 20 --k-max 40 --seed 0

Only pandas / pyarrow / numpy are needed (no ROS, no LeRobot).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

COLUMN = 'back_event'
LABELS_FILE = 'recovery_labels.json'
TAGS_FILE = 'recovery_episodes.json'
_QUANTILES = {'q01': 0.01, 'q10': 0.10, 'q50': 0.50, 'q90': 0.90, 'q99': 0.99}


def draw_k_rec(episodes, k_min, k_max, seed):
    """Deterministic K_rec per episode index (independent of processing order)."""
    return {int(ep): int(np.random.default_rng([seed, int(ep)]).integers(k_min, k_max + 1))
            for ep in episodes}


def _stat_values(labels: np.ndarray) -> dict:
    x = labels.astype(np.float64)
    out = {'min': x.min(), 'max': x.max(), 'mean': x.mean(), 'std': x.std(), 'count': len(x)}
    out.update({k: np.quantile(x, q) for k, q in _QUANTILES.items()})
    return out


def _write_atomic_table(table: pa.Table, path: Path) -> None:
    tmp = path.with_suffix('.parquet.tmp')
    pq.write_table(table, tmp, compression='snappy', use_dictionary=True)
    os.replace(tmp, path)


def label_dataset(root: Path, k_min: int, k_max: int, seed: int, dry_run: bool = False) -> dict:
    """Apply the tail labels to every episode under ``root``; returns {episode: K_rec}."""
    data_files = sorted(root.glob('data/**/*.parquet'))
    if not data_files:
        raise FileNotFoundError(f'No data parquet files under {root / "data"}')
    if COLUMN not in pq.read_schema(data_files[0]).names:
        raise ValueError(f"{root} has no '{COLUMN}' column; record it with use_back_event: true")

    episodes = sorted({int(e) for f in data_files
                       for e in pd.read_parquet(f, columns=['episode_index'])['episode_index']})
    k_rec = draw_k_rec(episodes, k_min, k_max, seed)
    lengths = {}

    for f in data_files:
        table = pq.read_table(f)
        df = table.select(['episode_index', 'frame_index']).to_pandas()
        labels = np.zeros(len(df), dtype=np.int64)
        for ep, idx in df.groupby('episode_index').indices.items():
            frames = df['frame_index'].to_numpy()[idx]
            n = int(frames.max()) + 1
            lengths[int(ep)] = n
            labels[idx] = (frames >= n - k_rec[int(ep)]).astype(np.int64)
        field = table.schema.field(COLUMN)
        if pa.types.is_list(field.type) or pa.types.is_fixed_size_list(field.type):
            col = pa.array([[int(v)] for v in labels], type=field.type)
        else:
            col = pa.array(labels, type=field.type)
        if not dry_run:
            _write_atomic_table(table.set_column(table.schema.get_field_index(COLUMN), field, col), f)

    short = {ep: n for ep, n in lengths.items() if n < k_rec[ep]}
    if short:
        print(f'[warn] {len(short)} episode(s) shorter than their K_rec are labelled entirely: '
              f'{sorted(short)[:10]}', file=sys.stderr)
    if not dry_run:
        _update_stats(root, k_rec, lengths)
        (root / 'meta' / LABELS_FILE).write_text(json.dumps(
            {'k_min': k_min, 'k_max': k_max, 'seed': seed,
             'k_rec': {str(ep): k for ep, k in sorted(k_rec.items())}}, indent=2), encoding='utf-8')
    return k_rec


def _update_stats(root: Path, k_rec: dict, lengths: dict) -> None:
    """Rewrite the back_event entries of meta/episodes/* and meta/stats.json."""
    all_labels = []
    for f in sorted(root.glob('meta/episodes/**/*.parquet')):
        df = pd.read_parquet(f)
        cols = [c for c in df.columns if c.startswith(f'stats/{COLUMN}/')]
        if not cols:
            continue
        per_episode = []
        for ep in df['episode_index'].to_numpy():
            n = lengths[int(ep)]
            labels = (np.arange(n) >= n - k_rec[int(ep)]).astype(np.int64)
            all_labels.append(labels)
            per_episode.append(_stat_values(labels))
        for c in cols:
            stat = c.rsplit('/', 1)[1]
            if stat in per_episode[0]:
                dtype = np.int64 if stat == 'count' else np.float64
                df[c] = [np.array([v[stat]], dtype=dtype) for v in per_episode]
        tmp = f.with_suffix('.parquet.tmp')
        df.to_parquet(tmp, index=False)
        os.replace(tmp, f)

    stats_path = root / 'meta' / 'stats.json'
    if stats_path.exists() and all_labels:
        stats = json.loads(stats_path.read_text(encoding='utf-8'))
        values = _stat_values(np.concatenate(all_labels))
        stats[COLUMN] = {k: [int(v)] if k == 'count' else [float(v)] for k, v in values.items()}
        stats_path.write_text(json.dumps(stats, indent=4), encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, type=Path, help='Recovery dataset root (LeRobot v3)')
    ap.add_argument('--k-min', type=int, default=20)
    ap.add_argument('--k-max', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true', help='Report K_rec without writing anything')
    args = ap.parse_args()
    if not 0 < args.k_min <= args.k_max:
        sys.exit('--k-min must be positive and not larger than --k-max')
    if not (args.root / 'meta' / TAGS_FILE).exists():
        print(f'[warn] {args.root} has no meta/{TAGS_FILE}: is this a recovery dataset '
              f'(collection.mode: recovery)?', file=sys.stderr)

    k_rec = label_dataset(args.root, args.k_min, args.k_max, args.seed, args.dry_run)
    ks = np.array(list(k_rec.values()))
    print(f'{"Would label" if args.dry_run else "Labelled"} {len(ks)} episode(s): '
          f'K_rec in [{ks.min()}, {ks.max()}], mean {ks.mean():.1f}')


if __name__ == '__main__':
    main()
