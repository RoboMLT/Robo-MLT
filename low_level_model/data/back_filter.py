"""Back-event filtering for System-1 (VLA) training.

The shared LeRobot v3 dataset may contain frames flagged ``back_event=1`` — states deliberately driven
into an erroneous, unrecoverable pose to supervise System-2's back head. System 1 must *not* imitate
these bad actions, so any training sample whose action-chunk window overlaps a back-event frame is
excluded here.

A sample at global frame index ``i`` supervises the action chunk ``[i, i+chunk_size)``; with the
forward delay augmentation (``DelayAugmentedDataset`` / ``SharedObservationDataset``) the window can
shift forward by up to ``max_delay_steps``. We therefore exclude ``i`` whenever any frame in
``[i, i + chunk_size + max_delay_steps)`` (clamped to the episode) is a back-event frame. lerobot
clamps/pads action chunks within the episode and never crosses episode boundaries, so the per-episode
scan below is exact.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["build_back_filtered_indices"]


def _coerce_int(x) -> int:
    """Robustly coerce a hf_dataset cell (scalar or shape-(1,) array/tensor) to int."""
    try:
        return int(x)
    except (TypeError, ValueError):
        return int(np.asarray(x).reshape(-1)[0])


def build_back_filtered_indices(
    dataset,
    chunk_size: int,
    max_delay_steps: int = 0,
    back_event_column: str = "back_event",
) -> Optional[List[int]]:
    """Return the global frame indices whose action-chunk window contains no back-event frame.

    Returns ``None`` when there is nothing to filter — the ``back_event`` column is absent or has no
    positives — so the caller keeps the default behavior (sample every frame).

    Args:
        dataset: a ``LeRobotDataset`` (or subclass) with ``hf_dataset`` and ``meta.episodes``.
        chunk_size: System-1 action-chunk length.
        max_delay_steps: max forward offset of the delay augmentation (0 if disabled).
        back_event_column: name of the per-frame 0/1 flag column.
    """
    hf = getattr(dataset, "hf_dataset", None)
    if hf is None or back_event_column not in getattr(hf, "column_names", []):
        return None
    back = [_coerce_int(x) for x in hf[back_event_column]]
    if not any(back):
        return None

    window = int(chunk_size) + max(0, int(max_delay_steps))
    allowed: List[int] = []
    for ep_idx in range(dataset.num_episodes):
        ep = dataset.meta.episodes[ep_idx]
        ep_start = int(ep["dataset_from_index"])
        ep_end = int(ep["dataset_to_index"])
        # next_err[k-ep_start] = index of the next back-event frame at or after k within the episode,
        # or +inf if none. A start index k is allowed iff the next back-event is >= k + window. The
        # +inf sentinel (NOT ep_end) is essential: the action chunk is clamped/padded at the episode
        # boundary, so trailing in-episode frames with no later error — and entire clean episodes —
        # must pass rather than be dropped.
        INF = 1 << 62
        nxt = INF
        next_err = [INF] * (ep_end - ep_start)
        for k in range(ep_end - 1, ep_start - 1, -1):
            if back[k]:
                nxt = k
            next_err[k - ep_start] = nxt
        for k in range(ep_start, ep_end):
            if next_err[k - ep_start] - k >= window:
                allowed.append(k)

    n_total = len(back)
    logger.info(
        "Back-event filter: kept %d/%d frames for System-1 training (window=%d, excluded %d).",
        len(allowed), n_total, window, n_total - len(allowed),
    )
    return allowed
