"""Dataset for training the System 2 completion gate (completion + back heads, Eq. 3-5).

Reuses :class:`high_level_dataset.HighLevelSequenceDataset` for history-frame sampling, and adds:
    - **current skill text**: the subtask label of the current frame (NOT the future-offset frame).
    - **back target** (beta_hat): 1 inside an annotated DAgger correction window, else 0. If no
      annotations are provided, all 0 (the back head simply won't learn until correction data exists).
    - **completion target** (gamma_hat): 1 on the last ``completion_window`` frames of a segment that
      genuinely ends with the subtask *done*; else 0. A segment ending at a real subtask transition
      (``ends_at_transition``) always qualifies. The episode's *final* run qualifies only when
      ``completion_at_episode_end=True``, which asserts that captures run to completion rather than
      being cut off mid-skill. Leave it off for truncated captures, where the last frames would teach
      the head that "unfinished" looks done. Note that with it off, single-subtask episodes contain
      **no** completion positives at all.

Segment boundaries are precomputed once from the subtask column (no image decoding); ``seg_id``
(the segment's start frame) groups frames from the same subtask occurrence for
:class:`SegmentBatchSampler`, which balances positives per batch for the weighted BCE losses.
"""

import json
import logging
import random
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import torch
from torch.utils.data import DataLoader

from high_level_model.data.high_level_dataset import HighLevelSequenceDataset, _get_ep_bounds

logger = logging.getLogger(__name__)

__all__ = ["CompletionGateDataset", "Segment", "completion_gate_collate_fn"]


class Segment(NamedTuple):
    """One contiguous run of an identical subtask label inside a single episode.

    ``start``/``end`` are *global* (dataset-wide) frame indices, half-open ``[start, end)``.
    ``seg_id`` (== ``start``) is unique per subtask *occurrence*, so two cycles of the same skill
    in one episode get different ids — the trainer uses it to form within-segment ranking pairs.

    ``ends_at_transition`` distinguishes a segment that ends because the *next* subtask begins
    (a real completion boundary → the last frame is genuinely ``progress == 1``) from one that
    ends at episode end. The latter can be a deliberately-partial capture (this collection style
    initialises from different moments and may store only a subset of subtasks) **or** a truncated
    trajectory; the flag lets downstream code treat the ``progress == 1`` anchor accordingly.
    """

    start: int
    end: int
    skill_text: str
    subtask_index: int
    ends_at_transition: bool

    @property
    def length(self) -> int:
        return self.end - self.start


class CompletionGateDataset(HighLevelSequenceDataset):
    """Yields (image_sequence, current_skill_text, back, complete, seg_id)."""

    def __init__(
        self,
        *args,
        back_annotations_path: Optional[str] = None,
        annotation_is_local: bool = True,
        min_segment_len: int = 0,
        completion_window: int = 5,
        completion_at_episode_end: bool = False,
        back_from_column: bool = True,
        back_event_column: str = "back_event",
        random_skip_range: int = 0,
        randomize_skip: bool = False,
        **kwargs,
    ) -> None:
        # Random inter-frame stride (train-time augmentation): instead of a fixed cadence, draw the
        # history stride per sample from U(1, Δ) so the gate sees many time-scales and overfits the
        # fixed cadence less. The deterministic stride (used for val and when randomization is off) is
        # the configured ``history_skip_frame``; we still reserve head-room for the *largest* possible
        # stride when building sample anchors, so a big random draw never reaches before the episode.
        self.random_skip_range = int(random_skip_range)
        self.randomize_skip = bool(randomize_skip)
        self._det_skip = int(kwargs.get("history_skip_frame", 10))
        if self.random_skip_range > self._det_skip:
            kwargs["history_skip_frame"] = self.random_skip_range   # reserve max-span head-room
        super().__init__(*args, **kwargs)
        # The last ``completion_window`` frames of each *truly-completed* segment are completion
        # positives; built in _precompute_segments. ``completion_at_episode_end`` additionally counts
        # the episode's final run as completed (see the module docstring) — required for
        # single-subtask episodes to contribute any positives at all.
        self.completion_window = int(completion_window)
        self.completion_at_episode_end = bool(completion_at_episode_end)
        # Runs shorter than this many frames are treated as label-boundary noise: they are dropped
        # from the segment map AND their samples are pruned. 0/1 disables.
        self.min_segment_len = int(min_segment_len)
        # Decode-free label source: lerobot 3.0 stores the (sub)task as an integer column in
        # hf_dataset and the text in meta.{subtasks,tasks}. Reading those never touches video.
        self.index_key = "task_index" if self.use_command_in_meta else "subtask_index"
        self.index_to_text: Dict[int, str] = self._build_index_to_text()
        self.segments: List[Segment] = []
        self.frame_to_seg: Dict[int, Segment] = {}
        self.completion_frames: set = set()
        self._precompute_segments()
        self.back_frames: set = set()
        # Prefer the inline ``back_event`` column written at collection time (decode-free); fall back to
        # the legacy post-hoc JSON annotation only when the column is absent (backward compatibility).
        loaded = False
        if back_from_column:
            loaded = self._load_back_from_column(back_event_column)
        if not loaded:
            self._load_back_annotations(back_annotations_path, annotation_is_local)

    # ------------------------------------------------------------------ segments
    def _episodes_in_use(self) -> List[int]:
        return sorted({ep for ep, _ in self.samples})

    def _build_index_to_text(self) -> Dict[int, str]:
        """Map (sub)task_index -> text from meta.{tasks,subtasks} (index=text, col=*_index)."""
        meta = self.dset.meta
        df = meta.tasks if self.use_command_in_meta else getattr(meta, "subtasks", None)
        if df is None or self.index_key not in getattr(df, "columns", []):
            logger.warning("meta has no '%s' map; skill_text will be empty.", self.index_key)
            return {}
        return {int(idx): str(text) for text, idx in df[self.index_key].items()}

    def _read_index_column(self) -> Optional[List[int]]:
        """Read the integer (sub)task index column from hf_dataset (no image/video decode)."""
        hf = getattr(self.dset, "hf_dataset", None)
        if hf is None or self.index_key not in hf.column_names:
            return None
        logger.info("Reading index column '%s' from hf_dataset (decode-free).", self.index_key)
        return [int(x) for x in hf[self.index_key]]

    def _precompute_segments(self) -> None:
        indices = self._read_index_column()
        if indices is None:
            raise RuntimeError(
                f"Column '{self.index_key}' not found in hf_dataset; refusing to fall back to "
                "per-frame decoding. Check the dataset's metadata columns."
            )

        n_dropped_short = 0
        for episode_idx in self._episodes_in_use():
            ep_start, ep_end = _get_ep_bounds(self.dset.meta, episode_idx)
            # Run-length scan over the integer label column: each contiguous run is one segment.
            # Episodes here may carry only a *subset* of the task's subtasks (deliberately-partial
            # captures), and a subtask may even recur (e.g. a retry); the run-length scan handles
            # both — every contiguous run becomes its own segment with progress renormalised to
            # [0, 1] independently, so a stand-alone subtask and an in-sequence one are comparable.
            runs = [(idx_val, sum(1 for _ in group))
                    for idx_val, group in groupby(indices[ep_start:ep_end])]
            i = ep_start
            for run_pos, (idx_val, n) in enumerate(runs):
                j = i + n
                # The last run in the episode ends at episode end, not at a real subtask boundary —
                # so its final frame is only a true progress==1 anchor when the capture is complete.
                ends_at_transition = run_pos < len(runs) - 1
                if n < self.min_segment_len:
                    n_dropped_short += n
                    i = j
                    continue
                seg = Segment(
                    start=i, end=j, skill_text=self.index_to_text.get(int(idx_val), ""),
                    subtask_index=int(idx_val), ends_at_transition=ends_at_transition,
                )
                self.segments.append(seg)
                for k in range(i, j):
                    self.frame_to_seg[k] = seg
                # completion positives: last `completion_window` frames of a segment that genuinely
                # ends with the subtask done. A real subtask transition is always such an anchor; the
                # episode's final run counts only when the captures are known to run to completion
                # (``completion_at_episode_end``). Without that flag a single-subtask episode yields
                # no positives at all, and the head learns "this skill never finishes".
                is_completion = ends_at_transition or self.completion_at_episode_end
                if is_completion and self.completion_window > 0:
                    for k in range(max(i, j - self.completion_window), j):
                        self.completion_frames.add(k)
                i = j

        # Prune samples whose current frame is no longer covered (its run was dropped as too short),
        # so __getitem__ never falls back to a bogus progress==0 / empty-skill target.
        if n_dropped_short:
            kept = [(ep, f) for (ep, f) in self.samples if f in self.frame_to_seg]
            logger.info("min_segment_len=%d dropped %d frames in short runs; pruned %d/%d samples.",
                        self.min_segment_len, n_dropped_short,
                        len(self.samples) - len(kept), len(self.samples))
            self.samples = kept

        self._log_segment_inventory()

    def _log_segment_inventory(self) -> None:
        """Surface how the (possibly-partial) episodes were segmented, so the deliberately-partial
        collection style is verifiable at a glance: per-subtask counts split by how each segment
        ends. ``ends-at-transition`` = the next subtask begins in-episode (a real boundary);
        ``ends-at-episode`` = the run reaches episode end — normal for a task's *final* subtask or a
        deliberately-partial capture, but worth a look if a *non-terminal* subtask shows up here."""
        at_transition: Counter = Counter()
        at_episode_end: Counter = Counter()
        for seg in self.segments:
            (at_transition if seg.ends_at_transition else at_episode_end)[seg.skill_text or seg.subtask_index] += 1
        logger.info("Precomputed %d subtask segments over %d frames.",
                    len(self.segments), len(self.frame_to_seg))
        for key in sorted(set(at_transition) | set(at_episode_end), key=str):
            logger.info("  subtask %-48s | ends-at-transition=%d  ends-at-episode=%d",
                        f"'{key}'", at_transition.get(key, 0), at_episode_end.get(key, 0))

        # Completion-label census. A silently-empty positive set trains the head to output 0 forever
        # while still showing a *low* BCE (the labels really are all-negative), so surface it loudly.
        n_frames = max(1, len(self.frame_to_seg))
        n_pos = len(self.completion_frames)
        logger.info("Completion positives: %d / %d frames (%.2f%%) | window=%d "
                    "completion_at_episode_end=%s",
                    n_pos, len(self.frame_to_seg), 100.0 * n_pos / n_frames,
                    self.completion_window, self.completion_at_episode_end)
        if n_pos == 0 and self.completion_window > 0:
            logger.warning(
                "No completion positives in this split — the completion head cannot learn and will "
                "predict ~0 everywhere. Every segment ends at episode end (e.g. single-subtask "
                "episodes). Set dataset.completion_at_episode_end=true if these captures do run to "
                "completion.")

    # ------------------------------------------------------------------ back labels
    def _read_named_int_column(self, col_name: str) -> Optional[List[int]]:
        """Read an integer column (e.g. ``back_event``) from hf_dataset, decode-free.

        Returns a list indexed by *global* frame index, or None if the column is absent. Mirrors
        :meth:`_read_index_column`; the ``reshape(-1)[0]`` fallback handles shape-(1,) features that
        are stored as length-1 arrays rather than plain scalars.
        """
        hf = getattr(self.dset, "hf_dataset", None)
        if hf is None or col_name not in getattr(hf, "column_names", []):
            return None

        def _to_int(x) -> int:
            try:
                return int(x)
            except (TypeError, ValueError):
                import numpy as _np
                return int(_np.asarray(x).reshape(-1)[0])

        logger.info("Reading back column '%s' from hf_dataset (decode-free).", col_name)
        return [_to_int(x) for x in hf[col_name]]

    def _load_back_from_column(self, col_name: str) -> bool:
        """Populate ``self.back_frames`` from the inline ``back_event`` column.

        Returns True if the column was found (even if it has no positives), so the caller can skip
        the legacy JSON fallback. Each global frame index with a non-zero flag is a back-positive.
        """
        values = self._read_named_int_column(col_name)
        if values is None:
            return False
        for global_idx, v in enumerate(values):
            if v:
                self.back_frames.add(global_idx)
        if not self.back_frames:
            logger.warning(
                "Column '%s' exists but has ZERO positives across the dataset — the back head has no "
                "supervision and will learn to output 0 everywhere. Recovery then rests entirely on "
                "the pipeline's stall detector. Collect/annotate correction data before trusting "
                "back_prob, and keep loss.lambda_back at 0 until then.", col_name)
        else:
            logger.info("Loaded %d back-positive frames from column '%s'.",
                        len(self.back_frames), col_name)
        return True

    def _load_back_annotations(self, path: Optional[str], is_local: bool) -> None:
        if not path:
            logger.warning(
                "No back-annotation file provided; all back targets = 0 "
                "(back head will not learn until DAgger correction data exists)."
            )
            return
        p = Path(path)
        if not p.exists():
            logger.warning("Back-annotation file %s not found; all back targets = 0.", p)
            return
        entries = json.loads(p.read_text(encoding="utf-8"))
        for e in entries:
            ep = int(e["episode"])
            ep_start, _ = _get_ep_bounds(self.dset.meta, ep)
            f0, f1 = int(e["frame_start"]), int(e["frame_end"])
            if is_local:
                f0, f1 = ep_start + f0, ep_start + f1
            for f in range(f0, f1 + 1):
                self.back_frames.add(f)
        logger.info("Loaded %d back-positive frames from %s.", len(self.back_frames), p)

    # ------------------------------------------------------------------ item
    def draw_stride(self) -> int:
        """Inter-frame stride for one sample: random (train aug) or the fixed deterministic stride
        (val / when disabled). ``__init__`` reserved head-room for the *largest* possible stride, so
        the oldest selected frame never reaches before the current episode's first frame."""
        if self.randomize_skip and self.random_skip_range > 0:
            return random.randint(1, self.random_skip_range)
        return self._det_skip

    def history_indices(self, curr_frame_abs: int, stride: int) -> List[int]:
        """Global frame indices of the history window ending at ``curr_frame_abs`` (oldest first)."""
        return [curr_frame_abs - i * stride for i in range(self.history_len - 1, -1, -1)]

    def labels_for(self, index: int):
        """Stride-independent targets for sample ``index``: ``(skill_text, back, complete, seg_id)``.

        Split out of :meth:`__getitem__` so the frame-level feature cache can reuse the exact same
        label construction without decoding any images.
        """
        _, curr_frame_abs = self.samples[index]
        # current skill text from the current frame's segment. After sample pruning every current
        # frame is covered; the fallback only guards against an unexpected miss.
        seg = self.frame_to_seg.get(
            curr_frame_abs, Segment(curr_frame_abs, curr_frame_abs + 1, "", -1, False)
        )
        back = 1.0 if curr_frame_abs in self.back_frames else 0.0
        complete = 1.0 if curr_frame_abs in self.completion_frames else 0.0
        # seg_id = global frame index of the segment start: unique per subtask occurrence (so two
        # cycles of the same skill get different ids). Lets SegmentBatchSampler co-locate frames
        # from the same segment in a batch.
        return seg.skill_text, back, complete, int(seg.start)

    def __getitem__(self, index):
        _, curr_frame_abs = self.samples[index]
        image_sequence = []
        for frame_idx in self.history_indices(curr_frame_abs, self.draw_stride()):
            frame_data = self.dset[frame_idx]
            cams = [frame_data[cam] for cam in self.camera_names]
            image_sequence.append(torch.stack(cams, dim=0))
        image_sequence = torch.stack(image_sequence, dim=0)

        skill_text, back, complete, seg_id = self.labels_for(index)
        return image_sequence, skill_text, back, complete, seg_id


def completion_gate_collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    skill_texts = [b[1] for b in batch]
    back = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    complete = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    seg_ids = torch.tensor([b[4] for b in batch], dtype=torch.long)
    return images, skill_texts, back, complete, seg_ids


class SegmentBatchSampler(torch.utils.data.Sampler):
    """Yield batches that contain several frames from each of a few segments.

    Plain shuffling rarely co-locates enough frames from the same (rare, positive) segment in one
    batch, which starves the weighted back/completion BCE losses of positives. Each batch picks
    ``segs_per_batch`` segments and ``frames_per_seg`` random frames from each → batch_size =
    segs_per_batch * frames_per_seg.
    """

    def __init__(self, dataset: "CompletionGateDataset", segs_per_batch: int = 4,
                 frames_per_seg: int = 4, shuffle: bool = True, drop_last: bool = False) -> None:
        self.segs_per_batch = segs_per_batch
        self.frames_per_seg = frames_per_seg
        self.shuffle = shuffle
        self.drop_last = drop_last
        # group dataset sample positions by segment id (== segment start frame)
        self.seg_to_positions: Dict[int, List[int]] = {}
        for pos, (_, frame_abs) in enumerate(dataset.samples):
            seg = dataset.frame_to_seg.get(frame_abs)
            if seg is None:  # frame in a run dropped by min_segment_len — skip
                continue
            self.seg_to_positions.setdefault(int(seg.start), []).append(pos)
        self.seg_ids = list(self.seg_to_positions.keys())
        self._num_batches = max(1, len(dataset.samples) // (segs_per_batch * frames_per_seg))

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        import random
        seg_order = list(self.seg_ids)
        if self.shuffle:
            random.shuffle(seg_order)
        for b in range(self._num_batches):
            chosen = random.sample(self.seg_ids, min(self.segs_per_batch, len(self.seg_ids))) \
                if self.shuffle else seg_order[b * self.segs_per_batch:(b + 1) * self.segs_per_batch]
            batch: List[int] = []
            for sid in chosen:
                pool = self.seg_to_positions[sid]
                k = min(self.frames_per_seg, len(pool))
                batch.extend(random.sample(pool, k) if self.shuffle else pool[:k])
            if batch:
                yield batch


def make_completion_gate_loader(dataset: CompletionGateDataset, batch_size: int, shuffle: bool,
                                num_workers: int = 4, batch_sampler=None) -> DataLoader:
    kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=completion_gate_collate_fn,
    )
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
