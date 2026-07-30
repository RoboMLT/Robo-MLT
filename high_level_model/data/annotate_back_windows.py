"""Offline annotator for System-2 ``back`` (precondition-violation) windows.

Plays back an already-recorded LeRobot v3 dataset (e.g. the failure / wrong-init
rollouts captured by ``collect_dagger_dataset.py``) frame-by-frame and lets you mark
contiguous frame windows as ``back = 1`` — for example the span where the gripper is
empty while the subtask is *place*. The exported JSON is consumed verbatim by
:class:`high_level_model.data.completion_gate_dataset.CompletionGateDataset` via its
``back_annotations_path`` argument (episode-local frames, ``annotation_is_local=True``).

JSON schema (episode-local, end-inclusive — matches ``CompletionGateDataset``):

    [{"episode": 3, "frame_start": 120, "frame_end": 180}, ...]

Controls (OpenCV window):
    ./,      next / previous frame          n/p   next / previous episode
    [ / ]    mark window start / end        m     commit the current [start, end] window
    u        undo last committed window      w     write JSON to --out
    q/ESC    write JSON and quit            (trackbar scrubs within the episode)

Example::

    python -m high_level_model.data.annotate_back_windows \
        --repo_id your/dagger_dataset --root data/dagger \
        --out data/dagger/back_annotations.json
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from high_level_model.data.high_level_dataset import _get_ep_bounds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WINDOW = "back-window annotator"


# --------------------------------------------------------------------- helpers
def _to_hwc_bgr_uint8(img) -> np.ndarray:
    """LeRobot frame tensor/array -> HWC BGR uint8 for cv2.imshow."""
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):      # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 \
            else arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:           # RGB -> BGR
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(arr)


def _build_index_to_text(meta, index_key: str) -> Dict[int, str]:
    """Map subtask/task index -> text from meta.{subtasks,tasks} (index=text, col=*_index)."""
    df = meta.tasks if index_key == "task_index" else getattr(meta, "subtasks", None)
    if df is None or index_key not in getattr(df, "columns", []):
        logger.warning("meta has no '%s' map; subtask text will be empty.", index_key)
        return {}
    return {int(idx): str(text) for text, idx in df[index_key].items()}


def _read_index_column(dset, index_key: str) -> Optional[List[int]]:
    """Read the integer (sub)task index column from hf_dataset (no image decode)."""
    hf = getattr(dset, "hf_dataset", None)
    if hf is None or index_key not in hf.column_names:
        return None
    return [int(x) for x in hf[index_key]]


def _segments_for_episode(indices: List[int], ep_start: int, ep_end: int,
                          idx_to_text: Dict[int, str]) -> Dict[int, str]:
    """local_frame -> subtask text, via run-length scan over the index column."""
    out: Dict[int, str] = {}
    i = ep_start
    for idx_val, group in groupby(indices[ep_start:ep_end]):
        j = i + sum(1 for _ in group)
        text = idx_to_text.get(int(idx_val), "")
        for k in range(i, j):
            out[k - ep_start] = text
        i = j
    return out


# --------------------------------------------------------------------- annotator
class BackWindowAnnotator:
    def __init__(self, dset, episodes: List[int], index_key: str,
                 idx_to_text: Dict[int, str], camera: str, out_path: str) -> None:
        self.dset = dset
        self.episodes = episodes
        self.index_key = index_key
        self.idx_to_text = idx_to_text
        self.camera = camera
        self.out_path = out_path
        self._indices = _read_index_column(dset, index_key) or []

        self.ep_pos = 0                         # position into self.episodes
        self.local_f = 0
        self.mark_start: Optional[int] = None
        self.mark_end: Optional[int] = None
        self.windows: Dict[int, List[Tuple[int, int]]] = {}
        self.commit_order: List[Tuple[int, int, int]] = []  # (episode, start, end) for undo

        self._load_episode()

    # --- episode state -----------------------------------------------------
    @property
    def episode(self) -> int:
        return self.episodes[self.ep_pos]

    def _load_episode(self) -> None:
        self.ep_start, self.ep_end = _get_ep_bounds(self.dset.meta, self.episode)
        self.n_frames = self.ep_end - self.ep_start
        self.seg_text = _segments_for_episode(self._indices, self.ep_start, self.ep_end,
                                              self.idx_to_text) if self._indices else {}
        self.local_f = 0
        self.mark_start = self.mark_end = None
        cv2.setTrackbarMax(WINDOW, WINDOW,self.n_frames - 1) if self._trackbar_ready() else None
        self._sync_trackbar()
        logger.info("episode %d | %d frames", self.episode, self.n_frames)

    def _trackbar_ready(self) -> bool:
        return getattr(self, "_tb", False)

    def _sync_trackbar(self) -> None:
        if self._trackbar_ready():
            cv2.setTrackbarPos("frame", WINDOW, self.local_f)

    # --- rendering ---------------------------------------------------------
    def _frame_image(self) -> np.ndarray:
        data = self.dset[self.ep_start + self.local_f]
        return _to_hwc_bgr_uint8(data[self.camera])

    def _overlay(self, img: np.ndarray) -> np.ndarray:
        canvas = img.copy()
        h = canvas.shape[0]
        pad = np.full((96, canvas.shape[1], 3), (25, 25, 25), dtype=np.uint8)
        canvas = np.vstack([canvas, pad])
        skill = self.seg_text.get(self.local_f, "")
        mark = f"[{self.mark_start}, {self.mark_end}]"
        n_win = sum(len(v) for v in self.windows.values())

        def put(text, y, color=(225, 225, 225)):
            cv2.putText(canvas, text, (8, h + y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        put(f"ep {self.episode}  frame {self.local_f}/{self.n_frames - 1}  subtask: {skill[:40]}", 22)
        # is current frame inside a committed back window?
        in_back = any(s <= self.local_f <= e for s, e in self.windows.get(self.episode, []))
        put(f"mark {mark}   committed windows(this ep): {self.windows.get(self.episode, [])}",
            44, (40, 200, 40) if not in_back else (40, 140, 230))
        put(f"total windows: {n_win}   [ start  ] end  m commit  u undo  w write  q quit", 66)
        if in_back:
            put("BACK=1 here", 88, (40, 140, 230))
        return canvas

    # --- actions -----------------------------------------------------------
    def set_frame(self, f: int) -> None:
        self.local_f = max(0, min(self.n_frames - 1, f))

    def next_frame(self, d: int) -> None:
        self.set_frame(self.local_f + d)
        self._sync_trackbar()

    def switch_episode(self, d: int) -> None:
        self.ep_pos = (self.ep_pos + d) % len(self.episodes)
        self._load_episode()

    def commit(self) -> None:
        if self.mark_start is None or self.mark_end is None:
            logger.warning("set both '[' start and ']' end before committing")
            return
        s, e = sorted((self.mark_start, self.mark_end))
        self.windows.setdefault(self.episode, []).append((s, e))
        self.commit_order.append((self.episode, s, e))
        logger.info("committed back window ep %d [%d, %d]", self.episode, s, e)
        self.mark_start = self.mark_end = None

    def undo(self) -> None:
        if not self.commit_order:
            return
        ep, s, e = self.commit_order.pop()
        self.windows.get(ep, []).remove((s, e))
        if not self.windows.get(ep):
            self.windows.pop(ep, None)
        logger.info("undid back window ep %d [%d, %d]", ep, s, e)

    def write(self) -> None:
        entries = [
            {"episode": ep, "frame_start": s, "frame_end": e}
            for ep, s, e in sorted(self.commit_order)
        ]
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.out_path).write_text(json.dumps(entries, indent=2), encoding="utf-8")
        n_frames = sum(e - s + 1 for _, s, e in self.commit_order)
        logger.info("wrote %d windows (%d back-positive frames) -> %s",
                    len(entries), n_frames, self.out_path)

    # --- main loop ---------------------------------------------------------
    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("frame", WINDOW, 0, max(1, self.n_frames - 1),
                           lambda v: self.set_frame(v))
        self._tb = True
        cv2.setTrackbarMax(WINDOW,winname=WINDOW, maxval=self.n_frames - 1)

        while True:
            cv2.imshow(WINDOW, self._overlay(self._frame_image()))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):           # q / ESC
                self.write()
                break
            elif key == ord("."):
                self.next_frame(1)
            elif key == ord(","):
                self.next_frame(-1)
            elif key == ord("n"):
                self.switch_episode(1)
            elif key == ord("p"):
                self.switch_episode(-1)
            elif key == ord("["):
                self.mark_start = self.local_f
            elif key == ord("]"):
                self.mark_end = self.local_f
            elif key == ord("m"):
                self.commit()
            elif key == ord("u"):
                self.undo()
            elif key == ord("w"):
                self.write()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------- main
def main() -> None:
    args = build_arg_parser().parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # No image_transforms: keep frames as raw uint8 for display.
    dset = LeRobotDataset(repo_id=args.repo_id, root=args.root)

    index_key = "task_index" if args.use_command_in_meta else "subtask_index"
    idx_to_text = _build_index_to_text(dset.meta, index_key)

    episodes = args.episodes if args.episodes else list(range(dset.meta.total_episodes))

    camera = args.camera
    if camera is None:
        cams = [k for k in dset.features if k.startswith("observation.images.")]
        if not cams:
            raise RuntimeError("No 'observation.images.*' feature found; pass --camera explicitly.")
        camera = cams[0]
        logger.info("Using camera '%s' (override with --camera).", camera)

    BackWindowAnnotator(dset, episodes, index_key, idx_to_text, camera, args.out).run()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Annotate System-2 back (precondition-violation) windows.")
    p.add_argument("--repo_id", required=True, help="LeRobot dataset repo id")
    p.add_argument("--root", default=None, help="Local dataset root directory")
    p.add_argument("--out", required=True, help="Output JSON path (episode-local back windows)")
    p.add_argument("--episodes", type=int, nargs="*", default=None,
                   help="Subset of episode indices to review (default: all)")
    p.add_argument("--camera", default=None,
                   help="Image feature key to display (default: first observation.images.*)")
    p.add_argument("--use_command_in_meta", action="store_true", default=False,
                   help="Show task-level text instead of subtask text")
    return p


if __name__ == "__main__":
    main()
