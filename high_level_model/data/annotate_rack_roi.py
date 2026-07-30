"""Offline annotator for the sample-tube rack region-of-interest (ROI).

Draws a single rectangle over the rack in a top-down camera frame and writes it as
``{"roi": [x, y, w, h]}`` JSON. The System-2 planner consumes this ROI to crop the scene image
before sending it to the vision LLM — cropping away clutter (table, arms, background) measurably
improves the tube-sorting perception in :meth:`high_level_model.planning.planner.Planner.perceive_rack`
(passed via ``plan_tube_sorting(..., roi=...)``).

The frame can come from either a standalone image file or a recorded LeRobot dataset frame.

Controls (OpenCV ``selectROI`` window):
    drag    draw the rectangle      ENTER/SPACE   confirm        c   cancel (no ROI written)

Examples::

    # From a saved top-camera image
    python -m high_level_model.data.annotate_rack_roi \
        --image data/top_frame.png --out configs/skill_library/rack_roi_bloodtube.json

    # From a LeRobot dataset frame (episode 0, frame 0, camera cam_top)
    python -m high_level_model.data.annotate_rack_roi \
        --repo_id your/dataset --root data/dagger --episode 0 --frame 0 \
        --camera observation.images.cam_top --out configs/skill_library/rack_roi_bloodtube.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WINDOW = "rack ROI annotator (drag a box, ENTER to confirm, c to cancel)"


def _to_hwc_bgr_uint8(img) -> np.ndarray:
    """LeRobot frame tensor/array -> HWC BGR uint8 for cv2.imshow (mirrors annotate_back_windows)."""
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):      # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 \
            else arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:           # RGB -> BGR
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(arr)


def _load_frame(args) -> np.ndarray:
    """Load a single BGR uint8 frame from --image or a LeRobot dataset frame."""
    if args.image:
        frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        return frame
    if not args.repo_id:
        raise SystemExit("Provide either --image or --repo_id (+ --root/--episode/--frame).")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dset = LeRobotDataset(args.repo_id, root=args.root)
    from high_level_model.data.high_level_dataset import _get_ep_bounds

    ep_start, ep_end = _get_ep_bounds(dset.meta, args.episode)
    global_idx = ep_start + max(0, min(args.frame, ep_end - ep_start - 1))
    sample = dset[global_idx]
    if args.camera not in sample:
        raise KeyError(
            f"Camera '{args.camera}' not in frame keys: {sorted(k for k in sample if 'image' in k)}"
        )
    return _to_hwc_bgr_uint8(sample[args.camera])


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate the sample-tube rack ROI.")
    ap.add_argument("--image", default="", help="Path to a top-camera image file.")
    ap.add_argument("--repo_id", default="", help="LeRobot dataset id (alternative to --image).")
    ap.add_argument("--root", default=None, help="LeRobot dataset root.")
    ap.add_argument("--episode", type=int, default=0, help="Episode index (dataset mode).")
    ap.add_argument("--frame", type=int, default=0, help="Episode-local frame index (dataset mode).")
    ap.add_argument("--camera", default="observation.images.cam_top", help="Camera key (dataset mode).")
    ap.add_argument("--out", required=True, help="Output JSON path for the ROI.")
    args = ap.parse_args()

    frame = _load_frame(args)
    logger.info("Loaded frame %s (HxW=%dx%d). Drag the rack box, ENTER to confirm.",
                args.image or f"{args.repo_id}#{args.episode}:{args.frame}", frame.shape[0], frame.shape[1])

    x, y, w, h = cv2.selectROI(WINDOW, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        logger.warning("Empty ROI; nothing written.")
        return

    roi = [int(x), int(y), int(w), int(h)]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"roi": roi}, f, indent=2)
    logger.info("Wrote ROI %s -> %s", roi, out_path)


if __name__ == "__main__":
    main()
