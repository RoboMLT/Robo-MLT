"""Live operator view for data collection.

An OpenCV window that tiles the robot's camera feeds side-by-side with an
overlay panel showing task / subtask, recording state, collection mode (nominal /
recovery), FPS, and dataset counters.

Threading model (important)
---------------------------
OpenCV's HighGUI (Qt/GTK backend) only reliably maps and repaints a window
from the **main thread** — especially under Anaconda, where torch / lerobot /
pynput pull in their own Qt/X11 and a background GUI thread silently fails to
show the window.  So this class is *main-thread driven*:

  * :meth:`start` (called on the main thread) imports cv2, checks the display,
    and creates the window.
  * :meth:`update` is called from the collection/background thread and only
    stashes the latest (images, status) under a lock — it never touches cv2.
  * :meth:`render_once` must be called repeatedly from the **main thread**
    (e.g. the node's spin loop); it draws the stashed data and pumps the GUI
    event loop. It returns False once the user presses 'q'/ESC in the window.

cv2 is imported lazily inside :meth:`start`; if it (or the display) is
unavailable the view degrades to a no-op (logged once) so collection is never
blocked by a missing display dependency.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["StatusView"]


def _is_image(val) -> bool:
    shape = getattr(val, "shape", None)
    if shape is None:
        return False
    nd = len(shape)
    return nd == 2 or (nd == 3 and shape[2] in (1, 3, 4))


class StatusView:
    """Threaded OpenCV window showing camera feeds + collection status overlay."""

    def __init__(
        self,
        camera_names: Optional[list] = None,
        bgr: bool = False,
        max_width: int = 1280,
        window_name: str = "DualPiper Collect",
        fps: float = 15.0,
        panel_height: int = 300,
    ) -> None:
        # camera_names restricts/orders which image keys are shown; None = auto.
        self.camera_names = camera_names
        # bgr=True: incoming images are RGB and need RGB→BGR conversion for cv2.
        # bgr=False: images are already BGR (default for ROS/OpenCV pipelines).
        self.bgr = bgr
        self.max_width = max_width
        self.window_name = window_name
        self.frame_interval = 1.0 / max(1e-3, fps)
        self.panel_height = panel_height

        self._lock = threading.Lock()
        self._latest_images: Optional[dict] = None
        self._latest_status: Optional[dict] = None
        self._enabled = False
        self._window_created = False
        self._render_errors = 0
        self._cv2 = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> "StatusView":
        """Set up cv2 + the window. MUST be called from the main thread."""
        try:
            import cv2  # noqa: PLC0415
        except Exception as exc:
            logger.warning("StatusView disabled: OpenCV unavailable (%s).", exc)
            return self

        # Require a display; otherwise cv2 GUI calls crash the GUI backend.
        import os
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            logger.warning(
                "StatusView disabled: no DISPLAY/WAYLAND_DISPLAY set. "
                "Use a graphical session (or `ssh -X`) and set DISPLAY."
            )
            return self

        self._cv2 = cv2
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            # Pump the event loop a few times so the window actually maps.
            cv2.waitKey(1)
        except Exception as exc:
            logger.warning("StatusView disabled: cannot create window '%s' (%s).",
                           self.window_name, exc)
            self._cv2 = None
            return self

        self._window_created = True
        self._enabled = True
        logger.info("StatusView started: window='%s' (render driven from main thread)",
                    self.window_name)
        return self

    @property
    def enabled(self) -> bool:
        return self._enabled

    def update(self, images: Optional[dict], status: Optional[dict]) -> None:
        """Stash the latest images + status dict. Non-blocking; safe to call from any thread."""
        if not self._enabled:
            return
        with self._lock:
            self._latest_images = images
            self._latest_status = status

    def render_once(self) -> bool:
        """Draw the stashed frame + status and pump the GUI event loop.

        MUST be called from the **main thread** (repeatedly, e.g. from the node's
        spin loop). Returns False once the user presses 'q'/ESC in the window so
        the caller can shut down; True otherwise (including when disabled).
        """
        if not self._enabled:
            return True
        cv2 = self._cv2
        with self._lock:
            images = self._latest_images
            status = dict(self._latest_status) if self._latest_status else {}
        try:
            canvas = self._render(images, status)
            if canvas is not None:
                cv2.imshow(self.window_name, canvas)
            self._render_errors = 0
        except Exception as exc:
            self._render_errors += 1
            if self._render_errors <= 3:
                logger.warning("StatusView render error: %s", exc)
        try:
            key = cv2.waitKey(1) & 0xFF
        except Exception:
            key = 0
        if key in (ord("q"), 27):
            return False
        return True

    def stop(self) -> None:
        if not self._enabled and not self._window_created:
            return
        self._enabled = False
        try:
            if self._cv2 is not None:
                self._cv2.destroyWindow(self.window_name)
                self._cv2.waitKey(1)
        except Exception:
            pass
        self._window_created = False

    # ------------------------------------------------------------------ drawing helpers

    def _select_images(self, images: dict) -> list:
        out = []
        if self.camera_names:
            for cam in self.camera_names:
                if cam in images and _is_image(images[cam]):
                    out.append((cam, np.asarray(images[cam])))
        else:
            for key, val in images.items():
                if _is_image(val):
                    out.append((key, np.asarray(val)))
        return out

    def _tile(self, frames: list) -> Optional[np.ndarray]:
        cv2 = self._cv2
        if not frames:
            return None
        target_h = 360
        tiles = []
        for name, img in frames:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif self.bgr:
                # Incoming frames are RGB; convert to BGR for cv2 display.
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = np.ascontiguousarray(img)
            h, w = img.shape[:2]
            scale = target_h / h
            img = cv2.resize(img, (max(1, int(w * scale)), target_h))
            cv2.putText(img, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            tiles.append(img)
        strip = cv2.hconcat(tiles) if len(tiles) > 1 else tiles[0]
        if strip.shape[1] > self.max_width:
            scale = self.max_width / strip.shape[1]
            strip = cv2.resize(strip, (self.max_width, int(strip.shape[0] * scale)))
        return strip

    def _render(self, images: Optional[dict], status: dict) -> Optional[np.ndarray]:
        cv2 = self._cv2
        strip = self._tile(self._select_images(images)) if images else None
        width = strip.shape[1] if strip is not None else max(640, self.max_width // 2)

        panel = np.zeros((self.panel_height, width, 3), dtype=np.uint8)
        self._draw_panel(panel, status)

        if strip is None:
            return panel
        return cv2.vconcat([strip, panel])

    def _draw_panel(self, panel: np.ndarray, status: dict) -> None:
        cv2 = self._cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (235, 235, 235)
        grey  = (150, 150, 150)
        yellow = (0, 230, 230)
        green  = (60, 230, 60)
        red    = (60, 60, 235)

        def put(text, x, y, color=white, scale=0.5, thick=1):
            cv2.putText(panel, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

        def put_fit(text, x, y, color, scale, thick, max_w):
            """Draw text, shrinking the font so it never overflows max_w pixels."""
            s = scale
            while s > 0.4:
                (tw, _th), _ = cv2.getTextSize(text, font, s, thick)
                if tw <= max_w:
                    break
                s -= 0.1
            cv2.putText(panel, text, (x, y), font, s, color, thick, cv2.LINE_AA)

        y = 24

        task = status.get("task")
        if task:
            put(f"TASK: {task}", 10, y, green, 0.6, 1)
            y += 26

        # Loop / timing info
        meta = []
        if status.get("step") is not None:
            meta.append(f"step {status['step']}")
        if status.get("fps") is not None:
            meta.append(f"{status['fps']:.1f} fps")
        if status.get("elapsed") is not None:
            elapsed = status["elapsed"]
            meta.append(f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
        if meta:
            put(" | ".join(meta), 10, y, grey, 0.5)
            y += 24

        # Recording state + episode / frame counters
        if status.get("recording") is not None:
            rec  = bool(status.get("recording"))
            recovery = status.get("mode") == "recovery"
            ep       = status.get("episode_index")
            buffered = status.get("buffered_frames")
            if rec:
                extra = ""
                if ep is not None and buffered is not None:
                    extra = f"   episode #{ep}  ({buffered} frames)"
                put(f"REC{extra}", 10, y, red, 0.6, 2)
            else:
                put("idle", 10, y, grey, 0.6, 2)
            put("RECOVERY" if recovery else "nominal",
                panel.shape[1] - 130, y,
                red if recovery else grey, 0.6, 2)
            y += 26

            saved_eps    = status.get("saved_episodes")
            saved_frames = status.get("saved_frames")
            if saved_eps is not None and saved_frames is not None:
                put(f"dataset: {saved_eps} episodes | {saved_frames} steps saved",
                    10, y, green, 0.5)
                y += 24

        # Current subtask — rendered ~3x larger than the other lines so the
        # operator can read it at a glance from across the workspace.
        skill_text = status.get("skill_text")
        if skill_text:
            put("SUBTASK:", 10, y, yellow, 0.6, 1)
            y += 36
            put_fit(skill_text, 10, y + 12, yellow, 1, 3,
                    max_w=panel.shape[1] - 20)
