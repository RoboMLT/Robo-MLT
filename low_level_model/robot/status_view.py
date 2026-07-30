"""Live operator view for real-robot deployment.

A small, dependency-light OpenCV window that shows what the system is *seeing* and *thinking*:
the live observation images tiled side-by-side, with an overlay panel listing the System-2
planner's full skill plan (current step highlighted), the current atomic skill, the latest gate
signals (progress / back / completion), the last decision (stay/advance/recover/done), the
replanning budget, and the loop rate.

Design notes
------------
- The cv2 GUI runs entirely in a dedicated background thread; the 20 Hz control loop only ever
  calls :meth:`update`, which just stashes the latest (images, status) under a lock and returns
  immediately — it never touches cv2 and never blocks on rendering.
- cv2 is imported lazily inside :meth:`start`; if it is unavailable the view degrades to a no-op
  (logged once) so deployment is never blocked by a missing display dependency.

Smoke test (no robot needed)::

    python -m low_level_model.robot.status_view
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["StatusView"]


def _is_image(val) -> bool:
    arr = getattr(val, "shape", None)
    if arr is None:
        return False
    nd = len(val.shape)
    return nd == 2 or (nd == 3 and val.shape[2] in (1, 3, 4))


class StatusView:
    """Threaded OpenCV window showing observation images + System-2 status overlay."""

    def __init__(
        self,
        camera_names: Optional[list[str]] = None,
        bgr: bool = True,
        max_width: int = 1280,
        window_name: str = "Robo-MLT",
        fps: float = 15.0,
        panel_height: int = 260,
    ) -> None:
        # camera_names optionally restricts/orders which observation keys are shown; None = auto.
        self.camera_names = camera_names
        self.bgr = bgr
        self.max_width = max_width
        self.window_name = window_name
        self.frame_interval = 1.0 / max(1e-3, fps)
        self.panel_height = panel_height

        self._lock = threading.Lock()
        self._latest_images: dict | None = None
        self._latest_status: dict | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._enabled = False
        self._cv2 = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "StatusView":
        try:
            import cv2  # noqa: PLC0415  (lazy: optional display dependency)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("StatusView disabled: OpenCV unavailable (%s).", exc)
            return self
        self._cv2 = cv2
        self._enabled = True
        self._thread = threading.Thread(target=self._run, name="status-view", daemon=True)
        self._thread.start()
        return self

    def update(self, images: Optional[dict], status: Optional[dict]) -> None:
        """Stash the latest observation dict + status snapshot. Non-blocking; control-loop safe."""
        if not self._enabled:
            return
        with self._lock:
            self._latest_images = images
            self._latest_status = status

    def stop(self) -> None:
        if not self._enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self._cv2.destroyWindow(self.window_name)
            self._cv2.waitKey(1)
        except Exception:  # pragma: no cover
            pass
        self._enabled = False

    # ------------------------------------------------------------------ render thread
    def _run(self) -> None:
        cv2 = self._cv2
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self._lock:
                images = self._latest_images
                status = dict(self._latest_status) if self._latest_status else {}
            try:
                canvas = self._render(images, status)
                if canvas is not None:
                    cv2.imshow(self.window_name, canvas)
            except Exception as exc:  # pragma: no cover - never kill the loop over a draw error
                logger.debug("StatusView render error: %s", exc)
            # waitKey drives the GUI event loop; the (q) key here just closes the view.
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                self._stop.set()
                break
            dt = time.perf_counter() - t0
            if dt < self.frame_interval:
                time.sleep(self.frame_interval - dt)

    # ------------------------------------------------------------------ drawing helpers
    def _select_images(self, images: dict) -> list[tuple[str, np.ndarray]]:
        out: list[tuple[str, np.ndarray]] = []
        if self.camera_names:
            for cam in self.camera_names:
                key = cam if cam in images else cam.removeprefix("observation.images.")
                if key in images and _is_image(images[key]):
                    out.append((key, np.asarray(images[key])))
        else:
            for key, val in images.items():
                if _is_image(val):
                    out.append((key, np.asarray(val)))
        return out

    def _tile(self, frames: list[tuple[str, np.ndarray]]) -> Optional[np.ndarray]:
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
            elif self.bgr:  # incoming frames are RGB; cv2 displays BGR
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
        white, grey, yellow, green, red = (
            (235, 235, 235), (150, 150, 150), (0, 230, 230), (60, 230, 60), (60, 60, 235))

        def put(text, x, y, color=white, scale=0.5, thick=1):
            cv2.putText(panel, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

        y = 24
        task = status.get("task")
        if task:
            put(f"TASK: {task}", 10, y, green, 0.6, 1)
            y += 26

        # Loop / step info (present even without System 2).
        meta = []
        if status.get("step") is not None:
            meta.append(f"step {status['step']}")
        if status.get("fps") is not None:
            meta.append(f"{status['fps']:.1f} fps")
        if status.get("elapsed") is not None:
            meta.append(f"{status['elapsed']:.0f}s")
        if meta:
            put(" | ".join(meta), 10, y, grey, 0.5)
            y += 24

        # Current atomic skill + recorder state. Used by the DAgger collector, which
        # has no System-2 plan to render; the System-2 view sets `plan` so SKILL is
        # skipped there (the plan already highlights the active skill) and never sets
        # `recording`, so this whole block is a no-op for deployment.
        if status.get("recording") is not None:
            rec, back = bool(status.get("recording")), bool(status.get("back"))
            ep = status.get("episode_index")
            buffered = status.get("buffered_frames")
            # Recording state + which episode is being captured and how many frames so far.
            if rec:
                extra = ""
                if ep is not None and buffered is not None:
                    extra = f"   episode #{ep}  ({buffered} frames buffered)"
                put(f"REC{extra}", 10, y, red, 0.6, 2)
            else:
                put("idle", 10, y, grey, 0.6, 2)
            # Back (correction) flag — drawn as a distinct badge so it stands out
            # whether or not recording is active (operator toggles it with 'b'/'e').
            put("BACK=1" if back else "back=0", panel.shape[1] - 120, y,
                red if back else grey, 0.6, 2)
            y += 26
            # Dataset totals already on disk (episodes + steps saved so far).
            saved_eps = status.get("saved_episodes")
            saved_frames = status.get("saved_frames")
            if saved_eps is not None and saved_frames is not None:
                put(f"dataset: {saved_eps} episodes | {saved_frames} steps saved",
                    10, y, green, 0.5)
                y += 24
        skill_text = status.get("skill_text")
        if skill_text and not status.get("plan"):
            put(f"SKILL: {skill_text}", 10, y, yellow, 0.5)
            y += 24

        plan = status.get("plan")
        if plan:
            pointer = status.get("pointer", -1)
            put("PLAN:", 10, y, white, 0.55, 1)
            y += 22
            for i, sid in enumerate(plan):
                mark = ">" if i == pointer else " "
                color = yellow if i == pointer else (white if i > pointer else grey)
                put(f" {mark} {i}. {sid}", 18, y, color, 0.5)
                y += 20

            # Gate signals + last action on the right column.
            x2 = panel.shape[1] - 320 if panel.shape[1] > 360 else 10
            yr = 24
            action = status.get("action")
            if action:
                acolor = {"advance": green, "recover": red, "done": green}.get(action, white)
                put(f"action: {action}", x2, yr, acolor, 0.6, 1)
                yr += 26
            for label, key in (("progress", "progress"), ("back", "back_prob"),
                               ("completion", "completion_prob")):
                val = status.get(key)
                if val is not None:
                    put(f"{label:11s}: {val:.3f}", x2, yr, white, 0.5)
                    yr += 22
            if status.get("replan_budget") is not None:
                put(f"replan budget: {status['replan_budget']}", x2, yr, grey, 0.5)
                yr += 22
            if status.get("is_done"):
                put("TASK COMPLETE", x2, yr, green, 0.6, 2)


if __name__ == "__main__":
    # Offline smoke test: fake cameras + fake status, no robot/cv2-camera needed.
    logging.basicConfig(level=logging.INFO)
    view = StatusView(bgr=True).start()
    if not view._enabled:
        raise SystemExit("OpenCV not available; cannot run StatusView smoke test.")
    plan = ["grasp_tube", "remove_cap", "discard_cap", "dock_to_analyzer",
            "withdraw_tube", "discard_sample", "return_home"]
    try:
        for step in range(300):
            imgs = {
                "cam_top": (np.random.rand(180, 320, 3) * 255).astype(np.uint8),
                "cam_left": (np.random.rand(180, 320, 3) * 255).astype(np.uint8),
                "joint.pos": 0.5,  # non-image entry, should be ignored
            }
            status = {
                "task": "Run blood-gas analysis on the sample tube.",
                "plan": plan,
                "pointer": (step // 40) % len(plan),
                "skill_id": plan[(step // 40) % len(plan)],
                "progress": (step % 40) / 40.0,
                "back_prob": 0.05,
                "completion_prob": (step % 40) / 40.0,
                "action": "advance" if step % 40 == 39 else "stay",
                "replan_budget": 3,
                "step": step,
                "fps": 20.0,
                "is_done": False,
            }
            view.update(imgs, status)
            time.sleep(0.03)
    finally:
        view.stop()
