"""DAgger-style data collection for System 1 (Generative Executor).

Runs the trained PI0 policy on the robot (System-1 async inference) while
recording the resulting observation/action stream **directly into a LeRobot v3
dataset** (Parquet + MP4) — the exact format ``train_system1.py`` consumes — so
on-policy rollouts (optionally with human teleop correction) can be folded back
into training without any conversion step (DAgger).

The recorder captures, per frame:

    * ``observation.state`` / ``action`` / camera images;
    * ``task``          — the high-level task instruction (constant per run);
    * ``subtask``       — the current atomic skill instruction (stored as an
                          integer ``subtask_index`` column + ``meta/subtasks.parquet``
                          so LeRobot resolves ``item["subtask"]`` on reload);
    * ``back`` windows  — frames the operator marked as a precondition-violation
                          / recovery (manual 'e' toggle or human teleop takeover)
                          are written to
                          ``back_annotations.json`` next to the dataset, the
                          format ``CompletionGateDataset.back_annotations_path``
                          consumes for the System-2 back head.

Just like ``robot_inference.py``, the policy is fed ``task`` *and* ``subtask``
separately so the PI0 preprocessor builds the same
``"task: <high> subtask: <atomic>\\n"`` prompt at collection time as at
deployment time — keeping the DAgger data on-distribution.

Recording / correction is driven from the keyboard:

    'c'     start recording a new episode
    space   pause / resume recording
    'r'     discard the current (unsaved) episode buffer
    's'     save the current episode into the dataset
    1-9     switch to the N-th skill (declaration order)
    0       reset to the default (high-level) task text
    'e' toggle a manual back (correction) window on/off
    't'     toggle human teleop takeover (requires --teleop); takeover frames
            are recorded as back=1
    'q'     finalize the dataset and quit
    ESC     quit (finalizing first)

Captured frames are timestamped during the run and resampled to ``target_fps``
at save time, then written via ``LeRobotDataset.add_frame`` / ``save_episode``.
The dataset feature schema (state/action dims, image sizes) is inferred from the
first captured frame.

Dataset lifecycle (mirrors the ROS ``data_to_lerobot3_node`` handling):

    * ``resume: true``  — a corrupted-parquet repair runs at startup (drops any
      footer-less file left by a previous crash and resyncs ``meta/info.json``
      counters) so the dataset reloads, and new episodes/subtasks/back-windows
      append; the existing video codec is reused so videos stay uniform.
    * ``resume: false`` — if a dataset already exists at ``root`` the operator is
      asked to confirm before it is wiped (auto-yes on a non-interactive stdin).
    * Video encoding is configurable (``video_codec`` / ``streaming_encoding`` /
      ``encoder_threads``); NVENC + streaming make ``s`` near-instant.
    * ``finalize`` is idempotent and also runs on ``atexit`` / SIGTERM, so an
      abnormal exit still closes the video/parquet writers cleanly.

Configuration (same style as ``train_system1.py`` / ``robot_inference.py``): all
parameters live in a YAML file; individual fields can be overridden with
``key=value`` (dot-notation for nested keys)::

    python -m low_level_model.robot.collect_dagger_dataset \
        configs/system1/inference/collect_dagger.yaml

    # override individual fields
    python -m low_level_model.robot.collect_dagger_dataset \
        configs/system1/inference/collect_dagger.yaml \
        root=data/dagger_run2 target_fps=20 robot.type=so101_follower

The plain draccus dotlist CLI also still works::

    python -m low_level_model.robot.collect_dagger_dataset \
        --robot.type=so101_follower --robot.port=/dev/ttyACM0 \
        --robot.cameras="{cam_top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --policy_path=outputs/system1/model_best \
        --task="pick up the green tube" \
        --repo_id=your/dagger_dataset --root=data/dagger \
        --fps=30 --target_fps=20 --control_time_s=600
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from lerobot.robots import RobotConfig

# Importing the robot subpackages registers their RobotConfig subclasses with
# draccus' ChoiceRegistry so `robot.type=...` resolves (lerobot pattern).
from lerobot.robots import (  # noqa: F401
    bi_so100_follower,
    dual_piper,
    koch_follower,
    lekiwi,
    piper,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import TeleoperatorConfig

# Same pattern for teleoperators so `teleop.type=...` resolves (optional human
# correction arm used for DAgger takeover).
from lerobot.teleoperators import (  # noqa: F401
    bi_openarm_leader,
    bi_so_leader,
    gamepad,
    keyboard,
    koch_leader,
    openarm_leader,
    so_leader,
)

from low_level_model.robot.keyboard_control import drain_keys, start_keyboard_listener
from low_level_model.robot.robot_common import (
    RobotAsyncExecutor,
    build_dataset_features,
    prompt_to_start,
)
from low_level_model.robot.runtime_builder import (
    SmoothingConfig,
    build_action_blender,
    load_robot_policy_runtime,
)
from low_level_model.robot.status_view import StatusView
from low_level_model.robot.task_manager import SkillTaskManager, digit_of
from low_level_model.utils.config_loader import parse_with_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- config
@dataclass
class CollectDaggerConfig:
    """DAgger collection run config. Loaded from a YAML file (+ ``key=value``
    overrides) or a draccus dotlist CLI — see the module docstring and
    ``configs/system1/inference/collect_dagger.yaml``."""

    robot: RobotConfig                # Robot config (draccus ChoiceRegistry; `type` selects subclass)
    policy_path: str                  # System-1 (PI0) checkpoint directory
    task: str                         # High-level task description (recorded per-frame as `task`)
    repo_id: str                      # LeRobot dataset repo id
    root: str                         # Local dataset root directory
    device: str = "cuda"
    fps: int = 20                     # Control/capture frequency (Hz)
    target_fps: int = 20              # Dataset fps (frames resampled to this rate at save)
    control_time_s: float = 600.0
    overlap_steps: int = 0            # Chunk overlap for async prefetch (0 = synchronous)
    # torch.compile of the System-1 policy (PI0 only; mirrors robot_inference). None
    # leaves the checkpoint's saved setting untouched; ignored by policies without these fields.
    compile_model: Optional[bool] = None
    compile_mode: Optional[str] = None          # e.g. "max-autotune", "reduce-overhead"
    compile_cache_dir: Optional[str] = None
    max_timesteps: int = 10000        # Cap on buffered frames per episode
    use_videos: bool = True           # Store MP4 video (False → PNG images)
    resume: bool = True               # Resume an existing dataset at `root` (False → confirm + overwrite)
    video_codec: str = "h264_nvenc"
    streaming_encoding: bool = True
    encoder_threads: int = 4
    # Optional skill-library YAML: enables digit-key subtask switching during
    # collection (keys 1-9 → declaration-order skills, 0 resets to `task`).
    skill_library: Optional[str] = None
    # Optional human-correction teleoperator (e.g. a leader arm). When set, 't'
    # toggles teleop takeover; takeover frames are recorded as back=1.
    teleop: Optional[TeleoperatorConfig] = None
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    # Live operator view (OpenCV): observation images + task/subtask + REC/back state.
    enable_view: bool = False
    view_bgr: bool = True             # convert RGB->BGR for cv2 display (RealSense frames are RGB)
    view_max_width: int = 1280        # max width of the tiled camera strip (downscaled if larger)

# --------------------------------------------------------------------- session
class DaggerSession:
    """Live DAgger state shared by the control loop and the recorder.
    Owns the current atomic subtask (via :class:`SkillTaskManager`), the
    human-correction (``back``) flag, and optional teleop takeover.  Kept
    torch/lerobot-free so it can be unit-tested offline.
    ``back_active`` is True whenever the current frames should be labelled
    ``back=1``:
      * while a manual back window is open (toggled with 'e');
      * while the human has taken over via teleop.
    Subtask (digit-key) switching does not affect ``back_active``.
    """
    def __init__(self, manager: SkillTaskManager, teleop=None, executor=None,
                 default_task: str = "") -> None:
        self.manager = manager
        self.teleop = teleop
        self.executor = executor
        self.default_task = default_task
        self.back_active = False
        self.teleop_active = False

    # --- queries -----------------------------------------------------------
    def current_subtask(self) -> str:
        """The current atomic skill instruction (default task if none selected)."""
        if self.manager is None:
            return self.default_task
        return self.manager.current_text()

    # --- transitions -------------------------------------------------------
    def switch_to_digit(self, digit: int) -> None:
        """Switch the active subtask from a digit key.

        Subtask switching no longer touches the back flag: ``back_active`` is
        controlled solely by the explicit 'e' toggle (and teleop takeover), so a
        digit press never auto-opens or clears a back window.
        """
        if self.manager is None:
            return
        self.manager.switch_to_digit(digit)

    def toggle_back(self) -> None:
        """Manually open/close a back (correction) window."""
        self.back_active = not self.back_active

    def toggle_teleop(self) -> bool:
        """Toggle human teleop takeover.

        Returns True if control is now with the human (recording back=1), False
        if it was handed back to the policy.  Either way the System-1 executor
        is reset so the policy re-infers from the corrected state.  No-op (returns
        False) if no teleop device is attached.
        """
        if self.teleop is None:
            return False
        self.teleop_active = not self.teleop_active
        self.back_active = self.teleop_active
        if self.executor is not None:
            self.executor.reset()
        return self.teleop_active

    def reset_episode_state(self) -> None:
        """Clear per-episode correction flags (called after save/discard)."""
        self.back_active = False
        self.teleop_active = False


# --------------------------------------------------------------------- recorder
class LeRobotV3Recorder:
    """Buffer on-policy rollouts and write them as LeRobot v3 episodes.

    Frames are buffered in memory while recording; on save the buffer is
    resampled to ``target_fps`` and flushed into the dataset as a single episode
    via ``add_frame`` + ``save_episode``.  The dataset is created lazily from the
    first captured frame (so state/action/image shapes are inferred), or resumed
    if one already exists at ``root``.

    Two side-car artifacts are maintained alongside the dataset:

      * ``meta/subtasks.parquet`` — first-seen-ordered map of subtask text →
        ``subtask_index`` (mirrors lerobot's ``tasks.parquet`` layout); combined
        with the per-frame ``subtask_index`` feature it lets a reloaded
        ``LeRobotDataset`` resolve ``item["subtask"]``.
      * ``back_annotations.json`` — episode-local ``[start, end]`` (inclusive)
        windows of frames the operator marked as ``back`` corrections.
    """

    def __init__(self, repo_id: str, root: str, target_fps: int, robot_type: str,
                 use_videos: bool = True, resume: bool = True, max_timesteps: int = 10000,
                 video_codec: str = "libsvtav1", streaming_encoding: bool = False,
                 encoder_threads: int = 4):
        self.repo_id = repo_id
        self.root = root
        self.target_fps = target_fps
        self.robot_type = robot_type
        self.use_videos = use_videos
        self.resume = resume
        self.max_timesteps = max_timesteps
        self.video_codec = video_codec
        self.streaming_encoding = streaming_encoding
        self.encoder_threads = encoder_threads

        self.dataset = None
        self.enabled = False
        self.paused = True
        self._finalized = False
        self._buf: list[dict] = []

        # Resolve resume/overwrite up front: repair any half-written parquet left
        # by a previous crash (so resume can load), or confirm + wipe an existing
        # dataset when resume is off. Mirrors the ROS collection node.
        self._prepare_root()

        # First-seen subtask text -> subtask_index; back windows accumulated
        # across episodes. Restored from disk when resuming so labels/windows
        # stay stable and new ones append.
        self._subtask_to_index: dict[str, int] = {}
        self._back_windows: list[dict] = []
        self._restore_sidecars()

    # --- recording state ---------------------------------------------------
    def start(self) -> None:
        self.enabled, self.paused = True, False
        logger.info("[rec] recording started")

    def toggle_pause(self) -> None:
        if self.enabled:
            self.paused = not self.paused
            logger.info("[rec] %s", "paused" if self.paused else "resumed")

    def discard(self) -> None:
        self.enabled, self.paused = False, True
        self._buf.clear()
        logger.info("[rec] discarded current episode buffer")

    @property
    def recording(self) -> bool:
        return self.enabled and not self.paused

    @property
    def total_episodes(self) -> int:
        """Episodes already saved into the dataset.

        Uses the live dataset counter once it exists; otherwise (e.g. resuming
        before the first save this run, when the dataset is still created lazily)
        reads it from the on-disk ``meta/info.json`` so the count is accurate from
        the very first keypress. Returns 0 for a brand-new dataset.
        """
        if self.dataset is not None:
            return int(self.dataset.meta.total_episodes)
        info_path = os.path.join(self.root, "meta", "info.json")
        if os.path.isfile(info_path):
            try:
                with open(info_path, encoding="utf-8") as f:
                    return int(json.load(f).get("total_episodes", 0))
            except Exception:  # noqa: BLE001
                return 0
        return 0

    @property
    def total_frames(self) -> int:
        """Total frames (steps) already saved into the dataset.

        Like :attr:`total_episodes`, reads the live dataset counter once it
        exists, else the on-disk ``meta/info.json`` so the count is accurate
        before this run's first save. Returns 0 for a brand-new dataset.
        """
        if self.dataset is not None:
            return int(self.dataset.meta.total_frames)
        info_path = os.path.join(self.root, "meta", "info.json")
        if os.path.isfile(info_path):
            try:
                with open(info_path, encoding="utf-8") as f:
                    return int(json.load(f).get("total_frames", 0))
            except Exception:  # noqa: BLE001
                return 0
        return 0

    @property
    def current_episode_index(self) -> int:
        """Index the next saved episode will receive (= episodes saved so far).

        While an episode is being buffered this is the index it will be written
        under, so the live view can label the in-progress recording.
        """
        return self.total_episodes

    def status(self) -> str:
        """One-line recording status for keypress feedback (episodes saved so far,
        frames buffered for the current unsaved episode, recording state)."""
        state = "recording" if self.recording else ("paused" if self.enabled else "idle")
        return (f"episodes saved: {self.total_episodes} | "
                f"buffered frames: {len(self._buf)} | {state}")

    # --- capture -----------------------------------------------------------
    def record(self, state: np.ndarray, action: np.ndarray, images: dict[str, np.ndarray],
               task: str, subtask: str, back: bool = False) -> None:
        if not self.recording or len(self._buf) >= self.max_timesteps:
            return
        self._buf.append({
            "timestamp": time.time(),
            "state": np.array(state, dtype=np.float32),
            "action": np.array(action, dtype=np.float32),
            "images": {k: np.array(v) for k, v in images.items()},
            "task": task,
            "subtask": subtask,
            "back": bool(back),
        })

    # --- subtask / back bookkeeping ---------------------------------------
    def _subtask_index(self, text: str) -> int:
        """Return the (stable, first-seen) integer index for a subtask text."""
        if text not in self._subtask_to_index:
            self._subtask_to_index[text] = len(self._subtask_to_index)
        return self._subtask_to_index[text]

    @staticmethod
    def _contiguous_windows(flags: list[bool]) -> list[tuple[int, int]]:
        """Inclusive ``[start, end]`` ranges of consecutive True flags."""
        windows: list[tuple[int, int]] = []
        start: Optional[int] = None
        for i, flag in enumerate(flags):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                windows.append((start, i - 1))
                start = None
        if start is not None:
            windows.append((start, len(flags) - 1))
        return windows

    def _restore_sidecars(self) -> None:
        """On resume, reload the subtask map and back windows so they append."""
        if not self.resume:
            return
        try:
            from pathlib import Path

            from lerobot.datasets.utils import load_subtasks
            df = load_subtasks(Path(self.root))
            if df is not None and "subtask_index" in df.columns:
                self._subtask_to_index = {str(text): int(idx) for text, idx in df["subtask_index"].items()}
        except Exception:  # noqa: BLE001
            pass
        ann_path = os.path.join(self.root, "back_annotations.json")
        if os.path.isfile(ann_path):
            try:
                with open(ann_path, encoding="utf-8") as f:
                    self._back_windows = json.load(f)
            except Exception:  # noqa: BLE001
                self._back_windows = []

    # --- resume / overwrite safety ----------------------------------------
    def _prepare_root(self) -> None:
        """Resolve the on-disk dataset state before any recording.

        * resume + existing dataset → repair half-written parquet files left by a
          previous crash so :class:`LeRobotDataset` can load it again;
        * not resume + existing dataset → confirm with the operator, then wipe it
          (so a stale dataset is never silently appended to or half-overwritten).
        """
        has_dataset = os.path.isdir(os.path.join(self.root, "meta"))
        if not has_dataset:
            return
        if self.resume:
            self._repair_corrupted_parquets(self.root)
            return
        if not self._confirm_overwrite(self.root):
            logger.info("[rec] operator declined overwrite; aborting.")
            sys.exit(0)
        logger.warning("[rec] overwriting existing dataset at %s", self.root)
        shutil.rmtree(self.root)

    @staticmethod
    def _confirm_overwrite(root: str) -> bool:
        """Ask the operator before deleting an existing dataset (auto-yes on a
        non-interactive stdin so unattended runs are not deadlocked)."""
        if not sys.stdin or not sys.stdin.isatty():
            logger.warning("[rec] resume=False and dataset exists at %s; stdin not a TTY → overwriting.", root)
            return True
        try:
            answer = input(f"\n>>> Dataset already exists at {root}.\n"
                           f"    resume=False will DELETE it. Overwrite? [y/N]: ").strip().lower()
        except EOFError:
            answer = ""
        return answer in ("y", "yes")

    @staticmethod
    def _repair_corrupted_parquets(root: str) -> None:
        """Remove parquet files left without a footer by a previous crash.

        A killed process can leave a ``ParquetWriter`` without its footer bytes;
        PyArrow then raises on the next load. We drop such files so the remaining
        valid episodes still resume, and resync ``meta/info.json`` counters with
        the surviving episode metadata. Mirrors the ROS collection node; wrapped
        defensively so a repair failure never blocks startup.
        """
        try:
            from pathlib import Path

            import pyarrow.parquet as pq

            removed = []
            for pq_path in Path(root).rglob("*.parquet"):
                try:
                    pq.read_metadata(pq_path)
                except Exception:  # noqa: BLE001
                    removed.append(str(pq_path))
                    pq_path.unlink()
            if not removed:
                return
            logger.warning("[rec] removed %d corrupted parquet file(s) from a previous crash:\n  %s",
                           len(removed), "\n  ".join(removed))

            info_path = os.path.join(root, "meta", "info.json")
            episodes_dir = Path(root) / "meta" / "episodes"
            if not (os.path.isfile(info_path) and episodes_dir.exists()):
                return
            surviving = sorted(episodes_dir.rglob("*.parquet"))
            total_eps, total_frames = 0, 0
            for ep_pq in surviving:
                tbl = pq.read_table(ep_pq)
                total_eps += len(tbl)
                if "dataset_to_index" in tbl.column_names:
                    col = tbl.column("dataset_to_index").to_pylist()
                    if col:
                        total_frames = max(total_frames, max(col))
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
            info["total_episodes"] = total_eps
            info["total_frames"] = total_frames
            info["splits"] = {"train": f"0:{total_eps}"}
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
            logger.info("[rec] repaired info.json: total_episodes=%d total_frames=%d", total_eps, total_frames)
        except Exception as exc:  # noqa: BLE001
            logger.error("[rec] parquet repair failed (continuing): %s", exc)

    @staticmethod
    def _get_local_video_codec(root: str, default: str) -> str:
        """Read the ffmpeg encoder an existing dataset was created with, so new
        episodes stay byte-compatible with the existing videos on resume.

        LeRobot stores the *stream* codec name (e.g. 'av1', 'h264', 'hevc') under
        each video feature's ``info.video.codec``; map it back to the encoder
        name. Returns *default* when undeterminable.
        """
        info_path = os.path.join(root, "meta", "info.json")
        if not os.path.isfile(info_path):
            return default
        try:
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
            codec_to_encoder = {"av1": "libsvtav1", "h264": "h264_nvenc", "hevc": "hevc_nvenc"}
            for feat in info.get("features", {}).values():
                codec = feat.get("info", {}).get("video.codec")
                if codec:
                    return codec_to_encoder.get(codec, codec)
        except Exception:  # noqa: BLE001
            pass
        return default

    # --- dataset lifecycle -------------------------------------------------
    def _build_features(self, sample: dict) -> dict:
        features = {
            "observation.state": {"dtype": "float32", "shape": (sample["state"].shape[-1],),
                                  "names": [f"state_{i}" for i in range(sample["state"].shape[-1])]},
            "action": {"dtype": "float32", "shape": (sample["action"].shape[-1],),
                       "names": [f"action_{i}" for i in range(sample["action"].shape[-1])]},
            # Per-frame atomic-skill label; text lives in meta/subtasks.parquet.
            "subtask_index": {"dtype": "int64", "shape": (1,), "names": None},
        }
        for cam, img in sample["images"].items():
            h, w = img.shape[:2]
            features[cam] = {"dtype": "video" if self.use_videos else "image",
                             "shape": (h, w, 3), "names": ["height", "width", "channel"]}
        return features

    def _ensure_dataset(self, sample: dict) -> None:
        if self.dataset is not None:
            return
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if os.path.isdir(self.root) and os.path.isdir(os.path.join(self.root, "meta")) and self.resume:
            # Reuse the codec the dataset was originally encoded with so new
            # episodes stay consistent with the existing video files.
            codec = self._get_local_video_codec(self.root, self.video_codec)
            if codec != self.video_codec:
                logger.warning("[rec] existing dataset uses codec '%s'; keeping it on resume "
                               "(config requested '%s'). Start a fresh dataset to switch codecs.",
                               codec, self.video_codec)
            logger.info("[rec] resuming dataset at %s (codec=%s, streaming=%s)",
                        self.root, codec, self.streaming_encoding)
            self.dataset = LeRobotDataset(self.repo_id, root=self.root, tolerance_s=1e-4,
                                          revision=self._local_version(self.root),
                                          vcodec=codec, streaming_encoding=self.streaming_encoding,
                                          encoder_threads=self.encoder_threads)
            self.dataset.episode_buffer = self.dataset.create_episode_buffer()
            # The PNG image writer is only needed when frames are not streamed
            # straight into the encoder.
            if not self.streaming_encoding:
                self.dataset.start_image_writer(num_processes=0, num_threads=4)
        else:
            logger.info("[rec] creating dataset at %s (codec=%s, streaming=%s)",
                        self.root, self.video_codec, self.streaming_encoding)
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id, root=self.root, fps=self.target_fps,
                robot_type=self.robot_type, features=self._build_features(sample),
                use_videos=self.use_videos, tolerance_s=1e-4,
                image_writer_processes=0, image_writer_threads=4,
                vcodec=self.video_codec, streaming_encoding=self.streaming_encoding,
                encoder_threads=self.encoder_threads,
            )

    @staticmethod
    def _local_version(root: str) -> str:
        info_path = os.path.join(root, "meta", "info.json")
        if os.path.isfile(info_path):
            with open(info_path) as f:
                return json.load(f).get("codebase_version", "v3.0")
        return "v3.0"

    def _resample(self) -> list[dict]:
        """Nearest-neighbour resample of the captured buffer to ``target_fps``.

        Buffer timestamps are monotonically increasing (frames are appended in
        capture order), so nearest-neighbour selection is a single vectorised
        ``np.searchsorted`` (O(N log N)) instead of the old per-target linear
        scan (O(N*M) ≈ O(N²)) — that scan dominated the save wait once an episode
        held thousands of frames.
        """
        if not self._buf:
            return []
        ts = np.fromiter((f["timestamp"] for f in self._buf), dtype=np.float64, count=len(self._buf))
        start, end = ts[0], ts[-1]
        if end <= start:
            return list(self._buf)
        targets = np.arange(start, end, 1.0 / self.target_fps)
        # Insertion points bracket each target between neighbours ts[idx-1], ts[idx];
        # pick whichever neighbour is nearer.
        idx = np.clip(np.searchsorted(ts, targets), 1, len(ts) - 1)
        choose_left = (targets - ts[idx - 1]) <= (ts[idx] - targets)
        nearest = np.where(choose_left, idx - 1, idx)
        return [self._buf[int(i)] for i in nearest]

    def save(self) -> bool:
        if not self._buf:
            logger.warning("[rec] nothing to save")
            return False
        frames = self._resample()
        if len(frames) < 10:
            logger.warning("[rec] episode has only %d frame(s) after resampling — "
                           "consider discarding (r) instead of saving.", len(frames))
        self._ensure_dataset(frames[0])

        # Episode index of the episode about to be written (assigned by lerobot
        # at create_episode_buffer time = current total_episodes). Capture it
        # before save_episode increments the counter so back windows are tagged
        # with the right episode.
        episode_index = int(self.dataset.meta.total_episodes)

        for f in frames:
            frame = {
                "observation.state": f["state"],
                "action": f["action"],
                "task": f["task"],
                "subtask_index": np.array([self._subtask_index(f["subtask"])], dtype=np.int64),
            }
            for cam, img in f["images"].items():
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                frame[cam] = img
            self.dataset.add_frame(frame)
        self.dataset.save_episode()

        # Record back windows (episode-local, inclusive) over the resampled frames.
        for s, e in self._contiguous_windows([bool(f["back"]) for f in frames]):
            self._back_windows.append({"episode": episode_index, "frame_start": int(s), "frame_end": int(e)})

        try:
            self.dataset.meta._flush_metadata_buffer()
        except Exception:  # noqa: BLE001
            pass
        logger.info("[rec] saved episode %d | total_episodes=%d total_frames=%d back_windows=%d",
                    episode_index, self.dataset.meta.total_episodes, self.dataset.meta.total_frames,
                    len(self._back_windows))
        self.discard()
        return True

    def _write_subtasks(self) -> None:
        """Write meta/subtasks.parquet (index=text named 'subtask', col=subtask_index)."""
        if not self._subtask_to_index:
            return
        from pathlib import Path

        import pandas as pd
        ordered = sorted(self._subtask_to_index.items(), key=lambda kv: kv[1])
        texts = [t for t, _ in ordered]
        idxs = [i for _, i in ordered]
        df = pd.DataFrame({"subtask_index": idxs}, index=pd.Index(texts, name="subtask"))
        path = Path(self.root) / "meta" / "subtasks.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    def _write_back_annotations(self) -> None:
        path = os.path.join(self.root, "back_annotations.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._back_windows, f, indent=2)

    def finalize(self) -> None:
        """Flush encoders + side-cars and write final metadata.

        Idempotent: safe to call from the main ``finally`` block *and* from the
        atexit / signal shutdown hooks, so an abnormal exit (e.g. SIGTERM) still
        closes video/parquet writers cleanly instead of leaving a half-written
        dataset behind.
        """
        if self._finalized:
            return
        self._finalized = True
        if self.dataset is None:
            return
        try:
            if getattr(self.dataset, "image_writer", None) is not None:
                self.dataset.stop_image_writer()
            self.dataset.finalize()
            self._write_subtasks()
            self._write_back_annotations()
            logger.info("[rec] dataset finalized at %s (%d subtasks, %d back windows)",
                        self.root, len(self._subtask_to_index), len(self._back_windows))
        except Exception as exc:  # noqa: BLE001
            logger.error("[rec] finalize failed: %s", exc, exc_info=True)

# --------------------------------------------------------------------- keyboard
def handle_keys(event_queue: "queue.Queue", recorder: LeRobotV3Recorder,
                session: Optional["DaggerSession"] = None) -> bool:
    """Process all pending key events. Returns False to stop the loop (q / ESC).

    Recording controls:
        'c'    start a new episode
        space  pause / resume
        'r'    discard the current episode buffer
        's'    save the current episode

    Subtask / correction (requires a session with a skill library):
        1-9    switch to the N-th skill in declaration order
        0      reset to the default task text
        'e'    toggle a manual back (correction) window
        't'    toggle human teleop takeover (records back=1)
    """
    for key in drain_keys(event_queue):
        digit = digit_of(key)
        if digit is not None and session is not None:
            session.switch_to_digit(digit)
            print(f"[subtask] -> {session.current_subtask()}")
        elif key == "c":
            recorder.start()
            print(f"[rec] start    | {recorder.status()}")
        elif key == "space":
            recorder.toggle_pause()
            print(f"[rec] pause    | {recorder.status()}")
        elif key == "r":
            recorder.discard()
            if session is not None:
                session.reset_episode_state()
            print(f"[rec] discard  | {recorder.status()}")
        elif key == "s":
            saved = recorder.save()
            if session is not None:
                session.reset_episode_state()
            tag = "saved" if saved else "nothing to save"
            print(f"[rec] {tag:<8} | {recorder.status()}")
        elif key == "e" and session is not None:
            session.toggle_back()
            print(f"[back] {'on' if session.back_active else 'off'}")
        elif key == "t" and session is not None:
            on = session.toggle_teleop()
            print(f"[teleop] {'takeover (recording back=1)' if on else 'handed back to policy'}")
        elif key in ("q", "esc"):
            return False
    return True

# --------------------------------------------------------------------- control loop
@torch.inference_mode()
def run_loop(robot, executor: RobotAsyncExecutor, recorder: LeRobotV3Recorder, dataset_features: dict,
             camera_map: dict[str, str], task: str, fps: int,
             control_time_s: float, event_queue: "queue.Queue",
             session: Optional["DaggerSession"] = None, teleop=None,
             view: Optional["StatusView"] = None) -> None:
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.datasets.utils import build_dataset_frame

    hint = " | 1-9 switch subtask | 0 reset | 'e' back | 't' teleop" if session is not None else ""
    logger.info("Keys: 'c' start | space pause | 'r' discard | 's' save | 'q'/ESC quit%s", hint)
    if session is not None and session.manager is not None:
        session.manager.print_menu()

    start = time.perf_counter()
    step = 0
    meas_fps = float(fps)  # EMA of the true (full-period) loop rate, shown in the view
    prev_loop_start: Optional[float] = None
    while time.perf_counter() - start < control_time_s:
        loop_start = time.perf_counter()
        if prev_loop_start is not None:
            period = loop_start - prev_loop_start
            if period > 0:
                meas_fps = 0.9 * meas_fps + 0.1 * (1.0 / period)
        prev_loop_start = loop_start
        if not handle_keys(event_queue, recorder, session):
            break
        try:
            raw_observation = robot.get_observation()
        except TimeoutError as exc:
            logger.warning("[cam] frame read timed out at step %d (%s); skipping control step.", step, exc)
            precise_sleep(max(0.0, 1.0 / fps - (time.perf_counter() - loop_start)))
            step += 1
            continue
        observation_frame = build_dataset_frame(dataset_features, raw_observation, prefix="observation")

        # Action source: human teleop during takeover, otherwise the policy.
        if session is not None and session.teleop_active and teleop is not None:
            action = teleop.get_action()
        else:
            action = executor.get_action(observation_frame)
        robot.send_action(action)

        if recorder.recording:
            current_subtask = session.current_subtask() if session is not None else task
            back = session.back_active if session is not None else False
            state = np.asarray(observation_frame["observation.state"], dtype=np.float32)
            action_vec = np.asarray(list(action.values()), dtype=np.float32)
            images = {feat_key: _to_hwc_uint8(raw_observation[cam])
                      for cam, feat_key in camera_map.items() if cam in raw_observation}
            recorder.record(state, action_vec, images, task, current_subtask, back=back)

        if step % 100 == 0:
            current_subtask = session.current_subtask() if session is not None else task
            logger.info("step %d, elapsed %.1fs, recording=%s, back=%s, subtask: %s", step,
                        time.perf_counter() - start, recorder.recording,
                        session.back_active if session is not None else False, current_subtask)

        # Push the latest frame + collection state to the live view (non-blocking).
        if view is not None:
            view.update(raw_observation, {
                "task": task,
                "skill_text": session.current_subtask() if session is not None else task,
                "recording": recorder.recording,
                "back": session.back_active if session is not None else False,
                "step": step,
                "elapsed": time.perf_counter() - start,
                "fps": meas_fps,
                # DAgger dataset / episode bookkeeping for the live overlay.
                "episode_index": recorder.current_episode_index,
                "buffered_frames": len(recorder._buf),
                "saved_episodes": recorder.total_episodes,
                "saved_frames": recorder.total_frames,
            })

        precise_sleep(max(0.0, 1.0 / fps - (time.perf_counter() - loop_start)))
        step += 1


def _to_hwc_uint8(img) -> np.ndarray:
    arr = img.cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] == 3:  # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return arr


def main():

    cfg = parse_with_yaml(CollectDaggerConfig)
    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(cfg.robot)
    runtime = load_robot_policy_runtime(
        cfg.policy_path,
        device=cfg.device,
        robot=robot,
        compile_model=cfg.compile_model,
        compile_mode=cfg.compile_mode,
        compile_cache_dir=cfg.compile_cache_dir,
    )
    policy_config = runtime.config
    policy = runtime.policy
    preprocessor = runtime.preprocessor
    postprocessor = runtime.postprocessor

    # Optional skill library for digit-key subtask switching during DAgger collection.
    manager = None
    if cfg.skill_library:
        from high_level_model.planning.skill_library import SkillLibrary
        manager = SkillTaskManager(SkillLibrary.from_file(cfg.skill_library), cfg.task)
        manager.switch_to_digit(1)

    # Optional human-correction teleoperator (leader arm, gamepad, ...).
    teleop = None
    if cfg.teleop is not None:
        from lerobot.teleoperators import make_teleoperator_from_config
        teleop = make_teleoperator_from_config(cfg.teleop)

    session = DaggerSession(manager, teleop=teleop, executor=None, default_task=cfg.task)
    task_provider = lambda: cfg.task
    subtask_provider = session.current_subtask

    # Optional output-side anti-jitter: cross-fade each chunk into the previous one so
    # the recorded actions match robot_inference.py's smoothed deployment stream.
    blender = build_action_blender(cfg.smoothing, cfg.overlap_steps)

    executor = RobotAsyncExecutor(
        policy, robot, task_provider, overlap_steps=cfg.overlap_steps,
        preprocessor=preprocessor, postprocessor=postprocessor,
        subtask_provider=subtask_provider, blender=blender,
    )
    session.executor = executor

    # Warm up (and compile) on the executor's worker thread before connecting the robot.
    if getattr(policy_config, "compile_model", False):
        executor.warmup(task_provider(), preprocessor=preprocessor)
    dataset_features = build_dataset_features(robot)

    # cam name (robot key) -> LeRobot feature key.
    from lerobot.utils.constants import OBS_IMAGES
    camera_map = {cam: f"{OBS_IMAGES}.{cam}" for cam in robot.cameras.keys()}

    recorder = LeRobotV3Recorder(
        repo_id=cfg.repo_id, root=cfg.root, target_fps=cfg.target_fps,
        robot_type=getattr(robot, "robot_type", "robot"),
        use_videos=cfg.use_videos, resume=cfg.resume, max_timesteps=cfg.max_timesteps,
        video_codec=cfg.video_codec, streaming_encoding=cfg.streaming_encoding,
        encoder_threads=cfg.encoder_threads,
    )
    atexit.register(recorder.finalize)

    def _on_sigterm(signum, _frame):
        logger.warning("[rec] received signal %d; finalizing dataset and exiting.", signum)
        recorder.finalize()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    prompt_to_start("Press ENTER to connect the robot and start DAgger collection")

    event_queue: "queue.Queue" = queue.Queue()
    listener = start_keyboard_listener(event_queue)

    view = None
    if cfg.enable_view:
        view = StatusView(camera_names=list(camera_map.keys()), bgr=cfg.view_bgr,
                          max_width=cfg.view_max_width).start()

    robot.connect()
    if teleop is not None:
        teleop.connect()
    try:
        run_loop(robot, executor, recorder, dataset_features, camera_map,
                 cfg.task, cfg.fps, cfg.control_time_s, event_queue,
                 session=session, teleop=teleop, view=view)
    finally:
        executor.shutdown()
        recorder.finalize()
        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()
        if listener is not None:
            listener.stop()
        if view is not None:
            view.stop()


if __name__ == "__main__":
    main()
