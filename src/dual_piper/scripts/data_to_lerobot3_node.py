#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot v3 direct data-collection node.

Subscribes to the camera and joint-state topics, synchronises them, and streams every frame of an
episode straight into a LeRobot v3.0 dataset (Parquet + MP4). Each frame carries the active atomic
skill (``subtask_index``), annotated online by the operator, and a ``back_event`` column that stays
0 at recording time (recovery labels are added offline, see ``tools/label_recovery_tail.py``).

All parameters come from a task config (``config/<task>.yaml``); ROS parameters (``~key``) can
override individual values.

Two modes (``collection.mode``):
    nominal   complete workflows and atomic-skill episodes          -> save with  s
    recovery  episodes ended by an intervention or a failure        -> save with  i / f
              (kept in a separate dataset; tags in meta/recovery_episodes.json)

Controls:
    c       Start / resume collecting
    space   Pause collecting
    1-N     Switch the active subtask (while paused)
    s       Save the episode                         (nominal mode)
    i       Save as an operator-intervention episode (recovery mode)
    f       Save as a teleoperation-failure episode  (recovery mode)
    r       Discard the buffered frames
    q       Finalize the dataset and exit

Usage:
    roslaunch dual_piper collect.launch config:=blood_gas_analysis
    rosrun dual_piper data_to_lerobot3_node.py _config_file:=/path/to/config.yaml
"""
import os
import sys

# Make the sibling ``utils`` package importable under rosrun / roslaunch.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import signal
import atexit
import queue
import shutil
import threading
import collections
import time
import types
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import rospy
from tqdm import tqdm
from pynput import keyboard
from sensor_msgs.msg import Image, JointState, PointCloud2

from utils import (imgmsg_to_cv2, sample_pointcloud, StatusView,
                   repair_dataset_consistency)

# LeRobot v3.0
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_SUBTASKS_PATH


_key_queue: queue.Queue = queue.Queue()
COLLECTION_MODES = ('nominal', 'recovery')
RECOVERY_TYPES = {'i': 'intervention', 'f': 'failure'}
RECOVERY_TAGS_FILE = 'recovery_episodes.json'


def _on_key_press(e):
    """Keyboard listener callback: enqueue char or special-key name."""
    if hasattr(e, 'char') and e.char is not None:
        _key_queue.put(e.char)
    elif hasattr(e, 'name'):
        _key_queue.put(e.name)


def _load_yaml_config(yaml_path: str) -> dict:
    """Load and return the YAML configuration file as a nested dict."""
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f'Config file not found: {yaml_path}')
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f) or {}


def _confirm_overwrite(target_dir: str) -> bool:
    """Ask the user via stdin whether to overwrite an existing dataset.

    Returns True if the user confirms deletion, False otherwise.
    """
    print(f'\n{"=" * 60}')
    print('  WARNING: Dataset already exists at:')
    print(f'    {target_dir}')
    print('  resume is set to False.')
    print(f'{"=" * 60}')
    try:
        answer = input('  Overwrite existing dataset? [y/N]: ').strip().lower()
    except EOFError:
        answer = ''
    return answer in ('y', 'yes')


# ──────────────────────────── ROS Collector Node ──────────────────────────
class LeRobotCollectorNode:
    """
    ROS node that subscribes to camera / joint-state / point-cloud topics,
    streams synchronised frames live into a LeRobot v3.0 dataset (Parquet +
    MP4) as they are collected, so video encoding overlaps with recording and
    saving an episode only finalises the already-encoded data.
    """
    def __init__(self):
        # ── Load YAML Config ─────────────────────────────────────────────
        _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _default_config = os.path.join(_pkg_dir, 'config', 'blood_gas_analysis.yaml')
        config_file = rospy.get_param('~config_file', _default_config)
        ycfg = _load_yaml_config(config_file)
        rospy.loginfo(f'Loaded config from {config_file}')

        ds   = ycfg.get('dataset', {})
        rob  = ycfg.get('robot', {})
        col  = ycfg.get('collection', {})
        cam  = ycfg.get('camera', {})
        pc   = ycfg.get('pointcloud', {})
        sub  = ycfg.get('subtask', {})
        top  = ycfg.get('topics', {})

        # ── Build cfg namespace (YAML defaults, ROS param overrides) ──────
        P = rospy.get_param  # shorthand
        self.cfg = cfg = types.SimpleNamespace(
            # dataset
            repo_id          = P('~repo_id',          ds.get('repo_id', 'user/DualPiper_Dataset')),
            target_dir       = os.path.expanduser(
                P('~target_dir', ds.get('target_dir', '~/robomlt_data/dataset'))),
            task_name        = P('~task_name',        ds.get('task_name', 'default_task')),
            robot_type       = P('~robot_type',       ds.get('robot_type', 'dual_piper')),
            fps              = P('~fps',              ds.get('fps', 20)),
            max_timesteps    = P('~max_timesteps',    col.get('max_timesteps', 5000)),
            state_dim        = P('~state_dim',        rob.get('state_dim', 14)),
            action_dim       = P('~action_dim',       rob.get('action_dim', 14)),
            num_joints_per_arm = P('~num_joints_per_arm', rob.get('num_joints_per_arm', 7)),
            image_h          = P('~image_height',     cam.get('image_height', 180)),
            image_w          = P('~image_width',      cam.get('image_width', 320)),
            camera_names     = P('~camera_names',     cam.get('names', ['cam_top', 'cam_left', 'cam_right'])),
            use_depth        = P('~use_depth_image',  cam.get('use_depth', False)),
            save_pointcloud  = P('~save_pointcloud',  pc.get('enabled', False)),
            pc_target_points = P('~pc_target_points', pc.get('target_points', 5000)),
            use_subtask      = P('~use_subtask',      sub.get('enabled', True)),
            use_back_event   = P('~use_back_event',   col.get('use_back_event', True)),
            mode             = P('~mode',             col.get('mode', 'nominal')),
            use_videos       = P('~use_videos',       cam.get('use_videos', True)),
            resume           = P('~resume',           ds.get('resume', True)),
            video_codec        = P('~video_codec',        ds.get('video_codec', 'h264_nvenc')),
            streaming_encoding = P('~streaming_encoding', col.get('streaming_encoding', True)),
            encoder_threads    = P('~encoder_threads',    col.get('encoder_threads', 4)),
            data_flush_every   = P('~data_flush_every',   col.get('data_flush_every', 10)),
            # Live operator view (OpenCV window with camera feeds + status overlay)
            enable_view        = P('~enable_view',        col.get('enable_view', True)),
            # view_bgr=False: ROS/OpenCV images are already BGR — no conversion needed.
            # Set True only if your image source delivers RGB frames.
            view_bgr           = P('~view_bgr',           col.get('view_bgr', False)),
            view_max_width     = P('~view_max_width',     col.get('view_max_width', 1280)),

            img_top_topic              = P('~img_top_topic',              top.get('img_top', '/cam_top/cam_top/color/image_raw')),
            img_leftwrist_topic        = P('~img_leftwrist_topic',        top.get('img_left', '/cam_leftwrist/cam_leftwrist/color/image_raw')),
            img_rightwrist_topic       = P('~img_rightwrist_topic',       top.get('img_right', '/cam_rightwrist/cam_rightwrist/color/image_raw')),
            img_top_depth_topic        = P('~img_top_depth_topic',        top.get('img_top_depth', '/cam_top/cam_top/aligned_depth_to_color/image_raw')),
            img_leftwrist_depth_topic  = P('~img_leftwrist_depth_topic',  top.get('img_left_depth', '/cam_leftwrist/cam_leftwrist/aligned_depth_to_color/image_raw')),
            img_rightwrist_depth_topic = P('~img_rightwrist_depth_topic', top.get('img_right_depth', '/cam_rightwrist/cam_rightwrist/aligned_depth_to_color/image_raw')),
            left_master_topic          = P('~left_master_arm_topic',      top.get('left_master', 'left/master_joint_states')),
            left_slave_topic           = P('~left_slave_arm_topic',       top.get('left_slave', 'left/slave_joint_states')),
            right_master_topic         = P('~right_master_arm_topic',     top.get('right_master', 'right/master_joint_states')),
            right_slave_topic          = P('~right_slave_arm_topic',      top.get('right_slave', 'right/slave_joint_states')),
            pc_top_topic               = P('~pc_top_topic',               top.get('pc_top', '/cam_top/cam_top/depth/color/points')),
        )

        if cfg.mode not in COLLECTION_MODES:
            raise ValueError(f"collection.mode must be one of {COLLECTION_MODES}, got '{cfg.mode}'")

        # ── Robot-type selection (dual / single arm) ─────────────────────
        # Derive the active arm(s) from robot_type and adjust cameras + dims
        # accordingly, so single-arm collection doesn't record the unused arm's
        # wrist camera / joint state. MUST run before queue + dataset creation.
        self._configure_robot_type()

        # ── Load Subtask Definitions from dataset config ─────────────────
        raw_subtasks = ds.get('subtasks', {})
        if cfg.use_subtask and not raw_subtasks:
            raise ValueError(
                "subtask.enabled is true but dataset.subtasks is empty. "
                "Add subtask definitions under dataset.subtasks of the task config."
            )
        # Ensure keys are strings (YAML may parse '1' as int)
        self.subtasks = {str(k): v for k, v in raw_subtasks.items()}
        rospy.loginfo(f'Loaded {len(self.subtasks)} subtasks from config')
        # Build subtask_name → int index mapping for subtask_index feature
        names = list(self.subtasks.values())
        if len(names) != len(set(names)):
            raise ValueError(f'Duplicate subtask names found: {names}')
        self.subtask_name_to_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(names)
        }
        self._first_subtask_key = next(iter(self.subtasks))

        # ── Data Queues (topic → deque) ──────────────────────────────────
        self.queues: dict = {}
        self._init_queues()
        # ── ROS Subscribers ──────────────────────────────────────────────
        self._init_subscribers()
        self.collecting = False
        self.current_subtask = self.subtasks[self._first_subtask_key]
        # ── LeRobot Dataset ──────────────────────────────────────────────
        self.dataset = None
        self._episodes_since_flush = 0
        self._finalized = False
        self._finalize_lock = threading.Lock()
        self._create_dataset()
        self._recovery_tags = self._load_recovery_tags()
        # ── Live Operator View (OpenCV) ───────────────────────────────
        self._view = None
        if cfg.enable_view:
            self._view = StatusView(
                camera_names=cfg.camera_names,
                bgr=cfg.view_bgr,
                max_width=cfg.view_max_width,
                window_name='DualPiper Collect',
            ).start()
        # ── Shutdown Safety: ensure finalize runs on abnormal exit ─────
        rospy.on_shutdown(self._safe_finalize)
        atexit.register(self._safe_finalize)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        # ── Thread-Safe Keyboard State ───────────────────────────────────
        self._key_lock = threading.Lock()
        self._key_value = None
        for t in (
                threading.Thread(target=self._key_poll_loop, daemon=True),
                threading.Thread(target=self._command_loop, daemon=True),
        ):
            t.start()

        self._print_usage()

    # ─────────────────────── Thread-Safe Key Property ──────────────────
    @property
    def _key(self):
        with self._key_lock:
            return self._key_value

    @_key.setter
    def _key(self, val):
        with self._key_lock:
            self._key_value = val

    def _consume_key(self):
        """Atomically read and clear the current key."""
        with self._key_lock:
            val = self._key_value
            self._key_value = None
            return val

    # ─────────────────────── Robot-Type Selection ────────────────────────
    # robot_type → ordered list of active arms. The order fixes the layout of
    # the concatenated state/action vectors (left first, then right).
    _ROBOT_TYPE_ARMS = {
        'dual_piper':        ['left', 'right'],
        'single_piper_left': ['left'],
        'single_piper_right': ['right'],
    }
    # Each arm's wrist camera. cam_top (and any other non-wrist cam) is shared
    # and always kept regardless of which arm is active.
    _ARM_WRIST_CAM = {'left': 'cam_left', 'right': 'cam_right'}

    def _configure_robot_type(self):
        """Resolve robot_type into active arms and adjust cameras + dims.

        - Sets ``self.active_arms`` (ordered).
        - Drops the inactive arm(s)' wrist camera from ``cfg.camera_names`` while
          keeping cam_top / any shared camera.
        - Overrides ``cfg.state_dim`` / ``cfg.action_dim`` to
          ``len(active_arms) * num_joints_per_arm``.
        """
        cfg = self.cfg
        rt = cfg.robot_type
        if rt not in self._ROBOT_TYPE_ARMS:
            valid = ', '.join(sorted(self._ROBOT_TYPE_ARMS))
            raise ValueError(
                f"Unknown robot_type '{rt}'. Valid options: {valid}."
            )
        self.active_arms = self._ROBOT_TYPE_ARMS[rt]

        # ── Auto-filter cameras: drop inactive arms' wrist cameras ────────
        inactive_wrist_cams = {
            self._ARM_WRIST_CAM[arm]
            for arm in self._ARM_WRIST_CAM
            if arm not in self.active_arms
        }
        filtered = [c for c in cfg.camera_names if c not in inactive_wrist_cams]
        if filtered != list(cfg.camera_names):
            rospy.loginfo(
                f'robot_type={rt}: cameras {list(cfg.camera_names)} → {filtered} '
                f'(dropped {sorted(inactive_wrist_cams)})'
            )
        cfg.camera_names = filtered

        # ── Auto-compute state/action dims from active arms ───────────────
        derived_dim = len(self.active_arms) * cfg.num_joints_per_arm
        if cfg.state_dim != derived_dim or cfg.action_dim != derived_dim:
            rospy.logwarn(
                f'robot_type={rt}: overriding state_dim/action_dim '
                f'({cfg.state_dim}/{cfg.action_dim}) → {derived_dim} '
                f'({len(self.active_arms)} arm(s) × {cfg.num_joints_per_arm} joints)'
            )
        cfg.state_dim = cfg.action_dim = derived_dim

        rospy.loginfo(
            f'robot_type={rt}: active_arms={self.active_arms}, '
            f'cameras={cfg.camera_names}, state_dim={cfg.state_dim}'
        )

    # ─────────────────────── Dataset Creation ────────────────────────────
    def _create_dataset(self):
        """Create a new LeRobotDataset or resume an existing one."""
        cfg = self.cfg
        target = cfg.target_dir

        # ── Resume existing dataset ───────────────────────────────────
        if os.path.exists(target):
            if cfg.resume:
                rospy.loginfo(f'Resuming existing dataset at {target}')
                # Make the dataset self-consistent again after any previous crash.
                # Must run BEFORE LeRobotDataset(): if data and metadata disagree it
                # falls back to the HuggingFace Hub, which hangs for a local repo_id.
                repair_dataset_consistency(target)
                # Reuse the codec the dataset was originally created with so new
                # episodes stay consistent with the existing video files. We can
                # still enable streaming encoding (it works with any codec).
                existing_codec = self._get_local_video_codec(target, cfg.video_codec)
                if existing_codec != cfg.video_codec:
                    rospy.logwarn(
                        f'Existing dataset uses codec "{existing_codec}"; keeping it on '
                        f'resume (config requested "{cfg.video_codec}"). Start a fresh '
                        f'dataset to switch codecs.'
                    )
                self.dataset = LeRobotDataset(
                    repo_id=cfg.repo_id,
                    root=cfg.target_dir,
                    tolerance_s=1e-4,
                    # Use local revision to avoid contacting HuggingFace Hub
                    revision=self._get_local_codebase_version(target),
                    vcodec=existing_codec,
                    streaming_encoding=cfg.streaming_encoding,
                    encoder_threads=cfg.encoder_threads,
                )
                # Initialize recording-related attributes that __init__ doesn't set
                self.dataset.episode_buffer = self.dataset.create_episode_buffer()
                if not cfg.streaming_encoding:
                    self.dataset.start_image_writer(num_processes=0, num_threads=4)
                rospy.loginfo(
                    f'  codec={existing_codec}  streaming={cfg.streaming_encoding}  '
                    f'existing episodes={self.dataset.meta.total_episodes}  '
                    f'frames={self.dataset.meta.total_frames}'
                )
                # A resumed dataset cannot gain a new column mid-stream.
                if cfg.use_back_event and 'back_event' not in self.dataset.meta.features:
                    rospy.logwarn(
                        "use_back_event=True but the resumed dataset has no 'back_event' column; "
                        "disabling it for this run."
                    )
                    cfg.use_back_event = False
                return

            # ── Confirm before overwriting ─────────────────────────────
            if not _confirm_overwrite(target):
                rospy.loginfo('User declined overwrite. Exiting.')
                rospy.signal_shutdown('User declined dataset overwrite')
                sys.exit(0)

            rospy.logwarn(f'Overwriting existing dataset: {target}')
            shutil.rmtree(target)

        # ── Build feature schema ──────────────────────────────────────
        features = {
            'observation.state': {
                'dtype': 'float32',
                'shape': (cfg.state_dim,),
                'names': [f'joint_{i+1}.pos' for i in range(cfg.state_dim)],
            },
            'action': {
                'dtype': 'float32',
                'shape': (cfg.action_dim,),
                'names': [f'joint_{i+1}.pos' for i in range(cfg.action_dim)],
            },
        }
        for cam in cfg.camera_names:
            features[f'observation.images.{cam}'] = {
                'dtype': 'video' if cfg.use_videos else 'image',
                'shape': (cfg.image_h, cfg.image_w, 3),
                'names': ['height', 'width', 'channel'],
            }
        if cfg.save_pointcloud:
            features['observation.pointcloud'] = {
                'dtype': 'float32',
                'shape': (cfg.pc_target_points, 6),
                'names': ['x', 'y', 'z', 'r', 'g', 'b'],
            }
        # Subtask as int64 index (aligned with LeRobot subtask_index system)
        if cfg.use_subtask:
            features['subtask_index'] = {
                'dtype': 'int64',
                'shape': (1,),
                'names': None,
            }
        # Per-frame recovery label (0/1) of the System-2 recovery head. Written as 0 here; the final
        # K_rec frames of each recovery episode are set to 1 offline (tools/label_recovery_tail.py).
        if cfg.use_back_event:
            features['back_event'] = {
                'dtype': 'int64',
                'shape': (1,),
                'names': None,
            }

        self.dataset = LeRobotDataset.create(
            repo_id=cfg.repo_id,
            root=cfg.target_dir,
            fps=cfg.fps,
            robot_type=cfg.robot_type,
            features=features,
            use_videos=cfg.use_videos,
            tolerance_s=1e-4,
            image_writer_processes=0,
            image_writer_threads=4,
            vcodec=cfg.video_codec,
            streaming_encoding=cfg.streaming_encoding,
            encoder_threads=cfg.encoder_threads,
        )
        rospy.loginfo(
            f'Video encoding: codec={cfg.video_codec}  '
            f'streaming={cfg.streaming_encoding}'
        )

        # ── Write meta/subtasks.parquet ───────────────────────────────
        if cfg.use_subtask:
            self._write_subtasks_parquet()

        rospy.loginfo(f'LeRobot dataset created at {cfg.target_dir}')

    @staticmethod
    def _get_local_codebase_version(target_dir: str) -> str:
        """Read codebase_version from local info.json to avoid hub access."""
        info_path = os.path.join(target_dir, 'meta', 'info.json')
        if os.path.isfile(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
            return info.get('codebase_version', 'v3.0')
        return 'v3.0'

    @staticmethod
    def _get_local_video_codec(target_dir: str, default: str) -> str:
        """Read the video codec an existing dataset was encoded with.

        LeRobot stores the codec under each video feature's ``info`` block as
        ``video.codec`` (e.g. 'av1', 'h264', 'hevc'). We map the stream codec
        name back to the ffmpeg encoder name so it can be reused on resume.
        Returns *default* if no codec can be determined.
        """
        info_path = os.path.join(target_dir, 'meta', 'info.json')
        if not os.path.isfile(info_path):
            return default
        with open(info_path, 'r') as f:
            info = json.load(f)
        # Stream-codec-name → ffmpeg-encoder-name
        codec_to_encoder = {'av1': 'libsvtav1', 'h264': 'h264_nvenc', 'hevc': 'hevc_nvenc'}
        for feat in info.get('features', {}).values():
            codec = feat.get('info', {}).get('video.codec')
            if codec:
                return codec_to_encoder.get(codec, codec)
        return default

    def _write_subtasks_parquet(self):
        """Write subtask name → index mapping to meta/subtasks.parquet."""
        subtask_names = list(self.subtasks.values())
        df = pd.DataFrame(
            {'subtask_index': range(len(subtask_names))},
            index=pd.Index(subtask_names, name='subtask'),
        )
        parquet_path = Path(self.cfg.target_dir) / DEFAULT_SUBTASKS_PATH
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path)
        rospy.loginfo(f'Wrote {len(df)} subtasks to {parquet_path}')

    # ─────────────────────── Recovery Episode Tags ───────────────────────
    def _tags_path(self):
        return Path(self.cfg.target_dir) / 'meta' / RECOVERY_TAGS_FILE

    def _load_recovery_tags(self):
        """Load existing tags; drop entries for episodes a crash repair removed."""
        path = self._tags_path()
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            tags = json.load(f)
        total = self.dataset.meta.total_episodes
        stale = [k for k in tags if int(k) >= total]
        for k in stale:
            tags.pop(k)
        if stale:
            rospy.logwarn(f'Dropped {len(stale)} recovery tag(s) of episodes no longer on disk.')
            self._write_recovery_tags(tags)
        return tags

    def _write_recovery_tags(self, tags):
        path = self._tags_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.json.tmp')
        ordered = dict(sorted(tags.items(), key=lambda kv: int(kv[0])))
        with open(tmp, 'w') as f:
            json.dump(ordered, f, indent=2)
        os.replace(tmp, path)

    # ─────────────────────── Queue / Subscriber Init ─────────────────────
    def _arm_state_keys(self):
        """Ordered joint-state queue keys ('<arm>/master', '<arm>/slave') for
        the active arm(s)."""
        return [f'{arm}/{role}'
                for arm in self.active_arms
                for role in ('master', 'slave')]

    def _init_queues(self):
        cfg = self.cfg
        for cam in cfg.camera_names:
            self.queues[cam] = collections.deque(maxlen=5)
            if cfg.use_depth:
                self.queues[f'{cam}_depth'] = collections.deque(maxlen=5)
        for key in self._arm_state_keys():
            self.queues[key] = collections.deque(maxlen=50)
        if cfg.save_pointcloud:
            self.queues['pc_top'] = collections.deque(maxlen=10)

    def _init_subscribers(self):
        cfg = self.cfg
        Q = 50  # subscriber queue size
        # Camera image topics → queue mapping
        _cam_topic_map = {
            'cam_top':   cfg.img_top_topic,
            'cam_left':  cfg.img_leftwrist_topic,
            'cam_right': cfg.img_rightwrist_topic,
        }
        _cam_depth_topic_map = {
            'cam_top':   cfg.img_top_depth_topic,
            'cam_left':  cfg.img_leftwrist_depth_topic,
            'cam_right': cfg.img_rightwrist_depth_topic,
        }
        for cam in cfg.camera_names:
            if cam in _cam_topic_map:
                rospy.Subscriber(_cam_topic_map[cam], Image,
                                 lambda msg, c=cam: self.queues[c].append(msg), queue_size=Q)
            if cfg.use_depth and cam in _cam_depth_topic_map:
                rospy.Subscriber(_cam_depth_topic_map[cam], Image,
                                 lambda msg, c=cam: self.queues[f'{c}_depth'].append(msg), queue_size=Q)

        # Arm joint-state topics (only the active arm(s))
        _arm_topic_map = {
            'left/master':  cfg.left_master_topic,
            'left/slave':   cfg.left_slave_topic,
            'right/master': cfg.right_master_topic,
            'right/slave':  cfg.right_slave_topic,
        }
        for key in self._arm_state_keys():
            rospy.Subscriber(_arm_topic_map[key], JointState,
                             lambda msg, a=key: self.queues[a].append(msg), queue_size=Q)

        # Point cloud
        if cfg.save_pointcloud:
            rospy.Subscriber(cfg.pc_top_topic, PointCloud2,
                             lambda msg: self.queues['pc_top'].append(msg), queue_size=10)

    # ─────────────────────── Frame Synchronisation ───────────────────────
    def _get_synced_frame(self):
        """
        Try to pop one time-synchronised frame from the queues.
        Returns dict with 'images', 'depths', 'pointcloud', 'left/master', ... or None.
        """
        cfg = self.cfg
        needed_keys = list(cfg.camera_names) + self._arm_state_keys()
        if cfg.use_depth:
            needed_keys += [f'{c}_depth' for c in cfg.camera_names]

        # All queues must be non-empty
        if any(len(self.queues.get(k, [])) == 0 for k in needed_keys):
            return None

        # Sync time = latest among the oldest messages
        def _stamp(msg):
            return msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs

        sync_t = min(_stamp(self.queues[k][-1]) for k in needed_keys)

        # Drop older messages
        for k in needed_keys:
            while self.queues[k] and _stamp(self.queues[k][0]) < sync_t:
                self.queues[k].popleft()

        if any(len(self.queues.get(k, [])) == 0 for k in needed_keys):
            return None

        # Pop images
        images = {}
        for cam in cfg.camera_names:
            images[cam] = imgmsg_to_cv2(self.queues[cam].popleft())

        # Pop depth
        depths = {}
        if cfg.use_depth:
            for cam in cfg.camera_names:
                depths[cam] = imgmsg_to_cv2(self.queues[f'{cam}_depth'].popleft())

        # Pop point cloud  
        pointcloud = None
        if cfg.save_pointcloud and self.queues['pc_top']:
            pointcloud = sample_pointcloud(self.queues['pc_top'].popleft(), cfg.pc_target_points)

        frame = {
            'images': images,
            'depths': depths,
            'pointcloud': pointcloud,
        }
        # Pop joint states for the active arm(s) only
        for key in self._arm_state_keys():
            frame[key] = self.queues[key].popleft()
        return frame

    # ─────────────────────── Collection Loop ─────────────────────────────
    def _collect(self):
        """
        Stream synchronised frames straight into the LeRobot episode buffer
        until paused or max_timesteps reached.

        Feeding frames live (via _append_frame → dataset.add_frame) makes the
        streaming video encoder run *concurrently* with collection: each frame
        is encoded as it arrives instead of in one burst at save time. This
        also avoids encoder-queue overflow (feed_frame drops frames when the
        queue fills), and means 's' only finalises an already-encoded episode.
        """
        cfg = self.cfg
        rate = rospy.Rate(cfg.fps)
        count = self._buffered_frames()
        rospy.loginfo(f'▶ Collecting from frame {count} (max {cfg.max_timesteps})')
        rospy.loginfo(f'  Current subtask: {self.current_subtask}')

        pbar = tqdm(total=cfg.max_timesteps, desc='Collecting', ncols=80, initial=count)

        # FPS / elapsed tracking for the live view
        collect_start = time.time()
        meas_fps = float(cfg.fps)
        prev_loop_t = None

        while not rospy.is_shutdown() and count < cfg.max_timesteps:
            loop_t = time.time()
            if prev_loop_t is not None and (loop_t - prev_loop_t) > 0:
                meas_fps = 0.9 * meas_fps + 0.1 / (loop_t - prev_loop_t)
            prev_loop_t = loop_t

            # Handle in-collection keys (atomic read+clear): 'space' pauses, 'q' stops.
            k = self._consume_key()
            if k == 'space':
                rospy.loginfo(f'⏸ Paused at frame {count}')
                self.collecting = False
                pbar.close()
                return
            elif k == 'q':
                # Quit while recording: stop collecting and request a clean
                # shutdown. Teardown (view + finalize) runs on the main thread.
                rospy.loginfo(f'⏹ Stop & exit requested at frame {count}')
                self.collecting = False
                pbar.close()
                self._finalize_and_exit()
                return

            frame = self._get_synced_frame()
            if frame is None:
                rate.sleep()
                continue

            self._append_frame(frame)
            count += 1
            pbar.update(1)

            # Push the latest frame + collection state to the live view (non-blocking).
            if self._view is not None:
                self._view.update(frame['images'], {
                    'task': cfg.task_name,
                    'skill_text': self.current_subtask,
                    'recording': self.collecting,
                    'mode': cfg.mode,
                    'step': count,
                    'elapsed': time.time() - collect_start,
                    'fps': meas_fps,
                    'episode_index': self.dataset.meta.total_episodes,
                    'buffered_frames': count,
                    'saved_episodes': self.dataset.meta.total_episodes,
                    'saved_frames': self.dataset.meta.total_frames,
                })

            rate.sleep()

        pbar.close()
        if count >= cfg.max_timesteps:
            rospy.loginfo(f'⏹ Reached max_timesteps ({cfg.max_timesteps})')
            self.collecting = False

    def _append_frame(self, frame):
        """Convert one synced frame and stream it into the episode buffer.

        Builds the LeRobot frame dict and calls ``dataset.add_frame`` so the
        per-frame cost (BGR→RGB conversion + handing the image to the streaming
        encoder) is paid during recording, at the natural FPS pace, rather than
        all at once when the episode is saved.
        """
        cfg = self.cfg
        njp = cfg.num_joints_per_arm

        # slave topic -> observation.state, master topic -> action (see state_get_node.py for what
        # each contains). Concatenated over the active arm(s), njp DOF each (left before right).
        qpos = np.concatenate(
            [np.array(frame[f'{a}/slave'].position)[:njp] for a in self.active_arms])
        action = np.concatenate(
            [np.array(frame[f'{a}/master'].position)[:njp] for a in self.active_arms])

        lr_frame = {
            'observation.state': qpos.astype(np.float32),
            'action': action.astype(np.float32),
            'task': cfg.task_name,
        }
        for cam in cfg.camera_names:
            img = frame['images'].get(cam)
            if img is not None:
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                # Convert BGR (from OpenCV/ROS) → RGB (expected by LeRobot/PIL)
                lr_frame[f'observation.images.{cam}'] = img[:, :, ::-1].copy()

        if cfg.save_pointcloud and frame.get('pointcloud') is not None:
            lr_frame['observation.pointcloud'] = frame['pointcloud'].astype(np.float32)

        # Per-frame subtask_index — shape must be (1,) to match feature schema
        if cfg.use_subtask:
            lr_frame['subtask_index'] = np.array(
                [self.subtask_name_to_idx[self.current_subtask]], dtype=np.int64
            )
        if cfg.use_back_event:
            lr_frame['back_event'] = np.zeros(1, dtype=np.int64)

        self.dataset.add_frame(lr_frame)

    # ─────────────────────── Save Episode ────────────────────────────────
    def _save_episode(self, recovery_type=None):
        """
        Finalise the in-progress episode; in recovery mode also tag it.

        Frames were already streamed into the LeRobot episode buffer (and the
        video encoder) live during collection by _append_frame, so this only
        has to flush the encoder and write the episode's parquet/metadata —
        making 's' near-instant regardless of episode length.

        Task/subtask strategy (aligned with LeRobot v3):
          - task:  high-level task name (cfg.task_name), same for all frames.
                   LeRobot converts this to task_index automatically.
          - subtask_index:  per-frame int64 index into meta/subtasks.parquet,
                   allowing fine-grained step tracking within an episode.
        """
        n = self._buffered_frames()
        if n == 0:
            rospy.logwarn('Nothing to save — episode buffer is empty.')
            return
        if self.cfg.mode == 'recovery' and recovery_type is None:
            rospy.logwarn('Recovery mode: save with i (intervention) or f (failure).')
            return
        if self.cfg.mode == 'nominal' and recovery_type is not None:
            rospy.logwarn('Nominal mode: save with s. Record recovery episodes with a '
                          '*_recovery config into their own dataset.')
            return
        last_subtask = self.current_subtask

        if n < 10:
            rospy.logwarn(f'Episode has only {n} frames — consider discarding (r).')

        rospy.loginfo(f'💾 Finalizing episode with {n} frames …')

        self.dataset.save_episode()
        # Flush metadata buffer to disk immediately so that episode files
        # are written after every save, not only every N episodes.
        # This also makes crash-resume reliable (no buffered-only episodes lost).
        self.dataset.meta._flush_metadata_buffer()
        # Durability checkpoint. Episode DATA sits in an open ParquetWriter whose
        # footer is only written on close(), while metadata is flushed every episode.
        # A hard kill in between leaves metadata pointing at frames that never became
        # readable — exactly the corruption repair_dataset_consistency() has to undo.
        self._episodes_since_flush += 1
        if self._episodes_since_flush >= self.cfg.data_flush_every:
            self._checkpoint_data_writer()

        ep_idx = self.dataset.meta.total_episodes - 1
        if recovery_type is not None:
            self._recovery_tags[str(ep_idx)] = {
                'type': recovery_type, 'subtask': last_subtask, 'frames': int(n)}
            self._write_recovery_tags(self._recovery_tags)
        rospy.loginfo(
            f'✅ Episode {ep_idx} saved{f" as {recovery_type}" if recovery_type else ""}  |  '
            f'total_episodes={self.dataset.meta.total_episodes}  '
            f'total_frames={self.dataset.meta.total_frames}'
        )
        # save_episode() normally resets the episode buffer, but it pops 'size'
        # first and only recreates the buffer at the very end — guard against
        # versions/paths that leave the buffer in a popped state so the next
        # collection doesn't hit a KeyError.
        buf = getattr(self.dataset, 'episode_buffer', None)
        if not buf or 'size' not in buf:
            self.dataset.episode_buffer = self.dataset.create_episode_buffer()
        # Reset the subtask for the next episode.
        self.current_subtask = self.subtasks[self._first_subtask_key]

    def _checkpoint_data_writer(self):
        """Close the data parquet writer so its footer reaches disk.

        LeRobot keeps one ParquetWriter open across episodes and only emits the
        file footer on close(); until then the file is unreadable and a killed
        process loses every episode in it. Closing here makes all episodes so far
        durable.

        ``_writer_closed_for_reading`` is set so the next episode rolls into a NEW
        parquet file (see LeRobotDataset._save_episode_data): without it LeRobot
        would reopen a ParquetWriter on the SAME path and truncate everything
        already written there.
        """
        ds = self.dataset
        if ds is None or getattr(ds, 'writer', None) is None:
            return
        try:
            ds._close_writer()
            ds._writer_closed_for_reading = True
            self._episodes_since_flush = 0
            rospy.loginfo('💾 Data checkpointed — parquet footer written, '
                          'next episode starts a new file.')
        except Exception as e:
            rospy.logwarn(f'Could not checkpoint data writer: {e}')

    # ─────────────────────── Keyboard & Command Loop ─────────────────────
    def _key_poll_loop(self):
        """Start pynput listener and relay events to self._key."""
        listener = keyboard.Listener(on_press=_on_key_press)
        # Daemon so the listener thread never keeps the process alive after exit.
        listener.daemon = True
        listener.start()
        try:
            while not rospy.is_shutdown():
                try:
                    self._key = _key_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
        finally:
            listener.stop()

    def _command_loop(self):
        """Main state-machine driven by keyboard commands."""
        while not rospy.is_shutdown():
            k = self._consume_key()
            if k is None:
                rospy.sleep(0.05)
                continue

            if k == 'c' and not self.collecting:
                self.collecting = True
                self._collect()
            elif k == 's' and not self.collecting:
                self._save_episode()
            elif k in RECOVERY_TYPES and not self.collecting:
                self._save_episode(recovery_type=RECOVERY_TYPES[k])
            elif k == 'r':
                # Cancel the streaming encoder + drop partial video/images and
                # reset the episode buffer (no-op if nothing buffered yet).
                if self._buffered_frames() > 0:
                    self.dataset.clear_episode_buffer()
                self.current_subtask = self.subtasks[self._first_subtask_key]
                self.collecting = False
                rospy.loginfo('🗑 Buffer cleared, subtask reset.')
            elif k in self.subtasks and not self.collecting:
                self.current_subtask = self.subtasks[k]
                rospy.loginfo(f'🔄 Subtask → {self.current_subtask}')
            elif k == 'q':
                self._finalize_and_exit()
                return

            rospy.sleep(0.05)

    # ─────────────────────── Finalize ────────────────────────────────────
    def _teardown_view(self):
        """Stop the live OpenCV view. MUST run on the main thread — driving the
        cv2/Qt GUI from a background thread deadlocks (the same reason rendering
        is pumped from spin()). Idempotent."""
        view = self._view
        if view is not None:
            self._view = None
            try:
                view.stop()
            except Exception:
                pass

    def _safe_finalize(self):
        """Idempotent dataset finalize — safe to call from any thread / shutdown
        hook. Does NOT touch the cv2 view (see _teardown_view, main-thread only)."""
        with self._finalize_lock:
            if self._finalized:
                return
            self._finalized = True
        try:
            rospy.loginfo('📦 Finalizing dataset …')
            if self.dataset is not None:
                if self.dataset.image_writer is not None:
                    self.dataset.stop_image_writer()
                self.dataset.finalize()
            rospy.loginfo(f'✅ Dataset finalized at {self.cfg.target_dir}')
        except Exception as e:
            # Last resort: at least try to print the error
            sys.stderr.write(f'[data_to_lerobot3] finalize error: {e}\n')

    def _signal_handler(self, signum, _frame):
        """Handle SIGTERM/SIGINT: request shutdown only. The heavy teardown
        (view + dataset finalize) runs on the main thread via spin()/main()."""
        rospy.signal_shutdown(f'Received signal {signum}')

    def _finalize_and_exit(self):
        """Request a clean shutdown. View teardown + dataset finalize happen on
        the main thread (spin()/main()) so the cv2 GUI is never driven from a
        background thread."""
        rospy.signal_shutdown('User requested exit')

    # ─────────────────────── Usage Banner ────────────────────────────────
    def _print_usage(self):
        lines = [
            '═══════════════════════════════════════════',
            f'  LeRobot v3 Direct Collection Node — {self.cfg.mode} mode',
            '═══════════════════════════════════════════',
            '  c     → Start / resume collecting',
            '  space → Pause',
        ]
        for key, desc in self.subtasks.items():
            lines.append(f'  {key}     → Subtask: {desc}')
        if self.cfg.mode == 'recovery':
            lines += ['  i     → Save as intervention episode',
                      '  f     → Save as failure episode']
        else:
            lines.append('  s     → Save episode')
        lines += [
            '  r     → Discard buffer',
            '  q     → Finalize dataset & exit',
            '═══════════════════════════════════════════',
        ]
        rospy.loginfo('\033[32m' + '\n'.join(lines) + '\033[0m')

    # ─────────────────────── Main-Thread Spin ────────────────────────────
    def spin(self):
        """Block the main thread until shutdown.

        When the live view is enabled we drive its rendering here, because
        OpenCV's GUI must be pumped from the main thread (a background GUI
        thread silently fails to display under Anaconda's Qt). Subscriber and
        command threads keep running independently. Falls back to plain
        rospy.spin() when the view is disabled/unavailable.
        """
        try:
            if self._view is None or not self._view.enabled:
                rospy.spin()
                return
            rate = rospy.Rate(15)  # GUI refresh rate
            while not rospy.is_shutdown():
                # While NOT collecting, keep the view live with the latest camera
                # frames + current subtask so the operator can frame the scene and
                # pick a subtask before pressing 'c'. During collection, _collect()
                # already feeds the view with the synced frames it records, so we
                # leave it alone to avoid overwriting the recording status.
                if not self.collecting:
                    self._view.update(self._peek_latest_images(), self._build_idle_status())
                if not self._view.render_once():  # user pressed q/ESC in the window
                    rospy.signal_shutdown('User pressed q/ESC in view')
                    break
                rate.sleep()
        finally:
            # Teardown on the MAIN thread: stop the cv2 view first (GUI calls are
            # main-thread-only), then finalize the dataset. Runs no matter how the
            # loop ends (q in view, q via keyboard, signal, max_timesteps).
            self._teardown_view()
            self._safe_finalize()

    # ─────────────────────── Live-View Helpers ───────────────────────────
    def _peek_latest_images(self):
        """Latest frame from each camera queue for live preview (non-destructive).

        Reads (without popping) the newest ROS image in each camera deque and
        converts it to a cv2 array, so the preview never consumes frames the
        collector needs. Returns an empty dict until cameras start publishing.
        """
        images = {}
        for cam in self.cfg.camera_names:
            q = self.queues.get(cam)
            if q:
                try:
                    images[cam] = imgmsg_to_cv2(q[-1])
                except Exception:
                    pass
        return images

    def _buffered_frames(self):
        """Number of frames currently in the LeRobot episode buffer.

        Read defensively: LeRobotDataset.save_episode() pops the 'size' key from
        episode_buffer, and the buffer can be None before the first frame, so a
        direct episode_buffer['size'] access can raise KeyError/TypeError between
        episodes. Treat any missing/None buffer as 0 buffered frames.
        """
        if self.dataset is None:
            return 0
        buf = getattr(self.dataset, 'episode_buffer', None)
        if not buf:
            return 0
        return buf.get('size', 0)

    def _build_idle_status(self):
        """Status overlay shown when not actively collecting (recording=False)."""
        buffered = self._buffered_frames()
        total_eps = self.dataset.meta.total_episodes if self.dataset is not None else 0
        total_frames = self.dataset.meta.total_frames if self.dataset is not None else 0
        return {
            'task': self.cfg.task_name,
            'skill_text': self.current_subtask,
            'recording': False,
            'mode': self.cfg.mode,
            'episode_index': total_eps,
            'buffered_frames': buffered,
            'saved_episodes': total_eps,
            'saved_frames': total_frames,
        }


def main():
    try:
        rospy.init_node('data_to_lerobot3_node', anonymous=True)
        node = LeRobotCollectorNode()
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down LeRobot collection node.')


if __name__ == '__main__':
    main()