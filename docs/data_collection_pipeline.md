# Data-collection pipeline

This document walks through how the Robo-MLT datasets are recorded: from bringing up the dual-arm
AgileX Piper rig to a release-ready LeRobot v3 dataset with skill annotations and recovery labels.
It follows the protocol of the paper (Sec. IV-A-3).

## 1. Hardware

| Item | Details |
|---|---|
| Arms | Two AgileX Piper leader–follower pairs (left, right). Each pair shares one USB-CAN adapter. Leader→follower following runs in the arm firmware; the software only reads the bus. |
| Cameras | Three Intel RealSense D435i, colour only, 640×480: one fixed top view of the workspace (`cam_top`) and one on each wrist (`cam_left`, `cam_right`). |
| Host | Ubuntu 20.04, ROS Noetic, USB 3 ports for the cameras, an NVIDIA GPU for NVENC video encoding (or use `video_codec: libsvtav1` without one). |

## 2. Software environment

The nodes are plain Python ROS nodes but need `lerobot` (Python ≥ 3.10), newer than the Python 3.8
of ROS Noetic. Use a separate environment that can also import `rospy`:

```bash
conda create -n robomlt-collect python=3.10 -y && conda activate robomlt-collect
pip install -r requirements.txt            # lerobot, piper_sdk, pynput, opencv, ...
pip install rospkg catkin_pkg              # lets this interpreter import the Noetic Python packages
source /opt/ros/noetic/setup.bash          # in every terminal, before running a node
sudo apt install ros-noetic-realsense2-camera can-utils ethtool
```

Build the workspace once and source it:

```bash
cd <this repository>
catkin_make
source devel/setup.bash
```

The nodes carry a `#!/usr/bin/env python3` shebang, so `rosrun` uses the `python3` of the active
environment.

## 3. CAN bring-up

```bash
bash scripts/find_all_can_port.sh                                  # note each adapter's bus-info
sudo CAN_LEFT_USB=<bus-info> CAN_RIGHT_USB=<bus-info> bash scripts/can_config.sh
```

`can_config.sh` renames the two adapters to `can_left` / `can_right` and brings them up at 1 Mbit/s.
Run it after every reboot or re-plug.

## 4. Teleoperation and cameras

```bash
rs-enumerate-devices -s                                            # camera serial numbers
roslaunch dual_piper teleoperation.launch \
    serial_top:=<serial> serial_left:=<serial> serial_right:=<serial>
```

This starts `state_get_node.py` (200 Hz) and the three camera drivers. Per arm the state node
publishes `<side>/slave_joint_states` (recorded as `observation.state`) and
`<side>/master_joint_states` (recorded as `action`); `position` holds six joint angles in radians
and the gripper opening in metres.

`action_source:=follower_feedback` (default) is the setting the released datasets were recorded
with: the joints are the measured follower angles, so `action[t]` equals `observation.state[t]` and
a policy trained on action chunks learns to predict future joint positions. In both topics the
gripper value is the commanded opening. `action_source:=leader_command` records the leader's joint
commands as the action instead.

## 5. Recording workflows (nominal corpus)

```bash
roslaunch dual_piper collect.launch config:=blood_gas_analysis     # or tube_sorting
```

Keys (read globally, also when the terminal is not focused):

| Key | Effect |
|---|---|
| `1`…`N` | set the active atomic skill (while paused) |
| `c` | start / resume recording |
| `space` | pause |
| `s` | save the episode |
| `r` | discard the buffered frames |
| `q` | finalize the dataset and exit |

A **complete workflow** is recorded as one episode with the skill boundaries annotated online:
select skill 1, press `c`, perform the skill, press `space` at its end, select the next skill,
press `c`, and so on; press `s` after the last skill. Every frame carries the `subtask_index` of the
skill that was active while it was recorded.

For blood gas analysis one workflow processes the samples on the rack in order (grasp → remove cap →
discard cap → dock → insert → withdraw → discard sample, per sample), then returns home. For tube
sorting it grasps each tube with the skill of its category and places it into the plate assigned by
the task instruction, then returns home.

**Atomic-skill episodes** contain one skill each, started from a randomised initial pose of the
arms and objects. They broaden the state coverage of every skill. Select the skill, `c`, perform
it, `space`, `s`.

The live window shows the camera feeds, the active skill in large letters, the recording state and
the dataset counters.

## 6. Recording recovery episodes (recovery corpus)

Positive supervision for the recovery head comes from episodes that end because the active skill
can no longer be completed. They are recorded into their own dataset and never mixed into the
nominal corpus:

```bash
roslaunch dual_piper collect.launch config:=blood_gas_analysis_recovery
```

Two sources, recorded in the same sessions as the nominal data:

1. **Deliberate intervention.** While the robot is teleoperated through the workflow, a second
   person removes the sample from the gripper — while it is delivered to the analyzer (blood gas
   analysis) or while it is being placed (tube sorting). Pause right after the removal (`space`)
   and save with **`i`**.
2. **Spontaneous teleoperation failure.** When the operator mishandles a sample during collection —
   most often a tube dropped on the way to the analyzer — pause immediately and save with **`f`**
   instead of discarding the episode.

In this mode `s` is disabled. Each saved episode is tagged in `meta/recovery_episodes.json` with its
type and the skill that was active when it ended. Keep the active skill annotated as usual.

After recording, label the positives offline: the final K_rec frames of each episode get
`back_event = 1`, with K_rec drawn uniformly from [20, 40] independently per episode:

```bash
python src/dual_piper/tools/label_recovery_tail.py \
    --root ~/robomlt_data/blood_gas_analysis_recovery --k-min 20 --k-max 40 --seed 0
```

The draw is reproducible from the seed and recorded in `meta/recovery_labels.json`.

## 7. Post-processing and checks

| Step | Command |
|---|---|
| Crash repair | automatic: resuming a dataset runs `repair_dataset_consistency` (drops the half-written tail of an interrupted session) |
| Drop leading warm-up episodes | `python src/dual_piper/tools/trim_lerobotv3_episodes.py --dataset <root> --drop-first N` |
| Label recovery positives | `python src/dual_piper/tools/label_recovery_tail.py --root <root>_recovery` |
| Sanity check | `python src/dual_piper/tools/check_dataset.py --root <root>` |
| Replay / inspect an episode | `rosrun dual_piper data_replay_node.py --dataset_dir <root> --episode_idx 0 [--replay_on_robot]` |
| Release (dedupe, merge sessions) | `python -m high_level_model.data.build_release_dataset` in the Robo-MLT code release |
| Report the counts | `python -m high_level_model.data.dataset_stats` in the Robo-MLT code release |

`check_dataset.py` reports timestamp jitter, the action/state relation, skill segments per episode,
unused skills, and whether the recovery positives form the final block of each episode.
`dataset_stats.py` (code release) reports the figures quoted in the paper: distinct episodes,
complete workflows vs. atomic-skill episodes, recovery episodes by type, and a byte-for-byte check
of the subtask text against the skill library.

## 8. Dataset schema

LeRobot v3.0, 25 fps, `robot_type: dual_piper`:

| Key | Type | Content |
|---|---|---|
| `observation.state` | float32 (14) | per arm, left first: 6 measured joint angles (rad) + commanded gripper opening (m) |
| `action` | float32 (14) | same layout; see Sec. 4 for what it contains |
| `observation.images.cam_top` / `cam_left` / `cam_right` | video 480×640×3 | RGB |
| `subtask_index` | int64 | active atomic skill; text in `meta/subtasks.parquet` |
| `back_event` | int64 | recovery-head label (0 in nominal data; final K_rec frames of each recovery episode) |
| `task_index` (+ `meta/tasks.parquet`) | int64 | high-level task instruction |

Extra metadata of recovery datasets: `meta/recovery_episodes.json` (episode → type, last skill,
frames) and `meta/recovery_labels.json` (k_min, k_max, seed, episode → K_rec).

## 9. How the corpora are used for training

| Corpus | System 1 and baselines | System 2 (execution monitor) |
|---|---|---|
| nominal: complete workflows | ✓ | ✓ completion labels |
| nominal: atomic-skill episodes | ✓ | ✓ completion labels |
| recovery: intervention + failure | – | ✓ recovery labels |

## 10. Known issues

- Keys are captured globally by `pynput`; a key pressed in another window is also received.
  Two keys pressed within ~50 ms can drop one.
- LeRobot timestamps are `frame_index / fps`; if the host cannot keep up with 25 fps the recorded
  time base is compressed. `check_dataset.py` reports the jitter of the stored timestamps only.
- The collector relies on non-public LeRobot APIs (`_flush_metadata_buffer`, `_close_writer`); use
  the pinned version in `requirements.txt`.
