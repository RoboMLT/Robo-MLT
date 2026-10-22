# Robo-MLT Data Collection

ROS Noetic workspace for recording the Robo-MLT laboratory datasets on a dual-arm AgileX Piper rig
with three RealSense cameras. Teleoperated episodes are written straight into LeRobot v3 datasets
with online atomic-skill annotation.

## Contents

```
scripts/                       CAN bring-up helpers (from the AgileX SDK)
docs/data_collection_pipeline.md   the full recording protocol
src/dual_piper/
  config/                      one config per task and corpus:
                               blood_gas_analysis, tube_sorting (+ *_recovery)
  launch/teleoperation.launch  arm state publisher + three RealSense cameras
  launch/collect.launch        LeRobot v3 recorder for a given config
  scripts/state_get_node.py    CAN -> observation.state / action topics (200 Hz)
  scripts/data_to_lerobot3_node.py   recorder (online skill annotation, recovery tagging)
  scripts/data_replay_node.py  plot / video / robot replay of an episode
  tools/label_recovery_tail.py label the final K_rec ~ U[20, 40] frames of recovery episodes
  tools/check_dataset.py       sanity checks of a recorded dataset
  tools/trim_lerobotv3_episodes.py, tools/dualpiper_to_singlepiper.py
src/piper*, src/piper_msgs     AgileX Piper ROS packages (description, MoveIt, messages)
```

## Quick start

```bash
git clone -b data-collection https://github.com/RoboMLT/Robo-MLT.git robomlt_ws && cd robomlt_ws

# environment (see docs for details): Python >= 3.10 with lerobot, able to import rospy
pip install -r requirements.txt
source /opt/ros/noetic/setup.bash && catkin_make && source devel/setup.bash

# CAN adapters -> can_left / can_right
sudo CAN_LEFT_USB=<bus-info> CAN_RIGHT_USB=<bus-info> bash scripts/can_config.sh

# arms + cameras
roslaunch dual_piper teleoperation.launch serial_top:=<sn> serial_left:=<sn> serial_right:=<sn>

# nominal corpus: complete workflows and atomic-skill episodes
roslaunch dual_piper collect.launch config:=blood_gas_analysis

# recovery corpus: interventions (save with i) and failures (save with f)
roslaunch dual_piper collect.launch config:=blood_gas_analysis_recovery
python src/dual_piper/tools/label_recovery_tail.py --root ~/robomlt_data/blood_gas_analysis_recovery

# checks
python src/dual_piper/tools/check_dataset.py --root ~/robomlt_data/blood_gas_analysis
python src/dual_piper/tools/test_tools.py
```

Recorder keys: `1`–`N` select the active skill (while paused), `c` record, `space` pause, `s` save
(nominal) or `i` / `f` save as intervention / failure (recovery), `r` discard, `q` finalize and exit.

## Dataset format

LeRobot v3.0 at 25 fps with `observation.state` and `action` (14 = 2 × (6 joints + gripper)),
three RGB videos (`cam_top`, `cam_left`, `cam_right`), `subtask_index` (active atomic skill, text in
`meta/subtasks.parquet`) and `back_event` (recovery-head label). In the released datasets `action`
is the measured follower joint position, identical to `observation.state` at the same step; see
`docs/data_collection_pipeline.md` §4 and the `action_source` option of `state_get_node.py`.

## License

The `dual_piper` package and the tools are released under the MIT License (`LICENSE`). The Piper
description, MoveIt, message and control packages and the CAN scripts come from AgileX Robotics;
see `THIRD_PARTY_NOTICES.md`.
