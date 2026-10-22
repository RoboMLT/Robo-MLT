# Third-Party Notices

The `dual_piper` package (`src/dual_piper/`) and the documentation are original to this
repository and released under the MIT License (`LICENSE`). The following parts come from other
projects; verify each upstream license before further redistribution.

## AgileX Robotics `piper_ros` (ROS Noetic branch)

`src/piper/`, `src/piper_msgs/`, `src/piper_description/`, `src/piper_moveit/` and
`src/piper_noetic/` are copied from [agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros)
(MIT License on GitHub at the time of writing). Local changes: `piper/scripts/piper_ctrl_single_node.py`
imports `quaternion_from_euler` from `tf2_ros.transformations`, and
`piper_description/launch/piper_with_gripper/display_xacro.launch` remaps the robot state publisher
to `/left/slave_joint_states` and disables the joint-state GUI.

## AgileX Robotics `piper_sdk`

`scripts/can_config.sh` and `scripts/find_all_can_port.sh` are adapted from the CAN helper scripts
of [agilexrobotics/piper_sdk](https://github.com/agilexrobotics/piper_sdk) (MIT License). The
`piper_sdk` Python package itself is a pip dependency and is not vendored.

## `src/piper_x_description/`

URDF and meshes of the PiPER-X arm, distributed by AgileX Robotics. Confirm the license of the
source package before redistribution.

## LeRobot

The recorder writes datasets with [LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0),
a pip dependency.
