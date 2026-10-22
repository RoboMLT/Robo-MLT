cmake_minimum_required(VERSION 3.0.2)
project(dual_piper)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs
)

# ── Install Python nodes (+ the utils package they import from their own directory) ──
catkin_install_python(PROGRAMS
  scripts/state_get_node.py
  scripts/data_replay_node.py
  scripts/data_to_lerobot3_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY scripts/utils
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# ── Install launch and config files ──
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
