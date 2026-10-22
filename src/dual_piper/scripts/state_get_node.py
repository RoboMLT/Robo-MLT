#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-arm Piper state publisher.

Each leader–follower pair shares one CAN bus, so a single ``C_PiperInterface`` per side sees both
the follower's feedback frames and the leader's command frames. At 200 Hz this node publishes,
per side, two ``sensor_msgs/JointState`` topics that the collector records:

    <side>/slave_joint_states   -> observation.state
    <side>/master_joint_states  -> action

``position`` holds the six joint angles (rad) and the gripper opening (m) in slot 7; slot 8 is 0.

What ``action`` contains is selected by ``~action_source``:

    follower_feedback (default)  joints from ``GetArmJointMsgs`` (measured follower angles) — the
                                 setting the released Robo-MLT datasets were recorded with, where
                                 ``action[t]`` equals ``observation.state[t]`` and a policy learns
                                 to predict *future* joint positions.
    leader_command               joints from ``GetArmJointCtrl`` (the leader's command frames).

In both cases the gripper value is the commanded opening from ``GetArmGripperCtrl``, for the state
and for the action alike (again matching the released data).

Usage::

    rosrun dual_piper state_get_node.py _left_can_port:=can_left _right_can_port:=can_right
"""
import os
import sys

# Make the sibling ``utils`` package importable regardless of the working directory
# used by rosrun / roslaunch (catkin installs scripts through a relay).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import threading
import time

import rospy
from piper_sdk import C_PiperInterface
from sensor_msgs.msg import JointState

from utils import (
    DEG_TO_RAD,
    EFFORT_SCALE,
    ENABLE_ALL_JOINTS,
    GRIPPER_SCALE,
    JOINT_SCALE,
    PUBLISH_RATE_HZ,
)

_JOINT_NAMES = [f'joint{i}' for i in range(1, 9)]
_NUM_JOINTS = len(_JOINT_NAMES)
ACTION_SOURCES = ('follower_feedback', 'leader_command')


def _joint_angles(piper, source):
    """Six joint angles in radians from the follower feedback or the leader command frames."""
    if source == 'leader_command':
        joint = piper.GetArmJointCtrl().joint_ctrl
    else:
        joint = piper.GetArmJointMsgs().joint_state
    return [getattr(joint, f'joint_{i}') / JOINT_SCALE * DEG_TO_RAD for i in range(1, 7)]


class DualPiperStateNode:
    """ROS node that publishes the dual-arm Piper state / action topics."""

    def __init__(self):
        rospy.init_node('dual_piper_state_get_node', anonymous=True)

        self._can_left = rospy.get_param('~left_can_port', 'can_left')
        self._can_right = rospy.get_param('~right_can_port', 'can_right')
        self._auto_enable = rospy.get_param('~auto_enable', True)
        self._action_source = rospy.get_param('~action_source', 'follower_feedback')
        if self._action_source not in ACTION_SOURCES:
            raise ValueError(f"~action_source must be one of {ACTION_SOURCES}, "
                             f"got '{self._action_source}'")

        rospy.loginfo("Left  CAN port : %s", self._can_left)
        rospy.loginfo("Right CAN port : %s", self._can_right)
        rospy.loginfo("Auto enable    : %s", self._auto_enable)
        rospy.loginfo("Action source  : %s", self._action_source)

        self._piper_left = self._connect_arm('Left', self._can_left)
        self._piper_right = self._connect_arm('Right', self._can_right)

        self._pubs = {
            (side, role): rospy.Publisher(f'{side}/{role}_joint_states', JointState, queue_size=10)
            for side in ('left', 'right') for role in ('master', 'slave')
        }
        self._msgs = {key: self._make_joint_msg() for key in self._pubs}

        self._enabled_left = False
        self._enabled_right = False
        if self._auto_enable:
            self._enable_dual_arms()

        threading.Thread(target=self._publish_loop, daemon=True).start()

    # ── Helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _connect_arm(label, can_port):
        """Instantiate and connect a ``C_PiperInterface``."""
        rospy.loginfo("Connecting to %s arm on %s …", label, can_port)
        iface = C_PiperInterface(can_name=can_port)
        iface.ConnectPort()
        rospy.loginfo("%s arm connected.", label)
        return iface

    @staticmethod
    def _make_joint_msg():
        msg = JointState()
        msg.name = list(_JOINT_NAMES)
        msg.position = [0.0] * _NUM_JOINTS
        msg.velocity = [0.0] * _NUM_JOINTS
        msg.effort = [0.0] * _NUM_JOINTS
        return msg

    @staticmethod
    def _check_arm_enabled(piper):
        """``True`` if all six motors report ``driver_enable_status``."""
        status = piper.GetArmLowSpdInfoMsgs()
        return all(
            getattr(status, f'motor_{i}').foc_status.driver_enable_status
            for i in range(1, 7)
        )

    # ── Enable ───────────────────────────────────────────────────────────
    def _enable_dual_arms(self):
        """Block until both arms are enabled (10 s timeout)."""
        rospy.loginfo("Enabling dual arms …")
        t0 = time.time()
        while not (self._enabled_left and self._enabled_right):
            if rospy.is_shutdown():
                return
            if time.time() - t0 > 10.0:
                rospy.logerr("Timeout while enabling arms — exiting.")
                sys.exit(1)
            if not self._enabled_left:
                if self._check_arm_enabled(self._piper_left):
                    self._enabled_left = True
                    rospy.loginfo("Left arm enabled.")
                else:
                    self._piper_left.EnableArm(ENABLE_ALL_JOINTS)
            if not self._enabled_right:
                if self._check_arm_enabled(self._piper_right):
                    self._enabled_right = True
                    rospy.loginfo("Right arm enabled.")
                else:
                    self._piper_right.EnableArm(ENABLE_ALL_JOINTS)
            time.sleep(1)

    # ── Publishing ───────────────────────────────────────────────────────
    def _publish_loop(self):
        rate = rospy.Rate(PUBLISH_RATE_HZ)
        while not rospy.is_shutdown():
            stamp = rospy.Time.now()
            self._publish_arm('left', self._piper_left, stamp)
            self._publish_arm('right', self._piper_right, stamp)
            rate.sleep()

    def _publish_arm(self, side, piper, stamp):
        """Read one arm over its CAN bus and publish its state (slave) and action (master)."""
        try:
            gripper = piper.GetArmGripperCtrl().gripper_ctrl
            speed = piper.GetArmHighSpdInfoMsgs()
            velocity = [getattr(speed, f'motor_{i}').motor_speed / EFFORT_SCALE for i in range(1, 7)]
            velocity += [0.0, 0.0]
            effort = [0.0] * _NUM_JOINTS
            effort[6] = gripper.grippers_effort / EFFORT_SCALE
            opening = gripper.grippers_angle / GRIPPER_SCALE

            for role, source in (('slave', 'follower_feedback'), ('master', self._action_source)):
                msg = self._msgs[(side, role)]
                msg.header.stamp = stamp
                msg.position = _joint_angles(piper, source) + [opening, 0.0]
                msg.velocity = list(velocity)
                msg.effort = list(effort)
                self._pubs[(side, role)].publish(msg)
        except Exception as e:  # noqa: BLE001 - a bad CAN read must not kill the 200 Hz loop
            rospy.logerr("Error reading/publishing %s arm state: %s", side, e)


def main():
    try:
        DualPiperStateNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down dual_piper state node.")


if __name__ == '__main__':
    main()
