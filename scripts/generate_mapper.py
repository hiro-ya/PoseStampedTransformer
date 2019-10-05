#!/usr/bin/env python
# coding: utf-8

import signal
from argparse import ArgumentParser
from multiprocessing import Lock
from copy import copy
from bisect import bisect

import numpy as np

import rospy
import rospkg
import message_filters
from geometry_msgs.msg import PoseStamped

from posetf import PoseTF


class TFMapper(object):
    def __init__(self, position_tolerance=1.0, time_tolerance=1e-2):
        self._position_tolerance = position_tolerance
        self._time_tolerance = time_tolerance

        self._lock_poses = Lock()
        self._poses = []

        self._lock_ref_poses = Lock()
        self._ref_poses = []

        self._lock_pose_pairs = Lock()
        self._pose_pairs = []

        self._lock_transform_factors = Lock()
        self._transform_factors = []

    def callback_on_pose_pair_data(self, data, ref_data):
        pose_pair = [[
              [data.pose.position.x, data.pose.position.y, data.pose.position.z],
              [data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z]
            ],
            [
              [ref_data.pose.position.x, ref_data.pose.position.y, ref_data.pose.position.z],
              [ref_data.pose.orientation.w, ref_data.pose.orientation.x, ref_data.pose.orientation.y, ref_data.pose.orientation.z]
        ]]
        with self._lock_pose_pairs:
            if 0 == len(self._pose_pairs):
                self._pose_pairs.append(pose_pair)
            else:
                d = np.linalg.norm(np.array(pose_pair[1][0])-np.array(self._pose_pairs[-1][1][0]))
                if self._position_tolerance < d:
                    self._pose_pairs.append(pose_pair)

    def update_transform_factors(self):
        with self._lock_pose_pairs:
            pose_pairs = copy(self._pose_pairs)

        if len(pose_pairs) < 2:
            return

        with self._lock_transform_factors:
            self._transform_factors.extend(PoseTF.generate_transform_factors(pose_pairs))

        with self._lock_pose_pairs:
            self._pose_pairs = self._pose_pairs[bisect(self._pose_pairs, pose_pairs[-2]):]


    def get_transform_factors(self):
        with self._lock_transform_factors:
            transform_factors = copy(self._transform_factors)
        return transform_factors


if __name__ == '__main__':
    """
    subscribe pose and reference pose and save transform factors.
    """

    rospy.init_node('generate_mapper')

    ref_pose_topic = rospy.get_param('~ref_pose_topic')
    pose_topic = rospy.get_param('~pose_topic')
    hz = float(rospy.get_param('~hz'))
    time_tolerance = float(rospy.get_param('~time_tolerance'))
    position_tolerance = float(rospy.get_param('~position_tolerance'))
    transform_factors_file_path = rospy.get_param('~transform_factors_file_path')
    if transform_factors_file_path == '':
        transform_factors_file_path = rospkg.RosPack().get_path('posestamped_transformer')+'/resources/transform_factors.csv'

    tf_mapper = TFMapper(position_tolerance=position_tolerance, time_tolerance=time_tolerance)

    sub_ref_pose = message_filters.Subscriber(ref_pose_topic, PoseStamped)
    sub_pose = message_filters.Subscriber(pose_topic, PoseStamped)
    poses_message_filter = message_filters.ApproximateTimeSynchronizer([sub_pose, sub_ref_pose], queue_size=10, slop=time_tolerance)
    poses_message_filter.registerCallback(tf_mapper.callback_on_pose_pair_data)

    def signal_handler(_signal, _frame):
        ret = PoseTF.dump_transform_factors(transform_factors_file_path, tf_mapper.get_transform_factors())
        if ret:
            rospy.loginfo('Transform factors was saved to [{}].'.format(transform_factors_file_path))
        else:
            rospy.logwarn('Transform factors was not saved.')
        rospy.signal_shutdown('finish')
        rospy.spin()

    signal.signal(signal.SIGINT, signal_handler)

    r = rospy.Rate(hz)
    while not rospy.is_shutdown():
        tf_mapper.update_transform_factors()
        r.sleep()

