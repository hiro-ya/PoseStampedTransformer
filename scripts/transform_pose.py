#!/usr/bin/env python
# coding: utf-8

import signal
from copy import deepcopy
from threading import Lock
from math import modf

import numpy as np

import rospy
import rospkg
from geometry_msgs.msg import PoseStamped

from posetf import PoseTF


class Transformer(object):
    def __init__(self, pub_frame_id='/world', pose_smoother_queue_size=5):
        self._pub_frame_id = pub_frame_id
        self._pose_smoother_queue_size = pose_smoother_queue_size
        self._transform_factors = None

        self._lock_transformed_pose_array = Lock()
        self._transformed_pose_array = []

    def load(self, transform_factors_file_path):
        self._transform_factors = PoseTF.load_transform_factors(transform_factors_file_path)


    @staticmethod
    def rosstamp_to_timestamp(rosstamp):
        return rosstamp.secs + 1e-9 * rosstamp.nsecs

    @staticmethod
    def timestamp_to_rosstamp(timestamp):
        secs_f, secs = modf(timestamp)
        return rospy.Time(int(secs), int(1e+9*secs_f))

    def pose_to_message(self, pose):
        data = PoseStamped()
        data.header.frame_id = self._pub_frame_id
        data.header.stamp = Transformer.timestamp_to_rosstamp(pose[0])
        data.pose.position.x = pose[1]
        data.pose.position.y = pose[2]
        data.pose.position.z = pose[3]
        data.pose.orientation.w = pose[4]
        data.pose.orientation.x = pose[5]
        data.pose.orientation.y = pose[6]
        data.pose.orientation.z = pose[7]
        return data

    @staticmethod
    def message_to_pose(data):
        return [
            Transformer.rosstamp_to_timestamp(data.header.stamp),
            data.pose.position.x,
            data.pose.position.y,
            data.pose.position.z,
            data.pose.orientation.w,
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z
        ]

    def callback_on_pose_data(self, data):
        pose = Transformer.message_to_pose(data)
        pose_transformed = PoseTF.transform_pose(pose, self._transform_factors)
        with self._lock_transformed_pose_array:
            self._transformed_pose_array.append(pose_transformed)
            if self._pose_smoother_queue_size < len(self._transformed_pose_array):
                self._transformed_pose_array.pop(0)

    def generate_estimated_pose(self):
        with self._lock_transformed_pose_array:
            pose_array = deepcopy(self._transformed_pose_array)
        if len(pose_array) < self._pose_smoother_queue_size:
            return None
        current_time = Transformer.rosstamp_to_timestamp(rospy.Time.now())
        poses = list(map(PoseTF.deserialize_pose, pose_array))
        predict_positions = list(map(lambda x: x[1] + (poses[-1][1] - x[1]) * (current_time - x[0]), poses))
        predict_rotations = list(map(lambda x: (x[2] + (poses[-1][2] - x[2]) * (current_time - x[0])).elements, poses))
        position = np.median(predict_positions, axis=0)
        rotation = np.median(predict_rotations, axis=0)
        return self.pose_to_message([current_time] + position.tolist() + rotation.tolist())


if __name__ == '__main__':
    """
    subscribe pose and publish reference map matched pose
    """

    rospy.init_node('transform_pose')

    sub_pose_topic = rospy.get_param('~sub_pose_topic')
    pub_pose_topic = rospy.get_param('~pub_pose_topic')
    hz = float(rospy.get_param('~hz'))
    transform_factors_file_path = rospy.get_param('~transform_factors_file_path')
    if transform_factors_file_path == '':
        transform_factors_file_path = rospkg.RosPack().get_path('posestamped_transformer')+'/resources/transform_factors.csv'

    transformer = Transformer(pub_frame_id='/map')
    transformer.load(transform_factors_file_path)

    pose_estimated_publisher = rospy.Publisher(
        name=pub_pose_topic, data_class=PoseStamped, subscriber_listener=None,
        tcp_nodelay=False, latch=False, headers=None, queue_size=0)

    rospy.Subscriber(sub_pose_topic, PoseStamped, transformer.callback_on_pose_data)

    def signal_handler(_signal, _frame):
        rospy.signal_shutdown('finish')
        rospy.spin()

    signal.signal(signal.SIGINT, signal_handler)

    r = rospy.Rate(hz)
    while not rospy.is_shutdown():
        pose_estimated = transformer.generate_estimated_pose()
        print(pose_estimated)
        if pose_estimated is not None:
            pose_estimated_publisher.publish(pose_estimated)
        r.sleep()

