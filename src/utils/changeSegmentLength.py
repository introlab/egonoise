#!/usr/bin/env python3

import rosbag
import rospy
import numpy as np

from time import time

from audio_utils import get_format_information, convert_audio_data_to_numpy_frames, convert_numpy_frames_to_audio_data

from audio_utils import get_format_information, convert_audio_data_to_numpy_frames
from audio_utils.msg import AudioFrame
import kissdsp.io as io

bag_names = [
    # 'AL21',
    # 'AL23',
    # 'AL22',
    'mix'
]

for bag_name in bag_names:
    bag_path = f'/home/pierre-olivier/catkin_ws/src/bag/article/1004/{bag_name}.bag'
    bag_path_out = f'/home/pierre-olivier/catkin_ws/src/bag/article/1004-008/{bag_name}.bag'
    sf = 32000
    newL = 0.008

    frames_list = []

    for topic, msg, _ in rosbag.Bag(bag_path).read_messages():
        frames = np.array(convert_audio_data_to_numpy_frames(get_format_information(msg.format),
                                                                    msg.channel_count, msg.data))
        frames_list.append(frames)

    frames_list = np.hstack(frames_list)

    newStep = int(sf*newL)
    newNData = int(frames_list.shape[1]/newStep)

    new_frames_list= []
    for i in range(newNData):
        new_frames_list.append(frames_list[:, (i*newStep):((i+1)*newStep)])

    new_frames_list = np.array(new_frames_list)

    data = []
    for i, frames in enumerate(new_frames_list):
        data.append(convert_numpy_frames_to_audio_data(get_format_information(msg.format), frames))

    rospy.init_node('allo')
    bag = rosbag.Bag(bag_path_out, 'w')
    rate = rospy.Rate(int(32000 / 256))

    for i, d in enumerate(data):
        amsg = AudioFrame()

        amsg.header.seq = i
        amsg.format = msg.format
        amsg.channel_count = msg.channel_count
        amsg.sampling_frequency = msg.sampling_frequency
        amsg.frame_sample_count = newStep
        amsg.data = d

        bag.write('audio_out', amsg)

        rate.sleep()

    bag.close()




