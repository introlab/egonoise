#!/usr/bin/env python3

import rosbag
import numpy as np

from audio_utils import get_format_information, convert_audio_data_to_numpy_frames, convert_numpy_frames_to_audio_data

from audio_utils import get_format_information, convert_audio_data_to_numpy_frames
from audio_utils.msg import AudioFrame
import kissdsp.io as io

bag_names = [
    'AL1',
    'AL2',
    'AL11',
    'AL15',
    'AL21',
    'AL24',
    'AL3',
    'AL4',
    'AL5',
    'AL12',
    'AL14',
    'AL22',
    'AL23',
    'AL25',
    '237-126133-0000',
    '237-126133-0001',
    '237-126133-0003',
    '237-126133-0005',
    '237-126133-0023',
    '237-126133-0024',
    '260-123286-0003',
    '260-123286-0006',
    '260-123286-0016',
    '260-123286-0026',
    '260-123286-0027',
    '260-123286-0028',
    '1089-134686-0000',
    '1089-134686-0002',
    '1089-134686-0005',
    '1089-134686-0011',
    '1089-134686-0012',
    '1089-134686-0013',
    '1188-133604-0003',
    '1188-133604-0004',
    '1188-133604-0007',
    '1188-133604-0018',
    '1188-133604-0019',
    '1188-133604-0021',
    '3570-5694-0000',
    '3570-5694-0005',
    '3570-5694-0006',
    '3570-5694-0007',
    '3570-5694-0008',
    '3570-5694-0011',
    '4992-23283-0005',
    '4992-23283-0009',
    '4992-23283-0017',
    '4992-23283-0020',
    '4992-41797-0008',
    '4992-41797-0009'
]

for bag_name in bag_names:
    bag_path = f'/home/pierre-olivier/catkin_ws/src/bag/article/1004/{bag_name}.bag'
    bag_path_out = f'/home/pierre-olivier/catkin_ws/src/bag/article/1004-100/{bag_name}.bag'
    sf = 32000
    newL = 1.0

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

    bag = rosbag.Bag(bag_path_out, 'w')
    for i, frames in enumerate(new_frames_list):
        data = convert_numpy_frames_to_audio_data(get_format_information(msg.format), frames)

        amsg = AudioFrame()

        amsg.header.seq = i

        amsg.format = msg.format
        amsg.channel_count = msg.channel_count
        amsg.sampling_frequency = msg.sampling_frequency
        amsg.frame_sample_count = newStep
        amsg.data = data

        bag.write('audio_out', amsg)

    bag.close()




