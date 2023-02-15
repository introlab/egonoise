#!/usr/bin/env python3

import rosbag
import numpy as np

from audio_utils import get_format_information, convert_audio_data_to_numpy_frames
from audio_utils.msg import AudioFrame
import kissdsp.io as io

bag_names = [
    'MUSIC1',
    'MUSIC2',
    'MUSIC3',
    'MUSIC4',
    'MUSIC5',
    'MUSIC6',
    'MUSIC7',
    'MUSIC8',
    'MUSIC9',
    'MUSIC10',
    'MUSIC11',
    'MUSIC12',
    'MUSIC13',
    'MUSIC14',
    'MUSIC15',
    'MUSIC16'
]

for bag_name in bag_names:
    # bag_path = f'/home/pierre-olivier/catkin_ws/src/bag/article/1004/{bag_name}.bag'
    # wav_path_out = f'/home/pierre-olivier/catkin_ws/src/bag/article/1004/{bag_name}.wav'
    bag_path = f'/home/pierre-olivier/Documents/Data_Article/tagging/2008-music/{bag_name}.bag'
    wav_path_out = f'/home/pierre-olivier/Documents/Data_Article/tagging/2008-music/{bag_name}.wav'
    sf = 32000

    audio_frame_msg = AudioFrame()

    frames_list = []

    for topic, msg, _ in rosbag.Bag(bag_path).read_messages():
        frames = np.array(convert_audio_data_to_numpy_frames(get_format_information(msg.format),
                                                                    msg.channel_count, msg.data))
        frames_list.append(frames)

    frames_list = np.hstack(frames_list)
    io.write(frames_list, wav_path_out, sf)
