#!/usr/bin/env python3

import rosbag
import numpy as np

from audio_utils import get_format_information, convert_audio_data_to_numpy_frames
from audio_utils.msg import AudioFrame
import kissdsp.io as io

bag_names = [
    'CALIB_test'
]

for bag_name in bag_names:
    bag_path = f'/home/pierre-olivier/catkin_ws/src/bag/article/video/{bag_name}.bag'
    wav_path_out = f'/home/pierre-olivier/catkin_ws/src/bag/article/video/{bag_name}.wav'
    # bag_path = f'/home/pierre-olivier/Documents/Data_Article/tagging/scream/2008/{bag_name}.bag'
    # wav_path_out = f'/home/pierre-olivier/Documents/Data_Article/tagging/scream/2008/{bag_name}.wav'
    sf = 32000

    audio_frame_msg = AudioFrame()

    frames_list = []

    for topic, msg, _ in rosbag.Bag(bag_path).read_messages():
        frames = np.array(convert_audio_data_to_numpy_frames(get_format_information(msg.format),
                                                                    msg.channel_count, msg.data))
        frames_list.append(frames)

    frames_list = np.hstack(frames_list)
    io.write(frames_list, wav_path_out, sf)
