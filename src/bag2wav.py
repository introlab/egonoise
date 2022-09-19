#!/usr/bin/env python3

import rosbag
import numpy as np

from audio_utils import get_format_information, convert_audio_data_to_numpy_frames
from audio_utils.msg import AudioFrame
import kissdsp.sink as snk


bag_name = 'noise4micsAfter-STFT-iSTFT'
bag_path = f'/home/pierre-olivier/catkin_ws/src/bag/{bag_name}.bag'
wav_path_out = f'/home/pierre-olivier/catkin_ws/src/bag/{bag_name}.wav'

audio_frame_msg = AudioFrame()

frames_list = []
for topic, msg, _ in rosbag.Bag(bag_path).read_messages():
    frames = np.array(convert_audio_data_to_numpy_frames(get_format_information(msg.format),
                                                                msg.channel_count, msg.data))
    frames_list.append(frames)

frames_list = np.hstack(frames_list)
snk.write(frames_list, wav_path_out, msg.sampling_frequency)
