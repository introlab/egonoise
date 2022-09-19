#!/usr/bin/env python3

import sys
sys.path.append('/home/pierre-olivier/Git/kissdsp')

import rosbag
import numpy as np

from audio_utils import convert_audio_data_to_numpy_frames
import kissdsp.sink as snk


def save_db(bag_path, channel_keep, frame_size, overlap, input_format_information, database_path):
    last_window = None

    for idx, (_, msg, _) in enumerate(rosbag.Bag(bag_path).read_messages()):
        frames = convert_audio_data_to_numpy_frames(input_format_information, msg.channel_count, msg.data)
        frames = np.array(frames)[channel_keep]

        if last_window is None:
            last_window = frames[:, -int(overlap * frame_size):]
        else:
            frames = np.hstack((last_window, frames))
            last_window = frames[:, -int(overlap * frame_size):]
            snk.write(frames, f'{database_path}{idx}.wav', msg.sampling_frequency)