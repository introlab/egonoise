#!/usr/bin/env python3

import sys
sys.path.append('/home/pierre-olivier/Git/kissdsp')

import rospy
import numpy as np

from audio_utils.msg import AudioFrame
from audio_utils import get_format_information, convert_audio_data_to_numpy_frames, convert_numpy_frames_to_audio_data
import kissdsp.sink as snk


def save_db(msg, name, channel_keep, input_format_information, database_path):
    frames = convert_audio_data_to_numpy_frames(input_format_information, msg.channel_count, msg.data)
    frames = np.array(frames)[channel_keep]

    snk.write(frames, f'{database_path}{name}.wav', msg.sampling_frequency)