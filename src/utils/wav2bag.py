#!/usr/bin/env python3

import rosbag
import numpy as np

from audio_utils import get_format_information, convert_audio_data_to_numpy_frames, convert_numpy_frames_to_audio_data
from audio_utils.msg import AudioFrame
import kissdsp.io as io

bag_names = [
    'mix'
]

for bag_name in bag_names:
    wav_path_out = f'/home/pierre-olivier/catkin_ws/src/bag/article/1004-008/{bag_name}.wav'
    bag_path = f'/home/pierre-olivier/catkin_ws/src/bag/article/1004-008/{bag_name}.bag'
    sf = 32000
    newL = 1
    format = 'signed_32'

    audio_frame_msg = AudioFrame()

    frames_list = io.read(wav_path_out)

    newStep = int(sf*newL)
    newNData = int(frames_list.shape[1]/newStep)

    new_frames_list= []
    for i in range(newNData):
        new_frames_list.append(frames_list[:, (i*newStep):((i+1)*newStep)])

    new_frames_list = np.array(new_frames_list)

    bag = rosbag.Bag(bag_path, 'w')
    for i, frames in enumerate(new_frames_list):
        data = convert_numpy_frames_to_audio_data(get_format_information(format), frames)

        amsg = AudioFrame()

        amsg.header.seq = i

        amsg.format = format
        amsg.channel_count = 16
        amsg.sampling_frequency = sf
        amsg.frame_sample_count = newStep
        amsg.data = data

        bag.write('audio_out', amsg)

    bag.close()