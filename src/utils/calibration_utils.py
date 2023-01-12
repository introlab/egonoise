#!/usr/bin/env python3
import os

import rosbag
import numpy as np
import shutil

from audio_utils import convert_audio_data_to_numpy_frames
import kissdsp.io as io


def save_db(bag_path, channel_keep, frame_size, overlap, input_format_information, database_path, overwrite_dict):
    if overwrite_dict:
        try:
            shutil.rmtree(database_path)
            os.mkdir(database_path)
        except OSError as e:
            print("Error: %s : %s" % (database_path, e.strerror))
        start_idx = 0
    else:
        start_idx = len(os.listdir(database_path))


    frames_all  = []
    for idx, (_, msg, _) in enumerate(rosbag.Bag(bag_path).read_messages()):
        frames = convert_audio_data_to_numpy_frames(input_format_information, msg.channel_count, msg.data)
        frames = np.array(frames)[channel_keep]
        frames_all.append(frames)

    frames_all = np.hstack(frames_all)
    len_window = msg.frame_sample_count + int(overlap * frame_size)

    i = 0
    idx = 0
    step = 4000
    while (i+len_window)<frames_all.shape[1]:
        window = frames_all[:, i:(i+len_window)]
        io.write(window, f'{database_path}{idx + start_idx}.wav')
        i = i+step
        idx = idx+1