#!/usr/bin/env python3
import os
import sys

import rosbag
import numpy as np
import shutil
import time
import pickle

from sklearn.decomposition import PCA

from audio_utils import convert_audio_data_to_numpy_frames
import kissdsp.io as io
import kissdsp.filterbank as fb
import kissdsp.spatial as sp

from utils import list_info


def save_scm(wav, path, frame_size, hop):
    Rs = fb.stft(wav, frame_size=frame_size, hop_size=hop)
    RRs = sp.scm(sp.xspec(Rs))
    RRsInv = np.linalg.inv(RRs)

    idx = np.triu_indices(wav.shape[0])

    # Saving inverse RRs
    arrInv = np.array([RRsInv.real, RRsInv.imag])
    arrInv_t = arrInv[:, :, idx[0], idx[1]]
    arrInv_tf = arrInv_t.flatten()

    np.save(path, arrInv_tf)

    # Returning only the good part of RRs
    arr = np.array([RRs.real, RRs.imag])
    arr_t = arr[:, :, idx[0], idx[1]]
    arr_tf = arr_t.flatten()

    return arr_tf


def save_db(bag_path, channel_keep, frame_size, overlap, input_format_information, database_path):
    hop = 256
    tfs = []

    try:
        shutil.rmtree(database_path)
        os.mkdir(database_path)
    except OSError as e:
        print("Error: %s : %s" % (database_path, e.strerror))

    idx = 0
    for bag in list_info.list_bag_database:
        bag_path_ = f'{bag_path}{bag}.bag'

        frames_all  = []
        for _, msg, _ in rosbag.Bag(bag_path_).read_messages():
            frames = convert_audio_data_to_numpy_frames(input_format_information, msg.channel_count, msg.data)
            frames = np.array(frames)[channel_keep]
            frames_all.append(frames)

        frames_all = np.hstack(frames_all)
        len_window = msg.frame_sample_count + int(overlap * frame_size)

        i = 0
        step = 2000
        while (i+len_window)<frames_all.shape[1]:
            window = frames_all[:, i:(i+len_window)]
            tf = save_scm(window, f'{database_path}{idx}', frame_size, hop)
            tfs.append(tf)
            i = i+step
            idx = idx+1
        print(idx)

    tfs = np.array(tfs)
    pca = PCA(n_components=min(300, len(tfs)))
    pca.fit(tfs)
    pca_dict = pca.transform(tfs)

    file = open(f'{database_path}model', 'wb')
    pickle.dump(pca, file)
    file.close()
    np.save(f'{database_path}pca_dict', pca_dict)