#!/usr/bin/env python3

import os
import rosbag
import numpy as np
import shutil
import pickle

from sklearn.decomposition import PCA

from audio_utils import convert_audio_data_to_numpy_frames
import kissdsp.filterbank as fb

from utils import beamformer_utils as bu


def save_scm(wav, path, frame_size, hop_length):
    Rs = fb.stft(wav, frame_size=frame_size, hop_size=hop_length)
    RRs = bu.scm(Rs)
    RRsInv = np.linalg.inv(RRs)

    idx = np.triu_indices(wav.shape[0])

    # Saving inverse
    arrInv_tf = np.array((RRsInv.real, RRsInv.imag)).flatten()

    np.save(path, arrInv_tf)

    # Returning only the good part of RRs
    arr = np.array([RRs.real, RRs.imag])
    arr_t = arr[:, :, idx[0], idx[1]]
    arr_tf = arr_t.flatten()

    return arr_tf

def save_pca(tfs, database_path):
    pca = PCA(n_components=min(30, len(tfs)))
    pca.fit(tfs)
    pca_dict = pca.transform(tfs)

    file = open(f'{database_path}model', 'wb')
    pickle.dump(pca, file)
    file.close()
    np.save(f'{database_path}pca_dict', pca_dict)

def reset_database(database_path):
    try:
        shutil.rmtree(database_path)
        os.mkdir(database_path)
    except OSError as e:
        print("Error: %s : %s" % (database_path, e.strerror))

def calibration_run(bag_calibration_path, list_bag_calibration, frame_size, hop_length, overlap, input_format_information, database_path, n_frame_scm, step=2000):
    reset_database(database_path)
    tfs = []
    idx = 0
    len_window = int(n_frame_scm * hop_length)

    for b in list_bag_calibration:
        bag_path = f'{bag_calibration_path}{b}'
        frames_all = []
        print(b)
        for _, msg, _ in rosbag.Bag(bag_path).read_messages():
            frames = convert_audio_data_to_numpy_frames(input_format_information, msg.channel_count, msg.data)
            frames = np.array(frames)
            frames_all.append(frames)

        frames_all = np.hstack(frames_all)

        i = 0
        while (i+len_window)<frames_all.shape[1]:
            window = frames_all[:, i:(i+len_window)]
            tf = save_scm(window, f'{database_path}{idx}', frame_size, hop_length)
            tfs.append(tf)
            i = i+step
            idx = idx+1

    tfs = np.array(tfs)
    save_pca(tfs, database_path)
