#!/usr/bin/env python3

import rosbag
import rospy
import numpy as np

import matplotlib.pyplot as plt

from audio_utils import get_format_information, convert_audio_data_to_numpy_frames
from audio_utils.msg import AudioFrame
from utils.egonoise_utils import *

import kissdsp.io as io
import kissdsp.filterbank as fb
import kissdsp.spatial as sp
import kissdsp.beamformer as bf


dict_path = f'/home/pierre-olivier/catkin_ws/src/egonoise/src/database/'

audio_frame_msg = AudioFrame()

n_channel = 16
overlap = 1.5
frame_size = 512
hop_length = 64
hop = hop_length
sf = 16000

RRs_dict, RRs_inv_dict = load_dictionnary(dict_path, frame_size, hop_length)
pca, pca_dict = create_pca(RRs_dict=RRs_dict[:-1], n_components=3)


# print(pca_dict.shape)
# for i,  p in enumerate(pca_dict):
#     newD = np.delete(pca_dict, i, axis=0)
#     diff = np.sum(abs(p-newD)**2, axis=1)
#     m = min(diff)
#     print(f'Min {i}: {m},')
#
last_window = np.zeros((n_channel, int(overlap * frame_size)))
last_window_s = np.zeros((n_channel, int(overlap * frame_size)))
last_window_n = np.zeros((n_channel, int(overlap * frame_size)))


#
VODS9 = [0, 0, 1, 1, 1, 1, 0, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
       0, 1, 1, 1, 1, 1, 0, 1, 1, 1,
       1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 0]
#
# VODS14 = [1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
#           0, 1, 1, 1, 0, 0, 1, 1, 1, 1,
#           1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
#           0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
#           1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
#           1, 0, 0, 1, 1, 1, 0, 1]

prob = np.zeros((len(pca_dict)))
i = 0
j = 0
pca_n = []
pca_sn = []
rrs_n = []
snr_l = []

for bag_noise in [
    f'/home/pierre-olivier/catkin_ws/src/bag/11oct/AL9_30_8000.bag',
    #f'/home/pierre-olivier/catkin_ws/src/bag/11oct/AL24_30_8000.bag',
    #f'/home/pierre-olivier/catkin_ws/src/bag/11oct/AL25_30_8000.bag',
    #f'/home/pierre-olivier/catkin_ws/src/bag/11oct/AL26_30_8000.bag',
    #f'/home/pierre-olivier/catkin_ws/src/bag/11oct/AL1_60_8000.bag',
    #f'/home/pierre-olivier/catkin_ws/src/bag/11oct/AL2_60_8000.bag',
    #f'/home/pierre-olivier/catkin_ws/src/bag/11oct/AL10_45_8000.bag',
    #f'/home/pierre-olivier/catkin_ws/src/bag/11oct/AL11_45_8000.bag'
]:
    for bag_speech in [
        f'/home/pierre-olivier/catkin_ws/src/bag/11oct/S9_30_8000.bag',
        #f'/home/pierre-olivier/catkin_ws/src/bag/11oct/S10_30_8000.bag'
    ]:
        list_snr_b = []
        list_snr_a = []
        list_sdr_n_b = []
        list_sdr_n_a = []
        for (_, msg_speech, _), (_, msg_noise, _) in zip(rosbag.Bag(bag_speech).read_messages(),
                                                         rosbag.Bag(bag_noise).read_messages()):
            frames_speech = np.array(
                convert_audio_data_to_numpy_frames(
                    get_format_information(msg_speech.format),
                    msg_speech.channel_count,
                    msg_speech.data))
            frames_noise = np.array(
                convert_audio_data_to_numpy_frames(
                    get_format_information(msg_noise.format),
                    msg_noise.channel_count,
                    msg_noise.data))

            frames = frames_speech + frames_noise
            frames = np.hstack((last_window, frames))
            frames_speech = np.hstack((last_window_s, frames_speech))
            frames_noise = np.hstack((last_window_n, frames_noise))

            last_window = frames[:, -int(overlap * frame_size):]
            last_window_s = frames_speech[:, -int(overlap * frame_size):]
            last_window_n = frames_noise[:, -int(overlap * frame_size):]

            if VODS9[j] == 1:
                # STFT and SCM
                Ys = fb.stft(frames, frame_size=frame_size, hop_size=hop)
                YYs = sp.scm(sp.xspec(Ys))
                Rs = fb.stft(frames_noise, frame_size=frame_size, hop_size=hop)
                RRs_best = sp.scm(sp.xspec(Rs))
                Ts = fb.stft(frames_speech, frame_size=frame_size, hop_size=hop)
                TTs_best = sp.scm(sp.xspec(Ts))

                # BEST MVDR
                Zs_best, ws_best = compute_mvdr(Ys, TTs_best, RRs_best)
                zs_best = fb.istft(Zs_best, hop_size=hop)

                # PCA
                val_sn = compute_pca(YYs, pca)
                pca_sn.append(val_sn)
                val_n = compute_pca(RRs_best, pca)
                pca_n.append(val_n)

                diff = np.sum(abs(val_sn - pca_dict)**2, axis=1)
                idx = np.argmin(diff)
                RRs = RRs_dict[idx]

                rrs_n.append(pca_dict[idx])

                TTs = YYs - RRs
                Zs, ws = compute_mvdr(Ys, TTs, RRs)

                # SNR
                snr_begining = snr(Ts, Rs)
                snr_after = snr(bf.beam(Ts, ws), bf.beam(Rs, ws))
                snr_dt = snr_after-snr_begining

                snr_l.append(np.round(snr_dt, 2))

                prob[idx] = prob[idx] + 1
                i = i + 1
            j = j + 1
        print(i)


for p in prob:
    print(p/i*100)


pca_n = np.stack(pca_n)[:,0,:]
pca_sn = np.stack(pca_sn)[:,0,:]
rrs_n = np.stack(rrs_n)
snr_l = np.stack(snr_l)

figure = plt.figure(figsize=(5, 3), dpi=100)
ax = figure.add_subplot(projection='3d')
ax.set_title('PCA')
for i in range(10):
    ax.scatter(
        [pca_n[i, 0], pca_sn[i, 0]],
        [pca_n[i, 1], pca_sn[i, 1]],
        [pca_n[i, 2], pca_sn[i, 2]]
    )
ax.legend(snr_l[:10])

figure = plt.figure(figsize=(5, 3), dpi=100)
ax = figure.add_subplot(projection='3d')
ax.set_title('PCA')
for i in range(10, 20):
    ax.scatter(
        [pca_n[i, 0], pca_sn[i, 0], rrs_n[i, 0]],
        [pca_n[i, 1], pca_sn[i, 1], rrs_n[i, 1]],
        [pca_n[i, 2], pca_sn[i, 2], rrs_n[i, 2]]
    )
ax.legend(snr_l[10:20])

figure = plt.figure(figsize=(5, 3), dpi=100)
ax = figure.add_subplot(projection='3d')
ax.set_title('PCA')
for i in range(20, 30):
    ax.scatter(
        [pca_n[i, 0], pca_sn[i, 0], rrs_n[i, 0]],
        [pca_n[i, 1], pca_sn[i, 1], rrs_n[i, 1]],
        [pca_n[i, 2], pca_sn[i, 2], rrs_n[i, 2]]
    )
ax.legend(snr_l[20:30])

plt.show()

        #         TTs = YYs - RRs
        #         Zs, ws = compute_mvdr(Ys, TTs, RRs)
        #
        #         # ISTFT
        #         zs = fb.istft(Zs, hop_size=hop)
        #
        #         # SNR
        #         snr_begining = snr(Ts, Rs)
        #         snr_after = snr(bf.beam(Ts, ws), bf.beam(Rs, ws))
        #         snr_dt = snr_after-snr_begining
        #
        #         # SDR noise + speech
        #         sdr_n_begining = sdr(Tensor(frames), Tensor(frames_speech))
        #         sdr_n_after = sdr(Tensor(zs), Tensor(zs_best))
        #         sdr_n_dt= sdr_n_after-sdr_n_begining
        #
        #         # SDR speech only
        #         sdr_after = sdr(Tensor(fb.istft(bf.beam(Ts, ws), hop_size=hop)), Tensor(fb.istft(bf.beam(Ts, ws_best), hop_size=hop)))
        #
        #         print(f'SNR before: {np.round(snr_begining, 2)}, SNR amelioration: {np.round(snr_dt, 2)},'
        #               f', SDR before: {np.round(sdr_n_begining, 2)}, SDR after: {np.round(sdr_n_after, 2)}',
        #               f' SDR speech only: {np.round(sdr_after, 2)}')
        #
        #         list_snr_b.append(snr_begining)
        #         list_snr_a.append(snr_after)
        #         list_sdr_n_b.append(sdr_n_begining)
        #         list_sdr_n_a.append(sdr_n_after)
        #
        #     i += 1
        #
        # list_snr_b = np.array(list_snr_b)
        # list_snr_a = np.array(list_snr_a)
        # list_sdr_n_b = np.array(list_sdr_n_b)
        # list_sdr_n_a = np.array(list_sdr_n_a)
        #
        # print(f'speech bag: {bag_speech}')
        # print(f'mean SNR before: {np.mean(list_snr_b)}')
        # print(f'mean SNR dt: {np.mean(list_snr_a - list_snr_b)}')
        # print(f'mean SDR before: {np.mean(list_sdr_n_b)}')
        # print(f'mean SDR after: {np.mean(list_sdr_n_a)}')