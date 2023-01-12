#!/usr/bin/env python3

import rosbag
import rospy
import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from audio_utils import get_format_information, convert_audio_data_to_numpy_frames
from audio_utils.msg import AudioFrame
from utils.egonoise_utils import *

import kissdsp.io as io
import kissdsp.filterbank as fb
import kissdsp.spatial as sp
import kissdsp.beamformer as bf


dict_path = f'/home/pierre-olivier/catkin_ws/src/egonoise/src/database/'
speech_bag_path = f'/home/pierre-olivier/catkin_ws/src/bag/5dec/M2_30_8000.bag'
# noise_bag_path = f'/home/pierre-olivier/catkin_ws/src/bag/11nov/ALc5_30_8000.bag'

audio_frame_msg = AudioFrame()

n_channel = 16
overlap = 1.5
frame_size = 1024
hop_length = 128
hop = hop_length
sf = 16000

list_frames_noise = []
list_frames_speech = []
list_frames = []
list_RRs = []
for p in [
    f'/home/pierre-olivier/catkin_ws/src/bag/9jan/AL1.bag'
]:
    last_window = np.zeros((n_channel, int(overlap * frame_size)))
    last_window_s = np.zeros((n_channel, int(overlap * frame_size)))
    last_window_n = np.zeros((n_channel, int(overlap * frame_size)))
    for (_, msg_speech, _), (_, msg_noise, _) in zip(rosbag.Bag(speech_bag_path).read_messages(), rosbag.Bag(p).read_messages()):
        frames_speech = np.array(convert_audio_data_to_numpy_frames(get_format_information(msg_speech.format), msg_speech.channel_count, msg_speech.data))
        frames_noise = np.array(convert_audio_data_to_numpy_frames(get_format_information(msg_noise.format), msg_noise.channel_count, msg_noise.data))
        frames = frames_noise + frames_speech
        frames = np.hstack((last_window, frames))
        frames_speech = np.hstack((last_window_s, frames_speech))
        frames_noise = np.hstack((last_window_n, frames_noise))
        last_window = frames[:, -int(overlap * frame_size):]
        last_window_s = frames_speech[:, -int(overlap * frame_size):]
        last_window_n = frames_noise[:, -int(overlap * frame_size):]
        list_frames.append(frames)
        list_frames_speech.append(frames_speech)
        list_frames_noise.append(frames_noise)

        Rs = fb.stft(frames, frame_size=frame_size, hop_size=hop)
        RRs = sp.scm(sp.xspec(Rs))
        list_RRs.append(RRs)


list_frames = np.array(list_frames)
list_frames_speech = np.array(list_frames_speech)
list_frames_noise = np.array(list_frames_noise)
list_RRs = np.array(list_RRs)
#

RRs_dict, RRs_inv_dict = load_dictionnary_mp(dict_path, frame_size, hop_length)
max_real = np.max(RRs_dict.real)
max_imag = np.max(RRs_dict.imag)
pca, pca_dict = create_pca(RRs_dict=RRs_dict, mr=max_real, mi=max_imag, n_components=RRs_dict.shape[0])
pca_list = compute_pca(list_RRs, pca, pca_dict, mr=max_real, mi=max_imag)

std = np.std(pca_dict, axis=0)
mean = np.mean(pca_dict, axis=0)
idx = np.zeros(len(pca_dict))
out_per_dim = np.zeros(len(std))
for i in range(len(std)):
    a = pca_dict[:, i] > (std[i] * 3.0 + mean[i])
    b = pca_dict[:, i] < (- std[i] * 3.0 + mean[i])
    c = np.array(a + b, dtype=int)
    idx = idx + c
    out_per_dim[i] = np.sum(c)
    n = int(len(idx) / 2)


std_o = np.std(pca_dict)
mean_o = np.mean(idx)

idx_o = idx > (std_o*1.5 + mean_o)

print(idx)
#
j=0
for i, ix in enumerate(idx_o):
    if ix:
        os.remove(f'/home/pierre-olivier/catkin_ws/src/egonoise/src/database/{i}.wav')
    else:
        os.rename(f'/home/pierre-olivier/catkin_ws/src/egonoise/src/database/{i}.wav',
                  f'/home/pierre-olivier/catkin_ws/src/egonoise/src/database/{j}.wav')
        j += 1

#########################################################################################
# plt.figure()
# plt.hist(idx[:n], 20, density=False, facecolor='g', alpha=0.5)
# plt.hist(idx[n:], 20, density=False, facecolor='b', alpha=0.5)
# plt.hist(idx, 20, density=False, facecolor='r', alpha=0.5)
#
# plt.xlabel('Dimension')
# plt.ylabel('Number of outlier')
# plt.title('Histogram of the numerber of outlier per dimension')
# plt.grid(True)


# for n_dims in [1400]:#[10, 50, 100, 200, 300, 400, 500, 750, 1000, 1500]:
#     ix = idx[:, :n_dims]
#
#     print(f'Number of dimensions: {len(std)}')
#     ix = np.round(idx / max(idx) * 100, 0)
#     print(ix)
#
#     n1 = int(len(ix)/2)
#
#     # the histogram of the data
#     plt.figure()
#     n, bins, patches = plt.hist(ix[:n1], 30, density=False, facecolor='g', alpha=0.5)
#     n2, bins2, patches2 = plt.hist(ix[n1:], 30, density=False, facecolor='b', alpha=0.5)
#
#     plt.xlabel('Percentage of max')
#     plt.ylabel('ndata')
#     plt.title('Histogram of mispca')
#     plt.grid(True)
# plt.show()

# print(f'Mean: {mean}')
# print(f'std: {std}')


# std_idx = np.std(idx)
# mean_idx = np.mean(idx)
#
# idx_off = np.argwhere(idx > (std_idx*2 + mean_idx))


# print(list_frames_noise.shape)
# for i in idx_off:
#     io.write(list_frames_noise[i, 0], f'/home/pierre-olivier/catkin_ws/src/bag/result recherche/{i}.wav')

#### GRAPHIQUES
# nb_ax = 5
#
# figures = []
# axs = []
# os = 0
# for j in range(nb_ax):
#     figure = plt.figure(figsize=(5, 3), dpi=100)
#     ax = figure.add_subplot(projection='3d')
#     ax.scatter(
#         pca_dict[:, 0+3*j+os],
#         pca_dict[:, 1+3*j+os],
#         pca_dict[:, 2+3*j+os],
#     )
#     figures.append(figure)
#     axs.append(ax)
#
# pcaYYs = []
# pcaRRs = []
# for f, fn in zip(list_frames, list_frames_noise):
#     Ys = fb.stft(f, frame_size=frame_size, hop_size=hop)
#     YYs = sp.scm(sp.xspec(Ys))
#     Rs = fb.stft(fn, frame_size=frame_size, hop_size=hop)
#     RRs = sp.scm(sp.xspec(Rs))
#
#     pcaYY = compute_pca(YYs, pca, pca_dict, mi=max_imag, mr=max_real)
#     pcaRR = compute_pca(RRs, pca, pca_dict, mi=max_imag, mr=max_real)
#
#     pcaYYs.append(pcaYY)
#     pcaRRs.append(pcaRR)
#
# pcaYYs = np.array(pcaYYs)
# pcaRRs = np.array(pcaRRs)
# print(int(len(pcaYYs[0,0])/10))
# for i in range(int(len(pcaYYs[0,0])/10)):
#     print(f'Mean {i}: {np.sum(abs(pcaYYs[:, 0, i*10:i*10+10] - pcaRRs[:, 0, i*10:i*10+10]))/np.sum(abs(pcaYYs[:, 0, i*10:i*10+10] + pcaRRs[:, 0, i*10:i*10+10]))}')
#
# for i in range(1, 30):
#     Ys = fb.stft(list_frames[i], frame_size=frame_size, hop_size=hop)
#     YYs = sp.scm(sp.xspec(Ys))
#     Rs = fb.stft(list_frames_noise[i], frame_size=frame_size, hop_size=hop)
#     RRs = sp.scm(sp.xspec(Rs))
#
#     pcaYY = compute_pca(YYs, pca, pca_dict, mi=max_imag, mr=max_real)
#     pcaRR = compute_pca(RRs, pca, pca_dict, mi=max_imag, mr=max_real)
#
#     for j, ax in enumerate(axs):
#         ax.scatter(
#             [pcaYY[0, 0+3*j+os], pcaRR[0, 0+3*j+os]],
#             [pcaYY[0, 1+3*j+os], pcaRR[0, 1+3*j+os]],
#             [pcaYY[0, 2+3*j+os], pcaRR[0, 2+3*j+os]]
#         )


#
# km = KMeans(n_clusters=25)
# km.fit(pca_dict)
# k = km.cluster_centers_
# idx_m = 0
# diff = (np.sum(abs(pca_dict[:, :3]-k[idx_m, :3]), axis=1))
# m = np.quantile(diff, 0.01)
# pts = pca_dict[diff<m]
#
#
# figure = plt.figure(figsize=(5, 3), dpi=100)
# ax = figure.add_subplot(projection='3d')
# ax.set_title('PCA dict BD')
# for i in range(len(pts)):
#     ax.scatter(
#         pts[i, 0],
#         pts[i, 1],
#         pts[i, 2]
#     )
# ax.scatter(
#     k[idx, 0],
#     k[idx, 1],
#     k[idx, 2]
# )

# figure = plt.figure(figsize=(5, 3), dpi=100)
# ax = figure.add_subplot(projection='3d')
# ax.set_title('PCA dict BD')
# ax.scatter(
#     pts[:, 0],
#     pts[:, 1],
#     pts[:, 2]
# )
# ax.scatter(
#     k[idx, 0],
#     k[idx, 1],
#     k[idx, 2]
# )

# figure = plt.figure(figsize=(5, 3), dpi=100)
# ax = figure.add_subplot(projection='3d')
# ax.set_title('PCA dict HD')
# ax.scatter(
#     pts[:, 3],
#     pts[:, 4],
#     pts[:, 5]
# )
# ax.scatter(
#     k[idx, 3],
#     k[idx, 4],
#     k[idx, 5]
# )


plt.show()

#
# for j in range(20):
#     RRs_dict_temp = RRs_dict
#     e = list_exemples[1][j]
#     val = compute_pca(e, pca)
#     diff = np.sum(abs(val - pca_dict), axis=1)
#
#     m = []
#     for i in range(5):
#         idx = np.argmin(diff)
#         RRs = RRs_dict_temp[idx]
#         m.append(np.sum(abs(RRs)**2))
#         diff = np.delete(diff, idx, axis=0)
#         RRs_dict_temp = np.delete(RRs_dict, idx, axis=0)
#
#     cur = np.sum(abs(e)**2)
#     print(f'Current: {np.round(cur)}')
#     print(f'Closest: {np.round(m/cur*100)}')


####################################################
# figure = plt.figure(figsize=(5, 3), dpi=100)
# ax = figure.add_subplot(projection='3d')
# ax.set_title('PCA dict')
#
# ax.scatter(
#     pca_dict[:, 0],
#     pca_dict[:, 1],
#     pca_dict[:, 2]
# )
#
# for i in range(1, n_classes):
#     d = compute_pca(list_exemples[i], pca)
#     ax.scatter(
#         d[:, 0],
#         d[:, 1],
#         d[:, 2]
#     )
#
# plt.show()

###################################################
# min_diff_list_dict = []
# for i, p in enumerate(pca_dict):
#     pca_dict_temp = np.delete(pca_dict, i, axis=0)
#     diff = np.sum(abs(p - pca_dict_temp)**2, axis=1)/np.sum(abs(p))
#     min_diff_list_dict.append(min(diff))
#
# threshold = np.quantile(min_diff_list_dict, 0.50)
#
# pca_dict_t1 = pca_dict
# j = 0
# for i, p in enumerate(pca_dict):
#     pca_dict_t2 = np.delete(pca_dict_t1, i-j, axis=0)
#     diff = np.sum(abs(p - pca_dict_t2)**2, axis=1)/np.sum(abs(p))
#     if min(diff) < threshold:
#         os.remove(f'{dict_path}/{i}.wav')
#         pca_dict_t1 = pca_dict_t2
#         j = j + 1
#
# print(f'DICTIONNARY')
# print(f'Quantile 05%: {np.quantile(min_diff_list_dict, 0.05)}')
# print(f'Quantile 10%: {np.quantile(min_diff_list_dict, 0.10)}')
# print(f'Quantile 25%: {np.quantile(min_diff_list_dict, 0.25)}')
# print(f'Quantile 50%: {np.quantile(min_diff_list_dict, 0.50)}')
# print(f'Quantile 75%: {np.quantile(min_diff_list_dict, 0.75)}')
#
# figure = plt.figure(figsize=(5, 3), dpi=100)
# ax = figure.add_subplot(projection='3d')
# ax.set_title('PCA filter')
# ax.scatter(
#     pca_dict_t1[:, 0],
#     pca_dict_t1[:, 1],
#     pca_dict_t1[:, 2]
# )
#
# figure = plt.figure(figsize=(5, 3), dpi=100)
# ax = figure.add_subplot(projection='3d')
# ax.set_title('PCA dict')
# ax.scatter(
#     pca_dict[:, 0],
#     pca_dict[:, 1],
#     pca_dict[:, 2]
# )
#
# plt.show()

#############################################
# print(pca.explained_variance_ratio_)
# print('##########')
# print(pca.noise_variance_)
# print('##########')
# ptot = 0
# for i, p in enumerate(pca.explained_variance_ratio_):
#     ptot += p
#     print(f'{i}: {ptot}')
