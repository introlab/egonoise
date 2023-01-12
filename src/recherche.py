#!/usr/bin/env python3

import rosbag
import rospy
import numpy as np
import time

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
overlap = 5.0
frame_size = 1024
hop_length = 256
hop = hop_length
sf = 32000

RRs_dict, RRs_inv_dict = load_dictionnary_mp(dict_path, frame_size, hop_length)
max_real = np.max(RRs_dict.real)
max_imag = np.max(RRs_dict.imag)
pca, pca_dict = create_pca(RRs_dict=RRs_dict, mr=max_real, mi=max_imag, n_components=min(RRs_dict.shape[0],2000))

last_window = np.zeros((n_channel, int(overlap * frame_size)))
last_window_s = np.zeros((n_channel, int(overlap * frame_size)))
last_window_n = np.zeros((n_channel, int(overlap * frame_size)))

j = 0
s_filter = []

for bag_noise in [
    f'/home/pierre-olivier/catkin_ws/src/bag/12jan/AL3.bag'
]:
    for bag_speech in [
        f'/home/pierre-olivier/catkin_ws/src/bag/12jan/S5en.bag'
    ]:
        list_snr_b = []
        list_snr_a = []
        list_sdr_n_b = []
        list_sdr_n_a = []
        for (_, msg_speech, _), (_, msg_noise, _) in zip(rosbag.Bag(bag_speech).read_messages(),
                                                         rosbag.Bag(bag_noise).read_messages()):
            t_start = time()
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

            frames = frames_noise + frames_speech
            frames = np.hstack((last_window, frames))
            frames_speech = np.hstack((last_window_s, frames_speech))
            frames_noise = np.hstack((last_window_n, frames_noise))

            last_window = frames[:, -int(overlap * frame_size):]
            last_window_s = frames_speech[:, -int(overlap * frame_size):]
            last_window_n = frames_noise[:, -int(overlap * frame_size):]

            # STFT and SCM
            Ys = fb.stft(frames, frame_size=frame_size, hop_size=hop)
            YYs = sp.scm(sp.xspec(Ys))

            # PCA
            val = compute_pca(YYs, pca, pca_dict, max_real, max_imag)
            diff = np.sum(abs(val - pca_dict), axis=1)
            idx = np.argmin(diff)
            RRs = RRs_dict[idx]

            TTs = YYs
            Zs, ws = compute_mvdr(Ys, TTs, RRs)

            # ISTFT
            zs = fb.istft(Zs, hop_size=hop)

            t_end = time()

            # Save
            start = int(overlap/2*frame_size)
            end = -int(overlap/2*frame_size)
            Start = int((overlap/2)*frame_size/hop)
            End = -int((overlap/2)*frame_size/hop)
            s_filter.extend(zs[0, start:end])

            Rs = fb.stft(frames_noise, frame_size=frame_size, hop_size=hop)
            RRs_best = sp.scm(sp.xspec(Rs))
            Ts = fb.stft(frames_speech, frame_size=frame_size, hop_size=hop)
            TTs_best = sp.scm(sp.xspec(Ts))

            # BEST MVDR
            Zs_best, ws_best = compute_mvdr(Ys, TTs_best, RRs_best)
            zs_best = fb.istft(Zs_best, hop_size=hop)

            # SNR
            snr_begining = snr(Ts[:, Start:End],
                               Rs[:, Start:End])
            snr_after = snr(bf.beam(Ts[:, Start:End], ws),
                            bf.beam(Rs[:, Start:End], ws))
            snr_dt = snr_after-snr_begining

            # SDR noise + speech
            sdr_n_begining = sdr(Tensor(frames[:, start:end]), Tensor(frames_speech[:, start:end]))
            sdr_n_after = sdr(Tensor(zs[:, start:end]), Tensor(zs_best[:, start:end]))
            sdr_n_dt= sdr_n_after-sdr_n_begining

            # SDR speech only
            sdr_after = sdr(Tensor(fb.istft(bf.beam(Ts, ws), hop_size=hop)[:, start:end]), Tensor(fb.istft(bf.beam(Ts, ws_best), hop_size=hop)[:, start:end]))

            if sdr_after>6.5:
                print(f'{j}, {np.sum(abs(frames_speech))}, SNR before: {np.round(snr_begining, 2)}, SNR amelioration: {np.round(snr_dt, 2)},'
                      f'SDR before: {np.round(sdr_n_begining, 2)}, SDR after: {np.round(sdr_n_after, 2)},'
                      f' SDR speech only: {np.round(sdr_after, 2)}, Time: {np.round(t_end-t_start, 3)}')

                list_snr_b.append(snr_begining)
                list_snr_a.append(snr_after)
                list_sdr_n_b.append(sdr_n_begining)
                list_sdr_n_a.append(sdr_n_after)
            j += 1

        list_snr_b = np.array(list_snr_b)
        list_snr_a = np.array(list_snr_a)
        list_sdr_n_b = np.array(list_sdr_n_b)
        list_sdr_n_a = np.array(list_sdr_n_a)
        s_filter = np.array(s_filter)
        io.write(s_filter, f'/home/pierre-olivier/catkin_ws/src/bag/result.wav', sf)

        print(f'speech bag: {bag_speech}')
        print(f'mean SNR before: {np.mean(list_snr_b)}')
        print(f'mean SNR dt: {np.mean(list_snr_a - list_snr_b)}')
        print(f'mean SDR before: {np.mean(list_sdr_n_b)}')
        print(f'mean SDR after: {np.mean(list_sdr_n_a)}')