#!/usr/bin/env python3

import rosbag
import rospy
import numpy as np
import time
import os
import logging

import matplotlib.pyplot as plt

from datetime import datetime

from audio_utils import get_format_information, convert_audio_data_to_numpy_frames
from audio_utils.msg import AudioFrame
from utils.egonoise_utils import *

import kissdsp.io as io
import kissdsp.filterbank as fb
import kissdsp.spatial as sp
import kissdsp.beamformer as bf


# Create logger
logger = logging.getLogger('Recherche')
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Create row style for result table
list_metrics_names = [
    'idx',
    'SNR before', 'SNR after', 'SNR delta',
    'SDR before', 'SDR after', 'SDR delta', 'SDR speech',
    'Speech presence', 'Traiting time'
]
row_format = "{:^18}" * (len(list_metrics_names))

# Paths
dict_path = f'/home/pierre-olivier/catkin_ws/src/egonoise/src/database/'
bag_path = f'/home/pierre-olivier/catkin_ws/src/bag/January/12jan/'

# Create folder for current session
session_path = f'{bag_path}/{datetime.now()}/'
os.mkdir(session_path)

audio_frame_msg = AudioFrame()

n_channel = 16
overlap = 5.0
frame_size = 1024
hop_length = 256
hop = hop_length
sf = 32000
frame_sample_count = 16000

RRs_dict, RRs_inv_dict = load_dictionnary_mp(dict_path, frame_size, hop_length)
max_real = np.max(RRs_dict.real)
max_imag = np.max(RRs_dict.imag)
pca, pca_dict = create_pca(RRs_dict=RRs_dict, mr=max_real, mi=max_imag, n_components=min(RRs_dict.shape[0],2000))

last_window = np.zeros((n_channel, int(overlap * frame_size)))
last_window_s = np.zeros((n_channel, int(overlap * frame_size)))
last_window_n = np.zeros((n_channel, int(overlap * frame_size)))

list_all_metrics = []

for bag_speech in [
    'S1en',
    'S2en',
    'S3en',
    'S4en',
    'S5en'
]:
    for bag_noise in [
        # 'AL2',
        'AL3'
    ]:
        # Create folder to save results
        path_out = f'{session_path}{bag_speech}_{bag_noise}/'
        os.mkdir(path_out)

        # List to save all metrics during infering
        list_metrics = []
        signal_filter = []
        signal_voice = []
        signal_noisy =[]

        # Create logger and log the info of the current session
        fhdlr = logging.FileHandler(filename=f'{path_out}info.log', mode='a')
        fhdlr.setLevel(logging.INFO)
        fhdlr.setFormatter(formatter)
        logger.addHandler(fhdlr)

        # Log the info of the session
        logger.info(f'Bag noise: {bag_noise}')
        logger.info(f'Bag speech: {bag_speech}\n')

        logger.info(row_format.format(*list_metrics_names))

        for j, ((_, msg_speech, _), (_, msg_noise, _)) in enumerate(zip(rosbag.Bag(f'{bag_path}{bag_speech}.bag').read_messages(),
                                                         rosbag.Bag(f'{bag_path}{bag_noise}.bag').read_messages())):
            t_idx_start_wav = time()
            # Convert frames to numpy array
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

            # Creating a list with only and noisy signal to be able to save it as .wav file
            signal_voice.extend(np.sum(frames_speech, axis=0))
            signal_noisy.extend(np.sum(frames, axis=0))

            # Adding the overlap to frames
            frames = np.hstack((last_window, frames))
            frames_speech = np.hstack((last_window_s, frames_speech))
            frames_noise = np.hstack((last_window_n, frames_noise))

            # Saving overlap for the next frames
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

            # Traiting time
            time_dt = time() - t_idx_start_wav

            # Index of true trame
            idx_start_wav = int(overlap/2*frame_size)
            idx_end_wav = -int(overlap/2*frame_size)
            idx_start_spec = int((overlap/2)*frame_size/hop)
            idx_end_spec = -int((overlap/2)*frame_size/hop)
            signal_filter.extend(zs[0, idx_start_wav:idx_end_wav])

            Rs = fb.stft(frames_noise, frame_size=frame_size, hop_size=hop)
            RRs_best = sp.scm(sp.xspec(Rs))
            Ts = fb.stft(frames_speech, frame_size=frame_size, hop_size=hop)
            TTs_best = sp.scm(sp.xspec(Ts))

            # BEST MVDR
            Zs_best, ws_best = compute_mvdr(Ys, TTs_best, RRs_best)
            zs_best = fb.istft(Zs_best, hop_size=hop)

            # SNR
            snr_begining = snr(Ts[:, idx_start_spec:idx_end_spec],
                               Rs[:, idx_start_spec:idx_end_spec])
            snr_after = snr(bf.beam(Ts[:, idx_start_spec:idx_end_spec], ws),
                            bf.beam(Rs[:, idx_start_spec:idx_end_spec], ws))
            snr_dt = snr_after-snr_begining

            # SDR noise + speech
            sdr_noisy_begining = sdr(Tensor(frames[:, idx_start_wav:idx_end_wav]), Tensor(frames_speech[:, idx_start_wav:idx_end_wav]))
            sdr_noisy_after = sdr(Tensor(zs[:, idx_start_wav:idx_end_wav]), Tensor(zs_best[:, idx_start_wav:idx_end_wav]))
            sdr_noisy_dt = sdr_noisy_after-sdr_noisy_begining

            # SDR speech only
            sdr_speech = sdr(Tensor(fb.istft(bf.beam(Ts, ws), hop_size=hop)[:, idx_start_wav:idx_end_wav]), Tensor(fb.istft(bf.beam(Ts, ws_best), hop_size=hop)[:, idx_start_wav:idx_end_wav]))

            # Speech intensity
            speech_intensity = np.sum(abs(Ts[0, :, 100:])**2)

            # Logging info
            if np.sum(abs(Ts[0, :, 100:])**2)>5:
                metrics = np.array([
                    snr_begining, snr_after, snr_dt,
                    sdr_noisy_begining, sdr_noisy_after, sdr_noisy_dt, sdr_speech,
                    speech_intensity, time_dt
                ])
                logger.info(row_format.format(j, *np.round(metrics,2)))

                list_metrics.append(metrics)


        # Convert list to numpy array
        list_metrics = np.array(list_metrics)
        signal_filter, signal_voice, signal_noisy = np.array(signal_filter), np.array(signal_voice), np.array(signal_noisy)

        # Append list_metrics to a list containing all tests
        list_all_metrics.extend(list_metrics)

        # Write .wav for future analysis
        io.write(signal_filter, f'{path_out}result.wav', sf)
        io.write(signal_voice, f'{path_out}voice.wav', sf)
        io.write(signal_noisy, f'{path_out}noise.wav', sf)

        # Logging means
        mean_metrics = np.round(np.mean(list_metrics, axis=0), 2)
        metrics = np.array([
            mean_metrics[0], mean_metrics[1], mean_metrics[2],
            mean_metrics[3], mean_metrics[4], mean_metrics[5], mean_metrics[6],
            mean_metrics[7], mean_metrics[8]
        ])
        logger.info(f'\n{row_format.format("Mean", *metrics)}')

        logger.info(f'\nPARAMETERS')
        logger.info(f'Sampling rate: {sf} Hz')
        logger.info(f'overlap: {overlap}')
        logger.info(f'frame_size: {frame_size} ')
        logger.info(f'Hop length: {hop_length}')
        logger.info(f'frame_sample_count: {frame_sample_count}')
        logger.info(f'Dictionnary size: {len(pca_dict)}\n')

        # Delete logger for future run
        logger.removeHandler(fhdlr)

# Convert list to numpy array
list_all_metrics = np.array(list_all_metrics).T

# Graphics
## SNR
figure = plt.figure(figsize=(5, 3), dpi=300)
ax = figure.add_subplot()
ax.set_title('SNR before and after filtering on every trame')
ax.set_ylabel('SNR filtered signal (dB)')
ax.set_xlabel('SNR noisy signal (dB)')
ax.scatter(list_all_metrics[0], list_all_metrics[1], marker='.')
plt.savefig(fname=f'{session_path}SNR', format='pdf', bbox_inches='tight')

## SDR
figure = plt.figure(figsize=(5, 3), dpi=300)
ax = figure.add_subplot()
ax.set_title('SDR before and after filtering on every trame')
ax.set_ylabel('SDR filtered signal (dB)')
ax.set_xlabel('SDR noisy signal (dB)')
ax.scatter(list_all_metrics[3], list_all_metrics[4], marker='.')
plt.savefig(fname=f'{session_path}SDR', format='pdf', bbox_inches='tight')