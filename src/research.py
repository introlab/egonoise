#!/usr/bin/env python3

import rosbag
import os
import logging

import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime
from time import time
from torch import Tensor

from audio_utils import get_format_information, convert_audio_data_to_numpy_frames
from audio_utils.msg import AudioFrame
from utils.egonoise_utils import *
from utils.metrics import *
from utils import list_info

import kissdsp.io as io
import kissdsp.filterbank as fb
import kissdsp.spatial as sp
import kissdsp.beamformer as bf


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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
dict_path = list_info.dict_path#f'/home/pierre-olivier/catkin_ws/src/egonoise/src/database/'
bag_path = list_info.bag_path#f'/home/pierre-olivier/catkin_ws/src/bag/article/{list_info.local}/'

# Create folder for current session
session_path = f'{bag_path}/{datetime.now()}/'
os.mkdir(session_path)

audio_frame_msg = AudioFrame()

n_channel = list_info.n_channel
overlap = list_info.overlap
frame_size = list_info.frame_size
hop = list_info.hop
sf = list_info.sf
frame_sample_count = list_info.frame_sample_count

pca, pca_dict = load_pca(dict_path)

list_all_metrics = []

for bag_speech, g in list_info.list_bag_target:
    for bag_noise in list_info.list_bag_noise:

        # Create folder to save results
        path_out = f'{session_path}{bag_speech}_{bag_noise}/'
        os.mkdir(path_out)

        # List to save all metrics during infering
        list_metrics = []
        signal_filter, signal_voice, signal_noisy = [], [], []
        stft_filter, stft_voice, stft_noisy = [], [], []

        # Create logger and log the info of the current session
        fhdlr = logging.FileHandler(filename=f'{path_out}info.log', mode='a')
        fhdlr.setLevel(logging.INFO)
        fhdlr.setFormatter(formatter)
        logger.addHandler(fhdlr)

        # Log the info of the session
        logger.info(f'Bag noise: {bag_noise}')
        logger.info(f'Bag speech: {bag_speech}\n')

        logger.info(row_format.format(*list_metrics_names))

        # Creating bag generator
        speech_gen = rosbag.Bag(f'{bag_path}{bag_speech}.bag').read_messages()
        noise_gen = rosbag.Bag(f'{bag_path}{bag_noise}.bag').read_messages()

        # Get first message to create last window overlap
        # This message is also discarted
        (_, msg_speech, _), (_, msg_noise, _) = next(speech_gen), next(noise_gen)
        frames_speech = list_info.gain * np.array(convert_audio_data_to_numpy_frames(
                get_format_information(msg_speech.format), msg_speech.channel_count,  msg_speech.data))
        frames_noise = np.array(convert_audio_data_to_numpy_frames(
                get_format_information(msg_noise.format), msg_noise.channel_count, msg_noise.data))
        frames = frames_noise + frames_speech
        last_window = frames[:, -int(overlap * frame_size):]
        last_window_s = frames_speech[:, -int(overlap * frame_size):]
        last_window_n = frames_noise[:, -int(overlap * frame_size):]

        for j, ((_, msg_speech, _), (_, msg_noise, _)) in enumerate(zip(speech_gen, noise_gen)):
            t_idx_start_wav = time()
            # Convert frames to numpy array
            frames_speech = list_info.gain*np.array(
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
            signal_voice.extend(frames_speech[0])
            signal_noisy.extend(frames[0])

            # Adding the overlap to frames
            frames = np.hstack((last_window, frames))
            frames_speech = np.hstack((last_window_s, frames_speech))
            frames_noise = np.hstack((last_window_n, frames_noise))
            #
            # # Saving overlap for the next frames
            # last_window = frames[:, -int(overlap * frame_size):]
            # last_window_s = frames_speech[:, -int(overlap * frame_size):]
            # last_window_n = frames_noise[:, -int(overlap * frame_size):]
            #
            # # STFT and SCM
            # Ys = fb.stft(frames, frame_size=frame_size, hop_size=hop)
            # YYs = sp.scm(sp.xspec(Ys))
            #
            # # PCA
            # val = compute_pca(YYs, pca)
            # diff = np.sum(abs(val - pca_dict), axis=1)
            # idx = np.argmin(diff)
            # RRsInv = load_scm(dict_path, idx, frame_size, len(frames))
            #
            # Zs, ws = compute_mvdr(Ys, YYs, RRsInv)
            #
            # # ISTFT
            # zs = fb.istft(Zs, hop_size=hop)
            #
            # # Traiting time
            # time_dt = time() - t_idx_start_wav
            #
            # # Index of true trame
            # idx_start_wav = int(overlap / 2 * frame_size)
            # idx_end_wav = -int(overlap / 2 * frame_size)
            # idx_start_spec = int((overlap / 2) * frame_size / hop)
            # idx_end_spec = -int((overlap / 2) * frame_size / hop)
            # signal_filter.extend(zs[0, idx_start_wav:idx_end_wav])
            #
            # # Speech intensity
            # Ts = fb.stft(frames_speech, frame_size=frame_size, hop_size=hop)
            # speech_intensity = np.sum(abs(Ts[:, idx_start_spec:idx_end_spec, :])**2)
            #
            # # Logging info
            # if speech_intensity>list_info.bg_intensity:
            #     Rs = fb.stft(frames_noise, frame_size=frame_size, hop_size=hop)
            #     RRs_best = sp.scm(sp.xspec(Rs))
            #     TTs_best = sp.scm(sp.xspec(Ts))
            #
            #     # BEST MVDR
            #     # Zs_best, ws_best = compute_mvdr(Ys, TTs_best, RRs_best)
            #     # zs_best = fb.istft(Zs_best, hop_size=hop)
            #
            #     Ts_ws = bf.beam(Ts, ws)
            #     # Ts_ws_best = bf.beam(Ts, ws_best)
            #     ts_ws = fb.istft(Ts_ws, hop_size=hop)
            #     # ts_ws_best = fb.istft(Ts_ws_best, hop_size=hop)
            #
            #     Rs_ws = bf.beam(Rs, ws)
            #     # Rs_ws_best = bf.beam(Rs, ws_best)
            #     # rs_ws = fb.istft(Rs_ws, hop_size=hop)
            #     # rs_ws_best = fb.istft(Rs_ws_best, hop_size=hop)
            #
            #     # SNR
            #     snr_begining = snr(Ts[:, idx_start_spec:idx_end_spec], Rs[:, idx_start_spec:idx_end_spec])
            #     snr_after = snr(Ts_ws, Rs_ws)
            #     snr_dt = snr_after-snr_begining
            #
            #     # SDR noise + speech
            #     sdr_noisy_begining = sdr(Tensor(frames[:, idx_start_wav:idx_end_wav]), Tensor(frames_speech[:, idx_start_wav:idx_end_wav]))
            #     sdr_noisy_after = sdr(Tensor(zs[:, idx_start_wav:idx_end_wav]), Tensor(ts_ws[:, idx_start_wav:idx_end_wav]))
            #     sdr_noisy_dt = sdr_noisy_after-sdr_noisy_begining
            #
            #     # SDR speech only
            #     sdr_speech = 0 #sdr(Tensor(ts_ws[:, idx_start_wav:idx_end_wav]), Tensor(ts_ws_best[:, idx_start_wav:idx_end_wav]))
            #
            #     metrics = np.array([
            #         snr_begining, snr_after, snr_dt,
            #         sdr_noisy_begining, sdr_noisy_after, sdr_noisy_dt, sdr_speech,
            #         speech_intensity, time_dt
            #     ])
            #     logger.info(row_format.format(j, *np.round(metrics,2)))
            #
            #     list_metrics.append(metrics)
            #
            # stft_filter.extend(Zs[0, idx_start_spec:idx_end_spec])
            # stft_voice.extend(Ts[0, idx_start_spec:idx_end_spec])
            # stft_noisy.extend(Ys[0, idx_start_spec:idx_end_spec])

        # Write .wav for future analysis
        signal_filter, signal_voice, signal_noisy = np.array(signal_filter), np.array(signal_voice), np.array(signal_noisy)
        io.write(signal_filter, f'{path_out}result.wav', sf)
        io.write(signal_voice, f'{path_out}voice.wav', sf)
        io.write(signal_noisy, f'{path_out}noise.wav', sf)
#
#         # Convert list to numpy array
#         list_metrics = np.array(list_metrics)
#
#
#         # Spectrogram
#         stft_filter = fb.stft(signal_filter[None,...], frame_size=frame_size, hop_size=hop)
#         stft_noisy = fb.stft(signal_noisy[None,...], frame_size=frame_size, hop_size=hop)
#         stft_voice = fb.stft(signal_voice[None,...], frame_size=frame_size, hop_size=hop)
#
#
#         figure = plt.figure(figsize=(5, 6), dpi=300)
#         figure.subplots_adjust(left=0.1,
#                             bottom=0.1,
#                             right=0.9,
#                             top=0.9,
#                             wspace=0.4,
#                             hspace=0.4)
#
#         vmin = -25
#         csfont = {'fontname': 'Times New Roman'}
#
#         max_val = len(signal_filter) / sf
#         max_freq = sf/2/1000/2
#         ax = figure.add_subplot(3, 1, 1)
#         img_noisy = 10*np.log10(np.abs(stft_noisy[0, :, :512]) + 1e-10).T
#         imgplot = plt.imshow(img_noisy, aspect='auto', origin='lower', cmap='viridis',
#                              extent=[0,max_val,0, max_freq],
#                              vmin = vmin)
#         ax.set_title('Input spectrogram')
#         ax.set_ylabel('Frequency [kHz]')
#         ax.set_yticks([0, 4, 8])
#
#
#         ax = figure.add_subplot(3, 1, 2)
#         img_filter = 10*np.log10(np.abs(stft_filter[0, :, :512]) + 1e-10).T
#         imgplot = plt.imshow(img_filter, aspect='auto', origin='lower', cmap='viridis',
#                              extent=[0, max_val, 0, max_freq],
#                              vmin = vmin)
#         ax.set_title('Filtered spectrogram')
#         ax.set_ylabel('Frequency [kHz]')
#         ax.set_yticks([0, 4, 8])
#
#         ax = figure.add_subplot(3, 1, 3)
#         voice_img = 10*np.log10(np.abs(stft_voice[0, :, :512]) + 1e-10).T
#         imgplot = plt.imshow(voice_img, aspect='auto', origin='lower',
#                              cmap='viridis',
#                              extent=[0, max_val, 0, max_freq],
#                              vmin = vmin)
#         ax.set_title('Speech only spectrogram')
#         ax.set_ylabel('Frequency [kHz]')
#         ax.set_yticks([0, 4, 8])
#
#         ax.set_xlabel('Time [s]')
#         plt.savefig(fname=f'{path_out}Spectrograms.pdf', format='pdf', bbox_inches='tight')
#
#
#         # Append list_metrics to a list containing all tests
#         list_all_metrics.extend(list_metrics)
#
#         # Logging means
#         mean_metrics = np.round(np.mean(list_metrics, axis=0), 2)
#         metrics = np.array([
#             mean_metrics[0], mean_metrics[1], mean_metrics[2],
#             mean_metrics[3], mean_metrics[4], mean_metrics[5], mean_metrics[6],
#             mean_metrics[7], mean_metrics[8]
#         ])
#         logger.info(f'\n{row_format.format("Mean", *metrics)}')
#
#         # Delete logger for future run
#         logger.removeHandler(fhdlr)
#
#
# # Convert list to numpy array
# list_all_metrics = np.array(list_all_metrics).T
# np.save(f'{session_path}metrics', list_all_metrics)
#
# # Global metrics
# fhdlr = logging.FileHandler(filename=f'{session_path}Global test info.log', mode='a')
# fhdlr.setLevel(logging.INFO)
# fhdlr.setFormatter(formatter)
# logger.addHandler(fhdlr)
#
# logger.info(f'\nMean over all trame')
# logger.info(row_format.format(*list_metrics_names))
# mean_metrics = np.round(np.mean(list_all_metrics, axis=1), 2)
# metrics = np.array([
#     mean_metrics[0], mean_metrics[1], mean_metrics[2],
#     mean_metrics[3], mean_metrics[4], mean_metrics[5], mean_metrics[6],
#     mean_metrics[7], mean_metrics[8]
# ])
# logger.info(f'{row_format.format("Mean", *metrics)}')
#
# logger.info(f'\nPARAMETERS')
# logger.info(f'Sampling rate: {sf} Hz')
# logger.info(f'overlap: {overlap}')
# logger.info(f'frame_size: {frame_size} ')
# logger.info(f'Hop length: {hop}')
# logger.info(f'frame_sample_count: {frame_sample_count}')
# logger.info(f'Dictionnary size: {len(pca_dict)}\n')
#
# # Graphics
# ## SNR
# figure = plt.figure(figsize=(5, 3), dpi=300)
# ax = figure.add_subplot()
# ax.set_title('SNR of the input signal versus filtered signal')
# ax.set_ylabel('SNR filtered signal (dB)')
# ax.set_xlabel('SNR input signal (dB)')
# ax.scatter(list_all_metrics[0], list_all_metrics[1], marker='.')
# plt.plot([-20, 20],  [-20, 20])
# plt.savefig(fname=f'{session_path}SNR.pdf', format='pdf', bbox_inches='tight')
#
# ## SDR
# figure = plt.figure(figsize=(5, 3), dpi=300)
# ax = figure.add_subplot()
# ax.set_title('SDR of the input signal versus filtered signal')
# ax.set_ylabel('SDR filtered signal (dB)')
# ax.set_xlabel('SDR input signal (dB)')
# ax.scatter(list_all_metrics[3], list_all_metrics[4], marker='.')
# plt.plot([-20, 20],  [-20, 20])
# plt.savefig(fname=f'{session_path}SDR.pdf', format='pdf', bbox_inches='tight')


