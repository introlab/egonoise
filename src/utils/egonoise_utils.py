#!/usr/bin/env python3

import sys
sys.path.append('/home/pierre-olivier/Git/kissdsp')

import os
import rospy
import numpy as np

from torchmetrics import SignalNoiseRatio as SDR
from torch import Tensor
from torch import stft, istft

import kissdsp.source as src
import kissdsp.filterbank as fb
import kissdsp.spatial as sp
import kissdsp.beamformer as bf

hop = 64

def load_dictionnary(path, frame_size):
    files = os.listdir(path)
    RRs_list = []
    for file in files:
        p = f'{path}/{file}'
        wav = src.read(p)
        Rs = fb.stft(wav, frame_size=frame_size, hop_size=hop)
        RRs = sp.scm(Rs)
        RRs_list.append(RRs)
    RRs_dict = np.stack(RRs_list)
    RRs_dict_inv = np.linalg.inv(RRs_dict)
    return RRs_dict, RRs_dict_inv

def egonoise(frames, RRs_dict, RRs_inv_dict, frame_size, _channel_keep, verbose=False, frames_speech=None, frames_noise=None, use_mask=False):
    Ys = fb.stft(frames, frame_size=frame_size, hop_size=hop)

    YYs = sp.scm(Ys)
    diff = np.sum(abs(RRs_inv_dict@(YYs-RRs_dict)-np.eye(_channel_keep))**2)
    # diff = np.sum(abs(abs(YYs)**2 - abs(RRs_dict)**2))
    idx = np.argmin(diff)
    RRs = RRs_dict[idx]
    TTs = YYs - RRs
    Zs, vs, ws = compute_mvdr(Ys, TTs, RRs)

    if use_mask:
        mask = abs(Zs)**2 / np.mean(abs(Ys)**2, axis=0)
        TTs = sp.scm(Ys, mask)
        RRs = sp.scm(Ys, 1 - mask)
        Zs, vs, ws = compute_mvdr(Ys, TTs, RRs)

    zs = fb.istft(Zs, hop_size=hop) # Return to time domain

    if verbose:
        Ts = fb.stft(frames_speech, frame_size=frame_size, hop_size=hop)
        TTs_best= sp.scm(Ts)
        Rs = fb.stft(frames_noise, frame_size=frame_size, hop_size=hop)
        RRs_best= sp.scm(Rs)

        Zs_best, _, _ = compute_mvdr(Ys, TTs_best, RRs_best)

        snr_begining = snr(Ts, Rs)
        snr_dt = snr(bf.beam(Ts, ws), bf.beam(Rs, ws)) - snr_begining
        sdr_best = sdr(Tensor(abs(Zs)**2), Tensor(abs(Zs_best)**2))
        rospy.loginfo(f'SNR before: {np.round(snr_begining, 2)}, SNR amelioration: {np.round(snr_dt, 2)}, SDR: {np.round(sdr_best, 2)}')

    return zs

def compute_mvdr(Ys, TTs, RRs):
    vs = sp.steering(TTs)  # Compute steering vector
    ws = bf.mvdr(vs, RRs)  # Compute mvdr weights
    Zs = bf.beam(Ys, ws)  # Perform beamforming
    return Zs, vs, ws

def snr(spec_speech, spec_noise):
    mean_speech = np.mean(np.abs(spec_speech) ** 2)
    mean_noise = np.mean(np.abs(spec_noise) ** 2)

    return 10*np.log10(mean_speech/mean_noise)

def sdr(pred, target):
    c = SDR()
    return c(pred, target)