#!/usr/bin/env python3

import os
import rospy
import numpy as np
from time import time

from torchmetrics import SignalDistortionRatio as SDR
from torch import Tensor
from sklearn.decomposition import PCA

import kissdsp.io as io
import kissdsp.filterbank as fb
import kissdsp.spatial as sp
import kissdsp.beamformer as bf

from multiprocessing import Process, Pool
from itertools import repeat


def load_dictionnary(path, frame_size, hop):
    files = os.listdir(path)
    RRs_list = []
    for i in range(len(files)):
        p = f'{path}/{i}.wav'
        wav = io.read(p)
        Rs = fb.stft(wav, frame_size=frame_size, hop_size=hop)
        RRs = sp.scm(sp.xspec(Rs))
        RRs_list.append(RRs)
    RRs_dict = np.stack(RRs_list)
    RRs_dict_inv = None#np.linalg.inv(RRs_dict)
    return RRs_dict, RRs_dict_inv


def load_dictionnary_mp(path, frame_size, hop):
    files = os.listdir(path)
    wav_list = []
    for i in range(len(files)):
        p = f'{path}/{i}.wav'
        wav = io.read(p)
        wav_list.append(wav)

    with Pool(8) as p:
        RRs = p.starmap(wav_to_scm, zip(wav_list, repeat(frame_size), repeat(hop)))

    RRs_dict = np.array(RRs)
    RRs_dict_inv = None
    return RRs_dict, RRs_dict_inv

def wav_to_scm(wav, frame_size, hop):
    Rs = fb.stft(wav, frame_size=frame_size, hop_size=hop)
    RRs = sp.scm(sp.xspec(Rs))
    return RRs

def egonoise(frames, RRs_dict, RRs_inv_dict, pca, pca_dict,
             frame_size, n_channels, hop, verbose=False, frames_speech=None,
             frames_noise=None, use_mask=False):
    Ys = fb.stft(frames, frame_size=frame_size, hop_size=hop)
    YYs = sp.scm(sp.xspec(Ys))

    # RRs, TTs  = irm(Ys, frames_noise, frames_speech, frame_size, hop)

    # PCA
    val = compute_pca(YYs, pca)
    diff = np.sum(abs(val - pca_dict), axis=1)
    idx = np.argmin(diff)
    print(idx)
    RRs = RRs_dict[idx]
    TTs = YYs - RRs

    # MVDR
    Zs, ws = compute_mvdr(Ys, TTs, RRs)

    if use_mask:
        mask = abs(Zs) ** 2 / (np.mean(abs(Ys) ** 2, axis=0) + 1e-13)
        mask -= np.min(mask)
        mask /= np.max(mask)
        m = np.quantile(mask, 0.5)
        mask = (mask>m).astype(float)*mask

        TTs = sp.scm(Ys, mask)
        RRs = sp.scm(Ys, 1 - mask)
        Zs, ws = compute_mvdr(Ys, TTs, RRs)


    zs = fb.istft(Zs, hop_size=hop)  # Return to time domain

    if verbose:
        Ts = fb.stft(frames_speech, frame_size=frame_size, hop_size=hop)
        TTs_best = sp.scm(sp.xspec(Ts))
        Rs = fb.stft(frames_noise, frame_size=frame_size, hop_size=hop)
        RRs_best = sp.scm(sp.xspec(Rs))

        Zs_best, _ = compute_mvdr(Ys, TTs_best, RRs_best)
        Zs_ref, _ = compute_mvdr(Ts, TTs_best, RRs_best)
        zs_ref = fb.istft(Zs_ref, hop_size=hop)

        snr_begining = snr(Ts, Rs)
        snr_dt = snr(bf.beam(Ts, ws), bf.beam(Rs, ws)) - snr_begining
        sdr_begining = sdr(Tensor(frames), Tensor(frames_speech))
        sdr_best = sdr(Tensor(zs), Tensor(zs_ref))
        rospy.loginfo(
            f'SNR before: {np.round(snr_begining, 2)}, SNR amelioration: {np.round(snr_dt, 2)}, SDR: {np.round(sdr_best, 2)}')
        print(sdr_begining)
    else:
        snr_dt = None
        snr_begining = None

    return zs, snr_dt, snr_begining


def irm(Ys, frames_noise, frames_speech, frame_size, hop):
    Rs = fb.stft(frames_noise, frame_size=frame_size, hop_size=hop)
    Ts = fb.stft(frames_speech, frame_size=frame_size, hop_size=hop)

    Rs_sq = abs(Rs) ** 2
    Ts_sq = abs(Ts) ** 2
    mask_RRs = abs(Rs_sq / (Rs_sq + Ts_sq + 1e-7))
    mask_TTs = abs(Ts_sq / (Rs_sq + Ts_sq + 1e-7))

    RRs = sp.scm(Ys, mask_RRs)
    TTs = sp.scm(Ys, mask_TTs)

    return RRs, TTs


def compute_diff(YYs, RRs_dict, RRs_inv_dict, n_channels):
    # diff = np.sum(abs(RRs_inv_dict@(YYs-RRs_dict)-np.eye(n_channels))**2, axis=(1,2,3))
    diff = np.sum(abs(YYs - RRs_dict) ** 2, axis=(1, 2, 3))

    return diff


def compute_mvdr(Ys, TTs, RRs):
    dia = abs(np.diagonal(abs(RRs)**2, axis1=1, axis2=2))
    dia_norm = dia/np.max(dia, axis=0)
    diff = np.sum(dia_norm, axis=0)

    ref = np.argmin(diff)
    # if ref == (6 or 7 or 8 or 9):
    ref = 0
    # print(ref)

    ws = bf.mvdr(TTs, RRs, ref) # Compute mvdr weights
    Zs = bf.beam(Ys, ws)  # Perform beamforming
    return Zs, ws


def snr(spec_speech, spec_noise):
    mean_speech = np.mean(np.abs(spec_speech) ** 2)
    mean_noise = np.mean(np.abs(spec_noise) ** 2)

    return 10 * np.log10(mean_speech / mean_noise)


def sdr(pred, target):
    c = SDR(filter_length=512)
    return c(pred, target)

def create_pca(RRs_dict, mr=1, mi=1, n_components=18):
    # d = np.stack((RRs_dict.real/mr, RRs_dict.imag/mi), axis=-1)
    d = RRs_dict.real/mr
    dim = d.shape
    d = d.reshape((dim[0], dim[1]*dim[2]*dim[3]))
    # d = d.reshape((dim[0], dim[1]*dim[2]*dim[3]*dim[4]))

    pca = PCA(n_components=n_components)
    pca.fit(d)
    pca_dict = pca.transform(d)

    # pmin = np.min(pca_dict, axis=0)
    # pmax = np.max(pca_dict, axis=0)
    #
    # pca_dict = (pca_dict-pmin)/(pmax-pmin) # TEST
    return pca, pca_dict

def compute_pca(RRs_dict, pca, pca_dict, mr=1, mi=1):
    RRs_dict = RRs_dict[None, ...] if len(RRs_dict.shape)==3 else RRs_dict
    # d = np.stack((RRs_dict.real/mr, RRs_dict.imag/mi), axis=-1)
    d = RRs_dict.real/mr
    dim = d.shape
    # d = d.reshape((dim[0], dim[1]*dim[2]*dim[3]*dim[4]))
    d = d.reshape((dim[0], dim[1]*dim[2]*dim[3]))
    val = pca.transform(d)

    # pmin = np.min(pca_dict, axis=0)
    # pmax = np.max(pca_dict, axis=0)
    #
    # val = (val - pmin) / (pmax - pmin)

    return val