#!/usr/bin/env python3

import pickle
import numpy as np
from time import time

from torchmetrics import SignalDistortionRatio as SDR
from torch import Tensor

import kissdsp.io as io
import kissdsp.filterbank as fb
import kissdsp.spatial as sp
import kissdsp.beamformer as bf


def load_scm(data_base_path, idx, fs, m):
    path = f'{data_base_path}{idx}.npy'
    data = np.load(path)

    idx = np.triu_indices(m)

    brru = np.zeros((2, int(fs/2)+1, m, m))
    brr_t = data.reshape((2, int(fs/2)+1, len(idx[0])))

    brru[:, :, idx[0], idx[1]] = brr_t
    brru = brru[0] + 1j*brru[1]
    brrl = np.conj(np.transpose(np.triu(brru, 1), axes=(0,2,1)))
    scmInv = brru+brrl


    return scmInv

def load_pca(database_path):
    file = open(f'{database_path}model', 'rb')
    pca = pickle.load(file)
    file.close()
    pca_dict = np.load(f'{database_path}pca_dict.npy')


    return pca, pca_dict

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

def compute_mvdr(Ys, TTs, RRsInv):
    ref = 0

    ws = bf.mvdr(TTs, RRsInv, ref) # Compute mvdr weights
    Zs = bf.beam(Ys, ws)  # Perform beamforming
    return Zs, ws

def snr(spec_speech, spec_noise):
    mean_speech = np.mean(np.abs(spec_speech) ** 2)
    mean_noise = np.mean(np.abs(spec_noise) ** 2)

    return 10 * np.log10(mean_speech / mean_noise)

def sdr(pred, target):
    c = SDR(filter_length=512)
    return c(pred, target)

def compute_pca(scm, pca):
    arr = np.array([scm.real, scm.imag])
    idx = np.triu_indices(scm.shape[-1])
    arr_t = arr[:, :, idx[0], idx[1]]
    arr_tf = arr_t.flatten()[None, ...]

    val = pca.transform(arr_tf)
    return val