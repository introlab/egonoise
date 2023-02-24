#!/usr/bin/env python3

import pickle
import numpy as np

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

def compute_mvdr(Ys, TTs, RRsInv):
    ref = 0

    ws = bf.mvdr(TTs, RRsInv, ref) # Compute mvdr weights
    Zs = bf.beam(Ys, ws)  # Perform beamforming
    return Zs, ws


def compute_pca(scm, pca):
    arr = np.array([scm.real, scm.imag])
    idx = np.triu_indices(scm.shape[-1])
    arr_t = arr[:, :, idx[0], idx[1]]
    arr_tf = arr_t.flatten()[None, ...]

    val = pca.transform(arr_tf)
    return val