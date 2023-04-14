#!/usr/bin/env python3

import pickle
import numpy as np

from utils import beamformer_utils as bu


def load_scm(data_base_path, idx, fs, m):
    path = f'{data_base_path}{idx}.npy'
    data = np.load(path).reshape((2, int(fs/2)+1, m, m))
    scmInv = data[0] + 1j * data[1]

    return scmInv

def load_pca(database_path):
    file = open(f'{database_path}model', 'rb')
    pca = pickle.load(file)
    file.close()
    pca_dict = np.load(f'{database_path}pca_dict.npy')

    return pca, pca_dict

def compute_mvdr(Ys, TTs, RRsInv):
    ws = bu.mvdr(TTs, RRsInv)
    Zs = bu.beam(Ys, ws)  # Perform beamforming

    return Zs, ws


def compute_pca(scm, pca):
    arr = np.array([scm.real, scm.imag])

    idx = np.triu_indices(scm.shape[-1])
    arr_t = arr[:, :, idx[0], idx[1]]
    arr_tf = arr_t.flatten()[None, ...]

    val = pca.transform(arr_tf)
    return val