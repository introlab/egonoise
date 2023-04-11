import numpy as np

from time import time

def mvdr(SSs, NNsInv):
    nb_of_bins = SSs.shape[0]
    nb_of_channels = SSs.shape[1]

    mat = np.zeros_like(SSs, dtype=np.cdouble)

    for i in range(nb_of_channels):
        mat[:, i, i] = np.einsum('ij,ij->i', NNsInv[:, i, :], SSs[:, :, i])

    for i in range(1, nb_of_channels):
        mat[:, i, 0] = np.einsum('ij,ij->i', NNsInv[:, i, :], SSs[:, :, 0])

    tr = np.trace(mat, axis1=1, axis2=2)

    ws = mat[:, :, 0] / tr[..., None]

    return ws


def scm(Xs):
    nb_of_frames = Xs.shape[1]

    spec = Xs
    spec_conj = np.conj(Xs)
    XXs = np.einsum('ijk,ujk->kiu', spec, spec_conj)/nb_of_frames

    return XXs


def beam(Xs, ws):
    # nb_of_channels = Xs.shape[0]
    # nb_of_frames = Xs.shape[1]
    # nb_of_bins = Xs.shape[2]

    # print(Xs.shape)
    # Xs = np.expand_dims(np.moveaxis(Xs, 0, -1), axis=3)
    # print(Xs.shape)
    # print(ws.shape)
    # ws = np.tile(np.expand_dims(np.expand_dims(ws, axis=1), axis=0), reps=(nb_of_frames, 1, 1, 1))
    # print(ws.shape)
    # Ys = np.expand_dims(np.squeeze(np.squeeze(np.conj(ws) @ Xs, axis=-1), axis=-1), axis=0)
    # print(Ys.shape)

    ws_conj = np.conj(ws)
    Ys = np.einsum('ij,jki->ki', ws_conj, Xs)[None, ...]

    return Ys