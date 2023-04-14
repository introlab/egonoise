import numpy as np


def mvdr(SSs, NNsInv):
    nb_of_channels = SSs.shape[1]
    mat = np.zeros_like(SSs, dtype=np.cdouble)

    # Trace
    for i in range(nb_of_channels):
        mat[:, i, i] = np.einsum('ij,ij->i', NNsInv[:, i, :], SSs[:, :, i])

    # Reference channel
    for i in range(1, nb_of_channels):
        mat[:, i, 0] = np.einsum('ij,ij->i', NNsInv[:, i, :], SSs[:, :, 0])

    tr = np.trace(mat, axis1=1, axis2=2)

    # Weight
    ws = mat[:, :, 0] / tr[..., None]

    return ws


def scm(Xs):
    nb_of_frames = Xs.shape[1]

    spec = Xs
    spec_conj = np.conj(Xs)
    XXs = np.einsum('ijk,ujk->kiu', spec, spec_conj)/nb_of_frames

    return XXs


def beam(Xs, ws):
    ws_conj = np.conj(ws)
    Ys = np.einsum('ij,jki->ki', ws_conj, Xs)[None, ...]

    return Ys
