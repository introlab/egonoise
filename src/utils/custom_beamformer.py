import numpy as np


def mvdr(SSs, NNsInv):
    """
    Generate beamformer weights with MVDR. We compute the following equation:

    w(k) = ( phi_NN(k)^-1 phi_SS(k) / trace{phi_NN(k)^-1 phi_SS(k)} ) u

    Args:
        SSs (np.ndarray):
            The speech spatial covariance matrix (nb_of_bins, nb_of_channels, nb_of_channels).
        NNs (np.ndarray):
            The noise spatial covariance matrix (nb_of_bins, nb_of_channels, nb_of_channels).

    Returns:
        (np.ndarray):
            The beamformer weights in the frequency domain (nb_of_bins, nb_of_channels).
    """

    nb_of_bins = SSs.shape[0]
    nb_of_channels = SSs.shape[1]

    # NNsInv = np.linalg.inv(NNs)

    # mul = NNsInv @ SSs

    tr = np.einsum('kij,kji->k', NNsInv, SSs)
    mul0 = np.einsum('kij,kji->ki', NNsInv, SSs)

    ws = (mul0 / np.tile(np.expand_dims(tr, axis=(1)),
                                 reps=(1, nb_of_channels)))

    return ws