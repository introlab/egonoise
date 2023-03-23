import numpy as np

from torchmetrics import SignalDistortionRatio as SDR


def snr(spec_speech, spec_noise):
    mean_speech = np.mean(np.abs(spec_speech) ** 2)
    mean_noise = np.mean(np.abs(spec_noise) ** 2)

    return 10 * np.log10(mean_speech / mean_noise)

def sdr(pred, target):
    c = SDR()
    return c(pred, target)