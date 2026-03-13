import numpy as np
from scipy import signal


def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=1000, order=2):
    """
    Butterworth bandpass filter.

    Parameters
    ----------
    data : array-like
        Input signal.
    lowcut : float
        Low cutoff frequency (Hz).
    highcut : float
        High cutoff frequency (Hz).
    fs : float
        Sampling rate (Hz).
    order : int
        Filter order.

    Returns
    -------
    filtered_data : ndarray
        Filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def downsample_signal(data, original_fs=10000, target_fs=1000):
    """
    Downsample signal using anti-aliasing FIR decimation.

    Parameters
    ----------
    data : array-like
        Original signal.
    original_fs : float
        Original sampling rate (Hz).
    target_fs : float
        Target sampling rate (Hz).

    Returns
    -------
    downsampled_data : ndarray
        Downsampled signal.
    """
    if target_fs >= original_fs:
        raise ValueError("target_fs must be less than original_fs")
    downsample_factor = int(original_fs / target_fs)
    downsampled_data = signal.decimate(data, downsample_factor, ftype='fir')
    return downsampled_data


def preprocess_ecg(raw_signal, original_fs, target_fs=1000,
                   lowcut=0.5, highcut=100.0, order=2):
    """
    Full ECG preprocessing pipeline: bandpass filter then downsample.

    Parameters
    ----------
    raw_signal : ndarray
        Raw ECG signal (single channel).
    original_fs : int
        Original sampling rate (Hz).
    target_fs : int
        Target sampling rate after downsampling (Hz).
    lowcut : float
        Low cutoff frequency (Hz).
    highcut : float
        High cutoff frequency (Hz).
    order : int
        Filter order.

    Returns
    -------
    processed : ndarray
        Filtered and downsampled ECG signal.
    """
    filtered = bandpass_filter(raw_signal, lowcut, highcut, original_fs, order)
    processed = downsample_signal(filtered, original_fs, target_fs)
    return processed
