import numpy as np
from scipy import signal

def bandpass_filter(data, lowcut = 0.5, highcut=50.0, fs=1000, order=2):
    """
    帶通濾波器函式
    
    Parameters:
    -----------
    data : array-like
        輸入信號數據
    lowcut : float
        低頻截止頻率 (Hz)
    highcut : float
        高頻截止頻率 (Hz)
    fs : float
        採樣頻率 (Hz)
    order : int, optional
        濾波器階數，預設為 4
    
    Returns:
    --------
    filtered_data : ndarray
        濾波後的信號
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data


def adjustable_bandpass_filter(data, fs, lowcut=None, highcut=None, order=4):
    """
    可手動調整頻帶的帶通濾波函式
    
    Parameters:
    -----------
    data : array-like
        輸入信號數據
    fs : float
        採樣頻率 (Hz)
    lowcut : float, optional
        低頻截止頻率 (Hz)，預設 None (使用 0.5 Hz)
    highcut : float, optional
        高頻截止頻率 (Hz)，預設 None (使用 50 Hz)
    order : int, optional
        濾波器階數，預設為 4
    
    Returns:
    --------
    filtered_data : ndarray
        濾波後的信號
    """
    if lowcut is None:
        lowcut = 0.5
    if highcut is None:
        highcut = 50.0
    
    if lowcut >= highcut:
        raise ValueError("lowcut 必須小於 highcut")
    
    if highcut >= fs / 2:
        raise ValueError(f"highcut 必須小於 Nyquist 頻率 ({fs/2} Hz)")
    
    return bandpass_filter(data, lowcut, highcut, fs, order)