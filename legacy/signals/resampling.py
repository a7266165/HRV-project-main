import numpy as np
from scipy import signal

def downsample_signal(data, original_fs = 10000, target_fs = 1000):
    """
    降低信號的取樣頻率
    
    Parameters:
    -----------
    data : array-like
        原始信號數據
    original_fs : float
        原始取樣頻率 (Hz)
    target_fs : float
        目標取樣頻率 (Hz)
    
    Returns:
    --------
    downsampled_data : numpy.ndarray
        降低取樣頻率後的信號
    """
    if target_fs >= original_fs:
        raise ValueError("目標取樣頻率必須小於原始取樣頻率")
    
    # 計算降取樣比率
    downsample_factor = int(original_fs / target_fs)
    
    # 使用抗混疊濾波器進行降取樣
    downsampled_data = signal.decimate(data, downsample_factor, ftype='fir')
    
    return downsampled_data


def resample_signal(data, original_fs, target_fs):
    """
    重新取樣信號到目標頻率
    
    Parameters:
    -----------
    data : array-like
        原始信號數據
    original_fs : float
        原始取樣頻率 (Hz)
    target_fs : float
        目標取樣頻率 (Hz)
    
    Returns:
    --------
    resampled_data : numpy.ndarray
        重新取樣後的信號
    """
    # 計算新的樣本數
    num_samples = int(len(data) * target_fs / original_fs)
    
    # 使用 scipy 的 resample 函數
    resampled_data = signal.resample(data, num_samples)
    
    return resampled_data