import numpy as np
import neurokit2 as nk


def analyze_hrv(ecg_signal, sampling_rate=1000):
    """
    Extended HRV analysis returning metrics and intermediate data for plots.

    Parameters
    ----------
    ecg_signal : ndarray
        Preprocessed ECG signal (single channel).
    sampling_rate : int
        Sampling rate in Hz.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'metrics': dict with HRV_SDNN, HRV_LF, HRV_HF, HRV_LF_HF,
          HRV_DFA_alpha1, LFnu, HFnu
        - 'r_peaks': ndarray of R-peak sample indices
        - 'rr_intervals': ndarray of RR intervals in seconds
        - 'rr_times': ndarray of cumulative time for each RR interval
    """
    # Detect R-peaks
    peaks, info = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
    r_peak_indices = info["ECG_R_Peaks"]

    # Compute RR intervals in seconds
    rr_intervals = np.diff(r_peak_indices) / sampling_rate
    rr_times = np.cumsum(rr_intervals)

    # Compute HRV metrics
    hrv_metrics = nk.hrv(peaks, sampling_rate=sampling_rate, show=False)

    def _get(col):
        if col in hrv_metrics.columns:
            return round(hrv_metrics[col].iloc[0], 2)
        return None

    hrv_sdnn = _get("HRV_SDNN")
    hrv_lf = _get("HRV_LF")
    hrv_hf = _get("HRV_HF")
    hrv_lf_hf = _get("HRV_LFHF")
    hrv_dfa = _get("HRV_DFA_alpha1")
    hrv_rmssd = _get("HRV_RMSSD")

    hr_mean = None
    if len(rr_intervals) > 0:
        hr_mean = round(60 / np.mean(rr_intervals), 2)

    # Compute normalized units (LFnu, HFnu)
    if hrv_lf is not None and hrv_hf is not None and (hrv_lf + hrv_hf) > 0:
        lf_nu = round(hrv_lf / (hrv_lf + hrv_hf) * 100, 2)
        hf_nu = round(hrv_hf / (hrv_lf + hrv_hf) * 100, 2)
    else:
        lf_nu = None
        hf_nu = None

    return {
        'metrics': {
            'HRV_SDNN': hrv_sdnn,
            'HRV_LF': hrv_lf,
            'HRV_HF': hrv_hf,
            'HRV_LF_HF': hrv_lf_hf,
            'HRV_DFA_alpha1': hrv_dfa,
            'LFnu': lf_nu,
            'HFnu': hf_nu,
            'HR_mean': hr_mean,
            'HRV_RMSSD': hrv_rmssd,

        },
        'r_peaks': r_peak_indices,
        'rr_intervals': rr_intervals,
        'rr_times': rr_times,
    }
