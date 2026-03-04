import neurokit2 as nk

def hrv_analyzer(ecg_signal, sampling_rate=1000):
    
    # 偵測 R 峰值
    peaks, info = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
    
    # 計算 HRV 指標
    hrv_metrics = nk.hrv(peaks, sampling_rate=sampling_rate, show=False)
    
    hrv_SDNN = round(hrv_metrics["HRV_SDNN"].iloc[0], 2) if "HRV_SDNN" in hrv_metrics.columns else None
    hrv_LF = round(hrv_metrics["HRV_LF"].iloc[0], 2) if "HRV_LF" in hrv_metrics.columns else None
    hrv_HF = round(hrv_metrics["HRV_HF"].iloc[0], 2) if "HRV_HF" in hrv_metrics.columns else None
    hrv_LF_HF = round(hrv_metrics["HRV_LFHF"].iloc[0], 2) if "HRV_LFHF" in hrv_metrics.columns else None
    hrv_DFA = round(hrv_metrics["HRV_DFA_alpha1"].iloc[0], 2) if "HRV_DFA_alpha1" in hrv_metrics.columns else None

    hrv_analysis_resuls = {
        "HRV_SDNN": hrv_SDNN,
        "HRV_LF": hrv_LF,
        "HRV_HF": hrv_HF,
        "HRV_LF_HF": hrv_LF_HF,
        "HRV_DFA_alpha1": hrv_DFA,
    }

    return hrv_analysis_resuls

