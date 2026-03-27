"""
Faithful Python port of Marcus Vollmer's HRV MATLAB Toolkit.
Reference: https://github.com/MarcusVollmer/HRV
MIT License (MIT) Copyright (c) 2015-2020 Marcus Vollmer

Ported by: Claude (2026-03-27)
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.interpolate import CubicSpline
from scipy.ndimage import minimum_filter1d, maximum_filter1d


# ============================================================
# Layer 1: NaN-aware helper functions
# ============================================================

def nanstd(x, flag=1, axis=None):
    """NaN-aware std matching Vollmer's convention.
    flag=1 -> divide by n (ddof=0), flag=0 -> divide by n-1 (ddof=1).
    """
    ddof = 0 if flag == 1 else 1
    with np.errstate(invalid='ignore'):
        return np.nanstd(x, ddof=ddof, axis=axis)


def nanzscore(x, opt=0, axis=None):
    """NaN-aware z-score. opt passed to nanstd (0=n-1, 1=n)."""
    flag = opt  # nanstd convention: opt=0 -> ddof=1, opt=1 -> ddof=0
    # But Vollmer's nanzscore default opt=0 -> nanstd(x,0) -> divide by n-1
    ddof = 0 if opt == 1 else 1
    m = np.nanmean(x, axis=axis, keepdims=True)
    with np.errstate(invalid='ignore'):
        s = np.nanstd(x, ddof=ddof, axis=axis, keepdims=True)
    return (x - m) / s


# ============================================================
# Layer 2: Signal processing filters
# ============================================================

def windowed_extrema(sig, wl):
    """Windowed min/max matching Vollmer's MATLAB implementation.
    MATLAB version builds NxW lag matrix: ts(j:end, j) = sig(1:end-j+1).
    This is a causal (look-back) window of length wl.
    """
    sig = np.asarray(sig, dtype=float).ravel()
    n = len(sig)
    if wl < 1:
        wl = 1
    # Build NxW matrix with NaN padding
    ts = np.full((n, wl), np.nan)
    for j in range(wl):
        ts[j:, j] = sig[:n - j]
    wmin = np.nanmin(ts, axis=1)
    wmax = np.nanmax(ts, axis=1)
    return wmin, wmax


def tma_filter(sig, wl, pct=0.25):
    """Trimmed Moving Average high-pass filter (tma_filter.m).
    Returns (filt_sig, tmavg) where filt_sig = sig - tmavg.
    """
    sig = np.asarray(sig, dtype=float).ravel()
    n = len(sig)
    if wl % 2 != 1:
        wl += 1
    half = (wl - 1) // 2

    # Build (n+wl-1) x wl matrix, then extract centered rows
    ts = np.full((n + wl - 1, wl), np.nan)
    for j in range(wl):
        ts[j:n + j, j] = sig
    ts = ts[half:half + n, :]  # centered: rows (wl+1)/2-1 to n+(wl-1)/2-1

    # Sort each row and trim
    ts = np.sort(ts, axis=1)
    trim_left = round(wl * pct / 2)
    trim_right = round(wl * (1 - pct) / 2)
    trimmed = ts[:, trim_left:wl - trim_right]

    # nanmean along axis=1
    with np.errstate(invalid='ignore'):
        tmavg = np.nanmean(trimmed, axis=1)
    filt_sig = sig - tmavg
    return filt_sig, tmavg


def tmzscore_filter(sig, wl, pct=0.25):
    """Trimmed Moving Z-score normalization (tmzscore_filter.m).
    Returns (filt_sig, tmavg, tmstd).
    """
    sig = np.asarray(sig, dtype=float).ravel()
    n = len(sig)
    if wl % 2 != 1:
        wl += 1
    half = (wl - 1) // 2

    # --- Pass 1: trimmed moving average ---
    sig_tma, tmavg = tma_filter(sig, wl, pct)

    # --- Pass 2: trimmed moving std on residuals ---
    ts2 = np.full((n + wl - 1, wl), np.nan)
    for j in range(wl):
        ts2[j:n + j, j] = sig_tma
    ts2 = ts2[half:half + n, :]
    ts2 = np.sort(ts2, axis=1)

    trim_left = round(wl * pct / 2)
    trim_right = round(wl * (1 - pct) / 2)
    trimmed2 = ts2[:, trim_left:wl - trim_right]

    with np.errstate(invalid='ignore'):
        tmstd = np.nanstd(trimmed2, ddof=1, axis=1)  # opt=0 -> n-1
    tmstd[tmstd == 0] = np.nan

    filt_sig = sig_tma / tmstd
    return filt_sig, tmavg, tmstd


# ============================================================
# Layer 3: R-peak detection
# ============================================================

def mvqrs_checkbeat(sig, fs, wl, beat_min, beat_max, threshold):
    """Beat annotation via adaptive threshold + constancy criterion.
    Faithful port of mvqrs_checkbeat.m.
    Returns (Ann, wmin, wmax, thr).
    """
    sig = np.asarray(sig, dtype=float).ravel()
    n = len(sig)

    # Windowed extrema of the feature signal
    wl_max = round(fs * 60 / beat_min)
    wl_min = round(fs * 60 / (0.5 * beat_max))
    _, wmax = windowed_extrema(sig, wl_max)
    wmin, _ = windowed_extrema(sig, wl_min)

    # Causal moving average smoothing (MATLAB filter())
    kernel_max = np.ones(wl_max) / wl_max
    wmax = sp_signal.lfilter(kernel_max, 1, wmax)
    kernel_min = np.ones(round(fs * 60 / beat_max)) / round(fs * 60 / beat_max)
    wmin = sp_signal.lfilter(kernel_min, 1, wmin)

    # Adaptive threshold with shifted indices (MATLAB circular indexing)
    shift_max = round(fs * 60 / beat_min)
    shift_min = round(fs * 60 / beat_max)

    # wmax shifted: indices [shift_max:end, repeat last]
    idx_wmax = np.concatenate([
        np.arange(shift_max, n),
        np.full(shift_max, n - 1)
    ]).astype(int)
    idx_wmax = np.clip(idx_wmax, 0, n - 1)

    # wmin shifted: indices [shift_min:end, repeat last]
    idx_wmin = np.concatenate([
        np.arange(shift_min, n),
        np.full(shift_min, n - 1)
    ]).astype(int)
    idx_wmin = np.clip(idx_wmin, 0, n - 1)

    thr_range = wmax[idx_wmax] - wmin[idx_wmin]
    thr = threshold * thr_range + wmin[idx_wmin]
    thr8 = 0.8 * threshold * thr_range + wmin[idx_wmin]

    # Threshold check
    thr_g = sig > thr       # above main threshold
    thr_s = sig <= thr8     # below lower threshold

    # Constancy criterion: runs of diff(sig)==0 of length >= wl-1
    const = np.diff(sig) == 0  # length n-1
    target = np.ones(max(1, wl - 1), dtype=float)
    # strfind equivalent: find all positions where target pattern starts
    cc = []
    target_len = len(target)
    for i in range(len(const) - target_len + 1):
        if np.all(const[i:i + target_len]):
            cc.append(i)  # 0-indexed position in const, maps to sig index i
    cc = np.array(cc, dtype=int)

    if len(cc) == 0:
        return np.array([], dtype=int), wmin, wmax, thr

    # Keep only those above threshold
    cc = cc[thr_g[cc]]
    if len(cc) == 0:
        return np.array([], dtype=int), wmin, wmax, thr

    # Select beginning of constant parts (non-consecutive starts)
    mask = np.concatenate([[True], np.diff(cc) > 1])
    cc_accept = cc[mask]

    # Find inflection points (rising edges of thr_g)
    diff_thr_g = np.diff(thr_g.astype(int))
    ip = np.concatenate([[0], np.where(diff_thr_g == 1)[0]])  # 0-indexed

    # For each cc_accept, find last ip <= cc_accept
    Ann = np.zeros(len(cc_accept), dtype=int)
    for i in range(len(cc_accept)):
        valid_ip = ip[ip <= cc_accept[i]]
        if len(valid_ip) > 0:
            Ann[i] = valid_ip[-1]
        else:
            Ann[i] = 0
    Ann = np.unique(Ann)

    # Rectangle merge: check if consecutive annotations share same rectangle
    diff_thr_s = np.diff(thr_s.astype(int))
    ip2 = np.concatenate([[0], np.where(diff_thr_s == 1)[0]])

    if len(Ann) > 1:
        for i in range(1, len(Ann)):
            if np.sum(ip2 < Ann[i]) == np.sum(ip2 < Ann[i - 1]):
                Ann[i] = Ann[i - 1]
    Ann = np.unique(Ann)

    return Ann, wmin, wmax, thr


def mvqrs_ann(sig, fs, wl_we, beat_min, beat_max, threshold, R):
    """Beat annotation with signal quality validation.
    Faithful port of mvqrs_ann.m.
    Returns Ann (valid beat positions, 0-indexed).
    """
    sig = np.asarray(sig, dtype=float).ravel()
    n = len(sig)
    valid = np.ones(n, dtype=int)

    # Windowed extrema on the signal
    if np.isscalar(wl_we):
        tmpmin, tmpmax = windowed_extrema(sig, wl_we)
    else:
        tmpmin, _ = windowed_extrema(sig, wl_we[0])
        _, tmpmax = windowed_extrema(sig, wl_we[1])

    # Feature = peak-to-trough, then checkbeat
    feature = tmpmax - tmpmin
    wl_cb = round(fs / 25)
    Ann, wmin_cb, wmax_cb, thr = mvqrs_checkbeat(
        feature, fs, wl_cb, beat_min, beat_max, threshold
    )

    # Signal quality validation via range filter
    shift_wmax = round(fs * 60 / beat_min)
    shift_wmin = round(fs * 60 / beat_max)

    # range = wmax[shift_wmax:] - wmin[shift_wmin:end-shift_wmax+shift_wmin]
    end_idx = n - shift_wmax + shift_wmin
    if shift_wmax < n and shift_wmin < end_idx:
        rng = wmax_cb[shift_wmax:n] - wmin_cb[shift_wmin:end_idx]
        # Pad to length n
        rng_full = np.empty(n)
        rng_full[:len(rng)] = rng
        rng_full[len(rng):] = rng[-1] if len(rng) > 0 else 0
    else:
        rng_full = np.ones(n)

    # Mark invalid regions
    half_min = round(shift_wmin / 2)
    half_max = round(shift_wmax / 2)
    for k in range(n):
        if rng_full[k] <= R:
            lo = max(0, k - half_min)
            hi = min(n, k + half_max + 1)
            valid[lo:hi] = 0

    # Keep only valid annotations
    if len(Ann) > 0:
        Ann = Ann[(Ann >= 0) & (Ann < n)]
        Ann = Ann[valid[Ann] == 1]

    return Ann


def singleqrs(signal, fs, threshold=0.5, downsampling=None,
              wl_tma=None, wl_we=None, pct=0.25,
              R=0.4, beat_min=50, beat_max=220):
    """QRS detection. Port of singleqrs.m.
    Returns 0-indexed R-peak indices in original sample rate.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    if downsampling is None:
        downsampling = fs
    if wl_tma is None:
        wl_tma = int(np.ceil(0.2 * fs))
    if wl_we is None:
        wl_we = int(np.ceil(fs / 3))

    # 1. Downsample via causal moving average + decimation
    factor = max(1, int(np.floor(fs / downsampling)))
    if factor > 1:
        kernel = np.ones(factor) / factor
        sig = sp_signal.lfilter(kernel, 1, signal)[::factor]
    else:
        sig = signal.copy()
    fs_new = fs / factor

    # 2. TMA high-pass filter
    wl_tma_ds = int(np.ceil(wl_tma / factor))
    sig_tma, _ = tma_filter(sig, wl_tma_ds, pct)

    # 3. Global z-score (nanzscore with opt=0 → ddof=1)
    sig_z = nanzscore(sig_tma)
    sig_z = np.nan_to_num(sig_z, nan=0.0)

    # 4. Annotation detection
    if np.isscalar(wl_we):
        wl_we_ds = int(np.ceil(wl_we / factor))
    else:
        wl_we_ds = [int(np.ceil(w / factor)) for w in wl_we]
    ann = mvqrs_ann(sig_z, fs_new, wl_we_ds,
                    beat_min, beat_max, threshold, R)

    # 5. Valid range filter + map back to original fs
    ann = ann[(ann > 0) & (ann < len(sig_z))]
    ann_orig = ann * factor

    # 6. Refine peak position in original signal
    refined = []
    search = max(1, factor)
    for p in ann_orig:
        lo = max(0, int(p) - search)
        hi = min(len(signal), int(p) + search + 1)
        refined.append(lo + np.argmax(np.abs(signal[lo:hi])))
    return np.array(refined, dtype=int)


# ============================================================
# Layer 4: RR artifact rejection
# ============================================================

def rrx(RR, grade=1):
    """Relative RR intervals. Port of HRV.rrx."""
    RR = np.asarray(RR, dtype=float).ravel()
    prefix = np.full(grade, np.nan)
    rel = 2 * (RR[grade:] - RR[:-grade]) / (RR[grade:] + RR[:-grade])
    return np.concatenate([prefix, rel])


def RRfilter(RR, limit=20):
    """Multi-pass artifact rejection. Port of HRV.RRfilter.
    Returns RR array with NaN at artifact positions.
    """
    RR = np.asarray(RR, dtype=float).ravel().copy()
    lim = max(limit, 50)

    # Pass 1: RR > 4s
    RR[RR > 4] = np.nan

    # Pass 2: single unrecognized beat
    rr_pct = 100 * rrx(RR)
    for i in range(len(RR) - 1):
        if rr_pct[i] > lim and (i + 1 < len(rr_pct)):
            if rr_pct[i + 1] < -lim:
                RR[i] = np.nan
    rr_pct = 100 * rrx(RR)

    # Pass 3: iterative threshold (80 → limit, step -10)
    rem_val = limit % 10
    for wbp_lim in range(80 - rem_val, limit - rem_val - 1, -10):
        d = np.abs(np.diff(rr_pct))
        wbp = np.where(d > wbp_lim)[0]
        if len(wbp) < 3:
            rr_pct = 100 * rrx(RR)
            continue
        # Find consecutive triplets
        d1 = np.diff(wbp)
        mask1 = d1 == 1
        # wbp where wbp[i], wbp[i+1], wbp[i+2] consecutive
        mask2 = mask1[:-1] & mask1[1:]
        idx = wbp[:-2][mask2]
        for k in idx:
            if k + 2 < len(RR):
                RR[k + 1] = np.nan
            if k + 3 < len(RR):
                RR[k + 2] = np.nan
        rr_pct = 100 * rrx(RR)

    # Pass 4: unreasonable differences after NaN
    nan_pos = np.where(np.isnan(RR[:-2]))[0]
    rr_pct = 100 * rrx(RR)
    for p in nan_pos:
        if p + 2 < len(rr_pct):
            if np.abs(rr_pct[p + 2]) > 15:
                if p + 1 < len(RR):
                    RR[p + 1] = np.nan
    rr_pct = 100 * rrx(RR)

    # Pass 5: remaining large rr_pct
    postmp = np.where(np.abs(rr_pct) > lim)[0]
    for p in postmp:
        if p - 1 >= 0:
            RR[p - 1] = np.nan
        RR[p] = np.nan

    return RR


# ============================================================
# Layer 5: Time domain metrics
# ============================================================

def SDNN(RR, flag=1):
    """Standard deviation of NN intervals (seconds → ms).
    flag=1→÷n, flag=0→÷n-1.
    """
    return nanstd(RR, flag=flag) * 1000


def RMSSD(RR, flag=1):
    """Root mean square of successive differences (ms).
    flag=1→÷n, flag=0→÷(n-1).
    """
    RR = np.asarray(RR, dtype=float)
    valid = ~np.isnan(RR)
    rr_clean = RR[valid]
    dRR2 = np.diff(rr_clean) ** 2
    n = len(dRR2)
    if n == 0:
        return np.nan
    return np.sqrt(np.nansum(dRR2) / (n - 1 + flag)) * 1000


def SDSD(RR, flag=1):
    """Std dev of successive differences (ms)."""
    RR = np.asarray(RR, dtype=float)
    valid = ~np.isnan(RR)
    dRR = np.diff(RR[valid])
    return nanstd(dRR, flag=flag) * 1000


def pNN50(RR, flag=1):
    """Percentage of successive diffs > 50ms."""
    RR = np.asarray(RR, dtype=float)
    valid = ~np.isnan(RR)
    dRR = np.abs(np.diff(RR[valid]))
    n = len(dRR)
    if n == 0:
        return np.nan
    count = np.sum(dRR > 0.05)
    return count / (n - 1 + flag) * 100


def HR(RR):
    """Heart rate in bpm. Port of HRV.HR (num=0)."""
    RR = np.asarray(RR, dtype=float)
    valid = ~np.isnan(RR)
    n = np.sum(valid)
    if n == 0:
        return np.nan
    return 60 * n / np.nansum(RR)


# ============================================================
# Layer 6: Frequency domain (FFT-based)
# ============================================================

def fft_val_fun(RR, Fs, interp_type='spline'):
    """Spectral analysis via FFT on interpolated RR tachogram.
    Port of HRV.fft_val_fun.
    Returns dict with pLF, pHF, LFHFratio, VLF, LF, HF, f, NFFT.
    """
    RR = np.asarray(RR, dtype=float).ravel()
    nan_result = {
        'pLF': np.nan, 'pHF': np.nan, 'LFHFratio': np.nan,
        'VLF': np.nan, 'LF': np.nan, 'HF': np.nan,
        'f': np.nan, 'NFFT': np.nan,
    }

    if np.any(np.isnan(RR)) or len(RR) <= 1:
        return nan_result

    # Interpolation to uniform grid at Fs Hz
    ANN = np.cumsum(RR) - RR[0]
    t_uniform = np.arange(0, ANN[-1], 1.0 / Fs)
    if interp_type == 'spline':
        cs = CubicSpline(ANN, RR)
        RR_rsmp = cs(t_uniform)
    elif interp_type == 'linear':
        RR_rsmp = np.interp(t_uniform, ANN, RR)
    else:
        RR_rsmp = RR

    L = len(RR_rsmp)
    if L == 0:
        return nan_result

    # Z-score normalize (opt=0 → ddof=1)
    RR_z = nanzscore(RR_rsmp)

    # FFT
    NFFT = int(2 ** np.ceil(np.log2(L)))
    Y = np.fft.fft(RR_z, NFFT) / L
    f = (Fs / 2) * np.linspace(0, 1, NFFT // 2 + 1)

    YY = 2 * np.abs(Y[:NFFT // 2 + 1])
    YY = YY ** 2

    # Band power by sum (matching MATLAB)
    VLF = np.sum(YY[f <= 0.04])
    LF = np.sum(YY[f <= 0.15]) - VLF
    HF = np.sum(YY[f <= 0.4]) - VLF - LF
    TP = VLF + LF + HF

    denom = TP - VLF
    pLF = LF / denom * 100 if denom > 0 else np.nan
    pHF = HF / denom * 100 if denom > 0 else np.nan
    LFHFratio = LF / HF if HF > 0 else np.nan

    return {
        'pLF': pLF, 'pHF': pHF, 'LFHFratio': LFHFratio,
        'VLF': VLF, 'LF': LF, 'HF': HF,
        'f': f, 'NFFT': NFFT,
    }


# ============================================================
# Layer 7: Nonlinear metrics
# ============================================================

def DFA(RR, boxsize_short=None, boxsize_long=None, grade=1):
    """Detrended Fluctuation Analysis. Port of HRV.DFA.
    Returns (alpha1, alpha2).
    """
    RR = np.asarray(RR, dtype=float).ravel()
    RR = RR[~np.isnan(RR)]
    n = len(RR)
    if boxsize_short is None:
        boxsize_short = np.arange(4, 17)
    if boxsize_long is None:
        boxsize_long = np.arange(16, 65)
    boxsize = np.concatenate([boxsize_short, boxsize_long])

    y = np.cumsum(RR - np.nanmean(RR))

    F = np.full(len(boxsize), np.nan)
    for bs_idx, bs_tmp in enumerate(boxsize):
        trend = np.zeros(n)
        num_boxes = int(np.floor((n - 2) / bs_tmp))
        for i in range(num_boxes + 1):
            if i == num_boxes:
                x = np.arange(i * bs_tmp, n)
            else:
                x = np.arange(i * bs_tmp, (i + 1) * bs_tmp)
            if len(x) < grade + 1:
                continue
            coeffs = np.polyfit(x, y[x], grade)
            trend[x] = np.polyval(coeffs, x)
        F[bs_idx] = np.sqrt(np.sum((y - trend) ** 2) / n)

    # Log-log regression for short and long ranges
    n_short = len(boxsize_short)
    F_short = F[:n_short]
    F_long = F[n_short:]

    valid_s = ~np.isnan(F_short) & (F_short > 0)
    valid_l = ~np.isnan(F_long) & (F_long > 0)

    if np.sum(valid_s) > 1:
        p1 = np.polyfit(np.log(boxsize_short[valid_s]),
                        np.log(F_short[valid_s]), 1)
        alpha1 = p1[0]
    else:
        alpha1 = np.nan

    if np.sum(valid_l) > 1:
        p2 = np.polyfit(np.log(boxsize_long[valid_l]),
                        np.log(F_long[valid_l]), 1)
        alpha2 = p2[0]
    else:
        alpha2 = np.nan

    return alpha1, alpha2


def returnmap_val(RR, flag=1):
    """Poincare SD1/SD2 via 45-degree rotation. Port of HRV.returnmap_val.
    Returns (SD1, SD2, SD1SD2ratio) in ms.
    """
    RR = np.asarray(RR, dtype=float).ravel()
    valid = ~np.isnan(RR)
    rr = RR[valid]
    if len(rr) < 2:
        return np.nan, np.nan, np.nan

    X = np.array([rr[:-1], rr[1:]])  # 2 x (n-1)
    alpha = -45 * np.pi / 180
    R_mat = np.array([[np.cos(alpha), -np.sin(alpha)],
                       [np.sin(alpha), np.cos(alpha)]])
    XR = R_mat @ X

    SD2 = nanstd(XR[0, :], flag=flag) * 1000
    SD1 = nanstd(XR[1, :], flag=flag) * 1000
    ratio = SD1 / SD2 if SD2 > 0 else np.nan
    return SD1, SD2, ratio


# ============================================================
# Layer 8: Top-level convenience function
# ============================================================

def analyze_vollmer(ecg_signal, fs, threshold=0.5,
                    rr_limit=20, fft_fs=1000):
    """Full Vollmer HRV pipeline.
    Parameters
    ----------
    ecg_signal : array — raw ECG (single channel)
    fs : int — sampling rate (Hz)
    threshold : float — QRS detection threshold (0-1)
    rr_limit : int — RR filter limit (%)
    fft_fs : int — resampling rate for FFT

    Returns dict with metrics, r_peaks, rr_intervals, etc.
    """
    ecg_signal = np.asarray(ecg_signal, dtype=float).ravel()

    # 1. R-peak detection
    r_peaks = singleqrs(ecg_signal, fs, threshold=threshold)

    # 2. RR intervals (seconds)
    rr_raw = np.diff(r_peaks) / fs

    # 3. Artifact rejection
    rr_filt = RRfilter(rr_raw, limit=rr_limit)
    valid_mask = ~np.isnan(rr_filt)
    rr_clean = rr_filt[valid_mask]

    if len(rr_clean) < 5:
        return {
            'metrics': {},
            'r_peaks': r_peaks,
            'rr_raw': rr_raw,
            'rr_filtered': rr_filt,
        }

    # 4. Time domain
    hr = HR(rr_clean)
    sdnn = SDNN(rr_clean, flag=1)
    rmssd = RMSSD(rr_clean, flag=1)
    sdsd = SDSD(rr_clean, flag=1)
    pnn50 = pNN50(rr_clean, flag=1)

    # 5. Frequency domain
    fft_result = fft_val_fun(rr_clean, fft_fs)

    # 6. Nonlinear
    alpha1, alpha2 = DFA(rr_clean)
    sd1, sd2, sd_ratio = returnmap_val(rr_clean, flag=1)

    return {
        'metrics': {
            'HR': round(hr, 2),
            'SDNN': round(sdnn, 2),
            'RMSSD': round(rmssd, 2),
            'SDSD': round(sdsd, 2),
            'pNN50': round(pnn50, 2),
            'nLF': round(fft_result['pLF'], 2),
            'nHF': round(fft_result['pHF'], 2),
            'LF/HF': round(fft_result['LFHFratio'], 4),
            'VLF': fft_result['VLF'],
            'LF': fft_result['LF'],
            'HF': fft_result['HF'],
            'DFA_alpha1': round(alpha1, 4),
            'DFA_alpha2': round(alpha2, 4),
            'SD1': round(sd1, 2),
            'SD2': round(sd2, 2),
            'SD1_SD2': round(sd_ratio, 4),
        },
        'r_peaks': r_peaks,
        'rr_raw': rr_raw,
        'rr_filtered': rr_filt,
        'rr_clean': rr_clean,
        'n_removed': int(np.sum(~valid_mask)),
    }
