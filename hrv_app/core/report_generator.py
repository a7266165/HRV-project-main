import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from .plotting import (
    create_taichi_plot,
    create_rr_tachogram,
    create_poincare_plot,
    create_spectrum_plot,
)


def _setup_chinese_font():
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False


def generate_report(output_path, patient_info, hrv_results,
                    analysis_text, recommendation_text):
    """
    Generate a single-page A4 PDF report.

    Parameters
    ----------
    output_path : str
        Path for the output PDF file.
    patient_info : dict
        Keys: 'record_number', 'name', 'exam_time', 'birth_date'
    hrv_results : dict
        Output of analyze_hrv().
    analysis_text : str
        Analysis text (may be edited by user).
    recommendation_text : str
        Recommendation text (may be edited by user).
    """
    _setup_chinese_font()

    metrics = hrv_results['metrics']
    rr_intervals = hrv_results['rr_intervals']
    rr_times = hrv_results['rr_times']

    # Create A4-sized figure (8.27 x 11.69 inches)
    fig = plt.figure(figsize=(8.27, 11.69))

    # Main GridSpec: title, patient info, charts, metrics, text
    gs = GridSpec(6, 2, figure=fig,
                  height_ratios=[0.06, 0.04, 0.28, 0.18, 0.14, 0.30],
                  hspace=0.3, wspace=0.3,
                  left=0.08, right=0.92, top=0.95, bottom=0.03)

    # === Row 0: Title ===
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, '高醫保健科自律神經報告 -- 心律變異分析',
                  ha='center', va='center', fontsize=16, fontweight='bold')

    # === Row 1: Patient info ===
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis('off')
    record_num = patient_info.get('record_number', '')
    name = patient_info.get('name', '')
    exam_time = patient_info.get('exam_time', '')
    birth_date = patient_info.get('birth_date', '')

    info_text = (f'病歷號：{record_num}          '
                 f'檢查時間：{exam_time}\n'
                 f'姓名：{name}          '
                 f'出生日期：{birth_date}')
    ax_info.text(0.02, 0.5, info_text, ha='left', va='center',
                 fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0',
                           edgecolor='gray'))

    # === Row 2: Taichi plot (large) ===
    ax_taichi = fig.add_subplot(gs[2, :])
    ax_taichi.axis('off')

    # Generate taichi as a sub-figure and render into this axes
    lf_hf = metrics.get('HRV_LF_HF', 1.0) or 1.0
    lf_nu = metrics.get('LFnu')
    hf_nu = metrics.get('HFnu')
    taichi_fig = create_taichi_plot(lf_hf, lf_nu, hf_nu)
    taichi_fig.savefig(output_path + '.taichi.tmp.png', dpi=150,
                       bbox_inches='tight', transparent=True)
    plt.close(taichi_fig)
    taichi_img = plt.imread(output_path + '.taichi.tmp.png')
    ax_taichi.imshow(taichi_img, aspect='equal')
    ax_taichi.set_xlim(0, taichi_img.shape[1])
    ax_taichi.set_ylim(taichi_img.shape[0], 0)

    # === Row 3: RR Tachogram + Poincaré (side by side) ===
    ax_rr = fig.add_subplot(gs[3, 0])
    ax_rr.plot(rr_times, rr_intervals, color='black', linewidth=0.5)
    ax_rr.set_facecolor('#FFFFCC')
    ax_rr.set_xlabel('time (s)', fontsize=8)
    ax_rr.set_ylabel('sec', fontsize=8)
    ax_rr.set_title('RR Tachogram', fontweight='bold', color='red', fontsize=9)
    ax_rr.tick_params(labelsize=7)

    ax_poincare = fig.add_subplot(gs[3, 1])
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]
    ax_poincare.scatter(rr_n, rr_n1, s=1, color='steelblue', alpha=0.5)
    ax_poincare.set_xlabel('RRn', fontsize=8)
    ax_poincare.set_ylabel('RRn+1', fontsize=8)
    ax_poincare.set_title('Global Return map', fontweight='bold', fontsize=9)
    ax_poincare.set_aspect('equal')
    ax_poincare.tick_params(labelsize=7)

    # === Row 4: Spectrum ===
    ax_spectrum = fig.add_subplot(gs[4, :])
    # Inline spectrum plot
    from scipy.interpolate import interp1d
    from scipy.signal import welch
    import numpy as np

    interp_fs = 4.0
    t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0 / interp_fs)
    f_interp = interp1d(rr_times, rr_intervals, kind='cubic',
                        fill_value='extrapolate')
    rr_uniform = f_interp(t_uniform)
    freqs, psd = welch(rr_uniform, fs=interp_fs,
                       nperseg=min(256, len(rr_uniform)))

    ax_spectrum.plot(freqs, psd, color='black', linewidth=0.8)
    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
    ax_spectrum.fill_between(freqs[lf_mask], psd[lf_mask], alpha=0.3,
                             color='blue')
    ax_spectrum.fill_between(freqs[hf_mask], psd[hf_mask], alpha=0.3,
                             color='green')
    ax_spectrum.set_title('Spectrum of RR Tachogram', fontweight='bold',
                          fontsize=9)
    ax_spectrum.set_xlabel('Frequency (Hz)', fontsize=8)
    ax_spectrum.set_xlim(0, 0.5)
    ax_spectrum.tick_params(labelsize=7)

    psd_max = psd.max() if psd.max() > 0 else 1
    if lf_nu is not None and hf_nu is not None:
        ax_spectrum.text(0.5, 0.95,
                         f'LF/HF ratio = {lf_hf:.3f}',
                         transform=ax_spectrum.transAxes,
                         ha='center', va='top', fontweight='bold', fontsize=8)
        ax_spectrum.text(0.095, psd_max * 0.7,
                         f'LFnu\n{lf_nu:.2f}%', ha='center', fontsize=7)
        ax_spectrum.text(0.275, psd_max * 0.7,
                         f'HFnu\n{hf_nu:.2f}%', ha='center', fontsize=7)

    # === Row 5: Metrics + Analysis + Recommendation ===
    ax_text = fig.add_subplot(gs[5, :])
    ax_text.axis('off')

    sdnn = metrics.get('HRV_SDNN', '--')
    lf = metrics.get('HRV_LF', '--')
    hf = metrics.get('HRV_HF', '--')
    lf_hf_val = metrics.get('HRV_LF_HF', '--')
    dfa = metrics.get('HRV_DFA_alpha1', '--')

    metrics_line = (f'SDNN: {sdnn}    LF: {lf}    HF: {hf}    '
                    f'LF/HF: {lf_hf_val}    DFA α1: {dfa}')

    # Wrap long text
    analysis_wrapped = textwrap.fill(analysis_text, width=60)
    recommendation_wrapped = textwrap.fill(recommendation_text, width=60)

    full_text = (
        f'HRV 指標\n'
        f'{metrics_line}\n\n'
        f'【分析】\n{analysis_wrapped}\n\n'
        f'【建議】\n{recommendation_wrapped}'
    )

    ax_text.text(0.02, 0.98, full_text, ha='left', va='top',
                 fontsize=9, transform=ax_text.transAxes,
                 linespacing=1.4,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#FAFAFA',
                           edgecolor='gray'))

    # Save PDF
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=150)
    plt.close(fig)

    # Clean up temp file
    import os
    tmp_file = output_path + '.taichi.tmp.png'
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
