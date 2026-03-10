from PyQt6.QtCore import QThread, pyqtSignal

from ..core.tff_reader import read_tff_file
from ..core.preprocessing import preprocess_ecg
from ..core.hrv_analysis import analyze_hrv
from ..core.report_generator import generate_report


class AnalysisWorker(QThread):
    """Background thread for running the full HRV analysis pipeline."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, file_path, channel_index=0):
        super().__init__()
        self.file_path = file_path
        self.channel_index = channel_index

    def run(self):
        try:
            self.progress.emit("讀取 TFF 檔案...")
            file_data = read_tff_file(self.file_path)

            self.progress.emit("訊號前處理（濾波 + 降取樣）...")
            ecg_signal = file_data['signal'][:, self.channel_index]
            ecg_processed = preprocess_ecg(ecg_signal,
                                           original_fs=file_data['fs'])

            self.progress.emit("HRV 分析中...")
            hrv_results = analyze_hrv(ecg_processed, sampling_rate=1000)
            hrv_results['file_data'] = file_data

            self.finished.emit(hrv_results)
        except Exception as e:
            self.error.emit(str(e))


class ReportWorker(QThread):
    """Background thread for generating the PDF report."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, output_path, patient_info, hrv_results,
                 analysis_text, recommendation_text):
        super().__init__()
        self.output_path = output_path
        self.patient_info = patient_info
        self.hrv_results = hrv_results
        self.analysis_text = analysis_text
        self.recommendation_text = recommendation_text

    def run(self):
        try:
            self.progress.emit("產生 PDF 報告...")
            generate_report(
                self.output_path,
                self.patient_info,
                self.hrv_results,
                self.analysis_text,
                self.recommendation_text,
            )
            self.finished.emit(self.output_path)
        except Exception as e:
            self.error.emit(str(e))
