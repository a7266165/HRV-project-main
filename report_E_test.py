import os
print("--- 腳本啟動中 ---") # 新增這一行
from hrv_app.core.report_generator_Eng import generate_report

def run_test():
    # 1. 根據截圖資訊建立假數據
    patient_info = {
        'record_number': '100001',
        'name': '張三',
        'exam_time': '2016-08-30 12:54:42',
        'birth_date': '20021201'
    }

    hrv_results = {
        'metrics': {
            'HRV_SDNN': 139.23,
            'HRV_LF': 0.03,
            'HRV_HF': 0.04,
            'HRV_LF_HF': 0.90,
            'HRV_DFA_alpha1': 0.65,
            'LFnu': 42.9,
            'HFnu': 57.1
        }
    }#

    # 2. 截圖中的分析與建議文字
    analysis_text = (
        "綜合以上數值，可以得到下列資訊：\n"
        "在時域分析中，您的SDNN比標準值稍微偏低，可能處於緊張狀態、壓力過大或身體不適之情況，"
        "並請小心潛在的心血管風險。在頻域分析中，您的數據顯示基礎期交感神經較活躍；壓力期仍偏向交感神經活躍；"
        "恢復期交感神經活性上升，您的整體自律神經系統偏向交感神經活躍。"
    )

    recommendation_text = (
        "建議可以多做運動來改善心率變異度。若有運動時胸悶、胸痛或氣促之症狀，請至心血管內科門診求診。"
    )

    # 3. 指定輸出路徑
    output_pdf = "E2.pdf"

    print(f"正在生成測試報告: {output_pdf}...")
    
    try:
        generate_report(
            output_pdf, 
            patient_info, 
            hrv_results, 
            analysis_text, 
            recommendation_text
        )
        print("生成成功！請查看檔案。")
    except Exception as e:
        print(f"發生錯誤: {e}")

if __name__ == "__main__":
    run_test()