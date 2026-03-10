TEMPLATES = {
    "sdnn_low": {
        "label": "SDNN偏低",
        "analysis": (
            "綜合以上數值，可以得到下列資訊：\n"
            "在時域分析中，您的SDNN比標準值稍微偏低，可能處於緊張狀態、"
            "壓力過大或身體不適之情況，並請小心潛在的心血管風險。\n"
            "在頻域分析中，您的數據顯示基礎期交感神經較活躍；壓力期仍偏向"
            "交感神經活躍；恢復期交感神經活性上升，您的整體自律神經已恢復至"
            "偏向交感神經活躍。"
        ),
        "recommendation": (
            "建議可以多做運動來改善心率變異度。\n"
            "若有運動時胸悶、胸痛或氣促之症狀，請至心血管內科門診求診。"
        ),
    },
    "arrhythmia": {
        "label": "心律不整",
        "analysis": (
            "綜合以上數值，可以得到下列資訊：\n"
            "您的檢測結果顯示有心律不整的情況，檢驗數值並無法依照本檢查之"
            "常規標準來判讀，不適用本項檢測。"
        ),
        "recommendation": (
            "在您的檢查中發現有心律不整。而在心律不整的情況下，並不適用於"
            "本項檢測。因此無法正確判讀。\n"
            "若有心悸、暈眩、昏倒等症狀，請至心臟內科門診求診。"
        ),
    },
    "noise": {
        "label": "雜訊干擾",
        "analysis": (
            "綜合以上數值，可以得到下列資訊：\n"
            "因本檢查需要安靜平穩狀態下檢測才能準確判斷，您的檢測結果發現"
            "有很多的雜訊，有可能因為說話、移動等等原因而引起雜訊的產生，"
            "因此無法判讀。"
        ),
        "recommendation": "因雜訊無法判讀。",
    },
    "normal": {
        "label": "Normal",
        "analysis": (
            "綜合以上數值，可以得到下列資訊：\n"
            "在時域分析中，您的SDNN在標準值之上，心血管風險相對較低。\n"
            "在頻域分析中，您的數據顯示基礎期副交感神經較活躍；壓力期副交感"
            "神經活躍；恢復期交感神經活性下降，您的整體自律神經已恢復至偏向"
            "副交感神經活躍。"
        ),
        "recommendation": (
            "經評估後您的心率變異度高於標準值，心血管風險相對較低，"
            "建議可以多做運動以維持心率變異度。"
        ),
    },
}

CONDITION_ORDER = ["sdnn_low", "arrhythmia", "noise", "normal"]


def get_dropdown_labels():
    return [TEMPLATES[key]["label"] for key in CONDITION_ORDER]


def get_template(condition_key):
    return TEMPLATES[condition_key]


def get_key_by_index(index):
    return CONDITION_ORDER[index]
