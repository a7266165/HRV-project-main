# import reportlab

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    print("ReportLab is installed and the necessary modules are accessible.")
    
    # 測試中文字體
    font_loaded = False
    try:
        # 嘗試註冊常見的中文字體
        pdfmetrics.registerFont(TTFont('Microsoft-YaHei', 'msyh.ttc'))
        print("成功載入微軟雅黑字體")
        font_name = 'Microsoft-YaHei'
        font_loaded = True
    except:
        try:
            pdfmetrics.registerFont(TTFont('SimSun', 'simsun.ttc'))
            print("成功載入宋體字體")
            font_name = 'SimSun'
            font_loaded = True
        except:
            print("警告：無法載入中文字體，請確認系統字體路徑")
    
    # 測試生成繁體中文PDF
    if font_loaded:
        c = canvas.Canvas("test_chinese.pdf", pagesize=letter)
        c.setFont(font_name, 16)
        c.drawString(100, 750, "測試繁體中文PDF生成")
        c.drawString(100, 720, "這是一個測試文件")
        c.drawString(100, 690, "包含繁體中文字元：台灣、資料、測試")
        c.save()
        print("成功生成 test_chinese.pdf 檔案")
            
except ImportError as e:
    print("ReportLab is not installed or some modules are missing:", e)