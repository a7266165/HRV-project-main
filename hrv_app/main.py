import sys

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import QApplication
from .gui.main_window import MainWindow


def main():
    # Chinese font support
    plt.rcParams['font.sans-serif'] = [
        'Microsoft JhengHei', 'SimHei', 'sans-serif'
    ]
    plt.rcParams['axes.unicode_minus'] = False

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
