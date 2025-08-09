from __future__ import annotations
import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from controllers.detect_controller import DetectController


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    ctrl = DetectController(win)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()