from __future__ import annotations
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
import numpy as np


class ImageView(QLabel):
    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("border: 1px solid #555; background:#222; color:#bbb;")

    def set_rgb(self, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))