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
        self._last_rgb: np.ndarray | None = None
        self._last_qimg: QImage | None = None

    def set_rgb(self, rgb: np.ndarray) -> None:
        # 1) 연속 메모리 보장
        rgb = np.ascontiguousarray(rgb)
        self._last_rgb = rgb

        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()  # 2) copy로 깊복
        self._last_qimg = qimg

        self._render()

    def resizeEvent(self, e):
        self._render()
        return super().resizeEvent(e)

    def _render(self):
        if self._last_qimg is None:
            return
        pix = QPixmap.fromImage(self._last_qimg)
        self.setPixmap(
            pix.scaled(self.width(), self.height(),
                       Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
        )