from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple


def ensure_rgb(img_bgr_or_gray: np.ndarray) -> np.ndarray:
    if img_bgr_or_gray.ndim == 2:
        return cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)


def ensure_bgr(img_rgb_or_gray: np.ndarray) -> np.ndarray:
    if img_rgb_or_gray.ndim == 2:
        return cv2.cvtColor(img_rgb_or_gray, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img_rgb_or_gray, cv2.COLOR_RGB2BGR)


def normalized_for_display(rgb: np.ndarray, max_side: int = 800) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        return cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return rgb.copy()


def to_qimage(rgb: np.ndarray):
    from PySide6.QtGui import QImage
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()