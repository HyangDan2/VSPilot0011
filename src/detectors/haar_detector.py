from __future__ import annotations
import cv2
import numpy as np
import os


class HaarFaceDetector:
    def __init__(self) -> None:
        haar_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if not os.path.exists(haar_path):
            raise FileNotFoundError(f"Haar cascade file not found: {haar_path}")

        lbp_path = os.path.join(cv2.data.haarcascades, "lbpcascade_frontalface.xml")
        if not os.path.exists(lbp_path):
            raise FileNotFoundError(f"LBP cascade file not found: {lbp_path}")
        self.haar = cv2.CascadeClassifier(haar_path)
        self.lbp = cv2.CascadeClassifier(lbp_path)
        self.scaleFactor = 1.15
        self.minNeighbors = 5
        self.minSize = 40
        self.clipLimit = 2.0
        self.use_invert = False

    def set_params(self, sf: float, mn: int, ms: int, clip: float, inv: bool) -> None:
        self.scaleFactor = float(sf)
        self.minNeighbors = int(mn)
        self.minSize = int(ms)
        self.clipLimit = float(clip)
        self.use_invert = bool(inv)

    def _clahe(self, gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _detect_once(self, gray: np.ndarray, classifier) -> np.ndarray:
        faces = classifier.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=(self.minSize, self.minSize),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return faces

    def detect(self, rgb_img: np.ndarray):
        # normalize for speed/robustness
        h, w = rgb_img.shape[:2]
        scale = 800.0 / max(h, w)
        if scale < 1.0:
            rgb = cv2.resize(rgb_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            rgb = rgb_img.copy()

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        g_plain = gray
        g_clahe = self._clahe(gray)
        g_inv = cv2.bitwise_not(gray)
        g_clahe_inv = cv2.bitwise_not(g_clahe)

        if self.use_invert:
            scenarios = [("haar", g_inv), ("haar", g_clahe_inv), ("lbp", g_inv), ("lbp", g_clahe_inv)]
        else:
            scenarios = [("haar", g_plain), ("haar", g_clahe), ("lbp", g_plain), ("lbp", g_clahe)]

        result = rgb.copy()
        for kind, g in scenarios:
            classifier = self.haar if kind == "haar" else self.lbp
            faces = self._detect_once(g, classifier)
            if len(faces) > 0:
                for (x, y, wf, hf) in faces:
                    cv2.rectangle(result, (x, y), (x + wf, y + hf), (0, 255, 0), 2)
                return result, faces.tolist()
        return result, []