from __future__ import annotations
import os
import cv2
import numpy as np

class HaarFaceDetector:
    def __init__(self) -> None:
        haar_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        lbp_path  = os.path.join(cv2.data.haarcascades, "lbpcascade_frontalface.xml")

        if not os.path.exists(haar_path):
            raise FileNotFoundError(f"Haar cascade file not found: {haar_path}")
        if not os.path.exists(lbp_path):
            raise FileNotFoundError(f"LBP cascade file not found: {lbp_path}")

        self.haar = cv2.CascadeClassifier(haar_path)
        self.lbp  = cv2.CascadeClassifier(lbp_path)

        self.scaleFactor = 1.15
        self.minNeighbors = 5
        self.minSize = 40
        self.clipLimit = 2.0
        self.use_invert = False

        # ★ 처리된 프레임(plain/clahe/invert) 자체를 프리뷰로 보여줄지 여부
        self.preview_preproc = True

    def set_params(self, sf, mn, ms, clip, inv):
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
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def detect(self, rgb_img: np.ndarray):
        # normalize size
        h, w = rgb_img.shape[:2]
        scale = 800.0 / max(h, w)
        if scale < 1.0:
            rgb = cv2.resize(rgb_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            rgb = rgb_img.copy()

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

       # haar_detector.py detect() 핵심 패치
        # 1) 먼저 모든 후보 미리 만들기
        g_plain = gray
        g_clahe = self._clahe(gray)
        g_inv = cv2.bitwise_not(gray)
        g_clahe_inv = cv2.bitwise_not(g_clahe)

        # 2) 프리뷰로 쓸 g를 "항상" 선택 (검출 성공/실패와 무관)
        if self.preview_preproc:
            # 원하는 규칙: invert면 clahe_inv, 아니면 clahe
            preview_g = g_clahe_inv if self.use_invert else g_clahe
        else:
            preview_g = rgb  # 원본

        # 3) 검출용 시나리오 (기존과 동일)
        scenarios = [("haar", g_plain), ("haar", g_clahe), ("lbp", g_plain), ("lbp", g_clahe)] \
                    if not self.use_invert else \
                    [("haar", g_inv), ("haar", g_clahe_inv), ("lbp", g_inv), ("lbp", g_clahe_inv)]

        # 4) 검출 성공/실패와 상관없이 프리뷰는 preview_g로 고정
        for kind, g in scenarios:
            classifier = self.haar if kind == "haar" else self.lbp
            faces = self._detect_once(g, classifier)
            if len(faces) > 0:
                result = cv2.cvtColor(preview_g, cv2.COLOR_GRAY2RGB) if self.preview_preproc else preview_g.copy()
                for (x, y, wf, hf) in faces:
                    cv2.rectangle(result, (x, y), (x + wf, y + hf), (0, 255, 0), 2)
                return result, faces.tolist()

        # 검출 실패도 동일 프리뷰 유지
        result = cv2.cvtColor(preview_g, cv2.COLOR_GRAY2RGB) if self.preview_preproc else preview_g.copy()
        return result, []
