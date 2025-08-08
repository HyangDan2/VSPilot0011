from __future__ import annotations
import cv2
import numpy as np
from PySide6.QtWidgets import QFileDialog
from ..ui.main_window import MainWindow
from ..detectors.haar_detector import HaarFaceDetector
from ..detectors.onnx_detector import ONNXFaceDetector, HAS_ORT
from ..utils.image_utils import ensure_rgb, ensure_bgr


class DetectController:
    def __init__(self, win: MainWindow) -> None:
        self.win = win
        self.original_rgb: np.ndarray | None = None
        self.result_rgb: np.ndarray | None = None
        self.haar = HaarFaceDetector()
        self.onnx: ONNXFaceDetector | None = None
        self.onnx_model_path: str | None = None

        # wiring
        win.backendChanged.connect(self._on_backend_change)
        win.onnxModelChosen.connect(self._load_onnx)
        win.haar_panel.changed.connect(self.run)  # autorun
        win.onnx_panel.changed.connect(self.run)  # autorun
        win.loadRequested.connect(self.load_image)
        win.runRequested.connect(self.run)
        win.saveRequested.connect(self.save_image)

    # --- actions ---
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self.win, "Open image", filter="Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        bgr = cv2.imread(path)
        if bgr is None:
            self.win.show_error("Open", "이미지를 열 수 없습니다.")
            return
        self.original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.win.left.set_rgb(self.original_rgb)
        self.run()

    def save_image(self):
        if self.result_rgb is None:
            return
        path, _ = QFileDialog.getSaveFileName(self.win, "Save result", filter="PNG (*.png)")
        if not path:
            return
        cv2.imwrite(path, cv2.cvtColor(self.result_rgb, cv2.COLOR_RGB2BGR))

    def _on_backend_change(self, backend: str):
        if backend == "ONNX":
            if not HAS_ORT:
                self.win.show_error("onnxruntime missing", "pip install onnxruntime 또는 onnxruntime-gpu 로 설치하세요.")
                self.win.backend.setCurrentText("Haar")
                return
            self.win.haar_panel.hide(); self.win.onnx_panel.show()
        else:
            self.win.onnx_panel.hide(); self.win.haar_panel.show()
        self.run()

    def _load_onnx(self, path: str):
        try:
            provider = self.win.onnx_panel.provider.currentText()
            import onnxruntime as ort
            avail = ort.get_available_providers()
            req = [provider] if provider in avail else ["CPUExecutionProvider"]
            if provider not in avail:
                self.win.show_warn("Provider fallback", f"{provider} 사용 불가. 사용 가능: {avail}\nCPUExecutionProvider로 폴백합니다.")
            self.onnx = ONNXFaceDetector(path, providers=req)
            self.onnx_model_path = path
            self.win.show_info("ONNX", f"모델 로드 완료: {path}\nProvider: {req[0]}")
        except Exception as e:
            self.onnx = None
            self.win.show_error("ONNX Load Error", str(e))

    def run(self):
        if self.original_rgb is None:
            return
        backend = self.win.backend.currentText()
        if backend == "Haar":
            sf, mn, ms, clip, inv = self.win.haar_panel.params()
            self.haar.set_params(sf, mn, ms, clip, inv)
            out, faces = self.haar.detect(self.original_rgb)
        else:
            if self.onnx is None:
                if not self.onnx_model_path:
                    self.win.show_warn("ONNX", "먼저 ONNX 모델을 로드하세요.")
                    return
                try:
                    provider = self.win.onnx_panel.provider.currentText()
                    import onnxruntime as ort
                    avail = ort.get_available_providers()
                    req = [provider] if provider in avail else ["CPUExecutionProvider"]
                    self.onnx = ONNXFaceDetector(self.onnx_model_path, providers=req)
                except Exception as e:
                    self.win.show_error("ONNX Init Error", str(e))
                    return
            in_w, in_h, conf, iou, clip, inv, _prov = self.win.onnx_panel.params()
            try:
                self.onnx.set_params(in_w, in_h, conf, iou, clip, inv)
                out, faces = self.onnx.detect(self.original_rgb)
            except Exception as e:
                self.win.show_error("ONNX Inference Error", str(e))
                return
        self.result_rgb = out
        self.win.right.set_rgb(out)