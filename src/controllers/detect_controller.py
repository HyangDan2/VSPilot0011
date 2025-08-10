from __future__ import annotations
import cv2
import numpy as np
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import QTimer
from ui.main_window import MainWindow
from detectors.haar_detector import HaarFaceDetector
from detectors.face_detector import ONNXFaceDetector, HAS_ORT


class DetectController:
    """
    - 파라미터 변경: 50ms 디바운스 후 자동 실행
    - Run 버튼: 디바운스 우회 즉시 실행
    - 백엔드 전환: 즉시 실행
    - Haar: 처리된 프레임 프리뷰 켜둠 (preview_preproc=True)
    """
    def __init__(self, win: MainWindow) -> None:
        self.win = win
        self.original_rgb: np.ndarray | None = None
        self.result_rgb: np.ndarray | None = None

        self.haar = HaarFaceDetector()
        self.haar.preview_preproc = True  # ★ 처리 프리뷰 켜기
        self.onnx: ONNXFaceDetector | None = None
        self.onnx_model_path: str | None = None

        # 디바운스 타이머
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._run_now)

        # 연결
        win.loadRequested.connect(self.load_image)
        win.saveRequested.connect(self.save_image)
        win.runRequested.connect(self._run_now)
        win.backendChanged.connect(self._on_backend_change)
        win.haar_panel.changed.connect(self.run)
        win.onnx_panel.changed.connect(self.run)
        win.onnxModelChosen.connect(self._load_onnx)

    # -- Actions ------------------------------------------------------------
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self.win, "Open image", filter="Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
        bgr = cv2.imread(path)
        if bgr is None:
            self.win.show_error("Open", "이미지를 열 수 없습니다.")
            return
        self.original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.win.left.set_rgb(self.original_rgb)
        self._run_now()

    def save_image(self):
        if self.result_rgb is None:
            return
        path, _ = QFileDialog.getSaveFileName(self.win, "Save result", filter="PNG (*.png)")
        if not path:
            return
        cv2.imwrite(path, cv2.cvtColor(self.result_rgb, cv2.COLOR_RGB2BGR))

    # -- Auto-run (debounced) ----------------------------------------------
    def run(self):
        self._debounce.start(50)  # 필요 시 30~100ms 조절

    def _sync_onnx_params(self):
        """UI의 ONNX 파라미터를 디텍터에 반영. (고정/가변 분리)"""
        if not self.onnx:
            return
        # 하위 호환 시그니처 유지: (in_w, in_h, conf, iou, clahe, inv, provider)
        in_w, in_h, conf, iou, clahe, inv, _prov = self.win.onnx_panel.params()

        # ✅ 입력 크기는 고정값 강제
        self.onnx.input_width  = 640
        self.onnx.input_height = 640

        # ✅ 스레시홀드만 반영
        self.onnx.conf_threshold = float(conf)
        self.onnx.iou_threshold  = float(iou)

        # ✅ 좌표 클립은 항상 활성화 (이전 clip 오용 제거)
        self.onnx.clip_coords = True

        # ✅ 색상 반전은 고정 비활성화
        self.onnx.invert_colors = False

    def _init_onnx_session_from_ui_provider(self):
        import onnxruntime as ort
        provider = self.win.onnx_panel.provider.currentText()
        avail = ort.get_available_providers()
        req = [provider] if provider in avail else ["CPUExecutionProvider"]
        if provider not in avail:
            self.win.show_warn(
                "Provider fallback",
                f"{provider} 사용 불가. 사용 가능: {avail}\nCPUExecutionProvider로 폴백합니다."
            )
        self.onnx = ONNXFaceDetector(self.onnx_model_path, providers=req)

        # ✅ 세션 직후에도 한 번 더 고정/가변 동기화
        self._sync_onnx_params()

    def _run_now(self):
        """즉시 실행(디바운스 우회). Run 버튼/이미지 로드/백엔드 전환 등에서 사용."""
        if self.original_rgb is None:
            return

        backend = self.win.backend.currentText()
        try:
            if backend == "Haar":
                # 파라미터 적용
                sf, mn, ms, clip, inv = self.win.haar_panel.params()
                self.haar.set_params(sf, mn, ms, clip, inv)
                # Haar는 detect가 (이미지, faces)로 리턴
                out, _faces = self.haar.detect(self.original_rgb)

            else:  # ONNX
                if not HAS_ORT:
                    self.win.show_error("onnxruntime missing", "pip install onnxruntime 또는 onnxruntime-gpu 로 설치하세요.")
                    self.win.backend.setCurrentText("Haar")
                    return

                if self.onnx is None:
                    if not self.onnx_model_path:
                        self.win.show_warn("ONNX", "먼저 ONNX 모델을 로드하세요.")
                        return
                    self._init_onnx_session_from_ui_provider()

                # ★ UI 파라미터를 ONNX 디텍터에 반영
                self._sync_onnx_params()

                # ONNXFaceDetector.detect는 '검출 리스트(또는 dets, ms)'를 리턴함
                det_res = self.onnx.detect(self.original_rgb)
                if isinstance(det_res, tuple):
                    dets = det_res[0]  # (dets, [ms]) 대응
                else:
                    dets = det_res

                # ★ 시각화 이미지 생성
                out = self.onnx.draw(self.original_rgb, dets)

        except Exception as e:
            if backend == "Haar":
                self.win.show_error("Haar Inference Error", str(e))
            else:
                self.win.show_error("ONNX Inference Error", str(e))
            return

        # 표시 (out은 반드시 HxWx3 uint8 이미지)
        self.result_rgb = np.ascontiguousarray(out)
        self.win.right.set_rgb(self.result_rgb)

    # -- Backend / ONNX helpers --------------------------------------------
    def _on_backend_change(self, backend: str):
        if backend == "ONNX":
            if not HAS_ORT:
                self.win.show_error("onnxruntime missing", "pip install onnxruntime 또는 onnxruntime-gpu 로 설치하세요.")
                self.win.backend.setCurrentText("Haar")
                return
            self.win.haar_panel.hide()
            self.win.onnx_panel.show()
        else:
            self.win.onnx_panel.hide()
            self.win.haar_panel.show()
        self._run_now()

    def _load_onnx(self, path: str):
        self.onnx_model_path = path
        try:
            self._init_onnx_session_from_ui_provider()
            self.win.show_info("ONNX", f"모델 로드 완료:\n{path}")
            self._run_now()
        except Exception as e:
            self.onnx = None
            self.win.show_error("ONNX Load Error", str(e))

    def _init_onnx_session_from_ui_provider(self):
        import onnxruntime as ort
        provider = self.win.onnx_panel.provider.currentText()
        avail = ort.get_available_providers()
        req = [provider] if provider in avail else ["CPUExecutionProvider"]
        if provider not in avail:
            self.win.show_warn(
                "Provider fallback",
                f"{provider} 사용 불가. 사용 가능: {avail}\nCPUExecutionProvider로 폴백합니다."
            )
        self.onnx = ONNXFaceDetector(self.onnx_model_path, providers=req)
        # 세션 생성 직후에도 UI 값 동기화 한 번
        self._sync_onnx_params()
