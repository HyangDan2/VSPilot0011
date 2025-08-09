from __future__ import annotations
import cv2
import numpy as np
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import QTimer
from ui.main_window import MainWindow
from detectors.haar_detector import HaarFaceDetector
from detectors.onnx_detector import ONNXFaceDetector, HAS_ORT


class DetectController:
    """
    - 파라미터 변경: 50ms 디바운스 후 자동 실행 (Tk의 Scale(command=...) 같은 UX)
    - Run 버튼: 디바운스 무시하고 즉시 실행
    - 백엔드 전환(Haar <-> ONNX): 바로 한 번 실행
    - 표시 안전성: np.ascontiguousarray로 버퍼 연속성 보장
    """
    def __init__(self, win: MainWindow) -> None:
        self.win = win
        self.original_rgb: np.ndarray | None = None
        self.result_rgb: np.ndarray | None = None

        self.haar = HaarFaceDetector()
        self.onnx: ONNXFaceDetector | None = None
        self.onnx_model_path: str | None = None

        # --- 디바운스 타이머 (파라미터 변경 폭주 방지) ---
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._run_now)

        # --- 시그널 연결 ---
        # UI 액션
        win.loadRequested.connect(self.load_image)
        win.saveRequested.connect(self.save_image)

        # Run 버튼: 즉시 실행 (디바운스 우회)
        win.runRequested.connect(self._run_now)

        # 백엔드 전환 시 즉시 한 번 실행 + 패널 토글
        win.backendChanged.connect(self._on_backend_change)

        # 파라미터 변경 시 자동 실행 (패널 내부에서 changed 시그널 발행)
        win.haar_panel.changed.connect(self.run)   # 디바운스
        win.onnx_panel.changed.connect(self.run)   # 디바운스

        # ONNX 모델 선택
        win.onnxModelChosen.connect(self._load_onnx)

    # --------------------------------------------------------------------- #
    # Public actions
    # --------------------------------------------------------------------- #
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



    def save_image(self):
        if self.result_rgb is None:
            return
        path, _ = QFileDialog.getSaveFileName(self.win, "Save result", filter="PNG (*.png)")
        if not path:
            return
        cv2.imwrite(path, cv2.cvtColor(self.result_rgb, cv2.COLOR_RGB2BGR))

    # --------------------------------------------------------------------- #
    # Auto-run (debounced)
    # --------------------------------------------------------------------- #
    def run(self):
        """파라미터 변경 등 잦은 이벤트 -> 50ms 내 모아서 한 번만 실행"""
        # 필요에 따라 30~100ms로 조절 가능
        self._debounce.start(50)

    def _run_now(self):
        """즉시 실행(디바운스 우회). Run 버튼/이미지 로드/백엔드 전환 등에서 사용."""
        if self.original_rgb is None:
            return

        backend = self.win.backend.currentText()
        try:
            if backend == "Haar":
                # 파라미터 읽고 적용
                sf, mn, ms, clip, inv = self.win.haar_panel.params()
                self.haar.set_params(sf, mn, ms, clip, inv)

                out, _faces = self.haar.detect(self.original_rgb)

            else:  # ONNX
                if not HAS_ORT:
                    self.win.show_error("onnxruntime missing", "pip install onnxruntime 또는 onnxruntime-gpu 로 설치하세요.")
                    self.win.backend.setCurrentText("Haar")
                    return

                if self.onnx is None:
                    # 모델이 아직 안 올라간 상태면 안내
                    if not self.onnx_model_path:
                        self.win.show_warn("ONNX", "먼저 ONNX 모델을 로드하세요.")
                        return
                    # 모델 경로는 있는데 세션 미초기화였던 경우 초기화
                    self._init_onnx_session_from_ui_provider()

                in_w, in_h, conf, iou, clip, inv, _prov = self.win.onnx_panel.params()
                self.onnx.set_params(in_w, in_h, conf, iou, clip, inv)

                out, _faces = self.onnx.detect(self.original_rgb)

        except Exception as e:
            # 추론 에러는 사용자에게 알려주고 중단
            if backend == "Haar":
                self.win.show_error("Haar Inference Error", str(e))
            else:
                self.win.show_error("ONNX Inference Error", str(e))
            return

        # 표시 안전성 확보
        self.result_rgb = np.ascontiguousarray(out)
        self.win.right.set_rgb(self.result_rgb)

    # --------------------------------------------------------------------- #
    # Backend / ONNX helpers
    # --------------------------------------------------------------------- #
    def _on_backend_change(self, backend: str):
        """백엔드 전환 시 패널 토글 + 즉시 한 번 실행"""
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

        # 전환 직후 현재 상태로 바로 반영
        self._run_now()

    def _load_onnx(self, path: str):
        """ONNX 모델 파일을 고르고 세션 생성"""
        self.onnx_model_path = path
        try:
            self._init_onnx_session_from_ui_provider()
            self.win.show_info("ONNX", f"모델 로드 완료:\n{path}")
            # 모델 로드 직후 현재 파라미터로 실행
            self._run_now()
        except Exception as e:
            self.onnx = None
            self.win.show_error("ONNX Load Error", str(e))

    def _init_onnx_session_from_ui_provider(self):
        """UI에서 선택한 Provider로 ONNX 세션 초기화 (가용성 체크 포함)"""
        import onnxruntime as ort
        provider = self.win.onnx_panel.provider.currentText()
        avail = ort.get_available_providers()
        req = [provider] if provider in avail else ["CPUExecutionProvider"]
        if provider not in avail:
            self.win.show_warn("Provider fallback",
                               f"{provider} 사용 불가. 사용 가능: {avail}\nCPUExecutionProvider로 폴백합니다.")
        self.onnx = ONNXFaceDetector(self.onnx_model_path, providers=req)
