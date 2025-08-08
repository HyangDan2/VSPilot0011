from __future__ import annotations
from PySide6.QtWidgets import QMainWindow, QWidget, QFileDialog, QMessageBox, QComboBox, QToolBar, QPushButton, QHBoxLayout, QVBoxLayout
from PySide6.QtCore import Qt, Signal
from .image_view import ImageView
from .panels import HaarPanel, OnnxPanel


class MainWindow(QMainWindow):
    loadRequested = Signal()
    runRequested = Signal()
    saveRequested = Signal()
    backendChanged = Signal(str)
    onnxModelChosen = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Face Detection (Haar / ONNX) – PySide6")
        self.resize(1100, 700)

        # Central
        cw = QWidget(); self.setCentralWidget(cw)
        main = QVBoxLayout(cw)
        # top controls
        top = QHBoxLayout()
        self.backend = QComboBox(); self.backend.addItems(["Haar", "ONNX"])
        btn_load_model = QPushButton("Load ONNX Model")
        btn_load = QPushButton("Load Image")
        btn_run = QPushButton("Run")
        btn_save = QPushButton("Save Result")
        for w in [self.backend, btn_load_model, btn_load, btn_run, btn_save]:
            top.addWidget(w)
        main.addLayout(top)

        # image views
        imgs = QHBoxLayout()
        self.left = ImageView("원본")
        self.right = ImageView("결과")
        imgs.addWidget(self.left, 1)
        imgs.addWidget(self.right, 1)
        main.addLayout(imgs, 1)

        # panels
        self.haar_panel = HaarPanel(); self.onnx_panel = OnnxPanel();
        main.addWidget(self.haar_panel)
        self.onnx_panel.hide()

        # signals
        self.backend.currentTextChanged.connect(self.backendChanged)
        btn_load_model.clicked.connect(self._choose_onnx)
        btn_load.clicked.connect(self.loadRequested)
        btn_run.clicked.connect(self.runRequested)
        btn_save.clicked.connect(self.saveRequested)

    def _choose_onnx(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose ONNX model", filter="ONNX model (*.onnx)")
        if path:
            self.onnxModelChosen.emit(path)

    # helpers
    def show_info(self, title: str, text: str):
        QMessageBox.information(self, title, text)
    def show_warn(self, title: str, text: str):
        QMessageBox.warning(self, title, text)
    def show_error(self, title: str, text: str):
        QMessageBox.critical(self, title, text)