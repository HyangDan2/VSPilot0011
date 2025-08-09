from __future__ import annotations
from PySide6.QtWidgets import QWidget, QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox
from PySide6.QtCore import Signal


class HaarPanel(QWidget):
    changed = Signal()
    def __init__(self) -> None:
        super().__init__()
        lay = QFormLayout(self)
        self.sf = QDoubleSpinBox(minimum=1.05, maximum=1.5, singleStep=0.01, value=1.15)
        self.mn = QSpinBox(minimum=1, maximum=12, value=5)
        self.ms = QSpinBox(minimum=20, maximum=200, singleStep=5, value=40)
        self.clip = QDoubleSpinBox(minimum=1.0, maximum=5.0, singleStep=0.1, value=2.0)
        self.inv = QCheckBox("Invert(반전) 사용")
        for w, label in [
            (self.sf, "scaleFactor"), (self.mn, "minNeighbors"), (self.ms, "minSize(px)"),
            (self.clip, "CLAHE clipLimit"), (self.inv, "Invert")
        ]:
            lay.addRow(label, w)
            # valueChanged: 슬라이더/스핀박스 값 변경 즉시
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self.changed)
            # editingFinished: 키보드 입력 후 엔터/포커스아웃
            if hasattr(w, "editingFinished"):
                w.editingFinished.connect(self.changed)

        self.inv.stateChanged.connect(self.changed)


    def params(self):
        return self.sf.value(), self.mn.value(), self.ms.value(), self.clip.value(), self.inv.isChecked()


class OnnxPanel(QWidget):
    changed = Signal()
    def __init__(self) -> None:
        super().__init__()
        lay = QFormLayout(self)
        from PySide6.QtWidgets import QComboBox
        self.in_w = QSpinBox(minimum=320, maximum=1280, singleStep=32, value=640)
        self.in_h = QSpinBox(minimum=320, maximum=1280, singleStep=32, value=640)
        self.conf = QDoubleSpinBox(minimum=0.05, maximum=0.9, singleStep=0.01, value=0.25)
        self.iou = QDoubleSpinBox(minimum=0.10, maximum=0.9, singleStep=0.01, value=0.45)
        self.clip = QDoubleSpinBox(minimum=1.0, maximum=5.0, singleStep=0.1, value=2.0)
        self.inv = QCheckBox("Invert(반전) 사용")
        self.provider = QComboBox(); self.provider.addItems(["CPUExecutionProvider", "CUDAExecutionProvider"])
        # OnnxPanel.__init__ 내
        for w, label in [
            (self.in_w, "Input Width"), (self.in_h, "Input Height"),
            (self.conf, "Conf Thres"), (self.iou, "IoU Thres"),
            (self.clip, "CLAHE clipLimit"), (self.inv, "Invert"),
            (self.provider, "Provider")
        ]:
            lay.addRow(label, w)
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self.changed)
            if hasattr(w, "editingFinished"):
                w.editingFinished.connect(self.changed)
            if hasattr(w, "currentIndexChanged"):
                w.currentIndexChanged.connect(self.changed)

        self.inv.stateChanged.connect(self.changed)

        # 입력 중에도 즉시 반응 원하면(스핀박스)
        for sb in (self.in_w, self.in_h, self.conf, self.iou, self.clip):
            if hasattr(sb, "setKeyboardTracking"):
                sb.setKeyboardTracking(True)  # 타이핑 중에도 valueChanged 발생


    def params(self):
        return (
            self.in_w.value(), self.in_h.value(), self.conf.value(), self.iou.value(), self.clip.value(), self.inv.isChecked(),
            self.provider.currentText()
        )