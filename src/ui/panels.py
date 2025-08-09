from __future__ import annotations
from PySide6.QtWidgets import QWidget, QFormLayout, QCheckBox, QSlider, QLabel, QHBoxLayout, QComboBox
from PySide6.QtCore import Qt, Signal


# ---------- Helper slider widgets ----------
class LFloatSlider(QWidget):
    """Float slider using integer QSlider with scale factor. Shows live value label."""
    changed = Signal()

    def __init__(self, minimum: float, maximum: float, step: float, value: float, fmt: str = "{:.2f}"):
        super().__init__()
        assert step > 0
        self._factor = int(round(1.0 / step))
        self._min = minimum
        self._max = maximum
        self._fmt = fmt

        self._sld = QSlider(Qt.Orientation.Horizontal)
        self._sld.setMinimum(int(round(minimum * self._factor)))
        self._sld.setMaximum(int(round(maximum * self._factor)))
        self._sld.setSingleStep(1)
        self._sld.setPageStep(1)
        self._sld.setValue(int(round(value * self._factor)))

        self._lbl = QLabel(self._fmt.format(self.value()))
        self._lbl.setMinimumWidth(64)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._sld, 1)
        lay.addWidget(self._lbl, 0)

        self._sld.valueChanged.connect(self._on_changed)

    def _on_changed(self, _):
        self._lbl.setText(self._fmt.format(self.value()))
        self.changed.emit()

    def value(self) -> float:
        return float(self._sld.value()) / self._factor

    # optional: programmatic set
    def setValue(self, v: float):
        self._sld.setValue(int(round(v * self._factor)))


class LIntSlider(QWidget):
    """Integer slider with live value label."""
    changed = Signal()

    def __init__(self, minimum: int, maximum: int, step: int, value: int):
        super().__init__()
        self._sld = QSlider(Qt.Orientation.Horizontal)
        self._sld.setMinimum(minimum)
        self._sld.setMaximum(maximum)
        self._sld.setSingleStep(step)
        self._sld.setPageStep(step)
        self._sld.setValue(value)

        self._lbl = QLabel(str(self.value()))
        self._lbl.setMinimumWidth(48)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._sld, 1)
        lay.addWidget(self._lbl, 0)

        self._sld.valueChanged.connect(self._on_changed)

    def _on_changed(self, _):
        self._lbl.setText(str(self.value()))
        self.changed.emit()

    def value(self) -> int:
        return int(self._sld.value())

    def setValue(self, v: int):
        self._sld.setValue(int(v))


class LDiscreteStepSlider(QWidget):
    """
    Discrete slider mapped as: value = base + index * step
    Useful for (320~1280 step 32) like widths/heights.
    """
    changed = Signal()

    def __init__(self, base: int, step: int, count: int, value: int):
        super().__init__()
        self._base = base
        self._step = step
        self._count = count  # inclusive max index
        self._sld = QSlider(Qt.Orientation.Horizontal)
        self._sld.setMinimum(0)
        self._sld.setMaximum(count)
        self._sld.setSingleStep(1)
        self._sld.setPageStep(1)

        idx = max(0, min(count, (value - base) // step))
        self._sld.setValue(idx)

        self._lbl = QLabel(str(self.value()))
        self._lbl.setMinimumWidth(64)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._sld, 1)
        lay.addWidget(self._lbl, 0)

        self._sld.valueChanged.connect(self._on_changed)

    def _on_changed(self, _):
        self._lbl.setText(str(self.value()))
        self.changed.emit()

    def value(self) -> int:
        return int(self._base + self._sld.value() * self._step)

    def setValue(self, v: int):
        idx = (v - self._base) // self._step
        self._sld.setValue(idx)


# ---------- Panels ----------
class HaarPanel(QWidget):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        lay = QFormLayout(self)

        # scaleFactor (1.05~1.5 step 0.01)
        self.sf = LFloatSlider(1.05, 1.50, 0.01, 1.15, fmt="{:.2f}")
        # minNeighbors (1~12 step 1)
        self.mn = LIntSlider(1, 12, 1, 5)
        # minSize px (20~200 step 5)
        self.ms = LIntSlider(20, 200, 5, 40)
        # CLAHE clipLimit (1.0~5.0 step 0.1)
        self.clip = LFloatSlider(1.0, 5.0, 0.1, 2.0, fmt="{:.1f}")
        # Invert
        self.inv = QCheckBox("Invert(반전) 사용")

        for w, label in [
            (self.sf, "scaleFactor"),
            (self.mn, "minNeighbors"),
            (self.ms, "minSize(px)"),
            (self.clip, "CLAHE clipLimit"),
            (self.inv, "Invert"),
        ]:
            lay.addRow(label, w)
            if hasattr(w, "changed"):
                w.changed.connect(self.changed)

        self.inv.stateChanged.connect(self.changed)

    def params(self):
        return (
            self.sf.value(),
            self.mn.value(),
            self.ms.value(),
            self.clip.value(),
            self.inv.isChecked(),
        )


class OnnxPanel(QWidget):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        lay = QFormLayout(self)

        # Input Width/Height: 320~1280 step 32 -> index 0..30
        self.in_w = LDiscreteStepSlider(base=320, step=32, count=(1280 - 320) // 32, value=640)
        self.in_h = LDiscreteStepSlider(base=320, step=32, count=(1280 - 320) // 32, value=640)

        # Conf/Iou: float sliders
        self.conf = LFloatSlider(0.05, 0.90, 0.01, 0.25, fmt="{:.2f}")
        self.iou = LFloatSlider(0.10, 0.90, 0.01, 0.45, fmt="{:.2f}")
        self.clip = LFloatSlider(1.0, 5.0, 0.1, 2.0, fmt="{:.1f}")
        self.inv = QCheckBox("Invert(반전) 사용")

        self.provider = QComboBox()
        self.provider.addItems(["CPUExecutionProvider", "CUDAExecutionProvider"])

        for w, label in [
            (self.in_w, "Input Width"),
            (self.in_h, "Input Height"),
            (self.conf, "Conf Thres"),
            (self.iou, "IoU Thres"),
            (self.clip, "CLAHE clipLimit"),
            (self.inv, "Invert"),
            (self.provider, "Provider"),
        ]:
            lay.addRow(label, w)
            if hasattr(w, "changed"):
                w.changed.connect(self.changed)
            if hasattr(w, "currentIndexChanged"):
                w.currentIndexChanged.connect(self.changed)

        self.inv.stateChanged.connect(self.changed)

    def params(self):
        return (
            self.in_w.value(),
            self.in_h.value(),
            self.conf.value(),
            self.iou.value(),
            self.clip.value(),
            self.inv.isChecked(),
            self.provider.currentText(),
        )
