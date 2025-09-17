# 🖥 Face Detection GUI (PySide6)

A face detection GUI application built with PySide6 in an OOP structure.  
Supports Haar/LBP (OpenCV) and ONNX backends, with CLAHE/Invert processing, slider-based parameter control, and real-time re-execution.

---

## ✨ Features
- **Backend Selection**: Haar/LBP (OpenCV) & ONNX
- **Image Processing Options**: CLAHE contrast enhancement, Invert
- **Real-time Parameter Control**: Automatic re-execution when sliders are adjusted
- **I/O Functions**: Load and save images
- **OOP Architecture**: Separated into UI / Controller / Detector / Utils layers
- **PySide6 UI**: Built on QMainWindow + FormLayout

---

## 📂 Project Structure
```bash
.
face-detector-gui/
├─ README.md
├─ LICENSE.txt
├─ requirements.txt
├─ src/
│  ├─ app.py                # Entry point
│  ├─ controllers/
│  │   └─ detect_controller.py
│  ├─ detectors/
│  │   ├─ haar_detector.py
│  │   └─ onnx_detector.py
│  ├─ ui/
│  │   ├─ main_window.py
│  │   ├─ panels.py
│  │   └─ image_view.py
│  └─ utils/
│      └─ image_utils.py
└─ assets/
   └─ qss/app.qss (optional)
```

## 🚀 How to Run
```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python -m src.app
```

## 📦 Build with PyInstaller
```bash
pip install pyinstaller
pyinstaller -F -w -n FaceDetector src/app.py
```

## License
MIT License - See [License](./License)

