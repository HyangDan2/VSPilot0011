# 🖥 Face Detection GUI (PySide6)

PySide6 기반 OOP 구조로 작성한 얼굴 검출 GUI 앱입니다.  
Haar/LBP(OpenCV)와 ONNX 백엔드를 지원하며, CLAHE/Invert, 슬라이더 기반 파라미터 조절, 실시간 재실행 기능이 있습니다.

---

## ✨ Features
- **백엔드 선택**: Haar/LBP (OpenCV) & ONNX
- **이미지 처리 옵션**: CLAHE 대비 향상, 반전(Invert)
- **실시간 파라미터 조절**: 슬라이더 변경 시 자동 재실행
- **입출력 기능**: 이미지 로드 / 저장
- **OOP 구조**: UI / Controller / Detector / Utils 계층 분리
- **PySide6 UI**: QMainWindow + FormLayout 기반

---

## 📂 프로젝트 구조
face-detector-gui/
├─ README.md
├─ LICENSE.txt
├─ requirements.txt
├─ src/
│ ├─ app.py # 실행 진입점
│ ├─ controllers/
│ │ └─ detect_controller.py
│ ├─ detectors/
│ │ ├─ haar_detector.py
│ │ └─ onnx_detector.py
│ ├─ ui/
│ │ ├─ main_window.py
│ │ ├─ panels.py
│ │ └─ image_view.py
│ └─ utils/
│ └─ image_utils.py
└─ assets/
└─ qss/app.qss (선택)
---

## 🚀 실행 방법
```bash
# 1. 가상환경 생성 및 활성화 (권장)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 실행
python -m src.app

📦 빌드 (PyInstaller)
```
pip install pyinstaller
pyinstaller -F -w -n FaceDetector src/app.py
```