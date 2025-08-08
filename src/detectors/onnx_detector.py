from __future__ import annotations
import numpy as np
import cv2
from typing import List, Tuple

try:
    import onnxruntime as ort
    HAS_ORT = True
except Exception:
    HAS_ORT = False


class ONNXFaceDetector:
    def __init__(
        self,
        onnx_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        use_invert: bool = False,
        clahe_clip: float = 2.0,
        clahe_grid: int = 8,
        providers: list[str] | None = None,
    ) -> None:
        if not onnx_path:
            raise ValueError("ONNX 모델 경로가 비어 있습니다.")
        if not HAS_ORT:
            raise RuntimeError("onnxruntime이 설치되지 않았습니다. pip install onnxruntime 또는 onnxruntime-gpu")

        self.onnx_path = onnx_path
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.use_invert = use_invert
        self.clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(clahe_grid, clahe_grid))

        ort_ver = getattr(ort, "__version__", "unknown")
        avail = set(ort.get_available_providers())
        req = list(providers or ["CPUExecutionProvider"])
        chosen = [p for p in req if p in avail] or (["CPUExecutionProvider"] if "CPUExecutionProvider" in avail else [])
        if not chosen:
            raise RuntimeError(f"사용 가능한 ExecutionProvider가 없습니다. available={list(avail)}")

        so = ort.SessionOptions()
        so.log_severity_level = 2
        try:
            self.sess = ort.InferenceSession(self.onnx_path, sess_options=so, providers=chosen)
        except TypeError as e:
            raise RuntimeError(
                f"InferenceSession 생성 실패: onnxruntime {ort_ver}. 최신 버전으로 업데이트하세요."
            ) from e
        except Exception as e:
            raise RuntimeError(
                "ONNX 세션 생성 실패\n"
                f"- ort version: {ort_ver}\n"
                f"- requested providers: {req}\n"
                f"- chosen providers: {chosen}\n"
                f"- available providers: {list(avail)}\n"
                f"- model: {self.onnx_path}\n"
                f"원인: {e}"
            ) from e

        try:
            self.inp_name = self.sess.get_inputs()[0].name
        except Exception as e:
            raise RuntimeError(f"모델 입력 정보를 읽는 중 오류: {e}")

    def set_params(self, input_w, input_h, conf, iou, clip, inv):
        self.input_size = (int(input_w), int(input_h))
        self.conf_thres = float(conf)
        self.iou_thres = float(iou)
        self.clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
        self.use_invert = bool(inv)

    def _clahe_proc(self, gray: np.ndarray) -> np.ndarray:
        return self.clahe.apply(gray)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float):
        if boxes.size == 0:
            return []
        idxs = scores.argsort()[::-1]
        keep = []
        while idxs.size > 0:
            i = idxs[0]
            keep.append(i)
            if idxs.size == 1:
                break
            rest = idxs[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / (area_i + area_r - inter + 1e-6)
            idxs = rest[iou < iou_thres]
        return keep

    def _preprocess(self, img_bgr_or_gray: np.ndarray):
        if img_bgr_or_gray.ndim == 2:
            inp = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2RGB)
        else:
            inp = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)
        in_w, in_h = self.input_size
        resized = cv2.resize(inp, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        return blob, (inp.shape[1], inp.shape[0])  # (W,H)

    def _parse_outputs(self, ort_out, scale_from, scale_to):
        preds = ort_out[0]
        if preds.ndim == 3:
            preds = preds[0]
        if preds.size == 0:
            return np.zeros((0, 4), dtype=np.int32), np.array([])
        conf = preds[:, 4]
        m = conf >= self.conf_thres
        preds = preds[m]
        if preds.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.int32), np.array([])
        in_w, in_h = scale_from
        W, H = scale_to
        sx, sy = W / in_w, H / in_h
        boxes = preds[:, :4].copy()
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        boxes = boxes.clip(min=0)
        scores = preds[:, 4]
        keep = self._nms(boxes, scores, self.iou_thres)
        return boxes[keep].astype(np.int32), scores[keep]

    def detect(self, rgb_img: np.ndarray):
        h, w = rgb_img.shape[:2]
        scale = 800.0 / max(h, w)
        if scale < 1.0:
            rgb = cv2.resize(rgb_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            rgb = rgb_img.copy()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        clahe_gray = self._clahe_proc(gray)
        proc_gray = cv2.bitwise_not(clahe_gray) if self.use_invert else clahe_gray
        display_img = cv2.cvtColor(proc_gray, cv2.COLOR_GRAY2RGB)
        blob, in_size = self._preprocess(proc_gray)
        out = self.sess.run(None, {self.inp_name: blob})
        boxes, scores = self._parse_outputs(out, self.input_size, in_size)
        faces = []
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            faces.append((x1, y1, x2 - x1, y2 - y1))
        return display_img, faces