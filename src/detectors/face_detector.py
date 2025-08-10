# detectors/face_detector.py
from __future__ import annotations
import time
from typing import List, Tuple, Optional
import numpy as np
import cv2

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

def _sigmoid_safe(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos], dtype=np.float32))
    ex = np.exp(x[~pos], dtype=np.float32)
    out[~pos] = ex / (1.0 + ex)
    return out.astype(np.float32)

def letterbox(im: np.ndarray, new_shape=(640, 640), color=(114,114,114), scaleup=True):
    h0, w0 = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0]/h0, new_shape[1]/w0)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(w0*r)), int(round(h0*r)))
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw *= 0.5; dh *= 0.5
    if (w0, h0) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thres=0.45, max_det=300) -> List[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


class ONNXFaceDetector:
    """
    얼굴 전용 onnx 디텍터 (출력 6채널: [cx, cy, w, h, obj, face])
    - 지원 출력: (1,N,6) / (N,6) / (1,6,N) / (6,N) / (6*k,)
    - conf = sigmoid(obj) * sigmoid(face)
    - 좌표 (정규화/픽셀) 자동 판별
    """
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        self.model_path = model_path
        self.providers = providers or ["CPUExecutionProvider"]

        self.input_width = 640
        self.input_height = 640
        self.conf_threshold = 0.4
        self.iou_threshold = 0.45
        self.clip_coords = True
        self.invert_colors = False
        self.center_score = True
        self.tiny_box_side_px = 22
        self.debug = False
        self.debug_topk = 8

        self._init_session()

    def _init_session(self):
        self.sess = ort.InferenceSession(self.model_path, providers=self.providers)
        self.input_name = self.sess.get_inputs()[0].name
        shp = self.sess.get_inputs()[0].shape  # [N,H,W,C] or [N,C,H,W] or dynamic
        if len(shp) == 4:
            if shp[-1] == 3:  # NHWC
                self.expect_nhwc = True
                h = shp[1] if isinstance(shp[1], int) else self.input_height
                w = shp[2] if isinstance(shp[2], int) else self.input_width
            else:             # NCHW
                self.expect_nhwc = False
                h = shp[2] if isinstance(shp[2], int) else self.input_height
                w = shp[3] if isinstance(shp[3], int) else self.input_width
        else:
            self.expect_nhwc = True
            h, w = self.input_height, self.input_width
        self.input_height, self.input_width = int(h), int(w)

    def _preprocess(self, img_bgr: np.ndarray):
        lb, ratio, (dw, dh) = letterbox(img_bgr, (self.input_height, self.input_width))
        if self.invert_colors:
            img_rgb = 255 - cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        inp = img_rgb.astype(np.float32) / 255.0
        if self.expect_nhwc:
            inp = np.expand_dims(inp, 0)
        else:
            inp = np.transpose(inp, (2,0,1))
            inp = np.expand_dims(inp, 0)
        return inp, ratio, (dw, dh), img_bgr.shape[:2]

    def _normalize_layout(self, out: np.ndarray) -> np.ndarray:
        if out.ndim == 3:
            if out.shape[0] == 1 and out.shape[2] == 6:
                out = out.reshape(-1, 6)
            elif out.shape[0] == 1 and out.shape[1] == 6:
                out = out.transpose(0,2,1).reshape(-1, 6)
            else:
                raise RuntimeError(f"Unexpected 3D output shape {out.shape}")
        elif out.ndim == 2:
            if out.shape[1] == 6:
                pass
            elif out.shape[0] == 6:
                out = out.transpose(1,0)
            else:
                raise RuntimeError(f"Unexpected 2D output shape {out.shape}")
        elif out.ndim == 1:
            if out.shape[0] % 6 != 0:
                raise RuntimeError(f"Unexpected 1D output length {out.shape[0]}")
            out = out.reshape(-1, 6)
        else:
            raise RuntimeError(f"Unexpected output ndim={out.ndim}")
        return out.astype(np.float32, copy=False)

    def _maybe_normalized_xywh(self, xywh: np.ndarray) -> bool:
        if xywh.size == 0: return False
        sample = xywh[np.random.choice(xywh.shape[0], min(32, xywh.shape[0]), replace=False)]
        ratio = np.mean((sample >= -0.01) & (sample <= 1.01))
        return ratio > 0.80

    def _decode_face6(self, out_any) -> Tuple[np.ndarray, np.ndarray]:
        out = self._normalize_layout(out_any)  # (N,6)
        if out.shape[1] != 6:
            raise RuntimeError(f"Expect 6 channels but got {out.shape[1]}")
        xywh = out[:, :4]
        obj = _sigmoid_safe(out[:, 4:5]).flatten()
        face = _sigmoid_safe(out[:, 5:6]).flatten()
        conf = (obj * face).astype(np.float32)

        keep = conf >= self.conf_threshold
        if not np.any(keep):
            return np.zeros((0,4), np.float32), np.zeros((0,), np.float32)

        xywh = xywh[keep]
        conf = conf[keep]

        x, y, w, h = xywh.T
        x1 = x - w*0.5; y1 = y - h*0.5
        x2 = x + w*0.5; y2 = y + h*0.5
        boxes = np.stack([x1,y1,x2,y2], axis=1).astype(np.float32)

        if self._maybe_normalized_xywh(xywh):
            boxes[:, [0,2]] *= self.input_width
            boxes[:, [1,3]] *= self.input_height

        return boxes, conf

    def _postprocess(self, outputs, ratio, dwdh, orig_hw):
        out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        boxes, conf = self._decode_face6(out)

        if boxes.shape[0] > 0:
            dh, dw = dwdh[1], dwdh[0]
            boxes[:, [0,2]] -= dw
            boxes[:, [1,3]] -= dh
            boxes /= max(ratio, 1e-9)

        if boxes.shape[0] > 0 and self.clip_coords:
            oh, ow = orig_hw
            boxes[:, [0,2]] = np.clip(boxes[:, [0,2]], 0, ow-1)
            boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, oh-1)

        keep = nms_numpy(boxes, conf, iou_thres=self.iou_threshold, max_det=300) if boxes.shape[0] else []
        dets = [(boxes[i], float(conf[i])) for i in keep]

        if self.debug and dets:
            k = min(self.debug_topk, len(dets))
            print(f"[DEBUG] keep top-{k}:")
            for i in range(k):
                b, sc = dets[i]
                print(f"  {i}: box={np.round(b,1)}, score={sc:.3f}")
        return dets

    def detect(self, img_bgr: np.ndarray):
        inp, ratio, dwdh, orig_hw = self._preprocess(img_bgr)
        t0 = time.time()
        outs = self.sess.run(None, {self.input_name: inp})
        ms = (time.time() - t0) * 1000.0
        dets = self._postprocess(outs if len(outs)>1 else outs[0], ratio, dwdh, orig_hw)
        return dets, ms

    def draw(self, img_bgr: np.ndarray, dets: List[Tuple[np.ndarray, float]]):
        vis = img_bgr.copy()
        for box, score in dets:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            w = x2 - x1; h = y2 - y1
            if self.center_score and min(w, h) > self.tiny_box_side_px:
                cx, cy = x1 + w//2, y1 + h//2
                cv2.circle(vis, (cx, cy), 3, (0,255,255), -1)
                cv2.putText(vis, f"{score:.2f}", (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            else:
                cv2.putText(vis, f"face {score:.2f}", (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        return vis
