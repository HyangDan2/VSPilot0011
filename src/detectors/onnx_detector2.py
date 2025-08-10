from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple, List, Optional
import time

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def _sigmoid_safe(x: np.ndarray) -> np.ndarray:
    # overflow-safe sigmoid
    # for x>=0: 1/(1+exp(-x)); for x<0: exp(x)/(1+exp(x))
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos], dtype=np.float32))
    ex = np.exp(x[~pos], dtype=np.float32)
    out[~pos] = ex / (1.0 + ex)
    return out.astype(np.float32)


def letterbox(im: np.ndarray, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    h0, w0 = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw *= 0.5
    dh *= 0.5
    if (w0, h0) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thres=0.45, max_det=300) -> List[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
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
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


# 표준 COCO80


COCO80 = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
    'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
    'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
    'oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]


# YOLOv5 anchor preset (stride 8/16/32)
_ANCHORS = np.array([
    [10,13, 16,30, 33,23],
    [30,61, 62,45, 59,119],
    [116,90, 156,198, 373,326]
], dtype=np.float32).reshape(3, 3, 2)


class ONNXFaceDetector:
    """
    ONNX 기반 얼굴/객체 검출 (YOLOv5/YOLOv8 onnx 호환)
    - 디버그 로그: raw → keep → 역변환 → NMS
    - 좌표 단위(정규화/픽셀) 자동 판별
    - obj/class sigmoid(overflow-safe)
    - letterbox 역변환(패딩 제거 + 1/scale) / 테스트용 스킵 토글
    - 스코어 중앙 표기(작은 박스는 코너 표기)
    """
    def __init__(self, model_path: str, providers: Optional[List[str]] = None,
                 center_score: bool = True, tiny_box_side_px: int = 22):
        if not HAS_ORT:
            raise ImportError("onnxruntime가 설치되지 않았습니다. pip install onnxruntime 또는 onnxruntime-gpu")

        self.model_path = model_path
        self.providers = providers or ["CPUExecutionProvider"]

        # 입력/후처리 파라미터(동적으로 갱신)
        self.input_width  = 640   # ✅ 고정 기본값
        self.input_height = 640   # ✅ 고정 기본값
        self.conf_threshold = 0.5
        self.iou_threshold  = 0.4
        self.clip_coords    = True
        self.invert_colors  = False  # ✅ 항상 미사용

        # 렌더 옵션
        self.center_score = center_score
        self.tiny_box_side_px = tiny_box_side_px

        # 디버그 옵션
        self.debug = True
        self.debug_topk = 6
        self.skip_depad_scale = False

        self.class_names = COCO80  # 필요시 외부에서 교체 가능
        self._init_session()

    # ---------------- Session & I/O ----------------
    def _init_session(self):
        self.sess = ort.InferenceSession(self.model_path, providers=self.providers)
        self.input_name = self.sess.get_inputs()[0].name
        shp = self.sess.get_inputs()[0].shape  # [N,H,W,C] or [N,C,H,W] or dynamic
        # 입력 레이아웃 및 크기 추론
        if len(shp) == 4:
            if shp[-1] == 3:  # NHWC
                self.expect_nhwc = True
                h = shp[1] if isinstance(shp[1], int) else self.input_height
                w = shp[2] if isinstance(shp[2], int) else self.input_width
            else:  # NCHW
                self.expect_nhwc = False
                h = shp[2] if isinstance(shp[2], int) else self.input_height
                w = shp[3] if isinstance(shp[3], int) else self.input_width
        else:
            self.expect_nhwc = True
            h = self.input_height
            w = self.input_width

        self.input_height = int(h)
        self.input_width = int(w)

    # ---------------- Preprocess ----------------
    def _preprocess(self, img_bgr: np.ndarray):
        lb, ratio, (dw, dh) = letterbox(img_bgr, (self.input_height, self.input_width))
        if self.invert_colors:
            img_rgb = 255 - cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        inp = img_rgb.astype(np.float32) / 255.0

        if self.expect_nhwc:
            inp = np.expand_dims(inp, 0)  # NHWC
        else:
            inp = np.transpose(inp, (2, 0, 1))
            inp = np.expand_dims(inp, 0)  # NCHW
        return inp, ratio, (dw, dh), img_bgr.shape[:2]

    # ---------------- Decode ----------------
    def _decode_detect(self, out: np.ndarray):
        """
        Detect 헤드 형태 처리: (1, 25200, 85) 고정 가정
        """
        # (1, 25200, 85) → (25200, 85)
        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]
        elif out.ndim == 2:
            pass
        else:
            raise RuntimeError(f"Unexpected Detect output shape: {out.shape}")

        if out.shape[1] != 85:
            raise RuntimeError(f"Expecting shape (*, 85) but got {out.shape}")

        # xywh, obj, cls 분리
        xywh = out[:, :4]
        obj  = _sigmoid_safe(out[:, 4:5])
        cls  = _sigmoid_safe(out[:, 5:])  # (25200, 80)

        # 최고 클래스 선택
        cls_idx = np.argmax(cls, axis=1)
        cls_score = cls[np.arange(cls.shape[0]), cls_idx]

        # conf = obj * cls_score
        conf = (obj.flatten() * cls_score).astype(np.float32)

        # conf 임계값 필터링
        keep = conf >= self.conf_threshold
        return xywh[keep], conf[keep], cls_idx[keep]


    def _decode_raw_heads(self, maps: List[np.ndarray]):
        """
        Raw 3-헤드(NHWC or NCHW) 처리. YOLOv5 방식.
        """
        ih, iw = self.input_height, self.input_width
        all_boxes, all_conf, all_cls = [], [], []
        # stride 레벨 순서가 뒤섞일 수 있어 순서는 신경쓰지 않음
        for i, fmap in enumerate(maps):
            if fmap.ndim != 4 or fmap.shape[0] != 1:
                raise RuntimeError(f"Unexpected raw head shape: {fmap.shape}")

            if fmap.shape[-1] % 3 == 0:  # NHWC
                gh, gw, ch = fmap.shape[1], fmap.shape[2], fmap.shape[3]
                per_anchor = ch // 3
                nc = per_anchor - 5
                arr = fmap.reshape(1, gh, gw, 3, per_anchor)[0]  # (gh,gw,na,85)
            elif fmap.shape[1] % 3 == 0:  # NCHW
                ch = fmap.shape[1]
                gh, gw = fmap.shape[2], fmap.shape[3]
                per_anchor = ch // 3
                nc = per_anchor - 5
                arr = fmap.transpose(0, 2, 3, 1).reshape(1, gh, gw, 3, per_anchor)[0]
            else:
                raise RuntimeError(f"Unknown raw head layout: {fmap.shape}")

            # stride 추정
            stride = ih / gh
            anchors = _ANCHORS[min(i, _ANCHORS.shape[0] - 1)]  # (3,2)

            t_xywh = _sigmoid_safe(arr[..., :4])
            t_obj = _sigmoid_safe(arr[..., 4:5])
            t_cls = _sigmoid_safe(arr[..., 5:5 + nc])

            gy, gx = np.meshgrid(np.arange(gh), np.arange(gw), indexing='ij')
            grid = np.stack([gx, gy], axis=-1)  # (gh,gw,2)

            xy = (t_xywh[..., :2] * 2.0 - 0.5 + grid[..., None, :]) * stride
            wh = (t_xywh[..., 2:4] * 2.0) ** 2 * anchors[None, None, :, :]

            x1y1 = xy - wh * 0.5
            x2y2 = xy + wh * 0.5
            boxes = np.concatenate([x1y1, x2y2], axis=-1).reshape(-1, 4)

            cls_idx = np.argmax(t_cls, axis=-1)
            cls_score = np.take_along_axis(t_cls, cls_idx[..., None], axis=-1).squeeze(-1)
            conf = (t_obj.squeeze(-1) * cls_score).reshape(-1).astype(np.float32)
            cls_idx = cls_idx.reshape(-1).astype(np.int32)

            keep = conf >= self.conf_threshold
            if np.any(keep):
                all_boxes.append(boxes[keep])
                all_conf.append(conf[keep])
                all_cls.append(cls_idx[keep])

        if not all_boxes:
            return (np.zeros((0, 4), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.int32))

        return (np.concatenate(all_boxes, axis=0),
                np.concatenate(all_conf, axis=0),
                np.concatenate(all_cls, axis=0))

    def _maybe_detect_normalized(self, boxes: np.ndarray) -> bool:
        """
        출력이 [0,1] 범위가 대부분이면 '정규화 좌표'로 판단.
        """
        if boxes.size == 0:
            return False
        # 랜덤 샘플로 히ュー리스틱
        sample = boxes[np.random.choice(boxes.shape[0], min(32, boxes.shape[0]), replace=False)]
        ratio = np.mean((sample >= -0.01) & (sample <= 1.01))
        return ratio > 0.85

    # ---------------- Postprocess ----------------
    def _postprocess(self, outputs, ratio, dwdh, orig_hw):
        """
        outputs: Detect (N,85) or raw 3-head list
        결과: [(xyxy(np.float32[4]), score(float), cls(int)), ...]
        """
        if isinstance(outputs, (list, tuple)):
            boxes, conf, cls_idx = self._decode_raw_heads(list(outputs))
            # raw head는 이미 input scale(boxes)임
        else:
            xywh, conf, cls_idx = self._decode_detect(outputs)
            if xywh.shape[0] == 0:
                return []

            # 좌표 단위 판별
            normalized = self._maybe_detect_normalized(xywh)
            x, y, w, h = xywh.T
            x1 = x - w / 2.0
            y1 = y - h / 2.0
            x2 = x + w / 2.0
            y2 = y + h / 2.0
            boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

            if normalized:
                boxes[:, [0, 2]] *= self.input_width
                boxes[:, [1, 3]] *= self.input_height

        # 역-Letterbox
        if not self.skip_depad_scale and boxes.shape[0] > 0:
            dh, dw = dwdh[1], dwdh[0]
            boxes[:, [0, 2]] -= dw
            boxes[:, [1, 3]] -= dh
            boxes /= max(ratio, 1e-9)

        # 클립
        if self.clip_coords and boxes.shape[0] > 0:
            oh, ow = orig_hw
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ow - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, oh - 1)

        # ✅ 최소 박스 크기 필터
        if boxes.shape[0] > 0:
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            min_w, min_h = 24, 24  # px
            keep_sz = (w >= min_w) & (h >= min_h)
            boxes, conf, cls_idx = boxes[keep_sz], conf[keep_sz], cls_idx[keep_sz]

        # NMS
        keep = nms_numpy(boxes, conf, iou_thres=self.iou_threshold, max_det=300) if boxes.shape[0] else []
        dets = [(boxes[i], float(conf[i]), int(cls_idx[i])) for i in keep]

        # 디버그 로그
        if self.debug:
            k = min(self.debug_topk, len(dets))
            if k > 0:
                print(f"[DEBUG] keep top-{k} (after NMS):")
                for i in range(k):
                    b, sc, cl = dets[i]
                    print(f"  {i}: box={b.round(1)}, score={sc:.3f}, cls={cl}")

        return dets


    # ---------------- Public API ----------------
    def detect(self, img_bgr):
        inp, ratio, dwdh, orig_hw = self._preprocess(img_bgr)
        t0 = time.time()
        outs = self.sess.run(None, {self.input_name: inp})
        ms = (time.time() - t0) * 1000.0

        outputs = outs if len(outs) > 1 else outs[0]
        dets = self._postprocess(outputs, ratio, dwdh, orig_hw)
        return dets, ms

    def draw(self, img_bgr: np.ndarray, dets: List[Tuple[np.ndarray, float, int]]) -> np.ndarray:
        vis = img_bgr.copy()
        for box, score, cls in dets:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            w = x2 - x1
            h = y2 - y1
            name = self.class_names[cls] if 0 <= cls < len(self.class_names) else str(cls)
            label = f"{name} {score:.2f}"

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if self.center_score and min(w, h) > self.tiny_box_side_px:
                cx, cy = x1 + w // 2, y1 + h // 2
                cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)
                # ✅ 중앙에 클래스+점수
                cv2.putText(vis, label, (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # 작은 박스는 기존처럼 코너에
                cv2.putText(vis, label, (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return vis
