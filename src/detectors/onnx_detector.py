from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple, List

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


class ONNXFaceDetector:
    """
    ONNX 기반 얼굴 검출 (YOLOv5/YOLOv8 onnx 호환)
    - 디버그 로그: raw → keep → 역변환 → NMS
    - 좌표 단위(정규화/픽셀) 자동 판별
    - obj/class sigmoid(overflow-safe)
    - letterbox 역변환(패딩 제거 + 1/scale) / 테스트용 스킵 토글
    - 스코어 중앙 표기(작은 박스는 코너 표기)
    """
    def __init__(self, model_path: str, providers: List[str] = None,
                 center_score: bool = True, tiny_box_side_px: int = 22):
        if not HAS_ORT:
            raise ImportError("onnxruntime가 설치되지 않았습니다. pip install onnxruntime 또는 onnxruntime-gpu")

        self.model_path = model_path
        self.providers = providers or ["CPUExecutionProvider"]

        # 입력/후처리 파라미터
        self.input_width = 640
        self.input_height = 640
        self.conf_threshold = 0.5
        self.iou_threshold = 0.4
        self.clip_coords = True
        self.invert_colors = False

        # 렌더 옵션
        self.center_score = center_score
        self.tiny_box_side_px = tiny_box_side_px

        # ✅ 디버그 옵션
        self.debug = True         # 콘솔 로그 ON/OFF
        self.debug_topk = 6       # 단계별 미리보기 개수
        self.skip_depad_scale = False  # True면 pad/scale 역변환을 건너뜀(원인 진단용)

        self._init_session()

    # ---------------------- 유틸/디버그 ----------------------
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # overflow-safe sigmoid
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _dbg_head(self, arr: np.ndarray, k: int | None = None) -> np.ndarray:
        if k is None:
            k = self.debug_topk
        k = min(k, len(arr)) if len(arr) else 0
        return arr[:k]

    def _dbg_stats(self, name: str, arr: np.ndarray):
        if arr.size == 0:
            print(f"[DBG] {name}: empty")
            return
        print(f"[DBG] {name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}")

    def _analyze_output(self, pred: np.ndarray) -> tuple[bool, bool]:
        """좌표 정규화 여부, conf가 logit인지 여부 추정"""
        coord_min, coord_max = float(pred[:, :4].min()), float(pred[:, :4].max())
        conf_min, conf_max = float(pred[:, 4].min()), float(pred[:, 4].max())
        coords_are_normalized = coord_max <= 1.5
        conf_is_logit = (conf_max > 1.0) or (conf_min < 0.0)
        if self.debug:
            print(f"[DBG] raw coord range: {coord_min:.4f} ~ {coord_max:.4f} -> "
                  f"{'NORMALIZED(0~1)' if coords_are_normalized else 'PIXEL'}")
            print(f"[DBG] raw conf  range: {conf_min:.4f} ~ {conf_max:.4f} -> "
                  f"{'LOGIT(sigmoid 필요)' if conf_is_logit else 'PROB(그대로 사용)'}")
        return coords_are_normalized, conf_is_logit

    # ---------------------- ONNX 세션 ----------------------
    def _init_session(self):
        try:
            self.session = ort.InferenceSession(self.model_path, providers=self.providers)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [o.name for o in self.session.get_outputs()]
            print(f"[ONNX] loaded: {self.model_path}")
            print(f"[ONNX] input shape: {self.input_shape}")
            print(f"[ONNX] providers : {self.session.get_providers()}")
        except Exception as e:
            raise RuntimeError(f"ONNX 모델 로드 실패: {e}")

    def set_params(self, input_width: int, input_height: int, conf_threshold: float,
                   iou_threshold: float, clip_coords: bool, invert_colors: bool,
                   center_score: bool | None = None, tiny_box_side_px: int | None = None):
        self.input_width = input_width
        self.input_height = input_height
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.clip_coords = clip_coords
        self.invert_colors = invert_colors
        if center_score is not None:
            self.center_score = center_score
        if tiny_box_side_px is not None:
            self.tiny_box_side_px = tiny_box_side_px

    # ---------------------- 전처리 ----------------------
    def preprocess_image(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, dict]:
        if self.invert_colors:
            image_rgb = 255 - image_rgb
        img_resized, t = self._letterbox_resize(image_rgb, (self.input_width, self.input_height))
        img = img_resized.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC->CHW
        img = np.expand_dims(img, 0)        # NCHW
        return img, t

    def _letterbox_resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, dict]:
        tw, th = target_size
        h, w = image.shape[:2]
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pw, ph = tw - nw, th - nh
        pl, pt = pw // 2, ph // 2
        pr, pb = pw - pl, ph - pt
        padded = cv2.copyMakeBorder(resized, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, {'scale': scale, 'pad_left': pl, 'pad_top': pt, 'new_width': nw, 'new_height': nh}

    # ---------------------- 후처리(디버그 포함) ----------------------
    def postprocess_detections(self, outputs: List[np.ndarray], t: dict,
                           original_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int, float]]:
        if not outputs:
            return []

        pred = outputs[0]
        if pred.ndim == 3:
            pred = pred[0]  # [N,C]
        if self.debug:
            print(f"[DBG] raw pred shape: {pred.shape}")

        if pred.shape[1] < 5:
            if self.debug:
                print("[DBG] unexpected output shape (<5 cols).")
            return []

        # ---- A) raw 분석 ----
        coord_min, coord_max = float(pred[:, :4].min()), float(pred[:, :4].max())
        conf_min, conf_max = float(pred[:, 4].min()), float(pred[:, 4].max())
        if self.debug:
            print(f"[DBG] raw coord range: {coord_min:.4f} ~ {coord_max:.4f}")
            print(f"[DBG] raw conf  range: {conf_min:.4f} ~ {conf_max:.4f}")

        # ---- B) YOLOv5 raw head인지 자동 판단 -> 디코딩 ----
        need_decode = False
        N = pred.shape[0]
        if N in (80*80*3 + 40*40*3 + 20*20*3, 25200):
            # 좌표가 작은 범위(그리드/로짓 전형)고 conf는 로짓이면 디코딩 필요
            if (coord_max <= 20.0 and coord_min >= -50.0) and (conf_max > 1.0 or conf_min < 0.0):
                need_decode = True

        if need_decode:
            if self.debug:
                print("[DBG] YOLOv5 raw head detected → applying grid/anchor decode")
            xywh, conf = self._decode_yolov5(pred)   # ← (입력해상도 픽셀 좌표계, 패딩 포함)
        else:
            # 기존 경로: 이미 디코딩된 xywh라고 가정
            xywh = pred[:, :4].astype(np.float32)
            # conf: logit일 수도 확률일 수도 → 안전하게
            obj = self._sigmoid(pred[:, 4].astype(np.float32)) if (conf_max > 1.0 or conf_min < 0.0) else pred[:, 4].astype(np.float32)
            if pred.shape[1] > 5:
                cls_raw = pred[:, 5:].astype(np.float32)
                cls_prob = self._sigmoid(cls_raw) if (conf_max > 1.0 or conf_min < 0.0) else cls_raw
                conf = obj * cls_prob.max(axis=1)
            else:
                conf = obj

        # ---- C) conf 필터
        keep = conf >= self.conf_threshold
        if self.debug:
            print(f"[DBG] keep >= {self.conf_threshold}: {int(keep.sum())}/{len(keep)}")
        if not np.any(keep):
            return []
        xywh = xywh[keep]
        conf = conf[keep]

        # xywh가 지금 "입력해상도 픽셀(패딩 포함)" 좌표라는 전제
        # letterbox 역변환 (패딩 제거 + 1/scale) — 필요 시 스킵 토글로 A/B
        x_c = xywh[:, 0].astype(np.float32)
        y_c = xywh[:, 1].astype(np.float32)
        w   = xywh[:, 2].astype(np.float32)
        h   = xywh[:, 3].astype(np.float32)

        if not getattr(self, "skip_depad_scale", False):
            x_c -= t['pad_left']
            y_c -= t['pad_top']
            s = t['scale'] if t['scale'] != 0 else 1.0
            x_c /= s; y_c /= s; w /= s; h /= s
            if self.debug:
                print(f"[DBG] x_c after depad/scale: shape={x_c.shape}, min={x_c.min():.4f}, max={x_c.max():.4f}")
                print(f"[DBG] y_c after depad/scale: shape={y_c.shape}, min={y_c.min():.4f}, max={y_c.max():.4f}")
        else:
            if self.debug:
                print("[DBG] ⚠ skip_depad_scale=True → depad/scale 복원 스킵")

        x1 = (x_c - w/2).astype(np.int32)
        y1 = (y_c - h/2).astype(np.int32)
        x2 = (x_c + w/2).astype(np.int32)
        y2 = (y_c + h/2).astype(np.int32)

        if self.clip_coords:
            H, W = original_shape
            x1 = np.clip(x1, 0, W-1); y1 = np.clip(y1, 0, H-1)
            x2 = np.clip(x2, 0, W-1); y2 = np.clip(y2, 0, H-1)

        if self.debug:
            xyxy = np.stack([x1, y1, x2, y2], axis=1)
            print(f"[DBG] xyxy before NMS (orig): shape={xyxy.shape}, min={xyxy.min()}, max={xyxy.max()}")
            print("[DBG] sample xyxy before NMS:\n", xyxy[:min(len(xyxy), 6)])
            print("[DBG] sample conf before NMS:\n", conf[:min(len(conf), 6)])

        boxes_xywh = np.stack([x1, y1, x2-x1, y2-y1], axis=1).tolist()
        confidences = conf.astype(float).tolist()
        if not boxes_xywh:
            return []

        idxs = cv2.dnn.NMSBoxes(boxes_xywh, confidences, self.conf_threshold, self.iou_threshold)
        faces: List[Tuple[int,int,int,int,float]] = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, ww, hh = boxes_xywh[i]
                if ww > 0 and hh > 0:
                    faces.append((int(x), int(y), int(x+ww), int(y+hh), float(confidences[i])))

        if self.debug:
            print(f"[DBG] NMS -> {len(faces)} boxes")
            for j, (xx1, yy1, xx2, yy2, cc) in enumerate(faces[:6]):
                print(f"  #{j}: x1={xx1}, y1={yy1}, x2={xx2}, y2={yy2}, conf={cc:.4f}")

        return faces


    # ---------------------- 텍스트 렌더링 ----------------------
    @staticmethod
    def _put_text_with_outline(img: np.ndarray, text: str, org: tuple[int, int],
                               font_scale: float, thickness: int, color_fg=(255, 255, 255)):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color_fg, thickness, cv2.LINE_AA)

    def _draw_centered_score(self, img: np.ndarray, box: tuple[int, int, int, int], score: float):
        x1, y1, x2, y2 = box
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        short_side = min(w, h)
        font_scale = max(0.4, min(1.2, short_side / 200.0))
        thickness = 2
        label = f"{score:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cx = x1 + w // 2; cy = y1 + h // 2
        ox = int(cx - tw / 2); oy = int(cy + th / 2)
        ox = max(0, min(ox, img.shape[1] - tw))
        oy = max(th, min(oy, img.shape[0] - base))
        self._put_text_with_outline(img, label, (ox, oy), font_scale, thickness)

    def _draw_corner_score(self, img: np.ndarray, box: tuple[int, int, int, int], score: float):
        x1, y1, x2, y2 = box
        label = f"{score:.2f}"
        fs, th = 0.5, 1
        (tw, thh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        bg_tl = (x1, max(0, y1 - thh - 6))
        bg_br = (x1 + tw + 6, y1)
        cv2.rectangle(img, bg_tl, bg_br, (0, 255, 0), -1)
        self._put_text_with_outline(img, label, (x1 + 3, y1 - 3), fs, th, color_fg=(0, 0, 0))

    # ---------------------- 실행 ----------------------
    def detect(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, float]]]:
        inp, t = self.preprocess_image(image_rgb)
        try:
            outputs = self.session.run(self.output_names, {self.input_name: inp})
        except Exception as e:
            raise RuntimeError(f"ONNX 추론 실패: {e}")

        faces = self.postprocess_detections(outputs, t, image_rgb.shape[:2])

        result = image_rgb.copy()
        for idx, (x1, y1, x2, y2, conf) in enumerate(faces):
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 박스 번호도 찍기(로그와 매칭)
            cv2.putText(result, f"#{idx}", (x1 + 4, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(result, f"#{idx}", (x1 + 4, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            w = x2 - x1; h = y2 - y1
            if self.center_score and min(w, h) >= self.tiny_box_side_px:
                self._draw_centered_score(result, (x1, y1, x2, y2), conf)
            else:
                self._draw_corner_score(result, (x1, y1, x2, y2), conf)

            print(f"Detected face #{idx}: ({x1}, {y1}) to ({x2}, {y2}), confidence: {conf:.2f}")

        return result, faces

    def _decode_yolov5(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        pred: [25200, 5+nc] (raw head outputs)
        return:
        xywh: [N,4] in input image pixels (letterbox 포함 좌표계)
        conf: [N]
        """
        no = int(pred.shape[1])          # 5 + nc
        nc = no - 5

        # strides는 정수! 640 입력 기준
        strides = np.array([8, 16, 32], dtype=np.int32)

        anchors = np.array([
            [10,13, 16,30, 33,23],        # P3/8
            [30,61, 62,45, 59,119],       # P4/16
            [116,90, 156,198, 373,326],   # P5/32
        ], dtype=np.float32).reshape(3, 3, 2)

        # 각 레벨의 (ny, nx)를 반드시 int로 계산
        shapes: list[tuple[int,int]] = []
        for s in strides:
            ny = int(self.input_height // int(s))
            nx = int(self.input_width  // int(s))
            shapes.append((ny, nx))

        counts = [int(ny * nx * 3) for (ny, nx) in shapes]
        assert sum(counts) == int(pred.shape[0]), f"Unexpected output size: {pred.shape[0]} vs {sum(counts)}"

        ofs = 0
        xywh_list = []
        conf_list = []

        for li, (ny, nx) in enumerate(shapes):
            cnt = int(ny * nx * 3)
            p = pred[int(ofs):int(ofs + cnt), :]     # <-- 슬라이싱 인덱스는 int
            ofs += cnt

            # [ny, nx, 3, no] 로 reshape (ny/nx도 int)
            p = p.reshape(int(ny), int(nx), 3, int(no))

            # overflow-safe sigmoid
            ps = 1.0 / (1.0 + np.exp(-np.clip(p, -50, 50)))

            xy = ps[..., 0:2] * 2.0 - 0.5
            wh = (ps[..., 2:4] * 2.0) ** 2

            # grid
            yv, xv = np.meshgrid(np.arange(ny, dtype=np.float32),
                                np.arange(nx, dtype=np.float32),
                                indexing="ij")
            grid = np.stack((xv, yv), axis=-1).reshape(ny, nx, 1, 2)

            anc = anchors[li].reshape(1, 1, 3, 2).astype(np.float32)
            stride = float(strides[li])

            xy = (xy + grid) * stride
            wh = wh * anc

            obj = ps[..., 4]
            if nc > 0:
                cls_max = ps[..., 5:].max(axis=-1)
                conf = obj * cls_max
            else:
                conf = obj

            xy = xy.reshape(-1, 2)
            wh = wh.reshape(-1, 2)
            conf = conf.reshape(-1)

            xywh_level = np.concatenate([xy, wh], axis=1)
            xywh_list.append(xywh_level.astype(np.float32))
            conf_list.append(conf.astype(np.float32))

        xywh = np.concatenate(xywh_list, axis=0)
        conf = np.concatenate(conf_list, axis=0)
        return xywh, conf


# quick run
if __name__ == "__main__":
    if HAS_ORT:
        print("ONNX Runtime OK:", ort.get_available_providers())
    else:
        print("Install onnxruntime / onnxruntime-gpu first.")
