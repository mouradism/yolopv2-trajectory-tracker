"""
YOLOP & YOLOPv2 Panoptic Driving Perception Pipeline
Hardware-accelerated via DirectML (Windows GPU) and CUDA.

Supports:
- YOLOPv2 (Default: yolopv2.onnx - E-ELAN / YOLOv7 architecture, higher accuracy)
- YOLOP v1 (yolop-640-640.onnx)
- 2D Aerial Bird's-Eye View (BEV) Live Radar Map Projection
"""

import os
import sys
import time
import collections
import cv2
import numpy as np
import onnxruntime as ort
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline


def _make_grid(nx=20, ny=20):
    xv, yv = np.meshgrid(np.arange(0, nx), np.arange(0, ny))
    return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)


def _sigmoid(arr):
    arr = np.array(arr, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-1.0 * arr))


def _xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _nms(boxes, scores, iou_threshold):
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)


def split_for_trace_model(pred, anchor_grid):
    z = []
    st = [8, 16, 32]
    for i in range(3):
        bs, _, ny, nx = pred[i].shape
        p = pred[i].reshape(bs, 3, 85, ny, nx).transpose(0, 1, 3, 4, 2)
        y = _sigmoid(p)
        gr = _make_grid(nx, ny)
        y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + gr) * st[i]
        y[..., 2:4] = (y[..., 2:4] * 2.0) ** 2 * anchor_grid[i]
        z.append(y.reshape(bs, -1, 85))
    return np.concatenate(z, 1)


def non_max_suppression(prediction, conf_thres=0.30, iou_thres=0.45):
    xc = prediction[..., 4] > conf_thres
    output = []
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            output.append(np.zeros((0, 6)))
            continue
        x[:, 5:] *= x[:, 4:5]
        box = _xywh2xyxy(x[:, :4])
        conf = np.max(x[:, 5:], axis=1, keepdims=True)
        j = np.argmax(x[:, 5:], axis=1).reshape((-1, 1))
        x = np.concatenate((box, conf, j.astype(np.float32)), 1)[conf.reshape(-1) > conf_thres]
        if not x.shape[0]:
            output.append(np.zeros((0, 6)))
            continue
        boxes, scores = x[:, :4], x[:, 4]
        keep = _nms(boxes, scores, iou_thres)
        output.append(x[keep[:100]])
    return output


class YOLOPDetector:
    def __init__(self, model_path="yolopv2.onnx", conf_thres=0.30, iou_thres=0.45, ego_x=198, ego_y=640):
        """
        Initialize YOLOP / YOLOPv2 Deep Learning Perception Detector.
        
        Args:
            model_path (str): Path to yolopv2.onnx or yolop-640-640.onnx.
            conf_thres (float): Object detection confidence threshold.
            iou_thres (float): NMS IoU threshold.
            ego_x (int): Lateral anchor for ego vehicle on BEV map (default 198 centered in lane).
            ego_y (int): Longitudinal anchor for ego vehicle on BEV map (default 640).
        """
        # If default yolopv2.onnx doesn't exist, fallback to yolop-640-640.onnx
        if not os.path.exists(model_path) and os.path.exists("yolop-640-640.onnx"):
            print(f"Notice: '{model_path}' not found, falling back to 'yolop-640-640.onnx'")
            model_path = "yolop-640-640.onnx"

        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Prioritize GPU providers: DirectML (DmlExecutionProvider) for Windows GPUs / CUDA
        available = ort.get_available_providers()
        priority_providers = []
        for p in ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]:
            if p in available:
                priority_providers.append(p)
        if not priority_providers:
            priority_providers = ["CPUExecutionProvider"]

        # Initialize ONNX Runtime session with GPU acceleration
        self.session = ort.InferenceSession(
            model_path,
            providers=priority_providers
        )
        self.active_providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        
        # Determine whether model is YOLOP v1 or YOLOPv2
        self.is_v2 = "input_image" in self.input_name or "v2" in model_path.lower()
        self.version_str = "YOLOPv2 (SOTA)" if self.is_v2 else "YOLOP v1"

        # ImageNet normalization stats
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # 2D Aerial BEV Homography Matrix
        src_bev = np.float32([[260, 680], [540, 485], [690, 485], [840, 680]])
        dst_bev = np.float32([[130, 620], [130, 100], [230, 100], [230, 620]])
        self.M_bev = cv2.getPerspectiveTransform(src_bev, dst_bev)
        self.M_inv = np.linalg.inv(self.M_bev)
        self.meters_scale = (620.0 - 100.0) / 45.0
        self.ego_x, self.ego_y = ego_x, ego_y
        self.prev_fit_x = None
        self.history_buffer = collections.deque(maxlen=10)
        self.lane_width = 72.0

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """Resize and pad image while meeting stride constraints."""
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (new_shape[1] - new_unpad[0]) // 2
        dh = (new_shape[0] - new_unpad[1]) // 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = dh, new_shape[0] - new_unpad[1] - dh
        left, right = dw, new_shape[1] - new_unpad[0] - dw
        padded_img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return padded_img, (r, dw, dh, new_unpad)

    def preprocess(self, img_bgr):
        """Preprocess frame for YOLOP inference."""
        padded, meta = self.letterbox(img_bgr, (640, 640))
        if self.is_v2:
            # YOLOPv2 uses standard 0..1 RGB scaling
            rgb = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            tensor = np.expand_dims(rgb, axis=0)
        else:
            # YOLOP v1 ImageNet normalization
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            normalized = (rgb - self.mean) / self.std
            tensor = np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)
        return tensor, meta

    def postprocess_v2(self, results, meta, orig_shape):
        """Post-process YOLOPv2 multi-task outputs."""
        r, dw, dh, new_unpad = meta
        h0, w0 = orig_shape[:2]

        # 1. Drivable area mask (results[4])
        da_raw = results[4][0, 0]
        seg_h, seg_w = da_raw.shape
        da_crop = 1.0 - da_raw[dh : seg_h - dh, dw : seg_w - dw]
        da_orig = (cv2.resize(da_crop, (w0, h0), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8)

        # 2. Lane line mask (results[5])
        ll_raw = results[5][0, 0]
        ll_h, ll_w = ll_raw.shape
        ll_crop = ll_raw[dh : ll_h - dh, dw : ll_w - dw]
        ll_orig = (cv2.resize(ll_crop, (w0, h0), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8)

        # 3. Object detection bounding boxes
        result_dets = [results[0][0], results[0][1], results[0][2]]
        anchor_grid = [results[1], results[2], results[3]]
        pred = split_for_trace_model(result_dets, anchor_grid)
        det_nms = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres)[0]

        boxes = []
        for det in det_nms:
            x1, y1, x2, y2, score, cls_id = det[:6]
            # Rescale coords back to original image
            x1_o = int(round(max(0, (x1 - dw) / r)))
            y1_o = int(round(max(0, (y1 - dh) / r)))
            x2_o = int(round(min(w0, (x2 - dw) / r)))
            y2_o = int(round(min(h0, (y2 - dh) / r)))
            bw = x2_o - x1_o
            bh = y2_o - y1_o
            if bw > 5 and bh > 5:
                boxes.append(([x1_o, y1_o, bw, bh], float(score)))

        return da_orig, ll_orig, boxes

    def postprocess_v1(self, det_out, da_seg, ll_seg, meta, orig_shape):
        """Post-process YOLOP v1 outputs."""
        r, dw, dh, new_unpad = meta
        h0, w0 = orig_shape[:2]

        da_mask = np.argmax(da_seg[0], axis=0).astype(np.uint8)
        ll_mask = np.argmax(ll_seg[0], axis=0).astype(np.uint8)

        da_crop = da_mask[dh : dh + new_unpad[1], dw : dw + new_unpad[0]]
        ll_crop = ll_mask[dh : dh + new_unpad[1], dw : dw + new_unpad[0]]

        da_orig = cv2.resize(da_crop, (w0, h0), interpolation=cv2.INTER_NEAREST)
        ll_orig = cv2.resize(ll_crop, (w0, h0), interpolation=cv2.INTER_NEAREST)

        # Boxes
        predictions = det_out[0]
        boxes_list = []
        conf_list = []
        for pred in predictions:
            obj_conf = pred[4]
            if obj_conf > self.conf_thres:
                class_scores = pred[5:]
                class_id = np.argmax(class_scores)
                score = obj_conf * class_scores[class_id]
                if score > self.conf_thres:
                    cx, cy, w, h = pred[0:4]
                    x1 = int(round((cx - w / 2 - dw) / r))
                    y1 = int(round((cy - h / 2 - dh) / r))
                    bw = int(round(w / r))
                    bh = int(round(h / r))
                    boxes_list.append([max(0, x1), max(0, y1), bw, bh])
                    conf_list.append(float(score))

        indices = cv2.dnn.NMSBoxes(boxes_list, conf_list, self.conf_thres, self.iou_thres)
        boxes = []
        if len(indices) > 0:
            for idx in indices.flatten():
                boxes.append((boxes_list[idx], conf_list[idx]))
        return da_orig, ll_orig, boxes

    def generate_aerial_map(self, da_mask, ll_mask, boxes, map_size=(360, 720)):
        """Generate a 2D Aerial Bird's-Eye View (BEV) live radar map."""
        map_w, map_h = map_size
        bev_map = np.full((map_h, map_w, 3), (18, 20, 24), dtype=np.uint8)

        # Warp Drivable Area & Lane Lines
        warped_da = cv2.warpPerspective(da_mask, self.M_bev, (map_w, map_h), flags=cv2.INTER_NEAREST)
        warped_ll = cv2.warpPerspective(ll_mask, self.M_bev, (map_w, map_h), flags=cv2.INTER_NEAREST)

        bev_map[warped_da == 1] = [35, 125, 45]     # Emerald green drivable surface
        bev_map[warped_ll == 1] = [0, 240, 255]     # Crisp lane lines

        # Distance range arcs (10m, 20m, 30m, 40m)
        for d_m in [10, 20, 30, 40]:
            r_pix = int(d_m * self.meters_scale)
            cv2.circle(bev_map, (self.ego_x, self.ego_y), r_pix, (45, 50, 60), 1, cv2.LINE_AA)
            cv2.putText(
                bev_map, f"{d_m}m", (self.ego_x + r_pix - 32, self.ego_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 130, 145), 1, cv2.LINE_AA
            )

        # Ego Vehicle
        car_w, car_h = 24, 48
        cv2.rectangle(
            bev_map,
            (self.ego_x - car_w // 2, self.ego_y - car_h // 2),
            (self.ego_x + car_w // 2, self.ego_y + car_h // 2),
            (240, 110, 20), -1
        )
        cv2.rectangle(
            bev_map,
            (self.ego_x - car_w // 2, self.ego_y - car_h // 2),
            (self.ego_x + car_w // 2, self.ego_y + car_h // 2),
            (255, 210, 60), 2
        )
        cv2.line(bev_map, (self.ego_x - 7, self.ego_y - car_h // 2), (self.ego_x - 12, self.ego_y - car_h // 2 - 14), (0, 255, 255), 2)
        cv2.line(bev_map, (self.ego_x + 7, self.ego_y - car_h // 2), (self.ego_x + 12, self.ego_y - car_h // 2 - 14), (0, 255, 255), 2)
        cv2.putText(bev_map, "EGO", (self.ego_x - 14, self.ego_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Project Surrounding Vehicles
        for (x, y, w, h), conf in boxes:
            ground_pt = np.array([[[x + w / 2.0, y + h]]], dtype=np.float32)
            bev_pt = cv2.perspectiveTransform(ground_pt, self.M_bev)[0, 0]
            bx, by = int(round(bev_pt[0])), int(round(bev_pt[1]))

            if 15 <= bx < map_w - 15 and 20 <= by < map_h - 20:
                dist_m = np.sqrt(((bx - self.ego_x) / (100.0 / 3.7)) ** 2 + ((self.ego_y - by) / self.meters_scale) ** 2)
                bw_bev, bh_bev = 24, 42
                cv2.rectangle(
                    bev_map,
                    (bx - bw_bev // 2, by - bh_bev // 2),
                    (bx + bw_bev // 2, by + bh_bev // 2),
                    (0, 140, 255), -1
                )
                cv2.rectangle(
                    bev_map,
                    (bx - bw_bev // 2, by - bh_bev // 2),
                    (bx + bw_bev // 2, by + bh_bev // 2),
                    (0, 230, 255), 2
                )
                cv2.putText(
                    bev_map, f"{dist_m:.1f}m", (bx - 20, by - bh_bev // 2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA
                )

        # Header Title
        cv2.rectangle(bev_map, (0, 0), (map_w, 45), (12, 14, 18), -1)
        cv2.putText(bev_map, "2D AERIAL RADAR (BEV)", (40, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2)
        cv2.line(bev_map, (0, 45), (map_w, 45), (0, 220, 255), 1)

        return bev_map

    def measure_average_lane_width(self, warped_ll, default_width=72.0):
        """
        Measure average distance between adjacent red lines (lane markings) in BEV space.
        """
        if warped_ll is None:
            return default_width
        widths = []
        for y in range(250, 620, 10):
            row = warped_ll[y, :]
            line_cols = np.where(row > 0)[0]
            if len(line_cols) >= 2:
                diffs = np.where(np.diff(line_cols) > 15)[0]
                clusters = np.split(line_cols, diffs + 1)
                if len(clusters) >= 2:
                    centroids = [float(np.mean(c)) for c in clusters if len(c) >= 2]
                    for i in range(len(centroids) - 1):
                        d = centroids[i + 1] - centroids[i]
                        if 45.0 <= d <= 115.0:
                            widths.append(d)
        if len(widths) >= 3:
            return float(np.median(widths))
        return default_width

    def compute_optimal_trajectory(self, da_mask, ll_mask=None, kappa_max=0.003, map_size=(360, 720)):
        """
        Compute optimal trajectory by generating lane candidates from drivable corridor
        and red line markings, and picking the candidate minimizing lateral departure from
        ego vehicle and the 10-frame rolling average, strictly limiting spline max curvature
        to minimize steering wheel jerking.
        """
        map_w, map_h = map_size
        warped_da = cv2.warpPerspective(da_mask, self.M_bev, (map_w, map_h), flags=cv2.INTER_NEAREST)
        warped_ll = cv2.warpPerspective(ll_mask, self.M_bev, (map_w, map_h), flags=cv2.INTER_NEAREST) if ll_mask is not None else None

        # Measure / update lane width
        meas_w = self.measure_average_lane_width(warped_ll, default_width=72.0)
        if meas_w is not None and 50.0 <= meas_w <= 110.0:
            self.lane_width = 0.85 * self.lane_width + 0.15 * meas_w
        lane_w = self.lane_width

        y_start = self.ego_y - 20
        y_end = 120
        tranche_h = 10
        eval_y = np.linspace(y_start, y_end, 70)

        tranche_pts_x = [float(self.ego_x)]
        tranche_pts_y = [float(y_start)]
        track_x = float(self.ego_x)

        # 10-frame historical average lookup
        hist_avg_func = None
        if len(self.history_buffer) > 0:
            hist_arr = np.array(list(self.history_buffer))
            mean_x = np.mean(hist_arr, axis=0)
            hist_avg_func = lambda y_val: np.interp(y_val, eval_y[::-1], mean_x[::-1])

        # Slice road into horizontal tranches from near to far
        for y_top in range(y_start, y_end, -tranche_h):
            y_bot = max(y_top - tranche_h, y_end)
            y_mid = (y_top + y_bot) / 2.0

            slice_da = warped_da[y_bot:y_top, :]
            da_sum = np.sum(slice_da == 1, axis=0)
            green_cols = np.where(da_sum >= (tranche_h * 0.25))[0]

            if len(green_cols) < 8:
                continue

            da_gaps = np.where(np.diff(green_cols) > 15)[0]
            da_segs = np.split(green_cols, da_gaps + 1)

            ll_clusters = []
            if warped_ll is not None:
                slice_ll = warped_ll[y_bot:y_top, :]
                ll_sum = np.sum(slice_ll > 0, axis=0)
                ll_cols = np.where(ll_sum >= 2)[0]
                if len(ll_cols) > 0:
                    ll_gaps = np.where(np.diff(ll_cols) > 15)[0]
                    for c in np.split(ll_cols, ll_gaps + 1):
                        if len(c) >= 2:
                            ll_clusters.append(float(np.mean(c)))

            hist_ref_x = float(hist_avg_func(y_mid)) if hist_avg_func is not None else float(self.ego_x)

            # Generate Candidates
            candidates = []

            # Option A: Red line pair midpoints
            if len(ll_clusters) >= 2:
                for i in range(len(ll_clusters) - 1):
                    d = ll_clusters[i + 1] - ll_clusters[i]
                    if 0.60 * lane_w <= d <= 1.45 * lane_w:
                        candidates.append(((ll_clusters[i] + ll_clusters[i + 1]) / 2.0, True))

            # Option B: Drivable corridor midpoints & sub-lane centers
            for seg in da_segs:
                if len(seg) >= 8:
                    seg_min = float(seg[0])
                    seg_max = float(seg[-1])
                    seg_w = seg_max - seg_min
                    seg_mid = 0.5 * (seg_min + seg_max)

                    if seg_w <= 1.35 * lane_w:
                        candidates.append((seg_mid, False))
                    else:
                        candidates.append((seg_mid, False))
                        candidates.append((seg_min + 0.5 * lane_w, False))
                        candidates.append((seg_max - 0.5 * lane_w, False))
                        if seg_w > 2.2 * lane_w:
                            k = 1
                            while seg_min + (k + 0.5) * lane_w < seg_max - 0.2 * lane_w:
                                candidates.append((seg_min + (k + 0.5) * lane_w, False))
                                k += 1

            if not candidates:
                continue

            # Candidate cost evaluation
            alpha = (y_start - y_mid) / max(y_start - y_end, 1)
            w_hist = 0.50
            w_track = 0.35 + 0.20 * alpha
            w_ego = max(0.0, 0.25 * (1.0 - alpha))

            best_cand = None
            min_cost = 1e9

            for cand_x, is_ll_pair in candidates:
                d_hist = abs(cand_x - hist_ref_x)
                d_track = abs(cand_x - track_x)
                d_ego = abs(cand_x - self.ego_x)

                cost = w_hist * d_hist + w_track * d_track + w_ego * d_ego
                if is_ll_pair:
                    cost -= 10.0
                if d_track > 0.30 * lane_w:
                    cost += ((d_track - 0.30 * lane_w) ** 1.8) * 1.5

                if cost < min_cost:
                    min_cost = cost
                    best_cand = cand_x

            if best_cand is not None:
                tranche_pts_x.append(float(best_cand))
                tranche_pts_y.append(float(y_mid))
                track_x = 0.85 * track_x + 0.15 * best_cand

        if len(tranche_pts_x) < 4:
            return None, None, None, None, lane_w, None

        # Fit soft quadratic spline with high regularization to prevent high-frequency S-bends
        y_arr = np.array(tranche_pts_y)
        x_arr = np.array(tranche_pts_x)
        sort_idx = np.argsort(y_arr)
        y_unique, u_idx = np.unique(y_arr[sort_idx], return_index=True)
        x_unique = x_arr[sort_idx][u_idx]

        try:
            k_order = min(2, len(y_unique) - 1)
            s_smooth = len(y_unique) * 35.0
            spl = UnivariateSpline(y_unique, x_unique, k=k_order, s=s_smooth, ext=3)
            raw_fit_x = spl(eval_y)
        except Exception:
            poly = np.polyfit(y_unique, x_unique, min(2, len(y_unique) - 1))
            raw_fit_x = np.polyval(poly, eval_y)

        # Helper relaxation to limit spline max curvature
        def _limit_curv(x_in, n_iter=25):
            x_out = x_in.copy()
            n_pts = len(x_out)
            dy_step = abs(eval_y[1] - eval_y[0])
            x_out[0] = float(self.ego_x)
            for _ in range(n_iter):
                if n_pts > 2:
                    x_out[1] = np.clip(x_out[1], x_out[0] - 0.08 * dy_step, x_out[0] + 0.08 * dy_step)
                for i in range(1, n_pts - 1):
                    dx_i = (x_out[i + 1] - x_out[i - 1]) / (2.0 * dy_step)
                    limit_d2 = kappa_max * ((1.0 + dx_i**2) ** 1.5) * (dy_step**2)
                    d2_i = x_out[i + 1] - 2.0 * x_out[i] + x_out[i - 1]
                    if abs(d2_i) > limit_d2:
                        x_out[i] += np.sign(d2_i) * (abs(d2_i) - limit_d2) * 0.5
                d2_lap = x_out[2:] - 2 * x_out[1:-1] + x_out[:-2]
                x_out[1:-1] += 0.12 * d2_lap
            return x_out

        raw_fit_x = gaussian_filter1d(raw_fit_x, sigma=3.0)
        raw_fit_x = _limit_curv(raw_fit_x, n_iter=35)

        # 10-Frame Rolling Average
        self.history_buffer.append(raw_fit_x)
        final_fit_x = np.mean(list(self.history_buffer), axis=0)
        final_fit_x = gaussian_filter1d(final_fit_x, sigma=2.0)
        final_fit_x = _limit_curv(final_fit_x, n_iter=25)

        # Calculate curvature telemetry
        dx = np.gradient(final_fit_x, eval_y)
        d2x = np.gradient(dx, eval_y)
        curv = np.abs(d2x) / np.power(1.0 + dx**2, 1.5)
        peak_kappa = float(np.max(curv))
        min_radius_m = float(1.0 / max(peak_kappa, 1e-6) / 11.55)
        curv_info = (peak_kappa, min_radius_m, kappa_max)

        # Project spline to front camera perspective
        pts_bev_f = np.stack([final_fit_x, eval_y], axis=1).reshape(-1, 1, 2).astype(np.float32)
        pts_cam = cv2.perspectiveTransform(pts_bev_f, self.M_inv).reshape(-1, 2)

        return final_fit_x, eval_y, pts_cam, (tranche_pts_x, tranche_pts_y), lane_w, curv_info

    def draw_trajectory(self, cam_view, bev_map, fit_x, eval_y, pts_cam, tranches=None, lane_w=None, curv_info=None):
        """
        Draw the optimal trajectory path in vibrant blue on both Camera View and 2D BEV Map.
        """
        if fit_x is None or eval_y is None or len(fit_x) < 2:
            return cam_view, bev_map

        # 1. Draw on 2D Aerial BEV Map (in vibrant electric blue)
        if bev_map is not None:
            # Draw subtle tranche centroid dots to visualize integration
            if tranches is not None:
                tx_list, ty_list = tranches
                for tx, ty in zip(tx_list[1:], ty_list[1:]):
                    cv2.circle(bev_map, (int(round(tx)), int(round(ty))), 2, (0, 240, 255), -1, lineType=cv2.LINE_AA)

            bev_traj = np.stack([fit_x, eval_y], axis=1).astype(np.int32)
            # Outer cyan/sky-blue glow (BGR)
            cv2.polylines(bev_map, [bev_traj], isClosed=False, color=(255, 200, 50), thickness=5, lineType=cv2.LINE_AA)
            # Inner deep electric blue core (BGR)
            cv2.polylines(bev_map, [bev_traj], isClosed=False, color=(255, 60, 0), thickness=2, lineType=cv2.LINE_AA)

            # Trajectory waypoint targets (10m, 20m, 30m ahead)
            for idx in [15, 30, 45, len(fit_x) - 1]:
                if idx < len(fit_x):
                    bx, by = int(fit_x[idx]), int(eval_y[idx])
                    cv2.circle(bev_map, (bx, by), 4, (255, 220, 0), -1, lineType=cv2.LINE_AA)
                    cv2.circle(bev_map, (bx, by), 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            # Telemetry badge: Lane Width & Curvature Limit
            if lane_w is not None or curv_info is not None:
                cv2.rectangle(bev_map, (10, 50), (350, 75), (14, 16, 20), -1)
                cv2.rectangle(bev_map, (10, 50), (350, 75), (0, 200, 255), 1)

                w_part = f"Lane: {lane_w:.1f}px" if lane_w is not None else ""
                if curv_info is not None:
                    peak_k, r_min_m, k_max = curv_info
                    telemetry_str = f"{w_part} | Rmin: {r_min_m:.1f}m (k<={k_max:.3f})"
                else:
                    telemetry_str = f"{w_part} | 10-Frame Cand Avg"

                cv2.putText(
                    bev_map,
                    telemetry_str,
                    (14, 67),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 230, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 2. Draw on Front Camera View (in vibrant blue ribbon + spline)
        if cam_view is not None and pts_cam is not None:
            h_cam, w_cam = cam_view.shape[:2]
            valid = (pts_cam[:, 0] >= -50) & (pts_cam[:, 0] <= w_cam + 50) & (pts_cam[:, 1] >= 0) & (pts_cam[:, 1] <= h_cam)
            if np.sum(valid) > 4:
                valid_fit_x = fit_x[valid]
                valid_eval_y = eval_y[valid]

                # Project guidance ribbon (+/- 20px in BEV for softer visual blending)
                ribbon_w = 20.0
                left_bev = np.stack([valid_fit_x - ribbon_w, valid_eval_y], axis=1).reshape(-1, 1, 2).astype(np.float32)
                right_bev = np.stack([valid_fit_x + ribbon_w, valid_eval_y], axis=1).reshape(-1, 1, 2).astype(np.float32)
                left_cam = cv2.perspectiveTransform(left_bev, self.M_inv).reshape(-1, 2).astype(np.int32)
                right_cam = cv2.perspectiveTransform(right_bev, self.M_inv).reshape(-1, 2).astype(np.int32)

                overlay = cam_view.copy()
                polygon = np.vstack([left_cam, right_cam[::-1]])
                cv2.fillPoly(overlay, [polygon], (235, 110, 0)) # Soft translucent blue guidance ribbon
                cam_view = cv2.addWeighted(cam_view, 0.76, overlay, 0.24, 0)

                # Centerline in vivid electric blue
                cam_traj = pts_cam[valid].astype(np.int32)
                cv2.polylines(cam_view, [cam_traj], isClosed=False, color=(255, 220, 80), thickness=4, lineType=cv2.LINE_AA)
                cv2.polylines(cam_view, [cam_traj], isClosed=False, color=(255, 60, 0), thickness=2, lineType=cv2.LINE_AA)

        return cam_view, bev_map

    def process_frame(self, frame_bgr, show_boxes=True, generate_bev=False, show_trajectory=True):
        """
        Run end-to-end panoptic perception on a single frame.
        """
        orig_shape = frame_bgr.shape
        tensor, meta = self.preprocess(frame_bgr)

        # Run ONNX inference
        raw_results = self.session.run(None, {self.input_name: tensor})

        if self.is_v2:
            da_orig, ll_orig, boxes = self.postprocess_v2(raw_results, meta, orig_shape)
        else:
            da_orig, ll_orig, boxes = self.postprocess_v1(raw_results[0], raw_results[1], raw_results[2], meta, orig_shape)

        # Draw overlays on camera view
        output = frame_bgr.copy()
        mask_overlay = output.copy()
        mask_overlay[da_orig == 1] = [0, 220, 50]  # Emerald green drivable surface
        mask_overlay[ll_orig == 1] = [0, 0, 255]   # Crimson red lane lines
        output = cv2.addWeighted(output, 0.65, mask_overlay, 0.35, 0)

        # Vehicle bounding boxes
        if show_boxes:
            for (x, y, bw, bh), conf in boxes:
                cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 220, 255), 2)
                label = f"Vehicle {conf:.2f}"
                cv2.putText(
                    output, label, (x, max(y - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2
                )

        # Generate 2D Aerial BEV Map
        bev_map = None
        if generate_bev:
            bev_map = self.generate_aerial_map(da_orig, ll_orig, boxes, map_size=(360, orig_shape[0]))

        # Optimal Trajectory Path (in Blue via Tranche Surface Integration + Spline)
        if show_trajectory:
            fit_x, eval_y, pts_cam, tranches, lane_w, curv_info = self.compute_optimal_trajectory(da_orig, ll_mask=ll_orig, map_size=(360, orig_shape[0]))
            output, bev_map = self.draw_trajectory(output, bev_map, fit_x, eval_y, pts_cam, tranches=tranches, lane_w=lane_w, curv_info=curv_info)

        # HUD Banner
        hud_h, hud_w = 125, 440
        cv2.rectangle(output, (20, 20), (20 + hud_w, 20 + hud_h), (20, 20, 20), -1)
        cv2.rectangle(output, (20, 20), (20 + hud_w, 20 + hud_h), (0, 220, 255), 2)
        cv2.putText(output, f"{self.version_str} PERCEPTION", (35, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.putText(output, "Green: Drivable Area  |  Red: Lane Lines", (35, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(output, "Blue: Optimal Trajectory Path", (35, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 180, 0), 1)
        cv2.putText(output, f"GPU: GTX 1060 | Model: {os.path.basename(self.model_path)}", (35, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)

        return output, da_orig, ll_orig, boxes, bev_map
