"""
Live Real-Time YOLOP / YOLOPv2 Panoptic Driving Perception Player with 2D Aerial Radar Map
and Optimal Trajectory Path Planning (in Blue) via Tranche Surface Area Integration & Cubic Spline.

Usage:
  import live_yolop_tracker
  live_yolop_tracker.run_live_tracker(video_path="2026-10-14 14-08-25.mp4", model_path="yolopv2.onnx")

Controls:
  [M]           : Cycle 2D Aerial Map Mode (Dual-View Side-by-Side -> PiP Inset -> Off)
  [T]           : Toggle Optimal Trajectory Path (in Blue)
  [SPACE] / [P] : Pause / Resume
  [B]           : Toggle Vehicle Bounding Boxes
  [S]           : Toggle Frame-Skip mode
  [[] / []]     : Move Blue Ego Car Left / Right (Fine-tuning)
  [Q] / [ESC]   : Quit
  [R]           : Restart from frame 0
  [RIGHT]       : Skip forward 1 sec
  [LEFT]        : Rewind 1 sec
"""

import collections
import sys
import time
import os
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from yolop_detector import YOLOPDetector


class LaneWidthTracker:
    """Tracks moving average of detected lane width (distance between red lines) across frames."""
    def __init__(self, default_w=72.0):
        self.val = float(default_w)

    def update(self, new_val):
        if new_val is not None and 50.0 <= new_val <= 110.0:
            self.val = 0.85 * self.val + 0.15 * float(new_val)
        return self.val


def measure_average_lane_width(warped_ll, default_width=72.0):
    """
    Measure the average distance between adjacent red lines (lane markings) in BEV space.
    Standard highway lane width corresponds to ~68-75 pixels (~3.4 - 3.7m).
    """
    if warped_ll is None:
        return default_width

    widths = []
    # Sample horizontal rows across the active driving range
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


def limit_spline_curvature(x_raw, eval_y, kappa_max=0.003, ego_x=198.0, n_iter=25):
    """
    Constrained Trajectory Smoother & Spline Maximum Curvature Limiter.
    Limits the maximum geometric curvature kappa(y) <= kappa_max to eliminate
    steering wheel jerking, sudden lateral snaps, and unstable high-frequency oscillations.

    Args:
        x_raw (np.ndarray): Lateral coordinates of the spline.
        eval_y (np.ndarray): Longitudinal sample positions.
        kappa_max (float): Maximum allowed curvature (default 0.003, turning radius R_min >= ~28m).
        ego_x (float): Lateral anchor for ego vehicle.
        n_iter (int): Optimization relaxation iterations.

    Returns:
        np.ndarray: Curvature-bounded smooth trajectory.
    """
    x = x_raw.copy()
    n = len(x)
    dy = abs(eval_y[1] - eval_y[0])
    x[0] = float(ego_x)

    for it in range(n_iter):
        # 1. Heading continuity at vehicle front bumper (prevents abrupt dogleg turns at vehicle hood)
        if n > 2:
            max_head_dx = 0.08 * dy  # Limit initial heading angle relative to vehicle body
            x[1] = np.clip(x[1], x[0] - max_head_dx, x[0] + max_head_dx)

        # 2. Curvature constraint: |d2x/dy2| <= kappa_max * (1 + dx^2)^1.5
        for i in range(1, n - 1):
            dx = (x[i + 1] - x[i - 1]) / (2.0 * dy)
            limit_d2x = kappa_max * ((1.0 + dx**2) ** 1.5) * (dy**2)
            d2x = x[i + 1] - 2.0 * x[i] + x[i - 1]
            if abs(d2x) > limit_d2x:
                excess = abs(d2x) - limit_d2x
                x[i] = x[i] + np.sign(d2x) * (excess * 0.5)

        # 3. Laplacian smoothing to minimize steering jerk (rate of curvature change)
        d2 = x[2:] - 2 * x[1:-1] + x[:-2]
        x[1:-1] += 0.12 * d2

    return x


def compute_optimal_trajectory_tranches(
    da_mask,
    ll_mask,
    M_bev,
    M_inv,
    ego_x,
    ego_y,
    history_buffer=None,
    lane_width_tracker=None,
    kappa_max=0.003,
    map_size=(360, 720),
):
    """
    Compute optimal trajectory by treating all possible lane solutions as candidates,
    relying on the average distance between red lines (lane width), and picking the candidate
    that avoids drastic position jumps, strictly capping spline maximum curvature to eliminate
    steering wheel jerking.

    Args:
        da_mask (np.ndarray): Binary drivable area mask (green road).
        ll_mask (np.ndarray): Binary lane line mask (red markings).
        M_bev (np.ndarray): Perspective transformation matrix to Bird's-Eye View.
        M_inv (np.ndarray): Inverse perspective matrix from BEV back to camera view.
        ego_x (int): Horizontal position of the ego vehicle anchor in BEV (e.g. 198).
        ego_y (int): Vertical position of the ego vehicle anchor in BEV (e.g. 640).
        history_buffer (collections.deque): Buffer of up to 10 past fitted trajectories for rolling average.
        lane_width_tracker (LaneWidthTracker): Moving average tracker for lane width.
        kappa_max (float): Maximum allowed curvature (default 0.006, R_min >= ~14m).
        map_size (tuple): Width and height of the BEV map (default 360x720).

    Returns:
        final_fit_x (np.ndarray): Curvature-bounded 10-frame smoothed spline x-coordinates in BEV space.
        eval_y (np.ndarray): Dense y-coordinates in BEV space.
        pts_cam (np.ndarray): Projected trajectory points in front camera space.
        tranches (tuple): Selected candidate coordinates (tranche_pts_x, tranche_pts_y).
        lane_w (float): Active average distance between red lines in pixels.
        curv_info (tuple): (peak_curvature, min_radius_meters, kappa_max).
    """
    map_w, map_h = map_size
    warped_da = cv2.warpPerspective(da_mask, M_bev, (map_w, map_h), flags=cv2.INTER_NEAREST)
    warped_ll = cv2.warpPerspective(ll_mask, M_bev, (map_w, map_h), flags=cv2.INTER_NEAREST) if ll_mask is not None else None

    # 1. Dynamically measure / track average lane width (distance between red lines)
    meas_w = measure_average_lane_width(warped_ll, default_width=72.0)
    if lane_width_tracker is not None:
        lane_w = lane_width_tracker.update(meas_w)
    else:
        lane_w = meas_w

    y_start = ego_y - 20   # Near vehicle front bumper
    y_end = 120             # Far horizon
    tranche_h = 10          # Height of each integration tranche
    eval_y = np.linspace(y_start, y_end, 70)

    tranche_pts_x = [float(ego_x)]
    tranche_pts_y = [float(y_start)]
    track_x = float(ego_x)

    # 2. Build 10-frame historical position lookup function: y -> average x
    hist_avg_func = None
    if history_buffer is not None and len(history_buffer) > 0:
        hist_arr = np.array(list(history_buffer))
        mean_x = np.mean(hist_arr, axis=0)
        hist_avg_func = lambda y_val: np.interp(y_val, eval_y[::-1], mean_x[::-1])

    # 3. Integrate road by horizontal tranches and evaluate candidates
    for y_top in range(y_start, y_end, -tranche_h):
        y_bot = max(y_top - tranche_h, y_end)
        y_mid = (y_top + y_bot) / 2.0

        slice_da = warped_da[y_bot:y_top, :]
        da_sum = np.sum(slice_da == 1, axis=0)
        green_cols = np.where(da_sum >= (tranche_h * 0.25))[0]

        if len(green_cols) < 8:
            continue

        # Segment contiguous drivable corridors (gaps > 15px indicate separate roads/curbs)
        da_gaps = np.where(np.diff(green_cols) > 15)[0]
        da_segs = np.split(green_cols, da_gaps + 1)

        # Detect red lane line clusters in this tranche
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

        # Reference position from the last 10 frames' rolling average
        hist_ref_x = float(hist_avg_func(y_mid)) if hist_avg_func is not None else float(ego_x)

        # 4. Generate all candidate trajectory solutions
        candidates = []

        # Candidate Option A: Direct red line pair midpoints (true lane center)
        if len(ll_clusters) >= 2:
            for i in range(len(ll_clusters) - 1):
                d = ll_clusters[i + 1] - ll_clusters[i]
                if 0.60 * lane_w <= d <= 1.45 * lane_w:
                    candidates.append(((ll_clusters[i] + ll_clusters[i + 1]) / 2.0, True))

        # Candidate Option B: Drivable area corridor midpoints & sub-lane centers
        for seg in da_segs:
            if len(seg) >= 8:
                seg_min = float(seg[0])
                seg_max = float(seg[-1])
                seg_w = seg_max - seg_min
                seg_mid = 0.5 * (seg_min + seg_max)

                if seg_w <= 1.35 * lane_w:
                    # Single-lane road or narrow corridor: corridor center
                    candidates.append((seg_mid, False))
                else:
                    # Wide road or multi-lane corridor: candidate sub-lanes spaced by lane_w
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

        # 5. Score candidates: pick solution that minimizes lateral jump & matches 10-frame history
        alpha = (y_start - y_mid) / max(y_start - y_end, 1)
        w_hist = 0.50
        w_track = 0.35 + 0.20 * alpha
        w_ego = max(0.0, 0.25 * (1.0 - alpha))

        best_cand = None
        min_cost = 1e9

        for cand_x, is_ll_pair in candidates:
            d_hist = abs(cand_x - hist_ref_x)
            d_track = abs(cand_x - track_x)
            d_ego = abs(cand_x - ego_x)

            cost = w_hist * d_hist + w_track * d_track + w_ego * d_ego

            # Bonus for true painted lane lines
            if is_ll_pair:
                cost -= 10.0

            # Progressive continuous penalty for lateral deviations to keep trajectory soft and straight
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

    # 6. Fit soft quadratic spline with high regularization to prevent high-frequency S-bends
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

    # 7. Apply 1D Gaussian kernel relaxation and strictly limit maximum curvature
    raw_fit_x = gaussian_filter1d(raw_fit_x, sigma=3.0)
    raw_fit_x = limit_spline_curvature(raw_fit_x, eval_y, kappa_max=kappa_max, ego_x=ego_x, n_iter=35)

    # 8. Apply 10-Frame Rolling Average for rock-solid stability without drastic jumps
    if history_buffer is not None:
        history_buffer.append(raw_fit_x)
        final_fit_x = np.mean(list(history_buffer), axis=0)
    else:
        final_fit_x = raw_fit_x

    # Ensure curvature limit and softening is preserved on the smoothed ensemble
    final_fit_x = gaussian_filter1d(final_fit_x, sigma=2.0)
    final_fit_x = limit_spline_curvature(final_fit_x, eval_y, kappa_max=kappa_max, ego_x=ego_x, n_iter=25)

    # Calculate real-time curvature telemetry
    dx = np.gradient(final_fit_x, eval_y)
    d2x = np.gradient(dx, eval_y)
    curv = np.abs(d2x) / np.power(1.0 + dx**2, 1.5)
    peak_kappa = float(np.max(curv))
    min_radius_m = float(1.0 / max(peak_kappa, 1e-6) / 11.55)
    curv_info = (peak_kappa, min_radius_m, kappa_max)

    # 9. Project spline to front camera perspective
    pts_bev_f = np.stack([final_fit_x, eval_y], axis=1).reshape(-1, 1, 2).astype(np.float32)
    pts_cam = cv2.perspectiveTransform(pts_bev_f, M_inv).reshape(-1, 2)

    return final_fit_x, eval_y, pts_cam, (tranche_pts_x, tranche_pts_y), lane_w, curv_info


def draw_optimal_trajectory(cam_view, bev_map, fit_x, eval_y, pts_cam, M_inv, tranches=None, lane_w=None, curv_info=None):
    """
    Draw the optimal trajectory path in vibrant blue on both the Front Camera View and 2D BEV Map.
    """
    if fit_x is None or eval_y is None or len(fit_x) < 2:
        return cam_view, bev_map

    # 1. Draw on 2D Aerial BEV Map (in vibrant electric blue)
    if bev_map is not None:
        # Draw subtle candidate tranche centroid dots
        if tranches is not None:
            tx_list, ty_list = tranches
            for tx, ty in zip(tx_list[1:], ty_list[1:]):
                cv2.circle(bev_map, (int(round(tx)), int(round(ty))), 2, (0, 240, 255), -1, lineType=cv2.LINE_AA)

        bev_traj = np.stack([fit_x, eval_y], axis=1).astype(np.int32)
        # Outer cyan/sky-blue glow (BGR)
        cv2.polylines(bev_map, [bev_traj], isClosed=False, color=(255, 200, 50), thickness=5, lineType=cv2.LINE_AA)
        # Inner deep electric blue core (BGR)
        cv2.polylines(bev_map, [bev_traj], isClosed=False, color=(255, 60, 0), thickness=2, lineType=cv2.LINE_AA)

        # Trajectory waypoint targets (10m, 20m, 30m, 40m ahead)
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
            left_cam = cv2.perspectiveTransform(left_bev, M_inv).reshape(-1, 2).astype(np.int32)
            right_cam = cv2.perspectiveTransform(right_bev, M_inv).reshape(-1, 2).astype(np.int32)

            overlay = cam_view.copy()
            polygon = np.vstack([left_cam, right_cam[::-1]])
            cv2.fillPoly(overlay, [polygon], (235, 110, 0)) # Soft translucent blue guidance ribbon
            cam_view = cv2.addWeighted(cam_view, 0.76, overlay, 0.24, 0)

            # Centerline in vivid electric blue
            cam_traj = pts_cam[valid].astype(np.int32)
            cv2.polylines(cam_view, [cam_traj], isClosed=False, color=(255, 220, 80), thickness=4, lineType=cv2.LINE_AA)
            cv2.polylines(cam_view, [cam_traj], isClosed=False, color=(255, 60, 0), thickness=2, lineType=cv2.LINE_AA)

    return cam_view, bev_map


def run_live_tracker(
    video_path="2026-10-14 14-08-25.mp4",
    model_path="yolopv2.onnx",
    window_name="YOLOPv2 Driving Perception & 2D Aerial Map (Press Q to quit)",
    frame_skip=1,
    map_mode="dual", # "dual" (side-by-side), "pip" (inset), or "off"
    ego_x=198,
    show_trajectory=True,
    kappa_max=0.003,
):
    """
    Run live interactive YOLOP / YOLOPv2 driving perception with 2D Aerial Live Map
    and Curvature-Limited Optimal Trajectory Path in blue.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Fallback if yolopv2.onnx not found
    if not os.path.exists(model_path) and os.path.exists("yolop-640-640.onnx"):
        print(f"Notice: '{model_path}' not found, falling back to 'yolop-640-640.onnx'")
        model_path = "yolop-640-640.onnx"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 65)
    print(f"[*] INITIALIZING {os.path.basename(model_path).upper()} WITH 2D AERIAL RADAR...")
    detector = YOLOPDetector(model_path=model_path, conf_thres=0.30, ego_x=ego_x)
    print(f"[+] Active Providers: {detector.active_providers}")
    print(f"[+] Model Version: {detector.version_str}")
    print(f"[+] Ego Car X Anchor: {detector.ego_x} (Lane Centered)")
    print(f"[+] Blue Optimal Trajectory: {'ENABLED' if show_trajectory else 'DISABLED'} (Curvature Capped: {kappa_max:.3f})")
    print(f"File: {video_path} ({total_frames} frames @ {orig_fps:.1f} FPS)")
    print("Controls:")
    print("  [M]           : Cycle Aerial Map Mode (Dual-View -> PiP -> Off)")
    print("  [T]           : Toggle Optimal Trajectory Path (in Blue)")
    print("  [K]           : Cycle Max Curvature Limit (0.002 -> 0.003 -> 0.005)")
    print("  [SPACE] / [P] : Pause / Resume")
    print("  [B]           : Toggle Vehicle Bounding Boxes")
    print("  [S]           : Toggle Frame-Skip mode")
    print("  [[] / []]     : Move Blue Ego Car Left / Right (Fine-tuning)")
    print("  [Q] / [ESC]   : Quit")
    print("  [R]           : Restart")
    print("=" * 65)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1640 if map_mode == "dual" else 1280, 720)

    modes = ["dual", "pip", "off"]
    curr_mode = map_mode if map_mode in modes else "dual"

    paused = False
    show_boxes = True
    show_traj = show_trajectory
    active_skip = frame_skip
    fps_smooth = orig_fps
    active_kappa = float(kappa_max)

    last_cam_view = None
    last_bev_map = None
    history_buffer = collections.deque(maxlen=10)
    lane_tracker = LaneWidthTracker(default_w=72.0)

    frame_idx = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached. Restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            frame_idx += 1
            t_start = time.time()

            # Run full inference every active_skip frames
            if last_cam_view is None or (frame_idx % active_skip == 0):
                cam_view, da_mask, ll_mask, boxes, last_bev_map = detector.process_frame(
                    frame, show_boxes=show_boxes, generate_bev=(curr_mode in ["dual", "pip"]), show_trajectory=False
                )

                # Compute and draw optimal trajectory directly in live_yolop_tracker
                if show_traj and da_mask is not None:
                    fit_x, eval_y, pts_cam, tranches, lane_w, curv_info = compute_optimal_trajectory_tranches(
                        da_mask, ll_mask, detector.M_bev, detector.M_inv, detector.ego_x, detector.ego_y,
                        history_buffer=history_buffer, lane_width_tracker=lane_tracker,
                        kappa_max=active_kappa, map_size=(360, frame.shape[0])
                    )
                    cam_view, last_bev_map = draw_optimal_trajectory(
                        cam_view, last_bev_map, fit_x, eval_y, pts_cam, detector.M_inv,
                        tranches=tranches, lane_w=lane_w, curv_info=curv_info
                    )

                last_cam_view = cam_view
            else:
                cam_view = last_cam_view.copy()

            proc_time = time.time() - t_start
            current_fps = 1.0 / max(proc_time, 1e-4)
            fps_smooth = 0.85 * fps_smooth + 0.15 * current_fps

            curr_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            progress_pct = (curr_frame / max(total_frames, 1)) * 100
            time_curr = curr_frame / orig_fps
            time_total = total_frames / orig_fps

            # Bottom info bar
            info_str = f"FPS: {fps_smooth:.1f} | Frame: {curr_frame}/{total_frames} ({progress_pct:.1f}%) | Time: {time_curr:.1f}s / {time_total:.1f}s"
            cv2.rectangle(cam_view, (20, 680), (1260, 710), (15, 15, 15), -1)
            cv2.putText(cam_view, info_str, (30, 702), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            traj_status = "ON" if show_traj else "OFF"
            cv2.putText(
                cam_view,
                f"[M] Map  [T] Trajectory:{traj_status}  [K] CurvLimit:{active_kappa:.3f}  [B] Boxes  [SPACE] Pause  [[]/[]] Ego X  [Q] Quit",
                (375, 702),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (0, 220, 255),
                1,
            )

            # Assemble Display based on curr_mode
            if curr_mode == "dual" and last_bev_map is not None:
                display_frame = np.hstack([cam_view, last_bev_map])
            elif curr_mode == "pip" and last_bev_map is not None:
                pip_h, pip_w = 300, 150
                pip_resized = cv2.resize(last_bev_map, (pip_w, pip_h))
                pip_x, pip_y = 1280 - pip_w - 20, 20
                cv2.rectangle(cam_view, (pip_x - 2, pip_y - 2), (pip_x + pip_w + 2, pip_y + pip_h + 2), (0, 220, 255), 2)
                cam_view[pip_y : pip_y + pip_h, pip_x : pip_x + pip_w] = pip_resized
                display_frame = cam_view
            else:
                display_frame = cam_view

        cv2.imshow(window_name, display_frame)

        wait_ms = 1 if not paused else 50
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in [ord("q"), 27]:
            print("Exiting YOLOP player.")
            break
        elif key in [ord(" "), ord("p")]:
            paused = not paused
            print(f"Playback {'PAUSED' if paused else 'RESUMED'}")
        elif key == ord("m"):
            idx = (modes.index(curr_mode) + 1) % len(modes)
            curr_mode = modes[idx]
            print(f"Switched Aerial Map Mode to: {curr_mode.upper()}")
            if curr_mode == "dual":
                cv2.resizeWindow(window_name, 1640, 720)
            else:
                cv2.resizeWindow(window_name, 1280, 720)
        elif key == ord("b"):
            show_boxes = not show_boxes
            print(f"Vehicle bounding boxes: {'ENABLED' if show_boxes else 'DISABLED'}")
        elif key == ord("s"):
            active_skip = 2 if active_skip == 1 else 1
            print(f"Frame-skip mode: {active_skip}x")
        elif key == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            print("Restarted from frame 0.")
        elif key in [83, ord("c")]:
            curr = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(curr + 30, total_frames - 1))
        elif key in [81, ord("x")]:
            curr = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(curr - 30, 0))
        elif key in [ord("k"), ord("K")]:
            presets = [0.002, 0.003, 0.005]
            curr_idx = presets.index(active_kappa) if active_kappa in presets else 1
            active_kappa = presets[(curr_idx + 1) % len(presets)]
            min_r_m = 1.0 / active_kappa / 11.55
            print(f"[Dynamics] Spline Curvature Limit updated: kappa_max = {active_kappa:.3f} (R_min >= {min_r_m:.1f}m)")
            if paused and 'frame' in locals() and frame is not None:
                cam_view, da_mask, ll_mask, boxes, last_bev_map = detector.process_frame(
                    frame, show_boxes=show_boxes, generate_bev=(curr_mode in ["dual", "pip"]), show_trajectory=False
                )
                if show_traj and da_mask is not None:
                    fit_x, eval_y, pts_cam, tranches, lane_w, curv_info = compute_optimal_trajectory_tranches(
                        da_mask, ll_mask, detector.M_bev, detector.M_inv, detector.ego_x, detector.ego_y,
                        history_buffer=history_buffer, lane_width_tracker=lane_tracker,
                        kappa_max=active_kappa, map_size=(360, frame.shape[0])
                    )
                    cam_view, last_bev_map = draw_optimal_trajectory(
                        cam_view, last_bev_map, fit_x, eval_y, pts_cam, detector.M_inv,
                        tranches=tranches, lane_w=lane_w, curv_info=curv_info
                    )
                last_cam_view = cam_view
        elif key in [ord("t"), ord("T")]:
            show_traj = not show_traj
            print(f"Optimal Trajectory Path: {'ENABLED' if show_traj else 'DISABLED'}")
            if paused and 'frame' in locals() and frame is not None:
                cam_view, da_mask, ll_mask, boxes, last_bev_map = detector.process_frame(
                    frame, show_boxes=show_boxes, generate_bev=(curr_mode in ["dual", "pip"]), show_trajectory=False
                )
                if show_traj and da_mask is not None:
                    fit_x, eval_y, pts_cam, tranches, lane_w, curv_info = compute_optimal_trajectory_tranches(
                        da_mask, ll_mask, detector.M_bev, detector.M_inv, detector.ego_x, detector.ego_y,
                        history_buffer=history_buffer, lane_width_tracker=lane_tracker,
                        kappa_max=active_kappa, map_size=(360, frame.shape[0])
                    )
                    cam_view, last_bev_map = draw_optimal_trajectory(
                        cam_view, last_bev_map, fit_x, eval_y, pts_cam, detector.M_inv,
                        tranches=tranches, lane_w=lane_w, curv_info=curv_info
                    )
                last_cam_view = cam_view
        elif key in [ord("["), ord("{")]:
            detector.ego_x = max(40, detector.ego_x - 2)
            history_buffer.clear()
            print(f"[Calibration] Ego car shifted Left -> X: {detector.ego_x}")
            if paused and 'frame' in locals() and frame is not None:
                cam_view, da_mask, ll_mask, boxes, last_bev_map = detector.process_frame(
                    frame, show_boxes=show_boxes, generate_bev=(curr_mode in ["dual", "pip"]), show_trajectory=False
                )
                if show_traj and da_mask is not None:
                    fit_x, eval_y, pts_cam, tranches, lane_w, curv_info = compute_optimal_trajectory_tranches(
                        da_mask, ll_mask, detector.M_bev, detector.M_inv, detector.ego_x, detector.ego_y,
                        history_buffer=history_buffer, lane_width_tracker=lane_tracker,
                        kappa_max=active_kappa, map_size=(360, frame.shape[0])
                    )
                    cam_view, last_bev_map = draw_optimal_trajectory(
                        cam_view, last_bev_map, fit_x, eval_y, pts_cam, detector.M_inv,
                        tranches=tranches, lane_w=lane_w, curv_info=curv_info
                    )
                last_cam_view = cam_view
        elif key in [ord("]"), ord("}")]:
            detector.ego_x = min(320, detector.ego_x + 2)
            history_buffer.clear()
            print(f"[Calibration] Ego car shifted Right -> X: {detector.ego_x}")
            if paused and 'frame' in locals() and frame is not None:
                cam_view, da_mask, ll_mask, boxes, last_bev_map = detector.process_frame(
                    frame, show_boxes=show_boxes, generate_bev=(curr_mode in ["dual", "pip"]), show_trajectory=False
                )
                if show_traj and da_mask is not None:
                    fit_x, eval_y, pts_cam, tranches, lane_w, curv_info = compute_optimal_trajectory_tranches(
                        da_mask, ll_mask, detector.M_bev, detector.M_inv, detector.ego_x, detector.ego_y,
                        history_buffer=history_buffer, lane_width_tracker=lane_tracker,
                        kappa_max=active_kappa, map_size=(360, frame.shape[0])
                    )
                    cam_view, last_bev_map = draw_optimal_trajectory(
                        cam_view, last_bev_map, fit_x, eval_y, pts_cam, detector.M_inv,
                        tranches=tranches, lane_w=lane_w, curv_info=curv_info
                    )
                last_cam_view = cam_view

    cap.release()
    cv2.destroyAllWindows()


# Backward-compatible alias
run_live_yolop = run_live_tracker

if __name__ == "__main__":
    default_vid = "2026-10-14 14-08-25.mp4" 
    vid = sys.argv[1] if len(sys.argv) > 1 else default_vid
    run_live_tracker(vid, model_path="yolopv2.onnx")
