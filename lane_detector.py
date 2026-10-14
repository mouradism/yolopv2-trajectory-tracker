"""
Advanced Lane Line Detection Pipeline (Option B - Calibrated for Ego Lane)
Features:
- Calibrated Ego-Lane Perspective Mapping (Bird's-Eye View)
- Ridge-based Top-Hat Morphology & Color Thresholding (Isolates lane stripes, rejects guardrails & mountains)
- Sliding Window Search & Prior-Frame Polynomial Tracking
- 2nd Degree Polynomial Curve Fitting: x = A*y^2 + B*y + C
- Real-world Radius of Curvature (m) and Lateral Vehicle Offset (m)
- Temporal Smoothing (EMA) across video frames
- Road Horizon Clipping & HUD Telemetry Overlay
"""

import cv2
import numpy as np


class LaneTracker:
    def __init__(
        self,
        img_size=(1280, 720),
        src_points=None,
        dst_points=None,
        smoothing_factor=0.20,
    ):
        """
        Initialize LaneTracker.
        
        Args:
            img_size (tuple): (width, height) of the input video frames.
            src_points (np.ndarray): 4 trapezoidal source points for ego-lane perspective warp.
            dst_points (np.ndarray): 4 rectangular destination points in bird's-eye view.
            smoothing_factor (float): Weight for exponential moving average (0 to 1).
        """
        self.w, self.h = img_size
        self.smoothing_factor = smoothing_factor

        # Calibrated ego-lane source points (focused strictly on ego lane between yellow divider and right white line)
        if src_points is None:
            self.src = np.float32([
                [280, 680],   # Bottom-left: yellow divider
                [540, 480],   # Top-left: horizon divider
                [690, 480],   # Top-right: horizon white line
                [840, 680]    # Bottom-right: right white line
            ])
        else:
            self.src = np.float32(src_points)

        # Destination points in warped top-down bird's-eye space
        if dst_points is None:
            self.dst = np.float32([
                [300, self.h],  # Bottom-left
                [300, 0],       # Top-left
                [800, 0],       # Top-right
                [800, self.h]   # Bottom-right
            ])
        else:
            self.dst = np.float32(dst_points)

        # Compute perspective transform matrices
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

        # Morphological structuring element for line extraction
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

        # Real-world metric conversions
        self.ym_per_pix = 30.0 / 720.0   # ~30 meters projected along height
        self.xm_per_pix = 3.7 / 500.0    # ~3.7 meters standard lane width (500 px in warped space)

        # State tracking for temporal smoothing
        self.detected = False
        self.best_left_fit = None
        self.best_right_fit = None

    def warp_perspective(self, img):
        """Transform camera view to top-down bird's-eye view."""
        return cv2.warpPerspective(img, self.M, (self.w, self.h), flags=cv2.INTER_LINEAR)

    def unwarp_perspective(self, img):
        """Transform top-down view back to original camera perspective."""
        return cv2.warpPerspective(img, self.Minv, (self.w, self.h), flags=cv2.INTER_LINEAR)

    def threshold_frame(self, warped_bgr):
        """
        Isolate lane markings on road asphalt while suppressing guardrails, mountains, and sand.
        Combines Morphological Top-Hat (bright line ridges) with HSV Yellow detection.
        """
        warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Morphological Top-Hat filter extracts thin bright ridges (lane markings) on darker road
        tophat = cv2.morphologyEx(warped_gray, cv2.MORPH_TOPHAT, self.kernel)
        _, bin_ridges = cv2.threshold(tophat, 18, 255, cv2.THRESH_BINARY)

        # 2. Yellow lane divider detection in HSV space
        hsv_w = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
        yellow_w = cv2.inRange(hsv_w, np.array([12, 50, 70]), np.array([35, 255, 255]))

        # Combined binary output
        combined = cv2.bitwise_or(bin_ridges, yellow_w)

        # Strict boundary suppression: zero out guardrails (x > 920) and opposite barrier (x < 180)
        combined[:, :180] = 0
        combined[:, 920:] = 0

        binary = (combined > 0).astype(np.uint8)
        return binary

    def find_lane_pixels_sliding_window(self, binary_warped, nwindows=9, margin=65, minpix=30):
        """Locate lane pixels across horizontal sliding window strips."""
        histogram = np.sum(binary_warped[self.h // 2 :, :], axis=0)

        # Constrained search for lane bases in warped space
        l_search = histogram[200:450]
        r_search = histogram[650:900]

        leftx_base = np.argmax(l_search) + 200 if len(l_search) > 0 and np.max(l_search) > 30 else 300
        rightx_base = np.argmax(r_search) + 650 if len(r_search) > 0 and np.max(r_search) > 30 else 800

        window_height = int(self.h // nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = self.h - (window + 1) * window_height
            win_y_high = self.h - window * window_height

            win_xleft_low = max(0, leftx_current - margin)
            win_xleft_high = min(self.w, leftx_current + margin)
            win_xright_low = max(0, rightx_current - margin)
            win_xright_high = min(self.w, rightx_current + margin)

            good_left_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_low)
                & (nonzerox < win_xleft_high)
            ).nonzero()[0]

            good_right_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_low)
                & (nonzerox < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def find_lane_pixels_from_prior(self, binary_warped, margin=55):
        """Search in a margin around the previously fitted polynomial curves."""
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_fit = self.best_left_fit
        right_fit = self.best_right_fit

        left_center = left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2]
        right_center = right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2]

        left_lane_inds = (nonzerox >= left_center - margin) & (nonzerox <= left_center + margin)
        right_lane_inds = (nonzerox >= right_center - margin) & (nonzerox <= right_center + margin)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def fit_polynomial(self, binary_warped):
        """Fit 2nd-degree polynomials to detected left & right lane pixels."""
        if self.detected and self.best_left_fit is not None and self.best_right_fit is not None:
            leftx, lefty, rightx, righty = self.find_lane_pixels_from_prior(binary_warped)
            if len(leftx) < 80 or len(rightx) < 80:
                leftx, lefty, rightx, righty = self.find_lane_pixels_sliding_window(binary_warped)
        else:
            leftx, lefty, rightx, righty = self.find_lane_pixels_sliding_window(binary_warped)

        valid_left = len(leftx) > 40
        valid_right = len(rightx) > 40

        if valid_left:
            curr_left_fit = np.polyfit(lefty, leftx, 2)
        else:
            curr_left_fit = self.best_left_fit

        if valid_right:
            curr_right_fit = np.polyfit(righty, rightx, 2)
        else:
            curr_right_fit = self.best_right_fit

        if curr_left_fit is None or curr_right_fit is None:
            return None, None, None, None

        ploty = np.linspace(0, self.h - 1, self.h)
        l_eval = curr_left_fit[0] * ploty**2 + curr_left_fit[1] * ploty + curr_left_fit[2]
        r_eval = curr_right_fit[0] * ploty**2 + curr_right_fit[1] * ploty + curr_right_fit[2]
        lane_width_bottom = r_eval[-1] - l_eval[-1]
        lane_width_top = r_eval[0] - l_eval[0]

        is_sane = (320 < lane_width_bottom < 680) and (280 < lane_width_top < 700)

        # Temporal smoothing with Exponential Moving Average (EMA)
        if self.best_left_fit is None or not is_sane:
            if is_sane or self.best_left_fit is None:
                self.best_left_fit = curr_left_fit
                self.best_right_fit = curr_right_fit
        else:
            alpha = self.smoothing_factor
            self.best_left_fit = (1 - alpha) * self.best_left_fit + alpha * curr_left_fit
            self.best_right_fit = (1 - alpha) * self.best_right_fit + alpha * curr_right_fit

        self.detected = True

        left_fitx = self.best_left_fit[0] * ploty**2 + self.best_left_fit[1] * ploty + self.best_left_fit[2]
        right_fitx = self.best_right_fit[0] * ploty**2 + self.best_right_fit[1] * ploty + self.best_right_fit[2]

        return left_fitx, right_fitx, ploty, (leftx, lefty, rightx, righty)

    def calculate_metrics(self, ploty, left_fitx, right_fitx):
        """Calculate road curvature radius (meters) and vehicle lateral offset (meters)."""
        y_eval = np.max(ploty)

        left_fit_cr = np.polyfit(ploty * self.ym_per_pix, left_fitx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_pix, right_fitx * self.xm_per_pix, 2)

        left_curverad = (
            (1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5
        ) / (np.absolute(2 * left_fit_cr[0]) + 1e-6)

        right_curverad = (
            (1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5
        ) / (np.absolute(2 * right_fit_cr[0]) + 1e-6)

        curverad = (left_curverad + right_curverad) / 2.0

        # Unwarp bottom lane centers to evaluate lateral offset in camera frame
        pts_bottom = cv2.perspectiveTransform(
            np.array([[[left_fitx[-1], self.h - 1], [right_fitx[-1], self.h - 1]]], dtype=np.float32),
            self.Minv
        )
        lx_orig = pts_bottom[0, 0, 0]
        rx_orig = pts_bottom[0, 1, 0]
        lane_center_x = (lx_orig + rx_orig) / 2.0
        car_center_x = self.w / 2.0
        offset_m = (car_center_x - lane_center_x) * (3.7 / max(rx_orig - lx_orig, 100))

        return curverad, offset_m

    def draw_lane_overlay(self, original_img, left_fitx, right_fitx, ploty, curverad, offset_m):
        """Draw unwarped lane polygon, lane boundary lines, and telemetry HUD on original frame."""
        warp_zero = np.zeros((self.h, self.w), dtype=np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Lane area in semi-transparent green
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 220, 0))

        # Left boundary in red, right boundary in blue
        cv2.polylines(color_warp, np.int_([pts_left]), False, (0, 0, 255), 20)
        cv2.polylines(color_warp, np.int_([pts_right]), False, (255, 0, 0), 20)

        # Unwarp back into driving perspective
        unwarped_lane = self.unwarp_perspective(color_warp)

        # Clip corridor so it doesn't extend beyond the road horizon ahead (y < 485)
        unwarped_lane[:485, :] = 0

        # Combine with original image
        result = cv2.addWeighted(original_img, 1.0, unwarped_lane, 0.45, 0)

        # HUD Overlay Banner
        hud_h = 130
        hud_w = 480
        overlay = result.copy()
        cv2.rectangle(overlay, (20, 20), (20 + hud_w, 20 + hud_h), (20, 20, 20), -1)
        result = cv2.addWeighted(overlay, 0.65, result, 0.35, 0)
        cv2.rectangle(result, (20, 20), (20 + hud_w, 20 + hud_h), (0, 200, 255), 2)

        curve_str = f"Curvature: {curverad:,.0f} m" if curverad < 8000 else "Curvature: Straight"
        direction = "Right" if offset_m > 0 else "Left"
        offset_str = f"Offset: {abs(offset_m):.2f} m ({direction} of center)"

        cv2.putText(result, "ADVANCED LANE TRACKER", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.putText(result, curve_str, (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(result, offset_str, (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return result

    def process_frame(self, frame):
        """End-to-end processing pipeline for a single frame."""
        warped = self.warp_perspective(frame)
        binary = self.threshold_frame(warped)
        left_fitx, right_fitx, ploty, _ = self.fit_polynomial(binary)

        if left_fitx is None or right_fitx is None:
            return frame

        curverad, offset_m = self.calculate_metrics(ploty, left_fitx, right_fitx)
        output_frame = self.draw_lane_overlay(frame, left_fitx, right_fitx, ploty, curverad, offset_m)
        return output_frame
