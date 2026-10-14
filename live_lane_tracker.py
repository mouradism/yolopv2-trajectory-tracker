"""
Live Real-Time Dashcam Lane Tracking with OpenCV
Features:
- Live playback window with augmented reality lane corridor
- Real-time telemetry HUD (Radius of curvature, lane offset, FPS)
- Picture-in-Picture (PiP) top-down Bird's-Eye view inset
- Interactive controls:
    [SPACE] / [P] : Pause / Resume
    [Q] / [ESC]   : Quit
    [R]           : Restart from frame 0
    [D]           : Toggle Bird's-Eye Mini Map
    [RIGHT]       : Skip forward 1 sec
    [LEFT]        : Rewind 1 sec
"""

import sys
import time
import cv2
import numpy as np
from lane_detector import LaneTracker


def run_live_tracker(
    video_path="2026-10-14 14-08-25.mp4",
    window_name="Live Dashcam Lane Detection (Press Q to quit)",
    show_pip=True,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame_time = 1.0 / orig_fps

    # Create resizable window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    tracker = LaneTracker(img_size=(1280, 720), smoothing_factor=0.20)

    paused = False
    pip_enabled = show_pip
    fps_smooth = orig_fps

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("[*] LIVE LANE DETECTION RUNNING")
    print(f"File: {video_path} ({total_frames} frames @ {orig_fps:.1f} FPS)")
    print("Controls:")
    print("  [SPACE] / [P] : Pause / Resume")
    print("  [Q] / [ESC]   : Quit")
    print("  [R]           : Restart")
    print("  [D]           : Toggle Top-Down View Inset")
    print("  [RIGHT/LEFT]  : Step 30 frames forward/back")
    print("=" * 60)

    prev_time = time.time()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Loop video or hold last frame
                print("End of video reached. Restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            t_start = time.time()

            # Process frame through advanced lane detector
            warped = tracker.warp_perspective(frame)
            binary = tracker.threshold_frame(warped)
            left_fitx, right_fitx, ploty, _ = tracker.fit_polynomial(binary)

            if left_fitx is not None and right_fitx is not None:
                curverad, offset_m = tracker.calculate_metrics(ploty, left_fitx, right_fitx)
                output = tracker.draw_lane_overlay(frame, left_fitx, right_fitx, ploty, curverad, offset_m)
            else:
                output = frame.copy()
                curverad, offset_m = 0, 0

            # Calculate FPS
            proc_time = time.time() - t_start
            current_fps = 1.0 / max(proc_time, 1e-4)
            fps_smooth = 0.9 * fps_smooth + 0.1 * current_fps

            # Telemetry display
            curr_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            progress_pct = (curr_frame / max(total_frames, 1)) * 100
            time_curr = curr_frame / orig_fps
            time_total = total_frames / orig_fps

            # Additional status strip at top right or bottom
            info_str = f"FPS: {fps_smooth:.1f} | Frame: {curr_frame}/{total_frames} ({progress_pct:.1f}%) | Time: {time_curr:.1f}s / {time_total:.1f}s"
            cv2.rectangle(output, (20, 680), (1260, 710), (15, 15, 15), -1)
            cv2.putText(output, info_str, (30, 702), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(
                output,
                "[Q] Quit  [SPACE] Pause  [D] Top-Down Inset  [R] Restart",
                (700, 702),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 255),
                1,
            )

            # Optional Picture-in-Picture (PiP) inset of bird's-eye view with binary mask
            if pip_enabled:
                pip_h, pip_w = 200, 320
                pip_x, pip_y = 1280 - pip_w - 25, 25

                # Colored bird's-eye representation
                binary_color = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
                if left_fitx is not None and right_fitx is not None:
                    # Draw fitted curves in inset
                    pts_l = np.transpose(np.vstack([left_fitx, ploty])).astype(np.int32)
                    pts_r = np.transpose(np.vstack([right_fitx, ploty])).astype(np.int32)
                    cv2.polylines(binary_color, [pts_l], False, (0, 0, 255), 10)
                    cv2.polylines(binary_color, [pts_r], False, (255, 0, 0), 10)

                pip_resized = cv2.resize(binary_color, (pip_w, pip_h))
                cv2.rectangle(output, (pip_x - 3, pip_y - 3), (pip_x + pip_w + 3, pip_y + pip_h + 3), (0, 220, 255), 2)
                output[pip_y : pip_y + pip_h, pip_x : pip_x + pip_w] = pip_resized
                cv2.putText(output, "Bird's-Eye Binary Inset", (pip_x + 10, pip_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

            display_frame = output

        # Show live frame
        cv2.imshow(window_name, display_frame)

        # Frame pacing
        wait_ms = 1 if not paused else 50
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in [ord("q"), 27]:  # 'q' or Esc
            print("Exiting live detection.")
            break
        elif key in [ord(" "), ord("p")]:  # Space or 'p'
            paused = not paused
            status_text = "PAUSED" if paused else "RESUMED"
            print(f"Playback {status_text}")
        elif key == ord("r"):  # Restart
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print("Restarted from frame 0.")
        elif key == ord("d"):  # Toggle PiP inset
            pip_enabled = not pip_enabled
        elif key in [83, ord("c")]:  # Right arrow
            curr = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(curr + 30, total_frames - 1))
        elif key in [81, ord("x")]:  # Left arrow
            curr = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(curr - 30, 0))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_arg = sys.argv[1] if len(sys.argv) > 1 else "2026-10-14 14-08-25.mp4"
    run_live_tracker(video_arg)
