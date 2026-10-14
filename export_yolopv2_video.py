"""
Export Dashcam Video with YOLOPv2 Panoptic Perception and 2D Aerial Live Map.
Creates a high-definition Dual-View side-by-side MP4 video (1640x720).
"""

import sys
import time
import os
import cv2
import numpy as np
from yolop_detector import YOLOPDetector


def export_dualview_video(
    input_video="2026-10-14 14-08-25.mp4",
    output_video="output_yolopv2_preview.mp4",
    model_path="yolopv2.onnx",
    max_frames=150,  # Default 5-second sample clip (150 frames @ 30 FPS)
    show_boxes=True
):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not os.path.exists(input_video):
        print(f"Error: Video file '{input_video}' not found.")
        return

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

    print("=" * 65)
    print(f"[*] EXPORTING DUAL-VIEW VIDEO WITH YOLOPv2...")
    print(f"Input:   {input_video} ({total_frames} frames @ {fps:.1f} FPS)")
    print(f"Output:  {output_video}")
    print(f"Frames:  {frames_to_process} ({frames_to_process / fps:.1f} seconds)")
    print("=" * 65)

    detector = YOLOPDetector(model_path=model_path, conf_thres=0.30)
    print(f"[+] Loaded Model: {detector.version_str} ({os.path.basename(model_path)})")
    print(f"[+] Active Providers: {detector.active_providers}")

    # Output dimensions: 1280 (Camera) + 360 (BEV Radar) = 1640 width x 720 height
    out_w, out_h = 1640, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (out_w, out_h))

    t0 = time.time()
    for i in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break

        cam_view, _, _, _, bev_map = detector.process_frame(
            frame, show_boxes=show_boxes, generate_bev=True
        )

        # Composite side-by-side Dual-View: [Camera (1280x720) | BEV Map (360x720)]
        dual_canvas = np.hstack([cam_view, bev_map])
        out.write(dual_canvas)

        if (i + 1) % 30 == 0 or (i + 1) == frames_to_process:
            elapsed = time.time() - t0
            current_fps = (i + 1) / elapsed
            percent = (i + 1) / frames_to_process * 100
            print(f"  Frame {i + 1}/{frames_to_process} [{percent:5.1f}%] - {current_fps:.1f} FPS (Elapsed: {elapsed:.1f}s)")

    cap.release()
    out.release()

    total_time = time.time() - t0
    print("=" * 65)
    print(f"[DONE] Successfully generated {output_video} in {total_time:.1f}s ({frames_to_process / total_time:.1f} FPS)")
    print(f"File size: {os.path.getsize(output_video) / (1024 * 1024):.2f} MB")
    print("=" * 65)


if __name__ == "__main__":
    export_dualview_video(
        input_video="2026-10-14 14-08-25.mp4",
        output_video="output_yolopv2_preview.mp4",
        max_frames=150
    )

