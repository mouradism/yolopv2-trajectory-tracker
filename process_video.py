"""
Batch Video Processing Script
Processes the dashcam video through the Hough Transform Lane Detection pipeline
and writes out the annotated output video.
"""

import sys
import time
from lane_detector import process_video_file


def main():
    input_video = sys.argv[1] if len(sys.argv) > 1 else "2026-10-14 14-08-25.mp4"
    output_video = sys.argv[2] if len(sys.argv) > 2 else "output_lane_detected.mp4"

    t0 = time.time()
    print("=" * 60)
    print("      LANE DETECTION VIDEO PROCESSING")
    print(f"Input:  {input_video}")
    print(f"Output: {output_video}")
    print("=" * 60)

    process_video_file(input_video, output_video)

    elapsed = time.time() - t0
    print(f"\n[DONE] Processing complete in {elapsed:.2f} seconds!")
    print(f"Result saved to: {output_video}")


if __name__ == "__main__":
    main()

