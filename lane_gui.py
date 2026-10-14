"""
Lane Line Detection Project GUI (Tkinter + OpenCV + PIL)
Based on the GUI architecture in mission.md:
- Dual-panel display: Original input video on the left, Lane-detected video on the right
- Real-time frame processing or dual video playback
- Interactive controls: Play/Pause, Restart, Video File Selector, Export Full Video, and Quit.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from lane_detector import LaneDetector, process_video_file


class LaneDetectionGUI:
    def __init__(self, root, default_video="2026-10-14 14-08-25.mp4"):
        self.root = root
        self.root.title("Road Lane-Line Detection - OpenCV & Python")
        self.root.geometry("1100x750+80+30")
        self.root.configure(bg="#1e1e24")

        self.video_path = default_video if os.path.exists(default_video) else ""
        self.cap = None
        self.detector = LaneDetector(alpha=0.25, draw_corridor=True)

        self.is_playing = False
        self.is_processing_export = False
        self.display_width = 520
        self.display_height = 390

        self._build_ui()

        if self.video_path:
            self._load_video(self.video_path)

    def _build_ui(self):
        # Header banner
        header_frame = tk.Frame(self.root, bg="#2b2d42", pady=12)
        header_frame.pack(fill=tk.X)

        title_lbl = tk.Label(
            header_frame,
            text="Autonomous Driving: Road Lane-Line Detection System",
            font=("Helvetica", 16, "bold"),
            fg="#edf2f4",
            bg="#2b2d42",
        )
        title_lbl.pack()

        subtitle_lbl = tk.Label(
            header_frame,
            text="Computer Vision with Color Masking, Canny Edges, Hough Transform & Temporal Smoothing",
            font=("Helvetica", 10),
            fg="#8d99ae",
            bg="#2b2d42",
        )
        subtitle_lbl.pack(pady=2)

        # Video preview container
        preview_container = tk.Frame(self.root, bg="#1e1e24", pady=15)
        preview_container.pack(fill=tk.BOTH, expand=True)

        # Left Column: Original Video
        left_box = tk.LabelFrame(
            preview_container,
            text=" Original Dashcam Video ",
            font=("Helvetica", 11, "bold"),
            fg="#00d2d3",
            bg="#22222b",
            padx=8,
            pady=8,
        )
        left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(15, 8))

        self.lbl_orig = tk.Label(left_box, bg="#000000")
        self.lbl_orig.pack(fill=tk.BOTH, expand=True)

        # Right Column: Lane Detected Video
        right_box = tk.LabelFrame(
            preview_container,
            text=" Lane Detection & Driving Corridor ",
            font=("Helvetica", 11, "bold"),
            fg="#10ac84",
            bg="#22222b",
            padx=8,
            pady=8,
        )
        right_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 15))

        self.lbl_detected = tk.Label(right_box, bg="#000000")
        self.lbl_detected.pack(fill=tk.BOTH, expand=True)

        # Telemetry / Status bar
        self.telemetry_var = tk.StringVar(value="Status: Ready | Select or play video to start tracking")
        status_bar = tk.Label(
            self.root,
            textvariable=self.telemetry_var,
            font=("Consolas", 10),
            fg="#f1f2f6",
            bg="#2f3542",
            pady=6,
        )
        status_bar.pack(fill=tk.X, padx=15, pady=(0, 10))

        # Progress bar for video export
        self.prog_var = tk.DoubleVar(value=0.0)
        self.prog_bar = ttk.Progressbar(
            self.root,
            variable=self.prog_var,
            maximum=100.0,
            mode="determinate",
        )
        self.prog_bar.pack(fill=tk.X, padx=15, pady=(0, 8))

        # Bottom Control Panel
        ctrl_frame = tk.Frame(self.root, bg="#1e1e24", pady=10)
        ctrl_frame.pack(fill=tk.X)

        btn_style = {
            "font": ("Helvetica", 10, "bold"),
            "padx": 16,
            "pady": 6,
            "bd": 0,
            "cursor": "hand2",
        }

        self.btn_select = tk.Button(
            ctrl_frame,
            text="Open Video",
            command=self._on_select_video,
            bg="#341f97",
            fg="white",
            **btn_style,
        )
        self.btn_select.pack(side=tk.LEFT, padx=(20, 8))

        self.btn_play = tk.Button(
            ctrl_frame,
            text="Play",
            command=self._toggle_play,
            bg="#10ac84",
            fg="white",
            **btn_style,
        )
        self.btn_play.pack(side=tk.LEFT, padx=8)

        self.btn_restart = tk.Button(
            ctrl_frame,
            text="Restart",
            command=self._restart_video,
            bg="#ff9f43",
            fg="white",
            **btn_style,
        )
        self.btn_restart.pack(side=tk.LEFT, padx=8)

        self.btn_export = tk.Button(
            ctrl_frame,
            text="Export Annotated MP4",
            command=self._export_video,
            bg="#ee5253",
            fg="white",
            **btn_style,
        )
        self.btn_export.pack(side=tk.LEFT, padx=8)

        self.btn_quit = tk.Button(
            ctrl_frame,
            text="Quit",
            command=self._on_close,
            bg="#576574",
            fg="white",
            **btn_style,
        )
        self.btn_quit.pack(side=tk.RIGHT, padx=(8, 20))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_video(self, path):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Failed to open video file:\n{path}")
            return
        self.video_path = path
        self.detector.reset()
        self._read_and_show_frame()

    def _on_select_video(self):
        selected = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if selected:
            self.is_playing = False
            self.btn_play.configure(text="Play", bg="#10ac84")
            self._load_video(selected)

    def _toggle_play(self):
        if self.cap is None or not self.cap.isOpened():
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.configure(text="Pause", bg="#ff9f43")
            self._update_loop()
        else:
            self.btn_play.configure(text="Play", bg="#10ac84")

    def _restart_video(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.detector.reset()
            self._read_and_show_frame()

    def _read_and_show_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            # Loop
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.detector.reset()
            ret, frame = self.cap.read()
            if not ret:
                return

        # Process frame
        processed_frame, telemetry = self.detector.process_image(frame)

        # Update telemetry text
        curr_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        offset = telemetry.get("offset_m", 0.0)
        direction = telemetry.get("direction", "CENTERED")
        self.telemetry_var.set(
            f"Frame: {curr_frame}/{total} | Vehicle Offset: {abs(offset):.2f}m {direction} | Lane Width: {telemetry.get('lane_width_px', 0)}px"
        )

        # Render original frame to GUI
        orig_rgb = cv2.cvtColor(cv2.resize(frame, (self.display_width, self.display_height)), cv2.COLOR_BGR2RGB)
        img_orig = ImageTk.PhotoImage(image=Image.fromarray(orig_rgb))
        self.lbl_orig.configure(image=img_orig)
        self.lbl_orig.image = img_orig

        # Render detected frame to GUI
        proc_rgb = cv2.cvtColor(cv2.resize(processed_frame, (self.display_width, self.display_height)), cv2.COLOR_BGR2RGB)
        img_proc = ImageTk.PhotoImage(image=Image.fromarray(proc_rgb))
        self.lbl_detected.configure(image=img_proc)
        self.lbl_detected.image = img_proc

    def _update_loop(self):
        if not self.is_playing:
            return
        self._read_and_show_frame()
        self.root.after(25, self._update_loop)

    def _export_video(self):
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showwarning("Warning", "Please select a valid input video first.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Output Video",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")],
            initialfile="output_lane_detected.mp4",
        )
        if not save_path:
            return

        self.btn_export.configure(state=tk.DISABLED)
        self.is_playing = False
        self.btn_play.configure(text="Play", bg="#10ac84")

        def task():
            try:
                def on_progress(cur, total):
                    pct = (cur / total) * 100
                    self.prog_var.set(pct)
                    self.telemetry_var.set(f"Exporting video: {cur}/{total} frames ({pct:.1f}%)")

                process_video_file(
                    self.video_path,
                    save_path,
                    max_frames=None,
                    progress_callback=on_progress,
                )
                messagebox.showinfo("Export Complete", f"Successfully exported annotated video to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Export Failed", str(e))
            finally:
                self.btn_export.configure(state=tk.NORMAL)
                self.prog_var.set(0.0)

        threading.Thread(target=task, daemon=True).start()

    def _on_close(self):
        self.is_playing = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    video_arg = sys.argv[1] if len(sys.argv) > 1 else "2026-10-14 14-08-25.mp4"
    app = LaneDetectionGUI(root, default_video=video_arg)
    root.mainloop()


if __name__ == "__main__":
    main()

