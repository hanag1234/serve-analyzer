"""
Core pose analysis engine for tennis serve video.

Pipeline:
1. Read video frame by frame
2. Run MediaPipe PoseLandmarker on each frame to get 33 body landmarks
3. Track key joint trajectories across the whole serve motion
4. Detect key phases of the serve (loading/trophy, racket drop, contact, follow-through)
5. Compute biomechanical metrics at each key phase
6. Return structured data + annotated frames for the overlay

NOTE ON MODEL FILE:
This uses MediaPipe's current Tasks API (PoseLandmarker), which requires a
small .task model file downloaded once. Run scripts/download_model.sh (or the
one-liner in the README) before first use:

    curl -L -o pose_landmarker.task \\
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

The lite model is fast and good enough for this use case. There's also a
"full" and "heavy" variant if you want more accuracy at the cost of speed.
"""

import os
import subprocess
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat


MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")

# MediaPipe Pose landmark indices we care about for serve analysis
# (same indexing as the classic BlazePose 33-point topology)
LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}

# Bone connections for drawing the skeleton overlay
POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


@dataclass
class FrameData:
    frame_idx: int
    timestamp: float
    landmarks: Optional[dict] = None  # name -> (x, y, z, visibility) in PIXEL coords


@dataclass
class ServeAnalysisResult:
    frames: list = field(default_factory=list)
    fps: float = 30.0
    width: int = 0
    height: int = 0
    contact_frame_idx: Optional[int] = None
    loading_frame_idx: Optional[int] = None
    racket_drop_frame_idx: Optional[int] = None
    metrics: dict = field(default_factory=dict)
    dominant_side: str = "right"


def _get_landmarker():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Pose model not found at {MODEL_PATH}.\n"
            "Download it once with:\n"
            "  curl -L -o backend/pose_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        )
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def _landmarks_to_dict(pose_landmarks_list, width, height):
    """pose_landmarks_list is result.pose_landmarks[0] -- a list of NormalizedLandmark."""
    out = {}
    for name, idx in LANDMARKS.items():
        lm = pose_landmarks_list[idx]
        out[name] = (lm.x * width, lm.y * height, lm.z, getattr(lm, "visibility", 1.0))
    return out


def _detect_dominant_side(frames: list[FrameData]) -> str:
    left_ys, right_ys = [], []
    for f in frames:
        if not f.landmarks:
            continue
        left_ys.append(f.landmarks["left_wrist"][1])
        right_ys.append(f.landmarks["right_wrist"][1])
    left_range = (max(left_ys) - min(left_ys)) if left_ys else 0.0
    right_range = (max(right_ys) - min(right_ys)) if right_ys else 0.0
    return "right" if right_range >= left_range else "left"


def _wrist_height_series(frames: list[FrameData], side: str):
    key = f"{side}_wrist"
    ys = []
    for f in frames:
        ys.append(f.landmarks[key][1] if f.landmarks else np.nan)
    return np.array(ys)


def _knee_angle(hip, knee, ankle):
    a = np.array(hip[:2]) - np.array(knee[:2])
    b = np.array(ankle[:2]) - np.array(knee[:2])
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _elbow_angle(shoulder, elbow, wrist):
    a = np.array(shoulder[:2]) - np.array(elbow[:2])
    b = np.array(wrist[:2]) - np.array(elbow[:2])
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _shoulder_hip_separation(landmarks):
    ls = np.array(landmarks["left_shoulder"][:2])
    rs = np.array(landmarks["right_shoulder"][:2])
    lh = np.array(landmarks["left_hip"][:2])
    rh = np.array(landmarks["right_hip"][:2])
    shoulder_vec, hip_vec = rs - ls, rh - lh
    cos_angle = np.dot(shoulder_vec, hip_vec) / (
        np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec) + 1e-8
    )
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _draw_skeleton(frame, landmarks, color=(0, 255, 0)):
    for name, (x, y, z, vis) in landmarks.items():
        if vis is not None and vis < 0.3:
            continue
        cv2.circle(frame, (int(x), int(y)), 5, color, -1)
    for a, b in POSE_CONNECTIONS:
        if a in landmarks and b in landmarks:
            xa, ya = int(landmarks[a][0]), int(landmarks[a][1])
            xb, yb = int(landmarks[b][0]), int(landmarks[b][1])
            cv2.line(frame, (xa, ya), (xb, yb), color, 2)


def _reencode_to_h264(input_path: str, output_path: str):
    """
    Re-encodes a video to H.264/AAC using the ffmpeg CLI so it's playable in
    standard browsers. OpenCV's own video writer often produces MP4 files
    using codecs (like mp4v / MPEG-4 Part 2) that browsers refuse to decode,
    even though the file extension is .mp4.

    If ffmpeg isn't installed on the system, falls back to just using the
    raw OpenCV output directly (renamed into place) so the pipeline doesn't
    crash -- but the video may not play in-browser in that case.
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",   # ensures broad compatibility (Safari in particular)
                "-movflags", "+faststart",  # lets browsers start playback before full download
                output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(input_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg missing or failed -- fall back to the raw OpenCV file so the
        # request still completes. Video may not play in every browser.
        if os.path.exists(input_path):
            os.replace(input_path, output_path)


def analyze_video(video_path: str, output_annotated_path: str) -> ServeAnalysisResult:
    """
    Main entry point: reads the video, runs pose estimation on every frame,
    detects serve phases, computes metrics, and writes an annotated video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    result = ServeAnalysisResult(fps=fps, width=width, height=height)
    raw_frames = []  # kept in memory for the annotation pass; fine for short serve clips

    landmarker = _get_landmarker()
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((frame_idx / fps) * 1000)

        pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks_dict = None
        if pose_result.pose_landmarks:
            landmarks_dict = _landmarks_to_dict(pose_result.pose_landmarks[0], width, height)

        result.frames.append(FrameData(
            frame_idx=frame_idx,
            timestamp=frame_idx / fps,
            landmarks=landmarks_dict,
        ))
        raw_frames.append(frame)
        frame_idx += 1

    cap.release()
    landmarker.close()

    # ---- Dominant (hitting) arm ----
    result.dominant_side = _detect_dominant_side(result.frames)
    side = result.dominant_side

    # ---- Phase detection ----
    wrist_y = _wrist_height_series(result.frames, side)
    valid_idx = np.where(~np.isnan(wrist_y))[0]

    if len(valid_idx) > 5:
        search_start = valid_idx[len(valid_idx) // 4]
        search_region = [i for i in valid_idx if i >= search_start]
        contact_idx = min(search_region, key=lambda i: wrist_y[i])
        result.contact_frame_idx = int(contact_idx)

        pre_contact = [i for i in valid_idx if i < contact_idx]
        if pre_contact:
            loading_idx = max(pre_contact, key=lambda i: wrist_y[i])
            result.loading_frame_idx = int(loading_idx)
            result.racket_drop_frame_idx = int((loading_idx + contact_idx) // 2)

    # ---- Metric computation ----
    metrics = {}

    def get_landmarks_at(idx):
        if idx is None or idx >= len(result.frames):
            return None
        return result.frames[idx].landmarks

    contact_lm = get_landmarks_at(result.contact_frame_idx)
    loading_lm = get_landmarks_at(result.loading_frame_idx)

    if contact_lm:
        shoulder = contact_lm[f"{side}_shoulder"]
        elbow = contact_lm[f"{side}_elbow"]
        wrist = contact_lm[f"{side}_wrist"]
        metrics["elbow_angle_at_contact_deg"] = round(_elbow_angle(shoulder, elbow, wrist), 1)

        knee = contact_lm[f"{side}_knee"]
        hip = contact_lm[f"{side}_hip"]
        ankle = contact_lm[f"{side}_ankle"]
        metrics["knee_angle_at_contact_deg"] = round(_knee_angle(hip, knee, ankle), 1)

        torso_len = abs(shoulder[1] - hip[1]) + 1e-6
        metrics["contact_height_rel_to_shoulder"] = round((shoulder[1] - wrist[1]) / torso_len, 2)
        metrics["shoulder_hip_separation_at_contact_deg"] = round(
            _shoulder_hip_separation(contact_lm), 1
        )

    if loading_lm:
        knee = loading_lm[f"{side}_knee"]
        hip = loading_lm[f"{side}_hip"]
        ankle = loading_lm[f"{side}_ankle"]
        metrics["knee_bend_at_loading_deg"] = round(_knee_angle(hip, knee, ankle), 1)

    toss_side = "left" if side == "right" else "right"
    toss_y = _wrist_height_series(result.frames, toss_side)
    if result.contact_frame_idx is not None and contact_lm:
        pre_contact_window = toss_y[: result.contact_frame_idx + 1]
        valid_toss = pre_contact_window[~np.isnan(pre_contact_window)]
        if len(valid_toss) > 0:
            toss_peak_y = float(np.min(valid_toss))
            contact_wrist_y = contact_lm[f"{side}_wrist"][1]
            torso_len = abs(contact_lm[f"{side}_shoulder"][1] - contact_lm[f"{side}_hip"][1]) + 1e-6
            metrics["toss_peak_vs_contact_height_rel"] = round(
                (contact_wrist_y - toss_peak_y) / torso_len, 2
            )

    result.metrics = metrics

    # ---- Render annotated video ----
    key_frames = {
        result.loading_frame_idx: ("LOADING", (255, 200, 0)),
        result.racket_drop_frame_idx: ("RACKET DROP", (0, 165, 255)),
        result.contact_frame_idx: ("CONTACT", (0, 0, 255)),
    }
    idx_to_label = {k: v for k, v in key_frames.items() if k is not None}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_path = output_annotated_path + ".raw.mp4"
    writer = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))

    for i, frame in enumerate(raw_frames):
        landmarks = result.frames[i].landmarks
        if landmarks:
            _draw_skeleton(frame, landmarks)
        if i in idx_to_label:
            label, color = idx_to_label[i]
            cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 0), -1)
            cv2.putText(frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
        writer.write(frame)

    writer.release()

    # OpenCV's mp4v codec produces files most browsers can't play. Re-encode
    # to H.264 with ffmpeg (the actual CLI binary, not via OpenCV) so the
    # video plays correctly in the frontend. Falls back to the raw file if
    # ffmpeg isn't installed, so the pipeline still completes either way.
    _reencode_to_h264(raw_path, output_annotated_path)

    return result

