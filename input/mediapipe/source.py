"""MediaPipe Hand Landmarker input.

MediaPipe outputs Standard21 world landmarks and handedness, but no fingertip
pointing, so this source deliberately leaves ``finger_pad_directions`` None.
"""

from pathlib import Path

import cv2
import numpy as np

from config import (
    CAMERA_DEVICE,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    MEDIAPIPE_MODEL_PATH,
    MEDIAPIPE_POINT_FILTER,
    MEDIAPIPE_POSITION_SCALE_TO_M,
)
from input.camera import OpenCVCamera
from input.frame import (
    SKELETON_EDGES,
    InputFrame,
    hand0_middle_tip_distance,
    initial_joint_angles_from_points,
)
from one_euro import OneEuro


def _points(landmarks):
    return np.asarray([(p.x, p.y, p.z) for p in landmarks], float)


def _preview(frame, landmarks=None):
    """Draw the detected skeleton on the shared dashboard canvas."""
    frame = frame.copy()
    if landmarks is not None:
        height, width = frame.shape[:2]
        pixels = [
            (int(p.x * width), int(p.y * height)) for p in landmarks
        ]
        for start, end in SKELETON_EDGES:
            cv2.line(
                frame, pixels[start], pixels[end], (80, 230, 255), 2,
                cv2.LINE_AA,
            )
        for pixel in pixels:
            cv2.circle(frame, pixel, 3, (255, 150, 70), -1)
    canvas = np.zeros((360, 1280, 3), np.uint8)
    canvas[:, 400:880] = cv2.resize(frame, (480, 360))
    return canvas


class MediaPipeSource:
    def __init__(self):
        path = Path(MEDIAPIPE_MODEL_PATH)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing MediaPipe hand model: {path} (see README.md)"
            )
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                f"MediaPipe is unavailable ({error}); see README.md"
            ) from error
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
        )
        self.mp = mp
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.camera = OpenCVCamera(
            CAMERA_DEVICE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
        )
        self.point_filter = OneEuro(*MEDIAPIPE_POINT_FILTER)
        self.handedness = None
        self.sequence = 0
        print(f"Input: MediaPipe; {self.camera.description}")

    def _image(self, frame):
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

    def read(self):
        self.sequence, frame, timestamp = self.camera.read(self.sequence)
        result = self.landmarker.detect_for_video(
            self._image(frame), int(timestamp * 1000)
        )
        if not result.hand_world_landmarks:
            self.point_filter.reset()
            self.handedness = None
            return InputFrame.empty(
                timestamp, "MEDIAPIPE WAITING", _preview(frame)
            )
        landmarks = result.hand_world_landmarks[0]
        handedness = result.handedness[0][0].category_name
        if handedness != self.handedness:
            self.point_filter.reset()
            self.handedness = handedness
        try:
            raw = _points(landmarks) * MEDIAPIPE_POSITION_SCALE_TO_M
            raw_palm_length = hand0_middle_tip_distance(raw)
            raw_palm_width = float(np.linalg.norm(raw[5] - raw[17]))
            points = self.point_filter(raw, timestamp)
            initial_angles = initial_joint_angles_from_points(points)
        except ValueError as error:
            self.point_filter.reset()
            return InputFrame.empty(
                timestamp, f"MEDIAPIPE INVALID · {error}",
                _preview(frame, result.hand_landmarks[0]),
            )
        return InputFrame(
            timestamp, points, handedness, True, f"MEDIAPIPE {handedness}",
            preview=_preview(frame, result.hand_landmarks[0]),
            initial_joint_angles=initial_angles,
            raw_palm_length=raw_palm_length,
            raw_palm_width=raw_palm_width,
        )

    def close(self):
        self.landmarker.close()
        self.camera.close()
