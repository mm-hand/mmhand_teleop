import types
import unittest
from unittest.mock import patch

import numpy as np

from input import create_source
from input.frame import InitialJointAngles
from input.mediapipe.source import MediaPipeSource
from one_euro import OneEuro


def standard_hand():
    points = np.zeros((21, 3), float)
    points[1:5] = ((0.02, 0.01, 0.01), (0.035, 0.02, 0.015),
                   (0.045, 0.03, 0.02), (0.055, 0.04, 0.025))
    for start, x in zip((5, 9, 13, 17), (0.03, 0.01, -0.01, -0.03)):
        points[start:start + 4] = ((x, 0.05, 0), (x, 0.075, 0),
                                   (x, 0.095, 0), (x, 0.11, 0))
    return points


class FakeCamera:
    def __init__(self):
        self.frame = np.zeros((48, 64, 3), np.uint8)
        self.sequence = 0
        self.closed = False

    def read(self, after):
        self.sequence = after + 1
        return self.sequence, self.frame.copy(), 1.0 + 0.033 * self.sequence

    def close(self):
        self.closed = True


class FakeLandmarker:
    def __init__(self, world=None, label="Right"):
        self.world, self.label = world, label
        self.closed = False

    def detect_for_video(self, _image, _timestamp):
        if self.world is None:
            return types.SimpleNamespace(
                hand_landmarks=[], hand_world_landmarks=[], handedness=[],
            )
        def landmark(row):
            return types.SimpleNamespace(x=row[0], y=row[1], z=row[2])
        return types.SimpleNamespace(
            hand_landmarks=[[landmark(row) for row in self.world]],
            hand_world_landmarks=[[landmark(row) for row in self.world]],
            handedness=[[types.SimpleNamespace(category_name=self.label)]],
        )

    def close(self):
        self.closed = True


def make_source(**landmarker):
    source = MediaPipeSource.__new__(MediaPipeSource)
    source.camera = FakeCamera()
    source.landmarker = FakeLandmarker(**landmarker)
    source.mp = types.SimpleNamespace(
        Image=lambda **options: options["data"],
        ImageFormat=types.SimpleNamespace(SRGB=1),
    )
    source.point_filter = OneEuro(0.5, 1.0, 1.0)
    source.handedness = None
    source.sequence = 0
    return source


class MediaPipeTests(unittest.TestCase):
    def test_create_source_branch(self):
        selected = object()
        with patch("input.INPUT_SOURCE", "mediapipe"), patch(
            "input.mediapipe.source.MediaPipeSource", return_value=selected
        ):
            self.assertIs(create_source(), selected)

    def test_read_returns_points_without_pad_directions(self):
        world = standard_hand()
        frame = make_source(world=world, label="Left").read()
        self.assertTrue(frame.ready)
        self.assertEqual(frame.handedness, "Left")
        self.assertEqual(frame.points.shape, (21, 3))
        self.assertIsNone(frame.finger_pad_directions)
        self.assertIsInstance(frame.initial_joint_angles, InitialJointAngles)
        np.testing.assert_allclose(frame.points, world, atol=1e-12)
        self.assertAlmostEqual(
            frame.raw_palm_length, np.linalg.norm(world[12] - world[0])
        )
        self.assertAlmostEqual(
            frame.raw_palm_width, np.linalg.norm(world[5] - world[17])
        )
        self.assertEqual(frame.preview.shape, (360, 1280, 3))
        self.assertIn("MEDIAPIPE Left", frame.status)

    def test_right_hand_is_reported_as_right(self):
        frame = make_source(world=standard_hand(), label="Right").read()
        self.assertEqual(frame.handedness, "Right")

    def test_position_scale_is_applied(self):
        world = standard_hand()
        source = make_source(world=world)
        with patch("input.mediapipe.source.MEDIAPIPE_POSITION_SCALE_TO_M", 2.0):
            frame = source.read()
        np.testing.assert_allclose(frame.points, 2 * world, atol=1e-12)
        self.assertAlmostEqual(
            frame.raw_palm_width, 2 * np.linalg.norm(world[5] - world[17])
        )

    def test_no_hand_returns_empty_frame(self):
        frame = make_source().read()
        self.assertIsNone(frame.points)
        self.assertFalse(frame.ready)
        self.assertIn("WAITING", frame.status)
        self.assertEqual(frame.preview.shape, (360, 1280, 3))

    def test_degenerate_landmarks_return_invalid_frame(self):
        frame = make_source(world=np.zeros((21, 3))).read()
        self.assertIsNone(frame.points)
        self.assertIn("INVALID", frame.status)

    def test_missing_model_fails_fast(self):
        with patch(
            "input.mediapipe.source.MEDIAPIPE_MODEL_PATH",
            "missing/hand_landmarker.task",
        ), self.assertRaises(FileNotFoundError):
            MediaPipeSource()


if __name__ == "__main__":
    unittest.main()
