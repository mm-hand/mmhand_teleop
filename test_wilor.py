import types
import unittest
from unittest.mock import patch

import numpy as np

from input import create_source
from input.frame import initial_joint_angles_from_points
from retarget import human_vector_features
from one_euro import OneEuro
from track import _parse_args
from input.wilor.source import (
    Detection,
    Detector,
    JOINT_ORDER,
    PAD_FACE_ROWS,
    Reconstructor,
    TIP_VERTICES,
    WilorSource,
    draw_mesh,
    mesh_hand,
    rotate,
)


def standard_hand():
    points = np.zeros((21, 3), np.float32)
    points[1:5] = ((0.02, 0.01, 0.01), (0.035, 0.02, 0.015),
                   (0.045, 0.03, 0.02), (0.055, 0.04, 0.025))
    for start, x in zip((5, 9, 13, 17), (0.03, 0.01, -0.01, -0.03)):
        points[start:start + 4] = ((x, 0.05, 0), (x, 0.075, 0),
                                   (x, 0.095, 0), (x, 0.11, 0))
    return points


def synthetic_mesh():
    vertices = np.zeros((778, 3), np.float32)
    regressor = np.zeros((16, 778), np.float32)
    regressor[np.arange(16), np.arange(16)] = 1
    raw = np.zeros((21, 3), np.float32)
    raw[JOINT_ORDER] = standard_hand()
    vertices[:16] = raw[:16]
    vertices[TIP_VERTICES] = raw[16:]
    faces = np.zeros((max(map(np.max, PAD_FACE_ROWS)) + 1, 3), np.int32)
    vertex = 100
    for finger, rows in enumerate(PAD_FACE_ROWS):
        for row in rows:
            faces[row] = (vertex, vertex + 1, vertex + 2)
            origin = np.array((finger * 0.01, 0, 0), np.float32)
            vertices[vertex:vertex + 3] = origin + ((0, 0, 0), (0.004, 0, 0),
                                                    (0, 0, 0.004))
            vertex += 3
    return vertices, regressor, faces


class FakeDetectorSession:
    def __init__(self, output):
        self.output = output

    def get_inputs(self):
        return [types.SimpleNamespace(type="tensor(float16)")]

    def run(self, *_):
        return [self.output]


class FakeReconstructorSession:
    def __init__(self, camera):
        self.camera = np.asarray(camera, np.float32)[None]
        self.input = None

    def get_inputs(self):
        return [types.SimpleNamespace(type="tensor(float16)")]

    def run(self, _, inputs):
        self.input = inputs["images"]
        return [np.zeros((1, 778, 3), np.float16), self.camera.copy()]


class WilorTests(unittest.TestCase):
    def test_cli_and_configured_source(self):
        args = _parse_args([])
        self.assertFalse(args.ros)
        self.assertFalse(args.sim)
        self.assertTrue(_parse_args(["--ros"]).ros)
        self.assertTrue(_parse_args(["--sim"]).sim)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            _parse_args(["--source", "wilor"])
        selected = object()
        with patch("input.INPUT_SOURCE", "wilor"), patch(
            "input.wilor.source.WilorSource", return_value=selected
        ):
            self.assertIs(create_source(), selected)

    def test_cmc_features_are_rigid_transform_invariant_but_preserve_scale(self):
        points = standard_hand()
        directions = np.tile((0.0, 0.0, -1.0), (5, 1))
        angle = 0.7
        rotation = np.array(((np.cos(angle), -np.sin(angle), 0),
                             (np.sin(angle), np.cos(angle), 0),
                             (0, 0, 1)))
        expected = human_vector_features(points, directions)
        actual = human_vector_features(
            points @ rotation.T + (2, 3, 4), directions @ rotation.T
        )
        for transformed, original in zip(actual[2:], expected[2:]):
            np.testing.assert_allclose(transformed, original, atol=1e-7)
        scaled = human_vector_features(points * 2, directions)
        np.testing.assert_allclose(scaled[2], expected[2] * 2)
        np.testing.assert_allclose(scaled[3], expected[3] * 2)
        np.testing.assert_allclose(scaled[4], expected[4])

    def test_mesh_to_standard21_and_pad_directions(self):
        vertices, regressor, faces = synthetic_mesh()
        with patch("input.wilor.source.WILOR_THUMB_PAD_ROTATION_DEG", 0.0):
            right_points, raw_right, _, right_raw_x, right_raw_width = mesh_hand(
                vertices, 1, regressor, faces
            )
            left_points, raw_left, _, left_raw_x, left_raw_width = mesh_hand(
                vertices, 0, regressor, faces
            )
        points, directions, handedness, raw_x, raw_width = mesh_hand(
            vertices, 1, regressor, faces
        )
        self.assertEqual(handedness, "Right")
        self.assertTrue(np.isfinite(raw_x))
        self.assertGreater(raw_width, 0.0)
        self.assertAlmostEqual(right_raw_x, left_raw_x)
        self.assertAlmostEqual(right_raw_width, left_raw_width)
        np.testing.assert_allclose(right_points, standard_hand(), atol=1e-8)
        np.testing.assert_allclose(
            left_points, standard_hand() * (-1, 1, 1), atol=1e-8
        )
        scaled_points = mesh_hand(vertices * 2, 1, regressor, faces)[0]
        np.testing.assert_allclose(scaled_points, right_points * 2, atol=1e-8)
        self.assertEqual(points.shape, (21, 3))
        self.assertEqual(directions.shape, (5, 3))
        self.assertTrue(np.isfinite(points).all())
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-7)

        left_points, left_directions, handedness, _, _ = mesh_hand(
            vertices, 0, regressor, faces
        )
        self.assertEqual(handedness, "Left")
        self.assertTrue(np.isfinite(left_points).all())
        np.testing.assert_allclose(np.linalg.norm(left_directions, axis=1), 1, atol=1e-7)
        np.testing.assert_allclose(
            directions[0], rotate(raw_right[0], right_points[4] - right_points[3], -np.pi / 4)
        )
        np.testing.assert_allclose(
            left_directions[0], rotate(raw_left[0], left_points[4] - left_points[3], np.pi / 4)
        )
        angles = initial_joint_angles_from_points(points)
        thumb = np.diff(points[1:5], axis=0)
        thumb /= np.linalg.norm(thumb, axis=1, keepdims=True)
        expected_thumb = np.arccos(np.clip((
            thumb[0] @ thumb[1], thumb[1] @ thumb[2]
        ), -1.0, 1.0))
        np.testing.assert_allclose(angles.thumb_bends, expected_thumb, atol=1e-9)
        self.assertEqual(angles.four_fingers.shape, (4, 4))

    def test_detector_threshold_coordinates_and_handedness(self):
        raw = np.array([[[320, 100], [320, 100], [100, 20], [100, 20],
                         [0.1, 0.2], [0.9, 0.1]]], np.float16)
        detector = Detector(FakeDetectorSession(raw), 0.3, 0.5)
        detection = detector(np.zeros((480, 640, 3), np.uint8))
        np.testing.assert_allclose(detection.box, (270, 190, 370, 290), atol=1)
        self.assertEqual(detection.handedness, 1)
        self.assertAlmostEqual(detection.score, 0.9, places=2)

    def test_point_and_direction_filters_reset_on_hand_switch(self):
        source = object.__new__(WilorSource)
        source.point_filter = OneEuro(0.5, 1.0, 1.0)
        source.direction_filter = OneEuro(0.5, 0.25, 1.0)
        source.handedness = None
        points = np.zeros((21, 3))
        directions = np.tile((0.0, 0.0, 1.0), (5, 1))
        first, vectors = source._filter(points, directions, "Left", 1.0)
        moved, _ = source._filter(points + 1, directions, "Left", 1.1)
        switched, _ = source._filter(points + 2, directions, "Right", 1.2)
        self.assertTrue(np.all(moved > first))
        np.testing.assert_array_equal(switched, points + 2)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)

    def test_reconstruction_camera_translation_and_mesh_overlay(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        box = np.array((270, 190, 370, 290), np.float32)
        for right, expected_x in ((1, 0.25), (0, -0.25)):
            session = FakeReconstructorSession((2, 0.25, -0.1))
            vertices, translation = Reconstructor(session, 2)(
                frame, Detection(box.copy(), right, 0.9)
            )
            self.assertEqual(vertices.shape, (778, 3))
            self.assertEqual(session.input.shape, (1, 3, 256, 192))
            np.testing.assert_allclose(
                translation, (expected_x, -0.1, 46.875), atol=1e-5
            )
        with self.assertRaisesRegex(ValueError, "camera scale"):
            Reconstructor(FakeReconstructorSession((-1, 0, 0)), 2)(
                frame, Detection(box.copy(), 1, 0.9)
            )

        vertices = np.array(((-0.01, -0.01, 0), (0.01, -0.01, 0),
                             (0, 0.01, 0)), np.float32)
        faces = np.array(((0, 1, 2),), np.int32)
        image = frame.copy()
        draw_mesh(image, vertices, (0, 0, 1), 1, faces)
        self.assertGreater(image[240, 320, 1], image[240, 320, 0])
        self.assertGreater(image[240, 320, 2], image[240, 320, 0])

if __name__ == "__main__":
    unittest.main()
