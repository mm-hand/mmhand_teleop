import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import track

from input import create_source
from input.frame import (
    InitialJointAngles, compute_cmc_frame,
    hand0_middle_tip_distance,
)
from input.manus.adapter import (
    MANUS_TO_STANDARD21,
    MANUS_THUMB_PAD_ROTATION_DEG,
    adapt_raw_skeleton,
    convert_manus25_to_standard21,
)
from input.manus.calibration import ensure_calibration
from input.manus.source import (
    CalibrationStep,
    MANUS_THUMB_DIP_TO_PIP_GAIN,
    MANUS_THUMB_PIP_DIP_SCALE,
    ManusPacket,
    ManusSource,
    _SdkFrame,
    _ergonomics_initial_angles,
    _sdk_frame_to_packet,
)


def standard_hand_world():
    points = np.zeros((21, 3), float)
    points[1:5] = ((0.01, -0.035, 0), (0.025, -0.045, 0), (0.04, -0.05, 0), (0.055, -0.052, 0))
    for mcp, lateral in zip((5, 9, 13, 17), (-0.03, -0.01, 0.01, 0.03)):
        points[mcp : mcp + 4] = [(0.04 + 0.025 * index, lateral, 0.002 * index) for index in range(4)]
    return points


def raw_packet(received_at=1.0):
    positions = np.zeros((25, 3), float)
    positions[MANUS_TO_STANDARD21] = standard_hand_world()
    positions[[5, 10, 15, 20]] = ((0.015, -0.03, 0), (0.02, -0.01, 0), (0.02, 0.01, 0), (0.015, 0.03, 0))
    return ManusPacket(
        positions,
        np.tile((1.0, 0.0, 0.0, 0.0), (25, 1)),
        list(range(25)),
        [],
        "Left",
        received_at,
        glove_id=0x1234,
    )


class FakeTransport:
    def __init__(self, *values):
        self.values = list(values)
        self.closed = False

    def read(self):
        return self.values.pop(0) if self.values else None

    def close(self):
        self.closed = True

    def mark_calibrated(self, path):
        self.calibration_path = path


class FakeCalibrationTransport(FakeTransport):
    def __init__(self, *values):
        super().__init__(*values)
        self.calls = []
        self.payload = b"official-manus-calibration"

    def calibration_steps(self, glove_id):
        self.calls.append(("steps", glove_id))
        return (
            CalibrationStep(3, "Flat", "Keep the hand flat", 1.0),
            CalibrationStep(7, "Fist", "Close the hand", -2.0),
        )

    def calibration_start(self, glove_id):
        self.calls.append(("start", glove_id))

    def calibration_run_step(self, glove_id, index):
        self.calls.append(("step", glove_id, index))

    def calibration_finish(self, glove_id):
        self.calls.append(("finish", glove_id))

    def calibration_stop(self, glove_id):
        self.calls.append(("stop", glove_id))

    def calibration_export(self, glove_id):
        self.calls.append(("export", glove_id))
        return self.payload

    def calibration_import(self, glove_id, payload):
        self.calls.append(("import", glove_id, bytes(payload)))


class ManusPhase1Tests(unittest.TestCase):
    def test_manus_25_to_standard21_mapping(self):
        points = np.repeat(np.arange(25, dtype=float)[:, None], 3, axis=1)
        output = convert_manus25_to_standard21(points)
        expected = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
        np.testing.assert_array_equal(output[:, 0], expected)
        self.assertTrue(set((5, 10, 15, 20)).isdisjoint(output[:, 0]))
        self.assertIn(1, output[:, 0])

    def test_official_sdk_packet_keeps_rotations(self):
        frame = _SdkFrame()
        frame.glove_id = 42
        frame.node_count = 25
        frame.side = 1
        frame.publish_time = 123456
        for row in range(25):
            frame.nodes[row].node_id = row
            frame.nodes[row].position[:] = (row, row + 0.25, row + 0.5)
            frame.nodes[row].rotation_wxyz[0] = 1
        packet = _sdk_frame_to_packet(frame, 8.0)
        self.assertEqual(packet.glove_id, 42)
        self.assertEqual(packet.rotations_wxyz.shape, (25, 4))
        np.testing.assert_array_equal(packet.rotations_wxyz[:, 0], 1)

    def test_official_sdk_packet_keeps_ergonomics_angles(self):
        frame = _SdkFrame()
        frame.node_count = 25
        frame.side = 1
        frame.has_ergonomics = 1
        frame.ergonomics[:] = np.arange(20, dtype=np.float32)
        for row in range(25):
            frame.nodes[row].rotation_wxyz[0] = 1
        packet = _sdk_frame_to_packet(frame, 8.0)
        self.assertEqual(packet.ergonomics.shape, (5, 4))
        np.testing.assert_array_equal(packet.ergonomics, np.arange(20).reshape(5, 4))

    def test_thumb_pip_dip_scale_is_configurable_and_thumb_only(self):
        ergonomics = np.arange(20, dtype=float).reshape(5, 4)
        with (
            patch("input.manus.source.MANUS_THUMB_PIP_DIP_SCALE", 2.0),
            patch("input.manus.source.MANUS_THUMB_DIP_TO_PIP_GAIN", 0.5),
        ):
            angles = _ergonomics_initial_angles(ergonomics)
        np.testing.assert_allclose(
            angles.four_fingers, np.radians(ergonomics[1:5])
        )
        np.testing.assert_allclose(
            angles.thumb_bends,
            np.radians((2 * 2 + 3 * 0.5 * 2, 3 * 2)),
        )
        self.assertEqual(angles.four_finger_space, "robot")

    def test_source_returns_common_frame_with_pad_directions(self):
        packet = raw_packet(received_at=5.0)
        packet.ergonomics = np.arange(20, dtype=float).reshape(5, 4)
        source = ManusSource(
            FakeTransport(packet),
            stale_seconds=0.2,
            clock=lambda: 5.0,
        )
        frame = source.read()
        self.assertTrue(frame.ready)
        self.assertEqual(frame.handedness, "Left")
        self.assertEqual(frame.points.shape, (21, 3))
        np.testing.assert_allclose(frame.points, standard_hand_world())
        self.assertEqual(frame.finger_pad_directions.shape, (5, 3))
        self.assertIsInstance(frame.initial_joint_angles, InitialJointAngles)
        np.testing.assert_allclose(
            frame.initial_joint_angles.four_fingers,
            np.radians(np.arange(20).reshape(5, 4)[1:5]),
        )
        np.testing.assert_allclose(
            frame.initial_joint_angles.thumb_bends,
            np.radians((
                MANUS_THUMB_PIP_DIP_SCALE
                * (2 + MANUS_THUMB_DIP_TO_PIP_GAIN * 3),
                MANUS_THUMB_PIP_DIP_SCALE * 3,
            )),
        )
        self.assertEqual(frame.initial_joint_angles.four_finger_space, "robot")
        raw = standard_hand_world()
        self.assertAlmostEqual(
            frame.raw_palm_length, hand0_middle_tip_distance(raw)
        )
        self.assertAlmostEqual(
            frame.raw_palm_width, np.linalg.norm(raw[5] - raw[17])
        )
        np.testing.assert_allclose(np.linalg.norm(frame.finger_pad_directions, axis=1), 1)
        source.close()

    def test_invalid_ergonomics_keeps_points_but_disables_retarget(self):
        first, second = raw_packet(5.0), raw_packet(5.1)
        first.ergonomics = np.zeros((5, 4))
        second.ergonomics = np.full((5, 4), np.nan)
        first.ergonomics[0] = (10, 20, 30, 40)
        source = ManusSource(
            FakeTransport(first, second), stale_seconds=0.2, clock=lambda: 5.1
        )
        valid = source.read()
        invalid = source.read()
        self.assertTrue(valid.ready)
        self.assertIsNotNone(valid.initial_joint_angles)
        self.assertFalse(invalid.ready)
        self.assertIsNotNone(invalid.points)
        self.assertIsNotNone(invalid.finger_pad_directions)
        self.assertIsNone(invalid.initial_joint_angles)
        self.assertIn("ergonomics invalid", invalid.status)
        source.close()

    def test_tip_quaternions_define_pad_directions(self):
        packet = raw_packet()
        root = np.sqrt(.5)
        packet.rotations_wxyz[MANUS_TO_STANDARD21[[4, 8, 12, 16, 20]]] = (
            root, root, 0, 0
        )
        adapted = adapt_raw_skeleton(
            packet.positions, rotations_wxyz=packet.rotations_wxyz
        )
        expected = np.tile((0, 1., 0), (5, 1))
        np.testing.assert_allclose(adapted.directions, expected, atol=1e-7)

    def test_adapter_preserves_source_translation_and_scale(self):
        packet = raw_packet()
        original = adapt_raw_skeleton(
            packet.positions, rotations_wxyz=packet.rotations_wxyz,
            handedness="Left",
        )
        offset = np.array((0.7, -0.4, 1.2))
        transformed = adapt_raw_skeleton(
            packet.positions * 2 + offset,
            rotations_wxyz=packet.rotations_wxyz, handedness="Left",
        )
        np.testing.assert_allclose(transformed.points, original.points * 2 + offset)
        np.testing.assert_allclose(transformed.directions, original.directions)
        self.assertAlmostEqual(transformed.raw_palm_length, original.raw_palm_length * 2)
        self.assertAlmostEqual(transformed.raw_palm_width, original.raw_palm_width * 2)

    def test_thumb_pad_alignment_uses_handed_ip_to_tip_rotation(self):
        packet = raw_packet()
        with patch("input.manus.adapter.MANUS_THUMB_PAD_ROTATION_DEG", 0.0):
            raw = adapt_raw_skeleton(
                packet.positions,
                rotations_wxyz=packet.rotations_wxyz,
                handedness="Left",
            )
        left = adapt_raw_skeleton(
            packet.positions,
            rotations_wxyz=packet.rotations_wxyz,
            handedness="Left",
        )
        right = adapt_raw_skeleton(
            packet.positions,
            rotations_wxyz=packet.rotations_wxyz,
            handedness="Right",
        )
        unknown = adapt_raw_skeleton(
            packet.positions,
            rotations_wxyz=packet.rotations_wxyz,
        )

        axis = raw.points[4] - raw.points[3]
        axis /= np.linalg.norm(axis)

        def rotate(vector, angle):
            cosine, sine = np.cos(angle), np.sin(angle)
            return (
                vector * cosine
                + np.cross(axis, vector) * sine
                + axis * np.dot(axis, vector) * (1 - cosine)
            )

        angle = np.radians(MANUS_THUMB_PAD_ROTATION_DEG)
        np.testing.assert_allclose(left.directions[0], rotate(raw.directions[0], angle))
        np.testing.assert_allclose(right.directions[0], rotate(raw.directions[0], -angle))
        np.testing.assert_allclose(left.directions[1:], raw.directions[1:])
        np.testing.assert_allclose(right.directions[1:], raw.directions[1:])
        np.testing.assert_allclose(unknown.directions, raw.directions)
        np.testing.assert_allclose(np.linalg.norm(left.directions, axis=1), 1.0)
        np.testing.assert_allclose(np.linalg.norm(right.directions, axis=1), 1.0)

    def test_stale_clears_points(self):
        source = ManusSource(
            FakeTransport(raw_packet(received_at=1.0)),
            stale_seconds=0.2,
            clock=lambda: 2.0,
        )
        frame = source.read()
        self.assertIsNone(frame.points)
        self.assertFalse(frame.ready)

    def test_missing_calibration_runs_terminal_sequence_and_saves(self):
        transport = FakeCalibrationTransport(raw_packet())
        answers = iter(("", "", ""))
        messages = []
        with TemporaryDirectory() as directory:
            path = ensure_calibration(
                transport,
                directory=directory,
                input_fn=lambda _prompt: next(answers),
                output=messages.append,
            )
            self.assertEqual(path, Path(directory) / "left_00001234.mcal")
            self.assertEqual(path.read_bytes(), transport.payload)
        self.assertEqual(transport.calls, [
            ("steps", 0x1234),
            ("start", 0x1234),
            ("step", 0x1234, 3),
            ("step", 0x1234, 7),
            ("finish", 0x1234),
            ("export", 0x1234),
        ])
        self.assertTrue(any("Step 1/2" in message for message in messages))

    def test_existing_calibration_loads_without_prompt(self):
        transport = FakeCalibrationTransport(raw_packet())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "left_00001234.mcal"
            path.write_bytes(b"saved")
            result = ensure_calibration(
                transport,
                directory=directory,
                input_fn=lambda _prompt: self.fail("must not prompt"),
                output=lambda _message: None,
            )
        self.assertEqual(result, path)
        self.assertEqual(transport.calls, [("import", 0x1234, b"saved")])

    def test_failed_calibration_stops_and_writes_no_file(self):
        transport = FakeCalibrationTransport(raw_packet())

        def fail_step(glove_id, index):
            transport.calls.append(("step", glove_id, index))
            raise RuntimeError("recording failed")

        transport.calibration_run_step = fail_step
        answers = iter(("", ""))
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "recording failed"):
                ensure_calibration(
                    transport,
                    directory=directory,
                    input_fn=lambda _prompt: next(answers),
                    output=lambda _message: None,
                )
            self.assertIn(("stop", 0x1234), transport.calls)
            self.assertFalse(list(Path(directory).glob("*.mcal")))

    def test_create_source_gates_manus_before_returning(self):
        source = FakeTransport()
        source.transport = object()
        with patch("input.INPUT_SOURCE", "manus"), patch(
            "input.manus.source.ManusSource", return_value=source
        ), patch(
            "input.manus.calibration.ensure_calibration", return_value="saved.mcal"
        ) as ensure:
            self.assertIs(create_source(), source)
        ensure.assert_called_once_with(source.transport)
        self.assertEqual(source.calibration_path, "saved.mcal")

    def test_create_source_closes_manus_when_calibration_gate_fails(self):
        source = FakeTransport()
        source.transport = object()
        with patch("input.INPUT_SOURCE", "manus"), patch(
            "input.manus.source.ManusSource", return_value=source
        ), patch(
            "input.manus.calibration.ensure_calibration",
            side_effect=RuntimeError("calibration required"),
        ), self.assertRaisesRegex(RuntimeError, "calibration required"):
            create_source()
        self.assertTrue(source.closed)

    def test_track_creates_nothing_after_failed_calibration_gate(self):
        arguments = type("Args", (), {"ros": False, "sim": False})()
        with patch("track._parse_args", return_value=arguments), patch(
            "track.create_source", side_effect=RuntimeError("calibration required")
        ), patch("track.Retargeter") as retargeter, patch(
            "track.ViewerProcess"
        ) as viewer, self.assertRaisesRegex(RuntimeError, "calibration required"):
            track.main()
        retargeter.assert_not_called()
        viewer.assert_not_called()

if __name__ == "__main__":
    unittest.main()
