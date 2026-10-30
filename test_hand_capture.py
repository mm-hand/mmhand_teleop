import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import config as C
from config import (
    KEYPOINT_LAYOUT,
    KEYPOINT_TOPIC,
    PAD_DIRECTION_LAYOUT,
    PAD_DIRECTION_TOPIC,
    ROBOT_JOINT_NAMES,
    ROBOT_LAYOUT,
    ROBOT_TOPIC,
)
from input.frame import (
    InitialJointAngles, InputFrame, initial_joint_angles_from_points,
    hand0_middle_tip_distance,
)
from retarget import (
    Retargeter, RetargetWorker, RobotModel, compute_cmc_frame,
    human_vector_features,
)
from ros import RosOutput
from viewer import (
    HUMAN_DISPLAY_PALM_WIDTH,
    VECTOR_ITEMS, Viewer, _angle_text, _arrow_points,
    _bind_dashboard_server,
    _human_display_frame, _human_residual_vector_data, _latency_text,
    _loss_text, _pad_angle_errors, _render_time_text,
    _robot_residual_vector_data, _timing_breakdown_text,
    _vector_group_info, _vector_label,
)

PAD_DIRECTIONS = np.tile((0.0, 0.0, -1.0), (5, 1))


class _SlowViewer:
    def update(self, *_args, **_kwargs):
        time.sleep(0.1)
        return True

    def close(self):
        pass


class _FailingViewer:
    def update(self, *_args, **_kwargs):
        raise RuntimeError("fake viewer failure")

    def close(self):
        pass


def hand():
    points = np.zeros((21, 3), float)
    for start, x, y in zip((5, 9, 13, 17), (-25, 0, 20, 35), (45, 50, 45, 38)):
        points[start:start + 4] = (
            (x, y, 0), (x, y + 30, 0), (x, y + 50, 0), (x, y + 65, 0)
        )
    points[1:5] = ((-30, 15, 0), (-45, 25, 0), (-58, 34, 4), (-70, 42, 8))
    return points / 1000.0


class RetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RobotModel()
        cls.points = hand()
        cls.angles = initial_joint_angles_from_points(cls.points)

    def test_robot_contract_and_fingertip_frames(self):
        model = self.model
        self.assertEqual(model.names, ROBOT_JOINT_NAMES)
        self.assertEqual(len(model.fk(model.seed)), 32)
        self.assertTrue(np.all(model.lower < model.upper))
        points, directions = model.fingertip_pads(model.seed)
        self.assertEqual((points.shape, directions.shape), ((5, 3), (5, 3)))
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1)
        arrows = _arrow_points(points, directions)
        np.testing.assert_allclose(arrows[:, 0], points)
        np.testing.assert_allclose(
            np.linalg.norm(arrows[:, 1] - points, axis=1), .025, atol=1e-8
        )
        width = model.palm_width(model.seed)
        self.assertAlmostEqual(width, 0.0743529722472478)
        self.assertAlmostEqual(
            width, model.palm_width((model.lower + model.upper) / 2)
        )
        mesh_tip = model.middle_fingertip_surface_point(model.seed)
        middle_link = model.fk(model.seed)["finger_2_fingertip_1"]
        np.testing.assert_allclose(
            mesh_tip,
            middle_link[:3, :3] @ model.middle_fingertip_surface_local
            + middle_link[:3, 3],
        )
        self.assertGreater(
            model.middle_fingertip_surface_local[0],
            model.joints["2-tip"]["origin"][0, 3],
        )
        self.assertAlmostEqual(
            model.middle_fingertip_surface_distance(model.seed),
            np.linalg.norm(mesh_tip - model.fk(model.seed)["base_link"][:3, 3]),
        )
        self.assertAlmostEqual(
            model.urdf_palm_length,
            model.middle_fingertip_surface_distance(model.seed),
        )
        bent = model.seed.copy()
        bent[model.finger_joints[1, 1:]] = model.upper[
            model.finger_joints[1, 1:]
        ]
        self.assertNotAlmostEqual(
            model.urdf_palm_length,
            model.middle_fingertip_surface_distance(bent),
        )

    def test_residual_vector_diagnostics_match_loss_features(self):
        retargeter = Retargeter(self.model)
        targets = retargeter._targets(
            self.points, finger_pad_directions=PAD_DIRECTIONS,
            initial_joint_angles=self.angles,
        )
        human = _human_residual_vector_data(self.points, PAD_DIRECTIONS)
        np.testing.assert_allclose(human["thumb_tip"][2], targets[0])
        np.testing.assert_allclose(human["fingertip_vectors"][2], targets[2])
        np.testing.assert_allclose(
            np.vstack((human["thumb_pad"][2], human["finger_pads"][2])),
            targets[3],
        )
        q = (self.model.lower + self.model.upper) / 2
        robot = _robot_residual_vector_data(self.model, q)
        values = self.model.features(q)[0]
        np.testing.assert_allclose(robot["thumb_tip"][2], values[0])
        np.testing.assert_allclose(robot["fingertip_vectors"][2], values[2])
        np.testing.assert_allclose(
            np.vstack((robot["thumb_pad"][2], robot["finger_pads"][2])),
            values[3],
        )

    def test_residual_vector_labels_angles_and_group_selection(self):
        self.assertEqual(
            _vector_label("Index", np.array((.01, -.02, .03))),
            "Index\nxyz [+10.0, -20.0, +30.0] mm\n|v| 37.4 mm",
        )
        direction_text = _vector_label(
            "Thumb", np.array((1.0, 0.0, 0.0)),
            direction=True, angle_error=90.0,
        )
        self.assertIn("|d| 1.000", direction_text)
        self.assertIn("angle error 90.0°", direction_text)
        np.testing.assert_allclose(
            _pad_angle_errors(np.eye(3), np.roll(np.eye(3), 1, axis=1)),
            (90.0, 90.0, 90.0),
        )
        self.assertIn("shared weight", _vector_group_info("fingertip_vectors"))
        finger_pad_info = _vector_group_info("finger_pads")
        self.assertIn("configured I=", finger_pad_info)
        self.assertIn("solver effective I=", finger_pad_info)
        self.assertIn(
            f"L={C.RETARGET_LITTLE_PAD_WEIGHT / 3:g}", finger_pad_info
        )

        class Handle:
            visible = False

        viewer = Viewer.__new__(Viewer)
        viewer.vector_lock = threading.Lock()
        viewer.vector_group = "fingertip_vectors"
        viewer.vector_available = {"human": True, "robot": True}
        viewer.human_vector_handles = {
            group: [(Handle(), Handle()) for _ in items]
            for group, items in VECTOR_ITEMS.items()
        }
        viewer.robot_vector_handles = {
            group: [(Handle(), Handle()) for _ in items]
            for group, items in VECTOR_ITEMS.items()
        }
        viewer.set_vector_group("thumb_pad")
        for handles in (viewer.human_vector_handles, viewer.robot_vector_handles):
            self.assertTrue(all(
                arrow.visible and label.visible
                for arrow, label in handles["thumb_pad"]
            ))
            self.assertTrue(all(
                not arrow.visible and not label.visible
                for group, items in handles.items() if group != "thumb_pad"
                for arrow, label in items
            ))
        with self.assertRaises(ValueError):
            viewer.set_vector_group("invalid")
        self.assertEqual(viewer.vector_group, "thumb_pad")
        viewer.set_vector_group("off")
        self.assertFalse(any(
            arrow.visible or label.visible
            for handles in (viewer.human_vector_handles, viewer.robot_vector_handles)
            for items in handles.values() for arrow, label in items
        ))

    def test_initial_joint_angle_contract(self):
        copied = self.angles.copy()
        self.assertIsNot(copied.four_fingers, self.angles.four_fingers)
        self.assertIsNot(copied.thumb_bends, self.angles.thumb_bends)
        self.assertEqual(copied.four_finger_space, "human")
        for four, thumb in (
            (np.zeros((3, 4)), np.zeros(2)),
            (np.zeros((4, 4)), np.zeros(3)),
            (np.full((4, 4), np.nan), np.zeros(2)),
        ):
            with self.assertRaises(ValueError):
                InitialJointAngles(four, thumb)
        with self.assertRaisesRegex(ValueError, "angle space"):
            InitialJointAngles(np.zeros((4, 4)), np.zeros(2), "unknown")
        with self.assertRaisesRegex(ValueError, "require initial"):
            InputFrame(
                0.0, self.points, "Left", True, "TRACKING",
                finger_pad_directions=PAD_DIRECTIONS,
            )
        optional = InputFrame(
            0.0, self.points, "Left", True, "TRACKING",
            initial_joint_angles=self.angles,
        )
        self.assertIsNone(optional.finger_pad_directions)

    def test_frames_and_direct_fingers(self):
        origin, rotation = compute_cmc_frame(self.points)
        np.testing.assert_allclose(origin, self.points[1])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        q = self.model.initial_angle_targets(self.angles)
        self.assertTrue(np.isfinite(q).all())
        self.assertTrue(np.all((q >= self.model.lower) & (q <= self.model.upper)))

    def test_robot_skeleton_reproduces_its_own_finger_angles(self):
        # The affine calibration must invert the reading the robot itself emits,
        # otherwise the angle prior drags every finger off the true pose.
        for q in (
            self.model.seed,
            np.clip(self.model.seed + 0.4 * (self.model.upper - self.model.seed),
                    self.model.lower, self.model.upper),
        ):
            points = self.model.skeleton_points(q)
            angles = initial_joint_angles_from_points(points)
            mapped = self.model.angle_targets(angles)
            indices = self.model.finger_joints.ravel()
            np.testing.assert_allclose(
                mapped[indices], q[indices], atol=0.12,
                err_msg="four-finger angle mapping must return the source pose",
            )

    def test_hand0_middle_tip_distance_is_direct(self):
        self.assertAlmostEqual(
            hand0_middle_tip_distance(self.points),
            np.linalg.norm(self.points[12] - self.points[0]),
        )

    def test_analytic_jacobians_and_loss_gradient(self):
        q = (self.model.lower + self.model.upper) / 2
        values, jacobians = self.model.features(q)
        np.testing.assert_allclose(values[1].ravel(), q[18:20])
        np.testing.assert_array_equal(jacobians[1][:, 0, 18:20], np.eye(2))
        numeric = [np.empty_like(item) for item in jacobians]
        step = 1e-6
        for column, joint in enumerate(range(len(q))):
            plus, minus = q.copy(), q.copy()
            plus[joint] += step
            minus[joint] -= step
            a, b = self.model.features(plus)[0], self.model.features(minus)[0]
            for group in range(len(values)):
                numeric[group][..., column] = (a[group] - b[group]) / (2 * step)
        for actual, expected in zip(jacobians, numeric):
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)

        retargeter = Retargeter(self.model)
        targets = retargeter._targets(
            self.points, finger_pad_directions=PAD_DIRECTIONS,
            initial_joint_angles=self.angles,
        )
        loss, gradient, _ = retargeter._losses(q, targets)
        numeric = np.empty(len(q))
        for column, joint in enumerate(range(len(q))):
            plus, minus = q.copy(), q.copy()
            plus[joint] += step
            minus[joint] -= step
            numeric[column] = (
                retargeter._losses(plus, targets)[0]
                - retargeter._losses(minus, targets)[0]
            ) / (2 * step)
        self.assertTrue(np.isfinite(loss))
        np.testing.assert_allclose(gradient, numeric, atol=1e-6, rtol=1e-5)

    def test_joint_initial_guess_and_independent_targets(self):
        retargeter = Retargeter(self.model)
        cold = retargeter._targets(
            self.points, finger_pad_directions=PAD_DIRECTIONS,
            initial_joint_angles=self.angles,
        )[5]
        self.assertEqual(cold[17], self.model.upper[17])
        np.testing.assert_allclose(
            cold[18:20],
            np.clip(self.angles.thumb_bends, self.model.lower[18:20], self.model.upper[18:20]),
        )
        retargeter.q[self.model.thumb] = .5 * (
            self.model.lower[self.model.thumb] + self.model.upper[self.model.thumb]
        )
        retargeter.has_previous = True
        targets = retargeter._targets(
            self.points, retargeter.q, PAD_DIRECTIONS, self.angles
        )
        initial = targets[5]
        np.testing.assert_array_equal(
            initial[self.model.finger_joints.ravel()], targets[4]
        )
        np.testing.assert_array_equal(initial[[16, 17, 20]], retargeter.q[[16, 17, 20]])
        np.testing.assert_allclose(initial[18:20], cold[18:20])
        _, jacobians = self.model.features(initial)
        relative_jacobian = jacobians[2]
        self.assertTrue(np.any(relative_jacobian[..., :16]))
        self.assertTrue(np.any(relative_jacobian[..., 16:]))

        before = retargeter._losses(initial, targets)[2]
        retargeter.solve(self.points, 0.0, PAD_DIRECTIONS, self.angles)
        after = retargeter._losses(retargeter.q, targets)[2]
        self.assertLess(after["total"], before["total"])
        self.assertLess(after["fingertip_vectors"], before["fingertip_vectors"])

        q = initial.copy()
        q[self.model.finger_joints.ravel()] += .05
        q = np.clip(q, self.model.lower, self.model.upper)
        with patch("retarget.C.RETARGET_FINGER_ANGLE_WEIGHT", 0.0):
            _, no_angles, terms = retargeter._losses(q, targets)
        with patch("retarget.C.RETARGET_FINGER_ANGLE_WEIGHT", 1.0):
            _, with_angles, _ = retargeter._losses(q, targets)
        self.assertEqual(terms["finger_angles"], 0)
        np.testing.assert_allclose(
            (with_angles - no_angles)[self.model.thumb], 0, atol=1e-15
        )
        self.assertTrue(np.any(with_angles[:16] != no_angles[:16]))

        with patch("retarget.C.RETARGET_FINGERTIP_VECTOR_WEIGHT", 0.0):
            _, no_vectors, terms = retargeter._losses(q, targets)
        with patch("retarget.C.RETARGET_FINGERTIP_VECTOR_WEIGHT", 2.0):
            _, with_vectors, _ = retargeter._losses(q, targets)
        self.assertEqual(terms["fingertip_vectors"], 0)
        vector_gradient = with_vectors - no_vectors
        self.assertTrue(np.any(vector_gradient[:16]))
        self.assertTrue(np.any(vector_gradient[16:]))

    def test_four_finger_flexion_uses_measured_zero(self):
        # Human zero flexion must land on the robot's own zero, not on its URDF
        # lower limit; that offset was a constant bias on every finger.
        index = self.model.finger_joints[0]
        rest = initial_joint_angles_from_points(
            self.model.skeleton_points(self.model.seed)
        )
        four = rest.four_fingers.copy()
        four[0] = (rest.four_fingers[0, 0], 0.0,
                   np.radians(30), np.radians(40))
        mapped = self.model.initial_angle_targets(
            InitialJointAngles(four, rest.thumb_bends)
        )
        gain, offset = self.model.finger_gain, self.model.finger_offset
        base = 0  # Index is the first row of self.model.finger_joints
        self.assertAlmostEqual(mapped[index[1]], (0.0 - offset[base + 1]) / gain[base + 1])
        self.assertAlmostEqual(
            mapped[index[2]], (np.radians(30) - offset[base + 2]) / gain[base + 2]
        )
        self.assertAlmostEqual(
            mapped[index[3]], (np.radians(40) - offset[base + 3]) / gain[base + 3]
        )

    def test_manus_angles_directly_seed_all_observed_robot_joints(self):
        four = np.radians(np.asarray((
            (10, 20, 30, 40),
            (11, 21, 31, 41),
            (12, 22, 32, 42),
            (13, 23, 33, 43),
        )))
        angles = InitialJointAngles(
            four, np.radians((24, 25)), four_finger_space="robot"
        )
        targets = self.model.angle_targets(angles)
        for row, indices in enumerate(self.model.finger_joints):
            np.testing.assert_allclose(
                targets[indices],
                np.clip(four[row], self.model.lower[indices], self.model.upper[indices]),
            )
        np.testing.assert_allclose(targets[18:20], np.radians((24, 25)))

        previous = (self.model.lower + self.model.upper) / 2
        initial = self.model.initial_guess(targets, previous)
        observed = np.concatenate((self.model.finger_joints.ravel(), (18, 19)))
        np.testing.assert_allclose(initial[observed], targets[observed])
        np.testing.assert_allclose(initial[[16, 17, 20]], previous[[16, 17, 20]])

        retarget_targets = Retargeter(self.model)._targets(
            self.points, previous, PAD_DIRECTIONS, angles
        )
        np.testing.assert_allclose(
            retarget_targets[4], targets[self.model.finger_joints.ravel()]
        )
        np.testing.assert_allclose(retarget_targets[1].ravel(), targets[18:20])
        np.testing.assert_allclose(retarget_targets[5], initial)

    def test_direct_manus_angles_are_clipped_to_each_urdf_limit(self):
        angles = InitialJointAngles(
            np.full((4, 4), np.radians(400.0)),
            np.radians((-400.0, 400.0)),
            four_finger_space="robot",
        )
        targets = self.model.angle_targets(angles)
        indices = self.model.finger_joints.ravel()
        np.testing.assert_allclose(targets[indices], self.model.upper[indices])
        self.assertEqual(targets[18], self.model.lower[18])
        self.assertEqual(targets[19], self.model.upper[19])

    def test_direct_manus_initial_angles_remain_soft_joint_targets(self):
        angles = InitialJointAngles(
            np.zeros((4, 4)), np.zeros(2), four_finger_space="robot"
        )
        retargeter = Retargeter(self.model)
        targets = retargeter._targets(
            self.points, finger_pad_directions=PAD_DIRECTIONS,
            initial_joint_angles=angles,
        )
        initial = targets[5]
        initial_loss = retargeter._losses(initial, targets)[2]["total"]
        _, losses = retargeter.solve(
            self.points, 0.0, PAD_DIRECTIONS, angles
        )
        observed = np.concatenate((self.model.finger_joints.ravel(), (18, 19)))
        self.assertTrue(np.any(np.abs(retargeter.q[observed] - initial[observed]) > 1e-6))
        self.assertGreater(losses["finger_angles"], 0.0)
        self.assertGreater(losses["thumb_proximal_bend"], 0.0)
        self.assertGreater(losses["thumb_distal_bend"], 0.0)
        self.assertLess(losses["total"], initial_loss)

    def test_thumb_bends_are_independent_and_clipped(self):
        retargeter = Retargeter(self.model)
        angles = InitialJointAngles(
            self.angles.four_fingers, np.radians((30.0, 400.0))
        )
        mapped = self.model.initial_angle_targets(angles)
        self.assertAlmostEqual(mapped[18], np.radians(30.0))
        self.assertAlmostEqual(mapped[19], self.model.upper[19])
        targets = retargeter._targets(
            self.points, finger_pad_directions=PAD_DIRECTIONS,
            initial_joint_angles=angles,
        )
        np.testing.assert_allclose(
            targets[1].ravel(), mapped[18:20]
        )
        changed = InitialJointAngles(
            self.angles.four_fingers, np.radians((45.0, 400.0))
        )
        changed_target = retargeter._targets(
            self.points, finger_pad_directions=PAD_DIRECTIONS,
            initial_joint_angles=changed,
        )[1].ravel()
        self.assertNotEqual(changed_target[0], targets[1].ravel()[0])
        self.assertEqual(changed_target[1], targets[1].ravel()[1])

    def test_thumb_pad_loss_and_solve(self):
        target_q = self.model.seed.copy()
        thumb = self.model.thumb
        target_q[thumb] = self.model.lower[thumb] + .25 * (
            self.model.upper[thumb] - self.model.lower[thumb]
        )
        target = self.model.fingertip_pads(target_q)[1][0] @ self.model.palm_frame
        _, rotation = compute_cmc_frame(self.points)
        directions = np.tile(target @ rotation.T, (5, 1))
        with patch("retarget.C.RETARGET_THUMB_PAD_WEIGHT", 0.0):
            baseline, _ = Retargeter(self.model).solve(
                self.points, 0.0, directions, self.angles
            )
        with patch("retarget.C.RETARGET_THUMB_PAD_WEIGHT", .25):
            retargeter = Retargeter(self.model)
            aligned, losses = retargeter.solve(
                self.points, 0.0, directions, self.angles
            )
        baseline_pad = self.model.fingertip_pads(baseline)[1][0] @ self.model.palm_frame
        aligned_pad = self.model.fingertip_pads(aligned)[1][0] @ self.model.palm_frame
        self.assertGreater(aligned_pad @ target, baseline_pad @ target)
        self.assertEqual(
            tuple(losses),
            ("thumb_tip", "thumb_proximal_bend", "thumb_distal_bend",
             "finger_angles", "fingertip_vectors", "thumb_pad",
             "finger_pads", "total"),
        )
        self.assertAlmostEqual(
            sum(value for name, value in losses.items() if name != "total"),
            losses["total"],
        )
        self.assertIn("thumb pad", _loss_text(losses))

        with self.assertRaisesRegex(ValueError, "Initial joint angles"):
            Retargeter(self.model).solve(self.points)
        for bad in (np.zeros((20, 3)), np.full((21, 3), np.nan)):
            self.assertIsNone(Retargeter(self.model).solve(bad, finger_pad_directions=PAD_DIRECTIONS))

    def test_missing_finger_pad_directions_disable_pad_losses(self):
        self.assertIsNone(human_vector_features(self.points, None)[4])
        retargeter = Retargeter(self.model)
        targets = retargeter._targets(
            self.points, finger_pad_directions=None,
            initial_joint_angles=self.angles,
        )
        self.assertIsNone(targets[3])
        q = (self.model.lower + self.model.upper) / 2
        loss, gradient, terms = retargeter._losses(q, targets)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(gradient).all())
        self.assertEqual((terms["thumb_pad"], terms["finger_pads"]), (0.0, 0.0))
        solved, losses = retargeter.solve(self.points, 0.0, None, self.angles)
        self.assertIsNotNone(solved)
        self.assertEqual(
            (losses["thumb_pad"], losses["finger_pads"]), (0.0, 0.0)
        )
        self.assertAlmostEqual(
            sum(value for name, value in losses.items() if name != "total"),
            losses["total"],
        )

    def test_four_finger_pad_residual_has_independent_weights(self):
        target_q = self.model.seed.copy()
        indices = self.model.finger_joints.ravel()
        target_q[indices] = self.model.lower[indices] + .3 * (
            self.model.upper[indices] - self.model.lower[indices]
        )
        target_pads = self.model.fingertip_pads(target_q)[1] @ self.model.palm_frame
        _, rotation = compute_cmc_frame(self.points)
        directions = target_pads @ rotation.T
        retargeter = Retargeter(self.model)
        targets = retargeter._targets(
            self.points, finger_pad_directions=directions,
            initial_joint_angles=self.angles,
        )
        names = (
            "RETARGET_INDEX_PAD_WEIGHT", "RETARGET_MIDDLE_PAD_WEIGHT",
            "RETARGET_RING_PAD_WEIGHT", "RETARGET_LITTLE_PAD_WEIGHT",
        )
        with patch.multiple("retarget.C", **dict.fromkeys(names, 0.0)):
            _, without, terms = retargeter._losses(self.model.seed, targets)
        with patch.multiple("retarget.C", **dict.fromkeys(names, 2.0)):
            _, with_pads, terms_with_pads = retargeter._losses(
                self.model.seed, targets
            )
        self.assertEqual(terms["finger_pads"], 0.0)
        self.assertGreater(terms_with_pads["finger_pads"], 0.0)
        gradient = with_pads - without
        self.assertTrue(np.any(np.abs(gradient[:16]) > 1e-12))
        np.testing.assert_allclose(gradient[16:], 0.0, atol=1e-15)

        individual = dict.fromkeys(names, 0.0)
        individual["RETARGET_LITTLE_PAD_WEIGHT"] = 8.0
        with patch.multiple("retarget.C", **individual):
            _, little_gradient, little_terms = retargeter._losses(
                self.model.seed, targets
            )
        self.assertGreater(little_terms["finger_pads"], 0.0)
        self.assertTrue(np.any(np.abs(little_gradient[:4] - without[:4]) > 1e-12))
        np.testing.assert_allclose(
            little_gradient[4:] - without[4:], 0.0, atol=1e-15
        )

    def test_robot_features_reproduce_robot_skeleton(self):
        # Feeding the robot's own FK hand back must produce zero residual; any
        # frame mismatch shows up here as a constant offset on every feature.
        # The tolerance covers the robot CMC frame drifting a couple of degrees
        # between the seed pose and a fully flexed one.
        for amplitude in (0.0, 0.3, 0.6):
            q = np.clip(
                self.model.seed + amplitude * (self.model.upper - self.model.seed),
                self.model.lower, self.model.upper,
            )
            points = self.model.skeleton_points(q)
            pads = self.model.fingertip_pads(q)[1]
            _, _, thumb_tip, relative, target_pads = human_vector_features(points, pads)
            values = self.model.features(q)[0]
            np.testing.assert_allclose(values[0], thumb_tip, atol=2.5e-3)
            np.testing.assert_allclose(values[2], relative, atol=2.5e-3)
            cosine = np.clip(np.sum(values[3] * target_pads, axis=1), -1, 1)
            self.assertLess(np.degrees(np.arccos(cosine)).max(), 2.5)

    def test_velocity_fidelity_on_a_curling_finger(self):
        # Local transfer: move the source hand a little and compare the fingertip
        # velocity the solver produces with the source's own velocity.
        retargeter = Retargeter(self.model)
        previous = None
        linear_gain, linear_cosine, angular_gain, angular_cosine = [], [], [], []
        for index in range(30):
            q = np.clip(
                self.model.seed + (index / 29.0) * 0.6
                * (self.model.upper - self.model.seed),
                self.model.lower, self.model.upper,
            )
            points = self.model.skeleton_points(q)
            pads = self.model.fingertip_pads(q)[1]
            solved, _ = retargeter.solve(
                points, index / 30.0, pads, initial_joint_angles_from_points(points)
            )
            human = human_vector_features(points, pads)
            robot = self.model.features(solved)[0]
            if previous is not None:
                source = np.asarray(human[3]) - previous[0]
                result = np.asarray(robot[2]) - previous[1]
                source_speed = np.linalg.norm(source, axis=1) + 1e-12
                result_speed = np.linalg.norm(result, axis=1) + 1e-12
                linear_gain.append(result_speed / source_speed)
                linear_cosine.append(
                    np.sum(source * result, axis=1) / (source_speed * result_speed)
                )
                source_turn = np.degrees(np.arccos(np.clip(
                    np.sum(previous[2] * np.asarray(human[4]), axis=1), -1, 1)))
                result_turn = np.degrees(np.arccos(np.clip(
                    np.sum(previous[3] * np.asarray(robot[3]), axis=1), -1, 1)))
                active = source_turn > 0.5
                if np.any(active):
                    angular_gain.append(np.where(
                        active, result_turn / (source_turn + 1e-12), np.nan))
                    source_axis = np.cross(previous[2], np.asarray(human[4]))
                    result_axis = np.cross(previous[3], np.asarray(robot[3]))
                    cosine = np.sum(source_axis * result_axis, axis=1) / (
                        np.linalg.norm(source_axis, axis=1)
                        * np.linalg.norm(result_axis, axis=1) + 1e-12
                    )
                    angular_cosine.append(np.where(active, cosine, np.nan))
            previous = (
                np.asarray(human[3]), np.asarray(robot[2]),
                np.asarray(human[4]), np.asarray(robot[3]),
            )
        gain = np.mean(linear_gain)
        self.assertGreater(gain, 0.9, "fingertip translation is too slow")
        self.assertLess(gain, 1.15, "fingertip translation overshoots")
        self.assertGreater(np.nanmean(linear_cosine), 0.94)
        self.assertGreater(np.nanmean(angular_gain), 0.75)
        self.assertGreater(np.nanmean(angular_cosine), 0.85)

    def test_unfiltered_output_and_pause(self):
        retargeter = Retargeter(self.model)
        first, _ = retargeter.solve(self.points, 0.0, PAD_DIRECTIONS, self.angles)
        raw = retargeter.q.copy()
        np.testing.assert_allclose(first, raw)
        moved = self.points.copy()
        moved[4] += (.02, -.01, .01)
        moved_angles = initial_joint_angles_from_points(moved)
        second, _ = retargeter.solve(moved, 1 / 30, PAD_DIRECTIONS, moved_angles)
        np.testing.assert_array_equal(second, np.clip(
            retargeter.q, self.model.lower, self.model.upper
        ))
        self.assertTrue(np.all(second >= self.model.lower))
        self.assertTrue(np.all(second <= self.model.upper))
        self.assertNotIn("output_filter", retargeter.last_stage_timings_ms)
        retargeter.pause()
        self.assertFalse(retargeter.has_previous)
        np.testing.assert_array_equal(retargeter.q, self.model.seed)

    def test_worker_keeps_latest_and_copies_directions(self):
        class Fake:
            model = object()

            def __init__(self):
                self.calls = []
                self.started, self.release = threading.Event(), threading.Event()

            def solve(self, points, _timestamp, directions, initial_angles):
                value = int(points[0, 0])
                self.calls.append((
                    value,
                    None if directions is None else directions.copy(),
                    None if initial_angles is None else initial_angles.copy(),
                ))
                if not self.release.is_set():
                    self.started.set()
                    self.release.wait(1)
                return np.full(21, value, float), {"total": float(value)}

            def pause(self):
                pass

        fake, worker = Fake(), None
        worker = RetargetWorker(fake, clock=lambda: 3.025)
        try:
            directions = np.ones((5, 3))
            angles = self.angles.copy()
            worker.submit(np.ones((21, 3)), 1.0, directions, angles)
            directions[:] = 9
            angles.four_fingers[:] = 9
            self.assertTrue(fake.started.wait(1))
            worker.submit(np.full((21, 3), 2), 2.0)
            worker.submit(np.full((21, 3), 3), 3.0)
            fake.release.set()
            deadline, output = time.monotonic() + 1, None
            while time.monotonic() < deadline and (output is None or output[0][0] != 3):
                output = worker.poll()
                time.sleep(.001)
            self.assertEqual([call[0] for call in fake.calls], [1, 3])
            np.testing.assert_array_equal(fake.calls[0][1], np.ones((5, 3)))
            np.testing.assert_array_equal(
                fake.calls[0][2].four_fingers, self.angles.four_fingers
            )
            self.assertEqual(output[1], {"total": 3.0})
            self.assertAlmostEqual(output[2], 0.0)
            self.assertEqual(output[3]["solve_total"], 0.0)
        finally:
            worker.close()

    def test_latency_text_contract(self):
        self.assertEqual(_latency_text(None), "waiting")
        self.assertEqual(_latency_text(float("nan")), "waiting")
        self.assertEqual(_latency_text(-1.0), "waiting")
        self.assertEqual(_latency_text(12.34), "12.3 ms")
        self.assertEqual(_render_time_text(None), "render waiting")
        self.assertEqual(_render_time_text(0.26), "render 0.3 ms")
        self.assertIn("SLSQP           3.2 ms", _timing_breakdown_text({"slsqp": 3.2}))
        self.assertNotIn("output filter", _timing_breakdown_text({"output_filter": 1.0}))
        self.assertEqual(
            _angle_text(
                "Input thumb", (), np.empty(0), 0.086, 0.1234, 0.0567, 0.0789
            ),
            "Input thumb\npalm width   86.0 mm\npalm length  123.4 mm"
            "\nraw palm length   56.7 mm\nraw palm width   78.9 mm",
        )
        self.assertEqual(
            _angle_text(
                "MMHand thumb", (), np.empty(0), 0.07435, 0.1312,
                palm_width_label="MMHand palm width",
                middle_tip_label="MMHand URDF palm length",
            ),
            "MMHand thumb\nMMHand palm width   74.3 mm"
            "\nMMHand URDF palm length  131.2 mm",
        )

    def test_viewer_process_is_non_blocking_and_reports_child_errors(self):
        from viewer_process import ViewerProcess

        frame = InputFrame.empty(time.monotonic(), "WAITING")
        viewer = ViewerProcess(_factory=_SlowViewer)
        try:
            started = time.monotonic()
            for _ in range(10):
                viewer.update(frame)
            self.assertLess(time.monotonic() - started, 0.08)
            with self.assertRaises(ValueError):
                ready = InputFrame(
                    time.monotonic(), self.points, "Left", True, "TRACKING",
                    finger_pad_directions=PAD_DIRECTIONS,
                    initial_joint_angles=self.angles,
                )
                viewer.update(ready, np.zeros(20))
        finally:
            viewer.close()

        viewer = ViewerProcess(_factory=_FailingViewer)
        try:
            viewer.update(frame)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    viewer.update(frame)
                except RuntimeError as error:
                    self.assertIn("fake viewer failure", str(error))
                    break
                time.sleep(0.01)
            else:
                self.fail("viewer child error was not propagated")
        finally:
            viewer.close()

    def test_dashboard_skips_occupied_web_port(self):
        import socket
        from http.server import BaseHTTPRequestHandler

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
            busy.bind(("0.0.0.0", 0))
            busy.listen(1)
            port = busy.getsockname()[1]
            server = _bind_dashboard_server(BaseHTTPRequestHandler, port)
            try:
                self.assertGreater(server.server_address[1], port)
            finally:
                server.server_close()

    def test_ros_contract(self):
        class Message:
            def __init__(self): self.layout, self.data = None, []
        class Dimension:
            def __init__(self, label="", size=0, stride=0):
                self.label, self.size, self.stride = label, size, stride
        class Layout:
            def __init__(self, dim=None, data_offset=0): self.dim = dim
        class Publisher:
            def __init__(self, topic): self.topic, self.messages = topic, []
            def publish(self, message): self.messages.append(message)
        class Node:
            def __init__(self): self.publishers = []
            def create_publisher(self, _type, topic, _depth):
                publisher = Publisher(topic); self.publishers.append(publisher); return publisher
            def destroy_node(self): pass

        node = Node()
        rclpy = types.ModuleType("rclpy")
        rclpy.init = lambda args=None: None
        rclpy.create_node = lambda _name: node
        rclpy.shutdown = lambda: None
        messages = types.ModuleType("std_msgs.msg")
        messages.Float32MultiArray, messages.MultiArrayDimension = Message, Dimension
        messages.MultiArrayLayout = Layout
        package = types.ModuleType("std_msgs"); package.msg = messages
        with patch.dict(sys.modules, {
            "rclpy": rclpy, "std_msgs": package, "std_msgs.msg": messages,
        }):
            output = RosOutput()
            output.hand(InputFrame(
                1.0, self.points, "Left", True, "TRACKING",
                finger_pad_directions=PAD_DIRECTIONS,
                initial_joint_angles=self.angles,
            ))
            output.hand(InputFrame(
                2.0, self.points, "Left", True, "TRACKING",
                initial_joint_angles=self.angles,
            ))
            output.joints(np.arange(21))
            output.close()
        self.assertEqual(
            [publisher.topic for publisher in node.publishers],
            ["/mmhand/keypoints", "/mmhand/finger_pad_directions",
             "/mmhand/joint_angle_deg"],
        )
        for publisher, expected in zip(node.publishers, (
            self.points.ravel(), PAD_DIRECTIONS.ravel(), np.arange(21)
        )):
            np.testing.assert_allclose(publisher.messages[0].data, expected)
        sizes = [len(publisher.messages[0].data) for publisher in node.publishers]
        self.assertEqual(sizes, [63, 15, 21])
        labels = [publisher.messages[0].layout.dim[0].label for publisher in node.publishers]
        self.assertEqual(labels, [
            "mmhand_teleop:keypoints:v2:source_m:hand=Left",
            "mmhand_teleop:finger_pad_directions:v2:source_unit:hand=Left",
            "mmhand:J00-J20:urdf_deg",
        ])
        self.assertEqual(
            [len(publisher.messages) for publisher in node.publishers],
            [2, 1, 1],
        )

class HumanDisplayTests(unittest.TestCase):
    def setUp(self):
        self.points = hand()
        self.directions = PAD_DIRECTIONS
        self.matrix, self.width, self.cmc = _human_display_frame(
            self.points, self.directions
        )

    def shown(self, points, matrix=None, width=None):
        matrix = self.matrix if matrix is None else matrix
        width = self.width if width is None else width
        return (
            (points - points[1]) @ matrix
            * (HUMAN_DISPLAY_PALM_WIDTH / width)
        )

    def test_palm_faces_screen_and_fingers_point_up(self):
        pads = self.directions @ self.matrix
        np.testing.assert_allclose(
            pads, np.tile((0.0, 0.0, 1.0), (5, 1)), atol=1e-9
        )
        fingers = (self.points[9] - self.points[0]) @ self.matrix
        np.testing.assert_allclose(
            fingers / np.linalg.norm(fingers), (0.0, 1.0, 0.0), atol=1e-9
        )
        shown = self.shown(self.points)
        self.assertAlmostEqual(
            np.linalg.norm(shown[17] - shown[5]), HUMAN_DISPLAY_PALM_WIDTH
        )
        self.assertGreater(shown[4, 0], shown[5, 0])

    def test_display_is_invariant_to_source_pose_and_size(self):
        rotation = Rotation.from_euler(
            "xyz", (27, -41, 63), degrees=True
        ).as_matrix()
        translation = np.array((0.12, -0.08, 0.35))
        factor = 1.7
        moved = self.points @ rotation.T * factor + translation
        turned = self.directions @ rotation.T
        matrix, width, _ = _human_display_frame(moved, turned)
        np.testing.assert_allclose(
            self.shown(self.points),
            self.shown(moved, matrix, width),
            atol=1e-9,
        )

    def test_degenerate_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            _human_display_frame(np.zeros((21, 3)), self.directions)

    def test_display_without_pad_directions(self):
        matrix, width, _ = _human_display_frame(self.points, None)
        self.assertTrue(np.isfinite(matrix).all())
        self.assertAlmostEqual(width, self.width)
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
