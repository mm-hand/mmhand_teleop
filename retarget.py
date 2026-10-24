"""Retarget source-frame human-hand observations to the 21-DOF MMHand URDF."""

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import threading
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, minimize

import config as C
from input.frame import InitialJointAngles, compute_cmc_frame

EPS = 1e-9
ROBOT_TIPS = ("5-tip_Link", "1-tip_Link", "2-tip_Link", "3-tip_Link", "4-tip_Link")
ROBOT_MIDDLE_FINGERTIP = "finger_2_fingertip_1"
ROBOT_FINGERS = tuple(
    (
        f"finger_{finger}_proximal_phalanx_1",
        f"finger_{finger}_distal_phalanx_1",
        f"finger_{finger}_fingertip_1",
        f"{finger}-tip_Link",
    )
    for finger in range(1, 5)
)
ROBOT_FINGER_JOINTS = tuple(
    (
        f"{name}_MCP_AA",
        f"{name}_MCP_FE",
        f"finger_{finger}_distal_phalanx_1_PIP_Joint",
        f"finger_{finger}_fingertip_1_DIP_Joint",
    )
    for name, finger in zip(("Index", "Middle", "Ring", "Little"), range(1, 5))
)


class _EvaluationLimit(Exception):
    """Stop SLSQP after the configured number of distinct evaluations."""

    pass


def _values(text, default):
    """Parse a whitespace-separated URDF vector with a fallback value."""

    return np.fromstring(text if text is not None else default, sep=" ")


def _unit(vector):
    """Normalize a vector while preventing division by a near-zero norm."""

    return vector / max(np.linalg.norm(vector), EPS)


def _checked_unit(vector):
    """Normalize a vector and reject degenerate geometry."""

    norm = np.linalg.norm(vector)
    if norm <= EPS:
        raise ValueError("Degenerate human palm frame")
    return vector / norm


def _orthogonal(vector):
    """Construct a stable unit vector orthogonal to the input vector."""

    basis = np.eye(3)[np.argmin(np.abs(vector))]
    return _checked_unit(np.cross(vector, basis))


def human_thumb_geometry(points):
    """Return thumb segments, unsigned 3D bend angles, and matching arc axes."""
    points = np.asarray(points, float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Thumb angles require 21 finite points")
    segments = _directions(points[1:5])
    angles, axes = [], []
    for first, second in zip(segments[:-1], segments[1:]):
        cross = np.cross(first, second)
        norm = np.linalg.norm(cross)
        angles.append(np.arctan2(norm, np.clip(np.dot(first, second), -1, 1)))
        axes.append(_orthogonal(first) if norm <= EPS else cross / norm)
    return segments, np.asarray(angles), np.asarray(axes)


def human_vector_features(points, directions=None):
    """Return the spatial and pad-direction targets used by retarget loss.

    ``pads`` is None when the source cannot observe fingertip pointing.
    """

    points = np.asarray(points, float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Retargeting requires 21 finite points")
    origin, rotation = compute_cmc_frame(points)
    local = (points - origin) @ rotation
    thumbs = local[4:5]
    relative = local[[8, 12, 16, 20]] - local[4]
    if directions is None:
        return origin, rotation, thumbs, relative, None
    directions = np.asarray(directions, float)
    if directions.shape != (5, 3) or not np.isfinite(directions).all():
        raise ValueError("Invalid finger-pad directions")
    pads = np.asarray([
        _checked_unit(direction @ rotation) for direction in directions
    ])
    return origin, rotation, thumbs, relative, pads


def _rotation(axis, angle):
    """Build a 3D Rodrigues rotation matrix for one axis-angle motion."""

    axis = np.asarray(axis, float)
    cross = np.array(((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0)))
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def _origin(node):
    """Convert a URDF origin element into a homogeneous transform."""

    xyz = _values(node.get("xyz"), "0 0 0")
    roll, pitch, yaw = _values(node.get("rpy"), "0 0 0")
    transform = np.eye(4)
    transform[:3, :3] = _rotation((0, 0, 1), yaw) @ _rotation((0, 1, 0), pitch) @ _rotation((1, 0, 0), roll)
    transform[:3, 3] = xyz
    return transform


def _stl_vertices(path):
    """Load triangle vertices from a binary or ASCII STL file."""

    data = Path(path).read_bytes()
    if len(data) >= 84:
        count = int.from_bytes(data[80:84], "little")
        if len(data) == 84 + 50 * count:
            records = np.frombuffer(
                data,
                dtype=np.dtype([
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attribute", "<u2"),
                ]),
                count=count,
                offset=84,
            )
            return records["vertices"].reshape(-1, 3).astype(float)
    vertices = [
        tuple(map(float, line.split()[1:4]))
        for line in data.decode("ascii").splitlines()
        if line.lstrip().startswith("vertex ")
    ]
    if not vertices:
        raise ValueError(f"STL contains no vertices: {path}")
    return np.asarray(vertices, float)


def _directions(points, jacobians=None):
    """Return unit segment directions and optional normalized Jacobians."""

    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= EPS):
        raise ValueError("Zero-length retarget segment")
    directions = vectors / lengths[:, None]
    if jacobians is None:
        return directions
    vector_jacobians = np.diff(jacobians, axis=0)
    projectors = np.eye(3) - directions[:, :, None] * directions[:, None, :]
    direction_jacobians = (
        np.einsum("kij,kjn->kin", projectors, vector_jacobians)
        / lengths[:, None, None]
    )
    return directions, direction_jacobians


class RobotModel:
    """Provide MMHand URDF kinematics, joint metadata, and target features."""

    def __init__(self, urdf_path=C.URDF_PATH):
        """Parse the URDF and precompute limits, topology, and neutral geometry."""

        urdf_path = Path(urdf_path)
        root = ET.parse(urdf_path).getroot()
        self.joints, self.children = {}, defaultdict(list)
        limits = {}
        for node in root.findall("joint"):
            origin, axis, limit = node.find("origin"), node.find("axis"), node.find("limit")
            joint = {
                "parent": node.find("parent").get("link"),
                "child": node.find("child").get("link"),
                "origin": _origin(origin) if origin is not None else np.eye(4),
                "axis": _unit(_values(axis.get("xyz"), "1 0 0")) if axis is not None else np.array((1.0, 0, 0)),
            }
            name = node.get("name")
            self.joints[name] = joint
            self.children[joint["parent"]].append(name)
            if node.get("type") == "revolute":
                limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
        self.names = tuple(name for name in C.ROBOT_JOINT_NAMES if name in limits)
        if self.names != C.ROBOT_JOINT_NAMES:
            raise ValueError(f"Unexpected MMHand joints: {self.names}")
        self.index = {name: index for index, name in enumerate(self.names)}
        for name, joint in self.joints.items():
            joint["index"] = self.index.get(name)
        self.order, stack = [], ["base_link"]
        while stack:
            parent = stack.pop()
            for name in self.children[parent]:
                self.order.append(name)
                stack.append(self.joints[name]["child"])
        ancestors = {"base_link": ()}
        for name in self.order:
            joint = self.joints[name]
            indices = ancestors[joint["parent"]]
            if joint["index"] is not None:
                indices += (joint["index"],)
            ancestors[joint["child"]] = indices
        self.influence = np.zeros((len(ROBOT_TIPS), len(self.names), 1))
        for row, name in enumerate(ROBOT_TIPS):
            self.influence[row, list(ancestors[name])] = 1
        self.lower, self.upper = np.asarray([limits[name] for name in self.names]).T
        self.thumb = np.arange(16, 21)
        self.finger_joints = np.asarray(
            [[self.index[name] for name in names] for names in ROBOT_FINGER_JOINTS]
        )
        if np.any(self.upper - self.lower <= EPS):
            raise ValueError("MMHand revolute joints must have nonzero ranges")
        self.seed = np.clip(np.zeros(len(self.names)), self.lower, self.upper)
        transforms = self.fk(self.seed)
        self.palm_frame = transforms["base_link"][:3, :3].copy()
        self.finger_neutral, self.finger_axis = [], []
        for chain, indices in zip(ROBOT_FINGERS, self.finger_joints):
            direction = (
                transforms[chain[1]][:3, 3] - transforms[chain[0]][:3, 3]
            ) @ self.palm_frame
            joint = self.joints[self.names[indices[0]]]
            frame = transforms[joint["parent"]] @ joint["origin"]
            axis = (frame[:3, :3] @ joint["axis"]) @ self.palm_frame
            if abs(axis[2]) <= EPS:
                raise ValueError("MMHand MCP A-A axis must follow the palm z axis")
            self.finger_axis.append(axis[2])
            self.finger_neutral.append(-np.arctan2(direction[1], direction[0]) / axis[2])
        self.finger_neutral = np.asarray(self.finger_neutral)
        self.finger_axis = np.asarray(self.finger_axis)
        cmc = self.joints["Thumb_CMC"]
        cmc_frame = transforms[cmc["parent"]] @ cmc["origin"]
        cmc_origin = cmc_frame[:3, 3]
        self.cmc_axis = _unit(cmc_frame[:3, :3] @ cmc["axis"])
        mcp_aa_origin = transforms[self.joints["Thumb_MCP_AA"]["child"]][:3, 3]
        self.palm_position = cmc_origin + self.cmc_axis * np.dot(
            mcp_aa_origin - cmc_origin, self.cmc_axis
        )
        middle_link = root.find(f"./link[@name='{ROBOT_MIDDLE_FINGERTIP}']")
        visual = None if middle_link is None else middle_link.find("visual")
        mesh = None if visual is None else visual.find("geometry/mesh")
        if mesh is None:
            raise ValueError("MMHand URDF has no Middle fingertip visual mesh")
        mesh_path = urdf_path.parent / mesh.get("filename")
        vertices = _stl_vertices(mesh_path)
        scale = _values(mesh.get("scale"), "1 1 1")
        vertices *= scale
        visual_origin = visual.find("origin")
        visual_transform = (
            _origin(visual_origin) if visual_origin is not None else np.eye(4)
        )
        vertices = (
            vertices @ visual_transform[:3, :3].T
            + visual_transform[:3, 3]
        )
        self.middle_fingertip_surface_local = vertices[np.argmax(vertices[:, 0])]
        self.urdf_palm_length = self.middle_fingertip_surface_distance(self.seed)

    def _forward(self, q, jacobian=False):
        """Run tree forward kinematics and optionally return joint frames."""

        transforms = {"base_link": np.eye(4)}
        origins = np.zeros((len(self.names), 3))
        axes = np.zeros_like(origins)
        for name in self.order:
            joint = self.joints[name]
            child = transforms[joint["parent"]] @ joint["origin"]
            index = joint["index"]
            if index is not None:
                origins[index] = child[:3, 3]
                axes[index] = child[:3, :3] @ joint["axis"]
                motion = np.eye(4)
                motion[:3, :3] = _rotation(joint["axis"], q[index])
                child = child @ motion
            transforms[joint["child"]] = child
        return (transforms, origins, axes) if jacobian else transforms

    def fk(self, q):
        """Return every URDF link transform for a 21-element joint vector."""

        return self._forward(np.asarray(q, float))

    def fingertips(self, q):
        """Return Thumb-to-Little virtual fingertip positions in robot space."""

        transforms = self.fk(q)
        return np.asarray([transforms[name][:3, 3] for name in ROBOT_TIPS])

    def palm_width(self, q):
        """Return the distance between the Index and Little MCP A-A origins."""

        _, origins, _ = self._forward(np.asarray(q, float), True)
        return float(np.linalg.norm(
            origins[self.index["Index_MCP_AA"]]
            - origins[self.index["Little_MCP_AA"]]
        ))

    def middle_fingertip_surface_point(self, q):
        """Return the Middle mesh's distal-most +X vertex in robot space."""

        transform = self.fk(q)[ROBOT_MIDDLE_FINGERTIP]
        return (
            transform[:3, :3] @ self.middle_fingertip_surface_local
            + transform[:3, 3]
        )

    def middle_fingertip_surface_distance(self, q):
        """Return Hand0-to-distal-Middle-mesh distance in robot space."""

        point = self.middle_fingertip_surface_point(q)
        hand0 = self.fk(q)["base_link"][:3, 3]
        return float(np.linalg.norm(point - hand0))

    def fingertip_pads(self, q):
        """Return fingertip positions and outward pad normals, Thumb to Little."""
        transforms = self.fk(q)
        positions = np.asarray([transforms[name][:3, 3] for name in ROBOT_TIPS])
        directions = np.asarray([-transforms[name][:3, 2] for name in ROBOT_TIPS])
        return positions, directions / np.linalg.norm(directions, axis=1, keepdims=True)

    def thumb_joint_frames(self, q):
        """Return J18/J19 origins, axes, and incoming bone directions."""
        _, origins, axes = self._forward(np.asarray(q, float), True)
        indices = np.array((18, 19))
        incoming = origins[indices] - origins[indices - 1]
        return origins[indices], axes[indices], incoming

    def angle_targets(self, initial_angles):
        """Convert input angles into clipped soft robot-angle targets."""

        if not isinstance(initial_angles, InitialJointAngles):
            raise ValueError("Retargeting requires InitialJointAngles")
        q = self.seed.copy()
        for row, indices in enumerate(self.finger_joints):
            human = initial_angles.four_fingers[row]
            if initial_angles.four_finger_space == "robot":
                angles = human
            else:
                aa = self.finger_neutral[row] + human[0] / self.finger_axis[row]
                angles = np.concatenate((
                    (aa,), self.lower[indices[1:]] + human[1:],
                ))
            q[indices] = np.clip(angles, self.lower[indices], self.upper[indices])
        q[18:20] = np.clip(
            initial_angles.thumb_bends, self.lower[18:20], self.upper[18:20]
        )
        return q

    def initial_guess(self, angle_targets, previous=None):
        """Overlay directly observed joints on the seed or previous warm start."""

        targets = np.asarray(angle_targets, float)
        if targets.shape != self.seed.shape or not np.isfinite(targets).all():
            raise ValueError("Robot angle targets must be a finite 21-element vector")
        q = self.seed.copy() if previous is None else np.asarray(previous, float).copy()
        if previous is None:
            q[17] = self.upper[17]
        observed = np.concatenate((self.finger_joints.ravel(), (18, 19)))
        q[observed] = targets[observed]
        return np.clip(q, self.lower, self.upper)

    def initial_angle_targets(self, initial_angles, previous=None):
        """Build the SLSQP initial guess from input angles and a warm start."""

        return self.initial_guess(self.angle_targets(initial_angles), previous)

    def features(self, q):
        """Evaluate all optimization features and their analytic Jacobians."""

        transforms, origins, axes = self._forward(np.asarray(q, float), True)
        world = np.asarray([transforms[name][:3, 3] for name in ROBOT_TIPS])
        delta = world[:, None] - origins
        world_jacobians = np.empty_like(delta)
        world_jacobians[:, :, 0] = axes[:, 1] * delta[:, :, 2] - axes[:, 2] * delta[:, :, 1]
        world_jacobians[:, :, 1] = axes[:, 2] * delta[:, :, 0] - axes[:, 0] * delta[:, :, 2]
        world_jacobians[:, :, 2] = axes[:, 0] * delta[:, :, 1] - axes[:, 1] * delta[:, :, 0]
        world_jacobians *= self.influence
        point_jacobians = np.einsum(
            "ij,knj->kin", self.palm_frame.T, world_jacobians
        )
        points = (world - self.palm_position) @ self.palm_frame
        relative = points[1:] - points[0]
        relative_jacobians = point_jacobians[1:] - point_jacobians[0]

        angles = np.asarray(q, float)[[18, 19], None]
        angle_jacobians = np.zeros((2, 1, len(self.names)))
        angle_jacobians[0, 0, 18] = angle_jacobians[1, 0, 19] = 1
        finger_indices = self.finger_joints.ravel()
        finger_angles = np.asarray(q, float)[finger_indices, None]
        finger_jacobians = np.zeros((len(finger_indices), 1, len(self.names)))
        finger_jacobians[np.arange(len(finger_indices)), 0, finger_indices] = 1
        pad_world = np.asarray([-transforms[name][:3, 2] for name in ROBOT_TIPS])
        pads = pad_world @ self.palm_frame
        pad_jacobians = (
            np.cross(axes[None], pad_world[:, None]) * self.influence
        )
        pad_jacobians = np.einsum(
            "ij,knj->kin", self.palm_frame.T, pad_jacobians
        )
        return (
            points[:1], angles, relative, pads, finger_angles,
        ), (
            point_jacobians[:1], angle_jacobians, relative_jacobians,
            pad_jacobians, finger_jacobians,
        )


class Retargeter:
    """Solve bounded full-hand retargeting and clip the resulting joint vector."""

    def __init__(self, model=None):
        """Initialize the robot model, SLSQP bounds, and warm start."""

        self.model = RobotModel() if model is None else model
        self.bounds = Bounds(self.model.lower, self.model.upper)
        self.q = self.model.seed.copy()
        self.has_previous = False
        self.last_stage_timings_ms = {}
        self.options = {"ftol": C.RETARGET_FTOL, "disp": False}
        weights = np.asarray((
            C.RETARGET_THUMB_PROXIMAL_BEND_WEIGHT,
            C.RETARGET_THUMB_DISTAL_BEND_WEIGHT,
        ), float)
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("Thumb bend weights must be finite and non-negative")

    def _targets(
        self,
        points,
        previous=None,
        finger_pad_directions=None,
        initial_joint_angles=None,
    ):
        """Build spatial, directional, and angle targets in the CMC frame.

        ``finger_pad_directions`` is optional; without it the pad targets are
        None and their losses are disabled.
        """

        _, _, thumb_tip, relative, pads = human_vector_features(
            points, finger_pad_directions
        )
        if not isinstance(initial_joint_angles, InitialJointAngles):
            raise ValueError("Initial joint angles are required for retargeting")
        angle_targets = self.model.angle_targets(initial_joint_angles)
        initial_q = self.model.initial_guess(angle_targets, previous)
        angles = angle_targets[18:20, None]
        return (
            thumb_tip, angles, relative, pads,
            angle_targets[self.model.finger_joints.ravel()], initial_q,
        )

    @staticmethod
    def _term(value, jacobian, target, scale, weight, normalizer=1.0):
        """Compute one weighted mean-squared residual and its joint gradient."""

        error = (value - scale * target) / normalizer
        loss = weight * np.mean(np.sum(error * error, axis=1))
        gradient = (
            2 * weight / (len(error) * normalizer)
            * np.einsum("ki,kin->n", error, jacobian)
        )
        return loss, gradient

    def _losses(self, q, targets):
        """Evaluate the complete weighted objective and per-term diagnostics."""

        values, jacobians = self.model.features(q)
        tip_loss, tip_gradient = self._term(
            values[0], jacobians[0], targets[0],
            C.RETARGET_THUMB_TIP_SCALE, C.RETARGET_THUMB_TIP_WEIGHT,
            C.RETARGET_POSITION_NORMALIZER_M,
        )
        thumb_terms = tuple(
            self._term(
                values[1][row:row + 1], jacobians[1][row:row + 1],
                targets[1][row:row + 1], 1.0, weight,
            )
            for row, weight in enumerate((
                C.RETARGET_THUMB_PROXIMAL_BEND_WEIGHT,
                C.RETARGET_THUMB_DISTAL_BEND_WEIGHT,
            ))
        )
        relative_term = self._term(
            values[2], jacobians[2], targets[2], 1.0,
            C.RETARGET_FINGERTIP_VECTOR_WEIGHT, C.RETARGET_POSITION_NORMALIZER_M,
        )
        # Without fingertip pointing (MediaPipe) both pad terms are exactly 0.
        pads = targets[3]
        pad_gain = 0.0 if pads is None else 1.0
        pad_targets = np.zeros((5, 3)) if pads is None else pads
        pad_term = self._term(
            values[3][:1], jacobians[3][:1], pad_targets[:1],
            1.0, pad_gain * C.RETARGET_THUMB_PAD_WEIGHT / 3,
        )
        finger_pad_terms = tuple(
            self._term(
                values[3][row:row + 1], jacobians[3][row:row + 1],
                pad_targets[row:row + 1], 1.0, pad_gain * weight / 3,
            )
            for row, weight in enumerate((
                C.RETARGET_INDEX_PAD_WEIGHT,
                C.RETARGET_MIDDLE_PAD_WEIGHT,
                C.RETARGET_RING_PAD_WEIGHT,
                C.RETARGET_LITTLE_PAD_WEIGHT,
            ), start=1)
        )
        finger_term = self._term(
            values[4], jacobians[4], targets[4][:, None], 1.0,
            C.RETARGET_FINGER_ANGLE_WEIGHT,
        )
        total = (
            tip_loss + sum(term[0] for term in thumb_terms) + relative_term[0]
            + pad_term[0] + sum(term[0] for term in finger_pad_terms)
            + finger_term[0]
        )
        gradient = (
            tip_gradient + sum(
                (term[1] for term in thumb_terms), np.zeros(len(self.model.names))
            ) + relative_term[1]
            + pad_term[1] + sum(
                (term[1] for term in finger_pad_terms),
                np.zeros(len(self.model.names)),
            ) + finger_term[1]
        )
        return (
            total,
            gradient,
            {
                "thumb_tip": tip_loss,
                "thumb_proximal_bend": thumb_terms[0][0],
                "thumb_distal_bend": thumb_terms[1][0],
                "finger_angles": finger_term[0],
                "fingertip_vectors": relative_term[0],
                "thumb_pad": pad_term[0],
                "finger_pads": sum(term[0] for term in finger_pad_terms),
                "total": total,
            },
        )

    def solve(
        self,
        points,
        timestamp=None,
        finger_pad_directions=None,
        initial_joint_angles=None,
    ):
        """Solve one frame, update the warm start, and return limit-clipped joints."""

        solve_started = time.perf_counter()
        self.last_stage_timings_ms = {}
        points = np.asarray(points, float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            return None
        previous = self.q.copy() if self.has_previous else None
        targets_started = time.perf_counter()
        targets = self._targets(
            points, previous, finger_pad_directions, initial_joint_angles
        )
        targets_ms = (time.perf_counter() - targets_started) * 1000.0
        initial = targets[5]
        cached_x, cached_value, evaluations, best = None, None, 0, None

        def evaluate(x):
            """Cache repeated SLSQP evaluations and retain the best finite state."""

            nonlocal cached_x, cached_value, evaluations, best
            if cached_x is None or not np.array_equal(x, cached_x):
                if evaluations >= C.RETARGET_MAX_EVALUATIONS:
                    raise _EvaluationLimit
                cached_x = np.asarray(x).copy()
                q = cached_x
                cached_value = self._losses(q, targets)
                evaluations += 1
                if np.isfinite(cached_value[0]) and np.isfinite(cached_value[1]).all():
                    candidate = cached_value[0], q, cached_x.copy()
                    if best is None or candidate[0] < best[0]:
                        best = candidate
            return cached_value[:2]

        slsqp_started = time.perf_counter()
        try:
            evaluate(initial)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Values in x were outside bounds during a minimize step, clipping to bounds",
                    category=RuntimeWarning,
                    module=r"scipy\.optimize\._slsqp_py",
                )
                minimize(
                    evaluate, initial, method="SLSQP", jac=True,
                    bounds=self.bounds, options=self.options,
                )
        except _EvaluationLimit:
            pass
        slsqp_ms = (time.perf_counter() - slsqp_started) * 1000.0
        if best is None:
            return None
        _, candidate, _ = best
        self.q = candidate.copy()
        self.has_previous = True
        output = np.clip(candidate, self.model.lower, self.model.upper)
        final_loss_started = time.perf_counter()
        losses = self._losses(output, targets)[2]
        final_loss_ms = (time.perf_counter() - final_loss_started) * 1000.0
        losses = {
            name: None if value is None else float(value)
            for name, value in losses.items()
        }
        solve_total_ms = (time.perf_counter() - solve_started) * 1000.0
        named_ms = targets_ms + slsqp_ms + final_loss_ms
        self.last_stage_timings_ms = {
            "targets": targets_ms,
            "slsqp": slsqp_ms,
            "final_loss": final_loss_ms,
            "solver_overhead": max(0.0, solve_total_ms - named_ms),
            "solve_total": solve_total_ms,
        }
        return output, losses

    def pause(self):
        """Clear warm-start state after tracking is interrupted."""

        self.q = self.model.seed.copy()
        self.has_previous = False
        self.last_stage_timings_ms = {}


class RetargetWorker:
    """Run retargeting on a latest-frame-wins background worker."""

    def __init__(self, retargeter=None, clock=time.monotonic):
        """Start a daemon worker around the supplied or default retargeter."""

        self.retargeter = Retargeter() if retargeter is None else retargeter
        self.clock = clock
        self.model = self.retargeter.model
        self.condition = threading.Condition()
        self.pending = self.result = self.error = None
        self.generation, self.running = 0, True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(
        self,
        points,
        timestamp=None,
        finger_pad_directions=None,
        initial_joint_angles=None,
    ):
        """Deep-copy and enqueue the newest common input frame for solving."""

        with self.condition:
            directions = (
                None if finger_pad_directions is None
                else np.asarray(finger_pad_directions, float).copy()
            )
            self.pending = (
                self.generation, np.asarray(points, float).copy(), timestamp,
                directions,
                None if initial_joint_angles is None else initial_joint_angles.copy(),
                self.clock(),
            )
            self.condition.notify()

    def poll(self):
        """Consume joints, losses, submit-to-solve latency, and stage timings."""

        with self.condition:
            if self.error is not None:
                raise RuntimeError("Retarget worker failed") from self.error
            result, self.result = self.result, None
            return result

    def pause(self):
        """Discard queued work and request a solver-state reset."""

        with self.condition:
            self.generation += 1
            self.pending = self.result = None
            self.condition.notify()

    def _run(self):
        """Process pending frames until shutdown, keeping only the latest result."""

        generation = 0
        while True:
            with self.condition:
                self.condition.wait_for(lambda: not self.running or self.pending is not None)
                if not self.running:
                    return
                (
                    current, points, timestamp, directions, initial_angles,
                    submitted_at,
                ) = self.pending
                self.pending = None
            try:
                if current != generation:
                    self.retargeter.pause()
                    generation = current
                solve_started_at = self.clock()
                solved = self.retargeter.solve(
                    points, timestamp, directions, initial_angles
                )
                solved_at = self.clock()
                if solved is None:
                    result = None
                else:
                    timings = dict(getattr(
                        self.retargeter, "last_stage_timings_ms", {}
                    ))
                    timings.update({
                        "worker_queue": max(
                            0.0, (solve_started_at - submitted_at) * 1000.0
                        ),
                    })
                    if "solve_total" not in timings:
                        timings["solve_total"] = max(
                            0.0, (solved_at - solve_started_at) * 1000.0
                        )
                    result = (
                        *solved,
                        max(0.0, (solved_at - submitted_at) * 1000.0),
                        timings,
                    )
            except Exception as error:
                with self.condition:
                    self.error, self.running = error, False
                    self.condition.notify_all()
                return
            with self.condition:
                if self.running and current == self.generation and result is not None:
                    self.result = result

    def close(self):
        """Stop the worker thread and wait for it to exit."""

        with self.condition:
            self.running, self.pending = False, None
            self.condition.notify()
        self.thread.join()
