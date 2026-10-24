from dataclasses import dataclass

import numpy as np

EPS = 1e-8

FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
SKELETON_EDGES = tuple(
    edge for chain in FINGER_CHAINS for edge in zip(chain[:-1], chain[1:])
)

FOUR_FINGER_CHAINS = (
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)


@dataclass(frozen=True, slots=True)
class InitialJointAngles:
    """Human joint-angle observations in radians for retarget initialization.

    ``four_fingers`` rows are Index, Middle, Ring, Little and columns are MCP
    Spread, MCP F-E, PIP, DIP. ``thumb_bends`` contains the first and second
    internal bends along the standard thumb chain. ``four_finger_space`` is
    ``"human"`` when retargeting must apply the WiLoR human-to-robot mapping,
    or ``"robot"`` when the values are already direct robot-angle targets.
    """

    four_fingers: np.ndarray
    thumb_bends: np.ndarray
    four_finger_space: str = "human"

    def __post_init__(self):
        four = np.asarray(self.four_fingers, dtype=float)
        thumb = np.asarray(self.thumb_bends, dtype=float)
        if four.shape != (4, 4) or not np.isfinite(four).all():
            raise ValueError("Four-finger initial angles must be finite with shape (4, 4)")
        if thumb.shape != (2,) or not np.isfinite(thumb).all():
            raise ValueError("Thumb initial bends must be finite with shape (2,)")
        if self.four_finger_space not in ("human", "robot"):
            raise ValueError("Four-finger angle space must be 'human' or 'robot'")
        object.__setattr__(self, "four_fingers", four.copy())
        object.__setattr__(self, "thumb_bends", thumb.copy())

    def copy(self):
        return InitialJointAngles(
            self.four_fingers, self.thumb_bends, self.four_finger_space
        )


@dataclass(slots=True)
class InputFrame:
    timestamp: float
    points: np.ndarray | None
    handedness: str | None
    ready: bool
    status: str
    # Outward unit normals for Thumb, Index, Middle, Ring, Little finger pads.
    # Coordinates use the same source frame as points. None when the source
    # cannot observe fingertip pointing (MediaPipe); pad losses are then off.
    finger_pad_directions: np.ndarray | None = None
    preview: np.ndarray | None = None
    initial_joint_angles: InitialJointAngles | None = None
    # Direct Hand0-to-Middle-tip distance before input filtering.
    raw_palm_length: float | None = None
    # Index-to-Little MCP distance in source metres before input filtering.
    raw_palm_width: float | None = None

    @classmethod
    def empty(cls, timestamp, status, preview=None):
        return cls(timestamp, None, None, False, status, preview=preview)

    def __post_init__(self):
        if self.points is None:
            if self.finger_pad_directions is not None:
                raise ValueError("Finger-pad directions require hand landmarks")
            if self.initial_joint_angles is not None:
                raise ValueError("Initial joint angles require hand landmarks")
            if self.raw_palm_length is not None:
                raise ValueError("Raw palm length requires hand landmarks")
            if self.raw_palm_width is not None:
                raise ValueError("Raw palm width requires hand landmarks")
            return
        points = np.asarray(self.points, float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError("Hand landmarks must be finite with shape (21, 3)")
        if self.finger_pad_directions is not None:
            directions = np.asarray(self.finger_pad_directions, float)
            if directions.shape != (5, 3) or not np.isfinite(directions).all():
                raise ValueError("Finger-pad directions must be finite with shape (5, 3)")
            if not np.allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-5):
                raise ValueError("Finger-pad directions must be unit vectors")
        if self.ready and self.initial_joint_angles is None:
            raise ValueError("Ready input frames require initial joint angles")
        if self.initial_joint_angles is not None and not isinstance(
            self.initial_joint_angles, InitialJointAngles
        ):
            raise ValueError("initial_joint_angles must be InitialJointAngles")
        if self.raw_palm_length is not None and not np.isfinite(
            self.raw_palm_length
        ):
            raise ValueError("Raw palm length must be finite")
        if self.raw_palm_width is not None and (
            not np.isfinite(self.raw_palm_width) or self.raw_palm_width <= 0
        ):
            raise ValueError("Raw palm width must be finite and positive")


def _unit(vector, fallback=(1.0, 0.0, 0.0)):
    norm = np.linalg.norm(vector)
    if norm > EPS:
        return vector / norm
    fallback = np.asarray(fallback, float)
    return fallback / max(np.linalg.norm(fallback), EPS)


def compute_cmc_frame(points):
    """Return the shared thumb-CMC retarget frame for standard-21 points."""
    points = np.asarray(points, float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Human landmarks must be finite with shape (21, 3)")
    longitudinal = points[9] - points[0]
    if np.linalg.norm(longitudinal) <= EPS:
        raise ValueError("Degenerate human palm longitudinal axis")
    x_axis = _unit(longitudinal)
    side = points[17] - points[5]
    y_axis = side - x_axis * np.dot(side, x_axis)
    if np.linalg.norm(y_axis) <= EPS:
        raise ValueError("Degenerate human palm side axis")
    y_axis = _unit(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    if np.linalg.norm(z_axis) <= EPS:
        raise ValueError("Degenerate human palm normal")
    z_axis = _unit(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    return points[1].copy(), np.column_stack((x_axis, y_axis, z_axis))


def hand0_middle_tip_distance(points):
    """Return the direct distance from Hand0 (P0) to Middle tip (P12)."""

    points = np.asarray(points, float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Hand landmarks must be finite with shape (21, 3)")
    distance = float(np.linalg.norm(points[12] - points[0]))
    if distance <= EPS:
        raise ValueError("Degenerate Hand0-to-Middle-tip distance")
    return distance


def _segment_directions(points):
    vectors = np.diff(np.asarray(points, float), axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= EPS):
        raise ValueError("Zero-length human finger segment")
    return vectors / lengths[:, None]


def initial_joint_angles_from_points(points):
    """Solve WiLoR-compatible human angles from filtered standard-21 points."""
    points = np.asarray(points, float)
    origin, rotation = compute_cmc_frame(points)
    local = (points - origin) @ rotation
    four = []
    for chain in FOUR_FINGER_CHAINS:
        directions = _segment_directions(local[np.asarray(chain)])
        proximal = directions[0]
        planar = np.linalg.norm(proximal[:2])
        if planar <= EPS:
            raise ValueError("Degenerate four-finger proximal projection")
        four.append((
            np.arctan2(proximal[1], proximal[0]),
            np.arctan2(-proximal[2], planar),
            np.arccos(np.clip(proximal @ directions[1], -1.0, 1.0)),
            np.arccos(np.clip(directions[1] @ directions[2], -1.0, 1.0)),
        ))
    thumb = _segment_directions(points[1:5])
    thumb_bends = np.asarray((
        np.arctan2(
            np.linalg.norm(np.cross(thumb[0], thumb[1])),
            np.clip(thumb[0] @ thumb[1], -1.0, 1.0),
        ),
        np.arctan2(
            np.linalg.norm(np.cross(thumb[1], thumb[2])),
            np.clip(thumb[1] @ thumb[2], -1.0, 1.0),
        ),
    ))
    return InitialJointAngles(np.asarray(four), thumb_bends)
