"""MANUS positions to the common input frame."""

from dataclasses import dataclass

import numpy as np

from config import (
    MANUS_PAD_LOCAL_AXIS, MANUS_THUMB_PAD_ROTATION_DEG,
)
from input.frame import hand0_middle_tip_distance


# Official 25-node layout fallback. Non-thumb metacarpals 5/10/15/20
# are deliberately omitted; Thumb Metacarpal (standard Thumb CMC) is retained.
MANUS_TO_STANDARD21 = np.array(
    (0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24),
    dtype=np.intp,
)
_CHAIN_NAMES = {
    5: "thumb",
    6: "index",
    7: "middle",
    8: "ring",
    9: "pinky",
    13: "hand",
}
_JOINT_NAMES = {0: "invalid", 1: "metacarpal", 2: "proximal", 3: "intermediate", 4: "distal", 5: "tip"}
_SIDE_NAMES = {1: "Left", 2: "Right"}


@dataclass(frozen=True, slots=True)
class AdaptedManusFrame:
    points: np.ndarray
    directions: np.ndarray
    mapping: np.ndarray
    mapping_source: str
    raw_palm_length: float
    raw_palm_width: float


def _field(value, name):
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _enum_name(value, numeric_names):
    if isinstance(value, (int, np.integer)):
        return numeric_names.get(int(value), str(int(value)))
    name = getattr(value, "name", value)
    name = str(name).lower()
    for prefix in ("chaintype_finger", "chaintype_", "fingerjointtype_", "side_"):
        name = name.removeprefix(prefix)
    return name


def handedness_from_node_info(node_info):
    """Return Left/Right from NodeInfo.side, or None if it is unavailable."""
    items = () if node_info is None else node_info
    sides = {
        _SIDE_NAMES.get(int(side), None)
        if isinstance(side, (int, np.integer))
        else str(getattr(side, "name", side)).removeprefix("Side_").title()
        for item in items
        if (side := _field(item, "side")) is not None
    }
    sides &= {"Left", "Right"}
    return sides.pop() if len(sides) == 1 else None


def _chain_order(entries):
    """Order (row, node_id, parent_id, joint) entries by their hierarchy."""
    remaining = list(entries)
    ids = {entry[1] for entry in remaining}
    ordered = []
    while remaining:
        candidates = [entry for entry in remaining if entry[2] not in ids or entry[2] in {item[1] for item in ordered}]
        if not candidates:
            candidates = remaining
        entry = min(candidates, key=lambda item: item[0])
        ordered.append(entry)
        remaining.remove(entry)
    return ordered


def semantic_standard21_mapping(node_info, node_ids=None):
    """Build standard-21 row indices from MANUS NodeInfo semantics.

    NodeInfo is authoritative when complete. The caller may then fall back to
    ``MANUS_TO_STANDARD21`` for the documented 25-row layout.
    """
    if node_info is None or len(node_info) == 0:
        raise ValueError("MANUS NodeInfo is unavailable")
    node_ids = None if node_ids is None else list(map(int, node_ids))
    row_by_id = None if node_ids is None else {node_id: row for row, node_id in enumerate(node_ids)}
    groups = {name: [] for name in ("thumb", "index", "middle", "ring", "pinky", "hand")}
    for fallback_row, item in enumerate(node_info):
        node_id = int(_field(item, "nodeId"))
        row = fallback_row if row_by_id is None else row_by_id.get(node_id)
        if row is None:
            continue
        chain = _enum_name(_field(item, "chainType"), _CHAIN_NAMES)
        joint = _enum_name(_field(item, "fingerJointType"), _JOINT_NAMES)
        if chain in groups:
            groups[chain].append((row, node_id, int(_field(item, "parentId")), joint))
    hands = _chain_order(groups["hand"])
    if not hands:
        raise ValueError("NodeInfo has no Hand/Wrist node")
    mapping = [hands[0][0]]
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        chain = _chain_order(groups[finger])
        if finger != "thumb":
            chain = [entry for entry in chain if entry[3] != "metacarpal"]
        if len(chain) != 4:
            raise ValueError(f"NodeInfo {finger} chain does not map to four standard joints")
        mapping.extend(entry[0] for entry in chain)
    result = np.asarray(mapping, dtype=np.intp)
    if result.shape != (21,) or len(set(result.tolist())) != 21:
        raise ValueError("NodeInfo produced an invalid standard-21 mapping")
    return result


def convert_manus25_to_standard21(points25, mapping=None, scale_to_m=1.0):
    """Validate and map finite MANUS 25x3 positions to standard 21x3 meters."""
    points = np.asarray(points25, dtype=float)
    if points.shape != (25, 3):
        raise ValueError("MANUS positions must have shape (25, 3)")
    if not np.isfinite(points).all():
        raise ValueError("MANUS positions must be finite")
    scale = float(scale_to_m)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("MANUS position scale must be finite and positive")
    indices = MANUS_TO_STANDARD21 if mapping is None else np.asarray(mapping, dtype=np.intp)
    if indices.shape != (21,) or np.any(indices < 0) or np.any(indices >= 25):
        raise ValueError("MANUS standard-21 mapping must contain 21 valid rows")
    return points[indices].copy() * scale


def _rotate_wxyz(quaternions, vector):
    quaternions = np.asarray(quaternions, float)
    if quaternions.shape != (5, 4) or not np.isfinite(quaternions).all():
        raise ValueError("MANUS tip rotations must be finite with shape (5, 4)")
    lengths = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(lengths < 1e-8):
        raise ValueError("MANUS tip rotations must be nonzero")
    normalized = quaternions / lengths
    w, xyz = normalized[:, :1], normalized[:, 1:]
    vectors = np.broadcast_to(np.asarray(vector, float), (5, 3))
    return vectors + 2 * w * np.cross(xyz, vectors) + 2 * np.cross(
        xyz, np.cross(xyz, vectors)
    )


def _rotate(vector, axis, angle):
    """Rotate and renormalize a direction around a source-frame axis."""
    axis = np.asarray(axis, float)
    axis /= max(np.linalg.norm(axis), 1e-8)
    cosine, sine = np.cos(angle), np.sin(angle)
    result = (
        vector * cosine
        + np.cross(axis, vector) * sine
        + axis * np.dot(axis, vector) * (1 - cosine)
    )
    return result / max(np.linalg.norm(result), 1e-8)


def adapt_raw_skeleton(
    positions25,
    *,
    rotations_wxyz,
    handedness=None,
    node_info=None,
    node_ids=None,
    scale_to_m=1.0,
):
    """Adapt one WORLD/GLOBAL Raw Skeleton."""
    if node_info is not None and len(node_info) > 0:
        mapping = semantic_standard21_mapping(node_info, node_ids)
        mapping_source = "NodeInfo"
    else:
        mapping, mapping_source = MANUS_TO_STANDARD21, "official-25 fallback"

    standard_world = convert_manus25_to_standard21(positions25, mapping, scale_to_m)
    raw_palm_length = hand0_middle_tip_distance(standard_world)
    rotations = np.asarray(rotations_wxyz, float)
    if rotations.shape != (25, 4):
        raise ValueError("MANUS rotations must have shape (25, 4)")
    tip_directions = _rotate_wxyz(
        rotations[np.asarray(mapping)[[4, 8, 12, 16, 20]]], MANUS_PAD_LOCAL_AXIS
    )
    points, directions = standard_world, tip_directions
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    if handedness in ("Left", "Right"):
        angle = np.radians(MANUS_THUMB_PAD_ROTATION_DEG)
        directions[0] = _rotate(
            directions[0], points[4] - points[3],
            angle if handedness == "Left" else -angle,
        )
    return AdaptedManusFrame(
        points=points,
        directions=directions,
        mapping=np.asarray(mapping).copy(),
        mapping_source=mapping_source,
        raw_palm_length=raw_palm_length,
        raw_palm_width=float(np.linalg.norm(standard_world[5] - standard_world[17])),
    )
