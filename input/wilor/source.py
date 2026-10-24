"""Monocular WiLoR ONNX input."""

from dataclasses import dataclass

import cv2
import numpy as np

from config import (
    WILOR_ASSET_DIR,
    CAMERA_DEVICE,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    WILOR_CONFIDENCE,
    WILOR_CROP_FACTOR,
    WILOR_DETECT_EVERY,
    WILOR_DEVICE_ID,
    WILOR_IOU,
    WILOR_THUMB_PAD_ROTATION_DEG,
    WILOR_DIRECTION_FILTER,
    WILOR_POINT_FILTER,
)
from input.frame import (
    InputFrame, initial_joint_angles_from_points,
    hand0_middle_tip_distance,
)
from one_euro import OneEuro
from input.camera import OpenCVCamera

MEAN = np.array((0.485, 0.456, 0.406), np.float32)[:, None, None]
STD = np.array((0.229, 0.224, 0.225), np.float32)[:, None, None]
TIP_VERTICES = np.array((744, 320, 443, 554, 671))
JOINT_ORDER = np.array((0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18,
                        10, 11, 12, 19, 7, 8, 9, 20))
# Outward-facing volar triangles, Thumb to Little. Thumb was manually selected.
PAD_FACE_ROWS = (
    np.array((1376, 1377, 1378, 1358, 1357, 1373, 1374, 1375, 1301, 1302,
              1306, 1305, 1379, 1380, 1363, 1364, 1365, 1366, 1353, 1354,
              1304, 1339, 1340, 1355, 1356, 1303, 1351, 1352)),
    np.array((500, 464, 497, 498)),
    np.array((698, 697, 731, 732)),
    np.array((950, 951, 952, 972)),
    np.array((1184, 1183, 1182, 1204)),
)


def make_session(path, device_id):
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("Install onnxruntime-gpu to use WiLoR") from error
    if hasattr(ort, "set_default_logger_severity"):
        ort.set_default_logger_severity(3)
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("WiLoR requires ONNX Runtime CUDAExecutionProvider")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(path), options,
        providers=[("CUDAExecutionProvider", {"device_id": device_id}),
                   "CPUExecutionProvider"],
    )
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("WiLoR CUDA session creation failed")
    return session


def letterbox(frame, size=640):
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(frame, (round(width * scale), round(height * scale)))
    x = (size - resized.shape[1]) / 2
    y = (size - resized.shape[0]) / 2
    left, top = round(x - 0.1), round(y - 0.1)
    canvas = cv2.copyMakeBorder(
        resized, top, size - resized.shape[0] - top,
        left, size - resized.shape[1] - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None]), scale, left, top


def box_iou(one, many):
    intersection_size = np.maximum(np.minimum(one[2:], many[:, 2:]) -
                                   np.maximum(one[:2], many[:, :2]), 0)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]
    one_area = np.prod(np.maximum(one[2:] - one[:2], 0))
    many_area = np.prod(np.maximum(many[:, 2:] - many[:, :2], 0), axis=1)
    return intersection / np.maximum(one_area + many_area - intersection, 1e-9)


def nms(boxes, scores, threshold):
    order, keep = scores.argsort()[::-1], []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        order = order[1:][box_iou(boxes[current], boxes[order[1:]]) <= threshold]
    return keep


class Detector:
    def __init__(self, session, confidence, iou):
        self.session, self.confidence, self.iou = session, confidence, iou
        self.dtype = np.float16 if "float16" in session.get_inputs()[0].type else np.float32

    def __call__(self, frame):
        tensor, scale, pad_x, pad_y = letterbox(frame)
        values = self.session.run(None, {"images": tensor.astype(self.dtype) / 255})[0][0].T.astype(np.float32)
        values = values[np.isfinite(values).all(1)]
        scores = values[:, 4:6].max(1)
        labels = values[:, 4:6].argmax(1).astype(np.int32)
        valid = scores >= self.confidence
        values, scores, labels = values[valid], scores[valid], labels[valid]
        if not len(values):
            return None
        boxes = np.c_[values[:, :2] - values[:, 2:4] / 2,
                      values[:, :2] + values[:, 2:4] / 2]
        selected = []
        for label in (0, 1):
            indices = np.flatnonzero(labels == label)
            selected.extend(indices[nms(boxes[indices], scores[indices], self.iou)])
        height, width = frame.shape[:2]
        if not selected:
            return None
        index = max(selected, key=lambda item: scores[item])
        box = boxes[index].astype(np.float32)
        box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
        box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
        return None if np.any(box[2:] - box[:2] < 4) else Detection(
            box, int(labels[index]), float(scores[index])
        )


@dataclass
class Detection:
    box: np.ndarray
    handedness: int
    score: float


def crop_hand(frame, detection, factor, dtype):
    height, width = frame.shape[:2]
    center = (detection.box[:2] + detection.box[2:]) / 2
    box_width, box_height = factor * (detection.box[2:] - detection.box[:2])
    box_height = max(box_height, box_width * 256 / 192)
    box_width = max(box_width, box_height * 192 / 256)
    box_size = max(box_width, box_height, 1)
    image, center_x = frame, center[0]
    if not detection.handedness:
        image, center_x = frame[:, ::-1], width - center_x - 1
    scale = 256 / box_size
    transform = np.array(((scale, 0, 128 - scale * center_x),
                          (0, scale, 128 - scale * center[1])), np.float32)
    patch = cv2.warpAffine(image, transform, (256, 256))
    rgb = patch[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255
    crop = ((rgb - MEAN) / STD)[:, :, 32:-32].astype(dtype)
    return np.ascontiguousarray(crop[None]), center, box_size


class Reconstructor:
    def __init__(self, session, factor):
        self.session, self.factor = session, factor
        self.dtype = np.float16 if "float16" in session.get_inputs()[0].type else np.float32

    def __call__(self, frame, detection):
        crop, center, box_size = crop_hand(
            frame, detection, self.factor, self.dtype
        )
        vertices, camera = self.session.run(None, {"images": crop})
        vertices, camera = vertices.astype(np.float32), camera.astype(np.float32)
        if (vertices.shape != (1, 778, 3) or camera.shape != (1, 3)
                or not np.isfinite(vertices).all() or not np.isfinite(camera).all()):
            raise ValueError("Invalid WiLoR model output")
        vertices, camera = vertices[0], camera[0]
        camera[1] *= 1 if detection.handedness else -1
        width, height = frame.shape[1], frame.shape[0]
        focal = 5000 / 256 * max(width, height)
        scaled_box = box_size * camera[0]
        if scaled_box <= 0:
            raise ValueError("Invalid WiLoR camera scale")
        translation = np.array((
            2 * (center[0] - width / 2) / scaled_box + camera[1],
            2 * (center[1] - height / 2) / scaled_box + camera[2],
            2 * focal / scaled_box,
        ), np.float32)
        return vertices, translation.astype(np.float32)


def mesh_hand(vertices, right, regressor, faces):
    vertices = np.asarray(vertices, np.float32)
    if vertices.shape != (778, 3) or not np.isfinite(vertices).all():
        raise ValueError("WiLoR vertices must be finite with shape (778, 3)")
    joints = np.vstack((regressor @ vertices, vertices[TIP_VERTICES]))[JOINT_ORDER]
    triangles = (vertices[faces[rows]] for rows in PAD_FACE_ROWS)
    normals = np.asarray([
        np.cross(group[:, 1] - group[:, 0], group[:, 2] - group[:, 0]).sum(0)
        for group in triangles
    ])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(lengths < 1e-8):
        raise ValueError("Degenerate WiLoR finger-pad surface")
    normals /= lengths
    if not right:
        joints[:, 0] *= -1
        normals[:, 0] *= -1
    raw_palm_length = hand0_middle_tip_distance(joints)
    raw_palm_width = float(np.linalg.norm(joints[5] - joints[17]))
    points = joints.astype(float)
    directions = normals.astype(float)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions[0] = rotate(
        directions[0], points[4] - points[3],
        np.radians(WILOR_THUMB_PAD_ROTATION_DEG) * (-1 if right else 1),
    )
    return (
        points, directions, "Right" if right else "Left",
        raw_palm_length, raw_palm_width,
    )


def rotate(vector, axis, angle):
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    cosine, sine = np.cos(angle), np.sin(angle)
    result = (vector * cosine + np.cross(axis, vector) * sine
              + axis * np.dot(axis, vector) * (1 - cosine))
    return result / max(np.linalg.norm(result), 1e-8)


def draw_mesh(frame, vertices, translation, right, faces):
    vertices = np.asarray(vertices, np.float32).copy()
    vertices[:, 0] *= 1 if right else -1
    points_3d = vertices + np.asarray(translation, np.float32)
    depth = np.maximum(points_3d[:, 2], 1e-5)
    focal = 5000 / 256 * max(frame.shape[:2])
    points = np.stack((focal * points_3d[:, 0] / depth + frame.shape[1] / 2,
                       focal * points_3d[:, 1] / depth + frame.shape[0] / 2), axis=1)
    triangles = np.clip(points[faces], -32768, 32767).astype(np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, triangles, (0, 235, 255), lineType=cv2.LINE_AA)
    cv2.polylines(overlay, triangles, True, (0, 90, 130), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, dst=frame)


def preview(frame, detection=None, mesh=None):
    frame = frame.copy()
    if mesh is not None:
        draw_mesh(frame, *mesh)
    if detection is not None:
        x1, y1, x2, y2 = detection.box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 230, 255), 2)
        cv2.putText(frame, "R" if detection.handedness else "L", (x1, max(20, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 230, 255), 2)
    canvas = np.zeros((360, 1280, 3), np.uint8)
    canvas[:, 400:880] = cv2.resize(frame, (480, 360))
    return canvas


class WilorSource:
    def __init__(self):
        paths = {name: WILOR_ASSET_DIR / name for name in (
            "detector_fp16.onnx", "wilor_fp16.onnx", "joint_regressor.npy",
            "mano_faces.npy",
        )}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing WiLoR assets: " + ", ".join(missing))
        detector = make_session(paths["detector_fp16.onnx"], WILOR_DEVICE_ID)
        model = make_session(paths["wilor_fp16.onnx"], WILOR_DEVICE_ID)
        self.detector = Detector(detector, WILOR_CONFIDENCE, WILOR_IOU)
        self.reconstructor = Reconstructor(model, WILOR_CROP_FACTOR)
        self.regressor = np.load(paths["joint_regressor.npy"], allow_pickle=False)
        self.faces = np.load(paths["mano_faces.npy"], allow_pickle=False)
        if self.regressor.shape != (16, 778) or self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("Invalid WiLoR geometry assets")
        self.detection = None
        self.camera = OpenCVCamera(
            CAMERA_DEVICE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
        )
        self.point_filter = OneEuro(*WILOR_POINT_FILTER)
        self.direction_filter = OneEuro(*WILOR_DIRECTION_FILTER)
        self.handedness = None
        self.fps = self.fps_timestamp = None
        self.sequence = self.frame_index = 0
        print(f"Input: WiLoR {self.camera.description}; CUDA")

    def _reset_filters(self):
        self.point_filter.reset()
        self.direction_filter.reset()
        self.handedness = self.fps = self.fps_timestamp = None

    def _filter(self, points, directions, handedness, timestamp):
        if handedness != self.handedness:
            self._reset_filters()
            self.handedness = handedness
        points = self.point_filter(points, timestamp)
        directions = self.direction_filter(directions, timestamp)
        lengths = np.linalg.norm(directions, axis=1, keepdims=True)
        if np.any(lengths < 1e-8):
            raise ValueError("degenerate filtered direction")
        return points, directions / lengths

    def _update_fps(self, timestamp):
        if self.fps_timestamp is not None:
            current = 1 / max(timestamp - self.fps_timestamp, 1e-6)
            self.fps = current if self.fps is None else .9 * self.fps + .1 * current
        self.fps_timestamp = timestamp
        return 0.0 if self.fps is None else self.fps

    def read(self):
        self.sequence, frame, timestamp = self.camera.read(self.sequence)
        detect = self.frame_index % WILOR_DETECT_EVERY == 0 or self.detection is None
        if detect:
            self.detection = self.detector(frame)
        self.frame_index += 1
        if self.detection is None:
            self._reset_filters()
            return InputFrame.empty(timestamp, "WILOR WAITING", preview(frame))
        try:
            vertices, translation = self.reconstructor(frame, self.detection)
            (
                points, directions, handedness, raw_palm_length, raw_palm_width,
            ) = mesh_hand(
                vertices, self.detection.handedness, self.regressor, self.faces
            )
        except ValueError as error:
            self._reset_filters()
            return InputFrame.empty(
                timestamp, f"WILOR INVALID: {error}", preview(frame, self.detection)
            )
        try:
            points, directions = self._filter(
                points, directions, handedness, timestamp
            )
            initial_angles = initial_joint_angles_from_points(points)
        except ValueError as error:
            self._reset_filters()
            return InputFrame.empty(
                timestamp, f"WILOR INVALID: filtered geometry: {error}",
                preview(frame, self.detection),
            )
        image = preview(frame, self.detection, (
            vertices, translation, self.detection.handedness, self.faces
        ))
        fps = self._update_fps(timestamp)
        return InputFrame(
            timestamp, points, handedness, True,
            f"WILOR {handedness} {self.detection.score:.2f} · {fps:.1f} FPS",
            finger_pad_directions=directions, preview=image,
            initial_joint_angles=initial_angles,
            raw_palm_length=raw_palm_length,
            raw_palm_width=raw_palm_width,
        )

    def close(self):
        self.camera.close()
