"""MANUS Core 3.1.1 input."""

import ctypes
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import numpy as np

from config import (
    MANUS_PINCH_COMPENSATION,
    MANUS_POSITION_SCALE_TO_M,
    MANUS_SDK_BRIDGE_PATH,
    MANUS_SDK_VERSION,
    MANUS_STALE_SECONDS,
    MANUS_THUMB_DIP_TO_PIP_GAIN,
    MANUS_THUMB_PIP_DIP_SCALE,
)
from input.frame import InitialJointAngles, InputFrame
from .adapter import MANUS_TO_STANDARD21, adapt_raw_skeleton, handedness_from_node_info

_MAX_SDK_NODES = 64


class _SdkNode(ctypes.Structure):
    _fields_ = (
        ("node_id", ctypes.c_uint32),
        ("parent_id", ctypes.c_uint32),
        ("side", ctypes.c_int32),
        ("chain_type", ctypes.c_int32),
        ("finger_joint_type", ctypes.c_int32),
        ("position", ctypes.c_float * 3),
        ("rotation_wxyz", ctypes.c_float * 4),
    )


class _SdkFrame(ctypes.Structure):
    _fields_ = (
        ("sequence", ctypes.c_uint64),
        ("publish_time", ctypes.c_uint64),
        ("glove_id", ctypes.c_uint32),
        ("node_count", ctypes.c_uint32),
        ("side", ctypes.c_int32),
        ("nodes", _SdkNode * _MAX_SDK_NODES),
        ("has_ergonomics", ctypes.c_int32),
        ("ergonomics", ctypes.c_float * 20),
    )


class _SdkCalibrationStep(ctypes.Structure):
    _fields_ = (
        ("index", ctypes.c_uint32),
        ("title", ctypes.c_char * 64),
        ("description", ctypes.c_char * 256),
        ("time", ctypes.c_float),
    )


@dataclass(frozen=True, slots=True)
class CalibrationStep:
    index: int
    title: str
    description: str
    duration: float


@dataclass(slots=True)
class ManusPacket:
    positions: np.ndarray
    rotations_wxyz: np.ndarray
    node_ids: list[int]
    node_info: list[dict]
    handedness: str | None
    received_at: float
    calibrated: bool | None = None
    ergonomics: np.ndarray | None = None
    glove_id: int | None = None


def _sdk_frame_to_packet(frame, received_at):
    """Copy the C bridge ABI into Python without changing any transform."""
    count = int(frame.node_count)
    if count <= 0 or count > _MAX_SDK_NODES:
        raise ValueError(f"Invalid MANUS SDK node count: {count}")
    positions, rotations = np.empty((count, 3)), np.empty((count, 4))
    node_ids = []
    node_info = []
    for row in range(count):
        node = frame.nodes[row]
        positions[row] = node.position
        rotations[row] = node.rotation_wxyz
        node_ids.append(int(node.node_id))
        node_info.append(
            {
                "nodeId": int(node.node_id),
                "parentId": int(node.parent_id),
                "side": int(node.side),
                "chainType": int(node.chain_type),
                "fingerJointType": int(node.finger_joint_type),
            }
        )
    ergonomics = (
        np.asarray(frame.ergonomics, dtype=float).reshape(5, 4).copy()
        if frame.has_ergonomics else None
    )
    return ManusPacket(
        positions, rotations, node_ids, node_info,
        {1: "Left", 2: "Right"}.get(int(frame.side)), float(received_at),
        ergonomics=ergonomics, glove_id=int(frame.glove_id),
    )


class OfficialSdkTransport:
    """Read Raw Skeleton callbacks from official Core SDK 3.1.1 Integrated."""

    def __init__(
        self,
        bridge_path=MANUS_SDK_BRIDGE_PATH,
        clock=time.monotonic,
        pinch_compensation=MANUS_PINCH_COMPENSATION,
    ):
        path = Path(bridge_path)
        if not path.is_file():
            raise RuntimeError(
                f"MANUS SDK bridge is missing: {path}; run `make -C input/manus/assets`"
            )
        self._lib = ctypes.CDLL(str(path))
        self._configure_api()
        self._clock = clock
        self.pinch_compensation = bool(pinch_compensation)
        self._last_sequence = 0
        self._closed = False
        self._connected = False
        self._initialized = False
        self._init_error = None
        if self._lib.manus_bridge_initialize() != 0:
            self._init_error = self._error()
            return
        self._initialized = True
        if self._lib.manus_bridge_set_pinch_compensation(
            int(self.pinch_compensation)
        ) != 0:
            self._init_error = self._error()
            return
        self._stop = threading.Event()
        self._connect_thread = threading.Thread(
            target=self._connect_loop, name="manus-sdk-connect", daemon=True
        )
        self._connect_thread.start()

    def _configure_api(self):
        self._lib.manus_bridge_initialize.restype = ctypes.c_int
        self._lib.manus_bridge_connect.restype = ctypes.c_int
        self._lib.manus_bridge_poll.argtypes = (ctypes.POINTER(_SdkFrame),)
        self._lib.manus_bridge_poll.restype = ctypes.c_int
        self._lib.manus_bridge_set_pinch_compensation.argtypes = (ctypes.c_int,)
        self._lib.manus_bridge_set_pinch_compensation.restype = ctypes.c_int
        self._lib.manus_bridge_get_pinch_compensation.argtypes = (
            ctypes.POINTER(ctypes.c_int),
        )
        self._lib.manus_bridge_get_pinch_compensation.restype = ctypes.c_int
        self._lib.manus_bridge_calibration_get_step_count.argtypes = (
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
        )
        self._lib.manus_bridge_calibration_get_step_count.restype = ctypes.c_int
        self._lib.manus_bridge_calibration_get_step.argtypes = (
            ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(_SdkCalibrationStep),
        )
        self._lib.manus_bridge_calibration_get_step.restype = ctypes.c_int
        for name in (
            "manus_bridge_calibration_start",
            "manus_bridge_calibration_finish",
            "manus_bridge_calibration_stop",
        ):
            function = getattr(self._lib, name)
            function.argtypes = (ctypes.c_uint32,)
            function.restype = ctypes.c_int
        self._lib.manus_bridge_calibration_run_step.argtypes = (
            ctypes.c_uint32, ctypes.c_uint32,
        )
        self._lib.manus_bridge_calibration_run_step.restype = ctypes.c_int
        byte_pointer = ctypes.POINTER(ctypes.c_ubyte)
        self._lib.manus_bridge_calibration_export.argtypes = (
            ctypes.c_uint32, byte_pointer, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        )
        self._lib.manus_bridge_calibration_export.restype = ctypes.c_int
        self._lib.manus_bridge_calibration_import.argtypes = (
            ctypes.c_uint32, byte_pointer, ctypes.c_uint32,
        )
        self._lib.manus_bridge_calibration_import.restype = ctypes.c_int
        self._lib.manus_bridge_last_error.restype = ctypes.c_char_p
        self._lib.manus_bridge_shutdown.restype = None

    def _error(self):
        message = self._lib.manus_bridge_last_error()
        return message.decode("utf-8", errors="replace") if message else "unknown SDK error"

    def _connect_loop(self):
        while not self._stop.is_set():
            self._connected = self._lib.manus_bridge_connect() == 0
            self._stop.wait(1.0)

    def read(self):
        if self._init_error is not None:
            raise RuntimeError(self._init_error)
        if not self._connected:
            return None
        frame = _SdkFrame()
        result = self._lib.manus_bridge_poll(ctypes.byref(frame))
        if result < 0:
            raise RuntimeError(self._error())
        if result == 0 or frame.sequence == self._last_sequence:
            return None
        self._last_sequence = frame.sequence
        return _sdk_frame_to_packet(frame, self._clock())

    def pinch_compensation_readback(self):
        """Return Core's effective setting, or None before connection."""
        enabled = ctypes.c_int()
        if self._lib.manus_bridge_get_pinch_compensation(ctypes.byref(enabled)) != 0:
            return None
        return bool(enabled.value)

    def calibration_steps(self, glove_id):
        count = ctypes.c_uint32()
        if self._lib.manus_bridge_calibration_get_step_count(
            int(glove_id), ctypes.byref(count)
        ) != 0:
            raise RuntimeError(self._error())
        steps = []
        for index in range(count.value):
            data = _SdkCalibrationStep()
            if self._lib.manus_bridge_calibration_get_step(
                int(glove_id), index, ctypes.byref(data)
            ) != 0:
                raise RuntimeError(self._error())
            steps.append(CalibrationStep(
                int(data.index),
                bytes(data.title).split(b"\0", 1)[0].decode("utf-8", "replace"),
                bytes(data.description).split(b"\0", 1)[0].decode(
                    "utf-8", "replace"
                ),
                float(data.time),
            ))
        return tuple(steps)

    def _calibration_call(self, name, glove_id, *args):
        function = getattr(self._lib, f"manus_bridge_calibration_{name}")
        if function(int(glove_id), *args) != 0:
            raise RuntimeError(self._error())

    def calibration_start(self, glove_id):
        self._calibration_call("start", glove_id)

    def calibration_run_step(self, glove_id, step_index):
        self._calibration_call("run_step", glove_id, int(step_index))

    def calibration_finish(self, glove_id):
        self._calibration_call("finish", glove_id)

    def calibration_stop(self, glove_id):
        self._calibration_call("stop", glove_id)

    def calibration_export(self, glove_id):
        required = ctypes.c_uint32()
        function = self._lib.manus_bridge_calibration_export
        if function(int(glove_id), None, 0, ctypes.byref(required)) != 0:
            raise RuntimeError(self._error())
        if required.value == 0:
            raise RuntimeError("MANUS returned an empty glove calibration")
        buffer = (ctypes.c_ubyte * required.value)()
        if function(
            int(glove_id), buffer, len(buffer), ctypes.byref(required)
        ) != 0:
            raise RuntimeError(self._error())
        return bytes(buffer[:required.value])

    def calibration_import(self, glove_id, payload):
        payload = bytes(payload)
        if not payload:
            raise ValueError("Calibration payload cannot be empty")
        buffer = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        self._calibration_call("import", glove_id, buffer, len(buffer))

    def close(self):
        if self._closed:
            return
        self._closed = True
        stop = getattr(self, "_stop", None)
        if stop is not None:
            stop.set()
        thread = getattr(self, "_connect_thread", None)
        if thread is not None:
            thread.join(timeout=3.0)
        if self._initialized:
            self._lib.manus_bridge_shutdown()
            self._initialized = False
class ManusSource:
    """Read fresh WORLD Raw Skeleton positions as common input frames."""

    def __init__(
        self,
        transport=None,
        *,
        stale_seconds=MANUS_STALE_SECONDS,
        clock=time.monotonic,
    ):
        self.transport = OfficialSdkTransport() if transport is None else transport
        self.stale_seconds = float(stale_seconds)
        self.clock = clock
        self.last_received_at = None
        self._waiting_reported = False
        self._stale_reported = False
        self._first_frame_reported = False
        self.calibration_path = None
        print("Input source: MANUS")
        print(f"MANUS SDK: official Core SDK {MANUS_SDK_VERSION} Integrated")
        print("MANUS coordinate mode: WORLD/GLOBAL")
        print("p_UseWorldCoordinates: true")
        print(f"MANUS Raw Skeleton pinch compensation requested: {MANUS_PINCH_COMPENSATION}")
        print("MANUS -> Standard21 fallback mapping: " + str(MANUS_TO_STANDARD21.tolist()))
        print("CMC/root frame implementation: compute_cmc_frame; origin=point1; rotation=R_world_from_cmc")

    def mark_calibrated(self, path):
        self.calibration_path = Path(path)

    def _handedness(self, packet):
        explicit = packet.handedness
        if explicit in ("Left", "Right"):
            return explicit
        return handedness_from_node_info(packet.node_info)

    def _select_packet(self, packets):
        packets = packets if isinstance(packets, (list, tuple)) else [packets]
        for packet in packets:
            if self._handedness(packet) == "Left":
                return packet
        return packets[0] if packets else None

    def _empty_frame(self, timestamp, status):
        return InputFrame.empty(timestamp, status)

    def read(self):
        now = self.clock()
        try:
            packets = self.transport.read()
        except (RuntimeError, ValueError) as error:
            return self._empty_frame(now, f"MANUS INVALID · {error}")
        if packets is None:
            if self.last_received_at is None:
                if not self._waiting_reported:
                    self._waiting_reported = True
                    return self._empty_frame(now, "MANUS WAITING · Core/glove data unavailable")
                return None
            if now - self.last_received_at > self.stale_seconds and not self._stale_reported:
                self._stale_reported = True
                return self._empty_frame(now, "MANUS STALE")
            return None

        packet = self._select_packet(packets)
        if packet is None:
            return self._empty_frame(now, "MANUS WAITING")
        received_at = packet.received_at
        if now - received_at > self.stale_seconds:
            self.last_received_at, self._stale_reported = received_at, True
            return self._empty_frame(now, "MANUS STALE")
        handedness = self._handedness(packet)
        try:
            adapted = adapt_raw_skeleton(
                packet.positions,
                rotations_wxyz=packet.rotations_wxyz,
                handedness=handedness,
                node_info=packet.node_info,
                node_ids=packet.node_ids,
                scale_to_m=MANUS_POSITION_SCALE_TO_M,
            )
        except (KeyError, TypeError, ValueError) as error:
            return self._empty_frame(now, f"MANUS INVALID · {error}")

        self.last_received_at, self._stale_reported = received_at, False
        self._waiting_reported = False
        calibrated = True if self.calibration_path is not None else packet.calibrated
        try:
            initial_angles = _ergonomics_initial_angles(packet.ergonomics)
        except ValueError as error:
            initial_angles = None
            ergonomics_text = f"ergonomics invalid: {error}"
        else:
            ergonomics_text = "ergonomics ready"
        ready = (
            handedness in ("Left", "Right")
            and calibrated is not False
            and initial_angles is not None
        )
        calibration_text = "calibration unknown" if calibrated is None else (
            "calibrated" if calibrated else "not calibrated"
        )
        status = (
            f"MANUS TRACKING · {handedness or 'unknown side'} · "
            f"{calibration_text} · {ergonomics_text}"
        )
        if not self._first_frame_reported:
            print(f"MANUS raw node count: {len(packet.positions)}")
            print(f"MANUS handedness: {handedness or 'Unknown (SDK side/NodeInfo unavailable)'}")
            print(f"MANUS mapping source: {adapted.mapping_source}")
            print(f"MANUS -> Standard21 mapping: {adapted.mapping.tolist()}")
            readback = getattr(self.transport, "pinch_compensation_readback", lambda: None)()
            print(f"MANUS Raw Skeleton pinch compensation Core readback: {readback}")
            self._first_frame_reported = True
        return InputFrame(
            timestamp=received_at,
            points=adapted.points,
            handedness=handedness,
            ready=ready,
            status=status,
            finger_pad_directions=adapted.directions,
            preview=None,
            initial_joint_angles=initial_angles,
            raw_palm_length=adapted.raw_palm_length,
            raw_palm_width=adapted.raw_palm_width,
        )

    def close(self):
        self.transport.close()


def _ergonomics_initial_angles(ergonomics):
    values = np.asarray(ergonomics, dtype=float)
    if values.shape != (5, 4):
        raise ValueError("expected 5x4")
    if not np.isfinite(values).all():
        raise ValueError("contains non-finite values")
    thumb_scale = float(MANUS_THUMB_PIP_DIP_SCALE)
    if not np.isfinite(thumb_scale) or thumb_scale <= 0:
        raise ValueError("thumb PIP/DIP scale must be finite and positive")
    dip_to_pip_gain = float(MANUS_THUMB_DIP_TO_PIP_GAIN)
    if not np.isfinite(dip_to_pip_gain):
        raise ValueError("thumb DIP-to-PIP gain must be finite")
    radians = np.radians(values)
    pip, dip = radians[0, 2:4]
    thumb_targets = np.asarray((
        thumb_scale * (pip + dip_to_pip_gain * dip),
        thumb_scale * dip,
    ))
    return InitialJointAngles(
        radians[1:5], thumb_targets, four_finger_space="robot"
    )
