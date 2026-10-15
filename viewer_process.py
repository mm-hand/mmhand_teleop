"""Run the Viser dashboard in an isolated spawned process."""

from dataclasses import dataclass
import multiprocessing as mp
from queue import Empty, Full
import signal
import time

import numpy as np


_STARTUP_TIMEOUT = 30.0
_SHUTDOWN_TIMEOUT = 5.0


@dataclass(slots=True)
class _ViewerUpdate:
    """One latest-only dashboard snapshot transferred to the child."""

    frame: object
    robot: np.ndarray | None
    losses: dict | None
    input_latency_ms: float | None
    retarget_latency_ms: float | None
    retarget_timings_ms: dict | None


def _send_status(connection, message):
    """Send child state without masking the original child exception."""

    try:
        connection.send(message)
    except (BrokenPipeError, EOFError, OSError):
        pass


def _run_viewer(updates, stop_event, status, factory):
    """Own Viser, HTTP, JPEG encoding, FK diagnostics, and rendering."""

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    viewer = None
    try:
        if factory is None:
            from retarget import RobotModel
            from viewer import Viewer

            viewer = Viewer(RobotModel())
        else:
            viewer = factory()
        _send_status(status, ("ready",))

        pending = None
        while not stop_event.is_set():
            try:
                update = updates.get(timeout=0.01 if pending is None else 0.001)
            except Empty:
                update = None
            if update is not None:
                pending = update
                # Rendering an old snapshot only adds latency. Drain all queued
                # snapshots and retain the newest one before touching Viser.
                while True:
                    try:
                        pending = updates.get_nowait()
                    except Empty:
                        break
            if pending is None:
                continue
            if viewer.update(
                pending.frame,
                pending.robot,
                pending.losses,
                input_latency_ms=pending.input_latency_ms,
                retarget_latency_ms=pending.retarget_latency_ms,
                retarget_timings_ms=pending.retarget_timings_ms,
            ):
                pending = None
        _send_status(status, ("closed",))
    except BaseException as error:
        _send_status(status, ("error", type(error).__name__, str(error)))
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        status.close()


class ViewerProcess:
    """Non-blocking latest-frame proxy for the Viser dashboard process."""

    def __init__(self, startup_timeout=_STARTUP_TIMEOUT, *, _factory=None):
        if not np.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive and finite")

        # Spawn does not inherit camera, CUDA, solver, or parent-thread state.
        context = mp.get_context("spawn")
        self._updates = context.Queue(maxsize=1)
        self._stop_event = context.Event()
        self._status, child_status = context.Pipe(duplex=False)
        self._process = context.Process(
            target=_run_viewer,
            args=(self._updates, self._stop_event, child_status, _factory),
            name="mmhand-viser-viewer",
            daemon=True,
        )
        self._state = "starting"
        self._error = None
        self._closed = False
        # Repeat the newest solver result in subsequent snapshots. Otherwise a
        # queue replacement could discard the only snapshot containing it.
        self._robot = None
        self._losses = None
        self._retarget_latency_ms = None
        self._retarget_timings_ms = None
        try:
            self._process.start()
            child_status.close()
            self._wait_until_ready(float(startup_timeout))
        except BaseException:
            child_status.close()
            self.close()
            raise

    def _read_status(self, timeout=0.0):
        if self._status is None:
            return
        try:
            if not self._status.poll(timeout):
                return
            message = self._status.recv()
        except (EOFError, OSError):
            return
        kind = message[0]
        if kind == "ready":
            self._state = "running"
        elif kind == "closed":
            self._state = "closed"
        elif kind == "error":
            self._state = "error"
            self._error = message[1:]

    def _raise_remote_error(self):
        name, message = self._error
        detail = f"{name}: {message}" if message else name
        raise RuntimeError(f"Viser viewer failed: {detail}")

    def _wait_until_ready(self, timeout):
        deadline = time.monotonic() + timeout
        while self._state == "starting":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Viser viewer did not start within {timeout:g}s"
                )
            self._read_status(min(remaining, 0.05))
            if self._process.exitcode is not None and self._state == "starting":
                self._read_status(0.05)
                raise RuntimeError(
                    "Viser viewer exited during startup "
                    f"(exit code {self._process.exitcode})"
                )
        if self._state == "error":
            self._raise_remote_error()
        if self._state != "running":
            raise RuntimeError("Viser viewer closed during startup")

    def _check_process(self):
        """Propagate child failure while keeping normal closure non-fatal."""

        self._read_status()
        if self._state == "error":
            self._raise_remote_error()
        if self._state == "closed":
            return False
        if self._process.exitcode is None:
            return True
        self._read_status(0.05)
        if self._state == "error":
            self._raise_remote_error()
        if self._state == "closed" or self._process.exitcode == 0:
            self._state = "closed"
            return False
        raise RuntimeError(
            "Viser viewer process exited unexpectedly "
            f"(exit code {self._process.exitcode})"
        )

    def update(
        self,
        frame,
        robot=None,
        losses=None,
        input_latency_ms=None,
        retarget_latency_ms=None,
        retarget_timings_ms=None,
    ):
        """Publish the newest complete display snapshot without rendering it."""

        if self._closed or not self._check_process():
            return False
        if frame.points is None or frame.handedness != "Left":
            self._robot = None
            self._losses = None
            self._retarget_latency_ms = None
            self._retarget_timings_ms = None
        elif robot is not None:
            robot = np.asarray(robot, dtype=float)
            if robot.shape != (21,) or not np.isfinite(robot).all():
                raise ValueError("robot must contain 21 finite radians")
            self._robot = robot.copy()
            self._losses = None if losses is None else dict(losses)
            self._retarget_latency_ms = retarget_latency_ms
            self._retarget_timings_ms = (
                None if retarget_timings_ms is None
                else dict(retarget_timings_ms)
            )

        update = _ViewerUpdate(
            frame=frame,
            robot=self._robot,
            losses=self._losses,
            input_latency_ms=input_latency_ms,
            retarget_latency_ms=self._retarget_latency_ms,
            retarget_timings_ms=self._retarget_timings_ms,
        )
        # Never wait for Viser. If its one-slot queue is occupied, replace that
        # stale snapshot. Queue feeder serialization also stays off this loop.
        try:
            self._updates.put_nowait(update)
            return True
        except Full:
            try:
                self._updates.get_nowait()
            except Empty:
                return False
            try:
                self._updates.put_nowait(update)
                return True
            except Full:
                return False

    def close(self):
        """Stop the dashboard child and reclaim all process resources."""

        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._process.pid is not None:
            self._process.join(_SHUTDOWN_TIMEOUT)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(_SHUTDOWN_TIMEOUT)
        if self._status is not None:
            self._status.close()
            self._status = None
        self._updates.close()
        self._updates.join_thread()
        self._process.close()
