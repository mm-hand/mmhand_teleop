import argparse
import time

import numpy as np

from input import create_source
from retarget import Retargeter, RetargetWorker
from ros import RosOutput
from viewer_process import ViewerProcess


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--ros", action="store_true")
    output.add_argument("--sim", action="store_true")
    return parser.parse_args(argv)


def main():
    args = _parse_args()
    source = create_source()
    retargeter = Retargeter()
    viewer = ViewerProcess()
    ros = RosOutput() if args.ros else None
    worker = RetargetWorker(retargeter)
    simulation = None
    pending_view_output = None
    try:
        if args.sim:
            from simulation.process import GraspSimulationProcess

            simulation = GraspSimulationProcess()
        while True:
            request_started = time.monotonic()
            frame = source.read()
            input_latency_ms = (
                time.monotonic() - request_started
            ) * 1000.0
            if frame is None:
                if simulation is not None and not simulation.update(None):
                    break
                time.sleep(0.002)
                continue
            if frame.points is not None and ros is not None:
                ros.hand(frame)
            if frame.ready and frame.handedness == "Left":
                worker.submit(
                    frame.points, frame.timestamp, frame.finger_pad_directions,
                    frame.initial_joint_angles,
                )
            else:
                worker.pause()
                pending_view_output = None
            output = worker.poll()
            robot = None if output is None else output[0]
            if ros is not None and robot is not None:
                ros.joints(np.degrees(robot))
            if output is not None:
                pending_view_output = output
            view_robot, view_losses, view_latency, view_timings = (
                (None, None, None, None)
                if pending_view_output is None else pending_view_output
            )
            displayed = viewer.update(
                frame, view_robot, view_losses,
                input_latency_ms=input_latency_ms,
                retarget_latency_ms=view_latency,
                retarget_timings_ms=view_timings,
            )
            if displayed:
                pending_view_output = None
            if simulation is not None and not simulation.update(robot):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if simulation is not None:
            simulation.close()
        worker.close()
        source.close()
        viewer.close()
        if ros is not None:
            ros.close()


if __name__ == "__main__":
    main()
