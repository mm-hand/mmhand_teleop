# mmhand_teleop

`mmhand_teleop` converts a monocular camera (WiLoR ONNX or MediaPipe) or a
MANUS glove into one common 21-point hand observation, retargets the left hand
to all 21 MMHand joints, and serves a browser dashboard. ROS 2 output is on by
default; `--no-ros` runs the dashboard without it. The SAPIEN grasp simulation
is optional (`--sim`).

## Installation

Python 3.12 and Git LFS are required. The WiLoR weights are stored with Git LFS.

```bash
git lfs install
git lfs pull
mamba env create -f environment.yml
conda activate mmhand
python -m pip install -r input/wilor/requirements.txt
```

The requirements file installs ONNX Runtime with its CUDA and cuDNN
dependencies; WiLoR requires `CUDAExecutionProvider` and an NVIDIA GPU.

MediaPipe is optional:

```bash
python -m pip install --no-deps -r input/mediapipe/requirements.txt
```

`--no-deps` keeps the environment's OpenCV instead of installing
`opencv-contrib-python`; the requirements file lists mediapipe's non-OpenCV
runtime dependencies. If you are not using the environment above, drop
`--no-deps`. Download
[hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
into `input/mediapipe/assets/`. The model file is not tracked by git.

MANUS requires the bundled Core SDK bridge; follow
`input/manus/assets/README.md`. Calibrate each glove once before tracking:

```bash
python -m input.manus.calibration --side left          # import the saved .mcal
python -m input.manus.calibration --side left --force  # run the wizard again
python -m input.manus.calibration --side right         # per glove, not per hand
```

The wizard waits up to `MANUS_CALIBRATION_CONNECT_TIMEOUT` for a connected
glove, lists the official SDK calibration steps, records each pose when you
press Enter, and saves
`input/manus/calibrations/<side>_<8 hex digit glove id>.mcal` (git-ignored).
`q` or Ctrl-C aborts without writing a file. `track.py` imports the saved blob
at startup and only runs the wizard when no file exists for that glove.

SAPIEN is optional and used only with `--sim`:
`python -m pip install -r simulation/requirements.txt`.

## Running

All runtime settings live in `config.py`:

```python
INPUT_SOURCE = "mediapipe"   # "mediapipe", "wilor", or "manus"
CAMERA_DEVICE = 0            # Windows camera index or Linux V4L2 index/path
```

```bash
python track.py          # dashboard and ROS 2 publishing
python track.py --no-ros # dashboard only, without ROS
python track.py --sim    # also open the optional SAPIEN grasp window
```

`--no-ros` and `--sim` are the only command-line arguments; every other setting
lives in `config.py`. Open <http://localhost:8080>; the two 3D views are served
on 8081 (input hand) and 8082 (MMHand). If one of these ports is already in use
(for example by another dashboard), each server moves to the next free port and
the `Dashboard: ...` URL printed at startup is the one to open.

Only a ready left hand enters retargeting; a right hand is displayed only. With
`INPUT_SOURCE = "manus"`, startup waits for a connected left glove and its
calibration file before the viewer, retargeter, ROS, or simulation starts. The
first WiLoR start loads the models and takes about 10-20 s.

The Input hand 3D panel uses a canonical display: the hand is expressed in its
palm frame, palm toward the screen, and normalized by palm width, so moving or
rotating the hand does not rotate or resize the view. Retargeting still uses
the raw source coordinates. Without finger-pad normals (MediaPipe) the display
uses the palm-frame normal and hides the pad arrows.

## How it works

**Input.** A camera frame goes through the WiLoR detector and reconstructor
ONNX models, producing MANO vertices. A joint regressor and fingertip vertices
yield Standard21 points and five outward finger-pad normals. One Euro filters
smooth points and directions; angles are solved from the filtered points and
packaged into `input.frame.InputFrame`. MediaPipe produces Standard21 world
landmarks in metres, with no finger-pad normals: those frames carry
`finger_pad_directions=None`. The MANUS path maps Raw25 SDK data to the same
Standard21 points, pad directions, and ergonomics-derived angles.

**Retargeting.** `retarget.Retargeter` builds a thumb-CMC-local frame from the
input points, maps the input angles to robot targets, and runs a bounded SLSQP
solve over all 21 joints with analytic Jacobians. The weighted objective
contains the thumb-tip position, the two thumb bends, the 16 four-finger joint
angles, four fingertip vectors, the thumb pad, and four finger pad directions.
The two pad terms are disabled (weight 0) on frames without
`finger_pad_directions`, so MediaPipe input never fakes fingertip pointing.
The previous solution is the next warm start; outputs are clipped to URDF
limits. Input, retargeting, visualization, ROS, and simulation contain no
device-specific branches.

**Output.** `track.py` reads one frame per loop, submits ready left-hand frames
to a background solver, and forwards the newest display snapshot to a separate
Viewer process. The Viewer owns the two Viser scenes and the HTTP dashboard, so
slow rendering never blocks acquisition or solving. ROS 2 output is published
by default and `--no-ros` disables it; it sends
`std_msgs/msg/Float32MultiArray`:

| Topic | Shape | Meaning |
|---|---:|---|
| `/mmhand/keypoints` | 21x3 | source-frame points, metres |
| `/mmhand/finger_pad_directions` | 5x3 | outward pad normals |
| `/mmhand/joint_angle_deg` | 21 | MMHand J00-J20, degrees |

The finger-pad topic is published only when the input source provides pointing.

## Repository layout

```text
input/
  frame.py              InputFrame, topology, CMC frame
  camera.py             latest-frame OpenCV capture
  mediapipe/source.py   Hand Landmarker points, filtering, InputFrame
  wilor/source.py       ONNX detector/reconstructor, filtering, InputFrame
  manus/adapter.py      Raw25 -> Standard21 mapping and pad directions
  manus/calibration.py  terminal calibration gate
  manus/source.py       official SDK transport
assets/mmhand/          MMHand URDF, visual meshes, collision meshes
config.py               runtime settings
one_euro.py             ndarray One Euro filter
retarget.py             URDF FK, analytic Jacobians, SLSQP retargeter
ros.py                  ROS 2 publishers
viewer.py               HTTP dashboard and two Viser scenes
viewer_process.py       isolated Viewer process proxy
track.py                process entry point
simulation/             optional SAPIEN grasp simulation
test_*.py               unit tests
```

## Tests and licenses

```bash
python -m unittest -v test_hand_capture.py test_wilor.py test_manus_phase1.py test_mediapipe.py
python -m unittest -v test_simulation.py
```

WiLoR model terms are in `input/wilor/assets/WILOR_MODEL_LICENSE.txt`; related
notices are in `input/wilor/assets/THIRD_PARTY_NOTICES.md`. MANUS SDK terms ship
with `input/manus/assets/`. The MMHand URDF and meshes come from
[`mm-hand/structure`](https://github.com/mm-hand/structure).
