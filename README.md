# mmhand_teleop

`mmhand_teleop` converts a monocular camera (WiLoR ONNX) or a MANUS glove into
one common 21-point hand observation, retargets the left hand to all 21 MMHand
joints, and serves a browser dashboard. ROS 2 and SAPIEN output are optional;
the base application runs without ROS.

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

MANUS requires the bundled Core SDK bridge; follow
`input/manus/assets/README.md`. SAPIEN is optional:
`python -m pip install -r simulation/requirements.txt`.

## Running

All runtime settings live in `config.py`:

```python
INPUT_SOURCE = "manus"   # "wilor" or "manus"
CAMERA_DEVICE = "/dev/video4"  # V4L2 index or /dev/videoN path
```

```bash
python track.py          # dashboard only, without ROS
python track.py --ros    # also publish ROS 2 topics
python track.py --sim    # optional SAPIEN grasp simulation
```

`--ros` and `--sim` are mutually exclusive. Open <http://localhost:8080>; the
two 3D views are served on 8081 (input hand) and 8082 (MMHand).

Only a ready left hand enters retargeting; a right hand is displayed only. With
`INPUT_SOURCE = "manus"`, startup waits for a connected left glove and its
calibration file before the viewer, retargeter, ROS, or simulation starts. The
first WiLoR start loads the models and takes about 10-20 s.

The Input hand 3D panel uses a canonical display: the hand is expressed in its
palm frame, palm toward the screen, and normalized by palm width, so moving or
rotating the hand does not rotate or resize the view. Retargeting still uses
the raw source coordinates.

## How it works

**Input.** A camera frame goes through the WiLoR detector and reconstructor
ONNX models, producing MANO vertices. A joint regressor and fingertip vertices
yield Standard21 points and five outward finger-pad normals. One Euro filters
smooth points and directions; angles are solved from the filtered points and
packaged into `input.frame.InputFrame`. The MANUS path maps Raw25 SDK data to
the same Standard21 points, pad directions, and ergonomics-derived angles.

**Retargeting.** `retarget.Retargeter` builds a thumb-CMC-local frame from the
input points, maps the input angles to robot targets, and runs a bounded SLSQP
solve over all 21 joints with analytic Jacobians. The weighted objective
contains the thumb-tip position, the two thumb bends, the 16 four-finger joint
angles, four fingertip vectors, the thumb pad, and four finger pad directions.
The previous solution is the next warm start; outputs are clipped to URDF
limits. Input, retargeting, visualization, ROS, and simulation contain no
device-specific branches.

**Output.** `track.py` reads one frame per loop, submits ready left-hand frames
to a background solver, and forwards the newest display snapshot to a separate
Viewer process. The Viewer owns the two Viser scenes and the HTTP dashboard, so
slow rendering never blocks acquisition or solving. Optional ROS 2 output
publishes `std_msgs/msg/Float32MultiArray`:

| Topic | Shape | Meaning |
|---|---:|---|
| `/mmhand/keypoints` | 21x3 | source-frame points, metres |
| `/mmhand/finger_pad_directions` | 5x3 | outward pad normals |
| `/mmhand/joint_angle_deg` | 21 | MMHand J00-J20, degrees |

## Repository layout

```text
input/
  frame.py              InputFrame, topology, CMC frame
  wilor/camera.py       latest-frame OpenCV capture
  wilor/source.py       ONNX detector/reconstructor, filtering, InputFrame
  manus/adapter.py      Raw25 -> Standard21 mapping and pad directions
  manus/calibration.py  terminal calibration gate
  manus/source.py       official SDK transport
assets/mmhand/          MMHand URDF, visual meshes, collision meshes
config.py               runtime settings
one_euro.py             ndarray One Euro filter
retarget.py             URDF FK, analytic Jacobians, SLSQP retargeter
ros.py                  optional ROS 2 publishers
viewer.py               HTTP dashboard and two Viser scenes
viewer_process.py       isolated Viewer process proxy
track.py                process entry point
simulation/             optional SAPIEN grasp simulation
test_*.py               unit tests
```

## Tests and licenses

```bash
python -m unittest -v test_hand_capture.py test_wilor.py test_manus_phase1.py
python -m unittest -v test_simulation.py
```

WiLoR model terms are in `input/wilor/assets/WILOR_MODEL_LICENSE.txt`; related
notices are in `input/wilor/assets/THIRD_PARTY_NOTICES.md`. MANUS SDK terms ship
with `input/manus/assets/`. The MMHand URDF and meshes come from
[`mm-hand/structure`](https://github.com/mm-hand/structure).
