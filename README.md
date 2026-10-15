# mmhand_teleop

`mmhand_teleop` converts a monocular camera or a MANUS glove into one common
hand-observation schema, retargets a left hand to all 21 MMHand joints, and
exposes a browser dashboard plus optional ROS 2 or SAPIEN output.

The checked-in configuration currently selects MANUS. Set `INPUT_SOURCE` to
`"wilor"` to use the CUDA ONNX camera path instead. A camera may be a USB/UVC
device or an integrated webcam exposed as `/dev/videoN`; no depth or
multi-camera calibration is required.

## Installation

The WiLoR models use Git LFS. Install the repository and Conda environment with:

```bash
git lfs install
git clone https://github.com/mm-hand/mmhand_teleop.git
cd mmhand_teleop
git lfs pull
mamba env create -f environment.yml
conda activate mmhand
python -m pip install -r input/wilor/requirements.txt
```

The last command installs ONNX Runtime with its CUDA and cuDNN dependencies.
WiLoR requires `CUDAExecutionProvider`. SAPIEN is optional:

```bash
python -m pip install -r simulation/requirements.txt
```

## Configuration

Runtime configuration lives in `config.py`. Select exactly one input:

```python
INPUT_SOURCE = "manus"  # "wilor" or "manus"
```

For WiLoR, configure any OpenCV source and its requested mode:

```python
CAMERA_DEVICE = 0       # V4L2 index or "/dev/videoN"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
WILOR_DEVICE_ID = 0
```

The camera prints the actual mode negotiated with the device. WiLoR keypoints
and finger-pad directions have independent One Euro settings:

```python
WILOR_POINT_FILTER = (0.5, 1.0, 1.0)
WILOR_DIRECTION_FILTER = (0.5, 0.25, 1.0)
WILOR_THUMB_PAD_ROTATION_DEG = 45.0
```

Each tuple is `(min_cutoff, beta, derivative_cutoff)`. Filters reset when the
hand disappears, an inference result is invalid, or handedness changes.
Retarget residual weights and solver limits are configured separately:

```python
RETARGET_THUMB_TIP_SCALE = 1.0
RETARGET_THUMB_TIP_WEIGHT = 1.0
RETARGET_THUMB_PAD_WEIGHT = 2.0
RETARGET_INDEX_PAD_WEIGHT = 2.0
RETARGET_MIDDLE_PAD_WEIGHT = 2.0
RETARGET_RING_PAD_WEIGHT = 2.0
RETARGET_LITTLE_PAD_WEIGHT = 4.0
RETARGET_THUMB_PROXIMAL_BEND_WEIGHT = 1.0
RETARGET_THUMB_DISTAL_BEND_WEIGHT = 1.0
RETARGET_FINGER_ANGLE_WEIGHT = 1.0
RETARGET_FINGERTIP_VECTOR_WEIGHT = 1.0
RETARGET_POSITION_NORMALIZER_M = 0.086
RETARGET_MAX_EVALUATIONS = 15
RETARGET_FTOL = 3e-5
```

`RETARGET_THUMB_TIP_SCALE` scales only the human CMC-to-thumb-tip position
target. Every angle target remains in radians at a fixed physical `1:1` scale.
`RETARGET_MAX_EVALUATIONS` limits distinct objective evaluations per frame; the
lowest finite candidate seen before the limit is retained.

For MANUS, place/build the bundled Core SDK bridge as described in
`input/manus/assets/README.md`, then set `INPUT_SOURCE = "manus"`.
`MANUS_PAD_LOCAL_AXIS` selects the outward pad axis in every Tip node's local
frame and defaults to `(0, 0, -1)`. `MANUS_THUMB_PAD_ROTATION_DEG = 30.0`
aligns the resulting thumb direction around the source-frame IP-to-TIP axis:
positive for Left and negative for Right. This correction is completed before
constructing `InputFrame`; retargeting receives the corrected direction without
any MANUS-specific branch. Handedness comes directly from the SDK frame or its
NodeInfo topology. `MANUS_PINCH_COMPENSATION` switches the official Raw Skeleton
MetaGlove pinch compensation; compare identical pinch poses with `False` and
`True`. The bridge reads the value back from Core and reports it on the first
valid frame. Official per-glove calibration blobs are stored under
`MANUS_CALIBRATION_DIR`; `MANUS_CALIBRATION_CONNECT_TIMEOUT` controls how long
startup waits for a Left glove.

MANUS Thumb PIPStretch/DIPStretch use a configurable fixed gain before entering
the common angle field:

```python
MANUS_THUMB_PIP_DIP_SCALE = 1.0
MANUS_THUMB_DIP_TO_PIP_GAIN = 0.5
```

The first gain scales both thumb bends. The second blends half of Thumb
DIPStretch into J18. Both values affect the SLSQP initial angles and their soft
targets. Four-finger ergonomics and all WiLoR-derived angles retain their
original scale.

## Running

```bash
python track.py
python track.py --ros
python track.py --sim
```

With `INPUT_SOURCE = "manus"`, startup is calibration-gated before the viewer,
retargeter, ROS output, or simulation is created. The program waits for a Left
glove and looks for `left_<GLOVE_ID>.mcal` in `MANUS_CALIBRATION_DIR`. An
existing file is loaded through the official SDK. If it is absent, a terminal
wizard displays each SDK-provided title, description, and duration, waits for
the user before recording every step, then calls the official finish/export
functions and saves the result. Tracking starts only after this succeeds. No
graphical calibration interface is used.

To run the same terminal workflow separately, or deliberately recalibrate:

```bash
python -m input.manus.calibration
python -m input.manus.calibration --force
```

Stop all other MANUS processes first because Core Integrated owns the device
connection exclusively. Enter `q` at any calibration prompt to cancel safely.

`--ros` and `--sim` are mutually exclusive. Every other option belongs in
`config.py`.

Input Standard21 points always retain source coordinates and physical size in
metres: MANUS uses WORLD/GLOBAL coordinates, while WiLoR retains its reconstructed
geometry after handedness handling. There is no wrist-centering, palm-frame
rotation, or hand-size scaling stage in the input pipeline. The retargeter still
builds a CMC-local frame for comparing features; this translates and rotates
geometry without scaling it. Finger-pad directions remain unit vectors.

Open <http://localhost:8080>. The dashboard contains the input preview, the
input hand, and the retargeted MMHand. WiLoR overlays the detected box and
yellow MANO mesh on the camera image. Its status shows confidence followed by
smoothed tracking FPS. Both hand views show finger-pad directions, thumb angle
arcs, live angle values in degrees, and palm width in millimetres. Input palm
width is the source Index-MCP-to-Little-MCP distance (`P5` to `P17`);
MMHand palm width is the FK distance between the corresponding MCP A-A joint
origins. Both panels also report a direct Hand0-to-Middle-tip distance. Input
Hand0 is Standard21 `P0`, so its value is simply `|P12-P0|` and decreases when
the Middle finger bends. MMHand Hand0 is the URDF `base_link` origin; its
endpoint is the distal-most `+X` surface vertex of
`finger_2_fingertip_1.stl` transformed by the zero-configuration URDF FK. The
robot value is their Euclidean distance, is not based on the `2-tip_Link`
virtual pad/contact-frame origin, and does not change with the retargeted pose.
The robot panel labels these physical URDF measurements as
`MMHand palm width` and `MMHand URDF palm length`; MMHand has no palm-size
normalization stage. The input panel
additionally reports `raw_palm_length`: the same direct Hand0-to-tip distance
computed from source points before input filtering. This is the displayed raw
palm length, not a
`P0`-to-`P9` Euclidean distance. It also reports
`raw_palm_width`, the P5-to-P17 Euclidean distance from those same source
points. All displayed distances use millimetres. The input thumb text comes from
`initial_joint_angles`; its arcs visualize the bends present in the 21-point
geometry. The robot text displays the output J18/J19 values. The loss panel
reports the seven weighted objective terms and their total.

The `Residual vectors` selector controls paired diagnostics in both 3D panels.
`Thumb tip` shows P1-to-P4 against the robot CMC-origin-to-Thumb-virtual-tip
feature. `Fingertip vectors` shows the four P4-to-fingertip targets against the
four robot virtual-tip vectors and is selected by default. `Thumb pad` and
`Finger pads` show unit directions plus their human-to-robot angular errors.
Position labels contain CMC-local XYZ components and magnitudes in millimetres;
direction labels contain local XYZ, unit norm, and angle error. The selector
also reports the configured group weights, each pad's effective `/3` weight,
and the `RETARGET_POSITION_NORMALIZER_M` normalizer for position terms. A zero
weight is
marked disabled even though its diagnostic remains selectable.

These diagnostics deliberately use the virtual `tip_Link` origins consumed by
the loss. They do not use the Middle fingertip mesh vertex reserved for the
fixed URDF palm-length display. Human values are the unweighted targets built
from the current `InputFrame`; robot values come from `RobotModel.features()`
at the output joint vector currently displayed by Viser.

The `Input hand` title reports the latency from entering `source.read()`
until that request returns common input data. The `Retargeted
MMHand` title starts when that completed frame is submitted to the background
retarget worker and ends when its solve completes. These are non-overlapping
pipeline measurements rather than two end-to-end intervals; optional ROS
publication and input copying can leave a small unmeasured gap between them.
Both use the same monotonic clock, are displayed in milliseconds, and exclude
browser refresh and rendering time.

Directly below each latency, the dashboard also reports `render X.X ms`. This
is the Python/Viser server-side time used to update the corresponding point
cloud, skeleton, direction arrows, angle arcs, or robot URDF state. It measures
scene-update submission and does not claim to measure completion of GPU drawing
inside the browser iframe.

The retargeted panel also decomposes this post-InputFrame latency. `worker
queue` is time waiting for the background solver. `targets`, `SLSQP`, and
`final loss` are measured inside `Retargeter.solve()`. `solve total` covers the complete solve; `solver overhead` is its small unclassified
remainder. Therefore `worker queue + solve total` approximately equals the
right-side latency. Browser refresh and GPU drawing remain outside these
measurements.

The complete Viewer stack runs in a separate process created with the
`spawn` multiprocessing context. The tracking process only publishes a
one-slot, latest-only display snapshot; when Viser, JPEG encoding, residual
diagnostics, WebSocket traffic, or the browser cannot keep up, an older display
snapshot is replaced instead of blocking input acquisition or retargeting.
The Viewer child owns both Viser servers and the Dashboard HTTP server.

Only a ready left hand enters retargeting. A right hand can still be displayed
and published.

## Common input format

Every input returns `input.frame.InputFrame`:

```python
@dataclass(slots=True)
class InputFrame:
    timestamp: float
    points: np.ndarray | None
    handedness: str | None
    ready: bool
    status: str
    finger_pad_directions: np.ndarray | None = None
    preview: np.ndarray | None = None
    initial_joint_angles: InitialJointAngles | None = None
    raw_palm_length: float | None = None
    raw_palm_width: float | None = None
```

`points` is a finite `21 x 3` array in metres. Its order is wrist followed by
Thumb, Index, Middle, Ring, and Little, with four joints from palm to tip per
finger. Points retain the source Standard21 coordinates and physical size in
metres. MANUS uses WORLD/GLOBAL coordinates; WiLoR uses its reconstructed source
geometry after handedness handling. No input centering, rotation, or size scaling
is applied.

`finger_pad_directions` is a finite `5 x 3` array in Thumb-to-Little order. Each
row is an outward unit vector in the same source frame as points. WiLoR produces
this
field from selected MANO pad faces. MANUS rotates `MANUS_PAD_LOCAL_AXIS` by the
five Tip-node WORLD quaternions, then applies its handed thumb-pad alignment before constructing `InputFrame`.
Retargeting compares the thumb row under `RETARGET_THUMB_PAD_WEIGHT`. Index,
Middle, Ring, and Little use their corresponding independent `*_PAD_WEIGHT`.

`initial_joint_angles` stores radians in a common structure:
four-finger rows are Index-to-Little with MCP Spread/MCP F-E/PIP/DIP columns,
and the two thumb values are the first and second bends along the thumb chain.
WiLoR derives these values from its filtered Standard21 points and marks the
four-finger values as human-space angles. Retargeting maps WiLoR MCP A-A with
the dynamic robot neutral/axis and maps its flexion angles from the URDF lower
limits. MANUS converts `ergonomics[1:5]` directly from degrees to radians and
marks them as robot-space angles, so MCP Spread/MCP F-E/PIP/DIP directly seed
the corresponding four robot joints after URDF clipping. Thumb
PIPStretch/DIPStretch (`ergonomics[0, 2:4]`) are converted to radians and
multiplied by `MANUS_THUMB_PIP_DIP_SCALE` before seeding J18/J19. No offset,
neutral, axis-sign, or activity-range remapping is applied. A MANUS frame with
missing or invalid ergonomics remains visible but is not ready for retargeting.

The direct MANUS mapping only defines the per-frame SLSQP initial guess and its
soft angle targets. All J00-J20 joints remain optimization variables, and the
spatial, pad-direction, and angle residuals are still minimized jointly. J16,
J17, and J20 retain the previous solution as their warm start because MANUS
ergonomics does not provide direct counterparts for them.

On the first solve, and again after `pause()`, Thumb MCP F-E (J17) starts from
its URDF upper limit to expose the strongly flexed thumb-to-little opposition
basin. It remains a free SLSQP variable. Later frames warm-start J17 from the
previous solution rather than resetting it to the upper limit.

For MANUS, both the initial guess and soft angle residual targets use `J18 =
scale * (PIPStretch + 0.5 * DIPStretch)` and `J19 = scale * DIPStretch`, where
`scale` is `MANUS_THUMB_PIP_DIP_SCALE`.

`timestamp` is the monotonic acquisition time. Missing landmark data uses
`points=None`, `finger_pad_directions=None`, and `ready=False`. A non-ready frame
may retain valid points and pad directions for display, but every ready frame
must also contain `initial_joint_angles`. Every frame with points must contain
five finite unit pad directions; malformed frames raise an error. Retargeting
receives the common points, pad directions, timestamp, and initial angles from
the same frame.

## Processing

```text
OpenCV frame -> detector/cached crop -> WiLoR MANO mesh
             -> Standard21 points + pad-face normals
             -> source geometry -> point/direction filters
             -> angles solved from filtered points -> InputFrame

MANUS Core Integrated -> per-glove calibration load/wizard
                      -> Raw25 WORLD positions + Tip WORLD rotations
                      -> semantic mapping, or fixed fallback when NodeInfo is absent
                      -> source points + 5 pad directions
                      -> handed thumb-pad alignment (+30° Left / -30° Right)
                      + ergonomics degrees -> radians
                        + configured thumb PIP/DIP mapping -> InputFrame

InputFrame -> human CMC target frame + angle-to-URDF target mapping
           -> full J00-J20 bounded SLSQP with analytic Jacobians
           -> URDF limit clipping -> dashboard / ROS 2 / SAPIEN
```

WiLoR runs its detector and reconstruction models in FP16, then performs box
and NMS arithmetic in FP32. The MANO mesh is regressed to standard landmarks.
Pad-face normals are averaged, mirrored for handedness, kept in source
coordinates, and normalized to unit length. The thumb normal is rotated by the
configured fixed
angle around its IP-to-TIP axis: positive for Left and negative for Right.

### Retarget target construction

`Retargeter._targets()` builds a human CMC frame with point 1 as its origin. Its
longitudinal axis follows wrist-to-middle-MCP, its lateral axis is the
Index-to-Little MCP direction projected onto the palm plane, and the third axis
is their cross product. Human positions and all five pad directions are
expressed in this frame before comparison with robot FK features.

The structured angle input is converted into robot targets as follows:

| Input angle | Robot target | Mapping before URDF clipping |
|---|---|---|
| Four-finger MCP Spread | Each MCP A-A joint | `robot_neutral + spread / axis_sign` |
| Four-finger MCP F-E | Each MCP F-E joint | `URDF lower + angle` |
| Four-finger PIP | Each PIP joint | `URDF lower + angle` |
| Four-finger DIP | Each DIP joint | `URDF lower + angle` |
| First thumb-chain bend | J18 | angle directly |
| Second thumb-chain bend | J19 | angle directly |

The mapped J00-J15 and J18/J19 values overwrite those channels in every
per-frame SLSQP initial guess. J16, J17, and J20 have no direct angle target and
retain the previous solution as their warm start; on a cold start they use the
URDF-clipped zero seed. This mapping is an initialization and a soft objective,
not a hard assignment: the optimizer may move every joint to improve the
combined objective.

### Retarget objective and solve

`RobotModel.features()` evaluates the FK features below together with analytic
Jacobians for all 21 joints:

| Loss key | Residual |
|---|---|
| `thumb_tip` | Robot palm-to-thumb-tip position versus the scaled human CMC-to-tip position |
| `thumb_proximal_bend` | Robot J18 versus the first input thumb bend |
| `thumb_distal_bend` | Robot J19 versus the second input thumb bend |
| `finger_angles` | All 16 four-finger robot joints versus their mapped targets |
| `fingertip_vectors` | Four Thumb-to-Index/Middle/Ring/Little tip vectors |
| `thumb_pad` | Robot thumb pad normal versus the input thumb pad direction |
| `finger_pads` | Four robot finger pad normals versus the corresponding input directions |

Position and relative-vector errors are normalized by `RETARGET_POSITION_NORMALIZER_M`;
this fixed residual scale does not resize the hand. Both `thumb_tip` and
`fingertip_vectors` retain dependence on input hand size. Angle errors are in
radians and direction errors are dimensionless. Index,
Middle, Ring, and Little pad rows have independent configured weights. Each
finger contributes directly without averaging across the four fingers; its
effective solver multiplier is `weight / 3`, where `/3` averages the XYZ
direction components. The total objective is the sum of the configured
weighted mean-squared terms. In
particular, the four complete thumb-to-fingertip vectors can adjust joints at
both ends, while thumb-tip and pad losses indirectly determine J16/J17/J20 and
may also refine J18/J19 away from their soft angle targets.

The human and robot vectors are compared component-wise in independently built
local frames rather than through one explicit human-to-robot rigid transform.
For human points, `compute_cmc_frame()` uses P1 as the origin, normalized
P0-to-P9 as X, the P5-to-P17 side vector projected orthogonal to X as Y, and
`X × Y` as Z. Row vectors use `(p - P1) @ R_h` and directions use `d @ R_h`,
where `R_h` has those axes as columns. For MMHand, the local origin is the
projection of the Thumb MCP A-A origin onto the Thumb CMC joint-axis line and
the axes are the URDF `base_link` rotation. Robot points and directions use the
same row-vector formulas with that origin and rotation. This frame conversion
translates and rotates but
does not scale input lengths. Only Left-hand frames currently enter the solver.

All J00-J20 variables are optimized together with bounded SLSQP. The URDF
supplies every lower/upper bound, and the best finite state encountered within
`RETARGET_MAX_EVALUATIONS` is retained even if the evaluation cap stops SLSQP.
The previous solution becomes the next warm start. Every output is clipped to
its URDF limits and returned directly in radians, without output filtering.

The tracking loop and Viewer-process proxy retain the latest completed robot
result. The proxy repeats that result in subsequent snapshots, so replacing a
queued display frame cannot lose the newest robot pose. `WEB_FPS` throttling
and slow browser rendering may drop intermediate visual states, but they do not
add a render wait to the control path.

## Robot assets

The robot model is `assets/mmhand/urdf/mmhand_collision_coacd.urdf`. It and its
27 visual meshes plus 162 convex collision meshes come from
[`mm-hand/structure`](https://github.com/mm-hand/structure) commit
`04eb24bedc300419d8556de1e3848c6d4e344d4e`. This repository adds five fixed,
geometry-free virtual fingertip links used by retargeting. Every STL is tracked
with Git LFS, so `git lfs pull` is required after cloning.

## ROS 2

`python track.py --ros` publishes `std_msgs/msg/Float32MultiArray`:

| Topic | Shape | Meaning |
|---|---:|---|
| `/mmhand/keypoints` | `21 x 3` | source-frame points in metres |
| `/mmhand/finger_pad_directions` | `5 x 3` | source-frame outward unit vectors |
| `/mmhand/joint_angle_deg` | `21` | MMHand J00-J20 in degrees |

Every frame published on the keypoint topic has a matching direction message.
Keypoint and direction layouts append `:hand=Left` or `:hand=Right`. The layouts are:

```text
mmhand_teleop:keypoints:v2:source_m
mmhand_teleop:finger_pad_directions:v2:source_unit
mmhand:J00-J20:urdf_deg
```

The messages have no Header, timestamp, or frame sequence. Input topics are
published whenever points are present, even if the frame is not ready for
retargeting. Joint angles are published only for completed ready-left-hand
solves. The asynchronous solver means input and robot messages are not strictly
synchronized. Missing input does not publish a stop or zero-angle message.

## Repository structure

```text
input/
  frame.py              common frame, topology, and CMC geometry
  wilor/
    camera.py           latest-frame OpenCV capture
    source.py           ONNX inference, mesh geometry, box caching, filtering
    assets/              ONNX/MANO data and license notices
  manus/
    adapter.py          25-to-21 mapping and quaternion pad directions
    calibration.py      terminal Integrated calibration and file gate
    source.py           official SDK transport and common-frame source
    assets/              SDK bridge, headers, library, and documentation
assets/mmhand/           MMHand URDF, meshes, and licenses
config.py                all runtime settings and public ROS constants
one_euro.py              ndarray One Euro filter
retarget.py              URDF FK, analytic Jacobians, full-hand optimization
ros.py                   keypoint, pad-direction, and joint publishers
viewer.py                dashboard and both 3D views
track.py                 minimal runtime lifecycle
simulation/              optional SAPIEN grasp simulation
test_hand_capture.py     retarget, viewer, and ROS tests
test_wilor.py            camera, inference geometry, filter tests
test_manus_phase1.py     MANUS mapping, calibration, and stale-data tests
test_simulation.py       optional headless simulation tests
```

The source implementations end at `InputFrame`. Retargeting, visualization,
ROS, and simulation contain no input-device branches.

## Tests and licenses

```bash
python -m unittest -v test_hand_capture.py test_wilor.py test_manus_phase1.py
python -m unittest -v test_simulation.py
```

Simulation tests skip when SAPIEN is unavailable. WiLoR model terms are in
`input/wilor/assets/WILOR_MODEL_LICENSE.txt`; related notices are in
`input/wilor/assets/THIRD_PARTY_NOTICES.md`. MANUS SDK terms are distributed
with its assets.
