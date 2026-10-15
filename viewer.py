from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import config as C
from config import URDF_PATH, WEB_FPS, WEB_PORT, VIEW_PORTS
from input.frame import SKELETON_EDGES, hand0_middle_tip_distance
from retarget import compute_cmc_frame, human_thumb_geometry, human_vector_features

ROBOT_CAMERA_DISTANCE = 0.42
PALM_FRAME_AXIS_LENGTH = 0.04
PALM_FRAME_AXIS_RADIUS = 0.0015
PALM_FRAME_ORIGIN_RADIUS = 0.004
PAD_DIRECTION_LENGTH = 0.025
ARROW_SHAFT_RADIUS = 0.0008
ARROW_HEAD_RADIUS = 0.0025
ARROW_HEAD_LENGTH = 0.006
TIP_INDICES = np.array((4, 8, 12, 16, 20))
LOSS_LABELS = (
    ("thumb_tip", "thumb tip"),
    ("thumb_proximal_bend", "thumb proximal"),
    ("thumb_distal_bend", "thumb distal"),
    ("finger_angles", "finger angles"),
    ("fingertip_vectors", "fingertip vectors"),
    ("thumb_pad", "thumb pad"),
    ("finger_pads", "finger pads"),
    ("total", "total"),
)
ANGLE_COLORS = ((80, 230, 120), (190, 100, 255))
VECTOR_GROUPS = (
    "off", "thumb_tip", "fingertip_vectors", "thumb_pad", "finger_pads",
)
VECTOR_GROUP_LABELS = {
    "off": "Off",
    "thumb_tip": "Thumb tip",
    "fingertip_vectors": "Fingertip vectors",
    "thumb_pad": "Thumb pad",
    "finger_pads": "Finger pads",
}
VECTOR_ITEMS = {
    "thumb_tip": (("Thumb", (235, 80, 200)),),
    "fingertip_vectors": (
        ("Index", (80, 220, 100)),
        ("Middle", (60, 210, 235)),
        ("Ring", (245, 205, 60)),
        ("Little", (180, 100, 235)),
    ),
    "thumb_pad": (("Thumb", (235, 80, 200)),),
    "finger_pads": (
        ("Index", (80, 220, 100)),
        ("Middle", (60, 210, 235)),
        ("Ring", (245, 205, 60)),
        ("Little", (180, 100, 235)),
    ),
}


def _loss_text(losses=None):
    lines = ["Weighted retarget loss"]
    if losses:
        total = losses["total"]
        for name, label in LOSS_LABELS:
            value = losses.get(name)
            text = "N/A" if value is None else (
                f"{value:.3e}  {(100 * value / total if total else 0):5.1f}%"
            )
            lines.append(f"{label:<18}{text}")
    else:
        lines.append("waiting")
    return "\n".join(lines)


def _latency_text(value):
    """Format an optional non-negative latency value for the dashboard."""
    if value is None or not np.isfinite(value) or value < 0:
        return "waiting"
    return f"{value:.1f} ms"


def _render_time_text(value):
    """Format one server-side visualization update duration."""
    return f"render {_latency_text(value)}"


def _timing_breakdown_text(timings=None):
    """Format the measured components of one retargeted-frame latency."""
    labels = (
        ("worker_queue", "worker queue"),
        ("targets", "targets"),
        ("slsqp", "SLSQP"),
        ("final_loss", "final loss"),
        ("solver_overhead", "solver overhead"),
        ("solve_total", "solve total"),
    )
    timings = {} if timings is None else timings
    return "\n".join(
        f"{label:<16}{_latency_text(timings.get(name))}"
        for name, label in labels
    )


def _human_view_wxyz(handedness, points):
    """Face the palm toward -X and level the Index-to-Little MCP line."""
    right = handedness == "Right"
    points = np.asarray(points, float)
    side = points[5] - points[17]
    if right:
        side = side * (-1, -1, 1)
    roll = -np.arctan2(side[2], side[1])
    roll = (roll + np.pi / 2) % np.pi - np.pi / 2
    cosine, sine = np.cos(roll / 2), np.sin(roll / 2)
    return (0.0, 0.0, -sine, cosine) if right else (cosine, sine, 0.0, 0.0)


def _frame_wxyz(frame):
    quaternion = Rotation.from_matrix(np.asarray(frame, float)).as_quat()
    return quaternion[[3, 0, 1, 2]]


def _arrow_points(starts, directions):
    return np.stack(
        (starts, starts + PAD_DIRECTION_LENGTH * directions), axis=1
    ).astype(np.float32)


def _vector_group_info(group):
    """Describe the selected residual group and its exact solver weight."""

    if group == "off":
        return "Residual vectors hidden"
    if group == "thumb_tip":
        disabled = " · disabled" if C.RETARGET_THUMB_TIP_WEIGHT == 0 else ""
        return (
            f"thumb tip · weight {C.RETARGET_THUMB_TIP_WEIGHT:g}{disabled} · "
            f"scale {C.RETARGET_THUMB_TIP_SCALE:g} · "
            f"normalizer {1000 * C.RETARGET_POSITION_NORMALIZER_M:g} mm"
        )
    if group == "fingertip_vectors":
        return (
            f"fingertip vectors · shared weight "
            f"{C.RETARGET_FINGERTIP_VECTOR_WEIGHT:g} · "
            f"normalizer {1000 * C.RETARGET_POSITION_NORMALIZER_M:g} mm"
        )
    if group == "thumb_pad":
        configured = C.RETARGET_THUMB_PAD_WEIGHT
        return (
            f"thumb pad · configured weight {configured:g} · "
            f"solver weight {configured / 3:g}"
        )
    configured = (
        C.RETARGET_INDEX_PAD_WEIGHT,
        C.RETARGET_MIDDLE_PAD_WEIGHT,
        C.RETARGET_RING_PAD_WEIGHT,
        C.RETARGET_LITTLE_PAD_WEIGHT,
    )
    weights = " ".join(
        f"{name[0]}={weight:g}" for name, weight in zip(
            ("Index", "Middle", "Ring", "Little"), configured
        )
    )
    effective = " ".join(
        f"{name[0]}={weight / 3:g}" for name, weight in zip(
            ("Index", "Middle", "Ring", "Little"), configured
        )
    )
    return (
        f"finger pads · configured {weights} · solver effective {effective}"
    )


def _vector_label(name, vector, *, direction=False, angle_error=None):
    """Format one CMC-local position or direction vector annotation."""

    vector = np.asarray(vector, float)
    if direction:
        xyz = ", ".join(f"{value:+.3f}" for value in vector)
        angle = "" if angle_error is None else f"\nangle error {angle_error:.1f}°"
        return f"{name}\nxyz [{xyz}]\n|d| {np.linalg.norm(vector):.3f}{angle}"
    millimetres = 1000 * vector
    xyz = ", ".join(f"{value:+.1f}" for value in millimetres)
    return f"{name}\nxyz [{xyz}] mm\n|v| {np.linalg.norm(millimetres):.1f} mm"


def _pad_angle_errors(human_pads, robot_pads):
    """Return row-wise angular errors between two sets of unit pad vectors."""

    dots = np.einsum("ij,ij->i", human_pads, robot_pads)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def _human_residual_vector_data(points, directions):
    """Build scene geometry and exact loss targets for the human side."""

    points = np.asarray(points, float)
    directions = np.asarray(directions, float)
    _, _, thumb_tip, relative, pads = human_vector_features(points, directions)
    tips = points[TIP_INDICES]
    return {
        "thumb_tip": (points[1:2], points[4:5], thumb_tip),
        "fingertip_vectors": (
            np.repeat(points[4:5], 4, axis=0), tips[1:], relative,
        ),
        "thumb_pad": (
            tips[:1], tips[:1] + PAD_DIRECTION_LENGTH * directions[:1], pads[:1],
        ),
        "finger_pads": (
            tips[1:], tips[1:] + PAD_DIRECTION_LENGTH * directions[1:], pads[1:],
        ),
    }


def _robot_residual_vector_data(model, q):
    """Build scene geometry and exact robot loss feature vectors."""

    q = np.asarray(q, float)
    values = model.features(q)[0]
    tips, directions = model.fingertip_pads(q)
    return {
        "thumb_tip": (
            model.palm_position[None], tips[:1], values[0],
        ),
        "fingertip_vectors": (
            np.repeat(tips[:1], 4, axis=0), tips[1:], values[2],
        ),
        "thumb_pad": (
            tips[:1], tips[:1] + PAD_DIRECTION_LENGTH * directions[:1],
            values[3][:1],
        ),
        "finger_pads": (
            tips[1:], tips[1:] + PAD_DIRECTION_LENGTH * directions[1:],
            values[3][1:],
        ),
    }


def _rotate(vector, axis, angle):
    axis = axis / np.linalg.norm(axis)
    return (vector * np.cos(angle) + np.cross(axis, vector) * np.sin(angle)
            + axis * np.dot(axis, vector) * (1 - np.cos(angle)))


def _angle_segments(origin, start, axis, angle, radius, end=None):
    start, axis = np.asarray(start, float), np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    start -= axis * np.dot(start, axis)
    start /= np.linalg.norm(start)
    values = np.linspace(0, angle, 25)
    curve = origin + radius * np.asarray([_rotate(start, axis, value) for value in values])
    end = _rotate(start, axis, angle) if end is None else end / np.linalg.norm(end)
    curve[-1] = origin + radius * end
    rays = np.asarray(((origin, origin + radius * start),
                       (origin, origin + radius * end)))
    return np.concatenate((rays, np.stack((curve[:-1], curve[1:]), axis=1))).astype(np.float32)


def _human_angle_segments(points):
    points = np.asarray(points, float)
    lengths = np.linalg.norm(np.diff(points[1:5], axis=0), axis=1)
    segments, angles, axes = human_thumb_geometry(points)
    return angles, tuple(
        _angle_segments(points[joint], segments[index], axes[index], angles[index],
                        .35 * min(lengths[index:index + 2]), segments[index + 1])
        for index, joint in enumerate((2, 3))
    )


def _angle_text(
    title, labels, angles=None, palm_width=None, middle_tip_x=None,
    raw_palm_length=None, raw_palm_width=None,
    palm_width_label="palm width", middle_tip_label="palm length",
):
    """Format thumb angles, palm width, and the CMC-frame Middle-tip x value."""

    lines = [title]
    if angles is None:
        lines.append("waiting")
    else:
        lines.extend(
            f"{label:<8}{np.degrees(value):+6.1f}°"
            for label, value in zip(labels, angles)
        )
    if palm_width is not None and np.isfinite(palm_width) and palm_width >= 0:
        lines.append(f"{palm_width_label} {1000 * palm_width:6.1f} mm")
    if middle_tip_x is not None and np.isfinite(middle_tip_x):
        lines.append(f"{middle_tip_label} {1000 * middle_tip_x:6.1f} mm")
    if raw_palm_length is not None and np.isfinite(raw_palm_length):
        lines.append(f"raw palm length {1000 * raw_palm_length:6.1f} mm")
    if raw_palm_width is not None and np.isfinite(raw_palm_width):
        lines.append(f"raw palm width {1000 * raw_palm_width:6.1f} mm")
    return "\n".join(lines)


def _robot_camera_pose(model):
    tips = model.fingertips(model.seed)
    look_at = np.vstack((model.palm_position, tips)).mean(0)
    palm_normal = model.palm_frame[:, 2]
    transforms = model.fk(model.seed)
    base = (
        transforms["finger_1_proximal_phalanx_1"][:3, 3]
        - transforms["finger_4_proximal_phalanx_1"][:3, 3]
    )
    normal = palm_normal - base * np.dot(palm_normal, base) / np.dot(base, base)
    normal /= np.linalg.norm(normal)
    return look_at - ROBOT_CAMERA_DISTANCE * normal, look_at, (
        0.9993984744,
        -0.0346479515,
        -0.0014862239,
    )


class Viewer:
    def __init__(self, model):
        import viser
        from viser.extras import ViserUrdf

        self.servers = [viser.ViserServer(port=port) for port in VIEW_PORTS]
        self.model = model
        input_server, robot_server = self.servers
        self._camera(
            input_server,
            (-0.26, 0, 0.06),
            (0, 0, 0.06),
            (0, 0.2588190451, 0.9659258263),
            42,
        )
        self._camera(robot_server, *_robot_camera_pose(model), 42)
        self.human_frame = input_server.scene.add_frame("/hand", show_axes=False)
        self.human_retarget_frame = input_server.scene.add_frame(
            "/hand/retarget_frame",
            show_axes=True,
            axes_length=PALM_FRAME_AXIS_LENGTH,
            axes_radius=PALM_FRAME_AXIS_RADIUS,
            origin_radius=PALM_FRAME_ORIGIN_RADIUS,
            visible=False,
        )
        empty_hand = np.zeros((21, 3), np.float32)
        self.human_cloud = input_server.scene.add_point_cloud(
            "/hand/points",
            empty_hand,
            (60, 170, 255),
            point_size=0.004,
            point_shape="circle",
            visible=False,
        )
        self.human_bones = input_server.scene.add_line_segments(
            "/hand/bones",
            empty_hand[np.asarray(SKELETON_EDGES)],
            (60, 170, 255),
            line_width=4,
            visible=False,
        )
        self.pad_directions = input_server.scene.add_arrows(
            "/hand/pad_directions",
            np.zeros((5, 2, 3), np.float32),
            (255, 150, 70),
            shaft_radius=ARROW_SHAFT_RADIUS,
            head_radius=ARROW_HEAD_RADIUS,
            head_length=ARROW_HEAD_LENGTH,
            visible=False,
        )
        self.human_angle_arcs = tuple(
            input_server.scene.add_line_segments(
                f"/hand/thumb_{name}_angle", np.zeros((26, 2, 3), np.float32),
                color, line_width=4, visible=False,
            )
            for name, color in zip(("mcp", "ip"), ANGLE_COLORS)
        )
        robot_server.scene.add_frame("/robot", show_axes=False)
        self.urdf = ViserUrdf(robot_server, URDF_PATH, root_node_name="/robot")
        self.robot_palm_frame = robot_server.scene.add_frame(
            "/robot/palm_frame",
            show_axes=True,
            position=model.palm_position,
            wxyz=_frame_wxyz(model.palm_frame),
            axes_length=PALM_FRAME_AXIS_LENGTH,
            axes_radius=PALM_FRAME_AXIS_RADIUS,
            origin_radius=PALM_FRAME_ORIGIN_RADIUS,
        )
        self.urdf_names = self.urdf.get_actuated_joint_names()
        self.robot_index = model.index
        self.urdf.update_cfg(
            np.asarray([model.seed[self.robot_index[name]] for name in self.urdf_names])
        )
        starts, directions = model.fingertip_pads(model.seed)
        self.robot_pad_directions = robot_server.scene.add_arrows(
            "/robot/pad_directions",
            _arrow_points(starts, directions),
            (255, 150, 70),
            shaft_radius=ARROW_SHAFT_RADIUS,
            head_radius=ARROW_HEAD_RADIUS,
            head_length=ARROW_HEAD_LENGTH,
        )
        self.robot_angle_arcs = tuple(
            robot_server.scene.add_line_segments(
                f"/robot/thumb_{name}_angle", np.zeros((26, 2, 3), np.float32),
                color, line_width=4, visible=False,
            )
            for name, color in zip(("pip", "dip"), ANGLE_COLORS)
        )
        self.vector_lock = threading.Lock()
        self.vector_group = "fingertip_vectors"
        self.human_vector_handles = self._add_residual_vector_handles(
            input_server, "/hand/residual_vectors"
        )
        self.robot_vector_handles = self._add_residual_vector_handles(
            robot_server, "/robot/residual_vectors"
        )
        self.vector_available = {"human": False, "robot": False}
        self.last_robot = None
        self.last_update, self.status = 0.0, "WAITING"
        self.loss_text = _loss_text()
        self.human_angle_text = _angle_text("Input thumb", (), None)
        self.robot_angle_text = _angle_text("MMHand thumb", (), None)
        self.input_latency_text = self.retarget_latency_text = "waiting"
        self.input_render_text = self.robot_render_text = _render_time_text(None)
        self.retarget_timings_text = _timing_breakdown_text()
        self.preview = cv2.imencode(".jpg", np.zeros((360, 1280, 3), np.uint8))[1].tobytes()
        self._start_dashboard()
        print(f"Dashboard: http://localhost:{WEB_PORT}")

    @staticmethod
    def _add_residual_vector_handles(server, root):
        """Create hidden arrows and labels for every residual vector group."""

        handles = {}
        for group, items in VECTOR_ITEMS.items():
            handles[group] = []
            for index, (name, color) in enumerate(items):
                path = f"{root}/{group}/{index}_{name.lower()}"
                arrow = server.scene.add_arrows(
                    f"{path}/arrow",
                    np.zeros((1, 2, 3), np.float32),
                    color,
                    shaft_radius=ARROW_SHAFT_RADIUS,
                    head_radius=ARROW_HEAD_RADIUS,
                    head_length=ARROW_HEAD_LENGTH,
                    visible=False,
                )
                label = server.scene.add_label(
                    f"{path}/label", name,
                    position=(0.0, 0.0, 0.0),
                    visible=False,
                    font_screen_scale=0.8,
                    anchor="bottom-center",
                )
                handles[group].append((arrow, label))
        return handles

    @staticmethod
    def _write_residual_vector_handles(handles, data, pad_errors=None):
        """Update one side's arrows and CMC-local value labels."""

        for group, items in handles.items():
            starts, ends, vectors = data[group]
            direction = group in ("thumb_pad", "finger_pads")
            errors = None if pad_errors is None else pad_errors.get(group)
            for row, ((name, _), (arrow, label)) in enumerate(
                zip(VECTOR_ITEMS[group], items)
            ):
                arrow.points = np.asarray(((starts[row], ends[row]),), np.float32)
                label.position = ends[row] if direction else .5 * (starts[row] + ends[row])
                label.text = _vector_label(
                    name, vectors[row], direction=direction,
                    angle_error=None if errors is None else errors[row],
                )

    def _apply_vector_visibility(self):
        """Show only the selected group on each side when its data is valid."""

        for side, handles in (
            ("human", self.human_vector_handles),
            ("robot", self.robot_vector_handles),
        ):
            for group, items in handles.items():
                visible = self.vector_available[side] and group == self.vector_group
                for arrow, label in items:
                    arrow.visible = label.visible = visible

    def set_vector_group(self, group):
        """Validate and apply a dashboard residual-vector selection."""

        if group not in VECTOR_GROUPS:
            raise ValueError(f"Unknown residual vector group: {group}")
        with self.vector_lock:
            self.vector_group = group
            self._apply_vector_visibility()

    def _update_residual_vectors(self, frame, robot):
        """Refresh paired human targets and robot feature vectors."""

        human = robot_data = None
        if frame.points is not None and frame.finger_pad_directions is not None:
            try:
                human = _human_residual_vector_data(
                    frame.points, frame.finger_pad_directions
                )
            except ValueError:
                human = None
        if robot is not None:
            robot_data = _robot_residual_vector_data(self.model, robot)
        pad_errors = None
        if human is not None and robot_data is not None:
            all_errors = _pad_angle_errors(
                np.vstack((human["thumb_pad"][2], human["finger_pads"][2])),
                np.vstack((robot_data["thumb_pad"][2], robot_data["finger_pads"][2])),
            )
            pad_errors = {
                "thumb_pad": all_errors[:1],
                "finger_pads": all_errors[1:],
            }
        with self.vector_lock:
            if human is not None:
                self._write_residual_vector_handles(
                    self.human_vector_handles, human, pad_errors
                )
            if robot_data is not None:
                self._write_residual_vector_handles(
                    self.robot_vector_handles, robot_data, pad_errors
                )
            self.vector_available["human"] = human is not None
            self.vector_available["robot"] = robot_data is not None
            self._apply_vector_visibility()

    @staticmethod
    def _camera(server, position, look_at, up, fov):
        server.scene.set_up_direction(up)
        server.gui.configure_theme(show_logo=False, show_share_button=False)
        server.initial_camera.position = position
        server.initial_camera.look_at = look_at
        server.initial_camera.up = up
        server.initial_camera.fov = np.radians(fov)

    def _start_dashboard(self):
        owner = self
        input_port, robot_port = [server.get_port() for server in self.servers]
        page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>mmhand_teleop</title><style>
*{{box-sizing:border-box}} body{{margin:0;height:100vh;background:#101318;color:#eef;
font:15px sans-serif;display:grid;grid-template:40vh 60vh/repeat(2,1fr);gap:6px}}
.panel{{position:relative;overflow:hidden;background:#181d25;border:1px solid #343b48}}
.input{{grid-column:1/3}} h2{{position:absolute;z-index:2;margin:0;padding:8px 12px;
font-size:15px;background:#101318cc;border-radius:0 0 6px 0}}
img,iframe{{width:100%;height:100%;display:block;border:0;object-fit:contain}}
#status{{color:#9bd;margin-left:12px;font-weight:normal}}
.latency{{color:#9bd;margin-left:10px;font-size:13px;font-weight:normal}}
.render-time{{display:block;color:#b8c5d6;margin-top:3px;font-size:12px;
font-weight:normal}}
#losses{{position:absolute;z-index:2;right:0;top:0;margin:0;padding:9px 12px;
background:#101318dd;color:#bde2ff;font:13px/1.35 monospace;pointer-events:none}}
.angles{{position:absolute;z-index:3;right:0;top:0;margin:0;padding:9px 12px;
background:#101318dd;color:#dff;font:13px/1.4 monospace;pointer-events:none}}
.timings{{position:absolute;z-index:3;left:0;top:66px;margin:0;padding:8px 12px;
background:#101318dd;color:#b8c5d6;font:12px/1.4 monospace;pointer-events:none}}
.vector-control{{position:fixed;z-index:10;left:50%;top:calc(40vh + 10px);
transform:translateX(-50%);padding:6px 10px;border:1px solid #526071;
border-radius:6px;background:#101318ee;color:#dff;font:12px sans-serif}}
.vector-control select{{margin:0 7px;background:#202834;color:#eef;border:1px solid #607086}}
#vectorInfo{{color:#9bd}}</style></head><body>
<section class="panel input"><h2>Input <span id="status">WAITING</span></h2>
<pre id="losses">Weighted retarget loss\nwaiting</pre><img id="preview"></section>
<div class="vector-control">Residual vectors
<select id="vectorGroup">
<option value="off">Off</option><option value="thumb_tip">Thumb tip</option>
<option value="fingertip_vectors" selected>Fingertip vectors</option>
<option value="thumb_pad">Thumb pad</option><option value="finger_pads">Finger pads</option>
</select><span id="vectorInfo"></span></div>
<section class="panel"><h2>Input hand <span class="latency" id="inputLatency">waiting</span><span class="render-time" id="inputRender">render waiting</span></h2><pre class="angles" id="humanAngles">Input thumb\nwaiting</pre><iframe id="input"></iframe></section>
<section class="panel"><h2>Retargeted MMHand <span class="latency" id="retargetLatency">waiting</span><span class="render-time" id="robotRender">render waiting</span></h2><pre class="timings" id="retargetTimings">waiting</pre><pre class="angles" id="robotAngles">MMHand thumb\nwaiting</pre><iframe id="robot"></iframe></section>
<script>
const host=location.hostname, statusText=document.getElementById("status"),
previewImage=document.getElementById("preview"),lossText=document.getElementById("losses"),
humanAngles=document.getElementById("humanAngles"),robotAngles=document.getElementById("robotAngles");
const inputLatency=document.getElementById("inputLatency"),
retargetLatency=document.getElementById("retargetLatency"),
inputRender=document.getElementById("inputRender"),
robotRender=document.getElementById("robotRender");
const retargetTimings=document.getElementById("retargetTimings"),
vectorGroup=document.getElementById("vectorGroup"),vectorInfo=document.getElementById("vectorInfo");
for(const [id,port] of [["input",{input_port}],["robot",{robot_port}]])
  document.getElementById(id).src=`http://${{host}}:${{port}}`;
vectorGroup.addEventListener("change",async()=>{{
  try{{
    const response=await fetch("/vector-group",{{method:"POST",
      headers:{{"Content-Type":"application/json"}},
      body:JSON.stringify({{group:vectorGroup.value}})}});
    if(!response.ok) throw new Error(await response.text());
  }}catch(error){{console.debug("vector group update failed",error)}}
}});
let previewUrl=null;
async function refresh(){{
  try{{
    const responses=await Promise.all([
      fetch("/preview.jpg?t="+Date.now()),fetch("/state")]);
    const [previewBlob,state]=await Promise.all([
      responses[0].blob(),responses[1].json()]);
    const nextUrl=URL.createObjectURL(previewBlob);
    previewImage.src=nextUrl;
    if(previewUrl!==null) URL.revokeObjectURL(previewUrl);
    previewUrl=nextUrl;
    statusText.textContent=state.status;lossText.textContent=state.losses;
    humanAngles.textContent=state.human_angles;robotAngles.textContent=state.robot_angles;
    inputLatency.textContent=state.input_latency;
    retargetLatency.textContent=state.retarget_latency;
    inputRender.textContent=state.input_render;
    robotRender.textContent=state.robot_render;
    retargetTimings.textContent=state.retarget_timings;
    if(document.activeElement!==vectorGroup) vectorGroup.value=state.vector_group;
    vectorInfo.textContent=state.vector_group_info;
  }}catch(error){{console.debug("dashboard refresh failed",error)}}
  setTimeout(refresh,{round(1000 / WEB_FPS)});
}}
refresh();
</script></body></html>""".encode()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/preview.jpg":
                    body, content_type = owner.preview, "image/jpeg"
                elif path == "/state":
                    body = json.dumps({
                        "status": owner.status,
                        "losses": owner.loss_text,
                        "human_angles": owner.human_angle_text,
                        "robot_angles": owner.robot_angle_text,
                        "input_latency": owner.input_latency_text,
                        "retarget_latency": owner.retarget_latency_text,
                        "input_render": owner.input_render_text,
                        "robot_render": owner.robot_render_text,
                        "retarget_timings": owner.retarget_timings_text,
                        "vector_group": owner.vector_group,
                        "vector_group_info": _vector_group_info(owner.vector_group),
                    }).encode()
                    content_type = "application/json"
                elif path == "/":
                    body, content_type = page, "text/html; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self):
                if self.path.split("?", 1)[0] != "/vector-group":
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 1024:
                        raise ValueError("Invalid request size")
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError("Expected a JSON object")
                    owner.set_vector_group(payload.get("group"))
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    self.send_error(400, str(error))
                    return
                body = json.dumps({
                    "vector_group": owner.vector_group,
                    "vector_group_info": _vector_group_info(owner.vector_group),
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_):
                pass

        self.http = ThreadingHTTPServer(("0.0.0.0", WEB_PORT), Handler)
        self.http_thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.http_thread.start()

    def update(
        self,
        frame,
        robot=None,
        losses=None,
        input_latency_ms=None,
        retarget_latency_ms=None,
        retarget_timings_ms=None,
    ):
        now = time.monotonic()
        if now - self.last_update < 1 / WEB_FPS:
            return False
        self.last_update = now
        self.status = frame.status
        self.input_latency_text = _latency_text(
            input_latency_ms if frame.points is not None else None
        )
        if losses is not None:
            self.loss_text = _loss_text(losses)
        elif frame.points is None or frame.handedness != "Left":
            self.loss_text = _loss_text()
        preview = frame.preview
        if preview is None:
            preview = np.zeros((360, 1280, 3), np.uint8)
        ok, encoded = cv2.imencode(".jpg", preview, (cv2.IMWRITE_JPEG_QUALITY, 82))
        if ok:
            self.preview = encoded.tobytes()
        input_render_started = time.perf_counter()
        points = frame.points
        if points is not None and frame.handedness in ("Left", "Right"):
            self.human_frame.wxyz = _human_view_wxyz(frame.handedness, points)
        visible = points is not None
        self.human_cloud.visible = self.human_bones.visible = visible
        self.pad_directions.visible = False
        for arc in self.human_angle_arcs:
            arc.visible = False
        if not visible:
            self.human_retarget_frame.visible = False
            self.human_angle_text = _angle_text("Input thumb", (), None)
        else:
            points = np.asarray(points, float)
            self.human_cloud.points = points.astype(np.float32)
            self.human_bones.points = points[np.asarray(SKELETON_EDGES)].astype(np.float32)
            directions = np.asarray(frame.finger_pad_directions, float)
            self.pad_directions.points = _arrow_points(points[TIP_INDICES], directions)
            self.pad_directions.visible = True
            _, arcs = _human_angle_segments(points)
            for handle, segments in zip(self.human_angle_arcs, arcs):
                handle.points, handle.visible = segments, True
            initial_angles = frame.initial_joint_angles
            middle_tip_x = None
            try:
                origin, palm_frame = compute_cmc_frame(points)
            except ValueError:
                self.human_retarget_frame.visible = False
            else:
                middle_tip_x = hand0_middle_tip_distance(points)
                self.human_retarget_frame.position = origin
                self.human_retarget_frame.wxyz = _frame_wxyz(palm_frame)
                self.human_retarget_frame.visible = True
            self.human_angle_text = _angle_text(
                "Input thumb", ("proximal", "distal"),
                None if initial_angles is None else initial_angles.thumb_bends,
                np.linalg.norm(points[5] - points[17]), middle_tip_x,
                frame.raw_palm_length, frame.raw_palm_width,
            )
        self.input_render_text = _render_time_text(
            (time.perf_counter() - input_render_started) * 1000.0
            if visible else None
        )
        if robot is not None:
            self.last_robot = np.asarray(robot, float).copy()
            robot_render_started = time.perf_counter()
            self.retarget_latency_text = _latency_text(retarget_latency_ms)
            self.retarget_timings_text = _timing_breakdown_text(
                retarget_timings_ms
            )
            self.urdf.update_cfg(
                np.asarray([robot[self.robot_index[name]] for name in self.urdf_names])
            )
            starts, directions = self.model.fingertip_pads(robot)
            self.robot_pad_directions.points = _arrow_points(starts, directions)
            origins, axes, incoming = self.model.thumb_joint_frames(robot)
            for row, handle in enumerate(self.robot_angle_arcs):
                length = .35 * np.linalg.norm(incoming[row])
                handle.points = _angle_segments(
                    origins[row], incoming[row], axes[row], robot[18 + row], length
                )
                handle.visible = True
            self.robot_angle_text = _angle_text(
                "MMHand thumb", ("J18/PIP", "J19/DIP"), robot[18:20],
                self.model.palm_width(robot),
                self.model.urdf_palm_length,
                palm_width_label="MMHand palm width",
                middle_tip_label="MMHand URDF palm length",
            )
            self.robot_render_text = _render_time_text(
                (time.perf_counter() - robot_render_started) * 1000.0
            )
        elif not visible or frame.handedness != "Left":
            self.last_robot = None
            self.retarget_latency_text = "waiting"
            self.retarget_timings_text = _timing_breakdown_text()
            self.robot_render_text = _render_time_text(None)
            for arc in self.robot_angle_arcs:
                arc.visible = False
            self.robot_angle_text = _angle_text("MMHand thumb", (), None)
        self._update_residual_vectors(frame, self.last_robot)
        return True

    def close(self):
        self.http.shutdown()
        self.http.server_close()
        for server in self.servers:
            server.stop()
