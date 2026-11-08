"""On the station table, centred between the two arms: an enclosed desktop 3D printer (36 x 35 x 40 cm, fixed), its
clear door shut. A spool of orange 1.75 mm PLA filament hangs on the spool holder on the printer's left wall: a 45 mm
steel rod sticking out to the robot's left, 22 cm above the table. On the roof, at the back right, stands the
filament feeder, a black block with a brass inlet on its left face, 7.5 cm above the roof, facing the spool's side: a
funnel 6 mm wide narrowing over 5 mm to a 2.4 mm bore, 3 cm deep, closed at its end. The filament comes off the front
of the spool, runs down to the table and round the printer's front left corner, and its last 28 cm lie straight
along y in front of the printer, over a pair of steel 1-2-3 blocks, the cut end toward the robot's right.

The task: push the filament's end into the feeder's inlet until 2 cm or more is inside, then let go.

The filament is flexible: 27 links 2.5 cm long on springy, damped joints (``filament.py``), its rest shape the shape
it is laid in, anchored to the spool, which turns on a damped revolute joint about the holder's rod when the filament
is pulled. The inlet's axis runs along -y, 47.5 cm above the table, so the hands carry the filament's straight end up
over the roof, the right hand 8.75 cm from the end and the left hand 17.5 cm further back, each in its own half of the
table, and feed it in together. ``inlet.py`` builds the funnel and the bore from box colliders, ``printer.py`` holds
the printer's geometry, the scorer reads each gripper's fingers to know the filament was let go, and ``looks.py``
draws the parts.

Run it:  inspect-robots run --embodiment yam_arena -E task=filament_into_extruder.py --scorer task_score ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import printer as pr  # noqa: E402
from bonds import on_reset, on_step  # noqa: E402
from colliders import shape  # noqa: E402
from filament import Spool, anchor, bends, lay, matrix, register  # noqa: E402
from grip import HandWaypoint  # noqa: E402
from inlet import Head, head  # noqa: E402
from looks import dress  # noqa: E402
from yam_arena.demo import FINGER_CLOSED  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0


def _at(x, y, h):
    """A station-frame position from a height above the table."""
    return (x, y, TOP + h)


# -- the printer, fixed, its door shut, the print head parked, the plate drawn on the bed ----------------------------
PRINTER = Box("printer", pos=_at(*pr.CENTRE), size=pr.SIZE, color=(0.78, 0.78, 0.8))
shape("printer", pr.printer_parts(bed_top=pr.PLATE_TOP))
DOOR = Box("door", pos=_at(*pr.DOOR_CENTRE), size=pr.DOOR_SIZE, color=(0.45, 0.52, 0.58))
shape("door", pr.door_parts())
PRINT_HEAD = Box("print_head", pos=_at(*pr.HEAD_CENTRE), size=pr.HEAD_SIZE, color=(0.78, 0.79, 0.8))

# -- the spool holder on the left wall: only its bracket collides (the spool turns on a joint about the rod's axis) --
HOLDER = Box("holder", pos=_at(*pr.HOLDER_CENTRE), size=pr.HOLDER_SIZE, color=(0.04, 0.04, 0.045))
shape("holder", [pr.bracket_part()])

# -- the feeder on the roof: its inlet's axis runs along -y (into the feeder), the mouth on its +y face ---------------
AXIS = (0.0, -1.0)  # unit vector, into the inlet
ACROSS = (-AXIS[1], AXIS[0])  # unit vector across the line, to its left (+x)
INLET_Z = TOP + pr.INLET_H
MOUTH = (pr.INLET_X, pr.MOUTH_Y)  # the funnel's mouth (the feeder's face), on the line
FUNNEL, BORE = pr.FUNNEL, pr.BORE
FUNNEL_DEPTH, BORE_DEPTH = pr.FUNNEL_DEPTH, pr.BORE_DEPTH
CHANNEL = FUNNEL_DEPTH + BORE_DEPTH  # m from the mouth to the bore's closed end
FEEDER_YAW = -math.pi / 2  # the box's x (the channel) along the station's -y
FEEDER = head(Head("feeder", pos=_at(*pr.FEEDER_CENTRE), size=pr.FEEDER_SIZE, yaw=FEEDER_YAW,
                   color=(0.03, 0.03, 0.035), inlet_z=pr.INLET_H - pr.FEEDER_CENTRE[2], funnel=FUNNEL, bore=BORE,
                   funnel_depth=FUNNEL_DEPTH, bore_depth=BORE_DEPTH))


def on_line(s: float, t: float = 0.0) -> tuple[float, float]:
    """The table point ``s`` metres along the line from the mouth (positive into the feeder) and ``t`` to its left."""
    return (MOUTH[0] + s * AXIS[0] + t * ACROSS[0], MOUTH[1] + s * AXIS[1] + t * ACROSS[1])


# -- the filament's end: straight along y in front of the printer, over the block, its end toward the right ----------
FIL_D = 0.00175  # m: 1.75 mm filament
R_FIL = FIL_D / 2
PICK_X = 0.25  # the filament's straight end lies along this x
TIP0 = (PICK_X, MOUTH[1] + 0.01)  # the free end at the start: 1 cm short of the mouth, in y
BLOCK_SIZE = (0.0762, 0.0508, 0.0254)  # a steel 1-2-3 block lying flat, its 3 inch side along the filament
BLOCK_AT = 0.165  # m: the block's centre, back from the filament's end (between the hands)
END_BLOCK_AT = 0.018  # m: the second block's centre, under the filament's end (the end 2 cm in from its far side), so
# the right hand takes the filament level, between the two blocks, not on a sagging overhang


def from_tip(s: float) -> tuple[float, float]:
    """The table point ``s`` metres from the filament's starting end, back along it (toward the spool, +y)."""
    return (TIP0[0] - s * AXIS[0], TIP0[1] - s * AXIS[1])


BLOCK = Box("block", pos=(*from_tip(BLOCK_AT), TOP + BLOCK_SIZE[2] / 2), size=BLOCK_SIZE, yaw=math.pi / 2,
            color=(0.45, 0.46, 0.48))
END_BLOCK = Box("end_block", pos=(*from_tip(END_BLOCK_AT), TOP + BLOCK_SIZE[2] / 2), size=BLOCK_SIZE, yaw=math.pi / 2,
                color=(0.45, 0.46, 0.48))
Z_BLOCK = TOP + BLOCK_SIZE[2] + R_FIL  # the filament's axis on the block
Z_TABLE = TOP + R_FIL  # and on the table

# -- the filament's path, from its free end back to the spool ---------------------------------------------------------
# Seen from above: straight along +y, then a quarter curl clockwise (6 cm radius) to run along +x beside the printer's
# left wall, in the spool's middle plane. It drops from the block to the table along the way (a half cosine). Then it
# bends up (4 cm radius) under the front of the spool's winding and rises straight to it, where it comes off the
# spool. The links are laid by length along it.
LINK = 0.025  # m: each link
LINK_MASS = 0.001  # kg: 40 g/m (real 1.75 mm PLA: 3 g/m)
CURL_R = 0.06
STRAIGHT = pr.SPOOL_Y - CURL_R - TIP0[1]  # the straight's plan length
DROP = (0.28, 0.39)  # m along the path in plan: from the block's height down to the table
RISE_R = 0.065  # m: the bend up under the spool
X_UP = pr.ROD_X - pr.WIND_R - R_FIL  # the rise's x: the filament's axis against the front of the winding


def _plan(t: float) -> np.ndarray:
    """The flat part of the path seen from above, ``t`` metres along it from the free end."""
    tip = np.array(TIP0)
    if t <= STRAIGHT:
        return tip + t * np.array([0.0, 1.0])
    centre = np.array([TIP0[0] + CURL_R, TIP0[1] + STRAIGHT])
    arc = CURL_R * math.pi / 2
    if t <= STRAIGHT + arc:
        a = math.pi - (t - STRAIGHT) / CURL_R
        return centre + CURL_R * np.array([math.cos(a), math.sin(a)])
    return centre + np.array([0.0, CURL_R]) + (t - STRAIGHT - arc) * np.array([1.0, 0.0])


def _flat_height(t: float) -> float:
    if t <= DROP[0]:
        return Z_BLOCK
    if t < DROP[1]:
        return Z_TABLE + (Z_BLOCK - Z_TABLE) * (1 + math.cos(math.pi * (t - DROP[0]) / (DROP[1] - DROP[0]))) / 2
    return Z_TABLE


def _table() -> tuple[np.ndarray, np.ndarray]:
    """Points along the whole path every 0.5 mm, and their length along it in 3D."""
    flat_end = STRAIGHT + CURL_R * math.pi / 2 + (X_UP - RISE_R - (TIP0[0] + CURL_R))
    pts = [[*_plan(t), _flat_height(t)] for t in np.arange(0.0, flat_end, 0.0005)]
    corner = np.array([X_UP - RISE_R, pr.SPOOL_Y, Z_TABLE + RISE_R])  # the bend's centre
    for a in np.arange(0.0, math.pi / 2, 0.0005 / RISE_R):
        pts.append(list(corner + RISE_R * np.array([math.sin(a), 0.0, -math.cos(a)])))
    for h in np.arange(Z_TABLE + RISE_R, TOP + pr.ROD_H + 0.05, 0.0005):
        pts.append([X_UP, pr.SPOOL_Y, h])
    pts = np.asarray(pts)
    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return pts, length


_PTS, _LEN = _table()
# As many links as reach the front of the winding at the rod's height, where the filament comes off the spool.
N_LINKS = int(round(float(np.interp(TOP + pr.ROD_H, _PTS[_PTS[:, 0] >= X_UP - 1e-9, 2],
                                    _LEN[_PTS[:, 0] >= X_UP - 1e-9])) / LINK))
FIL_LEN = LINK * N_LINKS  # from the free end to the spool


def centreline(s: float) -> tuple[float, float, float]:
    """The filament's axis ``s`` metres along it from its free end, as laid."""
    return tuple(float(np.interp(s, _LEN, _PTS[:, k])) for k in range(3))


FILAMENT = lay("fil_", centreline, N_LINKS, LINK, FIL_D, LINK_MASS, (1.0, 0.42, 0.0), inertia=1e-6, spin_damping=0.5)
LINKS = [link.name for link in FILAMENT]
BEND_K = 0.064  # N m/rad per joint: 1.75 mm PLA (E 3.5 GPa) over a 2.5 cm link, EI / L
BEND_C = 0.002  # N m s/rad per joint

# -- the spool on the holder's rod, turning about it (its box's x along the station's y) -------------------------------
SPOOL = Spool("spool", pos=_at(pr.ROD_X, pr.SPOOL_Y, pr.ROD_H), size=pr.SPOOL_SIZE, mass=0.5, yaw=math.pi / 2,
              color=(1.0, 0.42, 0.0), wind_r=pr.WIND_R, flange_r=pr.SPOOL_R, width=pr.SPOOL_W, flange_t=pr.FLANGE_T,
              damping=0.01, hold=60.0)

JOINTS = bends(FILAMENT, BEND_K, BEND_C) + [anchor(SPOOL, FILAMENT[-1], BEND_K, BEND_C, hold=25.0)]
register(FILAMENT, JOINTS, SPOOL)
ANCHOR_LOCAL = np.array(JOINTS[-1].pos0)  # where the filament leaves the spool, in the spool's frame (its front)

OBJECTS = [PRINTER, DOOR, PRINT_HEAD, HOLDER, FEEDER, BLOCK, END_BLOCK, SPOOL, *FILAMENT]

# What the boxes look like: meshes modelled in Blender (looks/make_looks.py), the colliders unchanged. The build plate
# is drawn on the printer (the bed's collider top is the plate's top).
_PLATE_IN_PRINTER = (pr.PLATE_XY[0] - pr.CENTRE[0], pr.PLATE_XY[1] - pr.CENTRE[1],
                     pr.BED_TOP + pr.PLATE_SIZE[2] / 2 - pr.CENTRE[2])
LOOKS = {"printer": ["printer.usdc", ("plate.usdc", _PLATE_IN_PRINTER)], "door": "door.usdc",
         "print_head": "head.usdc", "holder": "holder.usdc", "feeder": "feeder.usdc", "block": "block.usdc",
         "end_block": "block.usdc",
         "spool": "spool.usdc", LINKS[0]: "tip.usdc"}
LOOKS.update({name: "link.usdc" for name in LINKS[1:]})
dress(LOOKS)

# -- scoring ---------------------------------------------------------------------------------------------------------
FULL_DEPTH = 0.02  # m: the end this far past the mouth scores 1.0 (let go and still)
PART_DEPTH = 0.005  # m: this far scores 0.5
NEAR_MOUTH = 0.01  # m: an end this close to the mouth's centre scores 0.25
ON_AXIS = 0.0025  # m: the end within this of the inlet's axis is in the funnel or the bore
KINK = 60.0  # degrees: two links this far out of line kink the filament (a bend radius of 2.4 cm)
ATTACHED = 0.005  # m: the filament's last link this close to its point on the spool is still wound there
ON_AXLE = 0.01  # m: the spool's centre this close to where it started is still on its rod
HOLD_REACH = 0.015  # m: a closed gripper's TCP this close to a link's axis holds the filament
HOLD_GAP = 0.003  # m: pads at most this much wider apart than the filament are closed on it
OFF_TABLE_Z = -0.02  # m above the table: a link or the spool's centre lower than this has fallen off
STILL_MOVE = 0.002  # m: every link, and the filament's point on the spool, moved less than this over the last second
STILL_WINDOW = 20  # control steps: that second, at 20 Hz
STILL_SPEED, STILL_SPIN = 0.02, 0.5  # m/s and rad/s: before a second has been recorded, every link and the spool
# slower than this is still (a link's roll about its own axis is not counted: on a round filament it moves nothing)
MOUTH_3D = np.array([*MOUTH, INLET_Z])
AXIS_3D = np.array([*AXIS, 0.0])
FREE = [*LINKS, "spool"]

# What the episode remembers: whether a gripper is closed on the filament, the sharpest bend so far (degrees), and
# where each link and the filament's point on the spool were over the last second, read every step. A kink in PLA
# stays, so the peak counts.
STATE = {"held": False, "peak_bend": 0.0, "track": []}


@on_reset
def reset_memory(env=None) -> None:
    STATE.update(held=False, peak_bend=0.0, track=[])


def _finite(objects) -> bool:
    return all(bool(np.all(np.isfinite(objects[n].pos)) and np.all(np.isfinite(objects[n].quat_xyzw))
                    and np.all(np.isfinite(objects[n].lin_vel))) for n in FREE)


def _axis(state) -> np.ndarray:
    return matrix(state.quat_xyzw)[:, 0]


def link_ends(objects) -> np.ndarray:
    """Each link's two ends, (n, 2, 3): its end toward the free end first."""
    out = []
    for name in LINKS:
        half = LINK / 2 * _axis(objects[name])
        out.append([objects[name].pos + half, objects[name].pos - half])
    return np.asarray(out)


def bend_angles(objects) -> np.ndarray:
    """The angle between each two consecutive links' axes (degrees), as laid or not."""
    axes = np.array([_axis(objects[name]) for name in LINKS])
    return np.degrees(np.arccos(np.clip(np.sum(axes[:-1] * axes[1:], axis=1), -1.0, 1.0)))


class _Pose:
    def __init__(self, pos, quat_xyzw):
        self.pos, self.quat_xyzw = pos, quat_xyzw


def laid_bends() -> np.ndarray:
    """The bends of the filament as laid (its rest shape)."""
    return bend_angles({link.name: _Pose(np.array(link.pos), np.array(link.quat)) for link in FILAMENT})


def in_inlet_frame(p) -> tuple[float, float]:
    """A point's depth past the mouth along the inlet's axis, and its distance from that axis (m)."""
    rel = np.asarray(p, float) - MOUTH_3D
    depth = float(rel @ AXIS_3D)
    return depth, float(np.linalg.norm(rel - depth * AXIS_3D))


def tip_point(objects) -> np.ndarray:
    """The filament's free end."""
    return link_ends(objects)[0, 0]


def tip(objects) -> tuple[float, float]:
    """(depth, off axis) of the filament's free end."""
    return in_inlet_frame(tip_point(objects))


def inserted(objects, depth: float) -> bool:
    """The free end ``depth`` or more past the mouth, within 2.5 mm of the inlet's axis."""
    d, off = tip(objects)
    return bool(d >= depth and off <= ON_AXIS)


def near_mouth(objects) -> bool:
    return bool(np.linalg.norm(tip_point(objects) - MOUTH_3D) <= NEAR_MOUTH)


def spool_point(objects) -> np.ndarray:
    """Where the filament's last link is anchored on the spool, as the spool now stands."""
    s = objects["spool"]
    return s.pos + matrix(s.quat_xyzw) @ ANCHOR_LOCAL


def attached(objects) -> bool:
    return bool(np.linalg.norm(link_ends(objects)[-1, 1] - spool_point(objects)) <= ATTACHED)


def on_axle(objects) -> bool:
    return bool(np.linalg.norm(objects["spool"].pos - np.asarray(SPOOL.pos)) <= ON_AXLE)


def _fell(objects) -> bool:
    return any(objects[n].pos[2] < TOP + OFF_TABLE_Z for n in FREE)


def kinked(objects) -> bool:
    """The filament bent KINK degrees or more between two links, now or earlier in the episode."""
    return bool(max(STATE["peak_bend"], float(bend_angles(objects).max())) >= KINK)


def ruined(objects) -> bool:
    """A state that scores 0 whatever else: not finite, fallen, kinked, torn off the spool, spool off its rod."""
    if not _finite(objects):
        return True
    return _fell(objects) or kinked(objects) or not attached(objects) or not on_axle(objects)


def _points(objects) -> np.ndarray:
    """Each link's centre and the filament's point on the spool."""
    return np.vstack([[objects[n].pos for n in LINKS], [spool_point(objects)]])


def still(objects) -> bool:
    """Nothing moved: every link's centre, and the filament's point on the spool, within 2 mm of where it was a second
    ago. Before a second has been recorded: every link and the spool under 2 cm/s and turning under 0.5 rad/s (a
    link's roll about its own axis aside, and the rattle of the end in the bore shows only as speed)."""
    track = STATE["track"]
    if len(track) > STILL_WINDOW:
        return bool(np.linalg.norm(_points(objects) - track[-1 - STILL_WINDOW], axis=1).max() < STILL_MOVE)
    if not objects["spool"].resting(STILL_SPEED, STILL_SPIN):
        return False
    for name in LINKS:
        s = objects[name]
        w = np.asarray(s.ang_vel, float)
        across = w - (w @ _axis(s)) * _axis(s)
        if np.linalg.norm(s.lin_vel) >= STILL_SPEED or np.linalg.norm(across) >= STILL_SPIN:
            return False
    return True


def _segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((point - a) * ab).sum(axis=1) / (ab * ab).sum(axis=1), 0.0, 1.0)
    return np.linalg.norm(point - (a + t[:, None] * ab), axis=1)


def held_by(objects, grippers: dict) -> bool:
    """Whether some gripper is closed on the filament: its finger pads 1.25 to 4.75 mm apart (the filament's
    thickness, less 0.5 mm or plus 3 mm) and the point between them within 1.5 cm of a link's axis.
    ``grippers`` maps each arm to (that point, the pads' gap in m)."""
    ends = link_ends(objects)
    for point, gap in grippers.values():
        near = _segment_distance(np.asarray(point, float), ends[:, 0], ends[:, 1]).min() <= HOLD_REACH
        if near and FIL_D - 0.0005 <= gap <= FIL_D + HOLD_GAP:
            return True
    return False


def grippers(env) -> dict:
    """Each arm's TCP (the point between its finger pads) and its pads' gap, in the task file's frame."""
    from yam_arena.constants import TCP_OFFSET

    def arr(t) -> np.ndarray:
        t = getattr(t, "torch", t)
        return t.detach().cpu().numpy().astype(float)

    base = env.unwrapped
    origin = arr(base.scene.env_origins)[0]
    out = {}
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        bodies = list(robot.body_names)
        joints = list(robot.joint_names)
        i = bodies.index("gripper")
        pos, quat = arr(robot.data.body_pos_w)[0, i], arr(robot.data.body_quat_w)[0, i]
        q = arr(robot.data.joint_pos)[0]
        tcp = pos - origin + matrix(quat) @ np.asarray(TCP_OFFSET)
        out[arm] = (tcp, -float(q[joints.index("joint7")] + q[joints.index("joint8")]))
    return out


@on_step
def watch(env) -> None:
    """Every control step: whether a gripper holds the filament, its sharpest bend and where it is, into the
    episode's memory."""
    from yam_arena.scene import object_states

    objects = object_states(env, FREE)
    if _finite(objects):
        STATE["held"] = held_by(objects, grippers(env))
        STATE["peak_bend"] = max(STATE["peak_bend"], float(bend_angles(objects).max()))
        STATE["track"] = (STATE["track"] + [_points(objects)])[-(STILL_WINDOW + 1):]


def success(objects) -> bool:
    """The free end 2 cm or more into the inlet, the filament let go, and everything still."""
    if ruined(objects):
        return False
    return bool(inserted(objects, FULL_DEPTH) and not STATE["held"] and still(objects))


def score(objects) -> float:
    """1.0: success. 0.5: the end 5 mm or more into the inlet. 0.25: the end within 1 cm of the mouth's centre. 0
    otherwise, and 0 if the filament kinked, tore off the spool or fell, or the spool came off its rod."""
    if ruined(objects):
        return 0.0
    if success(objects):
        return 1.0
    if inserted(objects, PART_DEPTH):
        return 0.5
    return 0.25 if near_mouth(objects) else 0.0


def distance(objects) -> float:
    """How far the filament's end is from 2 cm inside the inlet (m)."""
    return float(np.linalg.norm(tip_point(objects) - (MOUTH_3D + FULL_DEPTH * AXIS_3D)))


# -- scripted demo ---------------------------------------------------------------------------------------------------
# Each hand pinches the filament from above, straight down, its fingers closing along x (across the filament), the
# right hand 8.75 cm from the end, between the two blocks, and the left hand 17.5 cm further back, beyond them. They
# lift it and, rising in front of the printer, turn their grips forward about the filament's own line (the gripped
# links level, so the turn doesn't swing the end sideways) until the fingers point
# along +x, one pad above the filament and one below. They carry it level over the roof to the inlet's height, its end
# 1 cm short of the mouth, and push it 3.5 cm along -y together (2.5 cm inside). The left hand lets go, backs off
# along -x and out over the printer's front, then the right hand does the same. The rest of the filament trails over
# the roof's left edge to the spool, which turns as it is drawn off. Numbers are TCP positions: 6 mm behind the
# filament along the fingers (the filament 7 mm in from the pad tips).
GRIP_R, GRIP_L = 3.5 * LINK, 10.5 * LINK  # m from the filament's end: each hand on a link's middle
IN_PADS = 0.006  # m: the TCP behind the filament's axis, along the fingers
SAG = {"right": 0.001, "left": 0.002}  # m: the filament's sag at each grip at rest
CLOSING = {"right": 1.0, "left": -1.0}  # which way each hand's fingers close (keeps joint6 inside its limits)
HOVER_H = 0.10  # the hands come down from this high (above the table)
LIFT_H = 0.12  # the filament lifted this high before the grips turn
RISE = (PICK_X + 0.02, 0.40, 60.0)  # x, height and pitch (degrees) on the way up, in front of the printer
OVER = (0.42, pr.INLET_H + 0.015)  # x and height as the hands come over the roof's front
AIM = (-0.0041, 0.0054)  # m: across the line (+x) and up: the hands carry the line this far off the inlet's axis,
# as the end, 8.75 cm ahead of the right hand, comes out of the grip's turn that far forward and low (measured)
START = -0.01  # m: the end this far short of the mouth before the push
PUSH = 0.035  # m along the axis: the end is then 2.5 cm inside
OPEN = -0.012  # finger joint to let go: pads 2.4 cm apart
BACK = 0.045  # m: each hand backs off this far along -x once open, its fingers sliding off the filament
CLEAR_X = 0.32  # and then back out level to here, in front of the printer, before it goes down to wait
HOLD_ON = 0  # 1: the hands never let go (the shortcut the 0.5 rung scores)


def grip(pitch_deg: float, arm: str) -> dict:
    """The grip turned ``pitch_deg`` from straight down toward +x, about the filament's line (y): the fingers point
    along (sin p, 0, -cos p) and close along (cos p, 0, sin p), so the filament (along y) stays between the pads."""
    p, k = math.radians(pitch_deg), CLOSING[arm]
    return {"approach": (math.sin(p), 0.0, -math.cos(p)), "closing": (k * math.cos(p), 0.0, k * math.sin(p))}


def tcp(point, pitch_deg: float) -> tuple[float, float, float]:
    """The TCP holding the filament at ``point`` with this grip."""
    p = math.radians(pitch_deg)
    return tuple(float(v) for v in np.asarray(point) - IN_PADS * np.array([math.sin(p), 0.0, -math.cos(p)]))


def demo():
    arms = {}
    for arm, s in (("right", GRIP_R), ("left", GRIP_L)):
        def w(duration, point, pitch, finger, label):
            return HandWaypoint(duration, tcp(point, pitch), finger, label, **grip(pitch, arm))

        x, y = from_tip(s)
        at_block = (x, y, Z_BLOCK - SAG[arm])
        y_line = on_line(START - s)[1]  # the hand's y with the end 1 cm short of the mouth
        y_push = on_line(START + PUSH - s)[1]
        line_x = MOUTH[0] + AIM[0]
        line_z = INLET_Z + AIM[1]
        arms[arm] = [
            w(1.5, (x, y, TOP + HOVER_H), 0.0, -0.02, "over the filament"),
            w(1.5, at_block, 0.0, -0.02, "down round it"),
            w(0.8, at_block, 0.0, FINGER_CLOSED, "close"),
            w(1.5, (x, y, TOP + LIFT_H), 0.0, FINGER_CLOSED, "lift"),
            w(3.5, (RISE[0], y, TOP + RISE[1]), RISE[2], FINGER_CLOSED, "rise, turning the grip forward"),
            w(2.5, (OVER[0], y_line, TOP + OVER[1]), 90.0, FINGER_CLOSED, "over the roof"),
            w(2.0, (line_x, y_line, line_z), 90.0, FINGER_CLOSED, "line up 1 cm short"),
            w(0.5, (line_x, y_line, line_z), 90.0, FINGER_CLOSED, "settle"),
            w(3.5, (line_x, y_push, line_z), 90.0, FINGER_CLOSED, "push it in"),
            w(0.5, (line_x, y_push, line_z), 90.0, FINGER_CLOSED, "hold"),
        ]
    if HOLD_ON:
        return arms
    for arm, wait, park in (("left", 0.0, (0.28, 0.24, TOP + 0.36)), ("right", 2.0, (0.28, -0.24, TOP + 0.36))):
        last = arms[arm][-1]
        x, y, z = last.pos
        g = grip(90.0, arm)
        steps = [HandWaypoint(wait, last.pos, FINGER_CLOSED, "hold while the left lets go", **g)] if wait else []
        steps += [
            HandWaypoint(1.0, last.pos, OPEN, "let go", **g),
            HandWaypoint(1.0, (x - BACK, y, z), OPEN, "back off along the fingers", **g),
            HandWaypoint(1.2, (CLEAR_X, y, z), OPEN, "back out over the front", **g),
            HandWaypoint(1.5, park, OPEN, "away", **grip(45.0, arm)),
        ]
        arms[arm] += steps
    return arms
