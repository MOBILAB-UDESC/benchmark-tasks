"""On the station table, between the two arms: two helping-hands stands, one on each side, each a heavy base with a
three-link arm on friction joints. Each arm ends in a black alligator clip holding a short wire stub, a red one on the
right stand and a blue one on the left. A yellow cross is taped on the table at (0.31, 0). Both stands start with their
wire ends 17 cm apart and away from the mark, the arms folded in a Z. The right stand is at (0.42, -0.27), the left
at (0.42, 0.27), so each arm works in its own half of the table.

The task: move the two stands' arms until the red and the blue wire ends cross over the yellow mark, then let go and
leave them there.

Each arm's three joints are dry-friction hinges (``bonds.Hinge``): a joint stays where it is under any torque below
0.10 N m, turns against a constant 0.10 N m once pushed past it, and has no spring that brings it back. Moving the
clip drags the two joints behind it. The wire is held in the clip by a bond that breaks at 2 N: pulled harder, the
wire comes out and the trial scores 0. The left stand stands 5 mm higher than the right one, exactly one wire
diameter, so the two wires lie one on the other where they overlap, touching, as in a lap joint ready to solder.

Run it:  inspect-robots run --embodiment yam_arena -E task=helping_hands_cross_wires.py --scorer task_score ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import contact  # noqa: E402  (hooks Isaac Lab to read the contact between the two wires)
from bonds import Bonded, Hinge, on_reset, on_step  # noqa: E402
from looks import dress  # noqa: E402
from yam_arena.constants import PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import FINGER_CLOSED, FINGER_OPEN, Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 3.0  # the crossing must stay, still and let go of, for this long

# -- the stands ------------------------------------------------------------------------------------------------------
MARK = (0.31, 0.0)  # the yellow cross
LINK = (0.11, 0.11, 0.12)  # m: hinge to hinge, hinge to hinge, and the third link's hinge to the clip's end face
REACH = (LINK[0], LINK[1], LINK[2])
EXT = 0.007  # a link's box reaches this far behind its first hinge and past its second one
LINK_W, LINK_H = 0.014, 0.024
LINK_MASS = 0.04
WIRE_SIZE = (0.05, 0.005, 0.005)  # a 5 mm wire stub 5 cm long sticking out of the clip
WIRE_MASS = 0.003
WIRE_HOLD = 2.0  # N: the pull that takes a wire out of its clip
CLIP_END = LINK[2]  # the wire starts at the clip's end face
TIP = LINK[2] + WIRE_SIZE[0]  # from the third hinge to the wire's free end
WIRE_PITCH = WIRE_SIZE[2]  # the left stand's arms stand one wire diameter higher: the two wires touch where they cross
LINK_Z = {"right": TOP + 0.045, "left": TOP + 0.045 + WIRE_PITCH}  # the links' centres
PLINTH_SIZE = (0.08, 0.08)
PLINTH_H = {"right": 0.030, "left": 0.030 + WIRE_PITCH}  # the base tops sit 3 mm under the first link
BASE_XY = {"right": (0.42, -0.27), "left": (0.42, 0.27)}
HOLD = (0.10, 0.10, 0.10)  # N m: the friction of each hinge, base to clip
GAIN = 20.0  # N m s/rad: the friction drive's gain (the hinge creeps at torque / gain below its hold)
JOINT_RANGE = math.radians(150)  # each hinge turns this far either way from straight

# Each stand's hinge angles, as (the first link's angle from +x, the second link's turn from the first, the third's turn
# from the second), in radians. The left stand mirrors the right one across y = 0.
_START_RIGHT = tuple(math.radians(a) for a in (60.0, 125.0, -40.0))
START = {"right": _START_RIGHT, "left": tuple(-a for a in _START_RIGHT)}
# Where each wire's free end goes: over the mark along x, and 6 mm past it along y, so the two overlap 12 mm.
HALF_OVERLAP = 0.006
FINAL_TIP = {"right": (MARK[0], MARK[1] + HALF_OVERLAP), "left": (MARK[0], MARK[1] - HALF_OVERLAP)}
FINAL_DIR = {"right": math.pi / 2, "left": -math.pi / 2}  # the third link's angle: the wires point at each other
ELBOW = {"right": -1, "left": +1}  # which way the first two links fold, the start's side


def _rot(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)])


def hinges(side: str, angles) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
    """The three hinges' places (x, y) and the three links' absolute angles, for a stand's hinge angles."""
    p0 = np.asarray(BASE_XY[side], float)
    a1 = angles[0]
    a2 = a1 + angles[1]
    a3 = a2 + angles[2]
    p1 = p0 + REACH[0] * _rot(a1)
    p2 = p1 + REACH[1] * _rot(a2)
    return p0, p1, p2, (a1, a2, a3)


def wire_end(side: str, angles) -> np.ndarray:
    """The wire's free end (x, y) for a stand's hinge angles."""
    _, _, p2, (_, _, a3) = hinges(side, angles)
    return p2 + TIP * _rot(a3)


def solve_arm(side: str, tip, a3: float) -> tuple[float, float, float]:
    """The hinge angles that put the wire's free end at ``tip`` with the third link at angle ``a3`` (the fold on the
    start's side), by two-link inverse kinematics of the first two links to the third hinge."""
    base = np.asarray(BASE_XY[side], float)
    p2 = np.asarray(tip, float) - TIP * _rot(a3)
    d = p2 - base
    r = float(np.hypot(*d))
    if not (abs(REACH[0] - REACH[1]) < r < REACH[0] + REACH[1]):
        raise ValueError(f"{side}: the third hinge at {p2} is out of the first two links' reach")
    a = math.acos((REACH[0] ** 2 + r * r - REACH[1] ** 2) / (2 * REACH[0] * r))
    a1 = math.atan2(d[1], d[0]) + ELBOW[side] * a
    p1 = base + REACH[0] * _rot(a1)
    a2 = math.atan2(*(p2 - p1)[::-1])
    return _wrap(a1), _wrap(a2 - a1), _wrap(a3 - a2)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def parts(side: str, angles) -> dict[str, tuple[tuple[float, float, float], float, tuple[float, float, float]]]:
    """Every moving part of a stand for its hinge angles: name -> (centre, yaw, size)."""
    p0, p1, p2, (a1, a2, a3) = hinges(side, angles)
    z = LINK_Z[side]
    spans = (
        (p0, a1, -EXT, REACH[0] + EXT),
        (p1, a2, -EXT, REACH[1] + EXT),
        (p2, a3, -EXT, CLIP_END),
    )
    out = {}
    for i, (origin, angle, lo, hi) in enumerate(spans, start=1):
        c = origin + (lo + hi) / 2 * _rot(angle)
        out[f"{side}_link{i}"] = ((float(c[0]), float(c[1]), z), angle, (hi - lo, LINK_W, LINK_H))
    c = p2 + (CLIP_END + WIRE_SIZE[0] / 2) * _rot(a3)
    out[f"{side}_wire"] = ((float(c[0]), float(c[1]), z), a3, WIRE_SIZE)
    return out


def _stand(side: str, color_clip, color_wire):
    """The boxes of one stand: its fixed base, three hinged links and the bonded wire."""
    angles = START[side]
    p0, p1, p2, _ = hinges(side, angles)
    layout = parts(side, angles)
    z = LINK_Z[side]
    base = Box(
        f"{side}_base",
        pos=(*BASE_XY[side], TOP + PLINTH_H[side] / 2),
        size=(*PLINTH_SIZE, PLINTH_H[side]),
        color=(0.10, 0.10, 0.11),
    )
    out = [base]
    previous = base.name
    for i, (hinge, hold) in enumerate(zip((p0, p1, p2), HOLD), start=1):
        pos, yaw, size = layout[f"{side}_link{i}"]
        rel = angles[i - 1]
        out.append(
            Hinge(
                f"{side}_link{i}",
                pos=pos,
                size=size,
                yaw=yaw,
                mass=LINK_MASS,
                color=color_clip if i == 3 else (0.30, 0.31, 0.33),
                to=previous,
                anchor=(float(hinge[0]), float(hinge[1]), z),
                lower=-JOINT_RANGE - rel,
                upper=JOINT_RANGE - rel,
                hold=hold,
                gain=GAIN,
            )
        )
        previous = f"{side}_link{i}"
    pos, yaw, size = layout[f"{side}_wire"]
    face = p2 + CLIP_END * _rot(yaw)
    out.append(
        Bonded(
            f"{side}_wire",
            pos=pos,
            size=size,
            yaw=yaw,
            mass=WIRE_MASS,
            color=color_wire,
            to=f"{side}_link3",
            hold=WIRE_HOLD,
            anchor=(float(face[0]), float(face[1]), z),
        )
    )
    return out


RED = (0.85, 0.06, 0.05)
BLUE = (0.05, 0.20, 0.90)
CLIP_BLACK = (0.05, 0.05, 0.055)
STAND_R = _stand("right", CLIP_BLACK, RED)
STAND_L = _stand("left", CLIP_BLACK, BLUE)
MARKS = [
    Box("mark_x", pos=(*MARK, TOP + 0.0005), size=(0.036, 0.007, 0.001), color=(0.95, 0.78, 0.05)),
    Box("mark_y", pos=(*MARK, TOP + 0.0005), size=(0.007, 0.036, 0.001), color=(0.95, 0.78, 0.05)),
]
OBJECTS = [*STAND_R, *STAND_L, *MARKS]

LINKS = [f"{side}_link{i}" for side in ("right", "left") for i in (1, 2, 3)]
WIRES = ["right_wire", "left_wire"]
MOVING = [*LINKS, *WIRES]

# What the boxes look like: meshes modelled in Blender (looks/make_looks.py), the boxes' colliders unchanged.
LOOKS = {name: f"{name}.usdc" for name in (*MOVING, "right_base", "left_base")}
LOOKS.update({"mark_x": "mark.usdc"})
dress(LOOKS, hide=["mark_y"])

# -- scoring ---------------------------------------------------------------------------------------------------------
OVERLAP_MIN = 0.005  # m: the free ends pass each other by at least this
SIDE_MAX = 0.002  # m: and the wires' centre lines are at most this far apart sideways
MARK_MAX = 0.005  # m: and the midpoint of the two free ends is at most this far from the mark's centre
TOUCH_GAP = contact.TOUCH_SEP  # m: and the wires touch: the simulator reports a contact point between them this close
NEAR_OVERLAP, NEAR_SIDE, NEAR_MARK = -0.010, 0.010, 0.010  # the 0.5 band: ends at most 10 mm short, 10 mm sideways
LOOSE_MARK = 0.030  # m: the 0.25 band: each free end is at most this far from the mark's centre
CLEAR_DIST = 0.03  # m: a gripper finger at least this far from every part of the stands has let go
OUT_OF_CLIP = 0.005  # m: a wire this far from where its clip holds it has come out
OFF_TABLE_Z = -0.02  # m above the table: a part whose centre is lower has fallen off the table
_WIRE_AT = CLIP_END + WIRE_SIZE[0] / 2  # the wire's centre, from the third hinge
_LINK3_AT = (CLIP_END - EXT) / 2  # the third link's centre, from the third hinge

# What the episode remembers: a wire came out of its clip at some step, whether a finger is near any part now, whether
# the simulator reports the two wires touching now, and for how many seconds the crossing has held, still, touching and
# let go of.
STATE = {"pulled": False, "clear": False, "touch": False, "held": 0.0}
CONTACT_LOG = True  # print a line to the run's log whenever the wires start or stop touching
_LOGGED = {"touch": False}
CONTROL_DT = 0.05  # s per control step (20 Hz), overwritten from the environment on the first step


@on_reset
def reset_memory(env=None) -> None:
    STATE.update(pulled=False, clear=False, touch=False, held=0.0)
    _LOGGED["touch"] = False


def _finite(objects) -> bool:
    return all(np.all(np.isfinite(objects[n].pos)) and np.all(np.isfinite(objects[n].quat_xyzw)) for n in MOVING)


def _axis(state) -> np.ndarray:
    """The unit vector (x, y) along a part's own long axis (its box's x axis)."""
    x, y, z, w = (float(v) for v in state.quat_xyzw)
    a = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w)])
    return a / np.linalg.norm(a)


def wire_tip(objects, side: str) -> tuple[np.ndarray, np.ndarray]:
    """A wire's free end (x, y) and the unit vector along the wire pointing out of its clip."""
    wire, link = objects[f"{side}_wire"], objects[f"{side}_link3"]
    u = _axis(wire)
    if float(u @ (wire.pos[:2] - link.pos[:2])) < 0:
        u = -u
    return wire.pos[:2] + u * WIRE_SIZE[0] / 2, u


def crossing(objects) -> tuple[float, float, float]:
    """How the two wire ends lie, as a ruler on a top photo would read it (m): the overlap (how far the free ends have
    passed each other along the wires, negative before they meet), the sideways gap (the larger distance of one free
    end from the other wire's centre line) and the distance from the midpoint between the free ends to the mark."""
    (pr, ur), (pl, ul) = wire_tip(objects, "right"), wire_tip(objects, "left")
    along = ur - ul
    along = along / np.linalg.norm(along)
    overlap = float((pr - pl) @ along)
    side = max(abs(float((pr - pl) @ np.array([-ul[1], ul[0]]))), abs(float((pl - pr) @ np.array([-ur[1], ur[0]]))))
    mid = float(np.linalg.norm((pr + pl) / 2 - np.asarray(MARK)))
    return overlap, side, mid


def wire_out(objects, side: str) -> bool:
    """A wire is out of its clip: its centre is 5 mm or more from where the clip holds it."""
    wire, link = objects[f"{side}_wire"], objects[f"{side}_link3"]
    u = _axis(link)
    held = link.pos[:2] + (_WIRE_AT - _LINK3_AT) * u
    return bool(float(np.linalg.norm(wire.pos[:2] - held)) >= OUT_OF_CLIP or abs(wire.pos[2] - link.pos[2]) >= OUT_OF_CLIP)


def still(objects) -> bool:
    """Every link and wire is still: under 1 cm/s and 0.5 rad/s."""
    return all(objects[n].resting() for n in MOVING)


def _remember(objects) -> None:
    if any(wire_out(objects, side) for side in ("right", "left")):
        STATE["pulled"] = True


def _violation(objects) -> bool:
    if not _finite(objects):
        return True
    return bool(any(objects[n].pos[2] < TOP + OFF_TABLE_Z for n in MOVING))


def wire_gap(objects) -> float:
    """The gap (m) between the two wires' facing faces, from their heights: the centres' height difference less one
    wire diameter (negative when they press into each other)."""
    return float(abs(objects["left_wire"].pos[2] - objects["right_wire"].pos[2]) - WIRE_SIZE[2])


def wires_touch(objects) -> bool:
    """The wires touch: the simulator reports a contact point between them at 0.5 mm or less (read every physics
    substep by ``contact.py``), and their faces are at most 0.5 mm apart by their heights."""
    return bool(STATE["touch"] and wire_gap(objects) <= TOUCH_GAP)


def crossed(objects) -> bool:
    """The wires cross over the mark: overlap 5 mm or more, sideways gap 2 mm or less, midpoint within 5 mm of the mark."""
    overlap, side, mid = crossing(objects)
    return bool(overlap >= OVERLAP_MIN and side <= SIDE_MAX and mid <= MARK_MAX)


def settled(objects) -> bool:
    """The crossing is made with the wires touching, everything is still and no finger is near the stands."""
    return bool(crossed(objects) and wires_touch(objects) and still(objects) and STATE["clear"])


def success(objects) -> bool:
    """The crossing is made with the wires touching, still, let go of, and has held for 3 s, with no wire pulled out."""
    if _violation(objects):
        return False
    _remember(objects)
    return bool(not STATE["pulled"] and settled(objects) and STATE["held"] >= SUCCESS_HOLD_S)


def score(objects) -> float:
    """1.0: success (the crossing made with the wires touching, still, let go of and held for 3 s). 0.5: the free
    ends are within 10 mm of meeting (at most 10 mm short of each other along the wires, at most 10 mm apart
    sideways) and their midpoint is within 10 mm of the mark, whether or not the wires touch or the crossing is
    still, let go of or held yet. 0.25: each free end is
    within 3 cm of the mark. 0 otherwise, and 0 for the rest of the trial once a wire has come out of its clip, or if
    a part has fallen off the table."""
    if _violation(objects):
        return 0.0
    _remember(objects)
    if STATE["pulled"]:
        return 0.0
    if settled(objects) and STATE["held"] >= SUCCESS_HOLD_S:
        return 1.0
    overlap, side, mid = crossing(objects)
    if overlap >= NEAR_OVERLAP and side <= NEAR_SIDE and mid <= NEAR_MARK:
        return 0.5
    near = [np.linalg.norm(wire_tip(objects, s)[0] - np.asarray(MARK)) for s in ("right", "left")]
    return 0.25 if max(near) <= LOOSE_MARK else 0.0


def distance(objects) -> float:
    """Metres between the two wire ends' midpoint and the mark, plus the sideways gap."""
    if not _finite(objects):
        return float("inf")
    _, side, mid = crossing(objects)
    return float(mid + side)


# -- fingers near the stands: has the robot let go? -------------------------------------------------------------------


def box_gap(points: np.ndarray, pos, quat_xyzw, size) -> np.ndarray:
    """Each point's distance (m) to the oriented box with centre ``pos``, orientation ``quat_xyzw`` and edge lengths
    ``size``."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    local = (points - np.asarray(pos, float)) @ rot  # the points in the box's frame
    out = np.maximum(np.abs(local) - np.asarray(size, float) / 2, 0.0)
    return np.linalg.norm(out, axis=1)


SIZES = {o.name: o.size for o in OBJECTS if o.name in MOVING}


def hands_clear(objects, points: np.ndarray) -> bool:
    """No finger point is within 3 cm of any link or wire."""
    for name in MOVING:
        s = objects[name]
        if float(box_gap(points, s.pos, s.quat_xyzw, SIZES[name]).min()) < CLEAR_DIST:
            return False
    return True


@on_step
def watch_hands(env) -> None:
    """Every step: whether the fingers have let go of the stands, whether the wires touch, and how long the crossing
    has held, still."""
    from fingers import finger_points
    from yam_arena.scene import object_states

    global CONTROL_DT
    CONTROL_DT = float(getattr(env.unwrapped, "step_dt", CONTROL_DT))
    objects = object_states(env, MOVING)
    if not _finite(objects):
        return
    _remember(objects)
    STATE["clear"] = hands_clear(objects, finger_points(env))
    STATE["touch"] = contact.touching()
    if CONTACT_LOG and STATE["touch"] != _LOGGED["touch"]:
        _LOGGED["touch"] = STATE["touch"]
        print(f"[wires] touching={STATE['touch']} gap={wire_gap(objects) * 1000:.2f} mm {contact.status()}", flush=True)
    STATE["held"] = STATE["held"] + CONTROL_DT if (not STATE["pulled"] and settled(objects)) else 0.0


# -- scripted demo ---------------------------------------------------------------------------------------------------
# Each arm closes on the third link, a straight-down grip across it, 8.5 cm behind the clip's face (where the stands'
# first link is not in the way) with the fingers open to 44 mm, drags it to the crossing in one move that turns the grip
# with the link, holds a second so the joints settle, lets go and stands clear. Numbers are TCP positions. The grip
# turns with the link: a yaw equal to the link's angle closes the fingers across it.
GRIP_U = 0.085  # m: from the third hinge, along the link
GRIP_DEPTH = 0.018  # m: the pad tips below the link's top edge
APPROACH = -0.022  # finger target while coming down over the link: 44 mm between the pads, 15 mm each side of it
HOVER = TOP + 0.13
# How far each arm's last waypoint is sent past the crossing, in x and y (m), to make up what the friction holds back.
AIM = {"right": (0.0034, 0.0), "left": (-0.0033, 0.0)}
DRAG_S = 4.0
CLEAR_XY = (0.24, 0.20)  # where each hand waits, above its own side of the table (y is mirrored)
# Cheat variants for the README's ablation: "wire" grips each wire stub instead of the clip link, "keep" never lets go.
CHEAT = ""


def grip_point(side: str, angles, cheat: str = "") -> tuple[float, float, float]:
    """The TCP, for a grip on the third link at ``GRIP_U`` from its hinge (or, in the "wire" cheat, on the wire's
    middle, the pad tips 4 mm below its top), when the stand has these hinge angles."""
    _, _, p2, (_, _, a3) = hinges(side, angles)
    if cheat == "wire":
        g = p2 + _WIRE_AT * _rot(a3)
        return float(g[0]), float(g[1]), LINK_Z[side] + WIRE_SIZE[2] / 2 - 0.004 + PAD_TIP_PAST_TCP
    g = p2 + GRIP_U * _rot(a3)
    return float(g[0]), float(g[1]), LINK_Z[side] + LINK_H / 2 - GRIP_DEPTH + PAD_TIP_PAST_TCP


FINAL = {side: solve_arm(side, FINAL_TIP[side], FINAL_DIR[side]) for side in ("right", "left")}


def demo():
    w = Waypoint
    plans = {}
    for side in ("right", "left"):
        sign = -1.0 if side == "right" else 1.0
        start_angle = hinges(side, START[side])[3][2]
        x0, y0, gz = grip_point(side, START[side], CHEAT)
        x1, y1, _ = grip_point(side, FINAL[side], CHEAT)
        x1, y1 = x1 + AIM[side][0], y1 + AIM[side][1]
        yaw0, yaw1 = start_angle, FINAL_DIR[side]
        steps = [
            w(1.5, (x0, y0, HOVER), APPROACH, "over the clip link", yaw0),
            w(1.5, (x0, y0, gz), APPROACH, "down across it", yaw0),
            w(1.6, (x0, y0, gz), FINGER_CLOSED, "close", yaw0),
            w(DRAG_S, (x1, y1, gz), FINGER_CLOSED, "drag the wire end to the mark", yaw1),
            w(1.0, (x1, y1, gz), FINGER_CLOSED, "let the joints settle", yaw1),
            w(0.8, (x1, y1, gz), APPROACH, "let go", yaw1),
            w(1.2, (x1, y1, HOVER), FINGER_OPEN, "lift off", yaw1),
            w(1.5, (CLEAR_XY[0], CLEAR_XY[1] * sign, HOVER), FINGER_OPEN, "stand clear", yaw1),
            w(4.5, (CLEAR_XY[0], CLEAR_XY[1] * sign, HOVER), FINGER_OPEN, "wait", yaw1),
        ]
        plans[side] = steps[:5] + [w(8.0, (x1, y1, gz), FINGER_CLOSED, "keep holding", yaw1)] if CHEAT == "keep" else steps
    return plans
