"""On the station table, centred between the two arms: an enclosed desktop 3D printer (36 x 35 x 40 cm, 2.5 kg) standing
on four smooth feet, its clear front door shut. The door (33 x 29 cm, tinted acrylic, 0.5 kg) hangs on a vertical
hinge at its left edge and has a bar handle near its right edge. A magnetic catch holds it shut. Inside, the print head
is parked at the front of its gantry and a finished orange print (a 5 x 5 x 9 cm vase, 30 g) stands on the gold
build plate, near the door.

The task: open the door all the way, until it stays open on its own, without moving the printer or touching the print
head or the print.

The door's hinge (``bonds.Hinged``): a magnetic catch holds it shut until 10 N is pulled at the handle (a joint that
breaks at 3.1 N m about the hinge), then it swings freely with a little friction. From 80 degrees on, a detent pulls
it into its open stop at 95 degrees and holds it there (the hinge stops at 110). The printer is light and its feet
slide (friction 0.28): a pull of about 7 N moves it, so pulling the handle without the other hand steadying the
printer drags it across the table instead of opening the catch. The print head hangs on a soft sprung slide
(``bonds.Sliding``) and the vase stands loose on the plate, so a knock shows.

Run it:  inspect-robots run --embodiment yam_arena -E task=open_printer_door.py --scorer task_score ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import printer as pr  # noqa: E402
from bonds import Hinged, Sliding, hinge_angle, on_reset, on_step  # noqa: E402
from colliders import shape, shapes  # noqa: E402,F401
from looks import dress  # noqa: E402
from wrist import TiltWaypoint  # noqa: E402
from yam_arena.constants import PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0


def _at(x, y, h):
    """A station-frame position from a height above the table."""
    return (x, y, TOP + h)


# -- the printer: free, light, on feet that slide ------------------------------------------------------------------------
PRINTER_MASS = 2.5  # kg
FEET_FRICTION = 0.28  # its four plastic feet on the table: 0.28 x 2.5 kg x 9.81 = 6.9 N to slide it
PRINTER = Box("printer", pos=_at(*pr.CENTRE), size=pr.SIZE, mass=PRINTER_MASS, color=(0.78, 0.78, 0.8))
shape("printer", pr.printer_parts(bed_top=pr.PLATE_TOP, feet_friction=FEET_FRICTION))

# -- the door: on a vertical hinge at its left edge, its angle negative as it opens (clockwise seen from above) ------
HANDLE_R = math.hypot(pr.HANDLE[0] - pr.HINGE[0], pr.HANDLE[1] - pr.HINGE[1])  # 0.309 m from the hinge axis
CATCH_PULL = 10.0  # N at the handle, square to the door, that opens the catch
CATCH_TORQUE = CATCH_PULL * HANDLE_R  # N m about the hinge
DETENT_FROM, DETENT_AT, STOP = 80.0, 95.0, 110.0  # degrees open: the detent's reach, its rest, the hinge's stop
DETENT_RATE = 1.5  # N m/rad: at 80 degrees it pulls 0.4 N m (1.3 N at the handle) toward 95
DETENT_DAMPING = 0.1  # N m s/rad
HINGE_FRICTION = 0.02  # N m s/rad
DOOR_FRICTION = 0.1  # the door's smooth acrylic against a finger sliding on it (its handle is grippy)
DOOR = Hinged(
    "door", pos=_at(*pr.DOOR_CENTRE), size=pr.DOOR_SIZE, mass=0.5, color=(0.45, 0.52, 0.58), to="printer", axis="Z",
    hinge=(pr.HINGE[0] - pr.DOOR_CENTRE[0], pr.HINGE[1] - pr.DOOR_CENTRE[1], 0.0), lower=-STOP, upper=0.0,
    catch=CATCH_TORQUE, zones=((-STOP, -DETENT_FROM, -DETENT_AT, DETENT_RATE, DETENT_DAMPING),),
    damping=HINGE_FRICTION,
)
shape("door", pr.door_parts(DOOR_FRICTION))

# -- the print head on a soft slide (along x), and the loose vase on the plate --------------------------------------
HEAD = Sliding("print_head", pos=_at(*pr.HEAD_CENTRE), size=pr.HEAD_SIZE, mass=0.15, color=(0.78, 0.79, 0.8),
               to="printer", axis="X", lower=-0.01, upper=0.01, rate=300.0, damping=5.0)
VASE_SIZE = (0.05, 0.05, 0.09)
VASE_XY = (0.43 + pr.SHIFT, -0.05)
VASE = Box("vase", pos=_at(*VASE_XY, pr.PLATE_TOP + VASE_SIZE[2] / 2), size=VASE_SIZE, mass=0.03,
           color=(0.95, 0.42, 0.08))

OBJECTS = [PRINTER, DOOR, HEAD, VASE]

# What the boxes look like: meshes modelled in Blender (looks/make_looks.py), each inside its box. The build plate is
# drawn on the printer (it is part of the printer's body here: the bed's collider top is the plate's top).
_PLATE_IN_PRINTER = (pr.PLATE_XY[0] - pr.CENTRE[0], pr.PLATE_XY[1] - pr.CENTRE[1],
                     pr.BED_TOP + pr.PLATE_SIZE[2] / 2 - pr.CENTRE[2])
LOOKS = {"printer": ["printer.usdc", ("plate.usdc", _PLATE_IN_PRINTER)], "door": "door.usdc",
         "print_head": "head.usdc", "vase": "tall_print.usdc"}
dress(LOOKS)

# -- scoring ---------------------------------------------------------------------------------------------------------
OPEN_FULL = 90.0  # degrees: fully open
OPEN_HALF = 45.0
OPEN_CRACKED = 10.0  # past the catch
MOVED = 0.01  # m: a corner of the printer's base this far from where it started: the printer was moved
HEAD_STRUCK = 0.003  # m: the head this far off its parked place on its slide
VASE_STRUCK = 0.005  # m: the vase this far from its place on the plate
VASE_TIPPED = 10.0  # degrees

# What the episode remembers: the printer was moved, and something struck the head or the vase. Either scores 0 for
# the rest of the trial.
STATE = {"moved": False, "struck": False}


@on_reset
def reset_memory(env=None) -> None:
    STATE.update(moved=False, struck=False)


def _rot(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


NAMES = ("printer", "door", "print_head", "vase")
_CORNERS = [np.array([sx * pr.SIZE[0] / 2, sy * pr.SIZE[1] / 2, -pr.SIZE[2] / 2]) for sx in (-1, 1) for sy in (-1, 1)]


def _finite(objects) -> bool:
    return all(np.all(np.isfinite(objects[n].pos)) and np.all(np.isfinite(objects[n].quat_xyzw)) for n in NAMES)


def door_open(objects) -> float:
    """How far the door is open (degrees), relative to the printer."""
    return -hinge_angle(DOOR, objects["door"].quat_xyzw, objects["printer"].quat_xyzw)


def printer_shift(objects) -> float:
    """How far the corner of the printer's base that moved most is from where it started (m)."""
    p = objects["printer"]
    r = _rot(p.quat_xyzw)
    start = np.asarray(PRINTER.pos)
    return float(max(np.linalg.norm(p.pos + r @ c - (start + c)) for c in _CORNERS))


def _in_printer(objects, name) -> np.ndarray:
    p = objects["printer"]
    return _rot(p.quat_xyzw).T @ (objects[name].pos - p.pos)


def head_shift(objects) -> float:
    """How far the print head is from its parked place, in the printer's frame (m)."""
    return float(np.linalg.norm(_in_printer(objects, "print_head") - (np.asarray(HEAD.pos) - PRINTER.pos)))


def vase_shift(objects) -> tuple[float, float]:
    """How far the vase is from its place on the plate (m), and how far it leans from the printer's up (degrees)."""
    d = float(np.linalg.norm(_in_printer(objects, "vase") - (np.asarray(VASE.pos) - PRINTER.pos)))
    up_v = _rot(objects["vase"].quat_xyzw)[:, 2]
    up_p = _rot(objects["printer"].quat_xyzw)[:, 2]
    return d, math.degrees(math.acos(max(-1.0, min(1.0, float(up_v @ up_p)))))


def struck(objects) -> bool:
    d, tilt = vase_shift(objects)
    return bool(head_shift(objects) >= HEAD_STRUCK or d >= VASE_STRUCK or tilt >= VASE_TIPPED)


def _remember(objects) -> None:
    if not _finite(objects):
        return
    if printer_shift(objects) >= MOVED:
        STATE["moved"] = True
    if struck(objects):
        STATE["struck"] = True


@on_step
def watch(env) -> None:
    """Every control step: remember a moved printer or a struck head or vase (between the scored steps too)."""
    from yam_arena.scene import object_states

    _remember(object_states(env, list(NAMES)))


def _zero(objects) -> bool:
    if not _finite(objects):
        return True
    _remember(objects)
    return STATE["moved"] or STATE["struck"]


def success(objects) -> bool:
    """The door open 90 degrees or more and still, the printer never moved 1 cm, the head and the vase never
    struck."""
    if _zero(objects):
        return False
    return bool(door_open(objects) >= OPEN_FULL and objects["door"].resting(max_speed=0.02, max_spin=0.1))


def score(objects) -> float:
    """1.0: success. 0.5: the door open 45 degrees or more. 0.25: open 10 degrees or more (past the catch). 0
    otherwise, and 0 for the rest of the trial once the printer has moved 1 cm or more, or the head or the vase was
    struck."""
    if _zero(objects):
        return 0.0
    if success(objects):
        return 1.0
    angle = door_open(objects)
    if angle >= OPEN_HALF:
        return 0.5
    return 0.25 if angle >= OPEN_CRACKED else 0.0


# -- scripted demo ---------------------------------------------------------------------------------------------------
# Left arm: presses its closed fingertips down on the ledge in front of the door, left of the middle, its grip tilted
# 30 degrees forward, so the printer cannot slide, and lets go once the catch is open, then waits out of the door's
# way. Right arm: grips the handle from the front and pulls the door open along its arc past the catch to 6 degrees,
# lets go, comes round the right of the printer, puts its closed fingertips behind the door's free edge and, the side
# of its fingers on the door's inside face, sweeps it open to about 85 degrees, where the detent takes it to 95. Only
# the catch needs a pull, and the left hand steadies the printer through it: the free door is swept, never dragged (a held handle fights the hand's path and slides the light printer). Numbers are TCP
# positions.
HANDLE_GRIP = -0.002  # the fingers on the 12 mm bar, a firm squeeze
GRIP_H = 0.23  # the grip's height on the handle bar (17 to 29 cm), midway between its standoffs
PUSH_H = 0.20
STEADY_PITCH = math.radians(30)
STEADY_XY = (0.345 + pr.SHIFT, 0.04)  # where the fingertips press the ledge, near the pull's line (the door's edge
# sweeps over it at 10 degrees)
STEADY_DEPTH = 0.03
_tip_off = PAD_TIP_PAST_TCP * np.array([math.sin(STEADY_PITCH), 0.0, -math.cos(STEADY_PITCH)])
STEADY = (STEADY_XY[0] - _tip_off[0], STEADY_XY[1], TOP + pr.BASE_TOP - STEADY_DEPTH - _tip_off[2])
PARK = (0.22, 0.32, TOP + 0.18)  # the left hand out of the door's sweep (the open door stands at y 0.17)
CATCH_OPEN = 6.0  # degrees: the right hand pulls the handle this far, past the catch, then lets go
HOOK = (CATCH_OPEN, 0.32)  # its closed fingertips then go behind the door's free edge: (degrees, radius)
HOOK_YAW = 65.0  # degrees: the hand turned only this far toward the door at first, so its palm stays clear of the edge
PUSH = tuple((deg, 0.30) for deg in (10.0, 14.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 78.0, 86.0))  # and sweep it open
# from the inside at a constant radius, so the fingers do not slide along the door: (degrees, radius from the hinge)
PUSH_OFFSET = 0.02  # m: the TCP this far inside the door's inner face (the fingers' half thickness)


def handle_at(deg: float) -> tuple[float, float]:
    """The handle bar's axis (x, y) with the door open ``deg`` degrees."""
    a = math.radians(deg)
    rx, ry = pr.HANDLE[0] - pr.HINGE[0], pr.HANDLE[1] - pr.HINGE[1]
    c, s = math.cos(-a), math.sin(-a)
    return (pr.HINGE[0] + rx * c - ry * s, pr.HINGE[1] + rx * s + ry * c)


def inside_face(deg: float, r: float, offset: float = PUSH_OFFSET) -> tuple[float, float]:
    """A point ``offset`` inside the door's inner face, ``r`` from the hinge axis, the door open ``deg`` degrees."""
    a = math.radians(deg)
    return (pr.HINGE[0] - r * math.sin(a) + offset * math.cos(a), pr.HINGE[1] - r * math.cos(a) - offset * math.sin(a))


def _at_label(steps, label: str) -> float:
    """When the waypoint ``label`` ends (s from the start)."""
    t = 0.0
    for x in steps:
        t += x.duration
        if x.label == label:
            return t
    raise KeyError(label)


def demo():
    def w(duration, pos, finger, label, yaw=0.0, pitch=0.0):
        return TiltWaypoint(duration, tuple(float(v) for v in pos), finger, label, math.radians(yaw), pitch)

    hx, hy = handle_at(0.0)
    p90 = math.pi / 2
    right = [
        w(1.5, (hx - 0.07, hy, TOP + GRIP_H), FINGER_OPEN, "in front of the handle", 0.0, p90),
        w(1.0, (hx, hy, TOP + GRIP_H), FINGER_OPEN, "fingers round the handle", 0.0, p90),
        w(0.6, (hx, hy, TOP + GRIP_H), HANDLE_GRIP, "grip it", 0.0, p90),
    ]
    for deg in (1.5, 3.0, 4.5, CATCH_OPEN):
        right.append(w(0.5, (*handle_at(deg), TOP + GRIP_H), HANDLE_GRIP, f"pull to {deg:.1f} deg", -deg, p90))
    a = math.radians(CATCH_OPEN)
    out_n = (-math.cos(a), math.sin(a))  # the door's outward normal
    hx, hy = handle_at(CATCH_OPEN)
    right.append(w(1.6, (hx, hy, TOP + GRIP_H), HANDLE_GRIP, "wait for the left hand", -CATCH_OPEN, p90))
    hook_deg, hook_r = HOOK
    beside = inside_face(hook_deg, hook_r, PUSH_OFFSET - 0.002)
    right += [
        w(0.6, (hx, hy, TOP + GRIP_H), FINGER_OPEN, "let go", -CATCH_OPEN, p90),
        w(0.8, (hx + 0.04 * out_n[0], hy + 0.04 * out_n[1], TOP + GRIP_H), FINGER_OPEN, "back off", -CATCH_OPEN,
          p90),
        w(1.0, (0.30, -0.27, TOP + PUSH_H), FINGER_CLOSED, "to the right", 60.0, math.pi / 4),
        w(1.0, (beside[0], -0.26, TOP + PUSH_H), FINGER_CLOSED, "beside the printer", HOOK_YAW, math.pi / 4),
        w(1.2, (*beside, TOP + PUSH_H), FINGER_CLOSED, "behind the free edge", HOOK_YAW, math.pi / 4),
    ]
    for deg, r in PUSH:
        right.append(w(1.5 if deg <= 14.0 else 1.0, (*inside_face(deg, r), TOP + PUSH_H), FINGER_CLOSED,
                       f"sweep to {deg:.0f} deg", min(90.0 - deg, HOOK_YAW), math.pi / 4))
    last_deg, last_r = PUSH[-1]
    right += [
        w(0.6, (*inside_face(last_deg, last_r), TOP + PUSH_H), FINGER_CLOSED, "hold", 90.0 - last_deg, math.pi / 4),
        w(1.2, (*inside_face(last_deg, last_r - 0.02, 0.08), TOP + PUSH_H + 0.05), FINGER_CLOSED, "away from the door",
          90.0 - last_deg, math.pi / 4),
        w(1.5, (0.25, -0.25, TOP + 0.18), FINGER_OPEN, "back to the right", 0.0, 0.0),
    ]
    # The left hand holds the ledge while the catch opens (the door at 6 degrees), then backs straight off before the
    # door's bottom edge sweeps over the ledge where it pressed (10 degrees there).
    caught = _at_label(right, f"pull to {CATCH_OPEN:.1f} deg") + 0.2
    left = [
        w(2.0, (STEADY[0] - 0.04, STEADY[1], STEADY[2] + 0.08), FINGER_CLOSED, "over the ledge", 0.0, STEADY_PITCH),
        w(1.0, STEADY, FINGER_CLOSED, "press the ledge", 0.0, STEADY_PITCH),
        w(caught - 3.0, STEADY, FINGER_CLOSED, "steady the printer", 0.0, STEADY_PITCH),
        w(0.6, (STEADY[0] - 0.05, STEADY[1], STEADY[2] + 0.03), FINGER_CLOSED, "back off the ledge", 0.0,
          STEADY_PITCH),
        w(1.2, PARK, FINGER_CLOSED, "out of the door's way", 0.0, 0.0),
    ]
    return {"right": right, "left": left}
