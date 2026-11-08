"""On the station table, centred between the two arms: an enclosed desktop 3D printer (36 x 35 x 40 cm, fixed), its
clear door shut, a filament feeder on its roof. On its left wall (+y) is an empty spool holder: a bracket plate with
a 45 mm steel rod sticking out 13 cm to the robot's left, 22 cm above the table, a 3 cm stop collar at its root and a
chamfered nose. A 1 kg spool of orange filament (20 cm across, 7 cm wide, a 52 mm bore) stands on its rims on the
table at the front right, its axis along y, in the right arm's half.

The task: hang the spool on the holder, its bore over the rod, pushed on until it touches the stop collar, and let
go. The spool starts in the right arm's half and the holder is out of the right arm's reach, so in the demo the
right hand brings the spool to the middle of the table and the left hand takes it from there.

The spool is one rigid body of box colliders round an open bore (``spool.py``), the rod and the collar cylinder
colliders on the fixed holder. The bore clears the rod by 3.5 mm all round, and the rod's last 2 cm are chamfered.
``printer.py`` holds the printer's geometry, ``looks.py`` draws the parts, ``fingers.py`` tells the scorer when a
finger is near the spool.

Run it:  inspect-robots run --embodiment yam_arena -E task=spool_onto_printer_holder.py --scorer task_score ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import printer as pr  # noqa: E402
from bonds import on_reset, on_step  # noqa: E402
from colliders import shape  # noqa: E402  (before spool: shape() strips the holder's cuboid, then rod() adds the rod)
from grip import HandWaypoint  # noqa: E402
from looks import dress  # noqa: E402
from spool import Rod, Spool, rod, spool  # noqa: E402
from yam_arena.demo import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0


def _at(x, y, h):
    """A station-frame position from a height above the table."""
    return (x, y, TOP + h)


# -- the printer, fixed, its door shut, the print head parked, the plate drawn on the bed, the feeder on its roof ----
PRINTER = Box("printer", pos=_at(*pr.CENTRE), size=pr.SIZE, color=(0.78, 0.78, 0.8))
shape("printer", pr.printer_parts(bed_top=pr.PLATE_TOP))
DOOR = Box("door", pos=_at(*pr.DOOR_CENTRE), size=pr.DOOR_SIZE, color=(0.45, 0.52, 0.58))
shape("door", pr.door_parts())
PRINT_HEAD = Box("print_head", pos=_at(*pr.HEAD_CENTRE), size=pr.HEAD_SIZE, color=(0.78, 0.79, 0.8))
FEEDER = Box("feeder", pos=_at(*pr.FEEDER_CENTRE), size=pr.FEEDER_SIZE, yaw=-math.pi / 2, color=(0.03, 0.03, 0.035))

# -- the holder: the bracket plate on the wall, the stop collar and the rod, along y ---------------------------------
ROD_Z = TOP + pr.ROD_H
ROD_FRICTION = 0.3  # a smooth steel rod: the spool slides along it (and turns on it)
HOLDER = Box("holder", pos=_at(*pr.HOLDER_CENTRE), size=pr.HOLDER_SIZE, color=(0.04, 0.04, 0.045))
shape("holder", [pr.bracket_part()])
rod(HOLDER, Rod("rod", pr.STOP_Y - 0.001, pr.ROD_TIP_Y, pr.ROD_X, ROD_Z, pr.ROD_R, *pr.NOSE), ROD_FRICTION)
rod(HOLDER, Rod("collar", pr.WALL_Y + pr.BRACKET_T, pr.STOP_Y, pr.ROD_X, ROD_Z, pr.COLLAR_R))

# -- the spool, standing on its rims at the front right, its axis along y ------------------------------------------
SPOOL_MASS = 1.0
SPOOL_W, SPOOL_R = pr.SPOOL_W, pr.SPOOL_R
START_XY = (0.20, -0.17)  # its back rim 1 cm in front of the printer's base
SPOOL = spool(Spool("spool", pos=_at(*START_XY, SPOOL_R), size=pr.SPOOL_SIZE, mass=SPOOL_MASS, yaw=math.pi / 2,
                    color=(1.0, 0.42, 0.0), bore=pr.BORE_R, core=pr.WIND_R, flange=pr.FLANGE_T, sides=24))

OBJECTS = [PRINTER, DOOR, PRINT_HEAD, FEEDER, HOLDER, SPOOL]

# What the boxes look like: meshes modelled in Blender (looks/make_looks.py), the colliders unchanged.
_PLATE_IN_PRINTER = (pr.PLATE_XY[0] - pr.CENTRE[0], pr.PLATE_XY[1] - pr.CENTRE[1],
                     pr.BED_TOP + pr.PLATE_SIZE[2] / 2 - pr.CENTRE[2])
LOOKS = {"printer": ["printer.usdc", ("plate.usdc", _PLATE_IN_PRINTER)], "door": "door.usdc",
         "print_head": "head.usdc", "feeder": "feeder.usdc", "holder": "holder.usdc", "spool": "spool.usdc"}
dress(LOOKS)

# -- scoring ---------------------------------------------------------------------------------------------------------
AXIS_TOL = math.radians(10)  # the spool's axis within this of the rod's
SEATED = 0.01  # m: the spool's inner flange within this of the stop
NEAR_TIP = 0.05  # m: the spool's centre within this of the rod tip's centre
PASSED = 0.15  # m: the spool's centre moved this far or more to the left (+y) of where it stood: passed over
UPRIGHT = math.radians(30)  # its axis within this of level: standing on its rims or held, not lying flat
LET_GO = 0.005  # m: no finger point this close to the spool's outside once it has been let go
OFF_TABLE_Z = -0.02  # m above the table: a spool whose centre is lower has fallen off the table
ROD_TIP = np.array([pr.ROD_X, pr.ROD_TIP_Y, ROD_Z])

# What the episode knows that the objects don't: whether a finger is on the spool, read every control step.
STATE = {"held": False}


@on_reset
def reset_memory(env=None) -> None:
    STATE.update(held=False)


def _rot(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _finite(objects) -> bool:
    s = objects["spool"]
    return bool(np.all(np.isfinite(s.pos)) and np.all(np.isfinite(s.quat_xyzw)) and np.all(np.isfinite(s.lin_vel)))


def spool_axis(objects) -> np.ndarray:
    """The spool's axis (a unit vector, either way along it)."""
    return _rot(objects["spool"].quat_xyzw)[:, 0]


def axis_angle(objects) -> float:
    """The angle between the spool's axis and the rod's (rad), 0 to 90 degrees."""
    return float(math.acos(min(1.0, abs(float(spool_axis(objects)[1])))))


def tilt(objects) -> float:
    """How far the spool's axis is from level (rad): 0 standing on its rims, 90 degrees lying flat."""
    return float(math.asin(min(1.0, abs(float(spool_axis(objects)[2])))))


def off_axis(objects) -> float:
    """How far the spool's centre is from the rod's axis line (m)."""
    c = objects["spool"].pos
    return float(math.hypot(c[0] - pr.ROD_X, c[2] - ROD_Z))


def faces_y(objects) -> tuple[float, float]:
    """Where the spool's two flange faces cross the rod's line, along y: (near the wall, away from it)."""
    c, a = objects["spool"].pos, abs(float(spool_axis(objects)[1]))
    return float(c[1] - SPOOL_W / 2 * a), float(c[1] + SPOOL_W / 2 * a)


def on_rod(objects) -> bool:
    """The rod runs through the bore across the spool's whole width: the spool's centre within 2.6 cm (the bore's
    radius) of the rod's axis, its axis within 10 degrees of the rod's, and both its faces between the stop and the
    rod's tip."""
    inner, outer = faces_y(objects)
    return bool(off_axis(objects) <= pr.BORE_R and axis_angle(objects) <= AXIS_TOL
                and inner >= pr.STOP_Y - 0.005 and outer <= pr.ROD_TIP_Y)


def seated(objects) -> bool:
    """On the rod, its inner flange within 1 cm of the stop collar."""
    return bool(on_rod(objects) and faces_y(objects)[0] - pr.STOP_Y < SEATED)


def at_tip(objects) -> bool:
    """The spool's centre (the bore's middle) within 5 cm of the centre of the rod's tip."""
    return bool(np.linalg.norm(objects["spool"].pos - ROD_TIP) <= NEAR_TIP)


def passed_over(objects) -> bool:
    """The spool's centre 15 cm or more to the left of where it stood (as far as the middle of the table, where
    the left hand can take it), standing on its rims or held, not lying flat."""
    return bool(objects["spool"].pos[1] >= START_XY[1] + PASSED and tilt(objects) <= UPRIGHT)


def fingers_on(objects, fingers) -> bool:
    """Any finger point within 5 mm of the spool's outside (a cylinder 20 cm across and 7 cm wide)."""
    if fingers is None or len(fingers) == 0:
        return False
    s = objects["spool"]
    local = (np.asarray(fingers) - s.pos) @ _rot(s.quat_xyzw)
    radial = np.hypot(local[:, 1], local[:, 2])
    return bool(np.any((np.abs(local[:, 0]) <= SPOOL_W / 2 + LET_GO) & (radial <= SPOOL_R + LET_GO)))


@on_step
def watch(env) -> None:
    """Every control step: is a finger on the spool?"""
    from fingers import finger_points
    from yam_arena.scene import object_states

    objects = object_states(env, ["spool"])
    STATE["held"] = fingers_on(objects, finger_points(env)) if _finite(objects) else False


def _violation(objects) -> bool:
    return bool(not _finite(objects) or objects["spool"].pos[2] < TOP + OFF_TABLE_Z)


def success(objects) -> bool:
    """Seated against the stop, let go (no finger within 5 mm of the spool) and still."""
    if _violation(objects):
        return False
    return bool(seated(objects) and not STATE["held"] and objects["spool"].resting(max_speed=0.01, max_spin=0.5))


def score(objects) -> float:
    """1.0: success (held 1 s). 0.5: on the rod, but not at the stop, or still held, or still moving. 0.25: the
    spool's centre within 5 cm of the rod's tip. 0.1: the spool moved 15 cm or more to the left, standing or held. 0
    otherwise, and 0 if the spool fell off the table."""
    if _violation(objects):
        return 0.0
    if success(objects):
        return 1.0
    if on_rod(objects):
        return 0.5
    if at_tip(objects):
        return 0.25
    return 0.1 if passed_over(objects) else 0.0


def distance(objects) -> float:
    """From the spool's centre to where it sits at the stop."""
    goal = np.array([pr.ROD_X, pr.STOP_Y + SPOOL_W / 2, ROD_Z + (pr.BORE_R - pr.ROD_R)])
    return float(np.linalg.norm(objects["spool"].pos - goal))


# -- scripted demo ---------------------------------------------------------------------------------------------------
# Each hand takes the spool by the top of its rim, its fingers closing along y on the outsides of both flanges (the
# full 50 N grip), tilted forward so the arm reaches. The spool hangs below the grip, its axis along y. The right hand
# lifts it, carries it to the middle of the table, sets it down on its rims, lets go and backs away. The left hand then
# takes it there, lifts it to the rod's height, carries it left in front of the printer and then forward along its
# side to beyond the rod's tip, lines its bore up with the rod, slides it on until it leans on the stop collar, lets
# go and draws back up and out. Numbers are TCP positions.
GRIP_DOWN = 0.012  # m: the TCP this far below the rim's top (the flanges are solid discs)
MID_XY = (0.20, -0.01)  # where the right hand sets the spool down for the left
PITCH = {"right": 30.0, "left": 70.0}  # degrees: each hand's grip tilted forward from straight down
CLOSING = {"right": -1.0, "left": -1.0}  # which way each hand's fingers close (keeps joint6 inside its limits)
SAG = 0.012  # m: how far the spool held by the right hand hangs below where the arm is sent (the arm gives under
# its weight)
LEFT_HOLD = (0.0075, 0.026, 0.100)  # m: how far the left hand is sent past where the spool should be (x, y, up):
# the left arm, reaching out to the rod with the 1 kg spool, gives that much (measured)
LIFT = 0.04  # m: the right hand lifts the spool this much to carry it
LINE_Y = pr.ROD_TIP_Y + SPOOL_W / 2 + 0.012  # the spool's centre lined up 1.2 cm off the rod's tip
SEAT_Y = pr.STOP_Y + SPOOL_W / 2 - 0.003  # 3 mm past the stop: the hand leans the spool on the collar
OPEN = FINGER_OPEN  # to let go: the pads 9.4 cm apart, 1.2 cm off each flange
LEFT_WAIT = 10.6  # s: the left hand waits this long out of the right hand's way
HOLD_ON = 0  # 1: the left hand never lets go (the shortcut the 0.5 rung scores)
ROUTE_X = 0.21  # the spool carried left along this x: its back rim 1.1 cm in front of the door's handle


def grip(arm: str) -> dict:
    p, k = math.radians(PITCH[arm]), CLOSING[arm]
    return {"approach": (math.sin(p), 0.0, -math.cos(p)), "closing": (0.0, k, 0.0)}


def hand_at(centre, arm: str, rise: float = 0.0) -> tuple[float, float, float]:
    """The TCP holding the spool by the top of its rim with its centre at ``centre`` (raised ``rise``): 1.2 cm
    below the rim's top, so the pads lie on the flanges' outsides whichever way the fingers point."""
    return tuple(float(v) for v in np.asarray(centre, float) + np.array([0.0, 0.0, SPOOL_R - GRIP_DOWN + rise]))


def demo():
    gr, gl = grip("right"), grip("left")
    c0 = np.array(SPOOL.pos)
    mid = np.array([*MID_XY, SPOOL.pos[2]])
    lifted = np.array([0.0, 0.0, LIFT])
    held = np.array([0.0, 0.0, SAG])  # the right arm sent this much higher than the spool should be
    held_l = np.array(LEFT_HOLD)  # and the left arm this far off

    def r(duration, centre, finger, label, rise=0.0):
        return HandWaypoint(duration, hand_at(centre, "right", rise), finger, label, **gr)

    def lw(duration, centre, finger, label, rise=0.0):
        return HandWaypoint(duration, hand_at(centre, "left", rise), finger, label, **gl)

    right = [
        r(1.5, c0, FINGER_OPEN, "over the spool", 0.06),
        r(1.2, c0, FINGER_OPEN, "down over the rim"),
        r(0.8, c0, FINGER_CLOSED, "take it"),
        r(1.2, c0 + lifted + held, FINGER_CLOSED, "lift"),
        r(2.5, mid + lifted + held, FINGER_CLOSED, "to the middle"),
        r(1.2, mid + np.array([0.0, 0.0, 0.002]) + held, FINGER_CLOSED, "set it down"),
        r(0.8, mid, OPEN, "let go"),
        r(0.8, mid, OPEN, "up off it", 0.07),
        HandWaypoint(1.2, (0.25, -0.26, TOP + 0.24), OPEN, "away", **gr),
    ]
    rod_c = np.array([pr.ROD_X, 0.0, ROD_Z])
    line = rod_c + np.array([0.0, LINE_Y, 0.0])
    seat = rod_c + np.array([0.0, SEAT_Y, 0.0])
    out = rod_c + np.array([0.0, LINE_Y + 0.05, 0.0])
    left = [
        HandWaypoint(LEFT_WAIT, (0.24, 0.22, TOP + 0.28), OPEN, "wait out of the right hand's way", **gl),
        lw(1.5, mid, OPEN, "over the spool", 0.06),
        lw(1.2, mid, OPEN, "down over the rim"),
        lw(0.8, mid, FINGER_CLOSED, "take it"),
        lw(1.5, mid + np.array([0.0, 0.0, ROD_Z - mid[2]]) + held_l, FINGER_CLOSED, "lift to the rod's height"),
        lw(2.6, np.array([ROUTE_X, out[1], out[2]]) + held_l, FINGER_CLOSED, "to the left, in front of the printer"),
        lw(2.2, out + held_l, FINGER_CLOSED, "forward, out to the left of the rod"),
        lw(1.5, line + held_l, FINGER_CLOSED, "line up with the rod"),
        lw(1.0, line + held_l, FINGER_CLOSED, "settle"),
        lw(3.0, seat + held_l, FINGER_CLOSED, "slide it on to the stop"),
        lw(0.6, seat + held_l, FINGER_CLOSED, "hold"),
    ]
    if HOLD_ON:
        return {"right": right, "left": left + [lw(3.0, seat + held_l, FINGER_CLOSED, "keep holding")]}
    left += [
        lw(0.8, seat + held_l, OPEN, "let go"),
        lw(1.0, seat + held_l, OPEN, "up off the rim", 0.06),
        lw(1.2, seat + np.array([-0.08, 0.10, 0.0]), OPEN, "out and away", 0.10),
    ]
    return {"right": right, "left": left}
