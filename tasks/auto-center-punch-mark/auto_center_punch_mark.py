"""On the station table, in front of the right arm: a blue-dyed steel plate (11 x 7 x 1.2 cm, fixed) with a scribed
cross on it, and a little further left a tool holder (a 4 cm block with a pocket) holding an automatic centre punch
upright, its red barrel on top. The punch is two bodies: a nose with a 3 mm hardened tip and a stem, and the barrel, a sleeve
that slides down over the stem against a spring (1500 N/m, 11 mm of travel).

The task: mark the cross's centre with the punch, once.

The punch fires when the barrel has been pushed 7 mm down over its nose, which takes 10.5 N through the tip. The
mark is made where the tip is at that moment, and a shot can't be taken back. The tip is hardened and smooth, so on
the dyed steel it slides (friction 0.15) as soon as the push has any sideways part, and the push is built up slowly
through the barrel: a punch set down off the cross, or tilted, or leaned on from the side, skates away from the
cross on the way to the trip. The scorer reads both bodies' poses every step, works out the barrel's travel and
remembers each shot (``STATE``), and ``score()`` reads the mark's offset from the cross.

The demo grips the barrel from above with the right arm, lifts the punch out of its holder, sets the tip on the cross,
pushes the barrel down until it fires, lifts the punch clear and holds it still. The left arm stays at READY.

Run it:  inspect-robots run --embodiment yam_arena -E task=auto_center_punch_mark.py --scorer task_score ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from bonds import Sliding, on_reset, on_step  # noqa: E402
from colliders import Part, shape  # noqa: E402
from looks import dress  # noqa: E402
from yam_arena.demo import FINGER_CLOSED, FINGER_OPEN, Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0


def _at(x, y, h):
    """A station-frame position from a height above the table."""
    return (x, y, TOP + h)


# -- the plate: blue layout dye on steel, fixed, with the scribed cross ---------------------------------------------
PLATE_C = (0.30, -0.33)
PLATE_SIZE = (0.11, 0.07, 0.012)
PLATE_TOP = TOP + PLATE_SIZE[2]
CROSS = (PLATE_C[0] + 0.010, PLATE_C[1] + 0.006)  # the cross's centre, off the plate's centre
CROSS_ARM = 0.012  # m: each scribed line runs 1.2 cm either side of the centre
PLATE = Box("plate", pos=_at(*PLATE_C, PLATE_SIZE[2] / 2), size=PLATE_SIZE, color=(0.10, 0.25, 0.75))

# -- the tool holder: a block with a pocket for the nose and a hole for its tip, fixed -------------------------------
HOLDER_C = (0.27, -0.14)
HOLDER_SIZE = (0.040, 0.040, 0.028)
HOLE = 0.007  # m: the hole under the tip, 2 mm clear of it each side
FLOOR_Z = 0.018  # the nose rests on the pocket's floor, this high above the table
POCKET = 0.021  # m: the pocket, 2.5 mm clear of the nose each side
BASE_Z = 0.010  # the hole's floor
HOLDER = Box("holder", pos=_at(*HOLDER_C, HOLDER_SIZE[2] / 2), size=HOLDER_SIZE, color=(0.14, 0.14, 0.16))


def _holder_parts():
    w, h, hh = HOLDER_SIZE[0], HOLDER_SIZE[2], HOLDER_SIZE[2] / 2
    a, p = w / 2, POCKET / 2
    k = HOLE / 2
    parts = [Part("base", (0, 0, BASE_Z / 2 - hh), (w, w, BASE_Z))]
    mid = (BASE_Z + FLOOR_Z) / 2 - hh  # the floor ring around the hole
    t = FLOOR_Z - BASE_Z
    parts += [Part("ring_l", (-(a + k) / 2, 0, mid), (a - k, w, t)), Part("ring_r", ((a + k) / 2, 0, mid), (a - k, w, t)),
              Part("ring_f", (0, -(a + k) / 2, mid), (HOLE, a - k, t)), Part("ring_b", (0, (a + k) / 2, mid), (HOLE, a - k, t))]
    top = (FLOOR_Z + h) / 2 - hh  # the pocket's walls
    wall = h - FLOOR_Z
    parts += [Part("wall_l", (-(a + p) / 2, 0, top), (a - p, w, wall)), Part("wall_r", ((a + p) / 2, 0, top), (a - p, w, wall)),
              Part("wall_f", (0, -(a + p) / 2, top), (POCKET, a - p, wall)),
              Part("wall_b", (0, (a + p) / 2, top), (POCKET, a - p, wall))]
    return parts


shape("holder", _holder_parts())

# -- the punch: a nose with the tip and a stem, and the barrel, a sleeve over the stem, sliding down on a spring -----
TIP_SIZE = (0.003, 0.003, 0.004)  # the hardened tip
SHOULDER_SIZE = (0.016, 0.016, 0.030)  # the nose, resting on the holder's pocket floor
STEM_SIZE = (0.010, 0.010, 0.034)  # the stem above the nose, going up into the barrel
TIP_BOX = (0.016, 0.016, TIP_SIZE[2] + SHOULDER_SIZE[2] + STEM_SIZE[2])  # the nose's body: tip, nose and stem
TIP_FRICTION = (0.15, 0.12)  # hardened steel on dyed steel: the lower of the two wins ("min")
BARREL_SIZE = (0.022, 0.022, 0.085)
SLEEVE = (0.012, 0.035)  # the barrel's hollow: its square width and its depth from the bottom
GAP = 0.013  # m: from the nose's top to the barrel's bottom (the stem shows here)
TRAVEL = 0.011  # m: the barrel slides this far down over the nose
RATE = 1500.0  # N/m: the spring
DAMPING = 10.0  # N s/m
TRIP = 0.007  # m: the punch fires when the barrel is this far down over the nose (10.5 N)
REARM = 0.003  # m: and can fire again once the barrel has come back up past this
TIP_Z = FLOOR_Z - TIP_SIZE[2]  # the tip's underside, standing in the holder
TIP_C = _at(*HOLDER_C, TIP_Z + TIP_BOX[2] / 2)
BARREL_BOTTOM = TIP_Z + TIP_SIZE[2] + SHOULDER_SIZE[2] + GAP  # above the table
BARREL_C = _at(*HOLDER_C, BARREL_BOTTOM + BARREL_SIZE[2] / 2)
REST_GAP = BARREL_C[2] - TIP_C[2]  # the barrel's centre above the nose's centre with the spring unloaded

TIP = Box("tip", pos=TIP_C, size=TIP_BOX, mass=0.04, color=(0.7, 0.7, 0.72))
shape("tip", [Part("point", (0, 0, -TIP_BOX[2] / 2 + TIP_SIZE[2] / 2), TIP_SIZE, friction=TIP_FRICTION, combine="min"),
              Part("shoulder", (0, 0, -TIP_BOX[2] / 2 + TIP_SIZE[2] + SHOULDER_SIZE[2] / 2), SHOULDER_SIZE),
              Part("stem", (0, 0, TIP_BOX[2] / 2 - STEM_SIZE[2] / 2), STEM_SIZE)])
BARREL = Sliding("barrel", pos=BARREL_C, size=BARREL_SIZE, mass=0.12, color=(0.85, 0.08, 0.06), to="tip", axis="Z",
                 lower=-TRAVEL, upper=0.0, rate=RATE, damping=DAMPING)


def _barrel_parts():
    w, h = BARREL_SIZE[0], BARREL_SIZE[2]
    s, d = SLEEVE
    t = (w - s) / 2
    c = (w + s) / 4
    z = -h / 2 + d / 2
    return [Part("wall_l", (-c, 0, z), (t, w, d)), Part("wall_r", (c, 0, z), (t, w, d)),
            Part("wall_f", (0, -c, z), (s, t, d)), Part("wall_b", (0, c, z), (s, t, d)),
            Part("cap", (0, 0, -h / 2 + d + (h - d) / 2), (w, w, h - d))]


shape("barrel", _barrel_parts())

OBJECTS = [PLATE, HOLDER, TIP, BARREL]

# What the boxes look like: meshes modelled in Blender (looks/make_looks.py), the colliders unchanged.
dress({"plate": "plate.usdc", "holder": "holder.usdc", "tip": "tip.usdc", "barrel": "barrel.usdc"})

# -- scoring ---------------------------------------------------------------------------------------------------------
BAND_FULL, BAND_HALF, BAND_QUARTER = 0.0005, 0.001, 0.002  # m: the mark's offset from the cross's centre
ON_PLATE = 0.002  # m: the tip's underside within this of the plate's top when it fires: the shot is on the plate
CLEAR = 0.02  # m: the tip's underside this far above the plate's top: the punch is lifted clear
OFF_TABLE_Z = -0.02  # m above the table: a body lower than this has fallen off
FREE = ["tip", "barrel"]
CROSS_3D = np.array([CROSS[0], CROSS[1], PLATE_TOP])

STILL_WINDOW = 20  # control steps: the last second, at 20 Hz
STILL_MOVE = 0.001  # m: still means neither part's centre moved this far over the last second

# What the episode remembers: each shot (where the tip's underside was when the punch fired, and whether it was on
# the plate), whether the punch is cocked (the barrel has come back up since the last shot), and where the two parts
# were over the last second (read every step: a held punch buzzes in the pads, so a speed is no test of stillness).
STATE = {"shots": [], "armed": True, "track": []}


@on_reset
def reset_memory(env=None) -> None:
    STATE.update(shots=[], armed=True, track=[])


def matrix(q) -> np.ndarray:
    """Rotation matrix of a quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _finite(objects) -> bool:
    return all(bool(np.all(np.isfinite(objects[n].pos)) and np.all(np.isfinite(objects[n].quat_xyzw))
                    and np.all(np.isfinite(objects[n].lin_vel))) for n in FREE)


def travel(objects) -> float:
    """How far the barrel is down over the nose (m), along the nose's own axis."""
    up = matrix(objects["tip"].quat_xyzw)[:, 2]
    return float(REST_GAP - (objects["barrel"].pos - objects["tip"].pos) @ up)


def point(objects) -> np.ndarray:
    """The centre of the tip's underside."""
    return objects["tip"].pos - matrix(objects["tip"].quat_xyzw)[:, 2] * (TIP_BOX[2] / 2)


def tip_height(objects) -> float:
    """The tip's underside above the plate's top (m)."""
    return float(point(objects)[2] - PLATE_TOP)


def offset(p) -> float:
    """A point's distance across the plate from the cross's centre (m)."""
    return float(np.linalg.norm(np.asarray(p, float)[:2] - np.asarray(CROSS)))


def watch_shots(objects) -> None:
    """Into the episode's memory: where the parts are, a shot when the barrel passes the trip while cocked, and the cocking when it
    comes back up."""
    if not _finite(objects):
        return
    STATE["track"] = (STATE["track"] + [_centres(objects)])[-(STILL_WINDOW + 1):]
    t = travel(objects)
    if STATE["armed"] and t >= TRIP:
        p = point(objects)
        STATE["shots"].append((float(p[0]), float(p[1]), bool(abs(p[2] - PLATE_TOP) <= ON_PLATE)))
        STATE["armed"] = False
    elif not STATE["armed"] and t <= REARM:
        STATE["armed"] = True


@on_step
def watch(env) -> None:
    from yam_arena.scene import object_states

    watch_shots(object_states(env, FREE))


def fell(objects) -> bool:
    return any(objects[n].pos[2] < TOP + OFF_TABLE_Z for n in FREE)


def _centres(objects) -> np.ndarray:
    return np.array([objects[n].pos for n in FREE], float)


def still(objects) -> bool:
    """Neither the nose's nor the barrel's centre moved 1 mm over the last second (read every step). False before a
    second has been recorded."""
    track = STATE["track"]
    if len(track) <= STILL_WINDOW:
        return False
    return bool(np.linalg.norm(_centres(objects) - track[-1 - STILL_WINDOW], axis=1).max() < STILL_MOVE)


def one_shot() -> tuple | None:
    """The one shot taken on the plate (x, y, on plate), or None when there were none or more than one."""
    return STATE["shots"][0] if len(STATE["shots"]) == 1 else None


def mark_offset() -> float | None:
    """The offset (m) of the mark from the cross's centre, or None when there is no single mark on the plate."""
    shot = one_shot()
    return offset(shot[:2]) if shot is not None and shot[2] else None


def success(objects) -> bool:
    """One shot within 0.5 mm of the cross's centre, the punch lifted clear and still."""
    return score(objects) == 1.0


def score(objects) -> float:
    """1.0: one shot within 0.5 mm of the cross's centre, and the punch lifted clear (the tip 2 cm above the plate)
    and still (neither part's centre moved 1 mm over the last second). 0.5: one shot within 1 mm (also a shot within 0.5 mm with the punch not yet clear and still). 0.25:
    one shot within 2 mm. 0 otherwise: no shot, two shots or more, a shot off the plate, or farther than 2 mm. 0 as
    well if the punch fell off the table or the state is not finite."""
    if not _finite(objects) or fell(objects):
        return 0.0
    off = mark_offset()
    if off is None:
        return 0.0
    if off <= BAND_FULL and tip_height(objects) >= CLEAR and still(objects):
        return 1.0
    if off <= BAND_HALF:
        return 0.5
    return 0.25 if off <= BAND_QUARTER else 0.0


def distance(objects) -> float:
    """How far the tip's underside is from the cross's centre (m)."""
    return float(np.linalg.norm(point(objects) - CROSS_3D))


# -- scripted demo ---------------------------------------------------------------------------------------------------
# The right arm grips the barrel from above, its fingers closing along y, takes the punch up out of its holder and
# carries it over the cross with the tip 1.2 cm above the plate, lowers it until the tip is 3 mm above the cross, then
# pushes down slowly through the barrel: the tip lands on the cross, the barrel slides down over the nose, and at 7 mm
# the punch fires. The arm leans on a little longer, then lifts the punch clear and holds it still. Numbers are TCP
# positions (the TCP is on the barrel's axis, level with its middle when the punch stands in the holder).
GRIP_DZ = 0.0  # m: the grip above the barrel's middle
HOVER_Z = 0.17  # the hand comes down onto the barrel from this high
LIFT = 0.04  # m: the punch is lifted this far out of its holder
APPROACH = 0.012  # m: the tip is carried over the cross this high above the plate
SETTLE = 0.003  # m: and lowered to this before the push
LEAN = 0.030  # m: the hand pushes this far past the tip's touch (the arm gives about two thirds of it, the barrel the rest)
AIM = (0.0, 0.00066)  # m: the hand's x and y offset from the cross (measured: the tip lands 0.66 mm short in y without it)
SHOTS = 1  # how many times the demo pushes (more is the shortcut that scores 0)
SIDE = 0.0  # m: a sideways push added while the barrel goes down (leaning on the punch scores less)


def _tcp(x, y, tip_h):
    """The TCP with the tip's underside ``tip_h`` above the plate: the TCP is REST_GAP + 1/2 the tip box above the
    tip's underside (the barrel's centre) plus the grip's offset."""
    return (x, y, PLATE_TOP + tip_h + REST_GAP + TIP_BOX[2] / 2 + GRIP_DZ)


def demo():
    hx, hy, _ = HOLDER_C[0], HOLDER_C[1], 0
    grasp = (hx, hy, BARREL_C[2] + GRIP_DZ)
    x, y = CROSS[0] + AIM[0], CROSS[1] + AIM[1]
    open_ = FINGER_OPEN + 0.012  # fingers 7 cm apart: the barrel is 2 cm across
    steps = [
        Waypoint(1.5, (hx, hy, TOP + HOVER_Z), open_, "over the punch"),
        Waypoint(1.5, grasp, open_, "down to the barrel"),
        Waypoint(0.8, grasp, FINGER_CLOSED, "close on it"),
        Waypoint(1.2, (hx, hy, grasp[2] + LIFT), FINGER_CLOSED, "lift it out of the holder"),
        Waypoint(2.5, (x, y, grasp[2] + LIFT), FINGER_CLOSED, "carry it over the cross"),
        Waypoint(1.2, _tcp(x, y, APPROACH), FINGER_CLOSED, "lower"),
        Waypoint(1.0, _tcp(x, y, SETTLE), FINGER_CLOSED, "tip just above the cross"),
        Waypoint(0.5, _tcp(x, y, SETTLE), FINGER_CLOSED, "settle"),
    ]
    for k in range(SHOTS):
        steps += [
            Waypoint(7.0, _tcp(x + SIDE, y, SETTLE - SETTLE - LEAN), FINGER_CLOSED, "push down slowly until it fires"),
            Waypoint(0.5, _tcp(x + SIDE, y, SETTLE - SETTLE - LEAN), FINGER_CLOSED, "lean on it"),
            Waypoint(1.5, _tcp(x, y, 0.06), FINGER_CLOSED, "lift the punch clear"),
        ]
    steps += [Waypoint(2.5, _tcp(x, y, 0.06), FINGER_CLOSED, "hold it still")]
    return {"right": steps}
