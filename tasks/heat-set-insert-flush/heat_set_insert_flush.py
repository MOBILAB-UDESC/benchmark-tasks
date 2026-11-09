"""On the station table, a little right of the robot's midline: a white 3D-printed bracket (3 x 5 x 8 cm, fixed) stands with
its face toward the robot. In that face, 6.6 cm up, is a round 5.6 mm hole 10.5 mm deep, and on its mouth sits a brass
heat-set threaded insert (6 mm across, 8 mm long), standing 8 mm proud of the face, its threaded end up toward the robot.
On the robot's right, a soldering iron (22 cm, blue handle, steel barrel, a 2.6 mm square pusher tip) lies in its stand,
its tip toward the bracket: the handle on a rest, the barrel in a cradle.

The task: with the iron, push the insert into the hole until its top is flush with the bracket's face and the insert is
square to it, then take the iron away. The plastic is hot where the iron presses and sets where it doesn't: the insert
sinks at a speed in proportion to the push (about 4 mm/s for every newton) for as long as the iron pushes, and stays
where it is when the push stops, so there is nothing to stop it at flush but the iron. A push that carries on buries it,
a tilted iron cocks it, and an iron that misses the hole touches the bracket's face.

``seat.py`` puts the insert on a damped sink-and-tilt joint, ``colliders.py`` builds the bracket (a hole in its face), the
iron and its stand from box colliders, ``grip.py`` gives the demo its grips, ``bonds.py`` the hooks, and ``looks.py``
draws the parts.

Run it:  inspect-robots run --embodiment yam_arena -E task=heat_set_insert_flush.py --scorer task_score ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from bonds import on_reset, on_step  # noqa: E402
from colliders import Part, shape  # noqa: E402
from grip import HandWaypoint  # noqa: E402
from looks import dress  # noqa: E402
from seat import Seated  # noqa: E402
from yam_arena.demo import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402
from yam_arena.urdf import quat_xyzw_to_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0

# -- the bracket: a block standing on the table, its face toward the robot, a hole in the face -------------------------
FACE_X = 0.44  # m: the bracket's face
HOLE_Y = -0.06  # m: the hole's axis, along x
HOLE_Z = TOP + 0.066  # m: and its height
PART_SIZE = (0.03, 0.05, 0.08)  # deep (x), wide (y), tall
PART_CENTRE = (FACE_X + PART_SIZE[0] / 2, HOLE_Y, TOP + PART_SIZE[2] / 2)
HOLE_DEPTH = 0.0105  # m: from the face to the hole's bottom
HOLE_DRAWN = 0.0056  # m: the drawn hole's diameter (the insert presses in, melting its way)
HOLE_OPEN = 0.008  # m: the colliders' opening (square), wider than the drawn hole so the insert can cock in it
HOLE_HALF = 0.003  # m: the tip's edge past this from the hole's axis, at the face, is on the bracket (the insert's half width)
_hz = HOLE_Z - PART_CENTRE[2]  # the hole's height above the bracket's centre
_ring = HOLE_DEPTH
_o = HOLE_OPEN / 2
PART = Box("part", pos=PART_CENTRE, size=PART_SIZE, color=(0.92, 0.92, 0.9))
shape("part", [
    Part("back", ((-PART_SIZE[0] / 2 + _ring + PART_SIZE[0] / 2) / 2, 0.0, 0.0),
         (PART_SIZE[0] - _ring, PART_SIZE[1], PART_SIZE[2])),
    *(Part(f"side_{k}", (-PART_SIZE[0] / 2 + _ring / 2, s * (_o + (PART_SIZE[1] / 2 - _o) / 2), 0.0),
           (_ring, PART_SIZE[1] / 2 - _o, PART_SIZE[2])) for k, s in (("left", 1), ("right", -1))),
    Part("above", (-PART_SIZE[0] / 2 + _ring / 2, 0.0, (_hz + _o + PART_SIZE[2] / 2) / 2),
         (_ring, HOLE_OPEN, PART_SIZE[2] / 2 - _hz - _o)),
    Part("below", (-PART_SIZE[0] / 2 + _ring / 2, 0.0, (_hz - _o - PART_SIZE[2] / 2) / 2),
         (_ring, HOLE_OPEN, PART_SIZE[2] / 2 + _hz - _o)),
])

# -- the insert: brass, 6 mm square in its box (drawn round), 8 mm long along x, on the hole's mouth --------------------
INS_L, INS_W = 0.008, 0.006
INSERT = Seated("insert", pos=(FACE_X - INS_L / 2, HOLE_Y, HOLE_Z), size=(INS_L, INS_W, INS_W), mass=0.004,
                color=(0.85, 0.6, 0.2), axis="X", travel=HOLE_DEPTH, damping=250.0, tilt=10.0, tilt_damping=0.05)
MOUTH = np.array([FACE_X, HOLE_Y, HOLE_Z])  # the hole's mouth, its centre

# -- the iron: one 22 x 2 x 2 cm box along x, its tip toward +x, colliding as a handle, a barrel and a pusher tip -------
IRON_SIZE = (0.22, 0.02, 0.02)
HANDLE = (0.11, 0.02, 0.02)
BARREL = (0.065, 0.008, 0.008)
TIP = (0.045, 0.0026, 0.0026)
IRON_PARTS = [
    Part("handle", (-IRON_SIZE[0] / 2 + HANDLE[0] / 2, 0.0, 0.0), HANDLE),
    Part("barrel", (-IRON_SIZE[0] / 2 + HANDLE[0] + BARREL[0] / 2, 0.0, 0.0), BARREL),
    Part("tip", (IRON_SIZE[0] / 2 - TIP[0] / 2, 0.0, 0.0), TIP),
]
TIP_LOCAL = np.array([IRON_SIZE[0] / 2, 0.0, 0.0])  # the tip's face, its centre
STAND_XY = (0.25, -0.26)  # the iron's centre when it lies in its stand
IRON_Z = TOP + 0.055
LIFT = HOLE_Z - IRON_Z  # m: how far the iron is lifted off its rest to line up with the hole
IRON = Box("iron", pos=(*STAND_XY, IRON_Z), size=IRON_SIZE, mass=0.06, color=(0.05, 0.2, 0.75))
shape("iron", IRON_PARTS)

# -- the stand: a base, a flat rest under the handle and a cradle under the barrel ----------------------------------------
REST_S = (-0.095, -0.075)  # the handle rest's ends, along the iron from its centre
CRADLE_S = (0.02, 0.06)  # the barrel cradle's ends
REST_TOP = IRON_Z - HANDLE[2] / 2
CRADLE_TOP = IRON_Z - BARREL[2] / 2
STAND_BASE = (0.19, 0.07, 0.012)
BASE_TOP = TOP + STAND_BASE[2]


def _stand(name, s0, s1, t, z0, z1, width, color=(0.62, 0.63, 0.65)):
    return Box(f"stand_{name}", pos=(STAND_XY[0] + (s0 + s1) / 2, STAND_XY[1] + t, (z0 + z1) / 2),
               size=(s1 - s0, width, z1 - z0), color=color)


STAND = [
    _stand("base", -0.095 - 0.0, 0.095, 0.0, TOP, BASE_TOP, STAND_BASE[1], (0.12, 0.12, 0.13)),
    _stand("rest", *REST_S, 0.0, BASE_TOP, REST_TOP, 0.016),
    _stand("cradle_post", 0.03, 0.05, 0.0, BASE_TOP, CRADLE_TOP - 0.003, 0.008),
    _stand("cradle_floor", *CRADLE_S, 0.0, CRADLE_TOP - 0.003, CRADLE_TOP, 0.016),
]

OBJECTS = [PART, *STAND, INSERT, IRON]

# What the boxes look like: meshes modelled in Blender (looks/make_looks.py), the colliders unchanged. The stand's base
# mesh draws the whole stand, and the part's the bracket with its hole.
LOOKS = {"part": "part.usdc", "insert": "insert.usdc", "iron": "iron.usdc", "stand_base": "stand.usdc"}
dress(LOOKS, hide=[b.name for b in STAND[1:]])

# -- scoring ---------------------------------------------------------------------------------------------------------
FLUSH_TIGHT = 0.0003  # m: the insert's top within this of the face is flush (1.0)
FLUSH_LOOSE = 0.001  # m: within this of it (0.5)
BURIED = 0.001  # m: the top further below the face than this is buried (0)
HALF_IN = 0.004  # m: the top no more than this proud of the face is at least half in (0.25)
SQUARE_TIGHT = 3.0  # degrees: the insert's axis off the hole's axis (1.0)
SQUARE_LOOSE = 8.0  # degrees (0.5)
CLEAR = 0.01  # m: the iron's tip at least this far from the hole's mouth is taken away
SURFACE_PLAY = 0.0005  # m: a point of the tip within this of the face's plane, or past it, is at the face
OFF_TABLE_Z = -0.02  # m above the table: an iron or insert whose centre is lower has fallen
TIP_PTS = np.array([[x, s * TIP[1] / 2, t * TIP[2] / 2] for x in (TIP_LOCAL[0], TIP_LOCAL[0] - 0.01)
                    for s in (-1, 1) for t in (-1, 1)])  # the tip's front corners and its corners 1 cm back

# What the episode remembers: whether the iron's tip ever touched the bracket's face outside the hole.
STATE = {"scorched": False}


@on_reset
def reset_memory(env=None) -> None:
    STATE.update(scorched=False)


def _finite(objects) -> bool:
    return all(bool(np.all(np.isfinite(objects[n].pos)) and np.all(np.isfinite(objects[n].quat_xyzw))
                    and np.all(np.isfinite(objects[n].lin_vel))) for n in ("insert", "iron"))


def _rot(state) -> np.ndarray:
    return quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, float))


def insert_axis(objects) -> np.ndarray:
    """The insert's own x axis in the world: along its length, pointing into the hole when it stands square."""
    return _rot(objects["insert"])[:, 0]


def proud(objects) -> float:
    """How far the insert's top stands out of the bracket's face (m), along the hole's axis: 8 mm at the start, 0 when
    flush, negative when buried. The top's centre is half the insert's length back from its centre along its axis."""
    top = np.asarray(objects["insert"].pos, float) - INS_L / 2 * insert_axis(objects)
    return float(FACE_X - top[0])


def square_off(objects) -> float:
    """The angle (degrees) between the insert's axis and the hole's."""
    return float(math.degrees(math.acos(np.clip(insert_axis(objects)[0], -1.0, 1.0))))


def tip_point(objects) -> np.ndarray:
    """The iron's tip face, its centre."""
    return np.asarray(objects["iron"].pos, float) + _rot(objects["iron"]) @ TIP_LOCAL


def touching_face(objects) -> bool:
    """A point of the tip (its front corners, or its corners 1 cm back) at the face's plane or past it, more than 3 mm
    from the hole's axis (across or up): the tip is on the bracket, not in its hole."""
    p = np.asarray(objects["iron"].pos, float) + TIP_PTS @ _rot(objects["iron"]).T
    at_face = p[:, 0] > FACE_X - SURFACE_PLAY
    off_axis = np.maximum(np.abs(p[:, 1] - HOLE_Y), np.abs(p[:, 2] - HOLE_Z)) > HOLE_HALF
    return bool(np.any(at_face & off_axis))


def tip_clear(objects) -> bool:
    return bool(np.linalg.norm(tip_point(objects) - MOUTH) >= CLEAR)


def _fell(objects) -> bool:
    return any(objects[n].pos[2] < TOP + OFF_TABLE_Z for n in ("insert", "iron"))


def observe(objects) -> None:
    """One control step of the episode's memory: the tip on the bracket's face."""
    if _finite(objects) and touching_face(objects):
        STATE["scorched"] = True


@on_step
def watch(env) -> None:
    from yam_arena.scene import object_states

    observe(object_states(env, ["insert", "iron"]))


def ruined(objects) -> bool:
    """A state that scores 0 whatever else: not finite, fallen, or the tip touched the bracket's face at any time."""
    if not _finite(objects) or _fell(objects):
        return True
    if touching_face(objects):
        STATE["scorched"] = True
    return STATE["scorched"]


def success(objects) -> bool:
    """The insert flush within 0.3 mm and square within 3 degrees, still, and the iron's tip 1 cm or more clear."""
    if ruined(objects):
        return False
    return bool(abs(proud(objects)) <= FLUSH_TIGHT and square_off(objects) <= SQUARE_TIGHT
                and objects["insert"].resting() and tip_clear(objects))


def score(objects) -> float:
    """1.0: success. 0.5: the insert's top within 1 mm of the face and its axis within 8 degrees otherwise. 0.25: the
    top no more than 4 mm proud of the face (half in or more), and no more than 1 mm below it, otherwise. 0 otherwise,
    0 if the insert is buried (its top more than 1 mm below the face), and 0 whatever else if the tip touched the
    bracket's face outside the hole at any time or the iron or insert fell off the table."""
    if ruined(objects):
        return 0.0
    p = proud(objects)
    if p < -BURIED:
        return 0.0
    if success(objects):
        return 1.0
    if abs(p) <= FLUSH_LOOSE and square_off(objects) <= SQUARE_LOOSE:
        return 0.5
    return 0.25 if p <= HALF_IN else 0.0


def distance(objects) -> float:
    """How far the insert's top is from flush (m)."""
    return abs(proud(objects))


# -- scripted demo ---------------------------------------------------------------------------------------------------
# Right arm, gripper straight down, its fingers closing across the iron: grip the handle over its centre of mass, lift
# the iron 11 mm off its rest (its axis is then at the hole's height), carry it across to the hole's line, bring its tip
# to 1.5 cm short of the insert, push 2.3 cm along x (the insert sinks 8 mm to flush as the arm leans on it), hold, and
# draw the iron back 4 cm. Numbers are TCP positions. The cheat variants change one constant each.
OVERSHOOT = 0.0  # m: push this much further than flush (the shortcut of pushing "all the way": buries the insert)
TILT_DEG = 0.0  # degrees: turn the held iron about the horizontal across its axis (its tip lower) before the push
OFFSET_Y = 0.0  # m: line the tip up this far left of the hole's axis
SHORT = 0.0  # m: stop this far short of flush
HOLD_ON = 0  # 1: leave the iron pressing at flush (it never withdraws)
GRIP_S = -0.045  # m along the iron from its centre: the handle, over its centre of mass
TIP_FROM_GRIP = IRON_SIZE[0] / 2 - GRIP_S
PICK_YAW = 0.0  # degrees: the fingers are sent turned this far about the vertical (toward -x) to close across the iron
CARRY_YAW = 0.0  # and this far while it is carried and pushed (the wrist tracks to about half a degree: tuned)
GRIP_DZ = 0.006  # m: the TCP is this far above the iron's axis, so the pads' tips (1.3 cm below it) stop 3 mm above the
# handle's underside and clear the rest
AIM = (0.0, -0.0057, 0.0003)  # m: x, y, z trims of the tip's aim, tuned in the simulation
PITCH_TRIM = 4.0  # degrees: the grip's pitch about y (tip lower) that leaves the insert square, tuned in the simulation
FLUSH_TRIM = 0.0  # m: the final push's end, from where the tip's face is level with the bracket's face
APPROACH = 0.015  # m: the tip stops this far short of the insert's top before the push
PUSH_S = 5.0
HOLD_S = 2.0
BACK = 0.04
LIFT_Z = IRON_Z + LIFT


def _grip(pitch_deg: float, yaw_deg: float = 0.0) -> dict:
    """The grip with its fingers closing along -y (the other way round puts joint 6 on its limit at the stand: the
    wrist would stop 7 degrees short of square), turned ``yaw_deg`` about the vertical, and pitched."""
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    return {"approach": (math.sin(p), 0.0, -math.cos(p)), "closing": (math.sin(y), -math.cos(y), 0.0)}


def _tcp(tip_x: float, y: float, z: float, pitch_deg: float = 0.0) -> tuple[float, float, float]:
    """The TCP that puts the held iron's tip face at ``tip_x`` and its axis at (y, z), with the grip turned
    ``pitch_deg``. A pitch about y swings the tip down by TIP_FROM_GRIP times the sine, which the TCP's height makes up."""
    p = math.radians(pitch_deg)
    return (tip_x - TIP_FROM_GRIP * math.cos(p) + AIM[0], y + AIM[1],
            z + GRIP_DZ + AIM[2] - TIP_FROM_GRIP * math.sin(p))


def demo():
    def w(duration, pos, finger, label, pitch=0.0, yaw=PICK_YAW):
        return HandWaypoint(duration, pos, finger, label, **_grip(pitch, yaw))

    gx, gy = STAND_XY[0] + GRIP_S, STAND_XY[1]
    y_line = HOLE_Y + OFFSET_Y
    pitch = PITCH_TRIM + TILT_DEG
    tip_near = FACE_X - INS_L - APPROACH
    tip_flush = FACE_X + FLUSH_TRIM + OVERSHOOT - SHORT
    right = [
        w(2.0, (gx, gy, IRON_Z + 0.06), FINGER_OPEN, "over the handle"),
        w(1.2, (gx, gy, IRON_Z + GRIP_DZ), FINGER_OPEN, "down round the handle"),
        w(0.8, (gx, gy, IRON_Z + GRIP_DZ), FINGER_CLOSED, "close"),
        w(0.8, (gx, gy, LIFT_Z + GRIP_DZ), FINGER_CLOSED, "lift"),
        w(3.0, (gx, y_line + AIM[1], LIFT_Z + GRIP_DZ), FINGER_CLOSED, "across to the hole's line", 0.0, CARRY_YAW),
        w(2.5, _tcp(tip_near, y_line, HOLE_Z, pitch), FINGER_CLOSED, "up to the insert", pitch, CARRY_YAW),
        w(0.8, _tcp(tip_near, y_line, HOLE_Z, pitch), FINGER_CLOSED, "settle", pitch, CARRY_YAW),
        w(PUSH_S, _tcp(tip_flush, y_line, HOLE_Z, pitch), FINGER_CLOSED, "push it in", pitch, CARRY_YAW),
        w(HOLD_S, _tcp(tip_flush, y_line, HOLE_Z, pitch), FINGER_CLOSED, "hold", pitch, CARRY_YAW),
    ]
    if not HOLD_ON:
        right.append(w(2.0, _tcp(tip_flush - BACK, y_line, HOLE_Z, pitch), FINGER_CLOSED, "draw the iron back", pitch, CARRY_YAW))
        right.append(w(1.5, _tcp(tip_flush - BACK, y_line, HOLE_Z, pitch), FINGER_CLOSED, "wait", pitch, CARRY_YAW))
    return {"right": right}
