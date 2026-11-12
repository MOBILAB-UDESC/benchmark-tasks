"""On the station table, between the two arms: a long thin blue circuit-board panel (60 x 6 cm, 0.8 mm FR-4, populated,
120 g) lies along y on a low black bench rail (3 cm high, centred at x = 0.26). It is a strip that bends: held by
its two ends it sags in the middle, and held near the middle it droops at both ends. An orange rack stands 14 cm
farther from the robot, a rail 8 cm high along y (centred at x = 0.40). Each arm works in its own half, the right arm
at y < 0 and the left arm at y > 0.

The task: carry the panel to the rack, set it down on the rack's rail and let go, holding it with both arms. The panel
is a chain of 20 rigid segments on springy hinges (``panelkit.py``), each 3 x 6 cm and 0.8 mm thick like the board
itself, that bends like the 0.8 mm FR-4 strip it stands for. Its flatness limit is a bow of 4 mm: the gap between
the panel and the straight line from one end to the other. Lifted by its ends, it bows several times that. Held about
a fifth of its length (13 cm) in from each end, it stays flat to about 1 mm. Full credit needs the panel lying on the
rack's rail, still, with both grippers' fingers clear of it (``fingers.py`` reads where they are).

Run it:  inspect-robots run --embodiment yam_arena -E task=lift_long_panel_no_sag.py --scorer task_score ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from fingers import finger_points  # noqa: E402
from panelkit import bow, dress, on_reset, on_step, panel, panel_look  # noqa: E402
from yam_arena.demo import FINGER_CLOSED, FINGER_OPEN, Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0

# -- the panel ---------------------------------------------------------------------------------------------------------
PANEL_LENGTH, PANEL_WIDTH, PANEL_THICK = 0.60, 0.06, 0.0008  # a 0.8 mm FR-4 strip, and its colliders are as thin
PANEL_MASS = 0.12  # kg, populated: about 53 g of bare laminate and 67 g of parts, six equal groups (see the README)
PANEL_EI = 0.07  # N m^2: measured on the real panel (cantilever test in the README): about 0.056 for the laminate
# (22 GPa) and the rest for the copper
SEGMENTS = 20
BLUE = (0.04, 0.22, 0.85)

# -- the bench rail and the rack: a bar along y on three legs ------------------------------------------------------
BAR = (0.036, 0.64, 0.012)  # x, y, z: 3.6 cm wide, a little longer than the panel
LEG = 0.02
LEG_Y = (-0.28, 0.0, 0.28)
START_X, RACK_X = 0.26, 0.40
START_TOP = TOP + 0.03  # the bench rail's top
RACK_TOP = TOP + 0.08  # the rack's rail: 5 cm higher
BENCH_COLOR, RACK_COLOR, LEG_COLOR = (0.10, 0.10, 0.12), (0.95, 0.42, 0.04), (0.18, 0.18, 0.20)


def _rail(prefix: str, x: float, top: float, bar_color) -> list[Box]:
    parts = [Box(f"{prefix}_bar", pos=(x, 0.0, top - BAR[2] / 2), size=BAR, color=bar_color)]
    h = top - BAR[2] - TOP
    for i, y in enumerate(LEG_Y):
        parts.append(Box(f"{prefix}_leg{i}", pos=(x, y, TOP + h / 2), size=(LEG, LEG, h), color=LEG_COLOR))
    return parts


BENCH = _rail("bench", START_X, START_TOP, BENCH_COLOR)
RACK = _rail("rack", RACK_X, RACK_TOP, RACK_COLOR)
START_Z = START_TOP + PANEL_THICK / 2 + 0.0005  # the panel's centre height at the start (0.5 mm off the rail)
RACK_Z = RACK_TOP + PANEL_THICK / 2  # and lying on the rack's rail

PANEL = panel("panel", (START_X, 0.0, START_Z), length=PANEL_LENGTH, width=PANEL_WIDTH, thickness=PANEL_THICK,
              segments=SEGMENTS, panel_mass=PANEL_MASS, bending_stiffness=PANEL_EI, color=BLUE)

OBJECTS = [*BENCH, *RACK, PANEL]

# What the objects look like (looks/make_looks.py): the colliders stay the boxes and the panel's segments.
_LOOKS = pathlib.Path(__file__).resolve().parent / "looks"
if (_LOOKS / "panel_00.usdc").is_file():
    panel_look("panel", "panel_{:02d}.usdc")
if (_LOOKS / "bench.usdc").is_file() and (_LOOKS / "rack.usdc").is_file():
    dress({"bench_bar": "bench.usdc", "rack_bar": "rack.usdc"}, hide=[b.name for b in (*BENCH[1:], *RACK[1:])])

# -- scoring -----------------------------------------------------------------------------------------------------------
BOW_LIMIT = 0.004  # m: the panel cracks at any step it bows more than this
BOW_CLEAN = 0.003  # m: a full score needs the peak bow of the whole trial at or under this
LIFT_CLEAR = 0.03  # m: every segment this far above its start height at some step: the panel was lifted clear
ON_RACK_X = 0.02  # m: every segment's centre within this of the rack's line
ON_RACK_Y = 0.02  # m: the panel's centre within this of the rack's middle along y
ON_RACK_Z = 0.001  # m: every segment's centre within this of lying on the rack's rail (its lower face on the rail)
FINGER_CLEAR = 0.003  # m: no finger of either gripper within this of any part of the panel: it is let go
OFF_TABLE_Z = -0.02  # m above the table: a segment lower than this has fallen off

# What the episode remembers: the peak bow so far, whether the whole panel was ever lifted clear, and whether a finger of
# either gripper is on the panel right now (read every control step from the running simulation).
STATE = {"peak": 0.0, "lifted": False, "touched": False}


@on_reset
def reset_memory(env=None) -> None:
    STATE.update(peak=0.0, lifted=False, touched=False)


def touching(objects, fingers: np.ndarray) -> bool:
    """Whether any point on a gripper's fingers (an (n, 3) array, the fingers' collision meshes) is within
    ``FINGER_CLEAR`` of the panel: its 20 boxes, each 3 cm x 6 cm x 0.8 mm, turned by the segment's pitch."""
    p = objects["panel"]
    ell = PANEL_LENGTH / SEGMENTS
    half = np.array([PANEL_WIDTH / 2, ell / 2, PANEL_THICK / 2])
    for centre, pitch in zip(p.points, p.tilts):
        d = np.asarray(fingers, float) - centre
        u = np.array([0.0, math.cos(pitch), math.sin(pitch)])  # along the segment
        w = np.array([0.0, -math.sin(pitch), math.cos(pitch)])  # its thickness
        local = np.stack([d[:, 0], d @ u, d @ w], axis=1)
        outside = np.maximum(np.abs(local) - half, 0.0)
        if np.any(np.linalg.norm(outside, axis=1) <= FINGER_CLEAR):
            return True
    return False


@on_step
def watch(env) -> None:
    """Every control step: whether a finger is on the panel, into the episode's memory."""
    from yam_arena.scene import object_states

    objects = object_states(env, ["panel"])
    if np.all(np.isfinite(objects["panel"].points)):
        STATE["touched"] = touching(objects, finger_points(env))


def _violation(objects) -> bool:
    p = objects["panel"].points
    return bool(not np.all(np.isfinite(p)) or np.min(p[:, 2]) < TOP + OFF_TABLE_Z)


def _remember(objects) -> None:
    p = objects["panel"].points
    STATE["peak"] = max(STATE["peak"], bow(p))
    if np.min(p[:, 2]) >= START_Z + LIFT_CLEAR:
        STATE["lifted"] = True


def peak_bow() -> float:
    """The largest bow (m) the panel has had at any step of the episode."""
    return STATE["peak"]


def on_rack(objects) -> bool:
    """Lying on the rack's rail: every segment's centre within 2 cm of its line along x and 4 mm of lying on it,
    and the panel's centre within 2 cm of the rail's middle along y."""
    p = objects["panel"].points
    return bool(
        np.all(np.abs(p[:, 0] - RACK_X) <= ON_RACK_X)
        and abs(p[:, 1].mean()) <= ON_RACK_Y
        and np.all(np.abs(p[:, 2] - RACK_Z) <= ON_RACK_Z)
    )


def _settled(objects) -> bool:
    return bool(objects["panel"].resting())


def _let_go(objects) -> bool:
    """On the rack's rail, still, and no finger on it."""
    return bool(on_rack(objects) and _settled(objects) and not STATE["touched"])


def success(objects) -> bool:
    """The panel lies still on the rack with the grippers off it, and its bow never went over 3 mm."""
    if _violation(objects):
        return False
    _remember(objects)
    return bool(STATE["lifted"] and STATE["peak"] <= BOW_CLEAN and _let_go(objects))


def distance(objects) -> float:
    """How far the panel's centre is from lying on the rack, in metres."""
    c = objects["panel"].pos
    return float(np.linalg.norm(c - np.array([RACK_X, 0.0, RACK_Z])))


def score(objects) -> float:
    """0 if the panel bowed more than 4 mm at any step (it cracked), fell off the table, or was never lifted clear.
    Otherwise 1.0 if it lies still on the rack, let go, with a peak bow of 3 mm or less, 0.75 if it does after a peak
    bow between 3 and 4 mm, and 0.5 for any other end (still carried or held, or set down off the rack)."""
    if _violation(objects):
        return 0.0
    _remember(objects)
    if STATE["peak"] > BOW_LIMIT or not STATE["lifted"]:
        return 0.0
    if _let_go(objects):
        return 1.0 if STATE["peak"] <= BOW_CLEAN else 0.75
    return 0.5


# -- scripted demo -----------------------------------------------------------------------------------------------------
# Both arms, straight down. Each hand pinches the panel across its width at one point (y = +-GRIP_Y), about a fifth of
# the panel's length in from its end, and the two carry it forward and up onto the rack together, so it stays flat.
# Numbers are TCP positions. A quarter turn closes the fingers along x, across the narrow panel: the left arm turns
# one way, the right arm the other, the way each wrist reaches.
GRIP_Y = 0.17  # m from the panel's middle: 13 cm in from each end
GRIP_Z_OFF = 0.0009  # the TCP this high above the rail: level with the panel's middle, so the pads' tips reach 1.3 cm under it
TRAVEL_Z = TOP + 0.16
TURN = math.pi / 2


def demo():
    w = Waypoint
    grab = START_TOP + GRIP_Z_OFF  # the TCP's height over the bench rail
    put = RACK_TOP + GRIP_Z_OFF  # and over the rack's rail
    out = {}
    for side, s, yaw in (("left", 1.0, -TURN), ("right", -1.0, TURN)):
        y = s * GRIP_Y
        out[side] = [
            w(1.5, (START_X, y, TRAVEL_Z), FINGER_OPEN, "over the panel", yaw),
            w(1.5, (START_X, y, grab), FINGER_OPEN, "down beside it", yaw),
            w(0.8, (START_X, y, grab), FINGER_CLOSED, "close", yaw),
            w(0.5, (START_X, y, grab), FINGER_CLOSED, "settle", yaw),
            w(1.8, (START_X, y, TRAVEL_Z), FINGER_CLOSED, "lift", yaw),
            w(2.2, (RACK_X, y, TRAVEL_Z), FINGER_CLOSED, "carry", yaw),
            w(1.8, (RACK_X, y, put), FINGER_CLOSED, "lower", yaw),
            w(0.6, (RACK_X, y, put), FINGER_OPEN, "release", yaw),
            w(1.2, (RACK_X, y, TRAVEL_Z), FINGER_OPEN, "retreat", yaw),
        ]
    return out
