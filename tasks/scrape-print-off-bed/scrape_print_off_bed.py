"""On the station table, centred between the two arms: an enclosed desktop 3D printer (36 x 35 x 40 cm, fixed) with
its clear front door swung fully open to the left (180 degrees). Inside, low and just behind the door opening, its
black heated bed carries a removable gold spring-steel build plate (24 x 24 cm, 1 mm, 250 g) held by the bed's
magnets. A green printed part (6 x 6 x 3 cm, 40 g) is stuck to the plate's front right. Its bottom edge is chamfered:
it stands on a 3.6 x 3.6 cm foot 2 mm tall, so a blade slid along the plate gets under its edge. A print scraper (a
10 x 4 cm steel blade, 1 mm thick, on a 12 cm black handle, 80 g) lies on the table right of the printer, its blade
toward the printer's front. A blue plastic basket (15 x 20 x 8.6 cm) stands on the table front left.

The task: use the scraper to free the print from the plate, put the print in the basket, leave the plate on the
printer and the scraper on the table.

The print is stuck to the plate by a bond that holds 60 N. The plate is held to the bed by a bond that gives at 5 N
(its magnets), so a hand pulling the print straight up lifts the plate. The scraper's push is less than that: the
magnets hold the plate while it scrapes, and one hand can do the whole task (the demo uses the left arm only to carry
the print to the basket on the left). Once the blade has been under the print's edge (3 mm or more in) for 0.5 s,
its top has met the edge (7 mm in at 8 degrees) and levered it up, the adhesion peels and the print's bond is
released. ``bonds.py`` makes both bonds, ``colliders.py`` the print's foot, the scraper's thin
blade and the printer's walls, ``printer.py`` holds the printer's geometry.

Run it:  inspect-robots run --embodiment yam_arena -E task=scrape_print_off_bed.py --scorer task_score ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import printer as pr  # noqa: E402
from bonds import Bonded, on_reset, on_step, release  # noqa: E402
from colliders import Part, shape, shapes  # noqa: E402
from looks import dress  # noqa: E402
from simready import SimReady  # noqa: E402
from wrist import TiltWaypoint  # noqa: E402
from yam_arena.demo import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0


def _at(x, y, h):
    """A station-frame position from a height above the table."""
    return (x, y, TOP + h)


# -- the printer: fixed, its colliders the base, bed, walls, roof, front frame and gantry ------------------------------
PRINTER = Box("printer", pos=_at(*pr.CENTRE), size=pr.SIZE, color=(0.12, 0.12, 0.13))
shape("printer", pr.printer_parts())
# The door swung open 180 degrees about its hinge: it stands flat beside the printer's left side, its handle away.
_dx, _dy = pr.DOOR_CENTRE[0] - pr.HINGE[0], pr.DOOR_CENTRE[1] - pr.HINGE[1]
DOOR_OPEN_XY = (pr.HINGE[0] - _dx, pr.HINGE[1] - _dy)
DOOR = Box("door", pos=_at(*DOOR_OPEN_XY, pr.DOOR_CENTRE[2]), size=pr.DOOR_SIZE, yaw=math.pi, color=(0.2, 0.22, 0.24))
shape("door", pr.door_parts())
HEAD = Box("print_head", pos=_at(*pr.HEAD_CENTRE), size=pr.HEAD_SIZE, color=(0.78, 0.79, 0.8))

# -- the build plate on the bed --------------------------------------------------------------------------------------
PLATE_SIZE = pr.PLATE_SIZE
PLATE_XY = pr.PLATE_XY
BED_TOP = TOP + pr.BED_TOP
PLATE_TOP = BED_TOP + PLATE_SIZE[2]
PLATE_HOLD = 5.0  # N: the magnets' grip on the sheet. Less than the print's bond, more than the scraper's push
PLATE_FRICTION = 0.3  # spring steel on the magnetic sheet
PLATE = Bonded("plate", pos=(*PLATE_XY, BED_TOP + PLATE_SIZE[2] / 2), size=PLATE_SIZE, mass=0.25,
               color=(0.72, 0.55, 0.22), to="printer", hold=PLATE_HOLD)
shape("plate", [Part("sheet", (0.0, 0.0, 0.0), PLATE_SIZE, (PLATE_FRICTION, PLATE_FRICTION), "min")])

# -- the print: a 6 x 6 x 3 cm block on a 3.6 x 3.6 cm foot 2 mm tall (its chamfered first layers) ----------------------
PRINT_SIZE = (0.06, 0.06, 0.03)
PRINT_XY = (0.45, -0.06)
PRINT_START_Z = PLATE_TOP + PRINT_SIZE[2] / 2
PRINT_TOP = PLATE_TOP + PRINT_SIZE[2]
FOOT = (0.036, 0.036, 0.002)  # the gap under the print's edge: 2 mm tall, 12 mm deep
PRINT_HOLD = 60.0  # N: a pull or push this hard breaks the print's bond. It does not give to a twist
PRINT = Bonded("print_part", pos=(*PRINT_XY, PRINT_START_Z), size=PRINT_SIZE, mass=0.04, color=(0.35, 0.85, 0.15),
               to="plate", hold=PRINT_HOLD)
_h = PRINT_SIZE[2] / 2
shape("print_part", [
    Part("foot", (0.0, 0.0, -_h + FOOT[2] / 2), FOOT),
    Part("body", (0.0, 0.0, FOOT[2] / 2), (PRINT_SIZE[0], PRINT_SIZE[1], PRINT_SIZE[2] - FOOT[2])),
])
PRINT_EDGE_X = PRINT_XY[0] - PRINT_SIZE[0] / 2  # the print's front edge, where the blade comes from

# -- the scraper: lying on the table along x, its blade toward the printer ------------------------------------------
# Its own frame: y along it (blade toward +y), z up. Turned -90 degrees, its +y end points along world +x.
SCRAPER_SIZE = (0.04, 0.22, 0.022)  # blade width, overall length, handle height
HANDLE = (0.028, 0.12, 0.022)
BLADE = (0.04, 0.10, 0.001)
BLADE_FRICTION = 0.15  # polished steel on PLA and on the plate's coating
_sl, _sh = SCRAPER_SIZE[1] / 2, SCRAPER_SIZE[2] / 2
HANDLE_Y = -_sl + HANDLE[1] / 2  # the handle's centre along the scraper (-5 cm)
TIP = (0.0, _sl, -_sh)  # the middle of the blade's front edge, at its bottom
SCRAPER_YAW = -math.pi / 2
SCRAPER_XY = (0.22, -0.27)
SCRAPER = Box("scraper", pos=(*SCRAPER_XY, TOP + _sh), size=SCRAPER_SIZE, mass=0.08, yaw=SCRAPER_YAW,
              color=(0.08, 0.08, 0.09))
shape("scraper", [
    Part("handle", (0.0, HANDLE_Y, 0.0), HANDLE),
    Part("blade", (0.0, _sl - BLADE[1] / 2, -_sh + BLADE[2] / 2), BLADE, (BLADE_FRICTION, BLADE_FRICTION), "min"),
])

# -- the basket: a SimReady plastic basket, fixed, front left ------------------------------------------------------------
BASKET_XY = (0.22, 0.28)
BASKET_SCALE = 0.75  # the 20.1 x 27.1 x 11.4 cm model to 15.1 x 20.3 x 8.6 cm
BASKET_SIZE = (0.151, 0.203, 0.086)
BASKET = SimReady("basket", pos=(*BASKET_XY, TOP), size=BASKET_SIZE, package="misc_basket_plastic_a01",
                  usd="sm_misc_basket_plastic_a01_01.usd", scale=BASKET_SCALE)
BASKET_RIM = TOP + BASKET_SIZE[2]
IN_BASKET_XY = (0.055, 0.08)  # m: the print's centre this close to the basket's centre is over its floor

OBJECTS = [PRINTER, DOOR, HEAD, PLATE, PRINT, SCRAPER, BASKET]

# What the boxes look like: meshes modelled in Blender (looks/make_looks.py), each inside its box, the colliders
# unchanged. The basket is the SimReady model itself.
LOOKS = {"printer": "printer.usdc", "door": "door.usdc", "print_head": "head.usdc", "plate": "plate.usdc",
         "print_part": "print.usdc", "scraper": "scraper.usdc"}
dress(LOOKS)

# -- scoring ---------------------------------------------------------------------------------------------------------
PLATE_MOVED = 0.005  # m: the plate's centre this far from its start has moved
PLATE_LIFTED = 0.02  # m: the plate's highest point this far above its start has been lifted off the bed
FREED = 0.05  # m: the print's centre this far from its place on the plate is off it
UNDER_GAP = 0.004  # m: the blade's tip within this of the print's bottom, 3 mm or more inside its outline, is
UNDER_DEPTH = 0.003  # under its edge (the blade's top meets the edge 7 mm in at 8 degrees)
WEDGE_S = 0.5  # s the tip must stay under the print's edge. Then the adhesion peels: the print's bond is released
ON_TABLE_Z = 0.005  # m: the scraper's lowest point within this of the table top lies on the table
OFF_TABLE_Z = -0.02  # m above the table: an object whose centre is lower has fallen off the table

# What the episode remembers: the blade's tip has been under the print's edge for 0.5 s, and the plate has been
# lifted 2 cm (which scores 0 for the rest of the trial).
STATE = {"under_s": 0.0, "wedged": False, "plate_lifted": False}


@on_reset
def reset_memory(env=None) -> None:
    STATE.update(under_s=0.0, wedged=False, plate_lifted=False)


def _rot(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


NAMES = ("plate", "print_part", "scraper")


def _finite(objects) -> bool:
    return all(np.all(np.isfinite(objects[n].pos)) and np.all(np.isfinite(objects[n].quat_xyzw)) for n in NAMES)


def blade_tips(scraper_pos, scraper_quat) -> np.ndarray:
    """The blade's front edge (both corners and the middle, at its bottom) in the world, from the scraper's body
    centre and orientation."""
    r = _rot(scraper_quat)
    edge = [np.array([s * BLADE[0] / 2, TIP[1], TIP[2]]) for s in (-1.0, 0.0, 1.0)]
    return np.array([np.asarray(scraper_pos) + r @ p for p in edge])


def tip_under_print(tips, print_pos, print_quat) -> bool:
    """A point of the blade's front edge is 3 mm or more inside the print's outline and within 4 mm of its bottom:
    under its edge, since the blade can't get into the foot."""
    r = _rot(print_quat).T
    inside = (PRINT_SIZE[0] / 2 - UNDER_DEPTH, PRINT_SIZE[1] / 2 - UNDER_DEPTH)
    for p in tips:
        q = r @ (np.asarray(p) - np.asarray(print_pos))
        if abs(q[0]) < inside[0] and abs(q[1]) < inside[1] and q[2] < -PRINT_SIZE[2] / 2 + UNDER_GAP:
            return True
    return False


def plate_moved(objects) -> bool:
    return bool(np.linalg.norm(objects["plate"].pos - np.asarray(PLATE.pos)) >= PLATE_MOVED)


def plate_lifted(objects) -> bool:
    return bool(objects["plate"].high[2] >= PLATE_TOP + PLATE_LIFTED)


def print_freed(objects) -> bool:
    """The print's centre is 5 cm or more from its place on the plate."""
    offset = objects["print_part"].pos - objects["plate"].pos
    start = np.asarray(PRINT.pos) - np.asarray(PLATE.pos)
    return bool(np.linalg.norm(offset - start) >= FREED)


def print_in_basket(objects) -> bool:
    """The print's centre is over the basket's floor (within 5.5 x 8 cm of its centre) and below its rim."""
    c = objects["print_part"].pos
    return bool(abs(c[0] - BASKET_XY[0]) < IN_BASKET_XY[0] and abs(c[1] - BASKET_XY[1]) < IN_BASKET_XY[1]
                and c[2] < BASKET_RIM)


def _over(xy, centre, size, margin=0.0) -> bool:
    return bool(abs(xy[0] - centre[0]) < size[0] / 2 + margin and abs(xy[1] - centre[1]) < size[1] / 2 + margin)


def scraper_on_table(objects) -> bool:
    """The scraper lies still on the table top: its lowest point within 5 mm of it, and its centre over neither the
    printer nor the basket."""
    s = objects["scraper"]
    return bool(s.low[2] < TOP + ON_TABLE_Z and not pr.inside_footprint(s.pos[0], s.pos[1])
                and not _over(s.pos, BASKET_XY, BASKET_SIZE) and s.resting())


def _violation(objects) -> bool:
    if not _finite(objects):
        return True
    return any(objects[n].pos[2] < TOP + OFF_TABLE_Z for n in ("print_part", "scraper"))


def _remember(objects) -> None:
    if plate_lifted(objects):
        STATE["plate_lifted"] = True


@on_step
def watch_the_blade(env) -> None:
    """Every control step: is the blade's tip under the print's edge, and for how long in a row. After 0.5 s the blade
    has levered the edge up and the print's bond is released for the rest of the episode."""
    from yam_arena.scene import object_states

    base = env.unwrapped
    world, origin = base.arena_world, base.scene.env_origins[0]
    pose = {n: world.get_pose_w(n)[0].detach().cpu().numpy().astype(float) for n in ("scraper", "print_part")}
    o = origin.detach().cpu().numpy().astype(float)
    if not all(np.all(np.isfinite(p)) for p in pose.values()):
        return
    dt = float(getattr(base, "step_dt", 0.05))
    tips = blade_tips(pose["scraper"][:3] - o, pose["scraper"][3:])
    if tip_under_print(tips, pose["print_part"][:3] - o, pose["print_part"][3:]):
        STATE["under_s"] += dt
        if STATE["under_s"] >= WEDGE_S - 1e-6:
            STATE["wedged"] = True
    else:
        STATE["under_s"] = 0.0
    if STATE["wedged"]:
        release("print_part")
    objects = object_states(env, ["plate"])
    if np.all(np.isfinite(objects["plate"].high)):
        _remember(objects)


def success(objects) -> bool:
    """The print lies still in the basket, the blade got under its edge, the plate never left the bed and is within
    5 mm of where it was, and the scraper lies on the table."""
    if _violation(objects):
        return False
    _remember(objects)
    return bool(not STATE["plate_lifted"] and STATE["wedged"] and print_in_basket(objects)
                and objects["print_part"].resting() and not plate_moved(objects) and scraper_on_table(objects))


def score(objects) -> float:
    """1.0: success. 0.5: the print is off its place on the plate (5 cm or more), but not all of success holds (not
    in the basket, the plate moved 5 mm or more, the scraper not on the table, or freed without the blade). 0.25: the
    blade's tip was under the print's edge for 0.5 s. 0 otherwise, and 0 for the rest of the trial once the plate was
    lifted 2 cm off the bed, or if the print or the scraper fell off the table."""
    if _violation(objects):
        return 0.0
    _remember(objects)
    if STATE["plate_lifted"]:
        return 0.0
    if success(objects):
        return 1.0
    if print_freed(objects):
        return 0.5
    return 0.25 if STATE["wedged"] else 0.0


# -- scripted demo ---------------------------------------------------------------------------------------------------
# Right arm: picks the scraper by the middle of its handle (straight down, fingers closing across it, along y), lifts
# it, carries it in front of the printer and tilts it 8 degrees, blade tip down, sets the tip on the plate's front
# edge, slides it along the plate under the print's edge and pushes on (the arm is position controlled, so the
# command runs past the print's edge and the arm leans on the blade), holds, pulls back out of the printer and lays
# the scraper back where it was. Left arm: waits in front of the door's left half, out of the right arm's way, then
# takes the freed print from the front (fingers either side of it), carries it out, lowers it into the basket and
# lets go 2 cm over its floor. The plate's magnets hold it against the scraper's push: no hand holds it down. Numbers are TCP positions.
TILT = math.radians(8)  # the blade's angle to the plate (the arm's wrist reaches no steeper here)
PITCH = -TILT  # the down grip tilted back about the fingers' axis (y): the scraper's +x end (the blade) goes down
HANDLE_GRIP = -(HANDLE[0] / 2 - 0.004)  # finger target: 4 mm past each side of the handle, a 20 N squeeze
PRINT_GRIP = -(PRINT_SIZE[1] / 2 - 0.004)
GRIP_UP = 0.006  # the TCP holds the handle 6 mm above its middle, so the pad tips stay 4 mm above the table
PICK = (SCRAPER_XY[0] + HANDLE_Y, SCRAPER_XY[1], TOP + _sh + GRIP_UP)  # the handle's middle (5 cm behind centre)
HOVER_Z = TOP + 0.15
SET_Z = PICK[2] + 0.002  # laid back down 2 mm above where it was picked
OUT_X = 0.20  # the TCP this far forward keeps the held scraper's tip out of the printer (its front is at x 0.31)


def tip_from_tcp(tilt: float = TILT) -> np.ndarray:
    """Where the blade's tip is from the TCP when the scraper is held tilted ``tilt`` (tip down, toward +x)."""
    d = np.array([TIP[1] - HANDLE_Y, TIP[2] - GRIP_UP])  # along the scraper, up through it
    c, s = math.cos(tilt), math.sin(tilt)
    return np.array([d[0] * c + d[1] * s, 0.0, -d[0] * s + d[1] * c])


def tcp_for_tip(tip) -> tuple[float, float, float]:
    """The TCP that puts the tilted blade's tip at ``tip``."""
    return tuple(float(v) for v in np.asarray(tip) - tip_from_tcp())


TIP_PRESS = 0.002  # m: the tip is sent this far below the plate's top, so it stays on it
TIP_START_X = pr.BED_X[0] + 0.008  # the tip set down 8 mm inside the plate's front edge
PUSH_PAST = 0.022  # m: the tip is sent this far past the print's edge (it meets the edge 3.6 mm in, the foot 12 mm in)
TIP_PUSH_X = PRINT_EDGE_X + PUSH_PAST

WAIT = (0.28, 0.10, TOP + 0.22)  # the left hand waiting, open, in front of the door's left half
PICK_PITCH = math.radians(75)  # the print taken from the front, fingers either side of it (along y), the hand
# tilted down 15 degrees so its palm clears the bed's front edge
PRINT_PICK_Z = PRINT_START_Z + 0.004
PRINT_PICK = (PRINT_XY[0], PRINT_XY[1], PRINT_PICK_Z)
PRINT_APPROACH = (0.33, PRINT_XY[1], PRINT_PICK_Z + 0.01)
LIFT = 0.025  # m: the print lifted off the plate before it is drawn out
DROP_PITCH = math.radians(60)
DROP = (BASKET_XY[0] + 0.03, BASKET_XY[1], TOP + 0.052)  # the print's lowest corner 2 cm over the basket's floor,
# the hand 3 cm toward the far side so its wrist clears the near rim


def demo():
    def r(duration, pos, finger, label, pitch=0.0):
        return TiltWaypoint(duration, tuple(pos), finger, label, 0.0, pitch)

    px, py, pz = PICK
    y = PRINT_XY[1]
    start = tcp_for_tip((TIP_START_X, y, PLATE_TOP + 0.01))
    down = tcp_for_tip((TIP_START_X, y, PLATE_TOP - TIP_PRESS))
    push = tcp_for_tip((TIP_PUSH_X, y, PLATE_TOP - TIP_PRESS))
    back = tcp_for_tip((TIP_START_X, y, PLATE_TOP + 0.01))
    outside = (OUT_X, y, start[2] + 0.01)
    right = [
        r(1.5, (px, py, HOVER_Z), FINGER_OPEN, "over the handle"),
        r(1.2, PICK, FINGER_OPEN, "down to the handle"),
        r(0.8, PICK, HANDLE_GRIP, "close on the handle"),
        r(1.0, (px, py, HOVER_Z), HANDLE_GRIP, "lift"),
        r(1.5, outside, HANDLE_GRIP, "in front of the printer", PITCH),
        r(1.5, start, HANDLE_GRIP, "blade into the printer", PITCH),
        r(1.0, down, HANDLE_GRIP, "tip on the plate", PITCH),
        r(2.5, push, HANDLE_GRIP, "slide under its edge", PITCH),
        r(1.2, push, HANDLE_GRIP, "lean on it", PITCH),
        r(1.0, back, HANDLE_GRIP, "pull back", PITCH),
        r(1.2, outside, HANDLE_GRIP, "out of the printer", PITCH),
        r(1.5, (px, py, HOVER_Z), HANDLE_GRIP, "back over its place"),
        r(1.0, (px, py, SET_Z), HANDLE_GRIP, "lay it down"),
        r(0.6, (px, py, SET_Z), FINGER_OPEN, "let go"),
        r(1.0, (px, py, HOVER_Z), FINGER_OPEN, "up"),
    ]
    pulled = sum(w.duration for w in right[:11])  # the blade is out of the printer by then

    def w(duration, pos, finger, label, pitch):
        return TiltWaypoint(duration, tuple(pos), finger, label, 0.0, pitch)

    left = [
        w(2.5, WAIT, FINGER_OPEN, "in front of the door", PICK_PITCH),
        w(pulled - 2.5, WAIT, FINGER_OPEN, "wait for the print", PICK_PITCH),
        w(1.5, PRINT_APPROACH, FINGER_OPEN, "in front of the print", PICK_PITCH),
        w(1.2, PRINT_PICK, FINGER_OPEN, "fingers either side", PICK_PITCH),
        w(0.8, PRINT_PICK, PRINT_GRIP, "close", PICK_PITCH),
        w(0.8, (PRINT_XY[0], y, PRINT_PICK_Z + LIFT), PRINT_GRIP, "lift", PICK_PITCH),
        w(1.5, (0.30, y, PRINT_PICK_Z + LIFT + 0.02), PRINT_GRIP, "out of the printer", PICK_PITCH),
        w(2.0, (DROP[0], DROP[1], DROP[2] + 0.12), PRINT_GRIP, "over the basket", DROP_PITCH),
        w(1.5, DROP, PRINT_GRIP, "lower into it", DROP_PITCH),
        w(0.6, DROP, FINGER_OPEN, "let go", DROP_PITCH),
        w(1.2, (DROP[0], DROP[1], DROP[2] + 0.12), FINGER_OPEN, "up", DROP_PITCH),
    ]
    return {"right": right, "left": left}
