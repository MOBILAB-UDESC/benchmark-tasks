"""On the station table, between the two arms: a dead desktop mainboard (24 x 30 cm, green) lies flat on a dark
anti-static mat, fixed, turned so its PCIe slot runs along y. The slot (two rails 9 cm long) at x = 0.28 holds a
graphics card (18 x 11 cm, 2 cm thick with its cooler, 350 g) standing upright, from y = -0.11 to y = 0.07. A small
orange latch (a 2 x 1.2 cm button standing 1.2 cm out of a black housing) sits on the board 5.5 cm before the slot's
right end, at y = -0.175, on the right arm's side. It locks the card in the slot until it is pressed down 3 mm. A
foam-lined tray (25 x 15 cm, 3 cm walls) stands on the robot's left, at (0.24, 0.26). The right arm presses the
latch and the left arm lifts the card, 15 cm apart, so the arms don't meet.

The task: press the latch and hold it, lift the graphics card straight up out of the slot and put it in the tray.

How the latch locks the card (the linkage, which ``bonds.py`` stands in for): the card's near end has a steel end plate
with a hole 9 mm tall, its lower edge 1.5 mm above the card's bottom. A 3 mm steel pin, axis 8 mm above the board,
passes through the hole, 5 mm above the hole's lower edge, so the card can rise 5 mm before the edge meets the pin.
The pin is on a 5 cm push rod in a covered groove along the slot's axis, and the rod's other end has a 45 degree
wedge under the orange button. Pressing the button 3 mm slides the rod 3 mm and pulls the pin 3 mm out of the hole
(2.5 mm engaged): the card is free while the button is down. Let go and the spring (500 N/m, 0 N at rest, 2 N at full
travel) pushes the pin back into the hole if the card is within 5 mm of seated. If the card is raised more than 5 mm,
the pin meets the plate's face instead and the card stays free.

In the simulation the lock is a bond that holds 60 N, between the card and the board. A step hook reads the button at
every control step: down 3 mm or more, the bond is off. Up, the bond is on again once the card is seated (within
1 mm), and it holds to 60 N. Pulled harder, or raised 5 mm or more with the button up after the pin was in the hole,
the pin is overloaded and the latch is broken: 0 for the rest of the trial, even if the card is then put in the
tray. So is a card pried 5 degrees or more in the slot. The slot's rails let the card out straight up only. To
score 1.0 the card must be out of both hands and lie still on the tray's floor.

Run it:  inspect-robots run --embodiment yam_arena -E task=latch_and_lift_graphics_card.py --scorer task_score ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from bonds import Bonded, Sliding, on_reset, on_step, relock, release  # noqa: E402
from looks import dress  # noqa: E402
from yam_arena.constants import PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import FINGER_CLOSED, FINGER_OPEN, Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0

# -- the layout -------------------------------------------------------------------------------------------------------
# Drawn in the board's own frame, (u, v): u along the slot, v across it. The board is turned a quarter turn onto the
# table, so the slot runs along y. The latch is on the right arm's side and the card reaches toward the left arm, so
# each arm works in its own half of the table.
TURN = math.pi / 2  # every board part's yaw
SLOT_LINE_X = 0.28  # m: the slot's line on the table (v = 0), close enough for the left arm to lift high
U_TO_Y = -0.37  # m: y = u + U_TO_Y


def on_table(u: float, v: float, z: float) -> tuple[float, float, float]:
    """A point of the board's frame on the table."""
    return (SLOT_LINE_X - v, u + U_TO_Y, z)


# -- the mat and the mainboard ---------------------------------------------------------------------------------------
MAT_UV = (0.33, -0.01)
MAT_SIZE = (0.34, 0.30, 0.003)  # u, v, thickness
MAT_TOP = TOP + MAT_SIZE[2]
MAT = Box("mat", pos=on_table(*MAT_UV, TOP + MAT_SIZE[2] / 2), size=MAT_SIZE, yaw=TURN, color=(0.16, 0.18, 0.24))

BOARD_UV = (0.33, -0.01)
BOARD_SIZE = (0.30, 0.24, 0.002)
BOARD_TOP = MAT_TOP + BOARD_SIZE[2]
BOARD = Box("mainboard", pos=on_table(*BOARD_UV, MAT_TOP + BOARD_SIZE[2] / 2), size=BOARD_SIZE, yaw=TURN,
            color=(0.10, 0.35, 0.18))

BLACK = (0.06, 0.06, 0.06)

# -- the slot: two rails, the card between them -----------------------------------------------------------------------
SLOT_V = 0.0
SLOT_U0, SLOT_LEN = 0.26, 0.09  # the slot's near end and length, along u
CARD_SIZE = (0.18, 0.02, 0.11)  # long, thick, tall
CARD_MASS = 0.35
RAIL_GAP = 0.002  # m of play on each side of the card: with 1 mm, a grip turned 1.3 degrees wedges it
RAIL_SIZE = (SLOT_LEN, 0.004, 0.012)
SLOT_DEPTH = RAIL_SIZE[2]
_rail_v = CARD_SIZE[1] / 2 + RAIL_GAP + RAIL_SIZE[1] / 2
RAILS = [
    Box(f"rail_{side}", pos=on_table(SLOT_U0 + SLOT_LEN / 2, SLOT_V + s * _rail_v, BOARD_TOP + RAIL_SIZE[2] / 2),
        size=RAIL_SIZE, yaw=TURN, color=BLACK)
    for side, s in (("left", 1), ("right", -1))
]

CARD_U = SLOT_U0 + CARD_SIZE[0] / 2  # the card's near end is at the slot's near end
CARD_START_Z = BOARD_TOP + CARD_SIZE[2] / 2
CARD_TOP = BOARD_TOP + CARD_SIZE[2]
LOCK_HOLD = 60.0  # N: the pull that breaks the latch and frees a locked card
CARD = Bonded(
    "card",
    pos=on_table(CARD_U, SLOT_V, CARD_START_Z),
    size=CARD_SIZE,
    yaw=TURN,
    mass=CARD_MASS,
    color=(0.13, 0.13, 0.15),
    to="mainboard",
    hold=LOCK_HOLD,
)

# -- the latch: a sprung button in a short housing just before the slot's near end --------------------------------
LATCH_U = 0.195  # 5.5 cm before the card, so the two grippers stay apart
LATCH_SIZE = (0.02, 0.012, 0.015)
LATCH_TRAVEL = 0.004  # m: how far it goes down
LATCH_RATE = 500.0  # N/m: 2 N at full travel
PRESSED = 0.003  # m down: the wedge has pulled the pin out of the card's hole (the pin is 2.5 mm into it)
PIN_ENGAGED = 0.0025  # m: how far the pin reaches into the hole in the card's end plate
# The card's end plate and the pin, heights above the card's bottom edge and the board (drawn nowhere, see the docstring)
PIN_AXIS, PIN_D = 0.008, 0.003  # m: the pin's axis above the board and its diameter, so its bottom is at 6.5 mm
HOLE_BOTTOM, HOLE_H = 0.0015, 0.009  # m: the hole's lower edge above the card's bottom edge, and its height
LATCH_BOTTOM = BOARD_TOP + LATCH_TRAVEL + 0.001  # it hangs in its housing, clear of the board at full travel
LATCH_TOP = LATCH_BOTTOM + LATCH_SIZE[2]
LATCH = Sliding(
    "latch",
    pos=on_table(LATCH_U, SLOT_V, LATCH_BOTTOM + LATCH_SIZE[2] / 2),
    size=LATCH_SIZE,
    yaw=TURN,
    mass=0.01,
    color=(0.95, 0.22, 0.01),
    to="mainboard",
    axis="Z",
    lower=-LATCH_TRAVEL,
    upper=0.0,
    rate=LATCH_RATE,
    damping=10.0,
)
HOUSING_H = 0.008  # m: four black walls round the latch, 2 mm thick, 1 mm from it, hiding the gap under it
HOUSING_T = 0.002
_hi = (LATCH_SIZE[0] / 2 + 0.001, LATCH_SIZE[1] / 2 + 0.001)  # the housing's inner half sizes
_hz = BOARD_TOP + HOUSING_H / 2
HOUSING = [
    Box(name, pos=on_table(u, v, _hz), size=(*size, HOUSING_H), yaw=TURN, color=BLACK)
    for name, u, v, size in (
        ("housing_near", LATCH_U - _hi[0] - HOUSING_T / 2, SLOT_V, (HOUSING_T, 2 * _hi[1] + 2 * HOUSING_T)),
        ("housing_far", LATCH_U + _hi[0] + HOUSING_T / 2, SLOT_V, (HOUSING_T, 2 * _hi[1] + 2 * HOUSING_T)),
        ("housing_left", LATCH_U, SLOT_V + _hi[1] + HOUSING_T / 2, (2 * _hi[0], HOUSING_T)),
        ("housing_right", LATCH_U, SLOT_V - _hi[1] - HOUSING_T / 2, (2 * _hi[0], HOUSING_T)),
    )
]

# -- the tray: 25 x 15 cm, 1 cm foam floor, 1 cm walls 3 cm high, on the left arm's side, not turned --------------
TRAY_XY = (0.24, 0.26)
TRAY_OUTER = (0.25, 0.15)
TRAY_WALL = 0.01
TRAY_WALL_H = 0.03
TRAY_FLOOR_T = 0.01
TRAY_FLOOR_Z = TOP + TRAY_FLOOR_T
FOAM = (0.25, 0.25, 0.27)
_tx, _ty = TRAY_XY
_hx, _hy = TRAY_OUTER[0] / 2 - TRAY_WALL / 2, TRAY_OUTER[1] / 2 - TRAY_WALL / 2
_wz = TOP + TRAY_WALL_H / 2
TRAY = [
    Box("tray_floor", pos=(_tx, _ty, TOP + TRAY_FLOOR_T / 2), size=(*TRAY_OUTER, TRAY_FLOOR_T), color=FOAM),
    Box("tray_wall_near", pos=(_tx - _hx, _ty, _wz), size=(TRAY_WALL, TRAY_OUTER[1], TRAY_WALL_H), color=FOAM),
    Box("tray_wall_far", pos=(_tx + _hx, _ty, _wz), size=(TRAY_WALL, TRAY_OUTER[1], TRAY_WALL_H), color=FOAM),
    Box("tray_wall_right", pos=(_tx, _ty - _hy, _wz), size=(TRAY_OUTER[0] - 2 * TRAY_WALL, TRAY_WALL, TRAY_WALL_H), color=FOAM),
    Box("tray_wall_left", pos=(_tx, _ty + _hy, _wz), size=(TRAY_OUTER[0] - 2 * TRAY_WALL, TRAY_WALL, TRAY_WALL_H), color=FOAM),
]

OBJECTS = [MAT, BOARD, *RAILS, *HOUSING, *TRAY, CARD, LATCH]

# What the boxes look like: meshes modelled in Blender (looks/make_looks.py), the boxes' colliders unchanged. The
# mainboard's mesh also draws the rails and the latch housing, and the tray's draws its walls.
LOOKS = {"card": "card.usdc", "latch": "latch.usdc", "mainboard": "mainboard.usdc", "mat": "mat.usdc",
         "tray_floor": "tray.usdc"}
dress(LOOKS, hide=[b.name for b in (*RAILS, *HOUSING, *TRAY[1:])])

# -- scoring ---------------------------------------------------------------------------------------------------------
LIFTED = 0.005  # m up from its start: the pin's play (pin bottom 6.5 mm, hole's lower edge 1.5 mm). A seated card
# raised this far with the button up has hit the pin
OUT_SHIFT = 0.03  # m sideways from its start: the card has left the slot
SEATED = 0.001  # m: a card this close to its start (up and sideways) is seated, and the released pin locks it again
PRY_DEG = 5.0  # degrees: a card tilted this far from upright, with its bottom still between the rails, pries the slot
IN_TRAY_XY = (0.09, 0.05)  # m: the card's centre within this of the tray's centre is over its 23 x 13 cm floor
ON_FLOOR = 0.003  # m: the card's four lowest corners this close to the tray floor's top are touching it
HOLD_REACH = 0.03  # m: a gripper whose TCP is this close to the card's box is on it
HOLD_GAP = 0.004  # m: pads at most this much wider apart than the card's thickness are closed on it, or on nothing
OFF_TABLE_Z = -0.02  # m above the table: an object whose centre is lower has fallen off the table

# What the episode remembers: whether the pin is in the hole (the button up and the card within 5 mm of seated at the
# last step), whether the latch is broken, and whether a gripper is on the card. The button is read at every step.
STATE = {"engaged": True, "broken": False, "held": False}


@on_reset
def reset_memory(env=None) -> None:
    STATE.update(engaged=True, broken=False, held=False)


def _finite(objects) -> bool:
    return all(np.all(np.isfinite(s.pos)) and np.all(np.isfinite(s.lin_vel)) for s in objects.values())


def matrix(q) -> np.ndarray:
    """The rotation matrix of a unit quaternion x y z w."""
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def latch_pressed(objects) -> bool:
    """The button is down 3 mm or more: the pin is out of the hole."""
    return bool(objects["latch"].pos[2] <= LATCH.pos[2] - PRESSED)


def card_lift(objects) -> float:
    """How far the card's centre is above its start."""
    return float(objects["card"].pos[2] - CARD_START_Z)


def card_moved(objects) -> bool:
    """The card has left where it was locked: up 5 mm or more, or 3 cm or more sideways."""
    d = objects["card"].pos - np.asarray(CARD.pos)
    return bool(d[2] >= LIFTED or np.hypot(d[0], d[1]) >= OUT_SHIFT)


def card_seated(objects) -> bool:
    """The card is within 1 mm of its start, up and sideways."""
    d = objects["card"].pos - np.asarray(CARD.pos)
    return bool(abs(d[2]) < SEATED and np.hypot(d[0], d[1]) < SEATED)


def out_of_slot(objects) -> bool:
    """The card's bottom is above the rails (up 1.2 cm or more), or it is 3 cm or more from the slot sideways."""
    d = objects["card"].pos - np.asarray(CARD.pos)
    return bool(d[2] >= SLOT_DEPTH or np.hypot(d[0], d[1]) >= OUT_SHIFT)


def card_tilt_deg(objects) -> float:
    """How far the card's height axis is from vertical, in degrees."""
    up = matrix(objects["card"].quat_xyzw)[2, 2]
    return float(math.degrees(math.acos(min(1.0, max(-1.0, up)))))


def slot_pried(objects) -> bool:
    """The card tilts 5 degrees or more from upright while its bottom is still between the rails."""
    return bool(not out_of_slot(objects) and card_tilt_deg(objects) >= PRY_DEG)


def hook_in_hole(objects) -> bool:
    """The pin is in the hole now: the button up and the card raised less than 5 mm."""
    return bool(not latch_pressed(objects) and not card_moved(objects))


def in_tray(objects) -> bool:
    """The card's centre is within 9 x 5 cm of the tray's centre."""
    c = objects["card"].pos
    return bool(abs(c[0] - TRAY_XY[0]) < IN_TRAY_XY[0] and abs(c[1] - TRAY_XY[1]) < IN_TRAY_XY[1])


def on_tray_floor(objects) -> bool:
    """The card is supported by the tray's floor: its four lowest corners within 3 mm of the floor's top, so it lies
    flat on a face (an edge, or an end, is a face too) and does not lean on a wall."""
    card = objects["card"]
    half = np.asarray(CARD_SIZE) / 2
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    heights = np.sort(card.pos[2] + corners @ matrix(card.quat_xyzw)[2])
    return bool(heights[3] - TRAY_FLOOR_Z <= ON_FLOOR and heights[0] >= TRAY_FLOOR_Z - ON_FLOOR)


def held_by(objects, grippers: dict) -> bool:
    """Whether a gripper is on the card: its TCP (the point between the pads) within 3 cm of the card's box and its
    pads at most 4 mm wider apart than the card's 2 cm thickness, which includes pads shut on nothing. A gripper
    opened wider than that, or away from the card, has let go. ``grippers`` maps each arm to (TCP, the pads' gap)."""
    card = objects["card"]
    rot = matrix(card.quat_xyzw)
    reach = np.asarray(CARD_SIZE) / 2 + HOLD_REACH
    for point, gap in grippers.values():
        local = rot.T @ (np.asarray(point, float) - card.pos)
        if bool(np.all(np.abs(local) <= reach)) and gap <= CARD_SIZE[1] + HOLD_GAP:
            return True
    return False


def _violation(objects) -> bool:
    if not _finite(objects):
        return True
    return bool(objects["card"].pos[2] < TOP + OFF_TABLE_Z)


def _remember(objects) -> None:
    """Read the button and the card at this step. The latch breaks if the pin, in the hole at the last step, has been
    forced up out of it with the button up, or if the slot is pried. Nothing about a press is remembered."""
    if STATE["engaged"] and card_moved(objects) and not latch_pressed(objects):
        STATE["broken"] = True
    if slot_pried(objects):
        STATE["broken"] = True
    STATE["engaged"] = hook_in_hole(objects)


def bond_command(objects) -> str:
    """What the card's bond does at this step, from the button now (never from an earlier press): "release" while it
    is down 3 mm or more, "relock" when it is up and the card is seated, "" otherwise (the card is raised and the pin
    is not in the hole, so there is nothing to lock)."""
    if latch_pressed(objects):
        return "release"
    return "relock" if card_seated(objects) else ""


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
    """Every control step: read the button, switch the card's bond with it, check for damage, and see whether a
    gripper is on the card."""
    from yam_arena.scene import object_states

    objects = object_states(env, ["card", "latch"])
    if not _finite(objects):
        return
    _remember(objects)
    STATE["held"] = held_by(objects, grippers(env))
    command = bond_command(objects)
    if command == "release":
        release("card")  # the pin is out of the hole: the card is free while the button is down
    elif command == "relock":
        relock("card")  # the button is up and the card is seated: the pin is back in the hole


def success(objects) -> bool:
    """The card lies still on the tray's floor, out of every gripper, and the latch is whole."""
    if _violation(objects):
        return False
    _remember(objects)
    return bool(
        not STATE["broken"] and not STATE["held"] and in_tray(objects) and on_tray_floor(objects)
        and objects["card"].resting()
    )


def score(objects) -> float:
    """1.0: success. 0.5: the card is out of the slot (held, held above the tray, or set down anywhere else, or
    still in a hand in the tray). 0.25: the card raised 5 mm or more (it can only have come with the button
    down), but it is still in the slot. 0 otherwise, and 0 for the rest of the trial once the pin was forced or the
    slot pried (the latch is broken), or if the card fell off the table."""
    if _violation(objects):
        return 0.0
    _remember(objects)
    if STATE["broken"]:
        return 0.0
    if not STATE["held"] and in_tray(objects) and on_tray_floor(objects) and objects["card"].resting():
        return 1.0
    if out_of_slot(objects):
        return 0.5
    return 0.25 if card_lift(objects) >= LIFTED else 0.0


# -- scripted demo ---------------------------------------------------------------------------------------------------
# Right arm, fingers closed, straight down: press the latch and hold it. Left arm, straight-down grip on the top 2 cm
# of the card above its centre: close once the latch is down, lift straight up out of the slot, turn the card a quarter
# turn on the way to the tray (it lies along y in the slot and along x in the tray), lower, let go. Numbers are TCP
# positions. A yaw of a quarter turn closes the fingers along x, across the turned card and latch: the left arm turns
# the other way (-TURN), the way its wrist reaches.
GRIP_U = CARD_U  # the left arm holds the card above its centre: off centre, it tilts and jams or slips
GRIP_XY = on_table(GRIP_U, SLOT_V, 0.0)[:2]
GRIP_DEPTH = 0.02  # m: the pad tips this far below the card's top edge
GRIP_Z = CARD_TOP - GRIP_DEPTH + PAD_TIP_PAST_TCP
TRAVEL_Z = TOP + 0.15  # the card's bottom clears the rails and the tray's walls
PLACE_Z = TRAY_FLOOR_Z + 0.003 + CARD_SIZE[2] - GRIP_DEPTH + PAD_TIP_PAST_TCP
PRESS_DEPTH = 0.012  # m: the pad tips are sent this far below the latch's top (its travel is 4 mm), so the
# arm leans on it: the arm is position controlled and needs the error to push
PRESS_Z = LATCH_TOP - PRESS_DEPTH + PAD_TIP_PAST_TCP
HOVER_Z = TOP + 0.14
PRESS_YAW = TURN  # the right gripper's turn over the latch
PRESS_HOLD_S = 4.0  # s: the right arm keeps the button down this long, past the end of the left arm's lift (3.3 s)


def demo():
    w = Waypoint
    lx, ly = LATCH.pos[:2]
    gx, gy = GRIP_XY
    tx, ty = TRAY_XY
    return {
        "right": [
            w(1.5, (lx, ly, HOVER_Z), FINGER_CLOSED, "over the latch", PRESS_YAW),
            w(1.5, (lx, ly, PRESS_Z), FINGER_CLOSED, "press", PRESS_YAW),
            w(PRESS_HOLD_S, (lx, ly, PRESS_Z), FINGER_CLOSED, "hold", PRESS_YAW),
            w(1.0, (lx, ly, HOVER_Z), FINGER_CLOSED, "retreat", PRESS_YAW),
        ],
        "left": [
            w(1.5, (gx, gy, TRAVEL_Z), FINGER_OPEN, "over the card", -TURN),
            w(1.5, (gx, gy, GRIP_Z), FINGER_OPEN, "down around its top edge", -TURN),
            w(0.8, (gx, gy, GRIP_Z), FINGER_CLOSED, "close", -TURN),
            w(1.0, (gx, gy, GRIP_Z), FINGER_CLOSED, "wait for the latch", -TURN),
            w(1.5, (gx, gy, TRAVEL_Z), FINGER_CLOSED, "lift", -TURN),
            w(2.0, (tx, ty, TRAVEL_Z), FINGER_CLOSED, "carry and turn"),
            w(1.2, (tx, ty, PLACE_Z), FINGER_CLOSED, "lower"),
            w(0.6, (tx, ty, PLACE_Z), FINGER_OPEN, "release"),
            w(1.0, (tx, ty, TRAVEL_Z), FINGER_OPEN, "retreat"),
        ],
    }
