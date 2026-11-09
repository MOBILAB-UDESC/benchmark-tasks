"""On the station table, in front of the robot: a spring-loaded desoldering pump standing in its holster on the right
(a 5 x 5 cm dark stand, 2.2 cm tall, with the pump's 5 mm nozzle through a 1.2 cm hole), and a green circuit board
(10 x 8 cm) fixed on the table between the arms, with three silver solder joints in a row 2 cm apart (7 mm across,
2.5 mm tall). The middle joint sits on a red square sticker: it is the one to clean.

The pump is a blue 2.4 cm barrel, 12 cm tall, with a black plunger knob (3 cm across) 2.6 cm above its top and a white
thumb button on its left side, 1.5 cm below the top. Pushing the knob down 25 mm against a 6 N spring
makes a latch catch it, which is the pump armed. Pressing the button 2.5 mm releases the latch and the plunger flies
up: a shot, and its recoil jolts the pump. The solder isn't simulated: the shot is the visible stroke of the plunger.

The task: arm the pump, rest its nozzle on the marked joint with the pump upright, fire it there, then stand the pump
back in its holster. One arm has to press the knob, lift the pump out and hold it on the joint. The other arm has to
press the button, which is 8 cm above where the first one holds it.

``bonds.py`` joins the pump's parts, makes the sprung plunger and button and the latch, ``looks.py`` draws the parts
(modelled in Blender, ``looks/make_looks.py``), and ``wrist.py`` adds the sideways grip the demo holds the pump with.

Run it:  inspect-robots run --embodiment yam_arena -E task=desoldering_pump_fire.py --scorer task_score ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from bonds import Bonded, Sliding, lock_slide, on_reset, on_step, unlock_slide  # noqa: E402
from looks import dress  # noqa: E402
from wrist import TiltWaypoint  # noqa: E402
from yam_arena.constants import PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, Cylinder  # noqa: E402
from yam_arena.urdf import quat_xyzw_to_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0

# -- the holster: four dark blocks round a 1.2 cm hole, 2.2 cm tall, on the robot's right ----------------------------------
HOLSTER_XY = (0.27, -0.15)
RING_OUT = 0.05  # m: the holster's side
HOLE = 0.012  # m: the hole's side
RING_H = 0.022
DARK = (0.13, 0.13, 0.14)
_hx, _hy = HOLSTER_XY
_w = (RING_OUT - HOLE) / 2  # m: a block's width
_c = (RING_OUT + HOLE) / 4  # m: a block's centre from the holster's
_rz = TOP + RING_H / 2
HOLSTER = [
    Box("holster_near", pos=(_hx - _c, _hy, _rz), size=(_w, RING_OUT, RING_H), color=DARK),
    Box("holster_far", pos=(_hx + _c, _hy, _rz), size=(_w, RING_OUT, RING_H), color=DARK),
    Box("holster_left", pos=(_hx, _hy + _c, _rz), size=(HOLE, _w, RING_H), color=DARK),
    Box("holster_right", pos=(_hx, _hy - _c, _rz), size=(HOLE, _w, RING_H), color=DARK),
]

# -- the pump: a barrel, the nozzle bonded under it, a sprung plunger knob above it and a sprung button on its side ------
NOZZLE_SIZE = (0.005, 0.005, 0.020)
NOZZLE_GAP = 0.002  # m: the nozzle's tip hangs this high above the table while the pump stands in its holster
BARREL_SIZE = (0.024, 0.024, 0.12)
BARREL_BOTTOM = TOP + RING_H  # the barrel stands on the holster's blocks, its nozzle through the hole
BARREL_TOP = BARREL_BOTTOM + BARREL_SIZE[2]
BARREL_POS = (_hx, _hy, BARREL_BOTTOM + BARREL_SIZE[2] / 2)
BLUE = (0.05, 0.22, 0.85)
STROKE = 0.025  # m: how far the plunger goes down to be latched
KNOB_H = 0.012  # m: the knob is 3 cm across and this thick, on a rod that goes down into the barrel
ROD_BELOW = 0.030  # m: the rod's collider reaches this far below the knob
KNOB_SIZE = (0.008, 0.008, KNOB_H + ROD_BELOW)  # the plunger's collider: the rod and the knob's thickness, 8 mm wide
KNOB_PLAY = 0.001  # m: the cocked plunger's rod ends this high above the barrel's top (the joint stops it)
SPRING = 240.0  # N/m: 6 N when cocked
KNOB_BOTTOM = BARREL_TOP + STROKE + KNOB_PLAY  # the knob's underside with the plunger out
KNOB_POS = (_hx, _hy, KNOB_BOTTOM + KNOB_H - KNOB_SIZE[2] / 2)
BUTTON_SIZE = (0.012, 0.008, 0.014)  # x, y (the way it is pressed), z
BUTTON_GAP = 0.004  # m: between the button and the barrel, which is its travel
BUTTON_TRAVEL = 0.004
BUTTON_RATE = 1500.0  # N/m: 6 N at the end of its travel
BUTTON_POS = (_hx, _hy + BARREL_SIZE[1] / 2 + BUTTON_GAP + BUTTON_SIZE[1] / 2, BARREL_TOP - 0.015)
BUTTON_CAP_Y = BUTTON_POS[1] - _hy + BUTTON_SIZE[1] / 2  # the button's outer face from the barrel's axis: +0.024
BUTTON_ABOVE_BOTTOM = BUTTON_POS[2] - BARREL_BOTTOM  # the button's centre above the barrel's bottom: 0.105

NOZZLE = Bonded("nozzle", pos=(_hx, _hy, BARREL_BOTTOM - NOZZLE_SIZE[2] / 2), size=NOZZLE_SIZE, mass=0.005,
                color=(0.92, 0.92, 0.9), to="barrel", collide=False)
BARREL = Box("barrel", pos=BARREL_POS, size=BARREL_SIZE, mass=0.06, color=BLUE)
PLUNGER = Sliding("plunger", pos=KNOB_POS, size=KNOB_SIZE, mass=0.04, color=(0.03, 0.03, 0.035), to="barrel", axis="Z",
                  lower=-STROKE, upper=0.0, rate=SPRING, damping=1.5, collide=False)
BUTTON = Sliding("button", pos=BUTTON_POS, size=BUTTON_SIZE, mass=0.005, color=(0.93, 0.93, 0.9), to="barrel",
                 axis="Y", lower=-BUTTON_TRAVEL, upper=0.0, rate=BUTTON_RATE, damping=15.0)
PUMP_MASS = 0.06 + 0.04 + 0.005 + 0.005
KNOB_REST_ABOVE = KNOB_POS[2] - BARREL_POS[2]  # the knob's centre above the barrel's, with the plunger out
BUTTON_REST_BESIDE = BUTTON_POS[1] - BARREL_POS[1]  # the button's centre from the barrel's, along y

# -- the board, its three solder joints and the red sticker ------------------------------------------------------------
BOARD_XY = (0.31, 0.0)
BOARD_SIZE = (0.10, 0.08, 0.0016)
BOARD_TOP = TOP + BOARD_SIZE[2]
BOARD = Box("board", pos=(*BOARD_XY, TOP + BOARD_SIZE[2] / 2), size=BOARD_SIZE, color=(0.10, 0.36, 0.20))
JOINT_PITCH = 0.02  # m between the joints, along y
JOINT_R, JOINT_H = 0.0035, 0.0025
JOINT_TOP = BOARD_TOP + JOINT_H
SILVER = (0.78, 0.79, 0.82)
JOINTS = [Cylinder(f"joint_{n}", pos=(BOARD_XY[0], BOARD_XY[1] + s * JOINT_PITCH, BOARD_TOP + JOINT_H / 2), color=SILVER,
                   radius=JOINT_R, height=JOINT_H) for n, s in (("a", -1), ("b", 0), ("c", 1))]
TARGET = JOINTS[1]  # the joint on the red sticker
TARGET_TOP = np.array([TARGET.pos[0], TARGET.pos[1], JOINT_TOP])  # the point the nozzle's tip must reach
STICKER = Box("sticker", pos=(TARGET.pos[0], TARGET.pos[1], BOARD_TOP + 0.0002), size=(0.016, 0.016, 0.0004),
              color=(0.90, 0.05, 0.04))

OBJECTS = [*HOLSTER, BOARD, STICKER, *JOINTS, BARREL, NOZZLE, PLUNGER, BUTTON]

# What the boxes look like: meshes modelled in Blender (looks/make_looks.py), the colliders unchanged. The joints stay
# plain silver cylinders.
LOOKS = {"barrel": "barrel.usdc", "nozzle": "nozzle.usdc", "plunger": "plunger.usdc", "button": "button.usdc",
         "holster_near": "block_x.usdc", "holster_far": "block_x.usdc", "holster_left": "block_y.usdc",
         "holster_right": "block_y.usdc", "board": "board.usdc"}
dress(LOOKS)

# -- scoring ---------------------------------------------------------------------------------------------------------
GOOD_MM = 2.0  # mm: the nozzle's tip this close to the marked joint's top is on it
NEAR_MM = 6.0  # mm: this close is on the joint's pad, nearly
MAX_TILT = 15.0  # degrees: the pump this far from upright or less is held upright enough
LATCH_BAND = 0.001  # m: the knob within this of the bottom of its stroke is latched
FIRED_RISE = STROKE / 2  # m: after a shot the knob must come back up this far before it can be latched again
PRESSED = 0.0025  # m: the button pressed this far lets the latch go
SEAT_XY = 0.008  # m: the pump's barrel within this of the holster's axis, standing in it
SEAT_Z = 0.004  # m: and within this of the height it stands at
SEAT_TILT = 5.0  # degrees
OFF_TABLE_Z = -0.02  # m above the table: an object whose centre is lower has fallen off the table
SEAT_BARREL_Z = BARREL_POS[2]

# What the episode remembers. ``armed``: the latch is holding the plunger down (so the plunger can fire). ``shots``: how
# many times a shot was fired. ``armed_on_joint``: at some step the pump was armed with its tip resting on the marked
# joint. ``near_shot`` and ``good_shot``: a shot was fired with the tip within 6 mm of the joint, and within 2 mm of
# it with the pump within 15 degrees of upright. ``rearm_ok`` is False between a shot and the knob coming back up.
STATE = {"armed": False, "rearm_ok": True, "shots": 0, "armed_on_joint": False, "near_shot": False, "good_shot": False,
         "last_error_mm": None, "last_tilt": None}


@on_reset
def reset_memory(env=None) -> None:
    STATE.update(armed=False, rearm_ok=True, shots=0, armed_on_joint=False, near_shot=False, good_shot=False,
                 last_error_mm=None, last_tilt=None)


def _finite(objects) -> bool:
    return all(np.all(np.isfinite(s.pos)) and np.all(np.isfinite(s.quat_xyzw)) and np.all(np.isfinite(s.lin_vel))
               for s in objects.values())


def _rot(state) -> np.ndarray:
    return quat_xyzw_to_matrix(np.asarray(state.quat_xyzw, float))


def plunger_travel(objects) -> float:
    """How far the knob is from where it stands with the plunger out, along the barrel's axis (m, negative: down)."""
    rel = _rot(objects["barrel"]).T @ (np.asarray(objects["plunger"].pos) - np.asarray(objects["barrel"].pos))
    return float(rel[2] - KNOB_REST_ABOVE)


def button_travel(objects) -> float:
    """How far the button is pressed in, along the barrel's y axis (m)."""
    rel = _rot(objects["barrel"]).T @ (np.asarray(objects["button"].pos) - np.asarray(objects["barrel"].pos))
    return float(BUTTON_REST_BESIDE - rel[1])


def tip(objects) -> np.ndarray:
    """The centre of the nozzle's bottom face."""
    return np.asarray(objects["nozzle"].pos) + _rot(objects["nozzle"]) @ np.array([0.0, 0.0, -NOZZLE_SIZE[2] / 2])


def tip_error_mm(objects) -> float:
    """Distance from the nozzle's tip to the top of the marked joint, in mm."""
    return float(np.linalg.norm(tip(objects) - TARGET_TOP) * 1000)


def tilt_deg(objects) -> float:
    """How far the pump's axis is from vertical, in degrees."""
    return float(np.degrees(np.arccos(np.clip(_rot(objects["barrel"])[2, 2], -1.0, 1.0))))


def on_joint(objects) -> bool:
    """The tip resting on the marked joint: within 2 mm of its top, the pump within 15 degrees of upright."""
    return bool(tip_error_mm(objects) <= GOOD_MM and tilt_deg(objects) < MAX_TILT)


def in_holster(objects) -> bool:
    """The barrel standing in the holster: its centre within 8 mm of the holster's axis and 4 mm of its height, upright
    within 5 degrees."""
    x, y, z = objects["barrel"].pos
    return bool(math.hypot(x - HOLSTER_XY[0], y - HOLSTER_XY[1]) <= SEAT_XY and abs(z - SEAT_BARREL_Z) <= SEAT_Z
                and tilt_deg(objects) < SEAT_TILT)


def _fell(objects) -> bool:
    return bool(objects["barrel"].pos[2] < TOP + OFF_TABLE_Z)


def observe(objects) -> str | None:
    """One control step of the pump's mechanism and of the episode's memory. Returns "lock" when the latch has caught
    the plunger this step, "unlock" when the button has let it go, and None otherwise."""
    if not _finite(objects):
        return None
    travel = plunger_travel(objects)
    if STATE["rearm_ok"] is False and travel > -FIRED_RISE:
        STATE["rearm_ok"] = True
    if not STATE["armed"]:
        if STATE["rearm_ok"] and travel <= -(STROKE - LATCH_BAND):
            STATE["armed"] = True
            STATE["armed_on_joint"] = STATE["armed_on_joint"] or on_joint(objects)
            return "lock"
        return None
    if on_joint(objects):
        STATE["armed_on_joint"] = True
    if button_travel(objects) >= PRESSED - 1e-9:
        err, tilt = tip_error_mm(objects), tilt_deg(objects)
        STATE.update(armed=False, rearm_ok=False, shots=STATE["shots"] + 1, last_error_mm=err, last_tilt=tilt)
        if err <= NEAR_MM:
            STATE["near_shot"] = True
        if err <= GOOD_MM and tilt < MAX_TILT:
            STATE["good_shot"] = True
        return "unlock"
    return None


@on_step
def work_the_latch(env) -> None:
    """Every control step: where the plunger and the button are, into the latch and the episode's memory."""
    from yam_arena.scene import object_states

    objects = object_states(env, ["barrel", "nozzle", "plunger", "button"])
    action = observe(objects)
    if action == "lock":
        lock_slide("plunger", -STROKE)
    elif action == "unlock":
        unlock_slide("plunger")


def success(objects) -> bool:
    """A good shot was fired, and the pump stands still in its holster."""
    if not _finite(objects) or _fell(objects) or not STATE["good_shot"]:
        return False
    return bool(in_holster(objects) and objects["barrel"].resting() and objects["plunger"].resting())


def score(objects) -> float:
    """1.0: success. 0.75: a shot was fired with the tip within 2 mm of the marked joint's top and the pump within 15
    degrees of upright, but the pump isn't standing still in its holster. 0.5: a shot was fired with the tip within 6
    mm of the joint. 0.25: the pump was armed with its tip resting on the joint, and never fired well. 0 otherwise, and
    0 if the pump fell off the table."""
    if not _finite(objects) or _fell(objects):
        return 0.0
    if STATE["good_shot"]:
        return 1.0 if success(objects) else 0.75
    if STATE["near_shot"]:
        return 0.5
    return 0.25 if STATE["armed_on_joint"] else 0.0


# -- scripted demo ---------------------------------------------------------------------------------------------------
# Right arm: closed fingers pressing the knob straight down (past the latch), then a forward grip (the gripper points
# along +x, 20 degrees down, its fingers closing along y on the barrel's flat sides) 2 cm above the barrel's bottom,
# lift the pump out, carry it over the board, set its tip on the marked joint, hold it there while the left arm presses
# the button, then lift, carry it back, lower it into the holster, let go. Left arm: closed fingers pointing at the button
# (along -y, 20 degrees down), press the button and hold it, retreat. Numbers are TCP positions, from the layout only.
COCK = True  # False: the right arm skips the knob (ablation: the pump is never armed)
FIRE = True  # False: the left arm never presses the button (ablation)
JOINT_USED = 1  # which joint the nozzle is set on, 0 to 2 (1 is the marked one)
PRESS_DEPTH = 0.020  # m: the pad tips are sent this far below the cocked knob's top, so the arm leans on it
GRIP_H = 0.025  # m: the grip's centre above the barrel's bottom
PITCH = math.radians(70)  # the sideways grips point 20 degrees below horizontal
PRE = 0.10  # m: the right gripper's pre-grasp distance behind the pump
PRESS_PITCH = 0.0  # the knob press is straight down
TIP_BELOW_TCP = GRIP_H + NOZZLE_SIZE[2]  # the nozzle's tip below the right TCP while the pump is held
AIM = (0.0002, 0.0071)  # m: the tip's aim off the joint's centre (x, y), tuned in the simulation
AIM_BACK = (0.0033, 0.0086)  # m: the same for the holster's hole, tuned in the simulation
SET_PRESS = 0.003  # m: the right TCP is sent this far below where the tip meets the joint, so the pump rests on it
BTN_PRESS = 0.008  # m: the left pad tips are sent this far past the button's face (its travel is 4 mm)
HOVER_Z = TOP + 0.215  # the right gripper's height over the knob and while it turns
CARRY_TIP_Z = JOINT_TOP + 0.045  # the carried nozzle's tip clears the joints
PRESS_YAW = 0.0
GRIP_YAW = 0.0  # the right gripper points along +x, its fingers closing along y on the barrel's flat sides
BTN_YAW = -math.pi / 2  # the left gripper points along -y, at the button on the barrel's left side


def _approach(pitch: float) -> np.ndarray:
    """The unit vector the TCP points along, for a grip tilted ``pitch`` (yaw 0: pitch 90 is +x, pitch 0 is down)."""
    return np.array([math.sin(pitch), 0.0, -math.cos(pitch)])


def knob_top_cocked() -> float:
    return BARREL_TOP + KNOB_PLAY + KNOB_H


def joint_xy(index: int) -> tuple[float, float]:
    return (JOINTS[index].pos[0], JOINTS[index].pos[1])


def right_grip_z() -> float:
    """The right TCP's height while it holds the pump in the holster."""
    return BARREL_BOTTOM + GRIP_H


def right_tcp_for_tip(x: float, y: float, tip_z: float) -> tuple[float, float, float]:
    """The right TCP that holds the pump with its nozzle's tip at (x, y, tip_z)."""
    return (x, y, tip_z + TIP_BELOW_TCP)


def left_tcp_for_cap(x_pump: float, y_pump: float, z_btn: float, past: float = 0.0) -> tuple[float, float, float]:
    """The left TCP whose pad tips touch the button's outer face (``past`` further in), for a pump with its axis at
    (``x_pump``, ``y_pump``) and its button's centre at height ``z_btn``. The pad tips are PAD_TIP_PAST_TCP ahead of
    the TCP, which points along -y and 20 degrees down."""
    a = np.array([0.0, -math.sin(PITCH), -math.cos(PITCH)])
    tcp = np.array([x_pump, y_pump + BUTTON_CAP_Y - past, z_btn]) - PAD_TIP_PAST_TCP * a
    return (float(tcp[0]), float(tcp[1]), float(tcp[2]))


def demo():
    w, t = TiltWaypoint, TiltWaypoint
    hx, hy = HOLSTER_XY
    jx, jy = joint_xy(JOINT_USED)
    px, py = jx + AIM[0], jy + AIM[1]
    zg = right_grip_z()
    # the knob press: the pad tips touch the cocked knob's top PRESS_DEPTH before the end of the press
    a = _approach(PRESS_PITCH)
    press_tcp = np.array([hx, hy, knob_top_cocked() - PRESS_DEPTH]) - PAD_TIP_PAST_TCP * a
    hover_tcp = (hx - PAD_TIP_PAST_TCP * a[0], hy, HOVER_Z)
    press_z = float(press_tcp[2])
    press_x = float(press_tcp[0])
    set_tcp = right_tcp_for_tip(px, py, JOINT_TOP - SET_PRESS)
    carry_tcp = right_tcp_for_tip(px, py, CARRY_TIP_Z)
    lift_z = zg + 0.05
    right = [
        t(2.0, hover_tcp, FINGER_CLOSED, "over the knob", pitch=PRESS_PITCH, yaw=PRESS_YAW),
        t(2.5, (press_x, hy, press_z) if COCK else hover_tcp, FINGER_CLOSED, "press the knob", pitch=PRESS_PITCH,
          yaw=PRESS_YAW),
        t(1.0, (press_x, hy, press_z) if COCK else hover_tcp, FINGER_CLOSED, "hold it down", pitch=PRESS_PITCH,
          yaw=PRESS_YAW),
        t(1.0, hover_tcp, FINGER_CLOSED, "up", pitch=PRESS_PITCH, yaw=PRESS_YAW),
        w(2.5, (hx - PRE, hy, HOVER_Z - 0.01), FINGER_OPEN, "turn the grip forward", pitch=PITCH, yaw=GRIP_YAW),
        w(1.5, (hx - PRE, hy, zg), FINGER_OPEN, "down behind the pump", pitch=PITCH, yaw=GRIP_YAW),
        w(1.5, (hx, hy, zg), FINGER_OPEN, "round the barrel", pitch=PITCH, yaw=GRIP_YAW),
        w(0.8, (hx, hy, zg), FINGER_CLOSED, "close", pitch=PITCH, yaw=GRIP_YAW),
        w(1.2, (hx, hy, lift_z), FINGER_CLOSED, "lift it out", pitch=PITCH, yaw=GRIP_YAW),
        w(2.5, carry_tcp, FINGER_CLOSED, "carry it over the joint", pitch=PITCH, yaw=GRIP_YAW),
        w(2.0, set_tcp, FINGER_CLOSED, "tip on the joint", pitch=PITCH, yaw=GRIP_YAW),
        w(1.5, set_tcp, FINGER_CLOSED, "settle", pitch=PITCH, yaw=GRIP_YAW),
        w(HOLD_FOR_SHOT, set_tcp, FINGER_CLOSED, "hold while the button is pressed", pitch=PITCH, yaw=GRIP_YAW),
        w(1.2, carry_tcp, FINGER_CLOSED, "lift off", pitch=PITCH, yaw=GRIP_YAW),
        w(2.5, (hx + AIM_BACK[0], hy + AIM_BACK[1], lift_z), FINGER_CLOSED, "carry it back", pitch=PITCH, yaw=GRIP_YAW),
        w(1.5, (hx + AIM_BACK[0], hy + AIM_BACK[1], zg + 0.003), FINGER_CLOSED, "into the holster", pitch=PITCH, yaw=GRIP_YAW),
        w(0.8, (hx + AIM_BACK[0], hy + AIM_BACK[1], zg + 0.003), FINGER_OPEN, "let go", pitch=PITCH, yaw=GRIP_YAW),
        w(1.5, (hx - PRE, hy, zg + 0.003), FINGER_OPEN, "back off", pitch=PITCH, yaw=GRIP_YAW),
        w(1.5, (hx - PRE, hy, HOVER_Z - 0.01), FINGER_OPEN, "up", pitch=PITCH, yaw=GRIP_YAW),
    ]
    # Left arm: it waits, comes up behind the button, touches it, presses past its travel, holds, and backs off.
    z_btn = JOINT_TOP + NOZZLE_SIZE[2] + BUTTON_ABOVE_BOTTOM  # the button's centre with the tip on the joint
    contact = left_tcp_for_cap(px, py, z_btn)
    press = left_tcp_for_cap(px, py, z_btn, past=BUTTON_TRAVEL + BTN_PRESS)
    stage = (contact[0], contact[1] + 0.10, contact[2] + 0.02)
    back = (contact[0], contact[1] + 0.06, contact[2] + 0.02)
    left = [
        t(LEFT_WAIT, stage, FINGER_CLOSED, "wait", pitch=PITCH, yaw=BTN_YAW),
        t(2.0, stage, FINGER_CLOSED, "beside the button", pitch=PITCH, yaw=BTN_YAW),
        t(1.5, (contact[0], contact[1] + 0.01, contact[2]), FINGER_CLOSED, "up to the button", pitch=PITCH, yaw=BTN_YAW),
        t(1.5, press if FIRE else contact, FINGER_CLOSED, "press the button", pitch=PITCH, yaw=BTN_YAW),
        t(1.5, press if FIRE else contact, FINGER_CLOSED, "hold it", pitch=PITCH, yaw=BTN_YAW),
        t(1.2, back, FINGER_CLOSED, "back off", pitch=PITCH, yaw=BTN_YAW),
    ]
    return {"right": right, "left": left}


# The left arm's wait, and how long the right arm holds the pump on the joint, so that the left presses once the pump has
# settled and has backed off before the right lifts: the right arm's steps before the hold take 20.2 s, the left's
# approach 3.5 s, its press and hold 3 s, its back-off 1.2 s.
LEFT_WAIT = 17.2
HOLD_FOR_SHOT = 6.0
