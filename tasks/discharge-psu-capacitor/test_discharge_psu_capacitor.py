"""Offline checks (no GPU): the task file loads, the layout leaves the board room to slide, the electrical model's
numbers hold, the scorer covers each outcome, real contact (a pin's tip on a pad, a finger on the board) is what counts,
the contact time and the episode's memory work, pulling the board first or touching it with a finger scores 0, a board
held still in the box does not score 1.0, and the demo discharges before it pulls and stays within reach. Run in the
yam_arena environment: python -m pytest -q test_discharge_psu_capacitor.py"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
from yam_arena.demo import ArmPlan
from yam_arena.station import BASES
from yam_arena.task_file import ObjectState, load

HERE = pathlib.Path(__file__).resolve().parent
TASK = load(HERE / "discharge_psu_capacitor.py")
T = TASK.module
DT = 0.05  # s: one control step at 20 Hz
STEPS_FULL = round(T.DISCHARGE_S / DT)  # 330 steps: five time constants

FLAT = np.array([0.0, 0.0, 0.0, 1.0])
DOWN = np.array([0.0, -math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)])  # a quarter turn about y: +x up, pins down


def _state(pos, quat=FLAT, vel=(0.0, 0.0, 0.0)):
    pos = np.asarray(pos, float)
    return ObjectState(pos=pos, quat_xyzw=np.asarray(quat, float), lin_vel=np.asarray(vel, float),
                       ang_vel=np.zeros(3), low=pos - 0.02, high=pos + 0.02)


BOARD0 = np.asarray(T.BOARD.pos)
CAP0 = np.asarray(T.CAPACITOR.pos)
SINK0 = np.asarray(T.HEATSINK.pos)
TOOL0 = np.asarray(T.TOOL.pos)


def _scene(board=None, tool=None, capacitor=None, heatsink=None):
    out = {"board": _state(BOARD0), "capacitor": _state(CAP0), "heatsink": _state(SINK0), "tool": _state(TOOL0)}
    for name, value in (("board", board), ("tool", tool), ("capacitor", capacitor), ("heatsink", heatsink)):
        if value is not None:
            out[name] = value if isinstance(value, ObjectState) else _state(value)
    return out


def _tool_on_pads(offset=(0.0, 0.0, 0.0), board=BOARD0):
    """The tool hanging pins down, its pins' tips on the pads (shifted by ``offset``), the board at ``board``."""
    pads_mid = np.asarray(board) + (T.PADS_XY[0] - T.BOARD_XY[0], T.PADS_XY[1] - T.BOARD_XY[1], T.BOARD_SIZE[2] / 2)
    centre = pads_mid + (0.0, 0.0, T.TOOL_SIZE[0] / 2) + np.asarray(offset)
    return _state(centre, DOWN)


def _discharge(steps=STEPS_FULL, offset=(0.0, 0.0, 0.0), fingers=None):
    for _ in range(steps):
        T.observe(_scene(tool=_tool_on_pads(offset)), fingers, DT)


def _board_at(dx=0.0, dy=0.0, dz=0.0, vel=(0.0, 0.0, 0.0)):
    return _state(BOARD0 + (dx, dy, dz), vel=vel)


IN_BOX = (*T.BOX_XY, T.BOX_FLOOR_TOP + T.BOARD_SIZE[2] / 2)
PULLED = BOARD0 + (-0.12, 0.0, 0.0)  # off the rails, 1.6 cm above the table in front of the case


def _fingers_on(name, gap=0.0):
    """Two finger points ``gap`` above the top of a part (the board's drawn parts reach 1.8 cm above it), and one far off."""
    top = {"board": T.BOARD_TOP + T.PARTS_H, "capacitor": T.CAP_TOP, "heatsink": T.CAP_TOP}[name]
    xy = {"board": (T.BOARD_XY[0] - 0.045, T.BOARD_XY[1] + 0.05), "capacitor": T.CAP_XY, "heatsink": T.SINK_XY}[name]
    return np.array([[*xy, top + gap], [0.0, 0.0, T.TOP + 0.5]])


def _quat_about_y(deg):
    h = math.radians(deg) / 2
    return np.array([0.0, math.sin(h), 0.0, math.cos(h)])


@pytest.fixture(autouse=True)
def fresh_episode():
    T.reset_memory()


# -- the scene -------------------------------------------------------------------------------------------------------


def test_loads():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {"board", "capacitor", "heatsink", "tool"}


def test_parts_ride_the_board():
    for part in (T.CAPACITOR, T.HEATSINK):
        assert part.to == "board" and math.isinf(part.hold) and math.isinf(part.twist)
        assert part.pos[2] - part.size[2] / 2 == pytest.approx(T.BOARD_TOP)
        for i in range(2):  # on the board, inside its edges
            assert abs(part.pos[i] - T.BOARD.pos[i]) + part.size[i] / 2 < T.BOARD_SIZE[i] / 2
    mass = T.BOARD.mass + T.CAPACITOR.mass + T.HEATSINK.mass
    assert 2.5 < mass * 9.81 * 1.0 < 3.5  # about 3 N of sliding friction (rubber-like 1.0) on the rails


def test_rails_leave_play():
    assert T.SIDE_PLAY >= 0.002 - 1e-9 and T.LIP_PLAY >= 0.002 - 1e-9  # 2 mm or more each side, so a turned grip doesn't wedge it
    lip_inner = T.INNER_HALF_Y - T.RAIL_W
    assert T.BOARD_SIZE[1] / 2 - lip_inner >= 0.005  # the lips hold the board's edges down by 5 mm or more
    for part in (T.CAPACITOR, T.HEATSINK):
        assert abs(part.pos[1] - T.CASE_Y) + part.size[1] / 2 < lip_inner - 0.005  # clear of the lips
    assert T.BOARD.pos[0] - T.BOARD_SIZE[0] / 2 == pytest.approx(T.RAIL_X0)
    assert T.RAIL_LEN == pytest.approx(0.10)  # 10 cm of travel before the board leaves the rails
    assert T.CAP_TOP < T.CASE_TOP and T.HEATSINK.pos[2] + T.SINK_SIZE[2] / 2 < T.CASE_TOP


def test_arms_work_in_their_own_halves():
    assert T.TOOL.pos[1] < -0.2 and T.CAP_XY[1] < 0  # the tool and the capacitor on the right
    assert T.SINK_XY[1] > 0 and T.BOX_XY[1] > 0  # the heatsink and the box on the left
    box_near = T.BOX_XY[1] - T.BOX_OUTER[1] / 2
    assert box_near > T.CASE_Y + T.CASE_SIZE[1] / 2  # the pulled board's path is clear of the box
    tool_far = T.TOOL.pos[1] + T.TOOL_SIZE[1] / 2
    assert tool_far < T.CASE_Y - T.CASE_SIZE[1] / 2 - 0.1


def test_pads_are_test_points_on_the_board_between_the_parts():
    """The terminals are two flush test pads on the board's top face, between the capacitor and the heatsink and 1 cm or
    more clear of each, round the tool's 1.5 cm end, so the tool rests on the board and not on a part."""
    p = T.pads(_scene())
    assert np.allclose(p[:, 2], T.BOARD_TOP) and np.allclose(p[:, 0], T.PADS_XY[0])
    assert p[1, 1] - p[0, 1] == pytest.approx(T.PAD_GAP) and T.PAD_D == 0.008
    half = T.TOOL_SIZE[1] / 2
    lo, hi = T.PADS_XY[1] - half, T.PADS_XY[1] + half  # the tool's end face along y
    assert lo - (T.CAP_XY[1] + T.CAP_SIZE[1] / 2) >= 0.01
    assert (T.SINK_XY[1] - T.SINK_SIZE[1] / 2) - hi >= 0.01
    assert abs(T.PADS_XY[0] - T.BOARD_XY[0]) + half < T.BOARD_SIZE[0] / 2  # on the board


def test_pins_and_pads_line_up():
    tips = T.pin_tips(_scene(tool=_tool_on_pads()))
    assert np.allclose(tips, T.pads(_scene()), atol=1e-9)  # turned pins down, both tips on both pads
    flat = T.pin_tips(_scene())
    assert np.allclose(flat[:, 0], T.TOOL_XY[0] - T.TOOL_SIZE[0] / 2)  # lying, the pins point toward the robot


# -- the electrical model --------------------------------------------------------------------------------------------


def test_electrical_model():
    assert T.CAP_F == pytest.approx(330e-6) and T.V_START == 400.0 and T.TOOL_OHM == 10e3
    assert T.TAU_S == pytest.approx(3.3)
    assert T.DISCHARGE_S == pytest.approx(16.5) and T.DISCHARGE_S == pytest.approx(5 * T.TAU_S)
    assert T.pad_voltage(0.0) == 400.0
    assert T.pad_voltage(T.DISCHARGE_S) == pytest.approx(2.695, abs=0.005)  # 400 V * e^-5
    assert T.pad_voltage(T.DISCHARGE_S) < T.V_SAFE
    assert T.pad_voltage(2.0) > 200  # the old 2 s hold left more than half the charge
    assert T.pad_voltage(4.4 * T.TAU_S) < T.V_SAFE < T.pad_voltage(4.3 * T.TAU_S)  # 5 V is reached at 4.38 time constants
    assert T.V_START ** 2 / T.TOOL_OHM <= 25.0  # peak power inside the tool resistor's rating
    assert 0.5 * T.CAP_F * T.V_START ** 2 == pytest.approx(26.4, abs=0.1)  # joules into the resistor


# -- scoring ---------------------------------------------------------------------------------------------------------


def test_start_scores_zero():
    T.observe(_scene(), None, DT)
    assert T.score(_scene()) == 0.0 and not T.success(_scene())


def test_five_time_constants_discharge():
    _discharge(STEPS_FULL - 1)
    assert not T.STATE["discharged"] and T.score(_scene()) == 0.0
    _discharge(1)
    assert T.STATE["discharged"]
    assert T.score(_scene()) == 0.25  # discharged, the board still in the case


def test_the_old_two_second_hold_is_not_enough():
    _discharge(round(2.0 / DT))
    assert not T.STATE["discharged"] and T.score(_scene(board=IN_BOX)) == 0.0


def test_contact_time_adds_up_over_breaks():
    """The capacitor keeps its charge while the pins are off, so the time counts in all, not in one go."""
    _discharge(STEPS_FULL // 2)
    T.observe(_scene(), None, DT)  # the pins off the pads for a step
    assert T.STATE["contact_s"] == pytest.approx(STEPS_FULL // 2 * DT)
    _discharge(STEPS_FULL - STEPS_FULL // 2)
    assert T.STATE["discharged"]


def test_discharge_is_remembered():
    _discharge()
    T.observe(_scene(), None, DT)  # the tool taken away
    assert T.STATE["discharged"] and T.score(_scene()) == 0.25


@pytest.mark.parametrize("offset", [(0.0, 0.0, 0.0), (0.0029, 0.0, 0.0), (0.0, -0.0029, 0.0), (0.002, 0.002, 0.0),
                                    (0.0, 0.0, 0.0014), (0.0, 0.0, -0.0014), (0.002, 0.0, 0.0014)])
def test_a_tip_on_the_pad_touches(offset):
    _discharge(offset=offset)
    assert T.STATE["discharged"]


@pytest.mark.parametrize("offset", [(0.0031, 0.0, 0.0), (0.0, 0.0031, 0.0), (0.0, 0.0, 0.0016), (0.0, 0.0, -0.0016),
                                    (0.0, 0.0, 0.004), (0.003, 0.003, 0.0), (0.004, 0.0, 0.0), (0.0, 0.01, 0.0)])
def test_a_tip_off_the_pad_or_above_it_does_not(offset):
    """Within 4 mm of the pad's centre used to count: now the tip must lie on the pad and touch its plane."""
    _discharge(offset=offset)
    assert not T.STATE["discharged"] and T.STATE["contact_s"] == 0.0


def test_pins_hovering_above_the_board_do_not_touch():
    _discharge(offset=(0.0, 0.0, 0.003))  # 3 mm up, the tool held above the pads with its tips over them
    assert T.STATE["contact_s"] == 0.0 and not T.STATE["discharged"]


def test_tool_either_way_round():
    turned = _tool_on_pads()
    q = DOWN  # the same, turned a half turn about the vertical: the pins swap pads
    half = np.array([0.0, 0.0, 1.0, 0.0])
    x1, y1, z1, w1 = half
    x2, y2, z2, w2 = q
    swapped = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
    for _ in range(STEPS_FULL):
        T.observe(_scene(tool=_state(turned.pos, swapped)), None, DT)
    assert T.STATE["discharged"]


def test_one_pin_is_not_enough():
    one = _tool_on_pads((0.0, T.PAD_GAP / 2, 0.0))  # both pins shifted half a gap: one between the pads, one off
    for _ in range(STEPS_FULL + 5):
        T.observe(_scene(tool=one), None, DT)
    assert not T.STATE["discharged"]


def test_pads_follow_the_board():
    """The pads sit on the board, not the capacitor: moved with the board (after the discharge starts the board must not
    move 2 cm, so test the geometry directly)."""
    moved = BOARD0 + (-0.015, 0.0, 0.0)
    scene = _scene(board=moved, tool=_tool_on_pads(board=moved))
    assert T.pins_on_pads(scene)
    assert not T.pins_on_pads(_scene(board=moved, tool=_tool_on_pads()))


def test_pulled_out_after_discharge_half():
    _discharge()
    assert T.score(_scene(board=PULLED)) == 0.5
    lifted = BOARD0 + (0.0, 0.0, T.CASE_SIZE[2])  # held up above the walls
    assert T.score(_scene(board=lifted)) == 0.5


def test_partly_out_is_still_in_the_case():
    _discharge()
    assert T.score(_scene(board=BOARD0 + (-0.06, 0.0, 0.0))) == 0.25  # on its rails still


def test_released_still_in_the_box_succeeds():
    _discharge()
    objects = _scene(board=IN_BOX)
    T.observe(objects, np.array([[0.3, 0.3, T.TOP + 0.3]]), DT)  # the fingers well away
    assert T.success(objects) and T.score(objects) == 1.0


def test_in_box_still_moving_half():
    _discharge()
    objects = _scene(board=_state(IN_BOX, vel=(0.05, 0.0, 0.0)))
    assert not T.success(objects) and T.score(objects) == 0.5


def test_held_still_in_the_box_air_is_not_full():
    """The named exploit: the board held motionless over the box (inside its walls, below their top), never put down."""
    _discharge()
    for height in (0.02, 0.05, 0.08):
        held = _scene(board=(*T.BOX_XY, T.BOX_FLOOR_TOP + T.BOARD_SIZE[2] / 2 + height))
        assert not T.success(held) and T.score(held) == 0.5, height


def test_held_in_fingers_on_the_floor_is_not_full():
    """Set down on the floor but still gripped (a finger on the heatsink): not released."""
    _discharge()
    objects = _scene(board=IN_BOX)
    T.observe(objects, _fingers_on("heatsink"), DT)
    assert T.STATE["held"]
    assert not T.success(objects) and T.score(objects) == 0.5
    T.observe(objects, np.array([[0.3, 0.3, T.TOP + 0.3]]), DT)  # let go
    assert T.success(objects) and T.score(objects) == 1.0


def test_leaning_in_the_box_is_not_full():
    _discharge()
    tilted = _state((*T.BOX_XY, T.BOX_FLOOR_TOP + 0.03), _quat_about_y(25.0))
    assert T.score(_scene(board=tilted)) == 0.5
    level = _state((*T.BOX_XY, T.BOX_FLOOR_TOP + T.BOARD_SIZE[2] / 2 + 0.001), _quat_about_y(8.0))
    assert T.score(_scene(board=level)) == 1.0


def test_on_the_box_edge_is_not_in():
    _discharge()
    edge = (T.BOX_XY[0] + T.BOX_OUTER[0] / 2, T.BOX_XY[1], T.BOX_FLOOR_TOP)
    assert T.score(_scene(board=edge)) == 0.5


def test_pull_first_scores_zero_for_good():
    """The dangerous shortcut: the board comes out 2 cm before the capacitor is discharged."""
    T.observe(_scene(board=_board_at(dx=-0.021)), None, DT)
    assert T.STATE["failed"]
    _discharge()
    assert not T.STATE["discharged"]
    for board in (_board_at(dx=-0.021), _state(PULLED), _state(IN_BOX)):
        objects = _scene(board=board)
        assert T.score(objects) == 0.0 and not T.success(objects)
    T.reset_memory()
    _discharge()
    assert T.score(_scene(board=IN_BOX)) == 1.0


def test_score_alone_catches_the_pull():
    assert T.score(_scene(board=_state(PULLED))) == 0.0  # even without a step hook in between
    assert T.STATE["failed"]


def test_small_shift_before_discharge_is_allowed():
    T.observe(_scene(board=_board_at(dx=-0.015)), None, DT)
    assert not T.STATE["failed"]
    _discharge()
    assert T.STATE["discharged"]


@pytest.mark.parametrize("name", ["board", "capacitor", "heatsink"])
def test_finger_on_an_exposed_part_before_the_discharge_scores_zero(name):
    """A finger touching the board's copper (or its drawn parts), the capacitor or the heatsink, before the discharge."""
    T.observe(_scene(), _fingers_on(name, 0.0), DT)
    assert T.STATE["failed"]
    _discharge()
    assert T.score(_scene(board=IN_BOX)) == 0.0 and not T.STATE["discharged"]


@pytest.mark.parametrize("name", ["board", "capacitor", "heatsink"])
def test_finger_clear_of_the_part_is_fine(name):
    _discharge(fingers=_fingers_on(name, 0.004))  # 4 mm clear, all along
    assert not T.STATE["failed"] and T.STATE["discharged"]


def test_finger_on_a_test_pad_scores_zero():
    pad = T.pads(_scene())[0]
    T.observe(_scene(), np.array([pad + (0.0, 0.0, 0.0015), pad + (0.0, 0.05, 0.05)]), DT)
    assert T.STATE["failed"]


def test_finger_inside_the_drawn_parts_scores_zero():
    """The transformer and the other drawn parts have no collider: a finger in their volume counts as touching them."""
    inside = np.array([[T.BOARD_XY[0] - 0.03, T.BOARD_XY[1] - 0.035, T.BOARD_TOP + 0.010]])
    T.observe(_scene(), inside, DT)
    assert T.STATE["failed"]


def test_finger_after_discharge_is_fine():
    _discharge()
    T.observe(_scene(), _fingers_on("board"), DT)
    assert not T.STATE["failed"] and T.STATE["held"]


def test_board_on_the_floor_zero():
    _discharge()
    assert T.score(_scene(board=(0.2, 0.0, T.TOP - 0.5))) == 0.0


def test_non_finite_zero():
    _discharge()
    assert T.score(_scene(board=(math.nan, 0.0, 0.0))) == 0.0


# -- the demo --------------------------------------------------------------------------------------------------------


def _at(steps, label):
    t = 0.0
    for w in steps:
        t += w.duration
        if w.label == label:
            return t, w
    raise KeyError(label)


def test_demo_discharges_before_the_left_arm_comes():
    plan = T.demo()
    on, _ = _at(plan["right"], "pins on the pads")
    held, hold = _at(plan["right"], "hold")
    lifted, _ = _at(plan["right"], "lift off")
    assert held - on >= T.DISCHARGE_S + 1.5 and hold.duration == T.HOLD_S  # the hold outlasts the 16.5 s of contact
    leave = _at(plan["left"], "wait for the discharge")[0]
    assert leave >= lifted - 0.5
    pull, _ = _at(plan["left"], "pull the board out")
    assert pull > held


def test_demo_pins_land_on_the_pads():
    """Where the demo sends the TCP, the turned tool's pins meet the pads (before the PRESS and the tuned AIM)."""
    plan = T.demo()
    _, poke = _at(plan["right"], "pins on the pads")
    tips_mid = np.asarray(poke.pos) + (T.ROD_AHEAD, 0.0, -T.PIN_BELOW)
    expect = np.array([*T.PADS_XY, T.BOARD_TOP]) + (T.AIM[0], T.AIM[1], -T.PRESS)
    assert np.allclose(tips_mid, expect, atol=1e-9)
    assert poke.pitch == pytest.approx(math.pi / 2)
    _, over = _at(plan["right"], "over the pads")
    assert over.pos[2] - T.PIN_BELOW > T.CASE_TOP + 0.02  # the hanging pins clear the case's walls


def test_demo_grips_the_tool_on_its_rod():
    _, close = _at(T.demo()["right"], "close")
    assert 0 < close.pos[0] - T.TOOL_XY[0] <= T.TOOL_SIZE[0] / 2 - 0.03  # the rod, past the middle, 3 cm in from its end and close.pos[1] == T.TOOL_XY[1]
    assert close.pos[2] - T.PAD_TIP_PAST_TCP > T.TOP  # the pad tips above the table


def test_demo_fingers_stay_off_the_pads():
    """The left arm's open fingers stay 1 cm or more from the capacitor's box when they come down round the
    heatsink, and the right arm's TCP stays 10 cm above the pads."""
    plan = T.demo()
    _, down = _at(plan["left"], "down round it")
    finger_y = down.pos[1] - abs(T.SINK_OPEN) - 0.01  # the inner pad face, and about 1 cm of finger
    assert finger_y - (T.CAP_XY[1] + T.CAP_SIZE[1] / 2) > 0.005
    _, poke = _at(plan["right"], "pins on the pads")
    assert poke.pos[2] - T.CAP_TOP > 0.10
    _, over = _at(plan["right"], "over the pads")
    assert over.pos[2] - T.PIN_BELOW - T.CAP_TOP > 0.02  # the hanging pins clear the capacitor and the heatsink on the way


def test_demo_places_the_board_in_the_box():
    plan = T.demo()
    _, lower = _at(plan["left"], "lower")
    board_xy = np.asarray(lower.pos[:2]) + T.TO_BOARD
    assert np.allclose(board_xy, T.BOX_XY)
    _, pull = _at(plan["left"], "pull the board out")
    back_edge = T.BOARD.pos[0] - T.PULL + T.BOARD_SIZE[0] / 2
    assert back_edge < T.CASE_FRONT  # off the rails and out of the case
    _, carry = _at(plan["left"], "carry")
    assert carry.pos[2] - T.BOARD_BELOW_TCP > T.TOP + T.BOX_OUTER[2] + 0.02  # clears the box's walls


def test_demo_keeps_the_arms_apart():
    """The right arm stays in the right half and the left in the left half. While the right arm is at the capacitor,
    the left waits 15 cm or more away."""
    plan = T.demo()
    assert max(w.pos[1] for w in plan["right"]) < 0.02  # the right arm reaches 2 cm past the midline, to the test pads
    assert min(w.pos[1] for w in plan["left"]) > 0.0
    _, poke = _at(plan["right"], "pins on the pads")
    stage = np.asarray(T.STAGE)
    assert np.linalg.norm(stage[:2] - np.asarray(poke.pos[:2])) > 0.15


@pytest.mark.parametrize("arm", ["left", "right"])
def test_demo_reachable(arm):
    plan = ArmPlan(T.demo()[arm], BASES[arm])
    for k in range(int(plan.duration * 20) + 1):
        plan.action7(k / 20)
    assert plan.worst_ik_mm < 2.0 and plan.worst_ik_deg < 2.0, (arm, plan.worst_ik_mm, plan.worst_ik_deg)


# -- the meshes --------------------------------------------------------------------------------------------------------


def test_every_dressed_box_exists_and_its_mesh_is_there():
    names = {o.name for o in T.OBJECTS}
    for name, file in T.LOOKS.items():
        assert name in names
        assert (HERE / "looks" / file).is_file()


@pytest.mark.parametrize("name", ["board", "capacitor", "tool"])
def test_scored_meshes_fit_in_their_boxes(name):
    """The scored parts' meshes stay inside their colliders, so their centres are their boxes' centres."""
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(HERE / "looks" / T.LOOKS[name]))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    half = np.asarray({"board": T.BOARD_SIZE, "capacitor": T.CAP_SIZE, "tool": T.TOOL_SIZE}[name]) / 2
    assert np.all(np.asarray(box.GetMax()) <= half + 1e-5)
    assert np.all(np.asarray(box.GetMin()) >= -half - 1e-5)


def test_finger_cloud_from_the_urdf():
    import fingers

    for tip in fingers.TIPS:
        pts = fingers.cloud(tip)
        assert 200 < len(pts) < 20000
        span = pts.max(0) - pts.min(0)
        assert 0.05 < span.max() < 0.15  # a finger, in metres
