"""Offline checks (no GPU): the task file loads, the layout (the lead on its break points, the nippers on their sliding
joint, the cup and mat, the arms in their halves), the clip rule (what counts as the nippers closed on the lead, flush
or high, and what does not), every scoring outcome, the offcut's snap and flight and the episode's memory, the
shortcuts (one hand, a high cut), and the demo's order and reach. Run in the yam_arena environment:
python -m pytest -q test_clip_lead_catch_offcut.py"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
from yam_arena.demo import ArmPlan
from yam_arena.station import BASES
from yam_arena.task_file import ObjectState, load
from yam_arena.urdf import axis_angle_matrix

HERE = pathlib.Path(__file__).resolve().parent
TASK = load(HERE / "clip_lead_catch_offcut.py")
T = TASK.module
MM = 0.001
DT = 0.05  # s: one control step at 20 Hz
Z = np.array([0.0, 0.0, 1.0])


def _quat_z(a: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(a / 2), math.cos(a / 2)])


def _matrix_quat(r: np.ndarray) -> np.ndarray:
    from yam_arena.ik import matrix_to_quat_xyzw

    return matrix_to_quat_xyzw(r)


def _state(pos, quat=(0.0, 0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0)):
    pos = np.asarray(pos, float)
    return ObjectState(pos=pos, quat_xyzw=np.asarray(quat, float), lin_vel=np.asarray(vel, float),
                       ang_vel=np.zeros(3), low=pos - 0.01, high=pos + 0.01)


LOW0, TOP0 = np.asarray(T.LEAD_LOW.pos), np.asarray(T.LEAD_TOP.pos)


def _nippers(origin=None, z=None, travel=0.0, tilt_deg=0.0):
    """The two halves with the cutter frame's origin at ``origin`` (x, y), its bottom plane at ``z``, the blade closed
    ``travel`` m (negative), and the nippers rolled ``tilt_deg`` about their own jaws' axis."""
    origin = T.CUT_O if origin is None else np.asarray(origin, float)
    z = T.Z_BT if z is None else z
    r = axis_angle_matrix(np.array([math.cos(T.AIM), math.sin(T.AIM), 0.0]), math.radians(tilt_deg)) \
        @ _rz(T.PHI)
    o = np.array([origin[0], origin[1], z])
    f = o + r @ np.array(T.C_F)
    m = f + r @ np.array([T.C_M[0] - T.C_F[0] + travel, 0.0, 0.0])
    q = _matrix_quat(r)
    return {"cutter": _state(f, q), "jaw": _state(m, q)}


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _scene(top=None, low=None, nip=None):
    out = {"lead_low": _state(LOW0), "lead_top": _state(TOP0), **(nip or _nippers(travel=0.0, z=T.MAT_TOP))}
    for name, value in (("lead_top", top), ("lead_low", low)):
        if value is not None:
            out[name] = value if isinstance(value, ObjectState) else _state(value)
    return out


CLOSED = -4 * MM  # the blade stopped on the lead: the face from 9 mm to 5 mm


def _clip(**kw):
    return _scene(nip=_nippers(travel=kw.pop("travel", CLOSED), **kw))


IN_CUP = (*T.CUP_XY, T.MAT_TOP + 0.01)
ON_MAT = (T.LEAD_XY[0], T.LEAD_XY[1] - 0.06, T.MAT_TOP + 0.001)  # beside the board, 6 cm from the lead
ON_BOARD = (*T.LEAD_XY, T.Z_BT + 0.001)


@pytest.fixture(autouse=True)
def fresh_episode():
    T.reset_memory()


def _after_clip(top):
    """Clip the lead flush, then let the offcut be at ``top``."""
    T.observe(_clip(), DT)
    assert T.STATE["cut"] == "flush"
    T.observe(_scene(top=top), DT)


# -- the scene -------------------------------------------------------------------------------------------------------


def test_loads():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {"lead_low", "lead_top", "cutter", "jaw"}


def test_lead_pieces_stack_on_the_break_points():
    bottom = lambda b: b.pos[2] - b.size[2] / 2  # noqa: E731
    top = lambda b: b.pos[2] + b.size[2] / 2  # noqa: E731
    assert top(T.LEAD_STUB) == pytest.approx(T.Z_BT + 0.8 * MM) and bottom(T.LEAD_STUB) == pytest.approx(T.Z_BT)
    assert bottom(T.LEAD_LOW) == pytest.approx(top(T.LEAD_STUB)) and top(T.LEAD_LOW) == pytest.approx(T.Z_BT + 4 * MM)
    assert bottom(T.LEAD_TOP) == pytest.approx(top(T.LEAD_LOW)) and top(T.LEAD_TOP) == pytest.approx(T.Z_BT + 14 * MM)
    assert T.LEAD_LOW.to == "" and T.LEAD_TOP.to == "lead_low"  # the chain: stub, low piece, top piece
    for piece in (T.LEAD_STUB, T.LEAD_LOW, T.LEAD_TOP):
        assert piece.pos[:2] == T.LEAD_XY and piece.size[:2] == (2 * MM, 2 * MM)
    assert math.isinf(T.LEAD_LOW.hold) and math.isinf(T.LEAD_TOP.hold)  # nothing but the nippers lets go


def test_nippers_are_two_halves_on_a_sliding_joint():
    assert T.JAW.to == "cutter" and T.JAW.axis == "X" and T.JAW.yaw == T.CUTTER.yaw
    assert T.JAW.lower == pytest.approx(-0.010) and T.JAW.upper == 0.0
    assert T.JAW.rate * T.JAW.target == pytest.approx(25.0)  # a 25 N detent holds the blade open against its stop
    assert T.JAW.target > T.JAW.upper
    handles = T.PLATE_M_X[1] - T.PLATE_F_X[0]
    assert handles <= 0.094  # the open grip's 94 mm
    assert handles - T.HOLD_SQUEEZE == pytest.approx(6 * MM)  # the hold squeezes 3 mm in on each plate
    # the hold's pad force: 5000 N/m (the finger's stiffness) over 3 mm is 15 N, under the 25 N detent
    assert 5000.0 * 0.003 < T.JAW.rate * T.JAW.target
    # the jaws: a 10 mm throat, the blade closing onto the anvil's face, the blade over the anvil with 0.3 mm between
    assert T.BLADE_X - T.ANVIL_X == pytest.approx(10 * MM)
    assert T.BLADE_BOTTOM - T.ANVIL_TOP == pytest.approx(0.3 * MM)
    assert T.TRAVEL >= T.BLADE_X - T.ANVIL_X - 1e-9
    # the plates never touch when the blade is closed all the way
    assert T.PLATE_M_X[0] - T.TRAVEL > T.ANVIL_X


def test_nippers_boxes_hold_their_parts():
    for half, parts in ((T.CUTTER, T.PARTS_F), (T.JAW, T.PARTS_M)):
        for _, centre, size in parts:
            for i in range(3):
                assert abs(centre[i]) + size[i] / 2 <= half.size[i] / 2 + 1e-9
    # the two boxes together are the whole nippers: the same length and height, side by side across the jaws
    assert T.CUTTER.size[1:] == pytest.approx(T.JAW.size[1:])
    assert T.CUTTER.mass + T.JAW.mass == pytest.approx(0.05)


def test_nippers_start_on_the_mat_clear_of_everything():
    for half in (T.CUTTER, T.JAW):
        assert half.pos[2] - half.size[2] / 2 == pytest.approx(T.MAT_TOP)
    corners = [T.START_O + T.rot((u, v)) for u in (T.PLATE_F_X[0], T.PLATE_M_X[1]) for v in (T.PLATE_Y[0], T.FOOT_Y[1])]
    for p in corners:
        assert all(abs(p[i] - T.MAT_XY[i]) < T.MAT_SIZE[i] / 2 for i in range(2))
        assert p[1] < T.LEAD_XY[1] - T.BOARD_SIZE[1] / 2 - 0.02  # right of the board and its rails


def test_board_cup_and_mat():
    assert T.BOARD.pos[2] + T.BOARD_SIZE[2] / 2 == pytest.approx(T.Z_BT)
    assert T.CAP.pos[2] + T.CAP_SIZE[2] / 2 < T.BOARD_BOTTOM  # the capacitor hangs under the board, clear of the mat
    assert T.CAP.pos[2] - T.CAP_SIZE[2] / 2 >= T.MAT_TOP
    # the cup stands on the mat beside the board, its far wall within 10 cm of the lead
    assert T.CUP.pos[2] == pytest.approx(T.MAT_TOP)
    near = T.CUP.pos[1] - T.CUP.size[1] / 2  # beside the board, clear of its edge
    assert near > T.LEAD_XY[1] + T.BOARD_SIZE[1] / 2
    assert np.hypot(T.CUP.pos[0] - T.LEAD_XY[0], T.CUP.pos[1] - T.LEAD_XY[1]) + T.CUP_IN_R < T.FLY_DIST
    assert T.CUP_RIM - T.MAT_TOP == pytest.approx(0.0609, abs=1e-3)
    assert T.CUP.package == "consumable_cup_paper_a01"


def test_the_offcut_is_inside_its_box_in_the_demo_and_the_lead_stands_14_mm():
    assert TOP0[2] + T.LEAD_TOP.size[2] / 2 - T.Z_BT == pytest.approx(0.014)


# -- the clip rule ---------------------------------------------------------------------------------------------------


def test_closed_on_the_lead_on_the_board_is_a_flush_cut():
    assert T.cut_class(_clip()) == "flush"


def test_nippers_resting_on_the_board_with_the_blade_open_do_nothing():
    assert T.cut_class(_clip(travel=0.0)) == ""
    assert T.cut_class(_clip(travel=-2 * MM)) == ""  # the blade moved 2 mm: short of 3 mm and of the lead


def test_blade_pushed_to_the_lead_by_the_arm_alone_is_not_a_clip():
    # the lead at 8 mm in the throat: the open blade (9 mm) touches it with no squeeze at all
    nip = _nippers(origin=T.CUT_O + T.rot((4 * MM, 0.0)), travel=0.0)
    assert T.cut_class(_scene(nip=nip)) == ""


def test_jaws_must_be_over_the_lead():
    assert T.cut_class(_clip(origin=T.CUT_O + T.rot((0.0, 30 * MM)))) == ""  # the lead behind the jaws
    assert T.cut_class(_clip(origin=T.CUT_O + T.rot((0.0, -30 * MM)))) == ""  # the lead ahead of them


def test_the_blade_must_be_on_the_lead_and_the_lead_past_the_anvil():
    # stopped 2 mm short of the lead's face: not on it
    assert T.cut_class(_clip(travel=-2 * MM)) == ""
    # the lead against the anvil's face: still clipped
    assert T.cut_class(_clip(origin=T.CUT_O + T.rot((4 * MM, 0.0)), travel=-8 * MM)) == "flush"
    # the lead on the wrong side of the anvil: no
    assert T.cut_class(_clip(origin=T.CUT_O + T.rot((-6 * MM, 0.0)), travel=-6 * MM)) == ""


def test_cutting_plane_height_makes_flush_or_high():
    assert T.cut_class(_clip(z=T.Z_BT + 0.5 * MM)) == "flush"  # 1.3 mm up: within 1.4
    assert T.cut_class(_clip(z=T.Z_BT + 1.0 * MM)) == "high"  # 1.8 mm
    assert T.cut_class(_clip(z=T.Z_BT + 6.0 * MM)) == "high"
    assert T.cut_class(_clip(z=T.Z_BT + 8.0 * MM)) == ""  # 8.8 mm: above the lead's break points
    assert T.cut_class(_clip(z=T.Z_BT - 0.1 * MM)) == "flush"


def test_nippers_must_be_level():
    assert T.cut_class(_clip(tilt_deg=8.0)) == "flush"
    assert T.cut_class(_clip(tilt_deg=15.0)) == ""


def test_a_clip_gives_the_offcut_a_snap_and_lets_go_the_right_joint():
    T.observe(_clip(), DT)
    names, v = T.STATE["kick"]
    assert names == ["lead_low", "lead_top"]  # flush: the low piece lets go of the stub, the top piece rides on it
    assert np.linalg.norm(v) == pytest.approx(T.KICK) and v[2] > 0
    along = np.array([math.cos(T.AIM), math.sin(T.AIM), 0.0])
    assert v @ along > 0 and abs(v @ np.array([math.cos(T.PHI), math.sin(T.PHI), 0.0])) < 1e-9  # forward along the jaws
    away = np.array(T.CUP_XY) - np.array(T.LEAD_XY)
    assert v[:2] @ away / np.linalg.norm(away) < 0.8 * np.linalg.norm(v[:2])  # more than 36 degrees off the cup, which is 7 cm away and 5.7 cm wide
    T.reset_memory()
    T.observe(_clip(z=T.Z_BT + 3 * MM), DT)
    assert T.STATE["cut"] == "high" and T.STATE["kick"][0] == ["lead_top"]


def test_the_clip_happens_once():
    T.observe(_clip(), DT)
    kick = T.STATE["kick"]
    T.observe(_clip(z=T.Z_BT + 3 * MM), DT)
    assert T.STATE["cut"] == "flush" and T.STATE["kick"] is kick


def test_non_finite_state_does_not_clip():
    bad = _clip()
    bad["cutter"] = _state([float("nan")] * 3)
    T.observe(bad, DT)
    assert T.STATE["cut"] == "" and T.score(bad) == 0.0 and not T.success(bad)


# -- flight ----------------------------------------------------------------------------------------------------------


def test_the_offcut_flies_when_it_goes_10_cm_from_the_board():
    T.observe(_clip(), DT)
    far = BASE = T.BASE_POINT + (0.0, 0.0, 0.0)
    near = BASE + (0.099, 0.0, 0.0)
    T.observe(_scene(top=near), DT)
    assert not T.STATE["flew"]
    T.observe(_scene(top=far + (0.101, 0.0, 0.0)), DT)
    assert T.STATE["flew"]


def test_flight_is_measured_in_three_dimensions():
    T.observe(_clip(), DT)
    T.observe(_scene(top=T.BASE_POINT + (0.0, 0.0, 0.12)), DT)
    assert T.STATE["flew"]  # straight up, 12 cm


def test_before_the_clip_nothing_flies():
    T.observe(_scene(top=T.BASE_POINT + (0.5, 0.0, 0.0)), DT)
    assert not T.STATE["flew"]


# -- scoring ---------------------------------------------------------------------------------------------------------


def test_start_scores_zero():
    T.observe(_scene(), DT)
    assert T.score(_scene()) == 0.0 and not T.success(_scene())


def test_clipped_flush_and_dropped_in_the_cup_is_success():
    _after_clip(IN_CUP)
    s = _scene(top=IN_CUP)
    assert T.score(s) == 1.0 and T.success(s)


def test_moving_offcut_in_the_cup_is_not_one():
    _after_clip(IN_CUP)
    moving = _scene(top=_state(IN_CUP, vel=(0.05, 0.0, 0.0)))
    assert T.score(moving) == 0.5 and not T.success(moving)  # still moving: within 10 cm, but not at rest anywhere
    spinning = _scene(top=ObjectState(pos=np.asarray(IN_CUP), quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3),
                                      ang_vel=np.array([0.0, 0.0, 2.0]), low=np.zeros(3), high=np.zeros(3)))
    assert T.score(spinning) == 0.5


def test_at_rest_on_the_mat_is_three_quarters():
    _after_clip(ON_MAT)
    s = _scene(top=ON_MAT)
    assert T.score(s) == 0.75 and not T.success(s)


def test_rest_on_the_board_or_in_the_gripper_is_half():
    _after_clip(ON_BOARD)
    assert T.score(_scene(top=ON_BOARD)) == 0.5
    T.reset_memory()
    _after_clip((*T.LEAD_XY, T.Z_BT + 0.05))  # held 5 cm up
    assert T.score(_scene(top=(*T.LEAD_XY, T.Z_BT + 0.05))) == 0.5


def test_above_the_cup_rim_is_not_in_the_cup():
    high = (*T.CUP_XY, T.CUP_RIM + 0.01)
    _after_clip(high)
    assert T.score(_scene(top=_state(high))) == 0.5
    low_outside = (T.CUP_XY[0], T.CUP_XY[1] + 0.025, T.MAT_TOP + 0.01)  # 2.5 cm off the axis: in the cup's wall
    T.reset_memory()
    _after_clip(low_outside)
    assert T.score(_scene(top=low_outside)) == 0.5


def test_high_cut_scores_a_quarter_whatever_the_offcut_does():
    T.observe(_clip(z=T.Z_BT + 3 * MM), DT)
    for pos in (IN_CUP, ON_MAT, ON_BOARD):
        T.observe(_scene(top=pos), DT)
        assert T.score(_scene(top=pos)) == 0.25
        assert not T.success(_scene(top=pos))


def test_flight_scores_zero_for_good_even_if_the_offcut_comes_back():
    T.observe(_clip(), DT)
    T.observe(_scene(top=T.BASE_POINT + (0.15, 0.0, 0.0)), DT)
    assert T.STATE["flew"]
    T.observe(_scene(top=IN_CUP), DT)
    assert T.score(_scene(top=IN_CUP)) == 0.0 and not T.success(_scene(top=IN_CUP))


def test_flying_offcut_scores_zero_even_in_the_step_before_it_is_seen():
    T.observe(_clip(), DT)
    assert T.score(_scene(top=T.BASE_POINT + (0.3, 0.0, 0.0))) == 0.0


def test_a_cut_must_happen():
    for pos in (IN_CUP, ON_MAT, TOP0):
        T.observe(_scene(top=pos), DT)
        assert T.score(_scene(top=pos)) == 0.0 and not T.success(_scene(top=pos))  # moving the offcut isn't a cut


def test_memory_resets():
    T.observe(_clip(), DT)
    T.observe(_scene(top=T.BASE_POINT + (0.3, 0.0, 0.0)), DT)
    assert T.STATE["cut"] and T.STATE["flew"]
    T.reset_memory()
    assert T.STATE == {"cut": "", "flew": False, "kick": None}


def test_the_score_ladder():
    outcomes = set()
    for name, steps in {
        "none": [],
        "high": [("z", 3 * MM)],
        "flush_board": [("flush", None)],
        "flush_mat": [("flush", ON_MAT)],
        "flush_cup": [("flush", IN_CUP)],
    }.items():
        T.reset_memory()
        if steps and steps[0][0] == "z":
            T.observe(_clip(z=T.Z_BT + steps[0][1]), DT)
            top = IN_CUP
        elif steps:
            T.observe(_clip(), DT)
            top = steps[0][1] or ON_BOARD
        else:
            top = TOP0
        T.observe(_scene(top=top), DT)
        outcomes.add(T.score(_scene(top=top)))
    assert outcomes == {0.0, 0.25, 0.5, 0.75, 1.0}


# -- the demo --------------------------------------------------------------------------------------------------------


def _times(waypoints):
    out, t = {}, 0.0
    for w in waypoints:
        t += w.duration
        out[w.label] = t
    return out


def test_demo_is_within_reach_of_both_arms():
    plans = T.demo()
    assert set(plans) == {"left", "right"}
    for arm in ("left", "right"):
        plan = ArmPlan(list(plans[arm]), BASES[arm])
        for k in range(int(plan.duration * 20) + 1):
            plan.action7((k + 1) / 20)
        assert plan.worst_ik_mm < 2.0 and plan.worst_ik_deg < 2.0, (arm, plan.worst_ik_mm, plan.worst_ik_deg)


def test_demo_run_fits_the_step_limit():
    plans = T.demo()
    assert max(sum(w.duration for w in plan) for plan in plans.values()) * 20 < 1200


def test_arms_work_in_their_own_halves():
    plans = T.demo()
    assert all(w.pos[1] <= 0.0 + 1e-9 for w in plans["right"])
    assert all(w.pos[1] >= 0.0 - 1e-9 for w in plans["left"])


def test_left_hand_holds_the_lead_before_the_squeeze_and_through_it():
    plans = T.demo()
    left, right = _times(plans["left"]), _times(plans["right"])
    squeeze_starts = right["squeeze: clip"] - 1.2
    assert left["hold the lead's top"] + 0.5 <= squeeze_starts  # the grip is closed 0.5 s or more before the squeeze
    assert left["hold through the clip"] >= right["let the jaws open"]  # and stays closed until the jaws are open again
    closed = [w for w in plans["left"] if w.label in ("hold the lead's top", "hold through the clip")]
    assert all(w.finger == 0.0 for w in closed)


def test_right_grip_holds_below_the_detent_and_squeezes_to_the_clip():
    plans = T.demo()
    holds = [w for w in plans["right"] if w.label in ("grip the plates", "lift", "slide the jaws round the lead")]
    assert all(w.finger == pytest.approx(-(T.HOLD_SQUEEZE - T.PAD_GAP0) / 2) for w in holds)
    assert T.PAD_GAP0 + 2 * -holds[0].finger == pytest.approx(T.HOLD_SQUEEZE)  # the pads' gap
    assert [w.finger for w in plans["right"] if w.label == "squeeze: clip"] == [0.0]


def test_wrists_stay_15_cm_apart_over_the_clip():
    plans = T.demo()
    left_tcp = np.array([w.pos for w in plans["left"] if w.label == "hold through the clip"][0])
    right_tcp = np.array([w.pos for w in plans["right"] if w.label == "squeeze: clip"][0])
    assert np.linalg.norm(left_tcp[:2] - right_tcp[:2]) > 0.15


def test_the_gripper_pad_tips_clear_the_blade():
    # left pad tips (TCP minus 13 mm) 8 mm over the board: over the blade's top (3.4 mm) with 4.6 mm to spare
    left_tcp = [w.pos for w in T.demo()["left"] if w.label == "hold the lead's top"][0]
    assert left_tcp[2] - 0.013 - T.Z_BT >= T.BLADE_TOP + 0.002
    assert left_tcp[2] - 0.013 - T.Z_BT <= 0.014 - 0.005  # and still on 5 mm or more of the lead's top piece


def test_nippers_slide_in_from_the_open_end_of_the_jaws():
    plans = T.demo()
    waypoints = {w.label: np.asarray(w.pos) for w in plans["right"]}
    slide = waypoints["slide the jaws round the lead"] - waypoints["down to the board"]
    along = np.array([math.cos(T.AIM), math.sin(T.AIM), 0.0])
    assert slide @ along == pytest.approx(T.SLIDE, abs=1e-6)  # forward along the jaws
    assert abs(slide[2]) < 1e-9


def test_demo_one_handed_shortcut_leaves_the_lead_uncaught(monkeypatch):
    monkeypatch.setattr(T, "HOLD_OFFCUT", False)
    left = T.demo()["left"]
    assert len(left) == 1 and left[0].finger == pytest.approx(-0.04695)  # the left hand stays open at READY height
    assert all(w.pos[2] > T.Z_BT + 0.05 for w in left)


def test_demo_high_cut_variant_holds_the_nippers_up(monkeypatch):
    monkeypatch.setattr(T, "CUT_HIGH", 0.006)
    base = {w.label: w.pos for w in T.demo()["right"]}
    monkeypatch.setattr(T, "CUT_HIGH", 0.0)
    flat = {w.label: w.pos for w in T.demo()["right"]}
    assert base["rest on the board"][2] - flat["rest on the board"][2] == pytest.approx(0.006)
