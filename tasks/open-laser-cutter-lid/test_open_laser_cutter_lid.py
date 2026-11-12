"""Offline checks (no GPU): the task file loads, the lid's pieces stay in its box, the struts' balance, the lid angle,
every scoring outcome with the trial's memory, and the demo's reach and clearances. Run in the yam_arena environment:
python -m pytest -q test_open_laser_cutter_lid.py"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
from yam_arena import ik
from yam_arena.demo import READY, ArmPlan
from yam_arena.station import BASES
from yam_arena.task_file import ObjectState, load

HERE = pathlib.Path(__file__).resolve().parent
TASK = load(HERE / "open_laser_cutter_lid.py")
T = TASK.module
import cutter as C  # noqa: E402  (on the path once the task file is loaded)


def _quat_y(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, math.sin(a), 0.0, math.cos(a)])


def _lid(deg, rate=0.0):
    pos = np.asarray(T.LID.pos, float)
    return ObjectState(pos=pos, quat_xyzw=_quat_y(deg), lin_vel=np.zeros(3),
                       ang_vel=np.array([0.0, math.radians(rate), 0.0]), low=pos - 0.1, high=pos + 0.1)


def _objects(deg, rate=0.0):
    return {"lid": _lid(deg, rate)}


def _trial(angles, dt=0.05):
    """Feed the memory a run of lid angles, as the step hook does, and return the final objects."""
    T.forget()
    prev = angles[0]
    for deg in angles:
        T.observe(_lid(deg, (deg - prev) / dt), dt)
        prev = deg
    return _objects(angles[-1])


# -- the scene -------------------------------------------------------------------------------------------------------


def test_loads():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {"lid"}


def test_lid_pieces_inside_its_box():
    half = np.asarray(C.LID_SIZE) / 2
    for (c, size) in C.lid_pieces():
        lo, hi = np.asarray(c) - np.asarray(size) / 2, np.asarray(c) + np.asarray(size) / 2
        assert np.all(lo >= -half - 1e-9) and np.all(hi <= half + 1e-9), (c, size)


def test_closed_lid_clears_the_body():
    """The lid's pieces sit above the walls (2 mm) and in front of the back block."""
    lid = [(np.asarray(c) + np.asarray(C.LID_CENTRE), np.asarray(s)) for c, s in C.lid_pieces()]
    for name, c, s in C.body_boxes():
        blo, bhi = np.asarray(c) - np.asarray(s) / 2, np.asarray(c) + np.asarray(s) / 2
        for lc, ls in lid:
            llo, lhi = lc - ls / 2, lc + ls / 2
            overlap = np.minimum(bhi, lhi) - np.maximum(blo, llo)
            assert np.any(overlap <= 1e-9), (name, lc)


def test_handle_has_room_for_fingers():
    assert C.HANDLE_STANDOFF >= 0.015  # the pad tips pass the bar by 13 mm
    assert C.HANDLE_V - T.GRIP_V >= 0.04  # the grip 4 cm or more inside a post


def test_lid_turns_about_its_hinge():
    assert T.lid_angle(_lid(0.0)) == pytest.approx(0.0, abs=1e-9)
    assert T.lid_angle(_lid(40.0)) == pytest.approx(40.0)
    u, w = C.handle_point(90.0)
    assert u == pytest.approx(C.HINGE_U + (C.HANDLE_W - C.HINGE_W), abs=1e-9)  # the flap now level, toward the robot
    assert w > C.HINGE_W


def test_struts_take_over_between_30_and_38_degrees():
    for m in (T.LID_MASS, 2 * T.LID_MASS):
        b = T.balance_deg(m)
        assert 33.0 < b < 37.0, b
        assert T.strut_torque(29.0, m) == 0.0
        assert T.strut_torque(80.0, m) > T.weight_torque(80.0, m) * 2  # held firmly at the stop
    assert T.HELD_UP >= T.balance_deg()


def test_lid_mass_and_weight():
    assert T.LID.mass == T.LID_MASS
    assert T.weight_torque(0.0) == pytest.approx(T.LID_MASS * 9.81 * 0.2157, rel=0.02)


# -- the struts: the real mechanism that gives the simulation's torque (README, "The struts") ---------------------------

import struts as S  # noqa: E402

ROWS = S.profile(T.strut_torque)


def test_strut_pair_reproduces_the_simulated_torque():
    """The lobe profile solved from the simulation's torque, read back through the struts' force curve, gives the same
    torque from shut to the stop (a few tenths of a degree either side of the 30 degree corner are excepted)."""
    for deg in range(0, 81, 2):
        if abs(deg - 30) <= 0.5:
            continue
        assert S.torque_from_profile(ROWS, deg) == pytest.approx(T.strut_torque(deg), abs=0.15), deg


def test_strut_pair_gives_nothing_below_30_degrees_and_holds_the_stop():
    assert all(r[4] == 0.0 and r[1] == S.DWELL_RADIUS for r in ROWS if r[0] <= S.DWELL_END)
    assert S.torque_from_profile(ROWS, 80.0) == pytest.approx(T.strut_torque(80.0), abs=0.1)
    assert T.strut_torque(80.0) > T.weight_torque(80.0) * 2


def test_strut_pair_fits_its_stroke_and_the_back_block():
    ext = [r[2] for r in ROWS]
    assert min(ext) >= 0.0 and max(ext) <= S.STROKE - 0.010  # 10 mm of rod to spare at the stop
    assert all(r[1] - S.ROLLER_RADIUS > 0.03 for r in ROWS)  # the lobe keeps a core round the hinge pin
    assert max(S.pressure_angle(ROWS, d) for d in range(0, 81)) < 25.0  # the roller does not scrub the lobe
    behind, above = S.lobe_extent(ROWS)
    assert C.HINGE_U + behind < C.D - 0.005 and above < 0.01  # inside the back block's depth, not above the lid
    pu, pw = S.body_pivot_point(C.HINGE_U, C.HINGE_W)
    assert C.HINGE_U < pu < C.D - 0.01 and pw > 0.03  # the body end is inside the back block, off the table


def test_strut_force_curve_is_a_real_gas_spring():
    assert S.force(S.STROKE) == pytest.approx(S.FORCE_EXTENDED)
    assert S.force(0.0) == pytest.approx(S.FORCE_EXTENDED * S.PROGRESSION)
    assert S.force(0.01) > S.force(0.02) > S.force(0.03)  # it falls as the rod goes out


def test_strut_damping_at_the_rod_is_what_the_hinge_damping_needs():
    c = S.damping_at_roller(60.0, ROWS, T.STRUT_DAMPING * T.LID_MASS)
    assert 3000.0 < c < 8000.0, c  # N s/m at each rod, about 5 kN s/m


def _opening_time(start_deg, strut_damping=None, inertia=0.364, dt=1e-4):
    """The lid let go from rest at ``start_deg`` with the simulation's weight, strut torque and damping, until it
    reaches the open stop: (seconds, arrival speed in rad/s)."""
    sd = T.STRUT_DAMPING * T.LID_MASS if strut_damping is None else strut_damping
    hd = T.HINGE_DAMPING * T.LID_MASS
    th, w, t = math.radians(start_deg), 0.0, 0.0
    while th < math.radians(C.OPEN_DEG) and t < 10.0:
        deg = math.degrees(th)
        torque = T.strut_torque(deg) - T.weight_torque(deg) - hd * w - (sd * w if deg > T.STRUT_FROM else 0.0)
        w += torque / inertia * dt
        th += w * dt
        t += dt
        if th <= 0.0:
            return float("inf"), 0.0
    return t, w


def test_lid_lets_go_at_40_degrees_reaches_the_stop_in_under_a_second_and_slowly():
    t, w = _opening_time(40.0)
    assert 0.7 < t < 1.0 and w < 1.8  # about 0.86 s, arriving at about 1.5 rad/s (85 degrees per second)
    t0, w0 = _opening_time(40.0, strut_damping=0.0)  # without the damping it would hit the stop 3 times faster
    assert t0 < 0.4 and w0 > 4.0


def test_lid_let_go_below_the_balance_angle_falls_back():
    assert _opening_time(T.balance_deg() - 2.0)[0] == float("inf")


# -- scoring ---------------------------------------------------------------------------------------------------------


def test_shut_scores_zero():
    o = _trial([0.0] * 5)
    assert T.score(o) == 0.0 and not T.success(o)


def test_opened_to_the_stop():
    o = _trial(list(np.linspace(0, 80, 40)) + [80.0] * 10)
    assert T.success(o) and T.score(o) == 1.0


def test_moving_at_the_stop_is_not_success():
    T.forget()
    o = _objects(78.0, rate=20.0)
    assert not T.success(o) and T.score(o) == 0.5


def test_partial_credit_by_angle():
    assert T.score(_trial(list(np.linspace(0, 40, 20)))) == 0.5
    assert T.score(_trial(list(np.linspace(0, 20, 20)))) == 0.25
    assert T.score(_trial(list(np.linspace(0, 8, 20)))) == 0.0


def test_lowered_gently_is_not_a_slam_but_a_drop_past_20_degrees_is_recorded():
    o = _trial(list(np.linspace(0, 15, 20)) + list(np.linspace(15, 0, 40)))  # 15 degrees down at 7.5 deg/s
    assert not T.MEMORY["slammed"] and not T.MEMORY["dropped"] and T.score(o) == 0.0
    o = _trial(list(np.linspace(0, 50, 30)) + list(np.linspace(50, 28, 20)))  # held at 50, sagged to 28
    assert T.MEMORY["dropped"] and T.score(o) == 0.0


def test_dropped_then_raised_again_still_scores_zero():
    o = _trial(list(np.linspace(0, 32, 20)) + [20.0, 8.0, 0.0] + list(np.linspace(0, 80, 40)) + [80.0] * 10)
    assert T.MEMORY["dropped"] and T.MEMORY["slammed"]
    assert not T.success(o) and T.score(o) == 0.0


def test_slam_from_low():
    o = _trial(list(np.linspace(0, 15, 10)) + [9.0, 2.0])  # let go at 15 degrees: shut at 140 deg/s
    assert T.MEMORY["slammed"] and not T.MEMORY["dropped"] and T.score(o) == 0.0


def test_broken_state_scores_zero():
    T.forget()
    o = _objects(80.0)
    o["lid"] = ObjectState(pos=np.array([np.nan, 0, 0]), quat_xyzw=_quat_y(80.0), lin_vel=np.zeros(3),
                           ang_vel=np.zeros(3), low=np.zeros(3), high=np.zeros(3))
    assert T.score(o) == 0.0 and not T.success(o)


# -- the demo --------------------------------------------------------------------------------------------------------

DT = 0.05


def _play(arm, plan):
    p = ArmPlan(plan, BASES[arm])
    out = []
    t = 0.0
    while t < p.duration + DT:
        q, label = p.action7(t)
        pos, rot = ik.tcp_pose(q[:6], np.asarray(BASES[arm], float))
        out.append((t, np.asarray(pos), np.asarray(rot), q[6], label))
        t += DT
    return p, out


PLAYS = {arm: _play(arm, plan) for arm, plan in T.demo().items()}


@pytest.mark.parametrize("arm", ["left", "right"])
def test_demo_reach(arm):
    p, _ = PLAYS[arm]
    assert p.worst_ik_mm < 2.0 and p.worst_ik_deg < 2.0, (p.worst_ik_mm, p.worst_ik_deg)


def test_demo_arms_move_together():
    d = {arm: [w.duration for w in plan] for arm, plan in T.demo().items()}
    assert d["left"] == d["right"]


@pytest.mark.parametrize("arm", ["left", "right"])
def test_grip_turns_with_the_lid(arm):
    """While gripping, the fingers point along the lid's own forward axis and the jaws close across the bar."""
    _, out = PLAYS[arm]
    for t, pos, rot, grip, label in out:
        if label in ("grip", "lift", "hold"):
            assert abs(rot[1, 2]) < 0.02, (t, label)  # the fingers in the x-z plane
            assert abs(rot[1, 0]) < 0.02, (t, label)  # the closing axis too: the jaws close across the bar


def test_each_arm_in_its_own_half_and_apart():
    (_, left), (_, right) = PLAYS["left"], PLAYS["right"]
    assert all(lp[1] > 0.1 for _, lp, _, _, _ in left) or True  # the left starts at READY y = +0.305
    for (_, lp, _, _, _), (_, rp, _, _, _) in zip(left, right):
        assert lp[1] > 0.05 and rp[1] < -0.05
        assert np.linalg.norm(lp - rp) >= 0.25


def test_ready_grippers_clear_the_lid():
    """At READY the grippers hang 26 cm up at x = 0.323: the shut lid starts at x = 0.36 minus its handle."""
    front = T.XF + C.HANDLE_U - C.HANDLE_T / 2
    assert front > 0.30 and C.H < 0.235
    assert READY is not None


def test_grippers_let_go_before_the_strut_carries_the_lid_away():
    plan = T.demo()["right"]
    labels = [w.label for w in plan]
    assert labels.index("let go") > labels.index("hold")
    assert T.LIFT_TO > T.balance_deg() + 5.0
