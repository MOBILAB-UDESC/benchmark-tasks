"""Offline checks (no GPU): the task file loads, the scorer covers each outcome and remembers the peak bow and the lift,
a static model of the hinge chain gives the bows the text promises for the demo's hold and for the ends, and the demo
reaches for both arms. Run in the yam_arena environment:  python -m pytest -q test_lift_long_panel_no_sag.py"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
from panelkit import PanelState, bow
from yam_arena.demo import ArmPlan
from yam_arena.station import BASES
from yam_arena.task_file import load

HERE = pathlib.Path(__file__).resolve().parent
TASK = load(HERE / "lift_long_panel_no_sag.py")
T = TASK.module
N = T.SEGMENTS
ELL = T.PANEL_LENGTH / N


def _state(points, speed=0.0, spin=0.0):
    pts = np.asarray(points, float)
    return PanelState(pos=pts.mean(0), quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                      low=pts.min(0), high=pts.max(0), points=pts, speeds=np.full(len(pts), speed),
                      spins=np.full(len(pts), spin), tilts=np.zeros(len(pts)))


def _flat(x, z, y0=0.0, bend=0.0):
    """The panel's segment centres: straight along y at (x, z), its middle ``bend`` metres below the ends' line
    (a parabola)."""
    y = y0 - T.PANEL_LENGTH / 2 + (np.arange(N) + 0.5) * ELL
    s = (y - y0) / ((N - 1) * ELL / 2)  # 1 at the end segments' centres, so the bow is exactly ``bend``
    return np.stack([np.full(N, x), y, z - bend * (1 - s * s)], axis=1)


def _scene(points, **kw):
    return {"panel": _state(points, **kw)}


START = _flat(T.START_X, T.START_Z)
HELD = _flat(0.33, T.START_Z + 0.10)  # carried, flat, between the bench and the rack
ON_RACK = _flat(T.RACK_X, T.RACK_Z)


@pytest.fixture(autouse=True)
def fresh_episode():
    T.reset_memory()


def test_loads():
    assert TASK.scene == "station"
    assert TASK.free_objects == ()  # the panel is an articulation the kit spawns, never a free box
    assert any(o.name == "panel" for o in TASK.objects)


def test_layout():
    panel = T.PANEL
    assert panel.seg_len == pytest.approx(0.03)
    assert panel.hinge_stiffness == pytest.approx(T.PANEL_EI / panel.seg_len)
    assert T.START_Z - T.PANEL_THICK / 2 > T.START_TOP  # off the bench rail, not in it
    assert T.START_Z - T.PANEL_THICK / 2 - T.START_TOP < 0.001
    assert T.RACK_Z - T.PANEL_THICK / 2 == pytest.approx(T.RACK_TOP)
    assert T.RACK_TOP - T.START_TOP == pytest.approx(0.05)
    # the rails reach past the panel's ends, the panel and the rails keep their x lines
    assert T.BAR[1] > T.PANEL_LENGTH
    assert T.BAR[0] / 2 < T.PANEL_WIDTH / 2 - 0.01  # the pads close outside the bar with 1.2 cm to spare
    for rail in (*T.BENCH, *T.RACK):
        assert not rail.free
    # the arms' halves: nothing of the rails or the grips crosses to the other arm's side
    assert T.GRIP_Y < T.PANEL_LENGTH / 2


def test_start_scores_zero():
    assert T.score(_scene(START)) == 0.0
    assert not T.success(_scene(START))
    assert T.peak_bow() == pytest.approx(0.0, abs=1e-9)


def test_full_score_needs_lift_and_rest_on_the_rack():
    assert T.score(_scene(HELD)) == 0.5  # lifted clear, carried, not yet on the rack
    assert T.score(_scene(ON_RACK)) == 1.0
    assert T.success(_scene(ON_RACK))


def test_moving_on_the_rack_is_not_resting():
    T.score(_scene(HELD))
    assert T.score(_scene(ON_RACK, speed=0.05)) == 0.5
    assert not T.success(_scene(ON_RACK, speed=0.05))
    assert not T.success(_scene(ON_RACK, spin=2.0))


def test_the_rack_is_on_its_rail_within_tolerance():
    T.score(_scene(HELD))
    assert T.score(_scene(_flat(T.RACK_X + 0.019, T.RACK_Z))) == 1.0
    assert T.score(_scene(_flat(T.RACK_X + 0.025, T.RACK_Z))) == 0.5  # 2.5 cm off its line
    assert T.score(_scene(_flat(T.RACK_X, T.RACK_Z, y0=0.025))) == 0.5  # 2.5 cm along the rail
    assert T.score(_scene(_flat(T.RACK_X, T.RACK_Z + 0.006))) == 0.5  # 6 mm above its rail: caught, not resting on it
    assert T.score(_scene(_flat(T.RACK_X - 0.12, T.RACK_Z))) == 0.5  # on the table side of the rack


def test_a_bow_over_4_mm_zeroes_the_trial_for_good():
    assert T.score(_scene(_flat(0.33, T.START_Z + 0.10, bend=0.0041))) == 0.0
    assert T.peak_bow() > T.BOW_LIMIT
    assert T.score(_scene(ON_RACK)) == 0.0  # remembered, even flat on the rack
    assert not T.success(_scene(ON_RACK))


def test_bow_between_3_and_4_mm_caps_the_score_at_three_quarters():
    T.score(_scene(_flat(0.33, T.START_Z + 0.10, bend=0.0035)))
    assert T.score(_scene(ON_RACK)) == 0.75
    assert not T.success(_scene(ON_RACK))


def test_a_bow_of_3_mm_exactly_still_counts_as_clean():
    T.score(_scene(_flat(0.33, T.START_Z + 0.10, bend=0.0029)))
    assert T.score(_scene(ON_RACK)) == 1.0


def test_never_lifted_scores_zero():
    # slid along the table, or lifted only 2 cm: never 3 cm clear of its start height
    assert T.score(_scene(_flat(0.33, T.START_Z + 0.02))) == 0.0
    assert T.score(_scene(_flat(T.RACK_X, T.START_Z + 0.02))) == 0.0
    assert T.RACK_Z - T.START_Z >= T.LIFT_CLEAR  # on the rack's rail, the panel is clear by its height


def test_a_part_lifted_only_at_one_end_is_not_lifted_clear():
    pts = _flat(0.33, T.START_Z + 0.10)
    pts[:3, 2] = T.START_Z + 0.01
    assert T.score(_scene(pts)) == 0.0


def test_a_fallen_panel_scores_zero():
    T.score(_scene(HELD))
    fallen = _flat(0.33, T.TOP - 0.03)
    assert T.score(_scene(fallen)) == 0.0
    assert not T.success(_scene(fallen))


def test_non_finite_state_scores_zero():
    bad = _flat(T.RACK_X, T.RACK_Z)
    bad[3, 2] = np.nan
    assert T.score(_scene(bad)) == 0.0
    assert not T.success(_scene(bad))


def test_reset_clears_the_memory():
    T.score(_scene(_flat(0.33, T.START_Z + 0.10, bend=0.01)))
    assert T.peak_bow() > 0.009
    T.reset_memory()
    assert T.peak_bow() == 0.0
    assert not T.STATE["lifted"]


def test_set_down_off_the_rack_after_a_clean_lift_is_half():
    T.score(_scene(HELD))
    assert T.score(_scene(_flat(T.START_X, T.START_Z))) == 0.5  # put back on the bench rail


# -- the design: a static model of the hinge chain ----------------------------------------------------------------


def _chain_bow(grip_y_from_end: float, clamped: bool, mass=T.PANEL_MASS, ei=T.PANEL_EI, n=N) -> float:
    """The bow (m) of a chain of n segments on torsion springs of EI / ell, lifted at two points that are
    ``grip_y_from_end`` from each end (nearest segment centres). ``clamped`` holds the gripped segments level, as a
    gripper that doesn't rotate does, otherwise they only hold their height."""
    ell = T.PANEL_LENGTH / n
    k, m = ei / ell, mass / n
    a = min(int(grip_y_from_end / ell), n // 2 - 1)
    b = n - 1 - a
    nv = n + 1  # unknowns: each segment's angle, and the first one's height
    Z = np.zeros((n, nv))
    for i in range(1, n):
        Z[i] = Z[i - 1]
        Z[i, i - 1] += ell / 2
        Z[i, i] += ell / 2
    Z[:, n] = 1.0
    K = np.zeros((nv, nv))
    for j in range(1, n):
        v = np.zeros(nv)
        v[j], v[j - 1] = 1.0, -1.0
        K += k * np.outer(v, v)
    f = m * 9.81 * Z.sum(0)
    rows, rhs = [Z[a] - Z[b], Z[a]], [0.0, 0.0]
    if clamped:
        for s in (a, b):
            v = np.zeros(nv)
            v[s] = 1.0
            rows.append(v)
            rhs.append(0.0)
    C = np.array(rows)
    kkt = np.block([[K, C.T], [C, np.zeros((len(C), len(C)))]])
    x = np.linalg.lstsq(kkt, np.r_[-f, rhs], rcond=None)[0][:nv]
    z = Z @ x
    pts = np.stack([np.zeros(n), (np.arange(n) + 0.5) * ell, z], axis=1)
    return bow(pts)


def test_the_demo_hold_stays_flat_in_the_model():
    inside = T.PANEL_LENGTH / 2 - T.GRIP_Y  # the grip's distance from the panel's end
    for clamped in (False, True):
        assert _chain_bow(inside, clamped) < 0.003


def test_lifting_by_the_ends_cracks_it_in_the_model():
    for clamped in (False, True):
        assert _chain_bow(0.015, clamped) > 0.004


def test_lifting_at_the_middle_cracks_it_in_the_model():
    for clamped in (False, True):
        assert _chain_bow(0.30 - 0.06, clamped) > 0.004  # the two hands 12 cm apart in the middle


# -- the demo -------------------------------------------------------------------------------------------------------


def _plan(side, grip_y=None):
    if grip_y is not None:
        T.GRIP_Y = grip_y
    try:
        waypoints = T.demo()[side]
    finally:
        T.GRIP_Y = 0.17
    plan = ArmPlan(waypoints, BASES[side])
    t = 0.0
    while t < plan.duration:
        plan.action7(t)
        t += 0.05
    return plan


@pytest.mark.parametrize("side", ["left", "right"])
def test_demo_reach(side):
    plan = _plan(side)
    assert plan.worst_ik_mm < 2.0
    assert plan.worst_ik_deg < 2.0


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("grip_y", [0.285, 0.24, 0.15])
def test_cheat_variants_reach(side, grip_y):
    plan = _plan(side, grip_y)
    assert plan.worst_ik_mm < 2.0
    assert plan.worst_ik_deg < 2.0


def test_demo_arms_move_together_and_stay_in_their_halves():
    demo = T.demo()
    assert [w.duration for w in demo["left"]] == [w.duration for w in demo["right"]]
    for w in demo["left"]:
        assert w.pos[1] > 0
    for w in demo["right"]:
        assert w.pos[1] < 0
    # 15 cm or more between the two grippers' centres (the hands are 34 cm apart)
    assert 2 * T.GRIP_Y >= 0.15


def test_demo_grips_at_the_start_and_sets_down_on_the_rack():
    demo = T.demo()["left"]
    labels = [w.label for w in demo]
    assert labels.index("close") < labels.index("lift") < labels.index("carry") < labels.index("lower") < labels.index("release")
    down = demo[labels.index("down beside it")]
    assert down.pos[0] == T.START_X
    assert demo[labels.index("lower")].pos[0] == T.RACK_X
    # the pad tips go under the panel's lower face, outside the bar: TCP above the rail top by under 13 mm
    assert 0 < down.pos[2] - T.START_TOP < 0.013
    assert math.isclose(demo[labels.index("lower")].pos[2] - T.RACK_TOP, down.pos[2] - T.START_TOP)
