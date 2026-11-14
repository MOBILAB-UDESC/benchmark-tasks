"""Offline checks (no GPU): the task file loads, the hinge's friction layers, the leaf's root and the rocking rule are
specified as the task needs, the angles are read from the quaternions, the rocking counter, the scorer covers each outcome
and remembers a snapped leaf, the leaf and the knuckle clear the base leaf as they swing, the demo's order, reach and
clearance, and that every mesh fits its box. Run in the yam_arena environment:  python -m pytest -q test_free_printed_hinge.py"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
from yam_arena import YAM_URDF
from yam_arena.constants import ARM_JOINTS
from yam_arena.demo import ArmPlan
from yam_arena.station import BASES
from yam_arena.task_file import ObjectState, load
from yam_arena.urdf import forward_kinematics

HERE = pathlib.Path(__file__).resolve().parent
TASK = load(HERE / "free_printed_hinge.py")
T = TASK.module
SPECS = {o.name: o for o in T.OBJECTS}


def _qz(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, 0.0, math.sin(a), math.cos(a)])


def _state(pos, size, quat=(0.0, 0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0)):
    pos = np.asarray(pos, float)
    half = np.asarray(size, float) / 2
    return ObjectState(pos=pos, quat_xyzw=np.asarray(quat, float), lin_vel=np.asarray(vel, float),
                       ang_vel=np.asarray(spin, float), low=pos - half, high=pos + half)


def _about_hinge(centre, deg):
    """A point turned ``deg`` degrees about the hinge's vertical axis."""
    h = np.array([T.HX, T.HY, centre[2]])
    a = math.radians(deg)
    r = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    return h + r @ (np.asarray(centre, float) - h)


def _scene(angle=0.0, layers=0, leaf_pos=None, vel=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0)):
    """The leaf turned ``angle`` degrees about the hinge, with ``layers`` of the stuck layers still on."""
    T.STATE["layers"] = layers
    pos = _about_hinge(T.LEAF.pos, angle) if leaf_pos is None else leaf_pos
    return {"leaf": _state(pos, T.LEAF.size, _qz(angle), vel, spin)}


@pytest.fixture(autouse=True)
def fresh_episode():
    T.reset_memory()


def test_loads():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {"leaf"}


def test_stuck_layers_and_the_snap():
    """The hinge is held harder than the leaf can take, and what the layers hold falls as they switch off."""
    leaf = T.LEAF
    assert len(leaf.layers) == 4 and leaf.to == "base" and leaf.axis == "Z"
    assert T.STUCK_TORQUE == pytest.approx(0.8) and T.LAYER_K * 4 == pytest.approx(8.0)
    assert T.STUCK_TORQUE > 1.5 * T.SNAP_TORQUE  # a hard push snaps the leaf long before the layers slip
    assert math.degrees(T.SNAP_TORQUE / (T.LAYER_K * 4)) == pytest.approx(3.5, abs=0.1)
    assert leaf.layers[0][2] * 4 > T.SNAP_TORQUE


def test_layers_hold_torque_by_angle():
    for layers, snap_deg in ((4, 3.5), (3, 4.67), (2, 7.0), (1, 14.03)):
        assert T.holding_torque(_scene(snap_deg - 0.1, layers)) < T.SNAP_TORQUE
        assert T.holding_torque(_scene(snap_deg + 0.1, layers)) > T.SNAP_TORQUE
    assert T.holding_torque(_scene(90.0, 0)) == 0.0
    assert T.holding_torque(_scene(-2.0, 4)) == pytest.approx(T.holding_torque(_scene(2.0, 4)))


def test_axis_frame_turns_x_onto_the_axis():
    from bonds import axis_frame

    def rotate(q, v):
        w, x, y, z = q
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        return r @ np.asarray(v, float)

    for axis, expect in (("X", (1, 0, 0)), ("Y", (0, 1, 0)), ("Z", (0, 0, 1))):
        assert rotate(axis_frame(axis), (1, 0, 0)) == pytest.approx(expect, abs=1e-9)


def test_angle_from_quats():
    for deg in (-10.0, 0.0, 30.0, 120.0, 130.0):
        assert T.leaf_angle(_scene(deg)) == pytest.approx(deg, abs=1e-6)


def test_seat_offset():
    for angle in (0.0, 35.0, 130.0):
        assert T.seat_offset(_scene(angle)) < 1e-6, angle
    off = _scene(0.0, leaf_pos=np.asarray(T.LEAF.pos) + np.array([0.0, 0.0, 0.0]) + np.array([0.02, 0.0, 0.0]))
    assert T.seat_offset(off) == pytest.approx(0.02, abs=1e-6)


def test_rocking_counter():
    r = T.Rocking()
    seq = [0.0, 0.5, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0, 0.0, -1.5, 1.5, -1.5, 1.5]
    done = [r.update(p) for p in seq]
    # first push (+1.0) starts it, -1.0 is half a cycle, +1.0 completes the first, -1.5 then +1.5 the second, ...
    assert done.index(True) == seq.index(1.0, 9)
    assert r.cycles == 3 and sum(done) == 3


def test_rocking_counter_ignores_small_and_same_side_pushes():
    r = T.Rocking()
    for p in (0.5, -0.5, 0.8, -0.8, 0.0):  # under 0.9 degrees: nothing counts
        assert not r.update(p)
    for p in (1.5, 2.0, 1.2, 0.3, 1.9, 1.0):  # the same side again and again: only the first push, no cycle
        assert not r.update(p)
    assert r.cycles == 0


def test_a_steady_push_never_loosens_it():
    r = T.Rocking()
    assert not any(r.update(p) for p in np.linspace(0.0, 3.4, 200))
    assert r.cycles == 0


def test_start_scores_zero():
    assert T.score(_scene(0.0, 4)) == 0.0 and not T.success(_scene(0.0, 4))


STEP_S = 1.0 / T.CONTROL_HZ
STILL_STEPS = round(T.STILL_S * T.CONTROL_HZ)  # 20 control steps: 1 s at 20 Hz


def _let_go(released=True):
    """What the step hook records: no finger within 3 mm of the leaf."""
    T.STATE["released"] = released


def _steps(objects, n):
    """What the step hook does each control step for ``n`` steps in this state (after the snap and the release are
    read): time the open, still leaf, and run the test push. Returns the push switches asked for."""
    out = []
    for _ in range(n):
        T._remember(objects)
        a = T.tick(objects, STEP_S)
        if a:
            out.append(a)
    return out


def _free_and_tested(start=130.0):
    """A freed leaf let go at ``start``, still 1 s, then turned back by the test push at its 30 degrees a second."""
    _let_go()
    assert _steps(_scene(start), STILL_STEPS) == ["start"]
    asked = []
    angle = start
    while T.STATE["test"] == "running":
        angle -= T.TEST_SPEED * STEP_S
        asked += _steps(_scene(angle), 1)
    return asked, angle


@pytest.mark.parametrize("deg, expected", [(5.0, 0.0), (59.9, 0.0), (60.01, 0.5), (119.9, 0.5), (120.01, 0.5),
                                           (130.0, 0.5)])
def test_ladder(deg, expected):
    """The angle alone gives at most 0.5: full credit needs the free-movement test."""
    _let_go()
    assert T.score(_scene(deg, layers=0)) == expected


def test_full_score_needs_release_stillness_and_the_test_push():
    asked, angle = _free_and_tested()
    assert asked == ["stop"] and T.STATE["test"] == "passed"
    assert T.STATE["test_from"] - angle >= T.TEST_DEG and T.STATE["test_s"] <= T.TEST_S
    assert T.success(_scene(angle)) and T.score(_scene(angle)) == 1.0


def test_held_at_120_never_scores_full():
    """The reviewers' case: a resistant leaf held at 120 degrees or more in a gripper is never tested, so never 1.0."""
    _let_go(False)
    held = _scene(125.0)
    assert _steps(held, 10 * STILL_STEPS) == [] and T.STATE["test"] is None
    assert not T.success(held) and T.score(held) == 0.5


def test_the_second_restarts_on_any_lapse():
    _let_go()
    _steps(_scene(125.0), STILL_STEPS - 1)
    assert T.STATE["test"] is None and T.score(_scene(125.0)) == 0.5
    for lapse in ({"vel": (0.05, 0.0, 0.0)}, {"spin": (0.0, 0.0, 1.0)}):
        _steps(_scene(125.0, **lapse), 1)  # swinging for one step
        assert T.STATE["open_s"] == 0.0
        _steps(_scene(125.0), STILL_STEPS - 1)
        assert T.STATE["test"] is None
    _let_go(False)
    _steps(_scene(125.0), 1)  # a finger back on the leaf
    _let_go(True)
    _steps(_scene(125.0), STILL_STEPS - 1)
    assert T.STATE["test"] is None
    assert _steps(_scene(125.0), 1) == ["start"]


def test_a_hinge_that_does_not_turn_fails_the_test():
    """A leaf that the push cannot turn 30 degrees within 3 s (a hinge still held) fails, for good: 0.5."""
    _let_go()
    _steps(_scene(125.0), STILL_STEPS)
    assert T.STATE["test"] == "running"
    asked = _steps(_scene(110.0), round(T.TEST_S * T.CONTROL_HZ))  # it turns only 15 degrees
    assert asked == ["stop"] and T.STATE["test"] == "failed"
    assert _steps(_scene(80.0), 100) == []  # not tested again
    assert not T.success(_scene(80.0)) and T.score(_scene(80.0)) == 0.5


def test_touching_the_leaf_during_the_test_fails_it():
    _let_go()
    _steps(_scene(125.0), STILL_STEPS)
    _let_go(False)
    assert _steps(_scene(90.0), 1) == ["stop"] and T.STATE["test"] == "failed"
    _let_go(True)
    assert T.score(_scene(90.0)) == 0.5


def test_no_full_score_while_any_condition_is_missing():
    """Each full-success condition missing in turn, for 10 s: never 1.0."""
    cases = {
        "held (not let go)": (False, _scene(125.0)),
        "under 120 degrees": (True, _scene(119.0)),
        "swinging": (True, _scene(125.0, vel=(0.05, 0.0, 0.0))),
        "turning": (True, _scene(125.0, spin=(0.0, 0.0, 1.0))),
    }
    for name, (released, objects) in cases.items():
        T.reset_memory()
        _let_go(released)
        _steps(objects, 10 * T.CONTROL_HZ)
        assert not T.success(objects) and T.score(objects) < 1.0, name


def test_finger_gap():
    leaf = _scene(0.0)
    tip = T.LEAF.pos + np.array([T.LEAF_BOX[0] / 2, 0.0, 0.0])
    side = T.LEAF.pos + np.array([0.0, T.LEAF_BOX[1] / 2, 0.0])
    assert T.finger_gap(leaf, [tip]) == pytest.approx(0.0, abs=1e-9)
    assert T.finger_gap(leaf, [side + (0.0, 0.0025, 0.0)]) == pytest.approx(0.0025)
    assert T.finger_gap(leaf, [side + (0.0, 0.009, 0.0), T.LEAF.pos]) == 0.0
    assert T.finger_gap(leaf, []) == math.inf and T.finger_gap(leaf, None) == math.inf
    assert 0.0025 < T.RELEASED_GAP < 0.009


def test_step_hook_reads_release_and_runs_the_test(monkeypatch):
    """watch() reads the fingers, times the second with the run's step and switches the test push on and off."""
    import sys
    import types

    import fingers

    state = {"angle": 128.0, "points": [T.LEAF.pos + np.array([0.0, 0.3, 0.0])]}
    monkeypatch.setitem(sys.modules, "yam_arena.scene",
                        types.SimpleNamespace(object_states=lambda env, n: _scene(state["angle"])))
    monkeypatch.setattr(fingers, "finger_points", lambda env: np.asarray(state["points"]))
    pushes = []
    monkeypatch.setattr(T, "test_push", lambda name, on: pushes.append(on))
    env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(step_dt=0.05))
    T.ROCK.side = 1  # the rocking is over: the leaf stays on one side
    for _ in range(STILL_STEPS):
        T.watch(env)
    assert T.STATE["released"] and pushes == [True] and T.STATE["test"] == "running"
    while T.STATE["test"] == "running":
        state["angle"] -= 1.5
        T.watch(env)
    assert pushes == [True, False] and T.STATE["test"] == "passed" and T.score(_scene(state["angle"])) == 1.0


def test_pushed_to_the_snap_is_zero_for_good():
    assert T.score(_scene(3.4, layers=4)) == 0.0 and not T.STATE["snapped"]  # held, not broken, and still at 0
    assert T.score(_scene(3.6, layers=4)) == 0.0 and T.STATE["snapped"]
    assert T.score(_scene(125.0, layers=0)) == 0.0  # recorded: a later step scores 0 too
    assert _steps(_scene(125.0), 5 * STILL_STEPS) == [] and T.STATE["test"] is None  # a snapped leaf is never tested
    T.reset_memory()
    asked, angle = _free_and_tested(125.0)
    assert T.score(_scene(angle, layers=0)) == 1.0


def test_a_leaf_off_its_hinge_is_snapped():
    s = _scene(125.0, leaf_pos=_about_hinge(T.LEAF.pos, 125.0) + np.array([0.0, 0.0, 0.0]) + np.array([0.02, 0.02, 0.0]))
    assert T.score(s) == 0.0 and T.STATE["snapped"]


def test_reset_clears_the_snap_and_the_layers():
    T.score(_scene(5.0, layers=4))
    assert T.STATE["snapped"]
    T.ROCK.cycles = 3
    T.reset_memory()
    assert not T.STATE["snapped"] and T.STATE["layers"] == 4 and T.ROCK.cycles == 0


def test_non_finite_zero():
    s = _scene(125.0)
    s["leaf"] = _state([np.nan, 0, 0.8], T.LEAF.size)
    assert T.score(s) == 0.0 and not T.success(s)


def test_distance():
    assert T.distance(_scene(0.0)) == pytest.approx(120.0)
    assert T.distance(_scene(140.0)) == 0.0


# -- the layout ------------------------------------------------------------------------------------------------------


def _rect(centre, size, deg):
    """The corners of a box's footprint (x, y) turned ``deg`` degrees about the hinge."""
    cx, cy = centre
    a = math.radians(deg)
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = cx + sx * size[0] / 2 - T.HX, cy + sy * size[1] / 2 - T.HY
            pts.append((T.HX + x * math.cos(a) - y * math.sin(a), T.HY + x * math.sin(a) + y * math.cos(a)))
    return pts


def _inside(point, rect_centre, rect_size, margin):
    return (abs(point[0] - rect_centre[0]) < rect_size[0] / 2 + margin
            and abs(point[1] - rect_centre[1]) < rect_size[1] / 2 + margin)


def _leaf_parts():
    """The leaf's colliders as (name, centre in the station frame, size)."""
    return [(p.name, np.asarray(T.LEAF.pos) + np.asarray(p.centre), p.size) for p in T.shapes()["leaf"]]


def test_the_swing_clears_the_base_leaf():
    """The leaf's plate and knuckle, turned about the hinge through the whole range, stay clear of the base leaf's plate."""
    plate = {p.name: p for p in T.shapes()["base"]}["plate"]
    c = np.asarray(T.BASE.pos) + np.asarray(plate.centre)
    for deg in np.arange(T.HINGE_STOPS[0], T.HINGE_STOPS[1] + 0.5, 1.0):
        for name, centre, size in _leaf_parts():
            a = math.radians(deg)
            lc = _about_hinge(centre, deg)
            for x in np.linspace(-size[0] / 2, size[0] / 2, 31):
                for y in np.linspace(-size[1] / 2, size[1] / 2, 3):
                    p = (lc[0] + x * math.cos(a) - y * math.sin(a), lc[1] + x * math.sin(a) + y * math.cos(a))
                    assert not _inside(p, c[:2], (T.BASE_LEN, T.PLATE_T), 0.0), (name, deg, p)


def test_layout():
    leaf, base = SPECS["leaf"], SPECS["base"]
    parts = {p.name: p for p in T.shapes()["leaf"]}
    # the knuckle is round the hinge axis, the plate starts at its face, and the leaf's box is exactly the two
    assert (np.asarray(leaf.pos) + np.asarray(leaf.hinge))[:2] == pytest.approx((T.HX, T.HY))
    assert (np.asarray(leaf.pos) + np.asarray(parts["knuckle"].centre))[:2] == pytest.approx((T.HX, T.HY))
    plate_x0 = leaf.pos[0] + parts["plate"].centre[0] - parts["plate"].size[0] / 2
    assert plate_x0 == pytest.approx(T.HX + T.KNUCKLE_SIZE[0] / 2)
    lo = min(c.centre[0] - c.size[0] / 2 for c in parts.values())
    hi = max(c.centre[0] + c.size[0] / 2 for c in parts.values())
    assert (lo, hi) == pytest.approx((-leaf.size[0] / 2, leaf.size[0] / 2))
    # the plates stand clear over the card
    for spec in (base, leaf):
        assert spec.pos[2] - spec.size[2] / 2 == pytest.approx(T.Z0)
    assert T.Z0 - (T.TOP + T.CARD_T) == pytest.approx(0.0015)
    # the right arm's half
    assert all(o.pos[1] < -0.04 for o in (base, leaf))
    top = leaf.pos[2] + leaf.size[2] / 2
    assert T.TOP + T.GRIP_H - 0.013 < top - 0.015  # the pads' tips reach 15 mm or more down the plate


def test_the_gauge_card_is_under_the_swing():
    card = SPECS["card"]
    assert card.pos[2] + card.size[2] / 2 == pytest.approx(T.TOP + T.CARD_T)
    for deg in (0, 60, 120, 130):
        tip = _about_hinge((T.HX + T.KNUCKLE_SIZE[0] / 2 + T.LEAF_LEN, T.HY, T.TOP), deg)
        assert abs(tip[0] - card.pos[0]) < card.size[0] / 2 and abs(tip[1] - card.pos[1]) < card.size[1] / 2, deg


@pytest.mark.parametrize("name", ["base", "leaf"])
def test_parts_inside_their_boxes(name):
    half = np.asarray(SPECS[name].size) / 2
    for p in T.shapes()[name]:
        c, s = np.asarray(p.centre), np.asarray(p.size) / 2
        assert np.all(c + s <= half + 1e-9) and np.all(c - s >= -half - 1e-9), p.name


# -- the demo ------------------------------------------------------------------------------------------------------


def _at(steps, label):
    t = 0.0
    for w in steps:
        t += w.duration
        if w.label == label:
            return t
    raise KeyError(label)


def test_demo_rocks_then_sweeps():
    steps = T.demo()["right"]
    labels = [w.label for w in steps]
    assert _at(steps, "close on the leaf") < _at(steps, "rock 1 one way") < _at(steps, "rock 5 the other way")
    assert _at(steps, "rock 5 the other way") < _at(steps, "let go") < _at(steps, f"sweep to {T.SWEEP_TO:.0f} deg")
    assert sum(w.label.startswith("rock") for w in steps) == 2 * T.ROCKS
    assert T.ROCKS >= len(T.LAYERS) + 1  # more cycles than the rule needs, one in reserve
    assert "left" not in T.demo()
    assert sum(w.duration for w in steps) < 50.0  # steps of 20 Hz: well under the 1,200-step limit
    assert labels[-1].startswith("wait")
    # after the hand is up and clear: the 1 s of stillness and the test push (at most TEST_S) fit in the last wait
    assert steps[-1].duration >= T.STILL_S + T.TEST_S + 1.0
    assert steps[-1].pos[2] - T.TOP >= 0.09


def test_demo_rocks_inside_the_limits():
    """The hand's push is bigger than the leaf's flex at its first push (so the stuck leaf is loaded) and the leaf is
    gripped on its plate, 6.5 cm from the hinge."""
    steps = T.demo()["right"]
    ys = [w.pos[1] for w in steps if w.label.startswith("rock")]
    assert max(ys) - T.HY == pytest.approx(T.SWING) and T.HY - min(ys) == pytest.approx(T.SWING)
    gx = [w.pos[0] for w in steps if w.label.startswith("rock")]
    assert all(x == pytest.approx(T.HX + T.GRIP_R) for x in gx)
    leaf = SPECS["leaf"]
    assert leaf.pos[0] - leaf.size[0] / 2 < T.HX + T.GRIP_R < leaf.pos[0] + leaf.size[0] / 2 - 0.01


def test_demo_force_variant_is_one_push():
    T.FORCE_S, saved = 1.5, T.FORCE_S
    try:
        steps = T.demo()["right"]
    finally:
        T.FORCE_S = saved
    assert not any(w.label.startswith("rock") for w in steps)
    push = [w for w in steps if w.label == "push it round"]
    assert len(push) == 1 and push[0].duration == 1.5


def test_demo_sweep_stays_clear_of_the_base_leaf():
    plate = {p.name: p for p in T.shapes()["base"]}["plate"]
    c = np.asarray(T.BASE.pos) + np.asarray(plate.centre)
    for w in T.demo()["right"]:
        if w.label.startswith("sweep") or w.label in ("hold", "back off"):
            assert not _inside(w.pos[:2], c[:2], (T.BASE_LEN, T.PLATE_T), T.FINGER_HALF + 0.01), w.label


def test_demo_sweep_follows_the_arc():
    steps = {w.label: w for w in T.demo()["right"]}
    for deg in (0, 12, 60, 120):
        w = steps[f"sweep to {deg:.0f} deg"] if f"sweep to {deg:.0f} deg" in steps else None
        if w is None:
            continue
        r = math.dist(w.pos[:2], (T.HX, T.HY))
        assert abs(r - math.hypot(T.SWEEP_R, T.PLATE_T / 2 + T.FINGER_HALF + 0.003)) < 1e-9


def test_demo_reachable():
    plan = ArmPlan(T.demo()["right"], BASES["right"])
    for k in range(int(plan.duration * 20) + 1):
        plan.action7(k / 20)
    assert plan.worst_ik_mm < 2.0 and plan.worst_ik_deg < 2.0, (plan.worst_ik_mm, plan.worst_ik_deg)


def test_demo_arm_clears_the_hinge_and_card():
    """Along the whole demo, the arm's links (link 3 to the gripper's body, 6 cm short of the TCP) stay 2 cm or more above
    the card and clear of the base leaf, apart from the fingers themselves."""
    plan = ArmPlan(T.demo()["right"], BASES["right"])
    base = np.asarray(BASES["right"])
    worst = 1.0
    for k in range(int(plan.duration * 20) + 1):
        action, label = plan.action7(k / 20)
        fk = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, action[:6])})
        pts = [fk[n][:3, 3] + base for n in ("link3", "link4", "link5", "gripper")]
        g = fk["gripper"]
        pts.append(g[:3, 3] + base + g[:3, :3] @ np.array([0.0, 0.0, 0.134 - 0.06]))
        for a, b in zip(pts, pts[1:]):
            for p in (a + (b - a) * s for s in np.linspace(0, 1, 6)):
                worst = min(worst, float(p[2] - (T.TOP + T.CARD_T + T.PLATE_H)))
    assert worst > 0.02, worst


# -- the meshes --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["base", "leaf", "card"])
def test_meshes_fit_in_their_boxes(name):
    """Each mesh stays inside its collider's box and fills it, so what the cameras see is what collides."""
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdGeom

    files = T.LOOKS[name] if isinstance(T.LOOKS[name], list) else [T.LOOKS[name]]
    half = np.asarray(SPECS[name].size) / 2
    lo, hi = np.full(3, 1e9), np.full(3, -1e9)
    for item in files:
        file, offset = (item, (0, 0, 0)) if isinstance(item, str) else item
        stage = Usd.Stage.Open(str(HERE / "looks" / file))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        box = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
        lo = np.minimum(lo, np.asarray(box.GetMin()) + offset)
        hi = np.maximum(hi, np.asarray(box.GetMax()) + offset)
    assert np.all(hi <= half + 1e-5) and np.all(lo >= -half - 1e-5)
    assert np.all(hi - lo >= 2 * half - 0.004)
