"""Offline checks (no GPU): the task file loads, the bracket's colliders leave exactly the hole open, the iron rests on
its stand, the looks match the boxes, the scorer covers each outcome (flush and square, flush but tilted or not still or
the iron still on it, within a millimetre, half in, not in, buried, the tip on the bracket's face, fallen, not finite),
and the demo stays within reach and in the right arm's half. Run in the yam_arena environment:
python -m pytest -q test_heat_set_insert_flush.py"""

from __future__ import annotations

import ast
import math
import pathlib

import numpy as np
import pytest
from yam_arena.demo import ArmPlan
from yam_arena.station import BASES
from yam_arena.task_file import ObjectState, load
from yam_arena.urdf import quat_xyzw_to_matrix

HERE = pathlib.Path(__file__).resolve().parent
TASK = load(HERE / "heat_set_insert_flush.py")
T = TASK.module
C = __import__("colliders")


def _quat(axis, deg):
    h = math.radians(deg) / 2
    v = np.zeros(3)
    v["xyz".index(axis)] = math.sin(h)
    return np.array([*v, math.cos(h)])


def _state(pos, quat=(0.0, 0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0)):
    pos = np.asarray(pos, float)
    return ObjectState(pos=pos, quat_xyzw=np.asarray(quat, float), lin_vel=np.asarray(vel, float),
                       ang_vel=np.asarray(spin, float), low=pos - 0.01, high=pos + 0.01)


def _insert(proud=0.0, tilt=0.0, axis="y", vel=(0.0, 0.0, 0.0)):
    """The insert with its top ``proud`` m out of the face (negative: buried), turned ``tilt`` degrees about its centre."""
    q = _quat(axis, tilt)
    rot = quat_xyzw_to_matrix(q)
    top = np.array([T.FACE_X - proud, T.HOLE_Y, T.HOLE_Z])
    return _state(top + T.INS_L / 2 * rot[:, 0], q, vel)


def _iron(tip=None, dy=0.0, dz=0.0, pitch=0.0):
    """The iron with its tip's face at ``tip`` (x) on the hole's line (shifted dy, dz), turned ``pitch`` about y."""
    q = _quat("y", pitch)
    rot = quat_xyzw_to_matrix(q)
    face = np.array([T.FACE_X - 0.05 if tip is None else tip, T.HOLE_Y + dy, T.HOLE_Z + dz])
    return _state(face - rot @ T.TIP_LOCAL, q)


def _objects(insert=None, iron=None):
    return {"insert": insert or _insert(0.008), "iron": iron or _iron()}


FLUSH = _insert(0.0)
CLEAR = _iron(T.FACE_X - 0.05)


@pytest.fixture(autouse=True)
def fresh_episode():
    T.reset_memory()


# -- the scene -------------------------------------------------------------------------------------------------------


def test_loads():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {"insert", "iron"}
    assert {"part", "stand_base"} <= {o.name for o in TASK.objects}


def _inside(parts, centre, p) -> bool:
    for part in parts:
        local = np.asarray(p) - np.asarray(centre) - np.asarray(part.centre)
        if np.all(np.abs(local) <= np.asarray(part.size) / 2 + 1e-9):
            return True
    return False


def test_bracket_colliders_leave_only_the_hole_open():
    parts = C.shapes()["part"]
    half = np.asarray(T.PART_SIZE) / 2
    hole_z = T.HOLE_Z - T.PART_CENTRE[2]
    rng = np.random.default_rng(0)
    for p in rng.uniform(-half, half, size=(20000, 3)):
        depth = p[0] + half[0]
        in_hole = depth < T.HOLE_DEPTH and max(abs(p[1]), abs(p[2] - hole_z)) < T.HOLE_OPEN / 2 - 1e-5
        outside = depth >= T.HOLE_DEPTH or max(abs(p[1]), abs(p[2] - hole_z)) > T.HOLE_OPEN / 2 + 1e-5
        if in_hole:
            assert not _inside(parts, (0, 0, 0), p), p
        elif outside:
            assert _inside(parts, (0, 0, 0), p), p


def test_hole_is_on_the_face_at_the_right_height_and_the_insert_sits_on_its_mouth():
    assert T.PART_CENTRE[0] - T.PART_SIZE[0] / 2 == pytest.approx(T.FACE_X)
    assert T.HOLE_Z - T.TOP == pytest.approx(0.066) and T.HOLE_Z < T.PART_CENTRE[2] + T.PART_SIZE[2] / 2
    start = {"insert": _state(T.INSERT.pos), "iron": _state(T.IRON.pos)}
    assert T.proud(start) == pytest.approx(T.INS_L)
    assert T.square_off(start) == pytest.approx(0.0, abs=1e-6)
    assert T.INSERT.travel == pytest.approx(T.HOLE_DEPTH) and T.INSERT.travel > T.INS_L  # it can be buried 2.5 mm
    assert T.HOLE_OPEN > T.INS_W + 0.001  # room to cock in the hole
    assert T.HOLE_DRAWN < T.INS_W < T.HOLE_OPEN


def test_the_tip_is_narrower_than_the_insert_and_the_drawn_hole():
    """So it can follow the insert into the hole, and never stops at the face."""
    assert T.TIP[1] < T.HOLE_DRAWN < T.INS_W
    assert T.TIP[0] > T.HOLE_DEPTH + 0.02  # the barrel is far behind the face when the insert bottoms out


def test_the_iron_lies_on_its_rest_and_cradle():
    iron_bottom = T.IRON_Z - T.HANDLE[2] / 2
    rest = next(b for b in T.STAND if b.name == "stand_rest")
    assert rest.pos[2] + rest.size[2] / 2 == pytest.approx(iron_bottom)
    barrel_bottom = T.IRON_Z - T.BARREL[2] / 2
    floor = next(b for b in T.STAND if b.name == "stand_cradle_floor")
    assert floor.pos[2] + floor.size[2] / 2 == pytest.approx(barrel_bottom)
    base = next(b for b in T.STAND if b.name == "stand_base")
    assert base.pos[2] + base.size[2] / 2 == pytest.approx(T.BASE_TOP)
    tip_bottom = T.IRON_Z - T.TIP[2] / 2
    assert tip_bottom > T.BASE_TOP + 0.005  # the tip hangs clear of the stand
    # the centre of mass lies between the rest and the cradle
    com = sum(p.size[0] * p.size[1] * p.size[2] * p.centre[0] for p in T.IRON_PARTS) / sum(
        p.size[0] * p.size[1] * p.size[2] for p in T.IRON_PARTS)
    s_rest, s_cradle = np.mean(T.REST_S), np.mean(T.CRADLE_S)
    assert s_rest < com < s_cradle
    assert T.LIFT == pytest.approx(0.011)


def test_the_stand_has_no_lips():
    """The pads' tips go 1.3 cm below the TCP and a yawed iron's barrel would catch on cradle lips: it is all flat."""
    assert not any("lip" in b.name for b in T.STAND)


def test_iron_and_insert_are_inside_their_boxes():
    for part in T.IRON_PARTS:
        assert np.all(np.abs(part.centre) + np.asarray(part.size) / 2 <= np.asarray(T.IRON_SIZE) / 2 + 1e-9)
    assert T.TIP_LOCAL[0] == pytest.approx(T.IRON_SIZE[0] / 2)


def _constants(path: pathlib.Path) -> dict:
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass
    return out


def test_looks_match_the_boxes():
    k = _constants(HERE / "looks" / "make_looks.py")
    assert tuple(k["PART_SIZE"]) == pytest.approx(T.PART_SIZE)
    assert k["HOLE_DRAWN"] == pytest.approx(T.HOLE_DRAWN) and k["HOLE_DEPTH"] == pytest.approx(T.HOLE_DEPTH)
    assert k["HOLE_Z"] == pytest.approx(T.HOLE_Z - T.PART_CENTRE[2])
    assert (k["INS_L"], k["INS_W"]) == pytest.approx((T.INS_L, T.INS_W))
    assert tuple(k["IRON_SIZE"]) == pytest.approx(T.IRON_SIZE)
    assert tuple(k["STAND_BASE"]) == pytest.approx(T.STAND_BASE)
    assert tuple(k["REST_S"]) == pytest.approx(T.REST_S) and tuple(k["CRADLE_S"]) == pytest.approx(T.CRADLE_S)
    assert k["IRON_ABOVE_BASE"] == pytest.approx(T.IRON_Z - (T.TOP + T.STAND_BASE[2] / 2))
    for name in ("part", "insert", "iron", "stand"):
        assert (HERE / "looks" / f"{name}.usdc").is_file()


# -- scoring: every outcome ------------------------------------------------------------------------------------------


def test_the_start_scores_zero():
    assert T.score(_objects()) == 0.0 and not T.success(_objects())


def test_flush_square_still_and_clear_scores_one():
    o = _objects(FLUSH, CLEAR)
    assert T.success(o) and T.score(o) == 1.0


@pytest.mark.parametrize("p", [-0.00029, -0.0002, 0.0002, 0.00029])
def test_flush_within_three_tenths_of_a_millimetre(p):
    assert T.score(_objects(_insert(p), CLEAR)) == 1.0


@pytest.mark.parametrize("p", [-0.0006, 0.0006, 0.00099, -0.00099])
def test_within_a_millimetre_scores_half(p):
    assert T.score(_objects(_insert(p), CLEAR)) == 0.5


def test_tilted_more_than_three_degrees_scores_half_and_more_than_eight_a_quarter():
    assert T.score(_objects(_insert(0.0, 2.9), CLEAR)) == 1.0
    assert T.score(_objects(_insert(0.0, 3.5), CLEAR)) == 0.5
    assert T.score(_objects(_insert(0.0, 7.5, axis="z"), CLEAR)) == 0.5
    assert T.score(_objects(_insert(0.0, 8.5), CLEAR)) == 0.25
    assert T.score(_objects(_insert(0.0, -8.5, axis="z"), CLEAR)) == 0.25


def test_flush_but_not_still_scores_half_not_one():
    moving = _insert(0.0, vel=(0.05, 0.0, 0.0))
    assert T.score(_objects(moving, CLEAR)) == 0.5 and not T.success(_objects(moving, CLEAR))


def test_the_iron_still_on_the_insert_scores_half_not_one():
    pressing = _iron(T.FACE_X)
    assert not T.success(_objects(FLUSH, pressing)) and T.score(_objects(FLUSH, pressing)) == 0.5
    near = _iron(T.FACE_X - 0.009)
    assert T.score(_objects(FLUSH, near)) == 0.5
    far = _iron(T.FACE_X - 0.011)
    assert T.score(_objects(FLUSH, far)) == 1.0


@pytest.mark.parametrize("p,expected", [(0.0041, 0.0), (0.0039, 0.25), (0.002, 0.25), (0.00101, 0.25), (0.008, 0.0)])
def test_half_in_scores_a_quarter_and_less_scores_nothing(p, expected):
    assert T.score(_objects(_insert(p), CLEAR)) == expected


def test_buried_more_than_a_millimetre_scores_zero():
    assert T.score(_objects(_insert(-0.00101), CLEAR)) == 0.0
    assert T.score(_objects(_insert(-0.0025), CLEAR)) == 0.0
    assert T.score(_objects(_insert(-0.0009), CLEAR)) == 0.5


def test_the_tip_on_the_face_outside_the_hole_scores_zero_for_the_rest_of_the_trial():
    o = _objects(FLUSH, _iron(T.FACE_X, dy=0.004))  # the pin's edge 5.3 mm out: on the bracket
    assert T.touching_face(o) and T.score(o) == 0.0
    assert T.STATE["scorched"]
    assert T.score(_objects(FLUSH, CLEAR)) == 0.0  # and the iron's taken away


def test_the_tip_off_to_the_side_but_short_of_the_face_is_not_touching():
    assert not T.touching_face(_objects(iron=_iron(T.FACE_X - 0.002, dy=0.006)))
    assert T.touching_face(_objects(iron=_iron(T.FACE_X - 0.0004, dy=0.006)))
    assert T.touching_face(_objects(iron=_iron(T.FACE_X + 0.002, dz=-0.004)))


def test_the_tip_inside_the_hole_is_not_touching():
    for dy, dz in ((0.0, 0.0), (0.0009, 0.0), (0.0, -0.0009), (-0.001, 0.001)):
        assert not T.touching_face(_objects(iron=_iron(T.FACE_X + 0.004, dy=dy, dz=dz)))
    assert T.touching_face(_objects(iron=_iron(T.FACE_X + 0.004, dy=0.0018)))  # the pin's edge past 3 mm


def test_observe_remembers_a_touch_even_if_the_iron_then_leaves():
    T.observe(_objects(iron=_iron(T.FACE_X, dy=0.004)))
    assert T.STATE["scorched"]
    assert T.score(_objects(FLUSH, CLEAR)) == 0.0
    T.reset_memory()
    assert T.score(_objects(FLUSH, CLEAR)) == 1.0


def test_a_fallen_or_non_finite_object_scores_zero():
    off = _state((T.FACE_X, T.HOLE_Y, T.TOP - 0.05))
    assert T.score(_objects(FLUSH, off)) == 0.0
    nan = _state((np.nan, 0.0, T.HOLE_Z))
    assert T.score(_objects(nan, CLEAR)) == 0.0
    assert T.score(_objects(FLUSH, nan)) == 0.0


def test_distance_is_how_far_from_flush():
    assert T.distance(_objects(_insert(0.003), CLEAR)) == pytest.approx(0.003)
    assert T.distance(_objects(_insert(-0.002), CLEAR)) == pytest.approx(0.002)


# -- the demo --------------------------------------------------------------------------------------------------------


def _plan():
    return ArmPlan(list(T.demo()["right"]), BASES["right"])


def test_demo_is_reachable():
    plan = _plan()
    assert plan.worst_ik_mm < 2.0


def test_demo_is_one_arm_in_its_own_half_and_the_tip_ends_flush():
    d = T.demo()
    assert set(d) == {"right"}
    assert all(w.pos[1] < 0 for w in d["right"])
    tip_x = T.FACE_X + T.FLUSH_TRIM
    push = next(w for w in d["right"] if w.label == "push it in")
    assert push.pos[0] == pytest.approx(tip_x - T.TIP_FROM_GRIP * math.cos(math.radians(T.PITCH_TRIM)) + T.AIM[0])
    first = d["right"][0]
    assert abs(first.pos[0] - (T.STAND_XY[0] + T.GRIP_S)) < 1e-9 and abs(first.pos[1] - T.STAND_XY[1]) < 1e-9


def test_demo_lifts_the_iron_to_the_hole_height_before_it_moves():
    d = T.demo()["right"]
    lift = next(w for w in d if w.label == "lift")
    assert lift.pos[2] == pytest.approx(T.HOLE_Z + T.GRIP_DZ)
    over = [w for w in d if w.label in ("across to the hole's line",)]
    assert all(w.pos[2] == pytest.approx(T.HOLE_Z + T.GRIP_DZ) for w in over)


def test_cheat_variants_are_off_by_default():
    assert (T.OVERSHOOT, T.TILT_DEG, T.OFFSET_Y, T.SHORT, T.HOLD_ON) == (0.0, 0.0, 0.0, 0.0, 0)
