"""Offline checks (no GPU): the task file loads, the bonds are specified as the trap needs, the scorer covers each
outcome and remembers a lifted plate and a wedged blade, the blade-under-the-edge test, the demo's geometry and reach,
that no arm link passes through the printer, and that every mesh fits its box. Run in the yam_arena environment:
python -m pytest -q test_scrape_print_off_bed.py"""

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
TASK = load(HERE / "scrape_print_off_bed.py")
T = TASK.module
PR = T.pr

FLAT = np.array([0.0, 0.0, 0.0, 1.0])
PRINT_START = np.asarray(T.PRINT.pos)
PLATE_START = np.asarray(T.PLATE.pos)
SCRAPER_START = np.asarray(T.SCRAPER.pos)


def _state(pos, size, vel=(0.0, 0.0, 0.0), quat=FLAT):
    pos = np.asarray(pos, float)
    half = np.asarray(size, float) / 2
    return ObjectState(pos=pos, quat_xyzw=np.asarray(quat, float), lin_vel=np.asarray(vel, float),
                       ang_vel=np.zeros(3), low=pos - half, high=pos + half)


SCRAPER_BOX = (T.SCRAPER_SIZE[1], T.SCRAPER_SIZE[0], T.SCRAPER_SIZE[2])  # lying along x


def _scene(print_part=None, plate=None, scraper=None):
    """The start, with any object replaced by a state or a centre."""
    out = {"print_part": _state(PRINT_START, T.PRINT_SIZE), "plate": _state(PLATE_START, T.PLATE_SIZE),
           "scraper": _state(SCRAPER_START, SCRAPER_BOX)}
    for name, value, size in (("print_part", print_part, T.PRINT_SIZE), ("plate", plate, T.PLATE_SIZE),
                              ("scraper", scraper, SCRAPER_BOX)):
        if value is not None:
            out[name] = value if isinstance(value, ObjectState) else _state(value, size)
    return out


IN_BASKET = (*T.BASKET_XY, T.TOP + 0.005 + T.PRINT_SIZE[2] / 2)
ON_PLATE_ASIDE = PRINT_START + (0.0, 0.06, 0.0)


@pytest.fixture(autouse=True)
def fresh_episode():
    T.reset_memory()


def _wedge():
    T.STATE["wedged"] = True


def test_loads():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {"plate", "print_part", "scraper"}


def test_the_trap_is_specified():
    assert T.PLATE.to == "printer" and not T.PRINTER.free  # the plate is held to the bench through the fixed printer
    assert T.PRINT.to == "plate"
    assert T.PRINT.hold == 60.0 and math.isinf(T.PRINT.twist)  # a gripper twisting the print can't break it
    assert T.PLATE.hold < T.PRINT.hold / 10  # pulled up, the plate's bond goes long before the print's
    plate = {p.name: p for p in T.shapes()["plate"]}["sheet"]
    assert plate.friction == (T.PLATE_FRICTION, T.PLATE_FRICTION) and plate.combine == "min"


def test_layout():
    assert T.PLATE.pos[2] - T.PLATE_SIZE[2] / 2 == pytest.approx(T.BED_TOP)
    assert T.PRINT.pos[2] - T.PRINT_SIZE[2] / 2 == pytest.approx(T.PLATE_TOP)
    assert T.SCRAPER.pos[2] - T.SCRAPER_SIZE[2] / 2 == pytest.approx(T.TOP)
    bed = {p.name: p for p in T.shapes()["printer"]}["bed"]
    assert T.PRINTER.pos[2] + bed.centre[2] + bed.size[2] / 2 == pytest.approx(T.BED_TOP)
    # the print on the plate, its front edge 2.5 cm inside the plate's front edge, inside the door opening's width
    assert T.PRINT_EDGE_X - PR.BED_X[0] == pytest.approx(0.025)
    assert abs(T.PRINT_XY[1]) + T.PRINT_SIZE[1] / 2 < PR.OPENING[0] - 0.05
    # the scraper on the table right of the printer, clear of it; the basket clear of the open door
    tip_x = T.SCRAPER.pos[0] + T.TIP[1]
    assert T.SCRAPER.pos[1] + T.SCRAPER_SIZE[0] / 2 < -PR.SIZE[1] / 2 - 0.05
    assert tip_x < PR.CENTRE[0] - PR.SIZE[0] / 2 + 0.03
    assert T.BASKET_XY[0] + T.BASKET_SIZE[0] / 2 < T.DOOR.pos[0] - T.DOOR.size[0] / 2 - 0.05


def test_open_door_beside_the_printer():
    """Turned 180 degrees about its hinge, the door stands beside the printer's left side, 2 mm or more clear."""
    near_y = T.DOOR.pos[1] - T.DOOR.size[1] / 2
    assert near_y >= PR.SIZE[1] / 2 + 0.002
    assert T.DOOR.pos[0] - T.DOOR.size[0] / 2 == pytest.approx(PR.HINGE[0])


def test_print_parts():
    parts = {p.name: p for p in T.shapes()["print_part"]}
    foot, body = parts["foot"], parts["body"]
    h = T.PRINT_SIZE[2] / 2
    assert foot.centre[2] - foot.size[2] / 2 == pytest.approx(-h)
    assert body.centre[2] - body.size[2] / 2 == pytest.approx(-h + T.FOOT[2])
    assert body.centre[2] + body.size[2] / 2 == pytest.approx(h)
    gap_depth = (T.PRINT_SIZE[0] - T.FOOT[0]) / 2
    assert gap_depth == pytest.approx(0.012)
    for tilt in (T.TILT, math.radians(7)):
        meet = (T.FOOT[2] - T.BLADE[2] / math.cos(tilt)) / math.tan(tilt)
        assert meet < gap_depth - 0.002, tilt


@pytest.mark.parametrize("name", ["printer", "door", "scraper", "print_part", "plate"])
def test_parts_inside_their_boxes(name):
    size = {o.name: o.size for o in T.OBJECTS}[name]
    half = np.asarray(size) / 2
    for p in T.shapes()[name]:
        c, s = np.asarray(p.centre), np.asarray(p.size) / 2
        assert np.all(c + s <= half + 1e-9) and np.all(c - s >= -half - 1e-9), p.name


def test_scraper_parts():
    parts = {p.name: p for p in T.shapes()["scraper"]}
    blade = parts["blade"]
    assert blade.friction == (T.BLADE_FRICTION, T.BLADE_FRICTION) and blade.combine == "min"
    assert blade.centre[1] + blade.size[1] / 2 == pytest.approx(T.TIP[1])
    assert blade.centre[2] - blade.size[2] / 2 == pytest.approx(T.TIP[2])


def test_start_scores_zero():
    assert T.score(_scene()) == 0.0
    assert not T.success(_scene())


def test_wedged_quarter():
    _wedge()
    assert T.score(_scene()) == 0.25


def test_freed_half():
    assert T.score(_scene(print_part=ON_PLATE_ASIDE)) == 0.5
    assert T.score(_scene(print_part=PRINT_START + (0.0, 0.0, 0.08))) == 0.5
    assert T.score(_scene(print_part=PRINT_START + (0.0, 0.03, 0.0))) == 0.0  # 3 cm is not off its place


def test_in_basket_succeeds():
    _wedge()
    objects = _scene(print_part=IN_BASKET)
    assert T.success(objects) and T.score(objects) == 1.0


def test_in_basket_without_the_blade_half():
    objects = _scene(print_part=IN_BASKET)
    assert not T.success(objects) and T.score(objects) == 0.5


def test_in_basket_plate_moved_half():
    _wedge()
    objects = _scene(print_part=IN_BASKET, plate=PLATE_START + (0.006, 0.0, 0.0))
    assert not T.success(objects) and T.score(objects) == 0.5
    objects = _scene(print_part=IN_BASKET, plate=PLATE_START + (0.004, 0.0, 0.0))
    assert T.success(objects)


def test_in_basket_scraper_held_or_in_printer_half():
    _wedge()
    held = SCRAPER_START + (0.0, 0.0, 0.05)
    assert T.score(_scene(print_part=IN_BASKET, scraper=held)) == 0.5
    in_printer = (0.50, 0.0, T.TOP + 0.08)
    assert T.score(_scene(print_part=IN_BASKET, scraper=in_printer)) == 0.5
    moving = _state(SCRAPER_START, SCRAPER_BOX, vel=(0.05, 0.0, 0.0))
    assert T.score(_scene(print_part=IN_BASKET, scraper=moving)) == 0.5
    elsewhere = (0.15, -0.35, SCRAPER_START[2])  # anywhere on the table is fine
    assert T.score(_scene(print_part=IN_BASKET, scraper=elsewhere)) == 1.0


def test_in_basket_still_moving_half():
    _wedge()
    objects = _scene(print_part=_state(IN_BASKET, T.PRINT_SIZE, vel=(0.0, 0.0, -0.1)))
    assert not T.success(objects) and T.score(objects) == 0.5


def test_held_over_the_basket_half():
    _wedge()
    assert T.score(_scene(print_part=(*T.BASKET_XY, T.BASKET_RIM + 0.03))) == 0.5


def test_plate_lifted_is_zero_for_good():
    lifted = PLATE_START + (0.0, 0.0, 0.021)
    _wedge()
    assert T.score(_scene(plate=lifted)) == 0.0
    assert T.STATE["plate_lifted"]
    put_back = _scene(print_part=IN_BASKET)
    assert T.score(put_back) == 0.0 and not T.success(put_back)
    T.reset_memory()
    _wedge()
    assert T.score(put_back) == 1.0


def test_plate_raised_a_little_is_not_lifted():
    objects = _scene(plate=PLATE_START + (0.0, 0.0, 0.015))
    assert T.score(objects) == 0.0 and not T.STATE["plate_lifted"]


def test_fallen_or_non_finite_zero():
    _wedge()
    assert T.score(_scene(print_part=(0.3, 0.0, T.TOP - 0.3))) == 0.0
    assert T.score(_scene(scraper=(0.3, -0.5, T.TOP - 0.3))) == 0.0
    assert T.score(_scene(print_part=(math.nan, 0.0, 0.0))) == 0.0


# -- the blade under the print's edge ------------------------------------------------------------------------------


def _tilted_scraper(tip, tilt=T.TILT):
    """The scraper's centre and orientation with its blade's tip at ``tip``, lying along +x and tilted ``tilt`` tip
    down (its own +y along world +x)."""
    yaw = np.array([0.0, 0.0, math.sin(T.SCRAPER_YAW / 2), math.cos(T.SCRAPER_YAW / 2)])
    down = np.array([0.0, math.sin(tilt / 2), 0.0, math.cos(tilt / 2)])  # about world y: +x end down
    r = T._rot(down) @ T._rot(yaw)
    q = _quat(r)
    centre = np.asarray(tip) - r @ np.asarray(T.TIP)
    return centre, q


def _quat(r):
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    return np.array([(r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w), w])


def test_tilt_puts_the_tip_down():
    centre, q = _tilted_scraper((0.40, -0.02, T.PLATE_TOP))
    tips = T.blade_tips(centre, q)
    assert tips[1] == pytest.approx((0.40, -0.02, T.PLATE_TOP))
    assert centre[2] > T.PLATE_TOP and centre[0] < 0.40


def test_tip_under_the_edge():
    edge = T.PRINT_EDGE_X
    for x, under in ((edge - 0.003, False), (edge + 0.002, False), (edge + 0.0035, True), (edge + 0.011, True)):
        centre, q = _tilted_scraper((x, T.PRINT_XY[1], T.PLATE_TOP))
        assert T.tip_under_print(T.blade_tips(centre, q), PRINT_START, FLAT) == under, x
    centre, q = _tilted_scraper((edge + 0.005, T.PRINT_XY[1], T.PLATE_TOP + 0.006))
    assert not T.tip_under_print(T.blade_tips(centre, q), PRINT_START, FLAT)


def test_tip_matches_the_demo_geometry():
    """``tip_from_tcp`` (used by the demo) agrees with the scraper's own geometry: the TCP 6 mm above the middle of
    the handle."""
    tcp = np.array([0.28, -0.02, 0.9])
    tip = tcp + T.tip_from_tcp()
    centre, q = _tilted_scraper(tip)
    grip_local = np.array([0.0, T.HANDLE_Y, T.GRIP_UP])
    assert centre + T._rot(q) @ grip_local == pytest.approx(tcp, abs=1e-9)


# -- the demo ------------------------------------------------------------------------------------------------------


def _at(steps, label):
    t = 0.0
    for w in steps:
        t += w.duration
        if w.label == label:
            return t, w
    raise KeyError(label)


def test_demo_takes_the_print_once_the_blade_is_out():
    plan = T.demo()
    out, _ = _at(plan["right"], "out of the printer")
    reach_in, _ = _at(plan["left"], "in front of the print")
    assert reach_in > out


def test_demo_blade_path():
    plan = {w.label: w for w in T.demo()["right"]}
    for label in ("tip on the plate", "slide under its edge"):
        tip = np.asarray(plan[label].pos) + T.tip_from_tcp()
        assert tip[2] == pytest.approx(T.PLATE_TOP - T.TIP_PRESS)
        assert tip[1] == pytest.approx(T.PRINT_XY[1])
    down = np.asarray(plan["tip on the plate"].pos) + T.tip_from_tcp()
    push = np.asarray(plan["slide under its edge"].pos) + T.tip_from_tcp()
    assert PR.BED_X[0] < down[0] < T.PRINT_EDGE_X < push[0] < T.PRINT_XY[0]
    # in front of the printer, the held scraper's tip is outside it
    out = np.asarray(plan["in front of the printer"].pos) + T.tip_from_tcp()
    assert out[0] < PR.FRONT - 0.02 and out[2] > T.TOP + PR.BASE_TOP + 0.03


def test_demo_keeps_the_wrists_apart():
    plan = {a: ArmPlan(T.demo()[a], BASES[a]) for a in ("left", "right")}
    dur = max(p.duration for p in plan.values())
    worst = 1.0
    for k in range(int(dur * 20) + 1):
        pos = [plan[a].target(min(k / 20, plan[a].duration))[0] for a in ("left", "right")]
        worst = min(worst, float(np.linalg.norm(pos[0] - pos[1])))
    assert worst > 0.15, worst
    assert max(w.pos[1] for w in T.demo()["right"]) < 0.0


@pytest.mark.parametrize("arm", ["left", "right"])
def test_demo_reachable(arm):
    plan = ArmPlan(T.demo()[arm], BASES[arm])
    for k in range(int(plan.duration * 20) + 1):
        plan.action7(k / 20)
    assert plan.worst_ik_mm < 2.0 and plan.worst_ik_deg < 2.0, (arm, plan.worst_ik_mm, plan.worst_ik_deg)


def _world_boxes():
    """The printer's and the open door's collider boxes in the world: (centre, half size, rotation)."""
    out = []
    for name in ("printer", "door"):
        spec = {o.name: o for o in T.OBJECTS}[name]
        c, s = math.cos(spec.yaw), math.sin(spec.yaw)
        r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for p in T.shapes()[name]:
            out.append((f"{name}.{p.name}", np.asarray(spec.pos) + r @ np.asarray(p.centre), np.asarray(p.size) / 2, r))
    return out


@pytest.mark.parametrize("arm", ["left", "right"])
def test_demo_arm_clears_the_printer(arm):
    """Along the whole demo, the arm's links (link 3 to the gripper's body, 6 cm short of the TCP) stay 2 cm or more
    outside every collider of the printer and the open door. The fingers may enter the door opening."""
    plan = ArmPlan(T.demo()[arm], BASES[arm])
    base = np.asarray(BASES[arm])
    boxes = _world_boxes()
    worst = (1.0, "")
    for k in range(int(plan.duration * 20) + 1):
        action, label = plan.action7(k / 20)
        fk = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, action[:6])})
        pts = [fk[n][:3, 3] + base for n in ("link3", "link4", "link5", "gripper")]
        g = fk["gripper"]
        pts.append(g[:3, 3] + base + g[:3, :3] @ np.array([0.0, 0.0, 0.134 - 0.06]))
        dense = []
        for a, b in zip(pts, pts[1:]):
            dense += [a + (b - a) * s for s in np.linspace(0, 1, 6)]
        for p in dense:
            for name, c, half, r in boxes:
                q = np.abs(r.T @ (p - c)) - half
                d = float(np.linalg.norm(np.maximum(q, 0.0))) if np.any(q > 0) else float(max(q))
                if d < worst[0]:
                    worst = (d, f"{name} at {label}")
    assert worst[0] > 0.02, (arm, worst)


# -- the meshes --------------------------------------------------------------------------------------------------------


def test_every_dressed_box_exists_and_its_mesh_is_there():
    names = {o.name for o in T.OBJECTS}
    for name, file in T.LOOKS.items():
        assert name in names
        assert (HERE / "looks" / file).is_file()


@pytest.mark.parametrize("name", ["printer", "door", "print_head", "plate", "print_part", "scraper"])
def test_meshes_fit_in_their_boxes(name):
    """Each mesh stays inside its collider's box and fills it, so what the cameras see is what collides."""
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(HERE / "looks" / T.LOOKS[name]))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    size = {o.name: o.size for o in T.OBJECTS}[name]
    half = np.asarray(size) / 2
    assert np.all(np.asarray(box.GetMax()) <= half + 1e-5)
    assert np.all(np.asarray(box.GetMin()) >= -half - 1e-5)
    assert np.all(np.asarray(box.GetMax()) - np.asarray(box.GetMin()) >= 2 * half - 0.004)
