"""Offline checks (no GPU): the task file loads, the hinge, catch, detent and slide are specified as the task needs,
the door's angle is read relative to the printer, the scorer covers each outcome and remembers a moved printer and a
struck head or vase, the demo's geometry and reach, that no arm link passes through the printer, and that every mesh
fits its box. Run in the yam_arena environment:  python -m pytest -q test_open_printer_door.py"""

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
TASK = load(HERE / "open_printer_door.py")
T = TASK.module
PR = T.pr
SPECS = {o.name: o for o in T.OBJECTS}


def _qz(deg):
    a = math.radians(deg) / 2
    return np.array([0.0, 0.0, math.sin(a), math.cos(a)])


def _state(pos, size, quat=(0.0, 0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0)):
    pos = np.asarray(pos, float)
    half = np.asarray(size, float) / 2
    return ObjectState(pos=pos, quat_xyzw=np.asarray(quat, float), lin_vel=np.asarray(vel, float),
                       ang_vel=np.asarray(spin, float), low=pos - half, high=pos + half)


def _door_at(open_deg, printer_pos=None, printer_yaw=0.0):
    """The door's state open ``open_deg`` degrees about its hinge, on a printer at ``printer_pos`` turned
    ``printer_yaw`` degrees."""
    p0 = np.asarray(T.PRINTER.pos)
    pp = p0 if printer_pos is None else np.asarray(printer_pos, float)
    c, s = math.cos(math.radians(printer_yaw)), math.sin(math.radians(printer_yaw))
    rp = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    hinge = np.array([PR.HINGE[0], PR.HINGE[1], T.DOOR.pos[2]])
    a = -math.radians(open_deg)
    rd = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    centre = hinge + rd @ (np.asarray(T.DOOR.pos) - hinge)
    centre = pp + rp @ (centre - p0)
    return _state(centre, T.DOOR.size, _qz(printer_yaw - open_deg))


def _scene(door=0.0, printer=None, head=None, vase=None, vase_quat=(0, 0, 0, 1.0)):
    out = {"printer": _state(T.PRINTER.pos if printer is None else printer, T.PRINTER.size),
           "door": door if isinstance(door, ObjectState) else _door_at(door),
           "print_head": _state(T.HEAD.pos if head is None else head, T.HEAD.size),
           "vase": _state(T.VASE.pos if vase is None else vase, T.VASE.size, vase_quat)}
    return out


@pytest.fixture(autouse=True)
def fresh_episode():
    T.reset_memory()


def test_loads():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {"printer", "door", "print_head", "vase"}


def test_the_catch_and_the_feet():
    """10 N at the handle opens the catch, the feet slide at about 7 N: a pull alone moves the printer."""
    assert T.DOOR.catch / T.HANDLE_R == pytest.approx(T.CATCH_PULL)
    slide = T.FEET_FRICTION * T.PRINTER_MASS * 9.81
    assert 4.5 < slide < 0.7 * T.CATCH_PULL
    feet = [p for p in T.shapes()["printer"] if p.name.startswith("foot")]
    assert len(feet) == 4 and all(p.friction == (T.FEET_FRICTION,) * 2 and p.combine == "min" for p in feet)


def test_door_zones():
    d = T.DOOR
    assert d.spring_at(-30.0) == 0.0 and d.spring_at(-79.0) == 0.0  # free, only the hinge's friction
    assert d.spring_at(-85.0) == pytest.approx(T.DETENT_RATE * math.radians(-10.0))  # pulled on open, to 95
    assert d.spring_at(-100.0) > 0.0 and d.spring_at(-95.0, -1.0) == pytest.approx(T.DETENT_DAMPING)
    assert d.lower == -T.STOP and d.upper == 0.0 and d.damping == T.HINGE_FRICTION


def test_hinge_on_the_printer_front():
    d = T.DOOR
    assert d.to == "printer" and d.axis == "Z"
    axis = np.asarray(d.pos) + np.asarray(d.hinge)
    assert axis[:2] == pytest.approx(PR.HINGE)
    assert axis[0] < PR.FRONT  # in front of the post, so the door swings outward clear of the frame
    door_left = d.pos[1] + d.size[1] / 2
    assert axis[1] - door_left > 0.005


def test_door_angle_from_quats():
    for deg in (0.0, 10.0, 45.0, 95.0):
        assert T.door_open(_scene(door=deg)) == pytest.approx(deg, abs=1e-6)
    # relative to the printer: a printer turned 3 degrees with the door shut on it reads shut
    turned = _scene(door=_door_at(0.0, printer_yaw=3.0))
    turned["printer"] = _state(T.PRINTER.pos, T.PRINTER.size, _qz(3.0))
    assert T.door_open(turned) == pytest.approx(0.0, abs=1e-6)


def test_start_scores_zero():
    assert T.score(_scene()) == 0.0 and not T.success(_scene())


@pytest.mark.parametrize("deg, expected", [(5.0, 0.0), (10.0, 0.25), (44.0, 0.25), (45.0, 0.5), (89.0, 0.5),
                                           (90.0, 1.0), (95.0, 1.0)])
def test_ladder(deg, expected):
    assert T.score(_scene(door=deg)) == expected


def test_success_needs_the_door_still():
    swinging = _door_at(95.0)
    swinging = ObjectState(pos=swinging.pos, quat_xyzw=swinging.quat_xyzw, lin_vel=np.zeros(3),
                           ang_vel=np.array([0.0, 0.0, -1.0]), low=swinging.low, high=swinging.high)
    assert not T.success(_scene(door=swinging)) and T.score(_scene(door=swinging)) == 0.5


def test_printer_moved_is_zero_for_good():
    moved = np.asarray(T.PRINTER.pos) + (-0.011, 0.0, 0.0)
    assert T.score(_scene(door=95.0, printer=moved)) == 0.0 and T.STATE["moved"]
    assert T.score(_scene(door=95.0)) == 0.0  # put back: still 0
    T.reset_memory()
    assert T.score(_scene(door=95.0)) == 1.0
    d = np.array([-0.009, 0.0, 0.0])  # slid 9 mm with everything on it
    s = _scene(door=_door_at(95.0, printer_pos=np.asarray(T.PRINTER.pos) + d), printer=np.asarray(T.PRINTER.pos) + d,
               head=np.asarray(T.HEAD.pos) + d, vase=np.asarray(T.VASE.pos) + d)
    assert T.score(s) == 1.0


def test_printer_turned_is_moved():
    """Turned 4 degrees about its centre, the base's corners move 1.7 cm."""
    s = _scene(door=_door_at(95.0, printer_yaw=4.0))
    s["printer"] = _state(T.PRINTER.pos, T.PRINTER.size, _qz(4.0))
    assert T.printer_shift(s) > 0.01 and T.score(s) == 0.0


def test_struck_head_or_vase_is_zero_for_good():
    head = np.asarray(T.HEAD.pos) + (0.004, 0.0, 0.0)
    assert T.score(_scene(door=95.0, head=head)) == 0.0 and T.STATE["struck"]
    assert T.score(_scene(door=95.0)) == 0.0
    T.reset_memory()
    assert T.score(_scene(door=95.0, vase=np.asarray(T.VASE.pos) + (0.0, 0.006, 0.0))) == 0.0
    T.reset_memory()
    tipped = (math.sin(math.radians(6)), 0.0, 0.0, math.cos(math.radians(6)))  # 12 degrees about x
    assert T.score(_scene(door=95.0, vase_quat=tipped)) == 0.0
    T.reset_memory()
    small = np.asarray(T.HEAD.pos) + (0.002, 0.0, 0.0)
    assert T.score(_scene(door=95.0, head=small, vase=np.asarray(T.VASE.pos) + (0.004, 0.0, 0.0))) == 1.0


def test_head_and_vase_read_in_the_printers_frame():
    """A printer slid 5 mm with everything on it: the head and the vase have not moved relative to it."""
    d = np.array([-0.005, 0.0, 0.0])
    s = _scene(door=_door_at(0.0, printer_pos=np.asarray(T.PRINTER.pos) + d), printer=np.asarray(T.PRINTER.pos) + d,
               head=np.asarray(T.HEAD.pos) + d, vase=np.asarray(T.VASE.pos) + d)
    assert not T.struck(s) and T.door_open(s) == pytest.approx(0.0, abs=1e-6)


def test_non_finite_zero():
    s = _scene(door=95.0)
    s["printer"] = _state((math.nan, 0, 0), T.PRINTER.size)
    assert T.score(s) == 0.0


def test_layout():
    head = SPECS["print_head"]
    beam = {p.name: p for p in T.shapes()["printer"]}["x_beam"]
    assert head.pos[0] + head.size[0] / 2 < T.PRINTER.pos[0] + beam.centre[0] - beam.size[0] / 2 - 0.001
    vase = SPECS["vase"]
    assert vase.pos[2] - vase.size[2] / 2 == pytest.approx(T.TOP + PR.PLATE_TOP)
    assert vase.pos[2] + vase.size[2] / 2 < head.pos[2] - head.size[2] / 2 - 0.03  # under the head with room
    door = SPECS["door"]
    assert door.pos[2] - door.size[2] / 2 > T.TOP + PR.BASE_TOP + 0.004  # clears the ledge as it swings


@pytest.mark.parametrize("name", ["printer", "door"])
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


def test_handle_path():
    assert T.handle_at(0.0) == pytest.approx(PR.HANDLE)
    r0 = math.dist(T.handle_at(0.0), PR.HINGE)
    assert math.dist(T.handle_at(T.CATCH_OPEN), PR.HINGE) == pytest.approx(r0)


def test_demo_steadies_while_the_catch_opens():
    plan = T.demo()
    pressed = _at(plan["left"], "press the ledge")
    held = _at(plan["left"], "steady the printer")
    gripped = _at(plan["right"], "grip it")
    opened = _at(plan["right"], "pull to 6.0 deg")
    swept = _at(plan["right"], "sweep to 10 deg")
    assert pressed < gripped and held >= opened
    assert _at(plan["left"], "back off the ledge") <= swept - 1.0  # the left hand off before the door sweeps there


def test_demo_wrists_apart():
    plan = {a: ArmPlan(T.demo()[a], BASES[a]) for a in ("left", "right")}
    dur = max(p.duration for p in plan.values())
    worst = 1.0
    for k in range(int(dur * 20) + 1):
        pos = [plan[a].target(min(k / 20, plan[a].duration))[0] for a in ("left", "right")]
        worst = min(worst, float(np.linalg.norm(pos[0] - pos[1])))
    assert worst > 0.15, worst


@pytest.mark.parametrize("arm", ["left", "right"])
def test_demo_reachable(arm):
    plan = ArmPlan(T.demo()[arm], BASES[arm])
    for k in range(int(plan.duration * 20) + 1):
        plan.action7(k / 20)
    assert plan.worst_ik_mm < 2.0 and plan.worst_ik_deg < 2.0, (arm, plan.worst_ik_mm, plan.worst_ik_deg)


@pytest.mark.parametrize("arm", ["left", "right"])
def test_demo_arm_clears_the_printer(arm):
    """Along the whole demo, the arm's links (link 3 to the gripper's body, 6 cm short of the TCP) stay 2 cm or more
    outside every collider of the printer."""
    plan = ArmPlan(T.demo()[arm], BASES[arm])
    base = np.asarray(BASES[arm])
    boxes = [(p.name, np.asarray(T.PRINTER.pos) + np.asarray(p.centre), np.asarray(p.size) / 2)
             for p in T.shapes()["printer"]]
    worst = (1.0, "")
    for k in range(int(plan.duration * 20) + 1):
        action, label = plan.action7(k / 20)
        fk = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, action[:6])})
        pts = [fk[n][:3, 3] + base for n in ("link3", "link4", "link5", "gripper")]
        g = fk["gripper"]
        pts.append(g[:3, 3] + base + g[:3, :3] @ np.array([0.0, 0.0, 0.134 - 0.06]))
        for a, b in zip(pts, pts[1:]):
            for p in (a + (b - a) * s for s in np.linspace(0, 1, 6)):
                for name, c, half in boxes:
                    q = np.abs(p - c) - half
                    d = float(np.linalg.norm(np.maximum(q, 0.0))) if np.any(q > 0) else float(max(q))
                    if d < worst[0]:
                        worst = (d, f"{name} at {label}")
    assert worst[0] > 0.02, (arm, worst)


# -- the meshes --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["printer", "door", "print_head", "vase"])
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
