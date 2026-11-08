"""Offline checks (no GPU): the task file loads, the spool's bore is open round the rod and the spool stands flat on
a rim facet, the holder sits on the printer's left wall where printer.py says, the looks match the boxes, every
scoring outcome and the let-go check, the demo's grips, reach and the hand-over (the spool out of the left arm's
reach, the holder out of the right arm's), and the hands' clearance from the printer. Run in the yam_arena
environment: python -m pytest -q test_spool_onto_printer_holder.py"""

from __future__ import annotations

import ast
import math
import pathlib
import re

import numpy as np
import pytest
from yam_arena import ik
from yam_arena.demo import ArmPlan
from yam_arena.station import BASES
from yam_arena.task_file import ObjectState, load

HERE = pathlib.Path(__file__).resolve().parent
TASK = load(HERE / "spool_onto_printer_holder.py")
T = TASK.module
PR = T.pr
G = __import__("grip")
SEAT = np.array([PR.ROD_X, PR.STOP_Y + PR.SPOOL_W / 2, T.ROD_Z + PR.BORE_R - PR.ROD_R])  # resting on the rod


def _q(axis_deg_from_y=0.0, about="z"):
    """The spool's turn: its axis (box x) along y, then tilted ``axis_deg_from_y`` about ``about``."""
    base = np.array([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])  # yaw 90: x along y
    a = math.radians(axis_deg_from_y)
    axis = {"z": (0, 0, 1), "x": (1, 0, 0)}[about]
    extra = np.array([*(math.sin(a / 2) * np.array(axis)), math.cos(a / 2)])
    x1, y1, z1, w1 = extra
    x2, y2, z2, w2 = base
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def _spool(pos, quat=None, vel=(0.0, 0.0, 0.0)):
    pos = np.array(pos, float)
    return {"spool": ObjectState(pos=pos, quat_xyzw=_q() if quat is None else np.asarray(quat, float),
                                 lin_vel=np.asarray(vel, float), ang_vel=np.zeros(3), low=pos - 0.1, high=pos + 0.1)}


@pytest.fixture(autouse=True)
def fresh_episode():
    T.reset_memory()


# -- the scene -------------------------------------------------------------------------------------------------------


def test_loads():
    assert TASK.scene == "station"
    assert list(TASK.free_objects) == ["spool"]
    assert {"printer", "door", "print_head", "feeder", "holder"} <= {o.name for o in TASK.objects}


def test_bore_is_open_round_the_rod():
    """No spool collider comes within 3 mm of the rod when the spool sits centred on it (the bore's flats are 26 mm
    from the axis, the rod 22.5 mm)."""
    for _, centre, turn, size in T.SPOOL.parts():
        radial_in = math.hypot(centre[1], centre[2]) - size[1] / 2
        assert radial_in >= PR.BORE_R - 1e-9 and radial_in - PR.ROD_R >= 0.003


def test_spool_stands_on_a_flat_and_matches_its_look():
    downs = [c for _, c, turn, _ in T.SPOOL.parts() if abs(turn - 1.5 * math.pi) < 1e-9]  # segments pointing down
    assert downs, "a segment of each ring points straight down: the spool rests on a facet"
    assert T.SPOOL.pos[2] - T.TOP == pytest.approx(PR.SPOOL_R)
    assert T.SPOOL.size == pytest.approx((PR.SPOOL_W, PR.SPOOL_D, PR.SPOOL_D))


def test_holder_on_the_left_wall():
    assert PR.HOLDER_LO[1] == pytest.approx(PR.WALL_Y)  # the bracket on the wall's outside
    assert PR.CENTRE[0] - PR.SIZE[0] / 2 < PR.HOLDER_LO[0] and PR.HOLDER_HI[0] < PR.CENTRE[0] + PR.SIZE[0] / 2
    assert PR.ROD_TIP_Y - (PR.STOP_Y + PR.SPOOL_W) > 0.02  # the rod sticks out 2 cm past a seated spool
    assert PR.STOP_Y - PR.WALL_Y >= 0.035  # room for a finger between the spool and the wall
    assert PR.ROD_H - PR.SPOOL_R > 0.1  # a hung spool clears the table


def test_looks_match_the_boxes():
    text = (HERE / "looks" / "make_looks.py").read_text()
    for name, want in (("ROD_X, ROD_H, ROD_R", (PR.ROD_X, PR.ROD_H, PR.ROD_R)),
                       ("COLLAR_R, COLLAR_T", (PR.COLLAR_R, PR.COLLAR_T)),
                       ("ROD_TIP_Y, NOSE", (PR.ROD_TIP_Y, PR.NOSE)),
                       ("BRACKET_T, BRACKET", (PR.BRACKET_T, PR.BRACKET)),
                       ("SPOOL_W, SPOOL_R, FLANGE_T, WIND_R, BORE_R, HUB_R",
                        (PR.SPOOL_W, PR.SPOOL_R, PR.FLANGE_T, PR.WIND_R, PR.BORE_R, PR.HUB_R))):
        got = ast.literal_eval(re.search(rf"^{re.escape(name)} = (.+)$", text, re.M).group(1))
        assert np.allclose(np.hstack(got), np.hstack(want)), name
    for name in ("printer", "door", "head", "plate", "holder", "feeder", "spool"):
        assert (HERE / "looks" / f"{name}.usdc").is_file(), name


# -- scoring ---------------------------------------------------------------------------------------------------------


def test_start_scores_zero():
    start = _spool(T.SPOOL.pos)
    assert T.score(start) == 0.0 and not T.on_rod(start) and not T.passed_over(start)


def test_seated_let_go_and_still_succeeds():
    objects = _spool(SEAT)
    assert T.seated(objects) and T.success(objects) and T.score(objects) == 1.0


def test_seated_but_held_or_moving_scores_half():
    T.STATE["held"] = True
    assert T.score(_spool(SEAT)) == 0.5
    T.STATE["held"] = False
    assert T.score(_spool(SEAT, vel=(0.0, 0.03, 0.0))) == 0.5


def test_on_the_rod_short_of_the_stop_scores_half():
    objects = _spool(SEAT + np.array([0.0, 0.02, 0.0]))
    assert T.on_rod(objects) and not T.seated(objects) and T.score(objects) == 0.5


def test_askew_or_off_the_rod_is_not_on_it():
    assert not T.on_rod(_spool(SEAT, _q(15.0)))  # turned 15 degrees about the vertical
    assert not T.on_rod(_spool(SEAT + np.array([0.03, 0.0, 0.0])))  # beside the rod
    beyond = SEAT + np.array([0.0, PR.ROD_TIP_Y - PR.STOP_Y - PR.SPOOL_W + 0.01, 0.0])
    assert not T.on_rod(_spool(beyond))  # its outer face past the rod's tip


def test_lined_up_at_the_tip_scores_a_quarter():
    objects = _spool((PR.ROD_X, T.LINE_Y, T.ROD_Z))
    assert T.at_tip(objects) and not T.on_rod(objects) and T.score(objects) == 0.25


def test_passed_over_scores_a_tenth():
    mid = np.array([*T.MID_XY, T.SPOOL.pos[2]])
    assert T.score(_spool(mid)) == 0.1
    assert T.score(_spool(mid, _q(80.0, "x"))) == 0.0  # lying flat
    assert T.score(_spool(np.asarray(T.SPOOL.pos) + np.array([0.0, 0.10, 0.0]))) == 0.0  # not far enough


def test_fallen_or_broken_scores_zero():
    assert T.score(_spool((0.3, 0.5, T.TOP - 0.3))) == 0.0
    bad = _spool(SEAT)
    bad["spool"].pos[0] = np.nan
    assert T.score(bad) == 0.0 and not T.success(bad)


def test_fingers_on_the_spool():
    objects = _spool(SEAT)
    top = SEAT + np.array([0.0, PR.SPOOL_W / 2 + 0.003, PR.SPOOL_R - 0.01])  # 3 mm off the outer flange, at the top
    assert T.fingers_on(objects, np.array([top]))
    assert not T.fingers_on(objects, np.array([top + np.array([0.0, 0.01, 0.0])]))
    assert not T.fingers_on(objects, None) and not T.fingers_on(objects, np.zeros((0, 3)))


# -- the demo --------------------------------------------------------------------------------------------------------


def _play(arm):
    plan = ArmPlan(T.demo()[arm], BASES[arm])
    qs, t = [], 0.0
    while t < plan.duration + 0.5:
        a, _ = plan.action7(t)
        qs.append((a[:6].copy(), plan.target(t)[2], t))
        t += 0.05
    return plan, qs


@pytest.mark.parametrize("arm", ["right", "left"])
def test_demo_within_reach(arm):
    plan, _ = _play(arm)
    assert plan.worst_ik_mm < 2.0 and plan.worst_ik_deg < 2.0, (plan.worst_ik_mm, plan.worst_ik_deg)


def _best_reach(arm, pos, g) -> float:
    rot = G.hand_rotation(g["approach"], g["closing"])
    best = 1e9
    for q0 in ([0.0, 1.0, 1.0, -0.6, 0.0, 0.0], [0.0, 1.5, 1.2, 0.0, 1.5, 0.0], [0.0, 2.0, 2.0, 0.5, 0.0, 1.5],
               [0.0, 2.0, 2.0, 0.5, 0.0, -1.5], [0.0, 1.5, 1.2, 0.0, 0.0, 0.0]):
        _, pe, re_ = ik.solve(np.asarray(pos), ik.matrix_to_quat_xyzw(rot), q0, base_pos=np.asarray(BASES[arm]),
                              iters=300)
        best = min(best, pe * 1000 + math.degrees(re_))
    return best


def test_it_takes_both_hands():
    """The spool starts in the right arm's half, and the right arm can't reach the rod with any of the demo's grips
    (more than 2 cm off), so the left hand hangs it."""
    assert T.SPOOL.pos[1] + PR.SPOOL_W / 2 < -0.1
    for pitch in (0.0, 30.0, 60.0, 90.0):
        for k in (1.0, -1.0):
            p = math.radians(pitch)
            g = {"approach": (math.sin(p), 0.0, -math.cos(p)), "closing": (0.0, k, 0.0)}
            for y in (T.LINE_Y, T.SEAT_Y):
                assert _best_reach("right", T.hand_at((PR.ROD_X, y, T.ROD_Z), "right"), g) > 20.0, (pitch, y)


def test_hand_over_in_the_middle_and_hands_apart():
    plans = T.demo()
    right = {w.label: w for w in plans["right"]}
    left = {w.label: w for w in plans["left"]}
    assert np.allclose(right["set it down"].pos[:2], left["take it"].pos[:2])  # the left takes it where it was left
    t_away = sum(w.duration for w in plans["right"][:[w.label for w in plans["right"]].index("away") + 1])
    t_over = plans["left"][0].duration  # the left hand waits out the right hand's moves
    assert t_over >= t_away - 1.2
    for w in plans["left"][1:]:
        assert w.pos[1] > -0.02, w.label
    _, rq = _play("right")
    _, lq = _play("left")
    from yam_arena.urdf import forward_kinematics
    from yam_arena import YAM_URDF
    from yam_arena.constants import ARM_JOINTS

    def wrist(q, arm):
        fk = forward_kinematics(YAM_URDF, dict(zip(ARM_JOINTS, q)) | {"joint7": 0.0, "joint8": 0.0})
        return fk["gripper"][:3, 3] + np.asarray(BASES[arm])

    closest = min(np.linalg.norm(wrist(a[0], "right") - wrist(b[0], "left")) for a, b in zip(rq, lq))
    assert closest >= 0.15, closest


def test_spool_lined_up_with_the_rod_before_the_slide():
    plans = T.demo()
    left = {w.label: w for w in plans["left"]}
    for label, y in (("line up with the rod", T.LINE_Y), ("slide it on to the stop", T.SEAT_Y)):
        pos = np.asarray(left[label].pos)
        centre = pos - np.array([0.0, 0.0, PR.SPOOL_R - T.GRIP_DOWN]) - np.array(T.LEFT_HOLD)
        assert centre == pytest.approx((PR.ROD_X, y, T.ROD_Z))
    assert T.LINE_Y - PR.SPOOL_W / 2 > PR.ROD_TIP_Y  # clear of the rod's tip before the slide
    assert T.SEAT_Y - PR.SPOOL_W / 2 < PR.STOP_Y  # leaned on the stop


def _printer_boxes():
    """The printer's and its door's colliders as (low, high) corners in the station frame."""
    out = []
    for centre, parts in ((PR.CENTRE, PR.printer_parts(PR.PLATE_TOP)), (PR.DOOR_CENTRE, PR.door_parts())):
        for p in parts:
            c = np.array([centre[0], centre[1], T.TOP + centre[2]]) + np.asarray(p.centre)
            out.append((c - np.asarray(p.size) / 2, c + np.asarray(p.size) / 2))
    return out


def _rim_points(centre) -> np.ndarray:
    """Points round both flanges' rims and on the spool's faces, its axis along y."""
    a = np.linspace(0, 2 * math.pi, 72, endpoint=False)
    pts = []
    for r in (PR.SPOOL_R, 0.6 * PR.SPOOL_R):
        for y in (-PR.SPOOL_W / 2, PR.SPOOL_W / 2):
            pts.append(np.column_stack([r * np.cos(a), np.full_like(a, y), r * np.sin(a)]))
    return np.vstack(pts) + np.asarray(centre)


def _clear(points, margin) -> float:
    worst = 1.0
    for lo, hi in _printer_boxes():
        gap = np.max(np.maximum(lo - points, points - hi), axis=1)
        worst = min(worst, float(gap.min()))
    return worst


def test_spool_clears_the_printer_where_it_stands_and_where_it_is_carried():
    """The spool where it starts and where it is set down, and where the demo's hands carry it (its centre below
    the TCP by the grip), stays 8 mm or more clear of the printer and its door."""
    assert _clear(_rim_points(T.SPOOL.pos), 0.0) >= 0.008
    assert _clear(_rim_points((*T.MID_XY, T.SPOOL.pos[2])), 0.0) >= 0.008
    for arm in ("right", "left"):
        plan = ArmPlan(T.demo()[arm], BASES[arm])
        t = 0.0
        while t < plan.duration:
            pos, _, finger, label = plan.target(t)
            t += 0.05
            if finger > -0.005 and label not in ("let go", "settle", "slide it on to the stop", "hold"):
                give = np.array(T.LEFT_HOLD) if arm == "left" else np.array([0.0, 0.0, T.SAG])
                centre = np.asarray(pos) - np.array([0.0, 0.0, PR.SPOOL_R - T.GRIP_DOWN]) - give
                assert _clear(_rim_points(centre), 0.0) >= 0.008, (arm, label)


def _boxes():
    out = [(np.asarray(T.PRINTER.pos) + p.centre, np.asarray(p.size), 0.0) for p in PR.printer_parts(PR.PLATE_TOP)]
    out.append((np.asarray(T.FEEDER.pos), np.asarray(T.FEEDER.size), T.FEEDER.yaw))
    b = PR.bracket_part()
    out.append((np.asarray(T.HOLDER.pos) + b.centre, np.asarray(b.size), 0.0))
    return out


@pytest.mark.parametrize("arm", ["right", "left"])
def test_hands_stay_clear_of_the_printer(arm):
    """Points on each gripper, its fingers and its wrist stay 5 mm or more outside the printer, the feeder and the
    holder's bracket."""
    from fingers import cloud
    from yam_arena import YAM_URDF
    from yam_arena.constants import ARM_JOINTS
    from yam_arena.urdf import forward_kinematics

    try:
        clouds = {link: cloud(link) for link in ("gripper", "tip_left", "tip_right", "link5", "link4")}
    except (FileNotFoundError, OSError):
        pytest.skip("the URDF's collision meshes are not here")
    _, qs = _play(arm)
    worst = 1.0
    for q, finger, _ in qs[::2]:
        joints = {n: float(v) for n, v in zip(ARM_JOINTS, q)} | {"joint7": finger, "joint8": finger}
        fk = forward_kinematics(YAM_URDF, joints)
        for link, pts in clouds.items():
            world = pts @ fk[link][:3, :3].T + fk[link][:3, 3] + np.asarray(BASES[arm])
            for centre, size, yaw in _boxes():
                c, s = math.cos(yaw), math.sin(yaw)
                turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                gap = np.max(np.abs((world - centre) @ turn) - size / 2, axis=1)
                worst = min(worst, float(gap.min()))
    assert worst >= 0.005, worst
