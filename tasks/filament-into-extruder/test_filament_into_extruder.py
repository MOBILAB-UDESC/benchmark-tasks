"""Offline checks (no GPU): the task file loads, the feeder's colliders leave exactly the funnel and the bore open
and sit where printer.py says, the filament's links meet end to end and their joints rest at the laid shape, the
filament comes off the front of the spool on the printer's holder, the layout keeps clear of the printer and the arms,
the looks match the boxes, the scorer covers each outcome (in, partly in, at the mouth, held, moving, kinked, torn off
the spool, spool off its rod, fallen), and the demo stays within reach, in each arm's own half and clear of the
printer and the feeder. Run in the yam_arena environment: python -m pytest -q test_filament_into_extruder.py"""

from __future__ import annotations

import ast
import math
import pathlib
import re

import numpy as np
import pytest
from yam_arena.demo import ArmPlan
from yam_arena.station import BASES
from yam_arena.task_file import ObjectState, load
from yam_arena.urdf import quat_xyzw_to_matrix

HERE = pathlib.Path(__file__).resolve().parent
TASK = load(HERE / "filament_into_extruder.py")
T = TASK.module
F = __import__("filament")
PR = T.pr
AXIS = np.array([*T.AXIS, 0.0])
ACROSS = np.array([*T.ACROSS, 0.0])
MOUTH = T.MOUTH_3D
READY = {"left": np.array([0.323, 0.305, 1.008]), "right": np.array([0.323, -0.305, 1.008])}


def _state(pos, quat, vel=(0.0, 0.0, 0.0)):
    pos = np.asarray(pos, float)
    return ObjectState(pos=pos, quat_xyzw=np.asarray(quat, float), lin_vel=np.asarray(vel, float),
                       ang_vel=np.zeros(3), low=pos - 0.01, high=pos + 0.01)


def _spool(dpos=(0.0, 0.0, 0.0)):
    return _state(np.asarray(T.SPOOL.pos) + dpos, F.quat(F._yaw_matrix(T.SPOOL.yaw)))


def _from_links(links, vel=(0.0, 0.0, 0.0)) -> dict:
    out = {link.name: _state(link.pos, link.quat, vel) for link in links}
    out["spool"] = _spool()
    return out


START = _from_links(T.FILAMENT)


def _fed(depth: float, off=(0.0, 0.0), vel=(0.0, 0.0, 0.0)) -> dict:
    """The filament with its end ``depth`` past the mouth (shifted ``off``: across, up), straight back along the
    inlet's axis for 20 cm (as the hands hold it), then a smooth curve that sags down to where it leaves the front of
    the spool, arriving there straight down (so it is still anchored and its length is right)."""
    tip = MOUTH + depth * AXIS + off[0] * ACROSS + np.array([0.0, 0.0, off[1]])
    q = tip - 0.20 * AXIS
    end = T.link_ends(START)[-1, 1]  # the anchor on the spool
    u = np.linspace(0.0, 1.0, 2000)[:, None]

    def connector(k):
        return ((2 * u**3 - 3 * u**2 + 1) * q + (u**3 - 2 * u**2 + u) * k * -AXIS + (-2 * u**3 + 3 * u**2) * end
                + (u**3 - u**2) * k * np.array([0.0, 0.0, -1.0]))

    lo, hi = 0.0, 3.0  # the tangents' length that makes the curve as long as the rest of the filament
    for _ in range(60):
        k = (lo + hi) / 2
        if np.linalg.norm(np.diff(connector(k), axis=0), axis=1).sum() < T.FIL_LEN - 0.20:
            lo = k
        else:
            hi = k
    pts = np.vstack([tip + np.linspace(0.0, 0.20, 200)[:, None] * -AXIS, connector((lo + hi) / 2)[1:]])
    length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    length *= T.FIL_LEN / length[-1]

    def path(s):
        return tuple(float(np.interp(s, length, pts[:, k])) for k in range(3))

    return _from_links(F.lay("fil_", path, T.N_LINKS, T.LINK, T.FIL_D, T.LINK_MASS, (1.0, 0.42, 0.0)), vel)


@pytest.fixture(autouse=True)
def fresh_episode():
    T.reset_memory()


# -- the scene -------------------------------------------------------------------------------------------------------


def test_loads():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {*T.LINKS, "spool"}
    assert len(T.LINKS) == T.N_LINKS == 27 and T.FIL_LEN == pytest.approx(0.675)
    assert {"printer", "door", "print_head", "holder", "feeder", "block", "end_block"} <= {o.name for o in TASK.objects}


def _inside_part(p) -> bool:
    for _, centre, size, quat in T.FEEDER.parts():
        local = quat_xyzw_to_matrix(np.asarray(quat)).T @ (np.asarray(p) - np.asarray(centre))
        if np.all(np.abs(local) <= np.asarray(size) / 2 + 1e-9):
            return True
    return False


def _channel_half(x_from_mouth: float) -> float:
    """The channel's half width at this depth (square), or -1 past its closed end."""
    if x_from_mouth < T.FUNNEL_DEPTH:
        return T.FUNNEL / 2 - (T.FUNNEL - T.BORE) / 2 * x_from_mouth / T.FUNNEL_DEPTH
    return T.BORE / 2 if x_from_mouth < T.CHANNEL else -1.0


def test_inlet_colliders_leave_only_the_channel_open():
    rng = np.random.default_rng(0)
    half = np.asarray(T.FEEDER.size) / 2
    zc = T.FEEDER.inlet_z
    for p in rng.uniform(-half, half, size=(20000, 3)):
        off = max(abs(p[1]), abs(p[2] - zc))
        h = _channel_half(p[0] + half[0])
        if off < h - 1e-5:
            assert not _inside_part(p), p  # the channel is open
        elif off > h + 1e-4:
            assert _inside_part(p), p  # the rest of the feeder is solid


def test_feeder_sits_on_the_roof_at_the_back_right():
    yaw = np.array([0.0, 0.0, math.sin(T.FEEDER_YAW / 2), math.cos(T.FEEDER_YAW / 2)])
    turn = quat_xyzw_to_matrix(yaw)
    mouth = np.asarray(T.FEEDER.pos) + turn @ np.asarray(T.FEEDER.mouth_local())
    assert np.allclose(mouth, MOUTH, atol=1e-9)
    assert np.allclose(turn @ [1.0, 0.0, 0.0], AXIS, atol=1e-9)  # the channel runs along -y
    assert T.INLET_Z - T.TOP == pytest.approx(0.475)
    assert T.FEEDER.pos[2] - T.FEEDER.size[2] / 2 == pytest.approx(T.TOP + PR.ROOF)  # standing on the roof
    assert PR.INLET_X > PR.CENTRE[0] and PR.MOUTH_Y < 0  # behind the roof's middle, right of the midline
    assert -PR.WALL_Y <= PR.FEEDER_LO[1] and PR.FEEDER_HI[0] < PR.CENTRE[0] + PR.SIZE[0] / 2  # over the roof


# -- the filament and the spool --------------------------------------------------------------------------------------


def test_links_meet_end_to_end_along_the_path():
    ends = T.link_ends(START)
    gaps = np.linalg.norm(ends[:-1, 1] - ends[1:, 0], axis=1)
    assert gaps.max() < 2e-4, gaps.max()
    assert np.allclose(ends[0, 0], [*T.TIP0, T.Z_BLOCK], atol=1e-4)  # the free end, on the block's line
    assert np.all(ends[:, :, 2] >= T.Z_TABLE - 2e-5)  # nothing under the table (the chords of the bend, 0.01 mm)
    for link in T.FILAMENT:  # each link on the path, its middle within 2 mm of it
        s = (int(link.name[4:]) + 0.5) * T.LINK
        assert np.linalg.norm(np.asarray(link.pos) - T.centreline(s)) < 2e-3


def test_the_laid_shape_is_gentle_and_at_rest():
    bends = T.laid_bends()
    assert bends.max() < 30.0 and bends.max() < T.KINK / 2, bends.max()
    assert bends[: int(T.DROP[0] / T.LINK) - 1].max() < 1e-3  # straight where the hands hold it
    for j in T.JOINTS:  # each joint's two frames coincide as laid, so its springs start relaxed
        frames = []
        for body, pos, rot in ((j.body0, j.pos0, j.rot0), (j.body1, j.pos1, j.rot1)):
            spec = TASK.objects[[o.name for o in TASK.objects].index(body)]
            p, r = F._frame(spec)
            frames.append((p + r @ np.asarray(pos), r @ F.matrix(rot)))
        assert np.allclose(frames[0][0], frames[1][0], atol=1e-9) and np.allclose(frames[0][1], frames[1][1])
    assert T.JOINTS[-1].body0 == "spool" and T.JOINTS[-1].hold < T.SPOOL.hold


def test_filament_comes_off_the_front_of_the_spool_on_the_holder():
    assert T.attached(START) and T.on_axle(START) and not T.ruined(START)
    end = T.link_ends(START)[-1, 1]
    assert np.allclose(end, T.spool_point(START), atol=1e-9)
    rel = end - np.asarray(T.SPOOL.pos)
    assert rel[0] < 0 and abs(rel[1]) < 1e-9  # in front of the axle, in the spool's middle plane
    assert math.hypot(rel[0], rel[2]) == pytest.approx(PR.WIND_R + T.R_FIL, abs=0.003)  # on the winding
    out = T.link_ends(START)[-1, 0] - end
    assert out[2] < -0.99 * T.LINK  # leaving it straight down
    # the spool on the rod: its axle along y on the rod's axis, its inner flange 2 mm off the stop collar
    assert np.allclose(T.SPOOL.pos, (PR.ROD_X, PR.SPOOL_Y, T.TOP + PR.ROD_H))
    assert T.SPOOL.pos[1] - PR.SPOOL_W / 2 - PR.STOP_Y == pytest.approx(0.002)
    assert PR.ROD_TIP_Y > T.SPOOL.pos[1] + PR.SPOOL_W / 2 + 0.02  # the rod sticks out 2 cm past it
    assert T.SPOOL.pos[2] - PR.SPOOL_R - T.TOP > 0.1  # it hangs clear of the table
    assert T.score(START) == 0.0 and not T.success(START)


def test_layout_stays_clear_of_the_printer_and_the_arms():
    for link in T.FILAMENT:  # the filament runs outside the printer
        assert not PR.inside_footprint(link.pos[0], link.pos[1], 0.01), link.name
    b = np.asarray(T.BLOCK.pos)
    assert b[0] + T.BLOCK_SIZE[1] / 2 < PR.CENTRE[0] - PR.SIZE[0] / 2 - 0.03  # the block in front of the printer
    for arm, ready in READY.items():  # the parked arms clear of the spool, the holder and the filament
        assert abs(ready[1] - T.SPOOL.pos[1]) > PR.SPOOL_W / 2 + 0.03 or abs(ready[0] - PR.ROD_X) > PR.SPOOL_R + 0.05
        for link in T.FILAMENT:
            assert np.linalg.norm(np.asarray(link.pos) - ready) > 0.06, (arm, link.name)


def test_looks_match_the_boxes():
    """make_looks.py models the holder, the feeder, the spool and the links inside boxes of the task file's sizes
    (it checks each mesh against its box when it exports)."""
    text = (HERE / "looks" / "make_looks.py").read_text()

    def const(name):
        return ast.literal_eval(re.search(rf"^{name} = (.+?)(?:  #.*)?$", text, re.M).group(1))

    assert const("SPOOL_SIZE") == pytest.approx(PR.SPOOL_SIZE)
    assert const("LINK_SIZE") == pytest.approx((T.LINK, T.FIL_D, T.FIL_D))
    assert const("BLOCK_SIZE") == pytest.approx(T.BLOCK_SIZE)
    m = re.search(r"^INLET_X, INLET_H, MOUTH_Y = (.+)$", text, re.M).group(1)
    assert ast.literal_eval(m) == pytest.approx((PR.INLET_X, PR.INLET_H, PR.MOUTH_Y))
    m = re.search(r"^ROD_X, ROD_H, ROD_R = (.+)$", text, re.M).group(1)
    assert ast.literal_eval(m) == pytest.approx((PR.ROD_X, PR.ROD_H, PR.ROD_R))
    m = re.search(r"^SPOOL_W, SPOOL_R, FLANGE_T, WIND_R, BORE_R, HUB_R = (.+)$", text, re.M).group(1)
    assert ast.literal_eval(m) == pytest.approx((PR.SPOOL_W, PR.SPOOL_R, PR.FLANGE_T, PR.WIND_R, PR.BORE_R,
                                                 PR.HUB_R))
    for name in ("printer", "door", "head", "plate", "holder", "feeder", "spool", "link", "tip", "block"):
        assert (HERE / "looks" / f"{name}.usdc").is_file(), name


def test_hands_work_in_their_own_halves():
    plans = T.demo()
    for arm, sign in (("right", -1), ("left", 1)):
        for w in plans[arm]:
            assert sign * w.pos[1] > 0.02, (arm, w.label)
    for r, lw in zip(plans["right"][:10], plans["left"][:10]):
        assert np.linalg.norm(np.subtract(r.pos, lw.pos)[:2]) >= 0.15


def test_grips_straddle_the_blocks_on_link_middles():
    pad_half = 0.0132
    lo, hi = T.BLOCK_AT - T.BLOCK_SIZE[0] / 2, T.BLOCK_AT + T.BLOCK_SIZE[0] / 2
    end_hi = T.END_BLOCK_AT + T.BLOCK_SIZE[0] / 2
    assert T.GRIP_R + pad_half + 0.01 < lo and T.GRIP_L - pad_half - 0.01 > hi  # 1 cm clear of the block's ends
    assert T.GRIP_R - pad_half - 0.01 > end_hi  # and of the end block's
    assert T.END_BLOCK_AT - T.BLOCK_SIZE[0] / 2 < -0.01  # the end rests on it, 1 cm or more from its edge
    assert T.GRIP_L + pad_half < T.DROP[0]  # the left pads on the straight, before it drops to the table
    for s in (T.GRIP_R, T.GRIP_L):
        assert (s / T.LINK) % 1 == pytest.approx(0.5)


# -- scoring ---------------------------------------------------------------------------------------------------------


def test_fed_states_are_whole():
    for depth in (-0.02, 0.0, 0.025):
        objects = _fed(depth)
        assert T.attached(objects) and T.bend_angles(objects).max() < T.KINK - 10, depth
        assert T.tip(objects)[0] == pytest.approx(depth, abs=1e-4)


def test_lined_up_short_of_the_mouth_scores_a_quarter():
    assert T.score(_fed(-0.009)) == 0.25
    assert T.score(_fed(-0.02)) == 0.0


@pytest.mark.parametrize("depth", [0.0055, 0.012, 0.019])
def test_partly_in_scores_half(depth):
    assert T.score(_fed(depth)) == 0.5


def test_shallow_in_the_funnel_scores_a_quarter():
    assert T.score(_fed(0.003)) == 0.25


def test_two_cm_in_let_go_and_still_succeeds():
    objects = _fed(0.025)
    assert T.success(objects) and T.score(objects) == 1.0
    assert T.score(_fed(0.0201)) == 1.0


def test_still_held_scores_half():
    T.STATE["held"] = True
    objects = _fed(0.025)
    assert not T.success(objects) and T.score(objects) == 0.5


def test_moving_scores_half():
    objects = _fed(0.025, vel=(0.05, 0.0, 0.0))
    assert not T.success(objects) and T.score(objects) == 0.5


def test_still_over_the_last_second_from_the_track():
    objects = _fed(0.025, vel=(0.05, 0.0, 0.0))  # a rattle in the bore: fast, but going nowhere
    T.STATE["track"] = [T._points(objects)] * (T.STILL_WINDOW + 1)
    assert T.still(objects) and T.score(objects) == 1.0
    moved = T._points(objects).copy()
    moved[0] += [0.0, 0.003, 0.0]  # the end was 3 mm away a second ago
    T.STATE["track"] = [moved] + [T._points(objects)] * T.STILL_WINDOW
    assert not T.still(objects) and T.score(objects) == 0.5


def test_a_link_rolling_about_its_own_axis_is_still():
    objects = _fed(0.025)
    s = objects[T.LINKS[0]]
    axis = F.matrix(s.quat_xyzw)[:, 0]
    objects[T.LINKS[0]] = ObjectState(pos=s.pos, quat_xyzw=s.quat_xyzw, lin_vel=np.full(3, 0.005),
                                      ang_vel=3.0 * axis, low=s.low, high=s.high)
    assert T.still(objects) and T.score(objects) == 1.0
    objects[T.LINKS[0]] = ObjectState(pos=s.pos, quat_xyzw=s.quat_xyzw, lin_vel=np.zeros(3),
                                      ang_vel=np.cross(axis, [0.0, 0.0, 1.0]) * 3.0, low=s.low, high=s.high)
    assert not T.still(objects) and T.score(objects) == 0.5


@pytest.mark.parametrize("off", [(0.004, 0.0), (0.0, -0.004), (0.03, 0.0)])
def test_beside_the_inlet_is_not_in(off):
    assert T.score(_fed(0.025, off)) in (0.0, 0.25)
    assert not T.inserted(_fed(0.025, off), T.PART_DEPTH)


def test_a_kink_scores_zero_and_is_remembered():
    objects = _fed(0.025)
    name = T.LINKS[5]
    s = objects[name]
    turn = F.matrix(s.quat_xyzw) @ quat_xyzw_to_matrix([0.0, 0.0, math.sin(math.radians(35)),
                                                         math.cos(math.radians(35))])
    objects[name] = _state(s.pos, F.quat(turn))  # 70 degrees out of line with its neighbours
    assert T.bend_angles(objects).max() >= T.KINK and T.score(objects) == 0.0
    T.STATE["peak_bend"] = 65.0  # kinked earlier, straightened since
    assert T.score(_fed(0.025)) == 0.0
    T.STATE["peak_bend"] = 45.0
    assert T.score(_fed(0.025)) == 1.0


def test_torn_off_the_spool_scores_zero():
    objects = _fed(0.025)
    last = T.LINKS[-1]
    objects[last] = _state(objects[last].pos + np.array([0.0, 0.0, 0.006]), objects[last].quat_xyzw)
    assert not T.attached(objects) and T.score(objects) == 0.0


def test_spool_off_its_rod_scores_zero():
    objects = _fed(0.025)
    objects["spool"] = _spool((0.0, 0.03, 0.0))
    assert not T.on_axle(objects) and T.score(objects) == 0.0


def test_spool_turned_on_its_rod_carries_the_anchor_round():
    objects = _fed(0.025)
    turn = F.matrix(objects["spool"].quat_xyzw) @ quat_xyzw_to_matrix([math.sin(0.2), 0.0, 0.0, math.cos(0.2)])
    objects["spool"] = _state(T.SPOOL.pos, F.quat(turn))
    assert T.on_axle(objects) and not T.attached(objects)  # the anchor point moved with it


def test_fallen_or_broken_scores_zero():
    objects = _fed(0.025)
    objects[T.LINKS[20]] = _state((0.3, 0.3, T.TOP - 0.3), objects[T.LINKS[20]].quat_xyzw)
    assert T.score(objects) == 0.0
    bad = _fed(0.025)
    bad[T.LINKS[0]].pos[0] = np.nan
    assert T.score(bad) == 0.0 and not T.success(bad)


def test_held_needs_a_gripper_closed_on_it():
    objects = _fed(0.025)
    tip = T.tip_point(objects)
    on = tip - 0.0875 * AXIS - np.array([0.006, 0.0, 0.0])  # a forward grip's TCP, 6 mm behind the filament
    assert T.held_by(objects, {"right_robot": (on, 0.0018)})
    assert T.held_by(objects, {"left_robot": (on + 0.005 * ACROSS, 0.0045)})
    assert not T.held_by(objects, {"right_robot": (on, 0.024)})  # open round it: let go
    assert not T.held_by(objects, {"right_robot": (on, 0.0)})  # closed on nothing
    assert not T.held_by(objects, {"right_robot": (on - 0.02 * ACROSS, 0.0018)})  # closed beside it
    assert not T.held_by(objects, {})


# -- the demo --------------------------------------------------------------------------------------------------------


def _play(arm):
    plan = ArmPlan(T.demo()[arm], BASES[arm])
    qs, t = [], 0.0
    while t < plan.duration + 0.5:
        a, _ = plan.action7(t)
        qs.append((a[:6].copy(), plan.target(t)[2]))
        t += 0.05
    return plan, qs


@pytest.mark.parametrize("arm", ["right", "left"])
def test_demo_within_reach(arm):
    plan, _ = _play(arm)
    assert plan.worst_ik_mm < 2.0 and plan.worst_ik_deg < 2.0, (plan.worst_ik_mm, plan.worst_ik_deg)


def _held_point(w) -> np.ndarray:
    """The filament's axis between the pads of a hand at waypoint ``w`` (6 mm ahead of its TCP along the fingers)."""
    return np.asarray(w.pos) + T.IN_PADS * np.asarray(w.approach)


def test_demo_pushes_the_end_two_and_a_half_cm_in():
    plans = T.demo()
    for arm, s in (("right", T.GRIP_R), ("left", T.GRIP_L)):
        at = {w.label: w for w in plans[arm]}
        picked = _held_point(at["close"]) + np.array([0.0, 0.0, T.SAG[arm]])
        assert np.allclose(picked, [*T.from_tip(s), T.Z_BLOCK], atol=1e-9)  # on the filament, s from its end
        for label, depth in (("line up 1 cm short", -0.01), ("push it in", 0.025)):
            end = _held_point(at[label]) + s * AXIS  # the end, s ahead of the grip along the axis
            depth_now, off = T.in_inlet_frame(end)
            assert depth_now == pytest.approx(depth, abs=1e-6)
            assert off == pytest.approx(math.hypot(*T.AIM), abs=1e-6)
            assert np.allclose(at[label].approach, (1.0, 0.0, 0.0), atol=1e-9)  # fingers forward, pads above, below
            assert abs(at[label].closing[2]) == pytest.approx(1.0)


def _boxes():
    """The fixed boxes a hand must keep out of: the printer's colliders, the feeder and the block (centre, size,
    yaw)."""
    out = [(np.asarray(T.PRINTER.pos) + p.centre, np.asarray(p.size), 0.0) for p in PR.printer_parts(PR.PLATE_TOP)]
    out.append((np.asarray(T.FEEDER.pos), np.asarray(T.FEEDER.size), T.FEEDER_YAW))
    for b in (T.BLOCK, T.END_BLOCK):
        out.append((np.asarray(b.pos), np.asarray(T.BLOCK_SIZE), b.yaw))
    return out


@pytest.mark.parametrize("arm", ["right", "left"])
def test_hands_stay_clear_of_the_printer_and_the_feeder(arm):
    """Points on each gripper, its fingers and its wrist stay 5 mm or more outside the printer, the feeder and the
    block."""
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
    for q, finger in qs[::2]:
        joints = {n: float(v) for n, v in zip(ARM_JOINTS, q)} | {"joint7": finger, "joint8": finger}
        fk = forward_kinematics(YAM_URDF, joints)
        for link, pts in clouds.items():
            world = pts @ fk[link][:3, :3].T + fk[link][:3, 3] + np.asarray(BASES[arm])
            for centre, size, yaw in _boxes():
                c, s = math.cos(yaw), math.sin(yaw)
                turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                local = (world - centre) @ turn
                gap = np.max(np.abs(local) - size / 2, axis=1)  # > 0 outside the box (Chebyshev)
                worst = min(worst, float(gap.min()))
    assert worst >= 0.005, worst
