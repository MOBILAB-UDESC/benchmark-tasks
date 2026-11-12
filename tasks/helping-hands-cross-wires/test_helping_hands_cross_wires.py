"""Offline checks (no GPU): the task file loads, the stands' geometry is consistent (hinges meet, the start is far from the
crossing, the final pose crosses), the scorer covers each outcome and remembers a wire pulled out, the hinges and the
bond are specified as the trap needs, the drag keeps clear of the arm's limits, and both arms' demos stay within reach.
Run in the yam_arena environment:  python -m pytest -q test_helping_hands_cross_wires.py"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
from yam_arena.demo import ArmPlan
from yam_arena.station import BASES
from yam_arena.task_file import ObjectState, load

HERE = pathlib.Path(__file__).resolve().parent
TASK = load(HERE / "helping_hands_cross_wires.py")
T = TASK.module
SIDES = ("right", "left")


def _quat(yaw):
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def _state(pos, yaw, vel=0.0):
    pos = np.asarray(pos, float)
    return ObjectState(pos=pos, quat_xyzw=_quat(yaw), lin_vel=np.array([vel, 0.0, 0.0]), ang_vel=np.zeros(3),
                       low=pos - 0.01, high=pos + 0.01)


def scene(right=None, left=None, shift_right=(0.0, 0.0), shift_left=(0.0, 0.0), vel=0.0):
    """Every moving part for the two stands' hinge angles (the start by default), the clip link and wire of each stand moved together by a shift (x, y)."""
    out = {}
    for side, angles, shift in (("right", right or T.START["right"], shift_right),
                                ("left", left or T.START["left"], shift_left)):
        for name, (pos, yaw, _) in T.parts(side, angles).items():
            p = np.asarray(pos, float)
            if name.endswith(("wire", "link3")):
                p[:2] += shift
            out[name] = _state(p, yaw, vel)
    return out


def final(**kw):
    return scene(T.FINAL["right"], T.FINAL["left"], **kw)


@pytest.fixture(autouse=True)
def fresh_episode():
    T.reset_memory()


def held(scenes, clear=True, held_s=3.0, touch=True):
    T.STATE.update(clear=clear, touch=touch, held=held_s)
    return scenes


def with_height(scenes, name, dz):
    """The scene with one part lifted by ``dz`` metres."""
    p = scenes[name]
    scenes[name] = _state(p.pos + np.array([0.0, 0.0, dz]), 2 * math.atan2(p.quat_xyzw[2], p.quat_xyzw[3]))
    return scenes


def test_loads():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {f"{s}_{p}" for s in SIDES for p in ("link1", "link2", "link3", "wire")}
    assert TASK.success_hold_s == 3.0


def test_hinges_meet_and_start_is_the_layout():
    for side in SIDES:
        p0, p1, p2, (a1, a2, a3) = T.hinges(side, T.START[side])
        spec = {o.name: o for o in T.OBJECTS}
        for i, (hinge, nxt) in enumerate(zip((p0, p1, p2), (p1, p2, None)), start=1):
            link = spec[f"{side}_link{i}"]
            assert np.allclose(link.anchor[:2], hinge)
            assert link.anchor[2] == pytest.approx(T.LINK_Z[side])
            assert link.to == (f"{side}_link{i - 1}" if i > 1 else f"{side}_base")
        for name, (pos, yaw, size) in T.parts(side, T.START[side]).items():
            assert tuple(spec[name].pos) == pytest.approx(pos)
            assert spec[name].yaw == pytest.approx(yaw)


def test_the_start_is_far_from_the_crossing():
    for side in SIDES:
        tip = T.wire_end(side, T.START[side])
        assert np.linalg.norm(tip - np.asarray(T.MARK)) > 0.10
    assert T.score(scene()) == 0.0
    assert not T.success(scene())


def test_final_pose_crosses_over_the_mark():
    for side in SIDES:
        assert np.allclose(T.wire_end(side, T.FINAL[side]), T.FINAL_TIP[side], atol=1e-9)
    overlap, side, mid = T.crossing(final())
    assert overlap == pytest.approx(2 * T.HALF_OVERLAP)
    assert side < 1e-9 and mid < 1e-9


def test_the_elbows_stay_on_the_starts_side():
    for side in SIDES:
        sign = np.sign(T.START[side][1])
        assert np.sign(T.FINAL[side][1]) == sign


def test_trap_is_specified():
    spec = {o.name: o for o in T.OBJECTS}
    for side in SIDES:
        assert not spec[f"{side}_base"].free  # the stand is on the bench
        for i in (1, 2, 3):
            h = spec[f"{side}_link{i}"]
            assert h.hold == pytest.approx(T.HOLD[i - 1]) and h.hold > 0
            assert h.lower < 0 < h.upper
            assert h.upper - h.lower == pytest.approx(2 * T.JOINT_RANGE)
        w = spec[f"{side}_wire"]
        assert w.to == f"{side}_link3" and w.hold == T.WIRE_HOLD and math.isinf(w.twist)


def test_wires_touch_by_design_the_left_stand_is_one_wire_diameter_higher():
    """The centres' heights differ by exactly one wire diameter, so the lower wire's top face is the upper wire's
    bottom face (the reviewers' 8 mm pitch left a 3 mm gap)."""
    assert T.WIRE_PITCH == pytest.approx(T.WIRE_SIZE[2]) == pytest.approx(0.005)
    assert T.LINK_Z["left"] - T.LINK_Z["right"] == pytest.approx(T.WIRE_SIZE[2])
    s = final()
    top_right = s["right_wire"].pos[2] + T.WIRE_SIZE[2] / 2
    bottom_left = s["left_wire"].pos[2] - T.WIRE_SIZE[2] / 2
    assert bottom_left == pytest.approx(top_right)
    assert T.wire_gap(s) == pytest.approx(0.0, abs=1e-9)
    # the base tops stay below the first links on both stands
    for side in SIDES:
        assert T.LINK_Z[side] - T.LINK_H / 2 - (T.TOP + T.PLINTH_H[side]) == pytest.approx(0.003)


def test_the_clips_stop_the_overlap():
    # clips meet end to end at an overlap of 2 * wire length - 2 * clip face distance: the overlap can be at most
    assert 2 * T.WIRE_SIZE[0] > 2 * T.HALF_OVERLAP


def test_full_score_needs_the_hold():
    assert T.score(final()) == 0.5  # crossed, but not let go of and held: the 0.5 band
    assert not T.success(final())
    assert T.score(held(final())) == 1.0
    assert T.success(held(final()))


def test_full_score_needs_the_wires_touching():
    """The exploit the reviewers named: the wires cross in plan view with a gap between them (centres 8 mm apart in
    height) and no contact: it does not score 1.0, with or without the simulator's contact flag."""
    s = with_height(final(), "left_wire", 0.003)  # the old pitch: 8 mm between centres, a 3 mm gap
    assert T.wire_gap(s) == pytest.approx(0.003)
    assert T.crossed(s) and not T.wires_touch(held(s))
    assert T.score(held(s)) == 0.5 and not T.success(held(s))
    # a simulator that reports no contact point blocks full score even with the poses touching
    assert T.score(held(final(), touch=False)) == 0.5
    assert not T.success(held(final(), touch=False))
    # and a contact report cannot make up for a gap in height
    assert T.score(held(s, touch=True)) == 0.5


def test_touch_gap_half_a_millimetre():
    assert T.wires_touch(held(with_height(final(), "left_wire", 0.0004)))
    assert not T.wires_touch(held(with_height(final(), "left_wire", 0.0006)))
    assert T.wires_touch(held(with_height(final(), "left_wire", -0.0004)))  # pressed into each other: touching
    assert T.wires_touch(held(with_height(final(), "right_wire", -0.0004)))
    assert not T.wires_touch(held(with_height(final(), "right_wire", -0.003)))
    assert not T.wires_touch(held(with_height(final(), "right_wire", 0.003)))


def test_full_score_needs_stillness():
    assert T.score(held(final(vel=0.05))) == 0.5
    assert not T.success(held(final(vel=0.05)))


def test_full_score_needs_hands_off():
    assert T.score(held(final(), clear=False)) == 0.5


def test_full_score_needs_three_seconds():
    assert T.score(held(final(), held_s=2.9)) == 0.5
    assert T.score(held(final(), held_s=3.0)) == 1.0


def test_sideways_gap_two_mm():
    assert T.score(held(final(shift_right=(0.0015, 0.0)))) == 1.0
    assert T.score(held(final(shift_right=(0.0025, 0.0)))) == 0.5
    assert T.score(held(final(shift_left=(-0.0025, 0.0)))) == 0.5


def test_overlap_five_mm():
    # the right wire pulled back 8 mm: overlap 12 -> 4 mm
    assert T.score(held(final(shift_right=(0.0, -0.008)))) == 0.5
    assert T.score(held(final(shift_right=(0.0, -0.006)))) == 1.0


def test_midpoint_over_the_mark():
    both = dict(shift_right=(0.0, 0.0), shift_left=(0.0, 0.0))
    assert T.score(held(final(shift_right=(0.0, 0.0)))) == 1.0
    # both wires slid 6 mm along y: overlap and gap unchanged, the midpoint 6 mm off the mark
    assert T.score(held(final(shift_right=(0.0, 0.006), shift_left=(0.0, 0.006)))) == 0.5
    assert T.score(held(final(shift_right=(0.0, 0.004), shift_left=(0.0, 0.004)))) == 1.0
    assert both


def test_near_band_ten_mm():
    # ends 6 mm short of meeting (overlap -6 mm), midpoint 9 mm off the mark: 0.5
    assert T.score(final(shift_right=(0.0, -0.018))) == 0.5
    # 18 mm short: out of the band, but both ends within 3 cm of the mark: 0.25
    assert T.score(final(shift_right=(0.0, -0.030))) == 0.25
    # 12 mm sideways
    assert T.score(final(shift_right=(0.012, 0.0))) == 0.25


def test_loose_band_three_cm():
    assert T.score(final(shift_right=(0.0, -0.035))) == 0.25  # the right end is 29 mm from the mark
    assert T.score(final(shift_right=(0.0, -0.037))) == 0.0  # 31 mm


def test_start_far_from_mark_is_zero():
    assert T.score(scene()) == 0.0


def test_pulled_wire_scores_zero_for_the_rest_of_the_trial():
    """A wire taken out of its clip (5 mm or more from where the clip holds it) is remembered."""
    bad = final()
    w = bad["right_wire"]
    bad["right_wire"] = _state(w.pos + np.array([0.0, 0.0, 0.0]) + np.array([0.01, 0.0, 0.0]), T.FINAL_DIR["right"])
    assert T.wire_out(bad, "right")
    assert T.score(held(bad)) == 0.0 and T.STATE["pulled"]
    back = held(final())
    assert T.score(back) == 0.0 and not T.success(back)
    T.reset_memory()
    assert T.score(held(final())) == 1.0


def test_a_wire_lifted_out_is_pulled():
    s = final()
    w = s["left_wire"]
    s["left_wire"] = _state(w.pos + np.array([0.0, 0.0, 0.006]), T.FINAL_DIR["left"] + math.pi)
    assert T.wire_out(s, "left")


def test_wire_in_place_is_not_pulled():
    assert not any(T.wire_out(final(), side) for side in SIDES)
    s = final()
    w = s["right_wire"]
    s["right_wire"] = _state(w.pos + np.array([0.0, 0.003, 0.0]), T.FINAL_DIR["right"])
    assert not T.wire_out(s, "right")


def test_fallen_or_nonfinite_scores_zero():
    s = final()
    s["right_link2"] = _state((0.3, 0.0, T.TOP - 0.05), 0.0)
    assert T.score(held(s)) == 0.0
    s = final()
    s["left_wire"] = _state((float("nan"), 0.0, 0.8), 0.0)
    assert T.score(held(s)) == 0.0 and not T.success(held(s))


def test_wire_tip_follows_the_wire_whichever_way_its_axis_points():
    s = final()
    w = s["right_wire"]
    flipped = dict(s)
    flipped["right_wire"] = _state(w.pos, T.FINAL_DIR["right"] + math.pi)
    assert np.allclose(T.wire_tip(s, "right")[0], T.wire_tip(flipped, "right")[0])


def test_hands_clear_measures_the_fingers():
    s = final()
    far = np.array([[0.0, 0.0, T.TOP + 0.30]])
    assert T.hands_clear(s, far)
    link = s["right_link3"]
    near = np.array([link.pos + np.array([0.0, 0.0, 0.012 + 0.01])])  # 1 cm over the link's top face
    assert not T.hands_clear(s, near)
    edge = np.array([link.pos + np.array([T.LINK_W / 2 + 0.05, 0.0, 0.0])])
    assert T.hands_clear(s, edge)


def test_box_gap_is_euclidean_to_the_box():
    pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0]])
    gap = T.box_gap(pts, (0, 0, 0), (0, 0, 0, 1), (0.02, 0.02, 0.02))
    assert gap == pytest.approx([0.0, 0.09, math.hypot(0.09, 0.09)])


def test_drag_path_stays_in_reach_and_clear():
    """Dragging the third link along the demo's straight path (grip point and angle both linear) is a smooth motion
    of the other joints: the third hinge keeps 4 cm or more clear of the first link's line and is never fully out."""
    for side in SIDES:
        a3_0 = T.hinges(side, T.START[side])[3][2]
        a3_1 = T.FINAL_DIR[side]
        g0 = np.array(T.grip_point(side, T.START[side])[:2])
        g1 = np.array(T.grip_point(side, T.FINAL[side])[:2])
        base = np.asarray(T.BASE_XY[side])
        prev = None
        for s in np.linspace(0, 1, 61):
            a3 = a3_0 + s * T._wrap(a3_1 - a3_0)
            g = g0 + s * (g1 - g0)
            tip = g - T.GRIP_U * T._rot(a3) + T.TIP * T._rot(a3)
            q = T.solve_arm(side, tip, a3)
            _, _, p2, _ = T.hinges(side, q)
            assert np.linalg.norm(p2 - base) < 0.9 * (T.REACH[0] + T.REACH[1])
            assert all(abs(a) < T.JOINT_RANGE * 0.85 for a in q)
            if prev is not None:
                assert max(abs(T._wrap(a - b)) for a, b in zip(q, prev)) < math.radians(8)  # no branch flip
            prev = q


def test_drag_keeps_the_links_apart():
    """The first and third links never come within 3 cm of each other along the drag (they are not jointed together,
    so they would collide)."""

    def seg_gap(a, b, c, d):
        def pt(p, a, b):
            t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
            return np.linalg.norm(p - (a + t * (b - a)))

        return min(pt(a, c, d), pt(b, c, d), pt(c, a, b), pt(d, a, b))

    for side in SIDES:
        a3_0 = T.hinges(side, T.START[side])[3][2]
        g0 = np.array(T.grip_point(side, T.START[side])[:2])
        g1 = np.array(T.grip_point(side, T.FINAL[side])[:2])
        for s in np.linspace(0, 1, 61):
            a3 = a3_0 + s * T._wrap(T.FINAL_DIR[side] - a3_0)
            tip = g0 + s * (g1 - g0) - T.GRIP_U * T._rot(a3) + T.TIP * T._rot(a3)
            q = T.solve_arm(side, tip, a3)
            p0, p1, p2, (_, _, b3) = T.hinges(side, q)
            end3 = p2 + T.LINK[2] * T._rot(b3)
            assert seg_gap(p0, p1, p2, end3) > 0.03


def test_demo_reach_for_both_arms():
    demo = T.demo()
    assert set(demo) == {"left", "right"}
    for arm in ("left", "right"):
        plan = ArmPlan(list(demo[arm]), BASES[arm])
        for t in np.arange(0.05, plan.duration, 0.05):
            plan.action7(t)
        assert plan.worst_ik_mm < 2.0 and plan.worst_ik_deg < 2.0, (arm, plan.worst_ik_mm, plan.worst_ik_deg)


def test_demo_grips_across_the_link_and_lets_go():
    demo = T.demo()
    for arm, side in (("right", "right"), ("left", "left")):
        labels = [w.label for w in demo[arm]]
        assert labels.index("close") < labels.index("drag the wire end to the mark") < labels.index("let go")
        gz = T.grip_point(side, T.START[side])[2]
        for w in demo[arm][1:6]:
            assert w.pos[2] == pytest.approx(gz)  # the grip stays at one height while it holds the link
        assert demo[arm][3].yaw == pytest.approx(T.FINAL_DIR[side])


def test_the_arms_stay_apart():
    """The two grips are 15 cm or more apart all along the drag, and in their own halves."""
    demo = T.demo()
    for w_r, w_l in zip(demo["right"], demo["left"]):
        assert w_r.pos[1] < 0 < w_l.pos[1]
        assert abs(w_l.pos[1] - w_r.pos[1]) > 0.15


def test_scored_boxes_are_inside_their_meshes_bounds():
    """The wires and clip links are the boxes the scorer reads: the layout puts the wire on the link's axis, level with
    it, starting at the clip's end face."""
    for side in SIDES:
        parts = T.parts(side, T.START[side])
        link, wire = parts[f"{side}_link3"], parts[f"{side}_wire"]
        axis = np.array([math.cos(link[1]), math.sin(link[1])])
        along = np.dot(np.asarray(wire[0][:2]) - np.asarray(link[0][:2]), axis)
        assert along == pytest.approx((T.CLIP_END + T.WIRE_SIZE[0] / 2) - (T.CLIP_END - T.EXT) / 2)
        assert wire[0][2] == link[0][2]
        assert wire[2][2] < link[2][2] and wire[2][1] < link[2][1]


def test_min_separation_reads_the_contact_views_data():
    """``contact.min_separation`` is the closest of all the reported points, ``inf`` for none."""
    import contact

    assert contact.min_separation([], [0], [0]) == float("inf")
    assert contact.min_separation([0.004, -0.0002, 0.001], [3], [0]) == pytest.approx(-0.0002)
    assert contact.min_separation([9.0, 0.002, 0.0001, 0.5], [0, 2], [0, 1]) == pytest.approx(0.0001)


def test_touching_follows_the_closest_point():
    import contact

    contact._MON["sep"] = float("inf")
    assert not contact.touching()
    contact._MON["sep"] = 0.0004
    assert contact.touching()
    contact._MON["sep"] = 0.003  # a point inside the 5 mm contact offset, still 3 mm apart
    assert not contact.touching()
    contact._MON["sep"] = float("inf")


def test_reset_forgets_the_contact():
    import contact

    contact._MON["sep"] = 0.0
    contact.reset()
    assert not contact.touching()
    T.STATE["touch"] = True
    T.reset_memory()
    assert T.STATE["touch"] is False
