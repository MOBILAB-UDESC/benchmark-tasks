"""Offline tests for desoldering_pump_fire.py: the layout, the latch and every scoring outcome, the exploits, the demo's
reach and the arms' clearance. They need yam_arena (plain Python parts only) and numpy, not Isaac Sim:

    scripts/task-sync.sh desoldering-pump-fire --test
"""

import ast
import inspect
import math
import pathlib
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import desoldering_pump_fire as T  # noqa: E402
from wrist import tilt_rotation  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402
from yam_arena.station import BASES  # noqa: E402
from yam_arena.task_file import ObjectState  # noqa: E402

REST = np.zeros(3)


def quat_y(deg: float) -> np.ndarray:
    """x y z w quaternion of a turn of ``deg`` degrees about world y."""
    h = math.radians(deg) / 2
    return np.array([0.0, math.sin(h), 0.0, math.cos(h)])


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def state(pos, quat=(0.0, 0.0, 0.0, 1.0), speed=0.0) -> ObjectState:
    return ObjectState(pos=np.asarray(pos, float), quat_xyzw=np.asarray(quat, float), lin_vel=np.array([speed, 0, 0]),
                       ang_vel=REST.copy(), low=np.asarray(pos, float) - 0.01, high=np.asarray(pos, float) + 0.01)


def pump(barrel=None, tilt=0.0, travel=0.0, button=0.0, tip_at=None, speed=0.0):
    """Every free object of the pump, as the simulator would report them: the barrel at ``barrel`` (or, with
    ``tip_at``, placed so the nozzle's tip is there), turned ``tilt`` degrees about y, the knob ``travel`` m from its
    rest (negative: down) and the button ``button`` m pressed in. The other bodies keep their place in the barrel's
    frame."""
    R = rot_y(tilt)
    q = quat_y(tilt)
    nozzle_local = np.array(T.NOZZLE.pos) - np.array(T.BARREL.pos)
    if tip_at is not None:
        tip_local = nozzle_local + np.array([0.0, 0.0, -T.NOZZLE_SIZE[2] / 2])
        barrel = np.asarray(tip_at, float) - R @ tip_local
    elif barrel is None:
        barrel = np.array(T.BARREL.pos)
    barrel = np.asarray(barrel, float)
    plunger_local = np.array(T.PLUNGER.pos) - np.array(T.BARREL.pos) + np.array([0.0, 0.0, travel])
    button_local = np.array(T.BUTTON.pos) - np.array(T.BARREL.pos) + np.array([0.0, -button, 0.0])
    return {
        "barrel": state(barrel, q, speed),
        "nozzle": state(barrel + R @ nozzle_local, q, speed),
        "plunger": state(barrel + R @ plunger_local, q, speed),
        "button": state(barrel + R @ button_local, q, speed),
    }


def at_joint(index=1, off=(0.0, 0.0, 0.0), **kw):
    top = np.array([T.JOINTS[index].pos[0], T.JOINTS[index].pos[1], T.JOINT_TOP])
    return pump(tip_at=top + np.asarray(off), **kw)


@pytest.fixture(autouse=True)
def fresh():
    T.reset_memory()
    yield
    T.reset_memory()


def cock(objects_fn, **kw):
    """Push the knob to the bottom of its stroke in the pose ``objects_fn`` builds."""
    return T.observe(objects_fn(travel=-T.STROKE, **kw))


# -- the layout ------------------------------------------------------------------------------------------------------


def test_the_pump_stands_in_its_holster_with_play_round_its_nozzle():
    assert T.in_holster(pump())
    hole_half = T.HOLE / 2
    assert hole_half - T.NOZZLE_SIZE[0] / 2 >= 0.002  # 2 mm of play each side
    assert T.NOZZLE.pos[2] - T.NOZZLE_SIZE[2] / 2 - T.TOP == pytest.approx(T.NOZZLE_GAP)
    assert T.BARREL.pos[2] - T.BARREL_SIZE[2] / 2 == pytest.approx(T.TOP + T.RING_H)  # on the blocks
    assert T.NOZZLE.pos[2] + T.NOZZLE_SIZE[2] / 2 == pytest.approx(T.BARREL_BOTTOM)  # the nozzle under the barrel


def test_the_parts_are_where_the_text_says():
    assert T.BARREL_SIZE[2] == 0.12 and T.STROKE == 0.025 and T.KNOB_H == 0.012
    assert T.SPRING * T.STROKE == pytest.approx(6.0)
    assert T.BUTTON_RATE * T.BUTTON_TRAVEL == pytest.approx(6.0)
    assert T.BARREL_TOP - T.BUTTON.pos[2] == pytest.approx(0.015)  # 1.5 cm below the top
    assert T.PRESSED < T.BUTTON_TRAVEL  # the latch lets go before the button bottoms out
    pitch = {round(abs(a.pos[1] - b.pos[1]), 4) for a, b in zip(T.JOINTS, T.JOINTS[1:])}
    assert pitch == {0.02}
    assert (T.JOINT_R * 2, T.JOINT_H) == (0.007, 0.0025)
    assert T.JOINTS[1] is T.TARGET


def test_the_cocked_plunger_never_touches_the_barrel_and_the_button_has_its_gap():
    # the cocked rod's end stops KNOB_PLAY above the barrel's top, and the button stands BUTTON_GAP off the barrel
    cocked_knob_bottom = T.PLUNGER.pos[2] + T.KNOB_SIZE[2] / 2 - T.KNOB_H - T.STROKE
    assert cocked_knob_bottom - T.BARREL_TOP == pytest.approx(T.KNOB_PLAY)
    assert T.PLUNGER.collide is False  # the rod goes into the barrel's box without colliding with it
    face = T.BARREL.pos[1] + T.BARREL_SIZE[1] / 2
    assert T.BUTTON.pos[1] - T.BUTTON_SIZE[1] / 2 - face == pytest.approx(T.BUTTON_GAP)
    assert T.BUTTON_GAP == T.BUTTON_TRAVEL


def test_every_pump_part_sits_above_the_table_and_the_board_is_clear_of_the_holster():
    for o in T.OBJECTS:
        z = o.pos[2] - (o.size[2] / 2 if hasattr(o, "size") else o.height / 2)
        assert z >= T.TOP - 1e-9
    assert T.BOARD.pos[1] - T.BOARD_SIZE[1] / 2 > T.HOLSTER_XY[1] + T.RING_OUT / 2 + 0.05  # 5 cm between them


def test_the_looks_files_use_the_task_files_sizes():
    text = (pathlib.Path(__file__).parent / "looks" / "make_looks.py").read_text()
    for name, value in (("BARREL_SIZE", T.BARREL_SIZE), ("NOZZLE_SIZE", T.NOZZLE_SIZE), ("ROD_SIZE", T.KNOB_SIZE),
                        ("BUTTON_SIZE", T.BUTTON_SIZE), ("BOARD_SIZE", T.BOARD_SIZE)):
        found = re.search(rf"^{name} = (.*)$", text, re.M)
        assert found and ast.literal_eval(found.group(1)) == pytest.approx(value), name
    for name, value in (("HOLE", T.HOLE), ("JOINT_PITCH", T.JOINT_PITCH)):
        found = re.search(rf"^{name} = (.*)$", text, re.M)
        assert found and float(found.group(1)) == pytest.approx(value), name
    for f in set(T.LOOKS.values()):
        assert (pathlib.Path(__file__).parent / "looks" / f).is_file(), f


# -- the mechanism ---------------------------------------------------------------------------------------------------


def test_pressing_the_button_of_an_uncocked_pump_does_nothing():
    for travel in (0.0, -0.01, -0.02):
        assert T.observe(at_joint(travel=travel, button=T.BUTTON_TRAVEL)) is None
    assert not T.STATE["armed"] and T.STATE["shots"] == 0
    assert T.score(at_joint()) == 0.0


def test_the_latch_catches_the_knob_at_the_bottom_of_its_stroke_and_not_before():
    assert T.observe(at_joint(travel=-(T.STROKE - 0.003))) is None
    assert not T.STATE["armed"]
    assert T.observe(at_joint(travel=-(T.STROKE - 0.0005))) == "lock"
    assert T.STATE["armed"]
    assert T.observe(at_joint(travel=-T.STROKE)) is None  # already armed: no second lock


def test_the_button_pressed_short_of_its_threshold_keeps_the_latch():
    cock(at_joint)
    assert T.observe(at_joint(travel=-T.STROKE, button=T.PRESSED - 0.0005)) is None
    assert T.STATE["armed"]
    assert T.observe(at_joint(travel=-T.STROKE, button=T.PRESSED)) == "unlock"
    assert not T.STATE["armed"] and T.STATE["shots"] == 1


def test_the_plunger_has_to_come_back_up_before_the_latch_can_catch_again():
    cock(at_joint)
    T.observe(at_joint(travel=-T.STROKE, button=T.PRESSED))
    assert T.observe(at_joint(travel=-T.STROKE + 0.002)) is None  # still low just after the shot
    assert not T.STATE["armed"]
    assert T.observe(at_joint(travel=-0.003)) is None  # up
    assert T.observe(at_joint(travel=-T.STROKE)) == "lock"  # pushed down again: armed again
    assert T.STATE["armed"]


def test_the_state_is_measured_in_the_barrels_frame_when_the_pump_is_tilted():
    for tilt in (0.0, 10.0, -12.0):
        o = pump(tilt=tilt, travel=-0.02, button=0.003)
        assert T.plunger_travel(o) == pytest.approx(-0.02, abs=1e-9)
        assert T.button_travel(o) == pytest.approx(0.003, abs=1e-9)
        assert T.tilt_deg(o) == pytest.approx(abs(tilt), abs=1e-6)


# -- the scoring ladder ----------------------------------------------------------------------------------------------


def shoot(off=(0.0, 0.0, 0.0), tilt=0.0, index=1):
    """Cock the pump with its tip at ``off`` from the joint's top and fire it there."""
    T.observe(at_joint(index, off, tilt=tilt, travel=-T.STROKE))
    T.observe(at_joint(index, off, tilt=tilt, travel=-T.STROKE, button=T.PRESSED + 0.001))


def test_full_score_needs_a_good_shot_then_the_pump_standing_still_in_its_holster():
    shoot()
    assert T.STATE["good_shot"] and T.STATE["last_error_mm"] < 0.01
    assert T.score(pump()) == 1.0 and T.success(pump())


def test_a_good_shot_with_the_pump_left_anywhere_else_scores_three_quarters():
    shoot()
    assert T.score(at_joint()) == 0.75  # still held on the joint
    assert T.score(pump(barrel=(0.2, 0.1, T.BARREL.pos[2]))) == 0.75  # set down somewhere else
    assert T.score(pump(tilt=30.0)) == 0.75  # toppled in the holster's place
    assert not T.success(at_joint())


def test_the_holster_has_to_hold_the_pump_still_and_upright():
    shoot()
    assert T.score(pump(speed=0.05)) == 0.75
    assert T.score(pump(tilt=T.SEAT_TILT + 1)) == 0.75
    assert T.score(pump(barrel=np.array(T.BARREL.pos) + [T.SEAT_XY + 0.002, 0, 0])) == 0.75
    assert T.score(pump(barrel=np.array(T.BARREL.pos) + [0, 0, T.SEAT_Z + 0.002])) == 0.75
    assert T.score(pump(barrel=np.array(T.BARREL.pos) + [T.SEAT_XY - 0.002, 0, 0])) == 1.0


def test_a_shot_within_two_millimetres_is_good_and_a_little_more_is_only_near():
    shoot(off=(0.0015, 0.0, 0.0))
    assert T.STATE["good_shot"] and T.score(pump()) == 1.0
    T.reset_memory()
    shoot(off=(0.0025, 0.0, 0.0))
    assert T.STATE["near_shot"] and not T.STATE["good_shot"]
    assert T.score(pump()) == 0.5


def test_a_shot_within_six_millimetres_scores_half_and_beyond_it_nothing():
    shoot(off=(0.0, 0.005, 0.0))
    assert T.score(pump()) == 0.5
    T.reset_memory()
    shoot(off=(0.0, 0.007, 0.0))
    assert T.STATE["shots"] == 1 and T.score(pump()) == 0.0


def test_a_shot_with_the_pump_tilted_fifteen_degrees_or_more_is_not_good():
    shoot(tilt=14.0)
    assert T.STATE["good_shot"]
    T.reset_memory()
    shoot(tilt=16.0)
    assert not T.STATE["good_shot"] and T.STATE["near_shot"]
    assert T.score(pump()) == 0.5


def test_a_shot_a_few_millimetres_above_the_joint_is_not_on_it():
    shoot(off=(0.0, 0.0, 0.003))
    assert not T.STATE["good_shot"] and T.STATE["near_shot"]


def test_armed_with_the_tip_on_the_joint_but_never_fired_scores_a_quarter():
    cock(at_joint)
    assert T.STATE["armed_on_joint"] and T.STATE["shots"] == 0
    assert T.score(at_joint(travel=-T.STROKE)) == 0.25
    assert T.score(pump()) == 0.25  # the event stands once the pump is put away


def test_armed_on_the_joint_but_fired_in_the_air_still_scores_a_quarter():
    cock(at_joint)
    T.observe(at_joint(off=(0.0, 0.0, 0.05), travel=-T.STROKE, button=T.PRESSED + 0.001))
    assert T.STATE["shots"] == 1 and not T.STATE["near_shot"]
    assert T.score(pump()) == 0.25


def test_cocking_alone_or_being_on_the_joint_uncocked_scores_nothing():
    cock(pump)
    assert T.score(pump()) == 0.0
    T.reset_memory()
    T.observe(at_joint())
    assert T.score(at_joint()) == 0.0


def test_a_shot_on_a_neighbouring_joint_is_a_wasted_shot():
    shoot(index=0)
    assert T.STATE["shots"] == 1 and not T.STATE["near_shot"] and not T.STATE["armed_on_joint"]
    assert T.score(pump()) == 0.0
    T.reset_memory()
    shoot(index=2)
    assert T.score(pump()) == 0.0


def test_a_wasted_shot_can_be_followed_by_a_good_one_after_cocking_again():
    shoot(index=0)
    T.observe(pump(travel=-0.003))  # the knob has come back up
    shoot(index=1)
    assert T.STATE["shots"] == 2 and T.STATE["good_shot"]
    assert T.score(pump()) == 1.0


def test_a_second_shot_cannot_be_fired_without_cocking_again():
    shoot()
    T.observe(pump(travel=-0.003))
    assert T.observe(at_joint(travel=0.0, button=T.BUTTON_TRAVEL)) is None
    assert T.STATE["shots"] == 1


def test_a_pump_that_fell_off_the_table_scores_nothing_even_after_a_good_shot():
    shoot()
    gone = pump(barrel=(0.3, 0.2, T.TOP - 0.2))
    assert T.score(gone) == 0.0 and not T.success(gone)


def test_a_pump_in_the_holster_cocked_on_the_table_is_not_on_the_joint():
    cock(pump)
    assert not T.STATE["armed_on_joint"] and T.STATE["armed"]


def test_the_memory_is_cleared_by_a_reset():
    shoot()
    assert T.STATE["good_shot"]
    T.reset_memory()
    assert T.STATE == {"armed": False, "rearm_ok": True, "shots": 0, "armed_on_joint": False, "near_shot": False,
                       "good_shot": False, "last_error_mm": None, "last_tilt": None}
    assert T.score(pump()) == 0.0


def test_a_non_finite_state_scores_nothing_and_changes_nothing():
    o = at_joint(travel=-T.STROKE)
    o["barrel"] = state([np.nan, 0, 0])
    assert T.observe(o) is None and T.score(o) == 0.0 and not T.STATE["armed"]


def test_the_pump_at_rest_in_its_holster_at_the_start_scores_nothing():
    assert T.score(pump()) == 0.0 and not T.success(pump())


# -- the demo ----------------------------------------------------------------------------------------------------------


def run(plan: ArmPlan):
    for k in range(int(plan.duration * 20)):
        plan.action7((k + 1) / 20)
    return plan


@pytest.mark.parametrize("arm", ["right", "left"])
def test_the_demo_is_reachable_for_each_arm(arm):
    plan = run(ArmPlan(list(T.demo()[arm]), BASES[arm]))
    assert plan.worst_ik_mm < 2.0 and plan.worst_ik_deg < 2.0


def test_the_demo_stays_within_the_run_limit():
    d = T.demo()
    total = max(sum(w.duration for w in d[arm]) for arm in d)
    assert total <= 45.0  # 900 steps, well inside the 1,500 the task allows


def test_the_grips_point_where_the_demo_assumes():
    a_right = tilt_rotation(T.GRIP_YAW, T.PITCH)[:, 2]
    a_left = tilt_rotation(T.BTN_YAW, T.PITCH)[:, 2]
    assert a_right == pytest.approx([math.sin(T.PITCH), 0.0, -math.cos(T.PITCH)], abs=1e-9)
    assert a_left == pytest.approx([0.0, -math.sin(T.PITCH), -math.cos(T.PITCH)], abs=1e-9)
    a_press = tilt_rotation(T.PRESS_YAW, T.PRESS_PITCH)[:, 2]
    assert a_press == pytest.approx([math.sin(T.PRESS_PITCH), 0.0, -math.cos(T.PRESS_PITCH)], abs=1e-9)


def test_the_demo_waypoints_hold_the_pump_where_the_layout_puts_the_tip_and_the_button():
    d = T.demo()
    by = {w.label: w for w in d["right"]}
    x, y, z = by["tip on the joint"].pos
    assert (x, y) == pytest.approx((T.TARGET.pos[0] + T.AIM[0], T.TARGET.pos[1] + T.AIM[1]))
    assert z == pytest.approx(T.JOINT_TOP - T.SET_PRESS + T.TIP_BELOW_TCP)
    # the right grip is one hand's width under the button: the arms' tool points are about 9 cm apart
    press = {w.label: w for w in d["left"]}["press the button"]
    held_button_z = T.JOINT_TOP + T.NOZZLE_SIZE[2] + T.BUTTON_ABOVE_BOTTOM
    assert held_button_z - (T.JOINT_TOP + T.TIP_BELOW_TCP) >= 0.079
    assert press.pos[2] == pytest.approx(held_button_z + T.PAD_TIP_PAST_TCP * math.cos(T.PITCH))


def test_the_left_arm_presses_after_the_pump_has_settled_and_leaves_before_the_right_lifts():
    d = T.demo()
    t = 0.0
    right_times = {}
    for w in d["right"]:
        right_times[w.label] = (t, t + w.duration)
        t += w.duration
    t = 0.0
    left_times = {}
    for w in d["left"]:
        left_times[w.label] = (t, t + w.duration)
        t += w.duration
    settled = right_times["settle"][1]
    assert left_times["press the button"][0] >= settled
    assert left_times["back off"][1] <= right_times["lift off"][0]
    # the press lasts long enough to cross the latch, and the hold afterwards lets the plunger finish its stroke
    assert left_times["hold it"][1] - left_times["press the button"][0] >= 2.0


def test_the_demo_reads_only_the_layout():
    src = inspect.getsource(T.demo)
    for banned in ("STATE", "object_states", "write_pose", "plunger_travel", "button_travel", "observe"):
        assert banned not in src


def test_the_ablation_switches_change_the_demo():
    base = [(w.label, tuple(w.pos)) for w in T.demo()["right"]]
    left = [(w.label, tuple(w.pos)) for w in T.demo()["left"]]
    T.COCK = False
    try:
        assert [(w.label, tuple(w.pos)) for w in T.demo()["right"]] != base
    finally:
        T.COCK = True
    T.FIRE = False
    try:
        assert [(w.label, tuple(w.pos)) for w in T.demo()["left"]] != left
    finally:
        T.FIRE = True
    T.JOINT_USED = 0
    try:
        assert [(w.label, tuple(w.pos)) for w in T.demo()["right"]] != base
    finally:
        T.JOINT_USED = 1


def test_the_arms_keep_their_distance_during_the_demo():
    """Link origins of both arms, centre to centre, stay 9 cm apart or more (fingers closed, which is closest)."""
    from yam_arena import YAM_URDF
    from yam_arena.constants import ARM_JOINTS
    from yam_arena.urdf import forward_kinematics

    d = T.demo()
    plans = {a: ArmPlan(list(d[a]), BASES[a]) for a in ("left", "right")}
    steps = int(max(p.duration for p in plans.values()) * 20)
    closest = 9.0
    for k in range(steps):
        t = (k + 1) / 20
        origins = {}
        for arm, p in plans.items():
            action, _ = p.action7(min(t, p.duration))
            q = {n: float(v) for n, v in zip(ARM_JOINTS, action[:6])}
            q["joint7"] = q["joint8"] = 0.0
            origins[arm] = np.array([m[:3, 3] + np.asarray(BASES[arm]) for m in forward_kinematics(YAM_URDF, q).values()])
        dist = np.linalg.norm(origins["left"][:, None, :] - origins["right"][None, :, :], axis=2).min()
        closest = min(closest, float(dist))
    assert closest >= 0.09, f"the arms come within {closest * 100:.1f} cm"
