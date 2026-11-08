"""Offline checks (no GPU): the task file loads, the punch's parts fit (the tip in the holder's hole, the nose in its
pocket, the stem in the barrel's sleeve at every travel), the looks exist and fit their boxes, the shot logic (a shot
when the barrel passes the trip, none before, a second only after the punch is cocked again, a shot off the plate), the
scorer covers each outcome (every band, two shots, none, off the plate, not clear, moving, fallen, not finite), the
demo's reach and its clearances. Run in the yam_arena environment: python -m pytest -q test_auto_center_punch_mark.py"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
from yam_arena.demo import ArmPlan
from yam_arena.station import BASES
from yam_arena.task_file import ObjectState, load

HERE = pathlib.Path(__file__).resolve().parent
TASK = load(HERE / "auto_center_punch_mark.py")
T = TASK.module
READY = {"left": np.array([0.323, 0.305, 1.008]), "right": np.array([0.323, -0.305, 1.008])}
UP = np.array([0.0, 0.0, 1.0])


def _state(pos, quat=(0.0, 0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0)):
    pos = np.asarray(pos, float)
    return ObjectState(pos=pos, quat_xyzw=np.asarray(quat, float), lin_vel=np.asarray(vel, float),
                       ang_vel=np.asarray(spin, float), low=pos - 0.01, high=pos + 0.01)


def _quat_tilt(deg, axis=(1.0, 0.0, 0.0)):
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    h = math.radians(deg) / 2
    return (*(a * math.sin(h)), math.cos(h))


def punch(at=T.CROSS, height=0.0, down=0.0, tilt=0.0, vel=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0)) -> dict:
    """The punch with its tip's underside over ``at`` (x, y), ``height`` above the plate's top, the barrel ``down`` m
    down over the nose, the whole turned ``tilt`` degrees about x about the tip's underside."""
    q = _quat_tilt(tilt)
    R = T.matrix(q)
    base = np.array([at[0], at[1], T.PLATE_TOP + height])
    tip = base + R @ np.array([0.0, 0.0, T.TIP_BOX[2] / 2])
    barrel = tip + R @ np.array([0.0, 0.0, T.REST_GAP - down])
    return {"tip": _state(tip, q, vel, spin), "barrel": _state(barrel, q, vel, spin),
            "plate": _state(T.PLATE.pos), "holder": _state(T.HOLDER.pos)}


@pytest.fixture(autouse=True)
def fresh():
    T.reset_memory()
    yield
    T.reset_memory()


def fire(at=T.CROSS, height=0.0, tilt=0.0):
    """Press the barrel down through the trip at ``at`` and let it come back up."""
    for d in (0.0, 0.003, 0.0069, T.TRIP, 0.009):
        T.watch_shots(punch(at, height, d, tilt))
    for d in (0.005, 0.0):
        T.watch_shots(punch(at, height, d, tilt))


# -- the file --------------------------------------------------------------------------------------------------------


def test_the_file_loads_with_four_objects():
    assert TASK.names == ("plate", "holder", "tip", "barrel")
    assert TASK.free_objects == ("tip", "barrel")
    assert callable(TASK.score) and callable(TASK.success) and callable(TASK.demo)
    assert T.SCENE == "station"


def test_the_barrel_is_a_sprung_slider_on_the_nose():
    b = T.BARREL
    assert b.to == "tip" and b.axis == "Z" and b.upper == 0.0 and b.lower == -T.TRAVEL
    assert b.rate * T.TRIP == pytest.approx(10.5)
    assert T.REARM < T.TRIP < T.TRAVEL
    assert T.BARREL.mass * 9.81 / T.RATE < T.REARM  # the barrel's own weight doesn't cock or fire it


# -- the parts fit ---------------------------------------------------------------------------------------------------


def test_the_nose_stands_on_the_pockets_floor_with_its_tip_in_the_hole():
    z_floor = T.TOP + T.FLOOR_Z
    tip_bottom = T.TIP.pos[2] - T.TIP_BOX[2] / 2
    assert tip_bottom + T.TIP_SIZE[2] == pytest.approx(z_floor)  # the shoulder's underside rests on the ring
    assert tip_bottom > T.TOP + T.BASE_Z  # the tip clears the hole's floor
    assert T.HOLE - T.TIP_SIZE[0] >= 0.004  # 2 mm each side
    assert (T.POCKET - T.TIP_BOX[0]) / 2 >= 0.002
    assert T.HOLDER.pos[:2] == T.HOLDER_C and T.TIP.pos[:2] == T.HOLDER_C and T.BARREL.pos[:2] == T.HOLDER_C


def test_the_holders_parts_stay_inside_its_block_and_leave_the_pocket_and_hole_open():
    parts = __import__("colliders").shapes()["holder"]
    for p in parts:
        for i in range(3):
            assert abs(p.centre[i]) + p.size[i] / 2 <= T.HOLDER_SIZE[i] / 2 + 1e-9
    # the points of the tip's underside and of the nose's corners are in no part
    def inside(pt):
        return any(all(abs(pt[i] - p.centre[i]) < p.size[i] / 2 - 1e-9 for i in range(3)) for p in parts)

    hh = T.HOLDER_SIZE[2] / 2
    z_tip = T.FLOOR_Z - hh + 0.001  # just above the tip's underside: inside the hole
    assert not inside((0.0, 0.0, z_tip))
    for sx in (-1, 1):
        for sy in (-1, 1):
            assert not inside((sx * T.TIP_BOX[0] / 2 * 0.999, sy * T.TIP_BOX[1] / 2 * 0.999, T.FLOOR_Z - hh + 0.001
                               + T.TIP_SIZE[2]))
    assert inside((0.0, 0.0, T.BASE_Z / 2 - hh))  # the hole has a floor


def test_the_stem_stays_in_the_sleeve_through_the_whole_travel_and_clears_the_cap():
    s, d = T.SLEEVE
    assert s - T.STEM_SIZE[0] >= 0.001  # 1 mm clear each side... half of it
    stem_top = T.TIP.pos[2] + T.TIP_BOX[2] / 2
    cap_under = T.BARREL.pos[2] - T.BARREL_SIZE[2] / 2 + d
    assert cap_under - T.TRAVEL - stem_top >= 0.002  # at full travel the stem is 2 mm short of the cap
    sleeve_bottom = T.BARREL.pos[2] - T.BARREL_SIZE[2] / 2
    shoulder_top = T.TIP.pos[2] - T.TIP_BOX[2] / 2 + T.TIP_SIZE[2] + T.SHOULDER_SIZE[2]
    assert sleeve_bottom - shoulder_top == pytest.approx(T.GAP)
    assert T.GAP - T.TRAVEL >= 0.0015  # the sleeve never reaches the shoulder
    assert T.BARREL_SIZE[0] <= 0.094 / 2  # the open gripper takes it


def test_the_barrel_hangs_clear_of_the_holders_walls():
    sleeve_bottom = T.BARREL.pos[2] - T.BARREL_SIZE[2] / 2
    assert sleeve_bottom > T.TOP + T.HOLDER_SIZE[2] + 0.02


def test_the_layout_keeps_to_the_right_arms_half_and_clear_of_the_ready_hand():
    for obj in (T.PLATE, T.HOLDER):
        lo, hi = np.asarray(obj.pos) - np.asarray(obj.size) / 2, np.asarray(obj.pos) + np.asarray(obj.size) / 2
        assert hi[1] < 0.0 and lo[0] > 0.1 and hi[0] < 0.5
        assert not (lo[0] < 0.36 < hi[0] and hi[2] > T.TOP + 0.2)  # nothing tall near the READY hand
    assert abs(T.CROSS[0] - T.PLATE_C[0]) > 0.005 and abs(T.CROSS[1] - T.PLATE_C[1]) > 0.003  # not the plate's centre
    assert abs(T.CROSS[0] - T.PLATE_C[0]) + T.CROSS_ARM < T.PLATE_SIZE[0] / 2
    assert abs(T.CROSS[1] - T.PLATE_C[1]) + T.CROSS_ARM < T.PLATE_SIZE[1] / 2


def test_the_looks_exist_and_fit_their_boxes():
    lk = __import__("looks").looks()
    assert set(lk) == {"plate", "holder", "tip", "barrel"}
    assert all(path.is_file() for entries in lk.values() for path, _ in entries)
    src = (HERE / "looks" / "make_looks.py").read_text()
    for token in ("PLATE = (0.11, 0.07, 0.012)", "TIP = (0.016, 0.016, 0.068)", "BARREL = (0.022, 0.022, 0.085)"):
        assert token in src
    assert T.TIP_BOX == pytest.approx((0.016, 0.016, 0.068)) and T.BARREL_SIZE == pytest.approx((0.022, 0.022, 0.085))
    assert T.PLATE_SIZE == pytest.approx((0.11, 0.07, 0.012))


# -- the shot --------------------------------------------------------------------------------------------------------


def test_the_travel_reads_the_barrels_position_over_the_nose():
    assert T.travel(punch(down=0.0)) == pytest.approx(0.0, abs=1e-9)
    assert T.travel(punch(down=0.007)) == pytest.approx(0.007)
    assert T.travel(punch(down=0.007, tilt=20.0)) == pytest.approx(0.007)  # along the nose's own axis


def test_no_shot_before_the_trip():
    for d in (0.0, 0.002, 0.004, 0.0069):
        T.watch_shots(punch(down=d))
    assert T.STATE["shots"] == []


def test_a_shot_when_the_barrel_passes_the_trip_where_the_tip_is():
    T.watch_shots(punch(at=(T.CROSS[0] + 0.0003, T.CROSS[1] - 0.0004), down=0.0071))
    (x, y, on), = T.STATE["shots"]
    assert (x, y) == pytest.approx((T.CROSS[0] + 0.0003, T.CROSS[1] - 0.0004)) and on
    assert T.mark_offset() == pytest.approx(0.0005)


def test_pressing_on_does_not_fire_again_until_the_barrel_comes_back_up():
    for d in (0.007, 0.009, 0.011, 0.009, 0.005, 0.0031, 0.008, 0.011):
        T.watch_shots(punch(down=d))
    assert len(T.STATE["shots"]) == 1  # it never came back past the re-arm point
    T.watch_shots(punch(down=0.002))
    T.watch_shots(punch(down=0.008))
    assert len(T.STATE["shots"]) == 2


def test_a_shot_in_the_air_or_in_the_holder_is_a_shot_off_the_plate():
    T.watch_shots(punch(height=0.015, down=0.008))
    assert T.STATE["shots"][0][2] is False
    assert T.mark_offset() is None


def test_a_shot_on_a_tilted_punch_is_taken_at_the_tips_underside():
    T.watch_shots(punch(down=0.008, tilt=15.0))
    (x, y, on), = T.STATE["shots"]
    assert (x, y) == pytest.approx(T.CROSS) and on


def test_a_reset_forgets_the_shots_and_cocks_the_punch():
    fire()
    T.reset_memory()
    assert T.STATE == {"shots": [], "armed": True, "track": []}


def test_a_state_that_is_not_finite_records_nothing():
    s = punch(down=0.01)
    s["tip"] = _state([np.nan, 0, 0])
    T.watch_shots(s)
    assert T.STATE["shots"] == []


# -- the scorer ------------------------------------------------------------------------------------------------------


def lifted(at=T.CROSS):
    return punch(at, height=0.03)


def settle(state, steps=T.STILL_WINDOW + 2):
    """Hold the punch in ``state`` for a little over a second, as the scorer's hook reads it every step."""
    for _ in range(steps):
        T.watch_shots(state)


def test_no_shot_scores_zero():
    assert T.score(punch()) == 0.0 and not T.success(punch())
    assert T.score(lifted()) == 0.0


@pytest.mark.parametrize("off,expect", [(0.0, 1.0), (0.0004, 1.0), (0.0005, 1.0), (0.0006, 0.5), (0.001, 0.5),
                                        (0.0012, 0.25), (0.002, 0.25), (0.0022, 0.0), (0.01, 0.0)])
def test_each_band_of_offset_when_the_punch_is_lifted_clear_and_still(off, expect):
    d = off - 1e-7 if off in (0.0005, 0.001, 0.002) else off  # the exact band edges, a hair inside
    at = (T.CROSS[0] + d * 0.6, T.CROSS[1] - d * 0.8)
    fire(at)
    settle(lifted(at))
    assert T.score(lifted(at)) == expect
    assert T.success(lifted(at)) == (expect == 1.0)


def test_a_mark_within_half_a_millimetre_scores_half_until_the_punch_is_clear_and_still():
    fire()
    assert T.score(punch()) == 0.5  # the tip still on the plate
    settle(punch(height=0.019))
    assert T.score(punch(height=0.019)) == 0.5  # nearly clear
    settle(punch(height=0.02))
    assert T.score(punch(height=0.02)) == 1.0


def test_the_punch_must_have_been_still_for_a_second():
    fire()
    settle(lifted(), steps=T.STILL_WINDOW)  # one step short of a second recorded
    assert T.score(lifted()) == 0.5
    T.watch_shots(lifted())
    assert T.score(lifted()) == 1.0
    moved = punch(T.CROSS, height=0.03)
    moved["tip"] = _state(moved["tip"].pos + np.array([0.0015, 0.0, 0.0]), moved["tip"].quat_xyzw)
    assert T.score(moved) == 0.5  # a part 1.5 mm from where it was a second ago
    nudged = punch(T.CROSS, height=0.03)
    nudged["barrel"] = _state(nudged["barrel"].pos + np.array([0.0, 0.0006, 0.0]), nudged["barrel"].quat_xyzw)
    assert T.score(nudged) == 1.0  # 0.6 mm: still


def test_a_buzzing_punch_is_still_when_it_has_not_moved():
    fire()
    settle(lifted())
    buzz = punch(T.CROSS, height=0.03, vel=(0.0, 0.0, 0.01), spin=(0.0, 1.6, 0.0))
    assert T.score(buzz) == 1.0  # speeds are noise in the pads: the positions are what count


def test_two_shots_score_zero_even_when_both_are_on_the_cross():
    fire()
    fire()
    assert len(T.STATE["shots"]) == 2
    assert T.score(lifted()) == 0.0


def test_a_good_shot_and_a_shot_in_the_air_score_zero():
    fire()
    fire(height=0.02)
    assert T.score(lifted()) == 0.0


def test_a_shot_off_the_plate_scores_zero():
    fire(at=(T.PLATE_C[0] + 0.07, T.PLATE_C[1]))
    assert T.score(lifted()) == 0.0


def test_the_punch_off_the_table_scores_zero():
    fire()
    gone = lifted()
    gone["tip"] = _state([0.3, -0.3, T.TOP - 0.05])
    assert T.score(gone) == 0.0
    gone = lifted()
    gone["barrel"] = _state([0.3, -0.3, T.TOP - 0.05])
    assert T.score(gone) == 0.0


def test_a_state_that_is_not_finite_scores_zero():
    fire()
    bad = lifted()
    bad["barrel"] = _state([np.nan, 0.0, 0.0])
    assert T.score(bad) == 0.0


def test_doing_nothing_scores_zero_and_the_start_is_not_a_success():
    start = {"tip": _state(T.TIP.pos), "barrel": _state(T.BARREL.pos), "plate": _state(T.PLATE.pos),
             "holder": _state(T.HOLDER.pos)}
    assert T.score(start) == 0.0 and not T.success(start)
    assert T.distance(start) > 0.1


def test_the_start_state_does_not_fire_the_punch():
    start = {"tip": _state(T.TIP.pos), "barrel": _state(T.BARREL.pos)}
    assert T.travel(start) == pytest.approx(0.0, abs=1e-9)
    T.watch_shots(start)
    assert T.STATE["shots"] == []


def test_distance_is_the_tips_underside_to_the_cross():
    assert T.distance(punch()) == pytest.approx(0.0, abs=1e-9)
    assert T.distance(punch(height=0.03)) == pytest.approx(0.03)


# -- the demo --------------------------------------------------------------------------------------------------------


def _right_plan():
    plan = ArmPlan(list(T.demo()["right"]), BASES["right"])
    t, points = 0.0, []
    while t < plan.duration:
        plan.action7(t)
        points.append(plan.target(t)[0])
        t += 0.05
    return plan, np.array(points)


def test_the_demo_uses_only_the_right_arm_and_reaches_every_waypoint():
    assert set(T.demo()) == {"right"}
    plan, _ = _right_plan()
    assert plan.worst_ik_mm < 2.0 and plan.worst_ik_deg < 2.0


def test_the_demos_hand_stays_in_the_right_half_and_above_the_table():
    _, pts = _right_plan()
    assert (pts[1:, 1] < 0.0).all()
    assert (pts[:, 2] > T.TOP + 0.05).all()


def test_the_demo_grips_the_barrel_not_the_nose():
    first = [w for w in T.demo()["right"] if w.label == "close on it"][0]
    assert abs(first.pos[2] - T.BARREL.pos[2]) < 0.02
    assert first.pos[2] - 0.013 > T.TIP.pos[2] + T.TIP_BOX[2] / 2 - 0.0  # the pad tips stay above the stem's bottom


def test_the_demo_presses_slowly_and_pushes_past_the_trip():
    push = [w for w in T.demo()["right"] if w.label.startswith("push down")][0]
    settle = [w for w in T.demo()["right"] if w.label == "settle"][0]
    assert settle.pos[2] - push.pos[2] > T.TRIP  # the hand goes further down than the barrel's trip
    assert (settle.pos[2] - push.pos[2]) / push.duration < 0.005  # 5 mm/s at most: the push builds slowly


def test_the_demo_ends_with_the_punch_lifted_clear_and_held():
    last = T.demo()["right"][-1]
    tip_height = last.pos[2] - (T.PLATE_TOP + T.REST_GAP + T.TIP_BOX[2] / 2 + T.GRIP_DZ)
    assert tip_height >= T.CLEAR + 0.02 and last.duration >= 1.0
