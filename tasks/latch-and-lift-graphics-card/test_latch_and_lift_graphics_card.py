"""Offline checks (no GPU): the task file loads, the scorer covers each outcome, reads the button at every step (a
press is not remembered) and remembers a broken latch, the card must be let go and lie on the tray's floor for 1.0,
the bond and the button are specified as the linkage needs, and the demo presses before it lifts and stays within
reach. Run in the yam_arena environment:  python -m pytest -q test_latch_and_lift_graphics_card.py"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
from yam_arena.demo import ArmPlan
from yam_arena.station import BASES
from yam_arena.task_file import ObjectState, load

HERE = pathlib.Path(__file__).resolve().parent
TASK = load(HERE / "latch_and_lift_graphics_card.py")
T = TASK.module

CARD_START = np.asarray(T.CARD.pos)
LATCH_START = np.asarray(T.LATCH.pos)
DOWN = np.array([0.0, 0.0, -T.PRESSED - 0.0005])


def _quat(axis, deg):
    """The quaternion x y z w of a turn of ``deg`` degrees about ``axis``."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    half = math.radians(deg) / 2
    return np.array([*(axis * math.sin(half)), math.cos(half)])


def _state(pos, vel=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0)):
    pos = np.asarray(pos, float)
    return ObjectState(pos=pos, quat_xyzw=np.asarray(quat, float), lin_vel=np.asarray(vel, float),
                       ang_vel=np.zeros(3), low=pos - 0.02, high=pos + 0.02)


def _scene(card=None, latch=None):
    """The start, with the card or the latch replaced by a state or a centre."""
    out = {"card": _state(CARD_START), "latch": _state(LATCH_START)}
    for name, value in (("card", card), ("latch", latch)):
        if value is not None:
            out[name] = value if isinstance(value, ObjectState) else _state(value)
    return out


def _play(*scenes):
    """The score at each of several consecutive control steps."""
    return [T.score(scene) for scene in scenes]


PRESSED = LATCH_START + DOWN
FLAT_ON_THE_FOAM = (*T.TRAY_XY, T.TRAY_FLOOR_Z + T.CARD_SIZE[1] / 2)  # lying on its face, the thickness axis up
LYING = _quat((1.0, 0.0, 0.0), 90.0)  # the card's thickness axis turned to vertical
STANDING_IN_TRAY = (*T.TRAY_XY, T.TRAY_FLOOR_Z + T.CARD_SIZE[2] / 2)  # on its 18 cm edge, upright
ON_ITS_END = (*T.TRAY_XY, T.TRAY_FLOOR_Z + T.CARD_SIZE[0] / 2)  # on its 11 cm end
ENDWISE = _quat((0.0, 1.0, 0.0), 90.0)  # the card's long axis turned to vertical
IN_TRAY = _state(FLAT_ON_THE_FOAM, quat=LYING)
RAISED = CARD_START + (0.0, 0.0, 0.006)  # up 6 mm, still between the rails
OUT = CARD_START + (0.0, 0.0, 0.05)
ABOVE_THE_TRAY = (*T.TRAY_XY, T.TRAY_FLOOR_Z + 0.06)  # upright, its bottom 0.5 cm above the foam
HAND_ON_THE_CARD = {"left_robot": (np.array([*T.TRAY_XY, T.TRAY_FLOOR_Z + 0.05]), 0.0205),
                    "right_robot": (np.array([0.3, -0.3, 0.9]), -0.0)}
HAND_AWAY = {"left_robot": (np.array([*T.TRAY_XY, T.TRAY_FLOOR_Z + 0.10]), 0.09),
             "right_robot": (np.array([0.3, -0.3, 0.9]), 0.0)}


@pytest.fixture(autouse=True)
def fresh_episode():
    T.reset_memory()


def test_loads():
    assert TASK.scene == "station"
    assert set(TASK.free_objects) == {"card", "latch"}


def test_the_trap_is_specified():
    assert T.CARD.to == "mainboard" and not T.BOARD.free  # locked to the bench, through the fixed board
    assert T.CARD.hold == 60.0 and math.isinf(T.CARD.twist)
    assert T.LATCH.axis == "Z" and T.LATCH.lower == -T.LATCH_TRAVEL and T.LATCH.upper == 0.0
    assert T.PRESSED < T.LATCH_TRAVEL
    assert T.LATCH_RATE * T.LATCH_TRAVEL == pytest.approx(2.0)


def test_the_linkage_numbers_agree():
    """The pin, the hole and the button as the README and the setup describe them."""
    assert T.PIN_ENGAGED < T.PRESSED < T.LATCH_TRAVEL  # 3 mm of button pulls a pin 2.5 mm in clear, with 4 mm of travel
    pin_bottom = T.PIN_AXIS - T.PIN_D / 2
    assert pin_bottom - T.HOLE_BOTTOM == pytest.approx(T.LIFTED)  # the card rises 5 mm before the hole's edge meets it
    assert T.HOLE_BOTTOM + T.HOLE_H > T.PIN_AXIS + T.PIN_D / 2  # and the pin sits inside the hole
    assert T.PIN_AXIS + T.PIN_D / 2 < 0.012  # the pin's guide, a little higher, stays under the board's 1.2 cm
    assert T.LIFTED < T.SLOT_DEPTH


def test_layout():
    assert T.CARD.pos[2] - T.CARD_SIZE[2] / 2 == pytest.approx(T.BOARD_TOP)  # the card's bottom edge on the board
    assert T.LATCH.pos[2] - T.LATCH_SIZE[2] / 2 - T.LATCH_TRAVEL > T.BOARD_TOP  # clear of the board when fully down
    assert T.LATCH_TOP - T.LATCH_TRAVEL > T.BOARD_TOP + T.HOUSING_H  # and still out of its housing
    for wall in T.HOUSING:
        assert not wall.free and wall.pos[2] + wall.size[2] / 2 == pytest.approx(T.BOARD_TOP + T.HOUSING_H)
    gap = T.CARD_U - T.CARD_SIZE[0] / 2 - (T.LATCH_U + T.LATCH_SIZE[0] / 2)
    assert 0.02 < gap < 0.06  # room for the pressing gripper, next to the card's near end
    assert gap == pytest.approx(0.055)  # the 5.5 cm the push rod spans
    for rail in T.RAILS:
        assert abs(rail.pos[0] - T.SLOT_LINE_X) - rail.size[1] / 2 == pytest.approx(T.CARD_SIZE[1] / 2 + T.RAIL_GAP)


# -- the button is read at every step --------------------------------------------------------------------------------


def test_start_scores_zero():
    assert T.score(_scene()) == 0.0
    assert not T.success(_scene())


def test_pressed_but_not_lifted_zero():
    assert T.score(_scene(latch=PRESSED)) == 0.0


def test_pressed_and_raised_quarter():
    assert T.score(_scene(card=RAISED, latch=PRESSED)) == 0.25


def test_lifted_while_held_down_then_let_go_is_fine():
    """The button held until the card is 6 mm up, then released: the pin meets the plate's face, nothing is forced."""
    scores = _play(_scene(latch=PRESSED), _scene(card=CARD_START + (0, 0, 0.003), latch=PRESSED),
                   _scene(card=RAISED, latch=PRESSED), _scene(card=RAISED), _scene(card=OUT))
    assert scores == [0.0, 0.0, 0.25, 0.25, 0.5]
    assert not T.STATE["broken"]


def test_pressed_less_than_three_mm_is_locked():
    shallow = LATCH_START + (0.0, 0.0, -0.002)
    assert T.score(_scene(card=RAISED, latch=shallow)) == 0.0
    assert T.STATE["broken"]


def test_raised_without_pressing_breaks_the_latch():
    assert T.score(_scene(card=RAISED)) == 0.0
    assert T.STATE["broken"]
    put_right = _scene(card=IN_TRAY, latch=PRESSED)
    assert T.score(put_right) == 0.0 and not T.success(put_right)
    T.reset_memory()
    assert T.score(put_right) == 1.0


def test_press_then_let_go_then_pull_breaks_the_latch():
    """The reviewers' exploit: press the button, let go, then pull the locked card past its 60 N. The press is not
    remembered, so the pin is back in the hole and the card pulled out of it breaks the latch."""
    steps = (_scene(latch=PRESSED), _scene(), _scene(card=RAISED), _scene(card=OUT), _scene(card=IN_TRAY))
    assert _play(*steps) == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert T.STATE["broken"]
    assert not T.success(steps[-1])


def test_press_lift_a_little_let_go_then_pull_breaks_the_latch():
    """Raised 3 mm with the button down, then let go: the pin drops into the hole (play 5 mm), and lifting on breaks it."""
    low = CARD_START + (0.0, 0.0, 0.003)
    assert _play(_scene(latch=PRESSED), _scene(card=low, latch=PRESSED), _scene(card=low)) == [0.0, 0.0, 0.0]
    assert not T.STATE["broken"] and T.STATE["engaged"]
    assert T.score(_scene(card=OUT)) == 0.0
    assert T.STATE["broken"]


def test_let_go_and_set_back_down_locks_again():
    """Out at 6 mm with the button up is free, but set back to under 5 mm the pin is in the hole once more."""
    low = CARD_START + (0.0, 0.0, 0.002)
    assert _play(_scene(latch=PRESSED), _scene(card=RAISED, latch=PRESSED), _scene(card=RAISED), _scene(card=low),
                 _scene(card=OUT)) == [0.0, 0.25, 0.25, 0.0, 0.0]
    assert T.STATE["broken"]


def test_pull_sideways_without_pressing_breaks_the_latch():
    assert T.score(_scene(card=CARD_START + (0.05, 0.0, 0.0))) == 0.0
    assert T.STATE["broken"]


def test_small_deflection_is_not_broken():
    assert T.score(_scene(card=CARD_START + (0.0, 0.0, 0.003))) == 0.0
    assert not T.STATE["broken"]


def test_pressed_at_the_step_of_the_pull_is_not_broken():
    assert T.score(_scene(card=RAISED, latch=PRESSED)) == 0.25
    assert not T.STATE["broken"]


def test_the_bond_follows_the_button():
    """What the step hook does with the bond: off while the button is down, on when it is up and the card seated, and
    nothing while the card is raised."""
    assert T.bond_command(_scene(latch=PRESSED)) == "release"
    assert T.bond_command(_scene(card=RAISED, latch=PRESSED)) == "release"
    assert T.bond_command(_scene()) == "relock"
    assert T.bond_command(_scene(card=CARD_START + (0.0, 0.0, 0.0005))) == "relock"
    assert T.bond_command(_scene(card=CARD_START + (0.0, 0.0, 0.003))) == ""
    assert T.bond_command(_scene(card=RAISED)) == ""
    # the first press does not release the bond for good
    assert [T.bond_command(x) for x in (_scene(latch=PRESSED), _scene())] == ["release", "relock"]


def test_relock_enables_the_released_joint_again(monkeypatch):
    import bonds

    calls = []
    monkeypatch.setattr(bonds, "_set_bond", lambda name, enabled: calls.append((name, enabled)))
    monkeypatch.setitem(bonds._PRIMS, "card", object())
    bonds._RELEASED.discard("card")
    bonds.relock("card")  # nothing to undo
    assert calls == []
    bonds.release("card")
    bonds.release("card")  # a second call changes nothing
    assert bonds.released("card") and calls == [("card", False)]
    bonds.relock("card")
    assert not bonds.released("card") and calls == [("card", False), ("card", True)]
    bonds.release("card")
    bonds._restore_bonds()  # a reset locks it again
    assert not bonds.released("card") and calls[-1] == ("card", True)


# -- the slot ----------------------------------------------------------------------------------------------------------


def test_prying_the_card_in_the_slot_breaks_it():
    tilted = _state(CARD_START + (0.0, 0.0, 0.003), quat=_quat((1.0, 0.0, 0.0), 6.0))
    assert T.score(_scene(card=tilted, latch=PRESSED)) == 0.0
    assert T.STATE["broken"]


def test_a_small_tilt_in_the_slot_is_fine():
    tilted = _state(CARD_START + (0.0, 0.0, 0.003), quat=_quat((1.0, 0.0, 0.0), 3.0))
    T.score(_scene(card=tilted, latch=PRESSED))
    assert not T.STATE["broken"]


def test_tilting_the_card_once_it_is_out_of_the_slot_is_fine():
    tilted = _state(OUT, quat=_quat((1.0, 0.0, 0.0), 40.0))
    assert T.score(_scene(card=tilted, latch=PRESSED)) == 0.5
    assert not T.STATE["broken"]


# -- let go and lying on the tray's floor ------------------------------------------------------------------------------


def test_out_of_slot_half():
    assert T.score(_scene(card=OUT, latch=PRESSED)) == 0.5
    assert T.score(_scene(card=(0.45, -0.10, T.BOARD_TOP + 0.01), latch=PRESSED)) == 0.5  # laid on the board


@pytest.mark.parametrize("pose,quat", [(FLAT_ON_THE_FOAM, LYING), (STANDING_IN_TRAY, (0, 0, 0, 1)),
                                       (ON_ITS_END, ENDWISE)])
def test_let_go_and_lying_on_the_floor_succeeds(pose, quat):
    objects = _scene(card=_state(pose, quat=quat), latch=PRESSED)
    T.STATE["held"] = False
    assert T.success(objects) and T.score(objects) == 1.0


def test_held_in_the_tray_is_half():
    """Full credit needs the hands off the card: a gripper on it, even with the card on the floor, is half."""
    objects = _scene(card=_state(STANDING_IN_TRAY), latch=PRESSED)
    T.STATE["held"] = True
    assert not T.success(objects) and T.score(objects) == 0.5


def test_held_above_the_tray_half_even_when_still():
    """The reviewers' exploit: a card held still above the tray's floor. Not on the floor, so not 1.0."""
    for z in (0.005, 0.02, 0.10):
        T.reset_memory()
        card = _state((*T.TRAY_XY, T.TRAY_FLOOR_Z + T.CARD_SIZE[2] / 2 + z))
        objects = _scene(card=card, latch=PRESSED)
        T.STATE["held"] = True
        assert not T.success(objects) and T.score(objects) == 0.5
        T.STATE["held"] = False  # even with no hand seen on it, a card in the air is not supported by the floor
        assert not T.success(objects) and T.score(objects) == 0.5


def test_leaning_on_a_wall_is_not_on_the_floor():
    """Tilted, its lower edge on the foam and its top against a wall: four corners are not on the floor."""
    leaning = _state((*T.TRAY_XY, T.TRAY_FLOOR_Z + 0.04), quat=_quat((0.0, 1.0, 0.0), 30.0))
    objects = _scene(card=leaning, latch=PRESSED)
    assert not T.on_tray_floor(objects) and not T.success(objects) and T.score(objects) == 0.5


def test_a_card_sunk_through_the_floor_is_not_on_it():
    sunk = _state((*T.TRAY_XY, T.TRAY_FLOOR_Z + T.CARD_SIZE[2] / 2 - 0.01))
    assert not T.on_tray_floor(_scene(card=sunk))


def test_in_tray_still_moving_half():
    objects = _scene(card=_state(STANDING_IN_TRAY, vel=(0.05, 0.0, 0.0)), latch=PRESSED)
    assert not T.success(objects) and T.score(objects) == 0.5


def test_on_the_board_beside_the_tray_is_half():
    beside = _state((0.24, 0.26 + 0.2, T.TOP + T.CARD_SIZE[2] / 2))
    objects = _scene(card=beside, latch=PRESSED)
    assert not T.in_tray(objects) and not T.success(objects) and T.score(objects) == 0.5


def test_held_by_needs_a_gripper_closed_on_the_card():
    objects = _scene(card=_state(STANDING_IN_TRAY))
    assert T.held_by(objects, HAND_ON_THE_CARD)
    assert not T.held_by(objects, HAND_AWAY)  # opened wide and back away: let go
    assert not T.held_by(objects, {})
    pads_open = {"left_robot": (np.array([*T.TRAY_XY, T.TRAY_FLOOR_Z + 0.05]), 0.09)}
    assert not T.held_by(objects, pads_open)  # round the card, opened to 9 cm: not touching it
    shut = {"left_robot": (np.array([*T.TRAY_XY, T.TRAY_FLOOR_Z + 0.05]), 0.0)}
    assert T.held_by(objects, shut)  # shut on or against it
    away = {"left_robot": (np.array([*T.TRAY_XY, T.TRAY_FLOOR_Z + 0.25]), 0.0205)}
    assert not T.held_by(objects, away)  # closed, far above it


def test_card_on_floor_zero():
    assert T.score(_scene(card=(0.2, 0.6, T.TOP - 0.5), latch=PRESSED)) == 0.0


def test_non_finite_zero():
    assert T.score(_scene(card=(math.nan, 0.0, 0.0))) == 0.0


def _at(steps, label):
    """When the waypoint ``label`` is reached (s from the start) and the waypoint."""
    t = 0.0
    for w in steps:
        t += w.duration
        if w.label == label:
            return t, w
    raise KeyError(label)


def test_demo_presses_before_it_lifts():
    plan = T.demo()
    pressed, press = _at(plan["right"], "press")
    held, _ = _at(plan["right"], "hold")
    closed, _ = _at(plan["left"], "close")
    lifted, lift = _at(plan["left"], "lift")
    assert pressed < closed < lifted <= held
    assert press.finger == T.FINGER_CLOSED  # pressing with the fingertips closed
    assert lift.pos[:2] == plan["left"][2].pos[:2] and lift.pos[2] - plan["left"][2].pos[2] > 0.03  # straight up


def test_demo_lets_go_just_above_the_floor():
    """The left hand opens with the card's bottom 3 mm above the tray's floor, so it lands and is let go of."""
    plan = T.demo()["left"]
    _, release = _at(plan, "release")
    _, lower = _at(plan, "lower")
    assert release.finger == T.FINGER_OPEN and release.pos == lower.pos
    bottom = release.pos[2] - T.PAD_TIP_PAST_TCP + T.GRIP_DEPTH - T.CARD_SIZE[2]
    assert bottom - T.TRAY_FLOOR_Z == pytest.approx(0.003)


def test_demo_press_depth():
    tips = T.PRESS_Z - T.PAD_TIP_PAST_TCP
    assert T.LATCH_TOP - 0.02 < tips < T.LATCH_TOP - T.LATCH_TRAVEL  # past full travel: the arm must lean on it


def test_demo_grips_the_top_of_the_card():
    tips = T.GRIP_Z - T.PAD_TIP_PAST_TCP
    assert T.CARD_TOP - 0.03 < tips < T.CARD_TOP
    assert T.TRAVEL_Z - T.PAD_TIP_PAST_TCP + T.GRIP_DEPTH - T.CARD_SIZE[2] > T.TOP + T.TRAY_WALL_H  # clears the walls


def test_demo_keeps_the_arms_apart():
    """Each arm works in its own half of the table: the right arm on the latch, the left on the card and the tray."""
    plan = T.demo()
    assert max(w.pos[1] for w in plan["right"]) < -0.15
    assert min(w.pos[1] for w in plan["left"]) > -0.03
    assert min(w.pos[1] for w in plan["left"]) - max(w.pos[1] for w in plan["right"]) > 0.15  # wrists 15 cm apart


def test_board_parts_turn_together():
    for box in (T.MAT, T.BOARD, *T.RAILS, *T.HOUSING, T.CARD, T.LATCH):
        assert box.yaw == T.TURN
    assert T.LATCH.pos[1] < T.CARD.pos[1] - T.CARD_SIZE[0] / 2  # the latch on the right of the card
    assert T.MAT.pos[1] + T.MAT_SIZE[0] / 2 < T.TRAY_XY[1] - T.TRAY_OUTER[1] / 2  # the tray clear of the mat


@pytest.mark.parametrize("arm", ["left", "right"])
def test_demo_reachable(arm):
    plan = ArmPlan(T.demo()[arm], BASES[arm])
    for k in range(int(plan.duration * 20) + 1):
        plan.action7(k / 20)
    assert plan.worst_ik_mm < 2.0 and plan.worst_ik_deg < 2.0, (arm, plan.worst_ik_mm, plan.worst_ik_deg)


# -- the meshes --------------------------------------------------------------------------------------------------------


def test_every_dressed_box_exists_and_its_mesh_is_there():
    names = {o.name for o in T.OBJECTS}
    for name, file in T.LOOKS.items():
        assert name in names
        assert (HERE / "looks" / file).is_file()


@pytest.mark.parametrize("name", ["card", "latch"])
def test_scored_meshes_fit_in_their_boxes(name):
    """The card's and the latch's meshes stay inside their colliders, so what the cameras see is what collides."""
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(HERE / "looks" / T.LOOKS[name]))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    half = np.asarray({"card": T.CARD_SIZE, "latch": T.LATCH_SIZE}[name]) / 2
    assert np.all(np.asarray(box.GetMax()) <= half + 1e-5)
    assert np.all(np.asarray(box.GetMin()) >= -half - 1e-5)
    assert np.all(np.asarray(box.GetMax()) - np.asarray(box.GetMin()) >= 2 * half - 0.002)  # and fill them
