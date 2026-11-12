# Press the slot latch and lift the graphics card out

Pulling a graphics card from a dead mainboard, the way an e-waste recycler or an IT asset disposal technician does it.
The PCIe slot's retention latch locks the card: pulled against the latch, the card does not come out until the latch
breaks, and a broken latch scores 0 for the rest of the trial. The slot's rails let the card out straight up only. One
hand presses the latch down and keeps it down while the other lifts the card out, lets go of it and sets it in a padded
tray, where it must lie on the tray's floor.

| File | What it is |
|---|---|
| `latch_and_lift_graphics_card.py` | The `yam_arena` task file: scene, scoring (`success`, `score`) and the scripted `demo()` |
| `bonds.py` | `Bonded`: a task-file box held by a bond that breaks at a set pull and twist, released and locked again on cue. `Sliding`: a box on a sprung prismatic joint |
| `looks.py` | Puts a detailed mesh on a box at spawn and hides the plain box, which still collides |
| `looks/` | The meshes (`*.usdc`) and their textures, and `make_looks.py`, the Blender script that models and exports them |
| `test_latch_and_lift_graphics_card.py` | Offline checks: the file loads, the linkage numbers agree, every scoring outcome, the press-release-pull and held-above-the-floor exploits, the demo's order, depths and reach (no GPU) |
| `task.json` | The task's details for the board |

## The lock and the latch

**The linkage (real hardware).** The card's near end carries a steel end plate with a hole 9 mm tall, its lower edge
1.5 mm above the card's bottom edge. A spring-loaded steel pin 3 mm across, its axis 8 mm above the board, passes
through the hole 5 mm above the hole's lower edge, so a seated card can rise 5 mm before the edge meets the pin. The pin
is on a 5 cm steel push rod in a covered groove along the slot's axis, flush with the board. The rod's other end has a
45 degree wedge under the orange button. Pressing the button 3 mm slides the rod 3 mm and pulls the pin 3 mm out of the
hole (it was 2.5 mm in): the card is free while the button is down. A spring under the button (500 N/m, 0 N at rest,
2 N at its full 4 mm travel) pushes the pin back into the hole when the button is let go. The pin holds 60 N.

| Case | What the hardware does |
|---|---|
| Button never pressed, card pulled | the hole's edge bears on the pin. Past 60 N the pin is overloaded: the latch is broken |
| Button down 3 mm or more | the pin is clear of the hole: the card lifts out freely |
| Button let go with the card raised less than 5 mm | the pin springs back into the hole. Raised 5 mm or more after that, the edge meets the pin: the latch is broken |
| Button let go with the card raised more than 5 mm | the pin meets the plate's face and cannot enter the hole: the card stays free |
| Card tilted 5 degrees or more while its bottom is still between the rails | the rails pry the slot: the slot is broken |

**How the simulation stands in for it.** The hook, rod and wedge are not bodies. Two PhysX joints, authored by
`bonds.py` when the boxes spawn, and a step hook that reads the button at every control step, do the same job:

| Part | Joint | Behaviour |
|---|---|---|
| lock | fixed joint between the card and the board, `physics:breakForce` 60 N | holds the card in the slot. Pulled harder than 60 N it breaks for good, which the scorer sees as the card rising with the button up |
| latch | prismatic joint along z between the latch and the board, 4 mm of travel, a 500 N/m spring back to the top (2 N at full travel) | the sprung button. Down 3 mm or more is "pressed" |
| step hook | reads the button and the card every control step | button down 3 mm or more: the lock joint is switched off. Button up and the card within 1 mm of seated: the lock joint is switched on again. A press is never remembered |

Everything else is `yam_arena` boxes. No assets are downloaded.

## What it looks like

Each box keeps its collider, and `looks.py` draws a mesh in its place:

| Box | Mesh |
|---|---|
| `card` | a dual-fan graphics card: shroud, fans, fin stack, heat pipes, backplate, gold edge connector, steel bracket with ports |
| `latch` | the orange button, with ridges and a hook toward the card |
| `mainboard` | the textured board, the PCIe slot under the card, the rails, the latch housing, the socket, memory slots and small parts |
| `mat`, `tray_floor` | the anti-static mat, and the tray with its walls and pick-and-pluck foam floor |

The card's and the latch's meshes fit inside their boxes (a test checks it), so what the cameras see is what
collides. The board's parts are drawn only: none is taller than 1.2 cm, and all are at least 1.5 cm from the card and the latch.
To change them, edit `looks/make_looks.py` and run it in Blender 4.2 or later:
`blender -b -P looks/make_looks.py`.

## Run it

On the YAM Brev launchable, from this folder, with the Arena environment active
(`source /data/IsaacLab-Arena/yam.env`):

```bash
# Dream-1 (needs DREAM_API_KEY), 2,400 steps
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=latch_and_lift_graphics_card.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Press and hold the orange latch with the right hand, then lift the graphics card straight up with the left hand and put it in the tray on the left." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/latch-and-lift-graphics-card-dream

# Scripted demo (both arms), 600 steps
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=latch_and_lift_graphics_card.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Press and hold the orange latch with the right hand, then lift the graphics card straight up with the left hand and put it in the tray on the left." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/latch-and-lift-graphics-card-demo
```

- The run prints `task_score` at the end. With `-E object_jitter=0` the layout is fixed and no seed is needed.
  Keep it at 0: jitter would move the card and the latch away from their joints.
- The layout keeps the arms apart: the slot runs along y, with the latch on the right arm's side and the card reaching
  toward the left arm, so each arm works in its own half of the table.
- The demo: the right arm, fingers closed, comes down on the latch and leans on it (the pad tips are sent 12 mm below its top,
  the button bottoms out at 4 mm), and keeps it down until the card is well out of the slot. The left arm closes on the
  top 2 cm of the card above its centre once the latch is down, lifts the card straight up out of the slot, turns it a
  quarter turn while carrying it over the tray, lowers it until its bottom is 3 mm above the foam, opens its fingers
  to 9 cm and backs off. The card lands and lies still on the floor. It reaches 1.0 and ends on success at
  step 215 (10.75 s at 20 Hz). A trial takes about 55 s on an L40S, including startup.
- The same demo with the right arm stopping above the latch (`PRESS_DEPTH = -0.02`) scores 0: the card stays locked
  in the slot, or the lock breaks and the trial is scored 0 from then on.
- The same demo with the right arm letting go of the button before the left arm lifts (`PRESS_HOLD_S = 0.2`) also
  scores 0: the pin is back in the hole, the pull breaks the latch, and the card in the tray does not score.
- Dream-1: 0.0 (ran to the step limit without finishing).
- Videos: `inspect-robots video ~/yam-eval/latch-and-lift-graphics-card-demo/<log>.json` writes one MP4 per camera
  next to the frames.
- Offline checks: `python -m pytest -q test_latch_and_lift_graphics_card.py`.

## Scoring

All positions are centres, in metres, in the station frame (x forward, y to the robot's left, from the midpoint of
the arm bases), and heights are above the table top. The simulation reads everything at every control step (20 Hz).

- **Button:** pressed when it is 3 mm or more below its rest. It is read at each step. Nothing about an earlier press
  is remembered.
- **Pin in the hole:** the button up and the card less than 5 mm above its start and less than 3 cm sideways from it.
- **Broken latch:** at any step, the card is 5 mm or more above its start, or 3 cm or more sideways from it, while the
  button is up and the pin was in the hole at the step before. It stays broken: the trial scores 0 from then on, even
  if the card is then put in the tray. Pulling without pressing, pressing and letting go before pulling, and letting go
  at under 5 mm of lift all break it. Letting go with the card already raised 5 mm or more does not.
- **Pried slot:** at any step, the card's height axis is 5 degrees or more from vertical while its centre is less than
  1.2 cm above its start and under 3 cm sideways (its bottom still between the rails). It stays broken: 0 from then on.
- **0** if the card has fallen off the table (centre more than 2 cm below the top) or the state is not finite.
- **Let go:** no gripper has its TCP (the point between the finger pads) within 3 cm of the card's box with the pads
  4 mm or less wider apart than the card's 2 cm thickness. Pads shut on nothing beside the card count as touching it.
  Opened wider, or moved away, the gripper has let go.
- **On the tray's floor:** the card's four lowest corners within 3 mm above the foam floor's top (1 cm above the table),
  so it lies flat on a face, its 18 x 11 cm edge, its 18 x 2 cm face or its 11 x 2 cm end. A card leaning on a wall, held
  up, or resting on another object is not on the floor.
- **1.0 (success, held 1 s):** not broken, no gripper on the card, the card's centre within 9 cm along x and 5 cm along y
  of the tray's centre (0.24, 0.26), on the tray's floor, and still (under 1 cm/s and 0.5 rad/s).
- **0.5:** the card out of the slot, its centre 1.2 cm or more above its start (its bottom edge clears the rails) or
  3 cm or more sideways, but not meeting all of the 1.0 conditions: held in the air, held above or in the tray, or set
  down on the board or anywhere out of the tray.
- **0.25:** the card raised 5 mm or more but still between the rails (it can only have come with the button down).
- **0** otherwise, including doing nothing, pressing the latch without lifting the card, pulling the card without
  pressing the latch, and any trial in which the latch or the slot was broken.

In a real trial, a tape measure gives the card's place in the tray, a ruler beside the card its height in the slot and
its clearance above the foam, and a look at the hands gives "let go". The latch and the slot are
checked by hand for damage after the trial: the pin must slide back in the hole, the rails must be unbent, and the
card must not have been tilted more than 5 degrees in the slot (the check is a protractor or a phone level).
