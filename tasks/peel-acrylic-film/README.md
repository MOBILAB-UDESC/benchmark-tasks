# Hold the acrylic sheet down and peel its protective film off by the tab

Peeling the protective film off a new acrylic sheet, an everyday chore (new screens, appliance panels, signs, and every
laser-cutting shop and makerspace).
The film peels when its edge is lifted, strip by strip, while the hand comes back over the sheet: dragged along the
sheet, the tab tears off before the glue lets go, and a torn tab scores 0. The sheet is light and loose, so the first
pull tips it up with the film unless the other hand holds it down, and a lifted sheet scores 0. One hand holds the
sheet, the other peels the film and drops it in a basket.

| File | What it is |
|---|---|
| `peel_acrylic_film.py` | The `yam_arena` task file: scene, scoring (`success`, `score`) and the scripted `demo()` |
| `bonds.py` | PhysX joints between task-file boxes, authored when they spawn: `Joint` (fixed, revolute or ball, breakable), `tune` (solver settings), and step and reset hooks |
| `simready.py` | Loads the SimReady basket, pinned to one dataset revision |
| `looks.py` | Puts a detailed mesh on a box at spawn and hides the plain box, which still collides |
| `looks/` | The meshes (`*.usdc`) and their textures, and `make_looks.py`, the Blender script that models and exports them |
| `test_peel_acrylic_film.py` | Offline checks: the file loads, the glue, hinges and tab, every scoring outcome, the demo's order, arc, arm clearance and reach, and the meshes inside their boxes (no GPU) |
| `task.json` | The task's details for the board |

## The film

The film is a chain of four rigid paper strips, 3.75 x 10 cm and 1.5 mm thick, on the sheet's right half. `bonds.py`
authors these PhysX joints when the boxes spawn:

| Part | Joint | Behaviour |
|---|---|---|
| glue | a fixed joint between each strip (its first body) and the sheet, under the strip's edge toward the robot, `physics:breakTorque` 0.06 N m, `physics:breakForce` 8 N | pulled up at its far edge, a strip twists its glue past 0.06 N m and comes unstuck. Pulled along the sheet, it holds to 8 N |
| joints between strips | a ball joint (a PhysX D6 joint with its translations locked) at the middle of each cut, free, lightly damped | the peeled strips swing, fold and twist like paper, and the peeled part pulls up the next strip's edge. No turning passes from one strip to the next, so only a pull up at a strip's edge breaks its glue |
| fold | a ball joint between the tab and the first strip, with a weak spring that keeps the tab upright, `physics:breakForce` 4 N | the tab turns on its fold. Pulled with 4 N, it tears off |

How hard the peel pulls was measured in the simulator, with the scripted demo and the left hand lifted 6 cm off the
sheet: with the glue at 0.02 N m the strips came off before the loose sheet moved, at 0.04 N m the sheet tipped up 62
degrees, at 0.06 N m it tipped 14 degrees and slid 2.7 cm. With the left hand pressing the near left corner (pad tips
sent 3 cm into the sheet), the sheet did not move at 0.06 N m. Revolute hinges between the strips let a hand that
twists the tab break the glue (the first versions came off without the sheet ever tipping), hence the ball joints.

A step hook records each strip that has come unstuck (its centre 1 cm or more from its place, measured on the sheet)
a torn tab (its centre 2 cm or more from its fold on the first strip), and a lifted or dragged sheet (its centre more
than 2 cm from its start, or tilted more than 10 degrees).

## Building the film for a real trial

Every number below is the one the simulator uses (`GLUE_TWIST`, `GLUE_HOLD`, `TAB_HOLD`, `STRIP_W`, `TAB_SIZE`). The
simulator's bonds are idealised, so a real film is built and then calibrated with a spring scale until it measures
the same. The materials and the starting widths are a starting point and have not been tested on a real robot: the
acceptance numbers in the calibration decide. Tools: a 0 to 10 N spring scale or digital force gauge (0.1 N steps), a
0.1 g kitchen scale, a ruler, a craft knife, a perforating wheel and a few offcuts of the same acrylic.

**Sheet and mat:** cast acrylic 150 x 200 x 3 mm (about 105 g: weigh it), on a 220 x 240 x 3 mm self-healing cutting
mat. The sheet is loose (nothing holds it), and its protective film is peeled off before the trial.

**Film:** 70 g/m2 brown kraft paper covering the sheet's right half (the robot's right: 150 mm along x by 100 mm
along y, y from -0.15 to -0.05), cut into four strips 37.5 x 100 mm. The strips lie side by side along x with their
cuts parallel to y, as in the simulator. The printing on the film is for the camera only.

| Part | Build | What it must do |
|---|---|---|
| Strip joins (3 cuts) | one 19 mm wide strip of clear cellophane tape along each cut on the top face, 100 mm long, centred on the cut (9.5 mm on each strip), the paper edges touching | bends freely like paper and does not come apart: it must hold at least 8 N pulled along the film (twice the tab's 4 N) |
| Glue (4 bands) | one 100 mm long band of low-tack removable tape (repositionable or painter's tape for delicate surfaces) on the underside of each strip, along its edge toward the robot, starting 6 mm wide, pressed on with one pass of a finger | lets the strip go when its far edge is lifted with 0.06 N m, which is 1.6 N pulled straight up at the far edge, 37.5 mm from the glued edge. Pulled along the sheet it must hold at least 8 N |
| Tab (1) | kraft card folded and glued in layers to 30 mm wide, 25 mm tall and 3 mm thick, trimmed to 4 g, standing on strip 1's far edge on a 30 mm wide paper hinge (a 30 mm wide kraft strip glued across the fold under the tab and on strip 1's top, a few mm on each side) with a row of perforations along the fold | stands upright on its own, leans over when pushed, and tears at the hinge at 4 N, in any pull direction |

**Calibration (do it on spare coupons from the same batch before each film, and again for each new roll of tape):**

1. **Peel force, 1.6 N.** Stick one spare 37.5 x 100 mm strip with its glue band on an acrylic offcut lying on the mat.
   Tape a 12 mm loop of thread to its far edge. Hook the scale to the loop and pull straight up, slowly (about 1
   cm/s). The strip must let go between 1.4 and 1.8 N. Too high: make the band narrower in 1 mm steps (a new coupon
   each time). Too low: widen it. Use the width that passes for all four glue bands. Test 5 coupons: all inside 1.4
   to 1.8 N.
2. **Hold along the sheet, 8 N.** On another coupon, pull the strip along the sheet with the scale held level at the
   strip's far edge, in the direction of the film's peel (toward the robot). It must not let go below 8 N. The
   simulator lets go at exactly 8 N. A real glue line only approximates that, and its job is the same: to outlast the
   tab, which tears at 4 N, so that a pull along the sheet tears the tab and not the glue. If a coupon lets go
   below 8 N at the width that passed step 1, switch to a tape with a stronger hold in shear and repeat step 1 with a
   narrower band.
3. **Tab tear, 4 N.** Stand the tab on a spare strip stuck as in step 1. Hook the scale to the tab's top (3 cm up)
   and pull along the sheet toward the robot, then once more at 45 degrees upward. It must tear at 4 N, between 3.5
   and 4.5 N, both ways. Too strong: perforate more (or cut one more bridge). Too weak: leave out perforations. Test
   5 tabs: none below 3.5 N or above 4.5 N.
4. **Strip joins hold.** Pull a joined pair of strips apart along the film: no failure below 8 N.
5. **The sheet tips without a hand (the premise of the task).** Make a whole film, lay it on the sheet with nothing
   holding it and peel the first strip by the tab straight up and back toward you. The sheet must tilt up or slide
   (the simulator: tilts 14 degrees and slides 2.7 cm at 0.06 N m). Then lay the film again, press the sheet's near
   left corner with one finger and peel: the sheet must stay flat. If it never moves, the glue is too weak. If it
   moves even with the finger down, it is too strong: go back to step 1.
6. **Mass and size.** Strips 3.75 x 10 cm, tab 3 x 2.5 cm and 3 mm thick, about 4 g (0.1 g scale), sheet about 105 g.
   The real strips weigh under 1 g (the simulator's 5 g is for numerical stability).

## What it looks like

Each box keeps its collider, and `looks.py` draws a mesh in its place:

| Box | Mesh |
|---|---|
| `mat` | a green self-healing cutting mat with its 1 cm grid, 5 cm lines, 45 degree guides and knife scores |
| `sheet` | glossy opal white acrylic |
| `strip_1` to `strip_4` | brown kraft masking paper, printed "PROTECTIVE FILM" across the four strips as one piece |
| `tab` | the same paper, with a printed arrow |

The basket is the SimReady `misc_basket_plastic_a01` (revision `9d4556bfe80868c1d27a7a7ba1c2774e63e8cedd`, CC-BY 4.0),
scaled to 0.6, fixed, colliding through its own signed distance field. `simready.py` downloads it into
`~/.cache/simready` the first time the scene is built. The scored meshes fit inside their boxes (a test checks it).
To change them, edit `looks/make_looks.py` and run it in Blender 4.2 or later: `blender -b -P looks/make_looks.py`.

## Run it

On the YAM Brev launchable, from this folder, with the Arena environment active
(`source /data/IsaacLab-Arena/yam.env`):

```bash
# Dream-1 (needs DREAM_API_KEY), 2,400 steps
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=peel_acrylic_film.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hold the white acrylic sheet down with your left hand. With your right hand, pinch the brown paper tab on the far edge of the film and peel the film off by lifting it up and back toward you, then drop the film in the blue basket on the right." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/peel-acrylic-film-dream

# Scripted demo (both arms), 600 steps
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=peel_acrylic_film.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hold the white acrylic sheet down with your left hand. With your right hand, pinch the brown paper tab on the far edge of the film and peel the film off by lifting it up and back toward you, then drop the film in the blue basket on the right." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/peel-acrylic-film-demo
```

- The run prints `task_score` at the end. With `-E object_jitter=0` the layout is fixed and no seed is needed.
  Keep it at 0: jitter would move the strips away from their glue and joints.
- Each run is one episode: a glue joint or tab that breaks stays broken until the scene is built again.
- The layout keeps the arms apart: the film is on the sheet's right half, so the right arm peels on the right while
  the left arm holds the bare left half by its near corner.
- The demo: the left arm, fingers closed, presses the bare acrylic 2 cm from the near left corner (the pad tips are sent
  3 cm below its top, so it leans on it) and holds it until the film is off. The right arm, turned so its fingers
  close across the tab, pinches the tab's top 1.4 cm (the pad tips 11 mm above the film), then peels in one 6 s
  stroke up and back toward the robot, from the far edge to 19.5 cm above the near edge: the strips come up one after another and the film leans up between
  the hand and the peel line. It carries the hanging film over the basket, turning the hand so the fingers open along
  the basket's long side, lowers the film onto the basket's floor and lets go. It reaches 1.0 and ends on success at
  step 348 (17.4 s at 20 Hz). A trial takes about 2.5 minutes on an L40S, including startup.
- The same demo with the left hand stopping 6 cm above the sheet (`PRESS_DEPTH = -0.06`) scores 0: the first pull
  tips the loose sheet up with the film, and the scorer records it as lifted.
- Dream-1 scored 0.0 in its one run: both hands reach over the sheet and the right one hovers by the tab, but it
  never pinches the tab and no strip comes off.
- Videos: `inspect-robots video ~/yam-eval/peel-acrylic-film-demo/<log>.json` writes one MP4 per camera next to the
  frames.
- Offline checks: `python -m pytest -q test_peel_acrylic_film.py`.

## Scoring

All positions are centres, in metres, in the station frame (x forward, y to the robot's left, from the midpoint of
the arm bases), and heights are above the table top.

- **Peeled:** a strip whose centre has been 1 cm or more from its place on the sheet, measured in the sheet's own frame
  (so a sheet dragged with its film still stuck peels nothing). The scorer remembers: a peeled strip stays peeled,
  even if it falls back onto the sheet.
- **Torn:** the tab's centre 2 cm or more from its fold on the first strip (turning on its fold, it stays 1.25 cm from
  it). The scorer remembers: the trial scores 0 from then on.
- **Lifted:** the sheet's centre more than 2 cm from its start, or the sheet tilted more than 10 degrees, at any control
  step (lifted with the film, or dragged). The scorer remembers: the trial scores 0 from then on.
- **0** if the sheet is off the table or the state is not finite.
- **1.0 (success, held 1 s):** all four strips peeled and the centre of every strip and of the tab within 6 cm along x
  and 8 cm along y of the basket's centre (0.235, -0.265), inside its walls, and lower than its 6.9 cm rim, the film
  still (under 2 cm/s and 1 rad/s).
- **0.5:** all four strips peeled, the film not all in the basket (held, dropped on the table, or hanging over the rim).
- **0.25:** two or three strips peeled.
- **0** otherwise, including doing nothing or peeling one strip.

In a real trial, a tape measure gives the sheet's place and the film's place in the basket, a look at the sheet tells
which strips have come off, and the tab is checked for a tear.
