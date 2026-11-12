# Lift a long thin panel at points that keep it flat

Carrying a long circuit-board panel to a rack, the way a production worker does. The panel is a 0.8 mm FR-4 strip and
bends: lifted by its two ends, its middle sags past the 4 mm bow the line allows, and held near its middle, both ends
droop. About a fifth of its length in from each end keeps it flat. The instruction does not say where to hold. The task
tests choosing where to hold a bending part with two hands, then setting it down and letting go.

| File | What it is |
|---|---|
| `lift_long_panel_no_sag.py` | The `yam_arena` task file: scene, scoring (`success`, `score`) and the scripted `demo()` |
| `panelkit.py` | `Panel`: a chain of rigid segments on springy hinges that bends like a strip of a set stiffness. Hooks the scene, the reset, the control step and the drawing of boxes |
| `fingers.py` | Points on the grippers' finger meshes, which the scorer uses to know the panel was let go |
| `looks/` | The meshes (`*.usdc`) and the texture, and `make_looks.py`, the Blender script that models and exports them |
| `test_lift_long_panel_no_sag.py` | Offline checks: the file loads, every scoring outcome and its memory, the hold-and-never-let-go exploit, the finger test, a static model of the chain, the demo's order and reach (no GPU) |
| `task.json` | The task's details for the board |

## The panel to build

One real panel, and the simulated one stands for exactly it:

| Property | Value |
|---|---|
| Laminate | FR-4, 0.8 mm thick, cut to 60 x 6 cm (60 cm along the length), about 22 GPa. Bare, it weighs 53 g (1.85 g/cm3) |
| Copper | 18 um (half ounce) on both faces, patterned to cover about 40 % of each face |
| Boards | Six 10 x 6 cm boards in one piece. The lines between them are printed, not cut: no V-score, so the strip bends like one piece |
| Population | Single-sided: every part is on the top face, within 2 cm of the centre line, so the lower face is flat and the long edges are clear. Each board carries the same parts: a 44-pin QFP, about 34 SMD resistors and capacitors, and heavy through-hole parts (two 470 uF radial electrolytic capacitors, a 2-way screw terminal, a 2 x 10 pin header) |
| Mass | 120 g, plus or minus 3 g, weighed with all parts. The parts add about 67 g in six equal groups, one at the middle of each 10 cm board (5, 15 ... 55 cm from an end). If the panel weighs light, solder or glue steel M3 nuts on the centre line, the same number on every board |
| Bending stiffness | EI = 0.07 N m2, plus or minus 10 %, measured as below. The laminate alone gives about 0.056, the copper the rest |

**Measure EI before the trials.** Clamp the panel flat on a table edge with 25 cm overhanging, along its length. Read the
free end's height with a ruler or a calibrated video, hang a 20 g weight from the end (0.196 N), and read it again. The
weight lowers the end by 14.6 mm if EI = 0.07 N m2 (EI = 0.196 x 0.25^3 / (3 x drop)). A drop between 13.3 and 16.2 mm is
within 10 %. The parts' own weight is not part of this difference, so the test works on the finished panel. A panel that
fails is reworked (more or less copper) until it passes.

**Why 120 g.** Bow is set by mass over stiffness. The bare strip (53 g) lifted level by its two ends held with its grippers keeping the held segments level bows only about
3 mm, which would let the ends pass. Populated to 120 g it bows over 7 mm. The mass is therefore part of the task, and
a real trial must weigh in.

**Why 4 mm.** 4 mm over 60 cm is 0.67 %, inside the usual 0.75 % bow and twist limit for surface-mount boards (IPC-6012).
It is a flatness limit the panel must still meet when it reaches the rack, a stand-in for the board having to stay
inside what the assembly line takes. A bow this size does not damage the board (the strain is tiny), so the panel does
not have to be replaced after a trial. The task scores bow, it does not model cracking.

## The simulated panel

`panelkit.py` spawns the panel as one PhysX articulation: 20 boxes of 3 x 6 x 0.08 cm along y, each joined to the next
by a revolute hinge about x with a spring and a damper. Each hinge's spring is EI divided by the segment length, so the
chain bends like a uniform strip of bending stiffness EI = 0.07 N m2. The colliders are as thin as the board, 0.8 mm, so
the grippers meet the same 0.8 mm edge a real gripper does. The panel weighs 120 g, spread evenly (the real one carries
its parts in six groups, see below). The hinges allow no stretch, no twist and no sideways bend. It starts flat on a
black bench rail 3 cm high. The rack's rail is orange, 8 cm high and 14 cm farther from the robot.

A static model of the chain (in the tests) lifted at two points gives these bows, for a gripper that keeps the held
segment level ("clamped") and one that lets it turn ("pinned"):

| Held at | Clamped | Pinned |
|---|---|---|
| the ends | 7.6 mm | 38 mm |
| 6 cm in from each end | 2.8 mm | 18 mm |
| 10.5 cm in from each end | 1.3 mm | 8.7 mm |
| 13.5 cm in from each end (the demo holds the segment 12 to 15 cm in) | 1.0 mm | 1.0 mm |
| 16.5 cm in from each end | 2.2 mm | 8.0 mm |

The same model with the real panel's mass in six lumps (a bare strip of 53 g and six groups of 11 g) gives the same bows
within 2 %. In the simulator the pads close on the panel's 0.8 mm edges along x, so a held segment can turn about x, the
bend axis: the "pinned" column is the one that applies, and the runs below agree with it, not with "clamped". The zone
that keeps the panel under 3 mm is then narrow, about a fifth of the length (13 cm) in from each end, plus or minus
1 cm. A real pad grip is somewhere between the two columns, so a real trial holds at least as well. With EI 10 % off either way the hold stays under 3 mm and the ends stay over 4 mm (all in the tests).

## Assets

No SimReady package fits: the panel has to bend, which a rigid SimReady body can't, and no package is a bench-scale
rack. The panel and the two rails are modelled in Blender (`looks/make_looks.py`, run with
`blender -b -P looks/make_looks.py`) as render-only meshes. The physics stays the panel's 20 segment boxes and the rails'
boxes, so what the cameras see is what collides. The panel's texture is drawn in the script with numpy, nothing is
downloaded. The parts and the traces are drawn flat: they are not modelled and add no thickness or mass of their own.

## Run it

On the YAM Brev launchable, from this folder, with the Arena environment active
(`source /data/IsaacLab-Arena/yam.env`):

```bash
# Dream-1 (needs DREAM_API_KEY), 600 steps
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=lift_long_panel_no_sag.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Carry the long blue circuit-board panel to the orange rack with both arms, set it down on the rack's rail and let go of it." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/lift-long-panel-no-sag-dream

# Scripted demo (both arms), 600 steps
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=lift_long_panel_no_sag.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Carry the long blue circuit-board panel to the orange rack with both arms, set it down on the rack's rail and let go of it." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/lift-long-panel-no-sag-demo
```

- The run prints `task_score` at the end. With `-E object_jitter=0` the layout is fixed and no seed is needed.
- Dream-1 was not run on this version of the task.
- The demo: both arms come down beside the panel at y = +-17 cm (13 cm in from each end), pinch its 6 cm width with the
  pad tips 1.3 cm under it, lift it 12 cm together, carry it 14 cm forward at that height, lower it onto the rack's rail,
  open and back off. It reaches 1.0 and ends on success at step 237 (11.9 s at 20 Hz): the panel is on the rail, still,
  with the fingers 1.7 cm clear of it and a peak bow of 3 mm or less. A trial takes about 75 s on an L40S, more when the
  GPU is shared.
- Offline checks: `python -m pytest -q test_lift_long_panel_no_sag.py`.

## Scoring

Heights are above the table top. Distances are in the task's frame, metres in the code. The bow is the greatest distance of any of the panel's 20 segment centres from the
straight line through the first and the last (the centres 1.5 cm in from each end), read at every control step.

- **Peak bow:** the largest bow at any step of the trial, from the start to the end. The scorer remembers it, and clears
  it on every reset.
- **Lifted clear:** at some step, every segment is at least 3 cm above its start height. The scorer remembers it too.
- **Let go:** at the step, no point on either gripper's fingers is within 3 mm of any of the panel's 20 segment boxes.
  `fingers.py` reads the finger meshes from the running simulation every control step and `touching()` measures the
  distance to each box, turned by the segment's pitch. The pads, the fingers' tips and their bodies all count.
- **0** if the peak bow goes over 4 mm at any step, if any segment falls more than 2 cm below the table top, if the
  state is not finite, or if the panel was never lifted clear.
- **1.0 (success, held 1 s):** the panel lies on the rack's rail, is still and is let go, with a peak bow of 3 mm or
  less. On the rail means every segment's centre within 2 cm of the rail's line along x and within 1 mm of lying on it
  (height of the rail's top plus half the panel's thickness, its lower face on the rail), and the panel's centre within
  2 cm of the rail's middle along y. Still means every segment under 1 cm/s and 0.5 rad/s. A panel carried to the rack
  and held there still, without letting go, does not score 1.0.
- **0.75:** the same, with a peak bow over 3 mm and up to 4 mm.
- **0.5:** lifted clear with a peak bow of 4 mm or less, but not on the rack, still and let go at the end: still
  carried or held (above the rack or on it), or set down anywhere else, the bench rail included.
- So the score is 0, 0.5, 0.75 or 1.0, and doing nothing scores 0.

### Measuring it in a real trial

Nothing touches the panel during the trial, and the peak bow is read over the whole transfer, not at the end.

1. **Camera.** One fixed camera, 4K (3840 px wide), 60 frames per second or more, exposure 1/500 s or shorter, at the
   height of the carry, 2.5 m from the panel and looking along x (from the robot's side, between the arms, or from behind
   the rack), so it sees the panel edge-on. Zoom so the 60 cm panel fills about 70 % of the width (about 0.25 mm per
   pixel). A matt white board behind the path gives a dark silhouette. Do not touch the camera during the trials.
2. **Calibrate once.** Film a 10 mm checkerboard in the panel's plane (an 8 x 6 board, 30 poses), correct the lens
   distortion and check the reprojection error is under 0.3 px. Then measure a 3.00 mm gauge block laid on the
   bench rail: it must read 3.00 mm, plus or minus 0.3 mm.
3. **Read each frame.** Find the panel's silhouette, take its top edge column by column, and move down half its
   thickness (0.4 mm) to get the centreline. The scale comes from the panel itself: the 60 cm between its two ends is
   the distance between the silhouette's end columns, so the reading holds as the panel moves toward the camera. Draw
   the straight line through the centreline's points 1.5 cm in from each end, and the bow of the frame is the largest
   distance of any point of the centreline from that line. A span hidden behind a gripper (at most 4 cm) is filled in
   straight from the visible columns either side of it.
4. **Peak bow** is the largest bow over every frame from the first lift-off to the end of the hold. Compare it with
   3 mm and 4 mm. Check before the trials that the panel lying flat on its rail reads under 0.5 mm.
5. **At the end, in the same video.** On the rack: a 1.0 mm feeler gauge does not slide between the rail's top and the
   panel's lower face anywhere along the rail, and a tape gives the position within 2 cm on x and y. Still: over the last
   second the silhouette moves under 1 cm. Let go: both grippers are open and a 3 mm feeler gauge slides between each pad
   and the panel's edge (or the video shows daylight round every finger).

## Cheat and ablation variants

`scripts/task-ablate.sh lift-long-panel-no-sag <name> GRIP_Y=<m>` runs the demo on a copy with both hands moved to
y = +-`GRIP_Y` (the demo's `GRIP_Y = 0.17` is 13 cm in from each end, and the panel's ends are at 0.30). Results of the
same carry, all scripted, `-E object_jitter=0`:

| Hands | Score | Why |
|---|---|---|
| `GRIP_Y = 0.17`, 13 cm from each end (the demo) | 1.0 | flat, set down on the rack, fingers clear |
| `GRIP_Y = 0.15`, 15 cm from each end | 0.0 | the pads sit on the edge of the next segment in: the model's pinned bow there is about 8 mm |
| `GRIP_Y = 0.24`, 6 cm from each end | 0.0 | the model's pinned bow is about 18 mm |
| `GRIP_Y = 0.285`, at the very ends | 0.0 | the model's pinned bow is about 38 mm |

The three failures are the panel bowing past 4 mm on the way: each ends 0.0 because the peak is remembered, wherever the
panel finally lies. Holding in the middle can't be scripted: with both hands within 15 cm of the centre line the arms
can't reach the rack. One-arm holds and drags along the bench rail were not run: the scorer gives 0 to a panel that never
has every segment 3 cm above its start (tested offline), and the chain model gives a bow far over 4 mm for any hold
within 6 cm of the middle.

The reviewers' exploit, carrying the panel to the rack and holding it there still without letting go, can't be scripted
(the demo's release is fixed). It is covered by the tests (`test_a_panel_held_on_the_rack_is_not_a_full_score`,
`test_a_panel_held_still_just_over_the_rack_is_not_a_full_score`), which put the finger flag up and show 0.5 and no
success, and by the finger-distance tests that decide the flag.
