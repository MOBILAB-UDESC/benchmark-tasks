# Free a stuck print-in-place hinge by rocking it

A print-in-place hinge comes off the printer fused solid, and whoever finishes the print frees it by rocking the leaf in
small pushes one way and the other. Forcing it round in one swing snaps the leaf at its thin neck instead. Category:
Other (fabrication and repair): the same skill frees any stuck or seized joint in repair work. The task tests one thing: **loosening
a stuck joint with small alternating loads that stay under the part's breaking load**, not with force or with a steady push.

- **Stuck harder than the leaf can take.** The hinge's four layers hold it with up to 0.8 N m. The leaf snaps when they hold
  it with 0.49 N m (3.5 degrees of turn with four layers on, about 7.5 N at the grip). A hard push gets to the snap long
  before it moves the hinge. A steady push below it moves nothing, and nothing loosens.
- **Rocking loosens it.** Each rocking cycle switches one of the four layers off. After four, the hinge turns freely, and the
  leaf swings round to its stop at 130 degrees.
- **Free means free.** Full credit needs the hinge shown to turn freely: once the leaf has been let go at 120 degrees or
  more and has lain still for 1 s, a light test push (at most 0.0325 N m, a force gauge's 0.5 N at 6.5 cm) must turn it
  back 30 degrees within 3 s. A leaf held at 120 degrees in a gripper never gets the test, and a hinge that still holds
  does not turn under it.

**The friction is a programmed rule, not the physics of fused plastic.** The layers are PhysX joints at the hinge, each a
torsion spring (2 N m/rad) capped at 0.2 N m, and the task file switches one off per rocking cycle. Nothing in the
simulation wears or heals the hinge.

| File | What it is |
|---|---|
| `free_printed_hinge.py` | The `yam_arena` task file: scene, rocking rule, scoring (`success`, `score`), the step watcher and the scripted `demo()` |
| `bonds.py` | `Hinged` (a revolute joint plus stuck layers that `unstick()` switches off, `snap()` that breaks the box off, and the free-movement test push that `test_push()` switches on and off), and the step and reset hooks |
| `fingers.py` | Where the grippers' fingers are: points on their collision meshes, from the robot's `tip_left` and `tip_right` bodies, for the "let go" check |
| `colliders.py` | Makes a box collide as several smaller boxes (the base leaf as its plate only, the orange leaf as its plate and knuckle) |
| `looks.py` | Puts detailed meshes on a box at spawn and hides the plain box, which still collides |
| `looks/` | The meshes (`*.usdc`), their textures and `make_looks.py`, the Blender script that models and exports them |
| `test_free_printed_hinge.py` | Offline checks (no GPU): the numbers, angle, rocking counter, every scoring outcome, the swing's clearance, the demo's reach and arm clearance, the meshes in their boxes |
| `task.json` | The task's details for the board |

## The scene

Positions in metres in the station frame (x forward, y to the robot's left, from the midpoint of the arm bases), heights
above the table top. The hinge's axis is vertical at (0.34, -0.12), in the right arm's half.

| Part | Body | Detail |
|---|---|---|
| base leaf | fixed box, 6.9 x 1.3 x 4 cm, colliding as its 5 x 1 x 4 cm plate | dark teal, two barrels drawn at the axis (the printed pin is inside them) |
| orange leaf | free box 8.9 x 1.4 x 4 cm, 50 g, `Hinged` on the base, colliding as a 7.5 x 1 x 4 cm plate and a 1.4 x 1.4 x 2.4 cm knuckle | orange, revolute at the axis, -12 to 130 degrees, damping 0.002 N m s/rad, four stuck layers |
| card | fixed box 28 x 17 x 0.2 cm | a printed protractor round the axis, 60 to 120 amber, 120 to 180 green |

The plates stand 1.5 mm over the card, so nothing drags on it. The leaf's mesh has a neck round the hinge, where the
base leaf's barrels and bridges reach in: a web 1.5 mm thick and 24 mm tall joins the plate to the knuckle (it collides as
the full plate). The hinge does not collide with the base leaf's plate through its joint.

### The layers, the rocking rule and the snap

| Part | Joint | Behaviour |
|---|---|---|
| layer (4) | a generic joint at the axis, its twist driven toward 0 by a torsion spring of 2 N m/rad, 0.05 N m s/rad, capped at 0.2 N m | together they hold the leaf at 0 with 8 N m/rad up to 0.8 N m |
| rocking rule | `on_step` | a push is the leaf turned 0.9 degrees or more to one side. The first push starts it, each later push to the other side is half a cycle, two halves are a cycle, and each cycle switches the next layer off (`unstick`) |
| snap | `on_step` | when the layers still on hold the leaf with 0.49 N m or more (their stiffness times its angle), or it is 8 mm off its hinge, its hinge and layers are switched off (`snap`): it is a free body, for the rest of the trial |
| test push | a generic joint at the axis, off at the start, its twist driven at 30 degrees a second toward closing with at most 0.0325 N m | switched on once (`test_push`) when the leaf has lain at 120 degrees or more, let go and still, for 1 s; off when the test is judged. A free hinge turns at once; a layer still on (0.2 N m) holds against it |

With four layers on, 0.49 N m is 3.5 degrees. With three it is 4.7, with two 7, with one 14, and with none the leaf is free.
The soft arm's push through a fingertip reaches 0.5 N m only when its hand is far ahead of the leaf, so a hard push snaps the
leaf, and a rocking push of about 0.25 N m (1.8 degrees with four layers on) does not.

## How the real hinge is made

The hinge is printed in one piece, the way the scene shows it:

| Setting | Value |
|---|---|
| Material, nozzle, layers | PLA, 0.4 mm nozzle, 0.2 mm layers, 210 C nozzle, 60 C bed, 100% infill |
| Orientation | standing as it lies on the card (hinge axis vertical), no supports |
| Pin and knuckle | a 5 mm pin printed with the base leaf's two barrels; the knuckle (13 mm across, 24 mm tall) has a 5.3 mm bore: 0.15 mm clearance all round, about half what a free print-in-place hinge needs |
| Gaps above and below the knuckle | one 0.2 mm layer: the printer bridges it and the bridge fuses to the knuckle |
| Neck | the leaf's plate joins the knuckle through a web 1.5 mm thick, 24 mm tall, 18.5 mm long. Bent about the axis, PLA (about 50 MPa along its layers) breaks there at about 0.5 N m: 7.5 N at 6.5 cm from the axis |

**Checking a batch.** Print six from one file, printer and spool, and check two with a force gauge (0 to 10 N, peak hold)
at the hole 6.5 cm from the axis. One is pushed round steadily: its web must snap at 6 to 9 N without the hinge turning
first (the fused hinge holds harder than the web). The other is rocked by hand with pushes of at most 4 N (0.26 N m)
each way: it must come free within 10 cycles with its web intact (no crack, no whitening) and then pass the free-movement
test below. If the first one turns before it snaps, or the second does not come free, make the bore 0.05 mm smaller or
larger and print the batch again. The four left are the trial hinges.

**Reset.** A fresh fused print from a checked batch for each trial: rocking frees the hinge for good, and a freed hinge
cannot be fused again. Stand it on the card with the leaf at 0 degrees, along the base leaf.

## Run it

On the YAM Brev launchable, from this folder, with the Arena environment active (`source /data/IsaacLab-Arena/yam.env`):

```bash
# Dream-1 (needs DREAM_API_KEY), 1,200 steps
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=free_printed_hinge.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The orange leaf of the printed hinge is fused solid. Free the hinge so the orange leaf swings round at least 120 degrees, past the amber part of the card, and stays there. Do not snap the leaf off." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/free-printed-hinge-dream

# Scripted demo (right arm), 1,200 steps (it ends on success)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=free_printed_hinge.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The orange leaf of the printed hinge is fused solid. Free the hinge so the orange leaf swings round at least 120 degrees, past the amber part of the card, and stays there. Do not snap the leaf off." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/free-printed-hinge-demo
```

- The run prints `task_score` at the end. With `-E object_jitter=0` the layout is fixed and no seed is needed.
- The demo (right arm only) puts its fingers, open to a 17 mm gap, either side of the leaf's outer end (6.5 cm from the hinge)
  and rocks it five times: the hand goes 3 cm to one side, then 3 cm to the other, each push 1.3 s, so each pad takes up
  the gap and pushes the leaf. A grip closed on the leaf does not work: the soft wrist tilts instead of pushing. After the
  layers have let go (four cycles are needed, so one is in reserve) it lifts its hand, comes round behind the leaf with its
  closed fingertips and sweeps the leaf along its arc to the stop at 130 degrees, then backs off and lifts its hand
  clear. The leaf lies still at its stop for 1 s, the scorer's test push turns it back 30 degrees, and the run ends on
  success. The demo reads nothing from the simulation: it is a fixed plan from the fixed layout. It reaches 1.0 and ends
  on success at step 615 (30.8 s at 20 Hz): the test push started at 130 degrees and passed 1.05 s later at 98.6 degrees. The trials here shared one GPU with several other runs and took far longer, so the time alone on an L40S was not measured.
- Review of the demo's motion: the arms' closest link origins are 32 cm apart (the left arm holds READY). The leaf's peak
  speed is 0.05 m/s and 1.3 rad/s, nothing fell, nothing snapped, and the layers' holding torque peaks at 0.23 N m.
- **Ablations** (the same demo with one constant changed, played in the simulator by a probe because `task-ablate.sh` kept
  finding the shared GPU busy), each scoring 0:
  - `FORCE_S = 1.5`: instead of rocking, the hand goes round to 60 degrees in 1.5 s. The leaf snaps (the layers hold it
    with 0.57 N m, 4 degrees of turn) and scores 0.
  - `FORCE_S = 40`: the same push spread over 40 s. The pad takes the gap up and pushes harder and harder, and the leaf
    snaps at about 0.6 N m (4.5 degrees), no cycle ever counted. It scores 0.
  - `ROCKS = 0`: the demo's sweep on the still-stuck leaf, with no rocking. The fingertip push takes the leaf to 3.6 degrees and
    snaps it, so it scores 0.
- Dream-1 has been run on this version (1,200 steps, the command above): it scored 0.0 and ran to the step limit. Both
  arms lift, draw back near their bases and stay there; neither reaches the hinge, and the leaf stays at 0 degrees.
- Videos: `inspect-robots video ~/yam-eval/free-printed-hinge-demo/<log>.json` writes one MP4 per camera next to the frames.
- Offline checks: `python -m pytest -q test_free_printed_hinge.py`.

## Scoring

All positions are in metres in the station frame, and heights are above the table top.

- **Angle:** the orange leaf's turn about the hinge axis, counterclockwise from above, 0 where it starts (the base leaf's line).
- **Layers:** four at the start, one fewer after each rocking cycle (see above).
- **Snapped:** at any time the layers still on hold the leaf with 0.49 N m or more, or its centre is 8 mm or more off the
  hinge. Recorded in the episode: the trial scores 0 from then on. A reset clears it, and the layers.
- **0** if the state is not finite.
- **Let go:** no point of any gripper finger within 3 mm of the leaf's box (the fingers are points on the robot's finger
  collision meshes, `fingers.py`), read every control step. Nothing counts as let go before the first step.
- **Free-movement test** (once): when the leaf has lain at 120 degrees or more, let go and still (under 10 mm/s and
  0.5 rad/s) for 1 s without a break (timed every control step; any lapse starts the second over), the test push is
  switched on: toward closing, 30 degrees a second, at most 0.0325 N m (0.5 N at 6.5 cm from the axis). It passes when the
  leaf has turned back 30 degrees or more within 3 s, not snapped and with no finger within 3 mm of it all along;
  otherwise it fails, for good. The push is switched off either way.
- **1.0 (success):** the test passed. `SUCCESS_HOLD_S` is 0, so the run ends on that step.
- **0.5:** not snapped, angle 60 degrees or more at the end: a leaf held at 120 degrees or more in a gripper, one still
  moving, one not yet tested, or one whose test failed.
- **0** otherwise, including doing nothing.

In a real trial a protractor on the card gives the angle, and a broken web (the leaf off its knuckle) shows. After the
robot has let go and the leaf has lain still for 1 s, the free-movement test is made by hand with a force gauge (0 to 5 N,
0.05 N resolution, peak hold) at the hole 6.5 cm from the axis, pushed slowly (about 1 s for the 30 degrees) toward
closing: it passes if the leaf turns back 30 degrees on the protractor with the peak at 0.5 N or less.

## Revision notes

The first review asked for the following; each is answered here.

- "README says Dream-1 has not been run, while run_scores and the supplied video report a run." The README now reports
  the Dream-1 run and its 0.0 (Run it).
- "Reaching 120 degrees while still gripping the leaf does not establish that a stuck hinge has been freed" and "Add a
  measurable test of movement after freeing so that merely holding a resistant leaf at 120 degrees cannot score full
  credit." Full credit now needs the free-movement test: the leaf let go and still at 120 degrees or more for 1 s, then
  turned back 30 degrees by a push of at most 0.0325 N m (0.5 N at 6.5 cm), in the simulation (`tick`, `test_push`) and
  in a real trial (a force gauge). A held leaf scores 0.5 (`test_held_at_120_never_scores_full`).
- "The real-world score ... does not check whether the joint now moves freely." The real-world test is the same push
  with a force gauge and the protractor (Scoring).
- "Specify the joint and print settings needed to produce a consistently fused hinge, and validate that it can be freed
  by rocking without breaking." How the real hinge is made: geometry, clearances, material, layers, orientation, the
  neck web that breaks at the stated torque, and a batch check that confirms each batch is fused, snaps when forced and
  comes free when rocked. The meshes now show that hinge: the pin is printed inside the barrels, the gaps are one layer,
  and the neck is the 1.5 mm web.
- "Fabrication or repair under Other is a better fit than impact on jobs." The category is now Other (fabrication and
  repair).
