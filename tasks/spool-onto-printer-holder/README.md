# Hang a filament spool on the 3D printer's side holder

Changing the spool on an enclosed 3D printer, a job fab lab and print farm technicians do all day: a new 1 kg spool,
set down standing on its rims at the front right of the bench, goes onto the spool holder on the printer's left wall,
a smooth rod only 7 mm narrower than the spool's bore. The spool starts in the right arm's half and the holder is out
of the right arm's reach, so the spool is passed from one hand to the other, then lifted level with the rod, threaded
on square to the wall and pushed back to the stop collar.

The printer is the one of the `scrape-print-off-bed` and `open-printer-door` tasks, with its door shut, the spool
holder on its left wall and the feeder on its roof of the `filament-into-extruder` task (same printer, holder and spool
meshes).

**Novelty.** Our own `spool-onto-holder` (PR #768) lifts a spool lying flat in front of the robot with both hands at
once, stands it up and slides it toward the robot's front, onto a free-standing rod that points at the robot. Here the
rod sticks out sideways from the printer's wall, the spool starts standing in the other arm's half, the arms pass it
from one hand to the other, and one hand alone carries the 1 kg spool round the printer's front corner, lines its bore
up with the rod and pushes it back to the wall. The scoring follows that: a rung for the hand-over, the rod along y,
the stop at the wall. #838 (another team) loads a spool into a filament dryer, a different motion.

| File | What it is |
|---|---|
| `spool_onto_printer_holder.py` | The `yam_arena` task file: scene, scoring (`success`, `score`) and the scripted `demo()` |
| `printer.py` | The printer's geometry and colliders, with the spool holder and the feeder |
| `spool.py` | `Spool`: one rigid body of box colliders round an open bore, and `rod`: the holder's rod and collar as cylinder colliders along y, the rod's nose a convex cone (adapted from `spool-onto-holder`) |
| `colliders.py` | A box that collides as several smaller boxes (the printer, its door, the holder's bracket) |
| `grip.py` | `HandWaypoint`: demo waypoints with the gripper pointing any way |
| `bonds.py` | The `on_spawn`, `on_step` and `on_reset` hooks |
| `fingers.py` | Points on the fingers' collision meshes: the scorer's let-go check, and the tests' clearance checks |
| `looks.py` | Puts a detailed mesh on a box at spawn and hides the plain box, which still collides |
| `looks/` | The meshes (`*.usdc`) and their textures, and `make_looks.py`, the Blender script that models and exports them |
| `test_spool_onto_printer_holder.py` | Offline checks: the file loads, the bore is open round the rod, the holder's place, the looks, every scoring outcome and the let-go check, the demo's reach, the hand-over, the spool's clearance from the printer and the hands' clearance (no GPU) |
| `task.json` | The task's details for the board |

## The scene

Positions are centres in metres from the midpoint between the arm bases (x forward, y to the robot's left), heights
above the table top.

| Part | Collider | Physics |
|---|---|---|
| printer | 36 x 35 x 40 cm, x 0.31 to 0.67, centred on y = 0, fixed | box colliders: base, bed, walls, roof, the front frame and the gantry. The door is shut and fixed. The feeder on the roof (x 0.495 to 0.595, y -0.175 to -0.13, 40 to 52 cm up) is a fixed box |
| holder | a bracket plate 10 x 14 cm on the left wall at x = 0.48, fixed | the bracket, a stop collar (cylinder, 7 cm across, y 0.181 to 0.211) and the rod (cylinder, 45 mm, its axis 22 cm up, from the collar to y = 0.31, its last 2 cm a convex cone narrowing to 28 mm). The rod's friction is 0.3 |
| spool | 7 x 20 x 20 cm, its axis along y, standing on its rims at (0.20, -0.17), 1 kg, free | 72 box colliders: a 24-sided ring round a 52 mm bore (the hub and the wound filament as one core, 16 cm across) and two 6 mm flanges out to the 20 cm rim. It stands on a flat of its rims |

The SimReady collection's two 3D printers are 1 m industrial machines with no spool holder, so every part is modelled
in Blender (`looks/make_looks.py`, the printer's script with the holder, the feeder and the spool added) and drawn on
its box: the printer, its door, print head and build plate, the holder (black bracket and collar, brushed steel rod),
the feeder (black body with a brass inlet on a riser, a stepper motor), and the spool (charcoal flanges with six
windows and a blank label, orange winding, grey hub round the bore). Every mesh fits inside its box (`make_looks.py`
checks it when it exports). To change them, edit `looks/make_looks.py` and run `blender -b -P looks/make_looks.py`
(Blender 4.2 or later) from this folder.

## Run it

On the YAM Brev launchable, from this folder, with the Arena environment active
(`source /data/IsaacLab-Arena/yam.env`):

```bash
# Dream-1 (needs DREAM_API_KEY), 1,200 steps
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=spool_onto_printer_holder.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hang the orange filament spool on the spool holder on the printer's left side: slide its hole over the rod until it touches the stop, then let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/spool-onto-printer-holder-dream

# Scripted demo (both arms), 600 steps
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=spool_onto_printer_holder.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hang the orange filament spool on the spool holder on the printer's left side: slide its hole over the rod until it touches the stop, then let go." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/spool-onto-printer-holder-demo
```

- The run prints `task_score` at the end. With `-E object_jitter=0` the layout is fixed and no seed is needed.
- The demo: each hand takes the spool by the top of its rim, its fingers closing along y on the outsides of both
  flanges (the full 50 N grip), tilted forward (the right hand 30 degrees, the left 70). The right hand lifts the spool,
  carries it to the middle of the table (0.20, -0.01), sets it down on its rims, lets go and backs away. The left hand,
  waiting out of its way, then takes it there, lifts it to the rod's height, carries it left in front of the printer
  and forward along its side to 1.2 cm beyond the rod's tip, lines its bore up with the rod, slides it on until it
  leans on the stop collar, lets go and draws back up and out. The spool ends on the rod, its inner face 1 mm from the
  collar, and the demo ends on success at step 553 (27.7 s at 20 Hz). A trial takes about 5 minutes on an L40S with
  the window open (`-E headless=false`), including startup.
- The left arm, reaching out to the rod with the 1 kg spool, gives under its weight: the demo sends it 7.5 mm forward,
  2.6 cm further left and 10 cm higher than where the spool should be (`LEFT_HOLD`, measured), and the right hand
  1.2 cm higher (`SAG`).
- Review of the demo: the arms' links stay 18.4 cm apart or more, the spool peaks at 0.33 m/s and 1.8 rad/s while
  carried, nothing falls.
- The shortcut of never letting go scores 0.5: the same demo with `HOLD_ON = 1` keeps the left hand on the seated
  spool (`scripts/task-ablate.sh spool-onto-printer-holder holdon HOLD_ON=1`: 0.5).
- Videos: `inspect-robots video ~/yam-eval/spool-onto-printer-holder-demo/<log>.json` writes one MP4 per camera next
  to the frames.
- Offline checks: `python -m pytest -q test_spool_onto_printer_holder.py`.

## Scoring

Positions are centres, in metres, in the station frame. The spool's axis is read from its turn.

- **On the rod:** the spool's centre within 2.6 cm (the bore's radius) of the rod's axis, its axis within 10 degrees
  of the rod's, and both its faces between the stop collar's face (y = 0.211, 5 mm of slack) and the rod's tip
  (y = 0.31).
- **Seated:** on the rod, its inner face within 1 cm of the collar's face.
- **Let go:** no point of any gripper finger within 5 mm of the spool's outside (a cylinder 20 cm across, 7 cm wide),
  read from the simulation every step.
- **1.0 (success, held 1 s):** seated, let go, and still (under 1 cm/s and 0.5 rad/s).
- **0.5:** on the rod, but not seated, or still held, or still moving.
- **0.25:** the spool's centre within 5 cm of the centre of the rod's tip (lined up at the rod).
- **0.1:** the spool's centre 15 cm or more left (+y) of where it stood, standing on its rims or held (its axis within
  30 degrees of level): passed over to the middle.
- **0** otherwise (where it stands, lying flat), and **0** if it fell off the table (its centre more than 2 cm below
  the table top) or the state is not finite.

In a real trial, a tape measure from the collar to the spool's inner face and the video showing the hand let go are
enough.

## Layout and reach

The holder's rod is at x = 0.48, 22 cm up: the left arm reaches it with a grip tilted 70 degrees forward, the right
arm can't (its IK misses the rod's line by more than 2 cm with any of the demo's grips). The rod's 3 cm stop collar
keeps the spool 3.8 cm off the wall, so a finger fits between them. The spool stands 1 cm in front of the printer's
base at (0.20, -0.17), in the right arm's half, and is passed over at (0.20, -0.01). `ArmPlan` worst over the demo:
0.1 mm and 0.01 degrees for both arms.

## Fidelity

- The spool is one rigid 1 kg body of 72 box colliders: a 24-sided bore (flats 26 mm from the axis), the hub and the
  wound filament as one solid core, two 6 mm flanges. The filament doesn't unwind. The meshes are drawn only.
- The rod (friction 0.3) and the collar are cylinder colliders, the rod's nose a convex cone. The printer, its door,
  the feeder and the holder are fixed.
- The gripper squeezes up to 50 N per pad. The arms are position controlled and give under the spool's weight, which
  the demo allows for.
