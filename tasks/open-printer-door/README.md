# Open a light 3D printer's door past its magnetic catch, steadying the printer

Opening an enclosed desktop 3D printer's front door, the way a makerspace or print farm technician does before every
print removal or filament change. Two things make it more than a door opening:

- **The catch and the light printer.** A magnetic catch holds the door shut until about 10 N is pulled at its handle,
  and the printer (2.5 kg, on hard plastic feet) slides on the table at about 7 N. Pulled with one hand, the printer comes
  across the table instead of the door coming open. The other hand has to steady it: pressing on the ledge in front
  of the door adds friction and holds it where it is.
- **Fully open, without a knock.** The door must go to 90 degrees or more, where a detent takes it into its open stop
  (95 degrees) and holds it, without anything striking the print head parked at the front of its gantry or the
  fragile finished print standing on the plate just behind the door.

| File | What it is |
|---|---|
| `open_printer_door.py` | The `yam_arena` task file: scene, scoring (`success`, `score`), the step watcher and the scripted `demo()` |
| `printer.py` | The printer's geometry and its box colliders (base, bed, walls, roof, front frame, gantry, door) |
| `bonds.py` | `Hinged` (a revolute joint with a breakable catch and spring zones), `Sliding` (a sprung slide), and the step and reset hooks |
| `colliders.py` | Makes a task-file box collide as several smaller boxes, each with its own friction (the printer, its feet, the door) |
| `wrist.py` | `TiltWaypoint`: demo waypoints with the grip tilted about the fingers' axis |
| `looks.py` | Puts detailed meshes on a box at spawn and hides the plain box, which still collides |
| `looks/` | The meshes (`*.usdc`) and their textures, and `make_looks.py`, the Blender script that models and exports them |
| `test_open_printer_door.py` | Offline checks: the catch, feet, hinge and detent, the door angle, every scoring outcome, the demo's order, reach and clearance from the printer, the meshes in their boxes (no GPU) |
| `task.json` | The task's details for the board |

## The printer, the door and its hinge

The printer is the same model as in the `scrape-print-off-bed` task (made by `looks/make_looks.py`), standing 9 cm
further back so its 33 cm door swings open clear of the arms: 36 x 35 x 40 cm, centred at x 0.58 on the table's
midline, its front toward the robot. It is one free rigid body of 2.5 kg that collides as its base, bed, walls, roof,
front frame and gantry, on four feet whose friction against the table is 0.28 (the lower of the two wins).

The door is a free box on a revolute joint (`bonds.Hinged`) at its left edge, on the printer:

| Part | Joint | Behaviour |
|---|---|---|
| hinge | revolute, 0 to 110 degrees, damping 0.02 N m s/rad | swings freely with a little friction |
| catch | a second joint at the hinge that locks only the door's turn, `physics:breakTorque` 3.1 N m | holds the door shut until 10 N is pulled at the handle (31 cm from the hinge), then lets go for the rest of the trial |
| detent | a spring torque, 1.5 N m/rad toward 95 degrees with 0.1 N m s/rad of damping, from 80 degrees open | pulls the door into its open stop and holds it there |

The detent is applied as a torque on the door, and the opposite one on the printer, every control step, because
PhysX does not take a joint drive's gains changed during a run. The print head hangs on a sprung slide along x
(`bonds.Sliding`, 300 N/m) on the printer, and the vase stands loose on the plate, so a knock moves them. The door's
acrylic has a friction of 0.1, its handle the usual grip.

## Run it

On the YAM Brev launchable, from this folder, with the Arena environment active
(`source /data/IsaacLab-Arena/yam.env`):

```bash
# Dream-1 (needs DREAM_API_KEY), 1,200 steps
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=open_printer_door.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Open the clear front door of the 3D printer all the way, until it stays open by itself. Hold the printer still with your other hand while you pull, and do not touch the print head or the orange print inside." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/open-printer-door-dream

# Scripted demo (both arms), 600 steps
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=open_printer_door.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Open the clear front door of the 3D printer all the way, until it stays open by itself. Hold the printer still with your other hand while you pull, and do not touch the print head or the orange print inside." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/open-printer-door-demo
```

- The run prints `task_score` at the end. With `-E object_jitter=0` the layout is fixed and no seed is needed.
- The demo: the left arm presses its closed fingertips down on the ledge in front of the door, just left of the
  middle (3 cm down into it, so it leans on it), its grip tilted 30 degrees forward. The right arm grips the handle
  from the front and pulls it along the door's arc to 6 degrees: the catch lets go. The left hand then backs off the
  ledge and parks out of the door's way, before the door's bottom edge sweeps over where it pressed. The right hand
  lets go of the handle, comes round the right of the printer, its wrist turned 65 degrees, puts its closed fingertips behind the door's free edge
  and, with the side of its fingers on the door's inside face, sweeps the door open at a constant 30 cm from the hinge
  to about 85 degrees, where the detent takes it to 95. The right hand never holds the free door: a held handle fights
  the hand's path and slides the light printer. It reaches 1.0 and ends on success at step 467 (23 s at 20 Hz). A
  trial takes about 2 minutes on an L40S, including startup.
- Dream-1 has not been run yet. 1,200 steps take about 4 minutes, including startup.
- The same demo with the left hand hovering 3 cm over the ledge instead of pressing it (`STEADY_DEPTH = -0.03`)
  scores 0: the right hand's pull drags the printer across the table before the catch gives, and the door stays shut.
- Videos: `inspect-robots video ~/yam-eval/open-printer-door-demo/<log>.json` writes one MP4 per camera next to the
  frames.
- Offline checks: `python -m pytest -q test_open_printer_door.py`.

## Scoring

All positions are in metres in the station frame (x forward, y to the robot's left, from the midpoint of the arm
bases), and heights are above the table top.

- **Door angle:** the door's turn about its hinge relative to the printer, 0 shut.
- **Printer moved:** a corner of the printer's base 1 cm or more from where it started (a slide or a turn). Recorded:
  the trial scores 0 from then on.
- **Struck:** the print head 3 mm or more off its parked place on its slide, or the vase 5 mm or more from its place on
  the plate or leaning 10 degrees or more, both measured relative to the printer. Recorded: the trial scores 0 from
  then on.
- **0** if the state is not finite.
- **1.0 (success, held 1 s):** the door open 90 degrees or more and still (under 0.1 rad/s).
- **0.5:** the door open 45 degrees or more.
- **0.25:** the door open 10 degrees or more (past the catch).
- **0** otherwise, including doing nothing.

In a real trial, a protractor on the hinge gives the door's angle, tape on the table marks the printer's feet, and
the video shows any knock on the head or the print.
