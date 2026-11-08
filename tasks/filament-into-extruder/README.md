# Feed filament into the feeder on top of a 3D printer

Loading an enclosed 3D printer by hand, the way a fab lab technician does it: the filament comes off the spool that
hangs on the printer's side, and its cut end is pushed into the feeder on the printer's roof, a funnel that narrows to
a bore barely wider than the filament (1.75 mm filament, 2.4 mm bore). Filament is springy: held a few centimetres
back, its end droops and wanders, and pushed off its axis it bends and kinks against the feeder. So both hands keep
the filament straight while they lift it over the roof and feed it in, and the spool turns as the filament is drawn
off it. Here the robot pushes the end 2 cm or more into the inlet and lets go, without kinking the filament or
pulling it off the spool.

The printer is the one of the `scrape-print-off-bed` and `open-printer-door` tasks, here with its door shut, a spool
holder on its left wall and a feeder on its roof. The `spool-onto-printer-holder` task uses the same printer, holder
and spool.

| File | What it is |
|---|---|
| `filament_into_extruder.py` | The `yam_arena` task file: scene, scoring (`success`, `score`) and the scripted `demo()` |
| `printer.py` | The printer's geometry and colliders, with the spool holder and the feeder |
| `filament.py` | The flexible filament: `Link` boxes laid end to end along a path, D6 joints with sprung, damped rotations between them (their rest is the laid shape), the anchor joint to the spool, and `Spool`, a box that turns on a damped revolute joint and collides as a winding and two flanges |
| `inlet.py` | `Head`: a fixed box (the feeder) whose collider is replaced by box colliders around a funnel and a bore |
| `colliders.py` | A box that collides as several smaller boxes (the printer, its door, the holder's bracket) |
| `grip.py` | `HandWaypoint`: demo waypoints with the gripper pointing any way (the forward grips over the roof) |
| `bonds.py` | The `on_object`, `on_spawn`, `on_step` and `on_reset` hooks |
| `fingers.py` | Points on the fingers' collision meshes, which the tests use to keep the hands clear of the printer |
| `looks.py` | Puts a detailed mesh on a box at spawn and hides the plain box, which still collides |
| `looks/` | The meshes (`*.usdc`) and their textures, and `make_looks.py`, the Blender script that models and exports them |
| `test_filament_into_extruder.py` | Offline checks: the file loads, the inlet's colliders, the chain and its joints, the anchor, the layout, every scoring outcome, the demo's reach and its clearance from the printer |
| `task.json` | The task's details for the board |

## The scene

Positions are centres in metres from the midpoint between the arm bases (x forward, y to the robot's left), heights
above the table top.

| Part | Collider | Physics |
|---|---|---|
| printer | 36 x 35 x 40 cm, x 0.31 to 0.67, centred on y = 0, fixed | box colliders: base, bed, walls, roof, the front frame and the gantry. The door is shut and fixed |
| spool holder | a bracket plate 10 x 14 cm on the left wall at x = 0.48, fixed | the bracket collides. The 45 mm rod (22 cm up, along y, from a 7 cm stop collar 3 cm long out to y = 0.31) is drawn: the spool turns on a joint about its axis |
| spool | a box 7 x 20 x 20 cm at (0.48, 0.248), 22 cm up, its axle along y, 0.5 kg | turns on a revolute joint about the rod's axis, damped 0.01 N m s/rad, comes off when pulled harder than 60 N. Collides as a 16 cm winding and two 20 cm flanges |
| feeder | 10 x 4.5 x 12 cm on the roof at the back right: x 0.495 to 0.595, y -0.175 to -0.13, 40 to 52 cm up, fixed | box colliders leave a square funnel 6 mm wide at the mouth, narrowing over 5 mm to a 2.4 mm square bore, 3 cm deep and closed. Its walls are slick (friction 0.15). The mouth's centre is at (0.545, -0.13), 47.5 cm up, facing +y |
| two 1-2-3 blocks | 2.54 x 5.08 x 7.62 cm, lying flat, long side along y, at (0.25, -0.102) and (0.25, 0.045), fixed | the filament's end lies across them |
| filament | 27 links, each a 2.5 x 0.175 x 0.175 cm box drawn as a round-ended tube, 1 g, free | 67.5 cm from its free end to the spool. The end at (0.25, -0.12) on the first block, pointing right. Its last 31 cm lie straight along y across both blocks, then it drops to the table, curls left (6 cm radius) to run along x at y = 0.248 beside the printer's left wall, bends up (6.5 cm radius) under the spool and rises to the front of its winding |

The joints: between each two links a D6 joint locks the three translations and springs the three rotations back to
the laid shape (0.064 N m/rad, the bending stiffness of 1.75 mm PLA over 2.5 cm, damped 0.002 N m s/rad). The last
link is held by the same kind of joint to the front of the spool's winding, and tears off when pulled harder than
25 N. Each link has its inertia raised to 1e-6 kg m^2, continuous collision detection and 32 solver position
iterations.

The SimReady collection's two 3D printers are 1 m industrial machines with no spool holder or feeder, so every part is
modelled in Blender (`looks/make_looks.py`, the printer's script with the holder, feeder, spool, filament and blocks
added) and drawn on its box: the printer (light grey base and panels, black frame, glass roof, the bed and gantry
inside, a small screen), the door, the print head and the build plate, the holder (black bracket and collar, brushed
steel rod), the feeder (aluminium riser, black stepper motor, black feeder body with a brass inlet funnel and an open
bore, a steel idler lever with its tension screw, a white PTFE tube into the roof), the spool (charcoal flanges with
six windows and a blank label, orange winding, grey hub), the filament links and the steel blocks. Every mesh fits
inside its box (`make_looks.py` checks it when it exports). To change them, edit `looks/make_looks.py` and run
`blender -b -P looks/make_looks.py` (Blender 4.2 or later) from this folder.

## Run it

On the YAM Brev launchable, from this folder, with the Arena environment active
(`source /data/IsaacLab-Arena/yam.env`):

```bash
# Dream-1 (needs DREAM_API_KEY), 1,200 steps
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=filament_into_extruder.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Feed the free end of the orange filament into the brass inlet of the black feeder on top of the printer, at its back right, until at least 2 cm of it is inside, then let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/filament-into-extruder-dream

# Scripted demo (both arms), 600 steps
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=filament_into_extruder.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Feed the free end of the orange filament into the brass inlet of the black feeder on top of the printer, at its back right, until at least 2 cm of it is inside, then let go." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/filament-into-extruder-demo
```

- The run prints `task_score` at the end. With `-E object_jitter=0` the layout is fixed and no seed is needed. Keep
  it at 0: jitter moves each link on its own and would pull the filament apart at its joints.
- The demo: each hand pinches the filament from above, straight down, the right hand 8.75 cm from the end (between
  the two blocks, where the filament lies level) and the left hand 17.5 cm further back (beyond them), each in its own
  half of the table. They lift it and, rising in front of the printer, turn their grips forward about the filament's
  own line until the fingers point along +x, one pad above the filament and one below (a straight-down grip can't
  reach 47 cm above the table). They carry it level over the roof, its end 1 cm short of the mouth, and push it 3.5 cm
  along -y together. The left hand lets go and backs out over the printer's front, then the right hand. The rest of
  the filament trails over the roof's left edge to the spool, which turns about 120 degrees as it is drawn off. The
  end rests 2.4 cm in, on the inlet's axis, and the demo ends on success at step 451 (22.6 s at 20 Hz). A trial
  takes about 5 minutes on an L40S with the window open (`-E headless=false`), including startup.
- The hands carry the line 4.1 mm back (-x) and 5.4 mm up from the inlet's axis (`AIM`): the end, 8.75 cm ahead of
  the right hand, comes out of the grip's turn that far forward and low (measured). Without it the end jams on the
  feeder's face and kinks.
- Review of the demo: the arms' links stay 11.8 cm apart or more, nothing falls, the filament's links peak at
  0.53 m/s while carried. The gripped links spin up to 13 rad/s for a moment while the grips turn (in the pads).
- The shortcut of never letting go scores 0.5: the same demo with `HOLD_ON = 1` pushes the end in and keeps both
  hands closed on it (`scripts/task-ablate.sh filament-into-extruder holdon HOLD_ON=1`: 0.5).
- Videos: `inspect-robots video ~/yam-eval/filament-into-extruder-demo/<log>.json` writes one MP4 per camera next to
  the frames.
- Offline checks: `python -m pytest -q test_filament_into_extruder.py`.

## Scoring

Positions are centres, in metres, in the station frame. Each link's two ends are worked out from its centre and its
turn. The free end is the front end of the first link.

- **Depth:** how far the free end is past the funnel's mouth (the feeder's face), along the inlet's axis (-y).
- **In the inlet:** the free end within 2.5 mm of the inlet's axis. The feeder is solid around the funnel and the
  bore, so an end that deep and that close to the axis is inside them.
- **Held:** a gripper is closed on the filament: its finger pads 1.25 to 4.75 mm apart and the point between them
  within 1.5 cm of a link's axis. The scorer reads both grippers from the robot every step.
- **Still:** no link's centre, and not the filament's point on the spool, moved 2 mm or more over the last second
  (read every step). The end rattles in the 2.4 mm bore without going anywhere, so this is a distance, not a speed.
- **Kinked:** two neighbouring links 60 degrees or more out of line (a bend radius of 2.4 cm, where PLA creases), now
  or at any step earlier in the trial. As laid, the sharpest bend is 25 degrees.
- **Torn off:** the filament's last link more than 5 mm from its point on the spool's winding (as the spool now
  stands). Its joint breaks above 25 N.
- **Off its rod:** the spool's centre more than 1 cm from where it started. Its axle breaks above 60 N.
- **1.0 (success, held 1 s):** the free end 2 cm or more deep and in the inlet, no gripper holding the filament, and
  the filament still.
- **0.5:** the free end 5 mm or more deep and in the inlet, but not all of the above.
- **0.25:** the free end within 1 cm of the mouth's centre.
- **0** otherwise, and **0** whatever else if the filament kinked, tore off the spool or fell off the table (a link's
  centre more than 2 cm below the table top), if the spool came off its rod, or if the state is not finite.

In a real trial, a mark on the filament 2 cm from its end shows the depth, a crease shows a kink, and the video shows
the hands let go.

## Layout and reach

The printer stands where it does in the scraping task (x 0.31 to 0.67). Its roof is 40 cm up, out of reach of a
straight-down grip, so the demo turns its grips forward, and the feeder's inlet sits 7.5 cm above the roof so the
grippers clear it. The inlet is at x = 0.545, behind the roof's middle (0.525) on the right: further back, the right
arm's IK fails by several millimetres at that height (`ArmPlan` worst over the demo: 1.0 mm and 0.1 degrees for the
right arm, 0.1 mm for the left). Its mouth faces left (+y), toward the spool's side, so the filament is pushed along
-y with the right hand near its end and the left hand further back, each in its own half.

## Fidelity

- The filament is 27 rigid links 2.5 cm long on sprung joints with PLA's bending stiffness and a little damping, and
  no hard limits. It bends only at the joints, every 2.5 cm, and never creases: a kink is the scored 60 degree bend.
- Each link weighs 1 g (40 g/m, 13 times real PLA), with its inertia raised to 1e-6 kg m^2, continuous collision
  detection and 32 solver iterations, for stable contacts with the 50 N grippers. Each collides as a 1.75 mm square
  box and is drawn as a round tube.
- The filament's end on the spool is a joint to a point on the winding: as the spool turns, that point moves with it,
  so a long pull would wrap the filament round the winding instead of paying it out.
- The spool turns on a revolute joint damped 0.01 N m s/rad. The funnel and the bore are box colliders at their true
  sizes (6 mm mouth, 2.4 mm bore), slick like a PTFE liner (friction 0.15), square where real ones are round (the
  meshes draw them round), and the bore ends closed after 3 cm: there are no drive gears.
- The printer, its door, the holder, the feeder and the blocks are fixed.
