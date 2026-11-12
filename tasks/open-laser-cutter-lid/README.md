# Open a heavy laser cutter lid with both hands until its gas struts hold it

The first step of every job on a desktop CO2 laser cutter, in a fab lab or a sign shop. The lid is a steel hood with
a glass window, hinged at the back, and its two gas struts only help once it is open past 30 degrees. Below that the
whole 6 kg hangs on the handle: it takes both hands, one at each end of the handle. Let go too early and it slams
shut on the machine's glass and optics, so a trial in which the lid falls back scores 0.

| File | What it is |
|---|---|
| `open_laser_cutter_lid.py` | The `yam_arena` task file: scene, struts, scoring (`success`, `score`) and the scripted `demo()` |
| `cutter.py` | The laser cutter's sizes, collider boxes and lid geometry, shared with the Blender script (the same file is in the laser-scrap-sheet-lift task) |
| `struts.py` | The real gas-strut and lobe pair that gives the simulation's torque: its dimensions, force curve, the lobe's profile solved from the torque, and the checks on it (no simulation) |
| `bonds.py` | `Hinged`: a box on a revolute joint with limits, and hooks into `yam_arena` (`on_spawn`, `on_reset`, `on_step`) |
| `plate.py` | `collider()`: gives a box a collider made of box pieces (the lid's panel, flap and handle). `surface()`: another friction for a box |
| `wrist.py` | `PinchWaypoint`: a grip that points forward and tilts, its jaws closing up and down, for the demo |
| `looks.py` | Puts a detailed mesh on a box at spawn and hides the plain box, which keeps its bounds |
| `looks/` | The meshes (`*.usdc`), their textures, `make_looks.py` (the Blender script) and `cutter_looks.py` (the machine's modelling, shared with the laser-scrap-sheet-lift task) |
| `test_open_laser_cutter_lid.py` | Offline checks: the file loads, the lid's pieces, the struts' balance, the real strut pair's torque, travel and damping, the lid angle, every scoring outcome with the trial's memory, the demo's reach and clearances (no GPU) |
| `task.json` | The task's details for the board |

## The machine and the lid

The laser cutter is 52 cm wide, 40 cm deep and 22 cm tall, with no brand: a charcoal body whose front wall is 7 cm
tall, side walls with vent slots, a back block the full height (the laser tube, the control panel with a screen,
buttons and an emergency stop, and a round exhaust duct out of the back), an aluminium honeycomb bed inside, and the
laser head on its gantry parked at the back. Every fixed part is a box collider, so an arm cannot pass through it.

The lid is a hood: a top panel with a tinted window and a 15 cm front flap that closes the front above the low wall,
36 cm from the handle to the hinge. A black handle bar, 16 mm square, runs along the flap's lower edge, from 21 cm
right to 21 cm left of the centre line, 18 mm out from the flap on two posts. The lid is one rigid body of 6 kg
(`LID_MASS`) on a revolute joint across the top of the back block, from shut (0) to its open stop at 80 degrees. Its
collider is five boxes: panel, flap, bar and posts. Its weight pulls it shut with 12.7 N m at 0 degrees, 35 N at the
handle.

The struts are a torque about the hinge, set every control step from the lid's angle and speed, and the next section
gives it as a table and as a real pair of gas struts you could build.

Why 6 kg: a probe (the demo's motion with one arm and with two, for several masses) measured that one arm lifts a 2
or 4 kg lid past the struts' balance by itself, but stalls at 12 degrees with a 6 kg lid, which then slips out of the
gripper and slams shut. Both arms lift the 6 kg lid to 44 degrees, lagging the plan by about 5 degrees.

## The struts

**What the simulation applies.** The struts are not bodies in the scene. Every control step (20 Hz, held for the 12
physics steps of 240 Hz) the task file puts one torque about the hinge axis on the lid, opening, from its angle
`a` (degrees) and angular speed `w` (rad/s), `strut_command` in `open_laser_cutter_lid.py`:

- `a` at 30 or below: 0.
- `a` above 30: `T(a) - 6 w`, in N m. `T` rises in a straight line from 0 at 30 degrees to 14.27 N m at 38 degrees and
  stays 14.27 N m up to the stop. 14.27 N m is 1.2 times the lid's weight torque at 30 degrees (11.89 N m). The 6 is
  the struts' damping, 6 N m s per rad (`STRUT_DAMPING` 1.0 per kg of lid, times 6 kg), only while they push.
- The hinge itself adds a viscous damping of 0.3 N m s per rad (`HINGE_DAMPING` 0.05 per kg of lid, times 6 kg), at
  every angle (the joint's drive). There is no spring and no dry friction in it. The stop at 80 degrees is a hard joint limit.

**The lid's weight.** 6 kg, from five box pieces (panel, flap, bar, two posts) of equal density. Its centre of mass is
215.7 mm in front of the hinge axis and 30.5 mm below it with the lid shut, so its weight torque is
`6 x 9.81 x (0.2157 cos a + 0.0305 sin a)` (`weight_torque`): 12.7 N m shut, 11.9 N m at 30
degrees, 4.0 N m at 80. The moment of inertia about the hinge is 0.36 kg m2 (computed from the pieces, and assumed the
same in the simulation, not read back from it).

| Lid angle (deg) | Weight (N m) | Struts (N m) | Net (N m) | Handle force to hold it (N) | Roller radius (mm) | Rod out (mm) | Force per strut (N) |
|---|---|---|---|---|---|---|---|
| 0 | 12.7 | 0.0 | -12.7 | +35 | 85.0 | 3.0 | 319 |
| 10 | 12.8 | 0.0 | -12.8 | +35 | 85.0 | 3.0 | 319 |
| 20 | 12.5 | 0.0 | -12.5 | +34 | 85.0 | 3.0 | 319 |
| 30 | 11.9 | 0.0 | -11.9 | +33 | 85.0 | 3.0 | 319 |
| 34 | 11.5 | 7.1 | -4.4 | +12 | 84.6 | 3.4 | 319 |
| 36 | 11.3 | 10.7 | -0.6 | +2 | 84.1 | 3.9 | 318 |
| 38 | 11.1 | 14.3 | +3.2 | -9 | 83.4 | 4.6 | 316 |
| 40 | 10.9 | 14.3 | +3.4 | -9 | 82.6 | 5.4 | 315 |
| 50 | 9.5 | 14.3 | +4.7 | -13 | 78.6 | 9.4 | 307 |
| 60 | 7.9 | 14.3 | +6.4 | -17 | 74.5 | 13.5 | 300 |
| 70 | 6.0 | 14.3 | +8.2 | -23 | 70.3 | 17.7 | 292 |
| 80 | 4.0 | 14.3 | +10.3 | -28 | 66.0 | 22.0 | 284 |

Net is positive when the lid opens by itself. The handle force is the push perpendicular to the line from the hinge to
the handle bar's centre (0.365 m long) that holds the lid still: `(weight - struts) / 0.365`. Plus lifts, minus holds
it back. The first columns are `weight_torque` and `strut_torque` in the task file. The rest are `struts.py`, below.

**An equivalent pair of struts.** A plain gas strut cannot give nothing below 30 degrees and a flat 14.3 N m above,
because its torque is its force times a lever arm that comes from where it is mounted, and that lever never stays at
zero for 30 degrees. So the equivalent is two gas struts that push a roller on a lobe fixed to the lid. The roller
sits on a line through the hinge axis, and the lobe is a circle round the hinge pin while the lid is below 30 degrees,
so the push goes straight through the hinge and gives no torque. From 30 degrees the lobe's radius falls, and the
torque is the strut's force times how fast the radius falls per radian of the lid's turn (the lever arm). The lobe is
cut so that this lever, times two struts and the strut's own force curve, is the table's torque. `struts.py` solves
the profile and the tests read the torque back from it.

| Part | Value |
|---|---|
| Struts | 2, one at each side of the exhaust duct, on the machine's lines `v = +0.200 m` and `v = -0.200 m` |
| Nominal force F1 | 250 N, fully extended (a 25 kgf strut) |
| Force fully compressed | 325 N (progression F2 / F1 = 1.3), rising in a straight line with the compression, so 319 N shut and 284 N at the stop |
| Stroke | 40 mm. Rod out 3 mm with the lid shut, 22 mm at the stop (18 mm to spare) |
| Length, body-end pivot to roller axle | 90 mm compressed, 130 mm extended |
| Roller | 20 mm diameter, on the rod end. The lobe's surface is 10 mm inside the axle's path |
| Axis | A straight line through the hinge axis, leaning 15 degrees from straight down toward the back. Hinge axis at `u = 0.330 m`, `w = 0.220 m` (machine frame: `u` forward from the front face, `w` up from the table) |
| Body-end pivot (fixed, on the back block's inside) | 178 mm from the hinge axis along that line: `u = 0.376 m`, `w = 0.048 m` |
| Roller axle | At 85.0 mm from the hinge axis with the lid shut (`u = 0.352 m`, `w = 0.138 m`), 66.0 mm at the stop (`u = 0.347 m`, `w = 0.156 m`) |
| Lobe | A sector of a plate on the lid on each side, centred on the hinge pin, covering 80 degrees. First 30 degrees: pitch radius 85.0 mm (surface 75.0 mm). Then the radius falls to 66.0 mm (surface 56.0 mm) at 80 degrees, by the table's roller radius column. At most 62.6 mm behind the hinge axis, so it stays inside the back block's 70 mm |
| Pressure angle | 7 degrees at 34, 15 at 40, 18 at 60 and 21 at 80 (the roller meets the lobe at a small slant) |
| Hinge load | The struts' push goes through the hinge pin below 30 degrees: 2 x 319 N = 0.64 kN |
| Open stop | A rubber bumper (hard in the simulation) that holds the lid at 80 degrees |

Damping, which the motion needs. Without the damping the lid let go at 40 degrees reaches the stop in 0.32 s at 4.6
rad/s (260 degrees per second), against 0.86 s at 1.5 rad/s (85 degrees per second) with it, and the results here are
for the second. About the hinge it is 6 N m s per rad above 30 degrees (and 0.3 below, from the hinge). At
each strut's rod that is a viscous damping of about 5 kN s per m (5.9 at 40 degrees, 5.3 at 60 and 4.7 at 80, from
`damping_at_roller`), a high value that a damped strut or a rotary damper on the hinge has to give. The simulation's 6
is flat from 30 degrees, a real damper's torque grows with the lever squared, so the two differ between 30 and 38
degrees only.

**What is not derived.** The strut maker, rod and tube diameters, end fittings, the lobe plate's thickness and the
bumper's material are not set: any strut with the force, stroke, length and damping above does it. Friction in the
roller, the struts' seals and the hinge is not modelled (the simulation has none beyond the viscous terms), so a
built version needs the checks below to be tuned with a different F1.

**Checks on a real lid.** Weigh the lid (6 kg) and its handle's distance from the hinge (0.365 m). With the struts
on:

1. Pull up on the handle with a spring scale, perpendicular to the line from the hinge, and hold the lid still at
   each angle (a protractor or a phone's level on the lid). The scale should read the table's handle force within
   about 20 percent: 35 N shut, 12 N at 34 degrees, then it must be pushed down, 9 N at 38 and 28 N at 80.
2. Let the lid go from rest at 34 degrees: it falls back. From 38 degrees up: it opens by itself to the stop. The
   balance angle is 36.4 degrees.
3. Let it go from rest at 40 degrees and time it to the stop: 0.8 to 0.9 s, arriving at about 85 degrees per second.
   A much shorter time means too little damping. (From a model of the lid's motion with the table's torques and the
   0.36 kg m2 inertia, 0.84 to 0.88 s for inertias from 0.30 to 0.45. It is not read from a simulation log.)
4. At 80 degrees the lid presses on the bumper with 28 N at the handle.

## Assets

| Object | Source |
|---|---|
| Laser cutter body, bed, gantry, head, lid | Task-file boxes (the lid's collider as above), drawn by meshes modelled in Blender (`looks/make_looks.py`, `looks/cutter_looks.py`): `cutter_body.usdc` on the front wall's box, `cutter_lid.usdc` on the lid's box (inside it) |

SimReady has no laser cutter. The textures (honeycomb, vents, screen) are drawn by the script. To change the meshes,
edit the scripts and run Blender 4.2 or later: `blender -b -P looks/make_looks.py`.

## Run it

On the YAM Brev launchable, from this folder, with the Arena environment active
(`source /data/IsaacLab-Arena/yam.env`):

```bash
# Dream-1 (needs DREAM_API_KEY), 1,200 steps
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=open_laser_cutter_lid.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Open the laser cutter's lid with both hands on the two ends of its black handle, and keep lifting until the gas struts take over and hold it fully open. Do not let it fall back." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/open-laser-cutter-lid-dream

# Scripted demo (both arms), 600 steps
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=open_laser_cutter_lid.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Open the laser cutter's lid with both hands on the two ends of its black handle, and keep lifting until the gas struts take over and hold it fully open. Do not let it fall back." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/open-laser-cutter-lid-demo
```

- The run prints `task_score` at the end. With `-E object_jitter=0` the layout is fixed and no seed is needed.
- The layout: the machine is centred on the robot's centre line, its front face 36 cm in front of the arm bases. Each
  arm works on its own end of the handle, 33 cm apart.
- The demo: each gripper points forward with its jaws closing up and down (`wrist.PinchWaypoint`), comes in round the
  bar 4.5 cm inside its post, closes and lifts the lid along its arc in 4 degree steps, turning with it, to 44
  degrees in 5.5 s. It lets go, backs off down and away, and the struts lift the lid on to its stop. It reaches 1.0
  and ends on success at step 268 (13.4 s at 20 Hz). A trial takes about 1.5 minutes on an L40S, including startup.
- Shortcuts, each the same demo with one constant changed (`scripts/task-ablate.sh`):
  - One hand only (`HANDS = 1`, the left arm stays at its ready pose): scores 0: the right arm alone lifts the 6 kg lid to 12 degrees and stalls, the handle slips out of its
    fingers and the lid slams shut.
  - Letting go before the struts take over (`LIFT_TO = 24.0`): scores 0: the lid falls back from 24 degrees and slams shut, which the trial records.
- Videos: `inspect-robots video ~/yam-eval/open-laser-cutter-lid-demo/<log>.json` writes one MP4 per camera next to
  the frames.
- Offline checks: `python -m pytest -q test_open_laser_cutter_lid.py`.

## Scoring

The lid's angle is how far it is open, in degrees, from shut (0) to its open stop (80): its turn about the hinge,
read from its orientation. In a real trial, a protractor or a phone's level app on the lid gives it.

- **0** if, at any time in the trial, the lid fell more than 20 degrees below the highest angle it had reached, or hit
  its closed stop (within 3 degrees of shut) closing faster than 60 degrees per second (a slam). The trial
  remembers both, so lifting the lid again afterwards does not help.
- **1.0:** the lid open 75 degrees or more and still (turning under 3 degrees per second). The episode ends in
  success once that has held for 1 s.
- **0.5:** the lid open 37 degrees or more at the end (past where the struts lift it by themselves).
- **0.25:** open 10 degrees or more at the end.
- **0** otherwise, including doing nothing.
