# Seat a heat-set insert flush with a soldering iron

Installing a threaded insert in a 3D-printed part, the way a team assembler does it: a brass heat-set insert sits on a
printed hole, a hot iron pushes it in, and the plastic flows round it and sets. The insert keeps sinking for as long
as the iron pushes and stays where it is when the push stops, so the push has to stop at flush. Push on and the insert
is buried and the thread is lost, tilt the iron and the insert sets crooked, miss the hole and the tip scorches the
part's face. Here the robot pushes the insert in until its top is flush with the bracket's face within 0.3 mm and
square within 3 degrees, then takes the iron away.

The ability: **pushing a tool into a yielding material and stopping at a depth**, with nothing hard to stop against.
It is not a peg in a hole (the part moves under the tool and stays), nor a placement.

| File | What it is |
|---|---|
| `heat_set_insert_flush.py` | The `yam_arena` task file: scene, scoring (`success`, `score`) and the scripted `demo()` with its cheat variants |
| `seat.py` | `Seated`: a box on a D6 joint that sinks along one axis against a damper (no spring) and cocks about its centre against another |
| `colliders.py` | A box that collides as several smaller boxes (the bracket with its hole, the iron, the stand's lips) |
| `grip.py` | `HandWaypoint`: demo waypoints with the gripper pointing any way (a small pitch levels the held iron) |
| `bonds.py` | The `on_object`, `on_spawn`, `on_step` and `on_reset` hooks |
| `looks.py` | Puts a detailed mesh on a box at spawn and hides the plain box, which still collides |
| `looks/` | The meshes (`*.usdc`) and their textures, and `make_looks.py`, the Blender script that models and exports them |
| `test_heat_set_insert_flush.py` | Offline checks: the file loads, the bracket's colliders leave only the hole open, the iron rests on its stand, the looks match the boxes, every scoring outcome, the demo's reach |
| `task.json` | The task's details for the board |

## The scene

Positions are centres in metres from the midpoint between the arm bases (x forward, y to the robot's left), heights
above the table top.

| Part | Collider | Physics |
|---|---|---|
| bracket | 3 x 5 x 8 cm block, face at x = 0.44, centre (0.455, -0.06), fixed | box colliders: a solid back and a 10.5 mm deep ring in front of it with an 8 mm square opening, 6.6 cm up. The drawn hole is round, 5.6 mm |
| insert | 8 x 6 x 6 mm box, 4 g, centre (0.436, -0.06), 6.6 cm up, its length along x | one D6 joint to the bench: slides along x from where it spawns to 10.5 mm in against a damper of 250 N s/m and no spring (4 mm/s for each newton), cocks up to 10 degrees about its centre against 0.001 N m s/rad and no spring, locked in the other three. Drawn as a knurled brass sleeve with a bore |
| iron | 22 x 2 x 2 cm box, 60 g, centre (0.25, -0.26), 5.5 cm up, tip toward +x | box colliders: a 2 cm handle (11 cm), an 8 mm barrel (6.5 cm) and a 2.6 mm square pusher tip (4.5 cm). Free. Drawn with a blue handle, steel barrel and a flat-faced pin |
| stand | base 19 x 7 x 1.2 cm at (0.25, -0.26), fixed | boxes: the base, a flat rest under the handle and a flat cradle under the barrel (no lips: a yawed iron would catch on them) |

The tip is narrower than the insert and the drawn hole, so it follows the insert into the hole and nothing but the
iron stops the insert at flush. The insert can sink 10.5 mm in all, 2.5 mm past flush, then it bottoms out.

Nothing in the scene is a hidden mechanism: the insert's state is its pose, and the scorer reads only poses.

SimReady has no soldering iron, insert or printed part (its bench vises are 15 kg articulated machines), so the bracket,
insert, iron and stand are modelled in Blender (`looks/make_looks.py`) and drawn on their boxes. Every mesh that is
scored fits inside its box (`make_looks.py` checks it when it exports). To change them, edit `looks/make_looks.py` and
run `blender -b -P looks/make_looks.py` (Blender 4.2 or later) from this folder.

## Run it

On the YAM Brev launchable, from this folder, with the Arena environment active
(`source /data/IsaacLab-Arena/yam.env`):

```bash
# Dream-1 (needs DREAM_API_KEY), 1,200 steps
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=heat_set_insert_flush.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "With the soldering iron (blue handle), push the brass threaded insert into the hole in the white printed bracket until its top is flush with the bracket's face and it sits straight, then take the iron away." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/heat-set-insert-flush-dream

# Scripted demo (right arm), 600 steps
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=heat_set_insert_flush.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "With the soldering iron (blue handle), push the brass threaded insert into the hole in the white printed bracket until its top is flush with the bracket's face and it sits straight, then take the iron away." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/heat-set-insert-flush-demo
```

- The run prints `task_score` at the end. With `-E object_jitter=0` the layout is fixed and no seed is needed.
- The demo (right arm only): grip the iron's handle over its centre of mass (fingers closing across it, the TCP 6 mm
  above its axis so the pads' tips clear the rest), lift it 11 mm off its rest, which brings its axis to the hole's
  height, carry it across to the hole's line, bring its tip to 1.5 cm short of the insert, push 2.3 cm along x over
  5 s (the insert sinks 8 mm to flush as the arm leans on it), hold 2 s, and draw the iron back 4 cm. The demo ends on
  success at step 395 (19.8 s at 20 Hz). A trial takes about 2.5 minutes on an L40S, including startup.
- The demo decides from the fixed layout only (no insert pose, joint state or scorer flag): `AIM` (the tip's aim,
  0.6 mm too low and 5.7 mm to the right of the hole's axis without it, measured as the iron sags and the wrist lags)
  and `PITCH_TRIM` (4 degrees, tip lower: a level iron leaves the insert 3.8 degrees off, 4 degrees leaves it 1.9, 8
  degrees 4.2) were tuned in the simulation. The fingers are sent closing along -y: the other way round puts joint 6
  on its limit at the stand and the wrist stops 7 degrees short of square.
- Review of the demo: only one arm moves (the arms' links stay 34 cm apart or more), nothing falls, the insert's peak
  speed is 0.03 m/s and the iron's 0.10 m/s (1.4 rad/s) while carried, the tip never touches the bracket's face, and
  the insert rocks to 9.5 degrees as the iron first presses it, then comes square (1.3 degrees at the end).
- Variants of the demo (constants in the task file, `scripts/task-ablate.sh heat-set-insert-flush <name> NAME=value`):

| Variant | Constant | Score | What happens |
|---|---|---|---|
| the demo | none | 1.0 | proud 0.0 mm, 1.3 degrees off, iron clear |
| push on to "the end" | `OVERSHOOT = 0.003` | 0 | the insert is buried 2.4 mm (ablation run) |
| stop early | `SHORT = 0.004` | 0.25 | the insert is 3.9 mm proud, half in |
| iron pitched 8 degrees more | `TILT_DEG = -8` | 0.5 | flush, 6.9 degrees off |
| iron 2 mm off the hole's axis | `OFFSET_Y = 0.002` | 0 | the tip's edge reaches the face: 0 for the rest of the trial |
| never take the iron away | `HOLD_ON = 1` | 0.5 | flush within 1 mm, 4 degrees off, iron on it |

- Videos: `inspect-robots video ~/yam-eval/heat-set-insert-flush-demo/<log>.json` writes one MP4 per camera next to the
  frames.
- Offline checks: `python -m pytest -q test_heat_set_insert_flush.py`.

## Scoring

Positions are in metres in the station frame. The insert's top is the centre of its end toward the robot, half its
length back from its centre along its own axis.

- **Proud:** how far the top stands out of the bracket's face, along the hole's axis: 8 mm at the start, 0 when flush,
  negative when buried.
- **Square:** the angle between the insert's axis and the hole's.
- **Clear:** the iron's tip face at least 1 cm from the hole's mouth.
- **Still:** the insert under 1 cm/s and 0.5 rad/s (`resting()`).
- **Touched the face:** at any step, a point of the tip (the four corners of its face, and the four 1 cm behind it)
  at the face's plane (within 0.5 mm) or past it, and more than 3 mm from the hole's axis across or up. The scorer
  reads the iron's pose every step and remembers it until the next reset.
- **1.0 (success, held 1 s):** proud within 0.3 mm of 0, square within 3 degrees, still, and the iron clear.
- **0.5:** proud within 1 mm of 0 and square within 8 degrees, but not all of the above.
- **0.25:** proud no more than 4 mm (at least half in) and not buried, but not the above.
- **0** otherwise, and **0** whatever else if the insert is buried at the end (its top more than 1 mm below the face),
  if the tip touched the face, if the iron or insert fell off the table (2 cm below its top), or if the state is not
  finite. Doing nothing scores 0.

In a real trial, a ruler or feeler gauge across the face shows how proud the insert is, a square shows how crooked
it is, and a scorch mark on the face shows the iron missed.

## Layout and reach

The bracket's face is at x = 0.44 and the hole's axis at y = -0.06, 6.6 cm up. The iron lies on its stand at (0.25,
-0.26), its handle 4.5 cm behind the grip. The right arm reaches the grip at the stand and the push (the TCP at x =
0.285 with the tip at the face) with the fingers closing along -y: `ArmPlan` worst over the demo is under 0.1 mm and 0.1
degrees. The left arm stays at READY, far from the work. The stand has no lips: the pads' tips go 1.3 cm below the TCP
and a yawed iron's barrel caught on lips and was flung (seen in the first runs).

## Fidelity

- Heat is not modelled. The insert is a 4 g box on one D6 joint to the bench: along the hole it moves only while
  pushed (a damper of 250 N s/m, no spring: 4 mm/s for each newton) and stays where it is left, over 10.5 mm (2.5 mm
  past flush, where it bottoms out). It cocks up to 10 degrees about its centre against 0.05 N m s/rad, and is locked
  in the other three degrees of freedom. Real plastic sets, so a push that stops leaves the insert where it is, as here.
- The hole's colliders are an 8 mm square opening 10.5 mm deep (the drawn hole is round, 5.6 mm), so the insert can
  cock in it. Whether the tip is "in the hole" is the scorer's 3 mm rule, not a collision.
- The iron is cold, 60 g, with a flat 2.6 mm square pusher 4.5 cm long, narrower than the insert so it follows the
  insert into the hole (a real tip has a pilot and a flat face). Its colliders are three boxes: handle, barrel, tip.
- The bracket, stand and iron stand are fixed or at rest on boxes. Friction is yam_arena's for task objects (1.2).
- The top camera sees the 6 mm insert as a few pixels at 320 x 176: the task is hard for a vision-only policy.
