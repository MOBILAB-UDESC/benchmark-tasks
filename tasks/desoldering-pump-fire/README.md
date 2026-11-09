# Cock a desoldering pump and fire it on the marked joint

Clearing a solder joint with a spring-loaded desoldering pump, the way an electronics repairer does it. The pump has to
be cocked first (the plunger pushed down until a latch catches it), then set upright with its nozzle on the joint and
fired with its button. Pressing the button of an uncocked pump does nothing, and a shot fired off the joint is wasted:
the pump has to be cocked again. One arm cocks the pump, lifts it out of its holster and holds it on the joint. The
other arm presses the button, 8 cm above the hold.

| File | What it is |
|---|---|
| `desoldering_pump_fire.py` | The `yam_arena` task file: scene, the latch and the episode's memory, scoring (`success`, `score`) and the scripted `demo()` |
| `bonds.py` | `Bonded`: a box fixed to another. `Sliding`: a box on a sprung prismatic joint. `lock_slide` and `unlock_slide`: hold a sliding box where it is, by closing and opening its joint's limits |
| `wrist.py` | `TiltWaypoint`: demo waypoints whose grip tilts forward, and turns about the vertical |
| `looks.py` | Puts a detailed mesh on a box at spawn and hides the plain box, which still collides |
| `looks/` | The meshes (`*.usdc`), the board's texture, and `make_looks.py`, the Blender script that models and exports them |
| `test_desoldering_pump_fire.py` | Offline checks: the layout, the latch, every scoring outcome and exploit, the demo's reach and the arms' clearance (no GPU) |
| `task.json` | The task's details for the board |

## The pump

Everything is `yam_arena` boxes with PhysX joints authored when they spawn. No assets are downloaded.

| Part | Box | Joint |
|---|---|---|
| barrel | 2.4 x 2.4 x 12 cm, 60 g, free | none: it stands on the holster's blocks |
| nozzle | 5 x 5 x 20 mm, 5 g | fixed to the barrel's underside, no collision with it. Its tip is the point scored |
| plunger | 8 x 8 x 42 mm collider (rod and knob thickness), 40 g | prismatic along the barrel's axis, 2.5 cm of travel, a 240 N/m spring back up (6 N when cocked), no collision with the barrel. The black knob, 3 cm across, is drawn wider than its collider |
| button | 12 x 8 x 14 mm, 5 g | prismatic along y, 4 mm of travel into the barrel's left side, a 1,500 N/m spring |
| latch | the plunger's joint limits | a step hook watches the plunger: at the bottom of its stroke (within 1 mm) it sets both limits to the bottom, so the spring can't lift it. With the pump armed, the button pressed 2.5 mm sets the limits back and the plunger flies up. It can't be latched again until it has come back up half way |

## What it looks like

Each box keeps its collider, and `looks.py` draws a mesh in its place: the barrel with its label band, grip ring and
button slot, the nozzle, the plunger's knob and rod, the button's cap and stem, the holster's blocks with rubber pads,
and the board with traces, pads and silkscreen. The three solder joints stay plain silver cylinders, and the sticker a
red box. The plunger's mesh is centred on its box in x and y and stays inside it in z, so what the scorer reads is the
box's centre. To change the meshes, edit `looks/make_looks.py` and run it in Blender 4.2 or later:
`blender -b -P looks/make_looks.py`.

## Run it

On the YAM Brev launchable, from this folder, with the Arena environment active
(`source /data/IsaacLab-Arena/yam.env`):

```bash
# Dream-1 (needs DREAM_API_KEY), 1,200 steps
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=desoldering_pump_fire.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Cock the blue desoldering pump, rest its nozzle on the solder joint that sits on the red sticker, fire the pump there with its white button, then stand the pump back in its holster." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/desoldering-pump-fire-dream

# Scripted demo (both arms), 1,200 steps
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=desoldering_pump_fire.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Cock the blue desoldering pump, rest its nozzle on the solder joint that sits on the red sticker, fire the pump there with its white button, then stand the pump back in its holster." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/desoldering-pump-fire-demo
```

- The run prints `task_score` at the end. With `-E object_jitter=0` the layout is fixed and no seed is needed. Keep it
  at 0: the demo is open loop and the joints are only 2 cm apart.
- The demo, from the fixed layout only. Right arm: closed fingers press the knob straight down past
  the latch, then the gripper points forward and 20 degrees down and closes on the barrel's flat sides 2.5 cm above its
  bottom, lifts the pump out, carries it over the board, sets the nozzle on the middle joint (the TCP is sent 3 mm
  lower so the pump rests on it) and holds it. Left arm: closed fingers point along -y and 20 degrees down, touch the
  button, press 8 mm past its face (4 mm of travel) for 1.5 s and back off. The right arm then lifts the pump, puts it
  back in the holster and lets go. It reaches 1.0 and ends on success at step 658 (32.9 s at 20 Hz). A trial took 6 minutes on an L40S
  shared with other runs, including startup.
- Cheat variants of the same demo, scored in the simulator (`scripts/task-ablate.sh`, 900 steps):
  - `COCK = False`: the right arm skips the knob, so the pump is never armed and the left arm's press does nothing. Scores 0.
  - `JOINT_USED = 0`: the whole demo, but the nozzle is set on the neighbouring joint 2 cm away, cocked and fired
    there and put back in the holster. Scores 0, because that shot is neither near the marked joint nor was the pump
    ever armed on it.
  - `FIRE = False` (left arm never presses): the pump is armed on the joint and never fired, which scores 0.25
    by the scoring text (checked offline in the test file, not rerun in the simulator).
- Videos: `inspect-robots video ~/yam-eval/desoldering-pump-fire-demo/<log>.json` writes one MP4 per camera next to
  the frames.
- Offline checks: `python -m pytest -q test_desoldering_pump_fire.py`.

## Scoring

All positions are centres, in metres, in the station frame (x forward, y to the robot's left, from the midpoint of
the arm bases), and heights are above the table top. The scorer reads the barrel's, nozzle's, plunger's and button's
poses after every control step.

- **Cocked:** the plunger 2.4 cm or more below its rest, along the barrel's axis. The latch then holds it.
- **A shot:** while cocked, the button pressed 2.5 mm or more along the barrel's y axis. The latch lets go and the
  plunger rises. The scorer remembers each shot's tip error and tilt at that step.
- **The tip:** the centre of the nozzle's bottom face. Its error is its distance to the top of the middle joint
  (0.31, 0, 2.5 mm above the board).
- **Good shot:** tip error 2 mm or less and the barrel within 15 degrees of vertical. **Near shot:** tip error 6 mm
  or less.
- **In the holster:** the barrel's centre within 8 mm of the hole's axis (0.27, -0.15) and 4 mm of its resting
  height, and within 5 degrees of vertical.
- **0** if the pump has fallen off the table (barrel centre more than 2 cm below the top) or a state is not finite.
- **1.0 (success, held 1 s):** a good shot at some step, and now the pump in the holster with the barrel and the
  plunger still (under 1 cm/s and 0.5 rad/s).
- **0.75:** a good shot at some step, and the pump not in the holster and still.
- **0.5:** no good shot, but a near shot at some step.
- **0.25:** no near shot, but at some step the pump was cocked with its tip within 2 mm of the joint's top and
  within 15 degrees of vertical.
- **0** otherwise, including doing nothing, pressing the button of an uncocked pump, or firing only at the outer joints.

The latch and the memory reset on every reset. In a real trial, a ruler and a side photo at the shot give the tip's
place and the tilt, and a held pump counts as set down only if the grippers have let go (the simulator can't tell, so
it scores the pump standing still in the holster).
