# Mark a cross's centre with an automatic centre punch

Machinists mark a hole's centre before drilling with an automatic centre punch: no hammer, the barrel is pushed down
over the nose against a spring and the punch fires at a set travel. The hardened tip skates on smooth steel while the
push builds, and the shot can't be undone. Here the robot lifts the punch out of its holder, sets the tip on a scribed
cross on a blue-dyed steel plate, pushes until it fires, and lifts it clear. One shot, within 0.5 mm of the cross
scores 1.0.

| File | What it is |
|---|---|
| `auto_center_punch_mark.py` | The task file: scene, shot logic (`watch_shots`), scoring, scripted `demo()` |
| `bonds.py`, `colliders.py`, `looks.py` | Hooks and the sprung slide, multi-box colliders, render-only meshes (as in the other tasks) |
| `looks/` | The meshes and `make_looks.py`, the Blender script (`blender -b -P looks/make_looks.py`) |
| `test_auto_center_punch_mark.py` | Offline checks: the parts fit, the shot logic, every scoring outcome, the demo's reach |
| `task.json` | The task's details for the board |

## The scene

Positions are centres in metres from the midpoint between the arm bases, heights above the table top.

| Part | Collider | Physics |
|---|---|---|
| plate | 11 x 7 x 1.2 cm at (0.30, -0.33), fixed | the scribed cross's centre is at (0.31, -0.324) |
| holder | 4 x 4 x 2.8 cm at (0.27, -0.14), fixed | a 2.1 cm pocket and a 7 mm hole for the tip, as box colliders |
| nose (`tip`) | 1.6 cm square nose, 10 mm stem, 3 mm tip, 40 g, free | the tip has friction 0.15 against the plate (lower of the two wins) |
| barrel | 2.2 x 2.2 x 8.5 cm sleeve, 0.12 kg, free | prismatic joint on the nose, 1500 N/m, 11 mm travel, collides as a hollow sleeve |

## Run it

On the YAM Brev launchable, from this folder, with the Arena environment active:

```bash
# Dream-1 (needs DREAM_API_KEY), 1,200 steps
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=auto_center_punch_mark.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mark the centre of the scribed cross on the blue steel plate with the red automatic centre punch, once, then lift the punch clear of the plate and hold it still." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/auto-center-punch-mark-dream

# Scripted demo (right arm), 600 steps
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=auto_center_punch_mark.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mark the centre of the scribed cross on the blue steel plate with the red automatic centre punch, once, then lift the punch clear of the plate and hold it still." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/auto-center-punch-mark-demo
```

- With `-E object_jitter=0` the layout is fixed and no seed is needed.
- The demo: the right hand grips the barrel from above (fingers along y), lifts the punch 4 cm out of its holder,
  carries it over the cross and lowers it until the tip is 3 mm above the plate. It then pushes down 3 cm over 7 s
  (about 4 mm/s): the tip lands, the barrel slides down over the nose and the punch fires at 7 mm of travel (about
  second 15). The hand leans on a little longer, lifts the punch 6 cm clear and holds it. The demo ends on success at
  step 460 (23 s at 20 Hz). The arm gives about two thirds of the hand's push, so the hand goes 3 cm past the touch
  for 10 mm of barrel travel. A trial takes about 8 minutes with the window open, and longer on a shared GPU.
- `AIM` is the hand's offset from the cross, 0.66 mm in y, measured: without it the tip lands 0.66 mm short and the
  demo scores 0.5. The mark lands 0.2 mm from the cross's centre.
- Review of the demo (`review_probe`): the arms stay 41.7 cm apart or more (the left arm holds READY), nothing falls,
  the punch's parts peak at 0.12 m/s and 3.8 rad/s while carried (the grip's lift and carry).
- Shortcuts, by `scripts/task-ablate.sh` constants: `SHOTS=2` (a second push after the first) scores 0.0.
  `SIDE=0.0015` (the hand pushes 1.5 mm to the side) scores 0.5 and `SIDE=0.004` scores 0.0: the tip skates with the hand.
- Offline checks: `python -m pytest -q test_auto_center_punch_mark.py`.
- Cheat variants of the demo, by constants: `SHOTS=2` (a second push, scores 0), `SIDE=0.004` (a sideways push while the
  barrel goes down).

## Scoring

The scorer reads the nose's and the barrel's poses every step (`on_step`). The barrel's travel is its centre's distance
along the nose's axis, less the unloaded distance.

- **Shot:** the barrel passes 7 mm of travel while cocked. The punch is cocked again once the travel is back under 3 mm.
- **Mark:** the centre of the tip's underside at the moment of the shot. On the plate if within 2 mm of the plate's top.
- **Clear:** the tip's underside 2 cm or more above the plate's top. **Still:** neither body's centre moved 1 mm or more over the last second (read every step: a held punch buzzes in the pads, so speeds are noise).
- **1.0 (held 1 s):** exactly one shot, on the plate, within 0.5 mm of the cross's centre, clear and still.
- **0.5:** exactly one shot on the plate within 1 mm (also within 0.5 mm when not clear and still).
- **0.25:** exactly one shot on the plate within 2 mm.
- **0** otherwise (no shot, two or more shots, off the plate, more than 2 mm off), and 0 if a part is 2 cm below the table top
  or the state is not finite. The shot memory is cleared by a reset hook.

In a real trial, a photo of the dent with a scale gives the offset, and the video shows the shots.

## Fidelity

- No hammer blow and no dent: the shot is a trip on the barrel's travel and the mark is a number the scorer keeps.
- The tip is a 3 mm square, not a cone, and the cross is a texture. Plate and holder are fixed.
- The demo reads nothing from the simulator: it plays fixed waypoints from the layout.
