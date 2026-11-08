# Lift a long thin panel at points that keep it flat

Carrying a long circuit-board panel to a rack, the way a production worker does. The panel is thin and bends: lifted by
its two ends, its middle sags past the 4 mm bow at which a populated panel cracks, and held near its middle, both ends
droop. About a fifth of its length in from each end keeps it flat. The instruction does not say where to hold. The task
tests choosing where to hold a bending part with two hands.

| File | What it is |
|---|---|
| `lift_long_panel_no_sag.py` | The `yam_arena` task file: scene, scoring (`success`, `score`) and the scripted `demo()` |
| `panelkit.py` | `Panel`: a chain of rigid segments on springy hinges that bends like a strip of a set stiffness. Hooks the scene, the reset, and the drawing of boxes |
| `looks/` | The meshes (`*.usdc`) and the texture, and `make_looks.py`, the Blender script that models and exports them |
| `test_lift_long_panel_no_sag.py` | Offline checks: the file loads, every scoring outcome and its memory, a static model of the chain, the demo's order and reach (no GPU) |
| `task.json` | The task's details for the board |

## The panel

`panelkit.py` spawns the panel as one PhysX articulation: 20 boxes of 3 x 6 x 0.6 cm along y, each joined to the next
by a revolute hinge about x with a spring and a damper. Each hinge's spring is EI divided by the segment length, so the
chain bends like a uniform strip of bending stiffness EI = 0.07 N m^2, which a 0.8 mm FR-4 strip 6 cm wide also has.
The panel weighs 120 g (populated). The hinges allow no stretch, no twist and no sideways bend. It starts flat on a
black bench rail 3 cm high. The rack's rail is orange, 8 cm high and 14 cm farther from the robot.

A static model of the chain (in the tests) lifted at two points gives these bows, for a gripper that keeps the held
segment level ("clamped") and one that lets it turn ("pinned"):

| Held at | Clamped | Pinned |
|---|---|---|
| the ends | over 7 mm | over 35 mm |
| 13 cm in from each end (the demo) | about 1 mm | about 1 mm |

## Assets

No SimReady package fits: the panel has to bend, which a rigid SimReady body can't, and no package is a bench-scale
rack. The panel and the two rails are modelled in Blender (`looks/make_looks.py`, run with
`blender -b -P looks/make_looks.py`) as render-only meshes. The physics stays the panel's 20 segment boxes and the rails'
boxes, so what the cameras see is what collides. The panel's texture is drawn in the script with numpy, nothing is
downloaded.

## Run it

On the YAM Brev launchable, from this folder, with the Arena environment active
(`source /data/IsaacLab-Arena/yam.env`):

```bash
# Dream-1 (needs DREAM_API_KEY), 600 steps
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=lift_long_panel_no_sag.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Carry the long blue circuit-board panel to the orange rack with both arms and set it down on the rack's rail." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/lift-long-panel-no-sag-dream

# Scripted demo (both arms), 600 steps
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=lift_long_panel_no_sag.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Carry the long blue circuit-board panel to the orange rack with both arms and set it down on the rack's rail." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/lift-long-panel-no-sag-demo
```

- The run prints `task_score` at the end. With `-E object_jitter=0` the layout is fixed and no seed is needed.
- A trial takes about 1 minute on an L40S, including startup.
- The demo: both arms come down beside the panel at y = +-17 cm (13 cm in from each end), pinch its width with the pad
  tips 1 cm under its lower face, lift it 12 cm together, carry it 14 cm forward at that height, lower it onto the
  rack's rail, open and back off. It reaches 1.0 and ends on success at step 232 (11.6 s at 20 Hz), with a peak bow of
  1.0 mm, a peak segment speed of 0.12 m/s and the arms at least 28 cm apart (link origins). A trial takes about 5 minutes
  on a shared L40S, 1 minute alone.
- Offline checks: `python -m pytest -q test_lift_long_panel_no_sag.py`.

## Scoring

Heights are above the table top. The bow is the greatest distance of any of the panel's 20 segment centres from the
straight line through the first and the last, read at every control step. A real trial lays a straightedge along the
panel's top face and reads the largest gap under it.

- **Peak bow:** the largest bow at any step of the trial. The scorer remembers it, and clears it on every reset.
- **Lifted clear:** at some step, every segment is at least 3 cm above its start height. The scorer remembers it too.
- **0** if the peak bow goes over 4 mm at any step, if any segment falls more than 2 cm below the table top, if the
  state is not finite, or if the panel was never lifted clear.
- **1.0 (success, held 1 s):** the panel lies on the rack's rail and is still, with a peak bow of 3 mm or less. On the
  rail means every segment's centre within 2 cm of the rail's line along x and within 4 mm of lying on it (height of
  the rail's top plus half the panel's thickness), and the panel's centre within 2 cm of the rail's middle along y.
  Still means every segment under 1 cm/s and 0.5 rad/s.
- **0.75:** the same, with a peak bow over 3 mm and up to 4 mm.
- **0.5:** lifted clear with a peak bow of 4 mm or less, but not still on the rack at the end: still carried, or set
  down anywhere else, the bench rail included.
- So the score is 0, 0.5, 0.75 or 1.0, and doing nothing scores 0.

## Cheat and ablation variants

`scripts/task-ablate.sh lift-long-panel-no-sag <name> GRIP_Y=<m>` runs the demo on a copy with both hands moved to
y = +-`GRIP_Y` (the demo's `GRIP_Y = 0.17` is 13 cm in from each end, and the panel's ends are at 0.30). Results of the
same carry, all scripted, `-E object_jitter=0`:

| Hands | Score | What happened |
|---|---|---|
| `GRIP_Y = 0.17`, 13 cm from each end (the demo) | 1.0 | peak bow 1.0 mm |
| `GRIP_Y = 0.15`, 15 cm from each end | 1.0 | still flat: the zone that stays under 3 mm is wide on the inner side |
| `GRIP_Y = 0.24`, 6 cm from each end | 0.75 | peak bow between 3 and 4 mm, set down on the rack |
| `GRIP_Y = 0.285`, at the very ends | 0.0 | peak bow 8.7 mm: the panel cracks on the way, though it ends still on the rack |

In the simulator a held segment stays level, so the "clamped" column of the model above is the one that applies.
Holding in the middle can't be scripted: with both hands within 15 cm of the centre line the arms can't reach the rack.
One-arm holds and drags along the bench rail were not run: the scorer gives 0 to a panel that never has every segment 3 cm above its start (tested offline), and the chain model gives a bow far over 4 mm for any hold within 6 cm of the middle.
