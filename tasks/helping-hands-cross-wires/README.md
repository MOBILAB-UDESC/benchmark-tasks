# Pose two friction-jointed helping hands to cross wire ends over a mark

Solderers (51-4121) set a helping-hands stand so two wire ends meet where a joint will be made. The arm's joints are
friction-held, so the work is in how it is moved:

- **Friction joints.** Each of the three joints stays put under a light push and turns only once pushed past its hold
  (0.10 N m), against a constant 0.10 N m, with no spring that brings it back. Moving the clip drags the two joints
  behind it, so the arm has to be moved by the body of its clip link, with the grip turned along with the link. Poking
  the clip moves whichever joint is weakest at the push point, and pulling the wire itself takes it out of its clip (a
  bond that breaks at 2 N), which scores 0.
- **Millimetres.** The wire ends must overlap by 5 mm with their centre lines within 2 mm of each other, over the mark.

| File | What it is |
|---|---|
| `helping_hands_cross_wires.py` | The `yam_arena` task file: layout, scoring (`success`, `score`), the finger watcher and the scripted `demo()` |
| `bonds.py` | `Hinge` (a vertical revolute joint with dry friction), `Bonded` (a breakable fixed joint), and the step and reset hooks |
| `fingers.py` | Where the grippers' fingers are, as points on their collision meshes (for "has the robot let go") |
| `looks.py`, `looks/` | Detailed meshes drawn on the boxes (`*.usdc`), and `make_looks.py`, the Blender script that builds them |
| `test_helping_hands_cross_wires.py` | Offline checks (no GPU) |
| `task.json` | The task's details for the board |

## The demo and what it shows

Each arm closes on its clip link (a straight-down grip across it, 8.5 cm behind the clip's face, the fingers first open
to 44 mm, then closed over 1.6 s), drags it to the crossing in one 4 s move that turns the grip along with the link,
holds 1 s so the joints settle, lets go, lifts off and waits at its own side of the table. The demo reads only the fixed
layout (a start that never changes with `object_jitter=0`) and no hidden state. The last waypoint of each arm is sent
3.3 mm past the mark in x (`AIM`) to make up what the friction holds back.

- **Demo:** 1.0, success at step 340 (17 s at 20 Hz, the 3 s hold included). A trial takes about 6 minutes on an L40S
  shared with other runs, startup included.
- **Final crossing in the demo:** overlap 13.6 mm, sideways gap 1.1 mm, midpoint 0.1 mm from the mark, held 3 s.
- **Review (motion):** the arms' closest link origins stay 10.5 cm apart. No object falls. Peaks of the free parts:
  links 1 and 2 at most 0.03 m/s and 0.6 rad/s, the right clip link 0.04 m/s and 1.7 rad/s, the wires 0.04 m/s
  (right) and 0.21 m/s (left, a 0.3 s jolt as the pads close on the clip link) and 3.6 rad/s. The left clip link
  reads 14.5 rad/s about the world y axis for about 1.5 s while it is carried, with no change in its orientation:
  the reading is a PhysX velocity artifact of a gripped body, not a spin (the same link at rest reads 0).
- **Ablations (cheat variants, `CHEAT` in the task file):**
  - `CHEAT = "wire"` (grip each wire stub instead of the clip link and drag it): scores 0, the wire comes out of its
    clip (the bond breaks at 2 N and the chain's friction needs about that much).
  - `CHEAT = "keep"` (drag the same way, but never let go): scores 0.5, the crossing is made but a gripper is still on
    the stands.
- **Creep:** after the arms let go, the clip links turn about 0.1 degree per second (0.2 to 0.4 mm/s at the wire
  ends) in the direction the drag came from, so a crossing set 1.1 mm sideways stays inside 2 mm for 3 s but not for
  much longer.

## Run it

On the YAM Brev launchable, from this folder, with the Arena environment active
(`source /data/IsaacLab-Arena/yam.env`):

```bash
# Dream-1 (needs DREAM_API_KEY), 600 steps
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=helping_hands_cross_wires.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Move the two helping-hands arms until the red wire end and the blue wire end cross over the yellow cross on the table, then let go of both arms and leave them there." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/helping-hands-cross-wires-dream

# Scripted demo (both arms), 600 steps
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=helping_hands_cross_wires.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Move the two helping-hands arms until the red wire end and the blue wire end cross over the yellow cross on the table, then let go of both arms and leave them there." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/helping-hands-cross-wires-demo
```

- With `-E object_jitter=0` the layout is fixed and no seed is needed. Keep it at 0: jitter would move the links away from
  their hinges.
- Offline checks: `python -m pytest -q test_helping_hands_cross_wires.py` (30 tests).
- Cheat variants: `scripts/task-ablate.sh helping-hands-cross-wires <name> 'CHEAT="wire"'` (or `"keep"`).
- To change the meshes, edit `looks/make_looks.py` and run it in Blender 4.2 or later: `blender -b -P looks/make_looks.py`.

## Scoring

Positions are in metres in the station frame (x forward, y to the robot's left, from the midpoint of the arm bases).
Measured on the two wires' free ends, seen from above:

- **Overlap:** how far the ends have passed each other along the wires (negative if short).
- **Sideways gap:** the larger distance of one end from the other wire's centre line.
- **Mark distance:** from the midpoint of the two ends to the cross's centre (0.31, 0).

Outcomes:

- **0** if at any step a wire's centre was 5 mm or more from where its clip holds it (remembered for the trial), or if
  a part fell off the table (centre more than 2 cm below the table top), or the state is not finite.
- **1.0** (success): overlap 5 mm or more, sideways gap 2 mm or less, mark distance 5 mm or less, every link and wire
  still (under 1 cm/s and 0.5 rad/s), no finger point within 3 cm of any link or wire, held for 3 s.
- **0.5:** overlap at least -10 mm, gap 10 mm or less, mark distance 10 mm or less, whether or not still, let go or held.
- **0.25:** each end within 3 cm of the cross's centre.
- **0** otherwise, including doing nothing.
