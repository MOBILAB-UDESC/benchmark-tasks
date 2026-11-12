# Pose two friction-jointed helping hands to cross wire ends over a mark

Solderers (51-4121) set a helping-hands stand so two wire ends meet where a joint will be made. The arm's joints are
friction-held, so the work is in how it is moved:

- **Friction joints.** Each of the three joints stays put under a light push and turns only once pushed past its hold
  (0.10 N m), against a constant 0.10 N m, with no spring that brings it back. Moving the clip drags the two joints
  behind it, so the arm has to be moved by the body of its clip link, with the grip turned along with the link. Poking
  the clip moves whichever joint is weakest at the push point, and pulling the wire itself takes it out of its clip (a
  bond that breaks at 2 N), which scores 0.
- **Millimetres.** The wire ends must overlap by 5 mm with their centre lines within 2 mm of each other, over the mark,
  and lie against each other there. The two stands differ in height by one wire diameter (5 mm), so the 5 mm wires
  touch one on the other where they cross, a lap joint ready to solder.

| File | What it is |
|---|---|
| `helping_hands_cross_wires.py` | The `yam_arena` task file: layout, scoring (`success`, `score`), the finger watcher and the scripted `demo()` |
| `contact.py` | Reads the contact between the two wires from the simulator (a PhysX contact view, every physics substep) |
| `bonds.py` | `Hinge` (a vertical revolute joint with dry friction), `Bonded` (a breakable fixed joint), and the step and reset hooks |
| `fingers.py` | Where the grippers' fingers are, as points on their collision meshes (for "has the robot let go") |
| `looks.py`, `looks/` | Detailed meshes drawn on the boxes (`*.usdc`), and `make_looks.py`, the Blender script that builds them |
| `test_helping_hands_cross_wires.py` | Offline checks (no GPU) |
| `task.json` | The task's details for the board |

## The demo and what it shows

Each arm closes on its clip link (a straight-down grip across it, 8.5 cm behind the clip's face, the fingers first open
to 44 mm, then closed over 1.6 s), drags it to the crossing in one 4 s move that turns the grip along with the link,
holds 1 s so the joints settle, lets go, lifts off and waits at its own side of the table. The left stand is 5 mm
higher than the right one, so its wire comes to rest on the right one's, face to face along the overlap. The demo
reads only the fixed layout (a start that never changes with `object_jitter=0`) and no hidden state. The last waypoint of each arm is sent 3.3 mm past the mark in x (`AIM`) to make up what the friction holds back.

- **Demo:** 1.0, success at step 340 (17 s at 20 Hz, the 3 s hold included, 341 steps logged). A trial takes about 6
  minutes on an L40S shared with other runs, startup included.
- **Wire contact in the demo:** the simulator reports the wires touching once the crossing is made, and it stays so
  to the end. The reported separation is 0.3 to 0.8 mm, not 0 or less: the chain of friction hinges sags a fraction of
  a millimetre, so the left wire hangs a little above the right one's top face instead of pressing on it. That is why
  touching is defined as 1 mm or less.
- **Dream-1:** not run on this version (the earlier 0.0 belonged to the version with an 8 mm height difference).
- **Ablations (variants set with `scripts/task-ablate.sh`):**
  - `WIRE_PITCH=0.008` (the earlier layout, wire centres 8 mm apart in height, the same demo): scores 0.5, not 1.0: the
    wires are 3 mm apart in height, so they never touch.
  - `CHEAT = "wire"` (grip each wire stub instead of the clip link and drag it): scores 0, the wire comes out of its
    clip (the bond breaks at 2 N and the chain's friction needs about that much).
  - `CHEAT = "keep"` (drag the same way, but never let go): scores 0.5, the crossing is made but a gripper is still on
    the stands.
  - Run each with `scripts/task-ablate.sh helping-hands-cross-wires <name> 'WIRE_PITCH=0.008'` (or `'CHEAT="wire"'`,
    `'CHEAT="keep"'`).

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
- Offline checks: `python -m pytest -q test_helping_hands_cross_wires.py` (36 tests).
- Ablations: see the list above (`scripts/task-ablate.sh`).
- To change the meshes, edit `looks/make_looks.py` and run it in Blender 4.2 or later: `blender -b -P looks/make_looks.py`.

## Scoring

Positions are in metres in the station frame (x forward, y to the robot's left, from the midpoint of the arm bases).
Measured on the two wires' free ends, seen from above:

- **Overlap:** how far the ends have passed each other along the wires (negative if short).
- **Sideways gap:** the larger distance of one end from the other wire's centre line.
- **Mark distance:** from the midpoint of the two ends to the cross's centre (0.31, 0).
- **Contact:** the simulator's contact data between the two wire bodies, read every physics substep (240 Hz). The
  wires touch when it reports a contact point between them at a separation of 1 mm or less (0 is touching, a
  negative separation is pressing into each other) and their faces are within 1 mm of each other by their heights
  (the centres' height difference differs from 5 mm by at most 1 mm).

Outcomes:

- **0** if at any step a wire's centre was 5 mm or more from where its clip holds it (remembered for the trial), or if
  a part fell off the table (centre more than 2 cm below the table top), or the state is not finite.
- **1.0** (success): overlap 5 mm or more, sideways gap 2 mm or less, mark distance 5 mm or less, every link and wire
  still (under 1 cm/s and 0.5 rad/s), the two wires touching, no finger point within 3 cm of any link or wire, held for 3 s.
- **0.5:** overlap at least -10 mm, gap 10 mm or less, mark distance 10 mm or less, whether or not the wires touch,
  still, let go or held.
- **0.25:** each end within 3 cm of the cross's centre.
- **0** otherwise, including doing nothing.

Real-world check: seen from the side, no daylight shows between the two wires along their overlap (a 1 mm feeler
gauge does not pass between them), and from above the overlap is 5 mm or more with the centre lines within 2 mm and the
midpoint within 5 mm of the cross. A top ruler cannot tell a lap with a 3 mm gap from a touching one, so the contact is
scored from the simulator and from the wires' heights, and a lap with a gap does not score 1.0.
