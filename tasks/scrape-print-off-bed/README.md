# Scrape a stuck print off the build plate inside a 3D printer

Taking a finished print off a 3D printer, the way a makerspace or print farm technician does it dozens of times a
day. The printer is an enclosed desktop FDM printer with its door swung open. The print is stuck to the removable
build plate, low inside, just behind the door opening. Pulled straight up, it does not come off: the plate's magnets
give first and the plate comes up with it. A scraper's blade slid along the plate under the print's chamfered edge
levers the edge up and breaks the print free, then the print goes in a basket and the scraper back on the table.

The plate's magnets hold it against the scraper's push, so one hand can do the whole task. The demo uses the left arm
to carry the print to the basket on the left, but holding the plate down is not needed (an earlier version claimed it
was). The scraper has to reach into the printer, under its frame, and the hands work around the enclosure.

| File | What it is |
|---|---|
| `scrape_print_off_bed.py` | The `yam_arena` task file: scene, scoring (`success`, `score`), the blade watcher and the scripted `demo()` |
| `printer.py` | The printer's geometry and its box colliders (base, bed, walls, roof, front frame, gantry, door) |
| `bonds.py` | `Bonded`: a task-file box held by a bond that breaks at a set pull and twist, and the step and reset hooks |
| `colliders.py` | Makes a task-file box collide as several smaller boxes, each with its own friction (the printer, the door, the print's foot, the scraper's blade) |
| `simready.py` | Loads the SimReady basket (pinned revision), fixed |
| `wrist.py` | `TiltWaypoint`: demo waypoints with the grip tilted about the fingers' axis (the scraper held at 8 degrees, the print taken from the front) |
| `looks.py` | Puts detailed meshes on a box at spawn and hides the plain box, which still collides |
| `looks/` | The meshes (`*.usdc`) and their textures, and `make_looks.py`, the Blender script that models and exports them |
| `test_scrape_print_off_bed.py` | Offline checks: the bonds, the parts, every scoring outcome, the blade-under-the-edge test, the meshes in their boxes, the demo's order, reach and clearance from the printer (no GPU) |
| `task.json` | The task's details for the board |

## The printer

A 36 x 35 x 40 cm enclosed printer, fixed, centred on the table's midline at x 0.49, its front toward the robot: a
light grey base with a sloped front and a small screen, a black aluminium frame, light grey side and back panels, a
glass roof, a light under the front beam. Its door, tinted clear acrylic in a black border with a bar handle, is
hinged at the front left and swung fully open (180 degrees), standing beside the printer's left side. The door
opening is 31 cm wide, from the base's top (8 cm) to 36 cm. Inside, the black heated bed is low (its top at 10 cm)
and just behind the opening, and the print head is parked at the front of its gantry, above the plate. The arms
cannot pass through any of it: the printer collides as its base, bed, side and back walls, roof, front frame and
gantry, the open door as its panel and handle (`printer.py`). The same model, made by `looks/make_looks.py`, is the
printer of the `open-printer-door` task.

## The adhesion, the magnets and the wedge

Both bonds are PhysX fixed joints authored by `bonds.py` when the boxes spawn, anchored at the centre of the box's
bottom face:

| Bond | Joint | Behaviour |
|---|---|---|
| print to plate | `physics:breakForce` 60 N | the print's adhesion |
| plate to bed | `physics:breakForce` 5 N | the bed's magnets. Pulled up harder than 5 N, the plate comes off the bed |

So a print pulled straight up takes the plate with it. The print stands on a 3.6 x 3.6 cm foot 2 mm tall (its
chamfered first layers), which leaves a 2 mm gap 12 mm deep under each edge. The scraper is one rigid body: a
12 x 2.8 x 2.2 cm handle and a 10 x 4 cm steel blade 1 mm thick, flush with the handle's bottom, with a friction of
0.15. Tilted 8 degrees and slid along the plate, the blade's tip gets under the print's edge and its top face meets the
edge 7 mm in. Peeling is not simulated: once the tip has been under the edge for 0.5 s, the print's bond is released.
The basket is NVIDIA's SimReady `misc_basket_plastic_a01` (dataset revision `9d4556bfe808`, CC-BY 4.0), scaled to
0.75. Everything else is modelled in Blender: each box keeps its collider and `looks.py` draws the mesh in its place,
inside the box (a test checks it). To change them, edit `looks/make_looks.py` and run it in Blender 4.2 or later:
`blender -b -P looks/make_looks.py`.

## Run it

On the YAM Brev launchable, from this folder, with the Arena environment active
(`source /data/IsaacLab-Arena/yam.env`):

```bash
# Dream-1 (needs DREAM_API_KEY), 2,400 steps
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=scrape_print_off_bed.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Use the scraper with the black handle to free the green print from the gold build plate inside the printer, and put the print in the blue basket. Leave the plate in the printer and the scraper on the table." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/scrape-print-off-bed-dream

# Scripted demo (both arms), 600 steps
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=scrape_print_off_bed.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Use the scraper with the black handle to free the green print from the gold build plate inside the printer, and put the print in the blue basket. Leave the plate in the printer and the scraper on the table." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/scrape-print-off-bed-demo
```

- The run prints `task_score` at the end. With `-E object_jitter=0` the layout is fixed and no seed is needed.
  Keep it at 0: jitter would move the print and the plate away from their bonds.
- The basket's files (34 MB of textures) download from Hugging Face on the first run, into `~/.cache/simready`.
- The demo: the right arm picks the scraper by the middle of its handle, straight down, lifts it and brings it in
  front of the printer, tilts it 8 degrees, tip down, sets the tip on the plate's front edge and slides it along the
  plate under the print's edge. The command runs 2.2 cm past the edge, so the position-controlled arm leans on the
  blade. After 0.5 s under the edge the bond lets go and the blade pushes the print 1 cm along. The right arm pulls
  the scraper out of the printer and lays it back where it was. The left arm waits in front of the door's left half,
  then takes the print from the front, its hand tilted down 15 degrees with the fingers either side of it, draws it
  out of the printer, lowers it into the basket and lets go 2 cm over its floor. It reaches 1.0 and ends on success
  at step 500 (25 s at 20 Hz). A trial takes about 2 minutes on an L40S, including startup.
- Dream-1 has not been run on this version yet. 2,400 steps take about 6 minutes, including startup.
- The same demo with the blade stopping 3 cm short of the print (`PUSH_PAST = -0.03`) scores 0: the left arm grips the
  still-stuck print and lifts it, the plate's magnets give at 5 N and the plate comes up with the print (2 cm or
  more), which scores 0 for the rest of the trial.
- Videos: `inspect-robots video ~/yam-eval/scrape-print-off-bed-demo/<log>.json` writes one MP4 per camera next to
  the frames.
- Offline checks: `python -m pytest -q test_scrape_print_off_bed.py`.

## Scoring

All positions are centres, in metres, in the station frame (x forward, y to the robot's left, from the midpoint of
the arm bases), and heights are above the table top.

- **Under the edge:** a point of the blade's front edge 3 mm or more inside the print's 6 x 6 cm outline and less than
  4 mm above the print's bottom (so under its edge) for 0.5 s in a row. The scorer remembers it for the rest of the
  trial.
- **Plate lifted:** the plate's highest point 2 cm or more above where it started. The trial scores 0 from then on.
- **0** if the print or the scraper has fallen off the table (centre more than 2 cm below the top), or the state is
  not finite.
- **1.0 (success, held 1 s):** the print's centre within 5.5 cm along x and 8 cm along y of the basket's centre and
  below its rim (8.6 cm), the print still (under 1 cm/s and 0.5 rad/s), the blade has been under its edge, the
  plate's centre within 5 mm of where it started, and the scraper lying still on the table (its lowest point within
  5 mm of the table top, its centre over neither the printer's footprint nor the basket).
- **0.5:** the print off its place: its centre 5 cm or more from where it started on the plate (held, set down, or
  in the basket with the rest of success not met).
- **0.25:** the blade's tip has been under the print's edge for 0.5 s.
- **0** otherwise, including doing nothing, and pulling the print up without the scraper (the plate comes up).

In a real trial, a tape measure gives the plate's shift and lift and the print's place, and the video shows the
blade under the print's edge.
