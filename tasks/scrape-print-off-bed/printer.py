"""printer: the enclosed desktop 3D printer's geometry, shared by the task file, its tests and looks/make_looks.py.

A 36 x 35 x 40 cm FDM printer, its front facing the robot (-x), centred on the table's midline. Positions here are in
the station frame (x forward, y to the robot's left), with heights ``h`` above the table top. Its colliders are boxes
(``colliders.Part``) in the printer box's own frame: the base, the bed, the side and back walls, the roof, the front
frame round the door opening, and the gantry, so an arm cannot pass through it. The door, the print head, the plate
and the prints are boxes of their own. Pure Python.
"""

from __future__ import annotations

from colliders import Part

# -- the printer's box -------------------------------------------------------------------------------------------------
CENTRE = (0.49, 0.0, 0.20)  # x, y, h
SIZE = (0.36, 0.35, 0.40)
FRONT = 0.38  # the enclosure's front face, where the door shuts
BASE_TOP = 0.08  # the base's top: the chamber floor and the ledge in front of the door
LEDGE = (0.335, 0.38)  # x: the flat top of the base in front of the door
OPENING = (0.155, 0.36)  # the door opening: |y| below this, from the base's top up to this height
BED_X = (0.395, 0.635)
BED_Y = 0.12  # half width
BED_TOP = 0.10  # the heated bed's top (the plate lies on it)
PLATE_SIZE = (0.24, 0.24, 0.001)
PLATE_XY = (0.515, 0.0)
PLATE_TOP = BED_TOP + PLATE_SIZE[2]

# -- the door: tinted acrylic on a vertical hinge at its left (+y) edge, a bar handle near its right edge ---------------
DOOR_CENTRE = (0.3555, 0.0, 0.229)
DOOR_SIZE = (0.047, 0.326, 0.288)
HINGE = (0.379, 0.170)  # x, y of the vertical hinge axis, on the front of the left post
HANDLE = (0.338, -0.136)  # x, y of the handle bar's axis, door shut
HANDLE_H = (0.17, 0.29)  # the bar's ends

# -- the print head, parked at the front of its gantry, left of the middle ---------------------------------------------
HEAD_CENTRE = (0.457, 0.07, 0.27)
HEAD_SIZE = (0.058, 0.06, 0.08)


def _part(name, x0, x1, y0, y1, h0, h1, centre=CENTRE, friction=None, combine="average"):
    c = ((x0 + x1) / 2 - centre[0], (y0 + y1) / 2 - centre[1], (h0 + h1) / 2 - centre[2])
    size = (x1 - x0, y1 - y0, h1 - h0)
    if friction is None:
        return Part(name, c, size)
    return Part(name, c, size, friction, combine)


def printer_parts(bed_top: float = BED_TOP, feet_friction: float | None = None) -> list[Part]:
    """The printer's colliders in its own frame. ``bed_top``: the bed's collider top (raise it by the plate's
    thickness when the plate is drawn on the bed and not a body of its own). ``feet_friction``: the four feet's
    friction against the table (the lower of the two wins), or None to stand on the base itself."""
    parts = []
    if feet_friction is None:
        parts.append(_part("base", 0.31, 0.67, -0.175, 0.175, 0.0, BASE_TOP))
    else:
        for i, x in enumerate((0.335, 0.645)):
            for j, y in enumerate((-0.15, 0.15)):
                parts.append(_part(f"foot{i}{j}", x - 0.015, x + 0.015, y - 0.015, y + 0.015, 0.0, 0.008,
                                   friction=(feet_friction, feet_friction), combine="min"))
        parts.append(_part("base", 0.31, 0.67, -0.175, 0.175, 0.008, BASE_TOP))
    w = OPENING[0]
    parts += [
        _part("bed", BED_X[0], BED_X[1], -BED_Y, BED_Y, BASE_TOP, bed_top),
        _part("wall_left", FRONT, 0.67, 0.165, 0.175, BASE_TOP, 0.40),
        _part("wall_right", FRONT, 0.67, -0.175, -0.165, BASE_TOP, 0.40),
        _part("back", 0.655, 0.67, -0.165, 0.165, BASE_TOP, 0.40),
        _part("roof", FRONT, 0.67, -0.165, 0.165, 0.385, 0.40),
        _part("post_left", FRONT, FRONT + 0.015, w, 0.165, BASE_TOP, 0.385),
        _part("post_right", FRONT, FRONT + 0.015, -0.165, -w, BASE_TOP, 0.385),
        _part("beam", FRONT, FRONT + 0.015, -w, w, OPENING[1], 0.385),
        _part("rails", 0.40, 0.65, -0.155, 0.155, 0.322, 0.344),
        _part("x_beam", 0.488, 0.508, -0.143, 0.143, 0.318, 0.336),
    ]
    return parts


def door_parts() -> list[Part]:
    """The door's colliders in its own frame (door shut): the panel, the bar handle and its two standoffs."""
    c = DOOR_CENTRE
    hx, hy = HANDLE
    return [
        _part("panel", 0.372, 0.379, -0.163, 0.163, 0.085, 0.373, centre=c),
        _part("bar", hx - 0.006, hx + 0.006, hy - 0.006, hy + 0.006, HANDLE_H[0], HANDLE_H[1], centre=c),
        _part("standoff_low", hx + 0.006, 0.372, hy - 0.005, hy + 0.005, 0.176, 0.186, centre=c),
        _part("standoff_high", hx + 0.006, 0.372, hy - 0.005, hy + 0.005, 0.274, 0.284, centre=c),
    ]


def inside_footprint(x: float, y: float, margin: float = 0.0) -> bool:
    """Whether (x, y) is over the printer's footprint."""
    return (CENTRE[0] - SIZE[0] / 2 - margin < x < CENTRE[0] + SIZE[0] / 2 + margin
            and abs(y - CENTRE[1]) < SIZE[1] / 2 + margin)
