"""printer: the enclosed desktop 3D printer's geometry, with its side spool holder and its roof feeder, shared by the
task file, its tests and looks/make_looks.py.

The printer of the scrape-print-off-bed and open-printer-door tasks, in the scraping task's place: a 36 x 35 x 40 cm
FDM printer, its front facing the robot (-x), centred on the table's midline, its clear door shut. Positions here are
in the station frame (x forward, y to the robot's left), with heights ``h`` above the table top. Its colliders are
boxes (``colliders.Part``) in the printer box's own frame: the base, the bed, the side and back walls, the roof, the
front frame round the door opening, and the gantry, so an arm cannot pass through it.

Two parts are added to that printer, each a fixed box of its own:

- the spool holder on its left wall (+y): a bracket plate on the wall, a 45 mm steel rod sticking out 13 cm to the
  left at mid height, with a 3 cm stop collar at its root and a chamfered nose. A spool hangs on it, its axle along y.
- the filament feeder on its roof, at the back right: a black feeder on a bracket, its brass inlet facing left (+y),
  toward the spool's side, 7.5 cm above the roof. Filament pushed into it runs along -y.

Pure Python.
"""

from __future__ import annotations

from colliders import Part

# -- the printer's box -------------------------------------------------------------------------------------------------
CENTRE = (0.49, 0.0, 0.20)  # x, y, h
SIZE = (0.36, 0.35, 0.40)
FRONT = 0.38  # the enclosure's front face, where the door shuts
BASE_TOP = 0.08  # the base's top: the chamber floor and the ledge in front of the door
OPENING = (0.155, 0.36)  # the door opening: |y| below this, from the base's top up to this height
BED_X = (0.395, 0.635)
BED_Y = 0.12  # half width
BED_TOP = 0.10  # the heated bed's top (the plate lies on it)
PLATE_SIZE = (0.24, 0.24, 0.001)
PLATE_XY = (0.515, 0.0)
PLATE_TOP = BED_TOP + PLATE_SIZE[2]
ROOF = 0.40  # the roof's top
WALL_Y = 0.175  # the side walls' outsides (+-)

# -- the door, shut, and the print head parked at the front of its gantry ------------------------------------------
DOOR_CENTRE = (0.3555, 0.0, 0.229)
DOOR_SIZE = (0.047, 0.326, 0.288)
HANDLE = (0.338, -0.136)  # x, y of the handle bar's axis
HANDLE_H = (0.17, 0.29)  # the bar's ends
HEAD_CENTRE = (0.457, 0.07, 0.27)
HEAD_SIZE = (0.058, 0.06, 0.08)

# -- the spool holder on the left wall -------------------------------------------------------------------------------
ROD_X, ROD_H = 0.48, 0.22  # the rod's axis
ROD_R = 0.0225  # 45 mm across
BRACKET_T = 0.006  # the bracket plate on the wall, from WALL_Y out
BRACKET = (0.10, 0.14)  # its width (along x) and height
COLLAR_R, COLLAR_T = 0.035, 0.03  # the stop collar at the rod's root: a standoff hub, so the spool
# hangs 3.8 cm off the wall and a finger fits between them
STOP_Y = WALL_Y + BRACKET_T + COLLAR_T  # 0.211: the collar's face, the stop
ROD_TIP_Y = 0.31  # the rod's free end
NOSE = (0.02, 0.014)  # its last 2 cm narrow to a 28 mm nose
HOLDER_LO = (ROD_X - BRACKET[0] / 2, WALL_Y, ROD_H - BRACKET[1] / 2)
HOLDER_HI = (ROD_X + BRACKET[0] / 2, ROD_TIP_Y, ROD_H + BRACKET[1] / 2)
HOLDER_CENTRE = tuple((a + b) / 2 for a, b in zip(HOLDER_LO, HOLDER_HI))
HOLDER_SIZE = tuple(b - a for a, b in zip(HOLDER_LO, HOLDER_HI))

# -- the spool: 20 cm across, 7 cm wide, a 52 mm bore, wound with filament to 16 cm across ---------------------------
SPOOL_D, SPOOL_W = 0.20, 0.07
SPOOL_R = SPOOL_D / 2
FLANGE_T = 0.006
WIND_R = 0.08  # the wound filament's surface
BORE_R = 0.026
HUB_R = 0.031  # the hub's outside, round the bore
SPOOL_Y = STOP_Y + 0.002 + SPOOL_W / 2  # 0.248: its middle on the holder, 2 mm off the collar
SPOOL_SIZE = (SPOOL_W, SPOOL_D, SPOOL_D)  # its box: along the axle, across, tall

# -- the feeder on the roof, at the back right ------------------------------------------------------------------------
INLET_X = 0.545  # the inlet's axis, x
INLET_H = 0.475  # and its height: 7.5 cm above the roof
MOUTH_Y = -0.13  # the inlet's mouth: the feeder's face toward the spool's side
FEEDER_W = 0.10  # along x
FEEDER_LO = (INLET_X - FEEDER_W / 2, -WALL_Y, ROOF)
FEEDER_HI = (INLET_X + FEEDER_W / 2, MOUTH_Y, 0.52)
FEEDER_CENTRE = tuple((a + b) / 2 for a, b in zip(FEEDER_LO, FEEDER_HI))
FEEDER_SIZE = (FEEDER_HI[1] - FEEDER_LO[1], FEEDER_W, FEEDER_HI[2] - FEEDER_LO[2])  # along the inlet, across, tall
FUNNEL, BORE = 0.006, 0.0024  # m: the funnel's mouth and the bore, square
FUNNEL_DEPTH, BORE_DEPTH = 0.005, 0.03


def _part(name, x0, x1, y0, y1, h0, h1, centre=CENTRE, friction=None, combine="average"):
    """A box collider from its station-frame extents, in the frame of the box centred at ``centre``."""
    c = ((x0 + x1) / 2 - centre[0], (y0 + y1) / 2 - centre[1], (h0 + h1) / 2 - centre[2])
    size = (x1 - x0, y1 - y0, h1 - h0)
    if friction is None:
        return Part(name, c, size)
    return Part(name, c, size, friction, combine)


def printer_parts(bed_top: float = BED_TOP) -> list[Part]:
    """The printer's colliders in its own frame. ``bed_top``: the bed's collider top (raise it by the plate's
    thickness when the plate is drawn on the bed and not a body of its own)."""
    w = OPENING[0]
    return [
        _part("base", 0.31, 0.67, -0.175, 0.175, 0.0, BASE_TOP),
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


def door_parts() -> list[Part]:
    """The door's colliders in its own frame (shut): the panel, the bar handle and its two standoffs."""
    c = DOOR_CENTRE
    hx, hy = HANDLE
    return [
        _part("panel", 0.372, 0.379, -0.163, 0.163, 0.085, 0.373, centre=c),
        _part("bar", hx - 0.006, hx + 0.006, hy - 0.006, hy + 0.006, HANDLE_H[0], HANDLE_H[1], centre=c),
        _part("standoff_low", hx + 0.006, 0.372, hy - 0.005, hy + 0.005, 0.176, 0.186, centre=c),
        _part("standoff_high", hx + 0.006, 0.372, hy - 0.005, hy + 0.005, 0.274, 0.284, centre=c),
    ]


def bracket_part() -> Part:
    """The holder's bracket plate on the wall, in the holder box's frame."""
    x0, x1 = ROD_X - BRACKET[0] / 2, ROD_X + BRACKET[0] / 2
    h0, h1 = ROD_H - BRACKET[1] / 2, ROD_H + BRACKET[1] / 2
    return _part("bracket", x0, x1, WALL_Y, WALL_Y + BRACKET_T, h0, h1, centre=HOLDER_CENTRE)


def inside_footprint(x: float, y: float, margin: float = 0.0) -> bool:
    """Whether (x, y) is over the printer's footprint."""
    return (CENTRE[0] - SIZE[0] / 2 - margin < x < CENTRE[0] + SIZE[0] / 2 + margin
            and abs(y - CENTRE[1]) < SIZE[1] / 2 + margin)
