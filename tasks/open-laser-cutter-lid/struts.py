"""struts: the real mechanism that gives the lid the torque the simulation applies (README, "The struts").

The simulation does not model struts as bodies: it puts a torque about the hinge on the lid, from its angle and speed
(``strut_command`` in the task file). This module is the equivalent mechanism, in dimensions you can build and measure:
two gas struts, one each side, each pushing a roller on a lobe that turns with the lid. The lobe's profile is solved
from the simulation's torque curve and the struts' own force curve, so the torque they give is the simulation's by
construction. The tests integrate the profile back and compare.

Pure Python (no NumPy, no yam_arena). Lengths in metres, forces in N, angles in degrees, torques in N m. Everything is
a design choice for the equivalent mechanism except the torque and the damping, which come from the task file.

How it works. The roller sits on the radial line through the hinge axis, at distance ``rho`` from it, and pushes
inward on the lobe, along that line. By virtual work the lobe turns with torque ``force * -d(rho)/d(angle)``: no
torque while ``rho`` is constant (the lobe's dwell, a circle round the hinge pin, below 30 degrees), and a torque
that is the force times the profile's slope once ``rho`` falls. The strut extends as ``rho`` falls.
"""

from __future__ import annotations

import math
from typing import Callable

N_STRUTS = 2  # one each side of the exhaust duct
SIDE_V = 0.20  # m: each strut's distance from the machine's centre line (v = +0.20 and -0.20)
AXIS_TILT = 15.0  # degrees: the radial line leans from straight down toward the back of the machine

FORCE_EXTENDED = 250.0  # N: one strut's force when fully extended (its nominal rating, F1)
PROGRESSION = 1.3  # force fully compressed / force fully extended (F2 / F1), rising linearly with the compression
STROKE = 0.040  # m: the rod's travel
MIN_LENGTH = 0.090  # m: pivot of the body end to the roller axle at the end of the rod, fully compressed
START_EXTENSION = 0.003  # m: how far the rod is out with the lid shut (it is nearly fully compressed)
DWELL_RADIUS = 0.085  # m: the roller axle's distance from the hinge axis while the lid is shut
ROLLER_RADIUS = 0.010  # m: the lobe's surface lies this much inside the roller axle's path (a 20 mm roller)
DWELL_END = 30.0  # degrees: the lobe's profile starts to fall here (the struts begin to give torque)
STEP = 0.01  # degrees: integration step for the profile

# Body-end pivot of each strut: on the radial line, this far from the hinge axis. It does not move: the radius of the
# roller axle plus the strut's length is constant.
BODY_PIVOT = DWELL_RADIUS + START_EXTENSION + MIN_LENGTH


def force(extension: float) -> float:
    """One strut's force (N) with its rod ``extension`` m out of its body: F1 fully extended, F1 * 1.3 fully
    compressed, linear between."""
    return FORCE_EXTENDED * (1.0 + (PROGRESSION - 1.0) * (STROKE - extension) / STROKE)


def profile(torque: Callable[[float], float], step: float = STEP):
    """Integrate the lobe's profile for the total torque ``torque(deg)`` (both struts, opening) from shut to the
    stop. Returns rows of (deg, rho, extension, force of one strut, lever) with lever = -d(rho)/d(angle) in m (the
    strut's force times it, times two, is the torque). The roller axle's radius ``rho`` falls as the lid opens."""
    rows, rho, ext, deg = [], DWELL_RADIUS, START_EXTENSION, 0.0
    n = int(round(80.0 / step))
    for _ in range(n + 1):
        lever = torque(deg) / (N_STRUTS * force(ext)) if deg > DWELL_END else 0.0
        rows.append((deg, rho, ext, force(ext), lever))
        mid = deg + step / 2  # midpoint rule: a second-order step
        ext_mid = ext + lever * math.radians(step) / 2
        lever_mid = torque(mid) / (N_STRUTS * force(ext_mid)) if mid > DWELL_END else 0.0
        drho = lever_mid * math.radians(step)
        rho, ext, deg = rho - drho, ext + drho, deg + step
    return rows


def at(rows, deg: float):
    """The row of :func:`profile` nearest ``deg``."""
    return min(rows, key=lambda r: abs(r[0] - deg))


def torque_from_profile(rows, deg: float) -> float:
    """The total torque the two struts give at ``deg``, read back from the profile alone: the force at the roller
    times the profile's slope, times two. A central difference over 0.1 degree."""
    h = round(0.1 / (rows[1][0] - rows[0][0]))
    i = min(max(rows.index(at(rows, deg)), h), len(rows) - 1 - h)
    slope = (rows[i - h][1] - rows[i + h][1]) / math.radians(rows[i + h][0] - rows[i - h][0])
    return N_STRUTS * rows[i][3] * slope


def pressure_angle(rows, deg: float) -> float:
    """The angle in degrees between the roller's push and the lobe's surface normal there: atan(slope / radius)."""
    r = at(rows, deg)
    return math.degrees(math.atan2(r[4], r[1]))


def body_pivot_point(hinge_u: float, hinge_w: float):
    """The body end's pivot in the machine frame (u forward, w up), for the radial line leaning back by AXIS_TILT."""
    t = math.radians(AXIS_TILT)
    return (hinge_u + BODY_PIVOT * math.sin(t), hinge_w - BODY_PIVOT * math.cos(t))


def roller_point(rho: float, hinge_u: float, hinge_w: float):
    """The roller axle's point in the machine frame at radius ``rho`` along the same line."""
    t = math.radians(AXIS_TILT)
    return (hinge_u + rho * math.sin(t), hinge_w - rho * math.cos(t))


def lobe_extent(rows):
    """The lobe's surface with the lid shut, as (greatest u behind the hinge axis in m, greatest height above the
    hinge axis in m). Its points are at radius ``rho - ROLLER_RADIUS`` and at the angle AXIS_TILT + deg from straight
    down toward the back, for deg from 0 to the stop."""
    pts = [((r[1] - ROLLER_RADIUS) * math.sin(math.radians(AXIS_TILT + r[0])),
            -(r[1] - ROLLER_RADIUS) * math.cos(math.radians(AXIS_TILT + r[0]))) for r in rows]
    return max(p[0] for p in pts), max(p[1] for p in pts)


def damping_at_roller(deg: float, rows, hinge_damping: float) -> float:
    """The viscous damping (N s/m) each strut would need at its rod to give the lid ``hinge_damping`` N m s/rad
    about the hinge at ``deg`` degrees: the rod moves at lever * (angular speed), so the torque is
    N * c * lever**2 * (angular speed)."""
    lever = at(rows, deg)[4]
    return hinge_damping / (N_STRUTS * lever * lever)
