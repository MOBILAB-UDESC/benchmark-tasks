"""filament: a flexible filament made of short rigid links on springy joints, anchored to a spool that turns on its
holder's axle.

yam_arena's task files spawn solid primitives with a yaw. This module adds:

- :class:`Link`: a free box, long along its own x axis, with a full starting orientation (``quat``, x y z w) in
  place of the yaw alone, its inertia raised to ``inertia`` (kg m^2) on every axis and its turning damped by
  ``spin_damping``. :func:`lay` lays ``n`` links end to end along a centreline, link 0 at the free end, each link's
  +x toward the free end.
- :class:`Spool`: a free box on a revolute joint about its own x axis (the axle) to the bench, damped by
  ``damping`` (N m s/rad, the bearing's drag), that comes off its axle when pulled harder than ``hold`` newtons.
  It collides as a cylinder for the wound filament and two thin flanges, its plain cuboid hidden.
- :class:`Joint`: a PhysX D6 joint between two bodies: the three translations locked and the three rotations free,
  each pulled back to the laid shape by a drive of ``stiffness`` (N m/rad) and ``damping`` (N m s/rad). The joint's
  frames are the laid shape, so that shape is its rest: :func:`bends` joins consecutive links, :func:`anchor` joins
  the last link's back end to a point on the spool. A joint with a finite ``hold`` (N) breaks when pulled harder.

:func:`register` hooks them in: the links get their pose when yam_arena builds them (``bonds.on_object``), and the
joints, colliders and body settings when they spawn (``bonds.on_spawn``). Pure Python at import time: ``pxr`` and
Isaac Lab-Arena are imported inside the hooks.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from bonds import on_object, on_spawn
from yam_arena.task_file import Box

__all__ = ["Joint", "Link", "Spool", "anchor", "bends", "lay", "matrix", "register"]


@dataclass(frozen=True)
class Link(Box):
    """A free box along its own x axis, square across, starting turned by ``quat`` (x y z w; ``yaw`` unused)."""

    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    inertia: float = 1e-6
    spin_damping: float = 0.5

    def _check(self) -> None:
        super()._check()
        if self.mass is None:
            raise ValueError(f"{self.name}: a link must have a mass")
        if not (self.size[1] == self.size[2] and self.size[0] > self.size[1]):
            raise ValueError(f"{self.name}: a link is long along x and square across")
        if abs(float(np.linalg.norm(self.quat)) - 1.0) > 1e-6:
            raise ValueError(f"{self.name}.quat must be a unit quaternion")

    @property
    def length(self) -> float:
        return self.size[0]


@dataclass(frozen=True)
class Spool(Box):
    """A free box on a revolute joint about its own x axis, at its centre, to the bench. It collides as a cylinder
    ``wind_r`` in radius and ``width`` wide less its flanges, and two flanges ``flange_r`` in radius and ``flange_t``
    thick at its faces (``width`` apart, outside to outside)."""

    wind_r: float = 0.085
    flange_r: float = 0.10
    width: float = 0.07
    flange_t: float = 0.003
    damping: float = 0.01
    hold: float = math.inf

    def _check(self) -> None:
        super()._check()
        if self.mass is None:
            raise ValueError(f"{self.name}: the spool turns, so it must have a mass")
        if not 0 < self.wind_r < self.flange_r <= min(self.size[1], self.size[2]) / 2:
            raise ValueError(f"{self.name}: the winding inside the flanges, the flanges inside the box")
        if not 2 * self.flange_t < self.width <= self.size[0]:
            raise ValueError(f"{self.name}: the flanges inside the box's width")


@dataclass(frozen=True)
class Joint:
    """A D6 joint: body 1's frame (``pos1``, ``rot1``) held on body 0's (``pos0``, ``rot0``, in the stage's frame
    when ``body0`` is empty), translations locked, rotations sprung back to it. Rotations are x y z w."""

    name: str
    body0: str
    body1: str
    pos0: tuple[float, float, float]
    rot0: tuple[float, float, float, float]
    pos1: tuple[float, float, float]
    rot1: tuple[float, float, float, float]
    stiffness: float
    damping: float
    hold: float = math.inf


# -- pure geometry ----------------------------------------------------------------------------------------------------


def matrix(q) -> np.ndarray:
    """The rotation matrix of a unit quaternion x y z w."""
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat(m) -> tuple[float, float, float, float]:
    """The unit quaternion x y z w of a rotation matrix."""
    m = np.asarray(m, float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)
    q = np.asarray(q) / np.linalg.norm(q)
    return tuple(float(v) for v in (q if q[3] >= 0 else -q))


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _frame(spec) -> tuple[np.ndarray, np.ndarray]:
    """A body's starting centre and rotation matrix."""
    rot = matrix(spec.quat) if isinstance(spec, Link) else _yaw_matrix(spec.yaw)
    return np.asarray(spec.pos, float), rot


def lay(prefix: str, centreline: Callable[[float], tuple[float, float, float]], n: int, length: float,
        thickness: float, mass: float, color, **link) -> list[Link]:
    """``n`` links ``length`` long, end to end along ``centreline(s)`` (a point ``s`` metres along the filament from
    its free end), link ``i`` from ``s = i * length`` to ``(i + 1) * length``. Each link's x axis points toward the
    free end and its y axis stays level."""
    out = []
    for i in range(n):
        front = np.asarray(centreline(i * length), float)
        back = np.asarray(centreline((i + 1) * length), float)
        x = (front - back) / np.linalg.norm(front - back)
        y = np.cross([0.0, 0.0, 1.0], x)
        if np.linalg.norm(y) < 1e-6:  # a vertical link: its y along the station's y
            y = np.array([0.0, 1.0, 0.0]) - x[1] * x
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        out.append(Link(f"{prefix}{i:02d}", pos=tuple(float(v) for v in (front + back) / 2), color=color,
                        mass=mass, size=(length, thickness, thickness), quat=quat(np.column_stack([x, y, z])),
                        **link))
    return out


def _between(name, a, b, at, stiffness, damping, hold=math.inf) -> Joint:
    """A joint at the point ``at`` (task-file frame), its frame body ``b``'s axes as laid."""
    pa, ra = _frame(a)
    pb, rb = _frame(b)
    at = np.asarray(at, float)
    tup = lambda v: tuple(float(c) for c in v)  # noqa: E731
    return Joint(name, a.name, b.name, tup(ra.T @ (at - pa)), quat(ra.T @ rb), tup(rb.T @ (at - pb)),
                 (0.0, 0.0, 0.0, 1.0), stiffness, damping, hold)


def bends(links: list[Link], stiffness: float, damping: float) -> list[Joint]:
    """A joint at each meeting of two consecutive links, its rest the laid shape."""
    out = []
    for i, (a, b) in enumerate(zip(links, links[1:])):
        pb, rb = _frame(b)
        at = pb + rb @ np.array([b.length / 2, 0.0, 0.0])  # b's front end, which is a's back end
        out.append(_between(f"bend_{i:02d}", a, b, at, stiffness, damping))
    return out


def anchor(spool: Spool, link: Link, stiffness: float, damping: float, hold: float) -> Joint:
    """A joint holding ``link``'s back end where it lies, on the spool (the filament still wound there)."""
    p, r = _frame(link)
    return _between("anchor", spool, link, p - r @ np.array([link.length / 2, 0.0, 0.0]), stiffness, damping, hold)


# -- spawning ---------------------------------------------------------------------------------------------------------

_LINKS: dict[str, Link] = {}
_SPOOLS: dict[str, Spool] = {}
_JOINTS: dict[str, list[Joint]] = {}  # authored when their body 1 spawns
_PHYSX = {"position_iterations": 32, "velocity_iterations": 4, "contact_offset": 0.002}


def register(links: list[Link], joints: list[Joint], spool: Spool, **physx) -> None:
    """Hook the links, their joints and the spool into the scene as it is built. ``physx`` sets the links'
    ``position_iterations``, ``velocity_iterations`` and ``contact_offset``."""
    _PHYSX.update(physx)
    for link in links:
        _LINKS[link.name] = link
    _SPOOLS[spool.name] = spool
    for j in joints:
        _JOINTS.setdefault(j.body1, []).append(j)


@on_object
def _pose(spec, obj) -> None:
    if spec.name not in _LINKS:
        return
    from isaaclab_arena.utils.pose import Pose

    obj.set_initial_pose(Pose(position_xyz=tuple(float(v) for v in spec.pos),
                              rotation_xyzw=tuple(float(v) for v in spec.quat)))


def _gf_quat(q):
    from pxr import Gf

    x, y, z, w = q
    return Gf.Quatf(float(w), float(x), float(y), float(z))


def _deg(per_rad: float) -> float:
    """USD's angular drives take N m per degree (and N m s per degree)."""
    return per_rad * math.pi / 180.0


def _author(prim, j: Joint) -> None:
    from pxr import Gf, UsdPhysics

    stage = prim.GetStage()
    joint = UsdPhysics.Joint.Define(stage, prim.GetPath().AppendChild(j.name))
    if j.body0:
        joint.CreateBody0Rel().SetTargets([prim.GetPath().GetParentPath().AppendChild(j.body0)])
    joint.CreateBody1Rel().SetTargets([prim.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*j.pos0))
    joint.CreateLocalRot0Attr(_gf_quat(j.rot0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*j.pos1))
    joint.CreateLocalRot1Attr(_gf_quat(j.rot1))
    joint.CreateCollisionEnabledAttr(False)
    if math.isfinite(j.hold):
        joint.CreateBreakForceAttr(float(j.hold))
    for axis in ("transX", "transY", "transZ"):  # low above high: locked
        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
        limit.CreateLowAttr(1.0)
        limit.CreateHighAttr(-1.0)
    for axis in ("rotX", "rotY", "rotZ"):
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(_deg(j.stiffness))
        drive.CreateDampingAttr(_deg(j.damping))
        drive.CreateTargetPositionAttr(0.0)


def _enable_ccd(stage) -> None:
    from pxr import PhysxSchema

    scene = stage.GetPrimAtPath("/physicsScene")
    if scene and scene.IsValid():  # per-body CCD needs it switched on for the scene
        PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableCCDAttr(True)


def _build_link(prim, spec: Link) -> None:
    from pxr import Gf, PhysxSchema, Usd, UsdPhysics

    for shape in [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]:
        contact = PhysxSchema.PhysxCollisionAPI.Apply(shape)
        contact.CreateContactOffsetAttr(float(_PHYSX["contact_offset"]))
        contact.CreateRestOffsetAttr(0.0)
    UsdPhysics.MassAPI.Apply(prim).CreateDiagonalInertiaAttr(Gf.Vec3f(*(spec.inertia,) * 3))
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    body.CreateAngularDampingAttr(spec.spin_damping)
    body.CreateEnableCCDAttr(True)
    body.CreateSolverPositionIterationCountAttr(int(_PHYSX["position_iterations"]))
    body.CreateSolverVelocityIterationCountAttr(int(_PHYSX["velocity_iterations"]))
    body.CreateMaxDepenetrationVelocityAttr(0.5)
    _enable_ccd(prim.GetStage())


def _build_spool(prim, spec: Spool) -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = prim.GetStage()
    for p in [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]:
        for api in ("PhysicsCollisionAPI", "PhysxCollisionAPI", "PhysicsMeshCollisionAPI"):
            if api in p.GetAppliedSchemas():
                p.RemoveAppliedSchema(api)
    UsdGeom.Xform.Define(stage, prim.GetPath().AppendChild("colliders"))
    half = spec.width / 2
    parts = [("winding", 0.0, spec.wind_r, spec.width - 2 * spec.flange_t)]
    parts += [(f"flange_{i}", s * (half - spec.flange_t / 2), spec.flange_r, spec.flange_t)
              for i, s in enumerate((-1, 1))]
    for name, x, radius, height in parts:
        cyl = UsdGeom.Cylinder.Define(stage, prim.GetPath().AppendPath(f"colliders/{name}"))
        cyl.CreateAxisAttr("X")
        cyl.CreateRadiusAttr(radius)
        cyl.CreateHeightAttr(height)
        cyl.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.0))
        UsdGeom.Imageable(cyl.GetPrim()).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    # The axle: a revolute joint about the box's x axis, at its centre, to the bench (the stage's frame).
    world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    axle = UsdPhysics.RevoluteJoint.Define(stage, prim.GetPath().AppendChild("axle"))
    axle.CreateBody1Rel().SetTargets([prim.GetPath()])
    axle.CreateLocalPos0Attr(Gf.Vec3f(*world))
    axle.CreateLocalRot0Attr(_gf_quat(quat(_yaw_matrix(spec.yaw))))
    axle.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    axle.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    axle.CreateAxisAttr("X")
    axle.CreateCollisionEnabledAttr(False)
    if math.isfinite(spec.hold):
        axle.CreateBreakForceAttr(float(spec.hold))
    drive = UsdPhysics.DriveAPI.Apply(axle.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(_deg(spec.damping))


@on_spawn
def _spawned(name: str, prim) -> None:
    if name in _LINKS:
        _build_link(prim, _LINKS[name])
    if name in _SPOOLS:
        _build_spool(prim, _SPOOLS[name])
    for j in _JOINTS.get(name, ()):
        _author(prim, j)
