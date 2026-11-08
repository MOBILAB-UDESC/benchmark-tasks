"""inlet: an extruder head whose inlet is a funnel and a bore made of box colliders.

yam_arena's task files spawn solid primitives. The head changes its collider when it spawns (through
``bonds.on_spawn``), and keeps its cuboid for the bounds the scorer reads:

- :class:`Head`: a fixed box whose collider is replaced by box colliders around a channel along its own +x axis. The
  channel opens on the box's -x face (the mouth) as a square funnel ``funnel`` wide, narrowing over ``funnel_depth``
  to a square bore ``bore`` wide, which runs ``bore_depth`` further and ends closed. The channel's axis is
  ``inlet_z`` above the box's centre. The walls are slick (``friction``, combined by minimum), like a PTFE-lined
  inlet.

Pure Python at import time: ``pxr`` is imported when a box spawns, inside Isaac Sim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bonds import on_spawn
from yam_arena.task_file import Box

__all__ = ["Head", "head"]

GRIPPY = (1.2, 1.0)  # yam_arena's friction for task objects (constants.GRIPPY_FRICTION)


@dataclass(frozen=True)
class Head(Box):
    """A fixed box with a funnel and a bore along its own +x axis, opening on its -x face."""

    inlet_z: float = 0.0
    funnel: float = 0.006
    bore: float = 0.0024
    funnel_depth: float = 0.005
    bore_depth: float = 0.03
    wall: float = 0.002  # m: the funnel plates' thickness
    friction: float = 0.15

    def _check(self) -> None:
        super()._check()
        if self.mass is not None:
            raise ValueError(f"{self.name}: the head is fixed (no mass)")
        hx, hy, hz = (s / 2 for s in self.size)
        if not 0 < self.bore < self.funnel < min(2 * hy, 2 * (hz - abs(self.inlet_z))):
            raise ValueError(f"{self.name}: the funnel must be wider than the bore and fit in the box")
        if not 0 < self.funnel_depth + self.bore_depth < 2 * hx:
            raise ValueError(f"{self.name}: the channel must end inside the box")

    def parts(self) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float], tuple]]:
        """Each collider as (name, centre, size, rotation as x y z w) in the box's own frame."""
        hx, hy, hz = (s / 2 for s in self.size)
        zc, b, f = self.inlet_z, self.bore / 2, self.funnel / 2
        x0, x1 = -hx, -hx + self.funnel_depth
        x2 = x1 + self.bore_depth
        flat = (0.0, 0.0, 0.0, 1.0)

        def slab(name, lo, hi):
            centre = tuple((a + c) / 2 for a, c in zip(lo, hi))
            return (name, centre, tuple(c - a for a, c in zip(lo, hi)), flat)

        out = []
        for zone, xa, xb, w in (("bore", x1, hx, b), ("funnel", x0, x1, f)):
            out += [
                slab(f"{zone}_top", (xa, -hy, zc + w), (xb, hy, hz)),
                slab(f"{zone}_bottom", (xa, -hy, -hz), (xb, hy, zc - w)),
                slab(f"{zone}_left", (xa, w, zc - w), (xb, hy, zc + w)),
                slab(f"{zone}_right", (xa, -hy, zc - w), (xb, -w, zc + w)),
            ]
        out.append(slab("plug", (x2, -b, zc - b), (hx, b, zc + b)))
        # The funnel: four plates, each with its inner face on the slope from the mouth's edge (f off the axis) to
        # the bore's (b off the axis), and its thickness behind that face.
        depth, drop = self.funnel_depth, f - b
        s = math.hypot(depth, drop)
        for name, n in (("top", (0, 0, 1)), ("bottom", (0, 0, -1)), ("left", (0, 1, 0)), ("right", (0, -1, 0))):
            u = (depth / s, -drop / s * n[1], -drop / s * n[2])  # down the slope, toward the bore
            w = (drop / s, depth / s * n[1], depth / s * n[2])  # out of the channel
            v = _cross(w, u)
            mid = ((x0 + x1) / 2, (f + b) / 2 * n[1], zc + (f + b) / 2 * n[2])
            centre = tuple(m + self.wall / 2 * c for m, c in zip(mid, w))
            out.append((f"funnel_plate_{name}", centre, (s, 2 * f, self.wall), _quat_from_axes(u, v, w)))
        return out

    def mouth_local(self) -> tuple[float, float, float]:
        """The centre of the funnel's mouth in the box's own frame."""
        return (-self.size[0] / 2, 0.0, self.inlet_z)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _quat_from_axes(u, v, w) -> tuple[float, float, float, float]:
    """The rotation (x y z w) whose matrix has columns u, v, w."""
    m = [[u[0], v[0], w[0]], [u[1], v[1], w[1]], [u[2], v[2], w[2]]]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return (0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return ((m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return ((m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s)


# -- spawning --------------------------------------------------------------------------------------------------------

_HEADS: dict[str, Head] = {}


def _material(stage, path, friction, combine: str):
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    static, dynamic = friction if isinstance(friction, tuple) else (friction, friction)
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr(combine)
    return material


def _bind(prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")


def _strip_colliders(prim) -> list:
    """Remove the collider from every shape under ``prim`` (the spawned cuboid), and return those shapes."""
    from pxr import Usd, UsdPhysics

    shapes = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
    for p in shapes:
        for api in ("PhysicsCollisionAPI", "PhysxCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"):
            if api in p.GetAppliedSchemas():
                p.RemoveAppliedSchema(api)
    return shapes


def _collide(prim, offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    contact.CreateContactOffsetAttr(offset)
    contact.CreateRestOffsetAttr(0.0)


def _build_head(prim, spec: Head) -> None:
    from pxr import Gf, UsdGeom

    stage = prim.GetStage()
    _strip_colliders(prim)
    material = _material(stage, prim.GetPath().AppendChild("inlet_material"), spec.friction, "min")
    UsdGeom.Xform.Define(stage, prim.GetPath().AppendChild("walls"))
    for name, centre, size, quat in spec.parts():
        cube = UsdGeom.Cube.Define(stage, prim.GetPath().AppendPath(f"walls/{name}"))
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
        x, y, z, w = quat
        cube.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
        _collide(cube.GetPrim(), 0.002)
        _bind(cube.GetPrim(), material)


@on_spawn
def _spawned(name: str, prim) -> None:
    if name in _HEADS:
        _build_head(prim, _HEADS[name])


def head(spec: Head) -> Head:
    """Register ``spec`` so its inlet's colliders are built when it spawns, and return it (for ``OBJECTS``)."""
    _HEADS[spec.name] = spec
    return spec

