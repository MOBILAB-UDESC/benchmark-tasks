"""seat: a task-file box seated in a hole on a damped joint, to sink under a push and tilt under a torque.

yam_arena's task files spawn plain primitives. :class:`Seated` is a ``Box`` with a mass that gets one PhysX D6 joint to
the bench when it spawns, with its centre at the joint's origin:

- along ``axis`` ("X", "Y" or "Z", the box's own, which are the world's at ``yaw`` 0) it moves from where it spawns
  (``lower`` = 0) to ``travel`` metres further, and only while something pushes it: a velocity damper of ``damping``
  N s/m resists, with no spring, so it sinks at ``force / damping`` metres per second and stays where it is left. Heat-set
  plastic under a heated insert: it flows while the insert is pushed, and sets when the push stops.
- it turns about its centre about the two other axes, up to ``tilt`` degrees, against a damper of ``tilt_damping``
  N m s/rad, with no spring: a torque off its axis cocks it, and it stays cocked.
- the other four degrees of freedom (the two sideways, and the turn about its own axis) are locked.

The joint does not stop the box colliding with what it sits in: the task file's colliders leave room for it.

Importing this module hooks ``bonds.on_object`` and ``bonds.on_spawn`` (and so needs ``bonds.py`` beside it). Pure
Python at import time: everything that needs Isaac Sim is imported when the box spawns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bonds import on_object, on_spawn
from yam_arena.task_file import Box

__all__ = ["Seated", "seated"]


@dataclass(frozen=True)
class Seated(Box):
    """A free box on a damped sink-and-tilt joint (see the module's docstring). ``inertia`` (kg m^2) raises its
    inertia about each axis, because a few grams of brass would otherwise spin faster than the solver follows."""

    axis: str = "X"
    travel: float = 0.01
    damping: float = 250.0
    tilt: float = 10.0
    tilt_damping: float = 1e-3
    inertia: float = 1e-7

    def _check(self) -> None:
        super()._check()
        if self.mass is None:
            raise ValueError(f"{self.name}: a seated box must have a mass")
        if self.axis not in ("X", "Y", "Z"):
            raise ValueError(f"{self.name}.axis must be X, Y or Z")
        if not (self.travel > 0 and self.damping > 0 and self.tilt_damping > 0 and 0 <= self.tilt < 90):
            raise ValueError(f"{self.name}: travel and both dampings must be positive, tilt in [0, 90) degrees")
        if self.yaw != 0.0:
            raise ValueError(f"{self.name}: a seated box is not turned (yaw 0)")


_SEATS: dict[str, Seated] = {}


def seated(name: str) -> Seated:
    """The :class:`Seated` box ``name`` registered when yam_arena built it (inside the simulation)."""
    return _SEATS[name]


@on_object
def _remember(spec, obj) -> None:
    if isinstance(spec, Seated):
        _SEATS[spec.name] = spec


def _deg(per_rad: float) -> float:
    """USD's angular drives take N m per degree (and N m s per degree)."""
    return per_rad * math.pi / 180.0


@on_spawn
def _seat(name: str, prim) -> None:
    spec = _SEATS.get(name)
    if spec is None:
        return
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = prim.GetStage()
    world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    joint = UsdPhysics.Joint.Define(stage, prim.GetPath().AppendChild("seat"))
    joint.CreateBody1Rel().SetTargets([prim.GetPath()])  # body 0 is the world: its frame is in stage coordinates
    joint.CreateLocalPos0Attr(Gf.Vec3f(float(world[0]), float(world[1]), float(world[2])))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateCollisionEnabledAttr(True)
    sink = "trans" + spec.axis
    others = [a for a in "XYZ" if a != spec.axis]
    for dof in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), dof)
        if dof == sink:
            limit.CreateLowAttr(0.0)
            limit.CreateHighAttr(float(spec.travel))
        elif dof.startswith("rot") and dof[3] in others:
            limit.CreateLowAttr(-float(spec.tilt))
            limit.CreateHighAttr(float(spec.tilt))
        else:  # low above high: locked
            limit.CreateLowAttr(1.0)
            limit.CreateHighAttr(-1.0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), sink)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(float(spec.damping))
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    for a in others:
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "rot" + a)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(_deg(spec.tilt_damping))
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
    UsdPhysics.MassAPI.Apply(prim).CreateDiagonalInertiaAttr(Gf.Vec3f(*(float(spec.inertia),) * 3))
    body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    body.CreateEnableCCDAttr(True)
    body.CreateSolverPositionIterationCountAttr(32)
    body.CreateSolverVelocityIterationCountAttr(4)
    scene = stage.GetPrimAtPath("/physicsScene")
    if scene and scene.IsValid():
        PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableCCDAttr(True)
