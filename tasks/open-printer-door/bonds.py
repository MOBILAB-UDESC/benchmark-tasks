"""bonds: task-file boxes held by a breakable bond, sliding on a sprung joint or turning on a hinge with a catch and detents.

yam_arena's task files spawn plain primitives. This module adds three boxes that get a PhysX joint when they spawn:

- :class:`Bonded`: a ``Box`` with a mass that starts fixed to the object named ``to`` (or to the bench, when ``to``
  is empty or names a fixed object) by a fixed joint whose ``physics:breakForce`` is ``hold`` newtons and
  ``physics:breakTorque`` is ``twist`` newton metres. Pull or push harder than ``hold``, or turn it harder than
  ``twist``, and the bond breaks for the rest of the trial. The joint sits at the centre of the box's bottom face.
  Old thermal paste, a glued-on label, a card locked in its slot. :func:`release` undoes a bond on cue (a latch that
  frees the card) by disabling its joint, and every reset enables the released bonds again.
- :class:`Sliding`: a ``Box`` with a mass on a prismatic joint along one of its own axes, between ``lower`` and
  ``upper`` (metres from where it spawns, which is its rest), pushed back to rest by a spring of ``rate`` N/m with
  ``damping`` N s/m. A button, a latch, a drawer. The other side of the joint is the object ``to``, or the bench.
- :class:`Hinged`: a ``Box`` with a mass on a revolute joint about one of its own axes through ``hinge`` (a point in
  the box's frame), turning from ``lower`` to ``upper`` degrees from where it spawns, its other side a free object or
  the bench. A breakable catch holds it shut until pulled hard enough, then it swings with a little friction, and
  springs that act only in zones of its angle (a detent that pulls it into its open stop) are applied as torques.
  :func:`hinge_angle` reads the angle relative to the other body. A printer's door, a cabinet door, a lid.

Adapted from the bonds module of the zif-socket-chip task (the catch, the zones and hinges on a free body are new).

Importing this module hooks ``yam_arena.scene.task_object`` when that module is imported (after Isaac Sim starts),
and ``yam_arena.inspect_embodiment.YamArenaEmbodiment`` so a task file can run code after every reset
(:func:`on_reset`) and before every scoring step (:func:`on_step`). Nothing in yam_arena is changed. ``success()``
and ``score()`` get the usual ``ObjectState``. Pure Python at import time: everything that needs Isaac Sim is
imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass

from yam_arena.task_file import Box

__all__ = ["Bonded", "Hinged", "Sliding", "hinge_angle", "hinge_angles", "install", "on_reset", "on_spawn", "on_step",
           "release", "released"]


@dataclass(frozen=True)
class Bonded(Box):
    """A free box bonded by its bottom face to the object ``to`` (a free box: the two move together until the bond
    breaks) or to the bench (``to`` empty, or a fixed object). ``hold`` is the force (N) and ``twist`` the torque
    (N m) that break the bond, ``math.inf`` for unbreakable."""

    to: str = ""
    hold: float = math.inf
    twist: float = math.inf

    def _check(self) -> None:
        super()._check()
        if self.mass is None:
            raise ValueError(f"{self.name}: a bonded box must have a mass")
        if not (self.hold > 0 and self.twist > 0):
            raise ValueError(f"{self.name}: hold and twist must be positive (math.inf for unbreakable)")
        if self.to and not self.to.isidentifier():
            raise ValueError(f"{self.name}.to must name an object")


@dataclass(frozen=True)
class Sliding(Box):
    """A free box on a prismatic joint along its ``axis`` ("X", "Y" or "Z", the box's own, which are the world's at
    ``yaw`` 0), free to move from ``lower`` to ``upper`` metres from where it spawns, and sprung back there with
    ``rate`` N/m and ``damping`` N s/m. The joint's other side is the free object ``to``, or the bench."""

    to: str = ""
    axis: str = "Z"
    lower: float = -0.004
    upper: float = 0.0
    rate: float = 1000.0
    damping: float = 10.0

    def _check(self) -> None:
        super()._check()
        if self.mass is None:
            raise ValueError(f"{self.name}: a sliding box must have a mass")
        if self.axis not in ("X", "Y", "Z"):
            raise ValueError(f"{self.name}.axis must be X, Y or Z")
        if not (self.lower <= 0.0 <= self.upper and self.lower < self.upper):
            raise ValueError(f"{self.name}: lower <= 0 <= upper is where it spawns, and the range must not be empty")
        if self.rate < 0 or self.damping < 0:
            raise ValueError(f"{self.name}: rate and damping must not be negative")
        if self.to and not self.to.isidentifier():
            raise ValueError(f"{self.name}.to must name an object")


@dataclass(frozen=True)
class Hinged(Box):
    """A free box on a revolute joint about its own ``axis`` ("X", "Y" or "Z", the box's own, which are the world's
    at ``yaw`` 0) through ``hinge`` (m, from the box's centre, in its own frame), free to turn from ``lower`` to
    ``upper`` degrees from where it spawns (the right-hand rule about the axis). The joint's other side is the free
    object ``to``, or the bench. The hinge's friction is ``damping`` (N m s/rad). ``catch``: a magnetic catch that
    holds the box at 0 until the torque about the hinge passes ``catch`` N m, then lets go for the rest of the trial
    (a second joint locking only that turn, which PhysX breaks). ``zones`` are (from, to, target, rate, damping):
    angles in degrees, ``rate`` in N m/rad and ``damping`` in N m s/rad. Inside a zone a spring pulls the box toward
    the zone's target (a detent): a torque applied to the box, and the opposite one to the body on the other side,
    every control step. PhysX does not take a joint drive's gains changed during a run, so the zones are torques."""

    to: str = ""
    axis: str = "Z"
    hinge: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lower: float = -90.0
    upper: float = 0.0
    zones: tuple[tuple[float, float, float, float, float], ...] = ()
    damping: float = 0.01
    catch: float = math.inf

    def _check(self) -> None:
        super()._check()
        if self.mass is None:
            raise ValueError(f"{self.name}: a hinged box must have a mass")
        if self.axis not in ("X", "Y", "Z"):
            raise ValueError(f"{self.name}.axis must be X, Y or Z")
        if not (self.lower <= 0.0 <= self.upper and self.lower < self.upper):
            raise ValueError(f"{self.name}: lower <= 0 <= upper is where it spawns, and the range must not be empty")
        for lo, hi, target, rate, damping in self.zones:
            if not (lo < hi and lo <= target <= hi and rate >= 0 and damping >= 0):
                raise ValueError(f"{self.name}: a zone is (from < to, its target inside, rate and damping >= 0)")
        if self.damping < 0 or not self.catch > 0:
            raise ValueError(f"{self.name}: damping must not be negative and catch must be positive")
        if self.to and not self.to.isidentifier():
            raise ValueError(f"{self.name}.to must name an object")

    def spring_at(self, angle: float, speed: float = 0.0) -> float:
        """The zones' torque (N m, about the hinge axis) at ``angle`` degrees turning at ``speed`` rad/s: 0 outside
        every zone."""
        for lo, hi, target, rate, damping in self.zones:
            if lo <= angle <= hi:
                return rate * math.radians(target - angle) - damping * speed
        return 0.0


# -- spawning --------------------------------------------------------------------------------------------------------

_SPECS: dict[str, Bonded | Sliding | Hinged] = {}
_PLACED: dict[str, tuple] = {}  # every task-file shape's (pos, yaw, free), to anchor a joint on it
_PRIMS: dict = {}  # each jointed box's spawned prim, to find its joint at run time
_RELEASED: set[str] = set()
_SPAWN: dict = {}
_SPAWNED: list = []  # callbacks run on every task-file box as it spawns


def _rz(yaw: float, v):
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1], v[2])


def _anchor(joint, prim, spec, anchor) -> None:
    """Author the joint's two frames: body 1 is the spawned box, body 0 the object ``spec.to`` or the world.
    ``anchor`` is the joint's position in the task file's frame."""
    from pxr import Gf, Sdf, Usd, UsdGeom

    world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    origin = world - Gf.Vec3d(*spec.pos)  # the environment's origin: where the task file's frame sits in the stage
    joint.CreateBody1Rel().SetTargets([prim.GetPath()])
    joint.CreateLocalPos1Attr(Gf.Vec3f(*_rz(-spec.yaw, tuple(a - p for a, p in zip(anchor, spec.pos)))))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    other = _PLACED.get(spec.to)
    if other is not None and other[2]:
        pos, yaw, _ = other
        joint.CreateBody0Rel().SetTargets([Sdf.Path(prim.GetPath().GetParentPath().AppendChild(spec.to))])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*_rz(-yaw, tuple(a - p for a, p in zip(anchor, pos)))))
    else:  # the bench: the joint's other side is the world, so its frame is in stage coordinates
        yaw = 0.0
        joint.CreateLocalPos0Attr(Gf.Vec3f(*(Gf.Vec3d(*anchor) + origin)))
    # Frame 1 is the box's own axes, so frame 0 is turned by the box's yaw relative to body 0: the joint holds the
    # box as placed, and a sliding box moves along its own axis.
    turn = (spec.yaw - yaw) / 2
    joint.CreateLocalRot0Attr(Gf.Quatf(math.cos(turn), 0.0, 0.0, math.sin(turn)))


def _bond(prim, spec: Bonded) -> None:
    """The fixed joint under the spawned box, anchored at the centre of its bottom face."""
    from pxr import UsdPhysics

    joint = UsdPhysics.FixedJoint.Define(prim.GetStage(), prim.GetPath().AppendChild("bond"))
    _anchor(joint, prim, spec, (spec.pos[0], spec.pos[1], spec.pos[2] - spec.size[2] / 2))
    joint.CreateCollisionEnabledAttr(True)  # the bonded faces keep touching after the bond breaks
    joint.CreateBreakForceAttr(float(spec.hold))
    joint.CreateBreakTorqueAttr(float(spec.twist))


def _slide(prim, spec: Sliding) -> None:
    """The prismatic joint under the spawned box, anchored at its centre, with its limits and spring."""
    from pxr import UsdPhysics

    joint = UsdPhysics.PrismaticJoint.Define(prim.GetStage(), prim.GetPath().AppendChild("slide"))
    _anchor(joint, prim, spec, spec.pos)
    joint.CreateAxisAttr(spec.axis)
    joint.CreateLowerLimitAttr(float(spec.lower))
    joint.CreateUpperLimitAttr(float(spec.upper))
    joint.CreateCollisionEnabledAttr(True)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(spec.rate))
    drive.CreateDampingAttr(float(spec.damping))
    drive.CreateTargetPositionAttr(0.0)


_AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def _revolute(prim, spec: Hinged, name: str):
    from pxr import UsdPhysics

    joint = UsdPhysics.RevoluteJoint.Define(prim.GetStage(), prim.GetPath().AppendChild(name))
    h = _rz(spec.yaw, spec.hinge)
    _anchor(joint, prim, spec, tuple(p + d for p, d in zip(spec.pos, h)))
    joint.CreateAxisAttr(spec.axis)
    joint.CreateCollisionEnabledAttr(True)
    return joint


def _hinge(prim, spec: Hinged) -> None:
    """The revolute joint of a hinged box, anchored at its hinge, with its limits and its friction, and its catch: a
    second joint at the same place that locks only the turn about the hinge and breaks at ``spec.catch`` N m."""
    from pxr import UsdPhysics

    joint = _revolute(prim, spec, "hinge")
    joint.CreateLowerLimitAttr(float(spec.lower))
    joint.CreateUpperLimitAttr(float(spec.upper))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(float(spec.damping) * math.pi / 180.0)  # USD angular drives: N m s per degree
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    if math.isfinite(spec.catch):
        catch = UsdPhysics.Joint.Define(prim.GetStage(), prim.GetPath().AppendChild("catch"))
        h = _rz(spec.yaw, spec.hinge)
        _anchor(catch, prim, spec, tuple(p + d for p, d in zip(spec.pos, h)))
        limit = UsdPhysics.LimitAPI.Apply(catch.GetPrim(), "rot" + spec.axis)
        limit.CreateLowAttr(1.0)  # low above high: that turn is locked, every other one free
        limit.CreateHighAttr(-1.0)
        catch.CreateCollisionEnabledAttr(True)
        catch.CreateBreakTorqueAttr(float(spec.catch))


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def hinge_angle(spec: Hinged, quat_xyzw, other_xyzw=(0.0, 0.0, 0.0, 1.0), other_yaw: float = 0.0) -> float:
    """The hinged box's angle (degrees) from where it spawned: its turn about its hinge axis relative to the body on
    the joint's other side (orientation ``other_xyzw``, spawned at ``other_yaw``; the bench by default)."""
    ox, oy, oz, ow = (float(v) for v in other_xyzw)
    rel = _quat_mul((-ox, -oy, -oz, ow), tuple(float(v) for v in quat_xyzw))  # the box in the other's frame
    turn = (spec.yaw - other_yaw) / 2  # remove the spawn turn between them
    rel = _quat_mul((0.0, 0.0, -math.sin(turn), math.cos(turn)), rel)
    qx, qy, qz, qw = rel
    k = {"X": qx, "Y": qy, "Z": qz}[spec.axis]
    if qw < 0:
        qw, k = -qw, -k
    return math.degrees(2.0 * math.atan2(k, qw))


def hinge_angles(env) -> dict[str, float]:
    """Every hinged box's angle now (degrees), keyed by name."""
    hinges = [s for s in _SPECS.values() if isinstance(s, Hinged) and s.name in _PRIMS]
    if not hinges:
        return {}
    from yam_arena.scene import object_states

    names = sorted({s.name for s in hinges} | {s.to for s in hinges if s.to and _PLACED.get(s.to, (0, 0, False))[2]})
    states = object_states(env, names)
    out = {}
    for spec in hinges:
        q = states[spec.name].quat_xyzw
        other, other_yaw = (0.0, 0.0, 0.0, 1.0), 0.0
        if spec.to in states:
            other, other_yaw = states[spec.to].quat_xyzw, _PLACED[spec.to][1]
        if all(math.isfinite(float(v)) for v in (*q, *other)):
            out[spec.name] = hinge_angle(spec, q, other, other_yaw)
    return out


def _wrench(env, name: str, torque) -> None:
    """Hold a torque (world frame, N m) on the moving object ``name`` until changed."""
    import torch

    obj = env.unwrapped.scene.rigid_objects[name]
    t = torch.tensor([[list(torque)]], dtype=torch.float32, device=obj.device)
    composer = obj.permanent_wrench_composer
    composer.reset()
    composer.add_forces_and_torques(torch.zeros_like(t), t, is_global=True)


_TORQUE: dict[str, float] = {}


def _drives(env) -> None:
    """Every step: each hinge's zone spring, as a torque on the box and the opposite one on the other body."""
    hinges = [s for s in _SPECS.values() if isinstance(s, Hinged) and s.name in _PRIMS and s.zones]
    if not hinges:
        return
    from yam_arena.scene import object_states

    angles = hinge_angles(env)
    names = sorted({s.name for s in hinges} | {s.to for s in hinges if s.to in env.unwrapped.scene.rigid_objects})
    states = object_states(env, names)
    for spec in hinges:
        if spec.name not in angles:
            continue
        other = states.get(spec.to)
        q = other.quat_xyzw if other is not None else (0.0, 0.0, 0.0, 1.0)
        x, y, z, w = (float(v) for v in q)
        axes = {"X": (1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)),
                "Y": (2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)),
                "Z": (2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y))}
        axis = axes[spec.axis]
        spin = states[spec.name].ang_vel - (other.ang_vel if other is not None else 0.0)
        speed = float(sum(a * b for a, b in zip(axis, spin)))
        torque = spec.spring_at(angles[spec.name], speed)
        if torque == 0.0 and _TORQUE.get(spec.name, 0.0) == 0.0:
            continue
        _wrench(env, spec.name, [torque * a for a in axis])
        if other is not None:
            _wrench(env, spec.to, [-torque * a for a in axis])
        _TORQUE[spec.name] = torque


def _reset_drives(env=None) -> None:
    for spec in [s for s in _SPECS.values() if isinstance(s, Hinged) and s.name in _PRIMS]:
        if _TORQUE.pop(spec.name, 0.0) != 0.0 and env is not None:
            _wrench(env, spec.name, (0.0, 0.0, 0.0))
            if spec.to in env.unwrapped.scene.rigid_objects:
                _wrench(env, spec.to, (0.0, 0.0, 0.0))


def _spawn_cuboid(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Isaac Lab's ``spawn_cuboid``, then the joint of a bonded or sliding box. Installed in place of the original,
    which the spawner configurations name by module and attribute, so a replacement on one configuration would be
    lost."""
    prim = _SPAWN["cuboid"](prim_path, cfg, translation, orientation, **kwargs)
    name = prim_path.rstrip("/").rsplit("/", 1)[-1]
    spec = _SPECS.get(name)
    if isinstance(spec, Bonded):
        _bond(prim, spec)
    elif isinstance(spec, Sliding):
        _slide(prim, spec)
    elif isinstance(spec, Hinged):
        _hinge(prim, spec)
    if spec is not None:
        _PRIMS[spec.name] = prim
    if name in _PLACED:
        for fn in _SPAWNED:
            fn(name, prim)
    return prim


def _patch_spawner() -> None:
    if "cuboid" in _SPAWN:
        return
    import isaaclab.sim.spawners as spawners
    import isaaclab.sim.spawners.shapes as shapes
    import isaaclab.sim.spawners.shapes.shapes as module

    _SPAWN["cuboid"] = module.spawn_cuboid
    for m in (module, shapes, spawners):
        m.spawn_cuboid = _spawn_cuboid


# -- releasing a bond at run time ------------------------------------------------------------------------------------


def _set_bond(name: str, enabled: bool) -> None:
    from pxr import UsdPhysics

    joint = UsdPhysics.Joint(_PRIMS[name].GetChild("bond"))
    joint.CreateJointEnabledAttr(enabled)


def release(name: str) -> None:
    """Undo the bond of the Bonded box ``name`` for the rest of the episode: its joint is disabled, so the box is
    free from then on. A no-op if it is already released. Call it from an :func:`on_step` callback."""
    if name in _RELEASED:
        return
    if name not in _PRIMS:
        raise KeyError(f"{name} has not spawned (release() runs inside the simulation)")
    _set_bond(name, False)
    _RELEASED.add(name)


def released(name: str) -> bool:
    """Whether :func:`release` has undone ``name``'s bond this episode."""
    return name in _RELEASED


def _restore_bonds(env=None) -> None:
    for name in _RELEASED:
        _set_bond(name, True)
    _RELEASED.clear()


# -- hooking yam_arena -----------------------------------------------------------------------------------------------

_RESET: list = [_restore_bonds, _reset_drives]
_STEP: list = [_drives]


def on_spawn(fn):
    """Register ``fn(name, prim)`` to run on every task-file box right after it spawns, before the simulation
    starts (to add to its prim, as ``looks.py`` does)."""
    _SPAWNED.append(fn)
    return fn


def on_reset(fn):
    """Register ``fn(env)`` to run after every reset of the embodiment, once the scene stands at its start."""
    _RESET.append(fn)
    return fn


def on_step(fn):
    """Register ``fn(env)`` to run after every control step, before the task file's hooks score it."""
    _STEP.append(fn)
    return fn


def _patch_scene(scene) -> None:
    if getattr(scene, "_bonds", False):
        return
    _patch_spawner()
    task_object = scene.task_object

    def task_object_hook(spec):
        if hasattr(spec, "pos"):
            _PLACED[spec.name] = (tuple(spec.pos), float(spec.yaw), spec.free)
        if isinstance(spec, (Bonded, Sliding, Hinged)):
            _SPECS[spec.name] = spec
        return task_object(spec)

    scene.task_object = task_object_hook
    scene._bonds = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_bonds", False):
        return
    base_reset, base_task_step = cls.reset, cls._task_step

    def reset(self, *args, **kwargs):
        out = base_reset(self, *args, **kwargs)
        for fn in _RESET:
            fn(self._env)
        return out

    def _task_step(self, *args, **kwargs):
        for fn in _STEP:
            fn(self._env)
        return base_task_step(self, *args, **kwargs)

    cls.reset, cls._task_step = reset, _task_step
    cls._bonds = True


class _Finder(importlib.abc.MetaPathFinder):
    """Patches yam_arena's Isaac-side modules as they are imported (after the task file is loaded)."""

    targets = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader, patch = spec.loader, self.targets[fullname]
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    """Hook yam_arena now if it is already imported, and on import otherwise. Idempotent."""
    for name, patch in _Finder.targets.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
