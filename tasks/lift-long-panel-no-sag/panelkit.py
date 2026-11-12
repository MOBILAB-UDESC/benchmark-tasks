"""panelkit: a long thin panel that bends, for yam_arena task files, and render-only looks for fixed boxes.

A task file uses it like this (keep this file next to the task file):

    import panelkit
    from panelkit import Panel, dress, on_reset

    PANEL = Panel("panel", pos=(0.26, 0.0, 0.78), length=0.60, width=0.06, thickness=0.0008, panel_mass=0.12,
                  bending_stiffness=0.07)
    OBJECTS = [PANEL, ...]

The panel is a chain of rigid boxes along world y, joined by hinges about world x. Every hinge has a spring and a
damper that pull it back to flat, set so that the whole chain bends like a uniform strip of bending stiffness EI
(N m^2): each hinge gets EI divided by the segment length. Nothing else can move: the hinges allow no stretch, no
sideways bend and no twist.

It plugs into yam_arena without modifying it: importing this module hooks ``yam_arena.scene`` so a ``Panel`` in
``OBJECTS`` is spawned as a PhysX articulation and ``success()/score()/distance()`` receive a :class:`PanelState`
(an ``ObjectState`` with every segment's centre, velocity and tilt). It also hooks the embodiment's reset and control step, so a task
file can clear what it remembers (:func:`on_reset`) and read the scene every step (:func:`on_step`), and the spawner
of fixed boxes, so a box can be drawn by a mesh in ``looks/`` instead (:func:`dress`, :func:`panel_look`). Pure Python at import time: everything that needs Isaac
Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from yam_arena.task_file import Box, ObjectState

__all__ = ["Panel", "PanelState", "bow", "dress", "install", "on_reset", "on_step", "panel", "panel_look"]

FOLDER = Path(__file__).resolve().parent / "looks"


@dataclass(frozen=True)
class Panel(Box):
    """A flat panel lying along world y, centred on ``pos``, ``length`` long (y), ``width`` wide (x) and
    ``thickness`` thick (z). It is ``segments`` rigid boxes of equal length on springy hinges that give it the
    bending stiffness ``bending_stiffness`` (EI, N m^2). ``pos`` and ``size`` are its bounding box at the start."""

    length: float = 0.6
    width: float = 0.06
    thickness: float = 0.0008
    segments: int = 20
    panel_mass: float = 0.12  # kg, the whole panel
    bending_stiffness: float = 0.07  # EI in N m^2
    damping_ratio: float = 0.08  # hinge damping as a fraction of the hinge spring (N m s/rad per N m/rad)
    armature: float = 5e-5  # kg m^2 per hinge: keeps the light hinges stable at 240 Hz
    max_bend: float = 0.8  # rad per hinge
    friction: float = 1.0
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def free(self) -> bool:  # never jittered by yam_arena (its reset writes rigid poses only)
        return False

    @property
    def seg_len(self) -> float:
        return self.length / self.segments

    @property
    def hinge_stiffness(self) -> float:
        """N m/rad of each hinge: EI over the segment length."""
        return self.bending_stiffness / self.seg_len

    def _check(self) -> None:
        super()._check()
        if self.segments < 3 or not (self.length > 0 and self.width > 0 and self.thickness > 0):
            raise ValueError(f"{self.name}: need segments >= 3 and positive length, width, thickness")
        if self.panel_mass <= 0 or self.bending_stiffness <= 0:
            raise ValueError(f"{self.name}: mass and bending stiffness must be positive")

    def start_points(self) -> np.ndarray:
        """Starting segment centres, (segments, 3), from the low-y end to the high-y end."""
        y = self.pos[1] - self.length / 2 + (np.arange(self.segments) + 0.5) * self.seg_len
        return np.stack([np.full(self.segments, self.pos[0]), y, np.full(self.segments, self.pos[2])], axis=1)


def panel(name: str, centre: tuple[float, float, float], **kwargs) -> Panel:
    """A :class:`Panel` of the given centre and dimensions, with its bounding box filled in."""
    length = kwargs.get("length", 0.6)
    width = kwargs.get("width", 0.06)
    thickness = kwargs.get("thickness", 0.0008)
    return Panel(name=name, pos=tuple(centre), size=(width, length, thickness), **kwargs)


@dataclass(frozen=True)
class PanelState(ObjectState):
    """An ObjectState for a panel (``pos`` is the mean of the segment centres), plus its shape."""

    points: np.ndarray = None  # (segments, 3) segment centres, from the low-y end to the high-y end
    speeds: np.ndarray = None  # (segments,) each segment's speed, m/s
    spins: np.ndarray = None  # (segments,) each segment's angular speed, rad/s
    tilts: np.ndarray = None  # (segments,) each segment's pitch about world x, rad (0 = flat)

    def resting(self, max_speed: float = 0.01, max_spin: float = 0.5) -> bool:
        """Every segment under ``max_speed`` m/s and ``max_spin`` rad/s."""
        return bool(np.all(self.speeds < max_speed) and np.all(self.spins < max_spin))


def bow(points: np.ndarray) -> float:
    """How far (m) the segment centres lie from the straight line through the first and the last of them: the gap
    under a straightedge laid end to end along the panel."""
    a, b = points[0], points[-1]
    d = (b - a) / (np.linalg.norm(b - a) + 1e-12)
    v = points - a
    perp = v - np.outer(v @ d, d)
    return float(np.max(np.linalg.norm(perp, axis=1)))


# -- looks -----------------------------------------------------------------------------------------------------------

_LOOKS: dict[str, Path | None] = {}
_PANEL_LOOKS: dict[str, Path] = {}


def dress(looks: dict[str, str], hide=()) -> None:
    """Give each fixed box named in ``looks`` the mesh in ``looks/<file>`` (the box stays as its collider, made
    invisible), and make the boxes in ``hide`` invisible with no mesh of their own."""
    for name, file in looks.items():
        path = FOLDER / file
        if not path.is_file():
            raise FileNotFoundError(f"{name}: no {path} (run looks/make_looks.py in Blender)")
        _LOOKS[name] = path
    for name in hide:
        _LOOKS.setdefault(name, None)


def panel_look(name: str, pattern: str) -> None:
    """Draw the panel ``name`` with one file per segment: ``looks/<pattern>`` with the segment's number in place of
    ``{}`` (``panel_{:02d}.usdc``), each a USD file with a default prim modelled around its segment's centre. The
    segments' boxes still collide, and are made invisible."""
    paths = [FOLDER / pattern.format(i) for i in range(1000) if (FOLDER / pattern.format(i)).is_file()]
    if not paths:
        raise FileNotFoundError(f"{name}: no {FOLDER / pattern.format(0)} (run looks/make_looks.py in Blender)")
    _PANEL_LOOKS[name] = FOLDER / pattern


def _dress_box(name: str, prim) -> None:
    from pxr import Usd, UsdGeom

    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Gprim):
            UsdGeom.Imageable(p).MakeInvisible()
    path = _LOOKS[name]
    if path is not None:
        look = UsdGeom.Xform.Define(prim.GetStage(), prim.GetPath().AppendChild("look"))
        look.GetPrim().GetReferences().AddReference(str(path))


_SPAWN: dict = {}


def _spawn_cuboid(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Isaac Lab's ``spawn_cuboid``, then the look of a dressed box. Installed in place of the original, which the
    spawner configurations name by module and attribute, so a replacement on one configuration would be lost."""
    prim = _SPAWN["cuboid"](prim_path, cfg, translation, orientation, **kwargs)
    name = prim_path.rstrip("/").rsplit("/", 1)[-1]
    if name in _LOOKS:
        _dress_box(name, prim)
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


# -- the panel's articulation ----------------------------------------------------------------------------------------

_SPECS: dict[str, Panel] = {}


def _spawn_panel(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the panel's articulation under ``prim_path`` (pxr only). ``cfg.panel_name`` keys the Panel spec."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec: Panel = _SPECS[cfg.panel_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)  # Isaac Lab 3.0: (x, y, z, w)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    vis = f"{prim_path}/Looks/panel"
    look = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.45)
    look.func(vis, look)
    phys = f"{prim_path}/Looks/panel_physics"
    pm = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    pm.func(phys, pm)

    n, ell = spec.segments, spec.seg_len
    seg_mass = spec.panel_mass / n
    gap = 0.0004  # m between neighbouring boxes, so a bent hinge doesn't pinch their corners
    dressed = spec.name in _PANEL_LOOKS

    seg_paths = []
    for i in range(n):
        path = f"{prim_path}/seg_{i}"
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(i * ell, 0.0, 0.0))
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(seg_mass)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        rb.CreateSolverPositionIterationCountAttr(32)
        rb.CreateLinearDampingAttr(0.2)
        rb.CreateAngularDampingAttr(0.5)
        geom = UsdGeom.Cube.Define(stage, f"{path}/geom")
        geom.CreateSizeAttr(1.0)
        geom.AddScaleOp().Set(Gf.Vec3f(ell - gap, spec.width, spec.thickness))  # along the panel, across, thick
        gp = geom.GetPrim()
        UsdPhysics.CollisionAPI.Apply(gp)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(gp)
        cp.CreateContactOffsetAttr(0.002)
        cp.CreateRestOffsetAttr(0.0)
        bind_physics_material(str(gp.GetPath()), phys, stage=stage)
        bind_visual_material(str(gp.GetPath()), vis, stage=stage)
        if dressed:
            UsdGeom.Imageable(gp).MakeInvisible()
            mesh = UsdGeom.Xform.Define(stage, f"{path}/look")
            mesh.GetPrim().GetReferences().AddReference(str(_PANEL_LOOKS[spec.name]).format(i))
        seg_paths.append(path)

    stiffness, damping = spec.hinge_stiffness, spec.hinge_stiffness * spec.damping_ratio
    for i in range(1, n):
        child = seg_paths[i]
        j = UsdPhysics.RevoluteJoint.Define(stage, f"{child}/bend_{i}")
        j.CreateBody0Rel().SetTargets([Sdf.Path(seg_paths[i - 1])])
        j.CreateBody1Rel().SetTargets([Sdf.Path(child)])
        j.CreateLocalPos0Attr(Gf.Vec3f(ell / 2, 0, 0))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(-ell / 2, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateAxisAttr("Y")  # the panel's own across axis, which is world x
        j.CreateLowerLimitAttr(-math.degrees(spec.max_bend))
        j.CreateUpperLimitAttr(math.degrees(spec.max_bend))
        drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        drv.CreateTypeAttr("force")
        drv.CreateStiffnessAttr(stiffness * math.pi / 180)  # USD angular drive gains are per degree
        drv.CreateDampingAttr(damping * math.pi / 180)
        drv.CreateTargetPositionAttr(0.0)

    # Root API on seg_0 itself, so seg_0 is the root link and Isaac Lab's root pose is seg_0's pose.
    root_link = stage.GetPrimAtPath(seg_paths[0])
    UsdPhysics.ArticulationRootAPI.Apply(root_link)
    art = PhysxSchema.PhysxArticulationAPI.Apply(root_link)
    art.CreateEnabledSelfCollisionsAttr(False)
    art.CreateSolverPositionIterationCountAttr(32)
    art.CreateSolverVelocityIterationCountAttr(4)
    return root.GetPrim()


def _arena_object(spec: Panel):
    """The panel as an Arena articulation object."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class PanelSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_panel)
        panel_name: str = ""

    class PanelObject(Object):
        def _generate_articulation_cfg(self):
            cfg = ArticulationCfg(
                prim_path=self.prim_path,
                spawn=self.spawner_cfg,
                actuators={
                    "bend": ImplicitActuatorCfg(
                        joint_names_expr=[".*"],
                        stiffness=spec.hinge_stiffness,
                        damping=spec.hinge_stiffness * spec.damping_ratio,
                        armature=spec.armature,
                    )
                },
            )
            return self._add_initial_pose_to_cfg(cfg)

    first = spec.start_points()[0]
    turn = math.pi / 2  # the chain's own x axis along world y
    return PanelObject(
        name=spec.name,
        object_type=ObjectType.ARTICULATION,
        spawner_cfg=PanelSpawnerCfg(panel_name=spec.name),
        initial_pose=Pose(position_xyz=tuple(float(v) for v in first),
                          rotation_xyzw=(0.0, 0.0, math.sin(turn / 2), math.cos(turn / 2))),
    )


def _numpy(t) -> np.ndarray:
    t = getattr(t, "torch", t)
    return t.detach().cpu().numpy().astype(np.float64)


def _panel_state(env, spec: Panel) -> PanelState:
    base = env.unwrapped
    art = base.scene[spec.name]
    origin = _numpy(base.scene.env_origins[0])
    names = list(art.body_names)
    idx = [names.index(f"seg_{i}") for i in range(spec.segments)]
    pts = (_numpy(art.data.body_pos_w)[0] - origin)[idx]
    quat = _numpy(art.data.body_quat_w)[0][idx]  # Isaac Lab 3.0: x, y, z, w
    vel = _numpy(art.data.body_lin_vel_w)[0][idx]
    angv = _numpy(art.data.body_ang_vel_w)[0][idx]
    # The chain's own x axis lies along world y, so a segment's pitch is the angle its long axis (the first column of
    # its rotation matrix) makes with the horizontal.
    qx, qy, qz, qw = quat.T
    long_axis_z = 2 * (qx * qz - qw * qy)
    lo, hi = pts.min(0) - spec.thickness / 2, pts.max(0) + spec.thickness / 2
    return PanelState(
        pos=pts.mean(0),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        lin_vel=vel.mean(0),
        ang_vel=angv.mean(0),
        low=lo,
        high=hi,
        points=pts,
        speeds=np.linalg.norm(vel, axis=1),
        spins=np.linalg.norm(angv, axis=1),
        tilts=np.arcsin(np.clip(long_axis_z, -1.0, 1.0)),
    )


# -- hooking yam_arena -----------------------------------------------------------------------------------------------

_RESET: list = []
_STEP: list = []


def on_reset(fn):
    """Register ``fn(env)`` to run after every reset of the embodiment, once the scene stands at its start."""
    _RESET.append(fn)
    return fn


def on_step(fn):
    """Register ``fn(env)`` to run after every control step, before the task file's hooks score it."""
    _STEP.append(fn)
    return fn


def _patch_scene(scene) -> None:
    if getattr(scene, "_panelkit", False):
        return
    _patch_spawner()
    orig_task_object, orig_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Panel):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return orig_task_object(spec)

    def object_states(env, names):
        plain = [n for n in names if n not in _SPECS]
        out = orig_states(env, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                out[n] = _panel_state(env, _SPECS[n])
        return {n: out[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._panelkit = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_panelkit", False):
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
    cls._panelkit = True


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
