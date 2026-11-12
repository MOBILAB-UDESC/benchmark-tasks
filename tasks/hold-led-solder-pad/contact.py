"""contact: real contact between a gripper finger and the LED, and between the iron and the board, read from the
simulator's own contact solver.

Two PhysX rigid-contact views (the mechanism behind Isaac Lab's ContactSensor) are read after every physics substep
(240 Hz, so 12 times per control step):

- ``led``: the four gripper fingers (the YAM USD has contact reporting on every robot link) against the LED. A
  substep counts when any contact point between them carries a force of at least ``F_MIN`` newtons.
- ``board``: the soldering iron against the board. The iron gets a contact report when it spawns, and the board is
  made a kinematic body (it never moves) so that the view can filter on it.

A finger hovering 1 mm over the LED, or the tip hovering 1 mm over a pad, carries no force. A control step counts as
in contact when at least one of its substeps did (``take``): a held iron bounces a little, so a pressed tip carries its
force in only some substeps of a step, and sampling once per control step would miss a touch that was there. Pure Python at import time: everything that needs Isaac Sim is imported on first use.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys

import numpy as np
from bonds import on_reset, on_spawn

__all__ = ["F_MIN", "partial", "peaks", "reduce_contacts", "step_touched", "take"]

F_MIN = 0.1  # N: a contact point carrying this much force is a touch (a thin-film force sensor reads 0.1 N)
FINGERS = ("tip_left", "tip_right")  # the gripper fingers' links, on both arms
_MON: dict = {"views": None, "subs": 0, "hits": {"led": 0, "board": 0}, "peak": {"led": 0.0, "board": 0.0},
              "err": None}


def reduce_contacts(forces, counts) -> tuple[bool, float]:
    """Whether any contact point carries at least ``F_MIN`` newtons, and the largest force. ``forces`` is the view's
    force per contact point (N), ``counts`` its number of contact points for each sensor and filter pair."""
    f = np.asarray(forces, float)
    if f.ndim > 1:
        f = np.linalg.norm(f.reshape(len(f), -1), axis=1)
    f = f.reshape(-1)
    if not np.asarray(counts).any() or f.size == 0:
        return False, 0.0
    peak = float(f.max())
    return peak >= F_MIN, peak


def step_touched(substeps: int, hits: int) -> bool:
    """A control step is in contact when at least one of its physics substeps was."""
    return substeps > 0 and hits >= 1


def partial() -> tuple[int, int, int]:
    """(substeps, substeps with a finger on the LED, substeps with the iron on the board) so far this control step."""
    return _MON["subs"], _MON["hits"]["led"], _MON["hits"]["board"]


def take() -> tuple[bool, bool]:
    """(a finger on the LED, the iron on the board) over the control step that just ended, and start the next one."""
    subs = _MON["subs"]
    out = (step_touched(subs, _MON["hits"]["led"]), step_touched(subs, _MON["hits"]["board"]))
    _MON["subs"] = 0
    _MON["hits"] = {"led": 0, "board": 0}
    _MON["peak"] = {"led": 0.0, "board": 0.0}
    return out


def peaks() -> tuple[float, float]:
    """The largest contact force (N) of the LED and board views in this control step, for the run log."""
    return _MON["peak"]["led"], _MON["peak"]["board"]


@on_reset
def _reset(env=None) -> None:
    _MON.update(views=None, subs=0, hits={"led": 0, "board": 0}, peak={"led": 0.0, "board": 0.0})


# -- spawning: contact reports on the iron, a kinematic board -------------------------------------------------------------


@on_spawn
def _spawned(name: str, prim) -> None:
    from pxr import PhysxSchema, UsdPhysics

    if name == "iron":
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
    elif name == "board":
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)


# -- the views ---------------------------------------------------------------------------------------------------------


def _enable_contact_processing() -> None:
    """Isaac Lab turns PhysX contact processing off unless a ContactSensor exists (this embodiment builds none):
    switch it back on, exactly as Isaac Lab's own ContactSensor does."""
    try:
        import carb

        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    except Exception:  # pragma: no cover  (outside Isaac Sim)
        pass


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x)


def _make_views(mgr) -> None:
    from isaaclab.sim.utils.stage import get_current_stage

    sv = mgr.get_physics_sim_view()
    if sv is None:
        return
    stage = get_current_stage()
    fingers, bodies = [], {}
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if "/env_0/" not in path:
            continue
        leaf = path.rsplit("/", 1)[-1]
        if leaf in FINGERS and "PhysxContactReportAPI" in prim.GetAppliedSchemas():
            fingers.append(path)
        elif leaf in ("led", "iron", "board") and path.rsplit("/", 2)[-2] != "parts":
            bodies.setdefault(leaf, path)
    if not fingers or len(bodies) < 3:
        raise RuntimeError(f"contact: {len(fingers)} fingers, bodies {sorted(bodies)}")
    _MON["views"] = {
        "led": sv.create_rigid_contact_view(fingers, filter_patterns=[[bodies["led"]]] * len(fingers),
                                            max_contact_data_count=64 * len(fingers)),
        "board": sv.create_rigid_contact_view([bodies["iron"]], filter_patterns=[[bodies["board"]]],
                                              max_contact_data_count=256),
    }
    print(f"[contact] views: {len(fingers)} fingers ({fingers[0].rsplit('/', 3)[-3:]}), "
          f"led {bodies['led']}, iron {bodies['iron']}, board {bodies['board']}", flush=True)


def _poll(mgr) -> None:
    """Called after every physics substep."""
    try:
        if _MON["views"] is None:
            _make_views(mgr)
        views = _MON["views"]
        if views is None:
            return
        dt = mgr.get_physics_dt()
        seen = {}
        for key, view in views.items():
            forces, _, _, _, counts, _ = view.get_contact_data(dt=dt)
            seen[key] = reduce_contacts(_np(forces), _np(counts))
    except Exception as e:  # pragma: no cover
        if _MON["err"] is None:
            _MON["err"] = repr(e)
            print(f"[contact] error: {e!r}", flush=True)
        return
    _MON["subs"] += 1
    for key, (hit, peak) in seen.items():
        _MON["hits"][key] += int(hit)
        _MON["peak"][key] = max(_MON["peak"][key], peak)


_PHYSX_MGR = "isaaclab_physx.physics.physx_manager"


def _patch_manager(mod) -> None:
    mgr = mod.PhysxManager
    if getattr(mgr.step, "_led_contact_hooked", False):
        return
    base_cfg = mgr._configure_physics.__func__
    base_step = mgr.step.__func__

    def _configure_physics(cls) -> None:
        base_cfg(cls)
        _enable_contact_processing()

    def step(cls) -> None:
        base_step(cls)
        _poll(cls)

    step._led_contact_hooked = True
    mgr._configure_physics = classmethod(_configure_physics)
    mgr.step = classmethod(step)


class _MgrFinder(importlib.abc.MetaPathFinder):
    """Patches Isaac Lab's PhysxManager when it is first imported (it needs Kit, so not at task import)."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _PHYSX_MGR:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch_manager(module)

        spec.loader.exec_module = exec_module
        return spec


def install() -> None:
    """Hook Isaac Lab's PhysxManager now if it is already imported, and on import otherwise. Idempotent."""
    if _PHYSX_MGR in sys.modules:
        _patch_manager(sys.modules[_PHYSX_MGR])
    elif not any(isinstance(f, _MgrFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MgrFinder())


install()
