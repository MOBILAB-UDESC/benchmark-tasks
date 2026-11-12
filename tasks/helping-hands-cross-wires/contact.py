"""contact: has the red wire touched the blue wire? Read from the simulator's contact data, not from poses.

A PhysX rigid-contact view between the two wire bodies (the mechanism behind Isaac Lab's ``ContactSensor``) is read
after every physics substep (240 Hz). It reports each contact point between the two wires with its separation (m):
a negative value is penetration, 0 is touching, and the solver also reports points up to the contact offset (5 mm)
away, so :func:`touching` asks for a separation of ``TOUCH_SEP`` or less. ``yam_arena`` builds no contact sensor
here, so contact processing is switched back on, exactly as Isaac Lab's ``ContactSensor`` does.

Pure Python at import time: everything that needs Isaac Sim is imported lazily. Importing this module hooks Isaac
Lab's ``PhysxManager`` when that module is imported (after Isaac Sim starts) and registers the wire bodies for
contact reports as they spawn.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys

import numpy as np
from bonds import on_reset, on_spawn

__all__ = ["TOUCH_SEP", "min_separation", "reset", "status", "touching"]

TOUCH_SEP = 0.0005  # m: a contact point this close or closer (or penetrating) is touching
SENSOR, OTHER = "right_wire", "left_wire"  # the contact view's sensor body and its filter body

# What the view last saw: the smallest separation over the contact points of the latest physics substep (inf: no
# contact point at all), and how many substeps have been read and had any point.
_MON: dict = {"view": None, "sep": float("inf"), "polls": 0, "hits": 0, "err": None}


def min_separation(sep, counts, starts) -> float:
    """The smallest separation over all contact points of a contact view's data (``inf`` when it has none). ``sep``
    holds every point's separation, ``counts[i]`` points belong to pair ``i`` from index ``starts[i]``."""
    sep = np.asarray(sep, float).reshape(-1)
    counts = np.asarray(counts).reshape(-1).astype(int)
    starts = np.asarray(starts).reshape(-1).astype(int)
    if not counts.any():
        return float("inf")
    idx = np.concatenate([np.arange(s, s + c) for s, c in zip(starts[counts > 0], counts[counts > 0])])
    return float(sep[idx].min())


def touching() -> bool:
    """The two wires touch now: the latest substep's closest contact point is at ``TOUCH_SEP`` or less."""
    return bool(_MON["sep"] <= TOUCH_SEP)


def status() -> dict:
    """What the monitor has seen, for the log: the closest separation (mm) now, substeps read and substeps with a contact
    point, and the first error, if the view could not be built or read."""
    return {"sep_mm": _MON["sep"] * 1000, "polls": _MON["polls"], "hits": _MON["hits"], "err": _MON["err"]}


@on_reset
def reset(env=None) -> None:
    """A fresh episode: forget the contact and rebuild the view on the next physics step (the scene may be rebuilt)."""
    _MON.update(view=None, sep=float("inf"), polls=0, hits=0)


@on_spawn
def _report_contacts(name: str, prim) -> None:
    if name not in (SENSOR, OTHER):
        return
    from pxr import PhysxSchema

    api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
    api.CreateThresholdAttr(0.0)


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x.numpy() if hasattr(x, "numpy") else x)


def _enable_contact_processing() -> None:
    try:
        import carb

        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    except Exception:  # pragma: no cover  (outside Isaac Sim)
        pass


def _make_view(mgr) -> None:
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdPhysics

    sim_view = mgr.get_physics_sim_view()
    if sim_view is None:
        return
    found = {}
    for prim in get_current_stage().Traverse():
        path = prim.GetPath().pathString
        name = path.rsplit("/", 1)[-1]
        if "/env_0/" in path and name in (SENSOR, OTHER) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            found[name] = path
    if len(found) != 2:
        raise RuntimeError(f"wire contact monitor: found {sorted(found)}, need {SENSOR} and {OTHER}")
    _MON["view"] = sim_view.create_rigid_contact_view(
        [found[SENSOR]], filter_patterns=[[found[OTHER]]], max_contact_data_count=64
    )
    print(f"[wires] contact monitor: {found[SENSOR]} against {found[OTHER]}", flush=True)


def _poll(mgr) -> None:
    """Called after every physics substep."""
    try:
        if _MON["view"] is None:
            _make_view(mgr)
        if _MON["view"] is None:
            return
        _, _, _, sep, counts, starts = _MON["view"].get_contact_data(dt=mgr.get_physics_dt())
        _MON["sep"] = min_separation(_np(sep), _np(counts), _np(starts))
    except Exception as e:  # pragma: no cover
        _MON["sep"] = float("inf")
        if _MON["err"] is None:
            _MON["err"] = repr(e)
            print(f"[wires] contact monitor error: {e!r}", flush=True)
        return
    _MON["polls"] += 1
    _MON["hits"] += int(_MON["sep"] < float("inf"))


_PHYSX_MGR = "isaaclab_physx.physics.physx_manager"


def _patch_manager(mod) -> None:
    mgr = mod.PhysxManager
    if getattr(mgr.step, "_wires_hooked", False):
        return
    base_cfg = mgr._configure_physics.__func__
    base_step = mgr.step.__func__

    def _configure_physics(cls) -> None:
        base_cfg(cls)
        _enable_contact_processing()

    def step(cls) -> None:
        base_step(cls)
        _poll(cls)

    step._wires_hooked = True
    mgr._configure_physics = classmethod(_configure_physics)
    mgr.step = classmethod(step)


class _Finder(importlib.abc.MetaPathFinder):
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
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch_manager(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    """Hook Isaac Lab's PhysxManager now if it is already imported, and on import otherwise. Idempotent."""
    if _PHYSX_MGR in sys.modules:
        _patch_manager(sys.modules[_PHYSX_MGR])
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
