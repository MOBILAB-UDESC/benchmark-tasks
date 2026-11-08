"""make_looks.py: model the stands' visual meshes in Blender and export them as USD next to this file.

    blender -b -P looks/make_looks.py          (Blender 4.2 or later)

The physics stays the task file's boxes. Each look is a render-only mesh that ``looks.py`` puts on one of those boxes,
whose plain cuboid is then hidden but still collides. Every look is built in its box's own frame (x along the link, y
across it, z up, metres, the box's centre at the origin).

- ``<side>_link1.usdc``, ``<side>_link2.usdc``: a gunmetal bar with a chrome friction knob at its far hinge.
- ``<side>_link3.usdc``: the bar, a black insulating boot and a steel alligator clip whose jaws face the wire.
- ``<side>_wire.usdc``: the wire stub, red (right) or blue (left), with a bare copper tip.
- ``<side>_base.usdc``: the stand's black weighted base with its hinge post.
- ``mark.usdc``: the yellow tape cross on the table.

Each link's and wire's mesh is checked to lie inside its box when exported. Every datablock carries the folder's name
as a prefix, so the shared Blender session can hold other tasks' looks. Nothing is downloaded.
"""

import math
import os

import bmesh
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
P = "helping-hands-cross-wires__"

# -- the task file's layout (helping_hands_cross_wires.py), metres ---------------------------------------------------
LINK = (0.11, 0.11, 0.12)
EXT = 0.007
LINK_W, LINK_H = 0.014, 0.024
WIRE = (0.05, 0.005, 0.005)
PLINTH = (0.08, 0.08)
PLINTH_H = {"right": 0.030, "left": 0.038}
GAP = {"right": 0.003, "left": 0.003}  # from a base's top to the first link's underside
MARK_SIZE = (0.036, 0.007, 0.001)

_MATS = {}


def material(name, color, rough=0.5, metal=0.0):
    name = P + name
    if name in _MATS:
        return _MATS[name]
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nodes, links = m.node_tree.nodes, m.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    m.diffuse_color = (*color, 1.0)
    _MATS[name] = m
    return m


def materials():
    return {
        "bar": material("gunmetal", (0.20, 0.21, 0.23), 0.35, 0.85),
        "chrome": material("chrome", (0.78, 0.79, 0.80), 0.18, 1.0),
        "boot": material("boot_black", (0.02, 0.02, 0.022), 0.5),
        "steel": material("steel", (0.55, 0.56, 0.58), 0.3, 1.0),
        "base": material("base_black", (0.03, 0.03, 0.035), 0.55, 0.2),
        "rubber": material("rubber", (0.012, 0.012, 0.012), 0.9),
        "red": material("wire_red", (0.85, 0.04, 0.03), 0.4),
        "blue": material("wire_blue", (0.03, 0.12, 0.85), 0.4),
        "copper": material("copper", (0.93, 0.52, 0.33), 0.28, 1.0),
        "yellow": material("tape_yellow", (0.95, 0.74, 0.02), 0.6),
    }


class Look:
    def __init__(self, name):
        self.name = name
        self.coll = bpy.data.collections.new(P + "look_" + name)
        bpy.context.scene.collection.children.link(self.coll)

    def mesh(self, name, bm, mat, bevel=0.0, smooth=False):
        me = bpy.data.meshes.new(P + self.name + "_" + name)
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new(P + self.name + "_" + name, me)
        self.coll.objects.link(ob)
        ob.data.materials.append(mat)
        if smooth:
            for poly in me.polygons:
                poly.use_smooth = True
        if bevel > 0:
            m = ob.modifiers.new("bevel", "BEVEL")
            m.width, m.segments, m.limit_method = bevel, 2, "ANGLE"
        return ob


def bm_box(lo, hi):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((lo[0] if v.co.x < 0 else hi[0], lo[1] if v.co.y < 0 else hi[1], lo[2] if v.co.z < 0 else hi[2]))
    return bm


def bm_cyl(center, r, h, segs=32, r2=None, axis="Z"):
    """A cylinder (or cone) about ``axis`` through ``center``, ``h`` long."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r, radius2=r if r2 is None else r2, depth=h)
    if axis == "X":
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=__import__("mathutils").Matrix.Rotation(math.pi / 2, 3, "Y"))
    elif axis == "Y":
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=__import__("mathutils").Matrix.Rotation(math.pi / 2, 3, "X"))
    bmesh.ops.translate(bm, verts=bm.verts, vec=center)
    return bm


# -- the looks -------------------------------------------------------------------------------------------------------


def build_link(side, i, M):
    L = Look(f"{side}_link{i}")
    last = i == 3
    lo, hi = (-EXT, CLIP_END) if last else (-EXT, LINK[i - 1] + EXT)
    centre = (lo + hi) / 2
    p_in = -centre  # the link's first hinge, in the box frame
    p_out = (LINK[i - 1] - centre) if not last else None
    thick = {1: 0.018, 2: 0.020, 3: 0.022}[i]
    bar_hi = p_out if p_out is not None else 0.012 + (-centre)  # the bar ends where the boot starts
    if last:
        boot0, boot1 = 0.020 - centre, 0.062 - centre
        L.mesh("bar", bm_box((p_in, -0.006, -thick / 2), (boot0 + 0.002, 0.006, thick / 2)), M["bar"], bevel=0.0012)
        L.mesh("end_round", bm_cyl((p_in, 0, 0), 0.007, thick, 32), M["bar"], smooth=True)
        L.mesh("boot", bm_box((boot0, -0.0068, -0.0108), (boot1, 0.0068, 0.0108)), M["boot"], bevel=0.0025)
        # the alligator jaws: two steel plates, one over the other, opening toward the wire, with teeth
        jx0, jx1 = boot1, hi - centre - 0.0
        for k, (z0, z1) in enumerate(((-0.0105, -0.0012), (0.0012, 0.0105))):
            L.mesh(f"jaw{k}", bm_box((jx0, -0.0058, z0), (jx1, 0.0058, z1)), M["steel"], bevel=0.0006)
            for t in range(5):
                x = jx0 + 0.0025 + t * 0.0034
                zt = (z1 if k == 0 else z0)
                L.mesh(f"tooth{k}_{t}", bm_box((x, -0.0055, min(zt, zt + (0.0012 if k == 0 else -0.0012))),
                                              (x + 0.0017, 0.0055, max(zt, zt + (0.0012 if k == 0 else -0.0012)))), M["steel"])
        L.mesh("pin", bm_cyl((p_in, 0, thick / 2 + 0.0004), 0.0038, 0.0008 + 0.0, 24), M["chrome"], smooth=True)
    else:
        L.mesh("bar", bm_box((p_in, -0.006, -thick / 2), (p_out, 0.006, thick / 2)), M["bar"], bevel=0.0012)
        L.mesh("end_in", bm_cyl((p_in, 0, 0), 0.007, thick, 32), M["bar"], smooth=True)
        L.mesh("end_out", bm_cyl((p_out, 0, 0), 0.007, thick, 32), M["bar"], smooth=True)
        # the friction knob over the next hinge: chrome, up to the box's top
        L.mesh("knob", bm_cyl((p_out, 0, (LINK_H / 2 + thick / 2) / 2), 0.0068, LINK_H / 2 - thick / 2, 32), M["chrome"],
               smooth=True)
        L.mesh("knob_cap", bm_cyl((p_out, 0, LINK_H / 2 - 0.0005), 0.0045, 0.001, 24), M["steel"], smooth=True)
        L.mesh("pin", bm_cyl((p_in, 0, thick / 2 + 0.0004), 0.0038, 0.0008, 24), M["chrome"], smooth=True)
    return L


CLIP_END = LINK[2]


def build_wire(side, M):
    L = Look(f"{side}_wire")
    ins = M["red"] if side == "right" else M["blue"]
    r = 0.0024
    L.mesh("insulation", bm_cyl((-0.004, 0, 0), r, 0.040, 24, axis="X"), ins, smooth=True)
    L.mesh("copper", bm_cyl((0.0185, 0, 0), 0.0016, 0.013, 16, axis="X"), M["copper"], smooth=True)
    return L


def build_base(side, M):
    L = Look(f"{side}_base")
    h = PLINTH_H[side]
    L.mesh("plinth", bm_box((-PLINTH[0] / 2, -PLINTH[1] / 2, -h / 2), (PLINTH[0] / 2, PLINTH[1] / 2, h / 2)), M["base"],
           bevel=0.004)
    L.mesh("foot", bm_box((-PLINTH[0] / 2 + 0.002, -PLINTH[1] / 2 + 0.002, -h / 2 - 0.0), (PLINTH[0] / 2 - 0.002,
                                                                                      PLINTH[1] / 2 - 0.002, -h / 2 + 0.0015)),
           M["rubber"], bevel=0.001)
    L.mesh("post", bm_cyl((0, 0, h / 2 + GAP[side] / 2), 0.0065, GAP[side] + 0.0008, 32), M["chrome"], smooth=True)
    L.mesh("washer", bm_cyl((0, 0, h / 2 - 0.0005), 0.014, 0.001, 32), M["steel"], smooth=True)
    return L


def build_mark(M):
    L = Look("mark")
    L.mesh("bar_x", bm_box((-MARK_SIZE[0] / 2, -MARK_SIZE[1] / 2, -MARK_SIZE[2] / 2),
                           (MARK_SIZE[0] / 2, MARK_SIZE[1] / 2, MARK_SIZE[2] / 2)), M["yellow"])
    L.mesh("bar_y", bm_box((-MARK_SIZE[1] / 2, -MARK_SIZE[0] / 2, -MARK_SIZE[2] / 2),
                           (MARK_SIZE[1] / 2, MARK_SIZE[0] / 2, MARK_SIZE[2] / 2)), M["yellow"])
    return L


# -- export ----------------------------------------------------------------------------------------------------------


def _bounds(look):
    dg = bpy.context.evaluated_depsgraph_get()
    lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
    for ob in look.coll.objects:
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        for v in me.vertices:
            p = ob.matrix_world @ v.co
            lo = Vector(map(min, lo, p))
            hi = Vector(map(max, hi, p))
        ev.to_mesh_clear()
    return lo, hi


def export(look, size=None):
    bpy.context.view_layer.update()
    lo, hi = _bounds(look)
    if size is not None:
        over = max(max(-lo[i] - size[i] / 2, hi[i] - size[i] / 2) for i in range(3))
        if over > 1e-5:
            raise ValueError(f"{look.name}: the look sticks {over * 1000:.2f} mm out of its box ({lo}, {hi})")
    path = os.path.join(HERE, look.name + ".usdc")
    bpy.ops.wm.usd_export(filepath=path, collection=look.coll.name, export_materials=True,
                          generate_preview_surface=True, export_textures_mode="KEEP", relative_paths=True,
                          root_prim_path="/look", export_lights=False, export_cameras=False,
                          convert_world_material=False, export_custom_properties=False, author_blender_name=False,
                          evaluation_mode="RENDER", export_uvmaps=True, export_normals=True)
    return {"file": os.path.basename(path), "kB": round(os.path.getsize(path) / 1024),
            "low": tuple(round(v, 4) for v in lo), "high": tuple(round(v, 4) for v in hi)}


def clear():
    for coll in list(bpy.data.collections):
        if coll.name.startswith(P):
            for ob in list(coll.objects):
                bpy.data.objects.remove(ob)
            bpy.data.collections.remove(coll)
    for block in (bpy.data.meshes, bpy.data.materials):
        for item in list(block):
            if item.name.startswith(P) and item.users == 0:
                block.remove(item)
    _MATS.clear()


def build():
    clear()
    M = materials()
    out = {}
    for side in ("right", "left"):
        for i in (1, 2, 3):
            size = ((LINK[i - 1] + 2 * EXT) if i < 3 else CLIP_END + EXT, LINK_W, LINK_H)
            out[f"{side}_link{i}"] = export(build_link(side, i, M), size)
        out[f"{side}_wire"] = export(build_wire(side, M), WIRE)
        out[f"{side}_base"] = export(build_base(side, M))
    out["mark"] = export(build_mark(M))
    clear()
    return out


if __name__ == "__main__":
    print(build())
