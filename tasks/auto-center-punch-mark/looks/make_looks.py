"""make_looks.py: model the centre-punch task's four props in Blender and export them as USD next to this file.

    blender -b -P looks/make_looks.py          (Blender 4.2 or later, from the task folder)

The physics stays the task file's boxes. Each look is a render-only mesh that ``looks.py`` puts on one of those boxes,
whose plain cuboid is then hidden but still collides. Every look is built in its box's own frame (metres, the box's
centre at the origin, z up) and must fit inside the box (checked when exported).

- ``plate.usdc``: the steel plate, 11 x 7 x 1.2 cm, its top coated with blue layout dye (a texture drawn here) and
  scribed with a cross whose lines show bright steel.
- ``holder.usdc``: the dark tool holder, a 4 cm block with a pocket for the punch's nose and a hole for its tip.
- ``tip.usdc``: the punch's nose: a bright steel stem going up into the barrel, a knurled nose with a taper and the
  hardened tip.
- ``barrel.usdc``: the punch's red anodised barrel, a sleeve with a hollow for the stem, a steel collar and a black cap.

Textures are drawn here with numpy and saved in ``textures/``. Nothing is downloaded. Everything this script makes in
Blender is named with the prefix ``<task folder>__``, and only those datablocks are removed when it runs again.
"""

import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Vector

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # pasted into a console
    HERE = os.getcwd()
TEX = os.path.join(HERE, "textures")
FOLDER = os.path.basename(os.path.dirname(HERE))
P = FOLDER + "__"
RNG = np.random.default_rng(16)

# -- the boxes (the task file), metres ---------------------------------------------------------------------------------
PLATE = (0.11, 0.07, 0.012)
CROSS_AT = (0.010, 0.006)  # the cross's centre from the plate's centre
CROSS_ARM = 0.012
HOLDER = (0.040, 0.040, 0.028)
HOLE, POCKET, BASE_Z, FLOOR_Z = 0.007, 0.021, 0.010, 0.018
TIP = (0.016, 0.016, 0.068)
TIP_PARTS = {"point": 0.004, "shoulder": 0.030, "stem": 0.034}
BARREL = (0.022, 0.022, 0.085)
SLEEVE_DEPTH = 0.035


def srgb(c):
    return tuple(((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c)


def material(name, color, rough=0.5, metal=0.0, image=None):
    full = P + name
    m = bpy.data.materials.get(full) or bpy.data.materials.new(full)
    m.use_nodes = True
    nodes, links = m.node_tree.nodes, m.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if image is not None:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.extension = "REPEAT"
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    m.diffuse_color = (*color, 1.0)
    return m


def _save(name, rgb):
    """Save an sRGB image (rows from the bottom, as Blender stores them) as JPEG in textures/ and load it."""
    os.makedirs(TEX, exist_ok=True)
    path = os.path.join(TEX, name + ".jpg")
    h, w, _ = rgb.shape
    for old in [i for i in bpy.data.images if i.name.startswith(P + name)]:
        bpy.data.images.remove(old)
    img = bpy.data.images.new(P + name + "_draw", w, h, alpha=False)
    rgba = np.concatenate([np.clip(rgb, 0, 1), np.ones((h, w, 1))], axis=2).astype(np.float32)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = path
    img.file_format = "JPEG"
    img.save(filepath=path, quality=92)
    bpy.data.images.remove(img)
    loaded = bpy.data.images.load(path)
    loaded.name = P + name
    return loaded


def dye_texture():
    """The plate's top: blue layout dye over steel, uneven, with faint grinding marks, and the scribed cross (bright
    lines 0.7 mm wide) at its place on the plate."""
    w, h = 1100, 700  # 10 px per mm
    base = np.array([0.07, 0.20, 0.66])
    y, x = np.mgrid[:h, :w]
    streak = 0.93 + 0.07 * np.sin(x * 0.035 + 3 * np.sin(y * 0.004)) * 0.5 + 0.03 * RNG.random((h, w))
    rgb = np.ones((h, w, 3)) * base * streak[..., None]
    rgb[(y // 40 % 2 == 0) & (RNG.random((h, w)) > 0.995)] += 0.2  # a few flecks
    cx = int(round(w / 2 + CROSS_AT[0] * 10000))
    cy = int(round(h / 2 + CROSS_AT[1] * 10000))
    arm, half = int(CROSS_ARM * 10000), 3
    steel = np.array([0.80, 0.82, 0.85])
    rgb[cy - half:cy + half + 1, cx - arm:cx + arm + 1] = steel
    rgb[cy - arm:cy + arm + 1, cx - half:cx + half + 1] = steel
    return _save("plate_top", rgb)


def knurl_texture():
    """The barrel's grip: red anodised with fine diagonal knurling."""
    n = 256
    y, x = np.mgrid[:n, :n]
    ridge = (((x + y) % 16) < 3) * 0.10 + (((x - y) % 16) < 3) * 0.10
    rgb = np.ones((n, n, 3)) * np.array([0.80, 0.07, 0.05]) * (1.0 - ridge[..., None])
    rgb *= 0.97 + 0.03 * RNG.random((n, n, 1))
    return _save("barrel_knurl", rgb)


class Look:
    def __init__(self, name, size):
        self.name, self.size = name, size
        self.coll = bpy.data.collections.new(P + "look_" + name)
        bpy.context.scene.collection.children.link(self.coll)
        self.root = bpy.data.objects.new(P + name, None)
        self.coll.objects.link(self.root)

    def mesh(self, name, bm, mat, bevel=0.0, smooth=False):
        me = bpy.data.meshes.new(P + self.name + "_" + name)
        bm.to_mesh(me)
        bm.free()
        if smooth:
            me.shade_smooth()
        ob = bpy.data.objects.new(P + self.name + "_" + name, me)
        self.coll.objects.link(ob)
        ob.parent = self.root
        me.materials.append(mat)
        if bevel:
            m = ob.modifiers.new("bevel", "BEVEL")
            m.width, m.segments, m.limit_method = bevel, 2, "ANGLE"
        return ob


def cuboid(lo, hi):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=[hi[i] - lo[i] for i in range(3)], verts=bm.verts)
    bmesh.ops.translate(bm, vec=[(hi[i] + lo[i]) / 2 for i in range(3)], verts=bm.verts)
    return bm


def revolve(profile, segs=48, v_scale=1.0, u_scale=1.0):
    """A surface of revolution about z from a polyline of (radius, z), as an open strip (UVs: round and along)."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    rings = [[bm.verts.new((r * math.cos(2 * math.pi * i / segs), r * math.sin(2 * math.pi * i / segs), z))
              for i in range(segs)] for r, z in profile]
    for k, (a, b) in enumerate(zip(rings, rings[1:])):
        for i in range(segs):
            j = (i + 1) % segs
            f = bm.faces.new((a[i], a[j], b[j], b[i]))
            v0, v1 = profile[k][1] * v_scale, profile[k + 1][1] * v_scale
            for loop, (u, v) in zip(f.loops, ((i, v0), (i + 1, v0), (i + 1, v1), (i, v1))):
                loop[uv].uv = (u / segs * u_scale, v)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


# -- the four looks --------------------------------------------------------------------------------------------------


def build_plate():
    L = Look("plate", PLATE)
    hx, hy, hz = (s / 2 for s in PLATE)
    e = 0.0002
    steel = material("plate_steel", (0.55, 0.56, 0.58), 0.35, 1.0)
    L.mesh("body", cuboid((-hx, -hy, -hz), (hx, hy, hz - e)), steel, bevel=0.0006)
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    vs = [bm.verts.new(p) for p in ((-hx + 0.0007, -hy + 0.0007, hz), (hx - 0.0007, -hy + 0.0007, hz),
                                    (hx - 0.0007, hy - 0.0007, hz), (-hx + 0.0007, hy - 0.0007, hz))]
    f = bm.faces.new(vs)
    for loop, (u, v) in zip(f.loops, ((0.0064, 0.01), (0.9936, 0.01), (0.9936, 0.99), (0.0064, 0.99))):
        loop[uv].uv = (u, v)
    L.mesh("dye", bm, material("plate_dye", (0.07, 0.2, 0.66), 0.5, 0.0, dye_texture()))
    return L


def build_holder():
    L = Look("holder", HOLDER)
    w, h = HOLDER[0], HOLDER[2]
    a, hh = w / 2, h / 2
    k, p = HOLE / 2, POCKET / 2
    dark = material("holder_dark", (0.14, 0.14, 0.16), 0.45, 0.2)
    boxes = {
        "base": ((-a, -a, -hh), (a, a, BASE_Z - hh)),
        "ring_l": ((-a, -a, BASE_Z - hh), (-k, a, FLOOR_Z - hh)), "ring_r": ((k, -a, BASE_Z - hh), (a, a, FLOOR_Z - hh)),
        "ring_f": ((-k, -a, BASE_Z - hh), (k, -k, FLOOR_Z - hh)), "ring_b": ((-k, k, BASE_Z - hh), (k, a, FLOOR_Z - hh)),
        "wall_l": ((-a, -a, FLOOR_Z - hh), (-p, a, hh)), "wall_r": ((p, -a, FLOOR_Z - hh), (a, a, hh)),
        "wall_f": ((-p, -a, FLOOR_Z - hh), (p, -p, hh)), "wall_b": ((-p, p, FLOOR_Z - hh), (p, a, hh)),
    }
    for name, (lo, hi) in boxes.items():
        L.mesh(name, cuboid(lo, hi), dark, bevel=0.0004)
    return L


def build_tip():
    L = Look("tip", TIP)
    z0 = -TIP[2] / 2
    steel = material("tip_steel", (0.78, 0.79, 0.82), 0.22, 1.0)
    hard = material("tip_hard", (0.35, 0.36, 0.40), 0.25, 1.0)
    dark = material("nose_dark", (0.12, 0.12, 0.13), 0.4, 0.8)
    pt, sh, st = TIP_PARTS["point"], TIP_PARTS["shoulder"], TIP_PARTS["stem"]
    L.mesh("tip", revolve([(0.0002, z0), (0.0013, z0 + 0.0004), (0.0013, z0 + pt)]), hard, smooth=True)
    L.mesh("taper", revolve([(0.0013, z0 + pt), (0.0070, z0 + pt + 0.006), (0.0074, z0 + pt + 0.0065)]), dark, smooth=True)
    L.mesh("nose", revolve([(0.0074, z0 + pt + 0.0065), (0.0074, z0 + pt + sh - 0.001), (0.0068, z0 + pt + sh),
                            (0.0045, z0 + pt + sh)]), dark, smooth=True)
    L.mesh("stem", revolve([(0.0045, z0 + pt + sh), (0.0045, z0 + TIP[2] - 0.0005), (0.0035, z0 + TIP[2]),
                            (0.0, z0 + TIP[2])]), steel, smooth=True)
    return L


def build_barrel():
    L = Look("barrel", BARREL)
    z0, z1 = -BARREL[2] / 2, BARREL[2] / 2
    r = 0.0105
    red = material("barrel_red", (0.80, 0.07, 0.05), 0.35, 0.5, knurl_texture())
    dark_steel = material("barrel_collar", (0.30, 0.31, 0.34), 0.3, 1.0)
    black = material("barrel_cap", (0.05, 0.05, 0.055), 0.4, 0.2)
    inner = material("barrel_inner", (0.10, 0.10, 0.11), 0.5, 0.5)
    collar_h, cap_h = 0.005, 0.008
    L.mesh("collar", revolve([(0.0058, z0), (r, z0), (r, z0 + collar_h)]), dark_steel, smooth=True)
    L.mesh("hole", revolve([(0.0058, z0), (0.0058, z0 + SLEEVE_DEPTH), (0.0, z0 + SLEEVE_DEPTH)]), inner, smooth=True)
    L.mesh("sleeve", revolve([(r, z0 + collar_h), (r, z1 - cap_h)], v_scale=60.0, u_scale=4.0), red, smooth=True)
    L.mesh("cap", revolve([(r, z1 - cap_h), (r, z1 - 0.0015), (r - 0.0015, z1), (0.0, z1)]), black, smooth=True)
    return L


# -- export ----------------------------------------------------------------------------------------------------------


def _bounds(look):
    dg = bpy.context.evaluated_depsgraph_get()
    lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
    for ob in look.coll.objects:
        if ob.type != "MESH":
            continue
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        for v in me.vertices:
            p = ob.matrix_world @ v.co
            lo = Vector(map(min, lo, p))
            hi = Vector(map(max, hi, p))
        ev.to_mesh_clear()
    return lo, hi


def export(look):
    bpy.context.view_layer.update()
    lo, hi = _bounds(look)
    half = Vector(look.size) / 2
    over = max(max(-lo[i] - half[i], hi[i] - half[i]) for i in range(3))
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
    """Remove every datablock this script made (named with the prefix), and nothing else."""
    for coll in [c for c in bpy.data.collections if c.name.startswith(P)]:
        for ob in list(coll.objects):
            if ob.name.startswith(P):
                bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.collections.remove(coll)
    for ob in [o for o in bpy.data.objects if o.name.startswith(P)]:
        bpy.data.objects.remove(ob, do_unlink=True)
    for store in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for block in [b for b in store if b.name.startswith(P)]:
            store.remove(block)


BUILDERS = {"plate": build_plate, "holder": build_holder, "tip": build_tip, "barrel": build_barrel}


def build(names=None, hide=True):
    clear()
    looks = [BUILDERS[n]() for n in (names or BUILDERS)]
    out = {look.name: export(look) for look in looks}
    if hide:  # keep the session tidy for the others using it: hide the looks once exported
        for look in looks:
            bpy.context.view_layer.layer_collection.children[look.coll.name].exclude = True
    return out


if __name__ == "__main__":
    print(build())
