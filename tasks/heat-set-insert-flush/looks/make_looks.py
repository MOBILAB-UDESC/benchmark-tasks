"""make_looks.py: model the scene's visual meshes in Blender and export them as USD next to this file.

    blender -b -P looks/make_looks.py          (Blender 4.2 or later, from the task folder)

The physics stays the task file's boxes. Each look is a render-only mesh that ``looks.py`` puts on one of those boxes,
whose plain cuboid is then hidden but still collides. Every look is built in its box's own frame (metres, centred on the
box, x along the box, y across, z up), so it turns with the box.

- ``part.usdc``: the white 3D-printed bracket with its 5.6 mm hole in the face (layer lines in the texture).
- ``insert.usdc``: the brass heat-set insert (knurled, a bore through it), inside its 8 x 6 x 6 mm box (checked).
- ``iron.usdc``: the soldering iron (blue handle, steel barrel, a flat pusher tip), inside its box (checked).
- ``stand.usdc``: the iron's stand: a cast base, the handle's rest, the barrel's cradle and a brass-wool tin.

Every datablock is named with the task folder's prefix, so the script can run in a shared Blender session: it removes
only its own datablocks and touches no scene settings. Textures are drawn here with numpy and saved in ``textures/``.
Nothing is downloaded.
"""

import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "textures")
PREFIX = "heat-set-insert-flush__"
RNG = np.random.default_rng(5)

# -- the task file's layout (heat_set_insert_flush.py), metres, in the boxes' own frames --------------------------------
PART_SIZE = (0.03, 0.05, 0.08)
HOLE_DRAWN = 0.0056
HOLE_DEPTH = 0.0105
HOLE_Z = 0.026  # the hole's height above the bracket's centre
INS_L = 0.008
INS_W = 0.006
IRON_SIZE = (0.22, 0.02, 0.02)
STAND_BASE = (0.19, 0.07, 0.012)
IRON_ABOVE_BASE = 0.049  # the iron's axis above the base box's centre
REST_S = (-0.095, -0.075)
CRADLE_S = (0.02, 0.06)


def nm(name):
    return PREFIX + name


def clear():
    """Remove every datablock this script made (its prefix), and nothing else."""
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.images,
                 bpy.data.collections):
        for block in [b for b in coll if b.name.startswith(PREFIX)]:
            coll.remove(block)
    _MATS.clear()


_MATS = {}


def material(name, color=(0.5, 0.5, 0.5), rough=0.5, metal=0.0, image=None):
    if name in _MATS:
        return _MATS[name]
    m = bpy.data.materials.new(nm(name))
    try:
        m.use_nodes = True
    except AttributeError:  # Blender 5: always on
        pass
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
    _MATS[name] = m
    return m


def materials():
    M = {}
    M["brass"] = material("brass", (0.95, 0.55, 0.10), 0.22, 1.0)  # the one gold thing on the table
    M["steel"] = material("steel", (0.62, 0.63, 0.65), 0.25, 1.0)
    M["chrome"] = material("chrome", (0.85, 0.86, 0.88), 0.12, 1.0)
    M["handle"] = material("iron_handle", (0.02, 0.12, 0.55), 0.55)  # the one blue thing
    M["handle_dark"] = material("iron_handle_dark", (0.015, 0.015, 0.02), 0.5)
    M["tip"] = material("iron_tip", (0.72, 0.72, 0.70), 0.2, 1.0)
    M["black"] = material("black_plastic", (0.02, 0.02, 0.022), 0.45)
    M["dark"] = material("hole_dark", (0.12, 0.12, 0.12), 0.8)
    M["rubber"] = material("rubber", (0.02, 0.02, 0.02), 0.8)
    M["wool"] = material("brass_wool", (0.75, 0.52, 0.18), 0.5, 0.7)
    return M


# -- textures --------------------------------------------------------------------------------------------------------


def _noise(h, w, cell):
    gh, gw = h // cell + 2, w // cell + 2
    g = RNG.random((gh, gw))
    ys, xs = np.linspace(0, gh - 2, h), np.linspace(0, gw - 2, w)
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    a, b = g[y0][:, x0], g[y0][:, x0 + 1]
    c, d = g[y0 + 1][:, x0], g[y0 + 1][:, x0 + 1]
    return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy


def _save(name, rgb):
    """Save an sRGB image (rows from the bottom, as Blender stores them) as JPEG in textures/ and load it."""
    os.makedirs(TEX, exist_ok=True)
    path = os.path.join(TEX, name + ".jpg")
    h, w, _ = rgb.shape
    img = bpy.data.images.new(nm(name + "_draw"), w, h, alpha=False)
    rgba = np.concatenate([np.clip(rgb, 0, 1), np.ones((h, w, 1))], axis=2).astype(np.float32)
    img.pixels.foreach_set(rgba.ravel())
    img.file_format = "JPEG"
    img.save(filepath=path, quality=92)
    bpy.data.images.remove(img)
    loaded = bpy.data.images.load(path)
    loaded.name = nm(name)
    return loaded


def print_texture():
    """White PLA with fine horizontal layer lines (0.2 mm layers over 8 cm, so 400 lines) and a little grain."""
    H, W = 1024, 512
    rows = np.arange(H)[:, None]
    lines = 0.5 + 0.5 * np.sin(rows * 2 * math.pi / 10.24)  # a line every 10 pixels
    g = 0.90 - 0.05 * lines + 0.03 * (_noise(H, W, 3) - 0.5) + 0.02 * (_noise(H, W, 40) - 0.5)
    rgb = np.stack([g, g, g * 0.985], axis=2)
    return _save("print", rgb)


def cast_texture():
    H, W = 256, 512
    g = 0.85 + 0.2 * _noise(H, W, 2) + 0.08 * _noise(H, W, 30)
    rgb = np.array([0.20, 0.20, 0.22])[None, None, :] * g[..., None]
    return _save("cast", rgb)


# -- geometry --------------------------------------------------------------------------------------------------------


class Look:
    """One exported file: a collection with an empty at the origin (the box's centre) and its meshes."""

    def __init__(self, name, size=None):
        self.name, self.size = name, size
        self.coll = bpy.data.collections.new(nm("look_" + name))
        bpy.context.scene.collection.children.link(self.coll)
        self.root = bpy.data.objects.new(nm(name), None)
        self.coll.objects.link(self.root)

    def mesh(self, name, bm, mat, bevel=0.0, smooth=False):
        me = bpy.data.meshes.new(nm(f"{self.name}_{name}"))
        bm.to_mesh(me)
        bm.free()
        if smooth:
            me.shade_smooth()
        ob = bpy.data.objects.new(nm(f"{self.name}_{name}"), me)
        self.coll.objects.link(ob)
        ob.parent = self.root
        me.materials.append(mat)
        if bevel:
            m = ob.modifiers.new("bevel", "BEVEL")
            m.width, m.segments, m.limit_method = bevel, 2, "ANGLE"
        return ob


def bm_box(lo, hi):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    c, s = (Vector(lo) + Vector(hi)) / 2, Vector(hi) - Vector(lo)
    for v in bm.verts:
        v.co = c + Vector((v.co.x * s.x, v.co.y * s.y, v.co.z * s.z))
    return bm


def _axis_matrix(axis):
    return {"Z": Matrix.Identity(4), "Y": Matrix.Rotation(-math.pi / 2, 4, "X"),
            "X": Matrix.Rotation(math.pi / 2, 4, "Y")}[axis]


def bm_cyl(center, r, h, axis="Z", segs=32, r2=None):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r, radius2=r if r2 is None else r2, depth=h,
                          matrix=Matrix.Translation(Vector(center)) @ _axis_matrix(axis))
    return bm


def bm_lathe(profile, axis="X", segs=40):
    """A surface of revolution: ``profile`` is [(t along the axis, radius), ...] in order, capped at its ends unless
    it ends where it starts (a ring)."""
    bm = bmesh.new()
    rings = []
    for t, r in profile:
        if r <= 1e-7:
            rings.append([bm.verts.new((0.0, 0.0, t))])
        else:
            rings.append([bm.verts.new((r * math.cos(a), r * math.sin(a), t))
                          for a in np.linspace(0, 2 * math.pi, segs, endpoint=False)])
    for a, b in zip(rings, rings[1:]):
        if len(a) == 1 and len(b) == 1:
            continue
        if len(a) == 1:
            for i in range(segs):
                bm.faces.new((a[0], b[i], b[(i + 1) % segs]))
        elif len(b) == 1:
            for i in range(segs):
                bm.faces.new((a[i], b[0], a[(i + 1) % segs]))
        else:
            for i in range(segs):
                j = (i + 1) % segs
                bm.faces.new((a[i], b[i], b[j], a[j]))
    if tuple(profile[0]) == tuple(profile[-1]):  # a closed profile: a ring, no caps
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-9)
    else:
        for ring in (rings[0], rings[-1]):
            if len(ring) > 2:
                bm.faces.new(ring)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.transform(_axis_matrix(axis))
    return bm


# -- the bracket -------------------------------------------------------------------------------------------------------


def build_part(M):
    """A block with a round hole in its -x face: the block as a box with UVs, the hole as a boolean cut."""
    L = Look("part", PART_SIZE)
    sx, sy, sz = PART_SIZE
    bm = bm_box((-sx / 2, -sy / 2, -sz / 2), (sx / 2, sy / 2, sz / 2))
    layer = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:  # layer lines run along the sides: v follows the height
        for loop in f.loops:
            c = loop.vert.co
            loop[layer].uv = ((c.x + c.y) / sy + 0.5, c.z / sz + 0.5)
    mat = material("print_white", (0.90, 0.90, 0.89), 0.55, image=print_texture())
    block = L.mesh("block", bm, mat, bevel=0.0006)
    # the hole: a cylinder along x from the face (x = -sx/2) to its bottom, cut by a boolean, then applied
    cut = bpy.data.objects.new(nm("part_cut"), bpy.data.meshes.new(nm("part_cut_mesh")))
    bmc = bm_cyl((-sx / 2 + HOLE_DEPTH / 2 - 0.001, 0.0, HOLE_Z), HOLE_DRAWN / 2, HOLE_DEPTH + 0.002, "X", 48)
    bmc.to_mesh(cut.data)
    bmc.free()
    L.coll.objects.link(cut)
    mod = block.modifiers.new("hole", "BOOLEAN")
    mod.operation, mod.object, mod.solver = "DIFFERENCE", cut, "EXACT"
    # put the bevel after the cut: reorder so the boolean runs first
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    new = bpy.data.meshes.new_from_object(block.evaluated_get(dg))
    new.name = nm("part_block_cut")
    block.modifiers.clear()
    block.data = new
    bpy.data.objects.remove(cut)
    # a dark floor at the hole's bottom hides the open end of the cut
    L.mesh("hole_floor", bm_cyl((-sx / 2 + HOLE_DEPTH - 0.0004, 0.0, HOLE_Z), HOLE_DRAWN / 2 + 0.0002, 0.0008, "X", 48),
           M["dark"])
    return L


# -- the insert --------------------------------------------------------------------------------------------------------


def build_insert(M):
    """A knurled brass sleeve, bore through it, its top (the end that takes the iron) at -x."""
    L = Look("insert", (INS_L, INS_W, INS_W))
    h, ro, ri = INS_L / 2, INS_W / 2 - 0.0001, 0.002
    prof = [(-h, ri), (-h, ro - 0.0005), (-h + 0.0004, ro)]
    n = 7
    for k in range(n):  # knurl bands: ridged, then a plain band between
        t0 = -h + 0.0012 + k * (INS_L - 0.0030) / n
        t1 = t0 + (INS_L - 0.0030) / n * 0.55
        prof += [(t0, ro), (t0 + 0.00005, ro - 0.00035), (t1, ro - 0.00035), (t1 + 0.00005, ro)]
    prof += [(h - 0.0004, ro), (h, ro - 0.0005), (h, ri), (-h, ri)]
    L.mesh("sleeve", bm_lathe(prof, "X", 28), M["brass"], smooth=False)
    return L


# -- the soldering iron ------------------------------------------------------------------------------------------------


def build_iron(M):
    L = Look("iron", IRON_SIZE)
    x0 = -IRON_SIZE[0] / 2
    grip = bm_lathe([(x0, 0.0), (x0, 0.0062), (x0 + 0.004, 0.0075), (x0 + 0.012, 0.0085), (x0 + 0.03, 0.0088),
                     (x0 + 0.07, 0.0099), (x0 + 0.09, 0.0099), (x0 + 0.104, 0.0088), (x0 + 0.11, 0.0072),
                     (x0 + 0.11, 0.0)], "X", 48)
    L.mesh("grip", grip, M["handle"], smooth=True)
    for i, x in enumerate(np.arange(x0 + 0.074, x0 + 0.102, 0.004)):  # finger rings at the front of the grip
        L.mesh(f"ring_{i}", bm_cyl((x, 0, 0), 0.00995, 0.0012, "X", 48), M["handle_dark"], smooth=True)
    L.mesh("cap", bm_cyl((x0 + 0.003, 0, 0), 0.0064, 0.006, "X", 40, r2=0.0076), M["handle_dark"], smooth=True)
    L.mesh("nut", bm_cyl((0.004, 0, 0), 0.0036, 0.008, "X", 6), M["steel"], bevel=0.0004)
    L.mesh("barrel", bm_cyl((0.0325, 0, 0), 0.0036, 0.065, "X", 40), M["steel"], smooth=True)
    L.mesh("collar", bm_cyl((0.0665, 0, 0), 0.0038, 0.003, "X", 40), M["steel"], smooth=True)
    # the pusher tip: a 2.6 mm round pin with a flat face and a short pilot, inside the 2.6 mm square collider
    tip = bm_lathe([(0.068, 0.0), (0.068, 0.0013), (0.1085, 0.0013), (0.11, 0.0011), (0.11, 0.0)], "X", 32)
    L.mesh("tip", tip, M["tip"], smooth=True)
    return L


# -- the stand -------------------------------------------------------------------------------------------------------


def build_stand(M):
    L = Look("stand")
    bx, by, bz = STAND_BASE
    za = IRON_ABOVE_BASE
    cast = material("cast_tex", (0.2, 0.2, 0.22), 0.6, 0.3, image=cast_texture())
    base = bm_box((-bx / 2, -by / 2, -bz / 2), (bx / 2, by / 2, bz / 2))
    layer = base.loops.layers.uv.new("UVMap")
    for f in base.faces:
        for loop in f.loops:
            loop[layer].uv = ((loop.vert.co.x + loop.vert.co.z) / bx, loop.vert.co.y / by + 0.5)
    L.mesh("base", base, cast, bevel=0.003)
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        L.mesh(f"foot_{i}", bm_cyl((sx * (bx / 2 - 0.01), sy * (by / 2 - 0.01), -bz / 2 - 0.0005), 0.005, 0.001),
               M["rubber"], smooth=True)
    # a brass-wool tin past the cradle, under the tip
    L.mesh("tin", bm_cyl((0.078, 0.0, bz / 2 + 0.008), 0.014, 0.016, "Z", 40), M["steel"], smooth=True)
    L.mesh("wool", bm_cyl((0.078, 0.0, bz / 2 + 0.0165), 0.0125, 0.003, "Z", 24), M["wool"])
    rest_top = za - 0.01
    s0, s1 = REST_S
    L.mesh("rest_post", bm_box((s0 + 0.003, -0.006, bz / 2), (s1 - 0.003, 0.006, rest_top - 0.002)), M["chrome"],
           bevel=0.001)
    L.mesh("rest_saddle", bm_box((s0, -0.008, rest_top - 0.003), (s1, 0.008, rest_top)), M["chrome"], bevel=0.0008)
    c0, c1 = CRADLE_S
    floor_top = za - 0.004
    L.mesh("cradle_post", bm_box((0.03, -0.004, bz / 2), (0.05, 0.004, floor_top - 0.003)), M["chrome"], bevel=0.001)
    L.mesh("cradle_floor", bm_box((c0, -0.008, floor_top - 0.003), (c1, 0.008, floor_top)), M["chrome"], bevel=0.0008)
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
    if look.size is not None:
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


def build(keep=False):
    """Model and export every look. ``keep`` leaves the models in the scene to look at; otherwise they are removed."""
    clear()
    M = materials()
    looks = [build_part(M), build_insert(M), build_iron(M), build_stand(M)]
    out = {look.name: export(look) for look in looks}
    if not keep:
        clear()
    return out


if __name__ == "__main__":
    for name, info in build().items():
        print(name, info)
