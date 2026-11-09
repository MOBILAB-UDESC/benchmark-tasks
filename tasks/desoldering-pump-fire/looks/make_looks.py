"""make_looks.py: model the scene's visual meshes in Blender and export them as USD next to this file.

    blender -b -P looks/make_looks.py          (Blender 4.2 or later, from the task folder)

The physics stays the task file's boxes. Each look is a render-only mesh that ``looks.py`` puts on one of those boxes,
whose plain cuboid is then hidden but still collides. Every look is built around an empty at its box's centre (metres,
z up, the task file's axes) and exported with that empty at the origin. Each look must stay inside its box (checked
when it is exported), because the scored boxes' bounds are read from what is drawn.

- ``barrel.usdc``: the pump's blue barrel with its white label band, grip ring, black end cap and the button's slot on its +y side.
- ``nozzle.usdc``: the white nozzle, a thin cone.
- ``plunger.usdc``: the black knob (3 cm across) and the steel rod under it, on a 8 x 8 x 42 mm collider.
- ``button.usdc``: the white thumb button, a cap and a stem.
- ``block_x.usdc`` and ``block_y.usdc``: the stand's blocks (two along y, two along x) with a rubber pad on top.
- ``board.usdc``: the green circuit board with its traces, pads, holes and silkscreen.

Every datablock is named ``desoldering-pump-fire__...`` so that other tasks sharing a Blender session are not touched.
Textures are drawn here with numpy and saved in ``textures/``. Nothing is downloaded.
"""

import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "textures")
PREFIX = "desoldering-pump-fire__"
RNG = np.random.default_rng(23)

# -- the task file's layout (desoldering_pump_fire.py), metres --------------------------------------------------------
BARREL_SIZE = (0.024, 0.024, 0.12)
NOZZLE_SIZE = (0.005, 0.005, 0.020)
ROD_SIZE = (0.008, 0.008, 0.042)  # the plunger's collider: its rod and the knob's thickness
KNOB_D, KNOB_H = 0.030, 0.012  # the knob is wider than the collider, which only the render shows
BUTTON_SIZE = (0.012, 0.008, 0.014)
BUTTON_Z = 0.015  # the button's centre below the barrel's top
HOLSTER_SIZE = (0.05, 0.05, 0.022)
HOLE = 0.012
BOARD_SIZE = (0.10, 0.08, 0.0016)
JOINT_PITCH = 0.02

_MATS = {}


def material(name, color=(0.5, 0.5, 0.5), rough=0.5, metal=0.0, image=None):
    name = PREFIX + name
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
    if image is not None:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = image
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    m.diffuse_color = (*color, 1.0)
    _MATS[name] = m
    return m


def materials():
    M = {}
    M["blue"] = material("barrel_blue", (0.04, 0.17, 0.78), 0.35)
    M["white"] = material("white_plastic", (0.85, 0.85, 0.83), 0.4)
    M["black"] = material("black_plastic", (0.015, 0.015, 0.017), 0.42)
    M["steel"] = material("steel", (0.66, 0.66, 0.68), 0.3, 1.0)
    M["rubber"] = material("rubber_pad", (0.03, 0.03, 0.032), 0.8)
    M["stand"] = material("stand_grey", (0.12, 0.12, 0.13), 0.55)
    M["solder"] = material("solder", (0.78, 0.79, 0.82), 0.25, 1.0)
    return M


def _save(name, rgb):
    """Save an sRGB image (rows from the bottom, as Blender stores them) as JPEG in textures/ and load it."""
    os.makedirs(TEX, exist_ok=True)
    name = PREFIX + name
    path = os.path.join(TEX, name.replace(PREFIX, "") + ".jpg")
    h, w, _ = rgb.shape
    old = bpy.data.images.get(name)
    if old:
        bpy.data.images.remove(old)
    img = bpy.data.images.new(name, w, h, alpha=False)
    rgba = np.concatenate([np.clip(rgb, 0, 1), np.ones((h, w, 1))], axis=2).astype(np.float32)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = path
    img.file_format = "JPEG"
    img.save()
    bpy.data.images.remove(img)
    return bpy.data.images.load(path)


def board_texture():
    """The board's top: green mask, copper traces, tinned pads for the three joints and silkscreen lines, drawn in
    board coordinates (x along the board's 10 cm, y across its 8 cm), 2400 px for 10 cm."""
    w, h = 2400, 1920
    k = w / BOARD_SIZE[0]
    img = np.zeros((h, w, 3))
    img[:] = np.array([0.075, 0.30, 0.15])
    img *= (0.94 + 0.12 * RNG.random((h // 8 + 1, w // 8 + 1))).repeat(8, 0).repeat(8, 1)[:h, :w, None]
    yy, xx = np.mgrid[0:h, 0:w]
    X, Y = (xx + 0.5) / k - BOARD_SIZE[0] / 2, (yy + 0.5) / k - BOARD_SIZE[1] / 2
    copper = np.zeros((h, w), bool)
    for _ in range(26):  # parallel traces running along x or y
        along_x = RNG.random() < 0.5
        c = RNG.uniform(-0.038, 0.038) if along_x else RNG.uniform(-0.048, 0.048)
        lo, hi = sorted(RNG.uniform(-0.046, 0.046, 2)) if along_x else sorted(RNG.uniform(-0.036, 0.036, 2))
        for i in range(int(RNG.integers(2, 6))):
            off = c + i * 0.0012
            if along_x:
                copper |= (np.abs(Y - off) < 0.0003) & (X > lo) & (X < hi)
            else:
                copper |= (np.abs(X - off) < 0.0003) & (Y > lo) & (Y < hi)
    img[copper] = np.array([0.55, 0.45, 0.20])
    # tinned pads under the three joints, along y at x = 0 (the board's centre)
    for s in (-1, 0, 1):
        pad = (X ** 2 + (Y - s * JOINT_PITCH) ** 2) < 0.0055 ** 2
        img[pad] = np.array([0.72, 0.68, 0.55])
        hole = (X ** 2 + (Y - s * JOINT_PITCH) ** 2) < 0.0016 ** 2
        img[hole] = np.array([0.02, 0.02, 0.02])
        # a trace leaving each pad toward +x
        img[(np.abs(Y - s * JOINT_PITCH) < 0.0004) & (X > 0) & (X < 0.04)] = np.array([0.55, 0.45, 0.20])
    # silkscreen outline of a header and two labels' worth of marks
    silk = np.zeros((h, w), bool)
    silk |= (np.abs(Y) < 0.034) & (np.abs(np.abs(X + 0.02) - 0.01) < 0.0002)
    silk |= (np.abs(X + 0.02) < 0.01) & (np.abs(np.abs(Y) - 0.034) < 0.0002)
    for i in range(12):
        silk |= (np.abs(X - (-0.04 + i * 0.002)) < 0.0003) & (np.abs(Y + 0.033) < 0.0025 * (1 + i % 3))
    img[silk] = np.array([0.92, 0.92, 0.88])
    # mounting holes in the corners
    for sx in (-1, 1):
        for sy in (-1, 1):
            d2 = (X - sx * 0.045) ** 2 + (Y - sy * 0.035) ** 2
            img[d2 < 0.0032 ** 2] = np.array([0.72, 0.68, 0.55])
            img[d2 < 0.0018 ** 2] = np.array([0.03, 0.03, 0.03])
    return _save("board_top", img)


class Look:
    def __init__(self, name, origin=(0, 0, 0), size=None, overhang_xy=False):
        self.name, self.size, self.overhang_xy = name, size, overhang_xy
        self.coll = bpy.data.collections.new(PREFIX + "look_" + name)
        bpy.context.scene.collection.children.link(self.coll)
        self.root = bpy.data.objects.new(PREFIX + name, None)
        self.root.location = Vector(origin)
        self.coll.objects.link(self.root)

    def mesh(self, name, bm, mat, smooth=True):
        me = bpy.data.meshes.new(PREFIX + self.name + "_" + name)
        bm.to_mesh(me)
        bm.free()
        if smooth:
            me.shade_smooth()
        ob = bpy.data.objects.new(PREFIX + self.name + "_" + name, me)
        self.coll.objects.link(ob)
        ob.parent = self.root
        me.materials.append(mat)
        return ob


def bm_cyl(r1, r2, z0, z1, cx=0.0, cy=0.0, seg=48, caps=True):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=caps, segments=seg, radius1=r1, radius2=r2, depth=z1 - z0)
    bmesh.ops.translate(bm, vec=(cx, cy, (z0 + z1) / 2), verts=bm.verts)
    return bm


def bm_box(lo, hi, uv=None):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    c, s = (Vector(lo) + Vector(hi)) / 2, Vector(hi) - Vector(lo)
    for v in bm.verts:
        v.co = c + Vector((v.co.x * s.x, v.co.y * s.y, v.co.z * s.z))
    if uv:
        layer = bm.loops.layers.uv.new("UVMap")
        x0, y0, x1, y1 = uv
        for f in bm.faces:
            for loop in f.loops:
                co = loop.vert.co
                loop[layer].uv = ((co.x - x0) / (x1 - x0), (co.y - y0) / (y1 - y0))
    return bm


def build_barrel(M):
    L = Look("barrel", size=BARREL_SIZE)
    h = BARREL_SIZE[2] / 2
    L.mesh("body", bm_cyl(0.0112, 0.0112, -h, h), M["blue"])
    L.mesh("label", bm_cyl(0.0115, 0.0115, -0.002, 0.034), M["white"])  # a white band
    L.mesh("ring", bm_cyl(0.01195, 0.01195, -h + 0.012, -h + 0.034), M["black"])  # the grip ring
    L.mesh("cap", bm_cyl(0.01195, 0.01195, h - 0.006, h), M["black"])
    L.mesh("foot", bm_cyl(0.0105, 0.0105, -h, -h + 0.004), M["steel"])
    # the button's slot on the +y side: a dark plate, flush with the barrel's flat
    L.mesh("slot", bm_box((-0.0075, 0.0112, h - BUTTON_Z - 0.0095), (0.0075, 0.0120, h - BUTTON_Z + 0.0095)), M["black"],
           smooth=False)
    return L


def build_nozzle(M):
    L = Look("nozzle", size=NOZZLE_SIZE)
    h = NOZZLE_SIZE[2] / 2
    L.mesh("cone", bm_cyl(0.0022, 0.0025, -h, h), M["white"])
    return L


def build_plunger(M):
    """The knob (3 cm across, wider than the collider, centred on it) on the rod, which fills the collider's depth. The
    knob's top is the box's top and its thickness the box's top 12 mm, so the bounds' centre is the box's."""
    L = Look("plunger", size=ROD_SIZE, overhang_xy=True)
    h = ROD_SIZE[2] / 2
    r = KNOB_D / 2
    L.mesh("knob", bm_cyl(r, r, h - KNOB_H, h - 0.003), M["black"])
    L.mesh("dome", bm_cyl(r, 0.011, h - 0.003, h), M["black"])
    L.mesh("rod", bm_cyl(0.0028, 0.0028, -h, h - KNOB_H), M["steel"])
    return L


def build_button(M):
    """The cap on the outer (+y) side of the box, the stem on the barrel's side."""
    L = Look("button", size=BUTTON_SIZE)
    sx, sy, sz = BUTTON_SIZE
    L.mesh("cap", bm_box((-sx / 2, 0.0, -sz / 2), (sx / 2, sy / 2, sz / 2)), M["white"], smooth=False)
    stem = bmesh.new()
    bmesh.ops.create_cone(stem, cap_ends=True, segments=16, radius1=0.0022, radius2=0.0022, depth=sy / 2)
    bmesh.ops.rotate(stem, cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, "X"), verts=stem.verts)
    bmesh.ops.translate(stem, vec=(0, -sy / 4, 0), verts=stem.verts)
    L.mesh("stem", stem, M["steel"])
    return L


def build_block(M, name, size):
    """One of the stand's four dark blocks, with a rubber pad on its top."""
    L = Look(name, size=size)
    sx, sy, sz = size
    z1 = sz / 2
    L.mesh("block", bm_box((-sx / 2, -sy / 2, -sz / 2), (sx / 2, sy / 2, z1 - 0.0015)), M["stand"], smooth=False)
    L.mesh("pad", bm_box((-sx / 2 + 0.0005, -sy / 2 + 0.0005, z1 - 0.0015), (sx / 2 - 0.0005, sy / 2 - 0.0005, z1)),
           M["rubber"], smooth=False)
    return L


def build_holster(M):
    w = (HOLSTER_SIZE[0] - HOLE) / 2
    return [build_block(M, "block_x", (w, HOLSTER_SIZE[1], HOLSTER_SIZE[2])),
            build_block(M, "block_y", (HOLE, w, HOLSTER_SIZE[2]))]


def build_board(M, image):
    L = Look("board", size=BOARD_SIZE)
    mat = material("board_mask", (0.1, 0.3, 0.15), 0.4, image=image)
    sx, sy, sz = BOARD_SIZE
    bm = bm_box((-sx / 2, -sy / 2, -sz / 2), (sx / 2, sy / 2, sz / 2), uv=(-sx / 2, -sy / 2, sx / 2, sy / 2))
    L.mesh("pcb", bm, mat, smooth=False)
    return L


def _local_bounds(look):
    lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
    dg = bpy.context.evaluated_depsgraph_get()
    for ob in look.coll.objects:
        if ob.type != "MESH":
            continue
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        mw = look.root.matrix_world.inverted() @ ev.matrix_world
        for v in me.vertices:
            p = mw @ v.co
            lo = Vector(map(min, lo, p))
            hi = Vector(map(max, hi, p))
        ev.to_mesh_clear()
    return lo, hi


def export(look):
    bpy.context.view_layer.update()
    lo, hi = _local_bounds(look)
    if look.size is not None:
        half = Vector(look.size) / 2
        axes = [2] if look.overhang_xy else range(3)
        if look.overhang_xy and abs(lo[0] + hi[0]) + abs(lo[1] + hi[1]) > 1e-6:
            raise ValueError(f"{look.name}: the overhang must be centred on its box")
        over = max(max(-lo[i] - half[i], hi[i] - half[i]) for i in axes)
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
    """Remove what an earlier run of this script made, and nothing else (every name carries the prefix)."""
    for coll in list(bpy.data.collections):
        if coll.name.startswith(PREFIX):
            for ob in list(coll.objects):
                bpy.data.objects.remove(ob)
            bpy.data.collections.remove(coll)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for item in list(block):
            if item.name.startswith(PREFIX) and item.users == 0:
                block.remove(item)
    _MATS.clear()


def build():
    clear()
    M = materials()
    image = board_texture()
    looks = [build_barrel(M), build_nozzle(M), build_plunger(M), build_button(M), *build_holster(M),
             build_board(M, image)]
    out = {look.name: export(look) for look in looks}
    clear()
    return out


if __name__ == "__main__":
    print(build())
