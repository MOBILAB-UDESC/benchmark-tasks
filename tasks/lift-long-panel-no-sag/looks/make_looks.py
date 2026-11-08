"""make_looks.py: model the scene's visual meshes in Blender and export them as USD next to this file.

    blender -b -P looks/make_looks.py          (Blender 4.2 or later)

The physics stays the task file's boxes and the panel's 20 segment boxes. Each look is a render-only mesh that
``panelkit.py`` puts on a box, which is then hidden but still collides.

- ``panel_00.usdc`` .. ``panel_19.usdc``: the circuit-board panel, one file per segment, each modelled around its
  segment's centre in the segment's frame (x along the panel, y across, z up) and filling its 3 x 6 x 0.6 cm box
  exactly (checked). One texture runs over all of them: six small boards in one panel, with traces, pads,
  parts, silkscreen, V-score lines, tooling holes and fiducials.
- ``bench.usdc``: the black bench rail and its three legs, around the bar's centre.
- ``rack.usdc``: the orange rack's rail and its three legs, around the bar's centre.

Everything is prefixed ``lift-long-panel-no-sag__`` so it can share a Blender session with other tasks. Textures are
drawn here with numpy and saved in ``textures/``. Nothing is downloaded.
"""

import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "textures")
PREFIX = "lift-long-panel-no-sag__"
RNG = np.random.default_rng(20)

# -- the task file's layout (lift_long_panel_no_sag.py), metres -------------------------------------------------------
PANEL_LENGTH, PANEL_WIDTH, PANEL_THICK, SEGMENTS = 0.60, 0.06, 0.006, 20
SEG = PANEL_LENGTH / SEGMENTS
GAP = 0.0004
BAR = (0.036, 0.64, 0.012)
LEG = 0.02
LEG_Y = (-0.28, 0.0, 0.28)
TOP = 0.7475
BENCH_TOP, RACK_TOP = TOP + 0.03, TOP + 0.08

_MATS = {}


def material(name, color, rough=0.5, metal=0.0, image=None):
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
        tex.extension = "EXTEND"
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    m.diffuse_color = (*color, 1.0)
    _MATS[name] = m
    return m


# -- the panel's texture ---------------------------------------------------------------------------------------------


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
    old = bpy.data.images.get(PREFIX + name)
    if old:
        bpy.data.images.remove(old)
    img = bpy.data.images.new(PREFIX + name, w, h, alpha=False)
    rgba = np.concatenate([np.clip(rgb, 0, 1), np.ones((h, w, 1))], axis=2).astype(np.float32)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = path
    img.file_format = "JPEG"
    bpy.context.scene.render.image_settings.quality = 90
    img.save()
    bpy.data.images.remove(img)
    image = bpy.data.images.load(path)
    image.name = PREFIX + name
    return image


def pcb_texture():
    """The panel's top face: u along its 60 cm, v across its 6 cm. Six 10 x 6 cm boards, one after another."""
    W, H = 4800, 480  # 8 px per mm
    k = 8.0
    mask = {key: np.zeros((H, W), bool) for key in ("trace", "pad", "silk", "hole", "tan", "black", "score", "gold")}
    yy, xx = np.mgrid[0:H, 0:W]

    def rect(m, x0, y0, x1, y1):  # millimetres
        mask[m][max(int(y0 * k), 0):max(int(y1 * k), 0), max(int(x0 * k), 0):max(int(x1 * k), 0)] = True

    def disc(m, cx, cy, r):
        mask[m][(xx - cx * k) ** 2 + (yy - cy * k) ** 2 <= (r * k) ** 2] = True

    for b in range(6):
        x0 = b * 100.0
        # traces: horizontal and vertical runs, 0.2 to 0.4 mm
        for _ in range(26):
            y = RNG.uniform(8, 52)
            xa = x0 + RNG.uniform(6, 60)
            xb = xa + RNG.uniform(10, 30)
            wd = RNG.choice([0.25, 0.3, 0.4])
            rect("trace", xa, y, xb, y + wd)
            rect("trace", xb, y, xb + wd, y + wd + RNG.uniform(3, 10))
        # a QFP-like chip with pads, resistor/capacitor pairs, a connector row of gold fingers
        cx, cy = x0 + 55, 30.0
        rect("black", cx - 5, cy - 5, cx + 5, cy + 5)
        for i in range(12):
            t = -4.5 + i * 0.82
            rect("pad", cx + t, cy - 7, cx + t + 0.4, cy - 5)
            rect("pad", cx + t, cy + 5, cx + t + 0.4, cy + 7)
            rect("pad", cx - 7, cy + t, cx - 5, cy + t + 0.4)
            rect("pad", cx + 5, cy + t, cx + 7, cy + t + 0.4)
        for _ in range(34):
            px, py = x0 + RNG.uniform(8, 92), RNG.uniform(9, 51)
            if abs(px - cx) < 9 and abs(py - cy) < 9:
                continue
            horiz = RNG.random() < 0.5
            lw, wd = (2.0, 1.2) if RNG.random() < 0.7 else (3.2, 1.6)
            if horiz:
                rect("pad", px, py, px + 0.7, py + wd)
                rect("pad", px + lw - 0.7, py, px + lw, py + wd)
                rect("tan" if RNG.random() < 0.6 else "black", px + 0.7, py, px + lw - 0.7, py + wd)
            else:
                rect("pad", px, py, px + wd, py + 0.7)
                rect("pad", px, py + lw - 0.7, px + wd, py + lw)
                rect("tan" if RNG.random() < 0.6 else "black", px, py + 0.7, px + wd, py + lw - 0.7)
        for i in range(10):  # edge connector fingers
            rect("gold", x0 + 8 + i * 2.2, 1.0, x0 + 8 + i * 2.2 + 1.4, 6.0)
        # silkscreen: an outline round the chip, reference text as glyph blocks, a pin-1 dot
        rect("silk", cx - 8, cy - 8, cx + 8, cy - 7.7)
        rect("silk", cx - 8, cy + 7.7, cx + 8, cy + 8)
        rect("silk", cx - 8, cy - 8, cx - 7.7, cy + 8)
        rect("silk", cx + 7.7, cy - 8, cx + 8, cy + 8)
        for i in range(4):
            rect("silk", x0 + 10 + i * 1.6, 52, x0 + 10 + i * 1.6 + 1.0, 54.2)
        disc("silk", cx - 6.5, cy - 6.5, 0.6)
        # V-score line between this board and the next, and tooling holes and fiducials at the board's corners
        rect("score", x0 + 99.7, 0, x0 + 100.0, 60)
        for hx, hy in ((x0 + 4, 4), (x0 + 96, 4), (x0 + 4, 56), (x0 + 96, 56)):
            disc("hole", hx, hy, 1.6)
        disc("pad", x0 + 12, 56, 0.5)
        disc("pad", x0 + 88, 4, 0.5)

    base = np.array([0.03, 0.20, 0.78])
    rgb = np.ones((H, W, 3)) * base
    rgb *= (0.93 + 0.14 * _noise(H, W, 96))[..., None]
    rgb[mask["trace"]] = np.array([0.05, 0.28, 0.92])
    rgb[mask["pad"]] = np.array([0.78, 0.78, 0.76])
    rgb[mask["gold"]] = np.array([0.88, 0.70, 0.30])
    rgb[mask["tan"]] = np.array([0.62, 0.50, 0.34])
    rgb[mask["black"]] = np.array([0.04, 0.04, 0.045])
    rgb[mask["silk"]] = np.array([0.93, 0.93, 0.90])
    rgb[mask["score"]] = np.array([0.02, 0.10, 0.40])
    rgb[mask["hole"]] = np.array([0.03, 0.03, 0.03])
    rgb *= (0.97 + 0.06 * _noise(H, W, 3))[..., None]
    return _save("pcb_top", rgb)


# -- helpers ---------------------------------------------------------------------------------------------------------


def _new_object(name, bm, mats, collection):
    mesh = bpy.data.meshes.new(PREFIX + name)
    bm.to_mesh(mesh)
    bm.free()
    for m in mats:
        mesh.materials.append(m)
    ob = bpy.data.objects.new(PREFIX + name, mesh)
    collection.objects.link(ob)
    return ob


def _box(bm, centre, size, mat_index=0):
    """Add an axis-aligned box to bm and return its faces."""
    cx, cy, cz = centre
    sx, sy, sz = (s / 2 for s in size)
    verts = [bm.verts.new((cx + dx * sx, cy + dy * sy, cz + dz * sz)) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]
    faces = []
    for idx in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
        f = bm.faces.new([verts[i] for i in idx])
        f.material_index = mat_index
        faces.append(f)
    bmesh.ops.recalc_face_normals(bm, faces=faces)
    return faces


def _collection(name):
    coll = bpy.data.collections.new(PREFIX + name)
    bpy.context.scene.collection.children.link(coll)
    return coll


def _bounds(coll):
    pts = []
    for ob in coll.objects:
        pts += [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    return Vector(min(p[i] for p in pts) for i in range(3)), Vector(max(p[i] for p in pts) for i in range(3))


def _export(coll, filename, root=""):
    bpy.context.view_layer.update()
    path = os.path.join(HERE, filename)
    bpy.ops.wm.usd_export(filepath=path, collection=coll.name, export_materials=True, generate_preview_surface=True,
                          export_textures_mode="KEEP", relative_paths=True, root_prim_path=root, export_lights=False,
                          export_cameras=False, convert_world_material=False, export_custom_properties=False,
                          author_blender_name=False, evaluation_mode="RENDER", export_uvmaps=True,
                          export_normals=True)
    return path


# -- the looks -------------------------------------------------------------------------------------------------------


def build_panel():
    """One collection and one file per segment, panel_00.usdc .. panel_19.usdc: a reference to a part of a file
    would leave its materials behind, so each file holds one segment with its own materials."""
    image = pcb_texture()
    top = material("pcb_top", (0.03, 0.2, 0.8), 0.35, 0.0, image)
    edge = material("pcb_edge", (0.002, 0.012, 0.14), 0.5)
    length = SEG - GAP
    out = []
    for i in range(SEGMENTS):
        coll = _collection(f"look_panel_{i:02d}")
        bm = bmesh.new()
        faces = _box(bm, (0, 0, 0), (length, PANEL_WIDTH, PANEL_THICK))
        uv = bm.loops.layers.uv.new("UVMap")
        u0, u1 = i / SEGMENTS, (i + 1) / SEGMENTS
        for f in faces:
            if f.normal.z > 0.9:  # the top face carries the texture
                f.material_index = 0
                for loop in f.loops:
                    x, y = loop.vert.co.x, loop.vert.co.y
                    loop[uv].uv = (u0 + (x / length + 0.5) * (u1 - u0), y / PANEL_WIDTH + 0.5)
            else:
                f.material_index = 1
                for loop in f.loops:
                    loop[uv].uv = (0.0, 0.0)
        ob = _new_object(f"seg_{i:02d}", bm, [top, edge], coll)
        lo, hi = _bounds(coll)
        # each segment fills its box: 3 x 6 x 0.6 cm, within 0.01 mm
        over = max(hi.x - length / 2, hi.y - PANEL_WIDTH / 2, hi.z - PANEL_THICK / 2,
                   -lo.x - length / 2, -lo.y - PANEL_WIDTH / 2, -lo.z - PANEL_THICK / 2)
        if over > 1e-5:
            raise ValueError(f"panel look {i} sticks {over * 1000:.3f} mm out of its box")
        out.append((coll, f"panel_{i:02d}.usdc", {"low": tuple(lo), "high": tuple(hi)}))
    return out


def _rail(name, top, colour, bar_rough, metal):
    """A bar along y with three legs, around the bar's centre (the bar's box centre is the origin)."""
    bar_m = material(name + "_bar", colour, bar_rough, metal)
    leg_m = material(name + "_leg", (0.03, 0.03, 0.035), 0.4, 0.8)
    coll = _collection("look_" + name)
    bm = bmesh.new()
    _box(bm, (0, 0, 0), BAR, 0)
    ob = _new_object("bar", bm, [bar_m], coll)
    h = top - BAR[2] - TOP
    for i, y in enumerate(LEG_Y):
        bm = bmesh.new()
        _box(bm, (0, y, -BAR[2] / 2 - h / 2), (LEG, LEG, h), 0)
        _new_object(f"leg{i}", bm, [leg_m], coll)
    lo, hi = _bounds(coll)
    return coll, name + ".usdc", {"low": tuple(lo), "high": tuple(hi), "bar": ob.name}


def build():
    clear()
    out = {}
    looks = [*build_panel(), _rail("bench", BENCH_TOP, (0.006, 0.006, 0.008), 0.45, 0.6),
             _rail("rack", RACK_TOP, (0.85, 0.15, 0.004), 0.4, 0.2)]
    for coll, filename, info in looks:
        # every file holds one default prim, "/look", that the task's boxes and segments reference
        out[filename] = {"file": os.path.basename(_export(coll, filename, "/look")), **info}
    return {"files": len(out), "last": out[filename]}


def clear():
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


if __name__ == "__main__":
    print(build())
