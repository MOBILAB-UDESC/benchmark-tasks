"""make_looks.py: model an enclosed desktop 3D printer and its props in Blender and export them as USD next to this
file.

    blender -b -P looks/make_looks.py          (Blender 4.2 or later, from the task folder)

The same script is in the folders of two tasks (scrape-print-off-bed and open-printer-door) and makes the same
printer in both. The physics stays the task file's boxes. Each look is a render-only mesh that ``looks.py`` puts on one
of those boxes, whose plain cuboid is then hidden but still collides. Every look is built in its box's own frame
(metres, the box's centre at the origin, z up) and must fit inside the box (checked when exported). Positions below
are written in the station frame (x forward, y left, h above the table top) and moved into each box's frame.

- ``printer.usdc``: the printer without its door, plate and print head. A 36 x 35 x 40 cm enclosed FDM printer: a
  light grey base with a sloped front and a small screen, a black aluminium frame, light grey side and back panels, a glass
  roof, a light under the front beam, the black heated bed low inside, the gantry with its rails and belts.
- ``door.usdc``: the hinged front door, tinted clear acrylic in a black border, with a brushed aluminium bar handle.
- ``head.usdc``: the print head, a light grey shroud with a black fan and a brass nozzle.
- ``plate.usdc``: the spring-steel build plate, textured gold PEI on top (24 x 24 cm, 1 mm).
- ``print.usdc``: a green printed part, an open box with layer lines and a chamfered first layer (6 x 6 x 3 cm).
- ``tall_print.usdc``: an orange printed vase, twisted, 5 x 5 x 9 cm.
- ``scraper.usdc``: the print scraper, a bevelled steel blade in a black handle (4 x 22 x 2.2 cm, y along it).

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
RNG = np.random.default_rng(12)

# -- the boxes (the task files), station frame: x forward, y left, heights above the table, metres ---------------------
PRINTER_C = (0.49, 0.0, 0.20)
PRINTER = (0.36, 0.35, 0.40)
DOOR_C = (0.3555, 0.0, 0.229)
DOOR = (0.047, 0.326, 0.288)
HEAD_C = (0.457, 0.07, 0.27)
HEAD = (0.058, 0.06, 0.08)
PLATE = (0.24, 0.24, 0.001)
PRINT = (0.06, 0.06, 0.03)
FOOT_H = 0.002  # the print's chamfered first layers
TALL = (0.05, 0.05, 0.09)
SCRAPER = (0.04, 0.22, 0.022)
HANDLE_LEN, BLADE_LEN, BLADE_T = 0.12, 0.10, 0.001

WANT = {
    "scrape-print-off-bed": ["printer", "door", "head", "plate", "print", "scraper"],
    "open-printer-door": ["printer", "door", "head", "plate", "tall_print"],
}


def srgb(c):
    return tuple(((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c)


# -- materials -------------------------------------------------------------------------------------------------------


def material(name, color, rough=0.5, metal=0.0, image=None, alpha=1.0, emit=None, strength=0.0):
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
    bsdf.inputs["Alpha"].default_value = alpha
    if alpha < 1.0 and hasattr(m, "blend_method"):
        m.blend_method = "BLEND"
    if emit is not None:
        bsdf.inputs["Emission Color"].default_value = (*emit, 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    if image is not None:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.extension = "REPEAT"
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        if emit is not None:
            links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
    m.diffuse_color = (*color, alpha)
    return m


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
    for old in [i for i in bpy.data.images if i.name.startswith(P + name)]:
        bpy.data.images.remove(old)
    img = bpy.data.images.new(P + name + "_draw", w, h, alpha=False)
    rgba = np.concatenate([np.clip(rgb, 0, 1), np.ones((h, w, 1))], axis=2).astype(np.float32)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = path
    img.file_format = "JPEG"
    img.save(filepath=path, quality=90)
    bpy.data.images.remove(img)
    loaded = bpy.data.images.load(path)
    loaded.name = P + name
    return loaded


def pei_texture():
    """Textured gold PEI: a powder coat of fine light and dark specks over a warm gold, a little uneven."""
    n = 1024
    base = np.array([0.74, 0.55, 0.22])
    shade = 0.92 + 0.08 * _noise(n, n, 96)[..., None]
    rgb = base * shade
    speck = RNG.random((n, n))
    rgb[speck > 0.93] *= 1.18
    rgb[speck < 0.05] *= 0.72
    fine = 0.97 + 0.06 * RNG.random((n, n, 1))
    return _save("plate_top", rgb * fine)


def layer_texture(name, color, h=500):
    """A print's sides: PLA in 0.2 mm layers (one texture repeat is 1 cm tall, 50 layers)."""
    w = 64
    base = np.array(srgb(color)) ** (1 / 2.2)
    rows = np.arange(h) % 10
    ridge = 1.0 - 0.10 * (rows >= 8) + 0.04 * (rows <= 1)
    rgb = np.ones((h, w, 3)) * base * ridge[:, None, None]
    rgb *= 0.98 + 0.04 * RNG.random((h, w, 1))
    return _save(name, rgb)


def grip_texture():
    """The handle's rubber: black with fine diagonal ridges."""
    n = 256
    y, x = np.mgrid[:n, :n]
    ridge = ((x + y) % 16 < 3) * 0.05
    rgb = np.ones((n, n, 3)) * (0.045 + ridge[..., None]) + 0.01 * RNG.random((n, n, 1))
    return _save("handle_grip", rgb)


def screen_texture():
    """The printer's small screen: a dark blue page with a progress bar at 100 %, two lines of text as bars and a
    nozzle and bed temperature row. No words, no logo."""
    h, w = 120, 240
    rgb = np.ones((h, w, 3)) * np.array([0.03, 0.07, 0.16])
    def rect(r0, r1, c0, c1, col):
        rgb[h - r1:h - r0, c0:c1] = col
    rect(14, 24, 16, 150, [0.85, 0.88, 0.92])  # a title line
    rect(34, 40, 16, 110, [0.55, 0.62, 0.72])
    rect(56, 70, 16, 224, [0.15, 0.20, 0.30])  # the progress bar's track
    rect(56, 70, 16, 224, [0.20, 0.78, 0.40])  # full: the print is finished
    for c0 in (16, 128):  # two temperature tiles
        rect(84, 106, c0, c0 + 96, [0.08, 0.14, 0.26])
        rect(90, 100, c0 + 10, c0 + 60, [0.85, 0.88, 0.92])
    return _save("screen", rgb)


def vent_texture():
    """A side panel: matte dark grey with a grid of vent slots near the back."""
    h, w = 256, 256
    rgb = np.ones((h, w, 3)) * 0.66 * (0.97 + 0.03 * _noise(h, w, 32)[..., None])
    for r in range(30, 110, 12):
        for c in range(150, 236, 30):
            rgb[r:r + 5, c:c + 24] = 0.12
    return _save("side_panel", rgb)


# -- building --------------------------------------------------------------------------------------------------------


class Look:
    def __init__(self, name, size, centre):
        self.name, self.size, self.c = name, size, centre
        self.coll = bpy.data.collections.new(P + "look_" + name)
        bpy.context.scene.collection.children.link(self.coll)
        self.root = bpy.data.objects.new(P + name, None)
        self.coll.objects.link(self.root)
        self.n = 0

    def local(self, x, y, h):
        """A station-frame point in this box's frame."""
        return (x - self.c[0], y - self.c[1], h - self.c[2])

    def mesh(self, name, bm, mat, bevel=0.0, smooth=False):
        self.n += 1
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

    def box(self, name, x0, x1, y0, y1, h0, h1, mat, bevel=0.0, uv=None):
        a, b = self.local(x0, y0, h0), self.local(x1, y1, h1)
        bm = cuboid(a, b)
        if uv:
            uv_box(bm, *uv)
        return self.mesh(name, bm, mat, bevel)

    def cyl(self, name, x, y, h0, h1, r, mat, n=16, axis="z", bevel=0.0):
        """A cylinder: along z from h0 to h1 at (x, y), or along x / y between the first pair of numbers given."""
        bm = bmesh.new()
        if axis == "z":
            c0, c1 = self.local(x, y, h0), self.local(x, y, h1)
        elif axis == "x":
            c0, c1 = self.local(x, y, h0), self.local(h1, y, h0)  # x from x to h1 at height h0
        else:
            c0, c1 = self.local(x, y, h0), self.local(x, h1, h0)
        d = Vector(c1) - Vector(c0)
        bmesh.ops.create_cone(bm, cap_ends=True, segments=n, radius1=r, radius2=r, depth=d.length)
        rot = d.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4()
        bmesh.ops.transform(bm, matrix=rot, verts=bm.verts)
        bmesh.ops.translate(bm, vec=(Vector(c0) + Vector(c1)) / 2, verts=bm.verts)
        return self.mesh(name, bm, mat, bevel, smooth=True)


def cuboid(a, b):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    s = [b[i] - a[i] for i in range(3)]
    c = [(a[i] + b[i]) / 2 for i in range(3)]
    bmesh.ops.scale(bm, vec=s, verts=bm.verts)
    bmesh.ops.translate(bm, vec=c, verts=bm.verts)
    return bm


def uv_box(bm, su, sv):
    """Box mapping: each face projected on its dominant plane, ``su`` / ``sv`` texture repeats per metre."""
    layer = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        n = f.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        u_i, v_i = [(1, 2), (0, 2), (0, 1)][ax]
        for loop in f.loops:
            co = loop.vert.co
            loop[layer].uv = (co[u_i] * su, co[v_i] * sv)


def rounded_rect(hx, hy, r, n=5, cx=0.0, cy=0.0):
    """Points of a rectangle with rounded corners, counter-clockwise."""
    r = min(r, hx, hy)
    pts = []
    for ox, oy, a0 in ((hx - r, -hy + r, -90), (hx - r, hy - r, 0), (-hx + r, hy - r, 90), (-hx + r, -hy + r, 180)):
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + ox + r * math.cos(a), cy + oy + r * math.sin(a)))
    return pts


def loft(rings, cap_bottom=True, cap_top=True):
    """Connect rings of 3D points (all the same length) bottom to top."""
    bm = bmesh.new()
    vs = [[bm.verts.new(p) for p in ring] for ring in rings]
    n = len(rings[0])
    for a, b in zip(vs, vs[1:]):
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((a[i], a[j], b[j], b[i]))
    if cap_bottom:
        bm.faces.new(list(reversed(vs[0])))
    if cap_top:
        bm.faces.new(vs[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def slab(pts, z0, z1):
    return loft([[(x, y, z0) for x, y in pts], [(x, y, z1) for x, y in pts]])


def prism_y(profile, y0, y1):
    """A prism along y from a profile of (x, z) points (counter-clockwise seen from +y)."""
    return loft([[(x, y0, z) for x, z in profile], [(x, y1, z) for x, z in profile]])


def uv_planar(bm, x0, y0, x1, y1):
    layer = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        for loop in f.loops:
            co = loop.vert.co
            loop[layer].uv = ((co.x - x0) / (x1 - x0), (co.y - y0) / (y1 - y0))


def uv_sides(bm, z_scale, u_scale):
    """Side faces: u along the outline (x + y), v up (z); 1 unit is ``1 / scale`` metres."""
    layer = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        for loop in f.loops:
            co = loop.vert.co
            loop[layer].uv = ((co.x + co.y) * u_scale, co.z * z_scale)


# -- the printer -----------------------------------------------------------------------------------------------------

FRONT = 0.38  # the enclosure's front face (the door shuts against it)
BED_TOP = 0.10


def build_printer():
    L = Look("printer", PRINTER, PRINTER_C)
    shell = material("shell", srgb((0.78, 0.78, 0.80)), 0.5)
    frame = material("frame", srgb((0.035, 0.035, 0.04)), 0.4, 0.6)
    panel = material("panel", srgb((0.66, 0.66, 0.68)), 0.55, image=vent_texture())
    glass = material("glass", srgb((0.62, 0.68, 0.72)), 0.05, 0.0, alpha=0.16)
    dark = material("inside", srgb((0.22, 0.22, 0.23)), 0.7)
    bed = material("bed", srgb((0.025, 0.025, 0.03)), 0.35, 0.5)
    steel = material("steel", srgb((0.72, 0.73, 0.75)), 0.25, 1.0)
    rubber = material("rubber", srgb((0.02, 0.02, 0.02)), 0.9)
    led = material("led", srgb((1.0, 0.98, 0.92)), 0.3, emit=(1.0, 0.97, 0.9), strength=6.0)
    screen = material("screen", srgb((0.1, 0.2, 0.4)), 0.2, image=screen_texture(), emit=(1, 1, 1), strength=1.5)
    ptfe = material("ptfe", srgb((0.85, 0.85, 0.82)), 0.4)
    accent = material("accent", srgb((0.55, 0.56, 0.58)), 0.35, 0.8)

    for x in (0.335, 0.645):
        for y in (-0.15, 0.15):
            L.cyl(f"foot_{x}_{y}", x, y, 0.0, 0.009, 0.013, rubber)
    # the base: a box with a sloped front holding the screen, its top flat in front of the door
    lx = lambda x: x - PRINTER_C[0]
    lz = lambda h: h - PRINTER_C[2]
    profile = [(lx(0.67), lz(0.008)), (lx(0.67), lz(0.08)), (lx(0.335), lz(0.08)), (lx(0.31), lz(0.05)),
               (lx(0.31), lz(0.008))]
    L.mesh("base", prism_y(profile, -0.175, 0.175), shell, bevel=0.002)
    # the screen on the slope, and a knob beside it
    s = math.atan2(0.03, 0.025)
    n = Vector((-math.sin(s), 0.0, math.cos(s)))  # the slope's outward normal (x, -, z)
    t = Vector((math.cos(s), 0.0, math.sin(s)))  # up the slope
    o = Vector((0.31, 0.0, 0.05)) + n * 0.0008
    bm = bmesh.new()
    corners = []
    for a, yy in ((0.007, -0.13), (0.007, -0.045), (0.034, -0.045), (0.034, -0.13)):
        p = o + t * a
        corners.append(bm.verts.new(L.local(p.x, yy, p.z)))
    f = bm.faces.new(corners)
    uv = bm.loops.layers.uv.new("UVMap")
    for loop, (u, v) in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
        loop[uv].uv = (1 - u, v)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if f.normal.x > 0:
        bmesh.ops.reverse_faces(bm, faces=[f])
    L.mesh("screen", bm, screen)
    # frame: four posts and the beams round the roof
    for x0 in (FRONT, 0.655):
        for s_ in (-1, 1):
            y0, y1 = sorted((s_ * 0.155, s_ * 0.175))
            L.box(f"post_{x0}_{s_}", x0, x0 + 0.015, y0, y1, 0.08, 0.40, frame, 0.0015)
    L.box("beam_front", FRONT, FRONT + 0.015, -0.155, 0.155, 0.36, 0.40, frame, 0.0015)
    L.box("beam_back", 0.655, 0.67, -0.155, 0.155, 0.385, 0.40, frame, 0.0015)
    for s_ in (-1, 1):
        y0, y1 = sorted((s_ * 0.155, s_ * 0.175))
        L.box(f"beam_side_{s_}", FRONT + 0.015, 0.655, y0, y1, 0.385, 0.40, frame, 0.0015)
    L.box("roof_glass", FRONT + 0.015, 0.655, -0.155, 0.155, 0.391, 0.396, glass)
    # side and back panels (opaque), the chamber floor
    for s_ in (-1, 1):
        y0, y1 = sorted((s_ * 0.1645, s_ * 0.1715))
        L.box(f"side_{s_}", FRONT + 0.015, 0.655, y0, y1, 0.08, 0.385, panel, uv=(3.0, 3.0))
    L.box("back", 0.66, 0.667, -0.155, 0.155, 0.08, 0.385, shell)
    L.box("floor", FRONT + 0.015, 0.655, -0.155, 0.155, 0.08, 0.082, dark)
    # the hinge knuckles on the front left post, the magnetic catch on the right one
    for h0 in (0.11, 0.33):
        L.cyl(f"hinge_{h0}", 0.379, 0.170, h0, h0 + 0.03, 0.0045, frame)
    L.box("catch", 0.3792, 0.3802, -0.165, -0.157, 0.205, 0.245, rubber)
    # the light under the front beam
    L.box("led", FRONT + 0.015, FRONT + 0.02, -0.14, 0.14, 0.356, 0.36, led)
    # the heated bed, low, on its carriage, with the lead screws and guide rods at the back
    L.box("bed_carriage", 0.41, 0.63, -0.10, 0.10, 0.082, 0.09, frame)
    L.box("bed", 0.395, 0.635, -0.12, 0.12, 0.09, BED_TOP, bed, 0.001)
    L.box("bed_arm_l", 0.60, 0.655, 0.08, 0.10, 0.083, 0.092, frame)
    L.box("bed_arm_r", 0.60, 0.655, -0.10, -0.08, 0.083, 0.092, frame)
    for y in (-0.13, 0.13):
        L.cyl(f"screw_{y}", 0.648, y, 0.082, 0.385, 0.004, steel)
    for y in (-0.095, 0.095):
        L.cyl(f"rod_{y}", 0.648, y, 0.082, 0.385, 0.0035, steel)
    # the gantry: rails along both sides, the cross beam the head rides on, belts, motors, the filament tube
    for s_ in (-1, 1):
        y0, y1 = sorted((s_ * 0.143, s_ * 0.155))
        L.box(f"rail_{s_}", 0.40, 0.65, y0, y1, 0.322, 0.332, steel, 0.0008)
        y0, y1 = sorted((s_ * 0.137, s_ * 0.139))
        L.box(f"belt_{s_}", 0.405, 0.645, y0, y1, 0.334, 0.34, rubber)
        y0, y1 = sorted((s_ * 0.135, s_ * 0.155))
        L.box(f"carriage_{s_}", 0.484, 0.512, y0, y1, 0.332, 0.344, frame, 0.001)
        y0, y1 = sorted((s_ * 0.112, s_ * 0.154))
        L.box(f"motor_{s_}", 0.612, 0.654, y0, y1, 0.342, 0.384, frame, 0.002)
    L.box("x_beam", 0.488, 0.508, -0.135, 0.135, 0.318, 0.336, frame, 0.002)
    pts = [Vector(L.local(0.457, 0.07, 0.31)), Vector(L.local(0.47, 0.06, 0.35)), Vector(L.local(0.55, 0.03, 0.375)),
           Vector(L.local(0.64, 0.0, 0.375))]
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        bm = bmesh.new()
        d = b - a
        bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=0.002, radius2=0.002, depth=d.length)
        bmesh.ops.transform(bm, matrix=d.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4(), verts=bm.verts)
        bmesh.ops.translate(bm, vec=(a + b) / 2, verts=bm.verts)
        L.mesh(f"tube_{i}", bm, ptfe, smooth=True)
    return L


def build_door():
    """The door shut: tinted acrylic in a black border, the handle on its outside, near its free (right) edge."""
    L = Look("door", DOOR, DOOR_C)
    glass = material("door_glass", srgb((0.45, 0.52, 0.58)), 0.04, alpha=0.25)
    border = material("door_border", srgb((0.03, 0.03, 0.035)), 0.45)
    alu = material("handle", srgb((0.80, 0.81, 0.83)), 0.3, 1.0)
    L.box("glass", 0.374, 0.379, -0.163, 0.163, 0.085, 0.373, glass)
    w = 0.016
    L.box("border_bottom", 0.3722, 0.374, -0.163, 0.163, 0.085, 0.085 + w, border)
    L.box("border_top", 0.3722, 0.374, -0.163, 0.163, 0.373 - w, 0.373, border)
    L.box("border_left", 0.3722, 0.374, 0.163 - w, 0.163, 0.085 + w, 0.373 - w, border)
    L.box("border_right", 0.3722, 0.374, -0.163, -0.163 + w, 0.085 + w, 0.373 - w, border)
    L.cyl("handle_bar", 0.338, -0.136, 0.17, 0.29, 0.006, alu, bevel=0.0)
    for h in (0.181, 0.279):
        L.cyl(f"standoff_{h}", 0.343, -0.136, h, 0.3722, 0.005, alu, axis="x")
    return L


def build_head():
    L = Look("head", HEAD, HEAD_C)
    body = material("head_body", srgb((0.78, 0.79, 0.80)), 0.45)
    black = material("head_fan", srgb((0.02, 0.02, 0.02)), 0.6)
    brass = material("nozzle", srgb((0.78, 0.60, 0.25)), 0.3, 1.0)
    alu = material("heater", srgb((0.6, 0.6, 0.62)), 0.35, 0.9)
    hx, hy, hz = HEAD[0] / 2, HEAD[1] / 2, HEAD[2] / 2
    L.mesh("shroud", slab(rounded_rect(hx - 0.0005, hy - 0.0005, 0.008), -hz + 0.022, hz), body, bevel=0.002)
    L.cyl("fan", 0.4285, 0.07, 0.288, 0.4295 + 0.0005, 0.017, black, n=24, axis="x")
    for i in range(5):
        L.box(f"grille_{i}", 0.4282, 0.4292, 0.054 + i * 0.008, 0.056 + i * 0.008, 0.272, 0.304, body)
    L.box("duct", 0.432, 0.484, 0.046, 0.094, 0.244, 0.253, black, 0.002)
    L.box("heater", 0.448, 0.466, 0.06, 0.08, 0.236, 0.247, alu, 0.0008)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=0.001, radius2=0.004, depth=0.006)
    bmesh.ops.translate(bm, vec=L.local(0.457, 0.07, 0.233), verts=bm.verts)
    L.mesh("nozzle", bm, brass, smooth=True)
    return L


def build_plate():
    L = Look("plate", PLATE, (0, 0, 0))
    hx, hy, hz = PLATE[0] / 2, PLATE[1] / 2, PLATE[2] / 2
    steel = material("plate_steel", srgb((0.30, 0.30, 0.31)), 0.4, 1.0)
    pei = material("plate_pei", srgb((0.74, 0.55, 0.22)), 0.55, 0.35, image=pei_texture())
    L.mesh("plate_steel", slab(rounded_rect(hx, hy, 0.006), -hz, hz - 0.0003), steel)
    top = slab(rounded_rect(hx, hy, 0.006), hz - 0.0003, hz)
    uv_planar(top, -hx, -hy, hx, hy)
    L.mesh("plate_pei", top, pei)
    return L


def build_print():
    """An open box (a printed enclosure), 2.5 mm walls, rounded corners, on a 45 degree chamfered first layer."""
    L = Look("print", PRINT, (0, 0, 0))
    hx, hy, hz = PRINT[0] / 2, PRINT[1] / 2, PRINT[2] / 2
    pla = material("print_pla", srgb((0.35, 0.85, 0.15)), 0.45, image=layer_texture("print_layers", (0.35, 0.85, 0.15)))
    r = 0.004
    outer = [
        [(x, y, -hz) for x, y in rounded_rect(hx - FOOT_H, hy - FOOT_H, r - FOOT_H * 0.5)],
        [(x, y, -hz + FOOT_H) for x, y in rounded_rect(hx, hy, r)],
        [(x, y, hz) for x, y in rounded_rect(hx, hy, r)],
    ]
    wall = 0.0025
    floor = -hz + 0.004
    inner = [
        [(x, y, hz) for x, y in rounded_rect(hx - wall, hy - wall, r - wall * 0.6)],
        [(x, y, floor) for x, y in rounded_rect(hx - wall, hy - wall, r - wall * 0.6)],
    ]
    bm = loft(outer + inner, cap_bottom=True, cap_top=True)
    uv_sides(bm, 100.0, 30.0)
    L.mesh("print_body", bm, pla, bevel=0.0004)
    return L


def build_tall_print():
    """A twisted vase: rounded squares turning 45 degrees bottom to top, swelling a little in the middle."""
    L = Look("tall_print", TALL, (0, 0, 0))
    orange = (0.95, 0.42, 0.08)
    pla = material("tall_pla", srgb(orange), 0.45, image=layer_texture("tall_layers", orange))
    hz = TALL[2] / 2
    rings = []
    k = 24
    for i in range(k + 1):
        f = i / k
        z = -hz + f * TALL[2]
        half = 0.0215 + 0.0035 * math.sin(math.pi * f)
        a = math.radians(45 * f)
        pts = rounded_rect(half, half, 0.008, 4)
        c, s = math.cos(a), math.sin(a)
        scale = 1.0 / (abs(c) + abs(s))  # keep the turned square inside the box
        rings.append([((x * c - y * s) * scale * 1.0, (x * s + y * c) * scale * 1.0, z) for x, y in pts])
    wall = 0.0015
    inner = [[(x * (1 - wall / 0.02), y * (1 - wall / 0.02), z) for x, y, z in rings[-1]],
             [(x * (1 - wall / 0.02), y * (1 - wall / 0.02), -hz + 0.003) for x, y, z in rings[-1]]]
    bm = loft(rings + inner)
    uv_sides(bm, 100.0, 30.0)
    L.mesh("vase", bm, pla, smooth=True)
    return L


def build_scraper():
    """The handle from y = -11 cm to +1 cm, the blade from there to y = +11 cm, its bottom flush with the handle's."""
    L = Look("scraper", SCRAPER, (0, 0, 0))
    hz = SCRAPER[2] / 2
    y0 = -SCRAPER[1] / 2
    y_handle_end = y0 + HANDLE_LEN
    steel = material("scraper_steel", srgb((0.78, 0.78, 0.80)), 0.22, 1.0)
    rubber = material("scraper_handle", srgb((0.05, 0.05, 0.055)), 0.7, image=grip_texture())
    ferrule = material("scraper_ferrule", srgb((0.55, 0.56, 0.58)), 0.3, 1.0)

    def section(y, hx, top):
        mid = (top - hz) / 2
        return [(x, y, mid + z) for x, z in rounded_rect(hx, (top + hz) / 2, 0.006, 4)]
    grip_end = y_handle_end - 0.014
    rings = [section(y0, 0.0115, hz - 0.002), section(y0 + 0.02, 0.014, hz), section(grip_end - 0.02, 0.014, hz),
             section(grip_end, 0.0125, hz - 0.002)]
    bm = loft(rings)
    uv_sides(bm, 40.0, 40.0)
    L.mesh("handle", bm, rubber, bevel=0.001, smooth=True)
    collar = [section(grip_end, 0.0115, hz - 0.006), section(y_handle_end, 0.0105, hz - 0.008)]
    L.mesh("ferrule", loft(collar), ferrule, bevel=0.0005, smooth=True)
    zb = -hz

    def blade_ring(y, hx, t):
        return [(-hx, y, zb), (hx, y, zb), (hx, y, zb + t), (-hx, y, zb + t)]
    tip = SCRAPER[1] / 2
    blade = loft([blade_ring(y_handle_end - 0.012, 0.016, BLADE_T), blade_ring(tip - 0.006, 0.0197, BLADE_T),
                  blade_ring(tip, 0.02, 0.0002)])
    L.mesh("blade", blade, steel)
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


BUILDERS = {"printer": build_printer, "door": build_door, "head": build_head, "plate": build_plate,
            "print": build_print, "tall_print": build_tall_print, "scraper": build_scraper}


def build(names=None, hide=True):
    clear()
    names = names or WANT.get(FOLDER, list(BUILDERS))
    looks = [BUILDERS[n]() for n in names]
    out = {look.name: export(look) for look in looks}
    if hide:  # keep the session tidy for the others using it: hide the looks once exported
        for look in looks:
            bpy.context.view_layer.layer_collection.children[look.coll.name].exclude = True
    return out


if __name__ == "__main__":
    print(build())
