"""make_looks.py: model an enclosed desktop 3D printer, its side spool holder, its roof feeder and their props in
Blender and export them as USD next to this file.

    blender -b -P looks/make_looks.py          (Blender 4.2 or later, from the task folder)

The same script is in the folders of two tasks (filament-into-extruder and spool-onto-printer-holder) and makes the
same printer, holder, feeder and spool in both. Its printer, door, head and plate are those of the scrape-print-off-bed
and open-printer-door tasks. The physics stays the task file's boxes. Each look is a render-only mesh that ``looks.py``
puts on one of those boxes, whose plain cuboid is then hidden but still collides. Every look is built in its box's own
frame (metres, the box's centre at the origin, turned by the box's yaw, z up) and must fit inside the box (checked when
exported). Positions below are written in the station frame (x forward, y left, h above the table top) and moved into
each box's frame.

- ``printer.usdc``: the printer without its door, plate and print head. A 36 x 35 x 40 cm enclosed FDM printer: a
  light grey base with a sloped front and a small screen, a black aluminium frame, light grey side and back panels, a
  glass roof, a light under the front beam, the black heated bed low inside, the gantry with its rails and belts.
- ``door.usdc``: the front door, tinted clear acrylic in a black border, with a brushed aluminium bar handle.
- ``head.usdc``: the print head, a light grey shroud with a black fan and a brass nozzle.
- ``plate.usdc``: the spring-steel build plate, textured gold PEI on top (24 x 24 cm, 1 mm).
- ``holder.usdc``: the spool holder on the printer's left wall: a black bracket plate with four screws, a 45 mm
  brushed steel rod sticking out 13 cm with a chamfered nose, and a black stop collar 3 cm long at its root.
- ``feeder.usdc``: the filament feeder on the roof's back right: an aluminium riser plate, a black stepper motor, a
  black anodised feeder body with a brass inlet funnel on its left face (the funnel and the bore open, as in the
  colliders), a steel idler lever with its tension screw, and a white PTFE tube down into the roof.
- ``spool.usdc``: a spool of orange filament, its axle along the box's x: black flanges with six windows and a blank
  label, the wound filament (a shadow line between turns), a grey hub round a 52 mm bore.
- ``link.usdc`` and ``tip.usdc``: a 2.5 cm piece of 1.75 mm orange filament, round ended (the tip cut flat at its
  free end). Every link of the filament wears one.
- ``block.usdc``: a steel 1-2-3 block (1 x 2 x 3 inch) with its holes.

Textures are drawn here with numpy and saved in ``textures/``. Nothing is downloaded. Everything this script makes in
Blender is named with the prefix ``<task folder>__``, and only those datablocks are removed when it runs again.
"""

import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

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

# the holder, the feeder and the spool (printer.py)
WALL_Y = 0.175
ROD_X, ROD_H, ROD_R = 0.48, 0.22, 0.0225
BRACKET_T, BRACKET = 0.006, (0.10, 0.14)
COLLAR_R, COLLAR_T = 0.035, 0.03
ROD_TIP_Y, NOSE = 0.31, (0.02, 0.014)
HOLDER_LO = (ROD_X - BRACKET[0] / 2, WALL_Y, ROD_H - BRACKET[1] / 2)
HOLDER_HI = (ROD_X + BRACKET[0] / 2, ROD_TIP_Y, ROD_H + BRACKET[1] / 2)
INLET_X, INLET_H, MOUTH_Y = 0.545, 0.475, -0.13
FEEDER_LO = (INLET_X - 0.05, -WALL_Y, 0.40)
FEEDER_HI = (INLET_X + 0.05, MOUTH_Y, 0.52)
FEEDER_YAW = -math.pi / 2  # the feeder's box x (the inlet's axis, into the feeder) is the station's -y
FUNNEL, BORE, FUNNEL_DEPTH, BORE_DEPTH = 0.006, 0.0024, 0.005, 0.03
SPOOL_SIZE = (0.07, 0.20, 0.20)  # the spool's box: along its axle, across, tall
SPOOL_W, SPOOL_R, FLANGE_T, WIND_R, BORE_R, HUB_R = 0.07, 0.10, 0.006, 0.08, 0.026, 0.031
LINK_SIZE = (0.025, 0.00175, 0.00175)
BLOCK_SIZE = (0.0762, 0.0508, 0.0254)

WANT = {
    "scrape-print-off-bed": ["printer", "door", "head", "plate", "print", "scraper"],
    "open-printer-door": ["printer", "door", "head", "plate", "tall_print"],
    "filament-into-extruder": ["printer", "door", "head", "plate", "holder", "feeder", "spool", "link", "tip",
                               "block"],
    "spool-onto-printer-holder": ["printer", "door", "head", "plate", "holder", "feeder", "spool"],
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
    def __init__(self, name, size, centre, yaw=0.0):
        self.name, self.size, self.c, self.yaw = name, size, centre, yaw
        self.coll = bpy.data.collections.new(P + "look_" + name)
        bpy.context.scene.collection.children.link(self.coll)
        self.root = bpy.data.objects.new(P + name, None)
        self.coll.objects.link(self.root)
        self.n = 0

    def local(self, x, y, h):
        """A station-frame point in this box's frame (turned back by the box's yaw)."""
        dx, dy = x - self.c[0], y - self.c[1]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (c * dx + s * dy, -s * dx + c * dy, h - self.c[2])

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
    a, b = [min(u, v) for u, v in zip(a, b)], [max(u, v) for u, v in zip(a, b)]
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


# -- the holder, the feeder, the spool, the filament and the block ----------------------------------------------------


def _grain(h, w, along_x=True, base=0.62, amp=0.06):
    """Ground steel: fine streaks along one direction and a little mottling."""
    streak = RNG.normal(0, 1, (1, w) if not along_x else (h, 1))
    streak = np.repeat(streak, h if not along_x else 1, axis=0) if not along_x else np.repeat(streak, w, axis=1)
    v = base + amp * 0.5 * streak + amp * (_noise(h, w, 24) - 0.5)
    return np.repeat(v[:, :, None], 3, axis=2) * np.array([0.98, 0.99, 1.0])


def _holes(rgb, centres_px, r_px):
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy in centres_px:
        d = np.hypot(xx - cx, yy - cy)
        rgb[d < r_px] = 0.04
        ring = (d >= r_px) & (d < r_px + 2.5)
        rgb[ring] *= 0.55
    return rgb


def block_textures():
    """The 1-2-3 block's faces: 3 x 2 inch top with 5 holes, 3 x 1 inch sides with 3, 2 x 1 inch ends with 2."""
    k = 300  # px per inch
    top = _holes(_grain(2 * k, 3 * k, True), [(k * x, k * y) for x, y in ((0.5, 0.5), (2.5, 0.5), (1.5, 1.0),
                                                                          (0.5, 1.5), (2.5, 1.5))], 0.17 * k)
    side = _holes(_grain(k, 3 * k, True, 0.58), [(k * x, 0.5 * k) for x in (0.5, 1.5, 2.5)], 0.17 * k)
    end = _holes(_grain(k, 2 * k, False, 0.58), [(k * x, 0.5 * k) for x in (0.5, 1.5)], 0.17 * k)
    return _save("block_top", top), _save("block_side", side), _save("block_end", end)


def bm_box(lo, hi):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    c, s = (Vector(lo) + Vector(hi)) / 2, Vector(hi) - Vector(lo)
    for v in bm.verts:
        v.co = c + Vector((v.co.x * s.x, v.co.y * s.y, v.co.z * s.z))
    return bm


def bm_quad(corners, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
    bm = bmesh.new()
    vs = [bm.verts.new(c) for c in corners]
    f = bm.faces.new(vs)
    layer = bm.loops.layers.uv.new("UVMap")
    for loop, t in zip(f.loops, uv):
        loop[layer].uv = t
    return bm


def _axis_matrix(axis):
    return {"Z": Matrix.Identity(4), "Y": Matrix.Rotation(math.pi / 2, 4, "X"),
            "X": Matrix.Rotation(math.pi / 2, 4, "Y")}[axis]


def bm_cyl(center, r, h, axis="Z", segs=32, r2=None, caps=True):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=caps, segments=segs, radius1=r, radius2=r if r2 is None else r2, depth=h,
                          matrix=Matrix.Translation(Vector(center)) @ _axis_matrix(axis))
    return bm


def bm_ring(center, r_in, r_out, h, axis="Z", segs=48):
    """A tube around ``axis``."""
    bm = bmesh.new()
    rings = [[bm.verts.new((r * math.cos(a), r * math.sin(a), z)) for a in np.linspace(0, 2 * math.pi, segs,
                                                                                      endpoint=False)]
             for r, z in ((r_out, -h / 2), (r_out, h / 2), (r_in, h / 2), (r_in, -h / 2))]
    for a, b in zip(rings, rings[1:] + rings[:1]):
        for i in range(segs):
            j = (i + 1) % segs
            bm.faces.new((a[i], a[j], b[j], b[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.transform(Matrix.Translation(Vector(center)) @ _axis_matrix(axis))
    return bm


def bm_disc_x(x, r_in, r_out, t, segs=72):
    """A flat ring across the x axis at ``x``, ``t`` thick, its faces mapped flat (u, v from y, z)."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    angles = np.linspace(0, 2 * math.pi, segs, endpoint=False)
    rings = {}
    for side, xs in (("in", x - t / 2), ("out", x + t / 2)):
        rings[side] = [[bm.verts.new((xs, r * math.cos(a), r * math.sin(a))) for a in angles] for r in (r_in, r_out)]
    for i in range(segs):
        j = (i + 1) % segs
        for side, flip in (("in", True), ("out", False)):
            inner, outer = rings[side]
            quad = [inner[i], inner[j], outer[j], outer[i]]
            f = bm.faces.new(quad[::-1] if flip else quad)
            for loop in f.loops:
                co = loop.vert.co
                loop[uv].uv = (0.5 + co.y / (2 * r_out), 0.5 + co.z / (2 * r_out))
        for k in (0, 1):  # the rims
            a, b = rings["in"][k], rings["out"][k]
            bm.faces.new((a[i], a[j], b[j], b[i]) if k else (a[i], b[i], b[j], a[j]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def bm_drum_x(r, width, segs=96):
    """The outside of a cylinder along x, ``width`` wide, mapped u around and v across (no ends)."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    rows = [[bm.verts.new((x, r * math.cos(a), r * math.sin(a))) for a in np.linspace(0, 2 * math.pi, segs + 1)]
            for x in (-width / 2, width / 2)]
    for i in range(segs):
        f = bm.faces.new((rows[0][i], rows[0][i + 1], rows[1][i + 1], rows[1][i]))
        for loop, (u, v) in zip(f.loops, ((i, 0), (i + 1, 0), (i + 1, 1), (i, 1))):
            loop[uv].uv = (u / segs, v)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def winding_texture():
    """Wound 1.75 mm filament seen from outside: orange, a thin shadow line between each turn across the width."""
    h, w = 512, 64  # v across the width (rows), u around (columns)
    v = (np.arange(h) + 0.5) / h * 36.6  # 36.6 turns across the 64 mm
    shade = 0.78 + 0.22 * np.sin(np.pi * (v % 1.0)) ** 0.6
    shade = shade[:, None] * (0.97 + 0.06 * _noise(h, w, 16))
    return _save("winding", np.repeat(shade[:, :, None], 3, axis=2) * np.array([1.0, 0.42, 0.0]))


def bm_capsule_x(length, r, flat_front=False, segs=12, rings=8):
    """A capsule along x, ``length`` long overall and ``r`` in radius (its +x end cut flat if ``flat_front``)."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=rings, radius=r,
                              matrix=Matrix.Rotation(math.pi / 2, 4, "Y"))
    for v in bm.verts:
        v.co.x += (length / 2 - r) * (1 if v.co.x > 1e-9 else -1 if v.co.x < -1e-9 else 0)
        if flat_front and v.co.x > length / 2 - r:
            v.co.x = length / 2
    return bm


def spool_flange_texture():
    """A spool's flange face: charcoal plastic, six round windows near the rim, a blank white label by the hub."""
    n = 512
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) / n * 2 - 1
    r = np.hypot(xx, yy)
    rgb = np.ones((n, n, 3)) * 0.13 * (0.9 + 0.2 * _noise(n, n, 32))[:, :, None]
    for k in range(6):
        c = 0.62 * np.array([math.cos(k * math.pi / 3 + math.pi / 6), math.sin(k * math.pi / 3 + math.pi / 6)])
        d = np.hypot(xx - c[0], yy - c[1])
        rgb[d < 0.15] = 0.012
        rgb[(d >= 0.15) & (d < 0.165)] *= 1.8
    label = (np.abs(xx) < 0.2) & (yy > 0.33) & (yy < 0.47)
    rgb[label] = 0.86
    for rr in (0.36, 0.97):
        rgb[np.abs(r - rr) < 0.008] = 0.09
    return _save("spool_flange", rgb)


def tube(L, name, p0, p1, r_in, r_out, mat, n=48):
    """A tube (a ring with thickness) between two station-frame points."""
    a, b = Vector(L.local(*p0)), Vector(L.local(*p1))
    d = b - a
    bm = bm_ring((0, 0, 0), r_in, r_out, d.length, "Z", n)
    bm.transform(Matrix.Translation((a + b) / 2) @ d.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4())
    return L.mesh(name, bm, mat, smooth=True)


def cone(L, name, p0, p1, r0, r1, mat, n=32, caps=True):
    """A cylinder or a cone from ``p0`` (radius ``r0``) to ``p1`` (radius ``r1``), station-frame points."""
    a, b = Vector(L.local(*p0)), Vector(L.local(*p1))
    d = b - a
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=caps, segments=n, radius1=r0, radius2=r1, depth=d.length)
    bm.transform(Matrix.Translation((a + b) / 2) @ d.normalized().to_track_quat("Z", "Y").to_matrix().to_4x4())
    return L.mesh(name, bm, mat, smooth=True)


def _centre(lo, hi):
    return tuple((a + b) / 2 for a, b in zip(lo, hi))


def build_holder():
    """The spool holder on the printer's left wall: a bracket plate with four screws, the stop collar and the rod."""
    size = tuple(b - a for a, b in zip(HOLDER_LO, HOLDER_HI))
    L = Look("holder", size, _centre(HOLDER_LO, HOLDER_HI))
    black = material("holder_black", srgb((0.035, 0.035, 0.04)), 0.45, 0.3)
    steel = material("holder_rod", srgb((0.74, 0.75, 0.77)), 0.22, 1.0)
    screw = material("holder_screw", srgb((0.5, 0.51, 0.53)), 0.35, 1.0)
    x0, x1 = HOLDER_LO[0], HOLDER_HI[0]
    h0, h1 = HOLDER_LO[2], HOLDER_HI[2]
    y_plate = WALL_Y + BRACKET_T
    L.box("bracket", x0, x1, WALL_Y, y_plate, h0, h1, black, 0.002)
    for i, (dx, dh) in enumerate(((-0.036, -0.054), (0.036, -0.054), (-0.036, 0.054), (0.036, 0.054))):
        cone(L, f"screw_{i}", (ROD_X + dx, y_plate - 0.0005, ROD_H + dh), (ROD_X + dx, y_plate + 0.0025, ROD_H + dh),
             0.0042, 0.0036, screw, 20)
    y_stop = y_plate + COLLAR_T
    cone(L, "collar", (ROD_X, y_plate - 0.0005, ROD_H), (ROD_X, y_stop, ROD_H), COLLAR_R, COLLAR_R, black, 48)
    cone(L, "set_screw", (ROD_X, y_plate + COLLAR_T / 2, ROD_H + COLLAR_R - 0.002),
         (ROD_X, y_plate + COLLAR_T / 2, ROD_H + COLLAR_R + 0.001), 0.003, 0.003, screw, 16)
    nose0 = ROD_TIP_Y - NOSE[0]
    cone(L, "rod", (ROD_X, y_stop - 0.001, ROD_H), (ROD_X, nose0, ROD_H), ROD_R, ROD_R, steel, 48, caps=False)
    cone(L, "nose", (ROD_X, nose0, ROD_H), (ROD_X, ROD_TIP_Y, ROD_H), ROD_R, NOSE[1], steel, 48)
    return L


def build_feeder():
    """The feeder on the roof: a riser plate, a stepper motor, the black feeder body with the brass inlet on its left
    face (+y), the funnel and the bore open, the idler lever and its tension screw, a PTFE tube into the roof."""
    size = (FEEDER_HI[1] - FEEDER_LO[1], FEEDER_HI[0] - FEEDER_LO[0], FEEDER_HI[2] - FEEDER_LO[2])
    L = Look("feeder", size, _centre(FEEDER_LO, FEEDER_HI), FEEDER_YAW)
    alu = material("feeder_riser", srgb((0.70, 0.71, 0.73)), 0.35, 1.0)
    black = material("feeder_black", srgb((0.03, 0.03, 0.034)), 0.42, 0.3)
    motor = material("feeder_motor", srgb((0.045, 0.045, 0.05)), 0.55)
    cap = material("feeder_motor_cap", srgb((0.62, 0.63, 0.65)), 0.3, 1.0)
    brass = material("feeder_brass", srgb((0.86, 0.66, 0.30)), 0.28, 1.0)
    dark = material("feeder_bore", srgb((0.01, 0.01, 0.01)), 0.9)
    steel = material("feeder_lever", srgb((0.62, 0.63, 0.66)), 0.3, 1.0)
    ptfe = material("feeder_ptfe", srgb((0.92, 0.92, 0.90)), 0.4)
    X = INLET_X
    yb, yf = -0.173, MOUTH_Y - 0.0005  # the body's back and front (the mouth's face)
    L.box("riser", X - 0.048, X + 0.048, yb, MOUTH_Y - 0.002, 0.400, 0.406, alu, 0.001)
    L.box("motor_cap_front", X + 0.002, X + 0.010, -0.171, MOUTH_Y - 0.004, 0.406, 0.446, cap, 0.002)
    L.box("motor_body", X + 0.010, X + 0.036, -0.170, MOUTH_Y - 0.005, 0.406, 0.446, motor, 0.002)
    L.box("motor_cap_back", X + 0.036, X + 0.044, -0.171, MOUTH_Y - 0.004, 0.406, 0.446, cap, 0.002)
    L.box("stand", X - 0.038, X, -0.165, -0.135, 0.406, 0.447, alu, 0.001)
    # the body round the channel: a square hole 6 mm wide through the funnel's depth, 2.4 mm after it
    xa, xb, ha, hb = X - 0.038, X + 0.040, 0.447, 0.505
    y1, y2 = MOUTH_Y - FUNNEL_DEPTH, MOUTH_Y - FUNNEL_DEPTH - BORE_DEPTH
    for zone, ya, yz, w in (("funnel", yf, y1, FUNNEL / 2), ("bore", y1, y2, BORE / 2)):
        L.box(f"{zone}_top", xa, xb, ya, yz, INLET_H + w, hb, black)
        L.box(f"{zone}_bottom", xa, xb, ya, yz, ha, INLET_H - w, black)
        L.box(f"{zone}_left", INLET_X + w, xb, ya, yz, INLET_H - w, INLET_H + w, black)
        L.box(f"{zone}_right", xa, INLET_X - w, ya, yz, INLET_H - w, INLET_H + w, black)
    L.box("body_back", xa, xb, y2, yb, ha, hb, black, 0.0015)
    tube(L, "inlet_collar", (INLET_X, MOUTH_Y - 0.0018, INLET_H), (INLET_X, MOUTH_Y - 0.0002, INLET_H), FUNNEL / 2,
         0.0085, brass)
    cone(L, "inlet_funnel", (INLET_X, MOUTH_Y - 0.0002, INLET_H), (INLET_X, y1, INLET_H), FUNNEL / 2, BORE / 2, brass,
         32, caps=False)
    cone(L, "bore", (INLET_X, y1, INLET_H), (INLET_X, y2, INLET_H), BORE / 2, BORE / 2, dark, 16)
    L.box("lever", X - 0.033, X + 0.036, -0.152, -0.138, hb, hb + 0.007, steel, 0.0015)
    cone(L, "tension_screw", (X - 0.028, -0.145, 0.497), (X - 0.028, -0.145, 0.519), 0.0028, 0.0028, steel, 16)
    cone(L, "tension_knob", (X - 0.028, -0.145, 0.512), (X - 0.028, -0.145, 0.519), 0.0065, 0.0065, black, 24)
    cone(L, "ptfe_coupler", (X - 0.045, -0.160, 0.441), (X - 0.045, -0.160, 0.4475), 0.0045, 0.0045, black, 20)
    cone(L, "ptfe_tube", (X - 0.045, -0.160, 0.4065), (X - 0.045, -0.160, 0.441), 0.002, 0.002, ptfe, 16)
    for i, (x, h) in enumerate(((X - 0.030, 0.497), (X + 0.032, 0.497), (X + 0.032, 0.455))):
        cone(L, f"face_screw_{i}", (x, MOUTH_Y - 0.0016, h), (x, MOUTH_Y - 0.0001, h), 0.0026, 0.0026, steel, 16)
    return L


def build_spool():
    """The spool in its box's frame, its axle along x through the centre: two black flanges with six windows, the
    orange filament wound out to 8 cm, a grey hub round the 52 mm bore."""
    L = Look("spool", SPOOL_SIZE, (0.0, 0.0, 0.0))
    flange = material("spool_flange", srgb((0.05, 0.05, 0.055)), 0.45, image=spool_flange_texture())
    winding = material("spool_winding", srgb((1.0, 0.42, 0.0)), 0.35, image=winding_texture())
    hub = material("spool_hub", srgb((0.55, 0.56, 0.58)), 0.4)
    half = SPOOL_W / 2
    for i, s in enumerate((-1, 1)):
        L.mesh(f"flange_{i}", bm_disc_x(s * (half - FLANGE_T / 2), HUB_R - 0.0005, SPOOL_R, FLANGE_T), flange)
    inner = SPOOL_W - 2 * FLANGE_T
    L.mesh("hub", bm_ring((0, 0, 0), BORE_R, HUB_R, SPOOL_W, "X", 48), hub, smooth=True)
    L.mesh("winding", bm_drum_x(WIND_R, inner), winding, smooth=True)
    L.mesh("winding_sides", bm_ring((0, 0, 0), HUB_R, WIND_R - 0.0005, inner - 0.0004, "X", 96), winding,
           smooth=True)
    return L


def build_link(name, flat_front):
    """A 2.5 cm piece of 1.75 mm orange PLA filament, round ended so the links read as one tube."""
    L = Look(name, LINK_SIZE, (0.0, 0.0, 0.0))
    pla = material("pla_orange", srgb((1.0, 0.42, 0.0)), 0.3)
    L.mesh("tube", bm_capsule_x(LINK_SIZE[0], LINK_SIZE[1] / 2 - 1e-6, flat_front), pla, smooth=True)
    return L


def build_block():
    L = Look("block", BLOCK_SIZE, (0.0, 0.0, 0.0))
    t_top, t_side, t_end = block_textures()
    top = material("block_top", (0.6, 0.6, 0.62), 0.3, 1.0, t_top)
    side = material("block_side", (0.6, 0.6, 0.62), 0.3, 1.0, t_side)
    end = material("block_end", (0.6, 0.6, 0.62), 0.3, 1.0, t_end)
    hx, hy, hz = (s / 2 for s in BLOCK_SIZE)
    e = 0.0002
    L.mesh("body", bm_box((-hx + e, -hy + e, -hz), (hx - e, hy - e, hz - e)), material("block_steel", (0.5, 0.5, 0.52),
                                                                                     0.35, 1.0))
    L.mesh("top", bm_quad([(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]), top)
    for s, name in ((1, "left"), (-1, "right")):
        c = [(-hx, s * hy, -hz), (hx, s * hy, -hz), (hx, s * hy, hz), (-hx, s * hy, hz)]
        L.mesh(f"side_{name}", bm_quad(c if s < 0 else c[::-1], ((0, 0), (1, 0), (1, 1), (0, 1)) if s < 0
                                       else ((0, 1), (1, 1), (1, 0), (0, 0))), side)
    for s, name in ((1, "back"), (-1, "front")):
        c = [(s * hx, -hy, -hz), (s * hx, hy, -hz), (s * hx, hy, hz), (s * hx, -hy, hz)]
        L.mesh(f"end_{name}", bm_quad(c if s > 0 else c[::-1], ((0, 0), (1, 0), (1, 1), (0, 1)) if s > 0
                                      else ((0, 1), (1, 1), (1, 0), (0, 0))), end)
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
            "print": build_print, "tall_print": build_tall_print, "scraper": build_scraper, "holder": build_holder,
            "feeder": build_feeder, "spool": build_spool, "link": lambda: build_link("link", False),
            "tip": lambda: build_link("tip", True), "block": build_block}


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
