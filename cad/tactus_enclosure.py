#!/usr/bin/env python3
# =============================================================================
# TACTUS - Brain-pack enclosure  (single compartment, ONE-PRINT base + lid)
# -----------------------------------------------------------------------------
# Simplified for the as-built rig (truth.md): the SK473 amp+driver units live on
# the VEST, so this box holds ONLY the audio source + power:
#
#   ONE COMPARTMENT (lidded), generous + loose (zip-tie / foam, no tight pockets):
#     - 2x Vantec NBA-200U USB 7.1 (100 x 58 x 26 each; web-verified vantecusa.com)
#       -> the 6x SK473 3.5mm audio plugs land here (6 of 8 jacks)
#     - 1x Inland 10-port USB hub (~163 x 48 x 23; SKU 885194)
#       -> the 6x SK473 USB leads + the 2 Vantec data leads land here (8 of 10)
#     - NO Pi, NO ESP32, NO amp boards in here (amps are on the vest, inside the
#       SK473 units). Anker 737 (Mode B) velcros OUTSIDE -- it has its own plug.
#   Wire holes on the walls for the 6 audio + 6 USB + the power/uplink bundle.
#
# ONE PRINT: main() emits tactus_enclosure_plate.stl = base + (flipped) lid laid
# side-by-side on the bed, and ASSERTS the pair fits the FlashForge 5M 220x220.
# (Individual base/lid STLs are written too.)
#
# Look: rounded corners + perimeter accent groove + a big, heavy engraved TACTUS
# wordmark on the lid (no logos). The engraving is a recess -> prints support-free.
#
# Built headless with manifold3d (the exact-boolean kernel modern OpenSCAD uses)
# via trimesh -- emits watertight printable STLs directly, no GUI (OpenSCAD is
# x86 and will not run on this arm64 Mac).
#
# RUN:  /tmp/tactuscad/bin/python cad/tactus_enclosure.py
# OUT:  cad/tactus_enclosure_base.stl, _lid.stl, _plate.stl (the one-print file)
# =============================================================================

import os
import numpy as np
import trimesh
from trimesh import creation, transformations as tf

ENGINE = "manifold"
SECT   = 48
EPS    = 0.1

# ---- wall stack -------------------------------------------------------------
WALL   = 2.6
FLOOR  = 2.6
LID_T  = 2.6
CORNER = 10.0
GROOVE_D = 1.6
GROOVE_H = 3.0
REBATE_D = 2.0
REBATE_H = 4.0

# ---- internal volume (GENEROUS - extra storage if a measurement is off) -----
# Sized so the LONGEST item (the ~163 mm hub) fits, with slack, AND so base+lid
# laid side-by-side on the bed stay < 220 mm (OY*2 + GAP).  ponytail: one loose
# bin, not precise pockets -- pack the 2 Vantec + hub however they sit.
# Packing: hub stood on its 23 mm edge along one wall (163 long, 48 tall) + the
# 2 Vantec stacked beside it (58 wide, 52 tall) -> ~81 wide content. Slack added.
IX = 170.0               # length  (163 hub + 7)
IY = 92.0                # width   (81 content + 11 slack)
IZ = 60.0                # height  (52 Vantec stack + 8 slack; 48 hub-on-side fits)
PLATE_GAP = 8.0          # gap between base and lid on the one-print bed

OX = IX + 2 * WALL                        # ~175.2
OY = IY + 2 * WALL                        # ~91.2
BASE_H = FLOOR + IZ                        # ~59.6

# ---- lid screw bosses: the 4 corners ----------------------------------------
BI = WALL + 11
BOSS = [(BI, BI), (OX - BI, BI), (BI, OY - BI), (OX - BI, OY - BI)]

# =============================================================================
# primitive helpers (min-corner placement; centered cylinders by axis)
# =============================================================================
def box_at(sx, sy, sz, x0, y0, z0):
    m = creation.box(extents=(sx, sy, sz))
    m.apply_translation((x0 + sx / 2.0, y0 + sy / 2.0, z0 + sz / 2.0))
    return m

def cyl(axis, r, h, cx, cy, cz):
    """Cylinder of length h centered at (cx,cy,cz), running along axis x/y/z."""
    m = creation.cylinder(radius=r, height=h, sections=SECT)
    if axis == "x":
        m.apply_transform(tf.rotation_matrix(np.pi / 2, (0, 1, 0)))
    elif axis == "y":
        m.apply_transform(tf.rotation_matrix(np.pi / 2, (1, 0, 0)))
    m.apply_translation((cx, cy, cz))
    return m

def U(parts):
    return trimesh.boolean.union(parts, engine=ENGINE)

def D(a, b):
    return trimesh.boolean.difference([a, b], engine=ENGINE)

def rounded_box(X, Y, Z, r, z0=0.0):
    """Vertical-edge rounded rectangular prism, min-corner at (0,0,z0)."""
    parts = [
        box_at(X - 2 * r, Y, Z, r, 0, z0),
        box_at(X, Y - 2 * r, Z, 0, r, z0),
        cyl("z", r, Z, r,     r,     z0 + Z / 2),
        cyl("z", r, Z, X - r, r,     z0 + Z / 2),
        cyl("z", r, Z, r,     Y - r, z0 + Z / 2),
        cyl("z", r, Z, X - r, Y - r, z0 + Z / 2),
    ]
    return U(parts)

def outer_band(X, Y, depth, h, z0, r):
    """Ring shell = the outer `depth` of a rounded wall over Z [z0, z0+h].
    Subtract it from a part to cut a perimeter groove or a top rebate."""
    outer = rounded_box(X + 2 * EPS, Y + 2 * EPS, h, r + EPS, z0)
    outer.apply_translation((-EPS, -EPS, 0))
    inner = rounded_box(X - 2 * depth, Y - 2 * depth, h + 2 * EPS, max(r - depth, 1.0), z0 - EPS)
    inner.apply_translation((depth, depth, 0))
    return D(outer, inner)

def make_text(s, target_w, depth, weight="bold", family=None):
    """Extruded wordmark mesh (min-corner at origin), scaled to target_w wide."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    from shapely.geometry import Polygon
    from functools import reduce
    prop = FontProperties(weight=weight) if family is None else FontProperties(family=family, weight=weight)
    tp = TextPath((0, 0), s, size=20, prop=prop)
    loops = [Polygon(p).buffer(0) for p in tp.to_polygons() if len(p) >= 4]
    geom = reduce(lambda a, b: a.symmetric_difference(b), loops)   # even-odd -> holes
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    tm = trimesh.util.concatenate(
        [trimesh.creation.extrude_polygon(g, height=depth) for g in parts])
    b = tm.bounds
    tm.apply_scale([target_w / (b[1][0] - b[0][0])] * 2 + [1.0])
    b = tm.bounds
    tm.apply_translation((-b[0][0], -b[0][1], 0))
    return tm

# =============================================================================
# BASE  (one open compartment)
# =============================================================================
def build_base():
    cuts = []
    shell = rounded_box(OX, OY, BASE_H, CORNER)
    cuts.append(box_at(IX, IY, IZ + EPS, WALL, WALL, FLOOR))            # the cavity

    # ---- wire holes: 6 high on the front (-Y) wall (audio -> Vantecs) and 6 on
    #      the back (+Y) wall (USB -> hub). Round Ø9 holes, press a cable through.
    zc = FLOOR + IZ - 20.0   # hole centre; Ø16 so a USB-A plug + its overmold threads through
    for i in range(6):
        x = 24.0 + i * 25.4
        cuts.append(cyl("y", 8.0, WALL + 2 * EPS, x, WALL / 2.0, zc))           # front: SK473 audio -> Vantec
        cuts.append(cyl("y", 8.0, WALL + 2 * EPS, x, OY - WALL / 2.0, zc))      # back:  SK473 USB   -> hub
    # ---- port window on the -X short wall: the hub's ports + the power/uplink
    #      bundle reach out here (leaves wall above/below -> stays one body).
    cuts.append(box_at(WALL + 2 * EPS, 48.0, 26.0, -EPS, OY / 2.0 - 24.0, FLOOR + 10.0))

    # ---- floor vents (modest -- this gear is cool, but airflow + looks) ------
    for ix in range(5):
        for iy in range(2):
            x = 30 + ix * 30
            y = WALL + 20 + iy * 34
            if x < IX - 10 and y + 26 < OY - WALL:
                cuts.append(box_at(6.0, 26.0, FLOOR + 2 * EPS, x, y, -EPS))

    # ---- zip-tie floor grid (lash the Vantecs + hub down anywhere) ----------
    for ix in range(5):
        for iy in range(2):
            x = 26 + ix * 32
            y = WALL + 22 + iy * 36
            if x < IX - 14 and y < IY - 6:
                cuts.append(box_at(2.6, 9.0, FLOOR + 2 * EPS, x,     y, -EPS))
                cuts.append(box_at(2.6, 9.0, FLOOR + 2 * EPS, x + 6, y, -EPS))

    # ---- harness lash points: Ø4 holes through the floor near the 4 corners --
    for (lx, ly) in [(15, 15), (OX - 15, 15), (15, OY - 15), (OX - 15, OY - 15)]:
        cuts.append(cyl("z", 2.0, FLOOR + 2 * EPS, lx, ly, FLOOR / 2.0))

    # ---- belt / waistband pass-through (optional lower-back wear) ------------
    for yy in (OY / 2.0 - 27.0, OY / 2.0 + 16.0):
        cuts.append(box_at(54.0, 11.0, FLOOR + 2 * EPS, OX / 2.0 - 27.0, yy, -EPS))

    base = D(shell, U(cuts))

    # ---- adds inside the cavity: lid bosses + 2 cable anchors ----------------
    addmesh = []
    for (bx, by) in BOSS:
        boss = cyl("z", 4.5, IZ + 1.0, bx, by, FLOOR + IZ / 2.0 - 0.5)
        pilot = cyl("z", 1.3, 16.0, bx, by, FLOOR + IZ - 8 + EPS)
        addmesh.append(D(boss, pilot))
    for x in (OX * 0.32, OX * 0.68):                       # anchors by the front wall
        post = box_at(8.0, 6.0, 15.0, x - 4, WALL - 0.5, FLOOR - 1.0)
        hole = cyl("y", 1.6, 8.0, x, WALL + 3.0, FLOOR + 8.0)
        addmesh.append(D(post, hole))
    base = U([base] + addmesh)

    # ---- aesthetic: perimeter accent groove + top-edge rebate ----------------
    base = D(base, U([
        outer_band(OX, OY, GROOVE_D, GROOVE_H, 0.45 * BASE_H, CORNER),
        outer_band(OX, OY, REBATE_D, REBATE_H, BASE_H - REBATE_H, CORNER),
    ]))
    return base

# =============================================================================
# LID  (full footprint; TACTUS + lightning bolt engraved on top)
# =============================================================================
def build_lid():
    plate = rounded_box(OX, OY, LID_T, CORNER)
    # nesting lip drops into the cavity (loose ~0.8 mm/side)
    lip_o = box_at(IX - 1.6, IY - 1.6, 5.0, WALL + 0.8, WALL + 0.8, -5.0)
    lip_i = box_at(IX - 6.6, IY - 6.6, 5.0 + EPS, WALL + 3.3, WALL + 3.3, -5.0 - EPS)
    lid = U([plate, D(lip_o, lip_i)])

    cuts = []
    for (bx, by) in BOSS:
        cuts.append(cyl("z", 1.8, LID_T + 2 * EPS, bx, by, LID_T / 2.0))
        cuts.append(cyl("z", 3.4, 1.8, bx, by, LID_T - 0.9))           # countersink
    lid = D(lid, U(cuts))

    lid = D(lid, outer_band(OX, OY, REBATE_D, 1.6, LID_T - 1.6, CORNER))
    # TACTUS wordmark only -- big, heavy, centred. No logos.
    try:
        tm = make_text("TACTUS", target_w=min(142.0, OX - 26), depth=1.3 + EPS,
                       weight="black",
                       family=["Arial Black", "Impact", "Helvetica Neue", "Arial", "DejaVu Sans"])
        tb = tm.bounds
        tm.apply_translation((OX / 2.0 - tb[1][0] / 2.0, OY / 2.0 - tb[1][1] / 2.0, LID_T - 1.3))
        lid = D(lid, tm)
    except Exception as e:
        print("  (lid text engrave skipped:", e, ")")
    return lid

# =============================================================================
def cleanup(m, name):
    """Drop tiny stray boolean flecks (<0.5 cm^3); ERROR if a real feature floats."""
    parts = m.split(only_watertight=False)
    if len(parts) <= 1:
        return m
    parts = sorted(parts, key=lambda p: -p.volume)
    big = [p for p in parts[1:] if p.volume > 500.0]
    if big:
        raise SystemExit(f"FAIL {name}: {len(big)} large disconnected body(ies) "
                         f"(vol {[round(p.volume/1000,2) for p in big]} cm^3) — fix geometry")
    print(f"  {name}: dropped {len(parts)-1} stray fleck(s) <0.5cm^3 "
          f"({[round(p.volume/1000,3) for p in parts[1:]]})")
    return parts[0]

def report(name, m):
    b = m.bounds
    dims = (b[1] - b[0])
    print(f"  {name:6s} {dims[0]:6.1f} x {dims[1]:6.1f} x {dims[2]:6.1f} mm"
          f"  | watertight={m.is_watertight}  bodies={len(m.split(only_watertight=False))}"
          f"  vol={m.volume/1000:6.1f} cm^3")

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    print("Building TACTUS one-print enclosure (manifold3d)...")
    base = cleanup(build_base(), "base")
    lid = cleanup(build_lid(), "lid")
    for m in (base, lid):
        m.process(validate=True)
        if not m.is_watertight:
            trimesh.repair.fill_holes(m)
            trimesh.repair.fix_normals(m)
    print("Results:")
    report("base", base)
    report("lid", lid)
    base.export(os.path.join(here, "tactus_enclosure_base.stl"))
    lid.export(os.path.join(here, "tactus_enclosure_lid.stl"))

    # ---- ONE-PRINT plate: base + lid flipped to print-orientation, side by side
    lidp = lid.copy()
    lidp.apply_transform(tf.rotation_matrix(np.pi, (1, 0, 0)))   # engraved face -> bed, lip up
    # the X-flip sent the lid to -Y/-Z; drop its min corner onto (0, OY+GAP, 0)
    lo = lidp.bounds[0]
    lidp.apply_translation((-lo[0], OY + PLATE_GAP - lo[1], -lo[2]))   # beside base, on the bed
    plate = trimesh.util.concatenate([base, lidp])
    plate.export(os.path.join(here, "tactus_enclosure_plate.stl"))
    pd = plate.bounds[1] - plate.bounds[0]
    report("plate", plate)
    print(f"ONE-PRINT plate: {pd[0]:.0f} x {pd[1]:.0f} mm on the 220x220 bed")
    assert pd[0] <= 220 and pd[1] <= 220, f"PLATE {pd[0]:.0f}x{pd[1]:.0f} EXCEEDS the 220 bed"
    print("  base + lid fit in ONE print, support-free.  PRINT: tactus_enclosure_plate.stl")

if __name__ == "__main__":
    main()
