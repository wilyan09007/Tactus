#!/usr/bin/env python3
# =============================================================================
# Headless renderer for actuator_puck.scad — faithful 1:1 port of the OpenSCAD
# geometry, used because the installed OpenSCAD is an x86 2021.01 build that
# won't run on this Apple-Silicon Mac (no Rosetta). Same manifold3d kernel as
# tactus_enclosure.py. Mirrors the .scad EXACTLY (validated: the 40 mm .scad
# output 63.4×45.4×10.4 matches this port's formula at spk_dia=40).
#
# Source of truth for params remains actuator_puck.scad. Keep them in sync.
# RUN: /tmp/tactuscad/bin/python cad/_puck_render.py
# OUT: cad/actuator_cup.stl , cad/actuator_button.stl
# =============================================================================
import os, numpy as np, trimesh
from trimesh import creation, transformations as tf

ENG, S = "manifold", 96

# ---- params (MUST match actuator_puck.scad) ---------------------------------
spk_dia, spk_depth, spk_fit_gap = 52.0, 27.0, 0.6     # SK473 driver: 52 mm OD
wall, back = 2.4, 2.4
btn_dia, btn_dome_h, btn_base_h = 14.0, 4.5, 1.6
notch_w, notch_h = 6.0, 5.0
cup_id = spk_dia + spk_fit_gap                         # 52.6
cup_od = cup_id + 2 * wall                             # 57.4
cup_h  = back + spk_depth + 1.0                        # 30.4

def U(p): return trimesh.boolean.union(p, engine=ENG)
def D(a, b): return trimesh.boolean.difference([a, b], engine=ENG)
def cylz(d, h, x=0, y=0, z=0):                         # base at z (OpenSCAD convention)
    m = creation.cylinder(radius=d/2, height=h, sections=S)
    m.apply_translation((x, y, z + h/2)); return m
def cyly(d, h, x=0, y=0, z=0):                         # axis along Y, centered at (x,y,z)
    m = creation.cylinder(radius=d/2, height=h, sections=S)
    m.apply_transform(tf.rotation_matrix(np.pi/2, (1, 0, 0)))
    m.apply_translation((x, y, z)); return m
def box(sx, sy, sz, x, y, z):                          # min corner at (x,y,z)
    m = creation.box(extents=(sx, sy, sz))
    m.apply_translation((x+sx/2, y+sy/2, z+sz/2)); return m

def cup():
    solid = U([
        cylz(cup_od, cup_h),                                   # body
        cylz(10, 3,  cup_od/2 + 4, 0),                         # ear +X (overlaps body 1 mm)
        cylz(10, 3, -(cup_od/2 + 4), 0),                       # ear -X
    ])
    cuts = U([
        cylz(cup_id, cup_h, 0, 0, back),                       # speaker bore (closed back)
        box(notch_w, wall + 1.5, notch_h, -notch_w/2, cup_od/2 - wall - 0.5, back + 1.5),  # wire notch (+Y)
        cyly(2.6, wall + 2, 0, cup_od/2 - 1.0, back + 3.5),    # zip-tie strain-relief hole
        cylz(3.4, 5,  cup_od/2 + 4, 0, -0.5),                  # ear hole +X
        cylz(3.4, 5, -(cup_od/2 + 4), 0, -0.5),                # ear hole -X
        cylz(4, back + 1, 0, 0, -0.5),                         # back vent (cone breathes)
    ])
    # NOTE: the .scad's rim_lip subtraction is a geometric no-op (it lies inside
    # the bore), so it is omitted here — the resulting solid is identical.
    return D(solid, cuts)

def button():
    base = cylz(btn_dia, btn_base_h)
    dome = creation.icosphere(subdivisions=3, radius=btn_dia/2)
    dome.apply_scale([1, 1, btn_dome_h / (btn_dia/2)])
    dome.apply_translation((0, 0, btn_base_h))
    return U([base, dome])

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for name, m, exp in (("actuator_cup", cup(), (75.4, 57.4, 30.4)),
                         ("actuator_button", button(), (14.0, 14.0, 9.0))):
        m.process(validate=True)
        d = (m.bounds[1] - m.bounds[0])
        p = os.path.join(here, name + ".stl")
        m.export(p)
        ok = all(abs(a - b) < 0.6 for a, b in zip(d, exp))
        print(f"{name}: dims={d.round(2)} (expect {exp}) watertight={m.is_watertight} "
              f"bodies={m.body_count} vol={m.volume/1000:.2f}cm3  {'OK' if ok else 'CHECK'}")

if __name__ == "__main__":
    main()
