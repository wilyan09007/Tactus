#!/usr/bin/env python3
"""Verify + visualize the TACTUS unified enclosure STLs. Outputs a PNG sheet."""
import os, numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
base = trimesh.load(os.path.join(HERE, "tactus_enclosure_base.stl"))
lid  = trimesh.load(os.path.join(HERE, "tactus_enclosure_lid.stl"))

for n, m in (("base", base), ("lid", lid)):
    b = m.bounds
    print(f"{n}: dims={np.round(b[1]-b[0],1)} watertight={m.is_watertight} "
          f"bodies={m.body_count} euler={m.euler_number} vol={m.volume/1000:.1f}cm3")

def iso(ax, m, title):
    T = m.triangles
    ax.add_collection3d(Poly3DCollection(T, facecolor="#7fb3d5", edgecolor="#234",
                                         linewidths=0.05, alpha=1.0))
    b = m.bounds; ctr = (b[0]+b[1])/2; r = (b[1]-b[0]).max()/2
    for setlim, c in ((ax.set_xlim, ctr[0]),(ax.set_ylim, ctr[1]),(ax.set_zlim, ctr[2])):
        setlim(c-r, c+r)
    ax.view_init(elev=28, azim=-58); ax.set_title(title, fontsize=9)
    ax.set_box_aspect((1,1,1)); ax.set_axis_off()

def slc(ax, m, normal, level, ha, va, title, lim):
    sec = m.section(plane_origin=np.array(normal)*0+level*np.array(normal),
                    plane_normal=normal)
    if sec is not None:
        for p in sec.discrete:
            ax.plot(p[:,ha], p[:,va], "-", color="#1a5276", lw=0.9)
    ax.set_title(title, fontsize=9); ax.set_aspect("equal")
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.grid(alpha=.25, lw=.4)

fig = plt.figure(figsize=(15, 9))
ax1 = fig.add_subplot(2,3,1, projection="3d"); iso(ax1, base, "BASE (iso)")
ax2 = fig.add_subplot(2,3,2, projection="3d"); iso(ax2, lid,  "LID (iso)")

# ---- component fit map (top-down) ------------------------------------------
ax3 = fig.add_subplot(2,3,3)
WALL=2.6; IX=192; EY0=2.6; EY1=114.6; PY0=118.6; PY1=188.6; OX=197.2; OY=191.2
ax3.add_patch(Rectangle((0,0), OX, OY, fill=False, ec="k", lw=1.5))
ax3.add_patch(Rectangle((WALL,EY0), IX, EY1-EY0, fc="#eaf2f8", ec="#2980b9", lw=1))   # elec bay
ax3.add_patch(Rectangle((WALL,PY0), IX, PY1-PY0, fc="#fef9e7", ec="#b9770e", lw=1))   # power bay
comp = [  # x,y,w,h,label,color
  (4.6,  8.6, 58,100, "Vantec\n7.1 #1", "#82e0aa"),
  (66.6, 8.6, 58,100, "Vantec\n7.1 #2", "#82e0aa"),
  (128,  8.6, 64,100, "amp bank\n(7x)",  "#f1948a"),
  (10.6,128.6,156,55, "Anker 737  (Mode B)", "#85c1e9"),
]
for x,y,w,h,l,c in comp:
    ax3.add_patch(Rectangle((x,y),w,h, fc=c, ec="#333", lw=.8, alpha=.75))
    ax3.text(x+w/2, y+h/2, l, ha="center", va="center", fontsize=6.5)
# hub alternative footprint (dashed) in the power bay
ax3.add_patch(Rectangle((8,121),180,65, fill=False, ec="#b9770e", lw=1, ls="--"))
ax3.text(98,118.5,"or 10-port hub  (Mode A, <=180x70)", ha="center", fontsize=6, color="#7e5109")
ax3.set_title("Component fit map (top) — generous leeway", fontsize=9)
ax3.set_aspect("equal"); ax3.set_xlim(-6, OX+6); ax3.set_ylim(-12, OY+6); ax3.grid(alpha=.25,lw=.4)

slc(fig.add_subplot(2,3,4), base, [0,0,1], 1.3, 0,1, "Floor slice z=1.3 (vents/slots)",
    ((-6,203),(-12,197)))
slc(fig.add_subplot(2,3,5), base, [0,0,1], 30, 0,1, "Mid slice z=30 (cavities/partition/bosses)",
    ((-6,203),(-12,197)))
slc(fig.add_subplot(2,3,6), base, [1,0,0], OX/2, 1,2, "Depth section x=mid (bay heights/open top)",
    ((-12,197),(-6,64)))

fig.suptitle("TACTUS unified enclosure — verification sheet", fontsize=12, y=.99)
fig.tight_layout()
out = os.path.join(HERE, "_verify_enclosure.png")
fig.savefig(out, dpi=110); print("wrote", out)
