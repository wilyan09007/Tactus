#!/usr/bin/env python3
"""Rotating GIF of the clean/buzz/muted semantic space (a drop-in-a-slide animation).

Same projection as the interactive viz: standardize → PCA(95%) → LDA (2 axes) + top
PCA component as the 3rd. Single notes only (unique event_ids → clean merge).
    .venv/bin/python software/ai/analysis/exp/make_rotating_gif.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "software/ai/analysis")
import schema

# features_fused already carries block + intended_class + the audio features, one
# row per event — no merge needed (and merging would suffix-collide intended_class).
sn = pd.read_csv("data/analysis/features_fused.csv")
sn = sn[sn.block == "core-grid"].copy()
assert sn.event_id.is_unique, "single-note ids must be unique"

X = sn[schema.AUDIO_FEATURES].apply(lambda c: c.fillna(c.median())).to_numpy()
y = sn["intended_class"].to_numpy()
Xs = StandardScaler().fit_transform(X)
Xp = PCA(0.95, svd_solver="full").fit_transform(Xs)
Xl = LDA().fit(Xp, y).transform(Xp)                 # 2 discriminant axes
pts = np.column_stack([Xl[:, 0], Xl[:, 1], Xp[:, 0]])   # + top PCA comp as axis-3

col = {"clean": "#2ca02c", "buzz": "#d62728", "muted": "#7f7f7f"}
c = [col[v] for v in y]
fig = plt.figure(figsize=(7, 6), facecolor="#0d1117")
ax = fig.add_subplot(111, projection="3d", facecolor="#0d1117")
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, s=20, alpha=.85, depthshade=True, edgecolors="none")
for lab, axis in (("LDA-1\n(hnr · mfcc1 · chroma)", ax.set_xlabel),
                  ("LDA-2\n(chroma · flatness)", ax.set_ylabel),
                  ("PC0 (brightness)", ax.set_zlabel)):
    axis(lab, color="#9aa7b4", fontsize=9)
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.set_pane_color((1, 1, 1, 0.02)); a.line.set_color("#283041")
ax.tick_params(colors="#283041", labelsize=0)
ax.set_title("Tactus — clean / buzz / muted (audio alone)", color="#e6edf3", fontsize=13)
handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=v, mec="none", label=k) for k, v in col.items()]
ax.legend(handles=handles, loc="upper left", facecolor="#161b22", labelcolor="#e6edf3", framealpha=.6)

def _f(i):
    ax.view_init(elev=16, azim=i * 4)
    return ()

out = "data/analysis/exp/rotating_clusters.gif"
FuncAnimation(fig, _f, frames=90, interval=60).save(out, writer=PillowWriter(fps=18), dpi=88)
print("wrote", out, "(%.1f MB)" % (os.path.getsize(out) / 1e6))
