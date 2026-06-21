#!/usr/bin/env python3
"""
Tactus — VIDEO assets: rotating 3D semantic-space models (MP4 + GIF + stills).

Renders the exact story for the video: a high-dim audio feature vector per note,
collapsed by eigenvectors (PCA -> LDA), with the clean / buzz / muted REGIONS drawn
as 2-sigma ellipsoid volumes in a 3D space whose axes are NAMED by their top
feature loadings. Two models:
  1. rotating_semantic_lda — the supervised SEPARATION (LD1, LD2, + top PC): the
     "named regions" view (this is where issues get surfaced — a note in the buzz
     ellipsoid = a buzz).
  2. rotating_eigen_pca    — the raw eigenvector (PCA-3) space, color by class.

Plotly figures -> kaleido PNG frames around a camera orbit -> imageio MP4 (30fps)
+ looping GIF. Outputs to data/analysis/exp/video/.

Run (.venv 3.14):  python3 software/ai/analysis/exp/make_video_assets.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import schema            # noqa: E402
import numpy as np       # noqa: E402
import pandas as pd      # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # noqa: E402

MATRIX = os.path.join(schema.OUT_DIR, "all", "matrix.csv")
OUT = os.path.join(schema.OUT_DIR, "exp", "video")
COLORS = {"clean": "#22c55e", "buzz": "#ef4444", "muted": "#9ca3af"}
N_FRAMES = 90
W, H = 1180, 820


def _feats(df):
    cols = [c for c in schema.AUDIO_FEATURES if c in df.columns and not df[c].isna().all()]
    X = df[cols].to_numpy(float).copy()                    # copy: pandas 3 returns read-only views
    mu = np.nanmean(X, 0); mu = np.where(np.isfinite(mu), mu, 0.0)
    bad = ~np.isfinite(X); X[bad] = np.take(mu, np.where(bad)[1])
    return X, cols


def _top_feat(vec, names, k=2):
    idx = np.argsort(np.abs(vec))[::-1][:k]
    return " · ".join("%s%s" % ("+" if vec[i] >= 0 else "-", names[i]) for i in idx)


def _ellipsoid(P, color, name):
    """2-sigma ellipsoid Surface for a class point cloud P (n,3)."""
    mu = P.mean(0); C = np.cov(P.T)
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-9, None)
    u = np.linspace(0, 2 * np.pi, 26); v = np.linspace(0, np.pi, 18)
    sx = np.outer(np.cos(u), np.sin(v)); sy = np.outer(np.sin(u), np.sin(v)); sz = np.outer(np.ones_like(u), np.cos(v))
    sph = np.stack([sx.ravel(), sy.ravel(), sz.ravel()])           # (3, N)
    pts = (V @ np.diag(2 * np.sqrt(w)) @ sph).T + mu               # 2 sigma
    shp = sx.shape
    return go.Surface(x=pts[:, 0].reshape(shp), y=pts[:, 1].reshape(shp), z=pts[:, 2].reshape(shp),
                      opacity=0.13, showscale=False, colorscale=[[0, color], [1, color]],
                      surfacecolor=np.zeros(shp), name=name, hoverinfo="skip", showlegend=False)


def _figure(coords, labels, axis_titles, title):
    fig = go.Figure()
    for c, col in COLORS.items():
        P = coords[labels == c]
        if len(P) < 4:
            continue
        fig.add_trace(_ellipsoid(P, col, c))
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                                   name=c, marker=dict(size=3.4, color=col, opacity=0.92,
                                   line=dict(width=0))))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18, color="#e8eaf0")),
        paper_bgcolor="#0a0d13", font=dict(color="#cdd5e3"),
        scene=dict(xaxis=dict(title=axis_titles[0], backgroundcolor="#0a0d13", gridcolor="#26324a", color="#94a0b4"),
                   yaxis=dict(title=axis_titles[1], backgroundcolor="#0a0d13", gridcolor="#26324a", color="#94a0b4"),
                   zaxis=dict(title=axis_titles[2], backgroundcolor="#0a0d13", gridcolor="#26324a", color="#94a0b4"),
                   aspectmode="cube"),
        legend=dict(font=dict(size=14), itemsizing="constant", x=0.02, y=0.95),
        margin=dict(l=0, r=0, t=46, b=0))
    return fig


def _render_orbit(fig, name):
    import imageio.v2 as imageio
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, "_f"); os.makedirs(tmp, exist_ok=True)
    frames = []
    for i in range(N_FRAMES):
        ang = 2 * np.pi * i / N_FRAMES
        fig.update_layout(scene_camera=dict(eye=dict(x=1.9 * np.cos(ang), y=1.9 * np.sin(ang), z=0.85)))
        fp = os.path.join(tmp, "f%03d.png" % i)
        fig.write_image(fp, width=W, height=H, scale=1)
        frames.append(imageio.imread(fp))
    mp4 = os.path.join(OUT, name + ".mp4")
    imageio.mimsave(mp4, frames, fps=30, quality=9, macro_block_size=8)
    gif = os.path.join(OUT, name + ".gif")
    imageio.mimsave(gif, frames[::2], duration=0.066, loop=0)        # ~15fps gif, smaller
    imageio.imwrite(os.path.join(OUT, name + ".png"), frames[0])     # still
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print("  wrote %s.mp4 (%d frames) + .gif + .png" % (name, N_FRAMES))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(MATRIX)
    core = df[(df["block"] == "core-grid") & df["intended_class"].isin(list(COLORS))].copy().reset_index(drop=True)
    X, names = _feats(core)
    y = core["intended_class"].to_numpy()
    Xs = StandardScaler().fit_transform(X)

    # ---- model 1: supervised SEPARATION space (LD1, LD2, + top PC) = the named regions
    pca = PCA(n_components=0.95).fit(Xs); Xp = pca.transform(Xs)
    lda = LDA(n_components=2).fit(Xp, y); L = lda.transform(Xp)        # (n,2)
    pc1 = PCA(n_components=1).fit(Xp).transform(Xp)                    # a 3rd spread axis
    coords = np.column_stack([L[:, 0], L[:, 1], pc1[:, 0]])
    # name axes via feature-space loadings W = pca.components^T @ lda.scalings
    Wf = pca.components_.T @ lda.scalings_[:, :2]
    pc_load = pca.components_.T @ PCA(n_components=1).fit(Xp).components_.T
    ax = ["LD1  (%s)" % _top_feat(Wf[:, 0], names), "LD2  (%s)" % _top_feat(Wf[:, 1], names),
          "PC  (%s)" % _top_feat(pc_load[:, 0], names)]
    fig1 = _figure(coords, y, ax,
                   "Tactus — the geometry of guitar mistakes<br><sup>clean / buzz / muted, regions = 2σ volumes · axes named by their features</sup>")
    fig1.write_html(os.path.join(OUT, "rotating_semantic_lda.html"), include_plotlyjs="cdn")
    _render_orbit(fig1, "rotating_semantic_lda")

    # ---- model 2: raw eigenvector PCA-3 space
    p3 = PCA(n_components=3).fit(Xs); C3 = p3.transform(Xs)
    evr = p3.explained_variance_ratio_ * 100
    ax2 = ["PC1 %.0f%% (%s)" % (evr[0], _top_feat(p3.components_[0], names)),
           "PC2 %.0f%% (%s)" % (evr[1], _top_feat(p3.components_[1], names)),
           "PC3 %.0f%% (%s)" % (evr[2], _top_feat(p3.components_[2], names))]
    fig2 = _figure(C3, y, ax2,
                   "Tactus — eigenvector collapse (PCA)<br><sup>28-dim audio vector → 3 principal axes, %.0f%% of variance</sup>" % evr[:3].sum())
    fig2.write_html(os.path.join(OUT, "rotating_eigen_pca.html"), include_plotlyjs="cdn")
    _render_orbit(fig2, "rotating_eigen_pca")

    print("video assets -> %s" % OUT)


if __name__ == "__main__":
    main()
