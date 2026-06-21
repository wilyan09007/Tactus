#!/usr/bin/env python3
"""
Tactus — AUTO-ROTATING self-contained 3D HTML (for screen-recording the video).

Same semantic models as make_video_assets, but written as standalone HTML that
spins on its own (a JS camera-orbit loop) and is fully offline (plotly inlined).
Open the file -> it rotates -> screen-record. You can also drag to explore.

Outputs to data/analysis/exp/video/:
  rotating_semantic_lda_auto.html   clean/buzz/muted regions (2 sigma ellipsoids)
  rotating_eigen_pca_auto.html      raw PCA-3 eigenvector space
  rotating_chord_auto.html          the 9-chord space (if chords present)

Run (.venv 3.14):  python3 software/ai/analysis/exp/make_rotating_html.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import make_video_assets as mva   # reuse _feats, _figure, _top_feat, COLORS, MATRIX
import schema                     # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # noqa: E402

OUT = mva.OUT
# camera-orbit loop; {plot_id} is filled by plotly write_html(post_script=...)
ORBIT = """
var gd = document.getElementById('{plot_id}');
var a = 0.0;
setTimeout(function () {{
  function spin() {{
    a += 0.0045;
    Plotly.relayout(gd, {{'scene.camera.eye': {{x: 1.95*Math.cos(a), y: 1.95*Math.sin(a), z: 0.82}}}});
    requestAnimationFrame(spin);
  }}
  requestAnimationFrame(spin);
}}, 500);
"""


def _write_auto(fig, name):
    p = os.path.join(OUT, name + ".html")
    fig.write_html(p, include_plotlyjs=True, full_html=True, post_script=ORBIT,
                   config={"displayModeBar": False})
    print("  wrote", os.path.relpath(p))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(mva.MATRIX)
    core = df[(df["block"] == "core-grid") & df["intended_class"].isin(list(mva.COLORS))].copy().reset_index(drop=True)
    X, names = mva._feats(core)
    y = core["intended_class"].to_numpy()
    Xs = StandardScaler().fit_transform(X)

    # semantic LDA separation space (named regions)
    pca = PCA(n_components=0.95).fit(Xs); Xp = pca.transform(Xs)
    lda = LDA(n_components=2).fit(Xp, y); L = lda.transform(Xp)
    p1 = PCA(n_components=1).fit(Xp); pc1 = p1.transform(Xp)
    coords = np.column_stack([L[:, 0], L[:, 1], pc1[:, 0]])
    Wf = pca.components_.T @ lda.scalings_[:, :2]
    pcl = pca.components_.T @ p1.components_.T
    ax = ["LD1 (%s)" % mva._top_feat(Wf[:, 0], names), "LD2 (%s)" % mva._top_feat(Wf[:, 1], names),
          "PC (%s)" % mva._top_feat(pcl[:, 0], names)]
    fig = mva._figure(coords, y, ax,
                      "Tactus — geometry of guitar mistakes<br><sup>clean / buzz / muted · regions = 2σ volumes · axes named by features · auto-rotating (drag to explore)</sup>")
    _write_auto(fig, "rotating_semantic_lda_auto")

    # raw eigenvector PCA-3
    p3 = PCA(n_components=3).fit(Xs); C3 = p3.transform(Xs); evr = p3.explained_variance_ratio_ * 100
    ax2 = ["PC1 %.0f%% (%s)" % (evr[0], mva._top_feat(p3.components_[0], names)),
           "PC2 %.0f%% (%s)" % (evr[1], mva._top_feat(p3.components_[1], names)),
           "PC3 %.0f%% (%s)" % (evr[2], mva._top_feat(p3.components_[2], names))]
    fig2 = mva._figure(C3, y, ax2,
                       "Tactus — eigenvector collapse (PCA)<br><sup>28-D audio vector → 3 principal axes (%.0f%% of variance) · auto-rotating</sup>" % evr[:3].sum())
    _write_auto(fig2, "rotating_eigen_pca_auto")

    # chord space (9 chords) if present
    ch = df[df["block"] == "chord-stream"].copy()
    ch = ch[ch["chord_name"].notna()].reset_index(drop=True) if "chord_name" in ch.columns else ch.iloc[0:0]
    if len(ch) > 30 and ch["chord_name"].nunique() > 2:
        Xc, nc = mva._feats(ch); ycc = ch["chord_name"].to_numpy()
        Xcs = StandardScaler().fit_transform(Xc)
        pc = PCA(n_components=3).fit(Xcs); Cc = pc.transform(Xcs); evc = pc.explained_variance_ratio_ * 100
        # color by chord (qualitative) — reuse _figure but with a chord palette
        fig3 = go.Figure()
        pal = ["#22c55e", "#ef4444", "#3b82f6", "#a78bfa", "#f59e0b", "#ec4899", "#14b8a6", "#eab308", "#94a0b4"]
        for i, c in enumerate(sorted(pd.unique(ycc))):
            P = Cc[ycc == c]
            fig3.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name=str(c),
                                        marker=dict(size=3.4, color=pal[i % len(pal)], opacity=0.9)))
        fig3.update_layout(title=dict(text="Tactus — the 9-chord space (PCA)<br><sup>color = chord identity · auto-rotating</sup>",
                                      x=0.5, xanchor="center", font=dict(size=18, color="#e8eaf0")),
                           paper_bgcolor="#0a0d13", font=dict(color="#cdd5e3"),
                           scene=dict(xaxis_title="PC1 %.0f%%" % evc[0], yaxis_title="PC2 %.0f%%" % evc[1],
                                      zaxis_title="PC3 %.0f%%" % evc[2], aspectmode="cube",
                                      xaxis=dict(backgroundcolor="#0a0d13", gridcolor="#26324a", color="#94a0b4"),
                                      yaxis=dict(backgroundcolor="#0a0d13", gridcolor="#26324a", color="#94a0b4"),
                                      zaxis=dict(backgroundcolor="#0a0d13", gridcolor="#26324a", color="#94a0b4")),
                           margin=dict(l=0, r=0, t=46, b=0), legend=dict(font=dict(size=12)))
        _write_auto(fig3, "rotating_chord_auto")
    print("auto-rotating HTML ->", OUT)


if __name__ == "__main__":
    main()
