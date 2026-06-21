#!/usr/bin/env python
"""
e1_viz.py — Tactus demo-centerpiece 3D semantic visualization.

Goal: make the GEOMETRY of clean vs. buzz vs. muted single-note guitar strikes
legible to a non-technical hackathon judge. We take 26 (named) + mfcc audio
features per strike, learn a supervised projection (PCA -> LDA) that pulls the
three intended classes apart, and render it as:

  1. A rotating, double-click-openable Plotly 3D scatter (self-contained HTML).
  2. A publication-quality matplotlib 3D money-shot PNG.
  3. A clean 2D LDA-plane PNG (the two discriminant axes).

RIGOR / HONESTY
---------------
* The label IS the prompt: `intended_class` is what we asked the player to do,
  not a human re-listen. We say so on the figures.
* ONE player, ONE session -> there is no leave-one-player-out option here. We
  use StratifiedKFold for the headline accuracy and an 80/20 stratified holdout
  to *fit the projector*, then project ALL points for the picture. Single-player
  numbers are optimistic about cross-player generalization; we state this.
* Scaler / PCA / LDA are fit on the TRAIN split only. The projector that places
  every dot in the 3D scene is train-fit; test points are merely transformed.
* No overclaiming: we print held-out accuracy next to the base rate (1/3).

Axes are labeled by their dominant *named* audio features, computed in the
original feature space via  W = pca.components_.T @ lda.scalings_  so a judge
reads "LDA-1 <- buzz_band_ratio, spec_flatness, hnr" rather than "LD1".

Outputs (data/analysis/exp/):
  viz_clean_buzz_muted_3d.html
  viz_clean_buzz_muted_3d.png
  viz_lda_2d.png

Usage:  .venv/bin/python software/ai/analysis/exp/e1_viz.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib

matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
# Resolve repo root from this file: software/ai/analysis/exp/e1_viz.py -> repo
REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "data" / "analysis"
OUT = DATA / "exp"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
PCA_VAR = 0.95          # keep components explaining 95% variance
TEST_SIZE = 0.20        # stratified holdout used to FIT the projector
N_SPLITS = 5            # StratifiedKFold for the headline accuracy

# The prompt's AUDIO_FEATURES list (15 named + 13 MFCC = 28 columns).
# (The brief says "26"; the enumerated list is 28. We use the enumerated list.)
NAMED = [
    "spec_centroid", "spec_bandwidth", "spec_flatness", "spec_rolloff",
    "spec_flux", "zcr", "rms", "log_attack_time", "attack_slope",
    "decay_rate", "hnr", "inharmonicity", "buzz_band_ratio",
    "pitch_cents_dev", "chroma_peak",
]
AUDIO_FEATURES = NAMED + [f"mfcc_{i}" for i in range(1, 14)]

CLASSES = ["clean", "buzz", "muted"]
COLORS = {"clean": "#2ca02c", "buzz": "#d62728", "muted": "#7f7f7f"}
# Marker styling tuned so grey "muted" still reads on a dark background.
EDGE = {"clean": "#0d3d12", "buzz": "#5c1010", "muted": "#2b2b2b"}

DARK_BG = "#0e1117"
PANEL_BG = "#11151c"
GRID = "#2a2f3a"
FG = "#e6e6e6"


# ----------------------------------------------------------------------------
# 1. Load + merge single-note events with audio features
# ----------------------------------------------------------------------------
def load_single_notes() -> pd.DataFrame:
    ev = pd.read_csv(DATA / "events.csv")
    ff = pd.read_csv(DATA / "features_fused.csv")

    # Single notes only.
    ev = ev[ev["block"] == "core-grid"].copy()

    meta_cols = ["event_id", "intended_class", "string_num", "target_fret", "chord_name"]
    ev = ev[[c for c in meta_cols if c in ev.columns]]

    # features_fused contains chord-stream rows too (whose ids repeat). Restrict
    # the feature table to the single-note ids first; those are unique in both
    # files (verified), so the merge is genuinely one-to-one.
    feat = ff.loc[ff["event_id"].isin(ev["event_id"]), ["event_id"] + AUDIO_FEATURES].copy()

    df = ev.merge(feat, on="event_id", how="inner", validate="one_to_one")
    df = df[df["intended_class"].isin(CLASSES)].reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# 2. Build the supervised projector (fit on TRAIN only) + report accuracy
# ----------------------------------------------------------------------------
def build_projection(df: pd.DataFrame):
    X_raw = df[AUDIO_FEATURES].to_numpy(dtype=float)
    y = df["intended_class"].to_numpy()

    # Median-impute any NaNs (only `inharmonicity` has a few). Impute stats are
    # computed on the full single-note set here purely to fill holes before the
    # train/test split; the discriminative fit (scaler/PCA/LDA) is train-only.
    col_median = np.nanmedian(X_raw, axis=0)
    nan_mask = np.isnan(X_raw)
    n_imputed = int(nan_mask.sum())
    X = X_raw.copy()
    if n_imputed:
        fill = np.broadcast_to(col_median, X.shape)
        X[nan_mask] = fill[nan_mask]

    # ---- Headline accuracy: StratifiedKFold over the full set --------------
    # Pipeline so every fold fits scaler/PCA/LDA on its own train portion.
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=PCA_VAR, random_state=RANDOM_STATE)),
        ("lda", LDA()),
    ])
    y_cv = cross_val_predict(cv_pipe, X, y, cv=skf)
    cv_acc = accuracy_score(y, y_cv)

    # ---- Projector for the picture: fit on an 80/20 stratified TRAIN split --
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    pca = PCA(n_components=PCA_VAR, random_state=RANDOM_STATE).fit(Xtr_s)
    Xtr_p, Xte_p = pca.transform(Xtr_s), pca.transform(Xte_s)

    lda = LDA().fit(Xtr_p, ytr)
    holdout_acc = accuracy_score(yte, lda.predict(Xte_p))

    # 3-class LDA -> 2 discriminant axes. For a 3D view we add the strongest
    # *remaining* PCA component (the PC whose direction is most orthogonal to
    # what LDA already used = the highest-variance PC with smallest projection
    # onto the LDA scalings). Simpler, defensible choice: the top-variance PC
    # not dominated by LDA. We take PC0 if LDA barely uses it, else the next.
    scal = lda.scalings_[:, :2]                       # (n_pca, 2)
    # How much each PCA axis contributes to the 2 LDA axes (row energy):
    lda_use = np.linalg.norm(scal, axis=1)            # (n_pca,)
    # Rank PCs by variance (PCA already sorts desc); pick the highest-variance
    # PC with the LOWEST LDA usage so axis-3 adds genuinely new structure.
    var_rank = np.arange(pca.n_components_)           # 0 = most variance
    score = var_rank + (lda_use / (lda_use.max() + 1e-12)) * pca.n_components_
    pc3_idx = int(np.argmin(score))

    # Project ALL points with the train-fit transforms.
    X_all_s = scaler.transform(X)
    X_all_p = pca.transform(X_all_s)
    lda_coords = lda.transform(X_all_p)               # (N, 2)
    pc3 = X_all_p[:, pc3_idx]                          # (N,)

    coords = np.column_stack([lda_coords[:, 0], lda_coords[:, 1], pc3])

    # ---- Named feature loadings -------------------------------------------
    # Map the LDA axes back to the ORIGINAL feature space:
    #   W = pca.components_.T @ lda.scalings_   (n_features, n_lda_axes)
    W = pca.components_.T @ lda.scalings_              # (n_feat, 2)
    # Axis 3 is a single PCA component: its feature loading is that PC's row.
    w3 = pca.components_[pc3_idx]                      # (n_feat,)

    def top_feats(weights, k=3):
        order = np.argsort(-np.abs(weights))[:k]
        return [(AUDIO_FEATURES[i], float(weights[i])) for i in order]

    loadings = {
        "LDA-1": top_feats(W[:, 0]),
        "LDA-2": top_feats(W[:, 1]),
        f"PC{pc3_idx}": top_feats(w3),
    }
    axis_titles = {
        "x": "LDA-1  ←  " + ", ".join(f for f, _ in loadings["LDA-1"]),
        "y": "LDA-2  ←  " + ", ".join(f for f, _ in loadings["LDA-2"]),
        "z": f"PC{pc3_idx} (residual var)  ←  " + ", ".join(f for f, _ in loadings[f"PC{pc3_idx}"]),
    }

    info = {
        "cv_acc": cv_acc,
        "holdout_acc": holdout_acc,
        "base_rate": 1.0 / len(CLASSES),
        "n": len(y),
        "n_train": len(ytr),
        "n_test": len(yte),
        "n_pca": int(pca.n_components_),
        "pca_var": float(pca.explained_variance_ratio_.sum()),
        "pc3_idx": pc3_idx,
        "n_imputed": n_imputed,
        "loadings": loadings,
        "axis_titles": axis_titles,
        "confusion": confusion_matrix(yte, lda.predict(Xte_p), labels=CLASSES),
        "report": classification_report(yte, lda.predict(Xte_p), labels=CLASSES, digits=3),
        "cv_confusion": confusion_matrix(y, y_cv, labels=CLASSES),
    }
    return coords, y, info


# ----------------------------------------------------------------------------
# 3. Plotly interactive 3D HTML (self-contained, auto-rotating)
# ----------------------------------------------------------------------------
def make_plotly(coords, y, info, path: Path):
    at = info["axis_titles"]
    traces = []
    for cls in CLASSES:
        m = y == cls
        traces.append(go.Scatter3d(
            x=coords[m, 0], y=coords[m, 1], z=coords[m, 2],
            mode="markers",
            name=f"{cls}  (n={int(m.sum())})",
            marker=dict(
                size=5,
                color=COLORS[cls],
                opacity=0.88,
                line=dict(width=0.6, color=EDGE[cls]),
            ),
            hovertemplate=(f"<b>{cls}</b><br>"
                           f"{at['x'].split('  ')[0]}: %{{x:.2f}}<br>"
                           f"{at['y'].split('  ')[0]}: %{{y:.2f}}<br>"
                           f"{at['z'].split('  ')[0]}: %{{z:.2f}}<extra></extra>"),
        ))

    # ---- Auto-rotation via camera-orbit frames -----------------------------
    n_frames = 60
    r, z_eye = 1.9, 0.55
    frames = []
    for k in range(n_frames):
        ang = 2 * np.pi * k / n_frames
        cam = dict(eye=dict(x=r * np.cos(ang), y=r * np.sin(ang), z=z_eye))
        frames.append(go.Frame(layout=dict(scene_camera=cam), name=str(k)))

    acc_pct = f"{info['holdout_acc']*100:.0f}%"
    cv_pct = f"{info['cv_acc']*100:.0f}%"
    base_pct = f"{info['base_rate']*100:.0f}%"

    title = (
        "<b>Tactus — the sound of a guitar mistake, in 3D</b><br>"
        "<span style='font-size:15px;color:#9aa0aa'>"
        "Each dot is one note strike, placed by a supervised audio projection "
        f"(PCA→LDA). Held-out accuracy {acc_pct} · 5-fold {cv_pct} "
        f"· chance {base_pct}. The label is the on-screen prompt.</span>"
    )

    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        title=dict(text=title, x=0.02, xanchor="left", y=0.96,
                   font=dict(size=26, color=FG)),
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=14, color=FG),
        legend=dict(title=dict(text="<b>intended class</b>"),
                    bgcolor="rgba(20,24,32,0.65)", bordercolor=GRID, borderwidth=1,
                    x=0.86, y=0.92, font=dict(size=15)),
        margin=dict(l=0, r=0, t=110, b=0),
        scene=dict(
            xaxis=dict(title=dict(text=at["x"], font=dict(size=12, color="#b9c0cc")),
                       backgroundcolor=PANEL_BG, gridcolor=GRID, zerolinecolor=GRID,
                       showspikes=False),
            yaxis=dict(title=dict(text=at["y"], font=dict(size=12, color="#b9c0cc")),
                       backgroundcolor=PANEL_BG, gridcolor=GRID, zerolinecolor=GRID,
                       showspikes=False),
            zaxis=dict(title=dict(text=at["z"], font=dict(size=12, color="#b9c0cc")),
                       backgroundcolor=PANEL_BG, gridcolor=GRID, zerolinecolor=GRID,
                       showspikes=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=r, y=0.0, z=z_eye)),
        ),
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.02, y=0.04, xanchor="left", yanchor="bottom",
            bgcolor="#1c2430", bordercolor=GRID, font=dict(color=FG, size=13),
            buttons=[
                dict(label="▶  Rotate", method="animate",
                     args=[None, dict(frame=dict(duration=70, redraw=True),
                                      transition=dict(duration=0),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸  Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
        )],
    )

    # Caption with the named-loading legend + honesty note.
    ld = info["loadings"]
    pc_key = f"PC{info['pc3_idx']}"
    def fmt(ax):
        return ", ".join(f"{f} ({w:+.2f})" for f, w in ld[ax])
    caption = (
        "Axis = direction in raw audio-feature space (W = PCAᵀ · LDA scalings), "
        "top-3 |loading|:<br>"
        f"&nbsp;&nbsp;<b>LDA-1</b>: {fmt('LDA-1')}<br>"
        f"&nbsp;&nbsp;<b>LDA-2</b>: {fmt('LDA-2')}<br>"
        f"&nbsp;&nbsp;<b>{pc_key}</b>: {fmt(pc_key)}<br>"
        "<i>Projector fit on an 80/20 stratified TRAIN split, then applied to all "
        f"{info['n']} points. One player, one session — single-player numbers are "
        "optimistic about new players.</i>"
    )
    fig.add_annotation(
        text=caption, xref="paper", yref="paper", x=0.0, y=-0.02,
        xanchor="left", yanchor="top", align="left", showarrow=False,
        font=dict(size=11.5, color="#8b929e"),
    )

    fig.write_html(
        str(path),
        include_plotlyjs=True,   # fully offline / double-click openable
        full_html=True,
        auto_open=False,
        config=dict(displaylogo=False,
                    toImageButtonOptions=dict(format="png", scale=3,
                                              filename="tactus_clean_buzz_muted_3d")),
    )
    return path


# ----------------------------------------------------------------------------
# 4 + 5. Matplotlib money shots (3D PNG + 2D LDA PNG)
# ----------------------------------------------------------------------------
def _style_dark(fig, ax3d=False):
    fig.patch.set_facecolor(DARK_BG)


def make_png_3d(coords, y, info, path: Path):
    at = info["axis_titles"]
    fig = plt.figure(figsize=(13, 10.5), dpi=200)
    fig.patch.set_facecolor(DARK_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(DARK_BG)

    for cls in CLASSES:
        m = y == cls
        ax.scatter(coords[m, 0], coords[m, 1], coords[m, 2],
                   s=34, c=COLORS[cls], edgecolors=EDGE[cls], linewidths=0.4,
                   alpha=0.9, depthshade=True, label=f"{cls}  (n={int(m.sum())})")

    # Pane + grid styling for a dark, publication look.
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.066, 0.082, 0.11, 1.0))
        pane._axinfo["grid"]["color"] = (0.18, 0.20, 0.25, 1.0)
        pane.label.set_color("#b9c0cc")
    ax.tick_params(colors="#6b7280", labelsize=8)

    ax.set_xlabel(_wrap(at["x"]), fontsize=10, labelpad=12, color="#b9c0cc")
    ax.set_ylabel(_wrap(at["y"]), fontsize=10, labelpad=12, color="#b9c0cc")
    ax.set_zlabel(_wrap(at["z"]), fontsize=9, labelpad=8, color="#b9c0cc")

    acc = f"{info['holdout_acc']*100:.0f}%"
    cv = f"{info['cv_acc']*100:.0f}%"
    base = f"{info['base_rate']*100:.0f}%"
    fig.suptitle("Tactus — clean vs. buzz vs. muted, learned from sound alone",
                 fontsize=20, color=FG, fontweight="bold", y=0.965)
    ax.set_title(f"supervised PCA→LDA projection  ·  held-out {acc}  ·  "
                 f"5-fold {cv}  ·  chance {base}",
                 fontsize=12, color="#9aa0aa", pad=16)

    leg = ax.legend(loc="upper left", frameon=True, fontsize=12,
                    facecolor="#161b24", edgecolor=GRID, labelcolor=FG,
                    title="intended class")
    leg.get_title().set_color(FG)
    leg.get_title().set_fontweight("bold")

    ax.view_init(elev=18, azim=-58)
    fig.text(0.5, 0.015,
             "The label is the on-screen prompt · projector fit on 80/20 train split, "
             "all points shown · one player / one session (optimistic for new players)",
             ha="center", fontsize=9.5, color="#7b8290", style="italic")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.06)
    fig.savefig(path, facecolor=DARK_BG, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return path


def make_png_2d(coords, y, info, path: Path):
    """The two LDA discriminant axes — the cleanest possible separation view."""
    at = info["axis_titles"]
    fig, ax = plt.subplots(figsize=(12, 9.5), dpi=200)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    for cls in CLASSES:
        m = y == cls
        ax.scatter(coords[m, 0], coords[m, 1],
                   s=72, c=COLORS[cls], edgecolors=EDGE[cls], linewidths=0.6,
                   alpha=0.85, label=f"{cls}  (n={int(m.sum())})")
        # 2-sigma covariance ellipse to make cluster shape obvious.
        _confidence_ellipse(coords[m, 0], coords[m, 1], ax,
                            edgecolor=COLORS[cls], facecolor="none",
                            lw=1.6, ls="--", alpha=0.55, n_std=2.0)

    ax.set_xlabel(_wrap(at["x"], 70), fontsize=12.5, color="#c4cad4", labelpad=10)
    ax.set_ylabel(_wrap(at["y"], 70), fontsize=12.5, color="#c4cad4", labelpad=10)
    ax.tick_params(colors="#6b7280", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)

    acc = f"{info['holdout_acc']*100:.0f}%"
    cv = f"{info['cv_acc']*100:.0f}%"
    base = f"{info['base_rate']*100:.0f}%"
    ax.set_title("Tactus — the LDA discriminant plane of guitar-note quality\n"
                 f"clean / buzz / muted from audio  ·  held-out {acc}  ·  "
                 f"5-fold {cv}  ·  chance {base}",
                 fontsize=16.5, color=FG, fontweight="bold", pad=16)

    leg = ax.legend(loc="best", frameon=True, fontsize=13,
                    facecolor="#161b24", edgecolor=GRID, labelcolor=FG,
                    title="intended class")
    leg.get_title().set_color(FG)
    leg.get_title().set_fontweight("bold")

    fig.text(0.5, 0.012,
             "Dashed = 2σ cluster ellipse · axes labeled by dominant raw audio "
             "features · label = prompt · projector train-fit · one player/session",
             ha="center", fontsize=9.5, color="#7b8290", style="italic")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.10)
    fig.savefig(path, facecolor=DARK_BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _wrap(label: str, width: int = 46) -> str:
    """Soft-wrap a long axis label after the arrow for readability."""
    if "←" in label and len(label) > width:
        head, tail = label.split("←", 1)
        return head + "←\n   " + tail.strip()
    return label


def _confidence_ellipse(x, y, ax, n_std=2.0, **kw):
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    if x.size < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    rx, ry = np.sqrt(1 + pearson), np.sqrt(1 - pearson)
    ell = Ellipse((0, 0), width=rx * 2, height=ry * 2, **kw)
    sx, sy = np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std
    tr = (transforms.Affine2D().rotate_deg(45).scale(sx, sy)
          .translate(np.mean(x), np.mean(y)))
    ell.set_transform(tr + ax.transData)
    ax.add_patch(ell)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    df = load_single_notes()
    if df.empty:
        sys.exit("No single-note (core-grid) events after merge — check inputs.")

    coords, y, info = build_projection(df)

    html_path = OUT / "viz_clean_buzz_muted_3d.html"
    png3d_path = OUT / "viz_clean_buzz_muted_3d.png"
    png2d_path = OUT / "viz_lda_2d.png"

    make_plotly(coords, y, info, html_path)
    make_png_3d(coords, y, info, png3d_path)
    make_png_2d(coords, y, info, png2d_path)

    # ---- Report ------------------------------------------------------------
    line = "=" * 74
    print(line)
    print("TACTUS e1_viz — single-note clean/buzz/muted 3D semantic projection")
    print(line)
    print(f"events (single-note, core-grid)     : {info['n']}  "
          f"(clean/buzz/muted balanced)")
    print(f"train / test (projector fit split)  : {info['n_train']} / {info['n_test']}")
    print(f"NaNs median-imputed                 : {info['n_imputed']} cells")
    print(f"PCA(95% var)                        : {info['n_pca']} comps "
          f"({info['pca_var']*100:.1f}% var)")
    print(f"axis-3 = strongest residual PC      : PC{info['pc3_idx']}")
    print("-" * 74)
    print(f"HELD-OUT 3-class accuracy (80/20)   : {info['holdout_acc']*100:.1f}%")
    print(f"StratifiedKFold (k={N_SPLITS}) accuracy   : {info['cv_acc']*100:.1f}%")
    print(f"base rate (chance, 3 classes)       : {info['base_rate']*100:.1f}%")
    print(f"CAVEAT: one player / one session — no leave-one-player-out possible;")
    print(f"        numbers are optimistic about generalizing to new players.")
    print("-" * 74)
    print("Held-out confusion (rows=true, cols=pred)  order:", CLASSES)
    print(info["confusion"])
    print("\nHeld-out classification report:")
    print(info["report"])
    print("-" * 74)
    print("NAMED-FEATURE AXIS LOADINGS (top-3 |loading| in raw feature space)")
    for ax_name, feats in info["loadings"].items():
        pretty = ", ".join(f"{f} ({w:+.3f})" for f, w in feats)
        print(f"  {ax_name:6s} <-  {pretty}")
    print("-" * 74)
    print("ARTIFACTS")
    for p in (html_path, png3d_path, png2d_path):
        sz = p.stat().st_size / 1024
        print(f"  {p}  ({sz:.0f} KB)")
    print(line)


if __name__ == "__main__":
    main()
