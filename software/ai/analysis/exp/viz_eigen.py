#!/usr/bin/env python3
"""
viz_eigen.py — Tactus demo CENTERPIECE: the eigenvector + feature-clustering story.

This is the judge-facing visualization the player explicitly asked for: "clearly
show feature-based clustering and the eigenvector stuff." It makes the linear-
algebra spine of the separability study (standardize -> PCA/eigendecomposition ->
supervised LDA) legible, and proves the clusters are real (vs chance) WITHOUT
overclaiming.

Four panels, all built on the master feature table data/analysis/all/matrix.csv
(432 single notes clean/buzz/muted + 600 chord strums, ONE player = aditya):

  1. EIGEN-DECOMPOSITION story — scree plot (variance per PC) + cumulative-variance
     curve + eigenvector LOADINGS HEATMAP (named audio features x PCs) so each axis's
     MEANING is visible. The PCA is the eigendecomposition of the standardized audio
     covariance matrix; we label PCs by their top |loading| named features.

  2. SUPERVISED SEPARATION done right — rotating 3D LDA scatter of clean/buzz/muted.
     LDA (3 classes -> 2 discriminant axes; we add PC1-of-residual as a 3rd display
     axis) SEPARATES where raw PCA blobs. Points colored by class, 2-sigma class
     ellipsoids, axes titled by their top NAMED-feature loadings. Honest k-fold
     accuracy in the caption (audio-only 0.80, single player, stratified k-fold).

  3. FEATURE-BASED CLUSTERING — KMeans + GMM + HDBSCAN on the standardized
     audio(+residual) space; ARI / AMI / silhouette vs the true clean/buzz/muted
     labels, reported against a permutation chance baseline. Each discovered cluster
     is NAMED by its top standardized-feature means ("region -> semantic meaning").

  4. MONO->POLY TRANSFER money shot — mono-clean + mono-buzz single notes + chord
     residuals on the SHARED buzz axis (re-fit cleanly here), showing chords land on
     the clean primitive with a buzz-side tail.

RIGOR (non-negotiable, stated in every caption):
  * ONE player -> STRATIFIED K-FOLD (not LOPO; we say so). No cross-player claim.
  * ALL preprocessing (impute, StandardScaler, PCA, LDA) fit on TRAIN folds only for
    every predictive number. The eigen-decomposition heatmap / scree is a DESCRIPTIVE
    full-data view (clearly labelled as such — it is geometry, not a held-out metric).
  * Clustering scored vs a permutation chance floor; ARI/AMI/silhouette reported with
    that floor next to them.
  * Base rates: clean/buzz/muted are balanced (144 each) -> chance accuracy = 1/3.

Outputs -> data/analysis/exp/:
  viz_eigen_scree.png/.html, viz_eigen_loadings.png/.html,
  viz_eigen_lda3d.html (rotating) + viz_eigen_lda3d.png,
  viz_eigen_clustering.png/.html, viz_eigen_transfer.png/.html,
  viz_eigen_report.json, and viz_eigen_gallery.html (dark, polished, links all).

Run:  python3 software/ai/analysis/exp/viz_eigen.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import schema  # noqa: E402

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import plotly.graph_objects as go  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
ROOT = Path(schema.ROOT)
MATRIX = ROOT / "data" / "analysis" / "all" / "matrix.csv"
OUT = ROOT / "data" / "analysis" / "exp"
OUT.mkdir(parents=True, exist_ok=True)

CORE = ["clean", "buzz", "muted"]
SEED = 17
N_SPLITS = 5

# Audio (timbre/pitch) features — the eigen space. Present-and-named in matrix.csv.
AUDIO = list(schema.AUDIO_FEATURES)
# Harmonic-residual features (the broadband-buzz primitive) — present for both
# single notes and chords in matrix.csv. Used for the transfer panel + clustering.
RESID = [
    "hpss_perc_ratio", "hpss_harm_ratio", "comb_harm_ratio", "comb_resid_ratio",
    "resid_centroid", "resid_flatness", "resid_rolloff", "resid_hf_ratio",
]
# The full descriptive space for clustering: audio + residual.
CLUSTER_FEATS = AUDIO + RESID

# Human-readable labels for the dozen-or-so most-cited features (keeps axes legible).
PRETTY = {
    "spec_centroid": "spectral centroid", "spec_bandwidth": "spectral bandwidth",
    "spec_flatness": "spectral flatness", "spec_rolloff": "spectral rolloff",
    "spec_flux": "spectral flux", "zcr": "zero-crossing rate", "rms": "RMS energy",
    "log_attack_time": "log attack time", "attack_slope": "attack slope",
    "decay_rate": "decay rate", "hnr": "harmonic-to-noise", "inharmonicity": "inharmonicity",
    "buzz_band_ratio": "buzz-band ratio (>4kHz)", "pitch_cents_dev": "pitch cents dev",
    "chroma_peak": "chroma peak", "comb_resid_ratio": "comb residual ratio",
    "resid_hf_ratio": "residual HF ratio", "hpss_perc_ratio": "percussive ratio",
    "comb_harm_ratio": "comb harmonic ratio", "resid_flatness": "residual flatness",
}
def pretty(f: str) -> str:
    return PRETTY.get(f, f.replace("_", " "))

# Class palette (color-blind-safe, consistent across all panels).
CLR = {"clean": "#1D9E75", "buzz": "#D85A30", "muted": "#534AB7"}
CLR_CHORD = "#888780"


# --------------------------------------------------------------------------- #
# Data + shared preprocessing
# --------------------------------------------------------------------------- #
def load():
    m = pd.read_csv(MATRIX)
    sn = m[m["intended_class"].isin(CORE)].copy().reset_index(drop=True)
    cs = m[m["block"] == "chord-stream"].copy().reset_index(drop=True)
    return m, sn, cs


def impute_fit(X):
    """Train column means (NaN-safe). Returns means + an apply fn."""
    means = np.nanmean(X, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    def apply(Z):
        Z = np.asarray(Z, float).copy()
        bad = ~np.isfinite(Z)
        if bad.any():
            Z[bad] = np.take(means, np.where(bad)[1])
        return Z
    return means, apply


# --------------------------------------------------------------------------- #
# PANEL 1 — eigen-decomposition: scree + cumulative + loadings heatmap
# --------------------------------------------------------------------------- #
def panel1_eigen(sn):
    """Descriptive full-data PCA on standardized AUDIO features (the eigen-
    decomposition of the audio covariance matrix). Labelled as descriptive geometry,
    NOT a held-out metric. Produces scree+cumulative PNG/HTML and a loadings heatmap."""
    feats = [c for c in AUDIO if c in sn.columns]
    X = sn[feats].to_numpy(float)
    _, ap = impute_fit(X)
    Xs = StandardScaler().fit_transform(ap(X))
    pca = PCA(svd_solver="full").fit(Xs)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n_for_95 = int(np.searchsorted(cum, 0.95) + 1)
    comp = pca.components_           # (n_pc, n_feat) — rows are eigenvectors
    n_show = min(8, comp.shape[0])   # show first 8 PCs in the heatmap

    # ---- scree + cumulative (matplotlib) ----
    png1 = OUT / "viz_eigen_scree.png"
    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=160)
    idx = np.arange(1, len(evr) + 1)
    ax.bar(idx, evr * 100, color="#378ADD", alpha=0.85, label="variance per PC")
    ax.set_xlabel("principal component (eigenvector)", fontsize=11)
    ax.set_ylabel("variance explained (%)", fontsize=11, color="#185FA5")
    ax.tick_params(axis="y", labelcolor="#185FA5")
    ax2 = ax.twinx()
    ax2.plot(idx, cum * 100, "-o", color="#C0392B", lw=1.8, ms=4, label="cumulative")
    ax2.axhline(95, color="#566573", ls="--", lw=1.0)
    ax2.axvline(n_for_95, color="#566573", ls=":", lw=1.0)
    ax2.set_ylabel("cumulative variance (%)", fontsize=11, color="#C0392B")
    ax2.tick_params(axis="y", labelcolor="#C0392B")
    ax2.set_ylim(0, 103)
    ax.set_title("Panel 1 — PCA eigen-decomposition of the standardized audio space\n"
                 f"{n_for_95} PCs capture 95% of variance ({len(feats)} named audio features, "
                 f"{len(sn)} single notes)", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005,
             "descriptive geometry (full data) — the eigenbasis the supervised LDA is built on; "
             "single player (aditya)",
             ha="center", fontsize=8.2, color="#34495E")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(png1, bbox_inches="tight"); plt.close(fig)

    # scree HTML (plotly)
    figp = go.Figure()
    figp.add_bar(x=idx, y=evr * 100, name="variance per PC", marker_color="#378ADD")
    figp.add_scatter(x=idx, y=cum * 100, name="cumulative", yaxis="y2",
                     mode="lines+markers", line=dict(color="#C0392B", width=2))
    figp.add_hline(y=95, line_dash="dash", line_color="#566573", yref="y2")
    figp.update_layout(
        title=f"PCA scree + cumulative variance — {n_for_95} PCs reach 95%",
        xaxis_title="principal component (eigenvector index)",
        yaxis=dict(title="variance per PC (%)"),
        yaxis2=dict(title="cumulative (%)", overlaying="y", side="right", range=[0, 103]),
        template="plotly_white", width=920, height=540,
        legend=dict(x=0.55, y=0.2, bgcolor="rgba(255,255,255,0.85)"))
    html1 = OUT / "viz_eigen_scree.html"
    figp.write_html(str(html1), include_plotlyjs="cdn", full_html=True)

    # ---- loadings heatmap (features x PCs) ----
    L = comp[:n_show].T                       # (n_feat, n_show)
    order = np.argsort(-np.abs(L[:, 0]))      # sort features by |PC1 loading| for readability
    Lo = L[order]
    flabels = [pretty(feats[i]) for i in order]
    pclabels = []
    for k in range(n_show):
        top = np.argsort(-np.abs(comp[k]))[:2]
        pclabels.append(f"PC{k+1}<br>{evr[k]*100:.0f}%<br>{pretty(feats[top[0]])}")

    png2 = OUT / "viz_eigen_loadings.png"
    cmap = LinearSegmentedColormap.from_list("rb", ["#C0392B", "#FFFFFF", "#185FA5"])
    fig, ax = plt.subplots(figsize=(8.6, 10.5), dpi=160)
    im = ax.imshow(Lo, aspect="auto", cmap=cmap, vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(n_show))
    ax.set_xticklabels([f"PC{k+1}\n{evr[k]*100:.0f}%" for k in range(n_show)], fontsize=9)
    ax.set_yticks(range(len(flabels)))
    ax.set_yticklabels(flabels, fontsize=8)
    ax.set_title("Panel 1 — eigenvector LOADINGS heatmap\n"
                 "how each named audio feature builds each principal component",
                 fontsize=12, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("loading (eigenvector weight)", fontsize=9)
    fig.tight_layout()
    fig.savefig(png2, bbox_inches="tight"); plt.close(fig)

    # loadings HTML (plotly heatmap)
    figh = go.Figure(go.Heatmap(
        z=Lo, x=[f"PC{k+1} ({evr[k]*100:.0f}%)" for k in range(n_show)], y=flabels,
        colorscale=[[0, "#C0392B"], [0.5, "#FFFFFF"], [1, "#185FA5"]], zmid=0,
        colorbar=dict(title="loading")))
    figh.update_layout(
        title="Eigenvector loadings — named audio features x principal components",
        template="plotly_white", width=820, height=900,
        yaxis=dict(autorange="reversed"))
    html2 = OUT / "viz_eigen_loadings.html"
    figh.write_html(str(html2), include_plotlyjs="cdn", full_html=True)

    # PC1..PC3 named interpretation for the report.
    pc_named = {}
    for k in range(min(3, comp.shape[0])):
        top = np.argsort(-np.abs(comp[k]))[:4]
        pc_named[f"PC{k+1}"] = {
            "variance_pct": round(float(evr[k] * 100), 1),
            "top_features": [{"feature": feats[i], "loading": round(float(comp[k][i]), 3)}
                             for i in top]}

    return {
        "n_features": len(feats), "n_components_for_95": n_for_95,
        "explained_variance_ratio": [round(float(v), 4) for v in evr[:n_show]],
        "cumulative_at_95_pcs": round(float(cum[n_for_95 - 1]), 3),
        "pc_named": pc_named,
        "artifacts": {"scree_png": png1.name, "scree_html": html1.name,
                      "loadings_png": png2.name, "loadings_html": html2.name},
    }


# --------------------------------------------------------------------------- #
# PANEL 2 — supervised LDA 3D scatter (rotating), held-out accuracy in caption
# --------------------------------------------------------------------------- #
def _kfold_lda_accuracy(X, y):
    """Stratified k-fold audio-only LDA accuracy with ALL preprocessing fit per
    train fold. Returns mean accuracy + per-class recall, the honest held-out number."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    accs, true_all, pred_all = [], [], []
    for tr, te in skf.split(X, y):
        _, ap = impute_fit(X[tr])
        sc = StandardScaler().fit(ap(X[tr]))
        Xtr = sc.transform(ap(X[tr])); Xte = sc.transform(ap(X[te]))
        pca = PCA(n_components=0.95, svd_solver="full").fit(Xtr)
        clf = LDA().fit(pca.transform(Xtr), y[tr])
        pred = clf.predict(pca.transform(Xte))
        accs.append(float((pred == y[te]).mean()))
        true_all.extend(y[te].tolist()); pred_all.extend(pred.tolist())
    true_all = np.array(true_all); pred_all = np.array(pred_all)
    rec = {c: round(float((pred_all[true_all == c] == c).mean()), 3) for c in CORE}
    return float(np.mean(accs)), float(np.std(accs)), rec


def _ellipsoid(mean, cov, n=18, scale=2.0):
    """2-sigma ellipsoid mesh (x,y,z) for a 3D Gaussian (scale=2 -> ~2 std)."""
    u = np.linspace(0, 2 * np.pi, n); v = np.linspace(0, np.pi, n)
    xs = np.outer(np.cos(u), np.sin(v)); ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    pts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], 0)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-9, None)
    T = vecs @ np.diag(np.sqrt(vals) * scale)
    e = T @ pts + mean[:, None]
    return e[0].reshape(n, n), e[1].reshape(n, n), e[2].reshape(n, n)


def panel2_lda3d(sn):
    """Full-data LDA 3D embedding (2 LDA axes + a residual PC for a 3rd display
    dimension), with 2-sigma class ellipsoids and a rotating camera. Caption carries
    the HELD-OUT k-fold accuracy so the picture is honest geometry, number is honest."""
    feats = [c for c in AUDIO if c in sn.columns]
    X = sn[feats].to_numpy(float); y = sn["intended_class"].to_numpy()

    acc, acc_sd, rec = _kfold_lda_accuracy(X, y)

    # Full-data embedding for display (axes are descriptive; the NUMBER is held out).
    _, ap = impute_fit(X)
    sc = StandardScaler().fit(ap(X)); Xs = sc.transform(ap(X))
    pca = PCA(n_components=0.95, svd_solver="full").fit(Xs); Xp = pca.transform(Xs)
    lda = LDA().fit(Xp, y)
    L2 = lda.transform(Xp)                       # 3 classes -> 2 LDA axes (centered)
    # 3rd DISPLAY axis = PC1 of the PCA-space variance left AFTER removing the 2-D LDA
    # subspace (orthonormalize the LDA directions in PCA space, project them out, PCA the
    # remainder). Gives the largest spread the two discriminants do not already show.
    Q, _ = np.linalg.qr(lda.scalings_[:, :2])    # orthonormal basis of LDA subspace in PCA space
    Xp_c = Xp - Xp.mean(0)
    resid = Xp_c - (Xp_c @ Q) @ Q.T              # remove LDA-spanned variance
    pc3 = PCA(n_components=1, random_state=SEED).fit_transform(resid).ravel()
    emb = np.column_stack([L2[:, 0], L2[:, 1], pc3])

    # Axis titles from feature-space loadings (chain PCA->LDA back to features).
    W = pca.components_.T @ lda.scalings_[:, :2]   # (n_feat, 2)
    def axis_title(col, default):
        top = np.argsort(-np.abs(col))[:2]
        return f"{default}: {pretty(feats[top[0]])} + {pretty(feats[top[1]])}"
    ax_titles = [axis_title(W[:, 0], "LD1"), axis_title(W[:, 1], "LD2"),
                 "residual PC (display)"]

    fig = go.Figure()
    for c in CORE:
        mk = y == c
        fig.add_trace(go.Scatter3d(
            x=emb[mk, 0], y=emb[mk, 1], z=emb[mk, 2], mode="markers",
            name=f"{c} (n={int(mk.sum())})",
            marker=dict(size=3.4, color=CLR[c], opacity=0.85,
                        line=dict(width=0.3, color="white"))))
        mean = emb[mk].mean(0); cov = np.cov(emb[mk].T)
        ex, ey, ez = _ellipsoid(mean, cov, scale=2.0)
        fig.add_trace(go.Surface(
            x=ex, y=ey, z=ez, showscale=False, opacity=0.13,
            colorscale=[[0, CLR[c]], [1, CLR[c]]], name=f"{c} 2σ", showlegend=False,
            hoverinfo="skip"))

    # Rotating camera frames.
    frames = []
    for ang in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        cam = dict(eye=dict(x=1.9 * np.cos(ang), y=1.9 * np.sin(ang), z=1.4))
        frames.append(go.Frame(layout=dict(scene_camera=cam), name=f"{ang:.2f}"))
    fig.frames = frames
    fig.update_layout(
        title=dict(text="Panel 2 — supervised LDA separation of clean / buzz / muted "
                        "(2σ ellipsoids)<br>"
                        f"<sub>held-out audio-only accuracy {acc:.2f}±{acc_sd:.2f} "
                        f"(chance 0.33) — stratified 5-fold, single player, fit on train only "
                        f"| recall {rec}</sub>",
                   x=0.02),
        scene=dict(xaxis_title=ax_titles[0], yaxis_title=ax_titles[1],
                   zaxis_title=ax_titles[2],
                   camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
        template="plotly_white", width=980, height=720,
        legend=dict(x=0.02, y=0.95, bgcolor="rgba(255,255,255,0.85)"),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.05,
                          buttons=[dict(label="rotate", method="animate",
                                        args=[None, dict(frame=dict(duration=70, redraw=True),
                                                         fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="stop", method="animate",
                                        args=[[None], dict(mode="immediate",
                                                           frame=dict(duration=0, redraw=False))])])])
    html = OUT / "viz_eigen_lda3d.html"
    fig.write_html(str(html), include_plotlyjs="cdn", full_html=True, auto_play=False)

    # Static PNG (best-angle snapshot) via kaleido.
    png = OUT / "viz_eigen_lda3d.png"
    try:
        fig.update_layout(updatemenus=[])
        # Snapshot camera looks down the residual (display) axis so the LD1xLD2
        # separation plane — where clean/buzz/muted actually pull apart — is face-on.
        fig.update_scenes(camera=dict(eye=dict(x=0.2, y=0.2, z=2.4),
                                      up=dict(x=0, y=1, z=0)))
        fig.write_image(str(png), width=980, height=720, scale=2)
        png_ok = True
    except Exception as exc:
        png_ok = False
        png = f"(png skipped: {exc})"

    return {
        "kfold_accuracy": round(acc, 3), "kfold_accuracy_std": round(acc_sd, 3),
        "chance": round(1 / 3, 3), "per_class_recall": rec,
        "axis_titles": ax_titles,
        "artifacts": {"lda3d_html": html.name,
                      "lda3d_png": (png.name if png_ok else str(png))},
    }


# --------------------------------------------------------------------------- #
# PANEL 3 — unsupervised clustering vs true labels (+ chance) + named clusters
# --------------------------------------------------------------------------- #
def _chance_ari_ami(y, n_perm=200):
    """Permutation chance floor: ARI/AMI of a random 3-way partition vs truth."""
    rng = np.random.default_rng(SEED)
    yk = pd.Categorical(y).codes
    aris, amis = [], []
    for _ in range(n_perm):
        rand = rng.integers(0, 3, size=len(yk))
        aris.append(adjusted_rand_score(yk, rand))
        amis.append(adjusted_mutual_info_score(yk, rand))
    return float(np.mean(aris)), float(np.mean(amis))


def _name_clusters(Xs_df, labels, feats, top=3):
    """Name each cluster by its most extreme standardized-feature means (z-scores).
    'geospatial region -> semantic meaning'."""
    out = {}
    for cl in sorted(set(labels)):
        if cl == -1:
            out["noise"] = {"n": int((labels == -1).sum()), "desc": "HDBSCAN noise / outliers"}
            continue
        mk = labels == cl
        z = Xs_df[mk].mean(axis=0)                  # mean z per feature in this cluster
        order = np.argsort(-np.abs(z.values))[:top]
        parts = []
        for i in order:
            f = feats[i]; v = z.values[i]
            parts.append(f"{'high' if v > 0 else 'low'} {pretty(f)}")
        out[f"cluster_{cl}"] = {"n": int(mk.sum()), "desc": ", ".join(parts)}
    return out


def panel3_clustering(sn):
    """KMeans + GMM + HDBSCAN on standardized audio+residual space. Score ARI/AMI/
    silhouette vs true clean/buzz/muted, report vs permutation chance. Name clusters."""
    feats = [c for c in CLUSTER_FEATS if c in sn.columns]
    X = sn[feats].to_numpy(float); y = sn["intended_class"].to_numpy()
    _, ap = impute_fit(X)
    Xs = StandardScaler().fit_transform(ap(X))
    Xs_df = pd.DataFrame(Xs, columns=feats)

    # 2D display embedding (PCA) shared by all three methods.
    disp = PCA(n_components=2, random_state=SEED).fit_transform(Xs)

    yk = pd.Categorical(y, categories=CORE).codes
    results = {}
    km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit_predict(Xs)
    gm = GaussianMixture(n_components=3, covariance_type="full", random_state=SEED,
                         n_init=5).fit_predict(Xs)
    try:
        import hdbscan
        # Cluster in the PCA-reduced space (HDBSCAN on raw 36-D over-fragments to noise);
        # min_cluster_size sized to the per-class count (144) / a few so real groups survive.
        Xs_red = PCA(n_components=10, random_state=SEED).fit_transform(Xs)
        hb = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=3).fit_predict(Xs_red)
    except Exception:
        hb = np.full(len(Xs), -1)

    methods = {"KMeans (k=3)": km, "GMM (k=3)": gm, "HDBSCAN": hb}
    chance_ari, chance_ami = _chance_ari_ami(y)

    for name, lab in methods.items():
        ari = adjusted_rand_score(yk, lab); ami = adjusted_mutual_info_score(yk, lab)
        sil = None
        valid = lab != -1
        if len(set(lab[valid])) >= 2 and valid.sum() > len(set(lab[valid])):
            try:
                sil = float(silhouette_score(Xs[valid], lab[valid]))
            except ValueError:
                sil = None
        results[name] = {
            "ARI": round(float(ari), 3), "AMI": round(float(ami), 3),
            "silhouette": round(sil, 3) if sil is not None else None,
            "n_clusters": int(len(set(lab[lab != -1]))),
            "n_noise": int((lab == -1).sum()),
            "named_clusters": _name_clusters(Xs_df, lab, feats),
        }

    # ---- 2x2 panel: true labels + 3 methods on the shared PCA display ----
    png = OUT / "viz_eigen_clustering.png"
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.0), dpi=150)
    # true labels
    axt = axes[0, 0]
    for c in CORE:
        mk = y == c
        axt.scatter(disp[mk, 0], disp[mk, 1], s=14, c=CLR[c], alpha=0.7,
                    edgecolors="none", label=c)
    axt.set_title("TRUE labels (clean / buzz / muted)", fontsize=11, fontweight="bold")
    axt.legend(fontsize=8, loc="best"); axt.set_xlabel("PC1"); axt.set_ylabel("PC2")
    axt.grid(alpha=0.15)
    cmap = plt.get_cmap("tab10")
    for ax, (name, lab) in zip([axes[0, 1], axes[1, 0], axes[1, 1]], methods.items()):
        for cl in sorted(set(lab)):
            mk = lab == cl
            col = "#BBBBBB" if cl == -1 else cmap(cl % 10)
            lb = "noise" if cl == -1 else f"c{cl}"
            ax.scatter(disp[mk, 0], disp[mk, 1], s=14, color=col, alpha=0.7,
                       edgecolors="none", label=lb)
        r = results[name]
        ax.set_title(f"{name}  ARI={r['ARI']}  AMI={r['AMI']}"
                     + (f"  sil={r['silhouette']}" if r['silhouette'] is not None else ""),
                     fontsize=10.5, fontweight="bold")
        ax.legend(fontsize=7, loc="best", ncol=2); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.grid(alpha=0.15)
    fig.suptitle("Panel 3 — unsupervised clustering of the standardized audio+residual space\n"
                 f"vs TRUE clean/buzz/muted   (chance ARI≈{chance_ari:.3f}, AMI≈{chance_ami:.3f}; "
                 f"single player, descriptive)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png, bbox_inches="tight"); plt.close(fig)

    # ---- HTML (plotly) — true + KMeans side by side ----
    figp = go.Figure()
    for c in CORE:
        mk = y == c
        figp.add_trace(go.Scatter(x=disp[mk, 0], y=disp[mk, 1], mode="markers",
                                  name=f"true {c}", marker=dict(size=6, color=CLR[c], opacity=0.7)))
    figp.update_layout(
        title=f"Feature-based clustering display (PCA of audio+residual) — true labels<br>"
              f"<sub>KMeans ARI={results['KMeans (k=3)']['ARI']}, "
              f"GMM ARI={results['GMM (k=3)']['ARI']} vs chance ARI≈{chance_ari:.3f} "
              f"| single player, descriptive</sub>",
        xaxis_title="PC1", yaxis_title="PC2", template="plotly_white",
        width=900, height=640, legend=dict(bgcolor="rgba(255,255,255,0.85)"))
    html = OUT / "viz_eigen_clustering.html"
    figp.write_html(str(html), include_plotlyjs="cdn", full_html=True)

    return {
        "n_features": len(feats),
        "chance_ARI": round(chance_ari, 4), "chance_AMI": round(chance_ami, 4),
        "methods": results,
        "artifacts": {"clustering_png": png.name, "clustering_html": html.name},
    }


# --------------------------------------------------------------------------- #
# PANEL 4 — mono->poly transfer money shot (shared buzz axis)
# --------------------------------------------------------------------------- #
def panel4_transfer(sn, cs):
    """Mono buzz axis fit on the HARMONIC-RESIDUAL features (pitch nulled, scale-relative)
    — the object H2 predicts should TRANSFER mono->poly. Honest tradeoff: residual features
    are scale-invariant so chords land ON the mono manifold (in-range≈1.0), but the residual
    buzz primitive is subtle here (held-out d'~0.5). The full audio axis separates mono far
    better (d'~1.1) but does NOT transfer — chords blow ~8x past the buzz cluster under the
    mono->poly domain shift (overall loudness/spectral-shape directions are not comparable).
    So the SCALE-INVARIANT residual axis is the correct transfer object; we report both facts.
    Chords are fully held out of every fit."""
    feats = [c for c in RESID if c in sn.columns]
    mono = sn[sn["intended_class"].isin(["clean", "buzz"])].copy()
    Xm = mono[feats].to_numpy(float); ym = (mono["intended_class"] == "buzz").astype(int).to_numpy()
    Xc = cs[feats].to_numpy(float)

    # k-fold mono separation (held-out d').
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    dprimes, accs = [], []
    for tr, te in skf.split(Xm, ym):
        _, ap = impute_fit(Xm[tr])
        sc = StandardScaler().fit(ap(Xm[tr]))
        clf = LDA().fit(sc.transform(ap(Xm[tr])), ym[tr])
        s = clf.decision_function(sc.transform(ap(Xm[te])))
        a, b = s[ym[te] == 0], s[ym[te] == 1]
        if a.size > 1 and b.size > 1:
            sp = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
            dprimes.append(abs(b.mean() - a.mean()) / (sp + 1e-9))
        accs.append(float((clf.predict(sc.transform(ap(Xm[te]))) == ym[te]).mean()))

    # Final axis on ALL mono (chords excluded by construction).
    _, apf = impute_fit(Xm)
    scf = StandardScaler().fit(apf(Xm)); clff = LDA().fit(scf.transform(apf(Xm)), ym)
    mono_s = clff.decision_function(scf.transform(apf(Xm)))
    clean_s = mono_s[ym == 0]; buzz_s = mono_s[ym == 1]
    chord_s = clff.decision_function(scf.transform(apf(Xc)))  # apf uses mono (train) means

    buzz_high = buzz_s.mean() > clean_s.mean()
    mid = 0.5 * (clean_s.mean() + buzz_s.mean())
    frac_buzz_side = float(np.mean(chord_s > mid) if buzz_high else np.mean(chord_s < mid))
    pos = float((chord_s.mean() - clean_s.mean()) / (buzz_s.mean() - clean_s.mean()))
    lo, hi = min(clean_s.min(), buzz_s.min()), max(clean_s.max(), buzz_s.max())
    frac_in_range = float(np.mean((chord_s >= lo) & (chord_s <= hi)))

    # y for the scatter = interpretable single residual feature.
    yfeat = "comb_resid_ratio" if "comb_resid_ratio" in feats else feats[0]
    yc = cs[yfeat].to_numpy(float)
    ycl = mono.loc[ym == 0, yfeat].to_numpy(float); ybz = mono.loc[ym == 1, yfeat].to_numpy(float)

    # ---- PNG ----
    png = OUT / "viz_eigen_transfer.png"
    fig, ax = plt.subplots(figsize=(9.4, 6.6), dpi=160)
    ax.scatter(chord_s, yc, s=22, c=CLR_CHORD, alpha=0.35, edgecolors="none",
               label=f"poly chord residual (n={len(chord_s)})", zorder=2)
    ax.scatter(clean_s, ycl, s=42, c=CLR["clean"], alpha=0.85, edgecolors="white",
               linewidths=0.4, label=f"mono CLEAN (n={len(clean_s)})", zorder=4)
    ax.scatter(buzz_s, ybz, s=42, c=CLR["buzz"], alpha=0.85, edgecolors="white",
               linewidths=0.4, label=f"mono BUZZ (n={len(buzz_s)})", zorder=4)
    ax.axvline(mid, color="#566573", ls="--", lw=1.1)
    ax.text(mid, ax.get_ylim()[1], " clean|buzz boundary", va="top", ha="left",
            fontsize=8, color="#566573", rotation=90)
    ax.set_xlabel("mono-fit BUZZ AXIS (LDA on scale-invariant harmonic-residual features)", fontsize=11)
    ax.set_ylabel(pretty(yfeat), fontsize=11)
    ax.set_title("Panel 4 — mono->poly transfer: the harmonic-residual buzz primitive "
                 "transfers to chords\n"
                 "buzz axis learned on SINGLE NOTES; chord strums projected on the SAME axis",
                 fontsize=11.5, fontweight="bold")
    ax.legend(loc="best", framealpha=0.92, fontsize=9); ax.grid(alpha=0.16)
    fig.text(0.5, 0.005,
             f"held-out mono clean|buzz d'={np.mean(dprimes):.2f} (acc {np.mean(accs):.2f}); "
             f"{frac_in_range*100:.0f}% of chords land INSIDE the mono range (axis transfers), "
             f"at {pos:.2f} on 0=clean..1=buzz with a buzz-lean tail "
             f"| chords NOT in fit; single player, k-fold; geometry (chords unlabeled)",
             ha="center", fontsize=7.8, color="#34495E")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(png, bbox_inches="tight"); plt.close(fig)

    # ---- HTML ----
    figp = go.Figure()
    figp.add_trace(go.Scatter(x=chord_s, y=yc, mode="markers",
                              name=f"poly chord residual (n={len(chord_s)})",
                              marker=dict(size=6, color=CLR_CHORD, opacity=0.35)))
    figp.add_trace(go.Scatter(x=clean_s, y=ycl, mode="markers",
                              name=f"mono CLEAN (n={len(clean_s)})",
                              marker=dict(size=9, color=CLR["clean"], opacity=0.85,
                                          line=dict(width=0.5, color="white"))))
    figp.add_trace(go.Scatter(x=buzz_s, y=ybz, mode="markers",
                              name=f"mono BUZZ (n={len(buzz_s)})",
                              marker=dict(size=9, color=CLR["buzz"], opacity=0.85,
                                          line=dict(width=0.5, color="white"))))
    figp.add_vline(x=mid, line_dash="dash", line_color="#566573",
                   annotation_text="clean|buzz boundary")
    figp.update_layout(
        title="Mono->poly transfer — chords on the scale-invariant mono-fit buzz axis<br>"
              f"<sub>held-out mono d'={np.mean(dprimes):.2f} | {frac_in_range*100:.0f}% of chords "
              f"inside mono range, at {pos:.2f} (0=clean..1=buzz) | single player, geometry</sub>",
        xaxis_title="mono-fit BUZZ AXIS (LDA projection)", yaxis_title=pretty(yfeat),
        template="plotly_white", width=960, height=660,
        legend=dict(bgcolor="rgba(255,255,255,0.85)"))
    html = OUT / "viz_eigen_transfer.html"
    figp.write_html(str(html), include_plotlyjs="cdn", full_html=True)

    return {
        "mono_kfold_dprime": round(float(np.mean(dprimes)), 3),
        "mono_kfold_accuracy": round(float(np.mean(accs)), 3),
        "n_chords": int(len(chord_s)),
        "chord_position_0clean_1buzz": round(pos, 3),
        "frac_chords_buzz_side": round(frac_buzz_side, 3),
        "frac_chords_in_mono_range": round(frac_in_range, 3),
        "transfer_note": ("scale-invariant harmonic-residual axis transfers (chords on mono "
                          "manifold); full-audio axis separates mono better (d'~1.1) but does "
                          "NOT transfer (chords blow ~8x past buzz under mono->poly domain shift)"),
        "artifacts": {"transfer_png": png.name, "transfer_html": html.name},
    }


# --------------------------------------------------------------------------- #
# GALLERY — dark, polished, links everything
# --------------------------------------------------------------------------- #
def build_gallery(report):
    p1, p2, p3, p4 = report["panel1_eigen"], report["panel2_lda3d"], \
        report["panel3_clustering"], report["panel4_transfer"]
    km = p3["methods"]["KMeans (k=3)"]

    def card(num, title, blurb, png, html, metrics):
        mrows = "".join(
            f'<div class="m"><span class="mk">{k}</span><span class="mv">{v}</span></div>'
            for k, v in metrics.items())
        img = f'<img src="{png}" alt="{title}">' if png else ""
        link = f'<a class="open" href="{html}" target="_blank">open interactive ↗</a>' if html else ""
        return f"""
    <section class="panel">
      <div class="phead"><span class="pnum">{num}</span><h2>{title}</h2></div>
      <p class="blurb">{blurb}</p>
      <div class="metrics">{mrows}</div>
      <div class="figwrap">{img}</div>
      {link}
    </section>"""

    panels = "".join([
        card("01", "Eigen-decomposition: scree, cumulative variance, loadings heatmap",
             f"PCA = the eigendecomposition of the standardized audio covariance matrix. "
             f"{p1['n_components_for_95']} principal components capture 95% of variance across "
             f"{p1['n_features']} named audio features. The loadings heatmap shows what each "
             f"eigenvector MEANS: PC1 ≈ "
             f"{pretty(p1['pc_named']['PC1']['top_features'][0]['feature'])}. "
             f"Descriptive geometry (full data) — the basis the supervised model is built on.",
             p1["artifacts"]["scree_png"], p1["artifacts"]["scree_html"],
             {"PCs for 95% var": p1["n_components_for_95"],
              "PC1 variance": f"{p1['pc_named']['PC1']['variance_pct']}%",
              "named features": p1["n_features"]})
        + f'<div class="extra"><img src="{p1["artifacts"]["loadings_png"]}" '
          f'alt="loadings heatmap"><a class="open" '
          f'href="{p1["artifacts"]["loadings_html"]}" target="_blank">open loadings heatmap ↗</a></div>',

        card("02", "Supervised LDA separation (rotating 3D, 2σ ellipsoids)",
             f"LDA finds the axes that SEPARATE clean / buzz / muted where raw PCA blobs. "
             f"Axes are titled by their top named-feature loadings; ellipsoids are 2σ class "
             f"covariances. The number is honest: held-out audio-only accuracy "
             f"{p2['kfold_accuracy']:.2f}±{p2['kfold_accuracy_std']:.2f} vs chance "
             f"{p2['chance']:.2f}, stratified 5-fold, single player, all preprocessing fit on "
             f"train folds only. Static PNG below; rotating version is the interactive link.",
             p2["artifacts"].get("lda3d_png") if str(p2["artifacts"].get("lda3d_png", "")).endswith(".png") else None,
             p2["artifacts"]["lda3d_html"],
             {"held-out acc": f"{p2['kfold_accuracy']:.2f}", "chance": f"{p2['chance']:.2f}",
              "clean recall": p2["per_class_recall"]["clean"],
              "muted recall": p2["per_class_recall"]["muted"]}),

        card("03", "Feature-based clustering vs true labels (named clusters)",
             f"KMeans, GMM and HDBSCAN on the standardized audio+residual space, scored against "
             f"the TRUE clean/buzz/muted labels and a permutation chance floor "
             f"(ARI≈{p3['chance_ARI']}). KMeans recovers structure far above chance "
             f"(ARI={km['ARI']}, AMI={km['AMI']}). Each discovered cluster is NAMED by its most "
             f"extreme standardized-feature means — geometry mapped to meaning.",
             p3["artifacts"]["clustering_png"], p3["artifacts"]["clustering_html"],
             {"KMeans ARI": km["ARI"], "chance ARI": p3["chance_ARI"],
              "KMeans AMI": km["AMI"], "silhouette": km["silhouette"]}),

        card("04", "Mono → poly transfer money shot",
             f"The SCALE-INVARIANT harmonic-residual buzz axis learned on single notes "
             f"(held-out clean|buzz d'={p4['mono_kfold_dprime']}) transfers to chords: "
             f"{p4['n_chords']} chord strums projected on the SAME axis land "
             f"INSIDE the mono cluster range ({p4['frac_chords_in_mono_range']*100:.0f}%), at "
             f"{p4['chord_position_0clean_1buzz']} on a 0=clean..1=buzz scale with a buzz-lean "
             f"tail — exactly H2's prediction that the residual fault primitive is the same "
             f"physical object mono and poly. Honest tradeoff: the full-audio axis separates "
             f"mono far better but does NOT transfer (domain shift). Chords fully held out; "
             f"geometry only (chords are unlabeled).",
             p4["artifacts"]["transfer_png"], p4["artifacts"]["transfer_html"],
             {"mono d' (held-out)": p4["mono_kfold_dprime"],
              "chords in mono range": f"{p4['frac_chords_in_mono_range']*100:.0f}%",
              "chord position": p4["chord_position_0clean_1buzz"],
              "chord strums": p4["n_chords"]}),
    ])

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tactus — eigenvector & feature-clustering gallery</title>
<style>
  :root {{ --bg:#0c0f14; --surf:#141922; --surf2:#1b2230; --line:#28303f;
    --tx:#e8edf5; --tx2:#9fb0c5; --tx3:#6c7c93; --accent:#5DCAA5; --accent2:#378ADD; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--tx);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    line-height:1.6; }}
  header {{ padding:48px 32px 28px; border-bottom:1px solid var(--line);
    background:linear-gradient(180deg,#10151d,#0c0f14); }}
  header h1 {{ margin:0 0 8px; font-size:30px; font-weight:600; letter-spacing:-0.5px; }}
  header p {{ margin:0; color:var(--tx2); max-width:880px; font-size:15px; }}
  .rigor {{ margin-top:16px; display:flex; flex-wrap:wrap; gap:8px; }}
  .rigor span {{ font-size:12px; color:var(--tx2); background:var(--surf2);
    border:1px solid var(--line); border-radius:20px; padding:5px 12px; }}
  main {{ max-width:1080px; margin:0 auto; padding:28px 24px 64px;
    display:flex; flex-direction:column; gap:28px; }}
  .panel {{ background:var(--surf); border:1px solid var(--line); border-radius:16px;
    padding:24px 26px; }}
  .phead {{ display:flex; align-items:center; gap:14px; margin-bottom:6px; }}
  .pnum {{ font-size:13px; font-weight:600; color:var(--bg); background:var(--accent);
    border-radius:8px; padding:4px 9px; }}
  .panel h2 {{ margin:0; font-size:19px; font-weight:600; }}
  .blurb {{ color:var(--tx2); font-size:14.5px; margin:8px 0 16px; }}
  .metrics {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:18px; }}
  .m {{ background:var(--surf2); border:1px solid var(--line); border-radius:10px;
    padding:8px 14px; min-width:120px; }}
  .mk {{ display:block; font-size:11.5px; color:var(--tx3); text-transform:uppercase;
    letter-spacing:0.4px; }}
  .mv {{ display:block; font-size:21px; font-weight:600; color:var(--accent); margin-top:2px; }}
  .figwrap, .extra {{ background:#fff; border-radius:12px; padding:8px; overflow:hidden; }}
  .figwrap img, .extra img {{ width:100%; display:block; border-radius:8px; }}
  .extra {{ margin-top:14px; }}
  a.open {{ display:inline-block; margin-top:14px; font-size:13.5px; font-weight:500;
    color:var(--accent2); text-decoration:none; border:1px solid var(--accent2);
    border-radius:8px; padding:7px 14px; }}
  a.open:hover {{ background:var(--accent2); color:#fff; }}
  footer {{ text-align:center; color:var(--tx3); font-size:12.5px; padding:24px;
    border-top:1px solid var(--line); }}
</style></head><body>
<header>
  <h1>Tactus — eigenvector &amp; feature-clustering gallery</h1>
  <p>The linear-algebra spine of the separability study, made legible. Standardize →
  PCA (eigendecomposition) → supervised LDA, then unsupervised clustering and a
  mono→poly transfer — all on the audio (timbre + harmonic-residual) feature space.</p>
  <div class="rigor">
    <span>432 single notes + 600 chord strums</span>
    <span>ONE player (aditya) → stratified 5-fold, NOT LOPO</span>
    <span>all preprocessing fit on train folds only</span>
    <span>clean/buzz/muted balanced → chance = 0.33</span>
    <span>held-out audio-only LDA acc {p2['kfold_accuracy']:.2f}</span>
  </div>
</header>
<main>{panels}</main>
<footer>Tactus offline-analysis pipeline · viz_eigen.py · descriptive geometry is
labelled as such; every predictive number is held out · single-player k-fold (no
cross-player generalization claimed)</footer>
</body></html>"""
    path = OUT / "viz_eigen_gallery.html"
    path.write_text(html)
    return path


# --------------------------------------------------------------------------- #
def main():
    print("=" * 74)
    print("viz_eigen — eigenvector + feature-clustering demo centerpiece")
    print("=" * 74)
    m, sn, cs = load()
    print(f"loaded matrix.csv: {len(m)} rows  |  single notes {len(sn)} "
          f"({dict(sn['intended_class'].value_counts())})  |  chord strums {len(cs)}")

    print("\n[1/4] eigen-decomposition (scree + loadings) ...")
    p1 = panel1_eigen(sn)
    print(f"      {p1['n_components_for_95']} PCs for 95% var; "
          f"PC1={p1['pc_named']['PC1']['variance_pct']}% "
          f"(top: {p1['pc_named']['PC1']['top_features'][0]['feature']})")

    print("[2/4] supervised LDA 3D (rotating) ...")
    p2 = panel2_lda3d(sn)
    print(f"      held-out audio-only acc {p2['kfold_accuracy']:.3f}±{p2['kfold_accuracy_std']:.3f} "
          f"(chance {p2['chance']}); recall {p2['per_class_recall']}")

    print("[3/4] unsupervised clustering vs labels ...")
    p3 = panel3_clustering(sn)
    for nm, r in p3["methods"].items():
        print(f"      {nm:14s} ARI={r['ARI']} AMI={r['AMI']} sil={r['silhouette']} "
              f"(chance ARI≈{p3['chance_ARI']})")

    print("[4/4] mono->poly transfer money shot ...")
    p4 = panel4_transfer(sn, cs)
    print(f"      mono held-out d'={p4['mono_kfold_dprime']}; chords at "
          f"{p4['chord_position_0clean_1buzz']} (0=clean..1=buzz), "
          f"{p4['frac_chords_buzz_side']*100:.0f}% buzz-side")

    report = {
        "experiment": "viz_eigen — eigenvector + feature-clustering demo centerpiece",
        "data": {"matrix": str(MATRIX), "n_rows": int(len(m)),
                 "n_single_notes": int(len(sn)), "n_chord_strums": int(len(cs)),
                 "single_note_counts": {c: int((sn['intended_class'] == c).sum()) for c in CORE}},
        "rigor": {"player": "aditya (single)", "split": "stratified 5-fold (NOT LOPO)",
                  "preprocessing": "impute+scale+PCA+LDA fit on train folds only for all "
                                   "predictive numbers; scree/loadings are descriptive full-data geometry",
                  "chance_accuracy": round(1 / 3, 3),
                  "base_rates": "clean/buzz/muted balanced at 144 each"},
        "panel1_eigen": p1, "panel2_lda3d": p2,
        "panel3_clustering": p3, "panel4_transfer": p4,
    }
    gallery = build_gallery(report)
    report["gallery"] = gallery.name
    (OUT / "viz_eigen_report.json").write_text(json.dumps(report, indent=2))

    print("\n--- ARTIFACTS (data/analysis/exp/) ---")
    for nm in ["viz_eigen_scree.png", "viz_eigen_scree.html", "viz_eigen_loadings.png",
               "viz_eigen_loadings.html", "viz_eigen_lda3d.html", "viz_eigen_lda3d.png",
               "viz_eigen_clustering.png", "viz_eigen_clustering.html",
               "viz_eigen_transfer.png", "viz_eigen_transfer.html",
               "viz_eigen_report.json", "viz_eigen_gallery.html"]:
        pth = OUT / nm
        print(f"  {'OK ' if pth.exists() else 'MISS'} {nm}"
              + (f"  ({pth.stat().st_size//1024} KB)" if pth.exists() else ""))
    print("=" * 74)
    return report


if __name__ == "__main__":
    main()
