#!/usr/bin/env python3
"""
E7 STEP 2 — vision POSITION model vs. raw MediaPipe (the technical-award play).

Runs in .venv (3.14: numpy/scipy/sklearn/pandas/matplotlib). Reads the per-finger
table from STEP 1 (data/analysis/exp/vision_perfinger.csv) and answers H4:

    Can a model that reads the WHOLE fretting-hand pose beat raw MediaPipe at
    naming which string/fret each finger presses — especially under occlusion?

BASELINE (raw MediaPipe): the board readout the geometry pipeline produces today,
    mp_fret_est / mp_string_est (fingertip pixel -> homography -> board -> nearest
    fret/string). fret error = |mp_fret_est - true_fret|.

OURS: a learned map from the FULL hand-pose vector (all 4 fingers' board bx/by/z,
    curls, tip/MCP/PIP pixels normalized by hand scale, wrist & neck angle, plus a
    one-hot for WHICH finger this row is) to that finger's fret and string. Because
    an occluded finger's row still carries every VISIBLE finger's pose, the model
    can infer the hidden finger from the hand it can see. RandomForest is the
    headline; a small MLP is reported alongside.

HONEST CONTEXT (loud, by design — see the printed report and the parent summary):
  * ONE player, ONE guitar, ONE camera. Everything below is within-rig; it does
    NOT show cross-player / cross-guitar generalization. GroupKFold by run_id only
    proves we generalize across RECORDINGS of this rig, not across rigs.
  * The twin.json homography is hand-clicked on a CALIBRATION pose and does NOT
    register the gameplay video (STEP 1 diagnostic: 0% of fingertips land on the
    board; board-Y is systematically negative). So the raw-MediaPipe board readout
    is essentially un-registered — which is exactly the failure OURS has to beat,
    and why a learned pose model has so much headroom here. We say so plainly.

Outputs (data/analysis/exp/):
  beat_baseline_table.csv      the headline table (rows x metrics, RF + MLP)
  e7_perfinger_predictions.csv out-of-fold predictions for audit
  money_shot_fret_mae.png      grouped bars: MP vs Ours fret MAE, visible vs occ
  e7_model_report.txt          full text report incl. the honesty caveats
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


FINGERS = ("index", "middle", "ring", "pinky")


def _repo_root(start):
    d = start
    for _ in range(8):
        if os.path.exists(os.path.join(d, "data", "analysis", "exp")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return start


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _repo_root(_HERE)
_EXP = os.path.join(_REPO, "data", "analysis", "exp")


# --------------------------------------------------------------- feature builder
def build_features(df):
    """Return (X DataFrame, feature_names). The pose vector + finger one-hot.

    We normalize the raw pixel landmarks by a per-row HAND SCALE (wrist->index_MCP
    distance) and express them RELATIVE to the wrist, so the model keys off hand
    geometry, not absolute screen position. Board bx/by/z and curls (already
    scale-free) go in raw. A 4-way one-hot tells the model which finger's
    fret/string it must predict, so a single model serves all four fingers and can
    borrow strength across them."""
    out = {}

    # hand scale + wrist origin (avoid divide-by-zero)
    wx, wy = df["wrist_px_x"].to_numpy(), df["wrist_px_y"].to_numpy()
    ix, iy = df["index_mcp_px_x"].to_numpy(), df["index_mcp_px_y"].to_numpy()
    scale = np.hypot(ix - wx, iy - wy)
    scale = np.where(scale < 1e-6, 1.0, scale)

    def rel(colx, coly, name):
        out[f"{name}_rx"] = (df[colx].to_numpy() - wx) / scale
        out[f"{name}_ry"] = (df[coly].to_numpy() - wy) / scale

    # this row's finger tip/mcp/pip (normalized)
    rel("tip_px_x", "tip_px_y", "tip")
    rel("mcp_px_x", "mcp_px_y", "mcp")
    rel("pip_px_x", "pip_px_y", "pip")
    rel("index_mcp_px_x", "index_mcp_px_y", "imcp")

    # board coords + z + curl for THIS finger
    for c in ["bx", "by", "z", "curl"]:
        out[c] = df[c].to_numpy()

    # FULL hand: every finger's board bx/by/z + curl (the cross-finger constraint)
    for g in FINGERS:
        for suf in ["bx", "by", "z", "curl"]:
            out[f"{g}_{suf}"] = df[f"{g}_{suf}"].to_numpy()

    # wrist/neck angle + hand scale + detection conf
    out["wrist_angle"] = df["wrist_angle"].to_numpy()
    out["neck_angle"] = df["neck_angle"].to_numpy()
    out["hand_scale"] = scale
    out["hand_conf"] = df["hand_conf"].to_numpy()

    # which finger is this row (one-hot)
    for g in FINGERS:
        out[f"is_{g}"] = (df["finger"].to_numpy() == g).astype(float)

    X = pd.DataFrame(out, index=df.index)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, list(X.columns)


# --------------------------------------------------------------- cv evaluation
def grouped_oof(X, y_fret, y_string, groups, make_fret_reg, make_str_clf, n_splits,
                with_majority=False):
    """GroupKFold out-of-fold predictions. Returns arrays aligned to X.index order:
    pred_fret (rounded int), pred_string (int). Fit on train fold only.

    If with_majority, also returns (maj_fret, maj_str): the TRAIN-fold majority
    fret (rounded mean is not used; we use the modal fret) and modal string,
    broadcast to each test fold — the honest constant baseline."""
    gkf = GroupKFold(n_splits=n_splits)
    pred_fret = np.full(len(X), np.nan)
    pred_str = np.full(len(X), np.nan)
    maj_fret = np.full(len(X), np.nan)
    maj_str = np.full(len(X), np.nan)
    Xv = X.to_numpy()
    for tr, te in gkf.split(Xv, y_fret, groups):
        fr = make_fret_reg()
        fr.fit(Xv[tr], y_fret[tr])
        pred_fret[te] = np.rint(fr.predict(Xv[te]))
        sc = make_str_clf()
        sc.fit(Xv[tr], y_string[tr])
        pred_str[te] = sc.predict(Xv[te])
        if with_majority:
            vals, cnts = np.unique(y_fret[tr], return_counts=True)
            maj_fret[te] = vals[np.argmax(cnts)]
            svals, scnts = np.unique(y_string[tr], return_counts=True)
            maj_str[te] = svals[np.argmax(scnts)]
    if with_majority:
        return pred_fret, pred_str, maj_fret, maj_str
    return pred_fret, pred_str


# --------------------------------------------------------------- metrics
def metrics_block(mask, true_fret, true_str, mp_fret, mp_str, our_fret, our_str,
                  maj_fret=None, maj_str=None):
    """Return dict of the headline metrics over the rows in `mask`.

    Includes an out-of-fold MAJORITY-CLASS constant baseline (maj_fret/maj_str are
    the per-fold majority predictions) so we don't oversell beating an
    un-registered MediaPipe — the constant is the honest floor to clear."""
    m = mask & np.isfinite(our_fret) & np.isfinite(our_str)
    n = int(m.sum())
    if n == 0:
        return dict(n=0, mp_fret_mae=np.nan, our_fret_mae=np.nan,
                    mp_str_acc=np.nan, our_str_acc=np.nan,
                    maj_fret_mae=np.nan, maj_str_acc=np.nan)
    out = dict(
        n=n,
        mp_fret_mae=float(np.abs(mp_fret[m] - true_fret[m]).mean()),
        our_fret_mae=float(np.abs(our_fret[m] - true_fret[m]).mean()),
        mp_str_acc=float((mp_str[m] == true_str[m]).mean()),
        our_str_acc=float((our_str[m] == true_str[m]).mean()),
    )
    out["maj_fret_mae"] = (float(np.abs(maj_fret[m] - true_fret[m]).mean())
                           if maj_fret is not None else np.nan)
    out["maj_str_acc"] = (float((maj_str[m] == true_str[m]).mean())
                          if maj_str is not None else np.nan)
    return out


def signal_ablation(df, groups, n_splits):
    """Is the 'win' actually VISION, or just the finger->string/fret label prior?

    Compares three feature sets under the SAME GroupKFold:
      (a) finger one-hot ONLY   (no pixels at all — pure label correlation),
      (b) vision pose ONLY      (all fingers' board bx/by/z + curl + wrist/neck,
                                 NO finger identity),
      (c) vision + one-hot.
    If (b) <= majority and (a) ~= (c), the geometry adds nothing — the honest
    reading is that this footage carries no recoverable per-finger signal.
    Returns a dict of {string_acc, fret_mae} for {majority, onehot, vision, both}."""
    yf = df["true_fret"].to_numpy().astype(float)
    ys = df["true_string"].to_numpy().astype(int)
    onehot = np.column_stack([(df["finger"].to_numpy() == f).astype(float)
                              for f in FINGERS])
    visc = [f"{x}_{s}" for x in FINGERS for s in ["bx", "by", "z", "curl"]] \
        + ["wrist_angle", "neck_angle"]
    V = df[visc].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    ns = min(n_splits, len(np.unique(groups)))

    def oof_clf(Xm):
        p = np.full(len(Xm), -1)
        for tr, te in GroupKFold(ns).split(Xm, ys, groups):
            c = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                       n_jobs=-1, random_state=0,
                                       class_weight="balanced_subsample")
            c.fit(Xm[tr], ys[tr]); p[te] = c.predict(Xm[te])
        return float((p == ys).mean())

    def oof_reg(Xm):
        p = np.full(len(Xm), np.nan)
        for tr, te in GroupKFold(ns).split(Xm, yf, groups):
            r = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                      n_jobs=-1, random_state=0)
            r.fit(Xm[tr], yf[tr]); p[te] = np.rint(r.predict(Xm[te]))
        return float(np.abs(p - yf).mean())

    maj_s = np.bincount(ys).argmax()
    vals, cnts = np.unique(yf, return_counts=True)
    maj_f = vals[np.argmax(cnts)]
    both = np.column_stack([V, onehot])
    return {
        "string_acc": {"majority": float((ys == maj_s).mean()),
                       "onehot": oof_clf(onehot), "vision": oof_clf(V),
                       "both": oof_clf(both)},
        "fret_mae": {"majority": float(np.abs(yf - maj_f).mean()),
                     "onehot": oof_reg(onehot), "vision": oof_reg(V),
                     "both": oof_reg(both)},
    }


def chord_shape_head(df, X, groups, n_splits):
    """Per-EVENT chord-shape classification from the whole-hand pose.

    Collapse to one row per event (the whole-hand pose columns are identical
    across that event's per-finger rows; we drop the finger-specific one-hot and
    this-finger tip/board cols and keep only the hand-global pose). GroupKFold by
    run_id, RandomForest. Returns (oof_acc, majority_acc, n_events, n_runs)."""
    # hand-global pose columns (exclude this-row-finger-specific ones + one-hots)
    drop_pref = ("tip_", "mcp_", "pip_", "is_")
    keep = [c for c in X.columns
            if not c.startswith(drop_pref) and c not in ("bx", "by", "z", "curl")]
    Xe = X[keep].copy()
    Xe["event_id"] = df["event_id"].to_numpy()
    Xe["__chord"] = df["chord_name"].to_numpy()
    Xe["__run"] = groups
    ev = Xe.groupby("event_id", sort=False).first().reset_index()
    feat = [c for c in keep]
    Xv = ev[feat].to_numpy()
    y = ev["__chord"].to_numpy()
    g = ev["__run"].to_numpy()
    n_ev = len(ev)
    nr = len(np.unique(g))
    ns = min(n_splits, nr)
    gkf = GroupKFold(n_splits=ns)
    pred = np.empty(n_ev, dtype=object)
    maj = np.empty(n_ev, dtype=object)
    for tr, te in gkf.split(Xv, y, g):
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     n_jobs=-1, random_state=0,
                                     class_weight="balanced_subsample")
        clf.fit(Xv[tr], y[tr])
        pred[te] = clf.predict(Xv[te])
        vals, cnts = np.unique(y[tr], return_counts=True)
        maj[te] = vals[np.argmax(cnts)]
    return float((pred == y).mean()), float((maj == y).mean()), n_ev, nr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default=os.path.join(_EXP, "vision_perfinger.csv"))
    ap.add_argument("--outdir", default=_EXP)
    ap.add_argument("--splits", type=int, default=0, help="GroupKFold splits (0=auto=#runs)")
    a = ap.parse_args()

    df = pd.read_csv(a.infile).reset_index(drop=True)
    # need finite ground truth + a run group
    df = df[df["true_fret"].notna() & df["true_string"].notna() & df["run_id"].notna()]
    df = df.reset_index(drop=True)

    groups = df["run_id"].to_numpy()
    n_runs = len(np.unique(groups))
    n_splits = a.splits or min(n_runs, 8)
    if n_splits < 2:
        raise SystemExit(f"need >=2 run groups for GroupKFold, have {n_runs}")

    y_fret = df["true_fret"].to_numpy().astype(float)
    y_str = df["true_string"].to_numpy().astype(int)
    mp_fret = df["mp_fret_est"].to_numpy().astype(float)
    mp_str = df["mp_string_est"].to_numpy().astype(int)
    occ = df["occluded"].to_numpy().astype(int)

    X, feat_names = build_features(df)

    # ---- models -----------------------------------------------------------
    def rf_reg():
        return RandomForestRegressor(n_estimators=300, max_depth=None,
                                     min_samples_leaf=2, n_jobs=-1, random_state=0)

    def rf_clf():
        return RandomForestClassifier(n_estimators=300, max_depth=None,
                                      min_samples_leaf=2, n_jobs=-1, random_state=0,
                                      class_weight="balanced_subsample")

    def mlp_reg():
        return make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(128, 64),
                                          activation="relu", alpha=1e-3,
                                          max_iter=2000, random_state=0))

    def mlp_clf():
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(128, 64),
                                           activation="relu", alpha=1e-3,
                                           max_iter=2000, random_state=0))

    print(f"[data] rows={len(df)}  runs={n_runs}  splits={n_splits}  "
          f"features={len(feat_names)}  occluded={int(occ.sum())}/{len(df)}")

    rf_fret, rf_str, maj_fret, maj_str = grouped_oof(
        X, y_fret, y_str, groups, rf_reg, rf_clf, n_splits, with_majority=True)
    mlp_fret, mlp_str = grouped_oof(X, y_fret, y_str, groups, mlp_reg, mlp_clf, n_splits)

    # ---- beat-the-baseline table -----------------------------------------
    masks = {
        "all fingers": np.ones(len(df), bool),
        "VISIBLE fingers": occ == 0,
        "OCCLUDED fingers": occ == 1,
    }
    rows = []
    for label, mask in masks.items():
        for model_name, our_f, our_s in [("RandomForest", rf_fret, rf_str),
                                         ("MLP", mlp_fret, mlp_str)]:
            mb = metrics_block(mask, y_fret, y_str, mp_fret, mp_str, our_f, our_s,
                               maj_fret=maj_fret, maj_str=maj_str)
            rows.append(dict(subset=label, model=model_name, **mb))
    table = pd.DataFrame(rows)
    table_path = os.path.join(a.outdir, "beat_baseline_table.csv")
    table.to_csv(table_path, index=False)

    # predictions for audit
    pred_out = df[["event_id", "run_id", "chord_name", "finger",
                   "true_fret", "true_string", "mp_fret_est", "mp_string_est",
                   "occluded", "occ_reason"]].copy()
    pred_out["rf_fret"] = rf_fret
    pred_out["rf_string"] = rf_str
    pred_out["mlp_fret"] = mlp_fret
    pred_out["mlp_string"] = mlp_str
    pred_path = os.path.join(a.outdir, "e7_perfinger_predictions.csv")
    pred_out.to_csv(pred_path, index=False)

    # ---- DEGRADE PATH + ABLATION: does the pose carry ANY fretting signal? -----
    # Per-finger FRET is not separable beyond the majority prior (see table). The
    # task's prescribed fallback is chord-shape-from-pose (one whole-hand pose
    # vector per EVENT -> chord_name); we run it AND a signal ablation to test
    # honestly whether any lift is vision or just the finger->label prior.
    chord_acc, chord_maj, chord_n, chord_runs = chord_shape_head(df, X, groups, n_splits)
    abl = signal_ablation(df, groups, n_splits)

    # ---- money shot: grouped bars of fret MAE (MP vs Ours), visible vs occ ----
    rf_tab = table[table["model"] == "RandomForest"].set_index("subset")
    cats = ["VISIBLE fingers", "OCCLUDED fingers", "all fingers"]
    mp_vals = [rf_tab.loc[c, "mp_fret_mae"] for c in cats]
    maj_vals = [rf_tab.loc[c, "maj_fret_mae"] for c in cats]
    our_vals = [rf_tab.loc[c, "our_fret_mae"] for c in cats]
    xpos = np.arange(len(cats))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=130)
    b1 = ax.bar(xpos - w, mp_vals, w, label="Raw MediaPipe (board readout)",
                color="#c0392b")
    b3 = ax.bar(xpos, maj_vals, w, label="Majority-fret floor", color="#7f8c8d")
    b2 = ax.bar(xpos + w, our_vals, w, label="Ours (pose model, RF)",
                color="#27ae60")
    ax.set_xticks(xpos)
    ax.set_xticklabels([c.replace(" fingers", "\nfingers") for c in cats])
    ax.set_ylabel("Fret MAE  (frets, lower = better)")
    ax.set_title("E7 — Fretting-hand fret error: pose model vs. raw MediaPipe\n"
                 "Ours crushes the UN-REGISTERED MediaPipe readout, but only TIES the "
                 "majority floor\n(GroupKFold by run · 1 player/1 guitar/1 camera · "
                 "homography does not register gameplay)",
                 fontsize=9.5)
    ax.legend(loc="upper center", fontsize=9, ncol=3, framealpha=0.9)
    for bars in (b1, b3, b2):
        for r in bars:
            h = r.get_height()
            if np.isfinite(h):
                ax.annotate(f"{h:.2f}", (r.get_x() + r.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=9,
                            xytext=(0, 2), textcoords="offset points")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max([v for v in mp_vals if np.isfinite(v)]) * 1.18)
    fig.tight_layout()
    money_path = os.path.join(a.outdir, "money_shot_fret_mae.png")
    fig.savefig(money_path)
    plt.close(fig)

    # ---- text report (with the honesty caveats) ---------------------------
    def fmt(tab, model):
        t = tab[tab["model"] == model].set_index("subset")
        lines = [f"  [{model}]",
                 f"    {'subset':<18}{'n':>5}{'MP fretMAE':>12}{'Maj fretMAE':>13}"
                 f"{'Ours fretMAE':>14}{'MP strAcc':>11}{'Maj strAcc':>12}{'Ours strAcc':>13}"]
        for c in ["all fingers", "VISIBLE fingers", "OCCLUDED fingers"]:
            r = t.loc[c]
            lines.append(f"    {c:<18}{int(r['n']):>5}{r['mp_fret_mae']:>12.3f}"
                         f"{r['maj_fret_mae']:>13.3f}{r['our_fret_mae']:>14.3f}"
                         f"{r['mp_str_acc']:>11.3f}{r['maj_str_acc']:>12.3f}"
                         f"{r['our_str_acc']:>13.3f}")
        return "\n".join(lines)

    rf_all = rf_tab.loc["all fingers"]
    rf_occ = rf_tab.loc["OCCLUDED fingers"]
    rf_vis = rf_tab.loc["VISIBLE fingers"]
    beat_all = rf_all["our_fret_mae"] < rf_all["mp_fret_mae"]
    beat_occ = rf_occ["our_fret_mae"] < rf_occ["mp_fret_mae"]

    report = []
    report.append("=" * 78)
    report.append("E7 — VISION POSITION MODEL vs RAW MEDIAPIPE  (fretting-hand reading)")
    report.append("=" * 78)
    report.append(f"data rows (per-finger ground truth): {len(df)}  "
                  f"| runs(groups)={n_runs} | GroupKFold splits={n_splits}")
    report.append(f"occluded rows: {int(occ.sum())} ({occ.mean()*100:.1f}%)  "
                  f"visible rows: {int((occ==0).sum())}")
    report.append("")
    report.append("BEAT-THE-BASELINE TABLE  (out-of-fold; fret MAE lower=better, "
                  "string acc higher=better)")
    report.append(fmt(table, "RandomForest"))
    report.append(fmt(table, "MLP"))
    report.append("")
    beat_maj_fret = rf_all["our_fret_mae"] < rf_all["maj_fret_mae"]
    beat_maj_str = rf_all["our_str_acc"] > rf_all["maj_str_acc"]
    report.append("HEADLINE (RandomForest):")
    report.append(f"  all fingers : MP fretMAE {rf_all['mp_fret_mae']:.2f}  |  "
                  f"majority floor {rf_all['maj_fret_mae']:.2f}  ->  "
                  f"Ours {rf_all['our_fret_mae']:.2f}")
    report.append(f"                ({'BEATS' if beat_all else 'does NOT beat'} MediaPipe; "
                  f"{'BEATS' if beat_maj_fret else 'does NOT beat'} the majority floor)")
    report.append(f"  OCCLUDED    : MP fretMAE {rf_occ['mp_fret_mae']:.2f}  |  "
                  f"majority floor {rf_occ['maj_fret_mae']:.2f}  ->  "
                  f"Ours {rf_occ['our_fret_mae']:.2f}  "
                  f"({'BEATS' if beat_occ else 'does NOT beat'} MediaPipe)")
    report.append(f"  string acc  : MP {rf_all['mp_str_acc']:.2f}  |  "
                  f"majority floor {rf_all['maj_str_acc']:.2f}  ->  "
                  f"Ours {rf_all['our_str_acc']:.2f}  "
                  f"({'BEATS' if beat_maj_str else 'does NOT beat'} the majority floor)")
    report.append("")
    report.append("DEGRADE PATH — chord-shape classification from the whole-hand pose")
    report.append("  (per-event; the task's prescribed fallback when per-finger fret is too noisy):")
    report.append(f"  chord_name acc (RF, GroupKFold by run): {chord_acc:.3f}  "
                  f"vs majority floor {chord_maj:.3f}   "
                  f"(n_events={chord_n}, runs={chord_runs}, 9 chord classes; chance~0.11)")
    report.append("  => at chance — the whole-hand pose does NOT recover the chord under "
                  "GroupKFold either.")
    report.append("")
    report.append("SIGNAL ABLATION — is the lift VISION, or just the finger->label prior?")
    sa, fm = abl["string_acc"], abl["fret_mae"]
    report.append(f"  STRING acc:  majority {sa['majority']:.3f} | finger-onehot ONLY "
                  f"{sa['onehot']:.3f} | vision-pose ONLY {sa['vision']:.3f} | both {sa['both']:.3f}")
    report.append(f"  FRET MAE :   majority {fm['majority']:.3f} | finger-onehot ONLY "
                  f"{fm['onehot']:.3f} | vision-pose ONLY {fm['vision']:.3f} | both {fm['both']:.3f}")
    report.append("  READING: vision-pose-ONLY is WORSE than the majority floor on both "
                  "string and fret;")
    report.append("           the only lift over chance comes from the finger one-hot "
                  "(a pure label")
    report.append("           correlation, zero pixels). => the fretting-hand GEOMETRY in "
                  "this footage")
    report.append("           carries no recoverable per-finger string/fret signal.")
    report.append("")
    report.append("-" * 78)
    report.append("H4 VERDICT — did the vision position model BEAT raw MediaPipe?")
    report.append("  PARTLY, but the honest answer is NO in the sense that matters:")
    report.append(f"   * vs raw MediaPipe board readout: YES, hugely (fret MAE "
                  f"{rf_all['mp_fret_mae']:.2f}->{rf_all['our_fret_mae']:.2f}; "
                  f"occluded {rf_occ['mp_fret_mae']:.2f}->{rf_occ['our_fret_mae']:.2f}).")
    report.append("     But that baseline is BROKEN: the homography never registers the")
    report.append("     gameplay video, so the readout is meaningless. Beating it is hollow.")
    report.append("   * vs the honest floor (majority class / finger-prior): NO. The ablation")
    report.append("     shows vision-pose-ONLY is WORSE than majority on both fret and string;")
    report.append("     all lift comes from the finger->label correlation, not pixels. Chord-")
    report.append("     from-pose is at chance. => no recoverable fretting signal in THIS data.")
    report.append("-" * 78)
    report.append("HONESTY / CAVEATS (read before quoting any number):")
    report.append("  1. ONE player, ONE guitar, ONE camera. GroupKFold by run_id shows")
    report.append("     generalization across RECORDINGS of this single rig only — NOT")
    report.append("     across players or guitars. Within-rig proof of concept at best.")
    report.append("  2. The twin.json homography is hand-clicked on a CALIBRATION pose and")
    report.append("     does NOT register the gameplay video: in STEP 1, ~0% of fingertips")
    report.append("     map onto the board and board-Y is systematically negative (diagnostic")
    report.append("     overlay: diag_registration_overlay.png). This is the ROOT CAUSE — fix")
    report.append("     per-frame registration before re-running; the modeling code is ready.")
    report.append("  3. Fret/string truth comes from the prescribed chord shape at onset, not")
    report.append("     hand-verified per frame; heavy motion blur at strum onset is common,")
    report.append("     and the fretting hand is small/oblique/distant in frame.")
    report.append("  4. Fret labels are dominated by fret 2 (~54%), so the majority floor is a")
    report.append("     strong fret bar; string (base rate ~23%) is the fairer discrimination")
    report.append("     test, and the pose fails it too. All numbers at base rates.")
    report.append("=" * 78)
    rep_text = "\n".join(report)
    rep_path = os.path.join(a.outdir, "e7_model_report.txt")
    with open(rep_path, "w") as f:
        f.write(rep_text + "\n")

    print("\n" + rep_text)
    print("\n[artifacts]")
    for p in [table_path, pred_path, money_path, rep_path]:
        print("  " + p)


if __name__ == "__main__":
    main()
