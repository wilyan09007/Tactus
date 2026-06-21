#!/usr/bin/env python3
"""
Tactus offline-analysis pipeline — stage: collapse.py (separability / metrics).

The rigor centerpiece (docs/17 §The-separability-study, docs/20 D5, docs/23 §7.1):

    standardize -> PCA(95%) -> LDA -> Fisher / silhouette / pairwise d' / confusion,
    all **leave-one-player-out** (LOPO).

It exists to prove two scientific claims (docs/20-eng-review V1/V2):

    V1  Audio ALONE confuses the two buzz causes (buzz-light vs buzz-placement):
        pairwise d' on that pair is small (~< 1.0) for the audio-only feature set.
    V2  Adding the vision feature `d_active` SEPARATES them: the fused feature set's
        pairwise d' on the buzz pair is materially larger than audio's and >= ~1.0.

So every metric is computed for TWO matched feature sets and compared:
    'audio' = schema.AUDIO_FEATURES        (timbre/pitch only)
    'fused' = schema.FUSED_FEATURES        (audio + the vision pose geometry, incl. d_active)

ALL evaluation is leave-one-player-out — never a random split — and EVERY
preprocessing step (NaN-column drop is global; mean-imputation, StandardScaler,
PCA, LDA) is fit on the TRAIN players of each fold ONLY, then applied to the
held-out player. Predictions/projections are aggregated across folds and scored
once. With < 2 players LOPO is impossible, so we fall back to stratified 5-fold
CV ("split":"kfold-fallback") and say so.

Conventions match the rest of the pipeline (see segment.py): stdlib paths from
__file__, flat imports, no package install; `import schema` resolves once the
analysis dir is on sys.path. Everything returned is JSON-serializable.
"""
import os
import sys

# Flat imports, no package install (matches software/ai/capture / segment.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema  # noqa: E402  (frozen contract — imported, never edited)

schema.on_path()  # analysis dir + vision dir on sys.path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # noqa: E402
from sklearn.metrics import confusion_matrix, silhouette_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


# ----------------------------------------------------------------- tunables
PCA_VAR = 0.95          # PCA(n_components=0.95): keep 95% of train variance
KFOLD_N = 5             # stratified CV folds when LOPO is unavailable
MIN_PLAYERS_LOPO = 2    # LOPO needs >= 2 players (hold one out, train on the rest)
LOADINGS_TOP_K = 8      # named features reported per discriminant axis (readability)

# V1/V2 decision thresholds (docs/20-eng-review). The headline pair is the two
# buzz causes; "clean vs others" is reported too but does not gate the booleans.
BUZZ_PAIR = ("buzz-light", "buzz-placement")
V1_AUDIO_CONFUSE_DPRIME = 1.0   # audio d'(buzz pair) BELOW this  => audio confuses them
V2_FUSION_GAIN_DPRIME = 0.5     # fused must beat audio by at least this much, AND...
V2_FUSION_MIN_DPRIME = 1.0      # ...fused d'(buzz pair) must reach at least this.


# ----------------------------------------------------------------- numpy -> json
def _f(x):
    """numpy/py scalar -> plain float (None/NaN-safe)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # drop NaN


def _jsonify(obj):
    """Recursively cast numpy types to plain Python so json.dumps works."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _f(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ----------------------------------------------------------------- d-prime
def _dprime(proj, y, c0, c1):
    """Sensitivity index d' between two classes on a 1-D projection.

        d' = (mean1 - mean0) / sqrt(0.5 * (var0 + var1))

    Signal-detection d' with a pooled (equal-weight) within-class std in the
    denominator — the standard two-Gaussian separation measure used for the
    pressure-vs-placement pair (docs/20 §D5, docs/23 §7.1). Sign is made positive
    (we report separation magnitude, not which class sits higher). Returns None
    if either class is empty or the pooled variance is ~0 (degenerate)."""
    proj = np.asarray(proj, dtype=float).ravel()
    a = proj[np.asarray(y) == c0]
    b = proj[np.asarray(y) == c1]
    if a.size == 0 or b.size == 0:
        return None
    v0 = np.var(a, ddof=1) if a.size > 1 else 0.0
    v1 = np.var(b, ddof=1) if b.size > 1 else 0.0
    denom = np.sqrt(0.5 * (v0 + v1))
    if not np.isfinite(denom) or denom < 1e-12:
        return None
    return abs(float(np.mean(b) - np.mean(a)) / denom)


# ----------------------------------------------------------------- folds
def _lopo_folds(players):
    """Leave-one-player-out: yield (train_mask, test_mask) holding out each player
    in turn. `players` is a 1-D array aligned with the rows."""
    players = np.asarray(players)
    for p in pd.unique(players):
        test = players == p
        yield ~test, test


def _kfold_folds(y, n_splits, seed=0):
    """Stratified k-fold fallback when LOPO is impossible (< 2 players). Folds are
    capped at the smallest class count so every fold keeps every class present."""
    y = np.asarray(y)
    counts = pd.Series(y).value_counts()
    n = int(min(n_splits, counts.min())) if len(counts) else 1
    n = max(n, 2)
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    idx = np.arange(len(y))
    for tr, te in skf.split(idx, y):
        train = np.zeros(len(y), dtype=bool)
        test = np.zeros(len(y), dtype=bool)
        train[tr] = True
        test[te] = True
        yield train, test


# ----------------------------------------------------------------- one fold
def _fit_fold(Xtr, ytr):
    """Fit the per-fold preprocessing+model chain on TRAIN ONLY and return a
    callable that transforms a raw feature matrix the same way.

    Chain: mean-impute (train col means) -> StandardScaler -> PCA(95%) -> LDA.
    LDA is the supervised classifier AND the source of the discriminant axes used
    for d'/Fisher/silhouette. Returns (transform_to_lda, predict, lda, n_pca, means)."""
    col_means = np.nanmean(Xtr, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)  # all-NaN col -> 0

    def _impute(X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(col_means, np.where(bad)[1])
        return X

    Xtr_i = _impute(Xtr)
    scaler = StandardScaler().fit(Xtr_i)
    Xtr_s = scaler.transform(Xtr_i)

    # PCA(0.95). Guard tiny folds where 0.95-by-variance is ill-posed: cap at the
    # achievable component count.
    max_comp = max(1, min(Xtr_s.shape[0] - 1, Xtr_s.shape[1]))
    try:
        pca = PCA(n_components=PCA_VAR, svd_solver="full").fit(Xtr_s)
        if pca.n_components_ < 1:
            raise ValueError
    except (ValueError, np.linalg.LinAlgError):
        pca = PCA(n_components=max_comp, svd_solver="full").fit(Xtr_s)
    Xtr_p = pca.transform(Xtr_s)

    lda = LDA().fit(Xtr_p, ytr)

    def transform(X):
        return lda.transform(pca.transform(scaler.transform(_impute(X))))

    def predict(X):
        return lda.predict(pca.transform(scaler.transform(_impute(X))))

    return transform, predict, lda, int(pca.n_components_), scaler, pca


# ----------------------------------------------------------------- Fisher
def _fisher_per_axis(proj, y):
    """Fisher ratio (between-class scatter / within-class scatter) per projected
    axis, computed on the aggregated held-out points. One value per LDA axis."""
    proj = np.atleast_2d(np.asarray(proj, dtype=float))
    if proj.shape[0] == 1 and proj.shape[1] != 1:
        proj = proj.T  # (n_samples, n_axes)
    y = np.asarray(y)
    classes = pd.unique(y)
    out = []
    grand = proj.mean(axis=0)
    for ax in range(proj.shape[1]):
        col = proj[:, ax]
        sb = 0.0  # between
        sw = 0.0  # within
        for c in classes:
            m = col[y == c]
            if m.size == 0:
                continue
            sb += m.size * (m.mean() - grand[ax]) ** 2
            sw += ((m - m.mean()) ** 2).sum()
        out.append(_f(sb / sw) if sw > 1e-12 else None)
    return out


# ----------------------------------------------------------------- per-set run
def _run_feature_set(df, feat_cols, classes, label_col, player_col, use_lopo, want_loadings):
    """Run the full LOPO (or k-fold) separability analysis for ONE feature set.

    Returns the per-set metrics dict described in run()'s docstring."""
    # Drop feature columns that are entirely NaN across the whole subset — they
    # carry no signal and would just be imputed to a constant. (Global, label-free
    # decision; the per-fold mean-imputation still only sees train rows.)
    present = [c for c in feat_cols if c in df.columns]
    kept = [c for c in present if not df[c].isna().all()]
    dropped = [c for c in present if c not in kept]

    y = df[label_col].to_numpy()
    players = df[player_col].to_numpy()
    X = df[kept].to_numpy(dtype=float) if kept else np.empty((len(df), 0))

    folds = list(_lopo_folds(players) if use_lopo else _kfold_folds(y, KFOLD_N))

    # Aggregate held-out predictions and the held-out multi-axis LDA projection.
    agg_true, agg_pred, agg_proj = [], [], []
    # Per-fold pairwise d': a fresh 2-class LDA is fit per pair on the train subset
    # and the held-out pair is projected; we pool the held-out projections across
    # folds, then compute one d' per pair (so d' reflects generalization).
    pair_proj = {}  # (c0,c1) -> {"proj":[...], "y":[...]}

    last_lda = None
    last_scaler = None
    last_pca = None

    for train, test in folds:
        ytr = y[train]
        if len(pd.unique(ytr)) < 2:
            continue  # LDA needs >= 2 classes in train
        transform, predict, lda, _, scaler, pca = _fit_fold(X[train], ytr)
        last_lda, last_scaler, last_pca = lda, scaler, pca

        agg_true.extend(y[test].tolist())
        agg_pred.extend(predict(X[test]).tolist())
        agg_proj.extend(np.atleast_2d(transform(X[test])).tolist())

        # Pairwise 1-D LDA per class pair, fit on the two-class TRAIN subset.
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                c0, c1 = classes[i], classes[j]
                tr_pair = train & ((y == c0) | (y == c1))
                te_pair = test & ((y == c0) | (y == c1))
                if te_pair.sum() == 0:
                    continue
                ytr_p = y[tr_pair]
                if set(np.unique(ytr_p)) != {c0, c1}:
                    continue  # need both classes in train to fit the 1-D axis
                t2, _p2, _l2, _n2, _s2, _pc2 = _fit_fold(X[tr_pair], ytr_p)
                proj = np.asarray(t2(X[te_pair]), dtype=float)[:, 0]  # 2-class LDA -> 1 axis
                d = pair_proj.setdefault((c0, c1), {"proj": [], "y": []})
                d["proj"].extend(proj.tolist())
                d["y"].extend(y[te_pair].tolist())

    # ---- confusion + accuracy
    agg_true = np.asarray(agg_true)
    agg_pred = np.asarray(agg_pred)
    cm = confusion_matrix(agg_true, agg_pred, labels=classes)
    overall_acc = _f((agg_true == agg_pred).mean()) if agg_true.size else None
    per_class_recall = {}
    for k, c in enumerate(classes):
        row = cm[k]
        tot = int(row.sum())
        per_class_recall[c] = _f(row[k] / tot) if tot else None

    # ---- pairwise d'
    pairwise = {}
    for (c0, c1), d in pair_proj.items():
        pairwise["%s__vs__%s" % (c0, c1)] = _dprime(d["proj"], np.asarray(d["y"]), c0, c1)

    # ---- Fisher per retained LDA axis (on aggregated held-out projection)
    proj_arr = np.asarray(agg_proj, dtype=float) if agg_proj else np.empty((0, 0))
    fisher = _fisher_per_axis(proj_arr, agg_true) if proj_arr.size else []

    # ---- silhouette on the LDA-projected held-out points vs true labels
    sil = None
    if proj_arr.size:
        n_lab = len(np.unique(agg_true))
        if n_lab >= 2 and proj_arr.shape[0] > n_lab:  # needs >=2 clusters and > k points
            try:
                sil = _f(silhouette_score(proj_arr, agg_true))
            except ValueError:
                sil = None

    out = {
        "confusion": _jsonify(cm.tolist()),
        "class_order": list(classes),
        "accuracy": overall_acc,
        "per_class_recall": _jsonify(per_class_recall),
        "pairwise_dprime": _jsonify(pairwise),
        "fisher": _jsonify(fisher),
        "silhouette": sil,
        "n_features_used": len(kept),
        "dropped_all_nan": dropped,
    }

    # ---- LDA loadings (FUSED set only): map each discriminant axis back to the
    # named features that load it most. PCA sits between features and LDA, so the
    # feature-space loading is W = pca.components_^T @ lda.scalings_ (chain rule
    # through the linear scaler+PCA+LDA). Report top-|loading| features per axis.
    if want_loadings and last_lda is not None and last_pca is not None and kept:
        scalings = np.atleast_2d(last_lda.scalings_)              # (n_pca, n_axes)
        comp = last_pca.components_                               # (n_pca, n_feat)
        feat_loadings = comp.T @ scalings                        # (n_feat, n_axes)
        loadings = {}
        n_axes = feat_loadings.shape[1]
        for ax in range(n_axes):
            col = feat_loadings[:, ax]
            order = np.argsort(np.abs(col))[::-1][:LOADINGS_TOP_K]
            loadings["axis_%d" % ax] = [
                {"feature": kept[k], "loading": _f(col[k])} for k in order
            ]
        out["lda_loadings"] = _jsonify(loadings)

    return out


# ----------------------------------------------------------------- public API
def run(features_df, label_col=schema.LABEL, player_col=schema.PLAYER,
        classes=schema.CORE_CLASSES):
    """Separability study over audio-only vs fused features (LOPO).

    features_df : one row per event with columns including schema.EVENT_ID, the
                  label (`label_col`), the player (`player_col`), and the feature
                  columns (schema.AUDIO_FEATURES + schema.VISION_FEATURES).

    Runs the standardize -> PCA(95%) -> LDA pipeline under leave-one-player-out for
    the 'audio' (schema.AUDIO_FEATURES) and 'fused' (schema.FUSED_FEATURES) feature
    sets, restricted to `classes`, and returns a JSON-serializable metrics dict:

        {
          "n_events", "n_players", "split": "lopo"|"kfold-fallback",
          "classes", "counts": {class: n, ...},
          "audio":  {confusion, class_order, accuracy, per_class_recall,
                     pairwise_dprime, fisher, silhouette, ...},
          "fused":  {..., "lda_loadings": {axis: [{feature, loading}, ...]}},
          "v1_audio_confuses_buzz_pair":  bool,   # audio d'(buzz pair) < ~1.0
          "v2_fusion_separates_buzz_pair": bool,  # fused d'(buzz pair) materially > audio
        }
    """
    classes = list(classes)
    df = features_df[features_df[label_col].isin(classes)].copy()

    n_players = int(df[player_col].nunique())
    use_lopo = n_players >= MIN_PLAYERS_LOPO
    split = "lopo" if use_lopo else "kfold-fallback"
    counts = {c: int((df[label_col] == c).sum()) for c in classes}

    audio = _run_feature_set(df, schema.AUDIO_FEATURES, classes, label_col,
                             player_col, use_lopo, want_loadings=False)
    fused = _run_feature_set(df, schema.FUSED_FEATURES, classes, label_col,
                             player_col, use_lopo, want_loadings=True)

    # V1/V2 on the headline buzz pair. Key order in pairwise_dprime follows the
    # class order in `classes`, so look the pair up in both orientations.
    def _pair_dprime(metrics, c0, c1):
        pw = metrics.get("pairwise_dprime", {})
        return pw.get("%s__vs__%s" % (c0, c1), pw.get("%s__vs__%s" % (c1, c0)))

    a_pair = _pair_dprime(audio, *BUZZ_PAIR)
    f_pair = _pair_dprime(fused, *BUZZ_PAIR)

    v1 = bool(a_pair is not None and a_pair < V1_AUDIO_CONFUSE_DPRIME)
    v2 = bool(
        a_pair is not None and f_pair is not None
        and (f_pair - a_pair) >= V2_FUSION_GAIN_DPRIME
        and f_pair >= V2_FUSION_MIN_DPRIME
    )

    result = {
        "n_events": int(len(df)),
        "n_players": n_players,
        "split": split,
        "classes": classes,
        "counts": counts,
        "buzz_pair": list(BUZZ_PAIR),
        "buzz_pair_dprime": {"audio": _f(a_pair), "fused": _f(f_pair)},
        "audio": audio,
        "fused": fused,
        "v1_audio_confuses_buzz_pair": v1,
        "v2_fusion_separates_buzz_pair": v2,
    }
    return _jsonify(result)


# ----------------------------------------------------------------- CLI
def main(argv=None):
    """argparse front-end: load features_fused.csv from schema.out_dir(session,player),
    run the separability study, write metrics.json next to it."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Tactus separability study (PCA->LDA, LOPO).")
    ap.add_argument("--session", required=True, help="session_id (data/analysis/<session>/...)")
    ap.add_argument("--player", required=True, help="player_id (data/analysis/<session>/<player>/)")
    ap.add_argument("--features", default="features_fused.csv",
                    help="features CSV filename in the analysis dir (default: features_fused.csv)")
    ap.add_argument("--out", default="metrics.json",
                    help="output JSON filename in the analysis dir (default: metrics.json)")
    args = ap.parse_args(argv)

    out_d = schema.out_dir(args.session, args.player)
    feat_path = os.path.join(out_d, args.features)
    if not os.path.exists(feat_path):
        ap.error("features file not found: %s" % feat_path)

    df = pd.read_csv(feat_path)
    metrics = run(df)

    out_path = os.path.join(out_d, args.out)
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("wrote %s" % out_path)
    print("  split=%s  n_events=%d  n_players=%d"
          % (metrics["split"], metrics["n_events"], metrics["n_players"]))
    print("  buzz-pair d'  audio=%s  fused=%s"
          % (metrics["buzz_pair_dprime"]["audio"], metrics["buzz_pair_dprime"]["fused"]))
    print("  V1 audio confuses buzz pair = %s" % metrics["v1_audio_confuses_buzz_pair"])
    print("  V2 fusion separates buzz pair = %s" % metrics["v2_fusion_separates_buzz_pair"])
    return metrics


if __name__ == "__main__":
    main()
