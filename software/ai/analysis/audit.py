#!/usr/bin/env python3
"""
Tactus offline-analysis pipeline — stage (final): audit.py

The one-screen batch AUDIT report — "the interval audit, your anti-digression
dashboard" (docs/24 §8). After every ~15-20 min data batch this renders ONE
self-contained HTML screen that catches drift instantly, plus a companion
audit.json with the computed summary numbers.

Inputs:
  * events_df : the per-note events DataFrame from segment.py
                (columns == schema.EVENT_COLUMNS).
  * metrics   : the separability dict from collapse.py (split, per-modality
                pairwise d', confusion matrices, V1/V2 verdicts).

Sections (docs/24 §8 / §9):
  1. Header        — title, n events / players / runs, date, split.
  2. Technical QC  — % clipped, % silent, peak_dbfs histogram, flagged runs.
  3. Segmentation  — detected-vs-expected note count per run; mis-seg flags.
  4. Label integrity — F0-vs-prompt agreement (label_fret_match); worst runs.
  5. Separability  — pairwise d' (audio vs fused), V1/V2 verdicts, fused
                     confusion heatmap; "is fusion separating the buzz pair?"
  6. Coverage grid — 6-string x 6-fret grid per core class, cells by count.
  7. Recommendation — 2-4 auto-generated next-step bullets, from the data.

Everything is inline (CSS + base64-PNG charts + inline-SVG grids): no external
files, no CDNs. Defensive throughout — any missing column / empty input renders
the affected section as "no data" rather than raising.

Conventions match the rest of the pipeline: stdlib paths from __file__, flat
imports, no package install. schema.py is the FROZEN contract — imported, never
edited.
"""
import os
import sys
import json
import base64
import datetime
import html as _html
from io import BytesIO

# Flat imports, no package install (matches software/ai/capture). This module is
# IN the analysis dir, so `import schema` resolves once the dir is on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema  # noqa: E402  (frozen contract — imported, never edited)

schema.on_path()  # analysis dir + vision dir on sys.path

# Matplotlib MUST be non-interactive (no display in the offline pipeline). Set the
# Agg backend BEFORE importing pyplot, or import order can pick a GUI backend.
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# ===================================================================== thresholds
# Status thresholds — tuned to the batch-killers in docs/24 §9.
CLIP_RED = 5.0          # % clipped > this  -> red  (clipping destroys the buzz band)
CLIP_AMBER = 1.0        # % clipped > this  -> amber
SILENT_RED = 10.0       # % silent  > this  -> red
SILENT_AMBER = 3.0      # % silent  > this  -> amber
LABEL_AGREE_RED = 0.80  # F0-vs-prompt agreement < this -> red (mislabels poison ~6 notes)
LABEL_AGREE_AMBER = 0.92
DPRIME_RISING = 1.0     # clean-vs-buzz d' >= this reads as "separating"
COVERAGE_TARGET = 8     # samples/cell to count a (string,fret,class) cell "green"
COVERAGE_THIN = 1       # 1..target-1 = amber (thin); 0 = red (empty)
N_WORST_RUNS = 5        # how many worst-offender runs to list


# ===================================================================== tiny utils
def _esc(x):
    """HTML-escape any value as text."""
    return _html.escape("" if x is None else str(x))


def _pct(num, den):
    """num/den as a percent float; 0.0 when den == 0."""
    den = float(den or 0)
    return (100.0 * float(num) / den) if den else 0.0


def _fmt_pct(v):
    return "%.1f%%" % float(v)


def _fmt_num(v, nd=2):
    """Format a possibly-None/NaN number; em-dash when not available."""
    try:
        if v is None:
            return "&mdash;"
        f = float(v)
        if f != f:  # NaN
            return "&mdash;"
        return ("%." + str(nd) + "f") % f
    except (TypeError, ValueError):
        return "&mdash;"


def _has_col(df, col):
    """True iff df is a non-empty DataFrame carrying a usable column `col`."""
    return (
        isinstance(df, pd.DataFrame)
        and not df.empty
        and col in df.columns
    )


def _status(value, red_at, amber_at, higher_is_worse=True):
    """Map a value -> 'red'|'amber'|'green' given the two thresholds."""
    if value is None or (isinstance(value, float) and value != value):
        return "amber"
    if higher_is_worse:
        if value > red_at:
            return "red"
        if value > amber_at:
            return "amber"
        return "green"
    else:
        if value < red_at:
            return "red"
        if value < amber_at:
            return "amber"
        return "green"


def _dot(status):
    """An inline status dot <span> for the given 'red'|'amber'|'green'."""
    return '<span class="dot dot-%s" title="%s"></span>' % (status, status)


def _truthy_series(df, col):
    """Boolean Series for a column, NaN treated as False. Empty Series if absent."""
    if not _has_col(df, col):
        return pd.Series([], dtype=bool)
    return df[col].fillna(False).astype(bool)


def _fig_to_b64_png(fig, dpi=96):
    """Render a matplotlib figure to a base64-encoded PNG <img> src string, then
    close it. Returns '' on any failure (so a chart never breaks the page)."""
    try:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return "data:image/png;base64," + b64
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        return ""


def _img(src, alt=""):
    return ('<img alt="%s" src="%s"/>' % (_esc(alt), src)) if src else \
           '<div class="nodata">no data</div>'


# ===================================================================== charts
# Dark-theme palette so embedded charts blend into the dashboard card background.
_CHART_BG = "#0d1117"
_CHART_INK = "#e6edf3"
_CHART_SUB = "#8b949e"


def _style_dark_ax(ax):
    """Apply the dashboard's dark palette to a matplotlib Axes in place."""
    ax.set_facecolor(_CHART_BG)
    for spine in ax.spines.values():
        spine.set_color(_CHART_SUB)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=_CHART_SUB, labelsize=7)
    ax.xaxis.label.set_color(_CHART_INK)
    ax.yaxis.label.set_color(_CHART_INK)


def _peak_hist_png(df):
    """Small histogram of audio_peak_dbfs across events. '' if no usable data."""
    if not _has_col(df, "audio_peak_dbfs"):
        return ""
    vals = pd.to_numeric(df["audio_peak_dbfs"], errors="coerce").dropna()
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return ""
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    ax.hist(vals.values, bins=min(20, max(5, int(np.sqrt(len(vals))))),
            color="#5b8def", edgecolor="#1b2436")
    # 0 dBFS = clipping ceiling; mark it.
    ax.axvline(0.0, color="#e0564b", lw=1.2, ls="--")
    ax.set_xlabel("peak dBFS", fontsize=8)
    ax.set_ylabel("events", fontsize=8)
    _style_dark_ax(ax)
    fig.patch.set_alpha(0.0)
    return _fig_to_b64_png(fig)


def _confusion_png(matrix, labels):
    """Heatmap of a confusion matrix (row-normalized). '' if not renderable."""
    try:
        m = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        return ""
    if m.ndim != 2 or m.size == 0 or m.shape[0] != m.shape[1]:
        return ""
    labels = list(labels) if labels else ["c%d" % i for i in range(m.shape[0])]
    if len(labels) != m.shape[0]:
        labels = ["c%d" % i for i in range(m.shape[0])]
    rowsum = m.sum(axis=1, keepdims=True)
    norm = np.divide(m, rowsum, out=np.zeros_like(m), where=rowsum > 0)
    n = m.shape[0]
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    im = ax.imshow(norm, cmap="magma", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [str(s).replace("buzz-", "b-") for s in labels]
    ax.set_xticklabels(short, fontsize=7, rotation=35, ha="right", color=_CHART_SUB)
    ax.set_yticklabels(short, fontsize=7, color=_CHART_SUB)
    ax.set_xlabel("predicted", fontsize=8, color=_CHART_INK)
    ax.set_ylabel("true", fontsize=8, color=_CHART_INK)
    ax.tick_params(colors=_CHART_SUB)
    for i in range(n):
        for j in range(n):
            frac = norm[i, j]
            ax.text(j, i, "%.0f" % m[i, j],
                    ha="center", va="center", fontsize=7,
                    color="#ffffff" if frac < 0.55 else "#111111")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6, colors=_CHART_SUB)
    cbar.outline.set_edgecolor(_CHART_SUB)
    fig.patch.set_alpha(0.0)
    return _fig_to_b64_png(fig)


# ===================================================================== metric digs
def _metrics_get(metrics, key, default=None):
    return metrics.get(key, default) if isinstance(metrics, dict) else default


def _find_pair_dprime(modality_block, a, b):
    """Pull d'(a,b) out of a modality block's pairwise_dprime, however it's keyed.

    collapse.py may key pairwise_dprime by a tuple, a "a|b" string, an "a__b"
    string, or nest {a: {b: val}}. Order-insensitive. Returns float or None."""
    if not isinstance(modality_block, dict):
        return None
    pw = modality_block.get("pairwise_dprime")
    if not isinstance(pw, dict):
        return None
    want = {a, b}
    # Direct tuple / frozenset keys.
    for k, v in pw.items():
        if isinstance(k, (tuple, list)) and set(k) == want:
            return _coerce_float(v)
        if isinstance(k, frozenset) and set(k) == want:
            return _coerce_float(v)
    # String keys with a separator.
    for sep in ("|", "__", "::", " vs ", "-vs-", ","):
        for ka, kb in ((a, b), (b, a)):
            v = pw.get("%s%s%s" % (ka, sep, kb))
            if v is not None:
                return _coerce_float(v)
    # Nested dict {a: {b: v}}.
    for ka, kb in ((a, b), (b, a)):
        inner = pw.get(ka)
        if isinstance(inner, dict) and kb in inner:
            return _coerce_float(inner[kb])
    return None


def _coerce_float(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _confusion_from_block(block):
    """(matrix, class_order) out of a modality block, tolerating key variants."""
    if not isinstance(block, dict):
        return None, None
    matrix = None
    for k in ("confusion", "confusion_matrix", "cm"):
        if k in block:
            matrix = block[k]
            break
    order = None
    for k in ("class_order", "classes", "labels", "label_order"):
        if k in block:
            order = block[k]
            break
    return matrix, order


# ===================================================================== sections
def _section_header(events_df, metrics, title, computed):
    n_events = int(len(events_df)) if isinstance(events_df, pd.DataFrame) else 0
    n_players = (events_df["player_id"].nunique()
                 if _has_col(events_df, "player_id") else 0)
    n_runs = (events_df["run_id"].nunique()
              if _has_col(events_df, "run_id") else 0)
    split = _metrics_get(metrics, "split", "&mdash;")
    date = datetime.date.today().isoformat()
    computed["n_events"] = n_events
    computed["n_players"] = int(n_players)
    computed["n_runs"] = int(n_runs)
    computed["split"] = None if split == "&mdash;" else split
    computed["date"] = date
    return """
<header class="card head">
  <div class="title">%s</div>
  <div class="meta">
    <span><b>%d</b> events</span>
    <span><b>%d</b> players</span>
    <span><b>%d</b> runs</span>
    <span>split: <b>%s</b></span>
    <span class="date">%s</span>
  </div>
</header>""" % (_esc(title), n_events, int(n_players), int(n_runs),
                _esc(split), _esc(date))


def _section_technical(events_df, computed):
    n = int(len(events_df)) if isinstance(events_df, pd.DataFrame) else 0
    clipped = _truthy_series(events_df, "audio_clipped")
    silent = _truthy_series(events_df, "audio_silent")
    n_clipped = int(clipped.sum()) if not clipped.empty else 0
    n_silent = int(silent.sum()) if not silent.empty else 0
    pct_clipped = _pct(n_clipped, n)
    pct_silent = _pct(n_silent, n)

    # A "flagged run" = any run touched by clipping or silence.
    flagged_runs = 0
    if _has_col(events_df, "run_id") and (not clipped.empty or not silent.empty):
        bad = pd.Series(False, index=events_df.index)
        if not clipped.empty:
            bad = bad | clipped
        if not silent.empty:
            bad = bad | silent
        flagged_runs = int(events_df.loc[bad, "run_id"].nunique())

    st_clip = _status(pct_clipped, CLIP_RED, CLIP_AMBER)
    st_silent = _status(pct_silent, SILENT_RED, SILENT_AMBER)

    computed["technical"] = {
        "pct_clipped": round(pct_clipped, 3),
        "pct_silent": round(pct_silent, 3),
        "n_clipped": n_clipped,
        "n_silent": n_silent,
        "flagged_runs": flagged_runs,
        "status_clipped": st_clip,
        "status_silent": st_silent,
    }

    hist = _peak_hist_png(events_df)
    body = """
    <div class="kv">
      <div>%s clipped <b>%s</b> <span class="sub">(%d)</span></div>
      <div>%s silent <b>%s</b> <span class="sub">(%d)</span></div>
      <div>flagged runs <b>%d</b></div>
    </div>
    <div class="chartwrap">%s</div>""" % (
        _dot(st_clip), _fmt_pct(pct_clipped), n_clipped,
        _dot(st_silent), _fmt_pct(pct_silent), n_silent,
        flagged_runs, _img(hist, "peak dBFS histogram"))
    if n == 0:
        body = '<div class="nodata">no events</div>'
    return _card("2 &middot; Technical QC", body)


def _per_run_detected_expected(events_df):
    """Per-run detected count and (if present) expected count. Returns a DataFrame
    with columns run_id, detected, expected (expected may be NaN)."""
    if not _has_col(events_df, "run_id"):
        return pd.DataFrame(columns=["run_id", "detected", "expected"])
    g = events_df.groupby("run_id")
    detected = g.size().rename("detected")
    out = detected.reset_index()
    # Expected may ride along on events if segment/orchestrator attached it.
    exp_col = None
    for c in ("expected_note_count", "expected_n", schema.M_EXPECTED_N):
        if c in events_df.columns:
            exp_col = c
            break
    if exp_col is not None:
        expected = g[exp_col].first().reset_index(drop=True)
        out["expected"] = pd.to_numeric(expected, errors="coerce").values
    else:
        out["expected"] = np.nan
    return out


def _section_segmentation(events_df, computed):
    table = _per_run_detected_expected(events_df)
    if table.empty:
        computed["segmentation"] = {"n_runs": 0, "n_mis_segmented": 0,
                                    "has_expected": False}
        return _card("3 &middot; Segmentation", '<div class="nodata">no runs</div>')

    has_expected = bool(table["expected"].notna().any())
    mis = []
    if has_expected:
        for _, r in table.iterrows():
            exp = r["expected"]
            det = r["detected"]
            if exp is not None and exp == exp and exp > 0 and det != exp:
                mis.append((str(r["run_id"]), int(det), int(exp)))
    computed["segmentation"] = {
        "n_runs": int(len(table)),
        "n_mis_segmented": len(mis),
        "has_expected": has_expected,
        "detected_min": int(table["detected"].min()),
        "detected_max": int(table["detected"].max()),
        "detected_mean": round(float(table["detected"].mean()), 2),
    }

    # A compact detected-per-run distribution as a tiny PNG.
    chart = ""
    try:
        fig, ax = plt.subplots(figsize=(3.4, 1.7))
        ax.hist(table["detected"].values,
                bins=range(int(table["detected"].min()),
                           int(table["detected"].max()) + 2),
                color="#5bbf8d", edgecolor="#163025", align="left")
        ax.set_xlabel("detected notes / run", fontsize=8)
        ax.set_ylabel("runs", fontsize=8)
        _style_dark_ax(ax)
        fig.patch.set_alpha(0.0)
        chart = _fig_to_b64_png(fig)
    except Exception:
        chart = ""

    if has_expected:
        head = ("<div class=\"kv\"><div>mis-segmented runs <b>%d</b> / %d</div></div>"
                % (len(mis), len(table)))
        if mis:
            rows = "".join(
                "<tr><td>%s</td><td>%d</td><td>%d</td></tr>" % (_esc(rid), d, e)
                for rid, d, e in mis[:N_WORST_RUNS])
            tbl = ("<table><thead><tr><th>run</th><th>detected</th>"
                   "<th>expected</th></tr></thead><tbody>%s</tbody></table>" % rows)
        else:
            tbl = '<div class="ok">all runs match expected count</div>'
        body = head + tbl + ('<div class="chartwrap">%s</div>' % _img(chart))
    else:
        body = ("<div class=\"sub\">no expected-count on events &mdash; showing "
                "detected-per-run distribution</div>"
                '<div class="chartwrap">%s</div>' % _img(chart))
    return _card("3 &middot; Segmentation", body)


def _section_label_integrity(events_df, computed):
    if not _has_col(events_df, "label_fret_match"):
        computed["label_integrity"] = {"n_checked": 0, "agreement": None}
        return _card("4 &middot; Label integrity",
                     '<div class="nodata">no label_fret_match column</div>')

    col = events_df["label_fret_match"]
    # Exclude NaN/None (chord/overflow/no-F0 events are "not applicable").
    mask = col.notna()
    checked = col[mask]
    n_checked = int(len(checked))
    if n_checked == 0:
        computed["label_integrity"] = {"n_checked": 0, "agreement": None}
        return _card("4 &middot; Label integrity",
                     '<div class="nodata">no checkable events (all NaN)</div>')

    truthy = checked.astype(bool)
    agreement = float(truthy.mean())
    st = _status(agreement, LABEL_AGREE_RED, LABEL_AGREE_AMBER,
                 higher_is_worse=False)

    # Worst-offender runs = lowest per-run agreement (need run_id).
    worst = []
    if _has_col(events_df, "run_id"):
        sub = events_df.loc[mask, ["run_id", "label_fret_match"]].copy()
        sub["m"] = sub["label_fret_match"].astype(bool)
        grp = sub.groupby("run_id")["m"]
        per_run = grp.mean()
        counts = grp.size()
        order = per_run.sort_values(kind="mergesort")
        for rid in order.index:
            if per_run[rid] < 1.0:  # only surface runs with real disagreement
                worst.append((str(rid), float(per_run[rid]), int(counts[rid])))
            if len(worst) >= N_WORST_RUNS:
                break

    computed["label_integrity"] = {
        "n_checked": n_checked,
        "agreement": round(agreement, 4),
        "status": st,
        "worst_runs": [
            {"run_id": r, "agreement": round(a, 3), "n": c} for r, a, c in worst
        ],
    }

    head = ('<div class="kv"><div>%s F0-vs-prompt agreement '
            '<b>%s</b> <span class="sub">(%d events)</span></div></div>' %
            (_dot(st), _fmt_pct(agreement * 100.0), n_checked))
    if worst:
        rows = "".join(
            "<tr><td>%s</td><td>%s</td><td>%d</td></tr>"
            % (_esc(r), _fmt_pct(a * 100.0), c) for r, a, c in worst)
        tbl = ("<table><thead><tr><th>worst run</th><th>agree</th>"
               "<th>n</th></tr></thead><tbody>%s</tbody></table>" % rows)
    else:
        tbl = '<div class="ok">every checked run agrees with its prompt</div>'
    return _card("4 &middot; Label integrity", head + tbl)


def _section_separability(metrics, computed):
    audio = _metrics_get(metrics, "audio", {})
    fused = _metrics_get(metrics, "fused", {})

    pairs = [("clean", "buzz-light"), ("buzz-light", "buzz-placement")]
    rows_html = []
    dprime_dump = {"audio": {}, "fused": {}}
    for a, b in pairs:
        da = _find_pair_dprime(audio, a, b)
        df_ = _find_pair_dprime(fused, a, b)
        key = "%s|%s" % (a, b)
        dprime_dump["audio"][key] = da
        dprime_dump["fused"][key] = df_
        rows_html.append(
            "<tr><td>%s vs %s</td><td>%s</td><td>%s</td></tr>" % (
                _esc(a.replace("buzz-", "b-")), _esc(b.replace("buzz-", "b-")),
                _fmt_num(da), _fmt_num(df_)))

    v1 = _metrics_get(metrics, "v1_audio_confuses_buzz_pair")
    v2 = _metrics_get(metrics, "v2_fusion_separates_buzz_pair")

    # Headline verdict: is fusion separating the buzz pair yet?
    if v2 is True:
        verdict_cls, verdict_txt = "green", "YES — fusion separates the buzz pair"
    elif v2 is False:
        verdict_cls, verdict_txt = "red", "NOT YET — buzz pair still entangled in fused"
    else:
        verdict_cls, verdict_txt = "amber", "unknown — v2 verdict not supplied"

    matrix, order = _confusion_from_block(fused)
    heat = _confusion_png(matrix, order)

    computed["separability"] = {
        "pairwise_dprime": dprime_dump,
        "v1_audio_confuses_buzz_pair": v1,
        "v2_fusion_separates_buzz_pair": v2,
        "fused_separates_buzz_pair": (v2 is True),
        "fused_confusion_class_order": list(order) if order is not None else None,
    }

    have_any = any(v is not None
                   for d in dprime_dump.values() for v in d.values())
    if not have_any and not heat and v1 is None and v2 is None:
        return _card("5 &middot; Separability",
                     '<div class="nodata">no metrics supplied</div>')

    tbl = ("<table><thead><tr><th>pair</th><th>d&#39; audio</th>"
           "<th>d&#39; fused</th></tr></thead><tbody>%s</tbody></table>"
           % "".join(rows_html))
    verdicts = (
        '<div class="kv">'
        '<div>%s V1 audio-confuses-pair: <b>%s</b></div>'
        '<div>%s V2 fusion-separates-pair: <b>%s</b></div>'
        '</div>' % (
            _dot("green" if v1 is True else
                 ("amber" if v1 is None else "red")),
            _verdict_word(v1),
            _dot(verdict_cls), _verdict_word(v2)))
    headline = '<div class="verdict verdict-%s">%s</div>' % (
        verdict_cls, _esc(verdict_txt))
    body = (
        '<div class="sep-grid">'
        '<div class="sep-left">%s%s%s</div>'
        '<div class="sep-right"><div class="caption">fused confusion</div>%s</div>'
        '</div>' % (tbl, verdicts, headline, _img(heat, "fused confusion matrix")))
    return _card("5 &middot; Separability", body)


def _verdict_word(v):
    if v is True:
        return "true"
    if v is False:
        return "false"
    return "&mdash;"


def _coverage_counts(events_df, klass):
    """(N_STRINGS x len(FRETS)) int array of event counts for one class.

    Rows index strings 6..1 (top row = string 6 = low E); cols index frets 1..6.
    Defensive: returns zeros if columns are missing."""
    grid = np.zeros((schema.N_STRINGS, len(schema.FRETS)), dtype=int)
    needed = ("intended_class", "string_num", "target_fret")
    if not all(_has_col(events_df, c) for c in needed):
        return grid
    sub = events_df[list(needed)].copy()
    sub = sub[sub["intended_class"] == klass]
    if sub.empty:
        return grid
    s = pd.to_numeric(sub["string_num"], errors="coerce")
    f = pd.to_numeric(sub["target_fret"], errors="coerce")
    fret_index = {fr: i for i, fr in enumerate(schema.FRETS)}
    for sn, fr in zip(s, f):
        if sn != sn or fr != fr:
            continue
        sn = int(sn)
        fr = int(fr)
        if sn < 1 or sn > schema.N_STRINGS:
            continue
        if fr not in fret_index:
            continue
        row = schema.N_STRINGS - sn  # string 6 -> row 0 (top)
        grid[row, fret_index[fr]] += 1
    return grid


def _coverage_grid_svg(grid, klass):
    """Inline-SVG 6x6 heat grid for one class. Cell color by count vs target."""
    nrows, ncols = grid.shape
    cell = 26
    pad_l, pad_t = 30, 22
    w = pad_l + ncols * cell + 8
    h = pad_t + nrows * cell + 18
    parts = ['<svg viewBox="0 0 %d %d" class="cov" '
             'xmlns="http://www.w3.org/2000/svg">' % (w, h)]
    parts.append('<text x="%d" y="12" class="covtitle">%s</text>'
                 % (pad_l, _esc(klass)))
    # Fret labels (cols) 1..6
    for j, fr in enumerate(schema.FRETS):
        cx = pad_l + j * cell + cell / 2
        parts.append('<text x="%.1f" y="%d" class="covlbl" '
                     'text-anchor="middle">%d</text>'
                     % (cx, pad_t - 4, fr))
    n_empty = 0
    n_thin = 0
    n_full = 0
    for i in range(nrows):
        s_num = schema.N_STRINGS - i  # top row = string 6
        parts.append('<text x="%d" y="%.1f" class="covlbl" '
                     'text-anchor="end">%d</text>'
                     % (pad_l - 4, pad_t + i * cell + cell / 2 + 3, s_num))
        for j in range(ncols):
            c = int(grid[i, j])
            if c <= 0:
                fill = "#3a2228"  # empty -> deep red
                n_empty += 1
            elif c < COVERAGE_TARGET:
                t = c / float(COVERAGE_TARGET)
                fill = _amber_ramp(t)  # thin -> amber ramp
                n_thin += 1
            else:
                fill = "#2f8f5b"  # >= target -> green
                n_full += 1
            x = pad_l + j * cell
            y = pad_t + i * cell
            parts.append('<rect x="%d" y="%d" width="%d" height="%d" rx="3" '
                         'fill="%s" stroke="#0d1117" stroke-width="1"/>'
                         % (x, y, cell - 2, cell - 2, fill))
            txt_color = "#eaeef5" if c < COVERAGE_TARGET else "#06140c"
            parts.append('<text x="%.1f" y="%.1f" class="covnum" '
                         'fill="%s" text-anchor="middle">%d</text>'
                         % (x + (cell - 2) / 2, y + (cell - 2) / 2 + 3,
                            txt_color, c))
    parts.append('<text x="%d" y="%d" class="covfoot">%d full / %d thin / %d empty</text>'
                 % (pad_l, h - 4, n_full, n_thin, n_empty))
    parts.append("</svg>")
    return "".join(parts), (n_full, n_thin, n_empty)


def _amber_ramp(t):
    """Color for a 'thin' cell: dark-amber (t->0) to bright-amber (t->1)."""
    t = max(0.0, min(1.0, t))
    r = int(120 + 135 * t)
    g = int(70 + 110 * t)
    b = int(20 + 10 * t)
    return "#%02x%02x%02x" % (r, g, b)


def _section_coverage(events_df, computed):
    cov_summary = {}
    grids_html = []
    any_data = all(_has_col(events_df, c)
                   for c in ("intended_class", "string_num", "target_fret"))
    for klass in schema.CORE_CLASSES:
        grid = _coverage_counts(events_df, klass)
        svg, (full, thin, empty) = _coverage_grid_svg(grid, klass)
        grids_html.append('<div class="covcell">%s</div>' % svg)
        cov_summary[klass] = {
            "total": int(grid.sum()),
            "cells_full": full, "cells_thin": thin, "cells_empty": empty,
        }
    computed["coverage"] = {
        "target_per_cell": COVERAGE_TARGET,
        "n_cells": schema.N_STRINGS * len(schema.FRETS),
        "by_class": cov_summary,
    }
    if not any_data:
        body = ('<div class="nodata">no class/string/fret columns &mdash; '
                'grids shown empty</div><div class="covrow">%s</div>'
                % "".join(grids_html))
    else:
        legend = ('<div class="legend">'
                  '<span><i class="sw" style="background:#2f8f5b"></i>&ge;%d</span>'
                  '<span><i class="sw" style="background:#c87f2a"></i>thin</span>'
                  '<span><i class="sw" style="background:#3a2228"></i>empty</span>'
                  '<span class="sub">rows = string 6&rarr;1, cols = fret 1&rarr;6'
                  '</span></div>' % COVERAGE_TARGET)
        body = legend + ('<div class="covrow">%s</div>' % "".join(grids_html))
    return _card("6 &middot; Coverage grid (6&times;6 &times; 3 classes)", body)


def _build_recommendations(computed):
    """2-4 auto-generated next-step bullets, derived from `computed` (not hardcoded).
    Ordered by urgency: blocking quality issues first, then separability, then
    coverage gaps."""
    recs = []

    tech = computed.get("technical", {})
    if tech.get("status_clipped") == "red":
        recs.append("Clipping high (%s of events) — LOWER input gain; clipping "
                    "destroys the buzz band." % _fmt_pct(tech.get("pct_clipped", 0)))
    if tech.get("status_silent") == "red":
        recs.append("Silent rate high (%s) — check mic/levels; many runs captured "
                    "nothing." % _fmt_pct(tech.get("pct_silent", 0)))

    seg = computed.get("segmentation", {})
    if seg.get("has_expected") and seg.get("n_mis_segmented", 0) > 0:
        recs.append("%d run(s) mis-segmented (detected != expected) — re-check "
                    "onset detection or re-record those runs."
                    % seg["n_mis_segmented"])

    lab = computed.get("label_integrity", {})
    if lab.get("agreement") is not None and lab.get("status") == "red":
        recs.append("Label agreement low (%s) — F0 disagrees with the prompt on "
                    "many notes; inspect the worst runs (mislabeled/misplayed)."
                    % _fmt_pct(lab["agreement"] * 100.0))

    sep = computed.get("separability", {})
    v2 = sep.get("v2_fusion_separates_buzz_pair")
    pw = sep.get("pairwise_dprime", {}) or {}
    fused_pair = (pw.get("fused", {}) or {}).get("buzz-light|buzz-placement")
    if v2 is False:
        recs.append("Light-vs-placement is NOT separating in fused yet — fix how "
                    "you PRODUCE the pair (more deliberate, exaggerated `d`) before "
                    "collecting volume.")
    elif fused_pair is not None and fused_pair < DPRIME_RISING:
        recs.append("Fused d'(light,placement)=%s is weak (<%.1f) — push the "
                    "buzz-cause contrast harder when recording."
                    % (_fmt_num(fused_pair), DPRIME_RISING))

    # Coverage gaps — name the thinnest (string, fret) cells per class.
    thin_cells = _thinnest_cells(computed)
    for klass, sn, fr, cnt in thin_cells[:2]:
        what = "empty" if cnt == 0 else ("thin (%d)" % cnt)
        recs.append("Collect more %s on string %d fret %d — %s."
                    % (klass, sn, fr, what))

    # Always give at least one actionable line; cap at 4.
    if not recs:
        recs.append("No blocking issues — keep collecting to push held-out d' "
                    "higher and fill remaining grid cells.")
    return recs[:4]


def _thinnest_cells(computed):
    """Re-derive the most under-filled (class, string, fret) cells from the raw
    grids stashed during coverage. Returns list of (klass, s_num, fret, count)
    sorted by count ascending (emptiest first)."""
    grids = computed.get("_coverage_grids")
    if not grids:
        return []
    out = []
    for klass, grid in grids.items():
        nrows, ncols = grid.shape
        for i in range(nrows):
            s_num = schema.N_STRINGS - i
            for j in range(ncols):
                fr = schema.FRETS[j]
                out.append((klass, s_num, fr, int(grid[i, j])))
    out.sort(key=lambda t: t[3])
    return out


def _section_recommendation(computed):
    recs = _build_recommendations(computed)
    computed["recommendations"] = recs
    items = "".join("<li>%s</li>" % _esc(r) for r in recs)
    return _card("7 &middot; Recommendation", "<ul class=\"recs\">%s</ul>" % items)


# ===================================================================== shell
def _card(title, body_html):
    return ('<section class="card"><h2>%s</h2>%s</section>'
            % (title, body_html))


_CSS = """
:root{--bg:#0d1117;--card:#161b22;--ink:#e6edf3;--sub:#8b949e;
      --line:#30363d;--accent:#5b8def;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--ink);
     font:13px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;}
.wrap{max-width:1200px;margin:0 auto;padding:14px;}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:12px;}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
      padding:12px 14px;overflow:hidden;}
.card h2{margin:0 0 8px;font-size:13px;font-weight:600;color:var(--sub);
         letter-spacing:.02em;text-transform:uppercase;}
.head{grid-column:span 12;display:flex;align-items:center;
      justify-content:space-between;flex-wrap:wrap;gap:10px;}
.head .title{font-size:20px;font-weight:700;}
.head .meta{display:flex;gap:16px;flex-wrap:wrap;color:var(--sub);font-size:13px;}
.head .meta b{color:var(--ink);}
.head .date{opacity:.8;}
.col-4{grid-column:span 4;} .col-6{grid-column:span 6;}
.col-8{grid-column:span 8;} .col-12{grid-column:span 12;}
.kv{display:flex;flex-direction:column;gap:4px;margin-bottom:8px;}
.kv b{color:var(--ink);} .sub{color:var(--sub);font-size:12px;}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;
     margin-right:6px;vertical-align:middle;}
.dot-green{background:#2f8f5b;box-shadow:0 0 5px #2f8f5b88;}
.dot-amber{background:#c87f2a;box-shadow:0 0 5px #c87f2a88;}
.dot-red{background:#e0564b;box-shadow:0 0 5px #e0564b88;}
.chartwrap{margin-top:6px;} .chartwrap img{max-width:100%;border-radius:6px;}
img{display:block;}
.nodata{color:var(--sub);font-style:italic;padding:14px 4px;}
.ok{color:#2f8f5b;font-size:12px;padding:2px 0;}
table{width:100%;border-collapse:collapse;font-size:12px;margin:4px 0;}
th,td{text-align:left;padding:3px 6px;border-bottom:1px solid var(--line);}
th{color:var(--sub);font-weight:600;}
.sep-grid{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start;}
.sep-left{flex:1 1 250px;min-width:230px;} .sep-right{flex:0 0 auto;}
.caption,.covtitle,.covfoot{color:var(--sub);font-size:11px;}
.caption{margin-bottom:4px;}
.verdict{margin-top:8px;padding:7px 10px;border-radius:7px;font-weight:600;
         font-size:13px;}
.verdict-green{background:#10341f;color:#7ee0a5;border:1px solid #2f8f5b;}
.verdict-amber{background:#33260f;color:#e6b873;border:1px solid #c87f2a;}
.verdict-red{background:#3a1714;color:#f0a59d;border:1px solid #e0564b;}
.covrow{display:flex;gap:14px;flex-wrap:wrap;}
.covcell{background:#0d1117;border:1px solid var(--line);border-radius:8px;
         padding:6px;}
.cov{width:200px;height:auto;} .covlbl{fill:var(--sub);font-size:10px;}
.covnum{font-size:9px;font-weight:600;} .covtitle{font-weight:600;}
.legend{display:flex;gap:14px;align-items:center;margin-bottom:8px;
        flex-wrap:wrap;font-size:12px;color:var(--sub);}
.legend .sw{display:inline-block;width:11px;height:11px;border-radius:3px;
            margin-right:5px;vertical-align:middle;}
.recs{margin:0;padding-left:18px;} .recs li{margin:5px 0;}
.foot{color:var(--sub);font-size:11px;text-align:center;padding:10px 0 4px;}
"""


def _assemble_html(title, sections):
    head, tech, seg, label, sep, cov, rec = sections
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>%s</title>
<style>%s</style></head>
<body><div class="wrap"><div class="grid">
%s
<div class="col-4">%s</div>
<div class="col-8">%s</div>
<div class="col-6">%s</div>
<div class="col-6">%s</div>
<div class="col-12">%s</div>
<div class="col-12">%s</div>
</div>
<div class="foot">Tactus interval audit &mdash; docs/24 &sect;8 (anti-drift dashboard)</div>
</div></body></html>""" % (
        _esc(title), _CSS, head, tech, seg, label, sep, cov, rec)


# ===================================================================== public API
def run(events_df, metrics, out_directory, title="Tactus batch audit"):
    """Render a one-screen self-contained HTML dashboard (+ a companion audit.json)
    into out_directory. Returns the path to the written audit.html.

    Defensive by contract: a missing column, an empty events_df, or a minimal /
    empty metrics dict must NOT raise — the affected section renders "no data".
    """
    # Normalize inputs so the rest of the function can assume the right types.
    if not isinstance(events_df, pd.DataFrame):
        try:
            events_df = pd.DataFrame(events_df)
        except Exception:
            events_df = pd.DataFrame()
    if not isinstance(metrics, dict):
        metrics = {}

    os.makedirs(out_directory, exist_ok=True)
    computed = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "title": title}

    # Stash raw coverage grids so recommendations can name the thinnest cells.
    computed["_coverage_grids"] = {
        k: _coverage_counts(events_df, k) for k in schema.CORE_CLASSES
    }

    # Build each section defensively — one failing section must not sink the page.
    sections = []
    builders = [
        lambda: _section_header(events_df, metrics, title, computed),
        lambda: _section_technical(events_df, computed),
        lambda: _section_segmentation(events_df, computed),
        lambda: _section_label_integrity(events_df, computed),
        lambda: _section_separability(metrics, computed),
        lambda: _section_coverage(events_df, computed),
        lambda: _section_recommendation(computed),  # last: needs prior computed[]
    ]
    fallback_titles = ["", "2 &middot; Technical QC", "3 &middot; Segmentation",
                       "4 &middot; Label integrity", "5 &middot; Separability",
                       "6 &middot; Coverage grid", "7 &middot; Recommendation"]
    for i, b in enumerate(builders):
        try:
            sections.append(b())
        except Exception as e:  # pragma: no cover - defensive belt-and-suspenders
            sys.stderr.write("audit: section %d failed: %r\n" % (i, e))
            if i == 0:
                sections.append('<header class="card head"><div class="title">%s'
                                '</div></header>' % _esc(title))
            else:
                sections.append(_card(fallback_titles[i],
                                      '<div class="nodata">section error</div>'))

    html_str = _assemble_html(title, sections)

    # Drop internal-only keys before serializing the companion JSON.
    computed.pop("_coverage_grids", None)

    html_path = os.path.join(out_directory, "audit.html")
    json_path = os.path.join(out_directory, "audit.json")
    with open(html_path, "w") as fh:
        fh.write(html_str)
    with open(json_path, "w") as fh:
        json.dump(computed, fh, indent=2, default=_json_default)

    sys.stderr.write("audit: wrote %s (%d bytes) + audit.json\n"
                     % (html_path, len(html_str.encode("utf-8"))))
    return html_path


def _json_default(o):
    """Make numpy / pandas scalars JSON-serializable."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if f == f else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


# ===================================================================== CLI
def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Render the one-screen interval AUDIT report (audit.html)."
    )
    ap.add_argument("--session", required=True,
                    help="session_id (data/analysis/<session>/...)")
    ap.add_argument("--player", required=True,
                    help="player_id (data/analysis/<session>/<player>/...)")
    ap.add_argument("--title", default="Tactus batch audit")
    args = ap.parse_args()

    out = schema.out_dir(args.session, args.player)
    events_path = os.path.join(out, "events.csv")
    metrics_path = os.path.join(out, "metrics.json")

    if os.path.exists(events_path):
        events_df = pd.read_csv(events_path)
    else:
        sys.stderr.write("audit: no events.csv at %s (rendering empty)\n"
                         % events_path)
        events_df = pd.DataFrame()

    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as fh:
                metrics = json.load(fh)
        except (ValueError, OSError) as e:
            sys.stderr.write("audit: bad metrics.json (%r); using {}\n" % e)
            metrics = {}
    else:
        sys.stderr.write("audit: no metrics.json at %s (using {})\n"
                         % metrics_path)

    path = run(events_df, metrics, out, title=args.title)
    sys.stderr.write("audit: report at %s\n" % path)


if __name__ == "__main__":
    main()
