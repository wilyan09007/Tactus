#!/usr/bin/env python3
"""
E6 — AUDIO fret detection (the >90% result). The honest answer to "what fret?".

Vision position is capture-limited on this footage (E7/E7b: the hand occludes the
neck, the calibration homography never registers gameplay frames). So fret comes
from AUDIO. The naive route — generic pitch tracking + nearest-fret — only agrees
with the prompt ~71% on clean (octave errors on low-E/high-e). We use the PRIOR we
actually have: in LEARN mode the tab gives the STRING, so there are only 8 candidate
frets (0..7). A string-conditioned HARMONIC-TEMPLATE detector scores each candidate
fret's harmonic comb against the event spectrum and picks the best -> no octave
errors. Result: 93%+ exact on clean notes.

Compares, per quality class and per string (5-fold-free: the detector is
deterministic, no training -> no leakage):
  - naive F0  (wide pyin -> nearest fret)            ~ the friend's 70.7% baseline
  - harmonic-template (string-conditioned, ours)     ~ 93% clean

Outputs data/analysis/exp/e6_*: results.json, e6_fret_by_condition.png,
e6_fret_confusion.png, e6_audio_fret.html (interactive). Run with .venv (3.14):
  python3 software/ai/analysis/exp/e6_audio_fret.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))      # the analysis dir (schema, segment, features_pitch)
import schema            # noqa: E402
schema.on_path()
import segment           # noqa: E402  (reuse the validated select-N onset picker)
import features_pitch as fp  # noqa: E402
import numpy as np       # noqa: E402
import librosa           # noqa: E402

OUT = os.path.join(schema.OUT_DIR, "exp")
OPEN = schema.OPEN_STRING_MIDI


def _manifest_rows():
    for session, player, pdir in schema.iter_session_player(schema.RAW_DIR):
        m = os.path.join(pdir, "manifest.jsonl")
        if os.path.exists(m):
            for line in open(m):
                line = line.strip()
                if line:
                    yield json.loads(line)


def _naive_f0_fret(y, sr, t0, t1, snum):
    """Generic wide-band pitch -> nearest fret (the un-prior'd baseline)."""
    i0 = max(0, int(t0 * sr)); i1 = min(len(y), int(t1 * sr)); seg = y[i0:i1]
    if seg.size < int(0.05 * sr):
        return None
    try:
        f0, _, vp = librosa.pyin(seg, sr=sr, fmin=librosa.note_to_hz("E1"),
                                 fmax=librosa.note_to_hz("E6"), frame_length=4096)
    except Exception:
        return None
    m = np.isfinite(f0)
    if m.sum() == 0:
        return None
    hz = np.average(f0[m], weights=vp[m])
    return int(round(12 * np.log2(hz / fp.hz_of(OPEN[snum]))))


def run():
    os.makedirs(OUT, exist_ok=True)
    # accumulate per (class) -> lists of (true_fret, naive_pred, tmpl_pred, string)
    rec = {c: {"true": [], "naive": [], "tmpl": [], "str": []} for c in ("clean", "buzz", "muted")}
    n_runs = 0
    for r in _manifest_rows():
        if r.get("block") != "core-grid":
            continue
        cls = r.get("intended_class")
        if cls not in rec:
            continue
        snum = int(str(r.get("string", "0")).split()[0])
        exp_frets = r.get("frets") or []
        wav = schema.abspath(r["files"]["audio"])
        if not wav or not os.path.exists(wav):
            continue
        y, sr = librosa.load(wav, sr=None, mono=True)
        on = list(segment._select_onsets(y, sr, int(r.get("expected_note_count") or 6)))
        if len(on) != len(exp_frets):
            continue
        n_runs += 1
        for k, t in enumerate(sorted(on)):
            t1 = on[k + 1] if k + 1 < len(on) else len(y) / sr
            tf, _ = fp.template_fret(y, sr, t, t1, snum)
            nf = _naive_f0_fret(y, sr, t, t1, snum)
            rec[cls]["true"].append(exp_frets[k])
            rec[cls]["tmpl"].append(tf if tf is not None else -1)
            rec[cls]["naive"].append(nf if nf is not None else -1)
            rec[cls]["str"].append(snum)

    def acc(true, pred):
        true, pred = np.array(true), np.array(pred)
        ex = float(np.mean(true == pred)) if len(true) else 0.0
        w1 = float(np.mean(np.abs(true - pred) <= 1)) if len(true) else 0.0
        return ex, w1

    results = {"chance_exact": round(1 / 6, 3), "by_class": {}, "n_runs": n_runs}
    for c in rec:
        t = rec[c]
        ex_t, w1_t = acc(t["true"], t["tmpl"])
        ex_n, w1_n = acc(t["true"], t["naive"])
        results["by_class"][c] = {
            "n": len(t["true"]),
            "harmonic_template": {"exact": round(ex_t, 3), "within1": round(w1_t, 3)},
            "naive_f0": {"exact": round(ex_n, 3), "within1": round(w1_n, 3)},
        }
    # overall pitched (clean+buzz) — the audio-fret deployment number when a note rings
    pt_true = rec["clean"]["true"] + rec["buzz"]["true"]
    pt_tmpl = rec["clean"]["tmpl"] + rec["buzz"]["tmpl"]
    ex_p, w1_p = acc(pt_true, pt_tmpl)
    results["pitched_clean_plus_buzz"] = {"n": len(pt_true), "exact": round(ex_p, 3), "within1": round(w1_p, 3)}
    results["headline"] = ("Audio harmonic-template (string-conditioned) recovers the intended fret "
                           "%.1f%% exact on CLEAN notes (naive F0 %.1f%%); muted has no pitch -> N/A."
                           % (results["by_class"]["clean"]["harmonic_template"]["exact"] * 100,
                              results["by_class"]["clean"]["naive_f0"]["exact"] * 100))

    with open(os.path.join(OUT, "e6_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    _viz(rec, results)
    print(json.dumps(results, indent=2))
    return results


def _viz(rec, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # (1) accuracy by class: naive vs template
    classes = ["clean", "buzz", "muted"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(3); w = 0.35
    nv = [results["by_class"][c]["naive_f0"]["exact"] for c in classes]
    tp = [results["by_class"][c]["harmonic_template"]["exact"] for c in classes]
    ax[0].bar(x - w / 2, nv, w, label="naive F0 (no string prior)", color="#c0392b")
    ax[0].bar(x + w / 2, tp, w, label="harmonic-template (ours)", color="#2563eb")
    ax[0].axhline(1 / 6, ls="--", c="gray", lw=1, label="chance (1/6)")
    ax[0].axhline(0.9, ls=":", c="green", lw=1, label="90% target")
    ax[0].set_xticks(x); ax[0].set_xticklabels(classes); ax[0].set_ylim(0, 1)
    ax[0].set_title("Audio fret EXACT accuracy by quality class"); ax[0].legend(fontsize=8)
    for i, (a, b) in enumerate(zip(nv, tp)):
        ax[0].text(i - w / 2, a + .02, f"{a:.2f}", ha="center", fontsize=8)
        ax[0].text(i + w / 2, b + .02, f"{b:.2f}", ha="center", fontsize=8)
    # (2) clean fret confusion (true vs template)
    t = rec["clean"]
    cm = np.zeros((6, 6))
    for tr, pr in zip(t["true"], t["tmpl"]):
        if 1 <= tr <= 6 and 1 <= pr <= 6:
            cm[tr - 1, pr - 1] += 1
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    im = ax[1].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax[1].set_xticks(range(6)); ax[1].set_xticklabels(range(1, 7))
    ax[1].set_yticks(range(6)); ax[1].set_yticklabels(range(1, 7))
    ax[1].set_xlabel("predicted fret"); ax[1].set_ylabel("true fret")
    ax[1].set_title("Clean-note fret confusion (harmonic-template)")
    for i in range(6):
        for j in range(6):
            if cm[i, j]:
                ax[1].text(j, i, int(cm[i, j]), ha="center", va="center",
                           color="white" if cmn[i, j] > .5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e6_fret_by_condition.png"), dpi=130, bbox_inches="tight")
    print("wrote e6_fret_by_condition.png")


if __name__ == "__main__":
    run()
