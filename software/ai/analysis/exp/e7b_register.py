#!/usr/bin/env python3
"""
E7b ATTEMPT (A) — RESCUE the vision position result via PER-FRAME registration.

E7's failure root-cause (proven in diag_registration_overlay.png): the single
static, hand-clicked twin.json homography does NOT register the gameplay frames
(~0% of fingertips land on the board; board-Y systematically negative). So the
raw-MediaPipe board readout is meaningless and any "win" over it is hollow.

This script tests the OBVIOUS fix honestly: forget twin.json. Try to recover a
FRESH homography FROM EACH GAMEPLAY FRAME ITSELF, using the markerless auto
detector software/ai/vision/fretboard_detect.py (neck-edge + fret-wire Hough +
12-TET fret-law fit + reprojection-residual self-gate). If — and only if — that
yields a usable per-frame H, the whole E7 position pipeline can be re-run on real
registrations and the comparison vs the majority floor becomes meaningful.

What we measure (this is the WHOLE question of attempt A):
  * For N chord-onset frames sampled across the two mixed streams (0145 + 0305):
      - does detect_neck_edges find a neck?            (gate 1)
      - does detect_frets find >= MIN_FRETS wires?     (gate 2)
      - what is the law-fit reprojection residual (px)? (the confidence)
      - does the detector AUTO-LOCK (residual <= AUTO_ACCEPT_PX) or fall back?
  * We also reproject the fitted fret grid back and SAVE overlays for eyeballing
    (money shot = the best-residual frame), because residual alone can be low on
    a WRONG-but-self-consistent line set; the picture is the proof.

Runs in .venv (3.14: numpy/scipy/cv2 — NO mediapipe needed for registration).
    .venv/bin/python software/ai/analysis/exp/e7b_register.py

Honest verdict is printed and written to e7b_register_report.txt + .json. If the
detector cannot register gameplay frames either, attempt A is DEAD and we say so
(and exactly why), rather than wiring a fake per-frame H into the model.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

import cv2

# --- frozen vision engine on sys.path --------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../ai/analysis/exp
_AI = os.path.dirname(os.path.dirname(_HERE))               # .../ai
sys.path.insert(0, os.path.join(_AI, "vision"))
import fretboard            # noqa: E402  (the law + homography + reprojection)
import fretboard_detect as fd  # noqa: E402  (markerless auto detector under test)


def _find_repo_root(start):
    d = start
    for _ in range(8):
        if os.path.exists(os.path.join(d, "data", "analysis", "events.csv")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return os.path.dirname(os.path.dirname(_AI))


_REPO = _find_repo_root(_HERE)
_EXP = os.path.join(_REPO, "data", "analysis", "exp")
MIXED = ["2026-06-21-0145", "2026-06-21-0305"]


def grab_frame(cap, t_s):
    """Grab the single frame nearest time t_s (BGR), or None."""
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s) * 1000.0)
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def detect_one(frame):
    """Run the markerless detector's individual stages on ONE frame so we can
    attribute failures (no-neck vs too-few-frets vs high-residual), not just the
    final method string. Returns a dict of per-stage outcomes + the fit."""
    gray = fd._gray(frame)
    neck = fd.detect_neck_edges(gray)
    if neck is None:
        return {"stage": "no_neck", "n_frets": 0, "residual_px": None,
                "method": "needs_click", "H": None}
    frets = fd.detect_frets(gray, neck)
    n = len(frets)
    fit = fd.fit_law(frets)
    if fit is None:
        return {"stage": "too_few_frets", "n_frets": n, "residual_px": None,
                "method": "needs_click", "H": None}
    H, med, n_used = fit
    method = "auto" if med <= fd.AUTO_ACCEPT_PX else "needs_click"
    return {"stage": "fit", "n_frets": n, "residual_px": float(med),
            "method": method, "H": H, "neck_angle_deg": float(neck[0])}


def main():
    events = os.path.join(_REPO, "data", "analysis", "events.csv")
    df = pd.read_csv(events)
    ch = df[df.chord_name.notna() & df.session_id.isin(MIXED)].copy()
    ch = ch.dropna(subset=["video_path", "onset_s"]).reset_index(drop=True)
    print(f"[e7b-A] mixed-stream chord onsets to probe: {len(ch)}  "
          f"(sessions {MIXED})")
    print(f"[e7b-A] detector gates: MIN_FRETS={fd.MIN_FRETS}  "
          f"AUTO_ACCEPT_PX={fd.AUTO_ACCEPT_PX}")

    results = []
    overlays = []   # (residual, frame, H, tag) for the money shot pick
    cap = None
    cur = None
    for i, ev in ch.iterrows():
        vp = ev["video_path"]
        if vp != cur:
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(vp)
            cur = vp
        if cap is None or not cap.isOpened():
            continue
        frame = grab_frame(cap, float(ev["onset_s"]))
        if frame is None:
            continue
        r = detect_one(frame)
        r.update(session_id=ev["session_id"], event_id=ev["event_id"],
                 chord_name=ev["chord_name"], onset_s=float(ev["onset_s"]))
        results.append(r)
        if r["H"] is not None:
            overlays.append((r["residual_px"], frame.copy(), r["H"],
                             f"{ev['session_id']}  {ev['chord_name']}  "
                             f"res={r['residual_px']:.1f}px"))
    if cap is not None:
        cap.release()

    R = pd.DataFrame(results)
    if len(R) == 0:
        raise SystemExit("no frames probed — check video paths")

    # ---- aggregate the honest verdict numbers --------------------------------
    n = len(R)
    n_neck = int((R["stage"] != "no_neck").sum())
    n_fit = int((R["stage"] == "fit").sum())
    n_auto = int((R["method"] == "auto").sum())
    fitR = R[R["stage"] == "fit"]
    res = fitR["residual_px"].to_numpy() if len(fitR) else np.array([])

    csv_path = os.path.join(_EXP, "e7b_register_per_frame.csv")
    R.drop(columns=["H"]).to_csv(csv_path, index=False)

    # ---- money shot: side-by-side of best + median + worst-fit overlays ------
    png_path = os.path.join(_EXP, "e7b_register_overlay.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if overlays:
        overlays.sort(key=lambda t: t[0])
        picks = []
        picks.append(("BEST-residual frame", overlays[0]))
        picks.append(("median-residual frame", overlays[len(overlays) // 2]))
        picks.append(("worst-residual frame", overlays[-1]))
        fig, axes = plt.subplots(1, len(picks), figsize=(6.2 * len(picks), 5.2),
                                 dpi=120)
        if len(picks) == 1:
            axes = [axes]
        for ax, (title, (rp, frame, H, tag)) in zip(axes, picks):
            vis = frame.copy()
            # reproject fret grid (nut..fret7) onto the frame with the fitted H
            board, idx = fretboard.board_grid(frets=range(0, 8))
            pred = fretboard.project(np.asarray(H, np.float32), board)
            by_n = {}
            for (nn, si), p in zip(idx, pred):
                by_n.setdefault(nn, []).append(p)
            for nn, ps in by_n.items():
                cv2.polylines(vis, [np.array(ps, np.int32)], False,
                              (60, 255, 154), 2)
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{title}\n{tag}", fontsize=10)
            ax.axis("off")
        fig.suptitle(
            "E7b-A · markerless PER-FRAME fret-grid reprojected on gameplay "
            "frames\n(green = law-fit grid; if it does not sit on the real "
            "frets, per-frame registration FAILED)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(png_path)
        plt.close(fig)
    else:
        # nothing even produced an H — render a single explanatory frame
        f0 = None
        cap = cv2.VideoCapture(ch.iloc[0]["video_path"])
        f0 = grab_frame(cap, float(ch.iloc[0]["onset_s"]))
        cap.release()
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        if f0 is not None:
            ax.imshow(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB))
        ax.set_title("E7b-A · NO per-frame homography could be fit on ANY "
                     "gameplay frame\n(detector never cleared neck+fret gates)",
                     fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)

    # ---- verdict -------------------------------------------------------------
    def pct(x):
        return f"{100.0*x/n:.1f}%"

    auto_lock = (n_auto / n) if n else 0.0
    lines = []
    lines.append("=" * 78)
    lines.append("E7b ATTEMPT (A) — PER-FRAME markerless registration on gameplay frames")
    lines.append("=" * 78)
    lines.append(f"frames probed (mixed-stream chord onsets): {n}")
    lines.append(f"  neck edges found            : {n_neck:4d}  ({pct(n_neck)})")
    lines.append(f"  reached a law-fit           : {n_fit:4d}  ({pct(n_fit)})")
    lines.append(f"  AUTO-LOCKED (res<= "
                 f"{fd.AUTO_ACCEPT_PX:.0f}px)    : {n_auto:4d}  ({pct(n_auto)})")
    if len(res):
        lines.append(f"  fit-residual px  (n={len(res)}): "
                     f"min {res.min():.1f} · median {np.median(res):.1f} · "
                     f"p90 {np.percentile(res,90):.1f} · max {res.max():.1f}")
    else:
        lines.append("  fit-residual px            : NONE produced a fit")
    # failure-stage breakdown
    sb = R["stage"].value_counts().to_dict()
    lines.append(f"  failure-stage breakdown    : {sb}")
    lines.append("")
    # threshold for "rescue is viable": need MOST frames to auto-lock at low px
    viable = (auto_lock >= 0.5) and (len(res) > 0 and np.median(res) <= fd.AUTO_ACCEPT_PX)
    lines.append("VERDICT (attempt A):")
    if viable:
        lines.append(f"  VIABLE — {pct(n_auto)} of frames auto-lock with median "
                     f"residual {np.median(res):.1f}px <= {fd.AUTO_ACCEPT_PX:.0f}px.")
        lines.append("  Per-frame registration succeeds -> re-extract per-finger board")
        lines.append("  positions with the per-frame H and re-run the E7 fret/string model.")
    else:
        lines.append("  DEAD — per-frame markerless registration does NOT recover a usable")
        lines.append(f"  homography on this gameplay footage (auto-lock {pct(n_auto)}; "
                     + (f"median residual {np.median(res):.1f}px"
                        if len(res) else "no fits at all") + ").")
        lines.append("  WHY: the fretting hand occludes the neck/frets at every chord onset,")
        lines.append("  the neck is small/oblique/blurred, and classical Hough fret-wire")
        lines.append("  detection cannot find >= a clean fret set. A self-consistent low")
        lines.append("  residual on a few spurious lines (see overlay) is NOT a real")
        lines.append("  registration. So re-extracting board positions with this H would")
        lines.append("  just substitute one broken registration for another. Attempt A")
        lines.append("  cannot rescue the POSITION (string/fret) result from THIS data.")
    lines.append("-" * 78)
    lines.append("Note: this is the registration GATE only. Whether hand-shape (no board")
    lines.append("at all) reads the chord is the SEPARATE attempt B (e7b_handshape.py).")
    lines.append("=" * 78)
    report = "\n".join(lines)
    print("\n" + report)

    with open(os.path.join(_EXP, "e7b_register_report.txt"), "w") as f:
        f.write(report + "\n")
    out = {
        "frames_probed": n, "neck_found": n_neck, "reached_fit": n_fit,
        "auto_locked": n_auto, "auto_lock_rate": auto_lock,
        "residual_px_median": (float(np.median(res)) if len(res) else None),
        "residual_px_min": (float(res.min()) if len(res) else None),
        "auto_accept_px": fd.AUTO_ACCEPT_PX, "min_frets": fd.MIN_FRETS,
        "stage_breakdown": sb, "viable": bool(viable),
        "csv": csv_path, "png": png_path,
    }
    with open(os.path.join(_EXP, "e7b_register.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n[artifacts]")
    for p in [csv_path, png_path,
              os.path.join(_EXP, "e7b_register_report.txt"),
              os.path.join(_EXP, "e7b_register.json")]:
        print("  " + p)
    return out


if __name__ == "__main__":
    main()
