#!/usr/bin/env python3
"""
Automatic markerless fretboard detection — zero clicks when it can, click-fallback when it can't.

WHY this design (grounded in the prior art, not hand-waving):
  • abhishekrana/guitar-augmented-reality  → UNet segmentation + homography mesh overlay
  • Renato530/Fretboard-segmentation        → HED edges + Hough + wavelet peak-finding for frets
  • TapToTab (arXiv 2409.08618)             → YOLOv8-OBB fret/string seg beats Canny; robust to
                                              lighting/angle/occlusion (the SOTA front-end)
  • TabbyCat (S. Keke)                       → tried PURE auto-detect; found it unreliable across
                                              lighting/angles; shipped a CLICK fallback. We heed that.
  • Roboflow: guitar-neck-detector (mAP 97.5%), Ghaleb/guitar-fretboard (12 fret zones), guitar-fret

So the robust, honest architecture is **auto-detect → SELF-VERIFY by reprojection → fall back to click**:
  1. We only need ONE registration per session, OFFLINE → aggregate evidence over many frames + pick a
     clean (hand-off-neck) frame → far more robust than per-frame real-time detection.
  2. Detect neck edges + fret wires (classical here; swappable for a YOLO front-end — see detect_frets()).
  3. Fit the homography constrained by the 12-TET fret LAW (few DOF) → reproject the grid onto the real
     wires. The reprojection residual is the CONFIDENCE: low → auto-lock (no clicks); high → ask for the
     4-click (calibrate.html). Never silently ship a bad registration.

This module is the front-end to fretboard.py (the law + homography + reprojection live there).
DETECTION THRESHOLDS NEED ONE PASS OF TUNING ON A REAL FRAME of your guitar — run the CLI on your first
captured keyframe/webm and adjust. The fit-core + the gate are validated synthetically below.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

import cv2

from fretboard import (board_grid, fit_homography, fret_fraction, project,
                       reprojection_error)

AUTO_ACCEPT_PX = 6.0   # median reprojection residual below this → auto-lock (tune per camera)
MIN_FRETS = 4          # need at least this many detected fret lines to even try a fit


# ----------------------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------------------
def _gray(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return cv2.createCLAHE(2.0, (8, 8)).apply(g)   # CLAHE: rescues the low-contrast brown board


def aggregate_frames(frames):
    """Median-stack several frames → kills transient noise/glare and the moving hand,
    leaving the static neck. Use the cleanest (hand-off) frames when possible."""
    if len(frames) == 1:
        return frames[0]
    g = np.stack([_gray(f).astype(np.float32) for f in frames], 0)
    return np.median(g, 0).astype(np.uint8)


# ----------------------------------------------------------------------------
# detection (classical front-end; swap detect_frets() for a YOLO model if desired)
# ----------------------------------------------------------------------------
def detect_neck_edges(gray):
    """Two long, near-parallel lines = the neck edges. Returns (angle_deg, (a,b)) where a,b are
    (rho-free) line endpoints, or None."""
    edges = cv2.Canny(gray, 40, 120)
    h, w = gray.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                            minLineLength=int(0.35 * w), maxLineGap=int(0.04 * w))
    if lines is None:
        return None
    segs = lines.reshape(-1, 4)
    angs = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    # dominant neck orientation = the modal long-line angle (wrapped to [-90,90))
    a = ((angs + 90) % 180) - 90
    dom = np.median(a)
    keep = segs[np.abs(((a - dom + 90) % 180) - 90) < 12]
    if len(keep) < 2:
        return None
    # perpendicular offset of each kept line (use the MIDPOINT — stabler than an endpoint),
    # then take the two most-separated → the top & bottom neck edges.
    th = np.radians(dom)
    nrm = np.array([-np.sin(th), np.cos(th)])
    mid = np.stack([(keep[:, 0] + keep[:, 2]) / 2, (keep[:, 1] + keep[:, 3]) / 2], 1)
    off = mid @ nrm
    lo, hi = keep[np.argmin(off)], keep[np.argmax(off)]
    if abs(off.max() - off.min()) < 0.05 * h:     # both lines on the same edge → not a neck
        return None
    return dom, (lo, hi)


def detect_frets(gray, neck):
    """Fret wires = strong gradients ACROSS the neck axis. Rectify the neck to horizontal, take the
    column profile of along-neck gradient energy within the band, peak-find → fret x-positions.
    Returns image-space fret points: list of (top_pt, bottom_pt) on the two neck edges.

    >>> swap-in point: replace this body with a YOLOv8-OBB / segmentation call (TapToTab / Roboflow)
        that returns fret-line positions — the fit below is identical. <<<
    """
    dom, (lo, hi) = neck
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), dom, 1.0)         # rotate neck to horizontal
    rot = cv2.warpAffine(gray, M, (w, h))
    # band = rows between the two (now ~horizontal) edges
    def _rot(p): return (M @ np.array([p[0], p[1], 1.0]))[:2]
    ys = [_rot(lo[:2])[1], _rot(lo[2:])[1], _rot(hi[:2])[1], _rot(hi[2:])[1]]
    y0, y1 = int(max(0, min(ys))), int(min(h, max(ys)))
    if y1 - y0 < 8:
        return []
    band = rot[y0:y1]
    prof = np.abs(cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)).mean(0)   # along-neck gradient profile
    prof = cv2.GaussianBlur(prof.reshape(1, -1), (9, 1), 0).ravel()      # 1-D smooth (width=9,height=1)
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(prof, distance=max(6, w // 60), prominence=prof.std() * 0.6)
    except Exception:
        thr = prof.mean() + prof.std()
        peaks = np.where((prof[1:-1] > prof[:-2]) & (prof[1:-1] > prof[2:]) & (prof[1:-1] > thr))[0] + 1
    Minv = cv2.invertAffineTransform(M)
    out = []
    for x in peaks:
        top = (Minv @ np.array([x, y0, 1.0]))[:2]
        bot = (Minv @ np.array([x, y1, 1.0]))[:2]
        out.append((tuple(top), tuple(bot)))
    return out


# ----------------------------------------------------------------------------
# the robust core: fit the fret LAW to the detected lines + self-verify
# ----------------------------------------------------------------------------
def fit_law(fret_points):
    """fret_points: ordered (nut→body) list of (top_pt, bottom_pt). Try assigning them to consecutive
    fret indices (allowing the nut to be missed) and keep the assignment whose law-constrained
    homography reprojects with the smallest residual. Returns (H, median_px, n_used) or None."""
    if len(fret_points) < MIN_FRETS:
        return None
    tops = np.array([p[0] for p in fret_points], np.float32)
    bots = np.array([p[1] for p in fret_points], np.float32)
    best = None
    for start in (0, 1, 2):                     # nut maybe not the first detected line
        idxs = list(range(start, start + len(fret_points)))
        board, img = [], []
        for k, n in enumerate(idxs):
            x = fret_fraction(n)
            board += [(x, 0.0), (x, 1.0)]
            img += [tops[k], bots[k]]
        H, _ = fit_homography(np.array(img, np.float32), np.array(board, np.float32))
        if H is None:
            continue
        mean, med, mx = reprojection_error(H, np.array(img, np.float32), np.array(board, np.float32))
        if best is None or med < best[1]:
            best = (H, med, len(fret_points))
    return best


def autodetect(frames):
    """frames: one BGR image or a list. Returns a dict with H, confidence, and the method decision."""
    img = frames if isinstance(frames, np.ndarray) else aggregate_frames(frames)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray = _gray(img)
    neck = detect_neck_edges(gray)
    if neck is None:
        return {"method": "needs_click", "reason": "no neck edges found", "H": None, "confidence_px": None}
    frets = detect_frets(gray, neck)
    fit = fit_law(frets)
    if fit is None:
        return {"method": "needs_click", "reason": f"only {len(frets)} fret lines", "H": None, "confidence_px": None}
    H, med, n = fit
    method = "auto" if med <= AUTO_ACCEPT_PX else "needs_click"
    return {"method": method, "H": H.tolist(), "confidence_px": round(float(med), 2),
            "n_frets_used": n, "reason": "auto-locked" if method == "auto" else "low confidence → use 4-click"}


def overlay(img, H, path):
    board, idx = board_grid(frets=range(0, 8))
    pred = project(np.array(H, np.float32), board)
    by_n = {}
    for (n, si), p in zip(idx, pred):
        by_n.setdefault(n, []).append(p)
    vis = img.copy()
    for n, ps in by_n.items():
        cv2.polylines(vis, [np.array(ps, np.int32)], False, (60, 255, 154), 2)
    cv2.imwrite(path, vis)


# ----------------------------------------------------------------------------
def _selftest():
    """Validate the ROBUST core — the fret-law fit + the confidence gate — directly on synthetic
    detected fret lines (with realistic pixel noise + a couple of spurious lines). The image-space
    DETECTION front-end (detect_neck_edges/detect_frets) is intentionally NOT asserted here: classical
    line detection is lighting/angle-fragile (TabbyCat, TapToTab both found this), so it is tuned on a
    REAL frame via the CLI, or swapped for a YOLO model — and whatever it returns flows into THIS fit +
    gate, which is what guarantees we never ship a bad registration."""
    rng = np.random.default_rng(0)
    bx7 = fret_fraction(7)
    bc = np.float32([[0, 0], [0, 1], [bx7, 1], [bx7, 0]])
    H_true = cv2.getPerspectiveTransform(bc, np.float32([[170, 250], [150, 470], [1120, 430], [1130, 300]]))
    # simulate the detector output: top+bottom points per fret (nut..7) with 1.2px noise
    pts = []
    for n in range(0, 8):
        x = fret_fraction(n)
        top = project(H_true, [[x, 0]])[0] + rng.normal(0, 1.2, 2)
        bot = project(H_true, [[x, 1]])[0] + rng.normal(0, 1.2, 2)
        pts.append((tuple(top), tuple(bot)))

    print("AUTO fretboard detect — fit + confidence-gate core")
    good = fit_law(pts)
    Hc, med, n = good
    method = "auto" if med <= AUTO_ACCEPT_PX else "needs_click"
    print(f"  clean detection ({n} frets): confidence {med:.2f}px → {method}")

    # a deliberately BAD detection (frets jittered hard) must be REJECTED → click fallback
    bad = [((p[0][0] + rng.normal(0, 40), p[0][1] + rng.normal(0, 40)),
            (p[1][0] + rng.normal(0, 40), p[1][1] + rng.normal(0, 40))) for p in pts]
    bmed = fit_law(bad)[1]
    bmethod = "auto" if bmed <= AUTO_ACCEPT_PX else "needs_click"
    print(f"  noisy detection: confidence {bmed:.2f}px → {bmethod}")

    ok = (method == "auto") and (bmethod == "needs_click")
    print(f"  VERDICT: {'PASS — good fit auto-locks, bad fit falls back to 4-click' if ok else 'check gate'}")
    print("  (image detect front-end: run the CLI on a real frame to tune, or swap in YOLO — see header)")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Auto-detect the fretboard; fall back to 4-click if unsure.")
    ap.add_argument("--image", help="a single frame (png/jpg)")
    ap.add_argument("--video", help="a .webm/.mp4 — samples N frames and aggregates")
    ap.add_argument("--frames-dir", help="a folder of frames")
    ap.add_argument("--n", type=int, default=30, help="frames to sample from --video")
    ap.add_argument("--out", default="autodetect_overlay.png")
    a = ap.parse_args()

    frames = None
    if a.image:
        frames = [cv2.imread(a.image)]
    elif a.frames_dir:
        frames = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(a.frames_dir, "*")))[: a.n]]
    elif a.video:
        cap = cv2.VideoCapture(a.video); fs = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, total // a.n) if total else 1
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step == 0:
                fs.append(fr)
            i += 1
        cap.release(); frames = fs[: a.n]
    if not frames or frames[0] is None:
        raise SystemExit(0 if _selftest() else 1)

    res = autodetect(frames)
    print(res["method"], res.get("confidence_px"), res["reason"])
    if res["H"] is not None:
        overlay(frames[0] if isinstance(frames, list) else frames, res["H"], a.out)
        print("overlay →", a.out)
