#!/usr/bin/env python3
"""
Open-vocabulary fretboard / guitar-neck detector served over WebSocket.

WHY this design:
  Prior auto-detectors in this repo failed on a busy hackathon background:
    * OpenCV Hough (fretboard_detect.py) — edges/lines drown in clutter.
    * MediaPipe hands — needs the hand on the neck and a clean scene.
  Ultralytics YOLO-World is open-vocabulary: we give it the TEXT prompt
  "guitar neck"/"acoustic guitar" and it localizes the instrument with no
  training and no markers, robust to background clutter. We then refine the
  YOLO box into a 4-corner quad of the *neck* using a foreground mask that
  rejects skin (the fretting hand), so the wooden neck + strings dominate.

Pipeline per frame:
  YOLO-World box (most confident, whole guitar) -> SAM2 instance mask prompted
  by that box -> isolate the NECK from the whole-guitar mask (distance-transform
  body/neck split + march perpendicular widths along the body->headstock axis)
  -> minAreaRect on the neck pixels -> 4 corners (full-frame px) -> canonical
  ordering along the neck's long axis -> normalized [0..1] quad + YOLO confidence.

WHY a mask (not just the box): YOLO-World returns ONE box around the WHOLE
guitar (body + neck), so the box-derived quad covers the whole instrument. The
consumer maps a 6-string x 7-fret grid that must sit on the NECK only, so we
segment the guitar (SAM2) and extract just the thin fretboard from the mask.

Both models load ONCE at startup (never per frame): YOLO-World + CLIP text
classes, and SAM2-tiny. If SAM2 or neck extraction fails on a frame we fall
back to the previous whole-box quad (never crash, never return garbage).

Run as a server:
    .venv-yolo/bin/python fretboard_server.py            # ws://127.0.0.1:8772

Self-test on an image (headless, no camera):
    .venv-yolo/bin/python fretboard_server.py --selftest path/to/guitar.jpg
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

# MPS (Apple Silicon) can hit unimplemented ops; fall back to CPU for those
# instead of crashing. Must be set before torch / ultralytics import.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

import cv2

# The 12-TET fret law (fret_fraction(n) = 1 - 2^(-n/12)). Shared with fretboard.py /
# fretboard_detect.py so the silver-fret-wire grid uses the SAME physical law the
# consumer's nut->fret7 grid does. Import is cheap (pure numpy) and never per-frame.
try:
    from fretboard import fret_fraction
except Exception:  # keep the detector runnable even if the sibling import fails
    def fret_fraction(n: int) -> float:
        return 1.0 - 2.0 ** (-n / 12.0)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WS_HOST = "127.0.0.1"   # explicit IPv4 loopback
WS_PORT = 8772          # moved off 8770: macOS sharingd holds a dual-stack *:8770 listener (v4+v6)
# yolov8m-worldv2 (~55 MB) replaces yolov8x (~146 MB): on the two test images it
# returns near-identical guitar boxes (the box is all SAM2 needs) at ~26ms vs
# ~73ms/frame. SAM2 keys on the box, not the logits, so the smaller detector
# costs us nothing downstream while roughly halving the YOLO cost.
MODEL_NAME = "yolov8m-worldv2.pt"          # auto-downloads (~55 MB) on first load
SAM_NAME = "sam2_t.pt"                      # auto-downloads (~74 MB) on first load
CLASSES = ["guitar neck", "guitar fretboard", "acoustic guitar", "guitar"]
CONF_THRESH = 0.20

# --- Speed: SAM2 cadence (see detect_quad) ---
# SAM2 is ~216ms/frame and dominates latency; YOLO (~26ms) + neck-derive (<1ms)
# are cheap. So run SAM2 only every SAM_EVERY-th frame (or when the YOLO box
# jumps); between those, reproject the cached neck quad (stored RELATIVE to the
# YOLO box) onto the current frame's box. The box tracks motion per frame; the
# neck's pose within the box is stable, so this stays accurate between SAM2 runs.
SAM_EVERY = 4
# Re-run SAM2 early (ignore the cadence) when the YOLO box center moves more than
# this fraction of the box diagonal, or its size changes by more than SAM_BOX_*:
# a big box jump means a re-acquire / large motion where the cached relative
# neck geometry may no longer hold.
SAM_BOX_MOVE_FRAC = 0.20
SAM_BOX_SCALE_LO = 0.75
SAM_BOX_SCALE_HI = 1.33

# --- Temporal stability: EMA smoothing of the emitted quad (see _smooth_quad) ---
# A new quad "close" to the last one is jitter -> blend heavily toward the last
# (NEAR_BLEND weight on the NEW quad). A "far" quad is a real move/re-acquire ->
# snap (FAR_BLEND weight on the new quad) so we stay responsive.
SMOOTH_NEAR_BLEND = 0.4     # smoothed = 0.6*last + 0.4*new  (jitter rejection)
SMOOTH_FAR_BLEND = 0.8      # smoothed = 0.2*last + 0.8*new  (snap on big move)
# "close" test: centroid move < this fraction of the frame diagonal AND area
# ratio within [AREA_LO, AREA_HI]. Outside either -> treat as a real move.
SMOOTH_MOVE_FRAC = 0.08
SMOOTH_AREA_LO = 0.7
SMOOTH_AREA_HI = 1.4
# On a frame with NO detection, re-emit the last smoothed quad (confidence decayed
# by MISS_CONF_DECAY each miss) for up to MISS_HOLD frames before emitting null.
# Avoids the overlay vanishing on a single dropped detection.
MISS_HOLD = 5
MISS_CONF_DECAY = 0.8

# --- Neck-extraction tuning (see _neck_quad_from_mask) ---
# Distance-transform fraction: pixels with dist-to-edge > DT_FRAC * max are the
# thick "body core"; the rest (thin) is neck/headstock. The body is wide/round
# so it survives; the neck is thin so it doesn't.
DT_FRAC = 0.32
# Number of bins marched along the body->headstock axis.
NECK_BINS = 40
# A bin counts as "neck" if its perpendicular width < NECK_WIDTH_FRAC * max width.
NECK_WIDTH_FRAC = 0.30
# Bridge up to this many missing/occluded bins (the fretting hand) within the
# fretboard run...
NECK_MAX_GAP = 3
# ...but only resume across a gap if the width stays within this band of the
# running fretboard median (rejects the headstock: tuner pegs flare wider, the
# foreshortened head collapses narrower).
NECK_BAND_LO = 0.5
NECK_BAND_HI = 2.0

# --- Top-edge rectangle fit (see _rect_quad_from_band) ---
# The neck quad's ROTATION comes from a LINE fit to the fretboard's top long edge
# (the straight binding above the low-E), not from minAreaRect of the whole band:
# minAreaRect's angle wobbles on busy/occluded frames, but the top binding line is
# the strongest, most reliable anchor and gives a true rotated rectangle (parallel
# long edges, no shear / perspective). Robust line fit (cv2.fitLine, DIST_HUBER)
# resists the fretting-hand / clutter outliers on the band boundary.
# DIST type for the robust line fit of each long edge.
EDGE_FIT_DIST = cv2.DIST_HUBER
# Boundary points within this fraction of the half-thickness of the long axis are
# treated as "near the centre" and dropped from the side split, so the nut/end
# caps (which curve across both sides) don't pollute either long-edge fit.
EDGE_SIDE_DEADBAND = 0.15
# The rectangle's top boundary + WIDTH come from the band's perpendicular extent
# (distance along the perpendicular to the fitted top edge). Use a robust HIGH
# percentile for the bottom and (100 - this) for the top, so a few stray pixels
# above the binding or below the low strings don't blow the width up/down; the
# fretboard is near-constant width.
NECK_WIDTH_PCTL_HI = 98.0
# Along-neck ends: trim to this/(100-this) percentile of the along-axis projection
# so a few along-axis straggler pixels don't over-extend the rectangle past the
# visible fretboard, while keeping essentially the full nut..far-fret run.
ALONG_END_PCTL = 1.0
# Need at least this many boundary points on EACH long side for a trustworthy fit;
# below it we fall back to minAreaRect so we never emit a garbage rectangle.
EDGE_MIN_SIDE_PTS = 8
# Rotation guard (safety net, deliberately LOOSE): on the test images the robust
# top-edge fit matches the true neck tilt (from full neck+headstock PCA) to ~1-2
# deg and beats the march axis, so we trust the fit. We only fall back to the march
# along-axis if the fit is GROSSLY off (a degenerate / nearly-square band where the
# side-split collapses), which would otherwise emit a wildly mis-rotated rect.
EDGE_ANGLE_GUARD_DEG = 35.0

# --- Silver fret-wire detection -> 12-TET law -> real nut..fret7 (see
#     _detect_fret_wires / _fit_fret_law_1d) ---
# The top-edge rect's ALONG extent is the SAM2 band (mid-neck, truncated by the
# fretting hand), NOT the real nut. The silver fret WIRES are a strong high-contrast
# cue: we rectify the band so frets become vertical, pop the bright thin vertical
# wires, peak-find their along-neck positions, fit the 12-TET law, and EXTRAPOLATE
# the (occluded) nut + fret 7 to re-anchor the quad. Tuned on the realtest frames.
FRET_DETECT = True              # master switch; False -> behave exactly as before
# Rectified-ROI sampling: extend the band this fraction of its own along-length past
# EACH end so an extrapolated nut/fret-7 that falls just outside the band is drawable.
FRET_RECT_PAD_ALONG = 0.6
FRET_RECT_PAD_PERP = 0.08
# Profile only the top+bottom fractions of the neck height: fret WIRES span the full
# height; inlay DOTS sit in the centre, so skipping the centre rejects the dots.
FRET_EDGE_BAND = 0.32           # use rows in [0,0.32) and (0.68,1] of the rect height
# A rectified column must have at least this fraction of its edge-rows inside the
# fretboard band to contribute a profile value (drops the occluded / off-board cols).
FRET_MIN_COL_COV = 0.18
# Peak finder on the smoothed along-neck bright-line profile.
FRET_PEAK_PROMINENCE = 0.30     # x the profile's nonzero std
FRET_PEAK_MIN_DIST_FRAC = 1.0 / 80.0   # x rectified width
FRET_PROFILE_SMOOTH = 7         # 1-D Gaussian width (odd) on the column profile
# 12-TET law fit acceptance: need >= this many UNIQUE consecutive fret inliers and a
# spacing residual (normalized by scale) below the threshold, else fall back to the
# band-extent quad (never worse than before).
FRET_MIN_INLIERS = 5
FRET_MAX_FRET = 14              # search frets 0..this in the law fit
FRET_FIT_TOL_FRAC = 0.05        # inlier tolerance as a fraction of the peak span
FRET_MAX_RESID_FRAC = 0.020     # max mean residual / scale to TRUST the law fit
# Confidence blend: final conf = YOLO_conf * (FRET_CONF_BASE + (1-base)*fit_quality),
# where fit_quality in [0,1] grows with inliers and shrinks with residual. So a clean
# fret fit nudges confidence up; a weak one barely changes the YOLO confidence.
FRET_CONF_BASE = 0.85

# Skin HSV range (the fretting hand) — used only by the legacy box-refine
# fallback when SAM2 / neck extraction is unavailable.
SKIN_LO = np.array([0, 30, 60], dtype=np.uint8)
SKIN_HI = np.array([25, 180, 255], dtype=np.uint8)

# Lazily-initialized singletons (loaded once).
_MODEL = None
_SAM = None
_DEVICE = "mps"

# --- Per-stream mutable state (module-level; reset via reset_state()) ---
# SAM2 cadence cache: the YOLO box SAM2 last ran on, the neck quad it produced
# expressed in that box's local frame (so we can reproject it onto a new box),
# and how many frames since SAM2 last ran.
_LAST_SAM_BOX = None        # (x1, y1, x2, y2) the cached neck quad is relative to
_CACHED_NECK_REL = None     # (4,2) neck corners in normalized box-local coords
_FRAMES_SINCE_SAM = 0
# The silver-fret `frets` payload from the last SAM2 frame, cached so the cheap
# cache path (between SAM2 runs) can re-emit it: the u values are relative to the
# quad's nut->fret7 axis, which the cached quad preserves as it reprojects. None
# when the last SAM2 frame had no trustworthy fret fit.
_CACHED_FRETS = None        # list[{"n": int, "u": float}] or None
# Temporal smoothing state: the last quad we EMITTED (normalized px, full frame),
# its confidence, and the count of consecutive no-detection frames.
_LAST_EMIT_QUAD = None      # (4,2) float, normalized [0..1], canonical order
_LAST_EMIT_CONF = 0.0
_MISS_COUNT = 0


def reset_state():
    """Clear all per-stream state (SAM2 cache + smoothing history).

    Called when a WS client connects so a new session starts clean, and exposed
    for the self-test / unit tests so each starts from a known state.
    """
    global _LAST_SAM_BOX, _CACHED_NECK_REL, _FRAMES_SINCE_SAM, _CACHED_FRETS
    global _LAST_EMIT_QUAD, _LAST_EMIT_CONF, _MISS_COUNT
    _LAST_SAM_BOX = None
    _CACHED_NECK_REL = None
    _FRAMES_SINCE_SAM = 0
    _CACHED_FRETS = None
    _LAST_EMIT_QUAD = None
    _LAST_EMIT_CONF = 0.0
    _MISS_COUNT = 0


# ---------------------------------------------------------------------------
# Model loading (ONCE)
# ---------------------------------------------------------------------------
def load_model():
    """Load YOLO-World once and set the open-vocabulary classes once.

    set_classes() builds CLIP text embeddings and is slow — it MUST NOT run
    per frame. We pick the inference device here too, with a CPU fallback if
    MPS isn't usable.
    """
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL

    from ultralytics import YOLOWorld
    import torch

    if torch.backends.mps.is_available():
        _DEVICE = "mps"
    elif torch.cuda.is_available():
        _DEVICE = "cuda"
    else:
        _DEVICE = "cpu"

    print(f"[init] loading {MODEL_NAME} (auto-downloads ~146MB on first run)...",
          flush=True)
    t0 = time.time()
    model = YOLOWorld(MODEL_NAME)
    print(f"[init] setting open-vocab classes {CLASSES} (loads CLIP, slow)...",
          flush=True)
    model.set_classes(CLASSES)
    _MODEL = model
    print(f"[init] ready on device={_DEVICE} in {time.time() - t0:.1f}s",
          flush=True)

    # Load SAM2-tiny once too (segments the guitar so we can isolate the neck).
    # If it fails, we keep running and fall back to the box-derived quad.
    global _SAM
    try:
        from ultralytics import SAM
        print(f"[init] loading {SAM_NAME} (auto-downloads ~74MB on first run)...",
              flush=True)
        ts = time.time()
        _SAM = SAM(SAM_NAME)
        print(f"[init] SAM2 ready in {time.time() - ts:.1f}s", flush=True)
    except Exception as exc:
        _SAM = None
        print(f"[init] SAM2 unavailable ({exc!r}); will use box-only fallback",
              flush=True)
    return _MODEL


# ---------------------------------------------------------------------------
# Quad geometry
# ---------------------------------------------------------------------------
def _order_quad_along_long_axis(box_pts: np.ndarray) -> np.ndarray:
    """Order 4 rotated-rect corners canonically along the neck's LONG axis.

    The longer side of the rect is the neck "along" axis (u). We return:
      [along-start . sideA, along-end . sideA, along-end . sideB, along-start . sideB]
    i.e. walk the two long edges so the quad traces a consistent ring.

    box_pts: (4,2) float, the cv2.boxPoints output (ordered around the rect).
    """
    p = box_pts.astype(np.float64)
    # Edge lengths between consecutive corners (boxPoints is ordered around rect).
    e = [np.linalg.norm(p[(i + 1) % 4] - p[i]) for i in range(4)]
    # Edges 0 and 2 are one pair of opposite sides; 1 and 3 the other pair.
    pair01_len = e[0] + e[2]
    pair12_len = e[1] + e[3]

    if pair01_len >= pair12_len:
        # Long edges are p0->p1 and p2->p3 (sides A and B run along these).
        # sideA long edge: p0 (start) -> p1 (end)
        # sideB long edge: p3 (start) -> p2 (end)
        along_start_A, along_end_A = p[0], p[1]
        along_start_B, along_end_B = p[3], p[2]
    else:
        # Long edges are p1->p2 and p3->p0.
        along_start_A, along_end_A = p[1], p[2]
        along_start_B, along_end_B = p[0], p[3]

    return np.array([along_start_A, along_end_A, along_end_B, along_start_B],
                    dtype=np.float64)


def _box_corners(x1, y1, x2, y2) -> np.ndarray:
    """Axis-aligned box -> 4 corners as a rotated-rect-style ring (TL,TR,BR,BL)."""
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


# ---------------------------------------------------------------------------
# SAM2-cadence cache: encode a neck quad RELATIVE to its YOLO box, then decode
# it onto a later box so we can skip SAM2 between runs (see detect_quad).
# ---------------------------------------------------------------------------
def _quad_to_box_rel(quad_px, box):
    """Express a full-frame neck quad in its YOLO box's local coords ([0..1] per
    axis, where 0 = box left/top, 1 = box right/bottom). Lets us reproject the
    same neck pose onto a moved/resized box without rerunning SAM2."""
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1e-6)
    bh = max(y2 - y1, 1e-6)
    rel = np.array(quad_px, dtype=np.float64).copy()
    rel[:, 0] = (rel[:, 0] - x1) / bw
    rel[:, 1] = (rel[:, 1] - y1) / bh
    return rel


def _box_rel_to_quad(rel, box):
    """Inverse of _quad_to_box_rel: box-local neck quad -> full-frame px."""
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    out = np.array(rel, dtype=np.float64).copy()
    out[:, 0] = out[:, 0] * bw + x1
    out[:, 1] = out[:, 1] * bh + y1
    return out


def _box_jumped(prev_box, new_box):
    """True if the YOLO box moved/resized enough that the cached relative neck
    geometry is no longer trustworthy (forces an early SAM2 re-run)."""
    if prev_box is None:
        return True
    px1, py1, px2, py2 = prev_box
    nx1, ny1, nx2, ny2 = new_box
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    ncx, ncy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
    pw, ph = max(px2 - px1, 1e-6), max(py2 - py1, 1e-6)
    pdiag = float(np.hypot(pw, ph))
    move = float(np.hypot(ncx - pcx, ncy - pcy))
    if move > SAM_BOX_MOVE_FRAC * pdiag:
        return True
    nw, nh = max(nx2 - nx1, 1e-6), max(ny2 - ny1, 1e-6)
    sx, sy = nw / pw, nh / ph
    if not (SAM_BOX_SCALE_LO <= sx <= SAM_BOX_SCALE_HI):
        return True
    if not (SAM_BOX_SCALE_LO <= sy <= SAM_BOX_SCALE_HI):
        return True
    return False


# ---------------------------------------------------------------------------
# Temporal smoothing: align corners across frames, then EMA-blend so the
# emitted quad doesn't jitter (see _smooth_quad / detect_quad).
# ---------------------------------------------------------------------------
def _quad_centroid(quad):
    return np.array(quad, dtype=np.float64).mean(axis=0)


def _quad_area(quad):
    """Shoelace area of a 4-point ring (sign-agnostic)."""
    p = np.array(quad, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _align_corners(new_quad, ref_quad):
    """Reorder new_quad's 4 corners to best match ref_quad's ordering.

    Our quads are a ring ordered along the neck's long axis, but which physical
    corner lands at index 0 can flip between frames (the long-axis ordering picks
    a start arbitrarily, and a near-180-deg neck flip swaps ends). Averaging
    mismatched corners would smear the quad. We try both ring rotations that
    preserve winding (offset 0 and 2) plus, for each, the reversed winding, and
    keep whichever permutation minimizes total corner distance to ref."""
    new = np.array(new_quad, dtype=np.float64)
    ref = np.array(ref_quad, dtype=np.float64)
    best, best_cost = new, float("inf")
    for rev in (new, new[::-1]):
        for off in range(4):
            cand = np.roll(rev, -off, axis=0)
            cost = float(np.sum((cand - ref) ** 2))
            if cost < best_cost:
                best_cost = cost
                best = cand
    return best


def _smooth_quad(new_quad, conf):
    """EMA-blend a freshly detected normalized quad into the last emitted one and
    return (smoothed_quad, smoothed_conf). Updates module state.

    - First detection (no history): emit as-is.
    - Corners are aligned to the previous quad before blending so we never average
      mismatched corners.
    - "Close" to last (centroid move < SMOOTH_MOVE_FRAC of the frame diagonal AND
      area ratio in [SMOOTH_AREA_LO, SMOOTH_AREA_HI]) -> heavy smoothing
      (SMOOTH_NEAR_BLEND on the new quad): kills frame-to-frame jitter.
    - "Far" (a real move / re-acquire) -> snap (SMOOTH_FAR_BLEND on the new quad).
    """
    global _LAST_EMIT_QUAD, _LAST_EMIT_CONF, _MISS_COUNT
    new = np.array(new_quad, dtype=np.float64)

    if _LAST_EMIT_QUAD is None:
        _LAST_EMIT_QUAD = new
        _LAST_EMIT_CONF = conf
        _MISS_COUNT = 0
        return new, conf

    last = _LAST_EMIT_QUAD
    aligned = _align_corners(new, last)

    # Closeness test in normalized coords (frame diagonal in [0..1] space = sqrt2).
    move = float(np.linalg.norm(_quad_centroid(aligned) - _quad_centroid(last)))
    a_last = _quad_area(last)
    a_new = _quad_area(aligned)
    area_ratio = (a_new / a_last) if a_last > 1e-9 else 999.0
    is_close = (move < SMOOTH_MOVE_FRAC * np.sqrt(2.0)
                and SMOOTH_AREA_LO <= area_ratio <= SMOOTH_AREA_HI)

    blend = SMOOTH_NEAR_BLEND if is_close else SMOOTH_FAR_BLEND
    smoothed = (1.0 - blend) * last + blend * aligned

    _LAST_EMIT_QUAD = smoothed
    _LAST_EMIT_CONF = conf
    _MISS_COUNT = 0
    return smoothed, conf


def _hold_or_null():
    """Called when a frame yields no detection. Re-emit the last smoothed quad
    with a decayed confidence for up to MISS_HOLD frames, then give up (None).
    Returns (quad_list, conf) to emit, or None to emit a null quad."""
    global _LAST_EMIT_CONF, _MISS_COUNT
    if _LAST_EMIT_QUAD is None or _MISS_COUNT >= MISS_HOLD:
        return None
    _MISS_COUNT += 1
    _LAST_EMIT_CONF *= MISS_CONF_DECAY
    return _LAST_EMIT_QUAD, _LAST_EMIT_CONF


# ---------------------------------------------------------------------------
# Guitar segmentation (SAM2) + neck isolation from the whole-guitar mask
# ---------------------------------------------------------------------------
def _sam_guitar_mask(bgr_frame, box_xyxy):
    """Segment the guitar with SAM2, prompted by the YOLO-World box.

    Returns a full-frame uint8 mask (0/255) of the largest returned instance,
    or None if SAM2 is unavailable or produced nothing.
    """
    if _SAM is None:
        return None
    try:
        res = _SAM.predict(bgr_frame, bboxes=[list(box_xyxy)], device=_DEVICE,
                           verbose=False)
    except Exception as exc:
        print(f"[detect] SAM2 predict failed: {exc!r}", flush=True)
        return None
    if not res or res[0].masks is None or len(res[0].masks.data) == 0:
        return None
    m = res[0].masks.data.cpu().numpy()          # (n, H, W) in [0,1]
    areas = m.reshape(m.shape[0], -1).sum(1)
    mask = (m[int(np.argmax(areas))] > 0.5).astype(np.uint8) * 255
    h, w = bgr_frame.shape[:2]
    if mask.shape[:2] != (h, w):                  # SAM may return model-res mask
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def _fretboard_run(norm_w):
    """Given per-bin normalized perpendicular widths (marched body->headstock),
    return the (start, end) bin indices of the FRETBOARD: the narrow,
    roughly-constant-width run that begins where the body ends.

    We start at the first narrow bin (leaving the wide body shoulder) and walk
    toward the headstock through narrow bins. The fretting hand leaves a few
    missing bins, which we bridge (up to NECK_MAX_GAP) ONLY if the width resumes
    consistent with the fretboard (within [NECK_BAND_LO, NECK_BAND_HI] x the
    running median). The headstock breaks that consistency (pegs flare wider or
    the foreshortened head collapses narrower), which ends the run.
    """
    n = len(norm_w)
    narrow = (norm_w > 0.001) & (norm_w < NECK_WIDTH_FRAC)
    i = 0
    while i < n and not narrow[i]:
        i += 1
    if i >= n:
        return None
    start = end = i
    seen = [float(norm_w[i])]
    gap = 0
    i += 1
    while i < n:
        if narrow[i]:
            med = float(np.median(seen))
            if gap > 0 and med > 0 and not (NECK_BAND_LO * med <= norm_w[i]
                                            <= NECK_BAND_HI * med):
                break
            seen.append(float(norm_w[i]))
            end = i
            gap = 0
        else:
            gap += 1
            if gap > NECK_MAX_GAP:
                break
        i += 1
    return (start, end)


def _fit_line_dir(pts):
    """Robust line fit (cv2.fitLine, DIST_HUBER) -> (point_on_line, unit_dir).

    pts: (N,2) float. Returns ((px,py), (dx,dy)) with (dx,dy) a unit vector, or
    None if the fit is degenerate. Huber down-weights the outliers the fretting
    hand / background clutter leave on a long-edge point set."""
    if pts is None or len(pts) < 2:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), EDGE_FIT_DIST,
                                 0, 0.01, 0.01).ravel()
    d = np.array([float(vx), float(vy)], dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None
    return np.array([float(x0), float(y0)], dtype=np.float64), d / n


def _rect_quad_from_band(neck_pts, axis, perp):
    """Build a TRUE rotated rectangle for the fretboard from its band pixels,
    anchored on the fretboard's TOP long edge.

    WHY (user insight): the fretboard's top long edge — the straight binding line
    above the thick low-E string — is the strongest, most reliable anchor on hard
    live frames. minAreaRect of the whole band takes its angle from ALL the mask
    pixels, so a busy background / partial occlusion / the fretting hand wobble or
    mis-rotate it. Fitting a robust LINE to just the top edge and forcing the
    bottom edge PARALLEL to it gives a stable rotation and a clean rectangle with
    zero shear / perspective (the required pose: in-plane tilt + scale only).

    neck_pts: (N,2) float32 pixels of the fretboard band (largest CC).
    axis, perp: unit long-axis and its perpendicular (initial guess from the body
        ->headstock march; only used to split the boundary into two long sides and
        to pick along-neck ends).
    Returns (4,2) float64 corners in cv2.boxPoints-style ring order, or None to
    let the caller fall back to minAreaRect.
    """
    if neck_pts is None or len(neck_pts) < EDGE_MIN_SIDE_PTS * 2:
        return None
    pts = neck_pts.astype(np.float64)

    # Boundary only: the two long edges live on the band's contour, and fitting
    # the contour (not the filled interior) keeps the line fit on the real edge.
    h = int(pts[:, 1].max()) + 2
    w = int(pts[:, 0].max()) + 2
    canvas = np.zeros((h, w), np.uint8)
    canvas[pts[:, 1].astype(np.int32), pts[:, 0].astype(np.int32)] = 255
    cnts, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        bnd = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    else:
        bnd = pts  # degenerate; fit on all pixels rather than crash

    c = bnd.mean(axis=0)
    rel = bnd - c
    s = rel @ perp                       # signed perpendicular offset of each pt
    half = float(np.abs(s).max())
    if half < 1e-6:
        return None
    dead = EDGE_SIDE_DEADBAND * half     # drop the end-cap points near the centre

    # Split the boundary into the two long sides by the SIGN of the perpendicular
    # offset (relative to the long axis). Smaller image-y == nearer the top.
    side_pos = bnd[s > dead]             # one long side
    side_neg = bnd[s < -dead]            # the other long side
    if len(side_pos) < EDGE_MIN_SIDE_PTS or len(side_neg) < EDGE_MIN_SIDE_PTS:
        return None
    # TOP side = the long side nearer the top of the image (smaller mean y): the
    # low-E-side binding, the user's anchor. We fit only this side; the bottom edge
    # is then built parallel to it (so the opposite side need not be fit).
    top_side = (side_pos if side_pos[:, 1].mean() <= side_neg[:, 1].mean()
                else side_neg)

    # ROTATION = the TOP edge's robustly-fitted line DIRECTION. This is the whole
    # point: the angle comes from a Huber line fit to the top binding alone, so a
    # busy background / the fretting hand on the OTHER side can't twist it.
    top_fit = _fit_line_dir(top_side)
    if top_fit is None:
        return None
    p_fit, u = top_fit                   # u: unit direction ALONG the top edge
    # Align u's SENSE with the march along-axis (the global body->headstock
    # direction) so "start/end" and the perpendicular below are consistent.
    a = axis / (np.linalg.norm(axis) + 1e-12)
    if u @ a < 0:
        u = -u
    # Rotation-sanity GUARD: the march axis is the physically-correct neck
    # direction derived from the WHOLE-guitar shape; the top-edge fit refines it
    # locally. On a foreshortened / nearly-square band the local fit is ambiguous
    # and can swing wildly, so if it disagrees with the march axis by more than
    # EDGE_ANGLE_GUARD_DEG we keep the march axis (a stable, correct fallback)
    # rather than emit a mis-rotated rectangle. Parallel edges are preserved either
    # way (the bottom edge is always built parallel to u).
    cos_dev = float(np.clip(u @ a, -1.0, 1.0))
    if float(np.degrees(np.arccos(cos_dev))) > EDGE_ANGLE_GUARD_DEG:
        u = a
    # Perpendicular to the (possibly guarded) top edge, pointing from the top edge
    # toward the band interior (so positive depth == "into the fretboard"). We
    # measure width along THIS so it's exactly perpendicular to the edge we anchor.
    n_hat = np.array([-u[1], u[0]], dtype=np.float64)
    if n_hat @ (c - p_fit) < 0:
        n_hat = -n_hat

    # ANCHOR + WIDTH from ALL band pixels projected onto n_hat (perp to the top
    # edge). The line fit's point p_fit sits mid-cloud, so anchoring/measuring from
    # it undershoots; instead take the band's true perpendicular extent. Robust
    # percentiles (not min/max) reject a handful of stray pixels above the binding
    # or below the low strings. d_lo == the true TOP boundary; the span is the full
    # fretboard thickness -> the bottom edge is parallel, offset by that width
    # (constant), so zero shear / perspective.
    dproj = (pts - p_fit) @ n_hat
    d_lo = float(np.percentile(dproj, 100.0 - NECK_WIDTH_PCTL_HI))  # toward top
    d_hi = float(np.percentile(dproj, NECK_WIDTH_PCTL_HI))          # toward bottom
    width = d_hi - d_lo
    if width < 1.0:
        return None
    # Slide the fitted line out to the real top boundary (keep its direction/angle).
    p_top = p_fit + d_lo * n_hat

    # ALONG-NECK ends: project the band pixels onto the top edge's own direction u
    # and clip to their visible extent (nut .. far fret). Robust percentiles trim a
    # few along-axis stragglers without shortening the real fretboard run.
    tproj = (pts - p_top) @ u
    t0 = float(np.percentile(tproj, ALONG_END_PCTL))
    t1 = float(np.percentile(tproj, 100.0 - ALONG_END_PCTL))
    if t1 - t0 < 1.0:
        return None

    # Four corners of the true rectangle (ring order, matching boxPoints):
    #   top edge  : p_top + t*u           (start, end)
    #   bottom edge: same, shifted by +width along n_hat  (PARALLEL to top)
    a0 = p_top + t0 * u
    a1 = p_top + t1 * u
    b1 = a1 + width * n_hat
    b0 = a0 + width * n_hat
    return np.array([a0, a1, b1, b0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Silver fret-wire detection -> 12-TET law fit -> real nut..fret7 anchoring.
# The top-edge rect gives a stable rotation + string span but its ALONG extent is
# the SAM2 band (truncated by the fretting hand), not the nut. We rectify that band,
# pop the bright silver wires, fit the fret LAW, and extrapolate the nut + fret 7.
# ---------------------------------------------------------------------------
def _fit_fret_law_1d(peaks, max_fret=FRET_MAX_FRET, tol_frac=FRET_FIT_TOL_FRAC,
                     min_inliers=FRET_MIN_INLIERS):
    """Fit the 12-TET law to 1-D rectified fret-x peaks.

    Model: x(n) = x0 + scale * sign * fret_fraction(n), where the detected peaks are
    a (mostly) CONSECUTIVE run of frets (a few missed/spurious tolerated). This is the
    1-D analogue of fretboard_detect.fit_law (try nut offsets, assign consecutive fret
    indices, keep the lowest-residual assignment) made robust to the orientation and
    to occlusion: we try BOTH signs and BOTH anchor-label orientations (which end is
    the nut is exactly what we solve for) and score by inlier count minus a phantom-gap
    penalty, with the spacing residual as the tiebreak.

    peaks: iterable of rectified along-neck x positions of detected fret wires.
    Returns dict(x0, scale, sign, nut_x, fret7_x, frets=[(n, x)...], inliers, resid,
    gaps, fret_span, quality in [0,1]) or None if no usable fit.
    """
    p = np.array(sorted({round(float(x), 1) for x in peaks}), dtype=np.float64)
    if len(p) < min_inliers:
        return None
    span = float(p.max() - p.min())
    if span < 10.0:
        return None
    tol = max(5.0, tol_frac * span)
    fr = np.array([fret_fraction(n) for n in range(max_fret + 1)], dtype=np.float64)
    n = len(p)
    best = None

    for ia in range(n):
        for ib in range(ia + 1, n):
            xa, xb = p[ia], p[ib]
            if abs(xb - xa) < 8.0:
                continue
            for sign in (+1.0, -1.0):
                for na in range(0, 4):
                    for nb in range(na + 1, na + 5):     # up to 3 skipped frets
                        # Both label orientations: the smaller fret may sit on the
                        # lower-x OR the higher-x anchor peak.
                        for la, lb in ((na, nb), (nb, na)):
                            denom = (fr[lb] - fr[la]) * sign
                            if abs(denom) < 1e-9:
                                continue
                            scale = (xb - xa) / denom
                            if not np.isfinite(scale) or scale <= 0:
                                continue
                            x0 = xa - scale * sign * fr[la]
                            pred = x0 + scale * sign * fr
                            d = np.abs(p[:, None] - pred[None, :])
                            j = d.argmin(1)
                            dmin = d[np.arange(n), j]
                            inl = dmin <= tol
                            if int(inl.sum()) < min_inliers:
                                continue
                            # one peak per fret number (closest wins)
                            bynum = {}
                            for k in range(n):
                                if not inl[k]:
                                    continue
                                num = int(j[k])
                                if num not in bynum or dmin[k] < bynum[num][1]:
                                    bynum[num] = (float(p[k]), float(dmin[k]))
                            nums = sorted(bynum)
                            if len(nums) < min_inliers:
                                continue
                            fret_span = nums[-1] - nums[0] + 1
                            gaps = fret_span - len(nums)
                            resid = float(np.mean([bynum[q][1] for q in nums]))
                            score = (len(nums) - 1.3 * gaps) - (resid / tol)
                            key = round(score, 4)
                            if best is None or key > best[0]:
                                best = (key, x0, scale, sign, bynum, nums,
                                        fret_span, gaps, resid)
    if best is None:
        return None
    _, x0, scale, sign, bynum, nums, fret_span, gaps, resid = best
    # Quality in [0,1]: rewards more unique inliers, punishes phantom gaps + residual.
    resid_frac = resid / max(scale, 1.0)
    quality = float(np.clip(
        (len(nums) / 8.0)
        - 0.10 * gaps
        - (resid_frac / FRET_MAX_RESID_FRAC) * 0.5,
        0.0, 1.0))
    return {
        "x0": x0, "scale": scale, "sign": sign,
        "nut_x": float(x0 + scale * sign * fret_fraction(0)),
        "fret7_x": float(x0 + scale * sign * fret_fraction(7)),
        "frets": [(int(q), bynum[q][0]) for q in nums],
        "inliers": len(nums), "resid": resid, "gaps": gaps,
        "fret_span": fret_span, "resid_frac": resid_frac, "quality": quality,
    }


def _detect_fret_wires(bgr_frame, neck_pts, band_mask, axis, perp, rect_quad):
    """Detect the silver fret wires inside the rectified neck band, fit the 12-TET
    law, and return everything needed to re-anchor the quad to the real nut..fret7.

    bgr_frame  : full BGR frame.
    neck_pts   : (N,2) float32 fretboard-band pixels (largest CC) — unused directly
                 but kept for parity / future per-pixel gating.
    band_mask  : full-frame uint8 (0/255) raster of that band (the spatial gate).
    axis, perp : body->headstock unit axis and its perpendicular (sense reference).
    rect_quad  : (4,2) [a0(top-start), a1(top-end), b1(bot-end), b0(bot-start)] from
                 _rect_quad_from_band — gives the rectifying frame (u along the top
                 edge, n_hat across the neck) and the string-span width.

    Returns dict(nut_pt, fret7_pt, along_u, frets=[(u_frac, n)...], fit) in FULL-FRAME
    px, where nut_pt/fret7_pt are the top-edge points (on the a-side) at the fitted
    nut and fret 7, along_u is the unit top-edge direction, and u_frac is each fret's
    position as a fraction of nut->fret7 (for the WS `frets` payload). None on failure
    (the caller then keeps the band-extent quad). Never raises.
    """
    if bgr_frame is None or rect_quad is None:
        return None
    a0, a1, b1, b0 = (np.asarray(rect_quad[i], np.float64) for i in range(4))
    u = a1 - a0
    along_len = float(np.linalg.norm(u))
    if along_len < 20.0:
        return None
    u = u / along_len
    nvec = b0 - a0
    width = float(np.linalg.norm(nvec))
    if width < 6.0:
        return None
    n_hat = nvec / width

    h, w = bgr_frame.shape[:2]
    pad_a = int(FRET_RECT_PAD_ALONG * along_len)
    pad_p = int(FRET_RECT_PAD_PERP * width)
    W_out = int(along_len + 2 * pad_a)
    H_out = int(width + 2 * pad_p)
    if W_out < 32 or H_out < 8 or W_out > 8000 or H_out > 2000:
        return None

    # Inverse map: rectified (xo,yo) -> source px = a0 + (xo-pad_a)*u + (yo-pad_p)*n_hat.
    xo, yo = np.meshgrid(np.arange(W_out, dtype=np.float32),
                         np.arange(H_out, dtype=np.float32))
    sx = (a0[0] + (xo - pad_a) * u[0] + (yo - pad_p) * n_hat[0]).astype(np.float32)
    sy = (a0[1] + (xo - pad_a) * u[1] + (yo - pad_p) * n_hat[1]).astype(np.float32)
    rect = cv2.remap(bgr_frame, sx, sy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    rband = cv2.remap(band_mask, sx, sy, cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Spatial gate = the rectified BAND (fretboard-only: the distance-transform split
    # already drops the body, and the fret-run gap-bridging drops most of the hand).
    # We deliberately DON'T color-mask the hand: a fixed skin HSV range eats ~98% of a
    # warm-lit brown fretboard (verified on chord_5s), and the 12-TET fit rejects the
    # few non-conforming peaks a finger edge leaves anyway.
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Pop bright, thin, near-vertical lines: |Sobel_x| (along-neck gradient energy)
    # spikes on each silver wire. Profiled over the top+bottom edge-rows only so the
    # central inlay dots don't register.
    sob = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    edge_rows = np.zeros(H_out, dtype=bool)
    edge_rows[: int(FRET_EDGE_BAND * H_out)] = True
    edge_rows[int((1.0 - FRET_EDGE_BAND) * H_out):] = True
    valid = (rband > 0) & edge_rows[:, None]
    sob[~valid] = 0.0
    cov = valid.sum(axis=0).astype(np.float32)
    denom_rows = max(int(edge_rows.sum()), 1)
    prof = sob.sum(axis=0) / np.maximum(cov, 1.0)
    prof[cov < FRET_MIN_COL_COV * denom_rows] = 0.0
    k = FRET_PROFILE_SMOOTH | 1
    prof = cv2.GaussianBlur(prof.reshape(1, -1), (k, 1), 0).ravel()

    nz = prof[prof > 0]
    if nz.size < 4:
        return None
    base = float(nz.std()) or 1.0
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(
            prof, distance=max(8, int(FRET_PEAK_MIN_DIST_FRAC * W_out)),
            prominence=base * FRET_PEAK_PROMINENCE)
    except Exception:
        thr = float(prof.mean() + base * FRET_PEAK_PROMINENCE)
        peaks = np.where((prof[1:-1] > prof[:-2]) & (prof[1:-1] > prof[2:])
                         & (prof[1:-1] > thr))[0] + 1
    if len(peaks) < FRET_MIN_INLIERS:
        return None

    fit = _fit_fret_law_1d([float(x) for x in peaks])
    if fit is None:
        return None

    # Map a rectified along-x back to full-frame px on the TOP edge (a-side): the
    # rectified frame has x=pad_a at a0 and advances by u per pixel.
    def _to_img_top(xr):
        return a0 + (float(xr) - pad_a) * u

    nut_pt = _to_img_top(fit["nut_x"])
    fret7_pt = _to_img_top(fit["fret7_x"])
    # Per-fret TOP-EDGE image points (full-frame px). The caller turns these into the
    # WS `frets` payload (normalized u along the FINAL emitted quad) so the u stays
    # consistent regardless of any later canonical corner re-ordering.
    fret_img_pts = [(int(nfr), _to_img_top(xr)) for nfr, xr in fit["frets"]]
    return {
        "nut_pt": nut_pt, "fret7_pt": fret7_pt, "along_u": u,
        "fret_img_pts": fret_img_pts, "fit": fit,
        # debug geometry for the selftest overlay:
        "_rect_origin": a0, "_pad_a": pad_a, "_W_out": W_out,
    }


def _neck_quad_from_mask(mask, bgr_frame=None):
    """Isolate the NECK from a whole-guitar mask and return (corners, fret_info).

    corners: 4 neck corners (full-frame px, cv2.boxPoints order) or None.
    fret_info: the _detect_fret_wires() dict if the silver-fret 12-TET fit succeeded
        AND re-anchored the quad to the real nut..fret7, else None. When fret_info is
        not None the returned corners ALREADY span nut..fret7 (along) with the rect's
        string-span width preserved; otherwise corners are the band-extent rect.

    Method (orientation-robust; the round body otherwise dominates a global
    PCA/minAreaRect axis):
      1. Distance transform -> the wide body core is the only thick region; the
         thin remainder is neck + headstock.
      2. Body centroid = centroid of the thick core. Neck tip = the thin pixel
         farthest from it (the headstock). The body->tip vector is the neck axis.
      3. March NECK_BINS bins along that axis; per bin measure the mask's
         perpendicular width (5..95 pct span). Pick the fretboard run.
      4. Rasterize the fretboard band's pixels, keep the largest connected
         component (drops body/headstock pixels that share an along-axis bin on
         a tilted guitar). Then anchor a TRUE rotated rectangle on the band's
         fitted TOP long edge (_rect_quad_from_band) -> 4 neck corners; if that
         robust fit can't be trusted, fall back to minAreaRect of the band.
      5. (NEW) If a frame is given and FRET_DETECT is on: detect the silver fret
         WIRES inside that band, fit the 12-TET law, and replace the along-extent
         with the extrapolated real nut..fret7 (keeping the rect's string span).
    """
    if mask is None:
        return None, None
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) < 200:
        return None, None

    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    maxd = float(dt.max())
    if maxd <= 0:
        return None, None

    # Thick body core, dilated back toward the full body so the rim is excluded
    # from the "thin" set as much as possible.
    rad = int(DT_FRAC * maxd * 2) | 1          # odd kernel size
    thick = (dt > DT_FRAC * maxd).astype(np.uint8)
    thick = cv2.dilate(thick, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (rad, rad)))
    thick = cv2.bitwise_and(thick, m)
    tys, txs = np.where(thick > 0)
    if len(txs) < 10:
        return None, None
    body_c = np.array([txs.mean(), tys.mean()], dtype=np.float64)

    thin = cv2.bitwise_and(m, cv2.bitwise_not(thick))
    nys, nxs = np.where(thin > 0)
    if len(nxs) < 30:
        return None, None
    thin_pts = np.column_stack([nxs, nys]).astype(np.float64)
    tip = thin_pts[int(np.argmax(((thin_pts - body_c) ** 2).sum(1)))]
    axis = tip - body_c
    nrm = np.linalg.norm(axis)
    if nrm < 1e-6:
        return None, None
    axis /= nrm
    perp = np.array([-axis[1], axis[0]], dtype=np.float64)

    # Project ALL guitar pixels for a robust per-bin width measurement.
    ays, axs_ = np.where(m > 0)
    all_pts = np.column_stack([axs_, ays]).astype(np.float64)
    rel = all_pts - body_c
    t = rel @ axis
    wco = rel @ perp
    t_tip = float((tip - body_c) @ axis)
    if t_tip <= 0:
        return None, None
    edges = np.linspace(0.0, t_tip, NECK_BINS + 1)
    widths = np.zeros(NECK_BINS)
    bin_idx = np.clip(np.digitize(t, edges) - 1, 0, NECK_BINS - 1)
    for b in range(NECK_BINS):
        sel = bin_idx == b
        if int(sel.sum()) >= 3:
            wc = wco[sel]
            widths[b] = np.percentile(wc, 95) - np.percentile(wc, 5)
    mw = widths.max() if widths.max() > 0 else 1.0
    norm_w = widths / mw

    run = _fretboard_run(norm_w)
    if run is None:
        return None, None
    s, e = run
    tlo, thi = edges[s], edges[e + 1]
    sel = (t >= tlo) & (t <= thi)
    if int(sel.sum()) < 10:
        return None, None

    # Rasterize the fretboard band, close small gaps, keep the largest blob.
    h, w = m.shape[:2]
    band = np.zeros((h, w), np.uint8)
    band[all_pts[sel][:, 1].astype(np.int32),
         all_pts[sel][:, 0].astype(np.int32)] = 255
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(band, 8)
    if n_lab <= 1:
        return None, None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    band_cc = (labels == big).astype(np.uint8) * 255   # the band raster (fret gate)
    ys, xs = np.where(labels == big)
    neck_pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(neck_pts) < 10:
        return None, None

    # Primary: a TRUE rotated rectangle anchored on the fretboard's fitted TOP
    # long edge (stable rotation, parallel long edges, zero shear). Fall back to
    # minAreaRect of the band only if the robust line/side fit can't be trusted.
    try:
        rect_quad = _rect_quad_from_band(neck_pts, axis, perp)
    except Exception as exc:  # the rect fit must never crash a frame
        print(f"[detect] top-edge rect fit failed: {exc!r}", flush=True)
        rect_quad = None
    if rect_quad is None:
        rect = cv2.minAreaRect(neck_pts)
        rect_quad = cv2.boxPoints(rect).astype(np.float64)

    # --- NEW: silver fret-wire 12-TET fit -> re-anchor along-extent to nut..fret7 ---
    # The rect's ALONG extent is the SAM2 band (hand-truncated, mid-neck). Detect the
    # silver wires, fit the law, and rebuild the quad so nut..fret7 lands on the REAL
    # frets. Strictly gated: only replace when the fit is trustworthy; on any failure
    # / low quality we keep the band-extent rect (never worse than before).
    fret_info = None
    if FRET_DETECT and bgr_frame is not None:
        try:
            fi = _detect_fret_wires(bgr_frame, neck_pts, band_cc, axis, perp,
                                    rect_quad)
        except Exception as exc:  # fret detection must never crash a frame
            print(f"[detect] fret-wire detection failed: {exc!r}", flush=True)
            fi = None
        if fi is not None and _fret_fit_trusted(fi["fit"]):
            new_quad = _reanchor_quad_nut_fret7(rect_quad, fi)
            if new_quad is not None:
                rect_quad = new_quad
                fret_info = fi
    return rect_quad, fret_info


def _fret_fit_trusted(fit):
    """Accept the 12-TET fret fit only if it has enough consecutive inliers and a
    small spacing residual — else we keep the band-extent quad (never regress)."""
    return (fit is not None
            and fit["inliers"] >= FRET_MIN_INLIERS
            and fit["resid_frac"] <= FRET_MAX_RESID_FRAC)


def _reanchor_quad_nut_fret7(rect_quad, fret_info):
    """Rebuild the rect quad so its ALONG extent runs the fitted real nut..fret7,
    keeping the existing top/bottom edges (string span) and rotation.

    rect_quad: [a0(top-start), a1(top-end), b1(bot-end), b0(bot-start)] — top edge
        a0->a1 along u, perpendicular a0->b0 of length = string-span width.
    fret_info: from _detect_fret_wires (nut_pt/fret7_pt on the top edge, along_u).

    Returns [nut.top, fret7.top, fret7.bottom, nut.bottom] (a true rotated rectangle,
    same width vector), or None if degenerate.
    """
    a0, b0 = np.asarray(rect_quad[0], np.float64), np.asarray(rect_quad[3], np.float64)
    width_vec = b0 - a0                       # top -> bottom (string span), preserved
    nut_top = np.asarray(fret_info["nut_pt"], np.float64)
    f7_top = np.asarray(fret_info["fret7_pt"], np.float64)
    if not (np.all(np.isfinite(nut_top)) and np.all(np.isfinite(f7_top))):
        return None
    if float(np.linalg.norm(f7_top - nut_top)) < 5.0:
        return None
    nut_bot = nut_top + width_vec
    f7_bot = f7_top + width_vec
    return np.array([nut_top, f7_top, f7_bot, nut_bot], dtype=np.float64)


def detect_quad(bgr_frame, smooth=True):
    """Detect the guitar neck and return {quad, confidence} or None.

    quad: 4 [x,y] points normalized to [0..1] by frame width/height, ordered
          canonically along the neck's long axis.
    confidence: the YOLO-World box confidence (0..1).

    smooth: when True (the default, used by the live WS stream) the emitted quad
        is EMA-smoothed across frames and held for a few frames on a miss. The
        first call after reset_state() is always a pass-through (no history), so
        the single-shot self-test sees the raw quad. Pass smooth=False to get the
        raw per-frame quad with no temporal state touched.
    """
    global _LAST_SAM_BOX, _CACHED_NECK_REL, _FRAMES_SINCE_SAM, _CACHED_FRETS

    if bgr_frame is None or bgr_frame.size == 0:
        return _miss_result() if smooth else None

    h, w = bgr_frame.shape[:2]
    model = load_model()

    results = model.predict(bgr_frame, device=_DEVICE, conf=CONF_THRESH,
                            verbose=False)
    if not results:
        return _miss_result() if smooth else None
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return _miss_result() if smooth else None

    # Most-confident box.
    confs = res.boxes.conf.cpu().numpy()
    best = int(np.argmax(confs))
    confidence = float(confs[best])
    x1, y1, x2, y2 = res.boxes.xyxy.cpu().numpy()[best].tolist()

    # Clamp box to frame.
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return _miss_result() if smooth else None

    # --- Primary: SAM2 mask of the whole guitar -> isolate the NECK ---
    # SAM2 dominates latency, so don't run it every frame. Run it when there's no
    # cached neck geometry, when the cadence counter is due, OR when the YOLO box
    # jumped (a re-acquire / large move where the cached relative neck may be
    # stale). Otherwise reproject the cached box-relative neck quad onto the new
    # box: the box already tracked this frame's motion, the neck's pose within it
    # is stable, so the quad follows the guitar without paying for SAM2.
    quad_full = None
    method = "box"
    fret_info = None         # silver-fret 12-TET fit (set when SAM2+fret fit succeed)
    box_full = (float(x1), float(y1), float(x2), float(y2))
    due = (_FRAMES_SINCE_SAM >= SAM_EVERY)
    jumped = _box_jumped(_LAST_SAM_BOX, box_full)
    if _CACHED_NECK_REL is not None and not due and not jumped:
        # Cheap path: derive the neck quad from the cached relative geometry. The
        # cached quad already has its along-extent at the fitted nut..fret7 (the fret
        # fit ran on the SAM2 frame that produced the cache), so we keep that anchor
        # for free between SAM2 runs.
        quad_full = _box_rel_to_quad(_CACHED_NECK_REL, box_full)
        method = "sam_cache"
        _FRAMES_SINCE_SAM += 1
    else:
        try:
            mask = _sam_guitar_mask(bgr_frame, box_full)
            neck_box, fret_info = _neck_quad_from_mask(mask, bgr_frame)
            if neck_box is not None:
                quad_full = _order_quad_along_long_axis(neck_box)
                method = "sam_neck_fret" if fret_info is not None else "sam_neck"
                # Refresh the cache: store this (possibly fret-anchored) neck quad
                # relative to its box.
                _CACHED_NECK_REL = _quad_to_box_rel(quad_full, box_full)
                _LAST_SAM_BOX = box_full
                _FRAMES_SINCE_SAM = 0
        except Exception as exc:  # neck extraction must never crash a frame
            print(f"[detect] neck extraction failed: {exc!r}", flush=True)
        # SAM2 ran but produced nothing usable -> fall back to the cached
        # relative neck if we have one (better than the whole-box quad).
        if quad_full is None and _CACHED_NECK_REL is not None:
            quad_full = _box_rel_to_quad(_CACHED_NECK_REL, box_full)
            method = "sam_cache"
            _FRAMES_SINCE_SAM += 1

    # --- Fallback 1: legacy in-box skin-rejecting mask -> minAreaRect ---
    if quad_full is None:
        crop = bgr_frame[y1:y2, x1:x2]
        if crop.size > 0:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            skin = cv2.inRange(hsv, SKIN_LO, SKIN_HI)
            fg = cv2.bitwise_not(skin)  # everything that is NOT skin
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
            nz = cv2.findNonZero(fg)
            if nz is not None and len(nz) >= 4:
                rect = cv2.minAreaRect(nz)            # in crop coords
                box = cv2.boxPoints(rect)             # (4,2) crop coords
                box[:, 0] += x1                       # -> full-frame px
                box[:, 1] += y1
                quad_full = _order_quad_along_long_axis(box)
                method = "skin_box"

    # --- Fallback 2: the YOLO box's own 4 corners (ordered along long axis) ---
    if quad_full is None:
        quad_full = _order_quad_along_long_axis(_box_corners(x1, y1, x2, y2))
        method = "box"

    # Blend the YOLO box confidence with the 12-TET fret-fit quality: a clean fret
    # fit (lands the law on the wires) nudges confidence up; a weak/absent fit leaves
    # the YOLO confidence essentially unchanged (FRET_CONF_BASE is near 1).
    if fret_info is not None:
        q = float(fret_info["fit"]["quality"])
        confidence = float(confidence * (FRET_CONF_BASE + (1.0 - FRET_CONF_BASE) * q))

    # Normalize to [0..1].
    quad_norm = quad_full.copy()
    quad_norm[:, 0] /= float(w)
    quad_norm[:, 1] /= float(h)
    quad_norm = np.clip(quad_norm, 0.0, 1.0)

    # Temporal smoothing (EMA across frames). Operates in normalized space so the
    # thresholds are resolution-independent. No-op on the first call after a
    # reset (the self-test path) since there's no history yet.
    if smooth:
        quad_norm, confidence = _smooth_quad(quad_norm, confidence)
        quad_norm = np.clip(quad_norm, 0.0, 1.0)
        method = method + "+ema"

    # Recompute px from the (possibly smoothed) normalized quad so the self-test
    # overlay matches exactly what the client receives.
    quad_px = quad_norm.copy()
    quad_px[:, 0] *= float(w)
    quad_px[:, 1] *= float(h)

    # Build the optional `frets` array: each detected/fitted fret as its normalized
    # position u along the EMITTED quad's nut->fret7 axis plus its fret NUMBER. We
    # project each fret's image point onto the quad's along-edge (corner0->corner1)
    # so u stays consistent with whatever canonical corner order we emit. On a fresh
    # SAM2+fret frame we (re)build it and CACHE it; on the cheap SAM2-cache path we
    # re-emit the cached frets (their u is quad-relative, so it rides the reprojected
    # cached quad). The cache is cleared whenever a SAM2 frame has no trustworthy fit.
    if fret_info is not None:
        frets_payload = _frets_for_quad(fret_info, quad_px)
        _CACHED_FRETS = frets_payload
    elif method.startswith("sam_cache"):
        frets_payload = _CACHED_FRETS
    elif method.startswith("sam_neck"):
        frets_payload = None
        _CACHED_FRETS = None        # fresh SAM2 frame, no fit -> invalidate cache
    else:
        frets_payload = None

    out = {
        "quad": [[float(x), float(y)] for x, y in quad_norm],
        "confidence": confidence,
        "method": method,
        "_quad_px": [[float(x), float(y)] for x, y in quad_px],  # for selftest draw
    }
    if frets_payload is not None:
        out["frets"] = frets_payload
    # Internal: image-space fret geometry for the selftest overlay (NOT sent on WS).
    # Only present on a fresh SAM2+fret frame (not via the SAM2-cache path, which has
    # no per-wire points). Skipped under smoothing so we don't carry stale px.
    if fret_info is not None and not smooth:
        out["_fret_info"] = {
            "nut_pt": [float(v) for v in fret_info["nut_pt"]],
            "fret7_pt": [float(v) for v in fret_info["fret7_pt"]],
            "fret_img_pts": [[int(n), [float(p[0]), float(p[1])]]
                             for n, p in fret_info["fret_img_pts"]],
            "fit": {kk: fret_info["fit"][kk]
                    for kk in ("inliers", "resid", "resid_frac", "gaps",
                               "fret_span", "quality", "scale", "sign")},
        }
    return out


def _frets_for_quad(fret_info, quad_px):
    """Turn _detect_fret_wires' per-fret image points into the WS `frets` array:
    [{"n": <fret number>, "u": <0..1 along the quad's nut->fret7 axis>}, ...].

    u is the scalar projection of each fret's top-edge image point onto the emitted
    quad's along-edge (corner0 -> corner1), divided by that edge's length. By
    construction of the re-anchored quad corner0 is the nut (u~0) and corner1 is
    fret 7 (u~1), so the consumer's nut..fret7 grid maps straight onto these.
    Returns None if degenerate.
    """
    if fret_info is None or not fret_info.get("fret_img_pts"):
        return None
    p0 = np.asarray(quad_px[0], np.float64)
    p1 = np.asarray(quad_px[1], np.float64)
    e = p1 - p0
    L2 = float(e @ e)
    if L2 < 1e-6:
        return None
    out = []
    for nfr, pt in fret_info["fret_img_pts"]:
        ptf = np.asarray(pt, np.float64)
        if not np.all(np.isfinite(ptf)):
            continue
        u = float(((ptf - p0) @ e) / L2)
        out.append({"n": int(nfr), "u": round(u, 5)})
    out.sort(key=lambda d: d["n"])
    return out or None


def _miss_result():
    """Build the detect_quad return for a frame with no fresh detection: re-emit
    the last smoothed quad (decayed confidence) during the hold window, else None.
    Centralized so every no-detection exit in detect_quad behaves identically."""
    held = _hold_or_null()
    if held is None:
        return None
    quad_norm, conf = held
    return {
        "quad": [[float(x), float(y)] for x, y in quad_norm],
        "confidence": float(conf),
        "method": "hold",
        "_quad_px": None,  # px omitted on a hold (no source frame geometry)
    }


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------
async def _handle(websocket):
    peer = getattr(websocket, "remote_address", "?")
    print(f"[ws] client connected: {peer}", flush=True)
    # Start each client session with fresh smoothing + SAM2-cache state so one
    # client's last quad never bleeds into the next session.
    reset_state()
    try:
        async for message in websocket:
            try:
                if isinstance(message, str):
                    # Ignore stray text frames; we expect binary JPEG.
                    continue
                buf = np.frombuffer(message, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is None:
                    # Decode failure = a miss; hold the last quad briefly rather
                    # than blink the overlay off on one corrupt frame.
                    det = _miss_result()
                    if det is None:
                        await websocket.send(json.dumps({"quad": None,
                                                         "confidence": 0.0}))
                        print("[detect] bad frame (decode failed)", flush=True)
                    else:
                        await websocket.send(json.dumps(
                            {"quad": det["quad"], "confidence": det["confidence"]}))
                        print("[detect] bad frame (decode failed) -> hold", flush=True)
                    continue

                det = detect_quad(frame)
                if det is None:
                    await websocket.send(json.dumps({"quad": None, "confidence": 0.0}))
                    print("[detect] conf=0.00 got_quad=n", flush=True)
                else:
                    payload = {"quad": det["quad"], "confidence": det["confidence"]}
                    # Optional silver-fret anchoring: present only when the 12-TET fret
                    # fit succeeded this frame. Each entry = {"n": fret number,
                    # "u": position 0..1 along the quad's nut(corner0)->fret7(corner1)
                    # axis}. nut is u~0, fret 7 is u~1. Absent => consumer falls back
                    # to the plain band quad (unchanged old behavior).
                    if det.get("frets"):
                        payload["frets"] = det["frets"]
                    await websocket.send(json.dumps(payload))
                    print(f"[detect] conf={det['confidence']:.2f} got_quad=y "
                          f"method={det.get('method', '?')} "
                          f"frets={len(det.get('frets') or [])}", flush=True)
            except Exception as exc:  # never crash on a single bad frame
                print(f"[detect] error: {exc!r}", flush=True)
                try:
                    await websocket.send(json.dumps({"quad": None,
                                                     "confidence": 0.0}))
                except Exception:
                    pass
    except Exception as exc:
        print(f"[ws] connection closed: {exc!r}", flush=True)
    finally:
        print(f"[ws] client disconnected: {peer}", flush=True)


async def _serve():
    import websockets
    # Warm the model before accepting connections so the first frame is fast.
    load_model()
    print(f"[ws] listening on ws://{WS_HOST}:{WS_PORT}  (send binary JPEG frames)",
          flush=True)
    async with websockets.serve(_handle, WS_HOST, WS_PORT, max_size=None):
        await asyncio.Future()  # run forever


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------
def selftest(image_path: str) -> int:
    img = cv2.imread(image_path)
    if img is None:
        print(f"[selftest] could not read image: {image_path}", flush=True)
        return 2

    # Single-shot: fresh state, no temporal smoothing — report the raw per-frame
    # neck quad (the live stream layers smoothing on top of exactly this).
    reset_state()
    t0 = time.time()
    det = detect_quad(img, smooth=False)
    dt = time.time() - t0

    if det is None:
        print(f"[selftest] NO DETECTION (conf<{CONF_THRESH}) in {dt:.2f}s on "
              f"{image_path}", flush=True)
        return 1

    print(f"[selftest] DETECTED guitar  conf={det['confidence']:.3f}  "
          f"method={det.get('method', '?')}  in {dt:.2f}s", flush=True)
    print(f"[selftest] quad (normalized 0..1): {det['quad']}", flush=True)

    out = img.copy()
    quad_px = np.array(det["_quad_px"], dtype=np.float64)

    # --- Draw the detected silver fret WIRES + the fitted nut/fret7 (verification) ---
    # Each fret line is drawn from its top-edge image point along the quad's width
    # (top->bottom) vector so we can SEE the line sit on the real silver wire.
    fi = det.get("_fret_info")
    if fi is not None:
        width_vec = quad_px[3] - quad_px[0]   # nut.top -> nut.bottom (string span)
        f = fi["fit"]
        print(f"[selftest] FRET FIT: {f['inliers']} frets, span={f['fret_span']}, "
              f"gaps={f['gaps']}, resid={f['resid']:.1f}px "
              f"(resid/scale={f['resid_frac']*100:.2f}%), quality={f['quality']:.2f}",
              flush=True)
        nums = [n for n, _ in fi["fret_img_pts"]]
        print(f"[selftest] fret numbers detected: {sorted(nums)}", flush=True)
        # detected wires (cyan), labeled with their fret number
        for nfr, pt in fi["fret_img_pts"]:
            top = np.array(pt, np.float64)
            bot = top + width_vec
            cv2.line(out, tuple(top.astype(int)), tuple(bot.astype(int)),
                     (255, 255, 0), 2)
            cv2.putText(out, str(nfr), (int(top[0]) - 6, int(top[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        # fitted NUT (red) + FRET 7 (magenta), the re-anchored quad ends
        for pt0, col, lab in ((fi["nut_pt"], (0, 0, 255), "NUT(0)"),
                              (fi["fret7_pt"], (255, 0, 255), "FRET7")):
            top = np.array(pt0, np.float64)
            bot = top + width_vec
            cv2.line(out, tuple(top.astype(int)), tuple(bot.astype(int)), col, 3)
            cv2.putText(out, lab, (int(top[0]) - 10, int(top[1]) - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    else:
        print("[selftest] FRET FIT: none (kept band-extent quad)", flush=True)

    # --- Draw the emitted quad (green) on top so it's visible over the fret lines ---
    pts = quad_px.astype(np.int32)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], isClosed=True,
                  color=(0, 255, 0), thickness=3)
    labels = ["0", "1", "2", "3"]
    for (x, y), lab in zip(det["_quad_px"], labels):
        cv2.circle(out, (int(x), int(y)), 7, (0, 0, 255), -1)
        cv2.putText(out, lab, (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(out, f"conf={det['confidence']:.3f}  "
                f"frets={len(det.get('frets') or [])}", (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    base, _ = os.path.splitext(image_path)
    out_path = base + "_annotated.png"
    cv2.imwrite(out_path, out)
    print(f"[selftest] wrote annotated image: {out_path}", flush=True)
    return 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="YOLO-World fretboard detector")
    ap.add_argument("--selftest", metavar="IMAGE",
                    help="run detect_quad on an image and write an annotated PNG")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest(args.selftest))
    else:
        try:
            asyncio.run(_serve())
        except KeyboardInterrupt:
            print("\n[ws] shutting down", flush=True)


if __name__ == "__main__":
    main()
