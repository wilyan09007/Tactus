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
    .venv-yolo/bin/python fretboard_server.py            # ws://localhost:8770

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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WS_HOST = "localhost"
WS_PORT = 8770
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
    global _LAST_SAM_BOX, _CACHED_NECK_REL, _FRAMES_SINCE_SAM
    global _LAST_EMIT_QUAD, _LAST_EMIT_CONF, _MISS_COUNT
    _LAST_SAM_BOX = None
    _CACHED_NECK_REL = None
    _FRAMES_SINCE_SAM = 0
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


def _neck_quad_from_mask(mask):
    """Isolate the NECK from a whole-guitar mask and return its 4 corners
    (full-frame px, cv2.boxPoints order) or None.

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
    """
    if mask is None:
        return None
    m = (mask > 0).astype(np.uint8)
    if int(m.sum()) < 200:
        return None

    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    maxd = float(dt.max())
    if maxd <= 0:
        return None

    # Thick body core, dilated back toward the full body so the rim is excluded
    # from the "thin" set as much as possible.
    rad = int(DT_FRAC * maxd * 2) | 1          # odd kernel size
    thick = (dt > DT_FRAC * maxd).astype(np.uint8)
    thick = cv2.dilate(thick, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (rad, rad)))
    thick = cv2.bitwise_and(thick, m)
    tys, txs = np.where(thick > 0)
    if len(txs) < 10:
        return None
    body_c = np.array([txs.mean(), tys.mean()], dtype=np.float64)

    thin = cv2.bitwise_and(m, cv2.bitwise_not(thick))
    nys, nxs = np.where(thin > 0)
    if len(nxs) < 30:
        return None
    thin_pts = np.column_stack([nxs, nys]).astype(np.float64)
    tip = thin_pts[int(np.argmax(((thin_pts - body_c) ** 2).sum(1)))]
    axis = tip - body_c
    nrm = np.linalg.norm(axis)
    if nrm < 1e-6:
        return None
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
        return None
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
        return None
    s, e = run
    tlo, thi = edges[s], edges[e + 1]
    sel = (t >= tlo) & (t <= thi)
    if int(sel.sum()) < 10:
        return None

    # Rasterize the fretboard band, close small gaps, keep the largest blob.
    h, w = m.shape[:2]
    band = np.zeros((h, w), np.uint8)
    band[all_pts[sel][:, 1].astype(np.int32),
         all_pts[sel][:, 0].astype(np.int32)] = 255
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(band, 8)
    if n_lab <= 1:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(labels == big)
    neck_pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(neck_pts) < 10:
        return None

    # Primary: a TRUE rotated rectangle anchored on the fretboard's fitted TOP
    # long edge (stable rotation, parallel long edges, zero shear). Fall back to
    # minAreaRect of the band only if the robust line/side fit can't be trusted.
    try:
        rect_quad = _rect_quad_from_band(neck_pts, axis, perp)
    except Exception as exc:  # the rect fit must never crash a frame
        print(f"[detect] top-edge rect fit failed: {exc!r}", flush=True)
        rect_quad = None
    if rect_quad is not None:
        return rect_quad
    rect = cv2.minAreaRect(neck_pts)
    return cv2.boxPoints(rect)


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
    global _LAST_SAM_BOX, _CACHED_NECK_REL, _FRAMES_SINCE_SAM

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
    box_full = (float(x1), float(y1), float(x2), float(y2))
    due = (_FRAMES_SINCE_SAM >= SAM_EVERY)
    jumped = _box_jumped(_LAST_SAM_BOX, box_full)
    if _CACHED_NECK_REL is not None and not due and not jumped:
        # Cheap path: derive the neck quad from the cached relative geometry.
        quad_full = _box_rel_to_quad(_CACHED_NECK_REL, box_full)
        method = "sam_cache"
        _FRAMES_SINCE_SAM += 1
    else:
        try:
            mask = _sam_guitar_mask(bgr_frame, box_full)
            neck_box = _neck_quad_from_mask(mask)
            if neck_box is not None:
                quad_full = _order_quad_along_long_axis(neck_box)
                method = "sam_neck"
                # Refresh the cache: store this neck quad relative to its box.
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

    return {
        "quad": [[float(x), float(y)] for x, y in quad_norm],
        "confidence": confidence,
        "method": method,
        "_quad_px": [[float(x), float(y)] for x, y in quad_px],  # for selftest draw
    }


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
                    await websocket.send(json.dumps(payload))
                    print(f"[detect] conf={det['confidence']:.2f} got_quad=y "
                          f"method={det.get('method', '?')}", flush=True)
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

    # Draw the quad on a copy and save next to the input.
    out = img.copy()
    pts = np.array(det["_quad_px"], dtype=np.int32)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], isClosed=True,
                  color=(0, 255, 0), thickness=3)
    labels = ["0", "1", "2", "3"]
    for (x, y), lab in zip(det["_quad_px"], labels):
        cv2.circle(out, (int(x), int(y)), 7, (0, 0, 255), -1)
        cv2.putText(out, lab, (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(out, f"conf={det['confidence']:.3f}", (12, 36),
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
