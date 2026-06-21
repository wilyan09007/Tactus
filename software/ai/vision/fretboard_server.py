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
MODEL_NAME = "yolov8x-worldv2.pt"          # auto-downloads (~146 MB) on first load
SAM_NAME = "sam2_t.pt"                      # auto-downloads (~74 MB) on first load
CLASSES = ["guitar neck", "guitar fretboard", "acoustic guitar", "guitar"]
CONF_THRESH = 0.20

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

# Skin HSV range (the fretting hand) — used only by the legacy box-refine
# fallback when SAM2 / neck extraction is unavailable.
SKIN_LO = np.array([0, 30, 60], dtype=np.uint8)
SKIN_HI = np.array([25, 180, 255], dtype=np.uint8)

# Lazily-initialized singletons (loaded once).
_MODEL = None
_SAM = None
_DEVICE = "mps"


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
         a tilted guitar), and minAreaRect it -> 4 neck corners.
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
    rect = cv2.minAreaRect(neck_pts)
    return cv2.boxPoints(rect)


def detect_quad(bgr_frame):
    """Detect the guitar neck and return {quad, confidence} or None.

    quad: 4 [x,y] points normalized to [0..1] by frame width/height, ordered
          canonically along the neck's long axis.
    confidence: the YOLO-World box confidence (0..1).
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return None

    h, w = bgr_frame.shape[:2]
    model = load_model()

    results = model.predict(bgr_frame, device=_DEVICE, conf=CONF_THRESH,
                            verbose=False)
    if not results:
        return None
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None

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
        return None

    # --- Primary: SAM2 mask of the whole guitar -> isolate the NECK ---
    quad_full = None
    method = "box"
    box_full = (float(x1), float(y1), float(x2), float(y2))
    try:
        mask = _sam_guitar_mask(bgr_frame, box_full)
        neck_box = _neck_quad_from_mask(mask)
        if neck_box is not None:
            quad_full = _order_quad_along_long_axis(neck_box)
            method = "sam_neck"
    except Exception as exc:  # neck extraction must never crash a frame
        print(f"[detect] neck extraction failed: {exc!r}", flush=True)

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

    return {
        "quad": [[float(x), float(y)] for x, y in quad_norm],
        "confidence": confidence,
        "method": method,
        "_quad_px": [[float(x), float(y)] for x, y in quad_full],  # for selftest draw
    }


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------
async def _handle(websocket):
    peer = getattr(websocket, "remote_address", "?")
    print(f"[ws] client connected: {peer}", flush=True)
    try:
        async for message in websocket:
            try:
                if isinstance(message, str):
                    # Ignore stray text frames; we expect binary JPEG.
                    continue
                buf = np.frombuffer(message, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is None:
                    await websocket.send(json.dumps({"quad": None, "confidence": 0.0}))
                    print("[detect] bad frame (decode failed)", flush=True)
                    continue

                det = detect_quad(frame)
                if det is None:
                    await websocket.send(json.dumps({"quad": None, "confidence": 0.0}))
                    print("[detect] conf=0.00 got_quad=n", flush=True)
                else:
                    payload = {"quad": det["quad"], "confidence": det["confidence"]}
                    await websocket.send(json.dumps(payload))
                    print(f"[detect] conf={det['confidence']:.2f} got_quad=y",
                          flush=True)
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

    t0 = time.time()
    det = detect_quad(img)
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
