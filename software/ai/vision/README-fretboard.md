# Fretboard / guitar-neck detector (YOLO-World over WebSocket)

Open-vocabulary guitar-**neck** detector. No training, no markers. Uses
Ultralytics **YOLO-World** (`yolov8x-worldv2.pt`) with a text prompt to localize
the guitar, then **SAM2** (`sam2_t.pt`) to segment it, then isolates the thin
**neck/fretboard** from that mask and returns its 4-corner quad. Replaces the
earlier OpenCV-Hough and MediaPipe-hand auto-detectors, which both failed on
busy backgrounds.

Why the mask step: YOLO-World returns ONE box around the **whole** guitar (body
+ neck, conf ~0.93), so a box-derived quad covers the whole instrument. The
consumer maps a 6-string x 7-fret grid that must sit on the **neck only**, so we
segment the guitar and extract just the fretboard (see "How neck isolation
works" below).

Runs on Apple Silicon via **MPS** (PyTorch), with automatic CPU fallback for any
op MPS doesn't implement (`PYTORCH_ENABLE_MPS_FALLBACK=1`, set in code).

## Setup (already done; for reference)

```bash
python3.11 -m venv software/ai/vision/.venv-yolo
software/ai/vision/.venv-yolo/bin/pip install -U ultralytics opencv-python websockets
```

The venv (`.venv-yolo/`) and the auto-downloaded weights (`*.pt`) are gitignored
via `software/ai/vision/.gitignore` — do not commit them. Two weights download on
first run: `yolov8x-worldv2.pt` (~140 MB) and `sam2_t.pt` (~74 MB, SAM2-tiny).

## Run the server (one line)

```bash
software/ai/vision/.venv-yolo/bin/python software/ai/vision/fretboard_server.py
```

On startup it loads YOLO-World + CLIP text classes and SAM2 **once** (a few
seconds warm, longer the very first time while it downloads weights), then
listens.

## WebSocket protocol

- **URL / port:** `ws://127.0.0.1:8772` (moved off `:8770` — macOS `sharingd` holds a dual-stack listener there; use `127.0.0.1`, not `localhost`)
- **Client -> server:** one **BINARY** message per frame = raw **JPEG bytes**
  (e.g. a webcam frame encoded as JPEG in the browser).
- **Server -> client:** one **JSON text** message per frame:

  ```json
  {
    "quad": [[x,y],[x,y],[x,y],[x,y]],
    "confidence": 0.93,
    "frets": [ {"n": 0, "u": 0.0}, {"n": 1, "u": 0.169}, {"n": 7, "u": 1.0} ]
  }
  ```

  - `quad`: 4 `[x, y]` points **normalized to [0..1]** by the received frame's
    width/height, ordered canonically along the neck's long axis:
    `[along-start·sideA, along-end·sideA, along-end·sideB, along-start·sideB]`.
    When the silver-fret fit succeeds the **along extent is the real
    nut → fret 7** (see "Silver fret-wire anchoring" below): `corner0`/`corner3`
    are the **nut** edge, `corner1`/`corner2` are the **fret 7** edge — so the
    consumer's nut→fret-7 grid lands on the real frets even when the fretting hand
    occludes the nut.
  - `confidence`: the YOLO-World box confidence (0..1), nudged up slightly by the
    12-TET fret-fit quality when a fret fit is present.
  - `frets` *(optional)*: present **only** when the silver fret wires were detected
    and the 12-TET law fit was trustworthy this frame. Each entry is
    `{"n": <fret number, 0 = nut>, "u": <position 0..1 along the quad's
    nut(corner0)→fret7(corner1) axis>}`. So `n:0 → u≈0`, `n:7 → u≈1`, and frets
    past 7 (toward the body) have `u > 1`. `u` is the **measured** wire position
    (not the pure law prediction), projected onto the emitted quad's along-edge, so
    it stays valid across the SAM2-cache frames and any canonical corner reorder.
    **Absent** ⇒ fall back to the plain `quad` (old behavior, never worse).
  - If nothing is detected (or the frame can't be decoded):
    `{ "quad": null, "confidence": 0.0 }` (no `frets`).

The server never crashes on a bad frame (per-message try/except) and prints one
log line per detection, e.g. `[detect] conf=0.93 got_quad=y method=sam_neck_fret frets=9`.

### Browser sketch

```js
const ws = new WebSocket("ws://127.0.0.1:8772");
ws.binaryType = "arraybuffer";
// from a <canvas> drawing the <video>:
canvas.toBlob(b => b.arrayBuffer().then(buf => ws.send(buf)), "image/jpeg", 0.8);
ws.onmessage = e => {
  const { quad, confidence, frets } = JSON.parse(e.data);
  // quad is null or 4 normalized [x,y] corners; multiply by canvas w/h to draw.
  // frets (optional): [{n, u}] with u in [0..1] along corner0(nut)->corner1(fret7).
  // To place fret n's line: lerp(quad[0], quad[1], u) (top) .. lerp(quad[3], quad[2], u) (bottom).
};
```

## Headless self-test (no camera)

```bash
software/ai/vision/.venv-yolo/bin/python software/ai/vision/fretboard_server.py \
  --selftest software/ai/vision/guitar_test.jpg
```

Prints the detected quad + confidence and writes `<image>_annotated.png` next to
the input with the quad drawn.

## How neck isolation works

`detect_quad` runs YOLO-World -> SAM2 -> neck extraction:

1. **YOLO-World** finds the most-confident guitar box (whole instrument).
2. **SAM2** (`sam2_t.pt`), prompted with that box, returns an instance **mask**
   of the whole guitar.
3. **Neck extraction** from the mask (`_neck_quad_from_mask`), orientation-robust
   because the round body otherwise dominates a global axis:
   - **Distance transform**: the wide body core is the only "thick" region; the
     thin remainder is neck + headstock.
   - **Axis**: body centroid (centroid of the thick core) -> the thin pixel
     farthest from it (the headstock tip). That vector is the neck's long axis.
   - **March** `NECK_BINS` bins along the axis; per bin measure the mask's
     perpendicular width. The **fretboard** is the narrow, roughly-constant-width
     run starting where the body ends (`_fretboard_run`); the fretting hand's
     occluded bins are bridged, but the headstock (pegs flare wider / the
     foreshortened head collapses narrower) ends the run.
   - `minAreaRect` over the largest connected blob of that run -> 4 neck corners.
4. A **true rotated rectangle** is anchored on the fretboard's robustly-fitted
   **top long edge** (`_rect_quad_from_band`): stable rotation, parallel long
   edges, zero shear. Falls back to `minAreaRect` of the band if that fit can't be
   trusted.
5. **Silver fret-wire anchoring** (`_detect_fret_wires` → `_fit_fret_law_1d`): the
   rect from step 4 spans only the SAM2 band (mid-neck, truncated by the fretting
   hand), **not** the nut. So we re-anchor it to the **real nut → fret 7** — see
   the next section.
6. If SAM2 is unavailable or neck extraction fails, it falls back to the legacy
   in-box skin-rejecting mask, and finally to the YOLO box's own 4 corners — so
   it never crashes and never returns garbage. The `--selftest` line prints which
   path ran, e.g. `method=sam_neck_fret` (fret-anchored) / `method=sam_neck`
   (band-extent, fret fit declined) / `method=sam_cache` / `method=skin_box` /
   `method=box`.

## Silver fret-wire anchoring (real nut → fret 7)

The SAM2 band's along extent is the un-occluded mid-neck, so a nut→fret-7 grid
mapped onto the raw band quad does **not** line up with the real frets. The silver
**fret wires** are a strong, high-contrast cue, so we detect them and anchor the
grid to the real frets (`FRET_DETECT`, on by default):

1. **Rectify** the masked neck band by the top-edge angle (inverse `cv2.remap`
   along the rect's along-axis `u` / perpendicular `n_hat`, padded past both ends),
   so frets become ~vertical and the neck ~horizontal.
2. **Pop the wires**: `|Sobel_x|` (along-neck gradient energy) spikes on each bright
   thin silver wire. Profiled over the **top + bottom edge rows only**
   (`FRET_EDGE_BAND`) so the central inlay **dots** (which sit mid-neck, not
   full-height like wires) don't register. The spatial gate is the **fretboard band
   itself** (the distance-transform split already drops the body; the fret-run
   gap-bridging drops most of the hand). We deliberately **don't** color-mask the
   hand — a fixed skin HSV range eats ~98% of a warm-lit brown fretboard (verified
   on a real warm-lit frame), and the law fit rejects the few non-conforming peaks a
   finger edge leaves. `scipy.signal.find_peaks` → fret x-positions.
3. **Fit the 12-TET law** `x(n) = x0 + scale·sign·(1 − 2^(−n/12))` to the peaks
   (`_fit_fret_law_1d`, the 1-D analogue of `fretboard_detect.fit_law`): tries both
   signs and both anchor-label orientations (**which end is the nut is exactly what
   it solves**), assigns each peak a fret **number**, and scores by consecutive
   inliers minus a phantom-gap penalty with the spacing residual as the tiebreak.
   This **extrapolates the nut (fret 0) and fret 7** even when the hand occludes
   them.
4. **Re-anchor the quad** (`_reanchor_quad_nut_fret7`): map the fitted nut and
   fret-7 rectified-x back to image space via `u`, and rebuild
   `[nut·top, fret7·top, fret7·bottom, nut·bottom]` — keeping the step-4 top/bottom
   edges (string span) and rotation. Still a true rotated rectangle.
5. **Strict gate**: only replace the band quad when the fit has
   ≥ `FRET_MIN_INLIERS` consecutive inliers and residual/scale ≤
   `FRET_MAX_RESID_FRAC` (`_fret_fit_trusted`); otherwise keep the band-extent quad
   (**never worse than before**). The fit runs only on the SAM2 frames and is cached
   (quad-relative `frets` u) so the cheap SAM2-cache frames re-emit it for free.

Verified on the real night-courtyard frames (`realtest/`): the drawn fret lines sit
on the real silver wires and the quad extends toward the real nut past the hand. The
`--selftest` overlay draws the detected wires (cyan, numbered), the fitted **NUT**
(red) and **FRET7** (magenta), and the emitted quad (green).

## Tuning

- Confidence gate: `CONF_THRESH` (default `0.20`).
- Class prompts: `CLASSES` (default
  `["guitar neck","guitar fretboard","acoustic guitar","guitar"]`).
- Neck extraction (in `fretboard_server.py`): `DT_FRAC` (body/neck split),
  `NECK_BINS`, `NECK_WIDTH_FRAC` (a bin is "neck" if its width is below this
  fraction of the max), `NECK_MAX_GAP` (occluded bins to bridge), and
  `NECK_BAND_LO`/`NECK_BAND_HI` (width consistency band that rejects the
  headstock when bridging a gap).
- Legacy fallback only: skin-rejection HSV range `SKIN_LO` / `SKIN_HI`.
- Silver fret-wire anchoring (`fretboard_server.py`): `FRET_DETECT` (master switch;
  `False` ⇒ exact pre-feature behavior, no `frets`), `FRET_EDGE_BAND` (top/bottom
  rows used for the profile, rejects inlay dots), `FRET_PEAK_PROMINENCE` /
  `FRET_PEAK_MIN_DIST_FRAC` (peak finder), `FRET_MIN_INLIERS` /
  `FRET_MAX_RESID_FRAC` (12-TET fit acceptance — below them it falls back to the
  band quad), `FRET_RECT_PAD_ALONG` (how far past the band to look for the
  extrapolated nut), and `FRET_CONF_BASE` (how much the fret-fit quality is allowed
  to raise confidence).
