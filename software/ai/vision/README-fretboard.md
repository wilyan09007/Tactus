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

- **URL / port:** `ws://localhost:8770`
- **Client -> server:** one **BINARY** message per frame = raw **JPEG bytes**
  (e.g. a webcam frame encoded as JPEG in the browser).
- **Server -> client:** one **JSON text** message per frame:

  ```json
  { "quad": [[x,y],[x,y],[x,y],[x,y]], "confidence": 0.93 }
  ```

  - `quad`: 4 `[x, y]` points **normalized to [0..1]** by the received frame's
    width/height, ordered canonically along the neck's long axis:
    `[along-start·sideA, along-end·sideA, along-end·sideB, along-start·sideB]`.
  - `confidence`: the YOLO-World box confidence (0..1).
  - If nothing is detected (or the frame can't be decoded):
    `{ "quad": null, "confidence": 0.0 }`.

The server never crashes on a bad frame (per-message try/except) and prints one
log line per detection, e.g. `[detect] conf=0.93 got_quad=y`.

### Browser sketch

```js
const ws = new WebSocket("ws://localhost:8770");
ws.binaryType = "arraybuffer";
// from a <canvas> drawing the <video>:
canvas.toBlob(b => b.arrayBuffer().then(buf => ws.send(buf)), "image/jpeg", 0.8);
ws.onmessage = e => {
  const { quad, confidence } = JSON.parse(e.data);
  // quad is null or 4 normalized [x,y] corners; multiply by canvas w/h to draw.
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
4. If SAM2 is unavailable or neck extraction fails, it falls back to the legacy
   in-box skin-rejecting mask, and finally to the YOLO box's own 4 corners — so
   it never crashes and never returns garbage. The `--selftest` line prints which
   path ran, e.g. `method=sam_neck` / `method=skin_box` / `method=box`.

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
