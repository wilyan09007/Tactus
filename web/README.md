# Tactus — web

The **product UI** is **[`glow.html`](glow.html)** — the live fret-glow LEARN
interface. `index.html` just redirects to it, so opening `/` opens the real thing.

## Run

```bash
cd web
python3 -m http.server 8000
# open http://localhost:8000/   (redirects to glow.html)
```

Allow the camera, then **calibrate**: drag the 4 corner handles onto your real
fretboard. The glow markers then sit on the real strings/frets.

## What glow.html is

The canonical LEARN view. Notes are annotated **on the real neck** over the
webcam — no synth, render-only (right for a Deaf-focused app).

- **Per-string 6-hue palette** (CVD-aware) — each string is its own colour;
  nut labels back it up for accessibility.
- **ADSR brightness envelope** — a marker washes in pale when far, deepens to
  full saturation at the strum, then releases. Approach-frame shrinks to meet
  the marker = strum now; pop burst on the beat.
- **4-corner bilinear neck grid** — draggable handles map the grid to your neck
  (equal-temperament fret spacing, `2^(-n/12)`).
- **alphaTab real tab** on the left — actual notation/tab rendered from one
  `SONG` source that also builds the glow chart, so left and right are the same
  song, locked to one master clock. CDN-blocked → falls back to a chord-row list.

## Helpers (prepared, not yet wired)

- `alphatab-loader.js` — parse a GuitarPro/MusicXML file → Tactus chart, headless
  (no DOM/synth). For driving glow from real song files.
- `aruco-poselock.js` — js-aruco2 marker pose-lock to **auto**-align the neck grid
  (drops the manual 4-corner calibration). The next AR layer.

## `_prototypes/`

Earlier exploration UIs, **kept for reference only — not the product**:
`learn-dashboard-mock.html` (the old black 14-channel body-map mock),
`beatsaber.html` (notes-fly-in highway), `ar.html` (chord-field depth view).
Do not build on these — glow.html is the one.
