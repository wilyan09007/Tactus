# Archived prototypes — NOT the product UI

These are earlier exploration passes, kept only for reference. **The real Tactus
LEARN interface is [`../glow.html`](../glow.html)** (open `/` → it redirects there).
Do not build on or ship these.

| File | What it was | Why archived |
|------|-------------|--------------|
| `learn-dashboard-mock.html` | Black 14-channel dashboard: top-bar pills, body-map (6 chest + 6 forearm), note highway, 3-fix correction panel. Runnable mock of the WebSocket `frame`/`note`/`error`/`reference` contract (docs/16). | Uniform-blue strings, no AR-on-real-neck. Superseded by glow.html. |
| `beatsaber.html` | three.js Beat-Saber highway — note blades fly down the string lanes toward a "now" bar, composited over the webcam. | Single-hue (gold), depth-approach not on-the-real-neck. Concept lives in docs/22. |
| `ar.html` | Chord-field — fretboard faces the camera, blades light up in the chord shape, depth = hold length. | Single-hue, face-on grid. Exploration only. |

What made glow.html win: **per-string colour + on-the-real-neck annotation +
silent render-only alphaTab tab**, all on the live camera. See `../README.md`.
