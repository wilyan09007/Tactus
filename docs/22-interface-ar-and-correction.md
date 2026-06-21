# 22 — The interface: live AR play + targeted 2D correction (the centerpiece)

Status: **DRAFT — for `/plan-eng-review`**. The god-tier interactive experience. Supersedes the flat "sheet-left / camera-right" layout in `docs/16`; keeps doc 16's governing law: **every pixel is a sense the Deaf learner doesn't have** (so cool == useful).

> Two coupled views: **(A) a live-AR, Beat-Saber depth-approach play view on your real neck**, and **(B) a dead-simple 2D fretboard that shows exactly what each finger did wrong and animates the fix.** Decision: AR is the committed centerpiece (Aditya).

---

## A. LIVE AR PLAY — Beat-Saber depth-approach on your real fretboard
Notes fly in from **depth (Z axis)** and land on the exact **(string, fret)** cell on the beat — not a top-down scroll. Depth gives natural lead-time, and the note arrives *where your finger goes*, killing the chart→hand translation step.

- **Registration:** the **live AR uses an ArUco pose-lock** (`web/aruco-poselock.js` — js-aruco2 detect → POSIT pose → one-euro smooth → confidence-gated **ghost-neck** fallback; `ARUCO_MIP_36h12` marker, physical side measured in mm). Notes are rendered in the **neck coordinate frame** and projected through the pose, so they stick to the real frets as the neck moves. *(Independent subsystem note: the offline **training/data pipeline registers MARKERLESS** — the 12-TET fret-law homography, `software/ai/vision/fretboard.py` — because pose there is computed offline and the prompt is the label. AR pose for the demo, markerless for `d`/features; they don't need to share a marker.)*
- **Occlusion-friendly by construction:** notes fly in **above** the board (visible until the instant of contact); registration rides on the nut/neck geometry (and the high frets stay visible above the fretting hand).
- **Color = per-string (6 colors), SAME palette as the back body-zones** → one color language across screen + body + haptics. Finger = a numeral/icon on the note.
- **Sustain = a depth "tail"** (length = duration) — the visual twin of the haptic looped-shimmer (`docs/21`). **Chord** = notes arriving together (bloom) or as a strum sweep.
- **Customization:** speed (slow/fast = approach speed + lead distance), **loop-any-segment**, left-handed mirror, difficulty (note density), color themes.

### The mandatory fallback (demo insurance — build this FIRST)
A **"virtual neck"** mirror: the **identical Z-axis renderer** with a swappable backdrop — live camera+ArUco when the pose is solid, a clean **rendered neck** when it isn't (bad lighting / jitter). Same experience, only the backdrop + pose source change. The centerpiece can never die on stage. Jittery AR looks worse than none, so the virtual neck is not optional.

## B. TARGETED CORRECTION (on error) — the super-simple 2D fretboard
On a miss: **rewind the segment**, switch to a clean **top-down 2D fretboard schematic**.
- Per finger, show **WHAT YOU PLAYED**: position (string, fret) + **where on the fret** (sub-fret distance `d` from vision — dot placed within the cell: too-far-back vs on-the-wire) + **pressure** (dot encoding: hollow/pale = too light, solid = good, hot/over-saturated = too hard).
- Show the **TARGET** as ghost dots.
- **Dead-simple animated instructions:** arrows ("slide this finger toward the fret"), a press-down pulse on under-pressured fingers ("press harder with these fingers"). Plainest language / ASL-gloss-friendly.

## Per-finger position + pressure — what's supportable (ties to `docs/20`)
- **Per-finger PLACEMENT** (which finger, where, wire-distance `d`): **solid**, from vision (per-finger) → "slide *these* fingers" fully supported.
- **Per-finger PRESSURE:** a **vision-led estimate** (finger-load geometry: curl, knuckle collapse, plant) **cross-checked by the global audio buzz** (audio: someone's light; vision: which finger looks lightest). **Robust for single notes; best-effort attribution in chords.** "Press harder with your bottom 3 fingers" = vision-led best-effort, honestly scoped. The correction view IS the `docs/20` fusion output, drawn.

## Coherence (one phase-locked signal → screen + body + sound)
The AR notes, the body-aurora ambient, and the haptic drive all come from one stream; nail a note and all channels confirm together. That coherence is the "alive, not an app" feeling.

## Build order (de-risked)
1. **Virtual-neck Z-Beat-Saber play view + the 2D correction view** — fully functional, demo-safe, no AR yet.
2. **Layer live AR** (ArUco neck pose → render notes in-frame) on top, with the virtual neck as the instant fallback.
3. **Polish:** sustain tails, color themes, the body-aurora + cluster/response-surface ambient (`docs/20` §5).

## Open questions for eng review
- AR pose **stability** under neck motion + occlusion; smoothing filter choice; multi-marker; re-acquire cadence; jitter threshold that auto-switches to the virtual neck.
- **Rendering stack** (three.js / WebXR over the webcam; in-browser vs native) and the latency budget for AR + MediaPipe + F0 + haptic render on one laptop.
- **Per-finger pressure confidence** to display — show a hint, not a fake precise number.
- **Readability:** depth cues for the Z-approach; color-blind-safe palette.

## Award alignment (`docs/19`)
Best UI/UX (the AR + the clarity of the fix), Anthropic (Claude authors the dead-simple instructions from the fusion output), Accessibility / Ddoski's World, CV track (the AR neck pose + hand tracking).
</content>
