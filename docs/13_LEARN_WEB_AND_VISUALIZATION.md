# 13 — LEARN: Closing the Loop (feel the target · see the fix)

> **The core idea — and why it's the guaranteed winner.** A *hearing* person learns a song by closing a loop with their ears: **play → hear the difference between what you played and what it should sound like → correct.** A Deaf person has never had that error signal. **Tactus closes the same loop with two senses instead of ears:**
> 1. **Haptic ground-truth** — you can *feel what the song is supposed to feel like* (we haptically play the correct version on the vest), and feel **your version vs the correct version back-to-back**, so the error is as tangible as a wrong note is to a hearing player.
> 2. **Vision** — a camera + VLM watches your hands and tells you the **exact physical change** to make, highlighting the wrong finger on screen.
>
> Think **"Rocksmith for Deaf players"**: it listens to your real guitar, and when you get a segment wrong it **rewinds, replays the correct haptics, and shows you what your hands should do.** Haptics tell you *what it should be*; vision tells you *how to fix your body*. That pairing is the recipe.

**Scope for the hackathon: GUITAR ONLY** (it's the easiest instrument to analyze visually — a flat fretboard, discrete frets, visible finger positions). The architecture generalizes to other instruments later, but we ship guitar.

---

## 1. The learning loop (one flow — replaces the old practice modes)
*(We dropped "Recognize" and "Anticipate" — this single, tighter loop is the product.)*

```
 pick a song (guitar)            ┌──────────────── on error ────────────────┐
        │                        │                                          │
        ▼                        │   REWIND the wrong segment                │
 ① FEEL THE TARGET  ─────────►   │   ② replay it on the vest, two ways:      │
   (vest plays the correct        │      • what YOU just played (feel it)     │
    segment so you know            │      • the CORRECT version (feel it)      │
    what it should feel like)      │        → you FEEL the difference          │
        │                         │   ③ VISION shows the physical fix:        │
        ▼                         │      VLM highlights the wrong finger +    │
 ④ YOU PLAY IT on the real        │      the correct hand position            │
   guitar; mic + camera capture   │   ④ ONE specific instruction (3 kinds)    │
        │                         │      → loop the segment until it's right  │
        ▼                         └──────────────────────────────────────────┘
 ⑤ live assessment: are you
   playing it correctly?  ── yes ──►  advance to the next segment
```

The whole point: **you never need to hear anything.** You *feel* the target, you *feel* your error against it, you *see* the physical correction, and you get one concrete thing to change.

---

## 2. The four components

### Component A — the Haptic Reference ("what it should feel like")
The interface can **haptically play any song or piece of sheet music** on the vest — this is the *ground truth* the learner internalizes (the user's crucial point: *"be able to feel what a song actually sounds like"*).
- **Source = the target's notes + timing + fingering.** For guitar the ideal source is **tab / MIDI / MusicXML** (gives exact pitch, duration, *and* which string/fret/finger) — cleaner than raw audio for a learning target. Plain audio also works (transcribe it once via the perception engine, truth.md §6 / docs/20), but tab/MIDI gives us the **correct fingering** we need for the vision fix.
- **Feel-the-difference (the true analog to "hearing the gap"):** on an error we play **your version** (your mic audio → the perception engine's transcription → haptics) immediately followed by the **correct version** (reference → haptics). A wrong note lands at a different body location; a short note has a clipped envelope; a buzzed note has a rougher texture — so the *difference is physically obvious.* This is what makes the loop work without ears.

### Component B — real-time performance assessment (microphone)
As you play, a mic (or contact mic on the guitar) feeds real-time analysis comparing **what you played** to **the reference**, on three axes that map exactly to the three kinds of advice (§3):
- **Pitch** → live monophonic F0 (pYIN / YIN; CREPE offline, truth.md §6) → *which note* you played vs the target note.
- **Timing / duration** → onset + offset detection (librosa/aubio) → *when* and *how long* vs the target.
- **Note quality** → harmonic-to-noise ratio, decay envelope, spectral flatness → *clean vs buzzed/muted* (the audio signature of bad fretting pressure).

### Component C — Vision / VLM (the "see exactly what you're doing")
A camera watches your fretting hand; this is the **visual reinforcement** that tells a Deaf learner the *physical* change (haptics alone can't).
- **Hand/finger tracking:** **MediaPipe Hands** (21 landmarks/hand, in the browser) → where each fingertip is.
- **Fretboard registration:** detect the neck/frets/strings (CV line-detection, or a one-time calibration where the user frames the neck) → map fingertips to **(string, fret)**.
- **The VLM coach (Claude vision):** a frame (or short clip) of the hand + the *known target fingering* + the detected error → **Claude generates the specific, plain-language correction and which finger to highlight.** This is a deep, defensible Anthropic integration — Claude's vision *is* the coaching brain.
- **The highlight:** on a wrong note, the UI **highlights the offending finger** on the live camera feed and shows a ghost of the correct position.

### Component D — the Physical-Correction Engine (the novel core)
The hard, original part the user called out: *deaf people won't know what physical change to make.* This module **fuses mic + vision + the reference** and emits **exactly one** of three corrections. **Keep it to these three (per Aditya):**

| # | Error category | How we detect it | The physical fix we give |
|---|---|---|---|
| 1 | **Wrong note** | mic pitch ≠ target pitch | **Highlight the wrong finger** (VLM) and show the correct fret: *"ring finger → 3rd fret"* |
| 2 | **Incorrect note duration** | measured onset→offset vs target length (too short / too long / rushed / dragged) | *"hold it longer — let it ring 2 beats"* / *"you're early — wait for the beat"* (+ a haptic metronome on the vest) |
| 3 | **Wrong pressure** | poor note quality (low harmonic-to-noise = **buzz**, fast decay/thud = **muted/dead**) **+** VLM finger-placement (too far from the fret wire / collapsed joint) | *"press harder"* / *"move your finger right behind the fret"* / *"use your fingertip, not the pad"* |

**The logic (priority order):** wrong pitch → it's a **wrong note** (fix the fingering, highlight the finger). Right pitch but off-length → **duration**. Right pitch, right length, but rough/buzzed/dead → **pressure**. One clear error → one clear fix → re-loop the segment.

> **Why this is rigorous, not hand-wavy:** every category has a concrete, measurable signal (F0; onset/offset timing; harmonic-to-noise + decay), cross-checked against vision and a known reference. Pressure-from-audio (buzz/mute classification) is the newest piece — we ship a real v1 classifier (HNR + decay + spectral flatness) and are honest it's a heuristic refined by the VLM. This multimodal **audio + vision + reference → specific motor correction** loop is the research-grade contribution.

---

## 3. Local memory — practice anywhere, offline (Redis, baked in)
*"Play your favorite song anywhere, no wifi/cellular"* — deeply integrated, and it's the **Redis** sponsor story made real:
- When you save a song, we persist **the analyzed reference + its rendered haptic signal + the correct fingering + your calibration + your progress** to a **local Redis** instance on the laptop.
- A practice session then runs **fully offline** — feel the target, play, get assessed, get corrected — with zero network. Redis is the on-device library + session memory ("Tactus remembers your songs and your body").
- Sync to cloud Redis when a network is present (so a song saved on one device appears on another) — but the **core loop never needs the internet.** This is genuine, not a bolt-on: a learning device you carry has to work in a basement practice room.

---

## 4. The interface (camera-centric, every pixel real)
**Anti-slop rule stands: if a pixel isn't driven by real audio analysis, real vision, or the real per-node drive, cut it.** Layout:
1. **The live camera feed (center)** — your fretting hand with **VLM overlays**: wrong finger highlighted in red, correct position as a ghost, (string, fret) labels. The heart of the visual reinforcement.
2. **The note highway (Rocksmith-style scroll)** — upcoming notes/fingering flowing toward a "now" line, color-coded; your play-along guide.
3. **The correction panel (appears on error)** — the rewound segment, "you played ✕ / correct ✓," the **feel-the-difference** replay button (vest), and the one-line physical fix.
4. **The body map** — vest nodes pulsing at the exact intensity being sent (what you're feeling, made visible; also the demo money-shot).
5. **The pipeline view (explainability showpiece, optional tab)** — note → deconstruction → haptics, from real spectrograms + real node data (the "we did the real DSP/ML" proof; Most-Technical catnip).
6. **The offline library** — your Redis-saved songs, calibration, progress; works with no network.

**Aesthetic:** dark, high-contrast; one consistent **color = body-zone** mapping; motion is data-driven (spectrogram, node intensities, hand landmarks). Stunning *because* it's real.

---

## 5. Architecture + parallelization (the seam)
The web app talks to the **Python engine (on the laptop)** over **one localhost WebSocket against a fixed JSON contract** — so the web owner builds independently from hour one (against a mock) and meets the ML/haptics owner at the message format. **The browser owns camera + MediaPipe + ArUco + AR + viz; Python owns mic + F0 + the fusion model + the 12-ch haptic output.** Vision features flow browser→Python, timestamped. (Locked architecture: truth.md §2.)

```
[ Python engine (laptop) ] --WS(~60Hz)--> [ Web app (browser) ]
  - per-node vest drive (12 floats)        camera capture + MediaPipe Hands (in-browser)
  - mic analysis: pitch, onset/offset,     note highway + live body map (WebGL/Canvas)
    note-quality(HNR/decay)                correction panel + feel-the-difference
  - error event {category, target,         Claude-vision coach call (frame + target → fix)
    measured, finger_to_highlight}         offline library (reads local Redis)
  - song position / segment                Deepgram voice ("rewind", "slower") — hands-free
  - (offline) reference + fingering         the pipeline/explainability view
  ◄── vision features (timestamped) ─────── (browser → Python)
        └────────────── local Redis (songs, fingering, calibration, progress) ──────────────┘
```
- **Frontend:** React + WebGL/Canvas; **MediaPipe Hands** in-browser (the $0, real-time hand tracking); Web Audio `AnalyserNode` for the live spectrogram.
- **Lock the WebSocket `frame` + `error-event` + `reference` schema Saturday morning** (the browser→Python vision-feature schema is the one still open) — that's the seam that lets the web person work in parallel with a mock while the engine comes online. **This doc (this §5 contract) is the WebSocket schema source of truth that truth.md §2 references.**
- **No-hardware fallback:** the whole app runs on the laptop against a recorded reference + webcam (vest optional), so it demos even if hardware is down — and it's the shareable artifact.

---

## 6. Sponsors this unlocks (a much deeper story)
- **Anthropic (flagship):** **Claude's vision is the coaching brain** — it looks at the hand + the target + the error and produces the exact physical fix in plain/ASL-gloss-friendly language, and highlights the finger. A real, deep, multimodal Claude integration (far beyond a chatbot). Built with Claude Code.
- **Redis (now a headline, not secondary):** on-device local memory → **offline practice anywhere** + the song/fingering/progress library. Baked into the core loop.
- **Deepgram:** hands-free voice control — *"rewind," "slower," "again"* — essential because **both hands are on the guitar.**
- **Arize / Phoenix:** trace + eval the F0 engine, the note-quality classifier, and the VLM correction accuracy; fix failure modes live.
- **Annapurna / AWS Trainium:** train the contrastive multimodal embedding (the legitimate accelerator job; cut-first stretch).
- **Prizes:** Best UI/UX (Wacom), Most-Technical (the multimodal correction engine + pipeline view), Hacker's-Choice (a judge feels themselves get corrected), Ddoski's World (a genuinely usable accessibility tool). (QNX dropped — its value was a Pi real-time loop, and the Pi is cut.)

---

## 7. Honest scope — demoable this weekend vs research-bet
**Demoable (high confidence):**
- Feel-the-target on the vest from a tab/MIDI reference; the note highway play-along.
- Mic assessment of **wrong note** (pitch) and **duration** — solid, well-understood DSP.
- The **feel-the-difference** replay (your version vs correct, on the vest).
- MediaPipe-Hands finger tracking + **highlighting the wrong finger** against a known target fingering.
- Claude-vision generating the plain-language fix from a frame + the error.
- **Offline practice from local Redis** on 2–3 prepared guitar songs.

**Research-bet (flag as v1 / future work):**
- **Pressure as the buzz inverse** — buzz `B = f(P, d)`; vision pins `d`, audio measures `B`, invert the fitted surface → a **2-class ordinal (too-light / good)** ("too hard" is a separate pitch-cents fault). We ship a real v1 (HNR + decay + spectral flatness → the surface fit); robust accuracy is the hard part. (truth.md §6, docs/20 §3.)
- **Precise fretboard registration** (exact string/fret from vision in any lighting/angle) — pragmatic version uses MediaPipe + ArUco homography + the known target + a one-time neck calibration; full robust CV is the stretch.
- Generalizing past guitar; polyphonic chords (we scope to single-line/monophonic for assessment, truth.md §6).

> **Bottom line:** *"We can teach a Deaf person to play a song"* becomes something a judge watches: they play a wrong note, the vest replays the correct feel, the screen highlights the finger that was off, and one clear instruction appears — the hearing player's "hear it, fix it" loop, rebuilt out of touch and vision. Guitar-only, offline-capable, and parallelizable so one teammate owns the whole front end.
