# 17 — The AI core: audio + vision → high-granularity (string, fret, finger, quality) → targeted feedback

> The new direction is **LEARN + PLAY only** (Experience/Express cut). That makes the *whole* product an AI-perception problem: from a **webcam + a mic** recover **what you played, where, with which finger, and how cleanly** — at a resolution neither sense alone can reach — and turn it into one precise correction you can feel and see. This doc is the rigor argument for a Most-Technical / CV / Anthropic judge, and the map of how the **interface makes that AI visible.**

---

## 0. The thesis — AI where it's *unsolved*, determinism where it's *solved*
```
audio + video ─► [ AI PERCEPTION FRONT-END ] ─► symbolic (string,fret,finger,quality)
                  transcription · vision ·                    │
                  buzz-inverse · FUSION                        ▼
                                                  [ DETERMINISTIC RENDERER ]──► body (12 ch)
                                                   encoder + pulse synth (docs/07/21)
```
The note→vibration rendering is a direct, measurable signal→vibration transform, **and that is correct** — there is **no generative "haptic score"**; the hard, unsolved problem is *understanding arbitrary guitar playing from sound + video.* We put the ML exactly there and nowhere else. **That split is the rigor** — no black box "the AI made a vibration," every felt cue traces to a measured signal. (Mirrors `docs/11`, deepened.)

---

## 1. High-granularity POSITION — why fusion beats either modality alone

**The core difficulty:** a guitar is *redundant* — the same pitch lives at many `(string, fret)` positions (an E4 sits in ~4–6 places). So:
- **Audio alone** recovers *pitch*, but is **blind to which position** produced it, and blind to *which finger*.
- **Vision alone** recovers *hand geometry* (which string/fret/finger), but is **blind to whether the note actually sounded** and to its *quality*.

Neither is enough. We **fuse** them, and the fusion is the technical contribution.

### 1a. Audio branch (what + how-clean)
- **Live, monophonic (the real-time "feel-it-now" path):** **pYIN / YIN** → F0 contour, cents-from-target, onsets. ~20–50 ms — the latency-safe path (truth.md §2, `docs/15`).
- **Offline / per-phrase, polyphonic:** **CREPE** (and Spotify `basic-pitch` for chords) → note events `(pitch, onset, duration)`. Per phrase, not per note — so it runs against a *known target song* for the polished run.
- **Pitch → candidate positions:** map F0 → the **set** of `(string, fret)` that produce it under the current tuning, then pick the most playable with a small **cost model / Viterbi** over positions (minimise hand span, prefer staying in the current hand zone, low frets). This is a real MIR move (guitar tablature/position estimation), and it's what turns "a pitch" into "a *position hypothesis*."

### 1b. Vision branch (where + which finger) — runs in the **browser**
- **Hand pose:** **MediaPipe Hands** (in the browser) → 21 landmarks/hand → fingertip pixels + finger identity, ~30 FPS (free, real-time). Vision features flow browser→Python timestamped (truth.md §2).
- **Fretboard registration:** an **ArUco fiducial** on the headstock + **OpenCV homography** → image → fretboard coordinates (string lines × fret lines). Robust to angle/lighting once calibrated; a one-time neck framing handles the rest.
- **Fingertip → `(string, fret)`** by grid cell; **placement vs the fret-wire** (distance from fingertip to the fret line) = coarse placement quality; collapsed-joint / finger-too-far flags from the landmark geometry.

### 1c. Fusion (the part that's a real contribution, not a wrapper)
1. **Time-align:** the audio **onset** timestamps the event; grab the **vision frame at that t**.
2. **Reconcile:** vision proposes `(string, fret, finger)`; the position implies a pitch `P̂`; audio measures pitch `P` and quality `Q`. **Agreement** (`P ≈ P̂`) → high-confidence note. **Disagreement is the detected mistake** — e.g. vision says fret 3 but audio hears the pitch of fret 5 → finger slipped / wrong string; vision says a fretted note but audio reads a dead thud → pressure fault.
3. **Confidence = cross-modal agreement**, calibrated; when low, defer to the modality that's reliable *in that context* (vision for which-finger/chords, audio for timing/quality).
4. **Output per note: `(string, fret, finger, placement-quality, sound-quality, confidence)`** — strictly higher granularity than audio-only transcription (no position/finger) or vision-only (no sound/quality). **The disagreement channel literally *is* the error detector** — that framing is what lands with a technical judge.

---

## 2. High-granularity TARGETED FEEDBACK — three axes, each a measured signal

Exactly three corrections (`docs/07`/`13`), each grounded, cross-checked, and rendered to **screen + finger-highlight + body**:

| # | Error | The measured signal(s) | Cross-check | The one instruction | Haptic |
|---|---|---|---|---|---|
| 1 | **Wrong note** | audio: cents-from-target; fusion: `(string,fret) ≠ target` | vision finger position | "ring → fret 3" + **red ring on the wrong finger** (Claude vision) | wrong-spot pulse in the strum sweep |
| 2 | **Duration** | onset→offset vs target length (onset detect + energy-decay offset) | beat grid | "hold 2 beats" / "you rushed" | haptic metronome on a back zone |
| 3 | **Pressure (the buzz inverse)** | buzz `B` over HNR, inharmonicity, spectral centroid/flux, attack/decay, broadband-buzz energy; **vision pins `d`** (fingertip-to-wire), so `B → P` is identifiable | vision: finger-vs-fret-wire / collapsed joint | buzz + `d` large → "slide toward the fret"; buzz + `d` small → "press harder" | (quality flag; no extra cue) |

**Priority logic:** wrong pitch → wrong note; right pitch, off length → duration; right pitch, right length, rough → pressure. One clear error → one clear fix → re-loop. **Axis 3 is the novel piece** — recovering *pressure as an inverse problem*: buzz `B = f(P, d)` is cause-blind in audio alone, but **vision pins `d`**, so inverting the fitted surface recovers **pressure as a 2-class ordinal (too-light / good).** "Too hard" is a *separate* pitch-cents fault (sharp/choked), not a buzz class. We ship a real v1 (the surface fit) and report **ordinal, never Newtons** (`docs/20 §3`, `docs/23`).

**The phrase-level coach (Claude, Anthropic flagship):** per-note feedback is rule-based (the LLM is too slow per note); **per phrase**, the structured error log → Claude → prioritized, plain-language / ASL-gloss-friendly advice ("ring finger muted the G — press just behind fret 3"). Claude's **vision** also produces the finger-highlight + fix text from `(frame + target fingering + detected fault)`. A deep multimodal integration, not a chatbot.

---

## 3. The rigor moves (cheap, high-credibility — have these ready at the table)
- **The AI/determinism split** (§0) — answer to "is the AI just making it up?": no, it's a measured signal→skin transform; the ML is only where the problem is unsolved.
- **The separability study (the rigor centerpiece, `docs/20`/`23`):** PCA→LDA → Fisher ratio / silhouette / pairwise d′ / confusion matrix, **leave-one-player-out**, run **audio-only vs vision-only vs fused** — empirically proves fusion is necessary and decides the taxonomy from data.
- **Cross-modal fusion + disagreement-as-signal + confidence calibration** — a genuine ML-systems contribution. (Cluster→advice is a **hybrid**: supervised backbone + validated discovery, gated on held-out data, `docs/23`.)
- **Position constraint solver** (pitch → playable `(string,fret)` via a cost model/Viterbi) — real MIR, not hand-waving.
- **The buzz inverse** — `B = f(P, d)`; vision pins `d`, audio measures `B`, invert the surface → ordinal pressure. Closes the "how hard did they press" gap from audio; the original bit.
- **Quantitative evaluation (do these, show numbers):**
  - **F0 / transcription accuracy** on *our own* guitar recordings; **trace every inference in Arize/Phoenix**, catch octave errors live and fix the mapping during the hackathon.
  - **Ablation:** naive "pitch → one speaker's vibration frequency" vs the **2-axis spatial code** → identification-accuracy jump (this *is* the "feel the difference" proof, quantified).
  - **Mini discrimination study (n=3 teammates):** % of notes/positions correctly identified *by feel* → one real accuracy number.
  - **Information-theoretic framing:** bits/second of musical info through the haptic channel; a confusion matrix + **Information Transfer** (vs a 1-channel buzzer's ~1–2 bits).
- **Latency discipline:** state live-monophonic (~20–50 ms) vs offline-polyphonic explicitly; never claim real-time for the heavy path.
- **Perception grounding:** every encoding choice cites `docs/12` (Weber fraction, two-point acuity, Pacinian peak) — so a haptics-literate judge sees we did the real thing.

---

## 4. Where the AI SHINES — the interface is the showcase (`docs/22`, `web/`)
The interface is not decoration; it's the **proof surface**. Anti-slop rule: every pixel is driven by real audio analysis, real vision, or the real per-channel drive.
- **Pipeline / explainability tab** — `audio+video → transcription → vision (landmarks + homography overlay) → fusion (with the live confidence + the disagreement that became the error) → encoder → 12-ch drive`, running live on the real spectrogram. **Most-Technical catnip** — it shows the whole chain at once.
- **The live 3D cluster view (`docs/20 §5`)** — a rotating LDA/UMAP scatter colored by fault, the live note dropping in, with an **audio-only ↔ fused toggle** that visibly collapses the overlapping blob into separated clusters — the multimodal-necessity argument in one gesture, all real data.
- **Live camera + Claude-vision finger highlight** — the multimodal Anthropic flagship, *watchable*: wrong finger ringed red, correct ghost, `(string,fret)` labels.
- **Body map = the real `drive[]`** (`docs/13_LEARN_WEB_AND_VISUALIZATION.md §5`) — the audience sees exactly what the wearer feels, on the real **12 sites**.
- **Correction panel = the 3 fusion outputs**, each showing its *measured* signal (cents, ms, ordinal pressure) and confidence — rigor rendered into UI.
- **Confidence meter + Arize trace view** — "we evaluated and improved the model," visible.
- **Honesty in the UI:** placement labelled *vs the fret-wire (coarse)*, pressure labelled *ordinal (too-light / good)* — we never overclaim on screen, which is itself a rigor signal.

---

## 5. Award orientation (apply only where we can win — Pi/QNX cut)
| Prize / track | Why we're strong | What we lean on |
|---|---|---|
| **Most Technical Hack** | the inverse-problem fusion + the buzz inverse + the separability proof + the explainable pipeline + a real eval | §1–§4 |
| **Best UI/UX** | the interface that makes the AI visible (`docs/22`, `web/`) | §4 |
| **Anthropic** | Claude vision = the coaching brain (frame+target+fault → fix+finger), phrase coach | §2 |
| **Annapurna / AWS Trainium** | train the contrastive multimodal embedding (the legitimate accelerator job; cut-first stretch) | `docs/20 §5` |
| **Accessibility / "Ddoski's World"** | sensory-substitution guitar coaching for Deaf/HoH — feel + see the fix | whole product |
| **Computer Vision track** | MediaPipe hands (browser) + ArUco/OpenCV fretboard homography + occluded placement | §1b |
| **Redis** | coach memory: per-user mistake history, adaptive difficulty, **vector-search "your similar past mistakes"** (on the engineered feature vectors) | `docs/11`/`23` |

**Dropped / deprioritized:** **QNX** (needed the Pi real-time loop — Pi is cut), heavy-hardware framing. **Fetch (multi-agent)** is *optional* — only if time, as a light analyze/coach split; don't contort for it.

---

## 6. Honest scope (rigor = honesty; say these if probed)
- Placement is **vs the fret-wire (coarse)**, not centimeters (audio has zero cm info; vision is coarse).
- Pressure is a **2-class ordinal (too-light / good)** recovered by inverting the buzz surface, not measured Newtons. "Too hard" is a separate pitch-cents fault.
- **Polyphonic** transcription degrades on dense/fast playing → vision cross-checks chords; scope live assessment to single-line where possible.
- State **real-time vs pre-processed** in the demo. A known target song de-risks both paths.

> One line for the judge: *"From a webcam and a mic we recover which string, which fret, which finger, and how cleanly you fretted it — by fusing vision and audio so their disagreement is the mistake — then render one correction you feel on your body and see on your hand. The AI lives exactly where the problem is unsolved; everything downstream is a measurable transform you can point at."*
