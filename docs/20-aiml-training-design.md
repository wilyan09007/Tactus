# 20 — AIML training design: multimodal guitar-technique coaching (comprehensive spec)

Status: **DRAFT — for `/plan-eng-review`**. Mode: hackathon (AI Berkeley 2026).
Extends `docs/17-ai-rigor.md` with the locked model (the buzz response-surface + Bayesian position posterior). Produced via an office-hours session (decisions D1–D8 below).

> **One-line thesis:** from a webcam + a mic, recover **where each finger is, on which fret, and how cleanly it's fretted** — by fusing vision and audio through a learned physical model — and recover the **unobservable** quantity (finger pressure) as an *inverse problem*. The AI lives exactly where the problem is unsolved (reading an occluded fretting hand and a cause-blind buzz); everything downstream is a measurable transform.

---

## 0. The eight locked decisions
| # | Decision | Choice |
|---|---|---|
| D1 | Supervision | **Self-collected deliberate-technique dataset**, multimodal (synced video+audio). |
| D2 | Labeling | **Prompted + audio-verified** — the prompt is the intended label; an independent audio check must agree or the sample is relabeled/dropped. Dual-witness labels, balanced by a capture checklist. |
| D3 | Vision | **Hybrid:** MediaPipe Hands + **markerless** OpenCV homography (fret-law; ArUco optional, `software/ai/vision/fretboard.py`) → fretboard-relative pose [deterministic]; read fret directly when visible; when occluded (common), a **trained model maps visible hand-pose → occluded (string, fret, sub-fret, per-finger quality)**, with **audio (pitch→fret) as the free teacher.** Do **not** rewrite hand tracking. |
| D4→D6 | Fault model | **Buzz is cause-blind in audio** (empirically: a pressure-buzz and a placement-buzz sound the same). So model **buzz magnitude B = f(pressure P, wire-distance d)**; vision measures **d**, audio measures **B**, **invert the fitted surface to recover P.** Pressure is an *inverse problem*, not a sensor reading. |
| D5 | Validation | **Separability study** (LDA/PCA + Fisher ratio, silhouette, pairwise d′, confusion matrix) across audio-only / vision-only / fused — empirically proves fusion is required and decides the taxonomy from data. |
| D7 | Representation | **Both:** engineered-feature core (LDA/UMAP) for the dependable model + the **live 3D cluster viz**; a learned **contrastive multimodal embedding** as the research-grade stretch (+ Redis nearest-neighbor retrieval). Embedding is cut-first. |
| D8 | Theory | **Bayesian position posterior with a theory prior:** `P(pos|obs) ∝ P(pose|pos)·P(pitch|pos)·P(pos|context)`, where context = **the intended chord/note from the sheet music** (chroma fallback in free-play) + standard fingerings + hand continuity. Strong likelihood overrides the prior → that override **is** the error detection. |

---

## 1. The data pipeline (capture + labeling)

**Every sample = a synchronized `(video frame(s), audio window)` pair** at a known, prompted condition.

**Capture rig.** MacBook front camera on the neck; **markerless fretboard registration** (fret-law homography, off-the-hand fretboard frame survives occlusion; ArUco optional validator only); mic clipped inside the body near the soundhole. One Python capture process timestamps both streams on a monotonic clock; the **audio onset** stamps the event time, and the nearest video frame(s) are grabbed. (A clap/transient at session start cross-checks A/V sync.)

**Prompted, dual-witnessed labeling (D2).** The app prompts an exact condition — e.g. *"low-E, fret 3, ring finger, pressure TOO LIGHT"* — so the **prompt is the intended label** (no manual annotation). Then an independent **audio check** (buzz/HNR/decay) must agree; disagreements are relabeled by the audio outcome or dropped. Classes stay balanced because the capture **checklist** dictates what's played.

**Coverage checklist** (per the capture plan): a representative neck subset (e.g. open + frets 1/3/5/7 across all 6 strings) × conditions {clean, buzz-light-pressure, buzz-too-far-back, muted/dead, choked-sharp} × the responsible finger. ~10–15 takes each → a few hundred to ~1–2k labeled multimodal samples (enough for the classical core + the response surface; the embedding wants the upper end).

---

## 2. The vision stack (D3) — seeing through the hand

```
camera ─► MediaPipe Hands (21 landmarks/hand)  ─┐
       ─► OpenCV markerless homography ─────────┘─► hand pose in FRETBOARD-RELATIVE coords
                                                     │
   fingertip + local grid VISIBLE? ── yes ──► read (string,fret) deterministically
                                    ── no  ──► TRAINED pose→placement model:
                                               features {per-finger curl, joint angles,
                                               fingertip→wire distance d, neck position,
                                               wrist angle} → (string, fret, sub-fret d,
                                               per-finger quality)
                                               TEACHER at train time: audio pitch → true fret
```
- **Why learned:** the hand occludes the contact point; the visible pose still encodes the answer; audio gives the label for free, so occlusion is a *supervised* problem, not a blocker. This is the defensible AI contribution.
- **Fretboard-relative frame** is what makes it generalize across camera angles — train and infer on pose expressed in homography coordinates, not pixels.
- **Per-finger d** (fingertip-to-fret-wire distance) is the single most important engineered feature: it's the disambiguator for buzz cause (§3) and the placement-quality signal.

## 3. The audio stack + the buzz inverse (D4→D6)

**Audio features per note (the engineered representation, all citable):** spectral centroid / flux / flatness / rolloff, **harmonic-to-noise ratio**, **inharmonicity**, attack time, decay rate, **MFCCs**, a **broadband buzz-energy ratio**, zero-crossing rate, and **chroma** (pitch/harmony context). F0 (YIN/pYIN/CREPE) gives pitch → fret.

**The buzz inverse — the research-grade core:**
- Buzz `B = f(P, d)` is monotone (↓P→↑B, ↑d→↑B), plausibly **quadratic in d** (characterizing the curve is a real finding).
- Audio alone gives `B`, which depends on **both** P and d → **underdetermined** (this is *why* a pressure-buzz and placement-buzz are confusable from sound).
- **Vision pins d**; then `B → P` is **identifiable.** Recover **P** by inverting the fitted surface.
- **Ground-truth honesty:** no force sensor → P is supervised only by **ordinal intent** (too-light / good / too-hard). So `f` is fit as `B` vs `d` per pressure level, and inference is "given measured `(B, d)`, which pressure level best explains it." **Report P as ordinal/relative, never Newtons.**

**Feedback decomposition (the granular advice):**
1. Posterior says wrong fret/note → *"wrong note — ring finger to fret 3."*
2. Buzz + **d large** → *"slide toward the fret"* (placement).
3. Buzz + **d small** → *"press harder"* (pressure).
4. Muted/dead → *"press firmer / clear the neighbor string."*
5. Sharp/choked → *"ease off — you're bending it."*
6. Timing (onset/offset vs beat) handled on a separate axis.

## 4. The fusion model (D8) — one Bayesian framework

```
P(position, fault | V, A, C) ∝ P(V | position,fault)   ← learned pose likelihood (occlusion model)
                              · P(A | position,fault)   ← audio: pitch→fret (sharp), buzz→quality
                              · P(position | C)          ← THEORY PRIOR
   C (context) = intended chord/note from the sheet music  (chroma-derived chord/key in free-play)
                 + standard-fingering table + hand-position continuity
```
- **MAP estimate** = what they actually played (position + per-finger fault).
- **Error detection = divergence** between the posterior (played) and the prior mode (intended). The dimension of divergence picks the feedback category.
- **Safeguard = Bayesian behavior:** a confident vision+audio likelihood overrides the prior, so playing the "wrong" chord is detected, not hidden. Tune the prior weight so it informs without dominating.

## 5. Representation, embedding, and the live 3D cluster viz (D7)

- **Core (reliable):** the ~40–60 engineered audio+vision features; **LDA** (supervised, maximizes between/within-class scatter — the principled "axis along which a fault collapses") + **UMAP** for the cluster geometry.
- **Stretch (research-grade, cut-first):** a small **contrastive multimodal embedding** (triplet/InfoNCE) that pushes faults apart; report whether it **beats the engineered baseline** (an ablation = a result), and power **Redis vector retrieval** ("your mistake is nearest these past attempts").
- **★ The live 3D cluster view (visual centerpiece):** a rotating WebGL scatter on 3 LDA/UMAP axes, points colored by fault, the **live note dropping in as a glowing point** that animates into its cluster, with a **audio-only ↔ fused toggle** that visibly **collapses the overlapping blob into separated clusters** — the entire multimodal-necessity argument in one gesture, all real data.

## 6. Evaluation protocol (what we put on the slide)
- **The headline ablation:** identification accuracy **audio-only vs vision-only vs fused** → fused wins (especially on pressure-vs-placement). A confusion matrix shows audio-only confuses the two buzz causes; fusion separates them.
- **Separability (D5):** Fisher ratio, silhouette, pairwise **d′** per fault pair, per modality.
- **Position accuracy** vs the audio-derived ground-truth fret (incl. the occluded subset — the hard case).
- **Pressure-level accuracy** (ordinal) + the **response-surface fit** (R², shape — is it quadratic in d?).
- **Generalization:** collect from 2–3 players, **hold one out**, report the train/test gap honestly (single guitar is a stated limit).
- **Calibration:** are the model's confidences meaningful (reliability diagram). Trace inferences in **Arize/Phoenix**; catch octave/mapping errors live.

## 7. Data-collection logistics
- **2–3 players, 1 guitar** (state the single-guitar generalization caveat).
- Prompted-capture harness; ~10–15 takes × the coverage checklist → **~few hundred to ~1.5k samples**, ~**2–4 hrs**.
- Balanced by construction; audio-verified; held-out player reserved.

## 8. Honest scope + failure modes (rigor = honesty)
- **Pressure is ordinal/relative**, not Newtons (no force sensor).
- **Placement is fingertip-to-wire distance** (coarse longitudinal), not exact cm.
- **Per-finger is clean for single notes**; for chords it's **vision-attributed + audio-confirmed** (best-effort — the finger with worst geometry owns a detected buzz).
- **Polyphony degrades** dense/fast playing → vision cross-checks; scope live assessment toward single-line where possible.
- **Single guitar/player** → flag overfit; the held-out-player number is the honest generalization claim.
- State **real-time vs pre-processed** in the demo.

## 9. Build order (MVP slices — for the ENG review to sequence)
1. **Prompted-capture harness** + collect a small set on one or two strings.
2. **Feature extraction + the LDA separability study (D5)** → confirm fusion necessity, get the first cluster plot.
3. **Position:** deterministic-when-visible + audio pitch→fret; then the **occlusion pose-model** on the collected data.
4. **Buzz response surface** fit + inverse → ordinal pressure.
5. **Bayesian fusion** + the **theory prior** (intended chord from a loaded tab).
6. **The viz suite:** live 3D cluster view (centerpiece) + response-surface view + pipeline ribbon.
7. **Stretch:** contrastive embedding + Redis retrieval + the "embedding beats features" ablation.

## 10. Open questions for ENG review
- A/V **sync** method + tolerance (timestamp alignment under playing motion).
- **Homography robustness** while the neck moves; re-detect cadence; do we need IMU/optical-flow stabilization?
- **MediaPipe accuracy on guitar hands** (heavy self-occlusion) — does it need augmentation/fine-tuning, or is raw landmark confidence enough as a feature?
- **How many (string,fret) positions** to cover for the pose-model to generalize across the neck without collecting all 6×12.
- **Prior-weight calibration** (theory vs likelihood) — how to set it so deviations are caught but occlusion is still resolved.
- **Per-finger attribution in chords** — is vision geometry enough, or do we restrict per-finger claims to single notes for the demo?
- **Real-time budget** for fusion + the WebGL viz on the laptop alongside MediaPipe + F0.

## 11. Award alignment (why this wins, `docs/19`)
Most-Technical (the inverse-problem fusion + the separability proof + the response surface), Best UI/UX (the live 3D cluster + response-surface views), Anthropic (Claude turns the posterior + fault into the plain-language fix), CV track (pose→occluded-placement), Redis (embedding retrieval / mistake memory), Accessibility (the whole point).

---

### The assignment (do this next, you have the guitar)
Build the **prompted-capture harness** and run the **D5 separability study** on 1–2 strings: record ~12 takes each of clean / buzz-light / buzz-too-far-back / muted / choked, extract the audio features, run **PCA + LDA**, and read **Fisher ratio + silhouette + pairwise d′ + a confusion matrix**. That one experiment (a) confirms your "buzz is cause-blind in audio" finding quantitatively, (b) decides the taxonomy from data, and (c) produces your first real cluster plot — the seed of the centerpiece viz.
</content>
