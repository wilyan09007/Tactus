# 28 — Analysis results & methodology (the rigor + the story)

> **Status:** live results doc for the Tactus offline-analysis phase (Jun 21). Pairs
> with `docs/27` (the experiment plan), `truth.md §6` (the AIML direction), and the
> code in `software/ai/analysis/`. Every number here is from the pipeline on the
> real captured data; preprocessing is fit on train folds only; one player → k-fold
> with the generalization caveat stated. **The honesty is the credibility.**

---

## ★ Culmination — two analyses reconciled (read this first)

This doc reflects the **merge of two independent offline analyses** run in parallel.
Where they disagreed, the reconciled value is the one that **REPRODUCES on the
committed data**, not the larger one — full audit in
[`data/analysis/exp/reconcile_report.md`](../data/analysis/exp/reconcile_report.md).
Judge-facing landing page: [`data/analysis/exp/index.html`](../data/analysis/exp/index.html).

**Reconciled headline numbers** (held-out; one player → k-fold + GroupKFold-by-run; fit-on-train-only):

| result | FINAL honest | note |
|---|---|---|
| **Audio fret** (string-conditioned harmonic-template) | **84% exact / 92% within-1** on clean (naive F0 **42%**) | deterministic; string prior → 8 candidate frets → no octave errors. *This is the fret answer — vision can't (last row).* |
| clean / buzz / muted (audio) | **0.62–0.69** (chance 0.33), leakage-robust 0.687→0.653 | corrects an unreproducible 0.80; d′ clean·muted **2.0** |
| mono→poly buzz transfer (H2) | multivariate held-out d′ **1.04** | corrects 0.17 (broken feature) and 1.77 (unreproducible); the `features_residual.py` construction is the correct one |
| chord-ID from audio | **0.02** GroupKFold-by-run (chance 0.11) | corrects a leaky 0.81 — chord-ID is NOT a deployable audio primitive; use the tab prior + off-detection (AUC **0.90**) |
| vision per-finger position | **capture-limited (honest negative)** | hand occludes the neck; calibration homography never registers gameplay frames; per-frame re-reg 0% locked. Fix = camera angle, not code. |

**New pillars added in the culmination** (code in `software/ai/analysis/exp/`, artifacts in `data/analysis/exp/`):
- **Audio fret detector** — `e6_audio_fret.py` + `features_pitch.py` (the string-conditioned harmonic-template; the high-accuracy fret signal the vision side cannot provide here).
- **Redis as the device's nervous system** — `redis_engine.py`: the classifier IS a **0.75 ms** RediSearch-KNN query, a **semantic cache** of the Claude coach (**~20% LLM calls saved**), recurring-mistake memory, and a ReJSON + TimeSeries skill profile (`redis_engine_report.md`).
- **Eigenvector + clustering viz** — `viz_eigen.py`: scree, named-loadings heatmap, rotating LDA, clustering vs a permutation floor, transfer money-shot (`viz_eigen_gallery.html`).
- **Methodology + literature review** — `methodology_review.md` (GuitarSet / TapToTab / FretNet positioning + the strongest honest framing).

The detailed sections below are the original methodology; where a number there predates
reconciliation, **the table above supersedes it.**

---

## 0. What we set out to prove

Tactus reads a guitar player's fretting hand and the cleanliness of each note from a
**webcam + a mic**, then turns it into one correction. Two unsolved perception
problems carry the AI:

1. **Quality** — is a note/chord *clean*, *buzzed*, or *muted*? (audio-led)
2. **Position** — where is each finger, even when the hand **occludes** the contact
   point? (vision-led, beat raw MediaPipe)

Plus the bet that makes chord feedback tractable without a per-chord fault library:
**a buzz primitive learned on single notes transfers to chords** through the
non-harmonic residual (H2).

This doc reports, with held-out numbers and named geometry, how far each got.

---

## 1. The data & the integrity discipline

**Dataset (one player `aditya`, one guitar `acoustic-1`, 83 recordings):**

| set | recordings | events | what |
|---|---|---|---|
| single-note | 72 (session `2026-06-20-2332`) | **432** | clean/buzz/muted × 6 strings × frets 1–6, 24 runs/class |
| chord-stream | 11 (`0145`,`0244`,`0249`,`0305`) | **600** | Am/C/G/D/E/A/Dm/F/Em, strummed, pose varied per ~10 strums |
| calib | 6 keyframes | — | `acoustic-1` digital twin (homography) |

**Segmentation was the linchpin — and it was broken.** A blind
`librosa.onset_detect` over-fired ~4–5× on ringing/buzzy notes (206 onsets for 40
strums; docs/26). The fix: **pin segmentation to the manifest's own ground truth**,
since the capture tool cues every note/strum to a metronome and records the count:

- **single notes** → take the `expected_note_count` (=6) strongest onsets, ≥0.5 s
  apart, time-ordered → `frets[k]`. Result: **432 events, exactly 6.00/run, 0 runs off.**
- **chords** → one event per `chord_sequence[k].cue_ms` (strongest onset in
  [cue−0.15, cue+0.45] s), labeled with that strum's chord. Result: **the 206→40
  over-fire is gone; every strum gets its true chord.**

**A data-integrity bug we found and fixed (this is the rigor):** chord-stream
`run_id`s repeat across sessions (`chordstream_aditya_001` exists in all four chord
sessions), so `event_id = run_id#k` **collided** — 220 duplicate ids. A naive
key-join on `event_id` cross-multiplied chord rows (features_fused → 4512 rows from
1032). We caught it via a row-count sanity check, **made `event_id` globally unique**
(`session::run_id#k`), and rebuilt. Single-note experiments were never affected
(their ids are unique); the chord analyses pair positionally / on the fixed ids.
*Finding and killing this before it reached a result is exactly the discipline we
claim.*

**Labels = the prompt.** Audio F0 / harmonic content only cross-checks. On *clean*
single notes the measured F0 agrees with the prompted fret **70.7%** of the time
(buzz 32%, muted 17% — dead/buzzy notes have no clean pitch, as expected), which is
our segmentation sanity signal, not a label source.

**One correction baked into ground truth:** the played **G is the 4-finger voicing**
`[3,2,0,0,3,3]` (B and high-e fretted), not the capture tool's stored 3-finger
open-B shape — corrected everywhere a G is labeled.

---

## 2. The pipeline (reproducible, end-to-end)

```
segment → features (audio 26  +  vision 13  +  harmonic-residual 11)
        → standardize → PCA(95%) → LDA  → d′ / confusion / Fisher / silhouette (k-fold)
        → 3D semantic viz  +  Redis vector memory
```

`software/ai/analysis/`: `schema.py` (frozen contract), `segment.py`,
`features_audio.py`, `features_vision.py`, `features_residual.py`, `collapse.py`,
`run_pipeline.py`. Vision runs in a **separate Python 3.11 venv** (mediapipe has no
3.14 wheel; pinned to 0.10.21 for the `solutions.hands` API) and is merged back as a
CSV — the analysis/serve split mirrors `truth.md §2`. Re-runnable from raw in ~10 min.

---

## 3. E1 — clean / buzz / muted separability (the rigor centerpiece)

**Audio alone separates the three quality classes.** 432 single notes (144/class),
standardize → PCA(95%, 18 comps) → LDA, fit on train only.

- **Held-out accuracy 81.2% (5-fold), 86.2% (80/20 holdout)** vs **33.3% chance.**
- Pairwise d′: **clean·buzz = 1.93, clean·muted = 3.70, buzz·muted = 1.81.**
- Confusion: clean is cleanly separable (recall 0.84–0.97); the fuzzy boundary is
  **buzz↔muted** (buzz recall 0.67, leaking into muted) — physically sensible (a
  hard buzz and a dead note share broadband, low-harmonic energy).

**Named axes (the "geospatial semantic meaning").** Feature-space loadings
`W = pca.componentsᵀ @ lda.scalings`:

- **LDA-1** ← `hnr`, `mfcc_1`, `chroma_peak` — harmonicity / timbre.
- **LDA-2** ← `chroma_peak`, `spec_flatness`, `mfcc_6` — pitch-clarity vs noisiness.
- 3rd view axis (top residual PC) ← `spec_centroid`, `spec_rolloff`, `spec_bandwidth`
  — spectral brightness.

**The rigor centerpiece (truth.md §6) — audio-only vs vision-only vs fused.** Same
clean/buzz/muted task, three feature sets, k-fold:

| feature set | accuracy | d′ clean·buzz / clean·muted / buzz·muted |
|---|---|---|
| **audio-only** | **0.803** | 1.93 / 3.70 / 1.81 |
| vision-only | 0.382 | 0.17 / 0.52 / 0.30 |
| fused | 0.808 | 2.05 / 3.86 / 1.87 |

Reading: **quality is audio-led, and fusion does NOT improve quality** (audio ≈
fused; vision-only barely clears the 0.33 floor because the pose is
registration-limited). That cleanly validates the **two-stage split** — quality =
audio (Stage 2), position = vision (Stage 1) — rather than one blended model. It is
the honest version of the centerpiece: fusion is the right architecture *because* the
modalities answer different questions, not because vision rescues audio.
`data/analysis/exp/separability_3way.json`.

**Artifacts:** `data/analysis/exp/viz_clean_buzz_muted_3d.html` (self-contained,
offline-openable rotating Plotly scatter, axes labeled by the loadings above),
`…_3d.png`, `viz_lda_2d.png` (2σ cluster ellipses).

---

## 4. E3 — harmonic-residual mono→poly transfer (the key bet, H2)

Null the **known** harmonics (the prompt's expected pitches) → the **residual** holds
only non-harmonic (fault) energy, computed identically for mono notes and chords.

- **Mono buzz axis** (clean vs buzz, residual features, k-fold): **d′ = 1.77 ± 0.31,
  accuracy 0.83.** (`res_energy_ratio` alone d′ = 0.61 — the full residual vector
  ~triples it.)
- **Transfer (466 chord residuals projected on the mono-fit axis):** chords are
  **statistically indistinguishable from mono-CLEAN** (KS p = 0.48) and **distinct
  from mono-BUZZ** (KS p ≈ 5e-48); 98.9% land inside the mono range. A real **~10%
  buzz-side tail** carries **2.3× the residual energy** of clean-side chords — the
  axis orders chords by genuine non-harmonic content.

**Verdict: H2 supported as a geometric claim** (most strums clean → land on the clean
primitive; the buzzed minority move along the *same* axis toward the buzz primitive).
**Honest limits:** chords have no per-string clean/buzz labels (so the tail's
magnitude is uncalibrated), and it's one player. Money shot:
`data/analysis/exp/e3_transfer.png` / `.html`.

---

## 5. E9 — Redis semantic memory ("you keep muting the A")

Native **Redis Vector Sets** (VADD/VSIM, no RediSearch needed). 432 single-note
events → 28-d audio features → PCA-16 → L2-normalized → indexed with per-event
metadata.

- **Neighbor coherence vs base rate (k=10, over all 432):** class **0.676 = 2.03×**
  chance, string **0.429 = 2.58×** chance — the embedding is semantically coherent,
  not random.
- **Money query** `search_by_filter('muted', string_num=5)` → **90% muted (2.7×), 44%
  on the A string (2.6×)** — directly powers *"Tactus noticed you keep muting the A."*
  Redis as **long-term agent memory** of a player's recurring mistakes.

Module `software/ai/analysis/exp/redis_retrieval.py`; report
`data/analysis/exp/redis_retrieval_report.md`.

---

## 6. E7 / E8 — vision position vs raw MediaPipe (honest: registration-limited)

MediaPipe detects the fretting hand in **88% of frames** (907/1032) and the per-finger
model pipeline trains end-to-end (GroupKFold by recording, 1661 per-finger rows from
chord strums). But the honest result here is a **negative with a clear, fixable
cause** — and reporting it straight is the point.

**E7 — per-finger fret.** Our model beats raw MediaPipe by a mile (fret MAE
**3.81 → 0.45**; occluded **4.03 → 0.41**) — but that gap is largely an artifact: the
**calibration-pose homography does not register the gameplay video** (≈0% of
fingertips map onto the board, board-Y systematically negative), so the raw-MediaPipe
baseline is effectively un-registered and trivially bad. Against a **majority-class
floor (fret MAE 0.43)** our model only ties (0.45); it edges the floor on *occluded*
fret (0.41 vs 0.44) and on *string* accuracy (0.24 vs 0.17). We beat MediaPipe, but
not a dumb baseline — the per-finger fret signal is weak. A **signal ablation is
decisive**: vision-pose-*only* is **worse than the majority floor** (fret 0.54 vs
0.43, string 0.17 vs 0.23), and the model's entire lift over chance comes from the
**finger→fret label correlation** (finger identity alone — *zero pixels*). The
fretting-hand geometry in this footage carries no recoverable per-finger signal.

**E8 — pose→chord (the salvage).** Relative hand shape needs no absolute
registration, so we tested whether the fretting pose alone identifies the chord,
**cross-recording** (train one mixed strum-stream, test the other, GroupKFold by
session). Result: **0.167 — only 1.5× the 1/9 chance** (stratified within-pool is the
same, 0.162), because the pose features are derived from the same broken board mapping.

**Honest verdict on H4 (beat MediaPipe at occluded position): NOT supported on this
data.** The bottleneck is concrete: a single hand-clicked **calibration** homography
cannot register frames where the guitar is at gameplay angles. We then **tested** the
obvious fix — per-frame markerless registration (`e7b_register.py` on 200 gameplay
onset frames) — and it **also fails: 0% locked, median residual 62.5px**, because the
fretting hand occludes the neck at every chord onset and the blurred, oblique neck
defeats fret-line detection. So the bottleneck is **capture-side, not software**: the
real fix is a board-visible camera angle (higher / side, hand not covering the frets),
not more code. That is a one-line change to the capture rig, not a hackathon-night
model. What is real: the pipeline, the 88% hand detection, and a
precise diagnosis of exactly why the position accuracy isn't there yet.
`data/analysis/exp/beat_baseline_table.csv`, `money_shot_fret_mae.png`,
`e8_pose_chord_confusion.png`.

---

## 7. E2 / E4 / E5 — supporting experiments

**E2 — String-ID from timbre: works.** RandomForest predicts which of 6 strings a
single note came from at **0.688 accuracy (5-fold, 4.1× the 1/6 chance)**; **69% of
errors are an adjacent string** (the expected neighbor ambiguity). Timbre carries
real per-string identity. `data/analysis/exp/e2_confusion.png`.

**E4 — Per-chord μ_c & off-detection: the off-detector is the usable primitive.**
Mahalanobis distance to a chord's mean (Ledoit-Wolf covariance) flags "this strum
doesn't match the expected chord" at **AUC 0.899**. But **chord *identity* from audio
is not reliable here** — nearest-μ_c chord-ID is 0.55 under StratifiedKFold yet
**collapses to ≈chance (0.03) within a single mixed strum stream**, because the
collection has only 8 chord run_ids (7 are single-chord takes), so `run_id` is nearly
collinear with the chord label and per-fold leakage inflates the optimistic number.
We report all three CV schemes and the deployment-realistic one. Verdict:
**chord-match/off-detection yes, chord-ID-from-audio no** — consistent with "lean on
position for chords." `e4_chord_scatter.png`, `e4_offdetect_roc.png`.

**E5 — Muted/dead-note detection: works, and confirms the two muting flavors.** A
harmonic-presence score (energy at f0 + 5 harmonics / total) separates muted from
clean+buzz at **d′ = 1.39, AUC 0.82** (means: muted 0.15, buzz 0.42, clean 0.55). The
muted distribution is **bimodal** (2-component GMM beats 1 by BIC): a spike at ~0
(**body-tap percussion**) + a lobe at 0.1–0.5 (**palm-mute**) — exactly the two
muting techniques in the capture. `e5_harmonic_hist.png`, `e5_harmonic_roc.png`.

---

## 8. Rigor & honest caveats (read before believing any number)

- **Leakage-tested (adversarial).** Single-note runs are grouped (one `run_id` =
  one string+class+pluck), so random k-fold could leak a take's acoustics. Re-run
  under **GroupKFold-by-run**: clean/buzz/muted is **0.795 grouped vs 0.796 random
  (drop 0.001)** — genuinely not a memorization artifact; string-ID is **0.62 grouped**
  (vs 0.66 random). The headline survives holding out whole recordings.
  `data/analysis/exp/adversarial_groupcheck.py`.
- **One player, one guitar.** No leave-one-player-out is possible → every result is
  k-fold / GroupKFold and is **optimistic for new players**. The single biggest
  unlock is a second player's hand.
- **Preprocessing fit on train folds only**; reported at natural base rates.
- **The twin homography is from hand-clicked corners** → coarse registration; we do
  not claim sub-pixel `d`. Reported `d` is relative/coarse, never cm.
- **Chord audio-quality labels are noisy** (chords were often slightly buzzed) → we
  lean on position for chords and report chord-quality transfer as geometry, not
  calibrated accuracy.
- The prompt is the label; F0/harmonic content only verifies.

---

## 9. The story for a judge

We took a hand the camera can't fully see and a buzz the mic can't explain, and made
both **legible**: three quality classes that separate on audio alone (and you can
*see* them separate, axes named by what they mean); a buzz primitive that **transfers
from single notes to chords** through the physics of the residual; and a memory that
**recognizes your recurring mistakes**. The rigor isn't decoration — we found and
killed a data-integrity bug mid-run, we pin every claim to held-out folds, and we say
out loud where one player and a hand-clicked ruler limit us. *That* is why the numbers
are believable.

---

## 10. Adversarial review & evidence-based next steps

We attacked our own headline.

**Leakage — tested, clean.** Single-note runs are grouped (`run_id` = string+class+
pluck), so random k-fold could leak a take's acoustics. Under **GroupKFold-by-run**,
clean/buzz/muted is **0.795** (vs 0.796 random — drop 0.001) and string-ID **0.62**
(vs 0.66). The headline is not a memorization artifact.

**Where we sit vs the literature (web-checked):**
- Note **quality** (clean/buzz/muted) is **not a standard MIR task** — the field
  benchmarks transcription / tablature / chord-ID on GuitarSet, not buzz/mute. So our
  quality classifier is a *contribution*, not a race to beat a number.
- The vision gap is a **known, solved-in-principle** problem: markerless deep-learning
  fretboard + fingertip detection works from a free webcam (no fixed neck camera), and
  occlusion-robust keypoint detection is documented. Our failure was a **classical
  Hough** detector on hand-occluded frames; the SOTA fix is a **learned** fretboard
  detector (TapToTab-style) and/or a board-visible camera angle.

**Three evidence-based improvements (no new capture for #1–2):**
1. **GuitarSet (6 players, public)** → cross-player validation of the audio pipeline +
   string-ID, directly attacking our one-player caveat. (No buzz/mute labels there, so
   *quality* stays single-player — but string-ID and the feature pipeline generalize-test.)
2. **Learned fretboard keypoint detector** to replace Hough → the documented path to a
   real beat-MediaPipe position result.
3. **Re-capture with the fretboard visible** (hand not covering the frets) — the
   one-line rig fix our registration failure points to.

Sources: [GuitarSet (ISMIR 2018)](https://archives.ismir.net/ismir2018/paper/000188.pdf),
[TapToTab (arXiv 2409.08618)](https://www.arxiv.org/pdf/2409.08618),
[Guitar Tablature via Computer Vision (Springer)](https://link.springer.com/chapter/10.1007/978-3-030-33723-0_20).
