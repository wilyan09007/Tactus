# 25 — Corrected data & feature processing contract (the offline pipeline codes against THIS)

Status: **ACTIVE.** Corrects the processing contract behind `docs/23` (features/training) and
`docs/24` (capture/manifest) after an adversarial pass. **Recording is NOT affected** except for
four cheap hygiene items (§7). Everything else here is an **offline** fix applied to data already
captured — so it must never block collection. If anything conflicts, `truth.md` wins; this file
governs how raw `data/raw/` becomes features + the separability headline.

> TL;DR for the person recording: **clap once per session, keep video on, confirm the WAV-lossless
> pill is green, play tight to the click.** Then collect as much as possible. The rest is Aiden/me.

---

## 1. A/V sync is RECOVERED offline, not assumed [HIGH]
The lossless WAV (AudioWorklet tap) and the video (`MediaRecorder` webm) are **two capture paths with
independent start offsets.** The manifest's `sync:"…synced by construction"` is **wrong** for WAV↔video.
Because `d` is read from the frame at each audio onset, a 50–100 ms offset = wrong `d` = corrupted headline.

**Fix (offline):**
- The webm's **Opus track is the same mic signal as the WAV** → **cross-correlate WAV ↔ webm-audio** to
  recover the WAV→webm offset to sample accuracy. Inside the webm, audio↔video are container-synced, so
  once WAV→webm-audio is known, **WAV onset → video frame** is determined.
- A **1-second clap/transient at session start** is free insurance (a sharp shared transient to lock onto).
- Pipeline writes the recovered offset per run + a sync-confidence (cross-corr peak sharpness); low
  confidence → flag the run, don't trust its `d`.

## 2. Segmentation = beat-anchored priors + F0-assigned frets (NOT onset-detect + sweep-order) [HIGH]
Buzz/muted notes have **weak, atypical onsets** → a generic onset detector **systematically mistimes the
exact classes we care about**. And labeling fret by **position-in-sweep** cascades an **off-by-one** through
the whole run if one note is missed or double-struck = silent label corruption.

**Fix (offline):**
- Use the logged **`beat_times_ms`** (you play to the click) as **onset priors**; refine locally, don't
  detect from scratch.
- Assign each note's **fret by F0** (pitch → fret under tuning), **cross-checked** against the prompted
  sweep — never by sweep position alone.
- **Flag any run where `seg_count ≠ expected_note_count`** (and where F0-fret ≠ prompted fret) → that run
  is quarantined from the headline until reviewed. Log the prompt-vs-audio disagreement rate (never drop).

## 3. The audio-only vs fused ablation runs on the SAME events [HIGH]
Some runs are `audio_only` (no video → no `d`). Comparing "audio-only runs" against "video runs" measures
**sampling, not modality** — the headline would be confounded.

**Fix (offline):** restrict V1 (audio-only) and V2 (fused) to the **vision-having subset**, and produce the
audio arm by **masking the vision block** on those same events. Matched comparison or the proof doesn't hold.

## 4. Never mix lossy and lossless audio in the buzz study [MED-HIGH]
`audio.format` is `wav-pcm16` normally but **`webm-opus` if the AudioWorklet failed** — and **Opus mangles
the buzz band** (the signal the whole Stage-2 proof rests on). Sample rate also varies (44.1 vs 48 kHz).

**Fix (offline):** segregate by `audio.format`; build buzz-separability on **WAV-only** events; flag/exclude
opus events from the buzz study (they're still fine for pitch/coarse use). **Canonicalize sample rate**
(resample to 48 kHz) before feature extraction so MFCC filterbanks don't drift.

## 5. Control pitch with string/fret as COVARIATES, not per-cell z-score [MED-HIGH]
At ~100 events/class spread over 36 cells, **per-cell normalization runs on a handful of samples → noisy**,
and starves the estimate.

**Fix (offline):** do **not** z-score within each (string,fret) cell. Standardize globally (train-fold fit),
and include **string + fret as covariates** (one-hot, or partial them out via regression) so pitch is
controlled without starving cells. This is what makes "technique, not pitch" actually true.

## 6. Separability metric = pairwise Mahalanobis d′ in the LDA subspace [MED]
3 classes → **2 LDA axes**; a univariate d′ along one axis **mis-states** separation.

**Fix (offline):** report **pairwise Mahalanobis d′** in the (≤2-D) LDA subspace, **per-axis loadings**
(the readable eigenvector interpretation, `docs/23 §1`), and the **confusion matrix**. Headline pair =
`buzz-light` vs `buzz-placement`: audio-only confuses → fused(+`d`) separates.

## 6b. Cross-validation: the RUN is the correlated unit [MED]
A run sweeps frets 1→6 in one take → its events share string-dulling / session nuisance. Random event
splits **leak**.

**Fix (offline):**
- **Leave-one-player-out (LOPO):** split by `player_id` (never random).
- **Leave-one-position-out (LOPosO):** split at the **event level grouped so a single run's events don't
  straddle train/test** in a leaky way; hold out whole (string,fret) cells. Use grouped CV (group = run)
  for player splits; for position splits, hold out the position across all runs.
- Fit **all** preprocessing (scaler, covariate model, PCA, LDA) on **train folds only**.

## 6c. Quick pins
- **Per-finger `d`** = the **prompted finger's** fingertip (manifest `finger`/`pose_variant` selects the
  MediaPipe landmark), distance to the prompted fret's wire in fretboard-relative units (coarse/relative).
- **muted/choked** are cheap external gates, thresholds pinned at bring-up: muted = energy/sustain gate
  (dead thud); choked = pitch-cents-sharp gate. Log the thresholds; they're flags, not buzz classes.
- **Markerless registration** (`software/ai/vision/fretboard.py`): homography from the fret law; per-run
  reprojection residual stored as the registration-quality number (replaces the ArUco validation).

---

## 7. What this changes for RECORDING (the only 4 things)
1. **Clap once** at the start of each session (1 sec) — sync insurance.
2. **Keep video ON** (don't use audio-only) — needed for `d` and the matched ablation.
3. **Confirm the "wav ✓ (lossless)" pill is green** before a batch (Chrome). "webm only" → buzz band degraded.
4. **Play tight to the click**, gain **amber, never red**.

Everything else above is offline and applied to data you've already recorded. **Do not let any of it delay
collection.** Get maximum rich, well-framed, prompt-labeled data now.

## 8. Build order for the offline pipeline (so the headline is correct by construction)
`ingest manifest → recover WAV↔webm offset (§1) → beat-anchored+F0 segmentation w/ count audit (§2) →
format-segregated feature extraction @48k (§4) → markerless pose+d aligned to onsets (§1/§6c) →
covariate-controlled standardize→PCA→LDA (§5) → matched-modality Mahalanobis d′ + confusion (§3/§6)
under LOPO/LOPosO (§6b) → one-screen interval audit (docs/24 §8).`
