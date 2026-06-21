# 27 — Chord feedback, mono↔poly transfer, and the 3D semantic viz (analysis + experiment plan)

> **Status:** live plan for the analysis/ML phase (Jun 21). Pairs with `truth.md §6`,
> `docs/23` (cluster→advice semantics), `docs/24` (collection protocol), `docs/25`
> (data/feature format), `docs/26` (Aden handoff). The pipeline + capture tooling
> live on branch **`worktree-analysis-pipeline`** (`software/ai/analysis/`,
> `software/ai/capture/`, `software/ai/vision/`).
>
> This doc is the "go crazy with rigorous testing in many dimensions / many
> labelings" roadmap. Nothing here is committed-as-final ML; it is the set of
> first-principles experiments to run the moment the data lands, and which ones we
> believe will win.

---

## 0. The pivot (what changed — read first)

Fault taxonomy is now **clean / buzz / muted** (3 acoustically-distinct classes),
**not** the old `buzz-light` vs `buzz-placement` cause-split.

- **Why:** buzz vs muted vs clean are physically different sounds (rattle / dead
  thud / clean ring) → separable on **audio alone**, leave-one-player-out. The old
  light-vs-placement split is audio-ambiguous (same rattle, different cause) and
  needed vision `d` to even attempt — dropped from the core.
- **muted** is the new third class: too little pressure / not fretted → dead note.
- This supersedes `truth.md §6` Stage-2's buzz-cause inverse problem for the audio
  side. The vision side (occlusion → position) is unchanged and is now the headline.

---

## 1. Three deliverables

1. **Quality** — clean/buzz/muted, per note and (via transfer) per chord. Audio-led.
2. **Position (the technical-award play)** — vision pose → `(string, fret)` that
   **beats raw MediaPipe under occlusion.** Self-supervised by the known chord.
3. **The 3D semantic viz** — rotating PCA→LDA cluster space, named axes, showing
   the class separations + the mono→poly transfer + the beat-MediaPipe result.

---

## 2. Data inventory

| set | what | size | feeds |
|---|---|---|---|
| single-note | clean/buzz/muted × 6 strings × frets 1–6 × ~4 passes | ~432 events | quality classifier, string-ID, the buzz/mute **primitive** |
| chord-stream | 8 chords × ~100, clean, strummed, varied pose | ~800–1000 | per-chord **μ_c** signatures, **vision occlusion/position**, clean residual baseline |
| chord-fault (small) | a few chords × buzz/mute a known string | ~50–100 | **validate** the mono→poly transfer (NOT a full library) |
| calib | 6 keyframes (`acoustic-1`) → `twin.json` | 6 | homography for vision position |

**Labels = the prompt (manifest).** Per-chord rows carry `chord_sequence` +
`cue_ms` (stream), `strings_played` / `fingers_played` / `frets`, and
`string_classes` (fault). The prompt IS the label; audio F0 / harmonic presence
only cross-checks.

```
single notes ──► buzz/mute PRIMITIVE (mono)
                        │  (collapse harmonics)
clean chords  ──► μ_c   │
                        ▼
            shared low-dim ERROR subspace  ──►  chord feedback (no per-string library)
fault chords (small) ──► validate the transfer holds
```

---

## 3. Core hypotheses (first principles)

- **H1** clean/buzz/muted separate on audio alone. *(V: LOPO d′ / confusion.)*
- **H2 (the key bet) — harmonic-residual transfer.** Buzz is broadband,
  non-harmonic; the chord's *correct* content is harmonic and **known** (the
  prior). Collapse the harmonics → the **residual** carries the fault in a
  context-free subspace → the **mono buzz axis transfers to chords**.
  *(V: a buzz axis learned on mono separates clean-vs-buzz on **held-out chord
  residuals**.)*
- **H3** muted = a **missing expected harmonic** (known-chord prior) → per-string,
  no library. *(V: per-string muted recall via harmonic-presence check.)*
- **H4 (the award)** vision pose → `(string,fret)` **beats raw MediaPipe** under
  occlusion. *(V: fret error vs raw-MediaPipe vs audio-pitch truth, LOPO +
  leave-one-pose-out.)*
- **H5** fusion (position + quality + prior) gives per-string-ish chord feedback
  **without source separation**. *(V: per-string verdict accuracy vs best single
  modality.)*

---

## 4. Pipeline

```
segment → features (audio ~28  +  vision ~13  +  harmonic-residual)
        → standardize → PCA(95%) → LDA
        → metrics (pairwise d′, confusion, Fisher, silhouette) under LOPO
        → audit (one-screen) + 3D rotating viz
```
Branch `worktree-analysis-pipeline`, `software/ai/analysis/`
(`schema · segment · features_audio · features_vision · collapse · audit ·
run_pipeline`). Known issue: `segment.py` over-fires onsets on real ringing/buzzy
notes — tune via `expected_note_count` + `beat_times_ms` (see `docs/26`).

---

## 5. Experiment matrix (multiple labelings — run them all, let one win)

| id | hypothesis | method | labels used | metric | baseline |
|---|---|---|---|---|---|
| **E1** single-note 3-class | H1 | std→PCA→LDA | `intended_class` | LOPO d′ / confusion | chance 0.33 |
| **E2** string-ID from audio | — | LDA on audio feats | `string_num` | accuracy vs 1/6 | (have: clean 0.47, buzz 0.53, muted 0.43) |
| **E3** harmonic-residual buzz primitive | **H2** | HPSS / known-pitch harmonic subtraction → residual feats → LDA | mono clean/buzz | **transfer**: held-out chord clean-vs-buzz d′ | mono-only |
| **E4** per-chord μ_c off-detection | — | Mahalanobis distance to μ_c (+Σ_c) | `chord_name` | ROC clean-vs-off | — |
| **E5** muted-string detection | H3 | harmonic presence vs known chord (basic-pitch) | `string_classes` | per-string recall | — |
| **E6** per-string buzz attribution | H5 | per-pitch comb residual + vision + prior | `string_classes` | confusion (expect weak) | chord-level only |
| **E7** vision position (THE award) | **H4** | MediaPipe pose feats → RF / small MLP → `(string,fret)` | chord shape (+ audio-pitch truth) | **fret error vs raw MediaPipe**, LOPO + pose-out | raw MediaPipe |
| **E8** fusion | H5 | Bayesian position × quality × prior | all | per-string feedback acc | best single modality |
| **E9** clustering / retrieval | — | k-means / vector search on residual embeddings | — | recurring-mistake retrieval ("you keep muting the A") | — |

**Labeling schemes to compare** (the "many dimensions"): per-event class · per-string
class · residual-space class · chord-level off-detection. Report which labeling +
feature set gives the best held-out separation; that's the one we ship.

---

## 6. The 3D semantic viz (demo centerpiece)

- **std → PCA → LDA → 3D**, rotating scatter (plotly / three.js).
- color by **clean / buzz / muted**; **axes labeled by their named-feature loadings**
  (the "geospatial semantic meaning" — e.g. axis-1 = buzz-band-energy, axis-2 = HNR).
- **Money shots:**
  1. mono buzz cluster **and** poly chord-residual buzz cluster **overlapping** in
     the collapsed error space → *visual proof the primitive transfers* (H2).
  2. position error: **ours vs raw MediaPipe vs audio-truth** → *visual proof we beat
     the baseline* (H4).
- **Rigor (what makes it not a pretty blob):** leave-one-player-out, fit ALL
  preprocessing on train folds only, named eigenvector loadings, held-out d′,
  report at natural base rates. The viz shows the **validated** space.

---

## 7. Rigor rules (non-negotiable)

Split by player (LOPO; k-fold only when 1 player) · report by position · fit every
scaler/PCA/LDA on **train folds only** · natural base rates · the prompt is the
label + F0 / harmonic cross-check · **report only what clears held-out; degrade
gracefully** (per-string → chord-level when unsure).

---

## 8. Capture tooling (what the localhost does now)

One server: `python3 software/ai/capture/serve.py` → **localhost:8765** (auto-opens
Chrome). `/calibrate` = the digital twin (4-corner markerless homography, 6 poses).
Record-page blocks:

- **core-grid** — single-note clean/buzz/muted, 6 strings × frets 1–6 (clap +
  3·2·1 count-in, FRET 1→6 cue).
- **pose-variation · pluck-sweep · muted · choked · holdout** — supporting sets.
- **arpeggio / strum** — chords played clean+buzz, per-string ground truth.
- **chord-fault** — per-string fault (buzz/mute one known string) → `string_classes`.
- **chord-stream** — continuous ~100-chord batch in ONE recording, live fretboard
  diagram (dots + finger numbers), pose change every 10, logs `chord_sequence` +
  `cue_ms` per chord for offline alignment.

1080p video + 48k mono WAV. Per-run terminal confirmation with VIDEO/AUDIO
resolution checks. Resolution/audio pills on screen (green = MacBook spec).

---

## 9. Division of labor

- **Aritro + Aylin** — vision position (beat MediaPipe) + the 3D viz.
- **Aden** — segmentation (tune onsets) + the experiment matrix + clustering/retrieval.
- **William + others** — sound-to-haptic (uses the same recordings).

---

## 10. Open risks

1. `segment.py` over-fires onsets on real notes — tune (known; `docs/26`).
2. Per-string buzz attribution precision — validate (E6), degrade to chord-level.
3. H2 transfer assumption — validate (E3) before relying on it.
4. **1 player** so far — a 2nd player's hand is the biggest generalization unlock.
5. chord-stream batch file size (~150–250 MB / 100 chords) — hand off in chunks.
