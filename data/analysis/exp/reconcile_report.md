# TACTUS — Reconciliation report (source of truth for docs/28)

**Protocol (fixed for every number below):** ONE player (aditya), ONE guitar (acoustic-1). Evaluation is **stratified k-fold (NOT LOPO)** — stated explicitly because there is only one player, so no cross-player claim is possible. **All** preprocessing (median-impute -> StandardScaler -> PCA(95%)) is **fit on the train fold only**. Numbers are reported at the natural base rate with the **chance** line beside them. Where takes are grouped (one run = correlated strums), we report **both** random StratifiedKFold and **GroupKFold-by-run** and adopt the leakage-free one.

> **Headline:** every disputed number was re-derived from the committed code on the consolidated `data/analysis/all/` data. Where my pass and the friend's pass disagreed, the FINAL number is whatever is REPRODUCIBLE on this branch — not whichever was larger. Two friend numbers (E1 0.80, E3 1.77) were worktree-specific and do NOT reproduce here; the friend's *methods* are correct, but the honest *values* are lower. One of my numbers (E1 0.62) was right. The e6 93.3% was survivorship-biased.

## The reconciliation table

| Claim | My number | Friend number | **FINAL honest** | Why |
|---|---|---|---|---|
| **E1** clean/buzz/muted audio separability (held-out acc) | 0.62 | 0.80 | **0.62 ± 0.04** (chance 0.33) | MINE was right. The committed `features_audio.py` re-extracted from raw, run through the friend's OWN PCA(95%)->LDA machinery, reproduces 0.62 exactly. The friend's 0.80 was from a worktree `features_fused.csv` that is not on this branch and does not reproduce. |
| **E3** residual buzz axis d' (clean vs buzz) | ~0.17 | 1.77 | **standalone 0.2771 / multivariate 1.0402 (held-out)** | Friend's CONSTRUCTION is correct (null expected k*f0 from the prompt), my values were low because the consolidated `comb_resid_ratio` is broken (d'=0.0124, clean≈buzz). Re-running `features_residual.py` gives a real but MODEST axis: standalone 0.2771, multivariate 1.0402. The friend's 1.77 does not reproduce here; ~1.0 does. |
| **E9** chord-ID-from-audio (deployment) | — | 0.81 (and 0.55) | **0.02** GroupKFold-by-run (chance 0.11) | 0.81/0.55 is LEAKAGE-inflated: 9/11 chord runs are single-chord takes, so run_id ≈ chord label. Holding whole runs out collapses it to ~chance (stratified here = 0.72). Chord-ID is NOT a deployable audio primitive on this collection. |
| **e6** harmonic-template fret (clean, exact) | 93.3% | — | **84.0% full-coverage** (within-1 92%); 93.3% on the 71% of runs with clean onsets | Deterministic detector -> no leakage possible. But 93.3% was measured only on the 51/72 runs where the onset finder lands exactly 6 (survivorship). Full coverage (all 144 clean notes) = 84.0% exact. |

## 1) E1 — clean/buzz/muted audio separability

- Held-out 5-fold accuracy: **0.6227 ± 0.0367** (chance 0.333, n=432, 28 audio features, PCA kept 17–18 comps/fold).
- Per-fold: [0.6897, 0.6092, 0.6279, 0.6047, 0.5814]
- Per-class recall: clean 0.625, buzz 0.486, muted 0.757
- Pairwise d' (pooled held-out): clean__vs__buzz 1.1285, clean__vs__muted 2.0096, buzz__vs__muted 1.265

**Diagnosis of the 0.62 vs 0.80 gap** (same fold protocol, varying one lever):

| Regime | Held-out acc |
|---|---|
| full_audio_28__pca__standard (friend regime) | 0.623 |
| full_audio_28__noPCA__standard | 0.669 |
| full_audio_28__pca__noscaler | 0.407 |
| harmonic_only__pca__standard (likely my-0.62 regime) | 0.435 |
| buzz_band+inharm+flatness_only | 0.521 |

**Verdict:** the full 28-feature audio set through PCA(95%)->LDA, fit-on-train, gives **0.6226** on the consolidated data — i.e. the friend's EXACT regime reproduces 0.62, not 0.80. The scaler and PCA are not the lever; the residual/harmonic-only subset is what lands at ~0.62, and even the full audio set tops out near 0.62 here. The friend's 0.80 used a different (worktree-only) feature CSV that is not reproducible on this branch. **Adopt 0.62 ± 0.04** for the LDA pipeline; a RandomForest does a bit better (0.69 random, and crucially 0.65 held out WHOLE runs — see §4), so the defensible deployable range is ~0.62–0.65. Note buzz is the hard class (recall 0.49); clean-vs-muted is strong (d' 2.0096).

## 2) E3 — residual buzz axis (clean vs buzz)

- Residual feature (correct): `features_residual.res_energy_ratio (residual / total after nulling expected k*f0)`.
- Consolidated `comb_resid_ratio` is BROKEN here: standalone d' = **0.0124** (clean residual ≈ buzz residual ≈ 0.80, no separation). This is the source of my ~0.17.
- Re-running `features_residual.py` (nulls expected k*f0 from string+target_fret): mean residual clean=0.419 < buzz=0.51 (right direction — buzz is more non-harmonic, as H2 predicts).
- (A) standalone `res_energy_ratio` d' = **0.2771**.
- (B) held-out multivariate residual buzz axis d' = **1.0402** (5-fold 1-D LDA on the 11 residual features, fit-on-train, acc 0.684).

**Verdict:** the friend's residual CONSTRUCTION is correct and beats both broken variants; the residual buzz axis is REAL but MODEST. The friend's 1.77 does not reproduce on the consolidated data (the honest multivariate value here is ~1.0402); my ~0.17 was the broken `comb_resid_ratio`. **Adopt the `features_residual.py` construction; report standalone d'~0.2771, multivariate held-out d'~1.0402.**

## 3) E9 — chord-ID from audio (leakage audit)

- n=312 chord-stream events, 9 chords (chance 0.111), 11 runs, 9 of them single-chord takes.
- StratifiedKFold (random, **leaky**): **0.718** (6.46x chance).
- GroupKFold-by-run (**honest**): **0.022** (0.2x chance).
- run_id collinear with chord label: False.

**Verdict:** confirmed LEAKAGE-inflated. The stratified score learns 'which take' because each run is one chord; whole-run holdout collapses to ~chance. The deployment-realistic chord-ID-from-audio number is **~chance** on this collection. The real audio primitive from the chord block is off-/fault-detection (friend's E4 off-detect AUC ~0.90), NOT chord identity.

## 4) Adversarial group-check + e6 fret leakage/robustness

- clean/buzz/muted (RF audio): random 0.687 | GroupKFold-by-run 0.653 | drop +0.035 (chance 0.33) -> no meaningful leak.
- string-ID (RF audio): random 0.546 | GroupKFold-by-run 0.501 | drop +0.046 (chance 0.17) -> no meaningful leak.

- **e6 harmonic-template fret** is **deterministic / parameter-free** -> leakage is impossible by construction (no fit, no train/test split). FULL-COVERAGE exact-fret (all 144 notes/class via the validated forced-N segmenter, best-effort window): clean 84.0% (within-1 92%), buzz 60.4% (within-1 88%), muted 14.6% (dead note has no pitch -> vision's job, honestly N/A). Pitched (clean+buzz) exact 72.2%.

- **The 93.3% claim is survivorship-biased.** It was measured only on the 51/72 runs (71%) where the naive onset detector lands exactly 6 onsets — the crisp-onset takes. On that subset clean is 93.3% (reproducing the claim), but on the dropped takes it is far lower. The honest, deployable number is the FULL-COVERAGE 84.0% exact / 92% within-1.

**Note on robustness:** because the detector only uses the prompted STRING (given by the tab in LEARN mode) plus the event spectrum, its accuracy does not depend on any data-fit weights — it cannot overfit a take. Its two real failure modes are honest and physical: (1) a dead/muted note carries no pitch (the muted N/A), and (2) onset mis-segmentation reads the wrong window (the coverage gap above).

## Artifacts

- `data/analysis/exp/reconcile_report.md`
- `data/analysis/exp/reconcile_report.json`
- `data/analysis/exp/reconcile_e1.png`
- `data/analysis/exp/reconcile_e3.png`
- `data/analysis/exp/reconcile.html`
- `data/analysis/exp/e6_results.json`
