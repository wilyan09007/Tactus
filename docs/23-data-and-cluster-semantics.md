# 23 — Data, representation, training & the eigenvector→advice semantics (extends docs/20)

Status: **DRAFT** — produced in an office-hours session, extends `docs/20-aiml-training-design.md` and the eng review `docs/20-eng-review.md`. Mode: hackathon (AI Berkeley 2026).

> **Thesis:** a cluster in feature space has *no inherent meaning* — meaning is **assigned by prompted labels**, made **readable by named-feature eigenvectors**, made **trustworthy by held-out validation**, and made **actionable by causal clean/fault pairs**. We use a **hybrid** scheme: a supervised backbone (advice attaches to labels you already know) plus *validated* unsupervised discovery of error sub-families (the "families of errors" wow, gated so it never ships a confident-wrong fix).

## Decisions taken (office-hours)
| ID | Decision | Choice |
|----|----------|--------|
| OH1 | Cluster → advice | **Hybrid (supervised backbone + validated discovery).** Labels define the core classes; clustering may *discover* sub-types, but a sub-type becomes advice only after it passes the promotion gate (§6). |
| OH2 | Pressure classes | **2-class (too-light / good)** per `docs/20-eng-review` D5; "too hard" is a pitch-cents fault, not a buzz class. |
| OH3 | Representation | **Engineered named features** as the dependable core (interpretable eigenvectors); contrastive embedding stays cut-first/stretch. |

---

## 1. How a cluster becomes advice (the epistemics)
You cannot read meaning off geometry. A blob in ~50-D is just a blob. Meaning comes from five mechanisms, each also a guardrail:

1. **Labels assign meaning — we don't discover it.** Prompted capture means every point already carries its truth ("buzz from too-light pressure"). The label *defines* the class; the geometry only has to *prove the labels separate*. Advice attaches to the label:
   - `buzz-too-light` → "press harder" · `buzz-placement` → "slide toward the fret"
   This is why **LDA (supervised)** is the backbone, not k-means (unsupervised).

2. **The eigenvector is readable because the features are named.** "Collapse" = projecting the ~50-D vector onto a few **discriminant eigenvectors** (LDA = generalized eigenproblem on between- vs within-class scatter; PCA = eigenproblem on covariance). Because each input is a named physical quantity, the axis is a readable weighted sum:
   ```
   axis1 ≈ +0.71·buzz_energy − 0.62·d − 0.20·HNR + ...
           └────────────────────────────────────┘
        "buzziness not explained by placement" = the PRESSURE axis
   ```
   The loadings ARE the interpretation.

3. **Prototype (medoid) per class.** Take the real recorded sample nearest each class centroid → the canonical example you can *listen to* and inspect. (Also the Redis retrieval: nearest neighbors = "your mistake is like these past attempts.")

4. **Causal grounding, not correlation.** Record a **clean / fault pair at the same `(string,fret,finger)`** so the only change is the thing the advice targets. That proves "press harder fixes it," turning a correlation into a coachable cause.

5. **Held-out validation defines what advice is even allowed.** A class is real only if it separates on a **left-out player**. The **confusion matrix is the permission slip**: if `too-light` and `placement` overlap on held-out data, you are *not allowed* to coach that distinction — degrade to "a note buzzed, check your fretting."

```
prompted label ─► defines the class      (meaning ASSIGNED, not found)
LDA loadings   ─► names the axis         (eigenvector = Σ named features)
medoid sample  ─► the cluster exemplar   (point at it, listen to it)
clean/fault pair► proves the fix         (causation, not correlation)
held-out d′    ─► which advice is allowed (overlap ⇒ don't coach it)
```

## 2. The hybrid scheme (OH1)
- **Backbone (supervised):** the a-priori fault taxonomy from prompts → LDA → separability proof → advice per label. Demo-safe, defensible, the must-have.
- **Discovery layer (unsupervised, gated):** within each labeled class, run clustering (HDBSCAN/GMM) to surface *sub-types* (e.g., "muted" splits into "neighbor-string-damping" vs "weak-fingertip"). A discovered sub-type is a **hypothesis**, not advice, until it passes §6.

## 3. Data collection protocol
```
RIG:  fixed cam over neck · ArUco on headstock · mic at soundhole
      one Python process, monotonic clock; audio onset stamps t, grab nearest frame(s)
      clap at session start = A/V sync cross-check

LOOP: prompt exact condition ("low-E fret3 ring too-light MED-pluck")
      → player executes → foot-pedal advances
      → audio VERIFIES pitch matches (consistency check, NOT the label)
      → save (features, prompt-label, player, take, session_order, agree_flag)

RULES (from docs/20-eng-review):
  • prompt = the only label; never drop/relabel by audio — log the disagreement rate (D2)
  • INTERLEAVE conditions within a position — kills string-dulling/session drift (H4)
  • control pluck strength — so buzz ≠ pluck (D1, identifiability)

TWO PASSES:
  BREADTH (position): 6 strings × 3 frets (1,5,9), clean, fast → fret interpolation
  DEPTH (buzz/press): 2 strings × 2 frets × {clean, buzz-light, buzz-placement}
                      × many takes, pluck controlled → the response surface
```

## 4. How much data (with the why)
Binding constraint = **LDA stability on held-out data** ⇒ samples ≫ dim *after* reduction. We PCA to ~10–15 dims (~95% variance) then shrinkage-LDA.

| Component | Target | Why |
|---|---|---|
| Separability, per fault class | ~100 clean/class | ~5–10× the reduced dim for stable held-out LDA |
| 5 fault classes | ~500 | the rigor deliverable ("few hundred") |
| Position model (breadth) | 6×3 × ~15 ≈ 270 | interpolate fret within sampled range |
| Buzz surface (depth) | ~20–40 (B,d) pts / level / cell ≈ 240 | fit B-vs-d curve with a real R² |
| **Total** | **~1,000–1,500 takes, ~2–4 hrs** | matches docs/20 §7 once the grid is cut |
| Contrastive embedding (stretch) | thousands of pairs | ~1k is data-starved ⇒ cut-first |

**Honest line:** classical pipeline is feasible at hackathon scale; the deep embedding is not.

## 5. Data representation
```
ONE SAMPLE = (synced frame(s), audio window) at a prompted condition → feature vector

AUDIO  (~20-30): centroid/flux/flatness/rolloff, HNR, inharmonicity, attack, decay,
                 MFCC×13, buzz-energy ratio, ZCR, chroma, pitch-cents,
                 + PLUCK-PROXY (attack RMS / onset slope)            ← D1
VISION (~10-15): per-finger curl, joint angles, fingertip→wire d, neck pos, wrist angle
                 — ALL in FRETBOARD-relative (homography) coords, never pixels
CONCAT (~40-60) → the vector fed to PCA→LDA→cluster

LABELS (prompt): string, fret, finger, pressure∈{light,good}, fault∈{clean,buzz-light,
                 buzz-placement,muted,choked}
META:  player_id, take_id, session_order, timestamp, audio-agree flag
STORE: parquet of vectors+labels+meta; keep raw audio/frames for medoids + re-extraction
NORM:  z-score per feature (scaler fit on TRAIN only); condition on (string,fret)
```

## 6. Promotion gate — when a discovered sub-family earns advice
A sub-cluster found by the discovery layer becomes shippable advice ONLY if **all** hold:
1. **Stable:** appears across ≥2 players (not one person's quirk).
2. **Separable on held-out:** pairwise d′ vs its parent class and siblings is material on a left-out player.
3. **Causally labelable:** you can describe the physical cause and a one-line fix, and ideally show a clean/fault pair.
4. **Interpretable axis:** the LDA loadings that separate it read as named features, not noise.
If any fail → keep it folded into the parent class (coach the parent-level advice). Log rejected sub-families (don't silently drop — they're future work).

## 7. Training procedure (most of it isn't deep learning)
1. **Separability study:** standardize → PCA(~95%) → LDA → Fisher / silhouette / pairwise d′ / confusion, all **leave-one-player-out**. Output = cluster-viz axes + "fusion necessary" proof.
2. **Position/occlusion model:** vision-pose → fret (prompt label). Shallow RF / small MLP. Eval **leave-one-position-out AND leave-one-player-out**.
3. **Pressure (2-class) + buzz surface:** fit `B vs d` per level per (string,fret) (quadratic); classifier (logistic/LDA) on (buzz,d,pluck,string,fret) → {too-light, good}. LOPO.
4. **Bayesian fusion:** combine component likelihoods + theory prior; "training" = calibrate the prior weight (sweep vs deliberate-wrong trials; weight ∝ 1−vision_conf) and likelihood temperatures (reliability diagram).
5. **Contrastive embedding (stretch / Trainium):** triplet/InfoNCE; the deliverable is the ablation — does it beat the engineered LDA baseline on held-out d′? (A null result is a finding.)

**Iron rules:** split by player, report by position, never random-split; fit ALL preprocessing on train folds only; report at natural base rates or class-weight; classical trains in seconds–minutes on the M4; trace inferences in Arize/Phoenix to catch octave/mapping bugs.

## 8. What this does NOT claim
- Cluster geometry alone never means anything — meaning is always anchored by labels + validation.
- Discovered families are hypotheses until they pass §6.
- Pressure is binary ordinal (too-light/good); "too hard" is a separate pitch fault.
- The deep embedding is a stretch; the engineered, interpretable core is the deliverable.

## Next build steps
1. Capture harness with the §3 rules (pluck control, interleave, prompt-only labels) — 1–2 strings first.
2. Feature extractor (§5 vector incl. pluck-proxy).
3. Separability study (§7.1) under LOPO → first cluster plot + confusion matrix.
4. Wire the discovery layer + §6 gate (so "families of errors" is real, not theater).
5. Then position model, buzz surface, fusion (per docs/20 §9).
