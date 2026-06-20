# 20 — Eng review of the AIML training design (`docs/20-aiml-training-design.md`)

Reviewer: `/plan-eng-review` · Mode: hackathon (AI Berkeley 2026) · Branch: `refine/enclosure-build-interface`
Companion to doc 20 (design doc left untouched). Architecture held as LOCKED:
**BROWSER** = camera + MediaPipe + ArUco + AR + viz · **PYTHON** = mic + F0 + fusion + 12‑ch haptic out · localhost WebSocket, vision features → Python timestamped.

> **Verdict in one line:** the thesis is sound and genuinely defensible, but four things must change for the rigor to survive a technical judge — (1) pluck is an uncontrolled confound in the buzz inverse, (2) "audio → true fret" is ill‑posed and the relabel/drop policy leaks selection bias into your headline result, (3) the §1 capture checklist doesn't fit the §7 time budget, and (4) the live system needs an explicit two‑tier latency contract. Decisions taken in this review are recorded inline.

## Decisions taken (this review)
| ID | Decision | Choice |
|----|----------|--------|
| D1 (H1) | Buzz‑inverse identifiability | **Add a pluck/attack‑energy proxy feature + condition the surface on (string,fret).** Reframe pressure as ordinal classification, not continuous surface inversion. |
| D2 (H2) | Labeling policy | **Prompt = the only label; audio = verifier only.** Never relabel/drop by audio — report the disagreement rate. Evaluate position vs the prompt. Run separability on the **unfiltered** set. |
| D3 (H8) | Build order | **Keep §9 as written** (user is taking the full‑ambition bet knowingly). Demo‑floor insurance noted in Failure Modes, not imposed. |
| D4 (H5) | Real‑time | **Low risk on the M4 Pro, conditional on two cheap rules** (audio‑out isolation/buffer; browser worker + AR/cluster time‑share) + a two‑tier latency contract. |
| D5 (H1) | Pressure classes | **2 classes only — too‑light vs good.** "Too hard" makes *no buzz*, so it's detected on the **pitch‑sharpening (cents) axis**, not the buzz model. Binary matches the actionable output ("press harder" vs ok) and makes buzz **monotone in pressure** → the inverse is well‑posed. Halves the pressure data. |

---

## What already exists (reuse, don't rebuild)
- **Python→browser WebSocket contract** — `docs/13 §3` locks `frame` + `error-event` + `reference` @ ~60 Hz. Reuse. The **reverse** channel (vision features → Python) is *not* specified — see Task T9.
- **Deterministic chord/sustain renderer** — `docs/21` renders per‑string onset+envelope from tab/MIDI **with no perception ML**. This is the demo floor (Failure Modes §F0).
- **On‑body tuning plan** — `docs/18` already defines the drive‑freq / punch / coupling / per‑channel‑gain experiments. The pressure/buzz study should ride the same capture session.
- **Rigor predecessor** — `docs/17` already frames the AI/determinism split, disagreement‑as‑signal, and honest scope. Doc 20 extends it; keep them consistent (channel count in 17 is stale — Task T10).

---

## Verification plan (claim → experiment → pass bar → confound control)
Each row is a claim the doc makes. "Pass bar" is what you put on the slide; "control" is what stops it being refuted.

| # | Claim | Experiment | Metric / pass bar | Confound control (mandatory) |
|---|-------|-----------|-------------------|------------------------------|
| V1 | Buzz is cause‑blind in audio | audio‑only classify light‑buzz vs placement‑buzz | confusion matrix; **pass = they confuse** (pairwise d′ < ~1) | unfiltered set (D2); randomized capture order |
| V2 | Fusion separates the two buzz causes | fused vs **best single** modality, leave‑one‑player‑out | d′/accuracy on the pressure‑vs‑placement pair; **fused materially > best single** | matched feature dims; LOPO CV (not in‑sample) |
| V3 | Pressure identifiable given d | **2‑class ordinal (too‑light vs good)** conditioned on (string,fret), with pluck‑proxy feature | per‑position confusion; reliability diagram | buzz now monotone in pressure ⇒ better‑posed; "too hard" is a *separate* pitch‑cents fault, not in this model; control/measure pluck (D1) |
| V4 | Response surface B vs d | fit B vs d **per pressure level per (string,fret)** | R² + shape (is it quadratic in d?) + 95% CI | this is a *finding*, report shape honestly; don't over‑fit ~N samples |
| V5 | Occlusion pose→placement generalizes | **leave‑one‑position‑out** fret accuracy on the occluded subset, evaluated **vs the prompt label** | within‑range interpolation accuracy; report extrapolation separately | never random split; fretboard‑relative features only; sample neck extremes |
| V6 | Theory prior informs without dominating | sweep prior weight (fixed) **and** vision‑confidence‑adaptive; needs **deliberate‑wrong trials** | error‑detection F1 vs occluded‑correct resolution; pick the weight maximizing both | report the blind‑spot rate: occluded + correct‑pitch + faulty |
| V7 | A/V sync is tight enough | start‑of‑session clap; timestamp offset distribution under playing motion | offset jitter < one frame period (~33 ms), ideally < 15 ms | monotonic clock both streams; re‑check after neck motion |
| V8 | Live path is real‑time | measure onset→haptic p95; ALSA underruns over 5 min; browser fps with AR+MediaPipe+ArUco (±cluster) | onset→haptic p95 ≤ ~40 ms; **0 underruns**; ≥30 fps | pYIN live (not CREPE); isolated audio‑out; cluster off during AR |
| V9 | Confidence is meaningful | reliability diagram of fusion confidence | ECE small; confidence tracks correctness | calibrate on held‑out player |
| V10 | Generalization is honest | train on N‑1 players, test on held‑out | report the train/test gap as the headline number | single‑guitar caveat stated; LOPO + LOPosO both reported |

**The one experiment to run first (unchanged from doc 20's assignment, now de‑confounded):** 1–2 strings, ~12 takes each of {clean, buzz‑light, buzz‑far, muted, choked}, **interleaved (not blocked) capture order**, prompt‑labeled, audio used only to verify. Extract features incl. the pluck‑proxy → PCA + LDA → Fisher / silhouette / pairwise d′ / confusion **under LOPO**. That proves V1 + V2 and seeds the cluster viz.

---

## Failure modes (per new codepath: realistic failure · test? · error handling? · silent?)
| Codepath | Realistic failure | Test | Error handling | Silent? | Severity |
|----------|-------------------|------|----------------|---------|----------|
| Buzz inverse (P from B,d) | pluck velocity varies → "pressure" reading is actually pluck force | V3 w/ pluck‑proxy | condition on pluck proxy; widen confidence band | **was silent** → now flagged | **P1** |
| Position teacher | "audio→fret" used as truth; audio can't disambiguate fret → eval measures nothing | V5 vs **prompt** | eval against prompt; audio = consistency check only | **was silent** | **P1** |
| D2 relabel/drop | audio‑disagreeing samples dropped → separability inflated | V1/V2 on unfiltered set | keep all; report disagreement rate | **was silent** | **P1** |
| Bayesian prior | occluded + correct‑pitch + faulty → prior dominates → fault missed | V6 blind‑spot rate | adaptive prior weight ∝ (1 − vision_conf); state blind spot | **was silent** | **P2** |
| Free‑play prior | chroma prior derived from the same note's audio → double‑counts audio | V6 free‑play arm | free‑play prior = continuity + fingering table only | **was silent** | **P2** |
| Per‑finger pressure in chords | confidently‑wrong finger callout ("press harder, ring") | V2 chord subset | **confidence‑gate**: low conf → chord‑level message | **was silent** | **P2** |
| ALSA 12‑ch out | output callback starved by F0/fusion under GIL → click | V8 underrun count | isolate audio‑out process/thread; ~30 ms buffer; pYIN live | audible | **P2** |
| Browser render | MediaPipe+ArUco+AR+cluster on one main thread → jank/freeze | V8 fps | MediaPipe/ArUco in Worker; time‑share AR vs cluster | visible | **P2** |
| A/V sync | timestamp drift under neck motion → wrong frame fused to onset | V7 | clap cross‑check; re‑detect cadence; reject stale frames | **was silent** | **P2** |
| Vision train/serve skew | train on MediaPipe‑Python, infer on MediaPipe‑JS → feature mismatch | parity test on shared clips | one extractor for train+serve, or a parity harness | **was silent** | **P2** |

**Critical gaps (no test AND no handling AND silent — as the doc stands today):** the buzz‑inverse pluck confound, the "audio→fret" teacher, and the D2 selection‑bias leak. D1+D2 in this review close all three; they are the must‑land items.

**§F0 — demo‑floor insurance (optional; you chose to keep §9).** If Sunday gets tight, the deterministic path needs **zero perception ML** and cannot die on stage: tab/MIDI → `docs/21` renderer → body, + `docs/22` virtual‑neck AR, + Claude coaching text. The perception model then layers on top as the "AI is live" proof. You're not required to build to this floor; it's the parachute.

---

## NOT in scope (considered, explicitly deferred)
- **Newtons of pressure** — no force sensor; pressure is **binary ordinal (too‑light vs good)**, report relative, never absolute. "Too hard" is out of the buzz model — it lives on the pitch‑cents (sharp/choked) axis (D5).
- **Exact cm placement** — `d` is fingertip‑to‑wire distance (coarse longitudinal), not metric. (kept)
- **Per‑string buzz attribution from one mic in dense chords** — physically hard; use vision‑led heuristic + global‑audio confirm, confidence‑gated. Hexaphonic/piezo pickup would solve it but is out of scope (hardware).
- **Contrastive multimodal embedding** — cut‑first; data‑starved at ~1k samples, unlikely to beat the engineered LDA baseline. Attempt only as an ablation if everything else lands. **Redis is decoupled from it** — index the engineered feature vectors for "your similar past mistakes" and the sponsor track is won cheaply (Task T8).
- **High‑neck (fret ≥ 9) generalization** — only claim it if you collect samples there; the pose model interpolates within the sampled range, not beyond.
- **Natural‑beginner fault transfer** — acted faults ≠ a real beginner's; collect a small natural sample as a sanity test only, don't claim learning‑curve results.
- **Real‑time high‑quality source separation / neural audio→haptic *beating* DSP in 48 h** — research bets (consistent with `docs/04`), not the demo.

---

## Implementation tasks
Synthesized from the findings. P1 blocks the rigor claim; P2 should land this weekend; P3 is cleanup.

- [ ] **T1 (P1, human ~1h / CC ~20m)** — Capture harness: pluck‑proxy — add attack‑energy/RMS (and optional picking‑hand landmark) per take; prompt enforces "medium pluck."
  - Surfaced by: H1 (D1). Files: `software/ai/capture/`, feature extractor.
  - Verify: V3 — pressure classes separate per (string,fret) with pluck held; ablate the proxy to show it matters.
- [ ] **T2 (P1, human ~30m / CC ~10m)** — Labeling policy: prompt‑only label; audio = verifier; **stop dropping/relabeling**; log prompt‑vs‑audio disagreement rate.
  - Surfaced by: H2 (D2). Files: capture/labeling module.
  - Verify: dataset retains audio‑disagreeing samples; disagreement‑rate report exists.
- [ ] **T3 (P1, human ~30m / CC ~10m)** — Position eval against the **prompt** label (not "audio‑derived fret"); fix the §6/§1 wording in doc 20.
  - Surfaced by: H2. Files: eval harness, `docs/20` text.
  - Verify: V5 runs vs prompt; "audio→fret" language removed.
- [ ] **T4 (P1, human ~1h / CC ~20m)** — Separability study under **LOPO**, on the **unfiltered** set, **fused vs best‑single** (matched dims), a‑priori taxonomy, randomized capture order.
  - Surfaced by: H4. Files: `software/ai/separability/`.
  - Verify: V1 + V2; confusion matrix + d′ with CIs.
- [ ] **T5 (P1, human ~1h / CC ~15m)** — Split data collection into **breadth** (6 strings × 3 frets spanning neck, clean) and **depth** (2 strings × 2 frets × 2 pressure levels (light/good) × many takes); reconcile §1 checklist with the §7 time budget.
  - Surfaced by: H3. Files: capture checklist, `docs/20` §1/§7.
  - Verify: estimated take count fits 2–4 hrs at 8–15 s/take.
- [ ] **T6 (P2, human ~1h / CC ~20m)** — Bayesian prior: weight = f(vision confidence) (adaptive, not fixed); collect **deliberate‑wrong trials** to calibrate; free‑play prior = continuity + fingering only.
  - Surfaced by: H7. Files: `software/ai/fusion/`.
  - Verify: V6 — F1 vs occluded‑correct resolution; blind‑spot rate reported.
- [ ] **T7 (P2, human ~45m / CC ~15m)** — Per‑finger pressure confidence‑gating: low confidence → chord‑level message instead of a specific finger.
  - Surfaced by: H6. Files: fusion + correction view (`docs/22`).
  - Verify: chord subset shows no confident‑wrong finger callouts below threshold.
- [ ] **T8 (P2, human ~30m / CC ~15m)** — Redis nearest‑neighbor on **engineered** feature vectors ("similar past mistakes"); no contrastive training.
  - Surfaced by: H9. Files: `software/ai/retrieval/`.
  - Verify: query returns plausible past attempts; sponsor demo works.
- [ ] **T9 (P2, human ~45m / CC ~15m)** — Lock the **browser→Python vision‑feature schema** (fields, rate, clock) alongside `docs/13`; define A/V sync (monotonic clock + clap) as a Saturday‑morning lock.
  - Surfaced by: cross‑cutting + H5. Files: `docs/13`/`docs/08`, WS layer.
  - Verify: V7 — clap offset < frame period.
- [ ] **T10 (P2, human ~45m / CC ~15m)** — Real‑time hygiene: pYIN live (CREPE offline only); isolate 12‑ch audio‑out (separate process or buffered callback); MediaPipe/ArUco in a Web Worker; cluster viz off during AR play; two‑tier latency contract documented.
  - Surfaced by: H5 (D4). Files: `software/haptic/`, `web/`.
  - Verify: V8 — 0 underruns, ≥30 fps, onset→haptic p95 ≤ ~40 ms.
- [ ] **T11 (P3, human ~20m / CC ~5m)** — Fix channel‑count drift: `docs/17` (and 02/08/09/16) say 14/16; as‑built is **12** (`config/channel_map.json`, `docs/05`).
  - Surfaced by: cross‑cutting. Files: `docs/17`, `docs/08`, etc.
  - Verify: grep shows 12 consistently or an explicit "as‑built = 12" note.

---

## Diagrams

### Corrected identifiability (what's observed vs hidden)
```
HIDDEN:   pressure P (ordinal)      pluck force  (was unmodeled!)
              │                          │
              ▼                          ▼
OBSERVED: buzz B (audio) ◄── also depends on ── string, fret, action
              ▲
              │  d (vision: fingertip→wire)  ── pins ONE of the causes
RECOVER:  class(P) ∈ {too-light, good} = argmax p(·|B,d,pluck_proxy,string,fret)
                              "too hard" is NOT here — it makes no buzz;
                              it's a separate PITCH-cents (sharp/choked) fault.
   ⇒ a (string,fret)-conditioned 2-class ordinal (monotone buzz ⇒ well-posed),
     NOT a continuous surface inversion, and ONLY with pluck controlled/measured.
```

### Two‑tier latency contract (the live path)
```
mic ─► onset(~10ms) ─► pitch settle(~30ms) ─┬─► IMMEDIATE HAPTIC  (audio-only, ~40ms)  ✅ feel it now
                                            │
browser cam ─► MediaPipe+ArUco ─►(WS,timestamped)─► Python fusion ─► CORRECTION (~100ms+) ─► screen / 2nd haptic
   (~50–80ms vision latency is fine HERE; never on the immediate tier)

Isolation rule: 12-ch audio-out lives off the GIL-blocking path (own process / buffered callback).
Browser rule:   MediaPipe+ArUco in a Worker; AR and 3D-cluster time-share (never both live).
```

### Data collection — breadth vs depth (fits 2–4 hrs)
```
BREADTH pass  (position model)      DEPTH pass (buzz/pressure surface)
 6 strings × 3 frets (1,5,9)         2 strings × 2 frets
 clean only, fast takes              × 2 pressure levels (light/good) × many takes
 → pose→fret interpolation           → B-vs-d response surface + ordinal P
 INTERLEAVE condition order in both passes (kills the session/string-dulling confound)
```

---

## Completion summary
- Step 0 scope challenge: full ambition kept (D3=B); demo‑floor insurance noted, not imposed.
- Architecture review: 3 cross‑cutting issues (browser→Python schema unspecified, train/serve skew, channel drift) + H5.
- Code quality review: N/A — no implementation yet.
- Test review: verification plan V1–V10 produced; 3 critical gaps closed by D1+D2.
- Performance review: H5 — low risk on M4 Pro conditional on 2 cheap rules + two‑tier latency.
- NOT in scope: written. What already exists: written. Failure modes: 3 critical gaps flagged + closed.
- Decisions taken: D1=B, D2=A, D3=B, D4=conditional‑low‑risk.
- Outside voice (Codex): available on request — not yet run.
