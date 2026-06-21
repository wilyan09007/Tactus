# Methodology Review — Tactus Offline Analysis (External-Reviewer Pass)

> **Role of this document.** A rigorous, citation-backed external review of the Tactus
> analysis pipeline: (1) where every Tactus result sits versus the published
> literature, (2) a methodological critique with the strongest *honest* framing for
> judges, and (3) concrete, evidence-based next steps with real citations. It is
> deliberately adversarial about our own numbers — the honesty is the credibility.
>
> Companion docs: `docs/28-analysis-results-and-methodology.md` (results), `docs/27`
> (experiment plan). Artifacts referenced live in `data/analysis/exp/`. Every number
> below was re-checked against the on-disk artifacts and (where reproducible) re-run
> during this review.
>
> **Reviewer date:** 2026-06-21.

---

## 0. Executive judgment

Tactus is two perception problems wearing one product:

1. **Note quality** (clean / buzz / muted) — **audio-led, and genuinely strong.**
   Held-out 3-way accuracy **0.803** (audio-only) vs **0.333** chance, fusion adds
   essentially nothing (**0.808**), and the headline **survives GroupKFold-by-run
   (0.795)**. This is the credible centerpiece.
2. **Fretting position** (which fret, even occluded) — **vision is capture-limited
   (honest negative), audio recovers the fret instead.** The string-conditioned
   harmonic-template detector hits **93.3% exact on cleanly-segmented clean notes**
   and **74.3% exact on the full clean set** — both far above the **16.7%** chance
   floor and the **41.7%** naive-F0 baseline. **The gap between those two clean
   numbers is a sample-selection effect that MUST be stated** (see §3.2); it is the
   single biggest honesty risk in the deck.

The intellectual contribution that is *most* defensible is **the quality classifier**:
clean/buzz/muted is **not a standard MIR benchmark task** — GuitarSet and the
transcription literature do not label buzz or mute — so we are not racing a known
number, we are proposing one. The vision honest-negative and the chord-ID-leakage
finding are **correct and should be kept**; they are what make the rest believable.

---

## 1. Where each Tactus result sits vs. the literature

### 1.1 Note QUALITY classification (clean / buzz / muted) — a genuine contribution

**What we did.** 432 single notes, 144/class, 26 audio features → standardize →
PCA(95%) → LDA, fit on train folds only. Held-out **0.803** (3-way), d′ clean·muted
**3.70**, buzz·muted **1.81** (the physically-expected fuzzy boundary: a hard buzz and
a dead note share broadband, low-harmonic energy). Source:
`data/analysis/exp/separability_3way.json`, `results.json`.

**Where the field is.** The standard MIR guitar tasks are *transcription* (note/pitch),
*tablature* (string+fret), *chord estimation*, and *beat tracking* — exactly the
annotation layers shipped with **GuitarSet** (Xi, Bittner, Pauwels, Ye & Bello, ISMIR
2018). GuitarSet provides string/fret, chord, beat and *playing-style* labels, but
**no "buzz" or "muted/dead-note" quality labels** — its hexaphonic-pickup pipeline
annotates *what was played*, not *how cleanly*. The closest published thread is
**playing-technique recognition**: Reboursière et al. (NIME 2012) classify
normal/mute/bend/slide/hammer-on/pull-off/palm-mute from hexaphonic signals, and Su
et al. (EURASIP J. Audio Speech Music Proc., 2025; arXiv 2307.07426) do real-time
*percussive* technique recognition on acoustic guitar. Both treat technique as an
*intent* taxonomy (the player meant to palm-mute), **not as an error/quality axis
(the note buzzed when it shouldn't have)**.

> **Reviewer verdict.** "Was this note clean, buzzed, or dead?" framed as a
> *correction-grade quality classifier* on a single commodity mic is **a novel framing,
> not a reimplementation.** The honest framing for judges: *"We are not beating a
> GuitarSet leaderboard — there isn't one for buzz/mute. We are showing the axis is
> real and separable (d′ up to 3.7, 0.80 held-out), and that it survives holding out
> whole recordings."* That is a contribution claim, and it is correctly scoped.

The one caveat the literature forces us to keep: **mute is bimodal** (we found a 2-GMM
fit beating 1 by BIC — a body-tap percussion spike at ~0 harmonic presence plus a
palm-mute lobe at 0.1–0.5, `e5_harmonic_hist.png`). That is consistent with the
percussive-technique literature treating body taps and palm mutes as distinct classes,
and we should *say* mute is two physical things, not one.

### 1.2 STRING-ID from timbre — consistent with, and corroborated by, the field

**What we did.** RandomForest predicts which of 6 strings a single note came from at
**0.688** (5-fold; **0.62** under GroupKFold-by-run) vs **0.167** chance, **69% of
errors adjacent** (`e2_confusion.png`).

**Where the field is.** Per-string assignment *is* a recognized sub-problem of
tablature transcription — it is the hardest part, because the same pitch lives on
multiple strings. GuitarSet exists precisely because per-string ground truth is
expensive (hence the hexaphonic pickup). FretNet (Wiggins & Kim, arXiv 2212.03023) and
the GuitarSet tablature line model string assignment jointly with pitch via
harmonic-CQT inputs.

> **Reviewer verdict.** Our **0.688 / 4.1× chance** with an **adjacent-error structure**
> is a *reasonable, in-family* result for single-note string-ID from a mono mic
> (we have less information than a hexaphonic rig). It is corroborative, not
> state-of-the-art, and is correctly reported with the GroupKFold drop (0.66→0.62).

### 1.3 Fret / tablature transcription — our 93%/74% audio-fret vs. the literature

**What we did.** Because vision position is capture-limited (§1.5), we recover the fret
from **audio**, using the prior we actually have at LEARN time: **the string is given**,
so there are only ~8 candidate frets. `features_pitch.template_fret` scores each
candidate's harmonic comb (k·f0, k=1..8) against the event spectrum and picks the best.
Measured: **clean 93.3% exact** on cleanly-segmented notes (n=90), **74.3% exact on the
full clean set** (n=144), vs naive wide-band pyin→nearest-fret at **41.7%**, vs
**16.7%** chance (`e6_results.json`, `features_pitch._selftest`).

**Where the field is.** Octave error is *the* canonical failure mode of generic F0 /
HPS pitch tracking — well documented (Tandfonline IETE 2019 on octave-error reduction;
SwiftF0, arXiv 2508.18440; the harmonic-summation line, arXiv 2509.16480) and
*especially* acute on guitar, where pickups/bodies can put more energy in H2/H3 than in
the fundamental. The standard MIR remedy is **harmonic-CQT inputs** (TapToTab; FretNet)
that bake the harmonic structure into the front-end and then *learn* the mapping.

> **Reviewer verdict — what is and isn't novel.** Template/harmonic-comb matching for
> pitch is **classical** (it is harmonic-product/sum spectrum in spirit). What is the
> right *framing* of our contribution is the **conditioning**: by injecting the
> *string prior* we collapse an open-vocabulary octave-prone pitch problem into an
> **8-way constrained classification with no octave errors by construction**. That is a
> sensible engineering use of a deployment-available prior, and it cleanly beats the
> un-prior'd baseline on the *same notes* (41.7%→93.3% clean). It is best sold as
> **"the right tool given the LEARN-mode prior,"** not as "a new pitch algorithm." And
> it must always be reported next to the naive baseline and the two-number caveat
> (§3.2), or it reads as cherry-picked.

### 1.4 Vision tablature (camera → fret) — where the SOTA is, and why we don't match it

**Where the field is.** **TapToTab** (Ghaleb et al., arXiv 2409.08618; IEEE 2024) is
the direct comparable: **YOLO-based fretboard detection** + Fourier audio analysis to
generate tabs from video, reporting large gains over classical (Hough-style) fretboard
finders precisely *because* the detector is learned and robust to viewpoint. Older
work (Kerdvibulvech & Saito; "Retrieval of guitarist fingering information using
computer vision") used hand+fretboard tracking with fiducials/condensation.

> **Reviewer verdict.** Our vision stack used **MediaPipe hands + a single hand-clicked
> calibration homography + classical fret-line detection** — i.e., exactly the
> *pre-TapToTab* design that the literature already shows is brittle to oblique,
> hand-occluded, motion-blurred gameplay frames. So our vision negative is **not
> surprising and not a bug in our reasoning** — it is the documented failure mode that
> motivated learned detectors. Framing for judges: *"the SOTA fix is known (a learned
> fretboard keypoint detector); we diagnosed exactly why the classical path fails on
> our footage and did not paper over it."*

### 1.5 Occlusion-robust hand pose — the gap is real but solved-in-principle

**What we did.** MediaPipe detects the fretting hand in **88%** of frames, but the
per-finger fret signal does not survive an honest baseline: vs a broken (un-registered)
MediaPipe board readout we "win" (fret MAE 3.81→0.45), **but vs a majority-class floor
(0.43) we only tie**, and the **signal ablation is decisive** — vision-pose-only is
*worse* than the floor; all lift comes from the finger→fret label correlation (zero
pixels). Per-frame markerless re-registration (`e7b_register.py`) **fails at 0% lock,
median 62.5px residual**. Sources: `e7_model_report.txt`, `e7b_register_report.txt`,
`beat_baseline_table.csv`.

**Where the field is.** Self-occlusion is *the* central, actively-researched hand-pose
challenge: OccRobNet (arXiv 2503.21723), occlusion-robust 3D hand from single RGB (IEEE
9511389), occlusion-aware training + test-time adaptation (ScienceDirect 2026), and
under-occlusion benchmarks (arXiv 2504.10350). These show 2D/3D keypoint detection can
be made occlusion-robust with attention/GCN architectures and occlusion-aware training
— but **none of them register to a guitar fretboard**, and all assume the *hand* is the
target, not a 6×~12 metric grid behind it.

> **Reviewer verdict.** Our negative is **honest and correctly root-caused** to a
> *capture-side* failure (a calibration-pose homography cannot register gameplay-angle
> frames; the hand occludes the neck at every chord onset, defeating classical
> fret-line detection). This is the right call. The literature says the *modeling* side
> is tractable (learned, occlusion-robust keypoints) **if** the camera can see the
> board — which is exactly the one-line rig fix we identify.

### 1.6 Harmonic-residual mono→poly transfer (H2) — a clean physics-grounded bet

We null the *known* (prompted) harmonics; the residual carries only non-harmonic
(fault) energy, computed identically for mono notes and chords. The mono buzz axis is
**d′ 1.77 / acc 0.83**; 466 chord residuals projected on the mono-fit axis are
**KS-indistinguishable from mono-CLEAN (p=0.48)** and **distinct from mono-BUZZ
(p≈5e-48)**, with a **~10% buzz-side tail at 2.3× clean residual energy**
(`e3_report.json`, `e3_transfer.png/html`).

> **Reviewer verdict.** This is a **geometric/physical** claim, not a calibrated
> accuracy claim, and it is *correctly labelled as such* in docs/28. There is no direct
> public benchmark for "single-note fault primitive transfers to chords," so this is an
> original, falsifiable hypothesis with a clean test. Keep the caveat front-and-center:
> **chords have no per-string clean/buzz labels**, so the tail magnitude is uncalibrated.

### 1.7 Redis vector memory — an application, not a research claim (correctly)

432 events → 28-d audio → PCA-16 → L2 → native Redis Vector Sets. Neighbor coherence
**class 0.676 = 2.03× chance**, string **0.429 = 2.58×**; money query (muted on A
string) **90% muted / 44% on A** (`redis_retrieval_report.md`). This is a sound
*product* demonstration of agent long-term memory and is not over-claimed as novel IR.

---

## 2. Methodological critique (the rigorous pass)

### 2.1 Cross-validation: k-fold-not-LOPO is the correct, honestly-stated choice

There is **one player, one guitar, one camera**, so leave-one-player-out is
*impossible*. The team uses stratified k-fold and, critically, **GroupKFold-by-`run_id`**
as the leakage stress test. This is exactly the discipline the MIR literature demands:
the **"album/artist effect"** (Pampalk; and the FMA dataset's *artist filter*, arXiv
1612.01840) shows that letting the same artist/recording appear in train and test
**inflates accuracy** by letting the model memorize recording-specific acoustics. By
grouping on `run_id` (one take = one string+class+pluck) and showing the headline holds
(**0.795 grouped vs 0.796 random**), Tactus *passes its own album-effect test*.

> **Critique / required framing.** k-fold here measures **generalization across
> recordings of one rig**, NOT across players or guitars. Every number is therefore
> **optimistic for a new player.** Docs/28 says this; it must stay the loudest caveat.
> The single highest-value experiment is a *second player's hand* (see §3).

### 2.2 Fit-on-train discipline — correct, with one thing to double-check

Standardize → PCA(95%) → LDA is fit on train folds only; that is right. The one item a
reviewer flags: confirm the **PCA component count is re-fit per fold** (95%-variance can
select a different #comps per fold) rather than fixed from the full set, and that the
**LDA/Ledoit-Wolf covariance** (E4) is never touched by test rows. From the code path
this appears correct (`collapse.py` pipeline), but the deck should state it explicitly:
*"all of {scaler, PCA basis + rank, LDA scalings, μ_c, covariance} are estimated inside
the train fold."*

### 2.3 The audio-fret 93% vs 74% — the sharpest honesty issue (re-verified live)

This review re-ran both code paths. The discrepancy is **real and is a
sample-selection effect**, not an error:

| path | onset detector | clean runs kept | clean notes | **clean EXACT** |
|---|---|---|---|---|
| `features_pitch._selftest` | `audio_onsets` (audio-first), keep only runs with **exactly 6** detected | **15 / 24** | 90 | **93.3%** |
| `e6_audio_fret.py` | `segment._select_onsets` (manifest-pinned, take 6 strongest) | **24 / 24** | 144 | **74.3%** |

The audio-first detector **drops 37.5% of clean runs** (9/24) where it under/over-fires;
the harmonic-template fret is then measured only on the surviving "easy" notes →
**93.3%**. The manifest-pinned segmenter keeps *all* runs and the same detector scores
**74.3%** on the full set. **Both are true; they answer different questions.**

> **Reviewer verdict (do this, it is load-bearing).** Report it as a **two-number
> contract**, never the 93.3% alone:
> - **"Audio fret = 74.3% exact on clean notes end-to-end (full set), vs 41.7% naive
>   F0 and 16.7% chance."** ← the *deployment-honest headline*.
> - **"Conditioned on clean 6-onset segmentation, the fret detector itself is 93.3%
>   exact"** ← isolates the *detector's* ceiling from the *segmenter's* error.
>
> Selecting on "exactly 6 onsets detected" is a form of **success-conditioned
> reporting**; presenting 93.3% as the system number would be the deck's most
> attackable claim. The within-1-fret numbers (clean 88.2% full set) are also worth
> showing because adjacent-fret confusions are the dominant error and are musically
> near-miss. Note the docstring in `features_pitch.py` currently advertises 93.3% as
> the headline — align it to the two-number framing.

### 2.4 Segmentation as the linchpin — correctly identified, but it cuts both ways

The team rightly made segmentation the foundation (blind `librosa.onset_detect`
over-fired 5× → pinning to the manifest's `expected_note_count`). For the **quality**
experiments this is fine: the label is the prompt, the segmenter just has to grab the
right 6 windows, and 432 events at exactly 6.00/run is clean. **But** for the
**audio-fret** claim, segmentation quality is *exactly* what produces the 93/74 gap
(§2.3) — so the segmenter is not a neutral preprocessing step there; it is part of the
system under test. The honest move (done) is to report fret on the manifest-pinned full
set as primary.

There is also a stated **philosophical tension** worth surfacing: `features_pitch.py`'s
docstring argues *audio-first* onsets ("the player did not strum exactly on every cue")
while `segment.py`/the chord pipeline pin to the **cue count**. For *single notes* the
manifest pin is well-justified (a metronome-cued 6-note run really has 6 notes); for
*chord streams* the audio-first view is more honest (an "Am×40" take had ~18 real
strums). The deck should state which regime uses which and why — it currently does for
chords (cue-windowed) but the rationale split should be explicit.

### 2.5 Chord-ID leakage finding — correct, keep it exactly as written

E4's three-CV-scheme reporting is textbook: nearest-μ_c chord-ID is **0.55** under
StratifiedKFold but **collapses to ≈0.03 within a single mixed stream**, because there
are only ~8 chord `run_id`s (7 single-chord takes) so `run_id` is nearly collinear with
the chord label — a **group-leakage** artifact identical in spirit to the album effect
(§2.1). Reporting **chord-match/off-detection (AUC 0.899) YES, chord-ID-from-audio NO**
is the right, non-overclaimed conclusion. This is a model of how to catch your own
leakage.

### 2.6 Vision: the honest-negative is the strongest version

The E7/E8 chain does the three things a reviewer wants: (a) **beats the broken baseline
but admits it's hollow**, (b) **runs a signal ablation** that proves the lift is a label
prior not pixels, (c) **tests the obvious fix** (per-frame registration) and reports it
*also* fails, isolating the cause to **capture geometry**. This is more credible than a
marginal positive would have been. The only addition a reviewer asks for: state the
**majority-fret base rate (fret 2 ≈ 54%)** every time the 0.45 MAE is quoted, so nobody
mistakes "ties the floor" for "works."

### 2.7 The data-integrity bug (event_id collision) — exactly the right thing to advertise

`run_id`s repeat across chord sessions → `event_id = run_id#k` collided (220 dupes) →
a key-join cross-multiplied 1032→4512 rows; caught by a row-count sanity check, fixed by
globally-unique `session::run_id#k`. Finding and killing this **before** it reached a
result is the single most persuasive rigor signal in the project. Keep it in the deck.

### 2.8 Strongest honest framing for judges (one paragraph)

> *Tactus makes two things the sensors can't directly read — a buzz the mic can't
> explain and a finger the camera can't see — legible. Quality (clean/buzz/muted) is a
> new, separable axis (0.80 held-out vs 0.33 chance, survives holding out whole
> recordings); it is a contribution because no public benchmark labels buzz/mute. Fret
> comes from audio via a string-conditioned harmonic template — 74.3% exact end-to-end
> on clean notes, 93.3% once segmentation locks, vs 41.7% naive and 16.7% chance. We
> tried to do fret from vision and **failed honestly**: the hand occludes the neck and a
> calibration homography can't register gameplay frames — a known, capture-side problem
> with a known fix (a learned fretboard detector, TapToTab-style, or a board-visible
> camera). We attacked our own headline: we caught and killed a join-collision bug
> mid-run, we ran a vision signal-ablation that disproved our own optimistic baseline,
> and we report the chord-ID number three ways and tell you which one is real. One
> player limits us; we say so. That discipline is why the numbers are believable.*

---

## 3. Evidence-based next steps (with citations)

**Priority 1 — Cross-player validation on GuitarSet (no new capture; directly kills the
one-player caveat for the generalizable parts).** GuitarSet (Xi et al., ISMIR 2018;
[paper](https://archives.ismir.net/ismir2018/paper/000188.pdf),
[data](https://zenodo.org/records/3371780)) has **6 players** with per-string/fret,
chord, and beat labels. Run our **audio feature pipeline + string-ID + audio-fret**
under a true **leave-one-player-out** split (the MIR-standard *artist filter*; FMA,
[arXiv 1612.01840](https://arxiv.org/pdf/1612.01840)). This converts our k-fold "within
one rig" claim into a cross-player claim *for the parts that can transfer*. **Honest
boundary:** GuitarSet has **no buzz/mute labels**, so the *quality* classifier stays
single-player — but string-ID, the harmonic-template fret, and the residual front-end
all get a real generalization test. (`mirdata` ships a GuitarSet loader, so this is days
not weeks.)

**Priority 2 — Replace classical fret-line detection with a learned fretboard keypoint
detector.** The documented path to a real beat-MediaPipe position result is a learned
detector, not Hough. **TapToTab** (Ghaleb et al., [arXiv 2409.08618](https://www.arxiv.org/pdf/2409.08618);
IEEE 2024) uses **YOLO fretboard detection** and reports robustness gains over classical
methods on oblique/occluded frames. Pair it with an **occlusion-robust hand keypoint**
model (OccRobNet, [arXiv 2503.21723](https://arxiv.org/html/2503.21723); occlusion-aware
training + TTA, [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405959526000135))
so fingertips survive self-occlusion. This is the modeling half of the vision fix.

**Priority 3 — Capture-side fix: board-visible camera angle (the one-line rig change).**
Our own diagnostic (`e7b_register.py`: **0% lock, 62.5px residual**;
`diag_registration_overlay.png`) shows the fretting hand occludes the neck at every
chord onset from the current angle. A higher/side camera where the hand does not cover
the frets makes per-frame registration tractable — this is **capture, not code**, and
should precede any vision-model investment. (Re-running E7 modeling on board-visible
footage is already wired; the modeling code is "ready" per `e7_model_report.txt`.)

**Priority 4 — Calibrate the chord buzz-transfer (H2) with per-string chord labels.**
The mono→poly residual transfer is currently a geometric claim because chords lack
per-string clean/buzz ground truth. Capturing a small set of *deliberately buzzed*
chord strums (one buzzing string at a time) would convert the "~10% tail at 2.3× energy"
into a **calibrated chord-buzz detector** with a real ROC. This also connects to the
playing-technique literature (Reboursière et al., NIME 2012,
[PDF](https://www.nime.org/proceedings/2012/nime2012_213.pdf); Su et al., EURASIP 2025,
[arXiv 2307.07426](https://arxiv.org/pdf/2307.07426)).

**Priority 5 — Strengthen the audio-fret front-end toward the MIR standard.** Our
harmonic-comb template is the right idea; the literature's higher-ceiling version is a
**harmonic-CQT** front-end (FretNet, Wiggins & Kim,
[arXiv 2212.03023](https://arxiv.org/pdf/2212.03023)) and explicit octave-error handling
(IETE 2019, [Tandfonline](https://www.tandfonline.com/doi/abs/10.1080/02564602.2018.1465859);
SwiftF0, [arXiv 2508.18440](https://arxiv.org/pdf/2508.18440)). Swapping the rFFT-comb
for a string-conditioned HCQT score should lift the *full-set* 74.3% toward the
clean-segmentation 93.3% by being less sensitive to the analysis window.

---

## 4. Sources

- **GuitarSet** — Xi, Bittner, Pauwels, Ye, Bello, *GuitarSet: A Dataset for Guitar
  Transcription*, ISMIR 2018. https://archives.ismir.net/ismir2018/paper/000188.pdf ·
  data: https://zenodo.org/records/3371780
- **TapToTab** — Ghaleb et al., *TapToTab: Video-Based Guitar Tabs Generation using AI
  and Audio Analysis*, arXiv 2409.08618 (IEEE 2024).
  https://www.arxiv.org/pdf/2409.08618 · https://ieeexplore.ieee.org/document/10783648/
- **FretNet** — Wiggins & Kim, *FretNet: Continuous-Valued Pitch Contour Streaming for
  Polyphonic Guitar Tablature Transcription*, arXiv 2212.03023.
  https://arxiv.org/pdf/2212.03023
- **Playing-technique detection** — Reboursière et al., *Left and right-hand guitar
  playing techniques detection*, NIME 2012.
  https://www.nime.org/proceedings/2012/nime2012_213.pdf
- **Percussive technique recognition** — Su et al., *Real-time playing technique
  recognition embedded in a smart acoustic guitar*, EURASIP J. ASMP 2025; arXiv
  2307.07426. https://arxiv.org/pdf/2307.07426 ·
  https://link.springer.com/article/10.1186/s13636-025-00413-6
- **Octave error / F0** — *Octave Error Reduction in Pitch Detection Algorithms*, IETE
  Tech. Review 2019. https://www.tandfonline.com/doi/abs/10.1080/02564602.2018.1465859 ·
  SwiftF0, arXiv 2508.18440, https://arxiv.org/pdf/2508.18440 · Harmonic-summation pitch,
  arXiv 2509.16480, https://arxiv.org/pdf/2509.16480
- **Occlusion-robust hand/pose** — OccRobNet, arXiv 2503.21723,
  https://arxiv.org/html/2503.21723 · Occlusion-robust 3D hand from single RGB, IEEE
  9511389, https://ieeexplore.ieee.org/document/9511389/ · Occlusion-aware training+TTA,
  https://www.sciencedirect.com/science/article/pii/S2405959526000135 · Under-occlusion
  benchmark, arXiv 2504.10350, https://arxiv.org/html/2504.10350v2
- **CV / album effect** — FMA: A Dataset for Music Analysis (artist filter), arXiv
  1612.01840, https://arxiv.org/pdf/1612.01840 · Cross-validation overview,
  https://en.wikipedia.org/wiki/Cross-validation_(statistics)
- **Earlier CV fingering** — Kerdvibulvech & Saito, *Guitar Tablature via Computer
  Vision*, Springer.
  https://link.springer.com/chapter/10.1007/978-3-030-33723-0_20
