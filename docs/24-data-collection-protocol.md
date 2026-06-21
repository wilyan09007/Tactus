# 24 — Data collection & labeling protocol (execute this to feed the AIML pipeline)

Status: **ACTIVE — start now.** Extends `docs/20` (training design), `docs/23` (data + cluster semantics), `docs/20-eng-review`. Canonical hardware: `truth.md`.

> **Goal:** high-volume, *labeled-by-construction* data fast, with an interval audit so we never drift. Decisions locked in office-hours:
> - **D1** single-note-dominant; **D2** collapse regions = {clean, buzz-too-light, buzz-placement} per string, muted/choked as cheap external flags; **D3** balanced classes + a natural-ratio holdout; **D4** **prompted RUNS + arpeggiated chords + interval audit**; camera = **MacBook front cam** (train/serve match).
> - **SCOPE LOCK (D6):** **frets 1–6 only, 1:1 fret→zone** — we have 6 fret-zone motors, so we record/coach only what we can render (this *supersedes* the eng-review V5 neck-spanning ask; 12-via-intensity is post-hackathon). See §0b for *what* the data trains.

---

## 0. The principle — the prompt IS the label
We never tag clips by hand afterward. You **announce a condition, play it, and the whole take inherits that label.** Your hand-work = executing prompted runs + a 2-second "did it come out as intended?" (y/n) per take. Every sample's ground truth is the prompt; audio only *verifies* (never silently relabel — log disagreements). This is what makes the labels trustworthy for the Most-Technical / Redis / Annapurna pitch.

---

## 0b. What the data actually trains — the two "last miles"
Vision is mostly **solved for free**: MediaPipe Hands (pretrained by Google) + ArUco/OpenCV homography turn raw pixels into clean, named, **fretboard-relative pose features** — and when the contact point is *visible* you just **read** (string, fret) off the grid (no model at all). So this capture is **not** teaching perception; it pins exactly **two small residuals**, and one recording trains both:

| The "last mile" (the ONLY trained part) | What it needs from your playing | Fed by |
|---|---|---|
| **Stage 1 — occluded pose → (string, fret, finger)** | the *same* position played with *varied* hand shapes/angles | the position grid + **pose-variation passes (D5)** |
| **Stage 2 — buzz B + d → pressure / cause** | clean vs the two buzz causes **back-to-back at the same spot**, with vision's `d` | the **clean / too-light / placement** triplet |

The prompt is the label for both (`(string,fret,finger)` for Stage 1, `class` for Stage 2); audio only verifies. Even at inference the Stage-1 "last mile" isn't alone — **audio pitch** narrows the candidate frets and the **theory prior** (the prompted/intended chord) constrains it, fused by the Bayesian posterior. **This is why ~1,000 takes suffice** where a raw-pixel model would need ~100× — we label a few hundred *residual* examples by construction, not a perception dataset. That economy is the payoff of the AI-only-where-unsolved split (`docs/17`).

## 1. The rig
- **Mic:** Saramonic LavMicro-U clipped **inside the body at a marked, repeatable spot** (capsule toward the strings), USB-C into the Mac. Same spot every session AND the demo.
- **Camera: the MacBook front camera** — the *same* camera the live AR interface uses, in the *same* playing position → train/serve match. **ArUco marker printed + taped to the headstock** (visible to the front cam). We do **not** need live MediaPipe during capture — record video, run MediaPipe+ArUco **offline** to extract pose + `d`, aligned to audio by timestamp.
  - ⚠️ A front view foreshortens the neck and the hand self-occludes the contact point — so vision `d` is **noisier** than overhead. That's fine: it's the real deployment condition (the occlusion problem the model exists to solve), the ArUco homography still recovers the fretboard plane, and **labels are prompt-grounded** so we train the occlusion model to predict the prompted truth from the noisy pose.
  - If the marker isn't printed yet: **start audio-only** (still proves "audio alone can't separate buzz cause"); add video the moment the marker's up.
- **Levels:** gain low enough that the **hardest pluck doesn't clip**; disable macOS input "enhancement."
- **Click:** slow metronome (~40–60 BPM) so notes separate cleanly for auto-segmentation.
- **Room:** quiet for the core; one short **noisy block** at the end.
- **Align mode (use for BOTH capture and inference — this is how we get train≈serve):** before any take, the app reads the **ArUco marker's apparent size/pose + the fretboard-region brightness** to estimate camera distance / angle / lighting, and guides you ("move back / raise the camera / more light") until you're inside the trained range, then **locks**. Capturing AND playing in the same aligned pose makes the vision features match **by construction** — far stronger than hoping the model is robust. The homography already absorbs a lot of camera-angle variation (fretboard-relative features); align-mode closes the rest, and offline **augmentation** (re-project homography to nearby virtual angles + brightness/contrast jitter) is cheap insurance for residual drift. This is an interface feature for the team; **gate capture on it.**

---

## 2. Capture protocol — prompted RUNS, not one-note blocks
A **run** = one fixed condition swept across a string's frets in a single continuous take.
1. The **conductor prompts** the exact condition; the prompt *is* the label, auto-logged to the manifest (§5): `(string, class, pluck, finger)`. (Day-1 zero-build flow — QuickTime + the terminal conductor — is in §10.)
2. Record one continuous WAV (+ MP4) for the run.
3. Play the sweep to the click: e.g. **frets 1→6 on the low-E, all buzz-too-light**, one note per tick.
4. Stop. 2-sec check: did all notes come out as intended? Set `matched_intent` y/n (re-do if a buzz run rang clean).

Offline we onset-segment each run into its notes; **every note inherits the run's class label**, and its `(string, fret)` comes from the sweep order + F0 cross-check. No per-note files, no post-hoc tagging.

**Chords = arpeggiated** (played note-by-note): each chord note becomes a clean single-note event (F0 → which string, buzz attributable per string), AND the held shape gives the vision model **multi-finger pose** data. Strummed chords go in a **separate renderer bucket** (labeled by chord name only, not per-string pressure).

---

## 3. The capture matrix

### Class definitions (produce consistently — this IS the experiment)
| Class | How to produce it |
|---|---|
| **clean** | finger just behind the wire, **good** pressure, medium pluck → rings clear |
| **buzz-too-light** | finger in the **correct** spot, pressed **too light** → buzz from low pressure |
| **buzz-placement** | **firm** pressure but finger **too far back** from the wire → buzz from bad placement |
| muted/dead | finger lightly touching, not fretting → dead thud (cheap external flag: energy gate) |
| choked/sharp | pressing too hard / slight bend → pitch **sharp** (cheap external flag: cents) |

> ⭐ **buzz-too-light vs buzz-placement is the crux.** Same sound, the only reliable difference is `d`. Record them back-to-back on the same string. Nailing this pair = the inverse-problem proof.

### Position grid: 6 strings × 6 frets = 36 cells, via runs
> **Frets 1–6 is the FULL scope** (6 fret-zone motors, 1:1 fret→zone) — the Stage-1 model only needs to generalize *within* 1–6; **no high-neck collection.** This is a rigor line, not a shortcut: *we record only what the body can render.*
- Per string: a run for each of **clean / buzz-too-light / buzz-placement**, sweeping **frets 1→6** = 18 runs × 6 notes = 108 events/pass.
- Do **~3–4 passes** → ~350–430 single-note position events, ~40 min.
- + a few **muted** and **choked** runs (~15 min), + the **pluck-sweep** (below).

### Chords (arpeggiated, labeled)
- ~6–8 common shapes (Em, Am, C, G, D, E, A…). Per shape: arpeggiate **clean**, then with **one deliberate buzz/mute** on a chosen string. ~20 min.
- Gives clean per-string chord labels + multi-finger vision poses.

### Renderer bucket + holdout
- **Strummed chords** (the same shapes, a few strums each) — for the chord-renderer path, chord-name label only. ~5 min.
- **Natural-ratio holdout:** ~5–10 min "just play normally" (mostly clean) — HELD OUT, to calibrate the real-world false-alarm rate (D3).

### Day-1 minimum (~45–60 min, unblocks me today)
- **2 strings** (low-E + G) × frets **1→6**, runs for **clean / buzz-light / buzz-placement**, ~3 passes → ~108 events
- + a few muted/choked + 2 arpeggiated chords (clean + one-buzz)
- → enough to run PCA→LDA, first confusion matrix, confirm audio-only confuses the two buzz causes while +`d` separates them.

### Full set (~1.5–2 hr): the 36-grid passes + chords + pluck-sweep + holdout → ~700–900 labeled events.

---

## 4. Pluck control (confound)
**Medium, consistent pluck for the whole balanced core** (control it). Plus a **pluck-sweep**: at ~3–4 cells, clean + buzz at **soft / medium / hard** (~80 events) → lets us add the pluck-proxy feature and prove buzz ≠ pluck (eng-review D1). Log `pluck_strength` on every run.

---

## 4b. Pose diversity (Stage 1) + how much data
**Pose-variation passes (D5):** repeat the position-grid runs with **different fingers and wrist/neck angles** (index vs ring on the same fret; slight hand rotation) so the occlusion model generalizes across hand shapes instead of memorizing one posture. ~2 extra variants × the 18 grid runs ≈ **+200 events, ~15–20 min.** (Arpeggiated chords already add multi-finger poses; this rounds out single-note diversity.)

**How much data — targets** (run method ≈ **5–6 sec wall-clock per labeled event**):

| Tier | Events | Buys you | Wall-clock |
|---|---|---|---|
| Minimum (prove the thesis) | ~300 | first separability headline (audio-only confuses cause; +d separates) + occlusion-model sanity | ~45–60 min |
| **Solid v1 (target)** | **~800–1,000** | both models trainable, per-string-ish, fully demoable — the real training set | ~1.5–2 hr |
| Strong | ~1,500–2,000 | robust + leave-one-player-out (Aiden records a partial set) | ~3–4 hr (2 people) |

**Per-class floor:** ~**100 clean events per collapse class** before trusting the separability numbers.
**Stop criterion (let the audit decide, don't guess):** collect until held-out **d′ between {clean, buzz-light, buzz-placement} plateaus** AND the 36-cell grid is covered. Still climbing at 1,000 → collect more; plateaus at 600 → done early.

## 5. Metadata schema (collect once, slice many ways) — one row per RUN
| Field | Example |
|---|---|
| `run_id` | `lowE_f1-6_buzzlight_pluckmed_aditya_001` |
| `session_id`, `player_id` | `2026-06-20-pm`, `aditya`/`aiden` |
| `string`, `fret_range` | `6 (low-E)`, `1-6` (or single fret for chords) |
| `finger` | 1–4 (index…pinky) |
| `intended_class` | clean / buzz-light / buzz-placement / muted / choked |
| `intended_placement` | on-wire / too-far-back |
| `pluck_strength` | soft / medium / hard |
| `chord_name` | (arpeggio/strum runs only) Em, C, … |
| `is_arpeggio` / `is_strum` | bool |
| `matched_intent` | y / n |
| `room` | quiet / noisy |
| `source_wav`, `source_video`, `notes` | paths + free text |

Rich metadata = the "many hypotheses on one dataset" capability: re-bin by pressure-only, by cause, per-string, 2-class vs 3-class — offline, no re-recording.

---

## 6. File / folder convention
```
data/
  raw/{session}/{player}/audio/{run_id}.wav     # one continuous WAV per run
  raw/{session}/{player}/video/{run_id}.mp4     # front-cam video (optional Day-1)
  manifest.jsonl                                  # one row per run (§5)
  events/                                         # (generated) one row/clip per segmented note
```
Clap at each session start = A/V sync check. Name strictly → I can auto-generate the manifest from filenames.

---

## 7. What I need from you
1. **Rig:** mic inside body (marked), MacBook front cam in playing position + ArUco on headstock (or audio-only to start), gain not clipping, slow click.
2. **Run the Day-1 slice** (§3) — ~45–60 min.
3. **Prompted runs**, one continuous take each, then the y/n `matched_intent` check; **arpeggiate** chords.
4. **Name files + fill the manifest** (§5) as you go.
5. **Hand me** `raw/` WAVs (+MP4s) + manifest — even just the Day-1 audio batch unblocks me. Then broaden to the full set while I build the pipeline.

---

## 8. The pipeline + the interval audit (your anti-digression dashboard)
Hand me a batch every ~15–20 min; `/loop` me and I run:
1. **Onset-segment** runs → events; attach labels from the manifest + F0 cross-check.
2. (video) **MediaPipe + ArUco offline** → pose + `d` per event, aligned by onset.
3. **Features** (~40–60 named audio + `d` + pluck-proxy).
4. **Collapse:** standardize → PCA → **LDA** (readable eigenvectors) → UMAP viz.
5. **Separability** under leave-one-player/position-out: Fisher / silhouette / pairwise d′ / confusion — **audio-only vs fused (+d)**.
6. **Redis:** index the collapsed fault-embeddings for nearest-neighbor "your buzz is like these past attempts."

**The audit (runs on every batch — catches drift early):**
- **Technical:** clipped clips? level distribution sane?
- **Segmentation:** expected note count per run?
- **Label integrity:** does each note's F0 match the prompted string/fret? (mismatch = mislabeled/misplayed run → flagged)
- **Running separability:** clean-vs-buzz d′ — is it *rising*? Is the light-vs-placement pair separating once `d` is in? (stalls → fix how you're producing the pair NOW)
- **Coverage:** which of the 36 cells × classes are filled vs thin.
→ a one-screen report per batch. That's the loop's objective: **maximize held-out separation of {clean, buzz-light, buzz-placement}, minimize confusion**, and tell you what to collect more of.

---

## 9. Quality bars / batch-killers
- **Clipping** → buzz band destroyed. Keep gain low; watch the meter.
- **Mic or laptop moved** → features drift. Mark positions.
- **buzz that rang clean** → that's what `matched_intent` y/n + the F0 audit catch; re-do the run.
- **Inconsistent pluck** in the controlled core → leaks into the buzz signal. Medium only; use the sweep for variation.
- **Mislabeled run** → poisons ~6 notes. Check the manifest row before recording.

---

## 10. Day-1 recording — do this NOW (zero-build, can't-fail)
No localhost / align-mode harness needed to start. **QuickTime captures synced A/V in ONE file; a terminal conductor tells you what to play and writes the manifest.** (The browser align-mode harness is the *next* build, for the M4 with the camera.)

**Setup (once):**
1. **Mic:** plug in the Saramonic → *System Settings → Sound → Input = Saramonic*; **disable any input "enhancement"/noise-cancel**; set gain so a **hard strum doesn't clip**.
2. **Metronome:** phone app at **~50 BPM** (slow → notes separate for auto-segmentation).
3. **QuickTime → New Movie Recording** → click the **⌄** by the record button: **Camera = front cam** (you in playing position), **Microphone = Saramonic** → **Record**. *(Audio-only fallback if the ArUco marker isn't printed yet: New Audio Recording, mic = Saramonic — still proves audio alone can't separate the buzz cause.)*
4. **CLAP once, loud** at the very start = the A/V sync mark.

**Run the conductor** (`software/ai/capture/record_conductor.py` — stdlib only, no install):
```
cd ~/Desktop/tactus && python3 software/ai/capture/record_conductor.py --player aditya --passes 3
```
It prints each run in **interleaved** order (kills session drift, H4), waits for you to play it + hit ENTER, asks the y/n intent check, and appends a manifest row (§5) to `data/raw/<session>/<player>/manifest.jsonl`. **One QuickTime file per pass is fine** — save the movie into that session's `video/` (or `audio/`) folder, then hand the folder to the pipeline.

**Extra blocks** (rerun the conductor or just announce them on the recording): pose-variation (`--finger ring`, `--finger pinky`) → Stage-1 generalization; pluck-sweep (soft/med/hard at a few cells) → buzz ≠ pluck; a few muted/choked runs; ~6–8 **arpeggiated** chords (clean + one deliberate buzz); and a 5-min **natural holdout** ("just play normally" — held out, calibrates false-alarm rate).
