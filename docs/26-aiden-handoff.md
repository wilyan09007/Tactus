# 26 — Data handoff for the offline pipeline (Aiden)

Status: **ACTIVE.** Who-gets-what for post-processing. Aiden owns everything **after** capture:
ingest → segment → features → separability. The capture tool already exists
(`software/ai/capture/`) — **do not rebuild it.** Build the pipeline under `software/ai/`
(`pipeline/`, `features/`, `separability/`). The authoritative *processing contract* is
[`docs/25`](25-data-and-feature-format.md); this doc is the orientation + the exact data shapes.

---

## 0. Read these first (in order)
1. **[`docs/25`](25-data-and-feature-format.md)** ← THE contract you code against (the 6 correctness fixes). If anything here is ambiguous, docs/25 wins.
2. **[`docs/24`](24-data-collection-protocol.md)** — capture protocol, manifest schema (§5), file layout (§6), pipeline + interval-audit (§8).
3. **[`docs/23`](23-data-and-cluster-semantics.md)** — feature vector (§5), training procedure + **iron rules** (§7), cluster→advice epistemics.
4. **[`docs/20`](20-aiml-training-design.md)** + **[`docs/17`](17-ai-rigor.md)** — locked model (D1–D8) + the rigor thesis (AI only where unsolved; the separability study is the centerpiece).
5. **[`truth.md`](../truth.md)** §1 + §6 — canonical direction. Markerless registration is in **`software/ai/vision/fretboard.py`** + **`twin.py`** (use these; don't reinvent).

## 1. What you're getting
Multimodal **raw** streams + labels — **not** feature vectors (you compute those). Layout:

```
data/raw/<session_id>/<player_id>/
  audio/<run_id>.wav      # LOSSLESS PCM16, MONO — the canonical audio for all features
  video/<run_id>.webm     # front-cam (vp9/vp8 + an opus audio track)
  manifest.jsonl          # ONE JSON row per RUN  ← your index
data/calib/<guitar_id>/
  kf_*.png                # twin keyframes (neck at ~6 angles)
  meta.jsonl              # 4 clicked neck corners + board points per keyframe
  twin.json               # (from twin.py) fret-law intrinsics + per-pose homography
```

**One row = one RUN = one continuous take with multiple notes.** You segment each run into note
events; **every event inherits the run's label**; per-event `(string, fret)` comes from the beat grid
+ F0 (NOT sweep order — see §3.2).

## 2. Manifest row schema (real example — fields are authoritative, not `run_id`)
```json
{
  "run_id": "s6_16_buzzlight_plkmedium_aditya_001",
  "session_id": "2026-06-20-2057", "player_id": "aditya", "app_version": "capture-0.2",
  "block": "core-grid",
  "string": "6 (low-E)", "fret_range": "1->6", "frets": [1,2,3,4,5,6], "finger": "index",
  "intended_class": "buzz-light", "intended_placement": "on-wire",
  "pluck_strength": "medium", "pluck_variant": null, "pose_variant": null,
  "chord_name": null, "is_arpeggio": false, "is_strum": false,
  "pass": 1, "held_out": false, "expected_note_count": 6,
  "matched_intent": "y", "room": "quiet", "notes": "",
  "bpm": 50, "beat_times_ms": [1203.4, 2401.8, 3600.1, 4799.5, 5998.2, 7196.9],
  "recorded_at": "2026-06-21T03:58:12.345Z", "duration_ms": 8200, "t0_perf_ms": 10432.5,
  "audio": {"sample_rate": 48000, "channels": 1, "format": "wav-pcm16",
            "peak_dbfs": -6.2, "clipped": false, "clip_samples": 0, "silent": false},
  "video": {"present": true, "width": 1280, "height": 720, "frame_rate": 30, "mime": "video/webm;codecs=vp9,opus"},
  "devices": {"mic_label": "Saramonic LavMicro-U", "cam_label": "FaceTime HD Camera"},
  "aruco_marker_present": false, "audio_only": false,
  "sync": "single getUserMedia stream …",
  "files": {"audio": "data/raw/.../audio/<run_id>.wav", "audio_bytes": 787200,
            "video": "data/raw/.../video/<run_id>.webm", "video_bytes": 2400000}
}
```

**Fields that matter:**
| Field | Meaning / use |
|---|---|
| `string` | `"N (name)"` e.g. `"6 (low-E)"`; `"chord"` for chord blocks. Parse N. **Don't parse labels from `run_id`** (it strips chars, can collide). |
| `intended_class` | `clean \| buzz-light \| buzz-placement \| muted \| choked \| natural`. **THE label** (the prompt). Audio only *verifies* — never relabel/drop; log the disagreement rate. |
| `intended_placement` | `on-wire \| too-far-back \| null`. |
| `finger` / `pose_variant` | which finger to read `d` from; pose_variant = the wrist/neck-angle variation (Stage-1 diversity). |
| `expected_note_count` | segmentation audit target (6 sweep, 4 pluck-sweep/strum, 6 arpeggio, 0 holdout). |
| `beat_times_ms` | onset **priors** (one note per click tick), relative to `t0_perf_ms`. |
| `matched_intent` | `"n"` → take didn't come out as intended → quarantine, don't silently drop. |
| `audio.format` | `wav-pcm16` normally; `webm-opus` if the lossless tap failed → **segregate** (opus mangles the buzz band). |
| `audio.sample_rate` | 44100 or 48000 → **canonicalize to 48k** before features. |
| `aruco_marker_present` / `audio_only` | both `false` in normal capture → **markerless, video-on**; register with `fretboard.py`/`twin.py`. |
| `held_out` | `true` (natural block) → real-world false-alarm calibration; never train on it. |

## 3. The 6 processing gotchas (full detail in `docs/25` — don't skip)
1. **A/V sync is NOT free.** WAV (lossless tap) and webm start at slightly different offsets — ignore the optimistic `sync` field. **Cross-correlate the WAV against the webm's opus track** (same mic) to recover the offset → audio-onset → video frame. (Clap at session start = free anchor.) `d` is read at the onset frame, so a 50–100 ms error corrupts `d`.
2. **Segment with beat priors + F0, not generic onset + sweep order.** Buzz/muted notes have weak onsets; sweep-order labeling cascades off-by-one. Use `beat_times_ms` priors, assign fret by **F0**, **flag runs where `seg_count ≠ expected_note_count`**.
3. **Audio-only vs fused on the SAME events.** Restrict the ablation to vision-having events; produce the audio arm by **masking the vision block**.
4. **No lossy/lossless mixing.** Buzz study on **WAV-only**; flag/exclude opus; resample to 48k.
5. **Pitch control via covariates, not per-cell z-score.** Standardize globally (train-fold only); include `string`+`fret` as covariates.
6. **d′ = pairwise Mahalanobis in the LDA subspace** (3 classes → 2 axes) + per-axis loadings + confusion. Split by player (LOPO); for leave-one-position-out, group by **run** so a run's events never straddle folds.

## 4. What to build (mirrors `docs/25` §8)
`ingest manifest → recover WAV↔webm offset → beat-anchored+F0 segmentation w/ count audit →
format-segregated feature extraction @48k → markerless pose+d (MediaPipe Hands + fretboard.py homography)
aligned to onsets → covariate-controlled standardize→PCA(~95%)→LDA → matched-modality Mahalanobis d′ +
confusion (LOPO/LOPosO) → one-screen interval audit.`

**Feature vector (~40–60 dims, `docs/23` §5):** audio (centroid/flux/flatness/rolloff, HNR, inharmonicity,
attack/decay, MFCC×13, buzz-energy ratio, ZCR, chroma, pitch-cents) + **pluck-proxy** (attack RMS / onset
slope) + vision (per-finger curl, joint angles, **d=fingertip-to-wire**, neck pos, wrist angle) — all
fretboard-relative.

**Iron rules (violating these invalidates the result):** prompt = label (audio verifies, log disagreement);
split by player, never random; fit ALL preprocessing on train folds only; pressure = 2-class ordinal
(too-light/good), "too hard" = separate pitch-cents fault; `d` is coarse/relative (never cm/Newtons); log
inferences to catch octave / string-fret mapping bugs.

**Deliverable:** the first confusion matrix + pairwise d′ showing **audio-only confuses buzz-light vs
buzz-placement, and +d separates them** + the one-screen audit.

## 5. Deps
`software/requirements.txt` (librosa, mediapipe, opencv-contrib-python, scikit-learn, numpy, scipy, pandas,
plotly/matplotlib). Classical pipeline trains in seconds–minutes on the M4.
