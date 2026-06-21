# Graph Report - .  (2026-06-20)

## Corpus Check
- 13 files · ~59,915 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 690 nodes · 860 edges · 59 communities (53 shown, 6 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 84 edges (avg confidence: 0.85)
- Token cost: 110,000 input · 12,299 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Hardware BOM & Architecture|Hardware BOM & Architecture]]
- [[_COMMUNITY_AI Rigor & Fusion Core|AI Rigor & Fusion Core]]
- [[_COMMUNITY_CAD Enclosure (brain-pack)|CAD Enclosure (brain-pack)]]
- [[_COMMUNITY_LLM Coach & Learning Loop|LLM Coach & Learning Loop]]
- [[_COMMUNITY_HapticAudio Research Briefs|Haptic/Audio Research Briefs]]
- [[_COMMUNITY_Software Modules & Deps|Software Modules & Deps]]
- [[_COMMUNITY_Actuator Puck CAD|Actuator Puck CAD]]
- [[_COMMUNITY_Build Refinements & Gotchas|Build Refinements & Gotchas]]
- [[_COMMUNITY_As-Built Rig & AR Prototypes|As-Built Rig & AR Prototypes]]
- [[_COMMUNITY_12-Channel Map (strings+zones)|12-Channel Map (strings+zones)]]
- [[_COMMUNITY_AIML Training Design|AIML Training Design]]
- [[_COMMUNITY_Enclosure CAD Renders|Enclosure CAD Renders]]
- [[_COMMUNITY_Chest Plate & Socket CAD|Chest Plate & Socket CAD]]
- [[_COMMUNITY_ChordSustain Rendering|Chord/Sustain Rendering]]
- [[_COMMUNITY_Capture App & Protocol|Capture App & Protocol]]
- [[_COMMUNITY_Design Decisions (rejected)|Design Decisions (rejected)]]
- [[_COMMUNITY_LEARN Loop Components|LEARN Loop Components]]
- [[_COMMUNITY_AIML Eng Review|AIML Eng Review]]
- [[_COMMUNITY_Chord Rendering Spec|Chord Rendering Spec]]
- [[_COMMUNITY_Capture Save Server|Capture Save Server]]
- [[_COMMUNITY_Build Refinements (channels)|Build Refinements (channels)]]
- [[_COMMUNITY_BOM Accounting|BOM Accounting]]
- [[_COMMUNITY_Assembly Checklist Stages|Assembly Checklist Stages]]
- [[_COMMUNITY_Power Architecture & Safety|Power Architecture & Safety]]
- [[_COMMUNITY_Tuning & Calibration Experiments|Tuning & Calibration Experiments]]
- [[_COMMUNITY_Interface AR Play + Correction|Interface: AR Play + Correction]]
- [[_COMMUNITY_Capture Recording Functions|Capture Recording Functions]]
- [[_COMMUNITY_Haptic Encoding Design|Haptic Encoding Design]]
- [[_COMMUNITY_Vibrotactile Perception Refs|Vibrotactile Perception Refs]]
- [[_COMMUNITY_SK473KHD Tuning|SK473/KHD Tuning]]
- [[_COMMUNITY_AI Pitch & Scoping|AI Pitch & Scoping]]
- [[_COMMUNITY_Open Questions  To-Do|Open Questions / To-Do]]
- [[_COMMUNITY_Wiring Connectors (forbidden)|Wiring Connectors (forbidden)]]
- [[_COMMUNITY_System Block Diagram|System Block Diagram]]
- [[_COMMUNITY_Cluster to Advice Semantics|Cluster to Advice Semantics]]
- [[_COMMUNITY_Safety & Wire Weight|Safety & Wire Weight]]
- [[_COMMUNITY_Software Pipeline Overview|Software Pipeline Overview]]
- [[_COMMUNITY_Assembly Bring-up Gates|Assembly Bring-up Gates]]
- [[_COMMUNITY_Perception to Design Mapping|Perception to Design Mapping]]
- [[_COMMUNITY_ArUco Pose-Lock JS|ArUco Pose-Lock JS]]
- [[_COMMUNITY_Mode A Wall Power|Mode A Wall Power]]
- [[_COMMUNITY_Mode B Cordless Power|Mode B Cordless Power]]
- [[_COMMUNITY_Feature Vector & Separability|Feature Vector & Separability]]
- [[_COMMUNITY_Two-Stage Perception Model|Two-Stage Perception Model]]
- [[_COMMUNITY_System Architecture Chain|System Architecture Chain]]
- [[_COMMUNITY_Soldering Guide|Soldering Guide]]
- [[_COMMUNITY_Psychophysics Brief|Psychophysics Brief]]
- [[_COMMUNITY_Record Conductor (fallback)|Record Conductor (fallback)]]
- [[_COMMUNITY_Wiring Map (channel)|Wiring Map (channel)]]
- [[_COMMUNITY_Project Pivot & Scope|Project Pivot & Scope]]
- [[_COMMUNITY_Capture Run Planner|Capture Run Planner]]
- [[_COMMUNITY_Docs Map  Overview|Docs Map / Overview]]
- [[_COMMUNITY_alphaTab Loader JS|alphaTab Loader JS]]
- [[_COMMUNITY_Power Cradle CAD|Power Cradle CAD]]
- [[_COMMUNITY_Laptop to Vantec USB|Laptop to Vantec USB]]
- [[_COMMUNITY_Haptic-Score Removal|Haptic-Score Removal]]
- [[_COMMUNITY_Coupling & Amp Rules|Coupling & Amp Rules]]
- [[_COMMUNITY_Data Volume Target|Data Volume Target]]

## God Nodes (most connected - your core abstractions)
1. `Design Decisions (Why, and What We Rejected)` - 15 edges
2. `15 — Build refinements + the kinks that will bite (read before Saturday)` - 14 edges
3. `Data collection & labeling protocol` - 14 edges
4. `20 — AIML training design: multimodal guitar-technique coaching (comprehensive spec)` - 13 edges
5. `base() — enclosure base shell + mounts + vents` - 13 edges
6. `21 — Chord, sustain & percussive rendering (extends docs/07)` - 11 edges
7. `Vibrotactile Music Perception Research Brief` - 11 edges
8. `Assembly checklist + bring-up` - 10 edges
9. `KHD 3 Ohm / 5 W driver (the actuator, de-housed from SK473)` - 10 edges
10. `ai/ Module (transcription, vision, fusion, LLM coach)` - 10 edges

## Surprising Connections (you probably didn't know these)
- `MockEngine (WebSocket message emitter)` --semantically_similar_to--> `LLM Coach Loop`  [INFERRED] [semantically similar]
  web/index.html → software/README.md
- `glow draggable 4-corner neck calibration grid` --semantically_similar_to--> `Align mode (ArUco pose gate for train=serve)`  [INFERRED] [semantically similar]
  web/glow.html → docs/24-data-collection-protocol.md
- `Pivot: Hearth to Tactus` --rationale_for--> `Tactus direction & goal (Deaf-accessible guitar coaching)`  [INFERRED]
  CHANGELOG.md → truth.md
- `Scope lock: LEARN + PLAY guitar coaching` --conceptually_related_to--> `Tactus direction & goal (Deaf-accessible guitar coaching)`  [INFERRED]
  CHANGELOG.md → truth.md
- `Vision branch (MediaPipe hand pose + ArUco fretboard homography)` --conceptually_related_to--> `LEARN mode (Rocksmith for Deaf players)`  [INFERRED]
  docs/08-software-architecture.md → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **AR guitar-coaching interface prototypes** — web_ar_chord_field_ar, web_beatsaber_beat_saber_highway, web_glow_fret_glow_annotation [INFERRED 0.85]
- **Cluster-to-advice epistemic guardrails** — doc23_prompt_is_label, doc23_lda_supervised_backbone, doc23_causal_clean_fault_pair, doc23_held_out_validation, doc23_promotion_gate [EXTRACTED 0.85]
- **Capture record-to-save flow** — capture_capture_startrec, capture_capture_stoprec, capture_capture_buildrow, capture_capture_commit, capture_serve_save_server [INFERRED 0.85]

## Communities (59 total, 6 thin omitted)

### Community 0 - "Hardware BOM & Architecture"
Cohesion: 0.07
Nodes (44): Build-tonight TL;DR, Pick your lane, Start here (teammates), Status, What tactus is (30 seconds), AudioVox 18 AWG speaker wire (the only body wire), LEO SALES 40mm speakers (spare actuators), Raspberry Pi 5 (owned, cut from critical path) (+36 more)

### Community 1 - "AI Rigor & Fusion Core"
Cohesion: 0.05
Nodes (40): Buzz Inverse (Pressure from Sound), Localhost WebSocket JSON Contract (the Seam), 0. The thesis — AI where it's *unsolved*, determinism where it's *solved*, 17 — The AI core: audio + vision → high-granularity (string, fret, finger, quality) → targeted feedback, 1. High-granularity POSITION — why fusion beats either modality alone, 1a. Audio branch (what + how-clean), 1b. Vision branch (where + which finger) — runs in the **browser**, 1c. Fusion (the part that's a real contribution, not a wrapper) (+32 more)

### Community 2 - "CAD Enclosure (brain-pack)"
Cohesion: 0.08
Nodes (39): SK473 amp+driver units mount on the VEST (decision), FlashForge Adventurer 5M (220x220x220 bed), Headless manifold3d render pipeline (no OpenSCAD on arm64), tactus_box (superseded split enclosure), tactus_enclosure (base+lid, one print), tactus_enclosure_plate.stl (175x202 one-print bed), tactus_power_cradle (vented sled), tactus_box.scad — main electronics enclosure (brain-pack) (+31 more)

### Community 3 - "LLM Coach & Learning Loop"
Cohesion: 0.06
Nodes (33): LLM Coach (Anthropic, Phrase-Level), Redis Coach Memory (Mistake History, Vector Search), Anti-Slop Rule (No Pixel Without Real Data), Feel-the-Difference Replay (Your Version vs Correct), Component A — Haptic Reference (Ground Truth), Camera-Centric Interface (Every Pixel Real), The Learning Loop (Feel Target, Play, Correct), Offline Local Redis Memory (Practice Anywhere) (+25 more)

### Community 4 - "Haptic/Audio Research Briefs"
Cohesion: 0.07
Nodes (30): AI Where Unsolved, Deterministic Where Solved, Deterministic Renderer (Encoder + Pulse Synth), Audio Branch (pYIN/YIN live, CREPE/basic-pitch offline), Demo-Floor Insurance (Deterministic Path, Zero ML), Audio to Haptic ML/DSP Research Brief, Haptic-Score Idea Cut (Direct Signal to Vibration Now), Information Transfer Evaluation (IT bits, d-prime), Learned Model Prior Art (HapticGen, HapticLDM, Sound2Hap) (+22 more)

### Community 5 - "Software Modules & Deps"
Cohesion: 0.08
Nodes (30): ai/ Module (transcription, vision, fusion, LLM coach), ALSA Multichannel Write (3 Vantec / 8ch), Bring-up order, LLM Coach Loop, haptic/ Module (encoder, synth, alsa_out), Honest-Scoping Rules (coarse placement, no pressure sensor), Install, software (+22 more)

### Community 6 - "Actuator Puck CAD"
Cohesion: 0.09
Nodes (27): actuator_puck.scad — Ø52 driver coupling puck (cup + button), button() — domed contact button bonded to the dust cap, cup() — rigid coupling cup that grips the Ø52 driver frame, box(), button(), cup(), cyly(), cylz() (+19 more)

### Community 7 - "Build Refinements & Gotchas"
Cohesion: 0.10
Nodes (19): 0. The 60-second mental model of the whole rig, 10. Laptop connection + power — exactly what plugs into what, 11. SK473 reality (live teardown, Jun 20): it's 6 stereo PAIRS = **12 channels**, and the amp *is* separable, 12. UPDATE (Jun 20 teardown): use the SK473's OWN drivers (KHD 3 Ω/5 W), and the ≤2-at-once power win, 15 — Build refinements + the kinks that will bite (read before Saturday), 1. ⚠️ THE BIG ONE — PAM8403 is a BTL amp: never common the "−" outputs, 2. ⚠️ The Vantec/ALSA enumeration trap (this eats Saturday if you don't pre-empt it), 3. Wire gauge — you only have 18 AWG, so the build has to respect it (+11 more)

### Community 8 - "As-Built Rig & AR Prototypes"
Cohesion: 0.18
Nodes (17): Align mode (ArUco pose gate for train=serve), As-built rig (laptop to Vantec to PAM8403 to KHD), ALSA CM6206 enumeration trap (bind by-id), 12-channel map (6 back strings + 6 torso fret-zones), Scope lock: frets 1-6 only, 1:1 fret to zone, Localhost WebSocket browser-Python sync, MacBook front camera (train/serve match), Locked software architecture (browser vision + Python audio) (+9 more)

### Community 9 - "12-Channel Map (strings+zones)"
Cohesion: 0.13
Nodes (15): Back ch5 — A string site, Back ch2 — B string site, Back ch4 — D string site, Back ch3 — G string site, Back ch1 — high E string site, Back ch6 — low E string site, BACK — 6 strings (ch 1–6), config/channel_map.json (source of truth) (+7 more)

### Community 10 - "AIML Training Design"
Cohesion: 0.13
Nodes (14): 0. The eight locked decisions, 10. Open questions for ENG review, 11. Award alignment (why this wins, `docs/19`), 1. The data pipeline (capture + labeling), 20 — AIML training design: multimodal guitar-technique coaching (comprehensive spec), 2. The vision stack (D3) — seeing through the hand, 3. The audio stack + the buzz inverse (D4→D6), 4. The fusion model (D8) — one Bayesian framework (+6 more)

### Community 11 - "Enclosure CAD Renders"
Cohesion: 0.18
Nodes (14): Enclosure base box (open top, blue CAD preview), Corner screw bosses on base, Enclosure lid, interior view (blue CAD preview), Triangulated rib / support structure on lid underside, Corner screw holes on lid, Internal partitions / component mounting structure, Rectangular port window cutout in base side wall, Row of circular wire/vent holes (Ø16 mm, USB-A pass-through) (+6 more)

### Community 12 - "Chest Plate & Socket CAD"
Cohesion: 0.26
Nodes (14): tactus_chest_plate.scad — body-contoured plate holding 6 string drivers, bore_at(z) — through-hole so the cone/button reaches skin, shell() — curved torso-cylinder shell panel between z_lo..z_hi, socket_at(z) — places a driver_socket on the inner face at height z, strap_slots(z_lo, z_hi) — edge slots to lash onto the laser-tag vest, tile(z_lo, z_hi) — one print tile (shell + sockets - bores - slots), tile_abs(z_lo, z_hi) — print tile built at absolute z coordinates, tactus_node_mount.scad — per-node split-clamp + telescoping driver mount (+6 more)

### Community 13 - "Chord/Sustain Rendering"
Cohesion: 0.18
Nodes (13): Chord onset encodes how it was struck (strum sweep / pluck bloom / arpeggio), Chord, Sustain & Percussive Rendering, Fret-zones carry chord SHAPE (pulsed in sync) + root emphasis, GSTACK eng-review report (PAUSED; B1-B7 open, 3 critical silent-failure gates), Percussive & muted hits (sharp pop across motors; muted strings don't fire), Renderer input: per-string (onset, pitch->string/fret, envelope, decay) stream, Sustain = envelope-modulated looped shimmer (LOCKED), One phase-locked signal coherence (screen + body + sound from one stream) (+5 more)

### Community 14 - "Capture App & Protocol"
Cohesion: 0.23
Nodes (12): capture.html localhost capture UI, enable() - getUserMedia with DSP off, meterLoop() - live level meter + clip detection, setupAudio() - AudioWorklet PCM capture + analyser, Capture README (two recording routes), serve.py localhost save server, Clean/buzz-light/buzz-placement triplet (the crux), Capture matrix (class defs + position grid) (+4 more)

### Community 15 - "Design Decisions (rejected)"
Cohesion: 0.17
Nodes (12): Amplification — gut SK473 boards (PAM8403), Audio source / channels — USB 7.1 (Vantec NBA-200U, CM6206), Decision: bass dropped (was Logitech Z313), Bass — dropped (was Logitech Z313), Decision: brain is the laptop (M4 Pro 48 GB), Pi cut from critical path, Brain — laptop (Pi cut from critical path), Design Decisions (Why, and What We Rejected), Encoding — location + timing, not frequency (+4 more)

### Community 16 - "LEARN Loop Components"
Cohesion: 0.17
Nodes (12): 13 — LEARN: Closing the Loop (feel the target · see the fix), 1. The learning loop (one flow — replaces the old practice modes), 2. The four components, 3. Local memory — practice anywhere, offline (Redis, baked in), 4. The interface (camera-centric, every pixel real), 5. Architecture + parallelization (the seam), 6. Sponsors this unlocks (a much deeper story), 7. Honest scope — demoable this weekend vs research-bet (+4 more)

### Community 17 - "AIML Eng Review"
Cohesion: 0.17
Nodes (12): 20 — Eng review of the AIML training design (`docs/20-aiml-training-design.md`), Completion summary, Corrected identifiability (what's observed vs hidden), Data collection — breadth vs depth (fits 2–4 hrs), Decisions taken (this review), Diagrams, Failure modes (per new codepath: realistic failure · test? · error handling? · silent?), Implementation tasks (+4 more)

### Community 18 - "Chord Rendering Spec"
Cohesion: 0.17
Nodes (11): 1. Single note (recap of `docs/07`), 21 — Chord, sustain & percussive rendering (extends docs/07), 2. Chords — the onset encodes HOW it was struck, 3. Sustain = envelope-modulated looped shimmer (LOCKED), 4. The other axis + foundation, 5. Percussive & muted hits, 6. Power (reconciles `docs/15 §12`), 7. Tunable parameters (→ `docs/18` experiments) (+3 more)

### Community 19 - "Capture Save Server"
Cohesion: 0.29
Nodes (3): Handler, Aggregate a session's manifest so the page can cross-check what's on disk., safe()

### Community 20 - "Build Refinements (channels)"
Cohesion: 0.18
Nodes (11): Channel count — 12, Tightened connector list (3 rails: line-level / BTL drive / 5V power), 40mm speakers are not exciters — coupling (contact button + backer + preload) is make-or-break, Build refinements + the kinks that will bite, PAM8403 is BTL — never common the '-' outputs (isolated pair per channel), Mode A power correction: hub is a buck'd 12V brick, run at felt level, Single-channel-first build order (prove one full channel, then mass-produce), SK473 teardown (Jun 20): 6 stereo PAIRS = 12 channels, amp is separable (+3 more)

### Community 21 - "BOM Accounting"
Cohesion: 0.20
Nodes (10): Actuators (the SK473's own KHD drivers), Amplifiers, Audio source (DAC / channels), Bill of materials, Channel accounting, Not needed for the audio path, Owned now, Power (+2 more)

### Community 22 - "Assembly Checklist Stages"
Cohesion: 0.20
Nodes (10): Assembly checklist + bring-up, Go/no-go summary, Stage 0 — bench prep, Stage 1 — prove the audio path (BEFORE any soldering), Stage 2 — gut + tone-test each amp (×6), Stage 3 — power, Stage 4 — chord stress test, Stage 5 — mount + couple (+2 more)

### Community 23 - "Power Architecture & Safety"
Cohesion: 0.22
Nodes (9): Fewer-ports fallback, Hard rules (break these and you fry something), Mode A — wall (RECOMMENDED for the judged demo), Mode B — cordless (the portability "wow"), Per-bus loads (computed the right way), Power architecture + the chord-safety guarantee, Pre-power checklist (every session), The numbers (per channel / per amp) (+1 more)

### Community 24 - "Tuning & Calibration Experiments"
Cohesion: 0.22
Nodes (8): 18 — Tuning & calibration: the on-body experiments (fill in Saturday), Experiment 1 — drive-frequency sweep (do this FIRST, per zone), Experiment 2 — punchiness (pulse shape), Experiment 3 — coupling pressure, Experiment 4 — per-channel gain calibration, Record the winners (these become the renderer defaults), Safety bounds (don't cross while experimenting), The 4 knobs you're tuning

### Community 25 - "Interface: AR Play + Correction"
Cohesion: 0.22
Nodes (9): 22 — The interface: live AR play + targeted 2D correction (the centerpiece), A. LIVE AR PLAY — Beat-Saber depth-approach on your real fretboard, Award alignment (`docs/19`), B. TARGETED CORRECTION (on error) — the super-simple 2D fretboard, Build order (de-risked), Coherence (one phase-locked signal → screen + body + sound), Open questions for eng review, Per-finger position + pressure — what's supportable (ties to `docs/20`) (+1 more)

### Community 26 - "Capture Recording Functions"
Cohesion: 0.25
Nodes (8): buildRow() - maximal-metadata manifest row, commit() - POST blobs + manifest to server, encodeWav() - lossless PCM16 WAV encoder, offerDownload() - per-run download fallback, startRec() - MediaRecorder + visual metronome, stopRec() - finalize blobs + QC checks, File/folder convention (data/raw + manifest.jsonl), Metadata schema (one row per run)

### Community 27 - "Haptic Encoding Design"
Cohesion: 0.25
Nodes (8): Chords / strums, Design rules (or it turns to mush), Fret → 6 torso zones (12 frets onto 12 channels), Haptic encoding — note → actuator, Relationship to the coach, The 2-axis code, The premise (why this design exists), Worked example — string 6, fret 1 vs fret 2

### Community 28 - "Vibrotactile Perception Refs"
Cohesion: 0.25
Nodes (8): Decision: encode note as location + timing, not frequency, Perception References (Vibrotactile Findings), Funneling / sensory saltation (separate simultaneous cues across regions), Neosensory Buzz (Eagleman) — 4 wrist actuators, learnable code, Precedents: Model Human Cochlea, SoundShirt, Braille / the vOICe, Temporal acuity ~5ms gap detection (skin's strength) -> sharp pulses + sweep, Two-point discrimination (fingertip 2-4mm, forearm 35-40mm, torso/back 40-50mm), Vibrotactile frequency: peak 200-300 Hz, Weber ~20-30% (poor pitch)

### Community 29 - "SK473/KHD Tuning"
Cohesion: 0.29
Nodes (8): Decision: gut SK473 boards (PAM8403), one amp per 2 channels, Switch body actuators to SK473's own 12x KHD 3 Ohm/5 W drivers (de-housed), Tuning & Calibration Experiments (on-body), Experiment 1 — drive-frequency sweep per zone (back vs torso 60/100/150/200/250 Hz), Experiment 2 — punchiness (pulse shape 30/50/80 ms, flat vs sharp-decay), Experiment 4 — per-channel gain calibration (equal feel across 12 sites), The 4 tuning knobs (drive freq, pulse shape, coupling pressure, per-channel gain), Tuning safety bounds (never clip; surface <40C; felt level only)

### Community 30 - "AI Pitch & Scoping"
Cohesion: 0.25
Nodes (8): AI engine + pitch (for judges and the software team), Any song (generalizability), Build-first (MVP for max judge ROI), Honest scoping (this is rigor, not weakness), Rigor moves (cheap, high-credibility), The AI pieces (ranked by how non-gimmicky), The honest framing (say this to judges), Track / prize alignment

### Community 31 - "Open Questions / To-Do"
Cohesion: 0.25
Nodes (8): Open: ALSA Enumeration Reorder Across Reboots, Buy / make, Confirm, Decide, Open: KHD Driver Diameter Un-measured (spk_dia placeholder), Known limitations (state honestly in the demo), Repo / logistics, Open Questions + To-Do

### Community 32 - "Wiring Connectors (forbidden)"
Cohesion: 0.29
Nodes (7): 2 KHD Ø52 mm drivers (de-house → body), ⛔ NEVER connect (forbidden wiring rules), Powered hub / Anker 737 (5 V source, Mode A/B), SK473 captive 3.5 mm plug (box's 2 channels), SK473 board (L/R out, inside master speaker), SK473 captive USB-A lead (powers the PAM8403), Vantec output jack (1 of 4 stereo jacks)

### Community 33 - "System Block Diagram"
Cohesion: 0.29
Nodes (7): config/channel_map.json (source of truth), 12× KHD Ø52 mm 4Ω drivers (6 back + 6 torso), 🎸 guitar · 📷 webcam · 🎤 mic (inputs), LAPTOP — AI brain (transcribe · fuse · LLM coach · AR/UI), POWER · 5 V separate (Mode A hub / Mode B Anker 737), 6× SK473 box — PAM8403 amps (used intact), 2× Vantec NBA-200U 7.1 (CM6206) audio source

### Community 34 - "Cluster to Advice Semantics"
Cohesion: 0.33
Nodes (7): Causal clean/fault pair grounding, Eigenvector to advice semantics, Held-out (leave-one-player-out) validation, Hybrid scheme (supervised backbone + validated discovery), LDA supervised backbone (readable eigenvectors), Promotion gate for discovered sub-families, Prompted labels assign meaning (not discovered)

### Community 35 - "Safety & Wire Weight"
Cohesion: 0.29
Nodes (7): Electrical, Eliminate it — every joint, no exceptions, Mechanical — the wire weight problem (this breaks things), Mounting the actuator (so it's felt, not heard, and stays put), One-line pre-flight, Risk table (and how each is eliminated), Safety — eliminate every risk

### Community 36 - "Software Pipeline Overview"
Cohesion: 0.29
Nodes (7): "Any song", Compute placement (be honest in the demo), Pipeline, Software architecture, Stack, Suggested layout, Two timescales

### Community 37 - "Assembly Bring-up Gates"
Cohesion: 0.29
Nodes (7): Assembly checklist + bring-up (staged go/no-go gates), Stage 1 — prove audio path before any soldering (speaker-test -c8), Stage 2 — gut + tone-test each amp (12 channels buzz), Stage 4 — chord stress test (fire all 12 at once), Stage 5 — mount + couple (6 strings on back, 6 fret-zones on torso), Stage 6 — software bring-up (channel map, renderer, AI engine, coach), Vantec/ALSA enumeration trap (bind by USB port, verify channel order)

### Community 38 - "Perception to Design Mapping"
Cohesion: 0.29
Nodes (7): Funneling / sensory saltation — why we separate simultaneous hits, How each maps to our design, Perception references (why the encoding is shaped this way), Precedents (this approach works, with training), Spatial acuity — why sites are spread, and resolution lives on the limbs, Temporal acuity — the skin's strength, Vibrotactile frequency — why we don't encode pitch as frequency

### Community 40 - "Mode A Wall Power"
Cohesion: 0.33
Nodes (6): 10-port powered hub (~50 W AC, data + 5 V), Laptop (upstream USB data), MODE A — WALL (judged demo), 6× SK473 (USB-A — 5 V only), 3× Vantec (data + their 5 V), 🔌 Wall outlet

### Community 41 - "Mode B Cordless Power"
Cohesion: 0.33
Nodes (6): Anker 737 battery (24 000 mAh · 140 W), Laptop (Mode B data source), 6× SK473 USB-A (5 V), 3× Vantec (straight to laptop USB), MODE B — CORDLESS (walk-around), PD-powered USB-A hub

### Community 42 - "Feature Vector & Separability"
Cohesion: 0.33
Nodes (6): Concatenated feature vector (audio + vision named features), Pluck-proxy feature (attack RMS / onset slope), Redis nearest-neighbor mistake retrieval, Separability study (PCA to LDA, Fisher/silhouette/d-prime), Interval audit dashboard (anti-drift), Pluck control + pluck-sweep (confound)

### Community 43 - "Two-Stage Perception Model"
Cohesion: 0.40
Nodes (6): Pose-variation passes (D5), Stage 1 - occluded pose to (string,fret,finger), Stage 2 - buzz B + d to pressure/cause, The two last miles (occluded pose + buzz cause), Bayesian fusion with theory prior, Two-stage perception model (WHERE + HOW CLEAN)

### Community 44 - "System Architecture Chain"
Cohesion: 0.33
Nodes (6): Brain placement, Channel budget, Connector cheat-sheet (where confusion happens), System architecture, The full chain, every connector, Why two devices per "DAC + power"

### Community 45 - "Soldering Guide"
Cohesion: 0.33
Nodes (6): Job 1 — gut an SK473 (×6), Job 2 — attach actuators to the amp (×12), Job 3 — build the 5V solder bus (cordless / Mode B), Joint quality, Order of operations (matches [09-assembly-checklist.md](09-assembly-checklist.md)), Soldering guide

### Community 46 - "Psychophysics Brief"
Cohesion: 0.33
Nodes (5): 1. Tactile Psychophysics, 2. Tactile Illusions, 3. Prior Art, 4. Honest framing, REF — Vibrotactile Music Perception Research Brief (raw, cited)

### Community 47 - "Record Conductor (fallback)"
Cohesion: 0.60
Nodes (4): ask(), main(), show(), Zero-dep fallback (QuickTime + terminal conductor)

### Community 48 - "Wiring Map (channel)"
Cohesion: 0.40
Nodes (5): ALSA side, Labeling rule, The map, Why this layout, Wiring map — channel by channel

### Community 49 - "Project Pivot & Scope"
Cohesion: 0.50
Nodes (5): Scope lock: LEARN + PLAY guitar coaching, Pivot: Hearth to Tactus, AI only where the problem is unsolved (rigor split), Tactus direction & goal (Deaf-accessible guitar coaching), Honesty discipline (new sensory channel, ordinal pressure)

### Community 50 - "Capture Run Planner"
Cohesion: 0.50
Nodes (4): buildPlan() - interleaved run plan generator, Coverage grid + live QC stats, Arpeggiated chords (per-string labels + multi-finger pose), Prompted RUNS (sweep one condition across frets)

### Community 51 - "Docs Map / Overview"
Cohesion: 0.50
Nodes (4): Docs map, 🎸 Tactus, The 30-second version, The rig (as-built — full detail in [`truth.md`](truth.md))

### Community 52 - "alphaTab Loader JS"
Cohesion: 1.00
Nodes (3): AT(), loadChart(), loadScore()

## Knowledge Gaps
- **330 isolated node(s):** `The 30-second version`, `The rig (as-built — full detail in [`truth.md`](truth.md))`, `Docs map`, `What lives inside (per truth.md §3 — laptop is the brain, no Pi)`, `Render → STL (headless, no OpenSCAD)` (+325 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `KHD 3 Ohm / 5 W driver (the actuator, de-housed from SK473)` connect `Hardware BOM & Architecture` to `Actuator Puck CAD`?**
  _High betweenness centrality (0.132) - this node is a cross-community bridge._
- **Why does `actuator_puck (cup + button coupler, spk_dia=52)` connect `Actuator Puck CAD` to `Hardware BOM & Architecture`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Why does `Design Decisions (Why, and What We Rejected)` connect `Design Decisions (rejected)` to `Hardware BOM & Architecture`, `SK473/KHD Tuning`, `Build Refinements (channels)`, `Vibrotactile Perception Refs`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **What connects `# NOTE: the .scad's rim_lip subtraction is a geometric no-op (it lies inside`, `Cylinder of length h centered at (cx,cy,cz), running along axis x/y/z.`, `Vertical-edge rounded rectangular prism, min-corner at (0,0,z0).` to the rest of the system?**
  _368 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Hardware BOM & Architecture` be split into smaller, more focused modules?**
  _Cohesion score 0.06994535519125683 - nodes in this community are weakly interconnected._
- **Should `AI Rigor & Fusion Core` be split into smaller, more focused modules?**
  _Cohesion score 0.0545876887340302 - nodes in this community are weakly interconnected._
- **Should `CAD Enclosure (brain-pack)` be split into smaller, more focused modules?**
  _Cohesion score 0.08205128205128205 - nodes in this community are weakly interconnected._