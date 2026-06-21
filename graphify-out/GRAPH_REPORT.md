# Graph Report - .  (2026-06-21)

## Corpus Check
- 81 files · ~94,906 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 730 nodes · 1221 edges · 49 communities (46 shown, 3 thin omitted)
- Extraction: 64% EXTRACTED · 35% INFERRED · 0% AMBIGUOUS · INFERRED: 432 edges (avg confidence: 0.84)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Analysis Audit & Reporting|Analysis Audit & Reporting]]
- [[_COMMUNITY_Data Capture & Pipeline|Data Capture & Pipeline]]
- [[_COMMUNITY_Fretboard Vision Detection|Fretboard Vision Detection]]
- [[_COMMUNITY_Haptic SpeakerResonance Check|Haptic Speaker/Resonance Check]]
- [[_COMMUNITY_Haptic Rendering & Research|Haptic Rendering & Research]]
- [[_COMMUNITY_Audio Feature Extraction|Audio Feature Extraction]]
- [[_COMMUNITY_Hardware Build & BOM|Hardware Build & BOM]]
- [[_COMMUNITY_Separability Collapse Metrics|Separability Collapse Metrics]]
- [[_COMMUNITY_Audio Event Segmentation|Audio Event Segmentation]]
- [[_COMMUNITY_Docs Perception & Encoding|Docs: Perception & Encoding]]
- [[_COMMUNITY_Vision Feature Extraction|Vision Feature Extraction]]
- [[_COMMUNITY_Enclosure CAD (code)|Enclosure CAD (code)]]
- [[_COMMUNITY_Vision Registration & Changelog|Vision Registration & Changelog]]
- [[_COMMUNITY_BuildAssembly Docs|Build/Assembly Docs]]
- [[_COMMUNITY_Analysis Schema & Mapping|Analysis Schema & Mapping]]
- [[_COMMUNITY_Enclosure CAD Renders|Enclosure CAD Renders]]
- [[_COMMUNITY_12-Channel Body Map|12-Channel Body Map]]
- [[_COMMUNITY_Source of Truth & Architecture|Source of Truth & Architecture]]
- [[_COMMUNITY_AIML Rigor & Sponsors|AIML Rigor & Sponsors]]
- [[_COMMUNITY_LEARN Loop & Correction|LEARN Loop & Correction]]
- [[_COMMUNITY_Sponsors & Deaf Framing|Sponsors & Deaf Framing]]
- [[_COMMUNITY_LEARN Interface & Eval|LEARN Interface & Eval]]
- [[_COMMUNITY_AIML Design Decisions|AIML Design Decisions]]
- [[_COMMUNITY_Channel Map Config|Channel Map Config]]
- [[_COMMUNITY_AI Rigor & Pitch|AI Rigor & Pitch]]
- [[_COMMUNITY_Capture HTTP Server|Capture HTTP Server]]
- [[_COMMUNITY_Power Modes & Safety|Power Modes & Safety]]
- [[_COMMUNITY_BuzzFault Taxonomy|Buzz/Fault Taxonomy]]
- [[_COMMUNITY_Actuator Puck CAD|Actuator Puck CAD]]
- [[_COMMUNITY_Wiring Connections|Wiring Connections]]
- [[_COMMUNITY_PowerDevice Topology|Power/Device Topology]]
- [[_COMMUNITY_Calibration Server|Calibration Server]]
- [[_COMMUNITY_Demo Data Generator|Demo Data Generator]]
- [[_COMMUNITY_System Flow Diagram|System Flow Diagram]]
- [[_COMMUNITY_AR Interface & Viz|AR Interface & Viz]]
- [[_COMMUNITY_ArUco Pose-Lock JS|ArUco Pose-Lock JS]]
- [[_COMMUNITY_Audit Tests|Audit Tests]]
- [[_COMMUNITY_Segmentation Tests|Segmentation Tests]]
- [[_COMMUNITY_Audio Feature Tests|Audio Feature Tests]]
- [[_COMMUNITY_Vision Feature Tests|Vision Feature Tests]]
- [[_COMMUNITY_Pipeline Runner|Pipeline Runner]]
- [[_COMMUNITY_Collapse Tests|Collapse Tests]]
- [[_COMMUNITY_Experiment Matrix Plan|Experiment Matrix Plan]]
- [[_COMMUNITY_Record Conductor|Record Conductor]]
- [[_COMMUNITY_End-to-End Pipeline Test|End-to-End Pipeline Test]]
- [[_COMMUNITY_alphaTab Loader JS|alphaTab Loader JS]]
- [[_COMMUNITY_ArUco Marker Validation|ArUco Marker Validation]]

## God Nodes (most connected - your core abstractions)
1. `truth.md — Tactus Source of Truth` - 36 edges
2. `Tactus README` - 17 edges
3. `AI Rigor` - 16 edges
4. `Start Here` - 15 edges
5. `Perception References` - 15 edges
6. `run()` - 14 edges
7. `main()` - 14 edges
8. `Haptic Encoding` - 14 edges
9. `Sponsor & prize alignment` - 14 edges
10. `Offline processing contract` - 14 edges

## Surprising Connections (you probably didn't know these)
- `SoundShirt` --semantically_similar_to--> `12-Channel Body Map (6 back strings + 6 torso fret-zones)`  [INFERRED] [semantically similar]
  docs/12-perception-references.md → truth.md
- `MicroCenter Receipt 195-PO-419168` --shares_data_with--> `truth.md — Tactus Source of Truth`  [INFERRED]
  docs/01-bill-of-materials.md → truth.md
- `Pivot: Hearth → Tactus` --references--> `Tactus (Deaf-Accessible Guitar-Coaching Wearable)`  [INFERRED]
  CHANGELOG.md → truth.md
- `Tactus README` --references--> `As-Built Rig (Laptop→Vantec→SK473→KHD)`  [INFERRED]
  README.md → truth.md
- `Tactus README` --references--> `Locked Software Architecture (Browser/Python split)`  [INFERRED]
  README.md → truth.md

## Hyperedges (group relationships)
- **Multimodal Perception Fusion** — docs_08_software_architecture_audio_branch, docs_08_software_architecture_vision_branch, truth_fusion_model, docs_17_ai_rigor_buzz_inverse [INFERRED 0.85]
- **LEARN Closed-Loop Coaching** — docs_13_learn_web_and_visualization_closing_the_loop, docs_13_learn_web_and_visualization_haptic_reference, docs_13_learn_web_and_visualization_feel_the_difference, docs_13_learn_web_and_visualization_vlm_coach, docs_13_learn_web_and_visualization_physical_correction_engine [INFERRED 0.85]
- **12-Channel Audio to Haptic Signal Chain** — truth_vantec_nba_200u, docs_10_design_decisions_sk473_pam8403, truth_khd_driver, truth_channel_map_12 [INFERRED 0.85]
- **Offline data → features → separability pipeline** — docs_24_data_collection_protocol_prompted_runs, docs_25_data_and_feature_format_processing_contract, docs_26_aiden_handoff_pipeline_build, docs_23_data_and_cluster_semantics_training_procedure [INFERRED 0.85]
- **Separability cluster proof + named-axis viz** — docs_20_aiml_training_design_separability_study, docs_23_data_and_cluster_semantics_readable_eigenvector, docs_27_chord_feedback_and_experiment_plan_3d_semantic_viz [INFERRED 0.75]
- **Phase-locked screen+body+sound feedback** — docs_22_interface_ar_and_correction_coherence, docs_21_chord_and_sustain_rendering_per_string_envelope_principle, web_index_body_map [INFERRED 0.75]

## Communities (49 total, 3 thin omitted)

### Community 0 - "Analysis Audit & Reporting"
Cohesion: 0.08
Nodes (57): _amber_ramp(), _assemble_html(), _build_recommendations(), _card(), _coerce_float(), _confusion_from_block(), _confusion_png(), _coverage_counts() (+49 more)

### Community 1 - "Data Capture & Pipeline"
Cohesion: 0.05
Nodes (55): Capture UI (prompted data collection), Coverage grid + intent check, Device picker + clip/QC meter, Lossless WAV (AudioWorklet) capture, Manifest row writer, Run-plan blocks (core-grid/pose/arpeggio/strum/chord-stream), Visual metronome + count-in, Capture tooling overview (+47 more)

### Community 2 - "Fretboard Vision Detection"
Cohesion: 0.1
Nodes (35): board_grid(), aggregate_frames(), autodetect(), detect_frets(), detect_neck_edges(), fit_law(), _gray(), overlay() (+27 more)

### Community 3 - "Haptic Speaker/Resonance Check"
Cohesion: 0.11
Nodes (35): getkey(), main(), make_glide(), Read ONE keypress, no Enter required, returned lowercased.      Enter comes back, A linear-frequency chirp f0 -> f1 over `dur` s. Driving through the band     mak, ask(), channel_label(), diagnose() (+27 more)

### Community 4 - "Haptic Rendering & Research"
Cohesion: 0.07
Nodes (36): Channel-count drift fix (as-built = 12), Demo-floor insurance (§F0), Failure modes / critical silent gaps, Leave-one-player-out validation, Verification plan (V1–V10), Chord bloom (simultaneous pluck), Fret-zones carry chord shape, Sustain looped shimmer (LOCKED) (+28 more)

### Community 5 - "Audio Feature Extraction"
Cohesion: 0.09
Nodes (30): _chroma(), _extract_one(), _f0_features(), _flux(), _hnr(), _inharmonicity(), main(), _mfcc() (+22 more)

### Community 6 - "Hardware Build & BOM"
Cohesion: 0.14
Nodes (22): Actuator Puck CAD (cup + button), CAD — Enclosure and Pucks, Driver Ø 52 mm (open dimension), Tactus Enclosure (manifold3d, watertight), Bill of Materials, LEO 40 mm Speakers (spares), MicroCenter Receipt 195-PO-419168, Gut the SK473 (Job 1) (+14 more)

### Community 7 - "Separability Collapse Metrics"
Cohesion: 0.16
Nodes (20): _dprime(), _f(), _fisher_per_axis(), _fit_fold(), _jsonify(), _kfold_folds(), _lopo_folds(), main() (+12 more)

### Community 8 - "Audio Event Segmentation"
Cohesion: 0.16
Nodes (20): _as_int(), _detect_onsets(), _empty_frame(), _events_for_row(), _f0_median_hz(), _get(), _load_audio(), main() (+12 more)

### Community 9 - "Docs: Perception & Encoding"
Cohesion: 0.18
Nodes (21): Vest-Primary Form Factor, 12-Channel Wiring Table, 2-Axis Code (string→back, fret→torso), Encoding Design Rules (separate simultaneous sites), Fret → Torso Zone + Intensity, Haptic Encoding, Encode Pitch as Location + Timing, 160 Hz / 50 ms Pulse (+13 more)

### Community 10 - "Vision Feature Extraction"
Cohesion: 0.15
Nodes (19): _angle_at(), _default_provider(), _event_features(), _load_active_keyframe(), main(), _nearest_fret(), _pose_angles(), Board Y (0..1, 0 = low-E) -> string name 6..1 (low-E=6 .. high-e=1).     schema. (+11 more)

### Community 11 - "Enclosure CAD (code)"
Cohesion: 0.25
Nodes (17): box_at(), build_base(), build_lid(), cleanup(), cyl(), D(), main(), make_text() (+9 more)

### Community 12 - "Vision Registration & Changelog"
Cohesion: 0.12
Nodes (18): ArUco Marker Print Sheet, ArUco 4×4_50 id 0 Marker, ArUco Marker — Optional Validator, Dropped Bass Shakers from Wearable, LEARN Re-architected as Vision+Haptic Closed Loop, Localhost Capture App (synced A/V + metadata), Markerless Primary, ArUco Optional (reconciliation), Pivot: Hearth → Tactus (+10 more)

### Community 13 - "Build/Assembly Docs"
Cohesion: 0.22
Nodes (16): Start Here, Sense → Understand → Render, System Architecture, Power Architecture, Attach Actuators (Job 2), Soldering Guide, Wiring Map, Bounded Isolation (signal ground + isolated 5V) (+8 more)

### Community 14 - "Analysis Schema & Mapping"
Cohesion: 0.13
Nodes (14): abspath(), f0_to_string_fret(), hz_to_midi(), iter_session_player(), on_path(), out_dir(), Yield (session_id, player_id, dir) for every <session>/<player> under `base`., data/analysis/<session>/<player>/, created if missing. (+6 more)

### Community 15 - "Enclosure CAD Renders"
Cohesion: 0.19
Nodes (15): Base Shell, Corner Screw Bosses, Enclosure Preview (CAD), Internal Mounting Standoffs, Lid, Port Window, TACTUS Impression, Ø16 Wire Holes (USB-A Fits) (+7 more)

### Community 16 - "12-Channel Body Map"
Cohesion: 0.14
Nodes (15): Back String Sites (ch 1–6), Tactus Body Map — 12 Channels, Fret Zone 1 — Nut (ch 7), Fret Zone 2 (ch 8), Fret Zone 3 (ch 9), Fret Zone 4 (ch 10), Fret Zone 5 (ch 11), Fret Zone 6 — Body (ch 12) (+7 more)

### Community 17 - "Source of Truth & Architecture"
Cohesion: 0.19
Nodes (15): Song-Agnostic Engine (free play / coach), Haptic Renderer (12-ch ALSA), Software Architecture, Two Timescales (per-note vs per-phrase), WebSocket JSON Contract (the parallel seam), Pipeline / Explainability View, Honesty Discipline, LEARN Mode (+7 more)

### Community 18 - "AIML Rigor & Sponsors"
Cohesion: 0.17
Nodes (15): AWS Trainium contrastive embedding, Most Technical: fusion + buzz inverse, Redis coach memory, Contrastive multimodal embedding (D7), Engineered-feature core (LDA/UMAP), Separability study (D5), Causal clean/fault pair, Cluster→advice epistemics (+7 more)

### Community 19 - "LEARN Loop & Correction"
Cohesion: 0.2
Nodes (14): Audio Branch (F0 + technique), Decision: Pressure from Timbre (no sensor), Closing the Loop (feel target, see fix), Deepgram Hands-Free Voice, Feel-the-Difference Replay, Haptic Reference (Component A), LEARN Web and Visualization, Local Memory — Redis Offline Practice (+6 more)

### Community 20 - "Sponsors & Deaf Framing"
Cohesion: 0.16
Nodes (14): Accessibility / Ddoski's World, Coherent sponsor story, Deepgram hands-free voice, QNX dropped (Pi cut), Sponsor & prize alignment, Honest deaf framing (Deaf Gain), Frequency discrimination (Weber ~12-21%), Prior art devices (VEST/Emoti-Chair) (+6 more)

### Community 21 - "LEARN Interface & Eval"
Cohesion: 0.16
Nodes (14): Arize/Phoenix trace + eval, Evaluation protocol / headline ablation, Per-finger fingertip-to-wire distance d, Browser→Python vision-feature schema (T9), Two-class pressure too-light vs good (D5), Per-finger placement + pressure display, Targeted 2D correction fretboard, Bring-up order (+6 more)

### Community 22 - "AIML Design Decisions"
Cohesion: 0.18
Nodes (14): Computer Vision track, Bayesian position posterior w/ theory prior (D8), Eight locked decisions (D1–D8), Fretboard-relative frame, Hybrid vision stack (D3), Multimodal technique-coaching thesis, Occluded pose→placement model, Prompted + audio-verified labeling (D2) (+6 more)

### Community 23 - "Channel Map Config"
Cohesion: 0.15
Nodes (12): channels, _comment, intensity_levels, pulse, duration_ms, freq_hz, _note, spare (+4 more)

### Community 24 - "AI Rigor & Pitch"
Cohesion: 0.26
Nodes (12): Removed AI-Composed Haptic Score, Ablation: Frequency vs 2-Axis Spatial Code, AI and Pitch, Honest Framing for Judges, Mini Discrimination Study (n=3), AI Rigor, Arize / Phoenix Tracing & Eval, Live 3D Cluster View (audio-only ↔ fused toggle) (+4 more)

### Community 25 - "Capture HTTP Server"
Cohesion: 0.29
Nodes (3): Handler, Aggregate a session's manifest so the page can cross-check what's on disk., safe()

### Community 26 - "Power Modes & Safety"
Cohesion: 0.2
Nodes (11): Anker 737 PowerCore 24K, Chord-Safety Power Guarantee, IPSG 10-Port Powered Hub, Mode A — Wall Power (10-port hub), Mode B — Cordless (Anker 737 bus), Build 5V Solder Bus (Job 3), USB-C CC Resistor (5.1kΩ), Temporal Acuity (~5ms) (+3 more)

### Community 27 - "Buzz/Fault Taxonomy"
Cohesion: 0.22
Nodes (11): Buzz inverse problem (D4→D6), Feedback decomposition (granular advice), Pluck/attack-energy confound (H1), Pluck-proxy feature, Renderer input stream, Capture matrix (position grid + classes), Class definitions (clean/buzz-light/buzz-placement/muted/choked), Pluck control + pluck-sweep (+3 more)

### Community 28 - "Actuator Puck CAD"
Cohesion: 0.4
Nodes (9): box(), button(), cup(), cyly(), cylz(), D(), main(), # NOTE: the .scad's rim_lip subtraction is a geometric no-op (it lies inside (+1 more)

### Community 29 - "Wiring Connections"
Cohesion: 0.27
Nodes (10): Tactus Connections (No Soldering) Diagram, KHD Ø52 mm Drivers, Laptop USB-A x3, PAM8403 Amplifier, Powered Hub / Anker 737, SK473 Captive 3.5 mm Plug, SK473 Board (L / R Out), SK473 Captive USB-A Lead (+2 more)

### Community 30 - "Power/Device Topology"
Cohesion: 0.27
Nodes (10): 10-Port Powered Hub, 3× Vantec, 6× SK473, Anker 737 Battery, Tactus Power Modes Diagram, Laptop, Mode A — Wall (judged demo), Mode B — Cordless (walk-around) (+2 more)

### Community 32 - "Demo Data Generator"
Cohesion: 0.43
Nodes (6): build(), _hz(), main(), _note(), One plucked note: decaying harmonic stack, optional broadband buzz burst, OR, Write a synthetic batch under raw_dir. Returns the list of player dirs.

### Community 33 - "System Flow Diagram"
Cohesion: 0.52
Nodes (7): Tactus Render Chain (System Flow Diagram), Guitar, Webcam & Mic Inputs, 12x KHD 52mm 4-Ohm Drivers (Actuators), Laptop (AI Brain), 5V Power Rail (Separate), 6x SK473 Amp Box (PAM8403), 2x Vantec NBA-200U USB Audio Adapter (CM6206)

### Community 34 - "AR Interface & Viz"
Cohesion: 0.29
Nodes (7): Claude vision coaching brain (Anthropic), Best UI/UX (AI made visible), Live 3D cluster viz, Phase-locked coherence (screen+body+sound), Every pixel is a sense (governing law), Two coupled views (AR play + 2D correction), 3D semantic viz (centerpiece)

### Community 36 - "Audit Tests"
Cohesion: 0.47
Nodes (5): _build_events_df(), _build_metrics(), main(), ~30 rows across the 3 core classes / strings 6..1 / frets 1..6.      Determinist, Minimal metrics dict matching collapse.py's shape (docs/24 §8):       split, cla

### Community 37 - "Segmentation Tests"
Cohesion: 0.53
Nodes (5): _make_manifest(), _make_wav(), _pluck(), A decaying sinusoid ('pluck') followed by silence. The envelope is tapered     t, test_segment()

### Community 38 - "Audio Feature Tests"
Cohesion: 0.47
Nodes (5): _event_row(), main(), 0.6 s exponentially-decaying 110 Hz tone followed by 0.4 s of silence., One events.csv row filling every schema.EVENT_COLUMNS field., _write_a2_wav()

### Community 39 - "Vision Feature Tests"
Cohesion: 0.47
Nodes (5): _build_twin(), main(), _make_provider(), Synthetic twin: 4 image corners (a foreshortened-ish rectangle) matched to     b, Stub provider returning a known index-fingertip pixel per event.      event 'v#0

### Community 40 - "Pipeline Runner"
Cohesion: 0.6
Nodes (4): main(), _manifests(), Run the full pipeline over every manifest under raw_dir (optionally filtered, run_batch()

### Community 41 - "Collapse Tests"
Cohesion: 0.6
Nodes (4): _dprime(), main(), _make_df(), Pull pairwise d'(a,b) out of whatever key shape collapse used.

### Community 42 - "Experiment Matrix Plan"
Cohesion: 0.5
Nodes (5): Experiment matrix (E1–E9), Harmonic-residual mono→poly transfer (H2), Rigor rules (LOPO, train-fold only), Three deliverables (quality/position/viz), Vision position beats raw MediaPipe (H4)

### Community 43 - "Record Conductor"
Cohesion: 0.83
Nodes (3): ask(), main(), show()

### Community 44 - "End-to-End Pipeline Test"
Cohesion: 0.67
Nodes (3): _buzz_pair_dprime(), Pull d'(buzz-light, buzz-placement) out of whatever key shape collapse used., test_end_to_end()

### Community 45 - "alphaTab Loader JS"
Cohesion: 1.0
Nodes (3): AT(), loadChart(), loadScore()

## Ambiguous Edges - Review These
- `Channel-count drift fix (as-built = 12)` → `12-channel body map`  [AMBIGUOUS]
  docs/20-eng-review.md · relation: conceptually_related_to
- `Chord-field AR prototype (depth blades)` → `alphaTab tab sync (single source)`  [AMBIGUOUS]
  web/ar.html · relation: conceptually_related_to

## Knowledge Gaps
- **191 isolated node(s):** `_comment`, `intensity_levels`, `freq_hz`, `duration_ms`, `_note` (+186 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Channel-count drift fix (as-built = 12)` and `12-channel body map`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Chord-field AR prototype (depth blades)` and `alphaTab tab sync (single source)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `truth.md — Tactus Source of Truth` connect `Source of Truth & Architecture` to `Hardware Build & BOM`, `Docs: Perception & Encoding`, `Vision Registration & Changelog`, `Build/Assembly Docs`, `LEARN Loop & Correction`, `AI Rigor & Pitch`, `Power Modes & Safety`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **Why does `Sponsor & prize alignment` connect `Sponsors & Deaf Framing` to `AR Interface & Viz`, `AIML Rigor & Sponsors`, `LEARN Interface & Eval`, `AIML Design Decisions`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **Why does `Live AR Beat-Saber depth-approach play` connect `Data Capture & Pipeline` to `AR Interface & Viz`, `Haptic Rendering & Research`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._
- **Are the 23 inferred relationships involving `truth.md — Tactus Source of Truth` (e.g. with `Tactus (Deaf-Accessible Guitar-Coaching Wearable)` and `LEARN Mode`) actually correct?**
  _`truth.md — Tactus Source of Truth` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Tactus README` (e.g. with `As-Built Rig (Laptop→Vantec→SK473→KHD)` and `Locked Software Architecture (Browser/Python split)`) actually correct?**
  _`Tactus README` has 2 INFERRED edges - model-reasoned connections that need verification._