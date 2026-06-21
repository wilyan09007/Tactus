# Graph Report - .  (2026-06-20)

## Corpus Check
- 3 files · ~51,706 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 395 nodes · 535 edges · 23 communities (21 shown, 2 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 69 edges (avg confidence: 0.84)
- Token cost: 79,661 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_AI Engine & Coaching Core|AI Engine & Coaching Core]]
- [[_COMMUNITY_Project Overview & Hardware BOM|Project Overview & Hardware BOM]]
- [[_COMMUNITY_ML Training & Buzz Data Pipeline|ML Training & Buzz Data Pipeline]]
- [[_COMMUNITY_Assembly Stages & Design Decisions|Assembly Stages & Design Decisions]]
- [[_COMMUNITY_Software Modules & Dependencies|Software Modules & Dependencies]]
- [[_COMMUNITY_Haptic Perception Research & Prior Art|Haptic Perception Research & Prior Art]]
- [[_COMMUNITY_Enclosure CAD (box + lid)|Enclosure CAD (box + lid)]]
- [[_COMMUNITY_Actuator Puck Coupling CAD|Actuator Puck Coupling CAD]]
- [[_COMMUNITY_Enclosure Build Script (Python)|Enclosure Build Script (Python)]]
- [[_COMMUNITY_Channel Map (strings + fret-zones)|Channel Map (strings + fret-zones)]]
- [[_COMMUNITY_Enclosure CAD Renders|Enclosure CAD Renders]]
- [[_COMMUNITY_Body-Plate & Driver Socket CAD|Body-Plate & Driver Socket CAD]]
- [[_COMMUNITY_Chord Rendering & AR Interface|Chord Rendering & AR Interface]]
- [[_COMMUNITY_Multimodal Vision & Buzz Modeling|Multimodal Vision & Buzz Modeling]]
- [[_COMMUNITY_Power, Soldering & Safety|Power, Soldering & Safety]]
- [[_COMMUNITY_Audio Wiring Connections|Audio Wiring Connections]]
- [[_COMMUNITY_System Components Overview|System Components Overview]]
- [[_COMMUNITY_Power Mode A (Wall)|Power Mode A (Wall)]]
- [[_COMMUNITY_Power Mode B (Cordless)|Power Mode B (Cordless)]]
- [[_COMMUNITY_Open Questions & To-Do|Open Questions & To-Do]]
- [[_COMMUNITY_Power Cradle CAD|Power Cradle CAD]]
- [[_COMMUNITY_Vantec USB Audio Path|Vantec USB Audio Path]]

## God Nodes (most connected - your core abstractions)
1. `Data Collection & Labeling Protocol (Execute to Feed AIML Pipeline)` - 21 edges
2. `Data, Representation, Training & Eigenvector-to-Advice Semantics` - 18 edges
3. `The AI Core: Audio + Vision to (String, Fret, Finger, Quality)` - 14 edges
4. `base() — enclosure base shell + mounts + vents` - 13 edges
5. `KHD 3 Ohm / 5 W driver (the actuator, de-housed from SK473)` - 11 edges
6. `Vibrotactile Music Perception Research Brief` - 11 edges
7. `LEARN: Closing the Loop (Feel Target, See Fix)` - 10 edges
8. `Eng Review of the AIML Training Design` - 10 edges
9. `ai/ Module (transcription, vision, fusion, LLM coach)` - 10 edges
10. `build_lid()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `MockEngine (WebSocket message emitter)` --semantically_similar_to--> `LLM Coach Loop`  [INFERRED] [semantically similar]
  web/index.html → software/README.md
- `Dropped heavy bass shakers (weight/physics)` --rationale_for--> `KHD 3 Ohm / 5 W driver (the actuator, de-housed from SK473)`  [INFERRED]
  CHANGELOG.md → docs/01-bill-of-materials.md
- `LEARN vision+haptic closed-loop coaching` --conceptually_related_to--> `LEARN mode (Rocksmith for Deaf players)`  [INFERRED]
  CHANGELOG.md → README.md
- `Vision branch (MediaPipe hand pose + ArUco fretboard homography)` --conceptually_related_to--> `LEARN mode (Rocksmith for Deaf players)`  [INFERRED]
  docs/08-software-architecture.md → README.md
- `Reference diff + any-song engine` --conceptually_related_to--> `The one correction (3 axes: note, duration, pressure)`  [INFERRED]
  docs/08-software-architecture.md → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Five Mechanisms That Turn a Cluster Into Advice** — docs_23_data_and_cluster_semantics_cluster_to_advice, docs_23_data_and_cluster_semantics_lda_eigenvector, docs_23_data_and_cluster_semantics_medoid_prototype, docs_23_data_and_cluster_semantics_causal_clean_fault_pair, docs_23_data_and_cluster_semantics_heldout_validation [EXTRACTED 1.00]
- **Labeled-by-Construction Capture Flow (Prompt → Run → Segment → Manifest)** — docs_24_data_collection_protocol_prompt_is_label, docs_24_data_collection_protocol_prompted_runs, docs_24_data_collection_protocol_onset_segmentation, docs_24_data_collection_protocol_metadata_schema, docs_24_data_collection_protocol_file_convention [EXTRACTED 0.85]
- **Separability Study Pipeline (Features → PCA → LDA → LOPO d′)** — docs_23_data_and_cluster_semantics_feature_vector, docs_23_data_and_cluster_semantics_lda_eigenvector, docs_23_data_and_cluster_semantics_lopo_training, truth_separability_study [INFERRED 0.85]

## Communities (23 total, 2 thin omitted)

### Community 0 - "AI Engine & Coaching Core"
Cohesion: 0.05
Nodes (48): AI Engine + Pitch (for Judges and Software Team), AI Where Unsolved, Deterministic Where Solved, Buzz Inverse (Pressure from Sound), Deterministic Renderer (Encoder + Pulse Synth), Fusion Perception Model (Vision + Audio), Any-Song Generalizability (Free Play + Coach Mode), LLM Coach (Anthropic, Phrase-Level), MediaPipe Hand Pose + ArUco Fretboard Homography (+40 more)

### Community 1 - "Project Overview & Hardware BOM"
Cohesion: 0.09
Nodes (42): Pivot: Hearth to Tactus, LEARN vision+haptic closed-loop coaching, Removed score-based architecture (direct transcription), Dropped heavy bass shakers (weight/physics), Tactus Project Changelog, Start here (teammate on-ramp), Build-tonight TL;DR, Bill of materials (as-purchased) (+34 more)

### Community 2 - "ML Training & Buzz Data Pipeline"
Cohesion: 0.09
Nodes (41): Pluck Confound in Buzz Inverse (D1), Data, Representation, Training & Eigenvector-to-Advice Semantics, Bayesian Fusion (Likelihoods + Adaptive Theory Prior), Causal Clean/Fault Pairs (Proves the Fix), How a Cluster Becomes Advice (Epistemics), Contrastive Multimodal Embedding (Cut-First Stretch), Data Collection Protocol (Prompted, Interleaved, Two-Pass), Engineered Feature Vector (~40-60 Dims, Audio + Vision) (+33 more)

### Community 3 - "Assembly Stages & Design Decisions"
Cohesion: 0.06
Nodes (40): Assembly checklist + bring-up (staged go/no-go gates), Stage 1 — prove audio path before any soldering (speaker-test -c8), Stage 2 — gut + tone-test each amp (12 channels buzz), Stage 4 — chord stress test (fire all 12 at once), Stage 5 — mount + couple (6 strings on back, 6 fret-zones on torso), Stage 6 — software bring-up (channel map, renderer, AI engine, coach), Decision: bass dropped (was Logitech Z313), Decision: brain is the laptop (M4 Pro 48 GB), Pi cut from critical path (+32 more)

### Community 4 - "Software Modules & Dependencies"
Cohesion: 0.09
Nodes (27): ai/ Module (transcription, vision, fusion, LLM coach), ALSA Multichannel Write (3 Vantec / 8ch), LLM Coach Loop, haptic/ Module (encoder, synth, alsa_out), Honest-Scoping Rules (coarse placement, no pressure sensor), Tactus Software (laptop, two halves), anthropic, aubio (+19 more)

### Community 5 - "Haptic Perception Research & Prior Art"
Cohesion: 0.08
Nodes (26): Audio Branch (pYIN/YIN live, CREPE/basic-pitch offline), Audio to Haptic ML/DSP Research Brief, Information Transfer Evaluation (IT bits, d-prime), Learned Model Prior Art (HapticGen, HapticLDM, Sound2Hap), MPEG-I Part 31 Haptics Coding (ISO/IEC 23090-31:2025), Pitch Engines (CREPE, pYIN, basic-pitch, SwiftF0), Psychohaptic Mapping (Warp Not Transpose, Mel/Bark/ERB), Source Separation (Spleeter, Demucs, HT-Demucs) (+18 more)

### Community 6 - "Enclosure CAD (box + lid)"
Cohesion: 0.10
Nodes (22): SK473 amp+driver units mount on the VEST (decision), FlashForge Adventurer 5M (220x220x220 bed), Headless manifold3d render pipeline (no OpenSCAD on arm64), tactus_box (superseded split enclosure), tactus_enclosure (base+lid, one print), tactus_enclosure_plate.stl (175x202 one-print bed), tactus_power_cradle (vented sled), tactus_box.scad — main electronics enclosure (brain-pack) (+14 more)

### Community 7 - "Actuator Puck Coupling CAD"
Cohesion: 0.16
Nodes (18): actuator_puck.scad — Ø52 driver coupling puck (cup + button), button() — domed contact button bonded to the dust cap, cup() — rigid coupling cup that grips the Ø52 driver frame, box(), button(), cup(), cyly(), cylz() (+10 more)

### Community 8 - "Enclosure Build Script (Python)"
Cohesion: 0.25
Nodes (17): box_at(), build_base(), build_lid(), cleanup(), cyl(), D(), main(), make_text() (+9 more)

### Community 9 - "Channel Map (strings + fret-zones)"
Cohesion: 0.13
Nodes (15): Back ch5 — A string site, Back ch2 — B string site, Back ch4 — D string site, Back ch3 — G string site, Back ch1 — high E string site, Back ch6 — low E string site, BACK — 6 strings (ch 1–6), config/channel_map.json (source of truth) (+7 more)

### Community 10 - "Enclosure CAD Renders"
Cohesion: 0.18
Nodes (14): Enclosure base box (open top, blue CAD preview), Corner screw bosses on base, Enclosure lid, interior view (blue CAD preview), Triangulated rib / support structure on lid underside, Corner screw holes on lid, Internal partitions / component mounting structure, Rectangular port window cutout in base side wall, Row of circular wire/vent holes (Ø16 mm, USB-A pass-through) (+6 more)

### Community 11 - "Body-Plate & Driver Socket CAD"
Cohesion: 0.26
Nodes (14): tactus_chest_plate.scad — body-contoured plate holding 6 string drivers, bore_at(z) — through-hole so the cone/button reaches skin, shell() — curved torso-cylinder shell panel between z_lo..z_hi, socket_at(z) — places a driver_socket on the inner face at height z, strap_slots(z_lo, z_hi) — edge slots to lash onto the laser-tag vest, tile(z_lo, z_hi) — one print tile (shell + sockets - bores - slots), tile_abs(z_lo, z_hi) — print tile built at absolute z coordinates, tactus_node_mount.scad — per-node split-clamp + telescoping driver mount (+6 more)

### Community 12 - "Chord Rendering & AR Interface"
Cohesion: 0.18
Nodes (13): Chord onset encodes how it was struck (strum sweep / pluck bloom / arpeggio), Chord, Sustain & Percussive Rendering, Fret-zones carry chord SHAPE (pulsed in sync) + root emphasis, GSTACK eng-review report (PAUSED; B1-B7 open, 3 critical silent-failure gates), Percussive & muted hits (sharp pop across motors; muted strings don't fire), Renderer input: per-string (onset, pitch->string/fret, envelope, decay) stream, Sustain = envelope-modulated looped shimmer (LOCKED), One phase-locked signal coherence (screen + body + sound from one stream) (+5 more)

### Community 13 - "Multimodal Vision & Buzz Modeling"
Cohesion: 0.24
Nodes (11): Vision Branch (MediaPipe + ArUco Homography), AIML Training Design (Multimodal Coaching), Bayesian Position Posterior (Theory Prior), Buzz Inverse Problem (B=f(P,d)), Contrastive Multimodal Embedding, Live 3D Cluster Visualization, Occlusion Pose-to-Placement Model, Pressure as Ordinal (Not Newtons) (+3 more)

### Community 14 - "Power, Soldering & Safety"
Cohesion: 0.25
Nodes (9): USB-C 5.1 kohm CC resistor requirement, Per-bus load math (3/2/2 split), Power architecture + chord-safety guarantee, Job 1 — gut an SK473 into a bare amp brick, Soldering guide (3 jobs), Strain relief (heat-shrink + hot-glue blob), Bounded isolation (signal ground vs 5V rail), Safety checklist (electrical + mechanical) (+1 more)

### Community 15 - "Audio Wiring Connections"
Cohesion: 0.29
Nodes (7): 2 KHD Ø52 mm drivers (de-house → body), ⛔ NEVER connect (forbidden wiring rules), Powered hub / Anker 737 (5 V source, Mode A/B), SK473 captive 3.5 mm plug (box's 2 channels), SK473 board (L/R out, inside master speaker), SK473 captive USB-A lead (powers the PAM8403), Vantec output jack (1 of 4 stereo jacks)

### Community 16 - "System Components Overview"
Cohesion: 0.29
Nodes (7): config/channel_map.json (source of truth), 12× KHD Ø52 mm 4Ω drivers (6 back + 6 torso), 🎸 guitar · 📷 webcam · 🎤 mic (inputs), LAPTOP — AI brain (transcribe · fuse · LLM coach · AR/UI), POWER · 5 V separate (Mode A hub / Mode B Anker 737), 6× SK473 box — PAM8403 amps (used intact), 2× Vantec NBA-200U 7.1 (CM6206) audio source

### Community 17 - "Power Mode A (Wall)"
Cohesion: 0.33
Nodes (6): 10-port powered hub (~50 W AC, data + 5 V), Laptop (upstream USB data), MODE A — WALL (judged demo), 6× SK473 (USB-A — 5 V only), 3× Vantec (data + their 5 V), 🔌 Wall outlet

### Community 18 - "Power Mode B (Cordless)"
Cohesion: 0.33
Nodes (6): Anker 737 battery (24 000 mAh · 140 W), Laptop (Mode B data source), 6× SK473 USB-A (5 V), 3× Vantec (straight to laptop USB), MODE B — CORDLESS (walk-around), PD-powered USB-A hub

### Community 20 - "Open Questions & To-Do"
Cohesion: 0.67
Nodes (3): Open: ALSA Enumeration Reorder Across Reboots, Open: KHD Driver Diameter Un-measured (spk_dia placeholder), Open Questions + To-Do

## Knowledge Gaps
- **127 isolated node(s):** `PLAY mode (real-time tuning by feel)`, `tactus_enclosure_plate.stl (175x202 one-print bed)`, `tactus_power_cradle (vented sled)`, `actuator_cup.stl (Ø52 driver cup)`, `actuator_button.stl (dust-cap contact button)` (+122 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `The AI Core: Audio + Vision to (String, Fret, Finger, Quality)` connect `AI Engine & Coaching Core` to `ML Training & Buzz Data Pipeline`, `Multimodal Vision & Buzz Modeling`, `Assembly Stages & Design Decisions`, `Haptic Perception Research & Prior Art`?**
  _High betweenness centrality (0.345) - this node is a cross-community bridge._
- **Why does `Data, Representation, Training & Eigenvector-to-Advice Semantics` connect `ML Training & Buzz Data Pipeline` to `AI Engine & Coaching Core`?**
  _High betweenness centrality (0.289) - this node is a cross-community bridge._
- **Why does `CAD README — printed enclosure + coupling pucks` connect `Actuator Puck Coupling CAD` to `ML Training & Buzz Data Pipeline`, `Enclosure CAD (box + lid)`?**
  _High betweenness centrality (0.288) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `KHD 3 Ohm / 5 W driver (the actuator, de-housed from SK473)` (e.g. with `Dropped heavy bass shakers (weight/physics)` and `Job 1 — gut an SK473 into a bare amp brick`) actually correct?**
  _`KHD 3 Ohm / 5 W driver (the actuator, de-housed from SK473)` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `# NOTE: the .scad's rim_lip subtraction is a geometric no-op (it lies inside`, `Cylinder of length h centered at (cx,cy,cz), running along axis x/y/z.`, `Vertical-edge rounded rectangular prism, min-corner at (0,0,z0).` to the rest of the system?**
  _152 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AI Engine & Coaching Core` be split into smaller, more focused modules?**
  _Cohesion score 0.05319148936170213 - nodes in this community are weakly interconnected._
- **Should `Project Overview & Hardware BOM` be split into smaller, more focused modules?**
  _Cohesion score 0.08710801393728224 - nodes in this community are weakly interconnected._