# truth.md — Tactus single source of truth

> **This document is canonical.** If any other doc disagrees with this one, **this wins** and the other doc is stale — fix it. This is the human source of truth for *direction* and *hardware*; the machine source of truth for channel routing is [`config/channel_map.json`](config/channel_map.json). Last reconciled to the **Jun 20 live teardown** (the as-built rig), the MicroCenter receipt `195-PO-419168` (2026-06-19, Santa Clara), and the locked AIML design (`docs/17`, `docs/20`, `docs/23`).

---

## 0. What is true vs what is stale (read first)
Several early docs describe **plans we no longer build**. The product narrowed and the hardware changed after a live teardown. Current truth:

| Topic | ✅ CURRENT (truth) | ❌ SUPERSEDED (the abandoned approach) |
|---|---|---|
| Product scope | **LEARN + PLAY guitar coaching only** | Experience / Express / Body-Synth (the cut 3-pillar vision) |
| Brain | **Laptop** (M4 Pro 16", 48 GB) | Raspberry Pi 5 / ESP32 as the engine |
| Amps | **6× SK473 PAM8403** (stereo, gutted) | MAX98357A I2S / Teensy / TPA3116 |
| Actuators | **12× SK473 KHD 3 Ω / 5 W drivers**, de-housed | Dayton DAEX/TT25 exciters; the LEO 40 mm speakers (now spares) |
| Channel count | **12** (6 boards × stereo) | 14, or 16 |
| DAC | **2× Vantec NBA-200U used** (V3 spare) | "3× Vantec" framing — 3 owned, 2 used |

The legacy docs that carried this old info have been **removed from this repo** (recoverable from the original repo's git history). This file is canonical; every surviving doc (§9) is reconciled to it.

---

## 1. Direction & goal
**Tactus is a Deaf-accessible guitar-coaching wearable.** From a **webcam + a mic**, recover *what you played, where, with which finger, and how cleanly you fretted it* by fusing vision and audio through a learned physical model, turn it into **one precise correction you can feel on your body and see on your hand**, and render music to the skin as a direct, measurable signal→vibration transform (no AI-invented "score").

- **Tagline:** "Rocksmith for Deaf players."
- **Two modes:** **LEARN** (coach a target song: feel the correct version + your-version-vs-correct, see the physical fix) and **PLAY** (real-time "flat / sharp / in tune / on beat" on a real guitar).
- **The AI lives only where the problem is unsolved** (reading an occluded fretting hand + a cause-blind buzz). Everything downstream is a deterministic, pointable transform. That split *is* the rigor.
- **Honesty discipline:** a *new sensory channel* for music, not "Deaf people can hear." Pressure is reported **ordinal** (never Newtons); placement is fingertip-to-wire distance (coarse), not cm.
- **Event:** UC Berkeley AI Hackathon, **June 20–22, 2026.** Team of 4. Guitar donated ($0).

---

## 2. Locked software architecture
```
BROWSER (one tab)                          PYTHON process (laptop)
 ├ camera capture                          ├ mic capture
 ├ MediaPipe Hands (21 landmarks/hand)     ├ F0 (pYIN/YIN live; CREPE offline)
 ├ ArUco + OpenCV homography (fretboard)   ├ FUSION model (position + buzz inverse + Bayesian)
 ├ AR play view + 2D correction view       └ 12-ch haptic output (ALSA → Vantecs)
 └ live 3D cluster / response-surface viz
        │   vision features (timestamped) ─► localhost WebSocket ─►  ▲
        ◄── drive[] / frame / error-event / reference (~60 Hz) ──────┘
```
- **Browser owns vision + UI; Python owns audio + the model + haptic out.** They sync over **one localhost WebSocket**. Vision features flow browser→Python **timestamped on a monotonic clock**; a start-of-session clap cross-checks A/V sync.
- **Two latency tiers (non-negotiable):** immediate haptic = **audio-only, ~40 ms**; vision-fused correction = **slower (~100 ms+)**, to screen / secondary cue. A vision-fused per-note haptic at 10 ms is physically impossible (camera+MediaPipe+WS ≈ 50–80 ms).
- **Real-time hygiene:** keep heavy compute off the 12-ch audio-output callback (own process / buffered) so it never underruns; MediaPipe/ArUco in a Web Worker; AR and the 3D cluster viz **time-share** (never both rendering live).
- Schema source of truth: `docs/13 §3` (Python→browser). The **browser→Python vision-feature schema is not yet locked — lock it Saturday AM.**

---

## 3. Hardware — the as-built rig (exact)
```
LAPTOP ──USB data──► 2× VANTEC (USB→8 analog ch each; V3 spare)
                        │  3.5 mm STEREO jack per box (tip=L, ring=R, sleeve=GND) = 2 ch
                        ▼
                     6× SK473 box → bare PAM8403 stereo board (2 ch each)
                        │  one isolated 2-wire pair per channel (BTL: + and − both driven)
                        ▼
                     12× KHD 3 Ω / 5 W drivers (de-housed), on the body via coupling pucks

POWER (5 V, fully separate from audio): wall 10-port hub (Mode A) OR Anker 737 bus (Mode B)
        → feeds ONLY the amps' 5 V leads. Laptop + Vantecs run off the laptop's own USB.
ONE shared ground (laptop ↔ Vantec ↔ 3.5 mm sleeve ↔ amp). Isolation = "+5 V never touches a signal line."
```

### 3.1 Compute / brain
- **Primary:** MacBook **M4 Pro, 16", 48 GB RAM** (runs browser + Python + 12-ch audio). Dev machine: M3 MBP 14".
- **Raspberry Pi 5 (8 GB):** owned, **CUT from the critical path** — the laptop does everything.

### 3.2 Audio-out chain & channel map (12 channels)
- **2× Vantec NBA-200U** (CM6206) used: **V1 = all 4 jacks (ch 1–8), V2 = 2 jacks (ch 9–12)**; **V3 = spare** (swap-in for any jack that won't enumerate on Linux).
- **6× SK473 boxes**, each gutted to **one PAM8403 stereo board** = 2 channels.
- **12 channels = 6 chest (strings) + 6 forearm (fret-zones):**

| Ch | Vantec | Jack | Box | Side | Body site |
|----|--------|------|-----|------|-----------|
| 1–6 | V1 | front/rear/center | B1–B3 | L/R | chest column: high-E (top) → low-E (bottom) |
| 7–8 | V1 | side | B4 | L/R | forearm fret-zones z1–z2 (near nut) |
| 9–12 | V2 | front/rear | B5–B6 | L/R | forearm fret-zones z3–z6 (→ elbow) |

- 12 musical frets map onto the 6 forearm zones via **zone + intensity**. Sites kept **≥ 4–5 cm apart** (forearm 2-pt ~35–40 mm). The 2 reserved upper-back zones (old ch 13–14) are **dropped**.
- ⚠️ **ALSA enumeration trap:** the 3 CM6206 cards enumerate with the same name and reorder across reboots. Bind by `/dev/snd/by-id` (not `hw:0/1/2`), and verify the in-card channel order (`speaker-test`) — CM6206 often presents FL,FR,FC,LFE,RL,RR,SL,SR (center/LFE before rears). Fill `card_alsa`/`alsa_ch` empirically at bring-up. **(docs/15 §2.)**

### 3.3 Drivers & body coupling
- **Actuators = the SK473's own KHD 3 Ω / 5 W drivers, de-housed** (mass-y, "subwoofer-ish" → likely punchier than the 40 mm). Factory-matched to the PAM8403, so 3 Ω is safe at our felt level. The bought **40 mm LEO speakers are now spares.**
- **Coupling (make-or-break):** de-house the driver → **contact button on the dust-cap + rigid backer on the magnet + firm strap.** A bare cone radiates into air (heard, not felt); the button couples cone excursion into tissue. Foam-isolate each node; spacing ≥ 4–5 cm.
- **Puck (parametric, `cad/actuator_puck.scad`):** contact button **Ø 14 mm**, dome **4.5 mm** proud, glue base 1.6 mm; cup wall 2.4 mm, back 2.4 mm; wire notch **6 × 5 mm** (fits 18 AWG zip-cord). **`spk_dia` is set to 40 mm as a placeholder — MEASURE the KHD driver and re-render before final print.**
- ⚠️ **Driver diameter is NOT yet measured.** CAD placeholders: puck `spk_dia=40`, chest-plate/node-mount `drv_dia=58, drv_depth=24, drv_mag_dia=30`. **Measure the de-housed KHD driver; set `spk_dia`/`drv_dia` in the .scad files; re-render the STLs.** This is the top open dimension.

### 3.4 Amps (PAM8403, filterless Class-D, BTL)
- Each channel's two output terminals (`+`/`−`) are **both driven** — neither is ground. **Hard rules:** never tie any `−` to GND; never join two channels' `−`; **every actuator gets its own isolated 2-wire pair.** Keep `+→+` phase consistent across the array.
- **Volume pots: bypassed — set level per channel in software** (`docs/18` Exp. 4). You can't exceed the chip's fixed gain; punch comes from coupling + drive frequency, not overdrive (clipping cooks the coil).

### 3.5 Power
- **Mode A — wall (RECOMMENDED for the judged run):** IPSG **10-port powered hub** (label 10 A / 50 W; really a ~12 V brick stepped to ~8–10 A @ 5 V) → the 6 amps' 5 V leads. One USB upstream to the laptop also carries Vantec data. Verify the hub's real per-port limit; run at **felt level**.
- **Mode B — cordless:** **Anker 737 PowerCore 24K** (24000 mAh ≈ 87 Wh, 140 W; 2× USB-C + 1× USB-A) → soldered 5 V buses, **3/2/2 split**, the 3-amp bus on **USB-C** (never USB-A), **5.1 kΩ CC resistor** on cut USB-C feeds. Meter each bus (don't trust the Anker display).
- **The power win:** chords render as a **sequential strum sweep** → **never more than ~2 drivers active at once** → real peak ~**2–3 W**. The old "all-channels = 33 W" worst case **never happens.** Any 5 V source runs the rig with huge margin. Hard rules: audio and power never share a wire; 5 V↔5 V / GND↔GND only; never power amps from the laptop/Pi USB; vent the enclosure.

### 3.6 Wire
- **The only body wire on hand is AudioVox 18 AWG zip-cord** (100 ft, stranded). The BOM's "22 AWG solid-core" line is **stale — not physically present.** 18 AWG is fat for the PAM8403 pads: fan/trim strands, tin first, land fast, **hot-glue blob carries the load**. Strain-relieve **both ends** of every run. Best cheap upgrade if obtainable: a spool of 24–26 AWG silicone stranded. For the **Mode B power bus, 18 AWG is correct** (real current there).

### 3.7 CAD / enclosure (FlashForge Adventurer 5M, 220×220×220 mm bed)
Parametric OpenSCAD + pre-rendered STLs in `cad/`:
- **`tactus_box` (base+lid):** houses Vantecs + amp boards (+ Pi if used), vented, with strain-relief comb + zip-tie floor. **Base = 199.8 × 161.8 × 60.4 mm** (verified, fits the bed). Pi mount = 58 × 49 mm boss pattern.
- **`tactus_power_cradle`:** vented sled for the Anker 737 / 10-port hub.
- **`actuator_puck` (cup + button):** §3.3. ~12–16 needed.
- **`tactus_chest_plate` / `tactus_node_mount`:** body-side mounts. Use **VHB + zip-ties through a slot grid** (dimension-tolerant), not tight press-fit, because several sizes are estimated. **Recommended decision: amps stay in the box, splice-extend each driver's leads with 18 AWG to the vest** (drivers + wire on the body only).

### 3.8 Sensing
- **Camera** (USB webcam aimed down the neck, or laptop cam) → MediaPipe + ArUco. **Print the ArUco marker** for the headstock (`cad/README` has the slot).
- **Mic / contact mic** near the soundhole; keep F0 clean in a loud room.

---

## 4. Exact purchases — granular parts ledger (owned now)
> **Authoritative cost record = MicroCenter receipt `195-PO-419168`** (2026-06-19, Santa Clara). Prices below marked **≈** are known MSRP/approximations; ones marked **(confirm)** must be read off the receipt — they are NOT yet verified here, so do not quote them as exact. Guitar donated ($0).

### Electronics (the rig)
| Part | SKU | Qty | Spec / dimensions | Cost | Use |
|---|---|---|---|---|---|
| Vantec NBA-200U USB 7.1 audio adapter | 080325 | 3 | CM6206 chipset; USB dongle; **4× 3.5 mm stereo jacks = 8 ch each** | ≈$20 ea (confirm) | DAC. **2 used (V1 ch1-8, V2 ch9-12); V3 spare** |
| IPSG SK473 6 W powered speaker (gutted) | 593038 | 6 (+1 prior = 7) | each box = **1 PAM8403 stereo board (2 ch)** + 2× **KHD 3 Ω / 5 W** drivers | ≈$12 ea (confirm) | **6 amp boards + 12 KHD drivers = the actuators** |
| LEO 3 W 40 mm speaker (4-pack) | 714816 | 3 packs = 12 | 40 mm cone, ~4 Ω, 3 W | ≈$8/pack (confirm) | **SPARE** (was the fret-zone actuators) |
| LEO 2 W 40 mm speaker (6-pack) | 714808 | 2 packs = 12 | 40 mm cone, 2 W | ≈$8/pack (confirm) | **SPARE** (was the string actuators) |
| IPSG USB-A 3.0 10-port hub | 885194 | 1 | label 10 A/50 W; really ~12 V brick → ~8-10 A @ 5 V | ≈$35 (confirm) | **Mode A** amp 5 V power + Vantec data |
| Anker 737 PowerCore 24K | 549246 | 1 | 24000 mAh (~87 Wh), 140 W; 2× USB-C (15 W ea) + 1× USB-A (12 W) | ≈$150 | **Mode B** cordless amp power |
| Raspberry Pi 5 (8 GB) | 635649 | 1 | — | ≈$80 | owned; **CUT from critical path** (laptop is the brain) |
| Pi 27 W USB-C PSU | 620351 | 1 | 5.1 V / 5 A | ≈$12 | Pi power (only if Pi is used) |
| AudioVox 18 AWG speaker wire | 995381 | 1 | 18 AWG, 2-conductor zip-cord, 100 ft, stranded | ≈$20 | **the only body wire** + Mode B 5 V bus |
| Elegoo ESP-32 USB-C (3-pack) | 961193 | 1 (+2 prior = 5) | — | (confirm) | **NOT in the audio path** (sensor work only) |
| Adafruit 22 AWG solid-core (6-color) | 889089 | 2 | — | (confirm) | **stale — not physically present** (`docs/15 §3`) |
| Guitar | — | 1 | acoustic, from a friend | $0 | the instrument |

### Tools
| Part | SKU | Cost | Use |
|---|---|---|---|
| Soldering iron (Apex/Weller, LED halo) | 499129 | (confirm) | assembly |
| Newark rosin-core solder 60/40 | 461079 | (confirm) | assembly |
| Eclipse digital multimeter | 653428 | (confirm) | verify 5 V / polarity / no short |
| Eclipse 20-10 AWG wire stripper | 795880 | (confirm) | assembly |
| LEO heat-shrink assortment (thin-wall) | 797399 | (confirm) | insulate joints |

### Body-side fabricated parts (3D-printed, `cad/`) — dimensions
| Part | File | Dimensions | Use |
|---|---|---|---|
| Brain-pack box (base + lid) | `tactus_box.scad` | base **199.8 × 161.8 × 60.4 mm**; fits FlashForge 5M **220³ mm** bed; Pi boss = 58 × 49 mm | houses 2-3 Vantecs + 6 amp boards (+ Pi), vented, strain-relief comb |
| Power cradle | `tactus_power_cradle.scad` | vented sled for the Anker 737 / 10-port hub | holds the power source, ports/screen accessible |
| Actuator puck (cup + button) | `actuator_puck.scad` | contact button **Ø14 mm**, dome **4.5 mm** proud, base 1.6 mm; cup wall 2.4 mm, back 2.4 mm; wire notch 6 × 5 mm (18 AWG); **`spk_dia` = 40 mm PLACEHOLDER** | de-house driver → button on dust-cap couples cone into skin; ~12-16 needed |
| Chest plate / node mount | `tactus_chest_plate.scad`, `tactus_node_mount.scad` | torso radius 150 mm (est), string pitch 46 mm, plate 3 mm; **`drv_dia` = 58 mm PLACEHOLDER** | body-side driver mounts (VHB + zip-tie through a slot grid — dimension-tolerant) |
> ⚠️ **The KHD driver diameter is NOT yet measured.** `spk_dia`/`drv_dia` are placeholders (40 / 58 mm). Measure the de-housed driver → set the params → re-render the STLs before the final print.

**Still to get:** 2× sacrificial USB-C cables (Mode B buses), compression vest/shirt + VHB foam tape + velcro + zip ties + thin gloves, USB webcam (or laptop cam), optional 24-26 AWG silicone stranded wire (solders to the tiny PAM8403 pads far better than 18 AWG).

---

## 5. Haptic encoding (the deterministic renderer)
- **Single note:** `(string → chest channel) + (fret → forearm zone + intensity)`, fired as a burst whose amplitude **tracks the note's real ADSR envelope**.
- **Starting pulse (tune on-body, `docs/18`):** **160 Hz, 50 ms, 3 intensity levels** (per `config/channel_map.json`; heavy 3 Ω drivers may prefer ~80–160 Hz). Earlier docs' 200–250 Hz is a pre-teardown estimate.
- **Chords:** strum = spread **sequential sweep** across the string column (down = low-E→high-E); block/pluck = tight **bloom** (~10–20 ms). **Sustain = re-triggered shimmer** (envelope-modulated, ~2–3 strings live at once) — never a static hold. Fret-zones pulse the chord shape in sync. (`docs/21`.)
- **Tuning knobs (Saturday, `docs/18`):** drive frequency (sweep 60–250 Hz), pulse shape (punch), coupling pressure, per-channel software gain (the volume control).

---

## 6. The AI/ML (perception model)
Full spec: `docs/17` (rigor), `docs/20` (training design + decisions D1–D8), `docs/20-eng-review.md` (review), `docs/23` (data + cluster→advice semantics). In brief:
- **Stage 1 — WHERE (vision-led):** MediaPipe + ArUco homography → fretboard-relative pose → `(string, fret, finger, d=fingertip-to-wire)`; deterministic when visible, a **trained occlusion model** when the hand hides the contact point. **Prompt = the label; audio only verifies** (audio cannot give fret).
- **Stage 2 — HOW CLEAN (audio-led, uses `d`):** buzz `B` + `d` + **pluck-proxy** → recover **pressure as a 2-class ordinal (too-light / good)**; "too hard" is a separate **pitch-cents** fault, not a buzz class. Feedback: buzz+`d` large → "slide toward the fret"; buzz+`d` small → "press harder."
- **Fusion:** Bayesian posterior over position × fault with a **theory prior** (intended chord from the tab); divergence between played and intended = the detected mistake.
- **Rigor centerpiece:** the **separability study** (PCA→LDA → Fisher / silhouette / pairwise d′ / confusion, **leave-one-player-out**, audio-only vs vision-only vs fused) proves fusion is necessary. **Cluster→advice = hybrid** (supervised backbone + validated discovery; a discovered error sub-family ships only after passing the held-out promotion gate).
- **Data:** prompted + audio-verified, interleaved, pluck-controlled; breadth pass (6×3 frets, clean) + depth pass (2×2 cells × fault conditions); ~100 clean/class, ~1,000–1,500 takes, ~2–4 hrs. Engineered named features are the core; the contrastive embedding + Redis retrieval is cut-first (Redis runs on the engineered vectors regardless).

---

## 7. Sponsors / tracks targeted
Anthropic (Claude vision = the coaching brain: frame + target + fault → the plain-language fix), **Annapurna/AWS Trainium** (train the contrastive embedding — the legitimate accelerator job), Redis (mistake-retrieval / "your similar past mistakes"), CV track (MediaPipe + ArUco + occluded-placement), Accessibility / Ddoski's Lab (the whole product), Most-Technical (the inverse-problem fusion + separability proof). (`docs/19`.)

---

## 8. Top open items (measure / lock before they bite)
1. **Driver diameter** — un-measured; CAD uses placeholders. Measure → set `spk_dia`/`drv_dia` → re-render STLs. (§3.3)
2. **ALSA bring-up** — bind Vantecs by-id, verify in-card channel order, fill `channel_map.json`. (§3.2)
3. **Browser→Python vision-feature schema + A/V sync** — lock Saturday AM. (§2)
4. **Actuator coupling** — prove ONE puck feels strong before building 12. (§3.3)
5. **On-body tuning** — drive freq / pulse / coupling / per-channel gain. (`docs/18`)

---

## 9. Doc map (all current; legacy removed)
**Canonical:** this file (`truth.md`) + [`config/channel_map.json`](config/channel_map.json) (machine-readable channel routing).

**Current docs — all reconciled to this file:** `docs/00-start-here`, `01-bill-of-materials`, `02-system-architecture`, `03-power`, `04-soldering-guide`, `05-wiring-map`, `06-safety`, `07-haptic-encoding`, `08-software-architecture`, `09-assembly-checklist`, `10-design-decisions`, `11-ai-and-pitch`, `12-perception-references`, `13-open-questions`, `13_LEARN_WEB_AND_VISUALIZATION` (the browser↔engine WebSocket schema, see §2), `15-build-refinements` (the as-built teardown log), `17-ai-rigor`, `18-tuning-and-calibration`, `19-sponsors-refined`, `20-aiml-training-design` (+ `20-eng-review`), `21-chord-and-sustain-rendering`, `22-interface-ar-and-correction`, `23-data-and-cluster-semantics`, `cad/README.md`, and the `REF_*` research briefs (cited background).

**Removed (legacy old-architecture — deleted from this repo, recoverable from the original repo's git history):** the uppercase `00_NORTH_STAR`–`14_HARDWARE_INTEGRATION` brief series, `16-learn-interface` (superseded by `22`), `SHOPPING_LIST.md`, `REF_actuator_brief`, `REF_digital_platform_brief`. They described the abandoned Pi/ESP32/MAX98357A/exciter, 16/14-channel, 3-pillar/score-era version.
