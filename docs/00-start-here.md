# Start here (teammates)

> **Canonical source of truth:** [`../truth.md`](../truth.md). If any doc disagrees with it, truth.md wins and the other doc is stale.

## What tactus is (30 seconds)
A Deaf-accessible guitar-coaching wearable with two modes: **LEARN** (coach a target song) and **PLAY** (real-time "flat / sharp / in tune / on beat" on a real guitar). Every note becomes a vibration at a specific body spot (string = back spot, fret = torso spot). A laptop watches (webcam) + listens (mic), uses AI to figure out exactly what you played — notes, fingers, how clean — compares to any song, and gives feedback + a live fretboard map. For Deaf/HoH players and for learning by touch.

**The one idea everything hangs on:** skin can't feel pitch, but it's great at *location* and *timing*. So we encode pitch as **where + when**, not as vibration frequency. See [07-haptic-encoding.md](07-haptic-encoding.md).

## Pick your lane

| You're doing | Read, in order |
|---|---|
| **Building the hardware** | [01-bom](01-bill-of-materials.md) → [02-architecture](02-system-architecture.md) → [04-soldering](04-soldering-guide.md) → [05-wiring-map](05-wiring-map.md) → [03-power](03-power.md) → [06-safety](06-safety.md) → [09-assembly-checklist](09-assembly-checklist.md) |
| **Writing the software** | [08-software](08-software-architecture.md) → [../software/README.md](../software/README.md) → [../config/channel_map.json](../config/channel_map.json) |
| **Pitching / AI / judges** | [11-ai-and-pitch](11-ai-and-pitch.md) → [07-encoding](07-haptic-encoding.md) → [12-perception-references](12-perception-references.md) |
| **Why did we choose X?** | [10-design-decisions](10-design-decisions.md) |
| **What's still open** | [13-open-questions](13-open-questions.md) |

Diagrams for everything live in [../diagrams/](../diagrams/) (system flow, power modes, wiring map, body map).

## Build-tonight TL;DR
- **12 channels** (6 SK473 × 2) = **6 strings (back) + 6 fret-zones (torso)**. The 2 reserved upper-back zones are dropped.
- **Actuators = the SK473's own KHD 3 Ω / 5 W drivers, de-housed** (not the 40 mm LEO speakers — those are spares).
- **Power for the judged run: Mode A (wall).** A chord renders as a sequential strum sweep, so ≤2 channels fire at once → real peak ~2–3 W; the hub powers all amps with huge margin → cannot brown out.
- **Cordless (Mode B): split amps 3/2/2 across Anker ports, the 3-amp bus on USB-C.** And a cut USB-C feed needs **5.1 kΩ CC resistors** or it stays dead.
- **Every solder joint: pre-load heat-shrink, tone-test before gluing, strain-relief both ends.** 18 AWG weight tears unrelieved joints.
- **Brain = laptop** (M4 Pro 16"). The Raspberry Pi 5 is owned but **cut from the critical path**.

## Status
- Hardware purchased (MicroCenter, 2026-06-19). Build pending.
- Still to get: 2× USB-C cables, 3D-printed vented enclosure, mounting (compression top + VHB + velcro + gloves), a webcam. See [13-open-questions.md](13-open-questions.md).
