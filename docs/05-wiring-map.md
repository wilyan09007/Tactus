# Wiring map — channel by channel

> **⚠️ UPDATED Jun 20 → 12 channels.** Teardown showed each SK473 box = **1 stereo amp (2 ch) + 2 drivers**, so 6 boxes = **12 channels**, not 14. Each box's single 3.5 mm jack is **stereo = 2 independent channels** (L=driver A, R=driver B); the USB is shared 5 V power only. 6 stereo jacks → **2 Vantecs** (V3 spare). **The live source of truth is now [`config/channel_map.json`](../config/channel_map.json)** (12 entries, `box`=B1–B6); use the SK473's own **KHD 3 Ω drivers**, not the 40 mm (`docs/15 §11–§12`). The 2 reserved upper-back zones (old ch 13–14) are **dropped**.

12 channels = 2 Vantec cards → 6 SK473 boxes (stereo) → 12 SK473 KHD 3 Ω drivers. Label every cable to match the JSON.

## The map

| Ch | Vantec | Jack | Box | Amp side | Body site | Actuator |
|----|--------|------|-----|----------|-----------|----------|
| 1  | V1 | front  | B1 | L | string 1 (high E) — chest top    | KHD 3Ω |
| 2  | V1 | front  | B1 | R | string 2 (B) — chest             | KHD 3Ω |
| 3  | V1 | rear   | B2 | L | string 3 (G) — chest             | KHD 3Ω |
| 4  | V1 | rear   | B2 | R | string 4 (D) — chest             | KHD 3Ω |
| 5  | V1 | center | B3 | L | string 5 (A) — chest             | KHD 3Ω |
| 6  | V1 | center | B3 | R | string 6 (low E) — chest bottom  | KHD 3Ω |
| 7  | V1 | side   | B4 | L | fret-zone 1 — forearm (near nut) | KHD 3Ω |
| 8  | V1 | side   | B4 | R | fret-zone 2 — forearm            | KHD 3Ω |
| 9  | V2 | front  | B5 | L | fret-zone 3 — forearm            | KHD 3Ω |
| 10 | V2 | front  | B5 | R | fret-zone 4 — forearm            | KHD 3Ω |
| 11 | V2 | rear   | B6 | L | fret-zone 5 — forearm                | KHD 3Ω |
| 12 | V2 | rear   | B6 | R | fret-zone 6 — forearm (toward elbow) | KHD 3Ω |

- V3 (third Vantec) is **spare / fallback** — swap in for any V1/V2 jack that won't map on Linux.
- B1–B6 = the 6 SK473 amp boards.
- All 12 channels drive the SK473's own de-housed **KHD 3 Ω/5 W drivers** (strings on the chest column, fret-zones on the forearm). The 40 mm LEO speakers are spares.

## Why this layout
- **6 strings as a vertical chest column** mirrors the guitar (high E up, low E down) → intuitive.
- **Fret-zones z1–z6 run along the fretting forearm** (near nut → elbow). Sites stay ≥ 4–5 cm apart → within skin two-point resolution (forearm ~35–40 mm). A tight row on one spot would funnel into a blob; spreading prevents that. (The 2 reserved upper-back zones from the old 14-channel layout are dropped.)
- 12 musical frets map onto the 6 forearm zones via **zone + intensity** (see [07-haptic-encoding.md](07-haptic-encoding.md)).

## ALSA side
Each Vantec exposes its 8 channels as one card. The renderer routes a logical channel index → (card, channel) → jack → amp → actuator, per [config/channel_map.json](../config/channel_map.json). Keep this table and that file in sync.

## Labeling rule
Write the channel number on **both ends** of every speaker-wire run and on each amp. A mislabeled channel = a note felt in the wrong place = silent bug. Label as you solder, not after.
