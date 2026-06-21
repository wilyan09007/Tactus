# Bill of materials

Reconciled to MicroCenter receipt ref `195-PO-419168`, 2026-06-19, Santa Clara.

## Owned now

### Audio source (DAC / channels)
| Part | SKU | Qty | Notes |
|---|---|---|---|
| Vantec NBA-200U USB 7.1 audio adapter | 080325 | 3 | CM6206 chipset. 4 stereo 3.5mm output jacks each = 8 ch. **2 used (V1+V2 → ch 1–12), V3 spare** |

### Amplifiers
| Part | SKU | Qty | Notes |
|---|---|---|---|
| IPSG SK473 6W powered speaker | 593038 | 6 (+1 prior = 7) | **6 used.** Gutted for its PAM8403 board + its 2 KHD 3 Ω/5 W drivers. 2 ch each = **12 channels** ← binding constraint |

### Actuators (the SK473's own KHD drivers)
| Part | SKU | Qty | Role |
|---|---|---|---|
| KHD 3 Ω/5 W driver (de-housed from the SK473, SKU 593038 above) | — | 6 boxes × 2 = **12** | **the actuators** — mass-y, punchy; factory-matched to the PAM8403, 3 Ω safe at felt level. Diameter TBD — measure the driver, set `spk_dia`, re-render. |
| LEO SALES 3W 40mm speaker (4-pack) | 714816 | 3 packs = 12 | **spare** (was fret-zone actuators) |
| LEO SALES 2W 40mm speaker (6-pack) | 714808 | 2 packs = 12 | **spare** (was string actuators) |

12 KHD drivers used (all 12 channels). The 24 LEO 40 mm speakers are now **spares**.

### Power
| Part | SKU | Qty | Notes |
|---|---|---|---|
| IPSG USB-A 3.0 10-port hub, **10A (50W) AC adapter** | 885194 | 1 | Wall-mode (Mode A) amp power. 50W ≫ real peak ~2–3 W |
| Anker 737 PowerCore 24K (24000mAh, 140W, 2× USB-C + 1× USB-A) | 549246 | 1 | Cordless-mode (Mode B) amp power |
| Raspberry Pi 5 8GB | 635649 | 1 | owned; **cut from the critical path** (the laptop is the brain) |
| Raspberry Pi 27W USB-C PSU | 620351 | 1 | Pi power (if used) |

### Wire + tools
| Part | SKU | Qty | Notes |
|---|---|---|---|
| AudioVox 18 AWG speaker wire, 100 ft | 995381 | 1 | **the only body wire on hand** + Mode B power bus |
| Adafruit 22 AWG solid-core hookup wire, 6 colors × 25 ft | 889089 | 2 | **stale — not physically present** |
| Apex/Weller soldering iron w/ LED halo | 499129 | 1 | |
| Newark rosin-core solder 60/40 | 461079 | 1 | |
| Eclipse compact digital multimeter | 653428 | 1 | |
| Eclipse 20–10 AWG wire stripper | 795880 | 1 | |
| LEO thin-wall heat-shrink assortment | 797399 | 1 | |
| IPSG Pro Micro starter kit | 055269 | 1 | |
| Elegoo ESP-32 USB-C (3-pack) | 961193 | 1 (+2 prior = **5** ESP32) | **not in the audio path** (sensor work only) |

### Not needed for the audio path
Resistors (1/4W 1kΩ SKU 796854 ×2, 2W 5.6kΩ SKU 796938). Keep for ESP32 sensor work. **No resistor in the audio chain.**

## Still to get

| Item | Why |
|---|---|
| 2× USB-C cables (sacrificial) | cut + strip to feed the two 5V solder buses from the Anker (cordless mode) |
| 3D-printed enclosure | houses Vantecs, amp boards (+ Pi if used), hub/bank — vented |
| Compression shirt/vest + VHB foam tape + velcro + zip ties + thin gloves | mount + couple actuators to the body |
| USB webcam (or use the laptop cam) | the vision branch needs a fretboard-facing camera |

## Channel accounting

- Vantec DAC channels available: **16** used (2 cards × 8); 12 routed, V3 spare
- SK473 amp channels: **12** (6 boards × 2) ← the limit
- Actuators driven: **12** KHD 3 Ω drivers = 6 strings (back) + 6 fret-zones (torso)
- Spare actuators: 24 LEO 40 mm speakers
