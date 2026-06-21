# System architecture

Three layers: **sense → understand → render**. The brain (laptop) does sense + understand; the hardware chain does render. Vision (camera + MediaPipe + ArUco + AR) runs in the **browser**; audio + the fusion model + 12-ch haptic out run in **Python**; they sync over one localhost WebSocket. See [`../truth.md`](../truth.md) §2 for the locked software split.

## The full chain, every connector

```
[guitar]   [webcam]   [mic]
    \          |        /
     \         |       /          (vision + audio in)
      v        v      v
 +-------------------------+
 |   LAPTOP  (AI brain)    |  vision: MediaPipe + ArUco homography (in the BROWSER)
 |   - transcription       |  audio:  F0 + timbre/technique + RMS (Python)
 |   - fusion + diff       |  fusion: cross-check audio vs vision
 |   - LLM coach + UI      |  out:    note/feedback events -> haptic renderer
 +-----------+-------------+
             | USB (data)
             v
 +-------------------------+
 | 2x Vantec NBA-200U 7.1  |  USB digital -> 8 analog channels each
 |  4 stereo jacks each    |  V1 = ch 1-8, V2 = ch 9-12; V3 spare
 +-----------+-------------+  CONNECTOR: 3.5mm jacks (round holes)
             | 3.5mm PLUG  (SK473's own captive plug -> Vantec jack)
             v
 +-------------------------+        power (5V):
 | 6x SK473 amp boards     | <----- Mode A: 10-port hub (wall)
 |  PAM8403, 2 ch each     | <----- Mode B: Anker 737 via solder bus
 |  = 12 channels          |  CONNECTOR in: 3.5mm. CONNECTOR out: solder pads
 +-----------+-------------+
             | 18 AWG SPEAKER WIRE (soldered to output pads)
             v
 +-------------------------+
 | 12x KHD 3 Ω/5 W drivers |  6 strings (back) + 6 fret-zones (torso)
 |  (de-housed)            |
 +-------------------------+
```

## Connector cheat-sheet (where confusion happens)

| Link | Connector | Bare wire? |
|---|---|---|
| Vantec → SK473 | 3.5 mm plug (SK473's captive plug into the Vantec jack) | **No** |
| SK473 → actuator | speaker wire soldered to the amp's **output pads** | **Yes** |
| amp ← power | USB 5 V (from hub or bus) | (power only) |
| laptop → Vantec | USB data | (no audio wire) |

The amp is the translator: **3.5 mm line-level in → amplified power out on bare wire.** Bare wire never touches the Vantec; the Vantec only ever sees 3.5 mm plugs.

The 3.5 mm cable shares a signal ground between laptop and amp, but it carries only line-level current and the amp's 5 V rail is isolated from the signal lines — so amp power cannot flow back into the laptop (see [06-safety.md](06-safety.md)). This is bounded isolation, not a total absence of any DC path.

## Why two devices per "DAC + power"

- **Vantec (audio):** turns USB into the many analog channels. 1 Vantec = 4 stereo jacks = 8 ch = 4 SK473. We use **2 Vantecs** (V1 = ch 1–8, V2 = ch 9–12); V3 is a spare.
- **Hub / bank (power):** only delivers 5 V to the amps. Carries no audio.
These are independent layers; a dongle is a fallback for the *Vantec* (audio), never the hub (power).

## Brain placement

- **Primary: laptop** (M4 Pro 16" 48 GB; M3 MBP 14" dev). It runs the browser (vision + UI) + Python (audio + model + 12-ch haptic out) and drives the Vantecs over USB.
- **Raspberry Pi 5:** owned but **cut from the critical path** — the laptop does everything. The ESP32 is **not** in the audio path either (sensor work only).

## Channel budget

12 amp channels (6 SK473). Mapped as **6 string (back) + 6 fret-zone (torso)**. Vantec capacity and driver count exceed this; amps are the limit. The 2 reserved upper-back zones (old ch 13–14) are dropped.
