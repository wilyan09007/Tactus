# Design decisions (why, and what we rejected)

The repo says *what* to build. This says *why*, so nobody re-opens a settled question or repeats a dead end.

## Amplification — gut SK473 boards (PAM8403)
**Chosen:** harvest the amp board from cheap powered speakers (SK473), one per 2 channels.
**Why:** haptic actuators need only ~1 W/ch; small class-D amps are the right size, in stock, cheap. Powered speaker = a free amp in a box.
**Rejected:**
- **Car amp** — 50–100 W/ch is massive overkill, needs a 12 V supply (ATX), adds frying risk + weight, and the "find one tonight" hunt ate hours. Wrong tool for 1 W haptics.
- **Raspberry Pi DigiAMP+** — only 2 channels, and it rides the Pi's I2S bus (one stereo stream); stacking 3 gives 3 copies of the same 2 channels, not 6. Hardware-bus dead end.
- **MAX98357A** — out of stock; would have needed channel-select resistors. Abandoned, which is why **no resistor is in the audio path**.

## Audio source / channels — USB 7.1 (Vantec NBA-200U, CM6206)
**Chosen:** 3× USB 7.1 cards owned, **2 used** (V1 = ch 1–8, V2 = ch 9–12; V3 spare) = the 12 analog channels we need.
**Why:** one Vantec = 4 stereo jacks = 8 channels = feeds 4 SK473. Far fewer cables/devices than a pile of stereo dongles; fits the laptop's ports; CM6206 does real 8-ch on Linux.
**Rejected / fallback:** ~9 separate USB stereo **dongles** (Sabrent AU-MMSA) — works but cable sprawl; kept as a **per-channel fallback** if a Vantec jack won't map.

## Bass — dropped (was Logitech Z313)
**Chosen:** no dedicated bass node. The 12-driver array carries the whole concept.
**Why:** once the project became *note coaching*, "which note/string/fret" matters, not chest-thump bass. A subwoofer also needs its enclosure (can't wear it bare) and its own bigger amp. Out of scope.
**Rejected:**
- **Z313 sub** — bought, then dropped. Return it.
- **Harvesting a transducer from a Joy-Con / phone / controller** — those are **LRA/ERM** (single-frequency rumble buzzers), need a special haptic driver IC, and can't be driven from an audio amp. Wrong technology. Only **voice-coil exciters / shakers / woofers** are audio-drivable; the one MicroCenter sells (Dayton BST-2) wasn't needed once bass left scope.

## Encoding — location + timing, not frequency
**Chosen:** a note = (string → back actuator) + (fret → torso zone + intensity), as a short ~160 Hz pulse (tune on-body; 200–250 Hz was a pre-teardown estimate).
**Why:** skin can't discriminate pitch (vibrotactile Weber ~25%; ~5–9 distinguishable buzz-pitches total). A one-fret frequency change is unfeelable *as frequency*. But skin resolves *location* and *timing* well, so we render pitch there instead. See [12-perception-references.md](12-perception-references.md).
**Rejected:** mapping pitch → one speaker's vibration frequency (the obvious first idea) — physically unfeelable.

## Channel count — 12
**Chosen:** 12 = 6 strings (back) + 6 fret-zones (torso); 12 frets map onto the 6 zones via zone + intensity.
**Why:** we use 6 SK473 = 12 amp channels; that's the binding constraint. It's also within skin's spatial resolution on the back+torso, so 12 distinct sites is already near the useful ceiling. The 2 reserved upper-back zones (old ch 13–14) are **dropped**.

## Brain — laptop (Pi cut from critical path)
**Chosen:** the laptop (M4 Pro 16" 48 GB) runs the full stack — browser (vision + UI) + Python (audio + model + 12-ch haptic out) — and drives the Vantecs.
**Why:** a Pi 5 can't run MediaPipe + transcription + LLM + 12-ch audio in real time, and the architecture put MediaPipe in the browser anyway.
**Pi role:** owned but **cut from the critical path** — the laptop does everything. The ESP32 is **not** in the audio path either.

## Pressure — inferred from timbre, no sensor
**Chosen:** detect pressure *adequacy* from audio (buzz / dead-note / clean), not a force sensor.
**Why:** an FSR gives true Newtons but adds hardware; the timbre signal ("does it buzz / choke?") is what actually matters for coaching, and a learned model reads it. Framed honestly as **pressure adequacy**, not measured force.
**Rejected:** FSR-on-ESP32 — good for a v2 with real ground-truth, dropped to keep scope tight.

## Power — two modes
**Chosen:** Mode A wall (10A/50W hub, guaranteed) + Mode B cordless (Anker 737 + solder bus, split 3/2/2).
**Why:** the hub trivially powers all amps with margin for the reliable judged run; the bank gives the portability wow. See [03-power.md](03-power.md) for the per-bus math and the USB-C CC-resistor requirement.
