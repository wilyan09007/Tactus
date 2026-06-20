# 18 — Tuning & calibration: the on-body experiments (fill in Saturday)

> We're switching the body actuators to the **SK473's own 12× KHD 3 Ω / 5 W drivers** (de-housed) — they're mass-y, low-biased ("subwoofer") drivers, so they likely hit *harder* than the 40 mm, but their best drive parameters are entirely **surface/coupling-dependent** and must be found **on the body**. This doc is the experiment plan + a results table. It answers the three questions Aditya flagged: *optimal drive frequency, optimal pressure, how punchy.*

## The 4 knobs you're tuning
| Knob | What it changes | Sane start | Why it's not obvious |
|---|---|---|---|
| **Drive frequency** | which mechanoreceptors fire / felt strength | 150–200 Hz | Pacinian peak is ~250 Hz, but **heavy 3 Ω drivers often feel strongest LOWER (~80–160 Hz)** — more excursion, more force into tissue. Sweep it. |
| **Pulse shape (punchiness)** | how crisp/“tap”-like vs droney | 50 ms, sharp attack | Sharp attack + fast decay = punchy + more distinguishable; long flat = mushy. |
| **Coupling pressure** | felt strength + localization | snug strap + contact button | Firm preload drops the detection threshold a lot (docs/12); loose = heard, not felt. |
| **Per-channel gain (software)** | equal feel across all 12 sites | per-site, set by ear/feel | Body sensitivity varies by site; drivers vary unit-to-unit. **This is the volume control — in software, not the pots** (see docs/15 §12). |

## Experiment 1 — drive-frequency sweep (do this FIRST, per zone)
1. Mount one driver + puck on a chest site, firm strap.
2. Play 500 ms tones at **60 / 100 / 150 / 200 / 250 Hz**; rate **felt strength 0–5** and how **localized** it feels.
3. Pick the strongest+cleanest. Repeat on a **forearm** site (chest and forearm may prefer different freqs).

| Site | 60 | 100 | 150 | 200 | 250 | chosen Hz |
|---|---|---|---|---|---|---|
| chest |  |  |  |  |  |  |
| forearm |  |  |  |  |  |  |

## Experiment 2 — punchiness (pulse shape)
At the chosen freq, play pulses of **30 / 50 / 80 ms**, each with (a) flat envelope and (b) sharp-attack→fast-decay. Rate **punch** and **distinctness** (can you tell two quick taps apart?).

| ms | flat | sharp-decay | pick |
|---|---|---|---|
| 30 |  |  |  |
| 50 |  |  |  |
| 80 |  |  |  |

## Experiment 3 — coupling pressure
Loose vs snug vs tight strap; **with vs without the contact button** on the cone. Rate felt strength, localization, comfort. Pick the firmest that's still comfortable.

## Experiment 4 — per-channel gain calibration
Play the chosen pulse on **every** site at equal digital gain; rate relative strength; set a per-channel gain so all 12 feel **equal**. Store in `config/` (e.g. `encoding.json`). The 60-sec per-wearer staircase (docs/12) layers on top for each new body.

## Safety bounds (don't cross while experimenting)
- **Never drive to clipping.** A clipped Class-D output is near-DC into the voice coil → cooks the coil *and* feels worse (buzzy). Keep gain below visible/audible distortion.
- Driver surface **< ~40 °C**; 5 W is a thermal ceiling — you use ~1 W, so you're fine.
- Thanks to **≤ 2 channels ever active at once** (sequential strum, docs/15 §12) power is a non-issue — but per channel, stay at *felt* level, not max.

## Record the winners (these become the renderer defaults)
```
drive_freq_chest:   ___ Hz      drive_freq_forearm: ___ Hz
pulse:              ___ ms, envelope = ___
strap/coupling:     ___
per-channel gains:  [ ... 12 values ... ]
```
</content>
