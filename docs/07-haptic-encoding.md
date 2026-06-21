# Haptic encoding — note → actuator

## The premise (why this design exists)

Skin **cannot feel pitch**. Vibrotactile frequency discrimination is ~25% (Weber); the skin resolves maybe 5–9 distinct "buzz pitches" across its whole range, not the 100+ a musician hears. So a one-fret change (~6% frequency) is **unfeelable as a frequency**.

The fix: encode pitch as **location + timing**, which skin *is* excellent at (temporal resolution ~5 ms; spatial two-point: fingertip ~2–4 mm, torso ~35–40 mm, back ~40–50 mm). This is a sensory-substitution code, like Braille or the Neosensory Buzz — structured and learnable, not arbitrary.

## The 2-axis code

A note = **(string, fret)**. Render it as two simultaneous cues on two body axes:

```
STRING axis  -> which back actuator  (6 actuators, high E top -> low E bottom)
FRET axis    -> which torso zone + intensity  (6 torso zones x 2 levels = 12 frets)
```

A played note fires **one back actuator (string) + one fret-zone actuator (fret)** together as a short pulse.

### Worked example — string 6, fret 1 vs fret 2
- Both fire **back-actuator-6** (tells you: low-E string).
- Fret 1 → fret-zone 1 at intensity-low. Fret 2 → fret-zone 1 at intensity-high (or fret-zone 2).
- Two distinguishable cues on the fret axis → you feel the difference. The unfeelable frequency delta became a feelable location/intensity delta.

## Fret → 6 torso zones (12 frets onto 12 channels)

The **6 fret-zone actuators z1–z6 sit on the torso and carry the 12 frets** via **zone + intensity** (2 levels each).

| Frets | Zone (torso) | Intensity |
|---|---|---|
| 0 (open) | — | brief double-tap on the string actuator only |
| 1–2 | zone 1 | low / high |
| 3–4 | zone 2 | low / high |
| 5–6 | zone 3 | low / high |
| 7–8 | zone 4 | low / high |
| 9–10 | zone 5 | low / high |
| 11–12 | zone 6 | low / high |

Skin resolves ~4–8 intensity steps, so 2 levels per zone is safe.

## Pulse shape
- Starting point: drive each actuator with a **160 Hz burst, 50 ms, 3 intensity levels** (crisp tap, not a drone — taps are far more distinguishable). Tune on-body per `docs/18`; the heavy 3 Ω KHD drivers may prefer ~80–160 Hz. (The older 200–250 Hz figure was a pre-teardown estimate.)
- Amplitude = note velocity (how hard you picked).
- Onset aligned to the note onset.

## Chords / strums
- **Sequential (recommended):** fire each note's (string+fret) pair in fast succession low→high string, ~40 ms apart. Skin's timing acuity makes this feel like a *strum sweep* — and a wrong note feels like a wrong spot in the sweep. This is what makes "feel that you played it wrong" work.
- **Gestalt (optional):** common chord shapes get one memorized pattern (faster to recognize, less granular). The chord root rides the corresponding back string actuator (the 2 upper-back cue zones from the old layout are dropped).

## Design rules (or it turns to mush)
1. **Separate simultaneous activations spatially** (back vs torso). Two close actuators firing together *funnel* into one perceived blob.
2. **Sharp pulses, not drones.**
3. **Structured mapping only** — neighbors in music → neighbors on body. Arbitrary maps are unlearnable.
4. Keep the highest-resolution info (fret detail) on the higher-acuity sites (torso over the broader back).

## Relationship to the coach
The same encoder renders the **target** (what to play) and can flag the **error** (a buzz/wrong-spot pulse) from the AI diff. The screen carries the precise fretboard map; the body carries string + coarse fret + correctness. See [truth.md](../truth.md) and [22-interface-ar-and-correction.md](22-interface-ar-and-correction.md).
