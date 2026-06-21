# Perception references (why the encoding is shaped this way)

These are well-established vibrotactile findings — the grounding for every encoding choice. Use them in the pitch; they're the difference between "we guessed" and "we designed."

## Vibrotactile frequency — why we don't encode pitch as frequency
- Useful tactile vibration range ≈ **10–1000 Hz**, peak sensitivity ≈ **200–300 Hz** (Pacinian corpuscles). Our starting pulse is **160 Hz** (the heavy de-housed 3 Ω drivers may prefer ~80–160 Hz; tuned on-body, `18`) — near the sensitive band, traded off against what the actuators couple efficiently.
- **Frequency discrimination is poor:** vibrotactile Weber fraction ≈ **20–30%**. The skin resolves maybe **5–9 distinct "buzz pitches"** across its whole range — not the 100+ a musician hears.
- Consequence: a one-fret change (~6% frequency) is **unfeelable as a frequency**. → encode pitch as **location + timing** instead.

## Spatial acuity — why sites are spread, and resolution lives on the limbs
Two-point discrimination (minimum separation to feel two points as two):
- fingertip ≈ **2–4 mm**
- palm ≈ **10 mm**
- forearm ≈ **35–40 mm**
- torso / back ≈ **40–50 mm**
Consequences: keep actuators **≥ ~4–5 cm apart** (a tight row funnels into one blob); put the **higher-resolution info (frets) on the torso**, the coarser info (strings) on the back; the hands/fingers carry the most bits if you ever add them.

## Temporal acuity — the skin's strength
- Gap/successiveness detection ≈ **5 ms**. Skin times far better than it pitches.
- → use **sharp pulses (30–80 ms)** not drones; render strums as a timed **sweep** low→high (a wrong note feels like a wrong spot in the sweep).

## Funneling / sensory saltation — why we separate simultaneous hits
- Two near-simultaneous stimuli close together **merge into one perceived locus**. → fire simultaneous cues on **separated sites** (chest vs forearm vs back), never adjacent.

## Precedents (this approach works, with training)
- **Neosensory Buzz** (Eagleman) — encodes audio onto **4 wrist actuators**; users *learn* to discriminate words. Lesson: few well-placed actuators + a structured, learnable code beats a dumb pile.
- **Model Human Cochlea** — maps audio frequency bands spatially across the torso.
- **SoundShirt** — torso vibrotactile music for Deaf audiences.
- **Braille / the vOICe** — general proof that structured sensory-substitution codes are learnable.

## How each maps to our design
| Finding | Design choice |
|---|---|
| Can't feel pitch | encode pitch as location + timing ([07](07-haptic-encoding.md)) |
| Coarse spatial acuity on torso | 12 sites, ≥4–5 cm apart, frets on torso |
| Excellent temporal acuity | sharp pulses; strum = timed sweep |
| Funneling | separate simultaneous cues across body regions |
| Few-actuator precedents work | 12 channels is plenty; structure + learnability > raw count |

(These are standard textbook/literature values for vibrotactile perception; cite the named effects and systems above rather than memorizing page numbers.)
