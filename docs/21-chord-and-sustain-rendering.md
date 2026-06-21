# 21 — Chord, sustain & percussive rendering (extends docs/07)

Status: **DRAFT — for `/plan-eng-review`**. Extends the single-note encoder in `docs/07-haptic-encoding.md` to chords, sustain, strums, and percussive hits on the as-built rig (12 ch = 6 back strings + 6 torso fret-zones, SK473 KHD drivers).

> **Governing principle:** we render the **actual per-string onset + amplitude envelope** the engine detects (mic + vision), **never a chord *symbol*.** Whatever the voicing, strum, fingerpick, or arpeggio, we reproduce what physically happened — so "the many ways to play a chord" are handled for free, and it stays an honest signal→skin transform.

---

## 1. Single note (recap of `docs/07`)
A note = `(string → back actuator) + (fret → torso zone + intensity)`, fired as a 200–250 Hz burst whose amplitude **tracks the note's real ADSR envelope** (sharp attack pulse, then a buzz that fades exactly as the string fades). Length and sustain come for free from following the envelope.

## 2. Chords — the onset encodes HOW it was struck
The simultaneity question is resolved at the **attack**, because that's where the performance information lives:
- **Strum** → a **spread sequential sweep** across the string-column. Direction + speed carry the strum: **down-strum = low-E→high-E**, **up-strum = reverse**; faster sweep = harder/faster strum. A wrong note lands as a wrong spot *in the sweep* (this is what makes "feel that you mis-fretted" work).
- **Simultaneous fingerpick / block chord** → a tight **"bloom"**: all sounding strings fire within a ~15 ms window (near-together). It's a brief transient, so it's power-fine, and it reads as "plucked at once" vs the spread of a strum.
- **Arpeggio** → already sequential in time; just render each note's onset+envelope as it occurs. No special case.

## 3. Sustain = envelope-modulated looped shimmer (LOCKED)
After the onset, a held/ringing chord is rendered as a **re-triggered shimmer**, not a static hold:
- **Cycle through the ringing strings** at a fast stagger so each string-speaker is re-triggered before it perceptually fades. The cycle continues **for the chord's duration** and **stops when the energy decays below threshold.**
- **The whole shimmer's amplitude follows the chord's real decay envelope** → you *feel it ring out and die* exactly as the strings do. Duration AND dynamics are faithful.

**Why re-trigger instead of hold (the citable rigor, `docs/01`):**
1. **Adaptation:** sustained fixed-site/fixed-frequency vibration desensitizes (~10 dB fade) — a *held* chord literally disappears from perception. Skin feels **change**, not drones.
2. **Masking:** simultaneous neighbors mask each other — a held 6-string chord blurs into one numb blob. Re-triggering keeps voices temporally separated.
3. **Apparent motion / Tactile Brush** (Israr & Poupyrev): discrete taps in sequence read as continuous flowing "ringing" — exactly the percept we want for a sustain, from a sparse array.

## 4. The other axis + foundation
- **Fret-zones carry the chord SHAPE** (which frets) on the torso, **pulsed in sync** with the shimmer (not held — so they don't add steady channels or mask).
- **Root emphasis:** the lowest/root string gets slightly stronger / slightly more frequent re-trigger so the chord has a felt "foundation."

## 5. Percussive & muted hits
- **Percussive chunk / palm-muted strum / string slap** = a **quick simultaneous pop across all motors** — a single sharp transient. Brief, distinct, power-fine.
- **Dead/muted strings within a chord** simply don't fire their string-speaker (they didn't sound) → the gap is felt, which is itself information ("you muted the G").

## 6. Power (reconciles `docs/15 §12`)
The looped shimmer is sequential **in time**, so even during a *sustained* chord only ~**2–3 string-speakers are live at any instant** → the ≤2-3-simultaneous power win **survives sustains.** Only a static hold (rejected) would light 6+ at once. Transient blooms/percussive pops are ~15 ms, so the brief 6-channel spike is negligible. Net: power stays trivial; you can drive each active channel hard for punch.

## 7. Tunable parameters (→ `docs/18` experiments)
- **Sweep stagger** (strum): ~30–50 ms between strings.
- **Bloom window** (pluck): ~10–20 ms.
- **Shimmer re-trigger interval**: tune so the sustain feels like *ringing*, not *repeated strumming* (start ~100–250 ms per string; faster = smoother/more continuous, slower = more discrete). This is the key feel knob — find it on-body.
- **Envelope→amplitude curve**, **root-emphasis gain**, **percussive-pop intensity**.

## 8. What the renderer needs as input
Per active string, a stream of `(onset time, pitch→string/fret, amplitude envelope, offset/decay)` — produced by the engine (mic transcription + vision, `docs/08`/`20`) for live play, or read from the reference (tab/MIDI) for the target. The renderer turns that into: onset-type (sweep vs bloom) → looped shimmer (envelope-modulated) → fret-zone shape pulses → stop at decay threshold.

## 9. Honest scope
- A literal all-at-once pluck's "togetherness" is conveyed by a **tight bloom**, not a held simultaneous chord (the right tradeoff — a literal hold blurs and fades).
- The **continuous apparent-motion "ring"** (energy-model phantoms, `docs/01`) is a **stretch** on top of this — smoother/"wow," but many more params to tune; land the looped shimmer first.
- Per-string envelope fidelity for dense/fast polyphony degrades with transcription accuracy (`docs/20` §8); vision cross-checks.

---

## GSTACK REVIEW REPORT

_`/plan-eng-review` — PAUSED mid-review (user stepped away). Sections 1–4 complete and decided; required-output synthesis (failure modes, tasks, parallelization) delivered in-chat; TODOS pass + outside-voice not yet run._

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | ISSUES_OPEN (PAUSED) | 8 arch + 7 design-correctness; 3 critical silent-failure gates |

**Decisions locked (3 forks resolved by user):**
- **Scope** → spine-first sequencing (bring-up + coupling + mono-envelope → sweep/bloom → looped-shimmer → ring stretch). Chord rendering is reference/feel-the-target path; live free-play stays monophonic.
- **Onset model** → two-path renderer (live: rhythm-pop on onset-detect → located buzz on the 25–40 ms settle; reference: pre-scheduled typed onset on the beat).
- **Power** → √N simultaneous-onset amplitude clamp (reuses energy-model law docs/01:38) + Mode A + one metered Stage-4 pop check.

**Top open items (P1):** A3 concurrency-budget scheduler, A5 verify CM6206 alsa_ch order (notes-in-wrong-place until done), B1 adaptation overclaim, B2 SOA vs re-trigger conflation, B3 driver ringdown/mush (add to doc-18), B6 fret-amp envelope decoupling, B7 config-is-source-of-truth drift.

**3 CRITICAL silent-failure gates** (silent in bench, appear on stage): [Stage 0] channel-order map, [Stage 4] 12-ch pop current draw, [Stage 7] callback starvation under AR+MediaPipe load.

**UNRESOLVED (paused, not defaulted):** B-series design-correctness fixes (B1–B7) presented but not individually ratified; TODOS.md pass not run; outside-voice (codex/claude) not run; review readiness dashboard not rendered.

**VERDICT:** NOT CLEARED — eng review paused with issues open. Resume to ratify B1–B7, run TODOS + outside-voice, and render the dashboard. Verification plan + failure modes + NOT-in-scope + implementation tasks (T1–T10) are delivered.
</content>
