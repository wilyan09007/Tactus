# AI engine + pitch (for judges and the software team)

## The honest framing (say this to judges)
tactus is **not** "a deterministic pipe with AI bolted on." The rendering (note → vibration) is deterministic DSP — and that's fine, because the *hard, unsolved* part is **understanding arbitrary guitar playing**: which notes/chords, which fingers, how clean. That's the AI.

```
audio + video -> [ AI perception front-end ] -> symbolic notes + technique -> [ deterministic renderer ] -> body
                  transcription, vision,                                       encoder + pulse synth
                  timbre->technique, fusion
```

**AI where the problem is unsolved; deterministic where it's solved.** That split *is* the rigor.

## The AI pieces (ranked by how non-gimmicky)
1. **Fusion perception model** — recovers `(string, fret, finger, placement d)` by fusing vision + audio through a learned physical model; audio can't see fingers, vision can't hear buzz, and **disagreement between played and intended is literally the detected mistake**. This is the core contribution.
2. **Vision** — MediaPipe hand pose (in the **browser**) + OpenCV/ArUco fretboard homography → which finger, which string/fret, placement vs the fret-wire (coarse).
3. **Live F0** (pYIN / YIN; CREPE offline) → pitch → candidate `(string, fret)` positions; audio only verifies position (audio cannot give fret).
4. **Pressure (the buzz inverse)** — buzz `B = f(pressure P, wire-distance d)`; vision pins `d`, audio measures `B`, invert the fitted surface → **pressure as a 2-class ordinal (too-light / good).** "Too hard" is a separate **pitch-cents** fault (sharp/choked), not a buzz class. This is the novel bit — it closes the "how hard did they press" gap from sound alone, without a sensor.
5. **LLM coach** (Anthropic, phrase-level) — structured error log → prioritized natural-language feedback ("ring finger muted the G — press just behind fret 3"). Per-note feedback stays rule-based (the LLM is too slow per note).

## Honest scoping (this is rigor, not weakness)
- **No "exact cm" finger position.** Pitch is set by the fret wire, not where you press behind it — audio has zero cm info; vision is only coarse. We report **placement quality vs the fret wire**, not centimeters.
- **No measured pressure.** We report pressure as a **2-class ordinal (too-light / good)** recovered by inverting the buzz surface, not Newtons. "Too hard" is a separate pitch-cents fault.
- **Polyphonic accuracy degrades on dense/fast playing.** Vision cross-checks audio on chords. State which parts are real-time vs pre-processed in the demo.

## Any song (generalizability)
No per-song hardcoding. Two modes:
- **Free play:** transcribe whatever is played → render + map. Works on anything.
- **Coach vs a song:** auto-generate the target (fetch MIDI/tab, or run basic-pitch on the original recording), then diff on note / fret-finger / timing / dynamics + technique. New song = no code change.

## Track / prize alignment
- **Accessibility / tech-for-good — flagship.** Sensory substitution letting Deaf/HoH feel and *learn* music.
- **Computer vision** — hand tracking + fretboard homography.
- **LLM / agents** — the coaching agent.
- **Hardware / wearables** — the power-bank-worn vest.
- **Redis** — the coach's memory: per-user mistake history, adaptive difficulty, vector-search "your similar past mistakes." (Not dropped by the vision pivot — it has a real home here.)

## Rigor moves (cheap, high-credibility)
- Cite the perceptual science behind every design choice ([12-perception-references.md](12-perception-references.md)).
- **Mini discrimination study** (n=3 teammates): % of notes/chords correctly identified by feel → a real accuracy number.
- **Ablation:** naive frequency-per-speaker vs the 2-axis spatial code → show identification accuracy jumps.
- **Information-theoretic framing:** bits/sec of musical info through the haptic channel.

## Build-first (MVP for max judge ROI)
1. Mono note detection (audio) + MediaPipe overlay + correct/incorrect.
2. Haptic render of the note on the body + rule-based "wrong note / too quiet."
3. LLM phrase-level coaching.
Then: chords/polyphony, audio↔vision cross-check, buzz→"press harder", placement color. Skip the impossible-precision claims.
