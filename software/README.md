# software

Two halves, both on the laptop. See [../docs/08-software-architecture.md](../docs/08-software-architecture.md).

```
software/
  ai/       transcription (basic-pitch), vision (mediapipe + ArUco), fusion, LLM coach
  haptic/   engine (per-speaker control API + real-time voice mixer), rig (routing/device
            discovery/waveform synth), pair_test (66-pair sweep); speaker_check/resonance_check
            (bring-up). "Which speaker for a played note" is the ML's job, NOT here. See ../docs/28.
```

## Bring-up order
1. `aplay -l` → note the 3 Vantec card indices; fill `card_alsa` in [../config/channel_map.json](../config/channel_map.json).
2. `speaker-test -c8 -Dhw:CARD,0` on each Vantec → confirm all 8 channels reach the 4 jacks.
3. `haptic/` smoke test: play a known (string, fret) → correct body site buzzes.
4. `ai/` smoke test: webcam + mic → transcription + fretboard overlay.
5. Wire the coach loop on one phrase.

## Install
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Real-time note: `basic-pitch` is near-real-time per phrase, not per-note. For live coaching use a known target song (compare against its reference) — see the "any song" section in the architecture doc.

## What to build first (MVP slices)
Build in this order for the fastest working demo — full reasoning + track alignment in [../docs/11-ai-and-pitch.md](../docs/11-ai-and-pitch.md):
1. mono note detection (audio) + MediaPipe overlay + correct/incorrect
2. haptic render of the note on the body + rule-based "wrong note / too quiet"
3. LLM phrase-level coaching
4. then: chords/polyphony, audio↔vision cross-check, buzz→"press harder", placement color

Honest-scoping rules (don't over-claim): placement is **vs the fret-wire (coarse)**, pressure is **adequacy from timbre** (no sensor), polyphony degrades on dense playing. See [../docs/13-open-questions.md](../docs/13-open-questions.md) for known limitations.
