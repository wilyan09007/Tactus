# Software architecture

Runs on the **laptop** (the Raspberry Pi is owned but cut from the critical path). Split across two processes: the **browser** owns vision + UI (camera, MediaPipe, ArUco, AR, viz) and **Python** owns audio + the model + the **haptic renderer** that drives the **12 channels**. They sync over **one localhost WebSocket** (vision features flow browser→Python, timestamped). Works on **any song** — no per-song hardcoding. See [`../truth.md`](../truth.md) §2 for the locked split.

## Pipeline

```
INPUTS:  mic audio (Python)     webcam (browser: fretboard + hand)

- AUDIO branch  [PYTHON] --------------------------------------
 1. F0 / transcription       [ML]  -> notes + onsets + durations   (pYIN/YIN live; CREPE offline)
 2. Dynamics                 [det] -> RMS -> volume
 3. Technique/quality model  [ML]  -> {clean|buzz|muted|dead|ghost}  <- pressure-adequacy from timbre
       features: harmonic-to-noise ratio, inharmonicity, spectral centroid/flux,
                 attack/decay envelope, broadband-buzz energy   (librosa)

- VISION branch  [BROWSER] -----------------------------------
 4. Hand pose                [ML]  -> MediaPipe 21 landmarks/hand -> fingertips + which finger
 5. Fretboard geometry       [det] -> OpenCV + ArUco marker -> homography
       -> map fingertips to (string, fret) + placement vs fret-wire (coarse)
       (vision features stream browser -> Python over the localhost WebSocket, timestamped)

- FUSION [rules + light ML]  [PYTHON] ------------------------
 6. time-align audio onsets <-> vision frames
 7. reconcile: vision resolves WHICH string/fret made a pitch; audio confirms it sounded + quality
       buzz + finger-too-far-from-wire -> placement fault
       dead/weak + finger-present      -> pressure fault
       clean + matches target          -> correct
 8. confidence = cross-modal agreement

- REFERENCE DIFF ----------------------------------------------
 9. target (MIDI/tab, or basic-pitch on the recording) vs played
       discrepancies on: note | fret/finger choice | timing | dynamics | technique

- OUTPUTS -----------------------------------------------------
10. haptic renderer [PYTHON] -> 12 channels (target + correctness pulses)
11. LLM coach (Anthropic) [phrase-level] -> natural-language feedback
12. on-screen fretboard map [BROWSER] (precise fret/finger/placement/pressure/volume)
13. Redis [optional] -> per-user mistake history, adaptive difficulty, vector-search past mistakes
```

### Two timescales
- **Per note (real-time, deterministic):** instant haptic flag + screen highlight. The LLM is too slow per-note.
- **Per phrase (LLM coach):** structured error log → prioritized, encouraging advice ("ring finger muted the G — press just behind fret 3").

### "Any song"
The engine is song-agnostic. Two modes:
- **Free play:** transcribe whatever is played → render + map. No reference.
- **Coach vs a song:** auto-generate the target (fetch MIDI/tab, or run basic-pitch on the original recording), then diff. New song = no code change. Accuracy tracks musical complexity; vision cross-checks audio on chords.

## Haptic renderer

```
note event (string, fret, velocity, t)
   -> encoder (docs/07): (string -> chest ch) + (fret -> forearm zone + intensity)
   -> synth: ~160 Hz burst, 30-80 ms, amp = velocity (tune on-body; was 200-250 Hz pre-teardown)
   -> ALSA: write buffer to (card, channel) per config/channel_map.json
```

### ALSA / multichannel
- **2 Vantec cards used** (V1 = ch 1–8, V2 = ch 9–12; V3 spare), each an 8-channel device. `aplay -l` lists them.
- The renderer holds a logical-channel → (card, channel) map ([config/channel_map.json](../config/channel_map.json)).
- Slight clock drift between cards is fine for haptics (not phase-critical for pulses).
- ⚠️ The CM6206 cards enumerate with the same name and reorder across reboots — **bind by `/dev/snd/by-id`, not `hw:0/1/2`**, and verify in-card channel order with `speaker-test -c8` on each Vantec. See [`../truth.md`](../truth.md) §3.2.

## Suggested layout
```
software/ai/        transcription, vision, fusion, coach
software/haptic/    encoder, synth, alsa_out
config/             channel_map.json, encoding.json
```

## Stack
`basic-pitch`, `mediapipe`, `opencv-python` (+ ArUco), `librosa`/`aubio`/`crepe`, `sounddevice`/`pyalsaaudio`, `numpy`/`scipy`, Anthropic SDK, optional `redis`.

## Compute placement (be honest in the demo)
Everything runs on the laptop: vision + UI in the browser, audio + model + 12-ch haptic out in Python, over one localhost WebSocket. The Raspberry Pi is owned but cut from the critical path. Pre-process songs offline for the polished run; light real-time pitch (YIN/pYIN live, CREPE offline) for the live-guitar wow. State which is which.
