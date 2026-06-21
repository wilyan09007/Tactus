# 28 — Haptic output engine & per-speaker controls

> **Status: built + validated on the real 2-Vantec rig (2026-06-21).** This doc is
> the spec, the research log, and the assumptions ledger for `software/haptic/`.
> It is reconciled to [`truth.md`](../truth.md) (§2, §3.2/3.4/3.5, §5, §8) and to
> [`config/channel_map.json`](../config/channel_map.json) + [`config/encoding.json`](../config/encoding.json).

## 1. What this is (and what it is NOT)

This is the **low-level haptic output engine** — the surface the ML/fusion calls
to fire any of the 12 speakers. It makes **no musical decisions**.

| Layer | Owner | This PR? |
|---|---|---|
| "WHICH speaker should fire for the played note/chord" | ML / vision+audio fusion (truth.md §6) | ❌ no |
| "Play speaker N **now**, with this spec, non-blocking, while others play" | **the engine (`engine.py`)** | ✅ yes |
| "Compose notes into strum sweeps / blooms / shimmer" (§5 *encoding*) | the encoder built **on top** of the engine | ❌ no (engine just makes it possible) |

The §5 chord/strum/sustain *behaviours* are produced later by the ML/encoder
making **multiple `play()` calls** with their own timing + params. The engine only
has to expose every basic knob and be non-blocking + concurrent.

## 2. Files

```
software/haptic/
  rig.py        shared: channel routing (channel_map.json -> vantec+hw), device
                discovery (Win WDM-KS / mac CoreAudio / Linux ALSA), waveform
                synthesis (tone/glide + raised-cosine envelope), logging.
  engine.py     HapticEngine: persistent OutputStream(s) + real-time voice mixer +
                the per-speaker play() API. The deliverable.
  pair_test.py  auto-paced sweep of all C(12,2)=66 speaker pairs (real audio).
  speaker_check.py / resonance_check.py   pre-existing standalone bring-up tools
                (unchanged — proven; see §7 deferred cleanup).
config/encoding.json   intensity->amplitude map, per-channel software gain, clip ceiling.
```

## 3. The control surface exposed to the ML — `HapticEngine.play()`

```python
with HapticEngine() as eng:           # auto-detects the rig; opens persistent streams
    vid = eng.play(ch, intensity=2, amp=None, freq_hz=160.0, duration_ms=50.0,
                   waveform="tone", glide_hz=(60,250), attack_ms=4.0,
                   release_ms=12.0, delay_ms=0.0)   # returns immediately
    eng.play_pair(1, 7, intensity=3)  # two speakers at once (even across V1/V2)
    eng.stop_voice(vid); eng.stop_all()
    state = eng.drive_state()          # {ch: current amplitude} — telemetry hook
    eng.set_channel_gain(ch, g)        # per-site calibration at runtime
```

| Param | Values | Default | Controls | Source |
|---|---|---|---|---|
| `ch` | 1–12 | required | which speaker (→ body site) | channel_map |
| `intensity` | {1,2,3} | 2 | felt-strength level → amplitude | channel_map, docs/07 |
| `amp` | 0–0.95 | None→use intensity | continuous override = note velocity | docs/07, docs/18 |
| `freq_hz` | ~60–250 | 160 | drive/carrier freq (heavy 3Ω best 80–160) | docs/18, docs/07 |
| `duration_ms` | ~10–500 | 50 | burst length | docs/18 |
| `waveform` | tone/glide | tone | steady sine vs chirp | resonance_check |
| `glide_hz` | (f0,f1) | (60,250) | sweep range (glide only) | resonance_check |
| `attack_ms` | ≥0 | 4 | punch (sharp attack ramp) | docs/18 Exp2 |
| `release_ms` | ≥0 | 12 | decay ramp | docs/18 Exp2 |
| `delay_ms` | ≥0 | 0 | scheduled start (stagger without blocking) | docs/07 |

**Engine-level (not per call):** `channel_gain[ch]` (per-site software volume,
`encoding.json`, docs/18 Exp4) and an always-on **clip ceiling 0.95** (a clipped
Class-D output is near-DC into the coil → cooks it + feels worse, docs/18 safety).
Final amplitude = `(amp or intensity_amp[intensity]) × channel_gain[ch]`, clamped.

**Non-blocking & concurrent (verified):** `play()` synthesizes the burst up front,
appends a "voice" to a shared list, and returns a voice id. One persistent
`OutputStream` **per Vantec** sums the voices routed to it in its callback. Any
number of speakers play at once, **including one on V1 + one on V2 simultaneously**
— impossible with sequential blocking playback, which is the whole reason for the
streaming-mixer design.

## 4. Output modes (auto-selected; all REAL audio, no simulation)

| Mode | When | Behaviour |
|---|---|---|
| `rig` | 2 openable ≥8-ch adapters | two streams, V1=ch1-8, V2=ch9-12 (the real rig) |
| `aggregate` | one device serves both (macOS Aggregate) | one 16-ch stream, V2 at hw+8 |
| `single` | explicit `device=` (e.g. one 8-ch Vantec) | ch1-8 on it |
| `bench` | **no Vantec present** | the default device; 12 logical ch round-robin onto its outputs — lets the full pipeline run + be heard on a laptop. Plug in the Vantecs → `rig` mode addresses all 12 jacks discretely. |

## 5. Research log (what I had to learn to make this real)

1. **PortAudio device indices are unstable across runs.** The same physical
   Vantec re-indexes between process launches; an auto-picked index from one run
   can be a phantom in the next (`Error querying device`). → Never trust the
   listing; resolve fresh each run.
2. **WDM-KS validates but won't open on this dev box.** On Windows the 8-ch USB
   Sound Device enumerates under MME (idx ~9/11, name-tagged `3-`/`4-`),
   DirectSound (~21/23), and WDM-KS (~49/51). `check_output_settings` *passes* for
   the WDM-KS endpoints, but `Pa_OpenStream` then fails with **paInvalidDevice
   (-9996)**. → Device validation must do a **real open test** (construct + close
   an `OutputStream`), not just `check_output_settings`. Auto-selection prefers a
   mixer-bypassing host API (WDM-KS) but **falls back across APIs**
   (→ DirectSound → MME) and finally to **bench**. On this machine the engine
   settled on **DirectSound idx 21/23** for the two real Vantecs.
3. **Two physical adapters, one host API.** Distinct *indices* under one API = two
   different physical Vantecs; the `3-`/`4-` MME/DS name tag confirms it (WDM-KS
   shows both as plain "USB Sound Device" — distinguish by index, per
   speaker_check's note). Auto-pick takes two distinct indices in the preferred
   API; override with `v1=/v2=`.
4. **CM6206 channel reorder confirmed in code.** Logical→hardware uses
   `FL,FR,FC,LFE,RL,RR,SL,SR`, so e.g. logical ch3 (rear L) → hw index 4, ch5
   (center L) → hw index 2. Encoded in `rig.JACK_TO_HW`; matches truth.md's "ALSA
   enumeration trap." Confirm by ear on the rig with `speaker_check.py --sweep`.
5. **Cross-Vantec simultaneity is best-effort (~a few ms), not sample-locked.**
   V1 and V2 are two USB DACs with **independent clocks**; PortAudio gives each its
   own callback thread, so two voices started "together" begin within ~one audio
   block. truth.md §3.8 already establishes independent device clocks **don't
   matter for our pulse haptics** (analysis→render is decoupled), so this is fine.
6. **Real-time hygiene.** The whole burst waveform is synthesized in `play()` (off
   the audio thread); the callback only copies precomputed slices and holds the
   lock briefly — no synthesis/allocation in the callback, so it can't underrun
   (truth.md §2).

### Validation performed (real audio, no mocks)
- Engine smoke: single note, a simultaneous cross-axis pair, and a 6-note
  staggered "strum" (40 ms apart) — all played + cleaned up.
- `pair_test.py`: full **66 pairs (32 cross-Vantec, 34 intra-Vantec) + 12 singles**
  on the two real Vantecs (DirectSound 21/23).
- **Routing correctness assertion:** played single channels in isolation and
  confirmed from the captured 8-ch output that **>95 % of energy lands in the
  routed hardware column** (ch3→V1 hw4, ch6→V1 hw3, ch11→V2 hw4 — all PASS).
- Real `--wav` capture: two 8-ch WAV files written from the actual mixed output.

## 6. Assumptions (bench starting points — revisit on-body Saturday)

These are **new settings this PR introduces** that are not pinned by truth.md /
channel_map; all are tunable and recorded in `config/encoding.json`:

- **intensity → amplitude:** `{1:0.30, 2:0.60, 3:0.90}`. channel_map declares
  *3 levels* but not the amplitudes; these are the levels. Tune in docs/18 Exp2/4.
- **per-channel gain:** all `1.0` until the docs/18 Exp4 on-body equalization fills
  real per-site values.
- **clip ceiling:** `0.95` (never reach digital full-scale → never clip the amp).
- **pulse shape default:** attack 4 ms / release 12 ms (sharp = "punchy"); flat
  feel = raise both. Default `freq 160 Hz`, `duration 50 ms` from channel_map.
- **`delay_ms` scheduling** is per-device-callback granular (aligned to ~one audio
  block, a few ms) — ample for strum staggers (~40 ms), not sample-accurate.
- **bench mode** maps 12 logical channels round-robin onto a stereo/8-ch default
  device; it is a dev convenience, **not** discrete body addressing.

## 7. Deferred / follow-ups
- **Consolidate `speaker_check.py` / `resonance_check.py` onto `rig.py`.** They
  currently keep their own copies of `JACK_TO_HW` / device discovery / `make_tone`.
  Left unchanged here because they are proven, interactive, and already tracked —
  refactoring risks breaking working bring-up tools right before the event. rig.py
  is the engine's foundation; merging is a low-risk post-hackathon cleanup.
- **Wire `drive_state()` to the localhost WebSocket** so the browser 3D viz lights
  up (truth.md §2 `drive[]`). The hook exists; the socket is a separate PR.
- **On-body tuning** (docs/18) overwrites `encoding.json` gains + the drive freq.
- **WDM-KS open failure** on the dev box: if discrete routing needs the
  mixer-bypass path, resolve the PortAudio/WDM-KS `-9996` (or use the rig's
  Linux/ALSA `by-id` path, truth.md §3.2); DirectSound/MME work for bring-up.

## 8. Run it
```bash
cd software/haptic
python engine.py --verbose --intensity 1          # real smoke demo
python pair_test.py                                # all 66 pairs, both Vantecs
python pair_test.py --singles --channels 1,2,9,10  # subset incl. cross-Vantec
python pair_test.py --intensity 3 --freq 120 --glide --wav out/pairs --verbose
python pair_test.py --v1 21 --v2 23                # pin adapters if auto-order wrong
```
