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
2. **THE Windows collapse: streaming (2 simultaneous streams) can't use WDM-KS, so
   it folds.** The 8-ch USB Sound Device enumerates under MME (idx ~9/11, tagged
   `3-`/`4-`), DirectSound (~21/23), and WDM-KS (~49/51). Only **WDM-KS bypasses the
   OS mixer** (per-jack routing); **MME/DirectSound/WASAPI-shared down-mix an 8-ch
   stream onto the device's configured layout — if it's *Stereo*, ch3-8 COLLAPSE
   onto the FRONT jack.** But this CM6206 driver **cannot open two simultaneous
   WDM-KS streams** (the 2nd adapter's exclusive KS pin fails, `-9999`/`-9996`;
   both dongles also share one USB controller). The streaming engine needs one
   persistent stream **per Vantec at once** → KS is impossible → it falls to
   DirectSound → **fold**. This is the "ch3-8 play on the front box" bug.
3. **The discrete path is SEQUENTIAL single-stream WDM-KS** (what `resonance_check`
   / `speaker_check` do, and `pair_test --discrete`): open ONE KS stream at a time
   (`sd.play`, blocking), play, close. No 2-stream limit, no fold → every channel
   on its own jack. Trade-off: **cross-Vantec pairs can't be simultaneous** (two
   devices at once = two KS streams). `pair_test --discrete` plays **intra-Vantec
   pairs simultaneously** (both channels in one KS buffer) and **serializes
   cross-Vantec pairs**. True simultaneous cross-Vantec needs **macOS CoreAudio**
   (bypass + Aggregate Device) or **Windows devices set to 7.1** (then shared-mode
   stops folding and the 2-device DirectSound rig routes discretely).
4. **KS is fragile — never probe it, never attempt 2 at once.** A probe-open (even
   `check_output_settings` can instantiate the KS pin) or a failed 2nd-KS open
   leaves the pin in a bad state where *every* later KS open fails `-9996` — which
   also breaks `resonance_check`. Recovery = **unplug/replug the Vantec USB** (or
   reboot). So: device discovery is **non-destructive** (`check_output_settings`,
   and **enumeration-only** for the discrete KS path — `rig.bypass_adapters_noprobe`),
   the engine **never builds a 2-device KS rig**, and the discrete sweep opens each
   KS device exactly once.
5. **CM6206 channel reorder confirmed in code.** Logical→hardware uses
   `FL,FR,FC,LFE,RL,RR,SL,SR`, so e.g. logical ch3 (rear L) → hw index 4, ch5
   (center L) → hw index 2. Encoded in `rig.JACK_TO_HW`; matches truth.md's "ALSA
   enumeration trap." Confirm by ear with `speaker_check.py --sweep`.
6. **Two adapters, one host API.** Distinct *indices* under one API = two physical
   Vantecs; the `3-`/`4-` MME/DS tag confirms it (WDM-KS shows both as plain "USB
   Sound Device" — distinguish by index). Override auto-pick with `--v1/--v2`.
7. **Real-time hygiene.** The streaming engine synthesizes the burst in `play()`
   (off the audio thread); the callback only copies precomputed slices under a brief
   lock — no synthesis/alloc in the callback, so it can't underrun (truth.md §2).

### Two output paths (pick per platform)
| Path | How | Discrete? | Simultaneous cross-Vantec? | Use |
|---|---|---|---|---|
| **streaming** (`engine.py`, `pair_test --streaming`) | persistent OutputStream per Vantec + voice mixer | **macOS yes** (CoreAudio); **Windows NO** (falls to DirectSound → folds unless device=7.1) | yes | real-time render, the demo (on Mac) |
| **discrete** (`pair_test --discrete`, `resonance_check`) | one blocking WDM-KS stream at a time | **yes** (per-jack, Win+Linux) | no (intra-Vantec simultaneous; cross serialized) | Windows bring-up / verification |

`pair_test` auto-selects **discrete** when `shared_would_fold()` is true (Windows
device set to Stereo); `--streaming` / `--discrete` force either.

### Validation performed (real audio, no mocks)
- Engine smoke: single note, simultaneous cross-axis pair, 6-note staggered strum.
- `pair_test` streaming: **66 pairs (32 cross-Vantec) + singles** drive both Vantecs.
- **Routing assertion** (engine-side, pre-OS-mixer): single channels in isolation put
  **>95 % of energy in the routed hardware column** (ch3→hw4, ch6→hw3, ch11→hw4 PASS).
- ⚠️ That assertion taps `outdata` **before** the OS mixer, so it can't see the
  shared-mode fold — the discrete (WDM-KS) path is what guarantees per-jack output;
  verify on the rig by ear with `resonance_check` / `pair_test --discrete`.

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
- **`-9996` on any WDM-KS open (incl. `resonance_check`)** = the KS pin is in a bad
  state from a prior probe / failed 2nd-KS open → **unplug/replug the Vantec USB**
  (or reboot) to recover. The code now avoids the triggers (no KS probe, no 2-KS rig).
- **True simultaneous discrete 12-ch on Windows** needs the two devices set to
  **7.1 Surround** (then `--streaming` DirectSound stops folding) or **macOS**
  (CoreAudio Aggregate). The dev box without 7.1 is limited to the discrete
  *sequential* path (cross-Vantec serialized).

## 8. Run it
```bash
cd software/haptic
# DISCRETE per-jack (Windows bring-up; auto-selected when shared-mode would fold):
python pair_test.py                       # discrete WDM-KS sweep, all 12, per-jack
python pair_test.py --discrete --singles  # force discrete; warm up with singles
python pair_test.py --v1 49 --v2 51       # pin the two WDM-KS indices (low_note_all --list)
python resonance_check.py                 # one-at-a-time discrete resonance walk

# STREAMING (real-time render; discrete on macOS, folds on Windows-Stereo):
python pair_test.py --streaming           # engine sweep (true simultaneous cross-Vantec)
python engine.py --verbose --intensity 1  # smoke demo
python low_note_all.py                     # all 12 at once (guards against fold/concentration)
```
> If a run prints the **fold HEADS-UP / FOLD GUARD**, you're on shared-mode: either
> set both USB devices to 7.1, use `--discrete`, or run on the Mac.
