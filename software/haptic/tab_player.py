#!/usr/bin/env python3
r"""
tab_player.py -- Tactus tab -> haptic translator (the full encoding pipeline).

WHAT THIS DOES
==============
Turns a song into something you FEEL. Two inputs, one output:

  * INPUT 1 -- the TAB (authored JSON, see the EXAMPLE below): the sequence of
    events (a note or a chord). This says WHICH speakers fire. No durations.
  * INPUT 2 -- the MIC (live; the Saramonic clipped inside the guitar): a volume
    envelope + spectral-flux onsets. This says WHEN a strike happens, HOW LONG it
    rings, and HOW HARD it was hit. No pitch detection at all.

The two are fused by a deterministic state machine (the SEQUENCER): each mic
onset fires the tab's next event, the event is held (re-triggered "shimmer")
while the mic stays loud, and it stops on the next onset OR when the volume
falls -- whichever comes first. A chord is one event: it starts, holds, and stops
as a unit. WHICH speaker for a (string, fret) is the deterministic NoteEncoder
(encoder.py); playing them is the non-blocking HapticEngine (engine.py).

    tab.json --\
                +--> Sequencer (onset->fire, hold->shimmer, release->stop)
    mic -------/          |  (string,fret) per note
                          v
                    NoteEncoder  (string,fret) -> (back_ch, fret_ch)
                          v
                    HapticEngine.play()  -> 12 speakers (6 back strings + 6 torso frets)

=============================================================================
 COMPLETE TAB EXAMPLE  (this is the contract -- a merge agent should emit THIS)
=============================================================================
A song is one JSON object. `events` is the ordered sequence. Each event has a
`notes` array: ONE note = a single note; TWO+ notes = a chord (fired together as
a quick strum sweep). `string`: 1 = high-E .. 6 = low-E (standard). `fret`:
0 = open, 1..6 = fretted (the rig's scope). Durations are NOT in the tab -- the
mic supplies them.

{
  "format": "tactus-tab/v1",
  "title": "Demo - single notes + one chord",
  "artist": "",
  "tuning": ["E2", "A2", "D3", "G3", "B3", "E4"],
  "string_numbering": "1=high-E .. 6=low-E",
  "capo": 0,
  "bpm": 90,
  "chord_window_ms": 40,
  "events": [
    { "notes": [ { "string": 6, "fret": 0 } ] },
    { "notes": [ { "string": 6, "fret": 3 } ] },
    { "notes": [ { "string": 5, "fret": 2 } ] },
    { "notes": [ { "string": 5, "fret": 0 } ] },
    { "notes": [ { "string": 4, "fret": 2 } ] },
    { "notes": [ { "string": 5, "fret": 2 },
                 { "string": 4, "fret": 2 },
                 { "string": 3, "fret": 1 } ], "strum_ms": 30 }
  ]
}

  Per-event optional keys:  "strum_ms" (chord sweep spread; default 25),
  "arpeggio": true (advance one mic-onset PER note instead of treating the group
  as one event), "intensity" (1..3 fixed felt strength; otherwise mic-driven).

=============================================================================
 USAGE  (for the merge agent / integrator)
=============================================================================
  # Inspect a tab's speaker mapping -- NO audio device needed:
  python software/haptic/tab_player.py --tab example_song.json --print

  # Play along on the real rig, driven by the mic (the live demo):
  python software/haptic/tab_player.py --tab example_song.json --mic "LavMicro"

  # Demo the haptics WITHOUT a guitar (synthetic onsets on a timer):
  python software/haptic/tab_player.py --tab example_song.json --simulate

  # See input + output devices (find the mic and the Vantec/Aggregate):
  python software/haptic/tab_player.py --list-devices

As a library:
  from tab_player import load_tab, TabSequencer, MicListener
  from encoder import NoteEncoder
  from engine import HapticEngine
  song = load_tab("example_song.json")
  with HapticEngine(v1="Aggregate", v2="Aggregate") as eng:      # see Mac note
      seq = TabSequencer(song, NoteEncoder(), eng)
      with MicListener(seq.on_onset, seq.on_sustain, seq.on_release, device="LavMicro"):
          seq.wait_until_done()

=============================================================================
 MAC AUDIO OUTPUT  (the judged machine is the M4 Pro -- this matters)
=============================================================================
macOS CoreAudio routes a multichannel stream DISCRETELY -- each channel to its
own jack, no shared-mode down-mix (this is the Windows trap that folds ch3-8 onto
FRONT; macOS does not do that). To drive BOTH Vantecs (16 ch total, 12 used) as
one synchronized device:

  1. Audio MIDI Setup -> "+" -> "Create Aggregate Device".
  2. Tick BOTH "USB Sound Device" Vantecs. Add V1 FIRST (it becomes channels
     1-8), then V2 (channels 9-16). Order matters: the engine assumes V1 = the
     low 8 columns, V2 = the next.
  3. Set both to 48000 Hz. Pick one as Clock Source; tick "Drift Correction" on
     the OTHER one (keeps the two USB clocks from sliding apart).
  4. Run with  --aggregate "Aggregate"  (or it is auto-detected on macOS).

The engine then opens ONE 16-ch CoreAudio stream (mode="aggregate") and writes
logical ch 1-8 -> columns 0-7, ch 9-12 -> columns 8-11. Per-channel addressing
on CoreAudio can also be done with sounddevice's CoreAudioSettings(channel_map=
[...]) (output: list length = device out-channels, unused = -1); the engine uses
the equivalent full-width-frame write, which is discrete on CoreAudio.

Alternative (no Aggregate Device): pass --v1/--v2 with the two Vantec names ->
the engine opens two separate CoreAudio streams (mode="rig"). That works on Mac
too, but the two USB clocks are independent (slight drift); for our pulse haptics
that is fine, but the Aggregate Device is tighter and is the recommended path.

Sources: Apple "Combine audio devices" (support.apple.com/en-us/102171);
Rogue Amoeba drift-correction KB; python-sounddevice platform-specific settings;
truth.md S2/S3.2 + docs/28 (the engine's CoreAudio aggregate/rig modes).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rig import configure_logging, get_logger
from encoder import NoteEncoder

log = get_logger("tactus.haptic.tabplayer")

# Per-pulse burst spec (config/channel_map.json pulse + config/encoding.json;
# truth.md S5). Tunable on-body Saturday (docs/18).
DEFAULT_BURST = {"freq_hz": 160.0, "duration_ms": 50.0, "attack_ms": 4.0, "release_ms": 12.0}
DEFAULT_STRUM_MS = 25.0
TAB_FORMAT = "tactus-tab/v1"


# --------------------------------------------------------------------------- #
# 1) The tab data model + loader  (pure; no audio)
# --------------------------------------------------------------------------- #
class TabError(ValueError):
    """A tab JSON that doesn't satisfy the tactus-tab/v1 contract."""


@dataclass(frozen=True)
class Note:
    string: int          # 1 = high-E .. 6 = low-E
    fret: int            # 0 = open, 1..6 = fretted


@dataclass
class Event:
    notes: list[Note]
    strum_ms: float = DEFAULT_STRUM_MS
    arpeggio: bool = False
    intensity: int | None = None     # None -> mic-driven velocity

    @property
    def is_chord(self) -> bool:
        return len(self.notes) > 1


@dataclass
class Song:
    events: list[Event]
    title: str = ""
    artist: str = ""
    tuning: list[str] = field(default_factory=list)
    capo: int = 0
    bpm: float | None = None
    chord_window_ms: float = 40.0
    meta: dict = field(default_factory=dict)


def load_tab(src, *, strings=range(1, 7), frets=range(0, 7),
             on_out_of_scope: str = "error") -> Song:
    """Parse + validate a tactus-tab/v1 tab into a Song.

    `src` is a path (str/Path) or an already-parsed dict.
    `strings`/`frets` define the rig's playable range (string 1-6, fret 0-6).
    `on_out_of_scope`: "error" (raise), "skip" (drop the note; drop the event if
    it empties), or "clamp" (pull the fret into range).
    """
    if isinstance(src, (str, Path)):
        data = json.loads(Path(src).read_text(encoding="utf-8"))
    elif isinstance(src, dict):
        data = src
    else:
        raise TabError(f"load_tab: expected path or dict, got {type(src).__name__}")

    fmt = data.get("format")
    if fmt != TAB_FORMAT:
        raise TabError(f"unsupported tab format {fmt!r}; expected {TAB_FORMAT!r}")
    if "events" not in data or not isinstance(data["events"], list):
        raise TabError("tab has no 'events' array")

    smin, smax = min(strings), max(strings)
    fmin, fmax = min(frets), max(frets)

    def _resolve_note(raw, ev_i, n_i) -> Note | None:
        try:
            s, fr = int(raw["string"]), int(raw["fret"])
        except (KeyError, TypeError, ValueError) as e:
            raise TabError(f"event {ev_i} note {n_i}: needs int 'string' and 'fret' ({e})")
        if not (smin <= s <= smax):
            raise TabError(f"event {ev_i} note {n_i}: string {s} out of range {smin}-{smax} "
                           f"(1=high-E .. 6=low-E)")
        if not (fmin <= fr <= fmax):
            if on_out_of_scope == "error":
                raise TabError(f"event {ev_i} note {n_i}: fret {fr} out of scope {fmin}-{fmax} "
                               f"(rig renders frets 1-6 + open). Pass on_out_of_scope='skip'/'clamp' "
                               f"or pick a lower-position tab.")
            if on_out_of_scope == "skip":
                log.warning("event %d: dropping out-of-scope note string=%d fret=%d", ev_i, s, fr)
                return None
            if on_out_of_scope == "clamp":
                fr = max(fmin, min(fmax, fr))
            else:
                raise TabError(f"on_out_of_scope must be error/skip/clamp, got {on_out_of_scope!r}")
        return Note(s, fr)

    events: list[Event] = []
    for ev_i, raw_ev in enumerate(data["events"]):
        raw_notes = raw_ev.get("notes") if isinstance(raw_ev, dict) else raw_ev
        if not isinstance(raw_notes, list) or not raw_notes:
            raise TabError(f"event {ev_i}: 'notes' must be a non-empty array")
        notes = [n for n in (_resolve_note(rn, ev_i, n_i) for n_i, rn in enumerate(raw_notes))
                 if n is not None]
        if not notes:
            log.warning("event %d emptied by out-of-scope filtering; dropping the event", ev_i)
            continue
        intensity = raw_ev.get("intensity") if isinstance(raw_ev, dict) else None
        events.append(Event(
            notes=notes,
            strum_ms=float(raw_ev.get("strum_ms", DEFAULT_STRUM_MS)) if isinstance(raw_ev, dict) else DEFAULT_STRUM_MS,
            arpeggio=bool(raw_ev.get("arpeggio", False)) if isinstance(raw_ev, dict) else False,
            intensity=int(intensity) if intensity is not None else None,
        ))

    if not events:
        raise TabError("tab has no playable events after validation")

    song = Song(
        events=events,
        title=data.get("title", ""), artist=data.get("artist", ""),
        tuning=list(data.get("tuning", [])), capo=int(data.get("capo", 0)),
        bpm=data.get("bpm"), chord_window_ms=float(data.get("chord_window_ms", 40.0)),
        meta={k: v for k, v in data.items() if k != "events"},
    )
    log.info("loaded tab %r: %d events (%d chords), chord_window=%.0fms",
             song.title, len(song.events), sum(e.is_chord for e in song.events),
             song.chord_window_ms)
    return song


# --------------------------------------------------------------------------- #
# 2) Onset/envelope DSP  (pure; testable on synthetic signals)
# --------------------------------------------------------------------------- #
@dataclass
class GateConfig:
    samplerate: int = 48000
    block: int = 512                 # analysis block (~10.7 ms @ 48k)
    gate_on_dbfs: float = -45.0      # envelope above this = a sound is present
    gate_off_dbfs: float = -55.0     # below this (held) = released (hysteresis)
    flux_floor: float = 1e-4         # minimum flux to ever call an onset
    flux_sensitivity: float = 3.0    # onset if flux > floor + sensitivity * running_mean
    flux_ema: float = 0.05           # running-mean adaptation rate
    chord_window_ms: float = 40.0    # onsets within this collapse into ONE event
    min_release_ms: float = 60.0     # must stay below gate_off this long to release
    sustain_emit_ms: float = 20.0    # cadence of 'sustain' ticks while held


class OnsetGate:
    """Streaming onset + sustain + release detector. push() one audio block at a
    time; get back a list of ("onset"|"sustain"|"release", amplitude) events.

    onset  = spectral-flux jump above an adaptive threshold while loud, debounced
             so a strum's staggered string attacks collapse into one onset.
    sustain= periodic tick while the envelope stays above the gate (drives shimmer).
    release= envelope fell below the gate (with hysteresis + a min-hold debounce).
    """

    def __init__(self, cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig()
        self._win = np.hanning(self.cfg.block).astype(np.float32)
        self._prev_mag: np.ndarray | None = None
        self._ema_flux = 0.0
        self._last_onset_ms = -1e9
        self._sounding = False
        self._below_since: float | None = None
        self._last_sustain_ms = -1e9

    def _amp(self, dbfs: float) -> float:
        """Map envelope dBFS -> 0..1 velocity (gate_off -> 0, 0 dBFS -> 1)."""
        lo = self.cfg.gate_off_dbfs
        return float(max(0.0, min(1.0, (dbfs - lo) / (0.0 - lo))))

    def push(self, block: np.ndarray, t_ms: float) -> list[tuple[str, float]]:
        cfg = self.cfg
        x = np.asarray(block, dtype=np.float32).ravel()
        if x.size == 0:
            return []
        rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
        dbfs = 20.0 * math.log10(rms)

        # spectral flux (half-wave-rectified magnitude increase)
        n = min(len(x), cfg.block)
        mag = np.abs(np.fft.rfft(x[:n] * self._win[:n]))
        if self._prev_mag is not None and self._prev_mag.shape == mag.shape:
            flux = float(np.sum(np.maximum(0.0, mag - self._prev_mag)))
        else:
            flux = 0.0
        self._prev_mag = mag

        out: list[tuple[str, float]] = []
        loud = dbfs > cfg.gate_on_dbfs
        thresh = cfg.flux_floor + cfg.flux_sensitivity * self._ema_flux
        is_onset = loud and flux > thresh and (t_ms - self._last_onset_ms) > cfg.chord_window_ms

        if is_onset:
            self._last_onset_ms = t_ms
            self._sounding = True
            self._below_since = None
            self._last_sustain_ms = t_ms
            out.append(("onset", self._amp(dbfs)))
        elif self._sounding:
            if dbfs < cfg.gate_off_dbfs:
                if self._below_since is None:
                    self._below_since = t_ms
                elif (t_ms - self._below_since) >= cfg.min_release_ms:
                    self._sounding = False
                    self._below_since = None
                    out.append(("release", 0.0))
            else:
                self._below_since = None
                if (t_ms - self._last_sustain_ms) >= cfg.sustain_emit_ms:
                    self._last_sustain_ms = t_ms
                    out.append(("sustain", self._amp(dbfs)))

        # adapt the running-mean flux (slowly; not while a fresh onset spikes it)
        if not is_onset:
            self._ema_flux = cfg.flux_ema * flux + (1.0 - cfg.flux_ema) * self._ema_flux
        return out


def run_offline(gate: OnsetGate, signal: np.ndarray) -> list[tuple[str, float, float]]:
    """Drive an OnsetGate over a whole signal in blocks (for tests / batch).
    Returns (kind, t_ms, amp) using a sample-accurate clock."""
    sr, b = gate.cfg.samplerate, gate.cfg.block
    sig = np.asarray(signal, dtype=np.float32).ravel()
    out = []
    for i in range(0, len(sig) - b + 1, b):
        t_ms = (i / sr) * 1000.0
        for kind, amp in gate.push(sig[i:i + b], t_ms):
            out.append((kind, t_ms, amp))
    return out


# --------------------------------------------------------------------------- #
# 3) The live mic listener  (wraps sounddevice.InputStream)
# --------------------------------------------------------------------------- #
class MicListener:
    """Open the mic, run the OnsetGate, dispatch onset/sustain/release callbacks.

    Callbacks: on_onset(amp, t_ms), on_sustain(amp, t_ms), on_release(t_ms).
    Time is a SAMPLE clock (frames seen / sr), so it is monotonic and independent
    of wall-clock jitter. Mono capture (channel 0) -- that's all F0/buzz/onset want.
    """

    def __init__(self, on_onset, on_sustain, on_release, *, device=None,
                 samplerate: int = 48000, gate: OnsetGate | None = None):
        self.on_onset, self.on_sustain, self.on_release = on_onset, on_sustain, on_release
        self.device = device
        self.samplerate = samplerate
        self.gate = gate or OnsetGate(GateConfig(samplerate=samplerate))
        self._frames = 0
        self._stream = None

    def _callback(self, indata, frames, time_info, status):  # noqa: ANN001
        if status:
            log.warning("mic input status: %s", status)
        mono = indata[:, 0] if indata.ndim > 1 else indata
        t_ms = (self._frames / self.samplerate) * 1000.0
        self._frames += frames
        for kind, amp in self.gate.push(mono, t_ms):
            if kind == "onset":
                self.on_onset(amp, t_ms)
            elif kind == "sustain":
                self.on_sustain(amp, t_ms)
            elif kind == "release":
                self.on_release(t_ms)

    def start(self):
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=self.samplerate, device=self.device, channels=1,
            blocksize=self.gate.cfg.block, dtype="float32", callback=self._callback)
        self._stream.start()
        dev = sd.query_devices(self.device)["name"].strip() if self.device is not None else "default"
        log.info("MicListener started on %s @%dHz block=%d", dev, self.samplerate, self.gate.cfg.block)
        return self

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info("MicListener stopped")

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


# --------------------------------------------------------------------------- #
# 4) The sequencer  (the state machine: tab + onsets -> haptics)
# --------------------------------------------------------------------------- #
class TabSequencer:
    """Drives the tab through the haptics from mic events.

    State: `next_idx` (next event to fire) and `active` (the currently-sounding
    event, or None). on_onset fires the next event (after stopping the previous);
    on_sustain re-triggers a shimmer; on_release stops the active event. A chord
    is one event -- it starts/holds/stops as a group.
    """

    def __init__(self, song: Song, encoder: NoteEncoder, engine, *,
                 velocity: bool = True, velocity_gain: float = 1.0,
                 base_intensity: int = 2, shimmer: bool = True,
                 shimmer_ms: float = 100.0, burst: dict | None = None, on_event=None):
        self.song = song
        self.enc = encoder
        self.eng = engine
        self.velocity = velocity
        self.velocity_gain = velocity_gain
        self.base_intensity = base_intensity
        self.shimmer = shimmer
        self.shimmer_ms = shimmer_ms
        self.burst = dict(burst or DEFAULT_BURST)
        self.on_event = on_event           # optional hook(kind, idx, event) for UI/telemetry
        self.next_idx = 0
        self.active: dict | None = None
        self._lock = threading.RLock()
        self._done = threading.Event()

    # -- helpers --------------------------------------------------------- #
    def _amp(self, env_amp: float) -> float:
        return max(0.05, min(0.95, env_amp * self.velocity_gain))

    def _spec(self, env_amp: float, intensity_override: int | None) -> dict:
        spec = dict(self.burst)
        if intensity_override is not None:
            spec["intensity"] = intensity_override
        elif self.velocity:
            spec["amp"] = self._amp(env_amp)
        else:
            spec["intensity"] = self.base_intensity
        return spec

    def _fire(self, idx: int, env_amp: float, shimmer: bool = False) -> list[int]:
        ev = self.song.events[idx]
        spec = self._spec(env_amp, ev.intensity)
        notes = [(n.string, n.fret) for n in ev.notes]
        if len(notes) == 1:
            vids = list(self.enc.play_note(self.eng, notes[0][0], notes[0][1], **spec))
        else:
            vids = list(self.enc.play_chord(self.eng, notes, strum_ms=ev.strum_ms, **spec))
        if self.on_event:
            self.on_event("shimmer" if shimmer else "fire", idx, ev)
        return vids

    def _stop_active(self):
        if self.active:
            for v in self.active["vids"]:
                try:
                    self.eng.stop_voice(v)
                except Exception:  # noqa: BLE001
                    pass
            if self.on_event:
                self.on_event("stop", self.active["idx"], self.song.events[self.active["idx"]])
            self.active = None

    # -- the three mic events -------------------------------------------- #
    def on_onset(self, amp: float, t_ms: float = 0.0) -> bool:
        """A strike. Stop the previous event, fire the next. Returns False when the
        song is exhausted (no more events to fire)."""
        with self._lock:
            self._stop_active()
            if self.next_idx >= len(self.song.events):
                self._done.set()
                return False
            idx = self.next_idx
            vids = self._fire(idx, amp)
            self.active = {"idx": idx, "vids": vids, "last_shimmer_ms": t_ms}
            self.next_idx += 1
            if self.next_idx >= len(self.song.events):
                self._done.set()    # last event fired; release will end it
            return True

    def on_sustain(self, amp: float, t_ms: float = 0.0):
        """Held above threshold -> re-trigger the active event as shimmer."""
        with self._lock:
            if not self.active or not self.shimmer:
                return
            if (t_ms - self.active["last_shimmer_ms"]) >= self.shimmer_ms:
                self.active["vids"] = self._fire(self.active["idx"], amp, shimmer=True)
                self.active["last_shimmer_ms"] = t_ms

    def on_release(self, t_ms: float = 0.0):
        """Volume fell -> stop the active event."""
        with self._lock:
            self._stop_active()

    # -- lifecycle ------------------------------------------------------- #
    @property
    def finished(self) -> bool:
        return self.next_idx >= len(self.song.events) and self.active is None

    def wait_until_done(self, timeout: float | None = None) -> bool:
        """Block until the last event has been fired (then released). Returns True
        if done, False on timeout."""
        if not self._done.wait(timeout):
            return False
        # also wait for the final event to release
        end = time.monotonic() + (timeout or 30.0)
        while self.active is not None and time.monotonic() < end:
            time.sleep(0.02)
        return self.finished


# --------------------------------------------------------------------------- #
# 5) Device helpers (Mac-aware output resolution) + CLI
# --------------------------------------------------------------------------- #
def list_devices():
    import sounddevice as sd
    apis = sd.query_hostapis()
    print("idx  in/out  ch_in/ch_out  host-API            name")
    print("-" * 78)
    for i, d in enumerate(sd.query_devices()):
        io = ("I" if d["max_input_channels"] else " ") + ("O" if d["max_output_channels"] else " ")
        print(f"{i:>3}  {io:>5}  {d['max_input_channels']:>5}/{d['max_output_channels']:<5}  "
              f"{apis[d['hostapi']]['name']:<18}  {d['name'].strip()}")


def resolve_mic(spec, samplerate=48000):
    """Resolve the input device. Explicit spec wins; else hunt for the Saramonic /
    a USB mic; else the system default input."""
    import sounddevice as sd
    if spec is not None:
        try:
            return int(spec)
        except (ValueError, TypeError):
            pass
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and str(spec).lower() in d["name"].lower():
                return i
        raise TabError(f"no input device name contains {spec!r}")
    for hint in ("lavmic", "saramonic", "usb"):
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and hint in d["name"].lower():
                log.info("mic auto-picked idx %d (%s)", i, d["name"].strip())
                return i
    return None   # system default input


def resolve_output_kwargs(args) -> dict:
    """Translate CLI output flags into HapticEngine kwargs, Mac-aware:
    --aggregate -> v1=v2=that device (engine aggregate mode);
    --v1/--v2   -> two separate Vantec streams (rig mode);
    --device    -> a single forced device;
    else on macOS, auto-detect an Aggregate Device by name; else engine auto."""
    if args.aggregate:
        return {"v1": args.aggregate, "v2": args.aggregate}
    if args.v1 and args.v2:
        return {"v1": args.v1, "v2": args.v2}
    if args.device:
        return {"device": args.device}
    if sys.platform == "darwin":
        try:
            import sounddevice as sd
            for i, d in enumerate(sd.query_devices()):
                if d["max_output_channels"] >= 12 and "aggregate" in d["name"].lower():
                    log.info("macOS: auto-using Aggregate Device idx %d (%s) -> 12-ch discrete",
                             i, d["name"].strip())
                    return {"v1": i, "v2": i}
        except Exception as e:  # noqa: BLE001
            log.debug("aggregate auto-detect skipped: %s", e)
    return {}   # let the engine's ordered fallback choose


def _print_mapping(song: Song):
    enc = NoteEncoder()
    print(f"Tab: {song.title!r}  ({len(song.events)} events, "
          f"{sum(e.is_chord for e in song.events)} chords)\n")
    for i, ev in enumerate(song.events):
        kind = "CHORD" if ev.is_chord else "note "
        parts = []
        for n in ev.notes:
            sc, fc = enc.note_to_channels(n.string, n.fret)
            parts.append(f"(s{n.string},f{n.fret})->ch{sc}+{'open' if fc is None else f'ch{fc}'}")
        extra = f"  strum_ms={ev.strum_ms:.0f}" if ev.is_chord else ""
        print(f"  [{i:>3}] {kind} {'  '.join(parts)}{extra}")


def _simulate(seq: TabSequencer, hold_ms: float, gap_ms: float):
    """Drive the sequencer with synthetic onsets (no mic) so the haptics can be
    demoed/validated without a guitar."""
    log.info("SIMULATE: %d events, hold=%.0fms gap=%.0fms", len(seq.song.events), hold_ms, gap_ms)
    t = 0.0
    n = len(seq.song.events)
    for i in range(n):
        seq.on_onset(0.8, t)
        # a couple of shimmer ticks during the hold
        steps = max(1, int(hold_ms / max(1.0, seq.shimmer_ms)))
        for s in range(1, steps + 1):
            time.sleep(min(hold_ms, seq.shimmer_ms) / 1000.0)
            seq.on_sustain(0.7, t + s * seq.shimmer_ms)
        seq.on_release(t + hold_ms)
        t += hold_ms + gap_ms
        time.sleep(gap_ms / 1000.0)
    log.info("SIMULATE done")


def main(argv=None):
    p = argparse.ArgumentParser(description="Tactus tab -> haptic translator (full pipeline).")
    p.add_argument("--tab", help="path to a tactus-tab/v1 .json file")
    p.add_argument("--print", dest="do_print", action="store_true",
                   help="print the tab's (string,fret)->speaker mapping and exit (NO audio)")
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    p.add_argument("--simulate", action="store_true",
                   help="drive synthetic onsets instead of the mic (demo without a guitar)")
    p.add_argument("--mic", help="input device (index or name substring); default = auto")
    # output (Mac-aware)
    p.add_argument("--aggregate", help="CoreAudio Aggregate Device (index/name) -> 12ch discrete (Mac)")
    p.add_argument("--v1"); p.add_argument("--v2")
    p.add_argument("--device", help="force a single output device (bench/one-Vantec)")
    # tuning
    p.add_argument("--chord-window-ms", type=float, default=None)
    p.add_argument("--flux-sensitivity", type=float, default=None)
    p.add_argument("--shimmer-ms", type=float, default=100.0)
    p.add_argument("--no-shimmer", action="store_true")
    p.add_argument("--no-velocity", action="store_true", help="fixed intensity instead of mic-driven")
    p.add_argument("--intensity", type=int, default=2)
    p.add_argument("--freq", type=float, default=DEFAULT_BURST["freq_hz"])
    p.add_argument("--duration-ms", type=float, default=DEFAULT_BURST["duration_ms"])
    p.add_argument("--samplerate", type=int, default=48000)
    # simulate timing
    p.add_argument("--hold-ms", type=float, default=400.0)
    p.add_argument("--gap-ms", type=float, default=250.0)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    configure_logging(a.verbose)

    if a.list_devices:
        list_devices()
        return 0
    if not a.tab:
        p.error("--tab is required (or use --list-devices)")

    song = load_tab(a.tab)
    if a.chord_window_ms is not None:
        song.chord_window_ms = a.chord_window_ms

    if a.do_print:
        _print_mapping(song)
        return 0

    burst = dict(DEFAULT_BURST, freq_hz=a.freq, duration_ms=a.duration_ms)
    gate_cfg = GateConfig(samplerate=a.samplerate, chord_window_ms=song.chord_window_ms)
    if a.flux_sensitivity is not None:
        gate_cfg.flux_sensitivity = a.flux_sensitivity

    from engine import HapticEngine
    out_kwargs = resolve_output_kwargs(a)
    with HapticEngine(samplerate=a.samplerate, **out_kwargs) as eng:
        seq = TabSequencer(song, NoteEncoder(), eng, velocity=not a.no_velocity,
                           base_intensity=a.intensity, shimmer=not a.no_shimmer,
                           shimmer_ms=a.shimmer_ms, burst=burst)
        if a.simulate:
            _simulate(seq, a.hold_ms, a.gap_ms)
        else:
            mic = resolve_mic(a.mic, a.samplerate)
            with MicListener(seq.on_onset, seq.on_sustain, seq.on_release,
                             device=mic, samplerate=a.samplerate, gate=OnsetGate(gate_cfg)):
                log.info("playing tab %r -- play along; Ctrl-C to stop", song.title)
                try:
                    seq.wait_until_done(timeout=None)
                except KeyboardInterrupt:
                    log.info("stopped by user")
        eng.stop_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
