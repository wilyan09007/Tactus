#!/usr/bin/env python3
"""
engine.py -- HapticEngine: the real-time, non-blocking per-speaker control API.

This is the surface the ML/fusion calls to fire any of the 12 haptic speakers.
It makes NO musical decisions -- "which speaker for a played note" is the ML's
job (truth.md S6). The engine only does: "play speaker N with this spec, now,
without blocking, while other speakers may also be playing."

Why a streaming mixer (not blocking sd.play):
  A note's haptic = a back-string speaker + a torso fret-zone speaker firing
  TOGETHER; a strum = several speakers in fast succession; a chord-bloom =
  overlapping bursts. None of that is possible one-blocking-call-at-a-time, and
  a cross-Vantec pair (one speaker on V1, one on V2 = two separate USB devices)
  literally cannot be played simultaneously by sequential blocking calls. So the
  engine keeps ONE persistent sounddevice.OutputStream per Vantec (or one macOS
  Aggregate stream) and a shared list of "voices"; each stream's callback sums
  the voices routed to it. play() just appends a voice and returns immediately.

Real-time hygiene (truth.md S2):
  The whole burst waveform is synthesized up front in play() (off the audio
  thread). The callback only copies precomputed slices into the output frame --
  no allocation, no synthesis, minimal lock hold. Heavy compute never touches
  the audio callback, so it can't underrun.

Output modes (auto-selected, all REAL audio -- no mocks, no simulation):
  * "rig"       2 Vantecs found  -> two streams, V1=logical ch 1-8, V2=ch 9-12.
  * "aggregate" v1==v2 (macOS Aggregate Device) -> one 16-ch stream, V2 at hw+8.
  * "single"    one explicit device (e.g. a single 8-ch Vantec) -> ch 1-8 on it.
  * "bench"     no Vantec present -> the default (or chosen) device; the 12
                logical channels round-robin onto its real outputs. Lets the full
                pipeline run + be heard on a laptop; the instant the Vantecs are
                plugged in, "rig" mode addresses all 12 jacks discretely.

Per-call control surface (the values exposed to the ML) -- see play().
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sounddevice as sd

from rig import (
    CONFIG, ENCODING, RigError, _device_openable, configure_logging, get_logger,
    load_channel_plan, load_pulse_defaults, make_glide, make_tone, resolve_device,
    resolve_two_adapters,
)

log = get_logger("tactus.haptic.engine")

# Built-in defaults; overridden by config/encoding.json if present (docs/18 Exp4).
DEFAULT_INTENSITY_AMP = {1: 0.30, 2: 0.60, 3: 0.90}
DEFAULT_CLIP_CEILING = 0.95   # never clip the Class-D output -> cooks the coil (docs/18)


@dataclass
class Voice:
    """One scheduled/active burst on one logical channel."""
    vid: int
    ch: int
    stream_key: str          # which output stream carries it ("V1"/"V2"/"AGG"/"OUT")
    hw_index: int            # 0-based column within that stream's frame
    buffer: np.ndarray       # precomputed mono float32 (gain + clip already applied)
    pos: int = 0             # samples already emitted
    delay: int = 0           # samples still to wait before first sample
    peak: float = 0.0        # nominal peak amplitude (for drive_state)
    spec: dict = field(default_factory=dict)


class HapticEngine:
    """Persistent multichannel haptic output. Open once, play() many times.

    Usage:
        with HapticEngine() as eng:
            eng.play(1, intensity=2)              # fire speaker 1
            eng.play_pair(1, 7, intensity=3)      # two at once (cross-axis)
    """

    def __init__(self, config_path: Path = CONFIG, encoding_path: Path = ENCODING,
                 samplerate: int = 48000, v1=None, v2=None, device=None,
                 clip_ceiling: float | None = None, capture: bool = False):
        self.samplerate = int(samplerate)
        self.capture_enabled = bool(capture)
        self.plan = load_channel_plan(config_path)
        self.defaults = load_pulse_defaults(config_path)
        self._load_encoding(encoding_path, clip_ceiling)

        self._voices: list[Voice] = []
        self._lock = threading.RLock()
        self._next_vid = 1
        self._streams: dict[str, sd.OutputStream] = {}
        self._stream_channels: dict[str, int] = {}
        self._first_block: set[str] = set()
        self._underruns = 0
        self._capture: dict[str, list[np.ndarray]] = {}
        self._running = False

        # route: logical ch -> (stream_key, hw_index, stream_device_idx)
        self.route: dict[int, tuple[str, int]] = {}
        self.mode = "?"
        self._plan_streams(v1, v2, device)
        log.info("HapticEngine ready: mode=%s, sr=%d, streams=%s, intensity_amp=%s, clip=%.2f",
                 self.mode, self.samplerate,
                 {k: f"{self._stream_devices[k]}({self._stream_channels[k]}ch)" for k in self._streams_planned},
                 self.intensity_amp, self.clip_ceiling)
        self._log_routing()

    # ------------------------------------------------------------------ #
    # config
    # ------------------------------------------------------------------ #
    def _load_encoding(self, encoding_path: Path, clip_ceiling):
        self.intensity_amp = dict(DEFAULT_INTENSITY_AMP)
        self.channel_gain = {ch: 1.0 for ch in range(1, 13)}
        self.clip_ceiling = DEFAULT_CLIP_CEILING if clip_ceiling is None else float(clip_ceiling)
        try:
            data = json.loads(Path(encoding_path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            log.info("no encoding.json at %s; using built-in intensity/gain defaults", encoding_path)
            return
        except Exception as e:  # noqa: BLE001
            log.warning("could not parse %s (%s); using built-in defaults", encoding_path, e)
            return
        for k, v in data.get("intensity_amp", {}).items():
            self.intensity_amp[int(k)] = float(v)
        for k, v in data.get("channel_gain", {}).items():
            self.channel_gain[int(k)] = float(v)
        if clip_ceiling is None and "clip_ceiling" in data:
            self.clip_ceiling = float(data["clip_ceiling"])
        log.info("loaded encoding.json: intensity_amp=%s clip=%.2f gains=%s",
                 self.intensity_amp, self.clip_ceiling, self.channel_gain)

    # ------------------------------------------------------------------ #
    # output planning (which streams, which channel goes where)
    # ------------------------------------------------------------------ #
    def _plan_streams(self, v1, v2, device):
        self._stream_devices: dict[str, int] = {}
        self._streams_planned: list[str] = []
        self._device_forced = device is not None

        if device is not None:
            dev = resolve_device(device)
            self._single_device_plan(dev, label="OUT", mode="single")
            return

        # try the full 2-Vantec rig
        try:
            v1_idx, v2_idx = resolve_two_adapters(v1, v2, self.samplerate)
        except RigError as e:
            log.warning("rig (2-Vantec) mode unavailable: %s", e)
            self._bench_plan()
            return

        if v1_idx == v2_idx:
            # macOS Aggregate Device: one 16-ch handle, V2 channels at hw+8
            n = sd.query_devices(v1_idx)["max_output_channels"]
            self.mode = "aggregate"
            self._streams_planned = ["AGG"]
            self._stream_devices = {"AGG": v1_idx}
            self._stream_channels = {"AGG": n}
            for e in self.plan:
                hw = e["hw"]
                if hw is None:
                    continue
                base = 0 if e["vantec"] == "V1" else 8
                self.route[e["ch"]] = ("AGG", hw - 1 + base)
            log.info("aggregate mode: single device idx %d (%d ch), V2 offset +8", v1_idx, n)
            return

        # normal 2-device rig
        self.mode = "rig"
        self._streams_planned = ["V1", "V2"]
        self._stream_devices = {"V1": v1_idx, "V2": v2_idx}
        self._stream_channels = {
            "V1": sd.query_devices(v1_idx)["max_output_channels"],
            "V2": sd.query_devices(v2_idx)["max_output_channels"],
        }
        for e in self.plan:
            hw = e["hw"]
            if hw is None:
                continue
            self.route[e["ch"]] = (e["vantec"], hw - 1)
        log.info("rig mode: V1=idx %d (%d ch), V2=idx %d (%d ch)",
                 v1_idx, self._stream_channels["V1"], v2_idx, self._stream_channels["V2"])

    def _single_device_plan(self, dev: int, label: str, mode: str):
        n = sd.query_devices(dev)["max_output_channels"]
        self.mode = mode
        self._streams_planned = [label]
        self._stream_devices = {label: dev}
        self._stream_channels = {label: n}
        for e in self.plan:
            hw = e["hw"]
            # use the real CM6206 hw channel if it fits, else round-robin onto
            # whatever the device exposes (stereo bench).
            idx = (hw - 1) if (hw is not None and hw - 1 < n) else (e["ch"] - 1) % n
            self.route[e["ch"]] = (label, idx)
        log.info("%s mode: device idx %d (%d ch); 12 logical ch mapped onto %d outputs",
                 mode, dev, n, n)

    def _bench_plan(self):
        """No Vantec: drive the default output device for real (laptop speakers /
        headphones). 12 logical channels round-robin onto its outputs."""
        dev = sd.default.device[1]
        if dev is None or dev < 0:
            outs = [i for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] > 0]
            if not outs:
                raise RigError("no output devices at all -- cannot open audio.")
            dev = outs[0]
        log.warning("BENCH MODE (no Vantec found): driving real default device idx %d (%s). "
                    "12 logical channels round-robin onto its outputs; plug in the Vantecs "
                    "for discrete 12-ch addressing.", dev, sd.query_devices(dev)["name"].strip())
        self._single_device_plan(dev, label="OUT", mode="bench")

    def _log_routing(self):
        for e in self.plan:
            key_hw = self.route.get(e["ch"])
            log.debug("  ch %2d -> %-3s -> stream %s hw %s  (%s %s, %s)",
                      e["ch"], e["vantec"], key_hw[0] if key_hw else "--",
                      key_hw[1] if key_hw else "--", e["jack"], e["side"], e["site"])

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def start(self):
        if self._running:
            log.debug("start() called but already running")
            return self
        try:
            self._open_streams()
        except RigError as e:
            # device indices are flaky / some endpoints validate but won't open;
            # never leave the caller with a dead engine -- drop to BENCH on the
            # default device (still REAL audio) unless the user forced a device.
            if self.mode in ("rig", "aggregate") and not self._device_forced:
                log.warning("opening %s streams failed (%s); falling back to BENCH mode", self.mode, e)
                self._close_open_streams()
                self.route.clear()
                self._bench_plan()
                self._open_streams()
            else:
                self._close_open_streams()
                raise
        self._running = True
        log.info("HapticEngine running (%d stream(s), mode=%s)", len(self._streams), self.mode)
        return self

    def _open_streams(self):
        for key in self._streams_planned:
            dev = self._stream_devices[key]
            nch = self._stream_channels[key]
            sr = self.samplerate
            if not _device_openable(dev, nch, sr):
                alt = int(sd.query_devices(dev)["default_samplerate"])
                if alt != sr and _device_openable(dev, nch, alt):
                    log.warning("stream %s: %d Hz rejected on device %d; using its default %d Hz",
                                key, sr, dev, alt)
                    self.samplerate = sr = alt
            try:
                stream = sd.OutputStream(samplerate=sr, device=dev, channels=nch,
                                         dtype="float32", callback=self._make_callback(key, nch))
                stream.start()
            except Exception as e:  # noqa: BLE001
                raise RigError(f"could not open stream {key} on device {dev} @{nch}ch/{sr}Hz: {e}") from e
            self._streams[key] = stream
            self._capture[key] = []
            log.info("stream %s started: device=%d (%s) channels=%d sr=%d latency=%.1fms",
                     key, dev, sd.query_devices(dev)["name"].strip(), nch, sr, stream.latency * 1000.0)

    def _close_open_streams(self):
        for key, stream in list(self._streams.items()):
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        self._streams.clear()

    def stop(self):
        for key, stream in list(self._streams.items()):
            try:
                stream.stop()
                stream.close()
                log.info("stream %s stopped/closed", key)
            except Exception as e:  # noqa: BLE001
                log.warning("error closing stream %s: %s", key, e)
        self._streams.clear()
        with self._lock:
            self._voices.clear()
        self._running = False
        if self._underruns:
            log.warning("total output underruns this session: %d", self._underruns)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    # ------------------------------------------------------------------ #
    # the per-speaker control API  (what the ML calls)
    # ------------------------------------------------------------------ #
    def play(self, ch: int, intensity: int = 2, amp: float | None = None,
             freq_hz: float = 160.0, duration_ms: float = 50.0, waveform: str = "tone",
             glide_hz: tuple[float, float] = (60.0, 250.0),
             attack_ms: float = 4.0, release_ms: float = 12.0, delay_ms: float = 0.0) -> int:
        """Fire a haptic burst on ONE speaker. Non-blocking: returns a voice id
        immediately; the burst plays on the audio thread. Many voices (across both
        Vantecs) can be active at once.

        Parameters (the control surface exposed to the ML/fusion):
          ch          1..12  WHICH speaker (-> body site). Required.
          intensity   1|2|3  felt-strength level -> amplitude via the intensity map.
          amp         0..1   optional continuous override (= note velocity); wins
                             over intensity. Clamped to the clip ceiling.
          freq_hz     drive/carrier frequency (tune 60-250; heavy 3-ohm drivers
                             often strongest 80-160).
          duration_ms burst length (e.g. 30/50/80).
          waveform    "tone" (steady sine) or "glide" (chirp glide_hz[0]->[1]).
          glide_hz    (f0, f1) sweep range, used only when waveform="glide".
          attack_ms   raised-cosine attack ramp -> "punch" (sharp = crisp tap).
          release_ms  raised-cosine release ramp (fast = punchy; long = droney).
          delay_ms    schedule the start this many ms in the future (lets the ML
                             stagger e.g. a strum without blocking). 0 = now.

        Returns: an int voice id (pass to stop_voice()).
        """
        if not self._running:
            raise RuntimeError("HapticEngine.play() before start(); use `with HapticEngine() as e:`")
        if ch not in self.route:
            raise ValueError(f"channel {ch} is not routable (valid: {sorted(self.route)})")
        if intensity not in self.intensity_amp and amp is None:
            raise ValueError(f"intensity {intensity} not in {sorted(self.intensity_amp)}; "
                             f"or pass amp=0..1")
        if waveform not in ("tone", "glide"):
            raise ValueError(f"waveform must be 'tone' or 'glide', got {waveform!r}")

        base_amp = float(amp) if amp is not None else self.intensity_amp[intensity]
        gain = self.channel_gain.get(ch, 1.0)
        eff_amp = base_amp * gain
        clipped = False
        if eff_amp > self.clip_ceiling:
            log.warning("ch %d amp %.3f (base %.3f x gain %.3f) exceeds clip ceiling %.2f -> clamped",
                        ch, eff_amp, base_amp, gain, self.clip_ceiling)
            eff_amp = self.clip_ceiling
            clipped = True

        dur_s = duration_ms / 1000.0
        if waveform == "tone":
            buf = make_tone(freq_hz, dur_s, self.samplerate, eff_amp, attack_ms, release_ms)
        else:
            buf = make_glide(glide_hz[0], glide_hz[1], dur_s, self.samplerate, eff_amp,
                             attack_ms, release_ms)
        # hard safety clip in case envelope/rounding nudged a sample over
        np.clip(buf, -self.clip_ceiling, self.clip_ceiling, out=buf)

        key, hw_index = self.route[ch]
        delay_samples = max(0, int(round(delay_ms / 1000.0 * self.samplerate)))
        site = next((e["site"] for e in self.plan if e["ch"] == ch), "?")

        with self._lock:
            vid = self._next_vid
            self._next_vid += 1
            self._voices.append(Voice(
                vid=vid, ch=ch, stream_key=key, hw_index=hw_index, buffer=buf,
                delay=delay_samples, peak=float(np.max(np.abs(buf)) if buf.size else 0.0),
                spec={"intensity": intensity, "amp": round(eff_amp, 4), "freq_hz": freq_hz,
                      "duration_ms": duration_ms, "waveform": waveform, "delay_ms": delay_ms,
                      "clipped": clipped},
            ))
            n_active = len(self._voices)
        log.info("play vid=%d ch=%d (%s) -> stream %s hw %d | %s %.0fHz %.0fms amp=%.3f%s delay=%.0fms | active=%d",
                 vid, ch, site, key, hw_index, waveform, freq_hz, duration_ms, eff_amp,
                 " CLIPPED" if clipped else "", delay_ms, n_active)
        return vid

    def play_pair(self, ch_a: int, ch_b: int, **spec) -> tuple[int, int]:
        """Fire two speakers at once (e.g. a string + a fret-zone). Returns both
        voice ids. Truly simultaneous even across V1/V2 (separate USB devices)."""
        log.debug("play_pair(%d, %d, %s)", ch_a, ch_b, spec)
        return self.play(ch_a, **spec), self.play(ch_b, **spec)

    def stop_voice(self, vid: int):
        """Stop one voice early by id. (Lifecycle teardown is stop()/close().)"""
        with self._lock:
            before = len(self._voices)
            self._voices = [v for v in self._voices if v.vid != vid]
            removed = before - len(self._voices)
        log.info("stop voice vid=%d (%d removed)", vid, removed)

    def stop_all(self):
        with self._lock:
            n = len(self._voices)
            self._voices.clear()
        log.info("stop_all: cleared %d voice(s)", n)

    def active(self) -> list[int]:
        with self._lock:
            return [v.vid for v in self._voices]

    def drive_state(self) -> dict[int, float]:
        """Per-logical-channel current nominal amplitude (0 if idle). A hook for
        the browser drive[] telemetry (truth.md S2); not wired to a socket here."""
        state = {ch: 0.0 for ch in self.route}
        with self._lock:
            for v in self._voices:
                if v.delay <= 0 and 0 <= v.pos < len(v.buffer):
                    state[v.ch] = max(state[v.ch], v.peak)
        return state

    def set_channel_gain(self, ch: int, gain: float):
        self.channel_gain[ch] = float(gain)
        log.info("channel %d gain set to %.3f", ch, gain)

    def get_capture(self) -> dict[str, np.ndarray]:
        """The REAL mixed output per stream since start (capture=True). Not a
        simulation -- it's exactly what was sent to the device."""
        out = {}
        with self._lock:
            for key, blocks in self._capture.items():
                if blocks:
                    out[key] = np.concatenate(blocks, axis=0)
        return out

    # ------------------------------------------------------------------ #
    # the audio callback (one per stream) -- runs on PortAudio threads
    # ------------------------------------------------------------------ #
    def _make_callback(self, key: str, n_out: int):
        def _callback(outdata, frames, time_info, status):
            if status:
                self._underruns += 1
                log.warning("stream %s status: %s (count=%d)", key, status, self._underruns)
            if key not in self._first_block:
                self._first_block.add(key)
                log.debug("stream %s: first audio block (%d frames, %d ch)", key, frames, n_out)
            outdata.fill(0.0)
            finished = []
            with self._lock:
                for v in self._voices:
                    if v.stream_key != key:
                        continue
                    start = 0
                    if v.delay > 0:
                        if v.delay >= frames:
                            v.delay -= frames
                            continue
                        start = v.delay
                        v.delay = 0
                    avail = frames - start
                    chunk = v.buffer[v.pos:v.pos + avail]
                    n = len(chunk)
                    if n and v.hw_index < n_out:
                        outdata[start:start + n, v.hw_index] += chunk
                    v.pos += n
                    if v.pos >= len(v.buffer):
                        finished.append(v)
                for v in finished:
                    self._voices.remove(v)
                if self.capture_enabled:
                    self._capture[key].append(outdata.copy())
            for v in finished:
                log.debug("voice vid=%d ch=%d done on stream %s", v.vid, v.ch, key)
        return _callback

    def close(self):
        """Alias for stop() -- lifecycle teardown (close streams, clear voices)."""
        self.stop()


# --------------------------------------------------------------------------- #
# tiny REAL smoke demo (no mock): a single note, then a cross-axis pair
# --------------------------------------------------------------------------- #
def _demo(argv=None):
    import argparse
    import time
    p = argparse.ArgumentParser(description="HapticEngine real smoke demo.")
    p.add_argument("--device", help="force one output device (index or name substring)")
    p.add_argument("--v1"); p.add_argument("--v2")
    p.add_argument("--intensity", type=int, default=1)
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    configure_logging(a.verbose)
    with HapticEngine(samplerate=a.samplerate, v1=a.v1, v2=a.v2, device=a.device) as eng:
        log.info("DEMO: single note on ch 1")
        eng.play(1, intensity=a.intensity)
        time.sleep(0.4)
        log.info("DEMO: simultaneous pair ch 1 (back string) + ch 7 (torso fret-zone)")
        eng.play_pair(1, 7, intensity=a.intensity, duration_ms=120)
        time.sleep(0.5)
        log.info("DEMO: staggered 'strum' ch 1..6, 40 ms apart (non-blocking)")
        for i, ch in enumerate(range(1, 7)):
            eng.play(ch, intensity=a.intensity, delay_ms=i * 40, duration_ms=60)
        time.sleep(0.6)
    log.info("DEMO done.")


if __name__ == "__main__":
    _demo()
