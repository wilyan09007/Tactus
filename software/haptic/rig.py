#!/usr/bin/env python3
"""
rig.py -- shared low-level rig primitives for the Tactus haptic stack.

ONE source of truth for:
  * channel routing      logical ch 1..12 -> (vantec, hw channel, jack, side, body site)
  * device discovery     find the Vantec(s) on Win (WDM-KS) / mac (CoreAudio) / Linux (ALSA)
  * waveform synthesis   click-free tone / glide bursts with a raised-cosine envelope
  * logging              one verbose, thread-aware logger for the whole haptic package

Everything here is deterministic and hardware-facing. The "which speaker should
fire for a played note" decision is NOT here -- that belongs to the ML/fusion
(truth.md S6). This module only knows how to *address and drive* a speaker.

Imported by:  engine.py (the real-time control API), pair_test.py (the 66-pair
sweep), speaker_check.py + resonance_check.py (bring-up tools).

Routing facts (truth.md S3.2, config/channel_map.json):
  * 12 channels = 6 SK473 boxes x stereo. V1 = logical ch 1-8, V2 = ch 9-12.
  * The CM6206 inside each Vantec presents 8 channels in FL,FR,FC,LFE,RL,RR,SL,SR
    order, so a jack's hardware channel is NOT its contiguous index -- the REAR
    jack is hw 5/6, not 3/4. JACK_TO_HW encodes that empirical order; confirm by
    ear at bring-up (speaker_check.py) and write the numbers back into
    channel_map.json's alsa_ch.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# this file is at repo_root/software/haptic/rig.py
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "config" / "channel_map.json"
ENCODING = REPO_ROOT / "config" / "encoding.json"


# --------------------------------------------------------------------------- #
# Logging -- verbose + thread aware (audio callbacks run on their own threads)
# --------------------------------------------------------------------------- #
def get_logger(name: str = "tactus.haptic") -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(verbose: bool = False, level: int | None = None) -> logging.Logger:
    """Configure the package root logger once. verbose -> DEBUG, else INFO.

    Format carries the thread name so callback-thread events (the V1/V2 audio
    threads) are distinguishable from the main thread -- essential when chasing
    a glitch or an underrun."""
    root = logging.getLogger("tactus")
    if level is None:
        level = logging.DEBUG if verbose else logging.INFO
    root.setLevel(level)
    # don't stack duplicate handlers if called twice
    if not any(getattr(h, "_tactus", False) for h in root.handlers):
        h = logging.StreamHandler(sys.stderr)
        h._tactus = True  # type: ignore[attr-defined]
        h.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(h)
    root.debug("logging configured: level=%s", logging.getLevelName(level))
    return root


log = get_logger("tactus.haptic.rig")


class RigError(RuntimeError):
    """Anything wrong with the physical/audio rig (no adapter, bad device, ...)."""


# --------------------------------------------------------------------------- #
# Static rig maps
# --------------------------------------------------------------------------- #
# Name fragments a CM6206/Vantec 7.1 adapter tends to enumerate under.
VANTEC_HINTS = ("vantec", "cm6206", "c-media", "cmedia", "usb audio",
                "usb sound", "usb advanced audio", "7.1")

# Standard Windows 8-channel (7.1) order -> this device's jack labels (for the
# bring-up tools' human-facing prompts).
JACK_MAP = {
    1: ("FRONT",    "Left",    "green FRONT jack, tip"),
    2: ("FRONT",    "Right",   "green FRONT jack, ring"),
    3: ("CENTER",   "Center",  "orange CENTER/BASS jack, tip"),
    4: ("BASS",     "Sub/LFE", "orange CENTER/BASS jack, ring"),
    5: ("BACK",     "Left",    "BACK '7.1' jack, tip"),
    6: ("BACK",     "Right",   "BACK '7.1' jack, ring"),
    7: ("SURROUND", "Left",    "SURROUND '5.1' jack, tip"),
    8: ("SURROUND", "Right",   "SURROUND '5.1' jack, ring"),
}

# channel_map.json labels a jack by name ("front"/"rear"/"center"/"side") + side
# ("L"/"R").  CM6206 presents FL,FR,FC,LFE,RL,RR,SL,SR -> a jack's 1-based hw
# channel is NOT the contiguous alsa_ch in the map.  Empirical order, confirmed
# at bring-up (truth.md "ALSA enumeration trap").
JACK_TO_HW = {
    ("front",  "L"): 1, ("front",  "R"): 2,
    ("center", "L"): 3, ("center", "R"): 4,
    ("rear",   "L"): 5, ("rear",   "R"): 6,
    ("side",   "L"): 7, ("side",   "R"): 8,
}


def channel_label(ch: int):
    """7.1 jack/side label for a 1-based *hardware* channel (bring-up prompts)."""
    return JACK_MAP.get(ch, (f"ch{ch}", "", ""))


# --------------------------------------------------------------------------- #
# Config loaders
# --------------------------------------------------------------------------- #
def load_pulse_defaults(config_path: Path = CONFIG) -> dict:
    """Pull pulse defaults (freq/duration/intensity levels) from channel_map.json
    so every tool matches the rig spec."""
    try:
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        pulse = data.get("pulse", {})
        out = {
            "freq_hz": float(pulse.get("freq_hz", 160.0)),
            "duration_ms": float(pulse.get("duration_ms", 50.0)),
            "intensity_levels": int(data.get("intensity_levels", 3)),
        }
    except Exception as e:  # noqa: BLE001 -- defaults must always work
        log.warning("could not read pulse defaults from %s (%s); using built-ins", config_path, e)
        out = {"freq_hz": 160.0, "duration_ms": 50.0, "intensity_levels": 3}
    log.debug("pulse defaults: %s", out)
    return out


def load_channel_plan(config_path: Path = CONFIG) -> list[dict]:
    """Load the 12 logical channels, resolving each to a physical (vantec, hw
    channel) via JACK_TO_HW. Returns a list sorted by logical ch."""
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    plan = []
    for c in data["channels"]:
        hw = JACK_TO_HW.get((c["jack"], c["side"]))
        if hw is None:
            log.warning("ch %s: no hw mapping for jack=%s side=%s", c["ch"], c["jack"], c["side"])
        plan.append({
            "ch": c["ch"], "vantec": c["vantec"], "jack": c["jack"], "side": c["side"],
            "hw": hw, "box": c.get("box", ""), "site": c.get("site", ""),
            "axis": c.get("axis", ""), "index": c.get("index"),
            "actuator": c.get("actuator", ""),
        })
    plan.sort(key=lambda e: e["ch"])
    log.debug("loaded channel plan: %d channels", len(plan))
    return plan


# --------------------------------------------------------------------------- #
# Device discovery (cross-platform; mixer-bypassing endpoints)
# --------------------------------------------------------------------------- #
def list_output_devices() -> list[tuple[int, str, int, str]]:
    """(idx, name, out_channels, host_api) for every output-capable device."""
    hostapis = sd.query_hostapis()
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            out.append((i, d["name"].strip(), d["max_output_channels"],
                        hostapis[d["hostapi"]]["name"]))
    return out


def resolve_device(spec) -> int:
    """Resolve an int index or a name substring to an output device index.
    Raises RigError on no/ambiguous-with-no-good-pick match."""
    devices = sd.query_devices()
    if spec is None:
        raise RigError("resolve_device(None): pass an index or a name substring.")
    try:
        idx = int(spec)
        if devices[idx]["max_output_channels"] < 1:
            raise RigError(f"device {idx} ('{devices[idx]['name'].strip()}') has no outputs.")
        log.debug("resolve_device(%r) -> idx %d (%s)", spec, idx, devices[idx]["name"].strip())
        return idx
    except (ValueError, TypeError):
        pass  # not an int -> name substring
    matches = [i for i, d in enumerate(devices)
               if str(spec).lower() in d["name"].lower() and d["max_output_channels"] > 0]
    if not matches:
        raise RigError(f"no output device name contains {spec!r}.")
    if len(matches) > 1:
        apis = sd.query_hostapis()

        def _score(i):
            d = devices[i]
            wdmks = "wdm-ks" in apis[d["hostapi"]]["name"].lower()
            return (d["max_output_channels"], 1 if wdmks else 0)
        best = max(matches, key=_score)
        log.info("%r matched %d devices; picked idx %d (%s, %d ch)", spec, len(matches), best,
                 devices[best]["name"].strip(), devices[best]["max_output_channels"])
        return best
    return matches[0]


def _discrete_usb_endpoint(d, api_name: str) -> bool:
    """True if device dict `d` is our USB multichannel DAC on the OS path that
    BYPASSES any down-mix: Windows WDM-KS, macOS Core Audio, Linux ALSA."""
    if d["max_output_channels"] < 8:
        return False
    if not any(h in d["name"].lower() for h in VANTEC_HINTS):
        return False
    api = api_name.lower()
    if sys.platform.startswith("win"):
        return "wdm-ks" in api
    if sys.platform == "darwin":
        return "core audio" in api
    return "alsa" in api


# Host-API preference for discrete multichannel: mixer-bypassing first. WDM-KS /
# Core Audio / ALSA route each channel to its own jack; WASAPI(excl) next; MME &
# DirectSound shared-mode may fold ch3-8 onto FRONT (still openable -> last).
HOST_API_PREF = ("wdm-ks", "wasapi", "core audio", "alsa", "directsound", "mme")


def _silent_cb(outdata, frames, time_info, status):  # noqa: ANN001
    outdata.fill(0.0)


def _device_openable(idx: int, channels: int, samplerate: int) -> bool:
    """True iff `idx` supports this width/rate. Uses check_output_settings (a
    NON-destructive query), NOT a real open.

    Hard-won: do NOT probe by actually opening the stream. WDM-KS (the mixer-bypass
    path) is finicky -- a probe open/close leaves the KS pin in a state where the
    engine's real open then fails (-9999 GLE=1). check_output_settings doesn't grab
    the device, so the engine's commit is the FIRST and only real KS open (the
    pattern that opens cleanly). Any endpoint that still won't open is caught by the
    engine's option fallback. Indices are unstable across *processes*, but discovery
    and open are back-to-back in one process, so they agree."""
    try:
        sd.check_output_settings(device=idx, channels=channels, samplerate=samplerate)
        return True
    except Exception as e:  # noqa: BLE001
        log.debug("device %s rejects %dch/%dHz: %s", idx, channels, samplerate, e)
        return False


# Shared-mode host APIs whose mixer down-mixes a multichannel stream onto the
# device's configured layout (folds ch3-8 onto FRONT unless the device is set to
# 7.1 in the OS). Bypass APIs (WDM-KS / Core Audio / ALSA / WASAPI-exclusive)
# route each channel to its own jack.
SHARED_MODE_APIS = ("mme", "directsound", "wasapi")


def is_bypass_api(api_name: str) -> bool:
    """True if this host API routes channels discretely (no shared-mode fold)."""
    a = api_name.lower()
    return ("wdm-ks" in a) or ("core audio" in a) or ("alsa" in a)


def bypass_adapters_noprobe(min_channels: int = 8) -> list[int]:
    """Indices of >=min_channels hint-matching endpoints on a BYPASS host API
    (WDM-KS / CoreAudio / ALSA), by ENUMERATION ONLY -- no open/format probe.

    Probing WDM-KS (even check_output_settings can instantiate the pin) destabilizes
    it so a later real open fails. The proven discrete path (resonance_check) finds
    KS purely by name+api enumeration and opens it ONCE; this mirrors that."""
    apis = sd.query_hostapis()
    out = [i for i, d in enumerate(sd.query_devices())
           if d["max_output_channels"] >= min_channels
           and any(h in d["name"].lower() for h in VANTEC_HINTS)
           and is_bypass_api(apis[d["hostapi"]]["name"])]
    out = sorted(set(out))
    log.debug("bypass adapters (no probe): %s", out)
    return out


def shared_would_fold(min_channels: int = 8) -> bool:
    """Windows: would a shared-mode (MME/DirectSound/WASAPI) multichannel stream be
    DOWN-MIXED onto the device's configured layout (folding ch3-8 onto FRONT)?

    The tell is the WASAPI **shared mix-format width**: if no hint-matching WASAPI
    endpoint exposes >= min_channels, the USB device is configured Stereo (2.0) not
    7.1, so the OS mixer folds. Bypass paths (WDM-KS) are unaffected. Non-Windows
    OSes don't fold this way -> False."""
    if not sys.platform.startswith("win"):
        return False
    apis = sd.query_hostapis()
    widths = [d["max_output_channels"] for d in sd.query_devices()
              if d["max_output_channels"] > 0
              and "wasapi" in apis[d["hostapi"]]["name"].lower()
              and any(h in d["name"].lower() for h in VANTEC_HINTS)]
    folds = (not widths) or (max(widths) < min_channels)
    log.debug("shared_would_fold: wasapi USB widths=%s -> %s", widths, folds)
    return folds


def candidate_adapters(min_channels: int = 8, samplerate: int = 48000) -> list[dict]:
    """Hint-matching, >=min_channels, *currently-openable* output endpoints, each
    with its host API. This is the validated replacement for blindly trusting the
    device list (which contains stale/phantom indices)."""
    devices = sd.query_devices()
    apis = sd.query_hostapis()
    out = []
    for i, d in enumerate(devices):
        if d["max_output_channels"] < min_channels:
            continue
        if not any(h in d["name"].lower() for h in VANTEC_HINTS):
            continue
        api = apis[d["hostapi"]]["name"]
        # Do NOT probe-open bypass (WDM-KS) endpoints -- even a probe destabilizes the
        # KS pin (-9996 on the next real open; also breaks resonance_check). Include
        # them by enumeration; only shared-mode endpoints get the openability probe.
        if is_bypass_api(api) or _device_openable(i, min_channels, samplerate):
            out.append({"idx": i, "name": d["name"].strip(),
                        "ch": d["max_output_channels"], "api": api})
    log.debug("validated candidate adapters (>=%dch): %s", min_channels,
              [(c["idx"], c["name"], c["api"]) for c in out])
    return out


def discrete_output_endpoints(samplerate: int = 48000) -> list[int]:
    """Indices of currently-openable >=8-ch hint-matching adapters."""
    return [c["idx"] for c in candidate_adapters(8, samplerate)]


def _two_distinct(group: list[dict]) -> list[dict] | None:
    chosen, idxs = [], set()
    for c in sorted(group, key=lambda x: x["idx"]):
        if c["idx"] not in idxs:
            idxs.add(c["idx"])
            chosen.append(c)
        if len(chosen) == 2:
            return chosen
    return None


def resolve_two_adapters(v1_spec=None, v2_spec=None, samplerate: int = 48000) -> tuple[int, int]:
    """Return (v1_idx, v2_idx) for the two Vantecs. Explicit specs win; else
    auto-pick two DISTINCT, currently-openable >=8-ch endpoints, preferring a
    mixer-bypassing host API (WDM-KS/CoreAudio/ALSA) over shared-mode (MME/DS).
    Raises RigError if it can't find two (caller may fall back to bench mode)."""
    if v1_spec is not None and v2_spec is not None:
        return resolve_device(v1_spec), resolve_device(v2_spec)
    cands = candidate_adapters(8, samplerate)
    if len(cands) < 2:
        raise RigError(
            f"need 2 openable >=8-ch USB adapters; found {[(c['idx'], c['name'], c['api']) for c in cands]}. "
            f"Plug in both Vantecs, or pass v1=/v2= explicitly. (macOS: set each to its 8-ch format in "
            f"Audio MIDI Setup, or build ONE Aggregate Device and pass v1=v2=that idx.)")
    by_api: dict[str, list[dict]] = {}
    for c in cands:
        by_api.setdefault(c["api"].lower(), []).append(c)
    for pref in HOST_API_PREF:
        group = next((v for k, v in by_api.items() if pref in k), None)
        if group:
            pick = _two_distinct(group)
            if pick:
                log.info("rig adapters via %s: V1=idx %d (%s)  V2=idx %d (%s) "
                         "(lower=V1; swap with v1=/v2= if sites are reversed)",
                         group[0]["api"], pick[0]["idx"], pick[0]["name"], pick[1]["idx"], pick[1]["name"])
                return pick[0]["idx"], pick[1]["idx"]
    pick = _two_distinct(cands)  # cross-API fallback (may be the same physical adapter twice -> warn)
    if pick:
        log.warning("no single host API exposes two adapters; using cross-API %s + %s "
                    "(verify they are two DIFFERENT physical Vantecs)",
                    (pick[0]["idx"], pick[0]["api"]), (pick[1]["idx"], pick[1]["api"]))
        return pick[0]["idx"], pick[1]["idx"]
    raise RigError("could not select two distinct adapters.")


# --------------------------------------------------------------------------- #
# Waveform synthesis -- click-free bursts
# --------------------------------------------------------------------------- #
def apply_envelope(sig: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """Raised-cosine attack/release ramps in place. A sharp attack + fast decay =
    "punchy" / more distinguishable; flat = mushy (docs/18 Exp 2). Ramps also kill
    the start/stop click that sounds harsh AND stresses the voice coil.

    Attack+release are clamped so they never overrun a short burst."""
    n = len(sig)
    if n == 0:
        return sig
    a = min(int(round(attack_ms / 1000.0 * sr)), n // 2)
    r = min(int(round(release_ms / 1000.0 * sr)), n // 2)
    if a > 0:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(a) / a))
        sig[:a] *= ramp
    if r > 0:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(r) / r))
        sig[-r:] *= ramp[::-1]
    return sig


def make_tone(freq: float, dur: float, sr: int, amp: float,
              attack_ms: float = 10.0, release_ms: float | None = None) -> np.ndarray:
    """A single steady sine burst, enveloped. dur in seconds."""
    if release_ms is None:
        release_ms = attack_ms
    n = int(round(dur * sr))
    t = np.arange(n) / sr
    tone = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return apply_envelope(tone, sr, attack_ms, release_ms)


def make_glide(f0: float, f1: float, dur: float, sr: int, amp: float,
               attack_ms: float = 12.0, release_ms: float | None = None) -> np.ndarray:
    """A linear-frequency chirp f0 -> f1 over `dur` s, enveloped. Driving through
    the band makes a heavy driver's resonant peak audible/palpable (resonance
    finder, truth.md S5 tuning sweep 60-250 Hz)."""
    if release_ms is None:
        release_ms = attack_ms
    n = int(round(dur * sr))
    t = np.arange(n) / sr
    # instantaneous phase of a linear sweep
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * dur))
    tone = (amp * np.sin(phase)).astype(np.float32)
    return apply_envelope(tone, sr, attack_ms, release_ms)


# --------------------------------------------------------------------------- #
# Blocking single-channel playback (bring-up tools only; NOT the engine path)
# --------------------------------------------------------------------------- #
def play_signal(signal: np.ndarray, ch: int, device, sr: int, maxch: int) -> None:
    """Play `signal` on ONE 1-based channel inside a FIXED maxch-wide frame,
    blocking until done. Used by the bring-up tools. A fixed full-width frame on a
    mixer-bypassing endpoint routes each channel straight to its own jack (a
    narrow mapping=[ch] stream gets up/down-mixed and collapses onto FRONT).

    The real-time engine does NOT use this -- it streams overlapping voices via a
    persistent callback instead (engine.py)."""
    buf = np.zeros((len(signal), maxch), dtype=np.float32)
    if 1 <= ch <= maxch:
        buf[:, ch - 1] = signal
    sd.play(buf, samplerate=sr, device=device, blocking=True)
