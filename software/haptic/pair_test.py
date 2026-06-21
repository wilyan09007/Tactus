#!/usr/bin/env python3
"""
pair_test.py -- auto-paced sweep of ALL 66 speaker pairs (real audio, no mocks).

Fires every unordered pair of the 12 logical speakers C(12,2)=66, BOTH speakers
ON simultaneously, for a fixed burst + gap, logging exactly which (ch_a, ch_b)
and body sites fired and whether the pair is intra-Vantec or cross-Vantec. This
verifies, on the real rig:
  * simultaneous drive works -- especially CROSS-Vantec pairs (one speaker on V1,
    one on V2 = two separate USB devices), which sequential blocking playback
    cannot do at all;
  * the "<= 2 drivers active at once" power assumption (truth.md S3.5) holds;
  * the two body sites in each pair are distinguishable (funnelling check,
    docs/07 rule 1 -- separate simultaneous activations spatially).

It drives the REAL HapticEngine -> real sounddevice output. With both Vantecs
plugged in you get discrete 12-ch addressing; with none, the engine's bench mode
plays on the laptop's real output so the sweep + logs still run end to end.

Usage:
    python pair_test.py                       # all 66 pairs, defaults from channel_map
    python pair_test.py --singles             # warm up with the 12 singles first
    python pair_test.py --channels 1,2,7,9    # only pairs among these channels
    python pair_test.py --intensity 3 --freq 120 --dur 80 --gap 0.25
    python pair_test.py --glide               # glide bursts (resonance feel)
    python pair_test.py --device "USB Sound"  # force one device (e.g. a single Vantec)
    python pair_test.py --v1 12 --v2 13       # pin the two Vantecs
    python pair_test.py --wav out/pairs       # also CAPTURE the real mixed output to WAV
    python pair_test.py --verbose             # DEBUG logs (per-voice lifecycle)
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

from engine import HapticEngine
from rig import RigError, configure_logging, get_logger, load_channel_plan, load_pulse_defaults

log = get_logger("tactus.haptic.pair_test")


def parse_channel_list(spec: str | None) -> list[int]:
    if not spec:
        return list(range(1, 13))
    chans = [int(c) for c in spec.split(",") if c.strip()]
    bad = [c for c in chans if not 1 <= c <= 12]
    if bad:
        sys.exit(f"channels must be 1..12; got bad values {bad}")
    return sorted(set(chans))


def site_of(plan, ch):
    return next((e["site"] for e in plan if e["ch"] == ch), "?")


def vantec_of(plan, ch):
    return next((e["vantec"] for e in plan if e["ch"] == ch), "?")


def main(argv=None):
    d = load_pulse_defaults()
    p = argparse.ArgumentParser(description="Auto sweep of all 66 speaker pairs.")
    p.add_argument("--channels", help="comma list to restrict the pair pool (default all 12)")
    p.add_argument("--singles", action="store_true", help="play the 12 singles first as a warm-up")
    p.add_argument("--intensity", type=int, default=2, choices=[1, 2, 3], help="felt-strength level")
    p.add_argument("--amp", type=float, default=None, help="continuous amplitude override 0..1")
    p.add_argument("--freq", type=float, default=d["freq_hz"], help=f"drive Hz (default {d['freq_hz']:g})")
    p.add_argument("--dur", type=float, default=d["duration_ms"], help=f"burst ms (default {d['duration_ms']:g})")
    p.add_argument("--gap", type=float, default=0.25, help="seconds of silence between pairs (default 0.25)")
    p.add_argument("--glide", action="store_true", help="use glide (chirp) bursts instead of steady tone")
    p.add_argument("--glide-lo", type=float, default=60.0)
    p.add_argument("--glide-hi", type=float, default=250.0)
    p.add_argument("--attack", type=float, default=4.0, help="attack ms (punch)")
    p.add_argument("--release", type=float, default=12.0, help="release ms")
    p.add_argument("--device", help="force ONE output device (index or name substring)")
    p.add_argument("--v1", help="device index/name for Vantec V1 (ch1-8)")
    p.add_argument("--v2", help="device index/name for Vantec V2 (ch9-12)")
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--wav", help="capture the REAL mixed output and save per-stream WAVs to this prefix")
    p.add_argument("--verbose", action="store_true", help="DEBUG logging (per-voice lifecycle)")
    args = p.parse_args(argv)

    configure_logging(args.verbose)
    plan = load_channel_plan()
    chans = parse_channel_list(args.channels)
    pairs = list(itertools.combinations(chans, 2))

    spec = dict(
        intensity=args.intensity, amp=args.amp, freq_hz=args.freq, duration_ms=args.dur,
        waveform="glide" if args.glide else "tone", glide_hz=(args.glide_lo, args.glide_hi),
        attack_ms=args.attack, release_ms=args.release,
    )
    log.info("pair_test: %d channels -> %d pairs%s | spec: %s | gap=%.2fs",
             len(chans), len(pairs), " (+12 singles)" if args.singles else "", spec, args.gap)

    burst_s = args.dur / 1000.0
    cross_count = 0
    try:
        eng = HapticEngine(samplerate=args.samplerate, v1=args.v1, v2=args.v2,
                           device=args.device, capture=bool(args.wav))
    except RigError as e:
        sys.exit(f"could not build engine: {e}")

    with eng:
        if args.singles:
            log.info("=== SINGLES warm-up (%d) ===", len(chans))
            for n, ch in enumerate(chans, 1):
                log.info("[single %2d/%d] ch %2d  %-4s  %s", n, len(chans), ch,
                         vantec_of(plan, ch), site_of(plan, ch))
                eng.play(ch, **spec)
                time.sleep(burst_s + args.gap)

        log.info("=== PAIR SWEEP (%d pairs) ===", len(pairs))
        for n, (a, b) in enumerate(pairs, 1):
            va, vb = vantec_of(plan, a), vantec_of(plan, b)
            cross = va != vb
            cross_count += cross
            log.info("[pair %2d/%d] ch %2d (%-4s %-12s) + ch %2d (%-4s %-12s)  %s",
                     n, len(pairs), a, va, site_of(plan, a), b, vb, site_of(plan, b),
                     "CROSS-VANTEC" if cross else "intra-Vantec")
            eng.play_pair(a, b, **spec)
            time.sleep(burst_s + args.gap)

        log.info("sweep complete: %d pairs (%d cross-Vantec, %d intra-Vantec)",
                 len(pairs), cross_count, len(pairs) - cross_count)

        if args.wav:
            _save_capture(eng, args.wav, args.samplerate)

    log.info("done.")


def _save_capture(eng: HapticEngine, prefix: str, sr: int):
    import numpy as np
    from scipy.io import wavfile
    cap = eng.get_capture()
    if not cap:
        log.warning("--wav requested but nothing was captured")
        return
    out = Path(prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    for key, data in cap.items():
        path = out.with_name(f"{out.name}_{key}.wav")
        # float32 [-1,1] -> int16 for portability
        i16 = np.clip(data, -1.0, 1.0)
        wavfile.write(str(path), sr, (i16 * 32767).astype(np.int16))
        log.info("captured stream %s -> %s  (%.2fs, %d ch)", key, path, len(data) / sr, data.shape[1])


if __name__ == "__main__":
    main()
