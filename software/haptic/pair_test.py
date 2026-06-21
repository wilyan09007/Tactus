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

import numpy as np
import sounddevice as sd

from engine import HapticEngine
from rig import (
    RigError, bypass_adapters_noprobe, configure_logging, get_logger, is_bypass_api,
    load_channel_plan, load_pulse_defaults, make_glide, make_tone, resolve_device,
    shared_would_fold,
)

log = get_logger("tactus.haptic.pair_test")

# intensity level -> amplitude (mirror of engine.DEFAULT_INTENSITY_AMP, for the
# blocking discrete path which doesn't build a HapticEngine)
DISCRETE_INTENSITY_AMP = {1: 0.30, 2: 0.60, 3: 0.90}

_REPLUG = (
    "\n" + "=" * 74 + "\n"
    "WDM-KS won't open (-9996): the Kernel-Streaming pins are STUCK -- a prior run\n"
    "grabbed them and Windows hasn't released them (it also breaks resonance_check).\n\n"
    "FIX: unplug BOTH Vantec USB dongles, wait ~3s, replug, then re-run.\n"
    "     (A reboot also clears it.) Re-running alone won't help once stuck.\n\n"
    "AVOID re-sticking them: don't run the STREAMING path (engine.py /\n"
    "pair_test --streaming / low_note_all) in between -- on this box it can re-grab KS.\n\n"
    "No-KS alternative: set BOTH USB Sound Devices to '7.1 Surround' in Windows Sound,\n"
    "then:  python pair_test.py --streaming   (DirectSound then routes discretely).\n"
    + "=" * 74)


def _reinit_portaudio():
    """Re-init PortAudio so device enumeration is fresh and stuck KS pins from a
    prior run are released WITHOUT a physical replug (often enough on its own)."""
    try:
        sd._terminate()
        sd._initialize()
        log.debug("PortAudio re-initialized (fresh device enumeration)")
    except Exception as e:  # noqa: BLE001
        log.debug("PortAudio reinit skipped: %s", e)


def _preflight_open(dev: int, sr: int):
    """Actually open `dev` once (callback mode, what KS needs), with a samplerate
    fallback. Returns (ok, working_sr). A clean single open = the device is healthy."""
    nch = sd.query_devices(dev)["max_output_channels"]
    for cand in (sr, int(sd.query_devices(dev)["default_samplerate"])):
        try:
            s = sd.OutputStream(device=dev, channels=nch, samplerate=cand,
                                dtype="float32", callback=lambda o, f, t, st: o.fill(0.0))
            s.close()
            return True, cand
        except Exception as e:  # noqa: BLE001
            log.debug("preflight idx%d @%dHz failed: %s", dev, cand, e)
    return False, sr


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


def run_discrete_sweep(args, plan) -> None:
    """DISCRETE sweep via blocking WDM-KS, one stream at a time -- the proven path
    `resonance_check.py` uses, extended to pairs. No 2-simultaneous-KS limit and no
    shared-mode fold, so every channel hits its OWN jack.

      intra-Vantec pair -> both channels in ONE KS buffer = simultaneous + discrete.
      cross-Vantec pair -> SERIALIZED (this driver can't drive two USB devices at
                           once via KS). For simultaneous cross-Vantec use macOS
                           CoreAudio, or set both devices to 7.1 and run --streaming.
    """
    _reinit_portaudio()  # release any stuck KS pins from a prior run (often avoids a manual replug)
    amp = args.amp if args.amp is not None else DISCRETE_INTENSITY_AMP.get(args.intensity, 0.6)
    # resolve the two WDM-KS (bypass) adapters by ENUMERATION ONLY -- no probe, so the
    # KS pins stay healthy and open cleanly (the resonance_check pattern).
    if args.v1 is not None and args.v2 is not None:
        v1, v2 = resolve_device(args.v1), resolve_device(args.v2)
    else:
        eps = bypass_adapters_noprobe()
        if len(eps) < 2:
            sys.exit(f"discrete sweep needs two bypass (WDM-KS) adapters; found {eps}. "
                     f"Plug in both Vantecs, or pass --v1/--v2 (see `low_note_all.py --list`).")
        if len(eps) > 2:
            log.warning("%d bypass adapters %s; using first two -- override with --v1/--v2", len(eps), eps)
        v1, v2 = eps[0], eps[1]
    dev_of = {"V1": v1, "V2": v2}
    apis = sd.query_hostapis()

    def api_of(dev):
        return apis[sd.query_devices(dev)["hostapi"]]["name"]

    for k, dev in dev_of.items():
        api = api_of(dev)
        if not is_bypass_api(api):
            log.warning("%s (idx%d) resolved to '%s' -- NOT a mixer-bypass API, so it may STILL "
                        "fold. Pass the WDM-KS indices with --v1/--v2 (find them with "
                        "`python low_note_all.py --list`).", k, dev, api)
        else:
            log.info("%s = idx%d (%s, discrete)", k, dev, api)

    # PRE-FLIGHT: actually open each KS device once (the stuck-pin check). On failure
    # print the replug remedy instead of a raw -9996 traceback mid-sweep.
    sr = args.samplerate
    ok1, sr = _preflight_open(v1, sr)
    ok2, sr2 = _preflight_open(v2, sr) if ok1 else (False, sr)
    if not (ok1 and ok2):
        sys.exit(_REPLUG)
    if sr2 != sr:
        log.warning("V1/V2 disagree on rate (%d/%d Hz); using %d", sr, sr2, min(sr, sr2))
        sr = min(sr, sr2)
    log.info("pre-flight OK: both KS devices open at %d Hz", sr)
    maxch_of = {dev: sd.query_devices(dev)["max_output_channels"] for dev in (v1, v2)}

    if args.glide:
        sig = make_glide(args.glide_lo, args.glide_hi, args.dur / 1000.0, sr, amp, args.attack, args.release)
    else:
        sig = make_tone(args.freq, args.dur / 1000.0, sr, amp, args.attack, args.release)

    def play_on(dev, hw_list):
        n = maxch_of[dev]
        buf = np.zeros((len(sig), n), dtype=np.float32)
        for hw in hw_list:
            if hw is not None and hw - 1 < n:
                buf[:, hw - 1] = sig
        try:
            sd.play(buf, samplerate=sr, device=dev, blocking=True)
        except sd.PortAudioError as e:
            sd.stop()
            if "9996" in str(e) or "Invalid device" in str(e):
                sys.exit(_REPLUG)
            sys.exit(f"audio error on idx{dev}: {e}")

    chans = parse_channel_list(args.channels)
    pairs = list(itertools.combinations(chans, 2))
    log.info("DISCRETE sweep (blocking WDM-KS, per-jack, no fold): V1=idx%d V2=idx%d | "
             "%d channels -> %d pairs%s | %s amp=%.2f | gap=%.2fs",
             v1, v2, len(chans), len(pairs), " (+singles)" if args.singles else "",
             "glide" if args.glide else f"{args.freq:g}Hz/{args.dur:g}ms", amp, args.gap)

    if args.singles:
        log.info("=== SINGLES (%d) ===", len(chans))
        for n, ch in enumerate(chans, 1):
            e = plan[ch - 1]
            log.info("[single %2d/%d] ch %2d  %-4s %s", n, len(chans), ch, e["vantec"], e["site"])
            play_on(dev_of[e["vantec"]], [e["hw"]])
            time.sleep(args.gap)

    log.info("=== PAIR SWEEP (%d pairs) ===", len(pairs))
    cross = 0
    for n, (a, b) in enumerate(pairs, 1):
        ea, eb = plan[a - 1], plan[b - 1]
        if ea["vantec"] == eb["vantec"]:
            log.info("[pair %2d/%d] ch %2d + ch %2d  %-4s  %-12s + %-12s  SIMULTANEOUS-discrete",
                     n, len(pairs), a, b, ea["vantec"], ea["site"], eb["site"])
            play_on(dev_of[ea["vantec"]], [ea["hw"], eb["hw"]])
        else:
            cross += 1
            log.info("[pair %2d/%d] ch %2d (%-4s %-12s) + ch %2d (%-4s %-12s)  CROSS-VANTEC -> SERIALIZED",
                     n, len(pairs), a, ea["vantec"], ea["site"], b, eb["vantec"], eb["site"])
            play_on(dev_of[ea["vantec"]], [ea["hw"]])
            play_on(dev_of[eb["vantec"]], [eb["hw"]])
        time.sleep(args.gap)

    log.info("discrete sweep complete: %d pairs (%d cross-Vantec serialized, %d intra simultaneous). "
             "Simultaneous cross-Vantec needs macOS CoreAudio or a 7.1 + --streaming rig.",
             len(pairs), cross, len(pairs) - cross)
    log.info("done.")


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
    p.add_argument("--discrete", action="store_true",
                   help="force the DISCRETE blocking WDM-KS sweep (per-jack, no fold; intra-Vantec "
                        "pairs simultaneous, cross-Vantec serialized) -- like resonance_check")
    p.add_argument("--streaming", action="store_true",
                   help="force the streaming-engine sweep (true simultaneous cross-Vantec; uses "
                        "shared-mode if KS can't do 2 streams -> may FOLD on Windows)")
    p.add_argument("--verbose", action="store_true", help="DEBUG logging (per-voice lifecycle)")
    args = p.parse_args(argv)

    configure_logging(args.verbose)
    plan = load_channel_plan()
    requested = parse_channel_list(args.channels)

    # On a box where shared-mode would FOLD (Windows device set to Stereo) the
    # streaming engine can't stay discrete, so default to the blocking WDM-KS sweep
    # (per-jack, like resonance_check). --streaming forces the engine; --discrete forces KS.
    use_discrete = args.discrete or (not args.streaming and not args.device and shared_would_fold())
    if use_discrete:
        log.info("using DISCRETE blocking WDM-KS sweep%s",
                 " (--discrete)" if args.discrete else " (shared-mode would fold here; --streaming to override)")
        return run_discrete_sweep(args, plan)

    spec = dict(
        intensity=args.intensity, amp=args.amp, freq_hz=args.freq, duration_ms=args.dur,
        waveform="glide" if args.glide else "tone", glide_hz=(args.glide_lo, args.glide_hi),
        attack_ms=args.attack, release_ms=args.release,
    )
    burst_s = args.dur / 1000.0
    try:
        eng = HapticEngine(samplerate=args.samplerate, v1=args.v1, v2=args.v2,
                           device=args.device, capture=bool(args.wav))
    except RigError as e:
        sys.exit(f"could not build engine: {e}")

    with eng:
        # only sweep channels the chosen output mode can actually address (a single
        # Vantec exposes ch1-8 only; bench/rig expose all 12)
        chans = [c for c in requested if c in eng.route]
        dropped = [c for c in requested if c not in eng.route]
        if dropped:
            log.warning("channels %s are NOT routable in mode '%s' -> skipped", dropped, eng.mode)
        pairs = list(itertools.combinations(chans, 2))
        log.info("pair_test: mode=%s | %d routable channels -> %d pairs%s | spec: %s | gap=%.2fs",
                 eng.mode, len(chans), len(pairs), " (+singles)" if args.singles else "", spec, args.gap)
        cross_count = 0

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
