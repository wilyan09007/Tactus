#!/usr/bin/env python3
"""
low_note_all.py -- play one LOW note on ALL 12 haptic speakers, simultaneously.

Fires every routable logical channel (1..12, across BOTH Vantecs) at once with a
single low-frequency burst, through the real HapticEngine voice mixer. The whole
array hits together -- the "everything on" smoke test / attention buzz.

"CONCENTRATES TO ONE BOX" -- the bug, and how this script avoids it
(measured on the dev rig; truth.md S2 + S8.2, engine.py, rig.py):

  PRIMARY cause (what actually bites on the dev box): the two Vantecs cannot
  stream simultaneously. Both CM6206 adapters sit on ONE USB controller/hub, so
  once the first 8-ch/48k stream is open the second adapter's isochronous OUT
  endpoint can't be allocated -- it fails to open on EVERY host API (WDM-KS /
  DirectSound / MME), in either order. The engine then walks its option list down
  to a SINGLE-Vantec fallback: one box plays, and logical ch9-12 (the other box's
  torso zones) get round-robined onto it. Result: the whole array concentrates on
  one box. THE FIX is physical -- put the two Vantecs on DIFFERENT USB controllers
  (ports on opposite sides of the chassis / a separate USB card; a passive hub on
  the same controller will NOT help), then the engine opens mode=rig with both.
  (macOS: build one Aggregate Device of both -> mode=aggregate, all 12 at once.)

  SECONDARY cause (if both boxes DO open but on a shared-mode API): Windows'
  MME/DirectSound/WASAPI-shared hand the stream to the OS mixer, which down-mixes
  8ch onto the device's configured layout; if that's Stereo, each Vantec's
  channels fold onto its FRONT jack. Fix: use the WDM-KS endpoints (mixer bypass),
  or set the USB device to 7.1 Surround in Windows Sound.

  So this script does NOT just blast a tone and hope. After the streams open it
  checks what the engine actually got and STOPS (with the exact remedy) if the
  array can't reach both boxes -- i.e. it collapsed to single/bench mode, or a
  Vantec is on a folding shared-mode API. Pass --force to play anyway.

Discrete 12-ch IS the default path: the engine prefers a 2-Vantec WDM-KS rig,
opens one full-width OutputStream per Vantec, and writes each logical channel
into its own hardware column -- all 12 on their own jacks, both boxes, at once.

Usage:
    python low_note_all.py --list                 # find your WDM-KS device indices
    python low_note_all.py                         # low note on all 12, defaults
    python low_note_all.py --freq 70 --dur 600     # lower + longer
    python low_note_all.py --v1 12 --v2 13         # pin the two Vantecs (WDM-KS)
    python low_note_all.py --count 3 --gap 0.5     # pulse the whole array 3x
    python low_note_all.py --force                 # play even if it can't reach both boxes
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# run-from-anywhere: this file sits next to engine.py / rig.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sounddevice as sd

from engine import HapticEngine
from rig import (
    RigError, configure_logging, get_logger, is_bypass_api,
    list_output_devices, load_channel_plan,
)

log = get_logger("tactus.haptic.low_note_all")


def stream_report(eng: HapticEngine) -> list[tuple[str, int, str, str, bool]]:
    """(key, idx, name, host_api, is_bypass) for every stream the engine actually
    opened -- the ground truth for whether we're on a discrete or folding path."""
    apis = sd.query_hostapis()
    rows = []
    for key in eng._streams:                       # the streams that really opened
        idx = eng._stream_devices[key]
        d = sd.query_devices(idx)
        api = apis[d["hostapi"]]["name"]
        rows.append((key, idx, d["name"].strip(), api, is_bypass_api(api)))
    return rows


def guard_against_concentration(eng: HapticEngine, force: bool) -> None:
    """The whole reason this script exists: never SILENTLY play a fake "all
    speakers" that is really ONE box. Report what the engine actually opened, then
    refuse (unless --force) if the array can't reach both boxes -- either because
    it collapsed to a single device, or because a Vantec is on a folding
    shared-mode API. Each refusal carries the exact remedy."""
    rows = stream_report(eng)
    rig_options = [o for o in eng._options if o["mode"] in ("rig", "aggregate")]

    log.info("output mode=%s; opened stream(s):", eng.mode)
    for key, idx, name, api, bypass in rows:
        log.info("    %-3s idx %2d  %-30s %-22s %s", key, idx, name, api,
                 "DISCRETE (mixer-bypass)" if bypass else "SHARED-MODE (folds!)")

    # ---- PRIMARY guard: only ONE physical device opened -> can't reach both boxes
    if eng.mode in ("single", "bench"):
        if rig_options:
            cause = (
                "Both Vantecs were detected, but no 2-device rig option could open --\n"
                "the two adapters can't stream at the same time on this machine. On the\n"
                "dev box they share ONE USB controller/hub (C-Media CM6206 x2), so the\n"
                "second adapter's isochronous OUT endpoint can't be allocated while the\n"
                "first is streaming -> the engine fell back to a SINGLE box.\n\n"
                "FIX: plug the two Vantecs into ports on DIFFERENT USB controllers (e.g.\n"
                "opposite sides of the chassis, or a separate USB card) -- a passive hub\n"
                "on the SAME controller will NOT help. Re-run; you want  mode=rig  (V1+V2).\n"
                "(macOS: build one Aggregate Device of both in Audio MIDI Setup ->\n"
                "mode=aggregate, all 12 from one handle.)")
        else:
            cause = ("Only ONE >=8-ch USB adapter was detected. Plug in the SECOND Vantec\n"
                     "(check  --list), or pass --v1/--v2 explicitly.")
        one_idx = rows[0][1] if rows else "?"
        msg = (
            "\n" + "=" * 74 + "\n"
            f"CONCENTRATION GUARD: engine collapsed to '{eng.mode}' on a single device\n"
            f"(idx {one_idx}). Logical ch9-12 (the other box's torso zones) are\n"
            "round-robined onto this one adapter -- so playing 'all speakers' would buzz\n"
            "just ONE box. THIS is the bug you asked to avoid.\n\n"
            + cause + "\n\n"
            "Re-run with --force to drive the single connected box anyway (NOT all 12).\n"
            + "=" * 74)
        if force:
            log.warning(msg)
            log.warning(">>> --force: driving the single box anyway (not all 12).")
            return
        log.error(msg)
        raise SystemExit(2)

    # ---- SECONDARY guard: both boxes open, but on a folding shared-mode API
    folded = [r for r in rows if not r[4]]
    if not folded:
        log.info("rig up on a mixer-bypass API -> all 12 channels route to their own "
                 "jacks, both boxes. No concentration, no fold. Good.")
        return

    bad = ", ".join(f"{k}={api}" for k, _, _, api, _ in folded)
    msg = (
        "\n" + "=" * 74 + "\n"
        "FOLD GUARD: a stream opened on a SHARED-MODE host API -> " + bad + "\n"
        "The OS mixer down-mixes that 8-ch stream onto the device's configured layout.\n"
        "If it's Stereo, that Vantec's channels collapse onto its FRONT jack.\n\n"
        "FIX (either one):\n"
        "  (a) Use the WDM-KS endpoints (Kernel Streaming bypasses the mixer; no config\n"
        "      needed). Find their indices with  --list  and pass  --v1/--v2.\n"
        "  (b) Windows Sound -> the USB Sound Device -> Configure -> 7.1 Surround on\n"
        "      BOTH adapters, then re-run.\n\n"
        "Re-run with --force to play anyway (expect a fold onto FRONT jacks).\n"
        + "=" * 74)
    if force:
        log.warning(msg)
        log.warning(">>> --force set: playing into the folding path anyway.")
    else:
        log.error(msg)
        raise SystemExit(2)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Play one LOW note on ALL 12 haptic speakers simultaneously, "
                    "guarding against the Windows shared-mode fold that concentrates "
                    "the array onto one box.")
    p.add_argument("--list", action="store_true",
                   help="list output devices (find your WDM-KS indices) and exit")
    p.add_argument("--freq", type=float, default=80.0,
                   help="LOW drive frequency in Hz (default 80; low-E-ish, within the "
                        "3ohm drivers' strong 80-160 Hz band)")
    p.add_argument("--dur", type=float, default=400.0, help="burst length in ms (default 400)")
    p.add_argument("--intensity", type=int, default=2, choices=[1, 2, 3],
                   help="felt-strength level (default 2)")
    p.add_argument("--amp", type=float, default=None,
                   help="continuous amplitude 0..1 (overrides --intensity)")
    p.add_argument("--count", type=int, default=1,
                   help="number of times to fire the whole array (default 1)")
    p.add_argument("--gap", type=float, default=0.35,
                   help="seconds of silence between repeats (default 0.35)")
    p.add_argument("--lead", type=float, default=20.0,
                   help="ms of lead-in applied equally to every channel so they start "
                        "sample-aligned in the same callback (default 20)")
    p.add_argument("--attack", type=float, default=6.0, help="attack ms")
    p.add_argument("--release", type=float, default=40.0,
                   help="release ms (longer = droney low note; default 40)")
    p.add_argument("--channels",
                   help="comma list to restrict which logical channels fire (default all routable)")
    p.add_argument("--device", help="force ONE output device (single-Vantec / bench)")
    p.add_argument("--v1", help="device index/name for Vantec V1 (ch1-8)")
    p.add_argument("--v2", help="device index/name for Vantec V2 (ch9-12)")
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--force", action="store_true",
                   help="play even if the array can't reach both boxes (single/bench "
                        "collapse, or a folding shared-mode API)")
    p.add_argument("--verbose", action="store_true", help="DEBUG logging")
    a = p.parse_args(argv)

    configure_logging(a.verbose)

    if a.list:
        print("\n  idx  out  host API        name")
        print("  " + "-" * 62)
        for idx, name, nch, api in list_output_devices():
            print(f"  {idx:>3}  {nch:>3}  {api:<14}  {name}")
        print("\nWindows: pick the two WDM-KS 'USB Sound Device' indices for --v1/--v2")
        print("(WDM-KS = mixer-bypass = discrete 12-ch). The 8-ch DirectSound/MME copies")
        print("fold the array onto one box.\n")
        return 0

    plan = load_channel_plan()

    try:
        eng = HapticEngine(samplerate=a.samplerate, v1=a.v1, v2=a.v2, device=a.device)
        eng.start()
    except RigError as e:
        sys.exit(f"could not open the haptic output: {e}")

    try:
        guard_against_concentration(eng, a.force)

        chans = sorted(eng.route)
        if a.channels:
            want = {int(c) for c in a.channels.split(",") if c.strip()}
            chans = [c for c in chans if c in want]
        if not chans:
            sys.exit("no routable channels to play.")

        spec = dict(freq_hz=a.freq, duration_ms=a.dur, waveform="tone",
                    attack_ms=a.attack, release_ms=a.release, delay_ms=a.lead)
        if a.amp is not None:
            spec["amp"] = a.amp
        else:
            spec["intensity"] = a.intensity

        site_of = {e["ch"]: f"{e['box']} {e['site']}" for e in plan}
        log.info("LOW NOTE on ALL %d channels @ %.0f Hz, %.0f ms, %s, lead %.0f ms  (x%d)",
                 len(chans), a.freq, a.dur,
                 f"amp={a.amp}" if a.amp is not None else f"intensity={a.intensity}",
                 a.lead, a.count)
        log.debug("channels: %s", ", ".join(f"{c}->{site_of.get(c, '?')}" for c in chans))

        dur_s = a.dur / 1000.0
        lead_s = a.lead / 1000.0
        for r in range(a.count):
            if a.count > 1:
                log.info("--- burst %d/%d ---", r + 1, a.count)
            # Queue EVERY channel before the next audio callback. Each voice carries
            # the same lead-in delay, so they all start in the same callback ->
            # sample-aligned within a stream, and as tight as two USB clocks allow
            # across V1/V2 (the engine's documented cross-Vantec limit).
            for ch in chans:
                eng.play(ch, **spec)
            time.sleep(lead_s + dur_s + a.gap)

        log.info("done -- fired %d channel(s) x %d burst(s).", len(chans), a.count)
    finally:
        eng.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
