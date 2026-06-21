#!/usr/bin/env python3
"""
resonance_check.py -- one-speaker-at-a-time LOW / LOUD resonance walk of ALL 12.

A focused bring-up interface: it steps every one of the 12 logical actuators
(across BOTH Vantecs -- V1 ch1-8, V2 ch9-12), one speaker at a time, and plays a
LOW, LOUD tone so you feel each driver's resonance. Not 8-on-one-jack: the full
12.  This is the dedicated counterpart to `speaker_check.py --full`, tuned for a
"does this driver actually move / buzz at its resonance" feel-test.

Controls (single keypress, no Enter needed where the terminal allows it):
    [Enter]  -> next speaker
    r / R    -> REPEAT the current speaker
    q / Esc  -> quit

Why low + loud: the SK473 KHD 3 ohm drivers are heavy; punch comes from driving
near mechanical resonance, not from overdrive (truth.md 3.2 -- "clipping cooks
the coil"). The tuning knob is a 60-250 Hz drive sweep, so:

    default     : sustained 80 Hz, amp 0.8, 1.5 s  (a low, loud single tone)
    --glide      : sweep 60 -> 250 Hz on each channel so the resonant peak --
                  the frequency where the driver gets loudest -- reveals itself

Routing comes from config/channel_map.json + the CM6206 FL,FR,FC,LFE,RL,RR,SL,SR
hardware order; both are reused from speaker_check.py so there's one source of
truth for the rig.

Usage (Windows / cmd):
    python resonance_check.py                 # walk all 12, low/loud sustained tone
    python resonance_check.py --glide          # sweep 60->250 Hz per speaker
    python resonance_check.py --freq 100       # different sustained low tone
    python resonance_check.py --amp 0.9        # louder (kept <1 to spare the coil)
    python resonance_check.py --channels 9,10,11,12   # just the V2 torso speakers
    python resonance_check.py --v1 12 --v2 13  # pin adapters if auto-order is wrong

Usage (macOS / Terminal):  same script, just `python3` and a Unix path --
    cd <repo>/software/haptic && python3 resonance_check.py
    cd <repo>/software/haptic && python3 resonance_check.py --glide
  CoreAudio passes raw multichannel straight through (no WDM-KS / 7.1 gate), but
  each USB adapter defaults to 2-ch -- set BOTH to their 8-ch format in
  Audio MIDI Setup first, or the auto-detect won't see two discrete endpoints.
  To drive all 12 from one handle, build an Aggregate Device of both Vantecs and
  run  python3 resonance_check.py --v1 IDX --v2 IDX  (V2 hw auto-offsets by +8).
  Run `python3 speaker_check.py --list` to get the device indices.

Needs both Vantecs plugged in (only 8 channels exist per adapter; 12 needs two).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import sounddevice as sd

# Reuse the rig logic from the bring-up script next door: channel_map.json ->
# (vantec, hw channel) resolution, dual-adapter discovery, and the per-channel
# player that opens a full-width mixer-bypassing frame. One source of truth.
from speaker_check import (
    load_channel_plan,
    resolve_two_adapters,
    play_signal,
    make_tone,
)


def getkey() -> str:
    """Read ONE keypress, no Enter required, returned lowercased.

    Enter comes back as '\\r'. Falls back to a line read when stdin isn't a real
    TTY (piped input, some IDE consoles) -- there Enter still = next and you type
    'r'/'q' then Enter."""
    try:
        if sys.platform.startswith("win"):
            import msvcrt
            ch = msvcrt.getwch()
            # arrow/function keys arrive as a 2-char sequence led by \x00 / \xe0;
            # swallow the second byte so it isn't read as a command.
            if ch in ("\x00", "\xe0"):
                msvcrt.getwch()
                return ""
            return ch.lower()
        import termios, tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch.lower()
    except Exception:
        line = sys.stdin.readline()
        return (line[:1] or "\r").lower()


def make_glide(f0: float, f1: float, dur: float, sr: int, amp: float,
               fade_ms: float = 12.0) -> np.ndarray:
    """A linear-frequency chirp f0 -> f1 over `dur` s. Driving through the band
    makes the driver's resonant peak audible/palpable as the loudest moment."""
    n = int(round(dur * sr))
    t = np.arange(n) / sr
    # instantaneous phase of a linear sweep: 2*pi*(f0*t + (f1-f0)*t^2/(2*dur))
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * dur))
    tone = amp * np.sin(phase)
    f = int(round(fade_ms / 1000.0 * sr))
    if f > 0 and 2 * f < n:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(f) / f))
        tone[:f] *= ramp
        tone[-f:] *= ramp[::-1]
    return tone.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Low/loud resonance walk of all 12 speakers, one at a time.")
    p.add_argument("--glide", action="store_true",
                   help="sweep --f0 -> --f1 on each speaker to find its resonance "
                        "(default plays a steady --freq tone instead)")
    p.add_argument("--freq", type=float, default=80.0,
                   help="sustained tone Hz (default 80 -- low; try 60-160)")
    p.add_argument("--f0", type=float, default=60.0, help="--glide start Hz (default 60)")
    p.add_argument("--f1", type=float, default=250.0, help="--glide end Hz (default 250)")
    p.add_argument("--amp", type=float, default=0.8,
                   help="amplitude 0..1 (default 0.8 -- loud; kept <1 to spare the coil)")
    p.add_argument("--dur", type=float, default=1.5, help="seconds per speaker (default 1.5)")
    p.add_argument("--channels", help="comma-separated logical channels 1..12 (default all)")
    p.add_argument("--v1", help="device index/name for Vantec V1 (ch1-8); auto if omitted")
    p.add_argument("--v2", help="device index/name for Vantec V2 (ch9-12); auto if omitted")
    p.add_argument("--v2-base", type=int, default=0,
                   help="hw-channel offset for V2 (use 8 for a macOS Aggregate Device "
                        "merging both adapters into one 16-ch handle)")
    p.add_argument("--samplerate", type=int, default=48000, help="sample rate (default 48000)")
    args = p.parse_args()

    if not 0.0 < args.amp <= 1.0:
        sys.exit("--amp must be in (0, 1].")

    plan = load_channel_plan()
    v1, v2 = resolve_two_adapters(args.v1, args.v2)
    dev_of = {"V1": v1, "V2": v2}

    # macOS Aggregate Device: both Vantecs merged -> V2's channels live at hw 9-16.
    v2_base = args.v2_base
    if v1 == v2 and v2_base == 0:
        v2_base = 8
        print(f"  (single aggregate device idx{v1}: V2 hw channels offset by +{v2_base})")

    # one rate both adapters accept (identical CM6206 chips; verify + fall back)
    sr = args.samplerate
    for dev in (v1, v2):
        try:
            sd.check_output_settings(device=dev, channels=8, samplerate=sr)
        except Exception:
            sr = int(sd.query_devices(v1)["default_samplerate"])
            print(f"  ! {args.samplerate} Hz rejected; using {sr} Hz instead.")
            break
    maxch_of = {d: sd.query_devices(d)["max_output_channels"] for d in (v1, v2)}

    if args.channels:
        chans = [int(c) for c in args.channels.split(",") if c.strip()]
    else:
        chans = list(range(1, 13))
    chans = [c for c in chans if 1 <= c <= 12]

    # one waveform, reused for every channel (only the routing changes)
    if args.glide:
        sig = make_glide(args.f0, args.f1, args.dur, sr, args.amp)
        tone_desc = f"glide {args.f0:g} -> {args.f1:g} Hz"
    else:
        sig = make_tone(args.freq, args.dur, sr, args.amp)
        tone_desc = f"steady {args.freq:g} Hz"

    print(f"\nADAPTERS : V1 = idx {v1}   V2 = idx {v2}   (mixer-bypassed)")
    print(f"           lower index assumed = V1 (ch1-8); higher = V2 (ch9-12).")
    print(f"           if the sites come out swapped, re-run with  --v1 {v2} --v2 {v1}")
    print(f"TONE     : {tone_desc}, amp {args.amp:g}, {args.dur:g}s, {sr} Hz")
    print(f"WALKING  : {len(chans)} speaker(s) -> {chans}\n")
    print("RESONANCE WALK -- one speaker at a time.")
    print("  [Enter] = next     r = repeat     q = quit\n")

    i = 0
    while i < len(chans):
        e = plan[chans[i] - 1]
        dev = dev_of[e["vantec"]]
        hw = e["hw"]
        if hw is not None and e["vantec"] == "V2":
            hw += v2_base
        print(f"  [{i + 1:>2}/{len(chans)}]  ch {e['ch']:>2}  {e['vantec']} idx{dev} hw{hw}  "
              f"{e['jack']:6} {e['side']}  ->  {e['box']:3} {e['site']}")
        if hw is None:
            print("           (no hw mapping for this jack/side -- skipped)")
            i += 1
            continue
        play_signal(sig, hw, dev, sr, maxch_of[dev])
        print("           [Enter]=next  r=repeat  q=quit : ", end="", flush=True)
        key = getkey()
        print()  # finish the prompt line after the keypress
        if key in ("q", "\x1b", "\x03"):  # q, Esc, Ctrl-C
            print("\nStopped.")
            return
        if key == "r":
            continue  # replay the SAME speaker
        i += 1  # Enter (or anything else) advances

    print("\nDone -- all 12 logical channels walked.")


if __name__ == "__main__":
    main()
