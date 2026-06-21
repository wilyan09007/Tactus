#!/usr/bin/env python3
"""
Speaker / jack bring-up test for the Tactus rig.

Two modes:

  1) FRONT-jack L/R test (default): plays ~1s to LEFT then RIGHT of one jack,
     one driver at a time, with a pass/fail diagnosis. Confirms a jack's two
     channels work and are independent (not mono-summed, not swapped).

  2) --sweep: plays a tone on EVERY channel (1..8) one at a time and tells you
     which jack/side each is, so you can MAP which speaker is on which jack.
     Use this when "only one jack plays" -- a stereo source only fills FRONT,
     so the other jacks need a tone forced onto channels 3-8 directly.

Channel -> jack (standard Windows 7.1 order; CM6206 can reorder 3-8, trust ears):
    ch1 FRONT Left      ch2 FRONT Right
    ch3 CENTER          ch4 BASS/Sub(LFE)
    ch5 BACK Left       ch6 BACK Right
    ch7 SURROUND Left   ch8 SURROUND Right

NOTE (Windows): channels 3-8 reach the rear/center/side jacks only if the
shared-mode mixer isn't down-mixing your 8-ch stream back to stereo (which
dumps ch2-8 onto the FRONT jack). Two ways around it:
  (a) pick the WDM-KS endpoint -- "USB Sound Device" under "Windows WDM-KS"
      in --list. Kernel Streaming bypasses the mixer, so NO 7.1 config is
      needed. --device "USB Sound" auto-prefers this endpoint.
  (b) or set the device to 7.1 in the Sound Control Panel
      (Configure -> 7.1 Surround) and use the 8-channel index.
Either way you must FORCE a tone onto one channel (--step/--sweep); a plain
stereo source only ever fills FRONT.

NOTE (macOS): CoreAudio has NO shared-mode stereo fold -- it passes raw
multichannel straight to the USB endpoint, so there's no "7.1" gate and no
WDM-KS equivalent to hunt for; any CoreAudio device just works. Caveats when
porting: (1) set each adapter to its 8-ch format in Audio MIDI Setup (some USB
DACs default to 2-ch); (2) RE-VERIFY channel order by ear (--sweep) -- the
CM6206 reorder (rear=hw5/6, center=hw3/4) is a chip property that should carry
over, but CoreAudio may instead honor the device's reported channel layout;
(3) to drive all 12 from ONE handle, build an Aggregate Device of both adapters
in Audio MIDI Setup and run  --v1 IDX --v2 IDX --v2-base 8.  Device auto-detect,
play_signal, and JACK_TO_HW are all platform-agnostic; only endpoint selection
differs (handled by discrete_output_endpoints()).

Runs on Windows/macOS/Linux via sounddevice (PortAudio).

Usage:
    python speaker_check.py --list                       # find your device index
    python speaker_check.py --device "USB Sound" --step  # Windows: WDM-KS, YOU pace each channel
    python speaker_check.py --device 12                  # FRONT jack L/R test
    python speaker_check.py --device 12 --sweep          # map ALL jacks, one channel at a time
    python speaker_check.py --device 12 --sweep --channels 1,2,5,6
    python speaker_check.py --device 12 --no-interactive # just play, judge by ear
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

# channel_map.json lives at repo_root/config/ ; this file is at repo_root/software/haptic/
CONFIG = Path(__file__).resolve().parents[2] / "config" / "channel_map.json"

# Name fragments a CM6206/Vantec 7.1 adapter tends to enumerate under.
VANTEC_HINTS = ("vantec", "cm6206", "c-media", "cmedia", "usb audio",
                "usb sound", "usb advanced audio", "7.1")

# Standard Windows 8-channel (7.1) order -> this device's jack labels.
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
# ("L"/"R").  The CM6206 actually presents channels in FL,FR,FC,LFE,RL,RR,SL,SR
# order, so a jack's 1-based hardware channel is NOT the contiguous alsa_ch in the
# map -- e.g. the REAR jack is hw 5/6, not 3/4.  This is the empirical order
# confirmed at bring-up (truth.md "ALSA enumeration trap"); drive by THIS, then
# write the confirmed numbers back into channel_map.json's alsa_ch.
JACK_TO_HW = {
    ("front",  "L"): 1, ("front",  "R"): 2,
    ("center", "L"): 3, ("center", "R"): 4,
    ("rear",   "L"): 5, ("rear",   "R"): 6,
    ("side",   "L"): 7, ("side",   "R"): 8,
}


def channel_label(ch: int):
    return JACK_MAP.get(ch, (f"ch{ch}", "", ""))


def load_default_freq() -> float:
    """Pull the pulse frequency from channel_map.json so the test matches the rig."""
    try:
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
        return float(data.get("pulse", {}).get("freq_hz", 160.0))
    except Exception:
        return 160.0


def load_channel_plan() -> list:
    """Load the 12 logical channels from channel_map.json, resolving each to a
    physical (vantec, hw channel) via JACK_TO_HW. Returns a list sorted by ch."""
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    plan = []
    for c in data["channels"]:
        plan.append({
            "ch": c["ch"], "vantec": c["vantec"], "jack": c["jack"], "side": c["side"],
            "hw": JACK_TO_HW.get((c["jack"], c["side"])),
            "box": c.get("box", ""), "site": c.get("site", ""),
        })
    plan.sort(key=lambda e: e["ch"])
    return plan


def list_devices() -> None:
    hostapis = sd.query_hostapis()
    print("\nOutput-capable audio devices:\n")
    print(f"  {'idx':>3}  {'out':>3}  {'host API':<12}  name")
    print("  " + "-" * 64)
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            host = hostapis[d["hostapi"]]["name"]
            print(f"  {i:>3}  {d['max_output_channels']:>3}  {host:<12}  {d['name']}")
    print("\nThe Vantec shows up as 'Speakers (USB Sound Device)'. The 8-channel")
    print("copies (MME/DirectSound) are the ones to use for --sweep.")
    print("Re-run with   --device <idx>   or   --device \"<name substring>\".\n")


def resolve_device(spec) -> int:
    devices = sd.query_devices()

    if spec is not None:
        try:
            idx = int(spec)
            if devices[idx]["max_output_channels"] < 1:
                sys.exit(f"Device {idx} ('{devices[idx]['name']}') has no output channels.")
            return idx
        except ValueError:
            pass  # not an int -> treat as a name substring
        matches = [i for i, d in enumerate(devices)
                   if spec.lower() in d["name"].lower() and d["max_output_channels"] > 0]
        if not matches:
            sys.exit(f"No output device name contains '{spec}'. Run --list to see names.")
        if len(matches) > 1:
            # indices are unstable; prefer the widest endpoint, and among equally
            # wide ones prefer WDM-KS -- Kernel Streaming bypasses the shared-mode
            # mixer that otherwise collapses ch3-8 onto the FRONT jack.
            apis = sd.query_hostapis()
            def _score(i):
                d = devices[i]
                wdmks = "wdm-ks" in apis[d["hostapi"]]["name"].lower()
                return (d["max_output_channels"], 1 if wdmks else 0)
            best = max(matches, key=_score)
            api = apis[devices[best]["hostapi"]]["name"]
            print(f"'{spec}' matched {len(matches)} devices; using "
                  f"{best}: {devices[best]['name'].strip()} "
                  f"({devices[best]['max_output_channels']} ch, {api})")
            return best
        return matches[0]

    cands = [i for i, d in enumerate(devices)
             if d["max_output_channels"] > 0
             and any(h in d["name"].lower() for h in VANTEC_HINTS)]
    if len(cands) == 1:
        print(f"Auto-detected device {cands[0]}: {devices[cands[0]]['name']}")
        return cands[0]
    if not cands:
        list_devices()
        sys.exit("Could not auto-detect the Vantec. Pick one with --device <idx>.")
    print("Several USB-audio-like output devices found:")
    for i in cands:
        print(f"  {i}: {devices[i]['name']}  ({devices[i]['max_output_channels']} ch)")
    sys.exit("Pass the right one with --device <idx>.")


def _discrete_usb_endpoint(d, api_name: str) -> bool:
    """True if device dict `d` is our USB multichannel DAC on the platform path
    that BYPASSES any OS down-mix.

      - Windows: WDM-KS  (Kernel Streaming; shared-mode mixer would fold ch3-8
                 onto FRONT -- the whole reason this script exists).
      - macOS:   Core Audio  (passes raw multichannel straight to the endpoint;
                 no 7.1 gate, no forced stereo fold, so any CoreAudio device works).
      - Linux:   ALSA  (bind the hw device by-id at deploy; truth.md  3.2)."""
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


def discrete_output_endpoints() -> list:
    """Indices of every adapter that can take discrete multichannel output on this
    OS's mixer-bypassing path. Falls back to any >=8-ch USB endpoint if the
    platform-specific filter comes up empty (e.g. an unusual host-API name)."""
    devices = sd.query_devices()
    apis = sd.query_hostapis()
    found = [i for i, d in enumerate(devices)
             if _discrete_usb_endpoint(d, apis[d["hostapi"]]["name"])]
    if not found:
        found = [i for i, d in enumerate(devices)
                 if d["max_output_channels"] >= 8
                 and any(h in d["name"].lower() for h in VANTEC_HINTS)]
    return sorted(set(found))


def resolve_two_adapters(v1_spec, v2_spec):
    """Return (v1_idx, v2_idx) for the two Vantecs. Explicit --v1/--v2 win;
    otherwise auto-pick the two discrete-capable USB endpoints for this OS.

    Two identical adapters can only be told apart by a stable ID:
      - Windows WDM-KS: both enumerate as plain 'USB Sound Device' (the '2-' tag
        shows only under MME/DirectSound) -> distinguish by INDEX.
      - macOS: each has a persistent CoreAudio UID (the by-id analog); pin it in
        Audio MIDI Setup, or pass --v1/--v2.
      - Linux: bind by /dev/snd/by-id (truth.md  3.2).
    Auto-assignment is lower-index -> V1, higher -> V2 == USB enumeration order,
    NOT your wiring; confirm by ear and swap with --v1/--v2 if sites are reversed."""
    if v1_spec is not None and v2_spec is not None:
        return resolve_device(v1_spec), resolve_device(v2_spec)
    eps = discrete_output_endpoints()
    if len(eps) < 2:
        sys.exit(f"Full 12-ch test needs 2 discrete USB adapters; found {eps}. "
                 f"Plug in both Vantecs (check with --list), or pass --v1 <idx> --v2 <idx>.\n"
                 f"  (macOS: set each adapter to its 8-ch format in Audio MIDI Setup, or build\n"
                 f"   ONE Aggregate Device of both and use  --v1 IDX --v2 IDX --v2-base 8.)")
    if len(eps) > 2:
        print(f"  ! {len(eps)} discrete USB adapters {eps}; using first two. Override with --v1/--v2.")
    return eps[0], eps[1]


def make_tone(freq: float, dur: float, sr: int, amp: float, fade_ms: float = 10.0) -> np.ndarray:
    n = int(round(dur * sr))
    t = np.arange(n) / sr
    tone = amp * np.sin(2 * np.pi * freq * t)
    # raised-cosine fade in/out -> no click (clicks sound harsh and stress the coil)
    f = int(round(fade_ms / 1000.0 * sr))
    if f > 0 and 2 * f < n:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(f) / f))
        tone[:f] *= ramp
        tone[-f:] *= ramp[::-1]
    return tone.astype(np.float32)


def play_signal(signal: np.ndarray, ch: int, device, sr: int, maxch: int) -> None:
    """Play `signal` on ONE channel (1-based) inside a FIXED maxch-wide frame.

    Every other channel is zero-filled and we open a full maxch-channel stream
    instead of sd.play(..., mapping=[ch]).  mapping=[ch] opens a stream whose
    width == ch (ch5 -> a 5-ch stream), and Windows' shared-mode mixer then
    up/down-mixes that odd width -- which is exactly how channels 3-8 collapse
    onto the FRONT jack.  A fixed full-width frame on a mixer-bypassing endpoint
    (WDM-KS) routes each channel straight to its own jack.
    """
    buf = np.zeros((len(signal), maxch), dtype=np.float32)
    if 1 <= ch <= maxch:
        buf[:, ch - 1] = signal
    sd.play(buf, samplerate=sr, device=device, blocking=True)


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return ""


def parse_channels(spec, maxch: int):
    if spec:
        return [int(c) for c in spec.split(",") if c.strip()]
    return list(range(1, min(8, maxch) + 1))


def diagnose(label: str, expect: str, answer: str):
    """expect/answer in {l, r, b, n}. Returns (passed, message)."""
    names = {"l": "LEFT only", "r": "RIGHT only", "b": "BOTH", "n": "nothing"}
    got = names.get(answer, f"'{answer}'")
    if answer == expect:
        return True, f"  {label:5}: PASS  (heard {got})"
    if answer == "n":
        return False, (f"  {label:5}: FAIL  (heard nothing) -> dead channel. Check: 3.5mm plug fully "
                       f"seated, amp has 5V, wire landed on that speaker.")
    if answer == "b":
        return False, (f"  {label:5}: FAIL  (heard BOTH) -> not true stereo. Amp mono-summing, or both "
                       f"speakers share one channel. Check L vs R input wiring.")
    if {expect, answer} == {"l", "r"}:
        return False, (f"  {label:5}: SWAPPED (heard {got}, expected {names[expect]}) -> L/R reversed. "
                       f"Swap the two speaker leads (or --left-ch/--right-ch).")
    return False, f"  {label:5}: UNCLEAR (answer '{answer}')."


def run_sweep(args, device, sr, tone, maxch) -> None:
    chans = parse_channels(args.channels, maxch)
    print(f"Sweeping channels {chans}. Listen for which physical speaker/jack buzzes.\n")
    live = []
    for ch in chans:
        jack, side, where = channel_label(ch)
        if ch > maxch:
            print(f"  ch {ch:>2}: SKIP -- device only exposes {maxch} output channels")
            continue
        print(f"  > ch {ch:>2}:  {jack:8} {side:7} ({where})")
        play_signal(tone, ch, device, sr, maxch)
        if args.interactive:
            if ask("        did a speaker play? [y/N]: ") == "y":
                live.append((ch, jack, side))
        time.sleep(args.gap)

    if args.interactive:
        print("\n==== LIVE CHANNELS ====")
        if not live:
            print("  none played. -> The mixer is folding ch3-8 onto FRONT. Either:")
            print("    (a) use the WDM-KS 'USB Sound Device' from --list  (--device \"USB Sound\"),")
            print("        which bypasses the mixer -- no config needed; or")
            print("    (b) Sound Control Panel -> USB Sound Device -> Configure -> 7.1 Surround,")
            print("        then use the 8-channel index (not the 2-ch WASAPI copy).")
        else:
            for ch, jack, side in live:
                print(f"  ch {ch}: {jack} {side}")
            jacks = sorted({j for _, j, _ in live})
            print(f"\n  Jacks with working speakers: {', '.join(jacks)}")
    else:
        print("\nDone sweeping.")


def run_step(args, device, sr, tone, maxch) -> None:
    """One speaker at a time, user-paced: Enter=next, r=repeat, q=quit."""
    chans = [c for c in parse_channels(args.channels, maxch) if c <= maxch]
    print("STEP mode -- one speaker at a time.")
    print("  After each tone:  [Enter] = next speaker   r = repeat   q = quit\n")
    i = 0
    while i < len(chans):
        ch = chans[i]
        jack, side, where = channel_label(ch)
        print(f"  ch {ch}: {jack} {side}  ({where})")
        play_signal(tone, ch, device, sr, maxch)
        cmd = ask("      [Enter]=next  r=repeat  q=quit : ")
        if cmd == "q":
            break
        if cmd == "r":
            continue
        i += 1
    print("\nDone.")


def run_full(args) -> None:
    """Step ALL 12 logical channels across BOTH Vantecs, user-paced.

    Logical ch -> body site comes from channel_map.json; the physical hw channel
    comes from JACK_TO_HW (the CM6206 FL,FR,FC,LFE,RL,RR,SL,SR order). V1 carries
    logical ch1-8, V2 carries ch9-12."""
    plan = load_channel_plan()
    v1, v2 = resolve_two_adapters(args.v1, args.v2)
    dev_of = {"V1": v1, "V2": v2}

    # macOS Aggregate Device: both Vantecs merged into one 16-ch handle, so V2's
    # channels live at hw 9-16. Auto-offset when V1 and V2 resolve to one device.
    v2_base = args.v2_base
    if v1 == v2 and v2_base == 0:
        v2_base = 8
        print(f"  (single aggregate device idx{v1}: V2 hw channels offset by +{v2_base})")

    # one rate both adapters accept (identical CM6206 chips, but verify + fall back)
    sr = args.samplerate
    for dev in (v1, v2):
        try:
            sd.check_output_settings(device=dev, channels=8, samplerate=sr)
        except Exception:
            sr = int(sd.query_devices(v1)["default_samplerate"])
            print(f"  ! {args.samplerate} Hz rejected; using {sr} Hz instead.")
            break
    maxch_of = {d: sd.query_devices(d)["max_output_channels"] for d in (v1, v2)}

    print(f"\nADAPTERS:  V1 = idx {v1}   V2 = idx {v2}   (both WDM-KS, mixer-bypassed)")
    print(f"  Assumed lower index = V1 (all 4 jacks, ch1-8), higher = V2 (2 jacks, ch9-12).")
    print(f"  That's USB enum order, NOT your wiring -- if sites are reversed, re-run with")
    print(f"      --v1 {v2} --v2 {v1}\n")

    if args.channels:
        chans = [int(c) for c in args.channels.split(",") if c.strip()]
    else:
        chans = list(range(1, 13))
    chans = [c for c in chans if 1 <= c <= 12]
    tone = make_tone(args.freq, args.dur, sr, args.amp)

    print("FULL 12-ch step -- one speaker at a time.")
    print("  [Enter] = next   r = repeat   q = quit\n")
    i = 0
    while i < len(chans):
        e = plan[chans[i] - 1]
        dev = dev_of[e["vantec"]]
        hw = e["hw"]
        if hw is not None and e["vantec"] == "V2":
            hw += v2_base
        print(f"  ch {e['ch']:>2}: {e['vantec']} idx{dev}  hw{hw}  "
              f"{e['jack']:6} {e['side']}  ->  {e['box']:3} {e['site']}")
        if hw is None:
            print("        (no hw mapping for this jack/side -- skipped)")
            i += 1
            continue
        play_signal(tone, hw, dev, sr, maxch_of[dev])
        cmd = ask("      [Enter]=next  r=repeat  q=quit : ")
        if cmd == "q":
            break
        if cmd == "r":
            continue
        i += 1
    print("\nDone -- all 12 logical channels walked.")


def run_all(args, device, sr, tone, maxch) -> None:
    """Play the tone on ALL channels simultaneously -- loudest the array gets."""
    data = np.tile(tone.reshape(-1, 1), (1, maxch))
    print(f"Playing {args.freq:g} Hz on ALL {maxch} channels at amp {args.amp:g} for "
          f"{args.dur:g}s  (brief max-drive burst -- don't sustain it).")
    sd.play(data, samplerate=sr, device=device, blocking=True)
    print("Done.")


def run_pulse(args, device, sr, maxch) -> None:
    """Rapid-fire a short pulse train on ONE channel (one speaker)."""
    ch = parse_channels(args.channels, maxch)[0] if args.channels else args.left_ch
    on_n = int(round(args.on * sr))
    gap_n = max(0, int(round(args.interval * sr)) - on_n)
    pulse = make_tone(args.freq, args.on, sr, args.amp, fade_ms=4)
    silence = np.zeros(gap_n, dtype=np.float32)
    train = np.tile(np.concatenate([pulse, silence]), args.count)
    jack, side, _ = channel_label(ch)
    print(f"Rapid-fire ch {ch} ({jack} {side}): {args.count} pulses, "
          f"{args.on * 1000:.0f}ms on / {args.interval * 1000:.0f}ms period @ "
          f"{args.freq:g} Hz, amp {args.amp:g}  (total {len(train) / sr:.2f}s)")
    play_signal(train, ch, device, sr, maxch)
    print("Done.")


def main() -> None:
    default_freq = load_default_freq()
    p = argparse.ArgumentParser(description="Speaker / jack bring-up test.")
    p.add_argument("--list", action="store_true", help="list output devices and exit")
    p.add_argument("--device", help="device index or name substring (e.g. \"USB Sound\")")
    p.add_argument("--full", action="store_true",
                   help="step ALL 12 logical channels across BOTH Vantecs (V1 ch1-8, V2 ch9-12)")
    p.add_argument("--v1", help="device index/name for Vantec V1 (logical ch1-8); auto if omitted")
    p.add_argument("--v2", help="device index/name for Vantec V2 (logical ch9-12); auto if omitted")
    p.add_argument("--v2-base", type=int, default=0,
                   help="hw-channel offset for V2 (use 8 with a macOS Aggregate Device that "
                        "merges both adapters into one 16-ch device: V1=ch1-8, V2=ch9-16)")
    p.add_argument("--sweep", action="store_true",
                   help="play a tone on EACH channel/jack one at a time (1..8) to map them")
    p.add_argument("--step", action="store_true",
                   help="one speaker at a time, YOU control the pace (Enter=next, r=repeat, q=quit)")
    p.add_argument("--all", dest="all_ch", action="store_true",
                   help="play the tone on ALL channels at once (loudest); pair with --freq/--amp")
    p.add_argument("--pulse", action="store_true",
                   help="rapid-fire a pulse train on ONE channel (set --channels N or --left-ch)")
    p.add_argument("--count", type=int, default=20, help="number of pulses for --pulse (default 20)")
    p.add_argument("--interval", type=float, default=0.125,
                   help="seconds between pulse starts for --pulse (default 0.125 = eighth second)")
    p.add_argument("--on", type=float, default=0.05, help="pulse ON time in seconds (default 0.05)")
    p.add_argument("--channels", help="comma-separated 1-based channels to sweep, e.g. 1,2,5,6")
    p.add_argument("--left-ch", type=int, default=1, help="1-based channel for LEFT / front-tip (default 1)")
    p.add_argument("--right-ch", type=int, default=2, help="1-based channel for RIGHT / front-ring (default 2)")
    p.add_argument("--freq", type=float, default=default_freq,
                   help=f"tone Hz (default {default_freq:g}, from channel_map.json)")
    p.add_argument("--dur", type=float, default=1.0, help="seconds per channel (default 1.0)")
    p.add_argument("--amp", type=float, default=0.3,
                   help="amplitude 0..1 (default 0.3; keep <1 to avoid clipping the coil)")
    p.add_argument("--gap", type=float, default=0.6, help="silence between tones (s)")
    p.add_argument("--samplerate", type=int, default=48000, help="sample rate (default 48000)")
    p.add_argument("--repeat", type=int, default=1, help="repeat the L->R sequence N times")
    p.add_argument("--no-interactive", dest="interactive", action="store_false",
                   help="just play; don't ask pass/fail")
    args = p.parse_args()

    if args.list:
        list_devices()
        return

    if not 0.0 < args.amp <= 1.0:
        sys.exit("--amp must be in (0, 1].")

    if args.full:
        run_full(args)
        return

    device = resolve_device(args.device)
    dev = sd.query_devices(device)
    maxch = dev["max_output_channels"]
    sr = args.samplerate

    api_name = sd.query_hostapis()[dev["hostapi"]]["name"]
    print(f"\nDevice : {device}: {dev['name'].strip()}  ({maxch} out ch, {api_name})")
    print(f"Tone   : {args.freq:g} Hz, {args.dur:g}s, amp {args.amp:g}, {sr} Hz")

    if args.sweep or args.step:
        addressed = max(parse_channels(args.channels, maxch))
    elif args.all_ch:
        addressed = maxch
    elif args.pulse:
        addressed = parse_channels(args.channels, maxch)[0] if args.channels else args.left_ch
    else:
        addressed = max(args.left_ch, args.right_ch)
        print(f"Routing: LEFT = ch{args.left_ch} (FRONT tip)    RIGHT = ch{args.right_ch} (FRONT ring)")
    print()

    # we always open a full maxch-wide stream now (play_signal), so verify that
    # width opens at this rate; fall back to the device's native rate if not
    try:
        sd.check_output_settings(device=device, channels=maxch, samplerate=sr)
    except Exception as e:
        alt = int(dev["default_samplerate"])
        print(f"  ! Device rejected {sr} Hz ({e}); using its default {alt} Hz instead.")
        sr = alt

    if args.pulse:
        run_pulse(args, device, sr, maxch)
        return

    tone = make_tone(args.freq, args.dur, sr, args.amp)

    if args.all_ch:
        run_all(args, device, sr, tone, maxch)
        return

    if args.step:
        run_step(args, device, sr, tone, maxch)
        return

    if args.sweep:
        run_sweep(args, device, sr, tone, maxch)
        return

    results = []
    for r in range(args.repeat):
        if args.repeat > 1:
            print(f"--- pass {r + 1}/{args.repeat} ---")
        for label, ch, expect in (("LEFT", args.left_ch, "l"), ("RIGHT", args.right_ch, "r")):
            print(f"  > playing {label}  (channel {ch}) ...")
            play_signal(tone, ch, device, sr, maxch)
            if args.interactive:
                a = ask("    Which speaker made sound?  [l]eft only / [r]ight only / [b]oth / [n]one : ")
                ok, msg = diagnose(label, expect, a)
                results.append((ok, msg))
            time.sleep(args.gap)

    if args.interactive and results:
        print("\n==== RESULT ====")
        for _, msg in results:
            print(msg)
        passed = all(ok for ok, _ in results)
        print("\n  ==> " + ("ALL PASS -- both channels work and are independent.  OK"
                            if passed else "FAIL -- see the notes above.  X"))
        sys.exit(0 if passed else 1)
    print("\nDone. (no-interactive mode -- you judged it by ear.)")


if __name__ == "__main__":
    main()
