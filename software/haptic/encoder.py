#!/usr/bin/env python3
"""
encoder.py -- deterministic (string, fret) -> haptic speaker mapping.

This is the *renderer* half of Tactus: given a note as (string, fret), fire the
right speaker(s). It makes NO acoustic or ML decision -- WHAT note was played is
decided upstream (truth.md S6); this module only performs the fixed, pointable
transform "(string, fret) -> channel(s)" defined by config/channel_map.json
(truth.md S5: single note = (string -> back channel) + (fret -> torso fret-zone)).

Routing (config/channel_map.json; frets 1-6 only -- the hackathon scope):
  * 6 'string' channels (axis="string", index 1..6): the back string sites,
    index 1 = high-E ... index 6 = low-E (standard guitar string numbering).
  * 6 'fret'   channels (axis="fret",   index 1..6): the torso fret-zones
    z1..z6, mapped 1:1 (fret N -> zone N).

So one played note lights TWO speakers -- a back string site AND a torso
fret-zone site -- fired together (truth.md S5). An open string (fret 0) lights
the string site only.

note_to_channels() is PURE: it reads the channel plan and touches no audio
device, so it is fully unit-testable without hardware (see test_encoder.py). The
play_* helpers take any object with .play()/.play_pair() (the HapticEngine) and
issue the non-blocking bursts; a chord is just several notes, optionally strummed
by staggering each note's delay_ms.
"""
from __future__ import annotations

from pathlib import Path

from rig import CONFIG, configure_logging, get_logger, load_channel_plan

log = get_logger("tactus.haptic.encoder")

OPEN_FRET = 0   # open string -> string site only (no fret-zone)


class NoteEncoder:
    """Deterministic (string, fret) -> channel(s), per config/channel_map.json.

    Usage:
        enc = NoteEncoder()
        enc.note_to_channels(6, 1)        # -> (6, 7)  pure mapping, no audio
        with HapticEngine() as eng:
            enc.play_note(eng, 6, 1)              # low-E, fret 1 -> ch6 + ch7
            enc.play_chord(eng, [(6,1),(5,3),(4,3)], strum_ms=40)  # strummed
    """

    def __init__(self, config_path: Path = CONFIG, plan: list[dict] | None = None):
        self.plan = plan if plan is not None else load_channel_plan(config_path)
        self._string_ch = {e["index"]: e["ch"] for e in self.plan if e["axis"] == "string"}
        self._fret_ch = {e["index"]: e["ch"] for e in self.plan if e["axis"] == "fret"}
        self._site = {e["ch"]: e.get("site", "?") for e in self.plan}
        self.strings = sorted(self._string_ch)
        self.frets = sorted(self._fret_ch)
        if not self._string_ch or not self._fret_ch:
            raise ValueError("channel plan has no string/fret axes; check config/channel_map.json")
        log.debug("encoder ready: strings=%s frets=%s", self.strings, self.frets)

    # ------------------------------------------------------------------ #
    # the deterministic core -- pure, no audio device
    # ------------------------------------------------------------------ #
    def note_to_channels(self, string: int, fret: int) -> tuple[int, int | None]:
        """(string, fret) -> (string_ch, fret_ch).

        fret_ch is None for an open string (fret 0). Raises ValueError if string
        or fret is outside the rig's range.
          string: 1 = high-E .. 6 = low-E (standard guitar numbering)
          fret:   0 = open, 1..6 = fretted (hackathon scope)
        """
        if string not in self._string_ch:
            raise ValueError(f"string {string!r} not in {self.strings} (1=high-E .. 6=low-E)")
        if fret == OPEN_FRET:
            return (self._string_ch[string], None)
        if fret not in self._fret_ch:
            raise ValueError(
                f"fret {fret!r} not in {self.frets} (hackathon scope = frets 1-6; 0=open)")
        return (self._string_ch[string], self._fret_ch[fret])

    def describe(self, string: int, fret: int) -> str:
        """Human-readable mapping for one note (for CLI / logging)."""
        s_ch, f_ch = self.note_to_channels(string, fret)
        s = f"string {string} -> ch{s_ch} ({self._site.get(s_ch)})"
        f = ("open (no fret-zone)" if f_ch is None
             else f"fret {fret} -> ch{f_ch} ({self._site.get(f_ch)})")
        return f"{s}  |  {f}"

    # ------------------------------------------------------------------ #
    # play helpers -- need an engine exposing play()/play_pair()
    # ------------------------------------------------------------------ #
    def play_note(self, engine, string: int, fret: int, intensity: int = 2,
                  **spec) -> tuple[int, ...]:
        """Fire the speaker(s) for one note; returns the engine voice id(s).
        Extra kwargs (amp, freq_hz, duration_ms, waveform, attack_ms, release_ms,
        delay_ms, ...) pass straight through to HapticEngine.play()."""
        s_ch, f_ch = self.note_to_channels(string, fret)
        if f_ch is None:
            return (engine.play(s_ch, intensity=intensity, **spec),)
        return engine.play_pair(s_ch, f_ch, intensity=intensity, **spec)

    def play_chord(self, engine, notes, intensity: int = 2, strum_ms: float = 0.0,
                   **spec) -> list[int]:
        """Fire several notes. `notes`: iterable of (string, fret).
        strum_ms > 0 staggers successive notes (a strum sweep) via the engine's
        delay_ms; strum_ms = 0 is a simultaneous block/bloom. Purely deterministic
        routing -- the sweep order is just the order of `notes`."""
        base_delay = float(spec.pop("delay_ms", 0.0))
        vids: list[int] = []
        for i, (string, fret) in enumerate(notes):
            vids.extend(self.play_note(engine, string, fret, intensity=intensity,
                                       delay_ms=base_delay + i * strum_ms, **spec))
        return vids


# --------------------------------------------------------------------------- #
# CLI: `--map` prints the deterministic table (no audio); --note/--chord play it
# --------------------------------------------------------------------------- #
def _main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Deterministic (string,fret) -> speaker encoder.")
    p.add_argument("--map", action="store_true",
                   help="print the full string x fret -> channel table and exit (NO audio)")
    p.add_argument("--note", help="play one note 'string,fret' (e.g. 6,1) -- needs the rig/engine")
    p.add_argument("--chord", help="play a chord 'string:fret ...' (e.g. 6:1 5:3 4:3) -- needs the rig")
    p.add_argument("--strum-ms", type=float, default=0.0, help="strum stagger per note in ms (chord)")
    p.add_argument("--intensity", type=int, default=2)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    configure_logging(a.verbose)
    enc = NoteEncoder()

    if a.map or (not a.note and not a.chord):
        print("Deterministic (string, fret) -> speaker  "
              "[string 1=high-E .. 6=low-E ; fret 0=open, 1-6]")
        cols = ["open"] + [f"f{f}" for f in enc.frets]
        print("  s\\fret | " + " | ".join(f"{c:>5}" for c in cols))
        print("  " + "-" * (9 + 8 * len(cols)))
        for s in enc.strings:
            cells = []
            for fr in [OPEN_FRET] + enc.frets:
                sc, fc = enc.note_to_channels(s, fr)
                cells.append(f"{sc}/{'-' if fc is None else fc}")
            print(f"   s{s}    | " + " | ".join(f"{c:>5}" for c in cells))
        print("\ncell = string_ch/fret_ch   (e.g. 6/7 = back ch6 + torso ch7; '-' = open)")
        return

    import time
    from engine import HapticEngine   # lazy: only when actually driving the rig

    def _parse(tok):
        s, fr = (int(x) for x in tok.replace(":", ",").split(","))
        return s, fr

    with HapticEngine() as eng:
        if a.note:
            s, fr = _parse(a.note)
            log.info("note: %s", enc.describe(s, fr))
            enc.play_note(eng, s, fr, intensity=a.intensity)
            time.sleep(0.6)
        if a.chord:
            notes = [_parse(t) for t in a.chord.split()]
            for s, fr in notes:
                log.info("chord note: %s", enc.describe(s, fr))
            enc.play_chord(eng, notes, intensity=a.intensity, strum_ms=a.strum_ms)
            time.sleep(1.0 + len(notes) * a.strum_ms / 1000.0)


if __name__ == "__main__":
    _main()
