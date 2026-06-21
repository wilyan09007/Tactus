#!/usr/bin/env python3
"""
Unit tests for the deterministic (string, fret) -> speaker encoder (encoder.py).

Pure routing tests -- NO audio device required. The play_* helpers are checked
against a FakeEngine that records calls instead of making sound.

Run:  pytest software/haptic/test_encoder.py
  or: python software/haptic/test_encoder.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `import encoder/rig` works

from encoder import NoteEncoder, OPEN_FRET   # noqa: E402


class FakeEngine:
    """Records play()/play_pair() calls instead of driving hardware."""

    def __init__(self):
        self.calls = []      # list of (ch, spec)
        self._vid = 0

    def play(self, ch, **spec):
        self._vid += 1
        self.calls.append((ch, spec))
        return self._vid

    def play_pair(self, a, b, **spec):
        return (self.play(a, **spec), self.play(b, **spec))


def _enc():
    return NoteEncoder()


# ----------------------------------------------------------------- mapping core
def test_string_axis_high_e_to_ch1_low_e_to_ch6():
    enc = _enc()
    assert enc.note_to_channels(1, 1)[0] == 1    # high-E -> back ch1
    assert enc.note_to_channels(6, 1)[0] == 6    # low-E  -> back ch6
    assert enc.note_to_channels(3, 1)[0] == 3    # G      -> back ch3


def test_fret_axis_1_to_ch7_6_to_ch12():
    enc = _enc()
    assert enc.note_to_channels(1, 1)[1] == 7    # fret 1 -> torso z1 = ch7
    assert enc.note_to_channels(1, 6)[1] == 12   # fret 6 -> torso z6 = ch12
    assert enc.note_to_channels(1, 3)[1] == 9    # fret 3 -> torso z3 = ch9


def test_canonical_notes():
    enc = _enc()
    assert enc.note_to_channels(6, 1) == (6, 7)     # low-E,  fret 1
    assert enc.note_to_channels(1, 6) == (1, 12)    # high-E, fret 6
    assert enc.note_to_channels(3, 3) == (3, 9)


def test_open_string_has_no_fret_zone():
    enc = _enc()
    assert enc.note_to_channels(2, OPEN_FRET) == (2, None)


def test_out_of_range_raises():
    enc = _enc()
    for bad in (0, 7, -1):
        try:
            enc.note_to_channels(bad, 1)
            assert False, f"string {bad} should raise"
        except ValueError:
            pass
    for bad in (7, 13, -1):
        try:
            enc.note_to_channels(1, bad)
            assert False, f"fret {bad} should raise"
        except ValueError:
            pass


def test_every_string_fret_combo_resolves_uniquely():
    enc = _enc()
    seen = set()
    for s in enc.strings:
        for fr in enc.frets:
            pair = enc.note_to_channels(s, fr)
            assert pair not in seen, f"({s},{fr}) collides with another note: {pair}"
            seen.add(pair)
    assert len(seen) == len(enc.strings) * len(enc.frets)   # 36 distinct (ch,ch)


# ----------------------------------------------------------------- play helpers
def test_play_note_fires_both_sites_with_intensity():
    enc = _enc()
    fe = FakeEngine()
    vids = enc.play_note(fe, 6, 1, intensity=3)
    assert len(vids) == 2
    assert [c[0] for c in fe.calls] == [6, 7]              # string site then fret site
    assert all(c[1].get("intensity") == 3 for c in fe.calls)


def test_play_note_open_string_fires_one_site():
    enc = _enc()
    fe = FakeEngine()
    vids = enc.play_note(fe, 2, 0)
    assert len(vids) == 1
    assert fe.calls[0][0] == 2


def test_play_note_passes_spec_through():
    enc = _enc()
    fe = FakeEngine()
    enc.play_note(fe, 6, 1, amp=0.5, freq_hz=120, duration_ms=80)
    for _, spec in fe.calls:
        assert spec["amp"] == 0.5 and spec["freq_hz"] == 120 and spec["duration_ms"] == 80


def test_play_chord_strum_staggers_delay():
    enc = _enc()
    fe = FakeEngine()
    enc.play_chord(fe, [(6, 1), (5, 3), (4, 3)], strum_ms=40.0)
    # 3 notes x 2 sites = 6 calls; a note's pair shares its delay; delays step by 40
    assert [c[1].get("delay_ms") for c in fe.calls] == [0.0, 0.0, 40.0, 40.0, 80.0, 80.0]
    assert [c[0] for c in fe.calls] == [6, 7, 5, 9, 4, 9]


def test_play_chord_block_is_simultaneous():
    enc = _enc()
    fe = FakeEngine()
    enc.play_chord(fe, [(6, 1), (5, 3)], strum_ms=0.0)
    assert [c[1].get("delay_ms") for c in fe.calls] == [0.0, 0.0, 0.0, 0.0]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    n = 0
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
        n += 1
    print(f"\n{n}/{len(tests)} tests passed")
