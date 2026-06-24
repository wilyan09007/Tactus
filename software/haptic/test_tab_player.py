#!/usr/bin/env python3
"""
Tests for the tab -> haptic pipeline (tab_player.py). No audio device required:

  * tab loader  -- pure JSON -> Song validation
  * OnsetGate   -- onset/release detection on SYNTHETIC signals
  * TabSequencer-- the state machine, driven against a FakeEngine

Run:  pytest software/haptic/test_tab_player.py
  or: python software/haptic/test_tab_player.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np   # noqa: E402

from encoder import NoteEncoder   # noqa: E402
from tab_player import (   # noqa: E402
    GateConfig, OnsetGate, Song, TabError, TabSequencer, load_tab, run_offline,
)


# ----------------------------------------------------------------- fakes/helpers
class FakeEngine:
    def __init__(self):
        self.plays = []      # (ch, spec)
        self.stopped = []    # vid
        self._vid = 0

    def play(self, ch, **spec):
        self._vid += 1
        self.plays.append((ch, spec))
        return self._vid

    def play_pair(self, a, b, **spec):
        return (self.play(a, **spec), self.play(b, **spec))

    def stop_voice(self, vid):
        self.stopped.append(vid)


def _tab(events, **kw):
    d = {"format": "tactus-tab/v1", "title": "t", "events": events}
    d.update(kw)
    return load_tab(d)


def _two_event_song() -> Song:
    return _tab([
        {"notes": [{"string": 6, "fret": 0}]},                                   # open low-E
        {"notes": [{"string": 5, "fret": 2}, {"string": 4, "fret": 2},
                   {"string": 3, "fret": 1}], "strum_ms": 30},                    # chord
    ])


def _make_burst(sr, freq=196.0, dur=0.25, amp=0.5, attack=0.003, decay_tau=0.06):
    n = int(sr * dur)
    t = np.arange(n) / sr
    env = np.exp(-t / decay_tau)
    a = int(sr * attack)
    if a > 0:
        env[:a] *= np.linspace(0.0, 1.0, a)
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _place(sr, total_s, bursts):
    buf = np.zeros(int(sr * total_s), dtype=np.float32)
    for start_s, b in bursts:
        i = int(start_s * sr)
        seg = b[: max(0, len(buf) - i)]
        buf[i:i + len(seg)] += seg
    return buf


# ----------------------------------------------------------------- tab loader
def test_load_valid_tab():
    song = _two_event_song()
    assert len(song.events) == 2
    assert not song.events[0].is_chord
    assert song.events[1].is_chord and len(song.events[1].notes) == 3
    assert song.events[1].strum_ms == 30


def test_load_rejects_bad_format():
    try:
        load_tab({"format": "nope", "events": []})
        assert False, "bad format should raise"
    except TabError:
        pass


def test_load_rejects_bad_string():
    try:
        _tab([{"notes": [{"string": 7, "fret": 1}]}])
        assert False, "string 7 should raise"
    except TabError:
        pass


def test_out_of_scope_fret_policies():
    raw = [{"notes": [{"string": 1, "fret": 9}]}]
    try:
        load_tab({"format": "tactus-tab/v1", "events": raw})   # default = error
        assert False, "fret 9 should raise by default"
    except TabError:
        pass
    skipped = load_tab({"format": "tactus-tab/v1",
                        "events": [{"notes": [{"string": 1, "fret": 9},
                                              {"string": 1, "fret": 3}]}]},
                       on_out_of_scope="skip")
    assert len(skipped.events[0].notes) == 1 and skipped.events[0].notes[0].fret == 3
    clamped = load_tab({"format": "tactus-tab/v1", "events": raw}, on_out_of_scope="clamp")
    assert clamped.events[0].notes[0].fret == 6


def test_example_file_loads():
    here = os.path.dirname(os.path.abspath(__file__))
    song = load_tab(os.path.join(here, "example_song.json"))
    assert len(song.events) == 6 and song.events[-1].is_chord


# ----------------------------------------------------------------- OnsetGate DSP
def test_single_burst_one_onset_one_release():
    sr = 48000
    sig = _place(sr, 0.6, [(0.05, _make_burst(sr))])
    ev = run_offline(OnsetGate(GateConfig(samplerate=sr)), sig)
    assert sum(k == "onset" for k, _, _ in ev) == 1
    assert sum(k == "release" for k, _, _ in ev) >= 1


def test_two_separated_bursts_two_onsets():
    sr = 48000
    sig = _place(sr, 1.1, [(0.05, _make_burst(sr)), (0.55, _make_burst(sr))])
    ev = run_offline(OnsetGate(GateConfig(samplerate=sr)), sig)
    assert sum(k == "onset" for k, _, _ in ev) == 2


def test_two_close_bursts_collapse_to_one_onset():
    sr = 48000
    # second attack 20 ms after the first -> inside chord_window (40 ms) -> ONE event
    sig = _place(sr, 0.6, [(0.05, _make_burst(sr)), (0.07, _make_burst(sr))])
    ev = run_offline(OnsetGate(GateConfig(samplerate=sr, chord_window_ms=40.0)), sig)
    assert sum(k == "onset" for k, _, _ in ev) == 1


# ----------------------------------------------------------------- sequencer
def test_onset_fires_first_event():
    seq = TabSequencer(_two_event_song(), NoteEncoder(), FakeEngine())
    ok = seq.on_onset(0.8, 0.0)
    assert ok
    fe = seq.eng
    assert [c[0] for c in fe.plays] == [6]          # open low-E -> back ch6 only
    assert seq.next_idx == 1 and seq.active["idx"] == 0


def test_second_onset_stops_prev_and_fires_chord():
    fe = FakeEngine()
    seq = TabSequencer(_two_event_song(), NoteEncoder(), fe)
    seq.on_onset(0.8, 0.0)               # event0 -> vid 1
    seq.on_onset(0.7, 500.0)             # event1 chord; stops event0
    assert fe.stopped == [1]             # the previous voice was killed
    chord_plays = fe.plays[1:]           # after the first (event0) play
    assert [c[0] for c in chord_plays] == [5, 8, 4, 8, 3, 7]
    assert [c[1].get("delay_ms") for c in chord_plays] == [0.0, 0.0, 30.0, 30.0, 60.0, 60.0]
    assert seq.next_idx == 2


def test_release_stops_active():
    fe = FakeEngine()
    seq = TabSequencer(_two_event_song(), NoteEncoder(), fe)
    seq.on_onset(0.8, 0.0)
    seq.on_release(900.0)
    assert fe.stopped == [1]
    assert seq.active is None


def test_sustain_shimmer_is_throttled():
    fe = FakeEngine()
    seq = TabSequencer(_two_event_song(), NoteEncoder(), fe, shimmer_ms=100.0)
    seq.on_onset(0.8, 0.0)
    n_after_onset = len(fe.plays)
    seq.on_sustain(0.7, 50.0)            # < shimmer_ms since onset -> no refire
    assert len(fe.plays) == n_after_onset
    seq.on_sustain(0.7, 120.0)           # >= shimmer_ms -> refire (shimmer)
    assert len(fe.plays) > n_after_onset


def test_velocity_amp_passthrough():
    fe = FakeEngine()
    seq = TabSequencer(_two_event_song(), NoteEncoder(), fe, velocity=True, velocity_gain=1.0)
    seq.on_onset(0.5, 0.0)
    assert abs(fe.plays[0][1]["amp"] - 0.5) < 1e-6


def test_fixed_intensity_when_velocity_off():
    fe = FakeEngine()
    seq = TabSequencer(_two_event_song(), NoteEncoder(), fe, velocity=False, base_intensity=3)
    seq.on_onset(0.5, 0.0)
    assert fe.plays[0][1]["intensity"] == 3
    assert "amp" not in fe.plays[0][1]


def test_song_finishes_after_last_event():
    fe = FakeEngine()
    seq = TabSequencer(_two_event_song(), NoteEncoder(), fe)
    seq.on_onset(0.8, 0.0)
    seq.on_onset(0.8, 500.0)             # fires last event
    assert not seq.on_onset(0.8, 1000.0)  # nothing left -> False
    seq.on_release(1100.0)
    assert seq.finished


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    n = 0
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
        n += 1
    print(f"\n{n}/{len(tests)} tests passed")
