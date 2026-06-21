#!/usr/bin/env python3
"""
Standalone test for features_audio.py (no pytest).

Synthesizes a 48 kHz mono WAV (one 0.6 s decaying 110 Hz / A2 tone then silence),
builds a 1-row events DataFrame against schema.EVENT_COLUMNS, runs the audio
feature stage, and checks the contract: exact column set/order, one row per event,
event_id preserved, key features finite and positive. Also feeds a degenerate
0.01 s window to prove short windows return a row instead of raising.

Run: `python software/ai/analysis/test_features_audio.py`  ->  prints PASS.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
import schema           # noqa: E402
import features_audio    # noqa: E402


SR = 48000
A2_HZ = 110.0           # open A-string (string_num 5, fret 0)
TONE_S = 0.6


def _write_a2_wav(path):
    """0.6 s exponentially-decaying 110 Hz tone followed by 0.4 s of silence."""
    n = int(TONE_S * SR)
    t = np.arange(n) / SR
    env = np.exp(-3.0 * t)                       # clear decay for decay_rate
    tone = (0.5 * env * np.sin(2 * np.pi * A2_HZ * t)).astype(np.float32)
    silence = np.zeros(int(0.4 * SR), dtype=np.float32)
    sf.write(path, np.concatenate([tone, silence]), SR, subtype="PCM_16")


def _event_row(wav_path, event_id, onset_s, dur_s):
    """One events.csv row filling every schema.EVENT_COLUMNS field."""
    row = {c: "" for c in schema.EVENT_COLUMNS}
    row.update({
        "event_id": event_id,
        "run_id": "t", "session_id": "sess", "player_id": "p0",
        "block": 0, "pass": 0, "held_out": False,
        "intended_class": "clean", "intended_placement": "",
        "string_num": 5, "target_fret": 0,
        "finger": 0, "pluck_strength": "med",
        "onset_s": onset_s, "dur_s": dur_s,
        "f0_hz": np.nan, "f0_midi": np.nan,
        "f0_string_est": 5, "f0_fret_est": 0, "label_fret_match": True,
        "audio_peak_dbfs": -6.0, "audio_clipped": False, "audio_silent": False,
        "wav_path": wav_path, "video_path": "", "video_frame_rate": 0.0,
        "guitar_id": "",
    })
    return row


def main():
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a2.wav")        # absolute path
        _write_a2_wav(wav)

        # --- normal event: the full 0.6 s tone window ---
        events_df = pd.DataFrame(
            [_event_row(wav, "t#0", 0.0, 0.6)], columns=schema.EVENT_COLUMNS)

        feats = features_audio.run(events_df)

        expected_cols = [schema.EVENT_ID] + list(schema.AUDIO_FEATURES)
        assert list(feats.columns) == expected_cols, \
            "columns mismatch:\n got %s\n exp %s" % (list(feats.columns), expected_cols)
        assert len(feats) == 1, "expected 1 row, got %d" % len(feats)
        assert feats.iloc[0][schema.EVENT_ID] == "t#0", \
            "event_id not preserved: %r" % feats.iloc[0][schema.EVENT_ID]

        sc = feats.iloc[0]["spec_centroid"]
        rms = feats.iloc[0]["rms"]
        m1 = feats.iloc[0]["mfcc_1"]
        assert np.isfinite(sc) and sc > 0, "spec_centroid not finite>0: %r" % sc
        assert np.isfinite(rms) and rms > 0, "rms not finite>0: %r" % rms
        assert np.isfinite(m1), "mfcc_1 not finite: %r" % m1

        # --- degenerate event: 0.01 s window must return a row, never raise ---
        deg_df = pd.DataFrame(
            [_event_row(wav, "t#1", 0.0, 0.01)], columns=schema.EVENT_COLUMNS)
        deg = features_audio.run(deg_df)
        assert list(deg.columns) == expected_cols, "degenerate columns mismatch"
        assert len(deg) == 1, "degenerate: expected 1 row, got %d" % len(deg)
        assert deg.iloc[0][schema.EVENT_ID] == "t#1", "degenerate event_id lost"

    print("PASS: features_audio")


if __name__ == "__main__":
    main()
