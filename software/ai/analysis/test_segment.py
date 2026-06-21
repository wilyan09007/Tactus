#!/usr/bin/env python3
"""
Plain-assert test for segment.py (no pytest — not installed).

Synthesizes a 3-note plucked WAV (open low-E E2 + frets 1,2 = F2, F#2), writes a
one-line manifest.jsonl pointing at it, runs segment.run(), and checks the
emitted events DataFrame against the schema contract.

Run:  .venv/bin/python software/ai/analysis/test_segment.py
"""
import os
import sys
import json
import tempfile

# Flat imports: put the analysis dir (this file's dir) on sys.path, then import
# the frozen contract and the module under test.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema      # noqa: E402
import segment     # noqa: E402

import numpy as np         # noqa: E402
import soundfile as sf     # noqa: E402


SR = 48000


def _pluck(freq, tone_s=0.5, sil_s=0.4, sr=SR, decay=6.0):
    """A decaying sinusoid ('pluck') followed by silence. The envelope is tapered
    to zero at the end of the tone so there is no hard amplitude step into the
    silence — a real plucked note rings down smoothly, and a hard step would
    manufacture a spurious onset that no real recording has."""
    n_tone = int(tone_s * sr)
    t = np.arange(n_tone) / sr
    env = np.exp(-decay * t)
    taper = int(0.01 * sr)  # 10 ms fade-out kills the residual step
    if 0 < taper < n_tone:
        env[-taper:] *= np.linspace(1.0, 0.0, taper)
    tone = (0.6 * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sil = np.zeros(int(sil_s * sr), dtype=np.float32)
    return np.concatenate([tone, sil])


def _make_wav(path):
    # E2=82.41, F2=87.31, F#2=92.50  (low-E frets 0,1,2)
    freqs = [82.41, 87.31, 92.50]
    sig = np.concatenate([_pluck(f) for f in freqs]).astype(np.float32)
    sf.write(path, sig, SR, subtype="FLOAT")
    return len(freqs)


def _make_manifest(path, wav_abs):
    row = {
        "run_id": "s6_0to2_clean_plkmedium_p01_001",
        "session_id": "testsess",
        "player_id": "p01",
        "app_version": "test",
        "block": "B1",
        "string": "6 (low E)",
        "s_num": 6,
        "fret_range": "0->2",
        "frets": [0, 1, 2],
        "finger": "index",
        "intended_class": "clean",
        "intended_placement": "on-wire",
        "pluck_strength": "medium",
        "pluck_variant": None,
        "pose_variant": None,
        "chord_name": None,
        "is_arpeggio": False,
        "is_strum": False,
        "pass": 1,
        "held_out": False,
        "expected_note_count": 3,
        "matched_intent": None,
        "bpm": 50,
        "beat_times_ms": [],
        "audio": {"sample_rate": SR, "channels": 1, "format": "wav-pcm16",
                  "peak_dbfs": -12, "clipped": False, "silent": False},
        "video": {"present": False, "width": None, "height": None,
                  "frame_rate": 30, "mime": None},
        "files": {"audio": wav_abs, "audio_bytes": 0, "video": None, "video_bytes": 0},
    }
    with open(path, "w") as fh:
        fh.write(json.dumps(row) + "\n")


def test_segment():
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "run001.wav")
        manifest = os.path.join(td, "manifest.jsonl")
        n_notes = _make_wav(wav)
        _make_manifest(manifest, wav)

        df = segment.run(manifest)

        # --- returns a DataFrame ---
        import pandas as pd
        assert isinstance(df, pd.DataFrame), "run() must return a DataFrame"

        # --- columns == schema.EVENT_COLUMNS exactly, in order ---
        assert list(df.columns) == list(schema.EVENT_COLUMNS), (
            "columns mismatch:\n  got: %s\n  exp: %s"
            % (list(df.columns), list(schema.EVENT_COLUMNS))
        )

        # --- finds ~3 events (onset detection is fuzzy; allow 2..4) ---
        n = len(df)
        assert 2 <= n <= 4, "expected ~3 events (2..4), got %d" % n

        # --- onset_s strictly increasing ---
        onsets = df["onset_s"].tolist()
        assert all(onsets[i] < onsets[i + 1] for i in range(len(onsets) - 1)), (
            "onset_s not strictly increasing: %s" % onsets
        )

        # --- dur_s non-negative ---
        assert (df["dur_s"] >= 0).all(), "dur_s must be non-negative"

        # --- target_fret values come from the prompted frets [0,1,2] (or -1 overflow) ---
        allowed = {0, 1, 2, -1}
        tf = set(int(x) for x in df["target_fret"].tolist())
        assert tf.issubset(allowed), "target_fret %s not subset of %s" % (tf, allowed)
        # The first few detected events should map onto real frets (not all -1).
        assert any(int(x) in (0, 1, 2) for x in df["target_fret"].tolist()), (
            "no event mapped to a prompted fret"
        )

        # --- string_num carried from the run (s_num=6) ---
        assert set(int(x) for x in df["string_num"].tolist()) == {6}, (
            "string_num should all be 6, got %s" % df["string_num"].tolist()
        )

        # --- at least one event has a finite f0_hz ---
        f0 = df["f0_hz"].to_numpy(dtype=float)
        assert np.isfinite(f0).any(), "no event produced a finite f0_hz"

        # --- label column present and populated from the manifest ---
        assert schema.LABEL in df.columns, "label column missing"
        assert (df[schema.LABEL] == "clean").all(), (
            "label should be 'clean', got %s" % df[schema.LABEL].unique().tolist()
        )

        # --- event_id is '<run_id>#<k>' ---
        assert df["event_id"].iloc[0].endswith("#0"), (
            "event_id format unexpected: %r" % df["event_id"].iloc[0]
        )

        return n


if __name__ == "__main__":
    n = test_segment()
    print("PASS: segment  (%d events from 3-note synthetic WAV)" % n)
