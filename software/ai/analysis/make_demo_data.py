#!/usr/bin/env python3
"""
Synthetic capture batch for smoke-testing the analysis pipeline before any real
data exists. Writes a data/raw-shaped tree (manifest.jsonl + audio/*.wav) for N
players across a small string x fret x class grid.

The audio model is deliberately simple but designed to reproduce the V1 baseline:
  - clean           = decaying harmonic stack (low buzz-band energy, high HNR)
  - buzz-light      = clean + broadband high-freq noise burst
  - buzz-placement  = clean + the SAME broadband noise burst (audio-identical to
                      buzz-light by construction) -> audio cannot tell the two
                      buzz causes apart. The real disambiguator (vision `d`) needs
                      video, which a synthetic batch has none of, so vision stays
                      NaN here. (The fused proof lives in test_collapse.py.)

Run standalone to drop a demo batch and render an audit you can open:
    python3 software/ai/analysis/make_demo_data.py --raw-dir /tmp/tactus_demo/raw
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema   # noqa: E402

SR = 48000


def _hz(string_num, fret):
    midi = schema.OPEN_STRING_MIDI[string_num] + fret
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _note(freq, buzz=0.0, muted=False, seed=0, dur=0.7, sr=SR):
    """One plucked note: decaying harmonic stack, optional broadband buzz burst, OR
    a dead muted thud; then trailing silence so onset segmentation sees one event."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    if muted:
        # dead/muted: fast decay, fundamental only, quiet -> low energy + low HNR;
        # distinct from clean (rich harmonics) and buzz (broadband rattle).
        env = np.exp(-26.0 * t)
        sig = (0.25 * np.sin(2 * np.pi * freq * t + rng.uniform(0, 0.2)) * env).astype(np.float32) * 0.7
        return np.concatenate([sig, np.zeros(int(0.3 * sr), dtype=np.float32)])
    env = np.exp(-4.0 * t)
    sig = np.zeros_like(t)
    for k in range(1, 7):
        sig += (1.0 / k) * np.sin(2 * np.pi * freq * k * t + rng.uniform(0, 0.2))
    sig *= env
    if buzz > 0:
        sos = butter(4, [3000, 8000], btype="band", fs=sr, output="sos")
        nb = sosfilt(sos, rng.standard_normal(t.size)) * env * buzz
        sig = sig + nb
    peak = np.max(np.abs(sig)) or 1.0
    sig = (sig / peak) * 0.5
    sig[-int(0.05 * sr):] *= np.linspace(1, 0, int(0.05 * sr))   # taper tail
    sil = np.zeros(int(0.3 * sr), dtype=np.float32)
    return np.concatenate([sig.astype(np.float32), sil])


def build(raw_dir, players=("p1", "p2"), session="demo-session",
          strings=(6, 4), frets=(1, 3, 5), reps=3, seed0=0):
    """Write a synthetic batch under raw_dir. Returns the list of player dirs."""
    BUZZ_LEVEL = 0.28   # broadband rattle amount for the 'buzz' class
    made = []
    s = seed0
    for pi, player in enumerate(players):
        pdir = os.path.join(raw_dir, session, player)
        adir = os.path.join(pdir, "audio")
        os.makedirs(adir, exist_ok=True)
        rows = []
        gain = 1.0 + 0.05 * pi            # small per-player level offset
        detune = 1.0 + 0.0008 * pi        # ~1.4 cents per-player pitch offset
        n = 0
        for string_num in strings:
            for fret in frets:
                for cls in schema.CORE_CLASSES:
                    for r in range(reps):
                        s += 1
                        n += 1
                        freq = _hz(string_num, fret) * detune
                        wav = _note(freq, buzz=(BUZZ_LEVEL if cls == "buzz" else 0.0), muted=(cls == "muted"), seed=s) * gain
                        wav = np.clip(wav, -1.0, 1.0)
                        run_id = "s%d_f%d_%s_%s_%03d" % (string_num, fret,
                                                         cls.replace("-", ""), player, n)
                        wpath = os.path.join(adir, run_id + ".wav")
                        sf.write(wpath, wav, SR, subtype="PCM_16")
                        peak_db = 20 * np.log10(np.max(np.abs(wav)) or 1e-9)
                        rows.append({
                            schema.M_RUN: run_id, schema.M_SESSION: session,
                            schema.M_PLAYER: player, "app_version": "demo",
                            schema.M_BLOCK: "core-grid", schema.M_STRING: "%d" % string_num,
                            schema.M_STRING_NUM: string_num, schema.M_FRET_RANGE: str(fret),
                            schema.M_FRETS: [fret], schema.M_FINGER: "index",
                            schema.M_LABEL: cls, schema.M_PLACEMENT:
                                "on-wire" if cls != "buzz-placement" else "too-far-back",
                            schema.M_PLUCK: "medium", schema.M_EXPECTED_N: 1,
                            schema.M_MATCHED: "y", schema.M_PASS: r + 1,
                            schema.M_HELDOUT: False, schema.M_BPM: 50,
                            schema.M_BEATS: [0.0], "recorded_at": "2026-06-20T00:00:00Z",
                            schema.M_AUDIO: {"sample_rate": SR, "channels": 1,
                                             "format": "wav-pcm16", "peak_dbfs": float(peak_db),
                                             "clipped": bool(peak_db >= -1.0),
                                             "clip_samples": 0, "silent": False},
                            schema.M_VIDEO: {"present": False, "width": None, "height": None,
                                             "frame_rate": None, "mime": None},
                            "devices": {"mic_label": "demo", "cam_label": None},
                            schema.M_FILES: {"audio": os.path.abspath(wpath), "audio_bytes": 0,
                                             "video": None, "video_bytes": 0},
                        })
        with open(os.path.join(pdir, "manifest.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        made.append(pdir)
        print("demo: %s -> %d runs" % (player, len(rows)))
    return made


def main():
    ap = argparse.ArgumentParser(description="write a synthetic Tactus capture batch")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--players", default="p1,p2")
    args = ap.parse_args()
    build(args.raw_dir, players=tuple(args.players.split(",")))


if __name__ == "__main__":
    main()
