#!/usr/bin/env python3
"""
Tactus offline-analysis pipeline — stage 1: segment.py

Onset segmentation + label attachment + F0 cross-check.

Reads one capture manifest.jsonl (one JSON run-row per line, schema authored by
software/ai/capture/capture.html buildRow()), segments each run's WAV into note
events with librosa onset detection, attaches the run's prompted labels to every
event ("the prompt IS the label", docs/24 §The-Prompt-IS-the-Label), and measures
each event's F0 to cross-check the prompted fret (label integrity, docs/24 §Onset
Segmentation + F0 Cross-Check).

Emits a pandas.DataFrame whose columns are exactly schema.EVENT_COLUMNS — the
frozen contract the feature stages (features_audio.py / features_vision.py) read.

Conventions match the rest of the pipeline: stdlib paths from __file__, flat
imports (this module lives in the analysis dir, so `import schema` works when run
from there; schema.on_path() also puts the dir on sys.path defensively).
"""
import os
import sys

# Flat imports, no package install (matches software/ai/capture). This module is
# IN the analysis dir, so `import schema` resolves once the dir is on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema  # noqa: E402  (frozen contract — imported, never edited)

schema.on_path()  # analysis dir + vision dir on sys.path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# ----------------------------------------------------------------- small helpers
def _get(d, key, default=None):
    """dict.get that treats a non-dict (e.g. None) as empty — nested manifest
    fields may be missing or null."""
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def _as_int(v, default=0):
    try:
        if v is None:
            return default
        return int(v)
    except (TypeError, ValueError):
        return default


def _string_num_for_row(row):
    """Resolve the prompted string number (6..1; 0 = chord/free).

    Authoritative field is schema.M_STRING_NUM ("s_num"). Some manifest writers
    omit it from the saved row (capture.html buildRow() re-emits `string` but not
    `s_num`); fall back to parsing a leading int out of `string` (e.g. "6 (low E)"),
    else 0."""
    s = _get(row, schema.M_STRING_NUM)
    if s is not None:
        return _as_int(s, 0)
    label = _get(row, schema.M_STRING)  # e.g. "6 (low E)", "chord", "free"
    if isinstance(label, str):
        head = label.strip().split()[0] if label.strip() else ""
        if head.lstrip("-").isdigit():
            return _as_int(head, 0)
    return 0


# ----------------------------------------------------------------- audio / onset
def _load_audio(wav_path):
    """Load mono audio at native sample rate; fall back to librosa's default sr if
    sr=None fails. Returns (y, sr) or (None, None) on failure."""
    import librosa
    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        if y is None or len(y) == 0:
            return None, None
        return y, int(sr)
    except Exception:
        try:
            y, sr = librosa.load(wav_path, mono=True)
            if y is None or len(y) == 0:
                return None, None
            return y, int(sr)
        except Exception:
            return None, None


# If two onsets land closer than this, treat them as one (backtracking + transient
# wobble can split a single attack). ~60 ms is well under a plucked-note duration.
_MIN_ONSET_GAP_S = 0.06
# Window at the head of the file used to decide whether the first note starts at t=0.
_HEAD_WIN_S = 0.05


def _detect_onsets(y, sr):
    """Onset times (seconds), backtracked to the nearest preceding energy minimum.

    Two robustness fixes over a raw onset_detect call, both matching how capture
    runs actually look (docs/24: a run is a short take that begins on the first
    pluck):
      * librosa.onset_detect does not report an onset inside the first analysis
        frame, so a note that starts at the very top of the file is dropped —
        which would mis-shift the whole frets[]->onset assignment. If the file
        starts with energy and no onset was placed near 0, prepend t=0.
      * collapse onsets closer than _MIN_ONSET_GAP_S (split-attack artifacts)."""
    import librosa
    try:
        onsets = librosa.onset.onset_detect(
            y=y, sr=sr, backtrack=True, units="time"
        )
    except Exception:
        onsets = np.asarray([], dtype=float)
    onsets = np.asarray(onsets, dtype=float)
    if onsets.size:
        onsets = np.unique(onsets)
        onsets = onsets[onsets >= 0.0]

    # Prepend t=0 when the recording starts mid-note (first attack at file head).
    if len(y):
        head = y[: int(_HEAD_WIN_S * sr)]
        head_rms = float(np.sqrt(np.mean(head ** 2))) if head.size else 0.0
        full_rms = float(np.sqrt(np.mean(y ** 2)))
        starts_loud = full_rms > 0.0 and head_rms > 0.2 * full_rms
        if starts_loud and (onsets.size == 0 or onsets[0] > _MIN_ONSET_GAP_S):
            onsets = np.insert(onsets, 0, 0.0)

    # Collapse near-duplicate onsets (keep the earliest of each cluster).
    if onsets.size:
        kept = [float(onsets[0])]
        for x in onsets[1:]:
            if float(x) - kept[-1] >= _MIN_ONSET_GAP_S:
                kept.append(float(x))
        onsets = np.asarray(kept, dtype=float)
    return onsets


def _windows_from_onsets(onsets, total_dur):
    """(onset_s, dur_s) per event; last event runs to end-of-file."""
    spans = []
    n = len(onsets)
    for k in range(n):
        start = float(onsets[k])
        end = float(onsets[k + 1]) if k + 1 < n else float(total_dur)
        dur = max(0.0, end - start)
        spans.append((start, dur))
    return spans


def _f0_median_hz(y, sr, start_s, dur_s):
    """Median voiced F0 (Hz) over an event window via librosa.pyin. NaN if unvoiced
    or the window is too short to analyze."""
    import librosa
    if dur_s <= 0:
        return float("nan")
    i0 = max(0, int(round(start_s * sr)))
    i1 = min(len(y), int(round((start_s + dur_s) * sr)))
    seg = y[i0:i1]
    # pyin needs at least a couple of frames; guard tiny slivers.
    if seg.size < int(0.05 * sr):
        return float("nan")
    try:
        fmin = float(librosa.note_to_hz("E1"))   # ~41 Hz, below low-E (E2 ~82)
        fmax = float(librosa.note_to_hz("E6"))
        # frame_length must hold >=2 periods of fmin (41 Hz @ 48 kHz -> ~2331);
        # 4096 covers the whole fret-1..6 range and silences pyin's low-fmin warn.
        f0, voiced_flag, _ = librosa.pyin(
            seg, sr=sr, fmin=fmin, fmax=fmax, frame_length=4096
        )
    except Exception:
        return float("nan")
    if f0 is None:
        return float("nan")
    f0 = np.asarray(f0, dtype=float)
    voiced = f0[np.isfinite(f0)]
    if voiced.size == 0:
        return float("nan")
    return float(np.median(voiced))


# ----------------------------------------------------------------- per-run
def _events_for_row(row, guitar_id):
    """Segment one manifest run-row into a list of event dicts (schema.EVENT_COLUMNS
    keys). Returns (events, status) where status is one of:
    'ok', 'no_audio', 'silent', 'missing_file', 'load_failed', 'no_onsets'."""
    files = _get(row, schema.M_FILES, {}) or {}
    audio_meta = _get(row, schema.M_AUDIO, {}) or {}
    video_meta = _get(row, schema.M_VIDEO, {}) or {}

    wav_rel = _get(files, "audio")
    wav_path = schema.abspath(wav_rel)

    # Skip-but-count: no audio file declared, or run flagged silent.
    if not wav_path:
        return [], "no_audio"
    if bool(_get(audio_meta, "silent", False)):
        return [], "silent"
    if not os.path.exists(wav_path):
        return [], "missing_file"

    y, sr = _load_audio(wav_path)
    if y is None:
        return [], "load_failed"

    total_dur = len(y) / float(sr)
    onsets = _detect_onsets(y, sr)
    spans = _windows_from_onsets(onsets, total_dur)
    if not spans:
        return [], "no_onsets"

    # ---- shared per-run fields (same for every event of the run) ----
    run_id = _get(row, schema.M_RUN, "") or ""
    frets = _get(row, schema.M_FRETS, []) or []
    if not isinstance(frets, list):
        frets = []
    string_num = _string_num_for_row(row)
    # Chord / strum / free rows carry no per-note fret assignment.
    is_chordlike = (string_num == 0) or (len(frets) == 0)

    video_path = schema.abspath(_get(files, "video"))
    common = {
        "run_id": run_id,
        "session_id": _get(row, schema.M_SESSION, "") or "",
        "player_id": _get(row, schema.M_PLAYER, "") or "",
        "block": _get(row, schema.M_BLOCK),
        "pass": _get(row, schema.M_PASS),
        "held_out": bool(_get(row, schema.M_HELDOUT, False)),
        "intended_class": _get(row, schema.M_LABEL),
        "intended_placement": _get(row, schema.M_PLACEMENT),
        "string_num": string_num,
        "finger": _get(row, schema.M_FINGER),
        "pluck_strength": _get(row, schema.M_PLUCK),
        "audio_peak_dbfs": _get(audio_meta, "peak_dbfs"),
        "audio_clipped": bool(_get(audio_meta, "clipped", False)),
        "audio_silent": bool(_get(audio_meta, "silent", False)),
        "wav_path": wav_path,
        "video_path": video_path,
        "video_frame_rate": _get(video_meta, "frame_rate"),
        "guitar_id": guitar_id or "",
    }

    events = []
    for k, (onset_s, dur_s) in enumerate(spans):
        # target_fret: k-th onset -> frets[k]; chord/free or overflow -> -1.
        if is_chordlike or k >= len(frets):
            target_fret = -1
        else:
            target_fret = _as_int(frets[k], -1)

        f0_hz = _f0_median_hz(y, sr, onset_s, dur_s)
        f0_midi = schema.hz_to_midi(f0_hz)
        f0_string_est, f0_fret_est = schema.f0_to_string_fret(f0_midi, string_num)

        if target_fret >= 0 and f0_fret_est is not None and f0_fret_est >= 0:
            label_fret_match = bool(f0_fret_est == target_fret)
        else:
            # Unknown / not applicable (chord, overflow, or no usable F0).
            label_fret_match = None

        ev = dict(common)
        ev["event_id"] = "%s#%d" % (run_id, k)
        ev["target_fret"] = target_fret
        ev["onset_s"] = float(onset_s)
        ev["dur_s"] = float(dur_s)
        ev["f0_hz"] = float(f0_hz)
        ev["f0_midi"] = float(f0_midi)
        ev["f0_string_est"] = int(f0_string_est) if f0_string_est is not None else 0
        ev["f0_fret_est"] = int(f0_fret_est) if f0_fret_est is not None else -1
        ev["label_fret_match"] = label_fret_match
        events.append(ev)

    return events, "ok"


# ----------------------------------------------------------------- public API
def run(manifest_path, guitar_id=""):
    """Read one manifest.jsonl, segment every run's WAV into note events, attach
    labels from the manifest, and F0-cross-check each event.

    Returns a pandas.DataFrame with columns == schema.EVENT_COLUMNS (in order).
    Rows with no audio file or audio.silent are skipped (counted/logged, not
    emitted)."""
    import json

    manifest_path = schema.abspath(manifest_path)
    rows = []
    skipped = {"no_audio": 0, "silent": 0, "missing_file": 0,
               "load_failed": 0, "no_onsets": 0}
    n_runs = 0
    n_events = 0

    if not manifest_path or not os.path.exists(manifest_path):
        sys.stderr.write("segment: manifest not found: %r\n" % (manifest_path,))
        return _empty_frame()

    with open(manifest_path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                sys.stderr.write("segment: bad JSON at line %d, skipping\n" % lineno)
                continue
            if not isinstance(row, dict):
                continue
            n_runs += 1
            events, status = _events_for_row(row, guitar_id)
            if status != "ok":
                if status in skipped:
                    skipped[status] += 1
                rid = _get(row, schema.M_RUN, "?")
                sys.stderr.write("segment: skip run %s (%s)\n" % (rid, status))
                continue
            rows.extend(events)
            n_events += len(events)

    sys.stderr.write(
        "segment: %d runs -> %d events  (skipped: %s)\n"
        % (n_runs, n_events, ", ".join("%s=%d" % kv for kv in skipped.items()))
    )

    if not rows:
        return _empty_frame()

    df = pd.DataFrame(rows)
    # Enforce the frozen column contract: every column present, in order.
    for col in schema.EVENT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[schema.EVENT_COLUMNS]
    return df


def _empty_frame():
    """An empty DataFrame that still satisfies the column contract."""
    return pd.DataFrame({c: [] for c in schema.EVENT_COLUMNS})[schema.EVENT_COLUMNS]


# ----------------------------------------------------------------- CLI
def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Segment a capture run into labeled note events (events.csv)."
    )
    ap.add_argument("--session", required=True, help="session_id (data/raw/<session>/...)")
    ap.add_argument("--player", required=True, help="player_id (data/raw/<session>/<player>/...)")
    ap.add_argument("--guitar", default="", help="guitar_id for twin lookup (optional)")
    args = ap.parse_args()

    manifest_path = os.path.join(
        schema.RAW_DIR, args.session, args.player, "manifest.jsonl"
    )
    if not os.path.exists(manifest_path):
        sys.stderr.write("segment: no manifest at %s\n" % manifest_path)
        sys.exit(1)

    df = run(manifest_path, guitar_id=args.guitar)
    out = schema.out_dir(args.session, args.player)
    csv_path = os.path.join(out, "events.csv")
    df.to_csv(csv_path, index=False)
    sys.stderr.write("segment: wrote %d events -> %s\n" % (len(df), csv_path))


if __name__ == "__main__":
    main()
