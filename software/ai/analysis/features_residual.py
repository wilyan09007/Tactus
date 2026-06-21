#!/usr/bin/env python3
"""
Tactus analysis — HARMONIC-RESIDUAL features (features_residual.py).  [E3 / H2]

The key bet (docs/27 §3 H2): a buzz/rattle is broadband and NON-harmonic, while a
note's or chord's *correct* content is harmonic and KNOWN (the prompt is the
prior). So if we null the spectrum at every expected harmonic — k*f0 for each
intended pitch — what remains (the residual) carries the FAULT in a context-free
subspace. Computed identically for a single note (one known pitch) and a chord
(several known pitches), the residual lets a buzz axis learned on MONO transfer to
held-out CHORD residuals. That transfer is the whole hypothesis.

This module adds NO new label logic: the expected pitches come straight from the
event's prompt fields (string_num + target_fret for mono; the G-corrected
chord_shape 6-vector for chords; measured f0_hz as a last-resort fallback).

    .venv/bin/python software/ai/analysis/features_residual.py        # self-check
    run(events_df) -> DataFrame[event_id + RESIDUAL_FEATURES]
"""
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema  # noqa: E402

import librosa  # noqa: E402
import pandas as pd  # noqa: E402

# Named residual features (the column contract for E3). Each is computed on the
# spectrum AFTER the known harmonics are nulled, so it measures only the
# non-harmonic (fault) energy — comparable across mono notes and chords.
RESIDUAL_FEATURES = [
    "res_energy_ratio",    # residual energy / total energy  (THE broadband-fault axis)
    "res_centroid",        # spectral centroid of the residual (Hz)
    "res_flatness",        # spectral flatness of the residual (1 = white noise)
    "res_highband_ratio",  # residual energy above 4 kHz / residual energy
    "res_rolloff",         # 85% spectral rolloff of the residual (Hz)
    "perc_ratio",          # HPSS percussive / (harmonic+percussive) over the window
] + ["res_mfcc_%d" % i for i in range(1, 6)]   # 5 MFCCs of the residual magnitude

_N_HARM = 8          # null k*f0 for k = 1..8
_TOL_CENTS = 70.0    # +/- this around each harmonic is treated as "the note" -> nulled
_BUZZ_HZ = 4000.0
_NFFT_MAX = 4096     # bigger FFT than the timbre stage: harmonic nulling wants freq res
_NFFT_MIN = 512
_EPS = 1e-12
_A4 = 440.0


def _nan_row():
    return {k: float("nan") for k in RESIDUAL_FEATURES}


def _midi_to_hz(m):
    return _A4 * 2.0 ** ((m - 69.0) / 12.0)


def expected_hz(event):
    """Expected fundamental frequencies (Hz) for an event, from the PROMPT.

    Chord events: every non-muted string of the (G-corrected) chord_shape 6-vector
    [low-E..high-e], pitch = open-string MIDI + fret. Single notes: string_num +
    target_fret. Fallback: the measured f0_hz. Empty list if nothing is known."""
    cs = event.get("chord_shape")
    if isinstance(cs, str) and cs.strip().startswith("["):
        try:
            shape = json.loads(cs)
        except ValueError:
            shape = None
        if shape:
            out = []
            for i, fret in enumerate(shape):          # i: 0=low-E(string6) .. 5=high-e(string1)
                if fret is None or fret < 0:
                    continue
                s = schema.N_STRINGS - i               # 0->6 (low-E) .. 5->1 (high-e)
                if s in schema.OPEN_STRING_MIDI:
                    out.append(_midi_to_hz(schema.OPEN_STRING_MIDI[s] + int(fret)))
            if out:
                return out
    try:
        s = int(event.get("string_num") or 0)
        f = int(event.get("target_fret"))
    except (TypeError, ValueError):
        s, f = 0, -1
    if s in schema.OPEN_STRING_MIDI and f >= 0:
        return [_midi_to_hz(schema.OPEN_STRING_MIDI[s] + f)]
    f0 = event.get("f0_hz")
    try:
        f0 = float(f0)
        if np.isfinite(f0) and f0 > 0:
            return [f0]
    except (TypeError, ValueError):
        pass
    return []


def _pow2(n):
    if n < _NFFT_MIN:
        return None
    return min(_NFFT_MAX, 1 << int(np.floor(np.log2(n))))


def _harmonic_keep_mask(freqs, expected, n_harm=_N_HARM, tol_cents=_TOL_CENTS):
    """Boolean per-bin mask: True = RESIDUAL bin (keep), False = within tolerance of
    a known harmonic (null it). Tolerance is the wider of ~1.5 bins and the cents band."""
    keep = np.ones(freqs.shape, dtype=bool)
    if freqs.size < 2:
        return keep
    bin_hz = float(freqs[1] - freqs[0])
    nyq = float(freqs[-1])
    band = 2.0 ** (tol_cents / 1200.0) - 1.0
    for f in expected:
        if not (f and np.isfinite(f) and f > 0):
            continue
        for k in range(1, n_harm + 1):
            target = k * f
            if target >= nyq:
                break
            tol = max(1.5 * bin_hz, target * band)
            keep &= np.abs(freqs - target) > tol
    return keep


def residual_features(y, sr, expected):
    """RESIDUAL_FEATURES for one window given its expected fundamentals (Hz)."""
    feats = _nan_row()
    if y is None or sr is None or len(y) == 0:
        return feats
    y = np.asarray(y, dtype=np.float32)
    n_fft = _pow2(len(y))
    if n_fft is None:
        return feats
    hop = max(1, n_fft // 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))           # (bins, frames)
        if S.size == 0:
            return feats
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        power = (S ** 2)
        total = float(power.sum())
        if total <= _EPS:
            return feats

        keep = _harmonic_keep_mask(freqs, expected)
        res_S = S.copy()
        res_S[~keep, :] = 0.0
        res_power = (res_S ** 2)
        res_total = float(res_power.sum())

        feats["res_energy_ratio"] = res_total / (total + _EPS)

        rp = res_power.sum(axis=1)                  # per-bin residual energy
        s_rp = float(rp.sum())
        if s_rp > _EPS:
            feats["res_centroid"] = float((freqs * rp).sum() / s_rp)
            feats["res_highband_ratio"] = float(rp[freqs >= _BUZZ_HZ].sum() / s_rp)
            pos = rp[rp > 0]
            if pos.size:
                feats["res_flatness"] = float(np.exp(np.mean(np.log(pos))) / (rp.mean() + _EPS))
            csum = np.cumsum(rp)
            feats["res_rolloff"] = float(freqs[int(np.searchsorted(csum, 0.85 * csum[-1]))])
            # MFCCs of the residual magnitude spectrogram (timbre of the fault).
            try:
                n_mels = max(8, min(40, n_fft // 4))
                melS = librosa.feature.melspectrogram(S=res_power, sr=sr, n_mels=n_mels)
                mf = librosa.feature.mfcc(S=librosa.power_to_db(melS + _EPS), n_mfcc=5)
                m = np.nanmean(mf, axis=1)
                for i in range(5):
                    if i < m.shape[0] and np.isfinite(m[i]):
                        feats["res_mfcc_%d" % (i + 1)] = float(m[i])
            except Exception:
                pass

        try:
            yh, yp = librosa.effects.hpss(y, n_fft=n_fft)
            eh = float(np.sum(yh.astype(np.float64) ** 2))
            ep = float(np.sum(yp.astype(np.float64) ** 2))
            if eh + ep > _EPS:
                feats["perc_ratio"] = ep / (eh + ep)
        except Exception:
            pass
    return feats


def run(events_df):
    """One residual-feature row per event (key = schema.EVENT_ID), reading each
    event's WAV window [onset_s, onset_s+dur_s]. Caches WAVs by path."""
    cols = [schema.EVENT_ID] + list(RESIDUAL_FEATURES)
    cache = {}
    rows = []
    for _, ev in events_df.iterrows():
        wp = schema.abspath(ev.get("wav_path"))
        if wp and wp not in cache:
            try:
                cache[wp] = librosa.load(wp, sr=None, mono=True)
            except Exception:
                cache[wp] = (None, None)
        y, sr = cache.get(wp, (None, None))
        seg = None
        if y is not None and sr:
            try:
                on = max(0.0, float(ev.get("onset_s")))
                dur = float(ev.get("dur_s"))
                i0, i1 = int(on * sr), int((on + dur) * sr)
                seg = y[max(0, i0):min(len(y), max(i0, i1))]
            except (TypeError, ValueError):
                seg = None
        feats = residual_features(seg, sr, expected_hz(ev)) if seg is not None and seg.size else _nan_row()
        feats[schema.EVENT_ID] = ev.get(schema.EVENT_ID)
        rows.append(feats)
    return pd.DataFrame(rows, columns=cols)


def _selftest():
    """A pure harmonic tone has near-zero residual; add broadband noise and the
    residual-energy ratio jumps. That monotonic gap is the whole premise of E3."""
    sr = 48000
    t = np.arange(int(0.4 * sr)) / sr
    f0 = 110.0  # A2
    harm = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 9)).astype(np.float32)
    harm /= np.max(np.abs(harm)) + _EPS
    rng = np.random.default_rng(0)
    buzzy = harm + 0.30 * rng.standard_normal(harm.shape).astype(np.float32)

    r_clean = residual_features(harm, sr, [f0])["res_energy_ratio"]
    r_buzz = residual_features(buzzy, sr, [f0])["res_energy_ratio"]
    print("residual self-check (A2 + 8 harmonics):")
    print(f"  clean residual ratio = {r_clean:.4f}")
    print(f"  buzzy residual ratio = {r_buzz:.4f}")
    ok = r_clean < 0.15 and r_buzz > r_clean * 2
    print(f"  VERDICT: {'PASS — residual isolates the non-harmonic fault' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)
