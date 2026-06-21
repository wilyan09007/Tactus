#!/usr/bin/env python3
"""
Tactus offline-analysis pipeline -- audio feature stage (features_audio.py).

Stage 2 of the segment -> features -> PCA/LDA -> audit pipeline (see schema.py).
Given the per-note events DataFrame that segment.py emits (columns =
schema.EVENT_COLUMNS), read each event's WAV window [onset_s, onset_s+dur_s] and
extract the ~26 named timbre features in schema.AUDIO_FEATURES -- the column
contract consumed by collapse.py.

Design notes (docs/20-aiml-training-design §3 timbre; docs/23 §5; pluck-proxy =
attack_slope per 20-eng-review D1):

  * One feature ROW per input event, in input order; key column = schema.EVENT_ID.
  * WAVs are read ONCE per unique wav_path and cached; windows are sliced from the
    cached signal (events from the same run share a file). librosa is asked never
    to resample (sr=None) so analysis runs at native capture rate (48 kHz).
  * Window-length guards: n_fft is clamped to a power of two <= the window length,
    short-frame helpers fall back gracefully, and ANY feature that cannot be
    computed becomes NaN rather than raising. A degenerate window (a few samples)
    must still yield a full, correctly-named row.

Robust-proxy choices (documented so downstream readers know what the columns mean):

  * hnr            -- harmonic-to-noise ratio via librosa HPSS: 10*log10(harmonic
                      energy / percussive energy). Cheap, monotone with real HNR,
                      and robust on short windows where autocorrelation HNR is noisy.
  * inharmonicity  -- mean fractional deviation of the strongest spectral peaks from
                      integer multiples of f0 (f0 from librosa.yin over the window).
                      NaN when f0 is unvoiced/unavailable.
  * buzz_band_ratio-- energy above 4 kHz / total energy (the broadband-buzz
                      signature). Computed straight from the magnitude spectrum.
  * attack_slope   -- rise rate of the RMS envelope from onset to its first peak
                      (RMS units per second); the pluck-proxy.

Run directly: `python features_audio.py --session <s> --player <p>` reads
events.csv from data/analysis/<s>/<p>/ and writes features_audio.csv beside it.
"""
import os
import sys
import warnings

import numpy as np

# Flat imports, no package install -- mirror the rest of the analysis dir.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema  # noqa: E402

import librosa  # noqa: E402
import pandas as pd  # noqa: E402


# ----------------------------------------------------------------- constants
HOP_DIV = 4                # hop_length = n_fft // HOP_DIV
N_FFT_MAX = 2048           # ceiling for the analysis FFT
N_FFT_MIN = 256            # smallest FFT we will attempt
N_MFCC = 13                # schema.AUDIO_FEATURES carries mfcc_1..mfcc_13
N_MELS = 40                # mel filters backing the MFCCs (fits small n_fft)
BUZZ_HZ = 4000.0           # high-band cutoff for buzz_band_ratio
F0_FMIN = 60.0             # yin search band: below low-E(2) fundamental ...
F0_FMAX = 1200.0           # ... up to well above the in-scope fretted range
A4_HZ = 440.0
EPS = 1e-12                # log/divide guard


def _nan_row():
    """A feature row with every schema.AUDIO_FEATURES value = NaN."""
    return {name: float("nan") for name in schema.AUDIO_FEATURES}


def _pow2_n_fft(n_samples):
    """Largest power of two in [N_FFT_MIN, N_FFT_MAX] that is <= n_samples.
    Returns None if the window is too short to FFT at all."""
    if n_samples < N_FFT_MIN:
        return None
    n = min(N_FFT_MAX, 1 << int(np.floor(np.log2(n_samples))))
    return n if n >= N_FFT_MIN else None


def _safe_mean(arr):
    """Mean of finite entries; NaN if none are finite."""
    arr = np.asarray(arr, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


# ----------------------------------------------------------------- wav cache
class _WavCache:
    """Read each WAV at most once (sr=None -> native rate) and slice windows.
    segment.py emits many events per run/file, so caching by path is the win."""

    def __init__(self):
        self._cache = {}   # abs_path -> (samples float32 mono, sr)

    def get(self, wav_path):
        ap = schema.abspath(wav_path)
        if ap is None or not os.path.exists(ap):
            return None, None
        if ap not in self._cache:
            try:
                y, sr = librosa.load(ap, sr=None, mono=True)
                self._cache[ap] = (np.asarray(y, dtype=np.float32), int(sr))
            except Exception:
                self._cache[ap] = (None, None)
        return self._cache[ap]

    def window(self, wav_path, onset_s, dur_s):
        """Samples for [onset_s, onset_s+dur_s] from the cached file, and sr."""
        y, sr = self.get(wav_path)
        if y is None or sr is None or len(y) == 0:
            return None, None
        try:
            onset_s = max(0.0, float(onset_s))
            dur_s = float(dur_s)
        except (TypeError, ValueError):
            return None, sr
        if not np.isfinite(onset_s) or not np.isfinite(dur_s) or dur_s <= 0:
            return None, sr
        i0 = int(round(onset_s * sr))
        i1 = int(round((onset_s + dur_s) * sr))
        i0 = max(0, min(i0, len(y)))
        i1 = max(i0, min(i1, len(y)))
        seg = y[i0:i1]
        return (seg if seg.size else None), sr


# ----------------------------------------------------------------- features
def _spectral_features(feats, y, sr, n_fft, hop):
    """spec_centroid / bandwidth / flatness / rolloff (mean over frames) plus the
    magnitude-spectrum-derived buzz_band_ratio. S is reused across all of them."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) + 0.0   # (1+n_fft/2, frames)
    if S.size == 0:
        return
    feats["spec_centroid"] = _safe_mean(
        librosa.feature.spectral_centroid(S=S, sr=sr, n_fft=n_fft, hop_length=hop))
    feats["spec_bandwidth"] = _safe_mean(
        librosa.feature.spectral_bandwidth(S=S, sr=sr, n_fft=n_fft, hop_length=hop))
    feats["spec_flatness"] = _safe_mean(
        librosa.feature.spectral_flatness(S=S, n_fft=n_fft, hop_length=hop))
    feats["spec_rolloff"] = _safe_mean(
        librosa.feature.spectral_rolloff(S=S, sr=sr, n_fft=n_fft, hop_length=hop))

    # buzz_band_ratio: power above BUZZ_HZ / total power. Power = |S|^2 summed.
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = (S ** 2).sum(axis=1)                      # per-bin energy over time
    total = float(power.sum())
    if total > EPS:
        feats["buzz_band_ratio"] = float(power[freqs >= BUZZ_HZ].sum() / total)


def _flux(feats, y, sr, n_fft, hop):
    """spec_flux := mean spectral-flux onset strength over the window. n_fft is
    forwarded so onset_strength's internal melspectrogram honors the window clamp
    (its default n_fft=2048 would otherwise warn/zero-pad short windows)."""
    try:
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, n_fft=n_fft)
        feats["spec_flux"] = _safe_mean(env)
    except Exception:
        pass


def _zcr_rms_envelope(feats, y, n_fft, hop, sr):
    """zcr, rms, and the RMS-envelope-derived log_attack_time / attack_slope /
    decay_rate. The RMS envelope is the shared substrate for the three temporal
    timbre features."""
    feats["zcr"] = _safe_mean(
        librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop))

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop).ravel()
    feats["rms"] = _safe_mean(rms)
    if rms.size == 0 or not np.isfinite(rms).any():
        return

    # Frame times (s) for the RMS envelope; center=True is librosa's default.
    t = librosa.frames_to_time(np.arange(rms.size), sr=sr, hop_length=hop, n_fft=n_fft)
    peak = int(np.nanargmax(rms))
    t_peak = float(t[peak])

    # log_attack_time: log10(time from onset(=window start) to peak RMS), guard >0.
    # Single-frame / instantaneous-attack windows -> NaN (no measurable rise).
    if t_peak > EPS:
        feats["log_attack_time"] = float(np.log10(t_peak))

    # attack_slope: rise rate of the RMS envelope onset->peak (RMS units / second).
    if peak >= 1 and t_peak > EPS:
        feats["attack_slope"] = float((rms[peak] - rms[0]) / t_peak)

    # decay_rate: slope of log-RMS AFTER the peak (per second; expected negative).
    if peak < rms.size - 1:
        t_dec = t[peak:] - t[peak]
        log_rms = np.log(np.maximum(rms[peak:], EPS))
        good = np.isfinite(log_rms) & np.isfinite(t_dec)
        if good.sum() >= 2 and np.ptp(t_dec[good]) > EPS:
            slope = np.polyfit(t_dec[good], log_rms[good], 1)[0]
            feats["decay_rate"] = float(slope)


def _hnr(feats, y, n_fft):
    """hnr proxy: 10*log10(harmonic energy / percussive energy) from HPSS."""
    try:
        y_h, y_p = librosa.effects.hpss(y, n_fft=n_fft)
        e_h = float(np.sum(y_h.astype(np.float64) ** 2))
        e_p = float(np.sum(y_p.astype(np.float64) ** 2))
        if e_h > EPS:
            feats["hnr"] = float(10.0 * np.log10(e_h / (e_p + EPS)))
    except Exception:
        pass


def _f0_features(feats, y, sr, n_fft, hop):
    """f0-derived inharmonicity + pitch_cents_dev. f0 via librosa.yin over the
    window (frame_length clamped to the window). Both NaN if f0 is unavailable."""
    frame_length = min(n_fft, len(y))
    if frame_length < N_FFT_MIN:
        return
    try:
        f0 = librosa.yin(y, fmin=F0_FMIN, fmax=min(F0_FMAX, sr / 2.0 - 1.0),
                         sr=sr, frame_length=frame_length, hop_length=hop)
    except Exception:
        return
    f0 = np.asarray(f0, dtype=float)
    f0 = f0[np.isfinite(f0) & (f0 > 0)]
    if f0.size == 0:
        return
    f0_hz = float(np.median(f0))               # robust window f0
    if f0_hz <= 0:
        return

    # pitch_cents_dev: cents from the nearest equal-tempered semitone (choked/sharp).
    midi = 69.0 + 12.0 * np.log2(f0_hz / A4_HZ)
    feats["pitch_cents_dev"] = float((midi - round(midi)) * 100.0)

    # inharmonicity: mean fractional deviation of strong spectral peaks from k*f0.
    feats["inharmonicity"] = _inharmonicity(y, sr, n_fft, hop, f0_hz)


def _inharmonicity(y, sr, n_fft, hop, f0_hz):
    """Mean |f_peak - k*f0| / (k*f0) over the strongest peaks that line up with a
    harmonic. Uses the time-averaged magnitude spectrum. NaN if no peaks found."""
    try:
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
        if S.size == 0:
            return float("nan")
        mag = S.mean(axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        if mag.size < 3:
            return float("nan")
        # Local maxima above a relative floor.
        thresh = mag.max() * 0.1
        peaks = np.where(
            (mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:]) & (mag[1:-1] > thresh)
        )[0] + 1
        if peaks.size == 0:
            return float("nan")
        nyq = sr / 2.0
        devs = []
        k_max = int(min(20, np.floor(nyq / f0_hz)))
        bin_hz = freqs[1] - freqs[0] if freqs.size > 1 else f0_hz
        for k in range(1, k_max + 1):
            target = k * f0_hz
            j = peaks[np.argmin(np.abs(freqs[peaks] - target))]
            # only count a peak that is genuinely near this harmonic (within ~half a bin spacing or 3%)
            if abs(freqs[j] - target) <= max(1.5 * bin_hz, 0.03 * target):
                devs.append(abs(freqs[j] - target) / target)
        return float(np.mean(devs)) if devs else float("nan")
    except Exception:
        return float("nan")


def _chroma(feats, y, sr, n_fft, hop):
    """chroma_peak: max of the mean chroma vector. chroma_stft (not _cqt) so very
    short windows -- where the CQT's low octaves have too few samples -- still work."""
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
        if chroma.size:
            feats["chroma_peak"] = float(np.nanmax(chroma.mean(axis=1)))
    except Exception:
        pass


def _mfcc(feats, y, sr, n_fft, hop):
    """mfcc_1..mfcc_13, mean over frames. n_mels capped so the mel filterbank is
    valid for small n_fft."""
    try:
        n_mels = max(N_MFCC, min(N_MELS, n_fft // 2))
        mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft,
                                  hop_length=hop, n_mels=n_mels)
        means = np.nanmean(mf, axis=1)
        for i in range(N_MFCC):
            v = means[i] if i < means.shape[0] else float("nan")
            feats["mfcc_%d" % (i + 1)] = float(v) if np.isfinite(v) else float("nan")
    except Exception:
        pass


def _extract_one(y, sr):
    """All schema.AUDIO_FEATURES for a single window. Never raises: any feature
    that cannot be computed stays NaN. Returns a dict keyed by feature name."""
    feats = _nan_row()
    if y is None or sr is None or len(y) == 0:
        return feats

    y = np.asarray(y, dtype=np.float32)
    n_fft = _pow2_n_fft(len(y))
    if n_fft is None:
        # Window too short to FFT. RMS/ZCR can still describe the raw window;
        # everything spectral stays NaN.
        try:
            feats["rms"] = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
            feats["zcr"] = float(np.mean(np.abs(np.diff(np.sign(y))) > 0)) if y.size > 1 else float("nan")
        except Exception:
            pass
        return feats

    hop = max(1, n_fft // HOP_DIV)

    # Each helper is independently guarded so one failure cannot abort the row.
    # Short/quiet windows legitimately trip librosa UserWarnings (empty mel
    # filters at small n_fft, tuning estimate on unpitched frames); those paths
    # already fall back to NaN, so silence the chatter rather than spam the caller.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        _spectral_features(feats, y, sr, n_fft, hop)
        _flux(feats, y, sr, n_fft, hop)
        _zcr_rms_envelope(feats, y, n_fft, hop, sr)
        _hnr(feats, y, n_fft)
        _f0_features(feats, y, sr, n_fft, hop)
        _chroma(feats, y, sr, n_fft, hop)
        _mfcc(feats, y, sr, n_fft, hop)
    return feats


# ----------------------------------------------------------------- public API
def run(events_df):
    """Given an events DataFrame (columns = schema.EVENT_COLUMNS, from segment.py),
    extract one feature row per event by reading each event's WAV window
    [onset_s, onset_s+dur_s]. Returns a DataFrame with columns
    [schema.EVENT_ID] + schema.AUDIO_FEATURES, one row per input event (same order)."""
    cols = [schema.EVENT_ID] + list(schema.AUDIO_FEATURES)
    cache = _WavCache()
    rows = []
    for _, ev in events_df.iterrows():
        seg, sr = cache.window(ev.get("wav_path"), ev.get("onset_s"), ev.get("dur_s"))
        feats = _extract_one(seg, sr)
        feats[schema.EVENT_ID] = ev.get(schema.EVENT_ID)
        rows.append(feats)

    out = pd.DataFrame(rows, columns=cols)
    if len(out) == 0:                      # preserve schema on empty input
        out = pd.DataFrame(columns=cols)
    return out.reset_index(drop=True)


# ----------------------------------------------------------------- CLI
def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract per-event audio features (schema.AUDIO_FEATURES) "
                    "from events.csv into features_audio.csv.")
    ap.add_argument("--session", required=True, help="session_id (data/analysis/<session>/...)")
    ap.add_argument("--player", required=True, help="player_id (data/analysis/<session>/<player>/...)")
    args = ap.parse_args()

    d = schema.out_dir(args.session, args.player)
    events_csv = os.path.join(d, "events.csv")
    if not os.path.exists(events_csv):
        sys.exit("no events.csv at %s (run segment.py first)" % events_csv)

    events_df = pd.read_csv(events_csv)
    feats_df = run(events_df)
    out_csv = os.path.join(d, "features_audio.csv")
    feats_df.to_csv(out_csv, index=False)
    print("wrote %s  (%d events x %d audio features)"
          % (out_csv, len(feats_df), len(schema.AUDIO_FEATURES)))


if __name__ == "__main__":
    main()
