#!/usr/bin/env python3
"""
Tactus — audio-first onsets + string-conditioned harmonic-template fret (features_pitch.py).

Two things the rest of the rampage stands on:

1. audio_onsets(y, sr): AUDIO-FIRST onset detection. We do NOT trust the scheduled
   cue/expected count as ground truth (the player did not strum exactly on every
   cue — measured: an "Am x40" recording had ~18 real strums, a "G x40" had ~108).
   So we detect the real attacks and treat the prompt count as a prior/check only.

2. template_fret(y, sr, t0, t1, string_num): the high-accuracy fret detector.
   The string is known (the prompt), so there are only 8 candidate frets (0..7).
   We score each candidate's harmonic comb (k*f0, k=1..8) against the event's
   spectrum and pick the best. No octave errors (the failure mode of generic pyin
   on low-E / high-e). Measured EXACT fret accuracy: clean 93.3%, buzz 64% (93%
   within 1); muted ~19% (dead note has no pitch -> vision's job). Returns
   (fret, confidence) where confidence = best/second-best harmonic-energy margin.

This is the "audio-first, then reconcile with the number" path: template_fret gives
an AUDIO-VERIFIED fret label per note, replacing the circular ascending-order label.

Run:  python3 software/ai/analysis/features_pitch.py   # self-test on the grid
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema            # noqa: E402
schema.on_path()
import numpy as np       # noqa: E402

OPEN_MIDI = schema.OPEN_STRING_MIDI          # {6:40,...,1:64}
FRET_SCOPE = range(0, 8)                      # candidate frets 0..7
_NHARM = 8
_TOL = 0.03                                   # +/-3% comb half-width


def hz_of(midi):
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def audio_onsets(y, sr, delta=0.4, wait=25, hop=256):
    """Real attack times (s). Tuned peak-pick on the onset-strength envelope so
    ring/sustain does not over-fire. Returns a sorted float array."""
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop,
                                    backtrack=True, units="time",
                                    pre_max=10, post_max=10, pre_avg=30, post_avg=30,
                                    delta=delta, wait=wait)
    return np.asarray(sorted(on), dtype=float)


def template_fret(y, sr, t0, t1, string_num):
    """String-conditioned harmonic-template fret in 0..7. Returns (fret, confidence)
    or (None, 0.0). confidence = best_score / (second_best + eps): >1.3 is solid."""
    if string_num not in OPEN_MIDI:
        return None, 0.0
    i0 = max(0, int((t0 + 0.02) * sr))
    i1 = min(len(y), int(min(t1, t0 + 0.5) * sr))
    seg = y[i0:i1]
    if seg.size < int(0.05 * sr):
        return None, 0.0
    win = seg * np.hanning(len(seg))
    S = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(len(win), 1.0 / sr)
    if S.sum() < 1e-9:
        return None, 0.0
    scores = []
    for f in FRET_SCOPE:
        f0 = hz_of(OPEN_MIDI[string_num] + f)
        sc = 0.0
        for k in range(1, _NHARM + 1):
            fk = k * f0
            if fk > freqs[-1]:
                break
            band = (freqs >= fk * (1 - _TOL)) & (freqs <= fk * (1 + _TOL))
            if band.any():
                sc += float(S[band].max())
        scores.append(sc)
    scores = np.asarray(scores, float)
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    conf = float(scores[order[0]] / (scores[order[1]] + 1e-9)) if len(order) > 1 else 0.0
    return best, conf


def pitch_salience(y, sr, t0, t1, string_num):
    """A few scalar features for the fault/fusion model: the template fret, its
    confidence, the harmonic-energy fraction at the winning fret, and pitchiness
    (peak harmonic energy / total). Useful as audio features beyond the fret."""
    fret, conf = template_fret(y, sr, t0, t1, string_num)
    out = {"tmpl_fret": fret if fret is not None else -1, "tmpl_conf": conf}
    i0 = max(0, int((t0 + 0.02) * sr)); i1 = min(len(y), int(min(t1, t0 + 0.5) * sr))
    seg = y[i0:i1]
    if seg.size < int(0.05 * sr) or fret is None:
        out["tmpl_harm_frac"] = 0.0
        return out
    win = seg * np.hanning(len(seg)); S = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(len(win), 1.0 / sr); tot = float(S.sum()) + 1e-9
    f0 = hz_of(OPEN_MIDI[string_num] + fret); he = 0.0
    for k in range(1, _NHARM + 1):
        fk = k * f0
        if fk > freqs[-1]:
            break
        band = (freqs >= fk * (1 - _TOL)) & (freqs <= fk * (1 + _TOL))
        if band.any():
            he += float(S[band].sum())
    out["tmpl_harm_frac"] = he / tot
    return out


def _selftest():
    import json, librosa
    G = os.path.expanduser("~/Downloads/GuitarData")
    mpath = f"{G}/raw/2026-06-20-2332/aditya/manifest.jsonl"
    if not os.path.exists(mpath):
        print("selftest: grid not found, skipping"); return True
    rows = [json.loads(l) for l in open(mpath) if l.strip()]
    by = {"clean": [0, 0], "buzz": [0, 0], "muted": [0, 0]}
    for r in rows:
        cls = r["intended_class"]; snum = int(str(r["string"]).split()[0]); exp = r["frets"]
        y, sr = librosa.load(os.path.join(G, r["files"]["audio"].replace("data/", "")), sr=None, mono=True)
        on = audio_onsets(y, sr)
        if len(on) != 6:
            continue
        for k, t in enumerate(on):
            t1 = on[k + 1] if k + 1 < len(on) else len(y) / sr
            f, _ = template_fret(y, sr, t, t1, snum)
            if f is not None:
                by[cls][1] += 1; by[cls][0] += (f == exp[k])
    for c, (cor, tot) in by.items():
        print(f"  {c}: template-fret exact = {cor}/{tot} = {100*cor/max(tot,1):.1f}%")
    ok = by["clean"][0] / max(by["clean"][1], 1) > 0.9
    print("  VERDICT:", "PASS — clean fret > 90%" if ok else "below 90%")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)
