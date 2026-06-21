#!/usr/bin/env python3
"""
Deterministic self-test for collapse.py (no pytest).

Reflects the current taxonomy: **clean / buzz / muted** are 3 acoustically-distinct
classes, separable on AUDIO alone (the pivot; the old buzz-light vs buzz-placement
cause-split is dropped). We synthesize audio features with class-distinct means so
all three separate under leave-one-player-out, and assert that collapse.run reports
the separation (3x3 confusion, accuracy >> chance, pairwise d').

3 players, ~30 events/class, small per-player offsets + Gaussian noise; signal >>
offset so the result is deterministic (fixed RNG seed) and LOPO still holds.

Run:  .venv/bin/python software/ai/analysis/test_collapse.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema      # noqa: E402
import collapse    # noqa: E402

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402


def _make_df(seed=7, per_class=30):
    rng = np.random.default_rng(seed)
    players = ["P1", "P2", "P3"]
    classes = schema.CORE_CLASSES  # clean, buzz, muted

    player_offset = {
        p: {f: rng.normal(0, 0.20) for f in schema.FUSED_FEATURES}
        for p in players
    }

    # Audio drivers with a DISTINCT mean per class — the three faults sound
    # different (clean rings, buzz rattles, muted is dead):
    #   clean: high HNR, low buzz-band, normal energy
    #   buzz : low HNR,  high buzz-band, normal energy
    #   muted: low HNR,  low buzz-band, LOW energy + fast decay (dead thud)
    AUDIO_MEANS = {
        "clean": {"hnr": +3.0, "buzz_band_ratio": -3.0, "spec_flatness": -2.0,
                  "rms": +1.0, "decay_rate": -1.0},
        "buzz":  {"hnr": -3.0, "buzz_band_ratio": +3.0, "spec_flatness": +2.0,
                  "rms": +1.0, "decay_rate": -1.0},
        "muted": {"hnr": -3.0, "buzz_band_ratio": -1.0, "spec_flatness": 0.0,
                  "rms": -3.0, "decay_rate": +2.0},
    }

    def audio_mean(cls, feat):
        return AUDIO_MEANS.get(cls, {}).get(feat, 0.0)

    rows = []
    k = 0
    for p in players:
        for cls in classes:
            for _ in range(per_class):
                row = {schema.EVENT_ID: "R%d#%d" % (k, k), schema.PLAYER: p, schema.LABEL: cls}
                for f in schema.AUDIO_FEATURES:
                    row[f] = float(rng.normal(audio_mean(cls, f) + player_offset[p][f], 0.6))
                # vision features are pure noise here (audio carries clean/buzz/muted)
                for f in schema.VISION_FEATURES:
                    row[f] = float(rng.normal(player_offset[p][f], 0.6))
                rows.append(row)
                k += 1
    return pd.DataFrame(rows)


def _dprime(study, a, b):
    """Pull pairwise d'(a,b) out of whatever key shape collapse used."""
    pd_ = study.get("pairwise_dprime", {}) or {}
    for k, v in pd_.items():
        kl = k.lower()
        if a in kl and b in kl:
            return v
    return None


def main():
    df = _make_df()
    res = collapse.run(df)
    classes = schema.CORE_CLASSES
    n = len(classes)

    assert res["split"] == "lopo", "expected lopo split, got %r" % res["split"]
    assert res["n_players"] == 3, "expected 3 players, got %r" % res["n_players"]

    for name in ("audio", "fused"):
        cm = res[name]["confusion"]
        assert isinstance(cm, list) and len(cm) == n, "%s confusion not %dx%d" % (name, n, n)
        assert all(isinstance(r, list) and len(r) == n for r in cm), "%s confusion rows wrong width" % name
        assert res[name]["class_order"] == classes, "%s class_order mismatch" % name

    # all three classes separate on audio -> high accuracy + material pairwise d'
    acc = res["audio"]["accuracy"]
    assert acc > 0.6, "audio accuracy should be well above chance (0.33); got %.3f" % acc
    cb = _dprime(res["audio"], "clean", "buzz")
    cm_ = _dprime(res["audio"], "clean", "muted")
    bm = _dprime(res["audio"], "buzz", "muted")
    assert cb and cb > 1.0, "clean-vs-buzz d' should be material; got %r" % cb
    assert cm_ and cm_ > 1.0, "clean-vs-muted d' should be material; got %r" % cm_
    assert bm and bm > 1.0, "buzz-vs-muted d' should be material; got %r" % bm

    s = json.dumps(res)
    assert isinstance(s, str) and len(s) > 0, "result not JSON-serializable"

    print("split=%s  n_events=%d  n_players=%d" % (res["split"], res["n_events"], res["n_players"]))
    print("audio accuracy=%.3f (chance=0.33)" % acc)
    print("audio d':  clean-buzz=%.2f  clean-muted=%.2f  buzz-muted=%.2f" % (cb, cm_, bm))
    print("PASS: collapse")


if __name__ == "__main__":
    main()
