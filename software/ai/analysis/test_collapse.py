#!/usr/bin/env python3
"""
Deterministic self-test for collapse.py (no pytest).

We synthesize a dataset that bakes in exactly the V1/V2 story the separability
study is meant to detect (docs/20-eng-review V1/V2):

  * AUDIO features separate `clean` from the two buzz classes, BUT the two buzz
    classes (buzz-light vs buzz-placement) are drawn from the SAME audio
    distribution -> audio is cause-blind on the buzz pair (V1).
  * The VISION feature `d_active` strongly separates buzz-light (small d) from
    buzz-placement (large d) -> fusion separates the pair (V2).

3 players, ~30 events/class, small per-player offsets + Gaussian noise. Because
the signal >> the per-player offset, the result is separable under leave-one-
player-out and the asserts below are deterministic (fixed RNG seed).

Run:  .venv/bin/python software/ai/analysis/test_collapse.py
"""
import json
import os
import sys

# Same flat-import contract the modules use.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema      # noqa: E402
import collapse    # noqa: E402

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402


def _make_df(seed=7, per_class=30):
    rng = np.random.default_rng(seed)
    players = ["P1", "P2", "P3"]
    classes = schema.CORE_CLASSES  # clean, buzz-light, buzz-placement

    # Small per-player additive offset per feature (much smaller than the signal),
    # so LOPO still works but players aren't identical.
    player_offset = {
        p: {f: rng.normal(0, 0.20) for f in schema.FUSED_FEATURES}
        for p in players
    }

    # Audio class means: `clean` is far from both buzz classes; the two buzz
    # classes SHARE the same audio mean (cause-blind in audio -> V1). We push a
    # few well-known audio features so PCA keeps a clean->buzz axis.
    AUDIO_DRIVERS = ["buzz_band_ratio", "spec_flatness", "hnr", "zcr", "spec_flux"]

    def audio_mean(cls, feat):
        if feat not in AUDIO_DRIVERS:
            return 0.0
        if cls == "clean":
            # hnr is high (clean/harmonic), buzz-band low, flatness low for clean
            base = {"buzz_band_ratio": -3.0, "spec_flatness": -3.0,
                    "hnr": +3.0, "zcr": -2.0, "spec_flux": -2.0}
        else:  # BOTH buzz classes identical in audio
            base = {"buzz_band_ratio": +3.0, "spec_flatness": +3.0,
                    "hnr": -3.0, "zcr": +2.0, "spec_flux": +2.0}
        return base[feat]

    # Vision: d_active separates the buzz pair. clean sits in the "good" placement
    # region (small d, like buzz-light), so d_active does NOT help clean-vs-buzz —
    # only buzz-light vs buzz-placement. Other vision features are pure noise here.
    def d_active_mean(cls):
        return {"clean": 0.0, "buzz-light": 0.0, "buzz-placement": 6.0}[cls]

    rows = []
    k = 0
    for p in players:
        for cls in classes:
            for _ in range(per_class):
                row = {
                    schema.EVENT_ID: "R%d#%d" % (k, k),
                    schema.PLAYER: p,
                    schema.LABEL: cls,
                }
                # audio features (tight noise so clean<->buzz is clean,
                # buzz<->buzz overlaps because their means are identical)
                for f in schema.AUDIO_FEATURES:
                    mu = audio_mean(cls, f) + player_offset[p][f]
                    row[f] = float(rng.normal(mu, 0.6))
                # vision features
                for f in schema.VISION_FEATURES:
                    if f == "d_active":
                        mu = d_active_mean(cls) + player_offset[p][f]
                        row[f] = float(rng.normal(mu, 0.6))
                    else:
                        row[f] = float(rng.normal(player_offset[p][f], 0.6))
                rows.append(row)
                k += 1

    return pd.DataFrame(rows)


def main():
    df = _make_df()
    res = collapse.run(df)

    classes = schema.CORE_CLASSES
    n = len(classes)

    # ---- split must be LOPO (3 players)
    assert res["split"] == "lopo", "expected lopo split, got %r" % res["split"]
    assert res["n_players"] == 3, "expected 3 players, got %r" % res["n_players"]

    # ---- confusion matrices present + correct 3x3 shape for both sets
    for name in ("audio", "fused"):
        cm = res[name]["confusion"]
        assert isinstance(cm, list) and len(cm) == n, "%s confusion not %dx%d" % (name, n, n)
        assert all(isinstance(r, list) and len(r) == n for r in cm), \
            "%s confusion rows wrong width" % name
        assert res[name]["class_order"] == classes, "%s class_order mismatch" % name

    # ---- the headline: fused d'(buzz pair) > audio d'(buzz pair)
    a = res["buzz_pair_dprime"]["audio"]
    f = res["buzz_pair_dprime"]["fused"]
    assert a is not None and f is not None, "buzz-pair d' missing (audio=%r fused=%r)" % (a, f)
    assert f > a, "expected fused d' (%.3f) > audio d' (%.3f) on the buzz pair" % (f, a)

    # ---- V2 boolean must fire (fusion separates the buzz pair)
    assert res["v2_fusion_separates_buzz_pair"] is True, \
        "v2_fusion_separates_buzz_pair should be True (audio d'=%.3f fused d'=%.3f)" % (a, f)

    # ---- fused set must surface LDA loadings, and d_active should load the
    #      discriminant strongly (it is THE buzz-cause disambiguator).
    loadings = res["fused"].get("lda_loadings")
    assert loadings, "fused lda_loadings missing"
    all_top_feats = {e["feature"] for axis in loadings.values() for e in axis}
    assert "d_active" in all_top_feats, \
        "d_active should be a top LDA contributor in the fused set; got %s" % sorted(all_top_feats)

    # ---- everything JSON-serializable
    s = json.dumps(res)
    assert isinstance(s, str) and len(s) > 0

    # ---- informative echo
    print("split=%s  n_events=%d  n_players=%d"
          % (res["split"], res["n_events"], res["n_players"]))
    print("buzz-pair d':  audio=%.3f  fused=%.3f  (gain=%.3f)" % (a, f, f - a))
    print("audio accuracy=%.3f  fused accuracy=%.3f"
          % (res["audio"]["accuracy"], res["fused"]["accuracy"]))
    print("V1 audio confuses buzz pair = %s" % res["v1_audio_confuses_buzz_pair"])
    print("V2 fusion separates buzz pair = %s" % res["v2_fusion_separates_buzz_pair"])
    print("PASS: collapse")


if __name__ == "__main__":
    main()
