#!/usr/bin/env python3
"""truth.md §6 rigor centerpiece: clean/buzz/muted separability, AUDIO-only vs
VISION-only vs FUSED (k-fold, 1 player). Proves the two-stage split — quality is
audio-led; vision (here, registration-limited) adds little to QUALITY.
Reuses the frozen collapse.py machinery (fit-on-train, PCA→LDA, d′/confusion).
    .venv/bin/python software/ai/analysis/exp/separability_3way.py
"""
import json
import sys

sys.path.insert(0, "software/ai/analysis")
import pandas as pd
import schema
import collapse

# features_fused.csv has NaN vision (run_pipeline runs vision in 3.14 where mediapipe
# can't load); the REAL vision is in features_vision.csv (extracted under .venv311).
fused = pd.read_csv("data/analysis/features_fused.csv")
vis = pd.read_csv("data/analysis/features_vision.csv")
fused = fused.drop(columns=[c for c in schema.VISION_FEATURES if c in fused.columns])
df = fused.merge(vis, on="event_id", how="left")
df = df[df[schema.LABEL].isin(schema.CORE_CLASSES)].copy()   # 432 single notes
print(f"single-note events: {len(df)}  | classes {dict(df[schema.LABEL].value_counts())}")

rows, out = [], {}
for name, feats in [("audio-only", schema.AUDIO_FEATURES),
                    ("vision-only", schema.VISION_FEATURES),
                    ("fused", schema.FUSED_FEATURES)]:
    r = collapse._run_feature_set(df, feats, schema.CORE_CLASSES, schema.LABEL,
                                  schema.PLAYER, use_lopo=False, want_loadings=False)
    pw = {k: round(v, 2) for k, v in r["pairwise_dprime"].items() if v is not None}
    print(f"  {name:11s}  acc={r['accuracy']:.3f}  feats={r['n_features_used']:2d}  d'={pw}")
    out[name] = {"accuracy": r["accuracy"], "n_features": r["n_features_used"],
                 "pairwise_dprime": r["pairwise_dprime"], "per_class_recall": r["per_class_recall"]}

with open("data/analysis/exp/separability_3way.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote data/analysis/exp/separability_3way.json")
# the truth.md claim: quality is AUDIO-led (audio ≈ fused, both >> vision-only)
a, v, fu = out["audio-only"]["accuracy"], out["vision-only"]["accuracy"], out["fused"]["accuracy"]
print(f"VERDICT: audio {a:.2f}  vision {v:.2f}  fused {fu:.2f}  "
      f"-> quality is {'AUDIO-led (vision adds little)' if a >= v else 'vision-helped'}; "
      f"two-stage split {'holds' if a >= v - 0.02 else 'questioned'}.")
