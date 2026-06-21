#!/usr/bin/env python3
"""E8 (vision salvage): can the fretting-hand POSE alone identify the chord?

E7 showed per-finger FRET regression is registration-limited (the calib homography
doesn't register gameplay frames). But the RELATIVE hand shape — finger board
positions relative to each other, curls, wrist/neck angle — encodes the chord
WITHOUT needing absolute fretboard registration. We test that directly, and we do it
the honest way: train on one recording, test on a DIFFERENT recording (GroupKFold by
session over the two mixed-chord streams 0145 & 0305), so the model can't memorize a
single take. Chance = 1/n_chords.

    .venv/bin/python software/ai/analysis/exp/e8_pose_chord.py
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXP = "data/analysis/exp"
FEAT = ["index_bx", "middle_bx", "ring_bx", "pinky_bx",
        "index_by", "middle_by", "ring_by", "pinky_by",
        "index_z", "middle_z", "ring_z", "pinky_z",
        "index_curl", "middle_curl", "ring_curl", "pinky_curl",
        "wrist_angle", "neck_angle", "hand_conf"]
MIXED = ["2026-06-21-0145", "2026-06-21-0305"]  # multi-chord streams (no run==chord leak)


def _pipe():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         RandomForestClassifier(n_estimators=300, random_state=0))


def main():
    d = pd.read_csv(os.path.join(EXP, "vision_perfinger.csv"))
    # dedup on (session, event_id): chord-stream event_ids collide across sessions,
    # so event_id alone would silently merge 0145 & 0305 into one group.
    ev = d.drop_duplicates(["session_id", "event_id"]).copy()   # per-event hand-shape row
    mix = ev[ev.session_id.isin(MIXED)].copy()
    vc = mix.chord_name.value_counts()
    keep = vc[vc >= 4].index                             # drop singletons (F=1)
    mix = mix[mix.chord_name.isin(keep)].copy()
    X, y, g = mix[FEAT], mix.chord_name.values, mix.session_id.values
    chords = sorted(set(y))
    chance = 1.0 / len(chords)

    # honest headline: cross-RECORDING (train one stream, test the other)
    pred_x = cross_val_predict(_pipe(), X, y, groups=g, cv=GroupKFold(n_splits=2))
    acc_x = accuracy_score(y, pred_x)
    # optimistic reference: stratified within the pooled streams
    pred_s = cross_val_predict(_pipe(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    acc_s = accuracy_score(y, pred_s)

    print(f"E8 pose->chord  n={len(y)}  {len(chords)} chords  chance={chance:.3f}")
    print(f"  CROSS-RECORDING (GroupKFold by session): acc={acc_x:.3f}  ({acc_x/chance:.1f}x chance)")
    print(f"  stratified within-pool (optimistic):      acc={acc_s:.3f}  ({acc_s/chance:.1f}x chance)")

    cm = confusion_matrix(y, pred_x, labels=chords)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="magma")
    ax.set_xticks(range(len(chords))); ax.set_xticklabels(chords, rotation=45)
    ax.set_yticks(range(len(chords))); ax.set_yticklabels(chords)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Pose→chord, cross-recording\nacc={acc_x:.2f} vs chance {chance:.2f} ({acc_x/chance:.1f}x)")
    for i in range(len(chords)):
        for j in range(len(chords)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] < cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im); fig.tight_layout()
    png = os.path.join(EXP, "e8_pose_chord_confusion.png")
    fig.savefig(png, dpi=130); plt.close(fig)

    out = {"n": int(len(y)), "n_chords": len(chords), "chords": chords, "chance": chance,
           "acc_cross_recording": acc_x, "acc_cross_recording_x_chance": acc_x / chance,
           "acc_stratified_optimistic": acc_s, "png": png,
           "note": "Mixed streams 0145+0305 only (avoid run==chord leakage). One player; "
                   "cross-recording = train one stream/test the other. Relative hand-shape "
                   "features, so absolute fretboard registration is not required."}
    with open(os.path.join(EXP, "e8_pose_chord.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {png} + e8_pose_chord.json")
    # one runnable assert: the pipeline produced a prediction per event
    assert len(pred_x) == len(y)
    return out


if __name__ == "__main__":
    main()
