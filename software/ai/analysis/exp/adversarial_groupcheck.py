#!/usr/bin/env python3
"""Adversarial leakage check: random k-fold vs GroupKFold-by-run_id.

Single-note runs are grouped (one run_id = one string+class+pluck, 6 fret events).
Random StratifiedKFold can put same-run events in train AND test → leaks a take's
acoustics → inflated accuracy. The honest metric holds out whole runs. If GROUP
accuracy ~= random, the result is real; if it drops hard, we were leaking.
    .venv/bin/python software/ai/analysis/exp/adversarial_groupcheck.py
"""
import sys
sys.path.insert(0, "software/ai/analysis")
import pandas as pd
import schema
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sn = pd.read_csv("data/analysis/features_fused.csv")
sn = sn[sn[schema.LABEL].isin(schema.CORE_CLASSES)].copy()
groups = sn["run_id"].values
A = schema.AUDIO_FEATURES


def chk(ycol, clf, name, chance):
    y, X = sn[ycol].values, sn[A]
    rnd = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    grp = cross_val_score(clf, X, y, groups=groups, cv=GroupKFold(5)).mean()
    flag = "  <-- LEAKAGE" if (rnd - grp) > 0.07 else "  ok"
    print(f"{name:34s} random {rnd:.3f} | GROUP {grp:.3f} | drop {rnd-grp:+.3f} | chance {chance:.2f}{flag}")


lda = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LDA())
rf = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(300, random_state=0))
print(f"n={len(sn)} single notes, {sn.run_id.nunique()} runs\n")
chk(schema.LABEL, lda, "clean/buzz/muted (LDA audio)", 0.333)
chk(schema.LABEL, rf, "clean/buzz/muted (RF audio)", 0.333)
chk("string_num", rf, "string-ID (RF audio)", 0.167)
