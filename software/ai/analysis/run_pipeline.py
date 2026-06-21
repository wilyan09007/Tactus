#!/usr/bin/env python3
"""
Tactus offline-analysis pipeline — orchestrator (run_pipeline.py).

Wires the stages end to end over a batch of captured data:

    data/raw/<session>/<player>/{manifest.jsonl, audio/*.wav, video/*.webm}
    data/calib/<guitar>/twin.json                 (from software/ai/vision/twin.py)
      segment -> features_audio + features_vision -> collapse -> audit
    -> data/analysis/{events.csv, features_fused.csv, metrics.json, audit.html, audit.json}

Usage:
    python3 software/ai/analysis/run_pipeline.py --guitar acoustic-1
    python3 software/ai/analysis/run_pipeline.py --session 2026-06-20-1830 --player aditya --guitar acoustic-1

Notes:
- Leave-one-player-out separability needs >=2 players in the batch; with one
  player collapse falls back to k-fold and says so in metrics["split"].
- Vision features need the video files PLUS mediapipe + a calibrated twin
  (pip install -r software/requirements.txt; run software/ai/vision/twin.py).
  Without them the vision columns are NaN and the study runs audio-only — which
  is exactly the V1 baseline (audio alone should confuse the two buzz causes).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema           # noqa: E402
schema.on_path()        # put the vision dir on sys.path so features_vision -> fretboard works
import segment          # noqa: E402
import features_audio   # noqa: E402
import features_vision  # noqa: E402
import collapse         # noqa: E402
import audit            # noqa: E402

import pandas as pd     # noqa: E402


def _manifests(raw_dir, sessions=None, players=None):
    found = []
    for session, player, pdir in schema.iter_session_player(raw_dir):
        if sessions and session not in sessions:
            continue
        if players and player not in players:
            continue
        m = os.path.join(pdir, "manifest.jsonl")
        if os.path.exists(m):
            found.append((session, player, m))
    return found


def run_batch(raw_dir=None, out_dir=None, guitar=None, sessions=None,
              players=None, vision_provider=None, title=None):
    """Run the full pipeline over every manifest under raw_dir (optionally filtered
    by session/player). Returns a small summary dict and writes artifacts to out_dir.
    vision_provider is injectable for testing (see features_vision.run)."""
    raw_dir = raw_dir or schema.RAW_DIR
    out_dir = out_dir or schema.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    twin_path = schema.resolve_twin(guitar)   # None auto-picks if exactly one guitar calibrated
    if twin_path is None:
        print("note: no twin.json resolved (guitar=%r) -> vision features will be NaN (audio-only study)"
              % guitar)

    # 1) segment every manifest -> concat events
    frames = []
    for session, player, m in _manifests(raw_dir, sessions, players):
        ev = segment.run(m, guitar_id=(guitar or ""))
        if ev is not None and len(ev):
            frames.append(ev)

    if not frames:
        print("no events segmented under %s" % raw_dir)
        empty = pd.DataFrame(columns=schema.EVENT_COLUMNS)
        path = audit.run(empty, {}, out_dir, title=title or "Tactus batch audit (no data)")
        return {"events": 0, "players": 0, "out_dir": out_dir, "audit": path, "twin": twin_path}

    events = pd.concat(frames, ignore_index=True)

    # 2) features (audio always; vision NaN unless video + twin + mediapipe present)
    fa = features_audio.run(events)
    fv = features_vision.run(events, twin_path, landmarks_provider=vision_provider)

    # 3) merge by event_id -> one fused row per event (meta + audio + vision)
    fused = (events
             .merge(fa, on=schema.EVENT_ID, how="left")
             .merge(fv, on=schema.EVENT_ID, how="left"))

    # 4) separability metrics: audio-only vs fused, leave-one-player-out
    metrics = collapse.run(fused)

    # 5) one-screen audit
    audit_path = audit.run(events, metrics, out_dir, title=title or "Tactus batch audit")

    # persist artifacts
    events.to_csv(os.path.join(out_dir, "events.csv"), index=False)
    fused.to_csv(os.path.join(out_dir, "features_fused.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    n_players = int(events[schema.PLAYER].nunique())
    print("batch: %d events, %d player(s), split=%s -> %s"
          % (len(events), n_players, metrics.get("split"), audit_path))
    return {"events": int(len(events)), "players": n_players, "out_dir": out_dir,
            "audit": audit_path, "metrics": os.path.join(out_dir, "metrics.json"), "twin": twin_path}


def main():
    ap = argparse.ArgumentParser(description="Tactus offline-analysis pipeline")
    ap.add_argument("--guitar", default=None,
                    help="guitar_id -> data/calib/<guitar>/twin.json (vision side)")
    ap.add_argument("--session", action="append", help="limit to session id(s); repeatable")
    ap.add_argument("--player", action="append", help="limit to player id(s); repeatable")
    ap.add_argument("--raw-dir", default=None, help="override data/raw root")
    ap.add_argument("--out-dir", default=None, help="override data/analysis output dir")
    args = ap.parse_args()
    run_batch(raw_dir=args.raw_dir, out_dir=args.out_dir, guitar=args.guitar,
              sessions=args.session, players=args.player)


if __name__ == "__main__":
    main()
