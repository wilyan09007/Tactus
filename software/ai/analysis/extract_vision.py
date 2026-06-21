#!/usr/bin/env python3
"""Vision-feature extraction over the master events.csv — RUN UNDER .venv311.

mediapipe ships no Python 3.14 wheel, so the main pipeline (.venv, 3.14) emits
all-NaN vision rows. This driver runs features_vision.run() under the 3.11
mediapipe venv with a FAST injected landmark provider (the frozen module's default
re-loads the Hands model once per event — ~1000x — which is minutes-slow). Here we:
  * build ONE mediapipe Hands instance and reuse it for every frame, and
  * cache cv2.VideoCapture per video file (chord recordings hold 40-100 events each).
Output: data/analysis/features_vision.csv for the experiments to merge by event_id.

    .venv311/bin/python software/ai/analysis/extract_vision.py --guitar acoustic-1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema            # noqa: E402
import features_vision   # noqa: E402
import pandas as pd      # noqa: E402


def _fast_provider():
    """One reused Hands model + a per-path VideoCapture cache. Same output shape as
    features_vision._default_provider (tips/joints/conf in pixels) or None."""
    import cv2
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                     min_detection_confidence=0.3)
    ids = {"index": (8, 6, 5), "middle": (12, 10, 9),
           "ring": (16, 14, 13), "pinky": (20, 18, 17)}
    caps = {}

    def _cap(path):
        if path not in caps:
            caps[path] = cv2.VideoCapture(path)
        return caps[path]

    def provider(video_path, time_s):
        if not video_path:
            return None
        cap = _cap(video_path)
        if not cap or not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return None
        lms = res.multi_hand_landmarks[0].landmark

        def px(i):
            return (float(lms[i].x) * w, float(lms[i].y) * h)

        tips, joints = {}, {"wrist": px(0), "index_mcp": px(5)}
        for f, (tip_i, pip_i, mcp_i) in ids.items():
            tips[f] = px(tip_i)
            joints[f + "_tip"], joints[f + "_pip"], joints[f + "_mcp"] = px(tip_i), px(pip_i), px(mcp_i)
        conf = float("nan")
        if res.multi_handedness:
            try:
                conf = float(res.multi_handedness[0].classification[0].score)
            except Exception:
                conf = float("nan")
        return {"tips": tips, "joints": joints, "conf": conf}

    provider._caps = caps  # keep a handle so the caller can release
    return provider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guitar", default="acoustic-1")
    ap.add_argument("--events", default=os.path.join(schema.OUT_DIR, "events.csv"))
    ap.add_argument("--out", default=os.path.join(schema.OUT_DIR, "features_vision.csv"))
    a = ap.parse_args()

    if not os.path.exists(a.events):
        raise SystemExit(f"no events.csv at {a.events} — run run_pipeline.py first")
    events = pd.read_csv(a.events)
    twin = schema.resolve_twin(a.guitar)
    if not twin:
        raise SystemExit(f"no twin.json for guitar={a.guitar!r} — run vision/twin.py first")

    provider = _fast_provider()
    fv = features_vision.run(events, twin, landmarks_provider=provider)
    for cap in getattr(provider, "_caps", {}).values():
        try:
            cap.release()
        except Exception:
            pass

    fv.to_csv(a.out, index=False)
    got = int(fv["d_active"].notna().sum())
    print(f"wrote {a.out}  ({len(fv)} events, {got} with a detected hand / d_active)")


if __name__ == "__main__":
    main()
