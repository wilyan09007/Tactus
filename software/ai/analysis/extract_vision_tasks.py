#!/usr/bin/env python3
"""
Tactus rampage — MediaPipe vision extractor (extract_vision.py).  RUN UNDER py3.11.

mediapipe has no py3.14 wheel, so the analysis pipeline (3.14) gets its vision
features from here. This script:
  1. reads data/analysis/all/events.csv (event_id, video_path, onset_s, finger, ...),
  2. runs MediaPipe Hands at each event's frame (one VideoCapture + one Hands
     instance reused across a video -> fast),
  3. picks the FRETTING hand as the one whose landmarks fall on the fretboard
     (twin homography polygon),
  4. reuses the tested features_vision geometry for schema.VISION_FEATURES,
  5. ALSO dumps the RAW 21 hand landmarks per event -> the substrate for the
     "beat raw MediaPipe" position award (raw-MP fingertip->fret vs our model).

Outputs (data/analysis/all/):
  features_vision.csv   schema.VISION_FEATURES per event (single-best-H approx*)
  landmarks_raw.csv     event_id + handedness/score + lm0..20 (x,y,z norm) + w,h

*single static keyframe H is approximate while the guitar moves; per-frame neck
 detection is a downstream refinement (the award agent). Raw landmarks are exact.

Run:
  .venv311/bin/python software/ai/analysis/extract_vision.py --guitar acoustic-1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema            # noqa: E402
schema.on_path()
import fretboard         # noqa: E402
import features_vision   # noqa: E402  (reuse _event_features / _load_active_keyframe)

import numpy as np       # noqa: E402
import pandas as pd      # noqa: E402
import cv2               # noqa: E402
import mediapipe as mp   # noqa: E402
from mediapipe.tasks.python import BaseOptions                              # noqa: E402
from mediapipe.tasks.python.vision import (HandLandmarker,                  # noqa: E402
                                           HandLandmarkerOptions, RunningMode)

ALL_DIR = os.path.join(schema.OUT_DIR, "all")
# this mediapipe build (0.10.35) ships only the Tasks API (no legacy mp.solutions),
# so we use HandLandmarker + the downloaded .task model.
_MODEL = os.path.join(schema.VISION_DIR, "models", "hand_landmarker.task")
_IDS = {"index": (8, 6, 5), "middle": (12, 10, 9),
        "ring": (16, 14, 13), "pinky": (20, 18, 17)}


def _fret_polygon(H):
    """Fretboard quad (nut..fret7) in image px as an int polygon, for fretting-hand
    selection (a hand whose wrist/MCPs sit on the board is the fretting hand)."""
    if H is None:
        return None
    x7 = fretboard.fret_fraction(7)
    quad = fretboard.project(H, [[0, 0], [0, 1], [x7, 1], [x7, 0]])
    return quad.astype(np.int32)


def _hand_on_board(lms, w, h, poly):
    """Score how many of a hand's key landmarks fall inside the fretboard polygon."""
    if poly is None:
        return 0
    pts = [(0,), (5,), (9,), (13,), (17,)]  # wrist + MCPs
    s = 0
    for (i,) in pts:
        x, y = lms[i].x * w, lms[i].y * h
        if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
            s += 1
    return s


class BatchProvider:
    """features_vision landmark provider: caches one VideoCapture per path and one
    Hands instance; selects the fretting hand via the fretboard polygon; records
    the chosen hand's raw landmarks in `self.last_raw` for the award dump."""

    def __init__(self, poly):
        self.poly = poly
        self.caps = {}
        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL),
            running_mode=RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3)
        self.lm = HandLandmarker.create_from_options(opts)
        self.last_raw = None

    def _cap(self, path):
        if path not in self.caps:
            self.caps[path] = cv2.VideoCapture(path)
        return self.caps[path]

    def __call__(self, video_path, time_s):
        self.last_raw = None
        if not video_path:
            return None
        cap = self._cap(video_path)
        if not cap or not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_s)) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.hand_landmarks:
            return None
        # pick fretting hand = most landmarks on the board (fallback: first hand).
        # Tasks API: res.hand_landmarks[hi] is a list of 21 NormalizedLandmark.
        best, best_hi, best_s = None, 0, -1
        for hi, hand in enumerate(res.hand_landmarks):
            s = _hand_on_board(hand, w, h, self.poly)
            if s > best_s:
                best, best_hi, best_s = hand, hi, s
        lms = best

        def px(i):
            return (float(lms[i].x) * w, float(lms[i].y) * h)

        tips, joints = {}, {"wrist": px(0), "index_mcp": px(5)}
        for f, (tip_i, pip_i, mcp_i) in _IDS.items():
            tips[f] = px(tip_i)
            joints[f + "_tip"] = px(tip_i)
            joints[f + "_pip"] = px(pip_i)
            joints[f + "_mcp"] = px(mcp_i)
        handed, score = "", float("nan")
        try:
            cat = res.handedness[best_hi][0]
            handed, score = cat.category_name, float(cat.score)
        except Exception:
            pass
        raw = {"handedness": handed, "hand_score": score, "frame_w": w, "frame_h": h,
               "on_board": best_s, "n_hands": len(res.hand_landmarks)}
        for i in range(21):
            raw["lm%d_x" % i] = float(lms[i].x)
            raw["lm%d_y" % i] = float(lms[i].y)
            raw["lm%d_z" % i] = float(lms[i].z)
        self.last_raw = raw
        return {"tips": tips, "joints": joints, "conf": score}

    def release(self):
        for c in self.caps.values():
            try:
                c.release()
            except Exception:
                pass
        try:
            self.lm.close()
        except Exception:
            pass


def run(guitar="acoustic-1"):
    events_csv = os.path.join(ALL_DIR, "events.csv")
    if not os.path.exists(events_csv):
        raise SystemExit("no %s — run build_matrix.py first" % events_csv)
    events = pd.read_csv(events_csv)

    twin_path = schema.resolve_twin(guitar)
    H, reproj = features_vision._load_active_keyframe(twin_path)
    if H is None:
        raise SystemExit("no twin homography for guitar=%r" % guitar)
    provider = BatchProvider(_fret_polygon(H))

    vis_rows, raw_rows = [], []
    n_hand = 0
    for n, (_, ev) in enumerate(events.iterrows(), 1):
        feats = features_vision._event_features(ev, H, reproj, provider)
        feats[schema.EVENT_ID] = ev[schema.EVENT_ID]
        vis_rows.append(feats)
        raw = {schema.EVENT_ID: ev[schema.EVENT_ID]}
        if provider.last_raw:
            raw.update(provider.last_raw)
            n_hand += 1
        raw_rows.append(raw)
        if n % 100 == 0:
            print("  %d/%d events  (%d with a hand)" % (n, len(events), n_hand), flush=True)
    provider.release()

    vis = pd.DataFrame(vis_rows, columns=[schema.EVENT_ID] + schema.VISION_FEATURES)
    vis.to_csv(os.path.join(ALL_DIR, "features_vision.csv"), index=False)
    pd.DataFrame(raw_rows).to_csv(os.path.join(ALL_DIR, "landmarks_raw.csv"), index=False)
    print("extract_vision: %d events, hand found in %d (%.0f%%) -> features_vision.csv + landmarks_raw.csv"
          % (len(events), n_hand, 100.0 * n_hand / max(1, len(events))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--guitar", default="acoustic-1")
    run(ap.parse_args().guitar)
