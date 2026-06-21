#!/usr/bin/env python3
"""
Tactus offline-analysis pipeline — VISION feature extractor (features_vision.py).

Per-event vision features (schema.VISION_FEATURES) computed by REUSING the
markerless fretboard engine (../vision/fretboard.py) plus the guitar's digital
twin (../vision/twin.py output). This module adds NO new geometry: it only
locates a video frame per event, asks a landmark provider for the fretting-hand
fingertips in pixels, and pushes those pixels through the existing homography.

THE crux feature is `d_active` — the signed along-neck fingertip->fret-wire
distance from fretboard.fingertip_d, the disambiguator between the two buzz
causes (light pressure vs misplacement; docs/24 §0b, schema.py VISION_FEATURES).

Call sequence into the engine (per event, H = active keyframe homography):
    board = fretboard.to_board(H, [tip_px])[0]      -> (bx, by) board coords
    d     = fretboard.fingertip_d(H, tip_px, fret)  -> bx - fret_fraction(fret)
    frac  = fretboard.fret_fraction(n)              -> nearest-fret search 0..7

SIGN CONVENTION (verified empirically against fretboard.py, not its prose):
    fingertip_d returns `bx - fret_fraction(n)`. The nut is x=0, the bridge is
    larger x. A fingertip placed BEHIND the wire (toward the nut) therefore has
    bx < fret_fraction(n) => d is NEGATIVE; toward the bridge => POSITIVE.
    NOTE: the fretboard.fingertip_d docstring prose says ">0 means BEHIND
    (toward the nut)", but its own self-test feeds a toward-nut tip
    (f3 - 0.012) and labels the ground truth as -0.012 — i.e. the CODE and the
    self-test agree the toward-nut sign is NEGATIVE, and the prose is inverted.
    We follow the code (the frozen engine), so downstream consumers get exactly
    what fretboard.fingertip_d emits. The placement-buzz direction is d < 0.

Provider seam (dependency-injected so the math is testable with zero footage):
    landmarks_provider(video_path, time_s) -> dict | None
        {"tips":   {"index":(x,y),"middle":(x,y),"ring":(x,y),"pinky":(x,y)},
         "joints": {... optional, for curl/wrist angles ...},
         "conf":   float}        coordinates in PIXELS, or None (=> NaN row).
    Default provider lazy-imports mediapipe + cv2.VideoCapture; if mediapipe is
    unavailable (it is optional and may not support this Python) it returns None,
    so every feature is NaN but the event_id is preserved.

Run directly (no package install; paths via schema.on_path()):
    python3 software/ai/analysis/features_vision.py --session S --player P --guitar G
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

import schema

schema.on_path()            # analysis dir + vision dir onto sys.path
import fretboard            # the engine we REUSE (do not reinvent)


# Fingers we can map to MediaPipe tips, in a stable order.
FINGERS = ("index", "middle", "ring", "pinky")
# Names of the four per-finger d columns, parallel to FINGERS.
_D_COLS = {"index": "d_index", "middle": "d_middle",
           "ring": "d_ring", "pinky": "d_pinky"}
# Fret candidates for the nearest-fret estimate (nut .. fret 7, matching the
# twin's calibrated grid range in twin.build_twin / fretboard.board_grid).
_FRET_CANDIDATES = range(0, 8)

# An all-NaN feature vector (order == schema.VISION_FEATURES) reused for every
# "no video / no hand / provider returned None" event.
_NAN_ROW = {c: float("nan") for c in schema.VISION_FEATURES}


# ----------------------------------------------------------------- twin / keyframe
def _load_active_keyframe(twin_path):
    """Load twin.json once and return (H float32 (3,3), reproj_px float) for the
    keyframe with the SMALLEST corner_reproj_px['median'] (best registration).
    Returns (None, nan) if the twin is missing or has no usable keyframe."""
    if not twin_path:
        return None, float("nan")
    try:
        with open(twin_path) as f:
            twin = json.load(f)
    except (OSError, ValueError):
        return None, float("nan")

    best_H, best_med = None, float("inf")
    for kf in twin.get("keyframes", []):
        H = kf.get("H_board_to_image")
        if H is None:
            continue
        med = (kf.get("corner_reproj_px") or {}).get("median")
        med = float(med) if med is not None else float("inf")
        if med < best_med:
            best_med, best_H = med, H
    if best_H is None:
        return None, float("nan")
    reproj = best_med if math.isfinite(best_med) else float("nan")
    return np.asarray(best_H, dtype=np.float32), reproj


# ----------------------------------------------------------------- board geometry
def _string_name(by):
    """Board Y (0..1, 0 = low-E) -> string name 6..1 (low-E=6 .. high-e=1).
    schema.N_STRINGS-1 evenly spaced lines; round to the nearest, clamp to 1..6."""
    raw = int(round(by * (schema.N_STRINGS - 1)))          # 0..5 (0 = low-E)
    s = schema.N_STRINGS - raw                              # 0->6 (low-E), 5->1 (high-e)
    return float(min(schema.N_STRINGS, max(1, s)))


def _nearest_fret(bx):
    """Fret n in 0..7 minimizing |bx - fret_fraction(n)| (board X = along neck)."""
    return int(min(_FRET_CANDIDATES,
                   key=lambda n: abs(bx - fretboard.fret_fraction(n))))


# ----------------------------------------------------------------- pose geometry
def _angle_at(a, b, c):
    """Interior angle (degrees) at vertex b of the path a-b-c, NaN if degenerate.
    Used for finger curl (PIP joint) once joint landmarks are wired in."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _pose_angles(joints, H):
    """Return (finger_curl_active_placeholder is handled by caller, here we give
    per-finger curls dict, wrist_angle, neck_angle) from optional joint landmarks.

    v1 emits NaN whenever `joints` is absent (the common case: the default
    MediaPipe provider is not runnable here). The formulas, for when joints land:
      finger_curl[f] = interior angle at that finger's PIP joint (mcp-pip-tip),
                       so a straight finger ~180 deg, a curled one smaller.
      wrist_angle    = angle of the wrist->index_MCP vector in image space
                       (atan2 of the metacarpal direction), degrees.
      neck_angle     = orientation of the fretboard's along-neck axis in image
                       space, from project(H, [(0,.5),(X7,.5)]); the wrist/neck
                       relative angle is the fretting-hand approach angle.
    Returns ({} , nan, nan) when joints is falsy."""
    curls = {}
    wrist_angle = float("nan")
    neck_angle = float("nan")

    # neck_angle depends only on H (board axis projected to image), so we can
    # always compute it when H is valid — it is pose geometry of the instrument.
    if H is not None:
        try:
            x7 = fretboard.fret_fraction(7)
            p = fretboard.project(H, [[0.0, 0.5], [x7, 0.5]])
            dx, dy = (p[1][0] - p[0][0]), (p[1][1] - p[0][1])
            neck_angle = math.degrees(math.atan2(dy, dx))
        except Exception:
            neck_angle = float("nan")

    if not joints:
        return curls, wrist_angle, neck_angle

    for f in FINGERS:
        mcp, pip, tip = joints.get(f + "_mcp"), joints.get(f + "_pip"), joints.get(f + "_tip")
        if mcp and pip and tip:
            curls[f] = _angle_at(mcp, pip, tip)

    wrist, idx_mcp = joints.get("wrist"), joints.get("index_mcp")
    if wrist and idx_mcp:
        wrist_angle = math.degrees(math.atan2(idx_mcp[1] - wrist[1],
                                              idx_mcp[0] - wrist[0]))
    return curls, wrist_angle, neck_angle


# ----------------------------------------------------------------- per-event core
def _resolve_finger(name):
    """Map an event's `finger` cell to one of FINGERS, or None for
    chord/free/unknown/blank (caller then falls back to the smallest-|d| tip)."""
    if name is None:
        return None
    n = str(name).strip().lower()
    return n if n in _D_COLS else None


def _event_features(event, H, reproj_px, provider):
    """Compute the schema.VISION_FEATURES dict for one event row. Returns a
    pure all-NaN copy when there is no video, no homography, or the provider
    yields nothing (contract: 'no video_path => all-NaN row', so reproj_px is
    NaN too until the homography is actually used to place a fingertip)."""
    row = dict(_NAN_ROW)

    video_path = event.get("video_path")
    if H is None or not video_path or (isinstance(video_path, float) and video_path != video_path):
        return row

    onset = event.get("onset_s")
    try:
        time_s = float(onset)
    except (TypeError, ValueError):
        time_s = 0.0
    if time_s != time_s:                      # NaN onset -> first frame
        time_s = 0.0

    lm = provider(schema.abspath(video_path), time_s)
    if not lm or not lm.get("tips"):
        return row                            # no hand / no video -> NaN row (id kept by caller)

    tips = lm["tips"]
    target_fret = event.get("target_fret")
    try:
        target_fret = int(target_fret)
    except (TypeError, ValueError):
        target_fret = -1

    # 1) board coords + per-finger d for every available fingertip.
    board_by_finger = {}        # finger -> (bx, by)
    d_by_finger = {}            # finger -> signed d at its effective target fret
    for f in FINGERS:
        tip = tips.get(f)
        if tip is None:
            continue
        tip_px = [float(tip[0]), float(tip[1])]
        bx, by = (float(v) for v in fretboard.to_board(H, [tip_px])[0])
        board_by_finger[f] = (bx, by)
        # use the event's target fret, or this finger's own nearest fret if unknown.
        fret_for_f = target_fret if target_fret >= 0 else _nearest_fret(bx)
        d_by_finger[f] = fretboard.fingertip_d(H, tip_px, fret_for_f)
        row[_D_COLS[f]] = d_by_finger[f]

    if not board_by_finger:
        return row

    # We have used the homography to place at least one fingertip, so the
    # keyframe registration quality is now a meaningful feature of this row.
    row["reproj_px"] = reproj_px

    # 2) pick the ACTIVE finger: the event's named finger, else the most-likely
    #    fretting tip = smallest |d| (closest to a wire).
    active = _resolve_finger(event.get("finger"))
    if active is None or active not in d_by_finger:
        active = min(d_by_finger, key=lambda f: abs(d_by_finger[f]))

    row["d_active"] = d_by_finger[active]
    abx, aby = board_by_finger[active]
    row["string_est"] = _string_name(aby)
    row["fret_est"] = float(_nearest_fret(abx))

    # 3) pose angles (NaN unless joints provided); neck_angle from H regardless.
    curls, wrist_angle, neck_angle = _pose_angles(lm.get("joints"), H)
    if curls:
        if active in curls:
            row["finger_curl_active"] = curls[active]
        vals = [v for v in curls.values() if v == v]
        if vals:
            row["finger_curl_mean"] = float(np.mean(vals))
    row["wrist_angle"] = wrist_angle
    row["neck_angle"] = neck_angle

    # 4) hand confidence from the provider.
    conf = lm.get("conf")
    row["hand_conf"] = float(conf) if conf is not None else float("nan")
    return row


# ----------------------------------------------------------------- default provider
def _default_provider():
    """MediaPipe-backed landmarks provider (lazy). Returns a callable
    provider(video_path, time_s)->dict|None. If mediapipe cannot be imported,
    returns a provider that always yields None (=> NaN feature rows), so this
    module degrades gracefully and never hard-requires mediapipe."""
    try:
        import mediapipe as mp          # noqa: F401  (lazy; optional dependency)
    except Exception:
        return lambda video_path, time_s: None

    import cv2  # available in this env; only imported on the real path.

    mp_hands = mp.solutions.hands
    _IDS = {           # MediaPipe Hands landmark indices for fingertips / joints
        "index": (8, 6, 5), "middle": (12, 10, 9),
        "ring": (16, 14, 13), "pinky": (20, 18, 17),
    }

    def provider(video_path, time_s):
        if not video_path:
            return None
        cap = cv2.VideoCapture(video_path)
        if not cap or not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                                min_detection_confidence=0.3) as hands:
                res = hands.process(rgb)
            if not res.multi_hand_landmarks:
                return None
            lms = res.multi_hand_landmarks[0].landmark

            def px(i):
                return (float(lms[i].x) * w, float(lms[i].y) * h)

            tips, joints = {}, {"wrist": px(0), "index_mcp": px(5)}
            for f, (tip_i, pip_i, mcp_i) in _IDS.items():
                tips[f] = px(tip_i)
                joints[f + "_tip"] = px(tip_i)
                joints[f + "_pip"] = px(pip_i)
                joints[f + "_mcp"] = px(mcp_i)
            conf = float("nan")
            if res.multi_handedness:
                try:
                    conf = float(res.multi_handedness[0].classification[0].score)
                except Exception:
                    conf = float("nan")
            return {"tips": tips, "joints": joints, "conf": conf}
        finally:
            cap.release()

    return provider


# ----------------------------------------------------------------- public API
def run(events_df, twin_path, landmarks_provider=None):
    """For each event: find the video frame at its onset, get fretting-hand
    landmarks, map fingertips to the fretboard via the twin homography, and
    compute schema.VISION_FEATURES.

    Returns a DataFrame with columns [schema.EVENT_ID] + schema.VISION_FEATURES,
    one row per event in input order. Events with no video, no detected hand, or
    a provider that returns None get an all-NaN feature row (event_id preserved).
    If landmarks_provider is None, the default MediaPipe-backed provider is used;
    when mediapipe is unavailable that provider yields None for every event."""
    provider = landmarks_provider if landmarks_provider is not None else _default_provider()
    H, reproj_px = _load_active_keyframe(twin_path)

    cols = [schema.EVENT_ID] + schema.VISION_FEATURES
    records = []
    for _, event in events_df.iterrows():
        feats = _event_features(event, H, reproj_px, provider)
        rec = {schema.EVENT_ID: event.get(schema.EVENT_ID)}
        rec.update(feats)
        records.append(rec)

    return pd.DataFrame(records, columns=cols)


# ----------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Compute per-event vision features.")
    ap.add_argument("--session", required=True)
    ap.add_argument("--player", required=True)
    ap.add_argument("--guitar", default=None,
                    help="guitar_id for twin lookup (data/calib/<guitar>/twin.json)")
    a = ap.parse_args()

    out = schema.out_dir(a.session, a.player)
    import os
    events_csv = os.path.join(out, "events.csv")
    if not os.path.exists(events_csv):
        raise SystemExit(f"no events.csv at {events_csv} — run segment.py first")
    events_df = pd.read_csv(events_csv)

    twin_path = schema.resolve_twin(a.guitar)
    if not twin_path:
        print(f"WARNING: no twin for guitar={a.guitar!r}; vision features will be all-NaN")

    feats = run(events_df, twin_path)
    out_csv = os.path.join(out, "features_vision.csv")
    feats.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(feats)} events x {len(schema.VISION_FEATURES)} vision features)")


if __name__ == "__main__":
    main()
