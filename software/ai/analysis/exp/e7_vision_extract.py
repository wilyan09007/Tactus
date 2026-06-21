#!/usr/bin/env python3
"""
E7 STEP 1 — per-finger vision extraction for the "beat raw MediaPipe" experiment.

Runs in .venv311 (mediapipe 0.10.21 + cv2). For every CHORD event in
data/analysis/events.csv we:

  1. sample the frame(s) around onset_s (median pixel of up to 3 frames),
  2. run MediaPipe Hands (ONE reused Hands instance; VideoCapture cached/video),
  3. for each fretting fingertip (index=8, middle=12, ring=16, pinky=20) map the
     pixel -> board coords via the FROZEN engine fretboard.to_board(H, .),
  4. derive the RAW-MediaPipe estimate per finger:
         mp_string_est = string from board-Y  (engine convention, 6=low-E..1=high-e)
         mp_fret_est   = argmin_n |bx - fret_fraction(n)|  over n=0..7,
  5. attach the PER-FINGER GROUND TRUTH from chord_shape + chord_fingers,
  6. record the FULL hand-pose vector (all 4 tips bx/by/z, curls, wrist/neck
     angle, MCP/PIP px) so STEP 2's model can constrain an occluded finger by
     the visible hand,
  7. flag OCCLUSION per finger.

H comes from data/calib/<guitar>/twin.json: the keyframe with the smallest
corner_reproj_px['median'] (board->image). NOTE (honesty): those corners are
hand-clicked, so reproj median is 0.0 and the registration is COARSE — a single
static homography applied to every frame regardless of how the guitar actually
moved. We say so loudly in the report.

Output: data/analysis/exp/vision_perfinger.csv  (one row per (event,finger) with
chord_shape[i] > 0 and chord_fingers[i] in {1,2,3,4}).

This script reuses the frozen modules and adds NO geometry of its own:
  fretboard.to_board / fret_fraction / project    (../../vision/fretboard.py)
  the MediaPipe provider pattern                   (../extract_vision.py)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

# --- locate the frozen engine (software/ai/vision/fretboard.py) on sys.path ---
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../ai/analysis/exp
_AI = os.path.dirname(os.path.dirname(_HERE))               # .../ai
sys.path.insert(0, os.path.join(_AI, "vision"))
sys.path.insert(0, os.path.join(_AI, "analysis"))
import fretboard  # noqa: E402  (the frozen engine — REUSED, never modified)


def _find_repo_root(start):
    """Walk up from `start` until we find a dir containing data/analysis/events.csv."""
    d = start
    for _ in range(8):
        if os.path.exists(os.path.join(d, "data", "analysis", "events.csv")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    # fall back: two levels above software/ai (…/software/ai -> repo root)
    return os.path.dirname(os.path.dirname(_AI))


_REPO = _find_repo_root(_HERE)

# MediaPipe Hands landmark indices: tip, pip, mcp per finger (matches extract_vision).
_IDS = {
    "index": (8, 6, 5),
    "middle": (12, 10, 9),
    "ring": (16, 14, 13),
    "pinky": (20, 18, 17),
}
FINGERS = ("index", "middle", "ring", "pinky")
# chord_fingers code -> finger name (1=index,2=middle,3=ring,4=pinky).
_CODE2FINGER = {1: "index", 2: "middle", 3: "ring", 4: "pinky"}
_FRET_CANDIDATES = range(0, 8)


# ----------------------------------------------------------------- twin / keyframe
def load_active_H(twin_path):
    """(H float32 (3,3), reproj_px, label) for the keyframe with smallest
    corner_reproj_px['median']. Mirrors features_vision._load_active_keyframe."""
    with open(twin_path) as f:
        twin = json.load(f)
    best_H, best_med, best_label = None, float("inf"), None
    for kf in twin.get("keyframes", []):
        H = kf.get("H_board_to_image")
        if H is None:
            continue
        med = (kf.get("corner_reproj_px") or {}).get("median")
        med = float(med) if med is not None else float("inf")
        if med < best_med:
            best_med, best_H, best_label = med, H, kf.get("label")
    if best_H is None:
        raise SystemExit(f"no usable keyframe in {twin_path}")
    reproj = best_med if math.isfinite(best_med) else float("nan")
    return np.asarray(best_H, np.float32), reproj, best_label


# ----------------------------------------------------------------- board geometry
def string_from_by(by):
    """Board Y (0..1, 0=low-E) -> string number 6..1 (engine convention)."""
    raw = int(round(by * (fretboard.N_STRINGS - 1)))       # 0..5
    s = fretboard.N_STRINGS - raw                          # 0->6, 5->1
    return int(min(fretboard.N_STRINGS, max(1, s)))


def nearest_fret(bx):
    return int(min(_FRET_CANDIDATES,
                   key=lambda n: abs(bx - fretboard.fret_fraction(n))))


def angle_at(a, b, c):
    """Interior angle (deg) at vertex b of path a-b-c; NaN if degenerate."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def neck_angle_from_H(H):
    """Orientation (deg) of the along-neck board axis projected into the image."""
    try:
        x7 = fretboard.fret_fraction(7)
        p = fretboard.project(H, [[0.0, 0.5], [x7, 0.5]])
        return math.degrees(math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0]))
    except Exception:
        return float("nan")


# ----------------------------------------------------------------- ground truth
def per_finger_truth(chord_shape, chord_fingers):
    """{finger_name: (true_string_number, true_fret)} for fretted strings.

    For string index i (0=low-E..5=high-e): if chord_shape[i] > 0 and
    chord_fingers[i] in {1..4}, that finger presses string_number=(6-i)
    at fret=chord_shape[i]."""
    out = {}
    for i in range(6):
        fret = chord_shape[i]
        code = chord_fingers[i]
        if fret is not None and fret > 0 and code in _CODE2FINGER:
            out[_CODE2FINGER[code]] = (6 - i, int(fret))
    return out


# ----------------------------------------------------------------- mediapipe core
def make_hands():
    from mediapipe import solutions
    return solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                 min_detection_confidence=0.3)


def sample_landmarks(cap, hands, time_s, n_frames=3):
    """Run MediaPipe on up to n_frames around time_s; return a single landmark
    record using the MEDIAN pixel/z per landmark across the frames that detected
    a hand (robust to a single bad frame). None if no frame detects a hand.

    Returns dict: lm[i] -> (px_x, px_y, z) for i in 0..20, plus 'conf','w','h',
    'n_used' (#frames that contributed)."""
    import cv2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    half = (n_frames - 1) // 2
    offsets = [(-half + k) * dt for k in range(n_frames)]

    per_lm = {i: [] for i in range(21)}
    confs = []
    W = H_ = None
    for off in offsets:
        t = max(0.0, time_s + off)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        H_, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            continue
        lms = res.multi_hand_landmarks[0].landmark
        for i in range(21):
            per_lm[i].append((float(lms[i].x) * W, float(lms[i].y) * H_, float(lms[i].z)))
        if res.multi_handedness:
            try:
                confs.append(float(res.multi_handedness[0].classification[0].score))
            except Exception:
                pass

    n_used = len(per_lm[0])
    if n_used == 0:
        return None
    lm = {}
    for i in range(21):
        arr = np.asarray(per_lm[i], float)
        lm[i] = tuple(np.median(arr, axis=0))
    return {"lm": lm, "conf": float(np.median(confs)) if confs else float("nan"),
            "w": W, "h": H_, "n_used": n_used}


# ----------------------------------------------------------------- occlusion flag
def occlusion_flag(bx, by, lm_z, conf, z_thresh, conf_thresh):
    """1 if the finger is likely occluded / unreliable, else 0.

    Heuristic (any one triggers):
      - tip maps OFF the board: bx outside [-0.05, fret_fraction(7)+0.08] or
        by outside [-0.15, 1.15],
      - lm.z deeper than this event's z_thresh (low percentile of the 4 tip z's),
      - hand detection confidence below conf_thresh.
    Returns (flag:int, reason:str)."""
    x_lo, x_hi = -0.05, fretboard.fret_fraction(7) + 0.08
    y_lo, y_hi = -0.15, 1.15
    reasons = []
    if not (x_lo <= bx <= x_hi) or not (y_lo <= by <= y_hi):
        reasons.append("offboard")
    if lm_z == lm_z and lm_z < z_thresh:
        reasons.append("deepz")
    if conf == conf and conf < conf_thresh:
        reasons.append("lowconf")
    return (1 if reasons else 0), "|".join(reasons)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=os.path.join(_REPO, "data/analysis/events.csv"))
    ap.add_argument("--twin", default=os.path.join(_REPO, "data/calib/acoustic-1/twin.json"))
    ap.add_argument("--out", default=os.path.join(_REPO, "data/analysis/exp/vision_perfinger.csv"))
    ap.add_argument("--n-frames", type=int, default=3, help="frames to median around onset")
    ap.add_argument("--conf-thresh", type=float, default=0.6,
                    help="hand-conf below this => occlusion-flag contribution")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #events (0=all)")
    a = ap.parse_args()

    print(f"[paths] repo_root={_REPO}")
    H, reproj_px, kf_label = load_active_H(a.twin)
    print(f"[twin] active keyframe label={kf_label!r}  corner_reproj_median={reproj_px}px "
          f"(hand-clicked => coarse, single static H for all frames)")
    neck_ang = neck_angle_from_H(H)

    df = pd.read_csv(a.events)
    ch = df[df["chord_name"].notna()].copy().reset_index(drop=True)
    if a.limit:
        ch = ch.head(a.limit)
    print(f"[events] {len(ch)} chord events")

    hands = make_hands()
    import cv2

    rows = []
    cap = None
    cur_video = None
    n_event_nohand = 0
    try:
        for ei, ev in ch.iterrows():
            vp = ev.get("video_path")
            if not isinstance(vp, str) or not vp:
                continue
            try:
                shape = json.loads(ev["chord_shape"])
                fingers = json.loads(ev["chord_fingers"])
            except Exception:
                continue
            truth = per_finger_truth(shape, fingers)
            if not truth:
                continue

            if vp != cur_video:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(vp)
                cur_video = vp
            if cap is None or not cap.isOpened():
                continue

            onset = ev.get("onset_s")
            try:
                t = float(onset)
            except (TypeError, ValueError):
                t = 0.0
            if t != t:
                t = 0.0

            rec = sample_landmarks(cap, hands, t, n_frames=a.n_frames)
            if rec is None:
                n_event_nohand += 1
                continue
            lm, conf, w, h, n_used = (rec["lm"], rec["conf"], rec["w"], rec["h"], rec["n_used"])

            tip_board = {}
            for f in FINGERS:
                ti = _IDS[f][0]
                px, py, z = lm[ti]
                bx, by = (float(v) for v in fretboard.to_board(H, [[px, py]])[0])
                tip_board[f] = (bx, by, z, px, py)

            zs = np.asarray([tip_board[f][2] for f in FINGERS], float)
            z_thresh = float(np.percentile(zs, 25)) - 1e-6

            curls, mcp_px, pip_px = {}, {}, {}
            for f in FINGERS:
                tip_i, pip_i, mcp_i = _IDS[f]
                curls[f] = angle_at(lm[mcp_i][:2], lm[pip_i][:2], lm[tip_i][:2])
                mcp_px[f] = lm[mcp_i][:2]
                pip_px[f] = lm[pip_i][:2]
            wrist_px = lm[0][:2]
            index_mcp_px = lm[5][:2]
            wrist_angle = math.degrees(math.atan2(index_mcp_px[1] - wrist_px[1],
                                                  index_mcp_px[0] - wrist_px[0]))

            for f, (true_string, true_fret) in truth.items():
                bx, by, z, px, py = tip_board[f]
                mp_string_est = string_from_by(by)
                mp_fret_est = nearest_fret(bx)
                occ, occ_reason = occlusion_flag(bx, by, z, conf, z_thresh, a.conf_thresh)
                row = {
                    "event_id": ev.get("event_id"),
                    "run_id": ev.get("run_id"),
                    "session_id": ev.get("session_id"),
                    "player_id": ev.get("player_id"),
                    "chord_name": ev.get("chord_name"),
                    "onset_s": t,
                    "video_path": vp,
                    "finger": f,
                    "true_string": true_string,
                    "true_fret": true_fret,
                    "mp_string_est": mp_string_est,
                    "mp_fret_est": mp_fret_est,
                    "bx": bx, "by": by, "z": z, "tip_px_x": px, "tip_px_y": py,
                    "curl": curls[f],
                    "mcp_px_x": mcp_px[f][0], "mcp_px_y": mcp_px[f][1],
                    "pip_px_x": pip_px[f][0], "pip_px_y": pip_px[f][1],
                    **{f"{g}_bx": tip_board[g][0] for g in FINGERS},
                    **{f"{g}_by": tip_board[g][1] for g in FINGERS},
                    **{f"{g}_z": tip_board[g][2] for g in FINGERS},
                    **{f"{g}_curl": curls[g] for g in FINGERS},
                    "wrist_px_x": wrist_px[0], "wrist_px_y": wrist_px[1],
                    "index_mcp_px_x": index_mcp_px[0], "index_mcp_px_y": index_mcp_px[1],
                    "wrist_angle": wrist_angle,
                    "neck_angle": neck_ang,
                    "hand_conf": conf,
                    "reproj_px": reproj_px,
                    "n_frames_used": n_used,
                    "occluded": occ,
                    "occ_reason": occ_reason,
                }
                rows.append(row)
            if (ei + 1) % 50 == 0:
                print(f"  ...{ei + 1}/{len(ch)} events  rows={len(rows)}")
    finally:
        if cap is not None:
            cap.release()
        hands.close()

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\n[done] wrote {a.out}")
    print(f"  per-finger rows: {len(out)}")
    print(f"  events with NO hand detected (skipped): {n_event_nohand}")
    if len(out):
        print(f"  rows by occluded flag:\n{out['occluded'].value_counts().to_string()}")
        print(f"  finger distribution:\n{out['finger'].value_counts().to_string()}")
        fe = (out["mp_fret_est"] - out["true_fret"]).abs()
        sa = (out["mp_string_est"] == out["true_string"]).mean()
        print(f"  [raw-MP sanity] fret MAE={fe.mean():.3f}  string acc={sa:.3f}")


if __name__ == "__main__":
    main()
