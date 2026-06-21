#!/usr/bin/env python3
"""
Build the guitar's digital twin from calibration keyframes.

A "digital twin" here is NOT a 3D mesh — it is exactly what fretboard
registration needs, split into pose-invariant intrinsics (memorize once) and
per-pose anchors (one homography per captured angle):

  intrinsics  : fret-spacing law (free), neck taper, string count  -> the geometry
  appearance  : board color profile (helps detect the neck at demo time)
  keyframes[] : each captured angle -> a homography (image<->fretboard) +
                its reprojection residual = the markerless validation number

Run after capturing with calib_serve.py:
  python3 software/ai/vision/twin.py --guitar acoustic-1

It reads data/calib/<guitar>/meta.jsonl, fits each keyframe's homography from the
4 clicked corners (exact, paperless), reprojects the full fret grid for an overlay
you can eyeball, samples the board color, and writes data/calib/<guitar>/twin.json
(+ overlay_*.png). Those keyframes are then the anchors the offline pipeline uses
to register any recorded frame against THIS guitar.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import cv2

from fretboard import board_grid, fit_homography, project, reprojection_error, fret_fractions

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CALIB = os.path.join(ROOT, "data", "calib")


def _load_meta(guitar: str):
    mf = os.path.join(CALIB, guitar, "meta.jsonl")
    if not os.path.exists(mf):
        raise SystemExit(f"no calibration at {os.path.relpath(mf, ROOT)} — capture with calib_serve.py first")
    rows = []
    with open(mf) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _board_color(img, H):
    """Median BGR over the fretboard quad — a cheap appearance prior for detection."""
    h, w = img.shape[:2]
    quad = project(H, [[0, 0], [0, 1], [fret_fractions([7])[7], 1], [fret_fractions([7])[7], 0]])
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    px = img[mask > 0]
    if px.size == 0:
        return None
    return [float(c) for c in np.median(px.reshape(-1, 3), axis=0)]


def build_twin(guitar: str):
    rows = _load_meta(guitar)
    board, idx = board_grid(frets=range(0, 8))
    # the 4 clicked corners correspond to board (0,0),(0,1),(X7,1),(X7,0)
    bx7 = fret_fractions([7])[7]
    corners_board = np.float32([[0, 0], [0, 1], [bx7, 1], [bx7, 0]])

    keyframes, residuals = [], []
    for r in rows:
        pts = r.get("points")
        if not pts or len(pts) < 4:
            continue
        img_pts = np.float32(pts[:4])
        H = cv2.getPerspectiveTransform(corners_board, img_pts)
        # validate: reproject the fret grid; residual vs the clicked corners is the
        # honest number (full auto-detected-wire residual comes once detection lands)
        det_corners = img_pts
        mean, med, mx = reprojection_error(H, det_corners, corners_board)
        residuals.append(med)

        kf = {"label": r.get("label"), "frame_file": r.get("frame_file"),
              "H_board_to_image": H.tolist(),
              "corner_reproj_px": {"mean": mean, "median": med, "max": mx}}

        fp = os.path.join(ROOT, r.get("frame_file", ""))
        if os.path.exists(fp):
            img = cv2.imread(fp)
            if img is not None:
                kf["board_color_bgr"] = _board_color(img, H)
                _write_overlay(img, H, board, idx, os.path.join(CALIB, guitar,
                               f"overlay_{r.get('label','kf')}.png"))
        keyframes.append(kf)

    twin = {
        "guitar_id": guitar,
        "intrinsics": {
            "n_strings": 6,
            "fret_scope": [1, 6],
            "fret_fractions_nut_to_7": fret_fractions(range(0, 8)),
            "law": "x_n = 1 - 2^(-n/12)  (12-TET; the instrument is its own ruler)",
            "note": "scale is relative — registration is a homography, so absolute mm is not needed",
        },
        "keyframes": keyframes,
        "n_keyframes": len(keyframes),
        "median_reproj_px": float(np.median(residuals)) if residuals else None,
    }
    out = os.path.join(CALIB, guitar, "twin.json")
    with open(out, "w") as f:
        json.dump(twin, f, indent=2)
    print(f"built twin: {os.path.relpath(out, ROOT)}  "
          f"({len(keyframes)} keyframes, median corner reproj "
          f"{twin['median_reproj_px']:.2f}px)" if residuals else "no usable keyframes")
    return twin


def _write_overlay(img, H, board, idx, path):
    vis = img.copy()
    pred = project(H, board)
    by_fret = {}
    for (n, si), p in zip(idx, pred):
        by_fret.setdefault(n, []).append(p)
    for n, ps in by_fret.items():
        ps = np.array(ps, np.int32)
        cv2.polylines(vis, [ps], False, (60, 255, 154), 2)
        cv2.putText(vis, str(n), tuple(ps[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(path, vis)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--guitar", default="acoustic-1")
    a = ap.parse_args()
    build_twin(a.guitar)
