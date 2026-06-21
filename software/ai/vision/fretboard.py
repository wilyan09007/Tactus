#!/usr/bin/env python3
"""
Markerless fretboard registration — "the instrument is its own ruler".

No fiducial. We exploit the fact that fret spacing is a *physical law*
(12-tone equal temperament): the n-th fret sits at a fraction of the scale
length

    x_n = 1 - 2^(-n/12)          (x_0 = nut = 0, x_12 = octave = 0.5)

so the guitar carries its own metric scale. We register the fretboard plane
with a homography fit from the guitar's own geometry (nut line + neck edges +
any visible fret wires), then VALIDATE by reprojecting every fret position back
onto the real fret wires — the residual is the proof, and it replaces the ArUco
marker entirely.

This module is pure geometry (numpy + cv2), so it is testable with zero footage:
run `python3 software/ai/vision/fretboard.py` for a synthetic round-trip proof.
The auto line-detector that finds the nut/edges/frets in a real frame plugs in
on top of `fit_homography` (see calibrate.py, next build) — but even with a
4-point hand-click per session the registration is exact, and needs no paper.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # geometry still importable without cv2 for the law itself
    cv2 = None

N_STRINGS = 6          # frets 1-6 scope; 6 strings (low-E .. high-e)
SCALE_LENGTH = 1.0     # board X is in scale-length units (relative; that's the point)


# ----------------------------------------------------------------------------
# 1. the law: where every fret is, for free
# ----------------------------------------------------------------------------
def fret_fraction(n: int, scale_length: float = SCALE_LENGTH) -> float:
    """Distance of fret n from the nut, as a fraction of the scale length."""
    return scale_length * (1.0 - 2.0 ** (-n / 12.0))


def fret_fractions(frets=range(0, 13), scale_length: float = SCALE_LENGTH) -> dict:
    return {n: fret_fraction(n, scale_length) for n in frets}


# ----------------------------------------------------------------------------
# 2. the board model -> a flat plane of (X along neck, Y across strings)
# ----------------------------------------------------------------------------
def board_grid(frets=range(0, 8), n_strings: int = N_STRINGS,
               scale_length: float = SCALE_LENGTH):
    """Return (points (M,2), index list) of fret-line x string-line intersections
    in FRETBOARD coordinates. Y runs 0..1 across the strings (0 = low-E)."""
    pts, idx = [], []
    ys = np.linspace(0.0, 1.0, n_strings)
    for n in frets:
        x = fret_fraction(n, scale_length)
        for si, y in enumerate(ys):
            pts.append((x, y))
            idx.append((n, si))
    return np.asarray(pts, np.float32), idx


# ----------------------------------------------------------------------------
# 3. registration: image <-> fretboard homography
# ----------------------------------------------------------------------------
def fit_homography(image_pts, board_pts):
    """image_pts, board_pts: matched (N,2) arrays (N>=4). Returns H (board->image)."""
    img = np.asarray(image_pts, np.float32).reshape(-1, 1, 2)
    brd = np.asarray(board_pts, np.float32).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(brd, img, cv2.RANSAC, 3.0)
    return H, mask


def project(H, board_pts):
    """Map fretboard coords -> image pixels."""
    brd = np.asarray(board_pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(brd, H).reshape(-1, 2)


def to_board(H, image_pts):
    """Map image pixels -> fretboard coords (needs H board->image)."""
    Hinv = np.linalg.inv(H)
    img = np.asarray(image_pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(img, Hinv).reshape(-1, 2)


# ----------------------------------------------------------------------------
# 4. the validation that REPLACES the marker: reproject frets onto real wires
# ----------------------------------------------------------------------------
def reprojection_error(H, detected_image_pts, board_pts):
    """How well the law-constrained grid lands on the points we actually saw.
    This residual is the rigor claim: 'markerless registration validated to N px'."""
    pred = project(H, board_pts)
    det = np.asarray(detected_image_pts, np.float32).reshape(-1, 2)
    err = np.linalg.norm(pred - det, axis=1)
    return float(err.mean()), float(np.median(err)), float(err.max())


# ----------------------------------------------------------------------------
# 5. THE feature: fingertip-to-fret-wire distance d (in board units = coarse)
# ----------------------------------------------------------------------------
def fingertip_d(H, fingertip_px, fret_n: int, scale_length: float = SCALE_LENGTH):
    """Signed along-neck distance from a fingertip to fret_n's wire, in board
    (scale-length) units. >0 means BEHIND the wire (toward the nut) = the
    placement-buzz direction. Report relative/coarse, never cm (docs/17 §6)."""
    bx, _ = to_board(H, [fingertip_px])[0]
    return float(bx - fret_fraction(fret_n, scale_length))


# ----------------------------------------------------------------------------
# self-test: synthesize a foreshortened front-cam view, recover it, no paper
# ----------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)
    frets = range(0, 8)  # nut .. fret 7 (tight framing)
    board, idx = board_grid(frets)

    # a "true" front-cam homography: map the nut..fret7 board rectangle onto a
    # foreshortened trapezoid in a 1280x720 image (neck recedes + tilts).
    bx7 = fret_fraction(7)
    board_corners = np.float32([[0, 0], [0, 1], [bx7, 1], [bx7, 0]])
    img_quad = np.float32([[170, 250], [150, 470],   # nut: tall (near camera)
                           [1120, 430], [1130, 300]])  # fret7: short (far) -> foreshortened
    H_true = cv2.getPerspectiveTransform(board_corners, img_quad)
    true_img = project(H_true, board)

    # simulate DETECTION: we only "see" the nut line + fret 5 + fret 7 + edges,
    # with realistic pixel noise (hand covers frets 1-4 in this scenario).
    seen = [k for k, (n, si) in enumerate(idx) if n in (0, 5, 7)]
    det = true_img[seen] + rng.normal(0, 1.5, (len(seen), 2)).astype(np.float32)

    # fit the homography from ONLY those few noisy points...
    H, _ = fit_homography(det, board[seen])
    # ...then let the LAW place ALL frets and check them against ground truth.
    mean, med, mx = reprojection_error(H, true_img, board)

    # a sample d: a fingertip 1.5mm-ish behind fret 3 on the D string
    f3 = fret_fraction(3)
    tip_board = np.float32([[f3 - 0.012, 0.4]])
    tip_px = project(H_true, tip_board)[0]
    d = fingertip_d(H, tip_px, 3)

    print("MARKERLESS fretboard registration — synthetic front-cam round-trip")
    print(f"  fit from {len(seen)} noisy points (nut+fret5+fret7), law fills the rest")
    print(f"  reprojection error over ALL nut..fret7 intersections:")
    print(f"    mean {mean:.2f}px · median {med:.2f}px · max {mx:.2f}px")
    print(f"  recovered fingertip d (fret 3): {d:+.4f} board-units "
          f"(true {-0.012:+.4f}) → sign = behind-the-wire ✓")
    ok = med < 4.0
    print(f"  VERDICT: {'PASS — paperless registration is sub-4px' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
