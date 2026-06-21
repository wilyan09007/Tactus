#!/usr/bin/env python3
"""
Wiring test for features_vision.py — NO pytest, NO mediapipe.

The point is NOT to test mediapipe (absent here, and the module degrades to
all-NaN without it). It is to prove the fretboard-math WIRING with an INJECTED
landmark provider: synthesize a twin homography from known corners, feed known
fingertip pixels (obtained by projecting known board points), and verify the
features that come back match the real fretboard engine — especially the SIGN
of d_active, the crux buzz-cause feature.

Run:  .venv/bin/python software/ai/analysis/test_features_vision.py
"""
import json
import os
import tempfile

import numpy as np
import pandas as pd

import schema
schema.on_path()
import fretboard
import features_vision


def _build_twin(tmpdir):
    """Synthetic twin: 4 image corners (a foreshortened-ish rectangle) matched to
    board corners nut..fret7. Returns (twin_path, H_board_to_image)."""
    X7 = 1.0 - 2.0 ** (-7.0 / 12.0)
    # board corners: (0,0)=nut/low-E, (0,1)=nut/high-e, (X7,1)=f7/high-e, (X7,0)=f7/low-E
    board_pts = np.float32([[0, 0], [0, 1], [X7, 1], [X7, 0]])
    # image corners in the SAME order. image y=300 -> low-E, y=100 -> high-e.
    image_pts = np.float32([[100, 300], [100, 100], [500, 100], [500, 300]])

    H, _ = fretboard.fit_homography(image_pts, board_pts)   # board -> image (RANSAC, exact for 4 pts)

    twin = {
        "intrinsics": {"n_strings": 6},
        "keyframes": [
            {"H_board_to_image": H.tolist(),
             "corner_reproj_px": {"mean": 0.1, "median": 0.1, "max": 0.2}},
        ],
    }
    twin_path = os.path.join(tmpdir, "twin.json")
    with open(twin_path, "w") as f:
        json.dump(twin, f)
    return twin_path, np.asarray(H, np.float32)


def _make_provider(H):
    """Stub provider returning a known index-fingertip pixel per event.

    event 'v#0': index tip placed BEHIND fret 1's wire (toward the nut) on the
                 low-E string. In board coords that is bx < fret_fraction(1), so
                 fretboard.fingertip_d returns bx - frac < 0  => NEGATIVE d.
    event 'v#1': index tip placed roughly ON fret 3's wire (bx ~ fret_fraction(3))
                 to sanity-check fret_est lands on a small integer near 3.

    Pixels are produced by projecting the known board point with fretboard.project,
    so the test exercises the real to_board/fingertip_d round-trip."""
    f1 = fretboard.fret_fraction(1)
    f3 = fretboard.fret_fraction(3)
    # board points (X along neck, Y across strings; Y=0 = low-E = string 6).
    tip_behind_f1 = fretboard.project(H, [[f1 - 0.02, 0.0]])[0]   # toward nut of fret 1
    tip_on_f3 = fretboard.project(H, [[f3, 0.0]])[0]              # on fret 3 wire

    by_event = {
        "v#0": {"tips": {"index": (float(tip_behind_f1[0]), float(tip_behind_f1[1]))},
                "conf": 0.91},
        "v#1": {"tips": {"index": (float(tip_on_f3[0]), float(tip_on_f3[1]))},
                "conf": 0.88},
    }

    # the provider is keyed by which call it is; we stash the expected event_id on
    # the events so we can route. Simpler: route by the requested pixel? Instead we
    # use a small closure with a call counter mapped to event order.
    seq = ["v#0", "v#1"]
    state = {"i": 0}

    def provider(video_path, time_s):
        if not video_path:
            return None
        eid = seq[min(state["i"], len(seq) - 1)]
        state["i"] += 1
        return by_event[eid]

    return provider


def main():
    with tempfile.TemporaryDirectory() as tmp:
        twin_path, H = _build_twin(tmp)
        provider = _make_provider(H)

        events_df = pd.DataFrame([
            {schema.EVENT_ID: "v#0", "string_num": 6, "target_fret": 1,
             "finger": "index", "onset_s": 0.0, "video_path": "dummy.mp4"},
            {schema.EVENT_ID: "v#1", "string_num": 6, "target_fret": 3,
             "finger": "index", "onset_s": 0.0, "video_path": "dummy.mp4"},
            {schema.EVENT_ID: "v#2", "string_num": 6, "target_fret": 1,
             "finger": "index", "onset_s": 0.0, "video_path": None},  # no video -> NaN row
        ])

        out = features_vision.run(events_df, twin_path, landmarks_provider=provider)

        # --- contract: exact columns, exact order ---
        expected_cols = [schema.EVENT_ID] + schema.VISION_FEATURES
        assert list(out.columns) == expected_cols, \
            f"columns mismatch:\n got {list(out.columns)}\n exp {expected_cols}"
        assert len(out) == len(events_df), "one row per event expected"
        assert list(out[schema.EVENT_ID]) == ["v#0", "v#1", "v#2"], "event_id order/identity preserved"

        r0 = out.iloc[0]
        r1 = out.iloc[1]
        r2 = out.iloc[2]

        # --- d_active finite for events with a hand ---
        assert np.isfinite(r0["d_active"]), "v#0 d_active must be finite"
        assert np.isfinite(r1["d_active"]), "v#1 d_active must be finite"

        # --- SIGN: tip toward the nut of fret 1 => negative d (verified engine
        #     convention: fingertip_d = bx - fret_fraction(n); behind/toward-nut
        #     means bx < frac => d < 0; this is the placement-buzz direction). ---
        assert r0["d_active"] < 0, \
            f"v#0 tip is behind fret-1 wire (toward nut) => d_active must be < 0, got {r0['d_active']}"
        # cross-check directly against the engine for the same pixel:
        f1 = fretboard.fret_fraction(1)
        tip_px0 = list(fretboard.project(H, [[f1 - 0.02, 0.0]])[0])
        assert np.isclose(r0["d_active"], fretboard.fingertip_d(H, tip_px0, 1), atol=1e-4), \
            "d_active must equal fretboard.fingertip_d for the injected pixel"

        # --- fret_est: v#0 nearest wire is fret 1 (small int), v#1 ~ fret 3 ---
        assert float(r0["fret_est"]).is_integer(), "fret_est is an integer-valued float"
        assert 0 <= r0["fret_est"] <= 7, f"fret_est in 0..7, got {r0['fret_est']}"
        assert r0["fret_est"] == 1, f"v#0 fret_est should be 1, got {r0['fret_est']}"
        assert r1["fret_est"] == 3, f"v#1 fret_est should be 3, got {r1['fret_est']}"

        # --- string_est: low-E (Y=0) => string 6 ---
        assert r0["string_est"] == 6, f"v#0 string_est should be 6 (low-E), got {r0['string_est']}"

        # --- d_active mirrors the active finger's per-finger column (index) ---
        assert np.isclose(r0["d_active"], r0["d_index"]), "active finger is index => d_active == d_index"

        # --- reproj_px carried from the chosen keyframe ---
        assert np.isclose(r0["reproj_px"], 0.1), f"reproj_px should be the keyframe median 0.1, got {r0['reproj_px']}"

        # --- joints absent => curl/wrist NaN; neck_angle is from H (finite) ---
        assert np.isnan(r0["finger_curl_active"]), "no joints => finger_curl_active NaN (v1)"
        assert np.isnan(r0["wrist_angle"]), "no joints => wrist_angle NaN (v1)"
        assert np.isfinite(r0["neck_angle"]), "neck_angle derives from H, should be finite"
        assert np.isclose(r0["hand_conf"], 0.91), "hand_conf from provider"

        # --- no-video event => all VISION_FEATURES NaN, event_id preserved ---
        assert r2[schema.EVENT_ID] == "v#2"
        for c in schema.VISION_FEATURES:
            assert np.isnan(r2[c]), f"no-video event must have NaN {c}, got {r2[c]}"

    print("PASS: features_vision")


if __name__ == "__main__":
    main()
