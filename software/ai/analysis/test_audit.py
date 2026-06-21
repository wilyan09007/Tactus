#!/usr/bin/env python3
"""
Self-contained smoke test for audit.py — NO pytest.

Builds a small but realistic events DataFrame (the columns audit.py reads, a
subset of schema.EVENT_COLUMNS) and a minimal metrics dict shaped like
collapse.py's output, then exercises audit.run() on BOTH a populated batch and a
fully-empty one. Asserts the report is written, non-trivial, self-contained, and
that the companion audit.json parses.

Run:  .venv/bin/python software/ai/analysis/test_audit.py
"""
import os
import sys
import json
import tempfile

# Flat imports (matches the rest of the pipeline). This file lives in the
# analysis dir, so putting that dir on sys.path makes `import schema, audit` work.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema  # noqa: E402
import audit   # noqa: E402

import pandas as pd  # noqa: E402


def _build_events_df():
    """~30 rows across the 3 core classes / strings 6..1 / frets 1..6.

    Deterministically seeded so the QC / label / coverage numbers are stable:
      * a couple of clipped + silent rows (to drive the Technical QC dots),
      * a controlled fraction of label_fret_match == False (label integrity),
      * a few NaN label_fret_match rows (excluded from the agreement rate),
      * one run with a deliberately wrong note count is implicit via run_id mix.
    """
    rows = []
    classes = schema.CORE_CLASSES            # clean / buzz-light / buzz-placement
    strings = [6, 5, 4, 3, 2, 1]
    frets = schema.FRETS                     # 1..6
    players = ["p01", "p02", "p03"]

    n = 30
    for i in range(n):
        klass = classes[i % len(classes)]
        s_num = strings[i % len(strings)]
        fret = frets[(i // 2) % len(frets)]
        player = players[i % len(players)]
        run_id = "r%02d" % (i // 3)          # ~3 events per run

        # ~10% clipped, ~7% silent (one each early), peak dBFS spread incl. a clip.
        clipped = (i in (4, 17, 26))
        silent = (i in (9, 22))
        peak = -6.0 + (i % 7) * 0.8
        if clipped:
            peak = 0.4                        # over the 0 dBFS ceiling

        # label_fret_match: mostly True, a few False, a couple NaN (not applicable).
        if i in (3, 11, 19):
            match = False
        elif i in (7, 14):
            match = None                      # chord/overflow/no-F0 -> excluded
        else:
            match = True

        rows.append({
            "event_id": "%s#%d" % (run_id, i),
            "run_id": run_id,
            "session_id": "sess-test",
            "player_id": player,
            "intended_class": klass,
            "string_num": s_num,
            "target_fret": fret,
            "finger": 1 + (i % 3),
            "pluck_strength": "medium",
            "audio_peak_dbfs": peak,
            "audio_clipped": clipped,
            "audio_silent": silent,
            "label_fret_match": match,
        })

    df = pd.DataFrame(rows)
    return df


def _build_metrics():
    """Minimal metrics dict matching collapse.py's shape (docs/24 §8):
      split, classes, and per-modality (audio / fused) blocks each carrying a
      pairwise_dprime map (incl. the two study pairs), a 3x3 confusion matrix
      with class_order, plus the V1/V2 verdict booleans.
    """
    order = list(schema.CORE_CLASSES)
    return {
        "split": "lopo",
        "classes": order,
        "audio": {
            "pairwise_dprime": {
                ("clean", "buzz-light"): 1.9,
                # audio is EXPECTED to confuse the buzz pair -> small d'
                ("buzz-light", "buzz-placement"): 0.4,
            },
            "confusion": [[18, 1, 1], [1, 9, 10], [0, 11, 9]],
            "class_order": order,
        },
        "fused": {
            "pairwise_dprime": {
                ("clean", "buzz-light"): 2.3,
                # fusion (+ vision d) is expected to pull the pair apart
                ("buzz-light", "buzz-placement"): 1.6,
            },
            "confusion": [[19, 1, 0], [1, 16, 3], [0, 3, 17]],
            "class_order": order,
        },
        "v1_audio_confuses_buzz_pair": True,
        "v2_fusion_separates_buzz_pair": True,
    }


def main():
    failures = 0

    with tempfile.TemporaryDirectory() as tmp:
        # ---- populated batch ----
        df = _build_events_df()
        metrics = _build_metrics()
        assert len(df) >= 25, "fixture should have ~30 rows, got %d" % len(df)

        path = audit.run(df, metrics, tmp, title="Tactus batch audit (test)")

        assert isinstance(path, str), "run() must return a path string"
        assert os.path.exists(path), "audit.html was not written: %s" % path

        size = os.path.getsize(path)
        assert size > 2048, "audit.html too small (%d bytes); expected > 2KB" % size

        with open(path, "r") as fh:
            html = fh.read()
        low = html.lower()
        assert "<html" in low, "output is not HTML (no <html tag)"
        assert "coverage" in low, "report missing the Coverage section"
        assert "separability" in low, "report missing the Separability section"
        # Self-contained: no external resource fetches. (The SVG xmlns URI
        # legitimately contains "http://", so check for actual external refs:
        # remote src/href and <link>/<script src> tags, not the namespace.)
        for bad in ('src="http', "src='http", 'href="http', "href='http",
                    "<link", "<script src", "//cdn", "googleapis"):
            assert bad not in low, "report not self-contained (found %r)" % bad
        # Charts must be embedded as base64 data URIs, not files.
        assert "data:image/png;base64," in low, \
            "expected base64-embedded PNG charts"

        json_path = os.path.join(tmp, "audit.json")
        assert os.path.exists(json_path), "companion audit.json not written"
        with open(json_path, "r") as fh:
            blob = json.load(fh)
        assert isinstance(blob, dict), "audit.json should be a JSON object"
        # Spot-check a few computed summaries made it into the JSON.
        for key in ("technical", "label_integrity", "separability",
                    "coverage", "recommendations"):
            assert key in blob, "audit.json missing computed key %r" % key
        assert blob["separability"]["fused_separates_buzz_pair"] is True, \
            "v2=True should surface as fused_separates_buzz_pair"
        assert len(blob["recommendations"]) >= 1, "expected >=1 recommendation"

        # ---- empty / minimal inputs must NOT raise ----
        empty_dir = os.path.join(tmp, "empty")
        empty_df = pd.DataFrame()
        empty_path = audit.run(empty_df, {}, empty_dir)
        assert os.path.exists(empty_path), \
            "empty-input run() did not write a file"
        with open(empty_path) as fh:
            empty_html = fh.read().lower()
        assert "<html" in empty_html, "empty report is not HTML"
        assert "no data" in empty_html or "no events" in empty_html, \
            "empty report should say 'no data'/'no events' somewhere"
        assert os.path.exists(os.path.join(empty_dir, "audit.json")), \
            "empty run() must still write audit.json"

        # ---- DataFrame with MISSING columns must NOT raise ----
        partial_dir = os.path.join(tmp, "partial")
        partial_df = pd.DataFrame({"intended_class": schema.CORE_CLASSES * 2})
        partial_path = audit.run(partial_df, {"split": "lopo"}, partial_dir)
        assert os.path.exists(partial_path), \
            "partial-column run() did not write a file"

    if failures:
        print("FAIL: audit (%d assertion group failures)" % failures)
        sys.exit(1)
    print("PASS: audit")


if __name__ == "__main__":
    main()
