#!/usr/bin/env python3
"""
End-to-end integration smoke test for the analysis pipeline.

Builds a synthetic two-player batch (make_demo_data), runs the full orchestrator
on it in a temp tree, and asserts the real artifacts land: events.csv, a valid
metrics.json with both audio and fused studies under a leave-one-player-out split,
and a non-trivial self-contained audit.html. No pytest, no mediapipe, no network.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema           # noqa: E402
import make_demo_data   # noqa: E402
import run_pipeline     # noqa: E402


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        out = os.path.join(tmp, "analysis")
        make_demo_data.build(raw, players=("p1", "p2"))

        res = run_pipeline.run_batch(raw_dir=raw, out_dir=out, guitar=None)

        assert res["events"] > 0, "no events produced"
        assert res["players"] == 2, "expected 2 players, got %r" % res["players"]

        # artifacts exist
        for name in ("events.csv", "features_fused.csv", "metrics.json"):
            p = os.path.join(out, name)
            assert os.path.exists(p) and os.path.getsize(p) > 0, "missing/empty %s" % name

        # metrics shape
        with open(os.path.join(out, "metrics.json")) as f:
            metrics = json.load(f)
        assert metrics["split"] == "lopo", "expected lopo split, got %r" % metrics.get("split")
        for fs in ("audio", "fused"):
            assert fs in metrics and "confusion" in metrics[fs], "metrics missing %s study" % fs

        # audit html: exists, self-contained, non-trivial
        ap = res["audit"]
        assert ap and os.path.exists(ap), "no audit.html"
        html = open(ap, encoding="utf-8").read()
        assert len(html) > 2048, "audit.html too small (%d bytes)" % len(html)
        low = html.lower()
        assert "<html" in low and "coverage" in low and "separability" in low, \
            "audit.html missing expected sections"
        # self-contained: no external asset fetches (the svg xmlns URI is fine — it's
        # an identifier, not a fetch). charts must be inline (base64 png and/or svg).
        for ext in ('src="http', "src='http", "<link", "<script src", "cdn."):
            assert ext not in low, "audit.html not self-contained: found %r" % ext
        assert "data:image/png;base64," in low or "<svg" in low, "expected inline charts"

        # V1 sanity: on synthetic audio the two buzz causes should be hard to split
        bp = _buzz_pair_dprime(metrics["audio"])
        print("audio buzz-pair d' = %s  (V1: should be low)" % bp)

    print("PASS: pipeline  (%d events, 2 players, audit + metrics produced)" % res["events"])


def _buzz_pair_dprime(study):
    """Pull d'(buzz-light, buzz-placement) out of whatever key shape collapse used."""
    pd_ = study.get("pairwise_dprime", {})
    for k, v in pd_.items():
        key = k.lower()
        if "buzz-light" in key and "buzz-placement" in key:
            return v
    return None


if __name__ == "__main__":
    test_end_to_end()
