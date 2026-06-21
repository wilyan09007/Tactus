#!/usr/bin/env python3
"""Tactus — LIVE Redis Vector-Sets memory service (the in-session loop).

`redis_retrieval.py` proved the offline index. This wraps it in a tiny HTTP
service so the live interface can, on every note the player plays:

    1. extract the SAME 28-D multi-dimensional aux-audio representation used to
       train the system (`features_audio._extract_one`, librosa),
    2. project it onto the audio-feature **eigenbasis** (PCA; the principal axes
       are literally the eigenvectors of the standardized feature covariance) —
       16-D for retrieval, the first 3 of those for the live 3-D view,
    3. `VSIM` against the Redis Vector Set `tactus:events` (similarity search runs
       *inside* Redis, sub-ms approximate-NN) to vote the note's class/string, and
    4. `VADD` it into the growing per-session set `tactus:live` so the vector
       space literally builds up in Redis as you play, and error clusters emerge.

Redis is the long-term memory + the similarity engine here, not a cache.

Run:
    REDIS_PORT=6380 python3 software/ai/analysis/exp/redis_live.py   # serves :8771

Endpoints (CORS-open for the local UI):
    GET  /health
    GET  /projection                 -> 432 events as 2-D eigen-coords (legacy viz)
    GET  /space3d                     -> 432 events as 3-D eigen-coords + labelled regions
    GET  /mistake?cls=muted&string=5  -> recurring-mistake retrieval + insight (legacy viz)
    POST /embed  {sr, pcm_b64 | samples}  -> live note -> 3-D coords + VSIM class vote + VADD
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                  # exp/  -> redis_retrieval
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))  # analysis/ -> features_audio, schema
import redis_retrieval as R   # noqa: E402
import features_audio as FA   # noqa: E402  (librosa-backed _extract_one)

PORT = int(os.environ.get("TACTUS_REDIS_LIVE_PORT", "8771"))
LIVE_KEY = "tactus:live"                  # the growing per-session vector set
EMB_DIM = 16                              # retrieval embedding dim (matches tactus:events)

CLASS_COLOR = {"clean": "#3DDC84", "buzz": "#FF9F43", "muted": "#8A7BFF"}
ERROR_CLASSES = {"buzz", "muted"}


# --------------------------------------------------------------------------- #
# Eigenbasis projector: ONE PCA(16) over the 28-D aux-audio features.
#   - the L2-normalized 16-D rows == what lives in Redis (tactus:events)
#   - the first 3 components == the live 3-D semantic space we render
# Replicates redis_retrieval.build_embeddings preprocessing so live notes land
# in the same basis as the index (StandardScaler -> PCA(16, random_state=0)).
# --------------------------------------------------------------------------- #
class Projector:
    def __init__(self):
        feats = pd.read_csv(R.FEATURES_CSV)
        feats = feats[feats["block"] == R.SINGLE_NOTE_BLOCK].reset_index(drop=True)
        X = feats[R.AUDIO_FEATURES].astype(float)
        self.medians = X.median(numeric_only=True)
        Xs = self.scaler_fit(X.fillna(self.medians).values)
        self.pca = PCA(n_components=EMB_DIM, random_state=0).fit(Xs)
        emb = self.pca.transform(Xs)                       # (N,16)
        self.var = [float(v) for v in self.pca.explained_variance_ratio_[:3]]
        c3 = emb[:, :3]
        self.ids = feats["event_id"].tolist()
        self.coord = {eid: [float(c3[i, 0]), float(c3[i, 1]), float(c3[i, 2])]
                      for i, eid in enumerate(self.ids)}
        self.meta = {}
        for _, row in feats.iterrows():
            self.meta[row["event_id"]] = {
                "cls": R._safe(row.get("intended_class")),
                "string": R.string_name(row.get("string_num")),
                "string_num": R._safe_int(row.get("string_num")),
                "fret": R._safe_int(row.get("target_fret")),
            }
        self._regions = self._build_regions()

    def scaler_fit(self, X):
        self.scaler = StandardScaler().fit(X)
        return self.scaler.transform(X)

    def _centroid(self, ids):
        if not ids:
            return [0, 0, 0]
        a = np.array([self.coord[i] for i in ids])
        return [float(a[:, 0].mean()), float(a[:, 1].mean()), float(a[:, 2].mean())]

    def _build_regions(self):
        regions = []
        # one region per articulation class (the error clusters are buzz/muted)
        for cls in ("clean", "buzz", "muted"):
            ids = [i for i in self.ids if self.meta[i]["cls"] == cls]
            if ids:
                regions.append({"kind": "class", "label": cls, "color": CLASS_COLOR[cls],
                                "centroid": self._centroid(ids), "n": len(ids)})
        # one label per string (the note/pitch axis of meaning)
        for sn in (6, 5, 4, 3, 2, 1):
            ids = [i for i in self.ids if self.meta[i]["string_num"] == sn]
            if ids:
                regions.append({"kind": "string", "label": R.string_name(sn) + " string",
                                "color": "#ffffff", "centroid": self._centroid(ids), "n": len(ids)})
        return regions

    def points3d(self):
        out = []
        for eid in self.ids:
            m = self.meta[eid]; c = self.coord[eid]
            out.append({"id": eid, "x": c[0], "y": c[1], "z": c[2],
                        "cls": m["cls"], "string": m["string"], "color": CLASS_COLOR.get(m["cls"], "#888")})
        return out

    def points(self):  # legacy 2-D
        return [{"id": e, "x": self.coord[e][0], "y": self.coord[e][1],
                 "cls": self.meta[e]["cls"], "string": self.meta[e]["string"]} for e in self.ids]

    def exemplars(self, cls=None, string_num=None):
        return [e for e in self.ids
                if (cls is None or self.meta[e]["cls"] == cls)
                and (string_num is None or self.meta[e]["string_num"] == string_num)]

    # --- live note -> (3-D coords, L2-normalized 16-D query vector, raw features) ---
    def embed_live(self, y, sr):
        feats = FA._extract_one(np.asarray(y, dtype=np.float32), int(sr))
        vec = np.array([feats.get(n, np.nan) for n in R.AUDIO_FEATURES], dtype=float)
        for i, n in enumerate(R.AUDIO_FEATURES):
            if not np.isfinite(vec[i]):
                vec[i] = float(self.medians[n])
        Xs = self.scaler.transform(vec.reshape(1, -1))
        emb = self.pca.transform(Xs)[0]
        coords3 = [float(emb[0]), float(emb[1]), float(emb[2])]
        norm = float(np.linalg.norm(emb)) or 1.0
        q = (emb / norm).astype(float)
        return coords3, q, feats


PROJ: Projector | None = None
_LIVE_N = 0


def ensure_data():
    """Build data/analysis/{features_fused,events}.csv from the shared offline
    data (data/analysis/all/) if they're absent. data/analysis/ is gitignored, so
    this lets the service run after a fresh checkout without shipping data files."""
    need_fused = not os.path.exists(R.FEATURES_CSV)
    need_ev = not os.path.exists(R.EVENTS_CSV)
    if not (need_fused or need_ev):
        return
    all_dir = os.path.join(R.REPO_ROOT, "data", "analysis", "all")
    aev = os.path.join(all_dir, "events.csv")
    afa = os.path.join(all_dir, "features_audio.csv")
    if not (os.path.exists(aev) and os.path.exists(afa)):
        raise SystemExit("[redis_live] missing data/analysis/all/{events,features_audio}.csv "
                         "(shared offline data) — cannot build the index.")
    ev = pd.read_csv(aev)
    fa = pd.read_csv(afa)
    meta_cols = ["block", "intended_class", "string_num", "target_fret",
                 "run_id", "session_id", "player_id"]
    meta = ev[["event_id"] + [c for c in meta_cols if c in ev.columns]]
    fused = fa.merge(meta, on="event_id", how="inner")
    fused = fused[fused["block"] == R.SINGLE_NOTE_BLOCK].copy()
    if "chord_name" not in fused.columns:
        fused["chord_name"] = ""
    os.makedirs(os.path.dirname(R.FEATURES_CSV), exist_ok=True)
    if need_fused:
        fused.to_csv(R.FEATURES_CSV, index=False)
    if need_ev:
        ev2 = ev.copy()
        if "chord_name" not in ev2.columns:
            ev2["chord_name"] = ""
        ev2.to_csv(R.EVENTS_CSV, index=False)
    print("[redis_live] prepared features_fused.csv (%d core-grid notes) + events.csv" % len(fused))


def ensure_index():
    r = R.get_redis()
    try:
        vcard = int(r.execute_command("VCARD", R.VSET_KEY))
    except Exception:
        vcard = 0
    if vcard == 0:
        R.index_events()
        vcard = int(r.execute_command("VCARD", R.VSET_KEY))
    return vcard


# --------------------------------------------------------------------------- #
# Live embed: a played note -> 3-D point + VSIM class vote + VADD to tactus:live
# --------------------------------------------------------------------------- #
def embed_note(y, sr, requested=None) -> dict:
    global _LIVE_N
    r = R.get_redis()
    coords3, q, feats = PROJ.embed_live(y, sr)

    # too quiet / unvoiced -> ignore (don't pollute the space)
    rms = feats.get("rms")
    if rms is None or not np.isfinite(rms) or rms < 1e-3:
        return {"silent": True}

    # REAL Redis similarity search by the raw query vector (VSIM ... VALUES)
    args = ["VSIM", R.VSET_KEY, "VALUES", EMB_DIM]
    args += [float(x) for x in q]
    args += ["WITHSCORES", "COUNT", 12]
    pairs = R._parse_withscores(r.execute_command(*args))

    neighbors, cls_votes, str_votes = [], {}, {}
    for eid, score in pairs:
        m = PROJ.meta.get(eid, {})
        c = PROJ.coord.get(eid)
        if not c:
            continue
        neighbors.append({"x": c[0], "y": c[1], "z": c[2], "score": round(float(score), 3),
                          "cls": m.get("cls"), "string": m.get("string")})
        cls_votes[m.get("cls")] = cls_votes.get(m.get("cls"), 0) + 1
        str_votes[m.get("string")] = str_votes.get(m.get("string"), 0) + 1

    pred_cls = max(cls_votes, key=cls_votes.get) if cls_votes else None
    pred_str = max(str_votes, key=str_votes.get) if str_votes else None

    # VADD into the growing session set -> the space literally accrues in Redis
    _LIVE_N += 1
    live_id = "live:%d:%d" % (int(time.time()), _LIVE_N)
    vadd = ["VADD", LIVE_KEY, "VALUES", EMB_DIM] + [float(x) for x in q] + [live_id]
    try:
        r.execute_command(*vadd)
        r.execute_command("VSETATTR", LIVE_KEY, live_id,
                          json.dumps({"pred_class": pred_cls, "pred_string": pred_str}))
        live_vcard = int(r.execute_command("VCARD", LIVE_KEY))
    except Exception:
        live_vcard = _LIVE_N

    return {
        "point": {"x": coords3[0], "y": coords3[1], "z": coords3[2],
                  "cls": pred_cls, "string": pred_str, "color": CLASS_COLOR.get(pred_cls, "#FF4D8D"),
                  "id": live_id},
        "neighbors": neighbors,
        "pred_class": pred_cls, "pred_string": pred_str,
        "is_error": pred_cls in ERROR_CLASSES,
        "requested": requested,
        "match": (requested is None) or (pred_cls == requested.get("cls")
                                         and (requested.get("string") in (None, pred_str))),
        "live_vcard": live_vcard,
        "f0_hint": feats.get("pitch_cents_dev"),
    }


# --------------------------------------------------------------------------- #
# Legacy 2-D recurring-mistake query (kept for web/redis-memory.html)
# --------------------------------------------------------------------------- #
def live_mistake(intended_class: str, string_num: int, k: int = 10) -> dict:
    r = R.get_redis()
    sname = R.string_name(string_num)
    cands = sorted(PROJ.exemplars(cls=intended_class, string_num=string_num)
                   or PROJ.exemplars(cls=intended_class))
    if not cands:
        return {"error": "no matching history", "intended_class": intended_class, "string_num": string_num}
    seed = cands[len(cands) // 2]
    res = R.search_by_event(seed, k=k, with_baserate=False)
    neighbors = []
    for nb in res["neighbors"]:
        c = PROJ.coord.get(nb.event_id)
        if not c:
            continue
        neighbors.append({"x": c[0], "y": c[1], "score": round(float(nb.score), 3),
                          "cls": nb.intended_class, "string": nb.string_name})
    prof = R.search_by_filter(intended_class=intended_class, string_num=string_num,
                              k=k, n_seeds=50, with_baserate=True)
    agg = prof.get("aggregate", {})
    cc = agg.get("class_coherence", {}) or {}
    sc = agg.get("string_coherence", {}) or {}
    class_frac = cc.get("neighbor_frac", 0.0) or 0.0
    class_lift = cc.get("lift") or 0.0
    string_lift = sc.get("lift") or 0.0
    verb = {"muted": "muting", "buzz": "buzzing", "clean": "playing"}.get(intended_class, intended_class)
    headline = ("You keep %s the %s string \u2014 %d%% of your nearest mistakes match (%.1f\u00d7 chance)"
                % (verb, sname, round(class_frac * 100), class_lift)) if class_lift else \
               ("Logged a %s on the %s string" % (intended_class, sname))
    sx = PROJ.coord[seed]
    return {"point": {"x": sx[0], "y": sx[1], "cls": intended_class, "string": sname, "id": seed},
            "neighbors": neighbors,
            "insight": {"headline": headline, "class_lift": round(float(class_lift), 2),
                        "string_lift": round(float(string_lift), 2), "class_frac": round(float(class_frac), 2),
                        "k": k, "total_neighbors": agg.get("total_neighbors", 0)},
            "vcard": int(r.execute_command("VCARD", R.VSET_KEY))}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send({}, 204)

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            self._send({"ok": True, "vcard": ensure_index(), "pca_var": PROJ.var,
                        "live_vcard": _safe_vcard()})
        elif u.path == "/projection":
            self._send({"points": PROJ.points(), "var": PROJ.var, "n": len(PROJ.ids)})
        elif u.path == "/space3d":
            self._send({"points": PROJ.points3d(), "regions": PROJ._regions,
                        "var": PROJ.var, "n": len(PROJ.ids)})
        elif u.path == "/mistake":
            q = parse_qs(u.query)
            self._send(live_mistake((q.get("cls") or ["muted"])[0],
                                    int((q.get("string") or ["5"])[0])))
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", "0") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if u.path == "/embed":
            y = _decode_audio(body)
            if y is None or len(y) < 256:
                return self._send({"error": "no audio"}, 400)
            req = body.get("requested")
            try:
                self._send(embed_note(y, int(body.get("sr", 48000)), requested=req))
            except Exception as e:
                self._send({"error": str(e)}, 500)
        elif u.path == "/mistake":
            cls = body.get("intended_class") or body.get("cls") or "muted"
            try:
                s = int(body.get("string_num") or body.get("string") or 5)
            except Exception:
                s = 5
            self._send(live_mistake(cls, s))
        else:
            self._send({"error": "not found"}, 404)


def _decode_audio(body):
    """Accept {pcm_b64: base64 Int16LE} or {samples: [float]}; return float32 [-1,1]."""
    if body.get("pcm_b64"):
        try:
            raw = base64.b64decode(body["pcm_b64"])
            return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)
        except Exception:
            return None
    if isinstance(body.get("samples"), list):
        return np.asarray(body["samples"], dtype=np.float32)
    return None


def _safe_vcard():
    try:
        return int(R.get_redis().execute_command("VCARD", LIVE_KEY))
    except Exception:
        return 0


def main():
    global PROJ
    ensure_data()
    PROJ = Projector()
    vcard = ensure_index()
    print("[redis_live] index=%d  live=%d  PCA var=%.2f/%.2f/%.2f  serving :%d"
          % (vcard, _safe_vcard(), PROJ.var[0], PROJ.var[1], PROJ.var[2], PORT))
    ThreadingHTTPServer(("localhost", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
