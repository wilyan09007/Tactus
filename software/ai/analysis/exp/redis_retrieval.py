"""Redis Vector Sets semantic retrieval over guitar-mistake events (Tactus).

Redis-track deliverable: "Redis beyond caching" -> vector search / agent memory.

This module builds a standardized audio-feature embedding for every single-note
event (block == 'core-grid'), indexes it into a Redis **Vector Set** (key
``tactus:events``) via the native VADD/VSETATTR commands, and answers
"find events like this" / "you keep muting the A" queries with VSIM.

Why Vector Sets and not redisvl/RediSearch: this Redis build ships the
Vector Sets module (VADD/VSIM/VCARD/VSETATTR/VGETATTR/VEMB) but NOT RediSearch
(FT.* unavailable), so redisvl's index API cannot be used here. We therefore
talk to Vector Sets through raw redis-py ``execute_command`` calls.

Usage
-----
    from redis_retrieval import index_events, search_by_event, search_by_filter
    info = index_events()                       # build + index, returns {'vcard': N, ...}
    res  = search_by_event(some_event_id, k=10) # nearest neighbours + coherence
    res  = search_by_filter(intended_class='muted', string_num=5, k=10)

Run as a script to (re)build the index and print a smoke test:
    .venv/bin/python software/ai/analysis/exp/redis_retrieval.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import redis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Repo root = three levels up from this file (software/ai/analysis/exp/).
_THIS = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", "..", ".."))

EVENTS_CSV = os.path.join(REPO_ROOT, "data", "analysis", "events.csv")
FEATURES_CSV = os.path.join(REPO_ROOT, "data", "analysis", "features_fused.csv")

VSET_KEY = "tactus:events"

# Full enumerated audio-feature set from features_fused.csv. (The task prose says
# "26"; the explicit enumeration spec_* .. mfcc_1..13 is 15 scalar + 13 MFCC = 28.
# We index the full enumerated set.) Vision columns are intentionally excluded.
AUDIO_FEATURES = [
    "spec_centroid", "spec_bandwidth", "spec_flatness", "spec_rolloff", "spec_flux",
    "zcr", "rms", "log_attack_time", "attack_slope", "decay_rate", "hnr",
    "inharmonicity", "buzz_band_ratio", "pitch_cents_dev", "chroma_peak",
] + [f"mfcc_{i}" for i in range(1, 14)]

# Metadata kept per event (stored as VSETATTR JSON, used for filtering/summaries).
META_COLS = ["intended_class", "string_num", "target_fret", "chord_name", "run_id", "block"]

# Single notes only. The chord-stream block is polyphonic and not part of the
# single-note retrieval index.
SINGLE_NOTE_BLOCK = "core-grid"

# Standard guitar string names (string_num: 6=low E .. 1=high e; 0=chord).
STRING_NAMES = {6: "low-E", 5: "A", 4: "D", 3: "G", 2: "B", 1: "high-e", 0: "chord"}


def string_name(n) -> str:
    try:
        return STRING_NAMES.get(int(n), str(n))
    except (TypeError, ValueError):
        return str(n)


# --------------------------------------------------------------------------- #
# Redis connection
# --------------------------------------------------------------------------- #

def get_redis(decode_responses: bool = False) -> redis.Redis:
    """Return a redis-py client (localhost:6379 by default)."""
    return redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        decode_responses=decode_responses,
    )


def _as_text(v) -> str:
    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #

@dataclass
class EmbeddingResult:
    event_ids: list
    vectors: np.ndarray            # (N, dim) float32, L2-normalized rows
    meta: dict                     # event_id -> metadata dict
    dim: int
    used_pca: bool
    feature_names: list = field(default_factory=list)


def build_embeddings(use_pca: bool = True, pca_dims: int = 16) -> EmbeddingResult:
    """Merge events + features, build a standardized audio-feature vector per
    single-note event.

    Steps: select core-grid rows -> median-impute the few NaNs -> StandardScaler
    over all single notes -> optional PCA to ``pca_dims`` -> L2-normalize rows so
    that VSIM cosine == dot product and similarity lives in a clean [-1, 1] range.
    """
    events = pd.read_csv(EVENTS_CSV)
    feats = pd.read_csv(FEATURES_CSV)

    # features_fused.csv already carries the event metadata columns; use it as the
    # source of truth for features and pull any missing meta from events.csv.
    feats = feats[feats["block"] == SINGLE_NOTE_BLOCK].copy()

    # Ensure metadata columns are present (fill from events.csv if needed).
    ev_meta = events.set_index("event_id")
    for col in META_COLS:
        if col not in feats.columns:
            feats[col] = feats["event_id"].map(ev_meta[col])

    feats = feats.reset_index(drop=True)
    event_ids = feats["event_id"].tolist()

    # Feature matrix + median imputation for the handful of NaNs (inharmonicity).
    X = feats[AUDIO_FEATURES].astype(float).copy()
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    Xv = X.values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xv)

    used_pca = bool(use_pca and pca_dims < Xs.shape[1])
    if used_pca:
        pca = PCA(n_components=pca_dims, random_state=0)
        Xs = pca.fit_transform(Xs)
        feature_names = [f"pc_{i+1}" for i in range(pca_dims)]
    else:
        feature_names = list(AUDIO_FEATURES)

    # L2-normalize each row -> cosine similarity == dot product under VSIM.
    norms = np.linalg.norm(Xs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = (Xs / norms).astype(np.float32)

    meta = {}
    for _, row in feats.iterrows():
        eid = row["event_id"]
        cn = row.get("chord_name")
        meta[eid] = {
            "intended_class": _safe(row.get("intended_class")),
            "string_num": _safe_int(row.get("string_num")),
            "string_name": string_name(row.get("string_num")),
            "target_fret": _safe_int(row.get("target_fret")),
            "chord_name": None if pd.isna(cn) else _safe(cn),
            "run_id": _safe(row.get("run_id")),
            "block": _safe(row.get("block")),
        }

    return EmbeddingResult(
        event_ids=event_ids,
        vectors=Xn,
        meta=meta,
        dim=Xn.shape[1],
        used_pca=used_pca,
        feature_names=feature_names,
    )


def _safe(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return v


def _safe_int(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #

def index_events(use_pca: bool = True, pca_dims: int = 16, reset: bool = True) -> dict:
    """Build embeddings and index every single-note event into the Redis Vector
    Set ``tactus:events``.

    For each event: ``VADD tactus:events VALUES <dim> <floats...> <event_id>``
    then ``VSETATTR tactus:events <event_id> <json-metadata>``.

    Returns a small summary dict including the post-index ``VCARD``.
    """
    emb = build_embeddings(use_pca=use_pca, pca_dims=pca_dims)
    r = get_redis()

    if reset:
        r.delete(VSET_KEY)

    for eid, vec in zip(emb.event_ids, emb.vectors):
        args = ["VADD", VSET_KEY, "VALUES", emb.dim]
        args.extend(float(x) for x in vec)
        args.append(eid)
        r.execute_command(*args)
        r.execute_command("VSETATTR", VSET_KEY, eid, json.dumps(emb.meta[eid]))

    vcard = int(r.execute_command("VCARD", VSET_KEY))
    return {
        "vcard": vcard,
        "n_events": len(emb.event_ids),
        "dim": emb.dim,
        "used_pca": emb.used_pca,
        "key": VSET_KEY,
        "match": vcard == len(emb.event_ids),
    }


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

def _get_attr(r: redis.Redis, element: str) -> dict:
    raw = r.execute_command("VGETATTR", VSET_KEY, element)
    if not raw:
        return {}
    try:
        return json.loads(_as_text(raw))
    except (ValueError, TypeError):
        return {}


def _base_rates(r: redis.Redis) -> dict:
    """Index-wide base rates for intended_class and string_num across all members."""
    members = [_as_text(m) for m in r.execute_command("VSIM", VSET_KEY, "ELE",
                                                       _first_element(r),
                                                       "COUNT", r.execute_command("VCARD", VSET_KEY))]
    cls, strg = {}, {}
    total = 0
    for m in members:
        a = _get_attr(r, m)
        total += 1
        cls[a.get("intended_class")] = cls.get(a.get("intended_class"), 0) + 1
        strg[a.get("string_name")] = strg.get(a.get("string_name"), 0) + 1
    return {"total": total,
            "class": {k: v / total for k, v in cls.items()},
            "string": {k: v / total for k, v in strg.items()}}


def _first_element(r: redis.Redis) -> str:
    # VSIM needs a reference element; grab any one via a random member trick.
    # Vector Sets has no SCAN, but VSIM with a known element works; we cache one.
    el = r.execute_command("VRANDMEMBER", VSET_KEY)
    if isinstance(el, list):
        el = el[0]
    return _as_text(el)


@dataclass
class Neighbor:
    event_id: str
    score: float
    intended_class: Optional[str]
    string_num: Optional[int]
    string_name: Optional[str]
    target_fret: Optional[int]
    chord_name: Optional[str]
    run_id: Optional[str]


def _summarize(neighbors: list, base: Optional[dict] = None,
               focus_class: Optional[str] = None,
               focus_string: Optional[str] = None) -> dict:
    """Neighbour composition + coherence-vs-baserate lift."""
    n = len(neighbors)
    cls_counts, str_counts = {}, {}
    for nb in neighbors:
        cls_counts[nb.intended_class] = cls_counts.get(nb.intended_class, 0) + 1
        str_counts[nb.string_name] = str_counts.get(nb.string_name, 0) + 1

    out = {
        "n": n,
        "class_counts": cls_counts,
        "string_counts": str_counts,
    }
    if base and focus_class is not None:
        frac = cls_counts.get(focus_class, 0) / n if n else 0.0
        br = base["class"].get(focus_class, 0.0)
        out["class_coherence"] = {
            "focus_class": focus_class,
            "neighbor_frac": frac,
            "base_rate": br,
            "lift": (frac / br) if br else None,
        }
    if base and focus_string is not None:
        frac = str_counts.get(focus_string, 0) / n if n else 0.0
        br = base["string"].get(focus_string, 0.0)
        out["string_coherence"] = {
            "focus_string": focus_string,
            "neighbor_frac": frac,
            "base_rate": br,
            "lift": (frac / br) if br else None,
        }
    return out


def search_by_event(event_id: str, k: int = 10, include_self: bool = False,
                    with_baserate: bool = True) -> dict:
    """Find the ``k`` events most similar to ``event_id`` (VSIM WITHSCORES),
    join their stored metadata, and summarize neighbour composition + the
    coherence-vs-baserate lift relative to the query's own class/string.
    """
    r = get_redis()
    count = k + 1 if not include_self else k
    raw = r.execute_command("VSIM", VSET_KEY, "ELE", event_id,
                            "WITHSCORES", "COUNT", count)

    pairs = _parse_withscores(raw)
    neighbors = []
    for eid, score in pairs:
        if not include_self and eid == event_id:
            continue
        a = _get_attr(r, eid)
        neighbors.append(Neighbor(
            event_id=eid, score=score,
            intended_class=a.get("intended_class"),
            string_num=a.get("string_num"),
            string_name=a.get("string_name"),
            target_fret=a.get("target_fret"),
            chord_name=a.get("chord_name"),
            run_id=a.get("run_id"),
        ))
        if len(neighbors) >= k:
            break

    qa = _get_attr(r, event_id)
    base = _base_rates(r) if with_baserate else None
    summary = _summarize(neighbors, base,
                         focus_class=qa.get("intended_class"),
                         focus_string=qa.get("string_name"))

    return {
        "query": {"event_id": event_id, **qa},
        "neighbors": neighbors,
        "summary": summary,
    }


def search_by_filter(intended_class: Optional[str] = None,
                     string_num: Optional[int] = None,
                     k: int = 10, n_seeds: int = 5,
                     with_baserate: bool = True) -> dict:
    """Pick events matching a metadata filter (e.g. muted notes on the A string),
    use them as seeds, and report aggregate neighbour coherence vs base rate.

    This is the "you keep muting the A" query: it averages the neighbour
    composition over several matching seed events so the coherence number is not
    a single-event fluke.
    """
    r = get_redis()
    seeds = _members_matching(r, intended_class=intended_class, string_num=string_num)
    if not seeds:
        return {"error": "no events match filter", "filter": locals()}

    seeds = seeds[:n_seeds]
    base = _base_rates(r) if with_baserate else None

    per_seed = []
    agg_cls, agg_str = {}, {}
    total_neighbors = 0
    focus_string = string_name(string_num) if string_num is not None else None

    for seed in seeds:
        res = search_by_event(seed, k=k, with_baserate=False)
        per_seed.append({"seed": seed, "summary": res["summary"],
                         "neighbors": res["neighbors"]})
        for nb in res["neighbors"]:
            agg_cls[nb.intended_class] = agg_cls.get(nb.intended_class, 0) + 1
            agg_str[nb.string_name] = agg_str.get(nb.string_name, 0) + 1
            total_neighbors += 1

    agg = {"n_seeds": len(seeds), "total_neighbors": total_neighbors,
           "class_counts": agg_cls, "string_counts": agg_str}
    if base and intended_class is not None:
        frac = agg_cls.get(intended_class, 0) / total_neighbors if total_neighbors else 0
        br = base["class"].get(intended_class, 0.0)
        agg["class_coherence"] = {"focus_class": intended_class, "neighbor_frac": frac,
                                  "base_rate": br, "lift": (frac / br) if br else None}
    if base and focus_string is not None:
        frac = agg_str.get(focus_string, 0) / total_neighbors if total_neighbors else 0
        br = base["string"].get(focus_string, 0.0)
        agg["string_coherence"] = {"focus_string": focus_string, "neighbor_frac": frac,
                                   "base_rate": br, "lift": (frac / br) if br else None}

    return {"filter": {"intended_class": intended_class, "string_num": string_num,
                       "string_name": focus_string},
            "seeds": seeds, "aggregate": agg, "per_seed": per_seed}


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #

def _parse_withscores(raw) -> list:
    """VSIM ... WITHSCORES returns a flat [el, score, el, score, ...] list."""
    out = []
    for i in range(0, len(raw) - 1, 2):
        eid = _as_text(raw[i])
        try:
            score = float(_as_text(raw[i + 1]))
        except (ValueError, TypeError):
            score = float("nan")
        out.append((eid, score))
    return out


def _members_matching(r: redis.Redis, intended_class=None, string_num=None) -> list:
    """All indexed members whose stored attrs match the given filter."""
    vcard = int(r.execute_command("VCARD", VSET_KEY))
    seed = _first_element(r)
    members = [_as_text(m) for m in r.execute_command("VSIM", VSET_KEY, "ELE", seed,
                                                      "COUNT", vcard)]
    matched = []
    for m in members:
        a = _get_attr(r, m)
        if intended_class is not None and a.get("intended_class") != intended_class:
            continue
        if string_num is not None and a.get("string_num") != int(string_num):
            continue
        matched.append(m)
    return matched


# --------------------------------------------------------------------------- #
# Script entry point: rebuild + smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    info = index_events()
    print("[index_events]", info)
    r = get_redis()
    sample = _first_element(r)
    print("[smoke] search_by_event", sample)
    res = search_by_event(sample, k=10)
    q = res["query"]
    print(f"  query class={q.get('intended_class')} string={q.get('string_name')} "
          f"fret={q.get('target_fret')}")
    for nb in res["neighbors"][:5]:
        print(f"    {nb.score:.3f}  {nb.intended_class:<6} {nb.string_name:<6} "
              f"fret={nb.target_fret} {nb.event_id}")
    print("  summary:", res["summary"])
