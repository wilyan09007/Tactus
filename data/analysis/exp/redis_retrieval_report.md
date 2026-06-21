# Tactus — Redis Vector Search over Guitar-Mistake Events

**Redis track deliverable: "Redis beyond caching" → vector search / agent memory / context retrieval.**

Tactus is a Deaf-accessible guitar coach. A Deaf player cannot hear that they just
muted the A string — so the coach has to *recognize the mistake from signal* and,
crucially, **remember that it keeps happening**. This module turns every recorded
note into a vector, stores it in Redis, and answers two questions:

1. *"Find events like this one."* (nearest-neighbour retrieval)
2. *"You keep muting the A."* (recurring-mistake detection = **agent memory**)

Both are powered by **Redis Vector Sets** (native `VADD` / `VSIM` / `VSETATTR`),
not by an external vector DB.

---

## 1. What was built

| Component | Location |
|---|---|
| Reusable module | `software/ai/analysis/exp/redis_retrieval.py` |
| This report | `data/analysis/exp/redis_retrieval_report.md` |
| Redis Vector Set key | `tactus:events` |

Public API:

```python
index_events(use_pca=True, pca_dims=16)   # build + index, returns {'vcard': ...}
search_by_event(event_id, k=10)           # "find events like this"
search_by_filter(intended_class='muted', string_num=5, k=10, n_seeds=5)  # "you keep muting the A"
```

### Engine: native Redis Vector Sets (not RediSearch / redisvl)

This Redis build ships the **Vector Sets** module (`VADD`, `VSIM`, `VCARD`,
`VSETATTR`, `VGETATTR`, `VEMB`, `VRANDMEMBER`) but **not RediSearch** (`FT.*` is
unavailable). So `redisvl`'s index API does not apply here. We talk to Vector
Sets through raw `redis-py` `execute_command` calls. We verified the primitives
on a 3-vector toy first: `VSIM ... WITHSCORES` returns cosine similarity in
`[-1, 1]` (1.0 = identical), exactly as needed. **No brute-force fallback was
required — native Vector Sets worked end to end.**

### Embedding

- **Source**: `events.csv` ⋈ `features_fused.csv`, restricted to single-note
  events (`block == 'core-grid'`). The polyphonic `chord-stream` block is out of
  scope for the single-note index.
- **Features**: the full enumerated audio set — `spec_centroid, spec_bandwidth,
  spec_flatness, spec_rolloff, spec_flux, zcr, rms, log_attack_time,
  attack_slope, decay_rate, hnr, inharmonicity, buzz_band_ratio,
  pitch_cents_dev, chroma_peak, mfcc_1..13` = **28 dims**. (The task prose says
  "26"; the explicit enumeration is 15 scalar + 13 MFCC = 28. Vision columns are
  all-NaN and excluded.)
- **Cleaning**: 3 NaNs (in `inharmonicity`) median-imputed.
- **Standardize → PCA(16) → L2-normalize** each row, so `VSIM` cosine == dot
  product. PCA-16 retains **93.5%** of variance.
- **Metadata** stored per event via `VSETATTR` (JSON): `intended_class`,
  `string_num`, `string_name`, `target_fret`, `chord_name`, `run_id`, `block`.

This is a **retrieval index**, not a held-out accuracy claim — the scaler/PCA
are fit over all indexed single notes by design.

---

## 2. Index confirmation — `VCARD`

```
index_events() -> {'vcard': 432, 'n_events': 432, 'dim': 16, 'used_pca': True,
                   'key': 'tactus:events', 'match': True}
```

**`VCARD tactus:events` = 432 = number of single-note events indexed.** ✅

Dataset composition (perfectly balanced):

- 432 single notes = **144 buzz / 144 clean / 144 muted**.
- 6 strings × 72 notes each (24 per class per string).
- **Base rates**: each class = **33.3%**, each string = **16.7%**.

These base rates are what the semantic-quality metric is measured against.

---

## 3. Retrieval examples (`search_by_event`, k=10)

Score = `VSIM` cosine similarity (1.0 = identical timbre).

### Example A — a **muted A** note → neighbours are overwhelmingly muted

Query `s5_16_muted_plkmedium_aditya_006#5` (class=**muted**, string=**A**, fret 6):

| score | class | string | fret | event |
|---|---|---|---|---|
| 0.970 | muted | D | 6 | s4_16_muted…008#5 |
| 0.941 | buzz | B | 2 | s2_16_buzz…033#1 |
| 0.902 | muted | G | 6 | s3_16_muted…030#5 |
| 0.896 | muted | high-e | 1 | s1_16_muted…017#0 |
| 0.889 | muted | low-E | 6 | s6_16_muted…001#5 |
| … | | | | |

→ **9/10 neighbours are muted** (base 33%, **lift 2.7×**). The muting *timbre*
generalizes across strings, so this particular seed's neighbours span many
strings — which is exactly why the recurring-mistake query (§4) averages over
several seeds instead of trusting one.

### Example B — a **clean B** note → neighbours concentrate on the B string

Query `s2_16_clean_plkmedium_aditya_013#4` (class=**clean**, string=**B**, fret 5):

| score | class | string | fret | event |
|---|---|---|---|---|
| 0.908 | buzz | high-e | 4 | s1_16_buzz…035#3 |
| 0.903 | buzz | B | 5 | s2_16_buzz…033#4 |
| 0.902 | clean | B | 5 | s2_16_clean…031#4 |
| 0.899 | clean | B | 6 | s2_16_clean…031#5 |
| 0.897 | buzz | B | 5 | s2_16_buzz…014#4 |
| … | | | | |

→ **7/10 neighbours are on the B string** (base 17%, **lift 4.2×**). Pitch/MFCC
structure pins the string tightly here.

### Example C — a **buzz low-E** note → neighbours concentrate on low strings

Query `s6_16_buzz_plkmedium_aditya_056#5` (class=**buzz**, string=**low-E**, fret 6):
top neighbours are dominated by low-E / G / A faults; **4/10 on low-E** (base
17%, **lift 2.4×**). Buzz vs muted are the two most acoustically similar fault
classes, so buzz↔muted bleed is expected and visible — still far from random.

---

## 4. The money query — "you keep muting the A" (`search_by_filter`)

`search_by_filter(intended_class='muted', string_num=5, k=10, n_seeds=5)` picks
5 muted-A seed events and aggregates the composition of their **50** combined
nearest neighbours:

```
total_neighbors : 50
class_counts    : {muted: 45, buzz: 5}
string_counts   : {A: 22, D: 9, B: 7, low-E: 6, G: 5, high-e: 1}
```

- **90% of neighbours are MUTED** — base rate 33% → **lift 2.7×**
- **44% of neighbours are on the A string** — base rate 17% → **lift 2.6×**

So when a player mutes the A, the index reliably surfaces *their other muted-A
attempts* well above chance on **both** the fault and the string. That single
"neighbour-coherence-vs-baserate" number is the semantic-quality metric.

---

## 5. Semantic-quality metric (aggregate, all 432 events, k=10)

For every indexed event, fraction of its 10 nearest neighbours that share the
query's class / string, averaged over all 432:

| Coherence | Mean neighbour frac | Base rate | **Lift** |
|---|---|---|---|
| **Class** (clean/buzz/muted) | **0.676** | 0.333 | **2.03×** |
| **String** (which of 6) | **0.429** | 0.167 | **2.58×** |

Per-class class-coherence: clean 0.825, muted 0.720, buzz 0.483 — buzz is the
hardest (most easily confused with muted), as expected acoustically.

Sanity check on dimensionality: re-running on the **raw 28-dim** (no PCA) index
gives class 2.04× / string 2.62× — statistically the same as PCA-16
(2.03× / 2.58×). PCA-16 is the canonical index (smaller vectors, free recall).

**Interpretation:** neighbours are ~2× more likely to share the fault and ~2.6×
more likely to share the string than random retrieval. The embedding encodes
*how the mistake sounds*, not noise.

---

## 6. How this becomes "Tactus remembers your recurring mistakes" (agent memory)

Redis here is **long-term agent memory for the coach**, not a cache:

1. **Write on every attempt.** Each note a player records is `VADD`ed to
   `tactus:events` with its class/string/fret metadata — the player's mistake
   history accrues in Redis across sessions (filter by `run_id`/`session_id`).
2. **Recall on demand.** When the model flags a muted A in the live session, it
   calls `search_by_filter(intended_class='muted', string_num=5)` → "you've
   muted the A in 9 of your last 10 similar attempts." That is grounded,
   *personalized* coaching feedback — exactly what a Deaf player can't self-detect
   by ear.
3. **Retrieval-augmented feedback.** `search_by_event(new_event_id)` pulls the
   closest past attempts so the coach can say "this sounds like the buzz you had
   on the low E last week — same fix: more finger pressure," with the actual
   neighbour events as evidence.
4. **Scales natively in Redis.** `VSIM` is approximate-NN inside Redis, so this
   stays sub-millisecond as one player's history grows to thousands of notes and
   across many players — no separate vector service to operate.

---

## 7. Reproduce

```bash
# rebuild index + smoke test (prints VCARD and an example)
.venv/bin/python software/ai/analysis/exp/redis_retrieval.py

# in Python
import sys; sys.path.insert(0, 'software/ai/analysis/exp')
from redis_retrieval import index_events, search_by_event, search_by_filter
index_events()
search_by_filter(intended_class='muted', string_num=5, k=10, n_seeds=5)
```
