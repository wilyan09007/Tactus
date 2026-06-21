#!/usr/bin/env python3
"""
Tactus offline-analysis pipeline — shared contract (schema.py).

Single source of truth for the segment -> features -> PCA/LDA -> audit pipeline.
Every analysis module imports THIS; nothing else defines paths, the label set,
the fret scope, or the feature-name lists. Freeze this and the stages can be
built (and rebuilt) independently against it.

Pipeline (docs/20-aiml-training-design, /23-data-and-cluster-semantics, /24-data-collection-protocol):

    data/raw/<session>/<player>/{manifest.jsonl, audio/*.wav, video/*.webm}
    data/calib/<guitar>/twin.json                      (from software/ai/vision/twin.py)
      -> segment.py         : onset events, labels attached, F0 cross-check
      -> features_audio.py  : ~26 named audio features / event
      -> features_vision.py : ~13 named vision features / event (reuses ../vision/fretboard.py)
      -> collapse.py        : standardize -> PCA(95%) -> LDA; LOPO d'/confusion/Fisher/silhouette
      -> audit.py           : one-screen HTML (QC, label integrity, running d', coverage)
    -> data/analysis/<session>/<player>/{events.csv, features_*.csv, metrics.json, audit.html}

Conventions match software/ai/capture: stdlib paths from __file__, flat imports,
no package install. Run modules directly; the analysis dir is placed on sys.path
via on_path().
"""
import os
import sys

# ----------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))                # .../software/ai/analysis
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))    # repo root
RAW_DIR    = os.path.join(ROOT, "data", "raw")      # capture:  <session>/<player>/{audio,video,manifest.jsonl}
CALIB_DIR  = os.path.join(ROOT, "data", "calib")    # twin:     <guitar>/{meta.jsonl, twin.json, kf_*.png}
OUT_DIR    = os.path.join(ROOT, "data", "analysis") # pipeline: <session>/<player>/{events.csv, ...}
VISION_DIR = os.path.join(ROOT, "software", "ai", "vision")     # fretboard.py / twin.py live here


def on_path(*dirs):
    """Put dirs on sys.path so flat imports work whether a module is run directly
    or imported. Default = analysis dir + vision dir. Matches the repo's no-package
    style (see software/ai/capture/serve.py)."""
    for d in (dirs or (HERE, VISION_DIR)):
        if d not in sys.path:
            sys.path.insert(0, d)


# ----------------------------------------------------------------- labels / scope
# The 3 acoustically-distinct classes the study separates: good vs buzz vs dead.
# We do NOT split the buzz cause (light vs placement is audio-ambiguous); muted is
# an unambiguous dead note. All three separate on AUDIO ALONE — no vision `d` needed.
CORE_CLASSES = ["clean", "buzz", "muted"]
# Other prompted classes that may appear in the manifest but sit outside the core study.
EXTRA_CLASSES = ["choked", "natural", "one-buzz"]
ALL_CLASSES = CORE_CLASSES + EXTRA_CLASSES

# SCOPE LOCK: 6 fret-zone motors, 1:1 fret->zone. No fret above 6 (truth.md, docs/24 §0b).
FRETS = list(range(1, 7))          # 1..6
N_STRINGS = 6                      # s_num: low-E=6 .. high-e=1; board Y 0..1 maps low-E->high-e

# Standard-tuning open-string MIDI (low-E..high-e) for the F0 -> (string,fret) label check.
OPEN_STRING_MIDI = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}   # E2 A2 D3 G3 B3 E4
F0_FRET_MAX = 12                   # F0 cross-check may see open/above-scope frets


# The captured G is the 3-finger open-B voicing [3,2,0,0,0,3]; the PLAYER played
# the 4-finger G (B and high-e both fretted at 3, ring + pinky). Correct every G or
# the position ground truth is wrong on every G strum (docs/27 data nuance #2).
CHORD_SHAPE_CORRECTIONS = {
    # name: (shape6, fingers6)  low-E..high-e; -1 = not played, 0 = open, n = fret.
    "G": ([3, 2, 0, 0, 3, 3], [2, 1, 0, 0, 3, 4]),   # middle, index, -, -, ring, pinky
}


def correct_chord(strum):
    """(shape6, fingers6) for one chord_sequence strum, applying known shape
    corrections (the played G is the 4-finger variation). Falls back to the stored
    6-vectors. low-E..high-e; -1 = not played, 0 = open, n = fret."""
    fix = CHORD_SHAPE_CORRECTIONS.get(strum.get("chord"))
    if fix:
        return list(fix[0]), list(fix[1])
    return strum.get("shape"), strum.get("fingers")


# ----------------------------------------------------------------- manifest fields
# Centralize field names so manifest parsing lives in one place.
# Authoritative source: main software/ai/capture/capture.html buildRow().
M_LABEL      = "intended_class"
M_PLACEMENT  = "intended_placement"
M_FRETS      = "frets"             # list[int], e.g. [1,2,3,4,5,6]
M_FRET_RANGE = "fret_range"        # human string, e.g. "1->6"
M_STRING_NUM = "s_num"            # 6..1, 0 = chord/free
M_STRING     = "string"
M_FINGER     = "finger"
M_PLUCK      = "pluck_strength"
M_EXPECTED_N = "expected_note_count"
M_MATCHED    = "matched_intent"    # "y"/"n"
M_BPM        = "bpm"
M_BEATS      = "beat_times_ms"     # list[float], ms relative to recording start
M_BLOCK      = "block"
M_PASS       = "pass"
M_HELDOUT    = "held_out"
M_PLAYER     = "player_id"
M_SESSION    = "session_id"
M_RUN        = "run_id"
M_AUDIO      = "audio"             # nested {sample_rate, channels, peak_dbfs, clipped, silent, ...}
M_VIDEO      = "video"             # nested {present, width, height, frame_rate, ...}
M_FILES      = "files"            # nested {audio, video, ...} repo-root-relative paths
M_CHORD_SEQ  = "chord_sequence"   # list[strum] for chord-stream blocks
M_CUE_MS     = "cue_ms"           # per-strum onset cue (ms) inside a chord-stream run


# ----------------------------------------------------------------- event schema
# One row per segmented note. Columns of events.csv — the contract segment.py
# emits and the feature stages read. Keep order stable.
EVENT_COLUMNS = [
    "event_id",          # "<run_id>#<k>"
    "run_id", "session_id", "player_id",
    "block", "pass", "held_out",
    "intended_class",    # label from manifest
    "intended_placement",
    "string_num",        # 6..1
    "target_fret",       # fret this event is meant to be (frets[] by onset order), or -1
    "finger",
    "pluck_strength",
    "onset_s", "dur_s",                  # segmentation
    "f0_hz", "f0_midi",                  # measured pitch
    "f0_string_est", "f0_fret_est",      # nearest (string,fret) implied by F0
    "label_fret_match",                  # bool: F0-implied fret == target_fret (label integrity)
    "audio_peak_dbfs", "audio_clipped", "audio_silent",
    "wav_path", "video_path", "video_frame_rate",
    "guitar_id",         # for twin lookup (set by orchestrator --guitar), may be ""
    # chord-stream labels (None for single-note events); 6-vectors low-E..high-e,
    # JSON-encoded. -1 = not played, 0 = open, n = fret. G is shape-corrected.
    "chord_name", "chord_shape", "chord_fingers",
]
# Key linking a feature row back to its event.
EVENT_ID = "event_id"
LABEL = "intended_class"     # the class column used everywhere downstream
PLAYER = "player_id"         # the split key for leave-one-player-out


# ----------------------------------------------------------------- feature names
# AUDIO (~26). Names are the column contract between features_audio.py and collapse.py.
# (docs/20 §3 timbre features; docs/23 §5; pluck-proxy = attack_slope per 20-eng-review D1.)
AUDIO_FEATURES = [
    "spec_centroid", "spec_bandwidth", "spec_flatness", "spec_rolloff", "spec_flux",
    "zcr", "rms", "log_attack_time", "attack_slope", "decay_rate",
    "hnr", "inharmonicity", "buzz_band_ratio", "pitch_cents_dev", "chroma_peak",
] + ["mfcc_%d" % i for i in range(1, 14)]    # mfcc_1 .. mfcc_13

# VISION (~13). The crux is d_active: signed fingertip->fret-wire distance from
# fretboard.fingertip_d (>0 = behind the wire toward the nut => placement-buzz
# direction). Everything else is supporting pose geometry.
VISION_FEATURES = [
    "d_active",                                  # THE buzz-cause disambiguator
    "d_index", "d_middle", "d_ring", "d_pinky",
    "string_est", "fret_est",                    # from fretboard.to_board
    "finger_curl_active", "finger_curl_mean",
    "wrist_angle", "neck_angle",
    "hand_conf", "reproj_px",                    # reproj_px = twin keyframe registration quality
]

FUSED_FEATURES = AUDIO_FEATURES + VISION_FEATURES


# ----------------------------------------------------------------- helpers
def iter_session_player(base):
    """Yield (session_id, player_id, dir) for every <session>/<player> under `base`."""
    if not os.path.isdir(base):
        return
    for session in sorted(os.listdir(base)):
        sdir = os.path.join(base, session)
        if not os.path.isdir(sdir):
            continue
        for player in sorted(os.listdir(sdir)):
            pdir = os.path.join(sdir, player)
            if os.path.isdir(pdir):
                yield session, player, pdir


def out_dir(session, player):
    """data/analysis/<session>/<player>/, created if missing."""
    d = os.path.join(OUT_DIR, session, player)
    os.makedirs(d, exist_ok=True)
    return d


def abspath(repo_rel):
    """Manifest file paths are repo-root-relative; resolve against ROOT."""
    if not repo_rel:
        return None
    return repo_rel if os.path.isabs(repo_rel) else os.path.join(ROOT, repo_rel)


def resolve_twin(guitar_id=None):
    """Path to data/calib/<guitar>/twin.json. If guitar_id is None and exactly one
    guitar is calibrated, use it; otherwise return None (caller must pick).

    NOTE: the capture manifest carries no guitar_id, so the orchestrator supplies
    --guitar to link a session to its calibration. This is the one capture<->calib
    seam that is not automatic."""
    if guitar_id:
        p = os.path.join(CALIB_DIR, guitar_id, "twin.json")
        return p if os.path.exists(p) else None
    if not os.path.isdir(CALIB_DIR):
        return None
    cands = [g for g in sorted(os.listdir(CALIB_DIR))
             if os.path.exists(os.path.join(CALIB_DIR, g, "twin.json"))]
    return os.path.join(CALIB_DIR, cands[0], "twin.json") if len(cands) == 1 else None


def f0_to_string_fret(f0_midi, string_num=None):
    """Nearest (string_num, fret) for a measured MIDI pitch. If string_num is known
    (from the prompt) use it directly; else pick the string giving a fret in
    [0..F0_FRET_MAX]. Returns (string_num, fret); fret = -1 if nothing fits.
    Used by segment.py for label integrity (does F0 agree with the prompted fret?)."""
    if f0_midi is None or f0_midi != f0_midi:        # None or NaN
        return (string_num or 0, -1)
    if string_num and string_num in OPEN_STRING_MIDI:
        return (string_num, int(round(f0_midi - OPEN_STRING_MIDI[string_num])))
    best_s, best_f, best_err = 0, -1, 1e9
    for s, open_midi in OPEN_STRING_MIDI.items():
        fret = f0_midi - open_midi
        err = abs(fret - round(fret))
        if 0 <= round(fret) <= F0_FRET_MAX and err < best_err:
            best_s, best_f, best_err = s, int(round(fret)), err
    return (best_s, best_f)


def hz_to_midi(hz):
    """MIDI note number from frequency (Hz). NaN-safe."""
    import math
    if not hz or hz != hz or hz <= 0:
        return float("nan")
    return 69.0 + 12.0 * math.log2(hz / 440.0)
