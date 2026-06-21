#!/usr/bin/env python3
"""
Tactus data-collection CONDUCTOR  (Day-1, zero-build, stdlib only)

What this is: a terminal conductor for prompted-capture data collection.
It does NOT record media -- QuickTime does that (rock-solid, synced A/V in ONE file).
This script tells you EXACTLY what to play next (interleaved order, to kill session
drift), and writes the manifest (the prompt IS the label) so nothing is hand-tagged.

Why not localhost: the browser/align-mode harness is the NEXT build (for the M4 with
the camera). This conductor is what you run RIGHT NOW, outdoors, with zero setup.

THE RECIPE (read once):
  1. Mic: plug in the Saramonic. System Settings > Sound > Input = Saramonic.
     Disable any input "enhancement"/noise-cancel. Gain so a HARD strum doesn't clip.
  2. Metronome: phone app at ~50 BPM (slow -> notes separate for auto-segmentation).
  3. QuickTime > New Movie Recording. Click the v next to the record button:
     Camera = MacBook front cam (you in playing position), Microphone = Saramonic.
     Hit record. (Audio-only fallback if no ArUco marker yet: New Audio Recording.)
  4. CLAP once, loud, at the very start -- that's the A/V sync mark.
  5. Run this script. For each prompt: play the run to the click, then press ENTER.
  6. One QuickTime file per PASS is fine. Save it into the session's video/ folder
     this script prints at the end, then hand the folder to the pipeline.

Usage:
  python3 software/ai/capture/record_conductor.py --player aditya --passes 3
  python3 software/ai/capture/record_conductor.py --player aiden --passes 1 --finger ring
"""
import argparse, json, os, random, sys
from datetime import datetime

# low-E first (down-strum order), high-e last
STRINGS = [(6, "low-E"), (5, "A"), (4, "D"), (3, "G"), (2, "B"), (1, "high-e")]
CORE_CLASSES = [
    ("clean",          "finger JUST BEHIND the wire, GOOD pressure, medium pluck -> rings clear"),
    ("buzz-light",     "finger in the CORRECT spot, pressed TOO LIGHT -> buzz from low pressure"),
    ("buzz-placement", "FIRM pressure but finger TOO FAR BACK from the wire -> buzz from placement"),
]
FRETS = "1->6"  # SCOPE LOCK: 6 fret-zone motors, 1:1 fret->zone. No fret above 6.

STAG = {6: "lowE", 5: "A", 4: "D", 3: "G", 2: "B", 1: "highe"}
CTAG = {"clean": "clean", "buzz-light": "buzzlight", "buzz-placement": "buzzplace"}


def ask(msg, default=""):
    try:
        return (input(msg).strip() or default).lower()
    except (EOFError, KeyboardInterrupt):
        print("\n[stopped]")
        sys.exit(0)


def show(row, n, total):
    print("\n" + "=" * 66)
    print(f"  RUN {n}/{total}    [ {row['intended_class'].upper()} ]")
    print("=" * 66)
    print(f"  STRING : {row['string']}")
    print(f"  FRETS  : {row['fret_range']}   (one note per click, low -> high)")
    print(f"  FINGER : {row['finger']}")
    print(f"  PLUCK  : {row['pluck_strength']}")
    print(f"  DO     : {row['_how']}")
    print("-" * 66)


def main():
    ap = argparse.ArgumentParser(description="Tactus prompted-capture conductor")
    ap.add_argument("--player", default="aditya")
    ap.add_argument("--passes", type=int, default=3, help="position-grid passes (3-4 typical)")
    ap.add_argument("--finger", default="index", help="finger for this run set (vary on later passes)")
    ap.add_argument("--pluck", default="medium")
    ap.add_argument("--room", default="quiet", choices=["quiet", "noisy"])
    ap.add_argument("--no-interleave", action="store_true", help="keep class order fixed (NOT recommended)")
    args = ap.parse_args()

    session = datetime.now().strftime("%Y-%m-%d-%H%M")
    outdir = os.path.join("data", "raw", session, args.player)
    os.makedirs(os.path.join(outdir, "video"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "audio"), exist_ok=True)
    manifest = os.path.join(outdir, "manifest.jsonl")

    print("""
TACTUS conductor -- setup BEFORE you start:
  [ ] Saramonic = the Mac input; input 'enhancement' OFF; gain not clipping on a hard strum
  [ ] Metronome ~50 BPM running
  [ ] QuickTime > New Movie Recording: camera = front cam, mic = Saramonic  ->  RECORD
  [ ] CLAP once, loud, NOW (A/V sync mark)
""")
    ask("Press ENTER once recording is rolling and you've clapped... ")

    # Build the interleaved run plan: per pass, per string, the 3 classes shuffled.
    runs = []
    for p in range(1, args.passes + 1):
        for s_num, s_name in STRINGS:
            classes = CORE_CLASSES[:]
            if not args.no_interleave:
                random.Random(f"{session}-{p}-{s_num}").shuffle(classes)
            for cname, chow in classes:
                runs.append({
                    "string": f"{s_num} ({s_name})", "fret_range": FRETS, "finger": args.finger,
                    "intended_class": cname,
                    "intended_placement": "too-far-back" if cname == "buzz-placement" else "on-wire",
                    "pluck_strength": args.pluck, "_how": chow, "_pass": p, "_s": s_num,
                })

    total = len(runs)
    est_min = round(total * 6 * 5.5 / 60)
    print(f"PLAN: {total} runs x 6 notes ~= {total * 6} labeled events "
          f"({args.passes} passes x 18 runs).  ~{est_min} min.\n")

    done, i = 0, 0
    while i < len(runs):
        show(runs[i], i + 1, total)
        cmd = ask("Played it?  [ENTER]=done   r=redo prompt   s=skip   q=quit: ")
        if cmd == "q":
            break
        if cmd == "s":
            i += 1
            continue
        if cmd == "r":
            continue
        matched = ask("  matched intent?  [Y/n]: ", "y").startswith("y")
        done += 1
        r = runs[i]
        rid = (f"{STAG[r['_s']]}_f1-6_{CTAG[r['intended_class']]}"
               f"_pluck{r['pluck_strength']}_{args.player}_{done:03d}")
        row = {
            "run_id": rid, "session_id": session, "player_id": args.player,
            "string": r["string"], "fret_range": r["fret_range"], "finger": r["finger"],
            "intended_class": r["intended_class"], "intended_placement": r["intended_placement"],
            "pluck_strength": r["pluck_strength"], "chord_name": None,
            "is_arpeggio": False, "is_strum": False,
            "matched_intent": "y" if matched else "n", "room": args.room,
            "source_wav": "", "source_video": "", "notes": "",
            "logged_at": datetime.now().isoformat(timespec="seconds"), "pass": r["_pass"],
        }
        with open(manifest, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  ok logged {rid}    ({done} runs / ~{done * 6} events)")
        i += 1

    print(f"\nDONE. {done} runs -> {manifest}")
    print("Now: STOP QuickTime and save the movie(s) into:")
    print(f"  {os.path.join(outdir, 'video')}/   (or audio/ if audio-only)")
    print("Then hand the whole session folder to the pipeline.")
    print("\nNext blocks (rerun this, or just announce them on the recording):")
    print("  - pose-variation:  --finger ring   and   --finger pinky   (Stage-1 generalization)")
    print("  - pluck-sweep:     a few cells at soft / medium / hard (buzz != pluck)")
    print("  - muted + choked:  a few runs each (cheap external flags)")
    print("  - chords:          ~6-8 shapes ARPEGGIATED, clean + one deliberate buzz")
    print("  - natural holdout: ~5 min 'just play normally' (HELD OUT)")


if __name__ == "__main__":
    main()
