# Capture — prompted data collection

Two ways to record the prompted runs from [`docs/24`](../../../docs/24-data-collection-protocol.md).
Both write to `data/raw/<session>/<player>/` (`audio/` + `video/` + `manifest.jsonl`).
**The prompt IS the label** — no hand-tagging.

## Recommended: localhost app (capture + labeling + QC in one page)
```
python3 software/ai/capture/serve.py        # run from the repo root
```
Open the printed **http://localhost:8765** in **Chrome**. Pick the **Saramonic** mic + **front cam**,
hit **Build run plan**, then per prompt: **Record → play to the visual click → Stop → Yes/No** (auto-saves).
- synced webcam+mic per run (lossless **WAV** + **webm**), browser DSP disabled so the buzz band survives
- live **CLIP alarm** + silent/too-short checks + coverage grid + **maximal metadata** (for the harness loop)
- if the server is down it offers a per-run **download** so nothing is lost
- **audio-only** mode if the ArUco marker isn't on the headstock yet (Stage-2 only; Stage-1 needs video)

## Fallback: terminal conductor + QuickTime
```
python3 software/ai/capture/record_conductor.py --player aditya --passes 3
```
**QuickTime** (New Movie Recording, mic = Saramonic, clap once for sync) records the A/V; the script
prompts each run and writes the manifest.

## Files
| File | What |
|---|---|
| `serve.py` | stdlib localhost server — serves the page, saves blobs + manifest to `data/raw/` |
| `capture.html` | the capture UI (vanilla JS, no deps) |
| `record_conductor.py` | terminal conductor (stdlib; QuickTime does the A/V) |

Matrix, class definitions, targets, and the full metadata schema: **`docs/24`**.
