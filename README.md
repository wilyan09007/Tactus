# 🎸 Tactus

**A Deaf-accessible guitar-coaching wearable.** From a webcam + a mic, Tactus recovers *what you played, where, with which finger, and how cleanly you fretted it* — by fusing vision and audio through a learned physical model — then gives you **one precise correction you feel on your body and see on your hand.**

> Built for the **UC Berkeley AI Hackathon 2026 (June 20–22)**. Team of 4. Working name **Tactus** (*touch* + the conductor's *pulse*).
>
> 📌 **Canonical source of truth: [`truth.md`](truth.md).** If any doc disagrees with it, `truth.md` wins.

---

## The 30-second version
- **Problem.** Learning guitar is a hear-and-correct loop: play → hear the buzz / wrong note → fix. Deaf and hard-of-hearing players are locked out of that loop (~466M people have disabling hearing loss, WHO).
- **Tactus — two modes on one engine:**
  - **LEARN** — "Rocksmith for Deaf players." Feel the target song on your body, play it on a real guitar, and on any mistake the system rewinds, replays the correct haptics vs. yours, and shows the physical fix — Claude vision highlights the wrong finger.
  - **PLAY** — pick up the guitar and get real-time "flat / sharp / in tune / on the beat" by feel.
- **Where the AI is.** The hard, unsolved part is reading an *occluded* fretting hand and a *cause-blind* buzz from a webcam + mic. The ML lives exactly there; everything downstream is a deterministic, measurable signal→vibration transform (no AI-invented "score"). That split is the rigor.
- **The one correction.** Three axes: **wrong note · wrong duration · pressure** (too-light vs good; "too hard" surfaces as a sharp/choked pitch fault). Each is a measured signal, cross-checked across vision + audio.

## The rig (as-built — full detail in [`truth.md`](truth.md))
```
LAPTOP (M4 Pro) ──USB──► 2× Vantec NBA-200U (USB→8 ch; V3 spare)
                            └► 6× SK473 PAM8403 stereo amp boards (gutted)
                                 └► 12× KHD 3 Ω/5 W drivers (de-housed) on the body
                                      = 12 channels: 6 back (strings) + 6 torso (fret-zones)
```
- **Software split (locked):** the **browser** owns camera + MediaPipe + ArUco + AR + viz; a **Python** process owns mic + F0 + the fusion model + 12-ch haptic output; they sync over one localhost WebSocket (vision features flow browser→Python, timestamped).
- **Power:** wall (10-port hub) for the judged run, or cordless (Anker 737). Only ~2–3 channels fire at once (sequential strum sweep), so real peak is ~2–3 W.

## Docs map
| Doc | What's in it |
|---|---|
| **[`truth.md`](truth.md)** | ⭐ The canonical source of truth — direction + all hardware (parts, dimensions, cost, use) |
| [`docs/00-start-here.md`](docs/00-start-here.md) | On-ramp + build-tonight TL;DR |
| [`docs/01-bill-of-materials.md`](docs/01-bill-of-materials.md) | The as-purchased BOM (SKUs, qty, spares) |
| [`docs/02-system-architecture.md`](docs/02-system-architecture.md) · [`03-power.md`](docs/03-power.md) | System chain + the chord-safety power math |
| [`docs/04-soldering-guide.md`](docs/04-soldering-guide.md) · [`05-wiring-map.md`](docs/05-wiring-map.md) · [`06-safety.md`](docs/06-safety.md) · [`09-assembly-checklist.md`](docs/09-assembly-checklist.md) | Build: solder order, channel-by-channel wiring, safety, assembly gates |
| [`docs/07-haptic-encoding.md`](docs/07-haptic-encoding.md) · [`18-tuning-and-calibration.md`](docs/18-tuning-and-calibration.md) · [`21-chord-and-sustain-rendering.md`](docs/21-chord-and-sustain-rendering.md) | How notes / chords / sustain become vibration + on-body tuning |
| [`docs/08-software-architecture.md`](docs/08-software-architecture.md) · [`13_LEARN_WEB_AND_VISUALIZATION.md`](docs/13_LEARN_WEB_AND_VISUALIZATION.md) | The engine + the browser↔engine WebSocket contract |
| [`docs/17-ai-rigor.md`](docs/17-ai-rigor.md) · [`20-aiml-training-design.md`](docs/20-aiml-training-design.md) · [`20-eng-review.md`](docs/20-eng-review.md) · [`23-data-and-cluster-semantics.md`](docs/23-data-and-cluster-semantics.md) | The AI core: fusion, the buzz inverse, the separability study, data + training |
| [`docs/22-interface-ar-and-correction.md`](docs/22-interface-ar-and-correction.md) | The interface: live AR play + the 2D correction view |
| [`docs/10-design-decisions.md`](docs/10-design-decisions.md) · [`11-ai-and-pitch.md`](docs/11-ai-and-pitch.md) · [`12-perception-references.md`](docs/12-perception-references.md) · [`13-open-questions.md`](docs/13-open-questions.md) · [`15-build-refinements.md`](docs/15-build-refinements.md) | Decisions, perception science, open items, the as-built teardown log |
| [`docs/19-sponsors-refined.md`](docs/19-sponsors-refined.md) | Sponsor / prize alignment |
| `docs/REF_*.md` (ml · psychophysics · sponsors) | Cited research briefs (background) |
| [`cad/`](cad/) | Printable enclosure + actuator coupling pucks (FlashForge 5M) |
| [`config/channel_map.json`](config/channel_map.json) | Machine-readable 12-channel routing (wiring source of truth) |
| [`CHANGELOG.md`](CHANGELOG.md) | Chronological history (incl. the pre-pivot era) |

---
*Pivoted from an earlier multi-pillar "music for the Deaf" concept. The abandoned Pi / ESP32 / 16-channel / 3-pillar docs were removed from this repo (recoverable from git history). The current product is **LEARN + PLAY guitar coaching** — see [`truth.md`](truth.md).*
