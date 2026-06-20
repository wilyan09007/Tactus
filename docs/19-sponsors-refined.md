# 19 — Sponsor & prize alignment (refined to the as-built rig)

> Supersedes the earlier sponsors doc (retired); canonical source is [`../truth.md`](../truth.md). The rig is now **laptop brain → 2–3 Vantec USB-audio → 6 gutted PAM8403 amps → 12 SK473 drivers**, product = **LEARN + PLAY guitar coaching** (Experience/Express cut, **Pi + QNX cut**). Apply only where we can genuinely win, tiered by probability.

## The system as one coherent sponsor story
*A webcam + mic feed **MediaPipe + OpenCV** in the browser (vision) and **pYIN/YIN live (CREPE offline)** in Python (audio); a **fusion** layer turns their **disagreement into the detected mistake** and inverts the buzz surface for pressure; **Claude's vision** produces the exact physical fix and the finger to highlight; the correction is rendered to **12 independent haptic channels**; the contrastive embedding is **trained on AWS Trainium**; the whole pipeline is **traced + evaluated in Arize**, **voice-controlled by Deepgram**, and **remembered in Redis** so it works offline.* No contortion — every sponsor sits on the critical path.

## Tier 1 — go hard (highest win probability; build the demo around these)
| Prize | Why we win | Lean on |
|---|---|---|
| **Anthropic** | **Claude vision is the coaching brain** — (frame + target fingering + detected fault) → the one physical fix + which finger to ring. Multimodal, deep, built with Claude Code. Not a chatbot. | `docs/17 §2`, `22` |
| **Most Technical** | audio+vision **fusion (disagreement = the error)** + the **buzz inverse** (recover pressure as a 2-class ordinal) + the **separability study** + a real **eval/ablation**. AI where the problem is unsolved, deterministic where it's solved. | `docs/17`, `20`, `23` |
| **Best UI/UX** | the interface that **makes the AI visible** — pipeline view, live finger-highlight, the 12-node body map, the live 3D cluster view, confidence + Arize trace. | `docs/22`, `web/` |
| **Accessibility / "Ddoski's World"** | sensory-substitution that lets a **Deaf/HoH player feel + learn guitar** — feel the target, see the fix. Name a Deaf advisor. | whole product |

## Tier 2 — strong, low marginal effort (claim them, don't reorganize for them)
- **Computer Vision track** — MediaPipe Hands (21 landmarks, in the browser) + **ArUco/OpenCV fretboard homography** + occluded placement-vs-fret-wire. Real CV, already in the build.
- **Annapurna / AWS Trainium** — train the **contrastive multimodal embedding** (the legitimate accelerator job; Redis then retrieves on those vectors). Cut-first stretch — the engineered-feature core is the deliverable, so don't block the demo on it.
- **Redis** — the coach's **memory**: per-user mistake history, adaptive difficulty, **vector-search "your similar past mistakes,"** offline song/calibration library. A genuine home in the loop, not a cache.
- **Deepgram** — hands-free voice (*rewind / slower / again*) — **essential** because both hands are on the guitar.
- **Arize / Phoenix** — trace + eval the F0/transcription, the technique classifier, and the VLM correction accuracy; **catch octave errors live** and fix the mapping during the event ("used *and* improved the model").

## Tier 3 — only if it's free
- **Fetch (multi-agent)** — *optional* light analyze→coach split via uAgents. Don't contort the architecture for it.
- **Sentry** — only if the real-time loop is already worth instrumenting; cheap add, low prize weight.

## Dropped — do NOT apply or build for
- **QNX** — its whole value was a Pi real-time loop; **Pi is cut**. Drop cleanly.
- Robotics / chip / media-gen (Pika/Midjourney) / computer-use / browser-use — bolt-ons here.

## Action items
- [ ] Devpost: **Ddoski's World** primary + tag **Anthropic, Deepgram, Arize, Redis, Annapurna/AWS Trainium** (+ the CV track). Drop QNX/Fetch from the headline.
- [ ] Grab credits **early**: Anthropic, Deepgram ($200), Arize/Phoenix, Redis, AWS.
- [ ] Attend workshops: **Anthropic, Deepgram (40-min), Arize.** (Skip the QNX flashing session — Pi is cut.)
- [ ] **Name a Deaf advisor/tester** — accessibility multiplier and the right thing ("nothing about us without us").
- [ ] One-line pitch hook per sponsor ready at the table (`docs/17 §5` has drafts).

> The judge-facing throughline: *"The AI is the part that's actually hard — understanding arbitrary guitar playing from a camera and a mic. Claude watches your hands and tells you the one thing to change; we render it to your body on 12 channels; and we traced and evaluated every model to prove it works."*
</content>
