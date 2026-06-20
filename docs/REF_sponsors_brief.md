# REF — Instrument-Learning + Sponsor + Deaf-Framing Brief (raw, cited)

*Background research (raw, cited). The parent plan docs were retired. **Current sponsor strategy is [`19-sponsors-refined.md`](19-sponsors-refined.md); project truth is [`../truth.md`](../truth.md).** Key citations preserved.*

## Pitch engines
SwiftF0 (2025, ~96k params, ~42× faster than CREPE, 46.9–2093.75 Hz, https://arxiv.org/abs/2508.18440); pYIN (https://ieeexplore.ieee.org/document/6853678/); torchCREPE (https://github.com/maxrmorrison/torchcrepe); CREPE (https://github.com/marl/crepe); SPICE (https://research.google/blog/spice-self-supervised-pitch-estimation/); basic-pitch polyphonic (https://github.com/spotify/basic-pitch). Recommend SwiftF0 primary, pYIN/torchCREPE-tiny Pi fallback; avoid full CREPE (heavy) + basic-pitch (polyphonic) for live tuner.

## Instrument-haptics psychophysics
Drive ~250 Hz; ~5 intensity steps (14% Weber, IEEE ToH 2016 https://ieeexplore.ieee.org/document/7470536/); torso two-point ~52 mm (28 mm w/ 200 ms separation); intervals 3–12 semitones ≥70%, single-semitone ~61% (Hopkins 2023 https://journals.sagepub.com/doi/full/10.1177/10298649211015278); training reshapes deaf brain connectivity (PMC5292439). Encode error direction + ≤5 steps, NOT absolute pitch by frequency.

## Instrument prior art
Evelyn Glennie "hearing is a specialized form of touch," pitch→body location (https://www.evelyn.co.uk/hearing-essay/) — THE anchor; MuSS-Bits (Petry/Nanayakkara OzCHI 2016, https://dl.acm.org/doi/10.1145/3010915.3010939); Emoti-Chair / Haptic Chair (Nanayakkara, https://ahlab.org/project/hapticchair/). (Don't cite "Good Vibrations" — unverified.)

## Sponsor integrations (deep, non-bolt-on)
| Sponsor | Integration | Story |
|---|---|---|
| Anthropic ($5k) | Claude = generative haptic composer + adaptive tutor; favorite-song→haptic; built w/ Claude Code | "Claude turns any song into a body-felt experience + coaches a Deaf learner" |
| QNX ($1k+Pi) | F0→haptic real-time loop on QNX 8.0 / Pi 4B; jitter-free actuator timing = safety-of-experience (https://devblog.qnx.com/qnx-at-cuhacking-6/) | "Tactile feedback only teaches if perfectly timed" |
| Deepgram ($200) | hands-free voice control (violin under chin) + captions for mixed Deaf/hearing | "Can't touch a screen with a violin under your chin" |
| Arize ($1k) | Phoenix trace/eval F0 across instruments/noise; fix octave errors live (https://arize.com/phoenix/) | "Traced + evaluated our pitch engine, fixed octave errors live" |
| Fetch.ai ($1.5k) | uAgents: Music-Analysis + Haptic-Choreography + Tutor agents (https://innovationlab.fetch.ai/) | "Three agents turn a song into a haptic lesson" |
| Sentry | instrument real-time loop: dropped frames/actuator faults/latency violations | "Catches a dropped haptic frame mid-lesson" |
| Redis | cache choreographies by song + per-user calibration/progress | "Tactus remembers your body" |

Recommended set: QNX, Anthropic, Deepgram, Arize, Fetch (+Sentry). Coherent system: Claude composes → uAgents orchestrate → QNX real-time → Deepgram voice → Arize eval. Main track **Ddoski's World** (social impact) + **Ddoski's Lab** depth.

## Deaf-culture framing (do-or-die)
Medical vs cultural model of deafness; audism (Tom Humphries 1975, https://www.medicaldevice-network.com/features/audism-cochlear-implants/). Haptic ≠ auditory → sidesteps the cochlear-implant landmine.
- **Evelyn Glennie** (deaf percussionist, pitch→body) — top citation.
- **Christine Sun Kim** — Deaf sound artist; work is about owning/reclaiming sound — perfect anchor for the *creative-expression* angle.
- Deaf Rave (20+ yrs, Woojer backpacks, https://ukf.com/read/we-need-to-talk-about-deaf-rave/); "Deaf Gain" not "hearing loss" (Bauman & Murray, https://gallaudet.edu/deaf-studies/deaf-gain-authors-reflect-on-why-deaf-gain-is-more-important-than-ever/); "Nothing about us without us" (https://www.boia.org/blog/nothing-about-us-without-us-starting-digital-accessibility-conversations); De'VIA (Deaf View/Image Art) + Visual Vernacular = real Deaf art forms.
DO say: "new way to feel/make music," "built with the Deaf community," capital-D Deaf, additive. DON'T: cure/restore/fix/hear, "the deaf" (monolith), savior framing.

## Competitive white space (triad nobody fully occupies)
1. Wideband + multi-node + ML pitch/timbre mapping. 2. Generative "song→haptic" with no human choreographer (vs Music: Not Impossible's manual). 3. Haptic instrument learning (commercially unoccupied). + (this brief's new 4th) **haptic-native creative expression/composition.**

Judging: Cal Hacks weights Application + Functionality/Quality (https://www.calhacks.io/). WOW = judge feels + names music blindfolded, live generation, named Deaf tester. DISMISSIBLE = "just buzzes to the beat," slideware, cure framing, overclaiming exact pitch.

**Flags:** SwiftF0 single-author 2025 preprint (validate on own audio); confirm "~18% learning-error reduction" in PMC8439542 before slides; Sound2Hap/2026 arXiv IDs not deep-read.
