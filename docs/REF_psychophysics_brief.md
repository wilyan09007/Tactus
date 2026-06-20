# REF — Vibrotactile Music Perception Research Brief (raw, cited)

*Background research (raw, cited); the parent plan doc was retired — current project spec is in [`../truth.md`](../truth.md). Numbers inline with citations; evidence-quality flagged. Bottom line: the skin is a coarse receiver — ~5 octaves usable frequency, ~15–20 distinguishable vibration frequencies, ~3–5 cm torso spatial resolution. Win by place-coding pitch, leaning on rhythm/amplitude, and using funneling + apparent-motion illusions to fake spatial density. Frame deaf perception as a learnable tactile code + cross-modal plasticity — never "hearing."*

## 1. Tactile Psychophysics
**1.1 Frequency range.** Glabrous skin ~0.4 Hz to >500 Hz (Bolanowski et al. 1988, https://www.semanticscholar.org/paper/Four-channels-mediate-the-mechanical-aspects-of-Bolanowski-Gescheider/3022fea8c79bfa99b0ce870ae1762c2f3a7c3284). Pacinian channel ~40–800 Hz (Sci Reports 2017, https://www.nature.com/articles/s41598-017-02922-7). Realistic tactile *music* pitch range only C1 (~33 Hz)–G5 (~784 Hz); many can't feel C6 ~1047 Hz (Fletcher et al. 2018, https://pmc.ncbi.nlm.nih.gov/articles/PMC4871541/). ~5 octaves vs ear's ~10.

**1.2 Four mechanoreceptors** (Bolanowski 1988; Purves Neuroscience https://www.ncbi.nlm.nih.gov/books/NBK10895/): Pacinian (FA-II) fast, 40–800 Hz, best ~250–300 Hz; Meissner (FA-I) fast, 10–100 Hz, flutter; Merkel (SA-I) slow, DC–6 Hz, pressure; Ruffini (SA-II) slow, stretch (weakest evidence). Pacinian peak ~250–300 Hz robust — anchor the mapping there.

**1.3 Frequency discrimination.** Vibrotactile Weber fractions ~12–21% (k≈0.12–0.16 @20–40 Hz, ~21% @100 Hz, ~17% @≥150 Hz, https://www.researchgate.net/publication/220461002). Audition JND ~0.2–0.5% (CCRMA https://ccrma.stanford.edu/CCRMA/Courses/152/perceptual.html) — skin 40–100× worse. Net ~15–20 distinguishable vibration frequencies. (Correction: the original "20–30%" JND guess was high.)

**1.4 Spatial acuity (Weinstein 1968, https://en.wikipedia.org/wiki/Two-point_discrimination):** fingertip ~2–8 mm; forearm ~35–40 mm; back/torso ~30–45 mm; thigh ~40–45 mm. Space torso tactors ≥3–5 cm. (Flag: two-point discrimination methodologically criticized; grating-orientation now preferred, https://internal-journal.frontiersin.org/articles/10.3389/fnhum.2013.00579/full.)

**1.5 Detection thresholds.** U-shaped, min ~250–300 Hz, <1 µm displacement (Verrillo, https://www.researchgate.net/publication/272013262). In acceleration min ~80–160 Hz (Morioka & Griffin 2008, https://pubmed.ncbi.nlm.nih.gov/18570014/). Volar forearm least sensitive.

**1.6 Masking.** Forward masking ~25 ms+, grows with masker level (Gescheider 1995, https://pubmed.ncbi.nlm.nih.gov/8550943/); spatial separation + contralateral reduce it (Craig, https://link.springer.com/article/10.3758/BF03209553). Usable dynamic range ~7–48 dB → only ~1–6 loudness steps (Fletcher 2018). Separate voices in space + time; avoid <30–50 ms overlaps.

**1.7 Adaptation.** Threshold rises exponentially, τ≈1.5–2 min; 30 dB-SL adapter → ~10 dB elevation; minutes to recover (Hollins 1990, https://pubmed.ncbi.nlm.nih.gov/2378193/). A 10–20 s adapter can sharpen discrimination. Rotate tactors, modulate, insert silences.

## 2. Tactile Illusions
**2.1 Phantom/funneling.** Two simultaneous actuators fuse into one percept between them; amplitude ratio sets position (Békésy 1957; Alles 1970, https://doi.org/10.1109/TMMS.1970.299958). Energy-summation model (Israr & Poupyrev 2011, https://doi.org/10.1145/1978942.1979235): A₁=√(1−β)·A_v, A₂=√β·A_v. Hard position counts: Park CHI 2018 https://doi.org/10.1145/3173574.3173832; Kim/Schneider CHI 2020 https://doi.org/10.1145/3313831.3376335.
**2.2 Saltation (cutaneous rabbit).** Geldard & Sherrick 1972 (Science 178:178, https://doi.org/10.1126/science.178.4057.178). Window ISI ~20–300 ms; strongest on low-acuity sites.
**2.3 Apparent motion.** Sherrick & Rogers 1966 (https://doi.org/10.3758/BF03215780). SOA ≈ 0.32·d + 47.3 ms; low-freq easier to animate.
**2.4 Tactile Brush** (Israr & Poupyrev, Disney, CHI 2011): 4×3 grid, EAI C-2 tactors, 63 mm spacing → continuous 2D strokes from 12 actuators via phantom + apparent motion. (Flag: constants are actuator/site-specific.)

## 3. Prior Art
**Neosensory VEST/Buzz:** VEST 24/32/40 actuators (press varies; Rice 2015 https://news2.rice.edu/2015/04/08/vest-helps-deaf-feel-understand-speech-2/); Buzz = 4 motors, ~300–7500 Hz → 256 spatial locations (Perrotta/Kohler/Eagleman 2023). Novich & Eagleman 2015 (Exp Brain Res, https://link.springer.com/article/10.1007/s00221-015-4346-1) — ≥6 cm to resolve two back points; spatiotemporal > static; NO bits/s figure. Perrotta 2021 — sound category ID avg ~70% (up to 95%). VEST speech claims = press, not trials.
**Model Human Cochlea / Emoti-Chair** (Karam, Russo, Fels 2009, IEEE ToH 2(3):160–169, https://ieeexplore.ieee.org/document/5184836/): 16 voice coils in 2×8, 8 bands ~27.5 Hz–1 kHz, low→bottom/high→top. CORE prior art. Human outcomes small-N exploratory.
**Music: Not Impossible** (Ebeling): ~24 points (vest+wrist+ankle), live "Vibrotactile DJ" routing. Press/testimony only. (Correction: collaborator is Mandy Harvey; 2018 artist was Greta Van Fleet.)
**SubPac (~5–130 Hz) / Woojer (6 transducers, 1–200 Hz):** tactile bass only, not substitution.
**SoundShirt (CuteCircuit):** 16→30→28 actuators by version; instruments→body zones. Press only. **CymaSpace:** Deaf-owned; mainly cymatic lighting.
**Academic:** Tan et al. 1999 — ~6.5 bits/stimulus, ~12 bits/s (PMID 10497422). Russo/Ammirante/Fels 2012 — timbre via vibration above chance (https://pubmed.ncbi.nlm.nih.gov/22708743/). Fletcher (Southampton) electro-haptic — F0 discrimination to ~1.4%, melody recognition gains (https://pmc.ncbi.nlm.nih.gov/articles/PMC8439542/) but often tested on NH/CI-sim. Paisa/Nilsson/Serafin 2023 scoping review (63 papers, mean N~11, https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2023.1085539/full).

## 4. Honest framing
- Rhythm/tempo: STRONG (Tranchant 2017 https://pmc.ncbi.nlm.nih.gov/articles/PMC5601036/; González-Garrido 2017).
- Pitch direction & ≥3-semitone intervals: ≥70% post-training; 1-semitone ≈ chance (Hopkins 2023 https://journals.sagepub.com/doi/full/10.1177/10298649211015278).
- Timbre: thinner but above chance (Russo 2012).
- Cross-modal plasticity: deaf auditory cortex responds to vibration (Levänen 1998 — verify cite; Bola 2017 PNAS https://www.pnas.org/doi/10.1073/pnas.1609000114) — task-specific, NOT hearing.
- Training: Tadoma (Reed 1985 https://pubmed.ncbi.nlm.nih.gov/3973218/); 48%→62% in 5×1hr.
- **Overclaims to avoid:** "deaf hear via vibration," "brain processes as sound," fine/absolute pitch, full intelligibility w/o long training.

**Correction flags:** skin freq JND ~12–21% not 20–30%; don't attribute bits/s to Novich & Eagleman 2015 (12 bits/s is Tan 1999); VEST 24/32/40 + SoundShirt 16/30/28 version-dependent (stable: Buzz=4/256, Emoti-Chair=16/2×8); SubPac real band ~5–130 Hz; verify Levänen 1998 (Current Biology).
