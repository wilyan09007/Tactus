# REF — Audio→Haptic ML/DSP Research Brief (raw, cited)

*Background research (raw, cited). The parent plan doc was retired — **for current project truth see [`../truth.md`](../truth.md)**. Historical note: this brief predates the LEARN+PLAY pivot; the "haptic-score" idea was **cut** — the shipped pipeline is a direct, measurable signal→vibration transform with no generative score. Citations preserved for reference.*

## Front-end
- **Separation:** Spleeter (Deezer, MIT, ~100× RT, ~5.9 dB SDR, https://github.com/deezer/spleeter); Open-Unmix; Hybrid Demucs v3 ~7.7 dB; **HT-Demucs v4 9.0–9.2 dB SOTA but non-causal** (https://github.com/facebookresearch/demucs). No causal Demucs-quality separator exists → separate offline.
- **Rhythm:** madmom (offline RNN+DBN, https://github.com/CPJKU/madmom); **BeatNet** (causal CRNN+particle filter, ISMIR 2021, https://github.com/mjhydri/BeatNet); Beat-Transformer (offline, https://arxiv.org/abs/2209.07140); librosa onsets (hop 256 ≈ 23 ms).
- **Pitch:** CREPE (ICASSP 2018, https://github.com/marl/crepe, ~0.95–0.999 RPA, 22M params → use tiny live); pYIN (Mauch & Dixon 2014, `librosa.pyin`); basic-pitch (Spotify, polyphonic, https://github.com/spotify/basic-pitch); SwiftF0 (2025 preprint, ~96k params, ~42× faster than CREPE, range 46.9–2093.75 Hz, https://arxiv.org/abs/2508.18440).
- **Timbre:** chroma_cqt, spectral centroid/flux/flatness, MFCC; Essentia (streaming C++), aubio.
- **Latency:** audio-tactile asynchrony — haptic-leads tolerated ~451 ms vs audio-leads ~179 ms (IEEE ToH 2023 https://ieeexplore.ieee.org/document/10107431/); latency JND ~27–49 ms. Engineer ~50 ms, predict beats so haptics lead. STFT hop 256≈5.8 ms / 512≈11.6 ms / 1024≈23 ms.

## Psychohaptic mapping
- Tactile band ~20–500 Hz, Pacinian peak ~250 Hz (https://www.nature.com/articles/s41467-026-69251-0).
- Warp not transpose: Mel `2595·log10(1+f/700)`, Bark (Traunmüller `z=26.81/(1+1960/f)−0.53`), ERB (Glasberg & Moore 1990 `24.7·(4.37f/1000+1)`, https://ccrma.stanford.edu/~jos/sasp/Equivalent_Rectangular_Bandwidth.html). "Tactile-ERB" not formally established — borrow auditory partition (flag).
- Tonotopy: Model Human Cochlea (Karam/Russo/Fels, IEEE ToH 2009, https://pubmed.ncbi.nlm.nih.gov/27788080/).
- Loudness: Stevens power law β≈0.45–0.95 (https://en.wikipedia.org/wiki/Stevens's_power_law); ~40–55 dB range, ~20% Weber → ~10–15 steps.
- Masking: Gescheider — P-range only, no cross-channel P↔NP masking (https://pubmed.ncbi.nlm.nih.gov/8550943/).
- Why > buzzer: Novich & Eagleman 2015 (space+time > amplitude, https://www.researchgate.net/publication/278794632); Buzz 8 bands→8 motors above-chance day one; Miller 1956 single channel ~1–2 bits.

## Learned model
- Prior art (all single-channel): **HapticGen** (CHI 2025, MusicGen+EnCodec, code+weights MIT, https://github.com/HapticGen/HapticGen); HapticLDM (2026 latent diffusion, no code, https://arxiv.org/html/2605.09971v1); Sound2Hap (CHI 2026 CNN AE, https://arxiv.org/abs/2601.12245); Lee et al. CHI 2023 (event-driven).
- Standard: **MPEG-I Part 31 Haptics Coding** ISO/IEC 23090-31:2025 (HJIF/MIHS, https://www.iso.org/standard/86122.html; RFC 9695 `haptics` media type).
- Encoders: **MERT-v1-330M** (music SSL 1024-d @75 Hz, https://huggingface.co/m-a-p/MERT-v1-330M); **CLAP** (audio+text, https://github.com/LAION-AI/CLAP); **EnCodec** (75 Hz, https://github.com/facebookresearch/encodec) as decoder template.
- Architecture: MERT(frozen) → causal TCN/Conv1D+GRU (~1–3M params, ≤2–3 frame lookahead ≈27–40 ms) → 16-ch envelopes. Distill DSP teacher (Stage A, FMA/Jamendo) → perceptual finetune (Stage B) → DPO (Stage C bet).
- Loss: `L = λ_env·L_env + λ_ons·L_onset + λ_loud·L_loud + λ_sep·L_separation + λ_mask·L_mask + λ_smooth·L_smooth`; start λ_env=1, λ_ons=2, λ_loud=0.5, λ_sep=1, λ_mask=0.5, λ_smooth=0.1. L_separation = off-diagonal channel-corr + InfoNCE (optimizes the eval d′ directly).
- Perceptual grounding: Kim et al. "Sound-to-Touch Crossmodal Pitch Matching" IEEE ToH 2024 (https://engineering.purdue.edu/~hongtan/pubs/PDFfiles/J81_DGKim-etal_ToH2024_SoundToTouchMatching.pdf).

## Claude haptic-score JSON schema (down-compiles to MPEG-I HJIF)
```jsonc
{
  "schema": "haptic-score/1.0",
  "track": { "title":"…","bpm":128,"key":"F#min","duration_s":212.0,
             "device": { "n_actuators":16,"layout":"vest-8x2",
                         "tactile_band_hz":[20,1000],"render_rate_hz":8000 } },
  "zones": [
    {"id":"bass","actuators":[0,1,2,3],"carrier_hz":60,"body":"lower-back/hips"},
    {"id":"drums","actuators":[4,5],"carrier_hz":120,"body":"sternum"},
    {"id":"harmony","actuators":[6,7,8,9],"carrier_hz":250,"body":"mid-back"},
    {"id":"melody","actuators":[10,11,12,13],"carrier_hz":400,"body":"shoulders"},
    {"id":"vocal","actuators":[14,15],"carrier_hz":500,"body":"collarbone"}
  ],
  "sections": [
    {"label":"drop","start_s":96.0,"end_s":128.0,"energy":0.95,"events":[
      {"zone":"bass","pattern":"pulse","sync":"downbeat",
       "intensity_env":{"type":"adsr","a":0.005,"d":0.08,"s":0.0,"r":0.12,"peak":1.0}},
      {"zone":"melody","pattern":"sweep","direction":"hips->shoulders","speed_s":0.4,
       "intensity":0.7,"trigger":"every_2_bars"}
    ]}
  ],
  "motion_patterns": {
    "pulse":{"kind":"impulse"},"sweep":{"kind":"spatial_traverse","interp":"raised-cosine"},
    "rotate":{"kind":"circular","period_s":0.5},"expand":{"kind":"center-out"}
  }
}
```
A deterministic compiler fuses score (structure/motion from Claude) ⊕ per-frame envelopes (texture from the net) → 16×8 kHz drive → HJIF.

## Evaluation
- Information Transfer `IT=H(S)+H(R)−H(S,R)` bits; Miller–Madow bias correction. Benchmarks: Tan/Reed/Durlach ~6.5 bits (https://link.springer.com/article/10.3758/BF03207608); single attribute ~1.5–2 bits (Miller 1956); Tadoma ~12 bits/s (Reed & Durlach 1986, https://www.rle.mit.edu/media/pr142/23_Reed.pdf).
- d′: Macmillan & Creelman 2005; 2AFC `d′=√2·z(p_correct)`.
- Tapping: Repp 2005 (https://link.springer.com/article/10.3758/BF03206433); Negative Mean Asynchrony −20 to −80 ms = prediction.
- Study: N=5–8 within-subjects, 3 conditions (buzzer < psychohaptic < neural = hypothesis), 40–60 trials/condition, "IRB-exempt educational pilot, informed consent, anonymized." Report effect sizes + CIs.

**Flags:** real-time HQ separation unsolved; net beating DSP on human metric in 48 h unproven; multi-channel spatial neural generation is the novel contribution (not weekend-guaranteed); music bits/s thin (12–22 bits/s is a *speech* ceiling).
