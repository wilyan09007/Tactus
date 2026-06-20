# Open questions + to-do

Live list of what's unconfirmed or unbought. Update as you close items.

## Confirm
- [ ] **KHD driver diameter** — the de-housed SK473 KHD 3 Ω/5 W drivers are the actuators, but the diameter is **un-measured**; CAD uses placeholders (`spk_dia=40`). Measure the de-housed driver → set `spk_dia`/`drv_dia` in the `.scad` files → re-render the STLs. (truth.md §3.3, top open dim.)
- [ ] **ALSA enumeration** — the CM6206 Vantecs reorder across reboots. Bind by `/dev/snd/by-id`, verify in-card channel order (`speaker-test`), fill `config/channel_map.json` empirically. (truth.md §3.2.)
- [ ] **Webcam** — laptop cam enough, or buy an external USB webcam? Print an **ArUco marker** for the headstock for robust fretboard tracking.

## Buy / make
- [ ] **2× USB-C cables** for the Mode-B bus feeds (or USB-C→USB-A-female adapters to skip the CC resistors).
- [ ] **5.1 kΩ resistors** if feeding buses from bare-cut USB-C cables (see [04-soldering-guide.md](04-soldering-guide.md)).
- [x] **3D-printed enclosure** — **done: see `cad/`** (`tactus_box` = 2–3 Vantec + 6 amps (+ Pi if used), vented, strain-relief comb; `tactus_power_cradle` = Anker 737 / hub; `actuator_puck` = de-housed KHD-driver coupler — `spk_dia` is a 40 mm placeholder, re-render once the driver is measured). Pre-rendered STLs included, sized to the FlashForge 5M 220³ bed. Build notes in `docs/15-build-refinements.md`.
- [ ] **Mounting** — compression shirt/vest, VHB foam tape, rigid backers (corrugated plastic/acrylic), velcro, zip ties.

## Decide
- [ ] **Judged run on Mode A (wall) or Mode B (cordless)?** Recommend Mode A for reliability, Mode B as the "and it goes cordless" flex.
- [ ] **Real-time vs pre-processed transcription** for the demo. Live guitar → live F0 (pYIN / YIN); polished run → pre-process the song (CREPE offline). Known target song de-risks both.

## Repo / logistics
- [ ] **Push to the team remote.** Currently local-only. `git remote add origin <url> && git push -u origin <branch>`. Send the URL.
- [ ] Decide branch/PR workflow for teammates.

## Known limitations (state honestly in the demo)
- Polyphonic transcription accuracy degrades on dense/fast playing (vision cross-checks).
- "Pressure" is a **2-class ordinal (too-light / good)** recovered by inverting the buzz surface, not measured force. "Too hard" is a separate pitch-cents fault.
- Placement is **vs the fret-wire (coarse)**, not exact centimeters.
- Mode B at a hard-cranked chord is tight on the 3-amp USB-C bus — run at felt level, or use Mode A.
