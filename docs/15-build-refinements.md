# 15 — Build refinements + the kinks that will bite (read before Saturday)

> This doc **does not replace** `01`–`09`/`14` — it sharpens them. It collects the *non-obvious failure modes* in the as-purchased build (laptop → 3× Vantec → 7× SK473/PAM8403 → 14× 40 mm speakers) and gives the exact fix for each, plus a tightened "what connects to what" map and a single-channel-first build order. Every item here is a thing that, skipped, silently costs you hours or a fried board on Saturday.

---

## 0. The 60-second mental model of the whole rig

```
LAPTOP ── USB data ──> 3× VANTEC (USB→8 analog ch each)
                           │  3.5 mm STEREO PLUG  (each plug = 2 channels = one amp)
                           ▼
                        7× SK473 amp brick (gutted → bare PAM8403, 2 ch each)
                           │  2-wire SPEAKER PAIR per channel  (BTL: + and − are BOTH driven)
                           ▼
                        14× 40 mm speaker (felt, not heard) on the body

POWER (5 V, separate from all audio):  hub (wall, Mode A)  OR  Anker 737 bus (cordless, Mode B)
        → feeds ONLY the amps' 5 V USB leads.  Laptop + Vantecs powered by the laptop's own USB.
```

Three rails of truth that everything else hangs on:
1. **Audio is line-level on 3.5 mm; actuator drive is bare 2-wire from the amp.** They never touch.
2. **Power (5 V) is a third, independent thing.** It only ever reaches the amps.
3. **One ground is shared** (laptop ↔ Vantec ↔ 3.5 mm sleeve ↔ amp). That is intended and safe — but it means the *only* isolation you have is "+5 V is never tied to a signal line." Respect it (see `06`).

---

## 1. ⚠️ THE BIG ONE — PAM8403 is a BTL amp: never common the "−" outputs

This is the single easiest way to kill an amp on this build, and it's the kind of thing a tidy-minded person does *on purpose* while "cleaning up the grounds." **Do not.**

- The PAM8403 is a **filterless Class-D, bridge-tied-load (BTL)** amp. For each channel the **two output terminals (`+` and `−`) are BOTH actively driven** — they swing in opposite directions around ~½·VCC. **Neither one is ground.**
- Therefore:
  - ❌ **Never connect any output `−` to GND.** (You short a half-bridge to ground → over-current → dead chip.)
  - ❌ **Never join two channels' `−` together, or run a single shared "return" wire for several actuators.** (You short two half-bridges into each other.)
  - ✅ **Every actuator gets its own isolated 2-conductor pair** straight back to *its* channel's `+`/`−`. The 18 AWG you bought is 2-conductor zip cord — one length = one channel. Perfect, *as long as you never bond the returns.*
- **Phase:** keep `+ → +` consistent across the whole array (mark one conductor of the zip cord — the ribbed/striped side — as `+` everywhere). Two actuators fired together then move in phase, which the encoder (`07`) assumes.
- This is *different* from the old MAX98357A plan, which was capless-but-single-ended. The encoding/wiring docs already say "one pair per channel"; this section is the *why*, so nobody re-routes a ground bus and fries A3.

> Quick check before powering: meter resistance from each output `−` to the amp's power GND. It should **NOT** be ~0 Ω. If any `−` reads a dead short to GND, you have an accidental bond — find it before you apply 5 V.

---

## 2. ⚠️ The Vantec/ALSA enumeration trap (this eats Saturday if you don't pre-empt it)

You have **three identical CM6206 cards**. On Linux they all enumerate as the *same name* ("USB Sound Device" / "C-Media USB Audio"), and **their card indices (`hw:0/1/2`) reorder across reboots and re-plugs.** If your renderer addresses cards by index, channel 1 silently jumps to the wrong Vantec → notes felt in the wrong place → a "bug" that's really just a shuffled index. Two real gotchas, two fixes:

**A. Bind cards by physical USB port, not by index.** Plug the 3 Vantecs into 3 *specific, labelled* laptop/hub ports and address them by stable path. Get the stable names with:
```bash
ls -l /dev/snd/by-id/        # stable per-device symlinks
cat /proc/asound/cards       # index ↔ usb path
aplay -l                     # card/device numbers (these can move!)
```
Use the `by-id`/`by-path` name (or a tiny udev rule mapping each USB port → a fixed ALSA name) and store **that** in `config/channel_map.json`'s `card_alsa`, not a bare `0/1/2`. Label each Vantec **V1/V2/V3** physically and always plug it into its own port.

**B. The 8-channel *order within a card* is not the obvious one.** CM6206 commonly presents channels as **FL, FR, FC, LFE, RL, RR, SL, SR** — i.e. **center/LFE come before the rears**, not in jack order. The current `channel_map.json` assumes `front=0,1 / rear=2,3 / center=4,5 / side=6,7`. **Verify the true mapping before you trust it:**
```bash
speaker-test -D hw:CARD,0 -c 8 -t sine    # walks ch 0→7; note which JACK each one comes out of
```
Plug a known amp+speaker into one jack at a time and learn `(alsa index → jack)` empirically. Then fix `alsa_ch` in the map. **Do this in Stage 1, before any soldering** (`09` already gates on it — this is the *why*).

**C. The center jack's ch 4 slot is "LFE."** On a raw `hw:` device it passes full-band (bass-management is an OS mixer feature, not applied to a direct device), so it's fine for our 200–250 Hz pulses — but **confirm it isn't attenuated** during the speaker-test. If any one channel is quiet, move that *logical* channel onto the spare Vantec **V3** rather than fighting it.

---

## 3. Wire gauge — you only have 18 AWG, so the build has to respect it

> **Reality (per Aditya): the only wire on hand is the AudioVox 18 AWG speaker zip-cord.** The BOM's "22 AWG solid-core" line (`docs/01`) is **stale — that wire is not physically here.** So plan everything around 18 AWG. The good news: speaker zip-cord is **stranded**, so it flexes and won't snap; the bad news is it's heavy/stiff and **fat for the tiny amp pads**. Two consequences:

**A. It tears joints (mechanical).** 18 AWG weight + arm motion levers the joint — the whole reason `06` exists. With no lighter wire to fall back on, strain relief is **mandatory, not optional**:
- Keep every actuator run **as short as the layout allows** + one small **service loop**.
- **Strain-relieve both ends** of every run — the `actuator_puck` zip-tie notch at the body end, the box's `comb` + `cable_anchors` at the amp end. The puck/comb are doing the job the missing thin wire would have done.

**B. It's too fat to land cleanly on a PAM8403 output pad or a 40 mm speaker tab (electrical/solder).** Cramming a full 18 AWG conductor onto a ~1 mm pad will **lift the pad** (a permanent kill). Fix at the joint:
- **Fan and trim the strands** at the pad end — solder maybe half the strands, snipped flush, so you land a manageable tinned bundle, not the full crowbar. (0.3–0.5 A doesn't need the full copper anyway.)
- **Tin the pad and the wire first**, land fast, then a **hot-glue blob is non-negotiable** — it, not the solder, carries the load.

**C. The single best upgrade is still cheap: grab one spool of 24–26 AWG stranded silicone** (~$7, any hardware/hobby store). It solders onto the small pads easily *and* solves the weight problem in one move — by far the highest-leverage "still to get." If you can grab it before Saturday, do; if not, the all-18-AWG plan above works with discipline.

**Power bus (Mode B):** here 18 AWG is *correct* — that's where real current flows (~2.85 A peak/bus). Use it for the 5 V rail backbone with no reservations.

> I widened the `actuator_puck` wire notch to fit 18 AWG zip-cord. If a run is still tight, pass the two conductors through **separately** or bump `notch_w` in `actuator_puck.scad`.

---

## 4. The 40 mm "speakers" are not exciters — coupling is the make-or-break

A Dayton exciter is *designed* to inject vibration into a surface through a sticky puck. A 40 mm cone speaker is designed to throw sound into **air**. Pressed naively against a shirt it will be **heard, barely felt** — most of the energy radiates as sound and the frame stays still. To make it *felt*:

- **Load the moving mass into the skin.** Bond a small **contact button to the centre of the cone/dust-cap**, and press *that* button to the body, while the **magnet/frame is anchored to a rigid backer.** Now the cone's excursion couples straight into tissue. (This is exactly what the `cad/actuator_puck.scad` part does: a domed contact face on the cone side + a rigid backer on the magnet side + VHB + a wire strain-relief notch.)
- **Firm preload.** Snug compression fabric / a light elastic strap raises sensitivity a lot (contact force directly lowers the detection threshold — `12`/`14`). Loose = you hear it; firm = you feel it.
- **Foam-isolate the backer from the garment** so one actuator doesn't shake its neighbours into a blur (crosstalk — `07` design rule #1). Keep sites ≥ 4–5 cm apart (already in `05`).
- **Seal optional:** dropping the speaker into a small closed cup (the puck's pocket) kills its air-radiation and forces more energy into the body — quieter to the room, stronger on the skin. Good for a loud venue.

> Saturday's **#1 risk is still coupling** (same as every version of this project). Prove *one* actuator + puck + strap feels strong **before** you build 14. If a bare speaker feels weak, the puck — not more volume — is the fix.

---

## 5. Power: correct the "50 W rail can't collapse" claim (Mode A)

`03` treats the wall hub as "one 50 W 5 V rail → physically cannot collapse." A powered USB hub is **not** a dumb 5 V rail — verify two things or it can still droop on a chord:

- **The brick is almost certainly 12 V, ~4–5 A (≈48–60 W),** stepped down to 5 V *inside* the hub — not "10 A at 5 V." After buck efficiency that's ~**8–10 A available at 5 V**, which still clears the 6.65 A hard-chord peak — **but** only if the hub doesn't enforce a low **per-port** current limit (USB-3 default is 0.9 A/port; charging hubs do 1.5–2.4 A).
- **So the real guarantee is the same as Mode B: run at *felt* level.** At felt play the whole rig draws **~3.5 A / 18 W** — comfortably under any plausible hub budget and per-port limit, with big margin. A hard-cranked 33 W chord is the only thing that could find a per-port ceiling, and you never need to crank.
- **Action:** read the hub's actual adapter rating off its label; in the Stage-4 chord test, meter total 5 V draw. If the hub current-limits a port, spread the 7 amp leads across non-adjacent ports (the hub has 10). The wall path is still the reliable demo path — just don't oversell it as "infinite."

Everything else in `03` (Mode B 3/2/2 split, 3-amp bus on USB-C, the **5.1 kΩ CC resistor** on cut USB-C feeds, meter-don't-trust-the-display) stands — those are correct and important.

---

## 6. Tightened "what connects to what" — the exhaustive connector list

Every physical connection in the system, in order. If a link isn't here, it shouldn't exist.

| # | From | To | Connector / method | Carries | Gotcha |
|---|---|---|---|---|---|
| 1 | Laptop USB-A ×3 | Vantec V1/V2/V3 | USB-A captive cable | data + Vantec's own 5 V | Use **fixed, labelled ports** (§2A). |
| 2 | Vantec jack (×4/card) | SK473 captive 3.5 mm plug | 3.5 mm TRS plug → jack | line-level stereo (2 ch) | Tip=L, Ring=R, Sleeve=GND. One plug = one amp = 2 ch. |
| 3 | Wall **or** Anker | SK473 5 V lead (cut) | USB / soldered 5 V bus | **5 V power only** | Mode A: into hub port. Mode B: onto the 3/2/2 bus (§5, `03`). |
| 4 | SK473 output `L+ L−` | actuator (string/fret) | **soldered 2-wire pair** | amplified BTL drive | §1: isolated pair, never common `−`. |
| 5 | SK473 output `R+ R−` | the *other* actuator on that amp | soldered 2-wire pair | amplified BTL drive | same. |
| 6 | Laptop | webcam | USB (or built-in cam) | video | ArUco marker on headstock for homography (`08`). |
| 7 | Laptop | mic / contact mic | USB / 3.5 mm | guitar audio | clip near the soundhole; keep F0 clean in a loud room. |

**Things that must NEVER connect:** amp 5 V ↔ any 3.5 mm pin; actuator wire ↔ a Vantec; one channel's `−` ↔ another's anything; amp power from the laptop/Pi USB ports.

There are exactly **14 soldered actuator pairs (28 joints) + 7 power-lead taps + (Mode B) 3 bus feeds.** That's the entire soldering scope. Count your joints against this.

---

## 7. Soldering — the order that prevents rework (sharpens `04`)

Build **one full channel end-to-end first**, prove it, *then* mass-produce. Do not gut all 7 and solder 28 joints before testing one.

1. **Gut ONE SK473** (`04` Job 1). Keep: 5 V lead, 3.5 mm plug, volume pot (set ~60%).
2. **Solder ONE actuator** to `L+/L−` (pre-load heat-shrink first — it's a closed tube, can't be added later).
3. **Tone-test that one channel:** Vantec jack → this amp → 5 V → play a 230 Hz tone on the laptop. The 40 mm should buzz; mount it on a puck + strap and confirm it's **felt**. ← this is your real M1.
4. Only now: **gut the other 6, solder the remaining 13 actuators**, tone-testing **each amp** before gluing.
5. **Hot-glue strain relief only on boards that passed** (`04`/`06`). Glue carries the wire load; the joint never does. Anchor **both** ends of every run (board edge *and* actuator body).
6. **Label every wire on both ends** with its channel number *as you solder* (`05`). A mislabeled channel is a silent bug felt in the wrong place.
7. **(Mode B) build the 5 V buses last** (`04` Job 3): meter to find 5 V vs GND on each cut lead, join all 5 V to one rail / all GND to one rail, **CC resistor on USB-C feeds**, keep the two rails physically apart, shrink everything.

Joint quality: tin first, heat the joint not the solder, ~1–2 s, shiny cone; fine tip on the amp pads, inspect under light for bridges. (`04`)

---

## 8. The resolved open questions (update `13`)

| `13` item | Resolution |
|---|---|
| 2 W actuator impedance | Either 4 Ω or 8 Ω is fine on PAM8403. **Meter one coil** (probe the two tabs, expect ~3–4 Ω → it's a "4 Ω"; ~6–7 Ω → "8 Ω"). If 8 Ω, **boost those 6 string channels ~+3 dB in software** so strings aren't half as loud as frets. |
| Old SK473 present? | If only 6 amps → **12 channels**: drop the 2 reserved back zones (z7/z8), keep 6 strings + 6 torso fret-zones. No re-wire of the rest. Design accommodates this. |
| Webcam | Laptop cam works for a v1; an external USB cam aimed *down the neck* is better. **Print the ArUco marker** (`cad/README` has a slot for taping it to the headstock). |
| Enclosure | **Now specified** — `cad/` (box + power cradle + actuator pucks), sized to the FlashForge 5M 220³ bed, vented, with strain-relief comb and zip-tie anchors. |
| Wire for body runs | §3 above — light **stranded** for actuators, 18 AWG only for the power bus. |

---

## 9. The honest demo-reliability stack (aligns with `07` / `09`-assembly)

- **Judged run = Mode A (wall), known song, pre-processed transcription, on the pre-calibrated wearer.** Lowest failure surface.
- **Walk-around flex = Mode B (cordless)** at felt level.
- **Single biggest schedule risk = the 3-Vantec ALSA bring-up (§2).** Do it Stage-1, before soldering, exactly as `09` gates.
- **Single biggest physical risk = actuator coupling (§4).** Prove one puck before scaling.
- Keep a **pre-recorded video** of the working loop as the ultimate fallback (`07`).

---

## 10. Laptop connection + power — exactly what plugs into what

**The brain is the laptop, and the Pi is cut.** There are only ever **two USB "trees,"** and they're independent (they share only the common signal ground via the 3.5 mm sleeves, §0):

- **Tree 1 — DATA: laptop → 3 Vantecs.** The Vantecs are USB sound cards; they need data *and* draw their own power from USB (~1 W each, trivial). They must be downstream of the laptop.
- **Tree 2 — POWER (5 V): wall hub *or* Anker → the 7 amps.** The amps only take 5 V; they exchange no data.

### Recommended: **one powered hub does both** (cleanest "one plug + one cable to the laptop")
The 10-port powered hub is *both* a data hub and a 5 V source. So:
```
wall outlet ─► 10-port powered hub (its own 12V/≈50W adapter)
                 │ upstream USB cable ─► LAPTOP        (carries the 3 Vantecs' audio data)
                 ├─ 3 ports ─► 3 Vantecs              (DATA + their power)
                 └─ 7 ports ─► 7 SK473 amp USB leads  (5 V POWER only)   = 10 ports, exact fit
LAPTOP ─► its own charger/battery   ·   webcam + mic ─► laptop USB
```
That's **one wall plug (the hub) + one USB cable from the hub to the laptop + the laptop on its own power.** Yes — **portable/one-plug + a laptop link is sufficient.** (Per-port caveat from §5: at *felt* level every amp draws ~0.5 A, well under any port limit; don't crank.)

### Cordless variant (the walk-around flex)
```
LAPTOP ─► 3 Vantecs (straight into laptop USB-A, or a small bus-powered DATA hub)
ANKER 737 ─► soldered 5 V bus (3/2/2 split, 3-amp bus on USB-C, CC resistor) ─► 7 amps
LAPTOP + ANKER both on battery → no wall plug at all.
```
Here the amps leave the hub and run off the Anker; the Vantecs go straight to the laptop. The only tether left is the **3-Vantec USB run to the laptop** — short, and the laptop can sit on the cart next to the player.

### If your laptop is USB-C-only (e.g. a MacBook)
You need a **USB-C → multi-USB-A** adapter/dock for the upstream link (the hub's upstream or the 3 Vantecs). Confirm the dock enumerates **3 independent USB-audio devices** — some cheap docks collapse identical devices. The hub-does-both topology needs just **one** USB-C port on the laptop.

### The shopping delta this implies
- **1× USB upstream cable** for the hub→laptop link (and a **USB-C→USB-A(F)** adapter if the laptop is USB-C-only).
- Decide **Mode A vs B for the judged run** (recommend A). For Mode A you can leave the SK473 USB plugs **intact** (just plug into the hub); for Mode B you cut them for the solder bus — so don't cut them until you've committed (or add short USB-A pigtails to keep both options).

---

## 11. SK473 reality (live teardown, Jun 20): it's 6 stereo PAIRS = **12 channels**, and the amp *is* separable

What a "box" actually is: each SK473 box is a **2.0 desktop set** — **one stereo PAM8403 board** lives in the *master* speaker (with the USB lead, the 3.5 mm plug, the volume knob, and the master's own driver), and a **passive *slave* speaker** hangs off a 2-wire cable carrying the amplified **R** channel. So **per box = 1 board (2 ch) + 2 drivers.**

- **Count correction: 6 boxes × 2 ch = 12 channels** (not 24, not 14). Map → **6 back strings + 6 torso fret-zones**, drop the 2 reserved back zones (this is exactly the §8 fallback). Update `config/channel_map.json` (remove ch 13–14) + the body map. *(If a 7th prior SK473 board exists and opens, that's +2 = 14.)* The **24× 40 mm speakers stay our body actuators** — 12 used, 12 spare.
- **"The amp is connected to the speaker" ≠ fused.** The board's **L** output is soldered to the master's driver; its **R** output runs out the cable to the slave driver. Those are **2 snippable joints, not a weld.** Snip both stock drivers off and the board is free. The hard part you're hitting is **cracking the plastic case**, not separating the amp.
- **Open only the MASTER speaker per box** (the one with the board / USB / knob / 3.5 mm). Ignore the slave's enclosure entirely — you only want the board. So it's **6 cases to open, not 12.** You don't even have to fully extract the board — open just enough to disconnect the stock driver and reach the **L+/L−/R+/R−** output pads.
- **Even less soldering than `04` implied:** keep the **USB and 3.5 mm plugs intact** (Mode A) → USB into the hub for 5 V, 3.5 mm into a Vantec jack, **zero input-side cutting.** The only solder is: **remove the 2 stock drivers, attach 2 of our 40 mm to the board's L and R pads** → ~**12 output joints total.** (Cut the USB lead only if you go Mode B / Anker bus.)
- **Opening a stubborn case:** screws usually hide **under the rubber feet or the back label**; else pry the seam with a flat screwdriver; **ultrasonic-welded** seams → score along the seam with a utility knife / rotary tool. Patience here, not force (force cracks the board).

### ❌ Why NOT to strap the whole boxed speakers to the body (the tempting shortcut)
- A **boxed cone speaker radiates into AIR** — it's built to be *heard*. Strapped to the body it's **loud and barely felt**; the box wall isolates the cone's motion from your skin. Haptics needs the **opposite**: a *bare* driver with a contact button on the cone, pressed to the skin (the `actuator_puck`, §4). That coupling is the entire reason it's felt, not heard.
- **Bulk/weight:** ~12 plastic speaker boxes (~340 g each ≈ **4 kg**) is not a wearable vest, and you can't place 12 boxes at the ≥4–5 cm spacing the encoding needs.
- **The amp stays in the enclosure; only the bare 40 mm + wire + puck go on the vest.** The amp does **not** have to live on the body — that worry came from the false "amp is fused to the driver" premise. Snip the stock driver, run a long lead to a 40 mm on the vest, board stays in the box. **The `cad/` enclosure is unchanged** (now 6 boards = even more room).

---

## 12. UPDATE (Jun 20 teardown): use the SK473's OWN drivers (KHD 3 Ω/5 W), and the ≤2-at-once power win

Live finding: the SK473's built-in drivers are **bigger/better than the 40 mm we bought** and are **factory-matched to the amp (KHD, 3 Ω, 5 W)** — the *safest possible* load, because the board was literally built to drive them. **Switch the body actuators to the 12× de-housed SK473 drivers**; the 40 mm become spares. (This is *not* "strap the boxed speaker on" — §11 still stands: you **de-house** the driver so the bare cone couples to skin.)

- **3 Ω on PAM8403:** below the 4 Ω nominal min, but it's the **factory pairing** and we run **felt-level (~1 W) with ≤2 channels ever active** → nowhere near the current/thermal limit. No impedance worry. ("Subwoofer"-ish heavy drivers also tend to feel *punchier* — likely a win; tune freq in `docs/18`.)
- **Each driver is hardwired (red/black) to its amp.** Don't fight that joint — decide where the **amp** lives:
  - **(B) Amps in the box (RECOMMENDED — robust + the soldering is the easy kind):** splice-extend each driver's red/black with your **18 AWG** to reach the vest; amps + pots stay in the vented `cad/` box (sized for 6–7 boards). Only **drivers + wire** on the body → clean, survives being worn/handled. The ~24 joints are **wire-to-wire splices on the existing leads — not tiny-pad SMD work**, so they're beginner-easy.
  - **(A) Amps on the vest (zero output soldering):** keep the factory driver↔amp wiring intact; mount the 6 boards on the vest by their drivers; run **6× audio (3.5 mm) + 6× 5 V (USB)** out to the vest (needs cheap extension cables). Boards are ~10 g and barely warm (≤2 active). The printed box then shrinks to just **3 Vantecs + power**.
  - Pick **B** unless the splices are a dealbreaker; **A** trades a messier/more-fragile body for no soldering.
- **The volume pots — ignore/bypass them; set level per channel in SOFTWARE** (`docs/18` Exp. 4). You **can't "overclock" past the chip's fixed gain** — max pot is the ceiling, and *punch comes from coupling + drive-frequency, not overdrive* (clipping cooks the coil and feels worse). **So don't fear breaking a pot while cracking the case:** worst case, hardwire the input past it = permanent full gain, and control level digitally (which we want anyway for calibration). The pot is **optional**.
- **The ≤2-simultaneous power win (Aditya's insight — correct):** chords render as a **sequential strum sweep** (~40 ms stagger, 30–80 ms pulses) → **never more than ~2 drivers active at once.** This **collapses the power budget**: doc 03's "all-14-at-once ≈ 33 W" case **never happens**; real peak is ~**2–3 W**. Any 5 V source runs the rig with absurd margin, **and you can drive each active channel harder for punch** with zero brownout risk. (Supersedes the doc 03 worst-case framing.)
- **Coupling unchanged:** de-house the driver → **contact button on the cone + rigid backer on the magnet + firm strap** (§4 / `actuator_puck`). **Measure the driver Ø and set `spk_dia`** in `actuator_puck.scad`, then re-render (the puck is parametric).
- **Channel count stays 12** (6 boards × 2). Map → 6 back strings + 6 torso fret-zones.
