# Soldering guide

Three jobs: (1) gut each SK473 into a bare amp brick, (2) attach actuators, (3) build the 5V bus (cordless mode). Skill bar: basic soldering. Budget ~10 min per amp.

**Before you start:** read [06-safety.md](06-safety.md). Every joint gets heat-shrink + a hot-glue strain-relief blob — non-negotiable, because 18 AWG zip-cord weight tears unrelieved joints. **Always pre-load heat-shrink onto the wire before you solder** — it is a closed tube and cannot be slid over a joint once both ends are terminated.

---

## Job 1 — gut an SK473 (×6)

What's inside an SK473:

```
USB lead (5V power) --+
                      +--[ PAM8403 amp board ]-- 2 output pairs --> 2 stock KHD 3Ω/5W drivers
3.5mm plug (audio) ---+        + volume pot
```

1. **Open the housing** — 3–4 screws or pry the clips.
2. **Identify the board.** It has: a USB power lead (red 5V / black GND), a captive 3.5 mm plug, a volume pot, and two speaker-output pairs (L+/L−, R+/R−) running to the two stock KHD 3 Ω/5 W drivers.
3. **De-house the two KHD drivers, keep them.** They are the actuators (mass-y, punchy). Desolder them from the output pads, note pad polarity (silk marking or wire color), and set the bare drivers aside for coupling (`docs/15`). The 40 mm LEO speakers are spares — do not use them here.
4. **KEEP:** the USB power lead, the 3.5 mm plug, and the two KHD drivers. The volume pot is **bypassed — level is set per channel in software** ([06-safety.md](06-safety.md)), so leave it as found (you can't exceed the chip's fixed gain anyway).

Result: a bare 2-channel amp brick — power in, 3.5 mm audio in, two output pad-pairs — plus its two de-housed KHD drivers.

---

## Job 2 — attach actuators to the amp (×12)

The actuators are the de-housed **KHD 3 Ω/5 W drivers** from Job 1. The PAM8403 is **BTL/filterless** — both terminals of each channel (`+` and `−`) are driven, neither is ground. **Hard BTL rules:** never tie any output `−` to ground; never join two channels' `−`; **every driver gets its own isolated 2-wire pair.**

1. **Cut a 2-wire 18 AWG zip-cord pair to length** — long enough to reach the driver's body position from the amp, plus a service-loop allowance ([06-safety.md](06-safety.md)). These runs are variable length; cut per the wiring map.
2. **Pre-load** two pieces of heat-shrink onto each lead and slide them well clear of the ends.
3. **Tin** the amp output pads and the KHD driver tabs.
4. **Solder one isolated pair to L+/L−, one isolated pair to R+/R−.** Keep **+ to +** consistent across the *entire* array (phase matters for clean multi-actuator behavior).
5. **Slide the heat-shrink over each joint and shrink it.**
6. **Tone-test now (before any glue): feed 5 V + a tone through this amp → both KHD drivers should buzz.** Label the channel (e.g. `V1-front-L`). If a board fails, swap in a spare — **do not** pot a board you haven't verified.
7. **Only on a board that passed:** hot-glue a strain-relief blob anchoring the wire to the board edge and to the driver body. The glue carries the wire load, never the solder. (Also anchor the run to the enclosure within a few cm of the amp — see [06-safety.md](06-safety.md).)

Impedance note: the KHD drivers are **3 Ω/5 W** and factory-matched to the PAM8403, so 3 Ω is safe at our felt level. Per the hard BTL rule, **one driver per channel on its own isolated pair** — never common two channels' `−`, never parallel drivers on one channel.

---

## Job 3 — build the 5V solder bus (cordless / Mode B)

Goal: power all 6 amps from the Anker 737, split across ports so chords don't brown out. See [03-power.md](03-power.md) for the per-bus split math (3/2/2; the 3-amp bus must land on a USB-C port).

1. **Cut each SK473's USB power lead**, expose 5 V (red) + GND (black). **Meter to confirm** which is which — do not trust color blindly.
2. **Group the amps** to match the port split (Bus 1 = 3 amps → USB-C; Bus 2 = 2 amps → USB-C; Bus 3 = 2 amps → USB-A).
3. For each bus: **pre-load heat-shrink**, then **join all 5 V leads to one rail, all GND leads to one rail** (a short length of 18 AWG as the rail backbone).
4. **Feed each bus from an Anker port** via a sacrificial cable:
   - **USB-A feed (Bus 3):** any USB-A cable — cut, meter VBUS/GND, solder VBUS→5V rail, GND→GND rail. USB-A always outputs 5 V.
   - **USB-C feed (Bus 1, Bus 2):** a cut USB-C cable presents **no CC pull-down**, so a USB-C PD source like the Anker will **refuse to turn the port on**. Add a **5.1 kΩ resistor from each CC line (CC1, CC2) to GND** on the cable end going into the Anker, so the source detects a sink and enables 5 V. *Simpler alternative:* use a USB-C→USB-A-female adapter on the Anker port, then a USB-A cable (the adapter handles CC) — no resistors needed.
5. **Shrink all joints.** Keep the 5 V rail and GND rail physically separate — a 5V–GND touch shorts the Anker.
6. **Verify:** meter each bus's 5 V and (during the chord test) its current in series.
7. Wall-mode alternative (Mode A): skip the bus — plug each SK473's USB lead into the 10-port hub instead. The hub's 50 W handles all 6, no CC resistors, no split.

---

## Joint quality
- Tin first, heat the joint not the solder, ~1–2 s, shiny cone (not a dull blob).
- Fine tip on the small amp pads; don't bridge adjacent pads.
- Inspect every joint under light before powering.
- Heat-shrink (pre-loaded) + hot-glue **every** joint (insulation + strain relief), glue only after the tone test passes.

## Order of operations (matches [09-assembly-checklist.md](09-assembly-checklist.md))
**Gut all 6 → attach + tone-test each amp individually → build the bus → powered chord/stress test → ONLY THEN mount actuators on the body.** Never mount before a board passes its tone test, and never pot strain relief before the tone test.
