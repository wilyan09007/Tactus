# Power architecture + the chord-safety guarantee

This is the document that guarantees the system does not brown out when you play a chord. Read it fully before powering anything. **Compute every port load per-bus (amps_on_that_bus × per-amp draw), never as total ÷ ports** — an even-split shortcut understates the real load.

> **The real load is tiny.** Chords render as a **sequential strum sweep**, so **never more than ~2 channels are active at once** → real peak is ~**2–3 W**, not the old "all-channels = 33 W" worst case (which never happens). The per-bus math below is kept for the conservative worst case, but any 5 V source runs the rig with huge margin. See [`../truth.md`](../truth.md) §3.5.

## The numbers (per channel / per amp)

- Each driver at a **felt** level needs only ~0.5–1 W. You are *not* running them at the 5 W max.
- A PAM8403 (the SK473 amp) is ~85% efficient. So 1 W out ≈ 1.2 W in.
- Each SK473 = 2 channels:
  - **felt play:** ~0.5 A @ 5 V per SK473
  - **hard chord (≈2 W/ch):** ~0.95 A @ 5 V per SK473

**Whole-rig totals (all 6 amps) — conservative worst case only:**

```
felt play:  6 × 0.5  A = 3.0 A  = ~15 W
hard chord: 6 × 0.95 A = 5.7  A = ~28 W   (theoretical "all-on" worst case)
```

**In reality this never happens:** the sequential strum sweep keeps ≤2 channels live at once, so the actual peak is ~**2–3 W**. Pulses are also short (30–80 ms), so even a chord is a brief peak, not a continuous load. The all-on totals above are kept only as a safety ceiling.

---

## Mode A — wall (RECOMMENDED for the judged demo)

```
wall outlet -> IPSG 10-port hub (label 10A / 50W @ 5V) -> 6 SK473 amps
laptop      -> its own charger
2x Vantec   -> laptop USB (data; ~1W total, trivial)
```

**Guarantee:** the hub's adapter is labelled **10 A (50 W)** at 5 V (verify the real per-port limit).
- Real strum-sweep peak ~2–3 W → utterly trivial against a 50 W rail.
- Theoretical all-on worst case ~5.7 A (28 W) → headroom = 50/28 = **~1.8×** even if every channel somehow fired together.

The whole rig is on **one** 50 W rail, so there is no per-port split to get wrong. **It physically cannot collapse.** Use this mode whenever an outlet is available — it is the bulletproof demo.

---

## Mode B — cordless (the portability "wow")

```
laptop      -> its own battery (no cord)
2x Vantec   -> laptop USB
6 SK473     -> solder buses -> Anker 737 ports
```

The Anker 737 holds plenty of energy (≈87 Wh → many hours at the rig's ~2–3 W real draw). The limit is **per-port current at 5 V**:
- USB-C ports: 5 V × 3 A = **15 W** each (and see the CC-resistor note below)
- USB-A port: 5 V × 2.4 A = **12 W**

### Per-bus loads (computed the right way)

Split the amps **3 / 2 / 2** across the 3 ports (the 3-amp bus is the worst-case bus). Per-bus = (amps on bus) × (per-amp draw):

| Bus | Amps | Felt (×0.5 A) | All-on worst case (×0.95 A) | Put it on |
|---|---|---|---|---|
| Bus 1 | 3 | 1.5 A | **2.85 A** | a **USB-C** port (3 A) — never USB-A |
| Bus 2 | 2 | 1.0 A | 1.9 A | the other USB-C (3 A) |
| Bus 3 | 1 | 0.5 A | 0.95 A | the USB-A (2.4 A) |

- **In reality (sequential strum sweep, ≤2 channels live): ~2–3 W total — every bus far under its port limit. Big margin, fully safe.**
- Even at a theoretical hard-cranked all-on chord, Bus 1 hits 2.85 A = **95% of the 3 A USB-C limit** — within limit *only if Bus 1 is on a USB-C port* (the 2.85 A bus on the 2.4 A USB-A port would brown out). That case never occurs with the strum sweep, but keep Bus 1 on USB-C regardless.

**Mode B guarantee:** 3/2/2 split, the **3-amp bus on USB-C** (never USB-A), volume at felt level → every port ≪ its limit → chords safe with margin. For a zero-margin-of-error chord guarantee, use **Mode A**.

### Fewer-ports fallback
- **2 ports:** felt total ~3.0 A → ~1.5 A/port. Safe at felt level. A theoretical all-on chord = ~2.85 A/port, which approaches the **3 A USB-C limit** → only run 2-port at felt level, never crank.
- Building the buses: see [04-soldering-guide.md](04-soldering-guide.md) Job 3, including the **USB-C CC-resistor** requirement.

### Verify per-bus current for real
**Do not trust the Anker's display for per-port safety** — it reports total/aggregate, not reliable per-port current. During the chord stress test, measure each bus with the **multimeter in series** (or a cheap USB inline power meter on each cable). Size the split off the measured numbers.

---

## Hard rules (break these and you fry something)

1. **Audio and power never share a power wire.** The only laptop↔amp link is the 3.5 mm signal. Note: the 3.5 mm cable *does* share a signal ground, but it carries only line-level current (µA–mA) and the amp's 5 V rail is **never** tied to the signal lines — so a fault can't push amp power back into the laptop. The protection is "negligible signal-ground current + isolated 5 V rail," **not** "no DC path at all."
2. **Polarity is strict.** 5V↔5V, GND↔GND. **Never let 5V touch GND** (dead short → trips/damages the bank or hub). Meter every power cable before connecting.
3. **Never power the amps from the laptop's USB ports** (nor the Pi's) — that overloads those buses. Amps get power only from the hub (Mode A) or the bank (Mode B).
4. **Set level in software, at felt level.** The PAM8403 volume pots are bypassed (level is per-channel in software). Felt haptics need ~1 W/ch; the KHD drivers are rated 5 W, so headroom is fine, but cranking wastes power and pushes you toward the Mode-B port limits. Punch comes from coupling, not overdrive.
5. **Vent the enclosure.** The amps run warm; do not seal them airtight.

---

## Pre-power checklist (every session)

- [ ] Meter each power feed: confirm 5 V, correct polarity, no 5V–GND short.
- [ ] Mode B USB-C feeds: confirm the 5.1 kΩ CC resistors are in place (else the port stays off).
- [ ] Per-channel software level at minimum, then bring up to felt level.
- [ ] Mode A: hub on its adapter, to wall. Mode B: amps split 3/2/2, 3-amp bus on USB-C.
- [ ] Laptop on its own power; the 2 Vantecs on laptop USB.
- [ ] Play a single test tone on one channel before the full array.
- [ ] Full chord: Mode B — measure each bus with a multimeter/inline meter (not the Anker display) and confirm under each port's limit.
