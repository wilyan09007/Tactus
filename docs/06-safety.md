# Safety — eliminate every risk

Two classes of risk: **electrical** (frying parts) and **mechanical** (wire weight tearing joints, the thing you specifically worried about). Both are fully avoidable. This is a checklist, not a suggestion.

## Mechanical — the wire weight problem (this breaks things)

18 AWG is **stiff and heavy**. Its weight, plus your arm moving while you play, acts as a lever on every solder joint. Unmanaged, it does exactly two failures:

1. **Rips the output pad off the SK473 board** (the pad is tiny; the wire is a crowbar).
2. **Tears the lead off the KHD driver** (the driver tab is thin solder).

### Eliminate it — every joint, no exceptions
- **Strain-relief blob.** After soldering each joint, cover it with **hot glue or epoxy** so the wire is anchored to the board/actuator body, not hanging off the solder. The glue takes the load, not the joint.
- **Service loop.** Leave a small slack loop of wire near each actuator and **anchor the wire to the garment** (zip tie / stitched loop / VHB) a few cm before the actuator. Now the actuator **never bears wire weight** — the garment does.
- **Anchor to the body, not the actuator.** Bundle and zip-tie wire runs to the vest, not to the KHD driver. The actuator only sees its own short pigtail.
- **Anchor the amp end too (symmetric).** Within a few cm of the SK473, zip-tie/anchor each run to the enclosure so the wire load is carried by the housing, not by the amp's output pad. Both ends of every run are relieved — actuator end *and* board end — or the heavy 18 AWG levers a joint somewhere.
- **Heat-shrink over every joint** for both insulation and a second mechanical anchor.
- **Don't dead-hang the brain pouch by its cables.** The enclosure carries its own weight via a strap; cables stay slack.

### Mounting the actuator (so it's felt, not heard, and stays put)
- Couple the actuator **frame flat to the body** with VHB foam tape over a small rigid backer (corrugated plastic / acrylic). Firm coupling = felt; loose = you only hear it.
- The rigid backer also spreads load so the actuator doesn't peel off during movement.

## Electrical

- **Polarity, every power cable.** Meter before connecting: red = 5 V, black = GND. Consistent +/− across all actuators.
- **Never short 5V to GND.** On the solder bus this is the #1 way to trip/damage the Anker. Insulate the two rails separately; heat-shrink all bus joints.
- **BTL output — never ground a `−`.** The PAM8403 is BTL/filterless: both output terminals (`+` and `−`) are driven, neither is ground. **Never tie any channel's `−` to GND or common; never join two channels' `−`; every driver gets its own isolated 2-wire pair.**
- **Don't bridge amp output pads.** Fine tip, inspect under light, no solder bridges between L+/L−/R+/R−.
- **Audio ≠ power.** Brain links to amps only via the 3.5 mm signal. That cable *does* share a signal ground, but it carries only line-level current (µA–mA) and the amp's 5 V rail is never tied to the signal lines — so amp power can't flow back into the laptop. The protection is "negligible signal-ground current + isolated 5 V rail," not "no DC path exists."
- **Impedance.** The KHD drivers are **3 Ω/5 W**, factory-matched to the PAM8403 — safe at our felt level. **One driver per channel on its own isolated 2-wire pair** (the BTL rule above). Never parallel drivers on a channel, and never common two channels' `−`.
- **Power source discipline.** Amps from hub (Mode A) or bank (Mode B) only — never the Pi/laptop ports.
- **Thermal.** Amps get warm. Vent the enclosure; don't seal the boards airtight; don't run at high volume (you never need to).
- **Gain low to start.** The volume pots are **bypassed — level is set per channel in software**; start low and ramp to felt level (you can't exceed the chip's fixed gain). Clipping/distortion sends near-DC into the voice coil and kills it even at low average power.

## Risk table (and how each is eliminated)

| Risk | Cause | Eliminated by |
|---|---|---|
| Pad torn off amp | wire weight leverage | hot-glue strain relief + service loop anchored to garment |
| Lead torn off actuator | arm motion + stiff wire | actuator bears no wire weight; backer spreads load |
| Brown-out on a chord | too many amps on one 5V feed | Mode A 50W hub, or Mode B split 3/2/2 across ports |
| Amp cooked | drivers paralleled, `−` grounded/commoned, or clipping | one isolated pair per channel; software gain at felt level |
| `−` tied to ground/common | misreading BTL output as single-ended | every driver on its own isolated 2-wire pair; never ground a `−` |
| Bank tripped/damaged | 5V shorted to GND on the bus | meter polarity; insulate rails; heat-shrink |
| Voice coil burned | sustained clipping | start at low software gain, never run to distortion |
| Short between channels | solder bridge on output pads | fine tip, inspect, no bridges |
| Overheating in enclosure | sealed amps | vent the 3D housing |

## One-line pre-flight
Meter polarity → gain at min → one tone on one channel → full chord (watch Anker display in Mode B) → ramp gain to felt level. If anything buzzes/cuts out, stop and check the bus split + joints before continuing.
