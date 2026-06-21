# Assembly checklist + bring-up

Build in this order. Each stage has a go/no-go gate — do not proceed past a failed gate. The biggest risk now is **build time**, so prove a small slice end-to-end before scaling.

## Stage 0 — bench prep
- [ ] Lay out all 6 SK473 (their 12 KHD drivers are the actuators), 2 Vantecs + V3 spare, 18 AWG wire, iron, multimeter, heat-shrink, hot glue.
- [ ] Confirm the Anker 737 charged; confirm the 10-port hub's 10A adapter present.
- **Gate:** all parts present per [01-bill-of-materials.md](01-bill-of-materials.md).

## Stage 1 — prove the audio path (BEFORE any soldering)
- [ ] One Vantec into the laptop. `aplay -l` shows the card.
- [ ] `speaker-test -c8 -Dhw:CARD` — confirm all 8 channels map to the 4 jacks.
- [ ] Plug one un-gutted SK473 into a Vantec jack, power it, play a tone → its speaker sounds.
- [ ] Repeat for all 3 Vantecs (confirm each enumerates independently).
- **Gate:** every Vantec maps cleanly. If a jack misbehaves → use Vantec V3 / a Sabrent dongle for that channel. **Don't solder until this passes.**

## Stage 2 — gut + tone-test each amp (×6)
- [ ] Gut each SK473, keeping its 2 KHD drivers ([04-soldering-guide.md](04-soldering-guide.md) Job 1).
- [ ] Solder 2 KHD drivers per amp, each on its own isolated 2-wire pair (Job 2). Heat-shrink + hot-glue every joint.
- [ ] Tone-test each amp individually → both KHD drivers buzz. Label the channel.
- **Gate:** all 12 channels buzz on a tone. Replace any dead board from the spare SK473.

## Stage 3 — power
- **Mode A (wall):** plug all 6 SK473 USB leads into the 10-port hub; hub to wall.
- **Mode B (cordless):** build the 5V buses (Job 3), split 3/2/2 across Anker ports.
- [ ] Meter polarity on every feed. Gain knobs at minimum.
- **Gate:** [03-power.md](03-power.md) pre-power checklist passes; no shorts.

## Stage 4 — chord stress test
- [ ] Fire a single channel → felt.
- [ ] Fire all 12 at once (simulated chord) at felt gain. (Real playback is a sequential strum → ≤ 2 drivers active at once, ~2–3 W; firing all 12 is a worst-case stress check.)
- [ ] Mode B: watch the Anker display — each port under its limit.
- **Gate:** no brown-out, no cut-outs. If Mode B dips → add the 3rd port / lower gain.

## Stage 5 — mount + couple
- [ ] Mount actuators per [05-wiring-map.md](05-wiring-map.md): 6 strings on back, 6 fret-zones on torso.
- [ ] VHB + rigid backer, firm coupling. Service loops + garment anchors so no actuator bears wire weight ([06-safety.md](06-safety.md)).
- **Gate:** each site is clearly felt and mechanically secure under arm movement.

## Stage 6 — software bring-up
- [ ] Channel map ([config/channel_map.json](../config/channel_map.json)) matches the wiring table.
- [ ] Renderer plays a known (string, fret) → correct body site lights.
- [ ] AI engine: webcam + mic → transcription + vision overlay.
- [ ] Coach loop end-to-end on one phrase.
- **Gate:** play a note → felt in the right place + shown on screen + coached.

## Stage 7 — demo dry run
- [ ] Full song, Mode A (wall) for reliability.
- [ ] Then Mode B cordless for the walk-around flex.
- [ ] Time it; rehearse the failure fallback (Vantec→dongle, Mode B→Mode A).

## Go/no-go summary
Audio path ✓ → amps buzz ✓ → power holds a chord ✓ → mounted + coupled ✓ → software renders + coaches ✓ → demo rehearsed ✓.
