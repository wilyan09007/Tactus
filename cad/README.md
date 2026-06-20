# CAD — the printed enclosure (FlashForge Adventurer 5M)

Three parametric OpenSCAD parts that house **everything that is not the vest or the subwoofers**, plus the actuator coupling pucks that turn a 40 mm cone speaker into a body tactor.

| File | What it is | Print qty |
|---|---|---|
| `tactus_box.scad` | Main "brain-pack": 3× Vantec + 7× SK473/PAM8403 amp bricks + Pi 5, vented, with strain-relief comb + zip-tie floor. Renders `base` and `lid`. | 1 base + 1 lid |
| `tactus_power_cradle.scad` | Open vented sled for the **Anker 737** (Mode B) / **10-port hub** (Mode A). Removable, screen + ports stay accessible. | 1 |
| `actuator_puck.scad` | Cup that grips a 40 mm speaker + a dust-cap **contact button** → couples vibration into skin (docs/15 §4). Renders `cup`, `button`, or `both`. | ~16 cups + 16 buttons |

All three fit the **5M's 220 × 220 × 220 mm bed** with margin (largest part, the box base, is **199.8 × 161.8 × 60.4 mm** — verified).

> **Pre-rendered STLs are already in this folder** (`tactus_box_base.stl`, `tactus_box_lid.stl`, `tactus_power_cradle.stl`, `actuator_cup.stl`, `actuator_button.stl`) — drop them straight into the slicer. Re-render only if you change a parameter.

---

## Why these shapes (the 3 things they solve)
1. **Strain relief is built in, not bolted on.** The box's `comb_ledge` + per-wire holes + internal `cable_anchors` take the 14-wire umbilical load off the amp output pads — the single most-documented failure mode (`docs/06-safety.md`). Every actuator puck has its own wire notch + zip-tie hole.
2. **The cone speakers become tactors.** The puck seals the speaker (less sound to the room), and the **contact button** on the dust cap pokes the cone's motion straight into the body (`docs/15 §4`). Without this they are *heard, barely felt*.
3. **Dimension-tolerant.** Several of our part sizes are estimated, so boards mount with **VHB foam + zip-ties through a floor slot grid**, not tight press-fit pockets that fail if a number is off by 2 mm. Only the Pi's mount is a precise boss pattern (its holes are a known 58 × 49 mm).

---

## Render → STL
The team slices these, so export STL first:

**GUI:** open each `.scad` in OpenSCAD → set the `part =` line at the top → **F6** (full render) → **File ▸ Export ▸ Export as STL**.

**Headless (one-liner per part):**
```bash
OS="/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"   # macOS cask path
$OS -D 'part="base"' -o tactus_box_base.scad.stl  tactus_box.scad
$OS -D 'part="lid"'  -o tactus_box_lid.stl         tactus_box.scad
$OS -o tactus_power_cradle.stl                     tactus_power_cradle.scad
$OS -D 'part="cup"'    -o actuator_cup.stl         actuator_puck.scad
$OS -D 'part="button"' -o actuator_button.stl      actuator_puck.scad
```

Edit the parameters at the top of each file to fit your *measured* parts (e.g. `inner_z`, `dev_l`, `spk_dia`). They're all named and commented.

---

## Slicing for the FlashForge Adventurer 5M
The 5M is **open-frame** (no chamber). Use **PETG** for the box + cradle (better heat tolerance than PLA near warm amps + the power bank; tougher screw bosses). **PLA is fine** if you want the easiest, fastest print and keep volume low. Pucks: **PETG or TPU** (TPU = softer, kinder skin contact).

| Setting | Box base / lid | Power cradle | Actuator puck |
|---|---|---|---|
| Material | PETG (PLA ok) | PETG | PETG or TPU |
| Layer height | 0.2 mm | 0.2 mm | 0.2 mm |
| Walls / perimeters | 3 | 3 | 3 |
| Top/bottom layers | 4 | 4 | 4 |
| Infill | 15–20% gyroid | 15% gyroid | 20% |
| Supports | **none** | **none** | **none** |
| Build plate adhesion | brim (5 mm) — the base footprint is large | brim | skirt |
| Orientation | as-modelled (floor down, open top up) | floor down | cup open-side **up**; button flat-side down |
| Nozzle / bed (PETG) | 235 / 80 °C | 235 / 80 | 235 / 80 |
| Est. print time | base ~6–9 h, lid ~3–4 h | ~3–4 h | ~12 min each |

**Use the multiple printers + queue in parallel:** base on one, lid on a second, cradle + a tray of pucks on a third. Nothing here needs supports, so prints are hands-off.

> Bed margin check: box base is ~200 × 155 mm on a 220 mm bed → ~10 mm/side for brim. If you want more margin, drop `inner_x`/`inner_y` to 185/145 in `tactus_box.scad` (boards then stack 2-high — they're light).

---

## Assembly order (matches `docs/09-assembly-checklist.md`)
1. Print everything. Tap the Pi standoffs (M2.5) and lid bosses (M3) with the screw itself — self-tapping into PETG.
2. **Pucks first** (they gate Stage 2 tone-tests): solder the 2-wire pair to each 40 mm speaker, strain-relieve through the puck, glue a contact button to the dust cap. Verify *one* feels strong on skin before building 14.
3. Mount Pi (if used) on the standoffs; lash the 3 Vantecs + 7 amp bricks to the floor grid with zip-ties over foam; amps nearest the comb wall.
4. Run all 14 actuator wires out the comb; lash the bundle to the internal `cable_anchors`, then again at the comb → both-end strain relief.
5. USB cables out the −Y glands to the laptop; power feed out its gland to the cradle.
6. Drop the Anker (or hub) into the cradle; velcro it; belt the cradle beside the box.
7. Lid on, 4× M3.

## ArUco marker (vision)
The fretboard homography (`docs/08`) is far more robust with a fiducial. Print an **ArUco marker** (e.g. 4×4_50, id 0, ~30 mm) on paper and tape it to the guitar headstock — no CAD needed; generate one at any ArUco generator or with `cv2.aruco`. (A small printed clip for it is a nice-to-have, not modelled here.)

## Parameters you'll most likely touch
- `tactus_box.scad`: `inner_x/y/z` (volume), `n_wires` (comb slots), `pi_holes` (move the Pi).
- `tactus_power_cradle.scad`: `dev_l/w/h` (set to whatever you actually belt in).
- `actuator_puck.scad`: `spk_dia`, `spk_depth` (measure your speaker), `btn_dome_h` (skin poke).
