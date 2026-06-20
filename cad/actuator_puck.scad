// ============================================================================
// TACTUS — Actuator coupling puck for a 40 mm cone speaker
// ----------------------------------------------------------------------------
// WHY THIS PART EXISTS (see docs/15-build-refinements.md §4):
// A 40 mm cone speaker is built to throw sound into AIR. Pressed naively to a
// shirt it is HEARD, barely FELT. This puck turns it into a tactor:
//   - a closed, rigid CUP grips the speaker frame and kills air-radiation
//     (more energy goes into the body, less into the room),
//   - a domed CONTACT BUTTON bonds to the dust-cap so the cone's excursion
//     pokes straight into the skin,
//   - a WIRE NOTCH + zip-tie hole take all cable load off the speaker tabs,
//   - EARS let you zip-tie / sew the puck to the compression garment.
// Print ~16 cups + ~16 buttons (14 channels + spares). PETG, no supports.
// ============================================================================

part = "both";   // "cup" | "button" | "both" (both = side by side for preview/plate)

// ---- speaker + fit parameters (edit if your 40 mm speaker differs) ----------
spk_dia        = 40.0;   // speaker frame outer diameter
spk_depth      = 7.0;    // speaker total depth (frame face -> magnet back)
spk_fit_gap    = 0.6;    // radial clearance so the speaker drops in (PETG shrink-safe)
rim_lip        = 2.0;    // inward lip that the speaker rim rests against
wall           = 2.4;    // cup wall thickness
back           = 2.4;    // rigid back thickness (the coupling backer)

// ---- contact button (bonds to the dust cap, presses the skin) --------------
btn_dia        = 14.0;   // button diameter (sits on the dust cap)
btn_dome_h     = 4.5;    // how far it pokes proud -> touches skin first
btn_base_h     = 1.6;    // flat glue base

// ---- wire exit (sized for the 18 AWG zip-cord on hand — docs/15 §3) ---------
notch_w        = 6.0;    // wire-notch width: fits 18 AWG zip-cord (2 conductors)
notch_h        = 5.0;    // wire-notch height

// ---- derived ----------------------------------------------------------------
cup_id   = spk_dia + spk_fit_gap;          // inner bore that holds the speaker
cup_od   = cup_id + 2*wall;                // outer diameter
cup_h    = back + spk_depth + 1.0;         // closed back + speaker + a hair
$fn      = 96;

// ----------------------------------------------------------------------------
module cup() {
  difference() {
    union() {
      // outer body
      cylinder(d = cup_od, h = cup_h);
      // two mounting ears (zip-tie / sew to garment)
      for (a = [0, 180])
        rotate([0,0,a])
          translate([cup_od/2 + 4, 0, 0])
            cylinder(d = 10, h = 3);
    }

    // speaker bore (open at the SKIN side = top), leaving a closed rigid back
    translate([0, 0, back])
      cylinder(d = cup_id, h = cup_h);   // through to top

    // inward retaining lip at the open rim (speaker rim rests on it)
    translate([0, 0, cup_h - rim_lip])
      cylinder(d1 = cup_id, d2 = cup_id - 2*1.5, h = rim_lip + 0.1);

    // wire exit notch through the wall (cable leaves sideways, low) — 18 AWG
    translate([0, 0, back + 1.5])
      rotate([0,0,90])
        translate([cup_od/2 - wall - 0.5, -notch_w/2, 0])
          cube([wall + 1.5, notch_w, notch_h]);

    // strain-relief zip-tie hole right at the wire exit (anchor cable to cup)
    rotate([0,0,90])
      translate([cup_od/2 - 1.0, 0, back + 3.5])
        rotate([0,90,0]) cylinder(d = 2.6, h = wall + 2, center = true);

    // ear holes
    for (a = [0, 180])
      rotate([0,0,a])
        translate([cup_od/2 + 4, 0, -0.5])
          cylinder(d = 3.4, h = 5);

    // small back vent so the trapped air behind the cone can breathe a touch
    // (a fully sealed cone stalls; one small port keeps excursion alive)
    translate([0, 0, -0.5]) cylinder(d = 4, h = back + 1);
  }
}

// ----------------------------------------------------------------------------
// Contact button: flat base glued to the dust cap, dome pokes toward skin.
module button() {
  union() {
    cylinder(d = btn_dia, h = btn_base_h);
    translate([0, 0, btn_base_h])
      // shallow dome
      scale([1, 1, btn_dome_h / (btn_dia/2)])
        sphere(d = btn_dia);
  }
}

// ----------------------------------------------------------------------------
if (part == "cup")    cup();
else if (part == "button") button();
else {                       // both -> for a quick visual / single small plate
  cup();
  translate([cup_od + 8, 0, 0]) button();
}

// ASSEMBLY (per actuator):
//  1. Solder the 2-wire pair to the speaker tabs; thread the wire out the notch.
//  2. Zip-tie the wire to the cup through the strain-relief hole (cable load now
//     on the cup, never on the solder tab — see docs/06-safety.md).
//  3. Drop the speaker into the cup, cone facing OUT (toward the open rim).
//  4. Super-glue / VHB a contact BUTTON to the centre of the dust cap so it
//     stands ~2-3 mm proud of the cup rim -> it hits skin first.
//  5. Zip-tie or sew the ears to the compression garment, button to the body,
//     snug. Foam-isolate the cup back from the garment to limit crosstalk.
