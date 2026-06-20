// ============================================================================
// TACTUS — Main electronics enclosure ("brain-pack")
// ----------------------------------------------------------------------------
// Holds everything that is NOT the vest or the subwoofers (per the build):
//   - 3x Vantec NBA-200U USB 7.1 cards     (100 x 58 x 26 mm each)
//   - 7x gutted SK473 / PAM8403 amp bricks (~55 x 35 x 25 mm each)
//   - 1x Raspberry Pi 5 (optional)         (85 x 56 mm, mounted on bosses)
//   (the Anker 737 / 10-port hub live in the SEPARATE tactus_power_cradle.scad
//    so the battery vents + stays removable for Mode A / Mode B swaps)
//
// Design goals (see docs/15-build-refinements.md):
//   - DIMENSION-TOLERANT: boards mount with VHB foam + zip-ties through a floor
//     slot grid, not press-fit pockets (our part dims are partly estimated).
//   - STRAIN RELIEF FIRST: a comb wall takes the 14-wire umbilical load off the
//     amp output pads — the #1 documented failure mode (docs/06-safety.md).
//   - VENTED: amps run warm; floor + wall + lid vents, Pi on standoffs.
//   - WEARABLE or table: belt-strap slots on the base; rubber-foot friendly.
//   - Fits the FlashForge Adventurer 5M 220x220x220 bed with margin.
//
// RENDER: set `part` then F6 -> Export STL. Print base + lid separately.
// ============================================================================

part = "base";          // "base" | "lid"

// ---- internal volume (edit to taste; keep outer < ~210 for a safe 220 bed) --
inner_x   = 195;        // width  (X)
inner_y   = 185;        // depth  (Y) — enlarged: holds 3 Vantec + 6 amps with ~25% floor to spare
inner_z   = 62;         // height (Z) — 25 mm amp + headroom; amps can also stack 2-high here
corner_r  = 7;          // rounded vertical corners (product look, not a raw cube)
wall      = 2.4;        // wall thickness (3 perimeters @ 0.4 nozzle, PETG)
floor_t   = 2.4;        // floor thickness
lid_t     = 2.4;        // lid plate thickness

outer_x   = inner_x + 2*wall;     // ~199.8
outer_y   = inner_y + 2*wall;     // ~154.8
base_h    = floor_t + inner_z;    // ~60.4

n_wires   = 16;         // umbilical comb slots (14 used + 2 spare)
include_pi = false;     // Pi CUT from scope (laptop-only) — set true to restore the wearable-mode mount
$fn       = 64;
eps       = 0.05;

// ---- lid screw boss positions (shared by base + lid so holes align) ---------
boss_inset = wall + 8;
boss_pts = [
  [boss_inset,           boss_inset],
  [outer_x - boss_inset, boss_inset],
  [boss_inset,           outer_y - boss_inset],
  [outer_x - boss_inset, outer_y - boss_inset]
];

// ---- Raspberry Pi 5 mount (58 x 49 hole pattern), front-left corner ---------
pi_x0 = 22; pi_y0 = 20;            // first hole center
pi_holes = [
  [pi_x0,      pi_y0],
  [pi_x0 + 58, pi_y0],
  [pi_x0,      pi_y0 + 49],
  [pi_x0 + 58, pi_y0 + 49]
];

// rounded-rectangle prism + engraved logo -> reads as a designed unit, not a cube
module rprism(x, y, h, r) {
  linear_extrude(h) translate([r, r]) offset(r = r) square([x - 2*r, y - 2*r]);
}
module logo_cut() {
  translate([outer_x/2, outer_y/2, lid_t - 0.8])
    linear_extrude(1.2) text("TACTUS", size = 22, halign = "center", valign = "center");
}

// ============================================================================
module base() {
  union() {
    difference() {
      union() {
        rprism(outer_x, outer_y, base_h, corner_r);   // rounded outer shell
        comb_ledge();                          // external strain-relief comb (+Y wall)
      }

      // --- main cavity (open top) ---
      translate([wall, wall, floor_t])
        cube([inner_x, inner_y, inner_z + eps]);

      // --- umbilical wire pass-throughs + comb slots (+Y wall) ---
      wire_passthroughs();

      // --- laptop-side glands: 3x USB + 1x power feed (-Y wall) ---
      gland_holes();

      // --- vents ---
      floor_vents();
      shortwall_vents();

      // --- belt / strap slots in the floor near the -Y edge ---
      strap_slots();

      // --- zip-tie floor grid (mount any board anywhere) ---
      ziptie_floor_grid();
    }

    // --- adds (after the difference so the cavity doesn't eat them) ---
    if (include_pi) pi_standoffs();   // Pi cut from scope by default — frees floor for amps/Vantecs
    lid_bosses();
    cable_anchors();          // internal zip-tie posts at the wire wall
  }
}

// ----------------------------------------------------------------------------
module comb_ledge() {
  // external ledge on the +Y wall, holes/slots cut later in wire_passthroughs()
  translate([0, outer_y, floor_t + 2])
    cube([outer_x, 7, 14]);
}

module wire_passthroughs() {
  // a row of Ø7 holes through the +Y wall AND the comb ledge, open-topped slots
  spacing = (outer_x - 30) / (n_wires - 1);
  for (i = [0 : n_wires - 1]) {
    x = 15 + i * spacing;
    // through-hole (wall + comb)
    translate([x, outer_y + 8, floor_t + 9])
      rotate([90, 0, 0])
        cylinder(d = 7, h = wall + 10);
    // open-top slot so a wire can be pressed in from above (no threading)
    translate([x - 2, outer_y - eps, floor_t + 9])
      cube([4, wall + 8, 20]);
  }
}

module gland_holes() {
  // -Y wall: 3 USB cable glands + 1 power-feed gland
  usb_d = 13; pwr_d = 10;
  xs = [outer_x*0.30, outer_x*0.45, outer_x*0.60];
  for (x = xs)
    translate([x, -eps, floor_t + 16])
      rotate([-90, 0, 0]) cylinder(d = usb_d, h = wall + 2*eps);
  translate([outer_x*0.80, -eps, floor_t + 14])
    rotate([-90, 0, 0]) cylinder(d = pwr_d, h = wall + 2*eps);
}

module floor_vents() {
  // grid of slots through the floor (airflow under the boards + Pi)
  for (ix = [0 : 4], iy = [0 : 3]) {
    x = 35 + ix * 32;
    y = 28 + iy * 30;
    translate([x - 2.5, y - 14, -eps])
      cube([5, 28, floor_t + 2*eps]);
  }
}

module shortwall_vents() {
  // vertical slots in the two short walls (x = 0 and x = outer_x)
  for (iy = [0 : 4]) {
    y = 25 + iy * 26;
    // -X wall
    translate([-eps, y - 2, floor_t + 8])
      cube([wall + 2*eps, 4, inner_z - 16]);
    // +X wall
    translate([outer_x - wall - eps, y - 2, floor_t + 8])
      cube([wall + 2*eps, 4, inner_z - 16]);
  }
}

module strap_slots() {
  // two floor slots near the -Y edge -> thread a belt/strap to wear the pack
  for (x = [outer_x*0.30, outer_x*0.62])
    translate([x, 8, -eps])
      cube([46, 9, floor_t + 2*eps]);
}

module ziptie_floor_grid() {
  // pairs of thin slots (a zip tie loops up one, over the board, down the other)
  for (ix = [0 : 5], iy = [0 : 4]) {
    x = 28 + ix * 28;
    y = 24 + iy * 28;
    if (x < inner_x - 6 && y < inner_y - 6) {
      translate([x,     y, -eps]) cube([2.6, 9, floor_t + 2*eps]);
      translate([x + 6, y, -eps]) cube([2.6, 9, floor_t + 2*eps]);
    }
  }
}

module pi_standoffs() {
  for (p = pi_holes)
    translate([p[0], p[1], floor_t])
      difference() {
        cylinder(d = 7, h = 6);
        translate([0,0,-eps]) cylinder(d = 2.3, h = 6 + 2*eps);  // M2.5 self-tap
      }
}

module lid_bosses() {
  for (p = boss_pts)
    translate([p[0], p[1], floor_t])
      difference() {
        cylinder(d = 9, h = inner_z);
        translate([0,0, inner_z - 14]) cylinder(d = 2.6, h = 14 + eps); // M3 self-tap
      }
}

module cable_anchors() {
  // two posts just inside the wire wall to lash the internal bundle to
  for (x = [outer_x*0.25, outer_x*0.75])
    translate([x, outer_y - wall - 6, floor_t])
      difference() {
        cube([8, 4, 12], center = false);
        translate([4, -eps, 6]) rotate([-90,0,0]) cylinder(d = 3, h = 4 + 2*eps);
      }
}

// ============================================================================
module lid() {
  difference() {
    union() {
      rprism(outer_x, outer_y, lid_t, corner_r);       // rounded top plate
      // nesting lip (locates the lid into the cavity) — loose ~0.8mm/side fit
      translate([wall + 0.8, wall + 0.8, -5])
        difference() {
          cube([inner_x - 1.6, inner_y - 1.6, 5 + eps]);
          translate([2, 2, -eps])
            cube([inner_x - 5.6, inner_y - 5.6, 5 + 2*eps]);
        }
    }
    logo_cut();   // engraved TACTUS on the lid
    // corner screw holes (clearance + countersink) — align to boss_pts
    for (p = boss_pts) {
      translate([p[0], p[1], -eps]) cylinder(d = 3.4, h = lid_t + 2*eps);
      translate([p[0], p[1], lid_t - 1.6]) cylinder(d1 = 3.4, d2 = 6.4, h = 1.6 + eps);
    }
    // lid vents (slot array)
    for (ix = [0 : 5], iy = [0 : 4]) {
      x = 30 + ix * 28; y = 26 + iy * 28;
      if (x < outer_x - 24 && y < outer_y - 24)
        translate([x - 2.5, y - 16, -eps]) cube([5, 32, lid_t + 2*eps]);
    }
  }
}

// ============================================================================
if (part == "base") base();
else if (part == "lid") lid();

// ---------------------------------------------------------------------------
// SUGGESTED INTERNAL LAYOUT (lash with zip-ties to the floor grid):
//   - Pi CUT (laptop-only build) -> the front-left area is now free; spread the
//     3 Vantecs + 7 amps in one layer (no 2-high stacking needed).
//   - 7 amp bricks in 2 rows along the back half (closest to the +Y comb wall),
//     output wires going straight out the comb -> shortest unrelieved span.
//     If they don't all fit one layer, stack 2-high on foam (they're ~15 g).
//   - 3 Vantecs along one short wall; their 3.5 mm plugs reach the amps; their
//     USB cables exit the -Y glands to the laptop.
//   - All 14 actuator wires leave through the comb; lash the bundle to the two
//     internal cable_anchors, then again to the comb -> both-end strain relief.
// ---------------------------------------------------------------------------
