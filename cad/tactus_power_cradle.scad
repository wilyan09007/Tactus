// ============================================================================
// TACTUS — Power cradle (Anker 737 PowerCore 24K  /  10-port USB hub)
// ----------------------------------------------------------------------------
// Separate from the main box on purpose (docs/15-build-refinements.md §8):
//   - a 140 W Li-ion bank dumping ~33 W should VENT and stay REMOVABLE,
//   - you swap Anker (Mode B cordless) <-> 10-port hub (Mode A wall) per demo,
//   - you must SEE the Anker's screen + reach its ports/button.
// The cradle is an open-faced, vented sled. The hub (smaller, ~150x50x25) sits
// in the same pocket with a foam shim. Velcro strap slots hold either in place
// and let you belt it next to the brain-pack.
//
// Anker 737 (A1289): 155.7 x 54.6 x 49.5 mm, 630 g  (measured-spec).
// Print one. PETG, no supports.  RENDER: F6 -> Export STL.
// ============================================================================

// ---- pocket = device + clearance -------------------------------------------
dev_l   = 155.7;   // along X
dev_w   = 54.6;    // along Y
dev_h   = 49.5;    // along Z
clr     = 1.5;     // all-round clearance

wall    = 2.6;
floor_t = 2.6;
cradle_wall_h = 26;     // how far the side walls come up (leaves screen/ports open)

pl = dev_l + 2*clr;     // pocket length
pw = dev_w + 2*clr;     // pocket width
ol = pl + 2*wall;       // outer length
ow = pw + 2*wall;       // outer width
$fn = 64;
eps = 0.05;

module cradle() {
  difference() {
    // outer body up to wall height + a full floor
    cube([ol, ow, floor_t + cradle_wall_h]);

    // device pocket (open top — device drops in, screen/ports face up & out)
    translate([wall, wall, floor_t])
      cube([pl, pw, cradle_wall_h + eps]);

    // OPEN the +X end fully (Anker ports + button live on one end) ---------
    translate([ol - wall - eps, wall + 4, floor_t + 6])
      cube([wall + 2*eps, pw - 8, cradle_wall_h]);

    // big floor vents (heat off the bank) --------------------------------
    for (ix = [0:5], iy = [0:1])
      translate([18 + ix*22, ow*0.30 + iy*ow*0.30, -eps])
        cube([12, 12, floor_t + 2*eps]);

    // side-wall vents -----------------------------------------------------
    for (ix = [0:5]) {
      translate([16 + ix*22, -eps, floor_t + 6])
        cube([10, wall + 2*eps, cradle_wall_h - 10]);
      translate([16 + ix*22, ow - wall - eps, floor_t + 6])
        cube([10, wall + 2*eps, cradle_wall_h - 10]);
    }

    // velcro / strap slots through the floor (two pairs) ------------------
    for (x = [ol*0.22, ol*0.40, ol*0.58, ol*0.76])
      translate([x, ow*0.5 - 25, -eps])
        cube([10, 50, floor_t + 2*eps]);

    // a finger scoop on the -X end so you can pop the bank out ------------
    translate([-eps, ow/2, floor_t + cradle_wall_h])
      rotate([0,90,0]) cylinder(d = 34, h = wall + 2*eps);
  }

  // two retention lips at the top edges so the device can't slide out ------
  for (y = [wall + 2, ow - wall - 5])
    translate([wall + 10, y, floor_t + cradle_wall_h - eps])
      cube([pl - 20, 3, 3]);
}

cradle();

// ASSEMBLY:
//  - Drop the Anker in screen-up, USB end at the open +X end. Strap it with a
//    velcro loop through the floor slots. Belt the cradle beside the brain-pack.
//  - Mode A: swap in the 10-port hub (foam-shim the extra clearance); the hub's
//    12 V barrel adapter stays OUTSIDE the cradle and plugs to the wall.
//  - Keep both vent faces clear of fabric so the bank breathes.
