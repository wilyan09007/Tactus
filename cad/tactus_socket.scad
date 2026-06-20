// ============================================================================
// TACTUS — Self-adjusting driver SOCKET (the coupling-physics core)
// ----------------------------------------------------------------------------
// Solves the "optimal-pressure / goes-invisible" problem:
//   too LOOSE  -> the cone moves air, you HEAR it, don't feel it (no coupling)
//   too TIGHT  -> the cone/button is clamped, can't move, also invisible
//   sweet spot -> a moderate FLOATING preload.
//
// The trick that gives BOTH "mm-precise" AND "lots of room for error":
//   - the socket GRIPS the driver precisely (rigid, mm-tolerance bore),
//   - the SKIN side couples through a COMPLIANT, self-adjusting element
//     (a foam/TPU pad behind the driver) so the contact button floats into
//     the body and finds the sweet-spot preload across a RANGE of fits.
//   Rigid where it meets the speaker; springy where it meets the body.
//
// `use <tactus_socket.scad>` from the chest/back plates. Standalone = preview.
// ⚠️ Pass the MEASURED SK473 KHD driver dims (defaults are PLACEHOLDERS).
// ============================================================================

$fn = 96;
_eps = 0.05;

// ---- the socket as a parametric MODULE ------------------------------------
// z=0 is the SKIN side (front). Driver loads from the back. Cone faces OUT.
module driver_socket(dia=60, depth=22, mag_dia=26, fit=0.6,
                     foam=6, rim=2.2, wall=2.6) {
  socket_id = dia + fit;                 // precise grip bore
  socket_od = socket_id + 2*wall;
  socket_h  = depth + foam + rim;        // back cavity + driver + front rim
  difference() {
    cylinder(d = socket_od, h = socket_h);

    // precise driver bore (grip), stops at the front rim
    translate([0, 0, rim]) cylinder(d = socket_id, h = socket_h);

    // front rim aperture: the cone/button pokes through to the body
    translate([0, 0, -_eps]) cylinder(d = socket_id - 2*rim, h = rim + 2*_eps);

    // back compliant-pad / magnet-clearance well (foam disc drops in behind)
    translate([0, 0, socket_h - foam - 0.1])
      cylinder(d = mag_dia + 2.0, h = foam + 1.0);

    // wire-exit slot (sized for 18 AWG zip-cord, docs/15 §3)
    rotate([0,0,90]) translate([socket_od/2 - wall - 0.5, -3.0, rim + 3])
      cube([wall + 2, 6.0, 6.0]);

    // 3 grip-relief slots so the bore flexes a hair onto the driver
    for (a = [0, 120, 240]) rotate([0,0,a])
      translate([socket_id/2 - 0.6, -1.0, rim + 4])
        cube([wall + 1.2, 2.0, socket_h - rim - 6]);
  }
}

// outer diameter helper so plates can space sockets correctly
function socket_outer_d(dia=60, fit=0.6, wall=2.6) = dia + fit + 2*wall;

// ---- the CONTACT BUTTON (separate, glued to the dust cap) ------------------
module contact_button(btn_dia=20, proud=2.5) {
  union() {
    cylinder(d = btn_dia, h = 1.6);                          // glue base
    translate([0,0,1.6]) scale([1,1,(proud+2.5)/(btn_dia/2)])
      sphere(d = btn_dia);                                   // skin dome
  }
}

// ---- standalone output (NOT run when `use`d) ------------------------------
// the BARREL already contains the socket, so for printing you only need the
// button from this file. part="button" -> just the contact button (×12+).
part = "preview";   // "preview" | "button"
if (part == "button") contact_button();
else { driver_socket(); translate([socket_outer_d() + 10, 0, 0]) contact_button(); }

// ASSEMBLY (docs/15 §4 + docs/18 tuning):
//  1. Glue a contact_button to the driver's dust cap.
//  2. Drop a foam disc (mag_dia+, `foam` thick) into the back cavity.
//  3. Press the driver in from the back, cone toward the SKIN side; wire out the slot.
//  4. The plate strap pulls the socket to the body; the foam floats the button
//     to the sweet-spot preload. Tune with foam thickness/durometer.
