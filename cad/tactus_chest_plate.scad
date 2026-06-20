// ============================================================================
// TACTUS — Chest plate (body-contoured, holds the 6 "string" drivers)
// ----------------------------------------------------------------------------
// A cylindrical-SECTION shell (the torso ~= a cylinder) carrying a vertical
// column of 6 self-adjusting sockets (high-E top -> low-E bottom, docs/07),
// with strap slots to lash onto the laser-tag vest. Prints as 2 tiles so it
// fits the bed and flexes to the body.
//
// ⚠️ PLACEHOLDERS until measured: `drv_*` (the SK473 KHD driver) and
//    `torso_r` (chest curvature) + `col_pitch` (per the wearer). Update + re-render.
//
// `use`s the socket module. RENDER: set `part`, F6 -> export STL.
// ============================================================================
use <tactus_socket.scad>

part       = "full";   // "full" | "top" | "bottom"  (top/bottom = the 2 print tiles)

// ---- driver (measure!) ----------------------------------------------------
drv_dia    = 58;   drv_depth = 24;   drv_mag_dia = 30;   // ⚠️ PLACEHOLDER estimate — MEASURE the driver, update, re-render

// ---- body fit (measure the wearer + vest) ---------------------------------
torso_r    = 150;  // chest radius of curvature (PLACEHOLDER ~150mm)
col_pitch  = 46;   // vertical spacing between string sockets (>= skin 2-pt res)
n_string   = 6;    // 6 strings
plate_t    = 3.0;  // shell thickness
arc_half   = 58;   // half-width of the panel (chord), mm

$fn = 120; eps = 0.05;
sock_od = socket_outer_d(drv_dia);
col_h   = (n_string - 1) * col_pitch;          // column span
z0      = -col_h/2;                            // first socket z
split_z = 0;                                    // tile boundary (between #3 and #4)

// curved shell panel between z_lo..z_hi (a front section of the torso cylinder)
module shell(z_lo, z_hi) {
  intersection() {
    difference() {
      cylinder(r = torso_r + plate_t, h = z_hi - z_lo);
      translate([0,0,-1]) cylinder(r = torso_r, h = z_hi - z_lo + 2);
    }
    translate([0, -arc_half, 0]) cube([torso_r + plate_t + 1, 2*arc_half, z_hi - z_lo]);
  }
}

// place one socket on the front inner face at height z, skin-side toward the body
module socket_at(z) {
  translate([torso_r, 0, z]) rotate([0, 90, 0]) driver_socket(drv_dia, drv_depth, drv_mag_dia);
}
// through-hole in the panel so the cone/button reaches the skin
module bore_at(z) {
  translate([torso_r - 2, 0, z]) rotate([0, 90, 0])
    cylinder(d = drv_dia - 8, h = plate_t + 4);
}

// strap slots along the top + bottom edges (lash to the laser-tag vest)
module strap_slots(z_lo, z_hi) {
  for (z = [z_lo + 8, z_hi - 8], y = [-arc_half*0.6, 0, arc_half*0.6])
    translate([torso_r + plate_t/2, y, z]) rotate([0,90,0])
      cube([plate_t + 4, 10, 4], center = true);
}

module tile(z_lo, z_hi) {
  // which sockets fall in this tile's z-range
  difference() {
    union() {
      translate([0,0,z_lo]) shell(z_lo, z_hi);
      for (i = [0:n_string-1]) let(z = z0 + i*col_pitch)
        if (z >= z_lo && z < z_hi) socket_at(z);
    }
    for (i = [0:n_string-1]) let(z = z0 + i*col_pitch)
      if (z >= z_lo && z < z_hi) bore_at(z);
    strap_slots(z_lo, z_hi);
  }
}

// NOTE: shell() is modeled from z=0; tile() translates it. Re-place sockets in
// absolute z by building the shell at absolute coords instead:
module tile_abs(z_lo, z_hi) {
  difference() {
    union() {
      intersection() {
        difference() {
          translate([0,0,z_lo]) cylinder(r = torso_r + plate_t, h = z_hi - z_lo);
          translate([0,0,z_lo-1]) cylinder(r = torso_r, h = z_hi - z_lo + 2);
        }
        translate([0, -arc_half, z_lo]) cube([torso_r + plate_t + 1, 2*arc_half, z_hi - z_lo]);
      }
      for (i = [0:n_string-1]) let(z = z0 + i*col_pitch)
        if (z >= z_lo - eps && z < z_hi + eps) socket_at(z);
    }
    for (i = [0:n_string-1]) let(z = z0 + i*col_pitch)
      if (z >= z_lo - eps && z < z_hi + eps) bore_at(z);
    strap_slots(z_lo, z_hi);
  }
}

if (part == "full")        tile_abs(z0 - 20, z0 + col_h + 20);
else if (part == "top")    tile_abs(split_z, z0 + col_h + 20);
else if (part == "bottom") tile_abs(z0 - 20, split_z);

// ASSEMBLY: 2 tiles bolt/zip together at the split, lash to the laser-tag vest
// chest panel via the strap slots; sockets hold the 6 string-drivers (foam-backed,
// docs/15 §4); buttons protrude toward the body. Tune preload per docs/18.
