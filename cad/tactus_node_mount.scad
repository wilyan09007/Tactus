// ============================================================================
// TACTUS — Per-node adjustable speaker mount (clamp + micro-depth + lock)
// ----------------------------------------------------------------------------
// Per Aditya: "all you need is a way to clamp the speaker modules, micro-adjust
// their depth, keep them in place, and attach the fixture to the vest."
//
// Mechanism = a SPLIT-CLAMP COLLAR + a TELESCOPING socket-barrel (GoPro / tube-
// clamp style):
//   - the BARREL holds the driver and slides in the collar -> CONTINUOUS depth
//     adjust = the optimal-preload tuning (too shallow = invisible, too deep =
//     clamped; the slide finds the sweet spot, docs/18),
//   - one CLAMP SCREW squeezes the split collar -> locks at any depth, stays put,
//   - the BASE FOOT zip-ties / velcros onto the (laser-tag) vest.
// Fully printed except 1× M3 screw + nut (or print the thumb-screw variant).
//
// `use`s the socket module. Print: 1 base + 1 barrel per node (×12).
// ⚠️ `drv_*` are PLACEHOLDERS until the SK473 KHD driver is measured.
// ============================================================================
use <tactus_socket.scad>

part      = "both";   // "base" | "barrel" | "both"

// driver (measure!) + derived barrel size
drv_dia   = 58;  drv_depth = 24;  drv_mag_dia = 30;     // ⚠️ PLACEHOLDER estimate — MEASURE the SK473 driver, update, re-render
sock_od   = socket_outer_d(drv_dia);
bar_od    = sock_od + 5;          // sleeve around the socket
bore_id   = bar_od + 0.8;         // slide clearance (telescope fit) — loose, the clamp takes up slack

// collar / clamp
collar_wall = 4;
collar_od   = bore_id + 2*collar_wall;
collar_h    = 26;                 // travel + grip
m3          = 3.4;                // M3 clearance
$fn = 96; eps = 0.05;

// ---- BARREL: a sleeve carrying the driver socket, slides in the collar -----
module barrel() {
  bar_len = drv_depth + 14;
  difference() {
    union() {
      cylinder(d = bar_od, h = bar_len);                 // sleeve
      // finned grip ring at the outer (non-body) end for push/pull + feel
      translate([0,0,bar_len-5])
        for (a=[0:30:330]) rotate([0,0,a])
          translate([bar_od/2,0,0]) cylinder(d=2.5,h=5,$fn=12);
    }
    // hollow it and drop the socket in at the BODY end (z=0 = skin side)
    translate([0,0,3]) cylinder(d = sock_od + 0.6, h = bar_len);
  }
  // the actual gripping socket at the body end
  translate([0,0,3]) driver_socket(drv_dia, drv_depth, drv_mag_dia);
}

// ---- BASE: split-clamp collar + vest-mount foot ----------------------------
module base() {
  difference() {
    union() {
      cylinder(d = collar_od, h = collar_h);             // collar
      // clamp ears straddling the slit (+X side)
      translate([collar_od/2-2, -9, 0]) cube([10, 18, collar_h]);
      // vest-mount foot (flange) on the -X side
      translate([-collar_od/2-18, -16, 0]) cube([20, 32, 5]);
    }
    translate([0,0,-1]) cylinder(d = bore_id, h = collar_h+2);   // bore
    // the SLIT (+X), bore -> outside, so the collar can squeeze
    translate([collar_od/2-collar_wall-1, -1.6, -1])
      cube([collar_wall+14, 3.2, collar_h+2]);
    // clamp screw across the ears (Y), + a nut trap on one side
    translate([collar_od/2+3, -12, collar_h/2]) rotate([-90,0,0])
      cylinder(d = m3, h = 24);
    translate([collar_od/2+3, 6.5, collar_h/2]) rotate([-90,0,0])
      cylinder(d = 6.2, h = 3, $fn=6);                   // M3 nut trap
    // vest attachment: zip-tie slots + a velcro pocket in the foot
    for (y=[-10,10])
      translate([-collar_od/2-12, y-2.5, -1]) cube([8, 5, 7]);
  }
}

// ---- output ----------------------------------------------------------------
if (part == "base")        base();
else if (part == "barrel") barrel();
else { base(); translate([collar_od + 30, 0, 0]) barrel(); }

// ASSEMBLY (per node, ×12):
//  1. Foam disc behind the driver (compliant preload), driver into the socket,
//     contact button on the cone (docs/15 §4).
//  2. Barrel slides into the base collar; push to the depth where it feels
//     strongest on the body (the sweet spot, docs/18) -> tighten the M3 -> locked.
//  3. Zip-tie / velcro the base foot to the vest. Wire exits the socket slot.
//  4. Re-tune depth per wearer in seconds by loosening the one screw.
