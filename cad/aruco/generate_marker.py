#!/usr/bin/env python3
"""
Generate the Tactus fretboard fiducial: ArUco 4x4_50, id 0 (truth.md §3.8, cad/README).

Outputs (next to this file):
  - marker_4x4_50_id0.png   the marker bitmap (black border included, no quiet zone baked in)
  - print_marker.html       a print sheet that renders the BLACK square at an EXACT physical
                            size in mm (Chrome: Print -> "Actual size"/100% -> a real ruler confirms it)

The marker carries NO metric meaning for our pipeline: the fretboard homography maps the
marker's 4 detected corners onto fretboard-relative coordinates, so you do NOT need to know
its size precisely. We still print at a known size + log it so align-mode's distance hint
(solvePnP) and any optional metric checks have a scale to lean on.
"""
import base64
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PNG = os.path.join(HERE, "marker_4x4_50_id0.png")
HTML = os.path.join(HERE, "print_marker.html")

MODULE_PX = 200  # render resolution; 6 modules across (1 black border + 4 data + 1 black border)
MARKER_ID = 0
PRIMARY_MM = 30  # black-square edge length when printed at 100%
ALT_MM = (24, 36)  # alternates if 30 mm doesn't fit your nut/headstock area
QUIET_MM = 6  # white quiet zone around the black square (>= 1 module; detector needs it)


def make_png() -> None:
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side = MODULE_PX * 6  # 4x4 marker => 6x6 module grid including the black border ring
    img = cv2.aruco.generateImageMarker(dic, MARKER_ID, side)
    cv2.imwrite(PNG, img)
    print(f"wrote {os.path.relpath(PNG)}  ({img.shape[1]}x{img.shape[0]} px)")


def make_html() -> None:
    with open(PNG, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    src = f"data:image/png;base64,{b64}"

    def tile(mm: int, primary: bool) -> str:
        ring = "3px solid #111" if primary else "1px dashed #888"
        tag = "PRIMARY — use this one" if primary else "alternate"
        return f"""
      <div class="tile">
        <div class="frame" style="padding:{QUIET_MM}mm;border:{ring}">
          <img src="{src}" style="width:{mm}mm;height:{mm}mm;image-rendering:pixelated;display:block">
        </div>
        <div class="cap">{mm} mm marker · {tag}</div>
      </div>"""

    tiles = tile(PRIMARY_MM, True) + "".join(tile(m, False) for m in ALT_MM)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Tactus ArUco — 4x4_50 id 0</title>
<style>
  @page {{ size: letter; margin: 14mm; }}
  body {{ font: 13px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; color:#111; }}
  h1 {{ font-size:18px; margin:0 0 2px; }}
  .sub {{ color:#555; margin:0 0 14px; }}
  .sheet {{ display:flex; gap:22mm; align-items:flex-start; flex-wrap:wrap; }}
  .tile {{ text-align:center; }}
  .frame {{ background:#fff; display:inline-block; }}
  .cap {{ margin-top:6px; font-size:11px; color:#333; }}
  ol {{ max-width:660px; }} li {{ margin:3px 0; }}
  .warn {{ color:#a00; font-weight:600; }}
  .ruler {{ margin-top:10px; }}
  @media print {{ .noprint {{ display:none; }} }}
</style></head><body>
  <h1>Tactus fretboard fiducial — ArUco 4&times;4_50, id 0</h1>
  <p class="sub">Print this page in <b>Chrome</b> &rarr; Print &rarr; set <b>Scale: 100% / "Actual size"</b>
     (NOT "Fit to page"). Verify with a ruler: the bold square should measure exactly <b>{PRIMARY_MM} mm</b>.</p>
  <div class="sheet">{tiles}</div>
  <ol>
    <li>Cut out the <b>{PRIMARY_MM} mm</b> marker keeping the full <b>white border</b> (the quiet zone) — the detector needs it.</li>
    <li><b class="warn">Keep it perfectly FLAT — never fold it.</b> A crease destroys the corner geometry the homography rides on. Glue it to a thin stiff card if the paper curls.</li>
    <li>Tape it <b>flat on the neck just past fret 6&ndash;7</b> (toward the body), or right <b>at the nut</b> &mdash; either way <b>coplanar with the fretboard</b>, beside the 1&ndash;6 fingering zone so your hand never covers it. <b>Avoid the angled headstock</b>: it tilts ~15&deg; off the fretboard plane, so its homography isn't the fretboard's and your distance feature <i>d</i> gets noisy.</li>
    <li>Frame the camera <b>tight: nut &rarr; fret 7</b>, marker in view. Jot the printed size ({PRIMARY_MM} mm) in the session notes.</li>
  </ol>
  <p class="noprint" style="color:#777">marker_4x4_50_id0.png is the raw bitmap if you want to place it yourself.</p>
</body></html>"""
    with open(HTML, "w") as f:
        f.write(html)
    print(f"wrote {os.path.relpath(HTML)}")


if __name__ == "__main__":
    make_png()
    make_html()
    print("open the HTML in Chrome and print at 100% (Actual size).")
