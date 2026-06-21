#!/usr/bin/env python3
"""Render an annotated demo video of the guitar-neck detector tracking a REAL
training capture over time.

We import fretboard_server (import-safe: its server only starts under
``if __name__ == "__main__"``), load the YOLO-World + SAM2 models ONCE via
``load_model()``, then walk a short segment of a training .webm. For each sampled
frame we call ``detect_quad(frame, smooth=True)`` — the same live-stream path the
WebSocket server uses (EMA smoothing across frames, SAM2 only every 4th frame,
hold-on-miss) — and draw the detected neck grid the way the product registration
overlay does:

  * the detected quad as a bright rotated rectangle,
  * 6 string lines interpolated across the quad's v-axis (top edge -> bottom edge),
  * 7 fret lines (frets 0..7) placed by the 12-TET law along the quad's u-axis
    (nut -> fret7): u_f = (1 - 2^(-f/12)) / (1 - 2^(-7/12)),
  * a small HUD: confidence + frame index + "tracking" / "searching".

On a miss (detect returns None) we draw "searching" and re-draw the last good quad
faintly so the overlay doesn't flicker to nothing.

The annotated frames are written to an mp4 at the sampled fps (cv2.VideoWriter with
mp4v/avc1; if that fails we fall back to PNG frames + ffmpeg). Output is downscaled
to <= ~1280px wide so the file stays small.

Usage:
    .venv-yolo/bin/python annotate_video.py \
        [--video PATH] [--start 5] [--dur 12] [--fps 9] [--max-width 1280] \
        [--out realtest/training_demo.mp4]
"""
import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np

# Import the detector. This module is import-safe: it only starts the WS server
# under ``if __name__ == "__main__"``. We use its load + detect functions only.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fretboard_server as fb  # noqa: E402

DEFAULT_VIDEO = os.path.abspath(os.path.join(
    HERE, "..", "..", "..",
    ".claude", "worktrees", "analysis-pipeline", "data", "raw",
    "2026-06-21-0249", "aditya", "video", "chordstream_aditya_001.webm"))
DEFAULT_OUT = os.path.join(HERE, "realtest", "training_demo.mp4")
FFMPEG = "/opt/homebrew/bin/ffmpeg"

# --- Drawing palette (BGR) ---
QUAD_COLOR = (0, 255, 120)      # bright green: detected neck quad
STRING_COLOR = (255, 235, 90)   # cyan-ish: the 6 strings
FRET_COLOR = (60, 180, 255)     # orange: the 7 fret lines (12-TET)
NUT_COLOR = (80, 80, 255)       # red: nut (fret 0)
FRET7_COLOR = (255, 80, 255)    # magenta: fret 7
HOLD_COLOR = (120, 120, 120)    # faint grey: last quad while searching
HUD_GREEN = (80, 255, 120)
HUD_AMBER = (40, 200, 255)

N_STRINGS = 6
N_FRETS = 7  # draw fret lines 0..7 inclusive (nut..fret7), i.e. 8 lines


def fret_u(f: int) -> float:
    """12-TET position of fret f along the quad's nut->fret7 u-axis (0..1).

    u_f = (1 - 2^(-f/12)) / (1 - 2^(-7/12)) = fret_fraction(f) / fret_fraction(7).
    f=0 -> 0 (nut), f=7 -> 1 (fret 7).
    """
    return fb.fret_fraction(f) / fb.fret_fraction(N_FRETS)


def bilerp(quad_px, u, v):
    """Bilinear point inside the quad. Corner order is
    [0]=alongStart.top, [1]=alongEnd.top, [2]=alongEnd.bottom, [3]=alongStart.bottom.
    u runs nut(0)->fret7(1) along the top/bottom edges; v runs top(0)->bottom(1).
    """
    p0, p1, p2, p3 = (np.asarray(quad_px[i], np.float64) for i in range(4))
    top = p0 + (p1 - p0) * u      # along the top edge
    bot = p3 + (p2 - p3) * u      # along the bottom edge
    return top + (bot - top) * v


def draw_grid(img, quad_px, faint=False):
    """Draw the neck registration grid (quad + 6 strings + 8 fret lines)."""
    pts = np.asarray(quad_px, np.float64)
    if faint:
        # Just the quad outline, dim, so a searching frame still shows last pose.
        cv2.polylines(img, [pts.astype(np.int32).reshape(-1, 1, 2)],
                      isClosed=True, color=HOLD_COLOR, thickness=2,
                      lineType=cv2.LINE_AA)
        return

    # Fret lines 0..7 across the v-axis (top edge -> bottom edge), placed by 12-TET.
    for f in range(N_FRETS + 1):
        u = fret_u(f)
        a = bilerp(pts, u, 0.0)
        b = bilerp(pts, u, 1.0)
        if f == 0:
            col, th = NUT_COLOR, 3
        elif f == N_FRETS:
            col, th = FRET7_COLOR, 3
        else:
            col, th = FRET_COLOR, 2
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), col, th,
                 lineType=cv2.LINE_AA)

    # 6 string lines across the u-axis (nut -> fret7), interpolated on the v-axis.
    for s in range(N_STRINGS):
        v = (s + 0.5) / N_STRINGS
        a = bilerp(pts, 0.0, v)
        b = bilerp(pts, 1.0, v)
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), STRING_COLOR, 1,
                 lineType=cv2.LINE_AA)

    # The quad outline on top so it reads as the registration boundary.
    cv2.polylines(img, [pts.astype(np.int32).reshape(-1, 1, 2)], isClosed=True,
                  color=QUAD_COLOR, thickness=3, lineType=cv2.LINE_AA)
    # Corner dots (nut end emphasized).
    for i, (x, y) in enumerate(quad_px):
        cv2.circle(img, (int(x), int(y)), 5, (40, 40, 230), -1, lineType=cv2.LINE_AA)


def draw_hud(img, frame_idx, conf, tracking, fresh, n_frets_payload):
    """Top-left HUD: state, confidence, frame index."""
    h, w = img.shape[:2]
    state = "tracking" if tracking else "searching"
    col = HUD_GREEN if tracking else HUD_AMBER
    # translucent backdrop
    bar_h = 64
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    dot = (0, 230, 90) if tracking else (40, 200, 255)
    cv2.circle(img, (24, 32), 9, dot, -1, lineType=cv2.LINE_AA)
    txt = f"{state.upper()}   conf={conf:0.2f}   frame {frame_idx}"
    if tracking and not fresh:
        txt += "  (hold)"
    if n_frets_payload:
        txt += f"   frets:{n_frets_payload}"
    cv2.putText(img, txt, (44, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2,
                lineType=cv2.LINE_AA)


def open_writer(out_path, fps, size):
    """Try cv2.VideoWriter with a couple of codecs; return (writer, fourcc) or
    (None, None) if none open (caller then falls back to ffmpeg-from-PNGs)."""
    w, h = size
    for tag in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*tag)
        vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if vw.isOpened():
            print(f"[writer] cv2.VideoWriter opened with fourcc={tag}", flush=True)
            return vw, tag
        vw.release()
    print("[writer] cv2.VideoWriter could not open any codec; will use ffmpeg",
          flush=True)
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--start", type=float, default=5.0, help="segment start (s)")
    ap.add_argument("--dur", type=float, default=12.0, help="segment duration (s)")
    ap.add_argument("--fps", type=float, default=9.0, help="output sample fps")
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"[err] video not found: {args.video}", flush=True)
        return 2
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[init] loading detector models once (YOLO-World + SAM2)...", flush=True)
    fb.load_model()
    fb.reset_state()  # fresh EMA + SAM2-cadence state for this run

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[err] could not open video: {args.video}", flush=True)
        return 2
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {src_w}x{src_h} src_fps={src_fps:.2f} frames={total} "
          f"dur={total / src_fps:.1f}s", flush=True)

    # Output scale: keep <= max_width.
    scale = min(1.0, args.max_width / float(src_w))
    out_w = int(round(src_w * scale))
    out_h = int(round(src_h * scale))
    if out_w % 2:   # even dims keep encoders happy
        out_w -= 1
    if out_h % 2:
        out_h -= 1
    print(f"[video] output {out_w}x{out_h} (scale={scale:.3f}) @ {args.fps}fps",
          flush=True)

    # Frame sampling: step through source frames, emit one every `step` to hit fps.
    step = max(1, int(round(src_fps / args.fps)))
    start_f = int(round(args.start * src_fps))
    end_f = int(round((args.start + args.dur) * src_fps))
    end_f = min(end_f, total) if total > 0 else end_f
    print(f"[plan] segment {args.start:.1f}s..{args.start + args.dur:.1f}s -> "
          f"src frames [{start_f}, {end_f}) every {step} "
          f"(~{(end_f - start_f) // step} sampled frames)", flush=True)

    # Writer (lazy fallback to PNG sequence).
    vw, fourcc = open_writer(args.out, args.fps, (out_w, out_h))
    frames_dir = os.path.join(os.path.dirname(args.out), "_demo_frames")
    use_pngs = vw is None
    if use_pngs:
        os.makedirs(frames_dir, exist_ok=True)
        for old in os.listdir(frames_dir):
            if old.endswith(".png"):
                os.remove(os.path.join(frames_dir, old))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    last_quad_px_scaled = None     # for faint hold while searching
    confs = []
    n_lock = 0
    n_search = 0
    written = 0
    src_idx = start_f
    t_run = time.time()

    while src_idx < end_f:
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[video] read ended at src frame {src_idx}", flush=True)
            break
        if (src_idx - start_f) % step != 0:
            src_idx += 1
            continue

        # --- Detect on the FULL-RES frame (best for the model), then draw on a
        # downscaled copy so the output stays small. ---
        det = None
        try:
            det = fb.detect_quad(frame, smooth=True)
        except Exception as exc:  # never crash the render on a bad frame
            print(f"[detect] frame {src_idx} raised {exc!r}; treating as miss",
                  flush=True)
            det = None

        small = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # detect_quad returns normalized quad; scale to the OUTPUT image size.
        quad_px_scaled = None
        conf = 0.0
        tracking = False
        fresh = False
        method = ""
        n_frets = 0
        if det is not None and det.get("quad") is not None:
            conf = float(det.get("confidence", 0.0))
            method = str(det.get("method", ""))
            n_frets = len(det.get("frets") or [])
            q = np.asarray(det["quad"], np.float64)  # normalized 0..1
            quad_px_scaled = np.column_stack([q[:, 0] * out_w, q[:, 1] * out_h])
            tracking = True
            fresh = not method.startswith("hold")

        if quad_px_scaled is not None:
            draw_grid(small, quad_px_scaled, faint=False)
            last_quad_px_scaled = quad_px_scaled
            confs.append(conf)
            if fresh:
                n_lock += 1
            else:
                n_search += 1  # a "hold" frame = no fresh detection this frame
        else:
            # searching: keep the last pose faintly so it doesn't flicker to nothing
            if last_quad_px_scaled is not None:
                draw_grid(small, last_quad_px_scaled, faint=True)
            n_search += 1

        draw_hud(small, src_idx, conf, tracking, fresh, n_frets)

        if use_pngs:
            cv2.imwrite(os.path.join(frames_dir, f"f{written:05d}.png"), small)
        else:
            vw.write(small)
        written += 1

        if written % 10 == 0:
            elapsed = time.time() - t_run
            print(f"[run] {written} frames | last src={src_idx} conf={conf:.2f} "
                  f"method={method or '-'} | {elapsed:.1f}s "
                  f"({elapsed / max(1, written):.2f}s/frame)", flush=True)
        src_idx += 1

    cap.release()
    if not use_pngs:
        vw.release()

    if written == 0:
        print("[err] no frames written", flush=True)
        return 1

    # If we wrote PNGs, mux them with ffmpeg.
    if use_pngs:
        cmd = [FFMPEG, "-y", "-framerate", str(args.fps),
               "-i", os.path.join(frames_dir, "f%05d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
               args.out]
        print(f"[ffmpeg] {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[ffmpeg] FAILED rc={r.returncode}\n{r.stderr[-2000:]}",
                  flush=True)
            return 1

    # Some cv2 builds write an mp4 that not every player likes; if avc1/mp4v but the
    # file is suspiciously small or unreadable, we still leave it — verification reads
    # it back with ffmpeg, which is robust.
    size = os.path.getsize(args.out)
    total_frames = written
    cr = (min(confs), max(confs)) if confs else (0.0, 0.0)
    lock_frac = n_lock / total_frames if total_frames else 0.0
    print("\n==== SUMMARY ====", flush=True)
    print(f"out: {args.out}", flush=True)
    print(f"frames written: {total_frames} @ {args.fps}fps "
          f"=> ~{total_frames / args.fps:.1f}s", flush=True)
    print(f"size: {size / 1e6:.2f} MB  ({out_w}x{out_h})", flush=True)
    print(f"confidence range (fresh+hold drawn frames): "
          f"{cr[0]:.2f}..{cr[1]:.2f}", flush=True)
    print(f"fresh-lock frames: {n_lock}  search/hold frames: {n_search}  "
          f"lock fraction: {lock_frac * 100:.0f}%", flush=True)
    print(f"render wall time: {time.time() - t_run:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
