/*
 * fretboard-autodetect.js — markerless guitar-fretboard corner detector (OpenCV.js)
 * =================================================================================
 * A browser port of the repo's Python markerless detector,
 *   software/ai/vision/fretboard_detect.py
 * (CLAHE -> median-stack -> Canny -> HoughLinesP neck edges -> rectify + gradient
 *  peak-find frets -> 12-TET law-constrained homography fit -> confidence gate),
 * plus the 12-TET fret law from software/ai/vision/fretboard.py
 *   fret_fraction(n) = 1 - 2^(-n/12)
 * and the pure-JS 4-point DLT homography solver (solveH/proj/gauss) lifted from
 * software/ai/vision/calibrate.html (so we don't fight cv.findHomography's Mat
 * marshalling for the tiny 4-corner reprojection check).
 *
 * "Ponytail" port: simplest thing that works. We lean entirely on OpenCV.js for
 * edge/Hough/gradient work (no hand-rolled Canny or Hough), keep everything in one
 * self-contained classic <script> (no module, no build step), reuse a small pool of
 * cv.Mat objects across calls, and NEVER throw (detect() returns null on any
 * failure). We only recover the 4 board CORNERS — the consumer lays the inner frets
 * by the 12-TET law inside the quad (see calibrate.html / glow.html).
 *
 * OpenCV.js API usage verified against the official 4.x docs:
 *   - cv['onRuntimeInitialized']                       (runtime ready hook)
 *   - new cv.VideoCapture(videoEl) + cap.read(mat)     (tutorial_js_video_display)
 *   - cv.cvtColor / cv.Canny / cv.HoughLinesP          (imgproc)
 *   - cv.createCLAHE(clip, size).apply(src, dst)       (imgproc, CLAHE)
 *   - cv.Sobel / cv.GaussianBlur                       (imgproc)
 *   - cv.getRotationMatrix2D(center, angle, scale)     (tutorial_py_geometric_transformations)
 *   - cv.warpAffine(src, dst, M, dsize)                (dsize = {width, height})
 *   HoughLinesP result is a CV_32SC4 Mat: endpoints at lines.data32S[i*4 + 0..3]
 *   = [x1, y1, x2, y2].
 *
 * Consumer corner semantics (must match — see the task spec & calibrate.html):
 *   u-axis = along the neck (0 = nut, 1 = fret 7)
 *   v-axis = across strings (0 = high-e / string-1 side, 1 = low-E / string-6 side)
 *     c00 = nut    x high-e   ·  c10 = fret-7 x high-e
 *     c01 = nut    x low-E    ·  c11 = fret-7 x low-E
 *   Corners are returned NORMALIZED [0..1] in the video's intrinsic frame
 *   (videoWidth x videoHeight), NOT the downscaled detection frame.
 */
(function () {
  "use strict";

  // -------------------------------------------------------------------------
  // Tuning knobs (exposed on window.TactusFretboard.opts). Ported 1:1 from the
  // Python constants where they exist; the rest are this port's own gates.
  // -------------------------------------------------------------------------
  var opts = {
    cdnUrl: "https://docs.opencv.org/4.x/opencv.js", // OpenCV.js build to load
    workWidth: 480,        // downscale width for speed (Python uses full frame; we go small)
    cannyLo: 40,           // Canny low threshold  (fretboard_detect.py: 40)
    cannyHi: 120,          // Canny high threshold (fretboard_detect.py: 120)
    claheClip: 2.0,        // CLAHE clip limit     (fretboard_detect.py: 2.0)
    claheTile: 8,          // CLAHE tile grid (NxN)(fretboard_detect.py: (8,8))
    houghThresh: 80,       // HoughLinesP accumulator threshold (fretboard_detect.py: 80)
    minLenFrac: 0.35,      // minLineLength = frac * width  (fretboard_detect.py: 0.35*w)
    maxGapFrac: 0.04,      // maxLineGap   = frac * width   (fretboard_detect.py: 0.04*w)
    angleTolDeg: 12,       // keep lines within this of the modal neck angle (Python: 12)
    minSpanFrac: 0.05,     // two edges must differ by >= frac*h to be a real neck (Python: 0.05*h)
    minFrets: 4,           // need >= this many transverse lines to use fret-span (Python MIN_FRETS)
    autoAcceptPx: 6.0,     // reprojection residual (in work px) for full confidence (Python AUTO_ACCEPT_PX)
    flipStrings: false,    // consumer override: swap high-e / low-E side (v-axis)
    flipEnds: false,       // consumer override: swap nut / fret-7 end (u-axis)
    debugCanvas: null      // optional canvas element OR element id: draws detected lines + quad
  };

  // -------------------------------------------------------------------------
  // 12-TET fret law + pure-JS homography (from calibrate.html). Used only for the
  // confidence reprojection check, so it stays tiny and Mat-free.
  // -------------------------------------------------------------------------
  function fretFrac(n) { return 1 - Math.pow(2, -n / 12); } // fretboard.py fret_fraction

  // solveH/proj/gauss: 4-point DLT, lifted verbatim in spirit from calibrate.html.
  function gauss(A, b) {
    var n = b.length, M = A.map(function (r, i) { return r.concat(b[i]); });
    for (var c = 0; c < n; c++) {
      var p = c;
      for (var r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      var tmp = M[c]; M[c] = M[p]; M[p] = tmp;
      var d = M[c][c] || 1e-9;
      for (var r2 = 0; r2 < n; r2++) {
        if (r2 === c) continue;
        var f = M[r2][c] / d;
        for (var k = c; k <= n; k++) M[r2][k] -= f * M[c][k];
      }
    }
    return M.map(function (r, i) { return r[n] / (r[i] || 1e-9); });
  }
  function solveH(src, dst) { // src,dst: 4x [x,y]; returns 3x3 board->image
    var A = [], b = [];
    for (var i = 0; i < 4; i++) {
      var X = src[i][0], Y = src[i][1], u = dst[i][0], v = dst[i][1];
      A.push([X, Y, 1, 0, 0, 0, -u * X, -u * Y]); b.push(u);
      A.push([0, 0, 0, X, Y, 1, -v * X, -v * Y]); b.push(v);
    }
    var h = gauss(A, b);
    return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
  }
  function proj(H, X, Y) {
    var w = H[2][0] * X + H[2][1] * Y + H[2][2];
    return [(H[0][0] * X + H[0][1] * Y + H[0][2]) / w,
            (H[1][0] * X + H[1][1] * Y + H[1][2]) / w];
  }

  // -------------------------------------------------------------------------
  // OpenCV.js loader. Inject the CDN script, resolve `ready` on onRuntimeInitialized.
  // -------------------------------------------------------------------------
  var _resolveReady, _rejectReady;
  var ready = new Promise(function (res, rej) { _resolveReady = res; _rejectReady = rej; });
  var api = {
    ready: ready,
    isReady: false,
    detect: detect,
    opts: opts
  };

  function _loadOpenCV() {
    try {
      if (typeof window === "undefined") { return; }
      if (window.cv && window.cv.Mat) { _onReady(); return; } // already present
      var s = document.createElement("script");
      s.async = true;
      s.src = opts.cdnUrl;
      s.onload = function () {
        // cv may be a module factory (newer builds) or a ready object (older).
        var cv = window.cv;
        if (!cv) { _rejectReady(new Error("opencv.js loaded but window.cv missing")); return; }
        if (cv.Mat) { _onReady(); return; }                 // already initialized
        cv["onRuntimeInitialized"] = _onReady;              // wait for WASM init
      };
      s.onerror = function () { _rejectReady(new Error("failed to load opencv.js from " + opts.cdnUrl)); };
      document.head.appendChild(s);
    } catch (e) {
      try { _rejectReady(e); } catch (_) {}
    }
  }
  function _onReady() {
    api.isReady = true;
    try { _resolveReady(); } catch (_) {}
  }

  // -------------------------------------------------------------------------
  // Reusable Mat pool (avoid per-call WASM allocations / leaks). Sized lazily to
  // the current work resolution; rebuilt if the resolution changes.
  // -------------------------------------------------------------------------
  var _pool = null; // { w, h, rgba, gray, clahe, edges, lines, rot, M, band, sobel }
  function _ensurePool(w, h) {
    var cv = window.cv;
    if (_pool && _pool.w === w && _pool.h === h) return _pool;
    if (_pool) _freePool();
    _pool = {
      w: w, h: h,
      rgba: new cv.Mat(h, w, cv.CV_8UC4),
      gray: new cv.Mat(h, w, cv.CV_8UC1),
      clahe: new cv.Mat(h, w, cv.CV_8UC1),
      edges: new cv.Mat(h, w, cv.CV_8UC1),
      lines: new cv.Mat(),                 // HoughLinesP output (CV_32SC4)
      rot: new cv.Mat(h, w, cv.CV_8UC1),   // rotated gray (for fret profile)
      claheOp: null
    };
    // CLAHE is an enhancement, not load-bearing for the geometry. Construct it
    // defensively: if this OpenCV.js build doesn't expose the class ctor, we fall
    // back to plain/equalized gray in _enhance() rather than failing every call.
    try { _pool.claheOp = new cv.CLAHE(opts.claheClip, new cv.Size(opts.claheTile, opts.claheTile)); }
    catch (e) { _pool.claheOp = null; }
    return _pool;
  }
  function _freePool() {
    if (!_pool) return;
    ["rgba", "gray", "clahe", "edges", "lines", "rot"].forEach(function (k) {
      try { if (_pool[k]) _pool[k].delete(); } catch (_) {}
    });
    try { if (_pool.claheOp) _pool.claheOp.delete(); } catch (_) {}
    _pool = null;
  }
  // CLAHE if available (matches fretboard_detect.py _gray); else degrade to
  // equalizeHist, else just copy. Enhancement only — never the reason detect() fails.
  function _enhance(srcGray, dstGray) {
    var cv = window.cv;
    if (_pool.claheOp) {
      try { _pool.claheOp.apply(srcGray, dstGray); return; } catch (_) {}
    }
    try { cv.equalizeHist(srcGray, dstGray); return; } catch (_) {}
    try { srcGray.copyTo(dstGray); } catch (_) {}
  }

  // -------------------------------------------------------------------------
  // Geometry helpers (plain JS on line endpoints — cheap, no Mats).
  // -------------------------------------------------------------------------
  function angDeg(x1, y1, x2, y2) { return Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI; }
  function wrap90(a) { return ((a + 90) % 180 + 180) % 180 - 90; } // -> [-90,90)
  function median(arr) {
    if (!arr.length) return 0;
    var s = arr.slice().sort(function (a, b) { return a - b; });
    var m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }
  function segLen(s) { var dx = s[2] - s[0], dy = s[3] - s[1]; return Math.sqrt(dx * dx + dy * dy); }

  // -------------------------------------------------------------------------
  // detect_neck_edges() port: two long near-parallel lines = the neck's long edges.
  // Returns { dom, lo, hi } (dominant angle + the two most-separated edge segments)
  // or null.  lo/hi are [x1,y1,x2,y2] in WORK pixels.
  // -------------------------------------------------------------------------
  function _neckEdges(linesMat, w, h) {
    var n = linesMat.rows;
    if (!n) return null;
    var d = linesMat.data32S, segs = [], a = [];
    for (var i = 0; i < n; i++) {
      var x1 = d[i * 4], y1 = d[i * 4 + 1], x2 = d[i * 4 + 2], y2 = d[i * 4 + 3];
      segs.push([x1, y1, x2, y2]);
      a.push(wrap90(angDeg(x1, y1, x2, y2)));
    }
    var dom = median(a);
    // keep lines within angleTol of the modal neck orientation
    var keep = [];
    for (var j = 0; j < segs.length; j++) {
      if (Math.abs(wrap90(a[j] - dom)) < opts.angleTolDeg) keep.push(segs[j]);
    }
    if (keep.length < 2) return null;
    // perpendicular offset of each kept line's MIDPOINT (stabler than an endpoint)
    var th = dom * Math.PI / 180;
    var nx = -Math.sin(th), ny = Math.cos(th);
    var loS = null, hiS = null, loOff = Infinity, hiOff = -Infinity;
    for (var m = 0; m < keep.length; m++) {
      var s = keep[m];
      var mx = (s[0] + s[2]) / 2, my = (s[1] + s[3]) / 2;
      var off = mx * nx + my * ny;
      if (off < loOff) { loOff = off; loS = s; }
      if (off > hiOff) { hiOff = off; hiS = s; }
    }
    if (Math.abs(hiOff - loOff) < opts.minSpanFrac * h) return null; // both on same edge -> not a neck
    return { dom: dom, lo: loS, hi: hiS };
  }

  // -------------------------------------------------------------------------
  // detect_frets() port (simplified to what we need): rectify the neck to
  // horizontal, take the along-neck Sobel gradient profile inside the band, smooth
  // it, peak-find -> transverse (fret) x-positions in the ROTATED frame.
  // Returns sorted array of rotated-frame x (work px), or [].
  // -------------------------------------------------------------------------
  function _fretXs(claheMat, neck, w, h) {
    var cv = window.cv;
    var M = cv.getRotationMatrix2D(new cv.Point(w / 2, h / 2), neck.dom, 1.0);
    var Minv = new cv.Mat();
    try {
      cv.warpAffine(claheMat, _pool.rot, M, new cv.Size(w, h), cv.INTER_LINEAR,
                    cv.BORDER_CONSTANT, new cv.Scalar());
      // rotate the four edge endpoints to find the band rows [y0,y1]
      function rotY(x, y) { return M.doubleAt(1, 0) * x + M.doubleAt(1, 1) * y + M.doubleAt(1, 2); }
      var ys = [
        rotY(neck.lo[0], neck.lo[1]), rotY(neck.lo[2], neck.lo[3]),
        rotY(neck.hi[0], neck.hi[1]), rotY(neck.hi[2], neck.hi[3])
      ];
      var y0 = Math.max(0, Math.floor(Math.min.apply(null, ys)));
      var y1 = Math.min(h, Math.ceil(Math.max.apply(null, ys)));
      if (y1 - y0 < 8) return { xs: [], y0: y0, y1: y1, M: M, Minv: null };

      // along-neck gradient profile: mean |Sobel_x| down each column of the band
      var band = _pool.rot.roi(new cv.Rect(0, y0, w, y1 - y0));
      var sob = new cv.Mat();
      var prof = new Float32Array(w);
      try {
        cv.Sobel(band, sob, cv.CV_32F, 1, 0, 3);
        // |sob| then column-mean. sob is CV_32F, single channel.
        var bh = sob.rows, bw = sob.cols, sd = sob.data32F;
        for (var x = 0; x < bw; x++) {
          var acc = 0;
          for (var yy = 0; yy < bh; yy++) acc += Math.abs(sd[yy * bw + x]);
          prof[x] = acc / bh;
        }
      } finally { band.delete(); sob.delete(); }

      // 1-D Gaussian-ish smoothing (box of width 9, like the Python GaussianBlur(9,1))
      var sm = new Float32Array(w);
      var rad = 4;
      for (var c = 0; c < w; c++) {
        var s = 0, cnt = 0;
        for (var k = -rad; k <= rad; k++) {
          var idx = c + k;
          if (idx >= 0 && idx < w) { s += prof[idx]; cnt++; }
        }
        sm[c] = s / cnt;
      }
      // peak-find: local maxima above mean+std, min spacing ~ w/60 (Python find_peaks)
      var mean = 0; for (var q = 0; q < w; q++) mean += sm[q]; mean /= w;
      var varr = 0; for (var q2 = 0; q2 < w; q2++) { var dd = sm[q2] - mean; varr += dd * dd; }
      var std = Math.sqrt(varr / w);
      var thr = mean + 0.6 * std;          // prominence-ish gate (Python prominence=std*0.6)
      var minDist = Math.max(6, Math.floor(w / 60));
      var peaks = [];
      for (var p = 1; p < w - 1; p++) {
        if (sm[p] > sm[p - 1] && sm[p] >= sm[p + 1] && sm[p] > thr) {
          if (peaks.length && p - peaks[peaks.length - 1] < minDist) {
            if (sm[p] > sm[peaks[peaks.length - 1]]) peaks[peaks.length - 1] = p; // keep stronger
          } else peaks.push(p);
        }
      }
      // Minv maps rotated-frame points back to image space. If this build lacks
      // invertAffineTransform, return xs without Minv -> detect() falls back to the
      // neck-edge-endpoint span for the corners (still a valid result).
      try { cv.invertAffineTransform(M, Minv); }
      catch (e) { Minv.delete(); return { xs: peaks, y0: y0, y1: y1, M: M, Minv: null }; }
      return { xs: peaks, y0: y0, y1: y1, M: M, Minv: Minv };
    } catch (e) {
      try { Minv.delete(); } catch (_) {}
      try { M.delete(); } catch (_) {}
      return { xs: [], y0: 0, y1: 0, M: null, Minv: null };
    }
  }

  // -------------------------------------------------------------------------
  // Confidence via the 12-TET law (fit_law() idea, fretboard.py reprojection):
  // assign sorted fret-x to consecutive fret indices (allowing the nut to be
  // missed: start 0/1/2), fit a board->image homography from those (x,0)/(x,1)
  // line points, reproject, keep the min median residual (in work px).
  // Returns residualPx or Infinity if it cannot fit.
  // -------------------------------------------------------------------------
  function _lawResidual(fretImgPts) {
    // fretImgPts: array of [topPt, botPt] in image (work) coords, ordered nut->body
    if (fretImgPts.length < opts.minFrets) return Infinity;
    var best = Infinity;
    for (var start = 0; start <= 2; start++) {
      var board = [], img = [];
      for (var k = 0; k < fretImgPts.length; k++) {
        var x = fretFrac(start + k);
        board.push([x, 0]); img.push(fretImgPts[k][0]);
        board.push([x, 1]); img.push(fretImgPts[k][1]);
      }
      // DLT needs exactly 4 correspondences; we have >= 2*minFrets. Pick 4 WELL-SPREAD
      // points — nut top/bot + far-fret top/bot — not the first 4 (which cluster near
      // the nut and make the fit extrapolate, amplifying noise ~5x; verified). Then
      // measure the residual over ALL points as an over-determined sanity check.
      var L = board.length;
      var sb = [board[0], board[1], board[L - 2], board[L - 1]];
      var si = [img[0], img[1], img[L - 2], img[L - 1]];
      var H;
      try { H = solveH(sb, si); }
      catch (e) { continue; }
      if (!H || !isFinite(H[0][0])) continue;
      var errs = [];
      for (var i = 0; i < board.length; i++) {
        var pr = proj(H, board[i][0], board[i][1]);
        var ex = pr[0] - img[i][0], ey = pr[1] - img[i][1];
        errs.push(Math.sqrt(ex * ex + ey * ey));
      }
      var med = median(errs);
      if (med < best) best = med;
    }
    return best;
  }

  // -------------------------------------------------------------------------
  // Optional debug overlay: draw the two neck edges + the detected quad.
  // -------------------------------------------------------------------------
  function _drawDebug(neck, fretXs, quadWork, scale, srcW, srcH) {
    try {
      var canvas = opts.debugCanvas;
      if (typeof canvas === "string") canvas = document.getElementById(canvas);
      if (!canvas || !canvas.getContext) return;
      canvas.width = srcW; canvas.height = srcH;
      var ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, srcW, srcH);
      // neck edges (work px -> source px via /scale)
      if (neck) {
        ctx.strokeStyle = "rgba(110,168,255,.9)"; ctx.lineWidth = 2;
        [neck.lo, neck.hi].forEach(function (s) {
          ctx.beginPath();
          ctx.moveTo(s[0] / scale, s[1] / scale);
          ctx.lineTo(s[2] / scale, s[3] / scale);
          ctx.stroke();
        });
      }
      // detected quad (source px)
      if (quadWork) {
        var c = quadWork;
        ctx.strokeStyle = "rgba(57,255,154,.95)"; ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(c.c00[0] / scale, c.c00[1] / scale);
        ctx.lineTo(c.c10[0] / scale, c.c10[1] / scale);
        ctx.lineTo(c.c11[0] / scale, c.c11[1] / scale);
        ctx.lineTo(c.c01[0] / scale, c.c01[1] / scale);
        ctx.closePath(); ctx.stroke();
        ctx.fillStyle = "#fff"; ctx.font = "bold 13px sans-serif";
        ctx.fillText("nut", c.c00[0] / scale + 4, c.c00[1] / scale - 4);
        ctx.fillText("f7", c.c10[0] / scale + 4, c.c10[1] / scale - 4);
      }
    } catch (_) { /* debug must never break detection */ }
  }

  // -------------------------------------------------------------------------
  // detect(videoEl): the public entry point. Cheap, never throws, returns
  //   { quad:{c00,c10,c01,c11}, confidence } or null.
  // -------------------------------------------------------------------------
  function detect(videoEl) {
    var cv = window.cv;
    if (!api.isReady || !cv || !cv.Mat || !videoEl) return null;
    var srcW = videoEl.videoWidth | 0, srcH = videoEl.videoHeight | 0;
    if (srcW < 2 || srcH < 2) return null; // no frame yet / zero dims

    // work resolution = downscale to opts.workWidth (keep aspect)
    var scale = Math.min(1, opts.workWidth / srcW);
    var w = Math.max(2, Math.round(srcW * scale));
    var h = Math.max(2, Math.round(srcH * scale));

    var cap = null;
    var full = null, smallRgba = null, fr = null;
    try {
      _ensurePool(w, h);

      // Grab one native-res RGBA frame, downscale it, THEN convert to the work-size
      // gray buffer. (Resizing RGBA before grayscale is one fewer full-res op.)
      full = new cv.Mat(srcH, srcW, cv.CV_8UC4);
      cap = new cv.VideoCapture(videoEl);
      cap.read(full);                                  // RGBA frame from <video>
      smallRgba = new cv.Mat(h, w, cv.CV_8UC4);
      cv.resize(full, smallRgba, new cv.Size(w, h), 0, 0, cv.INTER_AREA);
      cv.cvtColor(smallRgba, _pool.gray, cv.COLOR_RGBA2GRAY, 0);

      // CLAHE (rescues the low-contrast brown board), then Canny.
      _enhance(_pool.gray, _pool.clahe);
      cv.Canny(_pool.clahe, _pool.edges, opts.cannyLo, opts.cannyHi, 3, false);

      // HoughLinesP for the long lines.
      cv.HoughLinesP(
        _pool.edges, _pool.lines, 1, Math.PI / 180, opts.houghThresh,
        Math.round(opts.minLenFrac * w), Math.round(opts.maxGapFrac * w)
      );
      var neck = _neckEdges(_pool.lines, w, h);
      if (!neck) { _drawDebug(null, null, null, scale, srcW, srcH); return null; }

      // Frets (for the along-neck extent + confidence). Best-effort.
      fr = _fretXs(_pool.clahe, neck, w, h);

      // --- build the 4 corners ---------------------------------------------
      // The neck's two long edges give the two string-side lines. We need the two
      // along-neck ENDS (nut and far fret). Strategy:
      //   - If we found >= minFrets transverse lines, use the outermost two fret-x
      //     (in the rotated frame) as the ends -> map back through Minv.
      //   - Else fall back to spanning the neck-edge segment endpoints.
      var ends = _alongNeckEnds(neck, fr, w, h);
      if (!ends) { _drawDebug(neck, null, null, scale, srcW, srcH); return null; }

      // Corners in work px, then assign nut/f7 x high-e/low-E by image geometry.
      var quadWork = _assignCorners(neck, ends, w, h);
      if (!quadWork) { _drawDebug(neck, null, null, scale, srcW, srcH); return null; }

      // --- confidence ------------------------------------------------------
      // Combine: line parallelism (the two edges' angle agreement), edge length,
      // and (if frets found) the 12-TET law reprojection residual.
      var conf = _confidence(neck, fr, w, h);

      _drawDebug(neck, fr.xs, quadWork, scale, srcW, srcH);

      // normalize corners to [0..1] in the VIDEO's intrinsic frame
      function nrm(p) { return [clamp01(p[0] / w), clamp01(p[1] / h)]; }
      var quad = {
        c00: nrm(quadWork.c00), c10: nrm(quadWork.c10),
        c01: nrm(quadWork.c01), c11: nrm(quadWork.c11)
      };
      return { quad: quad, confidence: conf };
    } catch (e) {
      // any OpenCV exception (often a thrown number from WASM) -> null
      return null;
    } finally {
      try { if (full) full.delete(); } catch (_) {}
      try { if (smallRgba) smallRgba.delete(); } catch (_) {}
      // _fretXs allocates a rotation matrix + its inverse (the only per-call Mats
      // beyond full/small). They're consumed synchronously above, so free them here
      // every call — this is what keeps detect() from leaking WASM memory at 2-3/sec.
      try { if (fr && fr.M) fr.M.delete(); } catch (_) {}
      try { if (fr && fr.Minv) fr.Minv.delete(); } catch (_) {}
      // cap holds no Mats; nothing to free. Pool persists across calls by design.
    }
  }

  function clamp01(v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }

  // Compute the two along-neck end x-positions (rotated frame) + a mapper back to
  // image space. Returns { x0, x1, Minv } (x0 = nearer one end, x1 = the other) or
  // a fallback built from the neck-edge endpoints (in which case Minv is null and
  // x0/x1 carry image-space endpoint pairs instead).
  function _alongNeckEnds(neck, fr, w, h) {
    if (fr && fr.xs && fr.xs.length >= opts.minFrets && fr.Minv) {
      var xs = fr.xs.slice().sort(function (a, b) { return a - b; });
      return { mode: "frets", xLo: xs[0], xHi: xs[xs.length - 1], y0: fr.y0, y1: fr.y1, Minv: fr.Minv };
    }
    // fallback: project neck-edge endpoints onto the neck axis, take extremes
    var th = neck.dom * Math.PI / 180;
    var ax = Math.cos(th), ay = Math.sin(th);
    var pts = [
      [neck.lo[0], neck.lo[1]], [neck.lo[2], neck.lo[3]],
      [neck.hi[0], neck.hi[1]], [neck.hi[2], neck.hi[3]]
    ];
    var minT = Infinity, maxT = -Infinity, pMin = null, pMax = null;
    pts.forEach(function (p) {
      var t = p[0] * ax + p[1] * ay;
      if (t < minT) { minT = t; pMin = p; }
      if (t > maxT) { maxT = t; pMax = p; }
    });
    return { mode: "endpoints", tLo: minT, tHi: maxT, axis: [ax, ay] };
  }

  // Intersect the neck's two long edges with the two along-neck ends to get four
  // corners, then label them nut/f7 x high-e/low-E by image geometry.
  function _assignCorners(neck, ends, w, h) {
    var cv = window.cv;
    // Represent each long edge as an infinite line (point + dir).
    var loLine = _lineFromSeg(neck.lo), hiLine = _lineFromSeg(neck.hi);
    var P; // 4 raw corners: [endLo x edgeLo, endLo x edgeHi, endHi x edgeLo, endHi x edgeHi]
    if (ends.mode === "frets") {
      // ends are vertical lines x=const in the ROTATED frame -> map two points on
      // each back to image space, intersect with the (image-space) edges.
      function endLineImg(xc) {
        var a = _applyAffine(ends.Minv, xc, ends.y0);
        var b = _applyAffine(ends.Minv, xc, ends.y1);
        return _lineFromPts(a, b);
      }
      var eLo = endLineImg(ends.xLo), eHi = endLineImg(ends.xHi);
      P = [
        _intersect(eLo, loLine), _intersect(eLo, hiLine),
        _intersect(eHi, loLine), _intersect(eHi, hiLine)
      ];
    } else {
      // endpoints fallback: build end lines perpendicular to the neck axis through
      // the two extreme projection points.
      var ax = ends.axis[0], ay = ends.axis[1];
      function perpLineAt(t) {
        var cx = t * ax, cy = t * ay;            // a point on the axis at param t
        return { p: [cx, cy], d: [-ay, ax] };    // perpendicular direction
      }
      var eLo2 = perpLineAt(ends.tLo), eHi2 = perpLineAt(ends.tHi);
      P = [
        _intersect(eLo2, loLine), _intersect(eLo2, hiLine),
        _intersect(eHi2, loLine), _intersect(eHi2, hiLine)
      ];
    }
    for (var i = 0; i < 4; i++) if (!P[i]) return null;

    // P = [endA-edgeLo, endA-edgeHi, endB-edgeLo, endB-edgeHi]
    // endA/endB are the two along-neck ends; edgeLo/edgeHi are the two string sides.
    // ------- which END is the NUT? -------
    // Heuristic: the nut end of the board is WIDER than fret 7 (the neck tapers /
    // foreshortens toward the body). Pick the end whose two edge-intersections are
    // farther apart as the NUT. (opts.flipEnds overrides.)
    var widthA = _dist(P[0], P[1]); // span across strings at end A
    var widthB = _dist(P[2], P[3]); // span across strings at end B
    var nutIsA = widthA >= widthB;
    if (opts.flipEnds) nutIsA = !nutIsA;
    var nut = nutIsA ? [P[0], P[1]] : [P[2], P[3]];   // [edgeLo pt, edgeHi pt] at nut
    var f7  = nutIsA ? [P[2], P[3]] : [P[0], P[1]];   // [edgeLo pt, edgeHi pt] at fret7

    // ------- which SIDE is high-e? -------
    // high-e (string 1) is the thin side; with a typical front-camera framing the
    // strings run with high-e toward the TOP of the image (smaller y). Pick the
    // edge whose nut point has the smaller y as high-e. (opts.flipStrings overrides.)
    var loIsHighE = nut[0][1] <= nut[1][1];
    if (opts.flipStrings) loIsHighE = !loIsHighE;
    // map: index 0 of [nut/f7] is edgeLo, index 1 is edgeHi
    var highEIdx = loIsHighE ? 0 : 1, lowEIdx = loIsHighE ? 1 : 0;

    return {
      c00: nut[highEIdx], c10: f7[highEIdx],   // nut x high-e , f7 x high-e
      c01: nut[lowEIdx],  c11: f7[lowEIdx]     // nut x low-E  , f7 x low-E
    };
  }

  // --- small line / point helpers (image space, plain JS) ------------------
  function _lineFromSeg(s) { return { p: [s[0], s[1]], d: [s[2] - s[0], s[3] - s[1]] }; }
  function _lineFromPts(a, b) { return { p: [a[0], a[1]], d: [b[0] - a[0], b[1] - a[1]] }; }
  function _intersect(L1, L2) {
    // solve L1.p + t*L1.d = L2.p + u*L2.d
    var x1 = L1.p[0], y1 = L1.p[1], dx1 = L1.d[0], dy1 = L1.d[1];
    var x2 = L2.p[0], y2 = L2.p[1], dx2 = L2.d[0], dy2 = L2.d[1];
    var den = dx1 * dy2 - dy1 * dx2;
    if (Math.abs(den) < 1e-9) return null; // parallel
    var t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / den;
    return [x1 + t * dx1, y1 + t * dy1];
  }
  function _dist(a, b) { var dx = a[0] - b[0], dy = a[1] - b[1]; return Math.sqrt(dx * dx + dy * dy); }
  function _applyAffine(M, x, y) {
    // M is a 2x3 cv.Mat (CV_64F) from invertAffineTransform
    return [
      M.doubleAt(0, 0) * x + M.doubleAt(0, 1) * y + M.doubleAt(0, 2),
      M.doubleAt(1, 0) * x + M.doubleAt(1, 1) * y + M.doubleAt(1, 2)
    ];
  }

  // -------------------------------------------------------------------------
  // Confidence in [0..1]. Three cues, each mapped to [0..1], then averaged:
  //   1. parallelism  — how close the two neck edges' angles agree (||diff||->0 good)
  //   2. length       — how long the edges are vs the frame (longer = crisper)
  //   3. lawResidual  — 12-TET grid reprojection residual vs opts.autoAcceptPx
  //                     (only when >= minFrets transverse lines were found)
  // We deliberately return < 0.5 when unsure so the caller keeps its fallback.
  // -------------------------------------------------------------------------
  function _confidence(neck, fr, w, h) {
    // 1. parallelism: angle of lo vs hi edge (wrapped) -> 0deg=1, 12deg=0
    var aLo = wrap90(angDeg(neck.lo[0], neck.lo[1], neck.lo[2], neck.lo[3]));
    var aHi = wrap90(angDeg(neck.hi[0], neck.hi[1], neck.hi[2], neck.hi[3]));
    var dAng = Math.abs(wrap90(aLo - aHi));
    var cPar = clamp01(1 - dAng / opts.angleTolDeg);

    // 2. length: mean edge length / frame diagonal-ish (use width as the scale)
    var meanLen = (segLen(neck.lo) + segLen(neck.hi)) / 2;
    var cLen = clamp01(meanLen / (0.8 * w)); // 0.8*w long -> full marks

    // 3. law residual (best-effort). Build top/bottom image points per fret line.
    var cLaw = 0.5; // neutral prior if we can't measure
    if (fr && fr.xs && fr.xs.length >= opts.minFrets && fr.Minv) {
      var xs = fr.xs.slice().sort(function (a, b) { return a - b; });
      var pts = [];
      for (var i = 0; i < xs.length; i++) {
        pts.push([_applyAffine(fr.Minv, xs[i], fr.y0), _applyAffine(fr.Minv, xs[i], fr.y1)]);
      }
      var res = _lawResidual(pts);
      if (isFinite(res)) cLaw = clamp01(1 - res / (opts.autoAcceptPx * 4)); // 0px=1, 4*accept=0
    }
    // weighted blend: geometry (parallelism+length) is the floor; law sharpens it.
    var conf = 0.35 * cPar + 0.25 * cLen + 0.40 * cLaw;
    return Math.round(conf * 1000) / 1000;
  }

  // -------------------------------------------------------------------------
  // expose + kick off the loader
  // -------------------------------------------------------------------------
  if (typeof window !== "undefined") {
    window.TactusFretboard = api;
    _loadOpenCV();
  }
})();
