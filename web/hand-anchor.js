/* ===========================================================================
 * hand-anchor.js — markerless guitar-neck anchor via MediaPipe Hands
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS
 *   The OpenCV/Hough detector (web/fretboard-autodetect.js) needs strong, long,
 *   near-parallel straight lines to find the neck. On a busy hackathon stage —
 *   curtains, a Redis banner, table edges — those backgrounds have STRONGER
 *   straight lines than a thin diagonal guitar neck, so Hough locks onto the
 *   wrong edges and returns zero usable neck. truth.md §2 calls for MediaPipe
 *   Hands as the browser vision: it is trained for hands "in the wild" and is
 *   robust to clutter, and the fretting hand is the single most reliable cue to
 *   where the neck is. So instead of finding the neck directly, we find the
 *   FRETTING HAND and estimate the neck quad around it.
 *
 * APPROACH (deliberately simple — "ponytail" mindset)
 *   - Lean on the library: MediaPipe Tasks Vision HandLandmarker does ALL the
 *     vision. We only do cheap 2D vector math on the 21 returned landmarks.
 *   - One self-contained classic <script>. No build step, no ES-module export.
 *     The CDN ESM bundle is pulled in via a dynamic import() inside this script.
 *   - Never throw: detect() wraps everything and returns null on ANY failure
 *     (model not ready, 0-dim video, no hand, MediaPipe error, bad landmarks).
 *
 * THE QUAD IS APPROXIMATE BY DESIGN
 *   A hand covers only ~3–4 frets and tells us nothing certain about the far
 *   (fret-7) end or the exact string span — so the fret-7 edge is an estimate
 *   extrapolated from hand size and orientation. That is fine: the consumer
 *   (glow.html) eases toward this quad and lets the user fine-tune by dragging
 *   the four corner handles. Goal = "roughly right + stable", not pixel-perfect.
 *   Flip knobs (alongFlip / acrossFlip) let the consumer correct a mirrored
 *   guess live without reloading.
 *
 * CREDIT: Google MediaPipe Tasks Vision (HandLandmarker).
 *   https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
 *
 * CONSUMER CONTRACT (must match web/fretboard-autodetect.js exactly so this is
 * a drop-in alternative — glow.html reads window.TactusFretboard the same way):
 *   detect(video) -> { quad:{ c00,c10,c01,c11 }, confidence } | null
 *     each corner = [nx, ny] normalized [0..1] in the VIDEO's intrinsic frame.
 *   u-axis = along neck (0 = nut .. 1 = fret-7);  v-axis = across strings
 *     (0 = high-e / string-1 side .. 1 = low-E / string-6 side).
 *     c00 = nut   x high-e   ·  c10 = fret-7 x high-e
 *     c01 = nut   x low-E    ·  c11 = fret-7 x low-E
 *
 * ASSUMPTIONS the consumer must satisfy:
 *   - The FRETTING hand is visible on the neck (we detect a single hand).
 *   - One hand in frame is ideal (numHands:1). The strumming hand, if it
 *     dominates the frame instead, will produce a wrong-but-stable quad — the
 *     user drags to fix, or sets alongFlip/acrossFlip if it's mirrored.
 * =========================================================================== */
(function () {
  "use strict";

  // ---- tuning knobs (all overridable live by the consumer) -----------------
  var opts = {
    reachFrets:   6,      // how many frets the u-axis (nut→fret7) should span.
    spanScale:    1.6,    // multiply finger span to cover all 6 strings + margin.
    alongFlip:    false,  // flip the along-neck (u) direction if nut/f7 are swapped.
    acrossFlip:   false,  // flip the across-strings (v) direction if e/E are swapped.
    minConfidence: 0.4,   // floor we report; the consumer applies its own gate too.
    debugCanvas:  null    // canvas element OR element-id string -> draw landmarks + quad.
  };

  // ---- module state --------------------------------------------------------
  var landmarker = null;   // the MediaPipe HandLandmarker, once loaded.
  var isReady = false;     // cheap synchronous gate for the consumer's poll loop.
  var lastTs = 0;          // monotonic timestamp guard for detectForVideo().

  // CDN pins (verified to resolve 2026-06):
  //   bundle: https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs
  //   wasm:   https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm
  //   model:  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
  var VISION_BUNDLE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
  var WASM_ROOT     = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
  var MODEL_URL     = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

  // MediaPipe hand-landmark indices we use (the knuckle row + wrist).
  var WRIST = 0, IDX_MCP = 5, MID_MCP = 9, RING_MCP = 13, PINKY_MCP = 17;

  // ---- model loading -------------------------------------------------------
  // Dynamically import the ESM bundle from a classic script. `ready` resolves
  // when the model is usable, and rejects on failure so the consumer keeps its
  // manual fallback (it polls isReady, which stays false on failure).
  var ready = (async function load() {
    var vision = await import(VISION_BUNDLE);
    var HandLandmarker = vision.HandLandmarker;
    var FilesetResolver = vision.FilesetResolver;
    var fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: "VIDEO",
      numHands: 1
    });
    isReady = true;
  })();
  // Keep the rejection from becoming an unhandled promise rejection in the
  // console; the consumer still sees failure through isReady staying false.
  ready.catch(function () { isReady = false; });

  // ---- small 2D vector helpers (landmarks are {x,y} normalized 0..1) --------
  function sub(a, b) { return [a.x - b.x, a.y - b.y]; }
  function add(a, b) { return [a[0] + b[0], a[1] + b[1]]; }
  function mul(v, k) { return [v[0] * k, v[1] * k]; }
  function len(v)    { return Math.hypot(v[0], v[1]); }
  function norm(v)   { var L = len(v); return L > 1e-6 ? [v[0] / L, v[1] / L] : [0, 0]; }
  function perp(v)   { return [-v[1], v[0]]; }           // +90° rotation.
  function clamp01(x){ return x < 0 ? 0 : x > 1 ? 1 : x; }
  function cln(p)    { return [clamp01(p[0]), clamp01(p[1])]; }

  // ---- HAND -> NECK QUAD ----------------------------------------------------
  // Geometry (documented convention; flip knobs let the consumer correct it):
  //   u (ALONG neck, nut→fret7): the MCP knuckle row, indexMCP -> pinkyMCP.
  //     When fretting, the fingers spread ACROSS several frets, so this row runs
  //     roughly along the neck's LENGTH. Its sign is ambiguous (which end is the
  //     nut?), so we orient it and expose alongFlip.
  //   v (ACROSS strings, high-e→low-E): the wrist -> middleMCP direction, which
  //     is roughly perpendicular to the neck. We force it perpendicular to u
  //     (using the hand only to pick its sign), and expose acrossFlip.
  // Sizing:
  //   handAlong = |indexMCP - pinkyMCP| (knuckle-row width) sets one "hand unit".
  //   A hand covers ~3.5 frets, so u half-length = handAlong * reachFrets/3.5 / 2.
  //   The string span uses the finger reach (wrist->middle MCP) * spanScale.
  //   Everything scales with measured hand size, so the quad tracks distance.
  function quadFromHand(lm) {
    var idx = lm[IDX_MCP], mid = lm[MID_MCP], ring = lm[RING_MCP], pky = lm[PINKY_MCP], wr = lm[WRIST];
    if (!idx || !mid || !ring || !pky || !wr) return null;

    // u-axis: knuckle row direction (index -> pinky), along the neck length.
    var uVec = sub(pky, idx);
    var handAlong = len(uVec);
    if (handAlong < 1e-4) return null;          // degenerate / collapsed hand.
    var uHat = norm(uVec);
    if (opts.alongFlip) uHat = mul(uHat, -1);

    // v-axis: perpendicular to u, sign chosen to agree with wrist->middleMCP
    // (which points from the palm toward the fingertips, i.e. across the strings).
    var vHat = perp(uHat);
    var palmDir = sub(mid, wr);
    if (vHat[0] * palmDir[0] + vHat[1] * palmDir[1] < 0) vHat = mul(vHat, -1);
    if (opts.acrossFlip) vHat = mul(vHat, -1);

    // Center the quad on the knuckle row's midpoint.
    var center = [(idx.x + pky.x) / 2, (idx.y + pky.y) / 2];

    // Half-extents (in normalized video units), scaled by measured hand size.
    var uHalf = handAlong * (opts.reachFrets / 3.5) / 2;
    var palmLen = len(palmDir);
    var vHalf = Math.max(handAlong, palmLen) * opts.spanScale / 2;

    var uStep = mul(uHat, uHalf);
    var vStep = mul(vHat, vHalf);

    // Corners: c<u><v>. u=0 nut side (-uStep), u=1 fret7 (+uStep);
    //                   v=0 high-e (-vStep), v=1 low-E (+vStep).
    var nutHi = add(add(center, mul(uStep, -1)), mul(vStep, -1));
    var f7Hi  = add(add(center, uStep),          mul(vStep, -1));
    var nutLo = add(add(center, mul(uStep, -1)), vStep);
    var f7Lo  = add(add(center, uStep),          vStep);

    return { c00: cln(nutHi), c10: cln(f7Hi), c01: cln(nutLo), c11: cln(f7Lo) };
  }

  // ---- confidence ----------------------------------------------------------
  // Blend three cheap, bounded cues into 0..1:
  //   (a) the landmarker's own handedness/visibility score (presence),
  //   (b) how "spread" the knuckles are vs. a typical open hand (a fist or a
  //       hand seen edge-on gives a thin, unreliable row -> lower),
  //   (c) how planar/even the knuckle spacing is (the four MCPs should be a
  //       smooth arc; a jumbled row means a partial/occluded hand -> lower).
  // We return at least opts.minConfidence's neighborhood for a clean hand and
  // drop well below it for partial/edge hands so the consumer's gate rejects them.
  function confidenceOf(lm, handednessScore) {
    var idx = lm[IDX_MCP], mid = lm[MID_MCP], ring = lm[RING_MCP], pky = lm[PINKY_MCP];
    if (!idx || !mid || !ring || !pky) return 0;

    var rowW = len(sub(pky, idx));            // knuckle-row width (≈ hand size).
    // (b) spread: a healthy open hand's MCP row spans a decent fraction of frame.
    //     Map [~0.04 .. ~0.18] of frame width to [0..1].
    var spread = clamp01((rowW - 0.04) / (0.18 - 0.04));

    // (c) evenness: index→mid, mid→ring, ring→pinky gaps should be similar.
    var g1 = len(sub(mid, idx)), g2 = len(sub(ring, mid)), g3 = len(sub(pky, ring));
    var gAvg = (g1 + g2 + g3) / 3;
    var even = 1;
    if (gAvg > 1e-5) {
      var dev = (Math.abs(g1 - gAvg) + Math.abs(g2 - gAvg) + Math.abs(g3 - gAvg)) / (3 * gAvg);
      even = clamp01(1 - dev);                // 1 = perfectly even, 0 = jumbled.
    }

    var presence = (typeof handednessScore === "number") ? clamp01(handednessScore) : 0.5;

    // Weighted blend; presence dominates, geometry trims down bad poses.
    var conf = 0.5 * presence + 0.3 * spread + 0.2 * even;
    return clamp01(conf);
  }

  // ---- debug overlay (optional) --------------------------------------------
  // Draws the 21 landmarks (knuckles highlighted) + the estimated quad so the
  // consumer can eyeball whether the guess and the u/v orientation are right.
  function resolveCanvas(c) {
    if (!c) return null;
    if (typeof c === "string") { try { return document.getElementById(c); } catch (_) { return null; } }
    return (c && c.getContext) ? c : null;
  }
  function drawDebug(video, lm, quad, conf) {
    var cv = resolveCanvas(opts.debugCanvas);
    if (!cv) return;
    try {
      var W = video.videoWidth, H = video.videoHeight;
      if (!W || !H) return;
      if (cv.width !== W) cv.width = W;
      if (cv.height !== H) cv.height = H;
      var g = cv.getContext("2d");
      if (!g) return;
      g.clearRect(0, 0, W, H);

      if (lm) {
        // all landmarks
        g.fillStyle = "rgba(0,200,255,0.8)";
        for (var i = 0; i < lm.length; i++) {
          g.beginPath(); g.arc(lm[i].x * W, lm[i].y * H, 3, 0, 7); g.fill();
        }
        // knuckle row + wrist highlighted
        g.fillStyle = "rgba(255,80,80,0.95)";
        var hi = [WRIST, IDX_MCP, MID_MCP, RING_MCP, PINKY_MCP];
        for (var k = 0; k < hi.length; k++) {
          var p = lm[hi[k]];
          g.beginPath(); g.arc(p.x * W, p.y * H, 5, 0, 7); g.fill();
        }
      }

      if (quad) {
        g.strokeStyle = "rgba(255,215,0,0.95)"; g.lineWidth = 2;
        g.beginPath();
        g.moveTo(quad.c00[0] * W, quad.c00[1] * H);   // nut x high-e
        g.lineTo(quad.c10[0] * W, quad.c10[1] * H);   // f7  x high-e
        g.lineTo(quad.c11[0] * W, quad.c11[1] * H);   // f7  x low-E
        g.lineTo(quad.c01[0] * W, quad.c01[1] * H);   // nut x low-E
        g.closePath(); g.stroke();
        g.fillStyle = "rgba(255,215,0,0.95)"; g.font = "14px sans-serif";
        g.fillText("nut", quad.c00[0] * W + 4, quad.c00[1] * H - 4);
        g.fillText("f7",  quad.c10[0] * W + 4, quad.c10[1] * H - 4);
        if (typeof conf === "number") g.fillText("conf " + conf.toFixed(2), 6, 16);
      }
    } catch (_) { /* debug must never break detection */ }
  }

  // ---- public detect() — never throws, returns null on any failure ---------
  function detect(video) {
    try {
      if (!isReady || !landmarker) return null;
      if (!video || !video.videoWidth || !video.videoHeight) return null;

      // detectForVideo needs strictly-increasing timestamps (ms). performance.now()
      // can repeat between very fast polls; bump past lastTs to stay monotonic and
      // never pass 0/duplicate (MediaPipe errors on non-monotonic timestamps).
      var ts = Math.round(performance.now());
      if (ts <= lastTs) ts = lastTs + 1;
      lastTs = ts;

      var res = landmarker.detectForVideo(video, ts);
      if (!res || !res.landmarks || !res.landmarks.length) {
        drawDebug(video, null, null, null);
        return null;
      }

      var lm = res.landmarks[0];
      if (!lm || lm.length < 18) { drawDebug(video, lm || null, null, null); return null; }

      var quad = quadFromHand(lm);
      if (!quad) { drawDebug(video, lm, null, null); return null; }

      // handedness score (0..1) if MediaPipe provided it for this hand.
      var hScore;
      try {
        var h = (res.handednesses || res.handedness || [])[0];
        if (h && h[0] && typeof h[0].score === "number") hScore = h[0].score;
      } catch (_) {}

      var conf = confidenceOf(lm, hScore);
      drawDebug(video, lm, quad, conf);
      return { quad: quad, confidence: conf };
    } catch (_) {
      // any MediaPipe / DOM error -> null (consumer keeps its manual fallback).
      return null;
    }
  }

  // ---- expose the contract -------------------------------------------------
  // `isReady` is read live via a getter so the consumer's poll loop always sees
  // the current state (it flips true only after the async model load resolves).
  window.TactusHand = {
    ready: ready,
    detect: detect,
    opts: opts
  };
  Object.defineProperty(window.TactusHand, "isReady", {
    enumerable: true,
    get: function () { return isReady; }
  });
})();
