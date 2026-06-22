/* Tactus — occlusion-proof note input from the microphone.
 *
 * Pitch does not care whether the camera is blocked. This module runs an
 * autocorrelation pitch tracker on the mic and turns a stable pitch into a
 * (string,fret) observation the Sequencer can consume. It is context-aware:
 * given the step the player is *supposed* to land, it prefers the candidate
 * (string,fret) on an expected string, which is what makes a noisy pitch usable.
 *
 *   var mic = Tactus.PitchInput({
 *       expected: function(){ return seq.current(); },   // for disambiguation
 *       onNote: function(obs){ seq.feed(obs); },          // {string,fret,pitchMidi,articulation,conf,source}
 *       onLevel: function(rms,conf){ ... }                // optional meter
 *   });
 *   mic.start();
 */
(function (root) {
  "use strict";
  var OPEN_MIDI = { 6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64 }; // low-E..high-e
  var MAX_FRET = 7;

  function midiFromHz(hz) { return 69 + 12 * Math.log2(hz / 440); }

  // pitch -> best (string,fret), preferring strings in `preferred` (the expected step)
  function toStringFret(midi, preferred) {
    var best = null;
    for (var s = 1; s <= 6; s++) {
      var fret = Math.round(midi - OPEN_MIDI[s]);
      if (fret < 0 || fret > MAX_FRET) continue;
      var err = Math.abs((midi - OPEN_MIDI[s]) - fret);
      var pref = preferred && preferred.indexOf(s) >= 0 ? 0 : 0.5; // bias toward expected strings
      var cost = err + pref;
      if (!best || cost < best.cost) best = { string: s, fret: fret, cost: cost, err: err };
    }
    return best;
  }

  function autocorrelate(buf, sr) {
    var n = buf.length, rms = 0;
    for (var i = 0; i < n; i++) rms += buf[i] * buf[i];
    rms = Math.sqrt(rms / n);
    if (rms < 0.008) return { hz: -1, rms: rms, clarity: 0 };   // too quiet
    var r1 = 0, r2 = n - 1, thres = 0.2;
    for (var i = 0; i < n / 2; i++) if (Math.abs(buf[i]) < thres) { r1 = i; break; }
    for (var i = 1; i < n / 2; i++) if (Math.abs(buf[n - i]) < thres) { r2 = n - i; break; }
    buf = buf.slice(r1, r2); n = buf.length;
    var c = new Array(n).fill(0);
    for (var lag = 0; lag < n; lag++)
      for (var i = 0; i < n - lag; i++) c[lag] += buf[i] * buf[i + lag];
    var d = 0; while (d < n - 1 && c[d] > c[d + 1]) d++;
    var maxv = -1, maxp = -1;
    for (var i = d; i < n; i++) if (c[i] > maxv) { maxv = c[i]; maxp = i; }
    var T0 = maxp;
    if (T0 <= 0) return { hz: -1, rms: rms, clarity: 0 };
    // parabolic interpolation
    var x1 = c[T0 - 1] || 0, x2 = c[T0], x3 = c[T0 + 1] || 0;
    var a = (x1 + x3 - 2 * x2) / 2, b = (x3 - x1) / 2;
    if (a) T0 = T0 - b / (2 * a);
    var clarity = c[0] ? maxv / c[0] : 0;
    return { hz: sr / T0, rms: rms, clarity: clarity };
  }

  function PitchInput(opts) {
    opts = opts || {};
    var ctx, analyser, src, raf, running = false, buf;
    var lastMidi = null, stableCount = 0, emittedFor = null;

    function loop() {
      if (!running) return;
      analyser.getFloatTimeDomainData(buf);
      var p = autocorrelate(buf, ctx.sampleRate);
      var conf = Math.max(0, Math.min(1, (p.clarity - 0.5) * 2)) * Math.min(1, p.rms * 25);
      if (opts.onLevel) opts.onLevel(p.rms, conf);
      if (p.hz > 60 && p.hz < 1200 && p.clarity > 0.9) {
        var midi = Math.round(midiFromHz(p.hz));
        if (midi === lastMidi) stableCount++; else { lastMidi = midi; stableCount = 0; }
        // emit once per stable onset (debounce by note value)
        if (stableCount === 2 && emittedFor !== midi) {
          emittedFor = midi;
          var step = opts.expected ? opts.expected() : null;
          var preferred = step ? step.targets.map(function (t) { return t.string; }) : null;
          var sf = toStringFret(midi, preferred);
          if (sf) {
            // crude articulation: very low energy + low clarity => muted/dead
            var art = (p.rms < 0.02 && p.clarity < 0.95) ? "muted" : "clean";
            opts.onNote({
              string: sf.string, fret: sf.fret, pitchMidi: midi,
              articulation: art, conf: conf, source: "mic",
            });
          }
        }
      } else if (p.rms < 0.01) {
        emittedFor = null; // silence resets the onset latch
      }
      raf = requestAnimationFrame(loop);
    }

    return {
      start: function () {
        if (running) return Promise.resolve();
        return navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } })
          .then(function (stream) {
            ctx = new (window.AudioContext || window.webkitAudioContext)();
            analyser = ctx.createAnalyser(); analyser.fftSize = 2048;
            buf = new Float32Array(analyser.fftSize);
            src = ctx.createMediaStreamSource(stream); src.connect(analyser);
            running = true; loop();
          });
      },
      stop: function () { running = false; if (raf) cancelAnimationFrame(raf); if (ctx) ctx.close(); },
      toStringFret: toStringFret,
    };
  }

  root.Tactus = root.Tactus || {};
  root.Tactus.PitchInput = PitchInput;
})(typeof window !== "undefined" ? window : globalThis);
