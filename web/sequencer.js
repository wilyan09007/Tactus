/* Tactus — self-paced performance sequencer with error tracking.
 *
 * The digital tab has NO clock. Progress is event-driven: a step is "okayed"
 * only when the player actually lands it (detected by mic-pitch / vision / a
 * manual cue). The sequencer therefore is the single source of truth for:
 *
 *   - WHERE we are in the song          (cursor over steps)
 *   - what the actuators must anticipate (NOW step + NEXT step = "speaker memory")
 *   - every mistake, classified and kept (error log -> Redis memory, see redis_live)
 *
 * The crux the interface needs: when a note is played wrong we do NOT advance,
 * and we can rewind a step. Either way `onSpeakerMemory({now,next})` re-fires so
 * the actuators' knowledge of "what's next" is corrected the instant the player
 * slips. That rewind-and-re-anticipate loop is the error feedback.
 *
 * Framework-free. Exposes window.Tactus.Sequencer + window.Tactus.buildSteps.
 */
(function (root) {
  "use strict";

  var STRING_NAMES = ["e", "B", "G", "D", "A", "E"]; // index 0 -> string 1 (high e)
  function stringName(s) { return STRING_NAMES[s - 1] || ("s" + s); }

  /* ---- build an ordered step list from a song / flat note list ----
   * A "step" is one strum the player must land: a set of (string,fret) targets.
   * Accepts either {loop:[{name,shape:[[string,fret]...]}...], repeats} or a flat
   * NOTES array [{string,fret,ci,...}] grouped by chord index `ci`.            */
  function buildSteps(song, opts) {
    opts = opts || {};
    var steps = [];
    if (song && Array.isArray(song.loop)) {
      var reps = opts.repeats || 2;
      for (var r = 0; r < reps; r++) {
        song.loop.forEach(function (c) {
          steps.push(makeStep(c.name, c.shape));
        });
      }
    } else if (Array.isArray(song)) {
      // flat NOTES: group by chord index `ci` (fallback: each note its own step)
      var byCi = {}; var order = [];
      song.forEach(function (n) {
        var k = (n.ci != null ? n.ci : (n.chordId != null ? n.chordId : order.length));
        if (!byCi[k]) { byCi[k] = []; order.push(k); }
        byCi[k].push([n.string, n.fret]);
      });
      order.forEach(function (k, i) { steps.push(makeStep(song[0] && song[0].name || ("step " + (i + 1)), byCi[k])); });
    }
    steps.forEach(function (s, i) { s.index = i; });
    return steps;
  }

  function makeStep(name, shape) {
    return {
      index: 0,
      name: name || "?",
      targets: shape.map(function (sf) {
        return { string: sf[0], fret: sf[1], satisfied: false };
      }),
    };
  }

  function clamp(x, a, b) { return Math.min(b, Math.max(a, x)); }

  /* ---- the sequencer ---- */
  function Sequencer(steps, opts) {
    opts = opts || {};
    this.steps = steps || [];
    this.idx = 0;
    this.errors = [];          // full mistake log {stepIndex, kind, target, observed, t, stringName}
    this.attempts = 0;
    this.cb = {
      change: opts.onChange || noop,            // any state change (re-render)
      advance: opts.onAdvance || noop,          // a step was cleared
      error: opts.onError || noop,              // a mistake happened (-> Redis)
      speaker: opts.onSpeakerMemory || noop,    // {now,next} the actuators must hold
      haptic: opts.onHaptic || noop,            // {targets:[{string,fret,channel...}]}
      complete: opts.onComplete || noop,
    };
    // pitch tolerance & whether wrong notes auto-rewind
    this.autoRewindOnWrong = opts.autoRewindOnWrong !== false; // default true
    this._announce();
    this.cb.change(this.snapshot());   // render ribbon/HUD on load, before any input
  }

  Sequencer.prototype.current = function () { return this.steps[this.idx] || null; };
  Sequencer.prototype.next = function () { return this.steps[this.idx + 1] || null; };

  // emit the speaker memory + haptic anticipation for the current cursor
  Sequencer.prototype._announce = function () {
    var now = this.current(), nxt = this.next();
    this.cb.speaker({ now: now, next: nxt, idx: this.idx, total: this.steps.length });
    if (now) this.cb.haptic({ phase: "anticipate", step: now, targets: now.targets.map(hapticTarget) });
  };

  /* feed a detected note/observation.
   * obs = {string, fret, articulation?: 'clean'|'buzz'|'muted', source?}
   * returns {match, kind, advanced}                                            */
  Sequencer.prototype.feed = function (obs) {
    var step = this.current();
    if (!step) return { match: false, kind: "done", advanced: false };
    this.attempts++;

    var art = obs.articulation || "clean";
    var tgt = matchTarget(step, obs);

    // wrong string/fret entirely
    if (!tgt) {
      this._logError(step, "wrong-note", null, obs);
      if (this.autoRewindOnWrong) this._reanchor();
      this.cb.change(this.snapshot());
      return { match: false, kind: "wrong-note", advanced: false };
    }
    // right place, but buzzed / muted -> mistake, target stays unsatisfied
    if (art === "buzz" || art === "muted") {
      this._logError(step, art, tgt, obs);
      this.cb.haptic({ phase: "correct", step: step, targets: [hapticTarget(tgt)] });
      this.cb.change(this.snapshot());
      return { match: true, kind: art, advanced: false };
    }
    // clean hit on a real target
    tgt.satisfied = true;
    var advanced = false;
    if (step.targets.every(function (t) { return t.satisfied; })) {
      advanced = this._advance();
    } else {
      this.cb.change(this.snapshot());
    }
    return { match: true, kind: "clean", advanced: advanced };
  };

  // force-clear the current step ("okay this note") regardless of detection
  Sequencer.prototype.okay = function () {
    var step = this.current();
    if (!step) return false;
    step.targets.forEach(function (t) { t.satisfied = true; });
    return this._advance();
  };

  Sequencer.prototype._advance = function () {
    var done = this.current();
    this.cb.haptic({ phase: "strum", step: done, targets: done.targets.map(hapticTarget) });
    this.cb.advance(done);
    this.idx++;
    if (this.idx >= this.steps.length) {
      this.cb.change(this.snapshot());
      this.cb.complete({ errors: this.errors.length, attempts: this.attempts });
      return true;
    }
    this._announce();                 // <-- speaker memory updates to the new NOW/NEXT
    this.cb.change(this.snapshot());
    return true;
  };

  /* rewind one step: the player slipped, so the actuators' "next" must revert.
   * Clears the reverted step's satisfaction and re-announces speaker memory.   */
  Sequencer.prototype.rewind = function () {
    this.idx = clamp(this.idx - 1, 0, this.steps.length - 1);
    var step = this.current();
    if (step) step.targets.forEach(function (t) { t.satisfied = false; });
    this._announce();
    this.cb.change(this.snapshot());
    return step;
  };

  // wrong note on the current step: drop any partial progress so the actuators
  // re-anticipate the SAME step cleanly (no false "next").
  Sequencer.prototype._reanchor = function () {
    var step = this.current();
    if (step) step.targets.forEach(function (t) { t.satisfied = false; });
    this._announce();
  };

  Sequencer.prototype._logError = function (step, kind, tgt, obs) {
    var err = {
      stepIndex: step.index,
      stepName: step.name,
      kind: kind,                                   // wrong-note | buzz | muted
      target: tgt ? { string: tgt.string, fret: tgt.fret } : null,
      observed: { string: obs.string, fret: obs.fret, pitchMidi: obs.pitchMidi },
      stringName: tgt ? stringName(tgt.string) : (obs.string ? stringName(obs.string) : null),
      source: obs.source || "unknown",
      t: Date.now(),
      features: obs.features || null,               // 28-dim audio vector if present -> Redis
    };
    this.errors.push(err);
    this.cb.error(err);
    this.cb.haptic({ phase: "error", step: step, kind: kind, targets: tgt ? [hapticTarget(tgt)] : [] });
  };

  Sequencer.prototype.reset = function () {
    this.idx = 0; this.errors = []; this.attempts = 0;
    this.steps.forEach(function (s) { s.targets.forEach(function (t) { t.satisfied = false; }); });
    this._announce(); this.cb.change(this.snapshot());
  };

  Sequencer.prototype.snapshot = function () {
    return {
      idx: this.idx, total: this.steps.length,
      steps: this.steps, now: this.current(), next: this.next(),
      errors: this.errors, attempts: this.attempts,
      accuracy: this.attempts ? 1 - this.errors.length / this.attempts : 1,
    };
  };

  /* ---- helpers ---- */
  function matchTarget(step, obs) {
    // prefer exact (string,fret); allow string-only when fret unknown (-1/null)
    var exact = null, byString = null;
    step.targets.forEach(function (t) {
      if (t.string === obs.string && (obs.fret == null || obs.fret < 0 || t.fret === obs.fret)) {
        if (obs.fret === t.fret) exact = t; else if (!byString) byString = t;
      }
    });
    return exact || byString;
  }

  // (string,fret) -> haptic channels, matching config/channel_map.json axes:
  // back row = the 6 strings; torso zone = the fret. Abstract enough for sim or HW.
  function hapticTarget(t) {
    return {
      string: t.string, fret: t.fret,
      backChannel: t.string,                 // ch 1..6 (string axis)
      torsoZone: t.fret > 0 ? clamp(t.fret, 1, 6) : 0,  // ch 7..12 (fret axis), 0 = open
    };
  }

  function noop() {}

  root.Tactus = root.Tactus || {};
  root.Tactus.Sequencer = Sequencer;
  root.Tactus.buildSteps = buildSteps;
  root.Tactus.stringName = stringName;
  if (typeof module !== "undefined" && module.exports) module.exports = root.Tactus;
})(typeof window !== "undefined" ? window : globalThis);
