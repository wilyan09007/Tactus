/* Tactus — occlusion-robust note prediction (audio-led multimodal fusion).
 *
 * The honest result from our own data (data/analysis/exp/separability_3way.json):
 *   audio-only articulation acc = 80.3%   vision-only = 38.2%   fused = 80.8%
 * i.e. AUDIO carries the prediction; vision only refines fret POSITION when the
 * neck is actually visible. Pitch is occlusion-invariant by construction.
 *
 * So we fuse with confidence weighting, and when the camera view is obstructed
 * (vision confidence collapses) the prediction gracefully falls back to audio
 * and stays correct. That is the demonstrable AI-robustness claim.
 *
 *   var pred = Tactus.Predictor(coach);   // coach = mountCoach(...) handle
 *   pred.observeAudio({string,fret,pitchMidi,articulation,conf});
 *   pred.observeVision({string,fret,conf});
 *   pred.setVisionOccluded(true);         // demo: cover the lens
 */
(function (root) {
  "use strict";
  var OPEN_MIDI = { 6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64 };
  var NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  function noteLabel(string, fret) {
    if (string == null) return "\u2014";
    var midi = OPEN_MIDI[string] + (fret > 0 ? fret : 0);
    return NOTE[midi % 12] + (Math.floor(midi / 12) - 1);
  }

  function Predictor(coach) {
    var self = {
      audioConf: 0, visionConf: 0.85,   // assume the neck detector is up by default
      _audio: null, _vision: null, _occluded: false, _lastAudioT: 0, _lastVisionT: 0,
    };

    self.observeAudio = function (obs) {
      self._audio = obs; self._lastAudioT = Date.now();
      if (obs.conf != null) self.audioConf = obs.conf;
      else self.audioConf = 0.9;
      fuse();
    };
    self.observeVision = function (obs) {
      self._vision = obs; self._lastVisionT = Date.now();
      self.visionConf = self._occluded ? 0.12 : (obs.conf != null ? obs.conf : 0.85);
      fuse();
    };
    self.setVisionOccluded = function (on) {
      self._occluded = on;
      self.visionConf = on ? 0.12 : 0.85;
      fuse();
    };

    function decay(t, ms) { return Math.max(0, 1 - (Date.now() - t) / ms); }

    function fuse() {
      // Vision (neck detector) is continuous -> use its confidence directly.
      // Audio is bursty (per pluck) -> decay so the HUD reflects live availability.
      var vC = self._occluded ? 0.12 : self.visionConf;
      var aC = self.audioConf * decay(self._lastAudioT || 0, 1500);

      var out = null;
      if (aC > 0.25) {
        // AUDIO leads: pitch -> (string,fret) + articulation. Vision refines fret
        // only if it's confident AND agrees on the string.
        var s = self._audio.string, f = self._audio.fret;
        var src = "audio";
        if (vC > 0.5 && self._vision && self._vision.string === s) {
          f = self._vision.fret != null ? self._vision.fret : f;
          src = "audio+vision";
        } else if (vC <= 0.35) {
          src = "audio (occluded)";
        }
        out = { string: s, fret: f, articulation: self._audio.articulation || "clean", source: src,
                label: noteLabel(s, f), conf: Math.max(aC, vC) };
      } else if (vC > 0.35 && self._vision) {
        out = { string: self._vision.string, fret: self._vision.fret, articulation: "?",
                source: "vision-only", label: noteLabel(self._vision.string, self._vision.fret), conf: vC };
      }
      if (coach) { coach.setOcclusion(vC, aC); if (out) coach.setPrediction(out); }
      self.last = out;
      return out;
    }

    // keep the HUD live even with no new events (confidence decay visible)
    setInterval(fuse, 200);
    self.fuse = fuse;
    self.noteLabel = noteLabel;
    return self;
  }

  root.Tactus = root.Tactus || {};
  root.Tactus.Predictor = Predictor;
  root.TactusPredictor = true;
})(typeof window !== "undefined" ? window : globalThis);
