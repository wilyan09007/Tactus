/* Tactus — coach overlay (error feedback made visible).
 *
 * Drop-in panel for the live interface. Renders, in the interface itself:
 *   1. SPEAKER MEMORY  — the NOW step + the NEXT step the actuators are holding.
 *      This is the thing that must visibly correct itself the instant the player
 *      slips (wrong note -> NEXT reverts; rewind -> NOW steps back).
 *   2. A step ribbon    — done / now / next / upcoming, with a red error badge
 *      on any step the player mistook (the error track, on screen).
 *   3. Live verdict     — clean / buzz / muted / wrong-note for the last attempt.
 *   4. Occlusion HUD    — vision vs audio confidence + the fused prediction
 *      (filled by predictor.js for the AI-robustness story).
 *
 * Usage:
 *   var coach = Tactus.mountCoach(document.querySelector('.main'));
 *   var seq   = new Tactus.Sequencer(steps, coach.handlers);
 *   // later: coach.setPrediction({...}); coach.setOcclusion(0.8);
 */
(function (root) {
  "use strict";

  var CSS = "\
  .tc-wrap{position:absolute;left:16px;top:64px;z-index:6;width:300px;font-family:'Nunito',-apple-system,Segoe UI,sans-serif;color:#3A2A4A;pointer-events:none}\
  .tc-card{background:#FFFFFFe8;backdrop-filter:blur(6px);border:2.5px solid #FFB8D4;border-radius:18px;padding:12px 14px;margin-bottom:10px;box-shadow:0 8px 24px #ff4d8d22}\
  .tc-h{font:800 10px/1 'Fredoka','Nunito',sans-serif;letter-spacing:.16em;color:#C9356C;text-transform:uppercase;margin-bottom:8px;display:flex;align-items:center;gap:6px}\
  .tc-h .dot{width:7px;height:7px;border-radius:50%;background:#FF4D8D;box-shadow:0 0 0 0 #ff4d8d66;animation:tcpulse 1.4s infinite}\
  @keyframes tcpulse{0%{box-shadow:0 0 0 0 #ff4d8d66}70%{box-shadow:0 0 0 8px #ff4d8d00}100%{box-shadow:0 0 0 0 #ff4d8d00}}\
  .tc-mem{display:flex;align-items:stretch;gap:8px}\
  .tc-slot{flex:1;border-radius:14px;padding:8px 10px;text-align:center}\
  .tc-slot.now{background:linear-gradient(180deg,#FFE3EF,#FFD2E6);outline:3px solid #FF4D8D}\
  .tc-slot.next{background:#FFF3E0;outline:2px dashed #FF9F43}\
  .tc-slot .lbl{font:800 9px/1 'Fredoka';letter-spacing:.12em;color:#A8729A;margin-bottom:3px}\
  .tc-slot.next .lbl{color:#C98A33}\
  .tc-slot .nm{font:700 24px/1 'Fredoka','Nunito';color:#3A2A4A}\
  .tc-slot .sf{font:700 10px/1.3 ui-monospace,monospace;color:#9B7FB8;margin-top:4px;min-height:13px}\
  .tc-arrow{align-self:center;font-size:18px;color:#FF9F43;font-weight:900}\
  .tc-ribbon{display:flex;flex-wrap:wrap;gap:5px}\
  .tc-pip{width:26px;height:26px;border-radius:9px;display:flex;align-items:center;justify-content:center;font:700 11px 'Fredoka';position:relative;background:#EFE6F7;color:#B49AD0}\
  .tc-pip.done{background:#D6F7E6;color:#1f9d63}\
  .tc-pip.now{background:linear-gradient(180deg,#FF6FA3,#FF4D8D);color:#fff;box-shadow:0 3px 10px #ff4d8d55;animation:tcpulse 1.4s infinite}\
  .tc-pip.next{background:#FFF3E0;color:#C98A33;outline:2px dashed #FF9F43}\
  .tc-pip .err{position:absolute;top:-6px;right:-6px;min-width:15px;height:15px;border-radius:8px;background:#FF3B3B;color:#fff;font:800 9px/15px 'Nunito';text-align:center;padding:0 3px}\
  .tc-verdict{display:flex;align-items:center;gap:8px;font:800 13px 'Fredoka','Nunito'}\
  .tc-badge{font:800 11px 'Fredoka';padding:3px 9px;border-radius:999px;color:#fff}\
  .tc-clean{background:#1fb866}.tc-buzz{background:#FF9F43}.tc-muted{background:#8a7bd8}.tc-wrong{background:#FF3B3B}\
  .tc-sub{font:700 11px 'Nunito';color:#9B7FB8;margin-top:6px;display:flex;justify-content:space-between}\
  .tc-hud .bars{display:flex;flex-direction:column;gap:6px;margin-top:2px}\
  .tc-bar{display:flex;align-items:center;gap:8px;font:800 10px 'Fredoka';color:#6b5a82}\
  .tc-bar .track{flex:1;height:8px;border-radius:999px;background:#EFE6F7;overflow:hidden}\
  .tc-bar .fill{height:100%;border-radius:999px;transition:width .2s}\
  .tc-bar.vis .fill{background:linear-gradient(90deg,#72B6FF,#4D8DFF)}\
  .tc-bar.aud .fill{background:linear-gradient(90deg,#FF9F43,#FF4D8D)}\
  .tc-pred{font:800 12px 'Fredoka';margin-top:8px;color:#3A2A4A}\
  .tc-pred b{color:#C9356C}\
  .tc-occ{font:800 9px 'Fredoka';letter-spacing:.1em;padding:2px 8px;border-radius:999px;background:#D6F7E6;color:#1f9d63}\
  .tc-occ.bad{background:#FFE0E0;color:#d62b2b}\
  ";

  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function chips(step) {
    if (!step) return "";
    return step.targets.map(function (t) {
      return Tactus.stringName(t.string) + (t.fret > 0 ? t.fret : "\u25cb");
    }).join(" ");
  }

  function mountCoach(container, opts) {
    opts = opts || {};
    if (!document.getElementById("tc-style")) {
      var st = el("style"); st.id = "tc-style"; st.textContent = CSS; document.head.appendChild(st);
    }
    var wrap = el("div", "tc-wrap");

    // 1) speaker memory card
    var memCard = el("div", "tc-card");
    memCard.appendChild(el("div", "tc-h", "<span class='dot'></span> Actuator memory"));
    var mem = el("div", "tc-mem");
    var nowSlot = el("div", "tc-slot now", "<div class='lbl'>NOW</div><div class='nm' id='tc-now'>\u2014</div><div class='sf' id='tc-nowsf'></div>");
    var arrow = el("div", "tc-arrow", "\u2192");
    var nextSlot = el("div", "tc-slot next", "<div class='lbl'>NEXT</div><div class='nm' id='tc-next'>\u2014</div><div class='sf' id='tc-nextsf'></div>");
    mem.appendChild(nowSlot); mem.appendChild(arrow); mem.appendChild(nextSlot);
    memCard.appendChild(mem);
    wrap.appendChild(memCard);

    // 2) ribbon + 3) verdict
    var progCard = el("div", "tc-card");
    progCard.appendChild(el("div", "tc-h", "Error track"));
    var ribbon = el("div", "tc-ribbon"); progCard.appendChild(ribbon);
    var verdict = el("div", "tc-verdict"); verdict.style.marginTop = "10px";
    verdict.innerHTML = "<span id='tc-verdict-txt' style='color:#B49AD0'>waiting\u2026</span>";
    progCard.appendChild(verdict);
    var sub = el("div", "tc-sub", "<span id='tc-acc'>acc \u2014</span><span id='tc-errs'>0 errors</span>");
    progCard.appendChild(sub);
    wrap.appendChild(progCard);

    // 4) occlusion / prediction HUD (filled by predictor.js)
    var hud = el("div", "tc-card tc-hud");
    hud.appendChild(el("div", "tc-h", "<span style='flex:1'>Prediction</span><span class='tc-occ' id='tc-occ'>VISIBLE</span>"));
    var bars = el("div", "bars");
    bars.innerHTML =
      "<div class='tc-bar vis'>VIS<div class='track'><div class='fill' id='tc-vis' style='width:0%'></div></div></div>" +
      "<div class='tc-bar aud'>AUD<div class='track'><div class='fill' id='tc-aud' style='width:0%'></div></div></div>";
    hud.appendChild(bars);
    hud.appendChild(el("div", "tc-pred", "<span id='tc-pred'>\u2014</span>"));
    wrap.appendChild(hud);

    container.appendChild(wrap);

    var $ = function (id) { return wrap.querySelector("#" + id); };

    function renderRibbon(snap) {
      ribbon.innerHTML = "";
      var errByStep = {};
      snap.errors.forEach(function (e) { errByStep[e.stepIndex] = (errByStep[e.stepIndex] || 0) + 1; });
      snap.steps.forEach(function (s, i) {
        var cls = "tc-pip " + (i < snap.idx ? "done" : i === snap.idx ? "now" : i === snap.idx + 1 ? "next" : "");
        var pip = el("div", cls, s.name);
        if (errByStep[i]) pip.appendChild(el("span", "err", String(errByStep[i])));
        ribbon.appendChild(pip);
      });
    }

    var handlers = {
      onSpeakerMemory: function (m) {
        $("tc-now").textContent = m.now ? m.now.name : "\u2713 done";
        $("tc-nowsf").textContent = chips(m.now);
        $("tc-next").textContent = m.next ? m.next.name : "\u2014";
        $("tc-nextsf").textContent = chips(m.next);
      },
      onChange: function (snap) {
        renderRibbon(snap);
        $("tc-acc").textContent = "acc " + Math.round(snap.accuracy * 100) + "%";
        $("tc-errs").textContent = snap.errors.length + (snap.errors.length === 1 ? " error" : " errors");
      },
      onError: function (e) {
        var map = { "wrong-note": ["tc-wrong", "WRONG"], buzz: ["tc-buzz", "BUZZ"], muted: ["tc-muted", "MUTED"] };
        var m = map[e.kind] || ["tc-wrong", e.kind.toUpperCase()];
        var where = e.stringName ? (" on " + e.stringName) : "";
        verdict.innerHTML = "<span class='tc-badge " + m[0] + "'>" + m[1] + "</span>" +
          "<span id='tc-verdict-txt'>" + (e.target ? (e.stepName + where) : ("not " + e.stepName)) + "</span>";
        if (opts.onError) opts.onError(e);
      },
      onAdvance: function (step) {
        verdict.innerHTML = "<span class='tc-badge tc-clean'>CLEAN</span><span id='tc-verdict-txt'>" + step.name + " \u2713</span>";
      },
      onHaptic: opts.onHaptic || function () {},
      onComplete: function (r) {
        verdict.innerHTML = "<span class='tc-badge tc-clean'>DONE</span><span id='tc-verdict-txt'>" +
          r.errors + " mistakes logged</span>";
        if (opts.onComplete) opts.onComplete(r);
      },
    };

    return {
      el: wrap,
      handlers: handlers,
      // C: occlusion-robust prediction HUD
      setOcclusion: function (visConf, audConf) {
        $("tc-vis").style.width = Math.round(clamp01(visConf) * 100) + "%";
        $("tc-aud").style.width = Math.round(clamp01(audConf) * 100) + "%";
        var occ = $("tc-occ");
        if (visConf < 0.35) { occ.textContent = "OCCLUDED \u2192 AUDIO"; occ.className = "tc-occ bad"; }
        else { occ.textContent = "VISIBLE"; occ.className = "tc-occ"; }
      },
      setPrediction: function (p) {
        $("tc-pred").innerHTML = p ? ("\u2192 <b>" + p.label + "</b> &middot; " + (p.articulation || "") +
          " <span style='color:#9B7FB8'>(" + p.source + ")</span>") : "\u2014";
      },
    };
  }
  function clamp01(x) { return Math.min(1, Math.max(0, x || 0)); }

  root.Tactus = root.Tactus || {};
  root.Tactus.mountCoach = mountCoach;
})(typeof window !== "undefined" ? window : globalThis);
