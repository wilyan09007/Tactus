// alphatab-loader.js — sheet music (GuitarPro / MusicXML) -> Tactus chart.
//
// Pipeline: a .gp/.gp5/.gpx/.gp7 or .musicxml/.mxl file  ->  alphaTab parse (headless,
// no DOM/synth)  ->  { bpm, notes:[{t, string, fret, dur, chordId}], end }.
// Notes that sound together share the same `t` and a `chordId` -> the chord-field view
// lights their cells on the same frame for free.
//
// Requires the alphaTab UMD bundle on the page (exposes window.alphaTab):
//   <script src="https://cdn.jsdelivr.net/npm/@coderline/alphatab@latest/dist/alphaTab.min.js"></script>
//   (npm: `npm i @coderline/alphatab`)
//
// Verified against context7 /websites/alphatab_net (alphaTab agent, 2026-06-21):
//   alphaTab.importer.ScoreLoader.loadScoreFromBytes(Uint8Array, Settings?) -> Score
//   Beat.absolutePlaybackStart (ticks), Beat.playbackDuration (ticks),
//   Note.string (1=high-e..6=low-E), Note.fret, MasterBar.tempoAutomations,
//   MidiUtils.QuarterTime (= 960 ticks/quarter).

const AT = () => window.alphaTab;

// Parse bytes (or a URL) into an alphaTab Score, fully headless (no container, no synth).
export async function loadScore(urlOrBytes) {
  let data;
  if (typeof urlOrBytes === 'string') {
    const res = await fetch(urlOrBytes);
    data = new Uint8Array(await res.arrayBuffer());
  } else if (urlOrBytes instanceof ArrayBuffer) {
    data = new Uint8Array(urlOrBytes);
  } else {
    data = urlOrBytes; // already a Uint8Array
  }
  const settings = new (AT().Settings)();
  // ScoreLoader sniffs GuitarPro-binary vs MusicXML from the bytes automatically.
  return AT().importer.ScoreLoader.loadScoreFromBytes(data, settings);
}

// Load a score and flatten it to the Tactus chart.
// flipString: set true if a given file numbers strings 6=high-e..1=low-E (rare) -> uses 7-string.
export async function loadChart(urlOrBytes, { trackIndex = 0, flipString = false } = {}) {
  const score = await loadScore(urlOrBytes);
  const QUARTER = AT().model.MidiUtils.QuarterTime; // 960
  const track = score.tracks[trackIndex] || score.tracks[0];

  // ---- tick -> seconds, honoring simple tempo automation (one BPM per bar) ----
  let tick = 0, bpm = score.tempo || 120;
  const segs = []; // [{ startTick, bpm, startSec }]
  for (const mb of score.masterBars) {
    const autos = mb.tempoAutomations || [];
    if (autos.length) bpm = autos[autos.length - 1].value;
    segs.push({ startTick: tick, bpm });
    const num = mb.timeSignatureNumerator, den = mb.timeSignatureDenominator;
    tick += Math.round(QUARTER * 4 * num / den);
  }
  let acc = 0;
  for (let i = 0; i < segs.length; i++) {
    if (i > 0) { const p = segs[i - 1]; acc += (segs[i].startTick - p.startTick) * (60 / (p.bpm * QUARTER)); }
    segs[i].startSec = acc;
  }
  const tickToSec = (t) => {
    let seg = segs[0];
    for (let i = 1; i < segs.length; i++) { if (segs[i].startTick <= t) seg = segs[i]; else break; }
    return seg.startSec + (t - seg.startTick) * (60 / (seg.bpm * QUARTER));
  };

  // ---- walk tracks > staves > bars > voices > beats > notes ----
  const notes = [];
  let chordId = 0;
  for (const stave of track.staves) {
    for (const bar of stave.bars) {
      for (const voice of bar.voices) {
        for (const beat of voice.beats) {
          if (beat.isRest || !beat.notes || beat.notes.length === 0) continue;
          const onset = beat.absolutePlaybackStart;
          const t = +tickToSec(onset).toFixed(4);
          const dur = +Math.max(0.05, tickToSec(onset + beat.playbackDuration) - tickToSec(onset)).toFixed(4);
          const id = beat.notes.length > 1 ? ++chordId : null; // multi-note beat = a chord
          for (const note of beat.notes) {
            if (note.isTieDestination) continue;                 // sustain of an earlier hit, not a new onset
            if (note.string == null || note.fret == null) continue;
            notes.push({ t, string: flipString ? 7 - note.string : note.string, fret: note.fret, dur, chordId: id });
          }
        }
      }
    }
  }
  notes.sort((a, b) => a.t - b.t || a.string - b.string);
  const end = notes.length ? notes[notes.length - 1].t + notes[notes.length - 1].dur + 1.2 : 0;
  return { bpm: score.tempo || 120, notes, end };
}
