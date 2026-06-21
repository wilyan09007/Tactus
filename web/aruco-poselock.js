// aruco-poselock.js — lock a three.js Group onto a real ArUco marker in a webcam feed.
//
// Detect (js-aruco2) -> pose (POS.Posit) -> handedness fix -> one-euro smooth ->
// confidence gate (fade to a rendered "ghost" neck when the marker is lost).
// Verified against js-aruco2 source (aruco.js, posit1/2.js, samples/debug-posit) and
// three.js r160 (ArUco agent, 2026-06-21).
//
// Requires js-aruco2 globals on the page (cv.js -> aruco.js -> svd.js -> posit2.js),
// which attach window.AR / window.CV / window.POS. Print the marker from the
// ARUCO_MIP_36h12 dictionary and MEASURE its physical side length (mm).
//
// Units are MILLIMETRES throughout (MARKER_SIZE_MM seeds the model scale): build the
// fretboard geometry in mm, set camera near~10 / far~10000.

import * as THREE from 'three';

// adaptive 1-euro filter (low jitter when still, low lag when moving)
export class OneEuro {
  constructor(minCutoff = 1.2, beta = 0.02, dCutoff = 1.0) {
    Object.assign(this, { minCutoff, beta, dCutoff, xPrev: null, dxPrev: 0, tPrev: null });
  }
  _a(cutoff, dt) { const r = 2 * Math.PI * cutoff * dt; return r / (r + 1); }
  filter(x, t) {
    if (this.xPrev === null) { this.xPrev = x; this.tPrev = t; return x; }
    const dt = Math.max(1e-3, t - this.tPrev);
    const dx = (x - this.xPrev) / dt;
    const aD = this._a(this.dCutoff, dt);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;
    const a = this._a(this.minCutoff + this.beta * Math.abs(dxHat), dt);
    const xHat = a * x + (1 - a) * this.xPrev;
    this.xPrev = xHat; this.dxPrev = dxHat; this.tPrev = t;
    return xHat;
  }
}

// Set the PerspectiveCamera fov from the webcam focal length (vertical fov).
export function matchCameraToWebcam(camera, videoW, videoH, focalPx) {
  camera.fov = 2 * Math.atan(videoH / (2 * focalPx)) * 180 / Math.PI;
  camera.aspect = videoW / videoH;
  camera.updateProjectionMatrix();
}

// Build a per-frame pose updater. Pass the <video>, the target Group (matrixAutoUpdate=false),
// and an optional ghost-neck Object3D to fade in when tracking is lost.
export function makePoseLock({
  video, group, ghost = null,
  markerSizeMm = 80, markerId = 0,
  videoW = 1280, videoH = 720, focalPx = 1280,
  errorMax = 5.0,
}) {
  const detector = new AR.Detector({ dictionaryName: 'ARUCO_MIP_36h12' });
  const posit = new POS.Posit(markerSizeMm, focalPx);
  const grab = Object.assign(document.createElement('canvas'), { width: videoW, height: videoH });
  const gctx = grab.getContext('2d', { willReadFrequently: true });
  group.matrixAutoUpdate = false;

  const fx = new OneEuro(), fy = new OneEuro(), fz = new OneEuro();
  const smoothQ = new THREE.Quaternion();
  const _m = new THREE.Matrix4(), _pos = new THREE.Vector3(), _scl = new THREE.Vector3(1, 1, 1);
  const _qTarget = new THREE.Quaternion();
  let conf = 0, haveQ = false;

  function detect() {
    if (video.readyState !== video.HAVE_ENOUGH_DATA) return null;
    gctx.drawImage(video, 0, 0, grab.width, grab.height);
    const markers = detector.detect(gctx.getImageData(0, 0, grab.width, grab.height));
    return markers.find(m => m.id === markerId) || null;
  }

  function pose(marker) {
    const c = marker.corners.map(p => ({ x: p.x - videoW / 2, y: videoH / 2 - p.y })); // recenter + Y-up
    return posit.pose(c);
  }

  // CV (+Z into scene, Y-down-ish) -> three.js (-Z into scene, Y-up): negate rotation rows 1&2 and T[2].
  function rotToQuat(R, out) {
    _m.set(
       R[0][0],  R[0][1],  R[0][2], 0,
      -R[1][0], -R[1][1], -R[1][2], 0,
      -R[2][0], -R[2][1], -R[2][2], 0,
            0,        0,        0,  1);
    out.setFromRotationMatrix(_m);
  }

  // call every animation frame
  return function update(nowSec) {
    const marker = detect();
    let target = 0;
    if (marker) {
      const p = pose(marker);
      if (p.bestError >= 0 && p.bestError < errorMax) {
        rotToQuat(p.bestRotation, _qTarget);
        if (!haveQ) { smoothQ.copy(_qTarget); haveQ = true; } else { smoothQ.slerp(_qTarget, 0.35); }
        const T = p.bestTranslation;
        _pos.set(fx.filter(T[0], nowSec), fy.filter(T[1], nowSec), fz.filter(-T[2], nowSec)); // -z handedness
        group.matrix.compose(_pos, smoothQ, _scl);
        group.matrixWorldNeedsUpdate = true;
        target = 1;
      }
    }
    conf += (target - conf) * 0.15; // ~6-frame ease, no hard pop
    group.visible = conf > 0.02;
    group.traverse(o => { if (o.material) { o.material.transparent = true; o.material.opacity = conf; } });
    if (ghost) {
      ghost.visible = conf < 0.5;
      ghost.traverse(o => { if (o.material) { o.material.transparent = true; o.material.opacity = (1 - conf) * 0.6; } });
    }
    return conf;
  };
}
