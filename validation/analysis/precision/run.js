// Reproduce gamma=70.7 Float32 emulation numbers and run mixed-precision ablations.
'use strict';
const C = require('./core.js');
const { f32, id } = C;

const F32 = { Q: f32, P: f32, V: f32, A: f32, Rt: f32, Kr: f32 };
const F64 = { Q: id, P: id, V: id, A: id, Rt: id, Kr: id };
const S32 = { RS: f32, XS: f32, US: f32, AS: f32, clampLoop: true };
const S64 = { RS: id, XS: id, US: id, AS: id, clampLoop: false };
const KBIN = { 128: 9, 512: 11, 1024: 12, 2048: 13 };

function run(beta, M, arith, store, optOver = {}, nPts = 256, metricExtra = false) {
  const src = C.makeSource(beta, 2.40, 16);
  const K = C.populate(src, M, store);
  K.tEval = src.tEval; K.beta = beta;
  const opt = Object.assign({ kBin: KBIN[M] || 11, nNewton: 2, limiter: true, kappaSplineFloor: 5e-5, safeguard: false }, optOver);
  const sw = C.sweepPoints(1.15, src.tEval, beta, nPts);
  let eSq = 0, nSq = 0, alphaMin = 1, sec = 0;
  const out = [];
  for (const p of sw.pts) {
    // evaluator gets the f32 point (or the f64 point if Q=id); reference at the identical point
    const qx = arith.Q === id ? p.x64 : p.x, qy = arith.Q === id ? p.y64 : p.y;
    const r = C.evalF32(K, qx, qy, arith, opt);
    const ex = C.exactLW(qx, qy, src.tEval, beta);
    eSq += (r.ex - ex.ex) ** 2 + (r.ey - ex.ey) ** 2; nSq += ex.ex ** 2 + ex.ey ** 2;
    alphaMin = Math.min(alphaMin, r.alphaMin); if (r.sec) sec++;
    out.push({ p, r, ex });
  }
  const res = { L2: 100 * Math.sqrt(eSq / nSq), alphaMin, secClampedPts: sec, clampFired: K.clampFired || 0, out, sw, src, K };
  return res;
}
module.exports = { run, F32, F64, S32, S64, KBIN };

if (require.main === module) {
  const beta = 0.9999;
  console.log('== Reproduction (gamma=70.71), CPU Float32 emulation vs exact at identical f32 points ==');
  for (const M of [512, 1024, 2048]) {
    const a = run(beta, M, F32, S32);
    const b = run(beta, M, F64, S64, { limiter: false, kBin: 24, nNewton: 6, kappaSplineFloor: 1e-11 });
    console.log(`M=${M}: F32 emu ${a.L2.toFixed(3)}%  (alphaMin=${a.alphaMin.toFixed(3)}, secClampedPts=${a.secClampedPts}, chordShrinkFired=${a.clampFired})   F64 spline(no limiter, 24+6) ${b.L2.toFixed(5)}%`);
  }
  // coordinate-rounding shift of the reference
  const s = C.makeSource(beta, 2.40, 16), sw = C.sweepPoints(1.15, s.tEval, beta, 256);
  let d = 0, n = 0;
  for (const p of sw.pts) { const a = C.exactLW(p.x, p.y, s.tEval, beta), b = C.exactLW(p.x64, p.y64, s.tEval, beta); d += (a.ex - b.ex) ** 2 + (a.ey - b.ey) ** 2; n += a.ex ** 2 + a.ey ** 2; }
  console.log(`reference shift from rounding query coords to f32: ${(100 * Math.sqrt(d / n)).toFixed(2)}%   (paper: 3.23%)`);
  console.log(`sweep half-angle ${sw.half.toExponential(3)} rad, rObs ${sw.rObs.toFixed(4)}, arc ${(2 * sw.half * sw.rObs).toExponential(3)}, point spacing ${(2 * sw.half * sw.rObs / 255).toExponential(3)}`);
  const ux = new Set(sw.pts.map(p => p.x + ',' + p.y)).size;
  console.log(`distinct f32 query points among 256: ${ux}`);
}
