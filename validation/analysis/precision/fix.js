// Fix: double-float (df64) on the root-function critical path only + hi/lo knot-position storage.
// Everything else (spline, velocity, kernel, kappa) stays Float32.
'use strict';
const C = require('./core.js'); const { f32, id } = C;
const { F32, S32, KBIN } = require('./run.js');

function runFix(beta, M, variant, queryMode, betaMax) {
  C.state.betaMax = betaMax;
  const src = C.makeSource(beta, 2.40, 16);
  const K = C.populate(src, M, variant.store64 ? { RS: id, XS: id, US: id, AS: id } : S32); K.tEval = src.tEval; K.beta = beta;
  const kfloor = Math.min(5e-5, 1 - betaMax);
  const opt = Object.assign({ kBin: KBIN[M] || 11, nNewton: 2, limiter: true, kappaSplineFloor: kfloor, safeguard: false }, variant.opt);
  const sw = C.sweepPoints(1.15, src.tEval, beta, 256);
  let eSq = 0, nSq = 0; const rows = [];
  for (const p of sw.pts) {
    let F = Object.assign({}, F32, variant.F || {}), qx, qy, rx, ry;
    if (queryMode === 'f32') { qx = p.x; qy = p.y; rx = p.x; ry = p.y; }
    else { // df64 query: hi = f32(x64), lo = f32(x64 - hi); reference at x64 (unrounded)
      qx = p.x; qy = p.y; F.qlo = [f32(p.x64 - p.x), f32(p.y64 - p.y)]; rx = p.x64; ry = p.y64;
    }
    const r = C.evalF32(K, qx, qy, F, opt), ex = C.exactLW(rx, ry, src.tEval, beta);
    eSq += (r.ex - ex.ex) ** 2 + (r.ey - ex.ey) ** 2; nSq += ex.ex ** 2 + ex.ey ** 2;
    rows.push({ a: p.a, ex, r });
  }
  C.state.betaMax = 0.99995;
  return { L2: 100 * Math.sqrt(eSq / nSq), rows, sw, distinct: new Set(sw.pts.map(p => p.x + ',' + p.y)).size };
}
module.exports = { runFix };

if (require.main === module) {
  const variants = {
    'baseline F32 (solver as published)': { opt: {} },
    'df64 root path, f32 knot storage': { opt: { df64root: true } },
    'df64 root path + hi/lo knot storage': { opt: { df64root: true }, F: { splitStore: true } },
    'df64 root + hi/lo, 6 safeguarded Newton': { opt: { df64root: true, nNewton: 6, safeguard: true }, F: { splitStore: true } },
    'f32 bisect, widen 0.5, 4 df64 bisect + 2 Newton': { opt: { df64root: true, df64NewtonOnly: true, widen: 0.5, df64Bis: 4 }, F: { splitStore: true } },
    'all-F64 spline (model floor, limiter on)': { opt: { kBin: 24, nNewton: 6 }, F: { Q: (x) => x, P: (x) => x, V: (x) => x, A: (x) => x, Rt: (x) => x, Kr: (x) => x }, store64: true },
  };
  const cases = [[0.999, 0.99995], [0.9999, 0.99995], [0.9999875, 0.9999995]];
  for (const [beta, bmax] of cases) {
    const g = 1 / Math.sqrt((1 - beta) * (1 + beta));
    for (const M of [512, 2048]) {
      for (const qm of ['f32', 'df64']) {
        console.log(`\n## gamma=${g.toFixed(1)} M=${M} query=${qm} betaMax=${bmax}`);
        for (const [name, v] of Object.entries(variants)) {
          const r = runFix(beta, M, v, qm, bmax);
          console.log(`  ${name.padEnd(48)} ${r.L2.toFixed(4).padStart(9)}%   (distinct f32 pts ${r.distinct})`);
        }
      }
    }
  }
}
