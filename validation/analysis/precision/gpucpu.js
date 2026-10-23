// Does a legal reordering of a few f32 ops reproduce the paper's 6-9% "GPU vs CPU-Float32" discrepancy?
'use strict';
const C = require('./core.js'); const { F32, S32, KBIN } = require('./run.js');
for (const beta of [0.999, 0.9999]) for (const M of [512, 1024, 2048]) for (const fixed of [false, true]) {
  const src = C.makeSource(beta, 2.40, 16), K = C.populate(src, M, S32); K.tEval = src.tEval; K.beta = beta;
  const sw = C.sweepPoints(1.15, src.tEval, beta, 256);
  const F = Object.assign({}, F32, fixed ? { splitStore: true } : {});
  const base = { kBin: KBIN[M], nNewton: 2, limiter: true, kappaSplineFloor: 5e-5, df64root: fixed };
  let d = 0, n = 0;
  for (const p of sw.pts) {
    const a = C.evalF32(K, p.x, p.y, F, base), b = C.evalF32(K, p.x, p.y, F, Object.assign({}, base, { altOrder: true }));
    const e = C.exactLW(p.x, p.y, src.tEval, beta);
    d += (a.ex - b.ex) ** 2 + (a.ey - b.ey) ** 2; n += e.ex ** 2 + e.ey ** 2;
  }
  const g = 1 / Math.sqrt((1 - beta) * (1 + beta));
  console.log(`gamma ${g.toFixed(1)} M=${M} ${fixed ? 'df64-fix ' : 'published'}: discrepancy between two legal f32 op orderings = ${(100 * Math.sqrt(d / n)).toFixed(3)}%`);
}
