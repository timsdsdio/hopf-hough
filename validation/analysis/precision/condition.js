// Well-posedness: sensitivity of E to the query coordinate at the crest; fair metrics; gamma scaling.
'use strict';
const C = require('./core.js'); const { f32, id } = C;
const { runFix } = require('./fix.js');

function ulp32(x) { const a = Math.abs(f32(x)); if (a === 0) return 1.4e-45; const e = Math.floor(Math.log2(a)); return Math.pow(2, e - 23); }

function sensitivity(beta) {
  const src = C.makeSource(beta, 2.40, 16), sw = C.sweepPoints(1.15, src.tEval, beta, 256);
  const t = src.tEval; let Erms = 0, gradRms = 0, gradMax = 0, condMax = 0, perUlpMax = 0, mc = 0;
  const E = [];
  for (const p of sw.pts) {
    const e0 = C.exactLW(p.x64, p.y64, t, beta); E.push(Math.hypot(e0.ex, e0.ey)); Erms += e0.ex ** 2 + e0.ey ** 2;
  }
  Erms = Math.sqrt(Erms / sw.pts.length);
  const h = 2e-10;
  for (const p of sw.pts) {
    const ex1 = C.exactLW(p.x64 + h, p.y64, t, beta), ex0 = C.exactLW(p.x64 - h, p.y64, t, beta);
    const ey1 = C.exactLW(p.x64, p.y64 + h, t, beta), ey0 = C.exactLW(p.x64, p.y64 - h, t, beta);
    // Frobenius norm of dE/dx (2x2)
    const J = [(ex1.ex - ex0.ex) / (2 * h), (ex1.ey - ex0.ey) / (2 * h), (ey1.ex - ey0.ex) / (2 * h), (ey1.ey - ey0.ey) / (2 * h)];
    const g = Math.hypot(...J); gradRms += g * g; gradMax = Math.max(gradMax, g);
    const xn = Math.hypot(p.x64, p.y64);
    condMax = Math.max(condMax, g * xn / Erms);
    perUlpMax = Math.max(perUlpMax, g * ulp32(Math.max(Math.abs(p.x64), Math.abs(p.y64))) / Erms);
    // expected error from uniform +-ulp/2 per component: E|J d|^2 = sum J^2 * ulp^2/12
    mc += (J[0] ** 2 + J[1] ** 2) * ulp32(p.x64) ** 2 / 12 + (J[2] ** 2 + J[3] ** 2) * ulp32(p.y64) ** 2 / 12;
  }
  gradRms = Math.sqrt(gradRms / sw.pts.length);
  // FWHM of |E| along the sweep (arc length)
  const Emax = Math.max(...E), ds = 2 * sw.half * sw.rObs / 255; let n = 0; for (const e of E) if (e > Emax / 2) n++;
  return { gamma: 1 / Math.sqrt((1 - beta) * (1 + beta)), Erms, gradRms, gradMax, condMax, perUlpMax, predRound: 100 * Math.sqrt(mc / sw.pts.length) / Erms, fwhm: n * ds, x: sw.pts[128].x64, y: sw.pts[128].y64 };
}

// fair metrics on a result set (rows have a = angle, ex = ref at evaluated point, r = solver)
function metrics(rows, beta, tEval, rObs) {
  const n = rows.length; let e2 = 0, r2 = 0, fl = 0, flr = 0;
  for (const w of rows) { e2 += (w.r.ex - w.ex.ex) ** 2 + (w.r.ey - w.ex.ey) ** 2; r2 += w.ex.ex ** 2 + w.ex.ey ** 2; fl += w.r.ex ** 2 + w.r.ey ** 2; }
  flr = r2;
  // phase-aligned: best angular shift (search over +-0.05 sweep half-width) of a dense f64 reference
  const a0 = rows[0].a, a1 = rows[n - 1].a, span = a1 - a0;
  let best = Infinity, bestS = 0;
  for (let k = -60; k <= 60; k++) {
    const sh = k * span / 255 / 20; let s = 0;
    for (const w of rows) { const px = f32(rObs * Math.cos(w.a)), py = f32(rObs * Math.sin(w.a)); const x = px * Math.cos(sh) - py * Math.sin(sh), y = px * Math.sin(sh) + py * Math.cos(sh); const e = C.exactLW(x, y, tEval, beta); s += (w.r.ex - e.ex) ** 2 + (w.r.ey - e.ey) ** 2; }
    if (s < best) { best = s; bestS = sh; }
  }
  return { L2: 100 * Math.sqrt(e2 / r2), fluence: 100 * Math.abs(fl - flr) / flr, aligned: 100 * Math.sqrt(best / r2), shiftArc: bestS * rObs };
}
module.exports = { sensitivity, metrics };

if (require.main === module) {
  console.log('## Sensitivity of E(x) at the crest sweep (Float64 reference)');
  console.log('gamma | FWHM(arc) | FWHM/ulp | max |dE/dx||x|/E_rms | max rel. change per 1 ulp | predicted L2 shift from f32 query rounding');
  for (const b of [0.99, 0.999, 0.9995, 0.9999, 0.9999875]) {
    const s = sensitivity(b); const u = ulp32(Math.max(Math.abs(s.x), Math.abs(s.y)));
    console.log(`${s.gamma.toFixed(1).padStart(6)} | ${s.fwhm.toExponential(2)} | ${(s.fwhm / u).toFixed(1).padStart(7)} | ${s.condMax.toExponential(2)} | ${(100 * s.perUlpMax).toFixed(3)}% | ${s.predRound.toFixed(3)}%`);
  }
  console.log('\n## gamma scaling of the F32 floor: published solver vs df64-critical-path fix (M=512, f32 query, paper metric)');
  const base = { opt: {} }, fix = { opt: { df64root: true }, F: { splitStore: true } };
  for (const b of [0.99, 0.995, 0.999, 0.9995, 0.9998, 0.9999, 0.99995, 0.99998, 0.9999875]) {
    const g = 1 / Math.sqrt((1 - b) * (1 + b)); const bm = b < 0.99993 ? 0.99995 : 0.9999995;
    const r0 = runFix(b, 512, base, 'f32', bm), r1 = runFix(b, 512, fix, 'f32', bm), r2 = runFix(b, 2048, fix, 'f32', bm);
    console.log(`gamma ${g.toFixed(1).padStart(6)} (betaMax ${bm}): published ${r0.L2.toFixed(3).padStart(8)}%   fix M=512 ${r1.L2.toFixed(4).padStart(8)}%   fix M=2048 ${r2.L2.toFixed(4).padStart(8)}%`);
  }
  console.log('\n## Fair metrics (gamma=70.7): published vs fix');
  for (const M of [512, 2048]) for (const [nm, v] of [['published', base], ['fix', fix]]) {
    const r = runFix(0.9999, M, v, 'f32', 0.99995);
    const src = C.makeSource(0.9999, 2.40, 16);
    const m = metrics(r.rows, 0.9999, src.tEval, r.sw.rObs);
    console.log(`M=${M} ${nm.padEnd(10)} L2@represented ${m.L2.toFixed(3)}%  phase-aligned ${m.aligned.toFixed(3)}% (best shift ${m.shiftArc.toExponential(2)})  fluence ${m.fluence.toFixed(3)}%`);
  }
}
