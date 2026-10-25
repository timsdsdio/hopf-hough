// Accuracy of the 2D app's kernels vs the exact float64 LW field (evaluateExactAnalyticalLW), headless Chromium + SwiftShader.
// Same protocol as hopf-hough tests/gpu/web_app.mjs SWEEP: one synchrotron source, softEps = 0, 256 observers crossing the
// pulse at R_ret = 1.15. Modes: known (exact knots, C2 quintic), ring (seeded ring, cubic Hermite + limiter),
// live (ring after 600 live steps of 0.45 DT_STEP). Observers: f32 (rounded, compared at the rounded point) or df64.
// Usage: node measure.mjs <app.html> <out.json> <kernels comma-separated> [--quick]
import { launch } from './browser.mjs';
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const [app, outFile, kernArg] = process.argv.slice(2);
const kernels = (kernArg || 'v6').split(',');
const quick = process.argv.includes('--quick');
const browser = await launch();
const page = await browser.newPage({ viewport: { width: 640, height: 700 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto('file://' + resolve(app), { waitUntil: 'commit' });
await page.waitForFunction(() => typeof state !== 'undefined' && state.sources && state.sources.length > 0, null, { timeout: 300000, polling: 500 });
await page.evaluate('state.paused = true');
await page.waitForTimeout(500);

const SWEEP = (kernel, beta, M, mode, obs) => `(() => {
  state.paused = true; state.kernel = '${kernel}'; selectKernel();
  state.numParticles = 1; state.softEps = 0; state.knownPath = ${mode === 'known'};
  const kBin = kBinFor(${M});
  const t0 = 2.40; state.time = t0;
  seedExactRetardedHistory(state.sources[0], 0, 1, t0, ${beta}, 'synchrotron', ${M}, 2);
  for (let k = 0; k < 16; k++) stepSimulation(DT_STEP, ${beta}, 'synchrotron', ${M}, 2);
  for (let k = 0; k < ${mode === 'live' ? 600 : 0}; k++) stepSimulation(0.45 * DT_STEP, ${beta}, 'synchrotron', ${M}, 2);
  if (${mode === 'known'}) { state.sources[0].dragWins = []; populateKnownPathShells(state.sources[0], 0, 1, ${M}, 2, DT_STEP, ${beta}, 'synchrotron'); }
  const sw = getPulseCrossingSweepPoints(1.15, state.time, ${beta}, 256).pts;
  const pts = sw.map((p) => ${obs === 'df64'} ? { x: p.x64, y: p.y64 } : { x: p.x, y: p.y });
  const g = evaluateViaGPUFramebufferReadback(pts, ${M}, kBin, 2);
  let num = 0, den = 0, iMax = 0, eMax = 0; const ex = [], ey = [], rx = [], ry = [];
  pts.forEach((p, i) => { const e = evaluateExactAnalyticalLW(p.x, p.y, state.time, ${beta}, 'synchrotron');
    num += (g[i].ex - e.ex) ** 2 + (g[i].ey - e.ey) ** 2; den += e.mag ** 2; if (e.mag > eMax) { eMax = e.mag; iMax = i; }
    ex.push(g[i].ex); ey.push(g[i].ey); rx.push(e.ex); ry.push(e.ey); });
  const crest = Math.hypot(ex[iMax] - rx[iMax], ey[iMax] - ry[iMax]) / eMax;
  return { rel: Math.sqrt(num / den), crest, ex, ey, rx, ry };
})()`;

const cases = [];
const betas = quick ? [0.9999] : [0.99, 0.999, 0.9999, 0.99999];
for (const beta of betas) for (const M of quick ? [512] : [128, 512, 2048]) for (const mode of ['known', 'ring', 'live']) for (const obs of ['f32', 'df64']) {
  if (beta === 0.99999 && obs === 'f32') continue;   // float32 observers cannot resolve the pulse at gamma = 224 (paper table)
  if (mode === 'live' && M === 128) continue;
  cases.push({ beta, M, mode, obs });
}
const out = { errors, rows: [] };
for (const c of cases) {
  const row = { ...c, res: {} };
  for (const k of kernels) row.res[k] = await page.evaluate(SWEEP(k, c.beta, c.M, c.mode, c.obs));
  out.rows.push(row);
  const gam = (1 / Math.sqrt(1 - c.beta * c.beta)).toFixed(1);
  console.log(`beta ${c.beta} (gamma ${gam}) M ${c.M} ${c.mode} ${c.obs}: ` + kernels.map((k) => `${k} ${row.res[k].rel.toExponential(2)} (crest ${row.res[k].crest.toExponential(1)})`).join('  '));
}
console.log('errors:', errors.length ? errors.join('; ').slice(0, 2000) : 'none');
writeFileSync(outFile, JSON.stringify(out));
await browser.close();
