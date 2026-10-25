// Diagnostics on diag.html (copy of app2d with extra kernels; see make_diag.py), headless Chromium + SwiftShader.
//  attrib : accuracy of v6 / v6f / v6f_naive / v6f_dfbis on the pulse sweep (attribution of the floor; identical output
//           of float32 vs double-float bisection)
//  counts : per pixel-source evaluation counts (K df64 knot, F float32 knot, N in-segment) on the sweep (readback path)
//           and on the 32-source default scene (display path and full path)
//  timing : frame time of the rendered view, 32 sources (display path; full path with DIAG_FULL). SwiftShader: RELATIVE ONLY.
// Usage: node diag_test.mjs <section,...> <out.json>
import { launch } from './browser.mjs';
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const sections = (process.argv[2] || 'attrib,counts,timing').split(',');
const outFile = process.argv[3] || join(HERE, 'results_diag.json');
const browser = await launch();
const page = await browser.newPage({ viewport: { width: 640, height: 700 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto('file://' + join(HERE, 'diag.html'), { waitUntil: 'commit' });
await page.waitForFunction(() => typeof state !== 'undefined' && state.sources && state.sources.length > 0, null, { timeout: 300000, polling: 500 });
await page.evaluate('state.paused = true');
await page.waitForTimeout(500);
console.log('load errors:', errors.length ? errors.join('; ') : 'none');
const out = { attrib: [], counts: [], scene: [], timing: [] };

// one synchrotron source on the pulse sweep; returns rel L2 vs exact, E arrays, and (for _cnt kernels) counts
const SWEEP = (kernel, beta, M, mode, obs) => `(() => {
  window.DIAG_FAST = false; window.DIAG_FULL = false;
  state.paused = true; state.kernel = '${kernel}'; selectKernel();
  state.numParticles = 1; state.softEps = 0; state.knownPath = ${mode === 'known'};
  const kBin = kBinFor(${M}); const t0 = 2.40; state.time = t0;
  seedExactRetardedHistory(state.sources[0], 0, 1, t0, ${beta}, 'synchrotron', ${M}, 2);
  for (let k = 0; k < 16; k++) stepSimulation(DT_STEP, ${beta}, 'synchrotron', ${M}, 2);
  for (let k = 0; k < ${mode === 'live' ? 600 : 0}; k++) stepSimulation(0.45 * DT_STEP, ${beta}, 'synchrotron', ${M}, 2);
  if (${mode === 'known'}) { state.sources[0].dragWins = []; populateKnownPathShells(state.sources[0], 0, 1, ${M}, 2, DT_STEP, ${beta}, 'synchrotron'); }
  const sw = getPulseCrossingSweepPoints(1.15, state.time, ${beta}, 256).pts;
  const pts = sw.map((p) => ${obs === 'df64'} ? { x: p.x64, y: p.y64 } : { x: p.x, y: p.y });
  const g = evaluateViaGPUFramebufferReadback(pts, ${M}, kBin, 2);
  let num = 0, den = 0, K = 0, F = 0, N = 0; const ex = [], ey = [];
  pts.forEach((p, i) => { const e = evaluateExactAnalyticalLW(p.x, p.y, state.time, ${beta}, 'synchrotron');
    num += (g[i].ex - e.ex) ** 2 + (g[i].ey - e.ey) ** 2; den += e.mag ** 2; ex.push(g[i].ex); ey.push(g[i].ey);
    const f = Math.floor(g[i].emitX / 4096); F += f; K += g[i].emitX - 4096 * f; N += g[i].emitY; });
  const n = pts.length;
  return { rel: Math.sqrt(num / den), ex, ey, K: K / n, F: F / n, N: N / n };
})()`;

// default scene: 32 sources (swarm), M, known or live; 16x16 grid readback in display (fast) or full path
const SCENE = (kernel, M, known, fast) => `(() => {
  window.DIAG_FULL = false; window.DIAG_FAST = ${fast};
  state.paused = true; state.kernel = '${kernel}'; selectKernel(); state.knownPath = ${known};
  state.numParticles = 32; state.motionMode = 'swarm'; state.beta = 0.8; state.mT = ${M}; state.kBin = kBinFor(${M}); state.softEps = 0.06;
  state.time = 2.5; reseedAllSources();
  const pts = [];
  for (let j = 0; j < 16; j++) for (let i = 0; i < 16; i++) pts.push({ x: -2.0 + 4.0 * (i + 0.5) / 16, y: -1.45 + 2.9 * (j + 0.5) / 16 });
  const g = evaluateViaGPUFramebufferReadback(pts, ${M}, kBinFor(${M}), state.velMode);
  window.DIAG_FAST = false;
  let K = 0, F = 0, N = 0;
  g.forEach((o) => { const f = Math.floor(o.emitX / 4096); F += f; K += o.emitX - 4096 * f; N += o.emitY; });
  const n = pts.length * 32;
  return { K: K / n, F: F / n, N: N / n, e: g.map((o) => [o.ex, o.ey]) };
})()`;

const FRAMES = (kernel, M, known, full, beta) => `(() => {
  window.DIAG_FULL = ${full}; window.DIAG_FAST = false;
  state.paused = true; state.kernel = '${kernel}'; selectKernel(); state.knownPath = ${known};
  state.numParticles = 32; state.motionMode = 'swarm'; state.beta = ${beta}; state.mT = ${M}; state.kBin = kBinFor(${M}); state.softEps = 0.06;
  state.time = 2.5; reseedAllSources();
  const one = new Uint8Array(4);
  renderBulkField(); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, one);
  const ts = [];
  for (let k = 0; k < 3; k++) { const t0 = performance.now(); renderBulkField(); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, one); ts.push(performance.now() - t0); }
  const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  window.DIAG_FULL = false;
  ts.sort((a, b) => a - b);
  return { ms: ts[1], w, h, px: Array.from(px) };
})()`;

const gam = (b) => (1 / Math.sqrt(1 - b * b)).toFixed(1);
if (sections.includes('attrib')) {
  const K = ['v6', 'v6f', 'v6f_naive', 'v6f_dfbis'];
  for (const [beta, obsList] of [[0.9999, ['f32', 'df64']], [0.99999, ['df64']]]) for (const M of [512, 2048]) for (const mode of ['known', 'ring', 'live']) for (const obs of obsList) {
    const r = {};
    for (const k of K) r[k] = await page.evaluate(SWEEP(k, beta, M, mode, obs));
    let dmax = 0, emax = 0;
    r.v6f.ex.forEach((v, i) => { dmax = Math.max(dmax, Math.hypot(v - r.v6f_dfbis.ex[i], r.v6f.ey[i] - r.v6f_dfbis.ey[i])); emax = Math.max(emax, Math.hypot(v, r.v6f.ey[i])); });
    const row = { beta, M, mode, obs, rel: Object.fromEntries(K.map((k) => [k, r[k].rel])), bisectMaxDiff: dmax / emax };
    out.attrib.push(row);
    console.log(`attrib g${gam(beta)} M${M} ${mode} ${obs}: ` + K.map((k) => `${k} ${r[k].rel.toExponential(2)}`).join('  ') + `  | v6f vs v6f_dfbis max diff ${row.bisectMaxDiff.toExponential(1)}`);
  }
}
if (sections.includes('counts')) {
  for (const [beta, M, mode] of [[0.9999, 512, 'ring'], [0.9999, 2048, 'ring'], [0.9999, 512, 'known'], [0.99999, 2048, 'known'], [0.99999, 2048, 'live']]) {
    const a = await page.evaluate(SWEEP('v6_cnt', beta, M, mode, 'df64')), b = await page.evaluate(SWEEP('v6f_cnt', beta, M, mode, 'df64'));
    const row = { beta, M, mode, v6: { K: a.K, F: a.F, N: a.N }, v6f: { K: b.K, F: b.F, N: b.N } };
    out.counts.push(row);
    console.log(`counts sweep g${gam(beta)} M${M} ${mode}: v6 K ${a.K.toFixed(2)} F ${a.F.toFixed(2)} N ${a.N.toFixed(2)} | v6f K ${b.K.toFixed(2)} F ${b.F.toFixed(2)} N ${b.N.toFixed(2)}`);
  }
  for (const M of [128, 512, 2048]) for (const known of [true, false]) for (const fast of [true, false]) {
    const a = await page.evaluate(SCENE('v6_cnt', M, known, fast)), b = await page.evaluate(SCENE('v6f_cnt', M, known, fast));
    let num = 0, den = 0; a.e.forEach((v, i) => { num += (v[0] - b.e[i][0]) ** 2 + (v[1] - b.e[i][1]) ** 2; den += v[0] ** 2 + v[1] ** 2; });
    const row = { M, known, path: fast ? 'display' : 'full', v6: { K: a.K, F: a.F, N: a.N }, v6f: { K: b.K, F: b.F, N: b.N }, relDiff: Math.sqrt(num / den) };
    out.scene.push(row);
    console.log(`counts scene32 M${M} ${known ? 'known' : 'live'} ${row.path}: v6 K ${a.K.toFixed(2)} F ${a.F.toFixed(2)} N ${a.N.toFixed(2)} | v6f K ${b.K.toFixed(2)} F ${b.F.toFixed(2)} N ${b.N.toFixed(2)} | field v6f vs v6 ${row.relDiff.toExponential(1)}`);
  }
}
if (sections.includes('timing')) {
  const scenes = [[128, true, 0.8], [128, false, 0.8], [512, true, 0.8], [512, false, 0.8], [2048, false, 0.8], [512, false, 0.999]];
  for (const full of [false, true]) for (const [M, known, beta] of scenes) {
    const ms = { v5: [], v6: [], v6f: [] }; let pxA = null, pxB = null;
    for (let round = 0; round < 3; round++) for (const k of ['v5', 'v6', 'v6f']) {
      if (full && k === 'v5') continue;
      const r = await page.evaluate(FRAMES(k, M, known, full, beta)); ms[k].push(r.ms);
      if (round === 0 && k === 'v6') pxA = r.px; if (round === 0 && k === 'v6f') pxB = r.px;
    }
    let dmax = 0, ndiff = 0; for (let i = 0; i < pxA.length; i++) { const d = Math.abs(pxA[i] - pxB[i]); if (d) ndiff++; dmax = Math.max(dmax, d); }
    const med = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];
    const row = { M, known, beta, path: full ? 'full' : 'display', v5: ms.v5.length ? med(ms.v5) : null, v6: med(ms.v6), v6f: med(ms.v6f), pxMaxDiff: dmax, pxDiffFrac: ndiff / pxA.length };
    out.timing.push(row);
    console.log(`timing ${row.path} M${M} ${known ? 'known' : 'live'} beta ${beta}: ` + (row.v5 ? `v5 ${row.v5.toFixed(0)} ms, ` : '') + `v6 ${row.v6.toFixed(0)} ms, v6f ${row.v6f.toFixed(0)} ms (v6f/v6 ${(row.v6f / row.v6).toFixed(2)}); pixels differ: max ${dmax}/255 on ${(100 * row.pxDiffFrac).toFixed(2)}% of bytes`);
  }
}
console.log('errors:', errors.length ? errors.join('; ').slice(0, 2000) : 'none');
out.errors = errors;
writeFileSync(outFile, JSON.stringify(out));
await browser.close();
