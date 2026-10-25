// Known-path mode in the app (web/index.html, v6 kernel): accuracy against the exact field and relative cost.
// One synchrotron source (circular orbit r = 0.85, c = 2.35), no softening, float32 GPU readback (SwiftShader).
//   sweep  : 256 observers crossing the pulse at retarded radius 1.15 (the preprint's sweep)
//   view   : 24x24 grid over the default view (|x| <= 1.45 aspect, |y| <= 1.45), points within 0.05 of the orbit excluded
//   far    : 512 observers at 6 < R < 20 from the orbit centre (random angles, fixed seed)
// Ring = the v6 production path (ring buffer resampled onto shells, cubic segments).
// Known = exact knots of the analytic path + C2 quintic with the degree-8 certificate.
// Usage, from the repository root: node validation/knownpath/kp_measure.mjs [acc|cost|cpu|handoff|all]
// Writes validation/knownpath/results_<which>.json; results/knownpath.json holds the recorded acc, cost and handoff
// results (v6 kernel, before v6f). Software rendering: costs relative only.
import { launch } from '../../tests/gpu/browser.mjs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
const HERE = dirname(fileURLToPath(import.meta.url));
const APP = 'file://' + join(HERE, '..', '..', 'web', 'index.html');
import { writeFileSync } from 'node:fs';
const which = process.argv[2] || 'all';
const out = {};
const browser = await launch();
const page = await browser.newPage({ viewport: { width: 640, height: 700 } });
page.on('pageerror', (e) => console.error('pageerror', e.message));
page.setDefaultTimeout(0);
await page.goto(APP, { waitUntil: 'commit' });
await page.waitForFunction(() => typeof state !== 'undefined' && state.sources && state.sources.length > 0, null, { timeout: 300000, polling: 500 });
await page.evaluate('state.paused = true');

if (which === 'acc' || which === 'all') {
  out.acc = await page.evaluate(() => {
    const res = [];
    let seed = 12345; const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    function setup(beta, M, known) {
      state.kernel = 'v6'; selectKernel();
      state.numParticles = 1; state.softEps = 0; state.knownPath = false;
      const t0 = 2.40; state.time = t0;
      seedExactRetardedHistory(state.sources[0], 0, 1, t0, beta, 'synchrotron', M, 2);
      for (let k = 0; k < 16; k++) stepSimulation(DT_STEP, beta, 'synchrotron', M, 2);
      if (known) { state.sources[0].dragWins = []; populateKnownPathShells(state.sources[0], 0, 1, M, 2, DT_STEP, beta, 'synchrotron'); }
    }
    function err(beta, M, known, pts, df64) {
      setup(beta, M, known);
      const kb = Math.round(Math.log2(M)) + 2;
      const q = pts.map((p) => df64 ? { x: p.x64, y: p.y64 } : { x: p.x, y: p.y });
      const g = [];
      for (let i0 = 0; i0 < q.length; i0 += 256) g.push(...evaluateViaGPUFramebufferReadback(q.slice(i0, i0 + 256), M, kb, 2));
      let num = 0, den = 0;
      q.forEach((p, i) => {
        const e = evaluateExactAnalyticalLW(p.x, p.y, state.time, beta, 'synchrotron');
        num += (g[i].ex - e.ex) ** 2 + (g[i].ey - e.ey) ** 2; den += e.mag ** 2;
      });
      return 100 * Math.sqrt(num / den);
    }
    for (const beta of [0.99, 0.999, 0.9999, 0.99999]) {
      const gam = 1 / Math.sqrt((1 - beta) * (1 + beta));
      state.time = 2.40 + 16 * DT_STEP;
      const sweep = getPulseCrossingSweepPoints(1.15, state.time, beta, 256).pts;
      const aspect = 606 / 379;
      const view = [];
      for (let j = 0; j < 24; j++) for (let i = 0; i < 24; i++) {
        const x = (-1 + 2 * (i + 0.5) / 24) * 1.45 * aspect, y = (-1 + 2 * (j + 0.5) / 24) * 1.45;
        if (Math.abs(Math.hypot(x, y) - 0.85) > 0.05) view.push({ x: Math.fround(x), y: Math.fround(y), x64: x, y64: y });
      }
      const far = [];
      for (let k = 0; k < 512; k++) { const r = 6 + 14 * rnd(), a = 2 * Math.PI * rnd(); far.push({ x: Math.fround(r * Math.cos(a)), y: Math.fround(r * Math.sin(a)), x64: r * Math.cos(a), y64: r * Math.sin(a) }); }
      const df = beta > 0.9999;   // above gamma ~ 100 float32 observers cannot resolve the pulse
      for (const M of [32, 64, 128, 256, 512, 1024, 2048]) {
        const row = { beta, gamma: +gam.toFixed(1), M, df64: df };
        row.knownSweep = err(beta, M, true, sweep, df);
        row.knownView = err(beta, M, true, view, false);
        row.knownFar = err(beta, M, true, far, false);
        if ([128, 512, 2048].includes(M)) {
          row.ringSweep = err(beta, M, false, sweep, df);
          row.ringView = err(beta, M, false, view, false);
          row.ringFar = err(beta, M, false, far, false);
        }
        res.push(row);
      }
    }
    return res;
  });
  console.log('beta   gamma  M     | known sweep / view / far (%)      | ring sweep / view / far (%)');
  for (const r of out.acc) console.log(`${r.beta} ${String(r.gamma).padEnd(6)} ${String(r.M).padEnd(5)} | ${r.knownSweep.toExponential(2)} ${r.knownView.toExponential(2)} ${r.knownFar.toExponential(2)} | ${r.ringSweep !== undefined ? r.ringSweep.toExponential(2) + ' ' + r.ringView.toExponential(2) + ' ' + r.ringFar.toExponential(2) : ''}`);
}

if (which === 'cost' || which === 'cpu' || which === 'all') {
  if (which === 'cpu') await page.evaluate('window.__kpQuick = true');
  out.cost = await page.evaluate(() => {
    // default scene: 32 ensemble sources, M = 128; frame time of the rendered view (software rendering: relative only)
    const one = new Uint8Array(4);
    function frames(kernel, known, M) {
      state.kernel = kernel; selectKernel(); state.knownPath = known; state.numParticles = 32; state.motionMode = 'swarm';
      state.mT = M; state.kBin = Math.round(Math.log2(M)) + 2; state.time = 2.5; state.softEps = 0.06;
      reseedAllSources();
      renderBulkField(); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, one);
      const t0 = performance.now();
      for (let k = 0; k < 4; k++) { renderBulkField(); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, one); }
      return (performance.now() - t0) / 4;
    }
    function populate(known, M) {
      state.kernel = 'v6'; selectKernel(); state.knownPath = known; state.numParticles = 32; state.motionMode = 'swarm';
      state.mT = M; state.time = 2.5; reseedAllSources();
      const t0 = performance.now();
      for (let k = 0; k < 20; k++) stepSimulation(0.45 * DT_STEP);
      return (performance.now() - t0) / 20;
    }
    const r = {};
    for (const rep of (window.__kpQuick ? [] : [0, 1])) {
      for (const [k, kn, M] of [['v5', false, 128], ['v6', false, 128], ['v6', true, 128], ['v6', true, 64], ['v6', false, 512], ['v6', true, 512], ['v6', false, 2048]]) {
        const key = `${k}_${kn ? 'known' : 'ring'}_M${M}`;
        (r[key] = r[key] || []).push(frames(k, kn, M));
      }
    }
    for (const key of Object.keys(r)) r[key] = Math.min(...r[key]);
    r.cpuStep_ring_M128 = populate(false, 128); r.cpuStep_known_M128 = populate(true, 128);
    r.cpuStep_ring_M512 = populate(false, 512); r.cpuStep_known_M512 = populate(true, 512);
    r.cpuStep_ring_M2048 = populate(false, 2048); r.cpuStep_known_M2048 = populate(true, 2048);
    return r;
  });
  console.log(JSON.stringify(out.cost, null, 1));
}

if (which === 'handoff' || which === 'all') {
  // Drag source 0 (orbit preset, beta 0.8, M = 512) along a jerky path, release, let it rejoin; then compare the field of
  // known-path mode with the same run in live (ring) mode, and the frame-to-frame change across the hand-back.
  out.handoff = await page.evaluate(() => {
    const pts = [];
    for (let j = 0; j < 12; j++) for (let i = 0; i < 12; i++) pts.push({ x: -2.2 + 4.4 * (i + 0.5) / 12, y: -1.4 + 2.8 * (j + 0.5) / 12 });
    function run(known) {
      state.kernel = 'v6'; selectKernel(); state.knownPath = known; state.numParticles = 4; state.motionMode = 'synchrotron';
      state.beta = 0.8; state.mT = 512; state.kBin = 11; state.time = 2.5; state.softEps = 0.02;
      reseedAllSources();
      const dt = 0.45 * DT_STEP;
      // warm-up: refill the ring at the live step (the reseed writes it at DT_STEP spacing)
      for (let k = 0; k < 800; k++) stepSimulation(dt);
      const snaps = [];
      let phase = 'drag', k = 0, rejoinStep = -1;
      state.interactionTarget = 'source0'; state.sources[0].returningToOrbit = true; state.sources[0]._drag = null;
      while (k < 9000) {
        if (phase === 'drag') {
          const u = k * dt;
          state.draggedSourceTarget = { x: 0.6 * Math.sin(3.1 * u) + (k > 200 ? 0.4 : 0), y: 0.5 * Math.cos(2.3 * u) };
          if (k === 500) { phase = 'return'; state.interactionTarget = null; }
        } else if (phase === 'return' && !state.sources[0].returningToOrbit) { phase = 'rejoined'; rejoinStep = k; }
        stepSimulation(dt);
        const snapNow = k % 250 === 0 || (rejoinStep >= 0 && k - rejoinStep <= 3);
        if (snapNow) {
          const g = evaluateViaGPUFramebufferReadback(pts, 512, 11, 2);
          snaps.push({ k, phase, e: g.map((o) => [o.ex, o.ey]), q: g.map((o) => o.quality), nExactSrc0: Array.from(state.sources[0].emitExact.slice(0, 512)).reduce((a, b) => a + b, 0) });
        }
        if (rejoinStep >= 0 && k - rejoinStep > 3 && k > rejoinStep + 1500) break;
        k++;
      }
      return { snaps, rejoinStep };
    }
    const A = run(true), B = run(false);
    const rows = [];
    for (let i = 0; i < A.snaps.length && i < B.snaps.length; i++) {
      const a = A.snaps[i], b = B.snaps[i];
      let num = 0, den = 0, mx = 0;
      a.e.forEach((v, j) => { const d = Math.hypot(v[0] - b.e[j][0], v[1] - b.e[j][1]), m = Math.hypot(b.e[j][0], b.e[j][1]); num += d * d; den += m * m; mx = Math.max(mx, d / Math.max(m, 1e-12)); });
      // frame-to-frame change of the known-path field (only meaningful for consecutive snapshots around the rejoin)
      let dn = 0, dd = 0;
      if (i > 0 && a.k - A.snaps[i - 1].k === 1) a.e.forEach((v, j) => { const p = A.snaps[i - 1].e[j]; dn += (v[0] - p[0]) ** 2 + (v[1] - p[1]) ** 2; dd += v[0] ** 2 + v[1] ** 2; });
      rows.push({ k: a.k, phase: a.phase, knownVsRingRel: Math.sqrt(num / den), knownVsRingMax: mx, exactKnotsSrc0: a.nExactSrc0, stepChange: dd ? Math.sqrt(dn / dd) : null });
    }
    return { rejoinA: A.rejoinStep, rejoinB: B.rejoinStep, rows, dragWinsAfter: state.sources[0].dragWins.map((w) => w.slice()) };
  });
  for (const r of out.handoff.rows) console.log(r.k, r.phase, 'known vs ring rel', r.knownVsRingRel.toExponential(2), 'max', r.knownVsRingMax.toExponential(2), 'exact knots src0', r.exactKnotsSrc0, r.stepChange !== null ? 'step change ' + r.stepChange.toExponential(2) : '');
  console.log('rejoin step known/ring:', out.handoff.rejoinA, out.handoff.rejoinB);
}

writeFileSync(join(HERE, `results_${which}.json`), JSON.stringify(out, null, 1));
await browser.close();
