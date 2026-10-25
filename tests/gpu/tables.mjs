// Regenerates every table in the paper with the production solver (WebGL2 readback).
// Usage: node tests/gpu/tables.mjs [which=all] [kernel=v6]. results/tables.json (preprint v6.1, README, docs/ACCURACY.md
// sections 1-5) is the v6 kernel; pass v6f for the default kernel of the v6.2 library. The v6/v6f comparison of
// Table 2 is tests/gpu/table2_v6_2.mjs.
import { launch } from './browser.mjs';
import { readFileSync, writeFileSync } from 'node:fs';
const root = new URL('../../', import.meta.url).pathname;
const which = (process.argv[2] || 'all').split(',');
const kernel = process.argv[3] || 'v6';
if (!['v6', 'v6f'].includes(kernel)) throw new Error('kernel must be v6 or v6f');
const browser = await launch();
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('pageerror', e.message));
page.on('console', (m) => console.log('[page]', m.text()));
await page.setContent('<html><body><canvas id="c" width="64" height="64"></canvas><canvas id="l" width="64" height="64"></canvas><canvas id="r" width="640" height="400"></canvas></body></html>');
await page.addScriptTag({ content: readFileSync(root + 'dist/hopf-hough.iife.js', 'utf8') });
page.setDefaultTimeout(0);
const res = await page.evaluate(async ([which, kernel]) => {
  const H = HopfHough;
  const solver = new H.GpuSolver(document.getElementById('c'), H.kernelShaders(kernel));
  const B = H.gpuBackend(solver);
  const gl = solver.gl, ext = gl.getExtension('WEBGL_debug_renderer_info');
  const out = { kernel, renderer: String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)) };
  const prod = { ...H.DEFAULT_CONFIG, softEps: 0, prefilter: false };
  const legacy = { ...H.DEFAULT_CONFIG, newton: 2, certMode: 0, rootMode: 0, precision: 0, betaMax: 0.99995, softEps: 0, prefilter: false };
  const kb = { 64: 8, 128: 9, 256: 10, 512: 11, 1024: 12, 2048: 13 };
  const has = (k) => which.includes('all') || which.includes(k);

  // The version-5 algorithm runs on the deposited, regression-tested shader (LEGACY_SHADERS).
  const legacySolver = new H.GpuSolver(document.getElementById('l'), H.LEGACY_SHADERS);
  const LB = H.gpuBackend(legacySolver);
  const legacyBase = { ...H.DEFAULT_CONFIG, ...H.LEGACY_SETTINGS, softEps: 0, prefilter: false };
  if (has('gamma')) {
    out.gamma = [];
    for (const beta of [0.8, 0.95, 0.99, 0.999, 0.9999, 0.99999]) for (const M of [128, 512, 2048]) {
      const spec = { beta, motion: 'synchrotron', mT: M, kBin: kb[M], velMode: 2, nShock: 256 };
      const p = H.runCase(spec, B, { base: prod });
      const pd = H.runCase(spec, B, { base: prod, queryMode: 'df64' });
      const l = beta <= 0.9999 ? H.runCase(spec, LB, { base: legacyBase }) : null;
      out.gamma.push({ beta, gamma: 1 / Math.sqrt(1 - beta * beta), M, prodSweep: p.shockL2, prodSweepDf64: pd.shockL2, prodRing: p.bulkL2, legacySweep: l && l.shockL2, legacyRing: l && l.bulkL2, coordRound: p.coordRoundL2 });
    }
  }
  if (has('grids')) {
    out.grids = [];
    for (const g of ['compactified', 'log', 'power2', 'uniform']) {
      const r = H.runCase({ beta: 0.99, motion: 'synchrotron', mT: 128, kBin: 9, velMode: 2, nShock: 256, rMax: 220, rEval: 1.15, gridMode: g }, B, { base: prod });
      const cfg = { ...prod, mT: 128, gridMode: g, gridRMax: 220 };
      out.grids.push({ grid: g, sweep: r.shockL2, ring: r.bulkL2, spacingAt115: (() => { let best = 1e9, dR = 0; for (let m = 0; m < 127; m++) { const a = H.shellRadius(m, cfg).r64, b = H.shellRadius(m + 1, cfg).r64; if (Math.abs(a - 1.15) < best) { best = Math.abs(a - 1.15); dR = b - a; } } return dR; })(), firstSpacing: H.shellRadius(1, cfg).r64 - H.shellRadius(0, cfg).r64 });
    }
  }
  const ringErr = (sim, cfg, R, n = 64) => {
    const pts = [];
    for (let k = 0; k < n; k++) { const a = (k / n) * 2 * Math.PI + 0.1; pts.push({ x: Math.fround(R * Math.cos(a)), y: Math.fround(R * Math.sin(a)) }); }
    solver.upload(sim.shellHistories, cfg.mT);
    const g = solver.readFull(pts, cfg);
    let num = 0, den = 0, flagged = 0, numF = 0, denF = 0, numU = 0, denU = 0, maxU = 0;
    pts.forEach((p, i) => {
      const e = H.exactField(sim.trajectories, p.x, p.y, sim.time, { c: cfg.c, softEps: 0, maxLookback: 5000 });
      const d2 = (g[i].ex - e.ex) ** 2 + (g[i].ey - e.ey) ** 2, m2 = e.mag ** 2;
      num += d2; den += m2;
      const bad = g[i].quality < 0 || g[i].quality > 1.5;
      if (bad) { flagged++; numF += d2; denF += m2; } else { numU += d2; denU += m2; maxU = Math.max(maxU, Math.sqrt(d2 / m2) * 100); }
    });
    const pct = (a, b) => (b > 0 ? Math.sqrt(a / b) * 100 : null);
    return { R, l2: pct(num, den), flagged, n, l2Flagged: pct(numF, denF), l2Unflagged: pct(numU, denU), maxUnflagged: flagged < n ? maxU : null };
  };
  if (has('far')) {
    out.far = [];
    for (const beta of [0.5, 0.9]) for (const M of [128, 512, 2048]) for (const taper of [false, true]) {
      const cfg = { ...prod, mT: M, kBin: kb[M], prefilter: taper };
      const tr = H.circularOrbit({ radius: 0.85, beta, c: cfg.c });
      const sim = new H.Simulation(cfg, [tr], 2.4);
      for (let k = 0; k < 16; k++) sim.step(cfg.dt);
      const rows = [1.15, 3, 10, 30, 100, 200].map((R) => ringErr(sim, cfg, R));
      out.far.push({ beta, M, taper, rows, rMax: H.shellRadius(M - 1, cfg).r64 });
    }
  }
  if (has('hyper')) {
    out.hyper = [];
    // Long constant-proper-acceleration history; observers just AHEAD of the charge, whose retarded
    // points sit far back on the coarse outer shells where velocity changes a lot per segment.
    for (const [al, back] of [[0.5, 60], [0.25, 100]]) for (const M of [128, 512]) for (const cert of [2, 0, 1]) {
      const cfg = { ...prod, mT: M, kBin: kb[M], certMode: cert };
      const tEval = 2.4 + 16 * cfg.dt;
      const tr = H.hyperbolicMotion({ properAccel: al, c: cfg.c, t0: tEval - back, x0: 0 });
      const sim = new H.Simulation(cfg, [tr], 2.4);
      for (let k = 0; k < 16; k++) sim.step(cfg.dt);
      const now = sim.sources[0].current;
      const pts = [];
      for (let k = 0; k < 128; k++) { const d = 0.02 * Math.pow(1.04, k % 64), ang = (k >> 6) ? 0.004 : 0; pts.push({ x: Math.fround(now.x + d * Math.cos(ang)), y: Math.fround(now.y + d * Math.sin(ang)) }); }
      solver.upload(sim.shellHistories, cfg.mT);
      const g = solver.readFull(pts, cfg);
      let num = 0, den = 0, worst = 0, flagged = 0, inDomain = 0;
      pts.forEach((p, i) => {
        if (g[i].quality < 0) return;          // outside stored history: not comparable
        inDomain++;
        const e = H.exactField(sim.trajectories, p.x, p.y, sim.time, { c: cfg.c, softEps: 0, maxLookback: cfg.ringSize * cfg.dt * cfg.c * 0.999 });
        const d = Math.hypot(g[i].ex - e.ex, g[i].ey - e.ey);
        num += d * d; den += e.mag * e.mag; worst = Math.max(worst, d / Math.max(1e-30, e.mag));
        if (g[i].quality > 1.5) flagged++;
      });
      const h = sim.sources[0].shells; let peak = 0, peakR = 0;
      for (let m = 0; m < M - 1; m++) {
        const dR = h.emitR64[m + 1] - h.emitR64[m];
        const b = (i) => { const ux = h.emitUx64[i], uy = h.emitUy64[i], gg = Math.sqrt(1 + ux * ux + uy * uy); return [ux / gg, uy / gg]; };
        const [b1x, b1y] = b(m), [b2x, b2y] = b(m + 1);
        const vx = (h.emitX64[m + 1] - h.emitX64[m]) / dR, vy = (h.emitY64[m + 1] - h.emitY64[m]) / dR;
        const V = [[-b1x, -b1y], [3 * vx + b1x + b2x, 3 * vy + b1y + b2y], [-b2x, -b2y]];
        for (let k = 0; k <= 50; k++) { const s = k / 50, w = [(1 - s) ** 2, 2 * s * (1 - s), s * s]; const sp = Math.hypot(w[0] * V[0][0] + w[1] * V[1][0] + w[2] * V[2][0], w[0] * V[0][1] + w[1] * V[1][1] + w[2] * V[2][1]); if (sp > peak) { peak = sp; peakR = h.emitR64[m]; } }
      }
      out.hyper.push({ properAccel: al, accelTime: back, M, cert, inDomain, flagged, l2: den ? Math.sqrt(num / den) * 100 : null, worst: worst * 100, peakUnlimitedSpeed: peak, peakAtR: peakR, gammaNow: 1 / Math.sqrt(1 - (now.bx ** 2 + now.by ** 2)) });
    }
  }
  if (has('line')) {
    // Unaccelerated motion far from the charge: the validity flag must catch the end of the stored history.
    out.line = [];
    for (const M of [512, 2048]) {
      const cfg = { ...prod, mT: M, kBin: kb[M] };
      const tr = H.linearMotion({ beta: 0.5, c: cfg.c });
      const sim = new H.Simulation(cfg, [tr], 2.4);
      for (let k = 0; k < 16; k++) sim.step(cfg.dt);
      const now = sim.sources[0].current;
      const ds = [50, 100, 150, 200, 230, 300, 500, 800];
      const pts = ds.map((d) => ({ x: Math.fround(now.x), y: Math.fround(now.y + d) }));
      solver.upload(sim.shellHistories, M);
      const g = solver.readFull(pts, cfg);
      out.line.push({ M, ringDepth: cfg.c * cfg.dt * cfg.ringSize, rows: pts.map((p, i) => {
        const e = H.exactField(sim.trajectories, p.x, p.y, sim.time, { c: cfg.c, softEps: 0, maxLookback: 5000 });
        return { d: ds[i], rRet: e.rRet, err: Math.hypot(g[i].ex - e.ex, g[i].ey - e.ey) / e.mag * 100, quality: g[i].quality };
      }) });
    }
  }
  if (has('hypertarget')) {
    // Observers on-axis ahead of an accelerating charge, placed so their retarded points fall inside the
    // segment where the unlimited cubic is fastest (where a limiter can matter).
    out.hyperTarget = [];
    for (const [al, back, M] of [[0.25, 100, 128], [0.15, 105, 128], [0.25, 100, 64]]) {
      const base = { ...prod, mT: M, kBin: 9 };
      const tEval = 2.4 + 16 * base.dt;
      const tr = H.hyperbolicMotion({ properAccel: al, c: base.c, t0: tEval - back, x0: 0 });
      const sim = new H.Simulation(base, [tr], 2.4);
      for (let k = 0; k < 16; k++) sim.step(base.dt);
      const h = sim.sources[0].shells, now = sim.sources[0].current;
      let best = { peak: 0, m: 0 };
      for (let m = 0; m < M - 1; m++) {
        const dR = h.emitR64[m + 1] - h.emitR64[m];
        const b = (i) => { const ux = h.emitUx64[i], uy = h.emitUy64[i], gg = Math.sqrt(1 + ux * ux + uy * uy); return [ux / gg, uy / gg]; };
        const [b1x, b1y] = b(m), [b2x, b2y] = b(m + 1);
        const vx = (h.emitX64[m + 1] - h.emitX64[m]) / dR, vy = (h.emitY64[m + 1] - h.emitY64[m]) / dR;
        const V = [[-b1x, -b1y], [3 * vx + b1x + b2x, 3 * vy + b1y + b2y], [-b2x, -b2y]];
        for (let k = 0; k <= 100; k++) { const s = k / 100, w = [(1 - s) ** 2, 2 * s * (1 - s), s * s]; const sp = Math.hypot(w[0] * V[0][0] + w[1] * V[1][0] + w[2] * V[2][0], w[0] * V[0][1] + w[1] * V[1][1] + w[2] * V[2][1]); if (sp > best.peak) best = { peak: sp, m }; }
      }
      const R1 = h.emitR64[best.m], R2 = h.emitR64[best.m + 1];
      const xFor = (R) => tr.kinematics(sim.time - R / base.c).x + R;
      const pts = [];
      for (let k = 0; k < 256; k++) { const R = R1 + (R2 - R1) * (k + 0.5) / 256; pts.push({ x: Math.fround(xFor(R)), y: 0 }); }
      const row = { properAccel: al, accelTime: back, M, segment: best.m, R1, R2, peakUnlimitedSpeed: best.peak, gammaNow: 1 / Math.sqrt(1 - now.bx ** 2) };
      for (const cert of [2, 0, 1]) {
        const cfg = { ...base, certMode: cert };
        solver.upload(sim.shellHistories, M);
        const g = solver.readFull(pts, cfg);
        let num = 0, den = 0, worst = 0, flagged = 0;
        pts.forEach((p, i) => {
          const e = H.exactField(sim.trajectories, p.x, p.y, sim.time, { c: cfg.c, softEps: 0, maxLookback: 5000 });
          const d = Math.hypot(g[i].ex - e.ex, g[i].ey - e.ey);
          num += d * d; den += e.mag * e.mag; worst = Math.max(worst, d / e.mag);
          if (g[i].quality > 1.5 || g[i].quality < 0) flagged++;
        });
        row['limiter' + cert] = { l2: Math.sqrt(num / den) * 100, worst: worst * 100, flagged, n: pts.length };
      }
      out.hyperTarget.push(row);
    }
  }
  if (has('cost')) {
    const canvas = document.getElementById('r');
    const s2 = new H.GpuSolver(canvas, H.V6_SHADERS);   // the switches below are the v6 kernel's
    const cfg0 = { ...prod, mT: 512, kBin: 11 };
    const trs = Array.from({ length: 16 }, (_, i) => H.precessingEllipse({ index: i, beta: 0.8, omega: 1.6, c: cfg0.c }));
    const sim = new H.Simulation(cfg0, trs, 2.5);
    s2.upload(sim.shellHistories, 512);
    const sync = () => { const px = new Uint8Array(4); s2.gl.readPixels(0, 0, 1, 1, s2.gl.RGBA, s2.gl.UNSIGNED_BYTE, px); };
    const time = (c) => { s2.render(640, 400, c, { u_channel: 0, u_showWavePool: 1, u_viewCenter: [0, 0] }, { u_aspect: 1.6, u_viewHalfHeight: 1.45, u_gain: 1, u_maskThreshold: 1.5 }); sync(); const t0 = performance.now(); for (let f = 0; f < 3; f++) s2.render(640, 400, c, {}, {}); sync(); return (performance.now() - t0) / 3; };
    const variants = [
      ['production', cfg0],
      ['float32 root (otherwise production)', { ...cfg0, precision: 0 }],
      ['version-5 algorithm (switches)', { ...cfg0, precision: 0, certMode: 0, rootMode: 0, newton: 2 }],
    ];
    out.cost = variants.map(([name, c]) => ({ name, ms: [time(c), time(c), time(c)] }));
  }
  return out;
}, [which, kernel]);
const path = root + 'results/tables' + (which.includes('all') ? '' : '_' + which.join('_')) + (kernel === 'v6' ? '' : '_' + kernel) + '.json';
writeFileSync(path, JSON.stringify(res, null, 2));
console.log('wrote', path);
await browser.close();
