// Checks for the v6f kernel (v6 compiled with F32N) and for the v6 kernel it leaves unchanged.
// Headless Chromium + SwiftShader (IEEE float32 WebGL2). Library (dist/hopf-hough.iife.js) and the author's app
// (web/index.html):
//  1. v6 bit-identity: the library's V6_SHADERS / V6_KNOWN_PATH_SHADERS against the v6 shader of main at commit 904bcce
//     (κ-form radiation term and Newton fix; not the v6.0.0 kernel of preprint v6.1), kept in
//     reference/shader_v6_1.glsl (field readback and rendered pixels); the app's v6 kernel against the app of main at
//     904bcce, kept in reference/web_v6_1.html (readback and frames, ring and known path). The app's v5 kernel is checked
//     against reference/web_original.html in web_app.mjs.
//  2. v6f accuracy at γ = 70.7 and 223.6 against the float64 reference (pulse sweep, no softening), on the known path
//     and on the live path (library: ring-buffer history; app: the ring after 600 live steps); v6f also below v6.
//  3. v6f display-path pixels against v6 (library render and app frame).
import { launch } from './browser.mjs';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const APP = 'file://' + resolve(root, 'web/index.html');
const APP_V61 = 'file://' + resolve(root, 'reference/web_v6_1.html');

// Library-side checks, run inside the page (H = HopfHough). refFrag: the v6 production fragment shader of main at 904bcce.
function libraryChecks(refFrag) {
  const H = HopfHough, kb = { 128: 9, 512: 11, 2048: 13 };
  const canvas = (w = 16, h = 16) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
  const withDef = (src, defs) => src.replace('#version 300 es\n', '#version 300 es\n' + defs.map((d) => `#define ${d}\n`).join(''));
  const REF = { vertex: H.VERTEX_SHADER, fragment: refFrag, production: true };
  const REFK = { vertex: H.VERTEX_SHADER, fragment: withDef(refFrag, ['KNOWN_PATH']), production: true };
  const out = { identity: [], accuracy: [], pixels: [] };

  const sim1 = (beta, M, known) => {
    const cfg = { ...H.DEFAULT_CONFIG, softEps: 0, prefilter: false, knownPath: known, mT: M, kBin: kb[M], velMode: 2 };
    const sim = new H.Simulation(cfg, [H.legacyPreset('synchrotron', { index: 0, count: 1, beta, omega: cfg.omega, c: cfg.c })], 2.4);
    for (let k = 0; k < 16; k++) sim.step(cfg.dt);
    return { sim, cfg };
  };
  const scene = (n, beta, M, known) => {
    const cfg = { ...H.DEFAULT_CONFIG, softEps: 0.06, knownPath: known, mT: M, kBin: kb[M], velMode: 2 };
    const tr = []; for (let i = 0; i < n; i++) tr.push(H.legacyPreset('swarm', { index: i, count: n, beta, omega: cfg.omega, c: cfg.c }));
    const sim = new H.Simulation(cfg, tr, 2.5);
    for (let k = 0; k < 4; k++) sim.step(cfg.dt);
    return { sim, cfg };
  };
  const W = 160, Hh = 100;
  const frame = (solver, sim, cfg) => {
    solver.upload(sim.shellHistories, cfg.mT);
    solver.render(W, Hh, cfg, { u_channel: 0, u_showWavePool: true, u_showImpostors: false, u_showMask: false, u_viewCenter: [0, 0] },
      { u_aspect: W / Hh, u_viewHalfHeight: 1.45, u_time: sim.time, u_gain: 1, u_maskThreshold: 1.5 });
    const gl = solver.gl, px = new Uint8Array(W * Hh * 4);
    gl.readPixels(0, 0, W, Hh, gl.RGBA, gl.UNSIGNED_BYTE, px);
    return px;
  };
  const read = (solver, sim, cfg, pts) => { solver.upload(sim.shellHistories, cfg.mT); return solver.readRaw(pts, cfg, 1); };
  const sweepPts = (sim, beta, df64) => H.pulseCrossingSweep(1.15, sim.time, beta, 256, sim.config.omega, sim.config.c).pts
    .map((p) => (df64 ? p : { x: p.x, y: p.y }));
  const gridPts = () => { const p = []; for (let j = 0; j < 16; j++) for (let i = 0; i < 16; i++) p.push({ x: -1.4 + 2.8 * i / 15, y: -1.4 + 2.8 * j / 15 }); return p; };

  // 1. v6 bit-identity against the v6 shader of main at 904bcce
  const cv = canvas(W, Hh), cr = canvas(W, Hh), ckv = canvas(W, Hh), ckr = canvas(W, Hh);
  const v6 = new H.GpuSolver(cv, H.V6_SHADERS), ref = new H.GpuSolver(cr, REF);
  const v6k = new H.GpuSolver(ckv, H.V6_KNOWN_PATH_SHADERS), refk = new H.GpuSolver(ckr, REFK);
  for (const [beta, M, known, df64] of [[0.999, 512, false, false], [0.9999, 128, true, false], [0.99999, 2048, false, true], [0.99999, 512, true, true]]) {
    const { sim, cfg } = sim1(beta, M, known);
    const pts = sweepPts(sim, beta, df64).concat(gridPts());
    const a = read(known ? v6k : v6, sim, cfg, pts), b = read(known ? refk : ref, sim, cfg, pts);
    out.identity.push({ what: `readback β=${beta} M=${M} ${known ? 'known' : 'ring'}${df64 ? ' df64' : ''}`, n: a.length, same: a.length === b.length && a.every((v, i) => Object.is(v, b[i])) });
  }
  for (const known of [false, true]) {
    const { sim, cfg } = scene(16, 0.9, 128, known);
    const a = frame(known ? v6k : v6, sim, cfg), b = frame(known ? refk : ref, sim, cfg);
    out.identity.push({ what: `rendered frame ${W}×${Hh}, 16 sources ${known ? 'known' : 'ring'}`, n: a.length, same: a.every((v, i) => v === b[i]) });
  }

  // 2. v6f accuracy (shockL2 in %), against v6
  const solvers = { v6, v6k, v6f: new H.GpuSolver(canvas(W, Hh), H.V6F_SHADERS), v6fk: new H.GpuSolver(canvas(W, Hh), H.V6F_KNOWN_PATH_SHADERS) };
  for (const [beta, M, known, q] of [[0.9999, 512, true, 'f32'], [0.9999, 512, false, 'f32'], [0.9999, 2048, true, 'df64'], [0.9999, 2048, false, 'df64'],
    [0.99999, 512, true, 'df64'], [0.99999, 512, false, 'df64'], [0.99999, 2048, true, 'df64'], [0.99999, 2048, false, 'df64']]) {
    const o = { base: { ...H.DEFAULT_CONFIG, softEps: 0, prefilter: false, knownPath: known }, queryMode: q };
    const spec = { beta, motion: 'synchrotron', mT: M, kBin: kb[M], velMode: 2, nShock: 256 };
    const e6 = H.runCase(spec, H.gpuBackend(known ? solvers.v6k : solvers.v6), o).shockL2;
    const e6f = H.runCase(spec, H.gpuBackend(known ? solvers.v6fk : solvers.v6f), o).shockL2;
    out.accuracy.push({ beta, M, path: known ? 'known' : 'ring', obs: q, v6: e6, v6f: e6f });
  }

  // 3. display path: v6f vs v6 rendered pixels
  for (const [n, beta, M, known] of [[16, 0.9, 128, false], [16, 0.9, 512, true], [8, 0.999, 512, true]]) {
    const { sim, cfg } = scene(n, beta, M, known);
    const a = frame(known ? solvers.v6fk : solvers.v6f, sim, cfg), b = frame(known ? solvers.v6k : solvers.v6, sim, cfg);
    let diff = 0, maxd = 0; for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d) { diff++; maxd = Math.max(maxd, d); } }
    out.pixels.push({ what: `${n} sources β=${beta} M=${M} ${known ? 'known' : 'ring'}`, bytes: a.length, diff, maxd });
  }
  return out;
}

// App: one synchrotron source, pulse sweep (as web_app.mjs SWEEP), with df64 observers when asked.
const SWEEP = (kernel, beta, M, kBin, known, live, df64) => `(() => {
  state.paused = true; state.kernel = '${kernel}'; selectKernel();
  state.numParticles = 1; state.softEps = 0; state.knownPath = ${known};
  const t0 = 2.40; state.time = t0;
  seedExactRetardedHistory(state.sources[0], 0, 1, t0, ${beta}, 'synchrotron', ${M}, 2);
  for (let k = 0; k < 16; k++) stepSimulation(DT_STEP, ${beta}, 'synchrotron', ${M}, 2);
  for (let k = 0; k < ${live}; k++) stepSimulation(0.45 * DT_STEP, ${beta}, 'synchrotron', ${M}, 2);
  if (${known}) { state.sources[0].dragWins = []; populateKnownPathShells(state.sources[0], 0, 1, ${M}, 2, DT_STEP, ${beta}, 'synchrotron'); }
  const pts = getPulseCrossingSweepPoints(1.15, state.time, ${beta}, 256).pts.map((p) => ${df64} ? { x: p.x64, y: p.y64 } : { x: p.x, y: p.y });
  const g = evaluateViaGPUFramebufferReadback(pts, ${M}, ${kBin}, 2);
  let num = 0, den = 0;
  pts.forEach((p, i) => { const e = evaluateExactAnalyticalLW(p.x, p.y, state.time, ${beta}, 'synchrotron');
    num += (g[i].ex - e.ex) ** 2 + (g[i].ey - e.ey) ** 2; den += e.mag ** 2; });
  return 100 * Math.sqrt(num / den);
})()`;

// App: fixed scene, field readback on a grid and the rendered frame (raw bytes).
const SNAP = (kernel, mode, known, n, M, kBin) => `(() => {
  state.paused = true; state.kernel = '${kernel}'; selectKernel(); state.knownPath = ${known};
  state.numParticles = ${n}; state.motionMode = '${mode}'; state.mT = ${M}; state.kBin = ${kBin}; state.softEps = 0.06;
  state.time = 2.5; reseedAllSources();
  const pts = [];
  for (let j = 0; j < 16; j++) for (let i = 0; i < 16; i++) pts.push({ x: -1.4 + 2.8 * i / 15, y: -1.4 + 2.8 * j / 15 });
  const f = evaluateViaGPUFramebufferReadback(pts, state.mT, state.kBin, state.velMode).map((o) => [o.ex, o.ey, o.emitX, o.emitY]).flat();
  renderBulkField();
  const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  return { field: f, w, h, px: Array.from(px) };
})()`;

async function open(browser, url) {
  const page = await browser.newPage({ viewport: { width: 640, height: 700 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(url, { waitUntil: 'commit' });
  await page.waitForFunction(() => typeof state !== 'undefined' && state.sources && state.sources.length > 0, null, { timeout: 300000, polling: 500 });
  await page.evaluate('state.paused = true');
  await page.waitForTimeout(500);
  return { page, errors };
}

const pct = (v) => v.toPrecision(3) + '%';

export async function runV6fChecks(log) {
  let ok = true;
  const check = (cond, msg) => { log((cond ? 'ok   ' : 'FAIL ') + msg); ok = ok && cond; };
  const browser = await launch();
  try {
    // ---------------- library ----------------
    const page = await browser.newPage();
    page.setDefaultTimeout(0);
    const errs = [];
    page.on('pageerror', (e) => errs.push(e.message));
    await page.setContent('<html><body></body></html>');
    await page.addScriptTag({ content: readFileSync(resolve(root, 'dist/hopf-hough.iife.js'), 'utf8') });
    const refFrag = readFileSync(resolve(root, 'reference/shader_v6_1.glsl'), 'utf8').replace(/\n$/, '');
    const lib = await page.evaluate(libraryChecks, refFrag);
    const def = await page.evaluate(() => [HopfHough.PRODUCTION_SHADERS.kernel, HopfHough.KNOWN_PATH_SHADERS.kernel, new HopfHough.GpuSolver(document.createElement('canvas')).f32n]);
    await page.close();
    check(errs.length === 0, 'library page has no errors' + (errs.length ? ': ' + errs.join('; ') : ''));
    check(def[0] === 'v6f' && def[1] === 'v6f' && def[2] === true, `library default kernel is v6f (PRODUCTION_SHADERS ${def[0]}, KNOWN_PATH_SHADERS ${def[1]}, GpuSolver() f32n ${def[2]})`);
    for (const r of lib.identity) check(r.same, `library v6 bit-identical to the v6 shader at main 904bcce: ${r.what} (${r.n} values)`);
    // Limits about 2x the SwiftShader measurement (docs/ACCURACY.md section 7); v6 is 10-2000x larger on the known path.
    const LIM = { '0.9999|512|known': 3e-4, '0.9999|512|ring': 0.03, '0.9999|2048|known': 3e-4, '0.9999|2048|ring': 4e-3,
      '0.99999|512|known': 1.5e-3, '0.99999|512|ring': 0.12, '0.99999|2048|known': 1e-3, '0.99999|2048|ring': 0.012 };
    for (const a of lib.accuracy) {
      const lim = LIM[`${a.beta}|${a.M}|${a.path}`];
      const gam = a.beta === 0.9999 ? '70.7' : '224';
      const gain = a.path === 'known' ? 20 : 1;   // the known path is set by the float32 floor v6f removes
      check(a.v6f < lim && a.v6f * gain < a.v6, `library v6f γ=${gam}, M=${a.M}, ${a.path}, ${a.obs} obs: ${pct(a.v6f)} < ${lim}% (v6: ${pct(a.v6)}, ${(a.v6 / a.v6f).toFixed(1)}x)`);
    }
    for (const p of lib.pixels) check(p.maxd <= 3 && p.diff / p.bytes < 2e-3, `library display path v6f vs v6, ${p.what}: ${p.diff} of ${p.bytes} bytes differ, max ${p.maxd}/255`);

    // ---------------- app: v6 against the app of main at 904bcce ----------------
    const SCENES = [['synchrotron', false, 4, 512, 11], ['oscillator', false, 4, 128, 9], ['synchrotron', true, 4, 512, 11], ['swarm', true, 8, 128, 9]];
    const old = await open(browser, APP_V61);
    const A = [];
    for (const s of SCENES) A.push(await old.page.evaluate(SNAP('v6', ...s)));
    await old.page.close();
    const app = await open(browser, APP);
    check(app.errors.length === 0, `web app loads without errors${app.errors.length ? ': ' + app.errors.join('; ') : ''}`);
    for (let i = 0; i < SCENES.length; i++) {
      const [mode, known, n, M] = SCENES[i];
      const b = await app.page.evaluate(SNAP('v6', ...SCENES[i])), a = A[i];
      const sameF = a.field.length === b.field.length && a.field.every((v, j) => Object.is(v, b.field[j]));
      const sameP = a.w === b.w && a.h === b.h && a.px.every((v, j) => v === b.px[j]);
      check(sameF && sameP, `app v6 kernel bit-identical to main 904bcce, ${mode} ${known ? 'known' : 'ring'} ${n} sources M=${M}: readback (${a.field.length / 4} points) and frame (${b.w}×${b.h})`);
    }

    // ---------------- app: v6f accuracy ----------------
    // [beta, M, kBin, known, live steps, df64 observers, limit %]
    const CASES = [[0.9999, 512, 11, true, 0, false, 3e-4], [0.9999, 512, 11, false, 600, false, 0.03], [0.9999, 2048, 13, true, 0, true, 3e-4], [0.9999, 2048, 13, false, 600, true, 4e-3],
      [0.99999, 512, 11, true, 0, true, 1.5e-3], [0.99999, 512, 11, false, 600, true, 0.12], [0.99999, 2048, 13, true, 0, true, 1e-3], [0.99999, 2048, 13, false, 600, true, 0.012]];
    for (const [beta, M, kb, known, live, df64, lim] of CASES) {
      const ef = await app.page.evaluate(SWEEP('v6f', beta, M, kb, known, live, df64));
      const e6 = await app.page.evaluate(SWEEP('v6', beta, M, kb, known, live, df64));
      const gain = known ? 20 : 1;
      check(ef < lim && ef * gain < e6, `app v6f γ=${beta === 0.9999 ? '70.7' : '224'}, M=${M}, ${known ? 'known' : 'live (600 steps)'}, ${df64 ? 'df64' : 'f32'} obs: ${pct(ef)} < ${lim}% (v6: ${pct(e6)}, ${(e6 / ef).toFixed(1)}x)`);
    }

    // ---------------- app: v6f display path vs v6 ----------------
    for (const s of [['swarm', false, 32, 128, 9], ['swarm', true, 32, 128, 9], ['synchrotron', true, 4, 512, 11]]) {
      const a = await app.page.evaluate(SNAP('v6f', ...s)), b = await app.page.evaluate(SNAP('v6', ...s));
      let diff = 0, maxd = 0; for (let i = 0; i < a.px.length; i++) { const d = Math.abs(a.px[i] - b.px[i]); if (d) { diff++; maxd = Math.max(maxd, d); } }
      check(maxd <= 3 && diff / a.px.length < 2e-3, `app display path v6f vs v6, ${s[0]} ${s[1] ? 'known' : 'ring'} ${s[2]} sources: ${diff} of ${a.px.length} bytes differ, max ${maxd}/255`);
    }
    check(app.errors.length === 0, `web app: no console errors after the v6f checks${app.errors.length ? ': ' + app.errors.join('; ') : ''}`);
    await app.page.close();
  } finally {
    await browser.close();
  }
  return ok;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  runV6fChecks(console.log).then((ok) => process.exit(ok ? 0 : 1));
}
