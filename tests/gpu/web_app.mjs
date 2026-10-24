// Checks for the author's app (web/index.html, served at the site root) with the v6 kernel ported in.
//  1. loads without console errors; the kernel and mask buttons work;
//  2. the v5 kernel is bit-identical to the original file (git HEAD~ version kept in reference/web_original.html):
//     same field readback on a grid and same rendered pixels;
//  3. the v6 kernel reproduces the production accuracy on the pulse sweep (no softening);
//  4. known-path mode (exact knots, C2 quintic): pulse sweep accuracy, ring accuracy after live steps, drag hand-off;
//  5. relative frame time v6 / v5 / known path (software rendering: relative only).
import { launch } from './browser.mjs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const APP = 'file://' + resolve(root, 'web/index.html');
const ORIGINAL = 'file://' + resolve(root, 'reference/web_original.html');

async function open(browser, url) {
  // Small viewport: software rendering of a full-size frame with 32 sources takes seconds.
  const page = await browser.newPage({ viewport: { width: 640, height: 700 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  // The animation loop keeps the (software) GPU busy, so wait for the app's state, not for 'load'.
  await page.goto(url, { waitUntil: 'commit' });
  await page.waitForFunction(() => typeof state !== 'undefined' && state.sources && state.sources.length > 0, null, { timeout: 300000, polling: 500 });
  await page.evaluate('state.paused = true');
  await page.waitForTimeout(500);
  return { page, errors };
}

// Fixed scene, field on a 16×16 grid and the rendered frame (both read back as raw floats/bytes).
// Compared on the orbit and oscillator presets: the ensemble ('swarm') preset's kinematics were fixed in
// the known-path build (frame rotation counted once in the acceleration, constant time-scale speed cap),
// so its motion, not the kernel, differs from the original file there.
const SNAPSHOT = (mode) => `(() => {
  state.paused = true;
  if (typeof selectKernel === 'function') { state.kernel = 'v5'; selectKernel(); }
  state.motionMode = '${mode}';
  state.time = 2.5;
  reseedAllSources();
  const pts = [];
  for (let j = 0; j < 16; j++) for (let i = 0; i < 16; i++) pts.push({ x: -1.4 + 2.8 * i / 15, y: -1.4 + 2.8 * j / 15 });
  const f = evaluateViaGPUFramebufferReadback(pts, state.mT, state.kBin, state.velMode).map((o) => [o.ex, o.ey, o.emitX, o.emitY]).flat();
  renderBulkField();
  const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  let hash = 0; for (let i = 0; i < px.length; i++) hash = (hash * 31 + px[i]) >>> 0;
  return { field: f, w, h, hash };
})()`;

// known: exact knots of the analytic orbit (known-path mode); live: run the ring at the app's live step first.
const SWEEP = (kernel, beta, M, kBin, known = false, live = 0) => `(() => {
  state.paused = true; state.kernel = '${kernel}'; selectKernel();
  state.numParticles = 1; state.softEps = 0; state.knownPath = ${known};
  const t0 = 2.40; state.time = t0;
  seedExactRetardedHistory(state.sources[0], 0, 1, t0, ${beta}, 'synchrotron', ${M}, 2);
  for (let k = 0; k < 16; k++) stepSimulation(DT_STEP, ${beta}, 'synchrotron', ${M}, 2);
  for (let k = 0; k < ${live}; k++) stepSimulation(0.45 * DT_STEP, ${beta}, 'synchrotron', ${M}, 2);
  // calls with explicit beta/motion overrides keep the ring (that is the app's rule); resample the knots exactly here
  if (${known}) { state.sources[0].dragWins = []; populateKnownPathShells(state.sources[0], 0, 1, ${M}, 2, DT_STEP, ${beta}, 'synchrotron'); }
  const pts = getPulseCrossingSweepPoints(1.15, state.time, ${beta}, 256).pts;
  const g = evaluateViaGPUFramebufferReadback(pts, ${M}, ${kBin}, 2);
  let num = 0, den = 0;
  pts.forEach((p, i) => { const e = evaluateExactAnalyticalLW(p.x, p.y, state.time, ${beta}, 'synchrotron');
    num += (g[i].ex - e.ex) ** 2 + (g[i].ey - e.ey) ** 2; den += e.mag ** 2; });
  return 100 * Math.sqrt(num / den);
})()`;

// Grab source 0 of the orbit preset, drag it on a jerky path, release; once it has rejoined its orbit, the field in
// known-path mode must match the same run in ring mode (the dragged stretch of history comes from the ring in both).
const HANDOFF = (known) => `(() => {
  state.paused = true; state.kernel = 'v6'; selectKernel(); state.knownPath = ${known};
  state.numParticles = 4; state.motionMode = 'synchrotron'; state.beta = 0.8; state.mT = 512; state.kBin = 11;
  state.time = 2.5; state.softEps = 0.02; reseedAllSources();
  const dt = 0.45 * DT_STEP, s0 = state.sources[0];
  for (let k = 0; k < 400; k++) stepSimulation(dt);
  state.interactionTarget = 'source0'; s0.returningToOrbit = true; s0._drag = null;
  let k = 0, rejoin = -1;
  for (; k < 6000; k++) {
    if (k < 300) state.draggedSourceTarget = { x: 0.6 * Math.sin(3.1 * k * dt) + 0.3, y: 0.5 * Math.cos(2.3 * k * dt) };
    if (k === 300) state.interactionTarget = null;
    stepSimulation(dt);
    if (k > 300 && rejoin < 0 && !s0.returningToOrbit) rejoin = k;
    if (rejoin >= 0 && k > rejoin + 200) break;
  }
  const pts = [];
  for (let j = 0; j < 10; j++) for (let i = 0; i < 10; i++) pts.push({ x: -2.2 + 4.4 * (i + 0.5) / 10, y: -1.4 + 2.8 * (j + 0.5) / 10 });
  const g = evaluateViaGPUFramebufferReadback(pts, 512, 11, 2);
  return { rejoin, e: g.map((o) => [o.ex, o.ey]), exact: Array.from(s0.emitExact.slice(0, 512)).reduce((a, b) => a + b, 0) };
})()`;

const FRAMES = (kernel, known = false) => `(() => {
  state.paused = true; state.kernel = '${kernel}'; selectKernel(); state.knownPath = ${known};
  state.numParticles = 32; state.motionMode = 'swarm'; state.mT = 128; state.kBin = 9; state.softEps = 0.06;
  state.time = 2.5; reseedAllSources();
  const one = new Uint8Array(4);
  renderBulkField(); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, one);
  const t0 = performance.now();
  for (let k = 0; k < 5; k++) { renderBulkField(); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, one); }
  return (performance.now() - t0) / 5;
})()`;

export async function runWebAppChecks(log) {
  const browser = await launch();
  let ok = true;
  const check = (cond, msg) => { log((cond ? 'ok   ' : 'FAIL ') + msg); ok = ok && cond; };
  try {
    const orig = await open(browser, ORIGINAL);
    const MODES = ['synchrotron', 'oscillator'];
    const a = {};
    for (const mode of MODES) a[mode] = await orig.page.evaluate(SNAPSHOT(mode));
    await orig.page.close();

    const app = await open(browser, APP);
    check(app.errors.length === 0, `web app loads without errors${app.errors.length ? ': ' + app.errors.join('; ') : ''}`);
    const kernelAtLoad = await app.page.evaluate('state.kernel');
    check(kernelAtLoad === 'v6', `web app starts on the v6 kernel (${kernelAtLoad})`);
    await app.page.evaluate(() => document.getElementById('btn-kernel').click());
    const k2 = await app.page.evaluate('state.kernel + "|" + document.getElementById("btn-kernel").textContent.trim()');
    await app.page.evaluate(() => document.getElementById('btn-kernel').click());
    await app.page.evaluate(() => document.getElementById('btn-mask').click());
    const m = await app.page.evaluate('state.showMask');
    await app.page.evaluate(() => document.getElementById('btn-mask').click());
    check(k2.startsWith('v5|Kernel: v5') && m === true, `kernel and mask buttons toggle (${k2}, mask ${m})`);

    for (const mode of MODES) {
      const A = a[mode], b = await app.page.evaluate(SNAPSHOT(mode));
      const sameField = A.field.length === b.field.length && A.field.every((v, i) => Object.is(v, b.field[i]));
      check(sameField, `v5 kernel field readback bit-identical to the original, ${mode} (${A.field.length / 4} points)`);
      check(A.w === b.w && A.h === b.h && A.hash === b.hash, `v5 kernel rendered frame identical to the original, ${mode} (${b.w}×${b.h})`);
    }

    const targets = [[0.999, 0.009], [0.9999, 0.05]];   // the v6.0.0 kernel (0.0095% / 0.069%) would fail these
    for (const [beta, lim] of targets) {
      const e6 = await app.page.evaluate(SWEEP('v6', beta, 512, 11));   // ring mode
      const e5 = await app.page.evaluate(SWEEP('v5', beta, 512, 11));
      check(e6 < lim, `v6 kernel pulse sweep β=${beta}, M=512: ${e6.toPrecision(3)}% < ${lim}% (v5 kernel: ${e5.toPrecision(3)}%)`);
    }

    // known-path mode: exact knots + C2 quintic (equal shells: well below the ring; see docs/ACCURACY.md)
    const kTargets = [[0.99, 128, 9, 5e-4], [0.999, 512, 11, 2e-3]];
    for (const [beta, M, kb, lim] of kTargets) {
      const ek = await app.page.evaluate(SWEEP('v6', beta, M, kb, true));
      const er = await app.page.evaluate(SWEEP('v6', beta, M, kb, false));
      check(ek < lim, `known-path pulse sweep β=${beta}, M=${M}: ${ek.toPrecision(3)}% < ${lim}% (ring: ${er.toPrecision(3)}%)`);
    }
    // ring after live steps (0.45 DT_STEP): the ring's tangents use the step it was written at
    const el = await app.page.evaluate(SWEEP('v6', 0.999, 512, 11, false, 600));
    check(el < 0.02, `v6 ring pulse sweep after 600 live steps, β=0.999, M=512: ${el.toPrecision(3)}% < 0.02%`);

    const hA = await app.page.evaluate(HANDOFF(true)), hB = await app.page.evaluate(HANDOFF(false));
    let num = 0, den = 0;
    hA.e.forEach((v, j) => { num += (v[0] - hB.e[j][0]) ** 2 + (v[1] - hB.e[j][1]) ** 2; den += hB.e[j][0] ** 2 + hB.e[j][1] ** 2; });
    const rel = Math.sqrt(num / den);
    check(hA.rejoin > 0 && hA.rejoin === hB.rejoin && rel < 1e-3,
      `drag hand-off: rejoins at step ${hA.rejoin} (ring mode ${hB.rejoin}); known-path vs ring field ${rel.toExponential(2)} RMS < 1e-3; ${hA.exact}/512 exact knots`);

    const t5 = await app.page.evaluate(FRAMES('v5'));
    const t6 = await app.page.evaluate(FRAMES('v6'));
    const tk = await app.page.evaluate(FRAMES('v6', true));
    log(`info frame time, default scene (32 sources, M=128), software rendering: v5 ${t5.toFixed(0)} ms, v6 ${t6.toFixed(0)} ms (ratio ${(t6 / t5).toFixed(2)}), v6 known path ${tk.toFixed(0)} ms (ratio ${(tk / t5).toFixed(2)})`);
  } finally {
    await browser.close();
  }
  return ok;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  runWebAppChecks(console.log).then((ok) => process.exit(ok ? 0 : 1));
}
