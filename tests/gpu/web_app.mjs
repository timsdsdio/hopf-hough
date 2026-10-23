// Checks for the author's app (web/index.html, served at the site root) with the v6 kernel ported in.
//  1. loads without console errors; the kernel and mask buttons work;
//  2. the v5 kernel is bit-identical to the original file (git HEAD~ version kept in reference/web_original.html):
//     same field readback on a grid and same rendered pixels;
//  3. the v6 kernel reproduces the production accuracy on the pulse sweep (no softening);
//  4. relative frame time v6 / v5 (software rendering: relative only).
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
const SNAPSHOT = `(() => {
  state.paused = true;
  if (typeof selectKernel === 'function') { state.kernel = 'v5'; selectKernel(); }
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

const SWEEP = (kernel, beta, M, kBin) => `(() => {
  state.paused = true; state.kernel = '${kernel}'; selectKernel();
  state.numParticles = 1; state.softEps = 0;
  const t0 = 2.40; state.time = t0;
  seedExactRetardedHistory(state.sources[0], 0, 1, t0, ${beta}, 'synchrotron', ${M}, 2);
  for (let k = 0; k < 16; k++) stepSimulation(DT_STEP, ${beta}, 'synchrotron', ${M}, 2);
  const pts = getPulseCrossingSweepPoints(1.15, state.time, ${beta}, 256).pts;
  const g = evaluateViaGPUFramebufferReadback(pts, ${M}, ${kBin}, 2);
  let num = 0, den = 0;
  pts.forEach((p, i) => { const e = evaluateExactAnalyticalLW(p.x, p.y, state.time, ${beta}, 'synchrotron');
    num += (g[i].ex - e.ex) ** 2 + (g[i].ey - e.ey) ** 2; den += e.mag ** 2; });
  return 100 * Math.sqrt(num / den);
})()`;

const FRAMES = (kernel) => `(() => {
  state.paused = true; state.kernel = '${kernel}'; selectKernel(); state.time = 2.5; reseedAllSources();
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
    const a = await orig.page.evaluate(SNAPSHOT);
    await orig.page.close();

    const app = await open(browser, APP);
    check(app.errors.length === 0, `web app loads without errors${app.errors.length ? ': ' + app.errors.join('; ') : ''}`);
    const kernelAtLoad = await app.page.evaluate('state.kernel');
    check(kernelAtLoad === 'v6', `web app starts on the v6 kernel (${kernelAtLoad})`);
    await app.page.click('#btn-kernel');
    const k2 = await app.page.evaluate('state.kernel + "|" + document.getElementById("btn-kernel").textContent.trim()');
    await app.page.click('#btn-kernel');
    await app.page.click('#btn-mask');
    const m = await app.page.evaluate('state.showMask');
    await app.page.click('#btn-mask');
    check(k2.startsWith('v5|Kernel: v5') && m === true, `kernel and mask buttons toggle (${k2}, mask ${m})`);

    const b = await app.page.evaluate(SNAPSHOT);
    const sameField = a.field.length === b.field.length && a.field.every((v, i) => Object.is(v, b.field[i]));
    check(sameField, `v5 kernel field readback bit-identical to the original (${a.field.length / 4} points)`);
    check(a.w === b.w && a.h === b.h && a.hash === b.hash, `v5 kernel rendered frame identical to the original (${b.w}×${b.h})`);

    const targets = [[0.999, 0.03], [0.9999, 0.2]];
    for (const [beta, lim] of targets) {
      const e6 = await app.page.evaluate(SWEEP('v6', beta, 512, 11));
      const e5 = await app.page.evaluate(SWEEP('v5', beta, 512, 11));
      check(e6 < lim, `v6 kernel pulse sweep β=${beta}, M=512: ${e6.toPrecision(3)}% < ${lim}% (v5 kernel: ${e5.toPrecision(3)}%)`);
    }

    const t5 = await app.page.evaluate(FRAMES('v5'));
    const t6 = await app.page.evaluate(FRAMES('v6'));
    log(`info frame time, default scene (32 sources, M=128), software rendering: v5 ${t5.toFixed(0)} ms, v6 ${t6.toFixed(0)} ms (ratio ${(t6 / t5).toFixed(2)})`);
  } finally {
    await browser.close();
  }
  return ok;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  runWebAppChecks(console.log).then((ok) => process.exit(ok ? 0 : 1));
}
