// 1. v5 and v6 in the modified app2d are bit-identical to the unmodified copy (app2d_orig.html): field readback (16x16
//    grid, Ex, Ey, emitX, emitY) and rendered frame bytes, on several presets, known and live paths, M = 128 / 2048.
// 2. Drag physics with v6f: grab source 0, jerky drag, release, rejoin (the web_app.mjs HANDOFF protocol); v6f vs v6 field,
//    known vs ring hand-off, rendered frames mid-drag (display path) without errors.
// 3. The kernel button cycles v6 -> v5 -> v6f -> v6.
// Usage: node identity_drag.mjs <out.json>
import { launch } from './browser.mjs';
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const browser = await launch();
async function open(file) {
  const page = await browser.newPage({ viewport: { width: 640, height: 700 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto('file://' + file, { waitUntil: 'commit' });
  await page.waitForFunction(() => typeof state !== 'undefined' && state.sources && state.sources.length > 0, null, { timeout: 300000, polling: 500 });
  await page.evaluate('state.paused = true');
  await page.waitForTimeout(500);
  return { page, errors };
}
const SNAP = (kernel, mode, n, M, known) => `(() => {
  state.paused = true; state.kernel = '${kernel}'; selectKernel(); state.knownPath = ${known};
  state.motionMode = '${mode}'; state.numParticles = ${n}; state.mT = ${M}; state.kBin = kBinFor(${M}); state.softEps = 0.06; state.beta = 0.8;
  state.time = 2.5; reseedAllSources();
  const pts = [];
  for (let j = 0; j < 16; j++) for (let i = 0; i < 16; i++) pts.push({ x: -1.4 + 2.8 * i / 15, y: -1.4 + 2.8 * j / 15 });
  const f = evaluateViaGPUFramebufferReadback(pts, state.mT, state.kBin, state.velMode).map((o) => [o.ex, o.ey, o.emitX, o.emitY]).flat();
  renderBulkField();
  const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  let hash = 0; for (let i = 0; i < px.length; i++) hash = (hash * 31 + px[i]) >>> 0;
  return { field: f, w, h, hash };
})()`;
const HANDOFF = (kernel, known) => `(() => {
  state.paused = true; state.kernel = '${kernel}'; selectKernel(); state.knownPath = ${known};
  state.numParticles = 4; state.motionMode = 'synchrotron'; state.beta = 0.8; state.mT = 512; state.kBin = 11;
  state.time = 2.5; state.softEps = 0.02; reseedAllSources();
  const dt = 0.45 * DT_STEP, s0 = state.sources[0];
  for (let k = 0; k < 400; k++) stepSimulation(dt);
  state.interactionTarget = 'source0'; s0.returningToOrbit = true; s0._drag = null;
  const pts = [];
  for (let j = 0; j < 10; j++) for (let i = 0; i < 10; i++) pts.push({ x: -2.2 + 4.4 * (i + 0.5) / 10, y: -1.4 + 2.8 * (j + 0.5) / 10 });
  let k = 0, rejoin = -1, mid = null, hashes = [];
  for (; k < 6000; k++) {
    if (k < 300) state.draggedSourceTarget = { x: 0.6 * Math.sin(3.1 * k * dt) + 0.3, y: 0.5 * Math.cos(2.3 * k * dt) };
    if (k === 300) state.interactionTarget = null;
    stepSimulation(dt);
    if (k === 200) mid = evaluateViaGPUFramebufferReadback(pts, 512, 11, 2).map((o) => [o.ex, o.ey]);
    if (k % 100 === 50 && k < 600) { renderBulkField(); const px = new Uint8Array(4); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); hashes.push(Array.from(px)); }
    if (k > 300 && rejoin < 0 && !s0.returningToOrbit) rejoin = k;
    if (rejoin >= 0 && k > rejoin + 200) break;
  }
  const g = evaluateViaGPUFramebufferReadback(pts, 512, 11, 2);
  return { rejoin, mid, e: g.map((o) => [o.ex, o.ey]), q: g.length, exact: Array.from(s0.emitExact.slice(0, 512)).reduce((a, b) => a + b, 0), hashes };
})()`;
const rel = (a, b) => { let n = 0, d = 0; a.forEach((v, j) => { n += (v[0] - b[j][0]) ** 2 + (v[1] - b[j][1]) ** 2; d += b[j][0] ** 2 + b[j][1] ** 2; }); return Math.sqrt(n / d); };

const out = { identity: [], drag: {}, button: null };
const A = await open(join(HERE, '..', '..', '..', 'reference', 'web_v6_1.html')), B = await open(join(HERE, '..', '..', '..', 'web', 'index.html'));
const scenes = [['synchrotron', 4, 128, true], ['synchrotron', 4, 2048, false], ['oscillator', 4, 512, true], ['swarm', 32, 128, true], ['swarm', 32, 128, false], ['swarm', 32, 2048, true]];
for (const kernel of ['v6', 'v5']) for (const [mode, n, M, known] of scenes) {
  if (kernel === 'v5' && M === 2048) continue;
  const a = await A.page.evaluate(SNAP(kernel, mode, n, M, known)), b = await B.page.evaluate(SNAP(kernel, mode, n, M, known));
  const same = a.field.length === b.field.length && a.field.every((v, i) => Object.is(v, b.field[i]));
  const row = { kernel, mode, n, M, known, fieldIdentical: same, frameIdentical: a.hash === b.hash && a.w === b.w };
  out.identity.push(row);
  console.log(`identity ${kernel} ${mode} n=${n} M=${M} ${known ? 'known' : 'live'}: field ${same ? 'bit-identical' : 'DIFFERS'}, frame ${row.frameIdentical ? 'identical' : 'DIFFERS'} (${b.w}x${b.h})`);
}
await A.page.close();

const d6k = await B.page.evaluate(HANDOFF('v6', true)), dfk = await B.page.evaluate(HANDOFF('v6f', true)), dfr = await B.page.evaluate(HANDOFF('v6f', false)), d6r = await B.page.evaluate(HANDOFF('v6', false));
out.drag = { rejoin: { v6k: d6k.rejoin, v6fk: dfk.rejoin, v6fr: dfr.rejoin, v6r: d6r.rejoin }, v6f_vs_v6_known: rel(dfk.e, d6k.e), v6f_vs_v6_ring: rel(dfr.e, d6r.e),
  v6f_known_vs_ring: rel(dfk.e, dfr.e), v6_known_vs_ring: rel(d6k.e, d6r.e), mid_v6f_vs_v6_known: rel(dfk.mid, d6k.mid), mid_v6f_vs_v6_ring: rel(dfr.mid, d6r.mid), exact: dfk.exact };
console.log('drag:', JSON.stringify(out.drag));

const seq = [];
for (let i = 0; i < 4; i++) { seq.push(await B.page.evaluate('state.kernel + "|" + document.getElementById("btn-kernel").textContent.trim()')); await B.page.evaluate(() => document.getElementById('btn-kernel').click()); }
out.button = seq;
console.log('button cycle:', seq.join(' -> '));
out.errors = { orig: A.errors, app: B.errors };
console.log('errors:', JSON.stringify(out.errors));
writeFileSync(process.argv[2] || join(HERE, 'results_identity_drag.json'), JSON.stringify(out));
await browser.close();
