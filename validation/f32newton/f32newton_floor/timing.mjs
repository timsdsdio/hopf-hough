// Frame time of the rendered view, 32 sources (swarm), v6f before vs after (A: fix on all paths, B: fix off on the
// display path), display path and full path (DIAG_FULL). SwiftShader: RELATIVE ONLY. Pages interleaved, 3 rounds,
// median of 3 frames per round. Also: rendered bytes of the display path, after vs before.
import { launch } from './browser.mjs';
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
const HERE = dirname(fileURLToPath(import.meta.url));
const browser = await launch();
const names = (process.env.PAGES || 'before,A,B').split(',');
const pages = {}; const errors = [];
for (const n of names) {
  const p = await browser.newPage({ viewport: { width: 640, height: 700 } });
  p.on('pageerror', (e) => errors.push(n + ': ' + e.message));
  p.on('console', (m) => { if (m.type() === 'error') errors.push(n + ': ' + m.text()); });
  await p.goto('file://' + join(HERE, `timing_${n}.html`), { waitUntil: 'commit' });
  await p.waitForFunction(() => typeof state !== 'undefined' && state.sources && state.sources.length > 0, null, { timeout: 900000, polling: 500 });
  await p.evaluate('state.paused = true; window.requestAnimationFrame = () => 0;'); await p.waitForTimeout(1000); pages[n] = p;
}
const FRAMES = (M, known, full, beta) => `(() => {
  window.DIAG_FULL = ${full};
  state.paused = true; state.kernel = 'v6f'; selectKernel(); state.knownPath = ${known};
  state.numParticles = 32; state.motionMode = 'swarm'; state.beta = ${beta}; state.mT = ${M}; state.kBin = kBinFor(${M}); state.softEps = 0.06;
  state.time = 2.5; reseedAllSources();
  const one = new Uint8Array(4);
  renderBulkField(); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, one);
  const ts = [];
  for (let k = 0; k < 3; k++) { const t0 = performance.now(); renderBulkField(); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, one); ts.push(performance.now() - t0); }
  const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  window.DIAG_FULL = false; ts.sort((a, b) => a - b);
  return { ms: ts[1], w, h, px: Array.from(px) };
})()`;
const scenes = process.env.QUICK ? [[128, true, 0.8], [512, false, 0.8], [512, true, 0.9999]] : [[128, true, 0.8], [128, false, 0.8], [512, true, 0.8], [512, false, 0.8], [2048, false, 0.8], [512, false, 0.999], [512, true, 0.9999]];
const rows = [];
for (const full of (process.env.QUICK ? [false] : [false, true])) for (const [M, known, beta] of scenes) {
  const ms = Object.fromEntries(names.map((n) => [n, []])), px = {};
  for (let r = 0; r < 3; r++) for (const n of names) { const o = await pages[n].evaluate(FRAMES(M, known, full, beta)); ms[n].push(o.ms); if (r === 0) px[n] = o.px; }
  const med = (a) => a.slice().sort((x, y) => x - y)[1];
  const cmp = (a, b) => { let dmax = 0, nd = 0; for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d) nd++; dmax = Math.max(dmax, d); } return { dmax, frac: nd / a.length }; };
  const row = { M, known, beta, path: full ? 'full' : 'display' };
  for (const n of names) { row[n] = med(ms[n]); if (n !== 'before') row['px_' + n] = cmp(px.before, px[n]); }
  rows.push(row);
  console.log(`${row.path} M${M} ${known ? 'known' : 'live'} beta ${beta}: before ${row.before.toFixed(0)} ms` + names.filter((n) => n !== 'before').map((n) => `, ${n} ${row[n].toFixed(0)} (${(row[n] / row.before).toFixed(3)}; px max ${row['px_' + n].dmax}/255 on ${(100 * row['px_' + n].frac).toFixed(3)}%)`).join(''));
}
console.log('errors:', errors.length ? errors.join('; ') : 'none');
writeFileSync(join(HERE, process.env.OUT || 'results_timing.json'), JSON.stringify({ rows, errors }, null, 1));
await browser.close();
