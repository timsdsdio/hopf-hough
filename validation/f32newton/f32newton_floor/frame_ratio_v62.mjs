// Frame time of the app's rendered view (display path), v6 vs v6f vs v6f with F32N_NOFIX, ring and known path,
// default scene (32 ensemble sources, M = 128, softening 0.06) at beta = 0.8 (gamma 1.67: the field-floor fix's gate
// is false for every source) and beta = 0.999 (gamma 22.4: gate true). Same frame protocol as tests/gpu/web_app.mjs.
// SwiftShader: RELATIVE ONLY. Pages interleaved, ROUNDS rounds (default 5), each the mean of 3 frames after one warm-up
// frame; medians and ranges over rounds are reported.
// The NOFIX page is a temporary copy of web/index.html with `#ifdef F32N_NOFIX` -> `#if 1` (written to the OS temp
// directory and deleted afterwards). Usage, from the repository root: node validation/f32newton/f32newton_floor/frame_ratio_v62.mjs
import { launch } from './browser.mjs';
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
const HERE = dirname(fileURLToPath(import.meta.url));
const APP = join(HERE, '..', '..', '..', 'web', 'index.html');
const src = readFileSync(APP, 'utf8');
if (src.split('#ifdef F32N_NOFIX').length !== 2) throw new Error('F32N_NOFIX switch not found exactly once');
const tmp = mkdtempSync(join(tmpdir(), 'hh-nofix-'));
const NOFIX = join(tmp, 'index_nofix.html');
writeFileSync(NOFIX, src.replace('#ifdef F32N_NOFIX', '#if 1'));
const ROUNDS = +(process.env.ROUNDS || 5);
const browser = await launch();
const errors = [];
async function open(file, name) {
  const p = await browser.newPage({ viewport: { width: 640, height: 700 } });
  p.setDefaultTimeout(0);
  p.on('pageerror', (e) => errors.push(name + ': ' + e.message));
  p.on('console', (m) => { if (m.type() === 'error') errors.push(name + ': ' + m.text()); });
  await p.goto('file://' + file, { waitUntil: 'commit' });
  await p.waitForFunction(() => typeof state !== 'undefined' && state.sources && state.sources.length > 0, null, { timeout: 900000, polling: 500 });
  await p.evaluate('state.paused = true; window.requestAnimationFrame = () => 0;');
  await p.waitForTimeout(1000);
  return p;
}
const FRAMES = (kernel, known, beta) => `(() => {
  state.paused = true; state.kernel = '${kernel}'; selectKernel(); state.knownPath = ${known};
  state.numParticles = 32; state.motionMode = 'swarm'; state.beta = ${beta}; state.mT = 128; state.kBin = 9; state.softEps = 0.06;
  state.time = 2.5; reseedAllSources();
  const one = new Uint8Array(4);
  renderBulkField(); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, one);
  const t0 = performance.now();
  for (let k = 0; k < 3; k++) { renderBulkField(); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, one); }
  return (performance.now() - t0) / 3;
})()`;
try {
  const app = await open(APP, 'app'), nofix = await open(NOFIX, 'nofix');
  const variants = [['v6', app, 'v6'], ['v6f', app, 'v6f'], ['v6f_nofix', nofix, 'v6f']];
  const rows = [];
  for (const beta of [0.8, 0.999]) for (const known of [false, true]) {
    const ms = Object.fromEntries(variants.map(([n]) => [n, []]));
    for (let r = 0; r < ROUNDS; r++) for (const [n, page, k] of variants) ms[n].push(await page.evaluate(FRAMES(k, known, beta)));
    const med = (a) => { const s = a.slice().sort((x, y) => x - y), h = s.length >> 1; return s.length % 2 ? s[h] : 0.5 * (s[h - 1] + s[h]); };
    const ratios = (a, b) => a.map((v, i) => v / b[i]);
    const row = { beta, known, rounds: ROUNDS, ms,
      median: Object.fromEntries(Object.entries(ms).map(([n, a]) => [n, med(a)])),
      v6f_over_v6: { median: med(ratios(ms.v6f, ms.v6)), min: Math.min(...ratios(ms.v6f, ms.v6)), max: Math.max(...ratios(ms.v6f, ms.v6)) },
      nofix_over_v6: { median: med(ratios(ms.v6f_nofix, ms.v6)), min: Math.min(...ratios(ms.v6f_nofix, ms.v6)), max: Math.max(...ratios(ms.v6f_nofix, ms.v6)) },
      fix_over_nofix: { median: med(ratios(ms.v6f, ms.v6f_nofix)), min: Math.min(...ratios(ms.v6f, ms.v6f_nofix)), max: Math.max(...ratios(ms.v6f, ms.v6f_nofix)) } };
    rows.push(row);
    const f = (o) => `${o.median.toFixed(2)} (${o.min.toFixed(2)}–${o.max.toFixed(2)})`;
    console.log(`beta ${beta} ${known ? 'known' : 'ring '}: median v6 ${row.median.v6.toFixed(0)} ms, v6f ${row.median.v6f.toFixed(0)} ms, v6f NOFIX ${row.median.v6f_nofix.toFixed(0)} ms | v6f/v6 ${f(row.v6f_over_v6)}, NOFIX/v6 ${f(row.nofix_over_v6)}, fix/NOFIX ${f(row.fix_over_nofix)}`);
  }
  console.log('errors:', errors.length ? errors.join('; ') : 'none');
  writeFileSync(join(HERE, process.env.OUT || 'results_frame_ratio_v62.json'), JSON.stringify({ protocol: 'display path, 32 swarm sources, M=128, soft 0.06, mean of 3 frames per round, pages interleaved; SwiftShader, relative only', rows, errors }, null, 1));
} finally {
  await browser.close();
  rmSync(tmp, { recursive: true, force: true });
}
