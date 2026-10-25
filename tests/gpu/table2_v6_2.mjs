// Re-measures the paper's accuracy-across-Lorentz-factors table (preprint v6.1 Table 2, tab:gamma; docs/ACCURACY.md
// section 1) for the v6, v6f (default) and v6fp (v6f precise) kernels of the library, with the same protocol as tests/gpu/tables.mjs ('gamma'),
// plus the known-path sweep. Writes results/table2_v6_2.tsv and results/table2_v6_2.json.
// Headless Chromium + SwiftShader (IEEE float32). Accuracy only; no timings.
// Usage: npm run build && node tests/gpu/table2_v6_2.mjs
import { launch } from './browser.mjs';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
let commit = 'unknown';
try { commit = execFileSync('git', ['-C', root, 'rev-parse', '--short', 'HEAD']).toString().trim(); } catch { /* not a checkout */ }
const browser = await launch();
const page = await browser.newPage();
page.setDefaultTimeout(0);
page.on('pageerror', (e) => console.error('pageerror', e.message));
await page.setContent('<html><body></body></html>');
await page.addScriptTag({ content: readFileSync(resolve(root, 'dist/hopf-hough.iife.js'), 'utf8') });
const res = await page.evaluate(() => {
  const H = HopfHough, kb = { 128: 9, 512: 11, 2048: 13 };
  const mk = (sh) => H.gpuBackend(new H.GpuSolver(document.createElement('canvas'), sh));
  const B = { v6: mk(H.V6_SHADERS), v6f: mk(H.V6F_SHADERS), v6fp: mk(H.V6FP_SHADERS), v6k: mk(H.V6_KNOWN_PATH_SHADERS), v6fk: mk(H.V6F_KNOWN_PATH_SHADERS),
    v6fpk: mk(H.V6FP_KNOWN_PATH_SHADERS), legacy: mk(H.LEGACY_SHADERS) };
  const gl = document.createElement('canvas').getContext('webgl2'), ext = gl.getExtension('WEBGL_debug_renderer_info');
  const out = { renderer: String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)), rows: [] };
  const prod = { ...H.DEFAULT_CONFIG, softEps: 0, prefilter: false };
  const known = { ...prod, knownPath: true };
  const legacyBase = { ...H.DEFAULT_CONFIG, ...H.LEGACY_SETTINGS, softEps: 0, prefilter: false };
  for (const beta of [0.8, 0.95, 0.99, 0.999, 0.9999, 0.99999]) for (const M of [128, 512, 2048]) {
    const spec = { beta, motion: 'synchrotron', mT: M, kBin: kb[M], velMode: 2, nShock: 256 };
    const row = { beta, gamma: 1 / Math.sqrt(1 - beta * beta), M };
    for (const k of ['v6', 'v6f', 'v6fp']) {
      const p = H.runCase(spec, B[k], { base: prod });
      const pd = H.runCase(spec, B[k], { base: prod, queryMode: 'df64' });
      const q = H.runCase(spec, B[k + 'k'], { base: known });
      const qd = H.runCase(spec, B[k + 'k'], { base: known, queryMode: 'df64' });
      row[k] = { sweep: p.shockL2, sweepDf64: pd.shockL2, ring: p.bulkL2, knownSweep: q.shockL2, knownSweepDf64: qd.shockL2 };
      row.coordRound = p.coordRoundL2;
    }
    row.v5Sweep = beta <= 0.9999 ? H.runCase(spec, B.legacy, { base: legacyBase }).shockL2 : null;
    out.rows.push(row);
  }
  return out;
});
await browser.close();

// v6 must reproduce results/tables.json (the v6 kernel of main at 904bcce, unchanged; not the v6.0.0 kernel of preprint v6.1).
const ref = JSON.parse(readFileSync(resolve(root, 'results/tables.json'), 'utf8')).gamma;
let maxRel = 0;
for (const r of res.rows) {
  const t = ref.find((x) => x.beta === r.beta && x.M === r.M);
  for (const [a, b] of [[r.v6.sweep, t.prodSweep], [r.v6.sweepDf64, t.prodSweepDf64], [r.v6.ring, t.prodRing]]) maxRel = Math.max(maxRel, Math.abs(a - b) / b);
}
console.log(`v6 vs results/tables.json: max relative difference ${maxRel.toExponential(2)}`);

const f = (v) => (v === null || v === undefined ? '—' : v >= 100 ? '>100%' : (v < 0.001 ? v.toExponential(1) : v.toPrecision(2)) + '%');
const lines = [
  `# Preprint Table 2 (tab:gamma) re-measured for the v6.2 release: v6, v6f (default, F32N + F32N_NOFIX) and v6fp (v6f precise, F32N with the field-floor fix) kernels, library readback (tests/gpu/table2_v6_2.mjs), commit ${commit} + working tree`,
  `# Renderer: ${res.renderer} (IEEE float32 software rendering). Accuracy only, no timings.`,
  '# Relative L2 error vs the exact LW field, no softening. sweep: 256 observers across the synchrotron pulse at R_ret = 1.15 (f32 = rounded to float32 and',
  '# compared at the rounded point; df64 = double-float observer coordinates); ring: 64 observers at radius 1.15. Ring-buffer history (16 steps), as the paper.',
  '# known_*: the same sweep with known-path shells (exact knots, C2 quintic). v5: the version-5 algorithm (deposited shader). At gamma = 224 only df64 is meaningful.',
  `# v6 columns reproduce results/tables.json to ${maxRel.toExponential(1)} relative (unchanged kernel).`,
  ['beta', 'gamma', 'M', 'v6_sweep_f32', 'v6_sweep_df64', 'v6_ring', 'v6f_sweep_f32', 'v6f_sweep_df64', 'v6f_ring', 'v6fp_sweep_f32', 'v6fp_sweep_df64', 'v6fp_ring',
    'v5_sweep_f32', 'coord_rounding', 'v6_known_sweep_f32', 'v6_known_sweep_df64', 'v6f_known_sweep_f32', 'v6f_known_sweep_df64', 'v6fp_known_sweep_f32', 'v6fp_known_sweep_df64'].join('\t'),
  ...res.rows.map((r) => [r.beta, r.gamma.toFixed(1), r.M, f(r.v6.sweep), f(r.v6.sweepDf64), f(r.v6.ring), f(r.v6f.sweep), f(r.v6f.sweepDf64), f(r.v6f.ring),
    f(r.v6fp.sweep), f(r.v6fp.sweepDf64), f(r.v6fp.ring), f(r.v5Sweep), f(r.coordRound), f(r.v6.knownSweep), f(r.v6.knownSweepDf64),
    f(r.v6f.knownSweep), f(r.v6f.knownSweepDf64), f(r.v6fp.knownSweep), f(r.v6fp.knownSweepDf64)].join('\t')),
];
writeFileSync(resolve(root, 'results/table2_v6_2.tsv'), lines.join('\n') + '\n');
writeFileSync(resolve(root, 'results/table2_v6_2.json'), JSON.stringify({ commit, ...res, v6VsTablesJsonMaxRel: maxRel }, null, 2) + '\n');
console.log(lines.slice(6).join('\n'));
