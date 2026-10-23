// The ported GPU path (legacy shader) must reproduce the original suite exactly under SwiftShader.
import { launch } from './browser.mjs';
import { readFileSync } from 'node:fs';
const bundle = process.argv[2];
const base = JSON.parse(readFileSync(new URL('../../reference/baseline_suite.json', import.meta.url)));
const browser = await launch();
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('pageerror', e.message));
await page.setContent('<html><body><canvas id="c" width="16" height="16"></canvas></body></html>');
await page.addScriptTag({ content: readFileSync(bundle, 'utf8') });
const res = await page.evaluate(() => {
  const solver = new HH.GpuSolver(document.getElementById('c'));
  return HH.runSuite(HH.gpuBackend(solver));
});
const KEYS = ['bulkL2', 'shockL2', 'cpuF32L2', 'cpuF64L2', 'coordRoundL2', 'gpuCpuRelL2', 'posErr'];
let n = 0, bad = 0;
const cmp = (a, b, l) => { for (const k of KEYS) { n++; if (a[k] !== b[k]) { bad++; console.log('MISMATCH', l, k, a[k], b[k]); } } };
base.rows.forEach((r, i) => { cmp(res.rows[i].r128Norm, r.r128Norm, 'r128 ' + r.beta); cmp(res.rows[i].r5124Vel, r.r5124Vel, 'r512 ' + r.beta); cmp(res.rows[i].r20484Vel, r.r20484Vel, 'r2048 ' + r.beta); });
for (const k of ['diag1024', 'comp128Reach220', 'log128Reach220', 'pow128Reach220', 'unif128Reach220', 'compFar45', 'unifFar45']) cmp(res[k], base[k], k);
console.log(`${n - bad}/${n} values identical to the original solver`);
await browser.close();
process.exit(bad ? 1 : 0);
