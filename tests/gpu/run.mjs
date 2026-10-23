// GPU test suite (headless Chromium, software WebGL2). Run after `npm run build`.
//  1. Legacy shader reproduces the original solver's published suite exactly.
//  2. Production shader meets accuracy thresholds against the exact reference.
//  3. The single-file app loads without errors or external requests.
//  4. The author's app (web/index.html, site root): v5 kernel bit-identical to the original, v6 kernel accuracy.
import { launch } from './browser.mjs';
import { runWebAppChecks } from './web_app.mjs';
import { readFileSync } from 'node:fs';
const root = new URL('../../', import.meta.url).pathname;
const bundle = readFileSync(root + 'dist/hopf-hough.iife.js', 'utf8');
const base = JSON.parse(readFileSync(root + 'reference/baseline_suite.json', 'utf8'));
let failures = 0;
const check = (ok, msg) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) failures++; };

const browser = await launch();
try {
  const page = await browser.newPage();
  page.setDefaultTimeout(0);
  await page.setContent('<html><body><canvas id="a" width="16" height="16"></canvas><canvas id="b" width="16" height="16"></canvas></body></html>');
  await page.addScriptTag({ content: bundle });

  // 1. Legacy regression (a representative subset keeps CI fast; the full run is tests/gpu/legacy_regression.mjs).
  const legacy = await page.evaluate(() => {
    const H = HopfHough, s = new H.GpuSolver(document.getElementById('a'), H.LEGACY_SHADERS), B = H.gpuBackend(s), o = { base: H.LEGACY_SETTINGS };
    return {
      r512_0999: H.runCase({ beta: 0.999, motion: 'synchrotron', mT: 512, kBin: 11, velMode: 2, nShock: 256 }, B, o),
      r128_09999: H.runCase({ beta: 0.9999, motion: 'synchrotron', mT: 128, kBin: 9, velMode: 1, nShock: 256 }, B, o),
      comp: H.runCase({ beta: 0.99, motion: 'synchrotron', mT: 128, kBin: 9, velMode: 2, nShock: 256, rMax: 220, rEval: 1.15, gridMode: 'compactified' }, B, o),
    };
  });
  const same = (a, b) => ['bulkL2', 'shockL2', 'cpuF32L2', 'cpuF64L2'].every((k) => Math.abs(a[k] - b[k]) <= 1e-9 * Math.max(1, Math.abs(b[k])));
  check(same(legacy.r512_0999, base.rows[3].r5124Vel), `legacy shader reproduces original (β=0.999, M=512): ${legacy.r512_0999.shockL2.toFixed(4)}%`);
  check(same(legacy.r128_09999, base.rows[4].r128Norm), `legacy shader reproduces original (β=0.9999, M=128): ${legacy.r128_09999.shockL2.toFixed(3)}%`);
  check(same(legacy.comp, base.comp128Reach220), `legacy shader reproduces original (grid comparison): ${legacy.comp.shockL2.toFixed(4)}%`);

  // 2. Production accuracy thresholds (exact LW, no softening, float32-rounded observers). Tight enough that the
  //    v6.0.0 kernel (cross-product radiation term, open Newton bracket: 0.0095% / 0.069% at M = 512) fails them.
  const prod = await page.evaluate(() => {
    const H = HopfHough, s = new H.GpuSolver(document.getElementById('b'), H.PRODUCTION_SHADERS), B = H.gpuBackend(s);
    const o = { base: { ...H.DEFAULT_CONFIG, softEps: 0, prefilter: false } };
    const run = (beta, mT, kBin) => H.runCase({ beta, motion: 'synchrotron', mT, kBin, velMode: 2, nShock: 256 }, B, o);
    return { g22: run(0.999, 512, 11), g70: run(0.9999, 512, 11), g70c: run(0.9999, 128, 9), g7: run(0.99, 128, 9) };
  });
  check(prod.g7.shockL2 < 0.045, `production γ=7.1,  M=128: ${prod.g7.shockL2.toPrecision(3)}% < 0.045%`);
  check(prod.g22.shockL2 < 0.009, `production γ=22.4, M=512: ${prod.g22.shockL2.toPrecision(3)}% < 0.009%`);
  check(prod.g70.shockL2 < 0.05, `production γ=70.7, M=512: ${prod.g70.shockL2.toPrecision(3)}% < 0.05%`);
  check(prod.g70c.shockL2 < 0.5, `production γ=70.7, M=128: ${prod.g70c.shockL2.toPrecision(3)}% < 0.5%`);

  // 3. App smoke test.
  const app = await browser.newPage({ viewport: { width: 900, height: 700 }, deviceScaleFactor: 0.5 });
  const errs = [];
  app.on('pageerror', (e) => errs.push(e.message));
  app.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
  app.on('request', (r) => { if (!/^(file|blob|data):/.test(r.url())) errs.push('external request ' + r.url()); });
  await app.goto('file://' + root + 'dist/hopf-hough.html?n=2&scenario=orbit&beta=0.9');
  await app.waitForTimeout(3000);
  const glErr = await app.$eval('#gl-error', (e) => (e.classList.contains('hidden') ? null : e.textContent));
  check(!glErr, 'app initialises WebGL2' + (glErr ? ': ' + glErr : ''));
  await app.evaluate(() => window.hopfHough.checkAccuracy());
  const acc = await app.$eval('#stat-check', (e) => e.textContent);
  check(parseFloat(acc) < 0.1, `app accuracy check vs reference: ${acc}`);
  check(errs.length === 0, 'app has no errors or external requests' + (errs.length ? ': ' + errs.join('; ') : ''));
} finally {
  await browser.close();
}

// 4. The author's app with the v6 kernel.
if (!(await runWebAppChecks((line) => console.log(line.replace(/^ok   /, 'ok   ').replace(/^FAIL /, 'FAIL '))))) failures++;
console.log(failures ? `\n${failures} check(s) failed` : '\nall GPU checks passed');
process.exit(failures ? 1 : 0);
