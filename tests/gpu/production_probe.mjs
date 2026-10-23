// Production shader: compile check, then sweep errors for each switch combination.
import { launch } from './browser.mjs';
import { readFileSync } from 'node:fs';
const bundle = process.argv[2];
const browser = await launch();
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('pageerror', e.message));
page.on('console', (m) => console.log('[page]', m.text()));
await page.setContent('<html><body><canvas id="c" width="16" height="16"></canvas></body></html>');
await page.addScriptTag({ content: readFileSync(bundle, 'utf8') });
const res = await page.evaluate(() => {
  const solver = new HopfHough.GpuSolver(document.getElementById('c'), HopfHough.PRODUCTION_SHADERS);
  const B = HopfHough.gpuBackend(solver);
  const variants = {
    'legacy switches (eps .06)': { ...HopfHough.LEGACY_SETTINGS },
    'df64 only': { ...HopfHough.LEGACY_SETTINGS, precision: 1 },
    'df64 + sharp cert': { ...HopfHough.LEGACY_SETTINGS, precision: 1, certMode: 1 },
    'production (eps .06)': { softEps: 0.06 },
    'production (eps 0)': { softEps: 0 },
  };
  const out = {};
  for (const [name, base] of Object.entries(variants)) {
    const row = {};
    for (const [b, M, kb, vm] of [[0.999, 512, 11, 2], [0.9999, 128, 9, 1], [0.9999, 512, 11, 2], [0.9999, 2048, 13, 2]]) {
      const r = HopfHough.runCase({ beta: b, motion: 'synchrotron', mT: M, kBin: kb, velMode: vm, nShock: 256 }, B, { base });
      row[`γ=${(1/Math.sqrt(1-b*b)).toFixed(1)} M=${M}`] = [+r.shockL2.toPrecision(3), +r.bulkL2.toPrecision(3)];
    }
    out[name] = row;
  }
  return out;
});
console.log('sweep L2 % / bulk L2 % (f32 query metric)');
for (const [k, v] of Object.entries(res)) console.log(k.padEnd(28), JSON.stringify(v));
await browser.close();
