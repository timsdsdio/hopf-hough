// Runs the ORIGINAL solver's verification suite and saves its structured results.
import { launch } from '../tests/gpu/browser.mjs';
import { writeFileSync } from 'node:fs';
const browser = await launch();
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('pageerror', e.message));
await page.goto('file://' + new URL('./original_instrumented.html', import.meta.url).pathname);
await page.waitForFunction(() => window.__suite, null, { timeout: 900000, polling: 1000 });
const suite = await page.evaluate(() => window.__suite);
writeFileSync(new URL('./baseline_suite.json', import.meta.url), JSON.stringify(suite, null, 2));
console.log(JSON.stringify(suite.rows.map(r => [r.beta, r.r128Norm.shockL2, r.r5124Vel.shockL2, r.r20484Vel.shockL2])));
console.log('far', suite.compFar45.bulkL2, suite.unifFar45.bulkL2, 'grids', suite.comp128Reach220.shockL2, suite.log128Reach220.shockL2, suite.pow128Reach220.shockL2, suite.unif128Reach220.shockL2);
await browser.close();
