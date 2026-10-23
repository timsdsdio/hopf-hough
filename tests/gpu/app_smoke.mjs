// Loads the built app, checks for errors, exercises scenarios and the accuracy check, saves screenshots.
import { launch } from './browser.mjs';
const shots = process.argv[2] || '/tmp';
const browser = await launch();
const page = await browser.newPage({ viewport: { width: 1100, height: 800 }, deviceScaleFactor: 0.5 });
const errors = [];
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
page.on('request', (r) => { if (!r.url().startsWith('file:') && !r.url().startsWith('blob:') && !r.url().startsWith('data:')) errors.push('external request: ' + r.url()); });
await page.goto('file://' + new URL('../../dist/hopf-hough.html', import.meta.url).pathname + '?n=2&mt=512');
await page.waitForTimeout(4000);
const out = {};
out.glError = await page.$eval('#gl-error', (e) => e.classList.contains('hidden') ? null : e.textContent);
await page.screenshot({ path: `${shots}/01_default.png` });
await page.click('#btn-check'); await page.waitForTimeout(500);
out.checkEnsemble = await page.$eval('#stat-check', (e) => e.textContent);
out.probe = await page.$eval('#probe-err', (e) => e.innerText);
for (const sc of ['orbit', 'oscillator', 'linear', 'hyperbolic']) {
  await page.selectOption('#sel-scenario', sc); await page.waitForTimeout(2500);
  await page.click('#btn-check'); await page.waitForTimeout(400);
  out['check_' + sc] = await page.$eval('#stat-check', (e) => e.textContent);
  out['err_' + sc] = await page.$eval('#scenario-error', (e) => e.classList.contains('hidden') ? null : e.textContent);
  await page.screenshot({ path: `${shots}/02_${sc}.png` });
}
await page.selectOption('#sel-scenario', 'custom'); await page.waitForTimeout(300);
await page.click('#btn-apply-custom'); await page.waitForTimeout(2500);
await page.click('#btn-check'); await page.waitForTimeout(400);
out.check_custom = await page.$eval('#stat-check', (e) => e.textContent);
await page.screenshot({ path: `${shots}/03_custom.png` });
await page.fill('#in-x', '3*cos(3*t)'); await page.click('#btn-apply-custom'); await page.waitForTimeout(500);
out.superluminalRefused = await page.$eval('#scenario-error', (e) => e.textContent);
await page.selectOption('#sel-channel', '3'); await page.selectOption('#sel-scenario', 'orbit'); await page.waitForTimeout(2000);
await page.screenshot({ path: `${shots}/04_bz.png` });
out.errors = errors;
console.log(JSON.stringify(out, null, 2));
await browser.close();
