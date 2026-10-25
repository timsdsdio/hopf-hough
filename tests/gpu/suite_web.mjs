// Runs the verification suite of the author's app (web/index.html, "Run Evolved GPU Suite") headless for each kernel
// (v6f, v6fp, v6, v5) and writes results/suite_swiftshader_v6_2.tsv. The suite's computation is the app's own; only its
// number formatting is widened (toFixed(2)/(4)/(5) -> 3 significant digits) so the table keeps small values.
// Usage: node tests/gpu/suite_web.mjs [out.tsv]
import { launch } from './browser.mjs';
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const out = process.argv[2] || resolve(root, 'results/suite_swiftshader_v6_2.tsv');
let commit = 'unknown';
try { commit = execFileSync('git', ['-C', root, 'rev-parse', '--short', 'HEAD']).toString().trim(); } catch { /* not a checkout */ }

const browser = await launch();
const page = await browser.newPage({ viewport: { width: 640, height: 700 } });
page.setDefaultTimeout(0);
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto('file://' + resolve(root, 'web/index.html'), { waitUntil: 'commit' });
await page.waitForFunction(() => typeof state !== 'undefined' && state.sources && state.sources.length > 0, null, { timeout: 300000, polling: 500 });
await page.evaluate('state.paused = true');
await page.waitForTimeout(500);
const renderer = await page.evaluate(() => { const e = gl.getExtension('WEBGL_debug_renderer_info'); return String(e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)); });
const ua = await page.evaluate(() => navigator.userAgent);
// Same function, wider number formatting (redefined in the page's global scope).
await page.evaluate(() => {
  const src = runAutomatedGPUVerificationSuite.toString()
    .replace(/\.toFixed\((?:1|2|3|4|5)\)\}%/g, '.toPrecision(3)}%');
  (0, eval)(src);
});

const rows = [];
const notes = [];
for (const kernel of ['v6f', 'v6fp', 'v6', 'v5']) {
  const t0 = Date.now();
  const res = await page.evaluate((kernel) => {
    state.paused = true; state.kernel = kernel; selectKernel(); reseedAllSources();
    runAutomatedGPUVerificationSuite();
    const el = document.getElementById('suite-output');
    const head = el.querySelector('div').textContent.trim();
    const trs = [...el.querySelectorAll('tr')].slice(1).map((tr) => [...tr.querySelectorAll('td')].map((td) => td.textContent.trim()));
    const tail = el.querySelectorAll('div')[1].innerText.trim();
    return { head, trs, tail, active: state.kernel };
  }, kernel);
  console.log(`${kernel}: ${res.head} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
  if (res.active !== kernel || !res.head.includes(`Kernel: ${kernel} `)) throw new Error(`suite label does not show kernel ${kernel}: ${res.head}`);
  for (const r of res.trs) {
    const m = /^([\d.]+)c \(([\d.]+)\)$/.exec(r[0]);
    const pos = /^(\S+) \((\d+)×\)$/.exec(r[1]);
    rows.push([kernel, m[1], m[2], pos[1], pos[2], ...r.slice(2)]);
  }
  const conv = res.tail.split('\n').filter((l) => /Spline Float64|GPU Float32 vs Exact|Compactified L|Compactified:/.test(l)).map((l) => l.replace(/\s+/g, ' ').trim());
  notes.push(`# ${kernel}: ` + conv.join(' / '));
}
await browser.close();

const lines = [
  `# Author's app (web/index.html), "Run Evolved GPU Suite" for each kernel, commit ${commit} + working tree (release/v6.2)`,
  `# Headless Chromium (${ua.match(/Chrome\/[\d.]+/)?.[0] || 'Chromium'}), ${renderer}. SwiftShader is IEEE float32 software rendering; no timings here.`,
  '# Relative L2 error vs the exact LW field at the same float32 points, synchrotron pulse sweep (256 observers) and ring (64 observers, "bulk").',
  '# Ring-buffer (live) history, as the app suite. Number formatting widened to 3 significant digits (tests/gpu/suite_web.mjs); computation unchanged.',
  '# pos_err_128: max |emission point GPU - exact| on the ring at M = 128; ratio: M = 128 / M = 512.',
  ['kernel', 'beta', 'gamma', 'pos_err_128', 'pos_ratio_128_512', 'bulk_128', 'pulse_128_norm', 'pulse_512_4vel', 'pulse_2048_4vel'].join('\t'),
  ...rows.map((r) => r.join('\t')),
  '# gamma = 70.7 convergence (M = 512 | 1024 | 2048) and the v5-era grid / domain tests, as printed by the suite:',
  ...notes,
  `# console errors: ${errors.length ? errors.join('; ') : 'none'}`,
];
writeFileSync(out, lines.join('\n') + '\n');
console.log('wrote', out);
