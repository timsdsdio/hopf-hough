// The WebGL2 kernel on the same problems and grid as run_benchmark.py (here: SwiftShader, i.e. the CPU).
import { launch } from '../../tests/gpu/browser.mjs';
import { readFileSync, writeFileSync } from 'node:fs';
const root = new URL('../../', import.meta.url).pathname;
const N = +(process.argv[2] || 256);
const browser = await launch();
const page = await browser.newPage();
page.setDefaultTimeout(0);
await page.setContent('<html><body><canvas id="c" width="16" height="16"></canvas></body></html>');
await page.addScriptTag({ content: readFileSync(root + 'dist/hopf-hough.iife.js', 'utf8') });
const res = await page.evaluate((N) => {
  const H = HopfHough, solver = new H.GpuSolver(document.getElementById('c')), gl = solver.gl;
  const out = { renderer: String(gl.getParameter(gl.getExtension('WEBGL_debug_renderer_info').UNMASKED_RENDERER_WEBGL)), rows: [] };
  const pts = [];
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) pts.push({ x: -3 + 6 * i / (N - 1), y: -3 + 6 * j / (N - 1) });
  for (const [n, beta] of [[1, 0.9], [1, 0.99], [1, 0.999], [1, 0.9999], [16, 0.9]]) for (const M of [512, 2048]) {
    const c = 2.35, cfg = { ...H.DEFAULT_CONFIG, c, mT: M, kBin: M === 512 ? 11 : 13 };
    const trs = Array.from({ length: n }, (_, k) => H.circularOrbit({ radius: 0.85, beta, c, phase: 2 * Math.PI * k / n }));
    const sim = new H.Simulation(cfg, trs, 2.5);
    solver.upload(sim.shellHistories, M);
    solver.readFull(pts.slice(0, 256), cfg); // warm-up
    const ts = [];
    let f;
    for (let r = 0; r < 3; r++) { const t0 = performance.now(); f = solver.readFull(pts, cfg); ts.push((performance.now() - t0) / 1000); }
    let num = 0, den = 0;
    pts.forEach((p, i) => {
      if (Math.abs(Math.hypot(p.x, p.y) - 0.85) <= 0.05) return;
      const e = H.exactField(trs, Math.fround(p.x), Math.fround(p.y), sim.time, { c, softEps: 0 });
      num += (f[i].ex - e.ex) ** 2 + (f[i].ey - e.ey) ** 2; den += e.mag ** 2;
    });
    ts.sort((a, b) => a - b);
    out.rows.push({ sources: n, beta, M, seconds: ts[1], rel_l2: Math.sqrt(num / den) });
  }
  return out;
}, N);
writeFileSync(root + 'validation/benchmarks/results_webgl.json', JSON.stringify(res, null, 2));
console.log(res.renderer);
for (const r of res.rows) console.log(`N=${r.sources} beta=${r.beta} M=${r.M}: ${r.seconds.toFixed(3)} s err=${r.rel_l2.toExponential(2)}`);
await browser.close();
