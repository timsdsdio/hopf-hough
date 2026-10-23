// Validity indicator on dragged (live) histories: the acceleration terms must flag the field errors that
// acceleration jumps cause, and must not light up smooth presets. Reduced version of the tuning harness
// (two shell counts, one snapshot, 64x64 observers); see docs/ACCURACY.md for the full numbers.
import { launch } from './browser.mjs';
import { readFileSync } from 'node:fs';
const root = new URL('../../', import.meta.url).pathname;

export async function runDragMaskChecks(log) {
  const bundle = readFileSync(root + 'dist/hopf-hough.iife.js', 'utf8');
  const browser = await launch();
  let ok = true;
  const check = (cond, msg) => { log(`${cond ? 'ok  ' : 'FAIL'} ${msg}`); if (!cond) ok = false; };
  try {
    const page = await browser.newPage();
    page.setDefaultTimeout(0);
    await page.setContent('<html><body></body></html>');
    await page.addScriptTag({ content: bundle });
    const res = await page.evaluate(() => {
      const H = window.HopfHough, C = 2.35, DT = 0.00675;
      const solver = new H.GpuSolver(document.createElement('canvas'));
      // Pointer: rest, jump, hold, jump, rapid reversals, fast circle (the app's drag rules, consistent kinematics).
      const ev = []; let T = 0;
      const seg = (d, f) => { ev.push([T, T + d, f]); T += d; };
      seg(0.6, () => [0, 0]); seg(0.5, () => [1.6, 0.4]); seg(0.4, () => [1.6, 0.4]); seg(0.6, () => [-1.2, -0.8]);
      for (let k = 0; k < 6; k++) seg(0.12, () => (k % 2 ? [0.9, 0.2] : [-0.9, -0.2]));
      seg(1.2, (t) => [1.1 * Math.cos(5 * t), 1.1 * Math.sin(5 * t)]);
      const target = (t) => { for (const [a, b, f] of ev) if (t >= a && t < b) return f(t - a); return [0, 0]; };
      const N = 64, pts = [];
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) pts.push({ x: -3 + 6 * (i + 0.5) / N, y: -3 + 6 * (j + 0.5) / N });
      const ringTraj = (sim, s) => ({ kinematics(t) { const h = s.ring.sample((sim.time - t) / sim.config.dt, C * sim.config.dt); return { ...h, charge: s.current.charge }; } });
      function measure(sim, cfg) {
        solver.upload(sim.shellHistories, cfg.mT);
        const f = solver.readFull(pts, cfg);
        const refs = sim.sources.map((s) => ringTraj(sim, s));
        let bad = 0, caught = 0, flagged = 0, n = 0;
        pts.forEach((p, k) => {
          if (f[k].quality < 0) return;
          if (sim.sources.some((s) => Math.hypot(p.x - s.current.x, p.y - s.current.y) < 0.08)) return;
          const e = H.exactField(refs, Math.fround(p.x), Math.fround(p.y), sim.time, { c: C, softEps: 0 });
          const err = Math.hypot(f[k].ex - e.ex, f[k].ey - e.ey) / e.mag;
          const fl = f[k].quality > 1.5;
          n++; if (fl) flagged++;
          if (err > 1e-2) { bad++; if (fl) caught++; }
        });
        return { recall: bad ? caught / bad : 1, bad, flagged: flagged / n };
      }
      const out = {};
      for (const [M, kBin] of [[128, 9], [512, 11]]) {
        const cfg = { ...H.DEFAULT_CONFIG, c: C, dt: DT, mT: M, kBin, softEps: 0, prefilter: false };
        const sim = new H.Simulation(cfg, [H.linearMotion({ beta: 0, c: C, x0: 0, y0: 0 })], 2.5);
        sim.override = (i, t, dt, prev) => {
          const [gx, gy] = target(t - 2.5);
          let tbx = ((gx - prev.x) * 8.5) / C, tby = ((gy - prev.y) * 8.5) / C;
          const tb = Math.hypot(tbx, tby); if (tb > 0.9) { tbx *= 0.9 / tb; tby *= 0.9 / tb; }
          let nbx = prev.bx + 0.085 * (tbx - prev.bx), nby = prev.by + 0.085 * (tby - prev.by);
          const amax = 1.15 * C * dt, dbx = nbx - prev.bx, dby = nby - prev.by, db = Math.hypot(dbx, dby);
          if (db > amax) { nbx = prev.bx + (dbx * amax) / db; nby = prev.by + (dby * amax) / db; }
          return { x: prev.x + 0.5 * (prev.bx + nbx) * C * dt, y: prev.y + 0.5 * (prev.by + nby) * C * dt, bx: nbx, by: nby,
            bdx: (nbx - prev.bx) / (C * dt), bdy: (nby - prev.by) / (C * dt), charge: 1 };
        };
        while (sim.time - 2.5 < 0.62 * T) sim.step(DT, true);
        sim.repopulate();
        out['drag' + M] = measure(sim, cfg);
        const orbit = new H.Simulation(cfg, [H.circularOrbit({ radius: 0.999 * C / 1.6, beta: 0.999, c: C })], 2.5);
        out['orbit' + M] = measure(orbit, cfg);
        const osc = new H.Simulation(cfg, [H.harmonicOscillator({ amplitude: 0.6, omega: 0.05 * C / 0.6, c: C })], 2.5);
        out['rest' + M] = measure(osc, cfg);
      }
      return out;
    });
    for (const M of [128, 512]) {
      const d = res['drag' + M], o = res['orbit' + M], r = res['rest' + M];
      check(d.bad > 20 && d.recall >= 0.9, `validity mask catches drag errors, M=${M}: ${(100 * d.recall).toFixed(0)}% of ${d.bad} pixels with error > 1% flagged`);
      check(r.flagged <= 0.01, `validity mask quiet for a charge near rest, M=${M}: ${(100 * r.flagged).toFixed(1)}% flagged`);
      if (M === 512) check(o.flagged <= 0.05, `validity mask quiet on a smooth orbit (γ = 22), M=${M}: ${(100 * o.flagged).toFixed(1)}% flagged`);
    }
  } finally {
    await browser.close();
  }
  return ok;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const ok = await runDragMaskChecks(console.log);
  process.exit(ok ? 0 : 1);
}
