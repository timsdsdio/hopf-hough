// Export one pulse-sweep case for the numpy emulation (emu_field.py): source-0 knots (stored float32 and float64 truth),
// observers (df64 / f32), GPU field (layout 0), GPU root debug (layout 2: r1, dRSub, kappa, bMag), exact field and
// exact retarded kinematics. Usage: node export.mjs <app.html> <out.json> <kernel> [beta,M,mode,obs ...]
import { launch } from './browser.mjs';
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const [app, outFile, kernel, ...caseArgs] = process.argv.slice(2);
const cases = caseArgs.map((s) => { const [b, M, mode, obs] = s.split(','); return { beta: +b, M: +M, mode, obs }; });
const browser = await launch();
const page = await browser.newPage({ viewport: { width: 640, height: 700 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto('file://' + resolve(app), { waitUntil: 'commit' });
await page.waitForFunction(() => typeof state !== 'undefined' && state.sources && state.sources.length > 0, null, { timeout: 300000, polling: 500 });
await page.evaluate('state.paused = true');
await page.waitForTimeout(500);

const CASE = (c) => `(() => {
  state.paused = true; state.kernel = '${kernel}'; selectKernel();
  state.numParticles = 1; state.softEps = 0; state.knownPath = ${c.mode === 'known'};
  const M = ${c.M}, beta = ${c.beta}, kBin = kBinFor(M); state.time = 2.40;
  seedExactRetardedHistory(state.sources[0], 0, 1, 2.40, beta, 'synchrotron', M, 2);
  for (let k = 0; k < 16; k++) stepSimulation(DT_STEP, beta, 'synchrotron', M, 2);
  for (let k = 0; k < ${c.mode === 'live' ? 600 : 0}; k++) stepSimulation(0.45 * DT_STEP, beta, 'synchrotron', M, 2);
  if (${c.mode === 'known'}) { state.sources[0].dragWins = []; populateKnownPathShells(state.sources[0], 0, 1, M, 2, DT_STEP, beta, 'synchrotron'); }
  const sw = getPulseCrossingSweepPoints(1.15, state.time, beta, 256).pts;
  const pts = sw.map((p) => ${c.obs === 'df64'} ? { x: p.x64, y: p.y64 } : { x: p.x, y: p.y });
  const g = evaluateViaGPUFramebufferReadback(pts, M, kBin, 2);
  gl.useProgram(glProg); gl.uniform1i(glUniforms.readbackLayout, 2);
  const dbg = evaluateViaGPUFramebufferReadback(pts, M, kBin, 2, true);
  gl.useProgram(glProg); gl.uniform1i(glUniforms.readbackLayout, 0);
  // GPU readback inside evaluateViaGPU... returns ex, ey, emitX, emitY = the four channels
  const src = state.sources[0], A = (a) => Array.from(a.slice(0, M));
  const knots = {}; for (const k of ['emitX','emitY','emitXlo','emitYlo','emitUx','emitUy','emitBdx','emitBdy','emitR','emitJx','emitJy','emitExact',
      'emitX64','emitY64','emitUx64','emitUy64','emitBdx64','emitBdy64','emitR64']) knots[k] = A(src[k]);
  // exact kinematics at the stored (float32) shell radius: u, acc, position in float64
  const ke = { ux: [], uy: [], bdx: [], bdy: [], x: [], y: [] };
  for (let m = 0; m < M; m++) { const k = getExactTrajectoryKinematics(0, 1, state.time - src.emitR[m] / C_LIGHT, beta, 'synchrotron');
    const pv = packVelocityForStorage(k.bx, k.by, 2); ke.ux.push(pv.ux64); ke.uy.push(pv.uy64); ke.bdx.push(k.bdx); ke.bdy.push(k.bdy); ke.x.push(k.x); ke.y.push(k.y); }
  knots.exactAtR32 = ke;
  const ex = [], ret = [];
  for (const p of pts) {
    const e = evaluateExactAnalyticalLW(p.x, p.y, state.time, beta, 'synchrotron'); ex.push([e.ex, e.ey]);
    // exact retarded kinematics (same solve as evaluateExactAnalyticalLW)
    let tLow = state.time - 260 / C_LIGHT, tHigh = state.time;
    for (let b = 0; b < 32; b++) { const tm = 0.5 * (tLow + tHigh); const k = getExactTrajectoryKinematics(0, 1, tm, beta, 'synchrotron');
      if (C_LIGHT * (state.time - tm) - Math.hypot(p.x - k.x, p.y - k.y) > 0) tLow = tm; else tHigh = tm; }
    let tR = 0.5 * (tLow + tHigh);
    for (let it = 0; it < 12; it++) { const k = getExactTrajectoryKinematics(0, 1, tR, beta, 'synchrotron');
      const dx = p.x - k.x, dy = p.y - k.y, d = Math.hypot(dx, dy), f = C_LIGHT * (state.time - tR) - d;
      if (Math.abs(f) < 1e-13) break; const df = -C_LIGHT * Math.max(1e-11, 1 - (dx * k.bx + dy * k.by) / d);
      const cand = tR - f / df; tR = (cand >= tLow && cand <= tHigh) ? cand : 0.5 * (tLow + tHigh); if (f > 0) tLow = tR; else tHigh = tR; }
    const k = getExactTrajectoryKinematics(0, 1, tR, beta, 'synchrotron');
    ret.push([C_LIGHT * (state.time - tR), k.x, k.y, k.bx, k.by, k.bdx, k.bdy]);
  }
  return { ...${JSON.stringify(c)}, obsKind: '${c.obs}', C: C_LIGHT, betaMax: V6_BETA_MAX, knots,
    obs: pts.map((p) => [p.x, p.y]), gpu: g.map((o) => [o.ex, o.ey]), dbg: dbg.map((o) => [o.ex, o.ey, o.emitX, o.emitY]), exact: ex, ret };
})()`;

const out = { errors, cases: [] };
for (const c of cases) { out.cases.push(await page.evaluate(CASE(c))); console.log('exported', JSON.stringify(c)); }
console.log('errors:', errors.length ? errors.join('; ') : 'none');
writeFileSync(outFile, JSON.stringify(out));
await browser.close();
