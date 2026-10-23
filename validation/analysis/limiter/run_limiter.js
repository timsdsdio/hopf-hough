// Instrument the Bernstein limiter on the paper's synchrotron pulse benchmark.
const C = require('./core_inst.js');
const { state, f32 } = C;
state.numParticles = 1;
state.sources = [C.createCompactifiedSource(0, 1, 2.4)];
const fresh = () => ({ n: 0, act: 0, minA: 1 });

function setup(beta, motion, M, velMode, uMaxR, grid) {
  state.time = 2.40; globalThis.CLAMPS = 0;
  C.seedExactRetardedHistory(state.sources[0], 0, 1, 2.40, beta, motion, M, velMode, uMaxR, grid);
  for (let k = 0; k < 16; k++) C.stepSimulation(C.DT_STEP, beta, motion, M, velMode, uMaxR, grid);
  return globalThis.CLAMPS; // cumulative over 17 populates
}
function segStats(M, velMode) {
  const src = state.sources[0];
  globalThis.STATS = fresh();
  for (let m = 0; m < M - 1; m++) C.sampleSourceHistoryFloat32(src, m + 0.5, M, velMode);
  const s32 = globalThis.STATS; globalThis.STATS = null;
  // raw (alpha=1) cubic: max_s |z'(s)| in f64, and Bernstein control bound
  let maxRaw = 0, maxCtrl = 0, nSuper = 0, nCtrlSuper = 0, worstM = -1;
  for (let m = 0; m < M - 1; m++) {
    const dR = src.emitR64[m+1] - src.emitR64[m];
    const b = (i) => { const ux = src.emitUx64[i], uy = src.emitUy64[i];
      const g = velMode === 2 ? Math.sqrt(1 + ux*ux + uy*uy) : 1; return [ux/g, uy/g]; };
    const [b1x,b1y] = b(m), [b2x,b2y] = b(m+1);
    const vx = (src.emitX64[m+1]-src.emitX64[m])/dR, vy = (src.emitY64[m+1]-src.emitY64[m])/dR;
    const V = [[-b1x,-b1y],[3*vx+b1x+b2x,3*vy+b1y+b2y],[-b2x,-b2y]];
    let mx = 0;
    for (let k = 0; k <= 200; k++) { const s = k/200, w=[(1-s)*(1-s),2*s*(1-s),s*s];
      const zx = w[0]*V[0][0]+w[1]*V[1][0]+w[2]*V[2][0], zy = w[0]*V[0][1]+w[1]*V[1][1]+w[2]*V[2][1];
      mx = Math.max(mx, Math.hypot(zx,zy)); }
    const ctrl = Math.hypot(V[1][0],V[1][1]);
    if (mx > maxRaw) { maxRaw = mx; worstM = m; }
    maxCtrl = Math.max(maxCtrl, ctrl);
    if (mx >= 1) nSuper++; if (ctrl > 0.99995) nCtrlSuper++;
  }
  return { s32, maxRaw, worstM, Rworst: src.emitR64[worstM], maxCtrl, nSuper, nCtrlSuper };
}
function sweepErr(beta, M, kBin, velMode, nS, rEval) {
  const tEval = state.time;
  const sw = C.getPulseCrossingSweepPoints(rEval, tEval, beta, nS);
  const res = {};
  const ex32 = sw.pts.map(p => C.evaluateExactAnalyticalLW(p.x, p.y, tEval, beta, 'synchrotron'));
  const ex64 = sw.pts.map(p => C.evaluateExactAnalyticalLW(p.x64, p.y64, tEval, beta, 'synchrotron'));
  let den = 0; ex32.forEach(e => den += e.mag*e.mag);
  for (const lim of ['on','off']) {
    globalThis.LIM = lim; globalThis.STATS = fresh(); let e2 = 0;
    sw.pts.forEach((p,k) => { const r = C.evaluateShaderEquivalentFloat32(p.x, p.y, M, kBin, velMode);
      e2 += (r.ex-ex32[k].ex)**2 + (r.ey-ex32[k].ey)**2; });
    res['f32_'+lim] = 100*Math.sqrt(e2/den); if (lim==='on') res.sweepAct32 = globalThis.STATS;
    globalThis.STATS = null;
    globalThis.LIM64 = lim; globalThis.STATS64 = fresh(); e2 = 0;
    sw.pts.forEach((p,k) => { const r = C.evaluateCompactifiedFloat64(p.x64, p.y64, M, velMode);
      e2 += (r.ex-ex64[k].ex)**2 + (r.ey-ex64[k].ey)**2; });
    res['f64_'+lim] = 100*Math.sqrt(e2/den); if (lim==='on') res.sweepAct64 = globalThis.STATS64;
    globalThis.STATS64 = null;
  }
  globalThis.LIM = 'on'; globalThis.LIM64 = 'off';
  return res;
}
const fmtS = s => `${s.act}/${s.n} (minα=${s.minA.toFixed(4)})`;
const configs = [[128,9,1],[512,11,2],[1024,12,2],[2048,13,2]];
for (const beta of [0.99, 0.999, 0.9999]) {
  const g = 1/Math.sqrt((1-beta)*(1+beta));
  for (const [M,kBin,vm] of configs) {
    const clamps = setup(beta, 'synchrotron', M, vm);
    const st = segStats(M, vm);
    const e = sweepErr(beta, M, kBin, vm, 256, 1.15);
    console.log(`γ=${g.toFixed(2)} M=${M} vm=${vm} | chordClamps=${clamps} | seg α<1: ${fmtS(st.s32)} | raw max|z'|=${st.maxRaw.toFixed(6)} @R=${st.Rworst.toFixed(2)} nSeg|z'|≥1=${st.nSuper} nCtrl>βmax=${st.nCtrlSuper} | sweep α<1 f32 ${fmtS(e.sweepAct32)} f64 ${fmtS(e.sweepAct64)} | F32 L2 on/off ${e.f32_on.toFixed(4)}/${e.f32_off.toFixed(4)}% | F64 L2 on/off ${e.f64_on.toFixed(5)}/${e.f64_off.toFixed(5)}%`);
  }
}
