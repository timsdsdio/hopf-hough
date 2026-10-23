// Compare limiter variants in the Float64 evaluator; also probe larger retarded radii.
const C = require('./core_inst2.js'); const { state } = C;
state.numParticles = 1; state.sources = [C.createCompactifiedSource(0, 1, 2.4)];
function setup(beta, M, vm) { state.time = 2.40;
  C.seedExactRetardedHistory(state.sources[0], 0, 1, 2.40, beta, 'synchrotron', M, vm);
  for (let k = 0; k < 16; k++) C.stepSimulation(C.DT_STEP, beta, 'synchrotron', M, vm); }
function err(beta, M, vm, rEval, mode) {
  const t = state.time, sw = C.getPulseCrossingSweepPoints(rEval, t, beta, 256);
  globalThis.LIM64 = mode; globalThis.STATS64 = { n: 0, act: 0, minA: 1 };
  let e2 = 0, d = 0;
  for (const p of sw.pts) { const ex = C.evaluateExactAnalyticalLW(p.x64, p.y64, t, beta, 'synchrotron');
    const r = C.evaluateCompactifiedFloat64(p.x64, p.y64, M, vm);
    e2 += (r.ex-ex.ex)**2 + (r.ey-ex.ey)**2; d += ex.mag**2; }
  const S = globalThis.STATS64; globalThis.STATS64 = null;
  return `${(100*Math.sqrt(e2/d)).toExponential(2)}% [act ${(100*S.act/S.n).toFixed(0)}% minα ${S.minA.toFixed(2)}]`;
}
for (const beta of [0.99, 0.999, 0.9999]) for (const [M, vm] of [[128,1],[128,2],[512,2],[2048,2]]) {
  setup(beta, M, vm);
  for (const rEval of [1.15, 3.0, 8.0]) {
    const g = 1/Math.sqrt((1-beta)*(1+beta));
    console.log(`γ=${g.toFixed(1)} M=${M} vm=${vm} Rret=${rEval} | none ${err(beta,M,vm,rEval,'off')} | paper α* ${err(beta,M,vm,rEval,'on')} | |z'|²-Bernstein ${err(beta,M,vm,rEval,'sq')} | sharp ${err(beta,M,vm,rEval,'sharp')}`);
  }
}
