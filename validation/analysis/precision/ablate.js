// Mixed-precision ablations: (a) all-F32 with ONE stage promoted to F64; (b) all-F64 with ONE stage demoted to F32.
'use strict';
const { run, F32, F64, S32, S64 } = require('./run.js');
const C = require('./core.js'); const { f32, id } = C;

const arithStages = ['Q', 'P', 'V', 'A', 'Rt', 'Kr'];
const storeStages = ['XS', 'RS', 'US', 'AS'];
const names = { Q: 'query coord rounding', P: 'spline position/derivative arithmetic', V: '|u| interp + bHat arithmetic', A: 'accel Catmull-Rom arithmetic',
  Rt: 'root-find arithmetic (x - z, |.|, f)', Kr: 'kernel arithmetic (kappa, E)', XS: 'knot position storage', RS: 'R_shell rounding', US: 'u storage', AS: 'accel storage' };
const f64opt = { limiter: false, kBin: 24, nNewton: 6, kappaSplineFloor: 1e-11 };

for (const beta of [0.999, 0.9999]) {
  for (const M of [512, 2048]) {
    const g = 1 / Math.sqrt((1 - beta) * (1 + beta));
    const base32 = run(beta, M, F32, S32).L2, base64 = run(beta, M, F64, S64, f64opt).L2;
    console.log(`\n### gamma=${g.toFixed(2)} M=${M}: all-F32 ${base32.toFixed(3)}%   all-F64(24+6, no limiter) ${base64.toFixed(5)}%`);
    console.log('stage | F32 with this stage in F64 | F64 with only this stage in F32');
    for (const st of arithStages) {
      const a = Object.assign({}, F32, { [st]: id }), b = Object.assign({}, F64, { [st]: f32 });
      console.log(`${names[st].padEnd(40)} | ${run(beta, M, a, S32).L2.toFixed(3).padStart(8)}% | ${run(beta, M, b, S64, f64opt).L2.toFixed(4).padStart(9)}%`);
    }
    for (const st of storeStages) {
      const a = Object.assign({}, S32, { [st]: id }), b = Object.assign({}, S64, { [st]: f32 });
      console.log(`${names[st].padEnd(40)} | ${run(beta, M, F32, a).L2.toFixed(3).padStart(8)}% | ${run(beta, M, F64, b, f64opt).L2.toFixed(4).padStart(9)}%`);
    }
    // iteration / Newton conditioning / limiter
    console.log(`${'F32, 24 bisect + 6 safeguarded Newton'.padEnd(40)} | ${run(beta, M, F32, S32, { kBin: 24, nNewton: 6, safeguard: true }).L2.toFixed(3).padStart(8)}% |`);
    console.log(`${'F64, solver iters (kBin+2 Newton, clamps)'.padEnd(40)} |          | ${run(beta, M, F64, S64, { limiter: true }).L2.toFixed(4).padStart(9)}%`);
    console.log(`${'F32, limiter off'.padEnd(40)} | ${run(beta, M, F32, S32, { limiter: false }).L2.toFixed(3).padStart(8)}% |`);
    // all arithmetic F64 but storage F32 / all storage F64 but arithmetic F32
    console.log(`${'ALL arithmetic F64, storage F32'.padEnd(40)} | ${run(beta, M, F64, S32, f64opt).L2.toFixed(3).padStart(8)}% |`);
    console.log(`${'ALL storage F64, arithmetic F32'.padEnd(40)} | ${run(beta, M, F32, S64).L2.toFixed(3).padStart(8)}% |`);
    console.log(`${'F32 with Rt AND XS in F64'.padEnd(40)} | ${run(beta, M, Object.assign({}, F32, { Rt: id }), Object.assign({}, S32, { XS: id })).L2.toFixed(3).padStart(8)}% |`);
    console.log(`${'F32 with Rt,Kr,XS in F64'.padEnd(40)} | ${run(beta, M, Object.assign({}, F32, { Rt: id, Kr: id }), Object.assign({}, S32, { XS: id })).L2.toFixed(3).padStart(8)}% |`);
    console.log(`${'F32 with Rt,Kr,XS,P in F64'.padEnd(40)} | ${run(beta, M, Object.assign({}, F32, { Rt: id, Kr: id, P: id }), Object.assign({}, S32, { XS: id })).L2.toFixed(3).padStart(8)}% |`);
    // model ablations in F64
    for (const [lab, ek] of [['exact |u|', { u: true }], ['exact direction', { dir: true }], ['exact accel', { acc: true }]]) {
      console.log(`${('F64 model: ' + lab).padEnd(40)} |          | ${run(beta, M, F64, S64, Object.assign({}, f64opt, { exactKin: ek })).L2.toFixed(5).padStart(9)}%`);
    }
  }
}
