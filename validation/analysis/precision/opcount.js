// Rough cost: count Float32 rounding operations (~ALU ops) per pixel, published vs df64-critical-path fix.
'use strict';
const path = process.argv[2]; const C = require(path); const f32 = C.f32;
const F = { Q: f32, P: f32, V: f32, A: f32, Rt: f32, Kr: f32 };
const src = C.makeSource(0.9999, 2.40, 16); const K = C.populate(src, 512, { RS: f32, XS: f32, US: f32, AS: f32, clampLoop: false }); K.tEval = src.tEval; K.beta = 0.9999;
const sw = C.sweepPoints(1.15, src.tEval, 0.9999, 16);
for (const [nm, Fx, opt] of [['published', F, {}], ['df64 fix', Object.assign({}, F, { splitStore: true }), { df64root: true }], ['f32 bisect + widen + 4 df64 bisect', Object.assign({}, F, { splitStore: true }), { df64root: true, df64NewtonOnly: true, widen: 0.5, df64Bis: 4 }]]) {
  C.resetOps();
  for (const p of sw.pts) C.evalF32(K, p.x, p.y, Fx, Object.assign({ kBin: 11, nNewton: 2, limiter: true, kappaSplineFloor: 5e-5 }, opt));
  console.log(`${nm.padEnd(36)} ~${(C.getOps() / sw.pts.length).toFixed(0)} f32 ops/pixel (M=512, kBin=11, 2 Newton); texel fetches +${nm === 'published' ? 0 : 'one RG32F (hi/lo) per sampler call'}`);
}
