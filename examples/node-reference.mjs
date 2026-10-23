// The CPU parts of hopf-hough run in Node without a GPU: trajectories, histories and the
// double-precision Liénard–Wiechert reference. Run after `npm run build`:
//   node examples/node-reference.mjs > field.csv
import { circularOrbit, exactField, assertSubluminal } from '../dist/lib/core/index.js';

const c = 2.35;
const orbit = circularOrbit({ radius: 0.85, beta: 0.9, c });
assertSubluminal(orbit, -10, 10);

const t = 2.5;
console.log('x,y,Ex,Ey,Bz');
for (let j = 0; j < 64; j++) {
  for (let i = 0; i < 64; i++) {
    const x = -3 + (6 * (i + 0.5)) / 64, y = -3 + (6 * (j + 0.5)) / 64;
    const e = exactField([orbit], x, y, t, { c, softEps: 0 });
    const nx = (x - e.emitX) / e.rRet, ny = (y - e.emitY) / e.rRet;
    console.log(`${x},${y},${e.ex},${e.ey},${nx * e.ey - ny * e.ex}`);
  }
}
