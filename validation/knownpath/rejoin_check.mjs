// After a drag on the orbit preset, track the distance and velocity gap between source 0 and its prescribed path.
// Usage, from the repository root: node validation/knownpath/rejoin_check.mjs [beta=0.8] [preset=synchrotron] (prints only).
import { launch } from '../../tests/gpu/browser.mjs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
const HERE = dirname(fileURLToPath(import.meta.url));
const APP = 'file://' + join(HERE, '..', '..', 'web', 'index.html');
const b = await launch(); const p = await b.newPage({ viewport: { width: 640, height: 700 } });
await p.goto(APP, { waitUntil: 'commit' });
await p.waitForFunction(() => typeof state !== 'undefined' && state.sources && state.sources.length > 0, null, { timeout: 300000, polling: 500 });
await p.evaluate('state.paused = true');
const beta = process.argv[2] || '0.8', mode = process.argv[3] || 'synchrotron';
const r = await p.evaluate(([beta, mode]) => {
  state.numParticles = 4; state.motionMode = mode; state.beta = +beta; state.mT = 512; state.kBin = 11; state.time = 2.5;
  reseedAllSources();
  const dt = 0.45 * DT_STEP, s = state.sources[0], log = [];
  state.interactionTarget = 'source0'; s.returningToOrbit = true; s._drag = null;
  for (let k = 0; k < 9000; k++) {
    if (k < 500) state.draggedSourceTarget = { x: 0.6 * Math.sin(3.1 * k * dt) + (k > 200 ? 0.4 : 0), y: 0.5 * Math.cos(2.3 * k * dt) };
    if (k === 500) state.interactionTarget = null;
    stepSimulation(dt);
    if (k % 500 === 0 || !s.returningToOrbit) {
      const kin = getExactTrajectoryKinematics(0, 4, state.time);
      log.push({ k, ret: s.returningToOrbit, dpos: Math.hypot(s.x - kin.x, s.y - kin.y), dv: Math.hypot(s.bx - kin.bx, s.by - kin.by), b: Math.hypot(s.bx, s.by), bPath: Math.hypot(kin.bx, kin.by) });
      if (!s.returningToOrbit) break;
    }
  }
  return log;
}, [beta, mode]);
for (const l of r) console.log(l.k, l.ret, l.dpos.toExponential(2), l.dv.toExponential(2), l.b.toFixed(4), l.bPath.toFixed(4));
await b.close();
