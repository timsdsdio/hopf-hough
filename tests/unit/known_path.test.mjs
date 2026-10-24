import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Simulation, DEFAULT_CONFIG, circularOrbit } from '../../dist/lib/core/index.js';

const cfg = { ...DEFAULT_CONFIG, knownPath: true, gridMode: 'compactified', mT: 256 };
const orbit = () => circularOrbit({ radius: 0.6, beta: 0.9, c: cfg.c });

test('known path: knots are exact samples of the trajectory, with analytic jerk', () => {
  const sim = new Simulation(cfg, [orbit()], 1.3);
  const sh = sim.sources[0].shells, tr = sim.trajectories[0];
  let nExact = 0, worstX = 0, worstJ = 0;
  for (let m = 0; m < cfg.mT; m++) {
    if (!sh.emitExact[m]) continue;
    nExact++;
    const k = tr.kinematics(sim.time - sh.emitR64[m] / cfg.c);
    const kJ = tr.kinematics(sim.time - sh.emitR[m] / cfg.c);   // the jerk is taken at the float32 radius
    worstX = Math.max(worstX, Math.hypot(sh.emitX64[m] - k.x, sh.emitY64[m] - k.y));
    // circular orbit: |d acc/dR| = |acc| * omega / c, perpendicular to acc
    const a = Math.hypot(k.bdx, k.bdy), om = 0.9 * cfg.c / 0.6;
    const jAn = a * om / cfg.c, j = Math.hypot(sh.emitJx[m], sh.emitJy[m]);
    worstJ = Math.max(worstJ, Math.abs(j - jAn) / jAn);
    assert.ok(Math.abs(sh.emitJx[m] * kJ.bdx + sh.emitJy[m] * kJ.bdy) / (j * a) < 1e-3, 'jerk perpendicular to acceleration');
  }
  assert.ok(nExact > 0.95 * cfg.mT, `most knots exact (${nExact}/${cfg.mT})`);
  assert.ok(worstX < 1e-6, `positions exact (${worstX})`);
  assert.ok(worstJ < 1e-3, `jerk matches the analytic value (${worstJ})`);
});

test('known path: knots inside an off-path window come from the ring and are not exact', () => {
  const sim = new Simulation(cfg, [orbit()], 0);
  let n = 0;
  sim.override = (i, t, dt, prev, pres) => (n++ < 40 ? { ...pres, x: pres.x + 0.05 } : null);
  for (let s = 0; s < 120; s++) sim.step(cfg.dt);
  const src = sim.sources[0], sh = src.shells;
  assert.equal(src.offPath.length, 1);
  const [a, b] = src.offPath[0];
  let inWin = 0, outExact = 0, out = 0;
  for (let m = 0; m < cfg.mT; m++) {
    const tRet = sim.time - sh.emitR[m] / cfg.c;
    if (tRet > a + 4 * cfg.dt && tRet < b - 4 * cfg.dt) { inWin++; assert.equal(sh.emitExact[m], 0); }
    else if (tRet < a - 4 * cfg.dt || tRet > b + 4 * cfg.dt) { out++; outExact += sh.emitExact[m]; }
  }
  assert.ok(inWin > 0, 'some knots fall in the window');
  assert.ok(outExact > 0.9 * out, 'knots outside the window are exact again');
});

test('ring mode leaves every knot non-exact', () => {
  const sim = new Simulation({ ...cfg, knownPath: false }, [orbit()], 1.3);
  assert.equal(sim.sources[0].shells.emitExact.reduce((s, v) => s + v, 0), 0);
});
