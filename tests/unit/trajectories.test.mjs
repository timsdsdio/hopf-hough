import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  circularOrbit, linearMotion, harmonicOscillator, hyperbolicMotion, precessingEllipse,
  expressionTrajectory, csvTrajectory, sampledTrajectory, assertSubluminal, SuperluminalError, legacyPresetKinematics,
} from '../../dist/lib/core/index.js';

const C = 2.35;
// Velocity and acceleration must be the time derivatives of position (finite-difference check).
function consistency(tr, ts, h = 1e-4) {
  let worstV = 0, worstA = 0, scaleV = 1e-12, scaleA = 1e-12;
  for (const t of ts) {
    const m = tr.kinematics(t - h), p = tr.kinematics(t + h), k = tr.kinematics(t);
    const vx = (p.x - m.x) / (2 * h) / C, vy = (p.y - m.y) / (2 * h) / C;
    const ax = (p.bx - m.bx) / (2 * h) / C, ay = (p.by - m.by) / (2 * h) / C;
    worstV = Math.max(worstV, Math.hypot(vx - k.bx, vy - k.by)); scaleV = Math.max(scaleV, Math.hypot(k.bx, k.by));
    worstA = Math.max(worstA, Math.hypot(ax - k.bdx, ay - k.bdy)); scaleA = Math.max(scaleA, Math.hypot(k.bdx, k.bdy));
  }
  return { v: worstV / scaleV, a: worstA / scaleA };
}
const TS = Array.from({ length: 40 }, (_, i) => 0.37 * i - 3);

test('analytic trajectories are kinematically consistent', () => {
  for (const [name, tr] of Object.entries({
    circular: circularOrbit({ radius: 0.85, beta: 0.9, c: C }),
    linear: linearMotion({ beta: 0.6, c: C, angle: 0.3 }),
    oscillator: harmonicOscillator({ amplitude: 0.7, omega: 2.0, c: C }),
    hyperbolic: hyperbolicMotion({ properAccel: 3.0, c: C }),
    ellipse: precessingEllipse({ index: 5, beta: 0.8, omega: 1.6, c: C }),
  })) {
    const e = consistency(tr, TS);
    assert.ok(e.v < 1e-6 && e.a < 1e-5, `${name}: velocity error ${e.v}, acceleration error ${e.a}`);
  }
});

test('legacy swarm preset is NOT consistent (documents the fixed bug)', () => {
  const legacy = { kinematics: (t) => legacyPresetKinematics('swarm', { index: 5, count: 32, beta: 0.8, omega: 1.6, c: C }, t) };
  const e = consistency(legacy, TS);
  assert.ok(e.a > 0.05, `expected the legacy acceleration to be off by >5%, got ${e.a}`);
});

test('precessing ellipse respects its speed cap', () => {
  for (let i = 0; i < 32; i++) {
    const tr = precessingEllipse({ index: i, beta: 0.5, omega: 3.4, c: C });
    assert.ok(assertSubluminal(tr, -20, 20, 4000) <= 0.5 + 1e-3);
  }
});

test('superluminal requests are refused', () => {
  assert.throws(() => circularOrbit({ radius: 1, beta: 1.0, c: C }), SuperluminalError);
  assert.throws(() => harmonicOscillator({ amplitude: 2, omega: 2, c: C }), SuperluminalError);
  assert.throws(() => sampledTrajectory({ t: [0, 1, 2], x: [0, 3, 6], y: [0, 0, 0], c: C }), SuperluminalError);
});

test('expression and CSV trajectories reproduce a circular orbit', () => {
  const ref = circularOrbit({ radius: 0.8, beta: 0.7, c: C });
  const w = 0.7 * C / 0.8;
  const ex = expressionTrajectory({ x: `0.8*cos(${w}*t)`, y: `0.8*sin(${w}*t)`, c: C });
  const rows = ['t,x,y'];
  for (let k = 0; k <= 4000; k++) { const t = -2 + k * 0.001; const s = ref.kinematics(t); rows.push(`${t},${s.x},${s.y}`); }
  const csv = csvTrajectory(rows.join('\n'), C);
  for (const t of [-1.5, -0.3, 0.0, 0.77, 1.4]) {
    const r = ref.kinematics(t), a = ex.kinematics(t), b = csv.kinematics(t);
    assert.ok(Math.hypot(a.x - r.x, a.y - r.y) < 1e-12 && Math.hypot(a.bx - r.bx, a.by - r.by) < 1e-8 && Math.hypot(a.bdx - r.bdx, a.bdy - r.bdy) < 1e-5, 'expression');
    assert.ok(Math.hypot(b.x - r.x, b.y - r.y) < 1e-8 && Math.hypot(b.bx - r.bx, b.by - r.by) < 1e-5 && Math.hypot(b.bdx - r.bdx, b.bdy - r.bdy) < 1e-2, `csv at t=${t}`);
  }
});
