// The app's drag controller (web/index.html, physicalDragStep) must produce a physically consistent
// worldline: stored acceleration equals the applied velocity change, positions follow the trapezoid rule,
// speed stays below the cap, acceleration and jerk stay within budget, and a step move does not overshoot much.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const html = readFileSync(new URL('../../web/index.html', import.meta.url), 'utf8');
const a = html.indexOf('        const DRAG_PHYSICS = {');
const b = html.indexOf('        function stepSimulation(');
assert.ok(a > 0 && b > a, 'drag controller found in web/index.html');
const C = 2.35, DT = 0.00675, dt = 0.45 * DT;
const load = (mT, beta) => new Function('state', 'C_LIGHT', html.slice(a, b) + '\nreturn { physicalDragStep, DRAG_PHYSICS, dragAccelBudget };')({ mT, beta }, C);

function script() {
  const ev = []; let T = 0; const push = (d, f) => { ev.push({ t0: T, t1: T + d, f }); T += d; };
  push(0.6, () => ({ x: 0, y: 0 })); push(0.5, () => ({ x: 1.6, y: 0.4 })); push(0.4, () => ({ x: 1.6, y: 0.4 }));
  push(0.6, () => ({ x: -1.2, y: -0.8 }));
  for (let k = 0; k < 6; k++) push(0.12, () => (k % 2 ? { x: 0.9, y: 0.2 } : { x: -0.9, y: -0.2 }));
  push(1.2, (t) => ({ x: 1.1 * Math.cos(5 * t), y: 1.1 * Math.sin(5 * t) }));
  push(0.5, (t) => ({ x: 40 * t, y: 0 }));                       // pointer far faster than light
  const at = (t) => { for (const e of ev) if (t >= e.t0 && t < e.t1) return e.f(t - e.t0); return ev[ev.length - 1].f(0.5); };
  return { at, T };
}

for (const [mT, beta] of [[128, 0.99], [512, 0.9999], [2048, 0.5]]) {
  test(`drag worldline is consistent and bounded (M_T=${mT}, beta slider ${beta})`, () => {
    const app = load(mT, beta), { at, T } = script();
    const cap = Math.min(app.DRAG_PHYSICS.maxBeta, beta), aMax = app.dragAccelBudget(), jMax = aMax / app.DRAG_PHYSICS.jerkTau;
    let s = { x: 0, y: 0, bx: 0, by: 0, bdx: 0, bdy: 0 };
    for (let n = 0; n * dt < T; n++) {
      const q = app.physicalDragStep(s, at(n * dt), dt, null);
      const fx = (q.bx - s.bx) / (C * dt), fy = (q.by - s.by) / (C * dt);
      assert.ok(Math.hypot(q.bdx - fx, q.bdy - fy) < 1e-9, 'stored acceleration = applied velocity change');
      const tx = s.x + 0.5 * (s.bx + q.bx) * C * dt, ty = s.y + 0.5 * (s.by + q.by) * C * dt;
      assert.ok(Math.hypot(q.x - tx, q.y - ty) < 1e-12, 'trapezoid-rule position');
      assert.ok(Math.hypot(q.bx, q.by) <= cap + 1e-12, 'speed below the cap');
      assert.ok(Math.hypot(q.bdx, q.bdy) * C <= aMax * (1 + 1e-9), 'acceleration within budget');
      assert.ok(Math.hypot(q.bdx - s.bdx, q.bdy - s.bdy) * C / dt <= jMax * (1 + 1e-9), 'jerk within budget');
      s = { ...s, ...q };
    }
  });
}

test('a step move settles without large overshoot', () => {
  const app = load(128, 0.99);
  let s = { x: 0, y: 0, bx: 0, by: 0, bdx: 0, bdy: 0 }, peak = 0;
  for (let n = 0; n * dt < 3; n++) { s = { ...s, ...app.physicalDragStep(s, { x: 0.5, y: 0 }, dt, null) }; peak = Math.max(peak, s.x); }
  assert.ok(peak - 0.5 < 0.05, `overshoot ${peak - 0.5}`);
  assert.ok(Math.abs(s.x - 0.5) < 1e-3 && Math.hypot(s.bx, s.by) < 1e-3, 'settles at the target');
});

test('release returns to a moving path and hands off without a jump', () => {
  const app = load(128, 0.9);
  const path = (t) => ({ x: 0.85 * Math.cos(1.6 * t), y: 0.85 * Math.sin(1.6 * t), bx: -0.85 * 1.6 * Math.sin(1.6 * t) / C, by: 0.85 * 1.6 * Math.cos(1.6 * t) / C,
    bdx: -0.85 * 2.56 * Math.cos(1.6 * t) / (C * C), bdy: -0.85 * 2.56 * Math.sin(1.6 * t) / (C * C) });
  let s = { x: 1.5, y: -1.0, bx: 0.3, by: 0, bdx: 0, bdy: 0 }, t = 0, handed = false;
  for (let n = 0; n < 20000 && !handed; n++) {
    t += dt; const k = path(t);
    // same hand-off test as the app: the coasting end-of-step position meets the path
    if (Math.hypot(k.x - (s.x + s.bx * C * dt), k.y - (s.y + s.by * C * dt)) < 1e-4 && Math.hypot(s.bx - k.bx, s.by - k.by) < 1e-3 + Math.hypot(k.bdx, k.bdy) * C * dt) handed = true;
    else s = { ...s, ...app.physicalDragStep(s, { x: k.x, y: k.y }, dt, k) };
    assert.ok(Math.hypot(s.bx, s.by) < 1);
  }
  assert.ok(handed, 'reaches the hand-off tolerance');
});
