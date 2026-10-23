/**
 * Physically consistent trajectories. Every constructor returns positions, velocities and
 * accelerations that are exact derivatives of one another, and refuses superluminal input.
 *
 * Units: code units with speed of light `c` (the app uses c = 2.35 length units per time unit;
 * pass c = 1 for natural units). Velocities are reported as β = v/c and accelerations as a/c².
 */
import type { Kinematics, Trajectory } from './types.js';

export class SuperluminalError extends Error {
  constructor(message: string) { super(message); this.name = 'SuperluminalError'; }
}

function kin(x: number, y: number, vx: number, vy: number, ax: number, ay: number, c: number, charge: number): Kinematics {
  return { x, y, bx: vx / c, by: vy / c, bdx: ax / (c * c), bdy: ay / (c * c), charge };
}

/** Uniform circular motion. */
export function circularOrbit(o: { radius: number; beta: number; c: number; phase?: number; centerX?: number; centerY?: number; charge?: number; clockwise?: boolean }): Trajectory {
  if (!(o.beta >= 0 && o.beta < 1)) throw new SuperluminalError(`circularOrbit: beta must be in [0, 1), got ${o.beta}`);
  if (!(o.radius > 0)) throw new Error('circularOrbit: radius must be positive');
  const w = (o.beta * o.c) / o.radius * (o.clockwise ? -1 : 1);
  const cx = o.centerX ?? 0, cy = o.centerY ?? 0, ph = o.phase ?? 0, q = o.charge ?? 1, r = o.radius;
  return {
    kinematics(t) {
      const a = w * t + ph, ca = Math.cos(a), sa = Math.sin(a);
      return kin(cx + r * ca, cy + r * sa, -r * w * sa, r * w * ca, -r * w * w * ca, -r * w * w * sa, o.c, q);
    },
  };
}

/** Straight-line motion at constant velocity (no acceleration, so no radiation). */
export function linearMotion(o: { beta: number; c: number; angle?: number; x0?: number; y0?: number; t0?: number; charge?: number }): Trajectory {
  if (!(Math.abs(o.beta) < 1)) throw new SuperluminalError(`linearMotion: |beta| must be < 1, got ${o.beta}`);
  const ang = o.angle ?? 0, v = o.beta * o.c, vx = v * Math.cos(ang), vy = v * Math.sin(ang);
  const x0 = o.x0 ?? 0, y0 = o.y0 ?? 0, t0 = o.t0 ?? 0, q = o.charge ?? 1;
  return { kinematics: (t) => kin(x0 + vx * (t - t0), y0 + vy * (t - t0), vx, vy, 0, 0, o.c, q) };
}

/** Harmonic oscillation along a line: x(t) = A sin(ωt + φ). Peak speed Aω must be below c. */
export function harmonicOscillator(o: { amplitude: number; omega: number; c: number; angle?: number; phase?: number; x0?: number; y0?: number; charge?: number }): Trajectory {
  const peak = Math.abs(o.amplitude * o.omega) / o.c;
  if (!(peak < 1)) throw new SuperluminalError(`harmonicOscillator: peak speed A·ω/c = ${peak.toFixed(4)} must be < 1`);
  const ang = o.angle ?? 0, ux = Math.cos(ang), uy = Math.sin(ang);
  const A = o.amplitude, w = o.omega, ph = o.phase ?? 0, x0 = o.x0 ?? 0, y0 = o.y0 ?? 0, q = o.charge ?? 1;
  return {
    kinematics(t) {
      const s = Math.sin(w * t + ph), cs = Math.cos(w * t + ph);
      const d = A * s, v = A * w * cs, a = -A * w * w * s;
      return kin(x0 + d * ux, y0 + d * uy, v * ux, v * uy, a * ux, a * uy, o.c, q);
    },
  };
}

/**
 * Hyperbolic motion: constant proper acceleration `properAccel` along a direction,
 * at rest at time t0. The classic case where coarse splines overshoot c.
 */
export function hyperbolicMotion(o: { properAccel: number; c: number; angle?: number; t0?: number; x0?: number; y0?: number; charge?: number }): Trajectory {
  const al = o.properAccel, c = o.c, ang = o.angle ?? 0, ux = Math.cos(ang), uy = Math.sin(ang);
  const t0 = o.t0 ?? 0, x0 = o.x0 ?? 0, y0 = o.y0 ?? 0, q = o.charge ?? 1;
  return {
    kinematics(t) {
      const tau = (al * (t - t0)) / c;           // = γβ
      const g = Math.sqrt(1 + tau * tau);
      const d = (c * c / al) * (g - 1);
      const v = (c * tau) / g;
      const a = al / (g * g * g);
      return kin(x0 + d * ux, y0 + d * uy, v * ux, v * uy, a * ux, a * uy, c, q);
    },
  };
}

/**
 * Ensemble of precessing ellipses (the app's "swarm" scene), with kinematics derived
 * consistently in the rotating frame. If the requested speed cap would be exceeded, the
 * whole motion is slowed down in time (positions, velocities and accelerations stay consistent).
 */
export function precessingEllipse(o: { index: number; beta: number; omega: number; c: number }): Trajectory {
  const i = o.index;
  const golden = i * 2.399963229728653;
  const layer = (i % 4) / 3.0;
  const rBase = 0.28 + 0.58 * layer;
  const ecc = 0.28 * Math.sin(i * 1.7 + 0.5);
  const wOrb = o.omega * (0.75 + 0.55 * (1.0 - layer * 0.5));
  const wPre = 0.22 * o.omega * (i % 2 === 0 ? 1.0 : -1.0);
  const aS = rBase, bS = rBase * Math.sqrt(Math.max(0.25, 1.0 - ecc * ecc));
  const q = i % 2 === 0 ? 1.0 : -1.0;
  // Raw kinematics at time-scale factor 1.
  const raw = (t: number) => {
    const th = wOrb * t + golden, ph = wPre * t + golden * 0.5;
    const ct = Math.cos(th), st = Math.sin(th), cp = Math.cos(ph), sp = Math.sin(ph);
    const u = aS * ct, v = bS * st;
    const du = -aS * wOrb * st, dv = bS * wOrb * ct;
    const d2u = -aS * wOrb * wOrb * ct, d2v = -bS * wOrb * wOrb * st;
    const x = u * cp - v * sp, y = u * sp + v * cp;
    // Body-frame velocity rotated into the lab, plus the frame-rotation term.
    const vRx = du * cp - dv * sp, vRy = du * sp + dv * cp;
    const vx = vRx - wPre * y, vy = vRy + wPre * x;
    // a = R r̈ + 2Ω×(R ṙ) + Ω×(Ω×r), with Ω×(Ω×r) = −Ω² r.
    const ax = (d2u * cp - d2v * sp) - 2.0 * wPre * vRy - wPre * wPre * x;
    const ay = (d2u * sp + d2v * cp) + 2.0 * wPre * vRx - wPre * wPre * y;
    return { x, y, vx, vy, ax, ay };
  };
  // Peak speed over a long window (precession makes it vary slowly).
  let vmax = 0;
  const T = (2 * Math.PI) / Math.max(1e-6, Math.min(Math.abs(wOrb), Math.abs(wPre)));
  for (let k = 0; k < 4096; k++) { const r = raw((k / 4096) * 2 * T); vmax = Math.max(vmax, Math.hypot(r.vx, r.vy)); }
  const s = vmax > o.beta * o.c ? (o.beta * o.c) / vmax : 1.0;   // time-scale factor
  return {
    kinematics(t) {
      const r = raw(s * t);
      return kin(r.x, r.y, s * r.vx, s * r.vy, s * s * r.ax, s * s * r.ay, o.c, q);
    },
  };
}

/** Five-point central difference derivative with step h. */
function d1(f: (t: number) => number, t: number, h: number) {
  return (f(t - 2 * h) - 8 * f(t - h) + 8 * f(t + h) - f(t + 2 * h)) / (12 * h);
}
function d2(f: (t: number) => number, t: number, h: number) {
  return (-f(t - 2 * h) + 16 * f(t - h) - 30 * f(t) + 16 * f(t + h) - f(t + 2 * h)) / (12 * h * h);
}

/**
 * Trajectory from position functions x(t), y(t). Velocity and acceleration come from
 * high-order finite differences (relative accuracy ~1e-8 for smooth motion at the default
 * step), so supply analytic derivatives via `custom` if you need more.
 */
export function functionTrajectory(o: { x: (t: number) => number; y: (t: number) => number; c: number; charge?: number; step?: number }): Trajectory {
  const h = o.step ?? 1e-3, q = o.charge ?? 1;
  return {
    kinematics(t) {
      return kin(o.x(t), o.y(t), d1(o.x, t, h), d1(o.y, t, h), d2(o.x, t, h), d2(o.y, t, h), o.c, q);
    },
  };
}

/**
 * Compile `x(t)` and `y(t)` from JavaScript expressions (e.g. "0.6*cos(2*t)").
 * Math functions are in scope without the `Math.` prefix. Runs in the page; only use
 * expressions you wrote or trust.
 */
export function expressionTrajectory(o: { x: string; y: string; c: number; charge?: number }): Trajectory {
  const names = Object.getOwnPropertyNames(Math);
  const make = (src: string) => {
    // eslint-disable-next-line no-new-func
    const f = new Function(...names, 't', `"use strict"; return (${src});`) as (...a: number[]) => number;
    const args = names.map((n) => (Math as unknown as Record<string, number>)[n]);
    return (t: number) => {
      const v = f(...args, t);
      if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`expression "${src}" did not return a finite number at t=${t}`);
      return v;
    };
  };
  return functionTrajectory({ x: make(o.x), y: make(o.y), c: o.c, charge: o.charge });
}

/**
 * Trajectory interpolated from samples (t strictly increasing) with a natural cubic spline
 * per coordinate. Velocity is continuous, acceleration piecewise linear. Outside the sample
 * range the motion is extended at constant velocity.
 */
export function sampledTrajectory(o: { t: ArrayLike<number>; x: ArrayLike<number>; y: ArrayLike<number>; c: number; charge?: number }): Trajectory {
  const n = o.t.length;
  if (n < 2 || o.x.length !== n || o.y.length !== n) throw new Error('sampledTrajectory: need ≥ 2 samples with matching t, x, y lengths');
  const T = Float64Array.from(o.t as ArrayLike<number>);
  for (let i = 1; i < n; i++) if (!(T[i] > T[i - 1])) throw new Error(`sampledTrajectory: t must be strictly increasing (row ${i})`);
  const spline = (Y: Float64Array) => {
    // Natural cubic spline second derivatives (Thomas algorithm).
    const M = new Float64Array(n), u = new Float64Array(n);
    for (let i = 1; i < n - 1; i++) {
      const sig = (T[i] - T[i - 1]) / (T[i + 1] - T[i - 1]);
      const p = sig * M[i - 1] + 2;
      M[i] = (sig - 1) / p;
      const dd = (Y[i + 1] - Y[i]) / (T[i + 1] - T[i]) - (Y[i] - Y[i - 1]) / (T[i] - T[i - 1]);
      u[i] = (6 * dd / (T[i + 1] - T[i - 1]) - sig * u[i - 1]) / p;
    }
    M[n - 1] = 0;
    for (let k = n - 2; k >= 0; k--) M[k] = M[k] * M[k + 1] + u[k];
    return M;
  };
  const X = Float64Array.from(o.x as ArrayLike<number>), Y = Float64Array.from(o.y as ArrayLike<number>);
  const MX = spline(X), MY = spline(Y);
  const evalAt = (V: Float64Array, MV: Float64Array, t: number) => {
    if (t <= T[0] || t >= T[n - 1]) {
      const i = t <= T[0] ? 0 : n - 2;
      const h = T[i + 1] - T[i];
      const slope = (V[i + 1] - V[i]) / h - (h / 6) * (t <= T[0] ? (2 * MV[i] + MV[i + 1]) : -(MV[i] + 2 * MV[i + 1]));
      const t0 = t <= T[0] ? T[0] : T[n - 1], v0 = t <= T[0] ? V[0] : V[n - 1];
      return [v0 + slope * (t - t0), slope, 0];
    }
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (T[mid] > t) hi = mid; else lo = mid; }
    const h = T[hi] - T[lo], a = (T[hi] - t) / h, b = (t - T[lo]) / h;
    const val = a * V[lo] + b * V[hi] + ((a * a * a - a) * MV[lo] + (b * b * b - b) * MV[hi]) * (h * h) / 6;
    const der = (V[hi] - V[lo]) / h - ((3 * a * a - 1) * MV[lo] - (3 * b * b - 1) * MV[hi]) * h / 6;
    const acc = a * MV[lo] + b * MV[hi];
    return [val, der, acc];
  };
  const q = o.charge ?? 1;
  const tr: Trajectory = {
    kinematics(t) {
      const [x, vx, ax] = evalAt(X, MX, t);
      const [y, vy, ay] = evalAt(Y, MY, t);
      return kin(x, y, vx, vy, ax, ay, o.c, q);
    },
  };
  assertSubluminal(tr, T[0], T[n - 1], Math.max(4 * n, 512));
  return tr;
}

/** Parse "t,x,y" CSV text (header optional; '#' comments ignored) into a sampled trajectory. */
export function csvTrajectory(text: string, c: number, charge = 1): Trajectory {
  const t: number[] = [], x: number[] = [], y: number[] = [];
  for (const line of text.split(/\r?\n/)) {
    const s = line.trim();
    if (!s || s.startsWith('#')) continue;
    const parts = s.split(/[,\s;]+/).map(Number);
    if (parts.length < 3 || parts.slice(0, 3).some((v) => !Number.isFinite(v))) continue; // header or junk
    t.push(parts[0]); x.push(parts[1]); y.push(parts[2]);
  }
  return sampledTrajectory({ t, x, y, c, charge });
}

/** Throws SuperluminalError if |β| ≥ 1 anywhere on [t0, t1] (sampled at n points). Returns the peak β. */
export function assertSubluminal(tr: Trajectory, t0: number, t1: number, n = 2048): number {
  let peak = 0, at = t0;
  for (let k = 0; k <= n; k++) {
    const t = t0 + ((t1 - t0) * k) / n;
    const s = tr.kinematics(t);
    const b = Math.hypot(s.bx, s.by);
    if (!Number.isFinite(b)) throw new Error(`trajectory is not finite at t=${t}`);
    if (b > peak) { peak = b; at = t; }
  }
  if (peak >= 1) throw new SuperluminalError(`trajectory reaches |β| = ${peak.toFixed(6)} at t = ${at.toFixed(4)} (must stay below 1)`);
  return peak;
}
