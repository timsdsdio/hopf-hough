import type { Kinematics } from './types.js';

/** Interpolated sample of a source's past, returned by {@link RingBuffer.sample}. */
export interface HistorySample {
  x: number;
  y: number;
  bx: number;
  by: number;
  bdx: number;
  bdy: number;
}

/**
 * Append-only history of one source in retarded time, sampled every dt.
 * Index 0 steps back = the newest sample.
 */
export class RingBuffer {
  readonly size: number;
  head = 0;
  readonly x: Float64Array;
  readonly y: Float64Array;
  readonly bx: Float64Array;
  readonly by: Float64Array;
  readonly bdx: Float64Array;
  readonly bdy: Float64Array;

  constructor(size: number) {
    this.size = size;
    this.x = new Float64Array(size);
    this.y = new Float64Array(size);
    this.bx = new Float64Array(size);
    this.by = new Float64Array(size);
    this.bdx = new Float64Array(size);
    this.bdy = new Float64Array(size);
  }

  /** Fill the whole buffer from a trajectory, newest sample at time t0. */
  seed(kin: (t: number) => Kinematics, t0: number, dt: number): void {
    const n = this.size;
    this.head = 0;
    for (let k = 0; k < n; k++) {
      const s = kin(t0 - k * dt);
      const i = (n - k) % n;
      this.x[i] = s.x; this.y[i] = s.y;
      this.bx[i] = s.bx; this.by[i] = s.by;
      this.bdx[i] = s.bdx; this.bdy[i] = s.bdy;
    }
  }

  /** Append the newest sample. */
  push(s: Kinematics): void {
    this.head = (this.head + 1) % this.size;
    const i = this.head;
    this.x[i] = s.x; this.y[i] = s.y;
    this.bx[i] = s.bx; this.by[i] = s.by;
    this.bdx[i] = s.bdx; this.bdy[i] = s.bdy;
  }

  /**
   * Interpolate the history `stepsBack` samples into the past (fractional).
   * Position: 2-point kinematic cubic Hermite. Velocity: 4-tap Catmull–Rom with
   * arc-length speed preservation. Acceleration: Catmull–Rom.
   * `stepScale` = c·dt converts β (per c) into a per-step displacement.
   */
  sample(stepsBackFloat: number, stepScale: number): HistorySample {
    const RING_SIZE = this.size;
    const sClamped = Math.max(0.0, Math.min(RING_SIZE - 3.001, stepsBackFloat));
    const k1 = Math.floor(sClamped);
    const k0 = Math.max(0, k1 - 1);
    const k2 = Math.min(RING_SIZE - 1, k1 + 1);
    const k3 = Math.min(RING_SIZE - 1, k1 + 2);
    const t = sClamped - k1;
    const t2 = t * t;
    const t3 = t2 * t;
    const i0 = (this.head - k0 + RING_SIZE) % RING_SIZE;
    const i1 = (this.head - k1 + RING_SIZE) % RING_SIZE;
    const i2 = (this.head - k2 + RING_SIZE) % RING_SIZE;
    const i3 = (this.head - k3 + RING_SIZE) % RING_SIZE;
    const rx = this.x, ry = this.y, rbx = this.bx, rby = this.by, rbdx = this.bdx, rbdy = this.bdy;

    const tx1 = -rbx[i1] * stepScale;
    const ty1 = -rby[i1] * stepScale;
    const tx2 = -rbx[i2] * stepScale;
    const ty2 = -rby[i2] * stepScale;
    const h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    const h10 = t3 - 2.0 * t2 + t;
    const h01 = -2.0 * t3 + 3.0 * t2;
    const h11 = t3 - t2;
    const x = h00 * rx[i1] + h10 * tx1 + h01 * rx[i2] + h11 * tx2;
    const y = h00 * ry[i1] + h10 * ty1 + h01 * ry[i2] + h11 * ty2;

    const w0 = -0.5 * t3 + t2 - 0.5 * t;
    const w1 = 1.5 * t3 - 2.5 * t2 + 1.0;
    const w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
    const w3 = 0.5 * t3 - 0.5 * t2;

    let rawBx: number, rawBy: number, bdx: number, bdy: number;
    if (k1 === 0) {
      const bdx1 = -rbdx[i1] * stepScale;
      const bdy1 = -rbdy[i1] * stepScale;
      const bdx2 = -rbdx[i2] * stepScale;
      const bdy2 = -rbdy[i2] * stepScale;
      rawBx = h00 * rbx[i1] + h10 * bdx1 + h01 * rbx[i2] + h11 * bdx2;
      rawBy = rby[i1] * h00 + h10 * bdy1 + h01 * rby[i2] + h11 * bdy2;
      bdx = rbdx[i1] + t * (rbdx[i2] - rbdx[i1]);
      bdy = rbdy[i1] + t * (rbdy[i2] - rbdy[i1]);
    } else {
      rawBx = rbx[i0] * w0 + rbx[i1] * w1 + rbx[i2] * w2 + rbx[i3] * w3;
      rawBy = rby[i0] * w0 + rby[i1] * w1 + rby[i2] * w2 + rby[i3] * w3;
      bdx = rbdx[i0] * w0 + rbdx[i1] * w1 + rbdx[i2] * w2 + rbdx[i3] * w3;
      bdy = rbdy[i0] * w0 + rbdy[i1] * w1 + rbdy[i2] * w2 + rbdy[i3] * w3;
    }

    const bMag1 = Math.hypot(rbx[i1], rby[i1]);
    const bMag2 = Math.hypot(rbx[i2], rby[i2]);
    const bTarget = bMag1 + t * (bMag2 - bMag1);
    const rawBMag = Math.hypot(rawBx, rawBy);
    const bScale = rawBMag > 1e-12 ? (bTarget / rawBMag) : 1.0;
    return { x, y, bx: rawBx * bScale, by: rawBy * bScale, bdx, bdy };
  }
}
