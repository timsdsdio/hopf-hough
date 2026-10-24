import { RingBuffer, type HistorySample } from './ring.js';
import { MAX_M_T, type SolverConfig, type Trajectory, type VelocityMode } from './types.js';

export const f32 = Math.fround;

/** IEEE-754 single-precision 2D norm, matching GLSL length(vec2) without double intermediates. */
export function f32Len(x: number, y: number): number {
  const xx = f32(x * x);
  const yy = f32(y * y);
  return f32(Math.sqrt(f32(xx + yy)));
}

/** Speed limit used by the knot clamps and the causality limiter (legacy value). */
export const LEGACY_BETA_MAX = 0.99995;

export function packVelocityForStorage(bx: number, by: number, velMode: VelocityMode) {
  if (velMode === 2) {
    const bMag = Math.min(0.9999995, Math.hypot(bx, by));
    const gamma = 1.0 / Math.sqrt(Math.max(1e-14, (1.0 - bMag) * (1.0 + bMag)));
    return { ux: f32(gamma * bx), uy: f32(gamma * by), ux64: gamma * bx, uy64: gamma * by };
  }
  return { ux: f32(bx), uy: f32(by), ux64: bx, uy64: by };
}

/** Radius of shell m (double and single precision) and its local spacing. */
export function shellRadius(m: number, cfg: SolverConfig): { r64: number; r32: number; dR: number } {
  const M = cfg.mT;
  const u = (m + 0.5) / M;
  const rMaxTarget = cfg.gridRMax;
  switch (cfg.gridMode) {
    case 'uniform': {
      const dR = rMaxTarget / M;
      const r64 = (m + 0.5) * dR;
      return { r64, r32: f32(r64), dR };
    }
    case 'log': {
      const r0 = 0.05;
      const alpha = Math.log(1.0 + rMaxTarget / r0);
      const r64 = r0 * (Math.exp(alpha * u) - 1.0);
      return { r64, r32: f32(r64), dR: (r0 * alpha * Math.exp(alpha * u)) / M };
    }
    case 'power2': {
      const r64 = rMaxTarget * u * u;
      return { r64, r32: f32(r64), dR: (2.0 * rMaxTarget * u) / M };
    }
    default: {
      const dTheta = Math.PI / M;
      const th = (m + 0.5) * dTheta;
      const r64 = cfg.scaleL * Math.tan(0.5 * th);
      const r32 = f32(f32(cfg.scaleL) * f32(Math.tan(f32(0.5 * th))));
      return { r64, r32, dR: (cfg.scaleL * dTheta) / (1.0 + Math.cos(th)) };
    }
  }
}

/**
 * One source's history resampled onto M radial shells R_m = c·(t − t_ret).
 * Single-precision arrays feed the GPU textures; double-precision arrays feed
 * the CPU reference evaluators.
 */
export class ShellHistory {
  charge = 1.0;
  readonly emitX = new Float32Array(MAX_M_T);
  readonly emitY = new Float32Array(MAX_M_T);
  readonly emitUx = new Float32Array(MAX_M_T);
  readonly emitUy = new Float32Array(MAX_M_T);
  readonly emitBdx = new Float32Array(MAX_M_T);
  readonly emitBdy = new Float32Array(MAX_M_T);
  readonly emitR = new Float32Array(MAX_M_T);
  /** Low parts of the knot positions (x ≈ emitX + emitXlo to ~2^-48 relative). */
  readonly emitXlo = new Float32Array(MAX_M_T);
  readonly emitYlo = new Float32Array(MAX_M_T);
  /** Known path: d(acc)/dR at each knot, and 1 where the knot is an exact sample of the trajectory. */
  readonly emitJx = new Float32Array(MAX_M_T);
  readonly emitJy = new Float32Array(MAX_M_T);
  readonly emitExact = new Float32Array(MAX_M_T);
  readonly emitX64 = new Float64Array(MAX_M_T);
  readonly emitY64 = new Float64Array(MAX_M_T);
  readonly emitUx64 = new Float64Array(MAX_M_T);
  readonly emitUy64 = new Float64Array(MAX_M_T);
  readonly emitBdx64 = new Float64Array(MAX_M_T);
  readonly emitBdy64 = new Float64Array(MAX_M_T);
  readonly emitR64 = new Float64Array(MAX_M_T);

  /** Resample the ring buffer onto the shells (the original solver's populate step). */
  populate(ring: RingBuffer, cfg: SolverConfig): void {
    const M = cfg.mT;
    const dt = cfg.dt;
    const C_LIGHT = cfg.c;
    const stepScale = C_LIGHT * dt;
    const orbitLambda = (2.0 * Math.PI * C_LIGHT) / Math.max(0.5, cfg.omega);
    const velMode = cfg.velMode;
    this.emitExact.fill(0, 0, M);

    for (let m = 0; m < M; m++) {
      const { r64: rShell64, r32: rShellF32, dR } = shellRadius(m, cfg);
      const stepsBackCenter = (rShellF32 / C_LIGHT) / dt;
      const stepsBackCenter64 = (rShell64 / C_LIGHT) / dt;
      const filterRatio = dR / (0.5 * orbitLambda);

      let sample: HistorySample, sample64: HistorySample;
      if (cfg.prefilter && filterRatio > 0.25 && cfg.gridMode === 'compactified') {
        const dSteps = Math.min((dR / C_LIGHT) / dt, (2.0 * Math.PI / cfg.omega) / dt);
        const s0 = ring.sample(stepsBackCenter - 0.35 * dSteps, stepScale);
        const s1 = ring.sample(stepsBackCenter, stepScale);
        const s2 = ring.sample(stepsBackCenter + 0.35 * dSteps, stepScale);
        const wNyq = Math.exp(-Math.pow(filterRatio, 4.0));
        const avgBx = 0.25 * s0.bx + 0.5 * s1.bx + 0.25 * s2.bx;
        const avgBy = 0.25 * s0.by + 0.5 * s1.by + 0.25 * s2.by;
        const bTarget = 0.25 * Math.hypot(s0.bx, s0.by) + 0.5 * Math.hypot(s1.bx, s1.by) + 0.25 * Math.hypot(s2.bx, s2.by);
        const avgBMag = Math.hypot(avgBx, avgBy);
        const scaleB = avgBMag > 1e-12 ? (bTarget / avgBMag) : 1.0;
        sample = {
          x: s1.x,
          y: s1.y,
          bx: avgBx * scaleB,
          by: avgBy * scaleB,
          bdx: (0.25 * s0.bdx + 0.5 * s1.bdx + 0.25 * s2.bdx) * wNyq,
          bdy: (0.25 * s0.bdy + 0.5 * s1.bdy + 0.25 * s2.bdy) * wNyq,
        };
        sample64 = ring.sample(stepsBackCenter64, stepScale);
      } else {
        sample = ring.sample(stepsBackCenter, stepScale);
        sample64 = ring.sample(stepsBackCenter64, stepScale);
      }

      const packedV = packVelocityForStorage(sample.bx, sample.by, velMode);
      const packedV64 = packVelocityForStorage(sample64.bx, sample64.by, velMode);
      this.emitX[m] = f32(sample.x);
      this.emitY[m] = f32(sample.y);
      this.emitXlo[m] = f32(sample.x - this.emitX[m]);
      this.emitYlo[m] = f32(sample.y - this.emitY[m]);
      this.emitUx[m] = packedV.ux;
      this.emitUy[m] = packedV.uy;
      this.emitBdx[m] = f32(sample.bdx);
      this.emitBdy[m] = f32(sample.bdy);
      this.emitR[m] = rShellF32;
      this.emitX64[m] = sample64.x;
      this.emitY64[m] = sample64.y;
      this.emitUx64[m] = packedV64.ux64;
      this.emitUy64[m] = packedV64.uy64;
      this.emitBdx64[m] = sample64.bdx;
      this.emitBdy64[m] = sample64.bdy;
      this.emitR64[m] = rShell64;
    }

    // Chord clamp: pull any knot whose chord would be superluminal back inside the light cone
    // (only fires for unphysical input such as teleporting trajectories).
    if (cfg.precision === 1) {
      // Double precision on the full hi+lo positions: with float32 differences alone, close knots
      // (M = 2048) at γ ≳ 100 show phantom superluminal chords and good knots get moved.
      const maxSub = cfg.betaMax;
      for (let m = 0; m < M - 1; m++) {
        const dR = Math.max(1e-12, this.emitR[m + 1] - this.emitR[m]);
        const x0 = this.emitX[m] + this.emitXlo[m], y0 = this.emitY[m] + this.emitYlo[m];
        const dx = this.emitX[m + 1] + this.emitXlo[m + 1] - x0, dy = this.emitY[m + 1] + this.emitYlo[m + 1] - y0;
        const chordSpeed = Math.hypot(dx, dy) / dR;
        if (chordSpeed > maxSub) {
          const shrink = maxSub / chordSpeed;
          const nx = x0 + dx * shrink, ny = y0 + dy * shrink;
          this.emitX[m + 1] = f32(nx); this.emitXlo[m + 1] = f32(nx - this.emitX[m + 1]);
          this.emitY[m + 1] = f32(ny); this.emitYlo[m + 1] = f32(ny - this.emitY[m + 1]);
        }
      }
      return;
    }
    // Legacy float32 clamp (reproduces the original solver).
    const maxSub = f32(cfg.betaMax);
    for (let m = 0; m < M - 1; m++) {
      const r1 = this.emitR[m];
      const r2 = this.emitR[m + 1];
      const dR = f32(Math.max(f32(1e-6), f32(r2 - r1)));
      const dx = f32(this.emitX[m + 1] - this.emitX[m]);
      const dy = f32(this.emitY[m + 1] - this.emitY[m]);
      const chordSpeed = f32(f32Len(dx, dy) / dR);
      if (chordSpeed > maxSub) {
        const shrink = f32(maxSub / chordSpeed);
        this.emitX[m + 1] = f32(this.emitX[m] + f32(dx * shrink));
        this.emitY[m + 1] = f32(this.emitY[m] + f32(dy * shrink));
        this.emitXlo[m + 1] = 0;
        this.emitYlo[m + 1] = 0;
      }
    }
  }

  /**
   * Known-path populate: each knot is an exact sample of the trajectory at the shell's retarded time,
   * with the jerk d(acc)/dR for the quintic segments. Knots whose retarded time lies in one of
   * `offPath` (intervals when the source did not follow its trajectory) come from the ring buffer and
   * are flagged non-exact, as are knots the chord clamp moves.
   */
  populateExact(traj: Trajectory, t: number, ring: RingBuffer, cfg: SolverConfig, offPath: readonly [number, number][] = []): void {
    const M = cfg.mT, c = cfg.c, dt = cfg.dt, pad = 3 * dt, hJ = 1e-4;
    for (let m = 0; m < M; m++) {
      const { r64: rShell64, r32: rShellF32 } = shellRadius(m, cfg);
      const tRet = t - rShellF32 / c;
      const exact = !offPath.some(([a, b]) => tRet >= a - pad && tRet <= b + pad);
      let k: HistorySample, k64: HistorySample;
      if (exact) {
        k = traj.kinematics(tRet);
        const kp = traj.kinematics(tRet + hJ), km = traj.kinematics(tRet - hJ);
        this.emitJx[m] = f32(-(kp.bdx - km.bdx) / (2 * hJ) / c);
        this.emitJy[m] = f32(-(kp.bdy - km.bdy) / (2 * hJ) / c);
        const dtq = (rShellF32 - rShell64) / c, ax = k.bdx * c, ay = k.bdy * c;
        k64 = { x: k.x + c * (k.bx + 0.5 * ax * dtq) * dtq, y: k.y + c * (k.by + 0.5 * ay * dtq) * dtq, bx: k.bx + ax * dtq, by: k.by + ay * dtq, bdx: k.bdx, bdy: k.bdy };
      } else {
        k = ring.sample((rShellF32 / c) / dt, c * dt);
        k64 = ring.sample((rShell64 / c) / dt, c * dt);
        this.emitJx[m] = 0; this.emitJy[m] = 0;
      }
      const pv = packVelocityForStorage(k.bx, k.by, cfg.velMode);
      const pv64 = packVelocityForStorage(k64.bx, k64.by, cfg.velMode);
      this.emitX[m] = f32(k.x); this.emitY[m] = f32(k.y);
      this.emitXlo[m] = f32(k.x - this.emitX[m]); this.emitYlo[m] = f32(k.y - this.emitY[m]);
      this.emitUx[m] = pv.ux; this.emitUy[m] = pv.uy;
      this.emitBdx[m] = f32(k.bdx); this.emitBdy[m] = f32(k.bdy);
      this.emitR[m] = rShellF32;
      this.emitX64[m] = k64.x; this.emitY64[m] = k64.y;
      this.emitUx64[m] = pv64.ux64; this.emitUy64[m] = pv64.uy64;
      this.emitBdx64[m] = k64.bdx; this.emitBdy64[m] = k64.bdy;
      this.emitR64[m] = rShell64;
      this.emitExact[m] = exact ? 1 : 0;
    }
    const maxSub = cfg.betaMax;
    for (let m = 0; m < M - 1; m++) {
      const dR = Math.max(1e-12, this.emitR[m + 1] - this.emitR[m]);
      const x0 = this.emitX[m] + this.emitXlo[m], y0 = this.emitY[m] + this.emitYlo[m];
      const dx = this.emitX[m + 1] + this.emitXlo[m + 1] - x0, dy = this.emitY[m + 1] + this.emitYlo[m + 1] - y0;
      const chordSpeed = Math.hypot(dx, dy) / dR;
      if (chordSpeed > maxSub) {
        const shrink = maxSub / chordSpeed;
        const nx = x0 + dx * shrink, ny = y0 + dy * shrink;
        this.emitX[m + 1] = f32(nx); this.emitXlo[m + 1] = f32(nx - this.emitX[m + 1]);
        this.emitY[m + 1] = f32(ny); this.emitYlo[m + 1] = f32(ny - this.emitY[m + 1]);
        this.emitExact[m] = 0; this.emitExact[m + 1] = 0;
      }
    }
  }
}
