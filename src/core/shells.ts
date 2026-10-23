import { RingBuffer, type HistorySample } from './ring.js';
import { MAX_M_T, type SolverConfig, type VelocityMode } from './types.js';

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

    // Legacy chord clamp: pull any knot whose chord would be superluminal back inside the light cone.
    const maxSub = f32(LEGACY_BETA_MAX);
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
      }
    }
  }
}
