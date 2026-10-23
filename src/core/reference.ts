import type { Trajectory } from './types.js';

export interface ReferenceOptions {
  /** Speed of light in code units. */
  c: number;
  /** Plummer softening ε (0 = exact Liénard–Wiechert). */
  softEps: number;
  /** Maximum lookback distance searched for the retarded root. */
  maxLookback?: number;
}

export interface ReferenceField {
  ex: number;
  ey: number;
  mag: number;
  velMag: number;
  radMag: number;
  /** Retarded distance and emission point of the first source. */
  rRet: number;
  emitX: number;
  emitY: number;
}

/**
 * Double-precision Liénard–Wiechert field evaluated directly on analytic trajectories:
 * bracketed bisection on the retarded time followed by safeguarded Newton, then the
 * standard velocity + radiation field (Jackson eq. 14.14, 2D in-plane motion and observer).
 */
export function exactField(trajectories: readonly Trajectory[], px: number, py: number, t: number, opt: ReferenceOptions): ReferenceField {
  const C_LIGHT = opt.c;
  const lookback = opt.maxLookback ?? 260.0;
  let totEx = 0, totEy = 0, totVelMag = 0, totRadMag = 0;
  let primaryR = 0, primaryPosX = 0, primaryPosY = 0;

  for (let i = 0; i < trajectories.length; i++) {
    const kinAt = (tt: number) => trajectories[i].kinematics(tt);
    let tLow = t - lookback / C_LIGHT;
    let tHigh = t;
    for (let b = 0; b < 32; b++) {
      const tMid = 0.5 * (tLow + tHigh);
      const kin = kinAt(tMid);
      const fMid = C_LIGHT * (t - tMid) - Math.hypot(px - kin.x, py - kin.y);
      if (fMid > 0) tLow = tMid; else tHigh = tMid;
    }
    let tRet = 0.5 * (tLow + tHigh);
    for (let iter = 0; iter < 12; iter++) {
      const kin = kinAt(tRet);
      const dx = px - kin.x;
      const dy = py - kin.y;
      const dist = Math.hypot(dx, dy);
      const f = C_LIGHT * (t - tRet) - dist;
      if (Math.abs(f) < 1e-13) break;
      const nx = dist > 1e-12 ? dx / dist : 1.0;
      const ny = dist > 1e-12 ? dy / dist : 0.0;
      const df = -C_LIGHT * Math.max(1e-11, 1.0 - (nx * kin.bx + ny * kin.by));
      const cand = tRet - f / df;
      tRet = (cand >= tLow && cand <= tHigh) ? cand : 0.5 * (tLow + tHigh);
      if (f > 0) tLow = tRet; else tHigh = tRet;
    }

    const retKin = kinAt(tRet);
    const dx = px - retKin.x, dy = py - retKin.y;
    const r2 = dx * dx + dy * dy;
    const r = Math.sqrt(r2);
    const rSoft2 = r2 + opt.softEps * opt.softEps;
    const rSoft = Math.sqrt(rSoft2);
    const nx = r > 1e-12 ? dx / r : 1.0;
    const ny = r > 1e-12 ? dy / r : 0.0;
    const bMag = Math.hypot(retKin.bx, retKin.by);
    const oneMinusBeta2 = (1.0 - bMag) * (1.0 + bMag);
    const kappa = Math.max(1e-11, 1.0 - (nx * retKin.bx + ny * retKin.by));
    const kappa3 = kappa * kappa * kappa;
    const velPref = (retKin.charge * oneMinusBeta2) / (kappa3 * rSoft2);
    const eVelX = velPref * (nx - retKin.bx);
    const eVelY = velPref * (ny - retKin.by);
    const ux = nx - retKin.bx, uy = ny - retKin.by;
    const crossZ = ux * retKin.bdy - uy * retKin.bdx;
    const radPref = retKin.charge / (kappa3 * rSoft);
    const eRadX = radPref * (ny * crossZ);
    const eRadY = radPref * (-nx * crossZ);
    totEx += eVelX + eRadX;
    totEy += eVelY + eRadY;
    totVelMag += Math.hypot(eVelX, eVelY);
    totRadMag += Math.hypot(eRadX, eRadY);
    if (i === 0) { primaryR = r; primaryPosX = retKin.x; primaryPosY = retKin.y; }
  }
  return {
    ex: totEx, ey: totEy, mag: Math.hypot(totEx, totEy),
    velMag: totVelMag, radMag: totRadMag,
    rRet: primaryR, emitX: primaryPosX, emitY: primaryPosY,
  };
}
