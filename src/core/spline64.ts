/**
 * Double-precision evaluation of the compactified-shell representation (legacy algorithm:
 * cubic Hermite position with unlimited endpoint tangents, |u| interpolated linearly,
 * Catmull–Rom acceleration). Separates interpolation error from Float32 effects.
 */
import type { ShellHistory } from './shells.js';
import type { FieldSample, SolverConfig, VelocityMode } from './types.js';

function sample64(src: ShellHistory, mFloat: number, M: number, velMode: VelocityMode) {
  const mClamped = Math.max(0.0, Math.min(M - 1.0001, mFloat));
  const m1 = Math.floor(mClamped);
  const m0 = Math.max(0, m1 - 1);
  const m2 = Math.min(M - 1, m1 + 1);
  const m3 = Math.min(M - 1, m1 + 2);
  const s = m2 > m1 ? (mClamped - m1) : 0.0;
  const s2 = s * s, s3 = s2 * s;
  const r1 = src.emitR64[m1], r2 = src.emitR64[m2];
  const dR = m2 > m1 ? Math.max(1e-12, r2 - r1) : 1e-12;
  const dRSub = s * dR;
  const uSq1 = src.emitUx64[m1] * src.emitUx64[m1] + src.emitUy64[m1] * src.emitUy64[m1];
  const uSq2 = src.emitUx64[m2] * src.emitUx64[m2] + src.emitUy64[m2] * src.emitUy64[m2];
  const b1x = velMode === 2 ? src.emitUx64[m1] / Math.sqrt(1.0 + uSq1) : src.emitUx64[m1];
  const b1y = velMode === 2 ? src.emitUy64[m1] / Math.sqrt(1.0 + uSq1) : src.emitUy64[m1];
  const b2x = velMode === 2 ? src.emitUx64[m2] / Math.sqrt(1.0 + uSq2) : src.emitUx64[m2];
  const b2y = velMode === 2 ? src.emitUy64[m2] / Math.sqrt(1.0 + uSq2) : src.emitUy64[m2];
  const dxChord = src.emitX64[m2] - src.emitX64[m1];
  const dyChord = src.emitY64[m2] - src.emitY64[m1];
  const vSecX = dxChord / dR, vSecY = dyChord / dR;
  const v0x = -b1x, v0y = -b1y;
  const v1x = 3.0 * vSecX + b1x + b2x, v1y = 3.0 * vSecY + b1y + b2y;
  const v2x = -b2x, v2y = -b2y;
  const h10 = s3 - 2.0 * s2 + s;
  const h01 = -2.0 * s3 + 3.0 * s2;
  const h11 = s3 - s2;
  const dxSub = h01 * dxChord + h10 * (v0x * dR) + h11 * (v2x * dR);
  const dySub = h01 * dyChord + h10 * (v0y * dR) + h11 * (v2y * dR);
  const oms = 1.0 - s;
  const dzdRx = oms * oms * v0x + 2.0 * s * oms * v1x + s2 * v2x;
  const dzdRy = oms * oms * v0y + 2.0 * s * oms * v1y + s2 * v2y;
  const spMag = Math.hypot(dzdRx, dzdRy);
  const bHatX = spMag > 1e-12 ? -dzdRx / spMag : 1.0;
  const bHatY = spMag > 1e-12 ? -dzdRy / spMag : 0.0;
  const uMag1 = Math.hypot(src.emitUx64[m1], src.emitUy64[m1]);
  const uMag2 = Math.hypot(src.emitUx64[m2], src.emitUy64[m2]);
  const uTarget = uMag1 + s * (uMag2 - uMag1);
  const uSq = uTarget * uTarget;
  const bMag = velMode === 2 ? uTarget / Math.sqrt(1.0 + uSq) : uTarget;
  const oneMinusBeta2 = velMode === 2 ? 1.0 / (1.0 + uSq) : (1.0 - bMag) * (1.0 + bMag);
  const w0 = -0.5 * s3 + s2 - 0.5 * s;
  const w1 = 1.5 * s3 - 2.5 * s2 + 1.0;
  const w2 = -1.5 * s3 + 2.0 * s2 + 0.5 * s;
  const w3 = 0.5 * s3 - 0.5 * s2;
  const bdx = src.emitBdx64[m0] * w0 + src.emitBdx64[m1] * w1 + src.emitBdx64[m2] * w2 + src.emitBdx64[m3] * w3;
  const bdy = src.emitBdy64[m0] * w0 + src.emitBdy64[m1] * w1 + src.emitBdy64[m2] * w2 + src.emitBdy64[m3] * w3;
  return { x1: src.emitX64[m1], y1: src.emitY64[m1], dxSub, dySub, r1, dRSub, dRdm: dR, dzdRx, dzdRy, bHatX, bHatY, bMag, oneMinusBeta2, bdx, bdy };
}

/** Field of a single source evaluated from its shell history in Float64 (legacy algorithm). */
export function evaluateCompactifiedFloat64(src: ShellHistory, px: number, py: number, cfg: SolverConfig): FieldSample {
  const M = cfg.mT;
  const velMode = cfg.velMode;
  let mLow = 0.0, mHigh = M - 1.0001;
  for (let iter = 0; iter < 24; iter++) {
    const mMid = 0.5 * (mLow + mHigh);
    const h = sample64(src, mMid, M, velMode);
    const dist = Math.hypot((px - h.x1) - h.dxSub, (py - h.y1) - h.dySub);
    if (h.r1 + h.dRSub < dist) mLow = mMid; else mHigh = mMid;
  }
  let mRet = 0.5 * (mLow + mHigh);
  for (let nIter = 0; nIter < 6; nIter++) {
    const h = sample64(src, mRet, M, velMode);
    const dx = (px - h.x1) - h.dxSub, dy = (py - h.y1) - h.dySub;
    const dist = Math.hypot(dx, dy);
    const nx = dx / dist, ny = dy / dist;
    const fVal = (h.r1 - dist) + h.dRSub;
    const dfdm = h.dRdm * Math.max(1e-11, 1.0 + nx * h.dzdRx + ny * h.dzdRy);
    mRet = Math.max(mLow, Math.min(mHigh, mRet - fVal / dfdm));
  }
  const ret = sample64(src, mRet, M, velMode);
  const dx = (px - ret.x1) - ret.dxSub, dy = (py - ret.y1) - ret.dySub;
  const r2 = dx * dx + dy * dy, r = Math.sqrt(r2);
  const rSoft2 = r2 + cfg.softEps * cfg.softEps, rSoft = Math.sqrt(rSoft2);
  const nx = dx / r, ny = dy / r;
  const oneMinusBeta = ret.oneMinusBeta2 / (1.0 + ret.bMag);
  const diffX = nx - ret.bHatX, diffY = ny - ret.bHatY;
  const kappa = oneMinusBeta + 0.5 * ret.bMag * (diffX * diffX + diffY * diffY);
  const kappa3 = kappa * kappa * kappa;
  const ux = oneMinusBeta * nx + ret.bMag * diffX;
  const uy = oneMinusBeta * ny + ret.bMag * diffY;
  const velPref = ret.oneMinusBeta2 / (kappa3 * rSoft2);
  const crossZ = ux * ret.bdy - uy * ret.bdx;
  const radPref = 1.0 / (kappa3 * rSoft);
  const ex = velPref * ux + radPref * (ny * crossZ);
  const ey = velPref * uy + radPref * (-nx * crossZ);
  return { ex, ey, mag: Math.hypot(ex, ey) };
}
