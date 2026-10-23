/**
 * Strict IEEE-754 single-precision CPU emulation of the legacy fragment shader.
 * Used to test the GPU path without a GPU, and to separate algorithm error from
 * GPU arithmetic differences. Operation order mirrors the GLSL.
 */
import { f32, f32Len, LEGACY_BETA_MAX, type ShellHistory } from './shells.js';
import type { SolverConfig, VelocityMode } from './types.js';

function getKnotBetaFloat32(src: ShellHistory, mIdx: number, velMode: VelocityMode) {
  const ux = src.emitUx[mIdx];
  const uy = src.emitUy[mIdx];
  if (velMode === 2) {
    const uSq = f32(f32(ux * ux) + f32(uy * uy));
    const invGamma = f32(1.0 / f32(Math.sqrt(f32(1.0 + uSq))));
    return { bx: f32(ux * invGamma), by: f32(uy * invGamma) };
  }
  return { bx: ux, by: uy };
}

export function computeBernsteinAlphaFloat32(vSecX: number, vSecY: number, v1RawX: number, v1RawY: number, maxSub: number): number {
  const v1Sq = f32(f32(v1RawX * v1RawX) + f32(v1RawY * v1RawY));
  const maxSq = f32(maxSub * maxSub);
  if (v1Sq <= maxSq) return f32(1.0);
  const dx = f32(v1RawX - vSecX);
  const dy = f32(v1RawY - vSecY);
  const a = f32(f32(dx * dx) + f32(dy * dy));
  if (a <= f32(1e-12)) return f32(0.0);
  const bHalf = f32(f32(vSecX * dx) + f32(vSecY * dy));
  const c = f32(f32(f32(vSecX * vSecX) + f32(vSecY * vSecY)) - maxSq);
  if (c >= f32(0.0)) return f32(0.0);
  const disc = f32(Math.max(f32(0.0), f32(f32(bHalf * bHalf) - f32(a * c))));
  const alpha = f32(f32(-bHalf + f32(Math.sqrt(disc))) / a);
  return f32(Math.max(f32(0.0), Math.min(f32(1.0), alpha)));
}

export interface HistoryEval32 {
  x: number; y: number; x1: number; y1: number; dxSub: number; dySub: number;
  r1: number; dRSub: number; dRdm: number; dzdRx: number; dzdRy: number;
  bx: number; by: number; bHatX: number; bHatY: number; bMag: number; oneMinusBeta2: number;
  bdx: number; bdy: number;
}

export function sampleSourceHistoryFloat32(src: ShellHistory, mFloat: number, M: number, velMode: VelocityMode): HistoryEval32 {
  const mClamped = f32(Math.max(0.0, Math.min(f32(M - 1.0001), f32(mFloat))));
  const m1 = Math.floor(mClamped);
  const m0 = Math.max(0, m1 - 1);
  const m2 = Math.min(M - 1, m1 + 1);
  const m3 = Math.min(M - 1, m1 + 2);
  const t = f32(mClamped - m1);
  const t2 = f32(t * t);
  const t3 = f32(t2 * t);
  const r1 = src.emitR[m1];
  const r2 = src.emitR[m2];
  const dR = m2 > m1 ? f32(Math.max(f32(1e-6), f32(r2 - r1))) : f32(1e-6);
  const s = m2 > m1 ? t : f32(0.0);
  const dRSub = f32(s * dR);
  const x1 = src.emitX[m1], y1 = src.emitY[m1];
  const x2 = src.emitX[m2], y2 = src.emitY[m2];
  const dxChord = f32(x2 - x1), dyChord = f32(y2 - y1);
  const invDR = f32(1.0 / dR);
  const maxSubluminal = f32(LEGACY_BETA_MAX);
  let vSecX = f32(dxChord * invDR);
  let vSecY = f32(dyChord * invDR);
  const vSecMag = f32Len(vSecX, vSecY);
  if (vSecMag > maxSubluminal) {
    const sc = f32(maxSubluminal / vSecMag);
    vSecX = f32(vSecX * sc);
    vSecY = f32(vSecY * sc);
  }
  const kb1 = getKnotBetaFloat32(src, m1, velMode);
  const kb2 = getKnotBetaFloat32(src, m2, velMode);
  let b1x = kb1.bx, b1y = kb1.by;
  let b2x = kb2.bx, b2y = kb2.by;
  const b1Mag = f32Len(b1x, b1y);
  if (b1Mag > maxSubluminal) { b1x = f32(b1x * f32(maxSubluminal / b1Mag)); b1y = f32(b1y * f32(maxSubluminal / b1Mag)); }
  const b2Mag = f32Len(b2x, b2y);
  if (b2Mag > maxSubluminal) { b2x = f32(b2x * f32(maxSubluminal / b2Mag)); b2y = f32(b2y * f32(maxSubluminal / b2Mag)); }
  const v1RawX = f32(f32(3.0 * vSecX) + f32(b1x + b2x));
  const v1RawY = f32(f32(3.0 * vSecY) + f32(b1y + b2y));
  const alpha = computeBernsteinAlphaFloat32(vSecX, vSecY, v1RawX, v1RawY, maxSubluminal);
  const oma = f32(1.0 - alpha);
  const v0x = f32(f32(oma * vSecX) - f32(alpha * b1x));
  const v0y = f32(f32(oma * vSecY) - f32(alpha * b1y));
  const v1x = f32(vSecX + f32(alpha * f32(v1RawX - vSecX)));
  const v1y = f32(vSecY + f32(alpha * f32(v1RawY - vSecY)));
  const v2x = f32(f32(oma * vSecX) - f32(alpha * b2x));
  const v2y = f32(f32(oma * vSecY) - f32(alpha * b2y));
  const tx1 = f32(v0x * dR);
  const ty1 = f32(v0y * dR);
  const tx2 = f32(v2x * dR);
  const ty2 = f32(v2y * dR);
  const s2 = f32(s * s);
  const s3 = f32(s2 * s);
  const h10 = f32(s3 - f32(2.0 * s2) + s);
  const h01 = f32(f32(-2.0 * s3) + f32(3.0 * s2));
  const h11 = f32(s3 - s2);
  const dxSub = f32(f32(h01 * dxChord) + f32(h10 * tx1) + f32(h11 * tx2));
  const dySub = f32(f32(h01 * dyChord) + f32(h10 * ty1) + f32(h11 * ty2));
  const x = f32(x1 + dxSub);
  const y = f32(y1 + dySub);
  const oms = f32(1.0 - s);
  const b0 = f32(oms * oms);
  const b1 = f32(2.0 * f32(s * oms));
  const b2 = s2;
  const dzdRx = f32(f32(b0 * v0x) + f32(b1 * v1x) + f32(b2 * v2x));
  const dzdRy = f32(f32(b0 * v0y) + f32(b1 * v1y) + f32(b2 * v2y));
  const splineVelX = f32(-dzdRx);
  const splineVelY = f32(-dzdRy);
  const splineVelMag = f32Len(splineVelX, splineVelY);
  const w0 = f32(f32(-0.5 * t3) + f32(t2 - f32(0.5 * t)));
  const w1 = f32(f32(1.5 * t3) - f32(2.5 * t2) + 1.0);
  const w2 = f32(f32(-1.5 * t3) + f32(2.0 * t2) + f32(0.5 * t));
  const w3 = f32(f32(0.5 * t3) - f32(0.5 * t2));

  let bx: number, by: number, bHatX: number, bHatY: number, bMag: number, oneMinusBeta2: number;
  if (velMode === 2) {
    const uMag1 = f32Len(src.emitUx[m1], src.emitUy[m1]);
    const uMag2 = f32Len(src.emitUx[m2], src.emitUy[m2]);
    const uTarget = f32(uMag1 + f32(t * f32(uMag2 - uMag1)));
    bHatX = splineVelMag > 1e-7 ? f32(splineVelX / splineVelMag) : f32(1.0);
    bHatY = splineVelMag > 1e-7 ? f32(splineVelY / splineVelMag) : f32(0.0);
    const uSq = f32(uTarget * uTarget);
    const gamma = f32(Math.sqrt(f32(1.0 + uSq)));
    bMag = f32(uTarget / gamma);
    bx = f32(bHatX * bMag);
    by = f32(bHatY * bMag);
    oneMinusBeta2 = f32(1.0 / f32(1.0 + uSq));
  } else if (velMode === 1) {
    const bMag1 = f32Len(src.emitUx[m1], src.emitUy[m1]);
    const bMag2 = f32Len(src.emitUx[m2], src.emitUy[m2]);
    const bTarget = f32(bMag1 + f32(t * f32(bMag2 - bMag1)));
    bHatX = splineVelMag > 1e-7 ? f32(splineVelX / splineVelMag) : f32(1.0);
    bHatY = splineVelMag > 1e-7 ? f32(splineVelY / splineVelMag) : f32(0.0);
    bMag = f32(Math.min(0.9999995, bTarget));
    bx = f32(bHatX * bMag);
    by = f32(bHatY * bMag);
    oneMinusBeta2 = f32(f32(1.0 - bMag) * f32(1.0 + bMag));
  } else {
    const rawUx = f32(f32(src.emitUx[m0] * w0) + f32(src.emitUx[m1] * w1) + f32(src.emitUx[m2] * w2) + f32(src.emitUx[m3] * w3));
    const rawUy = f32(f32(src.emitUy[m0] * w0) + f32(src.emitUy[m1] * w1) + f32(src.emitUy[m2] * w2) + f32(src.emitUy[m3] * w3));
    bx = rawUx;
    by = rawUy;
    bMag = f32(Math.min(0.9999995, f32Len(bx, by)));
    bHatX = bMag > 1e-7 ? f32(bx / bMag) : f32(1.0);
    bHatY = bMag > 1e-7 ? f32(by / bMag) : f32(0.0);
    oneMinusBeta2 = f32(f32(1.0 - bMag) * f32(1.0 + bMag));
  }
  const bdx = f32(f32(src.emitBdx[m0] * w0) + f32(src.emitBdx[m1] * w1) + f32(src.emitBdx[m2] * w2) + f32(src.emitBdx[m3] * w3));
  const bdy = f32(f32(src.emitBdy[m0] * w0) + f32(src.emitBdy[m1] * w1) + f32(src.emitBdy[m2] * w2) + f32(src.emitBdy[m3] * w3));
  return { x, y, x1, y1, dxSub, dySub, r1, dRSub, dRdm: dR, dzdRx, dzdRy, bx, by, bHatX, bHatY, bMag, oneMinusBeta2, bdx, bdy };
}

function evalCancellationFreeKappaAndUFloat32(nx: number, ny: number, hist: HistoryEval32) {
  const oneMinusBeta = f32(hist.oneMinusBeta2 / f32(1.0 + hist.bMag));
  const diffX = f32(nx - hist.bHatX);
  const diffY = f32(ny - hist.bHatY);
  const diffSq = f32(f32(diffX * diffX) + f32(diffY * diffY));
  const kappa = f32(Math.max(f32(1e-7), f32(oneMinusBeta + f32(0.5 * f32(hist.bMag * diffSq)))));
  const ux = f32(f32(oneMinusBeta * nx) + f32(hist.bMag * diffX));
  const uy = f32(f32(oneMinusBeta * ny) + f32(hist.bMag * diffY));
  return { kappa, ux, uy };
}

export interface Field32 {
  ex: number; ey: number; mag: number; velMag: number; radMag: number;
  rRet: number; thetaRet: number; emitX: number; emitY: number;
}

/** Field at one point from all given sources, emulating the shader in Float32. */
export function evaluateShaderEquivalentFloat32(sources: readonly ShellHistory[], pxIn: number, pyIn: number, cfg: SolverConfig): Field32 {
  const px = f32(pxIn), py = f32(pyIn);
  const M = cfg.mT;
  const K_BIN = cfg.kBin;
  const vMode = cfg.velMode;
  const softEps = f32(cfg.softEps);
  const dTheta = f32(Math.PI / M);
  let totEx = f32(0.0), totEy = f32(0.0);
  let totVelMag = f32(0.0), totRadMag = f32(0.0);
  let primaryR = f32(0.0), primaryTheta = f32(0.0);
  let primaryPosX = f32(0.0), primaryPosY = f32(0.0);

  for (let s = 0; s < sources.length; s++) {
    const src = sources[s];
    let mLow = f32(0.0);
    let mHigh = f32(M - 1.001);
    for (let iter = 0; iter < K_BIN; iter++) {
      const mMid = f32(0.5 * f32(mLow + mHigh));
      const hist = sampleSourceHistoryFloat32(src, mMid, M, vMode);
      const dx = f32(f32(px - hist.x1) - hist.dxSub);
      const dy = f32(f32(py - hist.y1) - hist.dySub);
      const dist = f32Len(dx, dy);
      const rShell = f32(hist.r1 + hist.dRSub);
      if (rShell < dist) mLow = mMid; else mHigh = mMid;
    }
    let mRet = f32(0.5 * f32(mLow + mHigh));
    for (let nIter = 0; nIter < 2; nIter++) {
      const hist = sampleSourceHistoryFloat32(src, mRet, M, vMode);
      const dx = f32(f32(px - hist.x1) - hist.dxSub);
      const dy = f32(f32(py - hist.y1) - hist.dySub);
      const dist = f32Len(dx, dy);
      const nx = dist > 1e-6 ? f32(dx / dist) : f32(1.0);
      const ny = dist > 1e-6 ? f32(dy / dist) : f32(0.0);
      const fVal = f32(f32(hist.r1 - dist) + hist.dRSub);
      const nDotDzdR = f32(f32(nx * hist.dzdRx) + f32(ny * hist.dzdRy));
      const kappaSpline = f32(Math.max(f32(5e-5), f32(1.0 + nDotDzdR)));
      const dfdm = f32(Math.max(f32(1e-7), f32(hist.dRdm * kappaSpline)));
      const cand = f32(mRet - f32(fVal / dfdm));
      mRet = f32(Math.max(mLow, Math.min(mHigh, cand)));
    }
    const ret = sampleSourceHistoryFloat32(src, mRet, M, vMode);
    const dx = f32(f32(px - ret.x1) - ret.dxSub);
    const dy = f32(f32(py - ret.y1) - ret.dySub);
    const r2 = f32(f32(dx * dx) + f32(dy * dy));
    const r = f32(Math.sqrt(r2));
    const rSoft2 = f32(r2 + f32(softEps * softEps));
    const rSoft = f32(Math.sqrt(rSoft2));
    const nx = r > 1e-6 ? f32(dx / r) : f32(1.0);
    const ny = r > 1e-6 ? f32(dy / r) : f32(0.0);
    const ku = evalCancellationFreeKappaAndUFloat32(nx, ny, ret);
    const kappa = ku.kappa;
    const kappa3 = f32(kappa * f32(kappa * kappa));
    const chg = f32(src.charge);
    const velPref = f32(f32(chg * ret.oneMinusBeta2) / f32(kappa3 * rSoft2));
    const eVelX = f32(velPref * ku.ux);
    const eVelY = f32(velPref * ku.uy);
    const crossZ = f32(f32(ku.ux * ret.bdy) - f32(ku.uy * ret.bdx));
    const radPref = f32(chg / f32(kappa3 * rSoft));
    const eRadX = f32(radPref * f32(ny * crossZ));
    const eRadY = f32(radPref * f32(-nx * crossZ));
    totEx = f32(totEx + f32(eVelX + eRadX));
    totEy = f32(totEy + f32(eVelY + eRadY));
    totVelMag = f32(totVelMag + f32Len(eVelX, eVelY));
    totRadMag = f32(totRadMag + f32Len(eRadX, eRadY));
    if (s === 0) {
      primaryR = r;
      primaryTheta = f32(f32(mRet + 0.5) * dTheta);
      primaryPosX = ret.x;
      primaryPosY = ret.y;
    }
  }
  return {
    ex: totEx, ey: totEy, mag: f32Len(totEx, totEy),
    velMag: totVelMag, radMag: totRadMag,
    rRet: primaryR, thetaRet: primaryTheta, emitX: primaryPosX, emitY: primaryPosY,
  };
}
