import { legacyPresetKinematics } from './presets.js';
import { f32 } from './shells.js';

export interface QueryPoint {
  /** Single-precision coordinates actually sent to the GPU. */
  x: number;
  y: number;
  /** Unrounded double-precision coordinates. */
  x64: number;
  y64: number;
}

/**
 * Point reached by the ray emitted tangentially from the synchrotron orbit at
 * retarded radius rRet (the crest of the beamed pulse).
 */
export function synchrotronCrest(rRet: number, tObs: number, beta: number, omega: number, c: number) {
  const tRetStar = tObs - rRet / c;
  const kin = legacyPresetKinematics('synchrotron', { index: 0, count: 1, beta, omega, c }, tRetStar);
  const bMag = Math.hypot(kin.bx, kin.by);
  const bHatX = bMag > 1e-12 ? kin.bx / bMag : 1.0;
  const bHatY = bMag > 1e-12 ? kin.by / bMag : 0.0;
  const x = kin.x + rRet * bHatX;
  const y = kin.y + rRet * bHatY;
  return { x, y, rObs: Math.hypot(x, y), phiObs: Math.atan2(y, x), emitX: kin.x, emitY: kin.y, bHatX, bHatY, tRetStar };
}

/** Orbit-centred sweep of n observers across the pulse, half-width min(0.45, 4γ⁻³) radians. */
export function pulseCrossingSweep(rRet: number, tObs: number, beta: number, n: number, omega: number, c: number) {
  const gamma = 1.0 / Math.sqrt((1.0 - beta) * (1.0 + beta));
  const crest = synchrotronCrest(rRet, tObs, beta, omega, c);
  const halfSweep = Math.min(0.45, Math.max(1e-6, 4.0 * Math.pow(gamma, -3.0)));
  const pts: QueryPoint[] = [];
  for (let k = 0; k < n; k++) {
    const u = (k / (n - 1)) * 2.0 - 1.0;
    const ang = crest.phiObs + u * halfSweep;
    const x64 = crest.rObs * Math.cos(ang);
    const y64 = crest.rObs * Math.sin(ang);
    pts.push({ x: f32(x64), y: f32(y64), x64, y64 });
  }
  return { pts, halfSweep, crest };
}
