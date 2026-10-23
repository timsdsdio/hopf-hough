import type { Kinematics, Trajectory } from './types.js';

/**
 * Preset trajectories. These reproduce the original solver's presets exactly
 * (including its known defects) so that published numbers can be regenerated;
 * corrected versions live alongside them and are what the app uses by default.
 */
export type PresetName = 'synchrotron' | 'uniform' | 'oscillator' | 'swarm';

export interface PresetParams {
  /** Index of this source within the ensemble. */
  index: number;
  /** Ensemble size (sets phase offsets / spacing). */
  count: number;
  /** Characteristic speed β. */
  beta: number;
  /** Characteristic angular frequency. */
  omega: number;
  /** Speed of light in code units. */
  c: number;
}

/** Original solver presets, byte-for-byte behaviour (legacy, kept for reproducibility). */
export function legacyPresetKinematics(mode: PresetName, p: PresetParams, t: number): Kinematics {
  const { index: partIdx, count: totalParts, beta: betaVal, omega: stateOmega, c: C_LIGHT } = p;
  const phaseOffset = (partIdx / Math.max(1, totalParts)) * 2 * Math.PI;
  const charge = (partIdx % 2 === 0 || mode === 'synchrotron') ? 1.0 : -1.0;

  if (mode === 'synchrotron') {
    const omega = stateOmega;
    const v = betaVal * C_LIGHT;
    const rOrb = Math.min(0.85, v / Math.max(0.5, omega));
    const actualOmega = v / Math.max(0.05, rOrb);
    const ang = actualOmega * t + phaseOffset;
    const cosA = Math.cos(ang);
    const sinA = Math.sin(ang);
    return {
      x: rOrb * cosA,
      y: rOrb * sinA,
      bx: -betaVal * sinA,
      by: betaVal * cosA,
      bdx: (-betaVal * actualOmega * cosA) / C_LIGHT,
      bdy: (-betaVal * actualOmega * sinA) / C_LIGHT,
      charge: 1.0,
    };
  } else if (mode === 'uniform') {
    const period = 4.4;
    const wrappedT = ((t % period) + period) % period - 0.5 * period;
    const yOff = totalParts > 1 ? ((partIdx / (totalParts - 1)) - 0.5) * 1.2 : 0.0;
    const bEff = betaVal * 0.55;
    const vx = bEff * C_LIGHT;
    return { x: vx * wrappedT, y: yOff, bx: bEff, by: 0.0, bdx: 0.0, bdy: 0.0, charge };
  } else if (mode === 'swarm') {
    const goldenAngle = partIdx * 2.399963229728653;
    const shellLayer = (partIdx % 4) / 3.0;
    const rBase = 0.28 + 0.58 * shellLayer;
    const ecc = 0.28 * Math.sin(partIdx * 1.7 + 0.5);
    const orbitFreq = stateOmega * (0.75 + 0.55 * (1.0 - shellLayer * 0.5));
    const precessFreq = 0.22 * stateOmega * ((partIdx % 2 === 0) ? 1.0 : -1.0);
    const theta = orbitFreq * t + goldenAngle;
    const phiTilt = precessFreq * t + goldenAngle * 0.5;
    const aSemi = rBase;
    const bSemi = rBase * Math.sqrt(Math.max(0.25, 1.0 - ecc * ecc));
    const cosT = Math.cos(theta), sinT = Math.sin(theta);
    const u = aSemi * cosT;
    const v = bSemi * sinT;
    const du = -aSemi * orbitFreq * sinT;
    const dv = bSemi * orbitFreq * cosT;
    const d2u = -aSemi * orbitFreq * orbitFreq * cosT;
    const d2v = -bSemi * orbitFreq * orbitFreq * sinT;
    const cosP = Math.cos(phiTilt), sinP = Math.sin(phiTilt);
    const x = u * cosP - v * sinP;
    const y = u * sinP + v * cosP;
    const vxRaw = (du * cosP - dv * sinP) - precessFreq * y;
    const vyRaw = (du * sinP + dv * cosP) + precessFreq * x;
    const axRaw = (d2u * cosP - d2v * sinP) - 2.0 * precessFreq * vyRaw - precessFreq * precessFreq * x;
    const ayRaw = (d2u * sinP + d2v * cosP) + 2.0 * precessFreq * vxRaw - precessFreq * precessFreq * y;
    const speedRaw = Math.hypot(vxRaw, vyRaw);
    const maxAllowedV = betaVal * C_LIGHT;
    const vScale = speedRaw > maxAllowedV ? (maxAllowedV / speedRaw) : 1.0;
    return {
      x, y,
      bx: (vxRaw * vScale) / C_LIGHT,
      by: (vyRaw * vScale) / C_LIGHT,
      bdx: (axRaw * vScale) / (C_LIGHT * C_LIGHT),
      bdy: (ayRaw * vScale) / (C_LIGHT * C_LIGHT),
      charge,
    };
  } else {
    const omega = stateOmega;
    const maxA = Math.min(0.75, (betaVal * C_LIGHT) / Math.max(0.4, omega));
    const ang = omega * t + phaseOffset;
    const sinA = Math.sin(ang);
    const cosA = Math.cos(ang);
    const yOff = totalParts > 1 ? ((partIdx / (totalParts - 1)) - 0.5) * 1.1 : 0.0;
    return {
      x: maxA * sinA,
      y: yOff,
      bx: (maxA * omega * cosA) / C_LIGHT,
      by: 0.0,
      bdx: (-maxA * omega * omega * sinA) / (C_LIGHT * C_LIGHT),
      bdy: 0.0,
      charge,
    };
  }
}

/** Wraps the legacy preset as a Trajectory object. */
export function legacyPreset(mode: PresetName, p: PresetParams): Trajectory {
  return { kinematics: (t) => legacyPresetKinematics(mode, p, t) };
}
