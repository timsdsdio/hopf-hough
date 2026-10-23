/**
 * The verification suite that generates the preprint's tables: evolve a ring-buffer
 * history, then compare the field from the shell representation with the exact
 * double-precision reference on an azimuthal ring and on a pulse-crossing sweep.
 */
import { evaluateShaderEquivalentFloat32 } from './emulate32.js';
import { legacyPreset, type PresetName } from './presets.js';
import { exactField } from './reference.js';
import { f32 } from './shells.js';
import { Simulation } from './simulation.js';
import { evaluateCompactifiedFloat64 } from './spline64.js';
import { pulseCrossingSweep } from './sweep.js';
import { DEFAULT_CONFIG, type GridMode, type SolverConfig, type VelocityMode } from './types.js';

export interface PointField { ex: number; ey: number; emitX: number; emitY: number }

/** Something that evaluates the field of a simulation's shells at points (GPU readback or its emulation). */
export interface FieldBackend {
  name: string;
  evaluate(sim: Simulation, pts: readonly { x: number; y: number }[]): PointField[];
}

/** CPU stand-in for the GPU: the strict Float32 emulation of the shader. */
export const emulatedBackend: FieldBackend = {
  name: 'cpu-float32-emulation',
  evaluate(sim, pts) {
    const hs = sim.shellHistories;
    return pts.map((p) => evaluateShaderEquivalentFloat32(hs, p.x, p.y, sim.config));
  },
};

export interface CaseSpec {
  beta: number;
  motion: PresetName;
  mT: number;
  kBin: number;
  velMode: VelocityMode;
  nShock: number;
  /** Reach of non-compactified grids (legacy `uniformMaxR`). */
  rMax?: number;
  /** Ring radius for the bulk test and retarded radius of the sweep crest. */
  rEval?: number;
  gridMode?: GridMode;
}

export interface CaseResult {
  bulkL2: number;
  shockL2: number;
  cpuF32L2: number;
  cpuF64L2: number;
  coordRoundL2: number;
  gpuCpuRelL2: number;
  posErr: number;
}

export interface SuiteOptions {
  base?: Partial<SolverConfig>;
  tStart?: number;
  steps?: number;
}

/** Run one configuration (the legacy `evalEvolvedConfig`). All errors are relative L2 in percent. */
export function runCase(spec: CaseSpec, backend: FieldBackend, opt: SuiteOptions = {}): CaseResult {
  const base = { ...DEFAULT_CONFIG, ...opt.base };
  const rEval = spec.rEval ?? 1.15;
  const gridMode: GridMode = spec.gridMode ?? (spec.rMax && spec.rMax > 0 ? 'uniform' : 'compactified');
  const cfg: SolverConfig = { ...base, mT: spec.mT, kBin: spec.kBin, velMode: spec.velMode, gridMode, gridRMax: spec.rMax || 220.0 };
  const traj = legacyPreset(spec.motion, { index: 0, count: 1, beta: spec.beta, omega: cfg.omega, c: cfg.c });
  const tStart = opt.tStart ?? 2.40;
  const sim = new Simulation(cfg, [traj], tStart);
  const steps = opt.steps ?? 16;
  for (let k = 0; k < steps; k++) sim.step(cfg.dt);
  const tEval = sim.time;
  const ref = { c: cfg.c, softEps: cfg.softEps };
  const trajs = sim.trajectories;

  const bulkPts: { x: number; y: number }[] = [];
  for (let k = 0; k < 64; k++) {
    const phi = (k / 64) * 2.0 * Math.PI;
    bulkPts.push({ x: f32(rEval * Math.cos(phi)), y: f32(rEval * Math.sin(phi)) });
  }
  const gpuBulk = backend.evaluate(sim, bulkPts);
  let bulkErrSq = 0, bulkExSq = 0, maxPosErr = 0;
  for (let k = 0; k < 64; k++) {
    const ex = exactField(trajs, bulkPts[k].x, bulkPts[k].y, tEval, ref);
    const gp = gpuBulk[k];
    bulkErrSq += Math.pow(gp.ex - ex.ex, 2) + Math.pow(gp.ey - ex.ey, 2);
    bulkExSq += ex.mag * ex.mag;
    maxPosErr = Math.max(maxPosErr, Math.hypot(gp.emitX - ex.emitX, gp.emitY - ex.emitY));
  }

  const sweep = pulseCrossingSweep(rEval, tEval, spec.beta, spec.nShock, cfg.omega, cfg.c);
  const shockPts = sweep.pts;
  const gpuShock = backend.evaluate(sim, shockPts);
  const hs = sim.shellHistories;
  let shockErrSq = 0, shockExSq = 0, cpuF32ErrSq = 0, cpuF64ErrSq = 0, coordRoundDiffSq = 0, gpuCpuDiffSq = 0;
  for (let k = 0; k < spec.nShock; k++) {
    const p = shockPts[k];
    const exSameF32 = exactField(trajs, p.x, p.y, tEval, ref);
    const exUnrounded64 = exactField(trajs, p.x64, p.y64, tEval, ref);
    const gp = gpuShock[k];
    const f32Cpu = evaluateShaderEquivalentFloat32(hs, p.x, p.y, cfg);
    const f64Cpu = evaluateCompactifiedFloat64(hs[0], p.x64, p.y64, cfg);
    gpuCpuDiffSq += Math.pow(gp.ex - f32Cpu.ex, 2) + Math.pow(gp.ey - f32Cpu.ey, 2);
    shockErrSq += Math.pow(gp.ex - exSameF32.ex, 2) + Math.pow(gp.ey - exSameF32.ey, 2);
    cpuF32ErrSq += Math.pow(f32Cpu.ex - exSameF32.ex, 2) + Math.pow(f32Cpu.ey - exSameF32.ey, 2);
    cpuF64ErrSq += Math.pow(f64Cpu.ex - exUnrounded64.ex, 2) + Math.pow(f64Cpu.ey - exUnrounded64.ey, 2);
    shockExSq += exSameF32.mag * exSameF32.mag;
    coordRoundDiffSq += Math.pow(exSameF32.ex - exUnrounded64.ex, 2) + Math.pow(exSameF32.ey - exUnrounded64.ey, 2);
  }
  const rel = (num: number, den: number) => Math.sqrt(num / Math.max(1e-12, den)) * 100.0;
  return {
    bulkL2: rel(bulkErrSq, bulkExSq),
    shockL2: rel(shockErrSq, shockExSq),
    cpuF32L2: rel(cpuF32ErrSq, shockExSq),
    cpuF64L2: rel(cpuF64ErrSq, shockExSq),
    coordRoundL2: rel(coordRoundDiffSq, shockExSq),
    gpuCpuRelL2: rel(gpuCpuDiffSq, shockExSq),
    posErr: maxPosErr,
  };
}

export interface SuiteResult {
  backend: string;
  rows: { beta: number; r128Norm: CaseResult; r5124Vel: CaseResult; r20484Vel: CaseResult }[];
  diag512: CaseResult; diag1024: CaseResult; diag2048: CaseResult;
  comp128Reach220: CaseResult; log128Reach220: CaseResult; pow128Reach220: CaseResult; unif128Reach220: CaseResult;
  compFar45: CaseResult; unifFar45: CaseResult;
}

export const SUITE_BETAS = [0.80, 0.95, 0.99, 0.999, 0.9999];

/** The full legacy suite (Tables 1–3 of the preprint). */
export function runSuite(backend: FieldBackend, opt: SuiteOptions = {}, onProgress?: (label: string) => void): SuiteResult {
  const rows: SuiteResult['rows'] = [];
  let diag512: CaseResult | null = null, diag1024: CaseResult | null = null, diag2048: CaseResult | null = null;
  for (const b of SUITE_BETAS) {
    onProgress?.(`β=${b}`);
    const r128Norm = runCase({ beta: b, motion: 'synchrotron', mT: 128, kBin: 9, velMode: 1, nShock: 256 }, backend, opt);
    const r5124Vel = runCase({ beta: b, motion: 'synchrotron', mT: 512, kBin: 11, velMode: 2, nShock: 256 }, backend, opt);
    const r20484Vel = runCase({ beta: b, motion: 'synchrotron', mT: 2048, kBin: 13, velMode: 2, nShock: 256 }, backend, opt);
    if (b === 0.9999) {
      diag512 = r5124Vel;
      diag1024 = runCase({ beta: b, motion: 'synchrotron', mT: 1024, kBin: 12, velMode: 2, nShock: 256 }, backend, opt);
      diag2048 = r20484Vel;
    }
    rows.push({ beta: b, r128Norm, r5124Vel, r20484Vel });
  }
  onProgress?.('grids');
  const grid = (gridMode: GridMode) => runCase({ beta: 0.99, motion: 'synchrotron', mT: 128, kBin: 9, velMode: 2, nShock: 256, rMax: 220.0, rEval: 1.15, gridMode }, backend, opt);
  const comp128Reach220 = grid('compactified');
  const log128Reach220 = grid('log');
  const pow128Reach220 = grid('power2');
  const unif128Reach220 = grid('uniform');
  onProgress?.('far field');
  const compFar45 = runCase({ beta: 0.80, motion: 'uniform', mT: 128, kBin: 9, velMode: 2, nShock: 64, rMax: 18.0, rEval: 45.0, gridMode: 'compactified' }, backend, opt);
  const unifFar45 = runCase({ beta: 0.80, motion: 'uniform', mT: 128, kBin: 9, velMode: 2, nShock: 64, rMax: 18.0, rEval: 45.0, gridMode: 'uniform' }, backend, opt);
  return {
    backend: backend.name, rows,
    diag512: diag512!, diag1024: diag1024!, diag2048: diag2048!,
    comp128Reach220, log128Reach220, pow128Reach220, unif128Reach220, compFar45, unifFar45,
  };
}
