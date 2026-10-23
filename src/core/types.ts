/** Kinematic state of a point charge at one instant (2D motion in the x–y plane). */
export interface Kinematics {
  /** Position. */
  x: number;
  y: number;
  /** Velocity divided by c (β). */
  bx: number;
  by: number;
  /** Acceleration divided by c (dβ/dt, per unit time). */
  bdx: number;
  bdy: number;
  /** Charge in units of the elementary charge used by the kernel (±1 for the presets). */
  charge: number;
}

/** A prescribed worldline: anything that can report its kinematics at time t. */
export interface Trajectory {
  kinematics(t: number): Kinematics;
}

/** Radial layout of the history shells. */
export type GridMode = 'compactified' | 'uniform' | 'log' | 'power2';

/**
 * How velocity is stored and interpolated.
 * 0 = raw β with Catmull–Rom interpolation,
 * 1 = |β| interpolated linearly, direction from the spline,
 * 2 = proper velocity u = γβ stored; |u| interpolated linearly, direction from the spline.
 */
export type VelocityMode = 0 | 1 | 2;

/** Numerical configuration shared by the CPU and GPU paths. */
export interface SolverConfig {
  /** Speed of light in code units. */
  c: number;
  /** Ring-buffer timestep. */
  dt: number;
  /** Ring-buffer length (samples). Lookback reach is c·dt·ringSize. */
  ringSize: number;
  /** Number of history shells per source. */
  mT: number;
  /** Bisection steps in the retarded-time search. */
  kBin: number;
  velMode: VelocityMode;
  /** Compactification scale L in R = L·tan(θ/2). */
  scaleL: number;
  /** Plummer softening length ε (0 = exact LW). */
  softEps: number;
  /** Fundamental angular frequency used for the Nyquist prefilter. */
  omega: number;
  gridMode: GridMode;
  /** Reach of the non-compactified grids. */
  gridRMax: number;
  /** Apply the Nyquist prefilter/taper on coarse compactified shells. */
  prefilter: boolean;
}

export const DEFAULT_CONFIG: SolverConfig = {
  c: 2.35,
  dt: 0.00675,
  ringSize: 16384,
  mT: 128,
  kBin: 9,
  velMode: 2,
  scaleL: 1.35,
  softEps: 0.06,
  omega: 1.6,
  gridMode: 'compactified',
  gridRMax: 220.0,
  prefilter: true,
};

/** Largest number of shells per source supported by the texture layout. */
export const MAX_M_T = 2048;
/** Largest number of sources supported by the texture layout. */
export const MAX_SOURCES = 256;

/** Electric field sample. */
export interface FieldSample {
  ex: number;
  ey: number;
  mag: number;
}
