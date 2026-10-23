import { RingBuffer } from './ring.js';
import { ShellHistory } from './shells.js';
import type { Kinematics, SolverConfig, Trajectory } from './types.js';

/** One charge: its prescribed trajectory, its retarded-time history and its shell resampling. */
export class Source {
  trajectory: Trajectory;
  readonly ring: RingBuffer;
  readonly shells = new ShellHistory();
  /** Most recent kinematic state written to the ring. */
  current: Kinematics = { x: 0, y: 0, bx: 0, by: 0, bdx: 0, bdy: 0, charge: 1 };

  constructor(trajectory: Trajectory, ringSize: number) {
    this.trajectory = trajectory;
    this.ring = new RingBuffer(ringSize);
  }
}

/**
 * Optional per-step override of a source's kinematics (e.g. interactive dragging).
 * Return null to use the prescribed trajectory.
 */
export type KinematicsOverride = (index: number, t: number, dt: number, prev: Kinematics, prescribed: Kinematics) => Kinematics | null;

/** A set of sources advanced together in time. */
export class Simulation {
  config: SolverConfig;
  time: number;
  readonly sources: Source[] = [];
  override: KinematicsOverride | null = null;

  constructor(config: SolverConfig, trajectories: readonly Trajectory[], t0: number) {
    this.config = { ...config };
    this.time = t0;
    for (const tr of trajectories) this.sources.push(new Source(tr, config.ringSize));
    this.seed(t0);
  }

  /** Fill every history from the prescribed trajectories, ending at time t0. */
  seed(t0: number): void {
    this.time = t0;
    for (const s of this.sources) this.seedSource(s, t0);
  }

  seedSource(s: Source, t0: number): void {
    const now = s.trajectory.kinematics(t0);
    s.current = now;
    s.shells.charge = now.charge;
    s.ring.seed((t) => s.trajectory.kinematics(t), t0, this.config.dt);
    s.shells.populate(s.ring, this.config);
  }

  /** Replace trajectories (keeps the source count in sync) and reseed. */
  setTrajectories(trajectories: readonly Trajectory[]): void {
    this.sources.length = 0;
    for (const tr of trajectories) this.sources.push(new Source(tr, this.config.ringSize));
    this.seed(this.time);
  }

  /**
   * Advance by dt: append each source's new state to its ring buffer and
   * (unless skipShells) resample the shells.
   */
  step(dt: number, skipShells = false): void {
    this.time += dt;
    for (let i = 0; i < this.sources.length; i++) {
      const s = this.sources[i];
      const prescribed = s.trajectory.kinematics(this.time);
      const k = (this.override && this.override(i, this.time, dt, s.current, prescribed)) || prescribed;
      s.current = k;
      s.shells.charge = k.charge;
      s.ring.push(k);
      if (!skipShells) s.shells.populate(s.ring, this.config);
    }
  }

  /** Resample every source's shells (after a config change such as mT or scaleL). */
  repopulate(): void {
    for (const s of this.sources) s.shells.populate(s.ring, this.config);
  }

  get shellHistories(): ShellHistory[] {
    return this.sources.map((s) => s.shells);
  }

  get trajectories(): Trajectory[] {
    return this.sources.map((s) => s.trajectory);
  }
}
