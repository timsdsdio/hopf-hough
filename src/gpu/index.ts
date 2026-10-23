export * from './solver.js';
export { LEGACY_FRAGMENT_SHADER, LEGACY_VERTEX_SHADER } from './legacyShader.js';
export { FRAGMENT_SHADER, VERTEX_SHADER } from './shader.js';
import type { FieldBackend } from '../core/suite.js';
import type { GpuSolver } from './solver.js';

/** Verification backend that evaluates through real GPU readback. */
export function gpuBackend(solver: GpuSolver): FieldBackend {
  return {
    name: 'webgl2-readback',
    evaluate(sim, pts) {
      solver.upload(sim.shellHistories, sim.config.mT);
      return solver.readback(pts, sim.config);
    },
  };
}
