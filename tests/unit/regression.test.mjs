// The ported core must reproduce the original solver's CPU-side suite numbers exactly.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runCase, emulatedBackend } from '../../dist/lib/core/index.js';

const base = JSON.parse(readFileSync(new URL('../../reference/baseline_suite.json', import.meta.url)));
const KEYS = ['cpuF32L2', 'cpuF64L2', 'coordRoundL2'];
// Bit-exact inside Chromium (verified); Node's V8 math library rounds a few transcendental
// calls differently, which the γ=70 cases amplify to ~1e-7 relative at γ=70. Hence the tolerance.
const same = (a, b, label) => {
  for (const k of KEYS) {
    const rel = Math.abs(a[k] - b[k]) / Math.max(0.01, Math.abs(b[k])); // percent units; floor for tiny errors
    assert.ok(rel < 1e-6, `${label}.${k}: port ${a[k]} vs original ${b[k]} (rel ${rel})`);
  }
};

test('M_T=128 Norm-β and M_T=512 4-vel rows match the original bit for bit', () => {
  for (const row of base.rows) {
    const b = row.beta;
    same(runCase({ beta: b, motion: 'synchrotron', mT: 128, kBin: 9, velMode: 1, nShock: 256 }, emulatedBackend), row.r128Norm, `β=${b} 128`);
    same(runCase({ beta: b, motion: 'synchrotron', mT: 512, kBin: 11, velMode: 2, nShock: 256 }, emulatedBackend), row.r5124Vel, `β=${b} 512`);
  }
});

test('grid comparison and far-field cases match the original', () => {
  const grid = (gridMode) => runCase({ beta: 0.99, motion: 'synchrotron', mT: 128, kBin: 9, velMode: 2, nShock: 256, rMax: 220, rEval: 1.15, gridMode }, emulatedBackend);
  same(grid('compactified'), base.comp128Reach220, 'comp');
  same(grid('log'), base.log128Reach220, 'log');
  same(grid('power2'), base.pow128Reach220, 'pow2');
  same(grid('uniform'), base.unif128Reach220, 'unif');
  same(runCase({ beta: 0.8, motion: 'uniform', mT: 128, kBin: 9, velMode: 2, nShock: 64, rMax: 18, rEval: 45, gridMode: 'compactified' }, emulatedBackend), base.compFar45, 'far comp');
});

test('γ=70.7 convergence diagnostics (1024, 2048 shells) match the original', () => {
  same(runCase({ beta: 0.9999, motion: 'synchrotron', mT: 1024, kBin: 12, velMode: 2, nShock: 256 }, emulatedBackend), base.diag1024, '1024');
  same(runCase({ beta: 0.9999, motion: 'synchrotron', mT: 2048, kBin: 13, velMode: 2, nShock: 256 }, emulatedBackend), base.diag2048, '2048');
});
