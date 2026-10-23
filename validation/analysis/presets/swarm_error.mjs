// How wrong is the legacy "swarm" preset's stored acceleration? Compares it with the finite-difference
// derivative of its own velocity, and with the corrected precessingEllipse. Run after `npm run build`.
import { legacyPresetKinematics, precessingEllipse } from '../../../dist/lib/core/index.js';
const c = 2.35, beta = 0.8, omega = 1.6, h = 1e-4;
const rel = (k, tr) => {
  let num = 0, den = 0, worst = 0;
  for (let i = 0; i <= 4000; i++) {
    const t = -50 + i * 0.025;
    const m = tr(t - h), p = tr(t + h), s = tr(t);
    const ax = (p.bx - m.bx) / (2 * h) / c, ay = (p.by - m.by) / (2 * h) / c;
    const e = Math.hypot(ax - s.bdx, ay - s.bdy), a = Math.hypot(ax, ay);
    num += e * e; den += a * a; worst = Math.max(worst, e / Math.max(1e-12, a));
  }
  return { rms: Math.sqrt(num / den) * 100, worst: worst * 100 };
};
console.log('index  legacy RMS %  legacy max %  corrected RMS %');
let lo = Infinity, hi = 0;
for (let i = 0; i < 32; i++) {
  const L = rel(i, (t) => legacyPresetKinematics('swarm', { index: i, count: 32, beta, omega, c }, t));
  const P = precessingEllipse({ index: i, beta, omega, c });
  const F = rel(i, (t) => P.kinematics(t));
  lo = Math.min(lo, L.rms); hi = Math.max(hi, L.rms);
  console.log(String(i).padStart(5), L.rms.toFixed(2).padStart(12), L.worst.toFixed(2).padStart(12), F.rms.toExponential(1).padStart(15));
}
console.log(`legacy RMS acceleration error range: ${lo.toFixed(1)}–${hi.toFixed(1)}%`);
