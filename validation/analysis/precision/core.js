// Standalone extraction of the Float32 emulation path of hopf_hough_em_solver.html
// (synchrotron benchmark, N_p = 1), instrumented with per-stage rounding hooks.
// Faithful to: getExactTrajectoryKinematics, sampleFromRingBuffer, populateShellsFromRingBuffer,
// packVelocityForStorage, getKnotBetaFloat32, computeBernsteinAlphaFloat32,
// sampleSourceHistoryFloat32, evalCancellationFreeKappaAndUFloat32,
// evaluateShaderEquivalentFloat32, evaluateExactAnalyticalLW, getPulseCrossingSweepPoints.
'use strict';
const C_LIGHT = 2.35, RING_SIZE = 16384, DT_STEP = 0.00675;
const f32 = Math.fround, id = (x) => x;
const state = { omega: 1.6, scaleL: 1.35, softEps: 0.06, betaMax: 0.99995 };

function kin(t, beta) {
  const omega = state.omega, v = beta * C_LIGHT;
  const rOrb = Math.min(0.85, v / Math.max(0.5, omega));
  const w = v / Math.max(0.05, rOrb);
  const a = w * t, c = Math.cos(a), s = Math.sin(a);
  return { x: rOrb * c, y: rOrb * s, bx: -beta * s, by: beta * c,
           bdx: (-beta * w * c) / C_LIGHT, bdy: (-beta * w * s) / C_LIGHT };
}

// ---------- ring buffer (Float64, exact kinematics) ----------
function makeSource(beta, tStart, nSteps) {
  const src = { ringHead: 0, ringX: new Float64Array(RING_SIZE), ringY: new Float64Array(RING_SIZE),
    ringBx: new Float64Array(RING_SIZE), ringBy: new Float64Array(RING_SIZE),
    ringBdx: new Float64Array(RING_SIZE), ringBdy: new Float64Array(RING_SIZE), beta };
  for (let k = 0; k < RING_SIZE; k++) {
    const q = kin(tStart - k * DT_STEP, beta), r = (RING_SIZE - k) % RING_SIZE;
    src.ringX[r] = q.x; src.ringY[r] = q.y; src.ringBx[r] = q.bx; src.ringBy[r] = q.by;
    src.ringBdx[r] = q.bdx; src.ringBdy[r] = q.bdy;
  }
  let t = tStart;
  for (let st = 0; st < nSteps; st++) {   // stepSimulation: t += dt, push exact kinematics
    t += DT_STEP;
    const q = kin(t, beta);
    src.ringHead = (src.ringHead + 1) % RING_SIZE;
    const h = src.ringHead;
    src.ringX[h] = q.x; src.ringY[h] = q.y; src.ringBx[h] = q.bx; src.ringBy[h] = q.by;
    src.ringBdx[h] = q.bdx; src.ringBdy[h] = q.bdy;
  }
  src.tEval = t;
  return src;
}

function sampleFromRingBuffer(src, stepsBackFloat) {
  const sC = Math.max(0.0, Math.min(RING_SIZE - 3.001, stepsBackFloat));
  const k1 = Math.floor(sC), k0 = Math.max(0, k1 - 1), k2 = Math.min(RING_SIZE - 1, k1 + 1), k3 = Math.min(RING_SIZE - 1, k1 + 2);
  const t = sC - k1, t2 = t * t, t3 = t2 * t;
  const H = src.ringHead, R = RING_SIZE;
  const i0 = (H - k0 + R) % R, i1 = (H - k1 + R) % R, i2 = (H - k2 + R) % R, i3 = (H - k3 + R) % R;
  const ss = C_LIGHT * DT_STEP;
  const tx1 = -src.ringBx[i1] * ss, ty1 = -src.ringBy[i1] * ss, tx2 = -src.ringBx[i2] * ss, ty2 = -src.ringBy[i2] * ss;
  const h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
  const x = h00 * src.ringX[i1] + h10 * tx1 + h01 * src.ringX[i2] + h11 * tx2;
  const y = h00 * src.ringY[i1] + h10 * ty1 + h01 * src.ringY[i2] + h11 * ty2;
  const w0 = -0.5 * t3 + t2 - 0.5 * t, w1 = 1.5 * t3 - 2.5 * t2 + 1, w2 = -1.5 * t3 + 2 * t2 + 0.5 * t, w3 = 0.5 * t3 - 0.5 * t2;
  let rawBx, rawBy, bdx, bdy;
  if (k1 === 0) {
    const bdx1 = -src.ringBdx[i1] * ss, bdy1 = -src.ringBdy[i1] * ss, bdx2 = -src.ringBdx[i2] * ss, bdy2 = -src.ringBdy[i2] * ss;
    rawBx = h00 * src.ringBx[i1] + h10 * bdx1 + h01 * src.ringBx[i2] + h11 * bdx2;
    rawBy = src.ringBy[i1] * h00 + h10 * bdy1 + h01 * src.ringBy[i2] + h11 * bdy2;
    bdx = src.ringBdx[i1] + t * (src.ringBdx[i2] - src.ringBdx[i1]);
    bdy = src.ringBdy[i1] + t * (src.ringBdy[i2] - src.ringBdy[i1]);
  } else {
    rawBx = src.ringBx[i0] * w0 + src.ringBx[i1] * w1 + src.ringBx[i2] * w2 + src.ringBx[i3] * w3;
    rawBy = src.ringBy[i0] * w0 + src.ringBy[i1] * w1 + src.ringBy[i2] * w2 + src.ringBy[i3] * w3;
    bdx = src.ringBdx[i0] * w0 + src.ringBdx[i1] * w1 + src.ringBdx[i2] * w2 + src.ringBdx[i3] * w3;
    bdy = src.ringBdy[i0] * w0 + src.ringBdy[i1] * w1 + src.ringBdy[i2] * w2 + src.ringBdy[i3] * w3;
  }
  const bM1 = Math.hypot(src.ringBx[i1], src.ringBy[i1]), bM2 = Math.hypot(src.ringBx[i2], src.ringBy[i2]);
  const bT = bM1 + t * (bM2 - bM1), rawM = Math.hypot(rawBx, rawBy), sc = rawM > 1e-12 ? bT / rawM : 1;
  return { x, y, bx: rawBx * sc, by: rawBy * sc, bdx, bdy };
}

// ---------- shell population with storage-rounding hooks ----------
// cfg.RS: R-shell computation/storage rounding (f32 or id); cfg.XS: knot position storage;
// cfg.US: u storage; cfg.AS: acceleration storage; cfg.clampLoop: apply the f32 chord-shrink loop
// cfg.split: also store lo parts (x - f32(x)) for the df64 fix
function populate(src, M, cfg) {
  const RS = cfg.RS, XS = cfg.XS, US = cfg.US, AS = cfg.AS;
  const dTheta = Math.PI / M, orbitLambda = (2 * Math.PI * C_LIGHT) / Math.max(0.5, state.omega);
  const K = { M, R: new Float64Array(M), X: new Float64Array(M), Y: new Float64Array(M),
    Ux: new Float64Array(M), Uy: new Float64Array(M), Bdx: new Float64Array(M), Bdy: new Float64Array(M),
    Xlo: new Float64Array(M), Ylo: new Float64Array(M), Rlo: new Float64Array(M) };
  for (let m = 0; m < M; m++) {
    const th = (m + 0.5) * dTheta;
    const r64 = state.scaleL * Math.tan(0.5 * th);
    const rS = RS === id ? r64 : f32(f32(state.scaleL) * f32(Math.tan(f32(0.5 * th))));
    const dR = (state.scaleL * dTheta) / (1 + Math.cos(th));
    const sb = (rS / C_LIGHT) / DT_STEP, filterRatio = dR / (0.5 * orbitLambda);
    let smp;
    if (filterRatio > 0.25) {
      const dS = Math.min((dR / C_LIGHT) / DT_STEP, (2 * Math.PI / state.omega) / DT_STEP);
      const s0 = sampleFromRingBuffer(src, sb - 0.35 * dS), s1 = sampleFromRingBuffer(src, sb), s2 = sampleFromRingBuffer(src, sb + 0.35 * dS);
      const wN = Math.exp(-Math.pow(filterRatio, 4));
      const aBx = 0.25 * s0.bx + 0.5 * s1.bx + 0.25 * s2.bx, aBy = 0.25 * s0.by + 0.5 * s1.by + 0.25 * s2.by;
      const bT = 0.25 * Math.hypot(s0.bx, s0.by) + 0.5 * Math.hypot(s1.bx, s1.by) + 0.25 * Math.hypot(s2.bx, s2.by);
      const aM = Math.hypot(aBx, aBy), sc = aM > 1e-12 ? bT / aM : 1;
      smp = { x: s1.x, y: s1.y, bx: aBx * sc, by: aBy * sc,
        bdx: (0.25 * s0.bdx + 0.5 * s1.bdx + 0.25 * s2.bdx) * wN, bdy: (0.25 * s0.bdy + 0.5 * s1.bdy + 0.25 * s2.bdy) * wN };
    } else smp = sampleFromRingBuffer(src, sb);
    const bMag = Math.min(0.9999995, Math.hypot(smp.bx, smp.by));
    const g = 1 / Math.sqrt(Math.max(1e-14, (1 - bMag) * (1 + bMag)));
    K.R[m] = rS; K.X[m] = XS(smp.x); K.Y[m] = XS(smp.y);
    K.Xlo[m] = smp.x - f32(smp.x); K.Ylo[m] = smp.y - f32(smp.y); K.Rlo[m] = 0; // R is f32-exact when RS=f32
    K.Ux[m] = US(g * smp.bx); K.Uy[m] = US(g * smp.by);
    K.Bdx[m] = AS(smp.bdx); K.Bdy[m] = AS(smp.bdy);
  }
  if (cfg.clampLoop) {
    const maxSub = f32(state.betaMax);
    let fired = 0;
    for (let m = 0; m < M - 1; m++) {
      const dR = f32(Math.max(f32(1e-6), f32(K.R[m + 1] - K.R[m])));
      const dx = f32(K.X[m + 1] - K.X[m]), dy = f32(K.Y[m + 1] - K.Y[m]);
      const cs = f32(f32(Math.sqrt(f32(f32(dx * dx) + f32(dy * dy)))) / dR);
      if (cs > maxSub) { fired++; const sh = f32(maxSub / cs); K.X[m + 1] = f32(K.X[m] + f32(dx * sh)); K.Y[m + 1] = f32(K.Y[m] + f32(dy * sh)); }
    }
    K.clampFired = fired;
  }
  return K;
}

// ---------- spline sampler with arithmetic hooks ----------
// P: rounding for position/derivative arithmetic; V: velocity (|u| interp, bHat); A: acceleration CR
function makeLen(F) { return (x, y) => F(Math.sqrt(F(F(x * x) + F(y * y)))); }
function sampleHist(K, mFloat, F, opt) {
  const P = F.P, V = F.V, A = F.A, lenP = makeLen(P), lenV = makeLen(V), M = K.M;
  const mC = P(Math.max(0, Math.min(P(M - 1.0001), P(mFloat))));
  const m1 = Math.floor(mC), m0 = Math.max(0, m1 - 1), m2 = Math.min(M - 1, m1 + 1), m3 = Math.min(M - 1, m1 + 2);
  const t = P(mC - m1), t2 = P(t * t), t3 = P(t2 * t);
  const r1 = K.R[m1], r2 = K.R[m2];
  const dR = m2 > m1 ? P(Math.max(P(1e-6), P(r2 - r1))) : P(1e-6);
  const s = m2 > m1 ? t : 0, dRSub = P(s * dR);
  const x1 = K.X[m1], y1 = K.Y[m1], x2 = K.X[m2], y2 = K.Y[m2];
  let dxC = P(x2 - x1), dyC = P(y2 - y1);
  if (F.splitStore) { // chord from hi/lo pairs: (x2-x1) is Sterbenz-exact for adjacent knots; add lo difference
    dxC = P(dxC + P(f32(K.Xlo[m2]) - f32(K.Xlo[m1]))); dyC = P(dyC + P(f32(K.Ylo[m2]) - f32(K.Ylo[m1])));
  }
  const invDR = P(1 / dR), maxS = P(state.betaMax);
  let vSx = P(dxC * invDR), vSy = P(dyC * invDR);
  const vSm = lenP(vSx, vSy);
  let secClamped = false;
  if (opt.limiter && vSm > maxS) { const c = P(maxS / vSm); vSx = P(vSx * c); vSy = P(vSy * c); secClamped = true; }
  const kb = (mi) => { const ux = K.Ux[mi], uy = K.Uy[mi]; const uS = P(P(ux * ux) + P(uy * uy)); const ig = P(1 / P(Math.sqrt(P(1 + uS)))); return [P(ux * ig), P(uy * ig)]; };
  let [b1x, b1y] = kb(m1), [b2x, b2y] = kb(m2);
  if (opt.limiter) {
    const b1m = lenP(b1x, b1y); if (b1m > maxS) { b1x = P(b1x * P(maxS / b1m)); b1y = P(b1y * P(maxS / b1m)); }
    const b2m = lenP(b2x, b2y); if (b2m > maxS) { b2x = P(b2x * P(maxS / b2m)); b2y = P(b2y * P(maxS / b2m)); }
  }
  const v1Rx = P(P(3 * vSx) + P(b1x + b2x)), v1Ry = P(P(3 * vSy) + P(b1y + b2y));
  let alpha = 1;
  if (opt.limiter) {
    const v1Sq = P(P(v1Rx * v1Rx) + P(v1Ry * v1Ry)), mSq = P(maxS * maxS);
    if (v1Sq > mSq) {
      const ddx = P(v1Rx - vSx), ddy = P(v1Ry - vSy), a = P(P(ddx * ddx) + P(ddy * ddy));
      if (a <= P(1e-12)) alpha = 0; else {
        const bH = P(P(vSx * ddx) + P(vSy * ddy)), c = P(P(P(vSx * vSx) + P(vSy * vSy)) - mSq);
        if (c >= 0) alpha = 0; else { const disc = P(Math.max(0, P(P(bH * bH) - P(a * c)))); alpha = P(Math.max(0, Math.min(1, P(P(-bH + P(Math.sqrt(disc))) / a)))); }
      }
    }
  }
  const oma = P(1 - alpha);
  const v0x = P(P(oma * vSx) - P(alpha * b1x)), v0y = P(P(oma * vSy) - P(alpha * b1y));
  const v1x = P(vSx + P(alpha * P(v1Rx - vSx))), v1y = P(vSy + P(alpha * P(v1Ry - vSy)));
  const v2x = P(P(oma * vSx) - P(alpha * b2x)), v2y = P(P(oma * vSy) - P(alpha * b2y));
  const tx1 = P(v0x * dR), ty1 = P(v0y * dR), tx2 = P(v2x * dR), ty2 = P(v2y * dR);
  const s2 = P(s * s), s3 = P(s2 * s);
  const h10 = P(s3 - P(2 * s2) + s), h01 = P(P(-2 * s3) + P(3 * s2)), h11 = P(s3 - s2);
  const dxSub = P(P(h01 * dxC) + P(h10 * tx1) + P(h11 * tx2)), dySub = P(P(h01 * dyC) + P(h10 * ty1) + P(h11 * ty2));
  const oms = P(1 - s), B0 = P(oms * oms), B1 = P(2 * P(s * oms)), B2 = s2;
  const dzdRx = P(P(B0 * v0x) + P(B1 * v1x) + P(B2 * v2x)), dzdRy = P(P(B0 * v0y) + P(B1 * v1y) + P(B2 * v2y));
  const sVx = -dzdRx, sVy = -dzdRy, sVm = lenV(sVx, sVy);
  const w0 = A(A(-0.5 * t3) + A(t2 - A(0.5 * t))), w1 = A(A(1.5 * t3) - A(2.5 * t2) + 1), w2 = A(A(-1.5 * t3) + A(2 * t2) + A(0.5 * t)), w3 = A(A(0.5 * t3) - A(0.5 * t2));
  let bHatX = sVm > 1e-7 ? V(sVx / sVm) : 1, bHatY = sVm > 1e-7 ? V(sVy / sVm) : 0;
  let uT, bMag, omb2;
  const uM1 = lenV(K.Ux[m1], K.Uy[m1]), uM2 = lenV(K.Ux[m2], K.Uy[m2]);
  uT = V(uM1 + V(t * V(uM2 - uM1)));
  let bdx = A(A(K.Bdx[m0] * w0) + A(K.Bdx[m1] * w1) + A(K.Bdx[m2] * w2) + A(K.Bdx[m3] * w3));
  let bdy = A(A(K.Bdy[m0] * w0) + A(K.Bdy[m1] * w1) + A(K.Bdy[m2] * w2) + A(K.Bdy[m3] * w3));
  if (opt.exactKin) { // model ablation: replace |u|, direction and acceleration by exact values at this R
    const Rq = (r1 + dRSub);
    const q = kin(K.tEval - Rq / C_LIGHT, K.beta);
    const bm = Math.hypot(q.bx, q.by);
    if (opt.exactKin.u) uT = K.beta / Math.sqrt((1 - K.beta) * (1 + K.beta));
    if (opt.exactKin.dir) { bHatX = q.bx / bm; bHatY = q.by / bm; }
    if (opt.exactKin.acc) { bdx = q.bdx; bdy = q.bdy; }
  }
  const uSq = V(uT * uT), gam = V(Math.sqrt(V(1 + uSq)));
  bMag = V(uT / gam); omb2 = V(1 / V(1 + uSq));
  return { x1, y1, dxSub, dySub, r1, dRSub, dRdm: dR, dzdRx, dzdRy, bHatX, bHatY, bMag, oneMinusBeta2: omb2, bdx, bdy, alpha, secClamped, m1 };
}

// ---------- root + kernel with arithmetic hooks ----------
// F.Q: query rounding; F.Rt: root-finding arithmetic; F.Kr: kernel arithmetic
function evalF32(K, pxIn, pyIn, F, opt) {
  const Q = F.Q, Rt = F.Rt, Kr = F.Kr, lenR = makeLen(Rt), M = K.M;
  const px = Q(pxIn), py = Q(pyIn), kBin = opt.kBin, nNewt = opt.nNewton;
  let mLow = 0, mHigh = Rt(M - 1.001);
  let st = { alphaMin: 1, sec: false };
  const S = (m) => { const h = sampleHist(K, m, F, opt); if (h.alpha < st.alphaMin) st.alphaMin = h.alpha; if (h.secClamped) st.sec = true; return h; };
  const fOf = (h, phase) => {
    if (opt.df64root && !(opt.df64NewtonOnly && phase === 'bis')) return fDF(K, px, py, h, F);
    let dx, dy, dist;
    if (opt.altOrder) { // emulate a different-but-legal compiler schedule: reassociated subtraction + fused x*x+y*y
      dx = Rt(px - Rt(h.x1 + h.dxSub)); dy = Rt(py - Rt(h.y1 + h.dySub));
      dist = Rt(Math.sqrt(Rt(dx * dx + Rt(dy * dy))));
    } else { dx = Rt(Rt(px - h.x1) - h.dxSub); dy = Rt(Rt(py - h.y1) - h.dySub); dist = lenR(dx, dy); }
    return { f: Rt(Rt(h.r1 - dist) + h.dRSub), dx, dy, dist };
  };
  for (let i = 0; i < kBin; i++) {
    const mMid = Rt(0.5 * Rt(mLow + mHigh)), h = S(mMid), r = fOf(h);
    const below = (opt.df64root && !opt.df64NewtonOnly) ? (r.f < 0) : (Rt(h.r1 + h.dRSub) < r.dist); // solver: rShell < dist
    if (below) mLow = mMid; else mHigh = mMid;
  }
  if (opt.df64NewtonOnly && opt.widen) { // f32 bisection is unreliable within ~df32/(kappa dR) of the root: widen, then df64 bisection
    mLow = Math.max(0, mLow - opt.widen); mHigh = Math.min(Rt(M - 1.001), mHigh + opt.widen);
    for (let i = 0; i < opt.df64Bis; i++) { const mMid = Rt(0.5 * Rt(mLow + mHigh)), h = S(mMid), r = fDF(K, px, py, h, F); if (r.f < 0) mLow = mMid; else mHigh = mMid; }
  }
  let mRet = Rt(0.5 * Rt(mLow + mHigh));
  for (let it = 0; it < nNewt; it++) {
    const h = S(mRet), r = fOf(h);
    const nx = r.dist > 1e-6 ? Rt(r.dx / r.dist) : 1, ny = r.dist > 1e-6 ? Rt(r.dy / r.dist) : 0;
    const nD = Rt(Rt(nx * h.dzdRx) + Rt(ny * h.dzdRy));
    const kS = Rt(Math.max(Rt(opt.kappaSplineFloor), Rt(1 + nD)));
    const dfdm = Rt(Math.max(Rt(1e-7), Rt(h.dRdm * kS)));
    const cand = Rt(mRet - Rt(r.f / dfdm));
    if (opt.safeguard) { // keep bracket updated (safeguarded Newton)
      if (r.f < 0) mLow = mRet; else mHigh = mRet;
      mRet = (cand > mLow && cand < mHigh) ? cand : Rt(0.5 * Rt(mLow + mHigh));
    } else mRet = Rt(Math.max(mLow, Math.min(mHigh, cand)));
  }
  const ret = S(mRet);
  let dx, dy;
  if (opt.df64root) { const r = fDF(K, px, py, ret, F); dx = r.dx; dy = r.dy; }
  else { dx = Kr(Kr(px - ret.x1) - ret.dxSub); dy = Kr(Kr(py - ret.y1) - ret.dySub); }
  const r2 = Kr(Kr(dx * dx) + Kr(dy * dy)), r = Kr(Math.sqrt(r2));
  const se = Kr(state.softEps), rS2 = Kr(r2 + Kr(se * se)), rS = Kr(Math.sqrt(rS2));
  const nx = r > 1e-6 ? Kr(dx / r) : 1, ny = r > 1e-6 ? Kr(dy / r) : 0;
  const omb = Kr(ret.oneMinusBeta2 / Kr(1 + ret.bMag));
  const dX = Kr(nx - ret.bHatX), dY = Kr(ny - ret.bHatY), dS = Kr(Kr(dX * dX) + Kr(dY * dY));
  const kappa = Kr(Math.max(Kr(1e-7), Kr(omb + Kr(0.5 * Kr(ret.bMag * dS)))));
  const ux = Kr(Kr(omb * nx) + Kr(ret.bMag * dX)), uy = Kr(Kr(omb * ny) + Kr(ret.bMag * dY));
  const k3 = Kr(kappa * Kr(kappa * kappa));
  const vP = Kr(ret.oneMinusBeta2 / Kr(k3 * rS2));
  const cz = Kr(Kr(ux * ret.bdy) - Kr(uy * ret.bdx)), rP = Kr(1 / Kr(k3 * rS));
  const ex = Kr(Kr(vP * ux) + Kr(rP * Kr(ny * cz))), ey = Kr(Kr(vP * uy) + Kr(rP * Kr(-nx * cz)));
  return { ex, ey, mRet, Rret: ret.r1 + ret.dRSub, kappa, alphaMin: st.alphaMin, sec: st.sec };
}

// ---------- df64 (double-float, f32 pairs, no FMA) critical-path root function ----------
const SPLIT = 4097; // 2^12+1 for 24-bit mantissa
function twoSum(a, b) { const s = f32(a + b), bb = f32(s - a); const e = f32(f32(a - f32(s - bb)) + f32(b - bb)); return [s, e]; }
function quickTwoSum(a, b) { const s = f32(a + b); return [s, f32(b - f32(s - a))]; }
function split(a) { const t = f32(SPLIT * a), hi = f32(t - f32(t - a)); return [hi, f32(a - hi)]; }
function twoProd(a, b) { const p = f32(a * b); const [ah, al] = split(a), [bh, bl] = split(b);
  const e = f32(f32(f32(f32(ah * bh) - p) + f32(ah * bl) + f32(al * bh)) + f32(al * bl)); return [p, e]; }
function dfAdd(A, B) { let [s, e] = twoSum(A[0], B[0]); const [t, f] = twoSum(A[1], B[1]); e = f32(e + t); [s, e] = quickTwoSum(s, e); e = f32(e + f); return quickTwoSum(s, e); }
function dfNeg(A) { return [-A[0], -A[1]]; }
function dfMul(A, B) { let [p, e] = twoProd(A[0], B[0]); e = f32(e + f32(f32(A[0] * B[1]) + f32(A[1] * B[0]))); return quickTwoSum(p, e); }
function dfSqrt(A) { if (A[0] <= 0) return [0, 0]; const x = f32(1 / f32(Math.sqrt(A[0]))); const y = f32(A[0] * x);
  const [yy, ee] = twoProd(y, y); const d = dfAdd(A, dfNeg([yy, ee])); const corr = f32(f32(d[0] * x) * 0.5); return quickTwoSum(y, corr); }
// f = (r1 + rlo + dRSub) - |(px - x1 - x1lo) - dxSub|, all in df64; returns f32 f and f32 dx,dy
function fDF(K, px, py, h, F) {
  const m = h.m1;
  const xlo = F.splitStore ? f32(K.Xlo[m]) : 0, ylo = F.splitStore ? f32(K.Ylo[m]) : 0;
  const qxl = F.qlo ? F.qlo[0] : 0, qyl = F.qlo ? F.qlo[1] : 0;
  const DX = dfAdd(dfAdd(dfAdd(twoSum(px, -h.x1), [qxl, 0]), [-xlo, 0]), [-h.dxSub, 0]);
  const DY = dfAdd(dfAdd(dfAdd(twoSum(py, -h.y1), [qyl, 0]), [-ylo, 0]), [-h.dySub, 0]);
  const D2 = dfAdd(dfMul(DX, DX), dfMul(DY, DY)), dist = dfSqrt(D2);
  const Rr = dfAdd([h.r1, 0], [h.dRSub, 0]);
  const fdf = dfAdd(Rr, dfNeg(dist));
  return { f: f32(fdf[0] + fdf[1]), dx: f32(DX[0] + DX[1]), dy: f32(DY[0] + DY[1]), dist: f32(dist[0] + dist[1]) };
}

// ---------- exact Float64 reference ----------
function exactLW(px, py, t, beta) {
  let tLow = t - 260 / C_LIGHT, tHigh = t;
  for (let b = 0; b < 32; b++) { const tm = 0.5 * (tLow + tHigh), q = kin(tm, beta); const fm = C_LIGHT * (t - tm) - Math.hypot(px - q.x, py - q.y); if (fm > 0) tLow = tm; else tHigh = tm; }
  let tR = 0.5 * (tLow + tHigh);
  for (let it = 0; it < 12; it++) {
    const q = kin(tR, beta), dx = px - q.x, dy = py - q.y, d = Math.hypot(dx, dy), f = C_LIGHT * (t - tR) - d;
    if (Math.abs(f) < 1e-13) break;
    const nx = dx / d, ny = dy / d, df = -C_LIGHT * Math.max(1e-11, 1 - (nx * q.bx + ny * q.by));
    const c = tR - f / df; tR = (c >= tLow && c <= tHigh) ? c : 0.5 * (tLow + tHigh); if (f > 0) tLow = tR; else tHigh = tR;
  }
  const q = kin(tR, beta), dx = px - q.x, dy = py - q.y, r2 = dx * dx + dy * dy, r = Math.sqrt(r2);
  const rS2 = r2 + state.softEps ** 2, rS = Math.sqrt(rS2), nx = dx / r, ny = dy / r;
  // cancellation-free kappa in f64 (more accurate than the solver's 1 - n.beta, same value to ~1e-12 rel)
  const bm = Math.hypot(q.bx, q.by), omb2 = (1 - bm) * (1 + bm);
  const kappa = Math.max(1e-11, 1 - (nx * q.bx + ny * q.by)), k3 = kappa ** 3;
  const vP = omb2 / (k3 * rS2), ux = nx - q.bx, uy = ny - q.by, cz = ux * q.bdy - uy * q.bdx, rP = 1 / (k3 * rS);
  return { ex: vP * ux + rP * ny * cz, ey: vP * uy - rP * nx * cz, R: C_LIGHT * (t - tR), kappa };
}

function sweepPoints(rRet, tObs, beta, n) {
  const g = 1 / Math.sqrt((1 - beta) * (1 + beta));
  const tS = tObs - rRet / C_LIGHT, q = kin(tS, beta), bm = Math.hypot(q.bx, q.by);
  const cx = q.x + rRet * q.bx / bm, cy = q.y + rRet * q.by / bm;
  const rObs = Math.hypot(cx, cy), phi = Math.atan2(cy, cx), half = Math.min(0.45, Math.max(1e-6, 4 * g ** -3));
  const pts = [];
  for (let k = 0; k < n; k++) { const u = (k / (n - 1)) * 2 - 1, a = phi + u * half; const x64 = rObs * Math.cos(a), y64 = rObs * Math.sin(a); pts.push({ x: f32(x64), y: f32(y64), x64, y64, a }); }
  return { pts, half, rObs, phi };
}

module.exports = { C_LIGHT, DT_STEP, f32, id, state, kin, makeSource, populate, sampleHist, evalF32, exactLW, sweepPoints, dfAdd, twoSum };
