/**
 * Production shaders.
 *
 * Per fragment and per source: find the retarded radius R on the source's shell
 * history (f(R) = R − |x − z(R)| = 0), then evaluate the Liénard–Wiechert field.
 *
 * Switches (uniforms), each verified separately against the legacy behaviour:
 *  - u_precision: 0 = float32 root function (legacy), 1 = double-float root function
 *    with hi/lo knot positions (error-free transforms; every step guarded against reassociation by BAR).
 *  - u_certMode: 0 = Theorem 1 control-vector limiter (legacy), 1 = |z'|² Bernstein-coefficient
 *    certificate (sharper, never stricter), 2 = no limiter.
 *  - u_rootMode: 0 = legacy bisection on the spline + 2 clamped Newton steps,
 *    1 = bisection over stored knots, then safeguarded Newton inside the bracketing segment.
 */

export const VERTEX_SHADER = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

export const FRAGMENT_SHADER = `#version 300 es
precision highp float;
precision highp int;
in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_texPosVel;      // x_hi, y_hi, ux, uy
uniform sampler2D u_texAccChg;      // bdx, bdy, charge, R
uniform sampler2D u_texPosLo;       // x_lo, y_lo, -, -
uniform sampler2D u_texQueryCoords; // x_hi, y_hi, x_lo, y_lo
uniform int u_activeCount;
uniform int u_mT;
uniform int u_kBin;
uniform int u_newton;
uniform int u_velMode;
uniform int u_readbackMode;
uniform int u_readbackLayout;       // 0: (Ex, Ey, emitX0, emitY0)  1: (Ex, Ey, Bz, quality)
uniform int u_certMode;
uniform int u_rootMode;
uniform int u_precision;
uniform float u_betaMax;
uniform float u_c;
uniform float u_softEps;
uniform float u_scaleL;
uniform float u_one;                // 1.0 (kept for compatibility)
uniform uint u_zero;                // must be 0u; operand of BAR(), which blocks reassociation of error-free transforms
uniform float u_time;
uniform float u_aspect;
uniform vec2 u_viewCenter;
uniform float u_viewHalfHeight;
uniform int u_channel;              // 0 |E|, 1 E_rad, 2 E_vel, 3 B_z, 4 |S|
uniform float u_gain;
uniform int u_showWavePool;
uniform int u_showImpostors;
uniform int u_showMask;
uniform float u_maskThreshold;
uniform int u_taperOn;              // Nyquist prefilter/taper active on the CPU side
uniform float u_omega;              // its reference angular frequency
uniform float u_ringReach;          // depth of the stored history, c·dt·(ringSize − 4)

const float PI = 3.14159265359;

// ---------------- double-float arithmetic (pairs of float32) ----------------
// Every rounding step goes through BAR(), an identity the compiler cannot see through
// (XOR with a uniform that is 0). Some compilers (Apple's Metal, via ANGLE) reassociate
// float expressions; a guard on only some operations is not enough, because an unguarded
// sum such as (a - x) + (b - y) can be regrouped as (a + b) - (x + y), which loses the very
// bits the error-free transform recovers. Splitting uses bit masking, so the partial
// products are exact whether or not the compiler fuses multiply-adds.
float BAR(float x) { return uintBitsToFloat(floatBitsToUint(x) ^ u_zero); }
float ADD(float a, float b) { return BAR(a + b); }
float SUB(float a, float b) { return BAR(a - b); }
float MUL(float a, float b) { return BAR(a * b); }
vec2 twoSum(float a, float b) {
  float s = ADD(a, b);
  float bb = SUB(s, a);
  return vec2(s, ADD(SUB(a, SUB(s, bb)), SUB(b, bb)));
}
vec2 quickTwoSum(float a, float b) {
  float s = ADD(a, b);
  return vec2(s, SUB(b, SUB(s, a)));
}
vec2 splitF(float a) {
  float hi = BAR(uintBitsToFloat(floatBitsToUint(a) & 0xFFFFF000u));
  return vec2(hi, SUB(a, hi));
}
vec2 twoProd(float a, float b) {
  float p = MUL(a, b);
  vec2 A = splitF(a);
  vec2 B = splitF(b);
  float e = ADD(ADD(ADD(SUB(MUL(A.x, B.x), p), MUL(A.x, B.y)), MUL(A.y, B.x)), MUL(A.y, B.y));
  return vec2(p, e);
}
vec2 dfAdd(vec2 A, vec2 B) {
  vec2 s = twoSum(A.x, B.x);
  vec2 t = twoSum(A.y, B.y);
  s.y = ADD(s.y, t.x);
  s = quickTwoSum(s.x, s.y);
  s.y = ADD(s.y, t.y);
  return quickTwoSum(s.x, s.y);
}
vec2 dfMul(vec2 A, vec2 B) {
  vec2 p = twoProd(A.x, B.x);
  p.y = ADD(p.y, ADD(MUL(A.x, B.y), MUL(A.y, B.x)));
  return quickTwoSum(p.x, p.y);
}
vec2 dfSqrt(vec2 A) {
  if (A.x <= 0.0) return vec2(0.0);
  float x = 1.0 / sqrt(A.x);
  float y = MUL(A.x, x);
  vec2 yy = twoProd(y, y);
  vec2 d = dfAdd(A, -yy);
  return quickTwoSum(y, MUL(MUL(d.x, x), 0.5));
}

// ---------------- shell history access ----------------
struct Knot { vec2 phi; vec2 plo; vec2 u; vec2 acc; float chg; float r; };

Knot knotAt(int s, int m) {
  vec4 pv = texelFetch(u_texPosVel, ivec2(m, s), 0);
  vec4 ac = texelFetch(u_texAccChg, ivec2(m, s), 0);
  vec4 lo = texelFetch(u_texPosLo, ivec2(m, s), 0);
  Knot k;
  k.phi = pv.xy; k.u = pv.zw; k.acc = ac.xy; k.chg = ac.z; k.r = ac.w;
  k.plo = (u_precision == 1) ? lo.xy : vec2(0.0);
  return k;
}

vec2 unpackKnotBeta(vec2 raw) {
  if (u_velMode == 2) return raw / sqrt(1.0 + dot(raw, raw));
  return raw;
}

// Theorem 1: largest alpha keeping the middle control vector inside the speed ball.
float alphaLegacy(vec2 vSec, vec2 v1Raw, float maxSub) {
  float maxSq = maxSub * maxSub;
  if (dot(v1Raw, v1Raw) <= maxSq) return 1.0;
  vec2 d = v1Raw - vSec;
  float a = dot(d, d);
  if (a <= 1e-12) return 0.0;
  float bHalf = dot(vSec, d);
  float c = dot(vSec, vSec) - maxSq;
  if (c >= 0.0) return 0.0;
  float disc = max(0.0, bHalf * bHalf - a * c);
  return clamp((-bHalf + sqrt(disc)) / a, 0.0, 1.0);
}

// Smallest root in (0, cap) of A a^2 + B a + C with C <= 0; returns cap if none.
float firstRoot(float A, float B, float C, float cap) {
  float al = cap;
  if (abs(A) < 1e-30) {
    if (B > 0.0) { float r = -C / B; if (r > 0.0 && r < al) al = r; }
  } else {
    float disc = B * B - 4.0 * A * C;
    if (disc >= 0.0) {
      float q = sqrt(disc);
      float r1 = (-B - q) / (2.0 * A);
      float r2 = (-B + q) / (2.0 * A);
      if (r1 > 0.0 && r1 < al) al = r1;
      if (r2 > 0.0 && r2 < al) al = r2;
    }
  }
  return al;
}

// |z'|^2 = sum_k B_{k,4}(s) c_k with c1 = V0.V1, c2 = (2 V1.V1 + V0.V2)/3, c3 = V1.V2 (c0, c4 are always inside).
// Each c_k is quadratic in alpha; take the first alpha where any reaches betaMax^2.
float alphaSharp(vec2 v, vec2 m0, vec2 m1, float bm) {
  vec2 D0 = m0 - v;
  vec2 D1 = 2.0 * v - m0 - m1;
  vec2 D2 = m1 - v;
  float pp = dot(v, v) - bm * bm;
  if (pp >= 0.0) return 0.0;
  float al = 1.0;
  al = firstRoot(dot(D0, D1), dot(v, D1) + dot(D0, v), pp, al);
  al = firstRoot(dot(D1, D2), dot(v, D2) + dot(D1, v), pp, al);
  al = firstRoot((2.0 * dot(D1, D1) + dot(D0, D2)) / 3.0, (4.0 * dot(v, D1) + dot(v, D2) + dot(D0, v)) / 3.0, pp, al);
  return al;
}

struct Seg {
  vec2 p1hi; vec2 p1lo;   // knot m1 position
  vec2 dPosSub;           // z(s) - z(m1)
  float r1; float dRSub; float dR;
  vec2 dzdR;              // dz/dR at s
  vec2 bHat; float bMag; float omb2;
  vec2 acc; float chg;
  float duRel;            // |u2 - u1| / (mean |u| + 0.1) across the segment
};

// Segment m1 at fraction t in [0, 1). Index and fraction are kept separate so that float32
// resolution of a combined coordinate (≈ 6e-5 near m = 1000) never limits the retarded point.
Seg sampleSegAt(int s, int m1In, float tIn) {
  int m1 = clamp(m1In, 0, u_mT - 2);
  float t = clamp(tIn, 0.0, 0.99999);
  int m0 = max(0, m1 - 1);
  int m2 = min(u_mT - 1, m1 + 1);
  int m3 = min(u_mT - 1, m1 + 2);
  float t2 = t * t;
  float t3 = t2 * t;
  Knot k0 = knotAt(s, m0);
  Knot k1 = knotAt(s, m1);
  Knot k2 = knotAt(s, m2);
  Knot k3 = knotAt(s, m3);

  Seg g;
  g.r1 = k1.r;
  float dR = m2 > m1 ? max(1e-6, k2.r - k1.r) : 1e-6;
  float sRad = m2 > m1 ? t : 0.0;
  g.dR = dR;
  g.dRSub = sRad * dR;
  g.p1hi = k1.phi;
  g.p1lo = k1.plo;

  float maxSub = u_betaMax;
  vec2 dChord = (k2.phi - k1.phi) + (k2.plo - k1.plo);
  vec2 vSec = dChord / dR;
  float vSecMag = length(vSec);
  if (vSecMag > maxSub) vSec *= (maxSub / vSecMag);
  vec2 b1 = unpackKnotBeta(k1.u);
  vec2 b2 = unpackKnotBeta(k2.u);
  float b1Mag = length(b1);
  if (b1Mag > maxSub) b1 *= (maxSub / b1Mag);
  float b2Mag = length(b2);
  if (b2Mag > maxSub) b2 *= (maxSub / b2Mag);

  float alpha = 1.0;
  vec2 v1Raw = 3.0 * vSec + b1 + b2;
  if (u_certMode == 0) alpha = alphaLegacy(vSec, v1Raw, maxSub);
  else if (u_certMode == 1) alpha = alphaSharp(vSec, -b1, -b2, maxSub);
  float oma = 1.0 - alpha;
  vec2 v0 = oma * vSec - alpha * b1;
  vec2 v1 = vSec + alpha * (v1Raw - vSec);
  vec2 v2 = oma * vSec - alpha * b2;

  float s2 = sRad * sRad;
  float s3 = s2 * sRad;
  float h10 = s3 - 2.0 * s2 + sRad;
  float h01 = -2.0 * s3 + 3.0 * s2;
  float h11 = s3 - s2;
  g.dPosSub = h01 * dChord + h10 * (v0 * dR) + h11 * (v2 * dR);
  float oms = 1.0 - sRad;
  g.dzdR = (oms * oms) * v0 + (2.0 * sRad * oms) * v1 + s2 * v2;
  vec2 splineVel = -g.dzdR;
  float splineVelMag = length(splineVel);

  float w0 = -0.5 * t3 + t2 - 0.5 * t;
  float w1 = 1.5 * t3 - 2.5 * t2 + 1.0;
  float w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
  float w3 = 0.5 * t3 - 0.5 * t2;
  g.acc = k0.acc * w0 + k1.acc * w1 + k2.acc * w2 + k3.acc * w3;
  g.chg = k1.chg;
  g.duRel = length(k2.u - k1.u) / (0.5 * (length(k1.u) + length(k2.u)) + 0.1);

  if (u_velMode == 2) {
    float uTarget = mix(length(k1.u), length(k2.u), t);
    g.bHat = splineVelMag > 1e-7 ? splineVel / splineVelMag : vec2(1.0, 0.0);
    float uSq = uTarget * uTarget;
    g.bMag = uTarget / sqrt(1.0 + uSq);
    g.omb2 = 1.0 / (1.0 + uSq);
  } else if (u_velMode == 1) {
    float bTarget = mix(length(k1.u), length(k2.u), t);
    g.bHat = splineVelMag > 1e-7 ? splineVel / splineVelMag : vec2(1.0, 0.0);
    g.bMag = min(0.9999995, bTarget);
    g.omb2 = (1.0 - g.bMag) * (1.0 + g.bMag);
  } else {
    vec2 rawVel = k0.u * w0 + k1.u * w1 + k2.u * w2 + k3.u * w3;
    g.bMag = min(0.9999995, length(rawVel));
    g.bHat = g.bMag > 1e-7 ? rawVel / g.bMag : vec2(1.0, 0.0);
    g.omb2 = (1.0 - g.bMag) * (1.0 + g.bMag);
  }
  return g;
}

Seg sampleSeg(int s, float mFloat) {
  float mClamped = clamp(mFloat, 0.0, float(u_mT) - 1.0001);
  float mFloor = floor(mClamped);
  return sampleSegAt(s, int(mFloor), mClamped - mFloor);
}

// Root function f = R - |x - z| and the separation vector d = x - z (both single precision on return).
struct RootEval { float f; vec2 d; float dist; };

RootEval rootFn(vec2 xhi, vec2 xlo, vec2 phi, vec2 plo, vec2 dPosSub, float r1, float dRSub) {
  RootEval o;
  if (u_precision == 1) {
    vec2 DX = dfAdd(dfAdd(dfAdd(twoSum(xhi.x, -phi.x), vec2(xlo.x, 0.0)), vec2(-plo.x, 0.0)), vec2(-dPosSub.x, 0.0));
    vec2 DY = dfAdd(dfAdd(dfAdd(twoSum(xhi.y, -phi.y), vec2(xlo.y, 0.0)), vec2(-plo.y, 0.0)), vec2(-dPosSub.y, 0.0));
    vec2 dist = dfSqrt(dfAdd(dfMul(DX, DX), dfMul(DY, DY)));
    vec2 F = dfAdd(twoSum(r1, dRSub), -dist);
    o.f = F.x + F.y;
    o.d = vec2(DX.x + DX.y, DY.x + DY.y);
    o.dist = dist.x + dist.y;
  } else {
    o.d = (xhi - phi) - dPosSub;
    o.dist = length(o.d);
    o.f = (r1 - o.dist) + dRSub;
  }
  return o;
}

// Lower bound for the Newton derivative 1 + n·z'(R). The certificate guarantees ≥ 1 − betaMax;
// the legacy solver clamped at 5e-5, which stalls Newton above gamma ≈ 100.
float kappaFloor() { return u_rootMode == 0 && u_certMode == 0 ? 5e-5 : 0.5 * (1.0 - u_betaMax); }

// Retarded root for source s at observer x; returns m (shell coordinate) and sets inDomain.
void solveRoot(int s, vec2 xhi, vec2 xlo, out bool inDomain, out int segOut, out float fracOut) {
  inDomain = true;
  if (u_rootMode == 0) {
    float mLow = 0.0;
    float mHigh = float(u_mT) - 1.001;
    for (int iter = 0; iter < 16; iter++) {
      if (iter >= u_kBin) break;
      float mMid = 0.5 * (mLow + mHigh);
      Seg g = sampleSeg(s, mMid);
      RootEval r = rootFn(xhi, xlo, g.p1hi, g.p1lo, g.dPosSub, g.r1, g.dRSub);
      if (r.f < 0.0) mLow = mMid; else mHigh = mMid;
    }
    float mRet = 0.5 * (mLow + mHigh);
    for (int n = 0; n < 8; n++) {
      if (n >= u_newton) break;
      Seg g = sampleSeg(s, mRet);
      RootEval r = rootFn(xhi, xlo, g.p1hi, g.p1lo, g.dPosSub, g.r1, g.dRSub);
      vec2 nrm = r.dist > 1e-6 ? r.d / r.dist : vec2(1.0, 0.0);
      float dfdm = max(1e-7, g.dR * max(kappaFloor(), 1.0 + dot(nrm, g.dzdR)));
      mRet = clamp(mRet - r.f / dfdm, mLow, mHigh);
    }
    if (mRet > float(u_mT) - 1.5) inDomain = false;
    float mF = floor(mRet);
    segOut = int(mF); fracOut = mRet - mF;
    return;
  }

  // Knot-level bisection: f is monotone in R along the stored knots (subluminal chords).
  int lo = 0;
  int hi = u_mT - 1;
  Knot kl = knotAt(s, lo);
  Knot kh = knotAt(s, hi);
  float fLo = rootFn(xhi, xlo, kl.phi, kl.plo, vec2(0.0), kl.r, 0.0).f;
  float fHi = rootFn(xhi, xlo, kh.phi, kh.plo, vec2(0.0), kh.r, 0.0).f;
  if (fHi <= 0.0) { inDomain = false; segOut = u_mT - 2; fracOut = 0.9999; return; }
  if (fLo >= 0.0) { segOut = 0; fracOut = 0.0; return; }
  for (int i = 0; i < 12; i++) {
    if (hi - lo <= 1) break;
    int mid = (lo + hi) / 2;
    Knot km = knotAt(s, mid);
    float fm = rootFn(xhi, xlo, km.phi, km.plo, vec2(0.0), km.r, 0.0).f;
    if (fm < 0.0) { lo = mid; fLo = fm; } else { hi = mid; fHi = fm; }
  }
  // Safeguarded Newton inside [lo, lo+1], started from linear interpolation of f.
  float sLow = 0.0;
  float sHigh = 0.9999;
  float sc = clamp(fLo / (fLo - fHi), 0.0, 0.9999);
  // f is monotone on the segment (certificate) but can be strongly curved near the pulse crest at
  // high gamma, so iterate to convergence (u_newton = maximum steps) with a bisection fallback.
  for (int n = 0; n < 16; n++) {
    if (n >= u_newton) break;
    Seg g = sampleSegAt(s, lo, sc);
    RootEval r = rootFn(xhi, xlo, g.p1hi, g.p1lo, g.dPosSub, g.r1, g.dRSub);
    if (r.f < 0.0) sLow = sc; else sHigh = sc;
    vec2 nrm = r.dist > 1e-6 ? r.d / r.dist : vec2(1.0, 0.0);
    float dfds = max(1e-12, g.dR * max(kappaFloor(), 1.0 + dot(nrm, g.dzdR)));
    float cand = sc - r.f / dfds;
    float next = (cand > sLow && cand < sHigh) ? cand : 0.5 * (sLow + sHigh);
    bool done = abs(next - sc) < 1e-7 || (sHigh - sLow) < 1e-7;
    sc = next;
    if (done) break;
  }
  segOut = lo; fracOut = sc;
}

void evalCancellationFreeKappaAndU(vec2 n, vec2 bHat, float bMag, float omb2, out float kappa, out vec2 u) {
  float oneMinusBeta = omb2 / (1.0 + bMag);
  vec2 diff = n - bHat;
  kappa = max(1e-7, oneMinusBeta + 0.5 * bMag * dot(diff, diff));
  u = oneMinusBeta * n + bMag * diff;
}

struct FieldOut { vec2 e; vec2 eRad; vec2 eVel; float bz; float quality; vec2 emit0; vec4 dbg; };

FieldOut evalField(vec2 xhi, vec2 xlo) {
  FieldOut o;
  o.e = vec2(0.0); o.eRad = vec2(0.0); o.eVel = vec2(0.0); o.bz = 0.0; o.quality = 0.0; o.emit0 = vec2(0.0); o.dbg = vec4(0.0);
  for (int s = 0; s < 256; s++) {
    if (s >= u_activeCount) break;
    bool inDomain;
    int seg; float frac;
    solveRoot(s, xhi, xlo, inDomain, seg, frac);
    Seg g = sampleSegAt(s, seg, frac);
    float mRet = float(seg) + frac;
    RootEval r = rootFn(xhi, xlo, g.p1hi, g.p1lo, g.dPosSub, g.r1, g.dRSub);
    vec2 d = r.d;
    float r2 = dot(d, d);
    float rr = sqrt(r2);
    float rSoft2 = r2 + u_softEps * u_softEps;
    float rSoft = sqrt(rSoft2);
    vec2 n = rr > 1e-6 ? d / rr : vec2(1.0, 0.0);
    float kappa; vec2 u;
    evalCancellationFreeKappaAndU(n, g.bHat, g.bMag, g.omb2, kappa, u);
    float kappa3 = kappa * kappa * kappa;
    vec2 eVel = (g.chg * g.omb2 / (kappa3 * rSoft2)) * u;
    float crossZ = u.x * g.acc.y - u.y * g.acc.x;
    vec2 eRad = (g.chg / (kappa3 * rSoft)) * vec2(n.y * crossZ, -n.x * crossZ);
    vec2 eS = eVel + eRad;
    o.e += eS; o.eRad += eRad; o.eVel += eVel;
    o.bz += n.x * eS.y - n.y * eS.x;
    // Resolution indicator (heuristic, calibrated against the reference in results/):
    //  - bending: shell spacing x gamma over the local radius of curvature;
    //  - speeding up/slowing down: relative change of proper velocity across the segment.
    float b2 = max(1e-12, g.bMag * g.bMag);
    float aPerp = abs(g.bHat.x * g.acc.y - g.bHat.y * g.acc.x);
    float gam = inversesqrt(max(1e-12, g.omb2));
    float qBend = g.dR * gam * aPerp / b2;
    float qSpeed = 6.0 * g.duRel;
    float q = max(qBend, qSpeed);
    if (mRet < 1.0) q = max(q, 1.6);   // inside the first shell: too close to the charge to trust
    // Shells the Nyquist taper has damped no longer carry the true acceleration: flag them.
    if (u_taperOn == 1 && u_scaleL > 0.0) {
      float lambda = 2.0 * PI * u_c / max(0.5, u_omega);
      if (g.dR / (0.5 * lambda) > 0.25) q = max(q, 2.0);
    }
    // Beyond the ring buffer the shells repeat its oldest sample: not a valid history.
    if (u_ringReach > 0.0 && g.r1 + g.dR > u_ringReach) inDomain = false;
    if (!inDomain) o.quality = -1.0;
    else if (o.quality >= 0.0) o.quality = max(o.quality, q);
    if (s == 0) { o.emit0 = (g.p1hi + g.p1lo) + g.dPosSub; o.dbg = vec4(g.r1, g.dRSub, kappa, g.bMag); }
  }
  return o;
}

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Shared magnitude compression (the app's colorbar uses the same formula).
float compressMag(float m) {
  float lm = log(1.0 + 1.85 * m);
  return lm / (1.35 + lm);
}

vec3 diverging(float v) {
  // v in [-1, 1]: blue - near-black - red
  vec3 neg = vec3(0.23, 0.55, 0.98);
  vec3 pos = vec3(0.98, 0.42, 0.30);
  vec3 mid = vec3(0.04, 0.06, 0.10);
  return v < 0.0 ? mix(mid, neg, -v) : mix(mid, pos, v);
}

vec3 evalImpostors(vec2 wPos) {
  vec3 impCol = vec3(0.0);
  float shellSpace = 0.52;
  float phaseOff = fract((u_time * u_c) / shellSpace);
  vec3 lightDir = normalize(vec3(-0.35, 0.45, 0.82));
  for (int k = 0; k < 5; k++) {
    float rShell = (float(k) + phaseOff) * shellSpace;
    if (rShell < 0.06 || rShell > 2.65) continue;
    float th = 2.0 * atan(rShell / u_scaleL);
    float mFl = (th / PI) * float(u_mT) - 0.5;
    Seg g = sampleSeg(0, mFl);
    vec2 d = (wPos - (g.p1hi + g.p1lo)) - g.dPosSub;
    float d2 = dot(d, d);
    float r2 = rShell * rShell;
    if (d2 < r2) {
      float z = sqrt(r2 - d2);
      vec3 normal = vec3(d.x, d.y, z) / rShell;
      float nv = clamp(normal.z, 0.0, 1.0);
      float fresnel = pow(1.0 - nv, 4.0);
      float rim = exp(-abs(sqrt(d2) - rShell) * 48.0);
      vec2 n2 = d / max(1e-5, sqrt(d2));
      float kappa; vec2 uTmp;
      evalCancellationFreeKappaAndU(n2, g.bHat, g.bMag, g.omb2, kappa, uTmp);
      float doppler = 1.0 / max(0.18, kappa);
      float spec = pow(max(0.0, dot(reflect(-lightDir, normal), vec3(0.0, 0.0, 1.0))), 28.0);
      float fade = clamp(1.0 - rShell / 2.65, 0.0, 1.0);
      vec3 tint = mix(vec3(0.15, 0.68, 0.98), vec3(0.98, 0.75, 0.25), clamp((doppler - 1.0) * 0.35, 0.0, 1.0));
      impCol += fade * (tint * (0.28 * fresnel + 0.65 * rim) + vec3(0.85, 0.95, 1.0) * spec * 0.45);
    }
  }
  return impCol;
}

void main() {
  vec2 xhi;
  vec2 xlo = vec2(0.0);
  if (u_readbackMode == 1) {
    vec4 q = texelFetch(u_texQueryCoords, ivec2(int(gl_FragCoord.x), int(gl_FragCoord.y)), 0);
    xhi = q.xy;
    if (u_precision == 1) xlo = q.zw;
  } else {
    vec2 pNorm = vec2((v_uv.x - 0.5) * 2.0 * u_aspect, (v_uv.y - 0.5) * 2.0);
    xhi = u_viewCenter + pNorm * u_viewHalfHeight;
  }

  FieldOut F = evalField(xhi, xlo);

  if (u_readbackMode == 1) {
    if (u_readbackLayout == 2) outColor = F.dbg;
    else if (u_readbackLayout == 1) outColor = vec4(F.e, F.bz, F.quality);
    else outColor = vec4(F.e, F.emit0);
    return;
  }

  vec3 col = vec3(0.027, 0.039, 0.071);
  if (u_showWavePool == 1) {
    float multiScaleGain = u_gain / pow(float(max(1, u_activeCount)), 0.42);
    if (u_channel == 3) {
      float v = F.bz * multiScaleGain;
      float c = compressMag(abs(v));
      col = diverging(sign(v) * clamp(pow(c, 0.75) * 1.1, 0.0, 1.0));
    } else if (u_channel == 4) {
      float sMag = length(F.e) * abs(F.bz) * multiScaleGain * multiScaleGain;
      float c = compressMag(sMag);
      col = mix(vec3(0.02, 0.03, 0.06), vec3(1.0, 0.86, 0.55), clamp(pow(c, 0.8) * 1.05, 0.0, 1.0));
    } else {
      vec2 displayVec = u_channel == 1 ? F.eRad : (u_channel == 2 ? F.eVel : F.e);
      float compressed = compressMag(length(displayVec) * multiScaleGain);
      float val = clamp(0.14 + 0.82 * pow(compressed, 0.75), 0.0, 0.96);
      float hue = fract((atan(displayVec.y, displayVec.x) + PI) / (2.0 * PI));
      float sat = mix(0.78, 0.42, pow(compressed, 3.5));
      vec3 fieldRgb = hsv2rgb(vec3(hue, sat, val));
      vec3 poolBase = vec3(0.03, 0.11, 0.22) * (0.4 + 0.6 * compressed);
      float caustic = pow(clamp(compressed, 0.0, 1.0), 3.2) * 0.24;
      col = clamp(mix(poolBase, fieldRgb, clamp(compressed * 1.15 + 0.18, 0.0, 1.0)) + vec3(caustic * 0.6, caustic * 0.85, caustic), 0.0, 1.0);
    }
  }
  if (u_showImpostors == 1) col = clamp(col + evalImpostors(xhi) * 0.60, 0.0, 1.0);
  if (u_showMask == 1 && (F.quality < 0.0 || F.quality > u_maskThreshold)) {
    // Hatch pixels where the answer is outside the stored history or under-resolved.
    float stripe = step(0.5, fract((gl_FragCoord.x + gl_FragCoord.y) / 10.0));
    col = mix(col, F.quality < 0.0 ? vec3(0.45, 0.05, 0.08) : vec3(0.35, 0.30, 0.05), 0.55 * stripe + 0.15);
  }
  outColor = vec4(col, 1.0);
}`;
