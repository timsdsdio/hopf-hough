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
 *
 * Compile-time variants (prepended #defines; see solver.ts):
 *  - KNOWN_PATH: C2 quintic segments + degree-8 certificate on knots flagged exact.
 *  - F32N (kernel v6f): only f and d = x − z at the bracketing knot are double-float. Inside the segment the
 *    root function is a float32 sum of non-negative terms, with the chord deficit 1 − |v_chord|² supplied per
 *    knot by the CPU in float64 (u_texPosLo.z); the knot search is a float32 bisection with a double-float check
 *    of both bracket knots; at high γ (|u| > 4) the cancelling cross/dot products of the field are taken against
 *    the double-float separation with exact products. F32N_NOFIX removes that last part. F32N always uses the
 *    knot search + Newton (u_rootMode is not read), so the switch study above is meant for the v6 kernel.
 *    Without F32N the program is the v6 kernel, unchanged.
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
uniform sampler2D u_texPosLo;       // x_lo, y_lo, chord deficit 1 - |v_chord|^2 of segment m -> m+1 (v6f only), -
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
uniform int u_fastRender;
#ifdef KNOWN_PATH
uniform sampler2D u_texJerk;        // known-path knots: jx, jy (d acc / dR), exact flag (1 = exact sample of a known path), -
uniform int u_knownPath;            // 1: segments whose two knots are exact use the C2 quintic (degree-8 certificate)
#endif           // 1: cheaper root solve for on-screen rendering (never used for readback)

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
#ifndef F32N
struct Knot { vec2 phi; vec2 plo; vec2 u; vec2 acc; float chg; float r; };
#else
// v6f: dch = chord deficit 1 - |v_chord|^2 of segment m -> m+1, formed on the CPU in float64 (u_texPosLo.z)
struct Knot { vec2 phi; vec2 plo; vec2 u; vec2 acc; float chg; float r; float dch; };
#endif

Knot knotAt(int s, int m) {
  vec4 pv = texelFetch(u_texPosVel, ivec2(m, s), 0);
  vec4 ac = texelFetch(u_texAccChg, ivec2(m, s), 0);
  vec4 lo = texelFetch(u_texPosLo, ivec2(m, s), 0);
  Knot k;
  k.phi = pv.xy; k.u = pv.zw; k.acc = ac.xy; k.chg = ac.z; k.r = ac.w;
  k.plo = (u_precision == 1) ? lo.xy : vec2(0.0);
#ifdef F32N
  k.dch = lo.z;
#endif
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

#ifdef KNOWN_PATH
// Known-path segments: C2 quintic Hermite (z, z' = -beta, z'' = stored acceleration at both knots).
// Its hodograph is quartic, so |z'|^2 is degree 8 in Bernstein form: c_k = sum_{i+j=k} w_ij Q_i.Q_j.
// With Q(alpha) = vSec + alpha (Q - vSec), each c_k is quadratic in alpha; take the first alpha in (0, 1]
// where some c_k reaches betaMax^2 (c_0 and c_8 are the endpoint speeds, always inside). A limited
// segment (alpha < 1) no longer matches the endpoint derivatives: it is C0 there, as the cubic is.
float alphaQuint(vec2 v, vec2 D0, vec2 D1, vec2 D2, vec2 D3, vec2 D4, float bm) {
  float pp = dot(v, v) - bm * bm;
  if (pp >= 0.0) return 0.0;
  float e0 = dot(v, D0), e1 = dot(v, D1), e2 = dot(v, D2), e3 = dot(v, D3), e4 = dot(v, D4);
  float al = 1.0;
  al = firstRoot(dot(D0, D1), e0 + e1, pp, al);
  al = firstRoot((12.0 * dot(D0, D2) + 16.0 * dot(D1, D1)) / 28.0, (12.0 * (e0 + e2) + 32.0 * e1) / 28.0, pp, al);
  al = firstRoot((8.0 * dot(D0, D3) + 48.0 * dot(D1, D2)) / 56.0, (8.0 * (e0 + e3) + 48.0 * (e1 + e2)) / 56.0, pp, al);
  al = firstRoot((2.0 * dot(D0, D4) + 32.0 * dot(D1, D3) + 36.0 * dot(D2, D2)) / 70.0, (2.0 * (e0 + e4) + 32.0 * (e1 + e3) + 72.0 * e2) / 70.0, pp, al);
  al = firstRoot((8.0 * dot(D1, D4) + 48.0 * dot(D2, D3)) / 56.0, (8.0 * (e1 + e4) + 48.0 * (e2 + e3)) / 56.0, pp, al);
  al = firstRoot((12.0 * dot(D2, D4) + 16.0 * dot(D3, D3)) / 28.0, (12.0 * (e2 + e4) + 32.0 * e3) / 28.0, pp, al);
  al = firstRoot(dot(D3, D4), e3 + e4, pp, al);
  return al;
}

#endif

struct Seg {
  vec2 p1hi; vec2 p1lo;   // knot m1 position
  vec2 dPosSub;           // z(s) - z(m1)
  float r1; float dRSub; float dR;
  vec2 dzdR;              // dz/dR at s
  vec2 bHat; float bMag; float omb2;
  vec2 acc; float chg;
  float duRel;            // |u2 - u1| / (mean |u| + 0.1) across the segment
#ifdef F32N
  float sR;               // fraction used for the position (0 on a segment without a next knot)
#endif
};

// Segment m1 at fraction t in [0, 1). Index and fraction are kept separate so that float32
// resolution of a combined coordinate (≈ 6e-5 near m = 1000) never limits the retarded point.
// Everything about a segment that does not depend on the position inside it: fetched and
// certified once per (pixel, source), then evaluated at each Newton iterate by evalSeg().
struct SegC {
  vec2 p1hi; vec2 p1lo; float r1; float dR; bool hasNext;
  vec2 dChord; vec2 v0; vec2 v1; vec2 v2;
#ifdef KNOWN_PATH
  vec2 v3; vec2 v4; vec2 j1; vec2 j2;   // known-path quintic: hodograph control points 3-4, knot jerks
  bool exact; bool quint; bool uHerm;  // both knots exact; quintic in use; Hermite |u| in use
#endif
  vec2 acc0; vec2 acc1; vec2 acc2; vec2 acc3;
  vec2 u0; vec2 u1; vec2 u2; vec2 u3;
  float chg;
#ifdef F32N
  // Chord deficit 1 - |vbar(s)|^2 of the mean velocity over [0, s], as an affine combination of points P_i
  // (sum a_i = 1): 1 - |sum a_i P_i|^2 = sum a_i D_i + sum_{i<j} a_i a_j |P_i - P_j|^2, D_i = 1 - |P_i|^2.
  // Cubic: P = (v_chord, -b1, -b2). Quintic: P = (v_chord, Q0, Q1, Q3, Q4) (Q2 eliminated by sum Q = 5 v_chord).
  float alpha; float Dv; float D0; float D2;
  vec2 p2hi; vec2 p2lo;   // knot m2 position, for the double-float chord of the transverse-angle fix (FixC)
  vec4 qa;   // cubic: |Pv-P0|^2, |Pv-P2|^2, |P0-P2|^2, -     quintic: |Pv-Q0|^2, |Pv-Q1|^2, |Pv-Q3|^2, |Pv-Q4|^2
#ifdef KNOWN_PATH
  float D1; float D3;
  vec4 qb;   // quintic: |Q0-Q1|^2, |Q0-Q3|^2, |Q0-Q4|^2, |Q1-Q3|^2
  vec2 qc;   // quintic: |Q1-Q4|^2, |Q3-Q4|^2
#endif
#endif
};

SegC segCoef(int s, int m1In) {
  int m1 = clamp(m1In, 0, u_mT - 2);
  int m0 = max(0, m1 - 1);
  int m2 = min(u_mT - 1, m1 + 1);
  int m3 = min(u_mT - 1, m1 + 2);
  Knot k0 = knotAt(s, m0);
  Knot k1 = knotAt(s, m1);
  Knot k2 = knotAt(s, m2);
  Knot k3 = knotAt(s, m3);

  SegC c;
  c.r1 = k1.r;
  c.hasNext = m2 > m1;
  float dR = m2 > m1 ? max(1e-6, k2.r - k1.r) : 1e-6;
  c.dR = dR;
  c.p1hi = k1.phi;
  c.p1lo = k1.plo;
  c.acc0 = k0.acc; c.acc1 = k1.acc; c.acc2 = k2.acc; c.acc3 = k3.acc;
  c.u0 = k0.u; c.u1 = k1.u; c.u2 = k2.u; c.u3 = k3.u;
  c.chg = k1.chg;

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
  c.dChord = dChord;
  c.v0 = oma * vSec - alpha * b1;
  c.v1 = vSec + alpha * (v1Raw - vSec);
  c.v2 = oma * vSec - alpha * b2;
#ifdef F32N
  c.alpha = alpha; c.Dv = k1.dch;
  c.p2hi = k2.phi; c.p2lo = k2.plo;
  float bm1 = min(b1Mag, maxSub), bm2 = min(b2Mag, maxSub);
  c.D0 = (u_velMode == 2 && b1Mag <= maxSub) ? 1.0 / (1.0 + dot(k1.u, k1.u)) : (1.0 - bm1) * (1.0 + bm1);
  c.D2 = (u_velMode == 2 && b2Mag <= maxSub) ? 1.0 / (1.0 + dot(k2.u, k2.u)) : (1.0 - bm2) * (1.0 + bm2);
  vec2 e0 = vSec + b1, e2 = vSec + b2, e02 = b2 - b1;
  c.qa = vec4(dot(e0, e0), dot(e2, e2), dot(e02, e02), 0.0);
#ifdef KNOWN_PATH
  c.D1 = 0.0; c.D3 = 0.0; c.qb = vec4(0.0); c.qc = vec2(0.0);
#endif
#endif
#ifdef KNOWN_PATH
  c.v3 = vec2(0.0); c.v4 = vec2(0.0); c.j1 = vec2(0.0); c.j2 = vec2(0.0);
  c.exact = false; c.quint = false; c.uHerm = false;
  if (u_knownPath == 1) {
    vec4 jk1 = texelFetch(u_texJerk, ivec2(m1, s), 0);
    vec4 jk2 = texelFetch(u_texJerk, ivec2(m2, s), 0);
    c.exact = jk1.z > 0.5 && jk2.z > 0.5;
    if (c.exact && c.hasNext) {
      c.quint = true;
      vec2 Q0 = -b1, Q4 = -b2;
      vec2 Q1 = Q0 + (0.25 * dR) * k1.acc;
      vec2 Q3 = Q4 - (0.25 * dR) * k2.acc;
      vec2 Q2 = 5.0 * vSec - Q0 - Q1 - Q3 - Q4;
      float aq = alphaQuint(vSec, Q0 - vSec, Q1 - vSec, Q2 - vSec, Q3 - vSec, Q4 - vSec, maxSub);
      c.v0 = vSec + aq * (Q0 - vSec); c.v1 = vSec + aq * (Q1 - vSec); c.v2 = vSec + aq * (Q2 - vSec);
      c.v3 = vSec + aq * (Q3 - vSec); c.v4 = vSec + aq * (Q4 - vSec);
      c.j1 = jk1.xy; c.j2 = jk2.xy;
#ifdef F32N
      // Q1 = Q0 + w1, Q3 = Q4 - w2: their deficits from the knots' exact ones plus small terms
      vec2 w1 = (0.25 * dR) * k1.acc, w2 = (0.25 * dR) * k2.acc;
      c.alpha = aq;
      c.D1 = c.D0 + 2.0 * dot(b1, w1) - dot(w1, w1);
      c.D3 = c.D2 - 2.0 * dot(b2, w2) - dot(w2, w2);
      vec2 g1 = e0 - w1, g3 = e2 + w2, h03 = e02 + w2, h13 = h03 + w1, h14 = e02 + w1;
      c.qa = vec4(dot(e0, e0), dot(g1, g1), dot(g3, g3), dot(e2, e2));
      c.qb = vec4(dot(w1, w1), dot(h03, h03), dot(e02, e02), dot(h13, h13));
      c.qc = vec2(dot(h14, h14), dot(w2, w2));
#endif
      // Hermite |u| only where the segment is resolved (bend term of the validity indicator < 1) and
      // unlimited: on coarse segments it overshoots (e.g. 0.44 -> 1.1 at beta 0.99, M = 128), so fall
      // back to linear |u| there.
      float n1 = length(k1.u), n2 = length(k2.u);
      float um = 0.5 * (n1 + n2), bm = um / sqrt(1.0 + um * um);
      float ap = max(abs(b1.x * k1.acc.y - b1.y * k1.acc.x) / max(length(b1), 1e-7),
                     abs(b2.x * k2.acc.y - b2.y * k2.acc.x) / max(length(b2), 1e-7));
      c.uHerm = aq >= 1.0 && dR * sqrt(1.0 + um * um) * ap / max(0.04, bm * bm) < 1.0;
    }
  }
#endif
  return c;
}

Seg evalSeg(SegC c, float tIn) {
  float t = clamp(tIn, 0.0, 0.99999);
  float t2 = t * t;
  float t3 = t2 * t;
  float dR = c.dR;
  float sRad = c.hasNext ? t : 0.0;
  vec2 dChord = c.dChord, v0 = c.v0, v1 = c.v1, v2 = c.v2;
  Seg g;
  g.r1 = c.r1;
  g.dR = dR;
  g.dRSub = sRad * dR;
#ifdef F32N
  g.sR = sRad;
#endif
  g.p1hi = c.p1hi;
  g.p1lo = c.p1lo;

  float s2 = sRad * sRad;
  float s3 = s2 * sRad;
  float oms = 1.0 - sRad;
#ifdef KNOWN_PATH
  if (!c.quint) {
#endif
  float h10 = s3 - 2.0 * s2 + sRad;
  float h01 = -2.0 * s3 + 3.0 * s2;
  float h11 = s3 - s2;
  g.dPosSub = h01 * dChord + h10 * (v0 * dR) + h11 * (v2 * dR);
  g.dzdR = (oms * oms) * v0 + (2.0 * sRad * oms) * v1 + s2 * v2;
#ifdef KNOWN_PATH
  } else {
    // Bezier position control points relative to z1: P_k = dR/5 sum_{j<k} Q_j; P_5 = chord (exact).
    float o2 = oms * oms, o3 = o2 * oms, o4 = o3 * oms, t4 = s2 * s2;
    vec2 P1 = (0.2 * dR) * c.v0;
    vec2 P2 = P1 + (0.2 * dR) * c.v1;
    vec2 P3 = P2 + (0.2 * dR) * c.v2;
    vec2 P4 = dChord - (0.2 * dR) * c.v4;
    g.dPosSub = (5.0 * o4 * sRad) * P1 + (10.0 * o3 * s2) * P2 + (10.0 * o2 * s3) * P3 + (5.0 * oms * t4) * P4 + (t4 * sRad) * dChord;
    g.dzdR = o4 * c.v0 + (4.0 * o3 * sRad) * c.v1 + (6.0 * o2 * s2) * c.v2 + (4.0 * oms * s3) * c.v3 + t4 * c.v4;
  }
#endif
  vec2 splineVel = -g.dzdR;
  float splineVelMag = length(splineVel);

  float w0 = -0.5 * t3 + t2 - 0.5 * t;
  float w1 = 1.5 * t3 - 2.5 * t2 + 1.0;
  float w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
  float w3 = 0.5 * t3 - 0.5 * t2;
#ifndef KNOWN_PATH
  g.acc = c.acc0 * w0 + c.acc1 * w1 + c.acc2 * w2 + c.acc3 * w3;
#else
  float uHermVal = -1.0;
  if (!c.quint) g.acc = c.acc0 * w0 + c.acc1 * w1 + c.acc2 * w2 + c.acc3 * w3;
  else {
    float H00 = 2.0 * t3 - 3.0 * t2 + 1.0, H10 = t3 - 2.0 * t2 + t, H01 = -2.0 * t3 + 3.0 * t2, H11 = t3 - t2;
    g.acc = H00 * c.acc1 + H10 * (dR * c.j1) + H01 * c.acc2 + H11 * (dR * c.j2);
    if (c.uHerm) {
      // d|u|/dR = -gamma^3 (u_hat . acc)
      float n1 = length(c.u1), n2 = length(c.u2);
      float q1 = 1.0 + n1 * n1, q2 = 1.0 + n2 * n2;
      float d1 = -(q1 * sqrt(q1)) * dot(c.u1, c.acc1) / max(n1, 1e-20);
      float d2 = -(q2 * sqrt(q2)) * dot(c.u2, c.acc2) / max(n2, 1e-20);
      uHermVal = max(0.0, H00 * n1 + H10 * (dR * d1) + H01 * n2 + H11 * (dR * d2));
    }
  }
#endif
  g.chg = c.chg;
  g.duRel = length(c.u2 - c.u1) / (0.5 * (length(c.u1) + length(c.u2)) + 0.1);

  if (u_velMode == 2) {
#ifdef KNOWN_PATH
    float uTarget = uHermVal >= 0.0 ? uHermVal : mix(length(c.u1), length(c.u2), t);
#else
    float uTarget = mix(length(c.u1), length(c.u2), t);
#endif
    g.bHat = splineVelMag > 1e-7 ? splineVel / splineVelMag : vec2(1.0, 0.0);
    float uSq = uTarget * uTarget;
    g.bMag = uTarget / sqrt(1.0 + uSq);
    g.omb2 = 1.0 / (1.0 + uSq);
  } else if (u_velMode == 1) {
    float bTarget = mix(length(c.u1), length(c.u2), t);
    g.bHat = splineVelMag > 1e-7 ? splineVel / splineVelMag : vec2(1.0, 0.0);
    g.bMag = min(0.9999995, bTarget);
    g.omb2 = (1.0 - g.bMag) * (1.0 + g.bMag);
  } else {
    vec2 rawVel = c.u0 * w0 + c.u1 * w1 + c.u2 * w2 + c.u3 * w3;
    g.bMag = min(0.9999995, length(rawVel));
    g.bHat = g.bMag > 1e-7 ? rawVel / g.bMag : vec2(1.0, 0.0);
    g.omb2 = (1.0 - g.bMag) * (1.0 + g.bMag);
  }
  return g;
}

Seg sampleSegAt(int s, int m1In, float tIn) { return evalSeg(segCoef(s, m1In), tIn); }

Seg sampleSeg(int s, float mFloat) {
  float mClamped = clamp(mFloat, 0.0, float(u_mT) - 1.0001);
  float mFloor = floor(mClamped);
  return sampleSegAt(s, int(mFloor), mClamped - mFloor);
}

// Root function f = R - |x - z| and the separation vector d = x - z (both single precision on return).
#ifndef F32N
struct RootEval { float f; vec2 d; float dist; };
#else
struct RootEval { float f; vec2 d; float dist; vec2 dlo; };   // v6f: dlo = low part of the double-float d (knots only)
#endif

// g_fast: on-screen rendering (u_fastRender = 1). Readback (verification suite, probe, tables) always
// takes the full path, so every reported number is unaffected by the display shortcuts.
bool g_fast = false;

RootEval rootFn(vec2 xhi, vec2 xlo, vec2 phi, vec2 plo, vec2 dPosSub, float r1, float dRSub) {
  RootEval o;
  if (u_precision == 1 && !g_fast) {
    vec2 DX = dfAdd(dfAdd(dfAdd(twoSum(xhi.x, -phi.x), vec2(xlo.x, 0.0)), vec2(-plo.x, 0.0)), vec2(-dPosSub.x, 0.0));
    vec2 DY = dfAdd(dfAdd(dfAdd(twoSum(xhi.y, -phi.y), vec2(xlo.y, 0.0)), vec2(-plo.y, 0.0)), vec2(-dPosSub.y, 0.0));
    vec2 dist = dfSqrt(dfAdd(dfMul(DX, DX), dfMul(DY, DY)));
    vec2 F = dfAdd(twoSum(r1, dRSub), -dist);
    o.f = F.x + F.y;
    o.d = vec2(DX.x + DX.y, DY.x + DY.y);
    o.dist = dist.x + dist.y;
#ifdef F32N
    o.dlo = vec2(ADD(SUB(DX.x, o.d.x), DX.y), ADD(SUB(DY.x, o.d.y), DY.y));
#endif
  } else if (u_precision == 1) {
    // Display path: same double-float cancellation, but no double-float square root.
    // d = x - z: the large parts cancel exactly in twoSum; the small parts (observer and knot low words,
    // offset inside the segment) are added as a second double-float. Then f = (R² - |d|²) / (R + |d|):
    // the numerator in double-float, the division in float32 (it only sets f's relative error).
    vec2 DX = dfAdd(twoSum(xhi.x, -phi.x), twoSum(SUB(xlo.x, plo.x), -dPosSub.x));
    vec2 DY = dfAdd(twoSum(xhi.y, -phi.y), twoSum(SUB(xlo.y, plo.y), -dPosSub.y));
    vec2 R = twoSum(r1, dRSub);
    vec2 num = dfAdd(dfMul(R, R), -dfAdd(dfMul(DX, DX), dfMul(DY, DY)));
    o.d = vec2(DX.x + DX.y, DY.x + DY.y);
    o.dist = length(o.d);
    o.f = (num.x + num.y) / max(1e-30, (R.x + R.y) + o.dist);
#ifdef F32N
    o.dlo = vec2(ADD(SUB(DX.x, o.d.x), DX.y), ADD(SUB(DY.x, o.d.y), DY.y));
#endif
  } else {
    o.d = (xhi - phi) - dPosSub;
    o.dist = length(o.d);
    o.f = (r1 - o.dist) + dRSub;
#ifdef F32N
    o.dlo = vec2(0.0);
#endif
  }
  return o;
}

// Lower bound for the Newton derivative 1 + n·z'(R). The certificate guarantees ≥ 1 − betaMax;
// the legacy solver clamped at 5e-5, which stalls Newton above gamma ≈ 100.
float kappaFloor() { return u_rootMode == 0 && u_certMode == 0 ? 5e-5 : 0.5 * (1.0 - u_betaMax); }

// Retarded root for source s at observer x; returns m (shell coordinate) and sets inDomain.
// Root function at a stored knot in float32, used only for its sign during the knot bisection.
// It can be wrong only within a few rounding errors of zero, i.e. next to the root; the bracket
// is then checked once in double-float (below) and moved by one segment if needed.
float knotF(vec2 xhi, vec2 xlo, Knot k) {
  if (u_precision != 1 || !g_fast) return rootFn(xhi, xlo, k.phi, k.plo, vec2(0.0), k.r, 0.0).f;
  vec2 d = (xhi - k.phi) + (xlo - k.plo);
  return k.r - length(d);
}

void solveRoot(int s, vec2 xhi, vec2 xlo, out bool inDomain, out int segOut, out float fracOut, out SegC cOut) {
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
    cOut = segCoef(s, segOut);
    return;
  }

  // Knot-level bisection: f is monotone in R along the stored knots (subluminal chords).
  int lo = 0;
  int hi = u_mT - 1;
  Knot kl = knotAt(s, lo);
  Knot kh = knotAt(s, hi);
  float fLo = knotF(xhi, xlo, kl);
  float fHi = knotF(xhi, xlo, kh);
  if (fHi <= 0.0) { inDomain = false; segOut = u_mT - 2; fracOut = 0.9999; cOut = segCoef(s, segOut); return; }
  if (fLo >= 0.0) { segOut = 0; fracOut = 0.0; cOut = segCoef(s, 0); return; }
  for (int i = 0; i < 12; i++) {
    if (hi - lo <= 1) break;
    int mid = (lo + hi) / 2;
    Knot km = knotAt(s, mid);
    float fm = knotF(xhi, xlo, km);
    if (fm < 0.0) { lo = mid; fLo = fm; } else { hi = mid; fHi = fm; }
  }
  // Check the end of the bracket nearer zero in double-float (the only one whose float32 sign can be
  // wrong), move the bracket by one segment if it was, and refresh that end's value. (Display path only:
  // the full path bisects in double-float throughout.)
  if (u_precision == 1 && g_fast) {
    bool checkLo = abs(fLo) < abs(fHi);
    Knot kc = knotAt(s, checkLo ? lo : lo + 1);
    float fc = rootFn(xhi, xlo, kc.phi, kc.plo, vec2(0.0), kc.r, 0.0).f;
    if (checkLo && fc >= 0.0 && lo > 0) {
      fHi = fc; lo -= 1; Knot kn = knotAt(s, lo);
      fLo = rootFn(xhi, xlo, kn.phi, kn.plo, vec2(0.0), kn.r, 0.0).f;
    } else if (!checkLo && fc < 0.0 && lo + 2 <= u_mT - 1) {
      fLo = fc; lo += 1; Knot kn = knotAt(s, lo + 1);
      fHi = rootFn(xhi, xlo, kn.phi, kn.plo, vec2(0.0), kn.r, 0.0).f;
    } else if (checkLo) fLo = fc; else fHi = fc;
  }
  SegC c = segCoef(s, lo);
  // Safeguarded Newton inside [lo, lo+1], started from linear interpolation of f.
  float sLow = 0.0;
  float sHigh = 0.9999;
  float sc = clamp(fLo / (fLo - fHi), 0.0, 0.9999);
  // f is monotone on the segment (certificate) but can be strongly curved near the pulse crest at
  // high gamma, so iterate to convergence (u_newton = maximum steps) with a bisection fallback.
  for (int n = 0; n < 16; n++) {
    if (n >= u_newton) break;
    Seg g = evalSeg(c, sc);
    RootEval r = rootFn(xhi, xlo, g.p1hi, g.p1lo, g.dPosSub, g.r1, g.dRSub);
    if (r.f == 0.0) break;                       // on the root: hold (a strict bracket would bisect away)
    if (r.f < 0.0) sLow = sc; else sHigh = sc;
    vec2 nrm = r.dist > 1e-6 ? r.d / r.dist : vec2(1.0, 0.0);
    float dfds = max(1e-12, g.dR * max(kappaFloor(), 1.0 + dot(nrm, g.dzdR)));
    float cand = sc - r.f / dfds;
    float next = (cand >= sLow && cand <= sHigh) ? cand : 0.5 * (sLow + sHigh);   // closed bracket
    bool done = abs(next - sc) < 1e-7 || (sHigh - sLow) < 1e-7;
    sc = next;
    if (done) break;
  }
  segOut = lo; fracOut = sc; cOut = c;
}

#ifdef F32N
// ---------------- v6f: float32 incremental root inside the segment ----------------
// Only f1 = R1 - |x - z1| and d1 = x - z1 at the bracketing knot are double-float. Inside the segment,
// with dz = z(s) - z1, l = |dz|, dR = s dR_seg, L1 = |d1|, L = |d1 - dz|:
//   f(s) = f1 + [dR - l] + [l + L1 - L],  both brackets >= 0 and formed without cancellation:
//   dR - l = dR D / (1 + l/dR), D = 1 - |dz/dR|^2 from the affine form (segCoef), not from float32 dz;
//   l + L1 - L = 2 (d1 x dz)^2 / ((l L1 - d1.dz)(l + L1 + L)) if d1.dz <= 0, else 2 (l L1 + d1.dz) / (l + L1 + L).
// Plain float32 sums of non-negative terms: no BAR() needed (reassociation cannot cancel them).
float chordDeficit(SegC c, float s) {
  float o = 1.0 - s;
#ifdef KNOWN_PATH
  if (c.quint) {
    // mean of the quartic hodograph over [0, s]: I_k = (1/s) int_0^s B_{k,4}; coefficients a_k = aq (I_k - I_2)
    float o2 = o * o, o4 = o2 * o2, s2 = s * s;
    float I0 = (1.0 + o + o2 + o2 * o + o4) * 0.2, I1 = I0 - o4;
    float I2 = s2 * (10.0 * o2 + 5.0 * s * o + s2) * 0.2, I3 = s2 * s * (5.0 * o + s) * 0.2, I4 = s2 * s2 * 0.2;
    float a0 = c.alpha * (I0 - I2), a1 = c.alpha * (I1 - I2), a3 = c.alpha * (I3 - I2), a4 = c.alpha * (I4 - I2);
    float av = 1.0 - a0 - a1 - a3 - a4;
    return av * c.Dv + a0 * c.D0 + a1 * c.D1 + a3 * c.D3 + a4 * c.D2
         + av * (a0 * c.qa.x + a1 * c.qa.y + a3 * c.qa.z + a4 * c.qa.w)
         + a0 * (a1 * c.qb.x + a3 * c.qb.y + a4 * c.qb.z) + a1 * (a3 * c.qb.w + a4 * c.qc.x) + a3 * a4 * c.qc.y;
  }
#endif
  // mean of the quadratic hodograph over [0, s]: a0 = alpha (1-s)^2, a2 = -alpha s (1-s)
  float a0 = c.alpha * o * o, a2 = -c.alpha * s * o, av = 1.0 - a0 - a2;
  return av * c.Dv + a0 * c.D0 + a2 * c.D2 + av * a0 * c.qa.x + av * a2 * c.qa.y + a0 * a2 * c.qa.z;
}
// Transverse angles. Near the pulse crest n, beta_hat and the chord are parallel to within 1/gamma, so any float32
// rotation of one of them (rounding of d, of n = d/|d|, of the control points, of the hodograph and its normalisation)
// is a relative error ~ eps gamma in 1 - n.beta_hat, d1 x dz (root) and n.a. Instead, every such cross/dot product is
// taken against the double-float knot separation d1 (exact products, BAR-guarded), once per segment: the spline is
// linear in (chord, b1, b2, acc1, acc2), so psi_k = d1 x v_k (hodograph control points) and chi_k = d1 x P_k
// (position control points) are scalars, and d1 x z'(s), d1 x dz(s) are their Bernstein sums (no rounding of a direction).
struct FixC { vec4 psi; float psi4; vec4 chi; float chi5; vec4 Y; bool on; };
float crossX(vec2 dh, vec2 dl, vec2 w) {   // (dh + dl) x w, w float32: products exact, then summed
  vec2 p = twoProd(dh.x, w.y), q = twoProd(dh.y, w.x);
  return ADD(SUB(p.x, q.x), ADD(SUB(p.y, q.y), SUB(MUL(dl.x, w.y), MUL(dl.y, w.x))));
}
float dotX(vec2 dh, vec2 dl, vec2 w) {
  vec2 p = twoProd(dh.x, w.x), q = twoProd(dh.y, w.y);
  return ADD(ADD(p.x, q.x), ADD(ADD(p.y, q.y), ADD(MUL(dl.x, w.x), MUL(dl.y, w.y))));
}
// Only where it matters: the error is ~ eps gamma, so the fix is skipped (one branch per pixel-source, uniform across
// pixels) unless the segment has |u| > 4 (gamma > 4.1, proper-velocity storage).
FixC fixConsts(SegC c, vec2 dh, vec2 dl) {
  // F32N_NOFIX (compile-time) removes the fix: same output and frame time as v6f before it (SwiftShader runs the
  // gated-off branch anyway: +11-12% frame time there, validation/f32newton/f32newton_floor/results_frame_ratio_v62.json).
#ifdef F32N_NOFIX
  FixC X; X.on = false;
#else
  FixC X; X.on = u_velMode == 2 && max(dot(c.u1, c.u1), dot(c.u2, c.u2)) > 16.0;
#endif
  if (!X.on) return X;
  // double-float chord, and the scalars of vSec = vScale chord / dR and b_k = bs_k u_k (unpack and clamp, as segCoef)
  vec2 chx = twoSum(c.p2hi.x, -c.p1hi.x), chy = twoSum(c.p2hi.y, -c.p1hi.y);
  vec2 chHi = vec2(chx.x, chy.x), chLo = vec2(ADD(chx.y, SUB(c.p2lo.x, c.p1lo.x)), ADD(chy.y, SUB(c.p2lo.y, c.p1lo.y)));
  float maxSub = u_betaMax, vm = length(c.dChord) / c.dR;
  float vScale = vm > maxSub ? maxSub / vm : 1.0;
  float g1 = 1.0 / sqrt(1.0 + dot(c.u1, c.u1)), g2 = 1.0 / sqrt(1.0 + dot(c.u2, c.u2));
  float bs1 = g1 * min(1.0, maxSub / (g1 * length(c.u1))), bs2 = g2 * min(1.0, maxSub / (g2 * length(c.u2)));
  float xch = ADD(crossX(dh, dl, chHi), SUB(MUL(dh.x, chLo.y), MUL(dh.y, chLo.x)));   // d1 x chord
  float pv = vScale * xch / c.dR;                                                    // d1 x vSec
  float pb1 = bs1 * crossX(dh, dl, c.u1), pb2 = bs2 * crossX(dh, dl, c.u2);          // d1 x b1, d1 x b2
  X.Y = vec4(dot(dh, c.acc0), dotX(dh, dl, c.acc1), dotX(dh, dl, c.acc2), dot(dh, c.acc3));
  X.chi5 = xch;
#ifdef KNOWN_PATH
  if (c.quint) {
    vec2 w1 = (0.25 * c.dR) * c.acc1, w2 = (0.25 * c.dR) * c.acc2;
    float q0 = -pb1, q4 = -pb2, q1 = q0 + (dh.x * w1.y - dh.y * w1.x), q3 = q4 - (dh.x * w2.y - dh.y * w2.x);
    float q2 = 5.0 * pv - q0 - q1 - q3 - q4;
    X.psi = pv + c.alpha * (vec4(q0, q1, q2, q3) - pv); X.psi4 = pv + c.alpha * (q4 - pv);
    float h = 0.2 * c.dR;
    X.chi.x = h * X.psi.x; X.chi.y = X.chi.x + h * X.psi.y; X.chi.z = X.chi.y + h * X.psi.z; X.chi.w = xch - h * X.psi4;
    return X;
  }
#endif
  float om = 1.0 - c.alpha;
  X.psi = vec4(om * pv - c.alpha * pb1, pv + c.alpha * (2.0 * pv + pb1 + pb2), om * pv - c.alpha * pb2, 0.0);
  X.psi4 = 0.0; X.chi = vec4(0.0);
  return X;
}
float crossD1Dz(FixC X, SegC c, float s) {   // d1 x dz(s)
  float o = 1.0 - s, s2 = s * s, s3 = s2 * s;
#ifdef KNOWN_PATH
  if (c.quint) {
    float o2 = o * o;
    return dot(vec4(5.0 * o2 * o2 * s, 10.0 * o2 * o * s2, 10.0 * o2 * s3, 5.0 * o * s2 * s2), X.chi) + s2 * s3 * X.chi5;
  }
#endif
  return (3.0 * s2 - 2.0 * s3) * X.chi5 + c.dR * ((s3 - 2.0 * s2 + s) * X.psi.x + (s3 - s2) * X.psi.z);
}
float crossD1V(FixC X, SegC c, float s) {    // d1 x dz/dR(s)
  float o = 1.0 - s, s2 = s * s;
#ifdef KNOWN_PATH
  if (c.quint) {
    float o2 = o * o;
    return dot(vec4(o2 * o2, 4.0 * o2 * o * s, 6.0 * o2 * s2, 4.0 * o * s2 * s), X.psi) + s2 * s2 * X.psi4;
  }
#endif
  return o * o * X.psi.x + 2.0 * s * o * X.psi.y + s2 * X.psi.z;
}
RootEval incRoot(vec2 d1, float L1, float f1, SegC c, Seg g, FixC X) {
  vec2 dz = g.dPosSub; float dR = g.dRSub;
  float l = length(dz); vec2 b = d1 - dz; float L = length(b);
  float t1 = dR > 0.0 ? dR * chordDeficit(c, g.sR) / (1.0 + l / dR) : 0.0;
  float dd = dot(d1, dz), S = l + L1 + L, cr = X.on ? crossD1Dz(X, c, g.sR) : d1.x * dz.y - d1.y * dz.x;
  float slack = l > 0.0 ? (dd <= 0.0 ? 2.0 * cr * cr / ((l * L1 - dd) * S) : 2.0 * (l * L1 + dd) / S) : 0.0;
  RootEval o; o.f = f1 + (t1 + slack); o.d = b; o.dist = L; o.dlo = vec2(0.0);
  return o;
}
// Root function at a knot in double-float, also returning d = x - z (full form for readback, the display
// path's form without the double-float square root when g_fast).
float knotRootD(vec2 xhi, vec2 xlo, int s, int m, out vec2 d, out vec2 dl) {
  Knot k = knotAt(s, m);
  RootEval r = rootFn(xhi, xlo, k.phi, k.plo, vec2(0.0), k.r, 0.0);
  d = r.d; dl = r.dlo;
  return r.f;
}
// Sign-only float32 root function at a knot (high parts; two texel fetches).
float knotF32(vec2 xhi, int s, int m) {
  vec4 pv = texelFetch(u_texPosVel, ivec2(m, s), 0);
  float r = texelFetch(u_texAccChg, ivec2(m, s), 0).w;
  return r - length(xhi - pv.xy);
}
// Knot bisection in float32, then both bracket knots in double-float; a wrong float32 sign (possible only
// within rounding of the root) moves the bracket outward one knot at a time. Then Newton on incRoot.
void solveRootF(int s, vec2 xhi, vec2 xlo, out bool inDomain, out int segOut, out float fracOut, out SegC cOut,
                out vec2 d1Out, out float f1Out, out vec2 d1LoOut, out FixC fixOut) {
  inDomain = true;
  fixOut.on = false;
  int lo = 0, hi = u_mT - 1;
  float gLo = knotF32(xhi, s, lo), gHi = knotF32(xhi, s, hi);
  if (gHi <= 0.0) lo = hi - 1;
  else if (gLo >= 0.0) hi = 1;
  else {
    for (int i = 0; i < 12; i++) {
      if (hi - lo <= 1) break;
      int mid = (lo + hi) / 2;
      if (knotF32(xhi, s, mid) < 0.0) lo = mid; else hi = mid;
    }
  }
  vec2 dLo, dHi, lLo, lHi;
  float fLo = knotRootD(xhi, xlo, s, lo, dLo, lLo), fHi = knotRootD(xhi, xlo, s, hi, dHi, lHi);
  for (int w = 0; w < 16; w++) {
    if (fLo >= 0.0 && lo > 0) { hi = lo; fHi = fLo; dHi = dLo; lHi = lLo; lo -= 1; fLo = knotRootD(xhi, xlo, s, lo, dLo, lLo); }
    else if (fHi < 0.0 && hi < u_mT - 1) { lo = hi; fLo = fHi; dLo = dHi; lLo = lHi; hi += 1; fHi = knotRootD(xhi, xlo, s, hi, dHi, lHi); }
    else break;
  }
  d1Out = dLo; f1Out = fLo; d1LoOut = lLo;
  if (hi == u_mT - 1 && fHi <= 0.0) { inDomain = false; segOut = lo; fracOut = 0.9999; cOut = segCoef(s, lo); return; }
  if (lo == 0 && fLo >= 0.0) { segOut = 0; fracOut = 0.0; cOut = segCoef(s, 0); return; }
  SegC c = segCoef(s, lo);
  FixC X = fixConsts(c, dLo, lLo);
  fixOut = X;
  float L1 = length(dLo);
  float sLow = 0.0;
  float sHigh = 0.9999;
  float sc = clamp(fLo / (fLo - fHi), 0.0, 0.9999);
  for (int n = 0; n < 16; n++) {
    if (n >= u_newton) break;
    Seg g = evalSeg(c, sc);
    RootEval r = incRoot(dLo, L1, fLo, c, g, X);
    if (r.f == 0.0) break;
    if (r.f < 0.0) sLow = sc; else sHigh = sc;
    vec2 nrm = r.dist > 1e-6 ? r.d / r.dist : vec2(1.0, 0.0);
    float dfds = max(1e-12, g.dR * max(kappaFloor(), 1.0 + dot(nrm, g.dzdR)));
    float cand = sc - r.f / dfds;
    float next = (cand >= sLow && cand <= sHigh) ? cand : 0.5 * (sLow + sHigh);
    bool done = abs(next - sc) < 1e-7 || (sHigh - sLow) < 1e-7;
    sc = next;
    if (done) break;
  }
  segOut = lo; fracOut = sc; cOut = c;
}
#endif

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
    int seg; float frac; SegC cS;
#ifdef F32N
    vec2 d1, d1l; float f1; FixC X;
    solveRootF(s, xhi, xlo, inDomain, seg, frac, cS, d1, f1, d1l, X);
    Seg g = evalSeg(cS, frac);
    float mRet = float(seg) + frac;
    RootEval r = incRoot(d1, length(d1), f1, cS, g, X);
#else
    solveRoot(s, xhi, xlo, inDomain, seg, frac, cS);
    Seg g = evalSeg(cS, frac);
    float mRet = float(seg) + frac;
    RootEval r = rootFn(xhi, xlo, g.p1hi, g.p1lo, g.dPosSub, g.r1, g.dRSub);
#endif
    vec2 d = r.d;
    float r2 = dot(d, d);
    float rr = sqrt(r2);
    float rSoft2 = r2 + u_softEps * u_softEps;
    float rSoft = sqrt(rSoft2);
    vec2 n = rr > 1e-6 ? d / rr : vec2(1.0, 0.0);
    float kappa; vec2 u;
    evalCancellationFreeKappaAndU(n, g.bHat, g.bMag, g.omb2, kappa, u);
#ifdef F32N
    float nA = dot(n, g.acc);
    if (X.on && g.bMag > 0.5) {
      // sin(beta_hat, n) from d x z' = d1 x z' - dz x z' (FixC), 1 - cos = sin^2 / (1 + cos); n - beta_hat in the
      // beta_hat frame; n.a from the exact d1.acc_k. See FixC.
      float sR = cS.hasNext ? clamp(frac, 0.0, 0.99999) : 0.0;
      float sn = (crossD1V(X, cS, sR) - (g.dPosSub.x * g.dzdR.y - g.dPosSub.y * g.dzdR.x)) / (length(g.dzdR) * rr);
      float cs = dot(g.bHat, n);
      float ang = cs > 0.0 ? sn * sn / (1.0 + cs) : 1.0 - cs;
      vec2 diff = sn * vec2(-g.bHat.y, g.bHat.x) - ang * g.bHat;
      float omb = g.omb2 / (1.0 + g.bMag);
      kappa = max(1e-7, omb + g.bMag * ang);
      u = omb * (g.bHat + diff) + g.bMag * diff;
      float t = clamp(frac, 0.0, 0.99999), t2 = t * t, t3 = t2 * t;
      vec2 dz = g.dPosSub;
#ifdef KNOWN_PATH
      if (cS.quint) {
        float H00 = 2.0 * t3 - 3.0 * t2 + 1.0, H10 = t3 - 2.0 * t2 + t, H01 = -2.0 * t3 + 3.0 * t2, H11 = t3 - t2;
        nA = (H00 * (X.Y.y - dot(cS.acc1, dz)) + H01 * (X.Y.z - dot(cS.acc2, dz))
              + cS.dR * (H10 * dot(cS.j1, d) + H11 * dot(cS.j2, d))) / rr;
      } else
#endif
      {
        vec4 w = vec4(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
        nA = dot(w, X.Y - vec4(dot(cS.acc0, dz), dot(cS.acc1, dz), dot(cS.acc2, dz), dot(cS.acc3, dz))) / rr;
      }
    }
#endif
    float kappa3 = kappa * kappa * kappa;
    vec2 eVel = (g.chg * g.omb2 / (kappa3 * rSoft2)) * u;
    // Radiation term n x ((n - beta) x a) written as (n - beta)(n.a) - kappa a, with u = n - beta and the
    // cancellation-free kappa: the cross-product form rebuilds kappa ~ 1/(2 gamma^2) from O(1) float32 terms.
#ifdef F32N
    vec2 eRad = (g.chg / (kappa3 * rSoft)) * (u * nA - kappa * g.acc);
#else
    vec2 eRad = (g.chg / (kappa3 * rSoft)) * (u * dot(n, g.acc) - kappa * g.acc);
#endif
    vec2 eS = eVel + eRad;
    o.e += eS; o.eRad += eRad; o.eVel += eVel;
    o.bz += n.x * eS.y - n.y * eS.x;
    // Resolution indicator (heuristic, calibrated against the reference in results/):
    //  - bending: shell spacing x gamma over the local radius of curvature;
    //  - speeding up/slowing down: relative change of proper velocity across the segment.
    // The bend term's 1/beta^2 is floored at beta = 0.2: below that the radius of curvature is
    // irrelevant to the field (the velocity field dominates), and the bare 1/beta^2 flagged charges at rest.
    float b2 = max(0.04, g.bMag * g.bMag);
    float aPerp = abs(g.bHat.x * g.acc.y - g.bHat.y * g.acc.x);
    float gam = inversesqrt(max(1e-12, g.omb2));
    float qBend = g.dR * gam * aPerp / b2;
    float qSpeed = 6.0 * g.duRel;
    float q = max(qBend, qSpeed);
    //  - acceleration (live/dragged histories): (i) a jump in the stored acceleration, seen as the
    //    largest second difference over the Catmull-Rom support, which the cubic smooths across the
    //    jump's light sphere; (ii) stored acceleration inconsistent with the change of stored velocity
    //    across the segment (d beta/dR = -a/c^2). Each is turned into an estimated radiation-field error
    //    |delta a| (|u| + kappa) / (kappa^3 r), relative to this source's field; 0.2 maps to q = 1.5.
    //    Tuned on dragged histories (tests/gpu/drag_mask.mjs): >= 94% recall of 1% errors at M = 128-2048,
    //    <= 0.1% extra flags on smooth presets.
    if (cS.hasNext) {
      float cRad = (length(u) + kappa) / (kappa3 * rSoft);
      float d2A = max(length(cS.acc0 - 2.0 * cS.acc1 + cS.acc2), length(cS.acc1 - 2.0 * cS.acc2 + cS.acc3));
      vec2 bK1 = unpackKnotBeta(cS.u1);
      vec2 bK2 = unpackKnotBeta(cS.u2);
      float dAcons = length((bK2 - bK1) / cS.dR + 0.5 * (cS.acc1 + cS.acc2));
      float qAcc = 7.5 * cRad * max(d2A, dAcons) / max(length(eS), 1e-30);
      q = max(q, qAcc);
    }
    if (mRet < 1.0) q = max(q, 1.6);   // inside the first shell: too close to the charge to trust
    // Shells the Nyquist taper has damped no longer carry the true acceleration: flag them.
    if (u_taperOn == 1 && u_scaleL > 0.0) {
      float lambda = 2.0 * PI * u_c / max(0.5, u_omega);
      if (g.dR / (0.5 * lambda) > 0.25) q = max(q, 2.0);
    }
    // Beyond the ring buffer the shells repeat its oldest sample: not a valid history.
#ifdef KNOWN_PATH
    if (u_ringReach > 0.0 && !cS.exact && g.r1 + g.dR > u_ringReach) inDomain = false;
#else
    if (u_ringReach > 0.0 && g.r1 + g.dR > u_ringReach) inDomain = false;
#endif
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
  g_fast = (u_readbackMode == 0 && u_fastRender == 1);
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
