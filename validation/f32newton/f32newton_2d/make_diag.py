#!/usr/bin/env python3
"""Builds diag.html: a COPY of web/index.html (unchanged on disk) with extra, diagnostic-only kernels:
  v6_cnt     v6 + COUNT_ITERS (readback layout 0 returns (Ex, Ey, K + 4096 F, N) instead of (Ex, Ey, emitX, emitY))
  v6f_cnt    v6f + COUNT_ITERS
  v6f_dfbis  v6f with the knot bisection in double-float: output must equal v6f
  v6f_naive  v6f with the naive float32 chord deficit 1 - |dz/dR|^2: attributes the floor
  K = double-float knot evaluations, F = float32 knot evaluations, N = in-segment root evaluations (Newton loop),
  each summed over the sources of the pixel.
Hooks: window.DIAG_FULL = true renders the frame with the readback (full) path; window.DIAG_FAST = true runs readback
with the display path (fastRender, render Newton cap). Usage: python3 make_diag.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', '..', 'web', 'index.html')
src = open(SRC).read()


def rep(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, (n, old[:80])
    src = src.replace(old, new)


# ---- shader: counters
rep("""float kappaFloor() {""", """#ifdef COUNT_ITERS
int gK = 0, gN = 0, gF = 0;
#define CNT_K gK++;
#define CNT_N gN++;
#define CNT_F gF++;
#else
#define CNT_K
#define CNT_N
#define CNT_F
#endif
float kappaFloor() {""")
rep("""float knotF(vec2 xhi, vec2 xlo, Knot k) {
  if (u_precision != 1 || !g_fast) return rootFn(""", """float knotF(vec2 xhi, vec2 xlo, Knot k) {
  if (u_precision != 1 || !g_fast) { CNT_K } else { CNT_F }
  if (u_precision != 1 || !g_fast) return rootFn(""")
rep("    float fc = rootFn(", "    CNT_K float fc = rootFn(")
rep("      fLo = rootFn(xhi, xlo, kn.phi", "      CNT_K fLo = rootFn(xhi, xlo, kn.phi")
rep("      fHi = rootFn(xhi, xlo, kn.phi", "      CNT_K fHi = rootFn(xhi, xlo, kn.phi")
rep("""    Seg g = evalSeg(c, sc);
    RootEval r = rootFn(""", """    Seg g = evalSeg(c, sc); CNT_N
    RootEval r = rootFn(""")
rep("""    RootEval r = incRoot(dLo, L1, fLo, c, g);""", """    CNT_N RootEval r = incRoot(dLo, L1, fLo, c, g);""")
rep("""float knotRootD(vec2 xhi, vec2 xlo, int s, int m, out vec2 d) {
""", """float knotRootD(vec2 xhi, vec2 xlo, int s, int m, out vec2 d) {
  CNT_K
""")
rep("""float knotF32(vec2 xhi, int s, int m) {
""", """float knotF32(vec2 xhi, int s, int m) {
  CNT_F
""")
rep("    else outColor = vec4(F.e, F.emit0);", """#ifdef COUNT_ITERS
    else outColor = vec4(F.e, float(gK) + 4096.0 * float(gF), float(gN));
#else
    else outColor = vec4(F.e, F.emit0);
#endif""")
rep("  g_fast = (u_readbackMode == 0 && u_fastRender == 1);", "  g_fast = (u_fastRender == 1);")
# ---- shader: double-float bisection, naive deficit
rep("""  float gLo = knotF32(xhi, s, lo), gHi = knotF32(xhi, s, hi);""", """#ifdef F64BIS
  vec2 dT; float gLo = knotRootD(xhi, xlo, s, lo, dT), gHi = knotRootD(xhi, xlo, s, hi, dT);
#else
  float gLo = knotF32(xhi, s, lo), gHi = knotF32(xhi, s, hi);
#endif""")
rep("""      if (knotF32(xhi, s, mid) < 0.0) lo = mid; else hi = mid;""", """#ifdef F64BIS
      if (knotRootD(xhi, xlo, s, mid, dT) < 0.0) lo = mid; else hi = mid;
#else
      if (knotF32(xhi, s, mid) < 0.0) lo = mid; else hi = mid;
#endif""")
rep("""  float t1 = dR > 0.0 ? dR * chordDeficit(c, g.sR) / (1.0 + l / dR) : 0.0;""", """#ifdef NAIVE_D
  vec2 vb = dz / max(dR, 1e-30); float t1 = dR > 0.0 ? dR * (1.0 - dot(vb, vb)) / (1.0 + l / dR) : 0.0;
#else
  float t1 = dR > 0.0 ? dR * chordDeficit(c, g.sR) / (1.0 + l / dR) : 0.0;
#endif""")
# ---- programs
rep("""                if (v6fk) { glProgV6FK = v6fk.prog; glUniformsV6FK = v6fk.uni; }""", """                if (v6fk) { glProgV6FK = v6fk.prog; glUniformsV6FK = v6fk.uni; }
                const DEFS = { v6_cnt: '#define COUNT_ITERS\\n', v6f_cnt: '#define F32N\\n#define COUNT_ITERS\\n',
                               v6f_dfbis: '#define F32N\\n#define F64BIS\\n', v6f_naive: '#define F32N\\n#define NAIVE_D\\n' };
                for (const [name, d] of Object.entries(DEFS)) {
                    const a = buildV6(fsSourceV6.replace('#version 300 es\\n', '#version 300 es\\n' + d));
                    const b = buildV6(fsSourceV6.replace('#version 300 es\\n', '#version 300 es\\n' + d + '#define KNOWN_PATH\\n'));
                    DIAG[name] = { prog: a.prog, uni: a.uni, progK: b.prog, uniK: b.uni };
                }""")
rep("""        function isV6Family() { return state.kernel === 'v6' || state.kernel === 'v6f'; }""",
    """        const DIAG = {};
        function isV6Family() { return state.kernel === 'v6' || state.kernel === 'v6f' || !!DIAG[state.kernel]; }""")
rep("""        function selectKernel() {
""", """        function selectKernel() {
            if (DIAG[state.kernel]) { glProg = DIAG[state.kernel].prog; glUniforms = DIAG[state.kernel].uni; return; }
""")
rep("""            if (!isV6Family() || !glProgV6) return false;
""", """            if (!isV6Family() || !glProgV6) return false;
            if (DIAG[state.kernel]) {
                const D = DIAG[state.kernel], w = knownPathOnGPU ? D.progK : D.prog;
                if (glProg === w) return false;
                glProg = w; glUniforms = w === D.progK ? D.uniK : D.uni; return true;
            }
""")
rep("            if (glProg !== glProgV6 && glProg !== glProgV6K && glProg !== glProgV6F && glProg !== glProgV6FK) return;",
    "            if (glProg === glProgV5) return;\n            if (window.DIAG_FULL) readback = true; else if (window.DIAG_FAST) readback = false;")
rep("                    if (state.kernel === 'v6f') {", "                    if (state.kernel.startsWith('v6f')) {")
open(os.path.join(HERE, 'diag.html'), 'w').write(src)
print('wrote diag.html')
