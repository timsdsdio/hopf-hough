"""Certificates for |z'(s)| <= bmax on a cubic Hermite segment, z'(s)=sum_k B_{k,2}(s) V_k(alpha).
V0=(1-a)v+a m0, V2=(1-a)v+a m1, V1=v+a(3v-m0-m1-v), with m0=-beta1, m1=-beta2 (R-derivatives)."""
import numpy as np
S = np.linspace(0, 1, 2001)
W = np.stack([(1-S)**2, 2*S*(1-S), S**2])          # 3 x Ns

def ctrls(v, m0, m1, a):
    V0 = (1-a)*v + a*m0; V2 = (1-a)*v + a*m1; V1 = v + a*(2*v - m0 - m1)
    return V0, V1, V2

def maxspeed(v, m0, m1, a=1.0):
    V = np.stack(ctrls(v, m0, m1, a))               # 3 x D
    return np.linalg.norm(W.T @ V, axis=1).max()

def alpha_paper(v, m0, m1, bm):
    r = 3*v - m0 - m1
    if r @ r <= bm*bm: return 1.0
    d = r - v; A = d @ d; b = v @ d; c = v @ v - bm*bm
    if c >= 0: return 0.0
    return min(1.0, (-b + np.sqrt(max(0, b*b - A*c))) / A)

def _first_root(A, B, C, cap=1.0):
    # smallest root in (0, cap) of A a^2 + B a + C, with C<=0
    roots = []
    if abs(A) < 1e-300:
        if B > 0: roots.append(-C/B)
    else:
        disc = B*B - 4*A*C
        if disc >= 0:
            q = np.sqrt(disc); roots += [(-B-q)/(2*A), (-B+q)/(2*A)]
    al = cap
    for r in roots:
        if 0 < r < al: al = r
    return al

def alpha_sq(v, m0, m1, bm):
    """Bernstein coefficients of the quartic |z'|^2: c0=V0.V0, c1=V0.V1, c2=(2V1.V1+V0.V2)/3, c3=V1.V2, c4=V2.V2."""
    P = v; D0 = m0 - v; D1 = 2*v - m0 - m1; D2 = m1 - v
    pp = P @ P - bm*bm; al = 1.0
    for A, B in [(D0@D1, P@D1 + D0@P), (D1@D2, P@D2 + D1@P),
                 ((2*D1@D1 + D0@D2)/3, (4*P@D1 + P@D2 + D0@P)/3)]:
        al = _first_root(A, B, pp, al)
    return al

def alpha_sharp(v, m0, m1, bm, it=50):
    if maxspeed(v, m0, m1, 1.0) <= bm: return 1.0
    lo, hi = 0.0, 1.0            # max_s |z'_a(s)| is convex in a and = |v| <= bm at a=0
    for _ in range(it):
        md = 0.5*(lo+hi)
        if maxspeed(v, m0, m1, md) <= bm: lo = md
        else: hi = md
    return lo
