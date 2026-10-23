# (a) adversarial random segment data; (b) physical trajectories where the raw cubic CAN overshoot.
import numpy as np, sys; sys.path.insert(0, '.')
from certs import *
rng = np.random.default_rng(1); bm = 0.99995
def ball(n, r):
    x = rng.normal(size=2); x /= np.linalg.norm(x); return x * r * rng.random()**0.5
N = 4000; fireP = fireS = sup = 0; ratP = []; ratS = []; fp_P = fp_S = 0
for _ in range(N):
    v, m0, m1 = ball(2, bm), ball(2, bm), ball(2, bm)
    aP, aS, aH = alpha_paper(v, m0, m1, bm), alpha_sq(v, m0, m1, bm), alpha_sharp(v, m0, m1, bm, it=30)
    s = maxspeed(v, m0, m1) > bm; sup += s
    fireP += aP < 1; fireS += aS < 1
    fp_P += (aP < 1) and not s; fp_S += (aS < 1) and not s
    if aH < 1: ratP.append(aP/aH); ratS.append(aS/aH)
print(f"(a) random data in ball: raw cubic superluminal {sup}/{N}; paper fires {fireP} (false-pos {fp_P}); |z'|^2-Bernstein fires {fireS} (false-pos {fp_S})")
print(f"    when sharp alpha<1: median alpha_paper/alpha_sharp={np.median(ratP):.3f} (p10 {np.percentile(ratP,10):.3f}); alpha_sq/alpha_sharp median={np.median(ratS):.3f} (p10 {np.percentile(ratS,10):.3f})")

# (b1) 1D hyperbolic motion x(t)=(sqrt(1+(a t)^2)-1)/a, knots at t1,t2 (R-derivative sign irrelevant for norms)
print("(b1) hyperbolic motion, segment [t1,t2]: raw max|z'| vs knot speeds")
for a, t1, t2 in [(1, 2, 20), (1, 5, 50), (1, 10, 200), (1, 0, 100), (1, 3, 1000)]:
    x = lambda t: (np.sqrt(1+(a*t)**2)-1)/a; b = lambda t: a*t/np.sqrt(1+(a*t)**2)
    dR = t2-t1; v = np.array([(x(t2)-x(t1))/dR, 0.]); m0 = np.array([b(t1), 0.]); m1 = np.array([b(t2), 0.])
    print(f"   a={a} t=[{t1},{t2}] b1={b(t1):.6f} b2={b(t2):.6f} vsec={v[0]:.6f}  raw max|z'|={maxspeed(v,m0,m1):.6f}  a_paper={alpha_paper(v,m0,m1,bm):.3f} a_sharp={alpha_sharp(v,m0,m1,bm):.3f}")
# (b2) impulsive kick inside a segment: velocity jumps from b*e1 to b*e2 (angle psi) at fraction f of the segment
print("(b2) mid-segment kick (radiation-reaction/collision/laser-like), speed b, deflection psi at fraction f")
for b in [0.99, 0.9999]:
    for psi in [0.05, 0.3, 1.0, 2.5]:
        for f in [0.2, 0.5]:
            e1 = np.array([1, 0.]); e2 = np.array([np.cos(psi), np.sin(psi)])
            v = b*(f*e1 + (1-f)*e2); m0, m1 = b*e1, b*e2
            ms = maxspeed(v, m0, m1)
            print(f"   b={b} psi={psi} f={f}: raw max|z'|={ms:.5f} {'SUPERLUMINAL' if ms>=1 else ''} a_paper={alpha_paper(v,m0,m1,bm):.3f} a_sq={alpha_sq(v,m0,m1,bm):.3f} a_sharp={alpha_sharp(v,m0,m1,bm):.3f}")
# (b3) staggered (leapfrog/PIC) velocities: knot velocity taken half a step late on a circle
print("(b3) circle with knot velocities lagged by half a segment (leapfrog staggering)")
for beta in [0.99, 0.9999]:
    for phi in [0.02, 0.1, 0.3]:
        r = beta/phi; p = lambda t: r*np.array([np.sin(t), -np.cos(t)]); vel = lambda t: beta*np.array([np.cos(t), np.sin(t)])
        v = p(phi/2)-p(-phi/2); m0, m1 = vel(-phi/2 + phi/2), vel(phi/2 + phi/2)
        ms = maxspeed(v, m0, m1)
        print(f"   beta={beta} phi={phi}: raw max|z'|={ms:.6f} {'SUPERLUMINAL' if ms>=1 else ''} a_paper={alpha_paper(v,m0,m1,bm):.3f} a_sharp={alpha_sharp(v,m0,m1,bm):.3f}")
