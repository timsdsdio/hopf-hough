# C^1 existence questions.
import numpy as np, sys; sys.path.insert(0, '.')
from certs import *
from scipy.optimize import minimize
bm = 0.99
# (ii) free knot derivatives, C^1 piecewise cubic, 1D alternating chords +c,-c,... : minimize worst actual max speed
def ms1d(v, a, b):
    # 1D: z'(s) quadratic with z'(0)=a, z'(1)=b, mean v ; max |z'| via endpoints and vertex (exact)
    V1 = 3*v - a - b; A = a - 2*V1 + b; best = max(abs(a), abs(b))
    if abs(A) > 1e-15:
        s = (a - V1)/A
        if 0 < s < 1: best = max(best, abs((1-s)**2*a + 2*s*(1-s)*V1 + s*s*b))
    return best
def worst(ms, chords):
    return max(ms1d(c, ms[k], ms[k+1]) for k, c in enumerate(chords))
print("(ii) 1D sawtooth chords ±c, 8 segments, knot slopes free (C^1 cubic):")
for c in [0.5, 0.6, 0.65, 0.66, 0.67, 0.7, 0.8, 0.95]:
    chords = [c*(-1)**k for k in range(8)]
    best = min((minimize(worst, rng0, args=(chords,), method='Nelder-Mead', options={'maxiter': 20000, 'xatol':1e-9,'fatol':1e-10}).fun
                for rng0 in [np.zeros(9)] + [np.random.default_rng(i).uniform(-bm, bm, 9) for i in range(6)]))
    print(f"   c={c:.2f} (c/bm={c/bm:.3f}): min over knot slopes of worst max|z'| = {best:.4f} -> {'feasible' if best<=bm+1e-6 else 'INFEASIBLE'}  (prediction: feasible iff c <= 2bm/3 = {2*bm/3:.3f})")
# (iii) degree-n with exact knot velocities: minimal n such that |n v - m0 - m1| <= (n-2) bm (iff for the degree-n Bernstein certificate)
def nmin(v, m0, m1, b, nmax=10**6):
    for n in range(3, nmax):
        if np.linalg.norm(n*v - m0 - m1) <= (n-2)*b: return n
    return None
print("(iii) minimal degree n of a C^1 (exact-velocity) interpolant with Bernstein-certified |z'|<=0.99995")
b = 0.99995
cases = {
 'hyperbolic t=[2,20]': (np.array([0.988273,0]), np.array([0.894427,0]), np.array([0.998752,0])),
 'hyperbolic t=[0,100]': (np.array([0.990050,0]), np.array([0.0,0]), np.array([0.999950,0])),
 'kick b=.9999 psi=0.3 f=.2': (0.9999*(0.2*np.array([1,0])+0.8*np.array([np.cos(.3),np.sin(.3)])), 0.9999*np.array([1,0]), 0.9999*np.array([np.cos(.3),np.sin(.3)])),
 'kick b=.9999 psi=2.5 f=.2': (0.9999*(0.2*np.array([1,0])+0.8*np.array([np.cos(2.5),np.sin(2.5)])), 0.9999*np.array([1,0]), 0.9999*np.array([np.cos(2.5),np.sin(2.5)])),
}
for phi in [0.02, 0.1, 0.3, 1.0]:
    beta = 0.9999; r = beta/phi; p = lambda t: r*np.array([np.sin(t), -np.cos(t)]); vel = lambda t: beta*np.array([np.cos(t), np.sin(t)])
    cases[f'circle b=.9999 phi={phi}'] = (p(phi/2)-p(-phi/2), vel(-phi/2), vel(phi/2))
for k, (v, m0, m1) in cases.items():
    n = nmin(v, m0, m1, b); bound = 4*b/(b-np.linalg.norm(v))
    print(f"   {k:28s} |v|={np.linalg.norm(v):.6f}  n*={n}  (a-priori bound 4b/(b-|v|)={bound:.0f})")
