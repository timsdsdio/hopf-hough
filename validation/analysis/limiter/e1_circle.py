# Circular arc at speed beta, turning angle phi per segment: how do the certificates behave?
import numpy as np, sys; sys.path.insert(0, '.')
from certs import *
def arc(beta, phi):
    # unit-free: chord over dR for arc of angle phi traversed at speed beta (dR = c dt)
    r = beta/ (phi)            # radius so that arc length = beta*dR with dR=1
    th0, th1 = -phi/2, phi/2
    p = lambda t: r*np.array([np.sin(t), -np.cos(t)])
    vel = lambda t: beta*np.array([np.cos(t), np.sin(t)])
    v = p(th1) - p(th0)        # chord / dR (dR=1)
    return v, vel(th0), vel(th1)
bm = 0.99995
print("beta   phi     |V1|/beta-1  phi^2/8     raw max|z'|   a_paper  a_sq     a_sharp")
for beta in [0.99, 0.999, 0.9999]:
    g = 1/np.sqrt(1-beta**2)
    for phi in [0.5/g, 1/g, 2/g, 3/g, 5/g, 0.1, 0.3, 1.0]:
        v, b1, b2 = arc(beta, phi); m0, m1 = b1, b2   # forward-time parametrization; sign irrelevant for norms
        V1 = 3*v - m0 - m1
        print(f"{beta:<6} {phi:7.4f} {np.linalg.norm(V1)/beta-1: .3e}  {phi**2/8: .3e}  {maxspeed(v,m0,m1):.7f}    {alpha_paper(v,m0,m1,bm):.3f}    {alpha_sq(v,m0,m1,bm):.3f}    {alpha_sharp(v,m0,m1,bm):.3f}")
