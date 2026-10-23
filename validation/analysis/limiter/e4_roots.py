# When the raw cubic is superluminal, how big is the band of on-axis observers that see 3 light-cone roots?
import numpy as np
def herm(z1, z2, m0, m1, dR, s):
    h00=2*s**3-3*s**2+1; h10=s**3-2*s**2+s; h01=-2*s**3+3*s**2; h11=s**3-s**2
    return h00*z1 + h10*m0*dR + h01*z2 + h11*m1*dR
def band(z1, z2, m0, m1, dR, R1=0.0):
    s = np.linspace(0, 1, 200001); R = R1 + s*dR; z = herm(z1, z2, m0, m1, dR, s)
    # observer far ahead on +x: f(R) = R - (x - z(R)) = (R + z(R)) - x ; roots of g(R)=R+z(R) = x. multiple roots iff g non-monotone
    g = R + z; gmax_before = np.maximum.accumulate(g); dip = (gmax_before - g).max()
    return dip, (np.diff(g) < 0).mean()
# 1D hyperbolic motion, retarded parametrization: z(R) with dz/dR = -beta. Use t as coordinate time, R = t_obs - t (c=1).
for a, t1, t2 in [(1, 2, 20), (1, 0, 100), (1, 5, 50)]:
    x = lambda t: (np.sqrt(1+(a*t)**2)-1)/a; b = lambda t: a*t/np.sqrt(1+(a*t)**2)
    tobs = t2; R1, R2 = tobs - t2, tobs - t1     # knot 1 = latest emission (smallest R)
    dip, frac = band(x(t2), x(t1), -b(t2), -b(t1), R2 - R1)
    # observer ahead of motion (+x): n=+x, f = R - (x - z) -> f' = 1 + z' = 1 - beta
    s = np.linspace(0, 1, 200001); R = R1 + s*(R2-R1); z = herm(x(t2), x(t1), -b(t2), -b(t1), R2-R1, s)
    g = R + z; dip = (np.maximum.accumulate(g) - g).max(); frac = (np.diff(g) < 0).mean()
    print(f"hyperbolic a={a} t=[{t1},{t2}] segment dR={R2-R1}: f non-monotone on {100*frac:.1f}% of segment, 3-root observer band width = {dip:.4f} (units of c*t)")
