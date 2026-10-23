"""Branch C check 1b: decompose the scri-approximation error at finite r into
(a) the Fresnel arrival-time delay |z_perp|^2/(2r) (emission point offset from
the origin, transverse to n) and (b) the residual after removing it.
Reuses null_ray_stationarity.py. Units rho=c=q=1."""
import numpy as np
from null_ray_stationarity import E_exact, F_scri

def errs(gamma, r, psi_g=0.0):
    beta = np.sqrt(1-1/gamma**2); psi = psi_g/gamma
    n = np.array([np.cos(psi), 0, np.sin(psi)])
    u0 = -np.pi/2/beta; w = 1/gamma**3*(1+psi_g**2)**1.5
    u = u0 + np.linspace(-12, 12, 6001)*w
    F = F_scri(u, n, beta)
    G = r*E_exact(u + r, r*n[None, :], beta)
    e_raw = np.linalg.norm(G-F)/np.linalg.norm(F)
    # emission point for this direction: z=(0,-1,0); z_perp^2 = 1 - (n.z)^2 = 1
    zp2 = 1.0
    dt = zp2/(2*r)
    G2 = r*E_exact(u + r + dt, r*n[None, :], beta)   # read exact field shifted by Fresnel delay
    e_fr = np.linalg.norm(G2-F)/np.linalg.norm(F)
    return e_raw, e_fr, dt/w

print("gamma   r        raw_err   after_Fresnel_shift   delay/pulse_width")
for g in (7.09, 22.37, 70.71):
    for r in (3, 10, 30, 100, 1e3, 1e4):
        a, b, s = errs(g, r)
        print(f"{g:6.2f} {r:8g}  {a:9.2e}   {b:9.2e}             {s:9.1f}")
print("\nelevated psi=1/gamma, gamma=22.37")
for r in (3, 10, 30, 100, 1e3):
    a, b, s = errs(22.37, r, 1.0)
    print(f"  r={r:6g} raw={a:.2e} fresnel-corrected={b:.2e}")
