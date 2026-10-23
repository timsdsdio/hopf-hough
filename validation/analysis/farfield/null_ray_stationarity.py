"""Branch C check 1: is r*E stationary along outgoing null rays, and how far out
must the worldline->scri hand-off be for a synchrotron charge?

Units: rho (orbit radius) = 1, c = 1, q = 1.  Circular orbit in the xy plane.
Compare the exact Lienard-Wiechert field (velocity + acceleration terms) at
x = r n, t = u + r, multiplied by r, against the scri limit
    F(u, n) = n x ((n - beta) x betadot) / kappa^3   at t' with u = t' - n.z(t').
Error = relative L2 (vector) over a u-window spanning +-12 pulse widths.
"""
import numpy as np

def orbit(tp, beta):
    w = beta  # omega = beta c / rho
    z = np.stack([np.cos(w*tp), np.sin(w*tp), 0*tp], -1)
    b = beta*np.stack([-np.sin(w*tp), np.cos(w*tp), 0*tp], -1)
    bd = -beta*w*np.stack([np.cos(w*tp), np.sin(w*tp), 0*tp], -1)
    return z, b, bd

def solve_tret(t, x, beta, t0):
    # Newton on g(t') = t - t' - |x - z(t')|, g' = -1 + nhat.beta > 0 slope sign fixed
    tp = t0.copy()
    for _ in range(60):
        z, b, _ = orbit(tp, beta)
        d = x - z
        R = np.linalg.norm(d, axis=-1)
        g = t - tp - R
        n = d / R[..., None]
        gp = -1 + np.sum(n*b, -1)
        tp = tp - g/gp
    return tp

def E_exact(t, x, beta):
    # initial guess: far-field u-relation
    t0 = t - np.linalg.norm(x, axis=-1)
    tp = solve_tret(t, x, beta, t0)
    z, b, bd = orbit(tp, beta)
    d = x - z; R = np.linalg.norm(d, axis=-1); n = d/R[..., None]
    k = 1 - np.sum(n*b, -1)
    nb = n - b
    vel = (1-beta**2)*nb/(k**3*R**2)[..., None]
    acc = np.cross(n, np.cross(nb, bd))/(k**3*R)[..., None]
    return vel + acc

def F_scri(u, nvec, beta):
    # solve u = t' - n.z(t')
    tp = u.copy()
    for _ in range(80):
        z, b, _ = orbit(tp, beta)
        g = tp - z @ nvec - u
        gp = 1 - b @ nvec
        tp = tp - g/gp
    z, b, bd = orbit(tp, beta)
    n = np.broadcast_to(nvec, b.shape)
    k = 1 - b @ nvec
    return np.cross(n, np.cross(n - b, bd))/(k**3)[..., None]

def run(gamma, psi_over_gamma=0.0, rs=(10, 30, 100, 300, 1e3, 3e3, 1e4, 1e5)):
    beta = np.sqrt(1 - 1/gamma**2)
    psi = psi_over_gamma/gamma
    nvec = np.array([np.cos(psi), 0, np.sin(psi)])
    # pulse centre: velocity along +x at w t' = -pi/2, z = (0,-1,0) -> u0 = t'
    u0 = -np.pi/2/beta
    width = 1/gamma**3
    u = u0 + np.linspace(-12, 12, 4001)*width*(1 + (gamma*psi)**2)**1.5
    F = F_scri(u, nvec, beta)
    out = []
    for r in rs:
        G = r*E_exact(u + r, r*nvec[None, :], beta)
        err = np.linalg.norm(G - F)/np.linalg.norm(F)
        out.append((r, err))
    return out

if __name__ == "__main__":
    for gamma in (7.09, 22.37, 70.71):
        for pg in (0.0, 1.0):
            res = run(gamma, pg)
            s = "  ".join(f"r={r:g}:{e:.2e}" for r, e in res)
            print(f"gamma={gamma:6.2f} psi={pg}/gamma | {s}")
    # fit the hand-off radius for 1% error at each gamma (in-plane)
    print("\nr needed for 1% (in-plane), compare with rho*gamma:")
    for gamma in (3.2, 7.09, 22.37, 70.71):
        rs = np.logspace(0.5, 6.5, 49)
        res = run(gamma, 0.0, rs)
        r1 = next((r for r, e in res if e < 0.01), None)
        if r1 is None:
            print(f"  gamma={gamma:6.2f}: r_1% > {rs[-1]:.1e} (beyond the scanned range)")
        else:
            print(f"  gamma={gamma:6.2f}: r_1% ~ {r1:9.1f}   r_1%/gamma = {r1/gamma:7.1f}   r_1%/gamma^2 = {r1/gamma**2:7.2f}")
