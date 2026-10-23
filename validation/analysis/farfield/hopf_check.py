"""Branch C check 3: which bundle carries transverse polarization over S^2?
(a) Clutching winding of the tangent bundle TS^2 vs the Hopf bundle, computed
    numerically from the stereographic chart change w = 1/z on the equator.
(b) Spin-weight s functions as functions on SU(2)=S^3: check U(1) equivariance and
    whether f(-g)=f(g) (descends to SO(3)=RP^3 = unit tangent bundle) for s=1 vs s=1/2.
(c) Cartesian transverse storage needs no frame: n.E=0 is a linear constraint in
    the trivial bundle S^2 x R^3 = TS^2 (+) normal line."""
import numpy as np

# (a) winding numbers on the equator |z|=1
th = np.linspace(0, 2*np.pi, 20001); z = np.exp(1j*th)
J = -1/z**2                      # dw/dz: how a tangent vector's complex component transforms
wind = lambda f: np.round(np.sum(np.diff(np.unwrap(np.angle(f))))/(2*np.pi), 6)
print("(a) clutching winding: tangent bundle (dw/dz) =", wind(J),
      "| spinor/Hopf (dw/dz)^(1/2) =", wind(1j/z), "  -> Euler/Chern 2 vs 1")

# (b) SU(2) lift
sx = np.array([[0, 1], [1, 0]], complex); sy = np.array([[0, -1j], [1j, 0]]); sz = np.diag([1, -1]).astype(complex)
def su2_to_so3(U):
    S = [sx, sy, sz]
    return np.array([[0.5*np.trace(S[i] @ U @ S[j] @ U.conj().T).real for j in range(3)] for i in range(3)])
rng = np.random.default_rng(0)
def rand_su2():
    q = rng.normal(size=4); q /= np.linalg.norm(q); a, b, c, d = q
    return np.array([[a+1j*b, c+1j*d], [-c+1j*d, a-1j*b]])
Efield = lambda n: np.cross(n, np.cross(n, np.array([0.3, -1.0, 0.5])))   # any transverse field
def f_s1(U):        # spin-weight +1 component in the frame carried by U
    R = su2_to_so3(U); n = R[:, 2]; m = (R[:, 0] + 1j*R[:, 1])/np.sqrt(2)
    return m @ Efield(n)
def f_shalf(U):     # a spin-1/2 quantity: component of a fixed spinor in the dyad carried by U
    return (U.conj().T @ np.array([1.0, 0.4j]))[0]
U = rand_su2(); chi = 0.7
Uchi = U @ np.diag([np.exp(-1j*chi/2), np.exp(1j*chi/2)])
print("(b) s=1:   f(U e^{-i chi sz/2}) / f(U) =", np.round(f_s1(Uchi)/f_s1(U), 6), " expected e^{-i chi} =", np.round(np.exp(-1j*chi), 6))
print("    s=1:   f(-U)/f(U) =", np.round(f_s1(-U)/f_s1(U), 6), "  -> descends to SO(3) = RP^3 (no Hopf needed)")
print("    s=1/2: f(-U)/f(U) =", np.round(f_shalf(-U)/f_shalf(U), 6), " -> genuinely needs SU(2) = S^3 (Hopf)")

# (c) Cartesian storage: interpolate between nearby directions, re-project; error is O(h^2), no special points
def interp_err(n0, h):
    t1 = np.cross(n0, [0.36, 0.48, 0.8]); t1 /= np.linalg.norm(t1)
    nA = n0*np.cos(h) + t1*np.sin(h); nB = n0*np.cos(h) - t1*np.sin(h)
    EI = 0.5*(Efield(nA) + Efield(nB)); EI -= (EI @ n0)*n0
    return np.linalg.norm(EI - Efield(n0))/np.linalg.norm(Efield(n0))
for name, n0 in (("north pole", np.array([0, 0, 1.0])), ("equator", np.array([1.0, 0, 0])), ("random", np.array([0.2, -0.7, 0.68])/np.linalg.norm([0.2, -0.7, 0.68]))):
    print(f"(c) Cartesian+reproject at {name:10s}: h=1e-2 err {interp_err(n0, 1e-2):.2e}, h=5e-3 err {interp_err(n0, 5e-3):.2e}")
