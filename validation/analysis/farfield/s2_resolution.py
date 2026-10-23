"""Branch C check 2: sampling requirements for a (u, n) scri accumulator,
synchrotron at gamma=22.37 (rho=c=1). Interpolate stored Cartesian r*E_rad
between direction pixels at FIXED u (what a reader at an off-grid direction
must do) and measure relative L2 error over the pulse."""
import numpy as np
from null_ray_stationarity import F_scri

g = 22.37; beta = np.sqrt(1-1/g**2); w = 1/g**3
def nvec(phi, psi): return np.array([np.cos(psi)*np.cos(phi), np.cos(psi)*np.sin(phi), np.sin(psi)])
def proj(n, E): return E - (E@n)[:, None]*n[None, :]

u0 = -np.pi/2/beta
print("In-plane azimuth: linear interp between phi=0 and phi=dphi, evaluated at dphi/2")
for k in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
    dphi = k
    um = u0 + dphi/2/beta + np.linspace(-12, 12, 4001)*w
    nA, nB, nM = nvec(0, 0), nvec(dphi, 0), nvec(dphi/2, 0)
    FA, FB, FM = F_scri(um, nA, beta), F_scri(um, nB, beta), F_scri(um, nM, beta)
    FI = proj(nM, 0.5*(FA+FB))
    print(f"  dphi={dphi:7.0e} (= {dphi*g**3:8.2f} gamma^-3, {dphi*g:7.3f}/gamma): rel L2 err {np.linalg.norm(FI-FM)/np.linalg.norm(FM):.2e}")

print("\nElevation psi: linear interp at midpoints, psi in [-4,4]/gamma, phi=0")
for m in (2, 4, 8, 16, 32):
    dpsi = 1/(g*m)
    psis = np.arange(-4/g, 4/g + 1e-12, dpsi)
    num = den = 0.0
    for a, b in zip(psis[:-1], psis[1:]):
        c = 0.5*(a+b)
        uu = u0 + np.linspace(-15, 15, 1501)*w*(1+(g*c)**2)**1.5
        FA, FB, FC = (F_scri(uu, nvec(0, p), beta) for p in (a, b, c))
        FI = proj(nvec(0, c), 0.5*(FA+FB))
        num += np.sum((FI-FC)**2); den += np.sum(FC**2)
    print(f"  dpsi=1/({m}gamma): rel L2 err {np.sqrt(num/den):.2e}")

print("\nCharacteristic-aware azimuth interp (shift u by dphi/omega, rotate) -- only exact by symmetry:")
dphi = 1e-2
um = u0 + dphi/2/beta + np.linspace(-12, 12, 4001)*w
FA = F_scri(um - dphi/2/beta, nvec(0, 0), beta)
c, s = np.cos(dphi/2), np.sin(dphi/2); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
FM = F_scri(um, nvec(dphi/2, 0), beta)
print(f"  dphi=1e-2: rel err {np.linalg.norm(FA@Rz.T-FM)/np.linalg.norm(FM):.2e}")

# Memory estimate: accumulator vs worldline ring buffer, per orbital period
T = 2*np.pi/beta
ppw = 4   # samples per pulse width
Nu = T/(w/ppw)
Npsi = int(8*4)            # +-4/gamma at 1/(4 gamma)  (from table above)
Nphi_naive = int(2*np.pi/(w*beta/ppw))  # need dphi ~ pulse width in phase
print(f"\nPer orbital period at gamma={g}: N_u = {Nu:.2e} samples (pulse width/{ppw})")
Nwl = 512  # worldline is smooth on the orbital scale; paper uses dR~0.01-0.03 on a 5.3-long orbit
print(f"  worldline buffer (smooth, {Nwl}/period x 6 floats): {Nwl*6*4/1e3:.1f} kB  -- ratio u-samples/worldline-samples = {Nu/Nwl:.0f} ~ gamma^3/{g**3/(Nu/Nwl):.0f}")
print(f"  scri accumulator, belt |psi|<4/gamma, fixed-u-interpolable: N_phi={Nphi_naive:.1e}, N_psi={Npsi}"
      f" -> {Nu*Nphi_naive*Npsi*3*4/1e12:.2e} TB")
print(f"  scri accumulator, detector of 1e3 directions: {Nu*1e3*3*4/1e9:.2f} GB")
