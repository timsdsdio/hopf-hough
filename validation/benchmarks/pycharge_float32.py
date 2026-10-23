#!/usr/bin/env python3
"""What PyCharge's accuracy becomes in float32 — the precision consumer GPUs run at full speed
(Apple GPUs have no float64 at all). Accuracy only; same grid and reference as run_benchmark.py."""
import json, os, sys
import numpy as np
import jax                       # float64 deliberately NOT enabled
import jax.numpy as jnp
from scipy.constants import c as C_SI, e as E_CHARGE, epsilon_0, pi
from pycharge import Charge, potentials_and_fields
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_benchmark import C, R_ORB, T_OBS, orbit_params, exact_reference, grid, mask_near, rel_l2

N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
X, Y = grid(N)
out = []
for beta in (0.9, 0.99, 0.999, 0.9999):
    w, phases = orbit_params(1, beta)
    T0 = C / C_SI
    pos = lambda t, ph=phases[0]: jnp.array([R_ORB * jnp.cos(w * (t / T0) + ph), R_ORB * jnp.sin(w * (t / T0) + ph), 0.0])
    fn = jax.jit(potentials_and_fields([Charge(position_fn=pos, q=E_CHARGE)]))
    x = jnp.asarray(X.ravel(), jnp.float32); y = jnp.asarray(Y.ravel(), jnp.float32)
    q = fn(x, y, jnp.zeros_like(x), jnp.full_like(x, T_OBS * T0))
    s = 4 * pi * epsilon_0 / E_CHARGE
    E = np.asarray(q.electric, np.float64)
    Ex, Ey = (E[:, 0] * s).reshape(X.shape), (E[:, 1] * s).reshape(X.shape)
    Rx, Ry = exact_reference(1, beta, X, Y)
    m = mask_near(1, beta, X, Y) & np.isfinite(Ex) & np.isfinite(Ey)
    err = rel_l2(Ex, Ey, Rx, Ry, m)
    nonfinite = int((~np.isfinite(Ex)).sum())
    print(f"PyCharge float32, beta={beta}: rel L2 = {err:.3e}, non-finite points = {nonfinite}/{X.size}", flush=True)
    out.append({"beta": beta, "rel_l2": err, "nonfinite": nonfinite, "points": int(X.size), "dtype": str(q.electric.dtype)})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_pycharge_float32.json"), "w"), indent=2)
