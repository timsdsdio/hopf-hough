#!/usr/bin/env python3
"""Pulse-crossing sweep (the preprint's hardest test) for PyCharge and the JAX port of the
hopf-hough method. 256 observers cross the synchrotron pulse emitted tangentially at retarded
radius 1.15; observers are rounded to float32 and the exact field is evaluated at the rounded
points. Run once with --dtype 64 and once with --dtype 32 (separate processes)."""
import argparse, json, os, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--dtype", default="64"); args = ap.parse_args()
import jax
if args.dtype == "64":
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.constants import c as C_SI, e as E_CHARGE, epsilon_0, pi
from pycharge import Charge, potentials_and_fields
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_benchmark import C, R_ORB, T_OBS, orbit_params, exact_reference, kin_np

def sweep(beta, n=256, r_ret=1.15):
    w, ph = orbit_params(1, beta)
    g = 1 / np.sqrt(1 - beta**2)
    x, y, bx, by, *_ = kin_np(w, ph[0])(np.array([T_OBS - r_ret / C]))
    b = np.hypot(bx, by)
    cx, cy = x[0] + r_ret * bx[0] / b[0], y[0] + r_ret * by[0] / b[0]
    R, phi = np.hypot(cx, cy), np.arctan2(cy, cx)
    half = min(0.45, max(1e-6, 4 * g**-3))
    a = phi + np.linspace(-1, 1, n) * half
    return np.float32(R * np.cos(a)).astype(np.float64), np.float32(R * np.sin(a)).astype(np.float64)

out = []
for beta in (0.99, 0.999, 0.9999):
    X, Y = sweep(beta)
    Rx, Ry = exact_reference(1, beta, X, Y)
    den = np.sqrt(np.sum(Rx**2 + Ry**2))
    w, ph = orbit_params(1, beta); T0 = C / C_SI
    pos = lambda t, ph=ph[0]: jnp.array([R_ORB * jnp.cos(w * (t / T0) + ph), R_ORB * jnp.sin(w * (t / T0) + ph), 0.0])
    fn = jax.jit(potentials_and_fields([Charge(position_fn=pos, q=E_CHARGE)]))
    dt = jnp.float64 if args.dtype == "64" else jnp.float32
    q = fn(jnp.asarray(X, dt), jnp.asarray(Y, dt), jnp.zeros(len(X), dt), jnp.full(len(X), T_OBS * T0, dt))
    s = 4 * pi * epsilon_0 / E_CHARGE
    E = np.asarray(q.electric, np.float64) * s
    err = np.sqrt(np.nansum((E[:, 0] - Rx)**2 + (E[:, 1] - Ry)**2)) / den
    row = {"beta": beta, "gamma": 1 / np.sqrt(1 - beta**2), "dtype": "float" + args.dtype, "pycharge_sweep_rel_l2": float(err), "pycharge_nonfinite": int((~np.isfinite(E[:, 0])).sum())}
    if args.dtype == "64":
        import hh_jax
        for M in (128, 512, 2048):
            h = hh_jax.build_history(kin_np(w, ph[0]), T_OBS, M, 1.35, C)
            f = hh_jax.make_field_fn(M)
            Eh = np.asarray(f([h], jnp.asarray(np.stack([X, Y], -1))))
            row[f"hh_jax_M{M}_sweep_rel_l2"] = float(np.sqrt(np.sum((Eh[:, 0] - Rx)**2 + (Eh[:, 1] - Ry)**2)) / den)
    print(json.dumps(row), flush=True)
    out.append(row)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_sweep_float{args.dtype}.json"), "w"), indent=2)
