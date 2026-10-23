#!/usr/bin/env python3
"""Head-to-head: PyCharge 2 vs the hopf-hough method (JAX port) on the same problems.

Problems: charges on circular orbits (radius 0.85, code units c = 2.35), field E on an
N x N grid over [-3, 3]^2 at t = 2.5. Accuracy is the relative L2 error of E over the grid
against an exact float64 reference (vectorised bisection + Newton on the analytic orbit),
excluding points within 0.05 of a charge. Timings exclude JIT compilation (median of runs).

Usage: python3 run_benchmark.py [--grid 128] [--out results.json] [--quick]
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pycharge"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from scipy.constants import c as C_SI, e as E_CHARGE, epsilon_0, pi  # noqa: E402

import hh_jax  # noqa: E402
from pycharge import Charge, potentials_and_fields  # noqa: E402
from pycharge.charge import SolverConfig  # noqa: E402

C = 2.35
R_ORB = 0.85
T_OBS = 2.5


# ----------------------------------------------------------------------------- problems
def orbit_params(n, beta):
    w = beta * C / R_ORB
    phases = 2 * np.pi * np.arange(n) / n
    return w, phases


def kin_np(w, ph):
    def f(t):
        a = w * t + ph
        return (R_ORB * np.cos(a), R_ORB * np.sin(a), -R_ORB * w * np.sin(a) / C, R_ORB * w * np.cos(a) / C,
                -R_ORB * w * w * np.cos(a) / C**2, -R_ORB * w * w * np.sin(a) / C**2)
    return f


def exact_reference(n, beta, X, Y):
    """Float64 LW field (eps = 0), retarded time by 80-step bisection + 8 Newton steps."""
    w, phases = orbit_params(n, beta)
    Ex = np.zeros_like(X); Ey = np.zeros_like(X)
    for ph in phases:
        kin = kin_np(w, ph)
        lo = np.full_like(X, T_OBS - 20.0); hi = np.full_like(X, T_OBS)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            x, y, *_ = kin(mid)
            f = C * (T_OBS - mid) - np.hypot(X - x, Y - y)
            lo = np.where(f > 0, mid, lo); hi = np.where(f > 0, hi, mid)
        tr = 0.5 * (lo + hi)
        for _ in range(8):
            x, y, bx, by, *_ = kin(tr)
            dx, dy = X - x, Y - y; d = np.hypot(dx, dy)
            f = C * (T_OBS - tr) - d
            df = -C * (1 - (dx * bx + dy * by) / d)
            tr = tr - f / df
        x, y, bx, by, ax, ay = kin(tr)
        dx, dy = X - x, Y - y; r = np.hypot(dx, dy); nx, ny = dx / r, dy / r
        k = 1 - (nx * bx + ny * by); k3 = k**3; omb2 = 1 - (bx * bx + by * by)
        ux, uy = nx - bx, ny - by
        cz = ux * ay - uy * ax
        Ex += omb2 * ux / (k3 * r * r) + ny * cz / (k3 * r)
        Ey += omb2 * uy / (k3 * r * r) - nx * cz / (k3 * r)
    return Ex, Ey


def grid(N):
    s = np.linspace(-3, 3, N)
    X, Y = np.meshgrid(s, s)
    return X, Y


def mask_near(n, beta, X, Y):
    w, phases = orbit_params(n, beta)
    m = np.ones_like(X, bool)
    for ph in phases:
        # exclude points close to the orbit at any time (singular near a charge)
        m &= np.abs(np.hypot(X, Y) - R_ORB) > 0.05
    return m


def rel_l2(Ex, Ey, Rx, Ry, m):
    num = np.sum(((Ex - Rx) ** 2 + (Ey - Ry) ** 2)[m]); den = np.sum((Rx**2 + Ry**2)[m])
    return float(np.sqrt(num / den))


def timeit(fn, runs):
    fn()  # warm-up / compile
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [float(t) for t in ts]


# ----------------------------------------------------------------------------- contenders
def pycharge_fn(n, beta, cfg):
    w, phases = orbit_params(n, beta)
    T0 = C / C_SI  # code time unit in seconds (length unit = 1 m)
    charges = []
    for ph in phases:
        def pos(t_si, ph=ph):
            a = w * (t_si / T0) + ph
            return jnp.array([R_ORB * jnp.cos(a), R_ORB * jnp.sin(a), 0.0])
        charges.append(Charge(position_fn=pos, q=E_CHARGE, solver_config=cfg))
    fn = jax.jit(potentials_and_fields(charges))
    s = 4 * pi * epsilon_0 / E_CHARGE

    def run(X, Y):
        x = jnp.asarray(X.ravel()); y = jnp.asarray(Y.ravel())
        q = fn(x, y, jnp.zeros_like(x), jnp.full_like(x, T_OBS * T0))
        E = np.asarray(q.electric.block_until_ready())
        return (E[:, 0] * s).reshape(X.shape), (E[:, 1] * s).reshape(X.shape)
    return run


def hh_fn(n, beta, M, L=1.35, newton=8):
    w, phases = orbit_params(n, beta)
    hists = [hh_jax.build_history(kin_np(w, ph), T_OBS, M, L, C) for ph in phases]
    field = hh_jax.make_field_fn(M, newton=newton)

    def run(X, Y):
        x = jnp.asarray(np.stack([X.ravel(), Y.ravel()], -1))
        E = np.asarray(field(hists, x).block_until_ready())
        return E[:, 0].reshape(X.shape), E[:, 1].reshape(X.shape)
    return run, hists


PYCHARGE_DEFAULT = SolverConfig()
PYCHARGE_TUNED = {  # cheapest settings found that still match the hopf-hough accuracy (see --tune)
    "fp8_nt8": SolverConfig(fixed_point_max_steps=8, root_find_max_steps=8, root_find_rtol=1e-15, root_find_atol=1e-30, root_find_throw=False),
    "fp4_nt6": SolverConfig(fixed_point_max_steps=4, root_find_max_steps=6, root_find_rtol=1e-15, root_find_atol=1e-30, root_find_throw=False),
    "fp2_nt4": SolverConfig(fixed_point_max_steps=2, root_find_max_steps=4, root_find_rtol=1e-15, root_find_atol=1e-30, root_find_throw=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_python.json"))
    ap.add_argument("--cases", default="1:0.9,1:0.99,1:0.999,16:0.9")
    ap.add_argument("--skip-default", action="store_true", help="skip PyCharge with default (slow) settings")
    args = ap.parse_args()
    X, Y = grid(args.grid)
    out = {"machine": platform.platform(), "processor": platform.processor(), "cpus": os.cpu_count(),
           "jax": jax.__version__, "devices": [str(d) for d in jax.devices()], "grid": args.grid, "cases": []}
    for spec in args.cases.split(","):
        n, beta = int(spec.split(":")[0]), float(spec.split(":")[1])
        m = mask_near(n, beta, X, Y)
        Rx, Ry = exact_reference(n, beta, X, Y)
        case = {"sources": n, "beta": beta, "gamma": 1 / np.sqrt(1 - beta**2), "points": int(X.size), "rows": []}
        contenders = []
        if not args.skip_default:
            contenders.append(("PyCharge 2.0.1, default solver", pycharge_fn(n, beta, PYCHARGE_DEFAULT)))
        for name, cfg in PYCHARGE_TUNED.items():
            contenders.append((f"PyCharge 2.0.1, tuned ({name})", pycharge_fn(n, beta, cfg)))
        for M in (128, 512, 2048):
            run, _ = hh_fn(n, beta, M)
            contenders.append((f"hopf-hough method in JAX, M={M}", run))
        for name, run in contenders:
            try:
                med, all_t = timeit(lambda: run(X, Y), args.runs)
                Ex, Ey = run(X, Y)
                err = rel_l2(Ex, Ey, Rx, Ry, m)
            except Exception as ex:  # noqa: BLE001
                med, all_t, err = None, [], None
                print("  failed:", name, ex)
            row = {"name": name, "seconds": med, "all_seconds": all_t, "rel_l2": err,
                   "points_per_second": (X.size * n / med) if med else None}
            case["rows"].append(row)
            print(f"N={n:3d} beta={beta}: {name:45s} {med if med is None else f'{med:8.3f} s'}  err={err if err is None else f'{err:.2e}'}", flush=True)
        out["cases"].append(case)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
