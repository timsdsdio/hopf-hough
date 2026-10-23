#!/usr/bin/env python3
"""Time PyCharge with the cheapest settings that converge (from tune_pycharge.py) on the benchmark grid."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_benchmark import grid, exact_reference, mask_near, rel_l2, pycharge_fn, timeit
from pycharge.charge import SolverConfig
best = {0.9: (8, 8), 0.99: (32, 8), 0.999: (128, 16), 0.9999: (128, 8)}
X, Y = grid(int(sys.argv[1]) if len(sys.argv) > 1 else 256)
out = []
for n, beta in [(1, 0.9), (1, 0.99), (1, 0.999), (1, 0.9999), (16, 0.9)]:
    fp, nt = best[beta]
    cfg = SolverConfig(fixed_point_max_steps=fp, root_find_max_steps=nt, root_find_rtol=1e-15, root_find_atol=1e-30, root_find_throw=False)
    run = pycharge_fn(n, beta, cfg)
    med, all_t = timeit(lambda: run(X, Y), 3)
    Ex, Ey = run(X, Y)
    err = rel_l2(Ex, Ey, *exact_reference(n, beta, X, Y), mask_near(n, beta, X, Y))
    print(f"N={n:3d} beta={beta}: PyCharge tuned fp={fp} nt={nt}: {med:.3f} s err={err:.2e}", flush=True)
    out.append({"sources": n, "beta": beta, "fixed_point_max_steps": fp, "root_find_max_steps": nt, "seconds": med, "all_seconds": all_t, "rel_l2": err})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_pycharge_tuned.json"), "w"), indent=2)
