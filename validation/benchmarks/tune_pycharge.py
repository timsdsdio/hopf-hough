#!/usr/bin/env python3
"""Find the cheapest PyCharge solver settings that reach a target accuracy on the benchmark grid."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_benchmark import grid, exact_reference, mask_near, rel_l2, pycharge_fn
from pycharge.charge import SolverConfig
X, Y = grid(96)
res = []
for beta in (0.9, 0.99, 0.999, 0.9999):
    Rx, Ry = exact_reference(1, beta, X, Y); m = mask_near(1, beta, X, Y)
    for fp in (1, 8, 32, 128, 256):
        for nt in (4, 8, 16, 32):
            cfg = SolverConfig(fixed_point_max_steps=fp, root_find_max_steps=nt, root_find_rtol=1e-15, root_find_atol=1e-30, root_find_throw=False)
            Ex, Ey = pycharge_fn(1, beta, cfg)(X, Y)
            ok = np.isfinite(Ex).all()
            err = rel_l2(np.nan_to_num(Ex), np.nan_to_num(Ey), Rx, Ry, m)
            res.append({"beta": beta, "fixed_point_max_steps": fp, "root_find_max_steps": nt, "rel_l2": err, "finite": bool(ok)})
            print(f"beta={beta} fp={fp:3d} nt={nt:2d}: err={err:.2e}{'' if ok else ' (non-finite)'}", flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_tuning.json"), "w"), indent=2)
