#!/usr/bin/env python3
"""Cross-validate the ported Float64 analytic LW reference against PyCharge v2.

Writes results.json and prints a Markdown table. Run:  python3 compare.py
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from pycharge_fields import pycharge_fields  # noqa: E402
from reference_port import exact_lw, pulse_crossing_sweep  # noqa: E402

T_OBS = 2.508          # = 2.40 + 16*DT_STEP, the original GPU suite's tEval
RINGS = [0.5, 1.15, 3.0, 10.0]
N_RING = 64
N_SWEEP = 256
R_SWEEP = 1.15         # rTarget of the original suite
EPS_ORIG = 0.06


def ring(R, n=N_RING):
    return [[R * math.cos(2 * math.pi * k / n), R * math.sin(2 * math.pi * k / n)] for k in range(n)]


CASES = [
    ("synchrotron b=0.5", {"preset": "synchrotron", "beta": 0.5, "omega": 1.6}),
    ("synchrotron b=0.9", {"preset": "synchrotron", "beta": 0.9, "omega": 1.6}),
    ("synchrotron b=0.99", {"preset": "synchrotron", "beta": 0.99, "omega": 1.6}),
    ("oscillator b=0.3", {"preset": "oscillator", "beta": 0.3, "omega": 1.6}),
    ("oscillator b=0.8 (preset, capped)", {"preset": "oscillator", "beta": 0.8, "omega": 1.6}),
    ("oscillator b=0.8 (uncapped A=bc/w)", {"preset": "oscillator", "beta": 0.8, "omega": 1.6,
                                             "cap_amplitude": False}),
    ("linear b=0.44", {"preset": "linear", "beta": 0.44, "t0": T_OBS}),
    ("linear b=0.9", {"preset": "linear", "beta": 0.9, "t0": T_OBS}),
]


def port_fields(traj, pts, eps):
    out = [exact_lw(traj, x, y, T_OBS, eps=eps) for x, y in pts]
    return (np.array([[o["ex"], o["ey"]] for o in out]), np.array([o["bz"] for o in out]))


def errors(E_ref, E_pc, B_ref, B_pc):
    mag = np.linalg.norm(E_pc, axis=1)
    rms = math.sqrt(np.mean(mag ** 2))
    d = np.linalg.norm(E_ref - E_pc, axis=1)
    rel = d / np.maximum(mag, 1e-14 * rms)
    db = np.abs(B_ref - B_pc)
    relb = db / np.maximum(np.abs(B_pc), 1e-14 * rms)
    ratio = np.linalg.norm(E_ref, axis=1) / np.maximum(mag, 1e-300)
    return dict(max_rel=float(rel.max()), rms_rel=float(math.sqrt(np.mean(rel ** 2))),
                l2_rel=float(math.sqrt(np.sum(d ** 2) / np.sum(mag ** 2))),
                max_rmsnorm=float(d.max() / rms),
                max_rel_Bz=float(relb[np.abs(B_pc) > 1e-8 * rms].max()) if np.any(np.abs(B_pc) > 1e-8 * rms) else 0.0,
                mag_ratio_min=float(ratio.min()), mag_ratio_max=float(ratio.max()), rms_E=rms)


def main():
    results = []
    for name, traj in CASES:
        sets = [(f"ring R={R}", ring(R)) for R in RINGS]
        if traj["preset"] == "synchrotron" and traj["beta"] in (0.9, 0.99):
            pts, half, cr = pulse_crossing_sweep(R_SWEEP, T_OBS, traj["beta"], N_SWEEP)
            sets.append((f"sweep rRet={R_SWEEP} (±{half:.2e} rad, r_obs={cr['r_obs']:.4f})", pts))
        for sname, pts in sets:
            E_pc, B_pc = pycharge_fields(traj, T_OBS, pts)
            E0, B0 = port_fields(traj, pts, 0.0)
            E6, B6 = port_fields(traj, pts, EPS_ORIG)
            r = dict(case=name, set=sname, n=len(pts), eps0=errors(E0, E_pc, B0, B_pc),
                     eps006=errors(E6, E_pc, B6, B_pc))
            results.append(r)
            print(f"{name:38s} {sname:52s} eps0: max {r['eps0']['max_rel']:.2e} rms {r['eps0']['rms_rel']:.2e}"
                  f" Bz {r['eps0']['max_rel_Bz']:.2e} | eps.06: max {r['eps006']['max_rel']:.2e}"
                  f" rms {r['eps006']['rms_rel']:.2e}", flush=True)

    # Softening magnitude on a quasi-static charge, at distance R
    soft = []
    st = {"preset": "linear", "beta": 1e-9}
    for R in (0.3, 1.0, 3.0):
        a = exact_lw(st, R, 0.0, 0.0, eps=0.0)
        b = exact_lw(st, R, 0.0, 0.0, eps=EPS_ORIG)
        soft.append(dict(R=R, vel_rel=abs(b["ex"] - a["ex"]) / abs(a["ex"]),
                         rad_rel_analytic=1 - R / math.sqrt(R * R + EPS_ORIG ** 2)))
    print("softening:", soft)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w") as fh:
        json.dump(dict(t_obs=T_OBS, results=results, softening=soft), fh, indent=1)


if __name__ == "__main__":
    main()
