#!/usr/bin/env python3
"""Compute Liénard–Wiechert E and B_z with PyCharge (v2, JAX) for hopf-hough code-unit inputs.

Usage:
    python3 pycharge_fields.py input.json output.json
    python3 pycharge_fields.py input.json            # writes to stdout

Input/output schema: see README.md in this folder.

Unit mapping (code -> SI): length unit L0 metres (default 1 m), time unit
T0 = c_code * L0 / c_SI, so beta is preserved; charge q_code -> q_code * e.
PyCharge returns SI fields; we convert back with
    E_code   = E_SI   * 4*pi*eps0 * L0^2 / e
    B_z,code = B_z,SI * c_SI * 4*pi*eps0 * L0^2 / e
so a static unit charge gives |E| = 1/R^2 and, in the Gaussian convention,
B = n × E (same units as E). No softening is applied (eps = 0).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)  # PyCharge needs float64 for this to mean anything

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.constants import c as C_SI, e as E_CHARGE, epsilon_0, pi  # noqa: E402

from pycharge import Charge, potentials_and_fields  # noqa: E402
from trajectories import C_LIGHT_DEFAULT, params, position_jax  # noqa: E402


def make_charge(traj, c_code, L0):
    pr = params(traj, c_code)
    f_code = position_jax(pr)
    T0 = c_code * L0 / C_SI
    pos_si = lambda t_si: L0 * f_code(t_si / T0)  # noqa: E731
    return Charge(position_fn=pos_si, q=pr["q"] * E_CHARGE)


def pycharge_fields(trajectories, t, points, c_code=C_LIGHT_DEFAULT, L0=1.0):
    """trajectories: dict or list of dicts; t: code time; points: (N,2) code units.

    Returns (E (N,2) ndarray, B_z (N,) ndarray) in code units.
    """
    if isinstance(trajectories, dict):
        trajectories = [trajectories]
    charges = [make_charge(tr, c_code, L0) for tr in trajectories]
    T0 = c_code * L0 / C_SI
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    fn = jax.jit(potentials_and_fields(charges))
    x = jnp.asarray(pts[:, 0] * L0)
    y = jnp.asarray(pts[:, 1] * L0)
    z = jnp.zeros_like(x)
    tt = jnp.full_like(x, float(t) * T0)
    qty = fn(x, y, z, tt)
    E_si = np.asarray(qty.electric)
    B_si = np.asarray(qty.magnetic)
    s = 4 * pi * epsilon_0 * L0 ** 2 / E_CHARGE
    E = E_si[:, :2] * s
    Bz = B_si[:, 2] * s * C_SI
    if not (np.all(np.isfinite(E)) and np.all(np.isfinite(Bz))):
        raise FloatingPointError("PyCharge returned non-finite fields (observer on the trajectory?)")
    return E, Bz


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    with open(argv[1]) as fh:
        spec = json.load(fh)
    trajs = spec.get("trajectory", spec.get("trajectories"))
    E, Bz = pycharge_fields(trajs, spec["t"], spec["points"],
                            c_code=spec.get("c", C_LIGHT_DEFAULT), L0=spec.get("L0_m", 1.0))
    out = {"E": E.tolist(), "B_z": Bz.tolist(),
           "meta": {"pycharge_version": _pc_version(), "c": spec.get("c", C_LIGHT_DEFAULT),
                    "units": "code units, Gaussian-like (static unit charge: |E| = 1/R^2), eps = 0"}}
    txt = json.dumps(out)
    if len(argv) > 2:
        with open(argv[2], "w") as fh:
            fh.write(txt)
    else:
        print(txt)
    return 0


def _pc_version():
    try:
        from importlib.metadata import version
        return version("pycharge")
    except Exception:  # pragma: no cover
        return "unknown"


if __name__ == "__main__":
    sys.exit(main(sys.argv))
