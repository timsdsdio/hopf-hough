"""Trajectory definitions shared by the reference port and the PyCharge driver.

All quantities are in hopf-hough *code units*: c = 2.35 (default), lengths in
code length units, charge q in Gaussian-like units (E = q/R^2 for a static charge).

Trajectory dicts (the "trajectory" object of the JSON schema, see README.md):

  {"preset": "synchrotron", "beta": 0.9, "omega": 1.6}
  {"preset": "oscillator",  "beta": 0.3, "omega": 1.6, "cap_amplitude": true}
  {"preset": "linear", "beta": 0.44, "angle": 0.0, "x0": 0.0, "y0": 0.0, "t0": 0.0}

Optional keys on every trajectory: "q" (default 1.0), "phase" (synchrotron /
oscillator phase offset in rad, default 0).

`params()` resolves the preset into concrete orbit parameters EXACTLY as the
original solver's getExactTrajectoryKinematics does for partIdx=0,
totalParts=1 (including its clamps). `position_jax()` returns a JAX position
function built only from those parameters (radius, angular rate, amplitude,
velocity) -- velocity and acceleration are then obtained by autodiff inside
PyCharge, independently of the hand-written derivatives in the port.
"""

import math

C_LIGHT_DEFAULT = 2.35


def params(traj, c=C_LIGHT_DEFAULT):
    p = dict(traj)
    preset = p["preset"]
    beta = float(p["beta"])
    q = float(p.get("q", 1.0))
    phase = float(p.get("phase", 0.0))
    if preset == "synchrotron":
        omega = float(p.get("omega", 1.6))
        v = beta * c
        r_orb = min(0.85, v / max(0.5, omega))          # original clamp
        w_act = v / max(0.05, r_orb)
        return dict(preset=preset, q=q, r=r_orb, w=w_act, phase=phase, beta=beta)
    if preset == "oscillator":
        omega = float(p.get("omega", 1.6))
        if p.get("cap_amplitude", True):
            amp = min(0.75, (beta * c) / max(0.4, omega))  # original clamp
        else:
            amp = (beta * c) / omega
        return dict(preset=preset, q=q, A=amp, w=omega, phase=phase,
                    beta=beta, beta_max_actual=amp * omega / c)
    if preset == "linear":
        th = float(p.get("angle", 0.0))
        return dict(preset=preset, q=q, vx=beta * c * math.cos(th), vy=beta * c * math.sin(th),
                    x0=float(p.get("x0", 0.0)), y0=float(p.get("y0", 0.0)),
                    t0=float(p.get("t0", 0.0)), beta=beta)
    raise ValueError(f"unknown preset {preset!r} (supported: synchrotron, oscillator, linear)")


def position_jax(pr):
    """Return f(t_code) -> jnp.array([x, y, 0]) in code units."""
    import jax.numpy as jnp
    if pr["preset"] == "synchrotron":
        r, w, ph = pr["r"], pr["w"], pr["phase"]
        return lambda t: jnp.stack([r * jnp.cos(w * t + ph), r * jnp.sin(w * t + ph), 0.0 * t])
    if pr["preset"] == "oscillator":
        A, w, ph = pr["A"], pr["w"], pr["phase"]
        return lambda t: jnp.stack([A * jnp.sin(w * t + ph), 0.0 * t, 0.0 * t])
    if pr["preset"] == "linear":
        vx, vy, x0, y0, t0 = pr["vx"], pr["vy"], pr["x0"], pr["y0"], pr["t0"]
        return lambda t: jnp.stack([x0 + vx * (t - t0), y0 + vy * (t - t0), 0.0 * t])
    raise ValueError(pr["preset"])
