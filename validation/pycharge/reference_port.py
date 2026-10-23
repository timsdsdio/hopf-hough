"""Faithful Float64 port of the original solver's analytic reference.

Ports, from reference/original_solver.html:
  * getExactTrajectoryKinematics  (lines ~345-448) -- 'synchrotron' and the
    default/else branch ('oscillator') for partIdx=0, totalParts=1, plus a
    genuine straight-line 'linear' motion (NOT the sawtooth 'uniform' preset).
  * evaluateExactAnalyticalLW     (lines ~750-822) -- 32-step bisection on
    [t - 260/c, t], then <=12 safeguarded Newton steps, then the LW E field
    with Plummer softening R^2 -> R^2 + eps^2 (eps=0 disables it).
  * getExactShockCrestPoint / getPulseCrossingSweepPoints (lines ~451-490).

Conventions copied from the original: bx = v/c, bdx = a/c^2 (= (dβ/dt)/c).
The radiation term is computed as q n×((n-β)×bdx)/(κ³ R), which equals the
Gaussian q n×((n-β)×β̇)/(c κ³ R) with β̇ = dβ/dt.
"""

import math

from trajectories import C_LIGHT_DEFAULT, params


def kinematics(pr, t, c=C_LIGHT_DEFAULT):
    """Returns dict x, y, bx, by, bdx, bdy, charge (mirrors the JS)."""
    if pr["preset"] == "synchrotron":
        beta, w = pr["beta"], pr["w"]
        ang = w * t + pr["phase"]
        ca, sa = math.cos(ang), math.sin(ang)
        return dict(x=pr["r"] * ca, y=pr["r"] * sa,
                    bx=-beta * sa, by=beta * ca,
                    bdx=(-beta * w * ca) / c, bdy=(-beta * w * sa) / c, charge=pr["q"])
    if pr["preset"] == "oscillator":
        A, w = pr["A"], pr["w"]
        ang = w * t + pr["phase"]
        sa, ca = math.sin(ang), math.cos(ang)
        return dict(x=A * sa, y=0.0, bx=(A * w * ca) / c, by=0.0,
                    bdx=(-A * w * w * sa) / (c * c), bdy=0.0, charge=pr["q"])
    if pr["preset"] == "linear":
        dt = t - pr["t0"]
        return dict(x=pr["x0"] + pr["vx"] * dt, y=pr["y0"] + pr["vy"] * dt,
                    bx=pr["vx"] / c, by=pr["vy"] / c, bdx=0.0, bdy=0.0, charge=pr["q"])
    raise ValueError(pr["preset"])


def retarded_time(pr, px, py, t, c=C_LIGHT_DEFAULT):
    t_low, t_high = t - 260.0 / c, t
    for _ in range(32):
        t_mid = 0.5 * (t_low + t_high)
        k = kinematics(pr, t_mid, c)
        if c * (t - t_mid) - math.hypot(px - k["x"], py - k["y"]) > 0:
            t_low = t_mid
        else:
            t_high = t_mid
    t_ret = 0.5 * (t_low + t_high)
    for _ in range(12):
        k = kinematics(pr, t_ret, c)
        dx, dy = px - k["x"], py - k["y"]
        dist = math.hypot(dx, dy)
        f = c * (t - t_ret) - dist
        if abs(f) < 1e-13:
            break
        nx = dx / dist if dist > 1e-12 else 1.0
        ny = dy / dist if dist > 1e-12 else 0.0
        df = -c * max(1e-11, 1.0 - (nx * k["bx"] + ny * k["by"]))
        cand = t_ret - f / df
        t_ret = cand if (t_low <= cand <= t_high) else 0.5 * (t_low + t_high)
        if f > 0:
            t_low = t_ret
        else:
            t_high = t_ret
    return t_ret


def exact_lw(traj, px, py, t, eps=0.0, c=C_LIGHT_DEFAULT):
    """E field (and B_z = (n×E)_z, Gaussian) of one charge. Returns dict."""
    pr = params(traj, c)
    t_ret = retarded_time(pr, px, py, t, c)
    k = kinematics(pr, t_ret, c)
    dx, dy = px - k["x"], py - k["y"]
    r2 = dx * dx + dy * dy
    r = math.sqrt(r2)
    r_soft2 = r2 + eps * eps
    r_soft = math.sqrt(r_soft2)
    nx = dx / r if r > 1e-12 else 1.0
    ny = dy / r if r > 1e-12 else 0.0
    bmag = math.hypot(k["bx"], k["by"])
    omb2 = (1.0 - bmag) * (1.0 + bmag)
    kappa = max(1e-11, 1.0 - (nx * k["bx"] + ny * k["by"]))
    k3 = kappa ** 3
    vel = k["charge"] * omb2 / (k3 * r_soft2)
    evx, evy = vel * (nx - k["bx"]), vel * (ny - k["by"])
    ux, uy = nx - k["bx"], ny - k["by"]
    cz = ux * k["bdy"] - uy * k["bdx"]
    rad = k["charge"] / (k3 * r_soft)
    erx, ery = rad * (ny * cz), rad * (-nx * cz)
    ex, ey = evx + erx, evy + ery
    return dict(ex=ex, ey=ey, bz=nx * ey - ny * ex, t_ret=t_ret, R=r,
                vel=(evx, evy), rad=(erx, ery))


def shock_crest_point(r_ret, t_obs, beta, omega=1.6, c=C_LIGHT_DEFAULT):
    pr = params({"preset": "synchrotron", "beta": beta, "omega": omega}, c)
    t_star = t_obs - r_ret / c
    k = kinematics(pr, t_star, c)
    bm = math.hypot(k["bx"], k["by"])
    bhx = k["bx"] / bm if bm > 1e-12 else 1.0
    bhy = k["by"] / bm if bm > 1e-12 else 0.0
    cx, cy = k["x"] + r_ret * bhx, k["y"] + r_ret * bhy
    return dict(x=cx, y=cy, r_obs=math.hypot(cx, cy), phi_obs=math.atan2(cy, cx))


def pulse_crossing_sweep(r_ret, t_obs, beta, n, omega=1.6, c=C_LIGHT_DEFAULT):
    """Float64 sweep points (the JS also stores f32-rounded copies; we use x64)."""
    g = 1.0 / math.sqrt((1.0 - beta) * (1.0 + beta))
    cr = shock_crest_point(r_ret, t_obs, beta, omega, c)
    half = min(0.45, max(1e-6, 4.0 * g ** -3.0))
    pts = []
    for k in range(n):
        u = (k / (n - 1)) * 2.0 - 1.0
        a = cr["phi_obs"] + u * half
        pts.append([cr["r_obs"] * math.cos(a), cr["r_obs"] * math.sin(a)])
    return pts, half, cr
