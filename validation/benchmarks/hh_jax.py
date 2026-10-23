"""The hopf-hough method in JAX (float64), for a like-for-like comparison with PyCharge.

Same algorithm as the production shader, minus the GPU-specific parts:
  - each source's history is sampled at stereographic shells R_m = L tan(theta_m / 2)
    (here directly from the trajectory; the browser tool resamples a ring buffer);
  - per observer and source: bisection over knot indices on f(R) = R - |x - z(R)|,
    then safeguarded Newton inside the bracketing cubic Hermite segment, whose
    tangents are limited by the squared-speed Bernstein certificate;
  - velocity direction from the spline derivative, |u| interpolated linearly,
    acceleration Catmull-Rom in shell index; Lienard-Wiechert field (eps = 0).
Float64 needs no double-float tricks. Units: code units, speed of light c.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def shell_radii(M, L):
    th = (np.arange(M) + 0.5) * np.pi / M
    return L * np.tan(0.5 * th)


def build_history(kin_fn, t, M, L, c):
    """kin_fn(t_array) -> (x, y, bx, by, bdx, bdy) with bd = a/c^2. Returns knot arrays."""
    R = shell_radii(M, L)
    x, y, bx, by, bdx, bdy = kin_fn(t - R / c)
    b = np.stack([bx, by], -1)
    g = 1.0 / np.sqrt(np.maximum(1e-300, 1.0 - np.sum(b * b, -1)))
    u = b * g[:, None]
    return dict(R=jnp.asarray(R), Z=jnp.asarray(np.stack([x, y], -1)), U=jnp.asarray(u),
                A=jnp.asarray(np.stack([bdx, bdy], -1)))


def _first_root(A, B, C, cap):
    disc = B * B - 4 * A * C
    q = jnp.sqrt(jnp.maximum(disc, 0.0))
    safeA = jnp.where(jnp.abs(A) < 1e-300, 1.0, A)
    r1 = (-B - q) / (2 * safeA)
    r2 = (-B + q) / (2 * safeA)
    lin = jnp.where(B > 0, -C / jnp.where(B > 0, B, 1.0), cap)
    al = cap
    for r in (r1, r2):
        ok = (disc >= 0) & (jnp.abs(A) >= 1e-300) & (r > 0) & (r < al)
        al = jnp.where(ok, r, al)
    okl = (jnp.abs(A) < 1e-300) & (lin > 0) & (lin < al)
    return jnp.where(okl, lin, al)


def _alpha_sharp(v, m0, m1, bm):
    D0, D1, D2 = m0 - v, 2 * v - m0 - m1, m1 - v
    dot = lambda a, b: jnp.sum(a * b, -1)
    pp = dot(v, v) - bm * bm
    al = jnp.ones_like(pp)
    al = _first_root(dot(D0, D1), dot(v, D1) + dot(D0, v), pp, al)
    al = _first_root(dot(D1, D2), dot(v, D2) + dot(D1, v), pp, al)
    al = _first_root((2 * dot(D1, D1) + dot(D0, D2)) / 3, (4 * dot(v, D1) + dot(v, D2) + dot(D0, v)) / 3, pp, al)
    return jnp.where(pp >= 0, 0.0, al)


def make_field_fn(M, beta_max=0.9999995, newton=8):
    nbis = int(math.ceil(math.log2(M))) + 1

    def segment(h, m1, s):
        m1 = jnp.clip(m1, 0, M - 2)
        m0, m2, m3 = jnp.maximum(m1 - 1, 0), m1 + 1, jnp.minimum(m1 + 2, M - 1)
        R1, R2 = h["R"][m1], h["R"][m2]
        dR = R2 - R1
        z1, z2 = h["Z"][m1], h["Z"][m2]
        u1, u2 = h["U"][m1], h["U"][m2]
        b1 = u1 / jnp.sqrt(1 + jnp.sum(u1 * u1, -1, keepdims=True))
        b2 = u2 / jnp.sqrt(1 + jnp.sum(u2 * u2, -1, keepdims=True))
        chord = z2 - z1
        v = chord / dR[..., None]
        al = _alpha_sharp(v, -b1, -b2, beta_max)[..., None]
        V0, V2 = (1 - al) * v - al * b1, (1 - al) * v - al * b2
        V1 = v + al * (2 * v + b1 + b2)
        s_ = s[..., None]
        h10, h01, h11 = s_**3 - 2 * s_**2 + s_, -2 * s_**3 + 3 * s_**2, s_**3 - s_**2
        z = z1 + h01 * chord + h10 * V0 * dR[..., None] + h11 * V2 * dR[..., None]
        dzdR = (1 - s_) ** 2 * V0 + 2 * s_ * (1 - s_) * V1 + s_**2 * V2
        sp = jnp.linalg.norm(dzdR, axis=-1, keepdims=True)
        bhat = -dzdR / jnp.maximum(sp, 1e-300)
        um = (1 - s_) * jnp.linalg.norm(u1, axis=-1, keepdims=True) + s_ * jnp.linalg.norm(u2, axis=-1, keepdims=True)
        bmag = um / jnp.sqrt(1 + um * um)
        omb2 = 1 / (1 + um * um)
        t, t2, t3 = s_, s_**2, s_**3
        w0, w1 = -0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1
        w2, w3 = -1.5 * t3 + 2 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2
        acc = w0 * h["A"][m0] + w1 * h["A"][m1] + w2 * h["A"][m2] + w3 * h["A"][m3]
        return dict(R=R1 + s * dR, dR=dR, z=z, dzdR=dzdR, bhat=bhat, bmag=bmag[..., 0], omb2=omb2[..., 0], acc=acc)

    def one_source(h, x):  # x: (N, 2)
        fk = lambda m: h["R"][m] - jnp.linalg.norm(x - h["Z"][m], axis=-1)
        lo = jnp.zeros(x.shape[0], jnp.int32)
        hi = jnp.full(x.shape[0], M - 1, jnp.int32)

        def bis(_, st):
            lo, hi = st
            mid = (lo + hi) // 2
            neg = fk(mid) < 0
            go = hi - lo > 1
            return jnp.where(go & neg, mid, lo), jnp.where(go & ~neg, mid, hi)

        lo, hi = jax.lax.fori_loop(0, nbis, bis, (lo, hi))
        fl, fh = fk(lo), fk(lo + 1)
        s = jnp.clip(fl / (fl - fh), 0.0, 1.0 - 1e-12)
        sl, sh = jnp.zeros_like(s), jnp.ones_like(s)

        def newt(_, st):
            s, sl, sh = st
            g = segment(h, lo, s)
            d = x - g["z"]
            dist = jnp.linalg.norm(d, axis=-1)
            f = g["R"] - dist
            sl = jnp.where(f < 0, s, sl)
            sh = jnp.where(f < 0, sh, s)
            n = d / dist[..., None]
            dfds = g["dR"] * jnp.maximum(0.5 * (1 - beta_max), 1 + jnp.sum(n * g["dzdR"], -1))
            cand = s - f / dfds
            s = jnp.where((cand > sl) & (cand < sh), cand, 0.5 * (sl + sh))
            return s, sl, sh

        s, _, _ = jax.lax.fori_loop(0, newton, newt, (s, sl, sh))
        g = segment(h, lo, s)
        d = x - g["z"]
        r = jnp.linalg.norm(d, axis=-1)
        n = d / r[..., None]
        bm, omb2 = g["bmag"][..., None], g["omb2"][..., None]
        omb = omb2 / (1 + bm)
        diff = n - g["bhat"]
        kappa = omb + 0.5 * bm * jnp.sum(diff * diff, -1, keepdims=True)
        uvec = omb * n + bm * diff
        k3 = kappa**3
        ev = omb2 / (k3 * (r * r)[..., None]) * uvec
        cz = uvec[..., 0] * g["acc"][..., 1] - uvec[..., 1] * g["acc"][..., 0]
        er = (1 / (k3[..., 0] * r))[..., None] * jnp.stack([n[..., 1] * cz, -n[..., 0] * cz], -1)
        return ev + er

    @jax.jit
    def field(hists, x):
        # Stack the per-source histories and vectorise over sources (one fused kernel).
        stacked = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *hists)
        return jnp.sum(jax.vmap(lambda h: one_source(h, x))(stacked), axis=0)

    return field
