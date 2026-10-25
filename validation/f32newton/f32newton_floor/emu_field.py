#!/usr/bin/env python3
"""numpy emulation of the v6f field evaluation (web/index.html, F32N + KNOWN_PATH quintic, velMode 2) at the GPU's retarded root.
Every stage can run in float32 (as the shader) or float64, to attribute the remaining field-error floor.
Stages:
  store  knot data: stored float32 (u, acc, R, hi+lo positions) vs float64 truth (emitU*64, emitBd*64, emitR64, emitX64)
  coef   segCoef arithmetic (vSec, b1, b2, Q_k, alpha, control points)
  pos    evalSeg position offset dPosSub
  vel    evalSeg hodograph dzdR and bHat = -dzdR/|dzdR|
  umag   evalSeg |u| interpolation, bMag, omb2 = 1/(1+|u|^2)
  acc    evalSeg acceleration (quintic Hermite of acc, jerk)
  sep    separation d = d1 - dz (d1 = x - z1 from double-float, rounded to float32)
  norm   rr = |d|, n = d / rr
  kap    kappa = (1-b) + b/2 |n - bHat|^2 and u = (1-b) n + b (n - bHat)
  fvel   velocity-field arithmetic (omb2 / (kappa^3 R^2)) u
  frad   radiation arithmetic (1/(kappa^3 R)) (u (n.a) - kappa a)
  root   'gpu' = GPU's root (from readback layout 2), 'f64' = root of the float64 spline
Plus alternative formulas (fix candidates), selected with P['form'].
Usage: python3 emu_field.py exp_known.json"""
import json, sys
import numpy as np

F32, F64 = np.float32, np.float64


def V(T, a):
    return (T(a[0]), T(a[1]))


def add(a, b): return (a[0] + b[0], a[1] + b[1])
def sub(a, b): return (a[0] - b[0], a[1] - b[1])
def scl(k, a): return (k * a[0], k * a[1])
def dot(a, b): return a[0] * b[0] + a[1] * b[1]
def length(a): return np.sqrt(dot(a, a))
def cross(a, b): return a[0] * b[1] - a[1] * b[0]


class Case:
    def __init__(self, c):
        self.c = c
        k = c['knots']
        self.M = c['M']
        self.bm = F32(c['betaMax'])
        self.k = {n: (np.array(v) if isinstance(v, list) else v) for n, v in k.items()}
        self.obs = np.array(c['obs'])
        self.exact = np.array(c['exact'])
        self.dbg = np.array(c['dbg'])
        self.gpu = np.array(c['gpu'])

    def knot(self, m, store):
        k = self.k
        if store == 'f32':
            T = F32
            phi = (T(k['emitX'][m]), T(k['emitY'][m])); plo = (T(k['emitXlo'][m]), T(k['emitYlo'][m]))
            pos64 = (float(phi[0]) + float(plo[0]), float(phi[1]) + float(plo[1]))
            u = (T(k['emitUx'][m]), T(k['emitUy'][m])); acc = (T(k['emitBdx'][m]), T(k['emitBdy'][m])); r = T(k['emitR'][m])
            r64 = float(r)
        else:
            # exact kinematics at the stored float32 shell radius (positions: the stored hi+lo pair is already exact)
            e = k['exactAtR32']
            phi = (F32(k['emitX'][m]), F32(k['emitY'][m])); plo = (F32(k['emitXlo'][m]), F32(k['emitYlo'][m]))
            pos64 = (float(phi[0]) + float(plo[0]), float(phi[1]) + float(plo[1]))
            u = (F64(e['ux'][m]), F64(e['uy'][m])); acc = (F64(e['bdx'][m]), F64(e['bdy'][m])); r = F64(F32(k['emitR'][m]))
            r64 = float(r)
        j = (F32(k['emitJx'][m]), F32(k['emitJy'][m]))
        return dict(phi=phi, plo=plo, pos64=pos64, u=u, acc=acc, r=r, r64=r64, j=j)


def first_root(A, B, C, cap):
    al = cap
    if abs(A) < 1e-30:
        if B > 0:
            r = -C / B
            if 0 < r < al: al = r
    else:
        disc = B * B - 4 * A * C
        if disc >= 0:
            q = np.sqrt(disc)
            for r in ((-B - q) / (2 * A), (-B + q) / (2 * A)):
                if 0 < r < al: al = r
    return al


def alpha_quint(v, D, bm):
    D0, D1, D2, D3, D4 = D
    pp = dot(v, v) - bm * bm
    if pp >= 0: return 0.0
    e0, e1, e2, e3, e4 = (dot(v, x) for x in D)
    al = 1.0
    al = first_root(dot(D0, D1), e0 + e1, pp, al)
    al = first_root((12 * dot(D0, D2) + 16 * dot(D1, D1)) / 28, (12 * (e0 + e2) + 32 * e1) / 28, pp, al)
    al = first_root((8 * dot(D0, D3) + 48 * dot(D1, D2)) / 56, (8 * (e0 + e3) + 48 * (e1 + e2)) / 56, pp, al)
    al = first_root((2 * dot(D0, D4) + 32 * dot(D1, D3) + 36 * dot(D2, D2)) / 70, (2 * (e0 + e4) + 32 * (e1 + e3) + 72 * e2) / 70, pp, al)
    al = first_root((8 * dot(D1, D4) + 48 * dot(D2, D3)) / 56, (8 * (e1 + e4) + 48 * (e2 + e3)) / 56, pp, al)
    al = first_root((12 * dot(D2, D4) + 16 * dot(D3, D3)) / 28, (12 * (e2 + e4) + 32 * e3) / 28, pp, al)
    al = first_root(dot(D3, D4), e3 + e4, pp, al)
    return al


def seg_coef(case, m1, P):
    T = F32 if P['coef'] == 'f32' else F64
    k1, k2 = case.knot(m1, P['store']), case.knot(m1 + 1, P['store'])
    bm = T(case.bm)
    r1, r2 = T(k1['r']), T(k2['r'])
    dR = max(T(1e-6), r2 - r1)
    if T is F32:
        dChord = add(sub(k2['phi'], k1['phi']), sub(k2['plo'], k1['plo']))
    else:
        dChord = (F64(k2['pos64'][0] - k1['pos64'][0]), F64(k2['pos64'][1] - k1['pos64'][1]))
    vSec = scl(T(1) / dR, dChord) if False else (dChord[0] / dR, dChord[1] / dR)
    vm = length(vSec)
    if vm > bm: vSec = scl(bm / vm, vSec)
    u1, u2 = V(T, k1['u']), V(T, k2['u'])
    b1 = scl(T(1) / np.sqrt(T(1) + dot(u1, u1)), u1) if False else (u1[0] / np.sqrt(T(1) + dot(u1, u1)), u1[1] / np.sqrt(T(1) + dot(u1, u1)))
    b2 = (u2[0] / np.sqrt(T(1) + dot(u2, u2)), u2[1] / np.sqrt(T(1) + dot(u2, u2)))
    for b in (b1, b2):
        assert length(b) <= bm
    a1, a2 = V(T, k1['acc']), V(T, k2['acc'])
    Q0, Q4 = scl(T(-1), b1), scl(T(-1), b2)
    Q1 = add(Q0, scl(T(0.25) * dR, a1))
    Q3 = sub(Q4, scl(T(0.25) * dR, a2))
    Q2 = sub(sub(sub(sub(scl(T(5), vSec), Q0), Q1), Q3), Q4)
    Qs = (Q0, Q1, Q2, Q3, Q4)
    aq = T(alpha_quint(vSec, [sub(q, vSec) for q in Qs], bm))
    v = [add(vSec, scl(aq, sub(q, vSec))) for q in Qs]
    n1, n2 = length(u1), length(u2)
    um = T(0.5) * (n1 + n2); bmm = um / np.sqrt(T(1) + um * um)
    ap = max(abs(cross(b1, a1)) / max(length(b1), T(1e-7)), abs(cross(b2, a2)) / max(length(b2), T(1e-7)))
    uHerm = aq >= 1 and dR * np.sqrt(T(1) + um * um) * ap / max(T(0.04), bmm * bmm) < 1
    return dict(T=T, r1=r1, dR=dR, dChord=dChord, v=v, aq=aq, u1=u1, u2=u2, a1=a1, a2=a2, j1=V(T, k1['j']), j2=V(T, k2['j']),
                uHerm=uHerm, phi=k1['phi'], plo=k1['plo'], pos64=k1['pos64'], r1_64=k1['r64'], dR64=k2['r64'] - k1['r64'])


def seg_pos(c, t, T):
    v = [V(T, x) for x in c['v']]; dR = T(c['dR']); dChord = V(T, c['dChord']); s = T(t)
    oms = T(1) - s; o2 = oms * oms; o3 = o2 * oms; o4 = o3 * oms; s2 = s * s; s3 = s2 * s; t4 = s2 * s2
    P1 = scl(T(0.2) * dR, v[0]); P2 = add(P1, scl(T(0.2) * dR, v[1])); P3 = add(P2, scl(T(0.2) * dR, v[2]))
    P4 = sub(dChord, scl(T(0.2) * dR, v[4]))
    d = add(add(add(add(scl(T(5) * o4 * s, P1), scl(T(10) * o3 * s2, P2)), scl(T(10) * o2 * s3, P3)), scl(T(5) * oms * t4, P4)), scl(t4 * s, dChord))
    return d


def seg_vel(c, t, T):
    v = [V(T, x) for x in c['v']]; s = T(t)
    oms = T(1) - s; o2 = oms * oms; o3 = o2 * oms; o4 = o3 * oms; s2 = s * s; s3 = s2 * s; t4 = s2 * s2
    dzdR = add(add(add(add(scl(o4, v[0]), scl(T(4) * o3 * s, v[1])), scl(T(6) * o2 * s2, v[2])), scl(T(4) * oms * s3, v[3])), scl(t4, v[4]))
    sv = scl(T(-1), dzdR); m = length(sv)
    return dzdR, (sv[0] / m, sv[1] / m), sv


def hermite(t, T):
    t = T(t); t2 = t * t; t3 = t2 * t
    return T(2) * t3 - T(3) * t2 + T(1), t3 - T(2) * t2 + t, T(-2) * t3 + T(3) * t2, t3 - t2


def seg_acc(c, t, T):
    H00, H10, H01, H11 = hermite(t, T); dR = T(c['dR'])
    return add(add(add(scl(H00, V(T, c['a1'])), scl(H10, scl(dR, V(T, c['j1'])))), scl(H01, V(T, c['a2']))), scl(H11, scl(dR, V(T, c['j2']))))


def seg_umag(c, t, T):
    u1, u2 = V(T, c['u1']), V(T, c['u2']); n1, n2 = length(u1), length(u2); dR = T(c['dR'])
    if c['uHerm']:
        H00, H10, H01, H11 = hermite(t, T)
        a1, a2 = V(T, c['a1']), V(T, c['a2'])
        q1, q2 = T(1) + n1 * n1, T(1) + n2 * n2
        d1 = -(q1 * np.sqrt(q1)) * dot(u1, a1) / max(n1, T(1e-20))
        d2 = -(q2 * np.sqrt(q2)) * dot(u2, a2) / max(n2, T(1e-20))
        uT = max(T(0), H00 * n1 + H10 * (dR * d1) + H01 * n2 + H11 * (dR * d2))
    else:
        uT = n1 * (T(1) - T(t)) + n2 * T(t)
    uSq = uT * uT
    return uT / np.sqrt(T(1) + uSq), T(1) / (T(1) + uSq), uT


def field(case, i, P):
    xo = case.obs[i]
    # --- root
    if P['root'] == 'gpu':
        r1, dRSub = F32(case.dbg[i][0]), F32(case.dbg[i][1])
        idx = np.where(case.k['emitR'].astype(np.float32) == r1)[0]
        assert len(idx) == 1
        m1 = int(idx[0])
        c = seg_coef(case, m1, P)
        dR32 = F32(case.k['emitR'][m1 + 1]) - F32(case.k['emitR'][m1])
        t = float(dRSub) / float(dR32)
        # snap to the float32 s the GPU used
        cand = [np.nextafter(F32(t), F32(2) * np.sign(k) if k else F32(t)) for k in (0,)]
        best = F32(t)
        for k in range(-4, 5):
            s = F32(t)
            for _ in range(abs(k)): s = np.nextafter(s, F32(10) if k > 0 else F32(-10))
            if F32(s * dR32) == dRSub: best = s; break
        t = float(best)
    else:
        m1 = int(np.where(case.k['emitR'].astype(np.float32) == F32(case.dbg[i][0]))[0][0])
        t = None
        for mm in (m1, m1 - 1, m1 + 1):
            c = seg_coef(case, mm, dict(P, coef='f64'))
            def f(s):
                dz = seg_pos(c, s, F64)
                d = (xo[0] - c['pos64'][0] - dz[0], xo[1] - c['pos64'][1] - dz[1])
                return c['r1_64'] + s * c['dR64'] - np.hypot(*d)
            f0, f1 = f(0.0), f(0.99999)
            if f0 <= 0 <= f1 or f0 >= 0 >= f1:
                lo, hi = 0.0, 0.99999
                for _ in range(80):
                    md = 0.5 * (lo + hi)
                    if (f(md) < 0) == (f0 < 0): lo = md
                    else: hi = md
                t = 0.5 * (lo + hi); m1 = mm; break
        assert t is not None
        c = seg_coef(case, m1, P)
    T = lambda st: F32 if P[st] == 'f32' else F64
    dz = seg_pos(c, t, T('pos'))
    dzdR, bHat, sv = seg_vel(c, t, T('vel'))
    bMag, omb2, uT = seg_umag(c, t, T('umag'))
    acc = seg_acc(c, t, T('acc'))
    # --- separation
    Ts = T('sep')
    d1_64 = (xo[0] - c['pos64'][0], xo[1] - c['pos64'][1])
    if Ts is F32:
        d1 = (F32(d1_64[0]), F32(d1_64[1])); dzs = V(F32, dz)
        b = sub(d1, dzs)
    else:
        b = (F64(d1_64[0] - float(dz[0])), F64(d1_64[1] - float(dz[1])))
    # --- normalisation
    Tn = T('norm'); bn = V(Tn, b)
    r2 = dot(bn, bn); rr = np.sqrt(r2); n = (bn[0] / rr, bn[1] / rr)
    # --- kappa, u
    Tk = T('kap')
    nk, bh, bMk, ombk = V(Tk, n), V(Tk, bHat), Tk(bMag), Tk(omb2)
    form = P.get('form', 'base')
    omb = ombk / (Tk(1) + bMk)
    if form == 'base':
        diff = sub(nk, bh)
        kappa = max(Tk(1e-7), omb + Tk(0.5) * bMk * dot(diff, diff))
        u = add(scl(omb, nk), scl(bMk, diff))
    elif form in ('cross', 'cross_dfsep'):
        # |n - bHat|^2 = 2 (1 - cos) = 2 s^2 / (1 + cos), s = (d x sv)/(|d||sv|), cos = (d.sv)/(|d||sv|);
        # diff in the frame of bHat: diff = (cos - 1) bHat + s bHat_perp
        dd = V(Tk, b) if form == 'cross' else (F64(b[0]), F64(b[1]))
        svk = V(Tk, sv)
        if form == 'cross_dfsep':
            # cross product formed exactly (double-float) from the float32 inputs, rounded to Tk
            cr = Tk(float(svk[0]) * float(dd[1]) - float(svk[1]) * float(dd[0]))
        else:
            cr = cross(svk, dd)
        ld, ls = length(dd), length(svk)
        sn = cr / (ld * ls); cs = dot(dd, svk) / (ld * ls)
        q = Tk(2) * sn * sn / (Tk(1) + cs)
        kappa = max(Tk(1e-7), omb + Tk(0.5) * bMk * q)
        # diff = n - bHat = (cs - 1) bHat + sn perp(bHat), with cs - 1 = -q/2
        perp = (-bh[1], bh[0])
        diff = add(scl(-Tk(0.5) * q, bh), scl(sn, perp))
        u = add(scl(omb, nk), scl(bMk, diff))
    # --- field
    Tv = T('fvel'); rSoft2 = Tv(r2); rSoft = np.sqrt(rSoft2)
    k3 = Tv(kappa) * Tv(kappa) * Tv(kappa)
    eVel = scl(Tv(omb2) / (k3 * rSoft2), V(Tv, u))
    Tr = T('frad'); kr = Tr(kappa); k3r = kr * kr * kr; ur, nr, ar = V(Tr, u), V(Tr, n), V(Tr, acc)
    eRad = scl(Tr(1) / (k3r * Tr(rSoft)), sub(scl(dot(nr, ar), ur), scl(kr, ar)))
    return (float(eVel[0]) + float(eRad[0]), float(eVel[1]) + float(eRad[1])), dict(m1=m1, t=t, kappa=float(kappa), bMag=float(bMag))


STAGES = ['store', 'coef', 'pos', 'vel', 'umag', 'acc', 'sep', 'norm', 'kap', 'fvel', 'frad']
ALL32 = {s: 'f32' for s in STAGES}; ALL32['root'] = 'gpu'
ALL64 = {s: 'f64' for s in STAGES}; ALL64['root'] = 'f64'


def rel(case, P):
    E = np.array([field(case, i, P)[0] for i in range(len(case.obs))])
    return np.sqrt(((E - case.exact) ** 2).sum() / (case.exact ** 2).sum()), E


if __name__ == '__main__':
    data = json.load(open(sys.argv[1]))
    out = []
    for cd in data['cases']:
        case = Case(cd)
        g = 1 / np.sqrt(1 - cd['beta'] ** 2)
        gpu = np.sqrt(((case.gpu - case.exact) ** 2).sum() / (case.exact ** 2).sum())
        e32, E32 = rel(case, ALL32)
        nE = np.sqrt((case.exact ** 2).sum())
        contrib = lambda P: np.sqrt(((rel(case, P)[1] - E32) ** 2).sum()) / nE
        fid = np.sqrt(((E32 - case.gpu) ** 2).sum() / (case.exact ** 2).sum())
        e64, _ = rel(case, ALL64)
        row = dict(beta=cd['beta'], gamma=g, M=cd['M'], obs=cd.get('obsKind', ''), gpu=gpu, emu32=e32, fidelity=fid, all64=e64, one64={}, allbut32={}, forms={})
        for st in STAGES + ['root']:
            P = dict(ALL32); P[st] = 'f64' if st != 'root' else 'f64'
            row['one64'][st] = rel(case, P)[0]
            row.setdefault('contrib', {})[st] = contrib(P)
            Q = dict(ALL64); Q[st] = 'f32' if st != 'root' else 'gpu'
            row['allbut32'][st] = rel(case, Q)[0]
        for form in ('cross', 'cross_dfsep'):
            row['forms'][form] = rel(case, dict(ALL32, form=form))[0]
        out.append(row)
        print(f"gamma {g:.1f} M {cd['M']} {cd.get('obsKind','')}: GPU {gpu:.2e}  emu f32 {e32:.2e} (emu vs GPU {fid:.1e})  all f64 {e64:.2e}")
        print('  one stage -> f64 :', '  '.join(f"{k} {v:.2e}" for k, v in row['one64'].items()))
        print('  |dE| of stage    :', '  '.join(f"{k} {v:.2e}" for k, v in row['contrib'].items()))
        print('  only stage f32   :', '  '.join(f"{k} {v:.2e}" for k, v in row['allbut32'].items()))
        print('  forms (f32)      :', '  '.join(f"{k} {v:.2e}" for k, v in row['forms'].items()))
    json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else 'results_emu.json', 'w'), indent=1)
