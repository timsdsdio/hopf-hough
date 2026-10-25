#!/usr/bin/env python3
"""float32 emulation of the whole v6f known-path pipeline (quintic segment, chord-deficit constants, incremental Newton,
field at the root) and of the candidate fix: the transverse angle between n, beta_hat and a is formed from cross/dot
products with the double-float knot separation d1, so float32 rounding of the directions no longer enters.
  root : cr = d1 x dz in incRoot from per-segment constants chi_k = d1 x P_k (P_k quintic position control points)
  kap  : sin(n, beta_hat) = (d1 x dzdR - dz x dzdR) / (|dzdR| L) with d1 x dzdR = sum B_k psi_k, psi_k = d1 x v_k;
         1 - cos = sin^2 / (1 + cos); n - beta_hat rebuilt in the beta_hat frame
  na   : n.a from exact d1.a1, d1.a2 (and dz.a, j.d in float32)
Exact products (twoProd in the shader) are emulated by float64 products of the float32 / double-float inputs, rounded once.
Usage: python3 emu_fix.py exp_known.json"""
import json, sys
import numpy as np
from emu_field import Case, F32, F64, V, add, sub, scl, dot, length, cross, alpha_quint

f = F32


def knot(case, m):
    k = case.k
    return dict(phi=(f(k['emitX'][m]), f(k['emitY'][m])), plo=(f(k['emitXlo'][m]), f(k['emitYlo'][m])),
                u=(f(k['emitUx'][m]), f(k['emitUy'][m])), acc=(f(k['emitBdx'][m]), f(k['emitBdy'][m])), r=f(k['emitR'][m]),
                j=(f(k['emitJx'][m]), f(k['emitJy'][m])))


def pos64(kn): return (float(kn['phi'][0]) + float(kn['plo'][0]), float(kn['phi'][1]) + float(kn['plo'][1]))


def coef(case, m1):
    k1, k2 = knot(case, m1), knot(case, m1 + 1)
    bm = f(case.bm)
    dR = max(f(1e-6), k2['r'] - k1['r'])
    dChord = add(sub(k2['phi'], k1['phi']), sub(k2['plo'], k1['plo']))
    p1, p2 = pos64(k1), pos64(k2)
    dChord64 = (p2[0] - p1[0], p2[1] - p1[1])     # double-float chord (twoSum of the hi parts + lo parts)
    vSec = (dChord[0] / dR, dChord[1] / dR)
    assert length(vSec) <= bm
    u1, u2 = k1['u'], k2['u']
    g1, g2 = np.sqrt(f(1) + dot(u1, u1)), np.sqrt(f(1) + dot(u2, u2))
    b1, b2 = (u1[0] / g1, u1[1] / g1), (u2[0] / g2, u2[1] / g2)
    assert length(b1) <= bm and length(b2) <= bm
    a1, a2 = k1['acc'], k2['acc']
    w1, w2 = scl(f(0.25) * dR, a1), scl(f(0.25) * dR, a2)
    Q0, Q4 = scl(f(-1), b1), scl(f(-1), b2)
    Q1, Q3 = add(Q0, w1), sub(Q4, w2)
    Q2 = sub(sub(sub(sub(scl(f(5), vSec), Q0), Q1), Q3), Q4)
    Qs = (Q0, Q1, Q2, Q3, Q4)
    aq = f(alpha_quint(vSec, [sub(q, vSec) for q in Qs], bm))
    v = [add(vSec, scl(aq, sub(q, vSec))) for q in Qs]
    n1, n2 = length(u1), length(u2)
    um = f(0.5) * (n1 + n2); bmm = um / np.sqrt(f(1) + um * um)
    ap = max(abs(cross(b1, a1)) / max(length(b1), f(1e-7)), abs(cross(b2, a2)) / max(length(b2), f(1e-7)))
    uHerm = aq >= 1 and dR * np.sqrt(f(1) + um * um) * ap / max(f(0.04), bmm * bmm) < 1
    # chord-deficit constants (F32N)
    vv = min(np.hypot(*dChord64) / max(1e-6, float(k2['r']) - float(k1['r'])), float(bm))
    Dv = f((1 - vv) * (1 + vv))
    D0, D2 = f(1) / (f(1) + dot(u1, u1)), f(1) / (f(1) + dot(u2, u2))
    e0, e2, e02 = add(vSec, b1), add(vSec, b2), sub(b2, b1)
    D1 = D0 + f(2) * dot(b1, w1) - dot(w1, w1)
    D3 = D2 - f(2) * dot(b2, w2) - dot(w2, w2)
    g1v, g3v, h03 = sub(e0, w1), add(e2, w2), add(e02, w2)
    h13, h14 = add(h03, w1), add(e02, w1)
    qa = (dot(e0, e0), dot(g1v, g1v), dot(g3v, g3v), dot(e2, e2))
    qb = (dot(w1, w1), dot(h03, h03), dot(e02, e02), dot(h13, h13))
    qc = (dot(h14, h14), dot(w2, w2))
    return dict(k1=k1, k2=k2, dR=dR, dChord=dChord, dChord64=dChord64, vSec=vSec, b1=b1, b2=b2, g1=g1, g2=g2, w1=w1, w2=w2,
                aq=aq, v=v, uHerm=uHerm, Dv=Dv, D0=D0, D1=D1, D2=D2, D3=D3, qa=qa, qb=qb, qc=qc, u1=u1, u2=u2, a1=a1, a2=a2,
                j1=k1['j'], j2=k2['j'], p1=pos64(k1))


def chord_deficit(c, s):
    o = f(1) - s; o2 = o * o; o4 = o2 * o2; s2 = s * s
    I0 = (f(1) + o + o2 + o2 * o + o4) * f(0.2); I1 = I0 - o4
    I2 = s2 * (f(10) * o2 + f(5) * s * o + s2) * f(0.2); I3 = s2 * s * (f(5) * o + s) * f(0.2); I4 = s2 * s2 * f(0.2)
    al = c['aq']
    a0, a1, a3, a4 = al * (I0 - I2), al * (I1 - I2), al * (I3 - I2), al * (I4 - I2)
    av = f(1) - a0 - a1 - a3 - a4
    qa, qb, qc = c['qa'], c['qb'], c['qc']
    return (av * c['Dv'] + a0 * c['D0'] + a1 * c['D1'] + a3 * c['D3'] + a4 * c['D2']
            + av * (a0 * qa[0] + a1 * qa[1] + a3 * qa[2] + a4 * qa[3])
            + a0 * (a1 * qb[0] + a3 * qb[1] + a4 * qb[2]) + a1 * (a3 * qb[3] + a4 * qc[0]) + a3 * a4 * qc[1])


def bern(s):
    o = f(1) - s; o2 = o * o; o3 = o2 * o; o4 = o3 * o; s2 = s * s; s3 = s2 * s; s4 = s2 * s2
    B4 = (o4, f(4) * o3 * s, f(6) * o2 * s2, f(4) * o * s3, s4)
    B5 = (f(5) * o4 * s, f(10) * o3 * s2, f(10) * o2 * s3, f(5) * o * s4, s4 * s)
    return B4, B5


def evalseg(c, s):
    s = f(s); dR = c['dR']; v = c['v']
    B4, B5 = bern(s)
    P1 = scl(f(0.2) * dR, v[0]); P2 = add(P1, scl(f(0.2) * dR, v[1])); P3 = add(P2, scl(f(0.2) * dR, v[2]))
    P4 = sub(c['dChord'], scl(f(0.2) * dR, v[4]))
    dz = add(add(add(add(scl(B5[0], P1), scl(B5[1], P2)), scl(B5[2], P3)), scl(B5[3], P4)), scl(B5[4], c['dChord']))
    dzdR = add(add(add(add(scl(B4[0], v[0]), scl(B4[1], v[1])), scl(B4[2], v[2])), scl(B4[3], v[3])), scl(B4[4], v[4]))
    t2 = s * s; t3 = t2 * s
    H = (f(2) * t3 - f(3) * t2 + f(1), t3 - f(2) * t2 + s, f(-2) * t3 + f(3) * t2, t3 - t2)
    acc = add(add(add(scl(H[0], c['a1']), scl(H[1], scl(dR, c['j1']))), scl(H[2], c['a2'])), scl(H[3], scl(dR, c['j2'])))
    n1, n2 = length(c['u1']), length(c['u2'])
    if c['uHerm']:
        q1, q2 = f(1) + n1 * n1, f(1) + n2 * n2
        d1 = -(q1 * np.sqrt(q1)) * dot(c['u1'], c['a1']) / max(n1, f(1e-20))
        d2 = -(q2 * np.sqrt(q2)) * dot(c['u2'], c['a2']) / max(n2, f(1e-20))
        uT = max(f(0), H[0] * n1 + H[1] * (dR * d1) + H[2] * n2 + H[3] * (dR * d2))
    else:
        uT = n1 * (f(1) - s) + n2 * s
    sv = scl(f(-1), dzdR); svm = length(sv); bHat = (sv[0] / svm, sv[1] / svm)
    uSq = uT * uT
    return dict(s=s, dz=dz, dRSub=s * dR, dzdR=dzdR, acc=acc, bHat=bHat, bMag=uT / np.sqrt(f(1) + uSq), omb2=f(1) / (f(1) + uSq), H=H, B4=B4, B5=B5)


def crossX(a64, w):   # exact cross of a double-float 2-vector (as float64) with a float32 / double-float 2-vector, rounded once
    return f(a64[0] * float(w[1]) - a64[1] * float(w[0]))


def dotX(a64, w):
    return f(a64[0] * float(w[0]) + a64[1] * float(w[1]))


def fix_consts(c, d1_64, d1):
    dR = c['dR']
    psi_b1 = crossX(d1_64, c['u1']) / c['g1']; psi_b2 = crossX(d1_64, c['u2']) / c['g2']
    Xch = crossX(d1_64, c['dChord64']); psi_v = Xch / dR
    psi_w1, psi_w2 = cross(d1, c['w1']), cross(d1, c['w2'])
    q0, q4 = -psi_b1, -psi_b2
    q1, q3 = q0 + psi_w1, q4 - psi_w2
    q2 = f(5) * psi_v - q0 - q1 - q3 - q4
    psi = [psi_v + c['aq'] * (q - psi_v) for q in (q0, q1, q2, q3, q4)]
    h = f(0.2) * dR
    chi1 = h * psi[0]; chi2 = chi1 + h * psi[1]; chi3 = chi2 + h * psi[2]; chi4 = Xch - h * psi[4]
    return dict(psi=psi, chi=(chi1, chi2, chi3, chi4, Xch), Y1=dotX(d1_64, c['a1']), Y2=dotX(d1_64, c['a2']))


def inc_root(d1, L1, f1, c, g, fx, use_cr):
    dz = g['dz']; dR = g['dRSub']
    l = length(dz); b = sub(d1, dz); L = length(b)
    t1 = dR * chord_deficit(c, g['s']) / (f(1) + l / dR) if dR > 0 else f(0)
    dd = dot(d1, dz); S = l + L1 + L
    if use_cr:
        B5 = g['B5']; ch = fx['chi']
        cr = B5[0] * ch[0] + B5[1] * ch[1] + B5[2] * ch[2] + B5[3] * ch[3] + B5[4] * ch[4]
    else:
        cr = d1[0] * dz[1] - d1[1] * dz[0]
    slack = (f(2) * cr * cr / ((l * L1 - dd) * S) if dd <= 0 else f(2) * (l * L1 + dd) / S) if l > 0 else f(0)
    return f1 + (t1 + slack), b, L


def solve(case, i, c, m1, use_cr):
    xo = case.obs[i]
    def knotf(m):
        kn = knot(case, m); p = pos64(kn); d = (xo[0] - p[0], xo[1] - p[1])
        return f(float(kn['r']) - np.hypot(*d)), d
    fLo, d1_64 = knotf(m1); fHi, _ = knotf(m1 + 1)
    d1 = (f(d1_64[0]), f(d1_64[1])); L1 = length(d1)
    fx = fix_consts(c, d1_64, d1)
    kfl = f(0.5) * (f(1) - f(case.bm))
    sLow, sHigh = f(0), f(0.9999)
    sc = min(max(fLo / (fLo - fHi), f(0)), f(0.9999))
    for n in range(12):
        g = evalseg(c, sc)
        fv, b, L = inc_root(d1, L1, fLo, c, g, fx, use_cr)
        if fv == 0: break
        if fv < 0: sLow = sc
        else: sHigh = sc
        nrm = (b[0] / L, b[1] / L)
        dfds = max(f(1e-12), c['dR'] * max(kfl, f(1) + dot(nrm, g['dzdR'])))
        cand = sc - fv / dfds
        nxt = cand if sLow <= cand <= sHigh else f(0.5) * (sLow + sHigh)
        done = abs(nxt - sc) < 1e-7 or (sHigh - sLow) < 1e-7
        sc = f(nxt)
        if done: break
    return sc, d1_64, d1, L1, fLo, fx


def field_at(c, g, d1, d1_64, L1, fLo, fx, mode):
    _, b, L = inc_root(d1, L1, fLo, c, g, fx, False)
    r2 = dot(b, b); rr = np.sqrt(r2)
    n = (b[0] / rr, b[1] / rr)
    bh, bM, omb2 = g['bHat'], g['bMag'], g['omb2']
    omb = omb2 / (f(1) + bM)
    acc = g['acc']
    if 'kap' in mode:
        B4 = g['B4']; psi = fx['psi']
        Cv = B4[0] * psi[0] + B4[1] * psi[1] + B4[2] * psi[2] + B4[3] * psi[3] + B4[4] * psi[4]
        dz, dzdR = g['dz'], g['dzdR']
        sn = (Cv - cross(dz, dzdR)) / (length(dzdR) * rr)
        cs = dot(bh, n)
        ang = sn * sn / (f(1) + cs) if cs > 0 else f(1) - cs
        perp = (-bh[1], bh[0])
        diff = add(scl(-ang, bh), scl(sn, perp))
        nn = add(bh, diff)
        kappa = max(f(1e-7), omb + bM * ang)
        u = add(scl(omb, nn), scl(bM, diff))
    else:
        diff = sub(n, bh)
        kappa = max(f(1e-7), omb + f(0.5) * bM * dot(diff, diff))
        u = add(scl(omb, n), scl(bM, diff))
    if 'na' in mode:
        H, dz, dR = g['H'], g['dz'], c['dR']
        ad = H[0] * (fx['Y1'] - dot(c['a1'], dz)) + H[2] * (fx['Y2'] - dot(c['a2'], dz)) + dR * (H[1] * dot(c['j1'], b) + H[3] * dot(c['j2'], b))
        na = ad / rr
    else:
        na = dot(n, acc)
    k3 = kappa * kappa * kappa
    eVel = scl(omb2 / (k3 * r2), u)
    eRad = scl(f(1) / (k3 * rr), sub(scl(na, u), scl(kappa, acc)))
    return (float(eVel[0] + eRad[0]), float(eVel[1] + eRad[1]))


MODES = {'base': (False, ''), 'root': (True, ''), 'kap': (False, 'kap'), 'kap+na': (False, 'kap na'), 'root+kap': (True, 'kap'),
         'full': (True, 'kap na')}


def run(case):
    res = {m: [] for m in MODES}; tdiff = 0
    for i in range(len(case.obs)):
        m1 = int(np.where(case.k['emitR'].astype(np.float32) == f(case.dbg[i][0]))[0][0])
        c = coef(case, m1)
        sols = {}
        for ucr in (False, True):
            sols[ucr] = solve(case, i, c, m1, ucr)
        # emulated root vs GPU root
        tdiff = max(tdiff, abs(float(sols[False][0] * c['dR']) - float(case.dbg[i][1])))
        for m, (ucr, fm) in MODES.items():
            sc, d1_64, d1, L1, fLo, fx = sols[ucr]
            res[m].append(field_at(c, evalseg(c, sc), d1, d1_64, L1, fLo, fx, fm))
    nE = np.sqrt((case.exact ** 2).sum())
    out = {m: float(np.sqrt(((np.array(v) - case.exact) ** 2).sum()) / nE) for m, v in res.items()}
    out['emu_vs_gpu'] = float(np.sqrt(((np.array(res['base']) - case.gpu) ** 2).sum()) / nE)
    out['max_dRsub_vs_gpu'] = tdiff
    return out


if __name__ == '__main__':
    rows = []
    for cd in json.load(open(sys.argv[1]))['cases']:
        case = Case(cd); g = 1 / np.sqrt(1 - cd['beta'] ** 2)
        gpu = float(np.sqrt(((case.gpu - case.exact) ** 2).sum() / (case.exact ** 2).sum()))
        r = run(case); r.update(gamma=g, M=cd['M'], obs=cd['obsKind'], gpu=gpu); rows.append(r)
        print(f"gamma {g:.1f} M {cd['M']} {cd['obsKind']}: GPU {gpu:.2e} | " + '  '.join(f"{m} {r[m]:.2e}" for m in MODES)
              + f" | emu vs GPU {r['emu_vs_gpu']:.1e}, max |dRsub - GPU| {r['max_dRsub_vs_gpu']:.1e}")
    json.dump(rows, open(sys.argv[2] if len(sys.argv) > 2 else 'results_fix_emu.json', 'w'), indent=1)
