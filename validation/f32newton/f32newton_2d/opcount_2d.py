#!/usr/bin/env python3
"""ALU operations per pixel-source for the 2D app's v6 and v6f kernels: hand count of the GLSL in web/index.html
(one op per float add/sub/mul/div/sqrt, one per BAR() integer XOR, one per mantissa AND; texel fetches not counted),
times the per-pixel evaluation counts MEASURED on the GPU with the COUNT_ITERS builds (results_counts.json).
Shared parts (segCoef, evalSeg, kernel + validity indicator) are rough estimates, identical for both kernels (taken
from a hand count of the shared code).
Since v6.2 review: also counts the field-floor fix (fixConsts, crossD1Dz in every incRoot, the field-side sin/1-cos/n.a
terms), which ships on and runs where |u| > 4 at a knot of the segment (gamma > 4.1). v6f rows are given with the gate
false (fix off; only the gate test is paid) and true (fix on). Measured evaluation counts exist for the readback sweep
at gamma 71/224 (gate true) and for the 32-source default scene at beta = 0.8 (gamma 1.67, gate false). For the display
path above the gate no counts were measured; the scene's display-path counts are used with the fix forced on, and
the per-step fix term is also given with the sweep's Newton count (N about 7) as an upper case.
Usage: python3 opcount_2d.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
twoSum = (6, 6, 0); quick = (3, 3, 0); split = (1, 2, 1)
def s(*a): return tuple(sum(x[i] for x in a) for i in range(3))
def k(n, a): return tuple(n * v for v in a)
tot = lambda a: sum(a)
twoProd = s((1, 1, 0), k(2, split), (8, 8, 0))
dfAdd = s(k(2, twoSum), (2, 2, 0), k(2, quick))
dfMul = s(twoProd, (4, 4, 0), quick)
dfSqrt = s((2, 1, 0), twoProd, dfAdd, (2, 2, 0), quick)
# v6 rootFn, full path: per component twoSum + 3 dfAdd; |d|: 2 dfMul + dfAdd + dfSqrt; F: twoSum + dfAdd; 4 f32 adds
root_full = s(k(2, s(twoSum, k(3, dfAdd))), k(2, dfMul), dfAdd, dfSqrt, twoSum, dfAdd, (4, 0, 0))
# v6 rootFn, display path (no df64 sqrt): per component 2 twoSum + SUB + dfAdd; R twoSum; num 3 dfMul + 2 dfAdd; ~11 f32
root_fast = s(k(2, s(k(2, twoSum), (1, 1, 0), dfAdd)), twoSum, k(3, dfMul), k(2, dfAdd), (11, 0, 0))
OPS = dict(root_full=tot(root_full), root_fast=tot(root_fast), knot_f32_v6=11, knot_f32_v6f=7, inc_cubic=57, inc_quint=87,
           newton=25, evalSeg=110, segCoef=150, segCoef_f32n_extra=30, kernel=150)

# ---- field-floor fix (web/index.html: fixConsts, crossD1Dz, crossD1V, the `X.on && g.bMag > 0.5` block of evalField).
# Same convention; min/max/clamp count 1, comparisons and selects 0, vec2 ops count per component.
ADDB = (1, 1, 0)                                   # one BAR-guarded add/sub/mul (ADD, SUB, MUL)
crossX = s(k(2, twoProd), k(7, ADDB))              # 2 twoProd + 7 guarded ops
dotX = crossX
length2 = 4                                        # 2 mul, 1 add, 1 sqrt
fix_gate = 7                                       # 2 dot + max: paid by every v6f pixel-source with velMode 2
fix_consts_common = (2 * tot(twoSum)               # double-float chord hi parts
                     + 2 * tot(k(2, ADDB))         # chord lo parts
                     + (length2 + 1 + 1)           # vm, vScale
                     + 2 * (3 + 3)                 # g1, g2 (dot, add, sqrt, div)
                     + 2 * (length2 + 4)           # bs1, bs2
                     + tot(crossX) + tot(k(4, ADDB))   # xch = d1 x chord (hi via crossX, lo via 2 MUL + SUB + ADD)
                     + 2                           # pv
                     + 2 * (tot(crossX) + 1)       # pb1, pb2
                     + 2 * 3 + 2 * tot(dotX))      # X.Y: 2 plain dots (outer knots), 2 dotX (inner knots)
fix_psi_cubic = 12
fix_psi_quint = 6 + 4 + 4 + 5 + 12 + 3 + 1 + 7    # w1, w2, q1, q3, q2, psi, psi4, h, chi
fix_inc_cubic = 16 - 3                             # crossD1Dz (cubic) instead of the plain float32 d1 x dz
fix_inc_quint = 25 - 3                             # crossD1Dz (quintic)
fix_field_common = 1 + 10 + 3 + 3 + 6 + 2 + 3 + 8 + 3   # sR, sin, cos, 1-cos, n-bhat, 1-beta, kappa, n-beta, t powers
fix_field_cubic = fix_field_common + 10 + 40       # crossD1V (cubic), n.a from Catmull-Rom weights and d1.acc_k
fix_field_quint = fix_field_common + 22 + 34       # crossD1V (quintic), n.a from Hermite weights
OPS.update(fix_gate=fix_gate, fix_consts_cubic=fix_consts_common + fix_psi_cubic, fix_consts_quint=fix_consts_common + fix_psi_quint,
           fix_inc_cubic=fix_inc_cubic, fix_inc_quint=fix_inc_quint, fix_field_cubic=fix_field_cubic, fix_field_quint=fix_field_quint)
print('ops: ' + ', '.join(f'{a} {b}' for a, b in OPS.items()))
c = json.load(open(os.path.join(HERE, 'results_counts.json')))


def cost(kern, cnt, fast, known, fix=False, n_fix=None):
    """Ops per pixel-source. v6f: fix=True adds the field-floor fix (gate true); n_fix overrides the Newton count
    used for the fix's per-step term only (upper case for the display path above the gate)."""
    if kern == 'v6':
        r = OPS['root_fast'] if fast else OPS['root_full']
        return (cnt['K'] * r + cnt['F'] * OPS['knot_f32_v6'] + cnt['N'] * (r + OPS['evalSeg'] + OPS['newton'])
                + OPS['segCoef'] + OPS['evalSeg'] + r + OPS['kernel'])
    rk = OPS['root_fast'] if fast else OPS['root_full']
    inc = OPS['inc_quint'] if known else OPS['inc_cubic']
    c = (cnt['K'] * rk + cnt['F'] * OPS['knot_f32_v6f'] + cnt['N'] * (inc + OPS['evalSeg'] + OPS['newton'])
         + OPS['segCoef'] + OPS['segCoef_f32n_extra'] + OPS['evalSeg'] + inc + OPS['kernel'] + OPS['fix_gate'])
    if fix:
        q = 'quint' if known else 'cubic'
        n = cnt['N'] if n_fix is None else n_fix
        c += OPS['fix_consts_' + q] + OPS['fix_field_' + q] + (n + 1) * OPS['fix_inc_' + q]
    return c


rows = []
for r in c['counts']:
    known = r['mode'] == 'known'
    a = cost('v6', r['v6'], False, known)
    b0, b = cost('v6f', r['v6f'], False, known), cost('v6f', r['v6f'], False, known, fix=True)
    rows.append(dict(case=f"sweep gamma {1 / (1 - r['beta'] ** 2) ** 0.5:.0f} M{r['M']} {r['mode']}", path='readback', gate='on',
                     v6=a, v6f=b, v6f_nofix=b0))
n_hi = max(r['v6f']['N'] for r in c['counts'])
for r in c['scene']:
    fast = r['path'] == 'display'
    a = cost('v6', r['v6'], fast, r['known'])
    off = cost('v6f', r['v6f'], fast, r['known'])
    on = cost('v6f', r['v6f'], fast, r['known'], fix=True)
    on_hi = cost('v6f', r['v6f'], fast, r['known'], fix=True, n_fix=n_hi)
    case = f"32 sources M{r['M']} {'known' if r['known'] else 'live'}"
    path = 'display' if fast else 'readback'
    rows.append(dict(case=case + ' (beta 0.8, counts measured)', path=path, gate='off', v6=a, v6f=off))
    rows.append(dict(case=case + ' (beta-0.8 counts, fix forced on)', path=path, gate='on', v6=a, v6f=on, v6f_upper=on_hi))
for x in rows:
    extra = f", with N = {n_hi:.1f} in the fix term {x['v6f_upper'] / x['v6']:.2f}x" if 'v6f_upper' in x else ''
    nofix = f"; without the fix {x['v6f_nofix'] / x['v6']:.2f}x" if 'v6f_nofix' in x else ''
    print(f"{x['path']:8s} gate {x['gate']:3s} {x['case']}: v6 {x['v6']:.0f} ops, v6f {x['v6f']:.0f} ops ({x['v6f'] / x['v6']:.2f}x{extra}{nofix})")
summ = {}
for path in ('readback', 'display'):
    for gate in ('off', 'on'):
        rr = [x['v6f'] / x['v6'] for x in rows if x['path'] == path and x['gate'] == gate]
        rr += [x['v6f_upper'] / x['v6'] for x in rows if x['path'] == path and x['gate'] == gate and 'v6f_upper' in x]
        summ[f'{path}_gate_{gate}'] = [round(min(rr), 3), round(max(rr), 3)]
print('v6f/v6 ranges:', summ)
json.dump(dict(ops=OPS, rows=rows, ranges=summ), open(os.path.join(HERE, 'results_opcount.json'), 'w'), indent=1)
