#!/usr/bin/env python3
"""ALU operations per pixel-source for the 2D app's v6 and v6f kernels: hand count of the GLSL in web/index.html
(one op per float add/sub/mul/div/sqrt, one per BAR() integer XOR, one per mantissa AND; texel fetches not counted),
times the per-pixel evaluation counts MEASURED on the GPU with the COUNT_ITERS builds (results_counts.json).
Shared parts (segCoef, evalSeg, kernel + validity indicator) are rough estimates, identical for both kernels (taken
from a hand count of the shared code). Usage: python3 opcount_2d.py"""
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
print('ops: ' + ', '.join(f'{a} {b}' for a, b in OPS.items()))
c = json.load(open(os.path.join(HERE, 'results_counts.json')))


def cost(kern, cnt, fast, known):
    if kern == 'v6':
        r = OPS['root_fast'] if fast else OPS['root_full']
        return (cnt['K'] * r + cnt['F'] * OPS['knot_f32_v6'] + cnt['N'] * (r + OPS['evalSeg'] + OPS['newton'])
                + OPS['segCoef'] + OPS['evalSeg'] + r + OPS['kernel'])
    rk = OPS['root_fast'] if fast else OPS['root_full']
    inc = OPS['inc_quint'] if known else OPS['inc_cubic']
    return (cnt['K'] * rk + cnt['F'] * OPS['knot_f32_v6f'] + cnt['N'] * (inc + OPS['evalSeg'] + OPS['newton'])
            + OPS['segCoef'] + OPS['segCoef_f32n_extra'] + OPS['evalSeg'] + inc + OPS['kernel'])


rows = []
for r in c['counts']:
    known = r['mode'] == 'known'
    a, b = cost('v6', r['v6'], False, known), cost('v6f', r['v6f'], False, known)
    rows.append(dict(case=f"sweep gamma {1 / (1 - r['beta'] ** 2) ** 0.5:.0f} M{r['M']} {r['mode']} (readback)", v6=a, v6f=b))
for r in c['scene']:
    fast = r['path'] == 'display'
    a, b = cost('v6', r['v6'], fast, r['known']), cost('v6f', r['v6f'], fast, r['known'])
    rows.append(dict(case=f"32 sources M{r['M']} {'known' if r['known'] else 'live'} {r['path']}", v6=a, v6f=b))
for x in rows:
    print(f"{x['case']}: v6 {x['v6']:.0f} ops, v6f {x['v6f']:.0f} ops ({x['v6f'] / x['v6']:.2f}x)")
json.dump(dict(ops=OPS, rows=rows), open(os.path.join(HERE, 'results_opcount.json'), 'w'), indent=1)
