# Grouped stage swaps (float32 -> float64) in the field emulation: what fixing several float32 directions together buys.
import json, sys, numpy as np
from emu_field import Case, rel, ALL32
G = {'sep+norm': ['sep', 'norm'], 'coef+vel': ['coef', 'vel'], 'sep+norm+coef+vel': ['sep', 'norm', 'coef', 'vel'],
     '+acc+frad': ['sep', 'norm', 'coef', 'vel', 'acc', 'frad'], '+root': ['sep', 'norm', 'coef', 'vel', 'acc', 'frad', 'root'],
     'root only': ['root'], 'all but store': ['coef','pos','vel','umag','acc','sep','norm','kap','fvel','frad','root']}
out = []
for cd in json.load(open(sys.argv[1]))['cases']:
    case = Case(cd); g = 1 / np.sqrt(1 - cd['beta'] ** 2); row = {'gamma': g, 'M': cd['M'], 'obs': cd['obsKind']}
    row['all32'] = rel(case, ALL32)[0]
    for name, st in G.items():
        P = dict(ALL32); [P.__setitem__(s, 'f64') for s in st]; row[name] = rel(case, P)[0]
    out.append(row); print(' '.join(f"{k} {v:.2e}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
json.dump(out, open('results_groups.json', 'w'), indent=1)
