# Per-segment certificate activity on the actual benchmark shell data (Float64 and Float32-stored knots).
import json, numpy as np, sys; sys.path.insert(0, '.')
from certs import *
D = json.load(open('knots.json')); bm = np.float32(0.99995)
for key, d in D.items():
    for prec in ['64', '32']:
        sfx = '64' if prec == '64' else ''
        R = np.array(d['R'+sfx]); X = np.array(d['X'+sfx]); Y = np.array(d['Y'+sfx])
        Ux = np.array(d['Ux'+sfx]); Uy = np.array(d['Uy'+sfx])
        if prec == '32':
            f = np.float32; R, X, Y, Ux, Uy = [a.astype(f) for a in (R, X, Y, Ux, Uy)]
        if d['vm'] == 2:
            g = np.sqrt(1 + Ux*Ux + Uy*Uy); Bx, By = Ux/g, Uy/g
        else: Bx, By = Ux, Uy
        nP = nS = nSh = nSup = nSup1 = 0; worst = 0
        for m in range(d['M']-1):
            dR = R[m+1]-R[m]
            v = np.array([(X[m+1]-X[m])/dR, (Y[m+1]-Y[m])/dR], dtype=np.float64)
            m0 = -np.array([Bx[m], By[m]], dtype=np.float64); m1 = -np.array([Bx[m+1], By[m+1]], dtype=np.float64)
            for arr in (v, m0, m1):
                n = np.linalg.norm(arr)
                if n > bm: arr *= bm/n            # same pre-clamps as the shader
            ms = maxspeed(v, m0, m1); worst = max(worst, ms)
            nSup += ms > bm; nSup1 += ms >= 1
            nP += alpha_paper(v, m0, m1, bm) < 1
            nS += alpha_sq(v, m0, m1, bm) < 1
            nSh += alpha_sharp(v, m0, m1, bm) < 1
        print(f"{key:16s} F{prec}: segs={d['M']-1:5d} paper-fires={nP:4d} sq-fires={nS:4d} sharp-fires={nSh:4d} | raw cubic max|z'|>bmax: {nSup:3d}, >=1: {nSup1:3d}, worst={worst:.7f}")
