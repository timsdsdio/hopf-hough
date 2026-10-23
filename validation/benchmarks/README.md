# Head-to-head with PyCharge

The same problems, the same grids and the same exact reference, run with:

- **PyCharge 2.0.1** (JAX, float64): its default solver, and the cheapest solver settings that converge (found by `tune_pycharge.py` and checked on the full grid);
- **the hopf-hough method ported to JAX** (`hh_jax.py`, float64): same framework, same hardware as PyCharge, so the comparison is between algorithms rather than languages;
- **the hopf-hough WebGL2 kernel** (float32 with a double-float root function): `webgl_same_grid.mjs` here, `examples/benchmark.html` on a real GPU.

Problem: charges on circular orbits of radius 0.85 (c = 2.35), field E on a 256 × 256 grid over [−3, 3]² at t = 2.5, relative L2 error against the exact field, points within 0.05 of the orbit excluded. Timings are medians of three runs after JIT compilation.

Machine for the numbers below: 2-core x86-64 cloud container, 7 GB RAM, no GPU (`jax` 0.12, CPU). The WebGL row used SwiftShader, a software renderer, i.e. the same CPU.

## Speed at matched accuracy (float64, CPU)

One source, 65,536 points.

| β (γ) | PyCharge, default | PyCharge, cheapest converged | hopf-hough JAX, M = 2048 | hopf-hough JAX error | hopf-hough WebGL (SwiftShader), M = 2048 |
|---|---|---|---|---|---|
| 0.9 (2.3) | 0.27 s | 0.040 s (8 + 8 steps) | 0.038 s | 1.1e-6 | 0.092 s, 1.2e-6 |
| 0.99 (7.1) | 0.28 s | 0.066 s (32 + 8) | 0.061 s | 6.2e-6 | 0.091 s, 9.5e-6 |
| 0.999 (22.4) | 0.53 s | 0.18 s (128 + 16) | 0.034 s | 2.4e-5 | 0.088 s, 3.3e-5 |
| 0.9999 (70.7) | 0.33 s | 0.34 s (256 + 32) | 0.039 s | 2.9e-5 | 0.090 s, 1.8e-5 |
| 16 sources, β = 0.9 | 5.1 s | 0.61 s | 0.66 s | 1.0e-6 | 0.99 s, 1.1e-6 |

PyCharge evaluates the analytic trajectory, so when it converges it is exact (errors 1e-11 to 1e-14). Its fixed-point iteration slows as β → 1 (the contraction factor approaches 1), so the iteration budget it needs grows with γ; fewer iterations do not degrade gracefully, they fail (errors of order 1 or larger). The hopf-hough method interpolates a sampled history (errors 1e-6 to 3e-5 at M = 2048) and its cost does not depend on γ: a bounded search over the stored knots plus a few Newton steps. Raw data: `results_python.json`, `results_pycharge_tuned.json`, `results_tuning.json`, `results_webgl.json`.

**Reading:** a tie at γ ≲ 7 and for 16 sources; the hopf-hough method is 5× faster at γ = 22 and 9× faster at γ = 71, at an accuracy of a few times 1e-5 instead of machine precision.

## Accuracy in float32 (what GPUs run at full speed)

Consumer GPUs run float64 at 1/32–1/64 of float32 speed, and Apple GPUs have no float64 at all, while PyCharge requires float64 (`jax_enable_x64`). The pulse-crossing sweep (256 observers across the synchrotron pulse, rounded to float32, compared with the exact field at the rounded points; `sweep_compare.py`):

| γ | PyCharge, float32 | hopf-hough WebGL, float32 + double-float root, M = 512 | PyCharge, float64 | hopf-hough JAX, float64, M = 2048 |
|---|---|---|---|---|
| 7.1 | 0.027% | 0.0023% | 1e-10 % | 1.5e-4 % |
| 22.4 | 0.86% | 0.0095% | 3e-9 % | 7.5e-4 % |
| 70.7 | 27% | 0.069% | 1e-7 % | 5.3e-3 % |

(WebGL numbers from `results/tables.json`.) On a whole grid, which the smooth bulk of the field dominates, PyCharge in float32 stays within 0.11% (`results_pycharge_float32.json`); the difference shows where the field is sharp.

## What differs besides speed

- PyCharge needs the trajectory as a differentiable JAX function and supports 3D and self-consistent dynamics. hopf-hough works from any sampled history (analytic, tabulated or recorded), in 2D, with prescribed motion.
- hopf-hough reports where its answer is not trustworthy (the validity indicator); PyCharge, being exact on analytic input, does not need to.

## Reproduce

```bash
pip install pycharge --break-system-packages   # PyCharge 2.x, JAX
python3 run_benchmark.py --grid 256             # PyCharge default + JAX port
python3 tune_pycharge.py && python3 time_tuned.py 256
python3 sweep_compare.py --dtype 64 && python3 sweep_compare.py --dtype 32
node webgl_same_grid.mjs 256                    # after `npm run build`
```

On a machine with a GPU, open `examples/benchmark.html` (or the hosted copy on the project site) and press **Run**.
