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
| 0.9 (2.3) | 0.27 s | 0.040 s (8 + 8 steps) | 0.038 s | 1.1e-6 | 0.113 s, 1.1e-6 |
| 0.99 (7.1) | 0.28 s | 0.066 s (32 + 8) | 0.061 s | 5.9e-6 | 0.129 s, 6.2e-6 |
| 0.999 (22.4) | 0.53 s | 0.18 s (128 + 16) | 0.034 s | 3.7e-5 | 0.107 s, 3.9e-5 |
| 0.9999 (70.7) | 0.33 s | 0.34 s (256 + 32) | 0.039 s | 2.9e-5 | 0.111 s, 2.6e-5 |
| 16 sources, β = 0.9 | 5.1 s | 0.61 s | 0.66 s | 9.7e-7 | 1.09 s, 1.0e-6 |

PyCharge evaluates the analytic trajectory, so when it converges it is exact (errors 1e-11 to 1e-14). Its fixed-point iteration slows as β → 1 (the contraction factor approaches 1), so the iteration budget it needs grows with γ; fewer iterations do not degrade gracefully, they fail (errors of order 1 or larger). The hopf-hough method interpolates a sampled history (errors 1e-6 to 3e-5 at M = 2048) and its cost does not depend on γ: a bounded search over the stored knots plus a few Newton steps. Raw data: `results_python.json`, `results_pycharge_tuned.json`, `results_tuning.json`, `results_webgl.json`.

**Newton fix (after v6.0.0).** The JAX port's safeguarded Newton used an open bracket with no hold at f = 0, so a step that landed exactly on the root was rejected and the iteration fell back to bisection (found by the 3D baseline review). With the fix (closed bracket, hold at f = 0) the hopf-hough JAX error column above was re-measured: most cells fall slightly, but β = 0.999 rises from 2.4e-5 to 3.7e-5 at M = 2048 (and 6.3e-4 → 9.0e-4 at M = 512), because the unconverged roots had partly cancelled interpolation error. Pulse-sweep errors fall (table below). JAX timings did not change beyond run-to-run noise (0.037–0.047 s for one source), so the JAX and PyCharge timing columns are from the original interleaved run. The WebGL column was re-run with the current kernel (κ-form radiation term + Newton fix); its software-render times came out 15–25% higher than in the v6.0.0 run (0.088–0.092 s), and its grid errors moved as the JAX ones did (β = 0.999 and 0.9999 at M = 2048 rise from 3.3e-5 and 1.8e-5; the others fall). Software-render timings say little about GPUs; see the M1 Max section. The numbers in preprint v6 are those of tag v6.0.0.

**Reading:** a tie at γ ≲ 7 and for 16 sources; the hopf-hough method is 5× faster at γ = 22 and 9× faster at γ = 71, at an accuracy of a few times 1e-5 instead of machine precision.

## On a real GPU (Apple M1 Max)

`examples/benchmark.html` in Chrome 154 on an M1 Max (ANGLE → Metal), M = 2048, same problems and grid; `results_webgl_m1max.json`. Times include uploading the 65,536 observer coordinates and reading the field back.

| case | M1 Max GPU | error | PyCharge, cheapest converged (2-core CPU) | hopf-hough JAX (2-core CPU) |
|---|---|---|---|---|
| β = 0.9 | 6.6 ms | 1.2e-6 | 40 ms | 38 ms |
| β = 0.99 | 5.9 ms | 7.0e-6 | 66 ms | 61 ms |
| β = 0.999 | 5.1 ms | 3.8e-5 | 180 ms | 34 ms |
| β = 0.9999 | 7.4 ms | 3.0e-5 | 340 ms | 39 ms |
| 16 sources | 20 ms | 1.1e-6 | 610 ms | 660 ms |
| 64 sources | 71 ms | 9.6e-7 | — | — |
| 256 sources | 270 ms | 8.0e-7 | — | — |

The GPU and CPU columns come from different machines, so the ratio is indicative only; the CPU runs used a 2-core cloud container, not the M1 Max's CPU. Single-source times are dominated by fixed per-call cost (upload and readback). The marginal cost is about 1 ms per source per 65,536 points (≈ 6.5 × 10⁷ point-source evaluations per second). The errors match the software-renderer run, which confirms that the double-float root function survives the Metal compiler (see the shader comment on `BAR()` and `examples/precision-probe.html`).

## Accuracy in float32 (what GPUs run at full speed)

Consumer GPUs run float64 at 1/32–1/64 of float32 speed, and Apple GPUs have no float64 at all, while PyCharge requires float64 (`jax_enable_x64`). The pulse-crossing sweep (256 observers across the synchrotron pulse, rounded to float32, compared with the exact field at the rounded points; `sweep_compare.py`):

| γ | PyCharge, float32 | hopf-hough WebGL, float32 + double-float root, M = 512 | PyCharge, float64 | hopf-hough JAX, float64, M = 2048 |
|---|---|---|---|---|
| 7.1 | 0.027% | 0.0022% | 1e-10 % | 1.4e-4 % |
| 22.4 | 0.86% | 0.0077% | 3e-9 % | 4.5e-4 % |
| 70.7 | 27% | 0.036% | 1e-7 % | 1.5e-3 % |

(WebGL numbers from `results/tables.json`, current kernel with the κ-form radiation term and the Newton fix; the v6.0.0 kernel gave 0.0023%, 0.0095% and 0.069%. JAX numbers after the Newton fix; before it they were 1.5e-4, 7.5e-4 and 5.3e-3 %.) On a whole grid, which the smooth bulk of the field dominates, PyCharge in float32 stays within 0.11% (`results_pycharge_float32.json`); the difference shows where the field is sharp.

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
