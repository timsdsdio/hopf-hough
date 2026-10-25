# hopf-hough

**Exact retarded Liénard–Wiechert fields of moving point charges, evaluated per pixel on the GPU, with a built-in double-precision accuracy check.**

hopf-hough computes the electric field **E** (and the out-of-plane magnetic field B_z) of up to 256 point charges with prescribed motion, at every pixel of a 2D observation plane, in real time, in any browser with WebGL2. There is nothing to install: the app is a single HTML file. The same kernel is available as a TypeScript library for scripted, numeric use.

It adds no new physics. It is a fast, verifiable way to look at, and export, near-field and radiation-zone structure — for example how a synchrotron pulse forms — and to see how accurate the numbers are.

- **Live app:** https://timsdsdio.github.io/hopf-hough/ — the author's single-file app (`web/index.html`) running the v6f kernel by default; the **Kernel** button cycles v6f → v6f precise → v6 → v5 → v6f (v6f precise adds the field floor fix: more accurate at high γ, about 0.77× the frame rate of v6 on an M1 Max; v5 is the original kernel, bit-identical to the pre-port file kept as `reference/web_original.html`; v6 is bit-identical to the v6 kernel on main before this release, commit 904bcce, which includes the κ-form radiation term and the Newton fix, kept as `reference/web_v6_1.html`), and **Validity Mask** hatches untrusted pixels.
- **TypeScript build:** https://timsdsdio.github.io/hopf-hough/v6/ — the library's own app with the verification suite, CSV export and URL parameters (or open `dist/hopf-hough.html` after building); v6f by default, `?kernel=v6fp` for v6f precise, `?kernel=v6` for v6.
- **Preprint:** T. Huff, *A Causality-Preserving, Stereographically Compactified Worldline Representation for Retarded Liénard–Wiechert Electrodynamics*, Zenodo (2026), [doi:10.5281/zenodo.23581323](https://doi.org/10.5281/zenodo.23581323)

## What it computes

Charges are 3D point charges whose motion lies in the x–y plane; the field is evaluated in that plane, where **E** is in-plane and **B** = n̂ × **E** is along z. For each pixel and each charge the solver finds the retarded time on a compact stored history, then evaluates the standard Liénard–Wiechert field (Jackson eq. 14.14) from the interpolated history:

```
E = q (1 − β²)(n̂ − β) / (κ³ R²)  +  q n̂ × ((n̂ − β) × β̇) / (c κ³ R),     κ = 1 − n̂·β
```

Units are Gaussian-like code units with q = ±1 and c = 2.35 length units per time unit in the app (any c in the library). An optional Plummer softening ε replaces R² by R² + ε² near each charge; ε = 0 is exact Liénard–Wiechert, and every accuracy figure below uses ε = 0.

## How it works (short version)

1. **Gather, not scatter.** Grid solvers (FDTD/PIC) store the field and push sources into it. Here each pixel asks every source a question: *which event on your worldline lies on my past light cone?* That is a one-dimensional root solve per source, so it is exact and free of grid dispersion, and it is fast when sources are few and pixels many (N_sources ≪ N_pixels). The same split between "retarded-time" and "source-time" formulations is long established in aeroacoustics (Brentner 1996, 1997).
2. **Compact history.** Each charge's past is resampled onto M shells of retarded radius R = c(t − t_ret) using the stereographic map R = L tan(θ/2): nearly uniform spacing where radiation forms, reaching far with few shells. One texture row per charge.
3. **Causal interpolation.** Between shells the worldline is a cubic Hermite segment whose speed is certified below c (a Bernstein-coefficient bound on |z′|²), so the light-cone equation has exactly one root per observer.
4. **Precision where it matters.** Knot positions are stored as hi/lo pairs of float32, and the root function f(R) = R − |x − z(R)| is evaluated at the knots in double-float arithmetic (pairs of float32). In the default **v6f** kernel the root inside the segment is a float32 sum of non-negative terms, with the chord deficit 1 − |v_chord|² of each segment computed on the CPU in float64. The optional **v6f precise** kernel also takes the cancelling cross and dot products of the field near the pulse crest against the double-float separation with exact products (the field floor fix). This removes the float32 accuracy floors that otherwise appear from γ ≈ 30 (the v6 kernel evaluates the in-segment root in double-float too, but its float32 in-segment chord deficit still limits it at γ ≥ 71).
5. **Fast root search.** Bisection over the stored knots (v6f: in float32, with a double-float check of the bracket), then safeguarded Newton inside the bracketing segment.

Details, derivations and the original benchmarks are in the preprint; [docs/ACCURACY.md](docs/ACCURACY.md) has the current measured accuracy.

## Accuracy (summary)

Relative L2 error of GPU readback against the exact field (no softening), on 256 observers crossing the synchrotron pulse at retarded radius 1.15, for the default kernel (v6f), the v6 kernel and the original algorithm ([results/table2_v6_2.tsv](results/table2_v6_2.tsv)). Measured with WebGL2 in software rendering (SwiftShader). Numbers after v6.0.0 include the κ-form radiation term and the Newton fix; preprint v6 corresponds to tag v6.0.0 (doi:10.5281/zenodo.23601496); preprint v6.2 to tag v6.2.0 (doi:10.5281/zenodo.23649939). With the v6.0.0 kernel, an Apple M1 Max (Metal) gave the same figures as software rendering to within run-to-run scatter (e.g. 0.0089% and 0.059% at γ = 22 and 71, M = 512; [results/suite_m1max.tsv](results/suite_m1max.tsv)). The v6 kernel with the κ-form and Newton fixes (commit 06edf87, before the indicator's acceleration terms and known-path mode) was also measured there: 0.01% and 0.03% at M = 512 as printed by the in-app suite, and 62 fps for 32 sources in the default scene, against 120 fps (display-capped) for v5 ([results/suite_m1max_web_kform.txt](results/suite_m1max_web_kform.txt)). On the author's M1 Max (Safari, Metal; development builds with the v6.2 kernel code) the in-app suite with v6f and with v6f precise matched the float64 spline at γ = 71 (GPU 0.02% / 0.01% / 0.00% at M = 512 / 1024 / 2048, identical for both), and the default scene (32 sources, known path) ran at 43 fps with v6f, 33 fps with v6f precise, 43 fps with v6 and 120 fps with v5 ([results/suite_m1max_v6_2.txt](results/suite_m1max_v6_2.txt)). Run the in-app suite to measure on your own GPU.

<!-- ACCURACY_TABLE -->
| β | γ | v6f, M=128 | v6f, M=512 | v6f, M=2048 | v6, M=512 | original algorithm, M=512 |
|---|---|---|---|---|---|---|
| 0.8 | 1.7 | 0.0067% | 4.4e-4% | 3.7e-5% | 4.4e-4% | 0.0010% |
| 0.95 | 3.2 | 0.017% | 0.0010% | 6.6e-5% | 0.0010% | 0.0019% |
| 0.99 | 7.1 | 0.038% | 0.0022% | 1.6e-4% | 0.0022% | 0.0097% |
| 0.999 | 22.4 | 0.13% | 0.0076% | 5.5e-4% | 0.0077% | 0.22% |
| 0.9999 | 70.7 | 0.39% | 0.023% | 0.0018% | 0.036% | 7.6% |
| 0.99999 | 223.6 | 0.58%† | 0.066%† | 0.0060%† | 1.2%† | — |

† γ ≈ 224 uses double-precision observer coordinates (float32 coordinates cannot resolve the pulse there).
<!-- /ACCURACY_TABLE -->

The app hatches pixels where the answer should not be trusted: outside the stored history (red), or where the history is too coarse for the local motion (amber). In the far-field tests, no observer the flag leaves unmarked is more than 0.68% from the exact field (1.6% in preprint v6, before the acceleration terms below). For dragged sources the flag also watches the acceleration history: it now marks 96–99% of the pixels whose error exceeds 1% after sudden starts, stops and reversals (previously 0–35%), and adds at most 0.5% marked pixels on smooth presets. See [docs/ACCURACY.md](docs/ACCURACY.md) for the far-field envelope, the drag measurements, and how well the flag predicts the error.

**Known-path mode** (the app at the site root, and the library with `knownPath: true`): for preset motion the shells hold exact samples of the path, joined by quintic segments that match position, velocity and acceleration at both ends (C2 wherever the speed certificate does not have to limit a segment, C0 where it does). On the pulse sweep at M = 128, with the v6f kernel, it gives 9.2e-5% at γ = 7.1 (ring: 0.038%), 2.4e-4% at γ = 22 (ring: 0.13%) and 0.0011% at γ = 71 (ring: 0.39%); at M = 512 and γ = 224 it gives 0.0021% (ring: 0.066%). v6f precise lowers these to 3.3e-5%, 4.2e-5%, 1.1e-4% and 5.5e-4%. With the v6 kernel the known path is limited by v6's float32 floor at γ ≥ 22 (5.3e-3% at γ = 22, M = 128) and ties with the ring at γ ≥ 71; see section 7 of [docs/ACCURACY.md](docs/ACCURACY.md). The shell-count figures that follow were measured with v6. The shell count needed to match the ring at M = 2048 drops by at least 64× over the default view at γ ≤ 71 (4× at γ ≈ 224; 64× is the largest ratio measured), 2–16× on the sweep at γ ≤ 22, and 4× at 6 < R < 20 (single source, software rendering); see section 6 of [docs/ACCURACY.md](docs/ACCURACY.md). Dragged stretches of history still come from the ring buffer.

The app's **Speed** button (1×–4×, default 2×) runs that many simulation steps per animation frame, so the animation advances faster without changing the physics: the ring spacing and history depth are unchanged, and the shells are resampled once per frame.

## Using the app

This section describes the TypeScript build (`/v6/`). Open it, pick a scenario, and drag in the field to move the first charge (or switch to pan mode; the wheel zooms). The probe panel compares the GPU value at the green ring with an independent double-precision evaluation; **Check accuracy now** does the same on 64 nearby points. **Run verification suite** reproduces the accuracy table on your hardware; **Export field (CSV)** writes x, y, Ex, Ey, B_z and the validity flag for the current view.

Scenarios: circular orbit, an ensemble of precessing ellipses, harmonic oscillator, straight-line motion, hyperbolic motion, your own `x(t)`, `y(t)` expressions, or uploaded `t,x,y` samples. Superluminal input is refused.

Settings can be shared through the URL, e.g. `?scenario=orbit&beta=0.99&n=1&mt=2048&eps=0`.

## Using the library

```bash
npm install            # in a clone of this repository
npm run build          # dist/lib (ES modules + types), dist/hopf-hough.iife.js, dist/hopf-hough.html
```

```ts
import { DEFAULT_CONFIG, Simulation, expressionTrajectory, exactField, GpuSolver } from 'hopf-hough';

const c = 2.35;
const traj = expressionTrajectory({ x: '0.7*sin(1.9*t)', y: '0.35*sin(3.8*t)', c });
const cfg = { ...DEFAULT_CONFIG, c, mT: 512, kBin: 11, softEps: 0 };
const sim = new Simulation(cfg, [traj], 3.0);      // history ending at t = 3.0
sim.step(cfg.dt);                                   // advance as needed

const solver = new GpuSolver(document.createElement('canvas'));
solver.upload(sim.shellHistories, cfg.mT);
const field = solver.readFull([{ x: 1.2, y: 0.4 }], cfg);   // [{ ex, ey, bz, quality }]

const ref = exactField(sim.trajectories, 1.2, 0.4, sim.time, { c, softEps: 0 });  // float64 check
```

For known-path mode set `knownPath: true` in the config (compactified grid) and construct the solver with `new GpuSolver(canvas, KNOWN_PATH_SHADERS)`; with the default shaders the flag has no effect on the field.

Kernels: `PRODUCTION_SHADERS` = `V6F_SHADERS` (the default v6f, `F32N` + `F32N_NOFIX`, also used by `new GpuSolver(canvas)`) and `KNOWN_PATH_SHADERS` = `V6F_KNOWN_PATH_SHADERS`; `V6FP_SHADERS` / `V6FP_KNOWN_PATH_SHADERS` for v6f precise (`F32N` with the field floor fix); `V6_SHADERS` / `V6_KNOWN_PATH_SHADERS` for the v6 kernel (bit-identical to v6 on main at commit 904bcce, with the κ-form and Newton fixes; not the v6.0.0 kernel of preprint v6/v6.1); `LEGACY_SHADERS` for the version-5 algorithm. `kernelShaders('v6f' | 'v6fp' | 'v6' | 'legacy', knownPath)` picks one by name. v6f and v6fp read the chord deficit that `ShellHistory` computes during populate (`emitDch`), which `GpuSolver.upload` puts in the free channel of the low-part texture; use the same `SolverConfig` (in particular `betaMax`) for the history and the readback. The precision / certificate / root-search switches (`precision`, `certMode`, `rootMode`) are a study of the v6 kernel; v6f and v6fp always use the knot search + Newton.

Trajectory constructors: `circularOrbit`, `linearMotion`, `harmonicOscillator`, `hyperbolicMotion`, `precessingEllipse`, `functionTrajectory`, `expressionTrajectory`, `sampledTrajectory`, `csvTrajectory`; or any object with `kinematics(t)` returning position, β and a/c². The CPU parts (trajectories, histories, reference) run in Node; see [examples/](examples/).

## Verification

- **Independent code:** with softening off, a Python port of the double-precision reference agrees with [PyCharge](https://github.com/MatthewFilipovich/pycharge) 2.0.1 to ~1e-11 for circular, oscillating and straight-line motion up to γ = 7.1 ([validation/pycharge](validation/pycharge/REPORT.md)).
- **Regression:** the original single-file solver from the preprint is kept in [reference/](reference/); the port reproduces its verification suite exactly (154 of 154 values), and the CI checks this on every change.
- **Tests:** `npm test` (unit tests: trajectories, regression, drag physics, known-path knots) and `npm run test:gpu` (headless Chromium with software WebGL2: legacy reproduction, production accuracy thresholds for v6f, v6fp and v6, app smoke test; for `web/index.html`: v5 kernel bit-identical to the original file in `reference/web_original.html`, v6 kernel pulse-sweep accuracy, known-path accuracy and drag hand-off; the kernel button cycle; and for v6f and v6fp in the library and the app (`tests/gpu/v6f.mjs`): v6 bit-identical to the v6 kernel at main commit 904bcce (reference copies `reference/shader_v6_1.glsl`, `reference/web_v6_1.html`), v6f and v6fp accuracy at γ = 71 and 224 on known and live paths (v6fp below v6f on the known path), and their display pixels against v6).

## Compared with PyCharge

Same problems, grid and reference, same CPU ([validation/benchmarks](validation/benchmarks/README.md)). Ported to JAX, the method ties PyCharge 2.0.1 at γ ≲ 7 and for 16 sources, and is 5× faster at γ = 22 and 9× faster at γ = 71, at an accuracy of a few times 1e-5 (PyCharge is exact when it converges). In float32, the precision GPUs run at full speed, PyCharge's pulse error is 0.86% at γ = 22 and 27% at γ = 71; the WebGL kernel's is 0.0095% and 0.069% (v6.0.0 kernel, M = 512; 0.0077% and 0.036% after the κ-form and Newton fixes). On an M1 Max GPU the same problems take 5–7 ms for one source and 20 ms for 16 sources at the same accuracy (different machine from the CPU runs, so the ratio is indicative). To measure your GPU, open [examples/benchmark.html](https://timsdsdio.github.io/hopf-hough/examples/benchmark.html).

## Limitations

- 2D: motion in a plane, field in that plane.
- Prescribed motion: charges do not respond to each other's fields, and there is no radiation reaction.
- Accuracy falls with distance once the shell spacing can no longer resolve the motion; the validity mask marks this. The stored history is the ring buffer, about 260 length units deep in the default configuration; shells extend to R ≈ 4LM/π (220 for M = 128, L = 1.35) and anything beyond either limit is marked as outside the history.
- Very high γ: at γ ≈ 224 the pulse-sweep error with the v6f kernel is 0.066% (M = 512) and 0.0060% (M = 2048) on the ring history, 0.0021% and 0.0026% on the known path (v6f precise: 0.067%, 0.0050%, 5.5e-4% and 4.0e-4%; v6: 1.2% and 0.22% on the ring). Above γ ≈ 100, float32 observer coordinates cannot resolve the pulse; pass double-precision coordinates (`x64`, `y64`) to `readFull` (the rendered view uses float32 pixel coordinates).
- Accuracy has been measured in software rendering and, for the v6 kernel (as v6.0.0 and at commit 06edf87 with the κ-form and Newton fixes), on an Apple M1 Max; of the v6.2 kernels, v6f and v6f precise were both run through the in-app suite on that M1 Max (Safari, development builds with the release's kernel code) and printed identical values; its ring-buffer histories do not exercise the field floor fix. Apple's shader compiler reassociates floating-point sums, which undid an earlier version of the double-float code (it fell back to float32 accuracy). The shader now routes every rounding step through a bit-level barrier; `examples/precision-probe.html` checks each double-float building block on your GPU, and the in-app suite checks the whole solver.

## Citing

If you use hopf-hough, please cite the preprint (see [CITATION.cff](CITATION.cff)).

## Development and AI assistance

hopf-hough was conceived and directed by Tim Huff. AI systems contributed substantially: Gemini Argon (Google) to the original shader, the causality certificate and the first manuscript; Astra (OpenAI) to adversarial review of earlier drafts and code; Claude (Anthropic; Claude Fable 5.1 and Claude Opus 5.5) to the review of the preprint, the diagnosis of the accuracy floor and limiter, the TypeScript port, the production shader, the tests, the audit of the validity indicator, the PyCharge benchmark and JAX port, the diagnosis and fix of the Metal compiler issue, the v6f kernel and the attribution of its error floors, and this documentation. All results in this repository are produced by the code and tests in it, and were reviewed by the author.

## License

MIT — see [LICENSE](LICENSE).
