# hopf-hough

**Exact retarded Liénard–Wiechert fields of moving point charges, evaluated per pixel on the GPU, with a built-in double-precision accuracy check.**

hopf-hough computes the electric field **E** (and the out-of-plane magnetic field B_z) of up to 256 point charges with prescribed motion, at every pixel of a 2D observation plane, in real time, in any browser with WebGL2. There is nothing to install: the app is a single HTML file. The same kernel is available as a TypeScript library for scripted, numeric use.

It adds no new physics. It is a fast, verifiable way to look at, and export, near-field and radiation-zone structure — for example how a synchrotron pulse forms — and to see how accurate the numbers are.

- **Live app:** https://timsdsdio.github.io/hopf-hough/ (the author's original single-file solver, `web/index.html`)
- **TypeScript build (v6 kernel):** https://timsdsdio.github.io/hopf-hough/v6/ (or open `dist/hopf-hough.html` after building)
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
4. **Precision where it matters.** The root function f(R) = R − |x − z(R)| is evaluated in double-float arithmetic (pairs of float32) on hi/lo knot positions; everything else is float32. This removes the float32 accuracy floor that otherwise appears near γ ≈ 30–70.
5. **Fast root search.** Bisection over the stored knots, then safeguarded Newton inside the bracketing segment.

Details, derivations and the original benchmarks are in the preprint; [docs/ACCURACY.md](docs/ACCURACY.md) has the current measured accuracy.

## Accuracy (summary)

Relative L2 error of GPU readback against the exact field (no softening), on 256 observers crossing the synchrotron pulse at retarded radius 1.15. Measured with WebGL2 in software rendering (SwiftShader); an Apple M1 Max (Metal) gives the same figures to within run-to-run scatter, e.g. 0.0089% and 0.059% at γ = 22 and 71 with M = 512 ([results/suite_m1max.tsv](results/suite_m1max.tsv)). Run the in-app suite to measure on your own GPU.

<!-- ACCURACY_TABLE -->
| β | γ | this version, M=128 | this version, M=512 | this version, M=2048 | original algorithm, M=512 |
|---|---|---|---|---|---|
| 0.8 | 1.7 | 0.0067% | 4.5e-4% | 4.5e-5% | 0.0010% |
| 0.95 | 3.2 | 0.017% | 0.0010% | 1.5e-4% | 0.0019% |
| 0.99 | 7.1 | 0.038% | 0.0023% | 4.3e-4% | 0.0097% |
| 0.999 | 22.4 | 0.13% | 0.0095% | 0.0067% | 0.22% |
| 0.9999 | 70.7 | 0.37% | 0.069% | 0.051% | 7.6% |
| 0.99999 | 223.6 | 7.0%† | 1.3%† | 0.45%† | — |

† γ ≈ 224 uses double-precision observer coordinates (float32 coordinates cannot resolve the pulse there).
<!-- /ACCURACY_TABLE -->

The app hatches pixels where the answer should not be trusted: outside the stored history (red), or where the history is too coarse for the local motion (amber). In the far-field tests, no observer the flag leaves unmarked is more than 1.6% from the exact field. See [docs/ACCURACY.md](docs/ACCURACY.md) for the far-field envelope and how well that flag predicts the error.

## Using the app

Open the app, pick a scenario, and drag in the field to move the first charge (or switch to pan mode; the wheel zooms). The probe panel compares the GPU value at the green ring with an independent double-precision evaluation; **Check accuracy now** does the same on 64 nearby points. **Run verification suite** reproduces the accuracy table on your hardware; **Export field (CSV)** writes x, y, Ex, Ey, B_z and the validity flag for the current view.

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

Trajectory constructors: `circularOrbit`, `linearMotion`, `harmonicOscillator`, `hyperbolicMotion`, `precessingEllipse`, `functionTrajectory`, `expressionTrajectory`, `sampledTrajectory`, `csvTrajectory`; or any object with `kinematics(t)` returning position, β and a/c². The CPU parts (trajectories, histories, reference) run in Node; see [examples/](examples/).

## Verification

- **Independent code:** with softening off, a Python port of the double-precision reference agrees with [PyCharge](https://github.com/MatthewFilipovich/pycharge) 2.0.1 to ~1e-11 for circular, oscillating and straight-line motion up to γ = 7.1 ([validation/pycharge](validation/pycharge/REPORT.md)).
- **Regression:** the original single-file solver from the preprint is kept in [reference/](reference/); the port reproduces its verification suite exactly (154 of 154 values), and the CI checks this on every change.
- **Tests:** `npm test` (unit tests: trajectories, regression) and `npm run test:gpu` (headless Chromium with software WebGL2: legacy reproduction, production accuracy thresholds, app smoke test).

## Compared with PyCharge

Same problems, grid and reference, same CPU ([validation/benchmarks](validation/benchmarks/README.md)). Ported to JAX, the method ties PyCharge 2.0.1 at γ ≲ 7 and for 16 sources, and is 5× faster at γ = 22 and 9× faster at γ = 71, at an accuracy of a few times 1e-5 (PyCharge is exact when it converges). In float32, the precision GPUs run at full speed, PyCharge's pulse error is 0.86% at γ = 22 and 27% at γ = 71; the WebGL kernel's is 0.0095% and 0.069%. On an M1 Max GPU the same problems take 5–7 ms for one source and 20 ms for 16 sources at the same accuracy (different machine from the CPU runs, so the ratio is indicative). To measure your GPU, open [examples/benchmark.html](https://timsdsdio.github.io/hopf-hough/examples/benchmark.html).

## Limitations

- 2D: motion in a plane, field in that plane.
- Prescribed motion: charges do not respond to each other's fields, and there is no radiation reaction.
- Accuracy falls with distance once the shell spacing can no longer resolve the motion; the validity mask marks this. The stored history is the ring buffer, about 260 length units deep in the default configuration; shells extend to R ≈ 4LM/π (220 for M = 128, L = 1.35) and anything beyond either limit is marked as outside the history.
- Very high γ: at γ ≈ 224 the pulse-sweep error is 1.3% (M = 512) and 0.45% (M = 2048). Above γ ≈ 100, float32 observer coordinates cannot resolve the pulse; pass double-precision coordinates (`x64`, `y64`) to `readFull` (the rendered view uses float32 pixel coordinates).
- Accuracy has been measured in software rendering and on an Apple M1 Max. Apple's shader compiler reassociates floating-point sums, which undid an earlier version of the double-float code (it fell back to float32 accuracy). The shader now routes every rounding step through a bit-level barrier; `examples/precision-probe.html` checks each double-float building block on your GPU, and the in-app suite checks the whole solver.

## Citing

If you use hopf-hough, please cite the preprint (see [CITATION.cff](CITATION.cff)).

## Development and AI assistance

hopf-hough was conceived and directed by Tim Huff. AI systems contributed substantially: Gemini Argon (Google) to the original shader, the causality certificate and the first manuscript; Astra (OpenAI) to adversarial review of earlier drafts and code; Claude (Anthropic; Claude Fable 5.1 and Claude Opus 5.5) to the review of the preprint, the diagnosis of the accuracy floor and limiter, the TypeScript port, the production shader, the tests, the audit of the validity indicator, the PyCharge benchmark and JAX port, the diagnosis and fix of the Metal compiler issue, and this documentation. All results in this repository are produced by the code and tests in it, and were reviewed by the author.

## License

MIT — see [LICENSE](LICENSE).
