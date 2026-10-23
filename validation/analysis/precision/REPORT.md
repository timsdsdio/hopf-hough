# Branch D: Fixing the Float32 floor in the ultrarelativistic regime

## 1. Claim (falsifiable)
The 7–10% Float32 error at γ = 70.7 comes almost entirely from evaluating the light-cone residual f = R − |x − z(R)| in absolute single-precision coordinates. That residual has a fixed error of about 1 ulp (~6e-8), and the pulse width shrinks as γ⁻³, so the error grows as γ³. Two changes remove it, and nothing else in the shader needs to change:
- store knot positions as hi/lo float pairs;
- evaluate the single function f(R) in double-float (df64).

With both, the pulse error drops to 0.06% at γ = 70.7 and to 0.6–0.8% at γ = 200, while the spline, the κ evaluation and the LW kernel all stay in plain Float32.

## 2. Why it matters / who cares
**Venue: a section of the main paper.** It is too thin to stand alone as a paper. At most it could be a short note in a GPU or computational-physics venue, and only if it is validated on real GPUs. The section would do four things:
- replace the paper's attribution of the floor ("κ⁻³ ≈ 10¹² evaluation", "Newton conditioning", "hardware-level differences") with a measured error budget;
- turn the "GPU vs CPU-F32 discrepancy" from an open question into an explained effect;
- extend the usable γ from about 30 to at least 100;
- reframe the published Float32 result honestly as backward-stable to about 0.6 ulp in observer position.

Readers: authors of GPU retarded-time or LW codes, and WebGL/WebGPU visualization.

## 3. Nearest prior art and the delta
- **Double-float on GPUs.** The arithmetic itself is decades old:
  - Dekker/Knuth error-free transforms; Thall, *Extended-precision floating-point numbers for GPU computation* (2006), https://hgpu.org/?p=1806
  - Da Graça & Defour, *float-float operators on graphics hardware* (2006), https://arxiv.org/abs/cs/0603115
  - luma.gl/deck.gl fp64 in WebGL: https://luma.gl/next/docs/api-reference/shadertools/shader-modules/fp64 and https://deck.gl/docs/developer-guide/fp64. deck.gl later replaced fp64 with a cheaper offset (relative-coordinate) 32-bit projection mode: https://github.com/visgl/deck.gl/blob/9a77f7d30b6b5a17313a120aff47f85bd7814daa/dev-docs/RFCs/vNext/projection-mode-improvements-rfc.md
- **Relativistic pushers and precision.** These codes store u = γβ to avoid cancellation in 1 − β:
  - Vay pusher (2008); Higuera–Cary, https://arxiv.org/abs/1701.05605
  - Ripperda et al. comparison, https://arxiv.org/abs/1710.09164
  - Single and mixed precision in PIC: PIConGPU (https://hzdr.de/publications/Publ-16338) and mixed-precision implicit PIC (https://arxiv.org/abs/1111.5295)

  The paper's 4-velocity / cancellation-free κ trick belongs to this family.
- **The delta.** None of this new as arithmetic. What is new is the diagnosis for retarded-time root finding:
  - The quantity that limits accuracy is *not* 1 − β or κ (that contribution measures 0.04%).
  - It is the absolute error in f. An error in f shifts the effective observer position one-for-one, and the condition number grows as γ³.
  - A minimal df64 footprint fixes it: about 12 double-float ops per sampler call plus one extra RG32F texture.

## 4. Minimum viable experiment
1. Port `fDF` (in `core.js`) and a hi/lo position texture into the GLSL shader.
2. On the author's M1 Max (ANGLE→Metal), plus one NVIDIA or Intel GPU, rerun the γ = 70.7 sweep. Pass condition: GPU vs exact < 0.1% at M = 512 and 2048, and GPU vs CPU-df64 emulation < 0.05%.
3. Guard the error-free transforms against compiler reassociation. WebGL2 has no `precise` qualifier, so use the standard trick of multiplying by a uniform 1.0. Report the frame-time overhead.

## 5. What I actually checked
Scripts are in `./`:
- `core.js` is a faithful standalone extraction of the ring buffer, `populateShellsFromRingBuffer`, `sampleSourceHistoryFloat32`, `evaluateShaderEquivalentFloat32` and the exact reference, with a rounding hook for each stage.
- `ablate.js`, `fix.js`, `condition.js` and `gpucpu.js` produce the outputs saved as `*.out`.

**Reproduction.**
- CPU-F32 emulation vs exact at γ = 70.7: 7.23% / 9.72% / 7.79% at M = 512 / 1024 / 2048. The paper's GPU figures are 7.15 / 8.95 / 9.70%.
- Float64 spline: 0.0237 / 0.0060 / 0.00166% (paper: 0.0241 / 0.0062 / 0.00166%).
- Reference shift from rounding the query coordinates: **3.23%**, exactly matching the paper.

**Error budget at γ = 70.7.** Paper metric. "F64 + this stage F32" switches on only that stage's Float32 rounding inside an otherwise Float64 run. "F32 − this stage" promotes only that stage to Float64 inside the full Float32 run.

| Stage | F64 + this stage F32, M = 512 / 2048 | F32 − this stage, M = 512 / 2048 |
|---|---|---|
| **Root-function arithmetic (x − z, \|·\|, R − dist)** | **12.4% / 8.24%** | **1.23% / 2.09%** |
| **Knot-position storage (absolute coordinates)** | **2.42% / 2.29%** | 7.70% / 7.97% (masked) |
| R_shell rounding | 0.024% / 0.0017% (= F64 floor; knots are sampled at the rounded R) | — |
| Query rounding | 0 by construction (reference evaluated at the same point) | — |
| Newton conditioning / iteration count | F64 run with the solver's 11 + 2 iterations and clamps: no change | F32 with 24 + 6 safeguarded Newton: 9.75% / 9.08% (worse) |
| κ evaluation and LW kernel | 0.042% / 0.049% | no gain |
| \|u\| linear interpolation (arithmetic / model) | 0.037% / 0.029%; exact \|u\| gives no change | no gain |
| Acceleration Catmull–Rom (arithmetic / model) | arithmetic: no change. Model: exact acceleration gives 0.0006% (it is the dominant *F64* error) | no gain |
| Bernstein limiter | — | off: 7.23% / 7.66%. It does fire in F32 at M = 2048 (α_min = 0.28) because of chord noise |
| **Root arithmetic + knot storage both in F64** | — | **0.073% / 0.047%** |

**The GPU–CPU discrepancy.** Reordering two lines in a legal way (re-associating x − z₁ − Δz, and fusing x² + y²) changes the CPU-F32 result by 5.96% / 7.27% / 5.69%. The paper reports 5.97% / 7.82% / 8.70%. The discrepancy is ulp noise amplified by the conditioning; it is not a hardware fault.

**Well-posedness (`condition.js`, Float64 reference).**
- At γ = 70.7, max |∂E/∂x|·|x|/E_rms = 8.6e6.
- A 1-ulp move in x changes E by up to 71% of E_rms.
- The pulse FWHM is about 15 ulps at γ = 70.7, and 0.8 ulp at γ = 200.
- Rounding the query by ±½ ulp predicts a 3.37% rms shift; 3.23% is measured.
- The published 7.2% therefore corresponds to an effective observer error of about 0.6 ulp rms. The Float32 solver is close to backward-optimal; the "floor" is the condition number, not a bug.
- The paper's metric (exact field at the represented point) is legitimate, but a sentence should state this backward-error reading.
- Fair alternatives:
  - Fluence error: published 2.6% / 3.8%, fix 0.012% / 0.004%.
  - Phase-aligned L2: published 7.15% / 5.59%. Alignment does not rescue it, because the error is per-point noise rather than a shift.
- At γ = 200, an f32 query cannot even sample the pulse: only 101 of the 256 sweep points are distinct. Above γ ≈ 100, queries must be relative or df64.

**The fix (`fix.js`, M = 512 / 2048, paper metric).** At γ = 200 the fix row uses β_max = 0.9999995 and df64 queries; the "6 safeguarded Newton" variant gives 0.57% / 0.58%.

| γ | Published | df64 f(R) only | df64 f(R) + hi/lo knots |
|---|---|---|---|
| 22.4 | 0.19% / 0.30% | 0.068% / 0.080% | **0.010% / 0.005%** |
| 70.7 | 7.23% / 7.79% | 1.22% / 2.08% | **0.061% / 0.057%** |
| 200 (β_max raised) | 115% / 106% | 45% / 36% | **0.77% / 2.96%** (df64 query: 1.06% / 2.24%) |

- The published error follows roughly a γ³ law: 0.19, 0.60, 3.06, 7.23 and 13.1% at γ = 22, 32, 50, 71 and 100. The fix is about 100× lower across that range.
- The chord must also be formed from the hi/lo pairs. Without that, hi/lo storage gives no gain.
- β_max = 0.99995 is hard-coded, which caps γ at 100. γ = 200 is not runnable without changing it.
- Cost, from f32 op counts: about 2.7× ALU per pixel (2373 → 6372 ops; no FMA in GLSL ES 3.0), plus one extra texel fetch per sampler call and 1.5× texture memory. None of this has been measured on a GPU.
- A hybrid (f32 bisection, then a widened df64 bracket) works at γ ≤ 70 but fails at γ = 200.

## 6. Kill criteria
- The GPU compiler (ANGLE/Metal fast-math, or driver reassociation) breaks the TwoSum/TwoProd steps, no workaround restores better than 0.5% at γ = 70.7.
- The measured frame-time overhead exceeds about 4× in the visual (N_p = 32) mode.
- A reviewer considers df64 plus a relative frame too standard. Then the result belongs in the paper as a section, not as a separate publication.

## 7. Verdict
- **Strong as a section of the main paper** (confidence 0.85). It is cheap, the numbers are already computed, and it corrects wrong claims in §4.2 and §5.
- **Weak as a standalone paper** (confidence 0.75). The arithmetic is textbook; the novelty is the diagnosis plus a minimal footprint.
- Main open risk: GPU compilers preserving the double-float ops (untested; no GPU in this environment).
