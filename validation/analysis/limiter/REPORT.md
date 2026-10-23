# Branch A: the Bernstein convex-hull causality certificate as a standalone paper

## 1. Claim (falsifiable)
For cubic Hermite worldline interpolation in retarded radius, a closed-form tangent limiter can certify |z'(R)| ≤ β_max < 1 on every segment, which makes the light-cone root unique. **As written in Theorem 1, though, the limiter is about 4× more conservative than it needs to be. On the paper's own benchmark it never prevents a real superluminal overshoot, and it raises the pulse error by up to 100×.** A sharper closed-form certificate removes that cost.

## 2. Why it matters / who cares
Some codes evaluate retarded fields from tabulated trajectories: Liénard–Wiechert post-processors for PIC (PIConGPU/ClaRa-style radiation diagnostics), inverse-Compton and synchrotron spectra codes, and GPU visualizers. Those that move beyond piecewise-linear worldlines need the light-cone root to stay unique. The issue gets worse on compactified or log grids, because far shells are coarse (ΔR ∝ R²). Possible venues are a short note in *Computer Physics Communications* or *J. Comput. Phys.* (Short Note), or a CAGD-flavoured note. Realistically, this material is most likely to survive as a corrected section of a larger solver paper.

## 3. Nearest prior art, and the delta
- **Bernstein hodograph convex-hull derivative bounds** are textbook CAGD (Farin; Farouki, *Pythagorean-Hodograph Curves*; Farouki & Rajan's Bernstein-form polynomial algorithms). Robotics uses them routinely as velocity and acceleration feasibility constraints:
  - Zhou/Gao/Shen-style Bernstein-basis planners, e.g. Park et al., [arXiv:1909.10219](https://arxiv.org/pdf/1909.10219)
  - BeBOT, Kielas-Jensen & Cichella: [arXiv:2010.09992](https://arxiv.org/abs/2010.09992), [PMC8915131](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8915131/)
  - MINVO, a tighter enclosure than Bernstein: [arXiv:2010.10726](https://arxiv.org/abs/2010.10726)
  - CNC feedrate work: [Farouki, time-optimal CNC](https://faculty.engineering.ucdavis.edu/farouki/wp-content/uploads/sites/51/2021/07/Time-optimal-control-of-CNC-machines.pdf)
- **Scaling the endpoint tangents toward the secant** is the Fritsch–Carlson (1980) / Fritsch–Butland monotone-Hermite construction. Theorem 1's α is its vector, ball-constrained analogue. The CAGD literature on constrained Hermite interpolation is broad, e.g. [Constrained Interpolation via Cubic Hermite Splines](https://mmr.khu.ac.ir/article_8593.html?lang=en).
- **Retarded-field codes** usually avoid the problem with piecewise-linear worldlines. If every chord is subluminal, uniqueness is trivial. See e.g. [arXiv:2505.20596](https://arxiv.org/html/2505.20596) (linear interpolation plus a quadratic solve). [LW-integrator](https://lw-integrator.readthedocs.io/en/latest/overview.html) rejects |β| ≥ 1 states but says nothing about the interpolant.
- I found **no paper** that applies Bernstein or hodograph bounds to retarded-root uniqueness or to causal worldline interpolation.

**Honest delta.** The new part is narrow: applying a known bound to a known root-uniqueness argument, plus a closed-form α that is cheap on a GPU. Without new results, a referee will call it "Fritsch–Carlson plus the hodograph convex hull". Sections 4–5 list results that would be new: the conservatism analysis, a sharper certificate, the C¹ impossibility and degree-n results, and a far-shell failure demonstration.

## 4. Minimum viable proof or experiment
1. **Sharper closed-form certificate.** Write |z'(s)|² as a quartic in Bernstein form, with coefficients c₀ = V₀·V₀, c₁ = V₀·V₁, c₂ = (2V₁·V₁ + V₀·V₂)/3, c₃ = V₁·V₂, c₄ = V₂·V₂. Each cₖ is quadratic in α. Take α_sq as the first root in (0, 1] of cₖ(α) = β_max², which costs three quadratic solves. Whenever Theorem 1 accepts an α, this test also accepts it, so α_sq ≥ α\*. For a circular arc with turning angle φ, the paper's middle control obeys |V₁| = β(1 + φ²/8 + O(φ⁴)), whereas c₁ and c₂ equal β²(1 + O(φ⁴)).
2. **Well-posedness of the exact limiter.** Because z'_α(s) is affine in α, the map α ↦ max_s |z'_α(s)| is convex and equals |v_sec| at α = 0. The feasible set is therefore an interval [0, α_sharp], and bisection on α is rigorous.
3. **No global C¹ cubic theorem: counterexample.** Take chords alternating ±c with free knot slopes. A C¹ piecewise cubic with |z'| ≤ β exists **iff c ≤ 2β/3**. Theorem 1's piecewise-C¹ fallback cannot be upgraded.
4. **Degree-n C¹ fix with the exact physical knot velocities.** The degree-n Bernstein certificate succeeds iff ‖n·v_sec − m₀ − m₁‖ ≤ (n−2)β_max, with all interior controls set equal to (n·v_sec − m₀ − m₁)/(n−2). It always succeeds for n ≥ 4β/(β − |v_sec|). For circular arcs n = 6 always works, since the φ² terms cancel at n = 6.
5. **Failure demonstration.** Choose an accelerating source on coarse far shells, where the raw cubic produces three light-cone roots. Compare four methods there: linear, raw cubic, Theorem 1, and α_sq/α_sharp.

## 5. What I actually checked
All scripts are in `./`. I extracted the solver JS unmodified and instrumented a copy (`core_inst*.js`); the inputs were not touched. The harness reproduces the paper's Float64 numbers exactly: 0.02362%, 0.00602% and 0.00166% at M_T = 512/1024/2048.

- **Limiter activity** (`run_limiter.js`). At γ = 7, 22 and 71, α\* < 1 on 14–87% of shell segments (minimum α\* 0.13–0.28). In the R_ret = 1.15 sweep it fires only at γ = 71:
  - **M_T = 128:** all samples are limited. CPU-Float32 error is **46.8% with the limiter vs 7.05% without**. Float64 (limiter added) gives **46.2% vs 0.394%**. Table 1's 56% at M_T = 128 is therefore mostly caused by the limiter itself.
  - **M_T = 2048, Float32:** 383 of 4096 calls are limited, triggered by Float32 noise. The cost is 7.79% vs 7.66%.
  - **Table 2's Float64 column** never applies the limiter, so it measures a different algorithm from the GPU.
- **False positives** (`e2_knots.py`, about 9,000 segments). The raw α = 1 cubic is **never superluminal** in Float64. Its maximum |z'| equals the knot speed, because circular-arc Hermite cubics undershoot between knots (`e1_circle.py`, φ up to 1 rad). In Float32-stored knots, 5 segments exceed β_max = 0.99995, and none reaches 1. Activation counts for Theorem 1, α_sq and α_sharp were 18–386, 8–106 and 0–5.
- **Accuracy at larger R_ret** (`run_modes.js`, Float64). Applying the paper's limiter gives 30–99% error where the unlimited cubic gives 0.2–2%. Cases: γ = 22, M = 128, R = 3: 30.7% vs 1.08%. γ = 71, M = 512, R = 3: 38% vs 0.205%. γ = 71, M = 2048, R = 8: 94% vs 0.24%. With α_sq or α_sharp the result is **identical to the unlimited cubic** in every resolved case. Where even the unlimited cubic exceeds 70% error (unresolved), α_sq is erratic.
- **Conservatism on adversarial random data** (`e3_random.py`). When α_sharp < 1, the median α\*/α_sharp is **0.257** and the median α_sq/α_sharp is 0.464. Theorem 1 fires on 30% of segments that are actually subluminal.
- **Real overshoots do exist** (`e3_random.py`, `e4_roots.py`):
  - Coarse hyperbolic-motion segments give a raw maximum |z'| of 1.003–1.32. Observers ahead of the source then see 3 roots across a band 0.07–14 units wide.
  - Off-centre velocity kicks overshoot.
  - Leapfrog-staggered velocities on a circle overshoot even at φ = 0.02 when β = 0.9999.
  - The certificate's legitimate domain is coarse, accelerating or inconsistent data, not smooth circular orbits.
- **C¹ results** (`e5_c1.py`). The 2β/3 threshold is confirmed by optimization: feasible at c = 0.66, infeasible at c = 0.67 with β = 0.99. Minimal degrees n\* are 3–6 for circles, 5 for kicks, and 10/102 for the two hyperbolic segments.
- **Note:** β_max = 0.99995 is hard-coded. For γ > 100, knot velocities are clipped before the certificate runs.

## 6. Kill criteria
- No realistic test (coarse far shells with an accelerating or kicked source, or PIC-staggered data) shows raw-cubic multi-root errors that piecewise-linear plus a cheap direction fix does not already avoid.
- A referee-level search turns up prior relativistic or causal use of hodograph bounds.
- α_sq or α_sharp fail to match unlimited-cubic accuracy on resolved data, which they did not do here.

## 7. Verdict
**Weak as a standalone paper; viable as a corrected section** of the solver paper, or as a short note if it is rebuilt around items 4.1–4.5. Confidence about 75%. The most useful immediate result is negative and actionable for the existing preprint:
- Theorem 1's limiter is about 4× over-conservative.
- It is the dominant error source in the γ = 71, M_T = 128 entry.
- It makes the GPU and the Float64 "reference" different algorithms.

Replacing it with α_sq (closed form, three quadratics) or α_sharp removes all of these effects on the benchmark and keeps the guarantee.
