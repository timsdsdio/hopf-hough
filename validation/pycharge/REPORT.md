# Liénard–Wiechert cross-validation: original analytic reference vs PyCharge v2

## Setup
- **What was tested:** a Float64 Python port (`reference_port.py`) of `getExactTrajectoryKinematics`, `evaluateExactAnalyticalLW` and `getPulseCrossingSweepPoints`. The port follows the original step for step: 32 bisection steps on [t−260/c, t], at most 12 safeguarded Newton steps, then the LW E field. It uses a single charge, as the GPU suite does. B_z = (n×E)_z.
- **Independent check:** PyCharge 2.0.1 (JAX 0.12.0.1, float64 on). It implements Jackson eq. 14.14 in SI and gets v and a by differentiating a JAX position function (`trajectories.py`). That independently checks the original's hand-coded `bx`/`bdx`.
- **Cases:**
  - synchrotron (ω=1.6) at β = 0.5, 0.9, 0.99
  - oscillator at β_max = 0.3 and 0.8
  - straight-line motion along +x at β = 0.44 and 0.9, with the charge at the origin at t_obs
- **Observers:** rings of 64 points at R = 0.5, 1.15, 3, 10, at t = 2.508 (= 2.40 + 16·DT_STEP, the suite's `tEval`). For β = 0.9 and 0.99, also a 256-point pulse-crossing sweep at retarded distance 1.15, using Float64 coordinates.

## Unit mapping
- Code units: c = 2.35, q = 1.
- `bdx = a/c²` = (dβ/dt)/c. So the code's `q n×((n−β)×bdx)/(κ³R)` equals the Gaussian `q n×((n−β)×β̇)/(cκ³R)`. Its dimensions are q/length², the same as the velocity term, so it is correct.
- The radiation term is up to 105× the velocity term on the rings, so a stray factor of c would show up.
- SI mapping: 1 length unit = 1 m; 1 time unit = 2.35 m / c_SI (this keeps β the same); q = e.
- Conversion back: `E_code = E_SI·4πε0·L0²/e` and `B_z,code = c_SI·B_z,SI·4πε0·L0²/e`.
- The errors below are vector errors relative to |E_PyCharge| after this conversion, so they cover both the normalized comparison and the absolute scale.

## Results (eps = 0)
The last column is the max error of the original's default eps=0.06 reference against true LW, in %, for R = 0.5/1.15/3/10.

| case | set | max rel | RMS rel | max rel B_z | max \|ratio−1\| | eps=0.06 max err % |
|---|---|---|---|---|---|---|
| synchrotron β=0.5 | rings | 1.1e-14 | 1.4e-15 | 4.5e-13 | 5e-15 | 6.1 / 2.1 / 0.073 / 0.0043 |
| synchrotron β=0.9 | rings | 1.2e-13 | 1.1e-14 | 1.5e-13 | 1e-13 | 2.1 / 4.0 / 0.070 / 0.0021 |
| synchrotron β=0.9 | sweep | 7.6e-14 | 1.4e-14 | 2.1e-12 | 7e-14 | 0.97 |
| synchrotron β=0.99 | rings | 1.0e-13 | 7.5e-15 | 1.0e-13 | 3e-14 | 1.5 / 1.8 / 0.039 / 0.0021 |
| synchrotron β=0.99 | sweep | 1.3e-11 | 1.4e-12 | 2.3e-11 | 9e-12 | 1.1 |
| oscillator β=0.3 | rings | 3.7e-15 | 7.8e-16 | 9.2e-15 | 2e-15 | 7.8 / 0.30 / 0.051 / 0.0038 |
| oscillator "β=0.8" (preset, really 0.511) | rings | 5.9e-15 | 1.3e-15 | 1.0e-14 | 4e-15 | 54 / 0.35 / 0.059 / 0.0040 |
| oscillator β=0.8 (uncapped) | rings | 6.2e-15 | 1.4e-15 | 8.6e-15 | 5e-15 | 5.3 / 0.74 / 0.070 / 0.0045 |
| linear β=0.44 | rings | 2.9e-15 | 5.6e-16 | 4.0e-15 | 2e-15 | 2.9 / 0.56 / 0.083 / 0.0075 |
| linear β=0.9 | rings | 1.3e-14 | 3.2e-15 | 1.4e-14 | 1e-14 | 4.9 / 0.97 / 0.14 / 0.013 |

The sweeps do cross the pulse: |E| peaks at the centre sample, and the max/min |E| ratio is 81 at β=0.9 and 262 at β=0.99. The 1e-11 error at β=0.99 is ordinary rounding amplified by κ³ ≈ 1e-6.

**Conclusion:** with eps = 0, the original Float64 reference is correct to machine precision.

## Discrepancies and caveats in the original
1. **Softening is built into the "exact" reference.** `evaluateExactAnalyticalLW` replaces R² with R²+ε² (velocity term) and R with √(R²+ε²) (radiation term), using `state.softEps = 0.06`. `runAutomatedGPUVerificationSuite` never resets it, so the suite's "% vs exact" is measured against a softened field, not true LW. The bias at retarded distance R = 0.3/1/3 is 3.85%/0.36%/0.040% for the velocity term and 1.94%/0.18%/0.020% for the radiation term. A GPU error of about 1% near R ≈ 1 can't be told apart from this bias.
2. **Oscillator amplitude is capped:** `maxA = min(0.75, βc/max(0.4, ω))`, so at ω=1.6, β_max = min(β, 0.511). Every requested β ≥ 0.511 gives the same motion.
3. **Synchrotron radius is clamped:** `rOrb = min(0.85, …)`, so the actual angular rate is 2.488 at β=0.9 and 2.737 at β=0.99, not 1.6. The motion is still physically consistent; only the label is wrong.
4. **The `uniform` preset** is a sawtooth that jumps every 4.4 time units and moves at 0.55β, so a true straight line was used instead. `swarm` was excluded as instructed.
5. **Minor:** the bisection assumes the charge was always within 260 length units of the observer, and Newton stops at a fixed absolute tolerance of |f| < 1e-13. Neither matters for these cases.

## Rerun
```
pip install pycharge --break-system-packages
cd validation/pycharge && python3 compare.py            # ~20 s, writes results.json
python3 pycharge_fields.py example_input.json out.json  # format in README.md
```
