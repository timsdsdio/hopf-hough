# Audit of preprint v6 against the hopf-hough repository

Auditor: independent review, 2026-10-22. Nothing in the repository was edited.
Scope: `paper/preprint_v6.tex` vs `results/tables.json`, `docs/ACCURACY.md`, `validation/**`, `reference/**`, `src/core/*.ts`, `src/gpu/shader.ts`, `src/gpu/solver.ts`, `app/app.ts`.

**Checks I ran myself** (scripts are in the session scratchpad, not in the repo):
- `tests/gpu/legacy_regression.mjs`: **154/154 values identical**, so that claim reproduces.
- `npm test`: 8/8 unit tests pass.
- A Python check of Proposition 1 in float64: the Bernstein product formula, α_sq ≥ α*, and the speed bound, on 12 000 random cases.
- A float32 emulation of the shader's `alphaSharp`.
- A GPU comparison of the regression-tested legacy shader with the production shader in legacy-switch mode.
- A GPU test of the validity indicator for straight-line motion at large distance.
- A rerun of `validation/analysis/farfield/fresnel_decompose.py`.
- A recomputation of the legacy swarm-preset acceleration error.
- WebSearch on the doubtful references.

---

## Summary

The core tables are faithfully transcribed. Every cell of Tables 1–3 and every number in the abstract matches `results/tables.json` after rounding. The certificate formula is correct, and the shader implements it exactly. The chord clamp, the Newton floor, the hi/lo storage and the `u_one` guard all match the text.

The problems are in the prose around the tables:
- **Validity indicator.**
  - The abstract's claim about the indicator is stronger than the data.
  - The indicator can be shown to miss large errors.
- **Far-field estimate.** The hand-off radius is wrong by a factor of 100.
- **Untraceable or wrong numbers.**
  - One key experiment has no script or output.
  - Several percentages are miscopied or untraceable: 14–87%, 15–17%, "roughly 100×" and "±10% noise".
- **Version-5 column.** It is an emulation, not the regression-tested v5 shader.

---

## MUST FIX

### M1. "every unmarked observer is within 0.61% of the exact field" (abstract; also in README)
- **Evidence.** `tests/gpu/tables.mjs` (`ringErr`) records only *aggregate* quantities per 64-point ring:
  - `l2Unflagged = sqrt(Σ|ΔE|² / Σ|E|²)` over the unflagged points;
  - no per-observer error is stored.
- **The 0.61%.** It is the ring L2 for β = 0.9, M_T = 128, R = 3, taper on and off (`far[6..7].rows[1]`). An L2 over 64 points does not bound any single point.
- **The paper's own Table 3 caption** says it correctly: "the largest error over unmarked observers is 0.61%", meaning the L2 over the unmarked set.
- **Change.** Write "in the far-field tests, the relative L2 error over the observers the indicator leaves unmarked never exceeds 0.61%". Alternatively, record a per-observer maximum in `tables.mjs` and quote that. Fix README.md the same way.

### M2. The validity indicator misses large errors. "History reach" in Table 3 is not the history reach.
- **Paper claims.**
  - Abstract: the indicator "marks observers outside the stored history or where it is under-resolved".
  - Table 3 header: "M_T (reach)" = 220 / 880 / 3520.
  - §4.3: the indicator "never left a large error unmarked".
- **Evidence 1: the history is only about 260 units deep.**
  - The ring buffer has `ringSize = 16384`, `dt = 0.00675`, `c = 2.35`, so its depth is c·dt·ringSize = **259.9 length units**.
  - `RingBuffer.sample` clamps `stepsBack` to `RING_SIZE − 3.001`, so every shell with R ≳ 260 holds the *oldest ring sample, frozen*.
  - So for M_T = 512 and 2048, the "reach" of 880 and 3520 is shell placement, not stored history.
  - The shader's out-of-domain test (`fHi <= 0`) only fires beyond the last shell, so it never flags these points.
  - `exactField`'s default `maxLookback` is also 260. That is why the tested rings (R ≤ 200) never exposed this.
- **Evidence 2: GPU test** (production shader, ε = 0, straight-line motion at β = 0.5, observer at distance d perpendicular to the motion):

  | M_T | d = 100 | 200 | 300 | 500 | 800 | indicator q |
  |---|---|---|---|---|---|---|
  | 512 | 2.8e-5% | **3.3%** | **6.9%** | **69%** | **37%** | **0 (all unmarked)** |
  | 2048 | 2.8e-5% | 2.8e-5% | **10%** | **25%** | **62%** | **0 (all unmarked)** |

  - For unaccelerated motion both terms of q vanish: `acc = 0` and `Δu = 0`.
  - So coarse or frozen shells are never flagged, and errors up to 69% are shown as "resolved".
- **Change.**
  - In the paper:
    - state the ring-buffer depth;
    - define "reach" as min(outermost shell, c·dt·ringSize);
    - drop or qualify "marks observers outside the stored history";
    - add this failure mode to the limitations.
  - In the code (recommended): set `quality = −1` when the retarded R exceeds the ring depth, and add a ΔR-based term that does not vanish for straight-line motion.

### M3. Far-field hand-off radius is wrong by a factor of 100 (§5, item 2)
- **Paper:** "would only be valid beyond R ≳ 175 γρ/ε for relative error ε".
- **Evidence.**
  - `validation/analysis/farfield/REPORT.md`: the residual after removing the Fresnel delay is **1.75·γρ/r**, so the 1% radius is 175γρ.
  - My rerun of `fresnel_decompose.py` confirms it: γ = 22.37, r = 1000 gives 3.91e-2 = 1.75 × 22.37 / 1000; γ = 70.71, r = 10⁴ gives 1.23e-2.
  - Hence R ≳ **1.75 γρ/ε** with ε a fraction, which is 175γρ at ε = 1%.
- **Unstated precondition.** The bound applies only *after* subtracting the Fresnel arrival delay. That needs each source's emission point, which fails for superposed sources. Without that correction the 1% radius is ≈ 210 ρ γ³ (same report).
- **No saved output.** `farfield/` has no `*.out` files, unlike the other analysis folders.
- **Change.**
  - Write "R ≳ 1.75 γρ/ε (≈ 175 γρ for 1%), and only after removing the Fresnel delay |z⊥|²/2R; otherwise ≈ 210 ργ³".
  - Commit the script outputs.

### M4. The hyperbolic "targeted run" has no source. The "no accuracy cost" claim is contradicted by repo data. (§4.4)
- **Paper:** "Observers placed so that their retarded points fall in that segment see 21–73% error with the squared-speed certificate, 49–96% with Theorem 1 and 68–91% with no limiter, and the indicator marks all of them. The certificate's role is therefore a guarantee … at no accuracy cost."
- **Evidence: no source.**
  - These numbers exist only as a hard-coded string in `scripts/tables_md.mjs` (line 32), copied into `docs/ACCURACY.md`.
  - No script, JSON or `.out` file produces them.
  - `tables.mjs` (`hyper`) runs a *different* experiment, where all three limiter modes give *identical* results.
  - `docs/ACCURACY.md` says the run used proper accelerations 0.15 and 0.25. The paper mentions only "100 time units".
- **Evidence: repo data contradict "no accuracy cost".** `validation/analysis/limiter/run_modes.out` (v5 algorithm, Float64) shows that in under-resolved cases α_sq is much *worse* than no limiter:

  | case | α_sq | no limiter |
  |---|---|---|
  | γ = 22.4, M = 128, R = 8 | **296%** | 70% |
  | γ = 70.7, M = 128, R = 8 | **916%** | 103% |

- **Change.**
  - Add the script and its output for the targeted run, or remove the sentence.
  - Replace "at no accuracy cost" with "at no cost on resolved histories; on unresolved segments it can raise or lower the error, and such points should be flagged".

### M5. "roughly 100× reduction in error at γ = 22–71" (§4.1)
- **Evidence** (Table 1's own numbers, legacy / this version):

  | γ | M_T = 128 | 512 | 2048 |
  |---|---|---|---|
  | 22.4 | 0.26/0.13 = **2×** | 0.22/0.0095 = **23×** | 0.24/0.0067 = **36×** |
  | 70.7 | 130× | 104× | 108× |

- **Change.** "about 100× at γ = 70.7 and 20–40× at γ = 22.4 for M_T ≥ 512 (2× at M_T = 128)".

### M6. "it fired on 14–87% of segments that were never superluminal" (§2.3)
- **Evidence:** `validation/analysis/limiter/run_limiter.out`, segments with α* < 1:

  | γ | M = 128 | 512 | 1024 | 2048 |
  |---|---|---|---|---|
  | 7.1 | 18/127 = 14% | 37/511 = 7.2% | 53/1023 = 5.2% | 82/2047 = **4.0%** |
  | 22.4 | 28% | 14% | 10% | 7.3% |
  | 70.7 | 87% | 32% | 23% | 19% |

  The range is **4–87%**. "14–87%" holds only at M_T = 128. The limiter REPORT.md makes the same error.
- **Change.** "4–87% (14–87% at M_T = 128)".

### M7. Swarm preset: "15–17% error in β̇" (§6)
- **Evidence.**
  - No saved output contains this number.
  - The bug is real: `legacyPresetKinematics('swarm')` uses `2Ω×v_lab`, which counts Ω×(Ω×r) three times.
  - I measured the legacy acceleration against the finite-difference derivative of its own velocity (β = 0.8, ω = 1.6, all 32 sources, t ∈ [−50, 50]).
  - Result: the RMS relative error is **4.2–14.9%** depending on index (4.2 / 9.9 / 5.4 / 14.9% cycling mod 4), with a pointwise maximum of 15.4%.
  - The unit test only asserts > 5% for index 5, which gives 9.9%.
- **Change.** "4–15% depending on the source". Commit a script that prints it.

---

## SHOULD FIX

### S1. The abstract does not state the error metric
- **The abstract says** "Against the exact field (no softening) … 0.0095% at γ = 22.4 and 0.069% at γ = 70.7".
- **How these were measured.** Observers are rounded to float32, and the reference is evaluated *at the rounded point* (`runCase`, `queryMode: 'f32'`).
- **Why it matters.** At γ = 70.7, rounding the observers alone changes the exact field by 3.2% (Table 1, last column). The γ = 224 numbers (1.3%, 0.45%) are from a *different* metric: df64 observers.
- **Change.** Add "(exact field at the float32-represented observer position; at γ = 224 with double-float observers)".

### S2. The "version-5 algorithm" column is an emulation, not the deposited/regression-tested v5 shader
- **How the column is produced.** `tables.mjs` runs `PRODUCTION_SHADERS` with legacy switches (`newton 2, certMode 0, rootMode 0, precision 0, betaMax 0.99995`). The 154-value regression covers `LEGACY_SHADERS` instead.
- **My comparison** (same history, ε = 0, prefilter off):

  | case | legacy shader | production + switches (Table 1) |
  |---|---|---|
  | γ = 22.4, M = 512 | 0.21724% | 0.21721% |
  | γ = 70.7, M = 128 | 47.53% | 47.61% |
  | γ = 70.7, M = 512 | **7.577%** | **7.185%** |
  | γ = 7.1, M = 2048 | 0.01508% | 0.01495% |

  The arithmetic differs slightly, e.g. `(r1 + dRSub) < length(d)` against `(r1 − dist) + dRSub < 0`.
- **Effect on the paper.** The abstract's "7.2%" for v5 would read 7.6% with the actual v5 shader.
- **Change.** Either regenerate the column with `LEGACY_SHADERS`, or say "v5 algorithm, emulated by switches in the production shader (differs from the deposited shader by up to 5% relative)". §3 item 4 currently implies the regression-tested code was used.

### S3. "removes a single-precision accuracy floor" (abstract and contributions)
- **Evidence.**
  - Table 1 shows a residual plateau: γ = 70.7 goes 0.069% → 0.051% (M_T 512 → 2048), and γ = 22.4 goes 0.0095% → 0.0067%.
  - Over the same range the ring error converges by 15× (70.7) and 5× (22.4).
  - The float64 model floor at γ = 70.7, M = 2048 is 0.0017% (v5 F64 column).
  - The precision ablation (`ablate.out`) shows about 0.03–0.07% left after the df64 fix.
- **Change.** "lowers the floor about 100×, from ≈7% to ≈0.05% at γ = 70.7".

### S4. Cost claim (§4.5)
- **Paper:** the two kernels "take the same time to within the ±10% run-to-run noise".
- **Evidence.**
  - `tables.mjs` times each variant once, over 3 frames, and no repeated runs are stored, so the ±10% has no source.
  - The other cost file in `results/` (`tables_hyper_cost_grids.json`) gives production 2517 ms against legacy 3383 ms. That is production 26% *faster*, against 6% slower in `tables.json`.
  - Production timings differ by 50% between the two files, whether from noise or from code changes.
  - SwiftShader timing says nothing about GPU ALU cost. The repository's own op count (`precision/opcount.out`) estimates **2.7× ALU per pixel** (2373 → 6372 ops).
- **Change.** Report the measured ratios (0.74–1.06 across two runs), cite the 2.7× op-count estimate, and drop "±10%". Also remove or label the stale `results/tables_hyper*.json` files.

### S5. The knot search does not use one texture fetch per knot (§2.5)
- **Paper:** "evaluating f at knots only (one texture fetch each)".
- **Code:** `knotAt` does three `texelFetch` calls: `u_texPosVel` (x_hi), `u_texAccChg` (R in `.w`) and `u_texPosLo` (x_lo). All three are used by `rootFn`.
- **Change.** "three texel fetches, no spline evaluation". Alternatively pack R into the lo texture so one fetch is enough.

### S6. The validity-indicator formula is dimensionally inconsistent with the paper's notation (§2.6)
- **Paper:** q = ΔR γ |β̇⊥| / β², with β̇ = z̈/c defined in §2.1, which has units of 1/time.
- **Shader:** uses `acc` = z̈/c² (see `types.ts`: "Acceleration divided by c²"), so q = ΔR γ/ρ.
- **Change.** Write ΔR γ |β̇⊥| / (c β²) = ΔR γ/ρ. With c = 2.35 the current formula is off by 2.35×.

### S7. Proposition 1: precise statement and arithmetic caveat
- **Formula: verified.**
  - The Bernstein product identity holds to 1.3e-15 on 20 000 random control triples.
  - The shader's `alphaSharp` coefficients match it: D0 = −β₁ − v, D1 = 2v + β₁ + β₂, D2 = −β₂ − v; it solves c₁, c₂ and c₃.
  - In float64 there were 0 speed violations at α_sq in 12 000 random cases at β_max = 0.9, 0.99 and 0.9999.
  - α_sq ≥ α* held in every case.
- **Edge case ‖v‖ = β_max.**
  - Then c_k(0) = β_max², and c₁, c₂ and c₃ can exceed it immediately.
  - Example: c_k(10⁻³) − β² = +1e-4, +5.9e-4 and +4.9e-4.
  - "Smallest α ∈ (0,1] with c_k = β²" may then return 1, which is wrong.
  - The shader handles this correctly (`pp >= 0 → 0`).
  - State ‖v‖ < β_max, or define α_sq = 0 at equality.
- **Tangency.** The proof's "no crossing before α*" should say "no c_k exceeds β² on [0, α*]". A tangential touch c_k = β² before α* is possible in degenerate cases. `firstRoot` would then return α_sq < α*.
- **Float32.** The guarantee holds in exact arithmetic only.
  - I emulated `alphaSharp` in float32 at β_max = 0.9999995.
  - β ≤ 0.99999 (the tested range): no violations in 60 000 synthetic near-circular segments.
  - β = 0.999999: 19% of segments exceed β_max, by up to 4.9e-7 ≈ 1 − β_max, though none reached 1.
  - So "the certificate guarantees [½(1−β_max)] is below f′" (§2.4) is also exact-arithmetic only.
  - Add a sentence noting this and that β_max − β must exceed a few float32 ulps.
- **The "about 4× conservative" claim** (contributions) compares α* with the *exact* limit (bisection on max|z′|) on adversarial random data. My rerun: median α*/α_exact = 0.257. α_sq itself is about 2× conservative (median α_sq/α_exact = 0.463). The paper never defines α_exact. Say what the 4× is relative to and on which data.

### S8. "All numbers come from scripts in the repository (tests/gpu/tables.mjs)" (§3)
This is false for several groups of numbers:

| numbers | where they actually come from |
|---|---|
| precision attribution (12.4%, 2.4%, ≤0.05%, 15 ulps, 71%) | a **CPU float32 emulation of the v5 algorithm** (`validation/analysis/precision/core.js`), not the GPU and not the production code; the paper should say so |
| limiter statistics | `validation/analysis/limiter` |
| far-field estimates | scripts in `validation/analysis/farfield`, with no saved outputs |
| PyCharge | `validation/pycharge` |
| hyperbolic targeted run | nowhere (see M4) |
| swarm error | nowhere (see M7) |

`validation/analysis/README.md` itself says those reports are "working notes … superseded where they differ". Replace the sentence with an accurate source list.

### S9. Scope of the PyCharge cross-check (abstract, §3 item 1)
- **What was compared.** `validation/pycharge` compares PyCharge against `reference_port.py`, a Python port of the *v5 JS* `evaluateExactAnalyticalLW`. It does not test `src/core/reference.ts`, which produced the tables.
- **Indirect evidence for TS = v5.** The 154-value regression includes `cpuF64L2` and matches exactly. That is good, but it is indirect.
- **Range.** The pulse sweeps go only to β = 0.99 (γ = 7.1). The reference at γ = 22–224 is not externally checked; the error grows as κ⁻³.
- **Change.** Say "a Python port of the reference … up to γ = 7.1".

### S10. Sweep half-width at γ = 224 (§3 item 2)
- **Code:** `pulseCrossingSweep` uses `max(1e-6, 4γ⁻³)`.
- **At γ = 224,** 4γ⁻³ = 3.6e-7, so the half-width is **1e-6 rad**, 2.8× wider than stated.
- **Change.** State the clamp.

### S11. The taper is "off by default" only in the app (§4.3)
- **App:** `app.ts` sets `prefilter: false` and `softEps: 0.02`.
- **Library:** `DEFAULT_CONFIG` has `prefilter: true` and `softEps: 0.06`. The README library example (`{...DEFAULT_CONFIG, …, softEps: 0}`) therefore runs with the taper on.
- **Change.** Either change `DEFAULT_CONFIG`, or write "off by default in the app".

### S12. Double-float observers exist only in the readback API (§2.4, §5 item 3)
- **Code:** in render mode the shader sets `xlo = 0` and builds `xhi` from `v_uv` in float32.
- **Effect.** The interactive image cannot use df64 observers. Above γ ≈ 100 the displayed pulse is therefore not reliable.
- **Change.** Say "(library readback API)".

### S13. "Changes from version 5" items not fully supported
- **"three far-field cells were not computed by the deposited suite".**
  - The deposited suite computes only `compFar45` (stereographic, 1.62%) and `unifFar45` (113.9%).
  - The v5 far column has four cells. Only **two** (quadratic 1.58% and log 1.60%) are missing.
  - Say "two", or explain the third.
- **"max |Δz| column does not reproduce".** Plausible: `posErr` in `baseline_suite.json` is ~1e-7, against 1.8e-5 printed. But no saved output makes the comparison, and `posErr` is a different quantity (bulk-ring emission-point error). Add a pointer or soften.
- **"The 6–9% GPU-versus-CPU discrepancy … is reproduced by legal reorderings … and is not a hardware effect".**
  - `gpucpu.out` gives 5.96 / 7.27 / 5.69%, against 5.97 / 7.82 / 8.70% printed. The magnitude matches but the values are not reproduced (2048: 5.7 vs 8.7).
  - It shows the discrepancy is *explainable* by rounding. It cannot exclude hardware.
  - Write "of the same size as … so it need not be a hardware effect".
- **Supported as written:**
  - ε = 0.06 in the v5 reference (PyCharge REPORT);
  - quadratic 4.4% vs 0.21% (`baseline_suite.json` `pow128Reach220` = 4.42%);
  - β = 0.999, M = 128: 0.25% vs 2.47% (0.245%);
  - far 1.62% from the `uniform` preset at 0.44c, jumping every 4.4 units (`presets.ts`: 0.55·0.8; period 4.4);
  - v5 Float64 column had no limiter (`core_orig.js` `evaluateCompactifiedFloat64` has no α);
  - the limiter caused most of the γ = 70.7, M = 128 error (46.8% vs 7.05% F32; 46.2% vs 0.39% F64);
  - the swarm defect exists (magnitude: see M7).

### S14. Prior-work and novelty framing (§1)
- **PyCharge** is described as "in batch Python". PyCharge 2.x (the version used, 2.0.1 on JAX) is JIT-compiled and runs on GPUs through JAX. The contrast with "a GPU kernel" is therefore weaker than written.
- **"a fixed-cost … kernel".** Newton stops early at convergence (up to 12 steps). Use "bounded-cost".
- **"efficient when sources are few and observation points many".** No GPU timing exists to support "efficient". Use "suited to".

### S15. Acknowledgments contain unresolved TODOs
- The comments in lines 246 and 248 are hidden in the PDF.
- Line 248: "Astra" is not identified.
- Line 246: the Gemini model version is missing.
- Resolve both before publication.

### S16. References
- **zenginoglu2008: title and number do not match.** CQG **25 145002** is "Hyperboloidal foliations and scri-fixing". "Hyperboloidal evolution with the Einstein equations" (arXiv:0808.0810) is a different 2008 paper; I believe it is CQG 25 195025. Pick one and make the title and number agree.
- **huang2021 (CoSyR): incomplete.** The citable paper is C.-K. Huang, F.-Y. Li, H. N. Rakotoarivelo, B. Shen, J. Domine, B. Carlsten, G. Dilts, R. Garimella, T. Kwan, R. Robey, "CoSyR: a novel beam dynamics code for the modeling of synchrotron radiation effects", arXiv:2109.14901, published in NIM A (DOI 10.1016/j.nima.2022.166808). The current entry has a partial author list and no year or venue.
- **Verified:** brentner1997 (JSV 203(1), 87–100, 1997).
- **Consistent with my knowledge but not web-verified here:** vay2018, pycharge2022, filipovich2021, fritsch1980 and bishop1997.
- **shintake2003 (NIM A 507, 89–92):** I could not confirm it. Please check.
- **Minor:** MINVO (tordesillas2020) is an *alternative* to Bernstein enclosures, published in IEEE T-RO (2022). A Bernstein velocity-constraint planner (e.g. BeBOT, Kielas-Jensen & Cichella) fits the sentence better.
- **Minor:** for "established Cauchy-characteristic extraction", the canonical reference is Bishop, Gómez, Lehner & Winicour, PRD 54, 6153 (1996).
- **Not verifiable from here:** the GitHub URL. The local copy is not a git checkout. Make sure the repository is public before publishing. The CITATION.cff DOI also needs checking.

---

## NITS

1. **Table 3 caption.** Say the values shown are taper-off. Taper on differs, e.g. M = 512, R = 30: 36% vs 31%.
2. **Table 1, γ = 224, "this version" column.**
   - The data exist: 5.0, 0.98 and 0.50%.
   - By the paper's own argument ("exact field at the rounded point, which is the metric used here") they are meaningful, and at M = 512 better than df64.
   - Show them, or explain why the dashes are there. Many of the 256 points collapse to the same float32 coordinates.
3. **Indicator discrimination is weak.** For β = 0.5, M = 2048, R = 30, the flagged subset has *lower* error (0.097%) than the unflagged one (0.21%). Worth a sentence next to "conservative".
4. **C¹ claim.** "requires c ≤ 2β/3" was established by numerical optimization over 8 segments (`e5_c1.out`), not proved. "Raising the degree restores it" is likewise numerical. Say so.
5. **§5 item 2.** "the worldline is close to the most compact representation" is a heuristic argument. Soften it.
6. **§4.1.** "the remaining error … converges with M_T" holds for this version only. The legacy column worsens from M = 512 to 2048 at γ ≤ 7.
7. **Root-search corners.**
   - The segment fraction is capped at 0.9999 (`sHigh`, `sampleSegAt`).
   - The 12-step knot bisection limits M_T ≤ 4097, which is fine with `MAX_M_T = 2048`.
   - Roots inside the first shell are evaluated at knot 0. They are flagged with q ≥ 1.6, consistent with the text.
8. **Stale files.** `results/tables_hyper.json` disagrees with `tables.json` (e.g. flagged 0 vs 10 for 0.5/60/M=128) and should be deleted or labelled.
9. **Hyperbolic description (§4.4).** γ ≈ 11 is 10.7 in the data. "Over 100 time units" is proper acceleration 0.25 for 100 units. Fine, but name the acceleration.

---

## Verified as stated (no action needed)

- **Abstract numbers and tables against `results/tables.json`:**
  - abstract: 0.0095, 0.069, 0.22, 7.2, 1.3, 0.45;
  - Table 1: all 18 rows × 7 columns;
  - Table 2: all 16 cells;
  - Table 3: all 18 cells and flag counts (β = 0.9, taper off).
- **Precision section against `precision/ablate.out` and `condition.out`:** 12.4%, 2.4%, other stages ≤ 0.05%, about 15 ulps, 71%, 3.2%, δ ≈ 6e-8, and "rounding ~3e-5 > 1 − β at M = 2048".
- **Code matches the text:**
  - `kappaFloor()` = ½(1 − β_max) in production, against 5e-5 in legacy;
  - Newton is capped at 12 steps (`DEFAULT_CONFIG.newton`) with bisection fallback and a separate index and fraction;
  - the chord clamp uses hi + lo in double precision (`shells.ts` lines 144–160), with the legacy float32 path kept;
  - hi/lo knot textures (`u_texPosLo`), chords formed from both parts, and df64 `rootFn` guarded by `u_one`;
  - knots are sampled at the float32 shell radius, so R stored in float32 is consistent;
  - the cancellation-free κ and n̂ − β formulas and the LW kernel (`types.ts`: bdx = z̈/c²);
  - the validity thresholds q > 1.5 or q < 0 (`app.ts`), the first-shell rule (1.6), taper flagging (2.0) and q = −1 beyond the last shell.
- **Proposition 1:** the formula and α_sq ≥ α* hold numerically. The shader implements exactly the three-quadratic first-root rule. The edge cases are noted in S7.
- **Theorem 1:** `alphaLegacy` matches the statement. The control vectors in Eq. 4 are consistent with the Hermite derivative.
- **Outermost shell** L tan(π/2 − Δθ/4) ≈ 4LM/π gives 220 / 880 / 3520 (`tables.json` `rMax`).
- **Regression:** I reran it; 154/154 values are identical.
- **Other numbers:**
  - PyCharge agreement ≤ 1.3e-11 (REPORT, β = 0.99 sweep);
  - hyperbolic peak ‖z′‖ = 1.0573 at R ∈ [73.3, 220];
  - |V₁| = β(1 + φ²/8) (`e1_circle.out`);
  - the raw cubic is never superluminal on the synchrotron benchmark (`e2_knots.out`, F64).
