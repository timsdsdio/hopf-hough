# Branch C — Worldline → (τ, S²) radiation-accumulator hand-off

## 1. Claim (as tested)
*Hypothesis:* For a relativistic point charge, handing off at R ≈ R_Nyq from the stored worldline to an accumulator of r·E_rad on (u, n̂) ∈ ℝ × S² reproduces the far-field pulse to ≤1%, using less memory than the worldline buffer.
**Result: refuted on both counts.** The hand-off must sit about 200× further out than R_Nyq, and the accumulator needs roughly γ³·N_dir times more storage than the worldline it replaces.

## 2. Why it matters / who cares
If it had worked, this would be a bounded-memory, dispersion-free far-field engine. Venues: J. Comput. Phys., Comput. Phys. Commun., Phys. Rev. Accel. Beams. As it stands, the useful output is a negative result. §5 of the preprint should be corrected so that a referee does not find this first.

## 3. Nearest prior art, and the delta
- **CCE/CCM** (an inner worldtube plus an outer characteristic evolution out to scri): Bishop et al. [gr-qc/9705033](https://arxiv.org/pdf/gr-qc/9705033); SpECTRE CCE, Moxon et al. [2110.08635](https://arxiv.org/pdf/2110.08635); full CCM [2308.10361](https://arxiv.org/html/2308.10361v2); model-problem convergence [2306.13010](https://ar5iv.labs.arxiv.org/html/2306.13010). §5 is CCE with a Liénard–Wiechert interior. In flat vacuum the exterior evolution is trivial, so the only new piece would be the interior, and that interior already exists (it is the preprint).
- **Hyperboloidal compactification**, including Maxwell and spin-2 fields up to scri: Zenginoğlu et al. [1004.0760](https://arxiv.org/pdf/1004.0760); Frauendiener et al. [1207.5854](https://arxiv.org/pdf/1207.5854). These are Eulerian PDE solvers on hyperboloidal slices. Making §5 literally "hyperboloidal" means adopting one of them, which brings back the γ⁻³ grid-resolution problem.
- **Far-field accumulators for many charges**: PIConGPU radiation plugin and Clara2, Pausch et al. ([IPAC2014](https://jacow.org/IPAC2014/papers/MOPRI069.pdf), [Clara2](https://github.com/computationalradiationphysics/clara2)). They accumulate the Liénard–Wiechert far field on an (ω, n̂) detector grid, summed over particles on the GPU. **This is the many-source version of the idea, and it already exists.**
- **Time-domain near-to-far-field transforms** in FDTD (Luebbers 1991, [NASA](https://ntrs.nasa.gov/api/citations/19920010500/downloads/19920010500.pdf)) and **Kirchhoff/FW–H surfaces** in aeroacoustics ([Lyrintzis](https://ntrs.nasa.gov/api/citations/19940031524/downloads/19940031524.pdf)): a near-zone solution handed off to a surface integral or far-field record. This is the same architecture, used for decades.
- **Synchrotron near-field codes**: SRW ([ESRF](https://www.esrf.fr/Accelerators/Groups/InsertionDevices/Software/SRW)) handles finite-distance synchrotron radiation, because the Fraunhofer approximation fails there (see §5 below). Wavefront Liénard–Wiechert: CoSyR ([2109.14901](https://arxiv.org/abs/2109.14901)).

**Delta:** I can't find a defensible novel piece. A time-domain, direction-binned scatter accumulator is a minor variant of Pausch et al.

## The Hopf question (plain language)
Gemini's claim is **half right and doesn't matter computationally.**
- *Right:* a transverse field on the sphere of directions cannot be described by two components in one global frame. The hairy-ball theorem rules out any smooth, nowhere-vanishing frame. The polarization lives in the tangent bundle TS², whose "twist number" (Euler class) is 2.
- *Overstated:* the bundle that matches this is the unit tangent (frame) bundle SO(3) ≅ RP³, not the Hopf bundle S³ → S² (twist number 1). Hopf double-covers it. Spin-weight ±1 quantities (E_θ ± iE_φ, Newman–Penrose) can be written as functions on SU(2) = S³, but they are even under −1, so they already live on SO(3). Hopf is genuinely needed only for half-integer spin (spinors). See Boyle, [1604.08140](https://arxiv.org/pdf/1604.08140).
- *Kills the geometric angle:* store E as a **Cartesian 3-vector** with n̂·E = 0. TS² sits inside the trivial bundle S² × ℝ³, so no frame and no bundle are needed. After interpolation, project out the radial part. That costs 3 floats instead of 2 and has no singular points. Spin-weighted harmonics matter only if you want a spectral (Y_lm) representation, and for a γ = 22 beam that needs l_max ~ γ³.

## 4. Minimum viable experiment (already run, below)
Synchrotron orbit in 3D (ρ = c = 1). (a) Exact Liénard–Wiechert r·E at x = r n̂, t = u + r, compared with the scri data F(u, n̂), as a function of r and γ. (b) Interpolate F between direction pixels at fixed u. (c) Count memory per orbital period. A 2D prototype is not a meaningful test: in 2D the Green's function has a tail, so there is no clean scri.

## 5. What I actually checked (`./`)
- `null_ray_stationarity.py`: r·E does converge to F along outgoing null rays, but slowly. **The 1% radius is about 210·ρ·γ³** (γ = 3.2, 7.09, 22.4 give r/γ³ = 229, 210, 212). At γ = 22 that is r ≈ 2.4×10⁶ ρ.
- `fresnel_decompose.py`: most of that error is the Fresnel arrival delay |z⊥|²/2r, which is about 560 pulse widths at r = 10 for γ = 22. Removing it requires knowing each source's emission point, which fails once sources are superposed. Even after removing it, the residual is **1.75·γρ/r** (constant to 1% across γ = 7, 22, 71), which is parallax. So **R_h(1%) ≳ 175·γ·ρ ≈ 3300 paper units at γ = 22, against R_Nyq ≈ 10–16.** The §5 hand-off radius is too small by about 200–300×.
- `s2_resolution.py` (γ = 22.37): interpolating in elevation with Cartesian vectors plus reprojection is second order. Δψ = 1/(4γ) gives 2.3%, and 1/(8γ) gives 0.59%. **Interpolating in azimuth at fixed u fails** (error ≥ 100%) until Δφ ≲ 0.1·γ⁻³. The reason is that neighbouring directions see the same pulse shifted in time. Shift-aware interpolation reaches 5×10⁻¹² error, but only because of the orbit's symmetry: in general it needs the emission point, which means the worldline. **Memory per period:** a smooth worldline needs about 512 samples (12 kB). The accumulator needs about 2.8×10⁵ u-samples per direction (γ³ Doppler decompression). That is ~30 TB for an interpolable equatorial belt and 3.4 GB for a fixed detector of 10³ directions.
- `hopf_check.py`: the clutching winding is 2 for TS² and 1 for spinors/Hopf. The spin-1 lift satisfies f(−U) = f(U), so it descends to SO(3); spin-½ flips sign. Cartesian storage plus reprojection gives O(h²) error, the same at the poles as at the equator.

**Core insight:** the worldline *is* the compressed representation of the far field. The far field is the worldline after γ³ Doppler compression in u and 1/γ beaming in angle. Storing it on (u, S²) undoes that compression. The ring buffer in t_ret is already characteristic and costs nothing to append.

## 6. Kill criteria (status)
- R_h(1%) ≫ R_Nyq → **met (killed)**.
- Accumulator memory > worldline memory for one charge → **met (killed)**, by γ³·N_dir.
- Bundle machinery beats Cartesian transverse vectors → **met (killed)**.
- Remaining open case: N_p ≳ N_dir·γ³/(~20) sources, observed only at fixed far detectors. That regime is exactly what Pausch et al. already cover.
- Radiation reaction (local ALD/LL), coherent synchrotron radiation (CSR: near-field, inside the formation length), media and boundaries (the field is no longer freely outgoing, so you need a PDE): the scri accumulator gives nothing in any of these.

## 7. Verdict
**Dead as a standalone paper** (confidence ~80%). The only surviving version, a time-domain GPU detector accumulator for many charges, is **weak**, because Pausch et al. already do it in the frequency domain. Salvage: (i) rewrite §5 of the preprint to say the hand-off requires the Fraunhofer zone (R ≳ 175γρ/ε) and that the worldline buffer is near-optimal storage; (ii) delete "natural bundle representation", or replace it with "Cartesian transverse vectors, or spin-weight ±1 functions if a spectral basis is wanted". The γ³ decompression and Fraunhofer hand-off scaling could make a short note or an appendix, but not its own paper.
