# Stream Charter: p2_03_graviton_casimir_null_cone

**Title:** 1-Loop Pure-Graviton Casimir Energy and Null-Cone Focusing Consistency Across Spherical Space Forms S^3/Gamma
**Graduated:** 2026-11-26
**Status:** verified

## 1. Core Hypothesis
Across all 5 nontrivial isotropy-irreducible ADE spherical space forms (RP^3, S^3/Q_8, S^3/T*, S^3/O*, S^3/I*), the UV-finite 1-loop pure-vacuum-GR graviton topological Casimir stress-energy Delta C_top^EH(Gamma) (evaluated after full gauge-fixed TT + scalar/vector + Faddeev-Popov ghost and Killing-vector zero-mode accounting) is strictly positive (focusing) on S^3/I* (preliminary TT value pi^2 Delta C_top^{EH-TT} = +129.0613 > 0, reinforcing Chandrasekaran arXiv:2611.34924 null-cone focusing positivity Xi_y >= 0 with on-shell dS_4/I* area-ceiling tightening Delta s_* = -2.3365 and dark-photon flip threshold N_V >= 5), whereas anisotropic S^1-pinched lens spaces L(p,1) and binary dihedral quotients S^3/D_m* flip to negative (defocusing) pure-graviton Casimir null energy along the Hopf fiber above a sharp critical order p_crit / m_crit, delineating a precise geometric boundary for 1-loop closed-topology compatibility with Chandrasekaran's vacuum causal-diamond focusing bound.

## 2. Prior-Art Triage (arXiv / OpenAlex / Textbooks)
- V. Chandrasekaran, 'An algebraic quantization of gravity from the null cone', arXiv:2611.34924 [hep-th] (Nov 2026): pure vacuum 4D Lorentzian null-cone quantization; finite-dimensional causal-diamond Hilbert space rests on Xi_y >= 0 (Eqs. 4.22, 5.21c--d).
- J. S. Dowker, 'Spherical universe topology and the Casimir effect', Class. Quantum Grav. 21 (2004) 4247--4272 [hep-th/0404093]: exact conformal Casimir energies on S^3/Gamma for spins 0, 1/2, 1 via polyhedral degrees (omits physical Einstein-Hilbert spin-2 gravitons and null-cone focusing).
- J. S. Dowker, 'Arbitrary-spin theory in the Einstein universe', Phys. Rev. D 28 (1983) 3013: conformal arbitrary-spin (including spin-2) fields on round R x S^3 (no S^3/Gamma quotients or non-conformal EH dispersion omega_p = sqrt(p^2 - 1)).
- S. M. Christensen & M. J. Duff, 'Quantizing gravity with a cosmological constant', Nucl. Phys. B 170 (1980) 480--506; M. B. Altaie, Phys. Rev. D 65 (2002) 044028: covariant 1-loop graviton + ghost determinant structure on maximally symmetric spaces.
- D. Anninos, F. Denef, Y. T. A. Law, & Z. Sun, 'Quantum de Sitter horizon entropy from quasicanonical bulk, edge, sphere and topological string partition functions', JHEP 01 (2022) 088 [arXiv:2009.12464]: Killing-vector zero-mode handling and bulk-edge factorization in 1-loop dS_4 gravity (round S^4/dS_4 only, not dS_4/Gamma quotients).

**Hypothesized Delta:** First gauge-complete 1-loop pure-vacuum-GR graviton topological Casimir calculation across all 5 ADE isotropy-irreducible spherical space forms (RP^3, Q_8, T*, O*, I*) and anisotropic lens/dihedral towers (L(p,1), S^3/D_m*), directly classifying which closed 3-topologies reinforce vs obstruct Chandrasekaran's (arXiv:2611.34924) causal-diamond null-cone focusing positivity Xi_y >= 0 at 1 loop, alongside the exact dark-photon flip threshold N_V^crit(Gamma) and non-abelian SU(2) flat-holonomy switch (rho_0, rho_1 <-> rho_2).

## 3. Explicit Kill-Tests (Attempts to Falsify)
- KT1 (Gauge-Completeness & Zero-Mode Sign Test): Verify whether including off-shell longitudinal/trace metric modes, Faddeev-Popov vector ghosts, and the Killing-vector zero-mode mismatch (6 Killing vectors on S^3 vs 3 on S^3/I*) alters the sign or O(1) magnitude of Delta C_top^EH(S^3/I*) relative to the pure physical TT sector (+129.0613).
- KT2 (Independent Regulator & Integrator Oracle): Cross-check every space-form Casimir shift Delta C_top^EH(Gamma) between 50-digit Hurwitz-zeta + digamma analytic continuation and independent exponential heat-kernel / Abel-Plana summation (tolerance < 1e-10), and verify on-shell dS_4/Gamma Raychaudhuri integrals with DOP853 vs Radau (tolerance < 1e-9) plus exact round-S^3 dS_4 (Xi === 0) and flat-space (a -> inf) limits.
- KT3 (Negative Controls & Anisotropic Fiber Tensor Test): Compute the exact anisotropic stress tensor T_{uu}^grav(theta) (Hopf-parallel vs Hopf-transverse) on L(p,1) and S^3/D_m* (checking S^1 flat-circle limit u_grav -> -pi^2/(45 L_min^4) as p -> inf) and compare against flat cubic T^3 to isolate what is genuinely spherical-space-form-specific vs universal 4D conformal dimensional analysis.
- KT4 (EFT / Species-Scale & Matter-Coupled Boundary): Verify that for all 5 ADE space forms and for N_V dark photons or Ad(rho_i) flat SU(2) connections (rho_0, rho_1, rho_2), Chandrasekaran's finite-dimensionality bound dim H_D < inf remains intact throughout the EFT-controlled regime a >> a_Cas (finite bounded shift Delta s_*), and map the exact breakdown boundary where |Xi| ~ O(1) intersects CLM-T3-01 (L_min / l_sp^red < 1).

## 4. Verification & Domain-of-Validity Contract
- [x] Analytical derivation + dimensional / limiting-case checks
- [x] Un-clamped conservation / Ward / Bianchi / Gauss residuals logged in `results.json`
- [x] Independent dual-solver or high-precision oracle cross-check
- [x] Matched negative control (if attributing an effect to a specific mechanism)
- [x] EFT / UV / truncation boundaries disclosed explicitly
- [x] Cold `referee-critic` audit passed (0 P0, 0 P1)
- [x] Claims promoted to `ledger/claims.json` (`PROVEN_ANALYTIC | VERIFIED_NUMERIC | CAVEATED | RETRACTED`)
