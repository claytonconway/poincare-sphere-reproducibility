# Stream Charter: p2_02_nonlinear_kam_backreaction

**Title:** Nonlinear Scalaron-Radiation-Graviton Backreaction on the S^3/I* KAM Island (CLM-OPEN-02)
**Graduated:** 2026-11-25
**Status:** completed (Round-4 cold `referee-critic` PASS: 0 P0 / 0 P1; `physics_audit_results` PASS 4/4 KT; `CLM-OPEN-02` promoted to `ledger/claims.json` as `CAVEATED`)

## 1. Core Hypothesis
Coupling nonlinear scalaron condensate depletion (kappa_scal = 960.7 = 1.997 |kappa_V|), n=2 shear amplitude, and n=18,22 Mathieu-reheated inhomogeneous TT gravitons back into the self-consistent Starobinsky-Casimir Friedmann constraint E_00 = 0 determines whether the S^3/I* KAM breathing island exits to radiation-dominated expansion or relaxes into a damped cyclic attractor — and identifies the exact phase boundary across both sigma_t0 = 9/31 and the sub-unitarity EFT window sigma_t >= 4.

## 2. Prior-Art Triage (arXiv / OpenAlex / Textbooks)
- Starobinsky (1980), Phys. Lett. B 91, 99 & Vilenkin (1985), Phys. Rev. D 32, 2511 — Scalaron decay and reheating in trace-anomaly cosmology.
- Pelinson, Shapiro & Takakura (2003), Nucl. Phys. B 648, 417 — Stability of Starobinsky inflation with anomaly-induced action.
- Track III baseline: papers/track3_prd_semiclassical_poincare/main.tex (CLM-T3-02, CLM-T3-03).
- Track IV-A baseline: open_problems/track4_nonlocal_weyl_kam/T4A_Nonlocal_Weyl_KAM_Report.md (Manley-Rowe scalaron depletion & n=18,22 Mathieu gravitons; CLM-OPEN-02).

## 3. Explicit Kill-Tests (Attempts to Falsify)
- KT1 (Un-Clamped Friedmann & Covariant Energy Conservation): Enforce and log exact E_00 = 0 residual and total radiation + scalaron + graviton covariant energy balance without silent clamping across tau in [0, 10000].
- KT2 (Dual-Integrator Cross-Check): Verify DOP853 vs Radau agreement on both exit-time / attractor-amplitude observables and radiation energy density rho_rad(tau).
- KT3 (Sub-Unitarity EFT Regime Control): Compare dark-gauge sigma_t0 = 9/31 (4*c_anom*omega_scal^2 = 4.0) against sub-unitarity sigma_t in [4, 12] to separate true 1-loop EFT physics from trans-unitarity scalaron artifacts.
- KT4 (Zero-Dissipation & Decoupled Control): Setting radiative coupling / Mathieu transfer to zero must recover Track III / P05 adiabatic KAM island (x_ring = 21.9168, drift <= 1e-5) to machine/integrator tolerance.

## 4. Verification & Domain-of-Validity Contract
- [x] Analytical derivation + dimensional / limiting-case checks
- [x] Un-clamped conservation / Ward / Bianchi / Gauss residuals logged in `results.json`
- [x] Independent dual-solver or high-precision oracle cross-check
- [x] Matched negative control (if attributing an effect to a specific mechanism)
- [x] EFT / UV / truncation boundaries disclosed explicitly
- [x] Cold `referee-critic` audit passed (0 P0, 0 P1)
- [x] Claims promoted to `ledger/claims.json` (`PROVEN_ANALYTIC | VERIFIED_NUMERIC | CAVEATED | RETRACTED`)
