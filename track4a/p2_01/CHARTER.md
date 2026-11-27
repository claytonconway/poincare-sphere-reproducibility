# Stream Charter: p2_01_discrete_photon_pole_sum

**Title:** Discrete S^3/I* Two-Photon Spectral Pole Sum & Horowitz Pinch-Off Shift (CLM-OPEN-01)
**Graduated:** 2026-11-25
**Status:** verified

## 1. Core Hypothesis
Replacing Track IV-A's single-threshold continuum-log kernel above M_{I*}(2) = 26 with the exact discrete S^3/I* 2-photon spectral pole sum (summed over all opposite-chirality I*-invariant curl eigenmode pairs with exact Wigner-3j/triple-overlap weights and Hadamard/zeta UV subtraction) shifts the chiral Horowitz pinch-off beta_pinch(S^3/I*) quantitatively from 0.3836 and introduces narrow discrete sideband resonances without reopening the sub-threshold gap |Omega| < 26.

## 2. Prior-Art Triage (arXiv / OpenAlex / Textbooks)
- Horowitz (1980), Phys. Rev. D 21, 1445 — Semiclassical stability of flat space & nonlocal Weyl kernel pinch-off.
- Starobinsky (1981) / Mazur & Mottola (2001) — Conformal anomaly effective action and nonlocal C ln(-Box/mu^2) C form factor.
- Dowker (2004), arXiv:hep-th/0404093 — Exact Maxwell & spinor degeneracies and spectral geometry on S^3/I*.
- Track IV-A baseline: open_problems/track4_nonlocal_weyl_kam/T4A_Nonlocal_Weyl_KAM_Report.md (CLM-T4A-01, CLM-T4A-02, CLM-OPEN-01).

## 3. Explicit Kill-Tests (Attempts to Falsify)
- KT1 (Analyticity & Chiral Gap): Exact opposite-chirality selection (1 - s_1*s_2) preserves Im Pi_discrete(Omega) == 0 identically for all |Omega| < 26 and recovers exact M_{S^3}(2)=6 on round S^3.
- KT2 (Continuum & Weyl-Anomaly UV Recovery): High-frequency average of the discrete S^3/I* spectral density recovers the exact 4D conformal Weyl-anomaly coefficient c_anom = 9/31 (and continuum-log slope) as Omega -> infinity.
- KT3 (Cutoff Convergence & Pole-Repulsion Shift): Track beta_pinch(N_cut) and UHP/real Stelle roots across spectral cutoffs N_cut (e.g. 60, 100, 160, 240) plus dual-precision (float64 vs mpmath) check to confirm quantitative convergence and quantify shift relative to 0.3836.
- KT4 (Dynamical Volterra vs Discrete-Pole Sidebands): Run unreduced dynamical evolution on P05 x_ring and x_b=15 across Regime IV beta_0 in [0.44, 6.37] to test whether discrete 2-photon poles above Omega=26 destabilize any narrow sub-window via scalaron sidebands (Omega_grav + k*omega_scal).

## 4. Verification & Domain-of-Validity Contract
- [x] Analytical derivation + dimensional / limiting-case checks
- [x] Un-clamped conservation / Ward / Bianchi / Gauss residuals logged in `results.json`
- [x] Independent dual-solver or high-precision oracle cross-check
- [x] Matched negative control (if attributing an effect to a specific mechanism)
- [x] EFT / UV / truncation boundaries disclosed explicitly
- [x] Cold `referee-critic` audit passed (0 P0, 0 P1)
- [x] Claims promoted to `ledger/claims.json` (`PROVEN_ANALYTIC | VERIFIED_NUMERIC | CAVEATED | RETRACTED`)
