# Research Stream Charter: `p2_07_complete_model_b_kam_dynamics`

**Stream ID:** `p2_07_complete_model_b_kam_dynamics`  
**Target Claim ID:** `CLM-P2-07` (`CAVEATED`)  
**Parent Claims:** `CLM-T3-01`, `CLM-T3-02`, `CLM-T3-03`, `CLM-T4A-01`, `CLM-T4A-02`, `CLM-P2-01`, `CLM-OPEN-01`, `CLM-OPEN-02`, `CLM-P2-06`  
**Graduated From:** `sandbox/sbx_06_lane4_serendipity/`  
**Date:** 2026-11-27

***

## 1. Scope & Physical Framing (Non-Negotiable Epistemic Boundary)

This stream is a **formal large-`N_V` semiclassical nonlinear dynamical-systems study** extending `p2_02` (`CLM-OPEN-02`) with the exact topological and discrete-cavity physics established in `p2_06` (`CLM-P2-06`). Per `CLM-T3-01`, pure dark conformal gauge bosons on $S^3/I^*$ obey the `N_V`-invariant Dvali species-scale squeeze $L_{\min}(a_{\mathrm{Cas}})/\ell_{\mathrm{sp}}^{\mathrm{red}} = 0.4551 < 1$; accordingly, all P05-ring ($x_{\mathrm{ring}} \approx 21.92$) nonlinear backreaction results are explicitly framed as a formal large-`N_V` semiclassical dynamical system rather than a species-scale-safe UV effective field theory. Within this framework, exact analytical identities (dissipative Noether closure, Weyl invariance of the $n=2$ Casimir/bath action, and the Berger-squashed `M_R`-splitting law) are strictly separated from numerical nonlinear ODE integrations and honest kills.

## 2. Core Hypotheses & Kill-Test Charter (KT1--KT4)

1. **KT1 (Exact Lapse-Covariant Complete Model B Action, Shifted $n=2$ Mathieu Wall, `cand_0` Audit & Chiral Self-Quenching Ghost-Wall Stress Test):**
   - *Hypothesis:* Upgrading `p2_02`'s first-order lapse-covariant 5-mode action with `p2_06`'s Complete Model B $n=2$ Casimir kinetic remainder $\Pi_{\mathrm{kin}}$ (`3900/31` rel-`S^3`; `4344/31` abs MS) and static shear remainder $\Pi_{\mathrm{rem}}$ (`984896/465 = 2118.0559` rel-`S^3`; `2066.1264` abs MS) preserves exact dissipative Noether closure $\frac{d}{d\tau}(x^4 E_{00}) \equiv 0$, shifts the nonlinear $n=2$ 2:1 Mathieu wall into the sub-unitarity EFT window $\tilde\sigma \in [4, 12]$, narrows (but preserves) a unified 5-mode survival sub-desert above $\tilde\sigma = 4.20$, and stress-tests both `cand_0` (whether $x \to x_{\mathrm{kin}}$ kinetic chirping acts as a passive Fresnel detuner) and `p2_02`'s $\tilde\sigma_0 = 9/31$ chiral self-quenching attractor against the dynamical kinetic ghost wall $A_{R,2}(\tau) = A_R(\tau) - \Pi_{\mathrm{kin}}/x(\tau)^2 = 0$.
   - *Kill Criteria:* Verify $\frac{d}{d\tau}(x^4 E_{00}) < 10^{-30}$ at 35 digits in `mpmath`; map exact linear Floquet tongues and nonlinear 5-mode survival across $\tilde\sigma \in [3.5, 16.0]$; test `cand_0` unconfoundedly against an orbit-mean-frequency-matched constant-`k_2^2` control (logging honest kill if blowup is not capped); test chiral self-quenching under BOTH `p2_02`'s bare rate table and Complete Model B's self-computed shrinking-ring Parseval/MR rate table.

2. **KT2 (Exact Coherent-Talbot-to-Markovian Phase Transition $\gamma_{\mathrm{deph}} \cdot (\pi/2)$ in Self-Consistent Coupled Discrete-Pole Bath):**
   - *Hypothesis:* Coupling the explicit $J_{\mathrm{cut}} = 30$ (cross-checked at $J_{\mathrm{cut}} = 60$) discrete $S^3/I^*$ opposite-chirality 2-photon auxiliary oscillator bath ($E_j = 4j + 2$, $j = 6..J_{\mathrm{cut}}$, exact weights $w_j = 240 d_{I^*}(2j)/(2j+1)$) with tunable Lorentzian line-broadening $\gamma_{\mathrm{deph}} \ge 0$ directly into the nonlinear Starobinsky + Complete Model B background via an exact positive-definite, Weyl-invariant Caldeira--Leggett bilinear coupling preserves $\frac{d}{d\tau}(x^4 E_{00}) \equiv 0$ identically and resolves the sharp phase transition from coherent Talbot/Rabi trapping ($\gamma_{\mathrm{deph}} \cdot (\pi/2) \ll 1$, zero secular drain) to smooth Markovian radiative erosion ($\gamma_{\mathrm{deph}} \cdot (\pi/2) \ge 1$) matching `p2_02`'s continuum-log drain on the Markovian plateau.
   - *Kill Criteria:* Sweep $\gamma_{\mathrm{deph}} \in \{0.0, 0.05, 0.15, 0.35, 2/\pi, 1.0, 1.5, 3.0, 6.0\}$ at $\tilde\sigma_0 = 9/31$ and sub-unitarity $\tilde\sigma \in \{4.0, 6.0\}$; log exact un-clamped $E_{00}$ and bath + background + radiation energy balance ($< 10^{-7}$); verify $J_{\mathrm{cut}} = 30$ vs. $J_{\mathrm{cut}} = 60$ stability and contrast gapped $S^3/I^*$ ($M_{I^*} = 26$) against ungapped round $S^3$ ($M_{S^3} = 6$) at $\tilde\sigma = 4.0$.

3. **KT3 (Serendipity Investigation `cand_3` + `cand_4`: Autonomous Bianchi-IX `M_R`-Splitting Talbot Echo Scrambler):**
   - *Hypothesis:* Along the axisymmetric volume-preserving Bianchi IX $n=2(-)$ shear direction (canonical Misner exponential map $\lambda_B = e^{\sqrt{3} b_2}$), opposite-chirality pair energies $E_{\mathrm{pair}}(j, M_R; b_2)$ split at both linear $O(b_2)$ and non-vanishing quadratic $O(b_2^2)$ order (weighted by exact stretched-difference Clebsch--Gordan probabilities $p_{\mathrm{CG}}(j, M_R)$), autonomously scrambling half-antipodal $\Delta\eta = \pi/2$ Talbot revivals into transient pre-Heisenberg Landau-like phase mixing at perturbative $|b_2| \le 0.15$ with zero external $\gamma_{\mathrm{deph}}$.
   - *Kill Criteria:* Prove the exact $O(b_2)$ and $O(b_2^2)$ splitting laws and CG weight normalization to 50 digits / exact `Fraction`; quantify strictly conservative ($\gamma_{\mathrm{deph}} = 0$) echo suppression and $[10\pi, 20\pi]$ recurrence floors across $|b_2| \in [0, 0.15]$ for BOTH static $b_2$ ($O(b_2)$ mechanism) and zero-mean dynamically oscillating $b_2(\eta) = b_0 \cos(\Omega_2 \eta)$ ($O(b_2^2)$ mechanism, verified by direct time-varying oscillator ODE); test coupled Landau-like phase mixing at $\gamma_{\mathrm{deph}} = 0$ and report exact critical shear thresholds (or honest kill if splitting is insufficient at perturbative $|b_2| \le 0.15$).

4. **KT4 (Controls, Dual-Integrator Cross-Checks & Exact Limit Recovery):**
   - *Criteria:* Exact `p2_02` bare-limit recovery ($\Pi_{\mathrm{kin}} = \Pi_{\mathrm{rem}} = g_{\mathrm{cpl}} = 0$, $< 10^{-9}$), exact `p2_06` Talbot revival recovery ($b_2 = 0, \gamma_{\mathrm{deph}} = 0$, $< 10^{-13}$), matched round-`S^3` negative controls across KT1--KT3, `DOP853` vs. `Radau` dual-integrator verification ($< 10^{-5}$) across all key dynamical regimes, $J_{\mathrm{cut}} = 30$ vs. $J_{\mathrm{cut}} = 60$ convergence, and zero silent clamping of `E_00` or energy residuals.

## 3. Conventions Locked to `intent/global.md`
- Signature $(-,+,+,+)$; $8\pi G = M_{\mathrm{Pl}}^{-2} = 1$, $c = 1$, Starobinsky Euler scale $H_S = 1$ ($\tau = H_S t$, $x = H_S a$, $d\eta = d\tau/x$).
- Pure dark conformal gauge bosons: $\kappa_V = -44738/93$, $\tilde\sigma_0 = c_{\mathrm{anom}} = 9/31$, $x_{\mathrm{ring}}(\tilde\sigma_0) = \sqrt{44672/93} = 21.91675355$.
- Complete Model B constants from `p2_06` (`streams/p2_06/02_solve.json`):
  - `rel-S^3` topological: $\Pi_{\mathrm{kin}} = 3900/31 = 125.806452$ ($x_{\mathrm{kin}} = 11.216348$), $\Pi_{\mathrm{rem}} = 984896/465 = 2118.055914$.
  - `abs MS` (exact-derivative Gaussian MS): $\Pi_{\mathrm{kin}} = 4344/31 = 140.129032$ ($x_{\mathrm{kin}} = 11.837611$), $\Pi_{\mathrm{rem}} = 2066.126432$.
