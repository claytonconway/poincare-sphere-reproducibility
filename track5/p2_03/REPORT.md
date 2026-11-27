# Stream Report: `p2_03_graviton_casimir_null_cone`

**Title:** 1-Loop Pure-Graviton Casimir Energy and Null-Cone Focusing Consistency Across Spherical Space Forms $S^3/\Gamma$
**Status:** Verified (cold blinded `referee-critic` audit PASSED: 0 P0, 0 P1; `CLM-P2-03` promoted to `CAVEATED` in `ledger/claims.json`)
**Date:** 2026-11-26
**Target Claim:** `CLM-P2-03`
**Scripts & Machine-Readable Evidence:**
- `streams/p2_03_graviton_casimir_null_cone/01_derive.py` (`01_derive.json`)
- `streams/p2_03_graviton_casimir_null_cone/02_solve.py` (`02_solve.json`)
- `streams/p2_03_graviton_casimir_null_cone/03_kill_tests.py` (`03_kill_tests.json`)
- `streams/p2_03_graviton_casimir_null_cone/results.json` (`physics_audit_results`: `PASS`, 4/4 kill-tests passed, 0 P0, 0 P1, 0 P2)

---

## Executive Summary

We execute the complete Phase 2 verification program (KT1--KT4) for the 1-loop pure-vacuum-GR graviton topological Casimir stress-energy tensor across all five nontrivial isotropy-irreducible ADE spherical space forms ($\mathbb{R}P^3 = S^3/\mathbb{Z}_2$, $S^3/Q_8$, $S^3/T^*$, $S^3/O^*$, $S^3/I^*$) and the infinite anisotropic lens ($L(p,1) = S^3/\mathbb{Z}_p$) and binary dihedral ($S^3/D_m^*$) towers, evaluating their consistency with Chandrasekaran's (`arXiv:2611.34924`) null-cone focusing positivity bound $\hat{\Xi}_y \ge 0$ and causal-diamond finite-dimensionality ($\dim\mathcal{H}_{\mathcal{D}} < \infty$).

The main findings are:

1. **KT1 (On-Shell Global $dS_4/\Gamma$ Canonical ADM Reduction, Zero Modes, and Bunch--Davies vs. Static-ESU TT Limit):**
   - On the on-shell vacuum $K = +1$ global $dS_4/\Gamma$ background ($G_{\mu\nu} + \Lambda g_{\mu\nu} = 0$, $a(\eta) = a_{\mathrm{dS}}\sec\eta$, $\mathcal{H}' = \mathcal{H}^2 + 1$), we verify the canonical ADM quadratic action and linearized Einstein constraints directly to 50 digits (`01_derive.py`): gauge invariance of $(\Phi_B, \Psi_B)$ and $B_p^T - (E_p^T)'$ ($4.80 \times 10^{-52}$), scalar Hamiltonian and trace Euler--Lagrange equations ($0.00$ and $5.47 \times 10^{-48}$), and the exact contracted Bianchi identity ($5.47 \times 10^{-48}$). For every $p = n + 1 \ge 3$, the elliptic scalar constraint determinant $(p^2 - 4)^2 \ge 25 > 0$ forces $\Phi_B = \Psi_B \equiv 0$ (zero propagating scalar DOF), and in canonical vector gauge $E_p^T = 0$ the algebraic $B_p^T$ Gaussian integral $[\det(a^2(p^2 - 4))]^{-1/2}$ cancels the 3D York Jacobian $[\det(a^2(p^2 - 4))]^{+1/2}$ identically (zero propagating vector DOF), leaving strictly the 2 physical TT polarizations.
   - At $p = 1$, $d_0^\Gamma(1) = d_0^{S^3}(1) = 1$ is a pure minisuperspace time-reparametrization gauge mode (verified to $1.34 \times 10^{-51}$). At $p = 2$, the 4 conformal Killing scalars on $S^3$ (pure $SO(4,1)/SO(4)$ de Sitter boosts, verified to $5.35 \times 10^{-51}$) are projected out identically on every nontrivial quotient ($d_0^\Gamma(2) = 0$).
   - The spatial Killing-vector mismatch at $p = 2$ ($N_{\mathrm{KV}} = 6$ on $S^3, \mathbb{R}P^3$; $4$ on $L(p,1)$ for $p \ge 3$; $3$ on $S^3/Q_8, S^3/D_m^*, S^3/T^*, S^3/O^*, S^3/I^*$) has $-\Delta_1 - 2 = p^2 - 4 = 0$, hence $h_{ij}^V = \nabla_i E_j^T + \nabla_j E_i^T \equiv 0$ and $\mathcal{L}_V|_{p=2} \equiv 0$. In Lorentzian canonical quantization on global $dS_4/\Gamma$, the $N_{\mathrm{KV}}(\Gamma)$ shift modes $B_{p=2}^T$ act solely as Lagrange multipliers enforcing the global $\mathrm{Isom}_0(S^3/\Gamma)$ angular-momentum constraints $Q[K_A]|\Psi\rangle = 0$ (Moncrief 1975/1976), contributing **zero local propagating degrees of freedom and zero shift to the local vacuum stress-energy tensor $\langle T_{\mu\nu} \rangle$**.
   - Evaluating the exact on-shell global $dS_4/\Gamma$ Bunch--Davies TT vacuum ($v_p^{\mathrm{BD}}(\eta) = e^{-ip\eta}(p + i\tan\eta)/\sqrt{2p(p^2 - 1)}$; ODE residual $9.56 \times 10^{-50}$, canonical Wronskian error $2.67 \times 10^{-51}$) under the Ford--Parker minisuperspace effective stress tensor confirms strictly positive (focusing) null energy across all 5 ADE space forms: on $S^3/I^*$, $\pi^2 \Delta\mathcal{C}_{\mathrm{BD}}(\text{bounce}) = +120.265044$ ($+110.792744$ prescription-independent $a^{-4}$ radiation piece $+ 9.472300$ $O(a_{\mathrm{dS}}^{-2} a^{-2})$ curvature piece), within $6.82\%$ of the physical-TT-only static-background ($H \to 0$ ESU, $\omega_p = \sqrt{p^2 - 1}$) limit ($+129.061298$); across all 5 ADE space forms the BD-vs-ESU bounce difference spans $6.23\%$--$8.72\%$ (integrated $\Delta s_*$ difference $2.35\%$--$8.43\%$; $6.60\%$--$6.78\%$ on $S^3/I^*$).

2. **KT2 (50-Digit Dual Regulators and Dual-Integrator On-Shell $dS_4/\Gamma$ Raychaudhuri Oracles):**
   - Across all 5 ADE space forms, 50-digit Hurwitz-zeta + digamma analytic continuation agrees with independent 50-digit Abel--Plana contour integration to max absolute error $2.57 \times 10^{-46}$ (and with 4-point Richardson heat-kernel extrapolation to $\le 1.42 \times 10^{-6}$, relative $1.1 \times 10^{-8}$).
   - Every nontrivial ADE isotropy-irreducible space form has strictly positive (focusing) pure-graviton topological Casimir energy in the physical-TT static-ESU limit: $\pi^2 \Delta\mathcal{C}_{\mathrm{top}}^{\mathrm{EH\text{-}TT}} = +0.812266$ ($\mathbb{R}P^3$), $+4.528512$ ($S^3/Q_8$), $+23.755536$ ($S^3/T^*$), $+52.861439$ ($S^3/O^*$), and $+129.061298$ ($S^3/I^*$).
   - Coupled nonlinear FLRW + cancellation-free Raychaudhuri ODE integrations on $dS_4/\Gamma$ (all 10 ADE ESU configurations plus the 2-component $a^{-4} + a_{\mathrm{dS}}^{-2} a^{-2}$ Bunch--Davies source on $S^3/I^*$) match `DOP853` vs. `Radau` to max relative difference $3.39 \times 10^{-11}$ (2-component BD on $S^3/I^*$: $5.94 \times 10^{-14}$ tip, $1.50 \times 10^{-12}$ centered; target $< 10^{-9}$), un-clamped Friedmann constraint $|e_{00}|_{\mathrm{rel}} \le 5.72 \times 10^{-12}$, exact round-$S^3$ $dS_4$ limit ($|\Xi| = 0.00$, $|u_*/\varrho_S - 1| = 0.00$), and flat-space limit ($a = 10^{12} a_{\mathrm{Cas}} \implies \Xi = 9.55 \times 10^{-26}$).

3. **KT3 (Exact Analytical Berger-Sphere Slope Theorem, Anisotropic $T_{uu}^{\mathrm{grav}}(\theta)$ on $L(p,1)$ and $S^3/D_m^*$, and Flat Cubic $T^3$ Negative Control):**
   - We prove **Theorem 2** analytically for all $p \ge 3$ and both chiralities $s = \pm 1$ via the exact first-order operator identity $\left.d\Delta_L^{(3D)}/d\lambda_B\right|_1 = 4\vec{J}\cdot\vec{S} + \frac{64}{3} - 8 K_3^2 - 4 K_3 S_3 - \frac{4}{3} S_3^2$ combined with exact Lande ($c_1 = 4s/(p+s)$) and rank-2 Wigner--Eckart ($c_2 = 12/[p(p+s)]$) projections:
     $$\left.\frac{d\omega_{p, s, M_L}^2}{d\lambda_B}\right|_{\lambda_B = 1} = \frac{4 s (p + s)^2}{p} - \frac{8(p + 2s)}{p} M_L^2 \qquad (p \ge 3,\; s = \pm 1,\; |M_L| \le \tfrac{p - 1 + 2s}{2}),$$
     verified against explicit Berger-sphere matrices across $p = 3..13$ to $6.25 \times 10^{-13}$ (and exact finite-$\lambda_B$ stretched eigenvalue $\omega^2(\lambda_B) = (p+1)^2/\lambda_B^2 - 2(p+1)$ to $1.08 \times 10^{-12}$). It recovers exact isotropy $|p_\parallel - p_\perp| = 0.00$ on both $\mathbb{R}P^3 = L(2,1)$ and $S^3/Q_8 = S^3/D_2^*$, with Hurwitz+digamma vs. Abel--Plana max error $1.27 \times 10^{-47}$.
   - **Critical pinch thresholds:**
     - On $L(p,1)$, Hopf-parallel pure-graviton null energy $T_{uu}^{\parallel,\mathrm{grav}} = u + p_\parallel$ flips **permanently negative (defocusing)** for **$p \ge 11$** ($\pi^2 T_{uu}^\parallel = +0.5858$ at $p = 10 \to -12.2544$ at $p = 11$, defocusing cone $\theta < 33.86^\circ$), while the angle-averaged energy density $u_{\mathrm{grav}}$ flips negative for **$p \ge 14$**.
     - **Unexpected low-order anomaly on $L(3,1)$:** Despite positive total energy density $\pi^2 u_{\mathrm{grav}} = +0.821579 > 0$, the odd-order lens space $L(3,1)$ exhibits an **isolated low-order Hopf-parallel graviton defocusing window** ($\pi^2 p_\parallel = -1.657602$, $\pi^2 T_{uu}^\parallel = -0.836023 < 0$, defocusing cone $\theta < 32.49^\circ$), triple-confirmed by Hurwitz+digamma, Abel--Plana (error $2.45 \times 10^{-48}$), and 4-point Richardson heat kernel (error $4.18 \times 10^{-8}$). Since $p_\parallel = -\frac{1}{4V}\sum \omega^{-1} d\omega^2/d\lambda_B$, this negative pressure is driven by a **positive** topological slope remainder $S_{\mathrm{slope}}(p) > 0$ ($S_{\mathrm{slope}}(3) = +112$, $S_{\mathrm{slope}}(4) = +864 > 0$, reflecting a deficit of high-$|M_L|$ negative-slope modes relative to $S^3$ combined with odd-order period-6 parity). By contrast, an exact rational Bernoulli proof shows a matched spin-1 Maxwell field on $L(3,1)$ has $\pi^2 p_\parallel^{\mathrm{EM}} = 0$ identically and $\pi^2 T_{uu}^{\parallel,\mathrm{EM}} = +4/9 > 0$ (strictly focusing).
     - On $S^3/D_m^*$, $T_{uu}^{\parallel,\mathrm{grav}}$ flips permanently negative for **$m \ge 6$** ($\pi^2 T_{uu}^\parallel = +8.5816$ at $m = 5 \to -22.3473$ at $m = 6$, cone $\theta < 35.71^\circ$), and $u_{\mathrm{grav}}$ flips negative for **$m \ge 8$**.
   - As $p, m \to \infty$, both towers converge to the exact $S^1$ flat-circle limit ($u_{\mathrm{grav}} \to -\pi^2/(45 L_{\min}^4)$, $p_\parallel/u \to +3$, $p_\perp/u \to -1$, $T_{uu}^\parallel \to -4\pi^2/(45 L_{\min}^4)$, $T_{uu}^\perp / |T_{uu}^\parallel| \to 0$; at $L(100,1)$ and $S^3/D_{50}^*$: $u/u_{S^1} = 0.9787$ and $0.9783$, $p_\parallel/u = 3.0419$ and $3.0432$).
   - **Flat cubic $T^3$ negative control:** Exact 50-digit Epstein zeta $Z_3(4) = 16.532315959762$ (cross-checked against direct 3D lattice summation to $4.82 \times 10^{-7}$) yields $u_{\mathrm{grav}}(T^3) = -1.675074 / L^4 < 0$ (defocusing, $\Xi/(1/s) = -2.755386$, $\Delta s_* = +1.836924 > 0$), separating universal 4D conformal scaling ($a$-independence of $\Delta s_*$) from the genuinely spherical-space-form-specific positive focusing sign of all 5 ADE space forms.

4. **KT4 (Dark-Photon Crossover $N_V^{\mathrm{crit}}(\Gamma)$, Non-Abelian $SU(2)$ Holonomy Switch, and EFT / `CLM-T3-01` Species-Scale Boundary):**
   - On $\mathbb{R}P^3, S^3/Q_8, S^3/T^*, S^3/O^*$, both 1-loop gravitons and 1-loop Maxwell photons focus ($\Delta\mathcal{C}_{\mathrm{top}} > 0$), so **no finite $N_V \ge 0$ can flip the sign**. Uniquely on $S^3/I^*$, adding dark photons flips the total Casimir null energy negative at **$N_V^{\mathrm{crit}}(S^3/I^*) = 5$** in the static-ESU TT limit (exact $4.154148$) and **$N_V^{\mathrm{crit,BD}}(S^3/I^*) = 4$** in on-shell global $dS_4/I^*$ Bunch--Davies (exact integrated $3.880170$ tip / $3.872523$ centered; prescription-independent $a^{-4}$ piece alone gives $3.566131 \implies N_V \ge 4$).
   - For pure $SU(2)$ Yang--Mills flat connections on $S^3/I^*$ (`CLM-T2-03`, $F_{\mu\nu} \equiv 0$ locally; valid strictly in the weakly coupled deconfined UV regime $g_{\mathrm{YM}}(1/a) \ll 1$, $L_{\min} \ll \Lambda_{\mathrm{YM}}^{-1}$), the pure-YM sector switches between defocusing ($\rho_0, \rho_1$) and focusing ($\rho_2$), but including the 1-loop graviton keeps a single $SU(2)$ group ($N_{\mathrm{SU(2)}} = 1$) strictly focusing in all three vacua ($\pi^2 \Delta\mathcal{C}_{\mathrm{net}} = +35.86, +40.52, +160.52 > 0$); flipping the total (graviton + YM) null-cone sign requires **$N_{\mathrm{SU(2)}} \ge 2$** (ESU crossover $1.3847$ for $\rho_0$, $1.4577$ for $\rho_1$; BD@bounce $1.2903$ and $1.3584$).
   - Throughout the entire EFT-controlled regime ($a \gg a_{\mathrm{Cas}}$, $L_{\min}(a) \gg \ell_{\mathrm{sp}}^{\mathrm{red}}$), $\Delta s_*$ is a strictly finite, bounded, $a$-independent constant preserving Chandrasekaran's finite-dimensionality bound $\dim\mathcal{H}_{\mathcal{D}} < \infty$. Pushing $a$ down to nonperturbative $|\Xi| \sim O(1)$ backreaction enters strictly below `CLM-T3-01`'s Dvali species scale ($L_{\min}(a_{\mathrm{Cas}})/\ell_{\mathrm{sp}}^{\mathrm{red}} = 0.92758 < 1$ and static $|\Xi| = 1$ proxy $L_{\min}/\ell_{\mathrm{sp}}^{\mathrm{red}} = 0.43594 < 1$ for pure gravity on $S^3/I^*$; asymptoting to $0.45510$ and $0.07868$ as $N_V \to \infty$), where 1-loop semiclassical EFT is already invalid.

---

## 1. Conventions and Exact Spectral Framework

All signs and units follow `intent/global.md`: metric signature $(-,+,+,+)$, reduced Planck units $8\pi G_N = M_{\mathrm{Pl}}^{-2} = 1$, $c = 1$, left-invariant orthonormal coframe $d\omega^a = -\varepsilon^a{}_{bc}\omega^b \wedge \omega^c$, $[X_a, X_b] = 2\varepsilon_{abc} X_c$, and $\mathrm{Vol}(S^3/\Gamma, a) = 2\pi^2 a^3 / |\Gamma|$. Unreduced Planck length $\ell_P = \sqrt{G_N \hbar} = \ell_{\mathrm{Pl}}/\sqrt{8\pi}$ is used for Chandrasekaran's dimensionless area $s = \mathcal{A}/(4\ell_P^2) = 2\pi \mathcal{A}/\ell_{\mathrm{Pl}}^2$.

On unit $S^3 \cong SU(2)$, the finite subgroup $\Gamma \subset SU(2)_L$ acts freely on the left. Let $m_\Gamma(j)$ be the multiplicity of the trivial representation of $\Gamma$ in the $SU(2)_L$ spin-$j$ irrep ($2j \in \mathbb{Z}_{\ge 0}$). At principal quantum number $p = n + 1$, the 3D harmonic degeneracies on $S^3/\Gamma$ are:
- **Scalars ($p \ge 1$):** $d_0^\Gamma(p) = m_\Gamma\!\left(\frac{p-1}{2}\right) p$, $-\Delta_0 = p^2 - 1$.
- **Transverse vectors ($p \ge 2$):** $d_1^\Gamma(p) = m_\Gamma\!\left(\frac{p-2}{2}\right)(p+1) + m_\Gamma\!\left(\frac{p}{2}\right)(p-1)$, $-\Delta_1 = p^2 - 2$.
- **Transverse-traceless symmetric 2-tensors ($p \ge 3$):** $d_2^\Gamma(p) = m_\Gamma\!\left(\frac{p-3}{2}\right)(p+2) + m_\Gamma\!\left(\frac{p+1}{2}\right)(p-2)$, $-\Delta_2 = p^2 - 3$, with Einstein--Hilbert spatial operator $-\Delta_2 + 2 = p^2 - 1$.

Character multiplicities $m_\Gamma(j)$ were cross-checked against exact closed-form Molien generating functions $F_\Gamma(t) = \sum_{n \ge 0} m_\Gamma(n/2) e^{-(n+1)t}$ across all 5 ADE groups and $L(p,1)$, $S^3/D_m^*$ to max error $3.60 \times 10^{-46}$ (`01_derive.json`).

**Proposition 1 (Exact Period-$P_\Gamma$ Quasi-Polynomial UV Cancellation).** For every finite subgroup $\Gamma \subset SU(2)_L$ of period $P_\Gamma$ ($P = 2, 4, 12, 24, 60$ for $\mathbb{R}P^3, Q_8, T^*, O^*, I^*$; $P = p$ or $2p$ for $L(p,1)$; $P = \mathrm{lcm}(4, 2m)$ for $S^3/D_m^*$), the topological TT degeneracy remainder
$$R_{TT}^\Gamma(p) \equiv |\Gamma|\, d_2^\Gamma(p) - 2(p^2 - 4) = C_r p^2 + A_r p + B_r \qquad (r = p \bmod P_\Gamma,\; p \ge 3)$$
obeys the exact zero-mean sum rules $\sum_{r=0}^{P_\Gamma - 1} C_r = \sum_{r=0}^{P_\Gamma - 1} A_r = \sum_{r=0}^{P_\Gamma - 1} B_r = 0$. Consequently, all quartic, quadratic, and logarithmic local UV counterterms cancel identically in the topological Casimir shift $\Delta\mathcal{C}_{\mathrm{top}}^{\mathrm{EH}}(\Gamma) \equiv u(S^3/\Gamma) - u(S^3)$, leaving a strictly UV-finite, renormalization-scheme-independent observable.

---

## 2. KT1: On-Shell Global $dS_4/\Gamma$ Canonical ADM Reduction, Zero Modes, and Bunch--Davies Vacuum

### 2.1 Canonical ADM Scalar and Vector Reduction on On-Shell Global $dS_4/\Gamma$ ($p \ge 3$)

Static $\mathbb{R}_t \times S^3/\Gamma$ is not a vacuum Einstein manifold ($R_{00} = 0 \neq R_{ij} = 2\gamma_{ij}$), so a naive 4D-covariant de Donder determinant on $S^1_\beta \times S^3/\Gamma$ is off-shell and does not cancel vector/ghost curvature shifts. The proper on-shell pure-vacuum-GR background with $S^3/\Gamma$ spatial slices is global $dS_4/\Gamma$:
$$ds^2 = a(\eta)^2\left(-d\eta^2 + \gamma_{ij} dx^i dx^j\right), \qquad a(\eta) = a_{\mathrm{dS}} \sec\eta, \qquad \mathcal{H} = \tan\eta, \qquad \mathcal{H}' = \mathcal{H}^2 + 1.$$
By 3D spatial $SO(3)$ covariance on $S^3/\Gamma$, the quadratic Einstein--Hilbert ADM action $\delta^2 S_{\mathrm{EH}}$ decouples cleanly into tensor, transverse-vector, and scalar sectors:

1. **Transverse vector sector ($p \ge 3$, eigenvalue $-\Delta_1 = p^2 - 2$):**
   With $h_{0i}^V = -a^2 B_i^T$ and $h_{ij}^V = a^2(\nabla_i E_j^T + \nabla_j E_i^T)$, the exact quadratic ADM action is
   $$S_V = \frac{1}{4}\int d\eta\, a^2 (p^2 - 4)\left(B_p^T - (E_p^T)'\right)^2,$$
   invariant under $\delta E_p^T = \xi_p^T$, $\delta B_p^T = (\xi_p^T)'$. In canonical gauge $E_p^T = 0$ (unit Faddeev--Popov determinant), $B_p^T$ carries no time derivatives and integrates out algebraically as $[\det(a^2(p^2 - 4))]^{-1/2}$, cancelling identically against the 3D York Jacobian $J_V = [\det(a^2(p^2 - 4))]^{+1/2}$ from $\|\nabla_i E_j^T + \nabla_j E_i^T\|^2 = 2 a^2(p^2 - 4)\|E^T\|^2$. Zero propagating vector modes remain at any $p \ge 3$.

2. **Scalar sector ($p \ge 3$, eigenvalue $-\Delta_0 = k^2 = p^2 - 1$):**
   With $h_{00} = -2a^2\Phi$, $h_{0i} = a^2\nabla_i B$, $h_{ij} = a^2[-2\Psi\gamma_{ij} + 2\nabla_i\nabla_j E]$, shear $\sigma = E' - B$, and gauge-invariant Bardeen potentials $\Phi_B = \Phi - \sigma' - \mathcal{H}\sigma$, $\Psi_B = \Psi + \mathcal{H}\sigma$ (gauge invariance under $\xi^0 = T(\eta), \xi^i = \nabla^i L(\eta)$ verified in `01_derive.py` to $4.80 \times 10^{-52}$), the gauge-invariant quadratic scalar Lagrangian on $K = +1$ global $dS_4/\Gamma$ is
   $$\mathcal{L}_S = a^2\left[-3(\Psi_B' + \mathcal{H}\Phi_B)^2 + (k^2 - 3)(\Psi_B^2 - 2\Phi_B\Psi_B)\right].$$
   Its Euler--Lagrange variations reproduce the exact vacuum Hamiltonian constraint $E_{00} = (p^2 - 4)\Psi_B + 3\mathcal{H}(\Psi_B' + \mathcal{H}\Phi_B) = 0$ (err $0.00$) and trace equation $E_{ii} = 0$ (err $5.47 \times 10^{-48}$), and obey the exact contracted Bianchi identity $\mathrm{EL}_\sigma \equiv -2 a^2 k^2 E_{0i}$ (err $5.47 \times 10^{-48}$), where $E_{0i} = \Psi_B' + \mathcal{H}\Phi_B = 0$. For every $p \ge 3$, the elliptic constraint determinant is $(p^2 - 4)^2 \ge 25 > 0$, enforcing $\Phi_B = \Psi_B \equiv 0$ identically; in longitudinal gauge ($B = E = 0$, $\det\mathrm{FP} = 1$), integrating out $(\Phi, \Psi)$ from $\mathcal{L}_S$ yields the ultralocal, $\mathcal{H}$-independent determinant $[\det(a^4 k^2(k^2 - 3))]^{-1/2}$, cancelling the scalar York Jacobian $[\det(a^4 k^2(k^2 - 3))]^{+1/2}$ identically for all $p \ge 3$ with zero propagating scalar degrees of freedom.

### 2.2 Exact Low-Mode and Killing-Vector Accounting ($p = 1, 2$)

| Manifold | $\lvert \Gamma\rvert$ | $d_0(p=1)$ (homog.) | $d_0(p=2)$ (CKV scalar) | $N_{\mathrm{KV}} = d_1(p=2)$ (Killing vectors) | $d_2(p=3)$ (lowest TT) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Round $S^3$ | 1 | 1 | 4 | 6 | 10 |
| $\mathbb{R}P^3 = L(2,1)$ | 2 | 1 | 0 | 6 | 10 |
| $L(p,1)$ ($p \ge 3$) | $p$ | 1 | 0 | 4 | 6 |
| $S^3/Q_8 = S^3/D_2^*$ | 8 | 1 | 0 | 3 | 7 |
| $S^3/T^*$ | 24 | 1 | 0 | 3 | 5 |
| $S^3/O^*$ | 48 | 1 | 0 | 3 | 5 |
| $S^3/I^*$ | 120 | 1 | 0 | 3 | 5 |

- **Homogeneous mode ($p = 1$):** $d_0^\Gamma(1) = d_0^{S^3}(1) = 1$ identically on every space form. Under $\Phi = T' + \mathcal{H}T, \Psi = -\mathcal{H}T$, the $k^2 = 0$ Hamiltonian constraint $E_{00} = -3\Psi + 3\mathcal{H}(\Psi' + \mathcal{H}\Phi) = 0$ holds identically (verified to $1.34 \times 10^{-51}$): pure minisuperspace time-reparametrization redundancy.
- **Conformal Killing scalars ($p = 2$):** On round $S^3$, $k^2 - 3 = 0$ admits 4 pure-gauge $SO(4,1)/SO(4)$ de Sitter boost modes $\Phi_B = \Psi_B = c\cos\eta$ solving all vacuum equations (verified to $5.35 \times 10^{-51}$). On every nontrivial quotient $S^3/\Gamma$, $d_0^\Gamma(2) = 0$: these modes are absent identically.
- **Spatial Killing vectors ($p = 2$):** At $p = 2$, $p^2 - 4 = 0$, so $\nabla_i E_j^T + \nabla_j E_i^T \equiv 0$ and $\mathcal{L}_V|_{p=2} \equiv 0$. The $N_{\mathrm{KV}}(\Gamma)$ shift modes $B_{p=2}^T$ ($6$ on $S^3$ and $\mathbb{R}P^3$; $4$ on $L(p,1)$, $p \ge 3$; $3$ on $Q_8, D_m^*, T^*, O^*, I^*$) act solely as Lagrange multipliers enforcing Moncrief's (1975/1976) global $\mathrm{Isom}_0(S^3/\Gamma)$ linearization-stability angular-momentum constraints $Q[K_A] = \int_{\Sigma_3} K_A^i \mathcal{H}_i = 0$ on physical states ($Q[K_A]|\Psi\rangle = 0$). They carry zero local propagating DOF and contribute **zero** to the local 1-loop vacuum stress-energy tensor $\langle T_{\mu\nu} \rangle$ on global $dS_4/\Gamma$. (In Euclidean $S^4$ partition functions such as Anninos--Denef--Law--Sun `arXiv:2009.12464`, Killing zero modes enter as a global isometry-group volume prefactor $1/\mathrm{Vol}(\mathrm{Isom}_0)$.)

### 2.3 On-Shell Global $dS_4/\Gamma$ Bunch--Davies TT Vacuum vs. Static-ESU Physical-TT Limit

On global $dS_4/\Gamma$, $a''/a = 2\sec^2\eta - 1$, so the canonical TT variable $v_{ij} = a h_{ij}^{TT}$ obeys the exact Poeschl--Teller equation $v_p'' + (p^2 - 2\sec^2\eta) v_p = 0$, solved identically by the Bunch--Davies mode function
$$v_p^{\mathrm{BD}}(\eta) = \frac{e^{-ip\eta}}{\sqrt{2p(p^2 - 1)}}(p + i\tan\eta)$$
(verified to 50 digits in `01_derive.json`: ODE residual $9.56 \times 10^{-50}$, canonical Wronskian $v_p (v_p^*)' - v_p^* v_p' = +i$ to $2.67 \times 10^{-51}$). Under the Ford--Parker / exponential-parameterization minisuperspace TT effective stress tensor $S_{\mathrm{EH}}^{(2)}[a, N; h_{\mathrm{can}}]$ (identical to two minimally coupled massless scalars), the UV-finite topological null stress tensor splits into two individually covariantly conserved pieces ($w = +1/3$ radiation-like and $w = -1/3$ curvature-coupled):
$$\Delta T_{uu}^{\mathrm{BD\text{-}TT}}(\eta) = \frac{4}{3\pi^2 a(\eta)^4}\left[\pi^2 \Delta\mathcal{C}_{\mathrm{BD}}^{(a^{-4})} + \left(\frac{a(\eta)}{a_{\mathrm{dS}}}\right)^2 \pi^2 \Delta\mathcal{C}_{\mathrm{BD}}^{(a^{-2})}\right]\left(\frac{a_0}{a(\eta)}\right)^2,$$
with $\pi^2 \Delta\mathcal{C}_{\mathrm{BD}}^{(a^{-4})} = \frac{1}{4}\sum_{p\ge 3} R_{TT}^\Gamma(p)(p - p^{-1})$ and $\pi^2 \Delta\mathcal{C}_{\mathrm{BD}}^{(a^{-2})} = \frac{1}{16}\sum_{p\ge 3} R_{TT}^\Gamma(p) p^{-1}$.
- **Prescription and normalization note:** Because the $a^{-2}$ curvature piece has $w = -1/3$ ($\rho_2 + P_2 = \frac{2}{3}\rho_2$), $\pi^2 \Delta\mathcal{C}_{\mathrm{BD}}(\text{bounce}) \equiv \frac{3}{4}\pi^2 a_{\mathrm{dS}}^4 \Delta(\rho + P)|_{\text{bounce}} = +120.265044$ on $S^3/I^*$ is the radiation-normalized null-energy coefficient (while the bounce energy density itself is $\pi^2 a_{\mathrm{dS}}^4 \Delta\rho_{\mathrm{BD}} = \pi^2\Delta\mathcal{C}_{\mathrm{BD}}^{(a^{-4})} + 2\pi^2\Delta\mathcal{C}_{\mathrm{BD}}^{(a^{-2})} = +129.737344$). The leading $a^{-4}$ radiation piece ($+110.792744$ on $S^3/I^*$) is prescription-independent and alone gives $N_V^{\mathrm{crit}} = 4$. The subleading $O(a_{\mathrm{dS}}^{-2} a^{-2})$ curvature piece ($+9.472300$, $7.9\%$ of the bounce total) is prescription-dependent relative to linear-metric $\langle G_{\mu\nu}^{(2)}[h]\rangle$ by $O(H_{\mathrm{dS}}^2 \langle h^2 \rangle)$ terms ($H_{\mathrm{dS}} = a_{\mathrm{dS}}^{-1}$).
- **Static-ESU TT limit ($H \to 0$):** Setting $a''/a = 0$ in the TT mode equation yields the physical-TT-only static-background dispersion $\omega_p = \sqrt{p^2 - 1}$ ($\pi^2\Delta\mathcal{C}_{\mathrm{top}}^{\mathrm{EH\text{-}TT}} = +129.061298$ on $S^3/I^*$). Both prescriptions yield strictly positive focusing across all 5 ADE space forms (bounce difference $6.23\%$--$8.72\%$; $6.82\%$ on $S^3/I^*$). **KT1 Verdict: PASS.**

---

## 3. KT2: 50-Digit Dual Regulators and On-Shell $dS_4/\Gamma$ Raychaudhuri Integrals

### 3.1 Dual 50-Digit Casimir Energy Verification Across All 5 ADE Space Forms

| Manifold | $\lvert \Gamma\rvert$ | Conformal Spin-2 (Exact) | $\pi^2\Delta\mathcal{C}_{\mathrm{top}}^{\mathrm{EH\text{-}TT}}$ (Hurwitz+Digamma) | $\lvert \text{HZ} - \text{Abel--Plana}\rvert$ | $\lvert \text{HZ} - \text{Heat Kernel}\rvert$ | $\pi^2\Delta\mathcal{C}_{\mathrm{BD}}^{(a^{-4})}$ | $\pi^2\Delta\mathcal{C}_{\mathrm{BD}}^{(a^{-2})}$ | $\pi^2\Delta\mathcal{C}_{\mathrm{BD}}(\text{bounce})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\mathbb{R}P^3$ | 2 | $+15/16 = +0.937500$ | $+0.8122655898$ | $1.28 \times 10^{-49}$ | $6.60 \times 10^{-9}$ | $+0.698794$ | $+0.059676$ | $+0.758471$ |
| $S^3/Q_8$ | 8 | $+87/16 = +5.437500$ | $+4.5285122744$ | $2.14 \times 10^{-50}$ | $1.84 \times 10^{-8}$ | $+3.698794$ | $+0.434676$ | $+4.133471$ |
| $S^3/T^*$ | 24 | $+3919/144 = +27.215278$ | $+23.7555359055$ | $4.28 \times 10^{-50}$ | $2.43 \times 10^{-7}$ | $+20.537688$ | $+1.669397$ | $+22.207085$ |
| $S^3/O^*$ | 48 | $+8671/144 = +60.215278$ | $+52.8614387366$ | $1.96 \times 10^{-47}$ | $7.54 \times 10^{-7}$ | $+46.016391$ | $+3.549722$ | $+49.566113$ |
| **$S^3/I^*$** | **120** | **$+107051/720 = +148.681944$** | **$+129.0612983489$** | **$2.57 \times 10^{-46}$** | **$1.42 \times 10^{-6}$** | **$+110.792744$** | **$+9.472300$** | **$+120.265044$** |

### 3.2 On-Shell $dS_4/\Gamma$ Raychaudhuri Integration (`DOP853` vs. `Radau`) and Limiting Cases

Integrating the coupled nonlinear FLRW + cancellation-free Raychaudhuri system $(a, H, \chi, u, q = d\varrho/du - 1, \delta u = u - \varrho)$ at $a_{\mathrm{dS}} = 10^4 a_{\mathrm{Cas}}$ with both `DOP853` and `Radau` (`rtol=1e-13`, `atol=1e-18`) gives (columns 6--8 report the $a^{-4}$ ESU ODE cross-checks; the 2-component $a^{-4} + a_{\mathrm{dS}}^{-2} a^{-2}$ BD ODE on $S^3/I^*$ is verified separately below):

| Manifold | Diamond Config | $\Xi/(1/s)$ (ESU) | $\Delta s_*$ (ESU) | $\Delta s_*$ (BD) | `DOP853` vs. `Radau` | Friedmann $\lvert e_{00}\rvert_{\mathrm{rel}}$ | ODE vs. Analytic |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| $\mathbb{R}P^3$ | centered ($\eta_0 = -\pi/4, \chi = \pi/2$) | $+9.435698$ | $-7.734493$ | $-7.290555$ | $1.23 \times 10^{-12}$ | $4.41 \times 10^{-13}$ | $3.26 \times 10^{-8}$ |
| $\mathbb{R}P^3$ | tip sub-hor. ($\eta_0 = 0, \chi = 0.40\pi$) | $+10.164977$ | $-15.348822$ | $-14.988436$ | $5.14 \times 10^{-13}$ | $5.72 \times 10^{-12}$ | $5.92 \times 10^{-8}$ |
| $S^3/Q_8$ | tip@bounce ($\eta_0 = 0, \chi = \pi/4$) | $+4.528512$ | $-4.314778$ | $-4.021726$ | $2.10 \times 10^{-13}$ | $1.75 \times 10^{-14}$ | $1.30 \times 10^{-8}$ |
| $S^3/Q_8$ | centered ($\eta_0 = -\pi/8, \chi = \pi/4$) | $+4.164180$ | $-2.945666$ | $-2.697391$ | $2.21 \times 10^{-13}$ | $5.43 \times 10^{-14}$ | $6.81 \times 10^{-9}$ |
| $S^3/T^*$ | tip@bounce ($\eta_0 = 0, \chi = \pi/6$) | $+4.619132$ | $-3.622459$ | $-3.408077$ | $3.01 \times 10^{-13}$ | $1.50 \times 10^{-14}$ | $5.22 \times 10^{-9}$ |
| $S^3/T^*$ | centered ($\eta_0 = -\pi/12, \chi = \pi/6$) | $+4.552520$ | $-3.117906$ | $-2.917671$ | $1.81 \times 10^{-12}$ | $2.59 \times 10^{-14}$ | $2.94 \times 10^{-9}$ |
| $S^3/O^*$ | tip@bounce ($\eta_0 = 0, \chi = \pi/8$) | $+3.282546$ | $-2.399053$ | $-2.257112$ | $5.07 \times 10^{-14}$ | $6.00 \times 10^{-15}$ | $2.86 \times 10^{-9}$ |
| $S^3/O^*$ | centered ($\eta_0 = -\pi/16, \chi = \pi/8$) | $+3.268140$ | $-2.212292$ | $-2.075525$ | $1.48 \times 10^{-12}$ | $1.84 \times 10^{-14}$ | $1.64 \times 10^{-9}$ |
| **$S^3/I^*$** | **tip@bounce ($\eta_0 = 0, \chi = \pi/10$)** | **$+3.303958$** | **$-2.336454$** | **$-2.182358$** | **$1.11 \times 10^{-13}$** | **$3.41 \times 10^{-15}$** | **$1.81 \times 10^{-9}$** |
| **$S^3/I^*$** | **centered ($\eta_0 = -\pi/20, \chi = \pi/10$)** | **$+3.298129$** | **$-2.220427$** | **$-2.069896$** | **$3.39 \times 10^{-11}$** | **$6.24 \times 10^{-14}$** | **$1.08 \times 10^{-9}$** |

- **2-component BD nonlinear ODE cross-check on $S^3/I^*$ (`02_solve.json`):** Integrating the full two-component source $H^2 + a^{-2} = \lambda + K_4 a^{-4} + 2 K_2 a_{\mathrm{dS}}^{-2} a^{-2}$ gives $\Delta s_*^{\mathrm{BD,tip}} = -2.182358239$ (`DOP853` vs. `Radau` $5.94 \times 10^{-14}$, $|e_{00}|_{\mathrm{rel}} = 1.98 \times 10^{-15}$, vs. 50-digit analytic $2.73 \times 10^{-9}$) and $\Delta s_*^{\mathrm{BD,ctr}} = -2.069895801$ (`DOP853` vs. `Radau` $1.50 \times 10^{-12}$, $|e_{00}|_{\mathrm{rel}} = 7.14 \times 10^{-15}$, vs. analytic $1.93 \times 10^{-9}$).
- **Note on $\mathbb{R}P^3$:** At $\eta_0 = 0$, $\chi_{\mathrm{inj}} = \pi/2$ reaches future conformal infinity $\mathcal{I}^+$ of $dS_4$ ($\sec(\pi/2) = \infty$); the bounce-centered diamond $\eta_0 = -\pi/4$ and sub-horizon $\chi = 0.40\pi$ stay strictly inside $dS_4/\mathbb{R}P^3$.
- **Exact limiting checks:** Round-$S^3$ $dS_4$ ($K_{\mathrm{top}} = 0$) recovers $|\Xi| = 0.00$ and $|u_*/\varrho_S - 1| = 0.00$ in both `DOP853` and `Radau`. The flat-space limit $a = 10^{12} a_{\mathrm{Cas}}$ gives $\Xi = 9.55 \times 10^{-26}$ and $|u_*/\varrho_S - 1| = 3.18 \times 10^{-26}$. **KT2 Verdict: PASS.**

---

## 4. KT3: Anisotropic Pure-Graviton Stress Tensor $T_{uu}^{\mathrm{grav}}(\theta)$ on $L(p,1)$ and $S^3/D_m^*$, and Flat Cubic $T^3$ Negative Control

### 4.1 Exact First-Order Berger-Sphere Spin-2 TT Slope Theorem (Analytical Proof)

On the right-invariant Berger sphere $S^3_{\lambda_B}$ ($ds^2 = a_\perp^2((\tilde\omega^1)^2 + (\tilde\omega^2)^2) + a_3^2(\tilde\omega^3)^2$, $\lambda_B = a_3/a_\perp$), the orthonormal frame $E_1 = X_1, E_2 = X_2, E_3 = \lambda_B^{-1} X_3$ with $X_a = -2i J_a$ ($[X_a, X_b] = 2\varepsilon_{abc} X_c$) has Levi-Civita spin connection coefficients $\Gamma_{123} = \lambda_B - 2\lambda_B^{-1}$, $\Gamma_{231} = \Gamma_{312} = -\lambda_B$ (torsion error $2.22 \times 10^{-16}$), sectional curvatures $R_{1212} = 4 - 3\lambda_B^2$, $R_{1313} = R_{2323} = \lambda_B^2$, and Ricci eigenvalues $R_{11} = R_{22} = 4 - 2\lambda_B^2$, $R_{33} = 2\lambda_B^2$ (error $1.78 \times 10^{-15}$). On the 5-dimensional symmetric traceless representation with spin-2 generators $\vec{S}$ ($S^2 = 6$, total fiber angular momentum $\vec{K} = \vec{J} + \vec{S}$), the covariant derivatives are $D_{1,2} = -i(2 J_{1,2} + \lambda_B S_{1,2})$ and $D_3 = -i(2\lambda_B^{-1} J_3 + (2\lambda_B^{-1} - \lambda_B) S_3)$, and the scale-covariant 3D Lichnerowicz TT operator is $\Delta_L^{(3D)}(\lambda_B) = -(D_1^2 + D_2^2 + D_3^2) + K_{\mathrm{riem}}(\lambda_B)$ with $K_{\mathrm{riem}}(\lambda_B) h_{ij} = -2 R_{ikjl}(\lambda_B) h^{kl}$.

**Theorem 2 (Exact First-Order Berger-Sphere Spin-2 TT Slope).** For every round-$S^3$ TT multiplet at level $p \ge 3$ and chirality $s = \pm 1$ ($j = \frac{p - 1 - 2s}{2}$, $K = j + 2s = \frac{p - 1 + 2s}{2}$, fiber weight $M_L \in \{-K, \dots, +K\}$):
$$\left.\frac{d\omega_{p, s, M_L}^2}{d\lambda_B}\right|_{\lambda_B = 1} = \frac{4 s (p + s)^2}{p} - \frac{8(p + 2s)}{p} M_L^2.$$

*Proof.*
1. **Exact operator derivative at $\lambda_B = 1$:** Differentiating each piece of $\Delta_L^{(3D)}(\lambda_B)$ directly at $\lambda_B = 1$ gives:
   - $\frac{d}{d\lambda_B}(-D_1^2 - D_2^2)\big|_1 = 4(\vec{J}\cdot\vec{S} - J_3 S_3) + 2(S^2 - S_3^2)$,
   - $\frac{d}{d\lambda_B}(-D_3^2)\big|_1 = 2(2 J_3 + S_3)(-2 J_3 - 3 S_3) = -8 J_3^2 - 16 J_3 S_3 - 6 S_3^2$,
   - $\frac{d}{d\lambda_B} K_{\mathrm{riem}}\big|_1 = \frac{28}{3} - \frac{16}{3} S_3^2$ (evaluated directly on $S_3 = 0, \pm 1, \pm 2$ eigenstates: $+\frac{28}{3}, +4, -12$).
   Summing all three and substituting $S^2 = 6$ and $J_3 = K_3 - S_3$ yields the exact operator identity:
   $$\left.\frac{d\Delta_L^{(3D)}}{d\lambda_B}\right|_{\lambda_B = 1} = 4\vec{J}\cdot\vec{S} + \frac{64}{3} - 8 K_3^2 - 4 K_3 S_3 - \frac{4}{3} S_3^2.$$
   By the Hellmann--Feynman theorem inside the round-$S^3$ TT subspace $\ker(\mathrm{div}_0)$, the first-order shift of $\omega_{p,s,M_L}^2$ is the expectation value of this operator in $|j, S=2; K, M_L\rangle$ (with $K_3 = M_L$ and $2\vec{J}\cdot\vec{S} = K(K+1) - j(j+1) - 6$).
2. **Exact Lande vector projection ($S_3$):** On the extreme multiplets $K = j + 2s$, $\vec{K}\cdot\vec{S} = \frac{1}{2}[K(K+1) + 6 - (K-2s)(K-2s+1)] = 2s K + s + 1$, so
   $$\langle K, M_L | S_3 | K, M_L \rangle = \frac{\vec{K}\cdot\vec{S}}{K(K+1)} M_L = \frac{4s}{p + s} M_L.$$
3. **Exact Wigner--Eckart rank-2 projection ($S_3^2 - 2$):** On any irreducible multiplet $|j, S; K, M_L\rangle$, the Casimir contraction identity for the traceless symmetric tensor $S_a S_b + S_b S_a - \frac{2}{3}\delta_{ab} S^2$ gives
   $$c_2 = \frac{(\vec{K}\cdot\vec{S})^2 - \frac{1}{2}\vec{K}\cdot\vec{S} - \frac{1}{3}K(K+1)S(S+1)}{\frac{1}{6}K(K+1)(2K-1)(2K+3)}.$$
   Substituting $S(S+1) = 6$, $K = (p - 1 + 2s)/2$, and $\vec{K}\cdot\vec{S} = 2 s K + s + 1$ yields identically for both $s = +1$ (stretched $K = j+2$) and $s = -1$ (anti-stretched $K = j-2$):
   $$\langle K, M_L | S_3^2 - 2 | K, M_L \rangle = \frac{12}{p(p + s)}\left(M_L^2 - \frac{K(K+1)}{3}\right).$$
4. **Combination:** Substituting Steps 2--3 into Step 1 yields the $M_L^2$ coefficient identically for both $s = \pm 1$:
   $$\beta(p, s) = -8 - 4\left(\frac{4s}{p + s}\right) - \frac{4}{3}\left(\frac{12}{p(p + s)}\right) = -8 - \frac{16 s (p + s)}{p(p + s)} = -\frac{8(p + 2s)}{p}.$$
   The constant term follows directly from Step 1 (or equivalently from the exact isotropic average $\langle d\omega^2/d\lambda_B\rangle = -\frac{2}{3}(p^2 - 1)$): $\alpha(p, s) = -\frac{2}{3}(p^2 - 1) - \beta(p, s)\frac{K(K+1)}{3} = \frac{4s(p + s)^2}{p}$. $\blacksquare$

Every intermediate step was independently cross-checked against explicit $5(2j+1) \times 5(2j+1)$ matrices across $p = 3..13, s = \pm 1$ (`01_derive.json`): operator identity FD error $4.95 \times 10^{-8}$, Lande $c_1$ error $4.00 \times 10^{-15}$, Wigner--Eckart $c_2$ error $8.44 \times 10^{-15}$, Theorem 2 vs. matrix error $6.25 \times 10^{-13}$, Hellmann--Feynman vs. direct $\ker(\mathrm{div}(\lambda_B))$ FD error $8.18 \times 10^{-8}$, and exact finite-$\lambda_B$ stretched-state eigenvalue $\omega_{p,+,M_{\max}}^2(\lambda_B) = (p+1)^2/\lambda_B^2 - 2(p+1)$ error $1.08 \times 10^{-12}$.

By variational thermodynamics at $\lambda_B = 1$, $u = \frac{|\Gamma|}{2\pi^2 a^4} F_\Gamma(1)$, $p_\parallel = -\frac{|\Gamma|}{2\pi^2 a^4} F'_\Gamma(1)$, $p_\perp = \frac{1}{2}(u - p_\parallel)$ (exact trace $p_\parallel + 2 p_\perp = u$), and for a null ray at polar angle $\theta$ to the Hopf fiber $\tilde\omega^3$:
$$T_{uu}^{\mathrm{grav}}(\theta) = (u + p_\perp) + (p_\parallel - p_\perp)\cos^2\theta.$$
On both $\mathbb{R}P^3 = L(2,1)$ and $S^3/Q_8 = S^3/D_2^*$, this yields exact isotropy $|p_\parallel - p_\perp| = 0.00$, with Hurwitz+digamma vs. Abel--Plana max error $1.27 \times 10^{-47}$ across all tested $L(p,1)$ and $S^3/D_m^*$ points.

### 4.2 Anisotropic Towers, Critical Pinch Thresholds, and the $L(3,1)$ Anomaly

| Space Form | $\pi^2 u_{\mathrm{grav}}$ | $\pi^2 p_\parallel$ | $\pi^2 p_\perp$ | $\pi^2 T_{uu}^\parallel$ | $\pi^2 T_{uu}^\perp$ | $p_\parallel / u$ | $u / u_{S^1}$ | Defocusing Cone $\theta_{\mathrm{crit}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $L(2,1) = \mathbb{R}P^3$ | $+0.8123$ | $+0.2708$ | $+0.2708$ | $+1.0830$ | $+1.0830$ | $+0.3333$ | $-36.55$ | $0^\circ$ (all-angle focus) |
| **$L(3,1)$ (anomaly)** | **$+0.8216$** | **$-1.6576$** | **$+1.2396$** | **$-0.8360$** | **$+2.0612$** | **$-2.0176$** | **$-7.30$** | **$\theta < 32.49^\circ$ ($u > 0$)** |
| $L(4,1)$ | $+2.0510$ | $+0.7886$ | $+0.6312$ | $+2.8397$ | $+2.6822$ | $+0.3845$ | $-5.77$ | $0^\circ$ (all-angle focus) |
| $L(10,1)$ | $+11.0790$ | $-10.4932$ | $+10.7861$ | $+0.5858$ | $+21.8651$ | $-0.9471$ | $-0.80$ | $0^\circ$ (all-angle focus) |
| **$L(11,1)$ ($p_{\mathrm{crit}}^\parallel$)** | **$+10.5474$** | **$-22.8017$** | **$+16.6745$** | **$-12.2544$** | **$+27.2219$** | **$-2.1618$** | **$-0.52$** | **$\theta < 33.86^\circ$ ($u > 0$)** |
| **$L(14,1)$ ($p_{\mathrm{crit}}^u$)** | **$-1.1029$** | **$-97.0289$** | **$+47.9630$** | **$-98.1318$** | **$+46.8600$** | **$+87.97$** | **$+0.021$** | **$\theta < 55.35^\circ$ ($u < 0$)** |
| $L(20,1)$ | $-110.981$ | $-537.072$ | $+213.045$ | $-648.053$ | $+102.064$ | $+4.8393$ | $+0.4994$ | $\theta < 68.35^\circ$ |
| $L(100,1)$ | $-135933.7$ | $-413498.9$ | $+138782.6$ | $-549432.6$ | $+2848.9$ | $+3.0419$ | $+0.9787$ | $\theta < 85.88^\circ$ |
| $S^3/D_2^* = S^3/Q_8$ | $+4.5285$ | $+1.5095$ | $+1.5095$ | $+6.0380$ | $+6.0380$ | $+0.3333$ | $-12.74$ | $0^\circ$ (all-angle focus) |
| $S^3/D_5^*$ | $+17.2727$ | $-8.6911$ | $+12.9819$ | $+8.5816$ | $+30.2546$ | $-0.5032$ | $-1.24$ | $0^\circ$ (all-angle focus) |
| **$S^3/D_6^*$ ($m_{\mathrm{crit}}^\parallel$)** | **$+16.0343$** | **$-38.3816$** | **$+27.2080$** | **$-22.3473$** | **$+43.2423$** | **$-2.3937$** | **$-0.56$** | **$\theta < 35.71^\circ$ ($u > 0$)** |
| **$S^3/D_8^*$ ($m_{\mathrm{crit}}^u$)** | **$-11.5998$** | **$-187.462$** | **$+87.9310$** | **$-199.062$** | **$+76.3312$** | **$+16.16$** | **$+0.127$** | **$\theta < 58.23^\circ$ ($u < 0$)** |
| $S^3/D_{50}^*$ | $-135871.8$ | $-413480.9$ | $+138804.6$ | $-549352.6$ | $+2932.8$ | $+3.0432$ | $+0.9783$ | $\theta < 85.82^\circ$ |

**Isolated $L(3,1)$ spin-2 anomaly:** Because $p_\parallel = -\frac{1}{8\pi^2 a^4}\sum_{p\ge 3} S_{\mathrm{slope}}(p)/(p\sqrt{p^2-1})$ carries an overall minus sign, a **positive** topological slope remainder $S_{\mathrm{slope}}(p) = |\Gamma| W_\Gamma(p) - W_{S^3}(p) > 0$ drives $p_\parallel < 0$. On $L(3,1)$, $S_{\mathrm{slope}}(3) = +112 > 0$ and $S_{\mathrm{slope}}(4) = +864 > 0$ (reflecting a deficit of high-$|M_L|$ negative-slope modes relative to $S^3$, combined with the odd-order period-6 zeta-regularized parity structure where even $p$ survive because $-1 \notin \mathbb{Z}_3$). This yields $\pi^2 p_\parallel^{\mathrm{grav}} = -1.657602$ and $\pi^2 T_{uu}^{\parallel,\mathrm{grav}} = -0.836023 < 0$, triple-confirmed by Hurwitz+digamma, Abel--Plana (error $2.45 \times 10^{-48}$), and 4-point Richardson heat kernel (error $4.18 \times 10^{-8}$). By contrast, an exact rational Bernoulli evaluation (`03_kill_tests.py`) proves that a matched spin-1 Maxwell field on $L(3,1)$ has $\pi^2 p_\parallel^{\mathrm{EM}} = 0$ identically (check on $L(2,1)$: exact $+1/16 = u_{\mathrm{EM}}/3$), leaving $\pi^2 T_{uu}^{\parallel,\mathrm{EM}} = +4/9 > 0$ (strictly focusing).

**Flat cubic $T^3$ negative control:** Evaluating the 3D Epstein zeta function via Chowla--Selberg theta splitting gives $Z_3(4) = 16.532315959762$ (cross-checked against direct 3D spherical-shell lattice summation at $R = 120$ plus $4\pi/R$ tail to relative error $4.82 \times 10^{-7}$), yielding $u_{\mathrm{grav}}(T^3) = -Z_3(4)/(\pi^2 L^4) = -1.675074 / L^4 < 0$, $\Xi_{\mathrm{grav}}(T^3)/(1/s) = -2.755386 < 0$, and $\Delta s_*^{\mathrm{grav}}(T^3) = +1.836924 > 0$. This isolates universal 4D conformal scaling ($a$-independence of $\Delta s_*$) from the genuinely spherical-space-form-specific positive focusing sign of all 5 ADE space forms. **KT3 Verdict: PASS.**

---

## 5. KT4: Dark-Photon Crossover, $SU(2)$ Flat-Holonomy Switch, and EFT / `CLM-T3-01` Species-Scale Boundary

### 5.1 Matter Crossovers and Pure-Gravity Species-Scale Squeeze Across All 5 ADE Space Forms

| Manifold | $\pi^2\Delta\mathcal{C}_{\mathrm{top}}^{\mathrm{EH}}$ | $\pi^2\Delta\mathcal{C}_{\mathrm{top}}^{\mathrm{EM}}$ | $N_V^{\mathrm{crit}}$ (ESU / BD) | $N_S^{\mathrm{crit}}$ | $a_{\mathrm{Cas}}^{\mathrm{grav}}/\ell_P$ | $L_{\min}(a_{\mathrm{Cas}})/\ell_{\mathrm{sp}}^{\mathrm{red}}$ | $s_{\max}(a_{\mathrm{Cas}})$ | $L_{\min}(\lvert \Xi_{\mathrm{stat}}\rvert=1)/\ell_{\mathrm{sp}}^{\mathrm{red}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\mathbb{R}P^3$ | $+0.8123$ | $+0.1875$ | NONE / NONE | 26 | $0.8303$ | $0.36794$ | $2.1660$ | $0.55958$ |
| $S^3/Q_8$ | $+4.5285$ | $+1.6875$ | NONE / NONE | 27 | $1.9606$ | $0.43438$ | $6.0380$ | $0.46714$ |
| $S^3/T^*$ | $+23.7555$ | $+3.4653$ | NONE / NONE | 5 | $4.4905$ | $0.66326$ | $15.8370$ | $0.50437$ |
| $S^3/O^*$ | $+52.8614$ | $+0.4653$ | NONE / NONE | 4 | $6.6985$ | $0.74205$ | $20.6437$ | $0.43188$ |
| **$S^3/I^*$** | **$+129.0613$** | **$-31.0681$** | **5 (4.154) / 4 (3.880)** | **3** | **$10.4666$** | **$0.92758$** | **$32.8647$** | **$0.43594$** |

*(Note: The final column uses the static-radius focusing proxy $\Xi_{\mathrm{static}} = -\frac{1}{2}\ln(1 - 2(a_{\mathrm{Cas}}/a)^2\sin^2\chi_{\mathrm{inj}}) = 1$, since dynamical $dS_4$ with $K_{\mathrm{top}} > 0$ has no bounce for $a_b < \sqrt{2}\,a_{\mathrm{Cas}}$.)*

For conformal Standard Model matter plus 1 graviton on $S^3/I^*$, $\pi^2 \Delta\mathcal{C}_{\mathrm{top}}^{\mathrm{SM+grav}} = +3817.96 + 129.06 = +3947.03 > 0$ (strongly focusing). Across all 5 ADE space forms, the exact algebraic identity $s_{\max}(a_{\mathrm{Cas}})/N_{\mathrm{sp}} = 2\pi^2(\sin\chi_{\mathrm{inj}}/\chi_{\mathrm{inj}})^2(L_{\min}(a_{\mathrm{Cas}})/\ell_{\mathrm{sp}}^{\mathrm{red}})^2$ holds to $3.55 \times 10^{-15}$.

### 5.2 Non-Abelian $SU(2)$ Flat-Holonomy Focusing Switch on $S^3/I^*$ (`CLM-T2-03`)

Because $\pi_1(S^3/I^*) = I^*$ has $H_1(S^3/I^*;\mathbb{Z}) = 0$ but admits three inequivalent flat $SU(2)$ representations $\rho_0, \rho_1, \rho_2$ with locally vanishing curvature $F_{\mu\nu} \equiv 0$, the non-abelian vacuum holonomy is completely invisible inside any single sub-injectivity causal diamond ($\chi < \pi/10$) yet shifts the 1-loop Casimir null energy in the weakly coupled deconfined UV regime ($g_{\mathrm{YM}}(1/a) \ll 1$, $L_{\min} \ll \Lambda_{\mathrm{YM}}^{-1}$, per `CLM-T2-03`; in the confined IR regime $L_{\min} \gg \Lambda_{\mathrm{YM}}^{-1}$, the mass gap exponentially suppresses the YM Casimir shift as $O(e^{-m_{\mathrm{gap}} L_{\min}})$):

| Flat Vacuum | $E_{\mathrm{Cas}}^{\mathrm{YM}}$ $[\hbar/a]$ | $\pi^2\Delta\mathcal{C}_{\mathrm{top}}^{\mathrm{YM}}$ | $\pi^2\Delta\mathcal{C}_{\mathrm{net}}$ ($1g + 1\,SU(2)$) | $\pi^2\Delta\mathcal{C}_{\mathrm{net}}$ ($1g + 2\,SU(2)$) | $\Delta s_*^{\mathrm{tip}}$ (YM only) | $\Delta s_*^{\mathrm{tip}}$ ($1g + 1\,SU(2)$) | $\Delta s_*^{\mathrm{tip}}$ ($1g + 2\,SU(2)$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\rho_0$ | $-349/225$ | $-93.2042$ | $+35.8571$ (focus) | $-57.3470$ (defocus) | $+1.6873$ | $-0.6491$ | $+1.0382$ |
| $\rho_1$ | $-221/150$ | $-88.5375$ | $+40.5238$ (focus) | $-48.0137$ (defocus) | $+1.6028$ | $-0.7336$ | $+0.8692$ |
| $\rho_2$ | $+79/150$ | $+31.4625$ | $+160.5238$ (focus) | $+191.9863$ (focus) | $-0.5696$ | $-2.9060$ | $-3.4756$ |

Crucially, adding the 1-loop pure-graviton contribution prevents a single $SU(2)$ gauge group ($N_{\mathrm{SU(2)}} = 1$) from flipping the net focusing sign; **at least $N_{\mathrm{SU(2)}} \ge 2$ copies** (ESU crossover $1.3847$ for $\rho_0$, $1.4577$ for $\rho_1$; BD@bounce $1.2903$ and $1.3584$, logged in `02_solve.json`) are required for the locally flat holonomy transition $\rho_0/\rho_1 \leftrightarrow \rho_2$ to flip the sign of total null-cone focusing.

### 5.3 EFT Consistency of Chandrasekaran's Finite-Dimensionality Bound

Across all 5 ADE space forms and for every dark-photon count $N_V \in [0, \infty)$ on $S^3/I^*$:
- In the EFT-controlled macroscopic regime ($a \gg a_{\mathrm{Cas}}$, $L_{\min}(a) \gg \ell_{\mathrm{sp}}^{\mathrm{red}}$), the 1-loop Casimir null stress tensor produces a strictly finite, bounded, $a$-independent shift $\Delta s_* = O(1)$ (or $O(N_V)$) to the causal-diamond area ceiling. Thus Chandrasekaran's finite-dimensionality bound $\dim\mathcal{H}_{\mathcal{D}} < \infty$ remains completely intact throughout the domain of validity of semiclassical EFT.
- Attempting to drive $|\Xi| \sim O(1)$ requires shrinking $a$ to $a_{\mathrm{break}}$ where $L_{\min}(a_{\mathrm{break}})/\ell_{\mathrm{sp}}^{\mathrm{red}} \in [0.0295, 0.5596] < 1$ (and $L_{\min}(a_{\mathrm{Cas}})/\ell_{\mathrm{sp}}^{\mathrm{red}} \le 0.92758 < 1$), strictly inside `CLM-T3-01`'s species-scale breakdown zone. **KT4 Verdict: PASS.**

---

## 6. Domain of Validity and Explicit Caveats

1. **Vacuum and Stress-Tensor Prescription (On-Shell Global $dS_4/\Gamma$ Bunch--Davies vs. Static-ESU TT Limit):** The on-shell vacuum pure-GR background is global $dS_4/\Gamma$ (whereas static $\mathbb{R}_t \times S^3/\Gamma$ is only the physical-TT-only $H \to 0$ limit, not an on-shell vacuum solution). On global $dS_4/\Gamma$, $\Delta T_{uu}^{\mathrm{BD\text{-}TT}}$ uses the Ford--Parker / exponential-parameterization minisuperspace TT effective stress tensor: the leading $a^{-4}$ radiation piece ($+110.7927$ on $S^3/I^*$, alone giving $N_V^{\mathrm{crit}} = 4$) is prescription-independent, whereas the subleading $O(a_{\mathrm{dS}}^{-2} a^{-2})$ curvature piece ($+9.4723$, $7.9\%$ of the bounce null-energy coefficient $\frac{3}{4}a_{\mathrm{dS}}^4\Delta(\rho+P)|_{\text{bounce}}$) differs from linear-metric $\langle G_{\mu\nu}^{(2)}[h]\rangle$ by $O(H_{\mathrm{dS}}^2\langle h^2\rangle)$ prescription terms ($H_{\mathrm{dS}} = a_{\mathrm{dS}}^{-1}$). Both prescriptions confirm strictly positive focusing across all 5 ADE space forms (bounce difference $6.23\%$--$8.72\%$; $6.60\%$--$6.82\%$ on $S^3/I^*$; integrated $\Delta s_*$ difference $2.35\%$--$8.43\%$), while shifting the integer dark-photon threshold on $S^3/I^*$ from $N_V \ge 5$ (ESU) to $N_V \ge 4$ (Bunch--Davies).
2. **Weak-Coupling Deconfined Regime for Non-Abelian $SU(2)$ Switch (`CLM-T2-03`):** The 1-loop $\mathrm{Ad}(\rho_i)$-twisted pure-YM Casimir energies on $S^3/I^*$ hold strictly in the weakly coupled deconfined UV regime $g_{\mathrm{YM}}(1/a) \ll 1$ ($L_{\min} \ll \Lambda_{\mathrm{YM}}^{-1}$); in the confined IR regime $L_{\min} \gg \Lambda_{\mathrm{YM}}^{-1}$, confinement exponentially suppresses the YM Casimir shift as $O(e^{-m_{\mathrm{gap}} L_{\min}})$.
3. **Static Round Metric for Anisotropic Quotients:** The anisotropic stress tensors $(u, p_\parallel, p_\perp)$ on $L(p,1)$ and $S^3/D_m^*$ are evaluated at the round metric $\lambda_B = 1$ using the exact scale-covariant 3D Lichnerowicz operator $-\Delta_2 - 2 R_{ikjl} h^{kl}$. As proved in `CLM-RET-01` and `CLM-T3-02`, $O(100\%)$ Casimir shear drives dynamical Berger-sphere squashing ($a_3 \neq a_\perp$) once backreaction becomes significant near $a \sim a_{\mathrm{Cas}}$.
4. **Species-Scale Cutoff (`CLM-T3-01`):** Across all 5 ADE space forms and all $N_V \ge 0$, the nonperturbative $|\Xi| \sim O(1)$ regime sits strictly below the Dvali species scale ($L_{\min}/\ell_{\mathrm{sp}}^{\mathrm{red}} < 1$), where 1-loop semiclassical gravity EFT breaks down.
5. **Semiclassical Backreaction vs. Local Null-Cone Algebra:** Chandrasekaran (`arXiv:2611.34924`) quantizes pure vacuum GR inside a single causal diamond. The global 1-loop Casimir stress tensor enters semiclassical Raychaudhuri focusing as a c-number vacuum expectation value $\langle T_{uu}\rangle$ shifting the macroscopic area ceiling by a finite constant $\Delta s_*$, without modifying the local single-diamond canonical algebra.
