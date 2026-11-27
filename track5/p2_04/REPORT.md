# Stream Report: `p2_04_graviton_invariance_and_berger_stability` (`CLM-P2-04`)

**Title:** Parameterization-Invariant 1-Loop Graviton Focusing, Topological Prolate Stabilization, and Anisotropic Berger-Sphere Casimir Dynamics on $S^3/\Gamma$
**Date:** 2026-11-26
**Scripts & Data:** `01_derive.py` (`01_derive.json`), `02_solve.py` (`02_solve.json`), `03_kill_tests.py` (`03_kill_tests.json`), canonical `results.json`
**Automated Audit (`physics_audit_results`):** `PASS` (0 P0, 0 P1, 0 P2; 4/4 kill tests passed)
**Provenance Note:** Graduates surviving discoveries of `sandbox/sbx_02_graviton_prescription_and_L3_berger/` and executes from first principles the three targets planned for `sandbox/sbx_03_sachs_shear_and_oblate_dynamics/` (which was not pre-created prior to this graduation session).

## 1. Executive Summary & Major Refinements

This formal Phase 2 stream resolves all four Kill-Tests (`KT1`--`KT4`) with explicit analytical proofs, 50-digit `mpmath` conservation checks, Kahan-compensated Gaussian heat-kernel numerics across 15 spherical space forms up to $L(24,1)$ and $S^3/D_{12}^*$, an exact Spin-1 Maxwell Berger-$S^3$ matched control (`MC4`), and dual-solver (`DOP853` vs `Radau`) axisymmetric Bianchi IX integrations:

1. **KT1 (Exact Parameterization Invariance, Sachs-Einstein Shear Accounting, and Corrected $N_V^{\mathrm{crit}}(I^*)$):** Across the 1-parameter metric family $g_{ij}^{(\alpha)} = a^2(\gamma_{ij} + h_{ij} + \alpha (h^2)_{ij})$, the 24.4% to 37.7% discrepancy (26.7% on $S^3/I^*$) between Ford-Parker ($\alpha = 1/2$) and linear-metric ($\alpha = 0$) graviton backreaction cancels identically to `1.71e-49` across all 5 ADE global $dS_4/\Gamma$ space forms against the $O(h^2)$ null-geodesic shift. The exact Sachs-Einstein identity $-R_{\mu\nu}^{(2)}[g] k^\mu k^\nu \equiv \sigma_{AB}^{(1)}\sigma^{(1)AB} + \frac{d}{d\lambda}\delta\theta_\alpha^{(2)}$ (verified to `8.71e-24`) proves zero double-counting of gravitational-wave shear focusing. Full 2nd-order null-geodesic integration (general-angle nonlinear Gowdy shooting, including $\phi = \pi/4$, plus curved-$S^3$ triaxial Bianchi IX $p=3$ geodesics) reveals the strictly $\alpha$-independent Schur-complement metric-shear term $+\frac{1}{2} h_{nA} h_{nA} \to +\frac{1}{10}\langle h^2\rangle$ omitted by a straight coordinate line. This retracts the incomplete value $N_V^{\mathrm{crit}}(I^*) = 4$ and establishes the parameterization-invariant crossover **$N_V^{\mathrm{crit}}(I^*) \in \{5, 6\}$** (`x = 4.8208 -> 5` for the Schur-complement geodesic baseline; `x = 5.0328 -> 6` for the full 2nd-order fixed-endpoint WKB geodesic total).
2. **KT2 (Topological Prolate Stabilization Theorem & Interior Saturation Scan):** Exact SymPy symbolic proofs for ALL $j$ establish the stretched-mode spectrum $\omega^2 = 4(j+2)(j+2-\lambda_B^2)/\lambda_B^2$, the universal oblate instability $\lambda_{\mathrm{oblate}} = 1/2$ from $(j=0, M_L=0)$, strict positivity of the entire axisymmetric $M_L = 0$ TT sector for ALL $j \ge 1$ and ALL $\lambda_B > 0$ (via odd-parity divergence invertibility and Sylvester positivity of the even-parity $2\times 2$ block, proving exact $(-1)^j$ $D_m^*$ parity), and strict positivity of sub-stretched $|M_L|=j+1$ for all $j \ge 1, \lambda_B \ge 1$ with a unique isolated prolate tachyon at $(j=1/2, |M_L|=3/2, \lambda_B = 3.243808)$. These analytic proofs lock exact upper bounds on $\lambda_{\mathrm{prolate}}(\Gamma)$; finite-cutoff numerical scans across $2j \le 40$ on both oblate $\lambda_B \in [0.51, 0.95]$ (min non-$(0,0)$ eigenvalue `+8.2792`) and prolate $\lambda_B \in [1.2, 6.0]$ verify saturation with zero interior undercuts.
3. **KT3 (15-Quotient Landscape, Local 3D-TT $S^3_{\lambda_B}$ Renormalization Bound, and Spin-1 Maxwell Control):** Kahan-compensated Schwinger Gaussian heat-kernel evaluation matches 50-digit `p2_03` across all 15 quotients up to $L(24,1)$ and $S^3/D_{12}^*$ to max relative error `7.08e-07` in $u(1)$ and `1.87e-06` in $u'(1)$ (0 non-positive modes dropped on $[0.52, 1.38]$). For the 3D-TT operator $\Delta_L^{(3D)}$ on unquotiented Berger $S^3_{\lambda_B}$, $B_2 = -\frac{7}{48} R^{(3)}$ holds to `2.22e-10`, $A_4 = -\frac{5}{192}(R^{(3)})^2 + \frac{63}{320} C_{(4)}^2$ holds to `8.64e-07` on $[0.80, 1.20]$ (`1.37e-06` on $[0.66, 1.30]$), and exact $A_4''(1) = +\frac{554}{15} = +36.933333$ (4th-order FD `+36.934614`) yields $\pi^2 u''_{\mathrm{ren}}(S^3; 1; \mu) = +67.64 - 36.9334\ln(\mu\bar{a})$ (2-decade band `[-17.41, +152.68]`). Thus shallow $O(1)$ wells on $Q_8, L(6,1), D_3^*, L(3,1)$ are scheme-sensitive (while linearized slopes $u'(1)$ remain 100% scheme-independent by $SO(4)$ isotropy), whereas $q \ge 14$ is unconditionally 2-decade scheme-robust. Under the exact pure-gravity Dvali species scale $\ell_{\mathrm{sp}}^{(2)} = \sqrt{2}\,\ell_{\mathrm{Pl}} = 4\sqrt{\pi}\,\ell_P = 7.089815\,\ell_P$, Class 4 splits into **sub-species Class 4a** ($L(11,1), D_6^*, L(14,1)$ with static $L_{\min}/\ell_{\mathrm{sp}}^{(2)} = 0.697, 0.763, 0.939 < 1$) and **super-species Class 4b** starting statically at $L(16,1)$ (`1.092`) and reaching `1.682` at $L(24,1)$.
4. **KT4 (Coupled Bianchi IX Dynamics with Fixed-$\mu$ Trace Anomaly & Sub-Planckian Starobinsky $f(R)$):** Both traceless and exact covariantly conserved fixed-$\mu$ local trace-anomaly stress tensors satisfy 50-digit Bianchi conservation (`<= 1.34e-51` GR, `<= 1.21e-51` Starobinsky). Across 12 `DOP853` vs `Radau` pairs, max solver difference is `2.833e-10`, and max un-clamped relative Hamiltonian residual is `1.791e-10` (`2.532e-10` across all 146 runs). On pure-GR + $\Lambda$ bounces, genuine super-species oblate runaway ($\min L_{\min}/\ell_{\mathrm{sp}}^{(2)} > 1$ all the way to $\lambda_B = 0.525$) commences at $L(16,1)$ ($a_b = 19\,\ell_P$, `1.047`) and becomes deeply super-species on $L(20,1)/D_{10}^*$ (`1.323`) and $L(24,1)/D_{12}^*$ (`1.619`, invariant across fixed-$\mu$ scales $\mu\bar{a}_b \in \{0.1, 1, 10\}$). On Starobinsky $f(R)$ ($\sigma_t = 4.0$, $N_{\mathrm{sp}} = 3$ with $\ell_{\mathrm{sp}}^{(3)} = \sqrt{3}\,\ell_{\mathrm{Pl}} = 8.683\,\ell_P$), $a_b = 10\,\ell_P$ has super-Planckian curvature $R_0/M_{\mathrm{Pl}}^2 \approx 3.64 > 1$ (outside EFT); at genuinely sub-Planckian curvature $a_b = 36\,\ell_P$ ($R_0/M_{\mathrm{Pl}}^2 = 0.588 < 1, 2\sigma_t r_0 = 0.78 < 1$), **even Starobinsky $f(R)$ gravity does NOT rescue $L(24,1)$ or $D_{12}^*$ from oblate runaway to $\lambda_B = 0.525$** (remaining strictly super-species under $N_{\mathrm{sp}} = 3$ at `1.067` / `1.066`, subject to omitting the 1-loop massive-scalaron Casimir energy).

## 2. Conventions (Locked to `intent/global.md`, `CLM-T3-01`, `CLM-P2-03`)

- Metric signature `(-,+,+,+)`; reduced Planck units $8\pi G_N = M_{\mathrm{Pl}}^{-2} = 1$, $c = 1$, $\hbar = 1$; unreduced Planck length $\ell_P = \sqrt{G_N\hbar} = \ell_{\mathrm{Pl}}/\sqrt{8\pi}$.
- Dvali species scale: for pure GR ($N_{\mathrm{sp}} = 2$), $\ell_{\mathrm{sp}}^{(2)} = \sqrt{2}\,\ell_{\mathrm{Pl}} = 4\sqrt{\pi}\,\ell_P = 7.089815\,\ell_P$; for Starobinsky $f(R)$ ($N_{\mathrm{sp}} = 3$: 2 graviton helicities + 1 scalaron), $\ell_{\mathrm{sp}}^{(3)} = \sqrt{3}\,\ell_{\mathrm{Pl}} = \sqrt{24\pi}\,\ell_P = 8.683215\,\ell_P$.
- Left-invariant $S^3$ coframe: $d\omega^a = -\epsilon_{abc}\,\omega^b \wedge \omega^c$, $[X_a, X_b] = 2\epsilon_{abc} X_c$, $\mathrm{dvol} = +\omega^1 \wedge \omega^2 \wedge \omega^3$.
- Fixed-volume Berger metric: $ds_3^2 = \bar{a}^2[\lambda_B^{-2/3}((\omega^1)^2 + (\omega^2)^2) + \lambda_B^{4/3}(\omega^3)^2]$, $\bar{a} = (a_\perp^2 a_3)^{1/3}$, $\lambda_B = a_3/a_\perp$, $\mathrm{Vol}(S^3_{\lambda_B}/\Gamma) = 2\pi^2 \bar{a}^3 / |\Gamma|$.

## 3. KT1: 1-Parameter Field-Redefinition Invariance, Sachs-Einstein Shear Accounting, and Exact $N_V^{\mathrm{crit}}(I^*)$

### 3.1 Exact 1-Parameter Metric Redefinition Identity

For the 1-parameter transverse-traceless metric family $g_{ij}^{(\alpha)} = a(\eta)^2(\gamma_{ij} + h_{ij} + \alpha (h^2)_{ij} + O(h^3))$ on spatially curved ($K = +1$) or flat ($K = 0$) FLRW, the quadratic determinant shift induces an effective isotropic conformal-factor shift
$$\Phi_\alpha(\eta) = \frac{1}{6}\left(\alpha - \frac{1}{2}\right)\langle h_{kl} h^{kl}\rangle,$$
shifting the effective 1-loop graviton energy density and pressure relative to Ford-Parker ($\alpha = 1/2$, $\Phi_{1/2} \equiv 0$) by
$$\delta\rho_\alpha = -\frac{6\mathcal{H}}{a^2}\Phi_\alpha' + \frac{6K}{a^2}\Phi_\alpha, \qquad \delta P_\alpha = \frac{2}{a^2}(\Phi_\alpha'' + 2\mathcal{H}\Phi_\alpha' - K\Phi_\alpha).$$
Exact 50-digit nonlinear 4D Einstein-tensor verification across 6 values $\alpha \in \{-1, 0, 1/4, 1/2, 1, 3/2\}$ gives max error `4.976e-23` on triaxial Bianchi IX ($K=+1, k^2=8$), `5.567e-22` on 1+1D Gowdy waves ($K=0, k \in \{1, 3, 7\}$), and exact covariant conservation residual `1.165e-49`. Because $\delta\rho_\alpha + \delta P_\alpha$ integrates identically against the $dS_4$ horizon kernel, the background conformal-time shift transforms as $\Delta\eta_{\mathrm{BR}}(\alpha) = \Delta\eta_{\mathrm{BR}}^{\mathrm{FP}} + \frac{1}{6}(\alpha - \frac{1}{2})\int_{-\pi/2}^{+\pi/2} \Delta\langle h^2\rangle\,d\eta$.

### 3.2 Exact Sachs-Einstein Identity & Complete 2nd-Order Null-Geodesic Decomposition

Along a conformal null ray $k = \partial_\eta + \partial_z$ through a gravitational wave $h_{ij}^{(\alpha)}$ on flat space or flat $dS_4$, the unscaled 4D Ricci contraction obeys the exact identity (with zero extra $a^2$ factor on the LHS):
$$-R_{\mu\nu}^{(2)}[g] k^\mu k^\nu \equiv \sigma_{AB}^{(1)}\sigma^{(1)AB} + \frac{d}{d\lambda}\delta\theta_\alpha^{(2)},$$
verified to `8.710e-24` across both flat space and $dS_4$ for all $\alpha \in [-0.5, 1.25]$. Thus gravitational-wave shear focusing $-\langle\sigma^2\rangle$ is NOT an additive correction on top of $-\langle R_{\mu\nu}^{(2)} k^\mu k^\nu\rangle$: it IS the exact microscopic source of semiclassical backreaction, proving zero double-counting.

Computing the full 2nd-order fixed-endpoint null-geodesic arrival-time shift $\Delta\chi / (L\langle h^2\rangle)$ on a stochastic isotropic graviton background reveals four distinct $O(h^2)$ contributions:

| Contribution | Exact $S^2$-Averaged Coefficient | $\alpha$-Dependence | Physical Origin |
|---|---:|:---:|---|
| Straight coordinate line $(g_{nn}^{(\alpha)})^{-1/2} - 1$ | $-\frac{\alpha}{6} + \frac{1}{20}$ | Dependent | Restricted $1\times 1$ metric component along fixed line |
| **Schur-complement metric shear** $+\frac{1}{2} h_{nA} h_{nA}$ | $+\frac{1}{10}$ | **Invariant** | Exact inverse metric $\sqrt{g^{(\alpha)nn}} - 1$ (transverse wave-ray tilt) |
| **Schur-complement geodesic baseline** | $-\frac{\alpha}{6} + \frac{3}{20}$ | Cancels $\Delta\eta_{\mathrm{BR}}(\alpha)$ | Exact inverse-metric speed (verified on curved $S^3$ $p=3$) |
| Spatial-gradient Fermat bending | $+\frac{1}{40}$ | **Invariant** | Transverse geodesic deflection $\delta x_A$ |
| Longitudinal-redshift cross-term | $-\frac{1}{120}$ | **Invariant** | Coupled $\delta k_n \cdot h_{nn}$ path integral |
| **Full 2nd-order fixed-endpoint WKB geodesic** | **$\frac{1 - \alpha}{6}$** | **Cancels $\Delta\eta_{\mathrm{BR}}(\alpha)$** | **Total flat-space / WKB fixed-endpoint arrival shift** |

- **General-Angle Nonlinear Gowdy Shooting Check (including $\phi = \pi/4$):** Integrating exact 4D null geodesics across $(\theta, \phi)$ on standing Gowdy waves reproduces $\langle\Delta\chi/(L\langle h^2\rangle)\rangle_{S^2} = \frac{1-\alpha}{6}$ to max error `4.98e-08` (max null constraint residual `5.00e-15`), including the non-axisymmetric $\phi = \pi/4$ check (`+0.250000`, `+0.125000`, `0.000000` for $\alpha = 0, 1/2, 1$).
- **Curved-$S^3 = SU(2)$ Triaxial Bianchi IX ($p = 3$) Nonlinear Geodesics:** Integrating exact Hamiltonian null geodesics on curved $S^3$ via 12 Gauss-Legendre polar angles (exploiting axisymmetry $c_1 = c_2$) $\times$ 2 quadrature phases $\times$ 2 signs ($\pm\epsilon$, 48 ray shootings) gives `coeff = -alpha/6 + 0.149528` (within `-4.72e-4` of the exact Schur-complement value `3/20 = 0.150000`, with exact $-\alpha/6$ slope invariance error `3.09e-09`).
- **Direct 3D TT $S^2$-Averaged $2\times 2$ Screen-Area Determinant Check:** Projecting an explicit non-diagonal traceless symmetric 3D tensor $h_{ij}$ onto the $2\times 2$ screen metric $q_{AB}(\epsilon) = \delta_{AB} + \epsilon h_{AB} + \alpha\epsilon^2 (h^2)_{AB}$ across $16\times 16$ $S^2$ quadrature directly confirms both $\langle\delta\sqrt{\det q_{AB}}/\sqrt{\det\bar{q}_{AB}}\rangle_{S^2} = (\frac{\alpha}{3} - \frac{1}{10})\langle h^2\rangle$ (max err `5.08e-09`, giving total $\alpha$-invariant fractional area shift $+\frac{1}{15}\langle h^2\rangle$) and $\langle\delta\ln\sqrt{\det q_{AB}}\rangle_{S^2} = (\frac{\alpha}{3} - \frac{7}{60})\langle h^2\rangle$ (max err `3.86e-09`, giving total $\alpha$-invariant log-determinant shift $+\frac{1}{20}\langle h^2\rangle$).

### 3.3 50-Digit Parameterization-Invariant Observables Across All 5 ADE Space Forms & Retraction of $N_V^{\mathrm{crit}}(I^*) = 4$

Combining $\Delta\eta_{\mathrm{BR}}(\alpha)$ with the Schur-complement geodesic shift gives $\Delta\eta_{\mathrm{tot}}^{\mathrm{Schur}} = \Delta\eta_{\mathrm{BR}}^{\mathrm{FP}} + \frac{1}{15}\int\Delta\langle h^2\rangle\,d\eta$, while the full 2nd-order WKB geodesic gives $\Delta\eta_{\mathrm{tot}}^{\mathrm{WKB}} = \Delta\eta_{\mathrm{BR}}^{\mathrm{FP}} + \frac{1}{12}\int\Delta\langle h^2\rangle\,d\eta$. Both totals are strictly $\alpha$-invariant to `1.71e-49` across all 5 ADE space forms and 6 values of $\alpha$, and the fractional 2D screen-area shift $\langle\delta\sqrt{\det g_{\perp}}/\sqrt{\det\bar{g}_{\perp}}\rangle_{\mathrm{tot}} = +\frac{1}{15}\langle h^2\rangle$ is $\alpha$-invariant to `6.41e-50`:

| ADE Space Form | $\lvert \Gamma\rvert$ | $\pi^2\Delta\eta_{\mathrm{BR}}^{\mathrm{FP}}$ | $\pi^2\Delta\eta_{\mathrm{BR}}^{\mathrm{LIN}}$ | $\pi^2\Delta\eta_{\mathrm{tot}}^{\mathrm{Schur}}$ ($+\frac{1}{15}$) | $\pi^2\Delta\eta_{\mathrm{tot}}^{\mathrm{WKB}}$ ($+\frac{1}{12}$) |
|---|---:|---:|---:|---:|---:|
| `RP3` | 2 | +0.611325 | +0.432916 | **+0.754052** | **+0.789734** |
| `Q8` | 8 | +3.360218 | +2.093016 | **+4.373980** | **+4.627420** |
| `T*` | 24 | +17.878451 | +13.302390 | **+21.539301** | **+22.454513** |
| `O*` | 48 | +39.858449 | +30.152968 | **+47.622833** | **+49.563929** |
| `I*` | 120 | +96.935787 | +71.067740 | **+117.630225** | **+122.803835** |

Comparing against the exact $S^3/I^*$ single-Maxwell defocusing $\pi^2\Delta\eta_{\mathrm{EM}}(I^*) = -24.400794$:
- Ford-Parker backreaction alone (`BR_FP`): $x = 3.972649 \implies N_V^{\mathrm{crit}} = 4$ (parameterization-dependent).
- Naive linear-metric backreaction alone (`BR_LIN`): $x = 2.912518 \implies N_V^{\mathrm{crit}} = 3$ (parameterization-dependent).
- Incomplete straight coordinate line ($-1/30$, omits Schur metric shear): $x = 3.548596 \implies N_V^{\mathrm{crit}} = 4$ (**retracted**).
- **Schur-complement geodesic baseline ($+1/15$):** $x = 4.820754 \implies N_V^{\mathrm{crit}}(I^*) = 5$ (**parameterization-invariant**).
- **Full 2nd-order WKB geodesic total ($+1/12$):** $x = 5.032780 \implies N_V^{\mathrm{crit}}(I^*) = 6$ (**parameterization-invariant**).

Because $S^3/I^*$ has zero graviton modes for $p = 3..12$ ($p_{\min}(I^*) = 13$), any sub-WKB curved-$S^3$ reduction on $p = 3..12$ enters only the subtracted $S^3$ reference and pushes $x(I^*)$ slightly above `5.03`. We therefore report the exact parameterization-invariant crossover honestly as **$N_V^{\mathrm{crit}}(I^*) \in \{5, 6\}$**.

## 4. KT2: Topological Prolate Stabilization Theorem & Interior Numerical Verification

### 4.1 Theorem 2 (Analytically Proved Sectors for All $j$ and All $\lambda_B > 0$)

For the scale-covariant 3D spin-2 TT Lichnerowicz operator $\Delta_L^{(3D)} = -\Delta_2 - 2 R_{ikjl} h^{kl}$ on the unit-perpendicular Berger sphere $S^3_{\lambda_B}$ ($[X_a, X_b] = 2\epsilon_{abc} X_c$), exact SymPy block reduction in the $SU(2)_R \times U(1)_L$ basis $(j, M_L)$ proves the following four statements for ALL $2j \in \mathbb{Z}_{\ge 0}$:

1. **Stretched sector ($|M_L| = j + 2$, 1D TT block):** $\omega^2(j, \pm(j+2); \lambda_B) = \frac{4(j+2)(j+2-\lambda_B^2)}{\lambda_B^2}$, crossing zero at exact $\lambda_{\mathrm{tach}}(j) = \sqrt{j+2}$.
2. **Universal oblate mode ($j = 0, M_L = 0$):** $\omega^2(0, 0; \lambda_B) = \frac{32}{3}\lambda_B^2 - \frac{8}{3}$, crossing zero at universal $\lambda_{\mathrm{oblate}} = 1/2$ (invariant under every $\Gamma \subset SU(2)_L$).
3. **Strict positivity of the entire axisymmetric $M_L = 0$ sector for ALL $\lambda_B > 0$:**
   - For $j = 1, M_L = 0$ (1 TT mode, even under $j$-flip): $\omega^2 = \frac{8(4\lambda_B^4 + 10\lambda_B^2 + 1)}{3\lambda_B^2 + 2} > 0$ for all $\lambda_B > 0$.
   - For ALL $j \ge 2, M_L = 0$, the $3 \times 5$ divergence decouples by parity into a $2 \times 2$ odd-parity block with exact determinant $\det(\mathrm{Div}_{\mathrm{odd}}) = 2\sqrt{j(j-1)(j+1)(j+2)} \neq 0$ (proving zero odd-parity TT modes and establishing the exact $(-1)^j$ $S^3/D_m^*$ selection rule!) and a $1 \times 3$ even-parity block whose exact $2 \times 2$ TT Lichnerowicz matrix has strictly positive trace and strictly positive determinant (every polynomial coefficient in $u = j - 2 \ge 0$ and $\lambda_B^2 > 0$ is strictly positive, minimum coefficient $\ge +8$) for ALL $j \ge 2$ and ALL $\lambda_B > 0$.
4. **Sub-stretched sector ($|M_L| = j + 1$, 1D TT block):** Exact closed-form rational eigenvalue satisfies $\lim_{\lambda_B \to \infty} \omega^2 = 4j - 4$; writing $j = 1 + u$ ($u \ge 0$) and $\lambda_B^2 = 1 + y$ ($y \ge 0$), every numerator coefficient is $\ge 4.0 > 0$ and every denominator coefficient is $\ge 1.0 > 0$, proving strict positivity for ALL $j \ge 1, \lambda_B \ge 1$. At $j = 1/2, |M_L| = 3/2$, there is a unique isolated prolate tachyon crossing at $\lambda_B = 3.243808$ (strictly above $L(3,1)$'s $\sqrt{3}$ and projected out for all $L(q \ge 4, 1)$ and $S^3/D_m^*$).

**Analytic Corollary (Exact Upper/Lower Bounds):** Because quotient projection retains stretched modes iff $2(j+2)$ is divisible by $q$ (for $L(q,1)$) or $2m$ (for $S^3/D_m^*$), Theorem 2 establishes the exact analytic upper bounds $\lambda_{\mathrm{prolate}}(L(q,1)) \le \sqrt{2}$ ($q \in \{1,2,4\}$), $\le \sqrt{3}$ ($L(3,1)$), $\le \sqrt{q/2}$ ($q \ge 5$), $\le \sqrt{m}$ ($S^3/D_m^*$), and lower bound $\lambda_{\mathrm{oblate}}(\Gamma) \ge 1/2$.

### 4.2 Finite-Cutoff Numerical Verification of Interior Saturation ($2j \le 40$)

To verify that no general interior non-stretched sector $0 < |M_L| \le j$ undercuts Theorem 2's bounds, we diagonalize every block across $2j \le 40$ on both prolate ($\lambda_B \in [1.2, 6.0]$) and oblate ($\lambda_B \in [0.51, 0.95]$) grids:
- **Prolate interior scan ($2j \le 40, \lambda_B \in [1.2, 6.0]$):** Only the proved $(j=1/2, |M_L|=3/2)$ tachyon appears (above $\lambda_B = 3.243808$); zero other non-stretched prolate tachyons occur.
- **Oblate interior scan ($2j \le 40, \lambda_B \in [0.51, 0.95]$):** Minimum eigenvalue across ALL modes other than $(0,0)$ is `+8.279239` at `(lam, 2j, M_L) = [0.95, 4, 0.0]` (zero oblate undercuts of $\lambda_{\mathrm{oblate}} = 1/2$).
- **Matrix & Formula Cross-Checks:** Symbolic vs numerical block max error `2.274e-13`; full $5(2j+1)\times 5(2j+1)$ Lichnerowicz matrix vs $O(1)$ block max error `2.842e-13`; 16-quotient prolate threshold table ($S^3, \mathbb{R}P^3, L(3,4,5,6,8,11,14,20; 1), Q_8, D_3^*, D_4^*, D_6^*, D_8^*, D_{10}^*$) max error `2.045e-13`.

## 5. KT3: 15-Quotient Pure-Graviton Casimir Landscape, Local 3D-TT Renormalization Bound, and Spin-1 Maxwell Control (MC4)

### 5.1 High-Precision Benchmark vs 50-Digit `p2_03` & Dynamical Class Taxonomy

Using the Kahan-compensated Schwinger Gaussian heat-kernel regulator $f_t(\bar{\omega}) = \frac{1}{2}\bar{\omega} e^{-t\bar{\omega}^2}$ (`ts_xfine` at $\lambda_B = 1$, `max_2j = 853`; `ts_fine` across the 27-point grid $\lambda_B \in [0.52, 1.38]$ with **0 non-positive eigenvalues dropped**), we match 50-digit `p2_03` across all 15 quotients to max relative error `7.08e-07` in $u(1)$ (`1.49e-07` absolute for $q \le 14$) and `1.87e-06` in $u'(1)$. Convergence ladder (`ts_fine` vs `ts_xfine`) gives max relative differences `9.85e-06` ($\lambda_B = 1.00$), `3.15e-06` ($\lambda_B = 1.25$), and `1.96e-04` ($\lambda_B = 0.60$, $\le 5.30\times 10^{-6}$ for $q \le 16$).

| Quotient | Class | $\pi^2 u_{\mathrm{top}}(1)$ | $\pi^2 u'_{\mathrm{top}}(1)$ | $\pi^2 u''_{\mathrm{top}}(1)$ | $a_{\mathrm{destab}}/\ell_P$ | $L_{\min}/\ell_{\mathrm{Pl}}$ | $L_{\min}/\ell_{\mathrm{sp}}^{(2)}$ | 2-Decade $S^3$ Ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `D_2*` | `1_round_reinforced` | +4.5285 | +0.0000 | +2.972 | stable | N/A | N/A | 0.02x |
| `L(6,1)` | `2_prolate_minimum` | +5.6190 | -1.9875 | -2.168 | 1.017 | 0.213 | 0.150 | 0.01x |
| `D_3*` | `2_prolate_minimum` | +9.3353 | -1.8300 | +2.995 | stable | N/A | N/A | 0.02x |
| `L(3,1)` | `3_L3_oblate_zero` | +0.8216 | +1.9314 | -7.319 | 1.869 | 0.781 | 0.552 | 0.05x |
| `L(4,1)` | `intermediate` | +2.0510 | -0.1050 | -0.469 | 0.473 | 0.148 | 0.105 | 0.00x |
| `L(8,1)` | `intermediate` | +9.2755 | +1.1746 | -29.869 | 3.776 | 0.592 | 0.418 | 0.20x |
| `D_4*` | `intermediate` | +14.2305 | +1.3846 | -22.986 | 3.313 | 0.519 | 0.367 | 0.15x |
| `L(11,1)` | `4a_sub_species_oblate` | +10.5474 | +26.3173 | -156.860 | 8.654 | 0.986 | 0.697 | 1.03x |
| `D_6*` | `4a_sub_species_oblate` | +16.0343 | +43.7260 | -223.275 | 10.325 | 1.078 | 0.763 | 1.46x |
| `L(14,1)` | `4a_sub_species_oblate` | -1.1029 | +96.6606 | -460.358 | 14.826 | 1.327 | 0.939 | 3.02x |
| `L(16,1)` | `4b_super_species_oblate` | -21.5098 | +183.1741 | -813.390 | 19.707 | 1.544 | 1.092 | 5.33x |
| `L(20,1)` | `4b_super_species_oblate` | -110.9808 | +500.0726 | -2059.259 | 31.356 | 1.965 | 1.389 | 13.49x |
| `D_10*` | `4b_super_species_oblate` | -98.5933 | +500.5974 | -2042.051 | 31.225 | 1.957 | 1.384 | 13.37x |
| `L(24,1)` | `4b_super_species_oblate` | -298.2326 | +1096.4781 | -4345.778 | 45.552 | 2.379 | 1.682 | 28.46x |
| `D_12*` | `4b_super_species_oblate` | -283.3676 | +1097.1079 | -4325.128 | 45.443 | 2.373 | 1.678 | 28.33x |

**Species-Scale Classification:** With the exact pure-gravity Dvali species scale $\ell_{\mathrm{sp}}^{(2)} = \sqrt{2}\,\ell_{\mathrm{Pl}} = 4\sqrt{\pi}\,\ell_P = 7.089815\,\ell_P$, Class 4a quotients ($L(11,1), D_6^*, L(14,1)$) lie **below** the species scale at their static destabilization radii ($L_{\min}/\ell_{\mathrm{sp}}^{(2)} = 0.697, 0.763, 0.939 < 1$). True **super-species static destabilization (Class 4b)** begins at $L(16,1)$ (`1.092`) and climbs to `1.682` ($L(24,1)$) and `1.678` ($D_{12}^*$).

### 5.2 Universal 3D-TT Berger-$S^3$ Local Renormalization Bound & 6 Broken Killing/CKV Modes

- **Non-Einstein 3D-TT Truncation Disclosure:** On the static non-Einstein cylinder $\mathbb{R}_t \times S^3_{\lambda_B}$ ($\lambda_B \neq 1$), $\Delta_L^{(3D)}$ is the static 3D-TT Lichnerowicz operator and omits $O((\lambda_B-1)^2)$ coupling to 4D background-stress / gauge-constraint modes; the anomaly coefficient $c_C = 63/320$ below is specific to $\Delta_L^{(3D)}$.
- **6 Broken Round-$S^3$ Killing/CKV Modes (`continuous_s3=True`):** Round $S^3$ has 10 conformal Killing vectors (6 Killing + 4 pure-gradient CKVs). Squashing breaks 2 $SU(2)_L/U(1)_L$ Killing vectors (`2j=0, |2M|=2`) and 4 pure-gradient CKVs (`2j=1, |2M|=1`), giving those 6 modes an $O(\lambda_B - 1)$ nonzero 3D divergence singular value. Retaining them via `continuous_s3=True` preserves exact $SO(4)$ continuity at $\lambda_B = 1$ (contributing $\pi^2 u_6(1) = +5.287197$, $u_6'(1) = +5.309694$, $u_6''(1) = +4.509617$). Crucially, **zero** of these 6 modes survive quotient projection on any $L(q \ge 3, 1)$ or $S^3/D_m^*$ ($m \ge 2$), so they **cancel identically** in the total energy $u_{\mathrm{tot}} = u_{\mathrm{top}} + u_{\mathrm{ren}}(S^3)$.
- **Exact Heat-Kernel Divergences & $A_4''(1) = +\frac{554}{15}$:** Subtracting $B_0 = 1/4$ and $B_2(\lambda_B) = -\frac{7}{48} R^{(3)}(\lambda_B)$ (max rel error `2.22e-10` on $[0.66, 1.30]$) yields $A_4(1) = -0.93750087$ (exact $-15/16 = -0.93750000$, err `8.69e-07`) and $A_4(\lambda_B) = -\frac{5}{192}(R^{(3)})^2 + c_C C_{(4)}^2$ with fitted $c_C = +0.19687512$ vs exact rational $63/320 = +0.19687500$ (max abs error `8.64e-07` on $[0.80, 1.20]$, `1.37e-06` on $[0.66, 1.30]$). Since $R^{(3)''}(1) = -\frac{32}{3}$ gives $[(R^{(3)})^2]''(1) = -128$, the exact second derivative is:
$$A_4''(1) = \left(-\frac{5}{192}\right)(-128) + \left(\frac{63}{320}\right)\left(\frac{512}{3}\right) = \frac{10}{3} + \frac{168}{5} = +\frac{554}{15} = +36.933333\dots$$
cross-checked directly against the 4th-order FD(`0.03`) of $A_{4,\mathrm{fit}}(\lambda_B)$ (`+36.934614`, abs err `1.28e-03`). Direct 4th-order shell-sum $SO(4)$ isotropy holds to `3.206e-09` (`u'_ren(1)` fit `-0.000336`), and
$$\pi^2 u''_{\mathrm{ren}}(S^3; 1; \mu) = +67.6352 - 36.9334\ln(\mu\bar{a}), \qquad \mu\bar{a} \in [0.1, 10] \implies [-17.41, +152.68].$$

### 5.3 Matched Control MC4: Spin-1 Maxwell on Squashed Berger $S^3_{\lambda_B}/\Gamma$

Evaluating the exact closed-form Spin-1 Maxwell $*_B d$ spectrum under the identical Gaussian heat-kernel regulator (max rel error `5.50e-07` vs exact rational `t3_01` polynomials at $\lambda_B = 1$) confirms **zero tachyons for all $\lambda_B > 0$** (`min |omega| = +0.200000` across $2j \le 40, \lambda_B \in [0.10, 6.00]$) and cleanly isolates uniquely spin-2 effects from universal 2-helicity asymptotics:

| Quotient | Spin-1 $\pi^2 u_{\mathrm{EM}}(1)$ | Spin-1 $\pi^2 u'_{\mathrm{EM}}(1)$ | Spin-1 $\pi^2 u''_{\mathrm{EM}}(1)$ | Spin-2 $\pi^2 u''_{\mathrm{S2}}(1)$ | Qualitative Comparison |
|---|---:|---:|---:|---:|---|
| `D_2*` | +1.6875 | +0.0000 | -3.367 | +2.972 | **OPPOSITE SIGN**: Spin-1 destabilizes round $Q_8$; Spin-2 reinforces roundness |
| `L(6,1)` | +0.6319 | +4.1481 | -16.921 | -2.168 | **OPPOSITE SLOPE**: Spin-1 pulls oblate; Spin-2 pulls prolate |
| `D_3*` | +2.1319 | +3.1482 | -17.006 | +2.995 | **OPPOSITE SIGN & SLOPE**: Spin-1 oblate/unstable; Spin-2 prolate/stable |
| `L(3,1)` | +0.4444 | +0.1481 | -1.046 | -7.319 | Spin-2 curvature 7.0x stronger (oblate zero-crossing) |
| `L(11,1)` | -12.0000 | +52.0000 | -194.844 | -156.860 | Same sign; approaching 2-helicity asymptotics |
| `D_6*` | -15.8681 | +72.1481 | -276.751 | -223.275 | Same sign; approaching 2-helicity asymptotics |
| `L(14,1)` | -39.8125 | +138.6666 | -514.220 | -460.358 | Within 12% of 2-helicity asymptotics |
| `L(20,1)` | -194.5125 | +585.1997 | -2154.832 | -2059.259 | Within 4.6% of 2-helicity asymptotics |
| `L(24,1)` | -420.8678 | +1218.1460 | -4477.899 | -4345.778 | Within 3.0% (but Spin-2 uniquely hits Lichnerowicz wall $\lambda_B = 1/2$) |
| `D_12*` | -414.8678 | +1214.1460 | -4478.241 | -4325.128 | Within 3.5% (Spin-2 uniquely hits Lichnerowicz wall $\lambda_B = 1/2$) |

## 6. KT4: Coupled Axisymmetric Bianchi IX Dynamics (`DOP853` vs `Radau`) & Sub-Planckian Starobinsky $f(R)$

### 6.1 Exact 50-Digit Conservation (Traceless Topological & Fixed-$\mu$ Local Trace Anomaly)

When including the local Berger-$S^3$ background $u_{\mathrm{ren}}(S^3_\lambda; \mu)$ at a **fixed physical renormalization scale $\mu$**, exact 4D covariant conservation requires the trace anomaly $T^\mu{}_\mu = p_\parallel + 2 p_\perp - u = +A_4(\lambda)/(\pi^2 \bar{a}^4)$, entering the pure-GR Raychaudhuri equation as $-A_4/(6\pi^2\bar{a}^4)$ and the Starobinsky $f(R) = R + (\sigma_t/6) R^2$ ($R = 6r$) scalaron equation as $\dot{s} = -3\bar{H}s - r/\sigma_t - A_4/(6\pi^2\sigma_t\bar{a}^4)$. 50-digit `mpmath` verification gives:
- **Traceless topological (`A_4 = 0`):** trace `0.00e+00`, cov. cons. `8.35e-53`, pure-GR Bianchi `1.34e-51`, Starobinsky Bianchi `2.92e-52`.
- **Fixed-$\mu$ local trace anomaly (`A_4 != 0`):** trace `0.00e+00`, cov. cons. `4.18e-53`, pure-GR Bianchi `1.00e-51`, Starobinsky Bianchi `1.21e-51`.

### 6.2 Pure-GR + $\Lambda$ Bounce Dynamics: Sub-Species Class 4a vs Super-Species Class 4b

On pure-GR + $\Lambda$ global $dS_4/\Gamma$ bounces, the round-$S^3$ negative control (`Control_S3`) and Class 1--3 quotients ($Q_8, L(6,1), D_3^*, L(3,1)$) remain strictly stable and expansion-quenched (`max_rel_e00 <= 1e-14`). Across Class 4a and Class 4b:

| Quotient | Class | Highest Runaway $a_b/\ell_P$ (`tot_mu`) | Min $L_{\min}/\ell_{\mathrm{sp}}^{(2)}$ along Runaway | Next Quenched $a_b/\ell_P$ | EFT Status at Wall ($\lambda_B = 0.525$) |
|---|---|---:|---:|---:|---|
| `L(11,1)` | `4a_sub_species_oblate` | 8.0 | 0.624 | 9.0 | Sub-species ($< 1$) |
| `D_6*` | `4a_sub_species_oblate` | 10.0 | 0.708 | 11.0 | Sub-species ($< 1$) |
| `L(14,1)` | `4a_sub_species_oblate` | 14.0 | 0.878 | 15.0 | Sub-species ($< 1$) |
| `L(16,1)` | `4b_super_species_oblate` | 19.0 | 1.047 | 20.0 | **SUPER-SPECIES** ($> 1$) |
| `L(20,1)` | `4b_super_species_oblate` | 30.0 | 1.323 | 32.0 | **SUPER-SPECIES** ($> 1$) |
| `D_10*` | `4b_super_species_oblate` | 30.0 | 1.321 | 32.0 | **SUPER-SPECIES** ($> 1$) |
| `L(24,1)` | `4b_super_species_oblate` | 44.0 | 1.619 | 46.0 | **SUPER-SPECIES** ($> 1$) |
| `D_12*` | `4b_super_species_oblate` | 44.0 | 1.618 | 46.0 | **SUPER-SPECIES** ($> 1$) |

**2-Decade Fixed-$\mu$ Dynamical RG Robustness on $L(24,1)$ ($a_b = 42\,\ell_P$):** Varying $\mu\bar{a}_b \in \{0.1, 1.0, 10.0\}$ produces identical super-species oblate runaway to $\lambda_B = 0.525$ with wall-hit times `t_end = 9.203, 9.146, 9.091` ($\pm 0.6\%$) and $\min L_{\min}/\ell_{\mathrm{sp}}^{(2)} = 1.537, 1.536, 1.536$.

### 6.3 Starobinsky $f(R)$ ($\sigma_t = 4.0$): Super-Planckian Artifact at $a_b = 10\,\ell_P$ vs Sub-Planckian Runaway ($N_{\mathrm{sp}} = 3$)

Because Starobinsky $f(R)$ propagates $N_{\mathrm{sp}} = 3$ gravitational degrees of freedom (2 graviton helicities + 1 scalaron of mass $M_S = M_{\mathrm{Pl}}/\sqrt{2\sigma_t} = 0.354\,M_{\mathrm{Pl}}$), we report both $\min L_{\min}/\ell_{\mathrm{sp}}^{(2)}$ (pure-GR baseline) and $\min L_{\min}/\ell_{\mathrm{sp}}^{(3)}$ ($\ell_{\mathrm{sp}}^{(3)} = \sqrt{3}\,\ell_{\mathrm{Pl}} = 8.683\,\ell_P$):

| Case | $a_b/\ell_P$ | $R_0 / M_{\mathrm{Pl}}^2$ | $2\sigma_t r_0$ | Outcome | $\lambda_{\min}$ | $\min L_{\min}/\ell_{\mathrm{sp}}^{(2)}$ | $\min L_{\min}/\ell_{\mathrm{sp}}^{(3)}$ | EFT Regime ($N_{\mathrm{sp}}=3$) |
|---|---:|---:|---:|---|---:|---:|---:|---|
| `Control_S3_ab10` | 10.0 | 3.637 | 4.85 | `starobinsky_expansion_quenched` | 0.9800 | 8.744 | 7.139 | **Super-Planckian (outside EFT)** |
| `Q8_ab10` | 10.0 | 3.616 | 4.82 | `starobinsky_expansion_quenched` | 0.9800 | 2.186 | 1.785 | **Super-Planckian (outside EFT)** |
| `D6_ab10` | 10.0 | 3.564 | 4.75 | `starobinsky_expansion_quenched` | 0.9510 | 0.739 | 0.603 | **Super-Planckian (outside EFT)** |
| `L14_ab10` | 10.0 | 3.643 | 4.86 | `starobinsky_expansion_quenched` | 0.8854 | 0.633 | 0.517 | **Super-Planckian (outside EFT)** |
| `L20_ab24_subPl_tot` | 24.0 | 0.986 | 1.32 | `oblate_runaway_to_lichnerowicz_wall` | 0.5250 | 1.063 | 0.868 | Sub-Planckian, below $\ell_{\mathrm{sp}}^{(3)}$ |
| `L24_ab36_subPl_tot` | 36.0 | 0.588 | 0.78 | `oblate_runaway_to_lichnerowicz_wall` | 0.5250 | 1.307 | 1.067 | **Sub-Planckian & Super-Species ($N_{\mathrm{sp}}=3$)** |
| `L24_ab44_subPl_tot` | 44.0 | 0.450 | 0.60 | `oblate_runaway_to_lichnerowicz_wall` | 0.5250 | 1.075 | 0.878 | Sub-Planckian, below $\ell_{\mathrm{sp}}^{(3)}$ |
| `D12_ab36_subPl_tot` | 36.0 | 0.586 | 0.78 | `oblate_runaway_to_lichnerowicz_wall` | 0.5250 | 1.305 | 1.066 | **Sub-Planckian & Super-Species ($N_{\mathrm{sp}}=3$)** |

Thus the $a_b = 10\,\ell_P$ Starobinsky quenching of Class 4a ($D_6^*, L(14,1)$) occurs at **super-Planckian curvature $R_0 \approx 3.64\,M_{\mathrm{Pl}}^2$** ($f'(R_0) \approx 5.85$, outside EFT). Inside the genuinely **sub-Planckian, perturbative-$R^2$, super-species EFT regime** (`L24_ab36_subPl_tot` and `D12_ab36_subPl_tot`: $R_0/M_{\mathrm{Pl}}^2 = 0.59 < 1$, $2\sigma_t r_0 = 0.78 < 1$, $\min L_{\min}/\ell_{\mathrm{sp}}^{(3)} = 1.067$ and $1.066 > 1$), **Starobinsky $f(R)$ gravity does NOT rescue $L(24,1)$ or $D_{12}^*$ from dynamical oblate runaway to $\lambda_B = 0.525$** (noting explicitly that the 1-loop massive-scalaron Casimir energy is omitted in these pure-graviton stress-tensor runs).

### 6.4 Independent `DOP853` vs `Radau` Dual-Solver Cross-Checks (`rtol = 1e-12, atol = 1e-14`)

| Case ID | Outcome | `DOP853` vs `Radau` | Un-clamped `e00` (`DOP853`) | Un-clamped `e00` (`Radau`) | $\min L_{\min}/\ell_{\mathrm{sp}}^{(2)}$ |
|---|---|---:|---:|---:|---:|
| `GR_L24_ab42_top` | `oblate_runaway_to_lichnerowicz_wall` | 4.66e-11 | 7.50e-12 | 1.06e-13 | 1.536 |
| `GR_L24_ab42_tot_mu` | `oblate_runaway_to_lichnerowicz_wall` | 4.70e-11 | 9.62e-12 | 8.81e-14 | 1.536 |
| `GR_D12_ab42_top` | `oblate_runaway_to_lichnerowicz_wall` | 1.55e-10 | 1.50e-11 | 8.71e-14 | 1.534 |
| `GR_D12_ab42_tot_mu` | `oblate_runaway_to_lichnerowicz_wall` | 7.97e-11 | 5.22e-12 | 1.18e-13 | 1.534 |
| `GR_L20_ab28_top` | `oblate_runaway_to_lichnerowicz_wall` | 1.92e-11 | 7.15e-12 | 8.62e-14 | 1.217 |
| `GR_L14_ab10_tot_mu` | `oblate_runaway_to_lichnerowicz_wall` | 2.53e-12 | 2.47e-11 | 5.60e-13 | 0.480 |
| `GR_L3_ab10_top` | `dS_expansion_quenched` | 2.70e-14 | 1.44e-15 | 2.85e-15 | 2.954 |
| `GR_Q8_ab10_top` | `dS_expansion_quenched` | 2.62e-14 | 1.31e-15 | 1.88e-15 | 2.186 |
| `GR_S3_ctrl_ab10` | `dS_expansion_quenched` | 1.06e-14 | 8.84e-16 | 4.12e-15 | 8.744 |
| `Star_L24_ab36_tot_mu` | `oblate_runaway_to_lichnerowicz_wall` | 2.83e-10 | 7.21e-11 | 2.73e-13 | 1.307 |
| `Star_L24_ab44_tot_mu` | `oblate_runaway_to_lichnerowicz_wall` | 4.36e-11 | 1.79e-10 | 1.02e-13 | 1.075 |
| `Star_Q8_ab10_top` | `starobinsky_expansion_quenched` | 1.93e-14 | 3.50e-15 | 1.07e-12 | 2.186 |

Max dual-solver difference across all 12 cases: `2.833e-10`. Max un-clamped relative Hamiltonian residual: `1.791e-10` across the 12 dual-checked cases and `2.532e-10` across all 146 trajectories integrated in `03_kill_tests.py`.

## 7. Domain of Validity & Explicit Caveats

1. Dvali Species-Scale Threshold: Under CLM-T3-01 / CLM-P2-03, pure gravity (N_sp = 2) has l_sp^(2) = sqrt(2)*l_Pl = 4*sqrt(pi)*l_P = 7.0898 l_P. Consequently, Class 4a quotients L(11,1), D_6*, and L(14,1) lie in the sub-species regime at their destabilization radii (static L_min/l_sp^(2) = 0.697, 0.763, 0.939 < 1); genuine super-species static destabilization begins at L(16,1) (L_min/l_sp^(2) = 1.092), and genuine dynamical oblate runaway remaining super-species all the way to the Lichnerowicz wall lam_B = 0.525 also begins at L(16,1) (a_b = 19 l_P, min L_min/l_sp^(2) = 1.047) and becomes strongly super-species across L(20,1)/D_10* (min L_min/l_sp^(2) = 1.32) and L(24,1)/D_12* (min L_min/l_sp^(2) = 1.62).
2. 2nd-Order Geodesic Crossover N_V^crit(I*) in {5, 6}: Including the strictly alpha-independent Schur-complement metric-shear term (+1/10 <h^2> in sqrt(g^nn)) shifts the total parameterization-invariant null-cone delay on S^3/I* from the incomplete straight-line estimate (x=3.5486, N_V^crit=4, retracted) to x = 4.8208 (N_V^crit = 5, Schur-complement geodesic baseline) and x = 5.0328 (N_V^crit = 6, full 2nd-order fixed-endpoint WKB geodesic including flat-space Fermat +1/40 and longitudinal redshift -1/120). Because the exact curved-S^3 mode-by-mode Fermat coefficient c_geo(p) on global dS_4/I* is verified for p=3 (Bianchi IX, +0.149528 vs Schur 0.150000) and asymptotically via WKB (1/6), the exact integer crossover is bounded honestly as N_V^crit(I*) in {5, 6}.
3. 3D-TT Non-Einstein Operator Truncation, 6 Broken CKV Modes, and Scheme Sensitivity: On static non-Einstein R_t x S^3_lam (lam != 1), the 3D-TT Lichnerowicz operator Delta_L^(3D) omits O((lam-1)^2) coupling to 4D background-stress / vector-scalar gauge constraints, so c_C = 63/320 and A_4''(1) = +554/15 = +36.9333 characterize the 3D-TT operator specifically. Retaining the 6 broken round-S^3 Killing/CKV modes via continuous_s3=True (2j=0,|2M|=2 and 2j=1,|2M|=1; u_6''(1) = +4.510) preserves exact SO(4) continuity at lam=1 and cancels identically in u_tot = u_top + u_ren(S^3) for every L(q>=3,1) and D_m*. Shallow O(1) topological wells on Q_8, L(6,1), D_3*, L(3,1) are O(1) scheme-sensitive to C^2 counterterms, whereas q >= 14 is unconditionally 2-decade scheme-robust.
4. Quasi-Static Adiabatic Stress-Tensor Approximation, Starobinsky N_sp = 3 Species Scale, & Omitted Scalaron Casimir Energy (KT4): Dynamical Bianchi IX ODEs evaluate the quasi-static finite-lam_B graviton stress tensor (including exact covariantly conserved fixed-mu trace anomaly) instantaneously along (a_bar(t), lam_B(t)), omitting dynamical pair creation proportional to (d(lam_B)/dt)^2 that becomes O(1) as lam_B -> 0.50 (where omega_{0,0} -> 0); ODEs halt honestly at lam_B = 0.525. Starobinsky f(R) bounces at a_b = 10 l_P have super-Planckian curvature R_0/M_Pl^2 ~ 3.6 > 1 (outside EFT). Because Starobinsky f(R) propagates N_sp = 3 gravitational degrees of freedom (2 graviton helicities + 1 scalaron of mass M_S = M_Pl/sqrt(2*sigma_t) = 0.354 M_Pl), its Dvali species scale is l_sp^(3) = sqrt(3)*l_Pl = 8.683 l_P, and our runs omit the 1-loop massive-scalaron Casimir energy; under l_sp^(3), L(24,1) and D_12* at a_b = 36 l_P (R_0/M_Pl^2 = 0.59 < 1, 2*sigma_t*r_0 = 0.78 < 1) still undergo strictly super-species oblate runaway (min L_min/l_sp^(3) = 1.067 and 1.066 > 1), whereas L(20,1) at a_b = 24 l_P (0.868) and L(24,1) at a_b = 44 l_P (0.878) fall below l_sp^(3).
5. Analytic vs Finite-Cutoff Scope in KT2: The stretched formula (all j), oblate (0,0) formula, M_L=0 prolate gap (all j>=1, all lam_B>0), and sub-stretched |M_L|=j+1 positivity (all j>=1, lam>=1; unique tachyon at j=1/2, |M_L|=3/2, lam=3.243808) are proved analytically for ALL j (establishing exact analytic upper bounds on lam_prolate); general interior non-stretched sectors 0 < |M_L| <= j are verified numerically up to 2j <= 40 across both oblate lam in [0.51, 0.95] and prolate lam in [1.2, 6.0].

