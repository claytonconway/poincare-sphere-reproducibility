# Track IV-A Audited Technical Report: Causal Retarded 1-Loop Nonlocal $C\ln(-\Box/\mu^2)C$ Spin-2 Dynamics and Chiral Topological Stabilization on the $S^3/I^*$ KAM Breathing Island

**Stream:** `t4a_nonlocal_weyl_kam`  
**Workspace:** `open_problems/track4_nonlocal_weyl_kam/`  
**Background:** P05/T3-B Starobinsky–Casimir $S^3/I^*$ KAM breathing island ($\tilde\sigma_0 = c_{\mathrm{anom}} = 9/31 \approx 0.290323$, $\kappa_V = -44738/93 \approx -481.054$, $x_{\mathrm{ring}} = \sqrt{44672/93} = 21.916754$, $T_{\mathrm{macro}} = 136.88\,H_S^{-1}$, $M_{\mathrm{Pl}} = H_S = 1$)  
**Reproducibility Artifacts:**
- Goal 1 derivation & 40-digit verification: [`t4a_01_kernel_derivation_and_verify.py`](file://open_problems/track4_nonlocal_weyl_kam/t4a_01_kernel_derivation_and_verify.py) $\to$ [`t4a_01_kernel_verification.json`](file://open_problems/track4_nonlocal_weyl_kam/t4a_01_kernel_verification.json) (`"verified": true`)
- Goal 2 dual-solver implementation & cross-checks: [`t4a_02_volterra_and_order_reduced_solver.py`](file://open_problems/track4_nonlocal_weyl_kam/t4a_02_volterra_and_order_reduced_solver.py) $\to$ [`t4a_02_solver_benchmarks.json`](file://open_problems/track4_nonlocal_weyl_kam/t4a_02_solver_benchmarks.json)
- Goal 3 phase maps, kill-tests & commutator audit: [`t4a_03_kam_phase_map_and_kill_tests.py`](file://open_problems/track4_nonlocal_weyl_kam/t4a_03_kam_phase_map_and_kill_tests.py) $\to$ [`t4a_03_kam_phase_map.json`](file://open_problems/track4_nonlocal_weyl_kam/t4a_03_kam_phase_map.json)
- Goal 4 publication diagnostic figure: [`t4a_04_make_figure.py`](file://open_problems/track4_nonlocal_weyl_kam/t4a_04_make_figure.py) $\to$ [`t4a_nonlocal_weyl_kam_diagnostic.png`](file://open_problems/track4_nonlocal_weyl_kam/t4a_nonlocal_weyl_kam_diagnostic.png)

---

## Executive Summary

Track III-B proved that the local running Weyl-squared coupling $\beta_{\mathrm{eff}}(\tau) = \beta_0 - c_{\mathrm{anom}}\ln(a(\tau)/a_{\mathrm{ring}})$ admits a sharp 2:1 Stelle–scalaron parametric resonance tongue on $\beta_0 \in (0.2880, 0.2903)$ ($\gamma_{\mathrm{peak}} = +1.631 \times 10^{-3}\,H_S$) surrounded by a quiescent Regime IV window for $\beta_0 \gtrsim 0.30$. However, a local $\ln(a)$ coupling omits the causal retarded nonlocal partner $C_{\mu\nu\rho\sigma}\ln(-\Box/\mu^2)C^{\mu\nu\rho\sigma}$ mandated by the 1-loop Schwinger–Keldysh (in-in) effective action of conformally coupled Maxwell photons on $\mathbb{R}_\eta \times S^3/I^*$.

Deriving and solving the exact causal retarded spin-2 kernel on $S^3/I^*$—including the exact Maxwell stress-tensor helicity selection rule—establishes **four principal theorems**:

1. **Theorem 1 (Maxwell $s_1 s_2 = -1$ Chiral Gap & Dissipation Positivity):** Because the trace-free Maxwell stress tensor $T_{ij}^{\mathrm{TF}} = -(E_i E_j + B_i B_j)^{\mathrm{TF}} \propto (s_1 s_2 - 1)$ strictly forbids same-chirality 2-photon pair creation, the exact 2-photon threshold on round $S^3$ is $M_{S^3}(n) = n + 4$ ($M_{S^3}(2) = 6$, matching 4D CFT radial quantization $\Delta_{\min}(T_{\mu\nu}) = n+4$). On the Poincaré homology sphere $S^3/I^*$, the binary icosahedral group $I^*$ projects out all $+$-chirality photons below $\lambda_+ = +12$ (leaving $(-2, 3), (+12, 11), (-14, 15), \dots$), lifting the homogeneous $n=2$ shear threshold by **$4.33\times$** to **$M_{I^*}(2) = 26$** from $(+12, -14)$ (and $M_{I^*}(10)=14$, $M_{I^*}(14)=26$, $M_{I^*}(18)=22$, $M_{I^*}(22)=26$, with $n=12$ absent and $I^*$ triple overlaps verified nonzero). Because $M_{I^*}(n) \ge M_{S^3}(n) = n+4 > n+2$, every exact threshold lies strictly above the magnetic-Weyl band $(n, n+2)$ where the conformal Bach symbol $B_n(\Omega) = (\Omega^2 - n^2)(\Omega^2 - (n+2)^2) < 0$ ($\bar{B}^2 > \bar{E}^2$), eliminating the unphysical negative-radiation anti-damping ($+9.574 \times 10^{-4}\,H_S$) of naive $M=0$ or $M=\sqrt{6}$ kernels.
2. **Theorem 2 (Chiral Topological Horowitz Pinch-Off & Regime IV Stabilization on $S^3/I^*$):** Within the single-threshold continuum-log kernel $\ln((M^2 - (\Omega+i0^+)^2)/\mu_{\mathrm{cyl}}^2)$, raising $M$ from $M_{S^3} = 6$ to $M_{I^*} = 26$ lowers the static Horowitz upper-half-plane (UHP) Stelle-ghost pinch-off coupling by **$16.61\times$**: from $\beta_{\mathrm{pinch}}(S^3) = 6.371770$ down to **$\beta_{\mathrm{pinch}}(S^3/I^*) = 0.383639$** ($\Omega_{\mathrm{pinch}} = 21.1467$, $\omega_{\mathrm{pinch}} = 0.9649\,H_S$). On a dynamical KAM orbit, the instantaneous adiabatic pinch-off coupling $\beta_{\mathrm{pinch,inst}}(x, A_R, K_R)$ incorporates **both** the running logarithm $+c_{\mathrm{anom}}\ln(x/x_{\mathrm{ring}})$ and the dominant $x^2 A_R(\tau)$ Einstein–Hilbert scaling, yielding an adiabatic envelope maximum of **$\max_\tau \beta_{\mathrm{pinch,inst}} = 0.4206$** on $x_{\mathrm{ring}}$ (`0.4417` at frozen $A_R = K_R = 1$) and **`0.7326`** on eccentric $x_b = 15.0$ (`0.7544` at $A_R = K_R = 1$).
   **Consequence:** Across the entire Regime IV window **$\beta_0 \in [0.44, 6.37]$** where round $S^3$ suffers catastrophic Horowitz UHP runaway ($\gamma = +0.2406\,H_S$ at $\beta_0 = 0.50$, $+0.1129\,H_S$ at $\beta_0 = 1.0$, $+0.0417\,H_S$ at $\beta_0 = 2.0$, $+0.00859\,H_S$ at $\beta_0 = 5.0$), **$S^3/I^*$ IS COMPLETELY STABILIZED** in the full unreduced 4th-order causal Volterra solver over $\tau = 773\text{--}996\,H_S^{-1}$ ($5.6\text{--}7.3\,T_{\mathrm{macro}}$) with `max_ratio = 1.00–1.06`, $|\gamma| \le 5.3 \times 10^{-5}\,H_S$ at $\beta_0 \ge 0.50$ for ghost initial data (a **$4{,}544\times$ suppression** vs. round $S^3$ at $\beta_0 = 0.50$) and $|\gamma_E| \le 1.47 \times 10^{-5}\,H_S$ for smooth physical graviton initial data. Likewise, eccentric $x_b = 15.0$ stabilizes sharply above its `0.7326` envelope ($\beta_0 = 0.50$ blows up at $+3.57 \times 10^{-2}\,H_S$; $\beta_0 = 0.70$ drops to $+3.70 \times 10^{-3}\,H_S$; $\beta_0 = 0.80, 1.00$ drop to $+2.45 \times 10^{-4}, +1.76 \times 10^{-4}\,H_S$, bounded). Below $\max_\tau\beta_{\mathrm{pinch,inst}} = 0.4206$—including T3-B's bare 2:1 tongue at $\beta_0 = 0.2890$ ($\gamma_{\mathrm{dyn}} = +0.2111\,H_S$, $130\times$ T3-B's local tongue) and the adiabatic pinch-off band $\beta_0 \in (0.3836, 0.4206)$—unreduced Horowitz ghost runaway dominates.
3. **Theorem 3 (Manley–Rowe Floquet Sideband Cancellation, Non-Self-Adjoint Order-Reduction Commutator Defect, & Scalaron Backreaction Budget):** Because $4 c_{\mathrm{anom}} \omega_{\mathrm{scal}}^2 = 4.000$ on P05's $\tilde\sigma_0 = 9/31$ background, the scalaron frequency $\Omega_{\mathrm{scal}} \approx 41.05$ exceeds $M_{I^*} = 26$, launching graviton Floquet sidebands $\Omega_\pm = \Omega_{\mathrm{scal}} \pm \Omega_0 \approx \{38.22, 43.88\}$ into the 2-photon continuum ($P_{\mathrm{rad}}/(2E_{\mathrm{avg}}) = 3.292 \times 10^{-3}\,H_S$, $\tau_{\mathrm{rad}} = 303.8\,H_S^{-1} = 2.22\,T_{\mathrm{macro}}$, cross-checked to `0.48%` between Direct Volterra and Parseval FFT). Upper absorption ($\text{graviton}+\text{scalaron}\to 2\gamma$) and lower stimulated down-conversion ($\text{scalaron}\to \text{graviton}+2\gamma$) cancel to **$99.76\%$** on the graviton amplitude itself: centered finite-difference linear response ($\epsilon = 0.01$ and $0.02$, matching to 4 digits) yields $\Delta\lambda_2^{(1)} = +7.77 \times 10^{-6}\,H_S$ on $S^3/I^*$ (consistent with $|\gamma_E| \le 1.47 \times 10^{-5}\,H_S$ in the exact unreduced 4th-order solver), meaning virtually **$\approx 100\%$** of the emitted 2-photon power is supplied by the scalaron condensate (which carries up to $\kappa_{\mathrm{scal}} = 2 x_{\mathrm{ring}}^2 = 960.69 = 1.997\,|\kappa_V|$ vs. shear-only conversion $\Delta\kappa_{\mathrm{shear}} = \frac{8}{3}x_{\mathrm{ring}}^2 b_0^2 = 1280.9\,b_0^2$, $b_{\mathrm{crit}}^{\mathrm{shear}} = 0.6128$). Crucially, we prove and verify that **self-consistent finite-$\kappa_{\mathrm{mem}}$ Parker–Simon / Flanagan–Wald order reduction violates self-adjointness of the purely dispersive kernel $\mathrm{Re}\,\mathcal{K}_M$** because $[\mathrm{Re}\,\mathcal{K}_M, \mathcal{B}_0(\eta)] \ne 0$, generating spurious non-conservative drift of $O(\beta_0 \kappa_{\mathrm{mem}})$ (`+3.300e-3` at $\beta_0 = +1$ vs. `-3.238e-3` at $\beta_0 = -1$ at $\kappa=0.25$, cross-checked between Aux-ODE and Direct Volterra to `2.0e-3–3.0e-3`) and $O(\kappa_{\mathrm{mem}}^2)$ (`c_2 = +1.27e-3 H_S` on $x_{\mathrm{ring}}$).
4. **Theorem 4 (Inhomogeneous $S^3/I^*$ TT Gravitons & 0th-Order Mathieu Reheating at $n=18, 22$):** Using the analytic constant-curvature $S^3$ Myers-current identity $\delta^{(1)}{}^{(3)}G^i{}_j = \frac{1}{2a^2}(-\bar\nabla^2 + 2)H^i{}_j = +\frac{k_n^2}{2a^2}H^i{}_j$ (Lichnerowicz/curl matrix eigenvalues verified to `<= 2.84e-13`, `myers_deltaG1_matrix_rel_err <= 1.30e-15` for $n=2, 3, 10, 14$), $A_{\mathrm{Rieg}}, K_{\mathrm{Rieg}}$ are strictly $n$-independent for all TT modes $n \ge 2$. Modes $n=2, 10, 14$ stay bounded at 0th order ($|\lambda_n^{(0)}| \le 1.53 \times 10^{-5}\,H_S$); under centered $O(c_{\mathrm{anom}})$ linear response, $n=10$ ($M_{I^*}=14$) shows tiny positive anti-damping ($+9.81 \times 10^{-5}\,H_S$), whereas **$n=14$ ($M_{I^*}=26$) receives strong 1-loop damping ($\Delta\lambda_{14}^{(1)} = -1.241 \times 10^{-2}\,H_S$)** because $M_{I^*}(14) = 26$ sits right at the lower sideband $|\Omega_-| = 26.08$, choking off down-conversion while leaving upper absorption ($\Omega_+ = 56.02$) open. Modes $n=18$ ($\epsilon_{\mathrm{EFT}} = 0.870$) and $n=22$ ($\epsilon_{\mathrm{EFT}} = 1.277$) straddle the algebraically coincident 2:1 Mathieu / EFT-cutoff threshold $\omega_n = \omega_{\mathrm{scal}}/2 \equiv (\epsilon_{\mathrm{EFT}} = 1)$ at $n = 19.3625$, undergoing explosive 0th-order parametric reheating ($\lambda_{18}^{(0)} = +8.791 \times 10^{-2}\,H_S$, $\lambda_{22}^{(0)} = +1.309 \times 10^{-2}\,H_S$) only partially offset by $O(c_{\mathrm{anom}})$ damping ($-2.049 \times 10^{-2}, -2.21 \times 10^{-3}\,H_S$).

![Track IV-A 4-Panel Diagnostic Figure](file://open_problems/track4_nonlocal_weyl_kam/t4a_nonlocal_weyl_kam_diagnostic.png)

---

## 1. Exact Causal Retarded 1-Loop Nonlocal Spin-2 Action on $\mathbb{R}_\eta \times S^3/I^*$

### 1.1 Conformal Frame Reduction & Exact Separation of Local Running vs. Causal Memory

On a closed FLRW spacetime $ds^2 = a(\eta)^2(-d\eta^2 + \gamma_{ij}dx^i dx^j)$ with spatial 3-manifold $\Sigma_3 = S^3/I^*$ of unit curvature ($K=+1$), transverse-traceless (TT) tensor perturbations $h_{ij} = a(\eta)^2 H_{ij}(\eta, x)$ decompose into orthonormal TT tensor eigenmodes $Y_{ij}^{(n,s,\alpha)}(x)$ on unit $S^3/I^*$ satisfying
$$-\bar\nabla^2 Y_{ij} = (n^2 + 2n - 2)\,Y_{ij}, \qquad \mathrm{curl}\,Y_{ij} = s(n+1)\,Y_{ij}, \qquad k_n^2 \equiv n(n+2), \quad p_n \equiv n+1, \quad s = \pm 1\,.$$
Because the Weyl tensor $C^\mu{}_{\nu\rho\sigma}$ is conformally invariant, the 1-loop Maxwell vacuum polarization action on $g_{\mu\nu} = a(\eta)^2 \bar g_{\mu\nu}$ splits **identically** into a local conformal-anomaly running term plus a metric-static nonlocal term on the Einstein cylinder $\bar g_{\mu\nu} = \mathrm{diag}(-1, \gamma_{ij})$:
$$\Gamma_{\mathrm{1\text{-}loop}}^{(C^2)} = -\int d^4x\,\sqrt{-\bar g}\,\bar C_{\mu\nu\rho\sigma}\left[\beta_{\mathrm{eff}}(\eta) + \frac{c_{\mathrm{anom}}}{2}\ln\!\left(\frac{-\bar\Box_C}{\mu_{\mathrm{cyl}}^2}\right)\right]\bar C^{\mu\nu\rho\sigma}\,,$$
where $\mu_{\mathrm{cyl}} \equiv a_{\mathrm{ring}}\mu = x_{\mathrm{ring}}\mu$ ($\mu = H_S = 1$), $\beta_{\mathrm{eff}}(\eta) \equiv \beta_0 - c_{\mathrm{anom}}\ln(x(\eta)/x_{\mathrm{ring}})$, and $c_{\mathrm{anom}} = 9/31$ in T3-B's normalized units ($M_{\mathrm{Pl}} = H_S = 1$). Under an RG scale shift $\mu \to \lambda\mu$, $\beta_0 \to \beta_0 + c_{\mathrm{anom}}\ln\lambda$ cancels $-c_{\mathrm{anom}}\ln\lambda$ from $\frac{1}{2}\ln(-\bar\Box_C/\mu_{\mathrm{cyl}}^2)$ identically.

For each TT mode $b(\eta)$ normalized as in T3-B ($H_{ij} = b(\eta) Y_{ij}$, with homogeneous Bianchi IX shear $n=2, s=-1, k_2^2 = 8, p_2 = 3$), the electric and magnetic Weyl components on the unit cylinder are
$$Q(\eta) \equiv b''(\eta) - k_n^2 b(\eta), \qquad P(\eta) \equiv 2 p_n b'(\eta)\,,$$
and the quadratic Weyl invariant is $\bar C^2 \propto Q^2 - P^2$, whose variation yields the exact constant-coefficient conformal Bach operator:
$$\mathcal{B}_n[b] \equiv (\partial_\eta^2 - k_n^2)Q + 2 p_n \partial_\eta P = b^{(4)} + (4 p_n^2 - 2 k_n^2) b'' + k_n^4 b = (\partial_\eta^2 + n^2)(\partial_\eta^2 + (n+2)^2)\,b\,.$$
In frequency space ($\partial_\eta \to -i\Omega$), the conformal Bach symbol factorizes cleanly:
$$B_n(\Omega) = (\Omega^2 - n^2)(\Omega^2 - (n+2)^2)\,.$$

### 1.2 Exact Schwinger–Keldysh Causal Kernel & 40-Digit Verification

Applying the Schwinger–Keldysh (in-in) retarded prescription $\Omega \to \Omega + i0^+$ to the single-threshold 1-loop logarithm at conformal threshold $M > 0$ yields $\mathcal{K}_M(\Omega) \equiv \ln\!\frac{M^2 - (\Omega + i0^+)^2}{\mu_{\mathrm{cyl}}^2}$:
$$\mathrm{Re}\,\mathcal{K}_M(\Omega) = \ln\!\left(\frac{|M^2 - \Omega^2|}{\mu_{\mathrm{cyl}}^2}\right), \qquad \mathrm{Im}\,\mathcal{K}_M(\Omega) = -\pi\,\mathrm{sgn}(\Omega)\,\Theta(|\Omega| - M)\,.$$
Its exact time-domain retarded action on any smooth history $f(\eta)$ with constant pre-history ($f'(\eta) = 0$ for $\eta < 0$) is
$$\mathcal{K}_M[f](\eta) = \ln\!\left(\frac{M^2}{\mu_{\mathrm{cyl}}^2}\right) f(\eta) - 2 \int_0^\eta \mathrm{Ci}\!\left(M(\eta - \eta')\right) f'(\eta')\,d\eta'\,,$$
where the cosine integral admits the exact damped-oscillator Laplace representation for $s > 0$:
$$-\mathrm{Ci}(M s) = \mathrm{Re}\int_0^\infty \frac{e^{-(u - iM)s}}{u - iM}\,du\,.$$
In the gapless $M \to 0^+$ limit, this reduces smoothly to $\mathcal{K}_0[f](\eta) = -2\int_0^\eta [\ln(\mu_{\mathrm{cyl}}(\eta-\eta')) + \gamma_E]f'(\eta')\,d\eta'$. Direct 40-digit `mpmath` contour-rotated verification in [`t4a_01_kernel_derivation_and_verify.py`](file://open_problems/track4_nonlocal_weyl_kam/t4a_01_kernel_derivation_and_verify.py) confirms every identity (`"verified": true`):

| Kernel Identity Test (40-digit `mpmath`) | Max Absolute Error |
|---|---|
| UHP causal Fourier transform $\mathcal{K}_0(\Omega)$ ($M=0$, $\mathrm{Im}\,\Omega > 0$) | $4.59 \times 10^{-41}$ |
| UHP causal Fourier transform $\mathcal{K}_6(\Omega)$ (round $S^3$, $M=6$) | $7.35 \times 10^{-41}$ |
| UHP causal Fourier transform $\mathcal{K}_{26}(\Omega)$ ($S^3/I^*$, $M=26$) | $9.69 \times 10^{-42}$ |
| Laplace representation $-\mathrm{Ci}(26s) = \mathrm{Re}\int_0^\infty \frac{e^{-(u-26i)s}}{u-26i}\,du$ | $1.79 \times 10^{-43}$ |
| Gauss–Legendre auxiliary-ODE quadrature ($M=26$, $N_u=80$ / $N_u=120$) | $1.25 \times 10^{-5}$ / $1.17 \times 10^{-7}$ |

---

## 2. Exact Maxwell $s_1 s_2 = -1$ Chiral Thresholds, Myers $n$-Independence, and Dissipation Positivity

### 2.1 Maxwell Opposite-Chirality Selection Rule ($s_1 s_2 = -1$) & $S^3/I^*$ Spectrum

Transverse vector (photon) eigenmodes $V_i^{(\lambda)}$ on unit $S^3$ satisfy $\mathrm{curl}\,V_i = \lambda V_i$ with $\lambda = s(m+1)$, $m \ge 1$, $s = \pm 1$, carrying $\mathrm{SU}(2)_L \times \mathrm{SU}(2)_R$ quantum numbers $(j_L, j_R) = (\frac{m+1}{2}, \frac{m-1}{2})$ for $s = +1$ and $(\frac{m-1}{2}, \frac{m+1}{2})$ for $s = -1$. Under Maxwell's equations on the static cylinder, a single-frequency mode $A_i \propto e^{-i\omega\eta} V_i^{(\lambda)}$ with $\omega = |\lambda|$ has $E_i = i\omega A_i$ and $B_i = \mathrm{curl}\,A_i = s\omega A_i = -i s E_i$. Hence the bilinear trace-free Maxwell stress tensor for any photon pair $(1, 2)$ is
$$T_{ij}^{\mathrm{TF}}(1, 2) = -\left(E_i^{(1)} E_j^{(2)} + B_i^{(1)} B_j^{(2)}\right)^{\mathrm{TF}} = -\left(1 - s_1 s_2\right)\left(E_i^{(1)} E_j^{(2)}\right)^{\mathrm{TF}}\,.$$
Whenever $s_1 = s_2$ ($s_1 s_2 = +1$), $T_{ij}^{\mathrm{TF}}(1, 2) \equiv 0$ identically; every 2-photon creation channel MUST pair opposite chiralities ($s_1 s_2 = -1$).

- **Round $S^3$:** The lowest opposite-chirality pair coupling to a TT graviton of index $n$ has $(|\lambda_1|, |\lambda_2|) = (2, n+2)$, giving **$M_{S^3}(n) = n + 4$** (matching 4D CFT radial quantization $\Delta_{\min}(T_{\mu\nu}) = n+4$), so $M_{S^3}(2) = 6$ from $(+2, -4)$.
- **Poincaré Homology Sphere $S^3/I^*$:** Requiring $I^*$-invariant $j_L$ ($d_{I^*}(j_L) \ge 1$, $j_L \in \{0, 6, 10, 12, 15, 16, \dots\}$) gives first eight photon curl levels $(-2, 3), (+12, 11), (-14, 15), (+20, 19), (-22, 23), (+24, 23), (-26, 27), (+30, 29)$, projecting out **all $+$-chirality photons below $\lambda_+ = +12$** as well as TT gravitons $n=3..9$ and $n=12$ (`graviton_n12_absent_verified = true`). Explicit 120-element Wigner-$D$ projections verify nonzero $I^*$ triple overlaps (`I6*I6 -> I0: 0.2821`, `I6: 0.2089`, `I10: 0.1989`, `I12: 0.1712`), establishing the exact thresholds:

| TT Graviton Mode $n$ (chirality) | $(j_L, j_R)$ | Mult on $S^3/I^*$ | $\omega_{\mathrm{phys}}(x_{\mathrm{ring}})/H_S$ | $\epsilon_{\mathrm{EFT}}(x_{\mathrm{ring}})$ | Round $S^3$ $M_{S^3}$ (pair) | Exact $S^3/I^*$ $M_{I^*}$ (pair) |
|---|---|---|---|---|---|---|
| **$n = 2\,(-)$** (homog. shear) | $(0, 2)$ | 5 | $0.1291$ | $0.0193$ | $6$ $(+2, -4)$ | **$26$** $(+12, -14)$ |
| **$n = 10\,(+)$** (lowest inhom.) | $(6, 4)$ | 9 | $0.4998$ | $0.2901$ | $14$ $(-2, +12)$ | **$14$** $(-2, +12)$ |
| **$n = 14\,(-)$** | $(6, 8)$ | 17 | $0.6829$ | $0.5415$ | $18$ $(+2, -16)$ | **$26$** $(+12, -14)$ |
| **$n = 18\,(+)$** | $(10, 8)$ | 17 | $0.8657$ | $0.8703$ | $22$ $(-2, +20)$ | **$22$** $(-2, +20)$ |
| **$n = 22\,(-)$ / $(+)$** | $(10, 12)$ / $(12, 10)$ | $25$ / $21$ | $1.0484$ | $1.2765$ | $26$ $(\pm 2, \mp 24)$ | **$26$** $(+12,-14)$ / $(-2,+24)$ |

### 2.2 Analytic Myers-Current $n$-Independence of $A_{\mathrm{Rieg}}, K_{\mathrm{Rieg}}$ & Matrix Spectrum Check

On unit $S^3$ ($R^{(3)}_{ij} = 2\gamma_{ij}$, $R^{(3)} = 6$), the linearized spatial Einstein tensor of any TT mode $H^i{}_j$ satisfies the exact constant-curvature identity
$$\delta^{(1)}{}^{(3)}G^i{}_j = \frac{1}{2a^2}\left(-\bar\nabla^2 + 2\right)H^i{}_j = +\frac{k_n^2}{2a^2}\,H^i{}_j\,, \qquad k_n^2 = n(n+2)\,.$$
Because the Myers-current variation of the covariant Riegert term on conformally flat FLRW enters the quadratic TT action exclusively through $\delta^{(1)}{}^{(3)}G^i{}_j$ and $\partial_\tau H^i{}_j$, the harmonic index $n$ factors cleanly into $k_n^2$, leaving $A_{\mathrm{Rieg}} = 1 + 2\tilde\sigma_0 r - 2h^2 + \frac{2}{3x^2}$ and $K_{\mathrm{Rieg}} = 1 + 2\tilde\sigma_0 r + 2h^2 + \frac{2}{3x^2} - 4\dot h$ strictly $n$-independent for all $n \ge 2$. Given this analytic Myers reduction, `t4a_01` directly builds explicit $\mathrm{SU}(2)$ TT tensor-harmonic matrices in the left-invariant frame and verifies the Lichnerowicz, curl, and derived $\frac{1}{2}(-\bar\nabla^2 + 2)$ eigenvalues from the numerical matrix spectrum (`evals_L[idx_n]`):

| Mode $n$ | TT Mult in $j_{\mathrm{orb}}$ Rep | Max $\|-\bar\nabla^2 - (n^2+2n-2)\|$ | Max $\||\mathrm{curl}| - (n+1)\|$ | Matrix $\frac{1}{2}(\lambda_L + 2)$ Rel Err vs. $k_n^2/2$ |
|---|---|---|---|---|
| $n = 2$ | 5 | $2.66 \times 10^{-15}$ | $1.33 \times 10^{-15}$ | $3.33 \times 10^{-16}$ |
| $n = 3$ | 6 | $1.95 \times 10^{-14}$ | $4.88 \times 10^{-15}$ | $1.30 \times 10^{-15}$ |
| $n = 10$ | 9 | $1.28 \times 10^{-13}$ | $2.31 \times 10^{-14}$ | $1.07 \times 10^{-15}$ |
| $n = 14$ | 17 | $2.84 \times 10^{-13}$ | $3.02 \times 10^{-14}$ | $1.27 \times 10^{-15}$ |

### 2.3 Theorem 1: Dissipation Positivity Theorem

With the conformal energy norm $E_{\mathrm{loc}} \equiv x^2(A_R b'^2 + k_n^2 K_R b^2) = 2 H_{\mathrm{action}}$ used across `t4a_01`–`t4a_03`, the total energy radiated into photons by a localized wavepacket $b(\eta)$ is
$$\Delta E_{\mathrm{rad}} = 4 c_{\mathrm{anom}} \int_{-\infty}^{\infty} \frac{d\Omega}{2\pi}\,\Omega\left(-\mathrm{Im}\,\mathcal{K}_M(\Omega)\right) B_n(\Omega)\,|\tilde b(\Omega)|^2 = 2 c_{\mathrm{anom}} \int_{|\Omega| \ge M} |\Omega|\,B_n(\Omega)\,|\tilde b(\Omega)|^2\,d\Omega\,.$$
Because $B_n(\Omega) = (\Omega^2 - n^2)(\Omega^2 - (n+2)^2)$ is **strictly negative** throughout the magnetic-Weyl band $\Omega \in (n, n+2)$ (where $\bar B^2 > \bar E^2$, containing the on-shell Einstein–Hilbert graviton $\Omega_0 = \sqrt{n(n+2)}$ where $B_n(\Omega_0) = -4 k_n^2 < 0$), any kernel with $M < n+2$ violates unitarity:
- Naive $M = 0$ and Lichnerowicz $M = \sqrt{6}$ kernels yield $\Delta E_{\mathrm{rad}} = -1147.15$ and $-1138.50$ ($< 0$!) on an on-shell $n=2$ wavepacket, producing spurious static anti-damping $\gamma = +9.574 \times 10^{-4}\,H_S$.
- Exact opposite-chirality thresholds $M_{S^3}(n) = n+4$ and $M_{I^*}(n) \ge n+4$ lie strictly above $n+2$, guaranteeing $B_n(\Omega) > 0$ identically on the entire 2-photon cut $|\Omega| \ge M$ ($\Delta E_{\mathrm{rad}} = 0.0$ on-shell for $M_{I^*}=26$, $1.41 \times 10^{-67}$ for $M_{S^3}=6$, and $+1.032 \times 10^9 > 0$ at the scalaron sideband $\Omega = 43$).

---

## 3. Dual Retarded Volterra Solver Architecture & Goal 2 Cross-Check Benchmarks

In [`t4a_02_volterra_and_order_reduced_solver.py`](file://open_problems/track4_nonlocal_weyl_kam/t4a_02_volterra_and_order_reduced_solver.py), we implemented **two independent discretizations** for both the full unreduced 4th-order Ostrogradsky + causal Volterra system and the 2nd-order Parker–Simon / Flanagan–Wald order-reduced system: (A) **Spectral Auxiliary-ODE** (`Radau`/`BDF` with exact analytical Jacobian on $N_u$ Gauss–Legendre nodes in $\ln u$, guarding `M = 0` with nonzero $\Pi_2(0)$ via an explicit `ValueError`) and (B) **Direct Time-Domain Product-Integration Volterra** using closed-form panel integrals $I_m = \left[s\,\mathrm{Ci}(Ms) - \frac{\sin(Ms)}{M}\right]_{mh}^{(m+1)h}$ with zero exponential approximation. All benchmarks in [`t4a_02_solver_benchmarks.json`](file://open_problems/track4_nonlocal_weyl_kam/t4a_02_solver_benchmarks.json):

| Benchmark Test | Quantity Measured | Result / Agreement |
|---|---|---|
| KAM background $x_{\mathrm{ring}}$ ($\eta=40, \tau=884.6$) | Max Hamiltonian constraint error $\|\mathcal{E}_{00}\|_{\mathrm{rel}}$ | $1.17 \times 10^{-13}$ |
| Zero-memory reduction vs. T3-B ($\kappa_{\mathrm{mem}}=0, \tau=331.9$) | $\max_{\tau} |q_1^{\mathrm{T4A}} - q_1^{\mathrm{T3B}}|_{\mathrm{rel}}$ at $\beta_0 = 1.0$ / $0.289$ | $5.08 \times 10^{-10}$ / $8.04 \times 10^{-10}$ |
| Full Volterra: $M_{I^*}=26, \beta_0=0.289$ ($\eta=3.0$) | Aux vs. Volterra rel diff (static / dyn) | $8.85 \times 10^{-3}$ / $8.81 \times 10^{-3}$ |
| Full Volterra: $M_{I^*}=26, \beta_0=0.289$ (static $\gamma$) | Aux `+0.218122` / Volt `+0.217982` vs. Goal 1 `+0.218153` | Rel err **$1.41 \times 10^{-4}$** (dyn: `+0.20945`) |
| Full Volterra: $M_{I^*}=26, \beta_0=1.0$ ($\eta=5.0$, sub-thresh) | Aux vs. Volterra rel diff (static / dyn) | $4.43 \times 10^{-4}$ / $4.28 \times 10^{-4}$ |
| Full Volterra: $M_{I^*}=26, \beta_0=1.0$ (short $\eta=5$ vs. long $\tau=986$) | Short beat `+2.12e-3` (Goal 1: `0.0`); long static $\tau=986$ control | **`+5.09e-6` (bounded, $\gamma_{\mathrm{static}} \equiv 0$)** |
| Full Volterra: $M_{S^3}=6, \beta_0=1.0$ ($\eta=5.0$) | Aux `+0.112650` / Volt `+0.112622` vs. Goal 1 `+0.112651` | Rel err **$8.06 \times 10^{-6}$** (dyn: `+0.11287`) |
| Full Volterra: $M_{S^3}=6, \beta_0=0.289$ ($\eta=2.5$) | Aux `+0.363014` / Volt `+0.363157` vs. Goal 1 `+0.363149` | Rel err **$3.71 \times 10^{-4}$** (dyn: `+0.36280`) |
| Exact RG-scale invariance ($\mu \to e\mu$, $M_{I^*}=26$) | Full Aux-ODE / Full Direct Volterra rel diff | $8.28 \times 10^{-15}$ / $1.41 \times 10^{-12}$ |
| Order-reduced static $x_{\mathrm{ring}}$ ($\kappa_{\mathrm{mem}}=1$) | `M26` / `M6` / `M0_naive` ($\gamma_{\mathrm{Aux}}$ vs. $\gamma_{\mathrm{Volt}}$) | `-3.0e-9` / `+2.3e-9` / **`+9.574e-4` (spurious!)** |

---

## 4. Theorem 2: Chiral Topological Horowitz Pinch-Off & Regime IV Stabilization on $S^3/I^*$

### 4.1 30-Digit Static Dispersion ($M_{I^*} = 26$ vs. $M_{S^3} = 6$) & Full Adiabatic Pinch-Off Envelope

On a frozen background slice $(x, A_R, K_R)$, the real-axis Stelle dispersion relation $D_M(\Omega; x, A_R, K_R) = 0$ solves for $\beta_0$ as
$$\beta_0(\Omega; x, A_R, K_R) = \frac{x^2(A_R\Omega^2 - 8 K_R)}{4 B_2(\Omega)} - \frac{c_{\mathrm{anom}}}{2}\ln\!\left(\frac{M^2 - \Omega^2}{x_{\mathrm{ring}}^2}\right) + c_{\mathrm{anom}}\ln\!\left(\frac{x}{x_{\mathrm{ring}}}\right)\,.$$
Minimizing over $\Omega \in (4, M)$ gives the instantaneous pinch-off coupling $\beta_{\mathrm{pinch,inst}}(x, A_R, K_R)$ above which the UHP Horowitz ghost pole collides with its LHP partner on the real axis and bifurcates into two purely oscillatory sub-threshold real poles ($\gamma_{\mathrm{static}} \equiv 0$):

- **Static $x_{\mathrm{ring}}$ ($A_R = K_R = 1$):**
  - Round $S^3$ ($M_{S^3} = 6$): $\beta_{\mathrm{pinch}}(S^3) = \mathbf{6.371770}$ ($\Omega_{\mathrm{pinch}} = 5.9519$, $\omega = 0.2716\,H_S$).
  - $S^3/I^*$ ($M_{I^*} = 26$): $\beta_{\mathrm{pinch}}(S^3/I^*) = \mathbf{0.383639}$ ($\Omega_{\mathrm{pinch}} = 21.1467$, $\omega = 0.9649\,H_S$) — **$16.61\times$ lower than round $S^3$!**
- **Exact Dynamical Adiabatic Envelope ($\max_\tau \beta_{\mathrm{pinch,inst}}(x(\tau), A_R(\tau), K_R(\tau))$):** By the envelope theorem, the Einstein–Hilbert $x^2 A_R$ term contributes $1.90\times$ more to $\partial\beta_{\mathrm{pinch}}/\partial\ln x$ than the running logarithm $c_{\mathrm{anom}}\ln(x/x_{\mathrm{ring}})$ alone, while out-of-phase Riegert oscillations in $A_R(\tau)$ partially offset the pure $x_{\max}^2$ peak:
  - **KAM ring $x_{\mathrm{ring}}$** ($x \in [20.797, 23.423]$): frozen-$A_R=K_R=1$ max is `0.4417`, and the **full $(x, A_R, K_R)$ adiabatic envelope max is `0.4206`**!
  - **Eccentric orbit $x_b = 15.0$** ($x \in [15.0, 31.440]$): frozen-$A_R=K_R=1$ max is `0.7544`, and the **full $(x, A_R, K_R)$ adiabatic envelope max is `0.7326`**!
- **Negative $\beta_0 < 0$:** Both $M=6$ and $M=26$ exhibit IR tachyons plus secondary UV Landau-pole tachyons on the positive imaginary axis (at $\beta_0 = -1.0$: IR $+0.495\,H_S$ and UV $+31.28\,H_S$ for $M=26$; at $\beta_0 = -0.2903$: merged complex tachyon $\omega = 0.255 + 1.582\,i\,H_S$ for $M=26$, vs. IR $+0.877\,H_S$ and UV $+2.293\,H_S$ for $M=6$).

### 4.2 T3-B 2:1 Tongue Homotopy ($\beta_0 = 0.2890$) & Complete Pinch-Off Scan ($\beta_0 \in [0.35, 0.50]$)

Because T3-B's bare 2:1 tongue ($\beta_0 = 0.2890$) sits below $\beta_{\mathrm{pinch}}(I^*) = 0.383639$, causal memory $\kappa_{\mathrm{mem}} \in [0, 1]$ at $M_{I^*}=26$ drowns T3-B's $+1.631 \times 10^{-3}\,H_S$ tongue once $\kappa_{\mathrm{mem}} \gtrsim 0.02$:

| $\kappa_{\mathrm{mem}}$ ($M_{I^*}=26$) | $\gamma(\beta_0=0.2890)/H_S$ | $\gamma(\beta_0=0.2880)/H_S$ | Tongue Excess $\Delta\gamma / H_S$ | Regime |
|---|---|---|---|---|
| `0.0000` (T3-B local) | `+1.631e-3` | `-4.933e-4` | `+2.124e-3` | Sharp T3-B 2:1 tongue |
| `0.0020` | `+1.486e-3` | `-1.623e-3` | `+3.109e-3` | Bare tongue persists |
| `0.0100` | `-1.349e-4` | `-1.177e-3` | `+1.042e-3` | Real log detunes 2:1 tongue |
| `0.0500` | `+8.862e-4` | `+1.568e-3` | `-6.819e-4` | Horowitz runaway overtakes tongue |
| `0.2500` | `+1.152e-2` | `+1.416e-2` | `-2.635e-3` | Broad UHP ghost runaway |
| **`1.0000` (full 1-loop)** | **`+2.110e-1`** | **`+2.126e-1`** | `-1.565e-3` | **$130\times$ T3-B; flat UHP plateau** |

Sweeping finely across both $\beta_{\mathrm{pinch}}(I^*) = 0.3836$ and the full adiabatic envelope $\max_\tau\beta_{\mathrm{pinch,inst}} = \mathbf{0.4206}$ (**every row in `pinchoff_and_shifted_2to1_scan_M26` reported without omission**):

| $\beta_0$ ($M_{I^*}=26, \kappa_{\mathrm{mem}}=1$) | $\tau_{\mathrm{reached}}$ ($H_S^{-1}$) | $\gamma_{\mathrm{static}}(x_{\mathrm{ring}})/H_S$ | $\gamma_{\mathrm{dyn}}(\mathrm{KAM})/H_S$ | $\max|\Pi_2|/|\Pi_2|_0$ | Outcome |
|---|---|---|---|---|---|
| `0.350` | `132.6` | `+1.215e-1` | `+1.178e-1` | `7.69e6` | Below static pinch-off |
| `0.370` | `190.5` | `+7.576e-2` | `+1.049e-1` | `4.76e6` | Blowup |
| `0.380` | `274.0` | `+3.875e-2` | `+5.220e-2` | `2.56e6` | Blowup |
| `0.384` | `295.3` | `+1.91e-4` ($\equiv 0$) | `+4.020e-2` | `4.40e5` | Adiabatic band blowup |
| `0.386` | `318.6` | `-1.12e-3` ($\equiv 0$) | `+4.043e-2` | `4.58e5` | Adiabatic band blowup |
| `0.388` | `406.9` | `-4.46e-5` ($\equiv 0$) | `+3.298e-2` | `5.77e5` | Adiabatic band + shifted 2:1 blowup |
| `0.390` | `423.9` | `+3.48e-4` ($\equiv 0$) | `+3.388e-2` | `5.96e5` | Adiabatic band blowup |
| `0.395` | `536.3` | `+2.27e-4` ($\equiv 0$) | `+3.210e-2` | `1.10e6` | Adiabatic band blowup |
| `0.405` | `773.4` | `-3.41e-5` ($\equiv 0$) | `+1.687e-3` | `41.2` | Weak edge growth ($\le 0.4206$) |
| `0.420` | `773.4` | `-9.52e-5` ($\equiv 0$) | `+3.083e-3` | `5.50` | Upper edge of envelope (`0.4206`) |
| **`0.440`** | `773.4` | `-2.01e-5` ($\equiv 0$) | **`-1.472e-3`** | **`1.00`** | **Bounded immediately above `0.4206`!** |
| **`0.460`** | `773.4` | `-5.20e-5` ($\equiv 0$) | **`-1.844e-4`** | **`1.00`** | **Completely quiescent** |
| **`0.480`** | `773.4` | `-4.17e-5` ($\equiv 0$) | **`+1.320e-4`** | **`1.06`** | **Completely quiescent** |
| **`0.500`** | `884.6` | `+1.50e-5` ($\equiv 0$) | **`+5.296e-5`** | **`1.06`** | **Completely quiescent** |

### 4.3 Full Regime IV Comparison: $S^3/I^*$ ($M_{I^*}=26$) vs. Round $S^3$ ($M_{S^3}=6$) & Eccentric $x_b = 15$

| $\beta_0$ | $S^3/I^*$ ($M=26$) Static G1 | $S^3/I^*$ ($M=26$) Dyn KAM ($\tau$) | $S^3/I^*$ Phys-IC $\gamma_E$ ($\tau=885$) | Round $S^3$ ($M=6$) Static G1 | Round $S^3$ ($M=6$) Dyn KAM ($\tau$) | Topology Advantage |
|---|---|---|---|---|---|---|
| `-0.2903` | `+1.582e+0` | `+1.710e+0` (`10.1`) | — | `+2.293e+0` | `+2.307e+0` (`8.1`) | Both tachyonic |
| `+0.0500` | `+5.708e-1` | `+5.683e-1` (`30.8`) | — | `+6.262e-1` | `+6.285e-1` (`27.8`) | Both UHP runaway |
| `+0.2890` | `+2.182e-1` | `+2.111e-1` (`77.2`) | — | `+3.631e-1` | `+3.629e-1` (`49.7`) | Both UHP runaway |
| **`+0.5000`** | **`0.0`** | **`+5.30e-5`** (`884.6`) | **`+4.54e-6`** (`E/E0=0.89`) | `+2.410e-1` | **`+2.406e-1`** (`73.7`) | **$S^3/I^*$ $4{,}544\times$ stabilized!** |
| **`+1.0000`** | **`0.0`** | **`-1.82e-5`** (`995.5`) | **`+1.24e-5`** (`E/E0=4.38`) | `+1.127e-1` | **`+1.129e-1`** (`121.3`) | **$S^3/I^*$ completely bounded!** |
| **`+2.0000`** | **`0.0`** | **`-4.88e-6`** (`995.5`) | **`+1.47e-5`** (`E/E0=2.49`) | `+4.165e-2` | **`+4.167e-2`** (`242.7`) | **$S^3/I^*$ completely bounded!** |
| **`+5.0000`** | **`0.0`** | **`+1.21e-5`** (`995.5`) | **`-1.60e-6`** (`E/E0=0.76`) | `+8.483e-3` | **`+8.594e-3`** (`619.1`) | **$S^3/I^*$ completely bounded!** |
| `+10.000` | `0.0` | `-2.28e-5` (`995.5`) | — | `0.0` | `+1.79e-5` (`995.5`) | Both above pinch-off |

**Eccentric KAM Orbit ($x_b = 15.0$, $x \in [15.0, 31.440]$, `max_rel_e00 = 1.76e-13`, $\max_\tau\beta_{\mathrm{pinch,inst}} = 0.7326$):** Matches the full adiabatic envelope with surgical precision: $\beta_0 = 0.05$ (`+0.569 H_S`), $0.10$ (`+0.502 H_S`), and $0.50$ (`+3.57e-2 H_S`, blowup at $\tau = 210.0$, well below `0.7326`) blow up; $\beta_0 = 0.70$ (just inside the envelope edge) drops $10\times$ to `+3.70e-3 H_S` (`blowup = False`, $\tau = 707.5$); and **$\beta_0 = 0.80, 1.00$ ($> 0.7326$) drop another $15\times$ into complete quiescence (`+2.45e-4 H_S` and `+1.76e-4 H_S`, bounded over $\tau = 707.5$)!**

---

## 5. Theorem 3: Manley–Rowe Floquet Physics, Non-Self-Adjoint Order Reduction, and Backreaction Budget

### 5.1 Scalaron Sideband Emission & Positive-Definite Centered Order-Reduced Sweep Across the KAM Island

On P05's $\tilde\sigma_0 = 9/31$ background, the scalaron mass $\omega_{\mathrm{scal}} = \sqrt{31/9} = 1.8559\,H_S$ sits at $4 c_{\mathrm{anom}}\omega_{\mathrm{scal}}^2 = 4.000$, driving Floquet sidebands $\Omega_\pm = \Omega_{\mathrm{scal}} \pm \Omega_0 \approx \{38.22, 43.88\} > M_{I^*}=26$. Using the positive-definite energy norm $E_{\mathrm{pos}} = x^2(|A_R|v^2 + k_n^2|K_R|b^2)$ (eliminating negative-$K_R$ clamping during brief gradient-flash intervals on $x_b = 10, 50$) and centered finite differences $\Delta\lambda^{(1)} = (\lambda(+\epsilon)-\lambda(-\epsilon))/(2\epsilon)$ ($\epsilon = 0.01$, canceling $O(\kappa_{\mathrm{mem}}^2)$ commutator bias identically):

| KAM Orbit $x_b$ | $x \in [x_{\min}, x_{\max}]$ | $\min K_R$ | $T_{\mathrm{macro}}$ ($H_S^{-1}$) | $P_{\mathrm{rad}}/(2 E_{\mathrm{avg}})$ | $\tau_{\mathrm{rad}}$ ($H_S^{-1}$) | Centered $O(c)$ $\Delta\lambda_2^{(1)}$ | Naive $\kappa=1$ SC-OR ($E_{\mathrm{pos,end}}/E_{\mathrm{pos,0}}$) |
|---|---|---|---|---|---|---|---|
| `10.00` | $[10.00, 44.75]$ | `-0.516`* | `177.59` | `+4.049e-3` | `247.0` | **`-1.194e-4`** | `+4.102e-3` (`273.1`) |
| `15.00` | $[15.00, 31.44]$ | `+0.256` | `149.29` | `+4.343e-3` | `230.2` | **`+7.355e-6`** | `+1.139e-3` (`10.70`) |
| **`21.92` ($x_{\mathrm{ring}}$)** | $[20.79, 23.43]$ | `+0.560` | `142.23` (`136.88` long) | **`+3.292e-3`** | **`303.8`** | **`+6.416e-6`** | `+6.357e-4` (`2.34`) |
| `35.00` | $[12.86, 36.26]$ | `+0.032` | `160.92` | `+4.197e-3` | `238.3` | **`-1.726e-4`** | `+1.498e-3` (`13.93`) |
| `50.00` | $[8.48, 51.16]$ | `-1.056`* | `198.95` | `+3.752e-3` | `266.5` | **`-6.620e-5`** | `+5.415e-3` (`3.76e4`) |

*Note:* `*` marks eccentric orbits with brief $K_R < 0$ gradient-flash intervals near bounce. Notice that centered $O(c_{\mathrm{anom}})$ response reveals genuine net damping on $x_b = 10, 35, 50$ (`-1.19e-4, -1.73e-4, -6.62e-5 H_S`) and $99.8\%$ Manley–Rowe cancellation on $x_b = 15, x_{\mathrm{ring}}$ (`+7.36e-6, +6.42e-6 H_S`)—completely masked at $\kappa_{\mathrm{mem}} = 1$ by the $O(\kappa_{\mathrm{mem}}^2)$ commutator artifact!

### 5.2 Proof of Non-Self-Adjointness of Finite-$\kappa_{\mathrm{mem}}$ Order Reduction ($[\mathrm{Re}\,\mathcal{K}_M, \mathcal{B}_0(\eta)] \ne 0$)

**Complete `self_consistent_order_reduced_M26_sweep` on $x_{\mathrm{ring}}$ (all 18 rows reported without omission):**

| $\kappa_{\mathrm{mem}}$ | `loc=False` ($\beta_0=0$) | `loc=True, beta_0=0.289` | `loc=True, beta_0=1.000` |
|---|---|---|---|
| `0.02` | `+9.234e-7` (`1.046`) | `+7.384e-5` (`1.431`) | `+2.601e-4` (`1.234`) |
| `0.05` | `+2.441e-6` (`1.053`) | `+1.887e-4` (`1.610`) | `+6.587e-4` (`1.838`) |
| `0.10` | `+7.494e-6` (`1.067`) | `+3.825e-4` (`1.964`) | `+1.325e-3` (`3.577`) |
| `0.25` | `+4.156e-5` (`1.131`) | `+9.820e-4` (`3.618`) | `+3.338e-3` (`26.62`) |
| `0.50` | `+1.613e-4` (`1.330`) | `+2.041e-3` (`10.50`) | `+6.740e-3` (`765.0`) |
| `1.00` | `+6.357e-4` (`2.341`) | `+4.383e-3` (`102.6`) | `+1.373e-2` (`5.46e5`) |

In the exact 4th-order nonlocal action, $\mathrm{Re}\,\mathcal{K}_M \circ \mathcal{B}_n$ is strictly self-adjoint ONLY because the conformal Bach operator $\mathcal{B}_n = (\partial_\eta^2 + n^2)(\partial_\eta^2 + (n+2)^2)$ has constant coefficients and commutes with $\mathrm{Re}\,\mathcal{K}_M$. Substituting the 0th-order background EOM to form a time-dependent 2nd-order operator $\mathcal{B}_0(\eta) = c_b^B(\eta) + c_v^B(\eta)\partial_\eta$ inside $\mathcal{K}_M[\mathcal{B}_0(\eta)b]$ breaks $[\mathrm{Re}\,\mathcal{K}_M, \mathcal{B}_0(\eta)] = 0$:
- **Pure local control (`km = 0.0`, `loc = True`):** Conservative to $10^{-5}$ (`-9.035e-6 H_S` at $\beta_0 = +1.0$; `+2.524e-6 H_S` at $\beta_0 = -1.0$).
- **Odd-in-$\beta_0$ falsification proof (`km = 0.25`, `loc = True`, `part5`):** Flipping $\beta_0 = +1.0 \to -1.0$ flips the rate from **`+3.300e-3 H_S`** (`E/E0 = 16.36`, Aux-ODE vs. Direct Volterra `2.995e-3`) to **`-3.238e-3 H_S`** (`E/E0 = 0.224`, Aux-ODE vs. Direct Volterra `1.975e-3`)—exact odd parity in $\beta_0$ verified by both independent solvers! Physical 2-photon radiation cannot invert sign under $\beta_0 \to -\beta_0$; this proves the $O(\beta_0\kappa_{\mathrm{mem}})$ and $O(\kappa_{\mathrm{mem}}^2)$ drifts (`loc=False` grows quadratically as $\kappa_{\mathrm{mem}}^2$, $c_2 = +1.27\times 10^{-3}\,H_S$) are non-self-adjoint commutator artifacts of finite-$\kappa$ EOM-substitution order reduction.
- **Dynamically Faithful Extraction:** True physical 1-loop rates must come either from **centered** first-order linear response $(\lambda(+\epsilon)-\lambda(-\epsilon))/(2\epsilon)$ (where $O(\kappa^2)$ cancels identically!) or from the **chiral-stabilized full unreduced 4th-order Volterra solver** ($\beta_0 \ge 0.50$, where $\mathcal{B}_2$ is exact and yields $|\gamma_E| \le 1.47 \times 10^{-5}\,H_S$ over $\tau = 885\,H_S^{-1}$).

### 5.3 Radiative Backreaction Energy Budget: Shear vs. Catalyzed Scalaron Condensate

Shear energy alone converts into radiation as
$$\Delta\kappa_{\mathrm{shear}} = \frac{8}{3} x_{\mathrm{ring}}^2 b_0^2 = 1280.92\,b_0^2\,, \qquad b_{\mathrm{crit}}^{\mathrm{shear}} = \sqrt{\frac{3|\kappa_V|}{8 x_{\mathrm{ring}}^2}} = 0.6128$$
($0.07\%$ above $\sqrt{3/8} = 0.6124$ because $|\kappa_V|/x_{\mathrm{ring}}^2 = 44738/44672 = 1.00148$). For $b_0 = 0.10$, $\Delta\kappa_{\mathrm{shear}} = 12.81 \ll |\kappa_V| = 481.05$. However, because upper and lower Floquet sidebands cancel to $>99.7\%$ on the graviton amplitude itself, energy conservation gives $P_{\mathrm{scal}}/P_{\mathrm{rad}} = 1 - P_{\mathrm{grav}}/P_{\mathrm{rad}} \approx \mathbf{100\%}$ (slightly $>100\%$ whenever $\Delta\lambda_2^{(1)} > 0$): virtually the entire radiated 2-photon power is drawn from the **scalaron condensate**, whose total bounce energy budget from $\mathcal{E}_{00} = 0$ at $x_{\mathrm{ring}}$ is
$$\kappa_{\mathrm{scal}} = x_{\mathrm{ring}}^4\left(\tilde\sigma_0 \dot h_b^2\right) = 2 x_{\mathrm{ring}}^2 = \mathbf{960.69} = \mathbf{1.997\,|\kappa_V|}\,.$$
Thus catalyzed scalaron depletion taps a reservoir twice as large as $|\kappa_V|$, making nonlinear scalaron + radiation backreaction essential to determine the long-term fate of the $\Lambda = 0$ Casimir bounce.

---

## 6. Theorem 4: Inhomogeneous $S^3/I^*$ TT Gravitons & Mathieu Reheating at $n = 18, 22$

Using centered finite differences at BOTH $\epsilon = 0.01$ and $\epsilon = 0.02$ (eliminating $O(\kappa_{\mathrm{mem}}^2)$ commutator contamination identically and confirming 4-digit $\epsilon$-invariance across all bounded and $n=18$ modes), we map every $S^3/I^*$ TT graviton mode up to $n = 22$ alongside round-$S^3$ modes $n = 2, 3, 4$ on $x_{\mathrm{ring}}$:

| Mode & Exact Threshold | $\omega_{\mathrm{phys}}/H_S$ | $\epsilon_{\mathrm{EFT}}$ | 0th-Order $\lambda_n^{(0)}/H_S$ | Centered $O(c)$ $\Delta\lambda_n^{(1)}$ ($\epsilon=0.01$ / $0.02$) | Commutator $c_2 \equiv \partial_\kappa^2\lambda|_0$ | Bounded $P_{\mathrm{rad}}/(2E_{\mathrm{avg}})$ | Physical Status |
|---|---|---|---|---|---|---|---|
| `Istar_n2_M26` | `0.129` | `0.019` | `-2.96e-6` | **`+7.772e-6`** / `+7.771e-6` | `+1.27e-3` | `+3.293e-3` | Bounded; $99.76\%$ Manley–Rowe cancellation |
| `S3_n2_M6` | `0.129` | `0.019` | `-2.96e-6` | **`-5.180e-6`** / `-5.183e-6` | `-1.09e-2` | `+3.293e-3` | Tiny linear damping (but unreduced ghost blows up!) |
| `S3_only_n3_M7` | `0.177` | `0.036` | `+1.78e-6` | `+1.155e-6` / `+1.156e-6` | `-1.09e-2` | `+5.962e-3` | Projected out by $I^*$ |
| `S3_only_n4_M8` | `0.224` | `0.058` | `+1.62e-6` | `+7.127e-6` / `+7.143e-6` | `-1.17e-2` | `+9.387e-3` | Projected out by $I^*$ |
| `Istar_n10_M14` | `0.500` | `0.290` | `+7.06e-6` | **`+9.805e-5`** / `+9.844e-5` | `-5.63e-2` | `+4.529e-2` | Bounded at 0th order; weak MR anti-damping |
| **`Istar_n14_M26`** | `0.683` | `0.542` | `+1.53e-5` | **`-1.241e-2`** / `-1.241e-2` | `+4.82e-3` | `+6.165e-2` | **Strong 1-loop damping!** ($M=26 \approx |\Omega_-|=26.08$ suppresses down-conversion!) |
| **`Istar_n18_M22`** | `0.866` | `0.870` | **`+8.791e-2`** | **`-2.049e-2`** / `-2.050e-2` | `-1.79e-2` | *(Mathieu)* | **2:1 Mathieu reheating** ($E/E_0 = 2.39\times 10^9$ over $\eta=6$; $O(c)$ trims rate by $23\%$) |
| **`Istar_n22_M26`** | `1.048` | `1.277` | **`+1.309e-2`** | **`-2.205e-3`** / `-2.747e-3` | `-8.54e-1` | *(Mathieu)* | **2:1 Mathieu reheating** (above $\epsilon_{\mathrm{EFT}}=1$ species cutoff) |

Because $4 c_{\mathrm{anom}} \omega_{\mathrm{scal}}^2 = 4.000$ on $\tilde\sigma_0 = 9/31$, the 2:1 parametric reheating resonance $\omega_n = \omega_{\mathrm{scal}}/2$ coincides **algebraically** with the 1-loop graviton EFT boundary $\epsilon_{\mathrm{EFT}} = 1$ at $n = \sqrt{44672/108 + 1} - 1 = \mathbf{19.3625}$, right between $n = 18$ and $n = 22$.

---

## 7. Independent Falsification Kill-Tests (KT1–KT5)

All five independent kill-tests executed in [`t4a_03_kam_phase_map_and_kill_tests.py`](file://open_problems/track4_nonlocal_weyl_kam/t4a_03_kam_phase_map_and_kill_tests.py) passed cleanly:

| ID | Kill-Test Description | Key Quantitative Verification | Verdict |
|---|---|---|---|
| **KT1** | Time-domain Direct Volterra work rate vs. Parseval FFT at exact $M_{I^*}=26$ on $x_{\mathrm{ring}}$ | $\gamma_{\mathrm{TD}} = +3.2806 \times 10^{-3}\,H_S$ vs. $\gamma_{\mathrm{FFT}} = +3.2964 \times 10^{-3}\,H_S$ (**rel diff `0.48%`**) | **PASS** |
| **KT2** | Synthetic pump ($\Omega_s=12, M_{S^3}=6, \kappa=0.1$) exact Manley–Rowe sign flip & $\alpha^2$ scaling | $\alpha=0$: `+4.9e-7` vs `0`; $c_K=-5, \alpha=0.02/0.04$: `+4.54e-4 / +1.93e-3` vs `+4.29e-4 / +1.72e-3`; $c_K=+5$: **`-1.096e-3` vs `-1.099e-3`** | **PASS** |
| **KT3** | EFT-safe light scalaron ($\tilde\sigma_t=10.0$, $4 c_{\mathrm{anom}}\omega_{\mathrm{scal}}^2 = 0.116 \ll 1$, $\Omega_{\mathrm{scal}}=7.00$, long $\eta=80$ $>6$ beats, centered $\epsilon=0.02$): $n=2$ ($M_{I^*}=26$, sub-thresh $\Omega_+=9.83<26$) vs. $n=10$ ($M_{I^*}=14$, active $\Omega_+=17.95>14$) | $n=2$: $P/(2E) = 5.54 \times 10^{-7}$, centered $\Delta\lambda_2^{(1)} = +3.01 \times 10^{-7}\,H_S$ (**strictly zero**, $E/E_0=0.975$); $n=10$: $P/(2E) = 7.84 \times 10^{-5}\,H_S$ ($142\times$ turn-on!), centered **$\Delta\lambda_{10}^{(1)} = -2.201 \times 10^{-5}\,H_S$ (genuine EFT-safe damping!)**, $\lambda(1) = -1.709 \times 10^{-4}\,H_S$, **$E_{\mathrm{pos,end}}/E_{\mathrm{pos,0}} = 0.5194$** | **PASS** |
| **KT4** | Quadrature & step-size convergence at $M_{I^*}=26$: short ($\beta_0=0.289, \eta=2$) AND long Regime IV ($\beta_0=1.0, \eta=40, \tau=885$) | Short Aux-ODE $N_u=48\to 64$ vs. $96$: `7.89e-4` $\to$ `1.66e-5`; Volterra $d\eta=0.004\to 0.001$: `7.54e-2` $\to$ `2.14e-2` $\to$ `5.91e-3` (ratios `3.52, 3.63`); **Long Regime IV ($\beta_0=1.0, \eta=40$) $N_u=48, u_{\max}=10^6$ vs. $N_u=80, u_{\max}=10^7$: `rel_q1 = 1.54e-4`, $\gamma = -1.217\times 10^{-5}$ vs. $-1.219\times 10^{-5}\,H_S$** | **PASS** |
| **KT5** | Canonical Ostrogradsky energy sign signature ($M_{I^*}=26, \beta_0=0.289$) | Pure physical graviton $H_{\mathrm{ostro}}(0) = +3.844 \times 10^3 > 0$; late runaway $H_{\mathrm{ostro}}(\eta=3) = -4.313 \times 10^{14} < 0$ (strictly negative-energy ghost) | **PASS** |

---

## 8. Epistemic Boundaries, Caveats & Follow-Up Questions

1. **Discrete $S^3/I^*$ 2-Photon Pole Tower vs. Continuum-Log Kernel (`assumed` vs. `exact`):** On compact $S^3/I^*$, the **analyticity gap $|\Omega| < M_{I^*}(n)$** ($M_{I^*}(2) = 26$, $\mathrm{Im}\,\Pi(\Omega) \equiv 0$) is **exact**, whereas the numerical pinch-off value $\beta_{\mathrm{pinch}}(S^3/I^*) = 0.383639$ (and adiabatic max `0.4206`) relies on replacing the discrete 2-photon pole sum above $M_{I^*}$ by the single-threshold continuum logarithm $\ln((M^2 - (\Omega+i0^+)^2)/\mu_{\mathrm{cyl}}^2)$. Computing the exact discrete spectral sum on $S^3/I^*$ is needed to quantify how much the real-part pole repulsion shifts $\beta_{\mathrm{pinch}}$ and whether individual narrow 2-photon resonances intersect $\Omega_\pm \approx \{38.2, 43.9\}$.
2. **Pure Maxwell 1-Loop Closure & Graviton Self-Loops:** Our 1-loop action models $N_v$ conformally coupled Maxwell fields (large-$N_v$ dominance). Graviton self-loops open a 2-graviton cut at $2\Omega_0 = 2\sqrt{8} \approx 5.66 < 26$ suppressed by $O(1/N_v)$ relative to Maxwell loops; testing whether $O(1/N_v)$ graviton-loop decay re-opens a weak Horowitz instability inside $\beta_0 \in [0.44, 6.37]$ is an important multi-species check.
3. **EFT Boundary of P05's $\tilde\sigma_0 = 9/31$ Scalaron Condensate:** While $n=2$ shear is deeply EFT-safe ($\epsilon_{\mathrm{EFT}} = 0.0193$), P05's scalaron sits at $4 c_{\mathrm{anom}} \omega_{\mathrm{scal}}^2 = 4.000$ (and $n=18, 22$ Mathieu modes sit at $\epsilon_{\mathrm{EFT}} = 0.87\text{--}1.28$). Higher-loop $R^3 / R^2 C^2$ operators or lighter-scalaron regimes ($\tilde\sigma_t \gtrsim 4$, as in `KT3`) are required for strict perturbative control of the scalaron sector.
4. **Nonlinear Self-Consistent Scalaron + Radiation Backreaction (Future Nonlinear Backreaction Follow-Up):** Because catalyzed 2-photon emission taps the $\kappa_{\mathrm{scal}} = 960.7 \approx 2.0\,|\kappa_V|$ scalaron condensate energy budget, coupling $n=2$ shear, $n=18, 22$ Mathieu gravitons, and emitted radiation density $\Delta\kappa_{\mathrm{rad}}(\tau)$ nonlinearly back into the Friedmann constraint $\mathcal{E}_{00} = 0$ is the decisive next step to determine whether the $S^3/I^*$ KAM island thermally exits to radiation-dominated expansion or settles onto a damped breathing attractor.
