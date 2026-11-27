# Track IV-B Technical Report: Dynamical Standard-Model Hosotani Restoration, Real-Time $\rho_2 \to S_{02} \to \rho_0$ Avalanche Cosmology, and Nonlinear $n = 12$ Saturation on $S^3/I^*$

**Stream:** `t4b_dynamical_hosotani_rho2` · **Workspace:** `open_problems/track4_dynamical_hosotani_rho2/`  
**Baseline References:** [P02_Inhomogeneous_SU2_Higgs_Report.md](file://open_problems/p02_inhomogeneous_su2_higgs/P02_Inhomogeneous_SU2_Higgs_Report.md) · [P03_Galois_Rho2_Technical_Note.md](file://open_problems/p03_galois_rho2/P03_Galois_Rho2_Technical_Note.md) · [Poincare_Sphere_Gauge_Cosmology_Monograph.md](file://Poincare_Sphere_Gauge_Cosmology_Monograph.md) (§4)  
**Audit Status:** Cold `blinded-critic` audit completed (`3 P0 + 5 P1 + 2 P2` — **all 10 findings resolved and re-verified** in [t4b_05_audit_remediation.py](file://open_problems/track4_dynamical_hosotani_rho2/t4b_05_audit_remediation.py) / [t4b_05_audit_remediation.json](file://open_problems/track4_dynamical_hosotani_rho2/t4b_05_audit_remediation.json)).  
**Primary Summary Figure:** [t4b_dynamical_hosotani_rho2_summary.png](file://open_problems/track4_dynamical_hosotani_rho2/t4b_dynamical_hosotani_rho2_summary.png)

---

## Executive Summary

Extending Track II (Monograph §4, Problem 2, Problem 3) from frozen-$\phi$ stiff-Higgs perturbation theory and static 3D variational profiles to **full real-time dynamical coupled Einstein–$SU(2)_L$–Higgs–$\Lambda$ evolution** on the Poincaré homology sphere $S^3/I^*$ resolves four core dynamical questions with verified exact conservation laws, independent kill-tests, and explicit UV/gauge-validity boundaries:

1. **Exact Goldstone-Covariant Linearized Fluctuation Theory (Regular at $\phi = 0$, Goal 1):** Casting the quadratic $SU(2)_L$–Higgs action in gauge-scaled variables $q = (\alpha, \hat{\pi}, \tilde{h})^\top$ with generalized covariant derivative $C(\eta) = (D(f(\eta)), -\mu(\eta), 0)^\top$ ($\mu(\eta) = \frac{1}{2}g_{\text{YM}} a(\eta)\phi(\eta)$) eliminates the unitary-gauge $1/\phi$ singularity identically, satisfies the exact dynamical Gauge Ward identity $\mathcal{V}(\eta)C(\eta) + C''(\eta) \equiv 0$ to `2.885e-12` (`2.261e-12` at exact $\phi = 0$) across all $(n, K)$ blocks ($n \in \{0, 1, 2, 4, 12, 20\}$), and admits a globally nonsingular parallel-transport generalized Coulomb reduction on $C(\eta)^\perp$ ($\|C\|^2 = \|D\|^2 + \mu^2 \ge 20 > 0$ for all $n \ge 12$ on $S^3/I^*$). Across a Standard-Model ($\beta_H = m_H/M_W = 1.556$) Hosotani-restoring bounce with 3 zero crossings of $\phi(\eta)$, temporal gauge (Path A) and generalized Coulomb reduction (Path B) agree entry-for-entry to `3.881e-12` (`max Gauss drift = 1.447e-11`, `pure-gauge tracking = 1.045e-13`, `|det M - 1| = 1.175e-11`). On Kill-Test 1B against the full 14-variable nonlinear triaxial Bianchi IX ODE at SM ratio $\beta_H = 1.556$ ($\min\phi/v = -2.0638$, 3 zero crossings), our dynamical linear + TT-gravity ODE matches the nonlinear peak (`17.0445` vs. `17.0459`) to **`8.645e-5`**, closing P02's `22.2%` frozen-$\phi$ caveat by $2{,}500\times$.

2. **Dynamical SM Hosotani Restoration Preserves—and at High $E$ Enhances—the $S^3/I^*$ $n = 12$ Spectral Gap (Goals 1 & 3A):** Across all 11 surveyed $(E_{\text{init}}, M_W)$ bounce configurations ($\gamma_{\text{YM}} = 0.10$, $E_{\text{init}} \in [50, 2500]$, $M_W \in [1.0, 5.0]$ using P02's total initial gauge-plus-Higgs-gradient energy convention), SM dynamical $\phi$-rolling ($\phi \to 0 \to -2.84v$) shatters stiff-Higgs stabilization on $S^3$ ($n = 4$ amplifies to `87.62×` at `(500, 2.5)`, `478.94×` at `(1000, 2.5)`, and `144.78×` at `(2500, 4.0)`, while intermediate $\beta_H = 4.0$ excites a `3186.14×` Goldstone-coupled `2K=2` resonance at `(1000, 2.5)`). By contrast, **$S^3/I^*$ remains strongly protected across every SM bounce**: across all 11 configs ($M_W \in [1.0, 5.0]$), SM $n = 12$ stays within **`1.07×–8.73×`** (`8.73×` at weak Higgs $M_W = 1.0, E = 500$, still below pure-YM `12.13×` and stiff `9.01×`), and for all $M_W \ge 2.5$ stays strictly within **`1.07×–4.10×`** (`n = 20 <= 1.41×`). At `(2500, 4.0)` (where pure-YM $n=12$ hits `1788.71×` and stiff Higgs gives `12.37×`), SM dynamical Hosotani rolling (`4` zero crossings) **quenches $n = 12$ to `4.10×`** (`436×` below pure YM and `3.0×` below stiff Higgs; even under strict pure-YM-matched initial energy $E_{\text{YM},0} = 2500$, SM gives `9.45×` vs. stiff `13.97×`, confirming a genuine dynamical energy-drain + overshoot mechanism). Full nonlinear $k = 1$ ($N_{\max} = 20$, `D_A = 21, D_Phi = 10`) simulations at large finite seed $\epsilon_{12} = 0.1$ confirm this nonlinearly: at `(2500, 4.0)`, SM Hosotani rolling caps the peak inhomogeneous energy fraction at **`1.75%`** (`dT00_tot = 0.02%`, `H_res = 9.04e-13`), **`10.8×` lower than stiff Higgs (`18.91%`)**.

3. **Real-Time Cosmological $\rho_2 \to S_{02} \to \rho_0$ Avalanche Decay, Dynamical Supercooling, and the $\gamma_c = \frac{1}{4}$ Collapse Bifurcation (Goals 2 & 3B):** Built a unified temporal-gauge winding-$k$ ($k \in \{1, 7\}$) real-time 3+1D $SU(2)_L$ + full 4-component complex Higgs doublet + FLRW solver up to $N_{\max} = 36$ (`51` gauge + `24` Higgs DOF, FD gradients `< 4.3e-8`, exact $n=0$ global Gauss conservation `~1e-14`, exact Friedmann conservation `~5e-15`, nonlinear $k=1$ vs. Goal 1 linear Kill-Test 2A agreement `1.832e-8` at endpoint). Simulating real-time SM ($\beta_H = 1.556$) $\rho_2 \to S_{02} \to \rho_0$ avalanches from variationally relaxed flat $\rho_2$ ($E_{\text{YM}} \le 2.4 \times 10^{-6}$, $N_{\text{CS}} = \frac{49}{120}$ to 10 digits) reveals three regimes:
   - *Super-Horizon Expanding Avalanche ($M_W = 5.0$, $a_{\text{restore}} = 1.4428 > 1$, $a: 1.10 \to 25.0$):* Below $a_{\text{restore}}$, $\hat{\Phi}$ damps (`0.050 -> 0.0067` at $a_{\text{restore}}$, coasting to `0.00049` at $a = 1.531$), confirming Hosotani stability. Past $a_{\text{restore}}$, supercooling delays the plunge to $a_{\text{mid}} = 6.34$, whereupon $N_{\text{CS}}$ drops through $S_{02}$ (`0.2406`) to a first trough **`+0.00047` at $a = 7.33$** (`+0.00058` at $a = 7.24, N_{\max}=36$) before **coherently rebounding to `+0.4919` at $a = 9.46$** (converged across $N_{\max} = 20..36$) and oscillating toward `0` (`+0.00386` at $a = 25$), with `H_res <= 6.85e-15`, `dT00_tot <= 0.15%`, dense `G_n20(a<=8)` dropping monotonically `912×` (`1.66 -> 1.82e-3`), `G_rel(full) = 6.84e-4` at $N_{\max} = 36$, and post-avalanche `DOP853` vs. `Radau` agreement **`2.386e-13`** (`|dN_CS| = 3.56e-15`).
   - *Sub-Horizon False-Vacuum Bounce ($M_W = 22.0$, $a_{\text{restore}} = 0.3279 < 1/\sqrt{2}$):* Even when $a_{\text{bounce}} = 0.2902 < a_{\text{restore}} < 1/\sqrt{2}$, finite tachyonic growth supercools the transition by $R_{\text{sim}} = a_{\text{av}}/a_{\text{restore}} = 2.367$ ($a_{\text{av}} = 0.776 > 1/\sqrt{2}$, consistent with bare WKB $R = 1.944$, Hubble-diluted WKB $R = 2.212$, and exact-dS diluted WKB $R = 1.857$ plus pre-threshold damping and nonlinear gauge roll-off time). Because $\gamma_{\text{rad}} \propto R^4$, dynamical supercooling boosts $\gamma_{\text{rad}}$ to `1.463 > 1/4`, driving **supercooled escape** across all 10 tested sub-horizon runs (`H_res <= 3.5e-12`, `dT00_tot <= 9.92%`).
   - *Pre-Bounce Contracting Avalanche & $\gamma_c = \frac{1}{4}$ Bifurcation ($M_W = 5.0$, contracting from $a_0 = 4.5$):* When the $\rho_2 \to S_{02} \to \rho_0$ avalanche fires during contraction ($a: 4.5 \to 1.5$), pre-bounce Gauss residuals converge exponentially (`G_n20: 2.19 -> 0.392 -> 8.83e-4`, `G_rel: 8.76e-3 -> 6.80e-4` across $N_{\max} = 24 \to 36$) and the macroscopic radiation-bounce ceiling $\gamma_c = \frac{1}{4}$ converges to 5 digits: $\kappa_{\text{YM}} = 5\times 10^{-6}$ ($\gamma_{\text{tot}}(a \sim 1) = 0.1315 < \frac{1}{4}$) executes a **nonsingular bounce** (`a_min = 0.91872 -> 0.91870 -> 0.91869`), whereas $\kappa_{\text{YM}} = 10^{-5}$ ($\gamma_{\text{tot}}(a \sim 1) = 0.2622 > \frac{1}{4}$) **destroys the bounce and triggers singular Big Crunch collapse**. Explicitly flagging our cold-audit Gauss gate: once the hot contracting plasma compresses past $a \lesssim 1.09$ ($\eta \gtrsim 0.90$ at $N_{\max}=36$), classical UV cascade leaks above $N_{\max}$ (`G_rel` rising to `4.04–11.84`), so late-time contracting $N_{\text{CS}}$ excursions and $\delta_{T_{00}}$ beyond $a \lesssim 1.09$ are UV-cutoff-sensitive rather than continuum-converged.

4. **Nonlinear Mode-Mode Saturation of $n = 12$ in High-$E$ Pure-YM Plasma ($E_{\text{YM}} \ge 2100$, Goal 3C):** At $E_{\text{YM}} = 2500$ and $2843.9$ ($\gamma_{\text{YM}} = 0.10$, $k = 1$, $N_{\max} = 24$, `D_A = 30`), finite $n = 12, K = 4$ seeds saturate nonlinearly (compressing gain at $E = 2843.9$ from `150.07×` linear to `58.24×` at $\epsilon_{12} = 0.1$, with `79.64%` inhomogeneous energy share) while **strictly conserving the conformal radiation parameter** $\gamma_{\text{YM}}$ (`drift <= 1.37e-14`) and **preserving the exact bounce radius** (`a_min = 0.94195` vs. homogeneous `0.94195`, `dT00_tot <= 0.06%`). In long-time fixed-$a$ evolution ($\eta \in [0, 8.0]$, $E = 2500$, energy drift `3.21e-14`), the strictly positive **non-clamped kinetic inhomogeneous share** saturates at mean **`74.06%`** (`n=0: 4.87%, n=12: 36.12%, n=20: 27.03%, n=24: 31.98%` at $\eta = 8$, `dT00_mat = 32.4%`). Fixing the canonical cubic sign of $v_{k4}$ gives 6-digit convergence across $N_{\max} = 12 \to 30$ (`0.003737` for $+v_{k4}$, `0.004256` for $-v_{k4}$) and `DOP853` vs. `Radau` agreement **`1.290e-11`** (`|da_min| = 8.23e-7`).

![Track IV-B Audited Six-Panel Diagnostic Summary](file://open_problems/track4_dynamical_hosotani_rho2/t4b_dynamical_hosotani_rho2_summary.png)

---

## 1. Goal 1: Exact Goldstone-Covariant Fluctuations Around Dynamical SM Hosotani-Restoring Bounces

### 1.1 Exact Non-Unitary-Gauge Quadratic Lagrangian & Dynamical Gauge Ward Identity

Around an arbitrary dynamical isotropic FLRW background $(a(\eta), f(\eta), \phi(\eta))$ in conformal time $\eta$ ($ds^2 = a(\eta)^2[-d\eta^2 + \sum_i (\omega^i)^2]$, $\bar{A}_0^a = 0$, $\bar{A}_i^a = \lambda_f(\eta)\delta_i^a$ with $\lambda_f := 1 + f$, $\Phi_0 = (0, \phi(\eta)/\sqrt{2})^\top$, $8\pi G = 1$, $\Lambda = 3$, $\kappa_{\text{YM}} = g_{\text{YM}}^{-2}$, $v = 2 M_W \sqrt{\kappa_{\text{YM}}}$, $\lambda_H = m_H^2/(2v^2)$), the coupled background satisfies:
$$a'' = 2a^3 - a - \frac{a}{6}(\phi')^2 + \frac{a}{8}\lambda_f^2 \phi^2 + \frac{2}{3}a^3 V_H(\phi), \qquad f'' = -2f(f^2-1) - \mu(\eta)^2 \lambda_f,$$
$$(a\phi)'' + \left[\frac{3}{4}\lambda_f^2 + \lambda_H a^2(\phi^2 - v^2) - \frac{a''}{a}\right](a\phi) = 0, \qquad \mu(\eta) := \frac{1}{2}g_{\text{YM}} a(\eta)\phi(\eta).$$

Expanding $A_0^a = \alpha_0^a$, $A_i^a = \lambda_f \delta_i^a + \alpha_i^a$, and $\Phi = \frac{1}{\sqrt{2}}[(\phi + h)I + 2\pi^a T_a](0, 1)^\top$, and passing to the **gauge-scaled, conformally rescaled variables** (regular through $\phi = 0$ because $g_{\text{YM}} > 0$ and $a(\eta) > 0$):
$$\tilde{h} := g_{\text{YM}} a(\eta) h, \qquad \hat{\pi}^a := -i g_{\text{YM}} a(\eta) \pi^a, \qquad \hat{\alpha}_0^a := -i\alpha_0^a,$$
every reduced custodial-spin block $(n, K)$ (built via [istar_harmonics.py](file://open_problems/p02_inhomogeneous_su2_higgs/istar_harmonics.py)) decouples with real state vector $q := (\alpha, \hat{\pi}, \tilde{h})^\top \in \mathbb{R}^{d_\alpha + d_0 + d_h}$ and generalized covariant derivative $C(\eta) := (D(f(\eta)), -\mu(\eta), 0)^\top$ ($D(f) = D_0 + \lambda_f D_1$):

> **Theorem 1.1 (Exact Goldstone-Covariant Quadratic Action & Dynamical Ward Identity).** *On every dynamical solution $(a(\eta), f(\eta), \phi(\eta))$ of the coupled FLRW Einstein–$SU(2)_L$–Higgs–$\Lambda$ equations—including across arbitrary zeros $\phi(\eta) = 0$—the exact quadratic Yang–Mills + Higgs Lagrangian in each $(n, K)$ block is:*
> $$g_{\text{YM}}^2 L^{(2)} = \frac{1}{2}\|q' - C(\eta)\hat{\alpha}_0\|^2 + \hat{\alpha}_0^\top C'(\eta)^\top q - \frac{1}{2} q^\top \mathcal{V}(\eta)\, q,$$
> *where the real symmetric potential matrix $\mathcal{V}(\eta)$ is:*
> $$\mathcal{V}(\eta) = \begin{pmatrix} \mathcal{M}_{\text{YM}}^2(f) + \mu^2 I_{d_\alpha} & \mu D_0 & \sqrt{3}\lambda_f \mu e_{\text{trace}} \\ \mu D_0^\top & V_{\pi\pi} & \sqrt{3}\lambda_f D_0^\top e_{\text{trace}} \\ \sqrt{3}\lambda_f \mu e_{\text{trace}}^\top & \sqrt{3}\lambda_f e_{\text{trace}}^\top D_0 & V_{hh} \end{pmatrix},$$
> *with $V_{\pi\pi} = D_0^\top D_0 + \lambda_f D_0^\top D_1 + \frac{3}{4}\lambda_f^2 + \lambda_H a^2(\phi^2 - v^2) - \frac{a''}{a}$ and $V_{hh} = n(n+2) + \frac{3}{4}\lambda_f^2 + \lambda_H a^2(3\phi^2 - v^2) - \frac{a''}{a}$. The system satisfies the exact dynamical Gauge Ward identity:*
> $$\mathcal{V}(\eta)\, C(\eta) + C''(\eta) \equiv 0 \qquad \forall\,\eta \in \mathbb{R}.$$
> *Consequently: (i) in temporal gauge ($\hat{\alpha}_0 \equiv 0$), $q'' + \mathcal{V} q = 0$ admits $q_{\text{gauge}}(\eta) = C(\eta)$ as an exact analytical solution and conserves the Gauss constraint $\mathcal{G} = C^\top q' - C'^\top q$ identically; (ii) whenever $\|C(\eta)\|^2 = \|D(f)\|^2 + \mu^2 > 0$ (which holds globally on $S^3/I^*$ for all $n \ge 12$ since $\|D(f)\|^2 \ge 2n - 4 \ge 20$, even at $\phi = 0$), parallel-transporting an orthonormal frame $Q(\eta)$ spanning $C(\eta)^\perp$ ($Q' = -C(Q^\top C')^\top/\|C\|^2$) eliminates $\hat{\alpha}_0$ nonsingularly to give the exact physical equation:*
> $$u''(\eta) + \left[ Q(\eta)^\top \mathcal{V}(\eta) Q(\eta) + \frac{3}{\|C(\eta)\|^2} (Q(\eta)^\top C'(\eta))(Q(\eta)^\top C'(\eta))^\top \right] u(\eta) = 0.$$ $\square$

### 1.2 Verification Suite & Kill-Test 1B (`t4b_01_goldstone_covariant.json`)

Implemented and verified in [t4b_01_goldstone_covariant.py](file://open_problems/track4_dynamical_hosotani_rho2/t4b_01_goldstone_covariant.py) ([t4b_01_goldstone_covariant.json](file://open_problems/track4_dynamical_hosotani_rho2/t4b_01_goldstone_covariant.json)):

| Independent Verification Check | Scope | Max Error |
| :--- | :--- | :---: |
| Dynamical Gauge Ward identity $\|\mathcal{V}C + C''\|$ along SM Hosotani bounce | 15 $(n, K)$ blocks ($n \in \{0,1,2,4,12,20\}$), 10 times | `2.885e-12` |
| Dynamical Gauge Ward identity at **exact $\phi = 0, \phi' \neq 0$** | All 15 blocks | `2.261e-12` |
| Flat-connection Higgs+Goldstone spectra at $\rho_0$ ($n(n+2)$) & $\rho_1$ ($(n\pm 1)(n\pm 1+2)$) | $n \in \{1, 2, 4, 12, 20\}$ | `5.684e-14` (both) |
| **Path A (temporal) vs. Path B (generalized Coulomb)** across SM bounce (`3` $\phi$-crossings) | All $(n, K)$ for $n \in \{4, 12, 20\}$ | `3.881e-12` |
| Exact Gauss conservation $|\mathcal{G}(\eta_{\text{end}})|$ & pure-gauge tracking $\|q - C\|/\|C\|$ | Path A across SM bounce (`3` $\phi$-crossings) | `1.447e-11` / `1.045e-13` |
| Symplectic determinant $|\det M - 1|$ | Path A & Path B across SM bounce | `1.175e-11` |
| **Kill-Test 1B:** Dynamical $(a,f,\phi)$ linear + TT-gravity vs. 14-var nonlinear Bianchi IX at **SM $\beta_H = 1.556$** | $E=1000, M_W=2.5, \kappa=10^{-4}$ ($\min\phi/v = -2.0638$, 3 crossings) | **`8.645e-5`** (`17.0459` vs. `17.0445`; decoupled `2.65e-2`) |

### 1.3 Complete Dynamical SM Hosotani vs. Stiff-Higgs Bounce Stability Survey ($\gamma_{\text{YM}} = 0.10$)

*Conventions disclosed explicitly:* Following P02, $E_{\text{init}} := \frac{1}{2}(f'_0)^2 + \frac{1}{2}(f_0^2 - 1)^2 + \frac{1}{2}(a_0 M_W)^2(1 + f_0)^2$ holds total initial gauge + Higgs-gradient energy fixed at $a_0 = 2.0, f_0 = 0$ (which reduces initial pure-YM kinetic energy by $2 M_W^2$ relative to Pure YM—`1.3%` at `(2500, 4.0)`, `10%` at `(500, 5.0)`, `25%` at `(50, 2.5)`—separately verified below under strict pure-YM-matched initial energy). Bogoliubov spectral norms normalize the final state at $a_{\text{end}} = 6.0$ against phase-independent vacuum reference frequencies $\omega_{\text{ref}}(a; M_W, m_H)$ at $\phi = v$ (which adds an $O(1)$ phase-sensitive factor when SM $\phi(\eta_{\text{end}})$ is still oscillating around $v$). All values below are verified programmatically against [t4b_01_goldstone_covariant.json](file://open_problems/track4_dynamical_hosotani_rho2/t4b_01_goldstone_covariant.json):

| $(E_{\text{init}}, M_W)$ | Regime ($\beta_H$) | $\phi$-crossings ($\min\phi/v$) | $n = 0$ (hom.) | $n = 4$ ($S^3$) `[2K]` | **$n = 12$ ($S^3/I^*$) `[T / G, 2K]`** | **$n = 20$ ($S^3/I^*$)** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `(50, 2.5)` | Pure YM / SM (`1.556`) / Stiff (`20`) | `0` / `0 (+0.432)` / `0 (+0.999)` | `3.88` / `2.57` / `2.08` | `5.91` / `1.42` / `1.34` | `1.17` / **`1.07`** / `1.08` | `1.07` / **`1.04`** / `1.05` |
| `(200, 2.5)` | Pure YM / SM (`1.556`) / Stiff (`20`) | `0` / `2 (-1.896)` / `0 (+0.994)` | `87.25` / `2.03` / `2.05` | `151.42` / `3.68` / `3.96` | `3.78` / **`1.10`** / `1.11` | `1.16` / **`1.06`** / `1.05` |
| `(500, 1.0)` | **SM (`1.556`)** | **`3` (`-2.843`)** | `57.90×` | `243.47×` `[0]` | **`8.73×` `[8.73 / 1.27, 8]`** | **`1.07×`** |
| `(500, 1.0)` | Stiff (`20.0`) | `0` (`+0.848`) | `193.21×` | `224.57×` `[0]` | `9.01×` `[9.01 / 1.32, 8]` | `1.09×` |
| `(500, 2.5)` | Pure YM (`—`) | `0` (`0.000`) | `387.93×` | `417.56×` `[0]` | `12.13×` `[12.13 / 1.44, 8]` | `1.08×` |
| `(500, 2.5)` | **SM (`1.556`)** | **`3` (`-2.041`)** | `3.64×` | **`87.62×` `[0]`** | **`1.56×` `[1.56 / 1.14, 8]`** | **`1.08×`** |
| `(500, 2.5)` | Interm (`4.0`) | `0` (`+0.298`) | `17.02×` | `364.36×` `[2]` | `1.91×` `[1.39 / 1.91, 10]` | `1.20×` |
| `(500, 2.5)` | Stiff (`20.0`) | `0` (`+0.988`) | `17.33×` | `29.29×` `[2]` | `1.26×` `[1.26 / 1.18, 16]` | `1.06×` |
| `(500, 4.0)` | **SM (`1.556`)** / Stiff (`20`) | `1 (-1.210)` / `0 (+0.997)` | `2.78` / `1.26` | `4.70` / `2.47` | **`1.21`** / `1.11` | **`1.04`** / `1.06` |
| `(500, 5.0)` | **SM (`1.556`)** / Stiff (`20`) | `0 (+0.581)` / `0 (+0.999)` | `1.28` / `1.38` | `2.90` / `1.76` | **`1.07`** / `1.07` | **`1.04`** / `1.06` |
| `(1000, 2.5)` | Pure YM (`—`) | `0` (`0.000`) | `507.69×` | `988.75×` `[0]` | `2.62×` `[2.62 / 1.49, 8]` | `1.22×` |
| `(1000, 2.5)` | **SM (`1.556`)** | **`3` (`-2.065`)** | `19.16×` | **`478.94×` `[0]`** | **`2.88×` `[1.80 / 2.88, 12]`** | **`1.41×`** |
| `(1000, 2.5)` | Interm (`4.0`) | `1` (`-1.202`) | `52.83×` | **`3186.14×` `[2]`** | `5.46×` `[5.46 / 2.37, 8]` | `1.33×` |
| `(1000, 2.5)` | Stiff (`20.0`) | `0` (`+0.982`) | `74.88×` | `81.28×` `[0]` | `2.53×` `[2.53 / 1.31, 8]` | `1.22×` |
| `(1000, 4.0)` | **SM (`1.556`)** / Stiff (`20`) | `2 (-1.423)` / `0 (+0.995)` | `17.79` / `4.74` | `33.92` / `10.31` | **`1.51`** / `3.15` | **`1.09`** / `1.13` |
| `(1000, 5.0)` | **SM (`1.556`)** / Stiff (`20`) | `1 (-1.268)` / `0 (+0.998)` | `2.83` / `1.25` | `9.49` / `5.18` | **`1.49`** / `1.90` | **`1.05`** / `1.06` |
| `(2500, 4.0)` | Pure YM (`—`) | `0` (`0.000`) | `732.33×` | `1066.78×` `[0]` | **`1788.71×`** `[1788.7 / 2.70, 8]` | `1.71×` |
| `(2500, 4.0)` | **SM (`1.556`)** | **`4` (`-1.843`)** | `46.21×` | **`144.78×` `[6]`** | **`4.10×` `[3.44 / 4.10, 12]`** | **`1.32×`** |
| `(2500, 4.0)` | Interm (`4.0`) | `1` (`-1.366`) | `28.78×` | `1301.86×` `[2]` | `9.96×` `[1.80 / 9.96, 12]` | `2.34×` |
| `(2500, 4.0)` | Stiff (`20.0`) | `0` (`+0.992`) | `44.77×` | `42.10×` `[0]` | `12.37×` `[12.37 / 1.39, 8]` | `1.96×` |
| `(2500, 5.0)` | **SM (`1.556`)** | `0` (`+0.327`) | `95.08×` | `69.71×` `[4]` | **`1.89×` `[1.89 / 1.45, 16]`** | **`1.23×`** |
| `(2500, 5.0)` | Interm (`4.0`) | `0` (`+0.495`) | `3.33×` | `12.03×` `[8]` | `16.52×` `[2.26 / 16.52, 10]` | `2.08×` |
| `(2500, 5.0)` | Stiff (`20.0`) | `0` (`+0.995`) | `8.41×` | `6.65×` `[0]` | `4.23×` `[4.23 / 1.27, 16]` | `1.68×` |

*Pure-YM-Matched Initial Energy Cross-Check (`t4b_05_audit_remediation.json`):* Holding initial pure-YM energy $E_{\text{YM},0}$ strictly equal to Pure YM (adding $+2 M_W^2$ to $E_{\text{tot}}$ for Higgs runs) gives: at `(500, 2.5)`, SM $n=12 = 1.43\times$ vs. Stiff $1.22\times$ (vs. Pure YM `12.13×`); at `(2500, 4.0)`, SM $n=12 = 9.45\times$ vs. Stiff $13.97\times$ (vs. Pure YM `1788.71×`). This confirms that the high-$E$ SM quenching of $n = 12$ is a **genuine dynamical mechanism**: as $\phi(\eta)$ rolls through zero and overshoots to $-1.84v\dots -2.84v$, it siphons gauge excursion $f_{\max}$ below $f_{\text{crit}}(12) = +61/11$ while transiently amplifying $\mu^2 \propto \phi^2$, whereas low-threshold $S^3$ modes ($n = 4$, $f_{\text{crit}} = 5/3$) remain deeply tachyonic during every $\phi \approx 0$ crossing.

---

## 2. Goal 2: Unified Real-Time Winding-$k$ ($k \in \{1, 7\}$) 3+1D $SU(2)_L$ + Full 4-Component Higgs + FLRW Solver

### 2.1 Exact Temporal-Gauge $k$-Equivariant Formulation & Current Conservation

Implemented in [t4b_02_realtime_equivariant_solver.py](file://open_problems/track4_dynamical_hosotani_rho2/t4b_02_realtime_equivariant_solver.py) ([t4b_02_realtime_verification.json](file://open_problems/track4_dynamical_hosotani_rho2/t4b_02_realtime_verification.json)). For arbitrary Hopf winding $k \in \{1, 7\}$, $\mathcal{A}_k^{I^*}(N_{\max})$ obeys $(X_3 A)^a_i - 2\epsilon_{3ij} A^a_j - 2k\epsilon_{3ab} A^b_i = 0$ (verified on $A_{\rho_2}$ at $k=7$ to `2.41e-13` and on homogeneous $A = \omega^a T_a$ at $k=1$ to `< 1e-14`). Real-time temporal-gauge ($A_0 \equiv 0$) covariant current conservation $D_\mu J^\mu = 0$ requires the **full 4-real-component $k$-equivariant complex Higgs doublet** $\hat{\Phi} = (\hat{\Phi}_1, \hat{\Phi}_2)^\top$ ($X_3 \hat{\Phi}_1 = +2ik\hat{\Phi}_1$, $X_3 \hat{\Phi}_2 = 0$), reducing all spatial integrals **exactly** to quartic-exact 2D $S^2$ Hopf quadrature:
- **$k = 7$ sector:** Contains $\rho_0$, $S_{02}$, $\rho_2$, the exact $n=7$ Higgs ground state $\hat{\Phi}_0$, and all $k = 7$ Goldstone modes (`N_max = 12, 20, 24, 30, 36`: `(D_A, D_Phi, D_xi) = (6,4,2), (15,8,5), (24,12,8), (33,16,11), (51,24,17)`).
- **$k = 1$ sector:** Contains $\rho_0$, $\rho_1$, the homogeneous bounce $(f(\eta), \phi(\eta))$, homogeneous $n=0, K=2$ triaxial modes, and all $m_K = 0$ nonlinear inhomogeneous harmonics ($n \ge 12$, including $n = 12, K = 4$): `N_max = 12, 20, 24, 30` give `(D_A, D_Phi) = (12, 6), (21, 10), (30, 14), (39, 18)`.

In conformal time $\eta$, dividing the minisuperspace action by $V_0 = \pi^2/60$:
$$\tilde{L}_{\text{tot}} = -3(a')^2 + 3a^2 - 3a^4 + \kappa_{\text{YM}}\left[\frac{1}{2}\|c'\|^2 - \frac{1}{2}\langle|B|^2\rangle + 2a^2 M_W^2\left(\|\phi'\|^2 - \langle|D_A\hat{\Phi}|^2\rangle\right) - \frac{1}{2}\beta_H^2 a^4 M_W^4\langle(|\hat{\Phi}|^2 - 1)^2\rangle\right],$$
yielding exact Friedmann conservation $3(a'^2 + a^2 - a^4) = \kappa_{\text{YM}} E_{\text{tot}}$ and exact temporal-gauge Gauss constraint $\mathcal{G}^a(x) = (D_i E_i)^a + 4a^2 M_W^2 \operatorname{Re}((\partial_\eta\hat{\Phi})^\dagger T_a \hat{\Phi}) = 0$. Because constant $\xi^3$ ($n = 0$) is an exact finite-dimensional symmetry of the truncated $N_{\max}$ Hamiltonian, the global charge $\mathcal{G}_{n=0}$ is conserved to machine precision (`~1e-14–1e-22`) at every $N_{\max}$. By contrast, for finite-dimensional Galerkin truncations, testing $\mathcal{G}$ against gauge parameters $\xi^{(n_\xi)}$ all the way up to $n_\xi = N_{\max}$ suffers $O(1)$ cutoff aliasing whenever $D_A \xi$ scatters above $N_{\max}$ (observed directly in `t4b_02_realtime_verification.json` when starting from unrelaxed raw projected $c_{\rho_2}$: `max_gauss_l2_rel = 6.52, 2.32, 5.01` at $N=20, 24, 30$); proper continuum Gauss convergence is verified by (i) initializing from the **variationally relaxed flat representative $c_{\rho_2,\text{flat}}^{(N_{\max})}$** and (ii) tracking fixed low-harmonic projections $\|\mathcal{G}_{n \le 20}\|_2$ and full relative residual $\mathcal{G}_{\text{rel}}$ across $N_{\max} = 20 \to 36$.

### 2.2 Goal 2 Verification & Non-Circular Kill-Test 2A (`t4b_02_realtime_verification.json`)

- **Orthonormality, Analytical Gradients, & Rayleigh Limit:** Basis Gram errors `< 1e-12`, centered FD gradient checks `< 4.3e-8` (`fd_err_Vpot_phi <= 5.4e-11`), and the $k=7$ Higgs Rayleigh quotient at $A_{\rho_2}$ converges to exact `63.0` (`63.0493 -> 63.0010 -> 63.00016` across $N_{\max} = 12 \to 24$).
- **Kill-Test 2A (Nonlinear $k=1$ PDE vs. Goal 1 Linear ODE):** Evolving a `1e-7` $n=12, K=4$ seed (`eig(f=+1) = 99.9999999998` vs. `100.0`) through the SM Hosotani-restoring bounce ($E=500, M_W=2.5, \beta_H=1.556$, `3` $\phi$-crossings) inside the full nonlinear 3+1D $k=1$ PDE reproduces Goal 1's linear ODE to **`rel_end_err = 1.832e-8`** (`peak = 1.893360` vs. `1.893477`, `H_res = 1.53e-15`, `G_n0 = 2.61e-22`).

---

## 3. Goal 3: Three Core Nonlinear Dynamical Phenomena & Audited Convergence Kill-Tests

Implemented in [t4b_03_dynamical_phenomena.py](file://open_problems/track4_dynamical_hosotani_rho2/t4b_03_dynamical_phenomena.py), [t4b_03b_avalanche_bifurcations.py](file://open_problems/track4_dynamical_hosotani_rho2/t4b_03b_avalanche_bifurcations.py), and [t4b_05_audit_remediation.py](file://open_problems/track4_dynamical_hosotani_rho2/t4b_05_audit_remediation.py).

### 3.1 Phenomenon 3(a): Nonlinear $n = 12$ Bounce Stability During Dynamical SM Hosotani Restoration (`k=1, N_max=20`)

Seeding the full nonlinear $k=1$ (`D_A = 21, D_Phi = 10`) bounce ($\gamma_{\text{YM}} = 0.10$) with large finite $n=12, K=4$ amplitudes $\epsilon_{12} \in \{10^{-4}, 10^{-2}, 10^{-1}\}$ confirms Goal 1 nonlinearly (every run bounces cleanly with `max_H_res <= 2.2e-11`):

| Regime ($k=1, N_{\max}=20, \epsilon_{12} = 0.1$) | $a_{\min}$ | $\phi$-crossings | Nonlinear $q_{12}$ Gain | Peak $f_{E, n \ge 12}$ | $\max \delta_{T_{00}}^{\text{matter}}$ | $\max \delta_{T_{00}}^{\text{tot}}$ | $\max |\mathcal{H}_{\text{res}}|$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $E=500$, Pure YM ($M_W=0$) | `0.9419` | `0` | `10.57×` | `14.98%` | `7.80%` | `0.08%` | `8.86e-12` |
| $E=500$, Stiff Higgs ($M_W=2.5, \beta_H=20$) | `0.9594` | `0` | `1.93×` | `0.58%` | `0.23%` | `0.01%` | `8.17e-15` |
| **$E=500$, SM Hosotani ($M_W=2.5, \beta_H=1.556$)** | **`0.9471`** | **`3`** | **`1.91×`** | **`1.22%`** | **`0.15%`** | **`0.01%`** | **`2.45e-12`** |
| $E=2500$, Pure YM ($M_W=0$) | `0.9420` | `0` | `32.47×` | `14.56%` | `3.97%` | `0.01%` | `5.99e-12` |
| $E=2500$, Stiff Higgs ($M_W=4.0, \beta_H=20$) | `0.9592` | `0` | `14.51×` | `18.91%` | `3.78%` | `0.06%` | `2.51e-15` |
| **$E=2500$, SM Hosotani ($M_W=4.0, \beta_H=1.556$)** | **`0.9507`** | **`4`** | **`3.83×`** | **`1.75%`** | **`0.19%`** | **`0.02%`** | **`9.04e-13`** |

### 3.2 Phenomenon 3(b): Real-Time Cosmological $\rho_2 \to S_{02} \to \rho_0$ Avalanche Decay (`k=7, N_max=20..36`)

Initializing from the variationally relaxed flat representative $c_{\rho_2,\text{flat}}^{(N_{\max})}$ ($E_{\text{YM}} = 2.38 \times 10^{-6}$ at $N_{\max}=24$, $8.70 \times 10^{-8}$ at $N_{\max}=36$; $N_{\text{CS}} = 0.4083333333 = \frac{49}{120}$ to 10 digits) at Standard-Model ratio $\beta_H = 1.556$:

1. **Super-Horizon Expanding Avalanche & Coherent Rebound ($M_W = 5.0$, $a_{\text{restore}} = 1.4428 > 1$, $\kappa_{\text{YM}} = 2\times 10^{-4}$, $a_0 = 1.10 \to 25.0$):**
   - For $a < a_{\text{restore}}$, $\hat{\Phi}$ damps cleanly (`0.050 -> 0.0067` at $a = a_{\text{restore}}$, coasting to `0.00049` at $a = 1.531$) while $N_{\text{CS}} \equiv \frac{49}{120}$ and $E_{\text{YM}} \approx 0$.
   - Past $a_{\text{restore}}$, tachyonic condensation supercools until $a \approx 4.34$ ($\eta = 0.86$), where $E_{\text{YM}}$ crosses $S_{02}$ (`34.3` vs. `31.49`). Between $a = 4.34$ and $a = 7.33$, $N_{\text{CS}}$ plunges through $S_{02}$ (`0.2406`, $a_{\text{mid}} = 6.34$) to a first trough **`+0.00047`** (`+0.00058` at $N_{\max}=36$), **coherently rebounds to `+0.4929` at $a = 9.56$** (`+0.4919` at $a = 9.46, N_{\max}=36$ — a genuine continuum oscillation driven by undamped classical Higgs/YM energy exchange, not a settled vacuum!), bounces to `+0.1157` at $a = 18.31$, and passes through `+0.00386` at the arbitrary endpoint $a = 25.0$ (where $E_{\text{YM}}$ reaches `29,207` while $E_{H,\text{pot}}$ is still oscillating). Throughout the entire run up to $a = 25.0$, `max_H_res <= 6.85e-15`, `max delta_T00_matter = 9.25%`, `max delta_T00_tot = 0.15%`, and at $N_{\max} = 36$, `G_rel(full) = 6.84e-4` (`0.068%`).

2. **Sub-Horizon False-Vacuum Bounce & WKB Supercooling Law ($M_W = 22.0$, $a_{\text{restore}} = 0.3279 < 1/\sqrt{2}$):**
   At $\kappa_{\text{YM}} = 1.15\times 10^{-4}$ ($a_{\text{bounce}} = 0.29024 < a_{\text{restore}} = 0.32791$, $\epsilon_\Phi = 0.05$), integrating $\omega_{\text{tach}} = \sqrt{63[(a/a_{\text{restore}})^2 - 1]}$ over false-vacuum expansion yields:
   - Bare WKB ($\sqrt{63}(a_{\text{bnc}}/a_{\text{rest}})[\operatorname{arcosh} R - \sqrt{1 - R^{-2}}] = \ln(1/\epsilon_\Phi)$): exact root **$R = 1.944$**;
   - Adding $1/a$ Hubble dilution of $\phi = \chi/a$ ($+\ln R$ on RHS): **$R = 2.212$**;
   - Exact de Sitter $d\eta/da = [a^2\sqrt{a_{\text{bnc}}^{-2} - a^{-2}}]^{-1}$ with Hubble dilution: **$R = 1.857$**;
   - Full nonlinear $k=7$ simulation (including pre-threshold damping from $a_0 = 0.305$ to $a_{\text{rest}}$ plus nonlinear gauge roll-off time): **$R_{\text{sim}} = 2.367$** ($a_{\text{av}} = 0.7762 > 1/\sqrt{2}$).
   Because $\gamma_{\text{rad}} \propto R^4$, dynamical supercooling boosts $\gamma_{\text{rad}}$ to `1.463 > 1/4`, driving **supercooled escape** across all 10 tested sub-horizon runs (`kappa in [1.15, 2.60]e-4`, `seed in [0.05, 0.85]`; `dT00_tot <= 9.92%`, `H_res <= 3.5e-12`). Sub-horizon expanding recollapse requires extreme scale separation $a_{\text{restore}} \lesssim (2 R^2)^{-1} \approx 0.09$.

3. **Pre-Bounce Contracting Avalanche, $\gamma_c = \frac{1}{4}$ Bifurcation, & Audited UV-Cascade Validity Boundary ($M_W = 5.0$, contracting from $a_0 = 4.5$):**
   When $\rho_2$ contracts from $a_0 = 4.5 > a_{\text{restore}} = 1.4428$, the avalanche fires **pre-bounce** ($a \approx 2.2$, releasing $E_{\text{YM,peak}} \in [11{,}530,\; 34{,}953]$ across the sweep). Auditing BOTH sides of the $\gamma_c = \frac{1}{4}$ bifurcation across $N_{\max} = 24 \to 30 \to 36$ ([t4b_05_audit_remediation.json](file://open_problems/track4_dynamical_hosotani_rho2/t4b_05_audit_remediation.json)) cleanly separates **converged macroscopic cosmology** from **UV-cutoff-sensitive late-time turbulence**:

| Run ($M_W=5.0, a_0=4.5$) | $N_{\max}$ | Outcome | $a_{\min}$ | $\gamma_{\text{tot}}(1.5)$ / $\gamma_{\text{tot}}(1.0)$ | Pre-$a=1.5$ $\mathcal{G}_{\text{rel}}$ / $\|\mathcal{G}_{n \le 20}\|_2$ | $\mathcal{G}_{\text{rel}} > 0.1$ Onset ($a, \eta$) | Late-Time $\max\mathcal{G}_{\text{rel}}$ / $N_{\text{CS}}$ Range *(UV-sensitive)* |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| $\kappa = 5\times 10^{-6}$ | `24` | **Bounce** | `0.91872` | `0.1248` / `0.1315` | `8.76e-3` / `2.19e+0` | `a=1.143, eta=0.817` | `8.55` / `[-0.18, +8.26]` |
| $\kappa = 5\times 10^{-6}$ | `30` | **Bounce** | `0.91870` | `0.1249` / `0.1315` | `4.21e-3` / `3.92e-1` | `a=1.143, eta=0.817` | `7.22` / `[-0.23, +11.86]` |
| **$\kappa = 5\times 10^{-6}$** | **`36`** | **Bounce** | **`0.91869`** | **`0.1248` / `0.1315`** | **`6.80e-4` / `8.83e-4`** | **`a=1.090, eta=0.897`** | **`4.04` / `[-0.22, +8.86]`** |
| $\kappa = 1\times 10^{-5}$ | `24` | **Crunch** | `0.05000` | `0.2496` / `0.2621` | `9.05e-3` / `2.10e+0` | `a=1.127, eta=0.817` | `11.17` / `[-1.41, +5.72]` |
| $\kappa = 1\times 10^{-5}$ | `30` | **Crunch** | `0.05000` | `0.2496` / `0.2621` | `4.08e-3` / `3.70e-1` | `a=1.127, eta=0.817` | `23.83` / `[-1.32, +5.40]` |
| **$\kappa = 1\times 10^{-5}$** | **`36`** | **Crunch** | **`0.05000`** | **`0.2496` / `0.2622`** | **`6.94e-4` / `8.91e-4`** | **`a=1.059, eta=0.913`** | **`11.84` / `[-0.92, +6.48]`** |

**What IS converged vs. what is NOT:** Through the entire pre-bounce avalanche conversion ($a: 4.5 \to 1.5$), local Gauss conservation improves by **`2,480×`** (`G_n20: 2.19 -> 8.83e-4`, `G_rel = 6.8e-4`), and the macroscopic $\gamma_c = \frac{1}{4}$ bounce-vs.-crunch split (`a_min = 0.91869` at $\gamma_{\text{tot}} = 0.1315$ vs. singular collapse at $\gamma_{\text{tot}} = 0.2622$) is converged to 4–5 significant digits (also checked across $\kappa \in [2\times 10^{-7}, 10^{-5}]$ with `H_res <= 1.86e-12`). However, once the hot $E_{\text{YM}} \sim 10^4$ plasma compresses past $a \lesssim 1.09$ ($\eta \gtrsim 0.90$ at $N_{\max}=36$), classical UV turbulence hits the Galerkin cutoff and triggers $O(4\text{–}12)$ local Gauss leakage—so late-time $N_{\text{CS}}$ excursions and late-time contracting $\delta_{T_{00}}$ values beyond $a \lesssim 1.09$ are **not continuum-converged** and are shaded accordingly in Panel D.

### 3.3 Phenomenon 3(c): Nonlinear Mode-Mode Saturation of $n = 12$ in High-$E$ Pure-YM Plasma (`k=1, N_max=24`)

In pure 4D Yang–Mills ($M_W = 0$, `D_A = 30`), exact conformal invariance ($T^\mu{}_\mu \equiv 0$ pointwise) conserves $\gamma_{\text{YM}}$ identically across nonlinear saturation ($E_{\text{YM}} \in \{2500, 2843.9\}$, $\gamma_{\text{YM}} = 0.10$, $a: 2.0 \to a_{\min} \to 6.0$):

| $(E_{\text{YM}}, \epsilon_{12})$ | Bounced? | $a_{\min}$ (`exact hom`) | $q_{12}$ Gain | Final $f_{E, n \ge 12}$ | $\max \delta_{T_{00}}^{\text{matter}}$ | $\max \delta_{T_{00}}^{\text{tot}}$ | $\gamma_{\text{YM}}$ Rel. Drift |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `(2500.0, 1e-5)` | Yes | `0.94197` (`0.94197`) | `55.79×` | `0.00%` | `0.00%` | `0.00%` | `6.25e-15` |
| `(2500.0, 1e-2)` | Yes | `0.94197` (`0.94196`) | `53.06×` | `0.37%` | `0.07%` | `0.00%` | `1.05e-14` |
| `(2500.0, 1e-1)` | Yes | `0.94195` (`0.94195`) | `32.47×` | `14.56%` | `3.96%` | `0.01%` | `7.77e-15` |
| `(2843.9, 1e-5)` | Yes | `0.94197` (`0.94197`) | `150.07×` | `0.00%` | `0.00%` | `0.00%` | `5.55e-15` |
| `(2843.9, 5e-2)` | Yes | `0.94196` (`0.94196`) | `93.57×` | `67.17%` | `8.61%` | `0.03%` | `7.63e-15` |
| **`(2843.9, 1e-1)`** | **Yes** | **`0.94195` (`0.94195`)** | **`58.24×`** | **`79.64%`** | **`24.96%`** | **`0.06%`** | **`4.44e-15`** |

In the long-time fixed-$a$ plasma run ($E_{\text{YM}} = 2500, \epsilon_{12} = 10^{-3}, \eta \in [0, 8.0]$, energy drift `3.21e-14`), exponential linear growth saturates near $\eta \approx 4.5$ into broadband Molien turbulence: the strictly positive **non-clamped kinetic inhomogeneous share** $f_{\text{kin}, n \ge 12} = 1 - f_{\text{kin}, n=0}$ averages **`74.06%`** over $\eta \in [4.5, 8.0]$ (`n=0: 4.87%, n=12: 36.12%, n=20: 27.03%, n=24: 31.98%` at $\eta = 8$; mean `dT00_mat = 32.40%`). *(Note: the naive subtraction $1 - E_{\text{hom}}/E_{\text{tot}}$ transiently dips negative during turbulence because cubic/quadratic magnetic cross-terms with the tachyonic $n=12$ mode can be negative; using $f_{\text{kin}, n \ge 12}$ eliminates that clamping artifact cleanly.)*

### 3.4 Audited Convergence (`N_max` Sweeps) & Integrator (`DOP853` vs. `Radau`) Kill-Tests

1. **Dense-Sampled (`400` pts) $k = 7$ Expanding Avalanche $N_{\max} = 20 \to 36$ Convergence** (`t4b_05_audit_remediation.json`):

| $N_{\max}$ | $(D_A, D_\Phi)$ | $E_{\text{YM}}^{\text{flat}}$ | First Trough $N_{\text{CS}}$ ($a$) | Rebound $N_{\text{CS}}$ ($a$) | End $N_{\text{CS}}$ | $\max |\mathcal{H}_{\text{res}}|$ | $\|\mathcal{G}_{n \le 20}\|_2$ ($a \le 8$) | $\mathcal{G}_{\text{rel}}$ (full $a \le 25$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `20` | `(15, 8)` | `2.93e-2` | `+0.00104` (`7.23`) | `+0.4904` (`9.43`) | `+0.00386` | `6.85e-15` | `1.66e+00` | `7.33e-3` |
| `24` | `(24, 12)` | `2.38e-6` | `+0.00047` (`7.33`) | `+0.4929` (`9.56`) | `+0.00387` | `5.26e-15` | `5.29e-01` | `2.94e-3` |
| `30` | `(33, 16)` | `1.80e-7` | `+0.00043` (`7.32`) | `+0.4929` (`9.54`) | `+0.00387` | `5.16e-15` | `5.36e-02` | `5.47e-3` |
| **`36`** | **`(51, 24)`** | **`8.70e-8`** | **`+0.00058` (`7.24`)** | **`+0.4919` (`9.46`)** | **`+0.00386`** | **`6.06e-15`** | **`1.82e-03`** | **`6.84e-4`** |

2. **Canonical-Sign $k = 1$ Nonlinear $n = 12$ Saturation $N_{\max} = 12 \to 30$ Convergence** (`t4b_05_audit_remediation.json`, fixing $v_{k4}$ sign via the basis-independent cubic invariant $E_{\text{YM}}(c_{f=1} + 0.1 v_{k4}) > E_{\text{YM}}(c_{f=1} - 0.1 v_{k4})$):
   - $+v_{k4}$ seed ($\epsilon = +10^{-2}$): `fE_end = 0.003739 -> 0.003737 -> 0.003737 -> 0.003737`, `fkin_end = 0.001653 -> 0.001655 -> 0.001655 -> 0.001655`, `eta_end = 2.7587355838` (6-digit convergence across $N_{\max} = 12, 20, 24, 30$!).
   - $-v_{k4}$ seed ($\epsilon = -10^{-2}$): `fE_end = 0.004258 -> 0.004256 -> 0.004256 -> 0.004256`, `fkin_end = 0.002377 -> 0.002380 -> 0.002380 -> 0.002380`, `eta_end = 2.7587355816` (exact cubic asymmetry between $\pm v_{k4}$ cleanly resolved!).
3. **Independent `DOP853` vs. `Radau` Integrator Cross-Checks:**
   - **Post-avalanche $k = 7$ full decay** ($N_{\max}=20$, $\eta = 1.0549$, $a = 25.00$, $N_{\text{CS}} = +0.0039$): **`rel_state_diff = 2.386e-13`**, **`|dN_CS| = 3.561e-15`** (`Radau H_res = 3.05e-15`).
   - **$k = 1$ nonlinear $n = 12$ saturation bounce** ($N_{\max}=20$, $\eta = 2.00$): **`rel_state_diff = 1.290e-11`**, **`|da_min| = 8.229e-7`** (`Radau H_res = 1.03e-13`).

---

## 4. Honest Physical Scope Boundaries & Metric-Inhomogeneity Audit

1. **Spatially Averaged FLRW Backreaction Quantified via $\delta_{T_{00}}$ Across $S^2/I$:** Direct pointwise monitoring of $T_{00}(x)$ on $S^2/I$ bounds the minisuperspace FLRW approximation:
   - *Well-controlled (`< 1%` total):* Nonlinear SM Hosotani bounces (`dT00_tot <= 0.02%`), high-$E$ pure-YM $n=12$ saturation bounces (`dT00_tot <= 0.06%`), and super-horizon expanding $\rho_2$ avalanches (`dT00_tot <= 0.15%`, `dT00_mat <= 9.25%`).
   - *Moderately perturbed (`1%–10%` total):* Sub-horizon supercooled avalanches (`dT00_tot <= 9.92%`) and pre-bounce contracting avalanches prior to UV cascade (`dT00_tot <= 0.5%` for $a \ge 1.5$).
   - *Breakdown / UV-cutoff regime:* Late-time fixed-$a$ pure-YM turbulence (`dT00_mat = 32.4%`) and contracting post-avalanche plasma below $a \lesssim 1.09$ (where classical UV cascade triggers $O(4\text{–}12)$ local Gauss leakage at $N_{\max} \le 36$), requiring full 3+1D inhomogeneous numerical relativity and higher UV cutoffs.
2. **Inhomogeneous Metric Perturbations in Goal 1:** While Kill-Test 1B includes exact TT-graviton coupling at $n = 0, K = 2$, Goal 1's 11-configuration dynamical SM survey solves the Yang–Mills + Goldstone + radial-Higgs system on the dynamical FLRW background without coupled $n \ge 12$ scalar, vector, or TT metric perturbations (whose $O(\gamma_{\text{YM}})$ shift in $K = j \pm 2$ was bounded at `-1.2%` to `+2.4%` for $\gamma_{\text{YM}} = 0.10$ in P02).
3. **$U(1)_R$-Equivariant Submanifolds ($k = 1, 7$) & Magnetic Helicity Mixing in $N_{\text{CS}}$:** By Palais's Principle of Symmetric Criticality, $k = 1$ and $k = 7$ are exact consistent nonlinear submanifolds, but secondary non-axisymmetric ($m_K \neq 0$ or $k$-breaking) cascades during hot post-avalanche plasma evolution remain untested nonlinearly. Furthermore, during hot plasma oscillations with large $|B|$, $N_{\text{CS}}[A]$ mixes dynamical magnetic helicity with topological vacuum winding and cannot be interpreted strictly as vacuum-to-vacuum tunneling without cooling/gradient-flow projection onto flat connections.

---

## 5. Reproducibility Manifest

All scripts, JSON outputs, and figures reside strictly inside `open_problems/track4_dynamical_hosotani_rho2/` and run deterministically under `.venv/bin/python`:
- Goal 1: [t4b_01_goldstone_covariant.py](file://open_problems/track4_dynamical_hosotani_rho2/t4b_01_goldstone_covariant.py) · [t4b_01_goldstone_covariant.json](file://open_problems/track4_dynamical_hosotani_rho2/t4b_01_goldstone_covariant.json)
- Goal 2: [t4b_02_realtime_equivariant_solver.py](file://open_problems/track4_dynamical_hosotani_rho2/t4b_02_realtime_equivariant_solver.py) · [t4b_02_realtime_verification.json](file://open_problems/track4_dynamical_hosotani_rho2/t4b_02_realtime_verification.json)
- Goal 3: [t4b_03_dynamical_phenomena.py](file://open_problems/track4_dynamical_hosotani_rho2/t4b_03_dynamical_phenomena.py) · [t4b_03_dynamical_phenomena.json](file://open_problems/track4_dynamical_hosotani_rho2/t4b_03_dynamical_phenomena.json) · [t4b_03b_avalanche_bifurcations.py](file://open_problems/track4_dynamical_hosotani_rho2/t4b_03b_avalanche_bifurcations.py) · [t4b_03b_avalanche_bifurcations.json](file://open_problems/track4_dynamical_hosotani_rho2/t4b_03b_avalanche_bifurcations.json)
- Cold-Audit Remediation & Verification: [t4b_05_audit_remediation.py](file://open_problems/track4_dynamical_hosotani_rho2/t4b_05_audit_remediation.py) · [t4b_05_audit_remediation.json](file://open_problems/track4_dynamical_hosotani_rho2/t4b_05_audit_remediation.json)
- Diagnostic Summary Figure: [t4b_04_plot_summary.py](file://open_problems/track4_dynamical_hosotani_rho2/t4b_04_plot_summary.py) · [t4b_dynamical_hosotani_rho2_summary.png](file://open_problems/track4_dynamical_hosotani_rho2/t4b_dynamical_hosotani_rho2_summary.png)
