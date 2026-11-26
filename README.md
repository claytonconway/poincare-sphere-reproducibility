# Reproducibility Package: Gauge Cosmology and Semiclassical Gravity on the Poincaré Homology Sphere $S^3/I^*$

This self-contained repository reproduces every theorem, numerical table, and publication figure across the three companion papers on classical, Yang–Mills–Higgs, and semiclassical cosmology on the Poincaré homology sphere $S^3/I^*$ and spherical space forms $S^3/\Gamma$:

1. **Track I (`track1/`)** — *Rotating Triaxial Bianchi IX Bounces, Pure-Null Gauge Obstructions, and Spherical/Faceted Thin-Shell Horizons on the Poincaré Homology Sphere $S^3/I^*$*
2. **Track II (`track2/`)** — *Fractional Chern–Simons Vacua, Galois-Conjugate Sphalerons, Spectral-Gap Stabilization, and Twisted 1-Loop Casimir Lifting of $SU(2)$ Yang–Mills–Higgs Theory on the Poincaré Homology Sphere $S^3/I^*$*
3. **Track III (`track3/`)** — *Chiral Maxwell Spectral Geometry, Self-Consistent $O(\hbar)$ Starobinsky–Casimir Cosmology, Anisotropic Berger-Sphere Collapse, and Covariant Riegert–Weyl Spin-2 Stability on Spherical Space Forms $S^3/\Gamma$*

---

## Quick Start

### 1. Environment Setup

Verified on **Python 3.9.6** and **Python 3.14.7** (compatible with Python 3.9–3.14). Install portable dependencies via `requirements.txt` (or use `requirements-py39-lock.txt` for the exact original Python 3.9.6 wheel pins):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Fast Automated Smoke Tests (< 60 s)

Run live algebraic, spectral, 4D Einstein/Yang–Mills constraint, Starobinsky–Casimir KAM ODE, Berger-sphere spectrum, and covariant Riegert symplectic Wronskian checks across all three tracks:

```bash
python run_smoke_tests.py
```

### 3. Regenerate All 12 Publication Figures

Regenerate all 5 Track I figures, 3 Track II figures, and 4 Track III figures from the shipped scripts and reference data into `./generated_figures/` (or any custom `--out-dir`):

```bash
python generate_all_figures.py --out-dir ./generated_figures
```

Individual track figure builders can also be invoked standalone:

```bash
python track1/build_figures.py --out-dir ./generated_figures/track1
python track2/build_figures.py --out-dir ./generated_figures/track2
python track3/build_figures.py --out-dir ./generated_figures/track3
```

---

## Shared Mathematical Conventions

All scripts enforce consistent geometric and physical conventions across all three tracks:

- Metric signature $(-,+,+,+)$; reduced Planck units $8\pi G = M_{\mathrm{Pl}}^{-2} = 1$, $c = 1$, $\Lambda = 3$ ($a_{\mathrm{dS}} = 1$) unless restored explicitly.
- Left-invariant orthonormal coframe on $S^3/I^*$ obeying $d\omega^a = -\epsilon_{abc}\,\omega^b \wedge \omega^c$ and $[X_a, X_b] = 2\epsilon_{abc} X_c$, with $\mathrm{Vol}(S^3/I^*, a) = \pi^2 a^3 / 60$, $\mathcal{R} = 6/a^2$, injectivity radius $r_{\mathrm{inj}} = \pi a / 10$, and shortest closed geodesic $L_{\min} = \pi a / 5$.
- $SU(2)_L$ generators $T_a = -i\sigma_a / 2$ ($[T_a, T_b] = \epsilon_{abc} T_c$), homogeneous connection $A^a = (1 + f_a)\,\omega^a$, $\kappa_{\mathrm{YM}} = g_{\mathrm{YM}}^{-2}$, $M_W = \frac{1}{2} g_{\mathrm{YM}} v$, $m_H = \sqrt{2\lambda_H}\,v$.

---

## Track I: Classical GR, Rotating Triaxial Bianchi IX, Tilted $SU(2)_L$, and Thin Shells (`track1/`)

### Theorem, Table & Figure Map

| Paper Element | Description | Exact Verification Command | Reference Data / Output |
| :--- | :--- | :--- | :--- |
| **Theorem 1** | Rank-2 isotropic no-go for classical $U(1)$ EM ($\|M - \frac{2}{3}u I\|_{\mathrm{op}} \ge \frac{2}{3}u$) | `python track1/r2_01_axisymmetric_rank2_arrest.py` | `track1/data/r2_01.json` |
| **Theorem 2** | Warm/cold canonical scalar bounce no-go | `python track1/r2_03_warm_scalar_receding_ceiling.py` | `track1/data/r2_03.json` |
| **Table 1 & Fig. 1** | Closed-form Einstein–Maxwell–$\Lambda$ bounce ceiling $\gamma_c = 1/4$ & elliptic resonance tower $\Delta\eta(\gamma_k) = k\pi$ | `python track1/explore_einstein_maxwell_lambda.py`<br>`python track1/build_figures.py` | `fig1_einstein_maxwell_lambda.png` |
| **Theorem 3** | Complete axisymmetric homogeneous null-wave no-go across all $g_{33}$ causal signatures | `python track1/verify_exact_system.py` | `track1/data/goal1_symbolic_verification.json` |
| **Theorem 4 & Table 2** | Universal pure null-dust bounce ceiling $u_0^* = 8\sqrt{3}/27 = 0.513200$, rotating Einstein-static fixed point, & birefringent Model I split ($0.5132$ vs. $64/81$) | `python track1/verify_synchronous_and_bounds.py`<br>`python track1/synchronous_frame_audit.py`<br>`python track1/deep_phase_and_killtests.py` | `track1/data/synchronous_and_bounds.json`<br>`track1/data/synchronous_frame_audit.json`<br>`track1/data/deep_phase_and_killtests.json` |
| **Fig. 2** | 6-panel tilted rotating triaxial Bianchi IX diagnostic summary | `python track1/build_figures.py` | `fig2_tilted_bianchi9.png` |
| **Theorem 5, Table 3 & Fig. 3** | Pure-Null Gauss Ellipsoid Theorem for homogeneous $SU(2)_L$ Yang–Mills, 0/400 pure-null bounces, & matched $U(1)_A \times U(1)_B$ control sweep | `python track1/r3_05_tilted_su2_bianchi9.py`<br>`python track1/build_figures.py` | `track1/data/r3_05_tilted_su2_bianchi9.json`<br>`fig3_su2_tilted_bianchi9.png` |
| **Theorems 6–8 & Fig. 4** | Spherical Israel–Darmois bounce WEC ceiling $\sin^3(\pi/10) = 0.0295085$ vs. $\pi/80$ (ratio $1.330800$), static/turning-point NEC no-go, non-barotropic pure-dS no-go, & radiation bifurcation | `python track1/verify_junction_and_eom.py`<br>`python track1/test_separatrix.py`<br>`python track1/killtests_and_benchmarks.py`<br>`python track1/build_figures.py` | `track1/data/junction_verification.json`<br>`track1/data/separatrix_scan.json`<br>`track1/data/killtests_and_benchmarks.json`<br>`fig4_israel_darmois.png` |
| **Table 4 & Fig. 5** | $I^*$-equivariant $l=6$ ($n=12$) faceted thin-shell 8,254-point $S^2$ ceilings & nonlinear Hawking-mass collapse ($m_H^- < 0$ for $\epsilon > 0.00667$) | `python track1/r3_04_faceted_shell.py`<br>`python track1/build_figures.py` | `track1/data/r3_04_faceted_shell.json`<br>`fig5_faceted_thin_shells.png` |

---

## Track II: Fractional Chern–Simons Vacua, Galois $\rho_2$, Sphaleron $S_{02}$, and Twisted 1-Loop Lifting (`track2/`)

### Theorem, Table & Figure Map

| Paper Element | Description | Exact Verification Command | Reference Data / Output |
| :--- | :--- | :--- | :--- |
| **Sec. 2** | Homogeneous Einstein–$SU(2)_L$–Higgs–$\Lambda$ cosmology, $N_{\mathrm{CS}} = (2+3f-f^3)/480$, stiff-Higgs $m_H = 2\sqrt{3} M_W$ vs. SM Hosotani restoration | `python track2/r2_05_su2_ym_sphaleron_bounce.py`<br>`python track2/thread_a_su2_higgs_bounce.py` | `track2/data/r2_05.json`<br>`track2/data/thread_a_su2_higgs.json` |
| **Theorem 1 & Fig. 1** | Closed-form extremal $K = j \pm 2$ perturbation law, $280.08\times$ $I^*$ stabilization ($E_{\mathrm{tach}} = 442.5927$ vs. $128/81$), sub-tachyonic Floquet tongues, & gravity backreaction | `python track2/p02_01_operator_verification.py`<br>`python track2/p02_02_floquet_instability.py`<br>`python track2/p02_03_bounce_and_gravity.py`<br>`python track2/build_figures.py` | `track2/data/p02_01_operator_verification.json`<br>`track2/data/p02_02_floquet_instability.json`<br>`track2/data/p02_03_bounce_and_gravity.json`<br>`fig1_p02_inhomogeneous_spectrum.png` |
| **Theorem 2** | Minimal degree-7 $\mathbb{Z}[\sqrt{5}]$ holomorphic intertwiner $P_7$, flat connection $A_{\rho_2}$ ($\|B\| = 1.15 \times 10^{-13}$, $N_{\mathrm{CS}} = 49/120$ to $6.11 \times 10^{-16}$) | `python track2/goal1_construct_rho2.py` | `track2/data/goal1_rho2_verification.json` |
| **Sec. 4 & Table 1** | Full-3D 315-DOF Morse-index-1 sphaleron $S_{02}$ ($E_{\mathrm{YM}} = 31.493447$, $N_{\mathrm{CS}} = 0.240632$), V-shaped Morse graph $\rho_1 \leftrightarrow S_{01} \leftrightarrow \rho_0 \leftrightarrow S_{02} \leftrightarrow \rho_2$ | `python track2/goal2_sphaleron_solver.py`<br>`python track2/goal2_tier2_full3d.py`<br>`python track2/goal2_inspect_and_flow.py`<br>`python track2/goal2_morse_and_gauge_audit.py` | `track2/data/goal2_tier1_k7.json`<br>`track2/data/goal2_tier2_full3d.json`<br>`track2/data/goal2_saddles.npz`<br>`track2/data/goal2_connectivity_and_anatomy.json`<br>`track2/data/goal2_morse_and_gauge_audit.json` |
| **Theorem 3, Table 1 & Fig. 2** | Exact $21\times$ covariant Higgs gap at $\rho_2$ ($\lambda_{\min} = 63/a^2$), stiff-Higgs saddle-node $\mu_{W,\mathrm{sn}}(\rho_2) = 1.996199$, tricritical $\beta_{H,\mathrm{tri}} = 2.47698$, & SM avalanche | `python track2/goal3_higgs_lifting.py`<br>`python track2/goal3_coupled_finite_beta.py`<br>`python track2/build_figures.py` | `track2/data/goal3_higgs_lifting.json`<br>`track2/data/goal3_coupled_phase_diagram.json`<br>`fig2_p03_galois_rho2_sphaleron.png` |
| **Theorem 4, Table 2 & Fig. 3** | 50-digit $\mathrm{Ad}(\rho_0,\rho_1,\rho_2)$-twisted chiral curl spectra, exact pure-rational pure-YM Casimir energies $(-349/225, -221/150, +79/150)$, integer jump $+2$, & APS spectral flow ($\mathrm{SF}_1 = 1$, $\mathrm{SF}_2 = 5$) | `python track2/r3_01_twisted_casimir_eta.py`<br>`python track2/build_figures.py` | `track2/data/r3_01_twisted_casimir_eta.json`<br>`fig3_r3_01_twisted_casimir_eta.png` |

---

## Track III: Chiral Maxwell Spectral Geometry, Starobinsky–Casimir Cosmology, Berger Collapse, and Riegert–Weyl Spin-2 Stability (`track3/`)

### Theorem, Table & Figure Map

| Paper Element | Description | Exact Verification Command | Reference Data / Output |
| :--- | :--- | :--- | :--- |
| **Theorem 1, Table 1 & Fig. 1A** | Chiral Maxwell curl spectrum on $S^3/I^*$, ADE Casimir energies ($E_{\mathrm{Cas}}^{\mathrm{EM}} = -349/675$), APS $\eta_{\mathrm{curl}}(0) = -361/180$, & exact palindromic identity $\Delta E_{\mathrm{Cas}} \equiv 0$ | `python track3/r2_04_poincare_beltrami_spectrum.py`<br>`python track3/thread_b_chiral_casimir_eta.py`<br>`python track3/build_figures.py` | `track3/data/r2_04.json`<br>`track3/data/thread_b_chiral_casimir.json`<br>`fig1_chiral_ade_casimir_eta.png` |
| **Table 2 & Fig. 1B** | $\mathrm{Ad}(\rho_i)$-twisted pure-$SU(2)$ YM lifting & KAM ring persistence ($\rho_0, \rho_1$) vs. destruction ($\rho_2$) | `python track2/r3_01_twisted_casimir_eta.py`<br>`python track3/build_figures.py` | `track2/data/r3_01_twisted_casimir_eta.json`<br>`fig1_chiral_ade_casimir_eta.png` |
| **Theorems 2–3, Table 3 & Fig. 2A–B** | 50-digit nonsingular 4-variable Starobinsky–Casimir ODE, $S^3$ recovery, pure-Euler bounce destruction, $S^3/I^*$ adiabatic KAM island ($x_{\mathrm{ring}} = 21.9168$, $\lambda_{\mathrm{KAM}} = 1.545 \times 10^{-4}$) | `python track3/p05_01_derivation_and_identities.py`<br>`python track3/p05_02_solve_and_phase_space.py`<br>`python track3/p05_02b_deep_dynamics.py`<br>`python track3/p05_02c_cyclic_and_euler.py`<br>`python track3/p05_03_killtests_and_figures.py`<br>`python track3/p05_04_critic_fixes_and_final_fig.py` | `track3/data/p05_01_identities.json`<br>`track3/data/p05_02_phase_space.json`<br>`track3/data/p05_02b_deep_dynamics.json`<br>`track3/data/p05_02c_cyclic_breathing.json`<br>`track3/data/p05_03_killtests.json`<br>`track3/data/p05_04_critic_fixes.json`<br>`fig2_starobinsky_casimir_kam.png` |
| **Theorems 4–5, Table 4 & Fig. 2C–D** | Exact $N$-invariant Species-Scale Squeeze on $S^3/I^*$ ($\le 0.8981 < 1$), spin-2 unitarity bound $[3.648, 18.850] > 1$, & isotropy vs. species-scale dichotomy across all 5 isotropy-irreducible $S^3/\Gamma$ vs. $L(p,1)$ ($p \ge 34$) and $S^3/D_m^*$ ($m \ge 17$) | `python track3/r3_02_ade_lens_species_escape.py`<br>`python track3/build_figures.py` | `track3/data/r3_02_ade_lens_species_escape.json`<br>`fig2_starobinsky_casimir_kam.png` |
| **Theorem 6, Table 5 & Fig. 3** | Squashed Berger-sphere Maxwell/scalar Casimir stress tensor, static oblate saddle-node barrier ($x_{\perp,\min} \approx \frac{9}{2}x_{\mathrm{ring}}$ at $\lambda_B^* \approx \sqrt{2/3}$), & universal runaway Hopf-fiber collapse ($a_3 \to 0$; 0/47 valid bounces across 160 trials) | `python track3/t3_01_berger_casimir_backreaction.py`<br>`python track3/build_figures.py` | `track3/data/t3_01_berger_casimir_backreaction.json`<br>`fig3_berger_anisotropic_collapse.png` |
| **Theorem 7, Table 6 & Fig. 4** | Covariant variational Riegert anomaly action (exact symplectic Wronskian drift $5.67 \times 10^{-13}$, eliminates local ${}^{(3)}H$ non-variational drift & $x_{\mathrm{ghost}}^-$) and 4th-order Ostrogradsky Weyl $C^2$ spin-2 stability map (resolved 11-point 2:1 Stelle–scalaron resonance tongue peak at $\beta_0 \approx 0.2890$, $1995\times$) | `python track3/r3_03_kam_bianchi9_shear_stability.py`<br>`python track3/t3_02_riegert_weyl_c2_spin2.py`<br>`python track3/build_figures.py` | `track3/data/r3_03_kam_bianchi9_shear_stability.json`<br>`track3/data/t3_02_riegert_weyl_c2_spin2.json`<br>`fig4_riegert_weyl_c2_spin2.png` |

---

## Directory Layout

```text
reproducibility/
├── .zenodo.json                 # Automated Zenodo archival metadata (ORCID + MIT)
├── CITATION.cff                 # GitHub / Zenodo citation metadata
├── LICENSE                      # MIT License (code)
├── README.md                    # This reproducibility guide
├── requirements.txt             # Portable Python 3.9-3.14 dependencies
├── requirements-py39-lock.txt   # Exact Python 3.9.6 pinned lockfile
├── run_smoke_tests.py           # < 60 s end-to-end invariant check across Tracks I-III
├── generate_all_figures.py      # Regenerates all 12 publication figures
├── track1/                      # Classical GR, Bianchi IX, SU(2)_L, Thin Shells + data/
├── track2/                      # SU(2) YMH, Inhomogeneous Spectrum, Galois rho_2, S_02 + data/
└── track3/                      # Chiral Maxwell, Starobinsky-Casimir, Berger, Riegert-Weyl + data/
```

## License

Source code in this archive is released under the [MIT License](LICENSE). Reference numerical JSON datasets and documentation are provided under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
