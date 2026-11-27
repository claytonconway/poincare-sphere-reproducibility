# Poincare Homology Sphere Gauge Cosmology (`S³/I*`) : Open Reproducibility Archive (`v2.1.0`)

[![Smoke Tests](https://github.com/claytonconway/poincare-sphere-reproducibility/actions/workflows/smoke.yml/badge.svg)](https://github.com/claytonconway/poincare-sphere-reproducibility/actions/workflows/smoke.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24561720.svg)](https://doi.org/10.5281/zenodo.24561720)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This self-contained public repository contains the complete verification code, exact symbolic and 50-digit numerical oracles, machine-readable JSON datasets, permanent Claim Ledger, and seven standalone technical reports (Tracks I--VI) exploring classical general relativity, non-Abelian `SU(2)_L` Yang-Mills-Higgs dynamics, pure-graviton 1-loop spectral geometry, nonlocal semiclassical gravity, and nonlinear dynamical-systems backreaction on the Poincare homology sphere `S³/I*` and spherical space forms `S³/Γ`.

---

## Transparency & Verification Standard

This archive was produced through a **human-directed, AI-assisted symbolic and numerical exploration workflow** designed so that every mathematical identity and numerical claim is independently self-verifying rather than relying on credentials or authority:

- Every analytical identity and numerical table is backed by executable Python / NumPy / SciPy / 50-digit `mpmath` scripts.
- Dynamical solvers cross-check independent integrators (`DOP853` vs. `Radau`) and log un-clamped conservation, Bianchi, Ward, and Gauss residuals directly into `results.json`.
- Explicit EFT, unitarity, and Dvali species-scale validity boundaries are disclosed throughout.
- The [Claim Ledger](ledger/CLAIMS.md) (`ledger/claims.json`) permanently preserves **retracted / falsified dead ends (`RETRACTED`)** alongside verified theorems so future researchers never repeat falsified paths.

Independent verification, bug reports, and reuse of any calculator or dataset are welcomed.

---

## Quick Start (< 60 s Verification)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Automated end-to-end smoke suite across ALL 7 tracks, Claim Ledger, and PDFs (< 2 s):
python3 run_smoke_tests.py

# Regenerate all 12 Phase 1 (Tracks I-III) publication figures:
python3 generate_all_figures.py --out-dir ./generated_figures
```

---

## Complete Archive Map (Tracks I--VI)

| Track | Scope & Headline Verified Results | Technical Report | Verification Directory |
| :--- | :--- | :--- | :--- |
| **Track I** | Classical rotating triaxial Bianchi IX bounces, pure-null `SU(2)_L` Gauss ellipsoid theorem (`u_0* = 8*sqrt(3)/27`), and spherical / `I*`-faceted Israel-Darmois thin-shell horizons | [track1_grqc_bianchi9_thin_shells.pdf](reports/track1_grqc_bianchi9_thin_shells.pdf) (17 pp) | `track1/` |
| **Track II** | Exact `Z[√5]` degree-7 map `P_7` (`N_CS = +49/120`), 315-DOF Morse-index-1 sphaleron `S_02`, `280.08x` `K = j ± 2` gap stabilization, and exact rational twisted YM Casimir jump `+2 ℏc/a` | [track2_hepth_su2_galois_poincare.pdf](reports/track2_hepth_su2_galois_poincare.pdf) (14 pp) | `track2/` |
| **Track III** | Chiral Maxwell spectrum on `S³/I*`, Starobinsky-Casimir KAM breathing island, `N`-invariant Species-Scale Squeeze, Berger-sphere collapse, and covariant Riegert-Weyl spin-2 stability | [track3_prd_semiclassical_poincare.pdf](reports/track3_prd_semiclassical_poincare.pdf) (19 pp) | `track3/` |
| **Track IV-A** | Exact discrete `S³/I*` 2-photon spectral pole sum (Wigner `6j`, Bach factorization, digamma kernels), exact shell-by-shell Maxwell Contact Ward Identity (`-178952/93`), half-antipodal `Δη = π/2` Talbot revivals, and 2-graviton selection-rule desert | [track4a_grqc_nonlocal_weyl_kam.pdf](reports/track4a_grqc_nonlocal_weyl_kam.pdf) (24 pp) | `track4a/` (`t4a/`, `p2_01/`, `p2_06/`) |
| **Track IV-B** | Goldstone-covariant fluctuation theory regular at `φ = 0` (`V*C + C'' = 0`), dynamical SM Hosotani symmetry restoration, and real-time `(3+1)`D `k = 7` `ρ_2 -> S_02 -> ρ_0` avalanches | [track4b_hepth_dynamical_hosotani_rho2.pdf](reports/track4b_hepth_dynamical_hosotani_rho2.pdf) (16 pp) | `track4b/` |
| **Track V** | 50-digit pure-graviton Casimir tables across ADE / `L(p,1)` / `S³/D_m*`, exact Berger slope theorem, universal conformal chiral asymmetry `+(3/4) * order(Γ)`, uncapped 28-mode global focusing, topological prolate stabilization, and Witten-anomaly-free twisted-Dirac Hosotani inversion | [track5_hepth_pure_graviton_spaceforms.pdf](reports/track5_hepth_pure_graviton_spaceforms.pdf) (23 pp) | `track5/` (`p2_03/`, `p2_04/`, `p2_05/`) |
| **Track VI** | Formal large-`N_V` nonlinear KAM dynamical-systems study: Complete Kubo-plus-contact shifted `n=2` Mathieu wall (`σ̃ ∈ [8.75, 10.00]`), narrowed survival sub-desert `[4.5, 5.9]`, chiral self-quenching rescue above `x_kin`, coherent-Talbot-to-Markovian `γ_deph` crossover, and autonomous Misner `M_R`-splitting Talbot echo scrambler | [track6_grqc_nonlinear_kam_dynamical_systems.pdf](reports/track6_grqc_nonlinear_kam_dynamical_systems.pdf) (23 pp) | `track6/` (`p2_02/`, `p2_07/`) |

---

## Shared Mathematical Conventions

- Metric signature `(-,+,+,+)`; reduced Planck units `8*pi*G = M_Pl^-2 = 1`, `c = 1`, `Lambda = 3` (`a_dS = 1`) unless restored explicitly.
- Left-invariant orthonormal coframe on `S³/I*`: `d(omega^a) = -eps_abc omega^b ^ omega^c`, `[X_a, X_b] = 2 eps_abc X_c`, `Vol(S³/I*, a) = pi^2 a^3 / 60`, `Scal = 6/a^2`, `r_inj = pi*a/10`, `L_min = pi*a/5`.
- `SU(2)_L` generators `T_a = -i*sigma_a/2` (`[T_a, T_b] = eps_abc T_c`), homogeneous connection `A^a = (1 + f_a) omega^a`, Chern-Simons normalization `+(1/8pi^2)` yielding `N_CS(rho_0) = 0`, `N_CS(rho_1) = +1/120`, `N_CS(rho_2) = +49/120`.

---

## Directory Layout

```text
.
├── .zenodo.json                 # Zenodo v2.1.0 archival metadata (ORCID + MIT)
├── CITATION.cff                 # GitHub / Zenodo citation metadata
├── LICENSE                      # MIT License (code)
├── README.md                    # This guide
├── requirements.txt             # Portable Python 3.9-3.14 dependencies
├── requirements-py39-lock.txt   # Exact Python 3.9.6 pinned lockfile
├── run_smoke_tests.py           # < 2 s end-to-end invariant check across Tracks I-VI
├── generate_all_figures.py      # Regenerates all 12 Phase 1 figures (Tracks I-III)
├── ledger/                      # 24-claim ledger (CLAIMS.md + claims.json, incl. RETRACTED)
├── reports/                     # All 7 standalone technical reports (PDF)
├── track1/                      # Track I scripts, data/ & figure builder
├── track2/                      # Track II scripts, data/ & figure builder
├── track3/                      # Track III scripts, data/ & figure builder
├── track4a/                     # Track IV-A (t4a/, p2_01/, p2_06/) scripts, JSONs & reports
├── track4b/                     # Track IV-B (t4b_01..06) scripts, JSONs & report
├── track5/                      # Track V (p2_03/, p2_04/, p2_05/) scripts, JSONs & reports
└── track6/                      # Track VI (p2_02/, p2_07/) scripts, JSONs & reports
```

---

## License

Source code in this archive is released under the [MIT License](LICENSE). Reference numerical JSON datasets, Claim Ledger, and technical reports are provided under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
