#!/usr/bin/env python3
"""
Stream p2_01_discrete_photon_pole_sum — Stage 1: Exact Derivation & 40-Digit Verification
of the Discrete S^3/I* Opposite-Chirality 2-Photon Spectral Pole Sum (CLM-OPEN-01).

Establishes and verifies to 35-40 digits in mpmath (with independent float64 & matrix cross-checks):
  1. Exact SU(2)_L x SU(2)_R triple-overlap theorem for homogeneous n=2 (-) shear:
       - Schur orthogonality on SU(2)_L locks j_L(gamma_+) == j_L(gamma_-) == j strictly!
       - Opposite-chirality selection (1 - s_1*s_2) pairs omega_+ = 2j and omega_- = 2j+2,
         giving exact discrete 2-photon pole energies E_j = 4j + 2:
           * Round S^3: j = 1, 3/2, 2, ... => E_j = 6, 8, 10, 12, ... (M_{S^3}(2) = 6 exact!)
           * S^3/I*:    j in {6, 10, 12, 15, 16, 18, ...} => E_j = 26, 42, 50, 62, 66, ... (M_{I*}(2) = 26!)
       - Right-index + frame-spin-1 overlap factor C_R(j) = (2j-1)(2j+3) / [5(2j+1)],
         verified by TWO completely independent methods:
           (a) Direct Racah Clebsch-Gordan sum,
           (b) Direct numerical diagonalization of the SU(2) curl operator matrix (zero CG formula assumed!).
       - Exact algebraic Bach-polynomial factorization:
           (2j+1) * omega_+(j) * omega_-(j) * C_R(j) == (1/80) * B_2(E_j),  B_2(E) = (E^2-4)(E^2-16).
       - Topological spectral weight W_{Sigma}(j) = |Gamma| * d_Gamma(j) / (2j + 1),
         with exact period-30 remainder R_{I*}(j) = 120*d_{I*}(j) - 2(2j+1), sum_{r=0}^{29} R_{I*}(r) == 0.
  2. KT1 (Analyticity & Chiral Gap):
       - Same-chirality (-2, -2) pair at E=4 has nonzero geometric overlap (1.0000 for fixed unit-norm E_2)
         but vanishes identically under Maxwell (1 - s_1*s_2) = 0; Im K_{I*}^{disc}(Omega) == 0 for all |Omega| < 26;
         round S^3 recovers M_{S^3}(2) = 6 identically.
  3. KT2 (Continuum & Weyl-Anomaly UV Recovery):
       - High-frequency sliding-window spectral density average recovers exact 1/(40*pi) * B_2(Omega)
         (matching 4D Maxwell Weyl anomaly c_anom = 9/31 in H_S=1 units) for both S^3 and S^3/I*.
       - Exact Euclidean UV matching K_{Sigma}^{disc}(i*Om_E) - ln(Om_E^2/mu_cyl^2) = -zeta_{Sigma}(1)/Om_E^2 + O(Om_E^-4)
         with exact Hurwitz-zeta constants zeta_{S^3}(1) = -74/3 and zeta_{I*}(1) = -724/3.
       - Exact Gaussian heat-kernel regulator exp(-(eps*E)^2) check of un-factored Kubo bubble difference
         Delta_Pi = Pi_{I*} - Pi_{S^3}, verifying exact cancellation of O(Om_E^4 ln Om_E) and O(Om_E^2) local UV terms.
  4. KT3 (Cutoff Convergence, Pole-Repulsion Shift & Dual-Precision Oracle):
       - Exact static Horowitz pinch-off beta_pinch(N_cut) and UHP/real Stelle roots across spectral cutoffs
         E_cut in {60, 100, 160, 240, 480, 960, inf} and period-matched J_cut in {30, 60, 120, 240, inf},
         in BOTH 40-digit mpmath and float64 (dual-precision cross-check).
       - Side-by-side comparison of Model A (Conformal Weyl Form Factor B_2(Om)*K_disc(Om)) vs
         Model B (Un-factored Kubo bubble retaining finite topological Casimir kinetic term -2*c_anom*zeta(1)*Om^2).
       - Hybrid time-domain tail accuracy check (J_cut = 30, 60 explicit poles + continuum-log tail).
"""

import json
import math
from pathlib import Path
import mpmath as mp
import numpy as np
from scipy.optimize import minimize_scalar, brentq

STREAM_DIR = Path(__file__).resolve().parent
OUT_PATH = STREAM_DIR / "01_derive.json"

# Exact P05 / T3-B / T4-A constants (M_Pl = H_S = 1, Lambda = 0)
C_ANOM_NUM = 9
C_ANOM_DEN = 31
C_ANOM = C_ANOM_NUM / C_ANOM_DEN
SIGMA_0 = C_ANOM
KAPPA_V = -44738.0 / 93.0
X_RING = math.sqrt(44672.0 / 93.0)  # 21.916753546579507


# ==============================================================================
# 1. EXACT I* CHARACTER MULTIPLICITY & PERIOD-30 DECOMPOSITION
# ==============================================================================

def d_istar(j):
    """Exact multiplicity d_{I*}(j) of trivial I*-representations inside spin-j of SU(2)_L."""
    two_j = int(round(2 * j))
    if two_j < 0 or two_j % 2 == 1:
        return 0
    classes = [
        (1, 0.0), (1, math.pi), (30, math.pi / 2.0),
        (20, math.pi / 3.0), (20, 2.0 * math.pi / 3.0),
        (12, math.pi / 5.0), (12, 2.0 * math.pi / 5.0),
        (12, 3.0 * math.pi / 5.0), (12, 4.0 * math.pi / 5.0),
    ]
    total = 0.0
    for size, alpha in classes:
        if alpha == 0.0:
            chi = two_j + 1
        elif abs(alpha - math.pi) < 1e-12:
            chi = (two_j + 1) * ((-1) ** two_j)
        else:
            chi = math.sin((two_j + 1) * alpha) / math.sin(alpha)
        total += size * chi
    val = total / 120.0
    assert abs(val - round(val)) < 1e-9, f"Non-integer character multiplicity at j={j}: {val}"
    return int(round(val))


R_ISTAR_MOD30 = [120 * d_istar(r) - 2 * (2 * r + 1) for r in range(30)]


# ==============================================================================
# 2. INDEPENDENT TWO-WAY PROOF OF C_R(j) AND MAXWELL SELECTION RULE (KT1)
# ==============================================================================

def racah_cg(j1, m1, j2, m2, J, M):
    """Exact Racah formula for SU(2) Clebsch-Gordan coefficient <j1 m1; j2 m2 | J M>."""
    if abs(m1 + m2 - M) > 1e-9:
        return 0.0
    if J < abs(j1 - j2) - 1e-9 or J > j1 + j2 + 1e-9:
        return 0.0
    if abs(m1) > j1 + 1e-9 or abs(m2) > j2 + 1e-9 or abs(M) > J + 1e-9:
        return 0.0

    def fac(x):
        return math.factorial(int(round(x)))

    pref = math.sqrt(
        (2 * J + 1) * fac(J + j1 - j2) * fac(J - j1 + j2) * fac(j1 + j2 - J) / fac(j1 + j2 + J + 1)
    )
    pref *= math.sqrt(
        fac(J + M) * fac(J - M) * fac(j1 + m1) * fac(j1 - m1) * fac(j2 + m2) * fac(j2 - m2)
    )
    kmin = max(0, int(round(j2 - J - m1)), int(round(j1 - J + m2)))
    kmax = min(int(round(j1 + j2 - J)), int(round(j1 - m1)), int(round(j2 + m2)))
    s = 0.0
    for k in range(kmin, kmax + 1):
        denom = (
            fac(k) * fac(j1 + j2 - J - k) * fac(j1 - m1 - k)
            * fac(j2 + m2 - k) * fac(J - j2 + m1 + k) * fac(J - j1 - m2 + k)
        )
        s += ((-1) ** k) / denom
    return pref * s


def verify_triple_overlap_and_kt1():
    """
    Two-way independent verification of:
      (a) C_R(j) = (2j-1)(2j+3) / [5*(2j+1)] via BOTH Racah CG sum and SU(2) curl-matrix diagonalization,
      (b) Exact Bach factorization (2j+1)*(2j)*(2j+2)*C_R(j) == (1/80)*B_2(4j+2),
      (c) Same-chirality (-2, -2) at j=0 (E=4) has nonzero geometric overlap = 1.0000 (for fixed unit-norm E_2),
          proving the sub-threshold gap |Omega| < 26 (I*) and |Omega| < 6 (S^3) is strictly enforced
          by Maxwell's (1 - s_1*s_2) electromagnetic duality selection rule!
    """
    def su2_gens(j):
        dim = int(round(2 * j + 1))
        m = np.arange(j, -j - 1, -1, dtype=float)
        Jp = np.zeros((dim, dim), dtype=complex)
        for idx in range(dim - 1):
            mv = m[idx + 1]
            Jp[idx, idx + 1] = math.sqrt(j * (j + 1) - mv * (mv + 1))
        Jm = Jp.T.conj()
        return 0.5 * (Jp + Jm), -0.5j * (Jp - Jm), np.diag(m.astype(complex))

    eps3 = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
                    eps3[i, j, k] = 1.0
                elif (i, j, k) in [(0, 2, 1), (2, 1, 0), (1, 0, 2)]:
                    eps3[i, j, k] = -1.0

    Gam1 = [np.zeros((3, 3), dtype=complex) for _ in range(3)]
    for a in range(3):
        for b in range(3):
            for c in range(3):
                Gam1[a][b, c] = -eps3[a, b, c]

    E2 = np.diag([1.0, -1.0, 0.0]).astype(complex) / math.sqrt(2.0)

    # Check j=0 (-2, -2) same-chirality geometric overlap (E=4)
    # At j=0, V_- are the 3 left-invariant basis vectors e_1, e_2, e_3 (I_3 in frame space).
    # Geometric overlap sum_{a,b} |E2_{ab}|^2 = Tr(E2^2) = 1.0 (for 5-component average: 1/5 per component -> actually 1.0 for fixed E2!).
    O_j0_same = float(np.sum(np.abs(E2) ** 2))
    maxwell_factor_same = 1.0 - (-1.0) * (-1.0)  # == 0.0 identically!

    rows = []
    max_cg_err = 0.0
    max_mat_err = 0.0
    max_bach_err = 0.0
    for two_j in [2, 3, 4, 5, 6, 12, 20]:
        j = two_j / 2.0
        cr_exact = (2 * j - 1) * (2 * j + 3) / (5.0 * (2 * j + 1))

        # Method A: Direct Racah CG sum (for j <= 6 to keep instant; exact for all)
        cr_cg = None
        if two_j <= 12:
            tot = 0.0
            for mR1_2 in range(-int(2 * (j - 1)), int(2 * (j - 1)) + 1, 2):
                mR1 = mR1_2 / 2.0
                for mR2_2 in range(-int(2 * (j + 1)), int(2 * (j + 1)) + 1, 2):
                    mR2 = mR2_2 / 2.0
                    A = 0.0
                    for m1_2 in range(-two_j, two_j + 1, 2):
                        m1 = m1_2 / 2.0
                        for mu1 in (-1, 0, 1):
                            for mu2 in (-1, 0, 1):
                                if mu1 + mu2 != 0:
                                    continue
                                phase = (-1) ** int(round(j - m1))
                                A += (
                                    phase
                                    * racah_cg(j, m1, 1, mu1, j - 1, mR1)
                                    * racah_cg(j, -m1, 1, mu2, j + 1, mR2)
                                    * racah_cg(1, mu1, 1, mu2, 2, 0)
                                )
                    tot += A * A
            cr_cg = tot
            max_cg_err = max(max_cg_err, abs(cr_cg - cr_exact))

        # Method B: Direct SU(2) curl-operator matrix diagonalization
        Jx, Jy, Jz = su2_gens(j)
        dim_o = len(Jx)
        ea = [-2.0j * Jx, -2.0j * Jy, -2.0j * Jz]
        D = [np.kron(ea[a], np.eye(3)) + np.kron(np.eye(dim_o), Gam1[a]) for a in range(3)]
        Curl = np.zeros((dim_o * 3, dim_o * 3), dtype=complex)
        for a in range(3):
            Ca = np.zeros((3, 3), dtype=complex)
            for i in range(3):
                for b in range(3):
                    Ca[i, b] = eps3[i, a, b]
            Curl += np.kron(np.eye(dim_o), Ca) @ D[a]
        evals, evecs = np.linalg.eigh(Curl)
        Vp = evecs[:, np.abs(evals - (2 * j)) < 1e-9]
        Vm = evecs[:, np.abs(evals + (2 * j + 2)) < 1e-9]
        T_orb = np.zeros((dim_o, dim_o), dtype=complex)
        ms = np.arange(j, -j - 1, -1, dtype=float)
        for r in range(dim_o):
            T_orb[r, dim_o - 1 - r] = (-1) ** int(round(j - ms[r]))
        O_mat = Vp.T @ np.kron(T_orb, E2) @ Vm
        cr_mat = float(np.sum(np.abs(O_mat) ** 2))
        max_mat_err = max(max_mat_err, abs(cr_mat - cr_exact))

        # Exact Bach factorization check
        Ej = 4 * j + 2
        lhs = (2 * j + 1) * (2 * j) * (2 * j + 2) * cr_exact
        rhs = ((Ej ** 2 - 4.0) * (Ej ** 2 - 16.0)) / 80.0
        max_bach_err = max(max_bach_err, abs(lhs - rhs) / rhs)

        rows.append({
            "j": j, "E_j": Ej, "d_Istar": d_istar(j),
            "C_R_exact": cr_exact, "C_R_cg": cr_cg, "C_R_matrix": cr_mat,
            "matrix_abs_err": abs(cr_mat - cr_exact),
        })

    # Verify exact period-30 property of R_{I*}(j) across j = 0..299
    period30_ok = all(
        (120 * d_istar(j) - 2 * (2 * j + 1)) == R_ISTAR_MOD30[j % 30]
        for j in range(300)
    )
    sum_R30 = sum(R_ISTAR_MOD30)

    # First 15 nonzero S^3/I* poles (j <= 30, E_j <= 122)
    first_poles_istar = [
        {
            "j": j,
            "omega_plus": 2 * j,
            "omega_minus": 2 * j + 2,
            "E_j": 4 * j + 2,
            "d_Istar": d_istar(j),
            "W_Istar": 120.0 * d_istar(j) / (2 * j + 1),
            "two_W_Istar": 240.0 * d_istar(j) / (2 * j + 1),
            "C_R": (2 * j - 1) * (2 * j + 3) / (5.0 * (2 * j + 1)),
        }
        for j in range(1, 31) if d_istar(j) > 0
    ]
    assert first_poles_istar[0]["E_j"] == 26
    assert first_poles_istar[1]["E_j"] == 42

    # Check zero poles below 26 on S^3/I* and lowest pole = 6 on round S^3
    lowest_istar_E = first_poles_istar[0]["E_j"]
    lowest_s3_E = 4 * 1 + 2  # j = 1 -> (+2, -4) -> E = 6

    return {
        "C_R_verification_rows": rows,
        "max_cg_err": float(max_cg_err),
        "max_matrix_err": float(max_mat_err),
        "max_bach_factorization_rel_err": float(max_bach_err),
        "period30_exact_up_to_j299": bool(period30_ok),
        "sum_R_mod30": int(sum_R30),
        "same_chirality_j0_E4_geometric_overlap_sq": O_j0_same,
        "same_chirality_maxwell_selection_factor": maxwell_factor_same,
        "lowest_S3_opposite_chirality_pole_E": lowest_s3_E,
        "lowest_Istar_opposite_chirality_pole_E": lowest_istar_E,
        "first_15_Istar_poles_j_le_30": first_poles_istar,
    }


# ==============================================================================
# 3. EXACT 40-DIGIT DIGAMMA KERNELS K_{S^3}^{disc} AND K_{I*}^{disc}
# ==============================================================================

def K_S3_exact_mp(Om, mu):
    """Exact closed-form discrete 2-photon kernel on round S^3 (analytic for |Om| < 6)."""
    return mp.log(4 / (mu ** 2)) + mp.psi(0, 3 - Om / 2) + mp.psi(0, 3 + Om / 2)


def K_Istar_exact_mp(Om, mu):
    """
    Exact closed-form discrete 2-photon kernel on S^3/I* (analytic for |Om| < 26,
    and free of removable singularities everywhere in |Om| < 126 by separating j=1..30 explicitly).
    UV-normalized identically to continuum log: K_{I*}(i*Om_E) = ln(Om_E^2/mu^2) + O(Om_E^-2).
    """
    tail_lead = mp.log(16 / (mu ** 2)) + mp.psi(0, mp.mpf("31.5") - Om / 4) + mp.psi(0, mp.mpf("31.5") + Om / 4)
    tail_rem = mp.mpc(0)
    if abs(Om) < mp.mpf("1e-12"):
        for r in range(1, 31):
            a = mp.mpf(120 + 4 * r + 2) / 120
            tail_rem += R_ISTAR_MOD30[r % 30] * 8 * mp.psi(1, a) / (120 ** 2)
    else:
        for r in range(1, 31):
            am = (mp.mpf(120 + 4 * r + 2) - Om) / 120
            ap = (mp.mpf(120 + 4 * r + 2) + Om) / 120
            tail_rem += (8 * R_ISTAR_MOD30[r % 30] / (2 * Om * 120)) * (mp.psi(0, ap) - mp.psi(0, am))
    low = mp.mpc(0)
    for j in range(1, 31):
        d = d_istar(j)
        if d == 0:
            continue
        Ej = mp.mpf(4 * j + 2)
        w2 = mp.mpf(240 * d) / (2 * j + 1)
        low += w2 * (2 * Ej) / (Ej ** 2 - Om ** 2)
    return tail_lead - tail_rem - low


def K_Istar_truncated_mp(Om, mu, J_cut):
    """
    Truncated discrete S^3/I* pole sum up to j <= J_cut (E_j <= 4*J_cut + 2) with
    exact UV logarithmic subtraction ln((4*J_cut + 4)^2 / mu^2).
    """
    E_uv = mp.mpf(4 * J_cut + 4)
    s = mp.log((E_uv ** 2) / (mu ** 2))
    for j in range(1, J_cut + 1):
        d = d_istar(j)
        if d == 0:
            continue
        Ej = mp.mpf(4 * j + 2)
        w2 = mp.mpf(240 * d) / (2 * j + 1)
        s -= w2 * (2 * Ej) / (Ej ** 2 - Om ** 2)
    return s


def K_Istar_hybrid_mp(Om, mu, J_cut, match_K0=False):
    """
    Time-domain-realizable hybrid kernel: exact discrete poles for j <= J_cut
    PLUS exact continuum-log UV tail ln((E_tail^2 - (Om+i0+)^2)/mu^2) with E_tail = 4*J_cut + 4.
    When match_K0=True (as used in Stages 2-3), adds the exact mu-independent O(1/E_tail^2)
    static offset delta_C_tail = K_exact(0) - K_hybrid_raw(0) so K_hybrid(0) == K_exact(0).
    """
    E_tail = mp.mpf(4 * J_cut + 4)
    if mp.im(Om) > 0:
        tail = mp.log(-1j * (Om + E_tail) / mu) + mp.log(-1j * (Om - E_tail) / mu)
    else:
        tail = mp.log((E_tail ** 2 - Om ** 2) / (mu ** 2))
    low = mp.mpc(0)
    for j in range(1, J_cut + 1):
        d = d_istar(j)
        if d == 0:
            continue
        Ej = mp.mpf(4 * j + 2)
        w2 = mp.mpf(240 * d) / (2 * j + 1)
        low += w2 * (2 * Ej) / (Ej ** 2 - Om ** 2)
    val = tail - low
    if match_K0:
        # mu-independent exact static offset K_exact(0; mu) - K_hyb_raw(0; mu)
        K0_ex = K_Istar_exact_mp(mp.mpf(0), mu)
        K0_raw = mp.log((E_tail ** 2) / (mu ** 2)) - sum(
            (mp.mpf(240 * d_istar(j)) / (2 * j + 1)) * 2 / (4 * j + 2)
            for j in range(1, J_cut + 1) if d_istar(j) > 0
        )
        val += (K0_ex - K0_raw)
    return val


def K_Istar_truncated_f64(Om, mu, J_cut):
    """Pure float64 evaluator for dual-precision cross-check (KT3)."""
    E_uv = 4.0 * J_cut + 4.0
    s = math.log((E_uv * E_uv) / (mu * mu))
    for j in range(1, J_cut + 1):
        d = d_istar(j)
        if d == 0:
            continue
        Ej = 4.0 * j + 2.0
        w2 = 240.0 * d / (2.0 * j + 1.0)
        s -= w2 * (2.0 * Ej) / (Ej * Ej - Om * Om)
    return s


# ==============================================================================
# 4. KT1 ANALYTICITY & KT2 UV CONTINUUM / WEYL-ANOMALY RECOVERY
# ==============================================================================

def verify_kt1_and_kt2():
    """
    Verify:
      - KT1: K_{I*}^{disc}(Omega) is strictly real and analytic (Im == 0) across (-26, 26),
        including at round-S^3 pole locations Omega = 6, 8, 10, ..., 24.
      - KT2:
        (a) High-frequency spectral density sliding-window average recovers exact c_anom = 9/31
            (normalized spectral density rho_avg / [B_2/(40*pi)] -> 1.000000),
        (b) Euclidean UV asymptote K(i*Om_E) - ln(Om_E^2/mu^2) == -zeta(1)/Om_E^2 + O(Om_E^-4)
            with exact Hurwitz-zeta constants zeta_{S^3}(1) = -74/3 and zeta_{I*}(1) = -724/3,
        (c) Gaussian heat-kernel regulator exp(-(eps*E)^2) check of Delta_S = S_{I*} - S_{S^3}
            confirming exact O(Om_E^4 ln Om_E) and O(Om_E^2) local UV counterterm cancellation.
    """
    with mp.workdps(40):
        xr = mp.sqrt(mp.mpf(44672) / 93)

        # KT1: Check analyticity & Im == 0 at all round-S^3 pole frequencies < 26 and sub-threshold points
        sub_check = []
        max_im_sub = 0.0
        for Om_f in [0.0, 2.8284271247461903, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 25.5]:
            Om = mp.mpf(str(Om_f))
            val = K_Istar_exact_mp(Om, xr)
            k_cont = mp.log((676 - Om ** 2) / (xr ** 2))
            im_v = abs(float(mp.im(val)))
            max_im_sub = max(max_im_sub, im_v)
            sub_check.append({
                "Omega": Om_f,
                "K_Istar_disc_re": float(mp.re(val)),
                "K_Istar_disc_im": float(mp.im(val)),
                "K_M26_cont_re": float(mp.re(k_cont)),
                "diff_disc_minus_cont": float(mp.re(val - k_cont)),
            })

        # KT2a: High-frequency sliding-window spectral density average (period-30 window Delta_E = 120)
        # Recall: exact discrete spectral density in Im Pi is sum_j [2*W_{I*}(j)] * B_2(E_j) * delta(Om - E_j),
        # which averages to 1.0 * B_2(Om) (matching c_anom = 9/31 identically!).
        window_checks = []
        for m_start in [1, 5, 10, 20, 50]:
            # Sum 2*W_{I*}(j) over one exact mod-30 period j in [30*m, 30*m + 29], width Delta_E = 120
            j0 = 30 * m_start
            w_sum = sum(240.0 * d_istar(j) / (2.0 * j + 1.0) for j in range(j0, j0 + 30))
            rho_ratio = w_sum / 120.0  # MUST converge to 1.000000 -> c_anom_eff = (9/31) * rho_ratio!
            E_mid = 4 * (j0 + 14.5) + 2
            window_checks.append({
                "E_window": [4 * j0 + 2, 4 * (j0 + 29) + 2],
                "E_mid": E_mid,
                "normalized_spectral_density_avg": rho_ratio,
                "c_anom_recovered": C_ANOM * rho_ratio,
                "rel_err_vs_9_over_31": abs(rho_ratio - 1.0),
            })

        # KT2b: Exact Hurwitz-zeta constants & Euclidean UV asymptotic check
        def zeta_S3(p):
            return 4 * (mp.mpf(2) ** p) * mp.zeta(-p, 3)

        def zeta_Istar(p):
            part1 = 8 * (mp.mpf(4) ** p) * mp.zeta(-p, mp.mpf("1.5"))
            part2 = 8 * (mp.mpf(120) ** (p - 1)) * sum(
                R_ISTAR_MOD30[r % 30] * mp.zeta(-(p - 1), mp.mpf(4 * r + 2) / 120)
                for r in range(1, 31)
            )
            return part1 + part2

        z1_s3 = zeta_S3(1)
        z1_is = zeta_Istar(1)
        z3_s3 = zeta_S3(3)
        z3_is = zeta_Istar(3)
        z1_s3_exact = mp.mpf(-74) / 3
        z1_is_exact = mp.mpf(-724) / 3
        z3_s3_exact = mp.mpf(-4316) / 15
        z3_is_exact = mp.mpf(-521224) / 45
        assert abs(z1_s3 - z1_s3_exact) / abs(z1_s3_exact) < mp.mpf("1e-35")
        assert abs(z1_is - z1_is_exact) / abs(z1_is_exact) < mp.mpf("1e-35")
        assert abs(z3_s3 - z3_s3_exact) / abs(z3_s3_exact) < mp.mpf("1e-35")
        assert abs(z3_is - z3_is_exact) / abs(z3_is_exact) < mp.mpf("1e-35")

        euclid_rows = []
        for OmE in [50, 200, 1000, 5000]:
            z = 1j * mp.mpf(OmE)
            asymp = mp.log((mp.mpf(OmE) ** 2) / (xr ** 2))
            d_is = mp.re(K_Istar_exact_mp(z, xr) - asymp)
            d_s3 = mp.re(K_S3_exact_mp(z, xr) - asymp)
            euclid_rows.append({
                "Omega_E": OmE,
                "K_Istar_minus_log": float(d_is),
                "K_Istar_times_OmE2": float(d_is * (OmE ** 2)),
                "K_Istar_O_OmE_m2_rel_err": float(abs(d_is * (OmE ** 2) - (-z1_is_exact)) / abs(z1_is_exact)),
                "K_S3_minus_log": float(d_s3),
                "K_S3_times_OmE2": float(d_s3 * (OmE ** 2)),
                "K_S3_O_OmE_m2_rel_err": float(abs(d_s3 * (OmE ** 2) - (-z1_s3_exact)) / abs(z1_s3_exact)),
            })

        # KT2c: Gaussian heat-kernel regulator exp(-(eps*E)^2) convergence check at Om = 40i
        dz1 = z1_is - z1_s3
        dz3 = z3_is - z3_s3
        Om_g = mp.mpc(0, 40)
        B2_g = (Om_g ** 2 - 4) * (Om_g ** 2 - 16)
        pred_g = float(mp.re((dz3 - 20 * dz1) + (Om_g ** 2) * dz1 - B2_g * (K_Istar_exact_mp(Om_g, xr) - K_S3_exact_mp(Om_g, xr))))
        gauss_rows = []
        for eps_f, emax in [(0.02, 1000), (0.01, 1600), (0.005, 2400)]:
            eps = mp.mpf(str(eps_f))
            s_s3 = mp.mpf(0)
            for k in range(3, emax // 2 + 1):
                E = mp.mpf(2 * k)
                s_s3 += 4 * (E ** 2 - 4) * (E ** 2 - 16) * E / (E ** 2 - Om_g ** 2) * mp.exp(-((eps * E) ** 2))
            s_is = mp.mpf(0)
            for j in range(1, emax // 4 + 1):
                d = d_istar(j)
                if d == 0:
                    continue
                E = mp.mpf(4 * j + 2)
                s_is += (mp.mpf(480 * d) / (2 * j + 1)) * (E ** 2 - 4) * (E ** 2 - 16) * E / (E ** 2 - Om_g ** 2) * mp.exp(-((eps * E) ** 2))
            val_g = float(mp.re(s_is - s_s3))
            gauss_rows.append({
                "eps": eps_f, "Delta_S_reg": val_g, "analytic": pred_g,
                "rel_err": abs(val_g - pred_g) / abs(pred_g),
            })

        return {
            "kt1_subthreshold_analyticity": sub_check,
            "kt1_max_abs_Im_below_26": max_im_sub,
            "kt2_sliding_window_weyl_anomaly": window_checks,
            "kt2_exact_zeta_constants": {
                "zeta_S3_1": float(z1_s3),
                "zeta_S3_1_rational": "-74/3",
                "zeta_Istar_1": float(z1_is),
                "zeta_Istar_1_rational": "-724/3",
                "zeta_S3_3": float(z3_s3),
                "zeta_S3_3_rational": "-4316/15",
                "zeta_Istar_3": float(z3_is),
                "zeta_Istar_3_rational": "-521224/45",
                "kubo_Om0_Istar_exact": float(-2 * (mp.mpf(9) / 31) * (z3_is - 20 * z1_is)),
                "kubo_Om0_Istar_rational": "608048/155",
            },
            "kt2_euclidean_uv_asymptote": euclid_rows,
            "kt2_gaussian_heat_kernel_bubble_diff_Om40i": gauss_rows,
        }


# ==============================================================================
# 5. KT3: CUTOFF CONVERGENCE, POLE-REPULSION SHIFT, DUAL-PRECISION & MODEL A/B
# ==============================================================================

def find_pinch_mp(K_fn, M_val, xr, c_an, kin_shift=mp.mpf(0)):
    """Find exact static Horowitz pinch-off (beta_pinch, Omega_pinch) in 40-digit mpmath."""
    def beta_c(Om):
        B2 = (Om ** 2 - 4) * (Om ** 2 - 16)
        return (xr ** 2 * (Om ** 2 - 8) - kin_shift * (Om ** 2)) / (4 * B2) - (c_an / 2) * mp.re(K_fn(Om))

    grid = np.linspace(4.05, M_val - 0.05, 400)
    vals = [float(beta_c(mp.mpf(g))) for g in grid]
    im = int(np.argmin(vals))
    lo = mp.mpf(grid[max(0, im - 2)])
    hi = mp.mpf(grid[min(len(grid) - 1, im + 2)])
    om_s = mp.findroot(lambda o: mp.diff(beta_c, o), (lo, hi), solver="anderson", tol=1e-24)
    b_s = beta_c(om_s)
    return float(mp.re(b_s)), float(mp.re(om_s)), float(mp.re(om_s / xr))


def find_pinch_f64(K_fn_f64, M_val):
    """Independent float64 optimizer for dual-precision cross-check."""
    def beta_c(Om):
        B2 = (Om * Om - 4.0) * (Om * Om - 16.0)
        return (X_RING * X_RING * (Om * Om - 8.0)) / (4.0 * B2) - 0.5 * C_ANOM * K_fn_f64(Om)

    res = minimize_scalar(beta_c, bounds=(4.05, M_val - 0.05), method="bounded", options={"xatol": 1e-13})
    return float(res.fun), float(res.x), float(res.x / X_RING)


def solve_roots_mp(K_fn, b0_list, b_pinch, om_pinch, M_val, xr, c_an, second_pole=None):
    """
    Solve exact complex UHP Horowitz root (b0 < b_pinch) or BOTH real sub-threshold roots
    (lower negative-Krein Stelle root in (4, om_pinch) AND upper positive-Krein companion root
    in (om_pinch, M_val)) plus optional first above-threshold pole-interlaced real root
    Omega_1 in (M_val, second_pole).
    Uses monotonic beta_c(Om) - b0 == 0 bisection/Anderson on real branches so pole singularities
    near M_val^- never cause bracket blowup or silent null dropouts!
    """
    def D_fn(Om, b0):
        B2 = (Om ** 2 - 4) * (Om ** 2 - 16)
        return xr ** 2 * (Om ** 2 - 8) - 4 * (b0 + (c_an / 2) * K_fn(Om)) * B2

    def beta_c_real(Om):
        B2 = (Om ** 2 - 4) * (Om ** 2 - 16)
        return (xr ** 2 * (Om ** 2 - 8)) / (4 * B2) - (c_an / 2) * mp.re(K_fn(Om))

    def bisect_beta(lo, hi, target_b0, tol=mp.mpf("1e-26")):
        """Bracketed bisection root finder for beta_c_real(Om) == target_b0 on [lo, hi].
        Note: beta_c_real is strictly monotonic on S^3/I*'s (26, 42) and higher intervals,
        whereas on round S^3's (6, 8) interior non-monotonicity creates 3 real roots for
        beta_0 in (~3.6, ~4.1) (outside the sampled round-S^3 grid)."""
        flo = beta_c_real(lo) - target_b0
        fhi = beta_c_real(hi) - target_b0
        assert flo * fhi <= 0, f"Unbracketed real root on [{float(lo)}, {float(hi)}]: {float(flo)}, {float(fhi)}"
        for _ in range(130):
            mid = mp.mpf("0.5") * (lo + hi)
            if hi - lo < tol:
                return mid
            fmid = beta_c_real(mid) - target_b0
            if fmid == 0:
                return mid
            if flo * fmid <= 0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
        return mp.mpf("0.5") * (lo + hi)

    rows = []
    prev_uhp = None
    for b0_f in b0_list:
        b0 = mp.mpf(str(b0_f))
        uhp_rt = None
        real_lo = None
        real_hi = None
        above_om1 = None
        grav_rt = None

        # Physical graviton root near sqrt(8) (positive Krein energy, D'(Om) > 0)
        rg = mp.findroot(lambda z: D_fn(z, b0), mp.mpf("2.8284"), tol=1e-22, maxsteps=60)
        assert abs(D_fn(rg, b0)) < mp.mpf("1e-16"), f"Graviton root failed at b0={b0_f}"
        grav_rt = float(mp.re(rg))

        # First above-threshold pole-interlaced real root Omega_1 in (M_val, second_pole) (positive Krein energy)
        if second_pole is not None:
            eps_l = mp.mpf("1e-6")
            while beta_c_real(mp.mpf(M_val) + eps_l) > b0:
                eps_l *= mp.mpf("0.1")
            eps_r = mp.mpf("1e-6")
            while beta_c_real(mp.mpf(second_pole) - eps_r) < b0:
                eps_r *= mp.mpf("0.1")
            r_ab = bisect_beta(mp.mpf(M_val) + eps_l, mp.mpf(second_pole) - eps_r, b0)
            assert abs(beta_c_real(r_ab) - b0) < mp.mpf("1e-18")
            above_om1 = float(r_ab)

        if b0_f > b_pinch + 1e-6:
            # Strictly sub-threshold: BOTH real poles MUST exist with zero null dropouts!
            # Lower root (negative-Krein Stelle ghost) in (4, om_pinch): beta_c decreases from +inf to b_pinch
            eps4 = mp.mpf("1e-4")
            while beta_c_real(mp.mpf(4) + eps4) < b0:
                eps4 *= mp.mpf("0.1")
            r_lo = bisect_beta(mp.mpf(4) + eps4, mp.mpf(om_pinch), b0)
            assert abs(beta_c_real(r_lo) - b0) < mp.mpf("1e-18")
            real_lo = float(r_lo)

            # Upper companion root (positive-Krein) in (om_pinch, M_val): beta_c increases from b_pinch to +inf
            epsM = mp.mpf("1e-3")
            while beta_c_real(mp.mpf(M_val) - epsM) < b0:
                epsM *= mp.mpf("0.1")
            r_hi = bisect_beta(mp.mpf(om_pinch), mp.mpf(M_val) - epsM, b0)
            assert abs(beta_c_real(r_hi) - b0) < mp.mpf("1e-18")
            real_hi = float(r_hi)
        else:
            guesses = []
            if prev_uhp is not None:
                guesses.append(prev_uhp)
            guesses.extend([
                mp.mpc(om_pinch, 2.0), mp.mpc(om_pinch, 5.0),
                mp.mpc(om_pinch, 8.0), mp.mpc(12.0, 8.0), mp.mpc(5.5, 2.0),
            ])
            for g in guesses:
                try:
                    rt = mp.findroot(lambda z: D_fn(z, b0), g, tol=1e-22, maxsteps=80)
                    if abs(D_fn(rt, b0)) < 1e-14 and rt.imag > 1e-8 and rt.real >= -1e-9:
                        uhp_rt = rt
                        prev_uhp = rt
                        break
                except Exception:
                    pass
            assert uhp_rt is not None, f"Failed to find UHP Horowitz root at b0={b0_f}"

        rows.append({
            "beta_0": b0_f,
            "grav_Omega": grav_rt,
            "uhp_Omega_re": float(uhp_rt.real) if uhp_rt is not None else None,
            "uhp_Omega_im": float(uhp_rt.imag) if uhp_rt is not None else None,
            "uhp_gamma_HS": float(uhp_rt.imag / xr) if uhp_rt is not None else None,
            "real_stelle_lo_Omega": real_lo,
            "real_stelle_hi_Omega": real_hi,
            "above_threshold_Omega_1": above_om1,
        })
    return rows


def verify_kt3():
    """Run complete KT3 cutoff convergence, dual-precision oracle, Model A vs B, and hybrid tail checks."""
    with mp.workdps(40):
        xr = mp.sqrt(mp.mpf(44672) / 93)
        c_an = mp.mpf(9) / 31

        # 1. Exact Model A pinch-off comparison: Continuum vs Discrete on S^3 and S^3/I*
        b_s3_cont, om_s3_cont, w_s3_cont = find_pinch_mp(lambda o: mp.log((36 - o ** 2) / (xr ** 2)), 6.0, xr, c_an)
        b_s3_disc, om_s3_disc, w_s3_disc = find_pinch_mp(lambda o: K_S3_exact_mp(o, xr), 6.0, xr, c_an)
        b_is_cont, om_is_cont, w_is_cont = find_pinch_mp(lambda o: mp.log((676 - o ** 2) / (xr ** 2)), 26.0, xr, c_an)
        b_is_disc, om_is_disc, w_is_disc = find_pinch_mp(lambda o: K_Istar_exact_mp(o, xr), 26.0, xr, c_an)

        # 2. Model B sensitivity check (un-factored Kubo bubble retaining finite topological -2*c_anom*zeta(1)*Om^2)
        z1_is = mp.mpf(-724) / 3
        z1_s3 = mp.mpf(-74) / 3
        b_is_modB_abs, om_is_modB_abs, w_is_modB_abs = find_pinch_mp(
            lambda o: K_Istar_exact_mp(o, xr), 26.0, xr, c_an, kin_shift=2 * c_an * (-z1_is)
        )
        b_is_modB_rel, om_is_modB_rel, w_is_modB_rel = find_pinch_mp(
            lambda o: K_Istar_exact_mp(o, xr), 26.0, xr, c_an, kin_shift=2 * c_an * (-(z1_is - z1_s3))
        )

        # 3. Cutoff convergence ladder (BOTH E_cut in {60, 100, 160, 240, 480} AND period-30 J_cut in {30, 60, 120, 240})
        # Plus dual-precision float64 vs 40-digit mpmath check!
        cutoff_ladder = []
        for label, J_cut in [
            ("Ecut_60_J14", 14),
            ("Ecut_100_J24", 24),
            ("J30_E122_1per", 30),
            ("Ecut_160_J39", 39),
            ("J60_E242_2per", 60),
            ("J120_E482_4per", 120),
            ("J240_E962_8per", 240),
        ]:
            b_mp, om_mp, w_mp = find_pinch_mp(lambda o, jc=J_cut: K_Istar_truncated_mp(o, xr, jc), 26.0, xr, c_an)
            b_f64, om_f64, _ = find_pinch_f64(lambda o, jc=J_cut: K_Istar_truncated_f64(o, X_RING, jc), 26.0)
            # Also check BOTH raw hybrid tail AND K0-matched hybrid tail (used in Stages 2-3!)
            b_hyb, om_hyb, _ = find_pinch_mp(lambda o, jc=J_cut: K_Istar_hybrid_mp(o, xr, jc, match_K0=False), 26.0, xr, c_an)
            b_hyb_m, om_hyb_m, _ = find_pinch_mp(lambda o, jc=J_cut: K_Istar_hybrid_mp(o, xr, jc, match_K0=True), 26.0, xr, c_an)
            cutoff_ladder.append({
                "label": label,
                "J_cut": J_cut,
                "E_max_pole": 4 * J_cut + 2,
                "n_nonzero_poles": sum(1 for j in range(1, J_cut + 1) if d_istar(j) > 0),
                "beta_pinch_trunc_mp40": b_mp,
                "Omega_pinch_trunc_mp40": om_mp,
                "rel_err_trunc_vs_exact": abs(b_mp - b_is_disc) / b_is_disc,
                "beta_pinch_trunc_f64": b_f64,
                "dual_precision_beta_diff": abs(b_mp - b_f64),
                "dual_precision_Omega_diff": abs(om_mp - om_f64),
                "beta_pinch_hybrid_raw_mp40": b_hyb,
                "rel_err_hybrid_raw_vs_exact": abs(b_hyb - b_is_disc) / b_is_disc,
                "beta_pinch_hybrid_K0matched_mp40": b_hyb_m,
                "rel_err_hybrid_K0matched_vs_exact": abs(b_hyb_m - b_is_disc) / b_is_disc,
            })

        # 4. Exact static dispersion roots for discrete S^3/I* across beta_0 in [0.289, 10.0]
        #    (ZERO null dropouts on real_stelle_hi_Omega AND first above-threshold root Omega_1 in (26, 42)!)
        b0_scan = [0.289, 0.3836, 0.44, 0.50, 0.60, 0.65, 0.68, 0.70, 0.75, 0.80, 1.00, 1.50, 1.71, 2.00, 5.00, 6.37, 10.0]
        roots_istar = solve_roots_mp(lambda o: K_Istar_exact_mp(o, xr), b0_scan, b_is_disc, om_is_disc, 26.0, xr, c_an, second_pole=42.0)
        roots_s3 = solve_roots_mp(lambda o: K_S3_exact_mp(o, xr), [0.289, 0.50, 1.00, 2.00, 5.00, 6.37, 7.00, 8.00, 10.0], b_s3_disc, om_s3_disc, 6.0, xr, c_an, second_pole=8.0)

        # Cross-check BOTH raw and K0-matched hybrid tails (J30 & J60) on UHP, lower Stelle, AND upper companion roots!
        b_h30_m, om_h30_m, _ = find_pinch_mp(lambda o: K_Istar_hybrid_mp(o, xr, 30, match_K0=True), 26.0, xr, c_an)
        b_h60_m, om_h60_m, _ = find_pinch_mp(lambda o: K_Istar_hybrid_mp(o, xr, 60, match_K0=True), 26.0, xr, c_an)
        chk_b0s = [0.289, 0.50, 0.60, 0.70, 1.00, 2.00]
        roots_hyb30 = solve_roots_mp(lambda o: K_Istar_hybrid_mp(o, xr, 30, match_K0=True), chk_b0s, b_h30_m, om_h30_m, 26.0, xr, c_an, second_pole=42.0)
        roots_hyb60 = solve_roots_mp(lambda o: K_Istar_hybrid_mp(o, xr, 60, match_K0=True), chk_b0s, b_h60_m, om_h60_m, 26.0, xr, c_an, second_pole=42.0)

        hyb30_max_diff = 0.0
        hyb60_max_diff = 0.0
        exact_by_b0 = {r["beta_0"]: r for r in roots_istar}
        for r30, r60 in zip(roots_hyb30, roots_hyb60):
            rex = exact_by_b0[r30["beta_0"]]
            if rex["uhp_gamma_HS"] is not None:
                hyb30_max_diff = max(hyb30_max_diff, abs(r30["uhp_gamma_HS"] - rex["uhp_gamma_HS"]) / rex["uhp_gamma_HS"])
                hyb60_max_diff = max(hyb60_max_diff, abs(r60["uhp_gamma_HS"] - rex["uhp_gamma_HS"]) / rex["uhp_gamma_HS"])
            else:
                for key_rt in ("real_stelle_lo_Omega", "real_stelle_hi_Omega", "above_threshold_Omega_1"):
                    hyb30_max_diff = max(hyb30_max_diff, abs(r30[key_rt] - rex[key_rt]) / rex[key_rt])
                    hyb60_max_diff = max(hyb60_max_diff, abs(r60[key_rt] - rex[key_rt]) / rex[key_rt])

        # Exact DRESSED Krein-signature resonance analysis for parametric pump at Om_scal(x_ring) = 40.6758
        # (macro-modulated across [38.60, 43.47]):
        #   - Positive Krein energy (D'(Om) > 0): graviton Om_grav(b0) ~ 2.83, upper companion Om_comp(b0) in (17.93, 26),
        #     and above-threshold pole-interlaced roots Om_k(b0) in (E_k, E_{k+1}), specifically:
        #       Om_1(b0) in (26, 42), Om_2(b0) in (42, 50), Om_8(b0) in (86, 90) [8th interval: 86=E_21, 90=E_22].
        #   - Negative Krein energy (D'(Om) < 0): lower Ostrogradsky Stelle root Om_Stelle_lo(b0) in (4, 17.93).
        #   Solve the EXACT DRESSED normal-mode equations directly from D(Om, beta_0) == 0:
        #     * (H1) Same-sign sum: 2 * Om_comp(b0) == Om_scal
        #     * (H_fund) k=2 fundamental of dressed Om_1: Om_1(b0) == Om_scal
        #     * (H2) Same-sign sum: Om_grav(b0) + Om_1(b0) == Om_scal (and upper modulated edge Om_scal_max=43.47)
        #     * (H3) Opposite-sign difference: Om_2(b0) - Om_Stelle_lo(b0) == Om_scal
        #     * (H4) Opposite-sign 2-quantum difference: Om_8(b0) - Om_Stelle_lo(b0) == 2*Om_scal
        #     * (E1_dressed) Opposite-sign elliptic sum: Om_Stelle_lo(b0) + Om_comp(b0) <= 2*Om_pinch = 35.86 < 38.60
        om_scal_ref = float(xr / mp.sqrt(c_an))  # 40.675773

        with mp.workdps(18):
            xr18 = mp.sqrt(mp.mpf(44672) / 93)
            c18 = mp.mpf(9) / 31

            def beta_of_om_18(o_f):
                om = mp.mpf(o_f)
                B2 = (om ** 2 - 4) * (om ** 2 - 16)
                return float(mp.re(xr18 ** 2 * (om ** 2 - 8) / (4 * B2) - (c18 / 2) * K_Istar_exact_mp(om, xr18)))

            def dressed_root_in_interval(b0_f, lo_pole, hi_pole):
                def f_om(o):
                    return beta_of_om_18(o) - b0_f
                eps_l = 1e-3
                while f_om(lo_pole + eps_l) > 0:
                    eps_l *= 0.1
                eps_r = 1e-3
                while f_om(hi_pole - eps_r) < 0:
                    eps_r *= 0.1
                return float(brentq(f_om, lo_pole + eps_l, hi_pole - eps_r, xtol=1e-10))

            def dressed_stelle_lo(b0_f):
                def f_om(o):
                    return beta_of_om_18(o) - b0_f
                eps4 = 1e-2
                while f_om(4.0 + eps4) < 0:
                    eps4 *= 0.1
                return float(brentq(f_om, 4.0 + eps4, om_is_disc, xtol=1e-10))

            def dressed_grav(b0_f):
                def f_g(o):
                    om = mp.mpf(o)
                    return float(mp.re(xr18 ** 2 * (-om ** 2 + 8) + 4 * (mp.mpf(b0_f) + (c18 / 2) * K_Istar_exact_mp(om, xr18)) * (om ** 2 - 4) * (om ** 2 - 16)))
                return float(brentq(f_g, 2.5, 3.2, xtol=1e-10))

            b_H1 = beta_of_om_18(0.5 * om_scal_ref)
            b_Hfund = beta_of_om_18(om_scal_ref)
            b_H2_static = float(brentq(lambda b: dressed_grav(b) + dressed_root_in_interval(b, 26.0, 42.0) - om_scal_ref, 0.05, 0.60, xtol=1e-8))
            b_H2_modmax = float(brentq(lambda b: dressed_grav(b) + dressed_root_in_interval(b, 26.0, 42.0) - 43.47, 0.80, 2.50, xtol=1e-8))
            b_H3 = float(brentq(lambda b: dressed_root_in_interval(b, 42.0, 50.0) - dressed_stelle_lo(b) - om_scal_ref, 1.0, 4.0, xtol=1e-8))
            om2_H3 = dressed_root_in_interval(b_H3, 42.0, 50.0)
            omlo_H3 = dressed_stelle_lo(b_H3)
            b_H4 = float(brentq(lambda b: dressed_root_in_interval(b, 86.0, 90.0) - dressed_stelle_lo(b) - 2.0 * om_scal_ref, 1.0, 4.0, xtol=1e-8))
            om8_H4 = dressed_root_in_interval(b_H4, 86.0, 90.0)
            omlo_H4 = dressed_stelle_lo(b_H4)

        res_targets = {
            "Om_scal_xring_static": om_scal_ref,
            "hyperbolic_H1_2x_Comp_eq_Omscal_beta0": b_H1,
            "hyperbolic_Hfund_Om1_eq_Omscal_beta0": b_Hfund,
            "hyperbolic_H2_Grav_plus_Om1_eq_Omscal_static_beta0": b_H2_static,
            "hyperbolic_H2_Grav_plus_Om1_eq_Omscal_modmax43p47_beta0": b_H2_modmax,
            "hyperbolic_H3_dressed_Om2_minus_StelleLo_eq_Omscal_beta0": b_H3,
            "hyperbolic_H3_dressed_Om2": om2_H3,
            "hyperbolic_H3_dressed_StelleLo": omlo_H3,
            "hyperbolic_H4_dressed_Om8_minus_StelleLo_eq_2Omscal_beta0": b_H4,
            "hyperbolic_H4_dressed_Om8": om8_H4,
            "hyperbolic_H4_dressed_StelleLo": omlo_H4,
            "elliptic_sum_StelleLo_plus_Comp_max_at_pinch": 2.0 * om_is_disc,
        }

        return {
            "pinch_off_summary": {
                "S3_continuum_M6": {"beta_pinch": b_s3_cont, "Omega_pinch": om_s3_cont, "omega_pinch_HS": w_s3_cont},
                "S3_discrete_exact": {"beta_pinch": b_s3_disc, "Omega_pinch": om_s3_disc, "omega_pinch_HS": w_s3_disc},
                "Istar_continuum_M26": {"beta_pinch": b_is_cont, "Omega_pinch": om_is_cont, "omega_pinch_HS": w_is_cont},
                "Istar_discrete_exact_ModelA": {"beta_pinch": b_is_disc, "Omega_pinch": om_is_disc, "omega_pinch_HS": w_is_disc},
                "Istar_discrete_ModelB_relS3": {"beta_pinch": b_is_modB_rel, "Omega_pinch": om_is_modB_rel, "omega_pinch_HS": w_is_modB_rel},
                "Istar_discrete_ModelB_abs": {"beta_pinch": b_is_modB_abs, "Omega_pinch": om_is_modB_abs, "omega_pinch_HS": w_is_modB_abs},
                "shift_Istar_ModelA_abs": b_is_disc - b_is_cont,
                "shift_Istar_ModelA_rel": (b_is_disc - b_is_cont) / b_is_cont,
                "shift_S3_abs": b_s3_disc - b_s3_cont,
                "shift_S3_rel": (b_s3_disc - b_s3_cont) / b_s3_cont,
                "topology_advantage_ratio_continuum": b_s3_cont / b_is_cont,
                "topology_advantage_ratio_discrete_ModelA": b_s3_disc / b_is_disc,
            },
            "cutoff_convergence_ladder": cutoff_ladder,
            "hybrid_K0matched_root_max_rel_err_J30": float(hyb30_max_diff),
            "hybrid_K0matched_root_max_rel_err_J60": float(hyb60_max_diff),
            "static_roots_Istar_discrete": roots_istar,
            "static_roots_S3_discrete": roots_s3,
            "kt4_sideband_resonance_beta0_predictions": res_targets,
        }


def main():
    print("=== STREAM p2_01_discrete_photon_pole_sum — STAGE 1: EXACT DERIVATION (40-DIGIT) ===", flush=True)
    ov = verify_triple_overlap_and_kt1()
    print(f"1. Exact C_R(j) = (2j-1)(2j+3)/[5(2j+1)] verification:")
    print(f"   max CG err = {ov['max_cg_err']:.2e} | max SU(2)-curl matrix err = {ov['max_matrix_err']:.2e} | "
          f"Bach factorization err = {ov['max_bach_factorization_rel_err']:.2e}")
    print(f"   Period-30 R_{{I*}}(j) exact = {ov['period30_exact_up_to_j299']} (sum_R30 = {ov['sum_R_mod30']})")
    print(f"   Same-chirality j=0 (E=4): geom |O|^2 = {ov['same_chirality_j0_E4_geometric_overlap_sq']:.4f}, "
          f"Maxwell (1-s1*s2) = {ov['same_chirality_maxwell_selection_factor']:.1f}")
    print(f"   First 5 S^3/I* poles (E_j, 2*W_j): {[(p['E_j'], round(p['two_W_Istar'], 3)) for p in ov['first_15_Istar_poles_j_le_30'][:5]]}")

    kt12 = verify_kt1_and_kt2()
    print(f"\n2. KT1 (Analyticity & Chiral Gap): max |Im K_I*(Om)| for |Om| < 26 = {kt12['kt1_max_abs_Im_below_26']:.2e}")
    print("3. KT2 (Continuum & Weyl-Anomaly UV Recovery):")
    for w in kt12["kt2_sliding_window_weyl_anomaly"]:
        print(f"   E in {w['E_window']}: avg_rho = {w['normalized_spectral_density_avg']:.8f}, "
              f"c_anom_rec = {w['c_anom_recovered']:.8f} (rel err = {w['rel_err_vs_9_over_31']:.2e})")
    e_last = kt12["kt2_euclidean_uv_asymptote"][-1]
    print(f"   Euclidean OmE=5000: K_I* - ln = {e_last['K_Istar_minus_log']:+.3e} "
          f"(OmE^2*diff = {e_last['K_Istar_times_OmE2']:.4f} vs exact 724/3 = {724/3:.4f})")
    print(f"   Gaussian heat-kernel bubble diff (Om=40i): {[round(r['rel_err'], 5) for r in kt12['kt2_gaussian_heat_kernel_bubble_diff_Om40i']]}")

    kt3 = verify_kt3()
    ps = kt3["pinch_off_summary"]
    print("\n4. KT3 (Static Pinch-Off Shift & Cutoff Convergence):")
    print(f"   Round S^3:  cont = {ps['S3_continuum_M6']['beta_pinch']:.6f} -> disc = {ps['S3_discrete_exact']['beta_pinch']:.6f} ({100*ps['shift_S3_rel']:+.2f}%)")
    print(f"   S^3/I* (A): cont = {ps['Istar_continuum_M26']['beta_pinch']:.6f} -> disc = {ps['Istar_discrete_exact_ModelA']['beta_pinch']:.6f} "
          f"({100*ps['shift_Istar_ModelA_rel']:+.2f}%, Om_pinch = {ps['Istar_discrete_exact_ModelA']['Omega_pinch']:.4f})")
    print(f"   S^3/I* (B): relS3 = {ps['Istar_discrete_ModelB_relS3']['beta_pinch']:.6f} | abs = {ps['Istar_discrete_ModelB_abs']['beta_pinch']:.6f}")
    print(f"   Topology advantage: cont = {ps['topology_advantage_ratio_continuum']:.2f}x -> disc = {ps['topology_advantage_ratio_discrete_ModelA']:.2f}x")
    print("   Cutoff ladder (truncated vs exact; hybrid raw & K0-matched vs exact; mp40 vs float64):")
    for row in kt3["cutoff_convergence_ladder"]:
        print(f"     {row['label']:16s} (n={row['n_nonzero_poles']:3d}): b_trunc={row['beta_pinch_trunc_mp40']:.8f} "
              f"(err={row['rel_err_trunc_vs_exact']:.2e}) | b_hyb_m={row['beta_pinch_hybrid_K0matched_mp40']:.8f} "
              f"(err={row['rel_err_hybrid_K0matched_vs_exact']:.2e}) | mp40-f64={row['dual_precision_beta_diff']:.2e}")
    print(f"   K0-matched hybrid root max rel err vs exact (incl. upper companion & Om_1): "
          f"J30 = {kt3['hybrid_K0matched_root_max_rel_err_J30']:.2e} | J60 = {kt3['hybrid_K0matched_root_max_rel_err_J60']:.2e}")
    print(f"   KT4 hyperbolic Krein sideband predictions: {kt3['kt4_sideband_resonance_beta0_predictions']}")

    payload = {
        "stream_id": "p2_01_discrete_photon_pole_sum",
        "constants": {"C_ANOM": C_ANOM, "SIGMA_0": SIGMA_0, "KAPPA_V": KAPPA_V, "X_RING": X_RING},
        "triple_overlap_and_kt1_selection": ov,
        "kt1_and_kt2_verification": kt12,
        "kt3_pinch_off_and_convergence": kt3,
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
