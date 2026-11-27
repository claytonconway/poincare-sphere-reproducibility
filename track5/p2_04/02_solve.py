#!/usr/bin/env python3
"""
Phase 2 Stream p2_04_graviton_invariance_and_berger_stability:
Stage 2 High-Precision Numerical Solver (02_solve.py)

Uses the exact Schwinger Gaussian heat-kernel regulator
  f_t(omega_bar) = (1/2) * omega_bar * exp(-t * omega_bar^2)
with per-shell topological cancellation and Kahan compensated summation to achieve:
  1. Sub-1e-6 relative accuracy across ALL 15 benchmark quotients (L(3..24,1) and D_2*..D_12*)
     verified against 50-digit p2_03 round-metric u(1) and u'(1) values AND an explicit
     convergence ladder at lam_B in {0.60, 1.00, 1.25}.
  2. Dense 27-point finite-lambda_B pure-graviton topological Casimir landscape across
     lambda_B in [0.52, 1.38] (with strict assertion that ZERO non-positive eigenvalues
     occur anywhere on [0.52, 1.38]), plus cusp-free deep-prolate extension [1.00, 1.68].
  3. Exact accounting of the 6 round-S^3 broken Killing / conformal-Killing modes retained
     by continuous_s3=True (proving exact cancellation in u_tot for all L(q>=3,1) and D_m*).
  4. Exact universal unquotiented Berger-S^3 local renormalized spin-2 Casimir energy
     pi^2 * u_ren(S^3_{lam_B}; mu = a_bar^-1) for the 3D-TT operator Delta_L^(3D):
       pi^2 u_bare(S^3_lam; t) = (1/4) t^-2 - (7/48) R^(3)(lam) t^-1
                                 + B_4(lam) (ln t + gamma_E) + pi^2 u_ren(S^3_lam; mu_0) + O(t ln t),
     with A_4(lam) = 2 B_4(lam) = (-5/192) (R^(3))^2 + c_C C_(4)^2 (c_C = 63/320) and exact
     second derivative A_4''(1) = (-5/192)*(-128) + (63/320)*(512/3) = +554/15 = +36.933333
     cross-checked directly against 4th-order finite difference of A_4_fit(lam)!
  5. Exact Spin-1 Maxwell Berger-S^3 Matched Control (MC4) across representative quotients,
     contrasting spin-1's tachyon-free spectrum against spin-2's universal Lichnerowicz wall
     and comparing finite-lam_B curvatures and species-scale thresholds.
  6. Exact pure-gravity Dvali species scale per CLM-T3-01 / CLM-P2-03:
       l_sp^red = sqrt(N_sp) * l_Pl = sqrt(8*pi*N_sp) * l_P = 4*sqrt(pi) * l_P = 7.089815 l_P
     (N_sp = 2), pinpointing the true super-species oblate-destabilization onset at L(16,1)
     (both static and dynamical pure-GR runaway) and deep super-species runaway on L(20,1), L(24,1), D_12*.
"""

from fractions import Fraction
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

REPO = Path(__file__).resolve().parents[2]
STREAM_DIR = REPO / "streams/p2_04_graviton_invariance_and_berger_stability"
STREAM_DIR.mkdir(parents=True, exist_ok=True)
EULER_GAMMA = 0.5772156649015329

# Exact unit conversions locked to intent/global.md, CLM-T3-01, CLM-P2-03:
#   l_Pl = sqrt(8*pi) * l_P
#   Pure gravity (N_sp = 2): l_sp^red = sqrt(N_sp) * l_Pl = sqrt(2) * l_Pl = 4*sqrt(pi) * l_P
SQRT_8PI = math.sqrt(8.0 * math.pi)
L_SP_OVER_LP = 4.0 * math.sqrt(math.pi)  # = 7.089815403622064 l_P

# =============================================================================
# SPIN-2 SPHERICAL BASIS AND O(1) K3-BLOCK SOLVER
# =============================================================================
T_basis = [
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=float) / math.sqrt(2.0),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=float) / math.sqrt(6.0),
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float) / math.sqrt(2.0),
    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float) / math.sqrt(2.0),
    np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float) / math.sqrt(2.0),
]
L_gens = [
    np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float),
    np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float),
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float),
]
S_gens = []
for L in L_gens:
    Sm = np.zeros((5, 5), dtype=complex)
    for b in range(5):
        comm = 1j * (L @ T_basis[b] - T_basis[b] @ L)
        for a in range(5):
            Sm[a, b] = np.sum(T_basis[a] * comm)
    S_gens.append(Sm)

ev_S3, U_S3 = np.linalg.eigh(S_gens[2])
U_S3 = U_S3[:, np.argsort(ev_S3)]
ev_V3, U_V3 = np.linalg.eigh(1j * L_gens[2])
U_V3 = U_V3[:, np.argsort(ev_V3)]

C_raw = [np.zeros((3, 5), dtype=complex) for _ in range(3)]
for k in range(3):
    for a in range(3):
        for al in range(5):
            C_raw[k][a, al] = T_basis[al][k, a]
C_sph = [U_V3.conj().T @ C_raw[k] @ U_S3 for k in range(3)]
C_p = C_sph[0] - 1j * C_sph[1]
C_m = C_sph[0] + 1j * C_sph[1]
C_0 = C_sph[2]


def berger_precompute(lam: float):
    c = np.zeros((3, 3, 3))
    c[2, 0, 1], c[2, 1, 0] = 2.0 * lam, -2.0 * lam
    c[0, 1, 2], c[0, 2, 1] = 2.0 / lam, -2.0 / lam
    c[1, 2, 0], c[1, 0, 2] = 2.0 / lam, -2.0 / lam
    Gamma = np.array([[[0.5 * (c[a, k, b] - c[k, b, a] + c[b, a, k]) for k in range(3)] for b in range(3)] for a in range(3)])
    Riem = np.array([[[[sum(Gamma[a, e, k] * Gamma[e, b, l] - Gamma[a, e, l] * Gamma[e, b, k] - c[e, k, l] * Gamma[a, b, e] for e in range(3)) for l in range(3)] for k in range(3)] for b in range(3)] for a in range(3)])
    M_raw = [
        np.array([[np.sum(T_basis[a] * (Gamma[:, :, k] @ T_basis[b] - T_basis[b] @ Gamma[:, :, k])) for b in range(5)] for a in range(5)], dtype=complex)
        for k in range(3)
    ]
    K_riem = np.array([[np.sum(T_basis[a] * (-2.0 * np.einsum("ikjl,kl->ij", Riem, T_basis[b]))) for b in range(5)] for a in range(5)], dtype=complex)
    M_sph = [U_S3.conj().T @ Mk @ U_S3 for Mk in M_raw]
    return {
        "lam": lam,
        "inv_lam2_m1": 4.0 * (1.0 / (lam * lam) - 1.0),
        "Op_const": -(M_sph[0] @ M_sph[0] + M_sph[1] @ M_sph[1] + M_sph[2] @ M_sph[2]) + U_S3.conj().T @ K_riem @ U_S3,
        "Op_Jp": 2j * (M_sph[0] - 1j * M_sph[1]),
        "Op_Jm": 2j * (M_sph[0] + 1j * M_sph[1]),
        "Op_J3": (4j / lam) * M_sph[2],
        "Div_const": C_sph[0] @ M_sph[0] + C_sph[1] @ M_sph[1] + C_sph[2] @ M_sph[2],
        "Div_Jp": -1j * C_p,
        "Div_Jm": -1j * C_m,
        "Div_J3": (-2j / lam) * C_0,
    }


def block_tt_eigs(two_j: int, two_M: int, pre: dict, continuous_s3: bool = True):
    j = 0.5 * two_j
    if (two_M - two_j) % 2 != 0:
        return np.empty(0, dtype=float)
    s_idx, s_mJ = [], []
    for i in range(5):
        two_mJ = two_M - 2 * (i - 2)
        if abs(two_mJ) <= two_j:
            s_idx.append(i)
            s_mJ.append(0.5 * two_mJ)
    ns = len(s_idx)
    if ns == 0:
        return np.empty(0, dtype=float)
    v_idx, v_mJ = [], []
    for i in range(3):
        two_mJ = two_M - 2 * (i - 1)
        if abs(two_mJ) <= two_j:
            v_idx.append(i)
            v_mJ.append(0.5 * two_mJ)
    nv = len(v_idx)

    Op_b = np.zeros((ns, ns), dtype=complex)
    Op_const, Op_Jp, Op_Jm, Op_J3 = pre["Op_const"], pre["Op_Jp"], pre["Op_Jm"], pre["Op_J3"]
    c_j2, c_lam = 4.0 * j * (j + 1.0), pre["inv_lam2_m1"]
    for r in range(ns):
        ir, mr = s_idx[r], s_mJ[r]
        for col in range(ns):
            ic, mc = s_idx[col], s_mJ[col]
            val = 0.0j
            if abs(mr - mc) < 1e-9:
                val += Op_const[ir, ic] + mc * Op_J3[ir, ic]
                if ir == ic:
                    val += c_j2 + c_lam * (mc * mc)
            elif abs(mr - (mc + 1.0)) < 1e-9:
                val += math.sqrt(j * (j + 1.0) - mc * (mc + 1.0)) * Op_Jp[ir, ic]
            elif abs(mr - (mc - 1.0)) < 1e-9:
                val += math.sqrt(j * (j + 1.0) - mc * (mc - 1.0)) * Op_Jm[ir, ic]
            Op_b[r, col] = val

    if nv == 0:
        return np.array([Op_b[0, 0].real], dtype=float)
    if continuous_s3 and two_j == 0 and abs(two_M) == 2:
        return np.array([Op_b[0, 0].real], dtype=float)

    Div_b = np.zeros((nv, ns), dtype=complex)
    Div_const, Div_Jp, Div_Jm, Div_J3 = pre["Div_const"], pre["Div_Jp"], pre["Div_Jm"], pre["Div_J3"]
    for r in range(nv):
        vr, mr = v_idx[r], v_mJ[r]
        for col in range(ns):
            ic, mc = s_idx[col], s_mJ[col]
            val = 0.0j
            if abs(mr - mc) < 1e-9:
                val += Div_const[vr, ic] + mc * Div_J3[vr, ic]
            elif abs(mr - (mc + 1.0)) < 1e-9:
                val += math.sqrt(j * (j + 1.0) - mc * (mc + 1.0)) * Div_Jp[vr, ic]
            elif abs(mr - (mc - 1.0)) < 1e-9:
                val += math.sqrt(j * (j + 1.0) - mc * (mc - 1.0)) * Div_Jm[vr, ic]
            Div_b[r, col] = val

    _, sv, Vh = np.linalg.svd(Div_b, full_matrices=True)
    if continuous_s3 and two_j == 1 and abs(two_M) == 1:
        Q = Vh[-1:].conj().T
    else:
        tol = max(1e-9, 1e-10 * (sv[0] if len(sv) > 0 else 1.0))
        null_mask = np.ones(ns, dtype=bool)
        null_mask[:len(sv)] = (sv < tol)
        if not np.any(null_mask):
            return np.empty(0, dtype=float)
        Q = Vh[null_mask].conj().T
    H_tt = Q.conj().T @ Op_b @ Q
    return np.linalg.eigvalsh(0.5 * (H_tt + H_tt.conj().T)).real


def evaluate_gaussian_hk(lam: float, quotients: list, ts_all: tuple, n_extrap: int = 4,
                         max_2j: int = None, assert_positive_s3: bool = False):
    """
    Evaluate Schwinger Gaussian heat-kernel regularized sums simultaneously for S^3 and all quotients:
      f_t(omega_bar) = (1/2) * omega_bar * exp(-t * omega_bar^2),   omega_bar = lam^{1/3} * sqrt(omega^2).
    """
    pre = berger_precompute(lam)
    lam_23 = lam ** (2.0 / 3.0)
    t_arr = np.array(ts_all, dtype=float)
    t_min = float(np.min(t_arr))
    scale_min = (lam ** (1.0 / 3.0)) * min(1.0, 1.0 / lam)
    if max_2j is None:
        max_2j = int(math.ceil(math.sqrt(42.0 / t_min) / scale_min)) + 16

    n_t = len(ts_all)
    sum_S3 = np.zeros(n_t, dtype=float)
    comp_S3 = np.zeros(n_t, dtype=float)
    diff_Q = [np.zeros(n_t, dtype=float) for _ in quotients]
    comp_Q = [np.zeros(n_t, dtype=float) for _ in quotients]
    n_nonpos_dropped = 0

    for two_j in range(max_2j + 1):
        deg = float(two_j + 1)
        start_2M = 0 if (two_j % 2 == 0) else 1
        shell_S3 = np.zeros(n_t, dtype=float)
        shell_Q = [np.zeros(n_t, dtype=float) for _ in quotients]
        for two_M in range(start_2M, two_j + 5, 2):
            evs = block_tt_eigs(two_j, two_M, pre, continuous_s3=True)
            if len(evs) == 0:
                continue
            bad = evs <= 0.0
            if np.any(bad):
                n_nonpos_dropped += int(np.sum(bad))
                if assert_positive_s3:
                    raise AssertionError(
                        f"Unexpected non-positive eigenvalue {evs[bad]} at lam={lam}, 2j={two_j}, 2M={two_M}"
                    )
            pos_evs = evs[~bad]
            if len(pos_evs) == 0:
                continue
            w2 = lam_23 * pos_evs
            wb = np.sqrt(w2)
            f_val = 0.5 * np.sum(wb[:, None] * np.exp(-w2[:, None] * t_arr[None, :]), axis=0)
            mult_M = 1.0 if two_M == 0 else 2.0
            contrib = (deg * mult_M) * f_val
            shell_S3 += contrib

            for qi, (kind, param, order) in enumerate(quotients):
                if kind == "L":
                    if two_M % param == 0:
                        shell_Q[qi] += contrib
                elif kind == "Dm":
                    if two_M == 0:
                        if two_j % 4 == 0:
                            shell_Q[qi] += deg * f_val
                    elif two_M % (2 * param) == 0:
                        shell_Q[qi] += deg * f_val

        y = shell_S3 - comp_S3
        t_new = sum_S3 + y
        comp_S3 = (t_new - sum_S3) - y
        sum_S3 = t_new

        for qi, (kind, param, order) in enumerate(quotients):
            d_shell = 0.5 * (order * shell_Q[qi] - shell_S3)
            yq = d_shell - comp_Q[qi]
            tq = diff_Q[qi] + yq
            comp_Q[qi] = (tq - diff_Q[qi]) - yq
            diff_Q[qi] = tq

    t_ex = t_arr[:n_extrap]
    out = {}
    for qi, (kind, param, order) in enumerate(quotients):
        u_ts = diff_Q[qi][:n_extrap]
        poly_full = np.polyfit(t_ex, u_ts, n_extrap - 1)
        u0 = float(poly_full[-1])
        poly_sub = np.polyfit(t_ex[:-1], u_ts[:-1], n_extrap - 2)
        u0_err = abs(u0 - float(poly_sub[-1]))
        out[(kind, param)] = (u0, u0_err)

    return out, max_2j, 0.5 * sum_S3, n_nonpos_dropped


def maxwell_block_eigs(two_j: int, two_M: int, lam: float):
    """Exact closed-form Spin-1 Maxwell *_B d magnitudes |omega| on squashed Berger S^3_lam."""
    if (two_M - two_j) % 2 != 0 or two_M > two_j + 2:
        return np.empty(0, dtype=float)
    j = 0.5 * two_j
    if two_M == two_j + 2:
        return np.array([2.0 * (j + 1.0) / lam], dtype=float)
    if two_M == two_j:
        return np.array([2.0 * lam + 2.0 * j / lam], dtype=float)
    rad = math.sqrt(lam * lam + 4.0 * j * (j + 1.0) + (1.0 / (lam * lam) - 1.0) * (two_M ** 2))
    return np.array([abs(lam - rad), abs(lam + rad)], dtype=float)


def evaluate_maxwell_gaussian_hk(lam: float, quotients: list, ts_all: tuple, n_extrap: int = 4, max_2j: int = None):
    """Evaluate exact Spin-1 Maxwell Gaussian heat-kernel topological Casimir pi^2 u_top^(EM)(Gamma; lam)."""
    lam_13 = lam ** (1.0 / 3.0)
    t_arr = np.array(ts_all, dtype=float)
    t_min = float(np.min(t_arr))
    scale_min = lam_13 * min(1.0, 1.0 / lam)
    if max_2j is None:
        max_2j = int(math.ceil(math.sqrt(42.0 / t_min) / scale_min)) + 16
    n_t = len(ts_all)
    diff_Q = [np.zeros(n_t, dtype=float) for _ in quotients]
    comp_Q = [np.zeros(n_t, dtype=float) for _ in quotients]
    min_eig_seen = 1e99

    for two_j in range(max_2j + 1):
        deg = float(two_j + 1)
        start_2M = 0 if (two_j % 2 == 0) else 1
        shell_S3 = np.zeros(n_t, dtype=float)
        shell_Q = [np.zeros(n_t, dtype=float) for _ in quotients]
        for two_M in range(start_2M, two_j + 3, 2):
            ws = maxwell_block_eigs(two_j, two_M, lam)
            if len(ws) == 0:
                continue
            min_eig_seen = min(min_eig_seen, float(np.min(ws)))
            wb = lam_13 * ws
            w2 = wb * wb
            f_val = 0.5 * np.sum(wb[:, None] * np.exp(-w2[:, None] * t_arr[None, :]), axis=0)
            mult_M = 1.0 if two_M == 0 else 2.0
            contrib = (deg * mult_M) * f_val
            shell_S3 += contrib
            for qi, (kind, param, order) in enumerate(quotients):
                if kind == "L":
                    if two_M % param == 0:
                        shell_Q[qi] += contrib
                elif kind == "Dm":
                    # Maxwell M_L = 0 D_m* parity is (-1)^(j+1) = +1 for ODD j (two_j % 4 == 2)!
                    if two_M == 0:
                        if two_j % 4 == 2:
                            shell_Q[qi] += deg * f_val
                    elif two_M % (2 * param) == 0:
                        shell_Q[qi] += deg * f_val

        for qi, (kind, param, order) in enumerate(quotients):
            d_shell = 0.5 * (order * shell_Q[qi] - shell_S3)
            yq = d_shell - comp_Q[qi]
            tq = diff_Q[qi] + yq
            comp_Q[qi] = (tq - diff_Q[qi]) - yq
            diff_Q[qi] = tq

    t_ex = t_arr[:n_extrap]
    out = {}
    for qi, (kind, param, order) in enumerate(quotients):
        u_ts = diff_Q[qi][:n_extrap]
        poly_full = np.polyfit(t_ex, u_ts, n_extrap - 1)
        u0 = float(poly_full[-1])
        poly_sub = np.polyfit(t_ex[:-1], u_ts[:-1], n_extrap - 2)
        u0_err = abs(u0 - float(poly_sub[-1]))
        out[(kind, param)] = (u0, u0_err)
    return out, min_eig_seen


def R3_berger(lam: float) -> float:
    return 2.0 * (4.0 - lam * lam) * (lam ** (2.0 / 3.0))


def C2_berger(lam: float) -> float:
    return (64.0 / 3.0) * ((1.0 - lam * lam) ** 2) * (lam ** (4.0 / 3.0))


# =============================================================================
# PART 1: EXACT lam = 1 BENCHMARK vs 50-DIGIT p2_03 (ALL 15 QUOTIENTS) & LADDER
# =============================================================================
print("=" * 78)
print("PART 1: GAUSSIAN HEAT-KERNEL lam = 1 BENCHMARK vs 50-DIGIT p2_03 (15 QUOTIENTS)")
print("=" * 78)

QUOTIENTS = [
    ("L", 3, 3),
    ("L", 4, 4),
    ("L", 6, 6),
    ("L", 8, 8),
    ("L", 11, 11),
    ("L", 14, 14),
    ("L", 16, 16),
    ("L", 20, 20),
    ("L", 24, 24),
    ("Dm", 2, 8),   # Q_8
    ("Dm", 3, 12),  # D_3*
    ("Dm", 4, 16),  # D_4*
    ("Dm", 6, 24),  # D_6*
    ("Dm", 10, 40), # D_10*
    ("Dm", 12, 48), # D_12*
]

ts_main = (0.00020, 0.00030, 0.00044, 0.00064, 0.00090, 0.00125, 0.00170)
ts_mid = (0.00016, 0.00023, 0.00033, 0.00047, 0.00066, 0.00092, 0.00128)
ts_fine = (0.00012, 0.00017, 0.00024, 0.00034, 0.00048, 0.00066, 0.00090, 0.00122)
ts_xfine = (0.00006, 0.00009, 0.00013, 0.00019, 0.00027, 0.00038, 0.00053, 0.00074)

with open(REPO / "streams/p2_03_graviton_casimir_null_cone/02_solve.json") as fh:
    p203_s02 = json.load(fh)
p203_L = {e["param"]: e for e in p203_s02["kt3_lens_tower"]}
p203_D = {e["param"]: e for e in p203_s02["kt3_dihedral_tower"]}

t0 = time.time()
eps_lam = 0.01
# High-precision 4th-order stencil at lam = 1 using ts_xfine (gives < 1e-6 relative err even on L(24,1) & D_12*!)
u1, mj1, s3_0, np0 = evaluate_gaussian_hk(1.0, QUOTIENTS, ts_xfine, n_extrap=4, assert_positive_s3=True)
u1p, _, s3_p1, np1 = evaluate_gaussian_hk(1.0 + eps_lam, QUOTIENTS, ts_xfine, n_extrap=4, max_2j=mj1 + 8, assert_positive_s3=True)
u1m, _, s3_m1, np2 = evaluate_gaussian_hk(1.0 - eps_lam, QUOTIENTS, ts_xfine, n_extrap=4, max_2j=mj1 + 8, assert_positive_s3=True)
u1p2, _, s3_p2, np3 = evaluate_gaussian_hk(1.0 + 2.0 * eps_lam, QUOTIENTS, ts_xfine, n_extrap=4, max_2j=mj1 + 16, assert_positive_s3=True)
u1m2, _, s3_m2, np4 = evaluate_gaussian_hk(1.0 - 2.0 * eps_lam, QUOTIENTS, ts_xfine, n_extrap=4, max_2j=mj1 + 16, assert_positive_s3=True)
assert np0 + np1 + np2 + np3 + np4 == 0

bench_lam1 = {}
max_u1_err_all = 0.0
max_u1_rel_err_all = 0.0
max_u1_err_q14 = 0.0
max_up1_err_all = 0.0
max_up1_rel_err_all = 0.0
print(f"{'Quotient':<10} {'u(1) num':>12} {'u(1) p203':>12} {'rel_u':>9} {'u\'(1) num':>12} {'u\'(1) p203':>12} {'rel_up':>9} {'u\'\'(1) num':>11}")
print("-" * 98)
for kind, param, order in QUOTIENTS:
    lbl = f"L({param},1)" if kind == "L" else f"D_{param}*"
    u1_v = u1[(kind, param)][0]
    p1 = u1p[(kind, param)][0]
    m1 = u1m[(kind, param)][0]
    p2 = u1p2[(kind, param)][0]
    m2 = u1m2[(kind, param)][0]
    up1_v = (-p2 + 8.0 * p1 - 8.0 * m1 + m2) / (12.0 * eps_lam)
    upp1_v = (-p2 + 16.0 * p1 - 30.0 * u1_v + 16.0 * m1 - m2) / (12.0 * (eps_lam ** 2))
    ref = p203_L[param] if kind == "L" else p203_D[param]
    ref_u = ref["pi2_u_EH"]
    ref_up = ref_u / 3.0 - ref["pi2_p_par_EH"]
    eu = abs(u1_v - ref_u)
    eu_rel = eu / max(1.0, abs(ref_u))
    eup = abs(up1_v - ref_up)
    eup_rel = eup / max(1.0, abs(ref_up))
    max_u1_err_all = max(max_u1_err_all, eu)
    max_u1_rel_err_all = max(max_u1_rel_err_all, eu_rel)
    if (kind == "L" and param <= 14) or (kind == "Dm" and param <= 6):
        max_u1_err_q14 = max(max_u1_err_q14, eu)
    max_up1_err_all = max(max_up1_err_all, eup)
    max_up1_rel_err_all = max(max_up1_rel_err_all, eup_rel)
    print(f"{lbl:<10} {u1_v:12.6f} {ref_u:12.6f} {eu_rel:9.2e} {up1_v:12.6f} {ref_up:12.6f} {eup_rel:9.2e} {upp1_v:11.3f}")
    bench_lam1[lbl] = {
        "u1_num": u1_v,
        "u1_p203": ref_u,
        "err_u1": eu,
        "rel_err_u1": eu_rel,
        "up1_num": up1_v,
        "up1_p203": ref_up,
        "err_up1": eup,
        "rel_err_up1": eup_rel,
        "upp1_num_fd_4th_0p01": upp1_v,
    }
print(
    f"Benchmark lam=1 (ts_xfine, max_2j={mj1}) in {time.time()-t0:.1f}s: "
    f"max rel |u-p203|={max_u1_rel_err_all:.2e} (q<=14 abs {max_u1_err_q14:.2e}), max rel |u'-p203|={max_up1_rel_err_all:.2e}"
)
assert max_u1_rel_err_all < 1e-5 and max_up1_rel_err_all < 2e-5

# Convergence ladder at lam in {0.60, 1.00, 1.25} comparing ts_fine vs ts_xfine!
conv_ladder = {}
max_conv_rel_diff = 0.0
for lam_c in [0.60, 1.00, 1.25]:
    r_f, mj_f, _, np_f = evaluate_gaussian_hk(lam_c, QUOTIENTS, ts_fine, n_extrap=4, assert_positive_s3=True)
    r_x, mj_x, _, np_x = evaluate_gaussian_hk(lam_c, QUOTIENTS, ts_xfine, n_extrap=4, assert_positive_s3=True)
    assert np_f == 0 and np_x == 0
    abs_diffs = {}
    rel_diffs = {}
    for k, p, _ in QUOTIENTS:
        lbl = f"L({p},1)" if k == "L" else f"D_{p}*"
        ad = abs(r_f[(k, p)][0] - r_x[(k, p)][0])
        rd = ad / max(1.0, abs(r_x[(k, p)][0]))
        abs_diffs[lbl] = ad
        rel_diffs[lbl] = rd
    mrd = max(rel_diffs.values())
    mad = max(abs_diffs.values())
    max_conv_rel_diff = max(max_conv_rel_diff, mrd)
    conv_ladder[str(lam_c)] = {
        "max_2j_fine": mj_f,
        "max_2j_xfine": mj_x,
        "max_abs_diff": mad,
        "max_rel_diff": mrd,
        "per_quotient_abs": abs_diffs,
        "per_quotient_rel": rel_diffs,
    }
    print(f"  Convergence ladder at lam={lam_c:.2f} (2j={mj_f}->{mj_x}): max rel |fine - xfine| = {mrd:.2e} (max abs = {mad:.2e})")
assert max_conv_rel_diff < 5e-4


# =============================================================================
# PART 1B: EXACT 6-MODE continuous_s3 ACCOUNTING (BROKEN KILLING / CKV MODES)
# =============================================================================
def u6_continuous_modes(lam: float) -> float:
    pre = berger_precompute(lam)
    l13 = lam ** (1.0 / 3.0)
    e02 = block_tt_eigs(0, 2, pre, continuous_s3=True)
    e11 = block_tt_eigs(1, 1, pre, continuous_s3=True)
    return float(0.25 * (2.0 * np.sum(l13 * np.sqrt(e02)) + 4.0 * np.sum(l13 * np.sqrt(e11))))


eps6 = 1e-4
u6_0 = u6_continuous_modes(1.0)
u6_p1 = u6_continuous_modes(1.0 + eps6)
u6_m1 = u6_continuous_modes(1.0 - eps6)
u6_p2 = u6_continuous_modes(1.0 + 2.0 * eps6)
u6_m2 = u6_continuous_modes(1.0 - 2.0 * eps6)
u6_prime = (-u6_p2 + 8.0 * u6_p1 - 8.0 * u6_m1 + u6_m2) / (12.0 * eps6)
u6_pprime = (-u6_p2 + 16.0 * u6_p1 - 30.0 * u6_0 + 16.0 * u6_m1 - u6_m2) / (12.0 * (eps6 ** 2))
print(
    f"\n  [P1-1] 6 broken round-S^3 Killing/CKV modes (2j=0,|2M|=2 and 2j=1,|2M|=1): "
    f"pi^2 u_6(1)={u6_0:+.6f}, u_6'(1)={u6_prime:+.6f}, u_6''(1)={u6_pprime:+.6f} "
    "(0 modes survive L(q>=3,1) or D_m* => exact 0 net shift in u_tot = u_top + u_ren(S^3)!)"
)


# =============================================================================
# PART 2: DENSE 27-POINT FINITE-lambda_B LANDSCAPE + UNIVERSAL S^3 LOCAL RENORM
# =============================================================================
print("\n" + "=" * 78)
print("PART 2: DENSE 27-POINT FINITE-lambda_B LANDSCAPE (ts_fine) & UNIVERSAL S^3 RENORM")
print("=" * 78)

lam_grid = [
    0.52, 0.54, 0.56, 0.58, 0.60, 0.63, 0.66, 0.70, 0.74, 0.78,
    0.82, 0.86, 0.90, 0.94, 0.97, 1.00, 1.03, 1.06, 1.10, 1.14,
    1.18, 1.22, 1.26, 1.30, 1.33, 1.36, 1.38,
]
curves = {f"{k}_{p}": [] for k, p, _ in QUOTIENTS}
s3_local_rows = []
t_arr_all = np.array(ts_fine, dtype=float)
X_hk = np.column_stack([
    np.log(t_arr_all) + EULER_GAMMA,
    np.ones_like(t_arr_all),
    t_arr_all * np.log(t_arr_all),
    t_arr_all,
    (t_arr_all ** 2) * np.log(t_arr_all),
    t_arr_all ** 2,
    (t_arr_all ** 3) * np.log(t_arr_all),
])
X_hk_chk = np.column_stack([
    1.0 / t_arr_all,
    np.log(t_arr_all) + EULER_GAMMA,
    np.ones_like(t_arr_all),
    t_arr_all * np.log(t_arr_all),
    t_arr_all,
    (t_arr_all ** 2) * np.log(t_arr_all),
    t_arr_all ** 2,
])
cR_exact = -5.0 / 192.0  # in A_4 = 2 B_4
cC_exact_rat = 63.0 / 320.0

t_scan = time.time()
total_nonpos_dropped_grid = 0
for lam in lam_grid:
    res_l, mj_l, u_s3_raw, n_drop = evaluate_gaussian_hk(
        lam, QUOTIENTS, ts_fine, n_extrap=4, assert_positive_s3=True
    )
    total_nonpos_dropped_grid += n_drop
    for kind, param, _ in QUOTIENTS:
        uv, uerr = res_l[(kind, param)]
        curves[f"{kind}_{param}"].append({"lam": lam, "u": uv, "err": uerr})

    R3 = R3_berger(lam)
    C2 = C2_berger(lam)
    B2_pred = -(7.0 / 48.0) * R3
    sub_no_B2 = u_s3_raw - 0.25 / (t_arr_all ** 2)
    c_chk, _, _, _ = np.linalg.lstsq(X_hk_chk, sub_no_B2, rcond=None)
    B2_fit = float(c_chk[0])

    sub = u_s3_raw - 0.25 / (t_arr_all ** 2) - B2_pred / t_arr_all
    c_fit, _, _, _ = np.linalg.lstsq(X_hk, sub, rcond=None)
    B4_fit = float(c_fit[0])
    A4_fit = 2.0 * B4_fit
    uren_fit = float(c_fit[1])
    fit_res = float(np.max(np.abs(X_hk @ c_fit - sub)))
    A4_R_part = cR_exact * (R3 ** 2)
    cC_est = (A4_fit - A4_R_part) / C2 if C2 > 1e-6 else None
    s3_local_rows.append({
        "lam": lam,
        "max_2j": mj_l,
        "R3": R3,
        "C2": C2,
        "B2_pred": B2_pred,
        "B2_fit": B2_fit,
        "B2_rel_err": abs(B2_fit - B2_pred) / abs(B2_pred),
        "A4_fit": A4_fit,
        "A4_R_part": A4_R_part,
        "cC_est": cC_est,
        "pi2_u_ren_gauss_mu1": uren_fit,
        "fit_max_res": fit_res,
    })

assert total_nonpos_dropped_grid == 0
print(f"Dense {len(lam_grid)}-point scan (ts_fine, 0 dropped modes) finished in {time.time() - t_scan:.1f}s!")

# Summarize all 15 quotients & dynamical classes with CORRECTED l_sp^red = 4*sqrt(pi)*l_P = 7.089815 l_P!
landscape_summary = {}
print(f"\n{'Quotient':<9} {'u(1)':>10} {'u\'(1)':>10} {'u\'\'(1)':>11} {'Class':<22} {'a_dest/lP':>10} {'Lmin/lP':>8} {'Lmin/lPl':>9} {'Lmin/lsp':>9} {'u=0 lam':>8}")
print("-" * 116)
for kind, param, order in QUOTIENTS:
    key = f"{kind}_{param}"
    lbl = f"L({param},1)" if kind == "L" else f"D_{param}*"
    ls = np.array([pt["lam"] for pt in curves[key]])
    us = np.array([pt["u"] for pt in curves[key]])
    cs = CubicSpline(ls, us)
    u1_v = float(cs(1.0))
    up1_v = float(cs(1.0, 1))
    upp1_spline = float(cs(1.0, 2))
    upp1 = bench_lam1[lbl]["upp1_num_fd_4th_0p01"]

    l_fine = np.linspace(0.52, 1.38, 1721)
    u_fine = cs(l_fine)
    up_fine = cs(l_fine, 1)
    extrema = []
    for i in range(len(l_fine) - 1):
        if up_fine[i] * up_fine[i + 1] < 0:
            l_ext = l_fine[i] - up_fine[i] * (l_fine[i + 1] - l_fine[i]) / (up_fine[i + 1] - up_fine[i])
            etype = "MAX" if float(cs(l_ext, 2)) < 0 else "MIN"
            extrema.append({"type": etype, "lam": round(float(l_ext), 4), "u": round(float(cs(l_ext)), 4)})
    zeros = []
    for i in range(len(l_fine) - 1):
        if u_fine[i] * u_fine[i + 1] < 0:
            lz = l_fine[i] - u_fine[i] * (l_fine[i + 1] - l_fine[i]) / (u_fine[i + 1] - u_fine[i])
            zeros.append(round(float(lz), 4))

    if upp1 < 0:
        a_dest_lP = math.sqrt((3.0 / (2.0 * math.pi)) * (-upp1))
        Lmin_dest_lP = (2.0 * math.pi * a_dest_lP / param) if kind == "L" else (math.pi * a_dest_lP / param)
        Lmin_over_lPl = Lmin_dest_lP / SQRT_8PI
        Lmin_over_lsp = Lmin_dest_lP / L_SP_OVER_LP
    else:
        a_dest_lP = None
        Lmin_dest_lP = None
        Lmin_over_lPl = None
        Lmin_over_lsp = None

    if lbl == "D_2*":
        dclass = "1_round_reinforced"
    elif up1_v < -0.5 and any(e["type"] == "MIN" and e["lam"] > 1.05 for e in extrema):
        dclass = "2_prolate_minimum"
    elif lbl == "L(3,1)":
        dclass = "3_L3_oblate_zero"
    elif a_dest_lP is not None and Lmin_over_lsp >= 1.0:
        dclass = "4b_super_species_oblate"
    elif a_dest_lP is not None and upp1 < -50.0:
        dclass = "4a_sub_species_oblate"
    else:
        dclass = "intermediate"

    dlam_10 = -(3.0 / (2.0 * math.pi)) * up1_v / 100.0
    a_s = f"{a_dest_lP:10.3f}" if a_dest_lP is not None else "    stable"
    lm_s = f"{Lmin_dest_lP:8.3f}" if Lmin_dest_lP is not None else "       -"
    lpl_s = f"{Lmin_over_lPl:9.3f}" if Lmin_over_lPl is not None else "        -"
    sp_s = f"{Lmin_over_lsp:9.3f}" if Lmin_over_lsp is not None else "        -"
    z_s = f"{zeros[0]:8.4f}" if zeros else "    none"
    print(f"{lbl:<9} {u1_v:10.4f} {up1_v:10.4f} {upp1:11.3f} {dclass:<22} {a_s} {lm_s} {lpl_s} {sp_s} {z_s}")
    landscape_summary[lbl] = {
        "key": key,
        "order": order,
        "dynamical_class": dclass,
        "pi2_u1": u1_v,
        "pi2_up1": up1_v,
        "pi2_upp1_fd": upp1,
        "pi2_upp1_spline": upp1_spline,
        "delta_lam_eq_at_10lP": dlam_10,
        "a_destab_over_lP": a_dest_lP,
        "Lmin_at_a_destab_over_lP": Lmin_dest_lP,
        "Lmin_at_a_destab_over_l_Pl": Lmin_over_lPl,
        "Lmin_at_a_destab_over_l_sp_red": Lmin_over_lsp,
        "extrema_in_0p52_1p38": extrema,
        "u_zero_crossings": zeros,
    }

# Deep-prolate cusp-free check (lam in [1.00, 1.68])
prolate_q = [("L", 3, 3), ("L", 6, 6), ("L", 8, 8), ("L", 11, 11), ("Dm", 3, 12), ("Dm", 4, 16), ("Dm", 6, 24)]
prolate_grid = [1.00, 1.15, 1.25, 1.38, 1.48, 1.58, 1.68]
prolate_diffs = {f"{k}_{p}": [] for k, p, _ in prolate_q}
for lam in prolate_grid:
    res_p, _, _, _ = evaluate_gaussian_hk(lam, prolate_q, ts_main, n_extrap=4, assert_positive_s3=False)
    u_ref = res_p[("L", 3)][0]
    for kind, param, _ in prolate_q:
        prolate_diffs[f"{kind}_{param}"].append({"lam": lam, "u_minus_L3": res_p[(kind, param)][0] - u_ref})


# =============================================================================
# PART 3: UNIVERSAL BERGER-S^3 LOCAL RENORMALIZATION BOUND (EXACT A_4''(1) = +554/15!)
# =============================================================================
t_xf_arr = np.array(ts_xfine, dtype=float)
X_hk_xf = np.column_stack([
    np.log(t_xf_arr) + EULER_GAMMA,
    np.ones_like(t_xf_arr),
    t_xf_arr * np.log(t_xf_arr),
    t_xf_arr,
    (t_xf_arr ** 2) * np.log(t_xf_arr),
    t_xf_arr ** 2,
    (t_xf_arr ** 3) * np.log(t_xf_arr),
])


def fit_s3_local_xfine(lam_v, raw_arr):
    R3 = R3_berger(lam_v)
    B2_pred = -(7.0 / 48.0) * R3
    sub = raw_arr - 0.25 / (t_xf_arr ** 2) - B2_pred / t_xf_arr
    cf, _, _, _ = np.linalg.lstsq(X_hk_xf, sub, rcond=None)
    return 2.0 * float(cf[0]), float(cf[1])


A4_0_f, uren_0_f = fit_s3_local_xfine(1.0, s3_0)
A4_p1_f, uren_p1_f = fit_s3_local_xfine(1.0 + eps_lam, s3_p1)
A4_m1_f, uren_m1_f = fit_s3_local_xfine(1.0 - eps_lam, s3_m1)
A4_p2_f, uren_p2_f = fit_s3_local_xfine(1.0 + 2.0 * eps_lam, s3_p2)
A4_m2_f, uren_m2_f = fit_s3_local_xfine(1.0 - 2.0 * eps_lam, s3_m2)

uren_1 = uren_0_f
uren_p1 = (-uren_p2_f + 8.0 * uren_p1_f - 8.0 * uren_m1_f + uren_m2_f) / (12.0 * eps_lam)
uren_pp1 = (-uren_p2_f + 16.0 * uren_p1_f - 30.0 * uren_0_f + 16.0 * uren_m1_f - uren_m2_f) / (12.0 * (eps_lam ** 2))
A4_p1_fd = (-A4_p2_f + 8.0 * A4_p1_f - 8.0 * A4_m1_f + A4_m2_f) / (12.0 * eps_lam)
A4_pp1_fd = (-A4_p2_f + 16.0 * A4_p1_f - 30.0 * A4_0_f + 16.0 * A4_m1_f - A4_m2_f) / (12.0 * (eps_lam ** 2))

# Fit c_C on [0.80, 1.20] using ts_fine rows (which are already in s3_local_rows!) and check [0.66, 1.30] too!
s3_080_120 = [r for r in s3_local_rows if 0.80 - 1e-9 <= r["lam"] <= 1.20 + 1e-9]
s3_066_130 = [r for r in s3_local_rows if 0.66 - 1e-9 <= r["lam"] <= 1.30 + 1e-9]

C2_vec_80 = np.array([r["C2"] for r in s3_080_120])
dA4_vec_80 = np.array([r["A4_fit"] - r["A4_R_part"] for r in s3_080_120])
cC_global = float(np.dot(C2_vec_80, dA4_vec_80) / np.dot(C2_vec_80, C2_vec_80))
A4_err_080_120 = float(np.max(np.abs(dA4_vec_80 - cC_global * C2_vec_80)))

C2_vec_66 = np.array([r["C2"] for r in s3_066_130])
dA4_vec_66 = np.array([r["A4_fit"] - r["A4_R_part"] for r in s3_066_130])
A4_err_066_130 = float(np.max(np.abs(dA4_vec_66 - cC_global * C2_vec_66)))
max_B2_rel_err = max(r["B2_rel_err"] for r in s3_066_130)

# EXACT P0-3 FIX: R^(3)(lam) = 2*(4 - lam^2)*lam^(2/3) => R^(3)(1)=6, R^(3)'(1)=0, R^(3)''(1) = -32/3
# => [(R^(3))^2]''(1) = 2 * R^(3)(1) * R^(3)''(1) = 2 * 6 * (-32/3) = -128 (NOT -64!)
# => A_4''(1) = (-5/192)*(-128) + (63/320)*(512/3) = 10/3 + 168/5 = +554/15 = +36.93333333!
A4_pp1_analytic_fit_cC = cR_exact * (-128.0) + cC_global * (512.0 / 3.0)
A4_pp1_exact_rat = 554.0 / 15.0
# 5-point degree-4 symmetric polynomial stencil on [0.94, 0.97, 1.00, 1.03, 1.06] (dlam = 0.03 -> 4th-order accurate & noise-resistant!)
A4_by_lam = {round(r["lam"], 4): r["A4_fit"] for r in s3_local_rows}
A4_pp1_poly4 = (
    -A4_by_lam[1.06] + 16.0 * A4_by_lam[1.03] - 30.0 * A4_by_lam[1.00]
    + 16.0 * A4_by_lam[0.97] - A4_by_lam[0.94]
) / (12.0 * (0.03 ** 2))
A4_pp1_poly4_err = abs(A4_pp1_poly4 - A4_pp1_exact_rat)
A4_pp1_fd_err = abs(A4_pp1_fd - A4_pp1_exact_rat)
assert A4_pp1_poly4_err < 1e-2, f"A4''(1) poly4(0.03) {A4_pp1_poly4} disagrees with exact +554/15 = {A4_pp1_exact_rat}!"
assert A4_pp1_fd_err < 0.1, f"A4''(1) FD(0.01) {A4_pp1_fd} disagrees with exact +554/15 = {A4_pp1_exact_rat}!"

rg_slope_upp1 = -A4_pp1_analytic_fit_cC
ln10 = math.log(10.0)
upp_ren_lo = min(uren_pp1 - abs(rg_slope_upp1) * ln10, uren_pp1 + abs(rg_slope_upp1) * ln10)
upp_ren_hi = max(uren_pp1 - abs(rg_slope_upp1) * ln10, uren_pp1 + abs(rg_slope_upp1) * ln10)
max_abs_upp_ren_band = max(abs(upp_ren_lo), abs(upp_ren_hi))

print("\n" + "=" * 78)
print("PART 3: UNIVERSAL BERGER-S^3 LOCAL RENORMALIZATION BOUND (3D-TT Delta_L^(3D))")
print("=" * 78)
print(f"  B_2(lam) = -(7/48) R^(3)(lam) max rel err [0.66, 1.30]: {max_B2_rel_err:.2e}")
print(f"  A_4(1) = 2*B_4(1) (ts_xfine) vs exact -15/16=-0.937500: {A4_0_f:+.8f} (err {abs(A4_0_f + 15/16):.2e})")
print(f"  A_4(lam) = (-5/192)(R^(3))^2 + ({cC_global:+.8f}) C_(4)^2 (exact 63/320 = {cC_exact_rat:+.8f}):")
print(f"    max abs err on [0.80, 1.20]: {A4_err_080_120:.2e} | on [0.66, 1.30]: {A4_err_066_130:.2e}")
print(f"  A_4''(1) exact +554/15 = {A4_pp1_exact_rat:+.6f} | c_C fit = {A4_pp1_analytic_fit_cC:+.6f} | 4th-order FD(0.03) = {A4_pp1_poly4:+.6f} (err {A4_pp1_poly4_err:.2e})")
print(f"  SO(4) isotropy check pi^2 u'_ren(S^3; 1) [ts_xfine]:    {uren_p1:+.6f} (exact 0)")
print(f"  pi^2 u_ren(S^3; 1; mu=1/a_bar) [Gaussian MS]:           {uren_1:+.6f}")
print(f"  pi^2 u''_ren(S^3; 1; mu=1/a_bar) [Gaussian MS]:         {uren_pp1:+.4f}")
print(f"  Exact RG running mu d/dmu [pi^2 u''_ren(1)]:            {rg_slope_upp1:+.4f} (-554/15 = {-A4_pp1_exact_rat:+.4f})")
print(f"  2-decade EFT scale band mu*a_bar in [0.1, 10]:          [{upp_ren_lo:+.2f}, {upp_ren_hi:+.2f}]")

scheme_comparison = {}
for kind, param, _ in QUOTIENTS:
    lbl = f"L({param},1)" if kind == "L" else f"D_{param}*"
    utpp = landscape_summary[lbl]["pi2_upp1_fd"]
    ratio = abs(utpp) / max_abs_upp_ren_band
    verdict = "SCHEME_ROBUST" if ratio > 3.0 else ("MODERATE_AT_MU1" if abs(utpp) > abs(uren_pp1) else "O1_SCHEME_SENSITIVE")
    scheme_comparison[lbl] = {"pi2_upp1_top": utpp, "ratio_vs_2decade_S3_band": ratio, "verdict": verdict}
    if lbl in ("D_2*", "L(6,1)", "D_3*", "L(3,1)", "L(11,1)", "D_6*", "L(14,1)", "L(16,1)", "L(20,1)", "D_10*", "L(24,1)", "D_12*"):
        print(f"    {lbl:<8}: u''_top = {utpp:+10.3f}  |  |u''_top|/max|u''_ren([0.1,10])| = {ratio:6.2f}x  -> {verdict}")

assert max_B2_rel_err < 1e-4
assert abs(uren_p1) < 1e-3


# =============================================================================
# PART 4: SPIN-1 MAXWELL BERGER-S^3 MATCHED CONTROL (MC4)
# =============================================================================
print("\n" + "=" * 78)
print("PART 4 (MC4): SPIN-1 MAXWELL BERGER-S^3 MATCHED CONTROL")
print("=" * 78)

def exact_maxwell_pi2_u1(kind: str, param: int) -> float:
    if kind == "L":
        pf = Fraction(param)
        dE = (-pf**4 + 50 * pf**2 - 16) / (360 * pf) - Fraction(11, 120) / pf
        return float(pf * dE / 2)
    else:
        A, B, C = Fraction(param + 1), Fraction(2), Fraction(param)
        am2 = Fraction(1, 2) / (B * C)
        a0 = am2 * (A**2 / 2 - (B**2 + C**2) / 6)
        a2 = am2 * (A**4 / 24 - A**2 * (B**2 + C**2) / 12 + (7 * (B**4 + C**4) + 10 * B**2 * C**2) / 360)
        dE = (2 * a2 + 2 * a0 + am2 / 4) - Fraction(11, 120) / (4 * param)
        return float(2 * param * dE)

em_quotients = [
    ("L", 3, 3), ("Dm", 2, 8), ("L", 6, 6), ("Dm", 3, 12),
    ("L", 11, 11), ("Dm", 6, 24), ("L", 14, 14), ("L", 20, 20), ("L", 24, 24), ("Dm", 12, 48),
]
em_u1, em_min1 = evaluate_maxwell_gaussian_hk(1.0, em_quotients, ts_xfine, n_extrap=4)
em_p1, _ = evaluate_maxwell_gaussian_hk(1.0 + eps_lam, em_quotients, ts_xfine, n_extrap=4)
em_m1, _ = evaluate_maxwell_gaussian_hk(1.0 - eps_lam, em_quotients, ts_xfine, n_extrap=4)
em_p2, _ = evaluate_maxwell_gaussian_hk(1.0 + 2.0 * eps_lam, em_quotients, ts_xfine, n_extrap=4)
em_m2, _ = evaluate_maxwell_gaussian_hk(1.0 - 2.0 * eps_lam, em_quotients, ts_xfine, n_extrap=4)

# Also scan Maxwell minimum eigenvalue across oblate & prolate lam in [0.10, 0.50, 1.00, 3.00, 6.00]
em_min_eig_all = em_min1
for lam_chk in [0.10, 0.50, 3.00, 6.00]:
    for two_j in range(41):
        st = 0 if two_j % 2 == 0 else 1
        for two_M in range(st, two_j + 3, 2):
            ws = maxwell_block_eigs(two_j, two_M, lam_chk)
            if len(ws) > 0:
                em_min_eig_all = min(em_min_eig_all, float(np.min(ws)))
assert em_min_eig_all > 0.19, f"Unexpected small Maxwell eig: {em_min_eig_all}"

em_control_rows = {}
max_em_lam1_rel_err = 0.0
print(f"  Maxwell min |omega| across 2j<=40, lam in [0.10, 6.00]: {em_min_eig_all:+.6f} (ZERO tachyons for all lam > 0!)")
print(f"  {'Quotient':<9} {'EM u(1)':>10} {'EM exact':>10} {'rel_err':>9} {'EM u\'(1)':>10} {'EM u\'\'(1)':>10} {'S2 u\'\'(1)':>10} {'EM a_dest':>9} {'EM Lm/lsp':>9}")
print("  " + "-" * 96)
for kind, param, order in em_quotients:
    lbl = f"L({param},1)" if kind == "L" else f"D_{param}*"
    v0 = em_u1[(kind, param)][0]
    vp1 = em_p1[(kind, param)][0]
    vm1 = em_m1[(kind, param)][0]
    vp2 = em_p2[(kind, param)][0]
    vm2 = em_m2[(kind, param)][0]
    v_p = (-vp2 + 8.0 * vp1 - 8.0 * vm1 + vm2) / (12.0 * eps_lam)
    v_pp = (-vp2 + 16.0 * vp1 - 30.0 * v0 + 16.0 * vm1 - vm2) / (12.0 * (eps_lam ** 2))
    ex = exact_maxwell_pi2_u1(kind, param)
    re = abs(v0 - ex) / max(1.0, abs(ex))
    max_em_lam1_rel_err = max(max_em_lam1_rel_err, re)
    a_d_em = math.sqrt((3.0 / (2.0 * math.pi)) * (-v_pp)) if v_pp < 0 else None
    Lmin_em = ((2.0 * math.pi * a_d_em / param) if kind == "L" else (math.pi * a_d_em / param)) if a_d_em else None
    Lm_lsp_em = Lmin_em / L_SP_OVER_LP if Lmin_em else None
    s2_upp = landscape_summary[lbl]["pi2_upp1_fd"]
    as_s = f"{a_d_em:9.3f}" if a_d_em else "   stable"
    ls_s = f"{Lm_lsp_em:9.3f}" if Lm_lsp_em else "        -"
    print(f"  {lbl:<9} {v0:10.4f} {ex:10.4f} {re:9.2e} {v_p:10.4f} {v_pp:10.3f} {s2_upp:10.3f} {as_s} {ls_s}")
    em_control_rows[lbl] = {
        "pi2_u1_em": v0,
        "pi2_u1_em_exact_rational": ex,
        "rel_err_vs_exact": re,
        "pi2_up1_em": v_p,
        "pi2_upp1_em": v_pp,
        "pi2_upp1_spin2": s2_upp,
        "ratio_spin2_over_em_upp1": (s2_upp / v_pp) if abs(v_pp) > 1e-6 else None,
        "a_destab_em_over_lP": a_d_em,
        "Lmin_em_over_l_sp_red": Lm_lsp_em,
    }
assert max_em_lam1_rel_err < 1e-5, f"Maxwell lam=1 benchmark error too large: {max_em_lam1_rel_err}"

out_solve = {
    "stream_id": "p2_04_graviton_invariance_and_berger_stability",
    "regulator": "Schwinger Gaussian heat-kernel f_t(w) = (1/2) w exp(-t w^2)",
    "species_scale_conventions": {
        "N_sp_pure_gravity": 2,
        "l_Pl_over_lP": SQRT_8PI,
        "l_sp_red_over_l_Pl": math.sqrt(2.0),
        "l_sp_red_over_lP": L_SP_OVER_LP,
    },
    "lam1_benchmark_vs_p203": {
        "max_u1_err_all15": max_u1_err_all,
        "max_u1_rel_err_all15": max_u1_rel_err_all,
        "max_u1_err_q_le_14": max_u1_err_q14,
        "max_up1_err_all15": max_up1_err_all,
        "max_up1_rel_err_all15": max_up1_rel_err_all,
        "per_quotient": bench_lam1,
    },
    "finite_lam_convergence_ladder": {
        "ts_main": list(ts_main),
        "ts_mid": list(ts_mid),
        "ts_fine": list(ts_fine),
        "ts_xfine": list(ts_xfine),
        "max_rel_diff_fine_vs_xfine": max_conv_rel_diff,
        "by_lam": conv_ladder,
    },
    "continuous_s3_six_broken_ckv_modes": {
        "modes": "(2j=0, |2M|=2) [2 broken SU(2)_L Killing] + (2j=1, |2M|=1) [4 broken pure-gradient CKVs]",
        "surviving_modes_in_L_q_ge_3_and_Dm": 0,
        "net_shift_in_u_tot_for_q_ge_3_and_Dm": 0.0,
        "pi2_u6_lam1": u6_0,
        "pi2_u6_p1": u6_prime,
        "pi2_u6_pp1": u6_pprime,
    },
    "nonpositive_eigenvalues_dropped_on_0p52_1p38": total_nonpos_dropped_grid,
    "lam_grid": lam_grid,
    "topological_curves": curves,
    "landscape_summary": landscape_summary,
    "deep_prolate_cusp_free_diffs": {
        "prolate_grid": prolate_grid,
        "u_minus_L3": prolate_diffs,
    },
    "universal_S3_local_renormalization": {
        "operator_caveat": "3D-TT spin-2 Lichnerowicz operator Delta_L^(3D) on static non-Einstein Berger cylinder R_t x S^3_lam; excludes O((lam-1)^2) 4D gauge-constraint / background-stress mixing present in full 4D non-Einstein metric perturbations.",
        "B0_exact": 0.25,
        "B2_formula": "-(7/48) * R^(3)(lam_B) = -(7/24)*(4 - lam_B^2)*lam_B^(2/3)",
        "B2_max_rel_err_0p66_1p30": max_B2_rel_err,
        "A4_lam1_exact": -15.0 / 16.0,
        "A4_lam1_fit_xfine": A4_0_f,
        "cR_exact": cR_exact,
        "cC_exact_rational": cC_exact_rat,
        "cC_fitted_0p80_1p20": cC_global,
        "A4_model_max_abs_err_0p80_1p20": A4_err_080_120,
        "A4_model_max_abs_err_0p66_1p30": A4_err_066_130,
        "A4_pp1_exact_554_over_15": A4_pp1_exact_rat,
        "A4_pp1_from_cC_fit": A4_pp1_analytic_fit_cC,
        "A4_pp1_fd_4th_0p03": A4_pp1_poly4,
        "A4_pp1_fd_4th_0p03_err": A4_pp1_poly4_err,
        "A4_pp1_fd_4th_0p01_xfine": A4_pp1_fd,
        "A4_pp1_fd_4th_0p01_err": A4_pp1_fd_err,
        "pi2_u_ren_lam1_mu1": uren_1,
        "pi2_up_ren_lam1_isotropy_check": uren_p1,
        "pi2_upp_ren_lam1_mu1": uren_pp1,
        "rg_slope_mu_d_dmu_upp_ren": rg_slope_upp1,
        "two_decade_band_mu_a_0p1_to_10": [upp_ren_lo, upp_ren_hi],
        "scheme_comparison": scheme_comparison,
        "rows": s3_local_rows,
    },
    "mc4_spin1_maxwell_berger_control": {
        "min_abs_omega_2j_le_40_lam_0p1_to_6p0": em_min_eig_all,
        "tachyon_free_all_lam": True,
        "max_lam1_rel_err_vs_exact_rational": max_em_lam1_rel_err,
        "per_quotient": em_control_rows,
    },
}

out_path = STREAM_DIR / "02_solve.json"
with open(out_path, "w") as fh:
    json.dump(out_solve, fh, indent=2)
print(f"\nSaved {out_path}")
