#!/usr/bin/env python3
"""
Stream p2_06_poincare_shear_modulus_and_talbot_revivals -- Stage 2: Numerical Solvers,
Heat-Kernel Cross-Checks, Complete Model B Dispersion, & Coupled Talbot ODEs.

Executes:
  - Part 1 (KT1): Independent Kahan-compensated Gaussian heat-kernel evaluations for
      (a) Spin-1 Maxwell across all 5 ADE space forms (vs 50-digit Stage 1 exact oracle)
          and unquotiented Berger-S^3 local renormalization (free B_2/A_4 verification +
          exact-anomaly-subtracted ts_fine vs ts_xfine ladder);
      (b) Spin-2 pure graviton (3D-TT Delta_L^(3D), continuous_s3=True, explicitly isolating
          the 2 broken SU(2)_R/U(1)_R Killing modes at 2j=0, |2M_R|=2) across all 5 ADE space
          forms (vs p2_03 50-digit u(1) + ts_fine vs ts_xfine convergence + 2-decade C^2 band).
  - Part 2 (KT1): 40-digit mpmath + float64 dual-precision Complete Model B dispersion operator
      on P05's ring (Model A, partial Model B, Complete Model B rel-S^3 & abs Gaussian MS,
      hypothetical Kubo-only), computing lower beta_pinch, upper Krein collision ceiling beta_max,
      macroscopic-a Bach restoration radius x_Bach, and shifted p2_02 n=2 2:1 Mathieu wall.
  - Part 3 (KT2): Self-contained Complete-Model-B-capable discrete-pole aux-ODE solver over
      eta in [0, 10*pi] (zero UV tail), verifying static runs (exact H_tot conservation),
      all 6 periodic P05 breathing runs (un-clamped E_00 residual, coherent Rabi trapping),
      3 EFT-window runs, dual-solver DOP853 vs Radau and J_cut=30 vs J_cut=60 cross-checks,
      and the explicit shift of the EFT_sig6_xring_b1 opposite-Krein difference resonance
      under Complete Model B.
"""

from fractions import Fraction
import json
import math
import time
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import psi as sp_psi, polygamma as sp_polygamma

mp.mp.dps = 45

REPO = Path(__file__).resolve().parents[2]
STREAM_DIR = Path(__file__).resolve().parent
OUT_PATH = STREAM_DIR / "02_solve.json"

EULER_GAMMA = 0.5772156649015328606
SIGMA_0 = 9.0 / 31.0
C_ANOM = 9.0 / 31.0
KAPPA_V = -44738.0 / 93.0
X_RING_SQ = 44672.0 / 93.0
X_RING = math.sqrt(X_RING_SQ)
CONV_PI2U_TO_PI = 2160.0 / 31.0
PI_HALF = 0.5 * math.pi
ETA_END = 10.0 * math.pi

ADE_LIST = [("RP3", 2), ("Q8", 8), ("T*", 24), ("O*", 48), ("I*", 120)]


# ==============================================================================
# SHARED OPERATORS (ADE MULTIPLICITIES, SPIN-1 & SPIN-2 BERGER SPECTRUM)
# ==============================================================================

def d_ade(name: str, two_j: int) -> int:
    if two_j < 0:
        return 0
    if name == "I*":
        order = 120
        cls = [
            (1, 0.0), (1, math.pi), (30, math.pi / 2.0),
            (20, math.pi / 3.0), (20, 2.0 * math.pi / 3.0),
            (12, math.pi / 5.0), (12, 2.0 * math.pi / 5.0),
            (12, 3.0 * math.pi / 5.0), (12, 4.0 * math.pi / 5.0),
        ]
    elif name == "Q8":
        order = 8
        cls = [(1, 0.0), (1, math.pi), (6, math.pi / 2.0)]
    elif name == "T*":
        order = 24
        cls = [(1, 0.0), (1, math.pi), (6, math.pi / 2.0), (8, math.pi / 3.0), (8, 2.0 * math.pi / 3.0)]
    elif name == "O*":
        order = 48
        cls = [
            (1, 0.0), (1, math.pi), (18, math.pi / 2.0),
            (8, math.pi / 3.0), (8, 2.0 * math.pi / 3.0),
            (6, math.pi / 4.0), (6, 3.0 * math.pi / 4.0),
        ]
    elif name == "RP3":
        order = 2
        cls = [(1, 0.0), (1, math.pi)]
    else:
        raise ValueError(name)
    tot = 0.0
    for sz, th in cls:
        if th == 0.0:
            chi = two_j + 1
        elif abs(th - math.pi) < 1e-12:
            chi = (two_j + 1) * ((-1) ** two_j)
        else:
            chi = math.sin((two_j + 1) * th) / math.sin(th)
        tot += sz * chi
    iv = int(round(tot / order))
    assert abs(tot / order - iv) < 1e-9
    return iv


# Orthonormal 3x3 symmetric traceless basis & spherical operators for 3D-TT Delta_L^(3D)
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
    Gamma = np.array([
        [[0.5 * (c[a, k, b] - c[k, b, a] + c[b, a, k]) for k in range(3)] for b in range(3)]
        for a in range(3)
    ])
    Riem = np.array([
        [[[sum(Gamma[a, e, k] * Gamma[e, b, l] - Gamma[a, e, l] * Gamma[e, b, k] - c[e, k, l] * Gamma[a, b, e]
               for e in range(3)) for l in range(3)] for k in range(3)] for b in range(3)]
        for a in range(3)
    ])
    M_raw = [
        np.array([
            [np.sum(T_basis[a] * (Gamma[:, :, k] @ T_basis[b] - T_basis[b] @ Gamma[:, :, k])) for b in range(5)]
            for a in range(5)
        ], dtype=complex)
        for k in range(3)
    ]
    K_riem = np.array([
        [np.sum(T_basis[a] * (-2.0 * np.einsum("ikjl,kl->ij", Riem, T_basis[b]))) for b in range(5)]
        for a in range(5)
    ], dtype=complex)
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


def maxwell_block_eigs(two_j: int, two_M: int, lam: float):
    if (two_M - two_j) % 2 != 0 or two_M > two_j + 2:
        return np.empty(0, dtype=float)
    j = 0.5 * two_j
    if two_M == two_j + 2:
        return np.array([2.0 * (j + 1.0) / lam], dtype=float)
    if two_M == two_j:
        return np.array([2.0 * lam + 2.0 * j / lam], dtype=float)
    rad = math.sqrt(lam * lam + 4.0 * j * (j + 1.0) + (1.0 / (lam * lam) - 1.0) * (two_M ** 2))
    return np.array([abs(lam - rad), abs(lam + rad)], dtype=float)


# ==============================================================================
# PART 1 (KT1): GAUSSIAN HEAT-KERNEL SPIN-1 & SPIN-2 SHEAR MODULI
# ==============================================================================

def eval_maxwell_left_hk(lam: float, ade_list: list, ts_all: tuple, n_extrap: int = 4, max_2j: int = None):
    lam_13 = lam ** (1.0 / 3.0)
    t_arr = np.array(ts_all, dtype=float)
    t_min = float(np.min(t_arr))
    scale_min = lam_13 * min(1.0, 1.0 / lam)
    if max_2j is None:
        max_2j = int(math.ceil(math.sqrt(42.0 / t_min) / scale_min)) + 16
    n_t = len(ts_all)
    sum_S3 = np.zeros(n_t, dtype=float)
    comp_S3 = np.zeros(n_t, dtype=float)
    diff_Q = [np.zeros(n_t, dtype=float) for _ in ade_list]
    comp_Q = [np.zeros(n_t, dtype=float) for _ in ade_list]

    for two_j in range(max_2j + 1):
        start_2M = 0 if (two_j % 2 == 0) else 1
        shell_M = np.zeros(n_t, dtype=float)
        for two_M in range(start_2M, two_j + 3, 2):
            ws = maxwell_block_eigs(two_j, two_M, lam)
            if len(ws) == 0:
                continue
            wb = lam_13 * ws
            w2 = wb * wb
            f_val = 0.5 * np.sum(wb[:, None] * np.exp(-w2[:, None] * t_arr[None, :]), axis=0)
            shell_M += (1.0 if two_M == 0 else 2.0) * f_val

        deg_S3 = float(two_j + 1)
        sh_S3 = deg_S3 * shell_M
        y = sh_S3 - comp_S3
        tn = sum_S3 + y
        comp_S3 = (tn - sum_S3) - y
        sum_S3 = tn

        for qi, (name, order) in enumerate(ade_list):
            deg_Q = float(d_ade(name, two_j))
            d_sh = 0.5 * (order * deg_Q - deg_S3) * shell_M
            yq = d_sh - comp_Q[qi]
            tq = diff_Q[qi] + yq
            comp_Q[qi] = (tq - diff_Q[qi]) - yq
            diff_Q[qi] = tq

    out = {}
    for qi, (name, _) in enumerate(ade_list):
        u_ts = diff_Q[qi]
        if n_extrap <= 1:
            u0 = float(u_ts[0])
            u_err = 0.0
        else:
            t_ex = t_arr[:n_extrap]
            u0 = float(np.polyfit(t_ex, u_ts[:n_extrap], n_extrap - 1)[-1])
            u_sub = float(np.polyfit(t_ex[:-1], u_ts[:n_extrap - 1], n_extrap - 2)[-1])
            u_err = abs(u0 - u_sub)
        out[name] = (u0, u_err, float(u_ts[0]))
    return out, max_2j, 0.5 * sum_S3


def eval_spin2_left_hk(lam: float, ade_list: list, ts_all: tuple, n_extrap: int = 4,
                       max_2j: int = None, continuous_s3: bool = True):
    pre = berger_precompute(lam)
    lam_23 = lam ** (2.0 / 3.0)
    t_arr = np.array(ts_all, dtype=float)
    t_min = float(np.min(t_arr))
    scale_min = (lam ** (1.0 / 3.0)) * min(1.0, 1.0 / lam)
    if max_2j is None:
        max_2j = int(math.ceil(math.sqrt(42.0 / t_min) / scale_min)) + 16
    n_t = len(ts_all)
    diff_Q = [np.zeros(n_t, dtype=float) for _ in ade_list]
    comp_Q = [np.zeros(n_t, dtype=float) for _ in ade_list]

    for two_j in range(max_2j + 1):
        start_2M = 0 if (two_j % 2 == 0) else 1
        shell_M = np.zeros(n_t, dtype=float)
        for two_M in range(start_2M, two_j + 5, 2):
            evs = block_tt_eigs(two_j, two_M, pre, continuous_s3=continuous_s3)
            if len(evs) == 0:
                continue
            assert np.all(evs > 0.0)
            w2 = lam_23 * evs
            wb = np.sqrt(w2)
            f_val = 0.5 * np.sum(wb[:, None] * np.exp(-w2[:, None] * t_arr[None, :]), axis=0)
            shell_M += (1.0 if two_M == 0 else 2.0) * f_val

        deg_S3 = float(two_j + 1)
        for qi, (name, order) in enumerate(ade_list):
            deg_Q = float(d_ade(name, two_j))
            d_sh = 0.5 * (order * deg_Q - deg_S3) * shell_M
            yq = d_sh - comp_Q[qi]
            tq = diff_Q[qi] + yq
            comp_Q[qi] = (tq - diff_Q[qi]) - yq
            diff_Q[qi] = tq

    out = {}
    for qi, (name, _) in enumerate(ade_list):
        u_ts = diff_Q[qi]
        if n_extrap <= 1:
            u0 = float(u_ts[0])
            u_err = 0.0
        else:
            t_ex = t_arr[:n_extrap]
            u0 = float(np.polyfit(t_ex, u_ts[:n_extrap], n_extrap - 1)[-1])
            u_sub = float(np.polyfit(t_ex[:-1], u_ts[:n_extrap - 1], n_extrap - 2)[-1])
            u_err = abs(u0 - u_sub)
        out[name] = (u0, u_err, float(u_ts[0]))
    return out, max_2j


def run_part1_heat_kernel(derive_data: dict):
    print("\n" + "=" * 80)
    print("PART 1 (KT1): GAUSSIAN HEAT-KERNEL SPIN-1 & SPIN-2 SHEAR MODULI")
    print("=" * 80)
    t0 = time.time()
    ts_top = (0.00025, 0.00036, 0.00050, 0.00070)
    ts_mid = (0.0010, 0.0014, 0.0020, 0.0028)
    ts_top_fine = (0.00014, 0.00020, 0.00028, 0.00040)
    ts_fine = (0.00012, 0.00017, 0.00024, 0.00034, 0.00048, 0.00066, 0.00090, 0.00122)
    ts_xfine = (0.00006, 0.00009, 0.00013, 0.00019, 0.00027, 0.00038, 0.00053, 0.00074)

    # Scale-matched t-grids: RP3/Q8/T* (larger L_min) use ts_mid (eps=0.012); O*/I* (smaller L_min) use ts_top (eps=0.015)
    ADE_LARGE_L = [("RP3", 2), ("Q8", 8), ("T*", 24)]
    ADE_SMALL_L = [("O*", 48), ("I*", 120)]
    eps_em_l, eps_em_s, eps_iso = 0.012, 0.015, 0.001

    em0_l, mj_eml, _ = eval_maxwell_left_hk(1.0, ADE_LARGE_L, ts_mid, n_extrap=4)
    emp1_l, _, _ = eval_maxwell_left_hk(1.0 + eps_em_l, ADE_LARGE_L, ts_mid, n_extrap=4, max_2j=mj_eml + 8)
    emm1_l, _, _ = eval_maxwell_left_hk(1.0 - eps_em_l, ADE_LARGE_L, ts_mid, n_extrap=4, max_2j=mj_eml + 8)
    emp2_l, _, _ = eval_maxwell_left_hk(1.0 + 2.0 * eps_em_l, ADE_LARGE_L, ts_mid, n_extrap=4, max_2j=mj_eml + 16)
    emm2_l, _, _ = eval_maxwell_left_hk(1.0 - 2.0 * eps_em_l, ADE_LARGE_L, ts_mid, n_extrap=4, max_2j=mj_eml + 16)

    em0_s, mj_ems, _ = eval_maxwell_left_hk(1.0, ADE_SMALL_L, ts_top, n_extrap=4)
    emp1_s, _, _ = eval_maxwell_left_hk(1.0 + eps_em_s, ADE_SMALL_L, ts_top, n_extrap=4, max_2j=mj_ems + 8)
    emm1_s, _, _ = eval_maxwell_left_hk(1.0 - eps_em_s, ADE_SMALL_L, ts_top, n_extrap=4, max_2j=mj_ems + 8)
    emp2_s, _, _ = eval_maxwell_left_hk(1.0 + 2.0 * eps_em_s, ADE_SMALL_L, ts_top, n_extrap=4, max_2j=mj_ems + 16)
    emm2_s, _, _ = eval_maxwell_left_hk(1.0 - 2.0 * eps_em_s, ADE_SMALL_L, ts_top, n_extrap=4, max_2j=mj_ems + 16)

    em_ip1, _, _ = eval_maxwell_left_hk(1.0 + eps_iso, ADE_LIST, (0.0015,), n_extrap=1, max_2j=200)
    em_im1, _, _ = eval_maxwell_left_hk(1.0 - eps_iso, ADE_LIST, (0.0015,), n_extrap=1, max_2j=200)
    em_ip2, _, _ = eval_maxwell_left_hk(1.0 + 2.0 * eps_iso, ADE_LIST, (0.0015,), n_extrap=1, max_2j=200)
    em_im2, _, _ = eval_maxwell_left_hk(1.0 - 2.0 * eps_iso, ADE_LIST, (0.0015,), n_extrap=1, max_2j=200)

    ade_exact = derive_data["theorem1_maxwell_ward"]["ade_50dps_ward"]["ade_exact_table"]
    em_hk = {}
    for name, _ in ADE_LIST:
        if name in ("RP3", "Q8", "T*"):
            v0, eps_u = em0_l[name][0], eps_em_l
            vp1, vm1, vp2, vm2 = emp1_l[name][0], emm1_l[name][0], emp2_l[name][0], emm2_l[name][0]
        else:
            v0, eps_u = em0_s[name][0], eps_em_s
            vp1, vm1, vp2, vm2 = emp1_s[name][0], emm1_s[name][0], emp2_s[name][0], emm2_s[name][0]
        vp_t0 = (-em_ip2[name][0] + 8.0 * em_ip1[name][0] - 8.0 * em_im1[name][0] + em_im2[name][0]) / (12.0 * eps_iso)
        vpp = (-vp2 + 16.0 * vp1 - 30.0 * v0 + 16.0 * vm1 - vm2) / (12.0 * eps_u * eps_u)
        ex_u = float(Fraction(ade_exact[name]["pi2_u_EM_top_exact_rational"]))
        ex_upp = ade_exact[name]["pi2_upp_EM_top"]
        rel_upp = abs(vpp - ex_upp) / max(1.0, abs(ex_upp))
        assert abs(vp_t0) < 1e-8 and rel_upp < 1e-5
        em_hk[name] = {
            "hk_u1": v0, "exact_u1": ex_u, "abs_err_u1": abs(v0 - ex_u),
            "hk_up1_fixed_t": vp_t0, "hk_upp1": vpp, "exact_upp1": ex_upp, "rel_err_upp1": rel_upp,
        }
    print(f"  Spin-1 ADE heat-kernel PASS ({time.time()-t0:.1f}s): I* rel_err = {em_hk['I*']['rel_err_upp1']:.2e}, "
          f"max ADE rel_err = {max(v['rel_err_upp1'] for v in em_hk.values()):.2e}, "
          f"max |u'(1;t0)| = {max(abs(v['hk_up1_fixed_t']) for v in em_hk.values()):.2e}")

    # Unquotiented Berger-S^3 Spin-1 local renormalization:
    # Step A: Free fit at lam = 1.00, 1.10, 1.15 to verify B_2 = -(1/24)R^(3) and A_4 = (1/160)C_(4)^2
    t_xf = np.array(ts_xfine, dtype=float)
    X_free = np.column_stack([1.0 / t_xf, np.log(t_xf) + EULER_GAMMA, np.ones_like(t_xf),
                              t_xf * np.log(t_xf), t_xf, (t_xf ** 2) * np.log(t_xf), t_xf ** 2])
    s3_anomaly_checks = {}
    for lv in (1.00, 1.10, 1.15):
        _, _, raw_s3 = eval_maxwell_left_hk(lv, [("I*", 120)], ts_xfine)
        cf, _, _, _ = np.linalg.lstsq(X_free, raw_s3 - 0.25 / (t_xf ** 2), rcond=None)
        R3 = 2.0 * (4.0 - lv * lv) * (lv ** (2.0 / 3.0))
        C2 = (64.0 / 3.0) * ((1.0 - lv * lv) ** 2) * (lv ** (4.0 / 3.0))
        B2_ex = -(1.0 / 24.0) * R3
        A4_ex = C2 / 160.0
        s3_anomaly_checks[f"lam_{lv:.2f}"] = {
            "B2_fit": float(cf[0]), "B2_exact": B2_ex, "B2_err": abs(float(cf[0]) - B2_ex),
            "A4_fit": 2.0 * float(cf[1]), "A4_exact": A4_ex, "A4_err": abs(2.0 * float(cf[1]) - A4_ex),
        }

    # Step B: Exact analytical-derivative Gaussian heat-kernel extraction (zero FD noise!) on ts_fine & ts_xfine
    def extract_s3_em_ren_analytic_deriv(ts_grid):
        t_arr = np.array(ts_grid, dtype=float)
        max_2j = int(math.ceil(math.sqrt(42.0 / float(np.min(t_arr))))) + 16
        s0_tot = np.zeros_like(t_arr); c0 = np.zeros_like(t_arr)
        s1_tot = np.zeros_like(t_arr); c1 = np.zeros_like(t_arr)
        s2_tot = np.zeros_like(t_arr); c2 = np.zeros_like(t_arr)
        for two_j in range(max_2j + 1):
            j = 0.5 * two_j
            N = float(two_j + 1)
            sh0 = np.zeros_like(t_arr); sh1 = np.zeros_like(t_arr); sh2 = np.zeros_like(t_arr)
            start_2M = 0 if (two_j % 2 == 0) else 1
            for two_M in range(start_2M, two_j + 3, 2):
                mult = 1.0 if two_M == 0 else 2.0
                aM = float(two_M)
                modes = []
                if two_M == two_j + 2:
                    c = 2.0 * (j + 1.0)
                    modes.append((c, c * (-2.0 / 3.0), c * (10.0 / 9.0)))
                elif two_M == two_j:
                    modes.append((2.0 + 2.0 * j, 8.0 / 3.0 - 4.0 * j / 3.0, 8.0 / 9.0 + 20.0 * j / 9.0))
                else:
                    R0 = N
                    R1 = (1.0 - aM * aM) / R0
                    R2 = (1.0 + 3.0 * aM * aM) / R0 - (R1 * R1) / R0
                    for sgn in (+1.0, -1.0):
                        om0, om1, om2 = R0 + sgn, R1 + sgn, R2
                        modes.append((om0, om0 / 3.0 + om1, (-2.0 / 9.0) * om0 + (2.0 / 3.0) * om1 + om2))
                for w0, w1, w2 in modes:
                    wt = (w0 * w0) * t_arr
                    ex = np.exp(-wt)
                    sh0 += mult * 0.5 * w0 * ex
                    sh1 += mult * 0.5 * w1 * (1.0 - 2.0 * wt) * ex
                    sh2 += mult * 0.5 * (w2 * (1.0 - 2.0 * wt) + (w1 * w1) * (-6.0 * w0 * t_arr + 4.0 * w0 * wt * t_arr)) * ex
            for acc, comp, sh in ((s0_tot, c0, sh0), (s1_tot, c1, sh1), (s2_tot, c2, sh2)):
                d = 0.5 * N * sh
                y = d - comp
                tn = acc + y
                comp[:] = (tn - acc) - y
                acc[:] = tn
        X_sub = np.column_stack([np.ones_like(t_arr), t_arr * np.log(t_arr), t_arr, (t_arr ** 2) * np.log(t_arr), t_arr ** 2])
        u0 = float(np.linalg.lstsq(X_sub, s0_tot - 0.25 / (t_arr ** 2) + 0.25 / t_arr, rcond=None)[0][0])
        up1 = float(np.max(np.abs(s1_tot)))
        # Exact B2''(1) = -(1/24)*R3''(1) = -(1/24)*(-32/3) = +4/9, A4''(1) = +16/15
        upp1 = float(np.linalg.lstsq(X_sub, s2_tot - (4.0 / 9.0) / t_arr - 0.5 * (16.0 / 15.0) * (np.log(t_arr) + EULER_GAMMA), rcond=None)[0][0])
        return {"u_ren_1": u0, "u_ren_err_vs_11_240": abs(u0 - 11.0 / 240.0), "up_ren_1_max_all_t": up1, "upp_ren_1": upp1}

    s3_em_fine = extract_s3_em_ren_analytic_deriv(ts_fine)
    s3_em_xfine = extract_s3_em_ren_analytic_deriv(ts_xfine)
    s3_em_upp_conv = abs(s3_em_xfine["upp_ren_1"] - s3_em_fine["upp_ren_1"]) / max(1.0, abs(s3_em_xfine["upp_ren_1"]))
    em_s3_upp_mu1 = s3_em_xfine["upp_ren_1"]
    em_s3_band = [em_s3_upp_mu1 - (16.0 / 15.0) * math.log(10.0), em_s3_upp_mu1 + (16.0 / 15.0) * math.log(10.0)]
    assert s3_em_upp_conv < 1e-5
    print(f"  Berger-S^3 Spin-1 local renorm PASS: u_ren(1)={s3_em_xfine['u_ren_1']:.8f} (err vs 11/240: {s3_em_xfine['u_ren_err_vs_11_240']:.2e}), "
          f"|u'(1)|={s3_em_xfine['up_ren_1_max_all_t']:.2e}, u''_ren(1)={em_s3_upp_mu1:+.6f} (fine vs xfine rel={s3_em_upp_conv:.2e}), "
          f"2-dec band=[{em_s3_band[0]:+.4f}, {em_s3_band[1]:+.4f}]")

    # Spin-2 pure graviton (3D-TT Delta_L^(3D)) across all 5 ADE space forms:
    t_s2 = time.time()
    eps_s2 = 0.015
    s2_0, mj_s2 = eval_spin2_left_hk(1.0, ADE_LIST, ts_top, n_extrap=4, continuous_s3=True)
    s2_p1, _ = eval_spin2_left_hk(1.0 + eps_s2, ADE_LIST, ts_top, n_extrap=4, max_2j=mj_s2 + 8, continuous_s3=True)
    s2_m1, _ = eval_spin2_left_hk(1.0 - eps_s2, ADE_LIST, ts_top, n_extrap=4, max_2j=mj_s2 + 8, continuous_s3=True)
    s2_p2, _ = eval_spin2_left_hk(1.0 + 2.0 * eps_s2, ADE_LIST, ts_top, n_extrap=4, max_2j=mj_s2 + 16, continuous_s3=True)
    s2_m2, _ = eval_spin2_left_hk(1.0 - 2.0 * eps_s2, ADE_LIST, ts_top, n_extrap=4, max_2j=mj_s2 + 16, continuous_s3=True)

    s2_ip1, _ = eval_spin2_left_hk(1.0 + eps_iso, ADE_LIST, (0.0015,), n_extrap=1, max_2j=200, continuous_s3=True)
    s2_im1, _ = eval_spin2_left_hk(1.0 - eps_iso, ADE_LIST, (0.0015,), n_extrap=1, max_2j=200, continuous_s3=True)
    s2_ip2, _ = eval_spin2_left_hk(1.0 + 2.0 * eps_iso, ADE_LIST, (0.0015,), n_extrap=1, max_2j=200, continuous_s3=True)
    s2_im2, _ = eval_spin2_left_hk(1.0 - 2.0 * eps_iso, ADE_LIST, (0.0015,), n_extrap=1, max_2j=200, continuous_s3=True)

    # Independent ts_top_fine + eps = 0.01 convergence check on S^3/I*
    eps_s2_f = 0.01
    s2_0_f, mj_s2_f = eval_spin2_left_hk(1.0, [("I*", 120)], ts_top_fine, n_extrap=4, continuous_s3=True)
    s2_p1_f, _ = eval_spin2_left_hk(1.0 + eps_s2_f, [("I*", 120)], ts_top_fine, n_extrap=4, max_2j=mj_s2_f + 8, continuous_s3=True)
    s2_m1_f, _ = eval_spin2_left_hk(1.0 - eps_s2_f, [("I*", 120)], ts_top_fine, n_extrap=4, max_2j=mj_s2_f + 8, continuous_s3=True)
    s2_p2_f, _ = eval_spin2_left_hk(1.0 + 2.0 * eps_s2_f, [("I*", 120)], ts_top_fine, n_extrap=4, max_2j=mj_s2_f + 16, continuous_s3=True)
    s2_m2_f, _ = eval_spin2_left_hk(1.0 - 2.0 * eps_s2_f, [("I*", 120)], ts_top_fine, n_extrap=4, max_2j=mj_s2_f + 16, continuous_s3=True)
    s2_upp_I_fine = (-s2_p2_f["I*"][0] + 16 * s2_p1_f["I*"][0] - 30 * s2_0_f["I*"][0] + 16 * s2_m1_f["I*"][0] - s2_m2_f["I*"][0]) / (12 * eps_s2_f * eps_s2_f)

    # Isolate the 2 broken SU(2)_R/U(1)_R Killing modes at 2j=0, |2M_R|=2 on S^3/I*:
    # Compare unprojected 3D-TT Lichnerowicz omega_Lich^2 = 8*lam^(8/3) - 4*lam^(2/3) + 4*lam^(-4/3)
    # against exact volume-preserving Bianchi IX exponential variation omega_B9^2 = 4*(lam^(8/3) + lam^(2/3)),
    # which differs at O((lam-1)^2) by the omitted divergence term -4*lam^(-4/3)*(lam^2 - 1)^2 = -2*||div h||^2*lam^(2/3)!
    def broken_j0_contrib(lam):
        e02 = float(block_tt_eigs(0, 2, berger_precompute(lam), continuous_s3=True)[0])
        return 0.5 * 119.0 * (lam ** (1.0 / 3.0)) * math.sqrt(e02)

    def broken_j0_bianchi9_exp(lam):
        w2_b9 = 4.0 * (lam ** (8.0 / 3.0) + lam ** (2.0 / 3.0))
        return 0.5 * 119.0 * math.sqrt(w2_b9)

    h_b9 = 1e-3
    eb0 = broken_j0_contrib(1.0)
    ebp = (-broken_j0_contrib(1.0 + 2 * h_b9) + 8 * broken_j0_contrib(1.0 + h_b9) - 8 * broken_j0_contrib(1.0 - h_b9) + broken_j0_contrib(1.0 - 2 * h_b9)) / (12.0 * h_b9)
    ebpp = (-broken_j0_contrib(1.0 + 2 * h_b9) + 16 * broken_j0_contrib(1.0 + h_b9) - 30 * eb0 + 16 * broken_j0_contrib(1.0 - h_b9) - broken_j0_contrib(1.0 - 2 * h_b9)) / (12.0 * h_b9 * h_b9)

    eb9_0 = broken_j0_bianchi9_exp(1.0)
    eb9_p = (-broken_j0_bianchi9_exp(1.0 + 2 * h_b9) + 8 * broken_j0_bianchi9_exp(1.0 + h_b9) - 8 * broken_j0_bianchi9_exp(1.0 - h_b9) + broken_j0_bianchi9_exp(1.0 - 2 * h_b9)) / (12.0 * h_b9)
    eb9_pp = (-broken_j0_bianchi9_exp(1.0 + 2 * h_b9) + 16 * broken_j0_bianchi9_exp(1.0 + h_b9) - 30 * eb9_0 + 16 * broken_j0_bianchi9_exp(1.0 - h_b9) - broken_j0_bianchi9_exp(1.0 - 2 * h_b9)) / (12.0 * h_b9 * h_b9)
    # Exact analytical checks: f''_Lich(1) = 119*85/(18*sqrt(2)), f''_B9(1) = 119*13/(18*sqrt(2))
    assert abs(ebpp - 119.0 * 85.0 / (18.0 * math.sqrt(2.0))) < 1e-6
    assert abs(eb9_pp - 119.0 * 13.0 / (18.0 * math.sqrt(2.0))) < 1e-6
    broken_j0_div_shift = eb9_pp - ebpp  # -336.582828 (flips sign of Spin-2 u''_top!)

    with open(REPO / "streams/p2_03_graviton_casimir_null_cone/02_solve.json") as fh:
        p203_ade = {k: v["pi2_dC_EH_TT"] for k, v in json.load(fh)["kt1_kt2_ade_space_forms"].items()}
    with open(REPO / "streams/p2_04_graviton_invariance_and_berger_stability/02_solve.json") as fh:
        p204_u = json.load(fh)["universal_S3_local_renormalization"]
        s2_s3_band = p204_u["two_decade_band_mu_a_0p1_to_10"]
        s2_s3_mu1 = p204_u["pi2_upp_ren_lam1_mu1"]

    s2_hk = {}
    max_band_s2 = max(abs(s2_s3_band[0]), abs(s2_s3_band[1]))
    for name, _ in ADE_LIST:
        v0 = s2_0[name][0]
        vp1, vm1, vp2, vm2 = s2_p1[name][0], s2_m1[name][0], s2_p2[name][0], s2_m2[name][0]
        vp_t0 = (-s2_ip2[name][0] + 8.0 * s2_ip1[name][0] - 8.0 * s2_im1[name][0] + s2_im2[name][0]) / (12.0 * eps_iso)
        vpp = (-vp2 + 16.0 * vp1 - 30.0 * v0 + 16.0 * vm1 - vm2) / (12.0 * eps_s2 * eps_s2)
        ref_u = p203_ade[name]
        rel_u = abs(v0 - ref_u) / max(1.0, abs(ref_u))
        assert abs(vp_t0) < 1e-8 and rel_u < 1e-6
        s2_hk[name] = {
            "hk_u1": v0, "p203_u1": ref_u, "rel_err_u1": rel_u,
            "hk_up1_fixed_t": vp_t0, "hk_upp1": vpp, "ratio_vs_2dec_S3_band": abs(vpp) / max_band_s2,
        }
    s2_upp_I = s2_hk["I*"]["hk_upp1"]
    s2_conv_rel = abs(s2_upp_I - s2_upp_I_fine) / max(1.0, abs(s2_upp_I))
    s2_tot_band = [s2_upp_I + s2_s3_band[0], s2_upp_I + s2_s3_band[1]]
    s2_upp_I_b9 = s2_upp_I + broken_j0_div_shift
    s2_tot_band_b9 = [s2_upp_I_b9 + s2_s3_band[0], s2_upp_I_b9 + s2_s3_band[1]]
    print(f"  Spin-2 ADE heat-kernel PASS ({time.time()-t_s2:.1f}s): I* Delta_L^(3D) u''_top={s2_upp_I:+.4f} "
          f"(fine vs xfine rel={s2_conv_rel:.2e}; exact Bianchi IX j=0 |2M|=2 shift={broken_j0_div_shift:+.2f} -> {s2_upp_I_b9:+.4f}, "
          f"2-dec band [{s2_tot_band_b9[0]:+.2f}, {s2_tot_band_b9[1]:+.2f}] straddles 0!)")

    return {
        "maxwell_ade_hk": em_hk,
        "maxwell_S3_local_renorm": {
            "anomaly_free_fit_checks": s3_anomaly_checks,
            "exact_sub_ts_fine": s3_em_fine,
            "exact_sub_ts_xfine": s3_em_xfine,
            "upp_fine_vs_xfine_rel_diff": s3_em_upp_conv,
            "pi2_upp_ren_lam1_mu1": em_s3_upp_mu1,
            "two_decade_band": em_s3_band,
            "Istar_top_over_max_2dec_band": abs(ade_exact["I*"]["pi2_upp_EM_top"]) / max(abs(em_s3_band[0]), abs(em_s3_band[1])),
        },
        "spin2_ade_hk": s2_hk,
        "spin2_Istar_summary": {
            "ts_top_upp1_unprojected_DeltaL3D": s2_upp_I,
            "ts_top_fine_upp1_unprojected_DeltaL3D": s2_upp_I_fine,
            "fine_vs_xfine_rel_diff": s2_conv_rel,
            "broken_2j0_2M2_unprojected_Lich": {"u1": eb0, "up1": ebp, "upp1": ebpp},
            "broken_2j0_2M2_exact_BianchiIX_exp": {"u1": eb9_0, "up1": eb9_p, "upp1": eb9_pp},
            "broken_2j0_2M2_divergence_shift": broken_j0_div_shift,
            "pi2_upp_Istar_with_BianchiIX_j0": s2_upp_I_b9,
            "total_Istar_2dec_band_with_BianchiIX_j0": s2_tot_band_b9,
            "S3_2dec_band": s2_s3_band,
            "S3_mu1": s2_s3_mu1,
            "total_Istar_2dec_band_unprojected_DeltaL3D": s2_tot_band,
            "total_Istar_mu1_unprojected_DeltaL3D": s2_upp_I + s2_s3_mu1,
            "physical_sign_indeterminate_offshell_nonEinstein": True,
        },
    }


# ==============================================================================
# PART 2 (KT1): COMPLETE MODEL B DISPERSION OPERATOR (40-DIGIT + FLOAT64)
# ==============================================================================

R30_ISTAR = [120 * d_ade("I*", 2 * r) - 2 * (2 * r + 1) for r in range(30)]


def K_Istar_mp(Om, mu_cyl):
    Om_m = mp.mpc(Om)
    mu_m = mp.mpf(mu_cyl)
    lead = mp.log(16 / (mu_m ** 2)) + mp.psi(0, mp.mpf("31.5") - Om_m / 4) + mp.psi(0, mp.mpf("31.5") + Om_m / 4)
    if abs(Om_m) < mp.mpf("1e-12"):
        rem = sum(R30_ISTAR[r % 30] * 8 * mp.psi(1, mp.mpf(120 + 4 * r + 2) / 120) / (120 ** 2) for r in range(1, 31))
    else:
        rem = sum(
            R30_ISTAR[r % 30] / (30 * Om_m) * (
                mp.psi(0, (122 + 4 * r + Om_m) / 120) - mp.psi(0, (122 + 4 * r - Om_m) / 120)
            )
            for r in range(1, 31)
        )
    low = sum(
        (mp.mpf(240 * d_ade("I*", 2 * j)) / (2 * j + 1)) * (2 * (4 * j + 2)) / ((4 * j + 2) ** 2 - Om_m ** 2)
        for j in range(1, 31) if d_ade("I*", 2 * j) > 0
    )
    return lead - rem - low


def K_Istar_f64(Om: float, mu_cyl: float) -> float:
    lead = math.log(16.0 / (mu_cyl * mu_cyl)) + float(sp_psi(31.5 - 0.25 * Om)) + float(sp_psi(31.5 + 0.25 * Om))
    if abs(Om) < 1e-10:
        rem = sum(R30_ISTAR[r % 30] * 8.0 * float(sp_polygamma(1, (120.0 + 4.0 * r + 2.0) / 120.0)) / (120.0 ** 2) for r in range(1, 31))
    else:
        rem = sum(
            R30_ISTAR[r % 30] / (30.0 * Om) * (
                float(sp_psi((122.0 + 4.0 * r + Om) / 120.0)) - float(sp_psi((122.0 + 4.0 * r - Om) / 120.0))
            )
            for r in range(1, 31)
        )
    low = sum(
        (240.0 * d_ade("I*", 2 * j) / (2.0 * j + 1.0)) * (2.0 * (4.0 * j + 2.0)) / ((4.0 * j + 2.0) ** 2 - Om * Om)
        for j in range(1, 31) if d_ade("I*", 2 * j) > 0
    )
    return lead - rem - low


def run_part2_complete_model_B(derive_data: dict, part1_data: dict):
    print("\n" + "=" * 80)
    print("PART 2 (KT1): COMPLETE MODEL B DISPERSION OPERATOR (40-DIGIT vs FLOAT64)")
    print("=" * 80)
    xr = mp.sqrt(mp.mpf(44672) / 93)
    c_an = mp.mpf(9) / 31
    xsq = mp.mpf(44672) / 93

    # Exact constants from Stage 1 + Berger-S^3 local Gaussian MS
    K0_I_mu1 = mp.re(K_Istar_mp(0, 1))
    K0_S3_mu1 = mp.log(4) + 2 * mp.psi(0, 3)
    K0_top_diff = K0_I_mu1 - K0_S3_mu1
    Pi_tot_top_0 = mp.mpf(2160) / 31 * mp.mpf(derive_data["theorem1_maxwell_ward"]["ade_50dps_ward"]["ade_exact_table"]["I*"]["pi2_upp_EM_top_50dps"])
    Pi_rem_comp_relS3 = Pi_tot_top_0 - 2 * c_an * 64 * K0_top_diff

    em_s3_upp_mu1 = part1_data["maxwell_S3_local_renorm"]["pi2_upp_ren_lam1_mu1"]
    Pi_tot_S3_gauss_mu1 = mp.mpf(2160) / 31 * mp.mpf(em_s3_upp_mu1)
    Pi_tot_I_gauss_mu1 = Pi_tot_top_0 + Pi_tot_S3_gauss_mu1
    Pi_rem_comp_abs = Pi_tot_I_gauss_mu1 - 2 * c_an * 64 * K0_I_mu1

    Pi_kin_rel = mp.mpf(3900) / 31
    Pi_kin_abs = mp.mpf(4344) / 31
    Pi_kubo_rem_I = mp.mpf(608048) / 155

    def analyze_model(pi_kin_m, pi_rem_m, label):
        def beta_c_mp(om):
            om_m = mp.mpf(om)
            B2 = (om_m ** 2 - 4) * (om_m ** 2 - 16)
            return ((xsq - pi_kin_m) * om_m ** 2 - (8 * xsq + pi_rem_m)) / (4 * B2) - mp.mpf("0.5") * c_an * mp.re(K_Istar_mp(om_m, xr))

        def D_mp(om, b0):
            om_m = mp.mpc(om)
            B2 = (om_m ** 2 - 4) * (om_m ** 2 - 16)
            return (xsq - pi_kin_m) * om_m ** 2 - (8 * xsq + pi_rem_m) - 4 * mp.mpf(b0) * B2 - 2 * c_an * B2 * K_Istar_mp(om_m, xr)

        om0_sq = (8 * xsq + pi_rem_m) / (xsq - pi_kin_m)
        om0 = mp.sqrt(om0_sq)

        # Find lower pinch-off minimum on (4.2, 25.8) via coarse scan + golden section in 40-digit mpmath:
        def golden_ext(fn, a, b, find_max=False, n_iter=90):
            phi = (mp.sqrt(5) - 1) / 2
            c = b - phi * (b - a)
            d = a + phi * (b - a)
            fc, fd = fn(c), fn(d)
            for _ in range(n_iter):
                cond = (fc > fd) if find_max else (fc < fd)
                if cond:
                    b, d, fd = d, c, fc
                    c = b - phi * (b - a)
                    fc = fn(c)
                else:
                    a, c, fc = c, d, fd
                    d = a + phi * (b - a)
                    fd = fn(d)
            xm = (a + b) / 2
            return xm, fn(xm)

        def beta_c_f64(om: float) -> float:
            B2 = (om * om - 4.0) * (om * om - 16.0)
            return ((X_RING_SQ - float(pi_kin_m)) * om * om - (8.0 * X_RING_SQ + float(pi_rem_m))) / (4.0 * B2) - 0.5 * C_ANOM * K_Istar_f64(om, X_RING)

        def D_f64(om: float, b0: float) -> float:
            B2 = (om * om - 4.0) * (om * om - 16.0)
            return (X_RING_SQ - float(pi_kin_m)) * om * om - (8.0 * X_RING_SQ + float(pi_rem_m)) - 4.0 * b0 * B2 - 2.0 * C_ANOM * B2 * K_Istar_f64(om, X_RING)

        # Fast float64 coarse scan + 45-digit mpmath golden-section refinement!
        om_scan = np.linspace(4.05, 25.8, 1088)
        bc_scan = np.array([beta_c_f64(float(o)) for o in om_scan])
        # Lower pinch-off minimum sits in Omega in [10, 24]
        mask_p = (om_scan >= 10.0) & (om_scan <= 24.0)
        idx_p = int(np.argmin(bc_scan[mask_p]))
        om_p0 = float(om_scan[mask_p][idx_p])
        om_pinch_m, b_pinch_m = golden_ext(beta_c_mp, mp.mpf(om_p0 - 0.15), mp.mpf(om_p0 + 0.15), find_max=False)

        # Float64 independent golden-section cross-check of beta_pinch:
        a_f, b_f = om_p0 - 0.15, om_p0 + 0.15
        phi_f = (math.sqrt(5.0) - 1.0) / 2.0
        c_f, d_f = b_f - phi_f * (b_f - a_f), a_f + phi_f * (b_f - a_f)
        fc_f, fd_f = beta_c_f64(c_f), beta_c_f64(d_f)
        for _ in range(70):
            if fc_f < fd_f:
                b_f, d_f, fd_f = d_f, c_f, fc_f
                c_f = b_f - phi_f * (b_f - a_f)
                fc_f = beta_c_f64(c_f)
            else:
                a_f, c_f, fc_f = c_f, d_f, fd_f
                d_f = a_f + phi_f * (b_f - a_f)
                fd_f = beta_c_f64(d_f)
        b_pinch_f64 = beta_c_f64(0.5 * (a_f + b_f))
        pinch_dual_diff = abs(float(b_pinch_m) - b_pinch_f64)

        # Upper Krein collision ceiling beta_max (exists iff bare Omega_0 > 4!)
        beta_max_m = None
        om_coll_m = None
        x_bach_m = None
        if om0 > 4:
            mask_u = (om_scan >= float(om0) - 0.05) & (om_scan <= float(om_pinch_m) - 0.5)
            idx_u = int(np.argmax(bc_scan[mask_u]))
            om_u0 = float(om_scan[mask_u][idx_u])
            om_coll_m, beta_max_m = golden_ext(beta_c_mp, mp.mpf(max(4.005, om_u0 - 0.15)), mp.mpf(om_u0 + 0.15), find_max=True)
            x_bach_m = mp.sqrt(2 * pi_kin_m + pi_rem_m / 8)

        # Find all real sub-threshold roots in (0, 25.95) for beta_0 in [0.0, 0.75, 1.0, 2.0, 3.5, 4.0, 4.4, 5.0]
        roots_table = {}
        grid = np.linspace(0.0, 25.9, 1296)
        for b0 in (0.0, 0.75, 1.0, 2.0, 3.5, 4.0, 4.4, 5.0):
            D0 = float(mp.re(D_mp(0, b0)))
            dvals = [D_f64(float(o), b0) for o in grid]
            rts = []
            for i in range(len(grid) - 1):
                if dvals[i] * dvals[i + 1] <= 0.0:
                    lo, hi = mp.mpf(float(grid[i])), mp.mpf(float(grid[i + 1]))
                    flo = mp.re(D_mp(lo, b0))
                    for _ in range(90):
                        mid = (lo + hi) / 2
                        fmid = mp.re(D_mp(mid, b0))
                        if flo * fmid <= 0:
                            hi = mid
                        else:
                            lo, flo = mid, fmid
                    om_r = (lo + hi) / 2
                    sl = (mp.re(D_mp(om_r + mp.mpf("1e-9"), b0)) - mp.re(D_mp(om_r - mp.mpf("1e-9"), b0))) / mp.mpf("2e-9")
                    rts.append({"Omega": float(om_r), "Omega_sq": float(om_r ** 2), "krein": "+" if sl > 0 else "-"})
            om_grav = rts[0]["Omega"] if (rts and rts[0]["Omega"] < 10.0) else None
            # Shifted p2_02 n=2 2:1 Mathieu wall sigma_Math(2) = (|kappa_V| - 1) / (4*Omega_grav^2 - 1)
            sig_math = (abs(KAPPA_V) - 1.0) / (4.0 * om_grav * om_grav - 1.0) if (om_grav is not None) else None
            diff_krein = (rts[1]["Omega"] - rts[0]["Omega"]) if len(rts) >= 2 else None
            roots_table[str(b0)] = {
                "stable_at_Om0": D0 < 0.0,
                "D0": D0,
                "Omega_grav": om_grav,
                "Omega_grav_sq": (om_grav * om_grav if om_grav is not None else None),
                "diff_Stelle_minus_grav": diff_krein,
                "shifted_p202_mathieu_wall_sigma": sig_math,
                "all_subthreshold_roots": rts,
            }

        print(f"  [{label:<30}] Om0={float(om0):.4f} | beta_pinch={float(b_pinch_m):.6f} (Om={float(om_pinch_m):.3f}) | "
              f"beta_max={f'{float(beta_max_m):.4f} (Om={float(om_coll_m):.3f}, x_Bach={float(x_bach_m):.2f})' if beta_max_m else 'inf (Bach-protected)'} | "
              f"b0=1 Om_g={roots_table['1.0']['Omega_grav']:.4f}, sig_Math={roots_table['1.0']['shifted_p202_mathieu_wall_sigma']}")
        return {
            "Pi_kin": float(pi_kin_m),
            "Pi_rem": float(pi_rem_m),
            "bare_Omega_0": float(om0),
            "breaches_Bach_4": bool(om0 > 4),
            "beta_pinch": float(b_pinch_m),
            "Om_pinch": float(om_pinch_m),
            "beta_pinch_dual_prec_diff": pinch_dual_diff,
            "beta_max": float(beta_max_m) if beta_max_m is not None else None,
            "Om_coll": float(om_coll_m) if om_coll_m is not None else None,
            "x_Bach_restoration": float(x_bach_m) if x_bach_m is not None else None,
            "roots": roots_table,
        }

    models = {
        "Model_A_baseline": analyze_model(mp.mpf(0), mp.mpf(0), "Model A (p2_01 baseline)"),
        "Model_B_partial_kin": analyze_model(Pi_kin_rel, mp.mpf(0), "Model B partial (kin only)"),
        "Model_B_complete_relS3": analyze_model(Pi_kin_rel, Pi_rem_comp_relS3, "Model B COMPLETE (rel-S^3)"),
        "Model_B_complete_abs_gauss": analyze_model(Pi_kin_abs, Pi_rem_comp_abs, "Model B COMPLETE (abs MS)"),
        "Hypothetical_Kubo_only": analyze_model(Pi_kin_abs, Pi_kubo_rem_I, "Hypothetical Kubo-only"),
    }

    # Static stiffness summary on P05 ring
    stiff_EH = 8.0 * X_RING_SQ
    rg_shift_xring = -(2304.0 / 31.0) * math.log(X_RING)
    Pi_tot_I_muHS = float(Pi_tot_I_gauss_mu1) + rg_shift_xring
    return {
        "classical_8_xring_sq": stiff_EH,
        "Pi_tot_top_0": float(Pi_tot_top_0),
        "Pi_tot_top_pct_of_EH": 100.0 * float(Pi_tot_top_0) / stiff_EH,
        "Pi_tot_I_gauss_mu1": float(Pi_tot_I_gauss_mu1),
        "Pi_tot_I_gauss_muHS": Pi_tot_I_muHS,
        "Pi_rem_complete_relS3": float(Pi_rem_comp_relS3),
        "Pi_rem_complete_abs_gauss": float(Pi_rem_comp_abs),
        "static_stiffness_EH_plus_top": stiff_EH + float(Pi_tot_top_0),
        "static_stiffness_abs_muHS_b0": stiff_EH + Pi_tot_I_muHS,
        "static_stiffness_abs_muHS_b1": stiff_EH + Pi_tot_I_muHS + 256.0,
        "models": models,
    }


# ==============================================================================
# PART 3 (KT2): COUPLED DISCRETE AUX-ODE (MODEL A & COMPLETE MODEL B, DUAL SOLVER)
# ==============================================================================

def e00_constraint(x, h, r, s, sigma_t=SIGMA_0, lam=0.0, kappa=KAPPA_V):
    y = h * h + 1.0 / (x * x)
    return y - lam - y * y + sigma_t * (r * (2.0 * y - r) + 2.0 * h * s) - kappa / (x ** 4)


def integrate_kam_background_eta(x_b=X_RING, eta_end=ETA_END + 1.0, sigma_t=SIGMA_0, kappa=KAPPA_V):
    kappa_eff = 1.0 - sigma_t + kappa
    rhs_b = (1.0 / (x_b * x_b) - kappa_eff / (x_b ** 4)) / sigma_t
    assert rhs_b >= 0.0
    hdot_b = math.sqrt(rhs_b)
    r_b = 1.0 / (x_b * x_b) + hdot_b
    y0 = np.array([x_b, 0.0, r_b, 0.0, 0.0], dtype=float)

    def rhs_eta(eta, y):
        x, h, r, s, _ = y
        inv_x2 = 1.0 / (x * x)
        yy = h * h + inv_x2
        dx_t = x * h
        dh_t = r - 2.0 * h * h - inv_x2
        dr_t = s
        ds_t = -3.0 * h * s + (-r + 2.0 * (r - yy) * yy) / sigma_t
        return [x * dx_t, x * dh_t, x * dr_t, x * ds_t, x]

    sol = solve_ivp(rhs_eta, (0.0, eta_end), y0, method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True, max_step=0.02)
    assert sol.success
    etas_chk = np.linspace(0.0, eta_end, 1001)
    ys = sol.sol(etas_chk)
    e00_rel = []
    for i in range(len(etas_chk)):
        xi, hi, ri, si = ys[0, i], ys[1, i], ys[2, i], ys[3, i]
        res = abs(e00_constraint(xi, hi, ri, si, sigma_t, 0.0, kappa))
        yi = hi * hi + 1.0 / (xi * xi)
        sc = max(abs(yi), abs(yi * yi), abs(sigma_t * ri * (2.0 * yi - ri)), abs(2.0 * sigma_t * hi * si), abs(kappa) / (xi ** 4))
        e00_rel.append(res / sc)
    return sol, {
        "x_b": float(x_b), "tau_end": float(ys[4, -1]),
        "x_min": float(np.min(ys[0])), "x_max": float(np.max(ys[0])),
        "max_rel_e00": float(max(e00_rel)),
    }


def build_discrete_pole_spec(topology="Istar", J_cut=30):
    M_tail = 4.0 * J_cut + 4.0
    if topology == "Istar":
        E_l, w_l = [], []
        for j in range(1, J_cut + 1):
            d = d_ade("I*", 2 * j)
            if d > 0:
                E_l.append(4.0 * j + 2.0)
                w_l.append(240.0 * d / (2.0 * j + 1.0))
        E_p = np.array(E_l, dtype=float)
        w_p = np.array(w_l, dtype=float)
        alpha_p = w_p / E_p
        K0_exact_mu1 = float(mp.re(K_Istar_mp(0, 1)))
    elif topology == "S3":
        E_p = np.arange(6.0, 4.0 * J_cut + 3.0, 2.0, dtype=float)
        w_p = np.full_like(E_p, 2.0)
        alpha_p = w_p / E_p
        K0_exact_mu1 = math.log(4.0) + 3.0 - 2.0 * EULER_GAMMA
    else:
        raise ValueError(topology)
    K0_hyb_mu1 = math.log(M_tail * M_tail) - float(np.sum(2.0 * alpha_p))
    return {
        "topology": topology, "J_cut": int(J_cut), "E_p": E_p, "w_p": w_p, "alpha_p": alpha_p,
        "M_tail": float(M_tail), "delta_C_tail": float(K0_exact_mu1 - K0_hyb_mu1),
    }


def solve_discrete_aux_ode(
    bg_sol, eta_end, beta_0, pole_spec,
    kappa_mem=1.0, running=True, static_bg=False, x_static=X_RING,
    pi_kin=0.0, pi_rem=0.0, y0_4d=None, n_eval=12001,
    method="DOP853", rtol=1e-9, atol=1e-11, max_step=0.005,
    sigma_t=SIGMA_0, track_hamiltonian=False,
):
    """
    Self-contained unreduced 4th-order Ostrogradsky + Exact Discrete 2-Photon Pole Sum solver
    (zero continuum UV tail) supporting BOTH Model A (pi_kin=0, pi_rem=0) and Complete Model B
    (pi_kin > 0, pi_rem != 0).
    """
    kn2, pn = 8.0, 3.0
    mu_cyl = X_RING
    c_mem = kappa_mem * C_ANOM
    E_p, alpha_p = pole_spec["E_p"], pole_spec["alpha_p"]
    Np = len(E_p)
    uv_shift = 0.5 * c_mem * (math.log((pole_spec["M_tail"] ** 2) / (mu_cyl ** 2)) + pole_spec["delta_C_tail"])

    iYQ, iZQ, iYP, iZP = 4, 4 + Np, 4 + 2 * Np, 4 + 3 * Np
    dim = 4 + 4 * Np

    def get_bg(eta):
        if static_bg:
            x, A_R, K_R = x_static, 1.0, 1.0
        else:
            st = bg_sol.sol(eta)
            x, h, r = float(st[0]), float(st[1]), float(st[2])
            inv_x2 = 1.0 / (x * x)
            yy = h * h + inv_x2
            A_R = 1.0 + 2.0 * sigma_t * r - 2.0 * h * h + (2.0 / 3.0) * inv_x2
            K_R = 1.0 + 2.0 * sigma_t * r + 2.0 * h * h + (2.0 / 3.0) * inv_x2 - 4.0 * (r - yy)
        beff = beta_0 - (C_ANOM * math.log(x / X_RING) if running else 0.0)
        return x, A_R, K_R, beff + uv_shift

    _, _, _, buv0 = get_bg(0.0)
    q1_0, q2_0, pi1_0, pi2_0 = [float(v) for v in (y0_4d if y0_4d is not None else [0.0, 0.0, 1.0, 0.0])]
    P0_val = 2.0 * pn * q2_0
    b_init_eff = buv0 - c_mem * float(np.sum(alpha_p))
    Q0_val = -pi2_0 / (8.0 * b_init_eff)

    y0 = np.zeros(dim, dtype=float)
    y0[:4] = [q1_0, q2_0, pi1_0, pi2_0]
    y0[iYQ:iZQ] = Q0_val
    y0[iYP:iZP] = P0_val

    def rhs(eta, y):
        x, A_R, K_R, b_uv = get_bg(eta)
        q1, q2, pi1, pi2 = y[0], y[1], y[2], y[3]
        P = 6.0 * q2
        YQ, ZQ, YP, ZP = y[iYQ:iZQ], y[iZQ:iYP], y[iYP:iZP], y[iZP:dim]
        Q = (-0.125 * pi2 + c_mem * float(np.dot(alpha_p, YQ))) / b_uv
        Eff_P = b_uv * P - c_mem * float(np.dot(alpha_p, YP))
        dy = np.empty(dim, dtype=float)
        dy[0] = q2
        dy[1] = kn2 * q1 + Q
        dy[2] = -(2.0 * kn2 * x * x * K_R + 2.0 * pi_rem) * q1 - kn2 * pi2
        dy[3] = -pi1 + 2.0 * (x * x * A_R - pi_kin) * q2 + 48.0 * Eff_P
        dy[iYQ:iZQ] = E_p * ZQ
        dy[iZQ:iYP] = E_p * (Q - YQ)
        dy[iYP:iZP] = E_p * ZP
        dy[iZP:dim] = E_p * (P - YP)
        return dy

    def jac(eta, y):
        x, A_R, K_R, b_uv = get_bg(eta)
        J = np.zeros((dim, dim), dtype=float)
        c_over_b = c_mem / b_uv
        J[0, 1] = 1.0
        J[1, 0] = kn2
        J[1, 3] = -0.125 / b_uv
        J[2, 0] = -(2.0 * kn2 * x * x * K_R + 2.0 * pi_rem)
        J[2, 3] = -kn2
        J[3, 1] = 2.0 * (x * x * A_R - pi_kin) + 288.0 * b_uv
        J[3, 2] = -1.0
        J[1, iYQ:iZQ] = c_over_b * alpha_p
        J[3, iYP:iZP] = -48.0 * c_mem * alpha_p
        J[iYQ:iZQ, iZQ:iYP] = np.diag(E_p)
        J[iZQ:iYP, 3] = -0.125 * E_p / b_uv
        J[iZQ:iYP, iYQ:iZQ] = np.outer(E_p * c_over_b, alpha_p) - np.diag(E_p)
        J[iYP:iZP, iZP:dim] = np.diag(E_p)
        J[iZP:dim, 1] = 6.0 * E_p
        J[iZP:dim, iYP:iZP] = -np.diag(E_p)
        return J

    t_eval = np.linspace(0.0, eta_end, n_eval)
    kw = {"method": method, "t_eval": t_eval, "rtol": rtol, "atol": atol, "max_step": max_step}
    if method in ("Radau", "BDF"):
        kw["jac"] = jac
    sol = solve_ivp(rhs, (0.0, eta_end), y0, **kw)
    assert sol.success, f"ODE {method} failed: {sol.message}"

    q1, q2, pi1, pi2 = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    YQ, ZQ, YP, ZP = sol.y[iYQ:iZQ], sol.y[iZQ:iYP], sol.y[iYP:iZP], sol.y[iZP:dim]

    H_drift = None
    if track_hamiltonian and static_bg:
        # Exact conserved Ostrogradsky + Discrete-Pole Hamiltonian (for both Model A and Complete Model B):
        b_uv = buv0
        mem_Q = alpha_p @ YQ
        Q_arr = (-0.125 * pi2 + c_mem * mem_Q) / b_uv
        P_arr = 6.0 * q2
        H_tot = (
            pi1 * q2 + kn2 * pi2 * q1
            - (x_static * x_static - pi_kin) * (q2 ** 2)
            + (kn2 * x_static * x_static + pi_rem) * (q1 ** 2)
            - 4.0 * b_uv * (Q_arr ** 2)
            + 4.0 * c_mem * (alpha_p @ (YQ ** 2 + ZQ ** 2))
            - 4.0 * b_uv * (P_arr ** 2) + 8.0 * c_mem * P_arr * (alpha_p @ YP)
            - 4.0 * c_mem * (alpha_p @ (YP ** 2 + ZP ** 2))
        )
        H_scale = max(1.0, abs(float(H_tot[0])))
        H_drift = float(np.max(np.abs(H_tot - H_tot[0])) / H_scale)

    # Exact auxiliary diagnostics directly from (YQ, ZQ, YP, ZP) (NO FD approximation needed!)
    if static_bg:
        xs = np.full_like(t_eval, x_static)
    else:
        xs = bg_sol.sol(t_eval)[0]
    beff = beta_0 - (C_ANOM * np.log(xs / X_RING) if running else 0.0)
    b_uv_arr = beff + uv_shift
    Q_arr = (-0.125 * pi2 + c_mem * (alpha_p @ YQ)) / b_uv_arr
    P_arr = 6.0 * q2
    dYQ, dYP = YQ - Q_arr[None, :], YP - P_arr[None, :]
    amp2 = dYQ ** 2 + ZQ ** 2 + dYP ** 2 + ZP ** 2
    N_gamma = alpha_p @ amp2
    M_Q = alpha_p @ dYQ
    mean_per_pole = np.mean(alpha_p[:, None] * amp2, axis=1)
    top_idx = int(np.argmax(mean_per_pole))

    return {
        "eta": t_eval, "q1": q1, "q2": q2, "pi1": pi1, "pi2": pi2,
        "N_gamma": N_gamma, "M_Q": M_Q,
        "H_tot_max_rel_drift": H_drift,
        "dominant_pole_E": float(E_p[top_idx]),
        "dominant_pole_frac": float(mean_per_pole[top_idx] / max(1e-30, float(np.sum(mean_per_pole)))),
    }


def extract_talbot_and_drain(N_gamma, M_Q, etas):
    def ac_stats(sig):
        x = sig - np.mean(sig)
        d_eta = float(etas[1] - etas[0])
        n_lag = min(len(x) - 1, int(round(2.2 * math.pi / d_eta)))
        n_fft = 1 << (2 * len(x) - 1).bit_length()
        fx = np.fft.rfft(x, n=n_fft)
        ac = np.fft.irfft(fx * np.conj(fx), n=n_fft)[:n_lag + 1]
        counts = np.arange(len(x), len(x) - n_lag - 1, -1, dtype=float)
        ac = (ac / counts) / max(1e-30, ac[0] / len(x))
        lags = np.arange(n_lag + 1) * d_eta
        m_in = (lags >= 0.15) & (lags <= 1.2)
        ac_in, l_in = ac[m_in], lags[m_in]
        pks = np.where((ac_in[1:-1] > ac_in[:-2]) & (ac_in[1:-1] >= ac_in[2:]) & (ac_in[1:-1] > 0.05))[0] + 1
        return {
            "ac_pi_over_2": float(ac[int(round(PI_HALF / d_eta))]),
            "ac_pi": float(ac[int(round(math.pi / d_eta))]),
            "first_intra_lag": float(l_in[pks[0]]) if len(pks) > 0 else None,
        }

    win_means, win_maxs = [], []
    for k in range(20):
        m = (etas >= k * PI_HALF) & (etas < (k + 1) * PI_HALF)
        if np.any(m):
            win_means.append(float(np.mean(N_gamma[m])))
            win_maxs.append(float(np.max(N_gamma[m])))
    w_t = (np.arange(len(win_means)) + 0.5) * PI_HALF
    n0 = max(1e-30, win_means[0])
    slope_norm = float(np.polyfit(w_t, np.array(win_means) / n0, 1)[0])
    return {
        "M_Q": ac_stats(M_Q),
        "N_gamma": ac_stats(N_gamma),
        "late_over_early_mean": float(np.mean(win_means[-4:]) / max(1e-30, np.mean(win_means[:4]))),
        "late_over_early_peak": float(np.max(win_maxs[-4:]) / max(1e-30, np.max(win_maxs[:4]))),
        "normalized_slope_per_eta": slope_norm,
    }


def run_part3_coupled_talbot(part2_data: dict):
    print("\n" + "=" * 80)
    print("PART 3 (KT2): COUPLED DISCRETE AUX-ODE, DUAL-SOLVER & COMPLETE MODEL B SHIFT")
    print("=" * 80)
    spec_I30 = build_discrete_pole_spec("Istar", 30)
    spec_I60 = build_discrete_pole_spec("Istar", 60)
    spec_S3_30 = build_discrete_pole_spec("S3", 30)
    bg_ring, bg_ring_info = integrate_kam_background_eta(X_RING, ETA_END + 1.0, SIGMA_0)
    tau_ring_10pi = float(bg_ring.sol(ETA_END)[4])
    print(f"  P05 ring background: tau(10pi)={tau_ring_10pi:.2f}, max_rel_e00={bg_ring_info['max_rel_e00']:.2e}")

    # 1. Static x_ring runs (exact H_tot conservation!)
    static_runs = {}
    for lbl, sp, b0, kmem, y0_4d in [
        ("Istar_weak_b1_pulse", spec_I30, 1.0, 1e-3, [0.0, 0.0, 1.0, 0.0]),
        ("S3_weak_b1_pulse", spec_S3_30, 1.0, 1e-3, [0.0, 0.0, 1.0, 0.0]),
        ("Istar_full_b1_pulse", spec_I30, 1.0, 1.0, [0.0, 0.0, 1.0, 0.0]),
        ("Istar_full_b2_pulse", spec_I30, 2.0, 1.0, [0.0, 0.0, 1.0, 0.0]),
        ("Istar_full_b1_grav", spec_I30, 1.0, 1.0, [1.0, 0.0, 0.0, 0.0]),
        ("S3_full_b8p5_pulse", spec_S3_30, 8.5, 1.0, [0.0, 0.0, 1.0, 0.0]),
    ]:
        r = solve_discrete_aux_ode(bg_ring, ETA_END, b0, sp, kappa_mem=kmem, static_bg=True,
                                   y0_4d=y0_4d, method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.005,
                                   track_hamiltonian=True)
        st = extract_talbot_and_drain(r["N_gamma"], r["M_Q"], r["eta"])
        q1_ratio_st = float(np.max(np.abs(r["q1"][-2400:])) / max(1e-30, float(np.max(np.abs(r["q1"][:2400])))))
        assert r["H_tot_max_rel_drift"] < 1e-11
        static_runs[lbl] = {
            "H_tot_max_rel_drift": r["H_tot_max_rel_drift"],
            "q1_late_over_early_peak": q1_ratio_st,
            "dominant_pole_E": r["dominant_pole_E"],
            "dominant_pole_frac": r["dominant_pole_frac"],
            **st,
        }
        print(f"  [static {lbl:<20}] H_drift={r['H_tot_max_rel_drift']:.2e} | M_Q ac(pi/2)={st['M_Q']['ac_pi_over_2']:+.3f} | late/early={st['late_over_early_mean']:.3f}")

    # 2. All 6 periodic P05 breathing runs + 3 EFT runs (Model A)
    breathing_runs = {}
    max_bg_e00 = bg_ring_info["max_rel_e00"]
    bg_cache = {}
    bg_configs = [
        ("P05_sig0_xring_b1", SIGMA_0, X_RING, 1.0, [0.0, 0.0, 1.0, 0.0]),
        ("P05_sig0_xring_b2", SIGMA_0, X_RING, 2.0, [0.0, 0.0, 1.0, 0.0]),
        ("P05_sig0_xring_grav", SIGMA_0, X_RING, 1.0, [1.0, 0.0, 0.0, 0.0]),
        ("P05_sig0_xb20p5_b1p5", SIGMA_0, 20.5, 1.5, [0.0, 0.0, 1.0, 0.0]),
        ("P05_sig0_xb18p0_b1p5", SIGMA_0, 18.0, 1.5, [0.0, 0.0, 1.0, 0.0]),
        ("P05_sig0_xb15p0_b1p5", SIGMA_0, 15.0, 1.5, [0.0, 0.0, 1.0, 0.0]),
        ("EFT_sig4_xring_b1", 4.0, math.sqrt(abs(KAPPA_V) - 1.0 + 4.0), 1.0, [0.0, 0.0, 1.0, 0.0]),
        ("EFT_sig6_xring_b1", 6.0, math.sqrt(abs(KAPPA_V) - 1.0 + 6.0), 1.0, [0.0, 0.0, 1.0, 0.0]),
        ("EFT_sig10_xring_b1", 10.0, math.sqrt(abs(KAPPA_V) - 1.0 + 10.0), 1.0, [0.0, 0.0, 1.0, 0.0]),
    ]
    raw_cache = {}
    for lbl, sig_t, xb, b0, y0_4d in bg_configs:
        key_bg = (round(sig_t, 8), round(xb, 8))
        if key_bg not in bg_cache:
            bg_cache[key_bg] = integrate_kam_background_eta(xb, ETA_END + 1.0, sig_t)
        bg_s, bg_i = bg_cache[key_bg]
        max_bg_e00 = max(max_bg_e00, bg_i["max_rel_e00"])
        etas_p = np.linspace(0.0, ETA_END, 2001)
        xs_p = bg_s.sol(etas_p)[0]
        om_min, om_max = float(np.min(xs_p) / math.sqrt(sig_t)), float(np.max(xs_p) / math.sqrt(sig_t))

        r = solve_discrete_aux_ode(bg_s, ETA_END, b0, spec_I30, kappa_mem=1.0, static_bg=False,
                                   sigma_t=sig_t, y0_4d=y0_4d, method="DOP853", rtol=1e-9, atol=1e-11, max_step=0.005)
        raw_cache[lbl] = r
        st = extract_talbot_and_drain(r["N_gamma"], r["M_Q"], r["eta"])
        q1_ratio = float(np.max(np.abs(r["q1"][-2400:])) / max(1e-30, float(np.max(np.abs(r["q1"][:2400])))))
        breathing_runs[lbl] = {
            "sigma_t": sig_t, "x_b": xb, "beta_0": b0,
            "bg_max_rel_e00": bg_i["max_rel_e00"],
            "Omega_scal_range": [om_min, om_max],
            "Delta_Omega_scal": om_max - om_min,
            "q1_late_over_early_peak": q1_ratio,
            "dominant_pole_E": r["dominant_pole_E"],
            "dominant_pole_frac": r["dominant_pole_frac"],
            **st,
        }
        print(f"  [breath {lbl:<20}] e00={bg_i['max_rel_e00']:.2e} | Om_s=[{om_min:5.2f},{om_max:5.2f}] | "
              f"late/early={st['late_over_early_mean']:7.3f} | q1_rat={q1_ratio:.3f}")

    # 3. Dual-Solver (DOP853 vs Radau) & Cutoff Convergence (J_cut=30 vs J_cut=60)
    print("  Running dual-solver (DOP853 vs Radau) & cutoff (J_cut=30 vs J_cut=60) cross-checks...")
    cross_checks = {}
    for lbl, sig_t, xb, b0, y0_4d in [
        ("P05_sig0_xring_b1", SIGMA_0, X_RING, 1.0, [0.0, 0.0, 1.0, 0.0]),
        ("P05_sig0_xb15p0_b1p5", SIGMA_0, 15.0, 1.5, [0.0, 0.0, 1.0, 0.0]),
        ("EFT_sig6_xring_b1", 6.0, math.sqrt(abs(KAPPA_V) - 1.0 + 6.0), 1.0, [0.0, 0.0, 1.0, 0.0]),
    ]:
        bg_s, _ = bg_cache[(round(sig_t, 8), round(xb, 8))]
        r_dop = raw_cache[lbl]
        r_rad = solve_discrete_aux_ode(bg_s, ETA_END, b0, spec_I30, kappa_mem=1.0, static_bg=False,
                                       sigma_t=sig_t, y0_4d=y0_4d, method="Radau", rtol=1e-9, atol=1e-11, max_step=0.01)
        rel_q1 = float(np.max(np.abs(r_dop["q1"] - r_rad["q1"])) / max(1e-30, float(np.max(np.abs(r_dop["q1"])))))
        rel_Ng = float(np.max(np.abs(r_dop["N_gamma"] - r_rad["N_gamma"])) / max(1e-30, float(np.max(np.abs(r_dop["N_gamma"])))))
        st_rad = extract_talbot_and_drain(r_rad["N_gamma"], r_rad["M_Q"], r_rad["eta"])
        cross_checks[f"dop853_vs_radau_{lbl}"] = {
            "rel_diff_q1": rel_q1,
            "rel_diff_N_gamma": rel_Ng,
            "late_over_early_dop": breathing_runs[lbl]["late_over_early_mean"],
            "late_over_early_radau": st_rad["late_over_early_mean"],
        }
        print(f"    [DOP853 vs Radau {lbl:<20}] rel_q1={rel_q1:.2e}, rel_Ng={rel_Ng:.2e}")
        assert rel_q1 < 1e-4 and rel_Ng < 1e-4

    for lbl, sig_t, xb, b0, static_flag, y0_4d in [
        ("Istar_full_b1_pulse", SIGMA_0, X_RING, 1.0, True, [0.0, 0.0, 1.0, 0.0]),
        ("P05_sig0_xring_b1", SIGMA_0, X_RING, 1.0, False, [0.0, 0.0, 1.0, 0.0]),
        ("EFT_sig6_xring_b1", 6.0, math.sqrt(abs(KAPPA_V) - 1.0 + 6.0), 1.0, False, [0.0, 0.0, 1.0, 0.0]),
    ]:
        bg_s = bg_ring if static_flag else bg_cache[(round(sig_t, 8), round(xb, 8))][0]
        r_30 = (
            solve_discrete_aux_ode(bg_s, ETA_END, b0, spec_I30, kappa_mem=1.0, static_bg=True,
                                   y0_4d=y0_4d, method="DOP853", rtol=1e-9, atol=1e-11, max_step=0.005)
            if static_flag else raw_cache[lbl]
        )
        r_60 = solve_discrete_aux_ode(bg_s, ETA_END, b0, spec_I60, kappa_mem=1.0, static_bg=static_flag,
                                      sigma_t=sig_t, y0_4d=y0_4d, method="DOP853", rtol=1e-9, atol=1e-11, max_step=0.004)
        st_60 = extract_talbot_and_drain(r_60["N_gamma"], r_60["M_Q"], r_60["eta"])
        rel_q1 = float(np.max(np.abs(r_30["q1"] - r_60["q1"])) / max(1e-30, float(np.max(np.abs(r_30["q1"])))))
        le_30 = static_runs[lbl]["late_over_early_mean"] if static_flag else breathing_runs[lbl]["late_over_early_mean"]
        cross_checks[f"J30_vs_J60_{lbl}"] = {
            "rel_diff_q1": rel_q1,
            "late_over_early_J30": le_30,
            "late_over_early_J60": st_60["late_over_early_mean"],
            "M_Q_ac_pi_over_2_J60": st_60["M_Q"]["ac_pi_over_2"],
        }
        print(f"    [J30 vs J60      {lbl:<20}] rel_q1={rel_q1:.2e}, late/early J30={le_30:.3f} vs J60={st_60['late_over_early_mean']:.3f}")

    # 4. Complete Model B shift of the EFT_sig6_xring_b1 opposite-Krein difference resonance!
    # Under Complete Model B (rel-S^3), at beta_0 = 1.0:
    #   Omega_grav shifts 2.7857 -> 4.1158, Omega_Stelle_lo shifts 12.3623 -> 10.1698,
    #   so Delta_Omega_Krein = Omega_Stelle_lo - Omega_grav drops from 9.5766 -> 6.0540!
    #   Wait: test Complete Model B (rel-S^3) dynamically at sigma_t = 6.0, 10.0, 14.0, 16.0 (beta_0 = 1.0)
    #   AND at sigma_t = 6.0, beta_0 = 0.75 (where Delta_Omega_Krein = 12.219 - 4.111 = 8.108 sits inside sig=6 band!)
    print("  Checking Complete Model B shift of opposite-Krein difference resonance...")
    pi_kin_rel = part2_data["models"]["Model_B_complete_relS3"]["Pi_kin"]
    pi_rem_rel = part2_data["models"]["Model_B_complete_relS3"]["Pi_rem"]
    # Static Complete Model B Hamiltonian conservation check at beta_0 = 1.0:
    r_mb_stat = solve_discrete_aux_ode(
        bg_ring, ETA_END, 1.0, spec_I30, kappa_mem=1.0, static_bg=True,
        pi_kin=pi_kin_rel, pi_rem=pi_rem_rel, y0_4d=[0.0, 0.0, 1.0, 0.0],
        method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.005, track_hamiltonian=True,
    )
    assert r_mb_stat["H_tot_max_rel_drift"] < 1e-11
    print(f"    [Complete Model B static b0=1.0] H_tot_max_rel_drift = {r_mb_stat['H_tot_max_rel_drift']:.2e}")

    modelB_eft_runs = {}
    for lbl, sig_t, b0 in [
        ("ModelB_EFT_sig6_b1p0", 6.0, 1.0),
        ("ModelB_EFT_sig6_b0p75", 6.0, 0.75),
        ("ModelB_EFT_sig10_b1p0", 10.0, 1.0),
        ("ModelB_EFT_sig14_b1p0", 14.0, 1.0),
        ("ModelB_EFT_sig16_b1p0", 16.0, 1.0),
    ]:
        xb = math.sqrt(abs(KAPPA_V) - 1.0 + sig_t)
        key_bg = (round(sig_t, 8), round(xb, 8))
        if key_bg not in bg_cache:
            bg_cache[key_bg] = integrate_kam_background_eta(xb, ETA_END + 1.0, sig_t)
        bg_s, bg_i = bg_cache[key_bg]
        etas_p = np.linspace(0.0, ETA_END, 2001)
        xs_p = bg_s.sol(etas_p)[0]
        om_min, om_max = float(np.min(xs_p) / math.sqrt(sig_t)), float(np.max(xs_p) / math.sqrt(sig_t))
        r_b = solve_discrete_aux_ode(
            bg_s, ETA_END, b0, spec_I30, kappa_mem=1.0, static_bg=False, sigma_t=sig_t,
            pi_kin=pi_kin_rel, pi_rem=pi_rem_rel, y0_4d=[0.0, 0.0, 1.0, 0.0],
            method="DOP853", rtol=1e-9, atol=1e-11, max_step=0.005,
        )
        st_b = extract_talbot_and_drain(r_b["N_gamma"], r_b["M_Q"], r_b["eta"])
        q1_ratio = float(np.max(np.abs(r_b["q1"][-2400:])) / max(1e-30, float(np.max(np.abs(r_b["q1"][:2400])))))
        # Measure actual breathing frequency Omega_scal,eff from FFT of x(eta)
        fx_bg = np.abs(np.fft.rfft(xs_p - np.mean(xs_p)))
        fr_bg = np.fft.rfftfreq(len(xs_p), d=float(etas_p[1] - etas_p[0])) * 2.0 * math.pi
        om_scal_eff = float(fr_bg[int(np.argmax(fx_bg[1:])) + 1])
        modelB_eft_runs[lbl] = {
            "sigma_t": sig_t, "beta_0": b0, "x_b": xb,
            "Omega_scal_range": [om_min, om_max],
            "Omega_scal_eff_fft": om_scal_eff,
            "Delta_Omega_Krein_ModelB_relS3": part2_data["models"]["Model_B_complete_relS3"]["roots"][str(b0)]["diff_Stelle_minus_grav"],
            "q1_late_over_early_peak": q1_ratio,
            "N_gamma_late_over_early_mean": st_b["late_over_early_mean"],
            "N_gamma_late_over_early_peak": st_b["late_over_early_peak"],
        }
        print(f"    [{lbl:<22}] Om_scal_eff={om_scal_eff:.2f} vs dOm_Krein={modelB_eft_runs[lbl]['Delta_Omega_Krein_ModelB_relS3']:.2f} | "
              f"q1_rat={q1_ratio:.3f}, Ng_late/early={st_b['late_over_early_mean']:.3f}")

    # Cross-check ModelB_EFT_sig14_b1p0 with Radau
    xb14 = math.sqrt(abs(KAPPA_V) - 1.0 + 14.0)
    bg14 = bg_cache[(14.0, round(xb14, 8))][0]
    r_b14_dop = solve_discrete_aux_ode(bg14, ETA_END, 1.0, spec_I30, kappa_mem=1.0, static_bg=False, sigma_t=14.0,
                                       pi_kin=pi_kin_rel, pi_rem=pi_rem_rel, y0_4d=[0.0, 0.0, 1.0, 0.0],
                                       method="DOP853", rtol=1e-9, atol=1e-11, max_step=0.005)
    r_b14_rad = solve_discrete_aux_ode(bg14, ETA_END, 1.0, spec_I30, kappa_mem=1.0, static_bg=False, sigma_t=14.0,
                                       pi_kin=pi_kin_rel, pi_rem=pi_rem_rel, y0_4d=[0.0, 0.0, 1.0, 0.0],
                                       method="Radau", rtol=1e-9, atol=1e-11, max_step=0.01)
    rel_q1_b14 = float(np.max(np.abs(r_b14_dop["q1"] - r_b14_rad["q1"])) / max(1e-30, float(np.max(np.abs(r_b14_dop["q1"])))))
    cross_checks["dop853_vs_radau_ModelB_EFT_sig14_b1p0"] = {"rel_diff_q1": rel_q1_b14}
    assert rel_q1_b14 < 1e-4

    return {
        "bg_ring_tau_at_10pi": tau_ring_10pi,
        "max_bg_rel_e00": max_bg_e00,
        "ModelB_static_H_tot_max_rel_drift": r_mb_stat["H_tot_max_rel_drift"],
        "static_runs": static_runs,
        "breathing_runs": breathing_runs,
        "cross_checks": cross_checks,
        "modelB_opposite_krein_shift": modelB_eft_runs,
    }


def main():
    with open(STREAM_DIR / "01_derive.json") as fh:
        derive_data = json.load(fh)

    part1 = run_part1_heat_kernel(derive_data)
    part2 = run_part2_complete_model_B(derive_data, part1)
    part3 = run_part3_coupled_talbot(part2)

    out_data = {
        "stream_id": "p2_06_poincare_shear_modulus_and_talbot_revivals",
        "stage": "02_solve",
        "part1_heat_kernel_shear_moduli": part1,
        "part2_complete_model_B": part2,
        "part3_coupled_talbot_ode": part3,
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
