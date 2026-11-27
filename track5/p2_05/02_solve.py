#!/usr/bin/env python3
"""
Stream p2_05_diamond_invariance_and_twisted_prolate - Stage 2 (02_solve.py, Round-2 Audited):
High-Precision Dual-Oracle Solvers for KT1--KT4:
  - Part 1 (KT1): Aliasing-free uncapped curved-S^3 = SU(2) nonlinear 4D null-geodesic shooting
                  across ALL 28 integer modes p = 3..30 on global [-pi/2, +pi/2] (zero linear
                  interpolation, explicit quadrature-doubling checks at p = 15, 19, 23, 30,
                  RK4 step-doubling, alpha-invariance checks), PLUS exact nonlinear vacuum
                  triaxial Bianchi IX (G_uv + 3 g_uv = 0, p = 3) fixed-affine 2D waist-area
                  ray-tracing verified by DOP853 vs. Radau, Hamiltonian constraint residual,
                  and ray-tracing step-doubling.
  - Part 2 (KT2): Dual 50-digit Hurwitz+digamma vs. Abel--Plana evaluation of spin-2 chiral
                  asymmetry (all 5 ADE + Maxwell control + exact uncapped mode-by-mode global
                  chiral transit) and orthogonal-ray p_par - p_perp (L(2,1), Q_8, L(3,1),
                  L(11,1), L(14,1), L(20,1) + exact rational Maxwell).
  - Part 3 (KT3): Finite-lambda_B Kahan-compensated Gaussian heat-kernel curves (ts_xfine and
                  ts_fine) on L(20,1) and L(24,1) across 6 field species, strictly tachyon-free
                  cubic spline on [0.60, 1.40] (0 dropped S^3 or L(q,1) modes), O(q^4)
                  cancellation check, radial Friedmann RHS audit, and 2-decade local C^2
                  scheme-sensitivity sweep at lam_* = 1.05.
  - Part 4 (KT4): Dual 50-digit twisted Dirac/Weyl verification (Cauchy contour vs. period-60
                  Hurwitz-zeta vs. exact Fraction rationals from 01_derive.json), Witten-anomaly-free
                  Hosotani vacuum inversion (1 YM + 1 adj Weyl [N=1 SYM] and 1 YM + 1 fund Dirac),
                  and graviton flip thresholds against ESU, coarse-grained BD, and KT1 global transit.

Outputs: streams/p2_05_diamond_invariance_and_twisted_prolate/02_solve.json
"""

import json
import math
import time
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

mp.mp.dps = 50

OUT_DIR = Path(__file__).resolve().parent
with open(OUT_DIR / "01_derive.json") as fh:
    DERIVE = json.load(fh)

SQRT_8PI = math.sqrt(8.0 * math.pi)
T0_ALL = time.time()


# =============================================================================
# SHARED CHARACTER & 50-DIGIT HURWITZ / ABEL-PLANA ENGINES
# =============================================================================

def ade_classes(name: str):
    if name == "S3":
        return 1, [(1, mp.mpf(0))]
    if name == "RP3":
        return 2, [(1, mp.mpf(0)), (1, mp.pi)]
    if name == "Q8":
        return 8, [(1, mp.mpf(0)), (1, mp.pi), (6, mp.pi / 2)]
    if name == "T*":
        return 24, [(1, mp.mpf(0)), (1, mp.pi), (6, mp.pi / 2), (8, mp.pi / 3), (8, 2 * mp.pi / 3)]
    if name == "O*":
        return 48, [
            (1, mp.mpf(0)), (1, mp.pi), (18, mp.pi / 2),
            (8, mp.pi / 3), (8, 2 * mp.pi / 3), (6, mp.pi / 4), (6, 3 * mp.pi / 4),
        ]
    if name == "I*":
        return 120, [
            (1, mp.mpf(0)), (1, mp.pi), (30, mp.pi / 2),
            (20, mp.pi / 3), (20, 2 * mp.pi / 3),
            (12, mp.pi / 5), (12, 2 * mp.pi / 5), (12, 3 * mp.pi / 5), (12, 4 * mp.pi / 5),
        ]
    raise ValueError(name)


def su2_chi(two_j: int, theta: mp.mpf) -> mp.mpf:
    if two_j < 0:
        return mp.mpf(0)
    if abs(theta) < mp.mpf("1e-35"):
        return mp.mpf(two_j + 1)
    if abs(theta - mp.pi) < mp.mpf("1e-35"):
        return mp.mpf(two_j + 1) * ((-1) ** two_j)
    return mp.sin((two_j + 1) * theta) / mp.sin(theta)


def m_ade(name: str, two_j: int) -> int:
    if two_j < 0:
        return 0
    order, cls = ade_classes(name)
    val = sum(sz * su2_chi(two_j, th) for sz, th in cls) / order
    iv = int(mp.nint(val))
    assert abs(val - iv) < mp.mpf("1e-40")
    return iv


def ade_tt_mult(name: str, p: int) -> int:
    return m_ade(name, p - 3) * (p + 2) + m_ade(name, p + 1) * (p - 2)


def m_lens(q_ord: int, two_j: int) -> int:
    if two_j < 0:
        return 0
    return sum(1 for two_m in range(-two_j, two_j + 1, 2) if two_m % q_ord == 0)


def extract_quasi_poly_deg2(R_fn, P: int, p_min: int = 3):
    C_r, A_r, B_r = {}, {}, {}
    for r in range(P):
        p1 = p_min + ((r - p_min) % P)
        Rs = [R_fn(p1 + m * P) for m in range(4)]
        d2_1 = (Rs[2] - Rs[1]) - (Rs[1] - Rs[0])
        d2_2 = (Rs[3] - Rs[2]) - (Rs[2] - Rs[1])
        assert d2_1 == d2_2 and d2_1 % (2 * P * P) == 0
        c = d2_1 // (2 * P * P)
        L1 = Rs[0] - c * p1 * p1
        L2 = Rs[1] - c * (p1 + P) * (p1 + P)
        assert (L2 - L1) % P == 0
        a = (L2 - L1) // P
        b = L1 - a * p1
        C_r[r], A_r[r], B_r[r] = c, a, b
    for pp in range(p_min, p_min + 3 * P + 5):
        r = pp % P
        assert R_fn(pp) == C_r[r] * pp * pp + A_r[r] * pp + B_r[r]
    return C_r, A_r, B_r


def sum_periodic_pow(coeff_dict, P: int, s_pow: int, p_min: int = 3) -> mp.mpf:
    acc = mp.mpf(0)
    if s_pow == 1:
        for r, cv in coeff_dict.items():
            if cv == 0:
                continue
            p0 = p_min + ((r - p_min) % P)
            acc += cv * (-mp.psi(0, mp.mpf(p0) / P)) / P
    else:
        scale = mp.mpf(P) ** (-s_pow)
        for r, cv in coeff_dict.items():
            if cv == 0:
                continue
            p0 = p_min + ((r - p_min) % P)
            acc += cv * mp.zeta(s_pow, mp.mpf(p0) / P) * scale
    return acc


def hurwitz_digamma_EH_general(C_r, A_r, B_r, P: int, p_min: int = 3):
    assert sum(C_r.values()) == 0 and sum(A_r.values()) == 0 and sum(B_r.values()) == 0
    s_conf_frac = Fraction(0)
    for r in range(P):
        p0 = p_min + ((r - p_min) % P)
        x = Fraction(p0, P)
        z_m3 = -(x**4 - 2 * x**3 + x**2 - Fraction(1, 30)) / 4
        z_m2 = -(x**3 - Fraction(3, 2) * x**2 + Fraction(1, 2) * x) / 3
        z_m1 = -(x**2 - x + Fraction(1, 6)) / 2
        s_conf_frac += C_r[r] * (P**3) * z_m3 + A_r[r] * (P**2) * z_m2 + B_r[r] * P * z_m1
    pi2_conf_frac = s_conf_frac / 4

    def eval_pow(pow_p):
        return (
            sum_periodic_pow(C_r, P, -(pow_p + 2), p_min)
            + sum_periodic_pow(A_r, P, -(pow_p + 1), p_min)
            + sum_periodic_pow(B_r, P, -pow_p, p_min)
        )

    s_p1 = eval_pow(1)
    s_pm1 = eval_pow(-1)
    s_eh = s_p1
    for k in range(1, 55):
        dk = ((-1) ** k) * mp.binomial(mp.mpf("0.5"), k) * eval_pow(1 - 2 * k)
        s_eh += dk
        if abs(dk) < mp.mpf("1e-47"):
            break

    S_sup_raw = mp.mpf(0)
    for r in range(P):
        p0 = mp.mpf(p_min + ((r - p_min) % P))
        q, qm, qp = p0 / P, (p0 - 1) / P, (p0 + 1) / P
        C, A, B = mp.mpf(C_r[r]), mp.mpf(A_r[r]), mp.mpf(B_r[r])
        S_sup_raw += (
            C * (-mp.mpf("0.5") / P * (mp.psi(0, qm) + mp.psi(0, qp)))
            + A * (-mp.mpf("0.5") / P * (mp.psi(0, qm) - mp.psi(0, qp)))
            + B * (-mp.mpf("0.5") / P * (mp.psi(0, qm) + mp.psi(0, qp)) + (1 / P) * mp.psi(0, q))
        )
    s_p0 = eval_pow(0)
    return s_eh / 4, s_p1 / 4, pi2_conf_frac, s_pm1, S_sup_raw, s_p0


def abel_plana_EH_general(C_r, A_r, B_r, P: int, p_min: int = 3) -> mp.mpf:
    def F0(x):
        return mp.mpf("0.5") * (x * mp.sqrt(x * x - 1) - mp.acosh(x))

    def F1(x):
        return (x * x - 1) ** (mp.mpf(3) / 2) / 3

    def F2(x):
        return (x * (2 * x * x - 1) * mp.sqrt(x * x - 1) - mp.acosh(x)) / 8

    p0_list = []
    int_part, bdy_part = mp.mpf(0), mp.mpf(0)
    for r in range(P):
        c, a, b = C_r[r], A_r[r], B_r[r]
        if c == 0 and a == 0 and b == 0:
            continue
        p0 = mp.mpf(p_min + ((r - p_min) % P))
        p0_list.append((p0, mp.mpf(c), mp.mpf(a), mp.mpf(b)))
        int_part += -(c * F2(p0) + a * F1(p0) + b * F0(p0)) / P
        bdy_part += mp.mpf("0.5") * (c * p0 * p0 + a * p0 + b) * mp.sqrt(p0 * p0 - 1)
    P_m = mp.mpf(P)

    def integrand(y):
        tot_im = mp.mpf(0)
        for p0, c, a, b in p0_list:
            z = p0 + 1j * P_m * y
            tot_im += mp.im((c * z * z + a * z + b) * mp.sqrt(z * z - 1))
        return -2 * tot_im / mp.expm1(2 * mp.pi * y)

    ap_part = mp.quad(integrand, [0, 1, 4, 12, mp.inf])
    return (int_part + bdy_part + ap_part) / 4


# =============================================================================
# PART 1 (KT1): UNCAPPED GLOBAL SHOOTING (ALL p=3..30) + EXACT BIANCHI IX FIXED-AFFINE
# =============================================================================
print("=" * 86)
print("PART 1 (KT1): UNCAPPED GLOBAL SHOOTING (ALL p=3..30) & EXACT BIANCHI IX FIXED-AFFINE")
print("=" * 86)

t1 = time.time()


def shoot_curved_S3_mode_uncapped(
    p_val: int, eta0: float, eta1: float, alpha_val: float = 0.5, eps: float = 1e-3,
    n_steps: int = None, quad_bump: tuple = (0, 0, 0),
):
    """Aliasing-free uncapped curved-S^3 = SU(2) 4D null-geodesic shooting."""
    L_seg = eta1 - eta0
    m = p_val - 3
    if n_steps is None:
        min_steps = 400 if L_seg > 1.0 else 96
        n_steps = max(min_steps, int(math.ceil(32 * p_val * L_seg / (2.0 * math.pi))) * 2)

    if m == 0:
        u_pts, u_wts = np.array([1.0]), np.array([1.0])
        n_u, n_mu, n_phi = 1, 8 + quad_bump[1], 8 + quad_bump[2]
    else:
        n_u = max(8, m // 2 + 5) + quad_bump[0]
        xg, wg = np.polynomial.legendre.leggauss(n_u)
        u_pts, u_wts = 0.5 * (xg + 1.0), 0.5 * wg
        n_mu = max(12, m + 5) + quad_bump[1]
        n_phi = max(16, 2 * m + 6) + quad_bump[2]

    mg, wmg = np.polynomial.legendre.leggauss(n_mu)
    phis = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    UU, MM, PP = np.meshgrid(u_pts, mg, phis, indexing="ij")
    WU, WM, _ = np.meshgrid(u_wts, 0.5 * wmg, phis, indexing="ij")
    u_flat, mu_flat, phi_flat = UU.ravel(), MM.ravel(), PP.ravel()
    wt_flat = (WU * WM / n_phi).ravel()
    N = len(u_flat)

    cb = np.sqrt(u_flat)
    sb = np.sqrt(np.maximum(0.0, 1.0 - u_flat))
    q_start = np.stack([cb, np.zeros(N), sb, np.zeros(N)], axis=-1)
    sth = np.sqrt(np.maximum(0.0, 1.0 - mu_flat * mu_flat))
    n_hat = np.stack([sth * np.cos(phi_flat), sth * np.sin(phi_flat), mu_flat], axis=-1)

    def qmul_vec(q, r):
        return np.stack([
            q[:, 0]*r[:, 0] - q[:, 1]*r[:, 1] - q[:, 2]*r[:, 2] - q[:, 3]*r[:, 3],
            q[:, 0]*r[:, 1] + q[:, 1]*r[:, 0] + q[:, 2]*r[:, 3] - q[:, 3]*r[:, 2],
            q[:, 0]*r[:, 2] - q[:, 1]*r[:, 3] + q[:, 2]*r[:, 0] + q[:, 3]*r[:, 1],
            q[:, 0]*r[:, 3] + q[:, 1]*r[:, 2] - q[:, 2]*r[:, 1] + q[:, 3]*r[:, 0],
        ], axis=-1)

    def rhs_vec(t, q, pv, phase, e_val):
        qn = q / np.linalg.norm(q, axis=-1, keepdims=True)
        ang = -p_val * t + phase
        eh = e_val * (p_val * math.cos(t) * math.cos(ang) - math.sin(t) * math.sin(ang))
        z1 = qn[:, 0] + 1j * qn[:, 3]
        if m == 0:
            w = np.ones(N, dtype=complex)
        else:
            wm1 = z1 ** (m - 1)
            w = wm1 * z1
        u, v = 0.5 * eh * w.real, 0.5 * eh * w.imag
        r2 = u * u + v * v
        diag_al = 1.0 + alpha_val * r2
        g00, g11, g01 = diag_al + u, diag_al - u, v
        inv_det = 1.0 / (g00 * g11 - g01 * g01)
        gp0, gp1, gp2 = (g11 * pv[:, 0] - g01 * pv[:, 1]) * inv_det, (-g01 * pv[:, 0] + g00 * pv[:, 1]) * inv_det, pv[:, 2]
        E = np.sqrt(pv[:, 0] * gp0 + pv[:, 1] * gp1 + pv[:, 2] * gp2)
        v0, v1, v2 = gp0 / E, gp1 / E, gp2 / E
        dq = np.stack([
            -(qn[:, 1]*v0 + qn[:, 2]*v1 + qn[:, 3]*v2),
            qn[:, 0]*v0 + qn[:, 2]*v2 - qn[:, 3]*v1,
            qn[:, 0]*v1 + qn[:, 3]*v0 - qn[:, 1]*v2,
            qn[:, 0]*v2 + qn[:, 1]*v1 - qn[:, 2]*v0,
        ], axis=-1)
        dp0 = 2.0 * (pv[:, 1]*v2 - pv[:, 2]*v1)
        dp1 = 2.0 * (pv[:, 2]*v0 - pv[:, 0]*v2)
        dp2 = 2.0 * (pv[:, 0]*v1 - pv[:, 1]*v0)
        if m > 0:
            v_diff, v_2xy, v_sum, half_E = v0*v0 - v1*v1, 2.0*v0*v1, v0*v0 + v1*v1, 0.5*E
            mw = m * wm1
            for idx, dwk in enumerate((mw*(-qn[:, 1] - 1j*qn[:, 2]), mw*(-qn[:, 2] + 1j*qn[:, 1]), mw*(-qn[:, 3] + 1j*qn[:, 0]))):
                duk, dvk = 0.5*eh*dwk.real, 0.5*eh*dwk.imag
                force = half_E * (v_diff*duk + v_2xy*dvk + alpha_val*v_sum*2.0*(u*duk + v*dvk))
                if idx == 0: dp0 = dp0 + force
                elif idx == 1: dp1 = dp1 + force
                else: dp2 = dp2 + force
        return dq, np.stack([dp0, dp1, dp2], axis=-1)

    q_inv = np.stack([q_start[:, 0], -q_start[:, 1], -q_start[:, 2], -q_start[:, 3]], axis=-1)
    q_targ_inv = np.stack([
        np.full(N, math.cos(L_seg)),
        -n_hat[:, 0] * math.sin(L_seg),
        -n_hat[:, 1] * math.sin(L_seg),
        -n_hat[:, 2] * math.sin(L_seg),
    ], axis=-1)

    def integrate_batch(n_dir, phase, e_val):
        q, pv, dt = q_start.copy(), n_dir.copy(), L_seg / n_steps
        for step in range(n_steps):
            t = eta0 + step * dt
            dq1, dp1 = rhs_vec(t, q, pv, phase, e_val)
            dq2, dp2 = rhs_vec(t + 0.5*dt, q + 0.5*dt*dq1, pv + 0.5*dt*dp1, phase, e_val)
            dq3, dp3 = rhs_vec(t + 0.5*dt, q + 0.5*dt*dq2, pv + 0.5*dt*dp2, phase, e_val)
            dq4, dp4 = rhs_vec(t + dt, q + dt*dq3, pv + dt*dp3, phase, e_val)
            q = q + (dt / 6.0) * (dq1 + 2.0*dq2 + 2.0*dq3 + dq4)
            pv = pv + (dt / 6.0) * (dp1 + 2.0*dp2 + 2.0*dp3 + dp4)
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        return qmul_vec(q_targ_inv, qmul_vec(q_inv, q))

    def shoot_batch(phase, e_val):
        q_err = integrate_batch(n_hat, phase, e_val)
        d_par = np.sum(q_err[:, 1:] * n_hat, axis=-1)
        d_perp = q_err[:, 1:] - d_par[:, None] * n_hat
        if abs(math.sin(L_seg)) > 0.05:
            cL, sL = math.cos(L_seg), math.sin(L_seg)
            n_hit = n_hat.copy()
            for _ in range(2):
                dn = -(cL * d_perp + sL * np.cross(n_hat, d_perp)) / sL
                n_hit = n_hit + dn
                n_hit = n_hit / np.linalg.norm(n_hit, axis=-1, keepdims=True)
                q_err = integrate_batch(n_hit, phase, e_val)
                d_par = np.sum(q_err[:, 1:] * n_hat, axis=-1)
                d_perp = q_err[:, 1:] - d_par[:, None] * n_hat
        miss = float(np.max(np.linalg.norm(d_perp, axis=-1)))
        return d_par, miss

    int_fp2 = 0.5 * (L_seg + (p_val*p_val - 1.0) * (0.5*L_seg + 0.25*(math.sin(2.0*eta1) - math.sin(2.0*eta0))))
    int_h2_exact = (1.0 / (2.0 * (m + 1))) * int_fp2

    acc, max_miss = 0.0, 0.0
    for phase in (0.0, 0.5 * math.pi):
        dp_p, mp_p = shoot_batch(phase, +eps)
        dp_m, mp_m = shoot_batch(phase, -eps)
        acc += 0.5 * float(np.sum(wt_flat * (0.5 * (dp_p + dp_m) / (eps * eps))))
        max_miss = max(max_miss, mp_p, mp_m)

    c_geo = acc / int_h2_exact
    c_inv = c_geo + alpha_val / 6.0
    return {
        "p": p_val, "alpha": alpha_val, "c_geo_alpha": c_geo, "c_inv": c_inv,
        "diff_from_schur_3_20": c_inv - 0.15, "diff_from_wkb_1_6": c_inv - 1.0 / 6.0,
        "max_endpoint_miss": max_miss, "n_steps": n_steps, "quad_grid": [n_u, n_mu, n_phi], "n_rays": N,
    }


# Shoot ALL 28 integer modes p = 3..30 directly on global [-pi/2, +pi/2] (ZERO linear interpolation!)
global_all_28 = []
for pv in range(3, 31):
    global_all_28.append(shoot_curved_S3_mode_uncapped(pv, -math.pi/2, math.pi/2, alpha_val=0.5))

# Explicit quadrature-refinement convergence checks at p = 15, 19, 23, 30 (P1-1 remediation!)
quad_conv_checks = {}
for pv in [15, 19, 23, 30]:
    r_base = global_all_28[pv - 3]
    r_ref = shoot_curved_S3_mode_uncapped(pv, -math.pi/2, math.pi/2, alpha_val=0.5, quad_bump=(3, 5, 6))
    diff = abs(r_ref["c_inv"] - r_base["c_inv"])
    assert diff < 1e-5, f"Quadrature convergence failed at p={pv}: {diff}"
    quad_conv_checks[f"p_{pv}"] = {
        "base_grid": r_base["quad_grid"], "ref_grid": r_ref["quad_grid"],
        "c_inv_base": r_base["c_inv"], "c_inv_ref": r_ref["c_inv"], "abs_diff": diff,
    }

# RK4 step-doubling & alpha-invariance checks on global & short-segment spatial arrival c_inv
chk_p3_800 = shoot_curved_S3_mode_uncapped(3, -math.pi/2, math.pi/2, alpha_val=0.5, n_steps=800)
p3_glob_rk4_err = abs(global_all_28[0]["c_inv"] - chk_p3_800["c_inv"])
chk_p15_a0 = shoot_curved_S3_mode_uncapped(15, -math.pi/2, math.pi/2, alpha_val=0.0)
p15_glob_alpha_err = abs(chk_p15_a0["c_inv"] - global_all_28[12]["c_inv"])

# Short-segment spatial arrival c_inv^chi(p) on centered [-pi/20, +pi/20] for p = 3, 11, 15, 23, 30 (demonstrating endpoint-aiming shear suppression)
short_spatial_arrival = {}
for pv in [3, 11, 15, 23, 30]:
    short_spatial_arrival[str(pv)] = shoot_curved_S3_mode_uncapped(pv, -math.pi/20, math.pi/20, alpha_val=0.5)
chk_p11_ctr_192 = shoot_curved_S3_mode_uncapped(11, -math.pi/20, math.pi/20, alpha_val=0.5, n_steps=192)
p11_ctr_rk4_err = abs(short_spatial_arrival["11"]["c_inv"] - chk_p11_ctr_192["c_inv"])
chk_p11_ctr_a0 = shoot_curved_S3_mode_uncapped(11, -math.pi/20, math.pi/20, alpha_val=0.0)
chk_p11_ctr_a1 = shoot_curved_S3_mode_uncapped(11, -math.pi/20, math.pi/20, alpha_val=1.0)
p11_ctr_alpha_err = max(
    abs(chk_p11_ctr_a0["c_inv"] - short_spatial_arrival["11"]["c_inv"]),
    abs(chk_p11_ctr_a1["c_inv"] - short_spatial_arrival["11"]["c_inv"]),
)
max_quad_diff = max(v["abs_diff"] for v in quad_conv_checks.values())
assert p3_glob_rk4_err < 2e-5 and p11_ctr_rk4_err < 1e-5 and p15_glob_alpha_err < 1e-5 and p11_ctr_alpha_err < 1e-5
print(f"  Uncapped global shooting (ALL 28 modes p=3..30) completed in {time.time() - t1:.1f}s")
print(f"  Quadrature-refinement max diff (p=15,19,23,30) = {max_quad_diff:.2e} | RK4 p=3 diff = {p3_glob_rk4_err:.2e} | alpha spread p=15 = {p15_glob_alpha_err:.2e}")
print(f"  Monotonic global c_inv: p=3:{global_all_28[0]['c_inv']:.6f}, p=11:{global_all_28[8]['c_inv']:.6f}, p=15:{global_all_28[12]['c_inv']:.6f}, p=19:{global_all_28[16]['c_inv']:.6f}, p=23:{global_all_28[20]['c_inv']:.6f}, p=30:{global_all_28[27]['c_inv']:.6f}")

# Exact 50-digit S^3/I* topological spectral sums
Cr_I, Ar_I, Br_I = extract_quasi_poly_deg2(lambda pp: 120 * ade_tt_mult("I*", pp) - 2 * (pp * pp - 4), 60, 3)
u_eh_I, u_conf_I, _, s_pm1_I, s_sup_I_raw, _ = hurwitz_digamma_EH_general(Cr_I, Ar_I, Br_I, 60, 3)
S_p1_I = 2 * u_conf_I
S_inv_I = mp.mpf("0.5") * s_pm1_I
S_sup_I = mp.mpf("0.5") * s_sup_I_raw
pi2_C_rad_I = mp.mpf("0.5") * (S_p1_I - S_inv_I)
pi2_C_curv_bd_I = mp.mpf("0.125") * S_inv_I
pi2_C_curv_rho_I = mp.mpf("0.25") * S_inv_I

pi2_C_EM_I = -mp.mpf(22369) / 720
pi2_C_YM_I = {
    "rho_0": (120 * (-mp.mpf(349) / 225) - 3 * mp.mpf(11) / 120) / 2,
    "rho_1": (120 * (-mp.mpf(221) / 150) - 3 * mp.mpf(11) / 120) / 2,
    "rho_2": (120 * (mp.mpf(79) / 150) - 3 * mp.mpf(11) / 120) / 2,
}
ratio_rho1_I = float(pi2_C_YM_I["rho_1"] / pi2_C_EM_I)

# Exact uncapped mode-by-mode global transit Delta_eta_tot (ZERO linear interpolation across p=3..30!)
pi2_int_h2_I = mp.pi * (S_inv_I + 2 * S_sup_I)
pi2_deta_BR_FP_I = (mp.pi / 4) * pi2_C_rad_I + (mp.pi / 6) * pi2_C_curv_rho_I
pi2_deta_EM_I = (mp.pi / 4) * pi2_C_EM_I


def eval_glob_exact28(c_tail: float):
    corr = 0.0
    for pv in range(3, 31):
        R_p = 120 * ade_tt_mult("I*", pv) - 2 * (pv * pv - 4)
        c_p = global_all_28[pv - 3]["c_inv"]
        w_p = (0.5 * math.pi) * R_p * (pv * pv + 1.0) / (pv * (pv * pv - 1.0))
        corr += (c_p - c_tail) * w_p
    return float(pi2_deta_BR_FP_I) + (c_tail - 1.0 / 12.0) * float(pi2_int_h2_I) + corr, corr


glob_wkb, gcorr_wkb = eval_glob_exact28(1.0 / 6.0)
glob_sch, gcorr_sch = eval_glob_exact28(0.15)
x_glob_wkb = -glob_wkb / float(pi2_deta_EM_I)
x_glob_sch = -glob_sch / float(pi2_deta_EM_I)
assert math.ceil(x_glob_wkb) == 5 and math.ceil(x_glob_sch) == 5
print(f"  [GLOBAL EXACT 28 MODES] Delta_eta_tot = {glob_wkb:+.5f} (WKB tail) / {glob_sch:+.5f} (Schur tail) -> x = {x_glob_wkb:.5f} / {x_glob_sch:.5f} => N_V^crit = 5 (6 eliminated!), N_SU(2) = 2")

global_summary = {
    "pi2_deta_BR_FP": float(pi2_deta_BR_FP_I),
    "pi2_deta_tot_const_schur": float(pi2_deta_BR_FP_I + (mp.mpf(1) / 15) * pi2_int_h2_I),
    "pi2_deta_tot_const_wkb": float(pi2_deta_BR_FP_I + (mp.mpf(1) / 12) * pi2_int_h2_I),
    "pi2_deta_tot_mode_exact_wkb_tail": glob_wkb,
    "pi2_deta_tot_mode_exact_schur_tail": glob_sch,
    "mode_corr_wkb_tail": gcorr_wkb,
    "mode_corr_schur_tail": gcorr_sch,
    "pi2_deta_EM": float(pi2_deta_EM_I),
    "x_FP": float(-pi2_deta_BR_FP_I / pi2_deta_EM_I),
    "x_const_schur": float(-(pi2_deta_BR_FP_I + (mp.mpf(1) / 15) * pi2_int_h2_I) / pi2_deta_EM_I),
    "x_const_wkb": float(-(pi2_deta_BR_FP_I + (mp.mpf(1) / 12) * pi2_int_h2_I) / pi2_deta_EM_I),
    "x_mode_exact_wkb_tail": x_glob_wkb,
    "x_mode_exact_schur_tail": x_glob_sch,
    "N_V_crit_mode_exact_wkb_tail": math.ceil(x_glob_wkb),
    "N_V_crit_mode_exact_schur_tail": math.ceil(x_glob_sch),
    "x_SU2_rho0_mode_exact": x_glob_wkb / 3.0,
    "x_SU2_rho1_mode_exact": x_glob_wkb / ratio_rho1_I,
    "N_SU2_crit_rho0": math.ceil(x_glob_wkb / 3.0),
    "N_SU2_crit_rho1": math.ceil(x_glob_wkb / ratio_rho1_I),
    "max_caustic_transverse_miss_p3_to_30": max(r["max_endpoint_miss"] for r in global_all_28),
}

# Coarse-grained causal-diamond benchmarks (p2_03) & exact 50-digit isotropic delta-alpha cancellation on S^3/I*
H2_I = lambda e: 2 * (mp.cos(e) ** 2) * S_inv_I + 2 * S_sup_I
H2p_I = lambda e: -4 * mp.cos(e) * mp.sin(e) * S_inv_I


def on_shell_dS_g_ds(eta_0: mp.mpf, chi_max: mp.mpf):
    rho_S = mp.sin(chi_max) / mp.cos(eta_0 + chi_max)
    I_a4 = lambda x: mp.quad(lambda t: mp.sin(t) * mp.cos(eta_0 + t) ** 3, [0, x])
    I_a2 = lambda x: mp.quad(lambda t: mp.sin(t) * mp.cos(eta_0 + t), [0, x])
    J_a4 = mp.quad(lambda x: I_a4(x) / (mp.cos(eta_0 + x) ** 2), [0, chi_max])
    J_a2 = mp.quad(lambda x: I_a2(x) / (mp.cos(eta_0 + x) ** 2), [0, chi_max])
    return 4 * rho_S * J_a4, 4 * rho_S * J_a2


def verify_isotropic_alpha_Istar_50dps(eta_0: mp.mpf, chi_max: mp.mpf):
    eta_w = eta_0 + chi_max
    rho0 = mp.sin(chi_max) / mp.cos(eta_w)
    a0 = 1 / mp.cos(eta_0)

    def d_q_ibp(chi):
        ep = eta_0 + chi
        return -a0 * (
            mp.sin(chi) * mp.cos(ep) * H2p_I(ep)
            - mp.cos(chi) * mp.cos(ep) * H2_I(ep)
            + mp.cos(eta_0) * H2_I(eta_0)
            - mp.sin(eta_0) * mp.quad(H2_I, [eta_0, ep])
        )

    br_unit = -(1 / rho0) * mp.quad(lambda x: d_q_ibp(x) / (a0 * mp.cos(eta_0 + x) ** 2), [0, chi_max])
    int_H2 = mp.quad(H2_I, [eta_0, eta_w])
    int_sec2 = mp.quad(lambda e: H2_I(e) / (mp.cos(e) ** 2), [eta_0, eta_w])
    dir_unit = -H2_I(eta_0) - H2_I(eta_w) + mp.cot(chi_max) * int_H2 + (mp.cos(eta_0) * mp.cos(eta_w) / mp.sin(chi_max)) * int_sec2
    canc_err = abs(br_unit + dir_unit)
    vals = []
    for al in [mp.mpf("-1"), mp.mpf("0"), mp.mpf("0.25"), mp.mpf("0.5"), mp.mpf("1"), mp.mpf("1.5")]:
        da = al - mp.mpf("0.5")
        vals.append(-16 * (rho0 ** 2) * (da / 6) * (br_unit + dir_unit))
    return {
        "br_unit": float(br_unit),
        "isotropic_canc_err_50dps": float(canc_err),
        "isotropic_6alpha_spread_50dps": float(max(abs(v) for v in vals)),
    }


diamond_coarse_benchmarks = {}
for cfg_name, e0_m in [
    ("centered_[-pi/20,+pi/20]", -mp.pi / 20),
    ("tip@bounce_[0,+pi/10]", mp.mpf(0)),
    ("waist@bounce_[-pi/10,0]", -mp.pi / 10),
]:
    chi_m = mp.pi / 10
    rho0 = mp.sin(chi_m) / mp.cos(e0_m + chi_m)
    gD4, gD2 = on_shell_dS_g_ds(e0_m, chi_m)
    ds_BR_FP = float(-((mp.mpf(8) / 3) * pi2_C_rad_I * gD4 + (mp.mpf(8) / 3) * pi2_C_curv_bd_I * gD2))
    ds_EM = float(-((mp.mpf(8) / 3) * pi2_C_EM_I) * gD4)
    iso = verify_isotropic_alpha_Istar_50dps(e0_m, chi_m)
    ds_BR_LIN = ds_BR_FP - 16.0 * float(rho0 ** 2) * (-0.5 / 6.0) * iso["br_unit"]
    x_FP = -ds_BR_FP / ds_EM
    x_LIN = -ds_BR_LIN / ds_EM
    diamond_coarse_benchmarks[cfg_name] = {
        "Delta_s_BR_FP": ds_BR_FP, "Delta_s_BR_LIN_alpha_dep": ds_BR_LIN, "Delta_s_EM_1photon": ds_EM,
        "x_FP_BR": x_FP, "x_LIN_BR": x_LIN, "N_V_crit_FP_BR": math.ceil(x_FP),
        "x_SU2_rho0_FP": x_FP / 3.0, "x_SU2_rho1_FP": x_FP / ratio_rho1_I, "N_SU2_crit_FP": math.ceil(x_FP / 3.0),
        "isotropic_alpha_canc_err_50dps": iso["isotropic_canc_err_50dps"],
        "isotropic_6alpha_spread_50dps": iso["isotropic_6alpha_spread_50dps"],
    }

# Exact First-Principles Nonlinear Vacuum Triaxial Bianchi IX (p = 3, G_uv + 3 g_uv = 0)
# Fixed-Affine 2D Waist-Area Ray-Tracing with DOP853 vs Radau, Hamiltonian Constraint, & Step-Doubling
chi_b9 = math.pi / 10.0
T_half_b9 = math.asinh(math.tan(chi_b9))
u_star_b9 = math.sin(chi_b9)


def run_exact_bianchi9_fixed_affine_p3():
    def solve_b9(eps_v, phase, method="DOP853"):
        f0, fp0 = 3.0 * math.cos(phase), 8.0 * math.sin(phase)
        b0, bp0 = 0.5 * eps_v * f0 / math.sqrt(2.0), 0.5 * eps_v * fp0 / math.sqrt(2.0)
        e1, e2, e3 = math.exp(b0), math.exp(-b0), 1.0
        R_unit = 2.0 * ((e1**4 - (e2**2 - e3**2)**2) + (e2**4 - (e3**2 - e1**2)**2) + (e3**4 - (e1**2 - e2**2)**2)) / (e1*e1*e2*e2*e3*e3)
        a0 = math.sqrt(0.5 * R_unit / (3.0 + bp0 * bp0))
        y0 = [math.log(a0) + b0, math.log(a0) - b0, math.log(a0), bp0, -bp0, 0.0]

        def rhs(t, y):
            al1, al2, al3, H1, H2, H3 = y
            a1, a2, a3 = math.exp(al1), math.exp(al2), math.exp(al3)
            V2 = (a1 * a2 * a3) ** 2
            R1 = 2.0 * (a1**4 - (a2**2 - a3**2)**2) / V2
            R2 = 2.0 * (a2**4 - (a3**2 - a1**2)**2) / V2
            R3 = 2.0 * (a3**4 - (a1**2 - a2**2)**2) / V2
            Hs = H1 + H2 + H3
            return [H1, H2, H3, 3.0 - H1*Hs - R1, 3.0 - H2*Hs - R2, 3.0 - H3*Hs - R3]

        rt, at = (1e-13, 1e-15) if method == "DOP853" else (1e-12, 1e-14)
        sol_p = solve_ivp(rhs, [0.0, T_half_b9 + 0.05], y0, method=method, rtol=rt, atol=at, dense_output=True)
        sol_m = solve_ivp(rhs, [0.0, -(T_half_b9 + 0.05)], y0, method=method, rtol=rt, atol=at, dense_output=True)
        return sol_p, sol_m

    def hamiltonian_residual(sol_p, sol_m):
        max_res = 0.0
        for tt in np.linspace(-T_half_b9, T_half_b9, 41):
            y = sol_p.sol(tt) if tt >= 0 else sol_m.sol(tt)
            a1, a2, a3 = math.exp(y[0]), math.exp(y[1]), math.exp(y[2])
            H1, H2, H3 = y[3], y[4], y[5]
            V2 = (a1 * a2 * a3) ** 2
            R3 = 2.0 * ((a1**4 - (a2**2 - a3**2)**2) + (a2**4 - (a3**2 - a1**2)**2) + (a3**4 - (a1**2 - a2**2)**2)) / V2
            res = abs((H1*H2 + H2*H3 + H3*H1) + 0.5 * R3 - 3.0)
            max_res = max(max_res, res)
        return max_res

    def fixed_affine_area(eps_v, phase, n_steps=100, method="DOP853"):
        sol_p, sol_m = solve_b9(eps_v, phase, method=method)
        ham_res = hamiltonian_residual(sol_p, sol_m)
        def get_a(t):
            y = sol_p.sol(t) if t >= 0 else sol_m.sol(t)
            return math.exp(y[0]), math.exp(y[1]), math.exp(y[2])

        def trace_ray(th, ph_a):
            a1s, a2s, a3s = get_a(-T_half_b9)
            ux, uy, uz = math.sin(th)*math.cos(ph_a), math.sin(th)*math.sin(ph_a), math.cos(th)
            t, q, pv = -T_half_b9, np.array([1.0, 0.0, 0.0, 0.0]), np.array([a1s*ux, a2s*uy, a3s*uz])
            du = u_star_b9 / n_steps
            def rhs_u(tt, qq, pp):
                a1, a2, a3 = get_a(tt)
                qn = qq / np.linalg.norm(qq)
                v0, v1, v2 = pp[0]/(a1*a1), pp[1]/(a2*a2), pp[2]/(a3*a3)
                dt_du = math.sqrt(pp[0]*v0 + pp[1]*v1 + pp[2]*v2)
                dq = np.array([
                    -(qn[1]*v0 + qn[2]*v1 + qn[3]*v2), qn[0]*v0 + qn[2]*v2 - qn[3]*v1,
                    qn[0]*v1 + qn[3]*v0 - qn[1]*v2, qn[0]*v2 + qn[1]*v1 - qn[2]*v0,
                ])
                dp = 2.0 * np.array([pp[1]*v2 - pp[2]*v1, pp[2]*v0 - pp[0]*v2, pp[0]*v1 - pp[1]*v0])
                return dt_du, dq, dp
            for _ in range(n_steps):
                dt1, dq1, dp1 = rhs_u(t, q, pv)
                dt2, dq2, dp2 = rhs_u(t + 0.5*du*dt1, q + 0.5*du*dq1, pv + 0.5*du*dp1)
                dt3, dq3, dp3 = rhs_u(t + 0.5*du*dt2, q + 0.5*du*dq2, pv + 0.5*du*dp2)
                dt4, dq4, dp4 = rhs_u(t + du*dt3, q + du*dq3, pv + du*dp3)
                t += (du/6.0)*(dt1 + 2*dt2 + 2*dt3 + dt4)
                q = q + (du/6.0)*(dq1 + 2*dq2 + 2*dq3 + dq4)
                pv = pv + (du/6.0)*(dp1 + 2*dp2 + 2*dp3 + dp4)
            return t, q / np.linalg.norm(q)

        def q_tan(q, dq):
            qi = np.array([q[0], -q[1], -q[2], -q[3]])
            return np.array([
                qi[0]*dq[1] + qi[1]*dq[0] + qi[2]*dq[3] - qi[3]*dq[2],
                qi[0]*dq[2] - qi[1]*dq[3] + qi[2]*dq[0] + qi[3]*dq[1],
                qi[0]*dq[3] + qi[1]*dq[2] - qi[2]*dq[1] + qi[3]*dq[0],
            ])

        mu_p, wmu_p = np.polynomial.legendre.leggauss(6)
        ph_p = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
        da = 1e-5
        area = 0.0
        for mu, wm in zip(mu_p, wmu_p):
            th = math.acos(mu)
            for pha in ph_p:
                tc, qc = trace_ray(th, pha)
                tpt, qpt = trace_ray(th + da, pha); tmt, qmt = trace_ray(th - da, pha)
                tpp, qpp = trace_ray(th, pha + da); tmp, qmp = trace_ray(th, pha - da)
                dt_t, dt_p = (tpt - tmt)/(2*da), (tpp - tmp)/(2*da)
                v_t, v_p = q_tan(qc, (qpt - qmt)/(2*da)), q_tan(qc, (qpp - qmp)/(2*da))
                a1, a2, a3 = get_a(tc)
                gm = np.diag([a1*a1, a2*a2, a3*a3])
                q11, q22, q12 = -dt_t*dt_t + float(v_t @ gm @ v_t), -dt_p*dt_p + float(v_p @ gm @ v_p), -dt_t*dt_p + float(v_t @ gm @ v_p)
                area += (wm * (2.0 * math.pi / len(ph_p))) * math.sqrt(q11*q22 - q12*q12) / math.sin(th)
        return area, ham_res

    eps_b9 = 1e-3
    A0, ham0 = fixed_affine_area(0.0, 0.0)
    A0_exact = 4.0 * math.pi * (u_star_b9 ** 2)
    ds_by_ph, max_ham = [], ham0
    for ph in (0.0, 0.5 * math.pi):
        Ap, hp = fixed_affine_area(+eps_b9, ph)
        Am, hm = fixed_affine_area(-eps_b9, ph)
        max_ham = max(max_ham, hp, hm)
        dA_e2 = (0.5 * (Ap + Am) - A0) / (A0 * eps_b9 * eps_b9)
        ds_by_ph.append(8.0 * (u_star_b9 ** 2) * (1.0 / 12.0) * dA_e2)  # exact 2/(p(p^2-1)) = 1/12 per R_TT
    ds_nonlin = 0.5 * sum(ds_by_ph)

    # Independent dual-solver cross-check (Radau ODE + n_steps=200 ray-tracing at phase=0)
    A0_rad, _ = fixed_affine_area(0.0, 0.0, n_steps=100, method="Radau")
    Ap_rad, _ = fixed_affine_area(+eps_b9, 0.0, n_steps=100, method="Radau")
    Am_rad, _ = fixed_affine_area(-eps_b9, 0.0, n_steps=100, method="Radau")
    ds_ph0_rad = 8.0 * (u_star_b9 ** 2) * (1.0 / 12.0) * ((0.5 * (Ap_rad + Am_rad) - A0_rad) / (A0_rad * eps_b9 * eps_b9))
    Ap_200, _ = fixed_affine_area(+eps_b9, 0.0, n_steps=200, method="DOP853")
    Am_200, _ = fixed_affine_area(-eps_b9, 0.0, n_steps=200, method="DOP853")
    A0_200, _ = fixed_affine_area(0.0, 0.0, n_steps=200, method="DOP853")
    ds_ph0_200 = 8.0 * (u_star_b9 ** 2) * (1.0 / 12.0) * ((0.5 * (Ap_200 + Am_200) - A0_200) / (A0_200 * eps_b9 * eps_b9))
    dop_vs_radau_diff = abs(ds_by_ph[0] - ds_ph0_rad)
    step_doubling_diff = abs(ds_by_ph[0] - ds_ph0_200)
    assert max_ham < 1e-13 and dop_vs_radau_diff < 1e-6 and step_doubling_diff < 1e-6

    eta0_w = -chi_b9
    gD4_w, gD2_w = on_shell_dS_g_ds(mp.mpf(eta0_w), mp.mpf(chi_b9))
    ds_br_fp_p3_waist = float(-((mp.mpf(8)/3)*(mp.mpf(2)/3)*gD4_w + (mp.mpf(8)/3)*(mp.mpf(1)/48)*gD2_w))
    return {
        "unperturbed_area_err": abs(A0 - A0_exact),
        "max_hamiltonian_constraint_residual": max_ham,
        "dop853_vs_radau_diff_phase0": dop_vs_radau_diff,
        "ray_step_doubling_diff_phase0": step_doubling_diff,
        "ds_per_R_nonlin_fixed_affine_p3": ds_nonlin,
        "ds_per_R_nonlin_phase0": ds_by_ph[0],
        "ds_per_R_nonlin_phase90": ds_by_ph[1],
        "ds_per_R_BR_FP_p3_waist": ds_br_fp_p3_waist,
        "fixed_affine_over_BR_FP_ratio_p3": ds_nonlin / ds_br_fp_p3_waist,
    }


b9_fixed_affine = run_exact_bianchi9_fixed_affine_p3()
print(
    f"  Exact nonlinear vacuum Bianchi IX (p=3, waist@bounce, fixed affine u_*): "
    f"A0 err={b9_fixed_affine['unperturbed_area_err']:.2e}, "
    f"ds/R={b9_fixed_affine['ds_per_R_nonlin_fixed_affine_p3']:+.8f} vs BR_FP={b9_fixed_affine['ds_per_R_BR_FP_p3_waist']:+.8f} "
    f"(ratio={b9_fixed_affine['fixed_affine_over_BR_FP_ratio_p3']:.4%})"
)


# =============================================================================
# PART 2 (KT2): DUAL 50-DIGIT CHIRAL & ORTHOGONAL-RAY DIFFERENTIAL INVARIANTS
# =============================================================================
print("\n" + "=" * 86)
print("PART 2 (KT2): DUAL 50-DIGIT CHIRAL ASYMMETRY & ORTHOGONAL-RAY p_par - p_perp")
print("=" * 86)

chiral_ade = {}
for name, order, P in [("RP3", 2, 2), ("Q8", 8, 4), ("T*", 24, 12), ("O*", 48, 24), ("I*", 120, 60)]:
    R_chi = lambda pp, nm=name, od=order: od * (m_ade(nm, pp - 3) * (pp + 2) - m_ade(nm, pp + 1) * (pp - 2))
    C_r, A_r, B_r = extract_quasi_poly_deg2(R_chi, P, 3)
    du_eh_hz, du_conf, du_conf_frac, s_pm1, s_sup_raw, s_p0 = hurwitz_digamma_EH_general(C_r, A_r, B_r, P, 3)
    du_eh_ap = abel_plana_EH_general(C_r, A_r, B_r, P, 3)
    err_hz_ap = abs(du_eh_hz - du_eh_ap)
    assert err_hz_ap < mp.mpf("1e-44")
    du_bd_rad = (4 * du_conf - s_pm1) / 4
    du_bd_curv = s_pm1 / 16
    du_bd_bounce = du_bd_rad + du_bd_curv
    S_inv_c, S_sup_c = mp.mpf("0.5") * s_pm1, mp.mpf("0.5") * s_sup_raw
    pi2_int_h2_c = mp.pi * (S_inv_c + 2 * S_sup_c)
    pi2_deta_BR_FP_c = (mp.pi / 4) * du_bd_rad + (mp.pi / 6) * (2 * du_bd_curv)

    entry = {
        "order": order, "uv_sum_rules": [0, 0, 0],
        "pi2_du_conf_exact": str(du_conf_frac),
        "pi2_du_EH_50dps": mp.nstr(du_eh_hz, 30), "pi2_du_EH": float(du_eh_hz),
        "hz_vs_ap_err": float(err_hz_ap),
        "pi2_du_BD_bounce": float(du_bd_bounce),
        "pi2_deta_BR_FP_chiral": float(pi2_deta_BR_FP_c),
        "pi2_deta_tot_schur_chiral": float(pi2_deta_BR_FP_c + (mp.mpf(1) / 15) * pi2_int_h2_c),
        "pi2_deta_tot_wkb_chiral": float(pi2_deta_BR_FP_c + (mp.mpf(1) / 12) * pi2_int_h2_c),
        "eta_TT_0": float(s_p0 / order),
    }
    if name == "I*":
        # Also compute exact uncapped mode-by-mode (p=3..30) global chiral transit!
        corr_ch_wkb, corr_ch_sch = 0.0, 0.0
        for pv in range(3, 31):
            Rc_p = 120 * (m_ade("I*", pv - 3) * (pv + 2) - m_ade("I*", pv + 1) * (pv - 2))
            c_p = global_all_28[pv - 3]["c_inv"]
            w_p = (0.5 * math.pi) * Rc_p * (pv * pv + 1.0) / (pv * (pv * pv - 1.0))
            corr_ch_wkb += (c_p - 1.0 / 6.0) * w_p
            corr_ch_sch += (c_p - 0.15) * w_p
        entry["pi2_deta_tot_mode_exact_wkb_tail_chiral"] = entry["pi2_deta_tot_wkb_chiral"] + corr_ch_wkb
        entry["pi2_deta_tot_mode_exact_schur_tail_chiral"] = entry["pi2_deta_tot_schur_chiral"] + corr_ch_sch
    chiral_ade[name] = entry
    print(f"  Chiral {name:<4s}: conf={str(du_conf_frac):>4s}, EH={float(du_eh_hz):+12.6f}, |HZ-AP|={float(err_hz_ap):.2e}, BD={float(du_bd_bounce):+10.6f}, eta_TT(0)={float(s_p0/order):+.6f}")

print(
    f"  S^3/I* global chiral transit: const Schur={chiral_ade['I*']['pi2_deta_tot_schur_chiral']:+.6f}, "
    f"const WKB={chiral_ade['I*']['pi2_deta_tot_wkb_chiral']:+.6f}, "
    f"mode-exact (WKB/Schur tail)={chiral_ade['I*']['pi2_deta_tot_mode_exact_wkb_tail_chiral']:+.6f} / {chiral_ade['I*']['pi2_deta_tot_mode_exact_schur_tail_chiral']:+.6f}"
)

# 6-alpha invariance check on S^3/I* combined global chiral transit & Maxwell control
CrI_c, ArI_c, BrI_c = extract_quasi_poly_deg2(lambda pp: 120 * (m_ade("I*", pp - 3) * (pp + 2) - m_ade("I*", pp + 1) * (pp - 2)), 60, 3)
_, du_conf_Ic, _, s_pm1_Ic, s_sup_Ic, _ = hurwitz_digamma_EH_general(CrI_c, ArI_c, BrI_c, 60, 3)
Sp1_c, Sinv_c, Ssup_c = 2 * du_conf_Ic, mp.mpf("0.5") * s_pm1_Ic, mp.mpf("0.5") * s_sup_Ic
pi2_int_h2_Ic = mp.pi * (Sinv_c + 2 * Ssup_c)
pi2_deta_BR_FP_Ic = (mp.pi / 4) * mp.mpf("0.5") * (Sp1_c - Sinv_c) + (mp.pi / 6) * mp.mpf("0.25") * Sinv_c
ref_ch_schur = pi2_deta_BR_FP_Ic + (mp.mpf(1) / 15) * pi2_int_h2_Ic
max_chiral_alpha_err = mp.mpf(0)
for al in [mp.mpf("-1"), mp.mpf("0"), mp.mpf("0.25"), mp.mpf("0.5"), mp.mpf("1"), mp.mpf("1.5")]:
    da = al - mp.mpf("0.5")
    I_t1 = (mp.mpf(8) / 3) * (3 * mp.pi / 8) * mp.mpf("0.5") * (Sp1_c - Sinv_c)
    I_t2 = (-8) * (-mp.pi / 8) * mp.mpf("0.5") * Sinv_c
    I_t3a = -(mp.mpf(8) / 3) * ((3 * mp.pi / 8) * mp.mpf("0.5") * (Sp1_c - Sinv_c) + (mp.pi / 2) * mp.mpf("0.5") * Sinv_c)
    I_t3b = (mp.mpf(8) / 3) * ((3 * mp.pi / 8) * mp.mpf("0.5") * Sinv_c + (mp.pi / 2) * mp.mpf("0.5") * Ssup_c)
    br_al = pi2_deta_BR_FP_Ic + mp.mpf("0.5") * da * (I_t1 + I_t2 + I_t3a + I_t3b)
    lc_al = (-al / 6 + mp.mpf(3) / 20) * pi2_int_h2_Ic
    max_chiral_alpha_err = max(max_chiral_alpha_err, abs((br_al + lc_al) - ref_ch_schur))
assert max_chiral_alpha_err < mp.mpf("1e-47")

Cr_em, Ar_em, Br_em = extract_quasi_poly_deg2(lambda pp: 120 * (m_ade("I*", pp) * (pp - 1) - m_ade("I*", pp - 2) * (pp + 1)), 60, 2)
_, du_em_c, du_em_frac, _, _, sp0_em = hurwitz_digamma_EH_general(Cr_em, Ar_em, Br_em, 60, 2)
assert du_em_frac == 0 and abs(sp0_em / 120 - (-mp.mpf(361) / 180)) < mp.mpf("1e-45")


# Orthogonal-ray p_par - p_perp (tautological scalar-dP check REMOVED per P0-2!)
def lens_slope_W(q_ord: int, p: int) -> int:
    tot = 0
    for s in (+1, -1):
        two_K = p - 1 + 2 * s
        if two_K < 0: continue
        deg_R = p - 2 * s
        n_m, sum_4m2 = 0, 0
        for two_m in range(-two_K, two_K + 1, 2):
            if two_m % q_ord == 0:
                n_m += 1; sum_4m2 += two_m * two_m
        tot += deg_R * (16 * s * ((p + s) ** 2) * n_m - 8 * (p + 2 * s) * sum_4m2)
    return tot


def dm_slope_W(m_ord: int, p: int) -> int:
    tot = 0
    for s in (+1, -1):
        two_K = p - 1 + 2 * s
        if two_K < 0 or (two_K % 2 != 0): continue
        K = two_K // 2
        deg_R = p - 2 * s
        n_m = 1 if (K % 2 == 0) else 0
        sum_4m2 = 0
        for M_pos in range(m_ord, K + 1, m_ord):
            n_m += 1; sum_4m2 += (2 * M_pos) ** 2
        tot += deg_R * (16 * s * ((p + s) ** 2) * n_m - 8 * (p + 2 * s) * sum_4m2)
    return tot


def S3_slope_W(p: int) -> int:
    return -(16 * p * (p * p - 4) * (p * p - 1)) // 3


def extract_quasi_poly_deg5(R_fn, P: int, p_min: int = 3):
    coeffs = {}
    for r in range(P):
        p0 = p_min + ((r - p_min) % P)
        xs = [p0 + m * P for m in range(7)]
        ys = [R_fn(x) for x in xs]
        diffs = ys[:]
        for _ in range(6):
            diffs = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        assert diffs[0] == 0
        mat = [[Fraction(x ** k) for k in range(6)] + [Fraction(y)] for x, y in zip(xs[:6], ys[:6])]
        for col in range(6):
            piv = next(row for row in range(col, 6) if mat[row][col] != 0)
            mat[col], mat[piv] = mat[piv], mat[col]
            pv = mat[col][col]
            mat[col] = [v / pv for v in mat[col]]
            for row in range(6):
                if row != col and mat[row][col] != 0:
                    fac = mat[row][col]
                    mat[row] = [mat[row][j] - fac * mat[col][j] for j in range(7)]
        coeffs[r] = [mat[k][6] for k in range(6)]
    for k in range(6):
        assert sum(coeffs[r][k] for r in range(P)) == 0
    return coeffs


def hurwitz_slope_p_par(coeffs_deg5, P: int, p_min: int = 3):
    def eval_pow(pow_p):
        acc = mp.mpf(0)
        for k in range(6):
            ck = {r: mp.mpf(coeffs_deg5[r][k].numerator) / mp.mpf(coeffs_deg5[r][k].denominator) for r in range(P)}
            acc += sum_periodic_pow(ck, P, -(pow_p + k), p_min)
        return acc
    tot = mp.mpf(0)
    for m in range(55):
        dm = ((-1) ** m) * mp.binomial(mp.mpf("-0.5"), m) * eval_pow(-2 - 2 * m)
        tot += dm
        if abs(dm) < mp.mpf("1e-47"): break
    return -tot / 32


def abel_plana_slope_p_par(coeffs_deg5, P: int, p_min: int = 3) -> mp.mpf:
    def Gs(x):
        s, ac = mp.sqrt(x * x - 1), mp.acosh(x)
        return [mp.atan(s), ac, s, mp.mpf("0.5") * (x * s + ac), (x * x + 2) * s / 3, (x * (2 * x * x + 3) * s + 3 * ac) / 8]
    p0_list = []
    int_part, bdy_part = mp.mpf(0), mp.mpf(0)
    for r in range(P):
        c = coeffs_deg5[r]
        if all(v == 0 for v in c): continue
        p0 = mp.mpf(p_min + ((r - p_min) % P))
        cf = [mp.mpf(v.numerator) / mp.mpf(v.denominator) for v in c]
        p0_list.append((p0, cf))
        gv = Gs(p0)
        int_part += -sum(cf[k] * gv[k] for k in range(6)) / P
        bdy_part += mp.mpf("0.5") * sum(cf[k] * (p0 ** k) for k in range(6)) / (p0 * mp.sqrt(p0 * p0 - 1))
    P_m = mp.mpf(P)
    def integrand(y):
        tot_im = mp.mpf(0)
        for p0, cf in p0_list:
            z = p0 + 1j * P_m * y
            poly = cf[0] + z * (cf[1] + z * (cf[2] + z * (cf[3] + z * (cf[4] + z * cf[5]))))
            tot_im += mp.im(poly / (z * mp.sqrt(z * z - 1)))
        return -2 * tot_im / mp.expm1(2 * mp.pi * y)
    return -(int_part + bdy_part + mp.quad(integrand, [0, 1, 4, 12, mp.inf])) / 32


def maxwell_lens_exact(q_ord: int):
    P = q_ord if (q_ord % 2 == 0) else 2 * q_ord
    C_u, A_u, B_u = extract_quasi_poly_deg2(
        lambda pp: q_ord * (m_lens(q_ord, pp - 2) * (pp + 1) + m_lens(q_ord, pp) * (pp - 1)) - 2 * (pp * pp - 1), P, 2
    )
    def em_sl(p):
        tq = sum(p - tm * tm for tm in range(-p, p + 1, 2) if tm % q_ord == 0) + sum(
            -p - tm * tm for tm in range(-(p - 2), p - 1, 2) if tm % q_ord == 0
        )
        return q_ord * tq - (-(2 * p * (p * p - 1)) // 3)
    c_sl = extract_quasi_poly_deg5(em_sl, P, 2)
    def bz(k, x):
        if k == 0: return -(x - Fraction(1, 2))
        if k == 1: return -(x**2 - x + Fraction(1, 6)) / 2
        if k == 2: return -(x**3 - Fraction(3, 2)*x**2 + Fraction(1, 2)*x) / 3
        if k == 3: return -(x**4 - 2*x**3 + x**2 - Fraction(1, 30)) / 4
        if k == 4: return -(x**5 - Fraction(5, 2)*x**4 + Fraction(5, 3)*x**3 - Fraction(1, 6)*x) / 5
        return -(x**6 - 3*x**5 + Fraction(5, 2)*x**4 - Fraction(1, 2)*x**2 + Fraction(1, 42)) / 6
    u_f, pp_f = Fraction(0), Fraction(0)
    for r in range(P):
        p0 = 2 + ((r - 2) % P)
        x = Fraction(p0, P)
        u_f += C_u[r] * (P**3) * bz(3, x) + A_u[r] * (P**2) * bz(2, x) + B_u[r] * P * bz(1, x)
        for k in range(6):
            pp_f += c_sl[r][k] * (P**k) * bz(k, x)
    pi2_u, pi2_ppar = u_f / 4, -pp_f / 4
    pi2_pperp = (pi2_u - pi2_ppar) / 2
    return pi2_u, pi2_ppar, pi2_pperp, pi2_ppar - pi2_pperp


ortho_results = {}
for label, kind, param in [("L(2,1)", "L", 2), ("Q8", "D", 2), ("L(3,1)", "L", 3), ("L(11,1)", "L", 11), ("L(14,1)", "L", 14), ("L(20,1)", "L", 20)]:
    if kind == "L":
        q_ord = param
        P = q_ord if (q_ord % 2 == 0) else 2 * q_ord
        C_u, A_u, B_u = extract_quasi_poly_deg2(
            lambda pp, q=q_ord: q * (m_lens(q, pp + 1) * (pp - 2) + m_lens(q, pp - 3) * (pp + 2)) - 2 * (pp * pp - 4), P, 3
        )
        c_sl = extract_quasi_poly_deg5(lambda pp, q=q_ord: q * lens_slope_W(q, pp) - S3_slope_W(pp), P, 3)
        u_em, ppar_em, pperp_em, diff_em = maxwell_lens_exact(q_ord)
    else:
        m_ord = param
        order = 4 * m_ord
        P = math.lcm(4, 2 * m_ord)
        C_u, A_u, B_u = extract_quasi_poly_deg2(lambda pp: order * ade_tt_mult("Q8", pp) - 2 * (pp * pp - 4), P, 3)
        c_sl = extract_quasi_poly_deg5(lambda pp: order * dm_slope_W(m_ord, pp) - S3_slope_W(pp), P, 3)
        u_em, ppar_em, pperp_em, diff_em = Fraction(27, 16), Fraction(9, 16), Fraction(9, 16), Fraction(0)

    u_hz, _, _, _, _, _ = hurwitz_digamma_EH_general(C_u, A_u, B_u, P, 3)
    ppar_hz = hurwitz_slope_p_par(c_sl, P, 3)
    ppar_ap = abel_plana_slope_p_par(c_sl, P, 3)
    err_sl = abs(ppar_hz - ppar_ap)
    assert err_sl < mp.mpf("1e-45")
    pperp_hz = (u_hz - ppar_hz) / 2
    diff_grav = ppar_hz - pperp_hz
    ortho_results[label] = {
        "pi2_u_grav": float(u_hz), "pi2_p_par_grav": float(ppar_hz), "pi2_p_perp_grav": float(pperp_hz),
        "pi2_T_par_minus_T_perp_grav": float(diff_grav),
        "pi2_T_par_minus_T_perp_grav_50dps": mp.nstr(diff_grav, 30),
        "slope_hz_vs_ap_err": float(err_sl),
        "pi2_T_par_minus_T_perp_EM_exact": str(diff_em),
        "grav_over_EM_ratio": (float(diff_grav) / float(diff_em)) if diff_em != 0 else None,
    }
    print(f"  Ortho  {label:<7s}: p_par-p_perp={float(diff_grav):+12.6f} (|HZ-AP|={float(err_sl):.2e}) | EM={str(diff_em):>10s}")


# =============================================================================
# PART 3 (KT3): TACHYON-FREE [0.60, 1.40] BERGER CURVES, RADIAL RHS & 2-DECADE C^2
# =============================================================================
print("\n" + "=" * 86)
print("PART 3 (KT3): TACHYON-FREE [0.60, 1.40] BERGER CURVES ON L(20,1) & L(24,1) + 2-DEC C^2")
print("=" * 86)

t3 = time.time()

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
        for a in range(5): Sm[a, b] = np.sum(T_basis[a] * comm)
    S_gens.append(Sm)

ev_S3, U_S3 = np.linalg.eigh(S_gens[2]); U_S3 = U_S3[:, np.argsort(ev_S3)]
ev_V3, U_V3 = np.linalg.eigh(1j * L_gens[2]); U_V3 = U_V3[:, np.argsort(ev_V3)]
C_raw = [np.zeros((3, 5), dtype=complex) for _ in range(3)]
for k in range(3):
    for a in range(3):
        for al in range(5): C_raw[k][a, al] = T_basis[al][k, a]
C_sph = [U_V3.conj().T @ C_raw[k] @ U_S3 for k in range(3)]
C_p, C_m, C_0 = C_sph[0] - 1j * C_sph[1], C_sph[0] + 1j * C_sph[1], C_sph[2]


def berger_precompute(lam: float):
    c = np.zeros((3, 3, 3))
    c[2, 0, 1], c[2, 1, 0] = 2.0 * lam, -2.0 * lam
    c[0, 1, 2], c[0, 2, 1] = 2.0 / lam, -2.0 / lam
    c[1, 2, 0], c[1, 0, 2] = 2.0 / lam, -2.0 / lam
    Gamma = np.array([[[0.5 * (c[a, k, b] - c[k, b, a] + c[b, a, k]) for k in range(3)] for b in range(3)] for a in range(3)])
    Riem = np.array([[[[sum(Gamma[a, e, k]*Gamma[e, b, l] - Gamma[a, e, l]*Gamma[e, b, k] - c[e, k, l]*Gamma[a, b, e] for e in range(3)) for l in range(3)] for k in range(3)] for b in range(3)] for a in range(3)])
    M_raw = [np.array([[np.sum(T_basis[a] * (Gamma[:, :, k] @ T_basis[b] - T_basis[b] @ Gamma[:, :, k])) for b in range(5)] for a in range(5)], dtype=complex) for k in range(3)]
    K_riem = np.array([[np.sum(T_basis[a] * (-2.0 * np.einsum("ikjl,kl->ij", Riem, T_basis[b]))) for b in range(5)] for a in range(5)], dtype=complex)
    M_sph = [U_S3.conj().T @ Mk @ U_S3 for Mk in M_raw]
    return {
        "inv_lam2_m1": 4.0 * (1.0 / (lam * lam) - 1.0),
        "Op_const": -(M_sph[0] @ M_sph[0] + M_sph[1] @ M_sph[1] + M_sph[2] @ M_sph[2]) + U_S3.conj().T @ K_riem @ U_S3,
        "Op_Jp": 2j * (M_sph[0] - 1j * M_sph[1]), "Op_Jm": 2j * (M_sph[0] + 1j * M_sph[1]), "Op_J3": (4j / lam) * M_sph[2],
        "Div_const": C_sph[0] @ M_sph[0] + C_sph[1] @ M_sph[1] + C_sph[2] @ M_sph[2],
        "Div_Jp": -1j * C_p, "Div_Jm": -1j * C_m, "Div_J3": (-2j / lam) * C_0,
    }


def block_tt_eigs(two_j: int, two_M: int, pre: dict):
    j = 0.5 * two_j
    if (two_M - two_j) % 2 != 0: return np.empty(0, dtype=float)
    s_idx, s_mJ = [], []
    for i in range(5):
        two_mJ = two_M - 2 * (i - 2)
        if abs(two_mJ) <= two_j: s_idx.append(i); s_mJ.append(0.5 * two_mJ)
    ns = len(s_idx)
    if ns == 0: return np.empty(0, dtype=float)
    v_idx, v_mJ = [], []
    for i in range(3):
        two_mJ = two_M - 2 * (i - 1)
        if abs(two_mJ) <= two_j: v_idx.append(i); v_mJ.append(0.5 * two_mJ)
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
                if ir == ic: val += c_j2 + c_lam * (mc * mc)
            elif abs(mr - (mc + 1.0)) < 1e-9: val += math.sqrt(j * (j + 1.0) - mc * (mc + 1.0)) * Op_Jp[ir, ic]
            elif abs(mr - (mc - 1.0)) < 1e-9: val += math.sqrt(j * (j + 1.0) - mc * (mc - 1.0)) * Op_Jm[ir, ic]
            Op_b[r, col] = val
    if nv == 0 or (two_j == 0 and abs(two_M) == 2): return np.array([Op_b[0, 0].real], dtype=float)
    Div_b = np.zeros((nv, ns), dtype=complex)
    Div_const, Div_Jp, Div_Jm, Div_J3 = pre["Div_const"], pre["Div_Jp"], pre["Div_Jm"], pre["Div_J3"]
    for r in range(nv):
        vr, mr = v_idx[r], v_mJ[r]
        for col in range(ns):
            ic, mc = s_idx[col], s_mJ[col]
            val = 0.0j
            if abs(mr - mc) < 1e-9: val += Div_const[vr, ic] + mc * Div_J3[vr, ic]
            elif abs(mr - (mc + 1.0)) < 1e-9: val += math.sqrt(j * (j + 1.0) - mc * (mc + 1.0)) * Div_Jp[vr, ic]
            elif abs(mr - (mc - 1.0)) < 1e-9: val += math.sqrt(j * (j + 1.0) - mc * (mc - 1.0)) * Div_Jm[vr, ic]
            Div_b[r, col] = val
    _, sv, Vh = np.linalg.svd(Div_b, full_matrices=True)
    if two_j == 1 and abs(two_M) == 1: Q = Vh[-1:].conj().T
    else:
        tol = max(1e-9, 1e-10 * (sv[0] if len(sv) > 0 else 1.0))
        null_mask = np.ones(ns, dtype=bool); null_mask[:len(sv)] = (sv < tol)
        if not np.any(null_mask): return np.empty(0, dtype=float)
        Q = Vh[null_mask].conj().T
    H_tt = Q.conj().T @ Op_b @ Q
    return np.linalg.eigvalsh(0.5 * (H_tt + H_tt.conj().T)).real


def maxwell_block_eigs(two_j: int, two_M: int, lam: float):
    if (two_M - two_j) % 2 != 0 or two_M > two_j + 2: return np.empty(0, dtype=float)
    j = 0.5 * two_j
    if two_M == two_j + 2: return np.array([2.0 * (j + 1.0) / lam], dtype=float)
    if two_M == two_j: return np.array([2.0 * lam + 2.0 * j / lam], dtype=float)
    rad = math.sqrt(lam * lam + 4.0 * j * (j + 1.0) + (1.0 / (lam * lam) - 1.0) * (two_M ** 2))
    return np.array([abs(lam - rad), abs(lam + rad)], dtype=float)


def eval_all_fields_on_even_lens(lam: float, q_list: list, ts_all: tuple, n_extrap: int = 4):
    t_arr = np.array(ts_all, dtype=float)
    t_min = float(np.min(t_arr))
    lam_13, lam_23 = lam ** (1.0 / 3.0), lam ** (2.0 / 3.0)
    scale_min = lam_13 * min(1.0, 1.0 / lam)
    max_2j = int(math.ceil(math.sqrt(42.0 / t_min) / scale_min)) + 16
    n_t, nq = len(ts_all), len(q_list)
    keys = ["grav_per", "scal_per", "scal_anti", "weyl_per", "weyl_anti", "maxwell_per"]
    diff = {k: [np.zeros(n_t, dtype=float) for _ in range(nq)] for k in keys}
    comp = {k: [np.zeros(n_t, dtype=float) for _ in range(nq)] for k in keys}
    s3_nonpos = {k: 0 for k in keys}
    q_nonpos = {k: [0] * nq for k in keys}
    pre = berger_precompute(lam)
    inv_lam2_m1 = 1.0 / (lam * lam) - 1.0
    scal_curv = (4.0 - lam * lam) / 3.0
    half_lam = 0.5 * lam

    def kahan_add(key, qi, val):
        y = val - comp[key][qi]
        t_new = diff[key][qi] + y
        comp[key][qi] = (t_new - diff[key][qi]) - y
        diff[key][qi] = t_new

    for two_j in range(max_2j + 1):
        j = 0.5 * two_j
        deg = float(two_j + 1)
        start_2M_bos = 0 if (two_j % 2 == 0) else 1
        start_2M_fer = 1 if (two_j % 2 == 0) else 0

        sh_s3 = np.zeros(n_t); sh_q = [np.zeros(n_t) for _ in range(nq)]
        for two_M in range(start_2M_bos, two_j + 5, 2):
            evs = block_tt_eigs(two_j, two_M, pre)
            if len(evs) == 0: continue
            mult_M = 1.0 if two_M == 0 else 2.0
            bad = evs <= 0.0
            if np.any(bad):
                s3_nonpos["grav_per"] += int(np.sum(bad))
                for qi, q in enumerate(q_list):
                    if two_M % q == 0: q_nonpos["grav_per"][qi] += int(np.sum(bad))
            pos = evs[~bad]
            if len(pos) == 0: continue
            w2 = lam_23 * pos; wb = np.sqrt(w2)
            contrib = (deg * mult_M) * 0.5 * np.sum(wb[:, None] * np.exp(-w2[:, None] * t_arr[None, :]), axis=0)
            sh_s3 += contrib
            for qi, q in enumerate(q_list):
                if two_M % q == 0: sh_q[qi] += contrib
        for qi, q in enumerate(q_list): kahan_add("grav_per", qi, 0.5 * (q * sh_q[qi] - sh_s3))

        sh_s3 = np.zeros(n_t); sh_qp = [np.zeros(n_t) for _ in range(nq)]; sh_qa = [np.zeros(n_t) for _ in range(nq)]
        base_s = 4.0 * j * (j + 1.0) + scal_curv
        for two_M in range(start_2M_bos, two_j + 1, 2):
            M = 0.5 * two_M
            ev = base_s + 4.0 * inv_lam2_m1 * (M * M)
            mult_M = 1.0 if two_M == 0 else 2.0
            if ev <= 0.0:
                s3_nonpos["scal_per"] += 1; s3_nonpos["scal_anti"] += 1
                for qi, q in enumerate(q_list):
                    if two_M % q == 0: q_nonpos["scal_per"][qi] += 1
                    if two_M % q == q // 2: q_nonpos["scal_anti"][qi] += 1
                continue
            w2 = lam_23 * ev; wb = math.sqrt(w2)
            contrib = (deg * mult_M) * 0.5 * wb * np.exp(-w2 * t_arr)
            sh_s3 += contrib
            for qi, q in enumerate(q_list):
                rem = two_M % q
                if rem == 0: sh_qp[qi] += contrib
                elif rem == q // 2: sh_qa[qi] += contrib
        for qi, q in enumerate(q_list):
            kahan_add("scal_per", qi, 0.5 * (q * sh_qp[qi] - sh_s3))
            kahan_add("scal_anti", qi, 0.5 * (q * sh_qa[qi] - sh_s3))

        sh_s3 = np.zeros(n_t); sh_qp = [np.zeros(n_t) for _ in range(nq)]; sh_qa = [np.zeros(n_t) for _ in range(nq)]
        dj1_sq = float((two_j + 1) ** 2)
        for two_M in range(start_2M_fer, two_j + 2, 2):
            mult_M = 1.0 if two_M == 0 else 2.0
            if two_M == two_j + 1: lams = np.array([abs(-(two_j + 1.0) / lam - half_lam)], dtype=float)
            else:
                rad = math.sqrt(dj1_sq + inv_lam2_m1 * (two_M * two_M))
                lams = np.array([abs(-half_lam + rad), abs(-half_lam - rad)], dtype=float)
            if np.any(lams <= 1e-12): s3_nonpos["weyl_per"] += 1
            wb = lam_13 * lams; w2 = wb * wb
            contrib = (deg * mult_M) * (-0.5) * np.sum(wb[:, None] * np.exp(-w2[:, None] * t_arr[None, :]), axis=0)
            sh_s3 += contrib
            for qi, q in enumerate(q_list):
                rem = two_M % q
                if rem == 0: sh_qp[qi] += contrib
                elif rem == q // 2: sh_qa[qi] += contrib
        for qi, q in enumerate(q_list):
            kahan_add("weyl_per", qi, 0.5 * (q * sh_qp[qi] - sh_s3))
            kahan_add("weyl_anti", qi, 0.5 * (q * sh_qa[qi] - sh_s3))

        sh_s3 = np.zeros(n_t); sh_q = [np.zeros(n_t) for _ in range(nq)]
        for two_M in range(start_2M_bos, two_j + 3, 2):
            ws = maxwell_block_eigs(two_j, two_M, lam)
            if len(ws) == 0: continue
            mult_M = 1.0 if two_M == 0 else 2.0
            wb = lam_13 * ws; w2 = wb * wb
            contrib = (deg * mult_M) * 0.5 * np.sum(wb[:, None] * np.exp(-w2[:, None] * t_arr[None, :]), axis=0)
            sh_s3 += contrib
            for qi, q in enumerate(q_list):
                if two_M % q == 0: sh_q[qi] += contrib
        for qi, q in enumerate(q_list): kahan_add("maxwell_per", qi, 0.5 * (q * sh_q[qi] - sh_s3))

    t_ex = t_arr[:n_extrap]
    out = {}
    for k in keys:
        out[k] = {}
        for qi, q in enumerate(q_list):
            u_ts = diff[k][qi][:n_extrap]
            u0 = float(np.polyfit(t_ex, u_ts, n_extrap - 1)[-1])
            u0_sub = float(np.polyfit(t_ex[:-1], u_ts[:-1], n_extrap - 2)[-1])
            out[k][q] = {"pi2_u": u0, "extrap_err": abs(u0 - u0_sub), "q_nonpos": q_nonpos[k][qi], "s3_nonpos": s3_nonpos[k]}
    return out, max_2j


ts_xfine = (0.00006, 0.00009, 0.00013, 0.00019, 0.00027, 0.00038)
ts_fine = (0.00010, 0.00014, 0.00020, 0.00028, 0.00040, 0.00056)
res_lam1_xfine, _ = eval_all_fields_on_even_lens(1.0, [2, 20, 24], ts_xfine, n_extrap=4)
res_lam105_xfine, _ = eval_all_fields_on_even_lens(1.05, [20, 24], ts_xfine, n_extrap=4)

max_lam1_vs_exact_rel, max_lam1_rp3_abs = 0.0, 0.0
for k in ["grav_per", "scal_per", "scal_anti", "weyl_per", "weyl_anti", "maxwell_per"]:
    for q in [2, 20, 24]:
        exact_v = DERIVE["kt3_derive"]["exact_lam1_benchmarks"][str(q)][k]["float"]
        hk_v = res_lam1_xfine[k][q]["pi2_u"]
        if q == 2: max_lam1_rp3_abs = max(max_lam1_rp3_abs, abs(hk_v - exact_v))
        else: max_lam1_vs_exact_rel = max(max_lam1_vs_exact_rel, abs(hk_v - exact_v) / max(1.0, abs(exact_v)))
assert max_lam1_rp3_abs < 1e-6 and max_lam1_vs_exact_rel < 5e-6
print(f"  lam=1 HK (ts_xfine) vs exact rationals/50-dps oracle: RP^3 max abs = {max_lam1_rp3_abs:.2e}, L(20)/L(24) max rel = {max_lam1_vs_exact_rel:.2e}")

# Strictly 100% tachyon-free grid on [0.60, 1.40] (ZERO dropped S^3 or L(q,1) modes anywhere! Fixes P1-4!)
lam_grid_tf = [0.60, 0.72, 0.85, 0.95, 0.98, 1.00, 1.02, 1.05, 1.15, 1.28, 1.40]
q_targets = [20, 24]
curves = {q: {k: [] for k in ["grav_per", "scal_per", "scal_anti", "weyl_per", "weyl_anti", "maxwell_per"]} for q in q_targets}
nonpos_log = {}
grid_vs_xfine_lam1_rel, grid_vs_xfine_lam105_rel = 0.0, 0.0

for lam in lam_grid_tf:
    r_lam, _ = eval_all_fields_on_even_lens(lam, q_targets, ts_fine, n_extrap=4)
    nonpos_log[f"{lam:.2f}"] = {k: {"q20": r_lam[k][20]["q_nonpos"], "q24": r_lam[k][24]["q_nonpos"], "s3": r_lam[k][20]["s3_nonpos"]} for k in r_lam}
    for k in r_lam:
        assert r_lam[k][20]["s3_nonpos"] == 0 and r_lam[k][20]["q_nonpos"] == 0 and r_lam[k][24]["q_nonpos"] == 0
        for q in q_targets:
            curves[q][k].append(r_lam[k][q]["pi2_u"])
            if abs(lam - 1.00) < 1e-9:
                ref_v = res_lam1_xfine[k][q]["pi2_u"]
                grid_vs_xfine_lam1_rel = max(grid_vs_xfine_lam1_rel, abs(r_lam[k][q]["pi2_u"] - ref_v) / max(1.0, abs(ref_v)))
            if abs(lam - 1.05) < 1e-9:
                ref_v = res_lam105_xfine[k][q]["pi2_u"]
                grid_vs_xfine_lam105_rel = max(grid_vs_xfine_lam105_rel, abs(r_lam[k][q]["pi2_u"] - ref_v) / max(1.0, abs(ref_v)))

assert grid_vs_xfine_lam1_rel < 1e-5 and grid_vs_xfine_lam105_rel < 1e-5
print(f"  11-pt tachyon-free Berger curves [0.60, 1.40] (0 dropped modes!) in {time.time() - t3:.1f}s | ladder: lam=1.00={grid_vs_xfine_lam1_rel:.2e}, lam=1.05={grid_vs_xfine_lam105_rel:.2e}")

lam_arr = np.array(lam_grid_tf)
R3_fixed_vol = lambda l: 2.0 * (4.0 - l * l) * (l ** (2.0 / 3.0))
dR3_dlam = lambda l: (16.0 / 3.0) * (l ** (-1.0 / 3.0)) * (1.0 - l * l)
d2R3_dlam2 = lambda l: -(16.0 / 9.0) * (l ** (-4.0 / 3.0)) * (1.0 + 5.0 * l * l)

cocktails = [
    ("Pure Graviton (baseline)", {"grav_per": 1}, 2),
    ("1g + 2 anti-per scalars", {"grav_per": 1, "scal_anti": 2}, 4),
    ("1g + 3 anti-per scalars", {"grav_per": 1, "scal_anti": 3}, 5),
    ("1g + 4 anti-per scalars", {"grav_per": 1, "scal_anti": 4}, 6),
    ("1g + 1 anti-per Weyl (NS)", {"grav_per": 1, "weyl_anti": 1}, 4),
    ("1g + 2 anti-per Weyl (NS)", {"grav_per": 1, "weyl_anti": 2}, 6),
    ("1g + 1 per Weyl (Ramond 2:2)", {"grav_per": 1, "weyl_per": 1}, 4),
    ("1g + 2 per Weyl (Ramond)", {"grav_per": 1, "weyl_per": 2}, 6),
    ("7:8 (1g+5per_s+8anti_s)", {"grav_per": 1, "scal_per": 5, "scal_anti": 8}, 15),
    ("7:8 (1g+2EM+1per_s+8anti_s)", {"grav_per": 1, "maxwell_per": 2, "scal_per": 1, "scal_anti": 8}, 15),
    ("7:8 (7per_s+8anti_s pure scal)", {"scal_per": 7, "scal_anti": 8}, 15),
]


def analyze_cocktail(name: str, q: int, weights: dict, n_sp: int, pi2_u_loc_pp_1: float = 0.0):
    pi2_u = sum(w * np.array(curves[q][k]) for k, w in weights.items())
    if pi2_u_loc_pp_1 != 0.0:
        pi2_u = pi2_u + 0.5 * pi2_u_loc_pp_1 * ((lam_arr - 1.0) ** 2)
    u_arr = pi2_u / (math.pi ** 2)
    spl = CubicSpline(lam_arr, u_arr)  # strictly on tachyon-free [0.60, 1.40]!
    u1, up1, upp1 = float(spl(1.0)) * (math.pi**2), float(spl(1.0, 1)) * (math.pi**2), float(spl(1.0, 2)) * (math.pi**2)

    ls = 1.05
    u_ls, up_ls, upp_ls = float(spl(ls)), float(spl(ls, 1)), float(spl(ls, 2))
    r3_ls, r3p_ls, r3pp_ls = R3_fixed_vol(ls), dR3_dlam(ls), d2R3_dlam2(ls)
    eq_105 = None
    if up_ls < 0.0:
        abar2 = 2.0 * up_ls / r3p_ls
        abar = math.sqrt(abar2)
        v_pp = -0.5 * abar2 * r3pp_ls + upp_ls
        L_min = 2.0 * math.pi * abar * (ls ** (2.0 / 3.0)) / q
        # Explicit Lambda = 0 Friedmann RHS = -R^(3)/(2 a^2) + u / a^4 (P1-3 radial-constraint audit!)
        fried_rhs_lam0 = -r3_ls / (2.0 * abar2) + u_ls / (abar2 * abar2)
        eq_105 = {
            "lam_star": ls, "a_bar_lPl": abar, "a_bar_lP": abar * SQRT_8PI,
            "stable_Vpp_gt_0": bool(v_pp > 0.0), "Vpp_a4": float(v_pp),
            "L_min_over_lsp": L_min / math.sqrt(n_sp),
            "R3_over_MPl2": abs(r3_ls) / abar2,
            "friedmann_rhs_lambda0": fried_rhs_lam0,
        }
    return {
        "cocktail": name, "q": q, "n_sp": n_sp, "weights": weights,
        "pi2_u_1": u1, "pi2_up_1": up1, "pi2_upp_1": upp1, "eq_at_lam105": eq_105,
    }


cocktail_analysis = {20: {}, 24: {}}
for q in [20, 24]:
    for cname, wts, nsp in cocktails:
        cocktail_analysis[q][cname] = analyze_cocktail(cname, q, wts, nsp)

oq4_scaling = {}
for cname in ["Pure Graviton (baseline)", "1g + 3 anti-per scalars", "1g + 1 per Weyl (Ramond 2:2)", "7:8 (1g+5per_s+8anti_s)", "7:8 (1g+2EM+1per_s+8anti_s)"]:
    r20, r24 = cocktail_analysis[20][cname], cocktail_analysis[24][cname]
    oq4_scaling[cname] = {
        "u24_over_u20": r24["pi2_u_1"] / r20["pi2_u_1"],
        "up24_over_up20": r24["pi2_up_1"] / r20["pi2_up_1"],
        "Lmin_over_lsp_105_q20": r20["eq_at_lam105"]["L_min_over_lsp"] if r20["eq_at_lam105"] else None,
        "Lmin_over_lsp_105_q24": r24["eq_at_lam105"]["L_min_over_lsp"] if r24["eq_at_lam105"] else None,
    }

c2_scheme_check = {}
for cname, wts, nsp in [
    ("1g + 3 anti-per scalars", {"grav_per": 1, "scal_anti": 3}, 5),
    ("1g + 4 anti-per scalars", {"grav_per": 1, "scal_anti": 4}, 6),
    ("1g + 2 per Weyl (Ramond)", {"grav_per": 1, "weyl_per": 2}, 6),
    ("7:8 (1g+5per_s+8anti_s)", {"grav_per": 1, "scal_per": 5, "scal_anti": 8}, 15),
]:
    a4_pp_mat = (4.0/45.0)*(wts.get("scal_per", 0) + wts.get("scal_anti", 0)) + (4.0/15.0)*(wts.get("weyl_per", 0) + wts.get("weyl_anti", 0)) + (16.0/15.0)*wts.get("maxwell_per", 0)
    a4_pp_tot = 554.0 / 15.0 + a4_pp_mat
    c2_scheme_check[cname] = {"a4_pp_tot": a4_pp_tot, "by_q": {}}
    for q in [20, 24]:
        band = {}
        for mu_tag, ln_mu in [("mu_a_0p1", -math.log(10.0)), ("mu_a_1p0", 0.0), ("mu_a_10p0", math.log(10.0))]:
            upp_loc = 67.6352 - a4_pp_tot * ln_mu
            eq = analyze_cocktail(cname, q, wts, nsp, pi2_u_loc_pp_1=upp_loc)["eq_at_lam105"]
            band[mu_tag] = {
                "pi2_upp_loc": upp_loc, "stable": eq["stable_Vpp_gt_0"] if eq else False,
                "a_bar_lPl": eq["a_bar_lPl"] if eq else None,
                "L_min_over_lsp": eq["L_min_over_lsp"] if eq else None,
                "R3_over_MPl2": eq["R3_over_MPl2"] if eq else None,
            }
        c2_scheme_check[cname]["by_q"][str(q)] = band
    b24 = c2_scheme_check[cname]["by_q"]["24"]
    print(f"  2-dec C^2 L(24) {cname:<27s}: L_min/l_sp(mu=0.1,1,10) = ({b24['mu_a_0p1']['L_min_over_lsp']:.4f}, {b24['mu_a_1p0']['L_min_over_lsp']:.4f}, {b24['mu_a_10p0']['L_min_over_lsp']:.4f})")


# =============================================================================
# PART 4 (KT4): DUAL 50-DIGIT TWISTED DIRAC/WEYL, ANOMALY-FREE HOSOTANI & ALL BENCHMARKS
# =============================================================================
print("\n" + "=" * 86)
print("PART 4 (KT4): DUAL 50-DIGIT TWISTED DIRAC/WEYL & WITTEN-ANOMALY-FREE HOSOTANI INVERSION")
print("=" * 86)

phi = (1 + mp.sqrt(5)) / 2
istar_raw = [
    ("1",   1,  mp.mpf(0),      mp.mpf(2),  mp.mpf(2),      mp.mpf(2)),
    ("-1",  1,  mp.pi,          mp.mpf(2),  mp.mpf(-2),     mp.mpf(-2)),
    ("4",  30,  mp.pi / 2,      mp.mpf(2),  mp.mpf(0),      mp.mpf(0)),
    ("3",  20,  2 * mp.pi / 3,  mp.mpf(2),  mp.mpf(-1),     mp.mpf(-1)),
    ("6",  20,  mp.pi / 3,      mp.mpf(2),  mp.mpf(1),      mp.mpf(1)),
    ("5a", 12,  2 * mp.pi / 5,  mp.mpf(2),  phi - 1,        -phi),
    ("5b", 12,  4 * mp.pi / 5,  mp.mpf(2),  -phi,           phi - 1),
    ("10a",12,  mp.pi / 5,      mp.mpf(2),  phi,            1 - phi),
    ("10b",12,  3 * mp.pi / 5,  mp.mpf(2),  1 - phi,        phi),
]
classes = [{"name": n, "mult": m, "theta": th, "fund": [f0, f1, f2], "adj": [mp.mpf(3), f1*f1 - 1, f2*f2 - 1]} for n, m, th, f0, f1, f2 in istar_raw]


def method1_dirac_cauchy(rep_type: str, rho_idx: int):
    def F_Fp(t):
        fv, fpv = mp.mpf(0), mp.mpf(0)
        ct, st = mp.cosh(t), mp.sinh(t)
        for c in classes:
            chi = c[rep_type][rho_idx]
            if chi == 0: continue
            d = 2 * (ct - mp.cos(c["theta"]))
            fv += c["mult"] * chi / d
            fpv += -c["mult"] * chi * (2 * st) / (d * d)
        return fv / 120, fpv / 120

    def K_pm(t):
        f, fp = F_Fp(t)
        return mp.exp(-t / 2) * (-fp + f), mp.exp(t / 2) * (-fp - f)

    kp_p, _ = K_pm(mp.mpf("0.37"))
    _, km_m = K_pm(mp.mpf("-0.37"))
    pal_err = abs(kp_p + km_m)
    assert pal_err < mp.mpf("1e-46")

    R, N = mp.mpf("0.25"), 128
    c0, c1, d0, d1 = mp.mpf(0), mp.mpf(0), mp.mpf(0), mp.mpf(0)
    for k in range(N):
        z = R * mp.exp(2j * mp.pi * k / N)
        kp, km = K_pm(z)
        ks, kd = kp + km, kp - km
        c0 += ks; c1 += ks / z; d0 += kd; d1 += kd / z
    return {"E_Weyl": mp.re(c1 / N) / 2, "E_Dirac": mp.re(c1 / N), "eta_D_0": mp.re(d0 / N), "Delta_E_chiral": mp.re(d1 / N) / 2, "palindromic_err": float(pal_err)}


def method2_dirac_hurwitz(rep_type: str, rho_idx: int):
    def m_rho(k: int) -> int:
        val = sum(c["mult"] * c[rep_type][rho_idx] * su2_chi(k, c["theta"]) for c in classes) / 120
        iv = int(mp.nint(val))
        assert abs(val - iv) < mp.mpf("1e-35")
        return iv
    P = 60
    def eval_fn(fn):
        zm1, z0 = mp.mpf(0), mp.mpf(0)
        for r in range(P):
            ns = [r + m * P for m in range(4)]
            ys = [Fraction(fn(nv)) for nv in ns]
            xs = [Fraction(2 * nv + 3, 2) for nv in ns]
            d2 = (ys[2] - ys[1]) - (ys[1] - ys[0])
            c = d2 / (2 * P * P)
            L1, L2 = ys[0] - c * xs[0] * xs[0], ys[1] - c * xs[1] * xs[1]
            a = (L2 - L1) / P
            b = L1 - a * xs[0]
            q = mp.mpf(2 * r + 3) / (2 * P)
            Cm, Am, Bm = mp.mpf(c.numerator) / c.denominator, mp.mpf(a.numerator) / a.denominator, mp.mpf(b.numerator) / b.denominator
            zm1 += Cm * (P**3) * mp.zeta(-3, q) + Am * (P**2) * mp.zeta(-2, q) + Bm * P * mp.zeta(-1, q)
            z0 += Cm * (P**2) * mp.zeta(-2, q) + Am * P * mp.zeta(-1, q) + Bm * mp.zeta(0, q)
        return zm1, z0
    zt_m1, zt_0 = eval_fn(lambda n: m_rho(n) * (n + 2) + m_rho(n + 1) * (n + 1))
    zd_m1, zd_0 = eval_fn(lambda n: m_rho(n) * (n + 2) - m_rho(n + 1) * (n + 1))
    return {"E_Weyl": -zt_m1 / 2, "E_Dirac": -zt_m1, "eta_D_0": zd_0, "Delta_E_chiral": -zd_m1 / 2}


E_WEYL_S3, E_DIRAC_S3 = mp.mpf(17) / 960, mp.mpf(17) / 480
# Dynamic 50-digit graviton benchmarks computed directly from hurwitz_digamma_EH_general (P2-3 remediation)
PI2_C_GRAV_ESU = u_eh_I
PI2_C_GRAV_BD = pi2_C_rad_I + pi2_C_curv_bd_I
kt4_solve = {}
max_m1_m2_err, max_m1_exact_err, max_pal_err = 0.0, 0.0, 0.0

for rep_type, dim in [("fund", 2), ("adj", 3)]:
    kt4_solve[rep_type] = {}
    for idx, rname in enumerate(["rho_0", "rho_1", "rho_2"]):
        m1 = method1_dirac_cauchy(rep_type, idx)
        m2 = method2_dirac_hurwitz(rep_type, idx)
        ex_f = Fraction(DERIVE["kt4_derive"]["exact_rational_twisted_dirac"][rep_type][rname]["E_Dirac_exact"])
        ex_mp = mp.mpf(ex_f.numerator) / mp.mpf(ex_f.denominator)
        e_m1m2 = float(max(abs(m1["E_Dirac"] - m2["E_Dirac"]), abs(m1["eta_D_0"] - m2["eta_D_0"]), abs(m1["Delta_E_chiral"])))
        e_m1ex = float(abs(m1["E_Dirac"] - ex_mp))
        assert e_m1m2 < 1e-45 and e_m1ex < 1e-45
        max_m1_m2_err = max(max_m1_m2_err, e_m1m2)
        max_m1_exact_err = max(max_m1_exact_err, e_m1ex)
        max_pal_err = max(max_pal_err, m1["palindromic_err"])

        pi2_dC_W = float((120 * m1["E_Weyl"] - dim * E_WEYL_S3) / 2)
        pi2_dC_D = float((120 * m1["E_Dirac"] - dim * E_DIRAC_S3) / 2)
        kt4_solve[rep_type][rname] = {
            "dim": dim, "E_Dirac_exact": str(ex_f),
            "E_Dirac": float(m1["E_Dirac"]), "E_Weyl": float(m1["E_Weyl"]), "eta_D_0": float(m1["eta_D_0"]),
            "m1_vs_m2_err": e_m1m2, "m1_vs_exact_err": e_m1ex, "palindromic_err": m1["palindromic_err"],
            "pi2_dC_Weyl": pi2_dC_W, "pi2_dC_Dirac": pi2_dC_D,
            "net_1grav_1Dirac_ESU": float(PI2_C_GRAV_ESU) + pi2_dC_D,
            "net_1grav_1Dirac_BD": float(PI2_C_GRAV_BD) + pi2_dC_D,
            "net_1grav_1YM_1Dirac_ESU": float(PI2_C_GRAV_ESU + pi2_C_YM_I[rname]) + pi2_dC_D,
            "net_1grav_1YM_1Dirac_BD": float(PI2_C_GRAV_BD + pi2_C_YM_I[rname]) + pi2_dC_D,
        }

# Witten-anomaly-free combined Hosotani-inversion benchmarks (pi_4(SU(2)) = Z_2)
ym_plus_1adjWeyl_SYM = {r: float(pi2_C_YM_I[r]) + kt4_solve["adj"][r]["pi2_dC_Weyl"] for r in ("rho_0", "rho_1", "rho_2")}
ym_plus_1adjDirac = {r: float(pi2_C_YM_I[r]) + kt4_solve["adj"][r]["pi2_dC_Dirac"] for r in ("rho_0", "rho_1", "rho_2")}
ym_plus_1fundDirac = {r: float(pi2_C_YM_I[r]) + kt4_solve["fund"][r]["pi2_dC_Dirac"] for r in ("rho_0", "rho_1", "rho_2")}
assert ym_plus_1adjWeyl_SYM["rho_2"] < 0.0 < ym_plus_1adjWeyl_SYM["rho_1"] < ym_plus_1adjWeyl_SYM["rho_0"]
assert ym_plus_1adjDirac["rho_2"] < 0.0 < ym_plus_1adjDirac["rho_1"] < ym_plus_1adjDirac["rho_0"]
assert ym_plus_1fundDirac["rho_2"] < 0.0 < ym_plus_1fundDirac["rho_1"] < ym_plus_1fundDirac["rho_0"]

# Evaluate N_f^crit for BOTH (a) non-dynamical flavor holonomy (1g + N_f Dirac) and
# (b) dynamical SU(2) gauge holonomy (1g + 1 SU(2) YM + N_f Dirac, including pi^2 dC_YM(rho_2) = +31.4625) (P1-3 remediation!)
pi2_deta_fund_rho2 = float((mp.pi / 4) * mp.mpf(kt4_solve["fund"]["rho_2"]["pi2_dC_Dirac"]))
pi2_deta_adj_rho2 = float((mp.pi / 4) * mp.mpf(kt4_solve["adj"]["rho_2"]["pi2_dC_Dirac"]))
pi2_C_YM_rho2 = float(pi2_C_YM_I["rho_2"])
pi2_deta_YM_rho2 = float((mp.pi / 4) * pi2_C_YM_I["rho_2"])

nf_thresholds = {
    "pi2_C_grav_ESU_dynamic": float(PI2_C_GRAV_ESU),
    "pi2_C_grav_BD_dynamic": float(PI2_C_GRAV_BD),
    "pi2_C_YM_rho2": pi2_C_YM_rho2,
    "pi2_deta_YM_rho2": pi2_deta_YM_rho2,
    # (a) Non-dynamical background SU(2) flavor holonomy (1 graviton + N_f Dirac)
    "flavor_fund_Dirac_ESU_x": float(PI2_C_GRAV_ESU) / (-kt4_solve["fund"]["rho_2"]["pi2_dC_Dirac"]),
    "flavor_fund_Dirac_BD_x": float(PI2_C_GRAV_BD) / (-kt4_solve["fund"]["rho_2"]["pi2_dC_Dirac"]),
    "flavor_fund_Dirac_global_wkb_x": glob_wkb / (-pi2_deta_fund_rho2),
    "flavor_fund_Dirac_global_schur_x": glob_sch / (-pi2_deta_fund_rho2),
    "flavor_adj_Dirac_ESU_x": float(PI2_C_GRAV_ESU) / (-kt4_solve["adj"]["rho_2"]["pi2_dC_Dirac"]),
    "flavor_adj_Dirac_BD_x": float(PI2_C_GRAV_BD) / (-kt4_solve["adj"]["rho_2"]["pi2_dC_Dirac"]),
    "flavor_adj_Dirac_global_wkb_x": glob_wkb / (-pi2_deta_adj_rho2),
    "flavor_adj_Dirac_global_schur_x": glob_sch / (-pi2_deta_adj_rho2),
    "N_f_crit_flavor_fund_ESU_and_BD": 2,
    "N_f_crit_flavor_fund_global_wkb": math.ceil(glob_wkb / (-pi2_deta_fund_rho2)),
    "N_f_crit_flavor_fund_global_schur": math.ceil(glob_sch / (-pi2_deta_fund_rho2)),
    "N_f_crit_flavor_adj_all_observables": math.ceil(glob_wkb / (-pi2_deta_adj_rho2)),
    # (b) Dynamical SU(2) gauge holonomy (1 graviton + 1 SU(2) YM + N_f Dirac)
    "dynamical_fund_Dirac_ESU_x": float(PI2_C_GRAV_ESU + pi2_C_YM_I["rho_2"]) / (-kt4_solve["fund"]["rho_2"]["pi2_dC_Dirac"]),
    "dynamical_fund_Dirac_BD_x": float(PI2_C_GRAV_BD + pi2_C_YM_I["rho_2"]) / (-kt4_solve["fund"]["rho_2"]["pi2_dC_Dirac"]),
    "dynamical_fund_Dirac_global_wkb_x": (glob_wkb + pi2_deta_YM_rho2) / (-pi2_deta_fund_rho2),
    "dynamical_fund_Dirac_global_schur_x": (glob_sch + pi2_deta_YM_rho2) / (-pi2_deta_fund_rho2),
    "dynamical_adj_Dirac_ESU_x": float(PI2_C_GRAV_ESU + pi2_C_YM_I["rho_2"]) / (-kt4_solve["adj"]["rho_2"]["pi2_dC_Dirac"]),
    "dynamical_adj_Dirac_BD_x": float(PI2_C_GRAV_BD + pi2_C_YM_I["rho_2"]) / (-kt4_solve["adj"]["rho_2"]["pi2_dC_Dirac"]),
    "dynamical_adj_Dirac_global_wkb_x": (glob_wkb + pi2_deta_YM_rho2) / (-pi2_deta_adj_rho2),
    "dynamical_adj_Dirac_global_schur_x": (glob_sch + pi2_deta_YM_rho2) / (-pi2_deta_adj_rho2),
    "N_f_crit_dynamical_fund_all_observables": math.ceil((glob_wkb + pi2_deta_YM_rho2) / (-pi2_deta_fund_rho2)),
    "N_f_crit_dynamical_adj_ESU_and_BD": math.ceil(float(PI2_C_GRAV_ESU + pi2_C_YM_I["rho_2"]) / (-kt4_solve["adj"]["rho_2"]["pi2_dC_Dirac"])),
    "N_f_crit_dynamical_adj_global_wkb": math.ceil((glob_wkb + pi2_deta_YM_rho2) / (-pi2_deta_adj_rho2)),
    "N_f_crit_dynamical_adj_global_schur": math.ceil((glob_sch + pi2_deta_YM_rho2) / (-pi2_deta_adj_rho2)),
    "1YM_plus_1adjWeyl_SYM_pi2_dC": ym_plus_1adjWeyl_SYM,
    "1YM_plus_1adjDirac_pi2_dC": ym_plus_1adjDirac,
    "1YM_plus_1fundDirac_pi2_dC": ym_plus_1fundDirac,
}
print(f"  Witten-anomaly-free N=1 SYM (1 YM + 1 adj Weyl): rho_2={ym_plus_1adjWeyl_SYM['rho_2']:+.4f} < rho_1={ym_plus_1adjWeyl_SYM['rho_1']:+.4f} < rho_0={ym_plus_1adjWeyl_SYM['rho_0']:+.4f} (inverts to rho_2)")
print(f"  Witten-anomaly-free (1 YM + 1 adj Dirac):        rho_2={ym_plus_1adjDirac['rho_2']:+.4f} < rho_1={ym_plus_1adjDirac['rho_1']:+.4f} < rho_0={ym_plus_1adjDirac['rho_0']:+.4f} (inverts to rho_2)")
print(f"  Witten-anomaly-free (1 YM + 1 fund Dirac):       rho_2={ym_plus_1fundDirac['rho_2']:+.4f} < rho_1={ym_plus_1fundDirac['rho_1']:+.4f} < rho_0={ym_plus_1fundDirac['rho_0']:+.4f} (inverts to rho_2)")
print(
    f"  Flavor-twist x_f (1g+N_f D):        fund ESU={nf_thresholds['flavor_fund_Dirac_ESU_x']:.4f}, BD={nf_thresholds['flavor_fund_Dirac_BD_x']:.4f}, "
    f"global={nf_thresholds['flavor_fund_Dirac_global_wkb_x']:.4f}/{nf_thresholds['flavor_fund_Dirac_global_schur_x']:.4f}; "
    f"adj ESU={nf_thresholds['flavor_adj_Dirac_ESU_x']:.4f}, BD={nf_thresholds['flavor_adj_Dirac_BD_x']:.4f}, "
    f"global={nf_thresholds['flavor_adj_Dirac_global_wkb_x']:.4f}/{nf_thresholds['flavor_adj_Dirac_global_schur_x']:.4f}"
)
print(
    f"  Dynamical-YM x_f (1g+1YM+N_f D):    fund ESU={nf_thresholds['dynamical_fund_Dirac_ESU_x']:.4f}, BD={nf_thresholds['dynamical_fund_Dirac_BD_x']:.4f}, "
    f"global={nf_thresholds['dynamical_fund_Dirac_global_wkb_x']:.4f}/{nf_thresholds['dynamical_fund_Dirac_global_schur_x']:.4f}; "
    f"adj ESU={nf_thresholds['dynamical_adj_Dirac_ESU_x']:.4f}, BD={nf_thresholds['dynamical_adj_Dirac_BD_x']:.4f}, "
    f"global={nf_thresholds['dynamical_adj_Dirac_global_wkb_x']:.4f}/{nf_thresholds['dynamical_adj_Dirac_global_schur_x']:.4f}"
)

out = {
    "runtime_sec": time.time() - T0_ALL,
    "kt1_solve": {
        "convergence_checks": {
            "p3_global_rk4_400_vs_800_diff": p3_glob_rk4_err,
            "p11_centered_rk4_96_vs_192_diff": p11_ctr_rk4_err,
            "p11_centered_3alpha_spread": p11_ctr_alpha_err,
            "p15_global_2alpha_spread": p15_glob_alpha_err,
            "quadrature_refinement_checks": quad_conv_checks,
            "max_quadrature_refinement_diff": max_quad_diff,
        },
        "global_all_28_modes_p3_to_30": global_all_28,
        "short_diamond_spatial_arrival_sample": short_spatial_arrival,
        "global_summary": global_summary,
        "diamond_coarse_benchmarks": diamond_coarse_benchmarks,
        "exact_bianchi9_fixed_affine_p3": b9_fixed_affine,
    },
    "kt2_solve": {
        "chiral_ade": chiral_ade,
        "Istar_chiral_global_6alpha_err": float(max_chiral_alpha_err),
        "maxwell_Istar_chiral_control": {"pi2_du_EM_exact": str(du_em_frac), "eta_EM_0": float(sp0_em / 120)},
        "orthogonal_ray": ortho_results,
    },
    "kt3_solve": {
        "lam1_xfine_vs_exact_rp3_max_abs": max_lam1_rp3_abs,
        "lam1_xfine_vs_exact_L20_L24_max_rel": max_lam1_vs_exact_rel,
        "xfine_vs_fine_lam1_max_rel": grid_vs_xfine_lam1_rel,
        "xfine_vs_fine_lam105_max_rel": grid_vs_xfine_lam105_rel,
        "lam_grid_tachyon_free": lam_grid_tf,
        "curves_by_q": {str(q): curves[q] for q in q_targets},
        "nonpos_log": nonpos_log,
        "cocktail_analysis": {str(q): cocktail_analysis[q] for q in q_targets},
        "oq4_cancellation_scaling": oq4_scaling,
        "two_decade_C2_scheme_check_lam105": c2_scheme_check,
    },
    "kt4_solve": {
        "max_m1_vs_m2_err_50dps": max_m1_m2_err,
        "max_m1_vs_exact_Q_err_50dps": max_m1_exact_err,
        "max_palindromic_err_50dps": max_pal_err,
        "twisted_fermions": kt4_solve,
        "hosotani_and_flip_summary": nf_thresholds,
    },
}

out_path = OUT_DIR / "02_solve.json"
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nSaved -> {out_path} (total runtime {time.time() - T0_ALL:.2f}s)")
