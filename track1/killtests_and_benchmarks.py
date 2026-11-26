#!/usr/bin/env python3
"""
Stream p04_israel_darmois — Goal 3:
Independent Kill-Tests, BGG (1987) & FMM (1990) Benchmarks, Non-Time-Symmetric
Velocities, Super-Hubble Horizon Crossing, and Publication Diagnostic Figure.
"""

import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from verify_junction_and_eom import (
    flrw_a_and_derivs,
    flrw_rho_p_scaled,
    integrate_shell_cosmic_time,
    integrate_bounce_shell,
)

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHI_INJ = math.pi / 10.0


# =============================================================================
# 1. BLAU-GUENDELMAN-GUTH (1987) EXACT BENCHMARK: S^3 vs S^3/I*
# =============================================================================

def bgg_benchmark():
    """
    Reproduce exact Blau-Guendelman-Guth (PRD 35, 1747, 1987) critical mass mu_cr(kappa),
    barrier peak z_*(alpha), and integrate BOTH BGG daughter-universe branches on S^3
    (where chi in (0, pi) is embedded) vs S^3/I* (where chi >= pi/10 self-intersects).
    """
    kappas = [0.1, 0.2, 0.5, 0.8, 1.2]
    bgg_table = []
    for kappa in kappas:
        alpha = (1.0 - kappa ** 2) / (1.0 + kappa ** 2)
        w_star = 0.5 * (-alpha + math.sqrt(alpha ** 2 + 8.0))
        z_star = w_star ** (1.0 / 3.0)
        f_min = z_star ** (-4) - 2.0 * alpha / z_star + z_star ** 2
        # E_BGG = 4*kappa^2 / (mu^(2/3) * (1+kappa^2)^(4/3)) = f_min at critical mass mu_cr
        mu_cr = ((4.0 * kappa ** 2) / (f_min * ((1.0 + kappa ** 2) ** (4.0 / 3.0)))) ** 1.5
        R_star = ((mu_cr / (1.0 + kappa ** 2)) ** (1.0 / 3.0)) * z_star
        R_plus_cr = (mu_cr / (1.0 + kappa ** 2)) ** (1.0 / 3.0)
        R_minus_cr = (mu_cr / (1.0 - kappa ** 2)) ** (1.0 / 3.0) if kappa < 1.0 else None
        beta_m_at_star = (mu_cr - (1.0 - kappa ** 2) * (R_star ** 3)) / (2.0 * kappa * (R_star ** 2))
        beta_p_at_star = (mu_cr - (1.0 + kappa ** 2) * (R_star ** 3)) / (2.0 * kappa * (R_star ** 2))
        bgg_table.append({
            "kappa": kappa, "alpha": alpha, "z_star": z_star, "f_min": f_min,
            "mu_cr": mu_cr, "R_star": R_star, "R_plus_cr": R_plus_cr,
            "R_minus_cr": R_minus_cr,
            "beta_m_at_star": beta_m_at_star,
            "beta_p_at_star": beta_p_at_star,
            "beta_m_positive_at_star": beta_m_at_star > 0,
            "beta_p_negative_at_star": beta_p_at_star < 0,
        })

    # Integrate an explicit supercritical BGG monotone baby-universe trajectory on S^3:
    # kappa = 0.5 => mu_cr = 0.92505; choose mu = 1.5 * mu_cr = 1.38757 > mu_cr.
    # On S^3, keeping [0, chi] (s_keep = +1) starts at small R -> 0 (chi -> 0),
    # crosses equator chi = pi/2 (where beta_- flips + -> -), and expands into the northern hemisphere.
    kappa_bgg = 0.5
    alpha = (1.0 - kappa_bgg ** 2) / (1.0 + kappa_bgg ** 2)
    w_star = 0.5 * (-alpha + math.sqrt(alpha ** 2 + 8.0))
    z_star = w_star ** (1.0 / 3.0)
    f_min = z_star ** (-4) - 2.0 * alpha / z_star + z_star ** 2
    mu_cr_05 = ((4.0 * kappa_bgg ** 2) / (f_min * ((1.0 + kappa_bgg ** 2) ** (4.0 / 3.0)))) ** 1.5
    mu_sup = 1.5 * mu_cr_05

    # Place initial point at t = -2.2 (contracting leg of S^3 de Sitter) so the supercritical
    # BGG bubble expands smoothly across the S^3 equator chi = pi/2 = 90 deg into the northern hemisphere!
    t_start_s3 = -2.2
    a_st, a_t_st, _ = flrw_a_and_derivs(t_start_s3, 0.0)
    R_start = 0.05
    beta_m_start = (mu_sup - (1.0 - kappa_bgg ** 2) * (R_start ** 3)) / (2.0 * kappa_bgg * (R_start ** 2))
    beta_p_start = (mu_sup - (1.0 + kappa_bgg ** 2) * (R_start ** 3)) / (2.0 * kappa_bgg * (R_start ** 2))
    Rd_start = math.sqrt(beta_p_start ** 2 - 1.0 + mu_sup / R_start)
    chi_start = math.asin(R_start / a_st)
    denom_st = 1.0 - R_start ** 2
    td_start = (beta_m_start * math.cos(chi_start) - Rd_start * a_t_st * math.sin(chi_start)) / denom_st
    achid_start = (Rd_start * math.cos(chi_start) - beta_m_start * a_t_st * math.sin(chi_start)) / denom_st
    psi_start = math.asinh(achid_start)
    assert abs(math.cosh(psi_start) - td_start) / td_start < 1e-12, f"{math.cosh(psi_start)} vs {td_start}"

    def rhs_s3(t, state):
        chi, psi, tau = state
        psi_c = max(-20.0, min(20.0, psi))
        a, a_t, _ = flrw_a_and_derivs(t, 0.0)
        H = a_t / a
        ch, sh, th = math.cosh(psi_c), math.sinh(psi_c), math.tanh(psi_c)
        R = a * math.sin(chi)
        Ktt_m_exact = -mu_sup / (kappa_bgg * (R ** 3)) - (1.0 - kappa_bgg ** 2) / (2.0 * kappa_bgg)
        psidot = Ktt_m_exact - H * sh
        return [th / a, psidot / ch, 1.0 / ch]

    def ev_chi_pi(t, state):
        return (math.pi - 1e-3) - state[0]
    ev_chi_pi.terminal = True

    sol_s3 = solve_ivp(rhs_s3, [t_start_s3, 5.0], [chi_start, psi_start, 0.0],
                       method="DOP853", rtol=1e-12, atol=1e-14, events=[ev_chi_pi], max_step=0.02)
    t_arr = sol_s3.t
    chi_arr = sol_s3.y[0, :]
    psi_arr = sol_s3.y[1, :]
    R_arr = np.cosh(t_arr) * np.sin(chi_arr)
    beta_m_arr = np.cosh(psi_arr) * np.cos(chi_arr) + np.sinh(psi_arr) * np.sinh(t_arr) * np.sin(chi_arr)
    beta_p_arr = beta_m_arr - kappa_bgg * R_arr
    beta_m_alg = (mu_sup - (1.0 - kappa_bgg ** 2) * (R_arr ** 3)) / (2.0 * kappa_bgg * (R_arr ** 2))
    max_s3_res = float(np.max(np.abs(beta_m_arr - beta_m_alg) / np.maximum(1.0, np.abs(beta_m_alg))))

    idx_inj = int(np.argmax(chi_arr >= CHI_INJ))
    crosses_eq = bool(np.any(chi_arr >= 0.5 * math.pi))
    idx_eq = int(np.argmax(chi_arr >= 0.5 * math.pi)) if crosses_eq else None

    return {
        "bgg_critical_table": bgg_table,
        "s3_supercritical_demo": {
            "kappa": kappa_bgg, "mu_cr": mu_cr_05, "mu_used": mu_sup,
            "t_start": t_start_s3,
            "chi_start_deg": math.degrees(chi_start),
            "chi_max_deg": float(np.degrees(np.max(chi_arr))),
            "chi_end_deg": float(np.degrees(chi_arr[-1])),
            "R_start": R_start, "R_end": float(R_arr[-1]),
            "crosses_horizon_R_eq_rs": bool(R_arr[0] < mu_sup < R_arr[-1]),
            "crosses_equator_pi_over_2": crosses_eq,
            "t_at_chi_pi_over_10": float(t_arr[idx_inj]),
            "R_at_chi_pi_over_10": float(R_arr[idx_inj]),
            "beta_m_at_chi_pi_over_10": float(beta_m_arr[idx_inj]),
            "t_at_equator_pi_over_2": float(t_arr[idx_eq]) if idx_eq is not None else None,
            "R_at_equator_pi_over_2": float(R_arr[idx_eq]) if idx_eq is not None else None,
            "beta_m_start": float(beta_m_arr[0]),
            "beta_m_end": float(beta_m_arr[-1]),
            "beta_p_start": float(beta_p_arr[0]),
            "beta_p_end": float(beta_p_arr[-1]),
            "beta_m_flips_sign_on_S3": bool(beta_m_arr[0] > 0 and beta_m_arr[-1] < 0),
            "beta_p_flips_sign_into_QIII": bool(beta_p_arr[0] > 0 and beta_p_arr[-1] < 0),
            "max_constraint_rel_res": max_s3_res,
        },
        "s3_curve": {
            "t": t_arr.tolist(), "chi": chi_arr.tolist(), "R": R_arr.tolist(),
            "beta_m": beta_m_arr.tolist(), "beta_p": beta_p_arr.tolist(),
        },
    }


# =============================================================================
# 2. FROLOV-MARKOV-MUKHANOV (1989/1990) SPACELIKE JUNCTION BENCHMARK
# =============================================================================

def fmm_spacelike_benchmark():
    """
    Verify:
      (a) FMM (PRD 41, 383, 1990) spacelike transition at r = r_0 inside Schwarzschild
          interior (a_dS < r_0 < r_s): matches Kantowski-Sachs Schwarzschild interior to
          Kantowski-Sachs de Sitter interior (R x S^2, anisotropic 3-Ricci, pi_1 != I*).
      (b) Embedding a spherically symmetric constant-t=0 punctured annulus [chi_1, chi_2] x S^2
          (0 < chi_1 < chi_2 < pi/10; chi=0 maps to the r=0 Schwarzschild curvature singularity!)
          of S^3/I* inside Schwarzschild interior (r < r_s): proves exact (dT/dchi)^2 > 0,
          computes exact angular extrinsic curvature jump [K^theta_theta] on the annulus,
          and verifies topological obstruction (pi_1 = 0 != I*).
    """
    # (a) FMM Kantowski-Sachs junction at r = r_0 with a_dS = 1, r_s = 10, r_0 = 2.0
    mu = 10.0
    r0 = 2.0
    ric_ks = np.array([0.0, 1.0 / (r0 ** 2), 1.0 / (r0 ** 2)])
    aniso_ks = float(np.max(np.abs(ric_ks - np.mean(ric_ks))))
    f_s = mu / r0 - 1.0
    f_d = r0 ** 2 - 1.0
    Kth_schw = -math.sqrt(f_s) / r0
    KT_schw = 0.5 * (mu / (r0 ** 2)) / math.sqrt(f_s)
    Kth_dS = +math.sqrt(f_d) / r0
    KT_dS = +r0 / math.sqrt(f_d)
    # Spacelike Israel junction (n.n = -1): 8*pi*G S^i_j = +([K^i_j] - [K] delta^i_j)
    dKth = Kth_dS - Kth_schw
    dKT = KT_dS - KT_schw
    S_T_8piG = -2.0 * dKth
    S_th_8piG = -(dKT + dKth)

    # (b) Spherically symmetric constant-t=0 punctured annulus [0.01, pi/10) of S^3/I* (a0 = 1)
    # embedded inside Schwarzschild interior (r < r_s = mu_cap = 2.0):
    # Note: chi = 0 maps to r = 0 (Schwarzschild curvature singularity, Kretschmann -> inf)!
    mu_cap = 2.0
    a0 = 1.0
    chis = np.linspace(0.01, CHI_INJ - 1e-4, 50)
    R_cap = a0 * np.sin(chis)
    f_in = mu_cap / R_cap - 1.0
    dT_dchi_sq = (a0 ** 2) * (mu_cap / R_cap - np.sin(chis) ** 2) / (f_in ** 2)
    assert np.all(dT_dchi_sq > 0)
    # Extrinsic curvature K^theta_theta on the constant-t=0 S^3/I* slice is identically 0 (time-symmetric),
    # whereas on the Schwarzschild interior embedding (future timelike unit normal n^r < 0, n^T):
    # u_chi = (dT/dchi, dr/dchi) with g(u,u) = a0^2 => unit tangent e_chi = u_chi / a0,
    # future timelike unit normal n^r = -sqrt(f_in) * (sqrt(f_in) dT/dchi / a0) = -f_in (dT/dchi)/a0
    # => K^theta_theta,Schw = n^r / R = -sqrt(mu_cap / R - sin^2(chi)) / R < 0!
    Kth_cap_schw = -np.sqrt(mu_cap / R_cap - np.sin(chis) ** 2) / R_cap
    dKth_cap = 0.0 - Kth_cap_schw  # [K^theta_theta] = K_FLRW - K_Schw > 0

    return {
        "fmm_kantowski_sachs": {
            "mu": mu, "r0": r0,
            "intrinsic_3Ricci_eigenvalues_KS": ric_ks.tolist(),
            "intrinsic_3Ricci_anisotropy_norm": aniso_ks,
            "Kth_schw": Kth_schw, "KT_schw": KT_schw,
            "Kth_dS": Kth_dS, "KT_dS": KT_dS,
            "S_T_8piG": S_T_8piG, "S_theta_8piG": S_th_8piG,
            "isometric_to_S3_Istar": False,
            "topology": "R x S^2 (or S^1 x S^2), pi_1 != I*",
        },
        "s3_istar_spacelike_annulus_in_schw_interior": {
            "mu": mu_cap, "a0": a0,
            "chi_range_smooth": [0.01, CHI_INJ - 1e-4],
            "chi_zero_hits_r0_singularity": True,
            "min_dT_dchi_sq": float(np.min(dT_dchi_sq)),
            "min_jump_Kth": float(np.min(dKth_cap)),
            "max_jump_Kth": float(np.max(dKth_cap)),
            "embedded_topology": "punctured spherical annulus [chi_1, chi_2] x S^2 (pi_1 = 0, NOT S^3/I*)",
            "max_volume_fraction_of_S3_Istar": (60.0 / math.pi) * (2.0 * CHI_INJ - math.sin(2.0 * CHI_INJ)),
        },
    }


# =============================================================================
# 3. KILL-TESTS: NON-ZERO INITIAL RAPIDITY, SUPER-HUBBLE HORIZON, PURE dS BOUND
# =============================================================================

def killtest_non_symmetric_and_superhubble():
    """
    Kill-test 3a: Non-time-symmetric bounce shells (psi(0) != 0) integrated BOTH forward (+t) and backward (-t).
    Kill-test 3b: Pure de Sitter analytical upper bound t_breach <= arcosh(sec(pi/10)) = 0.319458.
    Kill-test 3c: Super-Hubble late-birth white-hole horizon emergence (R(t0) = r_s > a_dS,
                  Rdot > 0, beta_+ < 0 => Quadrant IV -> Quadrant III, U=0, V<0):
                  tests if it stays < pi/10 forward AND what happens backward inside Quadrant IV!
    Kill-test 3d: Page 7 kept-volume crossover sliver chi_b in (0.27212, pi/10).
    """
    # 3a: Non-time-symmetric initial rapidities at t=0
    nonsym = []
    for gamma in (0.0, 0.15, 0.247765741792):
        y_plus = math.sqrt(0.5 + math.sqrt(0.25 - gamma))
        chi_b = 0.15
        R0 = y_plus * math.sin(chi_b)
        mu = 0.3 * y_plus * (math.sin(chi_b) ** 3)
        for psi0 in (-1.5, -0.5, 0.5, 1.5):
            ch, sh = math.cosh(psi0), math.sinh(psi0)
            Rd0 = sh * math.cos(chi_b)
            beta_m0 = -ch * math.cos(chi_b)
            beta_p0 = -math.sqrt(Rd0 ** 2 + 1.0 - mu / R0)
            kappa0 = (beta_m0 - beta_p0) / R0
            if kappa0 <= 0:
                continue
            fwd = integrate_shell_cosmic_time(gamma, mu, 0.0, chi_b, psi0, kappa0, -1.0, s_keep=-1.0, t_end=8.0)
            bwd = integrate_shell_cosmic_time(gamma, mu, 0.0, chi_b, psi0, kappa0, -1.0, s_keep=-1.0, t_end=-8.0)
            # In backward integration (t decreasing from 0 to -8), check if chi crossed pi/10 or kappa -> 0
            chi_f_max = float(np.max(fwd["sol"].y[0, :]))
            chi_b_max = float(np.max(bwd["sol"].y[0, :]))
            survived_both = (chi_f_max < CHI_INJ and chi_b_max < CHI_INJ
                             and fwd["t_final"] > 7.9 and bwd["t_final"] < -7.9)
            nonsym.append({
                "gamma": gamma, "psi0": psi0, "kappa0": kappa0,
                "fwd_chi_max": chi_f_max, "fwd_t_final": fwd["t_final"], "fwd_breach": chi_f_max >= CHI_INJ,
                "bwd_chi_max": chi_b_max, "bwd_t_final": bwd["t_final"], "bwd_breach": chi_b_max >= CHI_INJ,
                "survived_eternally": survived_both,
                "max_res": max(fwd["max_constraint_rel_res"], bwd["max_constraint_rel_res"]),
            })

    # 3b: Pure de Sitter exact analytical upper bound arcosh(sec(pi/10)) — ZERO silent drops!
    t_bound_ds = math.acosh(1.0 / math.cos(CHI_INJ))
    ds_cases = []
    ds_dropped = 0
    for chi_b in (0.01, 0.05, 0.15, 0.25, 0.30):
        mu_max_loc = math.sin(chi_b) ** 3
        for mu_frac in (1e-5, 0.1, 0.5, 0.9):
            for w in (-1.0, 0.0, 1.0):
                res = integrate_bounce_shell(0.0, mu_frac * mu_max_loc, chi_b, w, s_keep=-1.0, t_end=4.0)
                if not res["feasible"] or res["t_breach"] is None:
                    ds_dropped += 1
                else:
                    ds_cases.append((chi_b, mu_frac, w, res["t_breach"], res["max_constraint_rel_res"]))
    assert ds_dropped == 0, f"Unexpected dropped cases in Kill-Test 3b: {ds_dropped}"
    max_t_breach_ds = max(c[3] for c in ds_cases)
    r_tiny = integrate_bounce_shell(0.0, 1e-12, 0.01, -1.0, s_keep=-1.0, t_end=2.0)
    t_exact_001 = math.acosh(math.cos(0.01) / math.cos(CHI_INJ))

    # 3c: Super-Hubble late-birth horizon-crossing shell (R(t0) = r_s = mu > 1 at t0 = 5.0, a(t0) = 74.2)
    # Choose mu = 5.0 (r_s = 5 > a_dS = 1), kappa = 0.5 (< 1), t0 = 5.0 => chi0 = arcsin(5 / cosh(5)) = 0.0674 < pi/10!
    superhubble = []
    for mu_sh, kappa_sh, t0_sh in [(3.0, 0.4, 4.5), (5.0, 0.5, 5.0), (10.0, 0.6, 5.5)]:
        a0, a_t0, _ = flrw_a_and_derivs(t0_sh, 0.0)
        chi0 = math.asin(mu_sh / a0)
        R0 = mu_sh
        # Exact BGG values at R = r_s = mu_sh:
        beta_m0 = (mu_sh - (1.0 - kappa_sh ** 2) * (R0 ** 3)) / (2.0 * kappa_sh * (R0 ** 2))
        beta_p0 = (mu_sh - (1.0 + kappa_sh ** 2) * (R0 ** 3)) / (2.0 * kappa_sh * (R0 ** 2))
        # At R = r_s, Rdot = +|beta_p0| (expanding out of white-hole/black-hole horizon into Quadrant III!)
        Rd0 = abs(beta_p0)
        # Inverse kinematics (s_keep = -1, outside dS horizon 1 - R0^2 < 0):
        denom = 1.0 - R0 ** 2
        td0 = (-beta_m0 * math.cos(chi0) - Rd0 * a_t0 * math.sin(chi0)) / denom
        achid0 = (Rd0 * math.cos(chi0) + beta_m0 * a_t0 * math.sin(chi0)) / denom
        if td0 <= 0:
            # Try opposite Rdot sign (contracting across horizon)
            Rd0 = -abs(beta_p0)
            td0 = (-beta_m0 * math.cos(chi0) - Rd0 * a_t0 * math.sin(chi0)) / denom
            achid0 = (Rd0 * math.cos(chi0) + beta_m0 * a_t0 * math.sin(chi0)) / denom
        psi0 = math.asinh(achid0)
        assert abs(math.cosh(psi0) - td0) < 1e-11, f"{math.cosh(psi0)} vs {td0}"

        # Integrate forward (t0 -> 10) and backward (t0 -> 0) using exact pure-dS K^tau_tau,-
        def rhs_sh(t, state):
            chi, psi, tau = state
            psi_c = max(-20.0, min(20.0, psi))
            a, a_t, _ = flrw_a_and_derivs(t, 0.0)
            H = a_t / a
            ch, sh, th = math.cosh(psi_c), math.sinh(psi_c), math.tanh(psi_c)
            R = max(1e-8, a * math.sin(chi))
            Ktt_m_exact = -mu_sh / (kappa_sh * (R ** 3)) - (1.0 - kappa_sh ** 2) / (2.0 * kappa_sh)
            # s_keep = -1 => Ktt_m = -(psidot + H sh) => psidot = -Ktt_m - H sh
            psidot = -Ktt_m_exact - H * sh
            return [th / a, psidot / ch, 1.0 / ch]

        def ev_eq(t, state):
            return (math.pi / 2.0 - 1e-3) - state[0]
        ev_eq.terminal = True

        def ev_zero(t, state):
            return state[0] - 1e-4
        ev_zero.terminal = True

        sol_f = solve_ivp(rhs_sh, [t0_sh, 10.0], [chi0, psi0, 0.0], method="DOP853",
                          rtol=1e-12, atol=1e-14, events=[ev_eq, ev_zero], max_step=0.02)
        sol_b = solve_ivp(rhs_sh, [t0_sh, -1.0], [chi0, psi0, 0.0], method="DOP853",
                          rtol=1e-12, atol=1e-14, events=[ev_eq, ev_zero], max_step=0.02)

        R_b = np.cosh(sol_b.t) * np.sin(sol_b.y[0, :])
        R_f = np.cosh(sol_f.t) * np.sin(sol_f.y[0, :])
        # Verify exact algebraic beta_m residual
        bm_f = -(np.cosh(sol_f.y[1, :]) * np.cos(sol_f.y[0, :]) + np.sinh(sol_f.y[1, :]) * np.sinh(sol_f.t) * np.sin(sol_f.y[0, :]))
        bm_f_alg = (mu_sh - (1.0 - kappa_sh ** 2) * (R_f ** 3)) / (2.0 * kappa_sh * (R_f ** 2))
        res_f = float(np.max(np.abs(bm_f - bm_f_alg) / np.maximum(1.0, np.abs(bm_f_alg))))
        bm_b = -(np.cosh(sol_b.y[1, :]) * np.cos(sol_b.y[0, :]) + np.sinh(sol_b.y[1, :]) * np.sinh(sol_b.t) * np.sin(sol_b.y[0, :]))
        bm_b_alg = (mu_sh - (1.0 - kappa_sh ** 2) * (R_b ** 3)) / (2.0 * kappa_sh * (R_b ** 2))
        res_b = float(np.max(np.abs(bm_b - bm_b_alg) / np.maximum(1.0, np.abs(bm_b_alg))))

        # Find where chi hits pi/10 going backward!
        idx_b_inj = np.where(sol_b.y[0, :] >= CHI_INJ)[0]
        t_b_inj = float(sol_b.t[idx_b_inj[0]]) if len(idx_b_inj) > 0 else None
        R_b_inj = float(R_b[idx_b_inj[0]]) if len(idx_b_inj) > 0 else None

        superhubble.append({
            "mu": mu_sh, "kappa": kappa_sh, "t0": t0_sh, "chi0": chi0, "R0_eq_rs": R0,
            "Rd0": Rd0, "td0": td0, "psi0": psi0, "beta_m0": beta_m0, "beta_p0": beta_p0,
            "fwd_chi_max": float(np.max(sol_f.y[0, :])),
            "fwd_stays_below_inj": bool(np.max(sol_f.y[0, :]) < CHI_INJ),
            "fwd_res": res_f,
            "bwd_t_end": float(sol_b.t[-1]),
            "bwd_chi_max": float(np.max(sol_b.y[0, :])),
            "bwd_R_min": float(np.min(R_b)),
            "bwd_breaches_inj": bool(np.max(sol_b.y[0, :]) >= CHI_INJ),
            "bwd_t_at_inj": t_b_inj,
            "bwd_R_at_inj": R_b_inj,
            "bwd_res": res_b,
        })

    # 3d: Page 7 kept-volume crossover sliver
    p7_kept = []
    for chi_b in (0.28, 0.29, 0.30, 0.31):
        f_exc = (60.0 / math.pi) * (2.0 * chi_b - math.sin(2.0 * chi_b))
        mu_keep = (math.pi / 80.0) * (1.0 - f_exc)
        res = integrate_bounce_shell(0.0, mu_keep, chi_b, -1.0, s_keep=-1.0, t_end=2.0)
        p7_kept.append({
            "chi_b": chi_b, "f_exc": f_exc, "mu_keep": mu_keep,
            "feasible_at_bounce": res["feasible"],
            "kappa0": res.get("kappa0"),
            "t_breach": res.get("t_breach"),
            "max_res": res.get("max_constraint_rel_res"),
        })

    return {
        "non_symmetric_bounce_tests": nonsym,
        "any_non_symmetric_survivor": any(r["survived_eternally"] for r in nonsym),
        "non_symmetric_max_res_all_12": max(r["max_res"] for r in nonsym),
        "pure_dS_total_cases": len(ds_cases),
        "pure_dS_dropped_cases": ds_dropped,
        "pure_dS_analytical_t_bound": t_bound_ds,
        "pure_dS_observed_max_t_breach": max_t_breach_ds,
        "pure_dS_mu0_check_err": abs(r_tiny["t_breach"] - t_exact_001),
        "superhubble_horizon_crossing_tests": superhubble,
        "page7_kept_volume_sliver": p7_kept,
    }


# =============================================================================
# 4. DIAGNOSTIC 4-PANEL PUBLICATION FIGURE
# =============================================================================

def make_diagnostic_figure(bgg_res, kt_res):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), dpi=200)

    # --- Panel A: Bounce WEC Mass Bound vs Page 7 Ansatz ---
    ax = axes[0, 0]
    chis = np.linspace(0.001, CHI_INJ, 300)
    mu_wec = np.sin(chis) ** 3
    f_exc = (60.0 / math.pi) * (2.0 * chis - np.sin(2.0 * chis))
    mu_p7_full = np.full_like(chis, math.pi / 80.0)
    mu_p7_keep = (math.pi / 80.0) * (1.0 - f_exc)

    ax.fill_between(chis / CHI_INJ, 0.0, mu_wec, color="#2b6cb0", alpha=0.18,
                    label=r"WEC-feasible bounce instant ($\mu/y_+ < \sin^3\chi_b$)")
    ax.plot(chis / CHI_INJ, mu_wec, color="#1a365d", lw=2.0,
            label=r"WEC ceiling $\sin^3\chi_b$ (max $0.02951$)")
    ax.plot(chis / CHI_INJ, mu_p7_full, color="#c53030", lw=2.0, ls="--",
            label=r"Page 7 full $\mathcal{E}_{\min}=Mc^2$: $\pi/80 = 0.03927$ ($1.331\times$ above)")
    ax.plot(chis / CHI_INJ, mu_p7_keep, color="#d69e2e", lw=1.8, ls="-.",
            label=r"Page 7 kept-vol: $\frac{\pi}{80}(1-f_{\rm exc})$ (crosses at $0.866\,\chi_{\rm inj}$)")
    ax.axvline(1.0, color="#111111", lw=1.2, ls=":")
    ax.set_xlim(0.0, 1.06)
    ax.set_ylim(0.0, 0.056)
    ax.set_xlabel(r"Bounce shell angle $\chi_b / \chi_{\rm inj}$  ($\chi_{\rm inj} = \pi/10 = 18^\circ$)")
    ax.set_ylabel(r"Scaled BH radius $r_s / a_{\min} = \mu / y_+(\gamma)$")
    ax.set_title(r"(a) Bounce-Epoch WEC Ceiling vs. Page 7 ($\mathcal{E}_{\min}=Mc^2$)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=7.6)
    ax.grid(alpha=0.25)

    # --- Panel B: Exact S^3 BGG Benchmark vs S^3/I* Injectivity Wall ---
    ax = axes[0, 1]
    t_s3 = np.array(bgg_res["s3_curve"]["t"])
    chi_s3 = np.array(bgg_res["s3_curve"]["chi"])
    ax.plot(t_s3, np.degrees(chi_s3), color="#2f855a", lw=2.2,
            label=r"$S^3$ BGG monotone ($\kappa=0.5, \mu=1.5\mu_{\rm cr}$, $s_{\rm keep}=+1$)")

    # Overlay representative punctured S^3/I* bounce shells (gamma=0, 0.2478)
    for gamma, mu_f, col, ls, lbl in [
        (0.0, 0.5, "#1a365d", "-", r"$S^3/I^*$ bounce $\gamma=0, w=-1$ (outward breach)"),
        (0.0, 0.5, "#3182ce", "--", r"$S^3/I^*$ bounce $\gamma=0, w=0$ (dust breach)"),
        (0.247765741792, 0.35, "#805ad5", "-", r"$S^3/I^*$ $k=2$ $\gamma_2$, $\mu/\mu_{\max}=0.35$ (rebound$\to$breach)"),
        (0.247765741792, 0.10, "#c53030", "-.", r"$S^3/I^*$ $k=2$ $\gamma_2$, $\mu/\mu_{\max}=0.10$ (inward WEC fail)"),
    ]:
        w_val = 0.0 if "dust" in lbl else -1.0
        y_p = math.sqrt(0.5 + math.sqrt(0.25 - gamma))
        chi_b = 0.15
        mu_v = mu_f * y_p * (math.sin(chi_b) ** 3)
        R0 = y_p * math.sin(chi_b)
        k0 = (-math.cos(chi_b) + math.sqrt(1.0 - mu_v / R0)) / R0
        out = integrate_shell_cosmic_time(gamma, mu_v, 0.0, chi_b, 0.0, k0, w_val,
                                          s_keep=-1.0, t_end=3.0, stop_after_inj=False)
        ax.plot(out["sol"].t, np.degrees(out["sol"].y[0, :]), color=col, lw=1.7, ls=ls, label=lbl)

    ax.axhline(18.0, color="#c53030", lw=1.6, ls=":", label=r"$S^3/I^*$ injectivity wall $\chi_{\rm inj}=18^\circ$ ($\pi/10$)")
    ax.axhline(90.0, color="#718096", lw=1.1, ls="--", label=r"$S^3$ equator $\chi=90^\circ$ ($\pi/2$)")
    ax.axhspan(18.0, 175.0, color="#c53030", alpha=0.07)
    ax.set_xlim(-2.25, 2.3)
    ax.set_ylim(0.0, 172.0)
    ax.set_xlabel(r"FLRW cosmic time $t / a_{\rm dS}$")
    ax.set_ylabel(r"Shell polar angle $\chi(t)$ [deg]")
    ax.set_title(r"(b) $S^3$ BGG Equator Crossing vs. $S^3/I^*$ $18^\circ$ Wall")
    ax.legend(loc="upper left", ncol=1, frameon=True, framealpha=0.95, fontsize=6.6)
    ax.grid(alpha=0.25)

    # --- Panel C: Radiation Bifurcation Phase Diagram (gamma vs mu/mu_max) at chi_b = 0.15 ---
    ax = axes[1, 0]
    chi_b = 0.15
    g_grid = np.linspace(0.0, 0.2495, 200)
    mu_psi0_curve = []
    for g in g_grid:
        yp = math.sqrt(0.5 + math.sqrt(0.25 - g))
        a_tt0 = (2.0 * yp * yp - 1.0) / yp
        R0 = yp * math.sin(chi_b)
        mu_m = yp * (math.sin(chi_b) ** 3)
        def f_psi(mu_val):
            bp = math.sqrt(1.0 - mu_val / R0)
            k0 = (bp - math.cos(chi_b)) / R0
            return a_tt0 * math.sin(chi_b) + 0.5 * mu_val / (R0 ** 2) - k0 * bp
        if f_psi(1e-12) >= 0:
            mu_psi0_curve.append(0.0)
        elif f_psi(mu_m * (1.0 - 1e-9)) <= 0:
            mu_psi0_curve.append(1.0)
        else:
            ms = brentq(f_psi, 1e-12, mu_m * (1.0 - 1e-9))
            mu_psi0_curve.append(ms / mu_m)
    mu_psi0_arr = np.array(mu_psi0_curve)

    # Compute EXACT dynamical separatrix between inward WEC collapse and outward breach
    g_dyn_pts = [0.1868, 0.195, 0.20, 0.21, 0.22, 0.23, 0.24, 0.245, 0.24776574, 0.2495]
    mu_dyn_pts = [0.0]
    for g in g_dyn_pts[1:]:
        yp = math.sqrt(0.5 + math.sqrt(0.25 - g))
        mu_m = yp * (math.sin(chi_b) ** 3)
        R0 = yp * math.sin(chi_b)
        def is_outward(mu_frac):
            mu_v = mu_frac * mu_m
            k0 = (math.sqrt(1.0 - mu_v / R0) - math.cos(chi_b)) / R0
            o = integrate_shell_cosmic_time(g, mu_v, 0.0, chi_b, 0.0, k0, -1.0,
                                            s_keep=-1.0, t_end=5.0, stop_after_inj=True)
            return o["breached_pi_over_10"]
        lo, hi = 1e-6, 0.6
        if not is_outward(lo):
            for _ in range(22):
                mid = 0.5 * (lo + hi)
                if is_outward(mid):
                    hi = mid
                else:
                    lo = mid
            mu_dyn_pts.append(0.5 * (lo + hi))
        else:
            mu_dyn_pts.append(0.0)

    g_full_dyn = np.concatenate([[0.0], np.array(g_dyn_pts)])
    mu_full_dyn = np.concatenate([[0.0], np.array(mu_dyn_pts)])
    mu_dyn_interp = np.interp(g_grid, g_full_dyn, mu_full_dyn)

    ax.fill_between(g_grid, mu_dyn_interp, 1.0, color="#2b6cb0", alpha=0.16,
                    label=r"Regime I: Outward blowout $\to$ $\chi=\pi/10$ breach")
    ax.fill_between(g_grid, 0.0, mu_dyn_interp, color="#c53030", alpha=0.24,
                    label=r"Regime II: Radiation-drained collapse $\to \sigma \to 0^+$")
    ax.plot(g_grid, mu_dyn_interp, color="#111111", lw=2.0,
            label=r"Exact dynamical separatrix ($\sigma_{\min}=0$ grazing)")
    ax.plot(g_grid, mu_psi0_arr, color="#4a5568", lw=1.4, ls="--",
            label=r"Initial dip locus $\dot\psi(0)=0$ ($\gamma_{\rm crit}=0.1868$)")
    ax.axvline(0.24776574, color="#805ad5", lw=1.3, ls=":", label=r"$k=2$ resonance $\gamma_2 = 0.2478$")
    ax.axvline(0.25, color="#111111", lw=1.1, ls="-")
    ax.set_xlim(0.0, 0.255)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Radiation parameter $\gamma = \Lambda B / (3 c^2)$")
    ax.set_ylabel(r"Normalized BH mass $\mu / (y_+ \sin^3\chi_b)$")
    ax.set_title(r"(c) Two-Regime No-Go Phase Diagram ($\chi_b=0.15, w=-1$)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.94, fontsize=7.2)
    ax.grid(alpha=0.25)

    # --- Panel D: Static/Comoving Minimal NEC Violation Floor ---
    ax = axes[1, 1]
    R_vals = np.linspace(0.02, math.sin(CHI_INJ), 200)
    for frac, col, ls in [(0.0, "#1a365d", "-"), (0.3, "#2b6cb0", "--"), (0.7, "#805ad5", "-."), (0.95, "#c53030", ":")]:
        mu_vals = frac * (R_vals ** 3)
        # Exact static NEC formula: 8*pi*G*(sigma + p_theta) = (1/R) * [(1 - 3 mu/(2 R))/sqrt(1 - mu/R) - 1/sqrt(1 - R^2)]
        nec_static = (1.0 / R_vals) * (
            (1.0 - 1.5 * mu_vals / R_vals) / np.sqrt(1.0 - mu_vals / R_vals)
            - 1.0 / np.sqrt(1.0 - R_vals ** 2)
        )
        ax.plot(R_vals, nec_static, color=col, lw=1.8, ls=ls,
                label=rf"Static $\mu/R_0^3 = {frac:.2f}$ (WEC $\sigma \geq 0$)")
    ax.axhline(0.0, color="#111111", lw=1.1)
    ax.axvline(math.sin(CHI_INJ), color="#c53030", lw=1.2, ls=":")
    ax.set_xlim(0.0, math.sin(CHI_INJ) * 1.05)
    ax.set_xlabel(r"Static shell radius $R_0 / a_{\rm dS}$  ($R_{\rm inj} = \sin(\pi/10) = 0.3090$)")
    ax.set_ylabel(r"$8\pi G(\sigma + p_\theta)\cdot a_{\rm dS}$  (NEC requires $\geq 0$)")
    ax.set_title(r"(d) Exact Static-Shell NEC Violation Theorem ($\sigma+p_\theta < 0$)")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.8)
    ax.grid(alpha=0.25)

    fig.tight_layout(pad=1.5)
    fig_path = OUT_DIR / "israel_darmois_poincare.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    return str(fig_path)


def main():
    bgg_res = bgg_benchmark()
    fmm_res = fmm_spacelike_benchmark()
    kt_res = killtest_non_symmetric_and_superhubble()
    fig_path = make_diagnostic_figure(bgg_res, kt_res)

    out = {
        "bgg_benchmark": {k: v for k, v in bgg_res.items() if k != "s3_curve"},
        "fmm_spacelike_benchmark": fmm_res,
        "killtests": kt_res,
        "figure_path": fig_path,
    }
    (OUT_DIR / "killtests_and_benchmarks.json").write_text(json.dumps(out, indent=2))

    print("=== BENCHMARK 1: BGG (1987) ON S^3 vs S^3/I* ===")
    for row in bgg_res["bgg_critical_table"]:
        rm = f"{row['R_minus_cr']:.4f}" if row["R_minus_cr"] is not None else "None"
        print(f"  k={row['kappa']:.1f} | mu_cr={row['mu_cr']:.5f} | 1-k^2={1-row['kappa']**2:+.2f} | "
              f"R*={row['R_star']:.4f} | R+={row['R_plus_cr']:.4f} | R-={rm} | "
              f"bm={row['beta_m_at_star']:+.4f} | bp={row['beta_p_at_star']:+.4f}")
    d = bgg_res["s3_supercritical_demo"]
    print(f"  S^3 supercritical demo: kappa={d['kappa']}, mu_cr={d['mu_cr']:.5f}, mu={d['mu_used']:.5f}, t_start={d['t_start']}")
    print(f"  chi: {d['chi_start_deg']:.2f} deg -> max {d['chi_max_deg']:.2f} deg -> chi(t=5)={d['chi_end_deg']:.2f} deg | crosses R=r_s: {d['crosses_horizon_R_eq_rs']} | crosses equator 90 deg: {d['crosses_equator_pi_over_2']}")
    print(f"  At chi=18 deg: t={d['t_at_chi_pi_over_10']:.4f}, R={d['R_at_chi_pi_over_10']:.4f}, beta_m={d['beta_m_at_chi_pi_over_10']:+.4f}")
    print(f"  beta_m flips (+ -> -) on S^3: {d['beta_m_flips_sign_on_S3']} | beta_p flips into QIII: {d['beta_p_flips_sign_into_QIII']} | max_res={d['max_constraint_rel_res']:.2e}")

    print("\n=== BENCHMARK 2: FMM (1990) SPACELIKE JUNCTION ===")
    print(f"  KS 3-Ricci anisotropy norm = {fmm_res['fmm_kantowski_sachs']['intrinsic_3Ricci_anisotropy_norm']:.4f} (!= 0)")
    ann = fmm_res["s3_istar_spacelike_annulus_in_schw_interior"]
    print(f"  S^3/I* spacelike annulus: min (dT/dchi)^2 = {ann['min_dT_dchi_sq']:.4f} > 0, "
          f"[K^th_th] in [{ann['min_jump_Kth']:.3f}, {ann['max_jump_Kth']:.3f}], chi=0 hits r=0 singularity")

    print("\n=== KILL-TEST 3a: NON-SYMMETRIC BOUNCE SHELLS (ALL 12 TESTS) ===")
    print(f"  Any eternal survivor: {kt_res['any_non_symmetric_survivor']} (max_res_all_12 = {kt_res['non_symmetric_max_res_all_12']:.2e})")
    for r in kt_res["non_symmetric_bounce_tests"]:
        print(f"    g={r['gamma']:.6f}, psi0={r['psi0']:+.1f} | fwd_breach={r['fwd_breach']} (chi_max={r['fwd_chi_max']:.4f}, t_f={r['fwd_t_final']:.3f}) | "
              f"bwd_breach={r['bwd_breach']} (chi_max={r['bwd_chi_max']:.4f}, t_b={r['bwd_t_final']:.3f}) | res={r['max_res']:.2e}")

    print("\n=== KILL-TEST 3b: PURE dS ANALYTICAL BREACH TIME BOUND ===")
    print(f"  Cases tested: {kt_res['pure_dS_total_cases']} (dropped: {kt_res['pure_dS_dropped_cases']})")
    print(f"  Analytical arcosh(sec(pi/10)) = {kt_res['pure_dS_analytical_t_bound']:.6f}")
    print(f"  Observed max t_breach (gamma=0) = {kt_res['pure_dS_observed_max_t_breach']:.6f}")
    print(f"  mu->0 exact check error = {kt_res['pure_dS_mu0_check_err']:.3e}")

    print("\n=== KILL-TEST 3c: SUPER-HUBBLE LATE-BIRTH HORIZON CROSSING ===")
    for r in kt_res["superhubble_horizon_crossing_tests"]:
        print(f"  mu={r['mu']}, k={r['kappa']}, t0={r['t0']}, chi0={r['chi0']:.4f} | "
              f"fwd_max={r['fwd_chi_max']:.4f} (<pi/10:{r['fwd_stays_below_inj']}, res={r['fwd_res']:.2e}) | "
              f"bwd_breach={r['bwd_breaches_inj']} (t_inj={r['bwd_t_at_inj']}, R_inj={r['bwd_R_at_inj']}, R_min={r['bwd_R_min']:.3f}, res={r['bwd_res']:.2e})")

    print("\n=== KILL-TEST 3d: PAGE 7 KEPT-VOLUME SLIVER ===")
    for r in kt_res["page7_kept_volume_sliver"]:
        print(f"  chi_b={r['chi_b']:.2f}, f_exc={r['f_exc']:.3f}, mu_keep={r['mu_keep']:.5f}, "
              f"k0={r['kappa0']:.5f}, t_breach={r['t_breach']:.4f}, res={r['max_res']:.2e}")
    print(f"\nFigure saved: {fig_path}")


if __name__ == "__main__":
    main()
