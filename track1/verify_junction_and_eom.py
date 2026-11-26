#!/usr/bin/env python3
"""
Stream p04_israel_darmois — Goals 1 & 2 (Post-Audit Hardened):
Exact Israel-Darmois spherical thin-shell junction conditions between
Schwarzschild (mass M) and closed FLRW Einstein-Maxwell-Lambda on S^3/I*.

Conventions (matching Poincare_Sphere_Gauge_Cosmology_Monograph.md):
  - Metric signature (-, +, +, +), c = 1
  - Dimensionless units: a_dS = sqrt(3/Lambda) = 1 (Lambda = 3),
    mu = 2 G M / a_dS = r_s / a_dS,
    kappa_sigma = 4 pi G sigma a_dS.
  - Closed FLRW on S^3/I*:
      ds_-^2 = -dt^2 + a(t)^2 (dchi^2 + sin^2(chi) dOmega_2^2),
      (da/dt)^2 = a^2 - 1 + gamma / a^2,  0 <= gamma < 1/4,
      a(t)^2 = 1/2 + sqrt(1/4 - gamma) cosh(2 t),  a_min = y_+(gamma).
  - Orientation sign s_keep:
      s_keep = -1 : keep punctured S^3/I* = (S^3/I*) \ B(chi), outward normal -> -d/dchi
      s_keep = +1 : keep small ball B(chi) (or S^3 cap [0, chi]), outward normal -> +d/dchi
"""

import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHI_INJ = math.pi / 10.0


# =============================================================================
# 1. EXACT GEOMETRY OF S^3/I* (BINARY ICOSAHEDRAL GROUP 2I SUBSET SU(2))
# =============================================================================

def build_binary_icosahedral_group():
    """Construct all 120 unit quaternions of I* = 2I subset S^3."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    pts = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = [0.0, 0.0, 0.0, 0.0]
            v[i] = s
            pts.append(tuple(v))
    for s0 in (-0.5, 0.5):
        for s1 in (-0.5, 0.5):
            for s2 in (-0.5, 0.5):
                for s3 in (-0.5, 0.5):
                    pts.append((s0, s1, s2, s3))
    even_perms = [
        (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
        (1, 0, 3, 2), (1, 2, 0, 3), (1, 3, 2, 0),
        (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
        (3, 0, 2, 1), (3, 1, 0, 2), (3, 2, 1, 0),
    ]
    base_vals = [0.0, 0.5, 0.5 * phi, 0.5 * inv_phi]
    for p in even_perms:
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                for s3 in (-1.0, 1.0):
                    raw = [0.0, s1 * base_vals[1], s2 * base_vals[2], s3 * base_vals[3]]
                    v = [0.0, 0.0, 0.0, 0.0]
                    for idx, pos in enumerate(p):
                        v[pos] = raw[idx]
                    pts.append(tuple(v))
    uniq = {tuple(np.round(row, 10)): row for row in np.array(pts, dtype=np.float64)}
    G = np.array(list(uniq.values()), dtype=np.float64)
    assert G.shape == (120, 4)
    assert np.max(np.abs(np.linalg.norm(G, axis=1) - 1.0)) < 1e-14
    return G


def verify_s3_istar_geometry():
    """Verify injectivity radius, L_min, dodecahedral circumradius, and volume fractions."""
    G = build_binary_icosahedral_group()
    e = np.array([1.0, 0.0, 0.0, 0.0])
    dots = G @ e
    non_id = dots < 1.0 - 1e-12
    dists = np.arccos(np.clip(dots[non_id], -1.0, 1.0))
    L_min = float(np.min(dists))
    n_nearest = int(np.sum(np.abs(dists - L_min) < 1e-12))
    chi_inj = 0.5 * L_min

    # Group closure check (all 120 x 120 products)
    keyset = {tuple(np.round(row, 9)) for row in G}
    for i in range(120):
        a = G[i]
        prods = np.column_stack([
            a[0]*G[:, 0] - a[1]*G[:, 1] - a[2]*G[:, 2] - a[3]*G[:, 3],
            a[0]*G[:, 1] + a[1]*G[:, 0] + a[2]*G[:, 3] - a[3]*G[:, 2],
            a[0]*G[:, 2] - a[1]*G[:, 3] + a[2]*G[:, 0] + a[3]*G[:, 1],
            a[0]*G[:, 3] + a[1]*G[:, 2] - a[2]*G[:, 1] + a[3]*G[:, 0],
        ])
        for row in prods:
            assert tuple(np.round(row, 9)) in keyset

    # Dirichlet dodecahedron vertices (20 vertices)
    nearest = G[non_id][np.abs(dists - L_min) < 1e-12]
    assert nearest.shape == (12, 4)
    v_uniq = {}
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                A = np.vstack([nearest[i] - e, nearest[j] - e, nearest[k] - e])
                _, s, Vt = np.linalg.svd(A)
                if s[2] < 1e-8:
                    continue
                v = Vt[3]
                if v[0] < 0:
                    v = -v
                v = v / np.linalg.norm(v)
                if np.all((nearest @ v) <= v[0] + 1e-10):
                    v_uniq[tuple(np.round(v, 9))] = v
    V = np.array(list(v_uniq.values()))
    chi_circ = float(np.max(np.arccos(np.clip(V[:, 0], -1.0, 1.0))))
    cos_circ_exact = math.sqrt(1.0 + 3.0 * math.cos(math.pi / 5.0)) / 2.0
    assert abs(math.cos(chi_circ) - cos_circ_exact) < 1e-13

    def vol_frac_excised(chi):
        return (60.0 / math.pi) * (2.0 * chi - math.sin(2.0 * chi))

    sin_inj = math.sin(chi_inj)
    assert abs(sin_inj - (math.sqrt(5.0) - 1.0) / 4.0) < 1e-15

    chi_keep_cross = float(brentq(
        lambda c: math.sin(c) ** 3 - (math.pi / 80.0) * (1.0 - vol_frac_excised(c)),
        1e-4, chi_inj
    ))

    return {
        "order_I_star": 120,
        "group_closure_verified": True,
        "L_min": L_min,
        "L_min_exact_pi_over_5": math.pi / 5.0,
        "n_nearest_neighbors": n_nearest,
        "chi_inj_rad": chi_inj,
        "chi_inj_deg": math.degrees(chi_inj),
        "sin_chi_inj": sin_inj,
        "cos_chi_inj": math.cos(chi_inj),
        "sin_chi_inj_cubed": sin_inj ** 3,
        "tan_half_chi_inj": math.tan(0.5 * chi_inj),
        "csc_chi_inj_2phi": 1.0 / sin_inj,
        "csc_2chi_inj": 1.0 / math.sin(2.0 * chi_inj),
        "n_dodecahedron_vertices": int(V.shape[0]),
        "chi_circ_rad": chi_circ,
        "chi_circ_deg": math.degrees(chi_circ),
        "cos_chi_circ_exact_err": abs(math.cos(chi_circ) - cos_circ_exact),
        "vol_frac_excised_at_chi_inj": vol_frac_excised(chi_inj),
        "vol_frac_kept_at_chi_inj": 1.0 - vol_frac_excised(chi_inj),
        "page7_mu_over_yplus": math.pi / 80.0,
        "inj_mu_max_over_yplus": sin_inj ** 3,
        "page7_over_inj_ratio": (math.pi / 80.0) / (sin_inj ** 3),
        "page7_kept_vol_crossover_chi_rad": chi_keep_cross,
        "page7_kept_vol_crossover_frac_of_inj": chi_keep_cross / chi_inj,
        "page7_kept_vol_excised_at_crossover": vol_frac_excised(chi_keep_cross),
    }


# =============================================================================
# 2. EXACT FLRW BACKGROUND (EINSTEIN-MAXWELL-LAMBDA, 0 <= gamma < 1/4)
# =============================================================================

def flrw_a_and_derivs(t, gamma):
    """Exact closed-form a(t), da/dt, d2a/dt2 for 0 <= gamma <= 1/4, a_dS = 1."""
    t_safe = max(-25.0, min(25.0, t))
    s = math.sqrt(max(0.0, 0.25 - gamma))
    a2 = 0.5 + s * math.cosh(2.0 * t_safe)
    a = math.sqrt(a2)
    a_t = s * math.sinh(2.0 * t_safe) / a
    a_tt = a - gamma / (a2 * a)
    return a, a_t, a_tt


def flrw_rho_p_scaled(a, gamma):
    """Return 8*pi*G*rho_tot, 8*pi*G*p_tot, 8*pi*G*rho_rad in a_dS = 1 units."""
    rho_tot_8piG = 3.0 * (1.0 + gamma / (a ** 4))
    rho_rad_8piG = 3.0 * gamma / (a ** 4)
    p_tot_8piG = 3.0 * (-1.0 + gamma / (3.0 * (a ** 4)))
    return rho_tot_8piG, p_tot_8piG, rho_rad_8piG


def remaining_conformal_time(t0, gamma):
    """Exact remaining conformal time int_{t0}^{inf} dt / a(t)."""
    if gamma == 0.0:
        return 2.0 * math.atan(math.exp(-t0))
    val, _ = quad(lambda u: 1.0 / flrw_a_and_derivs(u, gamma)[0], t0, t0 + 25.0, epsabs=1e-13, epsrel=1e-13)
    return float(val)


# =============================================================================
# 3. NUMERICAL VERIFICATION OF ALL ISRAEL-DARMOIS, CODAZZI & RAPIDITY IDENTITIES
# =============================================================================

def verify_israel_darmois_identities():
    """Verify all analytical junction, Codazzi, conservation, and rapidity identities."""
    max_errors = {
        "beta_minus_sq_err": 0.0,
        "codazzi_flrw_err": 0.0,
        "codazzi_schw_err": 0.0,
        "conservation_err": 0.0,
        "rapidity_Ktt_err": 0.0,
        "rapidity_Rdd_err": 0.0,
        "inverse_kinematics_err": 0.0,
    }
    eps = 1e-6

    for gamma in (0.0, 0.1, 0.24):
        for s_keep in (-1.0, 1.0):
            mu = 0.015

            def chi_fn(tau):
                return 0.18 + 0.04 * math.sin(0.7 * tau + 0.3)

            def chidot_fn(tau):
                return 0.04 * 0.7 * math.cos(0.7 * tau + 0.3)

            def chiddot_fn(tau):
                return -0.04 * 0.49 * math.sin(0.7 * tau + 0.3)

            sol_t = solve_ivp(
                lambda tau, y: [math.sqrt(1.0 + (flrw_a_and_derivs(y[0], gamma)[0] * chidot_fn(tau)) ** 2)],
                [0.0, 1.0], [0.4], rtol=1e-13, atol=1e-15, dense_output=True
            )

            def eval_state(tau):
                t = float(sol_t.sol(tau)[0])
                chi = chi_fn(tau)
                chid = chidot_fn(tau)
                chidd = chiddot_fn(tau)
                a, a_t, a_tt = flrw_a_and_derivs(t, gamma)
                td = math.sqrt(1.0 + (a * chid) ** 2)
                psi = math.asinh(a * chid)
                R = a * math.sin(chi)
                Rd = td * a_t * math.sin(chi) + a * chid * math.cos(chi)
                beta_m = s_keep * (td * math.cos(chi) + a * chid * a_t * math.sin(chi))
                twoGm_m_over_R = (1.0 + a_t ** 2) * (math.sin(chi) ** 2)
                Ktt_m = s_keep * (a * chidd / td + 2.0 * a_t * chid)
                beta_p = -math.sqrt(Rd ** 2 + 1.0 - mu / R)
                rho_8, p_8, _ = flrw_rho_p_scaled(a, gamma)
                flux_8piG = s_keep * (rho_8 + p_8) * a * td * chid
                return {
                    "t": t, "td": td, "chi": chi, "chid": chid, "psi": psi,
                    "a": a, "a_t": a_t, "a_tt": a_tt,
                    "R": R, "Rd": Rd, "beta_m": beta_m, "beta_p": beta_p,
                    "twoGm_m_over_R": twoGm_m_over_R, "Ktt_m": Ktt_m,
                    "flux_8piG": flux_8piG,
                }

            tau0 = 0.55
            st0 = eval_state(tau0)
            stp, stm = eval_state(tau0 + eps), eval_state(tau0 - eps)
            stp2, stm2 = eval_state(tau0 + 2 * eps), eval_state(tau0 - 2 * eps)

            def d_tau(key):
                return (-stp2[key] + 8 * stp[key] - 8 * stm[key] + stm2[key]) / (12 * eps)

            err1 = abs(st0["beta_m"] ** 2 - (st0["Rd"] ** 2 + 1.0 - st0["twoGm_m_over_R"]))
            max_errors["beta_minus_sq_err"] = max(max_errors["beta_minus_sq_err"], err1)

            dbeta_m = d_tau("beta_m")
            max_errors["codazzi_flrw_err"] = max(
                max_errors["codazzi_flrw_err"],
                abs((dbeta_m - st0["Rd"] * st0["Ktt_m"]) - (-0.5 * st0["R"] * st0["flux_8piG"]))
            )

            Rdd = d_tau("Rd")
            Ktt_p = (Rdd + 0.5 * mu / (st0["R"] ** 2)) / st0["beta_p"]
            max_errors["codazzi_schw_err"] = max(
                max_errors["codazzi_schw_err"], abs(d_tau("beta_p") - st0["Rd"] * Ktt_p)
            )

            def sigR2(st):
                return st["R"] * (st["beta_m"] - st["beta_p"])

            dsigR2 = (-sigR2(stp2) + 8 * sigR2(stp) - 8 * sigR2(stm) + sigR2(stm2)) / (12 * eps)
            p_theta_8piG = (Ktt_p - st0["Ktt_m"]) + (st0["beta_p"] - st0["beta_m"]) / st0["R"]
            lhs4 = dsigR2 + (0.5 * p_theta_8piG) * (2.0 * st0["R"] * st0["Rd"])
            rhs4 = -0.5 * (st0["R"] ** 2) * st0["flux_8piG"]
            max_errors["conservation_err"] = max(max_errors["conservation_err"], abs(lhs4 - rhs4))

            psid = d_tau("psi")
            H = st0["a_t"] / st0["a"]
            Ktt_m_rap = s_keep * (psid + H * math.sinh(st0["psi"]))
            max_errors["rapidity_Ktt_err"] = max(max_errors["rapidity_Ktt_err"], abs(Ktt_m_rap - st0["Ktt_m"]))

            ch, sh = math.cosh(st0["psi"]), math.sinh(st0["psi"])
            sc, cc = math.sin(st0["chi"]), math.cos(st0["chi"])
            Rdd_rap = (
                s_keep * st0["beta_m"] * psid
                + (ch ** 2) * st0["a_tt"] * sc
                + H * ch * sh * cc
                - (sh ** 2) * sc / st0["a"]
            )
            max_errors["rapidity_Rdd_err"] = max(max_errors["rapidity_Rdd_err"], abs(Rdd_rap - Rdd))

            denom = 1.0 - st0["twoGm_m_over_R"]
            td_inv = (s_keep * st0["beta_m"] * cc - st0["Rd"] * st0["a_t"] * sc) / denom
            achid_inv = (st0["Rd"] * cc - s_keep * st0["beta_m"] * st0["a_t"] * sc) / denom
            max_errors["inverse_kinematics_err"] = max(
                max_errors["inverse_kinematics_err"],
                max(abs(td_inv - st0["td"]), abs(achid_inv - st0["a"] * st0["chid"]))
            )

    return max_errors


# =============================================================================
# 4. EXACT RAPIDITY-COSMIC-TIME THIN-SHELL EOM SOLVER (EXPLICIT FATE CLASSIFIER)
# =============================================================================

def integrate_shell_cosmic_time(gamma, mu, t_init, chi_init, psi_init, kappa_init,
                                w_shell, s_keep=-1.0, t_end=8.0, stop_after_inj=True,
                                kappa_floor=1e-9, psi_cap=12.0, method="RK45"):
    """
    Integrate state y = [chi, psi, kappa, tau] as a function of FLRW cosmic time t.
    Explicitly inspects sol.status and terminal events — never misclassifies
    ultrarelativistic outward shells or WEC-collapse halts as 'survived'!
    """
    def rhs(t, state):
        chi, psi, kappa, _tau = state
        psi_c = max(-25.0, min(25.0, psi))
        a, a_t, a_tt = flrw_a_and_derivs(t, gamma)
        H = a_t / a
        ch = math.cosh(psi_c)
        sh = math.sinh(psi_c)
        th = math.tanh(psi_c)
        sc = math.sin(chi)
        cc = math.cos(chi)
        R = max(1e-12, a * sc)
        Rd = ch * a_t * sc + sh * cc
        beta_m = s_keep * (ch * cc + sh * a_t * sc)
        beta_p = beta_m - kappa * R

        Rdd_rem = (ch ** 2) * a_tt * sc + H * ch * sh * cc - (sh ** 2) * sc / a
        coeff = s_keep * (kappa * R) / beta_p
        rem = (Rdd_rem + 0.5 * mu / (R ** 2)) / beta_p - s_keep * H * sh
        psidot = ((1.0 + 2.0 * w_shell) * kappa - rem) / coeff

        rho_8, p_8, _ = flrw_rho_p_scaled(a, gamma)
        flux_8piG = s_keep * (rho_8 + p_8) * ch * sh
        kappadot = -2.0 * (1.0 + w_shell) * kappa * (Rd / R) - 0.5 * flux_8piG

        return [th / a, psidot / ch, kappadot / ch, 1.0 / ch]

    def ev_inj(t, state):
        return state[0] - CHI_INJ
    ev_inj.terminal = False
    ev_inj.direction = 1

    def ev_post_inj(t, state):
        return state[0] - (CHI_INJ + 0.05 if stop_after_inj else math.pi / 2.0 - 1e-3)
    ev_post_inj.terminal = True
    ev_post_inj.direction = 1

    def ev_chi_zero(t, state):
        return state[0] - 1e-5
    ev_chi_zero.terminal = True
    ev_chi_zero.direction = -1

    def ev_kappa_lo(t, state):
        return state[2] - kappa_floor
    ev_kappa_lo.terminal = True
    ev_kappa_lo.direction = -1

    def ev_psi_pos(t, state):
        return psi_cap - state[1]
    ev_psi_pos.terminal = True
    ev_psi_pos.direction = -1

    def ev_psi_neg(t, state):
        return state[1] + psi_cap
    ev_psi_neg.terminal = True
    ev_psi_neg.direction = -1

    def ev_betap_zero(t, state):
        chi, psi, kappa, _ = state
        a, a_t, _ = flrw_a_and_derivs(t, gamma)
        psi_c = max(-25.0, min(25.0, psi))
        beta_m = s_keep * (math.cosh(psi_c) * math.cos(chi) + math.sinh(psi_c) * a_t * math.sin(chi))
        return beta_m - kappa * a * math.sin(chi)
    ev_betap_zero.terminal = True

    sol = solve_ivp(
        rhs, [t_init, t_end], [chi_init, psi_init, kappa_init, 0.0],
        method=method, rtol=1e-10, atol=1e-12,
        events=[ev_inj, ev_post_inj, ev_chi_zero, ev_kappa_lo, ev_psi_pos, ev_psi_neg, ev_betap_zero],
        max_step=0.05,
    )

    max_res = 0.0
    max_beta_p = -1e18
    max_beta_m = -1e18
    for k in range(sol.y.shape[1]):
        t = sol.t[k]
        chi, psi, kappa, _ = sol.y[:, k]
        a, a_t, _ = flrw_a_and_derivs(t, gamma)
        ch, sh = math.cosh(psi), math.sinh(psi)
        R = a * math.sin(chi)
        Rd = ch * a_t * math.sin(chi) + sh * math.cos(chi)
        beta_m = s_keep * (ch * math.cos(chi) + sh * a_t * math.sin(chi))
        beta_p = beta_m - kappa * R
        beta_p_sq_true = Rd ** 2 + 1.0 - mu / R
        res = abs(beta_p ** 2 - beta_p_sq_true) / max(1.0, abs(beta_p_sq_true))
        max_res = max(max_res, res)
        max_beta_p = max(max_beta_p, beta_p)
        max_beta_m = max(max_beta_m, beta_m)

    t_breach = float(sol.t_events[0][0]) if len(sol.t_events[0]) > 0 else None
    tau_breach = float(sol.y_events[0][0][3]) if len(sol.t_events[0]) > 0 else None
    R_breach = None
    if t_breach is not None:
        ab, _, _ = flrw_a_and_derivs(t_breach, gamma)
        R_breach = float(ab * math.sin(CHI_INJ))

    # Explicit, unambiguous fate classification (closes BC-005!)
    chi_f = float(sol.y[0, -1])
    psi_f = float(sol.y[1, -1])
    kappa_f = float(sol.y[2, -1])
    t_f = float(sol.t[-1])

    if t_breach is not None or float(np.max(sol.y[0, :])) >= CHI_INJ:
        fate = "breach_outward"
    elif len(sol.t_events[4]) > 0 or (psi_f > 10.0 and sol.status == -1):
        # Ultra-relativistic outward shell (v > 1 - 1e-8) approaching null limit:
        # compute exact null-continuation breach time from (t_f, chi_f)
        if t_end > t_init:
            eta_rem = remaining_conformal_time(t_f, gamma)
            if chi_f + eta_rem >= CHI_INJ:
                # Solve int_{t_f}^{t_b} dt/a(t) = CHI_INJ - chi_f
                target = CHI_INJ - chi_f
                if gamma == 0.0:
                    # 2*arctan(e^-tf) - 2*arctan(e^-tb) = target
                    e_neg_tb = math.tan(math.atan(math.exp(-t_f)) - 0.5 * target)
                    t_breach = -math.log(e_neg_tb)
                else:
                    t_breach = float(brentq(
                        lambda tb: quad(lambda u: 1.0 / flrw_a_and_derivs(u, gamma)[0], t_f, tb)[0] - target,
                        t_f, t_f + 15.0
                    ))
                ab, _, _ = flrw_a_and_derivs(t_breach, gamma)
                R_breach = float(ab * math.sin(CHI_INJ))
                fate = "breach_outward_ultrarel"
            else:
                fate = "survived_ultrarel"
        else:
            fate = "ultrarel_backward"
    elif len(sol.t_events[3]) > 0 or len(sol.t_events[5]) > 0 or len(sol.t_events[2]) > 0 or (psi_f < -8.0 and kappa_f < 1e-5):
        fate = "inward_wec_collapse"
    elif sol.status == -1:
        fate = "solver_error"
    elif abs(t_f - t_end) < 1e-6:
        if t_end > t_init and chi_f + remaining_conformal_time(t_f, gamma) < CHI_INJ:
            fate = "survived_global"
        elif t_end < t_init:
            fate = "reached_t_end_backward"
        else:
            fate = "needs_longer_t_end"
    else:
        fate = "other_event"

    return {
        "sol": sol,
        "fate": fate,
        "t_breach": t_breach,
        "tau_breach": tau_breach,
        "R_breach": R_breach,
        "breached_pi_over_10": fate in ("breach_outward", "breach_outward_ultrarel"),
        "t_final": t_f,
        "chi_final": chi_f,
        "psi_final": psi_f,
        "kappa_final": kappa_f,
        "tau_final": float(sol.y[3, -1]),
        "max_constraint_rel_res": max_res,
        "max_beta_p": max_beta_p,
        "max_beta_m": max_beta_m,
        "status": int(sol.status),
    }


def integrate_bounce_shell(gamma, mu, chi_b, w_shell, s_keep=-1.0, t_end=8.0):
    a0, _, _ = flrw_a_and_derivs(0.0, gamma)
    R0 = a0 * math.sin(chi_b)
    beta_m0 = s_keep * math.cos(chi_b)
    if 1.0 - mu / R0 <= 0:
        return {"feasible": False, "fate": "infeasible", "reason": "R0 <= r_s"}
    beta_p0 = -math.sqrt(1.0 - mu / R0)
    kappa0 = (beta_m0 - beta_p0) / R0
    if kappa0 <= 0:
        return {"feasible": False, "fate": "infeasible", "reason": "kappa0 <= 0 (WEC violated)"}
    out = integrate_shell_cosmic_time(gamma, mu, 0.0, chi_b, 0.0, kappa0, w_shell, s_keep=s_keep, t_end=t_end)
    out.pop("sol")
    out.update({
        "feasible": True, "gamma": gamma, "mu": mu, "chi_b": chi_b,
        "w_shell": w_shell, "kappa0": kappa0, "R0": R0, "r_s_over_R0": mu / R0,
    })
    return out


def scan_parameter_space():
    """Scan across (gamma, mu, chi_b, w_shell) including w in (-1, -0.5) (closes BC-005 & BC-006)."""
    results = []
    gammas = [0.0, 0.10, 0.20, 0.247765741792]
    w_values = [-1.0, -0.9, -0.75, -0.5, 0.0, 0.5, 1.0]
    for gamma in gammas:
        y_plus = math.sqrt(0.5 + math.sqrt(0.25 - gamma))
        for chi_frac in (0.1, 0.5, 0.9):
            chi_b = chi_frac * CHI_INJ
            mu_local_max = y_plus * (math.sin(chi_b) ** 3)
            for mu_frac in (0.01, 0.5, 0.95):
                mu = mu_frac * mu_local_max
                for w in w_values:
                    results.append(integrate_bounce_shell(gamma, mu, chi_b, w, s_keep=-1.0, t_end=8.0))
    return results


def test_comoving_nec_violation():
    """
    Test comoving shell (chi(t) = chi_0) across mu/mu_max in {0.1, 0.5, 0.9} and gamma in
    {0, 0.10, 0.247765741792}, plus the exact unstable gamma = 1/4 Einstein-Static solution
    (closes BC-004!).
    """
    out = []
    for gamma in (0.0, 0.10, 0.247765741792):
        y_plus = math.sqrt(0.5 + math.sqrt(0.25 - gamma))
        for chi0 in (0.05, 0.15, 0.30):
            for mu_frac in (0.1, 0.5, 0.9):
                mu = mu_frac * y_plus * (math.sin(chi0) ** 3)
                ts = np.linspace(0.0, 8.0, 801)
                sig_8, sig_plus_p_8 = [], []
                for t in ts:
                    a, a_t, a_tt = flrw_a_and_derivs(t, gamma)
                    R = a * math.sin(chi0)
                    Rd = a_t * math.sin(chi0)
                    Rdd = a_tt * math.sin(chi0)
                    beta_m = -math.cos(chi0)
                    beta_p = -math.sqrt(Rd ** 2 + 1.0 - mu / R)
                    s8 = 2.0 * (beta_m - beta_p) / R
                    Ktt_p = (Rdd + 0.5 * mu / (R ** 2)) / beta_p
                    sp8 = 0.5 * s8 + Ktt_p
                    sig_8.append(s8)
                    sig_plus_p_8.append(sp8)
                sp_arr = np.array(sig_plus_p_8)
                # Find scale factor a_nec_zero where NEC flips + -> - if positive at bounce
                a_nec_zero = None
                if sp_arr[0] > 0:
                    idx_neg = int(np.argmax(sp_arr < 0))
                    t_zero = brentq(
                        lambda t: (
                            (math.sqrt((flrw_a_and_derivs(t, gamma)[1] * math.sin(chi0)) ** 2
                                       + 1.0 - mu / (flrw_a_and_derivs(t, gamma)[0] * math.sin(chi0)))
                             - math.cos(chi0)) / (flrw_a_and_derivs(t, gamma)[0] * math.sin(chi0))
                            - (flrw_a_and_derivs(t, gamma)[2] * math.sin(chi0)
                               + 0.5 * mu / ((flrw_a_and_derivs(t, gamma)[0] * math.sin(chi0)) ** 2))
                            / math.sqrt((flrw_a_and_derivs(t, gamma)[1] * math.sin(chi0)) ** 2
                                        + 1.0 - mu / (flrw_a_and_derivs(t, gamma)[0] * math.sin(chi0)))
                        ),
                        ts[max(0, idx_neg - 1)], ts[idx_neg]
                    )
                    a_nec_zero = flrw_a_and_derivs(t_zero, gamma)[0]
                out.append({
                    "gamma": gamma, "chi0": chi0, "mu_frac": mu_frac, "mu": mu,
                    "min_8piG_sigma": float(np.min(sig_8)),
                    "nec_at_bounce": float(sp_arr[0]),
                    "a_nec_zero": a_nec_zero,
                    "nec_at_t8": float(sp_arr[-1]),
                    "nec_violated_globally_late": bool(sp_arr[-1] < 0.0),
                    "nec_violated_everywhere_on_0_8": bool(np.all(sp_arr < 0.0)),
                })

    # Exact unstable gamma = 1/4 Einstein-Static comoving WEC+NEC+DEC shell:
    # a = 1/sqrt(2), a_t = a_tt = 0 => R = sin(chi0)/sqrt(2), beta_- = -cos(chi0), beta_+ = -sqrt(1 - mu/R)
    # 8piG sigma = 2*(sqrt(1-mu/R) - cos(chi0))/R > 0
    # 8piG (sigma + p_theta) = 4piG sigma - (mu/(2 R^2))/sqrt(1-mu/R)
    es_cases = []
    a_es = 1.0 / math.sqrt(2.0)
    for chi0 in (0.15, 0.28):
        R_es = a_es * math.sin(chi0)
        mu_max_es = a_es * (math.sin(chi0) ** 3)
        for mu_frac in (0.1, 0.4, 0.6):
            mu = mu_frac * mu_max_es
            bp = math.sqrt(1.0 - mu / R_es)
            sig_8 = 2.0 * (bp - math.cos(chi0)) / R_es
            sp_8 = 0.5 * sig_8 - (0.5 * mu / (R_es ** 2)) / bp
            p_8 = sp_8 - sig_8
            w_eff = p_8 / sig_8
            es_cases.append({
                "chi0": chi0, "mu_frac": mu_frac, "mu": mu,
                "sigma_8piG": sig_8, "sigma_plus_p_8piG": sp_8, "w_eff": w_eff,
                "satisfies_WEC_NEC_DEC": bool(sig_8 > 0 and sp_8 >= 0 and abs(p_8) <= sig_8),
            })
    return {"bounce_branch_cases": out, "einstein_static_gamma_quarter_cases": es_cases}


def test_late_birth_window():
    """
    Test BOTH late-birth thresholds (closes BC-001!):
      (a) General psi(t0) in R (inward-launched shell):
          eta_rem(t0) < 2*chi_inj = pi/5  <=>  t0 > -ln(tan(pi/10)) = 1.124177, a(t0) > csc(pi/5) = 1.701302.
          Includes exact sub-Hubble boosted CMC counterexample (alpha=-2.0, C=0.9998, t0=1.4, a0=2.1509, R0=0.4581 < 1).
      (b) Comoving/outward-launched sub-threshold (psi(t0) >= 0):
          eta_rem(t0) < chi_inj = pi/10   <=>  t0 > -ln(tan(pi/20)) = 1.842730, a(t0) > csc(pi/10) = 2*phi = 3.236068.
    Verifies backward obstruction to t = 0 in EVERY case!
    """
    t_crit_general = -math.log(math.tan(math.pi / 10.0))
    a_crit_general = math.cosh(t_crit_general)  # = csc(pi/5) = 1.7013016
    t_crit_outward = -math.log(math.tan(math.pi / 20.0))
    a_crit_outward = math.cosh(t_crit_outward)  # = csc(pi/10) = 2*phi = 3.2360680

    # (a) Exact boosted CMC sub-Hubble counterexample (BC-001):
    # cos(chi(t)) = (C + sinh(|alpha|)*sinh(t)) / (cosh(|alpha|)*cosh(t)), alpha = -2.0, C = 0.9998
    alpha_b = 2.0
    C_b = 0.9998
    kappa_b = math.sqrt((1.0 - C_b) / (1.0 + C_b))  # exact kappa for mu=0 CMC slice!
    # Check: (1 - k^2)/(1 + k^2) = C_b => k^2 = (1-C_b)/(1+C_b) => k = 0.0100005
    t0_sub = 1.4
    a0_sub = math.cosh(t0_sub)
    cos_chi0 = (C_b + math.sinh(alpha_b) * math.sinh(t0_sub)) / (math.cosh(alpha_b) * math.cosh(t0_sub))
    chi0_sub = math.acos(cos_chi0)
    R0_sub = a0_sub * math.sin(chi0_sub)
    # Exact dchi/dt from differentiating cos(chi(t)):
    # -sin(chi) dchi/dt = (sinh(alpha)*cosh(t)^2 - (C + sinh(alpha)*sinh(t))*sinh(t)) / (cosh(alpha)*cosh(t)^2)
    #                   = (sinh(alpha) - C*sinh(t)) / (cosh(alpha)*cosh(t)^2)
    dchi_dt0 = -(math.sinh(alpha_b) - C_b * math.sinh(t0_sub)) / (math.cosh(alpha_b) * (a0_sub ** 2) * math.sin(chi0_sub))
    v0_sub = a0_sub * dchi_dt0  # strictly in (-1, 0) (inward!)
    psi0_sub = math.atanh(v0_sub)

    # Use small nonzero Schwarzschild mass mu = 1e-8 and adjust kappa0 slightly to satisfy exact Israel constraint
    mu_sub = 1e-8
    ch0, sh0 = math.cosh(psi0_sub), math.sinh(psi0_sub)
    Rd0_sub = ch0 * math.sinh(t0_sub) * math.sin(chi0_sub) + sh0 * math.cos(chi0_sub)
    beta_m0_sub = -(ch0 * math.cos(chi0_sub) + sh0 * math.sinh(t0_sub) * math.sin(chi0_sub))
    beta_p0_sub = -math.sqrt(Rd0_sub ** 2 + 1.0 - mu_sub / R0_sub)
    kappa0_sub = (beta_m0_sub - beta_p0_sub) / R0_sub
    assert kappa0_sub > 0 and beta_p0_sub < beta_m0_sub < 0

    fwd_sub = integrate_shell_cosmic_time(0.0, mu_sub, t0_sub, chi0_sub, psi0_sub, kappa0_sub, -1.0,
                                          s_keep=-1.0, t_end=10.0, stop_after_inj=False, method="DOP853")
    bwd_sub = integrate_shell_cosmic_time(0.0, mu_sub, t0_sub, chi0_sub, psi0_sub, kappa0_sub, -1.0,
                                          s_keep=-1.0, t_end=-0.5, stop_after_inj=False, method="DOP853")
    sol_fs, sol_bs = fwd_sub["sol"], bwd_sub["sol"]
    sub_hubble_demo = {
        "t0": t0_sub, "a0": a0_sub, "chi0": chi0_sub, "R0": R0_sub,
        "sub_hubble_R0_lt_1": R0_sub < 1.0,
        "below_2phi": a0_sub < a_crit_outward,
        "above_csc_pi_over_5": a0_sub > a_crit_general,
        "psi0": psi0_sub, "v0": v0_sub, "mu": mu_sub, "kappa0": kappa0_sub,
        "beta_m0": beta_m0_sub, "beta_p0": beta_p0_sub,
        "fwd_chi_min": float(np.min(sol_fs.y[0, :])),
        "fwd_chi_max": float(np.max(sol_fs.y[0, :])),
        "fwd_chi_inf_exact": math.acos(math.tanh(alpha_b)),
        "fwd_stays_below_inj": bool(np.max(sol_fs.y[0, :]) + remaining_conformal_time(10.0, 0.0) < CHI_INJ),
        "fwd_max_res": fwd_sub["max_constraint_rel_res"],
        "bwd_t_end": float(sol_bs.t[-1]),
        "bwd_chi_max": float(np.max(sol_bs.y[0, :])),
        "bwd_breaches_inj": bool(np.max(sol_bs.y[0, :]) >= CHI_INJ),
        "bwd_max_res": bwd_sub["max_constraint_rel_res"],
        "exact_mu0_chi0_plus_chi_inf": math.acos(C_b / math.cosh(alpha_b)) + math.acos(math.tanh(alpha_b)),
    }

    # (b) Comoving-start (psi0 = 0) cases above t_crit_outward = 1.84273
    outward_cases = []
    for t0 in (2.0, 2.5, 3.5):
        a0, a_t0, _ = flrw_a_and_derivs(t0, 0.0)
        for chi0 in (0.05, 0.15):
            R0 = a0 * math.sin(chi0)
            mu = min(0.5 * (R0 ** 3), 0.5 * R0)
            Rd0 = a_t0 * math.sin(chi0)
            beta_m0 = -math.cos(chi0)
            beta_p0 = -math.sqrt(Rd0 ** 2 + 1.0 - mu / R0)
            kappa0 = (beta_m0 - beta_p0) / R0
            if kappa0 <= 0:
                continue
            fwd = integrate_shell_cosmic_time(0.0, mu, t0, chi0, 0.0, kappa0, -1.0, s_keep=-1.0, t_end=8.0, stop_after_inj=False)
            bwd = integrate_shell_cosmic_time(0.0, mu, t0, chi0, 0.0, kappa0, -1.0, s_keep=-1.0, t_end=-2.0, stop_after_inj=False)
            sol_f, sol_b = fwd["sol"], bwd["sol"]
            outward_cases.append({
                "t0": t0, "a0": a0, "chi0": chi0, "R0": R0, "mu": mu, "kappa0": kappa0,
                "fwd_chi_max": float(np.max(sol_f.y[0, :])),
                "fwd_stays_below_inj": bool(np.max(sol_f.y[0, :]) < CHI_INJ),
                "fwd_max_res": fwd["max_constraint_rel_res"],
                "bwd_t_end": float(sol_b.t[-1]),
                "bwd_chi_max": float(np.max(sol_b.y[0, :])),
                "bwd_breaches_inj": bool(np.max(sol_b.y[0, :]) >= CHI_INJ),
                "bwd_max_res": bwd["max_constraint_rel_res"],
            })

    return {
        "t_crit_general_inward": t_crit_general,
        "a_crit_general_csc_pi_over_5": a_crit_general,
        "t_crit_outward_psi_ge_0": t_crit_outward,
        "a_crit_outward_2phi": a_crit_outward,
        "sub_hubble_inward_counterexample": sub_hubble_demo,
        "outward_psi0_cases": outward_cases,
    }


def main():
    geom = verify_s3_istar_geometry()
    id_errs = verify_israel_darmois_identities()
    scan = scan_parameter_space()
    comoving = test_comoving_nec_violation()
    late = test_late_birth_window()

    fate_counts = {}
    for r in scan:
        f = r["fate"]
        fate_counts[f] = fate_counts.get(f, 0) + 1

    max_res_all = max((r["max_constraint_rel_res"] for r in scan if r["feasible"]), default=0.0)
    breach_ts = [r["t_breach"] for r in scan if r["feasible"] and r["t_breach"] is not None]

    summary = {
        "s3_istar_geometry": geom,
        "identity_verification_max_errors": id_errs,
        "bounce_shell_scan_summary": {
            "total_grid_points": len(scan),
            "feasible_at_bounce": sum(1 for r in scan if r["feasible"]),
            "fate_counts": fate_counts,
            "survived_global": fate_counts.get("survived_global", 0) + fate_counts.get("survived_ultrarel", 0),
            "solver_errors": fate_counts.get("solver_error", 0),
            "min_t_breach": min(breach_ts),
            "max_t_breach": max(breach_ts),
            "max_israel_constraint_rel_residual": max_res_all,
        },
        "comoving_nec_tests": comoving,
        "late_birth_tests": late,
    }

    out_path = OUT_DIR / "junction_verification.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print("=== BOUNCE SHELL SCAN (252 GRID POINTS, w in [-1, -0.9, -0.75, -0.5, 0, 0.5, 1]) ===")
    for k, v in summary["bounce_shell_scan_summary"].items():
        print(f"  {k}: {v}")
    assert summary["bounce_shell_scan_summary"]["survived_global"] == 0
    assert summary["bounce_shell_scan_summary"]["solver_errors"] == 0

    print("\n=== COMOVING NEC (INCLUDING POSITIVE BOUNCE PLATEAU & EINSTEIN-STATIC) ===")
    pos_bounce = [c for c in comoving["bounce_branch_cases"] if c["nec_at_bounce"] > 0]
    print(f"  Positive-NEC bounce cases: {len(pos_bounce)} / {len(comoving['bounce_branch_cases'])}")
    for c in pos_bounce[:4]:
        print(f"    gamma={c['gamma']:.6f}, chi0={c['chi0']}, mu_frac={c['mu_frac']}: "
              f"nec(0)={c['nec_at_bounce']:+.5f}, a_nec_zero={c['a_nec_zero']:.4f}, nec(t=8)={c['nec_at_t8']:+.4e}")
    print(f"  All 0<=gamma<1/4 violate NEC at late times: {all(c['nec_violated_globally_late'] for c in comoving['bounce_branch_cases'])}")
    print("  Einstein-Static (gamma=1/4) exact eternal comoving shells:")
    for c in comoving["einstein_static_gamma_quarter_cases"]:
        print(f"    chi0={c['chi0']}, mu_frac={c['mu_frac']}: w_eff={c['w_eff']:+.4f}, WEC+NEC+DEC={c['satisfies_WEC_NEC_DEC']}")

    print("\n=== LATE-BIRTH TRUE THRESHOLD (BC-001 VERIFIED) ===")
    s = late["sub_hubble_inward_counterexample"]
    print(f"  General psi in R: t_crit = {late['t_crit_general_inward']:.6f}, a_crit = csc(pi/5) = {late['a_crit_general_csc_pi_over_5']:.6f}")
    print(f"  Outward psi >= 0: t_crit = {late['t_crit_outward_psi_ge_0']:.6f}, a_crit = 2*phi = {late['a_crit_outward_2phi']:.6f}")
    print(f"  Sub-Hubble counterexample: t0={s['t0']}, a0={s['a0']:.4f} (<2phi:{s['below_2phi']}), R0={s['R0']:.4f} (<1:{s['sub_hubble_R0_lt_1']}), "
          f"fwd chi in [{s['fwd_chi_min']:.4f}, {s['fwd_chi_max']:.4f}] (<pi/10:{s['fwd_stays_below_inj']}, res={s['fwd_max_res']:.2e}), "
          f"bwd_breach={s['bwd_breaches_inj']} (chi_max={s['bwd_chi_max']:.4f}, res={s['bwd_max_res']:.2e})")


if __name__ == "__main__":
    main()
