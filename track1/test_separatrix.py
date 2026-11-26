#!/usr/bin/env python3
"""
Stream p04_israel_darmois — Post-Audit Hardened Separatrix, Cutoff-Sensitivity,
w in (-1, -0.5), and Non-Barotropic Switching Verification (closes BC-003 & BC-006).
"""

import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from verify_junction_and_eom import flrw_a_and_derivs, flrw_rho_p_scaled, remaining_conformal_time

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHI_INJ = math.pi / 10.0


def run_separatrix_case(gamma, mu, chi_b, w_shell=-1.0, t_end=10.0, kappa_cut=1e-8):
    """Integrate bounce shell with strict sol.status checks and Radau stiff fallback near kappa->0."""
    a0, _, _ = flrw_a_and_derivs(0.0, gamma)
    R0 = a0 * math.sin(chi_b)
    beta_m0 = -math.cos(chi_b)
    beta_p0 = -math.sqrt(1.0 - mu / R0)
    kappa0 = (beta_m0 - beta_p0) / R0
    if kappa0 <= 0:
        return {"outcome": "invalid_kappa0", "kappa0": kappa0}

    def rhs(t, state):
        chi, psi, kappa, _tau = state
        psi_c = max(-25.0, min(25.0, psi))
        a, a_t, a_tt = flrw_a_and_derivs(t, gamma)
        H = a_t / a
        ch, sh, th = math.cosh(psi_c), math.sinh(psi_c), math.tanh(psi_c)
        sc, cc = math.sin(chi), math.cos(chi)
        R = max(1e-12, a * sc)
        Rd = ch * a_t * sc + sh * cc
        beta_m = -(ch * cc + sh * a_t * sc)
        beta_p = beta_m - kappa * R
        Rdd_rem = (ch ** 2) * a_tt * sc + H * ch * sh * cc - (sh ** 2) * sc / a
        coeff = -(kappa * R) / beta_p
        rem = (Rdd_rem + 0.5 * mu / (R ** 2)) / beta_p + H * sh
        psidot = ((1.0 + 2.0 * w_shell) * kappa - rem) / coeff
        rho_8, p_8, _ = flrw_rho_p_scaled(a, gamma)
        flux_8piG = -(rho_8 + p_8) * ch * sh
        kappadot = -2.0 * (1.0 + w_shell) * kappa * (Rd / R) - 0.5 * flux_8piG
        return [th / a, psidot / ch, kappadot / ch, 1.0 / ch]

    def ev_inj(t, state):
        return state[0] - CHI_INJ
    ev_inj.terminal = True
    ev_inj.direction = 1

    def ev_kappa_zero(t, state):
        return state[2] - kappa_cut
    ev_kappa_zero.terminal = True
    ev_kappa_zero.direction = -1

    def ev_psi_lo(t, state):
        return state[1] + 18.0
    ev_psi_lo.terminal = True
    ev_psi_lo.direction = -1

    def ev_psi_hi(t, state):
        return 18.0 - state[1]
    ev_psi_hi.terminal = True
    ev_psi_hi.direction = -1

    def ev_chi_zero(t, state):
        return state[0] - 1e-6
    ev_chi_zero.terminal = True
    ev_chi_zero.direction = -1

    def ev_mass_locus(t, state):
        # Exact algebraic WEC boundary: 2*G*m_-(t, chi) - mu = 0 (forces kappa -> 0+)
        chi = state[0]
        a, a_t, _ = flrw_a_and_derivs(t, gamma)
        twoGm_m = a * (1.0 + a_t ** 2) * (math.sin(chi) ** 3)
        return twoGm_m - mu
    ev_mass_locus.terminal = True
    ev_mass_locus.direction = -1

    events = [ev_inj, ev_kappa_zero, ev_psi_lo, ev_psi_hi, ev_chi_zero, ev_mass_locus]
    sol = solve_ivp(
        rhs, [0.0, t_end], [chi_b, 0.0, kappa0, 0.0],
        method="DOP853", rtol=1e-12, atol=1e-14,
        events=events, max_step=0.04,
    )
    # If DOP853 hit step-size failure right near stiff kappa -> 0+ collapse, retry with Radau
    if sol.status == -1 and not (sol.y[1, -1] < -5.0 and sol.y[2, -1] < 1e-7):
        sol = solve_ivp(
            rhs, [0.0, t_end], [chi_b, 0.0, kappa0, 0.0],
            method="Radau", rtol=1e-10, atol=1e-12,
            events=events, max_step=0.04,
        )

    max_res = 0.0
    for k in range(sol.y.shape[1]):
        t = sol.t[k]
        chi, psi, kappa, _ = sol.y[:, k]
        a, a_t, _ = flrw_a_and_derivs(t, gamma)
        ch, sh = math.cosh(psi), math.sinh(psi)
        R = a * math.sin(chi)
        Rd = ch * a_t * math.sin(chi) + sh * math.cos(chi)
        beta_m = -(ch * math.cos(chi) + sh * a_t * math.sin(chi))
        beta_p = beta_m - kappa * R
        bp2 = Rd ** 2 + 1.0 - mu / R
        res = abs(beta_p ** 2 - bp2) / max(1.0, abs(bp2))
        max_res = max(max_res, res)

    chi_f = float(sol.y[0, -1])
    psi_f = float(sol.y[1, -1])
    kappa_f = float(sol.y[2, -1])
    t_f = float(sol.t[-1])

    if len(sol.t_events[0]) > 0 or float(np.max(sol.y[0, :])) >= CHI_INJ:
        outcome = "breach_outward"
    elif (len(sol.t_events[1]) > 0 or len(sol.t_events[2]) > 0
          or len(sol.t_events[4]) > 0 or len(sol.t_events[5]) > 0
          or (sol.status == -1 and psi_f < -5.0 and kappa_f < 1e-7)):
        outcome = "collapse_inward_wec"
    elif len(sol.t_events[3]) > 0:
        eta_rem = remaining_conformal_time(t_f, gamma)
        outcome = "SURVIVED_ULTRA_REL" if (chi_f + eta_rem < CHI_INJ) else "breach_outward_ultrarel"
    elif sol.status == 0 and abs(t_f - t_end) < 1e-6:
        eta_rem = remaining_conformal_time(t_f, gamma)
        outcome = "SURVIVED_GLOBAL" if (float(np.max(sol.y[0, :])) + eta_rem < CHI_INJ) else "needs_longer_t_end"
    else:
        raise RuntimeError(f"Unclassified ODE termination: status={sol.status}, msg={sol.message}, "
                           f"t={t_f}, chi={chi_f}, psi={psi_f}, kappa={kappa_f}")

    # Diagnostic 2Gm_- - mu at termination
    af, af_t, _ = flrw_a_and_derivs(t_f, gamma)
    twoGm_minus_mu = af * (1.0 + af_t ** 2) * (math.sin(chi_f) ** 3) - mu

    return {
        "outcome": outcome,
        "sol_status": int(sol.status),
        "gamma": gamma, "mu": mu, "chi_b": chi_b, "w_shell": w_shell,
        "kappa_cut": kappa_cut, "kappa0": kappa0,
        "t_end": t_f, "tau_end": float(sol.y[3, -1]),
        "chi_min": float(np.min(sol.y[0, :])), "chi_max": float(np.max(sol.y[0, :])),
        "chi_final": chi_f, "psi_final": psi_f,
        "kappa_min": float(np.min(sol.y[2, :])), "kappa_final": kappa_f,
        "twoGm_minus_mu_final": twoGm_minus_mu,
        "max_constraint_rel_res": max_res,
    }


def search_separatrix(gamma, chi_b, w_shell=-1.0, kappa_cut=1e-8, n_bisect=45):
    """Coarse scan with full outcome histogram + high-precision bisection."""
    y_plus = math.sqrt(0.5 + math.sqrt(0.25 - gamma))
    mu_max = y_plus * (math.sin(chi_b) ** 3) * 0.9999

    mus = np.geomspace(1e-7 * mu_max, mu_max, 50)
    coarse_counts = {}
    lo_mu, hi_mu = None, None
    survived = []
    for mu in mus:
        r = run_separatrix_case(gamma, float(mu), chi_b, w_shell=w_shell, kappa_cut=kappa_cut)
        o = r["outcome"]
        coarse_counts[o] = coarse_counts.get(o, 0) + 1
        if o.startswith("SURVIVED"):
            survived.append(r)
        elif o == "collapse_inward_wec":
            lo_mu = float(mu)
        elif o.startswith("breach_outward") and hi_mu is None and lo_mu is not None:
            hi_mu = float(mu)

    bisect_trace = []
    if lo_mu is not None and hi_mu is not None:
        a, b = lo_mu, hi_mu
        for it in range(n_bisect):
            mid = 0.5 * (a + b)
            r = run_separatrix_case(gamma, mid, chi_b, w_shell=w_shell, kappa_cut=kappa_cut)
            bisect_trace.append({
                "it": it, "mu": mid, "outcome": r["outcome"],
                "t_end": r["t_end"], "chi_max": r["chi_max"],
                "kappa_min": r["kappa_min"], "res": r["max_constraint_rel_res"],
            })
            if r["outcome"].startswith("SURVIVED"):
                survived.append(r)
                break
            elif r["outcome"] == "collapse_inward_wec":
                a = mid
            else:
                b = mid

    return {
        "gamma": gamma, "chi_b": chi_b, "w_shell": w_shell, "kappa_cut": kappa_cut,
        "mu_inj_local_max": mu_max,
        "coarse_outcome_counts": coarse_counts,
        "has_separatrix": (lo_mu is not None and hi_mu is not None),
        "n_bisect_executed": len(bisect_trace),
        "mu_sep_est": 0.5 * (lo_mu + hi_mu) if not bisect_trace and lo_mu and hi_mu else (bisect_trace[-1]["mu"] if bisect_trace else None),
        "window_found": len(survived) > 0,
        "last_bisect": bisect_trace[-4:] if bisect_trace else [],
    }


def test_nonbarotropic_optimal_switch():
    """
    BC-006 non-barotropic optimal-control test:
    Hold shell strictly comoving (chi = chi0, psi = 0) throughout the entire positive-NEC
    radiation plateau (a in [y_+, a_NEC]), then switch to the NEC-saturating domain wall
    (w = -1, p_theta = -sigma) at the exact instant t_NEC where sigma + p_theta = 0!
    """
    results = []
    for gamma, label in [(0.247765741792, "k=2"), (0.249973959691, "k=3")]:
        y_plus = math.sqrt(0.5 + math.sqrt(0.25 - gamma))
        for chi0 in (0.05, 0.15, 0.28):
            mu = 0.1 * y_plus * (math.sin(chi0) ** 3)
            def nec_comoving(t):
                a, a_t, a_tt = flrw_a_and_derivs(t, gamma)
                R = a * math.sin(chi0)
                Rd = a_t * math.sin(chi0)
                Rdd = a_tt * math.sin(chi0)
                bp = math.sqrt(Rd ** 2 + 1.0 - mu / R)
                s4 = (bp - math.cos(chi0)) / R
                return s4 - (Rdd + 0.5 * mu / (R ** 2)) / bp
            assert nec_comoving(0.0) > 0
            t_nec = float(brentq(nec_comoving, 0.0, 6.0))
            a_nec, a_t_nec, _ = flrw_a_and_derivs(t_nec, gamma)
            R_nec = a_nec * math.sin(chi0)
            Rd_nec = a_t_nec * math.sin(chi0)
            k_nec = (math.sqrt(Rd_nec ** 2 + 1.0 - mu / R_nec) - math.cos(chi0)) / R_nec
            # Now release at t = t_nec with w = -1 (saturating NEC!)
            from verify_junction_and_eom import integrate_shell_cosmic_time
            out = integrate_shell_cosmic_time(gamma, mu, t_nec, chi0, 0.0, k_nec, -1.0,
                                              s_keep=-1.0, t_end=t_nec + 8.0, stop_after_inj=True, method="DOP853")
            results.append({
                "label": label, "gamma": gamma, "chi0": chi0, "mu": mu,
                "t_nec_switch": t_nec, "a_nec_switch": a_nec, "kappa_at_switch": k_nec,
                "eta_rem_at_switch": remaining_conformal_time(t_nec, gamma),
                "post_switch_fate": out["fate"],
                "t_breach": out["t_breach"],
                "dt_after_switch": (out["t_breach"] - t_nec) if out["t_breach"] is not None else None,
                "max_res": out["max_constraint_rel_res"],
            })
    return results


def main():
    # 1. Core separatrix bisections (including fixed chi_b = 0.28 and w = -0.9, -0.75!)
    sep_configs = [
        (0.20, 0.15, -1.0),
        (0.23, 0.15, -1.0),
        (0.24, 0.15, -1.0),
        (0.247765741792, 0.05, -1.0),
        (0.247765741792, 0.15, -1.0),
        (0.247765741792, 0.28, -1.0),   # Fixed BC-003!
        (0.249973959691, 0.15, -1.0),   # k=3 resonance
        (0.247765741792, 0.15, -0.9),   # New BC-006!
        (0.247765741792, 0.15, -0.75),  # New BC-006!
        (0.247765741792, 0.15, -0.5),   # Outward-only control (1+2w = 0)
        (0.247765741792, 0.15, 0.0),    # Outward-only control (1+2w > 0)
    ]
    sep_results = []
    for gamma, chi_b, w in sep_configs:
        res = search_separatrix(gamma, chi_b, w_shell=w, kappa_cut=1e-8)
        sep_results.append(res)
        print(f"g={gamma:.6f}, chi_b={chi_b:.2f}, w={w:+.2f} | sep={res['has_separatrix']} "
              f"(n_bis={res['n_bisect_executed']}) | coarse={res['coarse_outcome_counts']} | "
              f"mu_sep={res['mu_sep_est']} | win={res['window_found']}")

    # 2. Cutoff-sensitivity check across kappa_cut in {1e-8, 1e-10, 1e-12} (BC-003)
    print("\n=== CUTOFF SENSITIVITY CHECK (gamma_2, chi_b=0.15, w=-1) ===")
    cutoff_results = []
    for kc in (1e-8, 1e-10, 1e-12):
        r = search_separatrix(0.247765741792, 0.15, w_shell=-1.0, kappa_cut=kc, n_bisect=40)
        cutoff_results.append({
            "kappa_cut": kc, "mu_sep": r["mu_sep_est"],
            "window_found": r["window_found"], "last_bisect": r["last_bisect"][-2:],
        })
        print(f"  kappa_cut={kc:.0e}: mu_sep={r['mu_sep_est']:.12e}, win={r['window_found']}, "
              f"t_out={r['last_bisect'][-1]['t_end']:.4f}, res={r['last_bisect'][-1]['res']:.2e}")

    # 3. Non-barotropic optimal comoving -> w=-1 switch test (BC-006)
    print("\n=== NON-BAROTROPIC OPTIMAL SWITCH (COMOVING PLATEAU -> w=-1 AT t_NEC) ===")
    nonbaro = test_nonbarotropic_optimal_switch()
    for r in nonbaro:
        print(f"  {r['label']} chi0={r['chi0']:.2f}: t_NEC={r['t_nec_switch']:.4f}, a_NEC={r['a_nec_switch']:.4f}, "
              f"eta_rem={r['eta_rem_at_switch']:.4f}, fate={r['post_switch_fate']}, "
              f"t_breach={r['t_breach']:.4f} (+{r['dt_after_switch']:.4f}), res={r['max_res']:.2e}")

    payload = {
        "separatrix_bisections": [r for r in sep_results if r["has_separatrix"]],
        "outward_only_controls": [r for r in sep_results if not r["has_separatrix"]],
        "cutoff_sensitivity": cutoff_results,
        "nonbarotropic_optimal_switch": nonbaro,
    }
    (OUT_DIR / "separatrix_scan.json").write_text(json.dumps(payload, indent=2))
    assert all(not r["window_found"] for r in sep_results)
    assert all(r["post_switch_fate"] == "breach_outward" for r in nonbaro)
    print("\nALL SEPARATRIX, CUTOFF-SENSITIVITY, w in (-1,-0.5), AND NON-BAROTROPIC CHECKS PASSED (ZERO WINDOWS).")


if __name__ == "__main__":
    main()
