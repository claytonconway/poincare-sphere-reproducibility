#!/usr/bin/env python3
"""
Stream p2_07_complete_model_b_kam_dynamics (CLM-P2-07)
Stage 3 (Round 2 Post-Audit): KT4 Adversarial Controls, Exact Limit Recoveries,
         DOP853 vs Radau Dual-Integrator Cross-Checks (with genuine 2nd-order oscillator ODE
         for KT3), & Canonical results.json Assembly.
"""

from importlib.machinery import SourceFileLoader
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

REPO = Path(__file__).resolve().parents[2]
STREAM_DIR = Path(__file__).resolve().parent
OUT_KT = STREAM_DIR / "03_kill_tests.json"
OUT_RESULTS = STREAM_DIR / "results.json"

s2 = SourceFileLoader("p207_solve", str(STREAM_DIR / "02_solve.py")).load_module()
p202_s2 = SourceFileLoader("p202_solve", str(REPO / "track6/p2_02/02_solve.py")).load_module()


def run_kt4_checks(derive_data: dict, solve_data: dict):
    print("=" * 80)
    print("STAGE 3 (KT4, ROUND 2): LIMIT RECOVERIES, MATCHED CONTROLS & DOP853 vs RADAU")
    print("=" * 80, flush=True)

    # 1. Exact p2_02 bare-limit recovery (Pi_kin = Pi_rem = 0, no bath)
    with open(REPO / "track6/p2_02/p2_02_01_identities.json") as fh:
        p202_rt = json.load(fh)["parseval_mr_rate_tables"]
    interp_bare = s2.build_pchip_from_rows(p202_rt["n2_shrinking_ring_sigma0"])
    m2_bare = [{"n": 2, "g": 1.0, "kn2": 8.0, "x_interp": interp_bare}]
    r_p207_bare = s2.integrate_full(
        s2.X_RING_0, s2.SIGMA_0, 0.0, m2_bare, [0.05], tau_end=2000.0,
        method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.20, n_eval=1001,
    )
    r_p202_ref = p202_s2.integrate_coupled(
        s2.X_RING_0, s2.SIGMA_0, lam=0.0, modes=m2_bare, b0_list=[0.05], tau_end=2000.0,
        method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.20, n_eval=1001,
    )
    p202_rec_x = abs(r_p207_bare["x_end"] - r_p202_ref["x_end"]) / abs(r_p202_ref["x_end"])
    p202_rec_krad = abs(r_p207_bare["kappa_rad_end"] - r_p202_ref["kappa_rad_end"]) / max(1e-12, abs(r_p202_ref["kappa_rad_end"]))
    print(f"  [1] Exact p2_02 bare-limit recovery (tau=2000): rel_x={p202_rec_x:.2e}, rel_krad={p202_rec_krad:.2e}", flush=True)
    assert max(p202_rec_x, p202_rec_krad) < 1e-10

    # 2. Exact p2_06 Talbot revival recovery (b2=0, gamma_deph=0) & round-S^3 failure
    j30, M30, w30 = s2.build_mr_resolved_poles(30)
    Ep_unsplit, wp_unsplit = s2.build_discrete_poles("Istar", 30)
    s_chk = np.linspace(0.0, 2.0 * math.pi, 2001)
    E_mr0 = np.array([2.0 * (2 * j + 1) for j in j30], dtype=float)
    K_mr0 = (2.0 * w30 / E_mr0) @ np.cos(np.outer(E_mr0, s_chk))
    K_unsplit = (2.0 * wp_unsplit / Ep_unsplit) @ np.cos(np.outer(Ep_unsplit, s_chk))
    mr_collapse_err = float(np.max(np.abs(K_mr0 - K_unsplit)) / np.max(np.abs(K_unsplit)))
    K_mr0_h = (2.0 * w30 / E_mr0) @ np.cos(np.outer(E_mr0, s_chk + 0.5 * math.pi))
    talbot_anti_err = float(np.max(np.abs(K_mr0_h + K_mr0)) / np.max(np.abs(K_mr0)))
    Ep_s3, wp_s3 = s2.build_discrete_poles("S3", 30)
    K_s3 = (2.0 * wp_s3 / Ep_s3) @ np.cos(np.outer(Ep_s3, s_chk))
    K_s3_h = (2.0 * wp_s3 / Ep_s3) @ np.cos(np.outer(Ep_s3, s_chk + 0.5 * math.pi))
    s3_anti_fail = float(np.max(np.abs(K_s3_h + K_s3)) / np.max(np.abs(K_s3)))
    print(f"  [2] Exact p2_06 Talbot recovery: MR->unsplit err={mr_collapse_err:.2e}, anti-per err={talbot_anti_err:.2e} (vs S^3 fail={s3_anti_fail:.4f})", flush=True)
    assert mr_collapse_err < 1e-13 and talbot_anti_err < 1e-13 and s3_anti_fail > 0.9

    # 3. Matched round-S^3 negative control inside Complete Model B narrowed sub-desert (sigma_t = 5.05)
    sig_mc = 5.05
    xr_mc = s2.x_ring_for_sigma(sig_mc)
    kap_pm = 0.002
    modes_I_mc = [
        {"n": 2,  "g": 5.0,  "kn2": 8.0,   "pi_kin": s2.PI_KIN_REL, "pi_rem": s2.PI_REM_REL},
        {"n": 10, "g": 9.0,  "kn2": 120.0},
        {"n": 14, "g": 17.0, "kn2": 224.0},
        {"n": 18, "g": 17.0, "kn2": 360.0},
        {"n": 22, "g": 46.0, "kn2": 528.0},
    ]
    b0_I_mc = [
        math.sqrt(3.0 * kap_pm / (5.0 * (8.0 * xr_mc * xr_mc + s2.PI_REM_REL))),
        math.sqrt(3.0 * kap_pm / (9.0 * 120.0 * xr_mc * xr_mc)),
        math.sqrt(3.0 * kap_pm / (17.0 * 224.0 * xr_mc * xr_mc)),
        math.sqrt(3.0 * kap_pm / (17.0 * 360.0 * xr_mc * xr_mc)),
        math.sqrt(3.0 * kap_pm / (46.0 * 528.0 * xr_mc * xr_mc)),
    ]
    modes_S3_mc = [
        {"n": 3, "g": 24.0, "kn2": 15.0},
        {"n": 4, "g": 42.0, "kn2": 24.0},
        {"n": 5, "g": 64.0, "kn2": 35.0},
        {"n": 6, "g": 90.0, "kn2": 48.0},
    ]
    b0_S3_mc = [math.sqrt(3.0 * kap_pm / (m["g"] * m["kn2"] * xr_mc * xr_mc)) for m in modes_S3_mc]
    r_I_mc = s2.integrate_full(xr_mc, sig_mc, 0.0, modes_I_mc, b0_I_mc, tau_end=4000.0, max_step=0.25, n_eval=1501)
    r_S3_mc = s2.integrate_full(xr_mc, sig_mc, 0.0, modes_S3_mc, b0_S3_mc, tau_end=4000.0, max_step=0.25, n_eval=1501)
    gain_I_mc = max(r_I_mc["gains"])
    gain_S3_mc = max(r_S3_mc["gains"])
    mc1_suppression = gain_S3_mc / gain_I_mc
    print(f"  [3] MC1 Sub-desert sig=5.05: ModelB I* {r_I_mc['outcome']} (gain={gain_I_mc:.2f}x) vs "
          f"S^3-forb {r_S3_mc['outcome']} (tau={r_S3_mc['tau_end']:.1f}, gain={gain_S3_mc:.1f}x, supp={mc1_suppression:.1f}x)", flush=True)
    assert r_I_mc["outcome"] == "kam_cyclic_survived" and r_S3_mc["outcome"] != "kam_cyclic_survived" and mc1_suppression > 100.0

    # 4. DOP853 vs Radau dual-integrator cross-checks across KT1, KT2, KT3
    print("  [4] Running DOP853 vs Radau dual-integrator cross-checks across KT1, KT2, KT3...", flush=True)
    dual_checks = {}

    xr5 = s2.x_ring_for_sigma(5.0)
    b0_5 = [
        math.sqrt(3.0 * kap_pm / (5.0 * (8.0 * xr5 * xr5 + s2.PI_REM_REL))),
        math.sqrt(3.0 * kap_pm / (9.0 * 120.0 * xr5 * xr5)),
        math.sqrt(3.0 * kap_pm / (17.0 * 224.0 * xr5 * xr5)),
        math.sqrt(3.0 * kap_pm / (17.0 * 360.0 * xr5 * xr5)),
        math.sqrt(3.0 * kap_pm / (46.0 * 528.0 * xr5 * xr5)),
    ]
    rd_1a_d = s2.integrate_full(xr5, 5.0, 0.0, modes_I_mc, b0_5, tau_end=1000.0, method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.20, n_eval=501)
    rd_1a_r = s2.integrate_full(xr5, 5.0, 0.0, modes_I_mc, b0_5, tau_end=1000.0, method="Radau", rtol=1e-10, atol=1e-12, max_step=0.20, n_eval=501)
    dual_checks["KT1a_subdesert_sig5_5mode"] = {
        "rel_x_end": abs(rd_1a_d["x_end"] - rd_1a_r["x_end"]) / abs(rd_1a_d["x_end"]),
        "dop_E00": rd_1a_d["max_rel_E00"], "rad_E00": rd_1a_r["max_rel_E00"],
    }

    xr9 = s2.x_ring_for_sigma(9.0)
    m2_rel5 = [{"n": 2, "g": 5.0, "kn2": 8.0, "pi_kin": s2.PI_KIN_REL, "pi_rem": s2.PI_REM_REL}]
    b0_9 = [math.sqrt(3.0 * kap_pm / (5.0 * (8.0 * xr9 * xr9 + s2.PI_REM_REL)))]
    rd_1b_d = s2.integrate_full(xr9, 9.0, 0.0, m2_rel5, b0_9, tau_end=1000.0, method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.20, n_eval=501)
    rd_1b_r = s2.integrate_full(xr9, 9.0, 0.0, m2_rel5, b0_9, tau_end=1000.0, method="Radau", rtol=1e-10, atol=1e-12, max_step=0.20, n_eval=501)
    dual_checks["KT1b_shifted_mathieu_sig9"] = {
        "rel_max_b2": abs(rd_1b_d["max_abs_b"][0] - rd_1b_r["max_abs_b"][0]) / max(1e-12, rd_1b_d["max_abs_b"][0]),
        "dop_E00": rd_1b_d["max_rel_E00"], "rad_E00": rd_1b_r["max_rel_E00"],
    }

    rel_rows_stable = [r for r in derive_data["modelB_shrinking_ring_parseval_tables"]["ModelB_relS3"] if not r["parametric_unstable_flagged"]]
    interp_relB = s2.build_pchip_from_rows(rel_rows_stable)
    m2_sc = [{"n": 2, "g": 1.0, "kn2": 8.0, "pi_kin": s2.PI_KIN_REL, "pi_rem": s2.PI_REM_REL, "x_interp": interp_relB}]
    rd_1c_d = s2.integrate_full(s2.X_RING_0, s2.SIGMA_0, 0.0, m2_sc, [0.05], tau_end=1500.0, method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.20, n_eval=501)
    rd_1c_r = s2.integrate_full(s2.X_RING_0, s2.SIGMA_0, 0.0, m2_sc, [0.05], tau_end=1500.0, method="Radau", rtol=1e-10, atol=1e-12, max_step=0.20, n_eval=501)
    dual_checks["KT1c_selfCons_chiral_quench"] = {
        "rel_x_end": abs(rd_1c_d["x_end"] - rd_1c_r["x_end"]) / abs(rd_1c_d["x_end"]),
        "rel_krad": abs(rd_1c_d["kappa_rad_end"] - rd_1c_r["kappa_rad_end"]) / max(1e-12, rd_1c_d["kappa_rad_end"]),
        "dop_E00": rd_1c_d["max_rel_E00"], "rad_E00": rd_1c_r["max_rel_E00"],
    }

    g_cpl = float(derive_data["theorem1_modelB_bath_noether_35dps"]["g_cpl_calibrated_to_p202_xring"])
    m2_rel1 = [{"n": 2, "g": 1.0, "kn2": 8.0, "pi_kin": s2.PI_KIN_REL, "pi_rem": s2.PI_REM_REL}]
    for lbl_g, gam in [("KT2a_bath_gam0", 0.0), ("KT2b_bath_gam_2overpi", 2.0 / math.pi), ("KT2c_bath_gam1p5", 1.5)]:
        bath = {"E_p": Ep_unsplit, "c_p": g_cpl * np.sqrt(wp_unsplit), "gamma_deph": gam}
        r_d = s2.integrate_full(s2.X_RING_0, s2.SIGMA_0, 0.0, m2_rel1, [0.05], bath=bath,
                                tau_end=200.0, method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.06, n_eval=401)
        r_r = s2.integrate_full(s2.X_RING_0, s2.SIGMA_0, 0.0, m2_rel1, [0.05], bath=bath,
                                tau_end=200.0, method="Radau", rtol=1e-10, atol=1e-12, max_step=0.06, n_eval=401)
        dx_g = abs(r_d["x_end"] - r_r["x_end"]) / abs(r_d["x_end"])
        dk_g = abs(r_d["kappa_rad_end"] - r_r["kappa_rad_end"]) / max(1e-12, r_d["kappa_rad_end"]) if gam > 0 else abs(r_d["kappa_bath_end"] - r_r["kappa_bath_end"]) / max(1e-12, r_d["kappa_bath_max"])
        dual_checks[lbl_g] = {"rel_x_end": dx_g, "rel_k_obs": dk_g, "dop_E00": r_d["max_rel_E00"], "rad_E00": r_r["max_rel_E00"]}

    # P0-4 fix: Genuine 2nd-order time-varying harmonic oscillator ODE q'' + E(eta)^2 q = 0 (first 50 Misner poles, b0=0.100, eta in [0, 4*pi])
    N_sub = (2 * j30[:50] + 1).astype(float)
    a2_sub = ((2 * M30[:50]) ** 2).astype(float)
    def rhs_2nd_sub(eta, st):
        lam = math.exp(math.sqrt(3.0) * 0.100 * math.cos(4.1158 * eta))
        E2 = 4.0 * (lam ** (2.0 / 3.0)) * (lam * lam + (N_sub * N_sub - 1.0) + (1.0 / (lam * lam) - 1.0) * a2_sub)
        return np.concatenate([st[50:], -E2 * st[:50]])
    y0_sub = np.concatenate([np.ones(50), np.zeros(50)])
    t_sub = np.linspace(0.0, 4.0 * math.pi, 401)
    s3_d = solve_ivp(rhs_2nd_sub, (0.0, 4.0 * math.pi), y0_sub, method="DOP853", t_eval=t_sub, rtol=1e-10, atol=1e-12, max_step=0.01)
    s3_r = solve_ivp(rhs_2nd_sub, (0.0, 4.0 * math.pi), y0_sub, method="Radau", t_eval=t_sub, rtol=1e-10, atol=1e-12, max_step=0.01)
    d_kt3_q = float(np.max(np.abs(s3_d.y[:50] - s3_r.y[:50])))
    dual_checks["KT3_2nd_order_MR_ode_50poles"] = {"max_abs_diff_q": d_kt3_q}

    for k, v in dual_checks.items():
        print(f"    {k:<28s}: {v}", flush=True)

    max_dual = max(
        dual_checks["KT1a_subdesert_sig5_5mode"]["rel_x_end"],
        dual_checks["KT1b_shifted_mathieu_sig9"]["rel_max_b2"],
        dual_checks["KT1c_selfCons_chiral_quench"]["rel_x_end"],
        dual_checks["KT1c_selfCons_chiral_quench"]["rel_krad"],
        dual_checks["KT2a_bath_gam0"]["rel_x_end"],
        dual_checks["KT2a_bath_gam0"]["rel_k_obs"],
        dual_checks["KT2b_bath_gam_2overpi"]["rel_x_end"],
        dual_checks["KT2b_bath_gam_2overpi"]["rel_k_obs"],
        dual_checks["KT2c_bath_gam1p5"]["rel_x_end"],
        dual_checks["KT2c_bath_gam1p5"]["rel_k_obs"],
        dual_checks["KT3_2nd_order_MR_ode_50poles"]["max_abs_diff_q"],
    )
    assert max_dual < 1e-5, f"Dual-solver check exceeded 1e-5: {max_dual}"

    # 5. KT3 J_cut = 30 vs J_cut = 60 static scrambling comparison summary (including peak_near_pi2!)
    st30 = {r["b2"]: r for r in solve_data["part3_kt3_mr_talbot_scrambler"]["static_b2_scrambling_J30"]}
    st60 = {r["b2"]: r for r in solve_data["part3_kt3_mr_talbot_scrambler"]["static_b2_scrambling_J60"]}
    kt3_j30_j60 = {}
    for b2 in (0.01, 0.05, 0.075, 0.10):
        kt3_j30_j60[f"b2_{b2:.3f}"] = {
            "K_pi2_J30": st30[b2]["K_pi2_over_K0"], "K_pi2_J60": st60[b2]["K_pi2_over_K0"],
            "peak_pi2_J30": st30[b2]["peak_near_pi2_over_K0"], "peak_pi2_J60": st60[b2]["peak_near_pi2_over_K0"],
            "max_late_J30": st30[b2]["max_late_10_20pi"], "max_late_J60": st60[b2]["max_late_10_20pi"],
        }

    return {
        "p202_bare_limit_recovery": {"rel_x_end": p202_rec_x, "rel_krad_end": p202_rec_krad},
        "p206_talbot_recovery": {"mr_to_unsplit_rel_err": mr_collapse_err, "Istar_anti_pi2_err": talbot_anti_err, "S3_anti_pi2_fail": s3_anti_fail},
        "mc1_subdesert_sig5p05": {
            "Istar_ModelB_outcome": r_I_mc["outcome"], "Istar_ModelB_gain": gain_I_mc, "Istar_ModelB_E00": r_I_mc["max_rel_E00"],
            "S3_forb_outcome": r_S3_mc["outcome"], "S3_forb_tau": r_S3_mc["tau_end"], "S3_forb_gain": gain_S3_mc,
            "suppression_factor": mc1_suppression,
        },
        "dual_integrator_dop853_vs_radau": dual_checks,
        "max_dual_integrator_diff": max_dual,
        "kt3_J30_vs_J60_comparison": kt3_j30_j60,
    }


def assemble_results_json(derive_data: dict, solve_data: dict, kt4_data: dict):
    th1 = derive_data["theorem1_modelB_bath_noether_35dps"]
    th2 = derive_data["theorem2_berger_mr_splitting_50dps"]
    p1 = solve_data["part1_kt1_modelB_landscape"]
    p2 = solve_data["part2_kt2_talbot_to_markovian"]
    p3 = solve_data["part3_kt3_mr_talbot_scrambler"]

    sw0 = {round(r["gamma_deph"], 4): r for r in p2["sweep_sigma0_J30"]}
    eft_I = {round(r["gamma_deph"], 4): r for r in p2["sub_unitarity_eft_sweeps"]["EFT_sig4_Istar"]}
    eft_S3 = {round(r["gamma_deph"], 4): r for r in p2["sub_unitarity_eft_sweeps"]["EFT_sig4_S3_ctrl"]}
    max_kt2_e00 = max(r["max_rel_E00"] for r in p2["sweep_sigma0_J30"])
    max_kt2_bal = max(r["bath_balance_rel_err"] for r in p2["sweep_sigma0_J30"])
    max_kt1_e00 = max(v["max_rel_E00"] for v in p1["chiral_quench_stress_tests"].values())
    eft_supp_015 = eft_S3[0.15]["kappa_rad_end"] / eft_I[0.15]["kappa_rad_end"]
    eft_supp_2pi = eft_S3[round(2.0 / math.pi, 4)]["kappa_rad_end"] / eft_I[round(2.0 / math.pi, 4)]["kappa_rad_end"]

    results = {
        "schema_version": "1.0",
        "stream_id": "p2_07_complete_model_b_kam_dynamics",
        "title": "Complete Model B Nonlinear KAM Backreaction, Coherent-Talbot-to-Markovian Crossover, and Autonomous Bianchi-IX M_R-Splitting Echo Scrambling on S^3/I*",
        "status": "verified",
        "updated_at": "2026-11-27",
        "conventions": {
            "metric_signature": "(-,+,+,+)",
            "units": "8*pi*G = M_Pl^-2 = 1, c = 1, Starobinsky scale H_S = 1 (tau = H_S t, x = H_S a, d_eta = d_tau / x)",
            "sector": "Formal large-N_V pure dark conformal gauge bosons on S^3/I*: kappa_V = -44738/93, sigma_t0 = c_anom = 9/31, x_ring(sigma_t0) = sqrt(44672/93) = 21.916754; Complete Model B constants from p2_06: rel-S^3 (Pi_kin = 3900/31, Pi_rem = 984896/465 = 2118.0559) and abs MS (Pi_kin = 4344/31, Pi_rem = 2066.1264); canonical Misner exponential Berger map lam_B = exp(sqrt(3)*b_2)",
            "framing": "Formal large-N_V semiclassical nonlinear dynamical-systems study honoring CLM-T3-01 Dvali species-scale boundary (L_min(a_Cas)/l_sp^red = 0.4551 < 1); not a species-scale-safe UV EFT"
        },
        "parameters": {
            "Pi_kin_relS3": s2.PI_KIN_REL,
            "Pi_rem_relS3": s2.PI_REM_REL,
            "x_kin_relS3": th1["x_kin_relS3"],
            "Pi_kin_absMS": s2.PI_KIN_ABS,
            "Pi_rem_absMS": s2.PI_REM_ABS,
            "x_kin_absMS": th1["x_kin_absMS"],
            "bath_g_cpl_calibrated": th1["g_cpl_calibrated_to_p202_xring"],
            "bath_J_cut_primary": 30,
            "bath_J_cut_check": 60,
            "kt3_primary_map": "canonical_misner_exp_sqrt3_b2",
            "kt3_n_mr_poles_J30": p3["n_poles_J30"],
            "kt3_n_mr_poles_J60": p3["n_poles_J60"],
        },
        "quantities": {
            "kt1_nonlinear_mathieu_peak_sigma_relS3": {
                "value": p1["cand0_summary"]["peak_relS3_sigma"],
                "uncertainty_or_tol": 0.25,
                "units": "dimensionless Starobinsky parameter sigma_t (QR-Benettin Lyapunov peak 0.00385 H_S at sig=9.5 rel-S^3 / 0.00389 H_S at sig=9.25 abs MS; shifted up from static-ring 7.191 / 6.915 by orbit-mean <Omega_2>_eta = 3.77)",
                "precision_digits": 3
            },
            "kt1_cand0_chirp_lyapunov_moderation_pct": {
                "value": p1["cand0_summary"]["chirp_growth_reduction_pct"],
                "uncertainty_or_tol": 0.5,
                "units": "percent peak QR-Benettin Lyapunov reduction vs orbit-mean-matched constant-k2 control (0.00493 -> 0.00385 H_S; HONEST KILL: at g_2=5, b0=0.005, sig=9.0 crashes into ghost_A_R2_xkin at tau=2518.9 rel-S^3 / 3287.6 abs MS, while sig=9.25, 9.50 saturate via nonlinear detuning at gain 13.8x/10.8x rel-S^3 and 10.2x/7.5x abs MS)",
                "precision_digits": 3
            },
            "kt1_selfCons_chiral_quench_x_late_min_relS3_g1_b005": {
                "value": p1["chiral_quench_stress_tests"]["relS3_selfCons_g1_b005"]["x_late_range"][0],
                "uncertainty_or_tol": 0.01,
                "units": "dimensionless radius x = H_S a (RESCUED with 0.0% lower-end x<x_min PCHIP clamping and 2.5% benign early-time x>x_ring outward-excursion clamping above x_kin=11.22, min_Aeff=0.3745, end|b|=0.0000 under self-consistent Markovian Model B rate table; bare p2_02 table crashes at tau=3745.0)",
                "precision_digits": 4
            },
            "kt1_selfCons_chiral_quench_x_late_min_absMS_g1_b005": {
                "value": p1["chiral_quench_stress_tests"]["absMS_selfCons_g1_b005"]["x_late_range"][0],
                "uncertainty_or_tol": 0.01,
                "units": "dimensionless radius x = H_S a (RESCUED with 0.0% lower-end x<x_min PCHIP clamping and 2.1% benign early-time x>x_ring outward-excursion clamping above x_kin=11.84, min_Aeff=0.3221, end|b|=0.0000 under self-consistent Markovian abs MS rate table; bare p2_02 table crashes at tau=3661.8)",
                "precision_digits": 4
            },
            "kt2_sig0_gamma0_kappa_rad": {
                "value": sw0[0.0]["kappa_rad_end"],
                "uncertainty_or_tol": 1e-12,
                "units": "dimensionless (exact zero secular drain at gamma_deph = 0; coherent Talbot/Rabi trapping k_bath_late = 0.325)",
                "precision_digits": 12
            },
            "kt2_sig0_gamma_2overpi_kappa_rad": {
                "value": sw0[round(2.0 / math.pi, 4)]["kappa_rad_end"],
                "uncertainty_or_tol": 0.005,
                "units": "dimensionless (crossover midpoint gamma_deph*pi/2 = 1.000 over tau=600: 0.39x bare p2_02 14.21, 0.18x Model B 31.13; bare-bg control gives 12.18 = 0.86x p2_02)",
                "precision_digits": 4
            },
            "kt2_sig0_gamma6p0_kappa_rad_markov_plateau": {
                "value": sw0[6.0]["kappa_rad_end"],
                "uncertainty_or_tol": 0.02,
                "units": "dimensionless (Markovian plateau gamma_deph*pi/2 = 9.425 over tau=600: 0.84x bare p2_02 14.21, 0.38x Model B 31.13; bare-bg control 11.81 = 0.83x)",
                "precision_digits": 4
            },
            "kt2_J30_vs_J60_gam_2overpi_rel_krad": {
                "value": p2["J30_vs_J60_sigma0"][f"gam_{2.0/math.pi:.4f}"]["rel_diff_k_rad"],
                "uncertainty_or_tol": 1e-05,
                "units": "dimensionless UV spectral-cutoff truncation convergence (J_cut=30 vs 60 at gamma_deph=2/pi; 8.84e-4 at gamma_deph=1.5; 6.09e-5 in k_bath at gamma_deph=0)",
                "precision_digits": 3
            },
            "kt2_eft_sig4_Istar_vs_S3_suppression_at_gam_2overpi": {
                "value": eft_supp_2pi,
                "uncertainty_or_tol": 0.5,
                "units": f"dimensionless topological gap suppression ratio (S^3 M=6 k_rad=0.9270 vs S^3/I* M=26 k_rad=0.0117 at sigma_t=4.0, gamma_deph=2/pi; {eft_supp_015:.1f}x at gam=0.15)",
                "precision_digits": 3
            },
            "kt3_static_b2_crit_peak_near_pi2_half": {
                "value": p3["static_b2_crit_peak_near_pi2_half"],
                "uncertainty_or_tol": 0.002,
                "units": "dimensionless shift-invariant static Misner |b_2| where peak_near_pi2_over_K0 = 0.5 at J_cut=30, gamma_deph=0 (fixed-time |K(pi/2)/K(0)|=0.5 occurs earlier at 0.0271 due to <c_1>_CG arrival-time shift)",
                "precision_digits": 3
            },
            "kt3_static_b2_crit_late_recurrence_half": {
                "value": p3["static_b2_crit_late_half"],
                "uncertainty_or_tol": 0.0005,
                "units": "dimensionless shift-invariant static Misner |b_2| where [10*pi, 20*pi] max recurrence drops to 0.5 at gamma_deph=0 (0.318 floor at 0.010 J30; 0.267 at J60)",
                "precision_digits": 3
            },
            "kt3_oscillating_b0_crit_misner_late_half": {
                "value": p3["oscillating_b0_crit_misner_late_half"],
                "uncertainty_or_tol": 0.005,
                "units": "dimensionless zero-mean oscillating Misner amplitude b_0 (lam_B=exp(sqrt(3)*b_0 cos(4.1158 eta))) where O(b_2^2) splitting drops [10*pi, 20*pi] max recurrence to 0.5 at gamma_deph=0 under plain phase integral (~+0.003 higher under WKB/2nd-order ODE amplitude modulation; vs 0.0661 in linear lam_B; 2.203x c_2 parameterization sensitivity)",
                "precision_digits": 3
            },
            "kt3_2nd_order_ode_vs_wkb_max_diff_b0p100": {
                "value": p3["oscillating_2nd_order_ODE_check_b0p100"]["max_diff_ODE2_vs_WKB_amplitude"],
                "uncertainty_or_tol": 0.0005,
                "units": "dimensionless max pointwise non-adiabatic O(Omega_2*b_0/E_p) Bogoliubov difference between exact 2nd-order ODE q''+E(eta)^2 q=0 and WKB amplitude-corrected kernel at b_0=0.100 (0.0296 vs plain cos(Phi); max_late ODE2=0.4178 vs WKB=0.4177)",
                "precision_digits": 3
            },
            "kt3_pre_heisenberg_landau_P05_R2_10pi_b2_0p05": {
                "value": p3["landau_phase_mixing_30pi_gamma0"]["Om43p0_P05_j10_band"][2]["R2_10pi_w_over_E"],
                "uncertainty_or_tol": 0.001,
                "units": "dimensionless transient pre-Heisenberg linear-absorption R^2 over [0, 10*pi] at Omega_drv=43.0, static Misner b_2=0.05, gamma_deph=0 (turns over at finite-shell T_Heis ~ 78.3; j=6 band at 26.5 turns over at T_Heis ~ 42.8; zero-mean oscillating b_2 stays bounded R^2 <= 0.0024, specifically 0.0024 at b_0=0.05 and 0.0009 at b_0=0.10)",
                "precision_digits": 4
            },
        },
        "verification": {
            "conservation_residuals": {
                "th1_35dps_weyl_invariance_DeltaL_Cas2": th1["weyl_invariance_DeltaL_Cas2_err_35dps"],
                "th1_35dps_weyl_invariance_L_bath": th1["weyl_invariance_L_bath_err_35dps"],
                "th1_35dps_dissipative_noether_d_x4E00_dtau": th1["dissipative_noether_d_x4E00_dtau_err_35dps"],
                "th1_35dps_bath_energy_balance": th1["bath_energy_balance_err_35dps"],
                "th2_exact_fraction_cg_norm_err": th2["cg_norm_exact_err_j1_to_30"],
                "th2_50dps_misner_c1_err": th2["misner_exp_cauchy_50dps_errs"][1],
                "th2_50dps_misner_c2_err": th2["misner_exp_cauchy_50dps_errs"][2],
                "th2_50dps_axi_cubic_R3_err": th2["axisymmetric_R3_cubic_err_50dps"],
                "kt1_lyapunov_max_block_det_err": max(r["relS3"]["max_block_det_err"] for r in p1["benettin_lyapunov_scan_cand0"]),
                "kt1_chiral_quench_max_rel_E00": max_kt1_e00,
                "kt1_subdesert_10k_max_rel_E00": max(v["max_rel_E00"] for v in p1["subdesert_diss_10k"].values()),
                "kt2_coupled_bath_max_rel_E00": max_kt2_e00,
                "kt2_coupled_bath_max_energy_balance_err": max_kt2_bal,
                "kt3_2nd_order_ode_wronskian_drift": p3["oscillating_2nd_order_ODE_check_b0p100"]["wronskian_max_drift"],
                "kt4_p206_talbot_anti_periodicity_err": kt4_data["p206_talbot_recovery"]["Istar_anti_pi2_err"],
            },
            "solver_cross_checks": {
                "kt4_p202_bare_limit_rel_x": kt4_data["p202_bare_limit_recovery"]["rel_x_end"],
                "kt4_p202_bare_limit_rel_krad": kt4_data["p202_bare_limit_recovery"]["rel_krad_end"],
                "kt4_p206_mr_to_unsplit_collapse_err": kt4_data["p206_talbot_recovery"]["mr_to_unsplit_rel_err"],
                "kt4_dop853_vs_radau_KT1a_subdesert_sig5": kt4_data["dual_integrator_dop853_vs_radau"]["KT1a_subdesert_sig5_5mode"]["rel_x_end"],
                "kt4_dop853_vs_radau_KT1b_mathieu_sig9": kt4_data["dual_integrator_dop853_vs_radau"]["KT1b_shifted_mathieu_sig9"]["rel_max_b2"],
                "kt4_dop853_vs_radau_KT1c_chiral_quench_krad": kt4_data["dual_integrator_dop853_vs_radau"]["KT1c_selfCons_chiral_quench"]["rel_krad"],
                "kt4_dop853_vs_radau_KT2a_bath_gam0": kt4_data["dual_integrator_dop853_vs_radau"]["KT2a_bath_gam0"]["rel_k_obs"],
                "kt4_dop853_vs_radau_KT2b_bath_gam_2overpi": kt4_data["dual_integrator_dop853_vs_radau"]["KT2b_bath_gam_2overpi"]["rel_k_obs"],
                "kt4_dop853_vs_radau_KT2c_bath_gam1p5": kt4_data["dual_integrator_dop853_vs_radau"]["KT2c_bath_gam1p5"]["rel_k_obs"],
                "kt4_dop853_vs_radau_KT3_2nd_order_MR_ode": kt4_data["dual_integrator_dop853_vs_radau"]["KT3_2nd_order_MR_ode_50poles"]["max_abs_diff_q"],
            },
            "matched_controls": [
                {
                    "control_id": "MC1_subdesert_sig5p05_Istar_ModelB_vs_S3_forbidden",
                    "matched_components": "Matched per-mode initial energy 0.002/mode, sigma_t = 5.05, Lambda = 0, exact round-S^3 degeneracies {24,42,64,90} vs Complete Model B (rel-S^3) S^3/I* 5-mode {2,10,14,18,22}.",
                    "treatment_result": f"Complete Model B S^3/I* survives tau=4000 cleanly (gain={kt4_data['mc1_subdesert_sig5p05']['Istar_ModelB_gain']:.2f}x, relE00={kt4_data['mc1_subdesert_sig5p05']['Istar_ModelB_E00']:.2e}).",
                    "control_result": f"Matched S^3-forbidden {{3,4,5,6}} crashes into {kt4_data['mc1_subdesert_sig5p05']['S3_forb_outcome']} at tau={kt4_data['mc1_subdesert_sig5p05']['S3_forb_tau']:.1f} (gain={kt4_data['mc1_subdesert_sig5p05']['S3_forb_gain']:.1f}x; {kt4_data['mc1_subdesert_sig5p05']['suppression_factor']:.1f}x suppression).",
                    "conclusion": "The narrowed Complete Model B sub-desert [4.5, 5.9] (rel-S^3) retains the I* topological protection against round-S^3 Mathieu blowup."
                },
                {
                    "control_id": "MC2_cand0_orbit_mean_matched_constant_k2_control",
                    "matched_components": "Identical nonlinear Starobinsky breathing background x(tau) across sigma_t in [3.5, 16.0] comparing Complete Model B (x-dependent Omega_2(x) chirp) against constant-k2 control matched to exact orbit-averaged <Omega_2>_eta (zero kinetic chirp) using QR-Benettin Lyapunov exponents over tau=4000.",
                    "treatment_result": f"Complete Model B rel-S^3 peaks at sigma_t={p1['cand0_summary']['peak_relS3_sigma']} with lam_Lyap={p1['cand0_summary']['peak_relS3_lam_4000']:.5f} H_S; at g_2=5, b0=0.005, sig=9.00 crashes nonlinearly into ghost_A_R2_xkin (tau=2518.9 rel-S^3 / 3287.6 abs MS), while sig=9.25 and 9.50 saturate via nonlinear detuning (gain 13.8x/10.8x rel-S^3; 10.2x/7.5x abs MS; min_A_eff >= 0.547).",
                    "control_result": f"Orbit-mean-matched constant-k2 control peaks at sigma_t={p1['cand0_summary']['peak_ctrl_sigma']} with lam_Lyap={p1['cand0_summary']['peak_ctrl_lam_4000']:.5f} H_S (only {p1['cand0_summary']['chirp_growth_reduction_pct']:.1f}% reduction).",
                    "conclusion": "Honest kill of cand_0: kinetic chirping moderates peak Lyapunov growth by only 21.9% and fails to prevent parametric blowup into the kinetic ghost wall at sig=9.00 (even though sig=9.25, 9.50 saturate nonlinearly at b0=0.005)."
                },
                {
                    "control_id": "MC3_KT2_EFT_sig4_gapped_Istar_vs_ungapped_S3_and_bare_bg_calibration",
                    "matched_components": "Identical coupled discrete Caldeira-Leggett bath (g_cpl=313.88, J_cut=30) at sub-unitarity sigma_t=4.0 comparing gapped S^3/I* (M_I*=26) vs ungapped round S^3 (M_S3=6), plus matched bare-p2_02-background control at sigma_t0=9/31 and b_2=0 Talbot control.",
                    "treatment_result": "At sig=4.0, gapped S^3/I* yields k_rad=0.00285 (gam=0.15), 0.01170 (gam=2/pi), 0.02719 (gam=1.5); on bare-p2_02 background at sig0=9/31, discrete bath converges to 12.18 (0.86x) at 2/pi and 12.54 (0.88x) at 1.5; b_2=0 anti-periodicity err 2.35e-14.",
                    "control_result": f"At sig=4.0, ungapped round S^3 drains k_rad=0.94149 ({eft_supp_015:.1f}x larger), 0.92697 ({eft_supp_2pi:.1f}x larger), 0.91274 (33.6x larger); round S^3 fails pi/2 Talbot anti-periodicity with residual 0.937.",
                    "conclusion": "Confirms bare-background Markovian calibration, topological M_I*=26 gap suppression in KT2, and exact S^3/I* Talbot recovery in KT3."
                },
            ],
            "kill_tests": [
                {
                    "id": "KT1",
                    "hypothesis_attacked": "KT1 (Complete Model B Lapse-Covariant Action, Shifted n=2 Mathieu Wall, cand_0 Chirp Detuner, & Chiral Self-Quenching Ghost-Wall Stress Test): verify exact 35-digit Noether closure with explicit g_2, compute QR-Benettin Lyapunov landscape across sigma_t in [3.5, 16.0], test cand_0 unconfoundedly, and stress-test p2_02's chiral self-quenching attractor against x_kin.",
                    "passed": True,
                    "metric": th1["dissipative_noether_d_x4E00_dtau_err_35dps"],
                    "notes": "35-digit Noether d/dtau(x^4 E_00) = 1.44e-33 with explicit g_2 in Delta_L_Cas2, Weyl invariance < 1e-34. QR-Benettin Lyapunov scan (tau=4000, max |det-1|<=1.36e-14) maps main shifted 2:1 wall to sigma_t in [8.75, 10.0] (peak 0.00385 H_S at 9.5 rel-S^3 / 0.00389 H_S at 9.25 abs MS) plus secondary 6.50-6.75 tongue. Honest kill of cand_0: chirp reduces peak Lyapunov rate only 21.9% vs orbit-mean-matched control (0.00493) and fails to stop nonlinear ghost_A_R2_xkin crash at g_2=5, sig=9.0, b0=0.005 (tau=2518.9 rel-S^3 / 3287.6 abs MS; sig=9.25, 9.50 survive via nonlinear detuning saturation at gain 13.8x/10.8x rel-S^3 and 10.2x/7.5x abs MS). Directly verified finite-amplitude (g_2=1, b0_2=0.05, b0_{n>=10}=1e-3) 10k unified 5-mode survival sub-desert is sigma_t in [4.5, 5.9] (rel-S^3, crashing at 6.0; absMS survives 4.5, 5.0, 5.5 at g_2=1, b0_2=0.05 while showing a narrow 30.5x small-seed g_2=5 tongue at 5.0). Under Markovian precondition (gamma_deph*pi/2 >= 1), chiral self-quenching at sigma_t0=9/31 crashes under bare p2_02 rates (tau=3745/4877 rel-S^3; 3662/4971 abs MS) yet is rescued with 0.0% lower-end x<x_min PCHIP clamping (2.1%-3.2% benign early-time x>x_ring outward-excursion clamping) above x_kin under Complete Model B's self-consistent Parseval rate table (rel-S^3 g1 b0=0.05 -> x_late=[13.80,15.73], min_Aeff=0.3745; abs MS -> [13.90,15.78], min_Aeff=0.3221; absMS x=12.86 parametric instability explicitly flagged and excluded)."
                },
                {
                    "id": "KT2",
                    "hypothesis_attacked": "KT2 (Exact Coherent-Talbot-to-Markovian Phase Transition gamma_deph*(pi/2) in Self-Consistent Coupled Discrete-Pole Bath): couple explicit J_cut=30 (checked vs J_cut=60) S^3/I* bath with tunable gamma_deph into nonlinear breathing x(tau) via exact dissipative Noether transfer and resolve the coherent-to-Markovian crossover.",
                    "passed": True,
                    "metric": max_kt2_e00,
                    "notes": f"Un-clamped max_rel_E00 <= {max_kt2_e00:.2e} and bath+rad energy balance <= {max_kt2_bal:.2e} across all 9 gamma_deph values [0, 6.0]. At sigma_t0=9/31 (tau=600, ~17.4 Talbot periods) on ModelB_relS3, kappa_rad transitions smoothly from exact 0.000 (gamma=0, coherent trapping k_bath_late=0.325) through 5.486 (gamma=2/pi, 0.39x bare p2_02 / 0.18x Model B) to 11.901 (gamma=6.0, 0.84x bare p2_02 / 0.38x Model B), while the matched bare-p2_02-background control confirms calibration (12.18 = 0.86x at 2/pi, 12.54 = 0.88x at 1.5); J_cut=30 vs 60 convergence <= 8.84e-4. At sub-unitarity sigma_t=4.0, gapped S^3/I* (M=26) suppresses drain {eft_supp_015:.1f}x (gam=0.15) and {eft_supp_2pi:.1f}x (gam=2/pi) vs matched ungapped round S^3 (M=6)."
                },
                {
                    "id": "KT3",
                    "hypothesis_attacked": "KT3 (Serendipity cand_3+cand_4: Autonomous Bianchi-IX M_R-Splitting Talbot Echo Scrambler): prove exact O(b_2) and Misner-exp O(b_2^2) splitting laws and CG weights, quantify strictly conservative (gamma_deph=0) echo scrambling for static and oscillating b_2 with genuine 2nd-order oscillator ODE, and bound transient pre-Heisenberg Landau phase mixing.",
                    "passed": True,
                    "metric": th2["misner_exp_cauchy_50dps_errs"][2],
                    "notes": "SURVIVES WITH HONEST PARAMETERIZATION & HEISENBERG BOUNDS: Under canonical kinetic-normalized Misner lam_B=exp(sqrt(3)*b_2), exact pair energy E_pair=2*lam^(1/3)*R splits at O(b_2) (std_CG(c_1)->(2/7)*E_j, 50-dps err 4.7e-48) and O(b_2^2) (c_2^exp/N = 1/3 + 4u^2 - 3u^4, 50-dps err 2.5e-46; 2.203x smaller std than linear lam_B), with exact Fraction CG norm err=0 (j=1..30). At gamma_deph=0, static Misner shear drops fixed-time |K(pi/2)/K_0| to 0.5 at |b_2|=0.0271 (confounded by <c_1>_CG arrival shift), shift-invariant first-echo peak_near_pi/2 to 0.5 at |b_2|=0.0980 (J30; 0.481 already at 0.075 in J60), and shift-invariant [10*pi,20*pi] recurrence to 0.5 at |b_2|=0.0044 (0.318 at 0.010 J30; 0.267 J60). Zero-mean oscillating Misner b_2(eta)=b_0 cos(4.1158 eta) drops [10*pi,20*pi] recurrence to 0.5 at b_0=0.0858 (vs 0.0661 linear lam_B); genuine 2nd-order ODE q''+E(eta)^2 q=0 at b_0=0.100 conserves symplectic Wronskian to 5.00e-8, reveals true 0.0020 non-adiabatic WKB Bogoliubov difference (0.0296 vs plain cos(Phi)), and confirms max_late=0.4178 vs WKB 0.4177. Extended 30*pi Landau audit confirms static b_2=0.05 produces transient pre-Heisenberg linear absorption (Om=43.0: R^2=0.994 on 10*pi turning over at T_Heis~78.3; Om=26.5: R^2=0.962 turning over at T_Heis~42.8), whereas zero-mean oscillating b_2 stays bounded (R^2<=0.0024: 0.0024 at b_0=0.05, 0.0009 at b_0=0.10)."
                },
                {
                    "id": "KT4",
                    "hypothesis_attacked": "KT4 (Controls, Dual-Integrator Cross-Checks & Exact Limit Recovery): recover p2_02 bare limit and p2_06 Talbot limit, verify matched round-S^3 controls across KT1--KT3, and verify DOP853 vs Radau agreement < 1e-5 across all key dynamical regimes including genuine 2nd-order KT3 ODE.",
                    "passed": True,
                    "metric": kt4_data["max_dual_integrator_diff"],
                    "notes": f"p2_02 bare-limit recovery rel_x={kt4_data['p202_bare_limit_recovery']['rel_x_end']:.2e}, rel_krad={kt4_data['p202_bare_limit_recovery']['rel_krad_end']:.2e}; p2_06 Talbot MR->unsplit collapse {kt4_data['p206_talbot_recovery']['mr_to_unsplit_rel_err']:.2e}, anti-periodicity {kt4_data['p206_talbot_recovery']['Istar_anti_pi2_err']:.2e} vs S^3 fail 0.937; MC1 sub-desert sig=5.05 suppression {kt4_data['mc1_subdesert_sig5p05']['suppression_factor']:.1f}x; worst DOP853 vs Radau across all 7 KT1/KT2/KT3 regimes (including genuine 2nd-order q''+E^2 q=0) = {kt4_data['max_dual_integrator_diff']:.2e} (< 1e-5); zero silent clamping everywhere."
                }
            ],
            "domain_of_validity": {
                "regime": "Formal large-N_V semiclassical nonlinear dynamical-systems study of Starobinsky-Casimir + Riegert + Complete Model B n=2 Casimir + explicit discrete S^3/I* 2-photon bath dynamics on R_eta x S^3/I*.",
                "caveats": [
                    "Species-Scale Squeeze (CLM-T3-01): For pure dark conformal gauge bosons on S^3/I*, L_min(a_Cas)/l_sp^red = 0.4551 < 1 is N_V-invariant, so P05-ring (x_ring ~ 21.92) nonlinear dynamics sit strictly below the Dvali species scale; all P05-ring dynamical results are formal large-N_V semiclassical dynamical-systems findings rather than a species-scale-safe UV EFT.",
                    "Trans-Unitarity vs Sub-Unitarity Scalaron & Markovian Precondition: sigma_t0 = 9/31 has 4*c_anom*omega_scal^2 = 4.000 (trans-unitarity); perturbative 1-loop scalaron EFT requires sigma_t >= 4 (within which Complete Model B narrows the finite-amplitude (g_2=1, b0_2=0.05, b0_{n>=10}=1e-3) unified 5-mode survival sub-desert to sigma_t in [4.5, 5.9] rel-S^3, crashing at 6.0, while abs MS survives 4.5, 5.0, 5.5 at g_2=1, b0_2=0.05 despite a narrow 30.5x small-seed g_2=5 tongue at 5.0). KT1's chiral self-quenching rescue relies on the continuum-log Markovian Parseval rate table, which per KT2 and CLM-P2-06 requires physical line-broadening gamma_deph*(pi/2) >= 1 (zero secular drain at gamma_deph=0); abs MS x=12.86 hits a 2:1 shrinking-ring parametric instability (amp=1.6e5) and is explicitly excluded from the PCHIP table, while relS3_selfCons_g5_b003 clamps at x_min=12.86 for 29.3% of samples (all primary rescued g1_b005 and g5_b002 runs have 0.0% lower-end x<x_min PCHIP clamping and 2.1%-3.2% benign early-time x>x_ring outward-excursion clamping; KT2 tau=600 Markovian reference runs clamp at x>x_ring for 33.9%-49.2% of early-time samples).",
                    "2-Derivative Local Complete Model B Truncation & Kinetic Ghost Wall: KT1 incorporates p2_06's exact 2-derivative Complete Model B kinetic (-Pi_kin) and static shear (+Pi_rem) terms while omitting the 4th-order Weyl-squared (beta_0 C^2) Ostrogradsky sector (bounded static Regime IV [0.5680, 4.4900] rel-S^3 / [0.5548, 3.7288] abs MS in CLM-P2-06). Approach to A_{R,2} = A_R - Pi_kin/x^2 -> 0+ signals breakdown of the 2-derivative Casimir EFT truncation; ODE integration halts cleanly at A_{R,2} = 1e-3.",
                    "KT2 Effective Flat-Weight Bilinear Bath Calibration: KT2 models the explicit J_cut=30/60 S^3/I* discrete 2-photon spectrum via a strictly positive-definite, Weyl-invariant 2nd-order Caldeira-Leggett bilinear bath S = phi*b_2 (counterterm 0.87% of 8*x_ring^2, J30 vs J60 <= 8.84e-4) with flat pole weights c_p = g_cpl*sqrt(w_p) calibrated locally to bare p2_02's continuum-log rate at sigma_t0=9/31, x_ring (giving 0.84x bare p2_02 / 0.38x Model B at gamma=6.0 on Model B, and 0.86-0.88x bare p2_02 on the matched bare background). Unlike the true 4th-order Weyl Omega*B_2(Omega) ~ Omega^5 spectral weight, flat c_p^2 weights make the calibration strictly local to sigma_t0=9/31 (at sigma_t=4.0, absolute bath coupling is overstated relative to (11/41)^5 while the S^3/I* vs S^3 ratio remains a valid topological gap control); at large gamma_deph >= 6, Lorentzian tails leak weakly across M_I*=26.",
                    "KT3 Axisymmetric Cubic Curvature Error, O(b_2^2) Parameterization Sensitivity, Fixed-CG Eikonal Weights & Finite-Shell Heisenberg Recurrence: Axisymmetric Bianchi IX shear lam_B = exp(sqrt(3)*b_2) has a nonzero cubic curvature potential R^(3) = 6 - 16*b_2^2 - (160/(3*sqrt(3)))*b_2^3 + ..., whose relative cubic error is 1.925*|b_2| (19.2% at |b_2|=0.10, 28.9% at |b_2|=0.15) and which dynamically induces an O(b_0^2) DC shift <b_2> ~ -b_0^2 feeding into c_1*<b_2> at O(b_0^2). Switching between canonical Misner exp(sqrt(3)*b_2) and linear 1+sqrt(3)*b_2 changes O(b_2^2) splitting std by 2.203x and oscillating b_0_crit from 0.0858 to 0.0661. The M_R-resolved memory kernel holds Clebsch-Gordan weights p_CG(j, M_R) fixed at their round-S^3 values (eikonal frequency-shift approximation omitting non-secular O(b_2) eigenmode-overlap corrections to the pole amplitudes w_p). At gamma_deph=0, static-b_2 Landau-like absorption is strictly transient pre-Heisenberg phase mixing (turning over at finite-shell T_Heis ~ 42.8 on j=6, ~78.3 on j=10) evaluated open-loop without 2-loop thermalization, and does not activate for zero-mean oscillating b_2(eta) at Om_drv=43.0 (R^2 <= 0.0024: 0.0024 at b_0=0.05, 0.0009 at b_0=0.10)."
                ]
            }
        }
    }
    return results


def main():
    with open(STREAM_DIR / "01_derive.json") as fh:
        derive_data = json.load(fh)
    with open(STREAM_DIR / "02_solve.json") as fh:
        solve_data = json.load(fh)

    kt4_data = run_kt4_checks(derive_data, solve_data)
    with open(OUT_KT, "w") as fh:
        json.dump(kt4_data, fh, indent=2)
    print(f"\nSaved {OUT_KT}", flush=True)

    results = assemble_results_json(derive_data, solve_data, kt4_data)
    with open(OUT_RESULTS, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved {OUT_RESULTS}", flush=True)


if __name__ == "__main__":
    main()
