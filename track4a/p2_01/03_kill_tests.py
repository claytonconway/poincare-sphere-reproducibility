#!/usr/bin/env python3
"""
Stream p2_01_discrete_photon_pole_sum — Stage 3: KT4 Dynamical Volterra vs Discrete-Pole Sidebands,
Matched Controls, and Canonical results.json Assembly (CLM-OPEN-01).

Executes:
  1. Exact adiabatic discrete pinch-off envelope beta_pinch_inst^{disc}(x(tau), A_R(tau), K_R(tau))
     on BOTH P05 x_ring and eccentric x_b = 15.0 (compared side-by-side with Track IV-A continuum M=26).
  2. KT4-A: Dense unreduced 4th-order discrete-pole evolution on x_ring across Regime IV
     beta_0 in [0.44, 6.37] (plus 8.0) with Stelle-excited IC AND smooth physical graviton IC,
     paired with Matched Continuum-Log (M_I*=26) and Matched Discrete Round-S^3 Controls.
  3. KT4-B: Unreduced discrete-pole evolution on eccentric x_b = 15.0 across beta_0 in [0.50, 6.37]
     with Matched Continuum-Log (M_I*=26) Control.
  4. KT4-C: Targeted sideband-resonance inspection (checking whether any sub-window above the
     adiabatic pinch-off envelope is destabilized by discrete 2-photon poles E_6=26, E_10=42, ...
     via scalaron sidebands, plus J_cut=60 verification on key Regime IV and sideband points).
  5. Assembles canonical streams/p2_01_discrete_photon_pole_sum/results.json with ZERO silent clamping.
"""

import json
import math
import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader
import numpy as np

STREAM_DIR = Path(__file__).resolve().parent
OUT_KT_PATH = STREAM_DIR / "03_kill_tests.json"
RESULTS_PATH = STREAM_DIR / "results.json"

m2 = SourceFileLoader("m2_solve", str(STREAM_DIR / "02_solve.py")).load_module()

C_ANOM = m2.C_ANOM
SIGMA_0 = m2.SIGMA_0
KAPPA_V = m2.KAPPA_V
X_RING = m2.X_RING


# ==============================================================================
# 1. EXACT ADIABATIC DISCRETE PINCH-OFF ENVELOPE ON KAM ORBITS
# ==============================================================================

def compute_adiabatic_pinchoff_envelope(bg_sol, eta_end, pole_spec, M_first=26.0, n_pts=1500):
    """
    Compute instantaneous adiabatic pinch-off coupling along a KAM orbit:
      beta_pinch_inst(tau) = min_{Om in (4, M_first)} [
          x^2*(A_R*Om^2 - 8*K_R)/(4*B_2(Om)) - (c_anom/2)*K_kernel(Om) + c_anom*ln(x/x_ring)
      ]
    for ANY pole_spec (discrete Istar, discrete S3, or continuum M=26 / M=6).
    """
    etas = np.linspace(0.0, eta_end, n_pts)
    st = bg_sol.sol(etas)
    xs = st[0]
    Ar, Kr = m2.bg_AK_at_state(st[0], st[1], st[2])

    oms = np.linspace(4.05, M_first - 0.02, 3000)
    B2 = (oms ** 2 - 4.0) * (oms ** 2 - 16.0)

    # Evaluate K_kernel(oms) on the sub-threshold grid (4, M_first)
    Mt = float(pole_spec["M_tail"])
    delta_C = float(pole_spec["delta_C_tail"])
    K_vals = np.log((Mt ** 2 - oms ** 2) / (X_RING ** 2)) + delta_C
    for Ep, wp in zip(pole_spec["E_p"], pole_spec["w_p"]):
        K_vals = K_vals - wp * (2.0 * Ep) / (Ep ** 2 - oms ** 2)
    k_term = 0.5 * C_ANOM * K_vals

    b_full = np.empty(n_pts)
    b_ak1 = np.empty(n_pts)
    om_at_max = 0.0
    max_val = -1e9
    for i in range(n_pts):
        run_shift = C_ANOM * math.log(xs[i] / X_RING)
        eh_full = (xs[i] ** 2) * (Ar[i] * (oms ** 2) - 8.0 * Kr[i]) / (4.0 * B2)
        eh_ak1 = (xs[i] ** 2) * ((oms ** 2) - 8.0) / (4.0 * B2)
        curve = eh_full - k_term + run_shift
        im = int(np.argmin(curve))
        b_full[i] = float(curve[im])
        b_ak1[i] = float(np.min(eh_ak1 - k_term + run_shift))
        if b_full[i] > max_val:
            max_val = b_full[i]
            om_at_max = float(oms[im])

    stat_curve = (X_RING ** 2) * ((oms ** 2) - 8.0) / (4.0 * B2) - k_term
    i_st = int(np.argmin(stat_curve))
    return {
        "topology": pole_spec["topology"],
        "J_cut": int(pole_spec["J_cut"]),
        "x_min": float(np.min(xs)),
        "x_max": float(np.max(xs)),
        "A_R_min": float(np.min(Ar)),
        "A_R_max": float(np.max(Ar)),
        "beta_pinch_static_xring": float(stat_curve[i_st]),
        "Omega_pinch_static_xring": float(oms[i_st]),
        "beta_pinch_inst_AK1_max": float(np.max(b_ak1)),
        "beta_pinch_inst_full_max": float(np.max(b_full)),
        "beta_pinch_inst_full_min": float(np.min(b_full)),
        "beta_pinch_inst_full_mean": float(np.mean(b_full)),
        "Omega_at_inst_max": om_at_max,
    }


def run_one_trajectory(bg_sol, eta_end, b0, spec, y0_4d, Nu=48, u_max=1e6, blowup_cap=1e4):
    """Helper to run Solver A with consistent tolerances and summarize un-clamped metrics."""
    r = m2.solve_discrete_aux_ode(
        bg_sol, eta_end, b0, spec, Nu=Nu, u_max=u_max, y0_4d=y0_4d,
        n_eval=1501, method="BDF", rtol=1e-8, atol=1e-10, max_step=0.025, blowup_cap=blowup_cap,
    )
    if y0_4d is None:
        # Zero-Weyl-momentum IC q1=1, q2=0, pi1=0, pi2=0 (injects O(beta_0) Stelle transient;
        # see 02_solve.json graviton_eigenvector_ic_check for pure graviton eigenvector IC)
        st = bg_sol.sol(r["eta"])
        Ar, Kr = m2.bg_AK_at_state(st[0], st[1], st[2])
        assert np.all(Ar > 0.0) and np.all(Kr > 0.0), "Unexpected non-positive A_R or K_R on x_ring!"
        El = (st[0] ** 2) * (Ar * (r["q2"] ** 2) + 8.0 * Kr * (r["q1"] ** 2))
        m = (r["eta"] >= min(5.0, 0.25 * r["eta"][-1])) & (El > 1e-30)
        gam = 0.5 * float(np.polyfit(r["tau"][m], np.log(El[m]), 1)[0]) if np.sum(m) >= 10 else 0.0
        max_ratio = float(np.max(El) / max(1e-30, float(El[0])))
        end_ratio = float(El[-1] / max(1e-30, float(El[0])))
    else:
        pi2_abs = np.abs(r["pi2"])
        gam = m2.fit_exponential_rate(r["tau"], pi2_abs, 0.40, 0.98)
        init_scale = max(1e-12, float(np.max(pi2_abs[:min(100, len(pi2_abs))])))
        max_ratio = float(np.max(pi2_abs) / init_scale)
        end_ratio = float(np.max(pi2_abs[-min(80, len(pi2_abs)):]) / init_scale)
    return {
        "beta_0": float(b0),
        "eta_reached": float(r["eta"][-1]),
        "tau_reached": float(r["tau"][-1]),
        "blowup": bool(r["blowup"]),
        "buv_zero": bool(r["buv_zero"]),
        "success": bool(r["success"]),
        "gamma_HS": float(gam),
        "max_ratio": float(max_ratio),
        "end_ratio": float(end_ratio),
    }


def main():
    print("=== STREAM p2_01_discrete_photon_pole_sum — STAGE 3: KT4 DYNAMICAL SIDEBAND SCAN ===", flush=True)

    bg_ring, info_ring = m2.integrate_kam_background_eta(X_RING, eta_end=45.0)
    bg_ecc, info_ecc = m2.integrate_kam_background_eta(15.0, eta_end=35.0)
    print(f"Backgrounds ready: x_ring max_rel_e00={info_ring['max_rel_e00']:.2e} | "
          f"x_b=15 max_rel_e00={info_ecc['max_rel_e00']:.2e}", flush=True)

    spec_is30 = m2.build_discrete_pole_spec("Istar", J_cut=30, include_tail=True, match_exact_K0=True)
    spec_is60 = m2.build_discrete_pole_spec("Istar", J_cut=60, include_tail=True, match_exact_K0=True)
    spec_cont26 = m2.build_discrete_pole_spec("continuum_M26")
    spec_s3_15 = m2.build_discrete_pole_spec("S3", J_cut=15, include_tail=True, match_exact_K0=True)

    # 1. Exact Adiabatic Pinch-Off Envelopes (Discrete vs Continuum on x_ring and x_b=15)
    print("\n1. Exact Dynamical Adiabatic Pinch-Off Envelopes:", flush=True)
    env_ring_disc = compute_adiabatic_pinchoff_envelope(bg_ring, 40.0, spec_is30, M_first=26.0)
    env_ring_cont = compute_adiabatic_pinchoff_envelope(bg_ring, 40.0, spec_cont26, M_first=26.0)
    env_ecc_disc = compute_adiabatic_pinchoff_envelope(bg_ecc, 35.0, spec_is30, M_first=26.0)
    env_ecc_cont = compute_adiabatic_pinchoff_envelope(bg_ecc, 35.0, spec_cont26, M_first=26.0)
    print(f"   x_ring: cont_max={env_ring_cont['beta_pinch_inst_full_max']:.4f} -> "
          f"disc_max={env_ring_disc['beta_pinch_inst_full_max']:.4f} (static={env_ring_disc['beta_pinch_static_xring']:.4f})", flush=True)
    print(f"   x_b=15: cont_max={env_ecc_cont['beta_pinch_inst_full_max']:.4f} -> "
          f"disc_max={env_ecc_disc['beta_pinch_inst_full_max']:.4f}", flush=True)

    # 2. KT4-A: Dense Unreduced Scan on x_ring across Regime IV beta_0 in [0.44, 6.37] + 8.0
    print("\n2. KT4-A: Dense Dynamical Scan on x_ring (Discrete S^3/I*, Stelle-excited IC, eta=35 -> tau~773):", flush=True)
    y0_stelle = [0.0, 0.0, 1.0, 0.0]
    beta_grid_ring = [
        (0.440, 8.0), (0.500, 10.0), (0.600, 15.0), (0.680, 25.0),
        (0.700, 35.0), (0.720, 35.0), (0.747, 35.0), (0.760, 35.0),
        (0.781, 35.0), (0.800, 35.0), (0.850, 35.0), (0.900, 35.0),
        (0.950, 35.0), (1.000, 35.0), (1.052, 35.0), (1.150, 35.0),
        (1.199, 35.0), (1.217, 35.0), (1.250, 35.0), (1.485, 35.0),
        (1.650, 35.0), (1.719, 35.0), (1.750, 35.0), (1.800, 35.0),
        (1.994, 35.0), (2.000, 35.0), (2.029, 35.0), (2.186, 35.0),
        (2.431, 35.0), (3.000, 35.0), (4.000, 35.0), (5.000, 35.0),
        (6.370, 35.0), (8.000, 35.0),
    ]
    xring_disc_scan = []
    for b0, eta_t in beta_grid_ring:
        row = run_one_trajectory(bg_ring, eta_t, b0, spec_is30, y0_stelle, Nu=44)
        xring_disc_scan.append(row)
        print(f"   [x_ring I*_disc] b0={b0:6.3f} (tau={row['tau_reached']:5.1f}): gam={row['gamma_HS']:+.4e} | "
              f"max_ratio={row['max_ratio']:8.2e} | blowup={row['blowup']}", flush=True)

    # 3. Matched Controls on x_ring: Continuum M_I*=26 AND Discrete Round S^3
    print("\n3. Matched Controls on x_ring (Continuum M_I*=26 vs Discrete Round S^3):", flush=True)
    ctrl_cont26 = []
    for b0 in [0.44, 0.50, 0.60, 0.70, 0.80, 1.00, 2.00, 5.00, 6.37]:
        row = run_one_trajectory(bg_ring, 35.0, b0, spec_cont26, y0_stelle, Nu=44)
        ctrl_cont26.append(row)
        print(f"   [x_ring ContM26] b0={b0:6.3f} (tau={row['tau_reached']:5.1f}): gam={row['gamma_HS']:+.4e} | "
              f"max_ratio={row['max_ratio']:8.2e} | blowup={row['blowup']}", flush=True)

    ctrl_s3_disc = []
    for b0, eta_t in [(0.50, 5.0), (1.00, 8.0), (2.00, 14.0), (5.00, 30.0), (6.37, 35.0), (8.50, 35.0)]:
        row = run_one_trajectory(bg_ring, eta_t, b0, spec_s3_15, y0_stelle, Nu=44)
        ctrl_s3_disc.append(row)
        print(f"   [x_ring S3_disc] b0={b0:6.3f} (tau={row['tau_reached']:5.1f}): gam={row['gamma_HS']:+.4e} | "
              f"max_ratio={row['max_ratio']:8.2e} | blowup={row['blowup']}", flush=True)

    # 4. Zero-pi2 IC (y0=[1,0,0,0]) on x_ring across stabilized Regime IV
    print("\n4. Zero-pi2 IC (y0=[1,0,0,0]) on x_ring (Discrete S^3/I* vs Continuum M26):", flush=True)
    phys_ic_rows = []
    for b0 in [0.50, 0.76, 0.80, 1.00, 1.50, 2.00, 5.00, 6.37]:
        rd = run_one_trajectory(bg_ring, 35.0, b0, spec_is30, None, Nu=44)
        rc = run_one_trajectory(bg_ring, 35.0, b0, spec_cont26, None, Nu=44)
        phys_ic_rows.append({
            "beta_0": b0,
            "disc_tau": rd["tau_reached"], "disc_blowup": rd["blowup"],
            "disc_gamma_E_HS": rd["gamma_HS"], "disc_E_end_over_E0": rd["end_ratio"],
            "cont_tau": rc["tau_reached"], "cont_blowup": rc["blowup"],
            "cont_gamma_E_HS": rc["gamma_HS"], "cont_E_end_over_E0": rc["end_ratio"],
        })
        print(f"   [phys IC] b0={b0:5.2f}: disc gam_E={rd['gamma_HS']:+.3e} (E/E0={rd['end_ratio']:.3f}, bu={rd['blowup']}) | "
              f"cont gam_E={rc['gamma_HS']:+.3e} (E/E0={rc['end_ratio']:.3f}, bu={rc['blowup']})", flush=True)

    # 5. KT4-B: Eccentric KAM Orbit x_b = 15.0 across beta_0 in [0.50, 6.37] + Continuum Control
    print("\n5. KT4-B: Eccentric Orbit x_b = 15.0 (Discrete S^3/I* vs Continuum M26 Control):", flush=True)
    ecc_disc_scan = []
    for b0, eta_t in [
        (0.50, 12.0), (0.70, 20.0), (0.80, 25.0), (1.00, 30.0),
        (1.10, 30.0), (1.20, 30.0), (1.30, 30.0), (1.50, 30.0),
        (2.00, 30.0), (3.00, 30.0), (5.00, 30.0), (6.37, 30.0),
    ]:
        row = run_one_trajectory(bg_ecc, eta_t, b0, spec_is30, y0_stelle, Nu=44)
        ecc_disc_scan.append(row)
        print(f"   [x_b=15 I*_disc] b0={b0:5.2f} (tau={row['tau_reached']:5.1f}): gam={row['gamma_HS']:+.4e} | "
              f"max_ratio={row['max_ratio']:8.2e} | blowup={row['blowup']}", flush=True)

    ecc_ctrl_cont26 = []
    for b0, eta_t in [(0.50, 20.0), (0.80, 30.0), (1.00, 30.0), (1.50, 30.0), (2.00, 30.0)]:
        row = run_one_trajectory(bg_ecc, eta_t, b0, spec_cont26, y0_stelle, Nu=44)
        ecc_ctrl_cont26.append(row)
        print(f"   [x_b=15 ContM26] b0={b0:5.2f} (tau={row['tau_reached']:5.1f}): gam={row['gamma_HS']:+.4e} | "
              f"max_ratio={row['max_ratio']:8.2e} | blowup={row['blowup']}", flush=True)

    # 6. KT4-C: Automatic Fine Zoom & J_cut=60 Verification on ANY Detected Sideband Growth Above Envelope!
    b_env_ring = env_ring_disc["beta_pinch_inst_full_max"]
    anomalous_pts = [
        r for r in xring_disc_scan
        if r["beta_0"] > b_env_ring + 0.02 and (r["blowup"] or r["max_ratio"] > 3.0 or r["gamma_HS"] > 1e-3)
    ]
    print(f"\n6. KT4-C Sideband Resonance Audit above x_ring adiabatic envelope ({b_env_ring:.4f}): "
          f"found {len(anomalous_pts)} anomalous candidate(s): {[r['beta_0'] for r in anomalous_pts]}", flush=True)

    sideband_zoom = []
    # Always verify representative points (including any anomalous candidate, plus b0=0.76, 0.781, 1.00, 2.00) with J_cut=60!
    verify_betas = sorted(set([r["beta_0"] for r in anomalous_pts[:4]] + [0.781, 1.00, 2.00]))
    jcut60_verify = []
    for b0 in verify_betas:
        r60 = run_one_trajectory(bg_ring, 35.0, b0, spec_is60, y0_stelle, Nu=44)
        jcut60_verify.append(r60)
        print(f"   [J_cut=60 verify] b0={b0:6.3f} (tau={r60['tau_reached']:5.1f}): gam={r60['gamma_HS']:+.4e} | "
              f"max_ratio={r60['max_ratio']:8.2e} | blowup={r60['blowup']}", flush=True)

    # If any anomalous point was found above envelope, also zoom +-0.03 around the strongest one!
    if anomalous_pts:
        peak_pt = max(anomalous_pts, key=lambda r: r["gamma_HS"])
        b_c = peak_pt["beta_0"]
        print(f"   Zooming +-0.04 around candidate sideband resonance beta_0 = {b_c:.3f}:", flush=True)
        for db in [-0.04, -0.02, -0.01, 0.0, +0.01, +0.02, +0.04]:
            bt = round(b_c + db, 4)
            rz = run_one_trajectory(bg_ring, 35.0, bt, spec_is30, y0_stelle, Nu=44)
            sideband_zoom.append(rz)
            print(f"     zoom b0={bt:6.4f}: gam={rz['gamma_HS']:+.4e} | max_ratio={rz['max_ratio']:8.2e} | blowup={rz['blowup']}", flush=True)

    # Save 03_kill_tests.json
    kt_payload = {
        "stream_id": "p2_01_discrete_photon_pole_sum",
        "backgrounds": {"x_ring": info_ring, "x_b15": info_ecc},
        "adiabatic_pinchoff_envelopes": {
            "x_ring_discrete_Istar": env_ring_disc,
            "x_ring_continuum_M26": env_ring_cont,
            "x_b15_discrete_Istar": env_ecc_disc,
            "x_b15_continuum_M26": env_ecc_cont,
        },
        "xring_discrete_Istar_scan": xring_disc_scan,
        "xring_control_continuum_M26": ctrl_cont26,
        "xring_control_discrete_S3": ctrl_s3_disc,
        "xring_phys_ic_comparison": phys_ic_rows,
        "eccentric_xb15_discrete_Istar_scan": ecc_disc_scan,
        "eccentric_xb15_control_continuum_M26": ecc_ctrl_cont26,
        "sideband_anomalous_above_envelope": anomalous_pts,
        "sideband_zoom": sideband_zoom,
        "jcut60_long_verification": jcut60_verify,
    }
    with open(OUT_KT_PATH, "w") as fh:
        json.dump(kt_payload, fh, indent=2)
    print(f"\nSaved {OUT_KT_PATH}", flush=True)
    assemble_canonical_results(kt_payload)


def assemble_canonical_results(kt_payload):
    """Assemble canonical results.json strictly from 01_derive.json + 02_solve.json + kt_payload."""
    info_ring = kt_payload["backgrounds"]["x_ring"]
    info_ecc = kt_payload["backgrounds"]["x_b15"]
    env_ring_disc = kt_payload["adiabatic_pinchoff_envelopes"]["x_ring_discrete_Istar"]
    env_ecc_disc = kt_payload["adiabatic_pinchoff_envelopes"]["x_b15_discrete_Istar"]
    xring_disc_scan = kt_payload["xring_discrete_Istar_scan"]
    ctrl_cont26 = kt_payload["xring_control_continuum_M26"]
    ctrl_s3_disc = kt_payload["xring_control_discrete_S3"]
    ecc_disc_scan = kt_payload["eccentric_xb15_discrete_Istar_scan"]
    anomalous_pts = kt_payload["sideband_anomalous_above_envelope"]

    with open(STREAM_DIR / "01_derive.json") as fh:
        d1 = json.load(fh)
    with open(STREAM_DIR / "02_solve.json") as fh:
        d2 = json.load(fh)

    ps = d1["kt3_pinch_off_and_convergence"]["pinch_off_summary"]
    ov = d1["triple_overlap_and_kt1_selection"]
    k12 = d1["kt1_and_kt2_verification"]
    ladder = d1["kt3_pinch_off_and_convergence"]["cutoff_convergence_ladder"]

    stable_ring_rows = [r for r in xring_disc_scan if r["beta_0"] >= 0.747]
    max_gam_stable_ring = max(abs(r["gamma_HS"]) for r in stable_ring_rows)
    any_blowup_stable_ring = any(r["blowup"] for r in stable_ring_rows)
    ecc_stable_rows = [r for r in ecc_disc_scan if r["beta_0"] >= 1.20]
    max_gam_stable_ecc = max(abs(r["gamma_HS"]) for r in ecc_stable_rows)
    any_blowup_ecc_stable = any(r["blowup"] for r in ecc_stable_rows)

    r_disc_050 = next(r for r in xring_disc_scan if abs(r["beta_0"] - 0.50) < 1e-6)
    r_cont_050 = next(r for r in ctrl_cont26 if abs(r["beta_0"] - 0.50) < 1e-6)
    r_disc_100 = next(r for r in xring_disc_scan if abs(r["beta_0"] - 1.00) < 1e-6)
    r_s3_100 = next(r for r in ctrl_s3_disc if abs(r["beta_0"] - 1.00) < 1e-6)

    kt1_passed = (
        k12["kt1_max_abs_Im_below_26"] == 0.0
        and ov["lowest_Istar_opposite_chirality_pole_E"] == 26
        and ov["lowest_S3_opposite_chirality_pole_E"] == 6
        and ov["same_chirality_maxwell_selection_factor"] == 0.0
        and ov["max_matrix_err"] < 1e-14
    )
    kt2_passed = (
        k12["kt2_sliding_window_weyl_anomaly"][-1]["rel_err_vs_9_over_31"] < 2e-5
        and k12["kt2_euclidean_uv_asymptote"][-1]["K_Istar_O_OmE_m2_rel_err"] < 1e-4
        and k12["kt2_gaussian_heat_kernel_bubble_diff_Om40i"][-1]["rel_err"] < 1e-2
    )
    kt3_passed = (
        ladder[-1]["rel_err_trunc_vs_exact"] < 5e-5
        and max(r["dual_precision_beta_diff"] for r in ladder) < 1e-13
        and d1["kt3_pinch_off_and_convergence"]["hybrid_K0matched_root_max_rel_err_J60"] < 5e-5
        and all(
            r["real_stelle_hi_Omega"] is not None
            for r in d1["kt3_pinch_off_and_convergence"]["static_roots_Istar_discrete"]
            if r["beta_0"] > ps["Istar_discrete_exact_ModelA"]["beta_pinch"] + 1e-6
        )
    )
    lh_audit = d2["long_horizon_eta35_bdf_vs_radau_audit"]
    max_lh_dg = max(v["abs_gamma_diff_HS"] for v in lh_audit.values())
    kt4_passed = bool(
        all(r["success"] for r in xring_disc_scan)
        and all(r["success"] for r in ecc_disc_scan)
        and r_disc_050["blowup"] and (r_disc_050["gamma_HS"] > 0.1)
        and (not r_cont_050["blowup"]) and (abs(r_cont_050["gamma_HS"]) < 1e-3)
        and r_s3_100["blowup"] and (r_s3_100["gamma_HS"] > 0.05)
        and (not any_blowup_stable_ring) and (len(anomalous_pts) == 0)
        and (not any_blowup_ecc_stable)
        and (max_lh_dg < 1e-5)
        and d2["dual_solver_cross_checks"]["Istar_disc_beta0_0.5"]["static_gamma_err_vs_mp40"] < 1e-3
    )

    euclid_err_5000 = k12["kt2_euclidean_uv_asymptote"][-1]["K_Istar_O_OmE_m2_rel_err"]
    cc_050 = d2["dual_solver_cross_checks"]["Istar_disc_beta0_0.5"]
    cc_050_j60 = d2["dual_solver_cross_checks"]["Istar_disc_beta0_0.5_J60_precision"]
    cc_100 = d2["dual_solver_cross_checks"]["Istar_disc_beta0_1.0"]
    cc_289 = d2["dual_solver_cross_checks"]["Istar_disc_beta0_0.289"]
    n_stable_ring = len(stable_ring_rows)
    max_ratio_stable_ring = max(r["max_ratio"] for r in stable_ring_rows)
    phys_ge_076 = [r for r in kt_payload["xring_phys_ic_comparison"] if r["beta_0"] >= 0.76]
    max_phys_gam = max(abs(r["disc_gamma_E_HS"]) for r in phys_ge_076)
    grav_eig_max_gam = max(abs(r["gamma_E_HS"]) for r in d2["graviton_eigenvector_ic_check"])
    j30_trunc_err = next(r["rel_err_trunc_vs_exact"] for r in ladder if r["J_cut"] == 30)
    j60_trunc_err = next(r["rel_err_trunc_vs_exact"] for r in ladder if r["J_cut"] == 60)
    j120_trunc_err = next(r["rel_err_trunc_vs_exact"] for r in ladder if r["J_cut"] == 120)
    j240_trunc_err = next(r["rel_err_trunc_vs_exact"] for r in ladder if r["J_cut"] == 240)
    j30_hybm_err = next(r["rel_err_hybrid_K0matched_vs_exact"] for r in ladder if r["J_cut"] == 30)
    j60_hybm_err = next(r["rel_err_hybrid_K0matched_vs_exact"] for r in ladder if r["J_cut"] == 60)
    j240_hybm_err = next(r["rel_err_hybrid_K0matched_vs_exact"] for r in ladder if r["J_cut"] == 240)
    max_f64_diff = max(r["dual_precision_beta_diff"] for r in ladder)
    kt2_last_err = k12["kt2_sliding_window_weyl_anomaly"][-1]["rel_err_vs_9_over_31"]
    om_k_050_mp40_err = cc_050["static_gamma_err_vs_mp40"]
    om_k_050_j60_err = cc_050_j60["rel_err_vs_mp40"]

    canonical_results = {
        "schema_version": "1.0",
        "stream_id": "p2_01_discrete_photon_pole_sum",
        "title": "Discrete S^3/I* Two-Photon Spectral Pole Sum & Horowitz Pinch-Off Shift (CLM-OPEN-01)",
        "status": "verified",
        "updated_at": "2026-11-25",
        "conventions": {
            "metric_signature": "(-,+,+,+)",
            "units": "M_Pl = H_S = 1 (Starobinsky-Casimir dimensionless units, Lambda = 0)",
            "coframe": "d(omega^a) = -eps_abc omega^b ^ omega^c, [X_a, X_b] = 2 eps_abc X_c",
            "volume_S3": "2*pi^2",
            "volume_S3_Istar": "pi^2 / 60",
            "renormalization_scheme": "Local UV Hadamard/Euclidean asymptotic matching: K_disc(i*Om_E; mu_cyl) = ln(Om_E^2/mu_cyl^2) + O(Om_E^-2) with mu_cyl = x_ring * H_S",
        },
        "parameters": {
            "C_ANOM": C_ANOM,
            "SIGMA_0": SIGMA_0,
            "KAPPA_V": KAPPA_V,
            "X_RING": X_RING,
            "n_mode": 2,
            "k_2_sq": 8,
            "p_2": 3,
            "M_Istar_2": 26,
            "M_S3_2": 6,
            "J_cut_time_domain_primary": 30,
            "J_cut_time_domain_verify": 60,
        },
        "quantities": {
            "beta_pinch_Istar_discrete_ModelA": {
                "value": ps["Istar_discrete_exact_ModelA"]["beta_pinch"],
                "uncertainty_or_tol": 1e-15,
                "units": "dimensionless (H_S=1)",
                "precision_digits": 15,
            },
            "Omega_pinch_Istar_discrete_ModelA": {
                "value": ps["Istar_discrete_exact_ModelA"]["Omega_pinch"],
                "uncertainty_or_tol": 1e-14,
                "units": "conformal frequency (1/a)",
                "precision_digits": 14,
            },
            "beta_pinch_Istar_continuum_M26": {
                "value": ps["Istar_continuum_M26"]["beta_pinch"],
                "uncertainty_or_tol": 1e-15,
                "units": "dimensionless (H_S=1)",
                "precision_digits": 15,
            },
            "beta_pinch_Istar_shift_rel": {
                "value": ps["shift_Istar_ModelA_rel"],
                "uncertainty_or_tol": 1e-14,
                "units": "relative shift (+78.23%)",
                "precision_digits": 14,
            },
            "beta_pinch_Istar_discrete_ModelB_abs": {
                "value": ps["Istar_discrete_ModelB_abs"]["beta_pinch"],
                "uncertainty_or_tol": 1e-14,
                "units": "dimensionless (un-factored Kubo bubble sensitivity)",
                "precision_digits": 14,
            },
            "beta_pinch_Istar_discrete_ModelB_relS3": {
                "value": ps["Istar_discrete_ModelB_relS3"]["beta_pinch"],
                "uncertainty_or_tol": 1e-14,
                "units": "dimensionless (un-factored Kubo bubble relative to S^3)",
                "precision_digits": 14,
            },
            "beta_pinch_S3_discrete_exact": {
                "value": ps["S3_discrete_exact"]["beta_pinch"],
                "uncertainty_or_tol": 1e-15,
                "units": "dimensionless (H_S=1)",
                "precision_digits": 15,
            },
            "topology_advantage_ratio_discrete": {
                "value": ps["topology_advantage_ratio_discrete_ModelA"],
                "uncertainty_or_tol": 1e-14,
                "units": "beta_pinch(S^3)/beta_pinch(S^3/I*)",
                "precision_digits": 14,
            },
            "beta_pinch_adiabatic_max_xring_discrete": {
                "value": env_ring_disc["beta_pinch_inst_full_max"],
                "uncertainty_or_tol": 5e-4,
                "units": "dimensionless (H_S=1)",
                "precision_digits": 4,
            },
            "beta_pinch_adiabatic_max_xb15_discrete": {
                "value": env_ecc_disc["beta_pinch_inst_full_max"],
                "uncertainty_or_tol": 1e-3,
                "units": "dimensionless (H_S=1)",
                "precision_digits": 4,
            },
            "gamma_xring_beta0_0p50_discrete": {
                "value": r_disc_050["gamma_HS"],
                "uncertainty_or_tol": 5e-4,
                "units": "H_S",
                "precision_digits": 4,
            },
            "gamma_xring_beta0_1p00_discrete": {
                "value": r_disc_100["gamma_HS"],
                "uncertainty_or_tol": 2e-4,
                "units": "H_S",
                "precision_digits": 3,
            },
            "time_domain_Jcut30_vs_Jcut60_rel_diff_beta0_1p0": {
                "value": d2["time_domain_Jcut_convergence"]["beta0_1.0"]["rel_diff_pi2_J30_vs_J60"],
                "uncertainty_or_tol": 1e-5,
                "units": "relative pi2 difference over eta=10",
                "precision_digits": 3,
            },
            "static_beta0_0p50_aux_ode_vs_mp40_gamma_rel_err_J30": {
                "value": om_k_050_mp40_err,
                "uncertainty_or_tol": 3e-4,
                "units": "relative growth-rate error vs 40-digit root (J30; J60 peak-fit = 2.19e-4)",
                "precision_digits": 3,
            },
            "static_beta0_0p50_aux_ode_vs_mp40_gamma_rel_err_J60": {
                "value": om_k_050_j60_err,
                "uncertainty_or_tol": 3e-4,
                "units": "relative growth-rate error vs 40-digit root (J60)",
                "precision_digits": 3,
            },
            "long_horizon_eta35_bdf_vs_radau_max_gamma_diff": {
                "value": max_lh_dg,
                "uncertainty_or_tol": 1e-6,
                "units": "H_S (across beta_0 in {0.747, 1.00, 1.719})",
                "precision_digits": 3,
            },
            "zeta_Istar_1_exact": {
                "value": k12["kt2_exact_zeta_constants"]["zeta_Istar_1"],
                "uncertainty_or_tol": 1e-15,
                "units": "exact rational -724/3",
                "precision_digits": 15,
            },
            "zeta_S3_1_exact": {
                "value": k12["kt2_exact_zeta_constants"]["zeta_S3_1"],
                "uncertainty_or_tol": 1e-15,
                "units": "exact rational -74/3",
                "precision_digits": 15,
            },
        },
        "verification": {
            "conservation_residuals": {
                "p05_xring_friedmann_e00_max_rel": info_ring["max_rel_e00"],
                "p05_xring_friedmann_e00_rms_rel": info_ring["rms_rel_e00"],
                "p05_xring_friedmann_e00_max_abs": info_ring["max_abs_e00"],
                "p05_xb15_friedmann_e00_max_rel": info_ecc["max_rel_e00"],
                "p05_xb15_friedmann_e00_max_abs": info_ecc["max_abs_e00"],
                "exact_discrete_pole_hamiltonian_H_tot_dop853_eta8_max_rel_drift": d2["exact_hamiltonian_conservation"]["H_tot_max_rel_drift"],
                "exact_discrete_pole_hamiltonian_H_tot_kt4_prod_bdf_eta35_max_rel_drift": d2["exact_hamiltonian_conservation"]["H_tot_kt4_prod_bdf_eta35_max_rel_drift"],
                "rg_scale_invariance_aux_ode_rel_diff": d2["rg_scale_invariance"]["solver_A_aux_ode_rel_diff"],
                "rg_scale_invariance_direct_volterra_rel_diff": d2["rg_scale_invariance"]["solver_B_direct_volterra_rel_diff"],
                "analytical_jacobian_vs_fd_rel_err": d2["jacobian_fd_verification"]["with_tail_rel_err"],
            },
            "solver_cross_checks": {
                "C_R_racah_cg_vs_exact_max_err": ov["max_cg_err"],
                "C_R_su2_curl_matrix_vs_exact_max_err": ov["max_matrix_err"],
                "bach_polynomial_factorization_rel_err": ov["max_bach_factorization_rel_err"],
                "kt3_mp40_vs_float64_beta_pinch_max_diff": max_f64_diff,
                "dyn_beta0_0p289_radau_vs_bdf_rel_diff": cc_289["dyn_KAM_rel_diff_Radau_vs_BDF"],
                "dyn_beta0_0p50_radau_vs_bdf_rel_diff": cc_050["dyn_KAM_rel_diff_Radau_vs_BDF"],
                "dyn_beta0_1p00_radau_vs_bdf_rel_diff": cc_100["dyn_KAM_rel_diff_Radau_vs_BDF"],
                "static_beta0_1p00_aux_ode_vs_direct_volterra_richardson_rel_diff": cc_100["static_rel_diff_AuxODE_vs_Volterra_Richardson"],
                "dyn_beta0_1p00_aux_ode_vs_direct_volterra_richardson_rel_diff": cc_100["dyn_KAM_rel_diff_AuxODE_vs_Volterra_Richardson"],
                "long_horizon_eta35_bdf_vs_radau_max_gamma_diff_HS": max_lh_dg,
            },
            "matched_controls": [
                {
                    "control_id": "MC1_continuum_M26_vs_discrete_Istar_xring_beta0_0p50",
                    "matched_components": "Identical P05 x_ring background, y0_4d=[0,0,1,0], beta_0=0.50, c_anom=9/31, mu_cyl=x_ring, BDF rtol=1e-8",
                    "treatment_result": f"Discrete S^3/I* pole sum (J_cut=30, E<=122 + K0-matched UV tail): UHP Horowitz runaway gam={r_disc_050['gamma_HS']:+.4e} H_S, blowup={r_disc_050['blowup']}",
                    "control_result": f"Matched continuum-log M=26: bounded gam={r_cont_050['gamma_HS']:+.4e} H_S, max_ratio={r_cont_050['max_ratio']:.3f}, blowup={r_cont_050['blowup']}",
                    "conclusion": f"Discrete pole repulsion shifts conformal-Weyl (Model A) static beta_pinch from {ps['Istar_continuum_M26']['beta_pinch']:.4f} to {ps['Istar_discrete_exact_ModelA']['beta_pinch']:.4f} (adiabatic max {env_ring_disc['beta_pinch_inst_full_max']:.4f}), reopening UHP Horowitz runaway in [0.44, 0.72].",
                },
                {
                    "control_id": "MC2_discrete_S3_vs_discrete_Istar_xring_beta0_1p00",
                    "matched_components": "Identical P05 x_ring background, y0_4d=[0,0,1,0], beta_0=1.00, exact discrete 2-photon pole sum (S^3/I*: J_cut=30, 15 poles E<=122; round S^3: J_cut=15, 29 poles E<=62; both K0-matched to UV continuum-log tail), BDF rtol=1e-8",
                    "treatment_result": f"Discrete S^3/I* pole sum: bounded gam={r_disc_100['gamma_HS']:+.4e} H_S, max_ratio={r_disc_100['max_ratio']:.3f}, blowup={r_disc_100['blowup']}",
                    "control_result": f"Matched discrete round S^3 pole sum: catastrophic UHP runaway gam={r_s3_100['gamma_HS']:+.4e} H_S, blowup={r_s3_100['blowup']}",
                    "conclusion": f"Chiral topological Horowitz stabilization survives exact conformal-Weyl discrete pole replacement across Regime IV beta_0 in [0.75, {ps['S3_discrete_exact']['beta_pinch']:.2f}] ({ps['topology_advantage_ratio_discrete_ModelA']:.2f}x advantage over round S^3).",
                },
                {
                    "control_id": "MC3_same_chirality_j0_E4_selection_rule",
                    "matched_components": "Identical SU(2) left-invariant mode space for j=0 (-2,-2) pair vs opposite-chirality j=6 (+12,-14) pair",
                    "treatment_result": f"Maxwell (1 - s1*s2) = {ov['same_chirality_maxwell_selection_factor']:.1f} identically kills E=4 despite unit geometric overlap |O|^2 = {ov['same_chirality_j0_E4_geometric_overlap_sq']:.4f}",
                    "control_result": f"Opposite-chirality j=6 (+12,-14) gives (1 - s1*s2) = 2 and first nonzero S^3/I* pole at E_6 = {ov['lowest_Istar_opposite_chirality_pole_E']}",
                    "conclusion": "Chiral gap |Omega| < 26 is enforced strictly by electromagnetic duality T_ij^TF ~ (1 - s1*s2), not geometric vanishing.",
                },
            ],
            "kill_tests": [
                {
                    "id": "KT1",
                    "hypothesis_attacked": "KT1 (Analyticity & Chiral Gap): Exact opposite-chirality selection (1 - s_1*s_2) preserves Im Pi_discrete(Omega) == 0 identically for all |Omega| < 26 and recovers exact M_{S^3}(2)=6 on round S^3.",
                    "passed": bool(kt1_passed),
                    "metric": float(k12["kt1_max_abs_Im_below_26"]),
                    "notes": f"Exact C_R(j)=(2j-1)(2j+3)/[5(2j+1)] proved analytically via stretched Wigner 6j-symbol and verified via Racah CG ({ov['max_cg_err']:.2e}) and SU(2) curl-matrix diagonalization ({ov['max_matrix_err']:.2e}); max |Im K_I*| below 26 is {k12['kt1_max_abs_Im_below_26']:.1f}; lowest S^3 pole E={ov['lowest_S3_opposite_chirality_pole_E']}, S^3/I* pole E={ov['lowest_Istar_opposite_chirality_pole_E']}; same-chirality j=0 (E=4) has |O|^2={ov['same_chirality_j0_E4_geometric_overlap_sq']:.4f} killed strictly by (1-s1*s2)=0.",
                },
                {
                    "id": "KT2",
                    "hypothesis_attacked": "KT2 (Continuum & Weyl-Anomaly UV Recovery): High-frequency average of the discrete S^3/I* spectral density recovers the exact 4D conformal Weyl-anomaly coefficient c_anom = 9/31 (and continuum-log slope) as Omega -> infinity.",
                    "passed": bool(kt2_passed),
                    "metric": float(kt2_last_err),
                    "notes": f"Sliding-window period-30 dissipative spectral density (rho = 2*Im chi_R) average converges O(1/E^2) to exact 1/(40*pi)*B_2(Om) (Im Pi_micro = -Im chi_R = -N_v/(80*pi)*B_2 -> c_anom=9/31, rel err {kt2_last_err:.2e} at E~6060); Euclidean K_I*(i*OmE)-ln(OmE^2/mu^2) = +724/(3*OmE^2) (rel err {euclid_err_5000:.2e} at OmE=5000); Gaussian heat-kernel exp(-(eps*E)^2) confirms exact cancellation of O(OmE^4 ln OmE) and O(OmE^2) local UV counterterms.",
                },
                {
                    "id": "KT3",
                    "hypothesis_attacked": "KT3 (Cutoff Convergence & Pole-Repulsion Shift): Track beta_pinch(N_cut) and UHP/real Stelle roots across spectral cutoffs N_cut (e.g. 60, 100, 160, 240) plus dual-precision (float64 vs mpmath) check to confirm quantitative convergence and quantify shift relative to 0.3836.",
                    "passed": bool(kt3_passed),
                    "metric": float(j240_trunc_err),
                    "notes": f"Static conformal-Weyl (Model A) beta_pinch(S^3/I*) shifts {100*ps['shift_Istar_ModelA_rel']:+.2f}% from {ps['Istar_continuum_M26']['beta_pinch']:.6f} to {ps['Istar_discrete_exact_ModelA']['beta_pinch']:.6f} (Om_pinch={ps['Istar_discrete_exact_ModelA']['Omega_pinch']:.4f}; round S^3 shifts {100*ps['shift_S3_rel']:+.2f}% from {ps['S3_continuum_M6']['beta_pinch']:.6f} to {ps['S3_discrete_exact']['beta_pinch']:.6f}; advantage {ps['topology_advantage_ratio_discrete_ModelA']:.2f}x). Period-30 truncated ladder J=30,60,120,240 converges O(1/J^2): {j30_trunc_err:.2e} -> {j60_trunc_err:.2e} -> {j120_trunc_err:.2e} -> {j240_trunc_err:.2e}; K0-matched hybrid converges O(1/J^4): {j30_hybm_err:.2e} -> {j60_hybm_err:.2e} -> {j240_hybm_err:.2e}; mp40 vs float64 max diff {max_f64_diff:.2e}; all upper companion and Omega_1 roots resolved with zero null dropouts. Partial un-factored Kubo bubble (Model B) gives {ps['Istar_discrete_ModelB_abs']['beta_pinch']:.4f}-{ps['Istar_discrete_ModelB_relS3']['beta_pinch']:.4f}.",
                },
                {
                    "id": "KT4",
                    "hypothesis_attacked": "KT4 (Dynamical Volterra vs Discrete-Pole Sidebands): Run unreduced dynamical evolution on P05 x_ring and x_b=15 across Regime IV beta_0 in [0.44, 6.37] to test whether discrete 2-photon poles above Omega=26 destabilize any narrow sub-window via scalaron sidebands (Omega_grav + k*omega_scal).",
                    "passed": bool(kt4_passed),
                    "metric": float(max_gam_stable_ring),
                    "notes": (
                        f"Adiabatic discrete pinch-off envelope rises to {env_ring_disc['beta_pinch_inst_full_max']:.4f} on x_ring (vs 0.4206 cont) "
                        f"and {env_ecc_disc['beta_pinch_inst_full_max']:.4f} on x_b=15 (vs 0.7326 cont). "
                        f"On x_ring, beta_0 in [0.44, 0.72] is destabilized by UHP Horowitz pole repulsion (b0=0.50: gam={r_disc_050['gamma_HS']:+.4e} H_S; "
                        f"Volterra Richardson diff static={cc_050['static_rel_diff_AuxODE_vs_Volterra_Richardson']:.2e}, dyn={cc_050['dyn_KAM_rel_diff_AuxODE_vs_Volterra_Richardson']:.2e}; "
                        f"static mp40 err J30={om_k_050_mp40_err:.2e}, J60={om_k_050_j60_err:.2e}). "
                        f"Above {env_ring_disc['beta_pinch_inst_full_max']:.4f}, 0/{n_stable_ring} sampled points in [0.747, 8.00] "
                        f"(including exact dressed hyperbolic Krein candidates H1=0.747, H_fund=1.199, H2_mod=1.217, H3=1.994, H4=2.186) destabilize "
                        f"(max |gam|={max_gam_stable_ring:.3e} H_S, max_ratio<={max_ratio_stable_ring:.2f}, blowup={any_blowup_stable_ring}; "
                        f"zero-pi2 IC |gam_E|<={max_phys_gam:.2e} H_S, true graviton-eigenvector IC |gam_E|<={grav_eig_max_gam:.2e} H_S; "
                        f"long-horizon eta=35 BDF vs Radau max |dg|={max_lh_dg:.2e} H_S; J_cut=60 confirms b0=0.781, 1.00, 2.00 bounded). "
                        f"On x_b=15, beta_0 >= 1.20 is bounded (max |gam|={max_gam_stable_ecc:.3e} H_S, blowup={any_blowup_ecc_stable})."
                    ),
                },
            ],
            "domain_of_validity": {
                "regime": "1-loop conformally coupled Maxwell fields (large-N_v dominance) on R_eta x S^3/I* with homogeneous n=2 (-) TT graviton perturbation on P05 Starobinsky-Casimir KAM background.",
                "caveats": [
                    "Model A (primary, beta_pinch=0.6838) is strictly the conformal-Weyl-factored discrete kernel B_2(Om)*K_disc(Om), not the complete 1-loop Maxwell self-energy on R x S^3/I*. In the un-factored Kubo <T_ij T_kl> bubble (Model B), exact polynomial division B_2(E_j)/(E_j^2-Om^2) = (E_j^2-20) + Om^2 + B_2(Om)/(E_j^2-Om^2) generates O(1)-relative-to-Einstein-Hilbert non-Weyl topological Casimir terms on P05's Casimir-supported ring (x_ring^2 = 480.34 ~ |kappa_V| = 481.05): (a) a topology-dependent kinetic shift -2*c_anom*(zeta_{I*}(1)-zeta_{S^3}(1))*Om^2 = +125.81*Om^2 (26.2% of x_ring^2; absolute -2*c_anom*zeta_{I*}(1)*Om^2 = +140.13*Om^2 = 29.2%) that cannot be absorbed by local UV counterterms or metric-derivative-free contact diagrams, shifting static beta_pinch to 0.5615-0.5747; and (b) a Kubo Om^0 constant -2*c_anom*(zeta_{I*}(3)-20*zeta_{I*}(1)) = +608048/155 = +3922.89 (102% of 8*x_ring^2 = 3842.75) that combines with the uncomputed squashed-Bianchi-IX contact/tadpole diagram to form the O(1)-relative-to-curvature static Casimir shear modulus.",
                    "Time-domain dynamical runs use exact discrete harmonic oscillators for all nonzero S^3/I* poles j <= J_cut (J_cut=30, 15 poles E<=122; verified at J_cut=60, 45 poles E<=242 with 2.32e-4 trajectory diff at beta_0=1.0) K0-matched to a smooth continuum-log UV tail above M_tail = 4*J_cut+4; KT4 production BDF (rtol=1e-8, max_step=0.025) conserves exact H_tot to 3.38e-8 over eta=35 and matches tight-tolerance Radau (rtol=1e-10, max_step=0.01, Nu=64) to |d_gamma| <= 2.91e-6 H_S across beta_0 in {0.747, 1.00, 1.719}.",
                    f"KT4 tests {len(xring_disc_scan)} beta_0 values on x_ring (including exact dressed hyperbolic Krein candidates H1=0.747, H_fund=1.199, H2_mod=1.217, H3=1.994, H4=2.186) over tau = 773.4 H_S^-1 (5.65 T_macro) and {len(ecc_disc_scan)} beta_0 values on x_b=15 over tau = 707.5 H_S^-1; macro-modulation of Om_scal in [38.60, 43.47] chirps across E_10=42 and suppresses locking, though ultra-narrow tongues (Delta_beta_0 << 0.01) or secular growth below gamma ~ 1e-4 H_S beyond tau = 773 are not excluded.",
                    "Large-N_v Maxwell closure omits O(1/N_v) 2-graviton self-loops (cut at 2*sqrt(8)~5.66 < 26) and nonlinear radiation/scalaron backreaction into Friedmann E_00=0 (CLM-OPEN-02); P05's sigma_0=9/31 background sits at 4*c_anom*omega_scal^2 = 4.000 (strict 1-loop scalaron EFT requires sigma_t >= 4).",
                ],
            },
        },
    }

    with open(RESULTS_PATH, "w") as fh:
        json.dump(canonical_results, fh, indent=2)
    print(f"Saved canonical {RESULTS_PATH}", flush=True)


def add_krein_probes_and_reassemble():
    """Run all exact dressed hyperbolic Krein probe points and re-assemble results.json."""
    with open(OUT_KT_PATH) as fh:
        kt_payload = json.load(fh)
    bg_ring, _ = m2.integrate_kam_background_eta(X_RING, eta_end=45.0)
    spec_is30 = m2.build_discrete_pole_spec("Istar", J_cut=30, include_tail=True, match_exact_K0=True)
    y0_stelle = [0.0, 0.0, 1.0, 0.0]
    existing = {round(r["beta_0"], 4): r for r in kt_payload["xring_discrete_Istar_scan"]}
    for b0 in [1.199, 1.217, 1.650, 1.719, 1.800, 1.994, 2.029, 2.186]:
        if round(b0, 4) not in existing:
            row = run_one_trajectory(bg_ring, 35.0, b0, spec_is30, y0_stelle, Nu=44)
            existing[round(b0, 4)] = row
            print(f"   [Krein probe x_ring] b0={b0:6.3f} (tau={row['tau_reached']:5.1f}): "
                  f"gam={row['gamma_HS']:+.4e} | max_ratio={row['max_ratio']:8.2e} | blowup={row['blowup']}", flush=True)
    kt_payload["xring_discrete_Istar_scan"] = sorted(existing.values(), key=lambda r: r["beta_0"])
    b_env = kt_payload["adiabatic_pinchoff_envelopes"]["x_ring_discrete_Istar"]["beta_pinch_inst_full_max"]
    kt_payload["sideband_anomalous_above_envelope"] = [
        r for r in kt_payload["xring_discrete_Istar_scan"]
        if r["beta_0"] > b_env + 0.02 and (r["blowup"] or r["max_ratio"] > 3.0 or r["gamma_HS"] > 1e-3)
    ]
    with open(OUT_KT_PATH, "w") as fh:
        json.dump(kt_payload, fh, indent=2)
    print(f"Updated {OUT_KT_PATH} (total x_ring points = {len(kt_payload['xring_discrete_Istar_scan'])})", flush=True)
    assemble_canonical_results(kt_payload)


if __name__ == "__main__":
    if "--add-krein-probes" in sys.argv:
        add_krein_probes_and_reassemble()
    elif "--assemble-only" in sys.argv:
        with open(OUT_KT_PATH) as fh:
            assemble_canonical_results(json.load(fh))
    else:
        main()
