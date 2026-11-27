"""T4B-03B (Targeted Avalanche Follow-Up): Complete rho_2 -> S_02 -> rho_0 Decay, True Pre-Bounce
Contracting gamma_c = 1/4 Collapse Bifurcation, Sub-Horizon Supercooling vs. Immediate Recollapse,
and Post-Avalanche DOP853 vs. Radau & N_max=12..36 Kill-Tests.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "track2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from istar_geometry import build_istar_and_galois, build_p7_equivariant_polynomial  # noqa: E402
from t4b_02_realtime_equivariant_solver import (  # noqa: E402
    RealtimeEquivariantYMHiggsCosmology,
    build_equivariant_basis,
)
from t4b_03_dynamical_phenomena import gauss_low_n_norm, relax_flat_rho2  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    t0 = time.time()
    beta_H = 1.556
    _, _, U1, U2 = build_istar_and_galois()
    _, poly_coeffs = build_p7_equivariant_polynomial(U1, U2)

    bas24 = build_equivariant_basis(U1, k_winding=7, max_n=24, poly_coeffs=poly_coeffs)
    c_flat24, E_flat24, ncs_flat24 = relax_flat_rho2(bas24)

    # ------------------------------------------------------------------
    # 1. Complete Expanding Avalanche (N_CS: 49/120 -> 0) to a = 25.0
    # ------------------------------------------------------------------
    print("=== 1. Complete Expanding rho_2 -> S_02 -> rho_0 Avalanche (M_W=5.0, a: 1.10 -> 25.0) ===", flush=True)
    eng_exp = RealtimeEquivariantYMHiggsCosmology(bas24, kappa_YM=2e-4, M_W=5.0, beta_H=beta_H)
    y0_exp, _ = eng_exp.build_initial_state(
        a0=1.10, c0=c_flat24, cp0=np.zeros(eng_exp.D_A),
        phi0=5e-2 * bas24["phi_rho2_ground"], phip0=np.zeros(eng_exp.D_Phi), contracting=False
    )
    res_exp = eng_exp.integrate(
        y0_exp, (0.0, 2.5), method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.004, a_stop_hi=25.0, n_eval=400
    )
    tr_exp = res_exp["trace"]
    a_mid = next((r["a"] for r in tr_exp if r["N_CS"] < 0.20), None)
    a_zero = next((r["a"] for r in tr_exp if r["N_CS"] < 0.05), None)
    E_ym_pk = max(r["E_YM_S3I"] for r in tr_exp)
    ncs_min = min(r["N_CS"] for r in tr_exp)
    print(
        f"  a: 1.10 -> {res_exp['a_final']:.2f} (a_mid={a_mid}, a_<0.05={a_zero}) | "
        f"N_CS: {tr_exp[0]['N_CS']:.4f} -> min {ncs_min:+.4f} -> end {tr_exp[-1]['N_CS']:+.4f} | "
        f"phi_rms_end={tr_exp[-1]['phi_rms']:.4f} | E_YM_pk={E_ym_pk:.2f} | "
        f"dT00_mat={res_exp['max_delta_T00_matter']:.4f}, dT00_tot={res_exp['max_delta_T00_tot']:.4f} | "
        f"H_res={res_exp['max_H_res_rel']:.2e}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 2. True Pre-Bounce Contracting Avalanche -> Exact gamma_c = 1/4 Recollapse Bifurcation!
    # Start contracting from a0 = 4.5 (M_W = 5.0, a_restore = 1.4428) with phi0 = 0.08 * phi_rho2_ground
    # so the rho_2 -> S_02 -> rho_0 avalanche FIRES & COMPLETES while contracting (before bounce)!
    # Sweep kappa_YM across the exact gamma_c = 1/4 bounce ceiling!
    # ------------------------------------------------------------------
    print("\n=== 2. True Pre-Bounce Contracting Avalanche vs. gamma_c = 1/4 Ceiling (M_W=5.0, a0=4.5) ===", flush=True)
    contract_sweep = []
    tr_contract_bounce = None
    tr_contract_crunch = None
    for kap in [2.0e-7, 5.0e-7, 1.0e-6, 1.5e-6, 2.0e-6, 3.0e-6, 5.0e-6, 1.0e-5]:
        eng_c = RealtimeEquivariantYMHiggsCosmology(bas24, kappa_YM=kap, M_W=5.0, beta_H=beta_H)
        y0_c, _ = eng_c.build_initial_state(
            a0=4.5, c0=c_flat24, cp0=np.zeros(eng_c.D_A),
            phi0=0.08 * bas24["phi_rho2_ground"], phip0=np.zeros(eng_c.D_Phi), contracting=True
        )
        res_c = eng_c.integrate(
            y0_c, (0.0, 5.0), method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.004,
            a_stop_hi=6.0, a_stop_lo=0.05, n_eval=300
        )
        tr = res_c["trace"]
        outcome = "singular_collapse" if res_c["collapsed"] else ("nonsingular_bounce" if res_c["escaped"] else "unresolved")
        idx_amin = int(np.argmin([r["a"] for r in tr]))
        # Measure N_CS when a first drops below 1.5 (confirming avalanche completed PRE-BOUNCE!)
        ncs_at_a15 = next((r["N_CS"] for r in tr if r["a"] <= 1.5), tr[idx_amin]["N_CS"])
        gam_at_a10 = next((r["gamma_tot"] for r in tr if r["a"] <= 1.05), tr[idx_amin]["gamma_tot"])
        gam_ym_at_a10 = next((r["gamma_ym"] for r in tr if r["a"] <= 1.05), tr[idx_amin]["gamma_ym"])
        row = {
            "kappa_YM": kap,
            "outcome": outcome,
            "a_min": res_c["a_min"],
            "a_final": res_c["a_final"],
            "N_CS_init": tr[0]["N_CS"],
            "N_CS_prebounce_a1p5": ncs_at_a15,
            "N_CS_at_amin": tr[idx_amin]["N_CS"],
            "N_CS_min": min(r["N_CS"] for r in tr),
            "phi_rms_at_a1p5": next((r["phi_rms"] for r in tr if r["a"] <= 1.5), tr[idx_amin]["phi_rms"]),
            "E_YM_peak_S3I": max(r["E_YM_S3I"] for r in tr),
            "gamma_ym_at_a1": gam_ym_at_a10,
            "gamma_tot_at_a1": gam_at_a10,
            "max_delta_T00_matter": res_c["max_delta_T00_matter"],
            "max_delta_T00_tot": res_c["max_delta_T00_tot"],
            "max_H_res_rel": res_c["max_H_res_rel"],
            "max_gauss_n0": res_c["max_gauss_n0"],
        }
        contract_sweep.append(row)
        if outcome == "nonsingular_bounce":
            tr_contract_bounce = tr
        if outcome == "singular_collapse" and tr_contract_crunch is None:
            tr_contract_crunch = tr
        print(
            f"  kap={kap:.1e}: {outcome:18s} | a_min={row['a_min']:.4f} | "
            f"N_CS(a=1.5)={ncs_at_a15:+.4f} (min {row['N_CS_min']:+.4f}) | "
            f"gam_ym(a~1)={gam_ym_at_a10:.4f}, gam_tot(a~1)={gam_at_a10:.4f} | "
            f"dT00_tot={row['max_delta_T00_tot']:.4f} | H_res={row['max_H_res_rel']:.2e}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 3. Sub-Horizon False-Vacuum Bounce: Supercooled Escape vs. Immediate-Avalanche Recollapse!
    # M_W = 22.0, a_restore = 0.3279 < 1/sqrt(2) = 0.7071, kappa = 1.15e-4 (a_bnc = 0.2902, u_vac/3 = 10.87).
    # Compare initial seed phi_seed in {0.05, 0.20, 0.40, 0.60, 0.80} at a0 = 0.34 (just past a_restore)!
    # When avalanche fires promptly at a_av ~ 0.35..0.42: gamma_rad ~ 10.87 * 0.38^4 = 0.226 < 0.25 => RECOLLAPSE!
    # When supercooled (small seed 0.05): avalanche fires at a_av ~ 0.55..0.65 => gamma_rad > 0.25 => ESCAPE!
    # ------------------------------------------------------------------
    print("\n=== 3. Sub-Horizon Supercooling vs. Immediate-Avalanche Recollapse (M_W=22.0, kap=1.15e-4) ===", flush=True)
    sub_seed_rows = []
    tr_sub_recollapse = None
    tr_sub_escape = None
    for seed_phi in [0.05, 0.25, 0.45, 0.65, 0.85]:
        eng_s = RealtimeEquivariantYMHiggsCosmology(bas24, kappa_YM=1.15e-4, M_W=22.0, beta_H=beta_H)
        # Start just above a_bounce at a0 = 0.305 (expanding!)
        y0_s, _ = eng_s.build_initial_state(
            a0=0.305, c0=c_flat24, cp0=np.zeros(eng_s.D_A),
            phi0=seed_phi * bas24["phi_rho2_ground"], phip0=np.zeros(eng_s.D_Phi), contracting=False
        )
        res_s = eng_s.integrate(
            y0_s, (0.0, 4.0), method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.0025,
            a_stop_hi=2.5, a_stop_lo=0.05, n_eval=300
        )
        tr = res_s["trace"]
        outcome = "avalanche_recollapse" if res_s["collapsed"] else ("avalanche_escape" if res_s["escaped"] else "unresolved")
        a_av = next((r["a"] for r in tr if r["N_CS"] < 0.20), None)
        row = {
            "phi_seed": seed_phi,
            "outcome": outcome,
            "a_av_mid": a_av,
            "a_max_reached": res_s["a_max"],
            "a_final": res_s["a_final"],
            "N_CS_min": min(r["N_CS"] for r in tr),
            "N_CS_final": tr[-1]["N_CS"],
            "gamma_ym_peak": max(r["gamma_ym"] for r in tr),
            "max_delta_T00_tot": res_s["max_delta_T00_tot"],
            "max_H_res_rel": res_s["max_H_res_rel"],
        }
        sub_seed_rows.append(row)
        if outcome == "avalanche_recollapse" and tr_sub_recollapse is None:
            tr_sub_recollapse = tr
        if outcome == "avalanche_escape" and tr_sub_escape is None:
            tr_sub_escape = tr
        print(
            f"  seed={seed_phi:.2f}: {outcome:20s} | a_av={a_av} | a_max={row['a_max_reached']:.4f} -> {row['a_final']:.4f} | "
            f"N_CS min={row['N_CS_min']:+.4f} end={row['N_CS_final']:+.4f} | gam_ym_pk={row['gamma_ym_peak']:.4f} | "
            f"dT00_tot={row['max_delta_T00_tot']:.4f} | H_res={row['max_H_res_rel']:.2e}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 4. Post-Avalanche DOP853 vs Radau Kill-Test (STRAIGHT ACROSS N_CS: 0.408 -> 0.0!)
    #    & Complete-Avalanche N_max = 20, 24, 30, 36 Convergence Check!
    # ------------------------------------------------------------------
    print("\n=== 4. Post-Avalanche DOP853 vs. Radau & N_max=20..36 Convergence Across FULL Decay ===", flush=True)
    # Use the complete expanding avalanche (phi0=5e-2, M_W=5.0, kap=2e-4) integrated to eta_end = res_exp["eta_final"]!
    eta_full = res_exp["eta_final"]
    bas20 = build_equivariant_basis(U1, k_winding=7, max_n=20, poly_coeffs=poly_coeffs)
    c_flat20, _, _ = relax_flat_rho2(bas20)
    eng20 = RealtimeEquivariantYMHiggsCosmology(bas20, kappa_YM=2e-4, M_W=5.0, beta_H=beta_H)
    y0_20, _ = eng20.build_initial_state(
        a0=1.10, c0=c_flat20, cp0=np.zeros(eng20.D_A),
        phi0=5e-2 * bas20["phi_rho2_ground"], phip0=np.zeros(eng20.D_Phi), contracting=False
    )
    res_dop = eng20.integrate(y0_20, (0.0, eta_full), method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.004, n_eval=100)
    res_rad = eng20.integrate(y0_20, (0.0, eta_full), method="Radau", rtol=1e-10, atol=1e-12, max_step=0.008, n_eval=100)
    yd = res_dop["sol"].y[:, -1]
    yr = res_rad["sol"].y[:, -1]
    rel_diff_post = float(np.linalg.norm(yd - yr) / np.linalg.norm(yd))
    ncs_diff_post = abs(res_dop["trace"][-1]["N_CS"] - res_rad["trace"][-1]["N_CS"])
    print(
        f"  Post-avalanche DOP853 vs Radau (N=20, eta={eta_full:.4f}, a={res_dop['a_final']:.2f}, "
        f"N_CS={res_dop['trace'][-1]['N_CS']:+.4f}): rel state diff = {rel_diff_post:.3e}, "
        f"|dN_CS| = {ncs_diff_post:.3e} (Radau H_res={res_rad['max_H_res_rel']:.2e})",
        flush=True,
    )

    conv_full = []
    for max_n in [20, 24, 30, 36]:
        bas = build_equivariant_basis(U1, k_winding=7, max_n=max_n, poly_coeffs=poly_coeffs)
        c_fl, E_fl, _ = relax_flat_rho2(bas)
        eng = RealtimeEquivariantYMHiggsCosmology(bas, kappa_YM=2e-4, M_W=5.0, beta_H=beta_H)
        y0, _ = eng.build_initial_state(
            a0=1.10, c0=c_fl, cp0=np.zeros(eng.D_A),
            phi0=5e-2 * bas["phi_rho2_ground"], phip0=np.zeros(eng.D_Phi), contracting=False
        )
        res = eng.integrate(y0, (0.0, eta_full), method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.004, n_eval=120)
        tr = res["trace"]
        g0_m, g20_m = 0.0, 0.0
        for k_i in np.linspace(0, res["sol"].y.shape[1] - 1, 50, dtype=int):
            a0g, a20g = gauss_low_n_norm(eng, res["sol"].y[:, k_i], n_cut=20)
            g0_m = max(g0_m, a0g)
            g20_m = max(g20_m, a20g)
        rec = {
            "N_max": max_n,
            "D_A": eng.D_A,
            "D_Phi": eng.D_Phi,
            "E_flat_init": E_fl,
            "a_final": res["a_final"],
            "N_CS_min": min(r["N_CS"] for r in tr),
            "N_CS_final": tr[-1]["N_CS"],
            "phi_rms_final": tr[-1]["phi_rms"],
            "E_YM_peak_S3I": max(r["E_YM_S3I"] for r in tr),
            "max_delta_T00_tot": res["max_delta_T00_tot"],
            "max_H_res_rel": res["max_H_res_rel"],
            "max_gauss_n0": g0_m,
            "max_gauss_n20": g20_m,
        }
        conv_full.append(rec)
        print(
            f"  Full-decay N={max_n:2d}: a_end={rec['a_final']:.4f} | N_CS min={rec['N_CS_min']:+.4f} end={rec['N_CS_final']:+.4f} | "
            f"E_YM_pk={rec['E_YM_peak_S3I']:.2f} | H_res={rec['max_H_res_rel']:.2e} | "
            f"G_n0={g0_m:.2e} | G_n20={g20_m:.2e}",
            flush=True,
        )

    payload = {
        "complete_expanding_avalanche": {
            "M_W": 5.0,
            "kappa_YM": 2e-4,
            "a_restore": 3.0 * math.sqrt(14.0) / (beta_H * 5.0),
            "a_av_mid": a_mid,
            "a_av_below_0p05": a_zero,
            "N_CS_init": tr_exp[0]["N_CS"],
            "N_CS_min": ncs_min,
            "N_CS_final": tr_exp[-1]["N_CS"],
            "phi_rms_final": tr_exp[-1]["phi_rms"],
            "E_YM_peak_S3I": E_ym_pk,
            "max_delta_T00_matter": res_exp["max_delta_T00_matter"],
            "max_delta_T00_tot": res_exp["max_delta_T00_tot"],
            "max_H_res_rel": res_exp["max_H_res_rel"],
            "trace_downsampled": tr_exp[::2],
        },
        "prebounce_contracting_gamma_c_bifurcation": {
            "M_W": 5.0,
            "a0": 4.5,
            "sweep": contract_sweep,
            "trace_bounce": tr_contract_bounce[::2] if tr_contract_bounce else None,
            "trace_crunch": tr_contract_crunch[::2] if tr_contract_crunch else None,
        },
        "subhorizon_supercooling_vs_recollapse": {
            "M_W": 22.0,
            "kappa_YM": 1.15e-4,
            "a_restore": 3.0 * math.sqrt(14.0) / (beta_H * 22.0),
            "sweep": sub_seed_rows,
            "trace_recollapse": tr_sub_recollapse[::2] if tr_sub_recollapse else None,
            "trace_escape": tr_sub_escape[::2] if tr_sub_escape else None,
        },
        "post_avalanche_dop853_vs_radau": {
            "N_max": 20,
            "eta_end": eta_full,
            "a_final": res_dop["a_final"],
            "N_CS_final_dop": res_dop["trace"][-1]["N_CS"],
            "N_CS_final_rad": res_rad["trace"][-1]["N_CS"],
            "rel_state_diff": rel_diff_post,
            "abs_N_CS_diff": ncs_diff_post,
            "dop_H_res": res_dop["max_H_res_rel"],
            "rad_H_res": res_rad["max_H_res_rel"],
        },
        "full_decay_N_max_convergence": conv_full,
        "elapsed_sec": round(time.time() - t0, 2),
    }

    out_path = os.path.join(OUT_DIR, "t4b_03b_avalanche_bifurcations.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {out_path} in {payload['elapsed_sec']}s.", flush=True)


if __name__ == "__main__":
    main()
