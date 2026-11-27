"""T4B-06 Final Referee Audit:
1. Sub-horizon supercooling N_max = 24 -> 30 convergence, exact barrier-crossing
   vs. post-avalanche trough vs. late-time gamma_YM / gamma_tot, and local G_rel.
2. Section 4.1 strong-seed (E_init=2500, M_W=4.0, eps_12=0.1) Pure-YM / SM / Stiff
   N_max = 20 -> 24 convergence and local G_rel audit.
3. Section 4.3 strong-seed pure-YM saturation (E_YM=2843.9, eps_12=0.1)
   N_max = 24 -> 30 convergence and local G_rel audit.
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
from t4b_03_dynamical_phenomena import (  # noqa: E402
    compute_ym_energy_by_harmonic,
    find_k1_n12_k4_mode,
    relax_flat_rho2,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def canonical_v_k4(eng: RealtimeEquivariantYMHiggsCosmology, bas: dict) -> np.ndarray:
    v = find_k1_n12_k4_mode(eng)
    c_f1 = 2.0 * bas["c_hom_id"]
    ep, _, _ = eng.ym.energy_and_grad(c_f1 + 0.1 * v)
    em, _, _ = eng.ym.energy_and_grad(c_f1 - 0.1 * v)
    return v if ep >= em else -v


def dense_local_gauss(eng: RealtimeEquivariantYMHiggsCosmology, sol, n_pts: int = 80) -> dict:
    idxs = np.unique(np.linspace(0, sol.y.shape[1] - 1, min(n_pts, sol.y.shape[1]), dtype=int))
    max_grel = 0.0
    max_gn0 = 0.0
    for idx in idxs:
        d = eng.eval_diagnostics(sol.y[:, idx])
        max_grel = max(max_grel, float(d["gauss_l2_rel"]))
        max_gn0 = max(max_gn0, float(d["gauss_n0_abs"]))
    return {"max_gauss_rel": max_grel, "max_gauss_n0": max_gn0}


def main():
    t0 = time.time()
    beta_H = 1.556
    _, _, U1, U2 = build_istar_and_galois()
    _, poly_coeffs = build_p7_equivariant_polynomial(U1, U2)

    out: dict = {}

    # ------------------------------------------------------------------
    # 1. Sub-horizon supercooling N_max = 24 -> 30 + exact barrier diagnostics
    # ------------------------------------------------------------------
    print("=== 1. Sub-Horizon Supercooling N_max = 24 -> 30 ===", flush=True)
    sub_rows = []
    for nmax in [24, 30]:
        bas = build_equivariant_basis(U1, k_winding=7, max_n=nmax, poly_coeffs=poly_coeffs)
        c_fl, E_fl, _ = relax_flat_rho2(bas)
        eng = RealtimeEquivariantYMHiggsCosmology(bas, kappa_YM=1.15e-4, M_W=22.0, beta_H=beta_H)
        y0, _ = eng.build_initial_state(
            a0=0.305, c0=c_fl, cp0=np.zeros(eng.D_A),
            phi0=0.05 * bas["phi_rho2_ground"], phip0=np.zeros(eng.D_Phi), contracting=False
        )
        res = eng.integrate(
            y0, (0.0, 4.0), method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.0025,
            a_stop_hi=2.5, a_stop_lo=0.05, n_eval=300
        )
        tr = res["trace"]
        g_aud = dense_local_gauss(eng, res["sol"], n_pts=100)
        a_av_mid = next((r["a"] for r in tr if r["N_CS"] < 0.20), None)
        r_bar = min(tr, key=lambda r: abs(r["a"] - 1.0 / math.sqrt(2.0)))
        idx_tr = min(range(len(tr)), key=lambda i: tr[i]["N_CS"] if tr[i]["a"] < 1.2 else 99.0)
        r_tr = tr[idx_tr]
        row = {
            "N_max": nmax,
            "D_A": eng.D_A,
            "D_Phi": eng.D_Phi,
            "escaped": bool(res["escaped"]),
            "a_av_mid": a_av_mid,
            "R_sim": a_av_mid / (math.sqrt(126.0) / (beta_H * 22.0)) if a_av_mid else None,
            "at_barrier": {
                "a": r_bar["a"], "gamma_ym": r_bar["gamma_ym"], "gamma_tot": r_bar["gamma_tot"],
                "N_CS": r_bar["N_CS"],
            },
            "at_first_trough": {
                "a": r_tr["a"], "gamma_ym": r_tr["gamma_ym"], "gamma_tot": r_tr["gamma_tot"],
                "N_CS": r_tr["N_CS"],
            },
            "gamma_ym_peak": max(r["gamma_ym"] for r in tr),
            "gamma_tot_peak": max(r["gamma_tot"] for r in tr),
            "max_H_res_rel": res["max_H_res_rel"],
            **g_aud,
        }
        sub_rows.append(row)
        print(
            f"  N={nmax}: a_av_mid={a_av_mid:.5f} (R={row['R_sim']:.4f}) | "
            f"barrier(a={r_bar['a']:.3f}): gam_ym={r_bar['gamma_ym']:.4f}, gam_tot={r_bar['gamma_tot']:.4f} | "
            f"trough(a={r_tr['a']:.3f}): gam_ym={r_tr['gamma_ym']:.4f}, gam_tot={r_tr['gamma_tot']:.4f} | "
            f"G_rel={g_aud['max_gauss_rel']:.2e}",
            flush=True,
        )
    out["subhorizon_N24_vs_N30"] = sub_rows

    # ------------------------------------------------------------------
    # 2. Section 4.1 strong-seed triplet (2500, 4.0, eps=0.1) N_max = 20 -> 24
    # ------------------------------------------------------------------
    print("\n=== 2. Sec 4.1 Strong-Seed Triplet (2500, 4.0, eps=0.1) N_max = 20 -> 24 ===", flush=True)
    sec41_rows = []
    for nmax in [20, 24]:
        bas = build_equivariant_basis(U1, k_winding=1, max_n=nmax)
        eng_tmp = RealtimeEquivariantYMHiggsCosmology(bas, kappa_YM=0.10 / 2500.0, M_W=4.0, beta_H=beta_H)
        v_k4 = canonical_v_k4(eng_tmp, bas)
        for MW_val, bH, label in [(0.0, 1.556, "E2500_pureYM"), (4.0, 1.556, "E2500_SM_MW4.0"), (4.0, 20.0, "E2500_Stiff_MW4.0")]:
            eng = RealtimeEquivariantYMHiggsCosmology(bas, kappa_YM=0.10 / 2500.0, M_W=MW_val, beta_H=bH)
            U_init = 0.5 + 0.5 * ((2.0 * MW_val) ** 2)
            fp0 = math.sqrt(max(0.0, 2.0 * (2500.0 - U_init)))
            c0 = bas["c_hom_id"] + (0.1 * math.sqrt(3.0)) * v_k4
            cp0 = fp0 * bas["c_hom_id"]
            phi0 = bas["phi_vac"].copy()
            phip0 = np.zeros(eng.D_Phi)
            y0, _ = eng.build_initial_state(a0=2.0, c0=c0, cp0=cp0, phi0=phi0, phip0=phip0, contracting=True)
            res = eng.integrate(y0, (0.0, 12.0), method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.015, a_stop_hi=6.0)
            sol = res["sol"]
            c_tr = sol.y[2 : 2 + eng.D_A, :]
            cp_tr = sol.y[2 + eng.D_A : 2 + 2 * eng.D_A, :]
            peak_fE = 0.0
            for k_idx in np.linspace(0, sol.y.shape[1] - 1, min(60, sol.y.shape[1]), dtype=int):
                pf = compute_ym_energy_by_harmonic(eng, c_tr[:, k_idx], cp_tr[:, k_idx])["inhom_energy_frac"]
                peak_fE = max(peak_fE, pf)
            end_fE = compute_ym_energy_by_harmonic(eng, c_tr[:, -1], cp_tr[:, -1])["inhom_energy_frac"]
            g_aud = dense_local_gauss(eng, sol, n_pts=80)
            row = {
                "N_max": nmax, "label": label, "peak_inhom_E_frac": peak_fE, "final_inhom_E_frac": end_fE,
                "max_H_res_rel": res["max_H_res_rel"], **g_aud,
            }
            sec41_rows.append(row)
            print(
                f"  N={nmax} {label:18s}: pk={peak_fE:.5f} end={end_fE:.5f} | "
                f"H_res={res['max_H_res_rel']:.2e} | G_rel={g_aud['max_gauss_rel']:.2e}",
                flush=True,
            )
    out["sec41_strong_seed_N20_vs_N24"] = sec41_rows

    # ------------------------------------------------------------------
    # 3. Section 4.3 strong-seed pure-YM saturation (2843.9, eps=0.1) N_max = 24 -> 30
    # ------------------------------------------------------------------
    print("\n=== 3. Sec 4.3 Strong-Seed Pure-YM Saturation (2843.9, eps=0.1) N_max = 24 -> 30 ===", flush=True)
    sec43_rows = []
    for nmax in [24, 30]:
        bas = build_equivariant_basis(U1, k_winding=1, max_n=nmax)
        eng = RealtimeEquivariantYMHiggsCosmology(bas, kappa_YM=0.10 / 2843.9, M_W=0.0, beta_H=beta_H)
        v_k4 = canonical_v_k4(eng, bas)
        fp0 = math.sqrt(2.0 * (2843.9 - 0.5))
        c0 = bas["c_hom_id"] + (0.1 * math.sqrt(3.0)) * v_k4
        cp0 = fp0 * bas["c_hom_id"]
        y0, _ = eng.build_initial_state(
            a0=2.0, c0=c0, cp0=cp0, phi0=np.zeros(eng.D_Phi), phip0=np.zeros(eng.D_Phi), contracting=True
        )
        res = eng.integrate(
            y0, (0.0, 6.0), method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.004, a_stop_hi=6.0, n_eval=250
        )
        sol = res["sol"]
        c_tr = sol.y[2 : 2 + eng.D_A, :]
        cp_tr = sol.y[2 + eng.D_A : 2 + 2 * eng.D_A, :]
        part_end = compute_ym_energy_by_harmonic(eng, c_tr[:, -1], cp_tr[:, -1])
        gam_vals = [r["gamma_ym"] for r in res["trace"]]
        gam_drift = max(abs(g - gam_vals[0]) for g in gam_vals) / gam_vals[0]
        g_aud = dense_local_gauss(eng, sol, n_pts=80)
        row = {
            "N_max": nmax, "D_A": eng.D_A, "a_min": res["a_min"],
            "final_inhom_E_frac": part_end["inhom_energy_frac"],
            "final_kin_frac_by_n": part_end["kin_frac_by_n"],
            "conformal_gamma_YM_drift": gam_drift,
            "max_H_res_rel": res["max_H_res_rel"],
            **g_aud,
        }
        sec43_rows.append(row)
        print(
            f"  N={nmax}: a_min={res['a_min']:.6f} end_fE={part_end['inhom_energy_frac']:.5f} | "
            f"kin_nmax={part_end['kin_frac_by_n'][str(nmax)]:.4f} | gam_drift={gam_drift:.2e} | "
            f"G_rel={g_aud['max_gauss_rel']:.2e}",
            flush=True,
        )
    out["sec43_strong_sat_N24_vs_N30"] = sec43_rows
    out["elapsed_sec"] = round(time.time() - t0, 2)

    out_path = os.path.join(OUT_DIR, "t4b_06_final_referee_audit.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved {out_path} in {out['elapsed_sec']}s.", flush=True)


if __name__ == "__main__":
    main()
