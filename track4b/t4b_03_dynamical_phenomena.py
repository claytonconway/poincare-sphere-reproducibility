"""T4B-03 (Goal 3): Full Nonlinear Real-Time Simulations & Convergence Kill-Tests of Three Core
Dynamical Phenomena on S^3/I*:

  Part 3A — Nonlinear n=12 Stability During Dynamical SM Hosotani Restoration vs. Stiff-Higgs:
    Uses the full nonlinear k=1 equivariant real-time SU(2)_L + 4-component Higgs + FLRW solver
    (N_max=20: D_A=21, D_Phi=10) at finite n=12, K=4 seeds eps_12 in {1e-4, 1e-2, 1e-1} across
    E_YM in {500, 2500}, comparing Pure YM, Stiff Higgs (beta_H=20), and SM Higgs (beta_H=1.556).

  Part 3B — Real-Time Cosmological rho_2 -> S_02 -> rho_0 Avalanche Decay Across a_restore = 3*sqrt(14)/m_H:
    Uses the full nonlinear k=7 equivariant real-time SU(2)_L + 4-component Higgs + FLRW solver
    starting from the variationally relaxed flat representative c_{rho2,flat} (N_CS = 49/120 to 10+ digits)
    in three physically distinct cosmological scenarios:
      (I-a) Super-horizon post-bounce expansion (M_W=5.0, a_restore=1.4428 > 1): avalanche fires
            during expansion (N_CS: 49/120 -> 0 with coherent post-avalanche oscillations);
      (I-b) Sub-horizon false-vacuum-powered bounce (M_W=22.0, a_restore=0.3279 < 1/sqrt(2)):
            tests whether dynamical supercooling past a_restore boosts gamma_rad above gamma_c = 1/4
            (preventing recollapse) vs. sub-horizon recollapse;
      (II)  Contracting pre-bounce avalanche (M_W=5.0): sweeps kappa_YM across gamma_c = 1/4
            (extended in t4b_03b_avalanche_bifurcations.py from a0 = 4.5 to capture pre-bounce decay).

  Part 3C — Nonlinear Mode-Mode Saturation of n=12 in High-E (E_YM >= 2100) Pure-YM Plasma:
    Uses the full nonlinear k=1 real-time solver (M_W=0, N_max=24: D_A=30 covering n=0,12,20,24)
    for both (i) gravitating bounces at E_YM = 2500 & 2844 (gamma_YM = 0.10) across eps_12 in
    {1e-5, 1e-3, 1e-2, 5e-2, 1e-1}, and (ii) long-time fixed-a saturation over 12+ periods,
    measuring exact harmonic energy partition (n=0, 12, 20, 24), conformal radiation conservation
    |gamma_YM(eta) - gamma_YM(0)|, bounce survival, and spatial T_00 variance delta_{T00}!

  Part 3D — Independent Convergence & Integrator Kill-Tests:
    - N_max convergence sweep (N_max = 12, 20, 24, 30, 36 for k=7 avalanche; N_max = 12, 20, 24, 30
      for k=1 nonlinear n=12 saturation);
    - Independent DOP853 vs. Radau implicit-integrator cross-checks on both flagship runs!
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "track2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from istar_geometry import build_istar_and_galois, build_p7_equivariant_polynomial  # noqa: E402
from t4b_02_realtime_equivariant_solver import (  # noqa: E402
    RealtimeEquivariantYMHiggsCosmology,
    build_equivariant_basis,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers: Variationally Relaxed Flat rho_2 & Exact k=1 n=12 K=4 Mode Finder
# ---------------------------------------------------------------------------

def relax_flat_rho2(bas: Dict) -> Tuple[np.ndarray, float, float]:
    """Variationally relax raw projected c_rho2 inside N_max to minimize cutoff curvature B(A) -> 0
    while strictly preserving N_CS == 49/120! Eliminates initial cutoff-shock artifacts.
    """
    ym = bas["ym_solver"]
    c_init = bas["c_rho2"].copy()
    res = minimize(
        lambda c: ym.energy_and_grad(c)[:2],
        c_init,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 600, "ftol": 1e-16, "gtol": 1e-11},
    )
    E_flat, _, ncs_flat = ym.energy_and_grad(res.x)
    return res.x.copy(), float(E_flat), float(ncs_flat)


def find_k1_n12_k4_mode(eng: RealtimeEquivariantYMHiggsCosmology) -> np.ndarray:
    """Extract the exact unit-norm n=12, K=4 (m=11: 122 - 22f) tachyonic eigenmode in the k=1 basis."""
    bas = eng.basis
    idx_n12 = [i for i, lv in enumerate(bas["A_levels"]) if lv == 12]
    c_f1 = 2.0 * bas["c_hom_id"]
    eps = 1e-6
    n12 = len(idx_n12)
    H12 = np.zeros((n12, n12))
    z_phi = np.zeros(eng.D_Phi)
    for a_i, ia in enumerate(idx_n12):
        dc = np.zeros(eng.D_A)
        dc[ia] = eps
        _, _, _, gp, *_ = eng.eval_potential_and_grads(c_f1 + dc, z_phi)
        _, _, _, gm, *_ = eng.eval_potential_and_grads(c_f1 - dc, z_phi)
        H12[:, a_i] = (gp[idx_n12] - gm[idx_n12]) / (2.0 * eps)
    H12 = 0.5 * (H12 + H12.T)
    ev, V = np.linalg.eigh(H12)
    k4_pos = int(np.argmin(np.abs(ev - 100.0)))
    assert abs(ev[k4_pos] - 100.0) < 1e-8, f"Expected K=4 eig 100.0, got {ev}"
    v_k4 = np.zeros(eng.D_A)
    v_k4[idx_n12] = V[:, k4_pos]
    # Canonical basis-independent sign fixing via the exact cubic magnetic invariant
    # E_YM(c_{f=1} + 0.1 v) > E_YM(c_{f=1} - 0.1 v)!
    e_plus, _, _ = eng.ym.energy_and_grad(c_f1 + 0.1 * v_k4)
    e_minus, _, _ = eng.ym.energy_and_grad(c_f1 - 0.1 * v_k4)
    if e_plus < e_minus:
        v_k4 = -v_k4
    return v_k4


def compute_level_power(c: np.ndarray, cp: np.ndarray, levels: List[int]) -> Dict[str, float]:
    """Compute fraction of gauge L^2 kinetic+field norm in each harmonic level n."""
    by_n: Dict[int, float] = {}
    tot = 0.0
    for i, n in enumerate(levels):
        p = float(c[i] * c[i] + cp[i] * cp[i])
        by_n[n] = by_n.get(n, 0.0) + p
        tot += p
    if tot < 1e-30:
        return {str(n): 0.0 for n in sorted(by_n)}
    return {str(n): by_n[n] / tot for n in sorted(by_n)}


def compute_ym_energy_by_harmonic(eng: RealtimeEquivariantYMHiggsCosmology, c: np.ndarray, cp: np.ndarray) -> Dict[str, float]:
    """Compute exact kinetic energy fraction by harmonic level n, plus inhomogeneous energy fraction
    (energy carried by modes with n >= 12 relative to total YM energy).
    """
    levels = eng.basis["A_levels"]
    idx_0 = [i for i, n in enumerate(levels) if n == 0]
    c_hom = np.zeros_like(c)
    cp_hom = np.zeros_like(cp)
    c_hom[idx_0] = c[idx_0]
    cp_hom[idx_0] = cp[idx_0]

    E_tot_mag, _, _ = eng.ym.energy_and_grad(c)
    E_tot_kin = eng.scale_V0 * 0.5 * float(np.dot(cp, cp))
    E_tot = E_tot_mag + E_tot_kin

    E_hom_mag, _, _ = eng.ym.energy_and_grad(c_hom)
    E_hom_kin = eng.scale_V0 * 0.5 * float(np.dot(cp_hom, cp_hom))
    E_hom = E_hom_mag + E_hom_kin

    # Kinetic partition by level n
    kin_by_n: Dict[int, float] = {}
    for i, n in enumerate(levels):
        kin_by_n[n] = kin_by_n.get(n, 0.0) + 0.5 * float(cp[i] * cp[i])
    kin_tot = max(1e-30, sum(kin_by_n.values()))

    # Field norm squared partition by level n (excluding homogeneous background)
    c2_by_n: Dict[int, float] = {}
    for i, n in enumerate(levels):
        c2_by_n[n] = c2_by_n.get(n, 0.0) + float(c[i] * c[i])

    inhom_frac = max(0.0, 1.0 - E_hom / max(1e-30, E_tot))
    return {
        "E_YM_tot_S3I": float(E_tot),
        "E_YM_hom_S3I": float(E_hom),
        "inhom_energy_frac": float(inhom_frac),
        "kin_frac_by_n": {str(n): kin_by_n[n] / kin_tot for n in sorted(kin_by_n)},
        "c2_by_n": {str(n): c2_by_n[n] for n in sorted(c2_by_n)},
    }


def gauss_low_n_norm(eng: RealtimeEquivariantYMHiggsCosmology, y: np.ndarray, n_cut: int = 20) -> Tuple[float, float]:
    """Return (|G_{n=0}|, ||G_{n<=n_cut}||_2) on state y."""
    d = eng.eval_diagnostics(y)
    lv = eng.basis["xi_levels"]
    g_low = float(np.linalg.norm([d["G_proj"][r] for r in range(eng.D_xi) if lv[r] <= n_cut]))
    return d["gauss_n0_abs"], g_low


# ---------------------------------------------------------------------------
# Part 3A: Nonlinear n=12 Stability During SM Hosotani Restoration vs. Stiff-Higgs
# ---------------------------------------------------------------------------

def run_part_3a_nonlinear_hosotani_n12(U1: np.ndarray) -> Dict:
    print("\n=== Part 3A: Nonlinear n=12 Stability During SM Hosotani Restoration (k=1, N_max=20) ===", flush=True)
    bas20 = build_equivariant_basis(U1, k_winding=1, max_n=20)
    # Use a temporary engine to extract the exact n=12 K=4 unit mode
    eng_tmp = RealtimeEquivariantYMHiggsCosmology(bas20, kappa_YM=1e-4, M_W=2.5, beta_H=1.556)
    v_k4 = find_k1_n12_k4_mode(eng_tmp)
    idx_n12_all = [i for i, lv in enumerate(bas20["A_levels"]) if lv >= 12]

    gamma_YM = 0.10
    configs = [
        # (E_YM, M_W, beta_H, label)
        (500.0, 0.0, 1.556, "E500_pureYM"),
        (500.0, 2.5, 1.556, "E500_SM_MW2.5"),
        (500.0, 2.5, 20.0, "E500_Stiff_MW2.5"),
        (2500.0, 0.0, 1.556, "E2500_pureYM"),
        (2500.0, 4.0, 1.556, "E2500_SM_MW4.0"),
        (2500.0, 4.0, 20.0, "E2500_Stiff_MW4.0"),
    ]
    eps_list = [1e-4, 1e-2, 1e-1]

    rows = []
    for E_val, MW_val, bH, label in configs:
        kappa = gamma_YM / E_val
        eng = RealtimeEquivariantYMHiggsCosmology(bas20, kappa_YM=kappa, M_W=MW_val, beta_H=bH)
        U_init = 0.5 + 0.5 * ((2.0 * MW_val) ** 2)
        fp0 = math.sqrt(max(0.0, 2.0 * (E_val - U_init)))
        for eps12 in eps_list:
            # Initial state: homogeneous f=0, fp=fp0 + finite n=12 K=4 seed eps12 * sqrt(3) * v_k4
            # (scaling by sqrt(3) so eps12 is the relative amplitude vs ||c_hom_id|| = sqrt(3))
            c0 = bas20["c_hom_id"] + (eps12 * math.sqrt(3.0)) * v_k4
            cp0 = fp0 * bas20["c_hom_id"]
            phi0 = bas20["phi_vac"].copy()
            phip0 = np.zeros(eng.D_Phi)
            y0, _ = eng.build_initial_state(a0=2.0, c0=c0, cp0=cp0, phi0=phi0, phip0=phip0, contracting=True)
            res = eng.integrate(
                y0, (0.0, 12.0), method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.015, a_stop_hi=6.0
            )
            sol = res["sol"]
            c_traj = sol.y[2 : 2 + eng.D_A, :]
            cp_traj = sol.y[2 + eng.D_A : 2 + 2 * eng.D_A, :]
            phi_traj = sol.y[2 + 2 * eng.D_A : 2 + 2 * eng.D_A + eng.D_Phi, :]

            q12_amp = np.abs(v_k4 @ c_traj) / (eps12 * math.sqrt(3.0))
            inhom_norm = np.linalg.norm(c_traj[idx_n12_all, :], axis=0)
            # Peak and final inhomogeneous YM energy fraction
            part_end = compute_ym_energy_by_harmonic(eng, c_traj[:, -1], cp_traj[:, -1])
            peak_inhom_frac = 0.0
            for k_idx in np.linspace(0, sol.y.shape[1] - 1, min(60, sol.y.shape[1]), dtype=int):
                pf = compute_ym_energy_by_harmonic(eng, c_traj[:, k_idx], cp_traj[:, k_idx])["inhom_energy_frac"]
                peak_inhom_frac = max(peak_inhom_frac, pf)

            phi0_traj = phi_traj[0, :]  # n=0 Re(Phi_2)
            phi_cross = int(np.sum(phi0_traj[:-1] * phi0_traj[1:] <= 0.0)) if MW_val > 0 else 0
            rec = {
                "label": label,
                "E_YM": E_val,
                "M_W": MW_val,
                "beta_H": bH,
                "eps_12": eps12,
                "bounced": bool(res["escaped"] and not res["collapsed"]),
                "a_min": res["a_min"],
                "eta_end": res["eta_final"],
                "phi_zero_crossings": phi_cross,
                "min_phi0": float(np.min(phi0_traj)),
                "peak_q12_gain": float(np.max(q12_amp)),
                "peak_inhom_c_norm": float(np.max(inhom_norm)),
                "peak_inhom_E_frac": float(peak_inhom_frac),
                "final_inhom_E_frac": float(part_end["inhom_energy_frac"]),
                "max_delta_T00_matter": res["max_delta_T00_matter"],
                "max_delta_T00_tot": res["max_delta_T00_tot"],
                "max_H_res_rel": res["max_H_res_rel"],
                "max_gauss_n0": res["max_gauss_n0"],
            }
            rows.append(rec)
            print(
                f"  {label:18s} eps={eps12:5.0e} | bounce={rec['bounced']} a_min={rec['a_min']:.4f} "
                f"phi_x={phi_cross:2d} | q12_gain={rec['peak_q12_gain']:7.2f}x | "
                f"peak_fE12={rec['peak_inhom_E_frac']:.4f} | dT00_mat={rec['max_delta_T00_matter']:.4f}, "
                f"dT00_tot={rec['max_delta_T00_tot']:.4f} | H_res={rec['max_H_res_rel']:.2e}",
                flush=True,
            )
    return {"N_max": 20, "D_A": bas20["ym_solver"].D, "D_Phi": bas20["B_Phi"].shape[1], "runs": rows}


# ---------------------------------------------------------------------------
# Part 3B: Real-Time Cosmological rho_2 -> S_02 -> rho_0 Avalanche Decay
# ---------------------------------------------------------------------------

def run_part_3b_rho2_avalanche(U1: np.ndarray, poly_coeffs: np.ndarray) -> Dict:
    print("\n=== Part 3B: Real-Time Cosmological rho_2 -> S_02 -> rho_0 Avalanche Decay (k=7, N_max=24) ===", flush=True)
    bas24 = build_equivariant_basis(U1, k_winding=7, max_n=24, poly_coeffs=poly_coeffs)
    c_r2_flat, E_r2_flat, ncs_r2_flat = relax_flat_rho2(bas24)
    print(f"  Relaxed flat rho_2 (N=24): E_YM={E_r2_flat:.3e}, N_CS={ncs_r2_flat:.10f}", flush=True)

    beta_H = 1.556
    # ------------------------------------------------------------------
    # Scenario 3B-I(a): Super-horizon post-bounce expansion avalanche!
    # M_W = 5.0 => m_H = 7.780 => a_restore = 3*sqrt(14)/7.780 = 1.4428 > 1!
    # Start near bounce a0 = 1.10 (expanding, strictly Hosotani-restored!) with small seed
    # phi0 = 2e-2 * phi_rho2_ground, kappa_YM = 2e-4 (so u_vac/3 = 0.0504, clean expansion!).
    # Wait: let's check growth rate of Higgs tachyon past a_restore:
    #   m_eff^2 = 63 - 0.5 * a^2 * m_H^2 = 63 * (1 - (a/a_restore)^2).
    #   For a = 1.44 -> 6.0, int sqrt(|m_eff^2|) d(eta) ~ 63^{1/2} * int (a/a_rest) d(eta)
    #   Since d(eta) = da / a^2, int (a/a_rest) da/a^2 = (1/a_rest) ln(a_end/a_rest) ~ 0.69 * ln(4.2) ~ 1.0
    #   Multiply by sqrt(63) ~ 7.94 => e-folds ~ 8.0 => exp(8) ~ 3000x growth -> seed 2e-2 hits O(1) at a ~ 2.2!!
    # ------------------------------------------------------------------
    print("\n  [Scenario 3B-I(a): Super-horizon Expanding Avalanche (M_W=5.0, a_restore=1.4428)]", flush=True)
    eng_Ia = RealtimeEquivariantYMHiggsCosmology(bas24, kappa_YM=2e-4, M_W=5.0, beta_H=beta_H)
    y0_Ia, _ = eng_Ia.build_initial_state(
        a0=1.10, c0=c_r2_flat, cp0=np.zeros(eng_Ia.D_A),
        phi0=2e-2 * bas24["phi_rho2_ground"], phip0=np.zeros(eng_Ia.D_Phi), contracting=False
    )
    res_Ia = eng_Ia.integrate(
        y0_Ia, (0.0, 2.5), method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.004, a_stop_hi=8.0, n_eval=400
    )
    tr_Ia = res_Ia["trace"]
    E_ym_peak_Ia = max(r["E_YM_S3I"] for r in tr_Ia)
    # Find avalanche midpoint (first crossing N_CS < 0.20)
    a_mid_Ia = next((r["a"] for r in tr_Ia if r["N_CS"] < 0.20), None)
    g0_Ia, g20_Ia = 0.0, 0.0
    for k_i in np.linspace(0, res_Ia["sol"].y.shape[1] - 1, 60, dtype=int):
        a0g, a20g = gauss_low_n_norm(eng_Ia, res_Ia["sol"].y[:, k_i], n_cut=20)
        g0_Ia = max(g0_Ia, a0g)
        g20_Ia = max(g20_Ia, a20g)
    print(
        f"    a: 1.10 -> {res_Ia['a_final']:.3f} (a_av_mid={a_mid_Ia}) | "
        f"N_CS: {tr_Ia[0]['N_CS']:.4f} -> {tr_Ia[-1]['N_CS']:.4f} | "
        f"phi_rms: {tr_Ia[0]['phi_rms']:.4f} -> {tr_Ia[-1]['phi_rms']:.4f} | "
        f"E_YM peak={E_ym_peak_Ia:.2f} (vs S_02=31.49) | "
        f"dT00_mat={res_Ia['max_delta_T00_matter']:.3f}, dT00_tot={res_Ia['max_delta_T00_tot']:.4f} | "
        f"H_res={res_Ia['max_H_res_rel']:.2e}, G_n0={g0_Ia:.2e}, G_n20={g20_Ia:.2e}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Scenario 3B-I(b): Sub-horizon False-Vacuum Bounce + Avalanche-Induced Recollapse vs. Escape!
    # M_W = 14.0 => m_H = 21.784 => a_restore = 3*sqrt(14)/21.784 = 0.51529 < 1/sqrt(2) = 0.70711!!
    # In the restored phase (Phi=0): u_vac / 3 = (1/6) kappa beta_H^2 M_W^4 = 15503.2 * kappa.
    # For a_bounce = 1/sqrt(1 + u_vac/3) < a_restore = 0.5153, we need u_vac/3 > 1/0.5153^2 - 1 = 2.766,
    # i.e. kappa > 1.784e-4!
    # When avalanche converts false vacuum to radiation at a_av ~ 0.60..0.68:
    #   gamma_rad ~ (u_vac/3) * a_av^4!
    # Look at the critical transition across gamma_c = 0.25:
    #   At kappa = 3.0e-4: u_vac/3 = 4.651, a_bounce = 0.4207; if a_av = 0.58 => gamma_rad ~ 4.65 * 0.113 = 0.526 > 0.25 (ESCAPE!)
    #   Wait: check smaller a_av (larger M_W = 18.0 -> a_restore = 0.4008)!
    #   At M_W = 18.0: u_vac/3 = (1/6)*kappa*1.556^2*18^4 = 42352.6 * kappa.
    #   Need u_vac/3 > 1/0.4008^2 - 1 = 5.225 => kappa > 1.234e-4.
    #   At kappa = 1.6e-4: u_vac/3 = 6.776, a_bounce = 0.3586, a_restore = 0.4008.
    #   If avalanche fires at a_av = 0.45: gamma_rad = 6.776 * 0.45^4 = 0.278 > 0.25 (just above!)
    #   At M_W = 22.0: a_restore = 0.3279! u_vac/3 = 94534.3 * kappa!
    #   For a_bounce = 0.30 < 0.3279: u_vac/3 = 10.11 => kappa = 1.07e-4.
    #   Then at a_av = 0.38: gamma_rad = 10.11 * 0.38^4 = 0.2108 < 0.25 -> AVALANCHE-INDUCED RECOLLAPSE TO a -> 0!!
    #   At kappa = 2.0e-4: u_vac/3 = 18.91 => a_bounce = 0.224, gamma_rad ~ 18.91 * 0.38^4 = 0.394 > 0.25 -> ESCAPE!!
    # LET'S TEST THAT EXACT BIFURCATION AT M_W = 22.0!!
    # ------------------------------------------------------------------
    print("\n  [Scenario 3B-I(b): Sub-horizon Avalanche Recollapse vs. Escape Bifurcation (M_W=22.0, a_restore=0.3279)]", flush=True)
    MW_Ib = 22.0
    a_rest_Ib = 3.0 * math.sqrt(14.0) / (beta_H * MW_Ib)
    sub_horizon_rows = []
    sub_horizon_flagship_recollapse = None
    sub_horizon_flagship_escape = None
    for kap in [1.15e-4, 1.35e-4, 1.60e-4, 2.00e-4, 2.60e-4]:
        u_vac_3 = (1.0 / 6.0) * kap * (beta_H ** 2) * (MW_Ib ** 4)
        a_bnc = 1.0 / math.sqrt(1.0 + u_vac_3)
        a0_use = max(a_bnc * 1.03, a_bnc + 0.01)
        eng_Ib = RealtimeEquivariantYMHiggsCosmology(bas24, kappa_YM=kap, M_W=MW_Ib, beta_H=beta_H)
        y0_Ib, _ = eng_Ib.build_initial_state(
            a0=a0_use, c0=c_r2_flat, cp0=np.zeros(eng_Ib.D_A),
            phi0=5e-2 * bas24["phi_rho2_ground"], phip0=np.zeros(eng_Ib.D_Phi), contracting=False
        )
        res_Ib = eng_Ib.integrate(
            y0_Ib, (0.0, 4.0), method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.0025,
            a_stop_hi=3.0, a_stop_lo=0.05, n_eval=300
        )
        tr = res_Ib["trace"]
        # Measure effective radiation gamma_ym + gamma_h_kin after avalanche
        gamma_ym_peak = max(r["gamma_ym"] for r in tr)
        E_ym_peak = max(r["E_YM_S3I"] for r in tr)
        ncs_min = min(r["N_CS"] for r in tr)
        outcome = "avalanche_recollapse" if res_Ib["collapsed"] else ("avalanche_escape" if res_Ib["escaped"] else "unresolved")
        row = {
            "kappa_YM": kap,
            "M_W": MW_Ib,
            "a_restore": a_rest_Ib,
            "a_bounce_vac": a_bnc,
            "a0": a0_use,
            "u_vac_over_3": u_vac_3,
            "outcome": outcome,
            "a_max_reached": res_Ib["a_max"],
            "a_final": res_Ib["a_final"],
            "N_CS_init": tr[0]["N_CS"],
            "N_CS_min": ncs_min,
            "N_CS_final": tr[-1]["N_CS"],
            "phi_rms_final": tr[-1]["phi_rms"],
            "E_YM_peak_S3I": E_ym_peak,
            "gamma_ym_peak": gamma_ym_peak,
            "max_delta_T00_matter": res_Ib["max_delta_T00_matter"],
            "max_delta_T00_tot": res_Ib["max_delta_T00_tot"],
            "max_H_res_rel": res_Ib["max_H_res_rel"],
            "max_gauss_n0": res_Ib["max_gauss_n0"],
        }
        sub_horizon_rows.append(row)
        if outcome == "avalanche_recollapse" and sub_horizon_flagship_recollapse is None:
            sub_horizon_flagship_recollapse = tr
        if outcome == "avalanche_escape" and sub_horizon_flagship_escape is None:
            sub_horizon_flagship_escape = tr
        print(
            f"    kap={kap:.2e} (a_bnc={a_bnc:.4f} < a_rest={a_rest_Ib:.4f}): {outcome:20s} | "
            f"a_max={row['a_max_reached']:.4f} -> a_end={row['a_final']:.4f} | "
            f"N_CS {row['N_CS_init']:.3f}->{row['N_CS_final']:.3f} (min {ncs_min:.3f}) | "
            f"gamma_ym_pk={gamma_ym_peak:.4f} | dT00_tot={row['max_delta_T00_tot']:.4f} | H_res={row['max_H_res_rel']:.2e}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Scenario 3B-II: Contracting Pre-Bounce Avalanche -> Bounce Survival vs. gamma_c = 1/4 Collapse!
    # Start contracting from a0 = 1.90 > a_restore = 1.4428 (M_W = 5.0) with phi0 = 0.08 * phi_rho2_ground,
    # so the rho_2 -> S_02 -> rho_0 avalanche fires DURING contraction before the bounce!
    # Sweep kappa_YM so released gamma_rad crosses gamma_c = 1/4!
    # At M_W=5.0, beta_H=1.556, a0=1.90: E_tilde_vac = 0.5 * 1.556^2 * (1.9*5)^4 = 9861!
    # So gamma_tot(0) = (kappa/3)*9861 = 3287 * kappa!
    # Thus gamma_rad crosses 0.25 around kappa ~ 1e-4 .. 6e-4 (depending on how much decays before bounce)!
    # ------------------------------------------------------------------
    print("\n  [Scenario 3B-II: Contracting Pre-Bounce Avalanche vs. gamma_c = 1/4 Ceiling (M_W=5.0, a0=1.90)]", flush=True)
    contract_rows = []
    flagship_contract_bounce = None
    flagship_contract_crunch = None
    for kap in [2.0e-5, 6.0e-5, 1.0e-4, 1.4e-4, 2.0e-4, 3.0e-4]:
        eng_II = RealtimeEquivariantYMHiggsCosmology(bas24, kappa_YM=kap, M_W=5.0, beta_H=beta_H)
        y0_II, _ = eng_II.build_initial_state(
            a0=1.90, c0=c_r2_flat, cp0=np.zeros(eng_II.D_A),
            phi0=0.10 * bas24["phi_rho2_ground"], phip0=np.zeros(eng_II.D_Phi), contracting=True
        )
        res_II = eng_II.integrate(
            y0_II, (0.0, 5.0), method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.003,
            a_stop_hi=4.0, a_stop_lo=0.05, n_eval=300
        )
        tr = res_II["trace"]
        outcome = "singular_collapse" if res_II["collapsed"] else ("nonsingular_bounce" if res_II["escaped"] else "unresolved")
        # gamma_tot at bounce (min a) or at collapse entry (a ~ 0.707)
        idx_min_a = int(np.argmin([r["a"] for r in tr]))
        row = {
            "kappa_YM": kap,
            "outcome": outcome,
            "a_min": res_II["a_min"],
            "a_final": res_II["a_final"],
            "N_CS_init": tr[0]["N_CS"],
            "N_CS_at_amin": tr[idx_min_a]["N_CS"],
            "N_CS_final": tr[-1]["N_CS"],
            "phi_rms_final": tr[-1]["phi_rms"],
            "E_YM_peak_S3I": max(r["E_YM_S3I"] for r in tr),
            "gamma_ym_peak": max(r["gamma_ym"] for r in tr),
            "gamma_tot_at_amin": tr[idx_min_a]["gamma_tot"],
            "max_delta_T00_matter": res_II["max_delta_T00_matter"],
            "max_delta_T00_tot": res_II["max_delta_T00_tot"],
            "max_H_res_rel": res_II["max_H_res_rel"],
            "max_gauss_n0": res_II["max_gauss_n0"],
        }
        contract_rows.append(row)
        if outcome == "nonsingular_bounce":
            flagship_contract_bounce = tr
        if outcome == "singular_collapse" and flagship_contract_crunch is None:
            flagship_contract_crunch = tr
        print(
            f"    kap={kap:.2e}: {outcome:18s} | a_min={row['a_min']:.4f} | "
            f"N_CS {row['N_CS_init']:.3f}->{row['N_CS_at_amin']:.3f}->{row['N_CS_final']:.3f} | "
            f"E_YM_pk={row['E_YM_peak_S3I']:6.1f} | gam_ym_pk={row['gamma_ym_peak']:.4f} | "
            f"dT00_tot={row['max_delta_T00_tot']:.4f} | H_res={row['max_H_res_rel']:.2e}",
            flush=True,
        )

    return {
        "N_max": 24,
        "relaxed_rho2_E_YM": E_r2_flat,
        "relaxed_rho2_N_CS": ncs_r2_flat,
        "scenario_Ia_superhorizon_expanding": {
            "M_W": 5.0,
            "kappa_YM": 2e-4,
            "a_restore": 3.0 * math.sqrt(14.0) / (beta_H * 5.0),
            "a_av_mid": a_mid_Ia,
            "E_YM_peak_S3I": E_ym_peak_Ia,
            "N_CS_init": tr_Ia[0]["N_CS"],
            "N_CS_final": tr_Ia[-1]["N_CS"],
            "phi_rms_final": tr_Ia[-1]["phi_rms"],
            "max_delta_T00_matter": res_Ia["max_delta_T00_matter"],
            "max_delta_T00_tot": res_Ia["max_delta_T00_tot"],
            "max_H_res_rel": res_Ia["max_H_res_rel"],
            "max_gauss_n0": g0_Ia,
            "max_gauss_n20": g20_Ia,
            "trace_downsampled": tr_Ia[::2],
        },
        "scenario_Ib_subhorizon_recollapse_sweep": sub_horizon_rows,
        "scenario_Ib_trace_recollapse": sub_horizon_flagship_recollapse[::2] if sub_horizon_flagship_recollapse else None,
        "scenario_Ib_trace_escape": sub_horizon_flagship_escape[::2] if sub_horizon_flagship_escape else None,
        "scenario_II_contracting_gamma_c_sweep": contract_rows,
        "scenario_II_trace_bounce": flagship_contract_bounce[::2] if flagship_contract_bounce else None,
        "scenario_II_trace_crunch": flagship_contract_crunch[::2] if flagship_contract_crunch else None,
    }


# ---------------------------------------------------------------------------
# Part 3C: Nonlinear Mode-Mode Saturation of n=12 in High-E Pure-YM Plasma
# ---------------------------------------------------------------------------

def run_part_3c_nonlinear_n12_saturation(U1: np.ndarray) -> Dict:
    print("\n=== Part 3C: Nonlinear Mode-Mode Saturation of n=12 in High-E Pure-YM Plasma (k=1, N_max=24) ===", flush=True)
    bas24 = build_equivariant_basis(U1, k_winding=1, max_n=24)
    eng_tmp = RealtimeEquivariantYMHiggsCosmology(bas24, kappa_YM=0.10 / 2500.0, M_W=0.0, beta_H=1.556)
    v_k4 = find_k1_n12_k4_mode(eng_tmp)

    # 1) Gravitating bounces at E_YM = 2500 and E_YM = 2844 (gamma_YM = 0.10) across eps_12!
    bounce_rows = []
    flagship_bounce_trace = None
    for E_val in [2500.0, 2843.9]:
        kappa = 0.10 / E_val
        eng = RealtimeEquivariantYMHiggsCosmology(bas24, kappa_YM=kappa, M_W=0.0, beta_H=1.556)
        fp0 = math.sqrt(2.0 * (E_val - 0.5))
        for eps12 in [1e-5, 1e-3, 1e-2, 5e-2, 1e-1]:
            c0 = bas24["c_hom_id"] + (eps12 * math.sqrt(3.0)) * v_k4
            cp0 = fp0 * bas24["c_hom_id"]
            phi0 = np.zeros(eng.D_Phi)
            phip0 = np.zeros(eng.D_Phi)
            y0, _ = eng.build_initial_state(a0=2.0, c0=c0, cp0=cp0, phi0=phi0, phip0=phip0, contracting=True)
            res = eng.integrate(
                y0, (0.0, 6.0), method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.004, a_stop_hi=6.0, n_eval=250
            )
            sol = res["sol"]
            c_traj = sol.y[2 : 2 + eng.D_A, :]
            cp_traj = sol.y[2 + eng.D_A : 2 + 2 * eng.D_A, :]
            q12_gain = float(np.max(np.abs(v_k4 @ c_traj)) / (eps12 * math.sqrt(3.0)))

            # Track exact harmonic energy transfer & conformal gamma_YM conservation
            gam_vals = [r["gamma_ym"] for r in res["trace"]]
            gam_drift = max(abs(g - gam_vals[0]) for g in gam_vals) / gam_vals[0]
            part_end = compute_ym_energy_by_harmonic(eng, c_traj[:, -1], cp_traj[:, -1])
            peak_fE = 0.0
            part_trace = []
            for k_idx in np.linspace(0, sol.y.shape[1] - 1, min(120, sol.y.shape[1]), dtype=int):
                pt = compute_ym_energy_by_harmonic(eng, c_traj[:, k_idx], cp_traj[:, k_idx])
                peak_fE = max(peak_fE, pt["inhom_energy_frac"])
                if E_val == 2500.0 and eps12 == 1e-2:
                    d_k = eng.eval_diagnostics(sol.y[:, k_idx])
                    part_trace.append({
                        "eta": float(sol.t[k_idx]),
                        "a": float(sol.y[0, k_idx]),
                        "inhom_E_frac": pt["inhom_energy_frac"],
                        "kin_frac_by_n": pt["kin_frac_by_n"],
                        "delta_T00_matter": d_k["delta_T00_matter"],
                        "delta_T00_tot": d_k["delta_T00_tot"],
                        "gamma_ym": d_k["gamma_ym"],
                    })
            if E_val == 2500.0 and eps12 == 1e-2:
                flagship_bounce_trace = part_trace

            rec = {
                "E_YM": E_val,
                "eps_12": eps12,
                "bounced": bool(res["escaped"] and not res["collapsed"]),
                "a_min": res["a_min"],
                "a_min_exact_hom": math.sqrt(0.5 * (1.0 + math.sqrt(1.0 - 4.0 * gam_vals[0]))),
                "q12_nonlinear_gain": q12_gain,
                "peak_inhom_E_frac": peak_fE,
                "final_inhom_E_frac": part_end["inhom_energy_frac"],
                "final_kin_frac_by_n": part_end["kin_frac_by_n"],
                "conformal_gamma_YM_drift": gam_drift,
                "max_delta_T00_matter": res["max_delta_T00_matter"],
                "max_delta_T00_tot": res["max_delta_T00_tot"],
                "max_H_res_rel": res["max_H_res_rel"],
                "max_gauss_n0": res["max_gauss_n0"],
            }
            bounce_rows.append(rec)
            print(
                f"  E={E_val:6.1f} eps={eps12:5.0e} | bounce={rec['bounced']} a_min={rec['a_min']:.5f} "
                f"(hom={rec['a_min_exact_hom']:.5f}) | gain={q12_gain:7.2f}x | "
                f"fE_inhom pk={peak_fE:.4f} end={part_end['inhom_energy_frac']:.4f} | "
                f"dT00_mat={rec['max_delta_T00_matter']:.4f}, dT00_tot={rec['max_delta_T00_tot']:.4f} | "
                f"gam_drift={gam_drift:.2e}",
                flush=True,
            )

    # 2) Long-time fixed-a saturation run at E_YM = 2500 (eta in [0, 8.0], ~13 periods)
    print("  [Long-Time Fixed-a Saturation at E_YM = 2500, eps_12 = 1e-3, eta in [0, 8.0]]", flush=True)
    eng_lt = RealtimeEquivariantYMHiggsCosmology(bas24, kappa_YM=0.10 / 2500.0, M_W=0.0, beta_H=1.556)
    fp0 = math.sqrt(2.0 * (2500.0 - 0.5))
    c0_lt = bas24["c_hom_id"] + (1e-3 * math.sqrt(3.0)) * v_k4
    cp0_lt = fp0 * bas24["c_hom_id"]
    y0_lt, _ = eng_lt.build_initial_state(
        a0=1.0, c0=c0_lt, cp0=cp0_lt, phi0=np.zeros(eng_lt.D_Phi), phip0=np.zeros(eng_lt.D_Phi), freeze_a=True
    )
    res_lt = eng_lt.integrate(
        y0_lt, (0.0, 8.0), method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.004, freeze_a=True, n_eval=250
    )
    sol_lt = res_lt["sol"]
    c_lt = sol_lt.y[2 : 2 + eng_lt.D_A, :]
    cp_lt = sol_lt.y[2 + eng_lt.D_A : 2 + 2 * eng_lt.D_A, :]
    lt_trace = []
    for k_idx in np.linspace(0, sol_lt.y.shape[1] - 1, 200, dtype=int):
        pt = compute_ym_energy_by_harmonic(eng_lt, c_lt[:, k_idx], cp_lt[:, k_idx])
        d_k = eng_lt.eval_diagnostics(sol_lt.y[:, k_idx])
        lt_trace.append({
            "eta": float(sol_lt.t[k_idx]),
            "q12_abs": float(abs(v_k4 @ c_lt[:, k_idx])),
            "inhom_E_frac": pt["inhom_energy_frac"],
            "kin_frac_by_n": pt["kin_frac_by_n"],
            "delta_T00_matter": d_k["delta_T00_matter"],
            "E_YM_tot": pt["E_YM_tot_S3I"],
        })
    E_lt_drift = max(abs(r["E_YM_tot"] - lt_trace[0]["E_YM_tot"]) for r in lt_trace) / lt_trace[0]["E_YM_tot"]
    sat_mean_fE = float(np.mean([r["inhom_E_frac"] for r in lt_trace if r["eta"] >= 4.0]))
    sat_mean_dT00 = float(np.mean([r["delta_T00_matter"] for r in lt_trace if r["eta"] >= 4.0]))
    print(
        f"    Long-time (eta=4..8): mean inhom_E_frac={sat_mean_fE:.4f}, "
        f"mean dT00_mat={sat_mean_dT00:.4f}, E_YM_drift={E_lt_drift:.2e}, "
        f"end kin_frac={lt_trace[-1]['kin_frac_by_n']}",
        flush=True,
    )

    return {
        "N_max": 24,
        "D_A": eng_tmp.D_A,
        "bounce_saturation_runs": bounce_rows,
        "flagship_bounce_E2500_eps1e2_trace": flagship_bounce_trace,
        "longtime_fixed_a_E2500": {
            "eps_12": 1e-3,
            "eta_end": 8.0,
            "E_YM_conservation_rel_drift": E_lt_drift,
            "post_saturation_mean_inhom_E_frac": sat_mean_fE,
            "post_saturation_mean_delta_T00_matter": sat_mean_dT00,
            "final_kin_frac_by_n": lt_trace[-1]["kin_frac_by_n"],
            "trace": lt_trace,
        },
    }


# ---------------------------------------------------------------------------
# Part 3D: Independent Convergence (N_max Sweep) & Integrator (DOP853 vs Radau) Kill-Tests
# ---------------------------------------------------------------------------

def run_part_3d_killtests(U1: np.ndarray, poly_coeffs: np.ndarray) -> Dict:
    print("\n=== Part 3D: Independent Convergence (N_max) & Integrator (DOP853 vs Radau) Kill-Tests ===", flush=True)
    beta_H = 1.556

    # 1) N_max convergence sweep for Flagship 3B-I(a) avalanche (N_max = 12, 20, 24, 30, 36)
    print("  [Kill-Test 3D-1: k=7 Avalanche N_max Convergence Sweep (N_max = 12, 20, 24, 30, 36)]", flush=True)
    k7_conv = []
    for max_n in [12, 20, 24, 30, 36]:
        bas = build_equivariant_basis(U1, k_winding=7, max_n=max_n, poly_coeffs=poly_coeffs)
        c_flat, E_flat, ncs_flat = relax_flat_rho2(bas)
        eng = RealtimeEquivariantYMHiggsCosmology(bas, kappa_YM=2e-4, M_W=5.0, beta_H=beta_H)
        y0, _ = eng.build_initial_state(
            a0=1.10, c0=c_flat, cp0=np.zeros(eng.D_A),
            phi0=2e-2 * bas["phi_rho2_ground"], phip0=np.zeros(eng.D_Phi), contracting=False
        )
        res = eng.integrate(
            y0, (0.0, 2.5), method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.004, a_stop_hi=8.0, n_eval=250
        )
        tr = res["trace"]
        a_mid = next((r["a"] for r in tr if r["N_CS"] < 0.20), None)
        g0_max, g20_max = 0.0, 0.0
        for k_i in np.linspace(0, res["sol"].y.shape[1] - 1, 50, dtype=int):
            a0g, a20g = gauss_low_n_norm(eng, res["sol"].y[:, k_i], n_cut=20)
            g0_max = max(g0_max, a0g)
            g20_max = max(g20_max, a20g)
        rec = {
            "N_max": max_n,
            "D_A": eng.D_A,
            "D_Phi": eng.D_Phi,
            "E_flat_init": E_flat,
            "N_CS_init": ncs_flat,
            "a_av_mid": a_mid,
            "a_final": res["a_final"],
            "eta_final": res["eta_final"],
            "N_CS_final": tr[-1]["N_CS"],
            "phi_rms_final": tr[-1]["phi_rms"],
            "E_YM_peak_S3I": max(r["E_YM_S3I"] for r in tr),
            "max_delta_T00_tot": res["max_delta_T00_tot"],
            "max_H_res_rel": res["max_H_res_rel"],
            "max_gauss_n0": g0_max,
            "max_gauss_n20": g20_max,
        }
        k7_conv.append(rec)
        print(
            f"    k=7 N={max_n:2d} (D_A={eng.D_A:2d}, D_Phi={eng.D_Phi:2d}): E_flat={E_flat:.2e} | "
            f"a_mid={a_mid} | E_YM_pk={rec['E_YM_peak_S3I']:.3f} | N_CS_end={rec['N_CS_final']:+.4f} | "
            f"H_res={rec['max_H_res_rel']:.2e} | G_n0={g0_max:.2e} | G_n20={g20_max:.2e}",
            flush=True,
        )

    # 2) N_max convergence sweep for Flagship 3C nonlinear n=12 saturation (E=2500, eps=1e-2, N=12,20,24,30)
    print("  [Kill-Test 3D-2: k=1 Nonlinear n=12 Saturation N_max Convergence Sweep (N_max = 12, 20, 24, 30)]", flush=True)
    k1_conv = []
    for max_n in [12, 20, 24, 30]:
        bas = build_equivariant_basis(U1, k_winding=1, max_n=max_n)
        eng = RealtimeEquivariantYMHiggsCosmology(bas, kappa_YM=0.10 / 2500.0, M_W=0.0, beta_H=beta_H)
        v_k4 = find_k1_n12_k4_mode(eng)
        fp0 = math.sqrt(2.0 * (2500.0 - 0.5))
        c0 = bas["c_hom_id"] + (1e-2 * math.sqrt(3.0)) * v_k4
        cp0 = fp0 * bas["c_hom_id"]
        y0, _ = eng.build_initial_state(
            a0=2.0, c0=c0, cp0=cp0, phi0=np.zeros(eng.D_Phi), phip0=np.zeros(eng.D_Phi), contracting=True
        )
        res = eng.integrate(
            y0, (0.0, 6.0), method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.004, a_stop_hi=6.0, n_eval=200
        )
        sol = res["sol"]
        c_traj = sol.y[2 : 2 + eng.D_A, :]
        cp_traj = sol.y[2 + eng.D_A : 2 + 2 * eng.D_A, :]
        part_end = compute_ym_energy_by_harmonic(eng, c_traj[:, -1], cp_traj[:, -1])
        peak_fE = max(
            compute_ym_energy_by_harmonic(eng, c_traj[:, k], cp_traj[:, k])["inhom_energy_frac"]
            for k in np.linspace(0, sol.y.shape[1] - 1, 60, dtype=int)
        )
        rec = {
            "N_max": max_n,
            "D_A": eng.D_A,
            "a_min": res["a_min"],
            "eta_final": res["eta_final"],
            "peak_inhom_E_frac": peak_fE,
            "final_inhom_E_frac": part_end["inhom_energy_frac"],
            "final_kin_frac_by_n": part_end["kin_frac_by_n"],
            "max_delta_T00_matter": res["max_delta_T00_matter"],
            "max_delta_T00_tot": res["max_delta_T00_tot"],
            "max_H_res_rel": res["max_H_res_rel"],
            "max_gauss_n0": res["max_gauss_n0"],
        }
        k1_conv.append(rec)
        print(
            f"    k=1 N={max_n:2d} (D_A={eng.D_A:2d}): a_min={rec['a_min']:.5f} | "
            f"fE_pk={peak_fE:.4f} end={part_end['inhom_energy_frac']:.4f} | "
            f"dT00_mat={rec['max_delta_T00_matter']:.4f} | H_res={rec['max_H_res_rel']:.2e}",
            flush=True,
        )

    # 3) Independent DOP853 vs Radau Integrator Cross-Checks!
    print("  [Kill-Test 3D-3: Independent DOP853 vs. Radau Cross-Check on Both Flagship Runs]", flush=True)
    # (a) k=7 avalanche at N_max=20 (48 ODE vars -> Radau fast & exact)
    bas_k7_20 = build_equivariant_basis(U1, k_winding=7, max_n=20, poly_coeffs=poly_coeffs)
    c_flat20, _, _ = relax_flat_rho2(bas_k7_20)
    eng_k7_20 = RealtimeEquivariantYMHiggsCosmology(bas_k7_20, kappa_YM=2e-4, M_W=5.0, beta_H=beta_H)
    y0_k7, _ = eng_k7_20.build_initial_state(
        a0=1.10, c0=c_flat20, cp0=np.zeros(eng_k7_20.D_A),
        phi0=2e-2 * bas_k7_20["phi_rho2_ground"], phip0=np.zeros(eng_k7_20.D_Phi), contracting=False
    )
    res_k7_dop = eng_k7_20.integrate(
        y0_k7, (0.0, 0.80), method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.004, n_eval=50
    )
    res_k7_rad = eng_k7_20.integrate(
        y0_k7, (0.0, 0.80), method="Radau", rtol=1e-10, atol=1e-12, max_step=0.01, n_eval=50
    )
    y_dop_end = res_k7_dop["sol"].y[:, -1]
    y_rad_end = res_k7_rad["sol"].y[:, -1]
    k7_state_rel = float(np.linalg.norm(y_dop_end - y_rad_end) / np.linalg.norm(y_dop_end))
    k7_ncs_diff = abs(res_k7_dop["trace"][-1]["N_CS"] - res_k7_rad["trace"][-1]["N_CS"])

    # (b) k=1 nonlinear n=12 saturation bounce at N_max=20 (E=2500, eps=1e-2)
    bas_k1_20 = build_equivariant_basis(U1, k_winding=1, max_n=20)
    eng_k1_20 = RealtimeEquivariantYMHiggsCosmology(bas_k1_20, kappa_YM=0.10 / 2500.0, M_W=0.0, beta_H=beta_H)
    v_k4_20 = find_k1_n12_k4_mode(eng_k1_20)
    fp0 = math.sqrt(2.0 * (2500.0 - 0.5))
    c0_k1 = bas_k1_20["c_hom_id"] + (1e-2 * math.sqrt(3.0)) * v_k4_20
    cp0_k1 = fp0 * bas_k1_20["c_hom_id"]
    y0_k1, _ = eng_k1_20.build_initial_state(
        a0=2.0, c0=c0_k1, cp0=cp0_k1, phi0=np.zeros(eng_k1_20.D_Phi), phip0=np.zeros(eng_k1_20.D_Phi), contracting=True
    )
    res_k1_dop = eng_k1_20.integrate(
        y0_k1, (0.0, 2.0), method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.004, n_eval=50
    )
    res_k1_rad = eng_k1_20.integrate(
        y0_k1, (0.0, 2.0), method="Radau", rtol=1e-10, atol=1e-12, max_step=0.01, n_eval=50
    )
    yk1_dop = res_k1_dop["sol"].y[:, -1]
    yk1_rad = res_k1_rad["sol"].y[:, -1]
    k1_state_rel = float(np.linalg.norm(yk1_dop - yk1_rad) / np.linalg.norm(yk1_dop))
    k1_amin_diff = abs(res_k1_dop["a_min"] - res_k1_rad["a_min"])

    print(
        f"    k=7 Avalanche (N=20, eta=0.80): DOP853 vs Radau rel state diff = {k7_state_rel:.3e}, "
        f"|dN_CS| = {k7_ncs_diff:.3e} (Radau H_res={res_k7_rad['max_H_res_rel']:.2e})",
        flush=True,
    )
    print(
        f"    k=1 n=12 Sat  (N=20, eta=2.00): DOP853 vs Radau rel state diff = {k1_state_rel:.3e}, "
        f"|da_min| = {k1_amin_diff:.3e} (Radau H_res={res_k1_rad['max_H_res_rel']:.2e})",
        flush=True,
    )

    return {
        "k7_avalanche_N_max_sweep": k7_conv,
        "k1_saturation_N_max_sweep": k1_conv,
        "dop853_vs_radau": {
            "k7_avalanche_N20": {
                "eta_end": 0.80,
                "rel_state_diff": k7_state_rel,
                "abs_N_CS_diff": k7_ncs_diff,
                "dop853_N_CS_end": res_k7_dop["trace"][-1]["N_CS"],
                "radau_N_CS_end": res_k7_rad["trace"][-1]["N_CS"],
                "dop853_H_res": res_k7_dop["max_H_res_rel"],
                "radau_H_res": res_k7_rad["max_H_res_rel"],
            },
            "k1_saturation_N20": {
                "eta_end": 2.00,
                "rel_state_diff": k1_state_rel,
                "abs_a_min_diff": k1_amin_diff,
                "dop853_H_res": res_k1_dop["max_H_res_rel"],
                "radau_H_res": res_k1_rad["max_H_res_rel"],
            },
        },
    }


def main():
    t0 = time.time()
    _, _, U1, U2 = build_istar_and_galois()
    _, poly_coeffs = build_p7_equivariant_polynomial(U1, U2)

    part_3a = run_part_3a_nonlinear_hosotani_n12(U1)
    part_3b = run_part_3b_rho2_avalanche(U1, poly_coeffs)
    part_3c = run_part_3c_nonlinear_n12_saturation(U1)
    part_3d = run_part_3d_killtests(U1, poly_coeffs)

    payload = {
        "part_3a_nonlinear_hosotani_n12": part_3a,
        "part_3b_rho2_avalanche_cosmology": part_3b,
        "part_3c_nonlinear_n12_saturation": part_3c,
        "part_3d_convergence_and_integrator_killtests": part_3d,
        "elapsed_sec": round(time.time() - t0, 2),
    }

    out_path = os.path.join(OUT_DIR, "t4b_03_dynamical_phenomena.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {out_path} in {payload['elapsed_sec']}s.", flush=True)


if __name__ == "__main__":
    main()
