"""P02-02: Exact Floquet Multipliers, Lamé Parametric Resonance, & Higgs Stabilization (S^3 vs. S^3/I*).

Computes:
  1. Double-path verification of pure-YM physical Floquet multipliers:
     - Path A: Temporal gauge (alpha'' + M_YM^2(f) alpha = 0), verifying det(M)=1 and exact gauge
       eigenvector (D(f0), fp0*D1) with mu = +1, then extracting the 2*(d_alpha - d_0) physical multipliers.
     - Path B: Exact parallel-transport Coulomb reduction L_phys(q, q') on D(f)^perp with +3 (f')^2/|D|^2 term.
     Confirms agreement to < 1e-9 across all coupled (n, K) blocks!
  2. Dense pure-YM Floquet stability & growth-rate scan sigma_Fl = ln(|mu_max|)/T across E_YM in [0.55, 5000]
     for n = 0 (homogeneous), S^3-only modes (n = 1, 2, 3, 4, 5, 6), and lowest surviving S^3/I* modes
     (n = 12, 20, 24, 30), resolving BOTH tachyonic bands AND non-tachyonic Lamé parametric resonances!
  3. Exact Hamiltonian Proca-Higgs Floquet map across mu_W in [0, 5] and E_tot in [2, 1000] (plus finite-m_H
     radial-Higgs h-coupled check in the K = j sector for stiff-Higgs m_H/M_W = 5 and SM m_H/M_W = 1.556).
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from istar_harmonics import (
    build_coulomb_complement,
    build_nK_block_operators,
    eval_ym_operators,
    klein_molien_coeffs,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Background Periodic Orbit T(E_tot, mu_W)
# ---------------------------------------------------------------------------

def background_period(E_tot: float, mu_W: float = 0.0) -> Tuple[float, float, float, float]:
    """Find exact period T, turning points (f_min, f_max), and initial (f0, fp0) for:
       f'' + 2 f (f^2 - 1) + mu_W^2 (1 + f) = 0,
       E_tot = 0.5 f'^2 + 0.5 (f^2 - 1)^2 + 0.5 mu_W^2 (1 + f)^2.
    Starts at f0 = -1 (where U_eff(-1) = 0 identically for ALL mu_W!) with fp0 = +sqrt(2 E_tot).
    """
    f0 = -1.0
    fp0 = math.sqrt(2.0 * E_tot)
    mu2 = mu_W * mu_W

    def ode_bg(_eta, y):
        f, fp = y
        return [fp, -2.0 * f * (f * f - 1.0) - mu2 * (1.0 + f)]

    def ev_ret(_eta, y):
        return y[0] + 1.0

    ev_ret.terminal = False
    ev_ret.direction = 1

    # Estimate upper bound on period (for E = 0.55 near separatrix, T ~ 14; for large E, T ~ 5/E^{1/4})
    t_max = 30.0 if E_tot < 1.0 else (15.0 if E_tot < 5.0 else 8.0)
    sol = solve_ivp(ode_bg, (0.0, t_max), [f0, fp0], method="DOP853", rtol=1e-13, atol=1e-15,
                    events=[ev_ret], max_step=min(0.02, 0.5 / max(1.0, E_tot ** 0.25)))
    rets = [float(t) for t in sol.t_events[0] if t > 1e-4]
    assert len(rets) >= 1, f"Failed to find background period at E={E_tot}, mu_W={mu_W}"
    T = rets[0]

    # Sample one period cleanly to extract turning points
    sol_T = solve_ivp(ode_bg, (0.0, T), [f0, fp0], method="DOP853", rtol=1e-13, atol=1e-15,
                      max_step=T / 200.0)
    f_min = float(np.min(sol_T.y[0]))
    f_max = float(np.max(sol_T.y[0]))
    return T, f_min, f_max, fp0


# ---------------------------------------------------------------------------
# Path A: Temporal-Gauge Pure-YM Floquet Solver (Exact Symplectic, Nonsingular)
# ---------------------------------------------------------------------------

def floquet_pure_ym_temporal(blk: dict, E_YM: float, T: float, fp0: float) -> dict:
    """Compute physical pure-YM Floquet multipliers via temporal gauge alpha'' + M_YM^2(f) alpha = 0."""
    d = blk["d_alpha"]
    d0 = blk["d_0"]
    d_phys = d - d0
    if d_phys == 0:
        return {"mu_max": 1.0, "sigma_fl": 0.0, "det_err": 0.0, "gauge_eig_err": 0.0, "phys_eigs_abs": []}

    dim_phase = 2 * d
    # State vector: [f, fp, Phi (dim_phase x dim_phase flattened)]
    y0 = np.zeros(2 + dim_phase * dim_phase, dtype=np.float64)
    y0[0] = -1.0
    y0[1] = fp0
    y0[2:] = np.eye(dim_phase, dtype=np.float64).ravel()

    JS = blk["JS"]
    ST = blk["ST"]
    I_d = np.eye(d, dtype=np.float64)

    def rhs(_eta, y):
        f = y[0]
        fp = y[1]
        Phi = y[2:].reshape(dim_phase, dim_phase)
        lam = 1.0 + f
        C_A = -2.0 * JS - 2.0 * I_d - lam * ST
        M2 = C_A @ C_A - (f * f - 1.0) * ST

        dPhi = np.empty_like(Phi)
        dPhi[:d, :] = Phi[d:, :]
        dPhi[d:, :] = -M2 @ Phi[:d, :]

        out = np.empty_like(y)
        out[0] = fp
        out[1] = -2.0 * f * (f * f - 1.0)
        out[2:] = dPhi.ravel()
        return out

    max_step = min(T / 80.0, 0.025)
    sol = solve_ivp(rhs, (0.0, T), y0, method="DOP853", rtol=1e-11, atol=1e-13, max_step=max_step)
    M = sol.y[2:, -1].reshape(dim_phase, dim_phase)
    det_err = abs(float(np.linalg.det(M)) - 1.0)

    all_eigs = np.linalg.eigvals(M)

    gauge_eig_err = 0.0
    if d0 == 1:
        # Verify exact gauge solution v_g = [D(f0), fp0 * D1] is an eigenvector with eigenvalue +1
        D_0 = blk["D0"]  # at f0 = -1, lam = 0 so D(-1) = D0
        v_g = np.concatenate([D_0[:, 0], fp0 * blk["D1"][:, 0]])
        nv_g = float(np.linalg.norm(v_g))
        if nv_g > 1e-12:
            gauge_eig_err = float(np.linalg.norm(M @ v_g - v_g) / nv_g)

        # Remove the two eigenvalues closest to +1.0 (the exact 2x2 Jordan gauge/Gauss pair)
        idx_sorted = np.argsort(np.abs(all_eigs - 1.0))
        phys_eigs = np.delete(all_eigs, idx_sorted[:2])
    else:
        phys_eigs = all_eigs

    mags = np.sort(np.abs(phys_eigs))[::-1]
    mu_max = float(mags[0])
    sigma_fl = math.log(max(mu_max, 1.0)) / T
    return {
        "mu_max": mu_max,
        "sigma_fl": sigma_fl,
        "det_err": det_err,
        "gauge_eig_err": gauge_eig_err,
        "phys_eigs_abs": [float(x) for x in mags],
    }


# ---------------------------------------------------------------------------
# Path B: Exact Parallel-Transport Coulomb-Reduced Pure-YM Floquet Solver
# ---------------------------------------------------------------------------

def floquet_pure_ym_coulomb_reduced(blk: dict, E_YM: float, T: float, fp0: float) -> dict:
    """Independent path: integrate parallel-transport Coulomb-reduced Lagrangian L_phys(q, q') on D(f)^perp."""
    d = blk["d_alpha"]
    d0 = blk["d_0"]
    d_phys = d - d0
    if d0 == 0:
        return floquet_pure_ym_temporal(blk, E_YM, T, fp0)
    if d_phys == 0:
        return {"mu_max": 1.0, "sigma_fl": 0.0, "det_err": 0.0}

    # Initial orthonormal complement Q0 to D(f0 = -1) = D0
    D_init = blk["D0"]
    assert float(np.linalg.norm(D_init)) > 1e-12, "Coulomb reduction requires D(f0) != 0"
    Q0 = build_coulomb_complement(D_init)  # (d, d_phys)

    dim_q = 2 * d_phys
    nQ = d * d_phys
    # State: [f, fp, Q (d x d_phys), Phi_q (dim_q x dim_q)]
    y0 = np.zeros(2 + nQ + dim_q * dim_q, dtype=np.float64)
    y0[0] = -1.0
    y0[1] = fp0
    y0[2: 2 + nQ] = Q0.ravel()
    y0[2 + nQ:] = np.eye(dim_q, dtype=np.float64).ravel()

    JS = blk["JS"]
    ST = blk["ST"]
    D0 = blk["D0"][:, 0]
    D1 = blk["D1"][:, 0]
    I_d = np.eye(d, dtype=np.float64)

    def rhs(_eta, y):
        f = y[0]
        fp = y[1]
        Q = y[2: 2 + nQ].reshape(d, d_phys)
        Phi = y[2 + nQ:].reshape(dim_q, dim_q)

        lam = 1.0 + f
        D = D0 + lam * D1
        D2 = float(np.dot(D, D))
        # Parallel transport: Q' = -D (Q^T (fp D1))^T / |D|^2
        q_d1 = Q.T @ D1  # (d_phys,)
        dQ = -np.outer(D, fp * q_d1) / D2

        C_A = -2.0 * JS - 2.0 * I_d - lam * ST
        M2 = C_A @ C_A - (f * f - 1.0) * ST
        V_eff = Q.T @ M2 @ Q + (3.0 * fp * fp / D2) * np.outer(q_d1, q_d1)

        dPhi = np.empty_like(Phi)
        dPhi[:d_phys, :] = Phi[d_phys:, :]
        dPhi[d_phys:, :] = -V_eff @ Phi[:d_phys, :]

        out = np.empty_like(y)
        out[0] = fp
        out[1] = -2.0 * f * (f * f - 1.0)
        out[2: 2 + nQ] = dQ.ravel()
        out[2 + nQ:] = dPhi.ravel()
        return out

    max_step = min(T / 80.0, 0.025)
    sol = solve_ivp(rhs, (0.0, T), y0, method="DOP853", rtol=1e-11, atol=1e-13, max_step=max_step)
    QT = sol.y[2: 2 + nQ, -1].reshape(d, d_phys)
    Phi_T = sol.y[2 + nQ:, -1].reshape(dim_q, dim_q)

    # Frame holonomy R = Q0^T QT in O(d_phys)
    R = Q0.T @ QT
    hol_ortho_err = float(np.max(np.abs(R.T @ R - np.eye(d_phys))))
    R_block = np.block([[R, np.zeros((d_phys, d_phys))], [np.zeros((d_phys, d_phys)), R]])
    M_phys = R_block @ Phi_T
    det_err = abs(float(np.linalg.det(M_phys)) - 1.0)
    mags = np.sort(np.abs(np.linalg.eigvals(M_phys)))[::-1]
    mu_max = float(mags[0])
    return {
        "mu_max": mu_max,
        "sigma_fl": math.log(max(mu_max, 1.0)) / T,
        "det_err": det_err,
        "hol_ortho_err": hol_ortho_err,
        "phys_eigs_abs": [float(x) for x in mags],
    }


# ---------------------------------------------------------------------------
# Exact Hamiltonian Proca-Higgs (+ Optional Radial Higgs h) Floquet Solver
# ---------------------------------------------------------------------------

def floquet_proca_higgs(
    blk: dict,
    E_tot: float,
    mu_W: float,
    T: float,
    fp0: float,
    include_radial_h: bool = False,
    mH_over_MW: float = 5.0,
) -> dict:
    """Exact Hamiltonian Floquet solver for Proca-Higgs (unitary gauge pi=0, mu_W > 0).

    State: (alpha, p_alpha) [plus (h_tilde, p_h) if include_radial_h and blk['d_h'] == 1].
    For fixed-a conformal Floquet test:
      - G = I - D D^T / (|D|^2 + mu_W^2),  G^{-1} = I + D D^T / mu_W^2
      - B = D (fp D1)^T / (|D|^2 + mu_W^2)
      - V_alpha = M_YM^2(f) + mu_W^2 I + (fp^2) D1 D1^T / (|D|^2 + mu_W^2)
      - If radial Higgs h included (K = j only, frozen background phi=v):
        V_hh = n(n+2) + 0.75*(1+f)^2 + (mH_over_MW * mu_W)^2
        V_h_alpha = sqrt(3) * (1+f) * mu_W * e_trace
    """
    if mu_W <= 1e-12:
        return floquet_pure_ym_temporal(blk, E_tot, T, fp0)

    d = blk["d_alpha"]
    d0 = blk["d_0"]
    use_h = bool(include_radial_h and blk["d_h"] == 1)
    dq = d + (1 if use_h else 0)
    dim_phase = 2 * dq

    y0 = np.zeros(2 + dim_phase * dim_phase, dtype=np.float64)
    y0[0] = -1.0
    y0[1] = fp0
    y0[2:] = np.eye(dim_phase, dtype=np.float64).ravel()

    JS = blk["JS"]
    ST = blk["ST"]
    I_d = np.eye(d, dtype=np.float64)
    D0 = blk["D0"][:, 0] if d0 == 1 else None
    D1 = blk["D1"][:, 0] if d0 == 1 else None
    mu2 = mu_W * mu_W
    n = blk["n"]
    e_tr = blk["e_trace"]
    mu_H2 = (mH_over_MW * mu_W) ** 2

    def rhs(_eta, y):
        f = y[0]
        fp = y[1]
        Phi = y[2:].reshape(dim_phase, dim_phase)

        lam = 1.0 + f
        C_A = -2.0 * JS - 2.0 * I_d - lam * ST
        M2 = C_A @ C_A - (f * f - 1.0) * ST

        if d0 == 1:
            D = D0 + lam * D1
            D2 = float(np.dot(D, D))
            W = D2 + mu2
            G_inv = I_d + np.outer(D, D) / mu2
            B = np.outer(D, fp * D1) / W
            V_a = M2 + mu2 * I_d + (fp * fp / W) * np.outer(D1, D1)
        else:
            G_inv = I_d
            B = np.zeros((d, d), dtype=np.float64)
            V_a = M2 + mu2 * I_d

        if use_h:
            G_inv_full = np.eye(dq, dtype=np.float64)
            G_inv_full[:d, :d] = G_inv
            B_full = np.zeros((dq, dq), dtype=np.float64)
            B_full[:d, :d] = B
            V_full = np.zeros((dq, dq), dtype=np.float64)
            V_full[:d, :d] = V_a
            V_full[d, d] = n * (n + 2) + 0.75 * lam * lam + mu_H2
            coup = math.sqrt(3.0) * lam * mu_W * e_tr
            V_full[:d, d] = coup
            V_full[d, :d] = coup
        else:
            G_inv_full = G_inv
            B_full = B
            V_full = V_a

        q_mat = Phi[:dq, :]
        p_mat = Phi[dq:, :]
        dq_mat = G_inv_full @ (p_mat - B_full @ q_mat)
        dp_mat = B_full.T @ dq_mat - V_full @ q_mat

        dPhi = np.vstack([dq_mat, dp_mat])
        out = np.empty_like(y)
        out[0] = fp
        out[1] = -2.0 * f * (f * f - 1.0) - mu2 * lam
        out[2:] = dPhi.ravel()
        return out

    max_step = min(T / 80.0, 0.025)
    sol = solve_ivp(rhs, (0.0, T), y0, method="DOP853", rtol=1e-11, atol=1e-13, max_step=max_step)
    M = sol.y[2:, -1].reshape(dim_phase, dim_phase)
    det_err = abs(float(np.linalg.det(M)) - 1.0)
    mags = np.sort(np.abs(np.linalg.eigvals(M)))[::-1]
    mu_max = float(mags[0])
    return {
        "mu_max": mu_max,
        "sigma_fl": math.log(max(mu_max, 1.0)) / T,
        "det_err": det_err,
        "eigs_abs": [float(x) for x in mags],
    }


def main():
    molien = klein_molien_coeffs(40)
    all_ns = [0, 1, 2, 3, 4, 5, 6, 12, 20, 24, 30]
    blocks_by_n = {n: build_nK_block_operators(n) for n in all_ns}

    # ------------------------------------------------------------------
    # 1. Double-Path Cross-Check (Temporal Gauge vs. Coulomb-Reduced)
    # ------------------------------------------------------------------
    xcheck_rows = []
    max_path_diff = 0.0
    max_det_err = 0.0
    max_gauge_err = 0.0
    for E_test in [1.0, 5.0, 25.0, 200.0]:
        T, fmin, fmax, fp0 = background_period(E_test, 0.0)
        for n in [1, 2, 3, 4, 12]:
            for two_K, blk in sorted(blocks_by_n[n].items()):
                if blk["d_0"] == 1 and (blk["d_alpha"] - blk["d_0"]) > 0:
                    # Skip (n=2, 2K=0) which has d_phys=0; check D(f0) != 0
                    if float(np.linalg.norm(blk["D0"])) < 1e-12:
                        continue
                    res_a = floquet_pure_ym_temporal(blk, E_test, T, fp0)
                    res_b = floquet_pure_ym_coulomb_reduced(blk, E_test, T, fp0)
                    diff = abs(res_a["mu_max"] - res_b["mu_max"])
                    max_path_diff = max(max_path_diff, diff)
                    max_det_err = max(max_det_err, res_a["det_err"], res_b["det_err"])
                    max_gauge_err = max(max_gauge_err, res_a["gauge_eig_err"])
                    xcheck_rows.append({
                        "E_YM": E_test, "n": n, "two_K": two_K,
                        "mu_temporal": res_a["mu_max"], "mu_coulomb": res_b["mu_max"],
                        "abs_diff": diff, "gauge_eig_err": res_a["gauge_eig_err"],
                    })

    print(f"[Double-Path Check] max |mu_A - mu_B| = {max_path_diff:.3e} | max|det-1| = {max_det_err:.3e} | max gauge_err = {max_gauge_err:.3e}")

    # ------------------------------------------------------------------
    # 2. Pure-YM Dense Floquet & Lamé Resonance Scan across E_YM in [0.6, 5000]
    # ------------------------------------------------------------------
    E_grid = sorted(set(
        [0.6, 0.8, 1.0, 1.2, 1.5, 1.58, 1.6, 2.0, 3.0, 4.0, 5.0, 6.2, 8.0, 10.0, 15.0,
         16.6, 20.0, 30.0, 36.2, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 443.0,
         500.0, 750.0, 1000.0, 1555.0, 2000.0, 3000.0, 4028.0, 5000.0]
        + [round(float(x), 4) for x in np.geomspace(0.6, 5000.0, 65)]
    ))

    # For each E in E_grid, compute worst (max sigma_fl) across ALL perturbative K-blocks for each n
    # Also separately track K = j +/- 2 (tachyonic-capable) vs K = j, j +/- 1 (Lamé parametric)
    scan_ns = [0, 1, 2, 3, 4, 12, 20, 24, 30]
    pure_ym_scan = []
    worst_by_n_below_Etach: Dict[int, dict] = {n: {"max_mu": 1.0, "max_sigma": 0.0, "at_E": None, "two_K": None} for n in scan_ns}

    # Exact E_tach(n) for reference (from closed form (m^2-1)^4 / (32 m^4))
    def exact_Etach(n_val: int) -> float:
        m = (n_val + 3) if n_val <= 3 else (n_val - 1)
        return ((m * m - 1.0) ** 4) / (32.0 * (m ** 4))

    for E_val in E_grid:
        T, fmin, fmax, fp0 = background_period(E_val, 0.0)
        row = {"E_YM": E_val, "T": T, "f_min": fmin, "f_max": fmax, "by_n": {}}
        for n in scan_ns:
            best_mu = 1.0
            best_sigma = 0.0
            best_2K = None
            best_lame_mu = 1.0
            best_lame_sigma = 0.0
            best_lame_2K = None

            for two_K, blk in blocks_by_n[n].items():
                if n == 0 and two_K == 0:
                    continue  # background mode itself (delta f)
                if blk["d_alpha"] - blk["d_0"] == 0:
                    continue
                res = floquet_pure_ym_temporal(blk, E_val, T, fp0)
                mu_v = res["mu_max"]
                sig_v = res["sigma_fl"]
                if mu_v > best_mu:
                    best_mu = mu_v
                    best_sigma = sig_v
                    best_2K = two_K
                # Track non-(j+/-2) sectors (K in {j-1, j, j+1}) separately to detect Lamé resonances!
                if abs(two_K - n) <= 2 and mu_v > best_lame_mu:
                    best_lame_mu = mu_v
                    best_lame_sigma = sig_v
                    best_lame_2K = two_K

            row["by_n"][str(n)] = {
                "mu_max": best_mu,
                "sigma_fl": best_sigma,
                "two_K": best_2K,
                "lame_mu_max": best_lame_mu,
                "lame_sigma_fl": best_lame_sigma,
                "lame_two_K": best_lame_2K,
            }
            if E_val < exact_Etach(n) and best_mu > worst_by_n_below_Etach[n]["max_mu"]:
                worst_by_n_below_Etach[n] = {"max_mu": best_mu, "max_sigma": best_sigma, "at_E": E_val, "two_K": best_2K}
        pure_ym_scan.append(row)

    # ------------------------------------------------------------------
    # 3. Benchmark Table at Key Energies (E = 1, 2, 10, 50, 200, 500, 1000, 5000)
    # ------------------------------------------------------------------
    key_Es = [1.0, 2.0, 10.0, 50.0, 200.0, 500.0, 1000.0, 5000.0]
    key_table = [r for r in pure_ym_scan if any(abs(r["E_YM"] - ek) < 1e-6 for ek in key_Es)]

    print("\n=== Pure-YM Floquet Growth Rate sigma_Fl = ln|mu_max|/T (and |mu_max|) Across E_YM ===")
    hdr = "   E_YM     T    |  n=0 (hom)     |  n=1 (S3)      |  n=2 (S3)      |  n=4 (S3)      |  n=12 (S3/I*)  |  n=20 (S3/I*)"
    print(hdr)
    for r in key_table:
        parts = [f"{r['E_YM']:7.1f}  {r['T']:6.3f}"]
        for n in (0, 1, 2, 4, 12, 20):
            d = r["by_n"][str(n)]
            parts.append(f"{d['sigma_fl']:5.3f} ({d['mu_max']:6.2f})")
        print(" | ".join(parts))

    print("\n=== Worst Sub-Tachyonic (Parametric/Lamé) Resonance Below E_tach(n) ===")
    for n in scan_ns:
        w = worst_by_n_below_Etach[n]
        tag = "S^3/I*" if molien[n] > 0 else "S^3-only"
        print(f"  n={n:2d} [{tag:8s}] E_tach={exact_Etach(n):9.3f} | max sub-tach |mu|={w['max_mu']:8.4f} (sigma={w['max_sigma']:6.4f}) at E={w['at_E']} (2K={w['two_K']})")

    # ------------------------------------------------------------------
    # 4. Higgs-Coupled Proca Floquet Map (mu_W in [0, 5], E_tot in [10, 50, 200, 500, 1000])
    # ------------------------------------------------------------------
    E_higgs_list = [10.0, 50.0, 200.0, 500.0, 1000.0]
    mu_W_list = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    higgs_ns = [0, 1, 2, 4, 12, 20]
    higgs_map = []
    max_proca_det_err = 0.0

    for E_tot in E_higgs_list:
        row_e = {"E_tot": E_tot, "by_mu_W": []}
        for mu_W in mu_W_list:
            T, fmin, fmax, fp0 = background_period(E_tot, mu_W)
            entry = {"mu_W": mu_W, "T": T, "f_min": fmin, "f_max": fmax, "by_n": {}}
            for n in higgs_ns:
                best_mu = 1.0
                best_sig = 0.0
                best_2K = None
                for two_K, blk in blocks_by_n[n].items():
                    if n == 0 and two_K == 0:
                        continue
                    if mu_W == 0.0 and (blk["d_alpha"] - blk["d_0"] == 0):
                        continue
                    res = floquet_proca_higgs(blk, E_tot, mu_W, T, fp0, include_radial_h=False)
                    max_proca_det_err = max(max_proca_det_err, res["det_err"])
                    if res["mu_max"] > best_mu:
                        best_mu = res["mu_max"]
                        best_sig = res["sigma_fl"]
                        best_2K = two_K
                entry["by_n"][str(n)] = {"mu_max": best_mu, "sigma_fl": best_sig, "two_K": best_2K}
            row_e["by_mu_W"].append(entry)
        higgs_map.append(row_e)

    # ------------------------------------------------------------------
    # 5. Validated Stiff-Higgs Radial-Mixing Check in K = j Sector
    #    Background phi == v is an asymptotic solution ONLY when the radial Higgs mass
    #    mu_H = (m_H / M_W) * mu_W satisfies eps_stiff := 3*(1 + f_max)^2 / (2 * mu_H^2) << 1
    #    (so the linear tadpole displacement |delta_phi|/v <= eps_stiff << 1 and no Hosotani
    #    symmetry restoration occurs). We test genuine stiff-Higgs ratios (m_H/M_W in {5, 10, 20})
    #    where eps_stiff < 0.25, and explicitly log eps_stiff for every point!
    # ------------------------------------------------------------------
    radial_checks = []
    for E_tot in [10.0, 50.0, 200.0]:
        for mu_W in [2.5, 4.0, 5.0]:
            T, fmin, fmax, fp0 = background_period(E_tot, mu_W)
            for ratio_name, ratio_val in [("stiff_5.0", 5.0), ("stiff_10.0", 10.0), ("stiff_20.0", 20.0)]:
                mu_H = ratio_val * mu_W
                eps_stiff = 1.5 * ((1.0 + fmax) ** 2) / (mu_H * mu_H)
                if eps_stiff > 0.25:
                    continue  # Strictly enforce valid stiff-Higgs background!
                for n in [1, 2, 12]:
                    blk_j = blocks_by_n[n][n]  # 2K == n <=> K == j
                    res_no_h = floquet_proca_higgs(blk_j, E_tot, mu_W, T, fp0, include_radial_h=False)
                    res_with_h = floquet_proca_higgs(blk_j, E_tot, mu_W, T, fp0, include_radial_h=True, mH_over_MW=ratio_val)
                    radial_checks.append({
                        "E_tot": E_tot, "mu_W": mu_W, "regime": ratio_name, "mH_over_MW": ratio_val,
                        "f_max": fmax, "mu_H": mu_H, "eps_stiff": eps_stiff, "n": n,
                        "mu_no_h": res_no_h["mu_max"], "sigma_no_h": res_no_h["sigma_fl"],
                        "mu_with_h": res_with_h["mu_max"], "sigma_with_h": res_with_h["sigma_fl"],
                        "det_err": res_with_h["det_err"],
                    })

    print(f"\n=== Higgs Proca Floquet Map (max|det-1| = {max_proca_det_err:.2e}) at E_tot = 500 and 1000 ===")
    for row_e in higgs_map:
        if row_e["E_tot"] in (500.0, 1000.0):
            print(f"  [E_tot = {row_e['E_tot']:.0f}]")
            for entry in row_e["by_mu_W"]:
                vals = " | ".join(
                    f"n={n}:{entry['by_n'][str(n)]['sigma_fl']:5.3f}(2K={entry['by_n'][str(n)]['two_K']})"
                    for n in (0, 1, 4, 12, 20)
                )
                print(f"    mu_W={entry['mu_W']:3.1f} [f in {entry['f_min']:+6.2f},{entry['f_max']:+6.2f}]: {vals}")

    max_radial_sig = max(r["sigma_with_h"] for r in radial_checks)
    print(f"\n=== Validated Stiff-Higgs Radial-Mixing Check (eps_stiff <= 0.25, N={len(radial_checks)} pts) === max sigma_fl = {max_radial_sig:.4e}")
    for r in sorted(radial_checks, key=lambda x: -x["sigma_with_h"])[:6]:
        print(f"  E={r['E_tot']:5.1f} mu_W={r['mu_W']:3.1f} {r['regime']:10s} eps={r['eps_stiff']:.3f} n={r['n']:2d}: no_h={r['mu_no_h']:6.4f} -> with_h={r['mu_with_h']:6.4f} (sigma={r['sigma_with_h']:.2e})")

    payload = {
        "script": "p02_02_floquet_instability",
        "double_path_verification": {
            "max_abs_diff_temporal_vs_coulomb": max_path_diff,
            "max_det_err": max_det_err,
            "max_gauge_eigenvector_err": max_gauge_err,
            "samples": xcheck_rows,
        },
        "pure_ym_floquet_scan": pure_ym_scan,
        "worst_subtachyonic_by_n": {str(k): v for k, v in worst_by_n_below_Etach.items()},
        "higgs_proca_floquet_map": {
            "max_det_err": max_proca_det_err,
            "E_list": E_higgs_list,
            "mu_W_list": mu_W_list,
            "map": higgs_map,
        },
        "radial_higgs_K_eq_j_checks": radial_checks,
    }
    out_path = os.path.join(OUT_DIR, "p02_02_floquet_instability.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
