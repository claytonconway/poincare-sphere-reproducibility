#!/usr/bin/env python3
"""
Stream p2_02_nonlinear_kam_backreaction (CLM-OPEN-02)
Stage 2 (Round 2): Self-Consistent Nonlinear Coupled Starobinsky-Casimir + Riegert +
                   Multi-Mode TT Graviton + Conformal 2-Photon Radiation Solver.

Round-2 Cold-Audit Remediations Incorporated:
  - [P0-1] Radius-dependent 1-loop 2-photon sideband rates (PCHIP C^1 interpolation of
           01_derive.py's exact shrinking-ring Parseval + Manley-Rowe table across
           x in [8.68, 21.92], capturing exact lower-sideband shutoff at x_lo=15.53,
           chiral upper-sideband damping in [12.5, 15.5], 1st-harmonic shutoff at
           x_up=12.49, and 2nd-harmonic continuation, normalized to local kap_scal=2*x^2).
  - [P0-2] Complete logged grids for Exp A (all 16 rows), Exp B (including G_scal=2e-4),
           Exp C (including n=10 1:1 tongue at sigma_t=4.20), and Exp D (all sigma_t in
           {9/31, 4, 6, 8, 10, 12}).
  - [P1-1] True un-floored relative Friedmann constraint residual (no hardcoded 1.0 floor!).
  - [P1-2] All 1-loop rate tables loaded directly from p2_02_01_identities.json, including
           exact upper-sideband Manley-Rowe scalaron share for n=18, 22 at sigma_t0=9/31.
  - [P1-3] Explicit g_2=1 (single axisymmetric polarization) vs g_2=5 (full 5-mode multiplet)
           tracking, exact round-S^3 TT degeneracies 2*(n+3)*(n-1)={24,42,64,90} with
           exact per-mode matched initial energy (0.002/mode), and RMS metric / Brill-Hartle
           spatial curvature validity diagnostics.
  - [P1-6] Pre-singularity safe radiation logging (kappa_rad_safe at last A_R>=0.1, |phi|<=50)
           separating physical budgets from singular 1/A_R spikes.
"""

import json
import math
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Exact P05 / Track IV-A parameters (pure dark gauge bosons, N_V >= 1)
KAPPA_V = -44738.0 / 93.0        # -481.0537634408602
SIGMA_0 = 9.0 / 31.0             # +0.2903225806451613
C_ANOM = 9.0 / 31.0              # (1/2)*(c_anom / a_anom) in H_S=1 units
X_RING_0 = math.sqrt(44672.0 / 93.0)  # 21.91675355...


def load_rate_tables():
    """Load reproducible Parseval/Manley-Rowe tables computed by 01_derive.py."""
    path = os.path.join(OUT_DIR, "p2_02_01_identities.json")
    with open(path) as fh:
        data = json.load(fh)
    return data["parseval_mr_rate_tables"]


def build_n2_x_dependent_interpolator(rate_tables):
    """
    Build smooth C^1 monotone PCHIP interpolators for n=2 (M_I*=26) at sigma_t0 = 9/31:
      g_tot(x) and gamma_b_MR(x) as functions of instantaneous radius x in [x_min_tbl, x_max_tbl].
    """
    rows = sorted(rate_tables["n2_shrinking_ring_sigma0"], key=lambda r: r["x_ring_eff"])
    xs = np.array([r["x_ring_eff"] for r in rows], dtype=float)
    gt = np.array([r["gamma_rad_tot"] for r in rows], dtype=float)
    gb = np.array([r["gamma_b_MR"] for r in rows], dtype=float)
    # Uses pure self-computed Parseval/MR table (gb = -3.40e-6 at x_ring_0, transitioning cleanly
    # into strong chiral upper-sideband damping across x_lo=15.53 and x_up=12.49).
    return {
        "x_min": float(xs[0]),
        "x_max": float(xs[-1]),
        "g_tot_fn": PchipInterpolator(xs, gt, extrapolate=False),
        "gam_b_fn": PchipInterpolator(xs, gb, extrapolate=False),
    }


def x_ring_for_sigma(sigma_t, kappa_V=KAPPA_V):
    """Exact Casimir-curvature ring radius sqrt(|1 - sigma_t + kappa_V|)."""
    return math.sqrt(abs(1.0 - sigma_t + kappa_V))


def kappa_scal_ref_for_sigma(sigma_t, kappa_V=KAPPA_V):
    """Exact bounce scalaron energy budget 2*x_ring^2 at h=0, s=0."""
    xr = x_ring_for_sigma(sigma_t, kappa_V)
    return 2.0 * xr * xr


# ==============================================================================
# 1. EXACT COUPLED VARIATIONAL + DISSIPATIVE ENGINE (UN-CLAMPED, UN-FLOORED E_00)
# ==============================================================================

def eval_E00_and_components(state, sigma_t, lam, kappa_V, modes):
    """
    Evaluate EXACT un-clamped Friedmann constraint E_00(state) with TRUE un-floored
    relative residual (P1-1 fix: NO artificial 1.0 floor in denominator!).
    Also computes RMS metric perturbation sqrt(sum_n g_n b_n^2) and spatial curvature ratio.
    """
    x, h, phi, s = state[0], state[1], state[2], state[3]
    kap_rad = state[-1]
    inv_x2 = 1.0 / (x * x)
    x4 = x ** 4
    y = h * h + inv_x2

    e00_bg = (
        y - lam - y * y
        + sigma_t * (phi * (2.0 * y - phi) + 2.0 * h * s)
        - (kappa_V + kap_rad) * inv_x2 * inv_x2
    )

    e00_tt = 0.0
    kap_tt_list = []
    tt_scale = 0.0
    rms_b2_sum = 0.0
    dR3_over_R3_sum = 0.0
    for idx, m in enumerate(modes):
        b = state[4 + 2 * idx]
        v = state[5 + 2 * idx]
        g = m["g"]
        kn2 = m["kn2"]
        c_v = 1.0 + 2.0 * sigma_t * phi - 6.0 * h * h + (2.0 / 3.0) * inv_x2
        c_b = 1.0 + 2.0 * sigma_t * phi - 2.0 * h * h + (2.0 / 3.0) * inv_x2
        term = (g / 3.0) * (c_v * v * v + kn2 * inv_x2 * c_b * b * b - 8.0 * kn2 * h * inv_x2 * b * v)
        e00_tt -= term
        kap_m = (g / 3.0) * x4 * (v * v + kn2 * inv_x2 * b * b)
        kap_tt_list.append(kap_m)
        tt_scale = max(tt_scale, abs(term))
        rms_b2_sum += g * b * b
        dR3_over_R3_sum += (g * 2.0 * kn2 * b * b) / 6.0

    E00 = e00_bg + e00_tt
    # P1-1 fix: TRUE un-floored scale (no 1.0 floor!)
    scale = max(
        1e-30,
        abs(y),
        abs(y * y),
        abs(lam),
        abs(sigma_t * phi * (2.0 * y - phi)),
        abs(2.0 * sigma_t * h * s),
        abs((kappa_V + kap_rad) * inv_x2 * inv_x2),
        tt_scale,
    )
    A_R = 1.0 + 2.0 * sigma_t * phi - 2.0 * h * h + (2.0 / 3.0) * inv_x2
    kap_scal = x4 * (sigma_t * (phi - y) ** 2 + (sigma_t * s) ** 2)
    kap_tt_tot = sum(kap_tt_list)
    kap_eff_tot = 1.0 - sigma_t + kappa_V + kap_rad + kap_tt_tot

    return {
        "E00": float(E00),
        "rel_E00": float(abs(E00) / scale),
        "x4_E00": float(x4 * E00),
        "A_R": float(A_R),
        "kappa_scal": float(kap_scal),
        "kappa_tt_list": kap_tt_list,
        "kappa_tt_tot": float(kap_tt_tot),
        "kappa_rad": float(kap_rad),
        "kappa_eff_tot": float(kap_eff_tot),
        "rms_b": float(math.sqrt(rms_b2_sum)),
        "dR3_over_R3": float(dR3_over_R3_sum),
    }


def make_bounce_ic(x_b, sigma_t, lam, kappa_V, modes, b0_list=None, v0_list=None, kap_rad0=0.0, s_b=0.0, sign_hdot=+1):
    """Exact bounce IC solving E_00(0) == 0 to machine epsilon (~1e-16)."""
    n_modes = len(modes)
    if b0_list is None:
        b0_list = [0.0] * n_modes
    if v0_list is None:
        v0_list = [0.0] * n_modes
    y0 = 1.0 / (x_b * x_b)
    S_tot = 0.0
    for m, b0, v0 in zip(modes, b0_list, v0_list):
        S_tot += (m["g"] / 3.0) * (v0 * v0 + m["kn2"] * y0 * b0 * b0)

    c_lin = y0 - S_tot
    rhs_sq = c_lin * c_lin + (
        y0 - lam - y0 * y0 - (kappa_V + kap_rad0) * y0 * y0 - (1.0 + (2.0 / 3.0) * y0) * S_tot
    ) / sigma_t
    if rhs_sq < 0.0:
        return None
    phi0 = c_lin + sign_hdot * math.sqrt(rhs_sq)
    state = [float(x_b), 0.0, float(phi0), float(s_b)]
    for b0, v0 in zip(b0_list, v0_list):
        state.extend([float(b0), float(v0)])
    state.append(float(kap_rad0))
    return np.array(state, dtype=float)


def coupled_rhs(tau, state, sigma_t, lam, kappa_V, modes, gamma_scal_const=0.0, kap_scal_ref=None, bilinear_mod=True):
    """
    Exact coupled nonlinear ODE preserving d/dtau(x^4 * E_00) == 0 identically.
    Supports BOTH fixed-ring 1-loop rates AND radius-dependent shrinking-ring Parseval/MR
    interpolators (mode['x_interp'] != None, normalized to local kap_scal_ring = 2*x^2).
    """
    x = state[0]
    h = state[1]
    phi = state[2]
    s = state[3]
    inv_x2 = 1.0 / (x * x)
    x4 = x ** 4
    y = h * h + inv_x2

    if kap_scal_ref is None:
        kap_scal_ref = kappa_scal_ref_for_sigma(sigma_t, kappa_V)

    kap_scal_inst = x4 * (sigma_t * (phi - y) ** 2 + (sigma_t * s) ** 2)

    Gamma_scal = gamma_scal_const
    gam_b_list = []
    tt_r_shift = 0.0
    for idx, m in enumerate(modes):
        b = state[4 + 2 * idx]
        v = state[5 + 2 * idx]
        g = m["g"]
        kn2 = m["kn2"]
        tt_r_shift += g * (v * v - kn2 * inv_x2 * b * b)

        x_interp = m.get("x_interp", None)
        if x_interp is not None:
            # P0-1 fix: exact radius-dependent Parseval + Manley-Rowe rate at instantaneous x(tau)
            xc = min(max(x, x_interp["x_min"]), x_interp["x_max"])
            g_rad = float(x_interp["g_tot_fn"](xc))
            gam_b_ref = float(x_interp["gam_b_fn"](xc))
            kap_scal_local = 2.0 * x * x  # Parseval table evaluated on shrinking ring with kap_scal = 2*x^2
            scal_ratio = (kap_scal_inst / kap_scal_local) if (bilinear_mod and kap_scal_local > 0.0) else 1.0
            rate_scal_share = max(0.0, g_rad - gam_b_ref)
            if rate_scal_share > 0.0:
                kap_tt_m = (g / 3.0) * x4 * (v * v + kn2 * inv_x2 * b * b)
                Gamma_scal += 2.0 * rate_scal_share * (kap_tt_m / kap_scal_local)
            gam_b_list.append(gam_b_ref * scal_ratio)
        else:
            g_rad = m.get("gamma_rad_tot", 0.0)
            d_lam = m.get("delta_lam_1", 0.0)
            gam_b_ref = -d_lam
            scal_ratio = (kap_scal_inst / kap_scal_ref) if (bilinear_mod and kap_scal_ref > 0.0) else 1.0
            rate_scal_share = max(0.0, g_rad - gam_b_ref)
            if rate_scal_share > 0.0:
                kap_tt_m = (g / 3.0) * x4 * (v * v + kn2 * inv_x2 * b * b)
                Gamma_scal += 2.0 * rate_scal_share * (kap_tt_m / kap_scal_ref)
            gam_b_list.append(gam_b_ref * scal_ratio)

    # Step 1: exact dh from EL_phi = Q_phi
    delta_dh = -tt_r_shift / 3.0 + sigma_t * Gamma_scal * s
    dh = phi - 2.0 * h * h - inv_x2 + delta_dh

    # Step 2: exact dv_n = a_n from EL_{b_n} = Q_{b_n}
    A_R = 1.0 + 2.0 * sigma_t * phi - 2.0 * h * h + (2.0 / 3.0) * inv_x2
    K_R = 1.0 + 2.0 * sigma_t * phi + 2.0 * h * h + (2.0 / 3.0) * inv_x2 - 4.0 * (dh + h * h)
    dA_R = 2.0 * sigma_t * s - 4.0 * h * dh - (4.0 / 3.0) * h * inv_x2

    out = np.empty_like(state)
    out[0] = x * h
    out[1] = dh
    out[2] = s

    tr_tt = 0.0
    dkap_tt = 0.0
    for idx, m in enumerate(modes):
        b = state[4 + 2 * idx]
        v = state[5 + 2 * idx]
        g = m["g"]
        kn2 = m["kn2"]
        gam_b = gam_b_list[idx]
        a_n = -(3.0 * h + dA_R / A_R + 2.0 * gam_b) * v - (kn2 * K_R * inv_x2 / A_R) * b
        out[4 + 2 * idx] = v
        out[5 + 2 * idx] = a_n
        tr_tt += g * (
            (-2.0 * (1.0 + 2.0 * sigma_t * phi) - 12.0 * h * h - 4.0 * dh - 8.0 * kn2 * inv_x2) * v * v
            + (kn2 * inv_x2) * (2.0 * (1.0 + 2.0 * sigma_t * phi) - 4.0 * h * h - 4.0 * dh) * b * b
            - 16.0 * kn2 * h * inv_x2 * b * v
            - 8.0 * h * v * a_n
            - 8.0 * kn2 * inv_x2 * b * a_n
        )
        dkap_tt += (4.0 / 3.0) * g * gam_b * x4 * A_R * v * v

    # Step 3: exact ds from Trace equation (x*EL_x - 3*x^3*E_00 = 0)
    out[3] = -3.0 * h * s + (
        2.0 * lam - phi + 2.0 * (phi - y) * y - (1.0 + 2.0 * sigma_t * phi - 2.0 * y) * delta_dh + tr_tt / 6.0
    ) / sigma_t

    # Step 4: exact d(kappa_rad)/dtau from dissipative Noether identity
    out[-1] = 2.0 * (sigma_t ** 2) * x4 * Gamma_scal * s * s + dkap_tt
    return out


def integrate_coupled(
    x_b,
    sigma_t,
    lam=0.0,
    kappa_V=KAPPA_V,
    modes=None,
    b0_list=None,
    v0_list=None,
    gamma_scal_const=0.0,
    bilinear_mod=True,
    tau_end=10000.0,
    method="DOP853",
    rtol=1e-10,
    atol=1e-12,
    max_step=0.25,
    n_eval=4001,
    x_exit_cap=2000.0,
):
    """Integrate the exact coupled system without silent clamping and log pre-singularity safe diagnostics."""
    if modes is None:
        modes = []
    y0 = make_bounce_ic(x_b, sigma_t, lam, kappa_V, modes, b0_list=b0_list, v0_list=v0_list)
    if y0 is None:
        return {"success": False, "outcome": "no_turning_point"}

    kap_scal_ref = kappa_scal_ref_for_sigma(sigma_t, kappa_V)

    def rhs_w(t, y):
        return coupled_rhs(
            t, y, sigma_t, lam, kappa_V, modes,
            gamma_scal_const=gamma_scal_const, kap_scal_ref=kap_scal_ref, bilinear_mod=bilinear_mod,
        )

    def ev_blowup(t, y):
        return 500.0 - max(abs(y[1]), abs(y[2]))
    ev_blowup.terminal = True
    ev_blowup.direction = -1

    def ev_collapse(t, y):
        return y[0] - 0.05
    ev_collapse.terminal = True
    ev_collapse.direction = -1

    def ev_ghost(t, y):
        inv_x2 = 1.0 / (y[0] * y[0])
        A_R = 1.0 + 2.0 * sigma_t * y[2] - 2.0 * y[1] * y[1] + (2.0 / 3.0) * inv_x2
        return A_R - 1e-3
    ev_ghost.terminal = True
    ev_ghost.direction = -1

    def ev_exit(t, y):
        return x_exit_cap - y[0]
    ev_exit.terminal = True
    ev_exit.direction = -1

    t_eval = np.linspace(0.0, tau_end, n_eval)
    sol = solve_ivp(
        rhs_w, (0.0, tau_end), y0, method=method, rtol=rtol, atol=atol,
        max_step=max_step, t_eval=t_eval, events=[ev_blowup, ev_collapse, ev_ghost, ev_exit],
    )

    ts = sol.t
    ys = sol.y
    ev_hit = None
    for idx_e, lbl_e in enumerate(["blowup", "collapse", "ghost_A_R_zero", "exit"]):
        if len(sol.t_events[idx_e]) > 0:
            ev_hit = lbl_e
            ts = np.append(ts, sol.t_events[idx_e][0])
            ys = np.column_stack([ys, sol.y_events[idx_e][0]])
            break

    n_pts = len(ts)
    rel_e00 = np.empty(n_pts)
    abs_x4_e00 = np.empty(n_pts)
    A_R_arr = np.empty(n_pts)
    kap_scal_arr = np.empty(n_pts)
    kap_tt_arr = np.empty(n_pts)
    kap_rad_arr = np.empty(n_pts)
    kap_eff_arr = np.empty(n_pts)
    rms_b_arr = np.empty(n_pts)
    dR3_arr = np.empty(n_pts)
    max_b_per_mode = [0.0] * len(modes)

    for i in range(n_pts):
        diag = eval_E00_and_components(ys[:, i], sigma_t, lam, kappa_V, modes)
        rel_e00[i] = diag["rel_E00"]
        abs_x4_e00[i] = abs(diag["x4_E00"])
        A_R_arr[i] = diag["A_R"]
        kap_scal_arr[i] = diag["kappa_scal"]
        kap_tt_arr[i] = diag["kappa_tt_tot"]
        kap_rad_arr[i] = diag["kappa_rad"]
        kap_eff_arr[i] = diag["kappa_eff_tot"]
        rms_b_arr[i] = diag["rms_b"]
        dR3_arr[i] = diag["dR3_over_R3"]
        for m_i in range(len(modes)):
            max_b_per_mode[m_i] = max(max_b_per_mode[m_i], abs(float(ys[4 + 2 * m_i, i])))

    x_end_val = float(ys[0, -1])
    h_end_val = float(ys[1, -1])
    phi_end_val = float(ys[2, -1])
    rho_scal_end = float(3.0 * kap_scal_arr[-1] / (x_end_val ** 4))
    rho_rad_end = float(3.0 * kap_rad_arr[-1] / (x_end_val ** 4))
    rho_lam = float(3.0 * lam)

    if ev_hit == "blowup":
        outcome = "blowup_pos" if ys[2, -1] > 0 else "blowup_neg"
    elif ev_hit == "collapse":
        outcome = "collapse"
    elif ev_hit == "ghost_A_R_zero":
        outcome = "ghost_A_R_zero"
    elif ev_hit == "exit":
        if lam > 0.0 and rho_scal_end < rho_lam and abs(h_end_val - math.sqrt(lam)) / math.sqrt(lam) < 0.25:
            outcome = "ir_dS_exit"
        else:
            outcome = "starobinsky_inflation_exit"
    else:
        outcome = "kam_cyclic_survived"

    # Pre-singularity safe index (last point with A_R >= 0.1 and |phi| <= 50.0) [P1-6 fix]
    safe_mask = (A_R_arr >= 0.1) & (np.abs(ys[2]) <= 50.0)
    idx_safe = int(np.where(safe_mask)[0][-1]) if np.any(safe_mask) else 0
    max_b_safe = [
        float(np.max(np.abs(ys[4 + 2 * m_i, :idx_safe + 1]))) for m_i in range(len(modes))
    ]
    end_b_per_mode = [abs(float(ys[4 + 2 * m_i, -1])) for m_i in range(len(modes))]

    xs = ys[0]
    kap_scal_0 = float(kap_scal_arr[0])
    kap_scal_safe = float(kap_scal_arr[idx_safe])
    depletion_frac_safe = 1.0 - (kap_scal_safe / kap_scal_0) if kap_scal_0 > 0 else 0.0

    cross_idx = np.where(kap_eff_arr >= 0.0)[0]
    tau_kappa_flip = float(ts[cross_idx[0]]) if len(cross_idx) > 0 else None

    m_early = (ts >= 0.05 * ts[-1]) & (ts <= 0.20 * ts[-1])
    m_late = (ts >= 0.80 * ts[-1]) & (ts <= 0.98 * ts[-1])
    x_early_min = float(np.min(xs[m_early])) if np.any(m_early) else float(np.min(xs))
    x_early_max = float(np.max(xs[m_early])) if np.any(m_early) else float(np.max(xs))
    x_late_min = float(np.min(xs[m_late])) if np.any(m_late) else float(xs[-1])
    x_late_max = float(np.max(xs[m_late])) if np.any(m_late) else float(xs[-1])

    return {
        "success": bool(sol.success),
        "method": method,
        "outcome": outcome,
        "tau_end": float(ts[-1]),
        "tau_safe": float(ts[idx_safe]),
        "tau_kappa_eff_zero_flip": tau_kappa_flip,
        "min_A_R": float(np.min(A_R_arr)),
        "x_min": float(np.min(xs)),
        "x_max": float(np.max(xs)),
        "x_end": x_end_val,
        "h_end": h_end_val,
        "phi_end": phi_end_val,
        "rho_scal_end": rho_scal_end,
        "rho_lam": rho_lam,
        "x_early_range": [x_early_min, x_early_max],
        "x_late_range": [x_late_min, x_late_max],
        "kappa_scal_0": kap_scal_0,
        "kappa_scal_safe": kap_scal_safe,
        "scalaron_depletion_frac_safe": float(depletion_frac_safe),
        "kappa_rad_safe": float(kap_rad_arr[idx_safe]),
        "kappa_rad_end": float(kap_rad_arr[-1]),
        "rho_rad_end": rho_rad_end,
        "kappa_tt_0": float(kap_tt_arr[0]),
        "kappa_tt_max_safe": float(np.max(kap_tt_arr[:idx_safe + 1])),
        "kappa_tt_end": float(kap_tt_arr[-1]),
        "kappa_eff_0": float(kap_eff_arr[0]),
        "kappa_eff_safe": float(kap_eff_arr[idx_safe]),
        "kappa_eff_end": float(kap_eff_arr[-1]),
        "max_abs_b_per_mode": max_b_per_mode,
        "max_abs_b_safe": max_b_safe,
        "end_abs_b_per_mode": end_b_per_mode,
        "max_rms_b_safe": float(np.max(rms_b_arr[:idx_safe + 1])),
        "max_dR3_over_R3_safe": float(np.max(dR3_arr[:idx_safe + 1])),
        "b_exceeded_0p15_safe": bool(max(max_b_safe, default=0.0) > 0.15),
        "max_rel_E00": float(np.max(rel_e00)),
        "max_rel_E00_safe": float(np.max(rel_e00[:idx_safe + 1])),
        "max_abs_x4_E00": float(np.max(abs_x4_e00)),
        "init_rel_E00": float(rel_e00[0]),
        "_ts": ts,
        "_ys": ys,
        "_A_R": A_R_arr,
        "_kap_scal": kap_scal_arr,
        "_kap_tt": kap_tt_arr,
        "_kap_rad": kap_rad_arr,
        "_kap_eff": kap_eff_arr,
        "_rel_e00": rel_e00,
    }


def strip_arrays(res):
    return {k: v for k, v in res.items() if not k.startswith("_")}


def build_five_modes(rate_tbl, g2=1.0, n2_x_interp=None):
    """Build the 5 S^3/I* TT mode dicts from a reproducible rate table in p2_02_01_identities.json."""
    modes = []
    for n in [2, 10, 14, 18, 22]:
        info = rate_tbl[str(n)]
        m = {
            "n": n,
            "g": float(g2) if n == 2 else float(info["mult"]),
            "kn2": float(n * (n + 2)),
            "gamma_rad_tot": float(info["gamma_rad_tot"]),
            "delta_lam_1": float(info["delta_lam_1"]),
        }
        if n == 2 and n2_x_interp is not None:
            m["x_interp"] = n2_x_interp
        modes.append(m)
    return modes


# ==============================================================================
# 2. FOUR SYSTEMATIC EXPERIMENTS (ROUND 2 — ALL P0/P1 REMEDIATED)
# ==============================================================================

def run_exp_A_mathieu_backreaction(rate_tables):
    """
    Exp A: Nonlinear backreaction of n=18 (g=17) & n=22 (g=46) Mathieu gravitons at sigma_t0 = 9/31.
    Uses exact upper-sideband Manley-Rowe scalaron share (~68.4% for n=18, ~64.1% for n=22) from 01_derive.py.
    """
    print("\n=== EXP A: NONLINEAR MATHIEU REHEATING BACKREACTION (n=18, 22 at sigma_t0 = 9/31) ===", flush=True)
    tbl0 = rate_tables["five_mode_tables_by_sigma"]["sigma0_9over31"]
    rows = []
    for diss_flag in [False, True]:
        m18 = {
            "n": 18, "g": 17.0, "kn2": 360.0,
            "gamma_rad_tot": tbl0["18"]["gamma_rad_tot"] if diss_flag else 0.0,
            "delta_lam_1": tbl0["18"]["delta_lam_1"] if diss_flag else 0.0,
        }
        m22 = {
            "n": 22, "g": 46.0, "kn2": 528.0,
            "gamma_rad_tot": tbl0["22"]["gamma_rad_tot"] if diss_flag else 0.0,
            "delta_lam_1": tbl0["22"]["delta_lam_1"] if diss_flag else 0.0,
        }
        modes = [m18, m22]
        for lam in [0.0, 5e-5, 1e-4]:
            b_seeds = [1e-5, 1e-4, 1e-3, 1e-2] if lam == 0.0 else [1e-4, 1e-3]
            for b0 in b_seeds:
                res = integrate_coupled(
                    X_RING_0, SIGMA_0, lam=lam, modes=modes, b0_list=[b0, b0],
                    tau_end=3000.0, max_step=0.15, n_eval=2001,
                )
                s = strip_arrays(res)
                s.update({"diss_1loop": diss_flag, "lam": lam, "b0_seed": b0})
                rows.append(s)
                fl_s = f"{s['tau_kappa_eff_zero_flip']:6.1f}" if s["tau_kappa_eff_zero_flip"] is not None else "  None"
                print(
                    f"  diss={str(diss_flag):5s} lam={lam:7.1e} b0={b0:7.1e} -> {s['outcome']:16s} "
                    f"tau={s['tau_end']:6.1f} flip={fl_s} | k_rad_safe={s['kappa_rad_safe']:6.1f} "
                    f"rms_b={s['max_rms_b_safe']:.3f} dR3/R3={s['max_dR3_over_R3_safe']:5.1f} relE00={s['max_rel_E00']:.2e}",
                    flush=True,
                )
    return rows


def run_exp_B_shear_and_scalaron_depletion(rate_tables, n2_interp):
    """
    Exp B (P0-1 remediated):
      1. Exact radius-dependent shrinking-ring n=2 catalyzed scalaron->2gamma depletion
         across b_0 in [0.02, 0.30] for BOTH g_2=1 (single axisymmetric Bianchi IX mode)
         AND g_2=5 (full 5-polarization multiplet), plus non-dissipative baseline!
      2. Complete direct (Gamma_scal, lambda) phase map including G_scal=2e-4 and lam=1.4e-4
         so every cited exit/crash window is archived verbatim.
    """
    print("\n=== EXP B: n=2 SHEAR + RADIUS-DEPENDENT CATALYZED DEPLETION (sigma_t0 = 9/31, P0-1 FIX) ===", flush=True)
    m2_nodiss = {"n": 2, "g": 1.0, "kn2": 8.0, "gamma_rad_tot": 0.0, "delta_lam_1": 0.0}
    m2_xdep_g1 = {"n": 2, "g": 1.0, "kn2": 8.0, "x_interp": n2_interp}
    m2_xdep_g5 = {"n": 2, "g": 5.0, "kn2": 8.0, "x_interp": n2_interp}

    shear_rows = []
    # (1) Pure variational non-dissipative baseline (g_2 = 1)
    for b0 in [0.05, 0.10, 0.15, 0.20]:
        res = integrate_coupled(X_RING_0, SIGMA_0, lam=0.0, modes=[m2_nodiss], b0_list=[b0],
                                tau_end=10000.0, max_step=0.25, n_eval=2501)
        s = strip_arrays(res)
        s.update({"mode_label": "nodiss_g1", "g2": 1.0, "lam": 0.0, "b0_shear": b0})
        shear_rows.append(s)
        print(f"  [nodiss g2=1] b0={b0:4.2f} -> {s['outcome']:20s} tau={s['tau_end']:7.1f} "
              f"x_late={[round(v,2) for v in s['x_late_range']]} max|b|={s['max_abs_b_safe'][0]:.4f} relE00={s['max_rel_E00']:.2e}", flush=True)

    # (2) Radius-dependent Parseval/MR catalyzed depletion (g_2 = 1, lam = 0 and 5e-5, 1e-4)
    for lam in [0.0, 5e-5, 1e-4]:
        b_grid = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30] if lam == 0.0 else [0.05, 0.10, 0.20]
        for b0 in b_grid:
            res = integrate_coupled(X_RING_0, SIGMA_0, lam=lam, modes=[m2_xdep_g1], b0_list=[b0],
                                    tau_end=10000.0, max_step=0.25, n_eval=2501)
            s = strip_arrays(res)
            s.update({"mode_label": "xdep_cat_g1", "g2": 1.0, "lam": lam, "b0_shear": b0})
            shear_rows.append(s)
            print(
                f"  [xdep g2=1] lam={lam:6.1e} b0={b0:4.2f} -> {s['outcome']:20s} tau={s['tau_end']:7.1f} "
                f"k_rad={s['kappa_rad_safe']:6.1f} x_late={[round(v,2) for v in s['x_late_range']]} "
                f"max|b|={s['max_abs_b_safe'][0]:.3f} end|b|={s['end_abs_b_per_mode'][0]:.4f} relE00={s['max_rel_E00']:.2e}",
                flush=True,
            )

    # (3) Explicit g_2 = 5 full-multiplet comparison (P1-3 fix)
    for b0 in [0.02, 0.05, 0.10, 0.15]:
        res = integrate_coupled(X_RING_0, SIGMA_0, lam=0.0, modes=[m2_xdep_g5], b0_list=[b0],
                                tau_end=10000.0, max_step=0.25, n_eval=2501)
        s = strip_arrays(res)
        s.update({"mode_label": "xdep_cat_g5", "g2": 5.0, "lam": 0.0, "b0_shear": b0})
        shear_rows.append(s)
        print(
            f"  [xdep g2=5] lam=0.0e+00 b0={b0:4.2f} -> {s['outcome']:20s} tau={s['tau_end']:7.1f} "
            f"k_rad={s['kappa_rad_safe']:6.1f} x_late={[round(v,2) for v in s['x_late_range']]} "
            f"max|b|={s['max_abs_b_safe'][0]:.3f} end|b|={s['end_abs_b_per_mode'][0]:.4f} relE00={s['max_rel_E00']:.2e}",
            flush=True,
        )

    # (4) Direct constant-Gamma_scal phase diagram across (Gamma_scal, lambda) — complete archived grid!
    print("\n  [Direct Scalaron->Radiation Phase Boundary across (Gamma_scal, lambda) at sigma_t0 = 9/31]:", flush=True)
    gamma_rows = []
    for lam in [0.0, 2e-5, 5e-5, 1e-4, 1.4e-4]:
        for G_s in [1e-5, 3e-5, 1e-4, 2e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
            res = integrate_coupled(
                X_RING_0, SIGMA_0, lam=lam, modes=[], gamma_scal_const=G_s,
                tau_end=10000.0, max_step=0.25, n_eval=2001,
            )
            s = strip_arrays(res)
            s.update({"lam": lam, "Gamma_scal": G_s})
            gamma_rows.append(s)
            print(
                f"    lam={lam:7.1e} G_scal={G_s:7.1e} -> {s['outcome']:24s} tau={s['tau_end']:7.1f} "
                f"k_rad_safe={s['kappa_rad_safe']:6.1f} x_end={s['x_end']:7.2f} relE00={s['max_rel_E00']:.2e}",
                flush=True,
            )

    return {"n2_shear_catalyzed_runs": shear_rows, "direct_gamma_scal_phase_map": gamma_rows}


def run_exp_C_sub_unitarity_eft_and_spectral_desert(rate_tables):
    """
    Exp C (P1-3 & P1-4 remediated):
      1. Full sigma_t sweep across [9/31, 17.5] with EXACT round-S^3 TT degeneracies
         2*(n+3)*(n-1)={24,42,64,90} and EXACT matched per-mode initial energy 0.002/mode
         (eliminating the 1.25x 4-vs-5 seeding asymmetry!).
      2. Includes sigma_t = 4.20 to capture the razor-narrow n=10 1:1 secondary Mathieu tongue
         (tested both pure variational AND with 1-loop 2-photon damping!).
      3. EFT-safe sigma_t = 10.0 n=10 (M_I*=14) finite-amplitude damping sweep.
    """
    print("\n=== EXP C: SUB-UNITARITY EFT WINDOW [4, 12], I* SPECTRAL DESERT & n=10 1:1 TONGUE ===", flush=True)
    sigma_grid = [SIGMA_0, 0.5, 1.0, 2.0, 4.0, 4.20, 5.05, 6.0, 8.60, 10.0, 12.0, 14.0, 15.5, 16.5, 17.0]

    desert_scan = []
    kap_per_mode = 0.002  # Exact per-mode matched initial energy!
    for sig in sigma_grid:
        xr = x_ring_for_sigma(sig)
        eps_scal = 4.0 * C_ANOM / sig
        modes_istar = [
            {"n": 2,  "g": 5.0,  "kn2": 8.0,   "gamma_rad_tot": 0.0, "delta_lam_1": 0.0},
            {"n": 10, "g": 9.0,  "kn2": 120.0, "gamma_rad_tot": 0.0, "delta_lam_1": 0.0},
            {"n": 14, "g": 17.0, "kn2": 224.0, "gamma_rad_tot": 0.0, "delta_lam_1": 0.0},
            {"n": 18, "g": 17.0, "kn2": 360.0, "gamma_rad_tot": 0.0, "delta_lam_1": 0.0},
            {"n": 22, "g": 46.0, "kn2": 528.0, "gamma_rad_tot": 0.0, "delta_lam_1": 0.0},
        ]
        # Exact round-S^3 TT degeneracies 2*(n+3)*(n-1): n=3->24, n=4->42, n=5->64, n=6->90
        modes_s3_forb = [
            {"n": 3, "g": 24.0, "kn2": 15.0, "gamma_rad_tot": 0.0, "delta_lam_1": 0.0},
            {"n": 4, "g": 42.0, "kn2": 24.0, "gamma_rad_tot": 0.0, "delta_lam_1": 0.0},
            {"n": 5, "g": 64.0, "kn2": 35.0, "gamma_rad_tot": 0.0, "delta_lam_1": 0.0},
            {"n": 6, "g": 90.0, "kn2": 48.0, "gamma_rad_tot": 0.0, "delta_lam_1": 0.0},
        ]
        b0_istar = [math.sqrt(3.0 * kap_per_mode / (m["g"] * m["kn2"] * xr * xr)) for m in modes_istar]
        b0_forb = [math.sqrt(3.0 * kap_per_mode / (m["g"] * m["kn2"] * xr * xr)) for m in modes_s3_forb]

        r_I = integrate_coupled(xr, sig, lam=0.0, modes=modes_istar, b0_list=b0_istar, tau_end=4000.0, max_step=0.25, n_eval=1501)
        r_F = integrate_coupled(xr, sig, lam=0.0, modes=modes_s3_forb, b0_list=b0_forb, tau_end=4000.0, max_step=0.25, n_eval=1501)
        sI = strip_arrays(r_I)
        sF = strip_arrays(r_F)

        dom_I_idx = int(np.argmax([sI["max_abs_b_per_mode"][i] / b0_istar[i] for i in range(len(modes_istar))]))
        dom_F_idx = int(np.argmax([sF["max_abs_b_per_mode"][i] / b0_forb[i] for i in range(len(modes_s3_forb))]))
        amp_I = sI["max_abs_b_per_mode"][dom_I_idx] / b0_istar[dom_I_idx]
        amp_F = sF["max_abs_b_per_mode"][dom_F_idx] / b0_forb[dom_F_idx]

        row = {
            "sigma_t": sig,
            "x_ring": xr,
            "eps_scal_4c_wscal2": eps_scal,
            "kap_per_mode_init": kap_per_mode,
            "Istar_outcome": sI["outcome"],
            "Istar_tau_end": sI["tau_end"],
            "Istar_kappa_tt_max_safe": sI["kappa_tt_max_safe"],
            "Istar_dom_mode_n": modes_istar[dom_I_idx]["n"],
            "Istar_dom_amp_gain": float(amp_I),
            "Istar_max_rel_E00": sI["max_rel_E00"],
            "S3_forb_outcome": sF["outcome"],
            "S3_forb_tau_end": sF["tau_end"],
            "S3_forb_kappa_tt_max_safe": sF["kappa_tt_max_safe"],
            "S3_forb_dom_mode_n": modes_s3_forb[dom_F_idx]["n"],
            "S3_forb_dom_amp_gain": float(amp_F),
            "S3_forb_max_rel_E00": sF["max_rel_E00"],
        }
        desert_scan.append(row)
        print(
            f"  sig={sig:6.3f} (eps={eps_scal:4.2f}): I*->{sI['outcome']:20s} (tau={sI['tau_end']:6.1f}, "
            f"kTTsafe={sI['kappa_tt_max_safe']:6.2f}, dom n={row['Istar_dom_mode_n']:2d} x{amp_I:6.1f}) | "
            f"S3_forb->{sF['outcome']:20s} (tau={sF['tau_end']:6.1f}, kTTsafe={sF['kappa_tt_max_safe']:6.2f}, "
            f"dom n={row['S3_forb_dom_mode_n']} x{amp_F:6.1f})",
            flush=True,
        )

    # Check 1-loop 2-photon damping quenching of the narrow n=10 1:1 tongue at sigma_t = 4.20!
    xr42 = x_ring_for_sigma(4.20)
    tbl4 = rate_tables["five_mode_tables_by_sigma"]["sigma_4p0"]
    b0_10_42 = math.sqrt(3.0 * kap_per_mode / (9.0 * 120.0 * xr42 * xr42))
    r_42_nod = integrate_coupled(
        xr42, 4.20, lam=0.0,
        modes=[{"n": 10, "g": 9.0, "kn2": 120.0, "gamma_rad_tot": 0.0, "delta_lam_1": 0.0}],
        b0_list=[b0_10_42], tau_end=4000.0, max_step=0.25, n_eval=1501,
    )
    r_42_diss = integrate_coupled(
        xr42, 4.20, lam=0.0,
        modes=[{"n": 10, "g": 9.0, "kn2": 120.0,
                "gamma_rad_tot": tbl4["10"]["gamma_rad_tot"], "delta_lam_1": tbl4["10"]["delta_lam_1"]}],
        b0_list=[b0_10_42], tau_end=4000.0, max_step=0.25, n_eval=1501,
    )
    n10_1to1_check = {
        "sigma_t": 4.20,
        "nodiss_outcome": r_42_nod["outcome"],
        "nodiss_gain": float(r_42_nod["max_abs_b_per_mode"][0] / b0_10_42),
        "nodiss_kappa_tt_max": r_42_nod["kappa_tt_max_safe"],
        "diss_1loop_outcome": r_42_diss["outcome"],
        "diss_1loop_gain": float(r_42_diss["max_abs_b_per_mode"][0] / b0_10_42),
        "diss_1loop_end_gain": float(r_42_diss["end_abs_b_per_mode"][0] / b0_10_42),
        "diss_1loop_kappa_rad": r_42_diss["kappa_rad_safe"],
    }
    print(
        f"  [n=10 1:1 tongue sigma=4.20] nodiss gain={n10_1to1_check['nodiss_gain']:.2f}x (kTT={n10_1to1_check['nodiss_kappa_tt_max']:.2f}) | "
        f"+1-loop damp gain={n10_1to1_check['diss_1loop_gain']:.2f}x -> end={n10_1to1_check['diss_1loop_end_gain']:.3f}x",
        flush=True,
    )

    # Part C2: EFT-safe sigma_t = 10.0 n=10 (M_I*=14) finite-amplitude damping sweep + unperturbed min_A_R
    print("\n  [EFT-Safe sigma_t = 10.0: unperturbed baseline & n=10 (M_I*=14) damping sweep over tau=10000]:", flush=True)
    xr10 = x_ring_for_sigma(10.0)
    r_p10 = integrate_coupled(xr10, 10.0, lam=0.0, modes=[], tau_end=10000.0, max_step=0.25, n_eval=2001)
    print(f"    Pure P05 background sigma=10.0: {r_p10['outcome']} x in [{r_p10['x_min']:.2f}, {r_p10['x_max']:.2f}], min_A_R={r_p10['min_A_R']:.4f}", flush=True)
    tbl10 = rate_tables["five_mode_tables_by_sigma"]["sigma_10p0"]
    m10_s10 = {"n": 10, "g": 9.0, "kn2": 120.0, "gamma_rad_tot": tbl10["10"]["gamma_rad_tot"], "delta_lam_1": tbl10["10"]["delta_lam_1"]}
    m2_s10 = {"n": 2, "g": 1.0, "kn2": 8.0, "gamma_rad_tot": tbl10["2"]["gamma_rad_tot"], "delta_lam_1": tbl10["2"]["delta_lam_1"]}
    eft_s10_runs = []
    for kap_n10_init in [1.0, 10.0, 50.0, 150.0]:
        b0_10 = math.sqrt(3.0 * kap_n10_init / (9.0 * 120.0 * xr10 * xr10))
        res = integrate_coupled(
            xr10, 10.0, lam=0.0, modes=[m2_s10, m10_s10], b0_list=[0.05, b0_10],
            tau_end=10000.0, max_step=0.25, n_eval=2501,
        )
        s = strip_arrays(res)
        s.update({"sigma_t": 10.0, "g2": 1.0, "kappa_n10_init": kap_n10_init, "b0_10": b0_10, "b0_2": 0.05})
        eft_s10_runs.append(s)
        print(
            f"    sig=10.0 kap_n10(0)={kap_n10_init:5.1f}: {s['outcome']:20s} tau={s['tau_end']:7.1f} "
            f"k_rad_safe={s['kappa_rad_safe']:6.2f} relE00={s['max_rel_E00']:.2e}",
            flush=True,
        )

    return {
        "sigma_t_spectral_desert_scan": desert_scan,
        "n10_1to1_tongue_sigma4p20": n10_1to1_check,
        "pure_bg_sigma10": strip_arrays(r_p10),
        "eft_sigma10_n10_damping_runs": eft_s10_runs,
    }


def run_exp_D_unified_multichannel(rate_tables, n2_interp):
    """
    Exp D (P0-2 & P1-2 remediated):
    Full unified 5-mode S^3/I* coupled system across ALL sigma_t in {9/31, 4, 6, 8, 10, 12}
    using each sigma_t's own reproducible Parseval rate table from 01_derive.py (plus g2=1 and g2=5 check!).
    """
    print("\n=== EXP D: UNIFIED 5-MODE S^3/I* COUPLED BACKREACTION (ALL sigma_t = 9/31, 4, 6, 8, 10, 12) ===", flush=True)
    tbls = rate_tables["five_mode_tables_by_sigma"]
    configs = [
        ("sigma0_lam0",     SIGMA_0, 0.0,  "sigma0_9over31", 1.0, n2_interp, [0.05, 1e-4, 1e-4, 1e-4, 1e-4], 5000.0),
        ("sigma0_lam5e5",   SIGMA_0, 5e-5, "sigma0_9over31", 1.0, n2_interp, [0.05, 1e-4, 1e-4, 1e-4, 1e-4], 5000.0),
        ("sigma4_lam0_g1",  4.0,     0.0,  "sigma_4p0",      1.0, None,      [0.05, 1e-3, 1e-3, 1e-3, 1e-3], 10000.0),
        ("sigma4_lam0_g5",  4.0,     0.0,  "sigma_4p0",      5.0, None,      [0.05, 1e-3, 1e-3, 1e-3, 1e-3], 10000.0),
        ("sigma4_lam5e5",   4.0,     5e-5, "sigma_4p0",      1.0, None,      [0.05, 1e-3, 1e-3, 1e-3, 1e-3], 10000.0),
        ("sigma6_lam0",     6.0,     0.0,  "sigma_6p0",      1.0, None,      [0.05, 1e-3, 1e-3, 1e-3, 1e-3], 10000.0),
        ("sigma8_lam0",     8.0,     0.0,  "sigma_8p0",      1.0, None,      [0.05, 1e-3, 1e-3, 1e-3, 1e-3], 10000.0),
        ("sigma10_lam0_05", 10.0,    0.0,  "sigma_10p0",     1.0, None,      [0.05, 1e-3, 1e-3, 1e-3, 1e-3], 10000.0),
        ("sigma10_lam0_02", 10.0,    0.0,  "sigma_10p0",     1.0, None,      [0.02, 1e-3, 1e-3, 1e-3, 1e-3], 10000.0),
        ("sigma10_lam5e5",  10.0,    5e-5, "sigma_10p0",     1.0, None,      [0.05, 1e-3, 1e-3, 1e-3, 1e-3], 10000.0),
        ("sigma12_lam0",    12.0,    0.0,  "sigma_12p0",     1.0, None,      [0.05, 1e-3, 1e-3, 1e-3, 1e-3], 10000.0),
    ]
    runs = []
    for label, sig, lam, tbl_key, g2, interp, b0s, t_end in configs:
        xr = x_ring_for_sigma(sig)
        modes = build_five_modes(tbls[tbl_key], g2=g2, n2_x_interp=interp)
        res = integrate_coupled(
            xr, sig, lam=lam, modes=modes, b0_list=b0s, tau_end=t_end, max_step=0.20, n_eval=2501,
        )
        s = strip_arrays(res)
        s.update({"label": label, "sigma_t": sig, "lam": lam, "g2": g2, "b0_list": b0s})
        runs.append(s)
        print(
            f"  {label:16s} (sig={sig:5.2f}, g2={g2:.0f}, lam={lam:6.1e}): {s['outcome']:20s} "
            f"tau={s['tau_end']:7.1f} k_rad_safe={s['kappa_rad_safe']:6.2f} k_tt_safe={s['kappa_tt_max_safe']:6.2f} "
            f"relE00={s['max_rel_E00']:.2e}",
            flush=True,
        )
    return runs


def main():
    print("=== STREAM p2_02 (CLM-OPEN-02) STAGE 2 (ROUND 4): NONLINEAR BACKREACTION SOLVER ===", flush=True)
    rate_tables = load_rate_tables()
    n2_interp = build_n2_x_dependent_interpolator(rate_tables)

    exp_A = run_exp_A_mathieu_backreaction(rate_tables)
    exp_B = run_exp_B_shear_and_scalaron_depletion(rate_tables, n2_interp)
    exp_C = run_exp_C_sub_unitarity_eft_and_spectral_desert(rate_tables)
    exp_D = run_exp_D_unified_multichannel(rate_tables, n2_interp)

    payload = {
        "stream_id": "p2_02_nonlinear_kam_backreaction",
        "stage": "02_solve",
        "round": 4,
        "exp_A_mathieu_n18_n22": exp_A,
        "exp_B_shear_and_scalaron_depletion": exp_B,
        "exp_C_sub_unitarity_eft_and_spectral_desert": exp_C,
        "exp_D_unified_multichannel": exp_D,
    }
    out_path = os.path.join(OUT_DIR, "p2_02_02_dynamics.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
