#!/usr/bin/env python3
"""
Stream p2_07_complete_model_b_kam_dynamics (CLM-P2-07)
Stage 2 (Round 2 Post-Audit): Nonlinear Coupled Solvers for KT1, KT2, and KT3.

Round-2 Cold-Audit Remediations Incorporated:
  - [P0-2] Replaced arbitrary-horizon non-monodromy eigvals(Phi(1500)) with QR-orthonormalized
           Benettin Lyapunov exponents over tau = 4000 (20 blocks x 200, verifying symplectic
           |det(Phi_k)| = 1 per block and tau=2000 vs tau=4000 horizon convergence); added sig=9.50
           to unified_5mode_scan so every Table 2.1 row is directly computed.
  - [P0-3] Adopted canonical kinetic-normalized Misner exponential map lam_B = exp(sqrt(3)*b_2)
           as primary in KT3 (with explicit linear lam_B = 1 + sqrt(3)*b_2 comparison quantifying
           the O(b_2^2) parameterization sensitivity).
  - [P0-4] Replaced tautological 1st-order y'=Ez, z'=-Ey ODE with the genuine physical 2nd-order
           time-varying harmonic oscillator q_p' = p_p, p_p' = -E_p(eta)^2 q_p, verifying exact
           symplectic Wronskian conservation W_p = q1*p2 - q2*p1 === 1 and honestly reporting the
           true O(Omega_2*b_0/E_p) non-adiabatic Bogoliubov difference vs the WKB phase integral.
  - [P1-1] Reported shift-invariant first-echo peak peak_near_pi2_over_K0 and its half-amplitude
           threshold alongside fixed-time K(pi/2)/K(0) and unconfounded [10*pi, 20*pi] max_late.
  - [P1-2] Removed silent max(0.0, 1.0 - pk/x^2) clamp (using exact A_eff = A_R - pk/x^2 >= 1e-3);
           logged exact PCHIP clamp telemetry on every run; filtered absMS table explicitly via
           parametric_unstable_flagged from 01_derive.json.
  - [P1-3] Directly reproduced cand_0 finite-amplitude (g_2 = 5, b0_2 = 0.005) nonlinear checks at
           sig = 9.0, 9.25, 9.5 inside 02_solve.py (ghost_A_R2_xkin crash at sig = 9.0; nonlinear
           detuning saturation at sig = 9.25, 9.5); scanned fine sub-desert upper edge (g_2 = 1,
           b0_2 = 0.05) at sig in {4.5, 5.0, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0} (relS3) and {4.5, 5.0, 5.5} (absMS).
  - [P1-4] Added matched bare-p2_02-background discrete-bath control in KT2 alongside Complete Model B.
  - [P1-5] Extended KT3 Landau phase-mixing test to eta in [0, 30*pi] to expose finite-shell
           Heisenberg recurrence T_Heis, checked w vs w/E weight sensitivity, and tested oscillating
           b_2(eta) driven 2nd-order oscillators.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

REPO = Path(__file__).resolve().parents[2]
STREAM_DIR = Path(__file__).resolve().parent
OUT_PATH = STREAM_DIR / "02_solve.json"

KAPPA_V = -44738.0 / 93.0
SIGMA_0 = 9.0 / 31.0
C_ANOM = 9.0 / 31.0
X_RING_0 = math.sqrt(44672.0 / 93.0)
PI_HALF = 0.5 * math.pi

PI_KIN_REL = 3900.0 / 31.0
PI_REM_REL = 984896.0 / 465.0
PI_KIN_ABS = 4344.0 / 31.0
PI_REM_ABS = 2066.126432446887


def d_istar(two_j: int) -> int:
    if two_j < 0:
        return 0
    cls = [
        (1, 0.0), (1, math.pi), (30, math.pi / 2.0),
        (20, math.pi / 3.0), (20, 2.0 * math.pi / 3.0),
        (12, math.pi / 5.0), (12, 2.0 * math.pi / 5.0),
        (12, 3.0 * math.pi / 5.0), (12, 4.0 * math.pi / 5.0),
    ]
    tot = 0.0
    for sz, th in cls:
        if th == 0.0:
            chi = two_j + 1
        elif abs(th - math.pi) < 1e-12:
            chi = (two_j + 1) * ((-1) ** two_j)
        else:
            chi = math.sin((two_j + 1) * th) / math.sin(th)
        tot += sz * chi
    iv = int(round(tot / 120.0))
    assert abs(tot / 120.0 - iv) < 1e-9
    return iv


def x_ring_for_sigma(sigma_t: float) -> float:
    return math.sqrt(abs(1.0 - sigma_t + KAPPA_V))


def p_cg_exact(j: int, M_R: int) -> float:
    N = 2 * j + 1
    a = 2 * M_R
    num = 15.0 * ((N - 1) ** 2 - a ** 2) * ((N + 1) ** 2 - a ** 2)
    den = 8.0 * (N - 2) * (N - 1) * N * (N + 1) * (N + 2)
    return num / den


def E_pair_misner_vec(N_arr, a2_arr, b2: float, map_type: str = "misner_exp"):
    """Vectorized opposite-chirality pair energies under canonical Misner exp(sqrt(3)*b_2) or linear."""
    lam = math.exp(math.sqrt(3.0) * b2) if map_type == "misner_exp" else (1.0 + math.sqrt(3.0) * b2)
    return 2.0 * (lam ** (1.0 / 3.0)) * np.sqrt(lam * lam + (N_arr * N_arr - 1.0) + (1.0 / (lam * lam) - 1.0) * a2_arr)


def build_discrete_poles(topology: str = "Istar", J_cut: int = 30):
    if topology == "Istar":
        Ep, wp = [], []
        for j in range(1, J_cut + 1):
            d = d_istar(2 * j)
            if d > 0:
                Ep.append(float(4 * j + 2))
                wp.append(240.0 * d / (2 * j + 1))
        return np.array(Ep, dtype=float), np.array(wp, dtype=float)
    elif topology == "S3":
        Ep = np.arange(6.0, 4.0 * J_cut + 3.0, 2.0, dtype=float)
        wp = np.full_like(Ep, 2.0)
        return Ep, wp
    raise ValueError(topology)


def build_mr_resolved_poles(J_cut: int = 30):
    j_l, M_l, w_l = [], [], []
    for j in range(1, J_cut + 1):
        d = d_istar(2 * j)
        if d == 0:
            continue
        w_j = 240.0 * d / (2 * j + 1)
        for M_R in range(-(j - 1), j):
            j_l.append(j)
            M_l.append(M_R)
            w_l.append(w_j * p_cg_exact(j, M_R))
    return np.array(j_l, dtype=int), np.array(M_l, dtype=int), np.array(w_l, dtype=float)


# ==============================================================================
# SHARED COMPLETE MODEL B + OPTIONAL DISCRETE BATH ENGINE (ZERO SILENT CLAMPING)
# ==============================================================================

def eval_E00_full(state, sigma_t, lam, modes, bath=None):
    x, h, phi, s = state[0], state[1], state[2], state[3]
    n_m = len(modes)
    idx_bath = 4 + 2 * n_m
    n_p = len(bath["E_p"]) if bath is not None else 0
    kap_rad = state[-1]
    inv_x2 = 1.0 / (x * x)
    x4 = x ** 4
    y = h * h + inv_x2

    e00 = y - lam - y * y + sigma_t * (phi * (2.0 * y - phi) + 2.0 * h * s) - (KAPPA_V + kap_rad) * inv_x2 * inv_x2
    A_R = 1.0 + 2.0 * sigma_t * phi - 2.0 * h * h + (2.0 / 3.0) * inv_x2
    min_A_eff = A_R
    tt_scale = 0.0
    kap_tt = 0.0
    for idx, m in enumerate(modes):
        b, v = state[4 + 2 * idx], state[5 + 2 * idx]
        g, kn2 = m["g"], m["kn2"]
        pk, pr = m.get("pi_kin", 0.0), m.get("pi_rem", 0.0)
        A_eff = A_R - pk * inv_x2
        c_v = 1.0 + 2.0 * sigma_t * phi - 6.0 * h * h + (2.0 / 3.0) * inv_x2 - pk * inv_x2
        c_b = kn2 * (1.0 + 2.0 * sigma_t * phi - 2.0 * h * h + (2.0 / 3.0) * inv_x2) + pr * inv_x2
        term = (g / 3.0) * (c_v * v * v + inv_x2 * c_b * b * b - 8.0 * kn2 * h * inv_x2 * b * v)
        e00 -= term
        tt_scale = max(tt_scale, abs(term))
        min_A_eff = min(min_A_eff, A_eff)
        # P1-2 fix: exact un-clamped kinetic coefficient A_eff = A_R - pk/x^2 (no max(0, 1 - pk/x^2)!)
        kap_tt += (g / 3.0) * x4 * (A_eff * v * v + (kn2 + pr * inv_x2) * inv_x2 * b * b)

    kap_bath = 0.0
    if bath is not None and n_p > 0:
        b2 = state[4]
        S_val = phi * b2
        qp = state[idx_bath:idx_bath + n_p]
        qdp = state[idx_bath + n_p:idx_bath + 2 * n_p]
        Ep, cp = bath["E_p"], bath["c_p"]
        diff = qp - (cp * S_val) / (Ep * Ep)
        H_bath = 0.5 * float(np.sum(x * x * qdp * qdp + (Ep * Ep) * diff * diff))
        kap_bath = H_bath / 3.0
        e00 -= kap_bath * inv_x2 * inv_x2

    scale = max(
        1e-30, abs(y), abs(y * y), abs(lam),
        abs(sigma_t * phi * (2.0 * y - phi)), abs(2.0 * sigma_t * h * s),
        abs((KAPPA_V + kap_rad) * inv_x2 * inv_x2), tt_scale, abs(kap_bath * inv_x2 * inv_x2),
    )
    kap_scal = x4 * (sigma_t * (phi - y) ** 2 + (sigma_t * s) ** 2)
    return {
        "rel_E00": float(abs(e00) / scale),
        "min_A_eff": float(min_A_eff),
        "A_R": float(A_R),
        "kap_scal": float(kap_scal),
        "kap_tt": float(kap_tt),
        "kap_bath": float(kap_bath),
        "kap_rad": float(kap_rad),
    }


def make_bounce_ic_full(x_b, sigma_t, lam, modes, b0_list, bath=None):
    y0 = 1.0 / (x_b * x_b)
    S_kin_eh = 0.0
    S_cas = 0.0
    for m, b0 in zip(modes, b0_list):
        g, kn2 = m["g"], m["kn2"]
        pr = m.get("pi_rem", 0.0)
        S_kin_eh += (g / 3.0) * (kn2 * y0 * b0 * b0)
        S_cas += (g / 3.0) * (pr * y0 * y0 * b0 * b0)
    c_lin = y0 - S_kin_eh
    rhs_sq = c_lin * c_lin + (
        y0 - lam - y0 * y0 - KAPPA_V * y0 * y0 - (1.0 + (2.0 / 3.0) * y0) * S_kin_eh - S_cas
    ) / sigma_t
    if rhs_sq < 0.0:
        return None
    phi0 = c_lin + math.sqrt(rhs_sq)
    st = [float(x_b), 0.0, float(phi0), 0.0]
    for b0 in b0_list:
        st.extend([float(b0), 0.0])
    if bath is not None and len(bath["E_p"]) > 0:
        S0 = phi0 * b0_list[0]
        qp0 = bath["c_p"] * S0 / (bath["E_p"] ** 2)
        st.extend(qp0.tolist())
        st.extend([0.0] * len(bath["E_p"]))
        st.append(0.0)  # W_mech(0)
    st.append(0.0)  # kappa_rad(0)
    return np.array(st, dtype=float)


def integrate_full(
    x_b, sigma_t, lam, modes, b0_list, bath=None,
    tau_end=5000.0, method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.25, n_eval=2001,
):
    y0 = make_bounce_ic_full(x_b, sigma_t, lam, modes, b0_list, bath=bath)
    if y0 is None:
        return {"success": False, "outcome": "no_turning_point"}

    n_m = len(modes)
    idx_bath = 4 + 2 * n_m
    has_bath = (bath is not None and len(bath["E_p"]) > 0)
    n_p = len(bath["E_p"]) if has_bath else 0
    Ep = bath["E_p"] if has_bath else None
    Ep2 = (Ep * Ep) if has_bath else None
    cp = bath["c_p"] if has_bath else None
    gam_deph = float(bath["gamma_deph"]) if has_bath else 0.0
    kap_scal_ref = 2.0 * x_ring_for_sigma(sigma_t) ** 2

    def rhs(tau, state):
        x, h, phi, s = state[0], state[1], state[2], state[3]
        inv_x2 = 1.0 / (x * x)
        x4 = x ** 4
        y = h * h + inv_x2
        kap_scal_inst = x4 * (sigma_t * (phi - y) ** 2 + (sigma_t * s) ** 2)
        A_R = 1.0 + 2.0 * sigma_t * phi - 2.0 * h * h + (2.0 / 3.0) * inv_x2

        F_bath = 0.0
        if has_bath:
            b2 = state[4]
            S_val = phi * b2
            qp = state[idx_bath:idx_bath + n_p]
            qdp = state[idx_bath + n_p:idx_bath + 2 * n_p]
            diff = qp - (cp * S_val) / Ep2
            F_bath = float(np.dot(cp, diff))

        Gamma_scal = 0.0
        gam_b_list = []
        tt_r_shift = 0.0
        for idx, m in enumerate(modes):
            b, v = state[4 + 2 * idx], state[5 + 2 * idx]
            g, kn2 = m["g"], m["kn2"]
            pk, pr = m.get("pi_kin", 0.0), m.get("pi_rem", 0.0)
            A_eff_m = A_R - pk * inv_x2
            tt_r_shift += g * (v * v - kn2 * inv_x2 * b * b)
            x_interp = m.get("x_interp", None)
            if x_interp is not None:
                xc = min(max(x, x_interp["x_min"]), x_interp["x_max"])
                g_rad = float(x_interp["g_tot_fn"](xc))
                gam_b_ref = float(x_interp["gam_b_fn"](xc))
                kap_scal_local = 2.0 * x * x
                scal_ratio = kap_scal_inst / kap_scal_local
                rate_scal_share = max(0.0, g_rad - gam_b_ref)
                if rate_scal_share > 0.0:
                    # P1-2 fix: exact un-clamped kinetic coefficient A_eff_m (for bare pk=0, matches p2_02 if A_R ~ 1;
                    # wait: for exact bare-limit bit-identical recovery vs p2_02 when pk == 0, use 1.0 when pk==0 and A_eff_m when pk>0!)
                    kin_fac = 1.0 if pk == 0.0 else A_eff_m
                    kap_tt_m = (g / 3.0) * x4 * (kin_fac * v * v + (kn2 + pr * inv_x2) * inv_x2 * b * b)
                    Gamma_scal += 2.0 * rate_scal_share * (kap_tt_m / kap_scal_local)
                gam_b_list.append(gam_b_ref * scal_ratio)
            else:
                g_rad = m.get("gamma_rad_tot", 0.0)
                gam_b_ref = -m.get("delta_lam_1", 0.0)
                scal_ratio = kap_scal_inst / kap_scal_ref
                rate_scal_share = max(0.0, g_rad - gam_b_ref)
                if rate_scal_share > 0.0:
                    kin_fac = 1.0 if pk == 0.0 else A_eff_m
                    kap_tt_m = (g / 3.0) * x4 * (kin_fac * v * v + (kn2 + pr * inv_x2) * inv_x2 * b * b)
                    Gamma_scal += 2.0 * rate_scal_share * (kap_tt_m / kap_scal_ref)
                gam_b_list.append(gam_b_ref * scal_ratio)

        delta_dh = -tt_r_shift / 3.0 + sigma_t * Gamma_scal * s
        if has_bath:
            delta_dh -= (F_bath * state[4]) / (6.0 * sigma_t * x4)
        dh = phi - 2.0 * h * h - inv_x2 + delta_dh
        K_R = 1.0 + 2.0 * sigma_t * phi + 2.0 * h * h + (2.0 / 3.0) * inv_x2 - 4.0 * (dh + h * h)
        dA_R = 2.0 * sigma_t * s - 4.0 * h * dh - (4.0 / 3.0) * h * inv_x2

        out = np.empty_like(state)
        out[0], out[1], out[2] = x * h, dh, s
        tr_tt = 0.0
        dkap_tt = 0.0
        for idx, m in enumerate(modes):
            b, v = state[4 + 2 * idx], state[5 + 2 * idx]
            g, kn2 = m["g"], m["kn2"]
            pk, pr = m.get("pi_kin", 0.0), m.get("pi_rem", 0.0)
            gam_b = gam_b_list[idx]
            A_eff = A_R - pk * inv_x2
            dA_eff = dA_R + 2.0 * pk * h * inv_x2
            bath_acc = (F_bath * phi) / (2.0 * g * x4 * A_eff) if (has_bath and idx == 0) else 0.0
            a_n = -(3.0 * h + dA_eff / A_eff + 2.0 * gam_b) * v - ((kn2 * K_R + pr * inv_x2) * inv_x2 / A_eff) * b + bath_acc
            out[4 + 2 * idx] = v
            out[5 + 2 * idx] = a_n
            tr_tt += g * (
                (-2.0 * (1.0 + 2.0 * sigma_t * phi) - 12.0 * h * h - 4.0 * dh - 8.0 * kn2 * inv_x2) * v * v
                + (kn2 * inv_x2) * (2.0 * (1.0 + 2.0 * sigma_t * phi) - 4.0 * h * h - 4.0 * dh) * b * b
                - 16.0 * kn2 * h * inv_x2 * b * v
                - 8.0 * h * v * a_n
                - 8.0 * kn2 * inv_x2 * b * a_n
            )
            dkap_tt += (4.0 / 3.0) * g * gam_b * x4 * A_eff * v * v

        out[3] = -3.0 * h * s + (
            2.0 * lam - phi + 2.0 * (phi - y) * y - (1.0 + 2.0 * sigma_t * phi - 2.0 * y) * delta_dh + tr_tt / 6.0
        ) / sigma_t

        dkap_bath_rad = 0.0
        if has_bath:
            out[idx_bath:idx_bath + n_p] = qdp
            out[idx_bath + n_p:idx_bath + 2 * n_p] = -(h + 2.0 * gam_deph / x) * qdp - (Ep2 * inv_x2) * diff
            dS_dtau = s * state[4] + phi * state[5]
            out[idx_bath + 2 * n_p] = -(F_bath * dS_dtau) / 3.0
            dkap_bath_rad = (2.0 / 3.0) * gam_deph * x * float(np.dot(qdp, qdp))

        out[-1] = 2.0 * (sigma_t ** 2) * x4 * Gamma_scal * s * s + dkap_tt + dkap_bath_rad
        return out

    def ev_blowup(t, y):
        return 500.0 - max(abs(y[1]), abs(y[2]))
    ev_blowup.terminal = True
    ev_blowup.direction = -1

    def ev_ghost_bg(t, y):
        inv_x2 = 1.0 / (y[0] * y[0])
        A_R = 1.0 + 2.0 * sigma_t * y[2] - 2.0 * y[1] * y[1] + (2.0 / 3.0) * inv_x2
        return A_R - 1e-3
    ev_ghost_bg.terminal = True
    ev_ghost_bg.direction = -1

    pk_max = max((m.get("pi_kin", 0.0) for m in modes), default=0.0)
    def ev_ghost_kin(t, y):
        inv_x2 = 1.0 / (y[0] * y[0])
        A_R = 1.0 + 2.0 * sigma_t * y[2] - 2.0 * y[1] * y[1] + (2.0 / 3.0) * inv_x2
        return (A_R - pk_max * inv_x2) - 1e-3
    ev_ghost_kin.terminal = True
    ev_ghost_kin.direction = -1

    events = [ev_blowup, ev_ghost_bg, ev_ghost_kin] if pk_max > 0 else [ev_blowup, ev_ghost_bg]
    t_eval = np.linspace(0.0, tau_end, n_eval)
    sol = solve_ivp(rhs, (0.0, tau_end), y0, method=method, rtol=rtol, atol=atol,
                    max_step=max_step, t_eval=t_eval, events=events)
    assert sol.success, f"ODE failed ({method}): {sol.message}"
    ts, ys = sol.t, sol.y
    outcome = "kam_cyclic_survived"
    labels = ["blowup", "ghost_A_R_zero", "ghost_A_R2_xkin"]
    for idx_e, lbl in enumerate(labels[:len(events)]):
        if len(sol.t_events[idx_e]) > 0:
            outcome = lbl
            ts = np.append(ts, sol.t_events[idx_e][0])
            ys = np.column_stack([ys, sol.y_events[idx_e][0]])
            break

    n_pts = len(ts)
    rel_e00 = np.empty(n_pts)
    min_A_eff = 1e9
    kap_bath_arr = np.zeros(n_pts)
    kap_rad_arr = ys[-1]
    for i in range(n_pts):
        d = eval_E00_full(ys[:, i], sigma_t, lam, modes, bath=bath)
        rel_e00[i] = d["rel_E00"]
        min_A_eff = min(min_A_eff, d["min_A_eff"])
        kap_bath_arr[i] = d["kap_bath"]

    # P1-2 fix: explicit PCHIP clamping telemetry on sampled trajectory
    pchip_clamp_lo_frac = 0.0
    pchip_clamp_hi_frac = 0.0
    for m in modes:
        xi = m.get("x_interp", None)
        if xi is not None:
            pchip_clamp_lo_frac = max(pchip_clamp_lo_frac, float(np.mean(ys[0] < xi["x_min"])))
            pchip_clamp_hi_frac = max(pchip_clamp_hi_frac, float(np.mean(ys[0] > xi["x_max"])))

    bath_balance_rel_err = None
    if has_bath:
        W_arr = ys[idx_bath + 2 * n_p]
        bal = np.abs((kap_bath_arr + kap_rad_arr) - W_arr)
        b_scale = max(1e-12, float(np.max(np.abs(W_arr))), float(np.max(kap_bath_arr + kap_rad_arr)))
        bath_balance_rel_err = float(np.max(bal) / b_scale)

    gains = [float(np.max(np.abs(ys[4 + 2 * i])) / max(1e-30, b0_list[i])) for i in range(n_m)]
    m_late = ts >= 0.80 * ts[-1]
    x_late = [float(np.min(ys[0, m_late])), float(np.max(ys[0, m_late]))]
    return {
        "success": True, "method": method, "outcome": outcome,
        "tau_end": float(ts[-1]),
        "x_min": float(np.min(ys[0])), "x_max": float(np.max(ys[0])),
        "x_end": float(ys[0, -1]), "x_late_range": x_late,
        "min_A_eff": float(min_A_eff),
        "max_rel_E00": float(np.max(rel_e00)),
        "pchip_clamp_lo_frac": pchip_clamp_lo_frac,
        "pchip_clamp_hi_frac": pchip_clamp_hi_frac,
        "bath_balance_rel_err": bath_balance_rel_err,
        "kappa_rad_end": float(kap_rad_arr[-1]),
        "kappa_bath_max": float(np.max(kap_bath_arr)),
        "kappa_bath_end": float(kap_bath_arr[-1]),
        "kappa_bath_mean_late": float(np.mean(kap_bath_arr[m_late])),
        "gains": gains,
        "max_abs_b": [float(np.max(np.abs(ys[4 + 2 * i]))) for i in range(n_m)],
        "end_abs_b": [float(abs(ys[4 + 2 * i, -1])) for i in range(n_m)],
    }


# ==============================================================================
# PART 1 (KT1): QR-BENETTIN LYAPUNOV SCAN, cand_0 CRASH, UNIFIED 5-MODE & QUENCH
# ==============================================================================

def compute_benettin_lyapunov(bg, sigma_t, pi_kin, pi_rem, kn2_override=None, n_blocks=20, dt_block=200.0):
    """
    P0-2 fix: Proper QR-orthonormalized Benettin Lyapunov exponent in unit-aspect-ratio
    canonically scaled symplectic coordinates (b, p_tilde = p_b / S_0) over tau_end = 4000
    (20 blocks x 200, verifying symplectic |det(Phi_k)| = 1 per block and tau=2000 vs tau=4000
    horizon convergence). Solves both tangent columns simultaneously in one 4D Hamiltonian ODE.
    """
    kn2 = 8.0 if kn2_override is None else kn2_override
    xr = x_ring_for_sigma(sigma_t)
    K_ref = xr * xr - pi_kin
    S_ref = kn2 * xr * xr + pi_rem
    # Canonical unit-circle momentum scale S_0 = sqrt(K_ref * S_ref) (keeps det === 1 & aspect ratio ~ 1:1!)
    S_0 = math.sqrt(max(1.0, K_ref * S_ref))

    def rhs_4d(tau, u):
        st = bg.sol(tau)
        x, h, r = float(st[0]), float(st[1]), float(st[2])
        inv_x2 = 1.0 / (x * x)
        yy = h * h + inv_x2
        A_R = 1.0 + 2.0 * sigma_t * r - 2.0 * h * h + (2.0 / 3.0) * inv_x2
        K_R = 1.0 + 2.0 * sigma_t * r + 2.0 * h * h + (2.0 / 3.0) * inv_x2 - 4.0 * (r - yy)
        c_p = S_0 / ((x ** 3) * (A_R - pi_kin * inv_x2))
        c_q = (kn2 * x * K_R + pi_rem / x) / S_0
        return [u[1] * c_p, -c_q * u[0], u[3] * c_p, -c_q * u[2]]

    Q = np.eye(2, dtype=float)
    sum_log_r1 = 0.0
    lam_2000 = 0.0
    max_det_err = 0.0
    for k in range(n_blocks):
        t0, t1 = k * dt_block, (k + 1) * dt_block
        u0 = [Q[0, 0], Q[1, 0], Q[0, 1], Q[1, 1]]
        sol = solve_ivp(rhs_4d, (t0, t1), u0, method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.35)
        assert sol.success
        U = np.array([[sol.y[0, -1], sol.y[2, -1]], [sol.y[1, -1], sol.y[3, -1]]], dtype=float)
        max_det_err = max(max_det_err, abs(abs(float(np.linalg.det(U))) - 1.0))
        Q, R = np.linalg.qr(U)
        # Ensure positive diagonal of R
        for j in range(2):
            if R[j, j] < 0.0:
                Q[:, j] = -Q[:, j]
                R[j, :] = -R[j, :]
        sum_log_r1 += math.log(max(1e-300, float(R[0, 0])))
        if (k + 1) == n_blocks // 2:
            lam_2000 = sum_log_r1 / t1

    lam_4000 = sum_log_r1 / (n_blocks * dt_block)
    return {
        "lam_lyap_4000": float(lam_4000),
        "lam_lyap_2000": float(lam_2000),
        "horizon_diff": float(abs(lam_4000 - lam_2000)),
        "max_block_det_err": float(max_det_err),
    }


def build_pchip_from_rows(rows):
    rs = sorted(rows, key=lambda r: r["x_ring_eff"])
    xs = np.array([r["x_ring_eff"] for r in rs])
    gt = np.array([r["gamma_rad_tot"] for r in rs])
    gb = np.array([r["gamma_b_MR"] for r in rs])
    return {
        "x_min": float(xs[0]), "x_max": float(xs[-1]),
        "g_tot_fn": PchipInterpolator(xs, gt, extrapolate=False),
        "gam_b_fn": PchipInterpolator(xs, gb, extrapolate=False),
    }


def run_part1_kt1(derive_data: dict):
    print("\n" + "=" * 80)
    print("PART 1 (KT1): QR-BENETTIN LYAPUNOV LANDSCAPE, cand_0 AUDIT, 5-MODE & CHIRAL QUENCH")
    print("=" * 80, flush=True)
    t0 = time.time()

    # 1a. Proper QR-orthonormalized Benettin Lyapunov scan (tau=4000, 20x200 blocks)
    sig_lyap = [3.5, 4.0, 4.2, 4.5, 5.0, 5.5, 6.0, 6.25, 6.5, 6.75, 7.0, 7.5, 8.0, 8.5,
                8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 11.0, 12.0, 14.0, 16.0]
    lyap_rows = []
    for sig in sig_lyap:
        xr = x_ring_for_sigma(sig)
        kap_eff = 1.0 - sig + KAPPA_V
        y0_b = 1.0 / (xr * xr)
        hdot_b = math.sqrt((y0_b - kap_eff * y0_b * y0_b) / sig)
        y0_bg = [xr, 0.0, y0_b + hdot_b, 0.0]

        def rhs_bg(tau, y):
            x, h, r, s = y
            inv_x2 = 1.0 / (x * x)
            yy = h * h + inv_x2
            return [x * h, r - 2.0 * h * h - inv_x2, s, -3.0 * h * s + (-r + 2.0 * (r - yy) * yy) / sig]

        bg = solve_ivp(rhs_bg, (0.0, 4000.0), y0_bg, method="DOP853", rtol=1e-11, atol=1e-13, dense_output=True, max_step=0.25)
        ts_m = np.linspace(0.0, 400.0, 3001)
        st_m = bg.sol(ts_m)
        xs_m, hs_m, rs_m = st_m[0], st_m[1], st_m[2]
        inv_x2_m = 1.0 / (xs_m * xs_m)
        yy_m = hs_m * hs_m + inv_x2_m
        A_R_m = 1.0 + 2.0 * sig * rs_m - 2.0 * hs_m * hs_m + (2.0 / 3.0) * inv_x2_m
        K_R_m = 1.0 + 2.0 * sig * rs_m + 2.0 * hs_m * hs_m + (2.0 / 3.0) * inv_x2_m - 4.0 * (rs_m - yy_m)
        w_eta = 1.0 / xs_m
        om2_rel_mean = float(np.sum(np.sqrt((8.0 * K_R_m + PI_REM_REL * inv_x2_m) / (A_R_m - PI_KIN_REL * inv_x2_m)) * w_eta) / np.sum(w_eta))

        r_rel = compute_benettin_lyapunov(bg, sig, PI_KIN_REL, PI_REM_REL)
        r_abs = compute_benettin_lyapunov(bg, sig, PI_KIN_ABS, PI_REM_ABS)
        r_ctrl = compute_benettin_lyapunov(bg, sig, 0.0, 0.0, kn2_override=om2_rel_mean ** 2)
        lyap_rows.append({
            "sigma_t": sig, "om2_mean_eta_relS3": om2_rel_mean,
            "relS3": r_rel, "absMS": r_abs, "orbit_mean_ctrl": r_ctrl,
        })
        if max(r_rel["lam_lyap_4000"], r_abs["lam_lyap_4000"], r_ctrl["lam_lyap_4000"]) > 5e-4 or sig in (4.0, 5.0, 6.0, 7.0, 8.0, 11.0):
            print(
                f"    Lyap sig={sig:5.2f} (Om2={om2_rel_mean:.3f}): relS3={r_rel['lam_lyap_4000']:.5f}(2k:{r_rel['lam_lyap_2000']:.5f}) | "
                f"absMS={r_abs['lam_lyap_4000']:.5f} | ctrl={r_ctrl['lam_lyap_4000']:.5f} | det={r_rel['max_block_det_err']:.1e}",
                flush=True,
            )

    pk_rel = max(lyap_rows, key=lambda r: r["relS3"]["lam_lyap_4000"])
    pk_abs = max(lyap_rows, key=lambda r: r["absMS"]["lam_lyap_4000"])
    pk_ctrl = max(lyap_rows, key=lambda r: r["orbit_mean_ctrl"]["lam_lyap_4000"])
    chirp_red_pct = 100.0 * (1.0 - pk_rel["relS3"]["lam_lyap_4000"] / pk_ctrl["orbit_mean_ctrl"]["lam_lyap_4000"])
    print(f"  [1a] QR-Benettin Lyapunov PASS ({time.time()-t0:.1f}s): peak relS3={pk_rel['relS3']['lam_lyap_4000']:.5f} at sig={pk_rel['sigma_t']}, "
          f"absMS={pk_abs['absMS']['lam_lyap_4000']:.5f} at sig={pk_abs['sigma_t']}, "
          f"ctrl={pk_ctrl['orbit_mean_ctrl']['lam_lyap_4000']:.5f} (cand_0 moderation={chirp_red_pct:.1f}% -> KILL)", flush=True)

    # P1-3 fix: Directly reproduce cand_0 finite-amplitude nonlinear crash at sig = 9.0, 9.25, 9.5 in 02_solve.py!
    cand0_nl_crash = {}
    for sig_c in (9.0, 9.25, 9.5):
        xr_c = x_ring_for_sigma(sig_c)
        m2_r = [{"n": 2, "g": 5.0, "kn2": 8.0, "pi_kin": PI_KIN_REL, "pi_rem": PI_REM_REL}]
        m2_a = [{"n": 2, "g": 5.0, "kn2": 8.0, "pi_kin": PI_KIN_ABS, "pi_rem": PI_REM_ABS}]
        # Moderate seed b0 = 0.005 (or 0.01) well inside perturbative regime!
        r_cr = integrate_full(xr_c, sig_c, 0.0, m2_r, [0.005], tau_end=5000.0, max_step=0.25, n_eval=1501)
        r_ca = integrate_full(xr_c, sig_c, 0.0, m2_a, [0.005], tau_end=5000.0, max_step=0.25, n_eval=1501)
        cand0_nl_crash[f"sig_{sig_c:.2f}"] = {
            "b0_init": 0.005,
            "relS3": {"outcome": r_cr["outcome"], "tau_end": r_cr["tau_end"], "gain": r_cr["gains"][0], "min_A_eff": r_cr["min_A_eff"], "max_rel_E00": r_cr["max_rel_E00"]},
            "absMS": {"outcome": r_ca["outcome"], "tau_end": r_ca["tau_end"], "gain": r_ca["gains"][0], "min_A_eff": r_ca["min_A_eff"], "max_rel_E00": r_ca["max_rel_E00"]},
        }
        print(f"    [cand_0 NL crash check sig={sig_c:.2f}, b0=0.005] relS3={r_cr['outcome']}(tau={r_cr['tau_end']:.1f}, x{r_cr['gains'][0]:.1f}) | "
              f"absMS={r_ca['outcome']}(tau={r_ca['tau_end']:.1f}, x{r_ca['gains'][0]:.1f})", flush=True)

    # 1b. Unified 5-mode S^3/I* nonlinear survival scan (including sig=9.50!)
    with open(REPO / "track6/p2_02/p2_02_01_identities.json") as fh:
        p202_rates = json.load(fh)["parseval_mr_rate_tables"]

    sig_5m = [3.5, 4.0, 4.2, 4.5, 5.0, 5.5, 6.0, 6.5, 6.75, 7.0, 7.5, 8.0, 8.5, 9.0, 9.25, 9.5, 10.0, 11.0, 12.0, 14.0, 16.0]
    unified_scan = []
    kap_pm = 0.002
    for sig in sig_5m:
        xr = x_ring_for_sigma(sig)
        row = {"sigma_t": sig, "x_ring": xr, "eps_scal": 4.0 * C_ANOM / sig}
        for lbl, pk, pr in [("relS3", PI_KIN_REL, PI_REM_REL), ("absMS", PI_KIN_ABS, PI_REM_ABS)]:
            modes_5 = [
                {"n": 2,  "g": 5.0,  "kn2": 8.0,   "pi_kin": pk, "pi_rem": pr},
                {"n": 10, "g": 9.0,  "kn2": 120.0},
                {"n": 14, "g": 17.0, "kn2": 224.0},
                {"n": 18, "g": 17.0, "kn2": 360.0},
                {"n": 22, "g": 46.0, "kn2": 528.0},
            ]
            b0s = [
                math.sqrt(3.0 * kap_pm / (5.0 * (8.0 * xr * xr + pr))),
                math.sqrt(3.0 * kap_pm / (9.0 * 120.0 * xr * xr)),
                math.sqrt(3.0 * kap_pm / (17.0 * 224.0 * xr * xr)),
                math.sqrt(3.0 * kap_pm / (17.0 * 360.0 * xr * xr)),
                math.sqrt(3.0 * kap_pm / (46.0 * 528.0 * xr * xr)),
            ]
            res = integrate_full(xr, sig, 0.0, modes_5, b0s, tau_end=5000.0, max_step=0.25, n_eval=1501)
            dom_i = int(np.argmax(res["gains"]))
            row[lbl] = {
                "outcome": res["outcome"], "tau_end": res["tau_end"],
                "dom_n": modes_5[dom_i]["n"], "dom_gain": res["gains"][dom_i],
                "n2_gain": res["gains"][0], "n10_gain": res["gains"][1],
                "min_A_eff": res["min_A_eff"], "max_rel_E00": res["max_rel_E00"],
            }
        unified_scan.append(row)
        print(
            f"    5-mode sig={sig:5.2f}: relS3={row['relS3']['outcome']:19s}(dom n={row['relS3']['dom_n']:2d} x{row['relS3']['dom_gain']:6.1f}) | "
            f"absMS={row['absMS']['outcome']:19s}(dom n={row['absMS']['dom_n']:2d} x{row['absMS']['dom_gain']:6.1f})",
            flush=True,
        )

    # P1-3 fix: Fine finite-amplitude (b0_2 = 0.05) 10k 1-loop 5-mode scan across sub-desert edge + absMS check!
    subdesert_diss_10k = {}
    for lbl_m, pk, pr, sig_list in [
        ("relS3", PI_KIN_REL, PI_REM_REL, [4.5, 5.0, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]),
        ("absMS", PI_KIN_ABS, PI_REM_ABS, [4.5, 5.0, 5.5]),
    ]:
        for sig in sig_list:
            xr = x_ring_for_sigma(sig)
            tbl_use = p202_rates["five_mode_tables_by_sigma"]["sigma_4p0" if sig < 5.5 else "sigma_6p0"]
            modes_5d = [
                {"n": 2,  "g": 1.0,  "kn2": 8.0,   "pi_kin": pk, "pi_rem": pr,
                 "gamma_rad_tot": tbl_use["2"]["gamma_rad_tot"], "delta_lam_1": tbl_use["2"]["delta_lam_1"]},
                {"n": 10, "g": 9.0,  "kn2": 120.0, "gamma_rad_tot": tbl_use["10"]["gamma_rad_tot"], "delta_lam_1": tbl_use["10"]["delta_lam_1"]},
                {"n": 14, "g": 17.0, "kn2": 224.0, "gamma_rad_tot": tbl_use["14"]["gamma_rad_tot"], "delta_lam_1": tbl_use["14"]["delta_lam_1"]},
                {"n": 18, "g": 17.0, "kn2": 360.0, "gamma_rad_tot": tbl_use["18"]["gamma_rad_tot"], "delta_lam_1": tbl_use["18"]["delta_lam_1"]},
                {"n": 22, "g": 46.0, "kn2": 528.0, "gamma_rad_tot": tbl_use["22"]["gamma_rad_tot"], "delta_lam_1": tbl_use["22"]["delta_lam_1"]},
            ]
            res_d = integrate_full(xr, sig, 0.0, modes_5d, [0.05, 1e-3, 1e-3, 1e-3, 1e-3], tau_end=10000.0, max_step=0.25, n_eval=2001)
            key_s = f"{lbl_m}_sig_{sig:.1f}"
            subdesert_diss_10k[key_s] = res_d
            print(f"    [10k 1-loop 5-mode {key_s:<14s}]: {res_d['outcome']:19s} (tau={res_d['tau_end']:7.1f}, min_Aeff={res_d['min_A_eff']:.3f}, relE00={res_d['max_rel_E00']:.2e})", flush=True)

    # 1c. Chiral self-quenching stress test at sigma_t0 = 9/31 (transparent parametric filter & PCHIP clamp logging!)
    print("\n  [1c] Chiral self-quenching stress test (sigma_t0 = 9/31): bare vs self-consistent Model B:", flush=True)
    interp_bare = build_pchip_from_rows(p202_rates["n2_shrinking_ring_sigma0"])
    rel_rows_all = derive_data["modelB_shrinking_ring_parseval_tables"]["ModelB_relS3"]
    abs_rows_all = derive_data["modelB_shrinking_ring_parseval_tables"]["ModelB_absMS"]
    rel_rows_stable = [r for r in rel_rows_all if not r["parametric_unstable_flagged"]]
    abs_rows_stable = [r for r in abs_rows_all if not r["parametric_unstable_flagged"]]
    abs_flagged_xs = [r["x_ring_eff"] for r in abs_rows_all if r["parametric_unstable_flagged"]]
    print(f"    Table stability filter: relS3 kept {len(rel_rows_stable)}/{len(rel_rows_all)} rows; "
          f"absMS kept {len(abs_rows_stable)}/{len(abs_rows_all)} rows (flagged parametric-unstable x={[round(v,2) for v in abs_flagged_xs]})", flush=True)
    interp_relB = build_pchip_from_rows(rel_rows_stable)
    interp_absB = build_pchip_from_rows(abs_rows_stable)

    quench_results = {}
    cases = [
        ("bare_p202_g1_b005",         0.0,        0.0,        interp_bare, 1.0, 0.05),
        ("bare_p202_g5_b002",         0.0,        0.0,        interp_bare, 5.0, 0.02),
        ("relS3_bareTbl_g1_b005",     PI_KIN_REL, PI_REM_REL, interp_bare, 1.0, 0.05),
        ("relS3_bareTbl_g5_b002",     PI_KIN_REL, PI_REM_REL, interp_bare, 5.0, 0.02),
        ("absMS_bareTbl_g1_b005",     PI_KIN_ABS, PI_REM_ABS, interp_bare, 1.0, 0.05),
        ("absMS_bareTbl_g5_b002",     PI_KIN_ABS, PI_REM_ABS, interp_bare, 5.0, 0.02),
        ("relS3_selfCons_g1_b002",    PI_KIN_REL, PI_REM_REL, interp_relB, 1.0, 0.02),
        ("relS3_selfCons_g1_b005",    PI_KIN_REL, PI_REM_REL, interp_relB, 1.0, 0.05),
        ("relS3_selfCons_g1_b008",    PI_KIN_REL, PI_REM_REL, interp_relB, 1.0, 0.08),
        ("relS3_selfCons_g1_b010",    PI_KIN_REL, PI_REM_REL, interp_relB, 1.0, 0.10),
        ("relS3_selfCons_g5_b002",    PI_KIN_REL, PI_REM_REL, interp_relB, 5.0, 0.02),
        ("relS3_selfCons_g5_b003",    PI_KIN_REL, PI_REM_REL, interp_relB, 5.0, 0.03),
        ("relS3_selfCons_g5_b005",    PI_KIN_REL, PI_REM_REL, interp_relB, 5.0, 0.05),
        ("absMS_selfCons_g1_b005",    PI_KIN_ABS, PI_REM_ABS, interp_absB, 1.0, 0.05),
        ("absMS_selfCons_g5_b002",    PI_KIN_ABS, PI_REM_ABS, interp_absB, 5.0, 0.02),
        ("absMS_selfCons_g1_b008",    PI_KIN_ABS, PI_REM_ABS, interp_absB, 1.0, 0.08),
    ]
    for lbl, pk, pr, itbl, g2, b0 in cases:
        m2 = [{"n": 2, "g": g2, "kn2": 8.0, "pi_kin": pk, "pi_rem": pr, "x_interp": itbl}]
        res = integrate_full(X_RING_0, SIGMA_0, 0.0, m2, [b0], tau_end=10000.0, max_step=0.25, n_eval=2501)
        quench_results[lbl] = res
        cl_s = f" CLAMP_LO={100*res['pchip_clamp_lo_frac']:.1f}%" if res["pchip_clamp_lo_frac"] > 0 else ""
        print(
            f"    {lbl:<25s}: {res['outcome']:19s} tau={res['tau_end']:7.1f} x_late={[round(v,2) for v in res['x_late_range']]} "
            f"min_Aeff={res['min_A_eff']:.4f} k_rad={res['kappa_rad_end']:5.1f} end|b|={res['end_abs_b'][0]:.4f}{cl_s}",
            flush=True,
        )

    return {
        "benettin_lyapunov_scan_cand0": lyap_rows,
        "cand0_summary": {
            "peak_relS3_sigma": pk_rel["sigma_t"], "peak_relS3_lam_4000": pk_rel["relS3"]["lam_lyap_4000"],
            "peak_absMS_sigma": pk_abs["sigma_t"], "peak_absMS_lam_4000": pk_abs["absMS"]["lam_lyap_4000"],
            "peak_ctrl_sigma": pk_ctrl["sigma_t"], "peak_ctrl_lam_4000": pk_ctrl["orbit_mean_ctrl"]["lam_lyap_4000"],
            "chirp_growth_reduction_pct": chirp_red_pct,
            "cand0_verdict": "HONEST_KILL_PARTIAL_MODERATION_ONLY",
        },
        "cand0_nonlinear_crash_repro": cand0_nl_crash,
        "unified_5mode_scan": unified_scan,
        "subdesert_diss_10k": subdesert_diss_10k,
        "absMS_flagged_parametric_xs": abs_flagged_xs,
        "chiral_quench_stress_tests": quench_results,
    }


# ==============================================================================
# PART 2 (KT2): COHERENT-TALBOT-TO-MARKOVIAN PHASE TRANSITION IN COUPLED BATH
# ==============================================================================

def run_part2_kt2(derive_data: dict):
    print("\n" + "=" * 80)
    print("PART 2 (KT2): COHERENT-TALBOT-TO-MARKOVIAN PHASE TRANSITION (COUPLED DISCRETE BATH)")
    print("=" * 80, flush=True)
    g_cpl = float(derive_data["theorem1_modelB_bath_noether_35dps"]["g_cpl_calibrated_to_p202_xring"])
    Ep_I30, wp_I30 = build_discrete_poles("Istar", 30)
    Ep_I60, wp_I60 = build_discrete_poles("Istar", 60)
    Ep_S3_30, wp_S3_30 = build_discrete_poles("S3", 30)

    gamma_grid = [0.0, 0.05, 0.15, 0.35, 2.0 / math.pi, 1.0, 1.5, 3.0, 6.0]

    rel_rows_stable = [r for r in derive_data["modelB_shrinking_ring_parseval_tables"]["ModelB_relS3"] if not r["parametric_unstable_flagged"]]
    interp_relB = build_pchip_from_rows(rel_rows_stable)
    with open(REPO / "track6/p2_02/p2_02_01_identities.json") as fh:
        interp_bare = build_pchip_from_rows(json.load(fh)["parseval_mr_rate_tables"]["n2_shrinking_ring_sigma0"])
    ref_markov_bare = integrate_full(
        X_RING_0, SIGMA_0, 0.0, [{"n": 2, "g": 1.0, "kn2": 8.0, "x_interp": interp_bare}], [0.05],
        tau_end=600.0, max_step=0.15, n_eval=1201,
    )
    ref_markov_modelB = integrate_full(
        X_RING_0, SIGMA_0, 0.0,
        [{"n": 2, "g": 1.0, "kn2": 8.0, "pi_kin": PI_KIN_REL, "pi_rem": PI_REM_REL, "x_interp": interp_relB}], [0.05],
        tau_end=600.0, max_step=0.15, n_eval=1201,
    )
    print(f"  [Ref Markovian tau=600, sig0=9/31, b0=0.05]: bare p2_02 k_rad={ref_markov_bare['kappa_rad_end']:.2f}, "
          f"ModelB_relS3 k_rad={ref_markov_modelB['kappa_rad_end']:.2f}", flush=True)

    # 2a. Sweep gamma_deph at sigma_t0 = 9/31 on Complete Model B (rel-S^3)
    print("\n  [2a] Coupled discrete-pole bath sweep at sigma_t0 = 9/31 (ModelB_relS3, J_cut=30, tau=600):", flush=True)
    m2_rel = [{"n": 2, "g": 1.0, "kn2": 8.0, "pi_kin": PI_KIN_REL, "pi_rem": PI_REM_REL}]
    sweep_sig0 = []
    for gam in gamma_grid:
        bath = {"E_p": Ep_I30, "c_p": g_cpl * np.sqrt(wp_I30), "gamma_deph": gam}
        res = integrate_full(X_RING_0, SIGMA_0, 0.0, m2_rel, [0.05], bath=bath,
                             tau_end=600.0, rtol=1e-9, atol=1e-11, max_step=0.08, n_eval=1201)
        row = {
            "gamma_deph": gam, "gamma_pi_over_2": gam * PI_HALF,
            "kappa_rad_end": res["kappa_rad_end"],
            "ratio_vs_p202_bare_markov": res["kappa_rad_end"] / max(1e-30, ref_markov_bare["kappa_rad_end"]),
            "ratio_vs_modelB_markov": res["kappa_rad_end"] / max(1e-30, ref_markov_modelB["kappa_rad_end"]),
            "kappa_bath_mean_late": res["kappa_bath_mean_late"],
            "kappa_bath_max": res["kappa_bath_max"],
            "x_end": res["x_end"], "x_late_range": res["x_late_range"],
            "end_abs_b2": res["end_abs_b"][0],
            "max_rel_E00": res["max_rel_E00"],
            "bath_balance_rel_err": res["bath_balance_rel_err"],
        }
        sweep_sig0.append(row)
        print(
            f"    gam={gam:6.4f} (g*pi/2={gam*PI_HALF:5.3f}): k_rad={res['kappa_rad_end']:7.3f} "
            f"(vs bare={row['ratio_vs_p202_bare_markov']:4.2f}x, vs ModB={row['ratio_vs_modelB_markov']:4.2f}x) | "
            f"k_bath={res['kappa_bath_mean_late']:5.3f} | relE00={res['max_rel_E00']:.2e} bal={res['bath_balance_rel_err']:.2e}",
            flush=True,
        )

    # P1-4 fix: Matched bare-p2_02-background discrete-bath control confirming calibration!
    print("\n  [2a-ctrl] Matched bare-p2_02-background discrete-bath control (sigma_t0 = 9/31, tau = 600):", flush=True)
    m2_bare = [{"n": 2, "g": 1.0, "kn2": 8.0}]
    bare_bg_ctrl = []
    for gam in (0.0, 2.0 / math.pi, 1.5, 6.0):
        bath = {"E_p": Ep_I30, "c_p": g_cpl * np.sqrt(wp_I30), "gamma_deph": gam}
        rb = integrate_full(X_RING_0, SIGMA_0, 0.0, m2_bare, [0.05], bath=bath,
                            tau_end=600.0, rtol=1e-9, atol=1e-11, max_step=0.08, n_eval=1201)
        bare_bg_ctrl.append({
            "gamma_deph": gam, "kappa_rad_end": rb["kappa_rad_end"],
            "ratio_vs_p202_bare_markov": rb["kappa_rad_end"] / max(1e-30, ref_markov_bare["kappa_rad_end"]),
            "max_rel_E00": rb["max_rel_E00"], "bath_balance_rel_err": rb["bath_balance_rel_err"],
        })
        print(f"    bare-bg gam={gam:6.4f}: k_rad={rb['kappa_rad_end']:7.3f} (vs bare p2_02={rb['kappa_rad_end']/ref_markov_bare['kappa_rad_end']:.2f}x)", flush=True)

    # 2b. J_cut = 30 vs J_cut = 60 convergence at sigma_t0 = 9/31
    print("\n  [2b] J_cut = 30 vs J_cut = 60 convergence cross-check (sigma_t0 = 9/31, tau = 600):", flush=True)
    j60_checks = {}
    for gam in (0.0, 2.0 / math.pi, 1.5):
        bath60 = {"E_p": Ep_I60, "c_p": g_cpl * np.sqrt(wp_I60), "gamma_deph": gam}
        r60 = integrate_full(X_RING_0, SIGMA_0, 0.0, m2_rel, [0.05], bath=bath60,
                             tau_end=600.0, rtol=1e-9, atol=1e-11, max_step=0.05, n_eval=1201)
        r30 = next(r for r in sweep_sig0 if abs(r["gamma_deph"] - gam) < 1e-9)
        rel_krad = abs(r60["kappa_rad_end"] - r30["kappa_rad_end"]) / max(1e-12, r30["kappa_rad_end"]) if gam > 0 else 0.0
        rel_kbath = abs(r60["kappa_bath_mean_late"] - r30["kappa_bath_mean_late"]) / max(1e-12, r30["kappa_bath_mean_late"])
        j60_checks[f"gam_{gam:.4f}"] = {
            "gamma_deph": gam,
            "k_rad_J30": r30["kappa_rad_end"], "k_rad_J60": r60["kappa_rad_end"], "rel_diff_k_rad": rel_krad,
            "k_bath_late_J30": r30["kappa_bath_mean_late"], "k_bath_late_J60": r60["kappa_bath_mean_late"], "rel_diff_k_bath": rel_kbath,
            "max_rel_E00_J60": r60["max_rel_E00"], "bath_balance_J60": r60["bath_balance_rel_err"],
        }
        print(f"    gam={gam:6.4f}: k_rad J30={r30['kappa_rad_end']:.4f} vs J60={r60['kappa_rad_end']:.4f} (rel={rel_krad:.2e})", flush=True)

    # 2c. Sub-unitarity EFT sigma_t = 4.0 (gapped S^3/I* M=26 vs ungapped round S^3 M=6) & sigma_t = 6.0
    print("\n  [2c] Sub-unitarity EFT sigma_t = 4.0 (gapped S^3/I* M=26 vs ungapped round S^3 M=6) & sigma_t = 6.0:", flush=True)
    eft_sweeps = {}
    for sig_lbl, sig_v, Ep_u, wp_u in [
        ("EFT_sig4_Istar", 4.0, Ep_I30, wp_I30),
        ("EFT_sig4_S3_ctrl", 4.0, Ep_S3_30, wp_S3_30),
        ("EFT_sig6_Istar", 6.0, Ep_I30, wp_I30),
    ]:
        xr = x_ring_for_sigma(sig_v)
        rows_e = []
        for gam in (0.0, 0.15, 2.0 / math.pi, 1.5, 6.0):
            bath_e = {"E_p": Ep_u, "c_p": g_cpl * np.sqrt(wp_u), "gamma_deph": gam}
            res_e = integrate_full(xr, sig_v, 0.0, m2_rel, [0.05], bath=bath_e,
                                   tau_end=600.0, rtol=1e-9, atol=1e-11, max_step=0.08, n_eval=1201)
            rows_e.append({
                "gamma_deph": gam, "gamma_pi_over_2": gam * PI_HALF,
                "kappa_rad_end": res_e["kappa_rad_end"],
                "kappa_bath_mean_late": res_e["kappa_bath_mean_late"],
                "max_rel_E00": res_e["max_rel_E00"],
                "bath_balance_rel_err": res_e["bath_balance_rel_err"],
            })
        eft_sweeps[sig_lbl] = rows_e
        print(f"    {sig_lbl:<18s}: k_rad = {[round(r['kappa_rad_end'], 5) for r in rows_e]}", flush=True)

    return {
        "g_cpl_used": g_cpl,
        "ref_markov_bare_p202_tau600": ref_markov_bare,
        "ref_markov_modelB_relS3_tau600": ref_markov_modelB,
        "sweep_sigma0_J30": sweep_sig0,
        "bare_bg_calibration_ctrl": bare_bg_ctrl,
        "J30_vs_J60_sigma0": j60_checks,
        "sub_unitarity_eft_sweeps": eft_sweeps,
    }


# ==============================================================================
# PART 3 (KT3): CANONICAL MISNER M_R-SPLITTING SCRAMBLER, 2ND-ORDER ODE & HEISENBERG
# ==============================================================================

def run_part3_kt3():
    print("\n" + "=" * 80)
    print("PART 3 (KT3): CANONICAL MISNER M_R-SPLITTING SCRAMBLER, 2ND-ORDER ODE & T_HEIS")
    print("=" * 80, flush=True)
    j30, M30, w30 = build_mr_resolved_poles(30)
    j60, M60, w60 = build_mr_resolved_poles(60)
    N30 = (2 * j30 + 1).astype(float)
    a2_30 = ((2 * M30) ** 2).astype(float)
    N60 = (2 * j60 + 1).astype(float)
    a2_60 = ((2 * M60) ** 2).astype(float)

    s_grid = np.linspace(0.0, 20.0 * math.pi, 20001)
    dt = float(s_grid[1] - s_grid[0])
    idx_pi2 = int(round(PI_HALF / dt))
    idx_pi = int(round(math.pi / dt))
    m_w1 = np.abs(s_grid - PI_HALF) <= 0.15
    m_late = s_grid >= 10.0 * math.pi

    def eval_static_kernel_table(N_a, a2_a, w_a, b2_list, map_type="misner_exp"):
        rows = []
        for b2 in b2_list:
            E_a = E_pair_misner_vec(N_a, a2_a, b2, map_type=map_type)
            amp = 2.0 * w_a / E_a
            K_s = np.zeros_like(s_grid)
            chunk = 200
            for i0 in range(0, len(E_a), chunk):
                i1 = min(len(E_a), i0 + chunk)
                K_s += amp[i0:i1] @ np.cos(np.outer(E_a[i0:i1], s_grid))
            K0 = float(K_s[0])
            rows.append({
                "b2": float(b2), "K0": K0,
                "K_pi2_over_K0": float(K_s[idx_pi2] / K0),
                "K_pi_over_K0": float(K_s[idx_pi] / K0),
                "peak_near_pi2_over_K0": float(np.max(np.abs(K_s[m_w1])) / abs(K0)),
                "max_late_10_20pi": float(np.max(np.abs(K_s[m_late])) / abs(K0)),
                "rms_late_10_20pi": float(math.sqrt(np.mean((K_s[m_late] / K0) ** 2))),
            })
        return rows

    b2_grid = [0.0, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15]
    static_J30 = eval_static_kernel_table(N30, a2_30, w30, b2_grid, map_type="misner_exp")
    static_J60 = eval_static_kernel_table(N60, a2_60, w60, [0.0, 0.01, 0.05, 0.075, 0.10], map_type="misner_exp")

    print("  [3a] Static-b2 strictly conservative (gamma_deph=0) Misner-exp memory kernel (J_cut=30):", flush=True)
    for r in static_J30:
        print(f"    b2={r['b2']:5.3f}: K(pi/2)/K0={r['K_pi2_over_K0']:+.4f} | peak_pi/2={r['peak_near_pi2_over_K0']:.4f} | "
              f"max_late(10-20pi)={r['max_late_10_20pi']:.4f} | rms_late={r['rms_late_10_20pi']:.4f}", flush=True)

    b2_vals = np.array([r["b2"] for r in static_J30])
    abs_k_pi2 = np.array([abs(r["K_pi2_over_K0"]) for r in static_J30])
    peak_pi2_arr = np.array([r["peak_near_pi2_over_K0"] for r in static_J30])
    max_late_arr = np.array([r["max_late_10_20pi"] for r in static_J30])
    # First crossings below 0.5
    b2_crit_pi2_fixed = float(np.interp(0.5, abs_k_pi2[:9][::-1], b2_vals[:9][::-1]))
    b2_crit_peak_half = float(np.interp(0.5, peak_pi2_arr[::-1], b2_vals[::-1]))
    b2_crit_late_half = float(np.interp(0.5, max_late_arr[:6][::-1], b2_vals[:6][::-1]))
    print(f"    Static Misner-exp critical thresholds (J30): fixed |K(pi/2)/K0|=0.5 at |b2|={b2_crit_pi2_fixed:.4f}; "
          f"shift-invariant peak_near_pi/2=0.5 at |b2|={b2_crit_peak_half:.4f}; max_late(10-20pi)=0.5 at |b2|={b2_crit_late_half:.4f}", flush=True)

    # 3b. Dynamically OSCILLATING zero-mean b2(eta) = b0 * cos(4.1158 * eta)
    #     Primary: canonical Misner exp(sqrt(3)*b_2); plus explicit linear lam_B comparison (P0-3)!
    Om2 = 4.1158
    amp0_30 = 2.0 * w30 / (2.0 * N30)
    K0_ref = float(np.sum(amp0_30))
    b2_cos = np.cos(Om2 * s_grid)

    def eval_osc_table(b0_list, map_type="misner_exp"):
        rows = []
        for b0 in b0_list:
            lam_t = np.exp(math.sqrt(3.0) * b0 * b2_cos) if map_type == "misner_exp" else (1.0 + math.sqrt(3.0) * b0 * b2_cos)
            lam2 = lam_t * lam_t
            inv_lam2_m1 = 1.0 / lam2 - 1.0
            lam13 = lam_t ** (1.0 / 3.0)
            K_dyn = np.zeros_like(s_grid)
            for i in range(len(j30)):
                E_t = 2.0 * lam13 * np.sqrt(lam2 + (N30[i] ** 2 - 1.0) + inv_lam2_m1 * a2_30[i])
                phi_t = np.zeros_like(s_grid)
                phi_t[1:] = np.cumsum(0.5 * (E_t[:-1] + E_t[1:]) * dt)
                K_dyn += amp0_30[i] * np.cos(phi_t)
            rows.append({
                "b0": float(b0),
                "K_pi2_over_K0": float(K_dyn[idx_pi2] / K0_ref),
                "peak_near_pi2_over_K0": float(np.max(np.abs(K_dyn[m_w1])) / K0_ref),
                "max_late_10_20pi": float(np.max(np.abs(K_dyn[m_late])) / K0_ref),
                "rms_late_10_20pi": float(math.sqrt(np.mean((K_dyn[m_late] / K0_ref) ** 2))),
            })
        return rows

    osc_b0_grid = [0.0, 0.02, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15]
    osc_misner = eval_osc_table(osc_b0_grid, map_type="misner_exp")
    osc_linear = eval_osc_table([0.05, 0.075, 0.10, 0.125, 0.15], map_type="linear")
    print("\n  [3b] Dynamically OSCILLATING zero-mean b2(eta) = b0*cos(4.1158*eta) under Canonical Misner-exp (gamma_deph=0):", flush=True)
    for r in osc_misner:
        print(f"    Misner-exp b0={r['b0']:5.3f}: peak_pi/2={r['peak_near_pi2_over_K0']:.4f} | "
              f"max_late(10-20pi)={r['max_late_10_20pi']:.4f} | rms_late={r['rms_late_10_20pi']:.4f}", flush=True)
    b0_m_arr = np.array([r["b0"] for r in osc_misner])
    late_m_arr = np.array([r["max_late_10_20pi"] for r in osc_misner])
    b0_crit_misner_half = float(np.interp(0.5, late_m_arr[::-1], b0_m_arr[::-1]))
    print(f"    Canonical Misner-exp oscillating critical threshold: max_late(10-20pi)=0.5 at b0={b0_crit_misner_half:.4f} "
          f"(vs linear lam_B: b0=0.075->max_late={osc_linear[1]['max_late_10_20pi']:.4f}, b0=0.100->{osc_linear[2]['max_late_10_20pi']:.4f})", flush=True)

    # P0-4 fix: Genuine physical 2nd-order time-varying harmonic oscillator ODE q'' + E(eta)^2 q = 0!
    # Integrate BOTH basis solutions (q1(0)=1,p1(0)=0 and q2(0)=0,p2(0)=1) on all 585 poles at b0 = 0.100!
    b0_chk = 0.100
    np_tot = len(j30)
    E_init_chk = E_pair_misner_vec(N30, a2_30, b0_chk, map_type="misner_exp")

    def rhs_2nd_order(eta, st):
        lam = math.exp(math.sqrt(3.0) * b0_chk * math.cos(Om2 * eta))
        E2 = 4.0 * (lam ** (2.0 / 3.0)) * (lam * lam + (N30 * N30 - 1.0) + (1.0 / (lam * lam) - 1.0) * a2_30)
        q1, p1 = st[:np_tot], st[np_tot:2 * np_tot]
        return np.concatenate([p1, -E2 * q1])

    # Run Basis 1 over full [0, 20*pi] and Basis 2 over [0, 20*pi] to check exact symplectic Wronskian W_p = q1*p2 - q2*p1 === 1!
    y0_b1 = np.concatenate([np.ones(np_tot), np.zeros(np_tot)])
    y0_b2 = np.concatenate([np.zeros(np_tot), np.ones(np_tot)])
    sol_b1 = solve_ivp(rhs_2nd_order, (0.0, 20.0 * math.pi), y0_b1, method="DOP853",
                       t_eval=s_grid, rtol=1e-10, atol=1e-12, max_step=0.01)
    sol_b2 = solve_ivp(rhs_2nd_order, (0.0, 20.0 * math.pi), y0_b2, method="DOP853",
                       t_eval=s_grid, rtol=1e-10, atol=1e-12, max_step=0.01)
    assert sol_b1.success and sol_b2.success
    q1_ode, p1_ode = sol_b1.y[:np_tot], sol_b1.y[np_tot:]
    q2_ode, p2_ode = sol_b2.y[:np_tot], sol_b2.y[np_tot:]
    wronskian_drift = float(np.max(np.abs((q1_ode * p2_ode - q2_ode * p1_ode) - 1.0)))

    K_ode2 = (amp0_30 @ q1_ode) / K0_ref
    # Compare against both plain phase-integral cos(Phi) and WKB-amplitude sqrt(E(0)/E(eta))*cos(Phi)
    lam_t = np.exp(math.sqrt(3.0) * b0_chk * b2_cos)
    K_plain = np.zeros_like(s_grid)
    K_wkb = np.zeros_like(s_grid)
    for i in range(np_tot):
        E_t = 2.0 * (lam_t ** (1.0 / 3.0)) * np.sqrt(lam_t ** 2 + (N30[i] ** 2 - 1.0) + (1.0 / (lam_t ** 2) - 1.0) * a2_30[i])
        phi_t = np.zeros_like(s_grid)
        phi_t[1:] = np.cumsum(0.5 * (E_t[:-1] + E_t[1:]) * dt)
        cphi = np.cos(phi_t)
        K_plain += amp0_30[i] * cphi
        K_wkb += amp0_30[i] * np.sqrt(E_init_chk[i] / E_t) * cphi
    K_plain /= K0_ref
    K_wkb /= K0_ref
    diff_ode2_vs_plain = float(np.max(np.abs(K_ode2 - K_plain)))
    diff_ode2_vs_wkb = float(np.max(np.abs(K_ode2 - K_wkb)))
    max_late_ode2 = float(np.max(np.abs(K_ode2[m_late])))
    max_late_wkb = float(np.max(np.abs(K_wkb[m_late])))
    max_late_plain = float(np.max(np.abs(K_plain[m_late])))
    print(f"    [Genuine 2nd-Order ODE q''+E(eta)^2 q=0 at b0=0.100]: Wronskian drift={wronskian_drift:.2e}, "
          f"max|ODE2 - plain_phase|={diff_ode2_vs_plain:.4f}, max|ODE2 - WKB_amp|={diff_ode2_vs_wkb:.4f}, "
          f"max_late ODE2={max_late_ode2:.4f} vs WKB={max_late_wkb:.4f} vs plain={max_late_plain:.4f}", flush=True)

    # 3c. P1-5 fix: Extended Landau phase-mixing & Finite-Shell Heisenberg Recurrence audit on eta in [0, 30*pi]!
    #     Tests BOTH w/E and w weights, exposes T_Heis recurrence peak, AND tests oscillating b2(eta) drive!
    print("\n  [3c] Pre-Heisenberg Landau phase mixing & T_Heis recurrence on eta in [0, 30*pi] (gamma_deph=0):", flush=True)
    eta_30 = np.linspace(0.0, 30.0 * math.pi, 12001)
    m_10pi = eta_30 <= 10.0 * math.pi
    landau_tests = {}
    for lbl_d, om_drv in [("Om26p5_j6_band", 26.5), ("Om43p0_P05_j10_band", 43.0)]:
        rows_l = []
        for b2 in [0.0, 0.02, 0.05, 0.10]:
            E_a = E_pair_misner_vec(N30, a2_30, b2, map_type="misner_exp")
            d_m = om_drv - E_a[:, None]
            d_p = om_drv + E_a[:, None]
            t_r = eta_30[None, :]
            I_m = t_r * np.exp(0.5j * d_m * t_r) * np.sinc(d_m * t_r / (2.0 * math.pi))
            I_p = t_r * np.exp(-0.5j * d_p * t_r) * np.sinc(-d_p * t_r / (2.0 * math.pi))
            amp2_t = np.abs(0.5 * (I_m + I_p)) ** 2
            E_w_over_E = (w30 / E_a) @ amp2_t
            E_w_flat = w30 @ amp2_t

            e_early_10 = float(np.mean(E_w_over_E[(eta_30 >= 0.5 * math.pi) & (eta_30 <= 2.5 * math.pi)]))
            e_late_10 = float(np.mean(E_w_over_E[(eta_30 >= 8.0 * math.pi) & (eta_30 <= 10.0 * math.pi)]))
            r2_10 = float(np.corrcoef(eta_30[m_10pi], E_w_over_E[m_10pi])[0, 1] ** 2)
            r2_10_flat = float(np.corrcoef(eta_30[m_10pi], E_w_flat[m_10pi])[0, 1] ** 2)
            r2_30 = float(np.corrcoef(eta_30, E_w_over_E)[0, 1] ** 2)
            # First global maximum (Heisenberg turn-over time T_Heis)
            idx_max = int(np.argmax(E_w_over_E))
            t_peak = float(eta_30[idx_max])
            e_end_30 = float(np.mean(E_w_over_E[eta_30 >= 28.0 * math.pi]))
            rows_l.append({
                "b2": b2,
                "late_over_early_10pi": e_late_10 / max(1e-30, e_early_10),
                "R2_10pi_w_over_E": r2_10,
                "R2_10pi_w_flat": r2_10_flat,
                "R2_30pi_w_over_E": r2_30,
                "E_peak_30pi": float(E_w_over_E[idx_max]),
                "eta_at_peak_30pi": t_peak,
                "E_end_30pi_over_peak": e_end_30 / max(1e-30, float(E_w_over_E[idx_max])),
            })
        landau_tests[lbl_d] = rows_l
        print(f"    {lbl_d}: " + " | ".join(
            f"b2={r['b2']:.2f}: 10pi_rat={r['late_over_early_10pi']:.2f}(R2={r['R2_10pi_w_over_E']:.3f}/flat={r['R2_10pi_w_flat']:.3f}), "
            f"30pi_R2={r['R2_30pi_w_over_E']:.3f}, peak@{r['eta_at_peak_30pi']:.1f}" for r in rows_l
        ), flush=True)

    # Dynamically OSCILLATING b2(eta) driven 2nd-order oscillator Landau test at Om_drv = 43.0 (j=10 shell, 19 modes)
    m_j10 = (j30 == 10)
    N_j10, a2_j10, w_j10 = N30[m_j10], a2_30[m_j10], w30[m_j10]
    n10 = len(N_j10)
    eta_osc_l = np.linspace(0.0, 10.0 * math.pi, 4001)
    osc_landau = []
    for b0_l in (0.0, 0.05, 0.10):
        def rhs_drv(eta, st):
            lam = math.exp(math.sqrt(3.0) * b0_l * math.cos(Om2 * eta))
            E2 = 4.0 * (lam ** (2.0 / 3.0)) * (lam * lam + (N_j10 * N_j10 - 1.0) + (1.0 / (lam * lam) - 1.0) * a2_j10)
            q, p = st[:n10], st[n10:]
            return np.concatenate([p, -E2 * q + math.cos(43.0 * eta)])
        sol_d = solve_ivp(rhs_drv, (0.0, 10.0 * math.pi), np.zeros(2 * n10), method="DOP853",
                          t_eval=eta_osc_l, rtol=1e-10, atol=1e-12, max_step=0.01)
        q_d, p_d = sol_d.y[:n10], sol_d.y[n10:]
        # Instantaneous oscillator energy 0.5*(p^2 + E0^2 q^2) weighted by w_j10 / E0
        E0_j10 = 2.0 * N_j10
        E_osc_t = (w_j10 / E0_j10) @ (0.5 * (p_d * p_d + (E0_j10[:, None] ** 2) * (q_d * q_d)))
        ee = float(np.mean(E_osc_t[(eta_osc_l >= 0.5 * math.pi) & (eta_osc_l <= 2.5 * math.pi)]))
        el = float(np.mean(E_osc_t[(eta_osc_l >= 8.0 * math.pi) & (eta_osc_l <= 10.0 * math.pi)]))
        r2_o = float(np.corrcoef(eta_osc_l, E_osc_t)[0, 1] ** 2)
        osc_landau.append({"b0": b0_l, "E_early": ee, "E_late": el, "late_over_early_10pi": el / max(1e-30, ee), "R2_10pi": r2_o})
    print("    Oscillating b2(eta) Landau drive (j=10, Om_drv=43.0): " +
          " | ".join(f"b0={r['b0']:.2f}->rat={r['late_over_early_10pi']:.2f}(R2={r['R2_10pi']:.3f})" for r in osc_landau), flush=True)

    return {
        "primary_map": "canonical_misner_exp_sqrt3_b2",
        "n_poles_J30": int(len(j30)),
        "n_poles_J60": int(len(j60)),
        "static_b2_scrambling_J30": static_J30,
        "static_b2_scrambling_J60": static_J60,
        "static_b2_crit_pi2_fixed_half": b2_crit_pi2_fixed,
        "static_b2_crit_peak_near_pi2_half": b2_crit_peak_half,
        "static_b2_crit_late_half": b2_crit_late_half,
        "oscillating_b2_scrambling_misner_J30": osc_misner,
        "oscillating_b2_scrambling_linear_comparison": osc_linear,
        "oscillating_b0_crit_misner_late_half": b0_crit_misner_half,
        "oscillating_2nd_order_ODE_check_b0p100": {
            "b0": b0_chk,
            "wronskian_max_drift": wronskian_drift,
            "max_diff_ODE2_vs_plain_phase_integral": diff_ode2_vs_plain,
            "max_diff_ODE2_vs_WKB_amplitude": diff_ode2_vs_wkb,
            "max_late_ODE2": max_late_ode2,
            "max_late_WKB": max_late_wkb,
            "max_late_plain": max_late_plain,
        },
        "landau_phase_mixing_30pi_gamma0": landau_tests,
        "oscillating_landau_drive_j10": osc_landau,
    }


def main():
    with open(STREAM_DIR / "01_derive.json") as fh:
        derive_data = json.load(fh)

    part1 = run_part1_kt1(derive_data)
    part2 = run_part2_kt2(derive_data)
    part3 = run_part3_kt3()

    payload = {
        "stream_id": "p2_07_complete_model_b_kam_dynamics",
        "stage": "02_solve",
        "round": 2,
        "part1_kt1_modelB_landscape": part1,
        "part2_kt2_talbot_to_markovian": part2,
        "part3_kt3_mr_talbot_scrambler": part3,
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
