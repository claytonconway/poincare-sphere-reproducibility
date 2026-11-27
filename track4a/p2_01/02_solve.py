#!/usr/bin/env python3
"""
Stream p2_01_discrete_photon_pole_sum — Stage 2: Dual Independent Time-Domain Solvers
for the Exact Discrete S^3/I* (and Round S^3) 2-Photon Spectral Pole Sum (CLM-OPEN-01).

Implements TWO completely independent time-domain discretizations of the full unreduced 4th-order
Ostrogradsky + Causal Nonlocal Discrete 2-Photon Pole Sum equation:
  - Solver A (`solve_discrete_aux_ode`):
      Exact undamped harmonic-oscillator auxiliary fields (Y_p, Z_p) for every discrete pole
      j <= J_cut (E_p <= 4*J_cut + 2) PLUS smooth continuum-log UV tail at M_tail = 4*J_cut + 4
      (Gauss-Legendre-in-ln(u) auxiliary ODEs), integrated with Radau/BDF and an EXACT analytical
      Jacobian (verified against central finite differences).
  - Solver B (`solve_discrete_direct_volterra`):
      COMPLETELY INDEPENDENT Direct Time-Domain Product-Integration Volterra solver using exact
      closed-form panel integrals of BOTH -2*Ci(M_tail * s) AND +(2*w_p/E_p)*cos(E_p * s) with
      implicit Crank-Nicolson (trapezoidal) marching on a uniform grid (ZERO auxiliary ODEs!).

Stage 2 Verification Suite (zero silent clamping):
  1. Exact analytical Jacobian verification (max rel error vs central FD).
  2. Exact background Friedmann Hamiltonian constraint residual ||E_00||_rel (un-clamped).
  3. Exact Ostrogradsky + Discrete-Oscillator Total Hamiltonian conservation H_tot(eta) on static
     x_ring (pure discrete truncation, no continuum tail -> exact conservative Hamiltonian!).
  4. Exact RG-scale invariance (mu_phys -> e * mu_phys, beta_0 -> beta_0 + c_anom) for BOTH Solver A & B.
  5. Solver A (Discrete Aux-ODE) vs Solver B (Direct Product-Volterra) vs 40-digit Stage 1 exact roots:
       - Static x_ring at beta_0 = 0.289 (UHP Horowitz pole, gam = +0.38725 H_S),
       - Static x_ring at beta_0 = 0.500 (NEW discrete UHP Horowitz pole! gam = +0.22485 H_S,
         whereas Track IV-A continuum-log had gam = 0!),
       - Static x_ring at beta_0 = 1.000 (sub-threshold real poles, bounded gam = 0.0),
       - Dynamical KAM x_ring at beta_0 = 0.500 and beta_0 = 1.000!
  6. Time-domain discrete-cutoff convergence check (J_cut = 30 vs J_cut = 60 on dynamical x_ring).
"""

import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import sici

from importlib.machinery import SourceFileLoader

STREAM_DIR = Path(__file__).resolve().parent
OUT_PATH = STREAM_DIR / "02_solve.json"

# Exact P05 / T3-B / T4-A constants (M_Pl = H_S = 1, Lambda = 0)
SIGMA_0 = 9.0 / 31.0
C_ANOM = 9.0 / 31.0
KAPPA_V = -44738.0 / 93.0
X_RING = math.sqrt(44672.0 / 93.0)  # 21.916753546579507


# ==============================================================================
# 1. PORTABLE P05 BACKGROUND INTEGRATOR (ZERO HARDCODED EXTERNAL PATHS)
# ==============================================================================

def e00_constraint(x, h, r, s, sigma_t=SIGMA_0, lam=0.0, kappa=KAPPA_V):
    """Exact dimensionless (00) Friedmann constraint e_00(x, h, r, s). Should equal 0."""
    y = h * h + 1.0 / (x * x)
    return y - lam - y * y + sigma_t * (r * (2.0 * y - r) + 2.0 * h * s) - kappa / (x ** 4)


def p05_ode_rhs(tau, state, sigma_t=SIGMA_0, lam=0.0):
    """Exact 4-variable nonsingular trace ODE for (x, h, r, s) in proper time tau."""
    x, h, r, s = state
    inv_x2 = 1.0 / (x * x)
    y = h * h + inv_x2
    dx = x * h
    dh = r - 2.0 * h * h - inv_x2
    dr = s
    ds = -3.0 * h * s + (2.0 * lam - r + 2.0 * (r - y) * y) / sigma_t
    return [dx, dh, dr, ds]


def bounce_initial_state(x_b, sigma_t=SIGMA_0, lam=0.0, kappa=KAPPA_V, s_b=0.0, sign_hdot=+1):
    """Construct exact bounce state (x_b, h=0, r_b, s_b) on e_00 = 0."""
    kappa_eff = 1.0 - sigma_t + kappa
    rhs = (1.0 / (x_b * x_b) - lam - kappa_eff / (x_b ** 4)) / sigma_t
    if rhs < 0.0:
        return None
    hdot_b = sign_hdot * math.sqrt(rhs)
    r_b = 1.0 / (x_b * x_b) + hdot_b
    return np.array([x_b, 0.0, r_b, s_b], dtype=float)


def integrate_kam_background_eta(x_b=X_RING, eta_end=50.0, sigma_t=SIGMA_0, kappa=KAPPA_V, sign_hdot=+1):
    """Integrate P05 background (x, h, r, s, tau) in conformal time eta in [0, eta_end]."""
    y0_bg = bounce_initial_state(x_b, sigma_t, 0.0, kappa, s_b=0.0, sign_hdot=sign_hdot)
    if y0_bg is None:
        raise ValueError(f"No turning point at x_b={x_b}")
    y0 = np.array([y0_bg[0], y0_bg[1], y0_bg[2], y0_bg[3], 0.0], dtype=float)

    def rhs_eta(eta, y):
        x, h, r, s, _ = y
        dx_t, dh_t, dr_t, ds_t = p05_ode_rhs(0.0, [x, h, r, s], sigma_t, 0.0)
        return [x * dx_t, x * dh_t, x * dr_t, x * ds_t, x]

    sol = solve_ivp(
        rhs_eta, (0.0, eta_end), y0, method="DOP853",
        rtol=1e-12, atol=1e-14, dense_output=True, max_step=0.02,
    )
    assert sol.success, f"Background integration failed: {sol.message}"
    etas_chk = np.linspace(0.0, eta_end, 1001)
    ys_chk = sol.sol(etas_chk)
    e00_abs = []
    e00_rel = []
    for i in range(len(etas_chk)):
        xi, hi, ri, si = ys_chk[0, i], ys_chk[1, i], ys_chk[2, i], ys_chk[3, i]
        res = abs(e00_constraint(xi, hi, ri, si, sigma_t, 0.0, kappa))
        yi = hi * hi + 1.0 / (xi * xi)
        scale = max(
            abs(yi),
            abs(yi * yi),
            abs(sigma_t * ri * (2.0 * yi - ri)),
            abs(2.0 * sigma_t * hi * si),
            abs(kappa) / (xi ** 4),
        )
        e00_abs.append(res)
        e00_rel.append(res / scale)
    return sol, {
        "x_b": float(x_b),
        "eta_end": float(eta_end),
        "tau_end": float(ys_chk[4, -1]),
        "x_min": float(np.min(ys_chk[0])),
        "x_max": float(np.max(ys_chk[0])),
        "max_abs_e00": float(max(e00_abs)),
        "rms_abs_e00": float(np.sqrt(np.mean(np.square(e00_abs)))),
        "max_rel_e00": float(max(e00_rel)),
        "rms_rel_e00": float(np.sqrt(np.mean(np.square(e00_rel)))),
    }


def bg_AK_at_state(x, h, r, sigma_t=SIGMA_0):
    """Exact variational Riegert coefficients A_Rieg and K_Rieg at (x, h, r)."""
    inv_x2 = 1.0 / (x * x)
    yy = h * h + inv_x2
    A_R = 1.0 + 2.0 * sigma_t * r - 2.0 * h * h + (2.0 / 3.0) * inv_x2
    K_R = 1.0 + 2.0 * sigma_t * r + 2.0 * h * h + (2.0 / 3.0) * inv_x2 - 4.0 * (r - yy)
    return A_R, K_R


# ==============================================================================
# 2. EXACT DISCRETE 2-PHOTON POLE SPECTRA (S^3/I* AND ROUND S^3)
# ==============================================================================

def d_istar(j):
    """Exact multiplicity d_{I*}(j) of trivial I*-representations inside spin-j of SU(2)_L."""
    two_j = int(round(2 * j))
    if two_j < 0 or two_j % 2 == 1:
        return 0
    classes = [
        (1, 0.0), (1, math.pi), (30, math.pi / 2.0),
        (20, math.pi / 3.0), (20, 2.0 * math.pi / 3.0),
        (12, math.pi / 5.0), (12, 2.0 * math.pi / 5.0),
        (12, 3.0 * math.pi / 5.0), (12, 4.0 * math.pi / 5.0),
    ]
    total = 0.0
    for size, alpha in classes:
        if alpha == 0.0:
            chi = two_j + 1
        elif abs(alpha - math.pi) < 1e-12:
            chi = (two_j + 1) * ((-1) ** two_j)
        else:
            chi = math.sin((two_j + 1) * alpha) / math.sin(alpha)
        total += size * chi
    return int(round(total / 120.0))


def build_discrete_pole_spec(topology="Istar", J_cut=30, include_tail=True, match_exact_K0=True):
    """
    Build exact discrete 2-photon pole array (E_p, w_p = 2*W(j_p), alpha_p = w_p / E_p)
    and UV tail parameters for topology in {'Istar', 'S3', 'continuum_M26', 'continuum_M6'}.

    For topology == 'Istar':
      - Explicit nonzero I*-invariant poles for integer j in [1, J_cut]: E_j = 4*j + 2, w_j = 240*d_{I*}(j)/(2j+1).
      - UV tail threshold M_tail = 4*J_cut + 4.
      - Optional exact O(1/M_tail^2) static offset delta_C_tail = K_exact(0) - K_hybrid(0) so K_hybrid(0) == K_exact(0)
        to machine precision (preserves exact RG invariance and causality!).
    For topology == 'S3':
      - Explicit poles for half-integer j in [1, J_cut]: E_k = 2*k (k = 3 .. 2*J_cut + 1), w_k = 2.0.
      - UV tail threshold M_tail = 4*J_cut + 4.
    """
    if topology == "continuum_M26":
        return {
            "topology": topology, "J_cut": 0, "E_p": np.array([], dtype=float),
            "w_p": np.array([], dtype=float), "alpha_p": np.array([], dtype=float),
            "M_tail": 26.0, "include_tail": True, "delta_C_tail": 0.0,
        }
    if topology == "continuum_M6":
        return {
            "topology": topology, "J_cut": 0, "E_p": np.array([], dtype=float),
            "w_p": np.array([], dtype=float), "alpha_p": np.array([], dtype=float),
            "M_tail": 6.0, "include_tail": True, "delta_C_tail": 0.0,
        }

    if topology == "Istar":
        E_list, w_list = [], []
        for j in range(1, J_cut + 1):
            d = d_istar(j)
            if d > 0:
                E_list.append(4.0 * j + 2.0)
                w_list.append(240.0 * d / (2.0 * j + 1.0))
        E_p = np.array(E_list, dtype=float)
        w_p = np.array(w_list, dtype=float)
        alpha_p = w_p / E_p
        M_tail = 4.0 * J_cut + 4.0
        delta_C = 0.0
        if match_exact_K0:
            # Compute exact mu-independent K_Istar(0; mu=1) via mpmath
            # (at mu = x_ring, K_Istar_exact(0; x_ring) = -0.464871174...)
            import mpmath as mp
            with mp.workdps(30):
                R30 = [120 * d_istar(r) - 2 * (2 * r + 1) for r in range(30)]
                lead0 = mp.log(16) + 2 * mp.psi(0, mp.mpf("31.5"))
                rem0 = sum(R30[r % 30] * 8 * mp.psi(1, mp.mpf(120 + 4 * r + 2) / 120) / (120 ** 2) for r in range(1, 31))
                low0 = sum((mp.mpf(240 * d_istar(j)) / (2 * j + 1)) * 2 / (4 * j + 2) for j in range(1, 31) if d_istar(j) > 0)
                K0_exact_mu1 = float(lead0 - rem0 - low0)
            K0_hyb_mu1 = math.log(M_tail * M_tail) - float(np.sum(2.0 * alpha_p))
            delta_C = K0_exact_mu1 - K0_hyb_mu1
        return {
            "topology": topology, "J_cut": int(J_cut), "E_p": E_p, "w_p": w_p, "alpha_p": alpha_p,
            "M_tail": float(M_tail), "include_tail": bool(include_tail), "delta_C_tail": float(delta_C),
        }

    if topology == "S3":
        # k = 3 .. 2*J_cut + 1 -> E_k = 2*k (from 6 to 4*J_cut + 2), w_k = 2.0
        ks = np.arange(3, 2 * J_cut + 2, dtype=float)
        E_p = 2.0 * ks
        w_p = np.full_like(E_p, 2.0)
        alpha_p = w_p / E_p
        M_tail = 4.0 * J_cut + 4.0
        delta_C = 0.0
        if match_exact_K0:
            # Exact K_S3(0; mu=1) = ln(4) + 2*psi(3) = ln(4) + 3 - 2*gamma_E
            K0_exact_mu1 = math.log(4.0) + 3.0 - 2.0 * 0.5772156649015328606
            K0_hyb_mu1 = math.log(M_tail * M_tail) - float(np.sum(2.0 * alpha_p))
            delta_C = K0_exact_mu1 - K0_hyb_mu1
        return {
            "topology": topology, "J_cut": int(J_cut), "E_p": E_p, "w_p": w_p, "alpha_p": alpha_p,
            "M_tail": float(M_tail), "include_tail": bool(include_tail), "delta_C_tail": float(delta_C),
        }

    raise ValueError(f"Unknown topology: {topology}")


def make_aux_quadrature(M_tail, u_min=1e-6, u_max=1e7, Nu=48):
    """Gauss-Legendre nodes in ln(u) over [u_min, u_max] for smooth UV continuum-log tail."""
    xi, wi = np.polynomial.legendre.leggauss(Nu)
    a_l, b_l = math.log(u_min), math.log(u_max)
    ln_u = 0.5 * (b_l - a_l) * xi + 0.5 * (b_l + a_l)
    u = np.exp(ln_u)
    du = 0.5 * (b_l - a_l) * wi * u
    denom = u * u + M_tail * M_tail
    w_re = du * u / denom
    w_im = du * M_tail / denom
    return {"M": float(M_tail), "u_min": float(u_min), "u_max": float(u_max), "Nu": int(Nu),
            "u": u, "w_re": w_re, "w_im": w_im}


# ==============================================================================
# 3. SOLVER A: DISCRETE-POLE HARMONIC OSCILLATORS + UV-TAIL AUX-ODE (EXACT JAC)
# ==============================================================================

def solve_discrete_aux_ode(
    bg_sol,
    eta_end,
    beta_0,
    pole_spec,
    kappa_mem=1.0,
    n_mode=2,
    running=True,
    static_bg=False,
    x_static=X_RING,
    mu_phys=1.0,
    Nu=48,
    u_min=1e-6,
    u_max=1e7,
    y0_4d=None,
    n_eval=1001,
    method="Radau",
    rtol=1e-9,
    atol=1e-11,
    max_step=0.02,
    blowup_cap=1e6,
    sigma_t=SIGMA_0,
    check_jacobian=False,
    track_hamiltonian=False,
):
    """
    Solve the FULL UNREDUCED 4th-order Ostrogradsky + Exact Discrete 2-Photon Pole Sum system.
    State layout:
      [q1, q2, pi1, pi2,
       YQ(Np), ZQ(Np), YP(Np), ZP(Np),            <-- exact undamped discrete pole oscillators
       UQ(Nu), VQ(Nu), UP(Nu), VP(Nu)]            <-- smooth UV tail (omitted if include_tail=False)
    """
    kn2 = float(n_mode * (n_mode + 2))
    pn = float(n_mode + 1)
    mu_cyl = X_RING * mu_phys
    use_mem = abs(kappa_mem) > 0.0
    c_mem = kappa_mem * C_ANOM

    E_p = pole_spec["E_p"]
    alpha_p = pole_spec["alpha_p"]
    Np = len(E_p)
    inc_tail = bool(pole_spec["include_tail"]) and use_mem
    M_tail = float(pole_spec["M_tail"])
    delta_C = float(pole_spec["delta_C_tail"])

    if inc_tail:
        quad = make_aux_quadrature(M_tail, u_min=u_min, u_max=u_max, Nu=Nu)
        u = quad["u"]
        w_re = quad["w_re"]
        w_im = quad["w_im"]
        uv_shift = 0.5 * c_mem * (math.log((u_max ** 2 + M_tail ** 2) / (mu_cyl ** 2)) + delta_C)
        sum_wre = float(np.sum(w_re))
    elif use_mem:
        Nu = 0
        u = w_re = w_im = np.array([], dtype=float)
        uv_shift = 0.5 * c_mem * (math.log((M_tail ** 2) / (mu_cyl ** 2)) + delta_C)
        sum_wre = 0.0
    else:
        Nu = 0
        Np = 0
        E_p = alpha_p = u = w_re = w_im = np.array([], dtype=float)
        uv_shift = 0.0
        sum_wre = 0.0

    # Index offsets
    iYQ = 4
    iZQ = iYQ + Np
    iYP = iZQ + Np
    iZP = iYP + Np
    iUQ = iZP + Np
    iVQ = iUQ + Nu
    iUP = iVQ + Nu
    iVP = iUP + Nu
    dim = iVP + Nu

    def get_bg(eta):
        if static_bg:
            x = x_static
            A_R, K_R = 1.0, 1.0
        else:
            st = bg_sol.sol(eta)
            x, h, r = float(st[0]), float(st[1]), float(st[2])
            A_R, K_R = bg_AK_at_state(x, h, r, sigma_t=sigma_t)
        beff = beta_0 - (C_ANOM * math.log(x / X_RING) if running else 0.0)
        b_uv = beff + uv_shift
        return x, A_R, K_R, beff, b_uv

    # Initial state (constant pre-history eta < 0 => Z_p(0) = 0, V_u(0) = 0, Y_p(0) = U_u(0) = f(0))
    _, _, _, _, buv0 = get_bg(0.0)
    if y0_4d is None:
        q1_0, q2_0, pi1_0, pi2_0 = 1.0, 0.0, 0.0, 0.0
        Q0_val, P0_val = 0.0, 0.0
    else:
        q1_0, q2_0, pi1_0, pi2_0 = [float(v) for v in y0_4d]
        P0_val = 2.0 * pn * q2_0
        b_init_eff = buv0 - c_mem * (sum_wre + float(np.sum(alpha_p)))
        if abs(b_init_eff) < 1e-10:
            raise ValueError(f"Initial effective beta near zero: {b_init_eff}")
        Q0_val = -pi2_0 / (8.0 * b_init_eff)

    y0 = np.zeros(dim, dtype=float)
    y0[:4] = [q1_0, q2_0, pi1_0, pi2_0]
    if Np > 0:
        y0[iYQ:iZQ] = Q0_val
        y0[iYP:iZP] = P0_val
    if Nu > 0:
        y0[iUQ:iVQ] = Q0_val
        y0[iUP:iVP] = P0_val

    def rhs(eta, y):
        x, A_R, K_R, _, b_uv = get_bg(eta)
        q1, q2, pi1, pi2 = y[0], y[1], y[2], y[3]
        P = 2.0 * pn * q2
        dy = np.empty(dim, dtype=float)

        mem_Q = 0.0
        mem_P = 0.0
        if Np > 0:
            YQ = y[iYQ:iZQ]
            ZQ = y[iZQ:iYP]
            YP = y[iYP:iZP]
            ZP = y[iZP:iUQ]
            mem_Q += float(np.dot(alpha_p, YQ))
            mem_P += float(np.dot(alpha_p, YP))
        if Nu > 0:
            UQ = y[iUQ:iVQ]
            VQ = y[iVQ:iUP]
            UP = y[iUP:iVP]
            VP = y[iVP:dim]
            mem_Q += float(np.dot(w_re, UQ) - np.dot(w_im, VQ))
            mem_P += float(np.dot(w_re, UP) - np.dot(w_im, VP))

        Q = (-0.125 * pi2 + c_mem * mem_Q) / b_uv
        Eff_P = b_uv * P - c_mem * mem_P

        dy[0] = q2
        dy[1] = kn2 * q1 + Q
        dy[2] = -2.0 * kn2 * x * x * K_R * q1 - kn2 * pi2
        dy[3] = -pi1 + 2.0 * x * x * A_R * q2 + 16.0 * pn * Eff_P

        if Np > 0:
            dy[iYQ:iZQ] = E_p * ZQ
            dy[iZQ:iYP] = E_p * (Q - YQ)
            dy[iYP:iZP] = E_p * ZP
            dy[iZP:iUQ] = E_p * (P - YP)
        if Nu > 0:
            dQ = Q - UQ
            dP = P - UP
            dy[iUQ:iVQ] = u * dQ - M_tail * VQ
            dy[iVQ:iUP] = -M_tail * dQ - u * VQ
            dy[iUP:iVP] = u * dP - M_tail * VP
            dy[iVP:dim] = -M_tail * dP - u * VP
        return dy

    def jac(eta, y):
        x, A_R, K_R, _, b_uv = get_bg(eta)
        J = np.zeros((dim, dim), dtype=float)
        c_over_b = c_mem / b_uv

        J[0, 1] = 1.0
        J[1, 0] = kn2
        J[1, 3] = -0.125 / b_uv
        J[2, 0] = -2.0 * kn2 * x * x * K_R
        J[2, 3] = -kn2
        J[3, 1] = 2.0 * x * x * A_R + 32.0 * pn * pn * b_uv
        J[3, 2] = -1.0

        # Row 1 (Q derivatives) and Row 3 (Eff_P derivatives)
        if Np > 0:
            J[1, iYQ:iZQ] = c_over_b * alpha_p
            J[3, iYP:iZP] = -16.0 * pn * c_mem * alpha_p
        if Nu > 0:
            J[1, iUQ:iVQ] = c_over_b * w_re
            J[1, iVQ:iUP] = -c_over_b * w_im
            J[3, iUP:iVP] = -16.0 * pn * c_mem * w_re
            J[3, iVP:dim] = +16.0 * pn * c_mem * w_im

        # Discrete pole oscillator blocks
        if Np > 0:
            J[iYQ:iZQ, iZQ:iYP] = np.diag(E_p)
            J[iZQ:iYP, 3] = -0.125 * E_p / b_uv
            J[iZQ:iYP, iYQ:iZQ] = np.outer(E_p * c_over_b, alpha_p) - np.diag(E_p)
            if Nu > 0:
                J[iZQ:iYP, iUQ:iVQ] = np.outer(E_p * c_over_b, w_re)
                J[iZQ:iYP, iVQ:iUP] = np.outer(-E_p * c_over_b, w_im)
            J[iYP:iZP, iZP:iUQ] = np.diag(E_p)
            J[iZP:iUQ, 1] = 2.0 * pn * E_p
            J[iZP:iUQ, iYP:iZP] = -np.diag(E_p)

        # UV tail blocks
        if Nu > 0:
            J[iUQ:iVQ, 3] = -0.125 * u / b_uv
            J[iUQ:iVQ, iUQ:iVQ] = np.outer(u * c_over_b, w_re) - np.diag(u)
            J[iUQ:iVQ, iVQ:iUP] = np.outer(-u * c_over_b, w_im) - M_tail * np.eye(Nu)
            J[iVQ:iUP, 3] = +0.125 * M_tail / b_uv
            J[iVQ:iUP, iUQ:iVQ] = np.outer(-M_tail * c_over_b, w_re) + M_tail * np.eye(Nu)
            J[iVQ:iUP, iVQ:iUP] = np.outer(+M_tail * c_over_b, w_im) - np.diag(u)
            if Np > 0:
                J[iUQ:iVQ, iYQ:iZQ] = np.outer(u * c_over_b, alpha_p)
                J[iVQ:iUP, iYQ:iZQ] = np.outer(-M_tail * c_over_b, alpha_p)
            J[iUP:iVP, 1] = 2.0 * pn * u
            J[iUP:iVP, iUP:iVP] = -np.diag(u)
            J[iUP:iVP, iVP:dim] = -M_tail * np.eye(Nu)
            J[iVP:dim, 1] = -2.0 * pn * M_tail
            J[iVP:dim, iUP:iVP] = +M_tail * np.eye(Nu)
            J[iVP:dim, iVP:dim] = -np.diag(u)
        return J

    jac_err = None
    if check_jacobian:
        rng = np.random.default_rng(42)
        y_probe = rng.standard_normal(dim)
        eta_probe = min(0.37, 0.5 * eta_end)
        J_an = jac(eta_probe, y_probe)
        J_fd = np.zeros_like(J_an)
        eps_fd = 1e-6
        for j in range(dim):
            yp = y_probe.copy(); yp[j] += eps_fd
            ym = y_probe.copy(); ym[j] -= eps_fd
            J_fd[:, j] = (rhs(eta_probe, yp) - rhs(eta_probe, ym)) / (2.0 * eps_fd)
        jac_err = float(np.max(np.abs(J_an - J_fd)) / max(1.0, float(np.max(np.abs(J_an)))))

    def ev_blowup(eta, y):
        return blowup_cap - math.hypot(y[0], y[1] / max(1.0, math.sqrt(kn2)))
    ev_blowup.terminal = True
    ev_blowup.direction = -1

    def ev_buv_zero(eta, y):
        _, _, _, _, buv = get_bg(eta)
        return abs(buv) - 1e-5
    ev_buv_zero.terminal = True
    ev_buv_zero.direction = -1

    t_eval = np.linspace(0.0, eta_end, n_eval)
    kwargs = {"method": method, "rtol": rtol, "atol": atol, "t_eval": t_eval,
              "events": [ev_blowup, ev_buv_zero], "max_step": max_step}
    if method in ("Radau", "BDF"):
        kwargs["jac"] = jac

    sol = solve_ivp(rhs, (0.0, eta_end), y0, **kwargs)
    etas, ys = sol.t, sol.y
    blowup = len(sol.t_events[0]) > 0
    buv_zero = len(sol.t_events[1]) > 0
    if blowup:
        etas = np.append(etas, sol.t_events[0][0])
        ys = np.column_stack([ys, sol.y_events[0][0]])
    elif buv_zero:
        etas = np.append(etas, sol.t_events[1][0])
        ys = np.column_stack([ys, sol.y_events[1][0]])

    taus = etas * x_static if static_bg else bg_sol.sol(etas)[4]

    H_tot_drift = None
    if track_hamiltonian and static_bg and (not inc_tail):
        H_vals = np.empty(len(etas))
        _, _, _, _, b_uv_st = get_bg(0.0)
        for k in range(len(etas)):
            yk = ys[:, k]
            q1k, q2k, pi1k, pi2k = yk[0], yk[1], yk[2], yk[3]
            Pk = 2.0 * pn * q2k
            YQk = yk[iYQ:iZQ]; ZQk = yk[iZQ:iYP]
            YPk = yk[iYP:iZP]; ZPk = yk[iZP:iUQ]
            Qk = (-0.125 * pi2k + c_mem * float(np.dot(alpha_p, YQk))) / b_uv_st
            osc_Q = float(np.dot(alpha_p, YQk * YQk + ZQk * ZQk))
            osc_P = float(np.dot(alpha_p, YPk * YPk + ZPk * ZPk))
            H_vals[k] = (
                pi1k * q2k + kn2 * q1k * pi2k
                + (x_static ** 2) * (kn2 * q1k * q1k - q2k * q2k)
                - 4.0 * b_uv_st * (Qk * Qk + Pk * Pk)
                + 8.0 * c_mem * Pk * float(np.dot(alpha_p, YPk))
                + 4.0 * c_mem * (osc_Q - osc_P)
            )
        H0 = H_vals[0]
        H_tot_drift = float(np.max(np.abs(H_vals - H0)) / max(1.0, abs(H0)))

    return {
        "eta": etas,
        "tau": taus,
        "q1": ys[0],
        "q2": ys[1],
        "pi1": ys[2],
        "pi2": ys[3],
        "Y_Q": ys[iYQ:iZQ] if Np > 0 else None,
        "Y_P": ys[iYP:iZP] if Np > 0 else None,
        "blowup": blowup,
        "buv_zero": buv_zero,
        "success": bool(sol.success),
        "dim": int(dim),
        "jac_fd_rel_err": jac_err,
        "H_tot_max_rel_drift": H_tot_drift,
    }


# ==============================================================================
# 4. SOLVER B: INDEPENDENT DIRECT TIME-DOMAIN PRODUCT-INTEGRATION VOLTERRA
# ==============================================================================

def build_discrete_volterra_weights(N_steps, d_eta, pole_spec):
    """
    Exact closed-form panel integrals W_m = (F((m+1)h) - F(m h)) / h for
      K_disc(s) = -2*Ci(M_tail * s) [if include_tail] + sum_p (2*w_p/E_p) * cos(E_p * s).
    Exact antiderivative:
      F(s) = -2 * [s*Ci(M_tail*s) - sin(M_tail*s)/M_tail] + sum_p (2*w_p/E_p^2) * sin(E_p * s).
    ZERO auxiliary ODEs, ZERO exponential approximation!
    """
    s = np.arange(N_steps + 1, dtype=float) * d_eta
    F = np.zeros(N_steps + 1, dtype=float)
    if pole_spec["include_tail"] and pole_spec["M_tail"] > 0.0:
        Mt = float(pole_spec["M_tail"])
        sp = s[1:]
        _, ci = sici(Mt * sp)
        F[1:] += -2.0 * (sp * ci - np.sin(Mt * sp) / Mt)
    E_p = pole_spec["E_p"]
    w_p = pole_spec["w_p"]
    for Ep, wp in zip(E_p, w_p):
        F += (2.0 * wp / (Ep * Ep)) * np.sin(Ep * s)
    return (F[1:] - F[:-1]) / d_eta


def solve_discrete_direct_volterra(
    bg_sol,
    eta_end,
    beta_0,
    pole_spec,
    kappa_mem=1.0,
    n_mode=2,
    running=True,
    static_bg=False,
    x_static=X_RING,
    mu_phys=1.0,
    d_eta=0.002,
    y0_4d=None,
    blowup_cap=1e6,
    sigma_t=SIGMA_0,
):
    """
    COMPLETELY INDEPENDENT Direct Time-Domain Product-Integration Volterra Solver
    with implicit Crank-Nicolson stepping (exact 4x4 linear solve per step).
    """
    N_steps = int(round(eta_end / d_eta))
    etas = np.arange(N_steps + 1, dtype=float) * d_eta
    kn2 = float(n_mode * (n_mode + 2))
    pn = float(n_mode + 1)
    mu_cyl = X_RING * mu_phys
    c_mem = kappa_mem * C_ANOM

    W = build_discrete_volterra_weights(N_steps, d_eta, pole_spec) if abs(kappa_mem) > 0 else np.zeros(N_steps)
    Mt = float(pole_spec["M_tail"])
    delta_C = float(pole_spec["delta_C_tail"])
    c_loc = (
        math.log((Mt * Mt) / (mu_cyl * mu_cyl)) + delta_C - float(np.sum(2.0 * pole_spec["alpha_p"]))
    ) if abs(kappa_mem) > 0 else 0.0

    if static_bg:
        xs = np.full(N_steps + 1, x_static)
        A_arr = np.ones(N_steps + 1)
        K_arr = np.ones(N_steps + 1)
        taus = etas * x_static
    else:
        st = bg_sol.sol(etas)
        xs, hs, rs, taus = st[0], st[1], st[2], st[4]
        inv_x2 = 1.0 / (xs * xs)
        yy = hs * hs + inv_x2
        A_arr = 1.0 + 2.0 * sigma_t * rs - 2.0 * hs * hs + (2.0 / 3.0) * inv_x2
        K_arr = 1.0 + 2.0 * sigma_t * rs + 2.0 * hs * hs + (2.0 / 3.0) * inv_x2 - 4.0 * (rs - yy)

    beff_arr = beta_0 - (C_ANOM * np.log(xs / X_RING) if running else 0.0)

    q1 = np.zeros(N_steps + 1)
    q2 = np.zeros(N_steps + 1)
    pi1 = np.zeros(N_steps + 1)
    pi2 = np.zeros(N_steps + 1)
    Q = np.zeros(N_steps + 1)
    P = np.zeros(N_steps + 1)
    dQ_hist = np.zeros(N_steps)
    dP_hist = np.zeros(N_steps)

    if y0_4d is None:
        q1[0], q2[0], pi1[0], pi2[0] = 1.0, 0.0, 0.0, 0.0
        Q[0], P[0] = 0.0, 0.0
    else:
        q1[0], q2[0], pi1[0], pi2[0] = [float(v) for v in y0_4d]
        P[0] = 2.0 * pn * q2[0]
        b_init = beff_arr[0] + 0.5 * c_mem * c_loc
        Q[0] = -pi2[0] / (8.0 * b_init)

    h2 = 0.5 * d_eta
    W0 = float(W[0])
    k_end = N_steps
    blowup = False

    rhs_prev = np.array([
        q2[0],
        kn2 * q1[0] + Q[0],
        -2.0 * kn2 * (xs[0] ** 2) * K_arr[0] * q1[0] - kn2 * pi2[0],
        -pi1[0] + 2.0 * (xs[0] ** 2) * A_arr[0] * q2[0] + 16.0 * pn * (beff_arr[0] + 0.5 * c_mem * c_loc) * P[0],
    ], dtype=float)

    for k in range(1, N_steps + 1):
        if k >= 2 and abs(kappa_mem) > 0:
            w_slice = W[k - 1:0:-1]
            hist_Q = float(np.dot(w_slice, dQ_hist[:k - 1]))
            hist_P = float(np.dot(w_slice, dP_hist[:k - 1]))
        else:
            hist_Q = 0.0
            hist_P = 0.0

        # -pi2_k / 8 = [beff_k + 0.5*c_mem*(c_loc + W0)] * Q_k + 0.5*c_mem*(-W0*Q_{k-1} + hist_Q)
        b_step = beff_arr[k] + 0.5 * c_mem * (c_loc + W0)
        q_const = -0.5 * c_mem * (-W0 * Q[k - 1] + hist_Q) / b_step
        p_const = 0.5 * c_mem * (-W0 * P[k - 1] + hist_P)

        xk2 = xs[k] * xs[k]
        M_mat = np.eye(4, dtype=float)
        M_mat[0, 1] -= h2 * 1.0
        M_mat[1, 0] -= h2 * kn2
        M_mat[1, 3] -= h2 * (-0.125 / b_step)
        M_mat[2, 0] -= h2 * (-2.0 * kn2 * xk2 * K_arr[k])
        M_mat[2, 3] -= h2 * (-kn2)
        M_mat[3, 1] -= h2 * (2.0 * xk2 * A_arr[k] + 32.0 * pn * pn * b_step)
        M_mat[3, 2] -= h2 * (-1.0)

        c_vec = np.array([0.0, q_const, 0.0, 16.0 * pn * p_const], dtype=float)
        y_prev = np.array([q1[k - 1], q2[k - 1], pi1[k - 1], pi2[k - 1]], dtype=float)
        y_cur = np.linalg.solve(M_mat, y_prev + h2 * (rhs_prev + c_vec))

        q1[k], q2[k], pi1[k], pi2[k] = y_cur
        Q[k] = -0.125 * pi2[k] / b_step + q_const
        P[k] = 2.0 * pn * q2[k]
        dQ_hist[k - 1] = Q[k] - Q[k - 1]
        dP_hist[k - 1] = P[k] - P[k - 1]

        rhs_prev = np.array([
            q2[k],
            kn2 * q1[k] + Q[k],
            -2.0 * kn2 * xk2 * K_arr[k] * q1[k] - kn2 * pi2[k],
            -pi1[k] + 2.0 * xk2 * A_arr[k] * q2[k] + 16.0 * pn * (b_step * P[k] + p_const),
        ], dtype=float)

        if math.hypot(q1[k], q2[k] / max(1.0, math.sqrt(kn2))) > blowup_cap:
            k_end = k
            blowup = True
            break

    return {
        "eta": etas[:k_end + 1],
        "tau": taus[:k_end + 1],
        "q1": q1[:k_end + 1],
        "q2": q2[:k_end + 1],
        "pi1": pi1[:k_end + 1],
        "pi2": pi2[:k_end + 1],
        "blowup": blowup,
        "d_eta": d_eta,
    }


def fit_exponential_rate(ts, amps, frac_start=0.35, frac_end=0.98):
    """Fit exponential envelope rate from local maxima (peaks) of amps(t)."""
    t_max = float(ts[-1])
    t0, t1 = frac_start * t_max, frac_end * t_max
    if len(amps) >= 5:
        pk_idx = np.where((amps[1:-1] >= amps[:-2]) & (amps[1:-1] >= amps[2:]) & (amps[1:-1] > 1e-30))[0] + 1
        pk_idx = pk_idx[(ts[pk_idx] >= t0) & (ts[pk_idx] <= t1)]
        if len(pk_idx) >= 4:
            return float(np.polyfit(ts[pk_idx], np.log(amps[pk_idx]), 1)[0])
    m = (ts >= t0) & (ts <= t1) & (amps > 1e-30)
    if np.sum(m) >= 3:
        return float(np.polyfit(ts[m], np.log(amps[m]), 1)[0])
    return 0.0


# ==============================================================================
# 5. STAGE 2 BENCHMARK & CROSS-CHECK SUITE
# ==============================================================================

def main():
    print("=== STREAM p2_01_discrete_photon_pole_sum — STAGE 2: DUAL-SOLVER VERIFICATION ===", flush=True)
    bg_sol, bg_info = integrate_kam_background_eta(X_RING, eta_end=40.0)
    print(f"1. P05 Background x_ring (eta=40 -> tau={bg_info['tau_end']:.2f}): "
          f"max_rel_e00={bg_info['max_rel_e00']:.2e}, rms_rel_e00={bg_info['rms_rel_e00']:.2e}", flush=True)

    spec_is30 = build_discrete_pole_spec("Istar", J_cut=30, include_tail=True, match_exact_K0=True)
    spec_is60 = build_discrete_pole_spec("Istar", J_cut=60, include_tail=True, match_exact_K0=True)
    spec_is30_notail = build_discrete_pole_spec("Istar", J_cut=30, include_tail=False, match_exact_K0=True)

    # 2. Analytical Jacobian check + Exact Conserved Total Hamiltonian Check (static, pure discrete, no tail)
    #    Check BOTH high-precision DOP853 (eta=8) AND KT4 production BDF (rtol=1e-8, max_step=0.025) over full eta=35!
    r_ham = solve_discrete_aux_ode(
        bg_sol, 8.0, 1.0, spec_is30_notail, static_bg=True, y0_4d=[1.0, 0.0, 0.0, 0.0],
        method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.005, n_eval=1601,
        check_jacobian=True, track_hamiltonian=True,
    )
    r_ham_bdf35 = solve_discrete_aux_ode(
        bg_sol, 35.0, 1.0, spec_is30_notail, static_bg=True, y0_4d=[1.0, 0.0, 0.0, 0.0],
        method="BDF", rtol=1e-8, atol=1e-10, max_step=0.025, n_eval=1501,
        check_jacobian=False, track_hamiltonian=True,
    )
    # Also verify Radau Jacobian with tail included
    r_jac_tail = solve_discrete_aux_ode(
        bg_sol, 0.5, 1.0, spec_is30, static_bg=False, Nu=48, y0_4d=[0.0, 0.0, 1.0, 0.0],
        method="Radau", n_eval=51, check_jacobian=True,
    )
    print(f"2. Analytical Jacobian rel err: no_tail={r_ham['jac_fd_rel_err']:.2e} | with_tail={r_jac_tail['jac_fd_rel_err']:.2e}", flush=True)
    print(f"3. Exact Conserved Discrete-Pole Hamiltonian H_tot drift (static, beta_0=1.0): "
          f"DOP853(eta=8)={r_ham['H_tot_max_rel_drift']:.2e} | KT4-prod BDF(eta=35)={r_ham_bdf35['H_tot_max_rel_drift']:.2e}", flush=True)

    # 3. Exact RG-scale invariance check (mu_phys: 1 -> e, beta_0: 1.0 -> 1.0 + C_ANOM) for BOTH solvers
    rA_rg1 = solve_discrete_aux_ode(bg_sol, 2.0, 1.0, spec_is30, mu_phys=1.0, Nu=56, n_eval=401)
    rA_rg2 = solve_discrete_aux_ode(bg_sol, 2.0, 1.0 + C_ANOM, spec_is30, mu_phys=math.e, Nu=56, n_eval=401)
    rg_aux = float(np.max(np.abs(rA_rg1["q1"] - rA_rg2["q1"])) / max(1.0, np.max(np.abs(rA_rg1["q1"]))))
    rB_rg1 = solve_discrete_direct_volterra(bg_sol, 1.5, 1.0, spec_is30, mu_phys=1.0, d_eta=0.001)
    rB_rg2 = solve_discrete_direct_volterra(bg_sol, 1.5, 1.0 + C_ANOM, spec_is30, mu_phys=math.e, d_eta=0.001)
    rg_volt = float(np.max(np.abs(rB_rg1["q1"] - rB_rg2["q1"])) / max(1.0, np.max(np.abs(rB_rg1["q1"]))))
    print(f"4. Exact RG-scale invariance (mu -> e*mu): Solver A={rg_aux:.2e} | Solver B={rg_volt:.2e}", flush=True)

    # 4. Solver A (Aux-ODE) vs Solver B (Direct Volterra) vs Stage 1 40-digit exact roots!
    with open(STREAM_DIR / "01_derive.json") as fh:
        d1 = json.load(fh)
    exact_roots = {round(r["beta_0"], 4): r for r in d1["kt3_pinch_off_and_convergence"]["static_roots_Istar_discrete"]}

    y0_st = [0.0, 0.0, 1.0, 0.0]
    cross_checks = {}
    print("5. Solver A (Discrete Aux-ODE) vs Solver B (Direct Product-Volterra + Richardson) vs 40-Digit Exact Roots:", flush=True)
    for b0, eta_t, d_eta_v in [(0.289, 1.8, 0.0005), (0.50, 2.5, 0.0005), (1.00, 3.0, 0.001)]:
        n_ev = int(round(eta_t / d_eta_v)) + 1
        # Static background: Solver A (Radau) vs Solver B (h & h/2 -> Richardson O(h^4)!)
        rA_st = solve_discrete_aux_ode(bg_sol, eta_t, b0, spec_is30, static_bg=True, Nu=64,
                                       y0_4d=y0_st, n_eval=n_ev, method="Radau", rtol=1e-10, atol=1e-12, max_step=0.01)
        rB_st = solve_discrete_direct_volterra(bg_sol, eta_t, b0, spec_is30, static_bg=True,
                                               d_eta=d_eta_v, y0_4d=y0_st)
        rB_st_fine = solve_discrete_direct_volterra(bg_sol, eta_t, b0, spec_is30, static_bg=True,
                                                    d_eta=0.5 * d_eta_v, y0_4d=y0_st)
        nm = min(len(rA_st["pi2"]), len(rB_st["pi2"]))
        pi2_rich_st = (4.0 * rB_st_fine["pi2"][:2 * nm - 1:2] - rB_st["pi2"][:nm]) / 3.0
        scale_st = max(1.0, float(np.max(np.abs(rA_st["pi2"][:nm]))))
        rel_AB_st = float(np.max(np.abs(rA_st["pi2"][:nm] - rB_st["pi2"][:nm])) / scale_st)
        rel_AB_st_fine = float(np.max(np.abs(rA_st["pi2"][:nm] - rB_st_fine["pi2"][:2 * nm - 1:2])) / scale_st)
        rel_AB_st_rich = float(np.max(np.abs(rA_st["pi2"][:nm] - pi2_rich_st)) / scale_st)
        gam_A = fit_exponential_rate(rA_st["tau"], np.abs(rA_st["pi2"]), 0.45, 0.98)
        gam_B = fit_exponential_rate(rB_st["tau"], np.abs(rB_st["pi2"]), 0.45, 0.98)
        g_ex = exact_roots[round(b0, 4)]["uhp_gamma_HS"]
        g_ex = 0.0 if g_ex is None else g_ex
        err_vs_ex = abs(gam_A - g_ex) / abs(g_ex) if g_ex > 1e-10 else abs(gam_A)

        # Dynamical KAM background: Solver A (Radau) vs Solver A (BDF) vs Solver B (h & h/2 -> Richardson)
        rA_dyn = solve_discrete_aux_ode(bg_sol, eta_t, b0, spec_is30, static_bg=False, Nu=64,
                                        y0_4d=y0_st, n_eval=n_ev, method="Radau", rtol=1e-10, atol=1e-12, max_step=0.01)
        rA_bdf = solve_discrete_aux_ode(bg_sol, eta_t, b0, spec_is30, static_bg=False, Nu=64,
                                        y0_4d=y0_st, n_eval=n_ev, method="BDF", rtol=1e-10, atol=1e-12, max_step=0.01)
        rB_dyn = solve_discrete_direct_volterra(bg_sol, eta_t, b0, spec_is30, static_bg=False,
                                                d_eta=d_eta_v, y0_4d=y0_st)
        rB_dyn_fine = solve_discrete_direct_volterra(bg_sol, eta_t, b0, spec_is30, static_bg=False,
                                                     d_eta=0.5 * d_eta_v, y0_4d=y0_st)
        nmd = min(len(rA_dyn["pi2"]), len(rB_dyn["pi2"]))
        pi2_rich_dyn = (4.0 * rB_dyn_fine["pi2"][:2 * nmd - 1:2] - rB_dyn["pi2"][:nmd]) / 3.0
        scale_dyn = max(1.0, float(np.max(np.abs(rA_dyn["pi2"][:nmd]))))
        rel_AB_dyn = float(np.max(np.abs(rA_dyn["pi2"][:nmd] - rB_dyn["pi2"][:nmd])) / scale_dyn)
        rel_AB_dyn_fine = float(np.max(np.abs(rA_dyn["pi2"][:nmd] - rB_dyn_fine["pi2"][:2 * nmd - 1:2])) / scale_dyn)
        rel_AB_dyn_rich = float(np.max(np.abs(rA_dyn["pi2"][:nmd] - pi2_rich_dyn)) / scale_dyn)
        rel_radau_bdf = float(np.max(np.abs(rA_dyn["pi2"][:nmd] - rA_bdf["pi2"][:nmd])) / scale_dyn)
        gam_dyn_A = fit_exponential_rate(rA_dyn["tau"], np.abs(rA_dyn["pi2"]), 0.45, 0.98)
        gam_dyn_B = fit_exponential_rate(rB_dyn["tau"], np.abs(rB_dyn["pi2"]), 0.45, 0.98)

        lbl = f"Istar_disc_beta0_{b0}"
        cross_checks[lbl] = {
            "beta_0": b0, "eta_end": eta_t, "d_eta": d_eta_v,
            "static_rel_diff_AuxODE_vs_Volterra_h": rel_AB_st,
            "static_rel_diff_AuxODE_vs_Volterra_h2": rel_AB_st_fine,
            "static_volterra_order2_ratio": rel_AB_st / max(1e-16, rel_AB_st_fine),
            "static_rel_diff_AuxODE_vs_Volterra_Richardson": rel_AB_st_rich,
            "static_gamma_AuxODE_HS": gam_A,
            "static_gamma_Volterra_HS": gam_B,
            "static_gamma_mp40_exact_HS": g_ex,
            "static_gamma_err_vs_mp40": err_vs_ex,
            "dyn_KAM_rel_diff_Radau_vs_BDF": rel_radau_bdf,
            "dyn_KAM_rel_diff_AuxODE_vs_Volterra_h": rel_AB_dyn,
            "dyn_KAM_rel_diff_AuxODE_vs_Volterra_h2": rel_AB_dyn_fine,
            "dyn_volterra_order2_ratio": rel_AB_dyn / max(1e-16, rel_AB_dyn_fine),
            "dyn_KAM_rel_diff_AuxODE_vs_Volterra_Richardson": rel_AB_dyn_rich,
            "dyn_KAM_gamma_AuxODE_HS": gam_dyn_A,
            "dyn_KAM_gamma_Volterra_HS": gam_dyn_B,
        }
        print(f"   [{lbl:22s}] st Volt(h/h2/Rich)={rel_AB_st:.2e}/{rel_AB_st_fine:.2e}/{rel_AB_st_rich:.2e} "
              f"(ratio={rel_AB_st/rel_AB_st_fine:.2f}) | dyn Radau/BDF={rel_radau_bdf:.2e} | "
              f"dyn Rich={rel_AB_dyn_rich:.2e} | gam_A={gam_A:+.6f} vs mp40={g_ex:+.6f} (err={err_vs_ex:.2e})", flush=True)

    # 5b. High-precision static beta_0 = 0.50 check at J_cut = 60, eta = 3.2 vs 40-digit exact root!
    rA_60_st = solve_discrete_aux_ode(bg_sol, 3.2, 0.50, spec_is60, static_bg=True, Nu=64,
                                      y0_4d=y0_st, n_eval=3201, method="Radau", rtol=1e-10, atol=1e-12, max_step=0.01)
    gam_60_st = fit_exponential_rate(rA_60_st["tau"], np.abs(rA_60_st["pi2"]), 0.55, 0.98)
    g_ex_050 = exact_roots[0.5]["uhp_gamma_HS"]
    err_60_st = abs(gam_60_st - g_ex_050) / abs(g_ex_050)
    cross_checks["Istar_disc_beta0_0.5_J60_precision"] = {
        "eta_end": 3.2, "gamma_AuxODE_J60_HS": gam_60_st,
        "gamma_mp40_exact_HS": g_ex_050, "rel_err_vs_mp40": err_60_st,
    }
    print(f"   [J60 static b0=0.50] gam_J60={gam_60_st:+.7f} vs mp40={g_ex_050:+.7f} (rel err={err_60_st:.2e})", flush=True)

    # 5c. Discrete-pole oscillator sideband excitation check on smooth physical graviton IC (beta_0=1.0, eta=20)
    r_osc = solve_discrete_aux_ode(bg_sol, 20.0, 1.0, spec_is30, static_bg=False, Nu=48,
                                   y0_4d=None, n_eval=2001, method="BDF", rtol=1e-9, atol=1e-11, max_step=0.015)
    # Index 0 is j=6 (E_6=26), index 1 is j=10 (E_10=42), index 2 is j=12 (E_12=50)
    YQ_rms = np.sqrt(np.mean(r_osc["Y_Q"] ** 2, axis=1))
    YP_rms = np.sqrt(np.mean(r_osc["Y_P"] ** 2, axis=1))
    pole_exc = [
        {"E_p": float(spec_is30["E_p"][k]), "alpha_p": float(spec_is30["alpha_p"][k]),
         "YQ_rms": float(YQ_rms[k]), "YP_rms": float(YP_rms[k]),
         "weighted_YQ_rms": float(spec_is30["alpha_p"][k] * YQ_rms[k])}
        for k in range(min(6, len(spec_is30["E_p"])))
    ]
    print("   [Discrete pole oscillator RMS on smooth phys IC, b0=1.0, eta=20]:", flush=True)
    for pe in pole_exc[:4]:
        print(f"     E={pe['E_p']:5.1f}: YQ_rms={pe['YQ_rms']:.3e}, YP_rms={pe['YP_rms']:.3e}, alpha*YQ={pe['weighted_YQ_rms']:.3e}", flush=True)

    # 5. Time-domain cutoff convergence check: J_cut = 30 vs J_cut = 60 on dynamical x_ring (beta_0 = 0.50 & 1.00)
    print("6. Time-Domain Discrete-Pole Cutoff Check (J_cut=30 vs J_cut=60, dynamical x_ring):", flush=True)
    jcut_td = {}
    for b0, eta_t in [(0.50, 2.5), (1.00, 10.0)]:
        r30 = solve_discrete_aux_ode(bg_sol, eta_t, b0, spec_is30, Nu=48, y0_4d=y0_st, n_eval=501, method="Radau")
        r60 = solve_discrete_aux_ode(bg_sol, eta_t, b0, spec_is60, Nu=48, y0_4d=y0_st, n_eval=501, method="Radau")
        nm = min(len(r30["pi2"]), len(r60["pi2"]))
        rel_j = float(np.max(np.abs(r30["pi2"][:nm] - r60["pi2"][:nm])) / max(1.0, np.max(np.abs(r60["pi2"][:nm]))))
        g30 = fit_exponential_rate(r30["tau"], np.abs(r30["pi2"]), 0.40, 0.98)
        g60 = fit_exponential_rate(r60["tau"], np.abs(r60["pi2"]), 0.40, 0.98)
        jcut_td[f"beta0_{b0}"] = {
            "eta_end": eta_t, "rel_diff_pi2_J30_vs_J60": rel_j,
            "gamma_J30_HS": g30, "gamma_J60_HS": g60,
        }
        print(f"   beta_0={b0:.2f} (eta={eta_t:.1f}): rel_diff_pi2={rel_j:.2e} | gam_J30={g30:+.5e} vs gam_J60={g60:+.5e}", flush=True)

    # 6b. P1-3 Long-Horizon (eta=35, tau=773.4) Numerical-Dissipation Audit of KT4 Production BDF vs High-Accuracy Radau!
    print("7. P1-3 Long-Horizon (eta=35, tau=773.4) KT4-Prod BDF vs Tight-Tol Radau Cross-Check:", flush=True)
    long_horizon_audit = {}
    for b0 in [0.747, 1.00, 1.719]:
        r_prod = solve_discrete_aux_ode(
            bg_sol, 35.0, b0, spec_is30, Nu=44, u_max=1e6, y0_4d=y0_st,
            n_eval=1501, method="BDF", rtol=1e-8, atol=1e-10, max_step=0.025, blowup_cap=1e4,
        )
        r_ref = solve_discrete_aux_ode(
            bg_sol, 35.0, b0, spec_is30, Nu=64, u_max=1e7, y0_4d=y0_st,
            n_eval=1501, method="Radau", rtol=1e-10, atol=1e-12, max_step=0.01, blowup_cap=1e4,
        )
        g_prod = fit_exponential_rate(r_prod["tau"], np.abs(r_prod["pi2"]), 0.40, 0.98)
        g_ref = fit_exponential_rate(r_ref["tau"], np.abs(r_ref["pi2"]), 0.40, 0.98)
        rel_q1 = float(np.max(np.abs(r_prod["q1"] - r_ref["q1"])) / max(1e-12, np.max(np.abs(r_ref["q1"]))))
        rel_pi2 = float(np.max(np.abs(r_prod["pi2"] - r_ref["pi2"])) / max(1e-12, np.max(np.abs(r_ref["pi2"]))))
        long_horizon_audit[f"beta0_{b0}"] = {
            "eta_end": 35.0, "tau_end": float(r_ref["tau"][-1]),
            "gamma_prod_BDF_HS": g_prod, "gamma_ref_Radau_HS": g_ref,
            "abs_gamma_diff_HS": abs(g_prod - g_ref),
            "rel_diff_q1": rel_q1, "rel_diff_pi2": rel_pi2,
            "prod_blowup": bool(r_prod["blowup"]), "ref_blowup": bool(r_ref["blowup"]),
        }
        print(f"   b0={b0:5.3f} (tau={r_ref['tau'][-1]:.1f}): gam BDF={g_prod:+.4e} vs Radau={g_ref:+.4e} "
              f"(|dg|={abs(g_prod-g_ref):.2e}) | rel_q1={rel_q1:.2e} | rel_pi2={rel_pi2:.2e}", flush=True)

    # 6c. P2-2 True Physical-Graviton Eigenvector IC check (eliminating pi2(0)=0 Stelle startup transient)
    print("8. P2-2 True Graviton-Eigenvector IC Check (beta_0=1.0, 2.0, eta=35):", flush=True)
    grav_eig_check = []
    K0_is = float(d1["kt1_and_kt2_verification"]["kt1_subthreshold_analyticity"][0]["K_Istar_disc_re"])
    for b0 in [1.0, 2.0]:
        # Static graviton frequency Om_g from Stage 1
        Om_g = float(exact_roots[round(b0, 4)]["grav_Omega"])
        # On pure graviton mode b(eta)=cos(Om_g*eta): Q(0) = -(Om_g^2 + 8), pi2(0) = -8*(b0 + 0.5*C_ANOM*K0_is)*Q(0)
        b_eff0 = b0 + 0.5 * C_ANOM * K0_is
        pi2_g = 8.0 * b_eff0 * (Om_g * Om_g + 8.0)
        r_ge = solve_discrete_aux_ode(
            bg_sol, 35.0, b0, spec_is30, Nu=48, u_max=1e6, y0_4d=[1.0, 0.0, 0.0, pi2_g],
            n_eval=1501, method="BDF", rtol=1e-9, atol=1e-11, max_step=0.02, blowup_cap=1e4,
        )
        st = bg_sol.sol(r_ge["eta"])
        Ar, Kr = bg_AK_at_state(st[0], st[1], st[2])
        assert np.all(Ar > 0.0) and np.all(Kr > 0.0)
        El = (st[0] ** 2) * (Ar * (r_ge["q2"] ** 2) + 8.0 * Kr * (r_ge["q1"] ** 2))
        m = r_ge["eta"] >= 5.0
        gE = 0.5 * float(np.polyfit(r_ge["tau"][m], np.log(El[m]), 1)[0])
        grav_eig_check.append({
            "beta_0": b0, "pi2_init": pi2_g, "tau_end": float(r_ge["tau"][-1]),
            "gamma_E_HS": gE, "E_end_over_E0": float(El[-1] / El[0]), "E_max_over_E0": float(np.max(El) / El[0]),
            "blowup": bool(r_ge["blowup"]),
        })
        print(f"   [grav-eig IC] b0={b0:.1f}: gam_E={gE:+.4e} H_S | E_end/E0={El[-1]/El[0]:.4f} | E_max/E0={np.max(El)/El[0]:.4f}", flush=True)

    payload = {
        "stream_id": "p2_01_discrete_photon_pole_sum",
        "background_info": bg_info,
        "jacobian_fd_verification": {
            "no_tail_rel_err": r_ham["jac_fd_rel_err"],
            "with_tail_rel_err": r_jac_tail["jac_fd_rel_err"],
        },
        "exact_hamiltonian_conservation": {
            "mode_dop853_eta8": "static_xring_pure_discrete_J30_no_tail_beta0_1p0_eta_8p0",
            "H_tot_max_rel_drift": r_ham["H_tot_max_rel_drift"],
            "mode_kt4_prod_bdf_eta35": "static_xring_pure_discrete_J30_no_tail_beta0_1p0_eta_35p0_BDF_rtol1e-8_maxstep0p025",
            "H_tot_kt4_prod_bdf_eta35_max_rel_drift": r_ham_bdf35["H_tot_max_rel_drift"],
        },
        "rg_scale_invariance": {
            "solver_A_aux_ode_rel_diff": rg_aux,
            "solver_B_direct_volterra_rel_diff": rg_volt,
        },
        "dual_solver_cross_checks": cross_checks,
        "discrete_pole_oscillator_excitation_b0_1p0": pole_exc,
        "time_domain_Jcut_convergence": jcut_td,
        "long_horizon_eta35_bdf_vs_radau_audit": long_horizon_audit,
        "graviton_eigenvector_ic_check": grav_eig_check,
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
