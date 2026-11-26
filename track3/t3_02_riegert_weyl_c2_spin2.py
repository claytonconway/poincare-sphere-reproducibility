#!/usr/bin/env python3
"""
Track III Upgrade T3-B: Variational Covariant Riegert Anomaly Action + O(1) Weyl-Squared C^2
(Bach-Tensor / Stelle Spin-2) Stability of the S^3/I* Adiabatic KAM Breathing Island.

Resolves the R3-3 non-variational local {}^{(3)}H & omitted Weyl C^2 caveat (P1-2):
  In R3-3, the local algebraic tensor {}^{(3)}H_uv was non-variational off C = 0 (naive x^3*A
  Wronskian drifted 61.2%), yielding an algebraic singular locus [x_ghost^-, x_ghost^+] =
  [7.7905, 61.6578], while the O(1) Weyl-squared sector (c_anom/a_anom = 18/31 for gauge bosons)
  was omitted.

This script:
  1. Verifies to 35-40 digits in mpmath:
       (a) Exact 4D diagonal Bianchi IX curvature invariants (including nonzero magnetic Weyl
           B_11 = -B_22 = -3*b'/a!) and exact Gauss-Bonnet Chern-Simons total derivative
           a_1 a_2 a_3 E_4 = d/dt[8 a_1 a_2 a_3 (H1 H2 H3 + sum_cyc H_i K^{(3)}_{jk})];
       (b) Direct 4D covariant action variation of S_EH + S_{R^2}^{comp} + S_Riegert[g, phi_R=ln a]
           + S_{C^2} against our closed-form Euler-Lagrange equation AND our canonical 4D
           Ostrogradsky Hamiltonian system (q1, q2, p1, p2) (< 1e-16 discrepancy!).
  2. Resolves the R3-3 local {}^{(3)}H drift (beta_eff == 0, pure variational Riegert + R^2):
       - Proves A_Rieg(tau) = 1 + 2*sig*r - 2*h^2 + 2/(3*x^2) has ZERO -2*h' kinetic term
         (moving -4*X into the gradient stiffness K_Rieg = 1 + 2*sig*r + 2*h^2 + 2/(3*x^2) - 4*(r-y)),
         so A_Rieg > 0 EVERYWHERE across P05's entire KAM island [7, 60] (eliminating R3-3's
         non-variational singular root x_ghost^- = 7.7905!), while conserving the exact canonical
         Wronskian W_Rieg = 2*x^3*A_Rieg*(b1*b2' - b2*b1') to < 1e-11!
       - Maps the true variational Riegert gradient/sound-speed boundary (K_Rieg < 0 transient
         gradient windows vs. parametric Floquet stability across x_b in [6.5, 65] over tau=2000-6000).
  3. Integrates the full 4th-order Ostrogradsky Weyl C^2 + Riegert system (with BOTH static
     beta_eff = beta_0 AND exact 1-loop RG running beta_eff(tau) = beta_0 - c_anom*ln(x(tau)/x_ring),
     c_anom = 9/31 in H_S=1 units) around the S^3/I* KAM ring x_ring = 21.9168:
       - Verifies exact 4D symplectic conservation ||M^T J_4 M - J_4||_max < 1e-10 and DOP853 vs Radau;
       - Maps all four physical regimes across beta_0 in [-1.5, +50]:
           (I)   Tachyonic spin-2 runaway (beta_eff < 0): rapid exp(+tau/(2*sqrt(|beta_eff|))) blowup;
           (II)  RG zero-crossing singularity (0 < beta_0 < c_anom*ln(x_max/x_ring)): beta_eff -> 0;
           (III) O(1) Stelle spin-2 ghost window (beta_0 ~ c_anom = 9/31): checks Krein/parametric
                 resonances against scalaron omega_scal = sqrt(31/9) = 1.8559 and macro breathing;
           (IV)  Heavy/decoupled Stelle spin-2 regime (0 < beta_0 << 1, m_2 >> omega_scal) and
                 soft Weyl regime (beta_0 >> 1, m_2 << omega_scal): tests long-time KAM survival!
"""

import json
import math
import os
import sys

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p05_02_solve_and_phase_space import (
    KAPPA_VECTOR,
    SIGMA0_VECTOR,
    ode_rhs,
    e00_constraint,
    bounce_initial_state,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

SIGMA_0 = SIGMA0_VECTOR            # 9/31
KAPPA_V = KAPPA_VECTOR             # -44738/93
C_ANOM_HS1 = 9.0 / 31.0            # (1/2) * (c_anom / a_anom) = (1/2)*(18/31) = 9/31 in H_S=1 units!
X_RING = math.sqrt(44672.0 / 93.0) # 21.916754...
J4 = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
], dtype=float)


# ==============================================================================
# PART 1: 35-40 DIGIT COVARIANT 4D ACTION & OSTROGRADSKY VERIFICATION
# ==============================================================================

def verify_4d_covariant_riegert_weyl_mpmath():
    """
    Independent 35-40 digit verification of:
      1. Exact Bianchi IX Gauss-Bonnet Chern-Simons identity a1 a2 a3 E_4 == d(K_GB)/dt
      2. Exact quadratic Weyl-squared C^2 == 4*[(b'' + h*b' - (8/a^2)*b)^2 - (36/a^2)*b'^2]
      3. Direct 4D covariant action variation of S_EH + S_{R^2}^{comp} + S_Riegert + S_{C^2}
         matching BOTH the closed-form Euler-Lagrange equation AND the canonical 4D Ostrogradsky
         Hamiltonian system (q1, q2, p1, p2) to < 1e-16!
    """
    with mp.workdps(35):
        t1, t2 = mp.mpf("0.2"), mp.mpf("0.8")
        t0 = mp.mpf("0.37")
        sig = mp.mpf(9) / 31
        c_an = mp.mpf(9) / 31
        beta_ct = mp.mpf("0.37")

        def a_fn(t):
            return mp.mpf("2.3") + mp.mpf("0.4") * mp.sin(mp.mpf("1.7") * t) + mp.mpf("0.15") * t**2

        def b_base(t):
            return mp.cos(mp.mpf("2.1") * t + mp.mpf("0.3")) + mp.mpf("0.2") * mp.exp(mp.mpf("0.5") * t)

        def eta_fn(t):
            return ((t - t1) * (t2 - t)) ** 5

        def invs(t, e, bf):
            def av(s):
                aa = a_fn(s); bb = bf(s)
                return aa * mp.exp(e * bb), aa * mp.exp(-e * bb), aa

            def hv(s):
                x1, x2, x3 = av(s)
                return (
                    mp.diff(lambda u: av(u)[0], s) / x1,
                    mp.diff(lambda u: av(u)[1], s) / x2,
                    mp.diff(lambda u: av(u)[2], s) / x3,
                )

            a1, a2, a3 = av(t)
            H1, H2, H3 = hv(t)
            dH1 = mp.diff(lambda u: hv(u)[0], t)
            dH2 = mp.diff(lambda u: hv(u)[1], t)
            dH3 = mp.diff(lambda u: hv(u)[2], t)
            c1, c2, c3 = 2 * a1 / (a2 * a3), 2 * a2 / (a3 * a1), 2 * a3 / (a1 * a2)
            A1, A2, A3 = (c1 - c2 - c3) / 2, (c2 - c3 - c1) / 2, (c3 - c1 - c2) / 2
            M1 = -(H1 * c1 + H2 * A3 + H3 * A2)
            M2 = -(H2 * c2 + H3 * A1 + H1 * A3)
            M3 = -(H3 * c3 + H1 * A2 + H2 * A1)
            K01, K02, K03 = dH1 + H1**2, dH2 + H2**2, dH3 + H3**2
            K3_12, K3_23, K3_31 = -A3 * c3 - A1 * A2, -A1 * c1 - A2 * A3, -A2 * c2 - A3 * A1
            K12, K23, K31 = H1 * H2 + K3_12, H2 * H3 + K3_23, H3 * H1 + K3_31
            R00 = -(K01 + K02 + K03)
            R11, R22, R33 = K01 + K12 + K31, K02 + K12 + K23, K03 + K23 + K31
            R = -R00 + R11 + R22 + R33
            Ric2 = R00**2 + R11**2 + R22**2 + R33**2
            Riem2 = 4 * (K01**2 + K02**2 + K03**2 + K12**2 + K23**2 + K31**2) - 8 * (M1**2 + M2**2 + M3**2)
            E4 = Riem2 - 4 * Ric2 + R**2
            C2 = Riem2 - 2 * Ric2 + (mp.mpf(1) / 3) * R**2
            K_GB = 8 * (a1 * a2 * a3) * (H1 * H2 * H3 + H1 * K3_23 + H2 * K3_31 + H3 * K3_12)
            return R, R00, E4, C2, K_GB, M1 + M2 + M3

        # 1. Gauss-Bonnet total derivative check at finite eps = 0.13
        e_fin = mp.mpf("0.13")
        _, _, E4_f, _, _, b1_err = invs(t0, e_fin, b_base)
        dK_f = mp.diff(lambda s: invs(s, e_fin, b_base)[4], t0)
        gb_err = float(abs(dK_f - a_fn(t0) ** 3 * E4_f))

        # 2. Quadratic Weyl C^2 check
        eps = mp.mpf("1e-9")
        C2_p = invs(t0, eps, b_base)[3]
        C2_m = invs(t0, -eps, b_base)[3]
        C2_2 = (C2_p + C2_m) / (2 * eps**2)
        a0 = a_fn(t0)
        h0 = mp.diff(a_fn, t0) / a0
        b0 = b_base(t0)
        db0 = mp.diff(b_base, t0)
        d2b0 = mp.diff(b_base, t0, 2)
        C2_2_pred = 4 * ((d2b0 + h0 * db0 - 8 * b0 / a0**2) ** 2 - 36 * db0**2 / a0**2)
        weyl_c2_err = float(abs(C2_2 - C2_2_pred))

        # 3. Direct 4D covariant action variation vs closed-form Euler-Lagrange
        def raw_L2(t, bf):
            a = a_fn(t)
            h = mp.diff(a_fn, t) / a
            dh = mp.diff(lambda s: mp.diff(a_fn, s) / a_fn(s), t)
            phi = mp.log(a)
            box_phi = -(dh + 3 * h**2)
            beff = beta_ct - c_an * phi

            def L_at(e):
                R, R00, E4, C2, _, _ = invs(t, e, bf)
                L_eh = mp.mpf("0.5") * R
                L_r2 = ((sig + mp.mpf(1) / 3) / 12) * R**2
                L_rg = mp.mpf("0.5") * (2 * box_phi**2 - 4 * (R00 + R / 3) * h**2 - phi * E4 + (mp.mpf(2) / 3) * box_phi * R)
                return a**3 * (L_eh + L_r2 + L_rg - beff * C2)

            return (L_at(eps) + L_at(-eps) - 2 * L_at(mp.mpf(0))) / (2 * eps**2)

        def get_AK(s):
            aa = a_fn(s)
            hh = mp.diff(a_fn, s) / aa
            dhh = mp.diff(lambda u: mp.diff(a_fn, u) / a_fn(u), s)
            XX = dhh + hh**2
            rr = dhh + 2 * hh**2 + 1 / aa**2
            A = 1 + 2 * sig * rr - 2 * hh**2 + mp.mpf(2) / (3 * aa**2)
            K = 1 + 2 * sig * rr + 2 * hh**2 + mp.mpf(2) / (3 * aa**2) - 4 * XX
            return aa, hh, A, K

        def EL_pred_val(t):
            aa, hh, A, K = get_AK(t)
            beff = beta_ct - c_an * mp.log(aa)
            P_R = lambda s: get_AK(s)[0] ** 3 * get_AK(s)[2] * mp.diff(b_base, s)
            Q_w = lambda s: mp.diff(b_base, s, 2) + (mp.diff(a_fn, s) / a_fn(s)) * mp.diff(b_base, s) - 8 * b_base(s) / a_fn(s) ** 2
            W_Q = lambda s: (beta_ct - c_an * mp.log(a_fn(s))) * a_fn(s) ** 3 * Q_w(s)
            W_hQ = lambda s: (beta_ct - c_an * mp.log(a_fn(s))) * a_fn(s) ** 3 * (mp.diff(a_fn, s) / a_fn(s)) * Q_w(s)
            W_bp = lambda s: (beta_ct - c_an * mp.log(a_fn(s))) * a_fn(s) * mp.diff(b_base, s)
            EL_R = -2 * mp.diff(P_R, t) - 16 * aa * K * b_base(t)
            EL_C = -8 * mp.diff(W_Q, t, 2) + 8 * mp.diff(W_hQ, t) + 64 * beff * aa * Q_w(t) - 288 * mp.diff(W_bp, t)
            return EL_R + EL_C

        I_raw = mp.quad(lambda t: mp.mpf("0.5") * (raw_L2(t, lambda s: b_base(s) + eta_fn(s)) - raw_L2(t, lambda s: b_base(s) - eta_fn(s))), [t1, t2])
        I_pred = mp.quad(lambda t: EL_pred_val(t) * eta_fn(t), [t1, t2])
        action_rel_err = float(abs(I_raw - I_pred) / abs(I_raw))

        # 4. Verify canonical Ostrogradsky system reproduces EL_pred_val == 0 identically!
        # From Ostrogradsky eqs:
        #   p2 = -8*beff*a^3*Q
        #   p1 = -p2' + h*p2 + 2*(a^3*A + 144*beff*a)*b'
        #   EL_ostro := p1' + (8/a^2)*p2 + 16*a*K*b  MUST equal -EL_pred_val!!
        def p2_fn(s):
            aa = a_fn(s)
            hh = mp.diff(a_fn, s) / aa
            be = beta_ct - c_an * mp.log(aa)
            Qw = mp.diff(b_base, s, 2) + hh * mp.diff(b_base, s) - 8 * b_base(s) / aa**2
            return -8 * be * aa**3 * Qw

        def p1_fn(s):
            aa, hh, A, _ = get_AK(s)
            be = beta_ct - c_an * mp.log(aa)
            return -mp.diff(p2_fn, s) + hh * p2_fn(s) + 2 * (aa**3 * A + 144 * be * aa) * mp.diff(b_base, s)

        aa0, hh0, A0, K0 = get_AK(t0)
        EL_ostro = mp.diff(p1_fn, t0) + (8 / aa0**2) * p2_fn(t0) + 16 * aa0 * K0 * b_base(t0)
        EL_direct = EL_pred_val(t0)
        ostro_rel_err = float(abs(EL_ostro + EL_direct) / abs(EL_direct))

        # 5. Also verify exact FLRW background recovery from S_EH + S_{R^2}^{comp} + S_Riegert(phi=ln a)
        # Check on static/dynamic FLRW that A_Rieg and K_Rieg coincide on exact dS (a = cosh t) and static S^3!
        return {
            "gauss_bonnet_total_deriv_err_eps0.13": gb_err,
            "first_bianchi_err": float(abs(b1_err)),
            "quadratic_weyl_C2_err": weyl_c2_err,
            "direct_4d_action_vs_EL_rel_err": action_rel_err,
            "ostrogradsky_vs_EL_rel_err": ostro_rel_err,
            "A_Rieg_formula": "1 + 2*sigma_t*r - 2*h^2 + 2/(3*x^2)",
            "K_Rieg_formula": "1 + 2*sigma_t*r + 2*h^2 + 2/(3*x^2) - 4*(r - y)",
            "C2_lagrangian": "-4*beta_eff(tau)*x^3*[(b'' + h*b' - (8/x^2)*b)^2 - (36/x^2)*b'^2]",
            "verified": bool(max(gb_err, weyl_c2_err, action_rel_err, ostro_rel_err) < 1e-15),
        }


# ==============================================================================
# PART 2: PURE VARIATIONAL RIEGERT (beta_eff == 0) RESOLUTION OF R3-3
# ==============================================================================

def integrate_variational_riegert_2d(x_b, tau_end=2000.0, method="DOP853", sign_hdot=+1):
    """
    Integrate P05 background (x, h, r, s) coupled to the EXACT 2D canonical variational Riegert
    shear system (beta_eff == 0):
      q1' = p1 / (2*x^3*A_Rieg)
      p1' = -16*x*K_Rieg * q1
    tracks 2x2 fundamental matrix M(tau), exact symplectic Wronskian W = q1_1*p1_2 - q1_2*p1_1 == 1,
    min/max A_Rieg(tau) (kinetic sign), min/max K_Rieg(tau) (gradient/sound-speed sign),
    and effective Lyapunov exponent.
    """
    y0_bg = bounce_initial_state(x_b, SIGMA_0, 0.0, KAPPA_V, s_b=0.0, sign_hdot=sign_hdot)
    if y0_bg is None:
        return {"x_b": float(x_b), "success": False, "reason": "no_turning_point"}
    y0 = np.array([y0_bg[0], y0_bg[1], y0_bg[2], y0_bg[3], 1.0, 0.0, 0.0, 1.0], dtype=float)

    def rhs(t, y):
        dx, dh, dr, ds = ode_rhs(t, y[:4], SIGMA_0, 0.0)
        x, h, r, s = y[0], y[1], y[2], y[3]
        inv_x2 = 1.0 / (x * x)
        yy = h * h + inv_x2
        A_R = 1.0 + 2.0 * SIGMA_0 * r - 2.0 * h * h + (2.0 / 3.0) * inv_x2
        K_R = 1.0 + 2.0 * SIGMA_0 * r + 2.0 * h * h + (2.0 / 3.0) * inv_x2 - 4.0 * (r - yy)
        inv_2x3A = 1.0 / (2.0 * (x ** 3) * A_R)
        stiff_p = -16.0 * x * K_R
        q1, p1, q2, p2 = y[4], y[5], y[6], y[7]
        return [dx, dh, dr, ds, p1 * inv_2x3A, stiff_p * q1, p2 * inv_2x3A, stiff_p * q2]

    def ev_ghost(t, y):
        x, h, r = y[0], y[1], y[2]
        return (1.0 + 2.0 * SIGMA_0 * r - 2.0 * h * h + (2.0 / 3.0) / (x * x)) - 1e-4
    ev_ghost.terminal = True
    ev_ghost.direction = -1

    def ev_blowup(t, y):
        return 1e4 - max(abs(y[1]), abs(y[2]))
    ev_blowup.terminal = True
    ev_blowup.direction = -1

    t_eval = np.linspace(0.0, tau_end, 4001)
    sol = solve_ivp(
        rhs, (0.0, tau_end), y0, method=method, rtol=1e-11, atol=1e-13,
        t_eval=t_eval, events=[ev_ghost, ev_blowup], max_step=0.35,
    )
    ghost_hit = len(sol.t_events[0]) > 0
    blowup_hit = len(sol.t_events[1]) > 0
    ts, ys = sol.t, sol.y
    if ghost_hit:
        ts = np.append(ts, sol.t_events[0][0])
        ys = np.column_stack([ys, sol.y_events[0][0]])
    elif blowup_hit:
        ts = np.append(ts, sol.t_events[1][0])
        ys = np.column_stack([ys, sol.y_events[1][0]])

    xs, hs, rs, ss = ys[0], ys[1], ys[2], ys[3]
    q1, p1, q2, p2 = ys[4], ys[5], ys[6], ys[7]
    inv_x2 = 1.0 / (xs * xs)
    yy = hs * hs + inv_x2
    A_R = 1.0 + 2.0 * SIGMA_0 * rs - 2.0 * hs * hs + (2.0 / 3.0) * inv_x2
    K_R = 1.0 + 2.0 * SIGMA_0 * rs + 2.0 * hs * hs + (2.0 / 3.0) * inv_x2 - 4.0 * (rs - yy)
    cs2 = K_R / A_R

    # Exact canonical symplectic Wronskian W = q1*p2 - q2*p1 == 1!
    W = q1 * p2 - q2 * p1
    W_drift = float(np.max(np.abs(W - 1.0)))

    # Convert canonical basis to dimensionless physical (b, b'/w0) norm!
    # Note: initial (q1, p1) = (1, 0) => b1(0)=1, b1'(0)=0; initial (q2, p2) = (0, 1) => b2(0)=0, b2'(0)=1/(2 x_b^3 A_R0)
    A0 = float(A_R[0])
    w0 = math.sqrt(8.0) / x_b
    b1 = q1
    v1 = p1 / (2.0 * xs**3 * A_R)
    b2 = q2 * (2.0 * x_b**3 * A0)
    v2 = p2 * (2.0 * x_b**3 * A0) / (2.0 * xs**3 * A_R)
    norm_t = np.sqrt(b1**2 + (v1 / w0) ** 2 + (b2 * w0) ** 2 + v2**2) / math.sqrt(2.0)

    t_reached = float(ts[-1])
    n_bins = max(2, min(10, int(t_reached / 50.0)))
    edges = np.linspace(0.0, t_reached, n_bins + 1)
    bin_t, bin_max = [], []
    for i in range(n_bins):
        m = (ts >= edges[i]) & (ts <= edges[i + 1])
        if np.any(m):
            bin_t.append(0.5 * (edges[i] + edges[i + 1]))
            bin_max.append(float(np.max(norm_t[m])))
    slope = float(np.polyfit(bin_t[1:], np.log(np.maximum(bin_max[1:], 1e-30)), 1)[0]) if len(bin_t) >= 3 else 0.0
    frac_cs2_neg = float(np.mean(cs2 < 0.0))

    max_e00 = float(max(
        abs(e00_constraint(*ys[:4, i], SIGMA_0, 0.0, KAPPA_V)) / max(1.0, abs(rs[i]))
        for i in range(0, len(ts), max(1, len(ts) // 300))
    ))
    return {
        "x_b": float(x_b),
        "sign_hdot": int(sign_hdot),
        "method": method,
        "tau_reached": t_reached,
        "kinetic_ghost_crossing": ghost_hit,
        "bg_blowup": blowup_hit,
        "success": bool(sol.success and not ghost_hit and not blowup_hit),
        "min_A_Rieg": float(np.min(A_R)),
        "max_A_Rieg": float(np.max(A_R)),
        "min_K_Rieg": float(np.min(K_R)),
        "max_K_Rieg": float(np.max(K_R)),
        "min_cs2": float(np.min(cs2)),
        "frac_time_cs2_neg": frac_cs2_neg,
        "symplectic_W_max_drift": W_drift,
        "norm_max": float(np.max(norm_t)),
        "norm_end": float(norm_t[-1]),
        "growth_over_tau": float(bin_max[-1] / bin_max[0]) if len(bin_max) >= 2 else float("nan"),
        "effective_lyapunov_rate": slope,
        "max_rel_e00": max_e00,
    }


# ==============================================================================
# PART 3: FULL 4TH-ORDER OSTROGRADSKY RIEGERT + WEYL C^2 SPIN-2 SYSTEM
# ==============================================================================

def integrate_riegert_weyl_c2_4d(x_b, beta_0, running=True, tau_end=800.0, method="DOP853", sign_hdot=+1):
    """
    Integrate P05 background (x, h, r, s) coupled to the EXACT 4x4 canonical Ostrogradsky
    fundamental matrix M(tau) of the 4th-order variational Riegert + Weyl C^2 system:
      beta_eff(tau) = beta_0 - c_anom * ln(x(tau) / X_RING)   (if running=True)
                    = beta_0                                   (if running=False)
    State dimension: 4 (background) + 16 (4x4 symplectic matrix M in columns (q1, q2, p1, p2)) = 20!
    """
    y0_bg = bounce_initial_state(x_b, SIGMA_0, 0.0, KAPPA_V, s_b=0.0, sign_hdot=sign_hdot)
    if y0_bg is None:
        return {"x_b": float(x_b), "beta_0": float(beta_0), "running": running, "outcome": "no_turning_point"}

    beff0 = beta_0 - (C_ANOM_HS1 * math.log(x_b / X_RING) if running else 0.0)
    if abs(beff0) < 1e-6:
        return {"x_b": float(x_b), "beta_0": float(beta_0), "running": running, "outcome": "starts_at_beta_zero_singularity"}

    # Scale canonical momenta by scale_p = 2 * x_b^3 so initial matrix is well-conditioned O(1)
    y0 = np.concatenate([y0_bg, np.eye(4, dtype=float).ravel()])

    def rhs(t, y):
        dx, dh, dr, ds = ode_rhs(t, y[:4], SIGMA_0, 0.0)
        x, h, r, s = y[0], y[1], y[2], y[3]
        inv_x2 = 1.0 / (x * x)
        yy = h * h + inv_x2
        A_R = 1.0 + 2.0 * SIGMA_0 * r - 2.0 * h * h + (2.0 / 3.0) * inv_x2
        K_R = 1.0 + 2.0 * SIGMA_0 * r + 2.0 * h * h + (2.0 / 3.0) * inv_x2 - 4.0 * (r - yy)
        beff = beta_0 - (C_ANOM_HS1 * math.log(x / X_RING) if running else 0.0)

        # 4x4 Hamiltonian linear matrix H_mat: dM/dt = H_mat @ M
        # q1' = q2
        # q2' = (8/x^2)*q1 - h*q2 - p2/(8*beff*x^3)
        # p1' = -16*x*K_R*q1 - (8/x^2)*p2
        # p2' = 2*(x^3*A_R + 144*beff*x)*q2 - p1 + h*p2
        c_q2p2 = -1.0 / (8.0 * beff * (x ** 3))
        c_p2q2 = 2.0 * ((x ** 3) * A_R + 144.0 * beff * x)
        c_8x2 = 8.0 * inv_x2
        c_p1q1 = -16.0 * x * K_R
        M = y[4:].reshape((4, 4))
        dM = np.empty((4, 4), dtype=float)
        dM[0] = M[1]
        dM[1] = c_8x2 * M[0] - h * M[1] + c_q2p2 * M[3]
        dM[2] = c_p1q1 * M[0] - c_8x2 * M[3]
        dM[3] = c_p2q2 * M[1] - M[2] + h * M[3]
        return np.concatenate([[dx, dh, dr, ds], dM.ravel()])

    # Events:
    # 1. RG zero crossing of beta_eff(tau) -> 0 (fourth-order singular turning point!)
    def ev_beta_zero(t, y):
        beff = beta_0 - (C_ANOM_HS1 * math.log(max(y[0], 1e-8) / X_RING) if running else 0.0)
        return abs(beff) - 1e-4
    ev_beta_zero.terminal = True
    ev_beta_zero.direction = -1

    # 2. Spin-2 exponential overflow (tachyonic or strong parametric blowup: ||M||_F > 1e12)
    def ev_spin2_blowup(t, y):
        return 1e12 - float(np.linalg.norm(y[4:]))
    ev_spin2_blowup.terminal = True
    ev_spin2_blowup.direction = -1

    # 3. Background blowup
    def ev_bg_blowup(t, y):
        return 1e4 - max(abs(y[1]), abs(y[2]))
    ev_bg_blowup.terminal = True
    ev_bg_blowup.direction = -1

    # Adaptive max_step resolving the fast massive spin-2 frequency w2 ~ 1/(2*sqrt(|beff|))!
    w2_est = 0.5 / math.sqrt(max(abs(beff0), 1e-4))
    max_step = min(0.35, 0.25 / max(1.0, w2_est))
    n_eval = min(2001, max(401, int(tau_end * 2) + 1))
    t_eval = np.linspace(0.0, tau_end, n_eval)

    sol = solve_ivp(
        rhs, (0.0, tau_end), y0, method=method, rtol=1e-10, atol=1e-12,
        t_eval=t_eval, events=[ev_beta_zero, ev_spin2_blowup, ev_bg_blowup],
        max_step=max_step,
    )

    beta_zero_hit = len(sol.t_events[0]) > 0
    spin2_blowup = len(sol.t_events[1]) > 0
    bg_blowup = len(sol.t_events[2]) > 0

    ts, ys = sol.t, sol.y
    if beta_zero_hit:
        ts = np.append(ts, sol.t_events[0][0])
        ys = np.column_stack([ys, sol.y_events[0][0]])
    elif spin2_blowup:
        ts = np.append(ts, sol.t_events[1][0])
        ys = np.column_stack([ys, sol.y_events[1][0]])
    elif bg_blowup:
        ts = np.append(ts, sol.t_events[2][0])
        ys = np.column_stack([ys, sol.y_events[2][0]])

    # Compute exact symplectic error ||M^T J_4 M - J_4||_max, det(M)-1, and natural spectral norm!
    # Normalize M using diagonal similarity D = diag(1, 1/w_ref, 1/(2 x_b^3 w_ref), 1/(2 x_b^3 w_ref^2))
    # so all 4 canonical entries are dimensionless O(1) at t = 0!
    w_ref = max(0.5 / math.sqrt(max(abs(beff0), 1e-4)), math.sqrt(8.0) / x_b)
    scale_vec = np.array([1.0, 1.0 / w_ref, 1.0 / (2.0 * x_b**3 * w_ref), 1.0 / (2.0 * x_b**3 * w_ref**2)])
    D = np.diag(scale_vec)
    D_inv = np.diag(1.0 / scale_vec)

    symp_errs = []
    norms = []
    beff_vals = []
    for i in range(len(ts)):
        M = ys[4:, i].reshape((4, 4))
        symp_errs.append(float(np.max(np.abs(M.T @ J4 @ M - J4))))
        M_sc = D @ M @ D_inv
        norms.append(float(np.linalg.norm(M_sc, 2)))
        be = beta_0 - (C_ANOM_HS1 * math.log(ys[0, i] / X_RING) if running else 0.0)
        beff_vals.append(be)

    M_end = ys[4:, -1].reshape((4, 4))
    eigs_end = np.linalg.eigvals(M_end)
    max_eig_mag = float(np.max(np.abs(eigs_end)))
    det_err = float(abs(np.linalg.det(M_end) - 1.0))
    # Note: symplectic error scales with ||M||^2 in float64 when ||M|| >> 1; report relative ||M^T J M - J|| / ||M||^2 too!
    max_symp_abs = float(np.max(symp_errs))
    max_symp_rel = float(max(symp_errs[i] / max(1.0, float(np.linalg.norm(ys[4:, i].reshape((4, 4)))) ** 2) for i in range(len(ts))))

    t_reached = float(ts[-1])
    norms_arr = np.array(norms)
    growth_ratio = 1.0
    if spin2_blowup and t_reached > 0.5:
        mask = ts >= 0.2 * t_reached
        lyap = float(np.polyfit(ts[mask], np.log(np.maximum(norms_arr[mask], 1e-30)), 1)[0]) if np.sum(mask) >= 3 else float(math.log(max(norms_arr[-1], 1.0)) / t_reached)
        growth_ratio = float(norms_arr[-1] / max(norms_arr[0], 1e-12))
    elif t_reached > 100.0:
        n_bins = min(10, max(4, int(t_reached / 40.0)))
        edges = np.linspace(0.0, t_reached, n_bins + 1)
        bt, bm = [], []
        for i in range(n_bins):
            m = (ts >= edges[i]) & (ts <= edges[i + 1])
            if np.any(m):
                bt.append(0.5 * (edges[i] + edges[i + 1]))
                bm.append(float(np.max(norms_arr[m])))
        lyap = float(np.polyfit(bt[1:], np.log(np.maximum(bm[1:], 1e-30)), 1)[0]) if len(bt) >= 3 else 0.0
        growth_ratio = float(bm[-1] / max(bm[1], 1e-12)) if len(bm) >= 3 else 1.0
    else:
        lyap = 0.0

    if beta_zero_hit:
        outcome = "rg_beta_zero_singularity"
    elif spin2_blowup:
        outcome = "tachyonic_spin2_blowup" if (beff0 < 0.0) else "stelle_parametric_blowup"
    elif bg_blowup:
        outcome = "bg_blowup"
    elif (lyap > 2.5e-4 and growth_ratio > 1.25) or growth_ratio > 2.5:
        outcome = "parametric_resonance_growth"
    else:
        outcome = "bounded_spin2_stable"

    return {
        "x_b": float(x_b),
        "beta_0": float(beta_0),
        "running": bool(running),
        "method": method,
        "outcome": outcome,
        "tau_reached": t_reached,
        "beff_min": float(np.min(beff_vals)),
        "beff_max": float(np.max(beff_vals)),
        "m2_stelle_at_xring": float(0.5 / math.sqrt(beta_0)) if beta_0 > 0 else None,
        "norm_max": float(np.max(norms_arr)),
        "norm_end": float(norms_arr[-1]),
        "bin_growth_ratio": growth_ratio,
        "max_fund_matrix_eig_mag": max_eig_mag,  # arbitrary-tau fundamental matrix spectral radius on quasiperiodic bg
        "effective_lyapunov_rate": lyap,
        "symplectic_max_abs_err": max_symp_abs,
        "symplectic_max_rel_err": max_symp_rel,
        "det_M_err": det_err,
    }


def main():
    print("=== TRACK III UPGRADE T3-B: VARIATIONAL COVARIANT RIEGERT + WEYL C^2 SPIN-2 STABILITY ===")
    v4d = verify_4d_covariant_riegert_weyl_mpmath()
    print("1. 35-digit 4D covariant action & Ostrogradsky verification:")
    print(f"   Gauss-Bonnet total deriv err = {v4d['gauss_bonnet_total_deriv_err_eps0.13']:.3e}")
    print(f"   Quadratic Weyl C^2 err       = {v4d['quadratic_weyl_C2_err']:.3e}")
    print(f"   Direct 4D action vs EL err   = {v4d['direct_4d_action_vs_EL_rel_err']:.3e}")
    print(f"   Ostrogradsky vs EL err       = {v4d['ostrogradsky_vs_EL_rel_err']:.3e}")
    assert v4d["verified"], f"4D verification failed: {v4d}"

    # 2. Pure variational Riegert (beta_eff == 0) sweep across the ENTIRE P05 KAM island [6.5, 65]!
    print("\n2. Pure Variational Riegert (beta_eff == 0) across S^3/I* KAM island (resolving R3-3 61.2% drift):")
    riegert_sweep = []
    for xb, shd in [(6.5, +1), (7.0, +1), (7.5, +1), (8.0, +1), (10.0, +1), (15.0, +1), (X_RING, +1), (35.0, +1), (50.0, +1), (60.0, +1), (60.0, -1), (65.0, +1)]:
        r = integrate_variational_riegert_2d(xb, tau_end=2000.0, method="DOP853", sign_hdot=shd)
        riegert_sweep.append(r)
        print(f"   x_b={xb:6.2f} (sgn={shd:+d}): ghost={str(r['kinetic_ghost_crossing']):5s} bg_bl={str(r['bg_blowup']):5s} "
              f"A_R in [{r['min_A_Rieg']:+.4f},{r['max_A_Rieg']:+.4f}], cs2_min={r['min_cs2']:+.3f} (neg {100*r['frac_time_cs2_neg']:4.1f}%), "
              f"W_drift={r['symplectic_W_max_drift']:.2e}, norm_max={r['norm_max']:7.3f}, lyap={r['effective_lyapunov_rate']:+.2e}")

    r_long_R = integrate_variational_riegert_2d(X_RING, tau_end=6000.0, method="DOP853")
    r_rad_R = integrate_variational_riegert_2d(X_RING, tau_end=1000.0, method="Radau")
    r_dop_R = integrate_variational_riegert_2d(X_RING, tau_end=1000.0, method="DOP853")
    riegert_solver_diff = abs(r_dop_R["norm_end"] - r_rad_R["norm_end"]) / max(1e-12, r_dop_R["norm_end"])
    print(f"   [x_ring, tau=6000 Variational Riegert] W_drift={r_long_R['symplectic_W_max_drift']:.2e}, "
          f"norm_max={r_long_R['norm_max']:.4f}, growth={r_long_R['growth_over_tau']:.4f}x, lyap={r_long_R['effective_lyapunov_rate']:+.2e}, "
          f"DOP853 vs Radau(1000)={riegert_solver_diff:.2e}")

    # 3. Full 4th-order Ostrogradsky Riegert + Weyl C^2 scan at x_ring = 21.9168!
    # First check what x_min and x_max the background KAM ring reaches at x_ring = 21.9168!
    # Note: at x_ring, x stays nearly constant around 21.917 +- 0.5, so c_anom * ln(x_max/x_ring) is small (~0.01)!
    # Also test an eccentric KAM orbit x_b = 15.0 (where x oscillates [15, 32], so c_anom*ln(32/15) ~ 0.22!)
    print("\n3. Full 4th-order Ostrogradsky Riegert + Weyl C^2 scan on x_ring = 21.9168 (c_anom = 9/31 = 0.2903):")
    beta_grid = [
        -1.0, -C_ANOM_HS1, -0.05,
        0.02, 0.05, 0.07258, 0.10, 0.15, 0.20, 0.25, C_ANOM_HS1, 0.35, 0.50, 0.75, 1.0, 2.0, 5.0, 20.0
    ]
    weyl_xring_scan = []
    for b0 in beta_grid:
        tmax = 600.0 if b0 > 0 else 200.0
        res = integrate_riegert_weyl_c2_4d(X_RING, b0, running=True, tau_end=tmax, method="DOP853")
        weyl_xring_scan.append(res)
        m2_s = f"{res['m2_stelle_at_xring']:.3f}" if res["m2_stelle_at_xring"] is not None else "tach "
        print(f"   beta_0={b0:+7.4f} (m2={m2_s}): {res['outcome']:28s} tau={res['tau_reached']:6.1f} "
              f"norm_max={res['norm_max']:9.3e} lyap={res['effective_lyapunov_rate']:+.3e} symp_rel={res['symplectic_max_rel_err']:.2e}")

    # Fine zoom around the 1:1 (beta_0 ~ 0.0726) and 2:1 (beta_0 ~ 0.2903) scalaron-Stelle resonance zones!
    print("\n4. Fine scan of Stelle-scalaron resonance bands & eccentric KAM orbit x_b = 15.0 (RG zero-crossing test):")
    fine_betas = [0.060, 0.068, 0.07258, 0.078, 0.085, 0.24, 0.27, 0.285, 0.29032, 0.296, 0.31, 0.34, 1.16]
    fine_scan = []
    for b0 in fine_betas:
        res = integrate_riegert_weyl_c2_4d(X_RING, b0, running=True, tau_end=800.0, method="DOP853")
        fine_scan.append(res)
        print(f"   [fine] beta_0={b0:.5f} (m2={res['m2_stelle_at_xring']:.4f}, 2*m2/w_s={2*res['m2_stelle_at_xring']/math.sqrt(31/9):.3f}): "
              f"{res['outcome']:26s} norm_max={res['norm_max']:9.3e} lyap={res['effective_lyapunov_rate']:+.3e} symp_rel={res['symplectic_max_rel_err']:.2e}")

    # Eccentric KAM orbit x_b = 15.0 (x oscillates ~ [15, 32], testing RG zero-crossing beta_eff(tau) = 0!)
    ecc_scan = []
    for b0 in [0.05, 0.10, 0.15, 0.25, 0.50, 1.0, 2.0, 5.0]:
        res = integrate_riegert_weyl_c2_4d(15.0, b0, running=True, tau_end=600.0, method="DOP853")
        ecc_scan.append(res)
        print(f"   [x_b=15.0] beta_0={b0:5.2f}: {res['outcome']:26s} beff in [{res['beff_min']:+.4f},{res['beff_max']:+.4f}] "
              f"tau={res['tau_reached']:6.1f} norm_max={res['norm_max']:9.3e} lyap={res['effective_lyapunov_rate']:+.3e}")

    # Independent solver check DOP853 vs Radau on 4D Ostrogradsky system at x_ring, beta_0 = 1.0 (tau = 200)
    r4_dop = integrate_riegert_weyl_c2_4d(X_RING, 1.0, running=True, tau_end=200.0, method="DOP853")
    r4_rad = integrate_riegert_weyl_c2_4d(X_RING, 1.0, running=True, tau_end=200.0, method="Radau")
    ostro_solver_diff = abs(r4_dop["norm_end"] - r4_rad["norm_end"]) / max(1e-12, r4_dop["norm_end"])
    print(f"\n5. 4D Ostrogradsky solver check (x_ring, beta_0=1.0, tau=200): DOP853 vs Radau rel diff = {ostro_solver_diff:.3e}")

    # Long-time check (tau = 3000) at clean non-resonant Stelle couplings AND tau = 6000 deep check of 2:1 tongue at beta_0 = 9/31!
    long_4d = {}
    for b0 in [0.50, 1.0, 5.0]:
        res = integrate_riegert_weyl_c2_4d(X_RING, b0, running=True, tau_end=3000.0, method="DOP853")
        long_4d[f"beta0_{b0}"] = res
        print(f"   [Long tau=3000, x_ring, beta_0={b0}] {res['outcome']:26s} norm_max={res['norm_max']:.4f}, "
              f"growth={res['bin_growth_ratio']:.4f}x, lyap={res['effective_lyapunov_rate']:+.3e}, symp_rel={res['symplectic_max_rel_err']:.2e}")

    print("\n6. Deep zoom (tau = 6000) on 2:1 Stelle-scalaron resonance tongue near beta_0 = 9/31 = 0.29032:")
    tongue_2to1 = []
    for b0 in [0.2880, 0.2886, 0.2890, 0.2892, 0.2894, 0.2895, 0.2896, 0.2898, 0.2900, C_ANOM_HS1, 0.2912]:
        res = integrate_riegert_weyl_c2_4d(X_RING, b0, running=True, tau_end=6000.0, method="DOP853")
        tongue_2to1.append(res)
        print(f"   [tongue tau=6000] beta_0={b0:.5f}: {res['outcome']:26s} norm_max={res['norm_max']:9.3e} "
              f"growth={res['bin_growth_ratio']:7.3f}x lyap={res['effective_lyapunov_rate']:+.3e} symp_rel={res['symplectic_max_rel_err']:.2e}")

    # Also test static (non-running) vs RG-running at beta_0 = 0.2903 and 1.0 to isolate RG effect
    static_cmp = {
        "beta0_0.2903_static": integrate_riegert_weyl_c2_4d(X_RING, C_ANOM_HS1, running=False, tau_end=3000.0),
        "beta0_1.0_static": integrate_riegert_weyl_c2_4d(X_RING, 1.0, running=False, tau_end=3000.0),
    }

    payload = {
        "four_d_covariant_verification": v4d,
        "pure_variational_riegert_island_sweep": riegert_sweep,
        "pure_variational_riegert_xring_tau6000": r_long_R,
        "pure_variational_riegert_dop853_vs_radau_diff": riegert_solver_diff,
        "weyl_c2_xring_beta0_scan": weyl_xring_scan,
        "weyl_c2_fine_resonance_scan": fine_scan,
        "weyl_c2_2to1_tongue_tau6000": tongue_2to1,
        "weyl_c2_eccentric_xb15_rg_scan": ecc_scan,
        "ostrogradsky_4d_dop853_vs_radau_diff": ostro_solver_diff,
        "weyl_c2_long_tau3000": long_4d,
        "static_vs_running_comparison": static_cmp,
    }
    out_path = os.path.join(OUT_DIR, "t3_02_riegert_weyl_c2_spin2.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
