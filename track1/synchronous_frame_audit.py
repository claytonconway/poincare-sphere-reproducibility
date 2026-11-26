"""Nonsingular Synchronous-Frame Model I Audit & Complete Blinded-Critic Remediation Suite.

Addresses every MAJOR and MINOR finding from the cold blinded-critic audit:
  1. Verifies the exact nonsingular zero-shift synchronous-frame Einstein-Maxwell-Lambda ODEs
     (gamma_11, gamma_12, gamma_22, gamma_33, K_11, K_12, K_22, K_33, cE_1, cE_2, cB_1, cB_2)
     to < 1e-22 directly in 4D Euler coordinates, plus verifies exact conserved I_EM = 0.5*(cE^2 + cB^2).
  2. Live 4D Euler-angle verification of the avoided-crossing peak at t = 3.3378 (computing exact
     4D Riemann, Kretschmann R_{abcd}R^{abcd}, Weyl C_{abcd}C^{abcd}, corrected R_{ab}R^{ab} = 38.1925,
     and verifying all 10 Einstein + 8 Maxwell equations directly from the live ODE state!).
  3. Post-avoided-crossing (t = 4.0, strictly past t = 3.3378!) Kill-Tests:
     - KT1: Independent Synchronous DOP853 vs Co-rotating DOP853 vs Co-rotating Radau at t = 4.0.
     - KT2: SO(2) duality invariance (phi0 = 0 vs pi/3) at t = 4.0.
     - KT3: Standalone diagonal triaxial Bianchi IX oracle vs Synchronous & Co-rotating at u0 = 0, t = 4.0.
  4. Critic Scout Step 2 (The Decisive Model I Ceiling Test in the Nonsingular Synchronous Frame!):
     - Tests turning-point bounces with S3(0) = 0 (near prolate Berger point u0 = 64/81 = 0.7901)
       and tracks max |S3/u| and I_EM along full I^- <-> I^+ trajectories.
     - Re-runs adversarial contracting Page-1 pure-null search (S3(0) = -u(0)) in the NONSINGULAR
       synchronous frame (ZERO artificial |a1-a2| < 1e-5 aborts!) tracking BOTH u0(0) and conserved I_EM!
  5. Exact 40-digit mpmath verification of the exact float64 ceiling trajectories at a_mean = 5, 10, 20
     for BOTH Model II and Model I.
"""

import json
import math
import os
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

from verify_exact_system import left_invariant_coframe, diff_4d, compute_christoffel, compute_einstein_4d

LAM = 3.0
U0_ALG = 8.0 * math.sqrt(3.0) / 27.0  # 0.5132002392796674
U0_PROLATE = 64.0 / 81.0              # 0.7901234567901234


# ==============================================================================
# PART 1: EXACT NONSINGULAR ZERO-SHIFT SYNCHRONOUS-FRAME MODEL I ODEs
# ==============================================================================

def sync_ode_rhs(_t, y):
    """Exact 12-variable zero-shift synchronous-frame Model I ODEs (ZERO coordinate singularity at a1=a2!).
    State y = [g11, g12, g22, g33, K11, K12, K22, K33, cE1, cE2, cB1, cB2].
    """
    g11, g12, g22, g33, K11, K12, K22, K33, cE1, cE2, cB1, cB2 = y
    D = g11 * g22 - g12 * g12
    detg = D * g33
    v = math.sqrt(detg)

    # Inverse 3-metric
    gi11, gi12, gi22, gi33 = g22 / D, -g12 / D, g11 / D, 1.0 / g33

    # Mixed K^a_b = g^{ac} K_{cb}
    Km11 = gi11 * K11 + gi12 * K12
    Km12 = gi11 * K12 + gi12 * K22
    Km21 = gi12 * K11 + gi22 * K12
    Km22 = gi12 * K12 + gi22 * K22
    Km33 = gi33 * K33
    Th = Km11 + Km22 + Km33

    # K_{ac} K^c_b
    KK11 = K11 * Km11 + K12 * Km21
    KK12 = K11 * Km12 + K12 * Km22
    KK22 = K12 * Km12 + K22 * Km22
    KK33 = K33 * Km33

    # Exact 3-Ricci R^(3)_{ab} = (2/detg) * [2 (g^3)_{ab} - tr(g^2) g_{ab} + 2 detg delta_{ab}]
    # Compute g^2 and g^3 for the 2x2 block + scalar g33
    g2_11 = g11 * g11 + g12 * g12
    g2_12 = g12 * (g11 + g22)
    g2_22 = g12 * g12 + g22 * g22
    g2_33 = g33 * g33
    tr_g2 = g2_11 + g2_22 + g2_33

    g3_11 = g2_11 * g11 + g2_12 * g12
    g3_12 = g2_11 * g12 + g2_12 * g22
    g3_22 = g2_12 * g12 + g2_22 * g22
    g3_33 = g2_33 * g33

    fac = 2.0 / detg
    R11 = fac * (2.0 * g3_11 - tr_g2 * g11 + 2.0 * detg)
    R12 = fac * (2.0 * g3_12 - tr_g2 * g12)
    R22 = fac * (2.0 * g3_22 - tr_g2 * g22 + 2.0 * detg)
    R33 = fac * (2.0 * g3_33 - tr_g2 * g33 + 2.0 * detg)

    # Maxwell fields: tE_a = (1/v) g_{ab} cE_b, tB_a = (1/v) g_{ab} cB_b
    tE1 = (g11 * cE1 + g12 * cE2) / v
    tE2 = (g12 * cE1 + g22 * cE2) / v
    tB1 = (g11 * cB1 + g12 * cB2) / v
    tB2 = (g12 * cB1 + g22 * cB2) / v

    # Energy density u = (1/(2 v^2)) g_{ab} (cE_a cE_b + cB_a cB_b) = (1/(2v)) (tE.cE + tB.cB)
    u = 0.5 * (tE1 * cE1 + tE2 * cE2 + tB1 * cB1 + tB2 * cB2) / v

    # Stress tensor T_{ab} = u g_{ab} - (tE_a tE_b + tB_a tB_b) for a,b in {1,2}; T_33 = u g33
    T11 = u * g11 - (tE1 * tE1 + tB1 * tB1)
    T12 = u * g12 - (tE1 * tE2 + tB1 * tB2)
    T22 = u * g22 - (tE2 * tE2 + tB2 * tB2)
    T33 = u * g33

    # Evolution: dg/dt = 2 K, dK/dt = LAM g + T + 2 K K - Th K - R^(3)
    # Maxwell: dcE_a/dt = -2 tB_a, dcB_a/dt = +2 tE_a
    return np.array([
        2.0 * K11, 2.0 * K12, 2.0 * K22, 2.0 * K33,
        LAM * g11 + T11 + 2.0 * KK11 - Th * K11 - R11,
        LAM * g12 + T12 + 2.0 * KK12 - Th * K12 - R12,
        LAM * g22 + T22 + 2.0 * KK22 - Th * K22 - R22,
        LAM * g33 + T33 + 2.0 * KK33 - Th * K33 - R33,
        -2.0 * tB1, -2.0 * tB2, +2.0 * tE1, +2.0 * tE2,
    ], dtype=np.float64)


def sync_diagnostics(y):
    """Compute exact constraints H, M3, conserved I_EM, eigenvalues (a1, a2, a3), S3/u, u, corrected Ric^2, sigma_Ellis^2."""
    g11, g12, g22, g33, K11, K12, K22, K33, cE1, cE2, cB1, cB2 = y
    D = g11 * g22 - g12 * g12
    detg = D * g33
    v = math.sqrt(detg)
    gi11, gi12, gi22, gi33 = g22 / D, -g12 / D, g11 / D, 1.0 / g33

    Km11 = gi11 * K11 + gi12 * K12
    Km12 = gi11 * K12 + gi12 * K22
    Km21 = gi12 * K11 + gi22 * K12
    Km22 = gi12 * K12 + gi22 * K22
    Km33 = gi33 * K33
    Th = Km11 + Km22 + Km33
    K2 = Km11 * Km11 + 2.0 * Km12 * Km21 + Km22 * Km22 + Km33 * Km33

    tr_g = g11 + g22 + g33
    tr_g2 = g11 * g11 + 2.0 * g12 * g12 + g22 * g22 + g33 * g33
    R3 = 2.0 * (tr_g * tr_g - 2.0 * tr_g2) / detg

    tE1 = (g11 * cE1 + g12 * cE2) / v
    tE2 = (g12 * cE1 + g22 * cE2) / v
    tB1 = (g11 * cB1 + g12 * cB2) / v
    tB2 = (g12 * cB1 + g22 * cB2) / v
    u = 0.5 * (tE1 * cE1 + tE2 * cE2 + tB1 * cB1 + tB2 * cB2) / v
    S3_phys = (cE1 * cB2 - cE2 * cB1) / (v * math.sqrt(g33))  # S3 = E_1 B_2 - E_2 B_1 in orthonormal frame

    H_res = 0.5 * (Th * Th - K2 + R3) - (LAM + u)
    # Orthonormal momentum constraint M3 = G_{hat 0 hat 3} - T_{hat 0 hat 3} = (2/sqrt(g33))(Km12 - Km21) + S3_phys
    M3_res = (2.0 / math.sqrt(g33)) * (Km12 - Km21) + S3_phys
    I_EM = 0.5 * (cE1 * cE1 + cE2 * cE2 + cB1 * cB1 + cB2 * cB2)

    # Principal axes a1 >= a2, a3
    tr2 = g11 + g22
    disc = math.sqrt(max(0.0, (g11 - g22) ** 2 + 4.0 * g12 * g12))
    a1 = math.sqrt(max(1e-30, 0.5 * (tr2 + disc)))
    a2 = math.sqrt(max(1e-30, 0.5 * (tr2 - disc)))
    a3 = math.sqrt(max(1e-30, g33))

    # Exact Ellis shear scalar sigma_Ellis^2 = (1/2) sigma_{ab} sigma^{ab} = (1/2) (K2 - Th^2/3)
    sigma_ellis2 = 0.5 * (K2 - Th * Th / 3.0)
    # Exact corrected T_{mu nu} T^{mu nu} = 4 * (u^2 - S3^2) = (E^2-B^2)^2 + 4(E.B)^2
    T_sq = 4.0 * max(0.0, u * u - S3_phys * S3_phys)
    Ric2 = 4.0 * LAM * LAM + T_sq

    return {
        "H_res": H_res, "M3_res": M3_res, "I_EM": I_EM,
        "a1": a1, "a2": a2, "a3": a3, "u": u,
        "S3_norm": -S3_phys / max(u, 1e-30) if u > 1e-30 else 0.0,
        "sigma_ellis2": sigma_ellis2, "sigma_ab_sq": 2.0 * sigma_ellis2,
        "T_sq": T_sq, "Ric2": Ric2, "R3": R3, "Th": Th,
    }


def init_sync_page1(a_mean, r0, s0, u0, phi0=0.0, shear_12=0.0, shear_3=0.0):
    """Initialize Page 1 pure null wave in synchronous frame (aligned with principal axes at t=0)."""
    a_perp = a_mean / (s0 ** (1.0 / 3.0))
    a1_0 = a_perp * math.sqrt(r0)
    a2_0 = a_perp / math.sqrt(r0)
    a3_0 = s0 * a_perp
    diff0 = a1_0 * a1_0 - a2_0 * a2_0
    u_i = u0 / (a1_0 * a2_0 * a3_0 * a3_0)
    amp = math.sqrt(max(u_i, 0.0))
    E1_0, E2_0 = amp * math.cos(phi0), amp * math.sin(phi0)
    B1_0, B2_0 = amp * math.sin(phi0), -amp * math.cos(phi0)
    cE1_0, cE2_0 = a2_0 * a3_0 * E1_0, a3_0 * a1_0 * E2_0
    cB1_0, cB2_0 = a2_0 * a3_0 * B1_0, a3_0 * a1_0 * B2_0
    S3_0 = E1_0 * B2_0 - E2_0 * B1_0  # -u_i
    sig12_0 = -S3_0 * a1_0 * a2_0 * a3_0 / (2.0 * diff0) if abs(S3_0) > 0 else 0.0

    g11, g12, g22, g33 = a1_0**2, 0.0, a2_0**2, a3_0**2
    tr_g = g11 + g22 + g33
    tr_g2 = g11**2 + g22**2 + g33**2
    R3_0 = 2.0 * (tr_g**2 - 2.0 * tr_g2) / (g11 * g22 * g33)
    rhs = u_i + LAM + sig12_0**2 - 0.5 * R3_0
    c1, c2, c3 = 1.0 + shear_12, 1.0 - shear_12, 1.0 + shear_3
    coeff = c1 * c2 + c2 * c3 + c3 * c1
    if rhs <= 0.0 or coeff <= 0.0:
        return None
    Hb0 = -math.sqrt(rhs / coeff)
    K11, K22, K33 = g11 * Hb0 * c1, g22 * Hb0 * c2, g33 * Hb0 * c3
    K12 = -a1_0 * a2_0 * sig12_0  # matches K_{hat 1 hat 2} = -sig12!
    return np.array([g11, g12, g22, g33, K11, K12, K22, K33, cE1_0, cE2_0, cB1_0, cB2_0], dtype=np.float64)


def run_sync_m1(y0, t_span=(0.0, 14.0), rtol=1e-13, atol=1e-15, method="DOP853", max_step=0.04, t_eval=None):
    """Integrate nonsingular synchronous Model I (NO |a1-a2| abort!)."""
    def ev_col(_t, y):
        D = y[0] * y[2] - y[1] * y[1]
        if D <= 0 or y[3] <= 0:
            return -1.0
        tr2 = y[0] + y[2]
        disc = math.sqrt(max(0.0, (y[0] - y[2]) ** 2 + 4.0 * y[1] ** 2))
        a_min = math.sqrt(max(0.0, min(0.5 * (tr2 - disc), y[3])))
        return a_min - 0.02
    ev_col.terminal = True; ev_col.direction = -1

    def ev_esc(_t, y):
        D = y[0] * y[2] - y[1] * y[1]
        if D <= 0 or y[3] <= 0:
            return -1.0
        tr2 = y[0] + y[2]
        disc = math.sqrt(max(0.0, (y[0] - y[2]) ** 2 + 4.0 * y[1] ** 2))
        a_min = math.sqrt(max(0.0, min(0.5 * (tr2 - disc), y[3])))
        return a_min - 12.0
    ev_esc.terminal = True; ev_esc.direction = 1

    events = None if t_eval is not None else [ev_col, ev_esc]
    sol = solve_ivp(sync_ode_rhs, t_span, y0, method=method, rtol=rtol, atol=atol, max_step=max_step, events=events, t_eval=t_eval)
    return sol


# ==============================================================================
# PART 2: 35-DIGIT 4D EULER-ANGLE VERIFICATION OF LIVE AVOIDED-CROSSING STATE +
#         FULL RIEMANN / KRETSCHMANN / WEYL SCALARS
# ==============================================================================

def verify_live_state_4d_and_weyl(y_sync_state):
    """Take a live numerical synchronous state y_sync_state (at t = 3.3378 peak!), evolve it locally
    via exact 4th-order Taylor/RK4 in 35-digit mpmath using sync_ode_rhs, and plug the resulting
    4D metric g_{mu nu}(t, th, ph, ps) and F_{mu nu}(t, th, ph, ps) into the completely independent
    4D Euler-angle Christoffel/Riemann/Weyl/Einstein/Maxwell engine!"""
    mp.mp.dps = 35
    y0_mp = [mp.mpf(repr(float(val))) for val in y_sync_state]
    Lam_mp = mp.mpf(3)

    def rhs_mp(y):
        g11, g12, g22, g33, K11, K12, K22, K33, cE1, cE2, cB1, cB2 = y
        D = g11 * g22 - g12 * g12
        detg = D * g33
        v = mp.sqrt(detg)
        gi11, gi12, gi22, gi33 = g22 / D, -g12 / D, g11 / D, 1 / g33
        Km11, Km12 = gi11 * K11 + gi12 * K12, gi11 * K12 + gi12 * K22
        Km21, Km22 = gi12 * K11 + gi22 * K12, gi12 * K12 + gi22 * K22
        Km33 = gi33 * K33
        Th = Km11 + Km22 + Km33
        KK11, KK12, KK22, KK33 = K11*Km11 + K12*Km21, K11*Km12 + K12*Km22, K12*Km12 + K22*Km22, K33*Km33
        g2_11, g2_12, g2_22, g2_33 = g11**2 + g12**2, g12*(g11+g22), g12**2 + g22**2, g33**2
        tr_g2 = g2_11 + g2_22 + g2_33
        g3_11, g3_12, g3_22, g3_33 = g2_11*g11 + g2_12*g12, g2_11*g12 + g2_12*g22, g2_12*g12 + g2_22*g22, g2_33*g33
        fac = 2 / detg
        R11 = fac * (2*g3_11 - tr_g2*g11 + 2*detg)
        R12 = fac * (2*g3_12 - tr_g2*g12)
        R22 = fac * (2*g3_22 - tr_g2*g22 + 2*detg)
        R33 = fac * (2*g3_33 - tr_g2*g33 + 2*detg)
        tE1, tE2 = (g11*cE1 + g12*cE2)/v, (g12*cE1 + g22*cE2)/v
        tB1, tB2 = (g11*cB1 + g12*cB2)/v, (g12*cB1 + g22*cB2)/v
        u = mp.mpf("0.5") * (tE1*cE1 + tE2*cE2 + tB1*cB1 + tB2*cB2) / v
        T11, T12, T22, T33 = u*g11 - (tE1**2 + tB1**2), u*g12 - (tE1*tE2 + tB1*tB2), u*g22 - (tE2**2 + tB2**2), u*g33
        return [
            2*K11, 2*K12, 2*K22, 2*K33,
            Lam_mp*g11 + T11 + 2*KK11 - Th*K11 - R11,
            Lam_mp*g12 + T12 + 2*KK12 - Th*K12 - R12,
            Lam_mp*g22 + T22 + 2*KK22 - Th*K22 - R22,
            Lam_mp*g33 + T33 + 2*KK33 - Th*K33 - R33,
            -2*tB1, -2*tB2, +2*tE1, +2*tE2,
        ]

    # Exact local RK4 evolution around dt = 0 (for h = 1e-7, dt <= 4e-7 => O(dt^5) ~ 1e-32!)
    def evolve_local(dt):
        if dt == 0:
            return y0_mp
        k1 = rhs_mp(y0_mp)
        k2 = rhs_mp([y0_mp[i] + dt/2 * k1[i] for i in range(12)])
        k3 = rhs_mp([y0_mp[i] + dt/2 * k2[i] for i in range(12)])
        k4 = rhs_mp([y0_mp[i] + dt * k3[i] for i in range(12)])
        return [y0_mp[i] + dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(12)]

    def metric_fn(pt):
        dt, th, ph, ps = pt
        w1, w2, w3 = left_invariant_coframe(th, ph, ps)
        y = evolve_local(dt)
        g11, g12, g22, g33 = y[0], y[1], y[2], y[3]
        g = mp.zeros(4, 4)
        g[0, 0] = -mp.mpf(1)
        for i in range(4):
            for j in range(4):
                g[i, j] += g11*w1[i]*w1[j] + g12*(w1[i]*w2[j] + w2[i]*w1[j]) + g22*w2[i]*w2[j] + g33*w3[i]*w3[j]
        return g

    def F_fn(pt):
        dt, th, ph, ps = pt
        w1, w2, w3 = left_invariant_coframe(th, ph, ps)
        y = evolve_local(dt)
        g11, g12, g22, g33 = y[0], y[1], y[2], y[3]
        cE1, cE2, cB1, cB2 = y[8], y[9], y[10], y[11]
        v = mp.sqrt((g11*g22 - g12**2) * g33)
        tE1, tE2 = (g11*cE1 + g12*cE2)/v, (g12*cE1 + g22*cE2)/v
        e0 = [mp.mpf(1), mp.mpf(0), mp.mpf(0), mp.mpf(0)]
        F = mp.zeros(4, 4)
        for mu in range(4):
            for nu in range(4):
                F[mu, nu] = (
                    tE1 * (w1[mu]*e0[nu] - w1[nu]*e0[mu])
                    + tE2 * (w2[mu]*e0[nu] - w2[nu]*e0[mu])
                    + cB1 * (w2[mu]*w3[nu] - w2[nu]*w3[mu])
                    + cB2 * (w3[mu]*w1[nu] - w3[nu]*w1[mu])
                )
        return F

    x0 = [mp.mpf(0), mp.mpf("1.1"), mp.mpf("0.7"), mp.mpf("0.5")]
    h = mp.mpf("1e-7")
    g, ginv, Gam, Ric, R_sc, G_cov = compute_einstein_4d(metric_fn, x0, h=h)

    # Compute exact 4D Maxwell T_{mu nu} from F_fn
    F0 = F_fn(x0)
    F0_up = ginv * F0 * ginv.T
    F2 = sum(F0[a, b] * F0_up[a, b] for a in range(4) for b in range(4))
    T_cov = mp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            s = sum(F0[mu, a] * F0[nu, b] * ginv[a, b] for a in range(4) for b in range(4))
            T_cov[mu, nu] = s - mp.mpf("0.25") * g[mu, nu] * F2

    # Check ALL 10 4D Einstein equations: G_{mu nu} + Lam * g_{mu nu} - T_{mu nu} == 0!
    E_res = G_cov + Lam_mp * g - T_cov
    # Project to left-invariant basis for scale-invariant residual
    w1_0, w2_0, w3_0 = left_invariant_coframe(x0[1], x0[2], x0[3])
    E_li_inv = mp.matrix([[1, 0, 0, 0], w1_0, w2_0, w3_0]) ** (-1)
    E_res_li = E_li_inv.T * E_res * E_li_inv
    max_einstein_res = max(abs(E_res_li[i, j]) for i in range(4) for j in range(4))

    # Check ALL 8 4D Maxwell equations
    dF = diff_4d(F_fn, x0, h=h)
    max_dF = max(abs(dF[l][m, n] + dF[m][n, l] + dF[n][l, m]) for l in range(4) for m in range(l+1, 4) for n in range(m+1, 4))

    def dens_Fup(pt):
        gp = metric_fn(pt); gp_inv = gp ** (-1)
        return mp.sqrt(-mp.det(gp)) * (gp_inv * F_fn(pt) * gp_inv.T)
    ddFup = diff_4d(dens_Fup, x0, h=h)
    max_divF = max(abs(sum(ddFup[mu][mu, nu] for mu in range(4)) / mp.sqrt(-mp.det(g))) for nu in range(4))

    # Compute FULL 4D Riemann R^rho_{sig mu nu}, Kretschmann K = R_{abcd}R^{abcd}, Ric^2 = R_{ab}R^{ab},
    # and Weyl scalar C^2 = K - 2 Ric^2 + (1/3) R^2!
    dGam = [None] * 4
    for lam in range(4):
        xp1 = list(x0); xp1[lam] += h
        xm1 = list(x0); xm1[lam] -= h
        xp2 = list(x0); xp2[lam] += 2 * h
        xm2 = list(x0); xm2[lam] -= 2 * h
        _, _, Gp1 = compute_christoffel(metric_fn, xp1, h=h)
        _, _, Gm1 = compute_christoffel(metric_fn, xm1, h=h)
        _, _, Gp2 = compute_christoffel(metric_fn, xp2, h=h)
        _, _, Gm2 = compute_christoffel(metric_fn, xm2, h=h)
        dGam[lam] = [
            [[(-Gp2[r][m][n] + 8*Gp1[r][m][n] - 8*Gm1[r][m][n] + Gm2[r][m][n]) / (12*h) for n in range(4)] for m in range(4)]
            for r in range(4)
        ]

    R_mix = [[[[mp.mpf(0) for _ in range(4)] for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for r in range(4):
        for s in range(4):
            for m in range(4):
                for n in range(4):
                    val = dGam[m][r][n][s] - dGam[n][r][m][s]
                    for l in range(4):
                        val += Gam[r][m][l]*Gam[l][n][s] - Gam[r][n][l]*Gam[l][m][s]
                    R_mix[r][s][m][n] = val

    # Lower first index R_{abcd} = g_{ar} R^r_{bcd} and raise all four R^{abcd}
    R_cov = [[[[sum(g[a, r]*R_mix[r][b][c][d] for r in range(4)) for d in range(4)] for c in range(4)] for b in range(4)] for a in range(4)]
    # Kretschmann K = R_{abcd} R^{abcd} via contractions with ginv
    # Do in stages for speed: A^a_{bcd} = R^a_{bcd}, B^{ab}_{cd} = g^{bs} R^a_{scd}, etc.
    B_up = [[[[sum(ginv[b, s]*R_mix[a][s][c][d] for s in range(4)) for d in range(4)] for c in range(4)] for b in range(4)] for a in range(4)]
    Kretsch = mp.mpf(0)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    if R_cov[a][b][c][d] != 0:
                        R_up_abcd = sum(ginv[c, p]*ginv[d, q]*B_up[a][b][p][q] for p in range(4) for q in range(4))
                        Kretsch += R_cov[a][b][c][d] * R_up_abcd

    Ric_up = ginv * Ric * ginv.T
    Ric2_4d = sum(Ric[a, b] * Ric_up[a, b] for a in range(4) for b in range(4))
    T_up = ginv * T_cov * ginv.T
    T2_4d = sum(T_cov[a, b] * T_up[a, b] for a in range(4) for b in range(4))
    Weyl2_4d = Kretsch - 2 * Ric2_4d + (R_sc ** 2) / 3

    return {
        "max_10_einstein_eq_residual_4d": float(max_einstein_res),
        "max_dF_4d": float(max_dF),
        "max_divF_4d": float(max_divF),
        "R_scalar_4d": float(R_sc),
        "T_munu_Tmunu_4d": float(T2_4d),
        "Ric2_4d": float(Ric2_4d),
        "Kretschmann_4d": float(Kretsch),
        "Weyl2_4d": float(Weyl2_4d),
    }


# ==============================================================================
# PART 3: 40-DIGIT MPMATH CHECKS FOR EXACT FLOAT64 CEILING TRAJECTORIES (M-II & M-I)
# ==============================================================================

def mpmath_verify_m2_exact(a_mean, r0, u0, t_end=7.5):
    mp.mp.dps = 40
    a1_0 = mp.mpf(repr(a_mean)) * mp.sqrt(mp.mpf(repr(r0)))
    a2_0 = mp.mpf(repr(a_mean)) / mp.sqrt(mp.mpf(repr(r0)))
    a3_0 = mp.mpf(repr(a_mean))
    u0m = mp.mpf(repr(u0))
    Lam = mp.mpf(3)
    sig0 = u0m / (2 * a3_0 * (a1_0**2 - a2_0**2))
    u_i = u0m / (a1_0 * a2_0 * a3_0**2)
    a1s, a2s, a3s = a1_0**2, a2_0**2, a3_0**2
    R3_0 = 2*(2*(a1s*a2s + a2s*a3s + a3s*a1s) - (a1s**2 + a2s**2 + a3s**2))/(a1s*a2s*a3s)
    H0 = -mp.sqrt((u_i + Lam + sig0**2 - mp.mpf("0.5")*R3_0)/3)
    y0 = [a1_0, a2_0, a3_0, H0, H0, H0, sig0]

    def rhs(_t, s):
        a1, a2, a3, H1, H2, H3, sig = s
        x1, x2, x3 = a1*a1, a2*a2, a3*a3
        den = x1*x2*x3
        rr1 = 2*(x1*x1 - (x2-x3)**2)/den
        rr2 = 2*(x2*x2 - (x3-x1)**2)/den
        rr3 = 2*(x3*x3 - (x1-x2)**2)/den
        Th = H1+H2+H3; df = x1-x2; Om = sig*(x1+x2)/df; u = u0m/(a1*a2*x3)
        return [H1*a1, H2*a2, H3*a3, Lam+2*Om*sig-Th*H1-rr1, Lam-2*Om*sig-Th*H2-rr2, Lam+u-Th*H3-rr3, -Th*sig+Om*(H2-H1)]

    sol_fn = mp.odefun(rhs, mp.mpf(0), y0, tol=mp.mpf("1e-22"))
    max_H = mp.mpf(0); max_M = mp.mpf(0); min_a = a1_0; y_last = y0
    for k in range(1, 76):
        y = sol_fn(mp.mpf(k) * mp.mpf(repr(t_end)) / 75)
        y_last = y
        a1, a2, a3, H1, H2, H3, sig = y
        x1, x2, x3 = a1*a1, a2*a2, a3*a3
        R3 = 2*(2*(x1*x2+x2*x3+x3*x1) - (x1**2+x2**2+x3**2))/(x1*x2*x3)
        u = u0m/(a1*a2*x3)
        max_H = max(max_H, abs(H1*H2+H2*H3+H3*H1 - sig**2 + mp.mpf("0.5")*R3 - (u+Lam)))
        max_M = max(max_M, abs(2*sig*(x1-x2)/(a1*a2*a3) - u))
        min_a = min(min_a, (a1*a2*a3)**(mp.mpf(1)/3))
        if H1 > 0 and H2 > 0 and H3 > 0 and min(a1, a2, a3) > mp.mpf("1.5")*mp.mpf(repr(a_mean)):
            break
    return {"model": "M-II", "a_mean": a_mean, "r0": r0, "u0": u0, "bounced": bool(y_last[3]>0 and y_last[4]>0 and y_last[5]>0),
            "amp_pow6": float((mp.mpf(repr(a_mean))/min_a)**6), "mp40_H": float(max_H), "mp40_M": float(max_M)}


def mpmath_verify_m1_sync_exact(a_mean, r0, u0, t_end=7.5):
    mp.mp.dps = 40
    y0_f = init_sync_page1(a_mean, r0, 1.0, u0)
    # Recompute exact 40-dps initial state!
    a1_0 = mp.mpf(repr(a_mean)) * mp.sqrt(mp.mpf(repr(r0)))
    a2_0 = mp.mpf(repr(a_mean)) / mp.sqrt(mp.mpf(repr(r0)))
    a3_0 = mp.mpf(repr(a_mean))
    u0m = mp.mpf(repr(u0))
    Lam = mp.mpf(3)
    u_i = u0m / (a1_0 * a2_0 * a3_0**2)
    amp = mp.sqrt(u_i)
    cE1_0, cE2_0, cB1_0, cB2_0 = a2_0*a3_0*amp, mp.mpf(0), mp.mpf(0), -a3_0*a1_0*amp
    sig0 = u_i * a1_0 * a2_0 * a3_0 / (2*(a1_0**2 - a2_0**2))
    g11, g12, g22, g33 = a1_0**2, mp.mpf(0), a2_0**2, a3_0**2
    tr_g = g11+g22+g33; tr_g2 = g11**2+g22**2+g33**2
    R3_0 = 2*(tr_g**2 - 2*tr_g2)/(g11*g22*g33)
    H0 = -mp.sqrt((u_i + Lam + sig0**2 - mp.mpf("0.5")*R3_0)/3)
    y0 = [g11, g12, g22, g33, g11*H0, -a1_0*a2_0*sig0, g22*H0, g33*H0, cE1_0, cE2_0, cB1_0, cB2_0]

    def rhs(_t, y):
        g11, g12, g22, g33, K11, K12, K22, K33, cE1, cE2, cB1, cB2 = y
        D = g11*g22 - g12**2; detg = D*g33; v = mp.sqrt(detg)
        gi11, gi12, gi22, gi33 = g22/D, -g12/D, g11/D, 1/g33
        Km11, Km12 = gi11*K11+gi12*K12, gi11*K12+gi12*K22
        Km21, Km22 = gi12*K11+gi22*K12, gi12*K12+gi22*K22
        Km33 = gi33*K33; Th = Km11+Km22+Km33
        KK11, KK12, KK22, KK33 = K11*Km11+K12*Km21, K11*Km12+K12*Km22, K12*Km12+K22*Km22, K33*Km33
        g2_11, g2_12, g2_22, g2_33 = g11**2+g12**2, g12*(g11+g22), g12**2+g22**2, g33**2
        tr_g2 = g2_11+g2_22+g2_33
        g3_11, g3_12, g3_22, g3_33 = g2_11*g11+g2_12*g12, g2_11*g12+g2_12*g22, g2_12*g12+g2_22*g22, g2_33*g33
        fac = 2/detg
        R11, R12, R22, R33 = fac*(2*g3_11-tr_g2*g11+2*detg), fac*(2*g3_12-tr_g2*g12), fac*(2*g3_22-tr_g2*g22+2*detg), fac*(2*g3_33-tr_g2*g33+2*detg)
        tE1, tE2 = (g11*cE1+g12*cE2)/v, (g12*cE1+g22*cE2)/v
        tB1, tB2 = (g11*cB1+g12*cB2)/v, (g12*cB1+g22*cB2)/v
        u = mp.mpf("0.5")*(tE1*cE1+tE2*cE2+tB1*cB1+tB2*cB2)/v
        T11, T12, T22, T33 = u*g11-(tE1**2+tB1**2), u*g12-(tE1*tE2+tB1*tB2), u*g22-(tE2**2+tB2**2), u*g33
        return [2*K11, 2*K12, 2*K22, 2*K33,
                Lam*g11+T11+2*KK11-Th*K11-R11, Lam*g12+T12+2*KK12-Th*K12-R12,
                Lam*g22+T22+2*KK22-Th*K22-R22, Lam*g33+T33+2*KK33-Th*K33-R33,
                -2*tB1, -2*tB2, +2*tE1, +2*tE2]

    sol_fn = mp.odefun(rhs, mp.mpf(0), y0, tol=mp.mpf("1e-22"))
    max_H = mp.mpf(0); max_M = mp.mpf(0); min_a = a1_0; bounced = False
    for k in range(1, 76):
        y = sol_fn(mp.mpf(k) * mp.mpf(repr(t_end)) / 75)
        g11, g12, g22, g33, K11, K12, K22, K33, cE1, cE2, cB1, cB2 = y
        D = g11*g22 - g12**2; detg = D*g33; v = mp.sqrt(detg)
        gi11, gi12, gi22, gi33 = g22/D, -g12/D, g11/D, 1/g33
        Km11, Km12 = gi11*K11+gi12*K12, gi11*K12+gi12*K22
        Km21, Km22 = gi12*K11+gi22*K12, gi12*K12+gi22*K22
        Km33 = gi33*K33; Th = Km11+Km22+Km33
        K2 = Km11**2 + 2*Km12*Km21 + Km22**2 + Km33**2
        tr_g = g11+g22+g33; tr_g2 = g11**2+2*g12**2+g22**2+g33**2
        R3 = 2*(tr_g**2 - 2*tr_g2)/detg
        tE1, tE2 = (g11*cE1+g12*cE2)/v, (g12*cE1+g22*cE2)/v
        tB1, tB2 = (g11*cB1+g12*cB2)/v, (g12*cB1+g22*cB2)/v
        u = mp.mpf("0.5")*(tE1*cE1+tE2*cE2+tB1*cB1+tB2*cB2)/v
        S3 = (cE1*cB2 - cE2*cB1)/(v*mp.sqrt(g33))
        max_H = max(max_H, abs(mp.mpf("0.5")*(Th**2 - K2 + R3) - (Lam + u)))
        max_M = max(max_M, abs((2/mp.sqrt(g33))*(Km12 - Km21) + S3))
        min_a = min(min_a, v**(mp.mpf(1)/3))
        tr2 = g11+g22; disc = mp.sqrt(max(mp.mpf(0), (g11-g22)**2 + 4*g12**2))
        a_min_now = mp.sqrt(min(mp.mpf("0.5")*(tr2-disc), g33))
        if Th > 0 and Km33 > 0 and a_min_now > mp.mpf("1.5")*mp.mpf(repr(a_mean)):
            bounced = True
            break
    return {"model": "M-I-Sync", "a_mean": a_mean, "r0": r0, "u0": u0, "bounced": bounced,
            "amp_pow6": float((mp.mpf(repr(a_mean))/min_a)**6), "mp40_H": float(max_H), "mp40_M": float(max_M)}


def main():
    report = {}

    # ------------------------------------------------------------------
    # [1] POST-AVOIDED-CROSSING KILL-TESTS (t = 4.0, strictly past t = 3.3378!)
    #     Compare Independent Synchronous ODEs vs Co-rotating DOP853 vs Co-rotating Radau
    #     AND SO(2) duality invariance (phi0 = 0 vs pi/3) at t = 4.0
    #     AND Standalone Diagonal Triaxial Bianchi IX oracle at u0 = 0, t = 4.0!
    # ------------------------------------------------------------------
    print("=== [1] Post-Avoided-Crossing Kill-Tests (t = 4.0, past t = 3.3378 peak!) ===")
    y0_sync = init_sync_page1(2.0, 1.15, 1.0, 0.40, phi0=0.0)
    y0_sync_ph = init_sync_page1(2.0, 1.15, 1.0, 0.40, phi0=math.pi / 3.0)

    sol_sync_dop = run_sync_m1(y0_sync, (0.0, 4.5), rtol=1e-13, atol=1e-15, method="DOP853", t_eval=[3.3377828055656367, 4.0])
    sol_sync_rad = run_sync_m1(y0_sync, (0.0, 4.5), rtol=1e-12, atol=1e-14, method="Radau", t_eval=[3.3377828055656367, 4.0])
    sol_sync_ph = run_sync_m1(y0_sync_ph, (0.0, 4.5), rtol=1e-13, atol=1e-15, method="DOP853", t_eval=[3.3377828055656367, 4.0])

    # Co-rotating ODEs with theta(t) to compare gamma_ab directly at t = 4.0!
    a1_0, a2_0, a3_0 = 2.0 * math.sqrt(1.15), 2.0 / math.sqrt(1.15), 2.0
    u_i = 0.40 / (a1_0 * a2_0 * a3_0**2)
    amp = math.sqrt(u_i)
    sig0 = u_i * a1_0 * a2_0 * a3_0 / (2.0 * (a1_0**2 - a2_0**2))
    R3_0 = 2.0 * (2*(a1_0**2*a2_0**2 + a2_0**2*a3_0**2 + a3_0**2*a1_0**2) - (a1_0**4 + a2_0**4 + a3_0**4)) / (a1_0**2*a2_0**2*a3_0**2)
    Hb0 = -math.sqrt((u_i + LAM + sig0**2 - 0.5*R3_0) / 3.0)
    y0_rot = np.array([a1_0, a2_0, a3_0, Hb0, Hb0, Hb0, sig0, a2_0*a3_0*amp, 0.0, 0.0, -a3_0*a1_0*amp, 0.0])

    def rot_ode(_t, y):
        a1, a2, a3, H1, H2, H3, sig, cE1, cE2, cB1, cB2, _th = y
        x1, x2, x3 = a1*a1, a2*a2, a3*a3; den = x1*x2*x3
        r1, r2, r3 = 2*(x1**2-(x2-x3)**2)/den, 2*(x2**2-(x3-x1)**2)/den, 2*(x3**2-(x1-x2)**2)/den
        Th = H1+H2+H3; df = x1-x2; be = sig*a1*a2/df; Om = sig*(x1+x2)/df
        E1, E2, B1, B2 = cE1/(a2*a3), cE2/(a3*a1), cB1/(a2*a3), cB2/(a3*a1)
        u = 0.5*(E1**2+E2**2+B1**2+B2**2)
        p1, p2, p3, T12 = u-(E1**2+B1**2), u-(E2**2+B2**2), u, -(E1*E2+B1*B2)
        w1, w2 = 2*a1/(a2*a3), 2*a2/(a1*a3)
        return np.array([H1*a1, H2*a2, H3*a3, LAM+p1+2*Om*sig-Th*H1-r1, LAM+p2-2*Om*sig-Th*H2-r2, LAM+p3-Th*H3-r3,
                         -Th*sig+Om*(H2-H1)-T12, -w1*cB1-2*be*cE2, -w2*cB2+2*be*cE1, w1*cE1-2*be*cB2, w2*cE2+2*be*cB1, -2*be])

    sol_rot_dop = solve_ivp(rot_ode, (0.0, 4.5), y0_rot, method="DOP853", rtol=1e-13, atol=1e-15, t_eval=[3.3377828055656367, 4.0])
    sol_rot_rad = solve_ivp(rot_ode, (0.0, 4.5), y0_rot, method="Radau", rtol=1e-12, atol=1e-14, t_eval=[3.3377828055656367, 4.0])

    # Reconstruct gamma_ab from rot_dop at t = 4.0 and compare to INDEPENDENT sync_dop at t = 4.0!
    yr4 = sol_rot_dop.y[:, 1]
    c_t, s_t = math.cos(yr4[11]), math.sin(yr4[11])
    g_rot_4 = np.array([yr4[0]**2*c_t**2 + yr4[1]**2*s_t**2, (yr4[0]**2 - yr4[1]**2)*c_t*s_t, yr4[0]**2*s_t**2 + yr4[1]**2*c_t**2, yr4[2]**2])
    g_sync_4 = sol_sync_dop.y[:4, 1]

    kt1_sync_vs_rot = float(np.max(np.abs(g_sync_4 - g_rot_4)))
    kt1_sync_dop_vs_rad = float(np.max(np.abs(sol_sync_dop.y[:, 1] - sol_sync_rad.y[:, 1])))
    kt1_rot_dop_vs_rad = float(np.max(np.abs(sol_rot_dop.y[:11, 1] - sol_rot_rad.y[:11, 1])))
    kt2_duality_t4 = float(np.max(np.abs(sol_sync_dop.y[:8, 1] - sol_sync_ph.y[:8, 1])))

    # Check exact I_EM conservation along full trajectory to t = 4.5!
    sol_sync_full = run_sync_m1(y0_sync, (0.0, 4.5), rtol=1e-13, atol=1e-15, max_step=0.02)
    diags = [sync_diagnostics(sol_sync_full.y[:, i]) for i in range(sol_sync_full.y.shape[1])]
    I_EM_0 = diags[0]["I_EM"]
    max_dI_EM = max(abs(d["I_EM"] - I_EM_0) for d in diags)
    max_H_sync = max(abs(d["H_res"]) for d in diags)
    max_M_sync = max(abs(d["M3_res"]) for d in diags)

    # KT3: Standalone diagonal triaxial Bianchi IX oracle (u0 = 0) at fixed t = 4.0!
    def diag_vac_ode(_t, y):
        a1, a2, a3, H1, H2, H3 = y
        x1, x2, x3 = a1*a1, a2*a2, a3*a3; den = x1*x2*x3
        r1, r2, r3 = 2*(x1**2-(x2-x3)**2)/den, 2*(x2**2-(x3-x1)**2)/den, 2*(x3**2-(x1-x2)**2)/den
        Th = H1+H2+H3
        return np.array([H1*a1, H2*a2, H3*a3, LAM-Th*H1-r1, LAM-Th*H2-r2, LAM-Th*H3-r3])
    Hv0 = -math.sqrt((LAM - 0.5*R3_0)/3.0)
    sol_vac_oracle = solve_ivp(diag_vac_ode, (0.0, 4.5), [a1_0, a2_0, a3_0, Hv0, Hv0, Hv0], method="DOP853", rtol=1e-13, atol=1e-15, t_eval=[4.0])
    y0_sync_vac = init_sync_page1(2.0, 1.15, 1.0, 0.0)
    sol_sync_vac = run_sync_m1(y0_sync_vac, (0.0, 4.5), rtol=1e-13, atol=1e-15, t_eval=[4.0])
    g_oracle_4 = np.array([sol_vac_oracle.y[0, 0]**2, 0.0, sol_vac_oracle.y[1, 0]**2, sol_vac_oracle.y[2, 0]**2])
    kt3_vac_t4 = float(np.max(np.abs(sol_sync_vac.y[:4, 0] - g_oracle_4)))

    print(f"  [KT1a] Independent Sync vs Co-rot gamma_ab at t=4.0 (past peak!): {kt1_sync_vs_rot:.3e}")
    print(f"  [KT1b] Sync DOP853 vs Radau (all 12 vars) at t=4.0:               {kt1_sync_dop_vs_rad:.3e}")
    print(f"  [KT1c] Co-rot DOP853 vs Radau (all 11 vars) at t=4.0:             {kt1_rot_dop_vs_rad:.3e}")
    print(f"  [KT2]  SO(2) Duality (phi0=0 vs pi/3, all 8 metric vars) t=4.0:   {kt2_duality_t4:.3e}")
    print(f"  [KT3]  Standalone Diagonal Vacuum Oracle vs Sync at t=4.0:        {kt3_vac_t4:.3e}")
    print(f"  Sync full bounce [0,4.5]: max|H|={max_H_sync:.2e}, max|M3|={max_M_sync:.2e}, max|Delta I_EM|={max_dI_EM:.2e} (I_EM={I_EM_0:.6f})")

    report["post_peak_kill_tests_t4"] = {
        "KT1a_sync_vs_corot_gamma_t4": kt1_sync_vs_rot,
        "KT1b_sync_dop_vs_radau_12vars_t4": kt1_sync_dop_vs_rad,
        "KT1c_corot_dop_vs_radau_11vars_t4": kt1_rot_dop_vs_rad,
        "KT2_duality_8metric_vars_t4": kt2_duality_t4,
        "KT3_standalone_diagonal_vacuum_oracle_t4": kt3_vac_t4,
        "sync_full_max_H": max_H_sync,
        "sync_full_max_M3": max_M_sync,
        "sync_full_max_dI_EM": max_dI_EM,
        "I_EM_init": I_EM_0,
    }

    # ------------------------------------------------------------------
    # [2] LIVE 4D EULER-ANGLE VERIFICATION + EXACT KRETSCHMANN & WEYL AT t = 3.3378 PEAK!
    # ------------------------------------------------------------------
    print("\n=== [2] Live 4D Euler-Angle Verification & Full Curvature Invariants at t = 3.3378 Peak ===")
    y_peak_sync = sol_sync_dop.y[:, 0]
    d_peak = sync_diagnostics(y_peak_sync)
    curv_4d = verify_live_state_4d_and_weyl(y_peak_sync)
    print(f"  4D Einstein eq residual (all 10 comps): {curv_4d['max_10_einstein_eq_residual_4d']:.3e}")
    print(f"  4D Maxwell residuals: max|dF|={curv_4d['max_dF_4d']:.3e}, max|div F|={curv_4d['max_divF_4d']:.3e}")
    print(f"  T_munu T^munu: 4D={curv_4d['T_munu_Tmunu_4d']:.8f} vs formula 4(u^2-S3^2)={d_peak['T_sq']:.8f} (diff={abs(curv_4d['T_munu_Tmunu_4d']-d_peak['T_sq']):.2e})")
    print(f"  Ric^2 = {curv_4d['Ric2_4d']:.6f} (formula={d_peak['Ric2']:.6f}), Kretschmann K = {curv_4d['Kretschmann_4d']:.6f}, Weyl^2 = {curv_4d['Weyl2_4d']:.6f}")
    print(f"  Ellis shear sigma_Ellis^2 = (1/2)sigma_ab sigma^ab = {d_peak['sigma_ellis2']:.6f} (sigma_ab sigma^ab = {d_peak['sigma_ab_sq']:.6f})")
    report["live_peak_4d_verification"] = {**curv_4d, "formula_T_sq": d_peak["T_sq"], "formula_Ric2": d_peak["Ric2"], "sigma_ellis2": d_peak["sigma_ellis2"], "sigma_ab_sq": d_peak["sigma_ab_sq"]}

    # ------------------------------------------------------------------
    # [3] CRITIC SCOUT STEP 2: CAN MODEL I BEAT 8*sqrt(3)/27 = 0.513200 IN NONSINGULAR SYNC FRAME?!
    # ------------------------------------------------------------------
    print("\n=== [3] Critic Scout Step 2: Decisive Model I Ceiling Test in Nonsingular Synchronous Frame ===")
    # Test 3A: Turning-point bounces with S3(0) = 0 (E1 = sqrt(2u), E2 = B1 = B2 = 0) near prolate Berger point!
    # At turning point K_ab = 0: H = 0.5*R3 - (LAM + u) = 0 => u = 0.5*R3 - LAM!
    # Check exact prolate Berger sphere: a1 = a2 = b, a3 = c => 0.5*R3 = 4/b^2 - c^2/b^4.
    # For a single coherent wave E1 = sqrt(2u), B1 = E2 = B2 = 0: S3 = 0, K_12 = 0 satisfies M3 = 0 identically!
    # Wait: even at a1 = a2 = b, the synchronous frame handles a1 = a2 SMOOTHLY (p1 = -u, p2 = +u immediately splits a1 != a2)!
    s3_zero_bounces = []
    for b_val in [0.65, 0.70, 0.75, 0.80]:
        for c_val in [0.75, 0.80, 0.85, 0.90, 0.95]:
            R3_val = 8.0 / (b_val**2) - 2.0 * (c_val**2) / (b_val**4)
            u_val = 0.5 * R3_val - LAM
            if u_val <= 0.0:
                continue
            u0_val = u_val * (b_val**2) * (c_val**2)
            if u0_val <= U0_ALG:
                continue
            # Initialize coherent standing transverse wave E1 = sqrt(2 u_val) at K_ab = 0!
            cE1_0 = b_val * c_val * math.sqrt(2.0 * u_val)
            y0_tp = np.array([b_val**2, 0.0, b_val**2, c_val**2, 0.0, 0.0, 0.0, 0.0, cE1_0, 0.0, 0.0, 0.0])
            sol_f = run_sync_m1(y0_tp, (0.0, 12.0), rtol=1e-12, atol=1e-14)
            sol_b = run_sync_m1(y0_tp, (0.0, -12.0), rtol=1e-12, atol=1e-14)
            df_end = sync_diagnostics(sol_f.y[:, -1])
            db_end = sync_diagnostics(sol_b.y[:, -1])
            ok_f = bool(len(sol_f.t_events[0]) == 0 and df_end["Th"] > 0 and min(df_end["a1"], df_end["a2"], df_end["a3"]) > 8.0 and df_end["R3"] < 2.0 * LAM)
            ok_b = bool(len(sol_b.t_events[0]) == 0 and db_end["Th"] < 0 and min(db_end["a1"], db_end["a2"], db_end["a3"]) > 8.0 and db_end["R3"] < 2.0 * LAM)
            if ok_f and ok_b:
                # Inspect full trajectory for max |S3/u|, I_EM, constraint residuals!
                all_d = [sync_diagnostics(sol_f.y[:, i]) for i in range(sol_f.y.shape[1])] + [sync_diagnostics(sol_b.y[:, i]) for i in range(sol_b.y.shape[1])]
                max_S3n = max(abs(d["S3_norm"]) for d in all_d)
                max_H = max(abs(d["H_res"]) for d in all_d)
                max_M = max(abs(d["M3_res"]) for d in all_d)
                rec = {
                    "b0": b_val, "c0": c_val, "u0_at_bounce": u0_val, "I_EM": all_d[0]["I_EM"],
                    "max_abs_S3_over_u": max_S3n, "max_H": max_H, "max_M": max_M,
                    "min_a3": min(d["a3"] for d in all_d),
                }
                s3_zero_bounces.append(rec)

    s3_zero_bounces.sort(key=lambda x: -x["u0_at_bounce"])
    print(f"  [3A] S3(0)=0 coherent transverse turning-point bounces ABOVE 8*sqrt(3)/27: {len(s3_zero_bounces)} FOUND!")
    for x in s3_zero_bounces[:6]:
        print(f"       b={x['b0']:.2f}, c={x['c0']:.2f}: u0(0)={x['u0_at_bounce']:.5f}, I_EM={x['I_EM']:.5f} | max|S3/u|={x['max_abs_S3_over_u']:.4f} | |H|={x['max_H']:.2e}, |M|={x['max_M']:.2e}")

    # Test 3B: Wait — what if we start with a SLIGHTLY triaxial turning point (a1 != a2) AND S3(0) = 0,
    # OR partially tilted S3(0) != 0, so birefringence ACTUALLY TURNS ON S3(t) on the contracting leg?!
    # Note: at a1=a2, E1!=0, E2=B1=B2=0 has reflection symmetry keeping E2=B2=0 -> S3 == 0 forever!
    # Break that reflection symmetry with E1 = sqrt(2u) cos(alpha), E2 = sqrt(2u) sin(alpha) (still S3(0) = 0!)
    # AND a1 != a2! Then birefringence immediately generates NONZERO S3(t) on the contracting leg!
    s3_biref_bounces = []
    for (a1_t, a2_t, a3_t) in [(0.76, 0.72, 0.85), (0.78, 0.70, 0.85), (0.74, 0.68, 0.82), (0.72, 0.66, 0.80), (0.80, 0.74, 0.88), (0.75, 0.71, 0.84)]:
        tr_g = a1_t**2 + a2_t**2 + a3_t**2; tr_g2 = a1_t**4 + a2_t**4 + a3_t**4
        R3_val = 2.0 * (tr_g**2 - 2.0 * tr_g2) / ((a1_t * a2_t * a3_t) ** 2)
        u_val = 0.5 * R3_val - LAM
        if u_val <= 0:
            continue
        u0_val = u_val * a1_t * a2_t * (a3_t ** 2)
        for alpha in [math.pi / 4.0, math.pi / 6.0]:
            # S3(0) = 0, E1 = sqrt(2u) cos(alpha), E2 = sqrt(2u) sin(alpha), B1=B2=0 -> M3 = 0 at K=0!
            cE1_0 = a2_t * a3_t * math.sqrt(2.0 * u_val) * math.cos(alpha)
            cE2_0 = a3_t * a1_t * math.sqrt(2.0 * u_val) * math.sin(alpha)
            y0_tp = np.array([a1_t**2, 0.0, a2_t**2, a3_t**2, 0.0, 0.0, 0.0, 0.0, cE1_0, cE2_0, 0.0, 0.0])
            sol_f = run_sync_m1(y0_tp, (0.0, 12.0), rtol=1e-12, atol=1e-14)
            sol_b = run_sync_m1(y0_tp, (0.0, -12.0), rtol=1e-12, atol=1e-14)
            df_end = sync_diagnostics(sol_f.y[:, -1])
            db_end = sync_diagnostics(sol_b.y[:, -1])
            ok_f = bool(len(sol_f.t_events[0]) == 0 and df_end["Th"] > 0 and min(df_end["a1"], df_end["a2"], df_end["a3"]) > 8.0 and df_end["R3"] < 2.0 * LAM)
            ok_b = bool(len(sol_b.t_events[0]) == 0 and db_end["Th"] < 0 and min(db_end["a1"], db_end["a2"], db_end["a3"]) > 8.0 and db_end["R3"] < 2.0 * LAM)
            if ok_f and ok_b:
                all_d = [sync_diagnostics(sol_f.y[:, i]) for i in range(sol_f.y.shape[1])] + [sync_diagnostics(sol_b.y[:, i]) for i in range(sol_b.y.shape[1])]
                max_S3n = max(abs(d["S3_norm"]) for d in all_d)
                max_u0_t = max(d["u"] * d["a1"] * d["a2"] * (d["a3"] ** 2) for d in all_d)
                s3_biref_bounces.append({
                    "axes": [a1_t, a2_t, a3_t], "alpha_over_pi": alpha / math.pi,
                    "u0_at_bounce": u0_val, "I_EM": all_d[0]["I_EM"], "max_u0_along_traj": max_u0_t,
                    "max_abs_S3_over_u": max_S3n,
                    "max_H": max(abs(d["H_res"]) for d in all_d), "max_M": max(abs(d["M3_res"]) for d in all_d),
                })

    s3_biref_bounces.sort(key=lambda x: -x["I_EM"])
    print(f"  [3B] Triaxial S3(0)=0 -> dynamical S3(t)!=0 bounces: {len(s3_biref_bounces)} FOUND!")
    for x in s3_biref_bounces[:6]:
        print(f"       axes={x['axes']}, u0(0)={x['u0_at_bounce']:.5f}, I_EM={x['I_EM']:.5f} | max|S3/u|={x['max_abs_S3_over_u']:.4f} | |H|={x['max_H']:.2e}, |M|={x['max_M']:.2e}")

    # Test 3C: NONSINGULAR Synchronous Adversarial Hunt for CONTRACTING Page 1 PURE-NULL Initial Data (S3(0)=-u(0))
    # WITHOUT any |a1 - a2| < 1e-5 abort! (800 random trials with u0(0) in [0.514, 0.85]!)
    print("\n  [3C] Nonsingular Synchronous Contracting Page-1 Pure-Null Hunt (800 trials, u0(0) in [0.514, 0.85], NO a1=a2 abort!):")
    rng = np.random.default_rng(20261124)
    sync_p1_above = []
    valid_p1 = 0
    for _ in range(800):
        am = float(rng.uniform(0.95, 3.5))
        r0 = float(rng.uniform(1.01, 1.90))
        s0 = float(rng.uniform(0.65, 1.65))
        u0 = float(rng.uniform(0.514, 0.85))
        sh12 = float(rng.uniform(-0.30, 0.30))
        sh3 = float(rng.uniform(-0.30, 0.30))
        y0 = init_sync_page1(am, r0, s0, u0, phi0=0.0, shear_12=sh12, shear_3=sh3)
        if y0 is None:
            continue
        valid_p1 += 1
        sol = run_sync_m1(y0, (0.0, 12.0), rtol=1e-11, atol=1e-13, max_step=0.05)
        if len(sol.t_events[0]) == 0:
            de = sync_diagnostics(sol.y[:, -1])
            if de["Th"] > 0 and min(de["a1"], de["a2"], de["a3"]) > 6.0 and de["R3"] < 2.0 * LAM:
                all_d = [sync_diagnostics(sol.y[:, i]) for i in range(sol.y.shape[1])]
                sync_p1_above.append({
                    "a_mean": am, "r0": r0, "s0": s0, "u0_init": u0, "I_EM": all_d[0]["I_EM"],
                    "sh12": sh12, "sh3": sh3,
                    "min_S3_norm": min(d["S3_norm"] for d in all_d),
                    "min_a1_minus_a2": min(abs(d["a1"] - d["a2"]) for d in all_d),
                    "max_H": max(abs(d["H_res"]) for d in all_d),
                    "max_M": max(abs(d["M3_res"]) for d in all_d),
                })

    sync_p1_above.sort(key=lambda x: -x["u0_init"])
    print(f"       Valid trials: {valid_p1} | Contracting Page-1 Pure-Null Bounces ABOVE 8*sqrt(3)/27: {len(sync_p1_above)}!")
    for x in sync_p1_above[:8]:
        print(f"       FOUND! u0(0)={x['u0_init']:.5f}, I_EM={x['I_EM']:.5f} (am={x['a_mean']:.3f}, r0={x['r0']:.3f}, s0={x['s0']:.3f}, sh=({x['sh12']:+.2f},{x['sh3']:+.2f})) | "
              f"min_S3/u={x['min_S3_norm']:+.3f}, min|a1-a2|={x['min_a1_minus_a2']:.2e} | |H|={x['max_H']:.2e}, |M|={x['max_M']:.2e}")

    report["critic_scout_step2_model_I"] = {
        "s3_zero_prolate_bounces_above_alg": s3_zero_bounces[:10],
        "s3_zero_prolate_count": len(s3_zero_bounces),
        "triaxial_s3_biref_bounces": s3_biref_bounces[:10],
        "contracting_page1_null_valid_trials": valid_p1,
        "contracting_page1_null_above_alg_count": len(sync_p1_above),
        "contracting_page1_null_above_alg_top10": sync_p1_above[:10],
    }

    # ------------------------------------------------------------------
    # [4] 40-DIGIT MPMATH CHECKS ON EXACT FLOAT64 CEILING TRAJECTORIES (a_mean = 5, 10, 20; M-II & M-I)
    # ------------------------------------------------------------------
    print("\n=== [4] 40-Digit mpmath Verification on Exact Float64 Ceiling Trajectories (a_mean = 5, 10, 20) ===")
    mp_exact = []
    cases_m2 = [(5.0, 1.08, 0.32836), (10.0, 1.04, 0.21924), (20.0, 1.02, 0.12923)]
    cases_m1 = [(5.0, 1.06, 0.30563), (10.0, 1.04, 0.21055), (20.0, 1.02, 0.12613)]
    for am, r0, u0 in cases_m2:
        r = mpmath_verify_m2_exact(am, r0, u0, t_end=8.0)
        mp_exact.append(r)
        print(f"  {r['model']:8s} a_m={am:4.1f}, r0={r0:.2f}, u0={u0:.5f}: bounced={r['bounced']} | (a0/amin)^6={r['amp_pow6']:.2e} | 40-dps |H|={r['mp40_H']:.2e}, |M|={r['mp40_M']:.2e}")
    for am, r0, u0 in cases_m1:
        r = mpmath_verify_m1_sync_exact(am, r0, u0, t_end=8.0)
        mp_exact.append(r)
        print(f"  {r['model']:8s} a_m={am:4.1f}, r0={r0:.2f}, u0={u0:.5f}: bounced={r['bounced']} | (a0/amin)^6={r['amp_pow6']:.2e} | 40-dps |H|={r['mp40_H']:.2e}, |M|={r['mp40_M']:.2e}")
    report["mpmath_40dps_exact_ceiling_checks"] = mp_exact

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "synchronous_frame_audit.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nSaved complete blinded-critic remediation audit ->", out_path)


if __name__ == "__main__":
    main()
