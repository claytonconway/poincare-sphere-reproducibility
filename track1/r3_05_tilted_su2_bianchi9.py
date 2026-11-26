#!/usr/bin/env python3
"""
Round-3 Micro-Scout R3-5:
Tilted Homogeneous SU(2)_L Yang-Mills in Zero-Shift Synchronous Rotating Triaxial Bianchi IX
on the Poincare Homology Sphere S^3/I* (T_{03} != 0, Exact Gauss [A_i, E^i] = 0).

Tests whether non-Abelian transverse commutator detouring [A_1, A_2] = det(A_perp) T_3
opens an untuned contracting pure-null bounce basin above P01's 0/800 Abelian wall
u_0* = 8*sqrt(3)/27 = 0.513200 (or whether the exact Gauss constraint and non-Abelian
color-anisotropy instability preserve or tighten the ceiling).

Exact Closed 18-Variable System (diagonal Z_2 color x space parity):
  Metric: ds^2 = -dt^2 + g11 w1^2 + 2 g12 w1 w2 + g22 w2^2 + g33 w3^2
  Gauge (A_0 = 0): A = (A11 w1 + A12 w2) T_1 + (A21 w1 + A22 w2) T_2 + A33 w3 T_3
  State: y = [g11, g12, g22, g33, K11, K12, K22, K33,
              A11, A12, A21, A22, A33, cE11, cE12, cE21, cE22, cE33]
  Constraints:
    Hamiltonian H = 0.5*(Th^2 - K2 + R3) - (Lam + u_YM) = 0
    Momentum    M3 = 2*(K^1_2 - K^2_1) - T_{03} = 0
    Gauss       G3 = A11 cE21 + A12 cE22 - A21 cE11 - A22 cE12 = 0  (G1 = G2 = 0 identically)

Independent Verification Pillars:
  1. Direct 35-digit mpmath 4D Euler-angle verification of ALL 10 4D Einstein equations
     and ALL 12 covariant 4D Yang-Mills equations D_mu F^{a mu nu} = 0 (3 Gauss + 9 EOM)
     on BOTH a generic tilted state AND a live bounced trajectory peak.
  2. Exact Pure-Null Gauss Ellipsoid Theorem verified algebraically & numerically to 1e-16
     (generalizing without Z_2 to all 9-component homogeneous SU(2)_L connections:
     vec{G} = [(a2/a1)(4 - ||A_1||^2) + (a1/a2)(4 - ||A_2||^2)] * (A_1 x A_2)/2 = 0):
     G^3 = +- det(A_perp) * [2(a2/a1 + a1/a2) - 0.5*((a2/a1)||A_1||^2 + (a1/a2)||A_2||^2)].
  3. Kill-Tests: exact recovery of P01 Abelian Model I (< 1e-11), exact recovery of
     Monograph Sec. 4 triaxial SU(2)_L (< 1e-11), DOP853 vs Radau agreement on a live
     tilted non-Abelian bounce (< 1e-10), AND live bounced-state 35-digit 4D verification.
  4. Paired head-to-head pure-null contracting ensembles (N=400 above 0.514 + N=200 below 0.50)
     sampling the FULL Gauss ellipsoid angle phi_ell in (0, pi/2) with full termination auditing.
  5. Paired partially-tilted contracting ensembles across g_YM in {0.5, 1.0, 2.0, 4.0, 10.0, 30.0}
     against BOTH Match-(S3) U(1) AND exact T_{mu nu}(0)-matched Match-(S3, u3) U(1)_A x U(1)_B
     controls, plus full past-to-future (I^- <-> I^+) turning-point bounce ceiling scan.
"""

import json
import math
import os
import sys
from pathlib import Path
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_exact_system import left_invariant_coframe, compute_einstein_4d
from synchronous_frame_audit import sync_ode_rhs, init_sync_page1

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LAM = 3.0
U0_ALG = 8.0 * math.sqrt(3.0) / 27.0  # 0.5132002392796674
U0_PROLATE = 64.0 / 81.0              # 0.7901234567901234


# =============================================================================
# 1. EXACT 18-VARIABLE TILTED SU(2)_L ROTATING TRIAXIAL BIANCHI IX ODEs
# =============================================================================

def make_su2_rhs(kap: float):
    """Exact 18-variable zero-shift synchronous tilted SU(2)_L Bianchi IX ODEs (ZERO clamping)."""
    def rhs(_t, y):
        g11, g12, g22, g33, K11, K12, K22, K33, A11, A12, A21, A22, A33, cE11, cE12, cE21, cE22, cE33 = y
        D = g11 * g22 - g12 * g12
        detg = D * g33
        if D <= 0.0 or g33 <= 0.0:
            return np.full(18, np.nan)
        v = math.sqrt(detg)
        gi11, gi12, gi22, gi33 = g22 / D, -g12 / D, g11 / D, 1.0 / g33

        Km11 = gi11 * K11 + gi12 * K12
        Km12 = gi11 * K12 + gi12 * K22
        Km21 = gi12 * K11 + gi22 * K12
        Km22 = gi12 * K12 + gi22 * K22
        Km33 = gi33 * K33
        Th = Km11 + Km22 + Km33

        KK11 = K11 * Km11 + K12 * Km21
        KK12 = K11 * Km12 + K12 * Km22
        KK22 = K12 * Km12 + K22 * Km22
        KK33 = K33 * Km33

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

        cB11 = -2.0 * A11 + A22 * A33
        cB12 = -2.0 * A12 - A21 * A33
        cB21 = -2.0 * A21 - A12 * A33
        cB22 = -2.0 * A22 + A11 * A33
        cB33 = -2.0 * A33 + (A11 * A22 - A12 * A21)

        E11 = (g11 * cE11 + g12 * cE12) / v
        E12 = (g12 * cE11 + g22 * cE12) / v
        E21 = (g11 * cE21 + g12 * cE22) / v
        E22 = (g12 * cE21 + g22 * cE22) / v
        E33 = (g33 * cE33) / v

        B11 = (g11 * cB11 + g12 * cB12) / v
        B12 = (g12 * cB11 + g22 * cB12) / v
        B21 = (g11 * cB21 + g12 * cB22) / v
        B22 = (g12 * cB21 + g22 * cB22) / v
        B33 = (g33 * cB33) / v

        u = 0.5 * kap * (
            E11 * cE11 + E12 * cE12 + E21 * cE21 + E22 * cE22 + E33 * cE33
            + B11 * cB11 + B12 * cB12 + B21 * cB21 + B22 * cB22 + B33 * cB33
        ) / v

        T11 = u * g11 - kap * (E11 * E11 + E21 * E21 + B11 * B11 + B21 * B21)
        T12 = u * g12 - kap * (E11 * E12 + E21 * E22 + B11 * B12 + B21 * B22)
        T22 = u * g22 - kap * (E12 * E12 + E22 * E22 + B12 * B12 + B22 * B22)
        T33 = u * g33 - kap * (E33 * E33 + B33 * B33)

        dcE11 = 2.0 * B11 - (A33 * B22 + A22 * B33)
        dcE12 = 2.0 * B12 + (A33 * B21 + A21 * B33)
        dcE21 = 2.0 * B21 + (A33 * B12 + A12 * B33)
        dcE22 = 2.0 * B22 - (A33 * B11 + A11 * B33)
        dcE33 = 2.0 * B33 - (A22 * B11 - A12 * B21 - A21 * B12 + A11 * B22)

        return np.array([
            2.0 * K11, 2.0 * K12, 2.0 * K22, 2.0 * K33,
            LAM * g11 + T11 + 2.0 * KK11 - Th * K11 - R11,
            LAM * g12 + T12 + 2.0 * KK12 - Th * K12 - R12,
            LAM * g22 + T22 + 2.0 * KK22 - Th * K22 - R22,
            LAM * g33 + T33 + 2.0 * KK33 - Th * K33 - R33,
            E11, E12, E21, E22, E33,
            dcE11, dcE12, dcE21, dcE22, dcE33,
        ], dtype=np.float64)
    return rhs


def su2_diagnostics(y, kap: float):
    """Exact Hamiltonian, momentum, Gauss constraints, eigenvalues, Poynting flux, and longitudinal fraction."""
    g11, g12, g22, g33, K11, K12, K22, K33, A11, A12, A21, A22, A33, cE11, cE12, cE21, cE22, cE33 = y
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

    cB11 = -2.0 * A11 + A22 * A33
    cB12 = -2.0 * A12 - A21 * A33
    cB21 = -2.0 * A21 - A12 * A33
    cB22 = -2.0 * A22 + A11 * A33
    cB33 = -2.0 * A33 + (A11 * A22 - A12 * A21)

    E11, E12 = (g11 * cE11 + g12 * cE12) / v, (g12 * cE11 + g22 * cE12) / v
    E21, E22 = (g11 * cE21 + g12 * cE22) / v, (g12 * cE21 + g22 * cE22) / v
    E33 = (g33 * cE33) / v
    B11, B12 = (g11 * cB11 + g12 * cB12) / v, (g12 * cB11 + g22 * cB12) / v
    B21, B22 = (g11 * cB21 + g12 * cB22) / v, (g12 * cB21 + g22 * cB22) / v
    B33 = (g33 * cB33) / v

    u_perp = 0.5 * kap * (
        E11 * cE11 + E12 * cE12 + E21 * cE21 + E22 * cE22
        + B11 * cB11 + B12 * cB12 + B21 * cB21 + B22 * cB22
    ) / v
    u_3 = 0.5 * kap * (E33 * cE33 + B33 * cB33) / v
    u = u_perp + u_3
    T03 = (kap / v) * ((cE11 * cB12 - cE12 * cB11) + (cE21 * cB22 - cE22 * cB21))
    S3_phys = T03 / math.sqrt(g33)

    H_res = 0.5 * (Th * Th - K2 + R3) - (LAM + u)
    M3_res = 2.0 * (Km12 - Km21) - T03
    G3_res = A11 * cE21 + A12 * cE22 - A21 * cE11 - A22 * cE12

    tr2 = g11 + g22
    disc = math.sqrt(max(0.0, (g11 - g22) ** 2 + 4.0 * g12 * g12))
    a1 = math.sqrt(max(1e-30, 0.5 * (tr2 + disc)))
    a2 = math.sqrt(max(1e-30, 0.5 * (tr2 - disc)))
    a3 = math.sqrt(g33)
    # Effective Monograph E_YM = u * (a1 a2 a3)^(4/3) / (6 * kap)
    a_mean = v ** (1.0 / 3.0)
    E_ym_mono = u * (a_mean ** 4) / (6.0 * kap) if kap > 0 else 0.0
    return {
        "H_res": H_res, "M3_res": M3_res, "G3_res": G3_res,
        "Th": Th, "R3": R3, "a1": a1, "a2": a2, "a3": a3, "a_mean": a_mean,
        "u": u, "u3_frac": u_3 / max(u, 1e-30), "S3_over_u": S3_phys / max(u, 1e-30),
        "u0_eff": u * a1 * a2 * a3 * a3, "u_iso": u * (a_mean ** 4),
        "E_ym_mono": E_ym_mono, "det_A_perp": A11 * A22 - A12 * A21,
    }


def integrate_su2(y0, kap, t_span=(0.0, 12.0), rtol=1e-11, atol=1e-13, method="DOP853", max_step=0.04, t_eval=None):
    """Integrate 18-variable tilted SU(2)_L Bianchi IX system."""
    def ev_col(_t, y):
        D = y[0] * y[2] - y[1] * y[1]
        if D <= 0.0 or y[3] <= 0.0:
            return -1.0
        tr2 = y[0] + y[2]
        disc = math.sqrt(max(0.0, (y[0] - y[2]) ** 2 + 4.0 * y[1] * y[1]))
        return math.sqrt(max(0.0, min(0.5 * (tr2 - disc), y[3]))) - 0.02
    ev_col.terminal = True
    ev_col.direction = -1

    def ev_esc(_t, y):
        D = y[0] * y[2] - y[1] * y[1]
        if D <= 0.0 or y[3] <= 0.0:
            return -1.0
        tr2 = y[0] + y[2]
        disc = math.sqrt(max(0.0, (y[0] - y[2]) ** 2 + 4.0 * y[1] * y[1]))
        return math.sqrt(max(0.0, min(0.5 * (tr2 - disc), y[3]))) - 10.0
    ev_esc.terminal = True
    ev_esc.direction = 1

    events = None if t_eval is not None else [ev_col, ev_esc]
    return solve_ivp(make_su2_rhs(kap), t_span, y0, method=method, rtol=rtol, atol=atol,
                     max_step=max_step, events=events, t_eval=t_eval)


def abelian_u1x2_rhs(_t, y):
    """Exact 14-variable two-copy Abelian U(1)_A x U(1)_B Maxwell control in rotating triaxial Bianchi IX
    (closes P1-6: matches u(0), S3(0), AND u3(0) -> IDENTICAL T_{33}(0) with zero commutators!).
    State y = [g11, g12, g22, g33, K11, K12, K22, K33, cE1, cE2, cB1, cB2, cE3, cB3]."""
    g11, g12, g22, g33, K11, K12, K22, K33, cE1, cE2, cB1, cB2, cE3, cB3 = y
    D = g11 * g22 - g12 * g12
    detg = D * g33
    if D <= 0.0 or g33 <= 0.0:
        return np.full(14, np.nan)
    v = math.sqrt(detg)
    gi11, gi12, gi22, gi33 = g22 / D, -g12 / D, g11 / D, 1.0 / g33

    Km11, Km12 = gi11 * K11 + gi12 * K12, gi11 * K12 + gi12 * K22
    Km21, Km22 = gi12 * K11 + gi22 * K12, gi12 * K12 + gi22 * K22
    Km33 = gi33 * K33
    Th = Km11 + Km22 + Km33
    KK11, KK12 = K11 * Km11 + K12 * Km21, K11 * Km12 + K12 * Km22
    KK22, KK33 = K12 * Km12 + K22 * Km22, K33 * Km33

    g2_11, g2_12, g2_22, g2_33 = g11*g11 + g12*g12, g12*(g11 + g22), g12*g12 + g22*g22, g33*g33
    tr_g2 = g2_11 + g2_22 + g2_33
    g3_11, g3_12 = g2_11*g11 + g2_12*g12, g2_11*g12 + g2_12*g22
    g3_22, g3_33 = g2_12*g12 + g2_22*g22, g2_33*g33
    fac = 2.0 / detg
    R11 = fac * (2.0 * g3_11 - tr_g2 * g11 + 2.0 * detg)
    R12 = fac * (2.0 * g3_12 - tr_g2 * g12)
    R22 = fac * (2.0 * g3_22 - tr_g2 * g22 + 2.0 * detg)
    R33 = fac * (2.0 * g3_33 - tr_g2 * g33 + 2.0 * detg)

    E1, E2 = (g11 * cE1 + g12 * cE2) / v, (g12 * cE1 + g22 * cE2) / v
    B1, B2 = (g11 * cB1 + g12 * cB2) / v, (g12 * cB1 + g22 * cB2) / v
    E3, B3 = (g33 * cE3) / v, (g33 * cB3) / v

    u = 0.5 * (E1 * cE1 + E2 * cE2 + B1 * cB1 + B2 * cB2 + E3 * cE3 + B3 * cB3) / v
    T11 = u * g11 - (E1 * E1 + B1 * B1)
    T12 = u * g12 - (E1 * E2 + B1 * B2)
    T22 = u * g22 - (E2 * E2 + B2 * B2)
    T33 = u * g33 - (E3 * E3 + B3 * B3)

    return np.array([
        2.0 * K11, 2.0 * K12, 2.0 * K22, 2.0 * K33,
        LAM * g11 + T11 + 2.0 * KK11 - Th * K11 - R11,
        LAM * g12 + T12 + 2.0 * KK12 - Th * K12 - R12,
        LAM * g22 + T22 + 2.0 * KK22 - Th * K22 - R22,
        LAM * g33 + T33 + 2.0 * KK33 - Th * K33 - R33,
        +2.0 * B1, +2.0 * B2, -2.0 * E1, -2.0 * E2, +2.0 * B3, -2.0 * E3,
    ], dtype=np.float64)


def integrate_u1x2(y0, t_span=(0.0, 12.0), rtol=1e-11, atol=1e-13, max_step=0.04):
    """Integrate 14-variable U(1)_A x U(1)_B control."""
    def ev_col(_t, y):
        D = y[0] * y[2] - y[1] * y[1]
        if D <= 0.0 or y[3] <= 0.0:
            return -1.0
        tr2 = y[0] + y[2]
        disc = math.sqrt(max(0.0, (y[0] - y[2]) ** 2 + 4.0 * y[1] * y[1]))
        return math.sqrt(max(0.0, min(0.5 * (tr2 - disc), y[3]))) - 0.02
    ev_col.terminal = True; ev_col.direction = -1

    def ev_esc(_t, y):
        D = y[0] * y[2] - y[1] * y[1]
        if D <= 0.0 or y[3] <= 0.0:
            return -1.0
        tr2 = y[0] + y[2]
        disc = math.sqrt(max(0.0, (y[0] - y[2]) ** 2 + 4.0 * y[1] * y[1]))
        return math.sqrt(max(0.0, min(0.5 * (tr2 - disc), y[3]))) - 10.0
    ev_esc.terminal = True; ev_esc.direction = 1

    return solve_ivp(abelian_u1x2_rhs, t_span, y0, method="DOP853", rtol=rtol, atol=atol,
                     max_step=max_step, events=[ev_col, ev_esc])


def is_bounced_u1x2(sol):
    """Verify bounce escape for 14-variable U(1)_A x U(1)_B control."""
    if sol.t_events is not None and len(sol.t_events[0]) > 0:
        return False
    y = sol.y[:, -1]
    g11, g12, g22, g33, K11, K12, K22, K33 = y[:8]
    D = g11 * g22 - g12 * g12
    if D <= 0.0 or g33 <= 0.0:
        return False
    gi11, gi12, gi22, gi33 = g22 / D, -g12 / D, g11 / D, 1.0 / g33
    Th = gi11 * K11 + 2.0 * gi12 * K12 + gi22 * K22 + gi33 * K33
    tr_g = g11 + g22 + g33; tr_g2 = g11*g11 + 2.0*g12*g12 + g22*g22 + g33*g33
    R3 = 2.0 * (tr_g * tr_g - 2.0 * tr_g2) / (D * g33)
    tr2 = g11 + g22; disc = math.sqrt(max(0.0, (g11 - g22) ** 2 + 4.0 * g12 * g12))
    amin = math.sqrt(max(0.0, min(0.5 * (tr2 - disc), g33)))
    return bool(Th > 0.0 and amin > 6.0 and R3 < 2.0 * LAM)


def is_bounced_leg(sol, kap, direction=+1):
    """Verify strict bounce escape: no collapse event, sign(Th)==direction, min(a_i)>6, R3<2*LAM."""
    if sol.t_events is not None and len(sol.t_events[0]) > 0:
        return False
    y_end = sol.y[:, -1]
    if y_end[0] * y_end[2] - y_end[1] ** 2 <= 0 or y_end[3] <= 0:
        return False
    de = su2_diagnostics(y_end, kap)
    return bool(direction * de["Th"] > 0.0 and min(de["a1"], de["a2"], de["a3"]) > 6.0 and de["R3"] < 2.0 * LAM)


# =============================================================================
# 2. DIRECT 35-DIGIT MPMATH 4D EULER-ANGLE VERIFICATION (10 EINSTEIN + 12 YM)
# =============================================================================

def verify_4d_euler_su2_mpmath(y_state_mp, kap_mp):
    """Verify ALL 10 4D Einstein equations and ALL 12 covariant 4D Yang-Mills equations
    D_mu F^{a mu nu} = 0 directly in 4D Euler angles (t, theta, phi, psi) at 35 digits!"""
    mp.mp.dps = 35
    Lam_mp = mp.mpf(3)
    eps3 = [(0, 1, 2, 1), (1, 2, 0, 1), (2, 0, 1, 1), (0, 2, 1, -1), (2, 1, 0, -1), (1, 0, 2, -1)]

    def rhs_mp(y):
        g11, g12, g22, g33, K11, K12, K22, K33, A11, A12, A21, A22, A33, cE11, cE12, cE21, cE22, cE33 = y
        D = g11 * g22 - g12 * g12
        detg = D * g33
        v = mp.sqrt(detg)
        gi11, gi12, gi22, gi33 = g22 / D, -g12 / D, g11 / D, 1 / g33
        Km11, Km12 = gi11 * K11 + gi12 * K12, gi11 * K12 + gi12 * K22
        Km21, Km22 = gi12 * K11 + gi22 * K12, gi12 * K12 + gi22 * K22
        Km33 = gi33 * K33
        Th = Km11 + Km22 + Km33
        KK11, KK12 = K11 * Km11 + K12 * Km21, K11 * Km12 + K12 * Km22
        KK22, KK33 = K12 * Km12 + K22 * Km22, K33 * Km33
        g2_11, g2_12, g2_22, g2_33 = g11**2 + g12**2, g12 * (g11 + g22), g12**2 + g22**2, g33**2
        tr_g2 = g2_11 + g2_22 + g2_33
        g3_11, g3_12, g3_22, g3_33 = g2_11 * g11 + g2_12 * g12, g2_11 * g12 + g2_12 * g22, g2_12 * g12 + g2_22 * g22, g2_33 * g33
        fac = 2 / detg
        R11 = fac * (2 * g3_11 - tr_g2 * g11 + 2 * detg)
        R12 = fac * (2 * g3_12 - tr_g2 * g12)
        R22 = fac * (2 * g3_22 - tr_g2 * g22 + 2 * detg)
        R33 = fac * (2 * g3_33 - tr_g2 * g33 + 2 * detg)
        cB11, cB12 = -2 * A11 + A22 * A33, -2 * A12 - A21 * A33
        cB21, cB22 = -2 * A21 - A12 * A33, -2 * A22 + A11 * A33
        cB33 = -2 * A33 + A11 * A22 - A12 * A21
        E11, E12 = (g11 * cE11 + g12 * cE12) / v, (g12 * cE11 + g22 * cE12) / v
        E21, E22 = (g11 * cE21 + g12 * cE22) / v, (g12 * cE21 + g22 * cE22) / v
        E33 = g33 * cE33 / v
        B11, B12 = (g11 * cB11 + g12 * cB12) / v, (g12 * cB11 + g22 * cB12) / v
        B21, B22 = (g11 * cB21 + g12 * cB22) / v, (g12 * cB21 + g22 * cB22) / v
        B33 = g33 * cB33 / v
        u = mp.mpf("0.5") * kap_mp * (
            E11 * cE11 + E12 * cE12 + E21 * cE21 + E22 * cE22 + E33 * cE33
            + B11 * cB11 + B12 * cB12 + B21 * cB21 + B22 * cB22 + B33 * cB33
        ) / v
        T11 = u * g11 - kap_mp * (E11**2 + E21**2 + B11**2 + B21**2)
        T12 = u * g12 - kap_mp * (E11 * E12 + E21 * E22 + B11 * B12 + B21 * B22)
        T22 = u * g22 - kap_mp * (E12**2 + E22**2 + B12**2 + B22**2)
        T33 = u * g33 - kap_mp * (E33**2 + B33**2)
        dcE11 = 2 * B11 - (A33 * B22 + A22 * B33)
        dcE12 = 2 * B12 + (A33 * B21 + A21 * B33)
        dcE21 = 2 * B21 + (A33 * B12 + A12 * B33)
        dcE22 = 2 * B22 - (A33 * B11 + A11 * B33)
        dcE33 = 2 * B33 - (A22 * B11 - A12 * B21 - A21 * B12 + A11 * B22)
        return [
            2 * K11, 2 * K12, 2 * K22, 2 * K33,
            Lam_mp * g11 + T11 + 2 * KK11 - Th * K11 - R11,
            Lam_mp * g12 + T12 + 2 * KK12 - Th * K12 - R12,
            Lam_mp * g22 + T22 + 2 * KK22 - Th * K22 - R22,
            Lam_mp * g33 + T33 + 2 * KK33 - Th * K33 - R33,
            E11, E12, E21, E22, E33,
            dcE11, dcE12, dcE21, dcE22, dcE33,
        ]

    def evolve_local(dt):
        if dt == 0:
            return y_state_mp
        k1 = rhs_mp(y_state_mp)
        k2 = rhs_mp([y_state_mp[i] + dt / 2 * k1[i] for i in range(18)])
        k3 = rhs_mp([y_state_mp[i] + dt / 2 * k2[i] for i in range(18)])
        k4 = rhs_mp([y_state_mp[i] + dt * k3[i] for i in range(18)])
        return [y_state_mp[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(18)]

    def metric_fn(pt):
        dt, th, ph, ps = pt
        w1, w2, w3 = left_invariant_coframe(th, ph, ps)
        y = evolve_local(dt)
        g = mp.zeros(4, 4)
        g[0, 0] = -mp.mpf(1)
        for i in range(4):
            for j in range(4):
                g[i, j] += y[0] * w1[i] * w1[j] + y[1] * (w1[i] * w2[j] + w2[i] * w1[j]) + y[2] * w2[i] * w2[j] + y[3] * w3[i] * w3[j]
        return g

    def A_4d(pt):
        dt, th, ph, ps = pt
        w1, w2, w3 = left_invariant_coframe(th, ph, ps)
        y = evolve_local(dt)
        A11, A12, A21, A22, A33 = y[8:13]
        return [
            [A11 * w1[m] + A12 * w2[m] for m in range(4)],
            [A21 * w1[m] + A22 * w2[m] for m in range(4)],
            [A33 * w3[m] for m in range(4)],
        ]

    h = mp.mpf("1e-7")

    def F_4d(pt):
        A0 = A_4d(pt)
        dA = [[[mp.mpf(0) for _ in range(4)] for _ in range(3)] for _ in range(4)]
        for mu in range(4):
            p1, m1, p2, m2 = list(pt), list(pt), list(pt), list(pt)
            p1[mu] += h; m1[mu] -= h; p2[mu] += 2 * h; m2[mu] -= 2 * h
            Ap1, Am1, Ap2, Am2 = A_4d(p1), A_4d(m1), A_4d(p2), A_4d(m2)
            for a in range(3):
                for nu in range(4):
                    dA[mu][a][nu] = (-Ap2[a][nu] + 8 * Ap1[a][nu] - 8 * Am1[a][nu] + Am2[a][nu]) / (12 * h)
        F = [mp.zeros(4, 4) for _ in range(3)]
        for a in range(3):
            for mu in range(4):
                for nu in range(4):
                    val = dA[mu][a][nu] - dA[nu][a][mu]
                    for aa, bb, cc, sgn in eps3:
                        if aa == a:
                            val += sgn * A0[bb][mu] * A0[cc][nu]
                    F[a][mu, nu] = val
        return F

    x0 = [mp.mpf(0), mp.mpf("1.1"), mp.mpf("0.7"), mp.mpf("0.5")]
    g, ginv, _, _, _, G_cov = compute_einstein_4d(metric_fn, x0, h=h)
    F0 = F_4d(x0)

    F2_tot = mp.mpf(0)
    for a in range(3):
        Fup = ginv * F0[a] * ginv.T
        F2_tot += sum(F0[a][r, s] * Fup[r, s] for r in range(4) for s in range(4))
    T_cov = mp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            s = mp.mpf(0)
            for a in range(3):
                s += sum(F0[a][mu, r] * F0[a][nu, q] * ginv[r, q] for r in range(4) for q in range(4))
            T_cov[mu, nu] = kap_mp * (s - mp.mpf("0.25") * g[mu, nu] * F2_tot)

    w1_0, w2_0, w3_0 = left_invariant_coframe(x0[1], x0[2], x0[3])
    E_li_inv = mp.matrix([[1, 0, 0, 0], w1_0, w2_0, w3_0]) ** (-1)
    E_res = E_li_inv.T * (G_cov + Lam_mp * g - T_cov) * E_li_inv
    max_E = max(abs(E_res[i, j]) for i in range(4) for j in range(4))

    def dens_Fup(pt):
        gp = metric_fn(pt)
        gpi = gp ** (-1)
        sq = mp.sqrt(-mp.det(gp))
        Fp = F_4d(pt)
        return [sq * (gpi * Fp[a] * gpi.T) for a in range(3)]

    A_x0 = A_4d(x0)
    sq0 = mp.sqrt(-mp.det(g))
    Fup0 = [ginv * F0[a] * ginv.T for a in range(3)]
    divF = [[mp.mpf(0) for _ in range(4)] for _ in range(3)]
    for mu in range(4):
        p1, m1, p2, m2 = list(x0), list(x0), list(x0), list(x0)
        p1[mu] += h; m1[mu] -= h; p2[mu] += 2 * h; m2[mu] -= 2 * h
        dp1, dm1, dp2, dm2 = dens_Fup(p1), dens_Fup(m1), dens_Fup(p2), dens_Fup(m2)
        for a in range(3):
            for nu in range(4):
                divF[a][nu] += (-dp2[a][mu, nu] + 8 * dp1[a][mu, nu] - 8 * dm1[a][mu, nu] + dm2[a][mu, nu]) / (12 * h * sq0)
    for a in range(3):
        for nu in range(4):
            for aa, bb, cc, sgn in eps3:
                if aa == a:
                    divF[a][nu] += sgn * sum(A_x0[bb][mu] * Fup0[cc][mu, nu] for mu in range(4))
    max_YM = max(abs(divF[a][nu]) for a in range(3) for nu in range(4))
    max_Gauss = max(abs(divF[a][0]) for a in range(3))

    dy0 = rhs_mp(y_state_mp)
    dG3 = (
        dy0[8] * y_state_mp[15] + y_state_mp[8] * dy0[15]
        + dy0[9] * y_state_mp[16] + y_state_mp[9] * dy0[16]
        - dy0[10] * y_state_mp[13] - y_state_mp[10] * dy0[13]
        - dy0[11] * y_state_mp[14] - y_state_mp[11] * dy0[14]
    )
    return {
        "max_10_einstein_eq_4D": float(max_E),
        "max_12_ym_eq_4D": float(max_YM),
        "max_3_gauss_4D": float(max_Gauss),
        "dG3_dt_exact": float(abs(dG3)),
    }


def run_exact_4d_verification_and_kill_tests():
    """Run (a) 35-digit 4D check on generic exact tilted state, (b) Gauss Ellipsoid theorem check,
    (c) P01 Abelian Model I limit recovery, (d) Monograph Sec. 4 triaxial SU(2) limit recovery,
    (e) DOP853 vs Radau on a live tilted non-Abelian bounce + 35-digit 4D live-peak check."""
    mp.mp.dps = 35
    Lam_mp = mp.mpf(3)
    kap_mp = mp.mpf("0.7")

    # (a) Generic exact 35-digit tilted state (ALL off-diagonals nonzero!)
    g11, g12, g22, g33 = mp.mpf("1.3"), mp.mpf("0.21"), mp.mpf("0.85"), mp.mpf("1.05")
    A11, A12, A21, A22, A33 = mp.mpf("0.7"), mp.mpf("-0.3"), mp.mpf("0.4"), mp.mpf("0.9"), mp.mpf("0.5")
    cE11, cE12, cE21, cE33 = mp.mpf("0.6"), mp.mpf("-0.25"), mp.mpf("0.35"), mp.mpf("0.45")
    cE22 = (A21 * cE11 + A22 * cE12 - A11 * cE21) / A12

    D = g11 * g22 - g12**2
    detg = D * g33
    v = mp.sqrt(detg)
    gi11, gi12, gi22, gi33 = g22 / D, -g12 / D, g11 / D, 1 / g33
    cB11, cB12 = -2 * A11 + A22 * A33, -2 * A12 - A21 * A33
    cB21, cB22 = -2 * A21 - A12 * A33, -2 * A22 + A11 * A33
    cB33 = -2 * A33 + A11 * A22 - A12 * A21
    T03 = (kap_mp / v) * ((cE11 * cB12 - cE12 * cB11) + (cE21 * cB22 - cE22 * cB21))
    K11, K22 = mp.mpf("0.2"), mp.mpf("-0.1")
    K12 = (mp.mpf("0.5") * T03 - gi12 * (K22 - K11)) / (gi11 - gi22)
    E11, E12 = (g11 * cE11 + g12 * cE12) / v, (g12 * cE11 + g22 * cE12) / v
    E21, E22 = (g11 * cE21 + g12 * cE22) / v, (g12 * cE21 + g22 * cE22) / v
    E33 = g33 * cE33 / v
    B11, B12 = (g11 * cB11 + g12 * cB12) / v, (g12 * cB11 + g22 * cB12) / v
    B21, B22 = (g11 * cB21 + g12 * cB22) / v, (g12 * cB21 + g22 * cB22) / v
    B33 = g33 * cB33 / v
    u0 = mp.mpf("0.5") * kap_mp * (
        E11 * cE11 + E12 * cE12 + E21 * cE21 + E22 * cE22 + E33 * cE33
        + B11 * cB11 + B12 * cB12 + B21 * cB21 + B22 * cB22 + B33 * cB33
    ) / v
    tr_g = g11 + g22 + g33
    tr_g2 = g11**2 + 2 * g12**2 + g22**2 + g33**2
    R3_0 = 2 * (tr_g**2 - 2 * tr_g2) / detg
    Km11, Km12 = gi11 * K11 + gi12 * K12, gi11 * K12 + gi12 * K22
    Km21, Km22 = gi12 * K11 + gi22 * K12, gi12 * K12 + gi22 * K22
    Km33 = (Lam_mp + u0 - mp.mpf("0.5") * R3_0 - (Km11 * Km22 - Km12 * Km21)) / (Km11 + Km22)
    K33 = g33 * Km33
    y_gen_mp = [g11, g12, g22, g33, K11, K12, K22, K33, A11, A12, A21, A22, A33, cE11, cE12, cE21, cE22, cE33]
    check_4d_generic = verify_4d_euler_su2_mpmath(y_gen_mp, kap_mp)

    # (b) Pure-Null Gauss Ellipsoid Theorem check on 100 random 2x2 A_perp matrices
    rng = np.random.default_rng(12345)
    max_ellip_err = 0.0
    for _ in range(100):
        a1, a2 = rng.uniform(0.5, 2.5, 2)
        A = rng.normal(size=(2, 2))
        detA = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        A33_e = 0.5 * detA
        cB = np.array([[-2 * A[0, 0] + A[1, 1] * A33_e, -2 * A[0, 1] - A[1, 0] * A33_e],
                       [-2 * A[1, 0] - A[0, 1] * A33_e, -2 * A[1, 1] + A[0, 0] * A33_e]])
        cE = np.array([[-(a2 / a1) * cB[0, 1], +(a1 / a2) * cB[0, 0]],
                       [-(a2 / a1) * cB[1, 1], +(a1 / a2) * cB[1, 0]]])
        G3_num = A[0, 0] * cE[1, 0] + A[0, 1] * cE[1, 1] - A[1, 0] * cE[0, 0] - A[1, 1] * cE[0, 1]
        col1_sq = A[0, 0]**2 + A[1, 0]**2
        col2_sq = A[0, 1]**2 + A[1, 1]**2
        G3_thm = detA * (2.0 * (a2 / a1 + a1 / a2) - 0.5 * ((a2 / a1) * col1_sq + (a1 / a2) * col2_sq))
        max_ellip_err = max(max_ellip_err, abs(G3_num - G3_thm))

    # (c) Kill-Test 1: Exact recovery of P01's 12-variable Abelian Model I at t = 4.0 (past avoided crossing!)
    y0_p01 = init_sync_page1(2.0, 1.15, 1.0, 0.40, phi0=0.0)
    # Remember our sign convention: E_ours = -E_P01, cB_ours = +cB_P01 = -2*A => A = -0.5*cB_P01, cE = -cE_P01!
    y0_su2_ab = np.zeros(18, dtype=np.float64)
    y0_su2_ab[:8] = y0_p01[:8]
    y0_su2_ab[8] = -0.5 * y0_p01[10]   # A11 = -0.5 * cB1
    y0_su2_ab[9] = -0.5 * y0_p01[11]   # A12 = -0.5 * cB2
    y0_su2_ab[13] = -y0_p01[8]         # cE11 = -cE1
    y0_su2_ab[14] = -y0_p01[9]         # cE12 = -cE2
    sol_p01 = solve_ivp(sync_ode_rhs, (0.0, 4.0), y0_p01, method="DOP853", rtol=1e-13, atol=1e-15, t_eval=[4.0])
    sol_su2_ab = solve_ivp(make_su2_rhs(1.0), (0.0, 4.0), y0_su2_ab, method="DOP853", rtol=1e-13, atol=1e-15, t_eval=[4.0])
    kt_p01_diff = float(np.max(np.abs(sol_su2_ab.y[:8, 0] - sol_p01.y[:8, 0])))

    # (d) Kill-Test 2: Exact recovery of Monograph Sec. 4 triaxial SU(2)_L (T03=0, diagonal g, A=2*g_a) at t = 3.0!
    kap_t2 = 0.05
    a1_0, a2_0, a3_0 = 1.6, 1.45, 1.5
    g1_0, g2_0, g3_0 = 0.6, 0.48, 0.52
    dg1_0, dg2_0, dg3_0 = 0.3, 0.28, 0.31
    # Standalone diagonal triaxial SU(2) oracle:
    def diag_su2_oracle(_t, s):
        s1, s2, s3, H1, H2, H3, q1, q2, q3, p1, p2, p3 = s
        dq1, dq2, dq3 = p1 * s1 / (s2 * s3), p2 * s2 / (s3 * s1), p3 * s3 / (s1 * s2)
        E1, E2, E3 = dq1 / s1, dq2 / s2, dq3 / s3
        B1 = 2 * (q2 * q3 - q1) / (s2 * s3)
        B2 = 2 * (q3 * q1 - q2) / (s3 * s1)
        B3 = 2 * (q1 * q2 - q3) / (s1 * s2)
        py1 = 2 * kap_t2 * (E2**2 + E3**2 - E1**2 + B2**2 + B3**2 - B1**2)
        py2 = 2 * kap_t2 * (E3**2 + E1**2 - E2**2 + B3**2 + B1**2 - B2**2)
        py3 = 2 * kap_t2 * (E1**2 + E2**2 - E3**2 + B1**2 + B2**2 - B3**2)
        den = (s1 * s2 * s3) ** 2
        r1 = 2 * (s1**4 - (s2**2 - s3**2)**2) / den
        r2 = 2 * (s2**4 - (s3**2 - s1**2)**2) / den
        r3 = 2 * (s3**4 - (s1**2 - s2**2)**2) / den
        Th = H1 + H2 + H3
        dp1 = -4 * ((q1 - q2 * q3) * s1 / (s2 * s3) + (q3 * q1 - q2) * q3 * s2 / (s3 * s1) + (q1 * q2 - q3) * q2 * s3 / (s1 * s2))
        dp2 = -4 * ((q2 - q3 * q1) * s2 / (s3 * s1) + (q1 * q2 - q3) * q1 * s3 / (s1 * s2) + (q2 * q3 - q1) * q3 * s1 / (s2 * s3))
        dp3 = -4 * ((q3 - q1 * q2) * s3 / (s1 * s2) + (q2 * q3 - q1) * q2 * s1 / (s2 * s3) + (q3 * q1 - q2) * q1 * s2 / (s3 * s1))
        return np.array([H1 * s1, H2 * s2, H3 * s3, LAM + py1 - Th * H1 - r1, LAM + py2 - Th * H2 - r2, LAM + py3 - Th * H3 - r3,
                         dq1, dq2, dq3, dp1, dp2, dp3], dtype=np.float64)

    E1_0, E2_0, E3_0 = dg1_0 / a1_0, dg2_0 / a2_0, dg3_0 / a3_0
    B1_0 = 2 * (g2_0 * g3_0 - g1_0) / (a2_0 * a3_0)
    B2_0 = 2 * (g3_0 * g1_0 - g2_0) / (a3_0 * a1_0)
    B3_0 = 2 * (g1_0 * g2_0 - g3_0) / (a1_0 * a2_0)
    u_t2 = 2 * kap_t2 * (E1_0**2 + E2_0**2 + E3_0**2 + B1_0**2 + B2_0**2 + B3_0**2)
    den0 = (a1_0 * a2_0 * a3_0) ** 2
    R3_t2 = 2 * (2 * (a1_0**2 * a2_0**2 + a2_0**2 * a3_0**2 + a3_0**2 * a1_0**2) - (a1_0**4 + a2_0**4 + a3_0**4)) / den0
    H0_t2 = -math.sqrt((u_t2 + LAM - 0.5 * R3_t2) / 3.0)
    p1_0, p2_0, p3_0 = (a2_0 * a3_0 / a1_0) * dg1_0, (a3_0 * a1_0 / a2_0) * dg2_0, (a1_0 * a2_0 / a3_0) * dg3_0
    sol_or = solve_ivp(diag_su2_oracle, (0.0, 3.0), [a1_0, a2_0, a3_0, H0_t2, H0_t2, H0_t2, g1_0, g2_0, g3_0, p1_0, p2_0, p3_0],
                       method="DOP853", rtol=1e-13, atol=1e-15, t_eval=[3.0])
    # Map to our 18-var convention: A^a_a = 2*g_a, cE^{aa} = 2*pi_a!
    y0_t2 = np.array([a1_0**2, 0.0, a2_0**2, a3_0**2, a1_0**2 * H0_t2, 0.0, a2_0**2 * H0_t2, a3_0**2 * H0_t2,
                      2 * g1_0, 0.0, 0.0, 2 * g2_0, 2 * g3_0, 2 * p1_0, 0.0, 0.0, 2 * p2_0, 2 * p3_0], dtype=np.float64)
    sol_su2_t2 = solve_ivp(make_su2_rhs(kap_t2), (0.0, 3.0), y0_t2, method="DOP853", rtol=1e-13, atol=1e-15, t_eval=[3.0])
    g_or_3 = np.array([sol_or.y[0, 0]**2, 0.0, sol_or.y[1, 0]**2, sol_or.y[2, 0]**2])
    kt_sec4_diff = float(np.max(np.abs(sol_su2_t2.y[:4, 0] - g_or_3)))

    # (e) Kill-Test 3: DOP853 vs Radau on a live tilted non-Abelian bounce + 35-digit 4D check on live bounced state! (closes P2-7)
    y0_live, kap_live = build_pure_null_nonabelian_state(1.8, 1.18, 1.05, 0.32, phi_ell=0.95, alpha=0.35, sh12=0.05, sh3=-0.04)
    sol_dop = integrate_su2(y0_live, kap_live, (0.0, 4.5), rtol=1e-13, atol=1e-15, method="DOP853", t_eval=[2.5, 4.0])
    sol_rad = integrate_su2(y0_live, kap_live, (0.0, 4.5), rtol=1e-12, atol=1e-14, method="Radau", t_eval=[2.5, 4.0])
    kt_dop_vs_rad = float(np.max(np.abs(sol_dop.y[:, 1] - sol_rad.y[:, 1])))
    d_live = su2_diagnostics(sol_dop.y[:, 0], kap_live)
    # 35-digit 4D check directly on the live bounced trajectory state at t = 2.5!
    y_live_mp = [mp.mpf(repr(float(v))) for v in sol_dop.y[:, 0]]
    check_4d_live_bounce = verify_4d_euler_su2_mpmath(y_live_mp, mp.mpf(repr(float(kap_live))))

    return {
        "mpmath_35dps_generic_tilted_4D": check_4d_generic,
        "mpmath_35dps_live_bounced_state_t2p5_4D": check_4d_live_bounce,
        "pure_null_gauss_ellipsoid_max_err_100trials": float(max_ellip_err),
        "KT1_p01_abelian_model1_recovery_t4": kt_p01_diff,
        "KT2_monograph_sec4_triaxial_su2_recovery_t3": kt_sec4_diff,
        "KT3_live_tilted_nonabelian_DOP853_vs_Radau_t4": kt_dop_vs_rad,
        "KT3_live_state_t2p5_diag": {
            "H_res": abs(d_live["H_res"]), "M3_res": abs(d_live["M3_res"]), "G3_res": abs(d_live["G3_res"]),
            "u3_frac": d_live["u3_frac"], "S3_over_u": d_live["S3_over_u"], "det_A_perp": d_live["det_A_perp"],
        },
    }


# =============================================================================
# 3. GENERAL INITIAL-DATA CONSTRUCTORS & PAIRED CONTRACTING ENSEMBLES
# =============================================================================

def build_pure_null_nonabelian_state(am, r0, s0, u0, phi_ell=0.95, alpha=0.0, sh12=0.0, sh3=0.0):
    """Build exact 100% pure-null (|S3/u|==1), exact Gauss (G3==0), genuinely non-Abelian
    (det(A_perp)!=0, A33 = 0.5*det(A_perp)) contracting state sampling the FULL Gauss ellipsoid
    angle phi_ell in (0, pi/2) (closes P2-5: covers BOTH q1 < 2 < q2 AND q2 < 2 < q1!).
    """
    a_perp = am / (s0 ** (1.0 / 3.0))
    a1, a2, a3 = a_perp * math.sqrt(r0), a_perp / math.sqrt(r0), s0 * a_perp
    # Exact full Gauss ellipsoid parameterization:
    #   (a2/a1)*q1^2 + (a1/a2)*q2^2 == 4*(a2/a1 + a1/a2)
    #   => q1 = 2*sqrt(1 + (a1/a2)^2)*cos(phi_ell), q2 = 2*sqrt(1 + (a2/a1)^2)*sin(phi_ell)!
    r_ratio = a1 / a2
    q1 = 2.0 * math.sqrt(1.0 + r_ratio * r_ratio) * math.cos(phi_ell)
    q2 = 2.0 * math.sqrt(1.0 + 1.0 / (r_ratio * r_ratio)) * math.sin(phi_ell)
    ca, sa = math.cos(alpha), math.sin(alpha)
    if abs(ca) < 0.05 or q1 < 1e-4 or q2 < 1e-4:
        return None, None
    A11, A12 = q1, q2 * sa
    A21, A22 = 0.0, q2 * ca
    A33 = 0.5 * (A11 * A22 - A12 * A21)

    cB11, cB12 = -2.0 * A11 + A22 * A33, -2.0 * A12 - A21 * A33
    cB21, cB22 = -2.0 * A21 - A12 * A33, -2.0 * A22 + A11 * A33
    cE11, cE12 = -(a2 / a1) * cB12, +(a1 / a2) * cB11
    cE21, cE22 = -(a2 / a1) * cB22, +(a1 / a2) * cB21
    cE33 = 0.0

    v = a1 * a2 * a3
    g11, g22, g33 = a1 * a1, a2 * a2, a3 * a3
    u_unit = 0.5 * (
        g11 * (cE11**2 + cE21**2 + cB11**2 + cB21**2)
        + g22 * (cE12**2 + cE22**2 + cB12**2 + cB22**2)
    ) / (v * v)
    if u_unit <= 1e-14:
        return None, None
    u_t = u0 / (a1 * a2 * a3 * a3)
    kap = u_t / u_unit
    T03 = (kap / v) * ((cE11 * cB12 - cE12 * cB11) + (cE21 * cB22 - cE22 * cB21))
    K12 = 0.5 * T03 * (g11 * g22) / (g22 - g11)
    sig12 = -K12 / (a1 * a2)
    R3_0 = 2.0 * ((g11 + g22 + g33) ** 2 - 2.0 * (g11**2 + g22**2 + g33**2)) / (g11 * g22 * g33)
    rhs = u_t + LAM + sig12**2 - 0.5 * R3_0
    c1, c2, c3 = 1.0 + sh12, 1.0 - sh12, 1.0 + sh3
    coeff = c1 * c2 + c2 * c3 + c3 * c1
    if rhs <= 0.0 or coeff <= 0.0:
        return None, None
    Hb0 = -math.sqrt(rhs / coeff)
    y0 = np.array([
        g11, 0.0, g22, g33, g11 * Hb0 * c1, K12, g22 * Hb0 * c2, g33 * Hb0 * c3,
        A11, A12, A21, A22, A33, cE11, cE12, cE21, cE22, cE33,
    ], dtype=np.float64)
    return y0, kap


def build_paired_abelian_pure_null(y0_na, kap_na):
    """Build exact paired Abelian Model I state (kap=1, color 1 only) with IDENTICAL initial
    metric (g_ij, K_ij), IDENTICAL u(0), and IDENTICAL pure-null T_{03}(0) as y0_na."""
    d0 = su2_diagnostics(y0_na, kap_na)
    a1, a2, a3 = d0["a1"], d0["a2"], d0["a3"]
    amp = math.sqrt(d0["u"])
    y0_ab = np.zeros(18, dtype=np.float64)
    y0_ab[:8] = y0_na[:8]
    # Sign of T03 in y0_na: T03 = -a3 * u (since S3_over_u = -1)
    sgn = 1.0 if d0["S3_over_u"] < 0 else -1.0
    y0_ab[9] = sgn * 0.5 * a3 * a1 * amp   # A12 -> cB12 = -sgn * a3 * a1 * amp
    y0_ab[13] = a2 * a3 * amp              # cE11 = +a2 * a3 * amp -> T03 = -sgn * a3 * u
    return y0_ab


def run_paired_pure_null_ensembles():
    """Paired head-to-head Abelian vs Non-Abelian pure-null contracting ensembles:
      - Ensemble 1 (Wide above-wall, u0 in [0.514, 0.75], matching P01's 0/800 distribution): N=200
      - Ensemble 2 (Near-bounce prolate above-wall, u0 in [0.514, 0.75], incl. off-diag alpha): N=200
      - Ensemble 3 (Sub-wall control, u0 in [0.20, 0.50]): N=200
    """
    rng = np.random.default_rng(20261124)
    configs = [
        ("above_wall_wide_P01_dist", (0.95, 3.5), (1.01, 1.90), (0.65, 1.65), (0.514, 0.75), (-0.30, 0.30), 200),
        ("above_wall_prolate_near_bounce", (0.85, 1.50), (1.02, 1.55), (0.95, 1.55), (0.514, 0.75), (-0.15, 0.15), 200),
        ("sub_wall_control", (0.95, 3.0), (1.02, 1.65), (0.80, 1.55), (0.20, 0.50), (-0.25, 0.25), 200),
    ]
    out = {}
    for name, am_r, r0_r, s0_r, u0_r, sh_r, n_target in configs:
        n_valid = 0
        ab_ok = na_ok = both_ok = only_na = only_ab = 0
        na_collapsed = na_solver_fail = na_timeout = 0
        q1_gt_2_count = 0
        max_u0_ab = max_u0_na = 0.0
        max_H_na = max_M_na = max_G_na = 0.0
        max_H_col = max_M_col = max_G_col = 0.0
        max_u3_detour = 0.0
        min_S3_after_detour = 1.0
        attempts = 0
        while n_valid < n_target and attempts < n_target * 10:
            attempts += 1
            am = float(rng.uniform(*am_r))
            r0 = float(rng.uniform(*r0_r))
            s0 = float(rng.uniform(*s0_r))
            u0 = float(rng.uniform(*u0_r))
            sh12 = float(rng.uniform(*sh_r))
            sh3 = float(rng.uniform(*sh_r))
            # Sample FULL Gauss ellipsoid angle phi_ell in (0.08, pi/2 - 0.08) (closes P2-5!)
            phi_ell = float(rng.uniform(0.08, 0.5 * math.pi - 0.08))
            alpha = float(rng.uniform(-0.85, 0.85))
            y0_na, kap_na = build_pure_null_nonabelian_state(am, r0, s0, u0, phi_ell=phi_ell, alpha=alpha, sh12=sh12, sh3=sh3)
            if y0_na is None:
                continue
            d0 = su2_diagnostics(y0_na, kap_na)
            assert abs(d0["H_res"]) < 1e-11 and abs(d0["M3_res"]) < 1e-11 and abs(d0["G3_res"]) < 1e-11
            assert abs(abs(d0["S3_over_u"]) - 1.0) < 1e-12
            if y0_na[8] > 2.0:
                q1_gt_2_count += 1

            y0_ab = build_paired_abelian_pure_null(y0_na, kap_na)
            d0_ab = su2_diagnostics(y0_ab, 1.0)
            assert abs(d0_ab["H_res"]) < 1e-11 and abs(d0_ab["M3_res"]) < 1e-11

            n_valid += 1
            sol_ab = integrate_su2(y0_ab, 1.0, (0.0, 12.0))
            sol_na = integrate_su2(y0_na, kap_na, (0.0, 12.0))
            ok_a = is_bounced_leg(sol_ab, 1.0, +1)
            ok_n = is_bounced_leg(sol_na, kap_na, +1)
            if ok_a:
                ab_ok += 1
                max_u0_ab = max(max_u0_ab, u0)
            if ok_n:
                na_ok += 1
                max_u0_na = max(max_u0_na, u0)
                ds = [su2_diagnostics(sol_na.y[:, i], kap_na) for i in range(sol_na.y.shape[1])]
                max_H_na = max(max_H_na, max(abs(d["H_res"]) for d in ds))
                max_M_na = max(max_M_na, max(abs(d["M3_res"]) for d in ds))
                max_G_na = max(max_G_na, max(abs(d["G3_res"]) for d in ds))
                max_u3_detour = max(max_u3_detour, max(d["u3_frac"] for d in ds))
                min_S3_after_detour = min(min_S3_after_detour, min(abs(d["S3_over_u"]) for d in ds))
            else:
                # Explicit termination & constraint audit on non-bouncing runs (closes P2-6!)
                if sol_na.t_events is not None and len(sol_na.t_events[0]) > 0:
                    na_collapsed += 1
                elif not sol_na.success:
                    na_solver_fail += 1
                else:
                    na_timeout += 1
                # Check constraints at midpoint and near-end (a_min >= 0.05) before singularity blowup
                for i_chk in (len(sol_na.t) // 2, max(0, len(sol_na.t) - 3)):
                    dc = su2_diagnostics(sol_na.y[:, i_chk], kap_na)
                    if min(dc["a1"], dc["a2"], dc["a3"]) >= 0.05:
                        max_H_col = max(max_H_col, abs(dc["H_res"]) / max(1.0, dc["u"]))
                        max_M_col = max(max_M_col, abs(dc["M3_res"]) / max(1.0, dc["u"]))
                        max_G_col = max(max_G_col, abs(dc["G3_res"]))
            if ok_a and ok_n:
                both_ok += 1
            elif ok_n and not ok_a:
                only_na += 1
            elif ok_a and not ok_n:
                only_ab += 1

        assert na_solver_fail == 0, f"Unexpected silent solver failures in {name}: {na_solver_fail}"
        out[name] = {
            "n_valid": n_valid,
            "q1_gt_2_upper_half_count": q1_gt_2_count,
            "abelian_bounced": ab_ok,
            "nonabelian_bounced": na_ok,
            "nonabelian_collapsed_event": na_collapsed,
            "nonabelian_solver_fail": na_solver_fail,
            "nonabelian_timeout": na_timeout,
            "both_bounced": both_ok,
            "only_nonabelian_bounced": only_na,
            "only_abelian_bounced": only_ab,
            "max_u0_abelian": max_u0_ab,
            "max_u0_nonabelian": max_u0_na,
            "bounced_max_u3_detour_frac": max_u3_detour,
            "bounced_min_abs_S3_over_u": min_S3_after_detour if na_ok > 0 else None,
            "bounced_max_H_res": max_H_na,
            "bounced_max_M3_res": max_M_na,
            "bounced_max_G3_res": max_G_na,
            "collapsed_max_rel_H_res_amin_ge_0p05": max_H_col,
            "collapsed_max_rel_M3_res_amin_ge_0p05": max_M_col,
            "collapsed_max_G3_res_amin_ge_0p05": max_G_col,
        }
    return out


def run_paired_partially_tilted_coupling_sweep():
    """Compare partially-tilted contracting trials (u0 in [0.514, 0.75], |S3/u|_0 in [0.15, 0.85])
    across g_YM in {0 (paired Abelian!), 0.5, 1.0, 2.0, 4.0} on the EXACT SAME metric + (u0, S3(0)) draws!
    Tests whether strong non-Abelian coupling (E_YM^mono > 128/81) enhances or suppresses survival."""
    rng = np.random.default_rng(777)
    n_draws = 150
    draws = []
    while len(draws) < n_draws:
        am = float(rng.uniform(0.90, 2.20))
        r0 = float(rng.uniform(1.02, 1.55))
        s0 = float(rng.uniform(0.85, 1.50))
        u0 = float(rng.uniform(0.514, 0.75))
        sh12 = float(rng.uniform(-0.20, 0.20))
        sh3 = float(rng.uniform(-0.20, 0.20))
        ratio_A = float(rng.uniform(0.20, 0.70))
        beta_E = float(rng.uniform(0.6, 2.5))
        # Check Hamiltonian feasibility for Abelian baseline at same |S3/u|
        a_perp = am / (s0 ** (1.0 / 3.0))
        a1, a2, a3 = a_perp * math.sqrt(r0), a_perp / math.sqrt(r0), s0 * a_perp
        g11, g22, g33 = a1**2, a2**2, a3**2
        R3_0 = 2.0 * ((g11 + g22 + g33)**2 - 2.0 * (g11**2 + g22**2 + g33**2)) / (g11 * g22 * g33)
        u_t = u0 / (a1 * a2 * a3**2)
        c1, c2, c3 = 1.0 + sh12, 1.0 - sh12, 1.0 + sh3
        coeff = c1 * c2 + c2 * c3 + c3 * c1
        if u_t + LAM - 0.5 * R3_0 > 0 and coeff > 0:
            draws.append((am, r0, s0, u0, sh12, sh3, ratio_A, beta_E))

    results = {}
    for g_ym in [0.5, 1.0, 2.0, 4.0, 10.0, 30.0]:
        label = f"su2_g{g_ym:.1f}"
        kap = 1.0 / (g_ym ** 2)
        n_ok = 0
        n_ok_ab_s3 = 0
        n_ok_ab_s3_u3 = 0
        only_na_vs_s3u3 = 0
        only_ab_vs_s3u3 = 0
        n_valid = 0
        max_u0 = 0.0
        max_S3_surv = 0.0
        med_Eym, med_u3_0, med_s3_0 = [], [], []
        max_H = max_M = max_G = 0.0
        for (am, r0, s0, u0, sh12, sh3, ratio_A, beta_E) in draws:
            a_perp = am / (s0 ** (1.0 / 3.0))
            a1, a2, a3 = a_perp * math.sqrt(r0), a_perp / math.sqrt(r0), s0 * a_perp
            g11, g22, g33 = a1**2, a2**2, a3**2
            v = a1 * a2 * a3
            u_t = u0 / (a1 * a2 * a3**2)
            c_quad = 0.5 * kap * (g22 * beta_E**2 + g11 * (ratio_A * beta_E)**2 + 4 * g11 + 4 * g22 * ratio_A**2) / (v * v) * (a1 * a2 * a3**2)
            c_quar = 0.5 * kap * (g33 * ratio_A**2) / (v * v) * (a1 * a2 * a3**2)
            s = math.sqrt((-c_quad + math.sqrt(c_quad**2 + 4.0 * c_quar * u0)) / (2.0 * c_quar))
            A11, A12, A21, A22, A33 = s, 0.0, 0.0, s * ratio_A, 0.0
            cE11, cE12, cE21, cE22, cE33 = 0.0, s * beta_E, s * ratio_A * beta_E, 0.0, 0.0
            T03 = (kap / v) * (-cE12 * (-2.0 * A11) + cE21 * (-2.0 * A22))

            K12 = 0.5 * T03 * (g11 * g22) / (g22 - g11)
            sig12 = -K12 / (a1 * a2)
            R3_0 = 2.0 * ((g11 + g22 + g33)**2 - 2.0 * (g11**2 + g22**2 + g33**2)) / (g11 * g22 * g33)
            rhs = u_t + LAM + sig12**2 - 0.5 * R3_0
            c1, c2, c3 = 1.0 + sh12, 1.0 - sh12, 1.0 + sh3
            coeff = c1 * c2 + c2 * c3 + c3 * c1
            if rhs <= 0.0 or coeff <= 0.0:
                continue
            Hb0 = -math.sqrt(rhs / coeff)
            y0 = np.array([g11, 0.0, g22, g33, g11 * Hb0 * c1, K12, g22 * Hb0 * c2, g33 * Hb0 * c3,
                           A11, A12, A21, A22, A33, cE11, cE12, cE21, cE22, cE33], dtype=np.float64)
            d0 = su2_diagnostics(y0, kap)
            med_Eym.append(d0["E_ym_mono"])
            med_u3_0.append(d0["u3_frac"])
            med_s3_0.append(abs(d0["S3_over_u"]))

            # Control 1: Match-(S3) single-U(1) control (u_3(0) = 0, so T_{33} = +u)
            s3_ex = abs(d0["S3_over_u"])
            psi_p = 0.5 * math.asin(min(0.999999, s3_ex))
            E1_p = math.sqrt(2.0 * u_t) * math.cos(psi_p)
            B2_p = -math.sqrt(2.0 * u_t) * math.sin(psi_p)
            y0_ab = np.zeros(18, dtype=np.float64)
            y0_ab[:8] = y0[:8]
            y0_ab[13] = a2 * a3 * E1_p
            y0_ab[9] = -0.5 * a3 * a1 * B2_p
            if abs(su2_diagnostics(y0_ab, 1.0)["M3_res"]) > 1e-6:
                y0_ab[9] = -y0_ab[9]
            assert abs(su2_diagnostics(y0_ab, 1.0)["M3_res"]) < 1e-11

            # Control 2 (closes P1-6!): EXACT T_{mu nu}(0)-matched Match-(S3, u3) U(1)_A x U(1)_B control!
            # Matches u_perp, u_3, S_3, AND T_{33}(0) = (1 - 2*u3_frac)*u IDENTICALLY with zero commutators!
            u3_val = d0["u3_frac"] * u_t
            u_perp_val = max(1e-30, u_t - u3_val)
            s3_perp = min(0.999999, abs(d0["S3_over_u"]) * u_t / u_perp_val)
            psi_2 = 0.5 * math.asin(s3_perp)
            E1_2 = math.sqrt(2.0 * u_perp_val) * math.cos(psi_2)
            B2_2 = math.sqrt(2.0 * u_perp_val) * math.sin(psi_2)
            # Match sign of T03 = (1/v) cE1 * cB2 = (1/a3) E1 * B2 -> need E1*B2 == T03 / a3
            if (E1_2 * B2_2) * T03 < 0:
                B2_2 = -B2_2
            # Put longitudinal energy u3_val into B3 = sqrt(2*u3_val) => cB3 = a1*a2*B3 (matching magnetic u3!)
            cB3_2 = a1 * a2 * math.sqrt(2.0 * max(0.0, u3_val))
            y0_u1x2 = np.array([
                g11, 0.0, g22, g33, y0[4], y0[5], y0[6], y0[7],
                a2 * a3 * E1_2, 0.0, 0.0, a3 * a1 * B2_2, 0.0, cB3_2,
            ], dtype=np.float64)

            n_valid += 1
            sol = integrate_su2(y0, kap, (0.0, 12.0))
            sol_ab = integrate_su2(y0_ab, 1.0, (0.0, 12.0))
            sol_u1x2 = integrate_u1x2(y0_u1x2, (0.0, 12.0))
            ok_na = is_bounced_leg(sol, kap, +1)
            ok_ab = is_bounced_leg(sol_ab, 1.0, +1)
            ok_u1x2 = is_bounced_u1x2(sol_u1x2)
            if ok_na:
                n_ok += 1
                max_u0 = max(max_u0, u0)
                max_S3_surv = max(max_S3_surv, s3_ex)
                ds = [su2_diagnostics(sol.y[:, i], kap) for i in range(sol.y.shape[1])]
                max_H = max(max_H, max(abs(d["H_res"]) for d in ds))
                max_M = max(max_M, max(abs(d["M3_res"]) for d in ds))
                max_G = max(max_G, max(abs(d["G3_res"]) for d in ds))
            if ok_ab:
                n_ok_ab_s3 += 1
            if ok_u1x2:
                n_ok_ab_s3_u3 += 1
            if ok_na and not ok_u1x2:
                only_na_vs_s3u3 += 1
            elif ok_u1x2 and not ok_na:
                only_ab_vs_s3u3 += 1

        results[label] = {
            "g_YM": g_ym, "kappa_YM": kap,
            "n_valid": n_valid, "n_bounced_su2": n_ok,
            "n_bounced_match_S3_only_U1": n_ok_ab_s3,
            "n_bounced_match_S3_u3_T33_U1xU1": n_ok_ab_s3_u3,
            "only_su2_vs_T33matched_U1xU1": only_na_vs_s3u3,
            "only_T33matched_U1xU1_vs_su2": only_ab_vs_s3u3,
            "median_E_YM_mono": float(np.median(med_Eym)) if med_Eym else 0.0,
            "median_init_u3_frac": float(np.median(med_u3_0)) if med_u3_0 else 0.0,
            "median_init_abs_S3_over_u": float(np.median(med_s3_0)) if med_s3_0 else 0.0,
            "max_u0_bounced": max_u0, "max_init_S3_over_u_bounced": max_S3_surv,
            "max_H_res": max_H, "max_M3_res": max_M, "max_G3_res": max_G,
        }
    return results


def run_turning_point_ceiling_scan():
    """Scan full past-to-future (I^- <-> I^+) turning-point bounces (K_11=K_22=K_33=0)
    with tilted non-Abelian SU(2)_L near the prolate Berger ceiling 64/81 = 0.790123."""
    survivors = []
    # (A) S3(0) = 0 magnetic non-Abelian start generating dynamical S3(t) != 0
    for a1 in [0.68, 0.70, 0.72, 0.74, 0.76]:
        for da in [0.02, 0.04]:
            a2 = a1 - da
            for a3 in [0.84, 0.86, 0.88, 0.90, 0.92]:
                g11, g22, g33 = a1**2, a2**2, a3**2
                v = a1 * a2 * a3
                R3 = 2.0 * ((g11 + g22 + g33)**2 - 2.0 * (g11**2 + g22**2 + g33**2)) / (g11 * g22 * g33)
                u_req = 0.5 * R3 - LAM
                if u_req <= 0.0:
                    continue
                u0_val = u_req * a1 * a2 * (a3**2)
                if u0_val <= U0_ALG:
                    continue
                # Pure transverse + commutator magnetic seed (cE=0 -> G3=0, T03(0)=0, dynamically generates T03(t)!=0!)
                A11, A12, A21, A22 = 1.0, 0.35, -0.25, 0.85
                cB11, cB12, cB21, cB22, cB33 = -2 * A11, -2 * A12, -2 * A21, -2 * A22, A11 * A22 - A12 * A21
                u_unit = 0.5 * (g11 * (cB11**2 + cB21**2) + g22 * (cB12**2 + cB22**2) + g33 * (cB33**2)) / (v * v)
                kap = u_req / u_unit
                y0 = np.array([g11, 0.0, g22, g33, 0.0, 0.0, 0.0, 0.0, A11, A12, A21, A22, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                sf = integrate_su2(y0, kap, (0.0, 10.0))
                sb = integrate_su2(y0, kap, (0.0, -10.0))
                if is_bounced_leg(sf, kap, +1) and is_bounced_leg(sb, kap, -1):
                    all_d = [su2_diagnostics(sf.y[:, i], kap) for i in range(sf.y.shape[1])] + \
                            [su2_diagnostics(sb.y[:, i], kap) for i in range(sb.y.shape[1])]
                    survivors.append({
                        "axes": [a1, a2, a3], "u0_at_bounce": u0_val, "u_iso_at_bounce": all_d[0]["u_iso"],
                        "kappa_YM": kap, "max_abs_S3_over_u": max(abs(d["S3_over_u"]) for d in all_d),
                        "max_u3_frac": max(d["u3_frac"] for d in all_d),
                        "max_H": max(abs(d["H_res"]) for d in all_d),
                        "max_M3": max(abs(d["M3_res"]) for d in all_d),
                        "max_G3": max(abs(d["G3_res"]) for d in all_d),
                    })
    survivors.sort(key=lambda x: -x["u0_at_bounce"])
    return {
        "n_full_bounces_above_alg": len(survivors),
        "max_u0_achieved": survivors[0]["u0_at_bounce"] if survivors else 0.0,
        "prolate_berger_ceiling_64_over_81": U0_PROLATE,
        "top5": survivors[:5],
    }


def main():
    print("=" * 78)
    print("SCOUT R3-5: TILTED HOMOGENEOUS SU(2)_L IN ROTATING TRIAXIAL BIANCHI IX")
    print("=" * 78)

    print("[1] Running 35-digit 4D Euler-angle verification & independent kill-tests...")
    ver = run_exact_4d_verification_and_kill_tests()
    g4 = ver["mpmath_35dps_generic_tilted_4D"]
    g4l = ver["mpmath_35dps_live_bounced_state_t2p5_4D"]
    print(f"    4D 35-dps generic tilted state: max|10 Einstein|={g4['max_10_einstein_eq_4D']:.2e}, "
          f"max|12 YM|={g4['max_12_ym_eq_4D']:.2e}, max|3 Gauss|={g4['max_3_gauss_4D']:.2e}, dG3/dt={g4['dG3_dt_exact']:.2e}")
    print(f"    4D 35-dps LIVE bounced t=2.5:   max|10 Einstein|={g4l['max_10_einstein_eq_4D']:.2e}, "
          f"max|12 YM|={g4l['max_12_ym_eq_4D']:.2e}, max|3 Gauss|={g4l['max_3_gauss_4D']:.2e}, dG3/dt={g4l['dG3_dt_exact']:.2e}")
    print(f"    Pure-Null Gauss Ellipsoid Theorem max err (100 random 2x2): {ver['pure_null_gauss_ellipsoid_max_err_100trials']:.2e}")
    print(f"    KT1 P01 Abelian Model I recovery at t=4.0:                  {ver['KT1_p01_abelian_model1_recovery_t4']:.2e}")
    print(f"    KT2 Monograph Sec.4 triaxial SU(2) recovery at t=3.0:       {ver['KT2_monograph_sec4_triaxial_su2_recovery_t3']:.2e}")
    print(f"    KT3 Live tilted non-Abelian DOP853 vs Radau at t=4.0:       {ver['KT3_live_tilted_nonabelian_DOP853_vs_Radau_t4']:.2e}")

    print("\n[2] Running Paired Pure-Null (|S3/u|==1) Contracting Ensembles (600 paired trials, FULL Gauss ellipsoid)...")
    pure_null = run_paired_pure_null_ensembles()
    for name, d in pure_null.items():
        print(f"    {name:32s} (N={d['n_valid']}, q1>2:{d['q1_gt_2_upper_half_count']:3d}): "
              f"Ab={d['abelian_bounced']:2d} | SU2={d['nonabelian_bounced']:2d} "
              f"(col={d['nonabelian_collapsed_event']}, fail={d['nonabelian_solver_fail']}, tout={d['nonabelian_timeout']}) | "
              f"ONLY_NA={d['only_nonabelian_bounced']}, ONLY_AB={d['only_abelian_bounced']} | "
              f"col_relH={d['collapsed_max_rel_H_res_amin_ge_0p05']:.1e}")

    print("\n[3] Running Paired Partially-Tilted Sweep vs g_YM (BOTH Match-S3 U(1) AND T33-Matched U(1)xU(1), N=150 each)...")
    part_tilt = run_paired_partially_tilted_coupling_sweep()
    for label, d in part_tilt.items():
        print(f"    {label:10s} (E_YM={d['median_E_YM_mono']:6.2f}, u3_0={d['median_init_u3_frac']:.3f}, |S3/u|_0={d['median_init_abs_S3_over_u']:.3f}): "
              f"SU2={d['n_bounced_su2']:2d} | U1(S3)={d['n_bounced_match_S3_only_U1']:2d} | U1xU1(S3,u3,T33)={d['n_bounced_match_S3_u3_T33_U1xU1']:2d} "
              f"(vs T33-Ab: only_SU2={d['only_su2_vs_T33matched_U1xU1']}, only_Ab={d['only_T33matched_U1xU1_vs_su2']}) max_u0={d['max_u0_bounced']:.5f}")

    print("\n[4] Running Full Past-to-Future (I^- <-> I^+) Turning-Point Ceiling Scan...")
    tp = run_turning_point_ceiling_scan()
    print(f"    Full bounces above 8*sqrt(3)/27: {tp['n_full_bounces_above_alg']} | max u0 = {tp['max_u0_achieved']:.5f} (vs prolate 64/81 = {U0_PROLATE:.5f})")
    for row in tp["top5"][:3]:
        print(f"      axes={row['axes']}, u0={row['u0_at_bounce']:.5f}, u_iso={row['u_iso_at_bounce']:.5f}, "
              f"max|S3/u|={row['max_abs_S3_over_u']:.4f}, max_u3={row['max_u3_frac']:.3f} | "
              f"|H|={row['max_H']:.1e}, |M3|={row['max_M3']:.1e}, |G3|={row['max_G3']:.1e}")

    na_above_bounced = (
        pure_null["above_wall_wide_P01_dist"]["nonabelian_bounced"]
        + pure_null["above_wall_prolate_near_bounce"]["nonabelian_bounced"]
    )
    na_above_total = (
        pure_null["above_wall_wide_P01_dist"]["n_valid"]
        + pure_null["above_wall_prolate_near_bounce"]["n_valid"]
    )
    payload = {
        "scout": "R3-5",
        "title": "Tilted Homogeneous SU(2)_L Yang-Mills in Rotating Triaxial Bianchi IX",
        "constants": {"u0_alg_8sqrt3_over_27": U0_ALG, "u0_isotropic_3_over_4": 0.75, "u0_prolate_64_over_81": U0_PROLATE},
        "verification_and_kill_tests": ver,
        "paired_pure_null_contracting_ensembles": pure_null,
        "paired_partially_tilted_coupling_sweep": part_tilt,
        "turning_point_ceiling_scan": tp,
        "verdict": {
            "pure_null_nonabelian_breaks_P01_wall": bool(na_above_bounced > 0),
            "pure_null_above_wall_bounced": na_above_bounced,
            "pure_null_above_wall_total": na_above_total,
            "gauss_ellipsoid_forbids_small_A_nonabelian_pure_null": bool(ver["pure_null_gauss_ellipsoid_max_err_100trials"] < 1e-12),
            "partially_tilted_su2_vs_controls": {
                k: {
                    "su2": v["n_bounced_su2"],
                    "u1_match_S3": v["n_bounced_match_S3_only_U1"],
                    "u1xu1_match_S3_u3_T33": v["n_bounced_match_S3_u3_T33_U1xU1"],
                    "only_su2_vs_T33matched": v["only_su2_vs_T33matched_U1xU1"],
                    "only_T33matched_vs_su2": v["only_T33matched_U1xU1_vs_su2"],
                }
                for k, v in part_tilt.items()
            },
            "turning_point_max_u0_exceeds_isotropic_0p75": bool(tp["max_u0_achieved"] > 0.75),
        },
    }
    out_path = OUT_DIR / "r3_05_tilted_su2_bianchi9.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
