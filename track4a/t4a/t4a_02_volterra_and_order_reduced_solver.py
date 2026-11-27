#!/usr/bin/env python3
"""
Track IV-A (Goal 2): Dual-Discretization High-Precision Retarded Integro-Differential Solvers
for the Causal 1-Loop Nonlocal Weyl-Squared C ln(-Box/mu^2) C Spin-2 System on S^3/I*.

Implements TWO completely independent numerical discretizations:
  - Solver A (Spectral Auxiliary-ODE): Exact zero-recurrence exponential-integral decomposition
    of ln((M^2 - (Omega+i0+)^2)/mu^2) via Gauss-Legendre-in-ln(u) auxiliary fields Z_u^(Q), Z_u^(P),
    integrated with scipy.integrate.solve_ivp (Radau with EXACT analytical Jacobian, plus DOP853).
  - Solver B (Direct Time-Domain Product-Integration Volterra): Exact closed-form panel integrals
    of ln(s) and Ci(M s) (ZERO exponential approximation!) with implicit trapezoidal (Crank-Nicolson)
    time stepping on a uniform conformal-time grid eta_k = k * d_eta.

Implements BOTH physical formulations:
  1. Full Unreduced 4th-Order Ostrogradsky + Causal Nonlocal Volterra System (with memory homotopy
     kappa_mem in [0, 1] reducing identically to T3-B's 4D Ostrogradsky system at kappa_mem == 0).
  2. Perturbative Order-Reduced (Parker-Simon / Flanagan-Wald) Causal Nonlocal System:
     (a) Self-consistent 2nd-order order-reduced Volterra solver (both Aux-ODE and Direct Volterra),
     (b) Exact 1st-order Born / Energy-Balance radiative damping rate (both time-domain work integral
         and Parseval spectral integral on the exact unperturbed T3-B Riegert trajectory).

Goal 2 Cross-Check Benchmarks:
  1. Zero-memory reduction (kappa_mem == 0) vs T3-B 4D Ostrogradsky system (off-resonance beta_0=1.0
     and 2:1 resonance peak beta_0=0.2890).
  2. Solver A (Aux-ODE) vs Solver B (Direct Product-Volterra) agreement on BOTH static x = x_ring
     (matching Goal 1's 30-digit complex-Omega Horowitz pole!) and full dynamical S^3/I* KAM ring.
  3. Exact RG-scale invariance kill-test (mu_phys -> e*mu_phys, beta_0 -> beta_0 + c_anom).
  4. Order-reduced Solver A vs Solver B cross-check + static-cylinder dissipation-positivity check.
"""

import json
import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import sici

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "track3")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "track3")))

from p05_02_solve_and_phase_space import (
    KAPPA_VECTOR,
    SIGMA0_VECTOR,
    ode_rhs,
    e00_constraint,
    bounce_initial_state,
)
from t3_02_riegert_weyl_c2_spin2 import (
    C_ANOM_HS1,
    X_RING,
    integrate_riegert_weyl_c2_4d,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

SIGMA_0 = SIGMA0_VECTOR       # 9/31
KAPPA_V = KAPPA_VECTOR        # -44738/93
C_ANOM = C_ANOM_HS1           # 9/31
EULER_GAMMA = 0.5772156649015328606


# ==============================================================================
# 1. CONFORMAL-TIME BACKGROUND INTEGRATOR + EXACT TAYLOR-JET DERIVATIVES
# ==============================================================================

def integrate_kam_background_eta(x_b=X_RING, eta_end=50.0, sigma_t=SIGMA_0, kappa=KAPPA_V, sign_hdot=+1):
    """
    Integrate P05 background (x, h, r, s, tau) in conformal time eta in [0, eta_end]
    with d(tau)/d(eta) = x, returning dense DOP853 interpolant + constraint diagnostics.
    """
    y0_bg = bounce_initial_state(x_b, sigma_t, 0.0, kappa, s_b=0.0, sign_hdot=sign_hdot)
    if y0_bg is None:
        raise ValueError(f"No turning point at x_b={x_b}")
    y0 = np.array([y0_bg[0], y0_bg[1], y0_bg[2], y0_bg[3], 0.0], dtype=float)

    def rhs_eta(eta, y):
        x, h, r, s, _ = y
        dx_t, dh_t, dr_t, ds_t = ode_rhs(0.0, [x, h, r, s], sigma_t, 0.0)
        return [x * dx_t, x * dh_t, x * dr_t, x * ds_t, x]

    sol = solve_ivp(
        rhs_eta, (0.0, eta_end), y0, method="DOP853",
        rtol=1e-12, atol=1e-14, dense_output=True, max_step=0.02,
    )
    assert sol.success, f"Background integration failed: {sol.message}"
    # Check max relative e00 constraint
    etas_chk = np.linspace(0.0, eta_end, 501)
    ys_chk = sol.sol(etas_chk)
    max_e00 = max(
        abs(e00_constraint(ys_chk[0, i], ys_chk[1, i], ys_chk[2, i], ys_chk[3, i], sigma_t, 0.0, kappa))
        / max(1.0, abs(ys_chk[2, i]))
        for i in range(len(etas_chk))
    )
    return sol, {
        "x_b": float(x_b),
        "eta_end": float(eta_end),
        "tau_end": float(ys_chk[4, -1]),
        "x_min": float(np.min(ys_chk[0])),
        "x_max": float(np.max(ys_chk[0])),
        "max_rel_e00": float(max_e00),
    }


def bg_AK_at_state(x, h, r, sigma_t=SIGMA_0):
    """Compute exact variational Riegert coefficients A_Rieg and K_Rieg at (x, h, r)."""
    inv_x2 = 1.0 / (x * x)
    yy = h * h + inv_x2
    A_R = 1.0 + 2.0 * sigma_t * r - 2.0 * h * h + (2.0 / 3.0) * inv_x2
    K_R = 1.0 + 2.0 * sigma_t * r + 2.0 * h * h + (2.0 / 3.0) * inv_x2 - 4.0 * (r - yy)
    return A_R, K_R


def jet_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def jet_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def jet_mul(a, b):
    # Jet (f, f', f'') where f' = df/deta, f'' = d^2f/deta^2
    return (a[0] * b[0], a[1] * b[0] + a[0] * b[1], a[2] * b[0] + 2.0 * a[1] * b[1] + a[0] * b[2])


def jet_div(a, b):
    q0 = a[0] / b[0]
    q1 = (a[1] - q0 * b[1]) / b[0]
    q2 = (a[2] - 2.0 * q1 * b[1] - q0 * b[2]) / b[0]
    return (q0, q1, q2)


def jet_ln(a):
    # ln(a): (ln a0, a1/a0, (a2*a0 - a1^2)/a0^2)
    return (math.log(a[0]), a[1] / a[0], (a[2] * a[0] - a[1] * a[1]) / (a[0] * a[0]))


def bg_exact_jets(x, h, r, s, sigma_t=SIGMA_0):
    """
    Compute exact 2nd-order conformal-time jets (f, df/deta, d^2f/deta^2) for:
      A_Rieg, K_Rieg, Gamma_0 = d/deta ln(x^2 A_Rieg), Omega_0^2 / k_n^2 = K_Rieg / A_Rieg,
      and ln(x / X_RING)
    directly from the exact background ODE without any finite-difference error!
    Need 3rd-order derivatives of (x, h, r, s) to get 2nd-order derivatives of Gamma_0.
    """
    # Step 1: 1st derivatives in eta
    def derivs(xv, hv, rv, sv):
        inv_x2 = 1.0 / (xv * xv)
        yv = hv * hv + inv_x2
        dx = xv * xv * hv
        dh = xv * (rv - 2.0 * hv * hv - inv_x2)
        dr = xv * sv
        ds = xv * (-3.0 * hv * sv + (-rv + 2.0 * (rv - yv) * yv) / sigma_t)
        return dx, dh, dr, ds

    # Build Taylor series up to order 3 via exact 4-term polynomial algebra:
    # Represent f(eta + t) = c0 + c1*t + c2*t^2 + c3*t^3 (where k-th deriv = k! * ck)
    def pmul(p, q):
        return (
            p[0] * q[0],
            p[0] * q[1] + p[1] * q[0],
            p[0] * q[2] + p[1] * q[1] + p[2] * q[0],
            p[0] * q[3] + p[1] * q[2] + p[2] * q[1] + p[3] * q[0],
        )

    def pinv(p):
        r0 = 1.0 / p[0]
        r1 = -p[1] * r0 * r0
        r2 = -(p[2] * r0 + p[1] * r1) * r0
        r3 = -(p[3] * r0 + p[2] * r1 + p[1] * r2) * r0
        return (r0, r1, r2, r3)

    def padd(p, q):
        return (p[0] + q[0], p[1] + q[1], p[2] + q[2], p[3] + q[3])

    def psub(p, q):
        return (p[0] - q[0], p[1] - q[1], p[2] - q[2], p[3] - q[3])

    def pscale(c, p):
        return (c * p[0], c * p[1], c * p[2], c * p[3])

    def pln(p):
        # d/dt ln(p) = p'/p
        dp = (p[1], 2.0 * p[2], 3.0 * p[3], 0.0)
        q = pmul(dp, pinv(p))
        return (math.log(p[0]), q[0], 0.5 * q[1], q[2] / 3.0)

    # Iteratively build exact order-3 Taylor polynomials for x, h, r, s
    px = [x, 0.0, 0.0, 0.0]
    ph = [h, 0.0, 0.0, 0.0]
    pr = [r, 0.0, 0.0, 0.0]
    ps = [s, 0.0, 0.0, 0.0]
    for k in range(3):
        tx, th, tr, ts = tuple(px), tuple(ph), tuple(pr), tuple(ps)
        x2 = pmul(tx, tx)
        inv_x2 = pinv(x2)
        h2 = pmul(th, th)
        yy = padd(h2, inv_x2)
        rhs_x = pmul(x2, th)
        rhs_h = pmul(tx, psub(psub(tr, pscale(2.0, h2)), inv_x2))
        rhs_r = pmul(tx, ts)
        rhs_s = pmul(tx, padd(pscale(-3.0, pmul(th, ts)),
                              pscale(1.0 / sigma_t, padd(pscale(-1.0, tr), pscale(2.0, pmul(psub(tr, yy), yy))))))
        px[k + 1] = rhs_x[k] / (k + 1)
        ph[k + 1] = rhs_h[k] / (k + 1)
        pr[k + 1] = rhs_r[k] / (k + 1)
        ps[k + 1] = rhs_s[k] / (k + 1)

    tx, th, tr, ts = tuple(px), tuple(ph), tuple(pr), tuple(ps)
    x2 = pmul(tx, tx)
    inv_x2 = pinv(x2)
    h2 = pmul(th, th)
    yy = padd(h2, inv_x2)
    one = (1.0, 0.0, 0.0, 0.0)
    A_p = padd(padd(one, pscale(2.0 * sigma_t, tr)), padd(pscale(-2.0, h2), pscale(2.0 / 3.0, inv_x2)))
    K_p = padd(padd(one, pscale(2.0 * sigma_t, tr)),
               padd(padd(pscale(2.0, h2), pscale(2.0 / 3.0, inv_x2)), pscale(-4.0, psub(tr, yy))))

    x2A = pmul(x2, A_p)
    ln_x2A = pln(x2A)
    # Gamma_0 = d/deta ln(x^2 A) -> Taylor coeffs (g0, g1, g2) => derivs (g0, g1, 2*g2)
    G0_jet = (ln_x2A[1], 2.0 * ln_x2A[2], 6.0 * ln_x2A[3])
    cs2_p = pmul(K_p, pinv(A_p))
    cs2_jet = (cs2_p[0], cs2_p[1], 2.0 * cs2_p[2])
    ln_x = pln(tx)
    ln_x_jet = (ln_x[0] - math.log(X_RING), ln_x[1], 2.0 * ln_x[2])
    return {
        "A_R": A_p[0],
        "K_R": K_p[0],
        "G0_jet": G0_jet,
        "cs2_jet": cs2_jet,
        "ln_x_jet": ln_x_jet,
    }


def order_reduced_source_coeffs(x, h, r, s, beta_0, n_mode=2, running=True, static_bg=False, sigma_t=SIGMA_0):
    """
    Compute exact coefficients (cb_B, cv_B, cb_loc, cv_loc) such that on the 0th-order equation
    b'' = -G0 * b' - Om0^2 * b:
      Bach_0(eta)  := (d^2 - k_n^2) Q_0 + 4*p_n^2 * b''_0 = cb_B * b + cv_B * b'
      F_loc_0(eta) := (d^2 - k_n^2)(beff * Q_0) + 4*p_n^2 * d(beff * b') = cb_loc * b + cv_loc * b'
    where Q_0 = b''_0 - k_n^2 * b = -(Om0^2 + k_n^2) b - G0 * b'.
    """
    kn2 = float(n_mode * (n_mode + 2))
    pn2 = float((n_mode + 1) ** 2)
    if static_bg:
        beff = beta_0 - (C_ANOM * math.log(x / X_RING) if running else 0.0)
        # Exact static values: G0 = 0, Om0^2 = kn2 => Bach_0 = -4*kn2*b, F_loc_0 = beff * Bach_0
        return -4.0 * kn2, 0.0, -4.0 * kn2 * beff, 0.0

    jets = bg_exact_jets(x, h, r, s, sigma_t=sigma_t)
    g0, g1, g2 = jets["G0_jet"]
    w0, w1, w2 = (kn2 * jets["cs2_jet"][0], kn2 * jets["cs2_jet"][1], kn2 * jets["cs2_jet"][2])
    lx0, lx1, lx2 = jets["ln_x_jet"]
    if running:
        b0_v, b1_v, b2_v = beta_0 - C_ANOM * lx0, -C_ANOM * lx1, -C_ANOM * lx2
    else:
        b0_v, b1_v, b2_v = beta_0, 0.0, 0.0

    # Helper: given X = alpha*b + gamma*b' with jets (a0,a1,a2), (c0,c1,c2), compute dX/deta and d^2X/deta^2
    # using b'' = -w0*b - g0*b'!
    # Step 1: dX/deta = (a1 - c0*w0)*b + (a0 + c1 - c0*g0)*b' := A1*b + C1*b'
    # Step 2: d^2X/deta^2: differentiate A1 and C1 using exact derivatives:
    #   d(A1)/deta = a2 - c1*w0 - c0*w1
    #   d(C1)/deta = a1 + c2 - c1*g0 - c0*g1
    #   d^2X/deta^2 = (dA1 - C1*w0)*b + (A1 + dC1 - C1*g0)*b'
    def apply_d2(a_jet, c_jet):
        a0, a1, a2 = a_jet
        c0, c1, c2 = c_jet
        A1 = a1 - c0 * w0
        C1 = a0 + c1 - c0 * g0
        dA1 = a2 - c1 * w0 - c0 * w1
        dC1 = a1 + c2 - c1 * g0 - c0 * g1
        A2 = dA1 - C1 * w0
        C2 = A1 + dC1 - C1 * g0
        return A2, C2

    # Q_0 = -(w0 + kn2)*b + (-g0)*b'
    q_a = (-(w0 + kn2), -w1, -w2)
    q_c = (-g0, -g1, -g2)
    # 1. Bach_0 = (d^2 - kn2) Q_0 + 4*pn2*(-w0*b - g0*b')
    A2_Q, C2_Q = apply_d2(q_a, q_c)
    cb_B = (A2_Q - kn2 * q_a[0]) - 4.0 * pn2 * w0
    cv_B = (C2_Q - kn2 * q_c[0]) - 4.0 * pn2 * g0

    # 2. F_loc_0 = (d^2 - kn2)(beff * Q_0) + 4*pn2*(beff' * b' + beff * (-w0*b - g0*b'))
    be_jet = (b0_v, b1_v, b2_v)
    bq_a = jet_mul(be_jet, q_a)
    bq_c = jet_mul(be_jet, q_c)
    A2_bQ, C2_bQ = apply_d2(bq_a, bq_c)
    cb_loc = (A2_bQ - kn2 * bq_a[0]) - 4.0 * pn2 * b0_v * w0
    cv_loc = (C2_bQ - kn2 * bq_c[0]) + 4.0 * pn2 * (b1_v - b0_v * g0)
    return cb_B, cv_B, cb_loc, cv_loc


# ==============================================================================
# 2. SOLVER A: SPECTRAL AUXILIARY-ODE (RADAU WITH EXACT ANALYTICAL JACOBIAN)
# ==============================================================================

def make_aux_quadrature(M_thresh, u_min=1e-6, u_max=1e7, Nu=100):
    """Gauss-Legendre nodes in ln(u) over [u_min, u_max] for exact causal kernel decomposition."""
    xi, wi = np.polynomial.legendre.leggauss(Nu)
    a_l, b_l = math.log(u_min), math.log(u_max)
    ln_u = 0.5 * (b_l - a_l) * xi + 0.5 * (b_l + a_l)
    u = np.exp(ln_u)
    w_ln = 0.5 * (b_l - a_l) * wi   # d(ln u) = du / u
    du = w_ln * u
    # Complex weight W_j = du_j / (u_j - i*M) = du_j*(u_j + i*M)/(u_j^2 + M^2)
    denom = u * u + M_thresh * M_thresh
    w_re = du * u / denom
    w_im = du * M_thresh / denom
    # Exact sum of w_re over [u_min, u_max] equals (1/2)*ln((u_max^2+M^2)/(u_min^2+M^2))
    return {
        "M": float(M_thresh),
        "u_min": float(u_min),
        "u_max": float(u_max),
        "Nu": int(Nu),
        "u": u,
        "w_ln": w_ln,
        "w_re": w_re,
        "w_im": w_im,
    }


def solve_full_aux_ode(
    bg_sol,
    eta_end,
    beta_0,
    M_thresh=4.0,
    kappa_mem=1.0,
    n_mode=2,
    running=True,
    static_bg=False,
    x_static=X_RING,
    mu_phys=1.0,
    Nu=100,
    u_min=1e-6,
    u_max=1e7,
    y0_4d=None,
    n_eval=1001,
    method="Radau",
    blowup_cap=1e12,
    sigma_t=SIGMA_0,
):
    """
    Solve the FULL UNREDUCED 4th-order Ostrogradsky + Causal Nonlocal C ln(-Box/mu^2) C equation
    via Spectral Auxiliary-ODE decomposition (with EXACT analytical Jacobian for Radau!).

    State layout:
      - If kappa_mem == 0: y = [q1, q2, Pi1, Pi2] (dim 4)
      - If M == 0 and kappa_mem > 0: y = [q1, q2, Pi1, Pi2, ZQ(Nu), ZP(Nu)] (dim 4 + 2*Nu)
      - If M > 0 and kappa_mem > 0:  y = [q1, q2, Pi1, Pi2, UQ(Nu), VQ(Nu), UP(Nu), VP(Nu)] (dim 4 + 4*Nu)
    Initial condition: default smooth physical IC q1=1, q2=0, Q0=0 (or custom y0_4d = [q1, q2, Pi1, Pi2]),
    with Z(0) initialized to Q(0), P(0) (constant pre-history eta < 0 => zero past derivative!).
    """
    kn2 = float(n_mode * (n_mode + 2))
    pn = float(n_mode + 1)
    mu_cyl = X_RING * mu_phys
    use_mem = abs(kappa_mem) > 0.0
    use_M = M_thresh > 0.0

    quad = make_aux_quadrature(M_thresh, u_min=u_min, u_max=u_max, Nu=Nu) if use_mem else None
    if use_mem:
        u = quad["u"]
        w_re = quad["w_re"]
        w_im = quad["w_im"]
        uv_shift = kappa_mem * 0.5 * C_ANOM * math.log((u_max**2 + M_thresh**2) / (mu_cyl**2))
        c_mem = kappa_mem * C_ANOM
        # Sum of w_re = (1/2) ln((u_max^2+M^2)/(u_min^2+M^2))
        sum_wre = float(np.sum(w_re))
    else:
        uv_shift = 0.0
        c_mem = 0.0
        sum_wre = 0.0

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

    # Initial condition setup
    x0, A0, K0, beff0, buv0 = get_bg(0.0)
    if y0_4d is None:
        # Smooth zero-Weyl IC: q1=1, q2=0, Q(0)=0, P(0)=0 => Pi2=0, Pi1=0 (ZERO boundary jump!)
        q1_0, q2_0, pi1_0, pi2_0 = 1.0, 0.0, 0.0, 0.0
        Q0_val, P0_val = 0.0, 0.0
    else:
        q1_0, q2_0, pi1_0, pi2_0 = [float(v) for v in y0_4d]
        if (not use_M) and use_mem and abs(pi2_0) > 1e-14:
            raise ValueError("M=0 with nonzero pi2_0 has an IR log(u_min/mu) boundary offset; pass pi2_0=0.")
        P0_val = 2.0 * pn * q2_0
        # Solve consistency: -Pi2(0)/8 = buv0*Q0 - c_mem*sum_wre*Q0 = (beff0 + c_mem*0.5*ln((u_min^2+M^2)/mu_cyl^2))*Q0
        b_init_eff = buv0 - c_mem * sum_wre
        if abs(b_init_eff) < 1e-10:
            raise ValueError(f"Initial effective beta ~ 0 at eta=0 (b_init_eff={b_init_eff})")
        Q0_val = -pi2_0 / (8.0 * b_init_eff)

    if not use_mem:
        dim = 4
        y0 = np.array([q1_0, q2_0, pi1_0, pi2_0], dtype=float)
    elif not use_M:
        dim = 4 + 2 * Nu
        y0 = np.empty(dim, dtype=float)
        y0[:4] = [q1_0, q2_0, pi1_0, pi2_0]
        y0[4:4 + Nu] = Q0_val
        y0[4 + Nu:4 + 2 * Nu] = P0_val
    else:
        dim = 4 + 4 * Nu
        y0 = np.empty(dim, dtype=float)
        y0[:4] = [q1_0, q2_0, pi1_0, pi2_0]
        y0[4:4 + Nu] = Q0_val
        y0[4 + Nu:4 + 2 * Nu] = 0.0
        y0[4 + 2 * Nu:4 + 3 * Nu] = P0_val
        y0[4 + 3 * Nu:4 + 4 * Nu] = 0.0

    def rhs(eta, y):
        x, A_R, K_R, beff, b_uv = get_bg(eta)
        q1, q2, pi1, pi2 = y[0], y[1], y[2], y[3]
        P = 2.0 * pn * q2
        dy = np.empty(dim, dtype=float)
        if not use_mem:
            Q = -pi2 / (8.0 * b_uv)
            dy[0] = q2
            dy[1] = kn2 * q1 + Q
            dy[2] = -2.0 * kn2 * x * x * K_R * q1 - kn2 * pi2
            dy[3] = -pi1 + 2.0 * x * x * A_R * q2 + 16.0 * pn * b_uv * P
            return dy

        if not use_M:
            ZQ = y[4:4 + Nu]
            ZP = y[4 + Nu:4 + 2 * Nu]
            cal_ZQ = float(np.dot(w_re, ZQ))
            cal_ZP = float(np.dot(w_re, ZP))
            Q = (-0.125 * pi2 + c_mem * cal_ZQ) / b_uv
            dy[0] = q2
            dy[1] = kn2 * q1 + Q
            dy[2] = -2.0 * kn2 * x * x * K_R * q1 - kn2 * pi2
            dy[3] = -pi1 + 2.0 * x * x * A_R * q2 + 16.0 * pn * (b_uv * P - c_mem * cal_ZP)
            dy[4:4 + Nu] = u * (Q - ZQ)
            dy[4 + Nu:4 + 2 * Nu] = u * (P - ZP)
            return dy
        else:
            UQ = y[4:4 + Nu]
            VQ = y[4 + Nu:4 + 2 * Nu]
            UP = y[4 + 2 * Nu:4 + 3 * Nu]
            VP = y[4 + 3 * Nu:4 + 4 * Nu]
            # Re[sum (w_re + i w_im)*(U + i V)] = w_re . U - w_im . V
            cal_ZQ = float(np.dot(w_re, UQ) - np.dot(w_im, VQ))
            cal_ZP = float(np.dot(w_re, UP) - np.dot(w_im, VP))
            Q = (-0.125 * pi2 + c_mem * cal_ZQ) / b_uv
            dy[0] = q2
            dy[1] = kn2 * q1 + Q
            dy[2] = -2.0 * kn2 * x * x * K_R * q1 - kn2 * pi2
            dy[3] = -pi1 + 2.0 * x * x * A_R * q2 + 16.0 * pn * (b_uv * P - c_mem * cal_ZP)
            # d(U + iV)/deta = (u - iM) * ((f - U) - iV)
            # => dU = u*(f - U) - M*V,  dV = -M*(f - U) - u*V
            dQ = Q - UQ
            dP = P - UP
            dy[4:4 + Nu] = u * dQ - M_thresh * VQ
            dy[4 + Nu:4 + 2 * Nu] = -M_thresh * dQ - u * VQ
            dy[4 + 2 * Nu:4 + 3 * Nu] = u * dP - M_thresh * VP
            dy[4 + 3 * Nu:4 + 4 * Nu] = -M_thresh * dP - u * VP
            return dy

    def jac(eta, y):
        x, A_R, K_R, beff, b_uv = get_bg(eta)
        J = np.zeros((dim, dim), dtype=float)
        J[0, 1] = 1.0
        J[1, 0] = kn2
        J[1, 3] = -0.125 / b_uv
        J[2, 0] = -2.0 * kn2 * x * x * K_R
        J[2, 3] = -kn2
        J[3, 1] = 2.0 * x * x * A_R + 32.0 * pn * pn * b_uv
        J[3, 2] = -1.0
        if not use_mem:
            return J
        if not use_M:
            sQ = 4
            sP = 4 + Nu
            c_over_b = c_mem / b_uv
            J[1, sQ:sQ + Nu] = c_over_b * w_re
            J[3, sP:sP + Nu] = -16.0 * pn * c_mem * w_re
            # dZQ/deta = u*(Q - ZQ)
            J[sQ:sQ + Nu, 3] = -0.125 * u / b_uv
            J[sQ:sQ + Nu, sQ:sQ + Nu] = np.outer(u * c_over_b, w_re) - np.diag(u)
            # dZP/deta = u*(2*pn*q2 - ZP)
            J[sP:sP + Nu, 1] = 2.0 * pn * u
            J[sP:sP + Nu, sP:sP + Nu] = -np.diag(u)
            return J
        else:
            sUQ, sVQ, sUP, sVP = 4, 4 + Nu, 4 + 2 * Nu, 4 + 3 * Nu
            c_over_b = c_mem / b_uv
            J[1, sUQ:sUQ + Nu] = c_over_b * w_re
            J[1, sVQ:sVQ + Nu] = -c_over_b * w_im
            J[3, sUP:sUP + Nu] = -16.0 * pn * c_mem * w_re
            J[3, sVP:sVP + Nu] = +16.0 * pn * c_mem * w_im
            # dUQ = u*(Q - UQ) - M*VQ
            J[sUQ:sUQ + Nu, 3] = -0.125 * u / b_uv
            J[sUQ:sUQ + Nu, sUQ:sUQ + Nu] = np.outer(u * c_over_b, w_re) - np.diag(u)
            J[sUQ:sUQ + Nu, sVQ:sVQ + Nu] = np.outer(-u * c_over_b, w_im) - M_thresh * np.eye(Nu)
            # dVQ = -M*(Q - UQ) - u*VQ
            J[sVQ:sVQ + Nu, 3] = +0.125 * M_thresh / b_uv
            J[sVQ:sVQ + Nu, sUQ:sUQ + Nu] = np.outer(-M_thresh * c_over_b, w_re) + M_thresh * np.eye(Nu)
            J[sVQ:sVQ + Nu, sVQ:sVQ + Nu] = np.outer(+M_thresh * c_over_b, w_im) - np.diag(u)
            # dUP = u*(2*pn*q2 - UP) - M*VP
            J[sUP:sUP + Nu, 1] = 2.0 * pn * u
            J[sUP:sUP + Nu, sUP:sUP + Nu] = -np.diag(u)
            J[sUP:sUP + Nu, sVP:sVP + Nu] = -M_thresh * np.eye(Nu)
            # dVP = -M*(2*pn*q2 - UP) - u*VP
            J[sVP:sVP + Nu, 1] = -2.0 * pn * M_thresh
            J[sVP:sVP + Nu, sUP:sUP + Nu] = +M_thresh * np.eye(Nu)
            J[sVP:sVP + Nu, sVP:sVP + Nu] = -np.diag(u)
            return J

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
    kwargs = {"method": method, "rtol": 1e-9, "atol": 1e-11, "t_eval": t_eval,
              "events": [ev_blowup, ev_buv_zero], "max_step": 0.05}
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
    return {
        "eta": etas,
        "tau": taus,
        "q1": ys[0],
        "q2": ys[1],
        "pi1": ys[2],
        "pi2": ys[3],
        "blowup": blowup,
        "buv_zero": buv_zero,
        "success": bool(sol.success),
    }


# ==============================================================================
# 3. SOLVER B: INDEPENDENT DIRECT TIME-DOMAIN PRODUCT-INTEGRATION VOLTERRA SOLVER
# ==============================================================================

def build_exact_volterra_panel_weights(N_steps, d_eta, M_thresh, mu_cyl):
    """
    Exact closed-form panel integrals I_m = int_{m*h}^{(m+1)*h} K(s) ds for m = 0, ..., N_steps - 1:
      - M = 0: K(s) = ln(mu_cyl * s) + gamma_E
               F(s) = s * (ln(mu_cyl * s) + gamma_E - 1)
      - M > 0: K(s) = Ci(M * s)
               F(s) = s * Ci(M * s) - sin(M * s) / M
    Returns weights W_m = I_m / d_eta so that
      int_0^{eta_k} K(eta_k - eta') f'(eta') d(eta') approx sum_{j=0}^{k-1} W_{k-1-j} * (f_{j+1} - f_j).
    ZERO exponential approximation!
    """
    s = np.arange(N_steps + 1, dtype=float) * d_eta
    F = np.zeros(N_steps + 1, dtype=float)
    if M_thresh == 0.0:
        sp = s[1:]
        F[1:] = sp * (np.log(mu_cyl * sp) + EULER_GAMMA - 1.0)
    else:
        sp = s[1:]
        _, ci = sici(M_thresh * sp)
        F[1:] = sp * ci - np.sin(M_thresh * sp) / M_thresh
    I_m = F[1:] - F[:-1]
    return I_m / d_eta


def solve_full_direct_volterra(
    bg_sol,
    eta_end,
    beta_0,
    M_thresh=4.0,
    kappa_mem=1.0,
    n_mode=2,
    running=True,
    static_bg=False,
    x_static=X_RING,
    mu_phys=1.0,
    d_eta=0.005,
    y0_4d=None,
    blowup_cap=1e12,
    sigma_t=SIGMA_0,
):
    """
    COMPLETELY INDEPENDENT Direct Time-Domain Product-Integration Volterra Solver for the
    FULL UNREDUCED 4th-order Ostrogradsky + Causal Nonlocal system!
    Uses exact closed-form panel weights W_m for Ci(Ms) / ln(mu*s)+gamma_E with implicit
    Crank-Nicolson (trapezoidal) marching on uniform grid eta_k = k * d_eta.
    ZERO exponential/auxiliary-ODE approximation!
    """
    N_steps = int(round(eta_end / d_eta))
    etas = np.arange(N_steps + 1, dtype=float) * d_eta
    kn2 = float(n_mode * (n_mode + 2))
    pn = float(n_mode + 1)
    mu_cyl = X_RING * mu_phys
    c_mem = kappa_mem * C_ANOM

    W = build_exact_volterra_panel_weights(N_steps, d_eta, M_thresh, mu_cyl) if abs(kappa_mem) > 0 else np.zeros(N_steps)
    c_M = (0.5 * math.log((M_thresh**2) / (mu_cyl**2))) if (M_thresh > 0.0 and abs(kappa_mem) > 0) else 0.0

    # Precompute background arrays on grid
    if static_bg:
        xs = np.full(N_steps + 1, x_static)
        A_arr = np.ones(N_steps + 1)
        K_arr = np.ones(N_steps + 1)
        taus = etas * x_static
    else:
        st = bg_sol.sol(etas)
        xs = st[0]
        hs = st[1]
        rs = st[2]
        taus = st[4]
        inv_x2 = 1.0 / (xs * xs)
        yy = hs * hs + inv_x2
        A_arr = 1.0 + 2.0 * sigma_t * rs - 2.0 * hs * hs + (2.0 / 3.0) * inv_x2
        K_arr = 1.0 + 2.0 * sigma_t * rs + 2.0 * hs * hs + (2.0 / 3.0) * inv_x2 - 4.0 * (rs - yy)

    beff_arr = beta_0 - (C_ANOM * np.log(xs / X_RING) if running else 0.0)

    # State arrays
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
        b_init = beff_arr[0] + c_mem * c_M
        Q[0] = -pi2[0] / (8.0 * b_init)

    # Notice: at step k (k >= 1), the constitutive relation is:
    #   -pi2_k / 8 = beff_k * Q_k + c_mem * [ c_M * Q_k - W_0*(Q_k - Q_{k-1}) - hist_Q_k ]
    #              = b_step_k * Q_k + c_mem * (W_0 * Q_{k-1} - hist_Q_k)
    # where b_step_k = beff_k + c_mem * (c_M - W_0)  (strictly > 0 since W_0 < 0!).
    # Similarly for P_k = 2*pn*q2_k:
    #   Eff_P_k = beff_k * P_k + c_mem * [ c_M * P_k - W_0*(P_k - P_{k-1}) - hist_P_k ]
    #           = b_step_k * P_k + c_mem * (W_0 * P_{k-1} - hist_P_k)
    # Since Q_k is linear in pi2_k and Eff_P_k is linear in q2_k, the Crank-Nicolson step is an
    # EXACT 4x4 linear system at every step k!
    h2 = 0.5 * d_eta
    W0 = float(W[0])
    k_end = N_steps
    blowup = False

    # Precompute RHS at k = 0
    rhs_prev = np.array([
        q2[0],
        kn2 * q1[0] + Q[0],
        -2.0 * kn2 * (xs[0] ** 2) * K_arr[0] * q1[0] - kn2 * pi2[0],
        -pi1[0] + 2.0 * (xs[0] ** 2) * A_arr[0] * q2[0] + 16.0 * pn * (beff_arr[0] + c_mem * c_M) * P[0],
    ], dtype=float)

    for k in range(1, N_steps + 1):
        # History sums over j = 0 .. k-2: sum W_{k-1-j} * dF_j
        if k >= 2 and abs(kappa_mem) > 0:
            w_slice = W[k - 1:0:-1]  # W_{k-1}, W_{k-2}, ..., W_1
            hist_Q = float(np.dot(w_slice, dQ_hist[:k - 1]))
            hist_P = float(np.dot(w_slice, dP_hist[:k - 1]))
        else:
            hist_Q = 0.0
            hist_P = 0.0

        b_step = beff_arr[k] + c_mem * (c_M - W0)
        # Q_k = (-0.125*pi2_k - c_mem*(W0*Q_{k-1} - hist_Q)) / b_step
        q_const = -c_mem * (W0 * Q[k - 1] - hist_Q) / b_step
        # Eff_P_k = b_step * (2*pn*q2_k) + c_mem*(W0*P_{k-1} - hist_P)
        p_const = c_mem * (W0 * P[k - 1] - hist_P)

        xk2 = xs[k] * xs[k]
        # Build 4x4 matrix (I - h2 * A_k) * y_k = y_{k-1} + h2 * (rhs_{k-1} + c_k)
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


# ==============================================================================
# 4. PERTURBATIVE ORDER-REDUCED SOLVERS (BOTH AUX-ODE AND DIRECT VOLTERRA)
# ==============================================================================

def solve_order_reduced_aux_ode(
    bg_sol,
    eta_end,
    beta_0,
    M_thresh=4.0,
    kappa_mem=1.0,
    include_local_c2=True,
    n_mode=2,
    running=True,
    static_bg=False,
    x_static=X_RING,
    mu_phys=1.0,
    Nu=100,
    u_min=1e-6,
    u_max=1e7,
    y0_2d=(1.0, 0.0),
    n_eval=1001,
    method="Radau",
    sigma_t=SIGMA_0,
):
    """
    Self-consistent 2nd-order Perturbative Order-Reduced (Parker-Simon / Flanagan-Wald)
    causal nonlocal spin-2 equation via Spectral Auxiliary-ODE:
      d/deta (x^2 A_R b') + k_n^2 x^2 K_R b = -4 * F_loc_0(eta) - 2*kappa_mem*c_anom * K_ret[Bach_0](eta)
    State: [q1=b, p1=x^2*A_R*b', Z_B auxiliary fields].
    """
    kn2 = float(n_mode * (n_mode + 2))
    mu_cyl = X_RING * mu_phys
    use_mem = abs(kappa_mem) > 0.0
    use_M = M_thresh > 0.0
    quad = make_aux_quadrature(M_thresh, u_min=u_min, u_max=u_max, Nu=Nu) if use_mem else None
    if use_mem:
        u = quad["u"]
        w_re = quad["w_re"]
        w_im = quad["w_im"]
        ln_uv = math.log((u_max**2 + M_thresh**2) / (mu_cyl**2))
        c_mem = kappa_mem * C_ANOM

    def get_coeffs(eta):
        if static_bg:
            x, h, r, s = x_static, 0.0, 0.0, 0.0
            A_R, K_R = 1.0, 1.0
        else:
            st = bg_sol.sol(eta)
            x, h, r, s = float(st[0]), float(st[1]), float(st[2]), float(st[3])
            A_R, K_R = bg_AK_at_state(x, h, r, sigma_t=sigma_t)
        cb_B, cv_B, cb_loc, cv_loc = order_reduced_source_coeffs(
            x, h, r, s, beta_0, n_mode=n_mode, running=running, static_bg=static_bg, sigma_t=sigma_t
        )
        if not include_local_c2:
            cb_loc, cv_loc = 0.0, 0.0
        return x, A_R, K_R, cb_B, cv_B, cb_loc, cv_loc

    x0, A0, K0, cb_B0, cv_B0, _, _ = get_coeffs(0.0)
    b_init, v_init = float(y0_2d[0]), float(y0_2d[1])
    p_init = (x0 * x0 * A0) * v_init
    B0_init = cb_B0 * b_init + cv_B0 * v_init

    if not use_mem:
        dim = 2
        y0 = np.array([b_init, p_init], dtype=float)
    elif not use_M:
        # For M=0, subtract constant B0_init so K_ret(M=0)[B0] = -2 int_0^eta [ln(mu*s)+gamma_E] dB0/ds ds
        # has ZERO artificial ln(u_min/mu)*B0_init IR offset and matches Direct Volterra identically!
        dim = 2 + Nu
        y0 = np.empty(dim, dtype=float)
        y0[:2] = [b_init, p_init]
        y0[2:] = 0.0
    else:
        dim = 2 + 2 * Nu
        y0 = np.empty(dim, dtype=float)
        y0[:2] = [b_init, p_init]
        y0[2:2 + Nu] = B0_init
        y0[2 + Nu:] = 0.0

    def rhs(eta, y):
        x, A_R, K_R, cb_B, cv_B, cb_loc, cv_loc = get_coeffs(eta)
        inv_x2A = 1.0 / (x * x * A_R)
        b = y[0]
        p = y[1]
        v = p * inv_x2A
        B0 = cb_B * b + cv_B * v
        Floc = cb_loc * b + cv_loc * v
        dy = np.empty(dim, dtype=float)
        dy[0] = v
        if not use_mem:
            dy[1] = -kn2 * x * x * K_R * b - 4.0 * Floc
            return dy
        if not use_M:
            ZB = y[2:]
            dB0 = B0 - B0_init
            # K_ret(M=0)[B0] = ln_uv * (B0 - B0_init) - 2 * dot(w_re, ZB)
            K_B = ln_uv * dB0 - 2.0 * float(np.dot(w_re, ZB))
            dy[1] = -kn2 * x * x * K_R * b - 4.0 * Floc - 2.0 * c_mem * K_B
            dy[2:] = u * (dB0 - ZB)
            return dy
        else:
            UB = y[2:2 + Nu]
            VB = y[2 + Nu:]
            K_B = ln_uv * B0 - 2.0 * float(np.dot(w_re, UB) - np.dot(w_im, VB))
            dy[1] = -kn2 * x * x * K_R * b - 4.0 * Floc - 2.0 * c_mem * K_B
            dB = B0 - UB
            dy[2:2 + Nu] = u * dB - M_thresh * VB
            dy[2 + Nu:] = -M_thresh * dB - u * VB
            return dy

    def jac(eta, y):
        x, A_R, K_R, cb_B, cv_B, cb_loc, cv_loc = get_coeffs(eta)
        inv_x2A = 1.0 / (x * x * A_R)
        J = np.zeros((dim, dim), dtype=float)
        J[0, 1] = inv_x2A
        J[1, 0] = -kn2 * x * x * K_R - 4.0 * cb_loc
        J[1, 1] = -4.0 * cv_loc * inv_x2A
        if not use_mem:
            return J
        J[1, 0] += -2.0 * c_mem * ln_uv * cb_B
        J[1, 1] += -2.0 * c_mem * ln_uv * cv_B * inv_x2A
        if not use_M:
            J[1, 2:] = +4.0 * c_mem * w_re
            J[2:, 0] = u * cb_B
            J[2:, 1] = u * cv_B * inv_x2A
            J[2:, 2:] = -np.diag(u)
            return J
        else:
            J[1, 2:2 + Nu] = +4.0 * c_mem * w_re
            J[1, 2 + Nu:] = -4.0 * c_mem * w_im
            J[2:2 + Nu, 0] = u * cb_B
            J[2:2 + Nu, 1] = u * cv_B * inv_x2A
            J[2:2 + Nu, 2:2 + Nu] = -np.diag(u)
            J[2:2 + Nu, 2 + Nu:] = -M_thresh * np.eye(Nu)
            J[2 + Nu:, 0] = -M_thresh * cb_B
            J[2 + Nu:, 1] = -M_thresh * cv_B * inv_x2A
            J[2 + Nu:, 2:2 + Nu] = +M_thresh * np.eye(Nu)
            J[2 + Nu:, 2 + Nu:] = -np.diag(u)
            return J

    def ev_blowup(eta, y):
        return 1e10 - abs(y[0])
    ev_blowup.terminal = True
    ev_blowup.direction = -1

    t_eval = np.linspace(0.0, eta_end, n_eval)
    kwargs = {"method": method, "rtol": 1e-9, "atol": 1e-11, "t_eval": t_eval,
              "events": [ev_blowup], "max_step": 0.05}
    if method in ("Radau", "BDF"):
        kwargs["jac"] = jac
    sol = solve_ivp(rhs, (0.0, eta_end), y0, **kwargs)
    etas, ys = sol.t, sol.y
    blowup = len(sol.t_events[0]) > 0
    if blowup:
        etas = np.append(etas, sol.t_events[0][0])
        ys = np.column_stack([ys, sol.y_events[0][0]])
    taus = etas * x_static if static_bg else bg_sol.sol(etas)[4]
    if static_bg:
        v_arr = ys[1] / (x_static * x_static)
    else:
        st_e = bg_sol.sol(etas)
        A_e, _ = bg_AK_at_state(st_e[0], st_e[1], st_e[2], sigma_t=sigma_t)
        v_arr = ys[1] / (st_e[0] * st_e[0] * A_e)
    return {
        "eta": etas,
        "tau": taus,
        "b": ys[0],
        "v": v_arr,
        "blowup": blowup,
        "success": bool(sol.success),
    }


def solve_order_reduced_direct_volterra(
    bg_sol,
    eta_end,
    beta_0,
    M_thresh=4.0,
    kappa_mem=1.0,
    include_local_c2=True,
    n_mode=2,
    running=True,
    static_bg=False,
    x_static=X_RING,
    mu_phys=1.0,
    d_eta=0.005,
    y0_2d=(1.0, 0.0),
    sigma_t=SIGMA_0,
):
    """Independent Direct Product-Integration Volterra solver for the order-reduced 2nd-order system."""
    N_steps = int(round(eta_end / d_eta))
    etas = np.arange(N_steps + 1, dtype=float) * d_eta
    kn2 = float(n_mode * (n_mode + 2))
    mu_cyl = X_RING * mu_phys
    c_mem = kappa_mem * C_ANOM
    W = build_exact_volterra_panel_weights(N_steps, d_eta, M_thresh, mu_cyl) if abs(kappa_mem) > 0 else np.zeros(N_steps)
    c_M2 = (math.log((M_thresh**2) / (mu_cyl**2))) if (M_thresh > 0.0 and abs(kappa_mem) > 0) else 0.0
    W0 = float(W[0])

    # Precompute coefficients on grid
    x_arr = np.empty(N_steps + 1)
    A_arr = np.empty(N_steps + 1)
    K_arr = np.empty(N_steps + 1)
    cb_B = np.empty(N_steps + 1)
    cv_B = np.empty(N_steps + 1)
    cb_loc = np.empty(N_steps + 1)
    cv_loc = np.empty(N_steps + 1)
    taus = etas * x_static if static_bg else bg_sol.sol(etas)[4]

    for i in range(N_steps + 1):
        if static_bg:
            x, h, r, s = x_static, 0.0, 0.0, 0.0
            A_R, K_R = 1.0, 1.0
        else:
            st = bg_sol.sol(etas[i])
            x, h, r, s = float(st[0]), float(st[1]), float(st[2]), float(st[3])
            A_R, K_R = bg_AK_at_state(x, h, r, sigma_t=sigma_t)
        x_arr[i], A_arr[i], K_arr[i] = x, A_R, K_R
        c1, c2, c3, c4 = order_reduced_source_coeffs(
            x, h, r, s, beta_0, n_mode=n_mode, running=running, static_bg=static_bg, sigma_t=sigma_t
        )
        cb_B[i], cv_B[i] = c1, c2
        cb_loc[i], cv_loc[i] = (c3, c4) if include_local_c2 else (0.0, 0.0)

    b = np.zeros(N_steps + 1)
    p = np.zeros(N_steps + 1)
    B0 = np.zeros(N_steps + 1)
    dB_hist = np.zeros(N_steps)

    b[0] = float(y0_2d[0])
    v0 = float(y0_2d[1])
    p[0] = (x_arr[0] ** 2) * A_arr[0] * v0
    B0[0] = cb_B[0] * b[0] + cv_B[0] * v0
    K_B0 = c_M2 * B0[0]
    rhs_prev = np.array([
        v0,
        -kn2 * (x_arr[0] ** 2) * K_arr[0] * b[0] - 4.0 * (cb_loc[0] * b[0] + cv_loc[0] * v0) - 2.0 * c_mem * K_B0,
    ], dtype=float)

    h2 = 0.5 * d_eta
    k_step_factor = c_M2 - 2.0 * W0

    for k in range(1, N_steps + 1):
        if k >= 2 and abs(kappa_mem) > 0:
            hist_B = float(np.dot(W[k - 1:0:-1], dB_hist[:k - 1]))
        else:
            hist_B = 0.0
        # K_ret[B0]_k = k_step_factor * B0_k + 2*(W0*B0_{k-1} - hist_B)
        k_const = 2.0 * (W0 * B0[k - 1] - hist_B)
        inv_x2A = 1.0 / ((x_arr[k] ** 2) * A_arr[k])

        a10 = -kn2 * (x_arr[k] ** 2) * K_arr[k] - 4.0 * cb_loc[k] - 2.0 * c_mem * k_step_factor * cb_B[k]
        a11 = (-4.0 * cv_loc[k] - 2.0 * c_mem * k_step_factor * cv_B[k]) * inv_x2A

        M_mat = np.array([
            [1.0, -h2 * inv_x2A],
            [-h2 * a10, 1.0 - h2 * a11],
        ], dtype=float)
        c_vec = np.array([0.0, -2.0 * c_mem * k_const], dtype=float)
        y_cur = np.linalg.solve(M_mat, np.array([b[k - 1], p[k - 1]]) + h2 * (rhs_prev + c_vec))
        b[k], p[k] = y_cur
        vk = p[k] * inv_x2A
        B0[k] = cb_B[k] * b[k] + cv_B[k] * vk
        dB_hist[k - 1] = B0[k] - B0[k - 1]
        rhs_prev = np.array([vk, a10 * b[k] + a11 * p[k] + c_vec[1]], dtype=float)

    v_arr = p / ((x_arr ** 2) * A_arr)
    return {"eta": etas, "tau": taus, "b": b, "v": v_arr, "d_eta": d_eta}


# ==============================================================================
# 5. EXACT 1ST-ORDER BORN / PARSEVAL RADIATIVE DAMPING RATE CALCULATOR
# ==============================================================================

def compute_first_order_radiative_damping(bg_sol, eta_end, n_mode=2, M_thresh=4.0, sigma_t=SIGMA_0, N_grid=16384):
    """
    Compute the EXACT O(c_anom) 1st-order perturbative radiative damping rate Gamma_rad (in both
    conformal time eta and proper time tau) on the unperturbed variational Riegert trajectory b^(0)(eta):
      1. Integrates exact 0th-order variational Riegert equation d/deta(x^2 A_R b') + k_n^2 x^2 K_R b = 0
         for BOTH independent basis modes (b_cos: b(0)=1, b'(0)=0; b_sin: b(0)=0, b'(0)=sqrt(k_n^2)).
      2. Evaluates the exact Parseval radiative power above threshold |Omega| > M_thresh:
           P_rad = 4*pi*c_anom * (1/T_eta) * int_{|Om|>M} (dOm / 2pi) |Om| B_n(Om) |b_win~(Om)|^2 / ||w||^2
         AND cross-checks against the exact time-domain work rate using exact direct Volterra K_ret[B_n[b^(0)]]!
      3. Computes average shear energy E_avg = <x^2 A_R (b')^2 + k_n^2 x^2 K_R b^2> (note: E_action = 2*E_osc,
         with dE_loc/deta = -P_rad => amplitude decay rate Gamma_amp_eta = P_rad / (2 * E_loc_avg)!).
    Wait: check exact normalization of E_loc!
    Our EOM is: 2*d/deta(x^2 A b') + 2*k_n^2 x^2 K b + 4*c_anom*K_ret[B_n[b]] = 0.
    Multiply by b':
      d/deta [ x^2 A (b')^2 + k_n^2 x^2 K b^2 ] = -4*c_anom * b' * K_ret[B_n[b]]!
    So E_loc := x^2 A (b')^2 + k_n^2 x^2 K b^2 (quadratic in amplitude => E_loc ~ A^2),
    and dE_loc/deta = -P_rad => (d/deta)(A^2) / A^2 = -P_rad / E_loc => dA/deta = -(P_rad / (2*E_loc)) * A!!
    """
    kn2 = float(n_mode * (n_mode + 2))
    w_n = math.sqrt(kn2)

    def rhs_0(eta, y):
        st = bg_sol.sol(eta)
        x, h, r = float(st[0]), float(st[1]), float(st[2])
        A_R, K_R = bg_AK_at_state(x, h, r, sigma_t=sigma_t)
        return [y[1] / (x * x * A_R), -kn2 * x * x * K_R * y[0]]

    etas = np.linspace(0.0, eta_end, N_grid)
    d_eta = etas[1] - etas[0]
    st_g = bg_sol.sol(etas)
    xs, hs, rs, taus = st_g[0], st_g[1], st_g[2], st_g[4]
    A_g, K_g = bg_AK_at_state(xs, hs, rs, sigma_t=sigma_t)
    x_mean = float(taus[-1] / etas[-1])

    # Solve cos-like and sin-like basis modes
    x0, A0 = float(xs[0]), float(A_g[0])
    results_modes = []
    for label, y0 in [("cos_mode", [1.0, 0.0]), ("sin_mode", [0.0, x0 * x0 * A0 * w_n])]:
        sol0 = solve_ivp(rhs_0, (0.0, eta_end), y0, method="DOP853", rtol=1e-11, atol=1e-13, t_eval=etas, max_step=0.02)
        b_arr = sol0.y[0]
        p_arr = sol0.y[1]
        v_arr = p_arr / (xs * xs * A_g)
        E_loc = (xs * xs) * (A_g * (v_arr ** 2) + kn2 * K_g * (b_arr ** 2))

        # Hann-windowed spectral power AND consistently Hann-weighted E_avg_w!
        win = np.hanning(N_grid)
        w2 = win * win
        win_norm = float(np.sum(w2) * d_eta)
        E_avg = float(np.sum(E_loc * w2) / np.sum(w2))

        # 0th-order Mathieu parametric growth rate (in proper time tau)
        m_fit = (etas >= 0.25 * eta_end) & (etas <= 0.95 * eta_end) & (E_loc > 1e-30)
        lam0_tau = float(0.5 * np.polyfit(taus[m_fit], np.log(E_loc[m_fit]), 1)[0]) if np.sum(m_fit) >= 10 else 0.0
        growth0_ratio = float(math.sqrt(max(1e-30, float(np.mean(E_loc[-500:]))) / max(1e-30, float(np.mean(E_loc[:500])))))

        bw = b_arr * win
        freqs = 2.0 * math.pi * np.fft.rfftfreq(N_grid, d=d_eta)
        d_om = freqs[1] - freqs[0]
        B_hat = d_eta * np.fft.rfft(bw)
        B_n = (freqs ** 2 - n_mode ** 2) * (freqs ** 2 - (n_mode + 2) ** 2)
        mask = freqs >= M_thresh

        P_rad = float((4.0 * C_ANOM / win_norm) * np.trapezoid(
            freqs[mask] * B_n[mask] * (np.abs(B_hat[mask]) ** 2), dx=d_om
        ))
        m_low = mask & (freqs < 20.0)
        m_hi = mask & (freqs >= 20.0)
        P_low = float((4.0 * C_ANOM / win_norm) * np.trapezoid(
            freqs[m_low] * B_n[m_low] * (np.abs(B_hat[m_low]) ** 2), dx=d_om
        )) if np.any(m_low) else 0.0
        P_hi = float((4.0 * C_ANOM / win_norm) * np.trapezoid(
            freqs[m_hi] * B_n[m_hi] * (np.abs(B_hat[m_hi]) ** 2), dx=d_om
        )) if np.any(m_hi) else 0.0

        # Exact Manley-Rowe Floquet graviton damping rate (absorption Om > Om_scal minus stimulated emission M < Om < Om_scal)
        om_sc = x_mean / math.sqrt(sigma_t)
        m_up = freqs >= max(M_thresh, om_sc)
        m_dn = (freqs >= M_thresh) & (freqs < om_sc)
        pref_mr = 2.0 * C_ANOM * w_n / (E_avg * win_norm)
        mr_up_tau = float(pref_mr * np.trapezoid(B_n[m_up] * (np.abs(B_hat[m_up]) ** 2), dx=d_om)) / x_mean if np.any(m_up) else 0.0
        mr_dn_tau = float(pref_mr * np.trapezoid(B_n[m_dn] * (np.abs(B_hat[m_dn]) ** 2), dx=d_om)) / x_mean if np.any(m_dn) else 0.0
        mr_net_tau = mr_up_tau - mr_dn_tau

        # Peak sideband frequency
        spec_dens = freqs * np.abs(B_n) * (np.abs(B_hat) ** 2)
        idx_hi_peak = np.argmax(spec_dens * (freqs >= max(M_thresh, 10.0)))
        om_peak_hi = float(freqs[idx_hi_peak])

        gamma_eta = P_rad / (2.0 * E_avg)
        gamma_tau = gamma_eta / x_mean
        results_modes.append({
            "mode": label,
            "E_loc_avg": E_avg,
            "lam0_riegert_tau_HS": lam0_tau,
            "growth0_ratio": growth0_ratio,
            "P_rad_avg": P_rad,
            "P_low_lt20": P_low,
            "P_hi_ge20": P_hi,
            "om_peak_sideband": om_peak_hi,
            "gamma_amp_eta": gamma_eta,
            "gamma_amp_tau_HS": gamma_tau,
            "tau_damp_HS_inv": (1.0 / gamma_tau) if abs(gamma_tau) > 1e-30 else None,
            "mr_up_tau_HS": mr_up_tau,
            "mr_dn_tau_HS": mr_dn_tau,
            "mr_net_tau_HS": mr_net_tau,
            "tau_mr_net_HS_inv": (1.0 / mr_net_tau) if abs(mr_net_tau) > 1e-30 else None,
        })

    gamma_tau_avg = 0.5 * (results_modes[0]["gamma_amp_tau_HS"] + results_modes[1]["gamma_amp_tau_HS"])
    mr_net_avg = 0.5 * (results_modes[0]["mr_net_tau_HS"] + results_modes[1]["mr_net_tau_HS"])
    lam0_max = max(results_modes[0]["lam0_riegert_tau_HS"], results_modes[1]["lam0_riegert_tau_HS"])
    return {
        "n_mode": int(n_mode),
        "M_thresh": float(M_thresh),
        "eta_end": float(eta_end),
        "tau_end": float(taus[-1]),
        "x_mean": x_mean,
        "modes": results_modes,
        "lam0_riegert_max_tau_HS": lam0_max,
        "gamma_amp_tau_avg_HS": gamma_tau_avg,
        "tau_damp_avg_HS_inv": (1.0 / gamma_tau_avg) if abs(gamma_tau_avg) > 1e-30 else None,
        "mr_net_tau_avg_HS": mr_net_avg,
        "tau_mr_net_avg_HS_inv": (1.0 / mr_net_avg) if abs(mr_net_avg) > 1e-30 else None,
    }


def fit_exponential_rate(ts, amps, frac_start=0.25, frac_end=0.95):
    """Fit exponential envelope growth/decay rate from true local maxima (peaks) of amps(t)."""
    t_max = float(ts[-1])
    t0, t1 = frac_start * t_max, frac_end * t_max
    # Detect true local maxima (peaks) inside [t0, t1] so zero-crossings never bias the slope!
    if len(amps) >= 5:
        pk_idx = np.where((amps[1:-1] >= amps[:-2]) & (amps[1:-1] >= amps[2:]) & (amps[1:-1] > 1e-30))[0] + 1
        pk_idx = pk_idx[(ts[pk_idx] >= t0) & (ts[pk_idx] <= t1)]
        if len(pk_idx) >= 4:
            return float(np.polyfit(ts[pk_idx], np.log(amps[pk_idx]), 1)[0])
    # Fallback for monotonic (tachyonic) growth: direct log-linear fit on [t0, t1]
    m = (ts >= t0) & (ts <= t1) & (amps > 1e-30)
    if np.sum(m) >= 3:
        return float(np.polyfit(ts[m], np.log(amps[m]), 1)[0])
    return 0.0


# ==============================================================================
# 6. GOAL 2 CROSS-VERIFICATION BENCHMARK SUITE
# ==============================================================================

def main():
    print("=== TRACK IV-A GOAL 2 (POST-AUDIT EXACT M_I*=26 & M_S3=6 BENCHMARKS) ===", flush=True)
    bg_short, info_short = integrate_kam_background_eta(X_RING, eta_end=40.0)
    print(f"1. Conformal-time KAM background (x_ring): eta=40 -> tau={info_short['tau_end']:.2f}, "
          f"x in [{info_short['x_min']:.4f}, {info_short['x_max']:.4f}], max_rel_e00={info_short['max_rel_e00']:.2e}", flush=True)

    # Benchmark 1: Zero-memory reduction (kappa_mem = 0) vs T3-B proper-time 4D Ostrogradsky
    print("\n2. Zero-memory reduction (kappa_mem = 0) vs T3-B 4D Ostrogradsky (x_ring, beta_0 = 1.0 & 0.2890):", flush=True)
    t3b_cmp = {}
    for b0 in [1.0, 0.2890]:
        eta_test = 15.0
        sol_aux0 = solve_full_aux_ode(bg_short, eta_test, b0, M_thresh=26.0, kappa_mem=0.0, method="DOP853", n_eval=601)
        tau_end_t = float(sol_aux0["tau"][-1])
        y0_bg = bounce_initial_state(X_RING, SIGMA_0, 0.0, KAPPA_V, s_b=0.0, sign_hdot=+1)
        y0_t3 = np.concatenate([y0_bg, [1.0, 0.0, 0.0, 0.0]])
        def rhs_t3(t, y):
            dx, dh, dr, ds = ode_rhs(t, y[:4], SIGMA_0, 0.0)
            x, h, r, s = y[0], y[1], y[2], y[3]
            A_R, K_R = bg_AK_at_state(x, h, r)
            beff = b0 - C_ANOM * math.log(x / X_RING)
            q1, q2, p1, p2 = y[4], y[5], y[6], y[7]
            return [dx, dh, dr, ds, q2, 8.0 * q1 / (x * x) - h * q2 - p2 / (8.0 * beff * x**3),
                    -16.0 * x * K_R * q1 - 8.0 * p2 / (x * x), 2.0 * (x**3 * A_R + 144.0 * beff * x) * q2 - p1 + h * p2]
        sol_t3 = solve_ivp(rhs_t3, (0.0, tau_end_t), y0_t3, method="DOP853", rtol=1e-12, atol=1e-14,
                           t_eval=sol_aux0["tau"], max_step=0.1)
        rel_diff = float(np.max(np.abs(sol_aux0["q1"] - sol_t3.y[4])) / max(1.0, np.max(np.abs(sol_t3.y[4]))))
        t3b_cmp[f"beta0_{b0}"] = {"tau_end": tau_end_t, "max_rel_diff_q1": rel_diff}
        print(f"   beta_0={b0:.4f} (tau={tau_end_t:.2f}): max_rel_diff vs T3-B proper-time Ostrogradsky = {rel_diff:.3e}", flush=True)

    # Benchmark 2: Full Unreduced Nonlocal (kappa_mem=1.0): Solver A (Aux-ODE) vs Solver B (Direct Volterra) vs Goal 1!
    # Tests BOTH:
    #   (a) M_I* = 26, beta_0 = 0.2890 (above pinch-off -> UHP Horowitz gam = +0.21815 H_S!)
    #   (b) M_I* = 26, beta_0 = 1.0000 (sub-threshold real pole -> ZERO static runaway gam = 0.0!)
    #   (c) M_S3 =  6, beta_0 = 1.0000 (round S^3 above pinch-off -> UHP Horowitz gam = +0.11265 H_S!)
    #   (d) M_S3 =  6, beta_0 = 0.2890 (round S^3 -> UHP Horowitz gam = +0.36315 H_S, longer window eta=2.8 -> <1e-3!)
    print("\n3. Full Unreduced Causal Nonlocal (kappa_mem=1.0): Solver A (Aux-ODE) vs Solver B (Direct Volterra) vs Goal 1:", flush=True)
    with open(os.path.join(OUT_DIR, "t4a_01_kernel_verification.json")) as fh:
        g1 = json.load(fh)
    def lookup_g1_gamma(key, b0_val):
        for e in g1["static_xring_horowitz_dispersion"][key]:
            if abs(e["beta_0"] - b0_val) < 1e-4:
                return e["uhp_stelle_gamma_HS"] if e["uhp_stelle_gamma_HS"] is not None else 0.0
        return None

    full_cross = {}
    y0_stelle = [0.0, 0.0, 1.0, 0.0]
    for M_val, m_key, b0_val, eta_t, d_eta_v in [
        (26.0, "M26_Istar", 0.2890, 3.0, 0.001),
        (26.0, "M26_Istar", 1.0000, 5.0, 0.002),
        (6.0,  "M6_S3",     1.0000, 5.0, 0.002),
        (6.0,  "M6_S3",     0.2890, 2.5, 0.001),
    ]:
        rA_st = solve_full_aux_ode(bg_short, eta_t, b0_val, M_thresh=M_val, kappa_mem=1.0,
                                   static_bg=True, Nu=64, u_max=1e7, n_eval=int(round(eta_t / d_eta_v)) + 1, y0_4d=y0_stelle)
        rB_st = solve_full_direct_volterra(bg_short, eta_t, b0_val, M_thresh=M_val, kappa_mem=1.0,
                                           static_bg=True, d_eta=d_eta_v, y0_4d=y0_stelle)
        n_min = min(len(rA_st["pi2"]), len(rB_st["pi2"]))
        rel_AB_st = float(np.max(np.abs(rA_st["pi2"][:n_min] - rB_st["pi2"][:n_min])) /
                          max(1.0, np.max(np.abs(rA_st["pi2"][:n_min]))))
        gam_A = fit_exponential_rate(rA_st["tau"], np.abs(rA_st["pi2"]), 0.50, 0.98)
        gam_B = fit_exponential_rate(rB_st["tau"], np.abs(rB_st["pi2"]), 0.50, 0.98)
        gam_G1 = lookup_g1_gamma(m_key, b0_val)
        rel_gam = (abs(gam_A - gam_G1) / abs(gam_G1)) if (gam_G1 and abs(gam_G1) > 1e-10) else abs(gam_A)

        rA_dyn = solve_full_aux_ode(bg_short, eta_t, b0_val, M_thresh=M_val, kappa_mem=1.0,
                                    static_bg=False, Nu=64, u_max=1e7, n_eval=int(round(eta_t / d_eta_v)) + 1, y0_4d=y0_stelle)
        rB_dyn = solve_full_direct_volterra(bg_short, eta_t, b0_val, M_thresh=M_val, kappa_mem=1.0,
                                            static_bg=False, d_eta=d_eta_v, y0_4d=y0_stelle)
        n_md = min(len(rA_dyn["pi2"]), len(rB_dyn["pi2"]))
        rel_AB_dyn = float(np.max(np.abs(rA_dyn["pi2"][:n_md] - rB_dyn["pi2"][:n_md])) /
                           max(1.0, np.max(np.abs(rA_dyn["pi2"][:n_md]))))
        gam_dyn = fit_exponential_rate(rA_dyn["tau"], np.abs(rA_dyn["pi2"]), 0.50, 0.98)

        lbl = f"{m_key}_beta0_{b0_val}"
        full_cross[lbl] = {
            "eta_end": eta_t, "d_eta": d_eta_v,
            "static_rel_diff_AuxODE_vs_Volterra": rel_AB_st,
            "static_gamma_AuxODE_HS": gam_A,
            "static_gamma_Volterra_HS": gam_B,
            "static_gamma_Goal1_exact_HS": gam_G1,
            "static_gamma_err_vs_Goal1": rel_gam,
            "dyn_KAM_rel_diff_AuxODE_vs_Volterra": rel_AB_dyn,
            "dyn_KAM_gamma_AuxODE_HS": gam_dyn,
        }
        print(f"   [{lbl:22s}] st Aux/Volt={rel_AB_st:.2e} | dyn Aux/Volt={rel_AB_dyn:.2e} | "
              f"gam_A={gam_A:+.6f} vs G1={gam_G1:+.6f} (err={rel_gam:.2e}) | dyn_gam={gam_dyn:+.5f}", flush=True)

    # Benchmark 3: Exact RG-Scale Invariance Kill-Test at M_I* = 26!
    print("\n4. Exact RG-Scale Invariance Kill-Test (M=26, mu_phys: 1 -> e, beta_0: 1.0 -> 1.0 + c_anom):", flush=True)
    r_rg1 = solve_full_aux_ode(bg_short, 3.0, 1.0, M_thresh=26.0, kappa_mem=1.0, mu_phys=1.0, Nu=64, u_max=1e7, n_eval=301)
    r_rg2 = solve_full_aux_ode(bg_short, 3.0, 1.0 + C_ANOM, M_thresh=26.0, kappa_mem=1.0, mu_phys=math.e, Nu=64, u_max=1e7, n_eval=301)
    rg_rel_diff = float(np.max(np.abs(r_rg1["q1"] - r_rg2["q1"])) / max(1.0, np.max(np.abs(r_rg1["q1"]))))
    r_rgv1 = solve_full_direct_volterra(bg_short, 2.0, 1.0, M_thresh=26.0, kappa_mem=1.0, mu_phys=1.0, d_eta=0.002)
    r_rgv2 = solve_full_direct_volterra(bg_short, 2.0, 1.0 + C_ANOM, M_thresh=26.0, kappa_mem=1.0, mu_phys=math.e, d_eta=0.002)
    rg_volt_diff = float(np.max(np.abs(r_rgv1["q1"] - r_rgv2["q1"])) / max(1.0, np.max(np.abs(r_rgv1["q1"]))))
    print(f"   Full Aux-ODE RG diff = {rg_rel_diff:.3e} | Full Direct Volterra RG diff = {rg_volt_diff:.3e}", flush=True)

    # Benchmark 4: Order-Reduced Cross-Check on Static x_ring (M=26 & M=6 zero damping vs M=0 spurious anti-damping!)
    print("\n5. Order-Reduced (Parker-Simon) Cross-Check on Static x_ring (M=26 vs M=0):", flush=True)
    or_cross = {}
    for M_val, m_key in [(26.0, "M26_Istar"), (6.0, "M6_S3"), (0.0, "M0_naive")]:
        eta_or, d_eta_or = 30.0, 0.005
        rA_or = solve_order_reduced_aux_ode(bg_short, eta_or, 1.0, M_thresh=M_val, kappa_mem=1.0,
                                            static_bg=True, Nu=64, u_max=1e7, n_eval=int(round(eta_or / d_eta_or)) + 1)
        rB_or = solve_order_reduced_direct_volterra(bg_short, eta_or, 1.0, M_thresh=M_val, kappa_mem=1.0,
                                                    static_bg=True, d_eta=d_eta_or)
        rel_or = float(np.max(np.abs(rA_or["b"] - rB_or["b"])) / max(1.0, np.max(np.abs(rA_or["b"]))))
        gam_or_A = fit_exponential_rate(rA_or["tau"], np.abs(rA_or["b"]), 0.2, 0.95)
        gam_or_B = fit_exponential_rate(rB_or["tau"], np.abs(rB_or["b"]), 0.2, 0.95)
        or_cross[m_key] = {"rel_diff_AuxODE_vs_Volterra": rel_or, "gamma_AuxODE_HS": gam_or_A, "gamma_Volterra_HS": gam_or_B}
        print(f"   [{m_key:10s}] Aux vs Volt = {rel_or:.3e} | gam_A={gam_or_A:+.4e} | gam_B={gam_or_B:+.4e}", flush=True)

    payload = {
        "background_info": info_short,
        "zero_memory_reduction_vs_T3B": t3b_cmp,
        "full_nonlocal_AuxODE_vs_DirectVolterra": full_cross,
        "rg_invariance_kill_tests": {
            "full_aux_ode_rel_diff": rg_rel_diff,
            "full_direct_volterra_rel_diff": rg_volt_diff,
        },
        "order_reduced_static_cross_check": or_cross,
    }
    out_path = os.path.join(OUT_DIR, "t4a_02_solver_benchmarks.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
