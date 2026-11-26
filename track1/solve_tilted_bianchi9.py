"""Goal 2 & 3 Solver: Exact Tilted/Rotating Triaxial Bianchi IX Einstein-Maxwell-Lambda on S^3/I*.

Integrates BOTH exact self-consistent systems WITHOUT clamping or projecting constraints:
  - Model II (Pure Null-Dust Congruence, 7 ODEs):
      T_{hat mu hat nu} = u k_{hat mu} k_{hat nu}, u = u_0 / (a_1 a_2 a_3^2), T_{12} = 0.
      Tracks exact Hamiltonian residual H and Momentum residual M_3.
  - Model I (Exact Source-Free Page 1 Beltrami-Maxwell Wave, 11 ODEs):
      Full (E_1, E_2, B_1, B_2) Maxwell equations with gravitational birefringence & torque
      d/dt(a_1 a_2 a_3^2 S_3) = 2 a_3 (a_1^2 - a_2^2) T_{12}.
      Tracks exact Hamiltonian residual H, Momentum residual M_3, Poynting reversal S_3/u,
      Lorentz invariants (E^2-B^2)/(2u) and (E.B)/u, and causality metrics g_33, g_00.

Also computes the exact linear stability spectrum of the stationary rotating triaxial
Einstein-static fixed point (a_1*=a_3*=sqrt(2/3), a_2*=sqrt(2)/3, beta*=sqrt(6)/4, u_0*=8*sqrt(3)/27).
"""

import json
import math
import os
import numpy as np
from scipy.integrate import solve_ivp

LAM = 3.0
U0_ALG_CEILING = 8.0 * math.sqrt(3.0) / 27.0  # 0.5132002392796674


def ricci3_triaxial(a1: float, a2: float, a3: float):
    """Exact 3-Ricci eigenvalues on unit-S^3 normalized triaxial Bianchi IX."""
    a1s, a2s, a3s = a1 * a1, a2 * a2, a3 * a3
    denom = a1s * a2s * a3s
    r1 = 2.0 * (a1s * a1s - (a2s - a3s) ** 2) / denom
    r2 = 2.0 * (a2s * a2s - (a3s - a1s) ** 2) / denom
    r3 = 2.0 * (a3s * a3s - (a1s - a2s) ** 2) / denom
    return r1, r2, r3, r1 + r2 + r3


def compute_fixed_point_stability():
    """Compute exact Jacobian eigenvalues at the rotating Einstein-static fixed point in Model II."""
    a1s = math.sqrt(2.0 / 3.0)
    a2s = math.sqrt(2.0) / 3.0
    a3s = math.sqrt(2.0 / 3.0)
    u0s = U0_ALG_CEILING
    sig12s = 1.0 / math.sqrt(2.0)

    # Full 7-var state: y = [a1, a2, a3, H1, H2, H3, sig12]
    def rhs7(y):
        a1, a2, a3, H1, H2, H3, sig12 = y
        r1, r2, r3, _ = ricci3_triaxial(a1, a2, a3)
        Th = H1 + H2 + H3
        diff = a1 * a1 - a2 * a2
        Om12 = sig12 * (a1 * a1 + a2 * a2) / diff
        u = u0s / (a1 * a2 * a3 * a3)
        return np.array([
            H1 * a1,
            H2 * a2,
            H3 * a3,
            LAM + 2.0 * Om12 * sig12 - Th * H1 - r1,
            LAM - 2.0 * Om12 * sig12 - Th * H2 - r2,
            LAM + u - Th * H3 - r3,
            -Th * sig12 + Om12 * (H2 - H1),
        ], dtype=np.float64)

    y_star = np.array([a1s, a2s, a3s, 0.0, 0.0, 0.0, sig12s], dtype=np.float64)
    f0 = rhs7(y_star)
    # Central finite-difference Jacobian
    eps = 1e-7
    J = np.zeros((7, 7), dtype=np.float64)
    for j in range(7):
        dy = np.zeros(7); dy[j] = eps
        J[:, j] = (rhs7(y_star + dy) - rhs7(y_star - dy)) / (2.0 * eps)

    # Constraint gradients at y_star:
    # H(y) = H1*H2 + H2*H3 + H3*H1 - sig12^2 + 0.5*R3 - (u + LAM)
    # M3(y) = 2*sig12*(a1^2 - a2^2)/(a1*a2*a3) - u
    def constraints(y):
        a1, a2, a3, H1, H2, H3, sig12 = y
        _, _, _, R3 = ricci3_triaxial(a1, a2, a3)
        u = u0s / (a1 * a2 * a3 * a3)
        H_c = H1 * H2 + H2 * H3 + H3 * H1 - sig12 * sig12 + 0.5 * R3 - (u + LAM)
        M_c = 2.0 * sig12 * (a1 * a1 - a2 * a2) / (a1 * a2 * a3) - u
        return np.array([H_c, M_c], dtype=np.float64)

    C_grad = np.zeros((2, 7), dtype=np.float64)
    for j in range(7):
        dy = np.zeros(7); dy[j] = eps
        C_grad[:, j] = (constraints(y_star + dy) - constraints(y_star - dy)) / (2.0 * eps)

    eigs_full = np.linalg.eigvals(J)
    # Project J onto the 5D tangent space of M3 = 0 and H = 0?
    # Note: at the critical point, grad(H) = 0 because y_star is a critical point of the effective potential V_eff!
    # Check rank of C_grad!
    svals = np.linalg.svd(C_grad, compute_uv=False)

    # Also compute the exact 3x3 Hessian of V_eff(a1, a2, a3) = u + sig12(a)^2 + LAM - 0.5*R3 in (ln a1, ln a2, ln a3)!
    # Because H_1 H_2 + H_2 H_3 + H_3 H_1 = V_eff(a_1, a_2, a_3) where V_eff = 0 and grad V_eff = 0 at y_star!
    def V_eff_log(q):
        a1, a2, a3 = math.exp(q[0]), math.exp(q[1]), math.exp(q[2])
        _, _, _, R3 = ricci3_triaxial(a1, a2, a3)
        u = u0s / (a1 * a2 * a3 * a3)
        sig = u0s / (2.0 * a3 * (a1 * a1 - a2 * a2))
        return u + sig * sig + LAM - 0.5 * R3

    q_star = np.array([math.log(a1s), math.log(a2s), math.log(a3s)], dtype=np.float64)
    Hess = np.zeros((3, 3), dtype=np.float64)
    h = 1e-5
    for i in range(3):
        for j in range(3):
            ei = np.zeros(3); ei[i] = h
            ej = np.zeros(3); ej[j] = h
            Hess[i, j] = (
                V_eff_log(q_star + ei + ej)
                - V_eff_log(q_star + ei - ej)
                - V_eff_log(q_star - ei + ej)
                + V_eff_log(q_star - ei - ej)
            ) / (4.0 * h * h)

    # Notice kinetic form K = H1 H2 + H2 H3 + H3 H1 = (1/2) H^T G_kin H with G_kin = [[0,1,1],[1,0,1],[1,1,0]]!
    # Check: G_kin^{-1} = (1/2) [[-1,1,1],[1,-1,1],[1,1,-1]] (signature (+,-,-)!).
    # Wait: check exact linearization of the reduced 6D system (eliminating sig12 via M3=0):
    def rhs6_reduced(z):
        a1, a2, a3, H1, H2, H3 = z
        r1, r2, r3, _ = ricci3_triaxial(a1, a2, a3)
        Th = H1 + H2 + H3
        diff = a1 * a1 - a2 * a2
        sig12 = u0s / (2.0 * a3 * diff)
        Om12 = sig12 * (a1 * a1 + a2 * a2) / diff
        u = u0s / (a1 * a2 * a3 * a3)
        return np.array([
            H1 * a1,
            H2 * a2,
            H3 * a3,
            LAM + 2.0 * Om12 * sig12 - Th * H1 - r1,
            LAM - 2.0 * Om12 * sig12 - Th * H2 - r2,
            LAM + u - Th * H3 - r3,
        ], dtype=np.float64)

    z_star = np.array([a1s, a2s, a3s, 0.0, 0.0, 0.0], dtype=np.float64)
    J6 = np.zeros((6, 6), dtype=np.float64)
    for j in range(6):
        dz = np.zeros(6); dz[j] = eps
        J6[:, j] = (rhs6_reduced(z_star + dz) - rhs6_reduced(z_star - dz)) / (2.0 * eps)
    eigs6 = np.linalg.eigvals(J6)
    eigs6_sorted = sorted(eigs6, key=lambda c: (-c.real, -abs(c.imag)))

    return {
        "rhs7_norm_at_star": float(np.linalg.norm(f0)),
        "constraint_grad_svals": [float(s) for s in svals],
        "V_eff_hessian_eigs": [float(x) for x in np.linalg.eigvalsh(Hess)],
        "reduced_6D_eigenvalues": [{"re": float(c.real), "im": float(c.imag)} for c in eigs6_sorted],
        "full_7D_eigenvalues": [{"re": float(c.real), "im": float(c.imag)} for c in sorted(eigs_full, key=lambda c: (-c.real, -abs(c.imag)))],
    }


def integrate_model_II_nulldust(
    a_mean: float,
    r0: float,
    s0: float,
    u0: float,
    shear_12: float = 0.0,
    shear_3: float = 0.0,
    t_max: float = 12.0,
    store_trace: bool = False,
):
    """Integrate Model II (7-variable Pure Null-Dust Tilted Bianchi IX) without constraint clamping.

    Initial geometry:
      a1/a2 = r0 (> 1), a3 / sqrt(a1*a2) = s0, (a1*a2*a3)^(1/3) = a_mean.
      => a_perp := sqrt(a1*a2) = a_mean / s0^(1/3), a3 = s0 * a_perp,
         a1 = a_perp * sqrt(r0), a2 = a_perp / sqrt(r0).
    Initial momentum constraint solved exactly:
      sig12_0 = u0 / (2 * a3 * (a1^2 - a2^2)).
    Initial Hubble rates parameterized by:
      H1 = H_base * (1 + shear_12), H2 = H_base * (1 - shear_12), H3 = H_base * (1 + shear_3),
      solving exact quadratic Hamiltonian constraint for H_base < 0 (contracting).
    """
    a_perp = a_mean / (s0 ** (1.0 / 3.0))
    a1_0 = a_perp * math.sqrt(r0)
    a2_0 = a_perp / math.sqrt(r0)
    a3_0 = s0 * a_perp
    diff0 = a1_0 * a1_0 - a2_0 * a2_0
    if abs(diff0) < 1e-12:
        return {"outcome": "invalid_axisymmetric_init"}

    sig12_0 = u0 / (2.0 * a3_0 * diff0)
    u_init = u0 / (a1_0 * a2_0 * a3_0 * a3_0)
    _, _, _, R3_0 = ricci3_triaxial(a1_0, a2_0, a3_0)
    rhs = u_init + LAM + sig12_0 * sig12_0 - 0.5 * R3_0
    c1, c2, c3 = 1.0 + shear_12, 1.0 - shear_12, 1.0 + shear_3
    coeff = c1 * c2 + c2 * c3 + c3 * c1
    if rhs <= 0.0 or coeff <= 0.0:
        return {"outcome": "invalid_initial_constraint", "rhs": rhs, "coeff": coeff}

    Hb0 = -math.sqrt(rhs / coeff)
    H1_0, H2_0, H3_0 = Hb0 * c1, Hb0 * c2, Hb0 * c3
    y0 = np.array([a1_0, a2_0, a3_0, H1_0, H2_0, H3_0, sig12_0], dtype=np.float64)

    def ode(_t, y):
        a1, a2, a3, H1, H2, H3, sig12 = y
        r1, r2, r3, _ = ricci3_triaxial(a1, a2, a3)
        Th = H1 + H2 + H3
        diff = a1 * a1 - a2 * a2
        Om12 = sig12 * (a1 * a1 + a2 * a2) / diff
        u = u0 / (a1 * a2 * a3 * a3)
        return np.array([
            H1 * a1,
            H2 * a2,
            H3 * a3,
            LAM + 2.0 * Om12 * sig12 - Th * H1 - r1,
            LAM - 2.0 * Om12 * sig12 - Th * H2 - r2,
            LAM + u - Th * H3 - r3,
            -Th * sig12 + Om12 * (H2 - H1),
        ], dtype=np.float64)

    def ev_collapse(_t, y):
        return min(y[0], y[1], y[2]) - 0.02
    ev_collapse.terminal = True; ev_collapse.direction = -1

    def ev_axisym(_t, y):
        return abs(y[0] - y[1]) - 1e-4
    ev_axisym.terminal = True; ev_axisym.direction = -1

    def ev_escape(_t, y):
        return min(y[0], y[1], y[2]) - 8.0
    ev_escape.terminal = True; ev_escape.direction = 1

    sol = solve_ivp(
        ode, (0.0, t_max), y0, method="DOP853",
        rtol=1e-12, atol=1e-14, max_step=0.05,
        events=[ev_collapse, ev_axisym, ev_escape],
    )

    a1, a2, a3, H1, H2, H3, sig12 = sol.y
    npts = sol.y.shape[1]
    H_res = np.empty(npts)
    M_res = np.empty(npts)
    g00_arr = np.empty(npts)
    beta_arr = np.empty(npts)
    for i in range(npts):
        _, _, _, R3 = ricci3_triaxial(a1[i], a2[i], a3[i])
        u = u0 / (a1[i] * a2[i] * a3[i] ** 2)
        H_res[i] = abs(H1[i] * H2[i] + H2[i] * H3[i] + H3[i] * H1[i] - sig12[i] ** 2 + 0.5 * R3 - (u + LAM))
        M_res[i] = abs(2.0 * sig12[i] * (a1[i] ** 2 - a2[i] ** 2) / (a1[i] * a2[i] * a3[i]) - u)
        be = sig12[i] * a1[i] * a2[i] / (a1[i] ** 2 - a2[i] ** 2)
        beta_arr[i] = be
        g00_arr[i] = -1.0 + (a3[i] * be) ** 2

    if len(sol.t_events[0]) > 0:
        # Classify which axis collapsed
        end_axes = (float(a1[-1]), float(a2[-1]), float(a3[-1]))
        idx_min = int(np.argmin(end_axes))
        outcome = f"collapse_a{idx_min + 1}"
    elif len(sol.t_events[1]) > 0:
        outcome = "axisymmetry_hit"
    elif H1[-1] > 0.0 and H2[-1] > 0.0 and H3[-1] > 0.0 and min(np.min(a1), np.min(a2), np.min(a3)) > 0.05:
        outcome = "nonsingular_bounce"
    else:
        outcome = "unresolved"

    # Compute total synchronous ellipsoid rotation angle Delta_theta = -2 int beta dt
    delta_theta = float(-2.0 * np.trapezoid(beta_arr, sol.t)) if hasattr(np, "trapezoid") else float(-2.0 * np.trapz(beta_arr, sol.t))

    out = {
        "model": "Model_II_NullDust",
        "outcome": outcome,
        "a_mean": a_mean,
        "r0": r0,
        "s0": s0,
        "u0": u0,
        "max_H_res": float(np.max(H_res)),
        "max_M_res": float(np.max(M_res)),
        "min_a1": float(np.min(a1)),
        "min_a2": float(np.min(a2)),
        "min_a3": float(np.min(a3)),
        "min_a1_minus_a2": float(np.min(np.abs(a1 - a2))),
        "min_g33": float(np.min(a3 ** 2)),
        "max_g00_corot": float(np.max(g00_arr)),
        "max_a3_abs_beta": float(np.max(np.abs(a3 * beta_arr))),
        "delta_theta_rad": delta_theta,
        "t_end": float(sol.t[-1]),
        "n_steps": int(npts),
    }
    if store_trace:
        out["trace"] = {
            "t": sol.t.tolist(),
            "a1": a1.tolist(),
            "a2": a2.tolist(),
            "a3": a3.tolist(),
            "sig12": sig12.tolist(),
            "beta": beta_arr.tolist(),
            "g00_corot": g00_arr.tolist(),
            "H_res": H_res.tolist(),
            "M_res": M_res.tolist(),
        }
    return out


def integrate_model_I_maxwell(
    a_mean: float,
    r0: float,
    s0: float,
    u0: float,
    phi0: float = 0.0,
    shear_12: float = 0.0,
    shear_3: float = 0.0,
    t_max: float = 12.0,
    store_trace: bool = False,
):
    """Integrate Model I (11-variable Exact Source-Free Page 1 Beltrami-Maxwell Wave + Tilted Bianchi IX).

    Initializes Page 1's exact circularly polarized null Beltrami wave at t = 0:
      E1 = sqrt(u_i) cos(phi0), E2 = sqrt(u_i) sin(phi0),
      B1 = sqrt(u_i) sin(phi0), B2 = -sqrt(u_i) cos(phi0),
    where u_i = u0 / (a1_0 * a2_0 * a3_0^2), giving S_3(0) = -u_i (T_03 = +u_i),
    E^2 - B^2 = 0, E.B = 0, T_12 = 0 identically at t = 0!
    """
    a_perp = a_mean / (s0 ** (1.0 / 3.0))
    a1_0 = a_perp * math.sqrt(r0)
    a2_0 = a_perp / math.sqrt(r0)
    a3_0 = s0 * a_perp
    diff0 = a1_0 * a1_0 - a2_0 * a2_0
    if abs(diff0) < 1e-12:
        return {"outcome": "invalid_axisymmetric_init"}

    u_init = u0 / (a1_0 * a2_0 * a3_0 * a3_0)
    amp = math.sqrt(max(u_init, 0.0))
    E1_0 = amp * math.cos(phi0)
    E2_0 = amp * math.sin(phi0)
    B1_0 = amp * math.sin(phi0)
    B2_0 = -amp * math.cos(phi0)

    cE1_0 = a2_0 * a3_0 * E1_0
    cE2_0 = a3_0 * a1_0 * E2_0
    cB1_0 = a2_0 * a3_0 * B1_0
    cB2_0 = a3_0 * a1_0 * B2_0

    S3_0 = E1_0 * B2_0 - E2_0 * B1_0  # = -u_init
    # Momentum constraint: 2 * sig12 * (a1^2 - a2^2)/(a1*a2*a3) + S3 = 0
    sig12_0 = -S3_0 * a1_0 * a2_0 * a3_0 / (2.0 * diff0)

    _, _, _, R3_0 = ricci3_triaxial(a1_0, a2_0, a3_0)
    rhs = u_init + LAM + sig12_0 * sig12_0 - 0.5 * R3_0
    c1, c2, c3 = 1.0 + shear_12, 1.0 - shear_12, 1.0 + shear_3
    coeff = c1 * c2 + c2 * c3 + c3 * c1
    if rhs <= 0.0 or coeff <= 0.0:
        return {"outcome": "invalid_initial_constraint", "rhs": rhs, "coeff": coeff}

    Hb0 = -math.sqrt(rhs / coeff)
    H1_0, H2_0, H3_0 = Hb0 * c1, Hb0 * c2, Hb0 * c3
    y0 = np.array([a1_0, a2_0, a3_0, H1_0, H2_0, H3_0, sig12_0, cE1_0, cE2_0, cB1_0, cB2_0], dtype=np.float64)

    def ode(_t, y):
        a1, a2, a3, H1, H2, H3, sig12, cE1, cE2, cB1, cB2 = y
        r1, r2, r3, _ = ricci3_triaxial(a1, a2, a3)
        Th = H1 + H2 + H3
        diff = a1 * a1 - a2 * a2
        beta = sig12 * a1 * a2 / diff
        Om12 = sig12 * (a1 * a1 + a2 * a2) / diff

        E1 = cE1 / (a2 * a3)
        E2 = cE2 / (a3 * a1)
        B1 = cB1 / (a2 * a3)
        B2 = cB2 / (a3 * a1)

        u = 0.5 * (E1 * E1 + E2 * E2 + B1 * B1 + B2 * B2)
        p1 = u - (E1 * E1 + B1 * B1)
        p2 = u - (E2 * E2 + B2 * B2)
        p3 = u
        T12 = -(E1 * E2 + B1 * B2)

        w1 = 2.0 * a1 / (a2 * a3)
        w2 = 2.0 * a2 / (a1 * a3)

        return np.array([
            H1 * a1,
            H2 * a2,
            H3 * a3,
            LAM + p1 + 2.0 * Om12 * sig12 - Th * H1 - r1,
            LAM + p2 - 2.0 * Om12 * sig12 - Th * H2 - r2,
            LAM + p3 - Th * H3 - r3,
            -Th * sig12 + Om12 * (H2 - H1) - T12,
            -w1 * cB1 - 2.0 * beta * cE2,
            -w2 * cB2 + 2.0 * beta * cE1,
            +w1 * cE1 - 2.0 * beta * cB2,
            +w2 * cE2 + 2.0 * beta * cB1,
        ], dtype=np.float64)

    def ev_collapse(_t, y):
        return min(y[0], y[1], y[2]) - 0.02
    ev_collapse.terminal = True; ev_collapse.direction = -1

    def ev_axisym(_t, y):
        return abs(y[0] - y[1]) - 1e-4
    ev_axisym.terminal = True; ev_axisym.direction = -1

    def ev_escape(_t, y):
        return min(y[0], y[1], y[2]) - 8.0
    ev_escape.terminal = True; ev_escape.direction = 1

    sol = solve_ivp(
        ode, (0.0, t_max), y0, method="DOP853",
        rtol=1e-12, atol=1e-14, max_step=0.04,
        events=[ev_collapse, ev_axisym, ev_escape],
    )

    a1, a2, a3, H1, H2, H3, sig12, cE1, cE2, cB1, cB2 = sol.y
    npts = sol.y.shape[1]
    H_res = np.empty(npts)
    M_res = np.empty(npts)
    g00_arr = np.empty(npts)
    beta_arr = np.empty(npts)
    S3_norm = np.empty(npts)
    I1_norm = np.empty(npts)
    I2_norm = np.empty(npts)
    u_arr = np.empty(npts)
    J_em = np.empty(npts)

    for i in range(npts):
        _, _, _, R3 = ricci3_triaxial(a1[i], a2[i], a3[i])
        E1 = cE1[i] / (a2[i] * a3[i])
        E2 = cE2[i] / (a3[i] * a1[i])
        B1 = cB1[i] / (a2[i] * a3[i])
        B2 = cB2[i] / (a3[i] * a1[i])
        u = 0.5 * (E1 * E1 + E2 * E2 + B1 * B1 + B2 * B2)
        S3 = E1 * B2 - E2 * B1
        u_arr[i] = u
        J_em[i] = -a1[i] * a2[i] * a3[i] * a3[i] * S3
        H_res[i] = abs(H1[i] * H2[i] + H2[i] * H3[i] + H3[i] * H1[i] - sig12[i] ** 2 + 0.5 * R3 - (u + LAM))
        M_res[i] = abs(2.0 * sig12[i] * (a1[i] ** 2 - a2[i] ** 2) / (a1[i] * a2[i] * a3[i]) + S3)
        be = sig12[i] * a1[i] * a2[i] / (a1[i] ** 2 - a2[i] ** 2)
        beta_arr[i] = be
        g00_arr[i] = -1.0 + (a3[i] * be) ** 2
        if u > 1e-30:
            S3_norm[i] = -S3 / u
            I1_norm[i] = 0.5 * ((E1 * E1 + E2 * E2) - (B1 * B1 + B2 * B2)) / u
            I2_norm[i] = (E1 * B1 + E2 * B2) / u
        else:
            S3_norm[i] = 0.0
            I1_norm[i] = 0.0
            I2_norm[i] = 0.0

    if len(sol.t_events[0]) > 0:
        end_axes = (float(a1[-1]), float(a2[-1]), float(a3[-1]))
        idx_min = int(np.argmin(end_axes))
        outcome = f"collapse_a{idx_min + 1}"
    elif len(sol.t_events[1]) > 0:
        outcome = "axisymmetry_hit"
    elif H1[-1] > 0.0 and H2[-1] > 0.0 and H3[-1] > 0.0 and min(np.min(a1), np.min(a2), np.min(a3)) > 0.05:
        outcome = "nonsingular_bounce"
    else:
        outcome = "unresolved"

    # Count Poynting sign flips (birefringent reversals)
    poynting_flips = int(np.sum(np.diff(np.sign(S3_norm)) != 0)) if u0 > 0 else 0
    delta_theta = float(-2.0 * np.trapezoid(beta_arr, sol.t)) if hasattr(np, "trapezoid") else float(-2.0 * np.trapz(beta_arr, sol.t))

    out = {
        "model": "Model_I_ExactMaxwell",
        "outcome": outcome,
        "a_mean": a_mean,
        "r0": r0,
        "s0": s0,
        "u0": u0,
        "phi0": phi0,
        "max_H_res": float(np.max(H_res)),
        "max_M_res": float(np.max(M_res)),
        "min_a1": float(np.min(a1)),
        "min_a2": float(np.min(a2)),
        "min_a3": float(np.min(a3)),
        "min_a1_minus_a2": float(np.min(np.abs(a1 - a2))),
        "min_g33": float(np.min(a3 ** 2)),
        "max_g00_corot": float(np.max(g00_arr)),
        "max_a3_abs_beta": float(np.max(np.abs(a3 * beta_arr))),
        "delta_theta_rad": delta_theta,
        "min_S3_norm": float(np.min(S3_norm)),
        "max_S3_norm": float(np.max(S3_norm)),
        "max_abs_I1_norm": float(np.max(np.abs(I1_norm))),
        "max_abs_I2_norm": float(np.max(np.abs(I2_norm))),
        "poynting_flips": poynting_flips,
        "t_end": float(sol.t[-1]),
        "n_steps": int(npts),
    }
    if store_trace:
        out["trace"] = {
            "t": sol.t.tolist(),
            "a1": a1.tolist(),
            "a2": a2.tolist(),
            "a3": a3.tolist(),
            "sig12": sig12.tolist(),
            "beta": beta_arr.tolist(),
            "g00_corot": g00_arr.tolist(),
            "S3_norm": S3_norm.tolist(),
            "I1_norm": I1_norm.tolist(),
            "I2_norm": I2_norm.tolist(),
            "J_em": J_em.tolist(),
            "H_res": H_res.tolist(),
            "M_res": M_res.tolist(),
        }
    return out


def binary_search_u0_max(integrator_fn, a_mean, r0, s0, u_hi=1.0, iters=24, **kwargs):
    """Binary search for dynamical bounce ceiling u0_max at fixed (a_mean, r0, s0)."""
    # First verify u0=0 bounces (if not, ceiling is 0)
    r_vac = integrator_fn(a_mean, r0, s0, 0.0, **kwargs)
    if r_vac["outcome"] != "nonsingular_bounce":
        return 0.0, r_vac
    lo, hi = 0.0, u_hi
    best_r = r_vac
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        res = integrator_fn(a_mean, r0, s0, mid, **kwargs)
        if res["outcome"] == "nonsingular_bounce":
            lo = mid
            best_r = res
        else:
            hi = mid
    return 0.5 * (lo + hi), best_r


def main():
    print("=" * 72)
    print("P01: TILTED ROTATING TRIAXIAL BIANCHI IX ON S^3/I* — FULL NUMERICAL STUDY")
    print("=" * 72)

    # 1. Fixed point linear stability
    fp_stab = compute_fixed_point_stability()
    print("\n[1] Rotating Triaxial Einstein-Static Fixed Point (u0* = 8*sqrt(3)/27 = %.6f):" % U0_ALG_CEILING)
    print("    ||rhs(y*)|| =", fp_stab["rhs7_norm_at_star"])
    print("    V_eff Hessian eigs (log-axes):", [round(x, 5) for x in fp_stab["V_eff_hessian_eigs"]])
    print("    Reduced 6D eigenvalues:", [(round(e["re"], 5), round(e["im"], 5)) for e in fp_stab["reduced_6D_eigenvalues"]])

    # 2. Model II (Pure Null Dust): Phase sweep over (r0, u0) at a_mean=2.0, s0=1.0
    print("\n[2] Model II (Pure Null Dust) Dynamical Ceilings vs Initial Triaxiality r0 = a1/a2 (a_mean=2.0, s0=1.0):")
    r0_list = [1.05, 1.15, 1.25, 1.40, 1.55, 1.73205, 1.90, 2.10, 2.30]
    m2_ceilings_s1 = []
    for r0 in r0_list:
        u_max, r_best = binary_search_u0_max(integrate_model_II_nulldust, 2.0, r0, 1.0, u_hi=0.60, iters=22)
        rec = {
            "r0": r0, "s0": 1.0, "a_mean": 2.0, "u0_max": u_max,
            "max_H_res": r_best.get("max_H_res"), "max_M_res": r_best.get("max_M_res"),
            "min_a1": r_best.get("min_a1"), "min_a2": r_best.get("min_a2"), "min_a3": r_best.get("min_a3"),
            "max_g00_corot": r_best.get("max_g00_corot"),
            "min_g33": r_best.get("min_g33"),
        }
        m2_ceilings_s1.append(rec)
        print(f"    r0={r0:7.4f} -> u0_max={u_max:.6f} (ratio to alg 0.513200: {u_max/U0_ALG_CEILING:.4f}) | "
              f"max|H|={rec['max_H_res']:.2e}, max|M|={rec['max_M_res']:.2e}, max_g00={rec['max_g00_corot']:.3f}")

    # 2b. Optimize (r0, s0) for Model II at a_mean = 2.0 AND test approaching the fixed-point shape (r0=sqrt(3), s0=3^(1/4)=1.31607)!
    # Notice at the fixed point: a1*=sqrt(2/3), a2*=sqrt(2)/3, a3*=sqrt(2/3)
    # => r* = a1*/a2* = sqrt(3) = 1.73205, s* = a3*/sqrt(a1*a2*) = 3^(1/4) = 1.316074!
    print("\n[2b] Model II 2D Grid over (r0, s0) at a_mean=2.0 (including exact fixed-point shape r*=1.7321, s*=1.3161):")
    s0_grid = [0.8, 1.0, 1.2, 3.0 ** 0.25, 1.5, 1.7]
    r0_grid_2d = [1.2, 1.4, 1.6, math.sqrt(3.0), 1.9, 2.1]
    m2_grid_2d = []
    best_m2_a2 = {"u0_max": 0.0}
    for s0 in s0_grid:
        for r0 in r0_grid_2d:
            u_max, r_best = binary_search_u0_max(integrate_model_II_nulldust, 2.0, r0, s0, u_hi=0.55, iters=20)
            entry = {
                "r0": r0, "s0": s0, "u0_max": u_max,
                "max_H_res": r_best.get("max_H_res"), "max_M_res": r_best.get("max_M_res"),
                "max_g00_corot": r_best.get("max_g00_corot"),
            }
            m2_grid_2d.append(entry)
            if u_max > best_m2_a2["u0_max"]:
                best_m2_a2 = entry
    print(f"    Best Model II (a_mean=2.0): r0={best_m2_a2['r0']:.4f}, s0={best_m2_a2['s0']:.4f} -> u0_max={best_m2_a2['u0_max']:.6f} "
          f"({100*best_m2_a2['u0_max']/U0_ALG_CEILING:.2f}% of exact algebraic ceiling 8*sqrt(3)/27)")

    # Test Model II at fixed-point shape across a_mean = 1.0, 2.0, 5.0, 10.0!
    print("\n[2c] Model II Ceiling vs Initial Scale a_mean at Fixed-Point Shape (r*=sqrt(3), s*=3^(1/4)) and Nearly-Isotropic (r0=1.2, s0=1.0):")
    m2_amean_scan = []
    for a_m in [0.9, 1.2, 2.0, 5.0, 10.0]:
        u_fp, r_fp = binary_search_u0_max(integrate_model_II_nulldust, a_m, math.sqrt(3.0), 3.0 ** 0.25, u_hi=0.55, iters=22, t_max=16.0)
        u_iso, r_iso = binary_search_u0_max(integrate_model_II_nulldust, a_m, 1.2, 1.0, u_hi=0.55, iters=22, t_max=16.0)
        m2_amean_scan.append({
            "a_mean": a_m,
            "u0_max_fp_shape": u_fp, "H_fp": r_fp.get("max_H_res"), "M_fp": r_fp.get("max_M_res"),
            "u0_max_r1p2": u_iso, "H_r1p2": r_iso.get("max_H_res"), "M_r1p2": r_iso.get("max_M_res"),
        })
        print(f"    a_mean={a_m:5.1f}: fp-shape u0_max={u_fp:.6f} (|H|={r_fp.get('max_H_res',0):.1e}, |M|={r_fp.get('max_M_res',0):.1e}) | "
              f"r0=1.2 u0_max={u_iso:.6f}")

    # 3. Model I (Exact Self-Consistent Page 1 Beltrami-Maxwell Wave): Phase sweep & comparison!
    print("\n[3] Model I (Exact Page 1 Beltrami-Maxwell Wave, 11 ODEs) Dynamical Ceilings (a_mean=2.0, s0=1.0, phi0=0):")
    m1_ceilings_s1 = []
    for r0 in r0_list:
        u_max, r_best = binary_search_u0_max(integrate_model_I_maxwell, 2.0, r0, 1.0, u_hi=1.10, iters=22, phi0=0.0)
        rec = {
            "r0": r0, "s0": 1.0, "a_mean": 2.0, "phi0": 0.0, "u0_max": u_max,
            "max_H_res": r_best.get("max_H_res"), "max_M_res": r_best.get("max_M_res"),
            "min_S3_norm": r_best.get("min_S3_norm"), "max_S3_norm": r_best.get("max_S3_norm"),
            "max_abs_I1": r_best.get("max_abs_I1_norm"), "max_abs_I2": r_best.get("max_abs_I2_norm"),
            "poynting_flips": r_best.get("poynting_flips"),
            "max_g00_corot": r_best.get("max_g00_corot"),
            "min_g33": r_best.get("min_g33"),
        }
        m1_ceilings_s1.append(rec)
        print(f"    r0={r0:7.4f} -> u0_max={u_max:.6f} | max|H|={rec['max_H_res']:.2e}, max|M|={rec['max_M_res']:.2e} | "
              f"S3/u in [{rec['min_S3_norm']:+.3f}, {rec['max_S3_norm']:+.3f}] (flips={rec['poynting_flips']}), max_g00={rec['max_g00_corot']:.3f}")

    # 3b. Model I phase sensitivity check (phi0 = 0, pi/8, pi/4, 3pi/8, pi/2) at r0 = 1.4, 1.732
    print("\n[3b] Model I Initial Polarization Phase Sensitivity (a_mean=2.0, s0=1.0):")
    m1_phase_sens = []
    for r0 in [1.25, 1.55, math.sqrt(3.0)]:
        for phi0 in [0.0, math.pi / 8.0, math.pi / 4.0, 3.0 * math.pi / 8.0, math.pi / 2.0]:
            u_max, r_best = binary_search_u0_max(integrate_model_I_maxwell, 2.0, r0, 1.0, u_hi=1.10, iters=20, phi0=phi0)
            m1_phase_sens.append({
                "r0": r0, "phi0_over_pi": phi0 / math.pi, "u0_max": u_max,
                "max_H_res": r_best.get("max_H_res"), "max_M_res": r_best.get("max_M_res"),
                "poynting_flips": r_best.get("poynting_flips"),
            })
        vals = [x["u0_max"] for x in m1_phase_sens if abs(x["r0"] - r0) < 1e-6]
        print(f"    r0={r0:.4f}: u0_max across phi0 in [0, pi/2] = [{min(vals):.5f}, {max(vals):.5f}]")

    # 3c. Model I 2D grid over (r0, s0) at a_mean=2.0, phi0=0 to find global dynamical max
    print("\n[3c] Model I 2D Grid over (r0, s0) at a_mean=2.0, phi0=0:")
    m1_grid_2d = []
    best_m1_a2 = {"u0_max": 0.0}
    for s0 in s0_grid:
        for r0 in r0_grid_2d:
            u_max, r_best = binary_search_u0_max(integrate_model_I_maxwell, 2.0, r0, s0, u_hi=1.10, iters=20, phi0=0.0)
            entry = {
                "r0": r0, "s0": s0, "u0_max": u_max,
                "max_H_res": r_best.get("max_H_res"), "max_M_res": r_best.get("max_M_res"),
                "min_S3_norm": r_best.get("min_S3_norm"), "max_g00_corot": r_best.get("max_g00_corot"),
                "poynting_flips": r_best.get("poynting_flips"),
            }
            m1_grid_2d.append(entry)
            if u_max > best_m1_a2["u0_max"]:
                best_m1_a2 = entry
    print(f"    Best Model I (a_mean=2.0): r0={best_m1_a2['r0']:.4f}, s0={best_m1_a2['s0']:.4f} -> u0_max={best_m1_a2['u0_max']:.6f}")

    # 3d. Model I across a_mean = 2.0, 5.0, 10.0 at r0 = 1.4, 1.732
    print("\n[3d] Model I Ceiling vs Initial Scale a_mean (s0=1.0, phi0=0):")
    m1_amean_scan = []
    for a_m in [2.0, 5.0, 10.0]:
        u_14, r_14 = binary_search_u0_max(integrate_model_I_maxwell, a_m, 1.4, 1.0, u_hi=1.10, iters=20, phi0=0.0, t_max=16.0)
        u_fp, r_fp = binary_search_u0_max(integrate_model_I_maxwell, a_m, math.sqrt(3.0), 3.0 ** 0.25, u_hi=1.10, iters=20, phi0=0.0, t_max=16.0)
        m1_amean_scan.append({
            "a_mean": a_m,
            "u0_max_r1p4": u_14, "H_14": r_14.get("max_H_res"), "M_14": r_14.get("max_M_res"),
            "u0_max_fp": u_fp, "H_fp": r_fp.get("max_H_res"), "M_fp": r_fp.get("max_M_res"),
        })
        print(f"    a_mean={a_m:5.1f}: r0=1.4 u0_max={u_14:.6f} (|H|={r_14.get('max_H_res',0):.1e}, |M|={r_14.get('max_M_res',0):.1e}) | "
              f"fp-shape u0_max={u_fp:.6f}")

    # 4. Detailed representative trajectories for plotting and deep inspection
    traces = {
        "m2_bounce_fp_nearmax": integrate_model_II_nulldust(2.0, math.sqrt(3.0), 3.0 ** 0.25, 0.45, store_trace=True),
        "m2_collapse_above_ceil": integrate_model_II_nulldust(2.0, math.sqrt(3.0), 3.0 ** 0.25, 0.55, store_trace=True),
        "m2_bounce_r1p4_u0p30": integrate_model_II_nulldust(2.0, 1.4, 1.0, 0.30, store_trace=True),
        "m1_bounce_r1p4_u0p30": integrate_model_I_maxwell(2.0, 1.4, 1.0, 0.30, phi0=0.0, store_trace=True),
        "m1_bounce_fp_u0p45": integrate_model_I_maxwell(2.0, math.sqrt(3.0), 3.0 ** 0.25, 0.45, phi0=0.0, store_trace=True),
        "m1_collapse_fp_u0p70": integrate_model_I_maxwell(2.0, math.sqrt(3.0), 3.0 ** 0.25, 0.70, phi0=0.0, store_trace=True),
        "m1_near_axisym_r1p05_u0p05": integrate_model_I_maxwell(2.0, 1.05, 1.0, 0.05, phi0=0.0, store_trace=True),
    }
    print("\n[4] Representative Trajectories Summary:")
    for k, v in traces.items():
        extra = f", S3/u=[{v['min_S3_norm']:+.2f},{v['max_S3_norm']:+.2f}], flips={v['poynting_flips']}" if "min_S3_norm" in v else ""
        print(f"    {k:28s}: {v['outcome']:18s} | |H|={v['max_H_res']:.2e}, |M|={v['max_M_res']:.2e} | "
              f"min_g33={v['min_g33']:.4f}, max_g00={v['max_g00_corot']:+.3f}{extra}")

    # Collect global max constraint residuals on ALL nonsingular bounces
    all_nonsing_H = []
    all_nonsing_M = []
    for coll in [m2_ceilings_s1, m2_grid_2d, m1_ceilings_s1, m1_phase_sens, m1_grid_2d, list(traces.values())]:
        for item in coll:
            if item.get("max_H_res") is not None and item.get("outcome", "nonsingular_bounce") == "nonsingular_bounce":
                all_nonsing_H.append(item["max_H_res"])
                all_nonsing_M.append(item["max_M_res"])

    payload = {
        "constants": {
            "Lambda": LAM,
            "u0_alg_ceiling_nulldust": U0_ALG_CEILING,
            "u0_iso_ceiling": 0.75,
            "u0_standing_berger_ceiling": 0.997,
        },
        "global_max_nonsingular_H_res": max(all_nonsing_H),
        "global_max_nonsingular_M_res": max(all_nonsing_M),
        "fixed_point_stability": fp_stab,
        "model_II_ceilings_s1": m2_ceilings_s1,
        "model_II_grid_2d": m2_grid_2d,
        "model_II_best_a2": best_m2_a2,
        "model_II_amean_scan": m2_amean_scan,
        "model_I_ceilings_s1": m1_ceilings_s1,
        "model_I_phase_sensitivity": m1_phase_sens,
        "model_I_grid_2d": m1_grid_2d,
        "model_I_best_a2": best_m1_a2,
        "model_I_amean_scan": m1_amean_scan,
        "traces": traces,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tilted_bianchi9_results.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nGlobal max nonsingular |H| = {max(all_nonsing_H):.3e}, |M_3| = {max(all_nonsing_M):.3e}")
    print("Saved results ->", out_path)


if __name__ == "__main__":
    main()
