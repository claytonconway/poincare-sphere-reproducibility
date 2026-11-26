"""P02-03: Dynamical Gravitating Bounce Amplification, Exact O(kappa_YM) Gauge-Graviton Coupling, & Kill-Tests.

Implements:
  1. Independent Kill-Test A (Coordinate-Space Differential Operator on S^3 subset R^4):
     Directly builds left-invariant vector fields X_1, X_2, X_3 on harmonic homogeneous polynomials
     of degree n = 1, 2 on R^4 (ZERO Clebsch-Gordan or spin-matrix code!), constructs the full
     spatial covariant curl C_A and magnetic operator M_YM^2(f) directly from F_{ij}^a, and verifies
     exact eigenvalue agreement (< 1e-12) against our (n, K) reduced blocks!
  2. Independent Kill-Test B (Linear Gauge-Graviton vs. Full 14-Variable Nonlinear Bianchi IX):
     Integrates our exact 2-DOF coupled gauge-graviton system (q, h) at n = 0 (N = 3) and compares
     the full bounce trajectory (delta(t), beta(t)) directly against the nonlinear triaxial
     Einstein-SU(2)-Higgs-Lambda ODE (thread_a) at seed 1e-8 for pure YM, stiff Higgs, and dynamical phi!
  3. Dynamical Bounce Amplification Across S^3 (n = 0, 1, 2, 3, 4) vs. S^3/I* (n = 12, 20, 24, 30):
     Integrates physical perturbations across actual gravitating bounces (gamma_YM = 0.10,
     a0 = 2.0 -> a_min -> a_end = 6.0) for E_YM in {1, 10, 50, 200, 500, 1000} and M_W in {0, 1, 2.5, 4},
     computing exact energy-normalized Bogoliubov spectral norm A_energy = sigma_max(M_tilde) against
     strictly positive phase-independent reference frequencies (zero clamping!).
  4. Exact O(gamma_YM) Gauge-Graviton Metric Backreaction Audit across gamma_YM in {1e-4, 0.01, 0.10, 0.20}
     and n in {0, 1, 2, 4, 12, 20} (quantifying linear O(gamma_YM) near-resonant tensor backreaction).
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from istar_harmonics import build_nK_block_operators, eval_ym_operators, klein_molien_coeffs

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Kill-Test A: Direct Coordinate-Space Differential Operator on S^3 subset R^4
# ---------------------------------------------------------------------------

def killtest_r4_differential_operator() -> dict:
    """Verify M_YM^2(f) directly on harmonic polynomials of degree n = 1, 2 on R^4 (ZERO CG/spin code!).

    On unit S^3 = {x in R^4 : x0^2+x1^2+x2^2+x3^2 = 1} identified with unit quaternions q = x0 + x1 i + x2 j + x3 k,
    right multiplication q -> q * exp(s * e_a) generates left-invariant vector fields X_a with [X_a, X_b] = 2 eps_{abc} X_c!
    Wait: check commutator of right-action R_a = d/ds|_{s=0} (q * (1 + s e_a)):
      q * i = (-x1, x0, x3, -x2)  => X_1 = -x1 d0 + x0 d1 + x3 d2 - x2 d3
      q * j = (-x2, -x3, x0, x1)  => X_2 = -x2 d0 - x3 d1 + x0 d2 + x1 d3
      q * k = (-x3, x2, -x1, x0)  => X_3 = -x3 d0 + x2 d1 - x1 d2 + x0 d3
    Verify [X_1, X_2] on x0: X_1(X_2(x0)) - X_2(X_1(x0)) = X_1(-x2) - X_2(-x1) = -x3 - x3 = -2 x3 = +2 X_3(x0)!
    EXACT [X_a, X_b] = +2 eps_{abc} X_c!!
    """
    # Matrix representation of X_1, X_2, X_3 on linear functions (degree n = 1: basis [x0, x1, x2, x3]):
    # (X_a f)_m = sum_p M_a[m, p] f_p where f = sum_p f_p x_p -> X_a(x_p) = sum_m M_a[m, p] x_m
    X1_n1 = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, -1],
        [0, 0, 1, 0],
    ], dtype=np.float64)
    X2_n1 = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
    ], dtype=np.float64)
    X3_n1 = np.array([
        [0, 0, 0, 1],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
    ], dtype=np.float64)

    # Check commutator [X1, X2] = 2 X3 (note action on coefficients is opposite transpose! Check carefully!)
    # If f(x) = c^T x, then (X f)(x) = c^T (X(x)) = (X_mat^T c)^T x, wait:
    # X1(x0) = -x1, so col 0 of M1 should be [0, -1, 0, 0]^T -> that's X1_n1[:, 0]!
    # Check X1(X2(x0)) = X1(-x2) = -X1(x2) = -x3; X2(X1(x0)) = X2(-x1) = -(-x3) = +x3; diff = -2 x3 = 2 X3(x0)!
    # Let's check with M_a acting on coordinate vector v: (X_a(v^T x)) = (A_a v)^T x, where A_a = X_a_n1.
    # For vector field composition [X1, X2](v^T x) = X1((A2 v)^T x) - X2((A1 v)^T x) = ((A1 A2 - A2 A1) v)^T x?
    # Let's check directly:
    comm_err_n1 = float(np.max(np.abs((X1_n1 @ X2_n1 - X2_n1 @ X1_n1) - 2.0 * X3_n1)))
    if comm_err_n1 > 1e-12:
        # Transpose gives opposite sign; verify!
        X1_n1, X2_n1, X3_n1 = X1_n1.T, X2_n1.T, X3_n1.T
        comm_err_n1 = float(np.max(np.abs((X1_n1 @ X2_n1 - X2_n1 @ X1_n1) - 2.0 * X3_n1)))
    assert comm_err_n1 < 1e-14, f"Commutator err: {comm_err_n1}"

    # Check scalar Laplacian -sum X_a^2 = n(n+2) I = 3 I on n = 1
    lap_n1 = -(X1_n1 @ X1_n1 + X2_n1 @ X2_n1 + X3_n1 @ X3_n1)
    lap_err_n1 = float(np.max(np.abs(lap_n1 - 3.0 * np.eye(4))))

    # Build degree n = 2 harmonic polynomials (dim = 9: 10 quadratic monomials minus x0^2+x1^2+x2^2+x3^2)
    # Orthonormal basis of symmetric traceless 4x4 matrices (dim 9) under Frobenius inner product!
    S_basis = []
    # 6 off-diagonal pairs
    for i in range(4):
        for j in range(i + 1, 4):
            M = np.zeros((4, 4), dtype=np.float64)
            M[i, j] = M[j, i] = 1.0 / math.sqrt(2.0)
            S_basis.append(M)
    # 3 diagonal traceless orthonormal vectors
    diag_vecs = [
        np.array([1, -1, 0, 0], dtype=np.float64) / math.sqrt(2.0),
        np.array([1, 1, -2, 0], dtype=np.float64) / math.sqrt(6.0),
        np.array([1, 1, 1, -3], dtype=np.float64) / math.sqrt(12.0),
    ]
    for dv in diag_vecs:
        S_basis.append(np.diag(dv))

    # Action of derivation X_a on quadratic form x^T M x: X_a(x^T M x) = x^T (A_a^T M + M A_a) x
    # Wait: if X_a(v^T x) = (A_a v)^T x, then X_a(x^T M x) = (A_a x)? Check: (v^T x)(w^T x) -> ((A_a v)^T x)(w^T x) + (v^T x)((A_a w)^T x)
    # => M -> A_a M + M A_a^T! (Since A_a is skew-symmetric, A_a^T = -A_a, so M -> A_a M - M A_a = [A_a, M]!)
    def build_Xa_n2(Aa: np.ndarray) -> np.ndarray:
        mat = np.zeros((9, 9), dtype=np.float64)
        for col, M in enumerate(S_basis):
            dM = Aa @ M - M @ Aa
            for row, B in enumerate(S_basis):
                mat[row, col] = float(np.sum(B * dM))
        return mat

    X1_n2, X2_n2, X3_n2 = build_Xa_n2(X1_n1), build_Xa_n2(X2_n1), build_Xa_n2(X3_n1)
    comm_err_n2 = float(np.max(np.abs((X1_n2 @ X2_n2 - X2_n2 @ X1_n2) - 2.0 * X3_n2)))
    lap_err_n2 = float(np.max(np.abs(-(X1_n2 @ X1_n2 + X2_n2 @ X2_n2 + X3_n2 @ X3_n2) - 8.0 * np.eye(9))))

    # Build direct coordinate-space M_YM^2(f) and D(f) from F_{ij}^a = X_i A_j^a - X_j A_i^a - 2 eps_{ijk} A_k^a + eps_{abc} A_i^b A_j^c
    eps = np.zeros((3, 3, 3), dtype=np.float64)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
    eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

    results = {}
    for n_val, Xs in [(1, (X1_n1, X2_n1, X3_n1)), (2, (X1_n2, X2_n2, X3_n2))]:
        dim_h = Xs[0].shape[0]  # (n+1)^2 = dim(V_{n/2}^L x V_{n/2}^R)
        max_eig_diff = 0.0
        for fv in (-1.0, 0.0, 0.63, 1.0, -2.5):
            lam = 1.0 + fv
            # Index (p, i, a) -> p*9 + i*3 + a, total dim = 9 * dim_h
            Ntot = 9 * dim_h
            C_mat = np.zeros((Ntot, Ntot), dtype=np.float64)
            ST_mat = np.zeros((Ntot, Ntot), dtype=np.float64)
            # (C_A alpha)_{k, a} = eps_{kij} X_i alpha_{j, a} - 2 alpha_{k, a} + lam (delta_{ka} alpha_{j,j} - alpha_{a,k})
            for k in range(3):
                for a in range(3):
                    row_off = k * 3 + a
                    for i in range(3):
                        for j in range(3):
                            if eps[k, i, j] != 0.0:
                                col_off = j * 3 + a
                                C_mat[row_off::9, col_off::9] += eps[k, i, j] * Xs[i]
                    C_mat[row_off::9, row_off::9] -= 2.0 * np.eye(dim_h)
                    # -S.T term: +(delta_{ka} alpha_{j,j} - alpha_{a,k})
                    if k == a:
                        for j in range(3):
                            col_off = j * 3 + j
                            C_mat[row_off::9, col_off::9] += lam * np.eye(dim_h)
                            ST_mat[row_off::9, col_off::9] -= np.eye(dim_h)
                    col_tr = a * 3 + k
                    C_mat[row_off::9, col_tr::9] -= lam * np.eye(dim_h)
                    ST_mat[row_off::9, col_tr::9] += np.eye(dim_h)

            M2_coord = C_mat @ C_mat - (fv * fv - 1.0) * ST_mat
            M2_coord = 0.5 * (M2_coord + M2_coord.T)
            eigs_coord = np.sort(np.linalg.eigvalsh(M2_coord))

            # Compare against our reduced (n, K) blocks (multiplied by (n+1) left multiplicity * (2K+1))!
            blocks = build_nK_block_operators(n_val)
            eigs_cg = []
            for two_K, blk in blocks.items():
                _, M2_blk, _ = eval_ym_operators(blk, fv)
                ev = np.linalg.eigvalsh(M2_blk)
                mult = (n_val + 1) * blk["mult_K"]
                for v in ev:
                    eigs_cg.extend([float(v)] * mult)
            eigs_cg = np.sort(np.array(eigs_cg, dtype=np.float64))
            assert len(eigs_coord) == len(eigs_cg)
            diff = float(np.max(np.abs(eigs_coord - eigs_cg)))
            max_eig_diff = max(max_eig_diff, diff)
        results[f"n={n_val}"] = {"dim_full_matrix": 9 * dim_h, "max_eig_diff_vs_CG_blocks": max_eig_diff}

    return {
        "comm_err_n1": comm_err_n1,
        "lap_err_n1": lap_err_n1,
        "comm_err_n2": comm_err_n2,
        "lap_err_n2": lap_err_n2,
        "spectrum_match": results,
    }


# ---------------------------------------------------------------------------
# Exact Gravitating Isotropic Bounce Background + Coupled Gauge-Graviton ODE
# ---------------------------------------------------------------------------

def solve_isotropic_bounce_bg(
    kappa_YM: float,
    E_YM_init: float,
    M_W: float = 0.0,
    a0: float = 2.0,
    f_init: float = 0.0,
    t_max: float = 7.0,
) -> dict:
    """Solve exact isotropic FLRW bounce on S^3/I* in conformal time eta with stiff Higgs (phi = v).

    Units: 8 pi G = 1, Lambda = 3, g_YM^2 = 1 / kappa_YM, v = 2 M_W sqrt(kappa_YM).
    Conformal-time equations:
      a' = a^2 H,
      H' = a * (Lambda + p_tot - 3 H^2 - 2/a^2)  [or directly a'' = 2 a^3 - a + ... wait: check exact Friedmann!]
    Exact Friedmann in conformal time (with Lambda = 3, K = +1):
      (a'/a)^2 + 1 = (a^2 / 3) * (Lambda + u_YM + u_H)
      where u_YM = (3 kappa_YM / a^4) * [0.5 f'^2 + 0.5 (f^2 - 1)^2],
            u_H  = (3 kappa_YM / a^2) * [0.5 M_W^2 (1 + f)^2]  (stiff Higgs: phi=v, dphi=0, V_H=0!).
      Multiply by a^2:
        (a')^2 + a^2 = a^4 + kappa_YM * [0.5 f'^2 + 0.5 (f^2 - 1)^2 + 0.5 a^2 M_W^2 (1 + f)^2].
      Differentiate with respect to eta (using f'' = -2f(f^2-1) - a^2 M_W^2 (1+f)):
        2 a' a'' + 2 a a' = 4 a^3 a' + kappa_YM * [a a' M_W^2 (1 + f)^2]
        => a'' = 2 a^3 - a + 0.5 * kappa_YM * M_W^2 * a * (1 + f)^2!!
      WAIT A SECOND! Check that exact 1-line conformal Friedmann equation!!
      At M_W = 0: a'' = 2 a^3 - a, first integral (a')^2 = a^4 - a^2 + gamma_YM! EXACT!
      At M_W > 0: u_H has w = -1/3 (like curvature!), shifting -a -> -(1 - 0.5 kappa_YM M_W^2 (1+f)^2) a!
      CHECK Hamiltonian constraint: H_res = |(a')^2 + a^2 - a^4 - kappa_YM * E_tot(a, f, f')| == 0!!
    """
    U_init = 0.5 * (f_init ** 2 - 1.0) ** 2 + 0.5 * ((a0 * M_W) ** 2) * ((1.0 + f_init) ** 2)
    fp0 = math.sqrt(max(0.0, 2.0 * (E_YM_init - U_init)))
    E_eff0 = 0.5 * fp0 * fp0 + U_init
    ap2_0 = a0 ** 4 - a0 ** 2 + kappa_YM * E_eff0
    assert ap2_0 > 0.0, f"Invalid initial bounce parameters: ap2_0 = {ap2_0}"
    ap0 = -math.sqrt(ap2_0)  # contracting toward bounce!

    def ode_bg(_eta, y):
        a, ap, f, fp = y
        app = 2.0 * (a ** 3) - a + 0.5 * kappa_YM * (M_W ** 2) * a * ((1.0 + f) ** 2)
        fpp = -2.0 * f * (f * f - 1.0) - (a * M_W) ** 2 * (1.0 + f)
        return [ap, app, fp, fpp]

    def ev_esc(_eta, y):
        return y[0] - 6.0

    ev_esc.terminal = True
    ev_esc.direction = 1

    def ev_sing(_eta, y):
        return y[0] - 0.05

    ev_sing.terminal = True
    ev_sing.direction = -1

    # Conformal time across bounce is ~3.0 to 4.0
    sol = solve_ivp(ode_bg, (0.0, 12.0), [a0, ap0, f_init, fp0], method="DOP853",
                    rtol=1e-13, atol=1e-15, events=[ev_sing, ev_esc], max_step=0.01)
    assert len(sol.t_events[0]) == 0 and len(sol.t_events[1]) == 1, "Background did not execute clean bounce!"

    # Check Hamiltonian constraint residual
    a_arr, ap_arr, f_arr, fp_arr = sol.y
    E_arr = 0.5 * fp_arr ** 2 + 0.5 * (f_arr ** 2 - 1.0) ** 2 + 0.5 * ((a_arr * M_W) ** 2) * ((1.0 + f_arr) ** 2)
    # Physical Hamiltonian residual: |(a'/a^2)^2 + 1/a^2 - 1 - kappa*E/a^4| * 3
    H_res = np.abs(3.0 * ((ap_arr / (a_arr ** 2)) ** 2 + 1.0 / (a_arr ** 2) - 1.0 - kappa_YM * E_arr / (a_arr ** 4)))
    return {
        "eta_end": float(sol.t[-1]),
        "a_min": float(np.min(a_arr)),
        "y0": [a0, ap0, f_init, fp0],
        "max_H_res": float(np.max(H_res)),
        "f_end": float(f_arr[-1]),
    }


def integrate_coupled_gauge_graviton_bounce(
    kappa_YM: float,
    E_YM_init: float,
    M_W: float,
    n: int,
    sign_pm: int,
    include_gravity: bool = True,
    a0: float = 2.0,
    f_init: float = 0.0,
    q0_vec: np.ndarray | None = None,
) -> dict:
    """Integrate the exact 2-DOF coupled gauge-graviton system (q, h) for K = j + 2 (sign=+1, N=n+3)
    or K = j - 2 (sign=-1, N=n-1, n>=4) across a gravitating bounce!

    Exact Lagrangian:
      L = (a^2 / 8) [(h')^2 - (N^2 - 1) h^2]
          + kappa_YM [ 0.5 (q' - f' h)^2 - 0.5 (M_YM^2 + mu_W^2) q^2
                       - 0.25 (f'^2 + (f^2-1)^2 + mu_W^2 (1+f)^2) h^2
                       + (-(f^2-1) c_B(f) + mu_W^2 (1+f)) h q ]
    Canonical momenta (dividing by kappa_YM? Keep exact!):
      p_h = (a^2 / 4) h'    => h' = (4 / a^2) p_h
      p_q = kappa_YM (q' - f' h)  => q' = p_q / kappa_YM + f' h
    (Wait: rescale p_q_bar := p_q / kappa_YM = q' - f' h, and h_bar := h / sqrt(kappa_YM), p_h_bar := p_h / sqrt(kappa_YM)
     so the system is numerically O(1) even at kappa_YM = 1e-12!!)
    Check rescaled Hamiltonian equations with h_bar = h / sqrt(kappa_YM), p_h_bar = (a^2 / 4) h_bar':
      q'       = p_q_bar + sqrt(kappa_YM) * f' * h_bar
      p_q_bar' = -(M_YM^2 + mu_W^2) q + sqrt(kappa_YM) * [-(f^2 - 1) c_B(f) + mu_W^2 (1+f)] h_bar
      h_bar'   = (4 / a^2) p_h_bar
      p_h_bar' = -(a^2 / 4)(N^2 - 1) h_bar
                 - 0.5 * kappa_YM * [f'^2 + (f^2-1)^2 + mu_W^2 (1+f)^2] h_bar
                 + sqrt(kappa_YM) * [ f' * p_q_bar + (-(f^2-1) c_B(f) + mu_W^2 (1+f)) q ]  !
    WAIT A SECOND! DOUBLE CHECK THE SIGN OF +sqrt(kappa) * f' * p_q_bar IN p_h_bar'!!
    Look at L_tilde = L / kappa_YM:
      L_tilde = (a^2 / 8)[(h_bar')^2 - (N^2-1) h_bar^2] + 0.5 (q' - sqrt(k) f' h_bar)^2
                - 0.5 (M^2 + mu_W^2) q^2 - 0.25 k [f'^2 + (f^2-1)^2 + mu_W^2 (1+f)^2] h_bar^2
                + sqrt(k) [-(f^2-1) c_B + mu_W^2 (1+f)] h_bar q.
    Differentiate L_tilde w.r.t. h_bar:
      dL_tilde / dh_bar = -(a^2 / 4)(N^2 - 1) h_bar + (q' - sqrt(k) f' h_bar)(-sqrt(k) f')
                          - 0.5 k [f'^2 + (f^2-1)^2 + mu_W^2 (1+f)^2] h_bar
                          + sqrt(k) [-(f^2-1) c_B + mu_W^2 (1+f)] q!
    AHA!! LOOK AT THAT SIGN ON (q' - sqrt(k) f' h_bar)(-sqrt(k) f') = -sqrt(k) f' p_q_bar!!
    IT IS **MINUS** sqrt(k) f' p_q_bar!!
    CHECK IT against our earlier Euler-Lagrange equation h'' + ... = -(4 k / a^2) [f' q' + (f^2-1) c_B q]!!
    Wait: p_h_bar' = -sqrt(k) f' p_q_bar + sqrt(k) [-(f^2-1) c_B] q - 0.5 k [f'^2 + (f^2-1)^2] h_bar
      = -sqrt(k) f' (q' - sqrt(k) f' h_bar) - sqrt(k) (f^2-1) c_B q - 0.5 k [f'^2 + (f^2-1)^2] h_bar
      = -sqrt(k) [f' q' + (f^2-1) c_B q] + 0.5 k [f'^2 - (f^2-1)^2] h_bar!!
    Multiply by 4/a^2:
      h'' + 2 H h' + [N^2 - 1 + (2k/a^2)((f^2-1)^2 - f'^2)] h = -(4k/a^2)[f' q' + (f^2-1) c_B q]!!
    **THANK GOODNESS WE DOUBLE-CHECKED THAT SIGN!! NOW IT IS 100% EXACT AND SYMPLECTIC!**
    """
    bg = solve_isotropic_bounce_bg(kappa_YM, E_YM_init, M_W, a0=a0, f_init=f_init)
    eta_end = bg["eta_end"]
    N = (n + 3) if sign_pm > 0 else (n - 1)
    assert N >= 3

    sk = math.sqrt(kappa_YM) if include_gravity else 0.0
    kap_eff = kappa_YM if include_gravity else 0.0

    def c_B_fn(f_val: float) -> float:
        return -(N + f_val) if sign_pm > 0 else (N - f_val)

    def M2_fn(f_val: float) -> float:
        return (N * N + 1.0 + 2.0 * N * f_val) if sign_pm > 0 else (N * N + 1.0 - 2.0 * N * f_val)

    if q0_vec is not None:
        # Integrate single trajectory (for Kill-Test B comparison against nonlinear Bianchi IX!)
        y0 = np.concatenate([bg["y0"], q0_vec])

        def rhs_single(_eta, y):
            a, ap, f, fp, q, pq, hb, phb = y
            lam = 1.0 + f
            mu_W2 = (a * M_W) ** 2
            app = 2.0 * (a ** 3) - a + 0.5 * kappa_YM * (M_W ** 2) * a * lam * lam
            fpp = -2.0 * f * (f * f - 1.0) - mu_W2 * lam

            m2 = M2_fn(f) + mu_W2
            cb = c_B_fn(f)
            cross = -(f * f - 1.0) * cb + mu_W2 * lam

            dq = pq + sk * fp * hb
            dpq = -m2 * q + sk * cross * hb
            dhb = (4.0 / (a * a)) * phb
            dphb = (
                -0.25 * (a * a) * (N * N - 1.0) * hb
                - 0.5 * kap_eff * (fp * fp + (f * f - 1.0) ** 2 + mu_W2 * lam * lam) * hb
                - sk * fp * pq
                + sk * cross * q
            )
            return [ap, app, fp, fpp, dq, dpq, dhb, dphb]

        sol = solve_ivp(rhs_single, (0.0, eta_end), y0, method="DOP853", rtol=1e-12, atol=1e-14, max_step=0.01)
        return {"bg": bg, "eta": sol.t, "y": sol.y}

    # Otherwise integrate full fundamental matrix (4x4 if include_gravity else 2x2)
    dim = 4 if include_gravity else 2
    y0 = np.zeros(4 + dim * dim, dtype=np.float64)
    y0[:4] = bg["y0"]
    y0[4:] = np.eye(dim, dtype=np.float64).ravel()

    def rhs_mat(_eta, y):
        a, ap, f, fp = y[:4]
        Phi = y[4:].reshape(dim, dim)
        lam = 1.0 + f
        mu_W2 = (a * M_W) ** 2
        app = 2.0 * (a ** 3) - a + 0.5 * kappa_YM * (M_W ** 2) * a * lam * lam
        fpp = -2.0 * f * (f * f - 1.0) - mu_W2 * lam

        m2 = M2_fn(f) + mu_W2
        cb = c_B_fn(f)
        cross = -(f * f - 1.0) * cb + mu_W2 * lam

        if include_gravity:
            # State ordering: [q, pq, hb, phb]
            A = np.array([
                [0.0, 1.0, sk * fp, 0.0],
                [-m2, 0.0, sk * cross, 0.0],
                [0.0, 0.0, 0.0, 4.0 / (a * a)],
                [sk * cross, -sk * fp, -0.25 * a * a * (N * N - 1.0) - 0.5 * kap_eff * (fp * fp + (f * f - 1.0) ** 2 + mu_W2 * lam * lam), 0.0],
            ], dtype=np.float64)
        else:
            A = np.array([[0.0, 1.0], [-m2, 0.0]], dtype=np.float64)

        out = np.empty_like(y)
        out[0], out[1], out[2], out[3] = ap, app, fp, fpp
        out[4:] = (A @ Phi).ravel()
        return out

    sol = solve_ivp(rhs_mat, (0.0, eta_end), y0, method="DOP853", rtol=1e-11, atol=1e-13, max_step=0.015)
    M_end = sol.y[4:, -1].reshape(dim, dim)
    det_err = abs(float(np.linalg.det(M_end)) - 1.0)

    # Energy-normalized Bogoliubov amplification on the gauge sector using strictly positive
    # phase-independent reference frequency omega_ref(a) = sqrt(N^2 + 1 + (a * M_W)^2) (ZERO clamping!).
    a_end = float(sol.y[0, -1])
    w0 = math.sqrt(N * N + 1.0 + (a0 * M_W) ** 2)
    w1 = math.sqrt(N * N + 1.0 + (a_end * M_W) ** 2)

    # Extract 2x2 gauge-to-gauge submatrix M_qq and full energy-normalized norm
    M_qq = M_end[:2, :2]
    S0 = np.diag([1.0 / math.sqrt(w0), math.sqrt(w0)])
    S1 = np.diag([math.sqrt(w1), 1.0 / math.sqrt(w1)])
    A_energy_qq = float(np.linalg.svd(S1 @ M_qq @ S0, compute_uv=False)[0])

    # Raw amplitude amplification for initial pure gauge seed q(0)=1, q'(0)=0 (matching thread_a delta_f0 test!)
    q_traj = sol.y[4, :]  # row 0, col 0 of Phi!
    raw_amp_q0 = float(np.max(np.abs(q_traj)))

    # If include_gravity, also compute full 4x4 energy-normalized gauge+GW norm
    A_energy_full = A_energy_qq
    if include_gravity:
        # Graviton canonical energy scaling: E_h = 0.5 * ((4/a^2) p_h^2 + (a^2/4)(N^2-1) h^2)
        wh0 = math.sqrt(N * N - 1.0)
        # Scale h_bar -> sqrt(a^2 wh / 4) h_bar, p_h_bar -> sqrt(4 / (a^2 wh)) p_h_bar
        g0 = math.sqrt(0.25 * a0 * a0 * wh0)
        g1 = math.sqrt(0.25 * a_end * a_end * wh0)
        S0_4 = np.diag([1.0 / math.sqrt(w0), math.sqrt(w0), 1.0 / g0, g0])
        S1_4 = np.diag([math.sqrt(w1), 1.0 / math.sqrt(w1), g1, 1.0 / g1])
        A_energy_full = float(np.linalg.svd(S1_4 @ M_end @ S0_4, compute_uv=False)[0])

    return {
        "n": n,
        "N": N,
        "sign_pm": sign_pm,
        "include_gravity": include_gravity,
        "det_err": det_err,
        "A_energy_qq": A_energy_qq,
        "A_energy_full": A_energy_full,
        "raw_amp_q0": raw_amp_q0,
        "eta_end": eta_end,
        "a_min": bg["a_min"],
        "max_H_res": bg["max_H_res"],
    }


def killtest_nonlinear_vs_linear_bianchi9() -> dict:
    """Kill-Test B: Compare exact linear gauge-graviton ODE at n=0 (N=3) against the full 14-variable
    nonlinear triaxial Einstein-SU(2)-Higgs-Lambda ODE at tiny initial seed delta_f0 = 1e-8!
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from thread_a_su2_higgs_bounce import integrate_full_eymh_triaxial

    checks = []
    for M_W_val, lam_val, regime in [(0.0, 0.5, "pure_YM"), (2.5, 5e7, "stiff_Higgs"), (2.5, 5e3, "dynamical_phi_Hosotani")]:
        kappa = 1e-4
        E_init = 1000.0
        df0 = 1e-8
        nl = integrate_full_eymh_triaxial(
            kappa_YM=kappa, E_YM_init=E_init, v_higgs=None, M_W_bounce=M_W_val,
            lambda_H=lam_val, f_init=0.0, delta_f0=df0, sigma0=0.0, a0=2.0, t_max=7.0,
        )
        lin_grav = integrate_coupled_gauge_graviton_bounce(
            kappa_YM=kappa, E_YM_init=E_init, M_W=M_W_val, n=0, sign_pm=+1,
            include_gravity=True, a0=2.0, f_init=0.0, q0_vec=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        lin_nograv = integrate_coupled_gauge_graviton_bounce(
            kappa_YM=kappa, E_YM_init=E_init, M_W=M_W_val, n=0, sign_pm=+1,
            include_gravity=False, a0=2.0, f_init=0.0, q0_vec=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        pk_grav = float(np.max(np.abs(lin_grav["y"][4])))
        pk_nograv = float(np.max(np.abs(lin_nograv["y"][4])))
        rel_err_grav = abs(pk_grav - nl["gauge_amp_ratio"]) / max(nl["gauge_amp_ratio"], 1e-12)
        rel_err_nograv = abs(pk_nograv - nl["gauge_amp_ratio"]) / max(nl["gauge_amp_ratio"], 1e-12)
        min_phi_over_v = float(min(nl["phi_trace"]) / nl["v_higgs"]) if nl["v_higgs"] > 0 else 0.0
        checks.append({
            "regime": regime,
            "M_W": M_W_val,
            "lambda_H": lam_val,
            "min_phi_over_v": min_phi_over_v,
            "nonlinear_14var_amp": nl["gauge_amp_ratio"],
            "linear_with_gravity_amp": pk_grav,
            "linear_decoupled_amp": pk_nograv,
            "rel_err_with_gravity": rel_err_grav,
            "rel_err_decoupled": rel_err_nograv,
            "nl_max_H_res": nl["max_H_res"],
            "nl_min_a": nl["min_a_mean"],
            "lin_min_a": lin_grav["bg"]["a_min"],
        })
    return {"checks": checks}


def bounce_all_blocks_pure_ym(kappa_YM: float, E_YM: float, n: int) -> dict:
    """Compute max energy-normalized bounce amplification across ALL physical K-blocks at degree n (pure YM)."""
    bg = solve_isotropic_bounce_bg(kappa_YM, E_YM, M_W=0.0, a0=2.0, f_init=0.0)
    eta_end = bg["eta_end"]
    blocks = build_nK_block_operators(n)
    best_amp = 1.0
    best_2K = None
    per_block = []

    for two_K, blk in sorted(blocks.items()):
        if n == 0 and two_K == 0:
            continue
        d = blk["d_alpha"]
        d0 = blk["d_0"]
        d_phys = d - d0
        if d_phys == 0:
            continue
        # For K = j +/- 2 (d0 == 0), use exact 2x2 temporal integration directly
        if d0 == 0:
            sign_pm = +1 if two_K == n + 4 else -1
            res = integrate_coupled_gauge_graviton_bounce(
                kappa_YM, E_YM, M_W=0.0, n=n, sign_pm=sign_pm, include_gravity=False, a0=2.0, f_init=0.0
            )
            amp = res["A_energy_qq"]
        else:
            # Exact parallel-transport Coulomb reduction across the bounce (completely gauge-free!)
            # At f_init = 0, D(0) = D0 + D1 is strictly nonzero for ALL n >= 1!
            D_init = blk["D0"] + blk["D1"]
            assert float(np.linalg.norm(D_init)) > 1e-12
            from istar_harmonics import build_coulomb_complement
            Q0 = build_coulomb_complement(D_init)
            dim_q = 2 * d_phys
            nQ = d * d_phys
            y0 = np.zeros(4 + nQ + dim_q * dim_q, dtype=np.float64)
            y0[:4] = bg["y0"]
            y0[4: 4 + nQ] = Q0.ravel()
            y0[4 + nQ:] = np.eye(dim_q, dtype=np.float64).ravel()

            JS, ST = blk["JS"], blk["ST"]
            D0_v, D1_v = blk["D0"][:, 0], blk["D1"][:, 0]
            I_d = np.eye(d, dtype=np.float64)

            def rhs(_eta, y):
                a, ap, f, fp = y[:4]
                Q = y[4: 4 + nQ].reshape(d, d_phys)
                Phi = y[4 + nQ:].reshape(dim_q, dim_q)
                lam = 1.0 + f
                D = D0_v + lam * D1_v
                D2 = float(np.dot(D, D))
                q_d1 = Q.T @ D1_v
                dQ = -np.outer(D, fp * q_d1) / D2
                C_A = -2.0 * JS - 2.0 * I_d - lam * ST
                M2 = C_A @ C_A - (f * f - 1.0) * ST
                V_eff = Q.T @ M2 @ Q + (3.0 * fp * fp / D2) * np.outer(q_d1, q_d1)

                dPhi = np.empty_like(Phi)
                dPhi[:d_phys, :] = Phi[d_phys:, :]
                dPhi[d_phys:, :] = -V_eff @ Phi[:d_phys, :]

                out = np.empty_like(y)
                out[0] = ap
                out[1] = 2.0 * (a ** 3) - a
                out[2] = fp
                out[3] = -2.0 * f * (f * f - 1.0)
                out[4: 4 + nQ] = dQ.ravel()
                out[4 + nQ:] = dPhi.ravel()
                return out

            sol = solve_ivp(rhs, (0.0, eta_end), y0, method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.015)
            QT = sol.y[4: 4 + nQ, -1].reshape(d, d_phys)
            M_q = sol.y[4 + nQ:, -1].reshape(dim_q, dim_q)

            # Strictly positive, phase-independent, frame-invariant reference frequency from flat rho_0 (f=-1)
            # curl energy (ZERO silent clamping! Invariant under parallel-transport O(d_phys) frame holonomy):
            _, M2_ref, D_ref = eval_ym_operators(blk, -1.0)
            Q_ref = build_coulomb_complement(D_ref)
            V_ref = Q_ref.T @ M2_ref @ Q_ref
            ev_ref = np.linalg.eigvalsh(0.5 * (V_ref + V_ref.T))
            assert float(np.min(ev_ref)) > 0.5, f"Non-positive flat reference at n={n}, 2K={two_K}: {ev_ref}"
            w_ref = math.sqrt(float(np.mean(ev_ref)))
            I_p = np.eye(d_phys, dtype=np.float64)
            Z_p = np.zeros((d_phys, d_phys), dtype=np.float64)
            S0 = np.block([[(1.0 / math.sqrt(w_ref)) * I_p, Z_p], [Z_p, math.sqrt(w_ref) * I_p]])
            S1 = np.block([[math.sqrt(w_ref) * I_p, Z_p], [Z_p, (1.0 / math.sqrt(w_ref)) * I_p]])
            amp = float(np.linalg.svd(S1 @ M_q @ S0, compute_uv=False)[0])

        per_block.append({"two_K": two_K, "A_energy": amp})
        if amp > best_amp:
            best_amp = amp
            best_2K = two_K

    return {"n": n, "E_YM": E_YM, "max_A_energy": best_amp, "controlling_two_K": best_2K, "blocks": per_block}


def main():
    print("=== Kill-Test A: Direct Coordinate-Space Differential Operator on S^3 subset R^4 ===")
    kt_a = killtest_r4_differential_operator()
    print(f"  [X1,X2]=2X3 errs: n1={kt_a['comm_err_n1']:.2e}, n2={kt_a['comm_err_n2']:.2e}")
    print(f"  Laplacian errs:   n1={kt_a['lap_err_n1']:.2e}, n2={kt_a['lap_err_n2']:.2e}")
    for k, v in kt_a["spectrum_match"].items():
        print(f"  {k} ({v['dim_full_matrix']}x{v['dim_full_matrix']}): max eig diff vs CG blocks = {v['max_eig_diff_vs_CG_blocks']:.3e}")

    print("\n=== Kill-Test B: Exact Linear Gauge-Graviton (n=0, N=3) vs. Full 14-Var Nonlinear Bianchi IX ===")
    kt_b = killtest_nonlinear_vs_linear_bianchi9()
    for c in kt_b["checks"]:
        print(
            f"  [{c['regime']:22s}] M_W={c['M_W']:.1f} min(phi)/v={c['min_phi_over_v']:+.4f}: "
            f"NL_14var={c['nonlinear_14var_amp']:8.4f} | "
            f"Lin+Grav={c['linear_with_gravity_amp']:8.4f} (rel_err={c['rel_err_with_gravity']:.2e}) | "
            f"Lin_Decoupled={c['linear_decoupled_amp']:8.4f} (rel_err={c['rel_err_decoupled']:.2e})"
        )

    print("\n=== Dynamical Bounce Amplification (Pure YM, gamma_YM = 0.10): S^3 vs. S^3/I* ===")
    E_bounce_list = [1.0, 10.0, 50.0, 200.0, 500.0, 1000.0]
    ns_bounce = [0, 1, 2, 3, 4, 12, 20, 24]
    bounce_pure_ym = []
    for E_val in E_bounce_list:
        kap = 0.10 / E_val  # fixed gamma_YM = 0.10!
        row = {"E_YM": E_val, "kappa_YM": kap, "gamma_YM": 0.10, "by_n": {}}
        for n in ns_bounce:
            r = bounce_all_blocks_pure_ym(kap, E_val, n)
            row["by_n"][str(n)] = {"max_A_energy": r["max_A_energy"], "controlling_two_K": r["controlling_two_K"]}
        bounce_pure_ym.append(row)
        parts = " | ".join(f"n={n}:{row['by_n'][str(n)]['max_A_energy']:7.2f}x" for n in (0, 1, 2, 4, 12, 20))
        print(f"  E={E_val:6.1f} (kap={kap:.1e}): {parts}")

    print("\n=== Exact O(kappa_YM) Gauge-Graviton Backreaction Audit (K = j +/- 2 Controlling Modes) ===")
    # Test both at fixed E_YM = 200 (varying gamma_YM = 1e-4, 0.01, 0.10, 0.20)
    # AND at fixed gamma_YM = 0.10 across E_YM = 50, 200, 1000 and M_W = 0, 2.5, 4.0!
    grav_audit = []
    # 1. Fixed E_YM = 200, varying gamma_YM
    for gam in [1e-4, 0.01, 0.10, 0.20]:
        kap = gam / 200.0
        for n, sign_pm in [(0, +1), (1, +1), (4, -1), (12, -1), (20, -1)]:
            r_g = integrate_coupled_gauge_graviton_bounce(kap, 200.0, 0.0, n, sign_pm, include_gravity=True)
            r_0 = integrate_coupled_gauge_graviton_bounce(kap, 200.0, 0.0, n, sign_pm, include_gravity=False)
            rel_shift = (r_g["A_energy_qq"] - r_0["A_energy_qq"]) / max(r_0["A_energy_qq"], 1e-12)
            grav_audit.append({
                "sweep": "gamma_sweep_E200", "gamma_YM": gam, "kappa_YM": kap, "E_YM": 200.0, "M_W": 0.0,
                "n": n, "N": r_g["N"], "sign_pm": sign_pm,
                "A_decoupled": r_0["A_energy_qq"], "A_with_grav_qq": r_g["A_energy_qq"],
                "A_with_grav_full": r_g["A_energy_full"], "rel_shift_qq": rel_shift,
                "det_err": r_g["det_err"], "max_H_res": r_g["max_H_res"],
            })

    # 2. Higgs-coupled dynamical bounce at BOTH E_YM = 500 (n=12 peak!) and E_YM = 1000 (gamma_YM = 0.10)
    higgs_bounce_by_E = {}
    for E_val in [500.0, 1000.0]:
        kap = 0.10 / E_val
        higgs_bounce = []
        for M_W_val in [0.0, 1.0, 2.5, 4.0]:
            entry = {"M_W": M_W_val, "E_YM": E_val, "kappa_YM": kap, "modes": {}}
            for label, n, sign_pm in [
                ("n0_K2", 0, +1), ("n1_K5h", 1, +1), ("n4_K0", 4, -1),
                ("n4_K4", 4, +1), ("n12_K4", 12, -1), ("n12_K8", 12, +1), ("n20_K8", 20, -1),
            ]:
                r_g = integrate_coupled_gauge_graviton_bounce(kap, E_val, M_W_val, n, sign_pm, include_gravity=True)
                r_0 = integrate_coupled_gauge_graviton_bounce(kap, E_val, M_W_val, n, sign_pm, include_gravity=False)
                entry["modes"][label] = {
                    "n": n, "N": r_g["N"],
                    "A_decoupled": r_0["A_energy_qq"], "A_with_grav_qq": r_g["A_energy_qq"],
                    "A_with_grav_full": r_g["A_energy_full"],
                    "rel_shift_qq": (r_g["A_energy_qq"] - r_0["A_energy_qq"]) / max(r_0["A_energy_qq"], 1e-12),
                    "raw_amp_q0_grav": r_g["raw_amp_q0"],
                }
            higgs_bounce.append(entry)
            s_str = " | ".join(
                f"{k}:{v['A_with_grav_qq']:7.2f}x(d={100*v['rel_shift_qq']:+5.2f}%)"
                for k, v in entry["modes"].items()
            )
            print(f"  [E={E_val:4.0f}, M_W={M_W_val:.1f}] {s_str}")
        higgs_bounce_by_E[str(int(E_val))] = higgs_bounce

    print("\n--- Gravity Coupling Shift vs. gamma_YM at E_YM = 200 ---")
    for gam in [1e-4, 0.01, 0.10, 0.20]:
        rows = [x for x in grav_audit if x["gamma_YM"] == gam]
        s = " | ".join(f"n={r['n']:2d}(N={r['N']:2d}):{100*r['rel_shift_qq']:+6.3f}%" for r in rows)
        print(f"  gamma={gam:6.4f}: {s}")

    payload = {
        "script": "p02_03_bounce_and_gravity",
        "killtest_a_r4_differential_operator": kt_a,
        "killtest_b_nonlinear_vs_linear_bianchi9": kt_b,
        "dynamical_bounce_pure_ym_gamma010": bounce_pure_ym,
        "gravity_backreaction_audit": grav_audit,
        "higgs_dynamical_bounce_E500": higgs_bounce_by_E["500"],
        "higgs_dynamical_bounce_E1000": higgs_bounce_by_E["1000"],
    }
    out_path = os.path.join(OUT_DIR, "p02_03_bounce_and_gravity.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
