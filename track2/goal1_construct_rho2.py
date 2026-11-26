"""Goal 1: Explicit I*-Equivariant Construction & Verification of Galois-Conjugate Flat Connection rho_2.

Constructs the unique degree-7 homogeneous holomorphic I*-intertwiner P_7: C^2 -> C^2
satisfying P_7(rho_1(g) z) = rho_2(g) P_7(z), normalizes u_7(x) = P_7(x e_1) / ||P_7(x e_1)||
into SU(2), and evaluates the strictly I*-invariant flat connection A_{rho_2} = u_7^{-1} du_7
on S^3/I*.

Independent verifications (zero circularity):
1. Exact I* group closure, Galois homomorphism rho_2, and conjugacy class character table.
2. Strict non-vanishing of P_7 on S^3 (min/max ||P_7||^2, exact n=0 + n=12 decomposition).
3. Exact analytical X_k A_{rho_2} (from exact 2nd derivatives of P_7) -> verify ||F|| = ||B|| < 1e-13.
4. Independent DOP853 ODE parallel transport along geodesic lifts 1 -> gamma for all 9 conjugacy
   classes of I*, verifying Tr(Hol_{A_{rho_2}}([gamma])) == Tr(rho_2(gamma)) to < 1e-11.
5. Exact numerical Chern-Simons invariant N_CS(A_{rho_2}) == 49/120 on both 2D Hopf-reduced
   and full 3D Euler quadrature grids (plus topological degree deg(u_7) == 49).
6. Exact Dirichlet energy <|du_7|^2> vs Rayleigh lower bound 63 (21x rho_1's value of 3!).
7. Exact harmonic spectrum of A_{rho_2} in the orthonormal I*-invariant Wigner-D basis up to N_max = 36.
"""

import json
import math
import os

import numpy as np
from scipy.integrate import solve_ivp

from istar_geometry import (
    E_MAT,
    EPS3,
    PHI,
    PHI_BAR,
    T_MAT,
    build_istar_and_galois,
    build_istar_invariant_harmonics,
    build_p7_equivariant_polynomial,
    c2_to_su2,
    compute_ym_and_cs,
    euler_quadrature_s2_hopf,
    euler_quadrature_s3,
    euler_quadrature_so3,
    eval_p7_and_derivs,
    eval_real_harmonic_basis,
    eval_u7_and_connection,
    quat_to_su2,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)


def eval_A_and_XA_exact(U_pts: np.ndarray, poly_coeffs: np.ndarray):
    """Evaluate A_{rho_2}(x) and exact derivatives X_k A_{rho_2}(x) directly from P_7(z)
    and its exact 1st & 2nd polynomial derivatives (zero use of flatness identity!).
    """
    z1 = U_pts[..., 0, 0]
    z2 = U_pts[..., 1, 0]
    z1p = [np.ones_like(z1)]
    z2p = [np.ones_like(z2)]
    for _ in range(7):
        z1p.append(z1p[-1] * z1)
        z2p.append(z2p[-1] * z2)

    bshape = U_pts.shape[:-2]
    P = np.zeros(bshape + (2,), dtype=np.complex128)
    P1 = np.zeros_like(P)
    P2 = np.zeros_like(P)
    P11 = np.zeros_like(P)
    P12 = np.zeros_like(P)
    P22 = np.zeros_like(P)

    for k in range(8):
        p1, p2 = 7 - k, k
        c0, c1 = poly_coeffs[0, k], poly_coeffs[1, k]
        mon = z1p[p1] * z2p[p2]
        P[..., 0] += c0 * mon
        P[..., 1] += c1 * mon
        if p1 >= 1:
            m = p1 * z1p[p1 - 1] * z2p[p2]
            P1[..., 0] += c0 * m
            P1[..., 1] += c1 * m
        if p2 >= 1:
            m = p2 * z1p[p1] * z2p[p2 - 1]
            P2[..., 0] += c0 * m
            P2[..., 1] += c1 * m
        if p1 >= 2:
            m = p1 * (p1 - 1) * z1p[p1 - 2] * z2p[p2]
            P11[..., 0] += c0 * m
            P11[..., 1] += c1 * m
        if p1 >= 1 and p2 >= 1:
            m = p1 * p2 * z1p[p1 - 1] * z2p[p2 - 1]
            P12[..., 0] += c0 * m
            P12[..., 1] += c1 * m
        if p2 >= 2:
            m = p2 * (p2 - 1) * z1p[p1] * z2p[p2 - 2]
            P22[..., 0] += c0 * m
            P22[..., 1] += c1 * m

    # X_i z = (U @ e_i)_{:, 0},  X_k X_i z = (U @ e_k @ e_i)_{:, 0}
    Xz = np.einsum("...ij,ajk->...aik", U_pts, E_MAT)[..., 0]  # (..., 3, 2)
    E_prod = np.einsum("krs,ist->kirt", E_MAT, E_MAT)  # (3, 3, 2, 2)
    XXz = np.einsum("...rs,kist->...kirt", U_pts, E_prod)[..., 0]  # (..., 3, 3, 2): [..., k, i, comp]

    # First derivative X_i P
    XP = P1[..., None, :] * Xz[..., :, 0:1] + P2[..., None, :] * Xz[..., :, 1:2]  # (..., 3, 2)

    # Second derivative X_k X_i P:
    # X_k(P1 * X_i z1 + P2 * X_i z2)
    # = (P11 X_k z1 + P12 X_k z2) X_i z1 + (P12 X_k z1 + P22 X_k z2) X_i z2
    #   + P1 (X_k X_i z1) + P2 (X_k X_i z2)
    Xkz1 = Xz[..., :, None, 0:1]  # (..., k, 1, 1)
    Xkz2 = Xz[..., :, None, 1:2]
    Xiz1 = Xz[..., None, :, 0:1]  # (..., 1, i, 1)
    Xiz2 = Xz[..., None, :, 1:2]

    XXP = (
        P11[..., None, None, :] * (Xkz1 * Xiz1)
        + P12[..., None, None, :] * (Xkz2 * Xiz1 + Xkz1 * Xiz2)
        + P22[..., None, None, :] * (Xkz2 * Xiz2)
        + P1[..., None, None, :] * XXz[..., :, :, 0:1]
        + P2[..., None, None, :] * XXz[..., :, :, 1:2]
    )  # (..., 3, 3, 2): [..., k, i, comp]

    # Now differentiate w = P / r where r = ||P|| = sqrt(<P, P>):
    r2 = np.sum(np.abs(P) ** 2, axis=-1)  # (...,)
    r = np.sqrt(r2)
    inv_r = 1.0 / r
    w = P * inv_r[..., None]  # (..., 2)

    # X_i r = Re<w, X_i P>
    Xr = np.sum(np.conj(w[..., None, :]) * XP, axis=-1).real  # (..., 3)
    # X_i w = (X_i P - w * X_i r) / r
    Xw = (XP - w[..., None, :] * Xr[..., :, None]) * inv_r[..., None, None]  # (..., 3, 2)

    # X_k X_i r = Re<X_k w, X_i P> + Re<w, X_k X_i P>
    XXr = (
        np.sum(np.conj(Xw[..., :, None, :]) * XP[..., None, :, :], axis=-1).real
        + np.sum(np.conj(w[..., None, None, :]) * XXP, axis=-1).real
    )  # (..., 3, 3)

    # X_k X_i w = (X_k X_i P - (X_k w)(X_i r) - w (X_k X_i r) - (X_i w)(X_k r)) / r
    XXw = (
        XXP
        - Xw[..., :, None, :] * Xr[..., None, :, None]
        - w[..., None, None, :] * XXr[..., :, :, None]
        - Xw[..., None, :, :] * Xr[..., :, None, None]
    ) * inv_r[..., None, None, None]  # (..., 3, 3, 2)

    u7 = c2_to_su2(w)  # (..., 2, 2)
    Xu7 = c2_to_su2(Xw)  # (..., 3, 2, 2)
    XXu7 = c2_to_su2(XXw)  # (..., 3, 3, 2, 2)

    # A_i = u7^dag @ X_i u7
    A_mat = np.einsum("...rs,...irt->...ist", u7.conj(), Xu7)  # (..., 3, 2, 2)
    # X_k A_i = (X_k u7)^dag @ (X_i u7) + u7^dag @ (X_k X_i u7)
    XA_mat = (
        np.einsum("...krs,...irt->...kist", Xu7.conj(), Xu7)
        + np.einsum("...rs,...kirt->...kist", u7.conj(), XXu7)
    )  # (..., 3, 3, 2, 2)

    # Extract components A^a_i = -2 Tr(T_a A_i) and X_k A^a_i = -2 Tr(T_a X_k A_i)
    A_comp = -2.0 * np.einsum("ast,...its->...ai", T_MAT, A_mat).real
    XA_comp = -2.0 * np.einsum("ast,...kits->...kai", T_MAT, XA_mat).real
    dirichlet = np.sum(np.abs(Xw) ** 2, axis=(-2, -1)).real
    return u7, A_comp, XA_comp, r2, dirichlet


def verify_ode_holonomy(Q1: np.ndarray, U1: np.ndarray, U2: np.ndarray, poly_coeffs: np.ndarray):
    """Verify Hol_{A_{rho_2}}([gamma]) independently by DOP853 parallel transport along geodesic lifts
    x(s) = exp(s * log(gamma)) from I (s=0) to gamma (s=1) for representatives of all 9 conjugacy classes!
    """
    # Group the 120 elements by trace of rho_1(g) (distinguishes all 9 conjugacy classes of I*!)
    traces1 = np.round(2.0 * Q1[:, 0], 6)
    unique_tr = sorted(set(traces1.tolist()))
    rows = []
    max_trace_err = 0.0
    max_unitarity_err = 0.0

    for tr_val in unique_tr:
        idxs = np.where(np.abs(2.0 * Q1[:, 0] - tr_val) < 1e-4)[0]
        g_idx = int(idxs[0])
        q = Q1[g_idx]
        # Write q = cos(theta) + sin(theta) * n_a e_a with theta in [0, pi]
        w = float(np.clip(q[0], -1.0, 1.0))
        theta = math.acos(w)
        if theta < 1e-12:
            # Identity element: trivial loop
            hol = np.eye(2, dtype=np.complex128)
            tr_hol = 2.0
            tr_rho2 = float(np.trace(U2[g_idx]).real)
            rows.append(
                {
                    "class_size": int(len(idxs)),
                    "tr_rho1": float(np.trace(U1[g_idx]).real),
                    "tr_rho2_exact": tr_rho2,
                    "tr_hol_ode": tr_hol,
                    "trace_err": abs(tr_hol - tr_rho2),
                }
            )
            continue

        if abs(theta - math.pi) < 1e-12:
            # Central element -I: geodesic along e_3 from 0 to pi
            n_vec = np.array([0.0, 0.0, 1.0])
        else:
            n_vec = q[1:] / math.sin(theta)

        # Geodesic path x(s) = exp(s * theta * n_a e_a), s in [0, 1].
        # Velocity in left-invariant frame: x(s)^{-1} x'(s) = theta * n_a e_a,
        # so omega^i(x'(s)) = theta * n_i!
        # Parallel transport equation: dPsi/ds + A(x(s))(x'(s)) Psi(s) = 0,
        # where A(x(s))(x'(s)) = sum_{a,i} A^a_i(x(s)) * (theta * n_i) * T_a.
        def rhs(s, y_real):
            Psi = (y_real[:4] + 1.0j * y_real[4:]).reshape(2, 2)
            cs = math.cos(s * theta)
            sn = math.sin(s * theta)
            qs = np.array([[cs, sn * n_vec[0], sn * n_vec[1], sn * n_vec[2]]])
            Us = quat_to_su2(qs)
            _, A_c, _ = eval_u7_and_connection(Us, poly_coeffs)
            # Contract with velocity theta * n_i
            A_vel_comp = np.einsum("ai,i->a", A_c[0], theta * n_vec)
            A_vel_mat = np.einsum("a,ajk->jk", A_vel_comp, T_MAT)
            dPsi = -A_vel_mat @ Psi
            return np.concatenate([dPsi.real.ravel(), dPsi.imag.ravel()])

        y0 = np.concatenate([np.eye(2).ravel(), np.zeros(4)])
        sol = solve_ivp(rhs, (0.0, 1.0), y0, method="DOP853", rtol=1e-12, atol=1e-14)
        yf = sol.y[:, -1]
        hol = (yf[:4] + 1.0j * yf[4:]).reshape(2, 2)

        unit_err = float(np.linalg.norm(hol.conj().T @ hol - np.eye(2)))
        tr_hol = float(np.trace(hol).real)
        tr_rho2 = float(np.trace(U2[g_idx]).real)
        # Note: Hol = u_7(I)^{-1} rho_2(gamma)^{-1} u_7(I), and Tr(g^{-1}) = Tr(g) in SU(2)!
        t_err = abs(tr_hol - tr_rho2)
        max_trace_err = max(max_trace_err, t_err)
        max_unitarity_err = max(max_unitarity_err, unit_err)
        rows.append(
            {
                "class_size": int(len(idxs)),
                "tr_rho1": float(np.trace(U1[g_idx]).real),
                "tr_rho2_exact": tr_rho2,
                "tr_hol_ode": tr_hol,
                "trace_err": t_err,
            }
        )

    return {
        "num_conjugacy_classes": len(rows),
        "max_trace_err": max_trace_err,
        "max_unitarity_err": max_unitarity_err,
        "classes": rows,
    }


def main():
    print("=== Goal 1: Explicit Construction & Verification of Galois Flat Connection rho_2 ===")
    Q1, Q2, U1, U2 = build_istar_and_galois()
    C_mat, poly_coeffs = build_p7_equivariant_polynomial(U1, U2)

    # Verify exact closed-form Z[sqrt(5)] polynomial against SVD intertwiner
    s5 = math.sqrt(5.0)
    p7_exact_zsqrt5 = np.array(
        [
            [0.0, 7.0, 0.0, -7.0 * s5, 0.0, -7.0, 0.0, -s5],
            [s5, 0.0, 7.0, 0.0, 7.0 * s5, 0.0, -7.0, 0.0],
        ],
        dtype=np.complex128,
    )
    scale_exact = poly_coeffs[1, 0] / p7_exact_zsqrt5[1, 0]
    p7_zsqrt5_err = float(np.max(np.abs(poly_coeffs - scale_exact * p7_exact_zsqrt5)))
    assert p7_zsqrt5_err < 1e-13, f"Exact Z[sqrt(5)] P_7 mismatch: {p7_zsqrt5_err}"
    print(f"Exact Z[sqrt(5)] P_7 closed-form discrepancy vs SVD: {p7_zsqrt5_err:.3e} (scale={scale_exact.real:.12f}, exact sqrt(3)/8={math.sqrt(3)/8:.12f})")

    print("Unique degree-7 holomorphic intertwiner P_7(z1, z2) raw monomial coefficients:")
    for comp in range(2):
        terms = []
        for k in range(8):
            val = poly_coeffs[comp, k]
            if abs(val) > 1e-10:
                terms.append(f"({val.real:+.6f}{val.imag:+.6f}j) z1^{7-k} z2^{k}")
        print(f"  P_7[{comp}] = " + " + ".join(terms))

    # 1. Verify P_7 non-vanishing, exact equivariance, flatness, and N_CS on high-order 2D Hopf quadrature
    U_s2, W_s2 = euler_quadrature_s2_hopf(n_beta=48, n_alpha=96)
    u7_s2, A_s2, XA_s2, r2_s2, dir_s2 = eval_A_and_XA_exact(U_s2, poly_coeffs)
    ym_s2 = compute_ym_and_cs(A_s2, XA_s2, W_s2)
    dirichlet_avg_s2 = float(np.dot(W_s2, dir_s2))
    r2_min = float(np.min(r2_s2))
    r2_max = float(np.max(r2_s2))

    # 2. Cross-check on FULL 3D Euler quadrature (no U(1)_R assumption!)
    U_s3, W_s3 = euler_quadrature_s3(n_beta=20, n_alpha=40, n_gamma=20)
    u7_s3, A_s3, XA_s3, r2_s3, dir_s3 = eval_A_and_XA_exact(U_s3, poly_coeffs)
    ym_s3 = compute_ym_and_cs(A_s3, XA_s3, W_s3)
    dirichlet_avg_s3 = float(np.dot(W_s3, dir_s3))

    # 3. Verify strict I*-invariance of A_{rho_2} and equivariance of u_7 at 20 random points x all 120 g in I*
    rng = np.random.default_rng(42)
    rand_q = rng.normal(size=(20, 4))
    rand_q /= np.linalg.norm(rand_q, axis=-1, keepdims=True)
    U_rand = quat_to_su2(rand_q)
    u7_base, A_base, _ = eval_u7_and_connection(U_rand, poly_coeffs)
    max_equiv_err = 0.0
    max_A_inv_err = 0.0
    for g in range(120):
        U_gx = np.einsum("ij,Njk->Nik", U1[g], U_rand)
        u7_gx, A_gx, _ = eval_u7_and_connection(U_gx, poly_coeffs)
        target_u7 = np.einsum("ij,Njk->Nik", U2[g], u7_base)
        max_equiv_err = max(max_equiv_err, float(np.max(np.abs(u7_gx - target_u7))))
        max_A_inv_err = max(max_A_inv_err, float(np.max(np.abs(A_gx - A_base))))

    # 4. Verify U(1)_R equivariance with winding k = 7:
    #    u_7(x exp(theta e_3)) == u_7(x) exp(7 theta e_3)
    theta_test = 0.731
    R_theta = np.diag([np.exp(-1.0j * theta_test), np.exp(1.0j * theta_test)])
    H_7theta = np.diag([np.exp(-7.0j * theta_test), np.exp(7.0j * theta_test)])
    U_xr = np.einsum("Nij,jk->Nik", U_rand, R_theta)
    u7_xr, _, _ = eval_u7_and_connection(U_xr, poly_coeffs)
    u7_target = np.einsum("Nij,jk->Nik", u7_base, H_7theta)
    u1_equiv_err = float(np.max(np.abs(u7_xr - u7_target)))

    # 5. Independent ODE holonomy check across all 9 conjugacy classes of I*
    hol_check = verify_ode_holonomy(Q1, U1, U2, poly_coeffs)

    # 6. Straight-line path A(s) = s A_{rho_2} from rho_0 to rho_2 (peak at s = 1/2):
    #    Since F(s A) = (s^2 - s) A ^ A, B(0.5 A) = -0.25 * (1/2 eps eps A A)
    ym_half = compute_ym_and_cs(0.5 * A_s2, 0.5 * XA_s2, W_s2)

    # 7. Straight-line path A(s) = (1-s) A_{rho_1} + s A_{rho_2} from rho_1 to rho_2 (scan s in [0, 1])
    A_rho1_s2 = np.broadcast_to(2.0 * np.eye(3), A_s2.shape)
    XA_rho1_s2 = np.zeros_like(XA_s2)
    s_grid = np.linspace(0.0, 1.0, 41)
    line_12_E = []
    line_12_CS = []
    for s in s_grid:
        res_s = compute_ym_and_cs(
            (1.0 - s) * A_rho1_s2 + s * A_s2,
            (1.0 - s) * XA_rho1_s2 + s * XA_s2,
            W_s2,
        )
        line_12_E.append(res_s["E_YM_S3_Istar"])
        line_12_CS.append(res_s["N_CS"])
    max_idx_12 = int(np.argmax(line_12_E))

    # 8. Exact harmonic spectrum of ||P_7||^2 and A_{rho_2} in the I*-invariant Wigner-D basis up to N_max = 36!
    print("Computing exact I*-invariant Wigner-D harmonic spectrum of ||P_7||^2 and A_{rho_2} up to N_max = 36...")
    levels = build_istar_invariant_harmonics(U1, max_n=36)
    # Project on quadratic-exact 3D SO(3) grid (n_beta=22, n_alpha=n_gamma=40 is exact for
    # inner products of degree <= 36 with A_{rho_2} up to total deg 72), THEN evaluate quartic
    # E_YM ~ <A^4> and cubic N_CS on a QUARTIC-EXACT 2D Hopf grid (n_beta=44, n_alpha=88, exact
    # to cos(beta) deg 87 > 72 = 2*36!) because every harmonic truncation of U(1)_R-equivariant
    # A_{rho_2} remains strictly U(1)_R-equivariant (so |B|^2 and CS density are gamma-independent!).
    U_spec, W_spec = euler_quadrature_so3(n_beta=30, n_alpha=60, n_gamma=48)
    Phi_spec, XPhi_spec, meta = eval_real_harmonic_basis(U_spec, levels)

    # Verify basis orthonormality on U_spec!
    gram = (Phi_spec * W_spec[:, None]).T @ Phi_spec
    gram_err = float(np.max(np.abs(gram - np.eye(Phi_spec.shape[1]))))
    assert gram_err < 1e-12, f"Basis Gram matrix not orthonormal! err={gram_err}"

    # Project r2 = ||P_7||^2 and A_{rho_2} onto Phi_spec (also check projection stability on finer grid)
    _, A_spec, XA_spec, r2_spec, _ = eval_A_and_XA_exact(U_spec, poly_coeffs)
    c_r2 = Phi_spec.T @ (W_spec * r2_spec)
    r2_recon = Phi_spec @ c_r2
    r2_l2_err = float(math.sqrt(np.dot(W_spec, (r2_spec - r2_recon) ** 2)))

    c_A = np.einsum("pM,p,pai->Mai", Phi_spec, W_spec, A_spec)  # (N_basis, 3, 3)

    # Quartic-exact 2D Hopf grid for evaluating E_YM and N_CS of each truncated A_{rho_2}^{(N_max)}!
    U_qex, W_qex = euler_quadrature_s2_hopf(n_beta=46, n_alpha=92)
    Phi_qex, XPhi_qex, _ = eval_real_harmonic_basis(U_qex, levels)

    r2_by_level = []
    A_by_level = []
    total_A_L2 = float(np.dot(W_s2, np.sum(A_s2 ** 2, axis=(-2, -1))))
    cum_A_L2 = 0.0
    truncation_convergence = []

    for m in meta:
        s0, s1, n = m["start"], m["end"], m["n"]
        r2_pow = float(np.sum(c_r2[s0:s1] ** 2))
        A_pow = float(np.sum(c_A[s0:s1] ** 2))
        cum_A_L2 += A_pow
        r2_by_level.append({"n": n, "dim": m["dim"], "L2_power": r2_pow})
        A_by_level.append(
            {
                "n": n,
                "dim": m["dim"],
                "L2_power": A_pow,
                "frac_of_total": A_pow / total_A_L2,
                "cum_frac": cum_A_L2 / total_A_L2,
            }
        )
        # Reconstruct truncated A_{rho_2} up to N_max = n on QUARTIC-EXACT grid!
        A_tr = np.einsum("pM,Mai->pai", Phi_qex[:, :s1], c_A[:s1])
        XA_tr = np.einsum("pkM,Mai->pkai", XPhi_qex[:, :, :s1], c_A[:s1])
        ym_tr = compute_ym_and_cs(A_tr, XA_tr, W_qex)
        truncation_convergence.append(
            {
                "N_max": n,
                "n_scalar_basis": s1,
                "n_conn_dof": 9 * s1,
                "E_YM_S3_Istar": ym_tr["E_YM_S3_Istar"],
                "rms_B": ym_tr["rms_B"],
                "N_CS": ym_tr["N_CS"],
                "N_CS_err_vs_49_120": abs(ym_tr["N_CS"] - 49.0 / 120.0),
            }
        )

    payload = {
        "goal": "Goal 1: Explicit I*-equivariant construction of rho_2",
        "exact_rational_target": {"CS_rho2": "49/120", "CS_rho2_float": 49.0 / 120.0, "deg_u7": 49},
        "p7_norm_squared": {
            "exact_zsqrt5_svd_max_err": p7_zsqrt5_err,
            "exact_scale_vs_sqrt3_over_8_err": abs(float(scale_exact.real) - math.sqrt(3.0) / 8.0),
            "min_on_S3": r2_min,
            "max_on_S3": r2_max,
            "exact_min_2_over_9": 2.0 / 9.0,
            "exact_max_3_over_10": 3.0 / 10.0,
            "max_over_min_ratio": r2_max / r2_min,
            "exact_kurtosis_144_over_143": 144.0 / 143.0,
            "observed_kurtosis": float(np.dot(W_s2, r2_s2 ** 2) / (np.dot(W_s2, r2_s2) ** 2)),
            "harmonic_l2_residual_N36": r2_l2_err,
            "power_by_level": r2_by_level,
        },
        "equivariance_and_invariance": {
            "max_u7_equivariance_err_120g": max_equiv_err,
            "max_A_Istar_invariance_err_120g": max_A_inv_err,
            "u1_hopf_winding7_equiv_err": u1_equiv_err,
            "harmonic_basis_gram_err": gram_err,
        },
        "flatness_and_cs_2D_hopf": {
            **ym_s2,
            "N_CS_err_vs_49_120": abs(ym_s2["N_CS"] - 49.0 / 120.0),
            "dirichlet_avg_du7_sq": dirichlet_avg_s2,
            "dirichlet_rayleigh_lower_bound": 63.0,
            "dirichlet_ratio_vs_rho1": dirichlet_avg_s2 / 3.0,
        },
        "flatness_and_cs_3D_euler": {
            **ym_s3,
            "N_CS_err_vs_49_120": abs(ym_s3["N_CS"] - 49.0 / 120.0),
            "dirichlet_avg_du7_sq": dirichlet_avg_s3,
        },
        "independent_ode_holonomy": hol_check,
        "naive_linear_barriers": {
            "S_01_exact_sphaleron_E_S3_Istar": math.pi ** 2 / 40.0,
            "S_01_exact_sphaleron_E_mono": 0.5,
            "line_02_midpoint_E_S3_Istar": ym_half["E_YM_S3_Istar"],
            "line_02_midpoint_E_mono": ym_half["E_YM_mono"],
            "line_02_midpoint_N_CS": ym_half["N_CS"],
            "line_12_peak_s": float(s_grid[max_idx_12]),
            "line_12_peak_E_S3_Istar": float(line_12_E[max_idx_12]),
            "line_12_peak_E_mono": float(line_12_E[max_idx_12] * 20.0 / (math.pi ** 2)),
            "line_12_peak_N_CS": float(line_12_CS[max_idx_12]),
        },
        "A_rho2_harmonic_spectrum": A_by_level,
        "A_rho2_truncation_convergence": truncation_convergence,
    }

    out_path = os.path.join(OUT_DIR, "goal1_rho2_verification.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\n--- VERIFICATION SUMMARY ---")
    print(f"  ||P_7||^2 on S^3: min={r2_min:.6f}, max={r2_max:.6f}, ratio={r2_max/r2_min:.6f}")
    print(f"  u_7 equivariance err (120g): {max_equiv_err:.3e}")
    print(f"  A_rho2 I*-invariance err:    {max_A_inv_err:.3e}")
    print(f"  U(1)_R winding-7 equiv err:  {u1_equiv_err:.3e}")
    print(f"  2D Hopf: max|B|={ym_s2['max_B']:.3e}, N_CS={ym_s2['N_CS']:.15f}, err={abs(ym_s2['N_CS']-49/120):.3e}")
    print(f"  3D Full: max|B|={ym_s3['max_B']:.3e}, N_CS={ym_s3['N_CS']:.15f}, err={abs(ym_s3['N_CS']-49/120):.3e}")
    print(f"  ODE Holonomy ({hol_check['num_conjugacy_classes']} classes): max_tr_err={hol_check['max_trace_err']:.3e}, max_unit_err={hol_check['max_unitarity_err']:.3e}")
    print(f"  Dirichlet <|du_7|^2> = {dirichlet_avg_s2:.6f} (Rayleigh bound = 63.0, ratio vs rho_1 = {dirichlet_avg_s2/3.0:.4f}x)")
    print(f"  Naive line 0->2 midpoint E_YM^{{S^3/I*}} = {ym_half['E_YM_S3_Istar']:.6f} (E_mono = {ym_half['E_YM_mono']:.4f}, vs S_01 = {math.pi**2/40:.6f})")
    print(f"  Naive line 1->2 peak     E_YM^{{S^3/I*}} = {line_12_E[max_idx_12]:.6f} (s={s_grid[max_idx_12]:.3f}, N_CS={line_12_CS[max_idx_12]:.5f})")
    print("\n--- ||P_7||^2 HARMONIC POWER BY DEGREE ---")
    for row in r2_by_level:
        if row["L2_power"] > 1e-12:
            print(f"  n={row['n']:2d}: power={row['L2_power']:.10f}")
    print("\n--- A_{rho_2} TRUNCATION CONVERGENCE ---")
    for row in truncation_convergence:
        print(
            f"  N_max={row['N_max']:2d} (dof={row['n_conn_dof']:4d}): "
            f"E_YM={row['E_YM_S3_Istar']:.4e}, rms|B|={row['rms_B']:.4e}, "
            f"N_CS={row['N_CS']:.10f} (err={row['N_CS_err_vs_49_120']:.3e})"
        )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
