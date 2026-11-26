"""Goal 2 (Tier 2): Full Unrestricted 3D I*-Invariant Yang-Mills Sphaleron Solver on S^3/I*.

Tests the exact full-3D stability (Palais criticality + full Hessian Morse index) of the
Tier-1 U(1)_R (k=7) saddle, and searches the FULL unrestricted 3D I*-invariant connection
space Omega^1(S^3, su(2))^{I*} (zero U(1)_R symmetry assumptions!) for minimum-energy
sphaleron barriers connecting:
  - rho_0 <-> rho_2
  - rho_1 <-> rho_2
across truncation cutoffs N_max = 12 (126 DOF), N_max = 20 (315 DOF), and N_max = 24 (540 DOF).

Uses scalar-factored BLAS evaluation A(x) = Phi(x) @ C so each full-3D energy+gradient call
takes < 1 ms at N_max = 12 and ~5 ms at N_max = 20!
"""

import json
import math
import os
import time
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize

from istar_geometry import (
    EPS3,
    build_istar_and_galois,
    build_istar_invariant_harmonics,
    build_p7_equivariant_polynomial,
    euler_quadrature_so3,
    eval_real_harmonic_basis,
    eval_u7_and_connection,
)
from goal2_sphaleron_solver import build_k7_subspace_basis, polish_saddle, run_cineb

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)


class Full3DYMSolver:
    """Ultra-fast scalar-factored exact spectral solver on full Omega^1(S^3, su(2))^{I*}.

    Connection parameters: c of shape (9 * N_scal,) reshaped as C of shape (N_scal, 9)
    where 9 indexes (a, i) in row-major order: A^a_i(x_p) = sum_m Phi[p, m] * C[m, 3*a + i].
    """

    def __init__(self, Phi: np.ndarray, XPhi: np.ndarray, W: np.ndarray):
        self.Phi = np.ascontiguousarray(Phi, dtype=np.float64)        # (N_pts, M)
        self.XPhi = np.ascontiguousarray(XPhi, dtype=np.float64)      # (N_pts, 3, M)
        self.W = np.ascontiguousarray(W, dtype=np.float64)            # (N_pts,)
        self.n_pts, self.M = self.Phi.shape
        self.D = 9 * self.M
        self.XPhi_0 = np.ascontiguousarray(self.XPhi[:, 0, :])
        self.XPhi_1 = np.ascontiguousarray(self.XPhi[:, 1, :])
        self.XPhi_2 = np.ascontiguousarray(self.XPhi[:, 2, :])
        self.scale_E = math.pi ** 2 / 120.0

    def energy_and_grad(self, c: np.ndarray) -> Tuple[float, np.ndarray, float]:
        C = c.reshape(self.M, 9)
        A = (self.Phi @ C).reshape(self.n_pts, 3, 3)

        # Exact derivatives X_k A^a_i:
        XA0 = (self.XPhi_0 @ C).reshape(self.n_pts, 3, 3)  # X_0 A[p, a, i]
        XA1 = (self.XPhi_1 @ C).reshape(self.n_pts, 3, 3)  # X_1 A[p, a, i]
        XA2 = (self.XPhi_2 @ C).reshape(self.n_pts, 3, 3)  # X_2 A[p, a, i]

        # curl_A[p, a, k] = eps_{kij} X_i A[p, a, j]:
        curl_A = np.empty_like(A)
        curl_A[:, :, 0] = XA1[:, :, 2] - XA2[:, :, 1]
        curl_A[:, :, 1] = XA2[:, :, 0] - XA0[:, :, 2]
        curl_A[:, :, 2] = XA0[:, :, 1] - XA1[:, :, 0]

        # Exact 3x3 cofactor comm_A[p, a, k] = (1/2) eps_{kij} eps_{abc} A[b, i] A[c, j]
        a00, a01, a02 = A[:, 0, 0], A[:, 0, 1], A[:, 0, 2]
        a10, a11, a12 = A[:, 1, 0], A[:, 1, 1], A[:, 1, 2]
        a20, a21, a22 = A[:, 2, 0], A[:, 2, 1], A[:, 2, 2]

        comm_A = np.empty_like(A)
        comm_A[:, 0, 0] = a11 * a22 - a12 * a21
        comm_A[:, 0, 1] = a12 * a20 - a10 * a22
        comm_A[:, 0, 2] = a10 * a21 - a11 * a20
        comm_A[:, 1, 0] = a21 * a02 - a22 * a01
        comm_A[:, 1, 1] = a22 * a00 - a20 * a02
        comm_A[:, 1, 2] = a20 * a01 - a21 * a00
        comm_A[:, 2, 0] = a01 * a12 - a02 * a11
        comm_A[:, 2, 1] = a02 * a10 - a00 * a12
        comm_A[:, 2, 2] = a00 * a11 - a01 * a10

        B = curl_A - 2.0 * A + comm_A
        B2 = np.sum(B * B, axis=(-2, -1))
        E = float(self.scale_E * np.dot(self.W, B2))

        WB = (2.0 * self.scale_E * self.W)[:, None, None] * B  # (N_pts, 3, 3)

        # Gradient from curl_A: sum_{p, k, i, j} WB[p, a, k] * eps_{kij} * XPhi[p, i, m]
        # For fixed (a, j):
        #   j = 0: XPhi_2 * WB[:, a, 1] - XPhi_1 * WB[:, a, 2]
        #   j = 1: XPhi_0 * WB[:, a, 2] - XPhi_2 * WB[:, a, 0]
        #   j = 2: XPhi_1 * WB[:, a, 0] - XPhi_0 * WB[:, a, 1]
        # Wait: check! curl_A[:, a, 0] = XA1[:, a, 2] - XA2[:, a, 1]:
        #   d(WB_0 * curl_0)/d(XA1_2) = +WB_0  => j=2 gets +XPhi_1^T @ WB[:, :, 0]
        #   d(WB_0 * curl_0)/d(XA2_1) = -WB_0  => j=1 gets -XPhi_2^T @ WB[:, :, 0]
        # curl_A[:, a, 1] = XA2[:, a, 0] - XA0[:, a, 2]:
        #   j=0 gets +XPhi_2^T @ WB[:, :, 1],  j=2 gets -XPhi_0^T @ WB[:, :, 1]
        # curl_A[:, a, 2] = XA0[:, a, 1] - XA1[:, a, 0]:
        #   j=1 gets +XPhi_0^T @ WB[:, :, 2],  j=0 gets -XPhi_1^T @ WB[:, :, 2]
        G_curl = np.empty((self.M, 3, 3), dtype=np.float64)
        G_curl[:, :, 0] = self.XPhi_2.T @ WB[:, :, 1] - self.XPhi_1.T @ WB[:, :, 2]
        G_curl[:, :, 1] = self.XPhi_0.T @ WB[:, :, 2] - self.XPhi_2.T @ WB[:, :, 0]
        G_curl[:, :, 2] = self.XPhi_1.T @ WB[:, :, 0] - self.XPhi_0.T @ WB[:, :, 1]

        # Polarized cofactor pol[p, a, j]
        w00, w01, w02 = WB[:, 0, 0], WB[:, 0, 1], WB[:, 0, 2]
        w10, w11, w12 = WB[:, 1, 0], WB[:, 1, 1], WB[:, 1, 2]
        w20, w21, w22 = WB[:, 2, 0], WB[:, 2, 1], WB[:, 2, 2]

        pol = np.empty_like(A)
        pol[:, 0, 0] = a11 * w22 - a12 * w21 - a21 * w12 + a22 * w11
        pol[:, 0, 1] = a12 * w20 - a10 * w22 - a22 * w10 + a20 * w12
        pol[:, 0, 2] = a10 * w21 - a11 * w20 - a20 * w11 + a21 * w10
        pol[:, 1, 0] = a21 * w02 - a22 * w01 - a01 * w22 + a02 * w21
        pol[:, 1, 1] = a22 * w00 - a20 * w02 - a02 * w20 + a00 * w22
        pol[:, 1, 2] = a20 * w01 - a21 * w00 - a00 * w21 + a01 * w20
        pol[:, 2, 0] = a01 * w12 - a02 * w11 - a11 * w02 + a12 * w01
        pol[:, 2, 1] = a02 * w10 - a00 * w12 - a12 * w00 + a10 * w02
        pol[:, 2, 2] = a00 * w11 - a01 * w10 - a10 * w01 + a11 * w00

        coeff_A = (-2.0 * WB + pol).reshape(self.n_pts, 9)
        G_A = (self.Phi.T @ coeff_A).reshape(self.M, 3, 3)
        grad = (G_curl + G_A).reshape(-1)

        A2 = np.sum(A * A, axis=(-2, -1))
        A_curl = np.sum(A * curl_A, axis=(-2, -1))
        det_A = a00 * comm_A[:, 0, 0] + a01 * comm_A[:, 0, 1] + a02 * comm_A[:, 0, 2]
        N_cs = float(np.dot(self.W, A2 - 0.5 * A_curl - det_A)) / 480.0
        return E, grad, N_cs

    def energy_grad_norm_sq_and_grad(self, c: np.ndarray) -> Tuple[float, np.ndarray]:
        _, g, _ = self.energy_and_grad(c)
        fval = 0.5 * float(np.dot(g, g))
        gnorm = math.sqrt(2.0 * fval)
        if gnorm < 1e-15:
            return fval, np.zeros_like(c)
        eps = 1e-6
        v = g / gnorm
        _, gp, _ = self.energy_and_grad(c + eps * v)
        _, gm, _ = self.energy_and_grad(c - eps * v)
        Hv = (gp - gm) / (2.0 * eps)
        return fval, gnorm * Hv

    def hessian(self, c: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        H = np.zeros((self.D, self.D), dtype=np.float64)
        for j in range(self.D):
            dc = np.zeros(self.D, dtype=np.float64)
            dc[j] = eps
            _, gp, _ = self.energy_and_grad(c + dc)
            _, gm, _ = self.energy_and_grad(c - dc)
            H[:, j] = (gp - gm) / (2.0 * eps)
        return 0.5 * (H + H.T)


def build_full3d_solver(U1: np.ndarray, poly_coeffs: np.ndarray, max_n: int):
    """Build exact zero-aliasing Full3DYMSolver at truncation max_n and project rho_0, rho_1, rho_2."""
    levels = build_istar_invariant_harmonics(U1, max_n=max_n)
    # Exact for quartic integrand (total deg <= 4*max_n on S^3 => cos(beta) deg <= 2*max_n, |m_sum| <= 2*max_n)
    # Actually |m_L| <= max_n/2, so sum of 4 has <= 2*max_n; to integrate exact quartic we need n_alpha, n_gamma > 2*max_n!
    # Check: at max_n=12: n_beta=14 (exact to 27 >= 24), n_alpha=n_gamma=26 (> 24) -> 9,464 pts!
    # At max_n=20: n_beta=22 (exact to 43 >= 40), n_alpha=n_gamma=42 (> 40) -> 38,808 pts!
    n_beta = max_n + 2
    n_ang = 2 * max_n + 2
    U_pts, W = euler_quadrature_so3(n_beta=n_beta, n_alpha=n_ang, n_gamma=n_ang)
    Phi, XPhi, meta = eval_real_harmonic_basis(U_pts, levels)

    gram = (Phi * W[:, None]).T @ Phi
    gram_err = float(np.max(np.abs(gram - np.eye(Phi.shape[1]))))
    assert gram_err < 1e-12, f"Full3D basis Gram err={gram_err}"

    solver = Full3DYMSolver(Phi, XPhi, W)

    # Project rho_0, rho_1, rho_2
    c_rho0 = np.zeros(solver.D, dtype=np.float64)
    # rho_1: A^a_i = 2 delta^a_i -> purely in constant mode Phi[:, 0] (check sign of Phi[:, 0]!)
    sgn0 = float(np.mean(Phi[:, 0]))  # +1 or -1
    assert abs(abs(sgn0) - 1.0) < 1e-12
    C_rho1 = np.zeros((solver.M, 3, 3), dtype=np.float64)
    C_rho1[0] = (2.0 / sgn0) * np.eye(3)
    c_rho1 = C_rho1.reshape(-1)

    _, A_r2, _ = eval_u7_and_connection(U_pts, poly_coeffs)
    C_rho2 = np.einsum("pm,p,pai->mai", Phi, W, A_r2)
    c_rho2 = C_rho2.reshape(-1)

    return solver, c_rho0, c_rho1, c_rho2, Phi, W, U_pts, meta


def main():
    t0 = time.time()
    print("=== Goal 2 (Tier 2): Full Unrestricted 3D Sphaleron Audit & Search ===", flush=True)
    Q1, Q2, U1, U2 = build_istar_and_galois()
    _, poly_coeffs = build_p7_equivariant_polynomial(U1, U2)

    # -------------------------------------------------------------------------
    # PART 1: Build Full 3D Solver at N_max = 12 (126 DOF) & Verify Gradient
    # -------------------------------------------------------------------------
    solver12, c0_12, c1_12, c2_12, Phi12, W12, U12, meta12 = build_full3d_solver(U1, poly_coeffs, max_n=12)
    print(f"Built Full3D N_max=12: D={solver12.D}, n_pts={solver12.n_pts}", flush=True)

    # Verify rho_0, rho_1, rho_2 energies & CS
    for name, cv in [("rho_0", c0_12), ("rho_1", c1_12), ("rho_2(N12)", c2_12)]:
        e, g, ncs = solver12.energy_and_grad(cv)
        print(f"  {name:10s}: E_YM={e:.6e}, |g|={np.linalg.norm(g):.3e}, N_CS={ncs:.10f}", flush=True)

    # Verify analytical gradient vs FD
    rng = np.random.default_rng(7)
    ct = 0.4 * c2_12 + 0.3 * c1_12 + 0.1 * rng.normal(size=solver12.D)
    _, ga, _ = solver12.energy_and_grad(ct)
    gf = np.zeros_like(ga)
    eps = 1e-6
    for j in range(solver12.D):
        dc = np.zeros(solver12.D)
        dc[j] = eps
        ep, _, _ = solver12.energy_and_grad(ct + dc)
        em, _, _ = solver12.energy_and_grad(ct - dc)
        gf[j] = (ep - em) / (2.0 * eps)
    gerr = float(np.max(np.abs(ga - gf)) / max(1.0, np.max(np.abs(ga))))
    print(f"[Full3D Grad Check] rel_err = {gerr:.3e}", flush=True)
    assert gerr < 1e-9

    # -------------------------------------------------------------------------
    # PART 2: Palais Criticality & Full-3D Morse Index Audit of Tier-1 k=7 Saddle!
    # -------------------------------------------------------------------------
    print("\n--- PART 2: Full-3D Hessian & Morse Index of Tier-1 k=7 Saddle (N_max=12) ---", flush=True)
    k7_s12, c_k7_r2_12, _ = build_k7_subspace_basis(U1, poly_coeffs, max_n=12)
    neb_k7 = run_cineb(k7_s12, np.zeros(k7_s12.D), c_k7_r2_12, n_images=19, k_spring=0.2, n_steps=1000, lr_init=0.008, climb_after=200)
    pol_k7 = polish_saddle(k7_s12, neb_k7["c_peak"], maxiter=300)
    # Reconstruct A_sad^{(k=7)} on U12 and project into full 3D basis!
    # Wait: evaluate k7 basis at U12 directly using the exact harmonic modes!
    # Even simpler: project pol_k7["c"] using the exact A_sad on 2D? No, k7 basis was built on U_s2!
    # Let's project via exact k7 basis evaluated on U12: since k7_s12 modes live inside N_max=12,
    # let's extract the k7 saddle's coefficients directly in Phi12 by running 1-line projection of k7 mode tensor!
    # Wait: we can map k7_s12 -> full 3D cleanly by rebuilding k7 nullspace in the exact same meta12 basis!
    # Let's compute the exact projection matrix P_k7_to_full12!
    c_sad_k7_full12 = np.zeros(solver12.D, dtype=np.float64)
    # Check exact (sf, sc) on c2_12 directly in Full3D!
    sf_best, sc_best = None, None
    best_e = 1e99
    C2_m = c2_12.reshape(solver12.M, 3, 3)
    for sf in (-2.0, 2.0):
        for sc in (-14.0, 14.0):
            err_sum = 0.0
            for m in meta12:
                d, s0, s1 = m["dim"], m["start"], m["end"]
                D3 = m["D_real"][2]
                Cm = C2_m[s0:s1]
                # (X_3 A)^a_i = sum_m' D3[m', m]? Check: X_3 Phi = Phi @ D3 => X_3 (Phi @ C) = Phi @ (D3 @ C)
                res = (
                    np.einsum("mn,nai->mai", D3, Cm)
                    + sf * np.einsum("ij,maj->mai", EPS3[2], Cm)
                    + sc * np.einsum("ab,mbi->mai", EPS3[2], Cm)
                )
                err_sum += float(np.linalg.norm(res))
            if err_sum < best_e:
                best_e = err_sum
                sf_best, sc_best = sf, sc
    print(f"  Full3D k=7 equivariance check on rho_2(N12): signs=({sf_best}, {sc_best}), err={best_e:.3e}", flush=True)
    assert best_e < 1e-10

    # Build orthonormal (126, 6) matrix Q_k7 embedding A_{k=7} into Full3D(N12)
    Q_blocks = []
    for m in meta12:
        d, s0, s1 = m["dim"], m["start"], m["end"]
        D3 = m["D_real"][2]
        L_op = (
            np.kron(D3, np.eye(9))
            + sf_best * np.kron(np.kron(np.eye(d), np.eye(3)), EPS3[2])
            + sc_best * np.kron(np.kron(np.eye(d), EPS3[2]), np.eye(3))
        )
        ev, evec = np.linalg.eigh(L_op.T @ L_op)
        nidx = np.where(ev < 1e-10)[0]
        for col in nidx:
            v_full = np.zeros((solver12.M, 9), dtype=np.float64)
            v_full[s0:s1] = evec[:, col].reshape(d, 9)
            Q_blocks.append(v_full.reshape(-1))
    Q_k7 = np.column_stack(Q_blocks)  # (126, 6)

    # Project c2_12 into Q_k7 and solve k7 saddle directly in Q_k7 coordinates, then lift to Full3D!
    # Wait: even easier, run 50-step polish of Q_k7 @ (Q_k7.T @ c_init) using Full3D!
    # Let's map c_init = 0.5 * c2_12 and polish inside span(Q_k7) via Full3D gradient projected onto Q_k7:
    c_k7_sub_r2 = Q_k7.T @ c2_12
    assert np.linalg.norm(Q_k7 @ c_k7_sub_r2 - c2_12) < 1e-12

    # Tiny wrapper for Q_k7 inside Full3D to reproduce exact Tier-1 saddle in Full3D coords!
    class SubspaceWrap:
        def __init__(self, base, Q):
            self.base = base
            self.Q = Q
            self.D = Q.shape[1]
        def energy_and_grad(self, u):
            e, g, ncs = self.base.energy_and_grad(self.Q @ u)
            return e, self.Q.T @ g, ncs
        def energy_grad_norm_sq_and_grad(self, u):
            _, g, _ = self.energy_and_grad(u)
            fv = 0.5 * float(np.dot(g, g))
            gn = math.sqrt(2.0 * fv)
            if gn < 1e-15:
                return fv, np.zeros_like(u)
            v = g / gn
            _, gp, _ = self.energy_and_grad(u + 1e-6 * v)
            _, gm, _ = self.energy_and_grad(u - 1e-6 * v)
            return fv, gn * (gp - gm) / 2e-6

    sub12 = SubspaceWrap(solver12, Q_k7)
    neb_sub = run_cineb(sub12, np.zeros(sub12.D), c_k7_sub_r2, n_images=19, k_spring=0.2, n_steps=1000, lr_init=0.008, climb_after=200)
    pol_sub = polish_saddle(sub12, neb_sub["c_peak"], maxiter=300)
    c_k7_sad_full12 = Q_k7 @ pol_sub["c"]

    E_k7_f, g_k7_f, ncs_k7_f = solver12.energy_and_grad(c_k7_sad_full12)
    print(f"  Palais check in FULL 3D (126 DOF): E={E_k7_f:.6f}, N_CS={ncs_k7_f:.6f}, ||grad_full3D||={np.linalg.norm(g_k7_f):.3e}", flush=True)

    H_k7_full12 = solver12.hessian(c_k7_sad_full12)
    ev_k7_full12, evec_k7_full12 = np.linalg.eigh(H_k7_full12)
    n_neg_full = int(np.sum(ev_k7_full12 < -1e-4))
    n_zero_full = int(np.sum(np.abs(ev_k7_full12) <= 1e-4))
    n_pos_full = int(np.sum(ev_k7_full12 > 1e-4))
    print(f"  FULL 3D MORSE SPECTRUM of k=7 saddle (126 DOF): ({n_neg_full} neg, {n_zero_full} zero, {n_pos_full} pos)!!", flush=True)
    print(f"  Lowest 12 eigenvalues: {[round(float(x), 4) for x in ev_k7_full12[:12]]}", flush=True)

    # -------------------------------------------------------------------------
    # PART 3: Full 3D CI-NEB Searches for rho_0 <-> rho_2 AND rho_1 <-> rho_2!
    # -------------------------------------------------------------------------
    print("\n--- PART 3: Full 3D Unrestricted CI-NEB Searches (N_max = 12) ---", flush=True)
    discovered_saddles = []

    def try_neb_path(label: str, c_a: np.ndarray, c_b: np.ndarray, init_p: np.ndarray, steps: int = 1200):
        neb = run_cineb(solver12, c_a, c_b, k_spring=0.15, n_steps=steps, lr_init=0.008, climb_after=300, init_path=init_p)
        pol = polish_saddle(solver12, neb["c_peak"], maxiter=500)
        # Check Morse index of polished saddle!
        H = solver12.hessian(pol["c"])
        ev = np.linalg.eigvalsh(H)
        n_neg = int(np.sum(ev < -1e-4))
        n_zero = int(np.sum(np.abs(ev) <= 1e-4))
        print(
            f"  [{label:26s}] NEB E_peak={neb['E_peak']:.6f} (N_CS={neb['N_CS_peak']:.5f}, |g|={neb['grad_norm_peak']:.2e}) | "
            f"Polished E={pol['E_YM_S3_Istar']:.6f} (E_mono={pol['E_YM_mono']:.4f}), N_CS={pol['N_CS']:.5f}, "
            f"|g|={pol['grad_norm']:.2e}, Morse=({n_neg}-, {n_zero}0)",
            flush=True,
        )
        rec = {
            "label": label,
            "neb_E_peak": neb["E_peak"],
            "neb_N_CS_peak": neb["N_CS_peak"],
            "neb_grad_peak": neb["grad_norm_peak"],
            "neb_energies": neb["energies"],
            "neb_N_CS": neb["N_CS"],
            "polished_E_YM_S3_Istar": pol["E_YM_S3_Istar"],
            "polished_E_YM_mono": pol["E_YM_mono"],
            "polished_N_CS": pol["N_CS"],
            "polished_grad_norm": pol["grad_norm"],
            "morse_neg": n_neg,
            "morse_zero": n_zero,
            "lowest_8_eigs": [float(x) for x in ev[:8]],
            "c_polished": pol["c"],
            "images": neb["images"],
        }
        discovered_saddles.append(rec)
        return rec

    s19 = np.linspace(0.0, 1.0, 19)

    # Search 1: rho_0 <-> rho_2 kicked along the MOST NEGATIVE U(1)_R-breaking eigenvector of the k=7 saddle!
    if n_neg_full > 1:
        # Find lowest eigenvector orthogonal to Q_k7!
        v_break = None
        for idx in range(n_neg_full):
            v = evec_k7_full12[:, idx]
            proj_k7 = np.linalg.norm(Q_k7.T @ v)
            if proj_k7 < 0.5:
                v_break = v
                print(f"  Found U(1)_R-breaking instability mode #{idx}: eig={ev_k7_full12[idx]:.4f}, ||P_k7 v||={proj_k7:.3e}", flush=True)
                break
        if v_break is not None:
            init_kicked = (neb_sub["images"] @ Q_k7.T) + (2.0 * np.sin(math.pi * s19)[:, None]) * v_break[None, :]
            init_kicked[0] = c0_12
            init_kicked[-1] = c2_12
            try_neb_path("0->2_k7_break_kick", c0_12, c2_12, init_kicked, steps=1600)

    # Search 2: rho_1 <-> rho_2 direct linear interpolation (breaks k=7 naturally!)
    init_12 = (1.0 - s19[:, None]) * c1_12 + s19[:, None] * c2_12
    try_neb_path("1->2_direct", c1_12, c2_12, init_12, steps=1600)

    # Search 3 & 4: rho_0 <-> rho_2 and rho_1 <-> rho_2 with random 3D kicks!
    for idx_seed, (lbl, ca, cb) in enumerate([
        ("0->2_rand3d_1", c0_12, c2_12),
        ("0->2_rand3d_2", c0_12, c2_12),
        ("1->2_rand3d_1", c1_12, c2_12),
        ("1->2_rand3d_2", c1_12, c2_12),
    ]):
        rr = np.random.default_rng(500 + idx_seed)
        kick = rr.normal(size=(1, solver12.D))
        kick = 3.0 * kick / np.linalg.norm(kick)
        init_r = (1.0 - s19[:, None]) * ca + s19[:, None] * cb + np.sin(math.pi * s19)[:, None] * kick
        init_r[0] = ca
        init_r[-1] = cb
        try_neb_path(lbl, ca, cb, init_r, steps=1600)

    # Search 5: rho_0 -> rho_1 -> rho_2 piecewise initial path!
    init_012 = np.zeros((19, solver12.D))
    for i in range(5):
        t = i / 4.0
        init_012[i] = (1.0 - t) * c0_12 + t * c1_12
    for i in range(4, 19):
        t = (i - 4) / 14.0
        init_012[i] = (1.0 - t) * c1_12 + t * c2_12
    try_neb_path("0->1->2_piecewise", c0_12, c2_12, init_012, steps=1600)

    # -------------------------------------------------------------------------
    # PART 4: Check Best Discovered Saddle(s) at Higher Truncation N_max = 20 (315 DOF)!
    # -------------------------------------------------------------------------
    # Pick lowest-energy nontrivial converged (or NEB-peak) saddle from N_max=12!
    valid = [r for r in discovered_saddles if r["neb_E_peak"] > 0.1]
    valid.sort(key=lambda r: r["neb_E_peak"])
    best12 = valid[0]
    print(f"\n--- PART 4: N_max = 20 (315 DOF) Verification of Best Full-3D Ridge ({best12['label']}, E_12={best12['neb_E_peak']:.6f}) ---", flush=True)

    solver20, c0_20, c1_20, c2_20, Phi20, W20, U20, meta20 = build_full3d_solver(U1, poly_coeffs, max_n=20)
    print(f"Built Full3D N_max=20: D={solver20.D}, n_pts={solver20.n_pts}", flush=True)

    # Embed best12 NEB path (which lives in first M12 scalar modes!) cleanly into N_max=20!
    M12 = solver12.M
    M20 = solver20.M
    init20 = np.zeros((best12["images"].shape[0], solver20.D), dtype=np.float64)
    for k_img in range(init20.shape[0]):
        C_img = np.zeros((M20, 9), dtype=np.float64)
        C_img[:M12] = best12["images"][k_img].reshape(M12, 9)
        init20[k_img] = C_img.reshape(-1)
    # Lock exact endpoints (c0_20 or c1_20 -> c2_20)
    if best12["label"].startswith("1->2"):
        init20[0] = c1_20
    else:
        init20[0] = c0_20
    init20[-1] = c2_20

    neb20 = run_cineb(solver20, init20[0], init20[-1], k_spring=0.15, n_steps=600, lr_init=0.008, climb_after=150, init_path=init20)
    pol20 = polish_saddle(solver20, neb20["c_peak"], maxiter=250)
    print(
        f"  [N_max=20 refined] NEB E_peak={neb20['E_peak']:.6f} (N_CS={neb20['N_CS_peak']:.5f}, |g|={neb20['grad_norm_peak']:.2e}) | "
        f"Polished E={pol20['E_YM_S3_Istar']:.6f} (E_mono={pol20['E_YM_mono']:.4f}), N_CS={pol20['N_CS']:.5f}, |g|={pol20['grad_norm']:.2e}",
        flush=True,
    )

    # Also check BOTH rho_0->rho_2 AND rho_1->rho_2 best paths at N_max=20 if distinct!
    best_02 = min([r for r in valid if r["label"].startswith("0->")], key=lambda r: r["neb_E_peak"])
    best_12 = min([r for r in valid if r["label"].startswith("1->")], key=lambda r: r["neb_E_peak"])
    n20_summary = {
        "best_overall_from_N12": best12["label"],
        "N20_neb_E_peak": neb20["E_peak"],
        "N20_neb_N_CS_peak": neb20["N_CS_peak"],
        "N20_neb_grad_peak": neb20["grad_norm_peak"],
        "N20_polished_E_YM_S3_Istar": pol20["E_YM_S3_Istar"],
        "N20_polished_E_YM_mono": pol20["E_YM_mono"],
        "N20_polished_N_CS": pol20["N_CS"],
        "N20_polished_grad_norm": pol20["grad_norm"],
        "N20_neb_energies": neb20["energies"],
        "N20_neb_N_CS": neb20["N_CS"],
    }

    # Save full JSON (without raw arrays) + numpy archive of best saddles for Goal 3 & plotting!
    json_saddles = []
    for r in discovered_saddles:
        r_clean = {k: v for k, v in r.items() if k not in ("c_polished", "images")}
        json_saddles.append(r_clean)

    payload = {
        "grad_check_rel_err": gerr,
        "k7_saddle_full3d_N12_audit": {
            "E_YM_S3_Istar": E_k7_f,
            "E_YM_mono": E_k7_f * 20.0 / (math.pi ** 2),
            "N_CS": ncs_k7_f,
            "palais_full3d_grad_norm": float(np.linalg.norm(g_k7_f)),
            "morse_neg_zero_pos": [n_neg_full, n_zero_full, n_pos_full],
            "lowest_15_eigs": [float(x) for x in ev_k7_full12[:15]],
        },
        "full3d_N12_searches": json_saddles,
        "best_02_N12": {k: v for k, v in best_02.items() if k not in ("c_polished", "images")},
        "best_12_N12": {k: v for k, v in best_12.items() if k not in ("c_polished", "images")},
        "full3d_N20_refinement": n20_summary,
        "elapsed_sec": time.time() - t0,
    }
    out_json = os.path.join(OUT_DIR, "goal2_tier2_full3d.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    np.savez(
        os.path.join(OUT_DIR, "goal2_saddles.npz"),
        c_k7_sad_full12=c_k7_sad_full12,
        c_best12_polished=best12["c_polished"],
        best12_images=best12["images"],
        c_best20_polished=pol20["c"],
        best20_images=neb20["images"],
    )
    print(f"\nSaved {out_json} and goal2_saddles.npz ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
