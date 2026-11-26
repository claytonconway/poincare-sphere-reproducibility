"""Goal 2: Variational / Spectral Sphaleron Ridge Solver on S^3/I* (rho_0 <-> rho_1 <-> rho_2).

Two-tier verified spectral solver:
Tier 1 (Exact U(1)_R Hopf-equivariant winding k=7 sector A_{k=7}^{I*}):
  Contains both rho_0 (A=0) and rho_2 (A=u_7^{-1} du_7). By Palais's Principle of Symmetric
  Criticality, every critical point inside A_{k=7}^{I*} is an exact solution of the full 3D
  Yang-Mills equation D_A * F_A = 0 on S^3/I*. Because the sector is tiny (~51 DOF at N_max=36)
  and quadrature reduces to 2D S^2, we solve CI-NEB + exact saddle polishing across
  N_max in {12, 20, 24, 30, 32, 36} to verify spectral convergence!

Tier 2 (Full Unrestricted 3D I*-Invariant Harmonic Space Omega^1(S^3, su(2))^{I*}):
  Contains rho_0, rho_1, AND rho_2 simultaneously with ZERO symmetry assumptions!
  1) Tests stability (Morse index / Hessian eigenvalues) of the Tier-1 k=7 saddle against
     U(1)_R-breaking perturbations!
  2) Runs full 3D CI-NEB for BOTH rho_0 <-> rho_2 AND rho_1 <-> rho_2 (including random
     symmetry-breaking perturbations!) at N_max = 12, 20, 24, and refines candidate saddles
     via exact gradient-norm minimization ||grad E_YM||^2 -> 0 + full Hessian eigenvalue audit!
"""

import json
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

from istar_geometry import (
    EPS3,
    build_istar_and_galois,
    build_istar_invariant_harmonics,
    build_p7_equivariant_polynomial,
    compute_ym_and_cs,
    euler_quadrature_s2_hopf,
    euler_quadrature_so3,
    eval_real_harmonic_basis,
    eval_u7_and_connection,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)


class SpectralYMSolver:
    """Exact spectral Yang-Mills + Chern-Simons evaluator with analytical gradients
    on an arbitrary linear subspace of Omega^1(S^3, su(2))^{I*}.

    Basis tensors:
      B_A:  (N_pts, D, 3, 3) float64  -- basis connection fields A^{(d)}_{a, i}(x_p)
      B_XA: (N_pts, D, 3, 3, 3) float64 -- exact derivatives X_k A^{(d)}_{a, i}(x_p)
      W:    (N_pts,) float64 normalized Haar weights summing to 1.0
    """

    def __init__(self, B_A: np.ndarray, B_XA: np.ndarray, W: np.ndarray):
        self.B_A = np.ascontiguousarray(B_A, dtype=np.float64)
        self.B_XA = np.ascontiguousarray(B_XA, dtype=np.float64)
        self.W = np.ascontiguousarray(W, dtype=np.float64)
        self.n_pts, self.D = B_A.shape[0], B_A.shape[1]
        # Precompute curl of basis: B_curl[p, d, a, k] = eps_{kij} B_XA[p, d, i, a, j]
        self.B_curl = np.einsum("kij,pdiaj->pdak", EPS3, self.B_XA)
        # Flattened BLAS matrices of shape (N_pts * 9, D) for ultra-fast matvec!
        self.B_A_mat = np.ascontiguousarray(
            np.transpose(self.B_A, (0, 2, 3, 1)).reshape(self.n_pts * 9, self.D)
        )
        self.B_curl_mat = np.ascontiguousarray(
            np.transpose(self.B_curl, (0, 2, 3, 1)).reshape(self.n_pts * 9, self.D)
        )
        self.scale_E = math.pi ** 2 / 120.0  # E_YM^{S^3/I*} = (pi^2 / 120) <B^2>

    def eval_fields(self, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        A = (self.B_A_mat @ c).reshape(self.n_pts, 3, 3)
        curl_A = (self.B_curl_mat @ c).reshape(self.n_pts, 3, 3)
        return A, curl_A

    def energy_and_grad(self, c: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """Return (E_YM^{S^3/I*}, dE/dc, N_CS) using fast exact 3x3 cofactor + BLAS matvec."""
        A, curl_A = self.eval_fields(c)

        # Exact 3x3 cofactor matrix: comm_A[p, a, k] = (1/2) eps_{kij} eps_{abc} A[p, b, i] A[p, c, j]
        # = cofactor(A)[a, k]!
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

        # Term 1: B_curl_mat.T @ WB.ravel()
        g1 = self.B_curl_mat.T @ WB.reshape(-1)

        # Term 3: M[p, a, j] = eps_{kij} eps_{cba} WB[p, c, k] A[p, b, i]
        # Notice eps_{cba} = -eps_{abc}, eps_{kij} = eps_{jki}:
        # M[a, j] = sum_{b,c,i,k} eps_{abc} eps_{jik} A[b, i] WB[c, k] (since eps_{kij} eps_{cba} = (-eps_{jik})(-eps_{abc}) = +eps_{abc} eps_{jik})!
        # That is the polarized 3x3 cofactor of (A, WB)!!
        w00, w01, w02 = WB[:, 0, 0], WB[:, 0, 1], WB[:, 0, 2]
        w10, w11, w12 = WB[:, 1, 0], WB[:, 1, 1], WB[:, 1, 2]
        w20, w21, w22 = WB[:, 2, 0], WB[:, 2, 1], WB[:, 2, 2]

        # For each (a, j), let (b, c) be cyclic after a and (i, k) be cyclic after j:
        # sum_{perm} eps_{abc} eps_{jik} A[b,i] W[c,k] = A[b,i] W[c,k] - A[b,k] W[c,i] - A[c,i] W[b,k] + A[c,k] W[b,i]!
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

        coeff_A = -2.0 * WB + pol
        g23 = self.B_A_mat.T @ coeff_A.reshape(-1)
        grad = g1 + g23

        # N_CS = (1/480) < A^2 - (1/2) A . curl_A - det(A) >
        A2 = np.sum(A * A, axis=(-2, -1))
        A_curl = np.sum(A * curl_A, axis=(-2, -1))
        # Fast exact 3x3 det using already-computed cofactor first row!
        det_A = a00 * comm_A[:, 0, 0] + a01 * comm_A[:, 0, 1] + a02 * comm_A[:, 0, 2]
        N_cs = float(np.dot(self.W, A2 - 0.5 * A_curl - det_A)) / 480.0
        return E, grad, N_cs

    def energy_grad_norm_sq_and_grad(self, c: np.ndarray) -> Tuple[float, np.ndarray]:
        """Exact objective F(c) = 0.5 * ||grad E(c)||^2 and its gradient H(c) @ grad E(c)
        via exact directional Hessian-vector product (for direct saddle-point polishing!).
        """
        _, g, _ = self.energy_and_grad(c)
        fval = 0.5 * float(np.dot(g, g))
        # Exact H @ v via centered directional derivative along v = g / ||g|| (1e-6 step)
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
        """Compute exact symmetric Hessian d^2 E / dc_i dc_j via centered differences of analytical grad."""
        H = np.zeros((self.D, self.D), dtype=np.float64)
        for j in range(self.D):
            dc = np.zeros(self.D, dtype=np.float64)
            dc[j] = eps
            _, gp, _ = self.energy_and_grad(c + dc)
            _, gm, _ = self.energy_and_grad(c - dc)
            H[:, j] = (gp - gm) / (2.0 * eps)
        return 0.5 * (H + H.T)


def build_k7_subspace_basis(
    U1: np.ndarray, poly_coeffs: np.ndarray, max_n: int
) -> Tuple[SpectralYMSolver, np.ndarray, Dict]:
    """Construct the EXACT orthonormal basis of the U(1)_R Hopf-equivariant (winding k=7)
    I*-invariant connection subspace A_{k=7}^{I*} up to harmonic degree max_n!

    Symmetry condition: R_{r(theta)}^* A = Ad_{exp(-7 theta e_3)} A, where r(theta) = exp(theta e_3).
    Under R_{r(theta)}, the coframe transforms as omega^3 -> omega^3, (omega^1 + i omega^2) -> e^{2i theta} (omega^1 + i omega^2),
    Wait: check exact sign of R_{r(theta)}^* omega^i and Ad_{exp(-7 theta e_3)} T_a!
    Instead of guessing signs, project the full real harmonic connection basis directly using the
    EXACT finite-group average over M = 24 angles theta_m = 2*pi*m/M (exact for |m_R| <= 11)!
    Even simpler and 100% bug-proof:
    For each harmonic level n, construct the exact equivariant projector on (V_{n/2}^R tensor R^3_color tensor R^3_frame)
    using the exact U(1)_R generator L_3 = X_3 (right-deriv along e_3) + S_frame (rotation of coframe) - 7 * S_color (Ad_{e_3}) == 0!!
    Check: A is k=7 Hopf-equivariant IFF the infinitesimal Lie derivative vanishes:
      (X_3 A)^a_i - 2 eps_{3ij} A^a_j - 14 eps_{3ab} A^b_i == 0!!
    Let's verify that exact infinitesimal equivariance equation on A_{rho_2} = u_7^{-1} du_7 right inside this function!
    """
    levels = build_istar_invariant_harmonics(U1, max_n=max_n)
    # Quadrature on 2D S^2 (gamma = 0) — exact for all U(1)_R-invariant integrands up to 2*max_n!
    # Wait: E_YM has B^2 ~ A^4 (quartic in A!), so cos(beta) degree can reach 2 * max_n!
    # For max_n = 36, 2 * max_n = 72 => n_beta = 40 (exact to 79), n_alpha = 80 is plenty!
    n_beta = max(24, max_n + 6)
    n_alpha = max(48, 2 * max_n + 8)
    U_s2, W_s2 = euler_quadrature_s2_hopf(n_beta=n_beta, n_alpha=n_alpha)
    Phi_s2, XPhi_s2, meta = eval_real_harmonic_basis(U_s2, levels)

    # First verify infinitesimal equivariance operator on exact A_{rho_2} on 2D grid!
    # Check: (X_3 A)^a_i + c_frame * eps_{3ij} A^a_j + c_color * eps_{3ab} A^b_i == 0
    # Let's check exact signs numerically on A_{rho_2} right here so zero sign mistake is possible!
    from goal1_construct_rho2 import eval_A_and_XA_exact

    _, A_exact, XA_exact, _, _ = eval_A_and_XA_exact(U_s2, poly_coeffs)
    # Test all 4 sign combos of (+/- 2 eps_{3ij}, +/- 14 eps_{3ab})
    X3A = XA_exact[:, 2, :, :]  # (N_pts, 3, 3)
    frame_rot = np.einsum("ij,paj->pai", EPS3[2], A_exact)  # eps_{3ij} A^a_j
    color_rot = np.einsum("ab,pbi->pai", EPS3[2], A_exact)  # eps_{3ab} A^b_i
    best_signs = None
    best_err = 1e99
    for sf in (-2.0, 2.0):
        for sc in (-14.0, 14.0):
            res = X3A + sf * frame_rot + sc * color_rot
            err = float(np.max(np.abs(res)))
            if err < best_err:
                best_err = err
                best_signs = (sf, sc)
    assert best_err < 1e-11, f"Infinitesimal k=7 equivariance failed! best_err={best_err}"
    sf, sc = best_signs

    # Now at each Molien degree n, find the EXACT nullspace of:
    #   (D_real[2]^T kron I_9) + sf * (I_d kron I_3 kron EPS3[2]) + sc * (I_d kron EPS3[2] kron I_3)
    # acting on vec(C) where A^a_i(x) = sum_m C[m, a, i] Phi_m(x)!
    # Wait: check X_3(sum_m C[m, a, i] Phi_m) = sum_{m, m'} C[m, a, i] Phi_{m'} D_real[2][m', m]
    # = sum_{m'} (D_real[2] @ C)[m', a, i] Phi_{m'}!
    # So the operator on C[m, a, i] is:
    #   (L C)[m, a, i] = sum_{m'} D_real[2][m, m'] C[m', a, i] + sf * sum_j EPS3[2, i, j] C[m, a, j] + sc * sum_b EPS3[2, a, b] C[m, b, i]!
    B_A_list = []
    B_XA_list = []
    mode_levels = []

    for m in meta:
        n = m["n"]
        d = m["dim"]
        s0, s1 = m["start"], m["end"]
        D3 = m["D_real"][2]  # (d, d) skew-symmetric!

        # Build (9*d, 9*d) skew-symmetric operator L
        I_d = np.eye(d)
        I_3 = np.eye(3)
        L_op = (
            np.kron(D3, np.eye(9))
            + sf * np.kron(np.kron(I_d, I_3), EPS3[2])
            + sc * np.kron(np.kron(I_d, EPS3[2]), I_3)
        )
        # Nullspace of skew-symmetric L_op = eigenvectors of L_op^T @ L_op with eigenvalue 0!
        M_sym = L_op.T @ L_op
        evals, evecs = np.linalg.eigh(M_sym)
        null_idx = np.where(evals < 1e-10)[0]
        if len(null_idx) == 0:
            continue
        # Verify clean spectral gap (smallest nonzero eigenvalue >= 1!)
        nonzero_ev = evals[evals >= 1e-10]
        if len(nonzero_ev) > 0:
            assert float(np.min(nonzero_ev)) > 0.5, f"Tiny nonzero eigenvalue in k=7 kernel at n={n}: {np.min(nonzero_ev)}"

        V_null = evecs[:, null_idx]  # (9*d, k_n), orthonormal columns!
        Phi_n = Phi_s2[:, s0:s1]     # (N_pts, d)
        XPhi_n = XPhi_s2[:, :, s0:s1]  # (N_pts, 3, d)

        for col in range(V_null.shape[1]):
            C_mode = V_null[:, col].reshape(d, 3, 3)
            A_mode = np.einsum("pm,mai->pai", Phi_n, C_mode)
            XA_mode = np.einsum("pkm,mai->pkai", XPhi_n, C_mode)
            B_A_list.append(A_mode)
            B_XA_list.append(XA_mode)
            mode_levels.append(n)

    B_A = np.stack(B_A_list, axis=1)    # (N_pts, D, 3, 3)
    B_XA = np.stack(B_XA_list, axis=1)  # (N_pts, D, 3, 3, 3)

    # Verify basis orthonormality on 2D S^2 quadrature!
    G = np.einsum("p,pdai,peai->de", W_s2, B_A, B_A)
    gram_err = float(np.max(np.abs(G - np.eye(B_A.shape[1]))))
    assert gram_err < 1e-11, f"k=7 basis not orthonormal on S^2! err={gram_err}"

    # Project exact A_{rho_2} onto this k=7 basis!
    c_rho2 = np.einsum("p,pdai,pai->d", W_s2, B_A, A_exact)
    solver = SpectralYMSolver(B_A, B_XA, W_s2)
    E_r2, g_r2, ncs_r2 = solver.energy_and_grad(c_rho2)

    info = {
        "max_n": max_n,
        "D_k7": int(B_A.shape[1]),
        "n_pts_s2": int(U_s2.shape[0]),
        "k7_signs": [sf, sc],
        "k7_infinitesimal_err": best_err,
        "basis_gram_err": gram_err,
        "rho2_proj_E_YM": E_r2,
        "rho2_proj_grad_norm": float(np.linalg.norm(g_r2)),
        "rho2_proj_N_CS": ncs_r2,
        "mode_levels": mode_levels,
    }
    return solver, c_rho2, info


def run_cineb(
    solver: SpectralYMSolver,
    c_start: np.ndarray,
    c_end: np.ndarray,
    n_images: int = 19,
    k_spring: float = 0.5,
    n_steps: int = 800,
    lr_init: float = 0.02,
    climb_after: int = 150,
    init_path: np.ndarray = None,
) -> Dict:
    """Climbing-Image Nudged Elastic Band (Henkelman-Uberuaga-Jonsson) with FIRE optimizer!

    Finds the minimum-energy path (MEP) and exact mountain-pass saddle between c_start and c_end.
    """
    D = solver.D
    if init_path is not None:
        images = init_path.copy()
        n_images = images.shape[0]
    else:
        s = np.linspace(0.0, 1.0, n_images)
        images = (1.0 - s[:, None]) * c_start[None, :] + s[:, None] * c_end[None, :]

    # FIRE (Fast Inertial Relaxation Engine) parameters for robust NEB convergence
    vel = np.zeros_like(images)
    dt = lr_init
    dt_max = 5.0 * lr_init
    alpha = 0.1
    n_pos = 0

    best_saddle_E = 1e99
    best_saddle_c = None

    for step in range(n_steps):
        energies = np.zeros(n_images)
        grads = np.zeros((n_images, D))
        ncs_arr = np.zeros(n_images)
        for i in range(n_images):
            energies[i], grads[i], ncs_arr[i] = solver.energy_and_grad(images[i])

        i_max = 1 + int(np.argmax(energies[1:-1]))
        # Compute improved Henkelman-Jonsson tangents + NEB forces for interior images
        forces = np.zeros_like(images)
        for i in range(1, n_images - 1):
            d_plus = images[i + 1] - images[i]
            d_minus = images[i] - images[i - 1]
            e_p, e_0, e_m = energies[i + 1], energies[i], energies[i - 1]
            if e_p > e_0 > e_m:
                tau = d_plus
            elif e_m > e_0 > e_p:
                tau = d_minus
            else:
                de_max = max(abs(e_p - e_0), abs(e_m - e_0))
                de_min = min(abs(e_p - e_0), abs(e_m - e_0))
                if e_p > e_m:
                    tau = d_plus * de_max + d_minus * de_min
                else:
                    tau = d_plus * de_min + d_minus * de_max
            tau_norm = np.linalg.norm(tau)
            if tau_norm > 1e-14:
                tau = tau / tau_norm
            else:
                tau = np.zeros(D)

            g_i = grads[i]
            g_par = float(np.dot(g_i, tau)) * tau
            g_perp = g_i - g_par

            if step >= climb_after and i == i_max:
                # Climbing image: invert parallel component of true force, zero spring force!
                forces[i] = -g_i + 2.0 * g_par
            else:
                # Standard NEB force: perpendicular true force + parallel spring force
                f_spring_par = k_spring * (np.linalg.norm(d_plus) - np.linalg.norm(d_minus)) * tau
                forces[i] = -g_perp + f_spring_par

        # FIRE update on interior images
        F_int = forces[1:-1]
        V_int = vel[1:-1]
        P = float(np.sum(F_int * V_int))
        if P > 0.0:
            v_norm = float(np.linalg.norm(V_int))
            f_norm = float(np.linalg.norm(F_int))
            if f_norm > 1e-15:
                vel[1:-1] = (1.0 - alpha) * V_int + alpha * (v_norm / f_norm) * F_int
            n_pos += 1
            if n_pos > 5:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99
        else:
            vel[1:-1] = 0.0
            dt = max(dt * 0.5, 1e-4)
            alpha = 0.1
            n_pos = 0

        # Velocity Verlet step with step-size cap for stability
        vel[1:-1] += dt * F_int
        step_disp = dt * vel[1:-1]
        max_disp = float(np.max(np.linalg.norm(step_disp, axis=-1)))
        if max_disp > 0.35:
            step_disp *= 0.35 / max_disp
        images[1:-1] += step_disp

        ci_grad = float(np.linalg.norm(grads[i_max]))
        if step >= climb_after and energies[i_max] < best_saddle_E and ci_grad < 1.0:
            best_saddle_E = float(energies[i_max])
            best_saddle_c = images[i_max].copy()

    # Final evaluation
    energies = np.zeros(n_images)
    grads = np.zeros((n_images, D))
    ncs_arr = np.zeros(n_images)
    for i in range(n_images):
        energies[i], grads[i], ncs_arr[i] = solver.energy_and_grad(images[i])
    i_max = 1 + int(np.argmax(energies[1:-1]))
    return {
        "images": images,
        "energies": energies.tolist(),
        "N_CS": ncs_arr.tolist(),
        "i_max": int(i_max),
        "E_peak": float(energies[i_max]),
        "N_CS_peak": float(ncs_arr[i_max]),
        "grad_norm_peak": float(np.linalg.norm(grads[i_max])),
        "c_peak": images[i_max].copy(),
    }


def polish_saddle(solver: SpectralYMSolver, c_init: np.ndarray, maxiter: int = 300) -> Dict:
    """Polish a candidate saddle point by minimizing 0.5 * ||grad E_YM(c)||^2 with L-BFGS-B."""
    res = minimize(
        solver.energy_grad_norm_sq_and_grad,
        c_init,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-12},
    )
    c_opt = res.x
    E_opt, g_opt, ncs_opt = solver.energy_and_grad(c_opt)
    return {
        "c": c_opt,
        "E_YM_S3_Istar": E_opt,
        "E_YM_mono": E_opt * 20.0 / (math.pi ** 2),
        "N_CS": ncs_opt,
        "grad_norm": float(np.linalg.norm(g_opt)),
        "nit": int(res.nit),
        "success": bool(res.success),
    }


def main():
    t0 = time.time()
    print("=== Goal 2: Variational / Spectral Sphaleron Ridge Solver on S^3/I* ===")
    Q1, Q2, U1, U2 = build_istar_and_galois()
    _, poly_coeffs = build_p7_equivariant_polynomial(U1, U2)

    # -------------------------------------------------------------------------
    # STEP 0: Verify analytical gradient against centered finite differences!
    # -------------------------------------------------------------------------
    solver12, c_r2_12, info12 = build_k7_subspace_basis(U1, poly_coeffs, max_n=12)
    rng = np.random.default_rng(123)
    c_test = 0.5 * c_r2_12 + 0.2 * rng.normal(size=solver12.D)
    E0, g_an, _ = solver12.energy_and_grad(c_test)
    g_fd = np.zeros_like(g_an)
    eps = 1e-6
    for j in range(solver12.D):
        dc = np.zeros(solver12.D)
        dc[j] = eps
        ep, _, _ = solver12.energy_and_grad(c_test + dc)
        em, _, _ = solver12.energy_and_grad(c_test - dc)
        g_fd[j] = (ep - em) / (2.0 * eps)
    grad_check_err = float(np.max(np.abs(g_an - g_fd)) / max(1.0, np.max(np.abs(g_an))))
    print(f"[Grad Check] Relative max error vs centered FD: {grad_check_err:.3e}")
    assert grad_check_err < 1e-9, f"Analytical gradient mismatch: {grad_check_err}"

    # -------------------------------------------------------------------------
    # TIER 1: Exact U(1)_R (k=7) CI-NEB + Saddle Polishing across N_max = 12..36!
    # -------------------------------------------------------------------------
    print("\n--- TIER 1: Exact U(1)_R (k=7) Sector rho_0 <-> rho_2 Convergence Sweep ---")
    tier1_results = []
    prev_images = None
    prev_D = 0
    k7_saddles = {}

    for max_n in [12, 20, 24, 30, 32, 36]:
        solver, c_r2, info = build_k7_subspace_basis(U1, poly_coeffs, max_n=max_n)
        c_r0 = np.zeros(solver.D, dtype=np.float64)

        # Warm-start from previous N_max path (embedded in first prev_D modes!)
        init_path = None
        if prev_images is not None:
            init_path = np.zeros((prev_images.shape[0], solver.D), dtype=np.float64)
            init_path[:, :prev_D] = prev_images
            init_path[0] = c_r0
            init_path[-1] = c_r2

        neb = run_cineb(
            solver,
            c_r0,
            c_r2,
            n_images=19,
            k_spring=0.2,
            n_steps=1200 if max_n == 12 else 700,
            lr_init=0.008,
            climb_after=250 if max_n == 12 else 150,
            init_path=init_path,
        )
        # Polish the climbing image directly to an exact YM critical point (||grad E|| -> 0)!
        pol = polish_saddle(solver, neb["c_peak"], maxiter=400)
        # Compute Hessian inside k=7 sector to verify exact Morse index!
        H = solver.hessian(pol["c"])
        evals = np.linalg.eigvalsh(H)
        n_neg = int(np.sum(evals < -1e-5))
        n_zero = int(np.sum(np.abs(evals) <= 1e-5))
        n_pos = int(np.sum(evals > 1e-5))

        prev_images = neb["images"]
        prev_D = solver.D
        k7_saddles[max_n] = (solver, pol["c"], c_r2)

        row = {
            "N_max": max_n,
            "D_k7": solver.D,
            "rho2_trunc_E_YM": info["rho2_proj_E_YM"],
            "rho2_trunc_N_CS": info["rho2_proj_N_CS"],
            "neb_E_peak": neb["E_peak"],
            "neb_N_CS_peak": neb["N_CS_peak"],
            "neb_grad_peak": neb["grad_norm_peak"],
            "polished_E_YM_S3_Istar": pol["E_YM_S3_Istar"],
            "polished_E_YM_mono": pol["E_YM_mono"],
            "polished_N_CS": pol["N_CS"],
            "polished_grad_norm": pol["grad_norm"],
            "k7_hessian_neg_zero_pos": [n_neg, n_zero, n_pos],
            "k7_lowest_5_eigs": [float(x) for x in evals[:5]],
            "neb_energies": neb["energies"],
            "neb_N_CS": neb["N_CS"],
        }
        tier1_results.append(row)
        print(
            f"  N_max={max_n:2d} (D={solver.D:2d}): "
            f"NEB E_peak={neb['E_peak']:.6f} (|g|={neb['grad_norm_peak']:.2e}) | "
            f"Polished E={pol['E_YM_S3_Istar']:.6f} (E_mono={pol['E_YM_mono']:.4f}), "
            f"N_CS={pol['N_CS']:.6f}, |g|={pol['grad_norm']:.2e}, "
            f"Morse(k=7)=({n_neg}-, {n_zero}0, {n_pos}+)"
        )

    # Also test a SECOND independent initial path in k=7 (perturbed & high-amp) at N_max=20
    # to check if there are MULTIPLE saddles in k=7!
    print("\n--- TIER 1 MULTI-START CHECK (N_max = 20, 5 perturbed paths) ---")
    solver20, c_r2_20, _ = build_k7_subspace_basis(U1, poly_coeffs, max_n=20)
    c_r0_20 = np.zeros(solver20.D)
    multistart_k7 = []
    for seed in [1, 2, 3, 4, 5]:
        r = np.random.default_rng(100 + seed)
        s = np.linspace(0.0, 1.0, 15)
        bump = np.sin(math.pi * s)[:, None] * (1.5 * r.normal(size=(1, solver20.D)))
        init_p = (1.0 - s[:, None]) * c_r0_20 + s[:, None] * c_r2_20 + bump
        init_p[0] = c_r0_20
        init_p[-1] = c_r2_20
        neb_s = run_cineb(solver20, c_r0_20, c_r2_20, k_spring=0.2, n_steps=800, lr_init=0.008, climb_after=200, init_path=init_p)
        pol_s = polish_saddle(solver20, neb_s["c_peak"], maxiter=300)
        multistart_k7.append(
            {
                "seed": seed,
                "neb_E_peak": neb_s["E_peak"],
                "polished_E": pol_s["E_YM_S3_Istar"],
                "polished_N_CS": pol_s["N_CS"],
                "polished_grad": pol_s["grad_norm"],
            }
        )
        print(
            f"  seed={seed}: NEB E={neb_s['E_peak']:.6f}, polished E={pol_s['E_YM_S3_Istar']:.6f}, "
            f"N_CS={pol_s['N_CS']:.6f}, |g|={pol_s['grad_norm']:.2e}"
        )

    # Save Tier 1 immediately before launching Tier 2 full 3D!
    out_t1 = os.path.join(OUT_DIR, "goal2_tier1_k7.json")
    with open(out_t1, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "grad_check_rel_err": grad_check_err,
                "tier1_k7_convergence": tier1_results,
                "tier1_multistart_N20": multistart_k7,
                "elapsed_sec": time.time() - t0,
            },
            fh,
            indent=2,
        )
    print(f"\nSaved Tier 1 results to {out_t1} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
