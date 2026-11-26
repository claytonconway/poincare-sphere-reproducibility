"""Core geometry, I* representation theory, Wigner-D harmonics, and quadrature on S^3/I*.

Shared mathematical conventions on S^3/I*:
- Unit quaternion q = (w, x, y, z) <-> SU(2) matrix U(q) = w I - i (x sigma_1 + y sigma_2 + z sigma_3)
  = w I + 2(x T_1 + y T_2 + z T_3), where T_a = -i sigma_a / 2, [T_a, T_b] = eps_{abc} T_c,
  Tr(T_a T_b) = -(1/2) delta_{ab}, Tr(T_a T_b T_c) = -(1/4) eps_{abc}.
- Left action of I* subset SU(2)_L: L_gamma(x) = gamma * x.
- Left-invariant orthonormal frame X_a(x) = x * e_a (e_a = 2 T_a = -i sigma_a):
  [X_a, X_b] = 2 eps_{abc} X_c, dual coframe d omega^a = -eps_{abc} omega^b ^ omega^c,
  Maurer-Cartan form x^{-1} dx = 2 omega^a T_a.
- Unit round metric Vol(S^3) = 2 pi^2, Vol(S^3/I*) = pi^2 / 60.
- Connection A = A^a_i omega^i T_a, curvature F = dA + A ^ A:
  B^a_k = eps_{kij} X_i A^a_j - 2 A^a_k + (1/2) eps_{kij} eps_{abc} A^b_i A^c_j.
- Chern-Simons functional on S^3/I*:
  N_CS(A) = (1/480) < A^a_k A^a_k - (1/2) eps_{kij} A^a_k X_i A^a_j - det(A) >_{S^3}.
- 3D Yang-Mills energy on unit S^3/I* (g_YM = 1, a = 1):
  E_YM^{S^3/I*}(A) = (pi^2 / 120) < sum_{a,k} (B^a_k)^2 >_{S^3},
  and monograph-normalized E_YM^{mono}(A) = (1/6) < sum_{a,k} (B^a_k)^2 >_{S^3}
  (so A = (1+f) omega^a T_a gives E_YM^{mono} = (1/2)(f^2 - 1)^2, sphaleron S_01 = 1/2).
"""

import math
from typing import Dict, List, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss

PHI = 0.5 * (1.0 + math.sqrt(5.0))
PHI_BAR = 0.5 * (1.0 - math.sqrt(5.0))  # Galois conjugate: sqrt(5) -> -sqrt(5)

# Pauli matrices and SU(2) generators T_a = -i sigma_a / 2, e_a = 2 T_a = -i sigma_a
SIGMA = np.array(
    [
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, -1.0j], [1.0j, 0.0]],
        [[1.0, 0.0], [0.0, -1.0]],
    ],
    dtype=np.complex128,
)
T_MAT = -0.5j * SIGMA
E_MAT = -1.0j * SIGMA  # e_1=i, e_2=j, e_3=k in SU(2)

# Levi-Civita tensor
EPS3 = np.zeros((3, 3, 3), dtype=np.float64)
EPS3[0, 1, 2] = EPS3[1, 2, 0] = EPS3[2, 0, 1] = 1.0
EPS3[0, 2, 1] = EPS3[2, 1, 0] = EPS3[1, 0, 2] = -1.0


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions (..., 4) with convention (w, x, y, z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=np.float64, copy=True)
    out[..., 1:] *= -1.0
    return out


def quat_to_su2(q: np.ndarray) -> np.ndarray:
    """Map unit quaternion (..., 4) = (w, x, y, z) to SU(2) matrix U(q) = w I - i (x s1 + y s2 + z s3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    mat = np.empty(q.shape[:-1] + (2, 2), dtype=np.complex128)
    mat[..., 0, 0] = w - 1.0j * z
    mat[..., 0, 1] = -y - 1.0j * x
    mat[..., 1, 0] = y - 1.0j * x
    mat[..., 1, 1] = w + 1.0j * z
    return mat


def su2_to_quat(U: np.ndarray) -> np.ndarray:
    """Inverse of quat_to_su2: U = [[w - i z, -y - i x], [y - i x, w + i z]]."""
    w = 0.5 * (U[..., 0, 0].real + U[..., 1, 1].real)
    z = 0.5 * (-U[..., 0, 0].imag + U[..., 1, 1].imag)
    x = -0.5 * (U[..., 0, 1].imag + U[..., 1, 0].imag)
    y = 0.5 * (U[..., 1, 0].real - U[..., 0, 1].real)
    return np.stack([w, x, y, z], axis=-1)


def c2_to_su2(w: np.ndarray) -> np.ndarray:
    """Map unit vector w = (w1, w2) in C^2 to SU(2) matrix [[w1, -conj(w2)], [w2, conj(w1)]].

    Satisfies c2_to_su2(M @ w) = M @ c2_to_su2(w) for all M in SU(2)!
    """
    w1 = w[..., 0]
    w2 = w[..., 1]
    mat = np.empty(w.shape[:-1] + (2, 2), dtype=np.complex128)
    mat[..., 0, 0] = w1
    mat[..., 0, 1] = -np.conj(w2)
    mat[..., 1, 0] = w2
    mat[..., 1, 1] = np.conj(w1)
    return mat


def build_istar_and_galois() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct all 120 elements of I* in exact Q(sqrt(5)) form and its Galois conjugate rho_2.

    Every element of I* has components (a + b*sqrt(5)) / 4 with integers a, b.
    Applying sqrt(5) -> -sqrt(5) elementwise gives the exact Galois automorphism rho_2: I* -> SU(2)!
    Returns:
      Q1: (120, 4) float64 quaternions of standard embedding rho_1(I*)
      Q2: (120, 4) float64 quaternions of Galois conjugate rho_2(I*)
      U1: (120, 2, 2) complex128 SU(2) matrices of rho_1
      U2: (120, 2, 2) complex128 SU(2) matrices of rho_2
    """
    # Represent each quaternion component as integer pair (a, b) meaning (a + b*sqrt(5)) / 4.
    # Standard 120 elements of I* (2T + 96 golden even permutations):
    # 1) 8 units: (+/- 1, 0, 0, 0) and permutations -> (a, b) = (+/-4, 0)
    # 2) 16 half-units: (+/- 1/2, +/- 1/2, +/- 1/2, +/- 1/2) -> (+/-2, 0)
    # 3) 96 golden elements: even permutations of (0, +/- 1/2, +/- phi/2, +/- (1/phi)/2)
    #    where 1/2 = (2,0)/4, phi/2 = (1,1)/4, (1/phi)/2 = (phi - 1)/2 = (-1,1)/4.
    exact_elems = set()

    # 8 axis elements
    for idx in range(4):
        for s in (-4, 4):
            v = [(0, 0)] * 4
            v[idx] = (s, 0)
            exact_elems.add(tuple(v))

    # 16 half-integer elements
    for s0 in (-2, 2):
        for s1 in (-2, 2):
            for s2 in (-2, 2):
                for s3 in (-2, 2):
                    exact_elems.add(((s0, 0), (s1, 0), (s2, 0), (s3, 0)))

    # 12 even permutations of (0, 1, 2, 3)
    even_perms = [
        (0, 1, 2, 3),
        (0, 2, 3, 1),
        (0, 3, 1, 2),
        (1, 0, 3, 2),
        (1, 2, 0, 3),
        (1, 3, 2, 0),
        (2, 0, 1, 3),
        (2, 1, 3, 0),
        (2, 3, 0, 1),
        (3, 0, 2, 1),
        (3, 1, 0, 2),
        (3, 2, 1, 0),
    ]
    # Note: phi/2 = (1 + sqrt(5))/4, (phi^-1)/2 = (-1 + sqrt(5))/4
    base_vals = [(0, 0), (2, 0), (1, 1), (-1, 1)]
    for p in even_perms:
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                for s3 in (-1, 1):
                    vals = [
                        (0, 0),
                        (s1 * base_vals[1][0], s1 * base_vals[1][1]),
                        (s2 * base_vals[2][0], s2 * base_vals[2][1]),
                        (s3 * base_vals[3][0], s3 * base_vals[3][1]),
                    ]
                    v = [None] * 4
                    for k, pos in enumerate(p):
                        v[pos] = vals[k]
                    exact_elems.add(tuple(v))

    exact_list = sorted(exact_elems)
    assert len(exact_list) == 120, f"Expected 120 elements, got {len(exact_list)}"

    sqrt5 = math.sqrt(5.0)
    Q1 = np.zeros((120, 4), dtype=np.float64)
    Q2 = np.zeros((120, 4), dtype=np.float64)
    for idx, elem in enumerate(exact_list):
        for c in range(4):
            a, b = elem[c]
            Q1[idx, c] = 0.25 * (a + b * sqrt5)
            Q2[idx, c] = 0.25 * (a - b * sqrt5)

    # Verify exact group closure and exact homomorphism rho_2(g1 * g2) = rho_2(g1) * rho_2(g2)
    # Build lookup map from Q1
    # Check all 120x120 products
    prod1 = quat_mul(Q1[:, None, :], Q1[None, :, :])  # (120, 120, 4)
    prod2 = quat_mul(Q2[:, None, :], Q2[None, :, :])  # (120, 120, 4)
    # Match each prod1[i,j] to an element k of Q1 via max dot product
    dots = np.einsum("ijk,lk->ijl", prod1, Q1)
    k_idx = np.argmax(dots, axis=-1)
    closure_err1 = float(np.max(np.abs(prod1 - Q1[k_idx])))
    hom_err2 = float(np.max(np.abs(prod2 - Q2[k_idx])))
    assert closure_err1 < 1e-14, f"I* closure failed: {closure_err1}"
    assert hom_err2 < 1e-14, f"Galois homomorphism failed: {hom_err2}"

    U1 = quat_to_su2(Q1)
    U2 = quat_to_su2(Q2)
    return Q1, Q2, U1, U2


def spin_j_generators(two_j: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Hermitian angular momentum matrices (Jx, Jy, Jz) for spin j = two_j / 2.

    Satisfies [Jx, Jy] = i Jz.
    In SU(2) convention U(q) = exp(theta * n_a e_a) = exp(-i theta * n_a sigma_a),
    the spin-j representation of e_a = -i sigma_a is:
      E_a^{(j)} = -2i J_a,   T_a^{(j)} = -i J_a,
    which satisfies [T_a^{(j)}, T_b^{(j)}] = eps_{abc} T_c^{(j)} and
    [E_a^{(j)}, E_b^{(j)}] = 2 eps_{abc} E_c^{(j)}!
    """
    j = 0.5 * two_j
    dim = two_j + 1
    ms = np.arange(j, -j - 0.5, -1.0)
    Jz = np.diag(ms).astype(np.complex128)
    Jp = np.zeros((dim, dim), dtype=np.complex128)
    for idx in range(dim - 1):
        m_low = ms[idx + 1]
        Jp[idx, idx + 1] = math.sqrt(j * (j + 1.0) - m_low * (m_low + 1.0))
    Jm = Jp.conj().T
    Jx = 0.5 * (Jp + Jm)
    Jy = -0.5j * (Jp - Jm)
    return Jx, Jy, Jz


def sym_power_matrix(U: np.ndarray, n: int) -> np.ndarray:
    """Compute exact (n+1)x(n+1) Sym^n(U) = D^{n/2}(U) representation of U in SU(2)
    with machine precision (~1e-15) and zero binomial cancellation for arbitrary n.

    Uses the exact Lie-algebraic exponential in the normalized monomial basis
    Psi_n(z)_k = sqrt(C(n,k)) z1^{n-k} z2^k (k = 0..n, corresponding to m = j, j-1, ..., -j):
      U = w I + x e_1 + y e_2 + z e_3 = exp(theta * n_a e_a)
      => D^{n/2}(U) = exp(-2 i theta * (n_x J_x + n_y J_y + n_z J_z)),
    evaluated via exact Euler decomposition U = exp((alpha/2) e_3) exp((beta/2) e_2) exp((gamma/2) e_3)
    so that only ONE eigendecomposition of J_y^{(n/2)} is needed for arbitrarily large batches!
    """
    batch_shape = U.shape[:-2]
    dim = n + 1
    if n == 0:
        return np.ones(batch_shape + (1, 1), dtype=np.complex128)
    if n == 1:
        return U.astype(np.complex128, copy=True)

    # Extract Euler angles of U = [[a, -conj(b)], [b, conj(a)]] where:
    # a = cos(beta/2) exp(-i(alpha+gamma)/2),  b = sin(beta/2) exp(i(alpha-gamma)/2).
    # Note: U = diag(e^{-i alpha/2}, e^{i alpha/2}) * [[c, -s], [s, c]] * diag(e^{-i gamma/2}, e^{i gamma/2})
    # gives U[0,0] = c e^{-i(alpha+gamma)/2}, U[1,0] = s e^{i(alpha-gamma)/2}!
    # Therefore:
    #   cos(beta/2) = |a|,  sin(beta/2) = |b| => beta = 2 * atan2(|b|, |a|) in [0, pi].
    #   Let phi_a = arg(a) = -(alpha + gamma)/2,  phi_b = arg(b) = (alpha - gamma)/2 (when |a|,|b| > 0).
    #   Then alpha = phi_b - phi_a,  gamma = -(phi_a + phi_b).
    #   Edge cases:
    #     |b| < 1e-15 (beta = 0): U = diag(a, conj(a)), set alpha = -2*arg(a), beta = 0, gamma = 0.
    #     |a| < 1e-15 (beta = pi): U = [[0, -conj(b)], [b, 0]], set alpha = 2*arg(b), beta = pi, gamma = 0.
    a = U[..., 0, 0]
    b = U[..., 1, 0]
    abs_a = np.abs(a)
    abs_b = np.abs(b)
    beta = 2.0 * np.arctan2(abs_b, abs_a)

    phi_a = np.angle(a)
    phi_b = np.angle(b)
    alpha = phi_b - phi_a
    gamma = -(phi_a + phi_b)

    small_b = abs_b < 1e-14
    small_a = abs_a < 1e-14
    alpha = np.where(small_b, -2.0 * phi_a, alpha)
    gamma = np.where(small_b, 0.0, gamma)
    alpha = np.where(small_a, 2.0 * phi_b, alpha)
    gamma = np.where(small_a, 0.0, gamma)

    # Diagonalize J_y^{(n/2)} once: J_y = V diag(ev) V^H
    _, Jy, Jz = spin_j_generators(n)
    ms = np.diag(Jz).real  # (dim,): j, j-1, ..., -j
    ev_y, V_y = np.linalg.eigh(Jy)  # ev_y in [-j, ..., +j]

    # Flatten batch for fast BLAS matmuls
    n_batch = int(np.prod(batch_shape)) if batch_shape else 1
    alpha_f = alpha.reshape(n_batch)
    beta_f = beta.reshape(n_batch)
    gamma_f = gamma.reshape(n_batch)

    # d^j(beta) = V_y @ diag(exp(-i beta ev_y)) @ V_y^H
    # Shape: phase_y (n_batch, dim)
    phase_y = np.exp(-1.0j * beta_f[:, None] * ev_y[None, :])
    # Compute V_y * phase_y @ V_y^H: (n_batch, dim, dim)
    # Using einsum: d[N, p, q] = sum_k V_y[p, k] * phase_y[N, k] * conj(V_y[q, k])
    # Faster via matmul: (V_y[None] * phase_y[:, None, :]) @ V_y.conj().T
    d_beta = (V_y[None, :, :] * phase_y[:, None, :]) @ V_y.conj().T

    # Multiply left by diag(exp(-i m alpha)) and right by diag(exp(-i m gamma))
    left_ph = np.exp(-1.0j * alpha_f[:, None] * ms[None, :])   # (n_batch, dim)
    right_ph = np.exp(-1.0j * gamma_f[:, None] * ms[None, :])  # (n_batch, dim)
    D = left_ph[:, :, None] * d_beta * right_ph[:, None, :]

    return D.reshape(batch_shape + (dim, dim))


def build_p7_equivariant_polynomial(U1: np.ndarray, U2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Construct the unique (up to scalar) degree-7 homogeneous holomorphic map P_7: C^2 -> C^2
    satisfying P_7(rho_1(g) z) = rho_2(g) P_7(z) for all g in I*.

    In the normalized monomial basis Psi_7(z)_k = sqrt(C(7, k)) z_1^{7-k} z_2^k (k = 0..7):
    P_7(z) = C @ Psi_7(z), where C is a (2, 8) complex matrix intertwining D^{7/2}(rho_1(g)) and rho_2(g):
      rho_2(g) C = C D^{7/2}(rho_1(g))   <=>   rho_2(g)^dag C D^{7/2}(rho_1(g)) = C.
    Average over g in I* to project onto the exact 1D intertwiner subspace!

    Returns:
      C_mat: (2, 8) complex128 matrix in normalized Psi_7 basis, scaled so ||C_mat||_F = sqrt(2)
             (and canonically phased/oriented).
      poly_coeffs: (2, 8) complex128 raw monomial coefficients so
                   P_7(z1, z2)_a = sum_{k=0}^7 poly_coeffs[a, k] * z1^{7-k} * z2^k.
    """
    D7 = sym_power_matrix(U1, 7)  # (120, 8, 8)
    # Linear operator on vec(C) in C^{16}: T(C) = (1/120) sum_g rho_2(g)^dag C D7(g)
    # Build 16x16 matrix directly: (rho_2(g)^dag x D7(g)^T)
    proj = np.zeros((16, 16), dtype=np.complex128)
    for g in range(120):
        # vec(A C B) = (A kron B^T) vec(C) in row-major order!
        proj += np.kron(U2[g].conj().T, D7[g].T)
    proj /= 120.0

    # Hermitize tiny numerical asymmetry and eigendecompose
    proj = 0.5 * (proj + proj.conj().T)
    evals, evecs = np.linalg.eigh(proj)
    # Verify exact 1D intertwiner space (one eigenvalue == 1, all others == 0)
    ones = np.where(evals > 0.5)[0]
    assert len(ones) == 1, f"Expected unique 1D intertwiner, got eigenvalues {evals}"
    assert abs(evals[ones[0]] - 1.0) < 1e-13
    assert np.max(np.abs(np.delete(evals, ones[0]))) < 1e-13

    C_mat = evecs[:, ones[0]].reshape(2, 8)

    # Canonicalize phase: make first largest-magnitude entry real positive
    idx_max = int(np.argmax(np.abs(C_mat)))
    phase = C_mat.flat[idx_max] / abs(C_mat.flat[idx_max])
    C_mat = C_mat / phase

    # Scale so max |poly_coeffs| or ||C_mat||_F = sqrt(2)
    C_mat = C_mat * (math.sqrt(2.0) / np.linalg.norm(C_mat))

    # Convert to raw monomial coefficients: sqrt(C(7, k)) * C_mat[:, k]
    binom_sqrt = np.array([math.sqrt(math.comb(7, k)) for k in range(8)], dtype=np.float64)
    poly_coeffs = C_mat * binom_sqrt[None, :]

    # Verify exact equivariance rho_2(g) C = C D7(g) across all 120 elements
    err = 0.0
    for g in range(120):
        err = max(err, float(np.linalg.norm(U2[g] @ C_mat - C_mat @ D7[g])))
    assert err < 1e-13, f"P_7 intertwiner verification failed: {err}"

    return C_mat, poly_coeffs


def eval_p7_and_derivs(
    U_pts: np.ndarray, poly_coeffs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate P_7(z) and its exact left-invariant derivatives X_a P_7(z) at batch of SU(2) points.

    Here z = U_pts[..., :, 0] = (z1, z2)^T (first column of x in SU(2)).
    Under right-infinitesimal flow x(s) = x * exp(s e_a),
      X_a(x) = x * e_a, so the first column transforms as:
      X_a z = (x * e_a)_{:, 0}.
    Since P_7(z1, z2) is holomorphic in (z1, z2), by exact chain rule:
      X_a P_7 = (dP_7/dz1) (X_a z1) + (dP_7/dz2) (X_a z2) !!

    Returns:
      P: (..., 2) complex128
      XP: (..., 3, 2) complex128, where XP[..., a, :] = X_a P_7
    """
    z1 = U_pts[..., 0, 0]
    z2 = U_pts[..., 1, 0]
    # Powers z1^k and z2^k for k=0..7
    z1p = [np.ones_like(z1)]
    z2p = [np.ones_like(z2)]
    for _ in range(7):
        z1p.append(z1p[-1] * z1)
        z2p.append(z2p[-1] * z2)

    P = np.zeros(U_pts.shape[:-2] + (2,), dtype=np.complex128)
    dP_dz1 = np.zeros_like(P)
    dP_dz2 = np.zeros_like(P)

    for k in range(8):
        p1 = 7 - k
        p2 = k
        mon = z1p[p1] * z2p[p2]
        P[..., 0] += poly_coeffs[0, k] * mon
        P[..., 1] += poly_coeffs[1, k] * mon
        if p1 > 0:
            m1 = p1 * z1p[p1 - 1] * z2p[p2]
            dP_dz1[..., 0] += poly_coeffs[0, k] * m1
            dP_dz1[..., 1] += poly_coeffs[1, k] * m1
        if p2 > 0:
            m2 = p2 * z1p[p1] * z2p[p2 - 1]
            dP_dz2[..., 0] += poly_coeffs[0, k] * m2
            dP_dz2[..., 1] += poly_coeffs[1, k] * m2

    # X_a z = (U_pts @ e_a)[..., :, 0]
    # shape (..., 3, 2): axis -2 is a in {0,1,2}, axis -1 is component {0,1}
    # U_pts has (..., 2, 2), E_MAT has (3, 2, 2) -> (U @ E_MAT)[..., a, i, 0]
    Xz = np.einsum("...ij,ajk->...aik", U_pts, E_MAT)[..., 0]  # (..., 3, 2)
    XP = (
        dP_dz1[..., None, :] * Xz[..., :, 0:1]
        + dP_dz2[..., None, :] * Xz[..., :, 1:2]
    )
    return P, XP


def eval_u7_and_connection(
    U_pts: np.ndarray, poly_coeffs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate u_7(x) = c2_to_su2(P_7 / ||P_7||), exact flat connection A_{rho_2} = u_7^{-1} du_7,
    and pointwise Dirichlet density |du_7|^2 = sum_i |X_i u_7|^2 (with |X_i u_7|^2 = |X_i w|^2 in C^2).

    Returns:
      u7: (..., 2, 2) SU(2) matrix
      A:  (..., 3, 3) float64 array A[..., a, i] such that A = A^a_i omega^i T_a
      dirichlet_density: (...,) float64 = sum_{i=1}^3 ||X_i (P_7 / ||P_7||)||_{C^2}^2
    """
    P, XP = eval_p7_and_derivs(U_pts, poly_coeffs)
    norm2 = np.sum(np.abs(P) ** 2, axis=-1)  # (...,)
    norm = np.sqrt(norm2)
    w = P / norm[..., None]  # unit vector in C^2

    # Derivative of unit vector w = P / ||P||:
    # X_i w = (X_i P)/||P|| - w * Re(<w, X_i P>) / ||P||
    re_inner = np.sum(np.conj(w[..., None, :]) * XP, axis=-1).real  # (..., 3)
    Xw = (XP - w[..., None, :] * re_inner[..., :, None]) / norm[..., None, None]  # (..., 3, 2)

    dirichlet_density = np.sum(np.abs(Xw) ** 2, axis=(-2, -1)).real

    u7 = c2_to_su2(w)  # (..., 2, 2)
    Xu7 = c2_to_su2(Xw)  # (..., 3, 2, 2) because c2_to_su2 is R-linear!

    # A_i = u7^{-1} (X_i u7) = u7^dag @ (X_i u7) = A^a_i T_a
    # Since Tr(T_a T_b) = -(1/2) delta_{ab}, A^a_i = -2 Tr(T_a A_i) = i Tr(sigma_a A_i).
    A_mat = np.einsum("...kj,...ikl->...ijl", u7.conj(), Xu7)  # (..., 3, 2, 2), axis -3 is i
    # Extract real components A^a_i:
    # Note: Tr(T_a A_i) = sum_{j,k} T_a[j,k] A_i[k,j]
    A_comp = -2.0 * np.einsum("ajk,...ikj->...ai", T_MAT, A_mat).real
    return u7, A_comp, dirichlet_density


def euler_quadrature_s3(
    n_beta: int, n_alpha: int, n_gamma: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact Gauss-Legendre x Fourier product quadrature on S^3 = SU(2) with normalized Haar weights.

    Parameterization: U(alpha, beta, gamma) = exp((alpha/2) e_3) exp((beta/2) e_2) exp((gamma/2) e_3),
      alpha in [0, 2*pi), cos(beta) in [-1, 1], gamma in [0, 4*pi).
    (Wait: covers SU(2) exactly once! Check: e^{2 pi e_3} = I, e^{pi e_3} = -I.
     With alpha in [0, 2*pi), gamma in [0, 4*pi):
     alpha/2 in [0, pi), gamma/2 in [0, 2*pi), beta/2 in [0, pi/2].
     u11 = cos(beta/2) exp(-i(alpha+gamma)/2), u21 = sin(beta/2) exp(i(alpha-gamma)/2).
     As (alpha, gamma) ranges over [0, 2*pi) x [0, 4*pi), phases phi1 = -(alpha+gamma)/2 and
     phi2 = (alpha-gamma)/2 have Jacobian det = 1/2, covering T^2 = [0, 2*pi)^2 exactly once!)
    Sum of weights == 1.0 exactly.
    """
    t_nodes, t_weights = leggauss(n_beta)  # cos(beta) in [-1, 1], sum(t_weights) = 2
    betas = np.arccos(np.clip(t_nodes, -1.0, 1.0))
    alphas = np.linspace(0.0, 2.0 * math.pi, n_alpha, endpoint=False)
    gammas = np.linspace(0.0, 4.0 * math.pi, n_gamma, endpoint=False)

    # Meshgrid with indexing='ij'
    B, A, G = np.meshgrid(betas, alphas, gammas, indexing="ij")
    cb = np.cos(0.5 * B)
    sb = np.sin(0.5 * B)
    # exp(theta e_3) = diag(e^{-i theta}, e^{+i theta})
    # exp(theta e_2) = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
    # Check: e_2 = -i sigma_2 = [[0, -1], [1, 0]], so exp(theta e_2) = [[cos, -sin], [sin, cos]].
    e_p = np.exp(-0.5j * (A + G))
    e_m = np.exp(-0.5j * (A - G))

    U = np.empty(B.shape + (2, 2), dtype=np.complex128)
    U[..., 0, 0] = cb * e_p
    U[..., 0, 1] = -sb * e_m
    U[..., 1, 0] = sb * np.conj(e_m)
    U[..., 1, 1] = cb * np.conj(e_p)

    # Weights: (t_weights / 2) / (n_alpha * n_gamma)
    w_b = t_weights / 2.0
    W = np.broadcast_to(
        w_b[:, None, None] / (n_alpha * n_gamma), B.shape
    ).copy()

    return U.reshape(-1, 2, 2), W.reshape(-1)


def euler_quadrature_so3(
    n_beta: int, n_alpha: int, n_gamma: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact Gauss-Legendre x Fourier quadrature on SO(3) = S^3 / {+/- 1} (gamma in [0, 2*pi)).

    Because -I is in I*, EVERY I*-invariant function on S^3 has even harmonic degrees n = 2j
    (integer m_L, m_R in [-j, +j]) and is strictly 2*pi-periodic in both alpha and gamma.
    Exact for products of harmonics of total degree <= min(2*n_beta - 1, n_alpha - 1, n_gamma - 1).
    """
    t_nodes, t_weights = leggauss(n_beta)
    betas = np.arccos(np.clip(t_nodes, -1.0, 1.0))
    alphas = np.linspace(0.0, 2.0 * math.pi, n_alpha, endpoint=False)
    gammas = np.linspace(0.0, 2.0 * math.pi, n_gamma, endpoint=False)

    B, A, G = np.meshgrid(betas, alphas, gammas, indexing="ij")
    cb = np.cos(0.5 * B)
    sb = np.sin(0.5 * B)
    e_p = np.exp(-0.5j * (A + G))
    e_m = np.exp(-0.5j * (A - G))

    U = np.empty(B.shape + (2, 2), dtype=np.complex128)
    U[..., 0, 0] = cb * e_p
    U[..., 0, 1] = -sb * e_m
    U[..., 1, 0] = sb * np.conj(e_m)
    U[..., 1, 1] = cb * np.conj(e_p)

    w_b = t_weights / 2.0
    W = np.broadcast_to(
        w_b[:, None, None] / (n_alpha * n_gamma), B.shape
    ).copy()
    return U.reshape(-1, 2, 2), W.reshape(-1)


def euler_quadrature_s2_hopf(n_beta: int, n_alpha: int) -> Tuple[np.ndarray, np.ndarray]:
    """2D Gauss-Legendre x Fourier quadrature on S^2 = S^3 / U(1)_R (setting gamma = 0).

    Exact for EVERY U(1)_R-invariant integrand on S^3 (such as |du_7|^2, |F(A)|^2, CS(A)
    for any U(1)_R-equivariant connection)!
    """
    return euler_quadrature_s3(n_beta, n_alpha, 1)


def build_istar_invariant_harmonics(
    U1: np.ndarray, max_n: int
) -> List[Dict]:
    """Construct exact I*-invariant left-vectors v in V_{n/2}^L for all Molien degrees 0 <= n <= max_n.

    For each degree n with m_n > 0:
      The I*-invariant harmonic basis functions on S^3/I* are:
        Y_{n, r, q}(x) = sqrt(n+1) * sum_{p=0}^n conj(v_{r, p}) * D^{n/2}_{p, q}(x),  q = 0..n.
      They satisfy:
        1) Y(gamma * x) = Y(x) for all gamma in I* (exact I*-invariance!),
        2) <conj(Y_{n,r,q}) Y_{n',r',q'}>_{S^3} = delta_{nn'} delta_{rr'} delta_{qq'},
        3) X_a Y_{n, r, q}(x) = sum_{q'} Y_{n, r, q'}(x) * E_a^{(n/2)}[q', q],
           where E_a^{(n/2)} = D^{n/2}(e_a) is the infinitesimal generator of right action!
    """
    levels = []
    for n in range(0, max_n + 1):
        # Quick filter: odd n < 30 have m_n = 0, and even n in {2,4,6,8,10,14,16,18,22,26,28} have m_n = 0
        # Still compute exact projector for every n so there are zero hardcoded assumptions!
        D = sym_power_matrix(U1, n)  # (120, n+1, n+1)
        P = np.mean(D, axis=0)
        P = 0.5 * (P + P.conj().T)
        evals, evecs = np.linalg.eigh(P)
        idx = np.where(evals > 0.5)[0]
        if len(idx) == 0:
            continue
        # Verify eigenvalues are strictly 0 or 1
        assert np.all(np.abs(evals[idx] - 1.0) < 1e-12)
        V_inv = evecs[:, idx]  # (n+1, mult)

        # Compute exact right-derivative matrices E_a^{(n/2)} via exact spin-j generators!
        # Let's verify sign: d/ds|_{s=0} D^{n/2}(x exp(s e_a)) = D^{n/2}(x) * d/ds|_{s=0} D^{n/2}(exp(s e_a)).
        # Since U(exp(s e_a)) = I + s e_a + O(s^2), compute E_a^{(n/2)} directly from spin_j_generators
        # AND cross-check against centered difference of sym_power_matrix(exp(eps e_a))!
        Jx, Jy, Jz = spin_j_generators(n)
        # Check whether sym_power_matrix(exp(s e_a)) has derivative -2i J_a:
        # For n=1: e_1 = -i sigma_1 = -2i (sigma_1/2) = -2i J_x.
        # In sym_power_matrix, basis is z1^{n-k} z2^k (k=0 is m=+j, k=n is m=-j), matching spin_j_generators!
        # Wait: let's verify e_2 = -i sigma_2 = [[0, -1], [1, 0]], while -2i J_y = -2i * [[0, -i/2], [i/2, 0]] = [[0, -1], [1, 0]]!
        # And e_3 = -i sigma_3 = [[-i, 0], [0, i]], while -2i J_z = -2i * [[1/2, 0], [0, -1/2]] = [[-i, 0], [0, i]]!
        # EXACT MATCH!
        E_gen = np.stack([-2.0j * Jx, -2.0j * Jy, -2.0j * Jz], axis=0)  # (3, n+1, n+1)

        levels.append(
            {
                "n": n,
                "mult": int(len(idx)),
                "dim_right": n + 1,
                "V_inv": V_inv,  # (n+1, mult)
                "E_gen": E_gen,  # (3, n+1, n+1)
            }
        )
    return levels


def eval_real_harmonic_basis(
    U_pts: np.ndarray, levels: List[Dict]
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Evaluate a REAL orthonormal I*-invariant scalar harmonic basis {Phi_M(x)} and its exact
    left-invariant derivatives X_i Phi_M(x) at U_pts.

    Because the space of I*-invariant harmonics at degree n is closed under complex conjugation,
    taking real and imaginary parts of Y_{n,r,q} and orthonormalizing via exact Gram-Schmidt / SVD
    inside each degree n yields a strictly real orthonormal basis {Phi_M} with EXACT real
    derivative matrices D_i^{(n)} (block-diagonal in n!):
      X_i Phi_M(x) = sum_{M'} Phi_{M'}(x) * D_i[M', M]  (Wait: or X_i Phi(x) = Phi(x) @ D_i!).

    Returns:
      Phi: (N_pts, N_basis) float64
      XPhi: (N_pts, 3, N_basis) float64, exact X_i Phi_M(x)
      meta: list of dicts per level with n, start_idx, end_idx, D_real (3, d, d)
    """
    n_pts = U_pts.shape[0]
    Phi_blocks = []
    XPhi_blocks = []
    meta = []
    cursor = 0

    for lv in levels:
        n = lv["n"]
        V_inv = lv["V_inv"]  # (n+1, mult)
        E_gen = lv["E_gen"]  # (3, n+1, n+1)
        D_pts = sym_power_matrix(U_pts, n)  # (n_pts, n+1, n+1)
        # Complex orthonormal basis Y: (n_pts, mult, n+1)
        Y = math.sqrt(n + 1.0) * np.einsum("pm,Npq->Nmq", V_inv.conj(), D_pts)
        # Exact derivatives XY: (n_pts, 3, mult, n+1)
        XY = np.einsum("Nmp,apq->Namq", Y, E_gen)

        d_c = lv["mult"] * (n + 1)
        Y_flat = Y.reshape(n_pts, d_c)
        XY_flat = XY.reshape(n_pts, 3, d_c)

        # Build exact unitary transformation U_real: C^{d_c} -> C^{d_c} mapping Y_flat -> strictly real basis!
        # Since {Y_flat} spans a conjugation-invariant space, compute the symmetric unitary matrix
        # C_conj = <Y^T Y> (actually on the full space, Y.conj() = Y @ S where S is unitary symmetric,
        # Takagi factorization S = W W^T => Phi = Y @ W is strictly real AND strictly orthonormal!).
        # Compute S cleanly from a small algebraic/pointwise least-squares or directly on S^3!
        # Better yet: use exact algebraic conjugation on D^{n/2}!!
        # Remember for U in SU(2): conj(U) = J U J^{-1} where J = [[0, -1], [1, 0]] = e_2!
        # Therefore conj(D^{n/2}(U)) = D^{n/2}(J) D^{n/2}(U) D^{n/2}(J)^{-1}!
        # That is 100% EXACT and independent of quadrature points!
        J_su2 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.complex128)
        DJ = sym_power_matrix(J_su2, n)  # (n+1, n+1) real orthogonal!
        # Check: Y_{r,q}(x) = sqrt(n+1) (V_r^dag D(x))_q
        # conj(Y_{r,q}(x)) = sqrt(n+1) (V_r^T conj(D(x)))_q = sqrt(n+1) (V_r^T DJ D(x) DJ^T)_q
        # Project V_r^T DJ back onto span(V_inv^dag):
        # M_L = V_inv^T @ DJ @ V_inv  -> Wait: (V_inv^T DJ) = M_L @ V_inv^dag => M_L = V_inv^T @ DJ @ V_inv!
        # Let's check: (M_L @ V_inv^dag)^T = V_inv^* M_L^T, wait:
        # Since V_inv spans an I*-invariant space and conj(I*) = J I* J^{-1} (wait: is J in I*? YES! j in I*!!),
        # DJ preserves V_inv (up to conj)! Check V_inv^T @ DJ @ V_inv:
        M_L = V_inv.T @ DJ @ V_inv  # (mult, mult)
        # Verify M_L is unitary!
        assert np.linalg.norm(M_L @ M_L.conj().T - np.eye(lv["mult"])) < 1e-11, (
            f"Left conjugation matrix not unitary at n={n}: err={np.linalg.norm(M_L @ M_L.conj().T - np.eye(lv['mult']))}"
        )
        # Then conj(Y_flat) = Y_flat @ S where S = kron(M_L^T, DJ)  (Wait: check index order!)
        # (conj(Y))_{r, q} = sum_{p} V_{p, r} (DJ D DJ^T)_{p, q}
        # = sum_{r', p', q'} (V^T DJ V)_{r, r'} (V^dag)_{r', p'} D_{p', q'} (DJ^T)_{q', q}
        # = sum_{r', q'} M_L[r, r'] Y_{r', q'} DJ^T[q', q]
        # = sum_{r', q'} Y_{r', q'} (M_L^T[r', r] * DJ^T[q', q])!
        S_mat = np.kron(M_L.T, DJ.T)  # (d_c, d_c), unitary symmetric (S^T = S, S S^dag = I)!
        assert np.linalg.norm(S_mat - S_mat.T) < 1e-11, f"S_mat not symmetric at n={n}"
        assert np.linalg.norm(S_mat @ S_mat.conj().T - np.eye(d_c)) < 1e-11, f"S_mat not unitary at n={n}"

        # Compute exact Takagi factorization S_mat = W W^T (W unitary) so that
        # Phi = Y_flat @ W satisfies conj(Phi) = conj(Y_flat) @ conj(W) = Y_flat @ S @ conj(W) = Y_flat @ W = Phi!!
        # Because S is symmetric unitary, S = exp(i H) for real symmetric H (or via real/imag blocks):
        # Specifically, let S = A + i B (A, B real symmetric, commuting, A^2 + B^2 = I).
        # Jointly diagonalize A and B via an orthogonal matrix Q (wait: eigendecompose A + 0.137 * B, or
        # better: for each real eigenvector q of A, B q = +/- sqrt(1-a^2) q; even simpler:
        # compute orthogonal eigenbasis of A + c*B for a generic irrational c, e.g. sqrt(2)!):
        A_re = S_mat.real
        B_im = S_mat.imag
        # Verify [A_re, B_im] == 0
        assert np.linalg.norm(A_re @ B_im - B_im @ A_re) < 1e-11
        _, Q_orth = np.linalg.eigh(A_re + math.sqrt(2.0) * B_im)
        # In basis Q_orth, S_diag = Q_orth^T @ S_mat @ Q_orth is diagonal unitary e^{i theta_k}!
        S_diag = Q_orth.T @ S_mat @ Q_orth
        off_err = np.linalg.norm(S_diag - np.diag(np.diag(S_diag)))
        if off_err > 1e-10:
            # Handle degenerate eigenspaces of A + sqrt(2) B by refining with B_im
            _, Q_orth = np.linalg.eigh(A_re + math.pi * B_im)
            S_diag = Q_orth.T @ S_mat @ Q_orth
            off_err = np.linalg.norm(S_diag - np.diag(np.diag(S_diag)))
        assert off_err < 1e-10, f"Takagi diagonalization failed at n={n}: off_err={off_err}"
        phases = np.angle(np.diag(S_diag))
        W = Q_orth @ np.diag(np.exp(0.5j * phases))  # satisfies W @ W.T == S_mat!
        assert np.linalg.norm(W @ W.T - S_mat) < 1e-11

        Phi_n = Y_flat @ W
        XPhi_n = np.einsum("Nam,mk->Nak", XY_flat, W)
        assert np.max(np.abs(Phi_n.imag)) < 1e-11, f"Nonzero imag part in Phi at n={n}: {np.max(np.abs(Phi_n.imag))}"
        assert np.max(np.abs(XPhi_n.imag)) < 1e-11, f"Nonzero imag part in XPhi at n={n}"

        # Real derivative matrix in Phi_n basis: X_a Phi_n = Phi_n @ D_real[a]
        E_full = np.array([np.kron(np.eye(lv["mult"]), E_gen[a]) for a in range(3)])
        D_real_c = np.einsum("ij,ajk,kl->ail", W.conj().T, E_full, W)
        assert np.max(np.abs(D_real_c.imag)) < 1e-11
        D_real = D_real_c.real

        Phi_blocks.append(Phi_n.real)
        XPhi_blocks.append(XPhi_n.real)
        meta.append(
            {
                "n": n,
                "mult": lv["mult"],
                "dim": d_c,
                "start": cursor,
                "end": cursor + d_c,
                "D_real": D_real,  # (3, d_c, d_c), skew-symmetric!
            }
        )
        cursor += d_c

    return (
        np.concatenate(Phi_blocks, axis=1),
        np.concatenate(XPhi_blocks, axis=2),
        meta,
    )


def compute_ym_and_cs(
    A: np.ndarray, XA: np.ndarray, weights: np.ndarray
) -> Dict[str, float]:
    """Compute exact B^a_k, E_YM^{S^3/I*}, E_YM^{mono}, and N_CS^{S^3/I*} at quadrature points.

    Args:
      A:  (N_pts, 3, 3) float64, A[p, a, i] = A^a_i(x_p)
      XA: (N_pts, 3, 3, 3) float64, XA[p, k, a, i] = X_k A^a_i(x_p)
      weights: (N_pts,) normalized Haar weights summing to 1.0
    """
    # B^a_k = eps_{kij} X_i A^a_j - 2 A^a_k + (1/2) eps_{kij} eps_{abc} A^b_i A^c_j
    curl_A = np.einsum("kij,piaj->pak", EPS3, XA)
    comm_A = 0.5 * np.einsum("kij,abc,pbi,pcj->pak", EPS3, EPS3, A, A)
    B = curl_A - 2.0 * A + comm_A  # (N_pts, 3, 3)

    B2 = np.sum(B * B, axis=(-2, -1))  # (N_pts,)
    avg_B2 = float(np.dot(weights, B2))
    max_B = float(np.sqrt(np.max(B2)))
    rms_B = math.sqrt(avg_B2)

    E_ym_s3istar = (math.pi ** 2 / 120.0) * avg_B2
    E_ym_mono = avg_B2 / 6.0

    # N_CS = (1/480) < A^a_k A^a_k - (1/2) eps_{kij} A^a_k X_i A^a_j - det(A) >
    A2 = np.sum(A * A, axis=(-2, -1))
    A_curl_A = np.sum(A * curl_A, axis=(-2, -1))
    det_A = np.linalg.det(A)
    cs_density = A2 - 0.5 * A_curl_A - det_A
    N_cs = float(np.dot(weights, cs_density)) / 480.0

    return {
        "E_YM_S3_Istar": E_ym_s3istar,
        "E_YM_mono": E_ym_mono,
        "N_CS": N_cs,
        "rms_B": rms_B,
        "max_B": max_B,
        "avg_B2": avg_B2,
    }
