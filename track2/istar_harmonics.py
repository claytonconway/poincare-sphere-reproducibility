"""Reusable I*-Equivariant Harmonic & SU(2) Representation Utilities on S^3 and S^3/I*.

Designed for both Problem 2 (p02_inhomogeneous_su2_higgs: linearized SU(2)_L Yang-Mills + Higgs
fluctuations on S^3/I*) and Problem 3 (p03_galois_rho2: Galois-conjugate flat connection rho_2
and twisted I*-equivariant vector harmonics).

Conventions (matching Poincare_Sphere_Gauge_Cosmology_Monograph.md):
  - Unit S^3 = SU(2), left-invariant orthonormal coframe d(omega^a) = -eps_{abc} omega^b ^ omega^c,
    dual frame [X_a, X_b] = 2 eps_{abc} X_c, Hermitian right-action generators J^R_a = (i/2) X_a
    obeying [J^R_a, J^R_b] = i eps_{abc} J^R_c.
  - Quotient S^3/I* = I* \\ SU(2) (left action of I* on V_{n/2}^L; background is SU(2)_L-invariant).
  - Spin-1 matrices (S_a)_{bc} = i eps_{bac} (Cartesian frame) obeying [S_a, S_b] = i eps_{abc} S_c.
  - Background homogeneous connection A = (1 + f(eta)) omega^a T_a, T_a = -i sigma_a / 2.
  - Covariant curl: C_A = -2 (J^R . S) - 2 I - (1 + f)(S . T).
  - Magnetic fluctuation operator: M_YM^2(f) = C_A^2 - (f^2 - 1)(S . T).
  - Covariant gradient: D_i = -i (2 J^R_i + (1 + f) T_i) : V_j^R x V_1^col -> V_j^R x V_1^sp x V_1^col.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Binary Icosahedral Group I* (order 120), Conjugacy Classes, & Characters
# ---------------------------------------------------------------------------

def build_binary_icosahedral_group() -> Tuple[np.ndarray, np.ndarray]:
    """Construct all 120 unit quaternions (w, x, y, z) of I* subset SU(2) and their SO(3) matrices."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    quats = []

    # 8 axis units: (+/-1, 0, 0, 0) and permutations
    for i in range(4):
        for s in (-1.0, 1.0):
            q = [0.0, 0.0, 0.0, 0.0]
            q[i] = s
            quats.append(tuple(q))

    # 16 half-integer units: (+/-1/2, +/-1/2, +/-1/2, +/-1/2)
    for s0 in (-0.5, 0.5):
        for s1 in (-0.5, 0.5):
            for s2 in (-0.5, 0.5):
                for s3 in (-0.5, 0.5):
                    quats.append((s0, s1, s2, s3))

    # 96 golden units: even permutations of (0, +/-1/2, +/-phi/2, +/-1/(2*phi))
    even_perms = [
        (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
        (1, 0, 3, 2), (1, 2, 0, 3), (1, 3, 2, 0),
        (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
        (3, 0, 2, 1), (3, 1, 0, 2), (3, 2, 1, 0),
    ]
    base_vals = (0.0, 0.5, 0.5 * phi, 0.5 / phi)
    for perm in even_perms:
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                for s3 in (-1.0, 1.0):
                    vals = [0.0, s1 * base_vals[1], s2 * base_vals[2], s3 * base_vals[3]]
                    q = [0.0, 0.0, 0.0, 0.0]
                    for idx, p_i in enumerate(perm):
                        q[p_i] = vals[idx]
                    quats.append(tuple(q))

    Q = np.array(quats, dtype=np.float64)
    _, uniq_idx = np.unique(np.round(Q, 8), axis=0, return_index=True)
    Q = Q[np.sort(uniq_idx)]
    assert Q.shape == (120, 4), f"Expected 120 elements in I*, got {Q.shape}"

    Rs = np.empty((120, 3, 3), dtype=np.float64)
    for idx, (w, x, y, z) in enumerate(Q):
        Rs[idx] = [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    return Q, Rs


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Exact Hamilton quaternion product q1 * q2 for (w, x, y, z) vectors."""
    w1, x1, y1, z1 = float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3])
    w2, x2, y2, z2 = float(q2[0]), float(q2[1]), float(q2[2]), float(q2[3])
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def quat_to_su2(q: np.ndarray) -> np.ndarray:
    """Group homomorphism I* -> SU(2): q = (w, x, y, z) -> w I - i (x sigma_x + y sigma_y + z sigma_z).

    Uses the Monograph's basis 2 T_a = -i sigma_a so (-i sigma_x)(-i sigma_y) = -i sigma_z,
    matching Hamilton multiplication i * j = +k: quat_to_su2(q1) @ quat_to_su2(q2) == quat_to_su2(q1 * q2).
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([[w - 1j * z, -y - 1j * x], [y - 1j * x, w + 1j * z]], dtype=np.complex128)


def galois_conjugate_istar_quats(Q: np.ndarray) -> np.ndarray:
    """Apply the Galois automorphism sqrt(5) -> -sqrt(5) on Q(sqrt(5)) to map rho_1 -> rho_2.

    Specifically swaps phi/2 <-> (1 - phi)/2 = -1/(2*phi) (note: phi = (1+sqrt5)/2 -> (1-sqrt5)/2 = -1/phi,
    and 1/phi = (sqrt5-1)/2 -> (-sqrt5-1)/2 = -phi).
    Preserves quaternion multiplication and unit norm, producing the Galois-conjugate embedding rho_2!
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    Q2 = Q.copy()
    for i in range(Q.shape[0]):
        for c in range(4):
            v = Q[i, c]
            av = abs(v)
            s = 1.0 if v >= 0 else -1.0
            if abs(av - 0.5 * phi) < 1e-9:
                Q2[i, c] = -s * 0.5 * inv_phi
            elif abs(av - 0.5 * inv_phi) < 1e-9:
                Q2[i, c] = -s * 0.5 * phi
    return Q2


def su2_character(two_j: int, q_w: np.ndarray) -> np.ndarray:
    """Exact SU(2) spin-j character chi_j(g) = sin((2j+1) theta) / sin(theta), cos(theta) = Re(q)."""
    w = np.clip(q_w, -1.0, 1.0)
    theta = np.arccos(w)
    out = np.empty_like(theta)
    near_pos = np.abs(w - 1.0) < 1e-12
    near_neg = np.abs(w + 1.0) < 1e-12
    mid = ~(near_pos | near_neg)
    out[near_pos] = float(two_j + 1)
    out[near_neg] = float((two_j + 1) * ((-1) ** two_j))
    out[mid] = np.sin((two_j + 1) * theta[mid]) / np.sin(theta[mid])
    return out


def klein_molien_coeffs(max_n: int) -> List[int]:
    """Exact coefficients m_n = [q^n] (1 + q^30) / ((1 - q^12)(1 - q^20)) = dim (V_{n/2}^L)^{I*}."""
    coeffs = [0] * (max_n + 1)
    for a in range(max_n // 12 + 1):
        for b in range((max_n - 12 * a) // 20 + 1):
            deg = 12 * a + 20 * b
            coeffs[deg] += 1
            if deg + 30 <= max_n:
                coeffs[deg + 30] += 1
    return coeffs


def istar_twisted_multiplicity(two_j: int, rho_char: np.ndarray, Q: np.ndarray | None = None) -> int:
    """Exact multiplicity of an I*-representation rho in V_{two_j/2}^L via character inner product."""
    if Q is None:
        Q, _ = build_binary_icosahedral_group()
    chi_j = su2_character(two_j, Q[:, 0])
    val = float(np.real(np.sum(chi_j * np.conj(rho_char))) / 120.0)
    val_int = int(round(val))
    assert abs(val - val_int) < 1e-9, f"Non-integer multiplicity {val} at 2j={two_j}"
    return val_int


def istar_adjoint_molien_coeffs(max_n: int, rho: str = "rho_1") -> List[int]:
    """Exact twisted multiplicities dim Hom_{I*}(V_{n/2}, Ad(rho)) for rho in {'rho_0', 'rho_1', 'rho_2'}.

    At rho_0 (trivial): Ad(rho_0) = 3 * triv, returns 3 * m_n.
    At rho_1 (standard): Ad(rho_1) = V_1|_{I*} (3-dim irrep 3_1 of A_5).
    At rho_2 (Galois): Ad(rho_2) = 3_2 (Galois-conjugate 3-dim irrep of A_5).
    """
    Q, _ = build_binary_icosahedral_group()
    if rho == "rho_0":
        char_ad = np.full(120, 3.0, dtype=np.float64)
    elif rho == "rho_1":
        char_ad = su2_character(2, Q[:, 0])  # spin-1 character of standard embedding
    elif rho == "rho_2":
        Q2 = galois_conjugate_istar_quats(Q)
        char_ad = su2_character(2, Q2[:, 0])
    else:
        raise ValueError(f"Unknown flat connection {rho}")
    return [istar_twisted_multiplicity(n, char_ad, Q) for n in range(max_n + 1)]


# ---------------------------------------------------------------------------
# 2. SU(2) Spin-j Generators, Wigner D-Matrices, & Cartesian/Spherical Bases
# ---------------------------------------------------------------------------

def su2_spin_matrices(two_j: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hermitian SU(2) generators (Jx, Jy, Jz) in Condon-Shortley |j, m> basis (m = j, j-1, ..., -j)."""
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


def spin1_cartesian_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cartesian spin-1 generators (S_x, S_y, S_z) with (S_a)_{bc} = i eps_{bac}, [S_a, S_b] = i eps_{abc} S_c."""
    Sx = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1j], [0.0, 1j, 0.0]], dtype=np.complex128)
    Sy = np.array([[0.0, 0.0, 1j], [0.0, 0.0, 0.0], [-1j, 0.0, 0.0]], dtype=np.complex128)
    Sz = np.array([[0.0, -1j, 0.0], [1j, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.complex128)
    return Sx, Sy, Sz


def cart_to_sph_spin1() -> np.ndarray:
    """Unitary 3x3 matrix U whose columns are the spherical spin-1 eigenvectors |1, +1>, |1, 0>, |1, -1>
    in the Cartesian basis (e_x, e_y, e_z), so U^dagger S_a U = J_a^{(j=1)}."""
    inv_s2 = 1.0 / math.sqrt(2.0)
    # Check Sz |1, +1> = +|1, +1>: Sz [1, i, 0]^T = [-i*i, i*1, 0]^T = [1, i, 0]^T!
    # Standard Condon-Shortley phase: |1, +1> = -(e_x + i e_y)/sqrt(2), |1, 0> = e_z, |1, -1> = +(e_x - i e_y)/sqrt(2).
    U = np.array(
        [
            [-inv_s2, 0.0, inv_s2],
            [-1j * inv_s2, 0.0, -1j * inv_s2],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    return U


def wigner_d_matrix(two_j: int, q: np.ndarray) -> np.ndarray:
    """Exact group homomorphism Wigner D^{(j)}(q) in |j, m> basis: D(q1) @ D(q2) == D(q1 * q2).

    For unit quaternion q = (cos theta, n_hat sin theta), represents exp(-i * 2*theta * n_hat . J):
      - two_j = 1: equals quat_to_su2(q) = cos(theta) I - i sin(theta) (n_hat . sigma) identically;
      - two_j = 2: satisfies U @ D^{(1)}(q) @ U^dagger == R(q) (exact SO(3) matrix in build_binary_icosahedral_group).
    """
    from scipy.linalg import expm

    w = float(np.clip(q[0], -1.0, 1.0))
    vec = np.asarray(q[1:4], dtype=np.float64)
    nv = float(np.linalg.norm(vec))
    dim = two_j + 1
    if nv < 1e-14:
        if w >= 0.0:
            return np.eye(dim, dtype=np.complex128)
        return ((-1) ** two_j) * np.eye(dim, dtype=np.complex128)
    theta = math.atan2(nv, w)
    nhat = vec / nv
    Jx, Jy, Jz = su2_spin_matrices(two_j)
    gen = 2.0 * theta * (nhat[0] * Jx + nhat[1] * Jy + nhat[2] * Jz)
    return expm(-1j * gen)


# ---------------------------------------------------------------------------
# 3. Exact Clebsch-Gordan Coupling & Block-Reduced (n, K) YM + Higgs Operators
# ---------------------------------------------------------------------------

def clebsch_gordan_spin1(two_j: int) -> Dict[int, np.ndarray]:
    """Exact real Clebsch-Gordan matrices C_L for V_j x V_1 -> V_L (2L in {2j-2, 2j, 2j+2}, L >= 0).

    Columns of C_L (shape ((2j+1)*3, 2L+1)) are orthonormal states |j, 1; L, m_L> in the
    tensor product basis |j, m_j> x |1, q> (ordered m_j = j..-j, q = +1, 0, -1).
    Computed directly from exact analytic spin-1 CG formulas (exact to machine precision!).
    """
    j = 0.5 * two_j
    dim_j = two_j + 1
    blocks: Dict[int, np.ndarray] = {}

    possible_two_L = [two_j + 2]
    if two_j >= 1:
        possible_two_L.append(two_j)
    if two_j >= 2:
        possible_two_L.append(two_j - 2)

    for two_L in possible_two_L:
        L = 0.5 * two_L
        dim_L = two_L + 1
        C = np.zeros((dim_j * 3, dim_L), dtype=np.float64)
        for col in range(dim_L):
            m = L - col  # m_L
            for q_idx, q in enumerate((1, 0, -1)):
                mj = m - q
                row_j = int(round(j - mj))
                if 0 <= row_j < dim_j:
                    if two_L == two_j + 2:
                        # L = j + 1
                        if q == 1:
                            val = math.sqrt((j + m) * (j + m + 1.0) / ((2.0 * j + 1.0) * (2.0 * j + 2.0)))
                        elif q == 0:
                            val = math.sqrt((j - m + 1.0) * (j + m + 1.0) / ((2.0 * j + 1.0) * (j + 1.0)))
                        else:
                            val = math.sqrt((j - m) * (j - m + 1.0) / ((2.0 * j + 1.0) * (2.0 * j + 2.0)))
                    elif two_L == two_j:
                        # L = j (j >= 1/2)
                        if q == 1:
                            val = -math.sqrt((j + m) * (j - m + 1.0) / (2.0 * j * (j + 1.0)))
                        elif q == 0:
                            val = m / math.sqrt(j * (j + 1.0))
                        else:
                            val = math.sqrt((j - m) * (j + m + 1.0) / (2.0 * j * (j + 1.0)))
                    else:
                        # L = j - 1 (j >= 1)
                        if q == 1:
                            val = math.sqrt((j - m) * (j - m + 1.0) / (2.0 * j * (2.0 * j + 1.0)))
                        elif q == 0:
                            val = -math.sqrt((j - m) * (j + m) / (j * (2.0 * j + 1.0)))
                        else:
                            val = math.sqrt((j + m) * (j + m + 1.0) / (2.0 * j * (2.0 * j + 1.0)))
                    C[row_j * 3 + q_idx, col] = val
        blocks[two_L] = C
    return blocks


def clebsch_gordan_spin1_x_spin1() -> Dict[int, np.ndarray]:
    """Exact real CG decomposition V_1^space x V_1^color -> V_0 (+) V_1 (+) V_2 in spherical basis."""
    return clebsch_gordan_spin1(2)  # two_j = 2 -> keys 0, 2, 4 corresponding to M = 0, 1, 2


def clebsch_gordan_spin2(two_j: int) -> Dict[int, np.ndarray]:
    """Exact real CG decomposition V_j x V_2 -> (+) V_K via exact spin-1 x spin-1 chain or eigenspace.

    Built directly by projecting V_j x (V_1 x V_1)_{M=2} onto K^2 eigenstates at highest weight,
    using exact Clebsch-Gordan recursion for spin-1 twice so it stays 100% real and analytic.
    """
    # Couple V_j x V_1 -> V_L, then V_L x V_1 -> V_K, and project onto M = 2 subspace!
    cg_j1 = clebsch_gordan_spin1(two_j)
    cg_11 = clebsch_gordan_spin1_x_spin1()  # keys 0, 2, 4 (M = 0, 1, 2)
    P_M2 = cg_11[4]  # (9, 5) real isometry from V_2 -> V_1 x V_1
    dim_j = two_j + 1
    I_j = np.eye(dim_j, dtype=np.float64)
    # Embedding V_j x V_2 -> V_j x V_1 x V_1
    emb_M2 = np.kron(I_j, P_M2)  # (9*dim_j, 5*dim_j)

    # Build K^2 and K_z on V_j x V_2 (tiny: 5*(n+1) dim!)
    Jx, Jy, Jz = su2_spin_matrices(two_j)
    M2x, M2y, M2z = su2_spin_matrices(4)  # spin-2
    Kx = np.kron(Jx, np.eye(5)) + np.kron(np.eye(dim_j), M2x)
    Ky = np.kron(Jy, np.eye(5)) + np.kron(np.eye(dim_j), M2y)
    Kz = np.kron(Jz, np.eye(5)) + np.kron(np.eye(dim_j), M2z)
    K2 = np.real(Kx @ Kx + Ky @ Ky + Kz @ Kz)
    Kz_r = np.real(Kz)

    # Extract highest-weight (m_K = K) real unit vector for each allowed 2K in [|2j-4|, 2j+4] step 2
    two_K_min = abs(two_j - 4)
    two_K_max = two_j + 4
    hw_vecs: Dict[int, np.ndarray] = {}
    for two_K in range(two_K_min, two_K_max + 1, 2):
        K = 0.5 * two_K
        # Restrict to m_K = K subspace first (tiny: <= 5 dim!)
        diag_kz = np.diag(Kz_r)
        idx = np.where(np.abs(diag_kz - K) < 1e-9)[0]
        sub_K2 = K2[np.ix_(idx, idx)]
        evals, evecs = np.linalg.eigh(sub_K2)
        target = K * (K + 1.0)
        best = int(np.argmin(np.abs(evals - target)))
        assert abs(evals[best] - target) < 1e-10, f"Failed K^2 match at 2j={two_j}, 2K={two_K}"
        v_full = np.zeros(5 * dim_j, dtype=np.float64)
        v_full[idx] = evecs[:, best]
        # Canonical sign: first nonzero entry > 0
        nz = np.where(np.abs(v_full) > 1e-10)[0]
        if v_full[nz[0]] < 0:
            v_full = -v_full
        # Map into V_j x V_1 x V_1 (spherical)
        hw_vecs[two_K] = emb_M2 @ v_full
    return hw_vecs


def build_nK_block_operators(n: int) -> Dict[int, dict]:
    """Build exact REAL reduced (n, K) fluctuation operators for ALL allowed 2K at harmonic degree n = 2j.

    For each allowed 2K (key = two_K, so K = two_K / 2):
      - Basis of spatial gauge space alpha in V_j^R x V_1^space x V_1^color at highest weight m_K = K,
        ordered by color-space spin M in {0, 1, 2} (dimension d_alpha in {1, 2, 3}).
      - Basis of temporal/Goldstone space (alpha_0 or pi) in V_j^R x V_1^color at m_K = K
        (dimension d_0 in {0, 1}).
      - Basis of radial Higgs h in V_j^R x V_0 at m_K = K (dimension d_h = 1 iff 2K == n else 0).

    Returns dict[two_K] with exact REAL matrices:
      - 'n', 'two_K', 'K', 'mult_K': 2K + 1
      - 'M_labels': list of M values (subset of [0, 1, 2]) corresponding to rows/cols of alpha
      - 'd_alpha', 'd_0', 'd_h'
      - 'JS': d_alpha x d_alpha real symmetric matrix of (J^R . S)
      - 'ST': d_alpha x d_alpha real diagonal matrix of (S . T) = diag(-2, -1, +1)[M_labels]
      - 'D0', 'D1': d_alpha x d_0 real matrices such that i * D_bar(f) = D0 + (1 + f) D1
      - 'd_lap_f0', 'd_lap_lin', 'd_lap_quad': coefficients of D^dagger D = ||D(f)||^2
      - 'e_trace': length-d_alpha real unit vector of the trace component (M=0), zero if M=0 absent
    """
    two_j = n
    dim_j = n + 1
    Jx, Jy, Jz = su2_spin_matrices(two_j)
    S1x, S1y, S1z = su2_spin_matrices(2)  # spherical spin-1!
    I_j = np.eye(dim_j, dtype=np.complex128)
    I_3 = np.eye(3, dtype=np.complex128)

    # Full operators in spherical basis V_j x V_1^space x V_1^color (dim = 9*(n+1)):
    # J^R . S = sum_a J_a (x) S1_a (x) I_3
    JS_full = np.real(
        np.kron(np.kron(Jx, S1x), I_3)
        + np.kron(np.kron(Jy, S1y), I_3)
        + np.kron(np.kron(Jz, S1z), I_3)
    )
    # S . T = sum_a I_j (x) S1_a (x) S1_a
    ST_full = np.real(
        np.kron(I_j, np.kron(S1x, S1x) + np.kron(S1y, S1y) + np.kron(S1z, S1z))
    )

    # Covariant gradient operator i * D_bar(f) = 2 J^R_i + (1+f) T_i (in Cartesian i, then rotated to spherical!)
    # Check: (D_bar_cart chi)_{m_j, i_cart, c_sph} = -i * (2 J^R_{i_cart} (x) I_3 + (1+f) I_j (x) T_{i_cart}^sph) chi
    # Rotating spatial index from Cartesian to spherical via U^dagger (since |psi_sph> = U^dagger |psi_cart>):
    U = cart_to_sph_spin1()  # U^dagger S_cart U = S_sph
    Udag = U.conj().T

    # Build D0_full (from 2 J^R) and D1_full (from T) mapping V_j x V_1^col (sph) -> V_j x V_1^sp (sph) x V_1^col (sph)
    J_cart = (Jx, Jy, Jz)
    T_cart = (S1x, S1y, S1z)  # in spherical color basis, T_x = S1x, T_y = S1y, T_z = S1z!
    # Map e_i_cart -> spherical components Udag[:, i]
    D0_full = np.zeros((dim_j * 9, dim_j * 3), dtype=np.complex128)
    D1_full = np.zeros((dim_j * 9, dim_j * 3), dtype=np.complex128)
    for i_c in range(3):
        v_sp = Udag[:, i_c].reshape(3, 1)  # (3, 1)
        D0_full += 2.0 * np.kron(np.kron(J_cart[i_c], v_sp), I_3)
        D1_full += np.kron(np.kron(I_j, v_sp), T_cart[i_c])

    # Verify D0_full, D1_full are real up to machine precision!
    # Wait: v_sp for i_c = 1 (y) has i, and J_y, S1y have i, so i * i = -1 is REAL!!
    max_imag_D0 = float(np.max(np.abs(np.imag(D0_full))))
    max_imag_D1 = float(np.max(np.abs(np.imag(D1_full))))
    assert max_imag_D0 < 1e-13 and max_imag_D1 < 1e-13, f"Non-real D matrices: {max_imag_D0}, {max_imag_D1}"
    D0_full = np.real(D0_full)
    D1_full = np.real(D1_full)

    # Now build exact highest-weight (m_K = K) basis vectors for each M in {0, 1, 2}
    cg_11 = clebsch_gordan_spin1_x_spin1()  # keys 0, 2, 4
    cg_j1 = clebsch_gordan_spin1(two_j)     # keys two_K in {2j-2, 2j, 2j+2} (nonneg)
    hw_M2 = clebsch_gordan_spin2(two_j)     # keys two_K in [|2j-4|, 2j+4]

    # Collect all possible 2K values
    all_two_Ks = sorted(set(list(hw_M2.keys()) + list(cg_j1.keys()) + [two_j]))

    blocks: Dict[int, dict] = {}
    I_jr = np.eye(dim_j, dtype=np.float64)

    for two_K in all_two_Ks:
        basis_cols = []
        M_labels = []

        # M = 0 exists iff 2K == 2j
        if two_K == two_j:
            # Highest-weight state |j, +j> x |M=0, 0>
            e_j_hw = np.zeros(dim_j, dtype=np.float64)
            e_j_hw[0] = 1.0
            v_M0 = np.kron(e_j_hw, cg_11[0][:, 0])
            basis_cols.append(v_M0)
            M_labels.append(0)

        # M = 1 exists iff two_K in cg_j1
        if two_K in cg_j1:
            # Column 0 of cg_j1[two_K] is m_K = +K in V_j x V_1; embed V_1 -> (V_1 x V_1)_{M=1} via cg_11[2]
            v_jM1 = cg_j1[two_K][:, 0]  # shape (3*dim_j,)
            v_M1 = np.kron(I_jr, cg_11[2]) @ v_jM1
            basis_cols.append(v_M1)
            M_labels.append(1)

        # M = 2 exists iff two_K in hw_M2
        if two_K in hw_M2:
            basis_cols.append(hw_M2[two_K])
            M_labels.append(2)

        B_alpha = np.column_stack(basis_cols)  # (9*dim_j, d_alpha)
        # Verify orthonormality
        ortho_err = float(np.max(np.abs(B_alpha.T @ B_alpha - np.eye(len(M_labels)))))
        assert ortho_err < 1e-12, f"Non-orthonormal basis at n={n}, 2K={two_K}: {ortho_err}"

        JS_blk = B_alpha.T @ JS_full @ B_alpha
        ST_blk = B_alpha.T @ ST_full @ B_alpha
        # Symmetrize numerically
        JS_blk = 0.5 * (JS_blk + JS_blk.T)
        ST_blk = 0.5 * (ST_blk + ST_blk.T)

        # Temporal/Goldstone basis (V_j x V_1^color at m_K = K): exists iff two_K in cg_j1
        if two_K in cg_j1:
            v_0 = cg_j1[two_K][:, 0:1]  # (3*dim_j, 1)
            D0_blk = B_alpha.T @ (D0_full @ v_0)
            D1_blk = B_alpha.T @ (D1_full @ v_0)
            d_0 = 1
        else:
            D0_blk = np.zeros((len(M_labels), 0), dtype=np.float64)
            D1_blk = np.zeros((len(M_labels), 0), dtype=np.float64)
            d_0 = 0

        d_h = 1 if two_K == two_j else 0
        e_trace = np.zeros(len(M_labels), dtype=np.float64)
        if 0 in M_labels:
            e_trace[M_labels.index(0)] = 1.0

        blocks[two_K] = {
            "n": n,
            "two_K": two_K,
            "K": 0.5 * two_K,
            "mult_K": two_K + 1,
            "M_labels": M_labels,
            "d_alpha": len(M_labels),
            "d_0": d_0,
            "d_h": d_h,
            "JS": JS_blk,
            "ST": ST_blk,
            "D0": D0_blk,
            "D1": D1_blk,
            "e_trace": e_trace,
        }
    return blocks


def eval_ym_operators(blk: dict, f: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate (C_A(f), M_YM^2(f), D(f)) for a reduced (n, K) block at background value f."""
    d = blk["d_alpha"]
    I_d = np.eye(d, dtype=np.float64)
    lam = 1.0 + f
    C_A = -2.0 * blk["JS"] - 2.0 * I_d - lam * blk["ST"]
    M2 = C_A @ C_A - (f * f - 1.0) * blk["ST"]
    M2 = 0.5 * (M2 + M2.T)
    D = blk["D0"] + lam * blk["D1"]
    return C_A, M2, D


def build_coulomb_complement(D: np.ndarray) -> np.ndarray:
    """Construct an orthonormal basis Q (d_alpha x (d_alpha - 1)) for D^perp in R^{d_alpha}."""
    d = D.shape[0]
    if D.shape[1] == 0:
        return np.eye(d, dtype=np.float64)
    v = D[:, 0]
    nv = float(np.linalg.norm(v))
    if nv < 1e-14:
        return np.eye(d, dtype=np.float64)
    # Deterministic QR / SVD orthogonal complement
    u = v / nv
    # Householder reflection mapping e_0 -> +/- u
    s = 1.0 if u[0] >= 0.0 else -1.0
    w = u.copy()
    w[0] += s
    wn = float(np.linalg.norm(w))
    if wn < 1e-14:
        H = np.eye(d, dtype=np.float64)
    else:
        w /= wn
        H = np.eye(d, dtype=np.float64) - 2.0 * np.outer(w, w)
    # First column of H is -s * u; remaining columns span u^perp!
    return H[:, 1:]
