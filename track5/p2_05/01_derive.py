#!/usr/bin/env python3
"""
Stream p2_05_diamond_invariance_and_twisted_prolate - Stage 1 (01_derive.py):
Exact Analytical Derivations, 50-Digit Differential/Algebraic Proofs, and Exact Rational Benchmarks for KT1--KT4.

Outputs: streams/p2_05_diamond_invariance_and_twisted_prolate/01_derive.json
"""

import json
import math
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np

mp.mp.dps = 50

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SHARED EXACT ARITHMETIC & CHARACTER HELPERS
# =============================================================================

def bern_zeta_neg(k: int, x: Fraction) -> Fraction:
    """Exact rational Hurwitz zeta(-k, x) = -B_{k+1}(x)/(k+1) for k in 0..5."""
    if k == 0:
        return -(x - Fraction(1, 2))
    if k == 1:
        return -(x**2 - x + Fraction(1, 6)) / 2
    if k == 2:
        return -(x**3 - Fraction(3, 2) * x**2 + Fraction(1, 2) * x) / 3
    if k == 3:
        return -(x**4 - 2 * x**3 + x**2 - Fraction(1, 30)) / 4
    if k == 4:
        return -(x**5 - Fraction(5, 2) * x**4 + Fraction(5, 3) * x**3 - Fraction(1, 6) * x) / 5
    if k == 5:
        return -(x**6 - 3 * x**5 + Fraction(5, 2) * x**4 - Fraction(1, 2) * x**2 + Fraction(1, 42)) / 6
    raise ValueError(k)


def ade_classes(name: str):
    if name == "S3":
        return 1, [(1, mp.mpf(0))]
    if name == "RP3":
        return 2, [(1, mp.mpf(0)), (1, mp.pi)]
    if name == "Q8":
        return 8, [(1, mp.mpf(0)), (1, mp.pi), (6, mp.pi / 2)]
    if name == "T*":
        return 24, [(1, mp.mpf(0)), (1, mp.pi), (6, mp.pi / 2), (8, mp.pi / 3), (8, 2 * mp.pi / 3)]
    if name == "O*":
        return 48, [
            (1, mp.mpf(0)), (1, mp.pi), (18, mp.pi / 2),
            (8, mp.pi / 3), (8, 2 * mp.pi / 3), (6, mp.pi / 4), (6, 3 * mp.pi / 4),
        ]
    if name == "I*":
        return 120, [
            (1, mp.mpf(0)), (1, mp.pi), (30, mp.pi / 2),
            (20, mp.pi / 3), (20, 2 * mp.pi / 3),
            (12, mp.pi / 5), (12, 2 * mp.pi / 5), (12, 3 * mp.pi / 5), (12, 4 * mp.pi / 5),
        ]
    raise ValueError(name)


def su2_chi(two_j: int, theta: mp.mpf) -> mp.mpf:
    if two_j < 0:
        return mp.mpf(0)
    if abs(theta) < mp.mpf("1e-35"):
        return mp.mpf(two_j + 1)
    if abs(theta - mp.pi) < mp.mpf("1e-35"):
        return mp.mpf(two_j + 1) * ((-1) ** two_j)
    return mp.sin((two_j + 1) * theta) / mp.sin(theta)


def m_ade(name: str, two_j: int) -> int:
    if two_j < 0:
        return 0
    order, cls = ade_classes(name)
    val = sum(sz * su2_chi(two_j, th) for sz, th in cls) / order
    iv = int(mp.nint(val))
    assert abs(val - iv) < mp.mpf("1e-40")
    return iv


def ade_tt_mult(name: str, p: int) -> int:
    return m_ade(name, p - 3) * (p + 2) + m_ade(name, p + 1) * (p - 2)


def m_lens(q_ord: int, two_j: int, anti_per: bool = False) -> int:
    if two_j < 0:
        return 0
    rem_target = (q_ord // 2) if anti_per else 0
    if anti_per:
        assert q_ord % 2 == 0
    return sum(1 for two_m in range(-two_j, two_j + 1, 2) if (two_m % q_ord) == rem_target)


def extract_quasi_poly_deg2_frac(R_fn, P: int, p_min: int):
    C_r, A_r, B_r = {}, {}, {}
    for r in range(P):
        p0 = p_min + ((r - p_min) % P)
        xs = [p0 + m * P for m in range(4)]
        ys = [Fraction(R_fn(x)) for x in xs]
        assert (ys[3] - 3 * ys[2] + 3 * ys[1] - ys[0]) == 0
        d2 = (ys[2] - ys[1]) - (ys[1] - ys[0])
        c = d2 / (2 * P * P)
        L1 = ys[0] - c * xs[0] * xs[0]
        L2 = ys[1] - c * xs[1] * xs[1]
        a = (L2 - L1) / P
        b = L1 - a * xs[0]
        C_r[r], A_r[r], B_r[r] = c, a, b
    for pp in range(p_min, p_min + 3 * P + 5):
        r = pp % P
        assert Fraction(R_fn(pp)) == C_r[r] * pp * pp + A_r[r] * pp + B_r[r]
    return C_r, A_r, B_r


# =============================================================================
# PART 1 (KT1): 50-DIGIT ISOTROPIC ALPHA-INVARIANCE PROOF, AFFINE-REDSHIFT COUNTEREXAMPLE, & TT AUDIT
# =============================================================================
print("=" * 86)
print("PART 1 (KT1): 50-DIGIT ISOTROPIC ALPHA-INVARIANCE, AFFINE-REDSHIFT IDENTITY & TT AUDIT")
print("=" * 86)

# 1A. Exact 50-digit differential-identity proof of both integration-by-parts steps
#     establishing BR_alpha + DIR_alpha === 0 for arbitrary H_2(eta), eta_0, chi_max
def _test_H2_poly(e):
    return mp.mpf("0.7") + mp.mpf("1.3") * mp.cos(2 * e) + mp.mpf("0.5") * mp.sin(3 * e) + mp.mpf("0.2") * mp.exp(e)

def _test_H2p(e):
    return -mp.mpf("2.6") * mp.sin(2 * e) + mp.mpf("1.5") * mp.cos(3 * e) + mp.mpf("0.2") * mp.exp(e)

def _test_H2pp(e):
    return -mp.mpf("5.2") * mp.cos(2 * e) - mp.mpf("4.5") * mp.sin(3 * e) + mp.mpf("0.2") * mp.exp(e)

def _test_IH2(e):
    return mp.mpf("0.7") * e + mp.mpf("0.65") * mp.sin(2 * e) - (mp.mpf("0.5") / 3) * mp.cos(3 * e) + mp.mpf("0.2") * mp.exp(e)

ibp1_max_err = mp.mpf(0)
ibp2_max_err = mp.mpf(0)
for e0_v, c_v in [(mp.mpf("-0.31"), mp.mpf("0.47")), (mp.mpf("0.13"), mp.mpf("0.29")), (mp.mpf("-0.55"), mp.mpf("0.83"))]:
    def F1(c):
        ep = e0_v + c
        return mp.sin(c) * mp.cos(ep) * _test_H2p(ep) - mp.cos(c) * mp.cos(ep) * _test_H2_poly(ep) - mp.sin(e0_v) * _test_IH2(ep)
    dF1 = mp.diff(F1, c_v)
    ep0 = e0_v + c_v
    rhs1 = mp.sin(c_v) * mp.cos(ep0) * (_test_H2pp(ep0) - mp.tan(ep0) * _test_H2p(ep0) + 2 * _test_H2_poly(ep0))
    ibp1_max_err = max(ibp1_max_err, abs(dF1 - rhs1))

    def A_anti(c):
        ep = e0_v + c
        return (
            (mp.sin(c) / mp.cos(ep)) * _test_H2_poly(ep)
            + mp.cos(e0_v) * _test_H2_poly(e0_v) * mp.tan(ep)
            - mp.sin(e0_v) * mp.tan(ep) * (_test_IH2(ep) - _test_IH2(e0_v))
        )
    dA = mp.diff(A_anti, c_v)
    integ2 = (
        mp.sin(c_v) * mp.cos(ep0) * _test_H2p(ep0)
        - mp.cos(c_v) * mp.cos(ep0) * _test_H2_poly(ep0)
        + mp.cos(e0_v) * _test_H2_poly(e0_v)
        - mp.sin(e0_v) * (_test_IH2(ep0) - _test_IH2(e0_v))
    ) / (mp.cos(ep0) ** 2)
    exp_rem2 = -_test_H2_poly(ep0) * (mp.cos(e0_v) / (mp.cos(ep0) ** 2) + mp.cos(e0_v))
    ibp2_max_err = max(ibp2_max_err, abs((integ2 - dA) - exp_rem2))

assert ibp1_max_err < mp.mpf("1e-44") and ibp2_max_err < mp.mpf("1e-44")
print(f"  Exact 50-dps differential IBP proof (BR_alpha + DIR_alpha === 0): ibp1={float(ibp1_max_err):.2e}, ibp2={float(ibp2_max_err):.2e}")

# 1B. 50-digit mpmath numerical verification across 6 alphas & 4 diamond placements + S^3/I* + round-S^3 control
#     (c_inv is an arbitrary alpha-independent dummy placeholder used solely to verify exact isotropic delta-alpha cancellation)
def verify_diamond_alpha_50dps(eta_0: mp.mpf, chi_max: mp.mpf, H2_fn, H2p_fn, H2pp_fn, c_inv: mp.mpf):
    eta_w = eta_0 + chi_max
    rho0 = mp.sin(chi_max) / mp.cos(eta_w)
    a0 = 1 / mp.cos(eta_0)

    def d_q_unit(chi):
        def integ(cp):
            ep = eta_0 + cp
            return (H2pp_fn(ep) - mp.tan(ep) * H2p_fn(ep) + 2 * H2_fn(ep)) * mp.cos(ep) * mp.sin(cp)
        return -a0 * mp.quad(integ, [0, chi])

    def d_q_unit_ibp(chi):
        ep = eta_0 + chi
        int_H2 = mp.quad(H2_fn, [eta_0, ep])
        return -a0 * (
            mp.sin(chi) * mp.cos(ep) * H2p_fn(ep)
            - mp.cos(chi) * mp.cos(ep) * H2_fn(ep)
            + mp.cos(eta_0) * H2_fn(eta_0)
            - mp.sin(eta_0) * int_H2
        )

    ibp_err = abs(d_q_unit(chi_max) - d_q_unit_ibp(chi_max))
    assert ibp_err < mp.mpf("1e-44")

    br_unit = -(1 / rho0) * mp.quad(lambda x: d_q_unit_ibp(x) / (a0 * mp.cos(eta_0 + x) ** 2), [0, chi_max])
    int_H2 = mp.quad(H2_fn, [eta_0, eta_w])
    int_sec2_H2 = mp.quad(lambda e: H2_fn(e) / (mp.cos(e) ** 2), [eta_0, eta_w])
    dir_unit = (
        -H2_fn(eta_0)
        - H2_fn(eta_w)
        + mp.cot(chi_max) * int_H2
        + (mp.cos(eta_0) * mp.cos(eta_w) / mp.sin(chi_max)) * int_sec2_H2
    )
    canc_err = abs(br_unit + dir_unit)

    ray_geom = (
        2 * H2_fn(eta_0)
        - mp.cot(chi_max) * int_H2
        - (mp.cos(eta_0) * mp.cos(eta_w) / mp.sin(chi_max)) * int_sec2_H2
    )
    vals = []
    for al in [mp.mpf("-1"), mp.mpf("0"), mp.mpf("0.25"), mp.mpf("0.5"), mp.mpf("1"), mp.mpf("1.5")]:
        da = al - mp.mpf("0.5")
        br_s = (da / 6) * br_unit
        dir_s = (al / 6 - mp.mpf(1) / 20) * (H2_fn(eta_0) - H2_fn(eta_w)) + (-al / 6 + c_inv) * ray_geom
        vals.append(-16 * (rho0 ** 2) * (br_s + dir_s))
    spread = max(abs(v - vals[3]) for v in vals)
    return {"ibp_err": float(ibp_err), "canc_err": float(canc_err), "alpha_spread_6": float(spread)}


test_H2 = lambda e: mp.mpf("0.7") + mp.mpf("1.3") * mp.cos(2 * e) + mp.mpf("0.5") * mp.sin(3 * e)
test_H2p = lambda e: -mp.mpf("2.6") * mp.sin(2 * e) + mp.mpf("1.5") * mp.cos(3 * e)
test_H2pp = lambda e: -mp.mpf("5.2") * mp.cos(2 * e) - mp.mpf("4.5") * mp.sin(3 * e)

kt1_alpha_50dps = {}
for lbl, e0, ch in [
    ("centered_[-pi/20,+pi/20]", -mp.pi / 20, mp.pi / 10),
    ("tip@bounce_[0,+pi/10]", mp.mpf(0), mp.pi / 10),
    ("waist@bounce_[-pi/10,0]", -mp.pi / 10, mp.pi / 10),
    ("wide_[-0.3,+0.5]", mp.mpf("-0.3"), mp.mpf("0.8")),
]:
    r = verify_diamond_alpha_50dps(e0, ch, test_H2, test_H2p, test_H2pp, mp.mpf(3) / 20)
    assert r["canc_err"] < 1e-45 and r["alpha_spread_6"] < 1e-45
    kt1_alpha_50dps[lbl] = r
    print(f"  50-dps {lbl:<26s}: |BR+DIR|={r['canc_err']:.3e}, 6-alpha spread={r['alpha_spread_6']:.3e}")

# Round-S^3 zero control (H_2 === 0):
zero_fn = lambda e: mp.mpf(0)
r_s3_zero = verify_diamond_alpha_50dps(-mp.pi / 20, mp.pi / 10, zero_fn, zero_fn, zero_fn, mp.mpf(1) / 6)
assert r_s3_zero["canc_err"] == 0.0 and r_s3_zero["alpha_spread_6"] == 0.0
kt1_alpha_50dps["round_S3_zero_control"] = r_s3_zero

# 1C. Exact S^3/I* TT multiplicity table p = 3..30 (correcting p2_04 index remark: p=13 absent!)
istar_tt_table = {p: ade_tt_mult("I*", p) for p in range(3, 31)}
surviving_Istar_p = [p for p, d in istar_tt_table.items() if d > 0]
assert surviving_Istar_p == [3, 11, 15, 19, 23, 27, 29]
assert istar_tt_table[3] == 5 and istar_tt_table[11] == 9 and istar_tt_table[13] == 0 and istar_tt_table[15] == 17
print(f"  S^3/I* surviving TT modes (p=3..30): {[(p, istar_tt_table[p]) for p in surviving_Istar_p]} (p=13 absent: d2=0)")

# 1D. Curved-S^3 = SU(2) TT eigenmode verification (div-free & (-Delta_2 + 2) H = (p^2 - 1) H)
def quat_mul(q, r):
    return np.array([
        q[0]*r[0] - q[1]*r[1] - q[2]*r[2] - q[3]*r[3],
        q[0]*r[1] + q[1]*r[0] + q[2]*r[3] - q[3]*r[2],
        q[0]*r[2] - q[1]*r[3] + q[2]*r[0] + q[3]*r[1],
        q[0]*r[3] + q[1]*r[2] - q[2]*r[1] + q[3]*r[0],
    ])


def mode_H_and_Xderivs(q, p_val: int):
    m = p_val - 3
    z1 = complex(q[0], q[3])
    if m == 0:
        w, dw1, dw2, dw3 = 1.0 + 0.0j, 0.0j, 0.0j, 0.0j
    else:
        wm1 = z1 ** (m - 1)
        w = wm1 * z1
        dw1 = m * wm1 * complex(-q[1], -q[2])
        dw2 = m * wm1 * complex(-q[2], q[1])
        dw3 = m * wm1 * complex(-q[3], q[0])

    def to_mat(cval):
        u, v = cval.real, cval.imag
        return 0.5 * np.array([[u, v, 0.0], [v, -u, 0.0], [0.0, 0.0, 0.0]], dtype=float)

    return to_mat(w), (to_mat(dw1), to_mat(dw2), to_mat(dw3))


q0 = np.array([0.53, -0.37, 0.61, 0.458], dtype=float)
q0 /= np.linalg.norm(q0)
L_mats = [
    np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float),
    np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float),
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float),
]
basis_i = [np.array([0.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])]
eps_fd = 1e-6
max_div_err, max_eig_err = 0.0, 0.0
for pv in [3, 4, 5, 11, 13, 15, 19, 23]:
    def cov_d(q):
        H, dH = mode_H_and_Xderivs(q, pv)
        return [dH[k] + (L_mats[k] @ H - H @ L_mats[k]) for k in range(3)]
    H0, _ = mode_H_and_Xderivs(q0, pv)
    nab0 = cov_d(q0)
    div_v = np.array([sum(nab0[k][k, b] for k in range(3)) for b in range(3)])
    lap = np.zeros((3, 3), dtype=float)
    for k in range(3):
        qp = q0 + eps_fd * quat_mul(q0, basis_i[k]); qp /= np.linalg.norm(qp)
        qm = q0 - eps_fd * quat_mul(q0, basis_i[k]); qm /= np.linalg.norm(qm)
        Xk_nabk = (cov_d(qp)[k] - cov_d(qm)[k]) / (2 * eps_fd)
        lap += Xk_nabk + (L_mats[k] @ nab0[k] - nab0[k] @ L_mats[k])
    res_m = (-lap + 2.0 * H0) - (pv * pv - 1.0) * H0
    max_div_err = max(max_div_err, float(np.linalg.norm(div_v)) / float(np.linalg.norm(H0)))
    max_eig_err = max(max_eig_err, float(np.linalg.norm(res_m)) / (float(np.linalg.norm(H0)) * (pv**2)))
assert max_div_err < 1e-12 and max_eig_err < 1e-7
print(f"  Curved-S^3 TT eigenmodes (p=3..23): max div err = {max_div_err:.2e}, max eig err = {max_eig_err:.2e}")

# 1E. Exact homogeneous-mode affine-redshift identity on S^3 (P0-1 & P0-2 remediation):
#     Along exact S^3 null geodesics in homogeneous g_ij^(alpha)(eta) = delta_ij + eps*h_ij(eta) + alpha*eps^2*(h^2)_ij(eta),
#     left-invariant momentum obeys the Euler--Arnold ODE dp_a/deta = 2 eps_{abc} p_b v_c, conserving |p|^2 === |p(eta_0)|^2.
#     The O(h) rotation delta p_a^(1) enters E(eta)^2 via -2 h^{ab} u_a delta p_b^(1), which is odd (cubic) in u in S^2
#     and vanishes identically upon sky averaging. Averaging over the PHYSICAL orthonormal sky Omega_phys(eta_0)
#     (p_a(eta_0) = (g^(alpha)(eta_0)^{1/2})_{ab} u_b, E(eta_0) === 1) yields:
#       <E(eta_0)/E(eta) - 1>_{Omega_phys(eta_0)} = (-alpha/6 + 1/20)*H2(eta_0) + (+alpha/6 - 7/60)*H2(eta) + (1/15)*<h_ij(eta_0) h^ij(eta)>.
#     Verify directly via exact S^2 Gauss-Legendre quadrature AND full RK4 integration of the S^3 Euler--Arnold ODE!
H0_mat = np.array([[0.7, -0.4, 0.3], [-0.4, -0.2, 0.6], [0.3, 0.6, -0.5]], dtype=float)
H1_mat = np.array([[-0.3, 0.5, -0.2], [0.5, 0.8, 0.1], [-0.2, 0.1, -0.5]], dtype=float)
assert abs(np.trace(H0_mat)) < 1e-15 and abs(np.trace(H1_mat)) < 1e-15

def h_homog(eta_v):
    return H0_mat * math.cos(3.0 * eta_v) + H1_mat * math.sin(3.0 * eta_v)

eta0_hom, eta1_hom = -0.2, 0.3
H_at_0, H_at_1 = h_homog(eta0_hom), h_homog(eta1_hom)
H2_0_val = float(np.sum(H_at_0 * H_at_0))
H2_1_val = float(np.sum(H_at_1 * H_at_1))
H01_corr = float(np.sum(H_at_0 * H_at_1))

def mat_sqrt_sym(M):
    w, V = np.linalg.eigh(M)
    return (V * np.sqrt(w)) @ V.T

mu_q, wmu_q = np.polynomial.legendre.leggauss(10)
ph_q = np.linspace(0.0, 2.0 * math.pi, 14, endpoint=False)
UU_h, PP_h = np.meshgrid(mu_q, ph_q, indexing="ij")
WU_h, _ = np.meshgrid(0.5 * wmu_q / len(ph_q), ph_q, indexing="ij")
st_h = np.sqrt(1.0 - UU_h.ravel() ** 2)
u_sky = np.stack([st_h * np.cos(PP_h.ravel()), st_h * np.sin(PP_h.ravel()), UU_h.ravel()], axis=-1)
wt_sky = WU_h.ravel()

eps_aff = 1e-3
affine_redshift_max_err = 0.0
euler_arnold_norm_max_err = 0.0
for al_v in [-1.0, 0.0, 0.25, 0.5, 1.0, 1.5]:
    def eval_ode_sky(ev):
        g0 = np.eye(3) + ev * H_at_0 + al_v * (ev ** 2) * (H_at_0 @ H_at_0)
        e0 = mat_sqrt_sym(g0)
        p_vec = u_sky @ e0.T  # p_a(eta_0) = e_a^b u_b => E(eta_0) === 1 identically
        p0_norm2 = np.sum(p_vec * p_vec, axis=-1)
        n_st = 80
        dt = (eta1_hom - eta0_hom) / n_st

        def rhs_ea(t, pv):
            ht = h_homog(t)
            ginv = np.linalg.inv(np.eye(3) + ev * ht + al_v * (ev ** 2) * (ht @ ht))
            gp = pv @ ginv.T
            E = np.sqrt(np.sum(pv * gp, axis=-1, keepdims=True))
            v = gp / E
            return 2.0 * np.cross(pv, v)

        for s_i in range(n_st):
            t = eta0_hom + s_i * dt
            k1 = rhs_ea(t, p_vec)
            k2 = rhs_ea(t + 0.5 * dt, p_vec + 0.5 * dt * k1)
            k3 = rhs_ea(t + 0.5 * dt, p_vec + 0.5 * dt * k2)
            k4 = rhs_ea(t + dt, p_vec + dt * k3)
            p_vec = p_vec + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        p1_norm2 = np.sum(p_vec * p_vec, axis=-1)
        norm_err = float(np.max(np.abs(p1_norm2 - p0_norm2)))
        g1_inv = np.linalg.inv(np.eye(3) + ev * H_at_1 + al_v * (ev ** 2) * (H_at_1 @ H_at_1))
        E1 = np.sqrt(np.sum(p_vec * (p_vec @ g1_inv.T), axis=-1))
        return 1.0 / E1 - 1.0, norm_err

    r_p, ne_p = eval_ode_sky(+eps_aff)
    r_m, ne_m = eval_ode_sky(-eps_aff)
    euler_arnold_norm_max_err = max(euler_arnold_norm_max_err, ne_p, ne_m)
    avg_num = float(np.sum(wt_sky * (0.5 * (r_p + r_m) / (eps_aff ** 2))))
    avg_th = (-al_v / 6.0 + 1.0 / 20.0) * H2_0_val + (al_v / 6.0 - 7.0 / 60.0) * H2_1_val + (1.0 / 15.0) * H01_corr
    affine_redshift_max_err = max(affine_redshift_max_err, abs(avg_num - avg_th))
assert affine_redshift_max_err < 1e-6 and euler_arnold_norm_max_err < 1e-13
print(f"  Homogeneous S^3 Euler--Arnold + physical-sky affine-redshift identity (6 alphas): max err = {affine_redshift_max_err:.2e}, |p|^2 err = {euler_arnold_norm_max_err:.2e}")


# =============================================================================
# PART 2 (KT2): PALINDROMIC-BREAKING THEOREM & EXACT NULL-SPACE ANNIHILATION
# =============================================================================
print("\n" + "=" * 86)
print("PART 2 (KT2): UNIVERSAL CONFORMAL CHIRAL THEOREM & PHI_ALPHA / R^2,C^2 ANNIHILATION")
print("=" * 86)

def F_and_Fp_ade(name: str, t: mp.mpf):
    order, cls = ade_classes(name)
    ct, st = mp.cosh(t), mp.sinh(t)
    fv, fpv = mp.mpf(0), mp.mpf(0)
    for sz, th in cls:
        d = 2 * (ct - mp.cos(th))
        fv += sz / d
        fpv += -sz * (2 * st) / (d * d)
    return fv / order, fpv / order


palindromic_checks = {}
t_test = mp.mpf("0.37")
for name, order in [("RP3", 2), ("Q8", 8), ("T*", 24), ("O*", 48), ("I*", 120)]:
    f_p, fp_p = F_and_Fp_ade(name, t_test)
    f_m, fp_m = F_and_Fp_ade(name, -t_test)
    m1 = m_ade(name, 2)  # j = 1 => 2j = 2
    assert m_ade(name, 0) == 1 and m_ade(name, 1) == 0
    # Spin-2 heat traces:
    #   K_+(t) = e^{-2t}(-F'(t) + 4 F(t))
    #   K_-(t) = e^{+2t}(-F'(t) - 4 F(t)) + 3 e^{+t} + m(1) e^{-t}
    Kp_t = mp.exp(-2 * t_test) * (-fp_p + 4 * f_p)
    Km_mt = mp.exp(-2 * t_test) * (-fp_m - 4 * f_m) + 3 * mp.exp(-t_test) + m1 * mp.exp(t_test)
    s2_pal_err = abs((Kp_t + Km_mt) - (3 * mp.exp(-t_test) + m1 * mp.exp(t_test)))
    # Spin-1 Maxwell heat traces:
    #   K_+^EM(t) = e^{+t}(-F'(t) - 2 F(t)) + 1,  K_-^EM(t) = e^{-t}(-F'(t) + 2 F(t))
    Kp_em = mp.exp(t_test) * (-fp_p - 2 * f_p) + 1
    Km_em_mt = mp.exp(t_test) * (-fp_m + 2 * f_m)
    em_pal_err = abs((Kp_em + Km_em_mt) - 1)
    assert s2_pal_err < mp.mpf("1e-47") and em_pal_err < mp.mpf("1e-47")

    # Exact theorem value for pi^2 (u_+^conf - u_-^conf) = ((3 - m1)/4) * |Gamma|:
    exact_du_conf = Fraction((3 - m1) * order, 4)
    # Cross-check against direct Bernoulli quasi-polynomial sum!
    P_ade = {"RP3": 2, "Q8": 4, "T*": 12, "O*": 24, "I*": 60}[name]
    C_r, A_r, B_r = extract_quasi_poly_deg2_frac(
        lambda pp, nm=name, od=order: od * (m_ade(nm, pp - 3) * (pp + 2) - m_ade(nm, pp + 1) * (pp - 2)),
        P_ade, 3,
    )
    assert sum(C_r.values()) == 0 and sum(A_r.values()) == 0 and sum(B_r.values()) == 0
    s_conf = Fraction(0)
    for r in range(P_ade):
        p0 = 3 + ((r - 3) % P_ade)
        x = Fraction(p0, P_ade)
        s_conf += C_r[r] * (P_ade**3) * bern_zeta_neg(3, x) + A_r[r] * (P_ade**2) * bern_zeta_neg(2, x) + B_r[r] * P_ade * bern_zeta_neg(1, x)
    assert s_conf / 4 == exact_du_conf
    palindromic_checks[name] = {
        "order": order,
        "m_Gamma_j1": m1,
        "spin2_palindromic_identity_err_50dps": float(s2_pal_err),
        "maxwell_palindromic_identity_err_50dps": float(em_pal_err),
        "pi2_du_conf_exact": str(exact_du_conf),
        "E_chiral_conf_exact": str(Fraction(3 - m1, 2)),
        "uv_sum_rules": [0, 0, 0],
    }
    print(f"  {name:<4s}: m(j=1)={m1}, pi^2(u_+^conf-u_-^conf)={str(exact_du_conf):>4s}, S2 pal err={float(s2_pal_err):.2e}, EM pal err={float(em_pal_err):.2e}")

# Also verify L(3,1) general formula (m_{L(3,1)}(j=1) = 1 => E_chiral_conf = +1, pi^2 du_conf = +3/2):
C_l3, A_l3, B_l3 = extract_quasi_poly_deg2_frac(
    lambda pp: 3 * (m_lens(3, pp - 3) * (pp + 2) - m_lens(3, pp + 1) * (pp - 2)), 6, 3
)
s_conf_l3 = sum(
    C_l3[r] * (6**3) * bern_zeta_neg(3, Fraction(3 + ((r - 3) % 6), 6))
    + A_l3[r] * (6**2) * bern_zeta_neg(2, Fraction(3 + ((r - 3) % 6), 6))
    + B_l3[r] * 6 * bern_zeta_neg(1, Fraction(3 + ((r - 3) % 6), 6))
    for r in range(6)
)
assert s_conf_l3 / 4 == Fraction(3, 2)
palindromic_checks["L(3,1)_extension"] = {"m_Gamma_j1": 1, "pi2_du_conf_exact": str(s_conf_l3 / 4), "E_chiral_conf_exact": "1"}

# 50-digit proof of round-metric d(R^2)/dlam|_1 = d(C^2)/dlam|_1 = d(A_4)/dlam|_1 === 0 and A_4''(1) = 554/15:
R3_mp = lambda l: 2 * (4 - l * l) * (l ** (mp.mpf(2) / 3))
C2_mp = lambda l: (mp.mpf(64) / 3) * ((1 - l * l) ** 2) * (l ** (mp.mpf(4) / 3))
A4_mp = lambda l: (-mp.mpf(5) / 192) * (R3_mp(l) ** 2) + (mp.mpf(63) / 320) * C2_mp(l)
dR2_1 = abs(mp.diff(lambda l: R3_mp(l) ** 2, mp.mpf(1)))
dC2_1 = abs(mp.diff(C2_mp, mp.mpf(1)))
dA4_1 = abs(mp.diff(A4_mp, mp.mpf(1)))
A4_pp_1_val = mp.diff(A4_mp, mp.mpf(1), 2)
A4_pp_1_err = abs(A4_pp_1_val - mp.mpf(554) / 15)
assert dR2_1 < mp.mpf("1e-47") and dC2_1 < mp.mpf("1e-47") and dA4_1 < mp.mpf("1e-47") and A4_pp_1_err < mp.mpf("1e-45")
print(f"  50-dps round-metric derivatives: d(R^2)/dlam|_1={float(dR2_1):.2e}, d(C^2)/dlam|_1={float(dC2_1):.2e}, d(A_4)/dlam|_1={float(dA4_1):.2e}, |A_4''(1)-554/15|={float(A4_pp_1_err):.2e}")

# Exact 50-dps verification of anisotropic tensor metric redefinition on Berger L(q>=3,1) (P0-2 & P1-1 & P1-2):
#   On fixed-volume Berger S^3_lam: G^3_3 - G^1_1 = R^3_3 - R^1_1 = 4*(lam^2 - 1)*lam^(2/3) / a_bar^2
#   => d(G^3_3 - G^1_1)/dlam|_1 = +8 / a_bar^2.
#   Mixed-index variation delta_alpha(T^3_3 - T^1_1) = -delta_alpha * d(G^3_3 - G^1_1)/deps|_0
#   (equivalently on-shell vacuum dS_4 E_uv := G_uv + Lambda g_uv = 0 for static Phi at the bounce) gives
#   delta_alpha(p_par - p_perp) = -4*delta_alpha*(Phi_par - Phi_perp)/a^2 (lowered-index off-shell static R_t x S^3
#   differs by -Phi^i_j/a^2 giving -3*delta_alpha*(Phi_par - Phi_perp)/a^2).
#   For the p=3, M_L=0 mode on L(3,1) (s = -1 in Theorem 2's 2j_L=p-3 convention; s = +1 in p2_03's 2K=p-1+2s slope convention),
#   H = diag(-1,-1,+2)/sqrt(6) => H^2 = (1/6) diag(1,1,4), so Phi_par - Phi_perp = 4/6 - 1/6 = +1/2 != 0.
G_aniso_mp = lambda l: 4 * (l * l - 1) * (l ** (mp.mpf(2) / 3))
dG_aniso_1 = mp.diff(G_aniso_mp, mp.mpf(1))
dG_aniso_err = abs(dG_aniso_1 - 8)
assert dG_aniso_err < mp.mpf("1e-47")
H_L3_p3 = np.diag([-1.0, -1.0, 2.0]) / math.sqrt(6.0)
H2_L3_p3 = H_L3_p3 @ H_L3_p3
phi_aniso_L3_p3 = float(H2_L3_p3[2, 2] - H2_L3_p3[0, 0])
assert abs(phi_aniso_L3_p3 - 0.5) < 1e-15
print(f"  Anisotropic L(q>=3,1) redefinition check: |d(G^3_3-G^1_1)/dlam|_1 - 8| = {float(dG_aniso_err):.2e}, p=3 L(3,1) Phi_par-Phi_perp = {phi_aniso_L3_p3:+.4f} (!= 0)")


# =============================================================================
# PART 3 (KT3): EXACT RATIONAL lam_B = 1 BENCHMARKS & O(q^0) SCALING NO-GO PROOF
# =============================================================================
print("\n" + "=" * 86)
print("PART 3 (KT3): EXACT RATIONAL lam_B = 1 BENCHMARKS (L(2), L(20), L(24)) & O(q^0) NO-GO")
print("=" * 86)

def exact_lens_lam1_conformal_species(q_ord: int):
    """Exact Fraction pi^2 u_top(L(q,1); lam_B = 1) for scal_per, scal_anti, weyl_per, weyl_anti, maxwell_per."""
    assert q_ord % 2 == 0
    P = q_ord

    # 1. Conformal scalar (p >= 1, omega = p, S^3 deg = p^2):
    #    pi^2 u_top = (1/4) * sum_{p>=1} (q * p * m_lens(q, p-1) - p^2) * p
    out = {}
    for key, anti in [("scal_per", False), ("scal_anti", True)]:
        C, A, B = extract_quasi_poly_deg2_frac(
            lambda pp, a=anti: q_ord * pp * m_lens(q_ord, pp - 1, anti_per=a) - pp * pp, P, 1
        )
        assert sum(C.values()) == 0 and sum(A.values()) == 0 and sum(B.values()) == 0
        s = sum(
            C[r] * (P**3) * bern_zeta_neg(3, Fraction(1 + ((r - 1) % P), P))
            + A[r] * (P**2) * bern_zeta_neg(2, Fraction(1 + ((r - 1) % P), P))
            + B[r] * P * bern_zeta_neg(1, Fraction(1 + ((r - 1) % P), P))
            for r in range(P)
        )
        out[key] = s / 4

    # 2. Weyl fermion (n >= 0, x = n + 3/2, S^3 deg = (n+1)(n+2) = x^2 - 1/4; FERMI MINUS SIGN!):
    #    d_Weyl(n) = m(n)*(n+2) + m(n+1)*(n+1) [wait: 1 Weyl has BOTH +- chiralities? Check probe3!]
    #    Check probe3 line 326 & 412:
    #      On Berger S^3 at lam=1:
    #        For 2j >= 0 (deg = 2j+1): 2M in {start_fer..2j+1} where start_fer has opposite parity to 2j:
    #        - 2M = 2j+1: 1 mode at |lam| = 2j + 3/2 -> n = 2j, |lam| = n + 3/2
    #        - |2M| <= 2j-1: 2 modes at |lam| = (2j+1) +- 1/2 -> (2j+3/2) and (2j+1/2)!
    #      Wait: total modes at |lam| = n + 3/2 (n >= 0):
    #        - from 2j = n: deg = n+1, modes with |2M| <= n+1 -> m(n+1) * (n+1)
    #        - from 2j = n+1: deg = n+2, modes with |2M| <= n -> m(n) * (n+2)
    #      Total S^3 deg at |lam| = n+3/2: (n+1)(n+2) + (n+2)(n+1) = 2(n+1)(n+2)!
    #      And probe3 applies fv = -0.5 * wb * exp(-w2*t) and kahan_add(0.5*(q*sh_q - sh_s3)) => factor -(1/4) * sum!
    for key, anti in [("weyl_per", False), ("weyl_anti", True)]:
        def R_weyl(n, a=anti):
            dq = m_lens(q_ord, n + 1, anti_per=a) * (n + 1) + m_lens(q_ord, n, anti_per=a) * (n + 2)
            return q_ord * dq - 2 * (n + 1) * (n + 2)
        # Express R_weyl(n) as C_r x^2 + A_r x + B_r where x = n + 3/2:
        z_m1 = Fraction(0)
        for r in range(P):
            ns = [r + m * P for m in range(4)]
            ys = [Fraction(R_weyl(nv)) for nv in ns]
            assert (ys[3] - 3 * ys[2] + 3 * ys[1] - ys[0]) == 0
            xs = [Fraction(2 * nv + 3, 2) for nv in ns]
            d2 = (ys[2] - ys[1]) - (ys[1] - ys[0])
            c = d2 / (2 * P * P)
            L1 = ys[0] - c * xs[0] * xs[0]
            L2 = ys[1] - c * xs[1] * xs[1]
            a = (L2 - L1) / P
            b = L1 - a * xs[0]
            x4 = Fraction(2 * (r + 4 * P) + 3, 2)
            assert Fraction(R_weyl(r + 4 * P)) == c * x4 * x4 + a * x4 + b
            q_frac = Fraction(2 * r + 3, 2 * P)
            z_m1 += c * (P**3) * bern_zeta_neg(3, q_frac) + a * (P**2) * bern_zeta_neg(2, q_frac) + b * P * bern_zeta_neg(1, q_frac)
        out[key] = -z_m1 / 4  # Fermi minus sign!

    # 3. Periodic Maxwell (p >= 2, omega = p, S^3 deg = 2*(p^2 - 1)):
    C, A, B = extract_quasi_poly_deg2_frac(
        lambda pp: q_ord * (m_lens(q_ord, pp - 2) * (pp + 1) + m_lens(q_ord, pp) * (pp - 1)) - 2 * (pp * pp - 1),
        P, 2,
    )
    assert sum(C.values()) == 0 and sum(A.values()) == 0 and sum(B.values()) == 0
    s_em = sum(
        C[r] * (P**3) * bern_zeta_neg(3, Fraction(2 + ((r - 2) % P), P))
        + A[r] * (P**2) * bern_zeta_neg(2, Fraction(2 + ((r - 2) % P), P))
        + B[r] * P * bern_zeta_neg(1, Fraction(2 + ((r - 2) % P), P))
        for r in range(P)
    )
    out["maxwell_per"] = s_em / 4
    return out


def exact_lens_lam1_grav_50dps(q_ord: int) -> mp.mpf:
    """50-digit Hurwitz+digamma physical EH graviton pi^2 u_top(L(q,1); lam_B = 1)."""
    P = q_ord if (q_ord % 2 == 0) else 2 * q_ord
    C_r, A_r, B_r = extract_quasi_poly_deg2_frac(
        lambda pp: q_ord * (m_lens(q_ord, pp + 1) * (pp - 2) + m_lens(q_ord, pp - 3) * (pp + 2)) - 2 * (pp * pp - 4),
        P, 3,
    )
    assert sum(C_r.values()) == 0 and sum(A_r.values()) == 0 and sum(B_r.values()) == 0

    def sum_pow(d, s_pow):
        acc = mp.mpf(0)
        if s_pow == 1:
            for r, cv in d.items():
                if cv == 0:
                    continue
                p0 = 3 + ((r - 3) % P)
                acc += (mp.mpf(cv.numerator) / cv.denominator) * (-mp.psi(0, mp.mpf(p0) / P)) / P
        else:
            sc = mp.mpf(P) ** (-s_pow)
            for r, cv in d.items():
                if cv == 0:
                    continue
                p0 = 3 + ((r - 3) % P)
                acc += (mp.mpf(cv.numerator) / cv.denominator) * mp.zeta(s_pow, mp.mpf(p0) / P) * sc
        return acc

    def eval_pow(pow_p):
        return sum_pow(C_r, -(pow_p + 2)) + sum_pow(A_r, -(pow_p + 1)) + sum_pow(B_r, -pow_p)

    s_eh = eval_pow(1)
    for k in range(1, 55):
        dk = ((-1) ** k) * mp.binomial(mp.mpf("0.5"), k) * eval_pow(1 - 2 * k)
        s_eh += dk
        if abs(dk) < mp.mpf("1e-47"):
            break
    return s_eh / 4


kt3_exact_lam1 = {}
for q_ord in [2, 20, 24]:
    ex = exact_lens_lam1_conformal_species(q_ord)
    g50 = exact_lens_lam1_grav_50dps(q_ord)
    if q_ord == 2:
        assert ex["scal_per"] == Fraction(-1, 32)
        assert ex["scal_anti"] == Fraction(1, 32)
        assert ex["weyl_per"] == 0 and ex["weyl_anti"] == 0
        assert ex["maxwell_per"] == Fraction(3, 16)
    kt3_exact_lam1[str(q_ord)] = {
        k: {"exact": str(v), "float": float(v)} for k, v in ex.items()
    }
    kt3_exact_lam1[str(q_ord)]["grav_per"] = {"50dps": mp.nstr(g50, 30), "float": float(g50)}
    u_s1_unit = (q_ord**4) / 1440.0
    print(
        f"  L({q_ord:2d},1) exact lam=1: grav={float(g50):+10.5f}, scal_p={float(ex['scal_per']):+9.5f}, "
        f"scal_a={float(ex['scal_anti']):+9.5f}, weyl_p={float(ex['weyl_per']):+9.5f}, "
        f"weyl_a={float(ex['weyl_anti']):+9.5f}, maxw={float(ex['maxwell_per']):+9.5f}"
    )

# Exact 50-digit scaling proof of O(q^0) no-go for O(q^4)-cancelled cocktails vs O(q^1) for unbalanced O(q^4):
def Lmin_over_lsp_scaling(q_val, power_q, c_coeff=mp.mpf("1.5"), N_sp=mp.mpf(15), lam_st=mp.mpf("1.05")):
    R3p_v = (mp.mpf(16) / 3) * (lam_st ** (-mp.mpf(1) / 3)) * (1 - lam_st * lam_st)
    abar_v = mp.sqrt(2 * (-c_coeff * (q_val ** power_q)) / R3p_v)
    return 2 * mp.pi * abar_v * (lam_st ** (mp.mpf(2) / 3)) / (q_val * mp.sqrt(N_sp))

oq4_canc_deriv = abs(mp.diff(lambda qv: Lmin_over_lsp_scaling(qv, 2), mp.mpf(24)))
unbal_ratio_err = abs(Lmin_over_lsp_scaling(mp.mpf(24), 4) / Lmin_over_lsp_scaling(mp.mpf(20), 4) - mp.mpf(24) / 20)
assert oq4_canc_deriv < mp.mpf("1e-48") and unbal_ratio_err < mp.mpf("1e-48")
print(f"  50-dps scaling theorem: O(q^4)-cancelled d(L_min/l_sp)/dq = {float(oq4_canc_deriv):.2e} (O(q^0) NO-GO); unbalanced ratio err = {float(unbal_ratio_err):.2e}")


# =============================================================================
# PART 4 (KT4): EXACT RATIONAL TWISTED DIRAC ON S^3/I* & GALOIS-INVARIANCE PROOF
# =============================================================================
print("\n" + "=" * 86)
print("PART 4 (KT4): EXACT RATIONAL TWISTED DIRAC/WEYL ON S^3/I* & GALOIS-INVARIANCE PROOF")
print("=" * 86)

phi = (1 + mp.sqrt(5)) / 2
phi_bar = (1 - mp.sqrt(5)) / 2  # Galois conjugate sigma(sqrt(5)) = -sqrt(5)

istar_raw = [
    ("1",   1,  mp.mpf(0),      mp.mpf(2),  mp.mpf(2),      mp.mpf(2)),
    ("-1",  1,  mp.pi,          mp.mpf(2),  mp.mpf(-2),     mp.mpf(-2)),
    ("4",  30,  mp.pi / 2,      mp.mpf(2),  mp.mpf(0),      mp.mpf(0)),
    ("3",  20,  2 * mp.pi / 3,  mp.mpf(2),  mp.mpf(-1),     mp.mpf(-1)),
    ("6",  20,  mp.pi / 3,      mp.mpf(2),  mp.mpf(1),      mp.mpf(1)),
    ("5a", 12,  2 * mp.pi / 5,  mp.mpf(2),  phi - 1,        -phi),
    ("5b", 12,  4 * mp.pi / 5,  mp.mpf(2),  -phi,           phi - 1),
    ("10a",12,  mp.pi / 5,      mp.mpf(2),  phi,            1 - phi),
    ("10b",12,  3 * mp.pi / 5,  mp.mpf(2),  1 - phi,        phi),
]
assert sum(c[1] for c in istar_raw) == 120

classes = []
for name, mult, th, f0, f1, f2 in istar_raw:
    assert abs(f1 - 2 * mp.cos(th)) < mp.mpf("1e-45")
    classes.append({
        "name": name, "mult": mult, "theta": th,
        "fund": [f0, f1, f2],
        "adj": [mp.mpf(3), f1 * f1 - 1, f2 * f2 - 1],
    })

# Verify exact character orthogonality norms & exact Galois invariance of F_rho(t)!
max_orth_err = mp.mpf(0)
for rep_type, exp_norms in [("fund", [4, 1, 1]), ("adj", [9, 1, 1])]:
    for idx in range(3):
        norm = sum(c["mult"] * (c[rep_type][idx] ** 2) for c in classes) / 120
        max_orth_err = max(max_orth_err, abs(norm - exp_norms[idx]))
assert max_orth_err < mp.mpf("1e-45")

# Galois invariance proof: under sigma(sqrt(5) -> -sqrt(5)):
#   cos(2pi/5) = (phi-1)/2 <-> cos(4pi/5) = -phi/2  (5a <-> 5b)
#   cos(pi/5)  = phi/2     <-> cos(3pi/5) = (1-phi)/2 (10a <-> 10b)
#   and chi_{rho_i}(5a) <-> chi_{rho_i}(5b), chi_{rho_i}(10a) <-> chi_{rho_i}(10b) for BOTH i=1,2!
# Therefore the 5a+5b and 10a+10b pair sums in F_rho(t) are IDENTICALLY fixed by sigma => F_rho(t) in Q[[t]]!
t_g = mp.mpf("0.37")
max_galois_err = mp.mpf(0)
for rep_type in ["fund", "adj"]:
    for idx in [1, 2]:
        # Evaluate 5a + 5b and 10a + 10b explicitly and check sigma(pair) == pair!
        c_by_name = {c["name"]: c for c in classes}
        for nA, nB in [("5a", "5b"), ("10a", "10b")]:
            cA, cB = c_by_name[nA], c_by_name[nB]
            # Original: chi(A)/(cosh t - cos thA) + chi(B)/(cosh t - cos thB)
            orig = cA[rep_type][idx] / (2 * (mp.cosh(t_g) - mp.cos(cA["theta"]))) + cB[rep_type][idx] / (2 * (mp.cosh(t_g) - mp.cos(cB["theta"])))
            # Apply sigma: swaps (chi_A, cos thA) <-> (chi_B, cos thB), so sigma(orig) = termB + termA == orig!
            # Check numerically that sigma maps cos(thA) -> cos(thB) and chi(A) -> chi(B):
            chi_conj_err = abs(cA[rep_type][idx] - cB[rep_type][3 - idx])  # since rho_2 = sigma(rho_1)!
            max_galois_err = max(max_galois_err, chi_conj_err)
            # Also verify numerator of combined fraction has ZERO sqrt(5) part!
            # termA + termB = [ (chiA + chiB)*cosh t - (chiA*cosB + chiB*cosA) ] / [ 2*(cosh^2 t - (cosA+cosB)*cosh t + cosA*cosB) ]
            s_chi = cA[rep_type][idx] + cB[rep_type][idx]
            s_cos = mp.cos(cA["theta"]) + mp.cos(cB["theta"])
            p_cos = mp.cos(cA["theta"]) * mp.cos(cB["theta"])
            cross = cA[rep_type][idx] * mp.cos(cB["theta"]) + cB[rep_type][idx] * mp.cos(cA["theta"])
            for v in (s_chi, s_cos, p_cos, cross):
                fv = Fraction(float(v)).limit_denominator(100)
                max_galois_err = max(max_galois_err, abs(v - mp.mpf(fv.numerator) / fv.denominator))
assert max_galois_err < mp.mpf("1e-47")
print(f"  Exact character orthogonality err = {float(max_orth_err):.2e}; exact Galois Q-invariance err = {float(max_galois_err):.2e}")

# Exact Fraction Bernoulli derivation for all 6 representations (zero floating-point PSLQ needed!):
E_WEYL_S3_FRAC = Fraction(17, 960)
E_DIRAC_S3_FRAC = Fraction(17, 480)
kt4_exact = {}
for rep_type, dim in [("fund", 2), ("adj", 3)]:
    kt4_exact[rep_type] = {}
    for idx, rname in enumerate(["rho_0", "rho_1", "rho_2"]):
        def m_rho_int(k: int, rt=rep_type, ri=idx) -> int:
            val = sum(c["mult"] * c[rt][ri] * su2_chi(k, c["theta"]) for c in classes) / 120
            iv = int(mp.nint(val))
            assert abs(val - iv) < mp.mpf("1e-35")
            return iv

        P = 60
        def eval_exact_frac(fn):
            zm1, z0 = Fraction(0), Fraction(0)
            for r in range(P):
                ns = [r + m * P for m in range(4)]
                ys = [Fraction(fn(nv)) for nv in ns]
                assert (ys[3] - 3 * ys[2] + 3 * ys[1] - ys[0]) == 0
                xs = [Fraction(2 * nv + 3, 2) for nv in ns]
                d2 = (ys[2] - ys[1]) - (ys[1] - ys[0])
                c = d2 / (2 * P * P)
                L1 = ys[0] - c * xs[0] * xs[0]
                L2 = ys[1] - c * xs[1] * xs[1]
                a = (L2 - L1) / P
                b = L1 - a * xs[0]
                x4 = Fraction(2 * (r + 4 * P) + 3, 2)
                assert Fraction(fn(r + 4 * P)) == c * x4 * x4 + a * x4 + b
                qf = Fraction(2 * r + 3, 2 * P)
                zm1 += c * (P**3) * bern_zeta_neg(3, qf) + a * (P**2) * bern_zeta_neg(2, qf) + b * P * bern_zeta_neg(1, qf)
                z0 += c * (P**2) * bern_zeta_neg(2, qf) + a * P * bern_zeta_neg(1, qf) + b * bern_zeta_neg(0, qf)
            return zm1, z0

        zt_m1, zt_0 = eval_exact_frac(lambda n: m_rho_int(n) * (n + 2) + m_rho_int(n + 1) * (n + 1))
        zd_m1, zd_0 = eval_exact_frac(lambda n: m_rho_int(n) * (n + 2) - m_rho_int(n + 1) * (n + 1))
        assert zd_m1 == 0, f"Dirac chiral energy asymmetry must vanish identically: {zd_m1}"
        E_dirac = -zt_m1
        E_weyl = -zt_m1 / 2
        pi2_dC_dirac = (120 * E_dirac - dim * E_DIRAC_S3_FRAC) / 2
        pi2_dC_weyl = (120 * E_weyl - dim * E_WEYL_S3_FRAC) / 2
        kt4_exact[rep_type][rname] = {
            "dim": dim,
            "E_Dirac_exact": str(E_dirac),
            "E_Weyl_exact": str(E_weyl),
            "eta_D_0_exact": str(zd_0),
            "zeta_D_0_exact": str(zt_0),
            "Delta_E_chiral_exact": str(-zd_m1 / 2),
            "pi2_dC_Dirac_exact": str(pi2_dC_dirac),
            "pi2_dC_Weyl_exact": str(pi2_dC_weyl),
            "E_Dirac_float": float(E_dirac),
            "pi2_dC_Dirac_float": float(pi2_dC_dirac),
            "eta_D_0_float": float(zd_0),
        }
        print(
            f"  {rep_type}({rname}): E_Dirac = {str(E_dirac):>14s} ({float(E_dirac):+11.7f}), "
            f"eta_D(0) = {str(zd_0):>10s}, pi^2 dC_Dirac = {float(pi2_dC_dirac):+10.4f}"
        )

# Verify exact rational Galois jumps:
jump_fund_E = Fraction(kt4_exact["fund"]["rho_2"]["E_Dirac_exact"]) - Fraction(kt4_exact["fund"]["rho_1"]["E_Dirac_exact"])
jump_adj_E = Fraction(kt4_exact["adj"]["rho_2"]["E_Dirac_exact"]) - Fraction(kt4_exact["adj"]["rho_1"]["E_Dirac_exact"])
jump_fund_eta = Fraction(kt4_exact["fund"]["rho_2"]["eta_D_0_exact"]) - Fraction(kt4_exact["fund"]["rho_1"]["eta_D_0_exact"])
jump_adj_eta = Fraction(kt4_exact["adj"]["rho_2"]["eta_D_0_exact"]) - Fraction(kt4_exact["adj"]["rho_1"]["eta_D_0_exact"])
assert jump_fund_E == Fraction(-29, 5) and jump_adj_E == Fraction(-26, 5)
assert jump_fund_eta == Fraction(-6, 5) and jump_adj_eta == Fraction(-4, 5)
print(f"  Exact Galois jumps verified: fund dE = {jump_fund_E}, d_eta = {jump_fund_eta}; adj dE = {jump_adj_E}, d_eta = {jump_adj_eta}")

out = {
    "kt1_derive": {
        "ibp1_diff_residual_50dps": float(ibp1_max_err),
        "ibp2_diff_residual_50dps": float(ibp2_max_err),
        "br_plus_dir_alpha_cancellation": "EXACT_ZERO_50DPS",
        "alpha_invariance_50dps": kt1_alpha_50dps,
        "Istar_tt_multiplicities_p3_to_30": {str(k): v for k, v in istar_tt_table.items()},
        "Istar_surviving_tt_p_le_30": surviving_Istar_p,
        "curved_S3_tt_eigenmode_max_rel_div_err": max_div_err,
        "curved_S3_tt_eigenmode_max_rel_eig_err": max_eig_err,
        "homogeneous_affine_redshift_identity_max_err": affine_redshift_max_err,
        "euler_arnold_p_norm_max_err": euler_arnold_norm_max_err,
    },
    "kt2_derive": {
        "palindromic_theorem_50dps": palindromic_checks,
        "round_metric_dR2_dlam_at_1_50dps": float(dR2_1),
        "round_metric_dC2_dlam_at_1_50dps": float(dC2_1),
        "round_metric_dA4_dlam_at_1_50dps": float(dA4_1),
        "exact_A4_pp_at_1": "554/15",
        "exact_A4_pp_at_1_err_50dps": float(A4_pp_1_err),
        "berger_d_G33_minus_G11_dlam_at_1_err_50dps": float(dG_aniso_err),
        "L3_p3_anisotropic_Phi_par_minus_Phi_perp": phi_aniso_L3_p3,
    },
    "kt3_derive": {
        "exact_lam1_benchmarks": kt3_exact_lam1,
        "oq4_cancelled_scaling_d_d_q_50dps": float(oq4_canc_deriv),
        "unbalanced_oq4_scaling_ratio_err_50dps": float(unbal_ratio_err),
    },
    "kt4_derive": {
        "character_orthogonality_max_err_50dps": float(max_orth_err),
        "galois_Q_invariance_max_err_50dps": float(max_galois_err),
        "exact_rational_twisted_dirac": kt4_exact,
        "galois_jumps_exact": {
            "fund_E_Dirac_rho2_minus_rho1": str(jump_fund_E),
            "adj_E_Dirac_rho2_minus_rho1": str(jump_adj_E),
            "fund_eta_D_rho2_minus_rho1": str(jump_fund_eta),
            "adj_eta_D_rho2_minus_rho1": str(jump_adj_eta),
        },
        "single_untwisted_weyl_E_exact": str(Fraction(kt4_exact["fund"]["rho_0"]["E_Weyl_exact"]) / 2),
    },
}

out_path = OUT_DIR / "01_derive.json"
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nSaved -> {out_path}")
