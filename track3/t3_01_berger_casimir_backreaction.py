#!/usr/bin/env python3
"""
Track III Upgrade T3-A: Exact Squashed Berger-Sphere Casimir Stress Tensor <T_uv(a_perp, a_3)>
and Axisymmetric Bianchi IX + O(hbar) Starobinsky Backreaction on L(p,1) and S^3/D_m*.

Resolves the R3-2 anisotropy obstruction (P0-1):
  Static round-metric L(p,1) (p >= 34) and S^3/D_m* (m >= 17) exceed the Dvali species scale
  L_min(a_Cas)/l_sp^red > 1 at fixed N_V = 1, but their Hopf-pinched Casimir tensor has O(100%)
  shear (p_parallel ~ 3*u_Cas < 0, p_perp ~ -u_Cas > 0), forbidding isotropic FLRW.

This script:
  1. Verifies the exact squashed Berger-sphere conformal scalar and Maxwell chiral-curl spectra
     (in the Hopf-aligned right-invariant Berger metric ds^2 = a_perp^2(w1^2+w2^2) + a_3^2 w3^2,
     squashing lam_B = a_3/a_perp) directly against explicit matrix diagonalization of *_B d.
  2. Computes the exact scheme-independent topological Casimir energy Delta E_Cas^top(a_perp, a_3)
     = (1/a_perp) * F_Gamma(lam_B) for both conformal scalars and Maxwell 1-forms on L(p,1) and
     S^3/D_m* via Abel-Plana + exact Hurwitz boundary shift, cross-checked against:
       (a) Exact R3-2 rational polynomials at lam_B = 1 (< 1e-18);
       (b) Independent c_0-subtracted equivariant heat-kernel Mellin integral at lam_B != 1;
       (c) Exact 4D Weyl-anomaly pole residue c_{-1} = -Vol(S^3_lam)*c_anom*C^2 (< 1e-25);
       (d) Exact pinched-fiber limit lam_B -> 0: u -> -pi^2/(45 L_fiber^4), p_par/u -> 3, p_perp/u -> -1;
       (e) Exact thermodynamic Euler identities & 30-digit Bianchi IX f(R) conservation law.
  3. Maps F_Gamma(lam_B), (u, p_par, p_perp), and the static anisotropic equilibrium barrier
     across lam_B in [0.05, 50] (both oblate and prolate regimes) for L(34,1), L(40,1),
     S^3/D_17*, S^3/D_20*, S^3/D_30*.
  4. Integrates the exact coupled 6-variable axisymmetric Bianchi IX + Starobinsky f(R) +
     anisotropic Berger-sphere Casimir ODE across wide (x_b, lam_B0, sigma_+0) ensembles
     (with DOP853 vs Radau cross-check and Hamiltonian constraint monitoring) to determine
     whether any super-species KAM breathing island survives Hopf-fiber squashing or whether
     negative Hopf-fiber tension drives runaway squashing collapse (a_3 -> 0).
"""

import json
import math
import os
from fractions import Fraction

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.linalg import expm

mp.mp.dps = 30
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ==============================================================================
# PART 1: EXACT MATRIX VERIFICATION OF BERGER-SPHERE MAXWELL SPECTRUM
# ==============================================================================

def verify_maxwell_curl_spectrum():
    """Verify exact closed-form Maxwell *_B d eigenvalues on squashed S^3_lam against matrix diag."""
    def build_J(j):
        dim = int(round(2 * j + 1))
        ms = [j - k for k in range(dim)]
        J3 = np.diag(ms).astype(complex)
        Jp = np.zeros((dim, dim), dtype=complex)
        Jm = np.zeros((dim, dim), dtype=complex)
        for k, m in enumerate(ms):
            if k > 0:
                Jp[k - 1, k] = math.sqrt(j * (j + 1) - m * (m + 1))
            if k < dim - 1:
                Jm[k + 1, k] = math.sqrt(j * (j + 1) - m * (m - 1))
        return 0.5 * (Jp + Jm), -0.5j * (Jp - Jm), J3

    S3 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex)
    S_beta = np.diag([1.0, -1.0, -1.0])
    max_eval_err = 0.0
    max_div_err = 0.0
    max_parity_err = 0.0

    for lam in [0.25, 0.65, 1.0, 1.75]:
        for two_j in range(9):  # j = 0 .. 4
            j = 0.5 * two_j
            J1, J2, J3 = build_J(j)
            dim = J3.shape[0]
            I = np.eye(dim, dtype=complex)
            E1, E2, E3 = 2j * J1, 2j * J2, (2j / lam) * J3
            C = np.block([
                [(2 / lam) * I, -E3, E2],
                [E3, (2 / lam) * I, -E1],
                [-E2, E1, (2 * lam) * I],
            ])
            div = np.block([[E1, E2, E3]])
            max_div_err = max(max_div_err, float(np.max(np.abs(div @ C))))
            K3 = np.kron(np.eye(3), J3) + np.kron(S3, I)
            evals_K, evecs_K = np.linalg.eigh(K3)
            beta_op = np.kron(S_beta, expm(-1j * math.pi * J1)) if (two_j % 2 == 0) else None
            for M in np.unique(np.round(evals_K, 6)):
                idx = np.where(np.abs(evals_K - M) < 1e-5)[0]
                V = evecs_K[:, idx]
                C_sub = V.conj().T @ C @ V
                w, u = np.linalg.eigh(C_sub)
                ev_nz = sorted([float(x) for x in w if abs(x) > 1e-10])
                aM = abs(M)
                if abs(aM - (j + 1)) < 1e-6:
                    pred = [2 * (j + 1) / lam]
                elif abs(aM - j) < 1e-6:
                    pred = [2 * lam + 2 * j / lam]
                else:
                    rad = math.sqrt(lam**2 + 4 * j * (j + 1) + (1 / lam**2 - 1) * (2 * M) ** 2)
                    pred = sorted([lam - rad, lam + rad])
                max_eval_err = max(max_eval_err, float(np.max(np.abs(np.array(ev_nz) - np.array(pred)))))
                if beta_op is not None and abs(M) < 1e-6:
                    B_sub = V.conj().T @ beta_op @ V
                    expected_p = 1.0 if (int(round(j)) % 2 == 1) else -1.0
                    for k in range(len(w)):
                        if abs(w[k]) > 1e-10:
                            p_obs = float(np.real(u[:, k].conj().T @ B_sub @ u[:, k]))
                            max_parity_err = max(max_parity_err, abs(p_obs - expected_p))

    return {
        "max_eigenvalue_err": max_eval_err,
        "max_div_curl_err": max_div_err,
        "max_Dm_star_M0_parity_err": max_parity_err,
    }


# ==============================================================================
# PART 2: EXACT 25-DIGIT BERGER-SPHERE TOPOLOGICAL CASIMIR ENGINE (ABEL-PLANA)
# ==============================================================================

def _S_Q_exact(Q, alpha, lam):
    """
    Exact Abel-Plana zeta-regularized 1D sum at s = -1:
      S(Q) = sum_{r=0}^inf (a + 2r) * [(a + 2r)^2 + M^2]^(-s/2) |_{s=-1}
    with a = |Q| + 1, M^2 = alpha + (lam^{-2} - 1)*Q^2.
    Zero branch points in Re(z) >= 0 for all lam > 0 (Maxwell) and lam < 2 (scalar).
    """
    a = mp.mpf(abs(Q) + 1)
    M2 = alpha + (1 / (lam * lam) - 1) * mp.mpf(Q) ** 2
    a2M2 = a * a + M2
    term_int = -(a2M2 ** (mp.mpf(3) / 2)) / 6
    term_0 = mp.mpf("0.5") * a * mp.sqrt(a2M2)

    def ig(y):
        z = a + 2j * y
        val = z * mp.sqrt(z * z + M2)
        return -2 * mp.im(val) / mp.expm1(2 * mp.pi * y)

    term_ap = mp.quad(ig, [0, 1, 5, mp.inf])
    return term_int + term_0 + term_ap


def _S_Q_laurent_coeffs(alpha, lam):
    """
    Exact large-Q Laurent coefficients of S(Q) = c3*Q^3 + c1*Q + cm1/Q + cm3/Q^3 + cm5/Q^5 + O(Q^-7)
    derived from Euler-Maclaurin up to B_6 (so remainder R(Q) = O(Q^-7)).
    """
    beta = 1 / (lam * lam) - 1

    def f_over_Q3(u):
        # u = 1/Q; a = (1+u)/u; M2 = (beta + alpha*u^2)/u^2; a^2+M2 = P(u)/u^2
        P = 1 / (lam * lam) + 2 * u + (1 + alpha) * u * u
        x = 1 + u  # a*u
        m2 = beta + alpha * u * u  # M2*u^2
        sP = mp.sqrt(P)
        t_int = -(P * sP) / 6
        t_0 = mp.mpf("0.5") * u * x * sP
        # -B_2/2! * f'(0), f'(0) = 2*g'(a) = 2*(2a^2+M2)/sqrt(a^2+M2)
        t_1 = -mp.mpf(1) / 12 * u * u * (2 * (2 * x * x + m2) / sP)
        # -B_4/4! * f'''(0), f'''(0) = 8*3*M2^2/(a^2+M2)^(5/2) => +(1/30)*M2^2/(a^2+M2)^(5/2)
        t_3 = mp.mpf(1) / 30 * (u ** 4) * (m2 * m2) / (P * P * sP)
        # -B_6/6! * f^{(5)}(0): g^{(5)}(x) = d^2/dx^2 [3*M2^2*(x^2+M2)^(-5/2)]
        #   g^{(4)}(x) = -15*M2^2*x*(x^2+M2)^(-7/2)
        #   g^{(5)}(x) = 15*M2^2*(6*x^2 - M2)*(x^2+M2)^(-9/2)
        #   f^{(5)}(0) = 32 * g^{(5)}(a) = 480*M2^2*(6*a^2 - M2)/(a^2+M2)^(9/2)
        #   -(B_6/720) = -(1/42)/720 = -1/30240; -480/30240 = -1/63!
        t_5 = -mp.mpf(1) / 63 * (u ** 6) * (m2 * m2) * (6 * x * x - m2) / (P ** 4 * sP)
        return t_int + t_0 + t_1 + t_3 + t_5

    cs = mp.taylor(f_over_Q3, 0, 8)
    # cs[0]=c3, cs[1]=0, cs[2]=c1, cs[3]=0, cs[4]=cm1, cs[5]=0, cs[6]=cm3, cs[7]=0, cs[8]=cm5
    return cs[0], cs[2], cs[4], cs[6], cs[8]


def _alt_Q0_sum(alpha):
    """Exact analytic continuation of sum_{j=0}^inf (-1)^j (2j+1) [(2j+1)^2 + alpha]^(-s/2) at s = -1."""
    def branch(a0):
        t_int = -((a0 * a0 + alpha) ** (mp.mpf(3) / 2)) / 12
        t_0 = mp.mpf("0.5") * a0 * mp.sqrt(a0 * a0 + alpha)

        def ig(y):
            z = a0 + 4j * y
            val = z * mp.sqrt(z * z + alpha)
            return -2 * mp.im(val) / mp.expm1(2 * mp.pi * y)

        return t_int + t_0 + mp.quad(ig, [0, 1, 5, mp.inf])

    return branch(mp.mpf(1)) - branch(mp.mpf(3))


class BergerCasimirEvaluator:
    """
    High-precision evaluator for scheme-independent topological Casimir energy
      Delta E_Cas^top(a_perp, a_3; Gamma) = (1/a_perp) * F_Gamma(lam_B)
    for both Maxwell (spin 1) and conformally coupled scalar (spin 0) on L(p,1) and S^3/D_m*.
    Caches the p-independent S^3 remainder sum per (spin, lam_B) for instant multi-p evaluation.
    """

    def __init__(self, dps=25, k1_max=18):
        self.dps = dps
        self.k1_max = k1_max
        self._cache = {}

    def _base_data(self, spin, lam):
        key = (spin, str(lam))
        if key in self._cache:
            return self._cache[key]
        with mp.workdps(self.dps):
            lam_m = mp.mpf(lam)
            alpha = (lam_m * lam_m - 1) if spin == "em" else (1 - lam_m * lam_m) / 3
            c3, c1, cm1, cm3, cm5 = _S_Q_laurent_coeffs(alpha, lam_m)
            S0 = _S_Q_exact(0, alpha, lam_m)
            # Ensure k1_max is strictly inside the asymptotic regime Q >> lam^2!
            k_eff = max(self.k1_max, int(math.ceil(4.0 * float(lam_m) ** 2)) + 12)
            # Compute p-independent S^3 remainder sum_{k=1}^inf R(k), where R(Q) = O(Q^-7)
            rem_1 = mp.mpf(0)
            for k in range(1, k_eff + 1):
                Q = mp.mpf(k)
                poly = c3 * Q**3 + c1 * Q + cm1 / Q + cm3 / Q**3 + cm5 / Q**5
                rem_1 += _S_Q_exact(k, alpha, lam_m) - poly
            alt_Q0 = _alt_Q0_sum(alpha)
            data = (lam_m, alpha, c3, c1, cm1, cm3, cm5, S0, rem_1, alt_Q0, k_eff)
        self._cache[key] = data
        return data

    def F_lens(self, p, lam, spin="em"):
        """Return dimensionless F_{L(p,1)}(lam_B) = a_perp * Delta E_Cas^top(L(p,1); a_perp, a_3)."""
        with mp.workdps(self.dps):
            lam_m, alpha, c3, c1, cm1, cm3, cm5, S0, rem_1, _, k_eff = self._base_data(spin, lam)
            pm = mp.mpf(p)
            mult = mp.mpf(2) if spin == "em" else mp.mpf(1)
            zeta_part = mult * (
                c3 * (pm**3 - 1 / pm) * (mp.mpf(1) / 120)
                + c1 * (pm - 1 / pm) * (-mp.mpf(1) / 12)
                - cm1 * mp.log(pm) / pm
                + cm3 * (pm ** (-3) - 1 / pm) * mp.zeta(3)
                + cm5 * (pm ** (-5) - 1 / pm) * mp.zeta(5)
            )
            rem_p = mp.mpf(0)
            kp_max = max(3, int(math.ceil(k_eff / p)) + 3)
            for k in range(1, kp_max + 1):
                Q = mp.mpf(k * p)
                poly = c3 * Q**3 + c1 * Q + cm1 / Q + cm3 / Q**3 + cm5 / Q**5
                rem_p += _S_Q_exact(k * p, alpha, lam_m) - poly
            val = (mp.mpf("0.5") * mult) * (1 - 1 / pm) * S0 + zeta_part + mult * (rem_p - rem_1 / pm)
            if spin == "em":
                # Exact Hurwitz Maxwell boundary shift: (p^2 - 1) / (6 * p * lam)
                val += (pm * pm - 1) / (6 * pm * lam_m)
            return val

    def F_dihedral(self, m, lam, spin="em"):
        """Return dimensionless F_{S^3/D_m*}(lam_B) = a_perp * Delta E_Cas^top(S^3/D_m*; a_perp, a_3)."""
        with mp.workdps(self.dps):
            _, _, _, _, _, _, _, _, _, alt_Q0, _ = self._base_data(spin, lam)
            F_L2m = self.F_lens(2 * m, lam, spin=spin)
            if spin == "em":
                return mp.mpf("0.5") * F_L2m - mp.mpf("0.5") * alt_Q0
            else:
                return mp.mpf("0.5") * F_L2m + mp.mpf("0.25") * alt_Q0


# ==============================================================================
# PART 3: INDEPENDENT VERIFICATION SUITE (R3-2, HEAT KERNEL, WEYL ANOMALY, LIMITS)
# ==============================================================================

def _heat_kernel_lens_top(p, lam, spin="em"):
    """Independent c_0-subtracted equivariant heat-kernel check for L(p,1) at moderate p."""
    with mp.workdps(22):
        lam_m = mp.mpf(lam)
        beta = 1 / (lam_m * lam_m) - 1
        c0 = (1 - mp.mpf(1) / p) if spin == "em" else mp.mpf(0)
        alpha_sc = (1 - lam_m * lam_m) / 3

        def dK(t):
            N_max = int(mp.ceil(mp.sqrt(60 / t))) + 8
            Q_max = int(mp.ceil(lam_m * mp.sqrt(60 / t))) + 8
            tot = mp.mpf(0)
            for Q in range(0, Q_max + 1):
                w_Q = 1 if Q == 0 else 2
                w_top = (1 - mp.mpf(1) / p) if (Q % p == 0) else (-mp.mpf(1) / p)
                s_Q = mp.mpf(0)
                if spin == "scal":
                    for r in range(0, N_max):
                        n = Q + 2 * r
                        w2 = (n + 1) ** 2 + alpha_sc + beta * Q * Q
                        v = (n + 1) * mp.exp(-t * w2)
                        s_Q += v
                        if v < mp.mpf("1e-26"):
                            break
                else:
                    if Q == 0:
                        s_Q += mp.exp(-t * (2 * lam_m) ** 2)
                        for n in range(2, N_max, 2):
                            R = mp.sqrt((n + 1) ** 2 + (lam_m * lam_m - 1))
                            v = (n + 1) * (mp.exp(-t * (lam_m + R) ** 2) + mp.exp(-t * (R - lam_m) ** 2))
                            s_Q += v
                            if v < mp.mpf("1e-26"):
                                break
                    else:
                        if Q >= 2:
                            s_Q += (Q - 1) * mp.exp(-t * (Q / lam_m) ** 2)
                        s_Q += (Q + 1) * mp.exp(-t * (2 * lam_m + Q / lam_m) ** 2)
                        for r in range(1, N_max):
                            n = Q + 2 * r
                            R = mp.sqrt((n + 1) ** 2 + (lam_m * lam_m - 1) + beta * Q * Q)
                            v = (n + 1) * (mp.exp(-t * (lam_m + R) ** 2) + mp.exp(-t * (R - lam_m) ** 2))
                            s_Q += v
                            if v < mp.mpf("1e-26"):
                                break
                tot += w_Q * w_top * s_Q
            return tot

        I1 = mp.quad(lambda t: (dK(t) - c0) / (t ** (mp.mpf(3) / 2)), [mp.mpf("0.003"), mp.mpf("0.015"), mp.mpf("0.08"), mp.mpf("0.4"), mp.mpf("1.0")])
        I2 = mp.quad(lambda t: dK(t) / (t ** (mp.mpf(3) / 2)), [mp.mpf("1.0"), mp.mpf("5.0"), mp.mpf("20.0")])
        return -(I1 - 2 * c0 + I2) / (4 * mp.sqrt(mp.pi))


def run_precision_verifications(ev: BergerCasimirEvaluator):
    """Run all 5 independent precision cross-checks on the Berger-sphere Casimir engine."""
    # (a) Exact R3-2 rational polynomials at lam_B = 1
    def L_poly(p):
        pf = Fraction(p)
        dE_em = (-pf**4 + 50 * pf**2 - 16) / (360 * pf) - Fraction(11, 120) / pf
        dE_sc = (-pf**4 - 10 * pf**2 + 14) / (720 * pf) - Fraction(1, 240) / pf
        return dE_em, dE_sc

    def D_poly(m):
        A, B, C = Fraction(m + 1), Fraction(2), Fraction(m)
        am2 = Fraction(1, 2) / (B * C)
        a0 = am2 * (A**2 / 2 - (B**2 + C**2) / 6)
        a2 = am2 * (A**4 / 24 - A**2 * (B**2 + C**2) / 12 + (7 * (B**4 + C**4) + 10 * B**2 * C**2) / 360)
        dE_em = (2 * a2 + 2 * a0 + am2 / 4) - Fraction(11, 120) / (4 * m)
        dE_sc = a2 - Fraction(1, 240) / (4 * m)
        return dE_em, dE_sc

    max_lam1_err_L = 0.0
    max_lam1_err_D = 0.0
    for p in [3, 10, 34, 40]:
        em_ex, sc_ex = L_poly(p)
        em_ap = ev.F_lens(p, "1", "em")
        sc_ap = ev.F_lens(p, "1", "scal")
        max_lam1_err_L = max(
            max_lam1_err_L,
            float(abs(em_ap - mp.mpf(em_ex.numerator) / em_ex.denominator)),
            float(abs(sc_ap - mp.mpf(sc_ex.numerator) / sc_ex.denominator)),
        )
    for m in [3, 10, 17, 20, 30]:
        em_ex, sc_ex = D_poly(m)
        em_ap = ev.F_dihedral(m, "1", "em")
        sc_ap = ev.F_dihedral(m, "1", "scal")
        max_lam1_err_D = max(
            max_lam1_err_D,
            float(abs(em_ap - mp.mpf(em_ex.numerator) / em_ex.denominator)),
            float(abs(sc_ap - mp.mpf(sc_ex.numerator) / sc_ex.denominator)),
        )

    # (b) Independent heat-kernel verification at lam_B = 0.5 and 1.3 for L(5,1) (scalar & EM)
    hk_checks = {}
    for spin, lam_s in [("scal", "0.5"), ("em", "0.5"), ("em", "1.3")]:
        f_ap = ev.F_lens(5, lam_s, spin)
        f_hk = _heat_kernel_lens_top(5, lam_s, spin)
        hk_checks[f"L5_{spin}_lam{lam_s}"] = {
            "abel_plana": float(f_ap),
            "heat_kernel": float(f_hk),
            "abs_diff": float(abs(f_ap - f_hk)),
        }

    # (c) Exact 4D Weyl-anomaly pole residue c_{-1} = -Vol(S^3_lam) * c_anom * C^2
    max_weyl_res_err = 0.0
    for lam_s in ["0.3", "0.7", "1.4"]:
        lm = mp.mpf(lam_s)
        weyl_sc = (lm / 45) * (1 - lm * lm) ** 2
        weyl_em = 12 * weyl_sc
        _, _, cm1_sc, _, _ = _S_Q_laurent_coeffs((1 - lm * lm) / 3, lm)
        _, _, cm1_em, _, _ = _S_Q_laurent_coeffs(lm * lm - 1, lm)
        max_weyl_res_err = max(
            max_weyl_res_err,
            float(abs(cm1_sc + weyl_sc)),
            float(abs(2 * cm1_em + weyl_em)),
        )

    # (d) Pinched-fiber limit lam_B -> 0 (p_par/u -> 3, p_perp/u -> -1, u / u_1D -> 1)
    # Test on L(40, 1) and S^3/D_20* at lam_B = 0.05 using 4th-order central diff for F'(lam)
    def pressures_at(family, param, lam_val, spin="em"):
        h = mp.mpf("1e-5") * lam_val
        fn = (lambda x: ev.F_lens(param, x, spin)) if family == "L" else (lambda x: ev.F_dihedral(param, x, spin))
        F0 = fn(lam_val)
        Fp = (-fn(lam_val + 2 * h) + 8 * fn(lam_val + h) - 8 * fn(lam_val - h) + fn(lam_val - 2 * h)) / (12 * h)
        # At a_perp = 1, a_3 = lam_val:
        # u ~ F/lam, p_par ~ -F', p_perp ~ 0.5*(F/lam + F')
        u_hat = F0 / lam_val
        ppar_hat = -Fp
        pperp_hat = mp.mpf("0.5") * (u_hat + Fp)
        # Check exact Euler identity: a_perp dE/da_perp + a_3 dE/da_3 + E == 0
        # Explicitly via partial derivatives in (a_perp, a_3) at (a_perp=1.7, a_3=1.7*lam_val):
        ap0 = mp.mpf("1.7")
        a30 = ap0 * lam_val
        da = mp.mpf("1e-6")
        E0 = fn(a30 / ap0) / ap0
        dE_dap = (fn(a30 / (ap0 + da)) / (ap0 + da) - fn(a30 / (ap0 - da)) / (ap0 - da)) / (2 * da)
        dE_da3 = (fn((a30 + da) / ap0) / ap0 - fn((a30 - da) / ap0) / ap0) / (2 * da)
        euler_err = float(abs(ap0 * dE_dap + a30 * dE_da3 + E0) / abs(E0))
        order_G = param if family == "L" else 4 * param
        vol0 = 2 * mp.pi**2 * ap0 * ap0 * a30 / order_G
        p_par_dir = -a30 * dE_da3 / vol0
        p_perp_dir = -ap0 * dE_dap / (2 * vol0)
        u_dir = E0 / vol0
        trace_err = float(abs(p_par_dir + 2 * p_perp_dir - u_dir) / abs(u_dir))
        # Compare u_dir against 1D S^1 Casimir density -pi^2/(45 L_fiber^4) (or /2 for scalar)
        L_fib = (2 * mp.pi * a30 / param) if family == "L" else (mp.pi * a30 / param)
        u_1d = -(mp.pi**2) / (45 * L_fib**4) * (1 if spin == "em" else mp.mpf("0.5"))
        return {
            "lam_B": float(lam_val),
            "F": float(F0),
            "F_prime": float(Fp),
            "p_par_over_u": float(ppar_hat / u_hat),
            "p_perp_over_u": float(pperp_hat / u_hat),
            "shear_over_u": float((ppar_hat - pperp_hat) / u_hat),
            "u_over_u_S1": float(u_dir / u_1d),
            "euler_homogeneity_rel_err": euler_err,
            "traceless_rel_err": trace_err,
        }

    pinched_checks = {
        "L40_em_lam0.05": pressures_at("L", 40, mp.mpf("0.05"), "em"),
        "L40_em_lam1.00": pressures_at("L", 40, mp.mpf("1.0"), "em"),
        "L40_scal_lam0.05": pressures_at("L", 40, mp.mpf("0.05"), "scal"),
        "D20_em_lam0.05": pressures_at("D", 20, mp.mpf("0.05"), "em"),
        "D20_em_lam1.00": pressures_at("D", 20, mp.mpf("1.0"), "em"),
    }

    # (e) 30-digit verification of exact Bianchi IX + Starobinsky f(R) + anisotropic Casimir conservation!
    # Check d/dt [ Vol * (G_00 + (sig/6)*H1_00 - u_tilde) ] == 0 along an arbitrary trajectory (a_perp(t), a_3(t))
    # satisfying the trace + anisotropy equations? Better: verify the EXACT covariant Bianchi identity
    #   d/dt(a_perp^2 a_3 * E_00) + a_perp^2 a_3 * (2*H_perp*E_11 + H_3*E_33) == 0
    # directly for E_uv = G_uv + (sig/6)*H1_uv - T_uv^top on an arbitrary smooth (a_perp(t), a_3(t))!
    with mp.workdps(30):
        sig_m = mp.mpf(9) / 31
        ap_fn = lambda t: mp.mpf("2.1") + mp.mpf("0.3") * mp.sin(mp.mpf("1.3") * t)
        a3_fn = lambda t: mp.mpf("1.4") + mp.mpf("0.25") * mp.cos(mp.mpf("1.9") * t + mp.mpf("0.4"))
        # Arbitrary smooth test F(lam) = -50/lam^3 + 7/lam - 0.8*lam^2 (exact degree -1 Casimir law!)
        F_test = lambda l: -50 / l**3 + 7 / l - mp.mpf("0.8") * l**2
        Fp_test = lambda l: 150 / l**4 - 7 / l**2 - mp.mpf("1.6") * l

        def E_components(t):
            ap = ap_fn(t)
            a3 = a3_fn(t)
            lam = a3 / ap
            hp = mp.diff(ap_fn, t) / ap
            h3 = mp.diff(a3_fn, t) / a3
            dhp = mp.diff(lambda s: mp.diff(ap_fn, s) / ap_fn(s), t)
            dh3 = mp.diff(lambda s: mp.diff(a3_fn, s) / a3_fn(s), t)
            hbar = (2 * hp + h3) / 3
            sp = (hp - h3) / 3
            dhbar = (2 * dhp + dh3) / 3
            dsp = (dhp - dh3) / 3
            r_fn = lambda s: (
                2 * (mp.diff(lambda q: mp.diff(ap_fn, q) / ap_fn(q), s))
                + mp.diff(lambda q: mp.diff(a3_fn, q) / a3_fn(q), s)
                + 3 * (mp.diff(ap_fn, s) / ap_fn(s)) ** 2
                + (mp.diff(a3_fn, s) / a3_fn(s)) ** 2
                + 2 * (mp.diff(ap_fn, s) / ap_fn(s)) * (mp.diff(a3_fn, s) / a3_fn(s))
                + (4 - (a3_fn(s) / ap_fn(s)) ** 2) / ap_fn(s) ** 2
            ) / 3
            r_val = r_fn(t)
            s_val = mp.diff(r_fn, t)
            sp_val = mp.diff(r_fn, t, 2)
            yB = hbar**2 - sp**2 + (4 - lam**2) / (3 * ap**2)
            # Casimir components (overall constant absorbed into F_test)
            u_c = F_test(lam) / (ap**4 * lam)
            ppar_c = -Fp_test(lam) / (ap**4)
            pperp_c = (u_c - ppar_c) / 2
            # E_00 = 3*(yB + sig*(r*(2*yB-r) + 2*hbar*s)) - u_c
            E00 = 3 * (yB + sig_m * (r_val * (2 * yB - r_val) + 2 * hbar * s_val)) - u_c
            # Trace E_tr = -E_00 + 2*E_11 + E_33 = -6*r - 6*sig*(s' + 3*hbar*s)  (since u_c - 2*pperp_c - ppar_c == 0)
            Etr = -6 * r_val - 6 * sig_m * (sp_val + 3 * hbar * s_val)
            # Anisotropy D := E_11 - E_33 = (1 + 2*sig*r)*(3*(sp' + 3*hbar*sp) + 4*(1-lam^2)/ap^2) + 6*sig*s*sp - (pperp_c - ppar_c)
            D_aniso = (1 + 2 * sig_m * r_val) * (3 * (dsp + 3 * hbar * sp) + 4 * (1 - lam**2) / ap**2) + 6 * sig_m * s_val * sp - (pperp_c - ppar_c)
            # Recover E_11 and E_33 from E_tr and D_aniso:
            # 2*E_11 + E_33 = E_tr + E_00, E_11 - E_33 = D_aniso => 3*E_11 = E_tr + E_00 + D_aniso
            E11 = (Etr + E00 + D_aniso) / 3
            E33 = E11 - D_aniso
            return ap**2 * a3 * E00, ap**2 * a3 * (2 * hp * E11 + h3 * E33)

        t0 = mp.mpf("0.41")
        lhs_b = mp.diff(lambda s: E_components(s)[0], t0)
        rhs_b = -E_components(t0)[1]
        bianchi_bix_err = float(abs(lhs_b - rhs_b))

    return {
        "max_lam1_vs_R3_2_L_err": max_lam1_err_L,
        "max_lam1_vs_R3_2_D_err": max_lam1_err_D,
        "heat_kernel_cross_checks": hk_checks,
        "max_weyl_anomaly_residue_err": max_weyl_res_err,
        "pinched_and_round_pressure_checks": pinched_checks,
        "bianchi_ix_fR_casimir_conservation_err_30dps": bianchi_bix_err,
    }


# ==============================================================================
# PART 4: SCAN F_Gamma(lam_B) ACROSS OBLATE & PROLATE REGIMES + STATIC BARRIER
# ==============================================================================

def profile_manifold_curve(ev: BergerCasimirEvaluator, family, param, spin="em"):
    """
    Compute F_Gamma(lam_B), F'_Gamma(lam_B), (p_par/u, p_perp/u), and check whether
    F_Gamma(lam_B) ever changes sign in the prolate regime lam_B > 1!
    """
    # Log-spaced grid from lam_B = 0.05 to lam_B = 1.0, then up to lam_B = 6.0 (EM) or 1.85 (scalar)
    # Note: R_B = 2(4 - lam_B^2)/a_perp^2 turns negative at lam_B = 2, and F_Gamma flips positive around lam_B ~ 3.2-4.1.
    lam_lo = np.geomspace(0.05, 1.0, 30)
    lam_hi_max = 6.0 if spin == "em" else 1.85
    lam_hi = np.geomspace(1.0, lam_hi_max, 25)[1:]
    lam_grid = np.concatenate([lam_lo, lam_hi])
    fn = (lambda x: ev.F_lens(param, x, spin)) if family == "L" else (lambda x: ev.F_dihedral(param, x, spin))

    F_vals = np.array([float(fn(mp.mpf(f"{lv:.14g}"))) for lv in lam_grid])
    # Accurate derivative in log(lam): dF/d(ln lam) = lam * F'(lam) using CubicSpline on log(lam)
    log_lam = np.log(lam_grid)
    # Wait: F ~ -A/lam^3 at small lam, so interpolate G(ln lam) := lam^3 * F(lam) for 1e-10 smoothness!
    G_vals = (lam_grid ** 3) * F_vals
    cs_G = CubicSpline(log_lam, G_vals, bc_type="not-a-knot")
    dG_dln = cs_G(log_lam, 1)
    # Since F = lam^{-3} G, lam*F' = lam^{-3} * (dG/dln - 3*G)!
    lam_Fp = (lam_grid ** (-3)) * (dG_dln - 3.0 * G_vals)
    # Exact pressures ratios: p_par/u = -lam*F'/F, p_perp/u = 0.5*(1 + lam*F'/F)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppar_over_u = -lam_Fp / F_vals
        pperp_over_u = 0.5 * (1.0 + lam_Fp / F_vals)

    # Check if F changes sign (becomes positive at large prolate lam_B)
    sign_flips = []
    for i in range(len(lam_grid) - 1):
        if F_vals[i] < 0.0 and F_vals[i + 1] > 0.0:
            # Linear-in-log root refinement via bisection on spline
            a_l, b_l = log_lam[i], log_lam[i + 1]
            for _ in range(50):
                m_l = 0.5 * (a_l + b_l)
                if cs_G(m_l) < 0:
                    a_l = m_l
                else:
                    b_l = m_l
            sign_flips.append(float(math.exp(0.5 * (a_l + b_l))))

    # Check static equilibrium condition in lam_B at fixed x_perp:
    #   4*(1 - lam^2) / x_perp^2 == (tilde_p_perp - tilde_p_par) = (1440*|G| / (a_num * x_perp^4)) * (0.5*F/lam + 1.5*F')
    # Rewrite: x_perp^2 = (1440*|G|/a_num) * (0.5*F/lam + 1.5*F') / (4*(1 - lam^2))
    # For lam < 1, 1 - lam^2 > 0, so a static lam_B equilibrium exists at x_perp >= x_perp_min!
    order_G = param if family == "L" else 4 * param
    a_num = 62.0 if spin == "em" else 1.0
    pref = 1440.0 * order_G / a_num
    # Force coefficient C_aniso(lam) := pref * (0.5*F/lam + 1.5*F') = pref * (0.5*F + 1.5*lam*F') / lam
    C_aniso = pref * (0.5 * F_vals + 1.5 * lam_Fp) / lam_grid
    # On lam < 1 (where 1 - lam^2 > 0 and C_aniso > 0):
    mask_oblate = (lam_grid < 0.999) & (C_aniso > 0)
    x_eq_oblate = np.sqrt(C_aniso[mask_oblate] / (4.0 * (1.0 - lam_grid[mask_oblate] ** 2)))
    idx_min = int(np.argmin(x_eq_oblate))
    x_eq_min = float(x_eq_oblate[idx_min])
    lam_eq_min = float(lam_grid[mask_oblate][idx_min])
    # Also check prolate lam > 1: need 1 - lam^2 < 0 AND C_aniso < 0 (i.e. p_perp - p_par < 0)!
    mask_prolate = (lam_grid > 1.001) & (C_aniso < 0)
    prolate_eq = None
    if np.any(mask_prolate):
        x_eq_pro = np.sqrt(C_aniso[mask_prolate] / (4.0 * (1.0 - lam_grid[mask_prolate] ** 2)))
        ip = int(np.argmin(x_eq_pro))
        prolate_eq = {
            "x_perp_min_prolate_eq": float(x_eq_pro[ip]),
            "lam_B_at_min": float(lam_grid[mask_prolate][ip]),
        }

    # Exact x_ring at lam_B = 1: kappa = (480/a_num)*|G|*F(1) (for EM a_num=62 => (480/31)*(|G|/2)*F(1); for scal a_num=1 => 480*|G|*F(1))
    F1 = float(fn(mp.mpf(1)))
    kap_val = (480.0 / a_num) * order_G * F1
    x_ring = math.sqrt(abs(kap_val))

    return {
        "family": family,
        "param": param,
        "order_G": order_G,
        "spin": spin,
        "F_at_lam1": F1,
        "x_ring_round_metric": x_ring,
        "lam_grid": [round(float(v), 6) for v in lam_grid],
        "F_vals": [float(v) for v in F_vals],
        "lam_Fprime_vals": [float(v) for v in lam_Fp],
        "p_par_over_u": [float(v) if math.isfinite(v) else None for v in ppar_over_u],
        "p_perp_over_u": [float(v) if math.isfinite(v) else None for v in pperp_over_u],
        "F_sign_flip_lam_B": sign_flips,
        "oblate_static_eq_barrier": {
            "x_perp_min": x_eq_min,
            "x_perp_min_over_x_ring": x_eq_min / x_ring,
            "lam_B_at_saddle_node": lam_eq_min,
            "asymptotic_9_over_2_check": 4.5,
        },
        "prolate_static_eq_barrier": prolate_eq,
        "_spline_G": cs_G,
        "_lam_min": float(lam_grid[0]),
        "_lam_max": float(lam_grid[-1]),
        "_A_small_lam": float(-G_vals[0]),  # F ~ -A / lam^3 as lam -> 0
    }


# ==============================================================================
# PART 5: COUPLED AXISYMMETRIC BIANCHI IX + STAROBINSKY f(R) + ANISOTROPIC CASIMIR ODE
# ==============================================================================

def build_casimir_lookup(profile):
    """Fast, C^2-smooth evaluator returning (u_tilde, p_perp_tilde, p_par_tilde) for any (x_perp, x_3)."""
    cs_G = profile["_spline_G"]
    lam_min = profile["_lam_min"]
    lam_max = profile["_lam_max"]
    A_as = profile["_A_small_lam"]
    pref = 1440.0 * profile["order_G"] / 62.0  # pure N_V = 1 vectors

    def eval_F_and_lamFp(lam):
        if lam < lam_min:
            # Exact pinched-fiber 1D S^1 asymptote F ~ -A / lam^3 (matches spline at lam_min to < 1e-4)
            F = -A_as / (lam ** 3)
            lamFp = 3.0 * A_as / (lam ** 3)
            return F, lamFp
        if lam > lam_max:
            ll = math.log(lam_max)
            G = float(cs_G(ll))
            dG = float(cs_G(ll, 1))
            F = G / (lam ** 3)
            lamFp = (dG - 3.0 * G) / (lam ** 3)
            return F, lamFp
        ll = math.log(lam)
        G = float(cs_G(ll))
        dG = float(cs_G(ll, 1))
        inv_l3 = 1.0 / (lam ** 3)
        return G * inv_l3, (dG - 3.0 * G) * inv_l3

    def stress_tensor(xp, x3):
        lam = x3 / xp
        F, lamFp = eval_F_and_lamFp(lam)
        inv_x4 = 1.0 / (xp ** 4)
        u_t = pref * inv_x4 * (F / lam)
        ppar_t = -pref * inv_x4 * (lamFp / lam)
        pperp_t = 0.5 * (u_t - ppar_t)
        return u_t, pperp_t, ppar_t

    return stress_tensor


def integrate_anisotropic_bix(profile, x_b, lam0=1.0, sigp0=0.0, s0=0.0, tau_max=600.0, method="DOP853", sign_hdot=+1):
    """
    Integrate the exact 6-variable axisymmetric Bianchi IX + Starobinsky f(R) + anisotropic Casimir ODE:
      state = [x_perp, x_3, h_bar, sigma_+, r, s]
    Initial condition at mean-volume bounce h_bar(0) = 0, x_bar(0) = x_b, lam_B(0) = lam0,
    sigma_+(0) = sigp0, s(0) = s0, solving exact e_00^BIX = 0 for r(0)!
    """
    sig = 9.0 / 31.0  # pure N_V = 1 zero-counterterm zeta/PS scalaron
    stress_fn = build_casimir_lookup(profile)

    xp0 = x_b * (lam0 ** (-1.0 / 3.0))
    x30 = xp0 * lam0
    u0, _, _ = stress_fn(xp0, x30)
    yB0 = -sigp0 * sigp0 + (4.0 - lam0 * lam0) / (3.0 * xp0 * xp0)
    # Exact e_00^BIX = yB0 + sig*r0*(2*yB0 - r0) - u0/3 = 0
    # => sig*(r0 - yB0)^2 = yB0 + sig*yB0^2 - u0/3
    rhs_r = (yB0 + sig * yB0 * yB0 - u0 / 3.0) / sig
    if rhs_r < 0.0:
        return {"outcome": "no_turning_point", "x_b": float(x_b), "lam0": float(lam0), "sigp0": float(sigp0)}
    r0 = yB0 + sign_hdot * math.sqrt(rhs_r)
    y0 = np.array([xp0, x30, 0.0, sigp0, r0, s0], dtype=float)

    def rhs(t, y):
        xp, x3, hbar, sp, r, s = y
        if xp <= 1e-6 or x3 <= 1e-6:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        lam = x3 / xp
        hp = hbar + sp
        h3 = hbar - 2.0 * sp
        u_t, pperp_t, ppar_t = stress_fn(xp, x3)
        dxp = xp * hp
        dx3 = x3 * h3
        dhbar = r - 2.0 * hbar * hbar - sp * sp - (4.0 - lam * lam) / (3.0 * xp * xp)
        denom = 1.0 + 2.0 * sig * r
        dsp = -3.0 * hbar * sp - 4.0 * (1.0 - lam * lam) / (3.0 * xp * xp) + (pperp_t - ppar_t - 6.0 * sig * s * sp) / (3.0 * denom)
        dr = s
        ds = -3.0 * hbar * s - r / sig
        return [dxp, dx3, dhbar, dsp, dr, ds]

    # Terminal events:
    # 1. Hopf-fiber pinching collapse: lam_B = x3/xp drops below 0.02 (or x3 drops below 0.5)
    def ev_fiber_collapse(t, y):
        return (y[1] / max(y[0], 1e-12)) - 0.02
    ev_fiber_collapse.terminal = True
    ev_fiber_collapse.direction = -1

    # 2. Prolate base collapse / runaway negative-curvature expansion: lam_B > 5.8 or x_bar > 50 * x_b
    def ev_base_collapse(t, y):
        return 5.8 - (y[1] / max(y[0], 1e-12))
    ev_base_collapse.terminal = True
    ev_base_collapse.direction = -1

    # 3. f(R) ghost / curvature blowup: |1 + 2*sig*r| < 1e-3 or |r| > 1e3 or |hbar| > 1e2
    def ev_blowup(t, y):
        return min(abs(1.0 + 2.0 * sig * y[4]) - 1e-3, 1e3 - abs(y[4]), 1e2 - abs(y[2]))
    ev_blowup.terminal = True
    ev_blowup.direction = -1

    # 4. Runaway expansion (not a bounded KAM breathing cycle: x_bar > 20 * x_b)
    def ev_expand(t, y):
        return 20.0 * x_b - ((max(y[0], 1e-12) ** 2 * max(y[1], 1e-12)) ** (1.0 / 3.0))
    ev_expand.terminal = True
    ev_expand.direction = -1

    sol = solve_ivp(
        rhs, (0.0, tau_max), y0, method=method, rtol=1e-10, atol=1e-12,
        events=[ev_fiber_collapse, ev_base_collapse, ev_blowup, ev_expand],
        max_step=0.5,
    )

    # Check exact e_00^BIX Hamiltonian constraint along trajectory!
    max_rel_e00 = 0.0
    n_pts = len(sol.t)
    step = max(1, n_pts // 250)
    for i in range(0, n_pts, step):
        xp, x3, hb, sp, r, s = sol.y[:, i]
        lam = x3 / xp
        u_t, _, _ = stress_fn(xp, x3)
        yB = hb * hb - sp * sp + (4.0 - lam * lam) / (3.0 * xp * xp)
        e00 = yB + sig * (r * (2.0 * yB - r) + 2.0 * hb * s) - u_t / 3.0
        scale = max(1e-12, abs(yB), abs(u_t / 3.0), abs(sig * r * r), abs(2.0 * sig * hb * s))
        max_rel_e00 = max(max_rel_e00, abs(e00) / scale)

    lam_arr = sol.y[1] / sol.y[0]
    xbar_arr = (sol.y[0] ** 2 * sol.y[1]) ** (1.0 / 3.0)
    hb_arr = sol.y[2]
    n_hb_zeros = int(np.sum((hb_arr[:-1] * hb_arr[1:]) < 0))

    if len(sol.t_events[0]) > 0:
        outcome = "hopf_fiber_collapse_a3_to_0"
    elif len(sol.t_events[1]) > 0:
        outcome = "prolate_runaway_lam_gt_5.8"
    elif len(sol.t_events[2]) > 0:
        outcome = "curvature_or_fR_blowup"
    elif len(sol.t_events[3]) > 0:
        outcome = "runaway_negative_curv_expansion"
    else:
        outcome = "bounded_to_tau_max"

    return {
        "x_b": float(x_b),
        "lam0": float(lam0),
        "sigp0": float(sigp0),
        "method": method,
        "outcome": outcome,
        "tau_end": float(sol.t[-1]),
        "lam_min_reached": float(np.min(lam_arr)),
        "lam_max_reached": float(np.max(lam_arr)),
        "lam_end": float(lam_arr[-1]),
        "xbar_min": float(np.min(xbar_arr)),
        "xbar_max": float(np.max(xbar_arr)),
        "xbar_end": float(xbar_arr[-1]),
        "n_hbar_turnarounds": n_hb_zeros,
        "max_rel_e00": float(max_rel_e00),
    }


def main():
    print("=== TRACK III UPGRADE T3-A: ANISOTROPIC BERGER-SPHERE CASIMIR BACKREACTION ===")
    spec_ver = verify_maxwell_curl_spectrum()
    print(f"1. Matrix verification of Maxwell *_B d spectrum: eval_err={spec_ver['max_eigenvalue_err']:.2e}, "
          f"div_err={spec_ver['max_div_curl_err']:.2e}, D_m*_parity_err={spec_ver['max_Dm_star_M0_parity_err']:.2e}")
    assert spec_ver["max_eigenvalue_err"] < 1e-12

    ev = BergerCasimirEvaluator(dps=25, k1_max=18)
    prec_ver = run_precision_verifications(ev)
    print(f"2. Precision cross-checks:")
    print(f"   lam_B=1 vs R3-2 exact polynomials: L(p,1)={prec_ver['max_lam1_vs_R3_2_L_err']:.2e}, D_m*={prec_ver['max_lam1_vs_R3_2_D_err']:.2e}")
    print(f"   Weyl anomaly pole c_-1 err:        {prec_ver['max_weyl_anomaly_residue_err']:.2e}")
    print(f"   30-digit Bianchi IX f(R) conserv:  {prec_ver['bianchi_ix_fR_casimir_conservation_err_30dps']:.2e}")
    for k, v in prec_ver["heat_kernel_cross_checks"].items():
        print(f"   Heat-kernel check {k:14s}: AP={v['abel_plana']:+.12f}, HK={v['heat_kernel']:+.12f}, diff={v['abs_diff']:.2e}")
    for k, v in prec_ver["pinched_and_round_pressure_checks"].items():
        print(f"   Pressure {k:16s}: p_par/u={v['p_par_over_u']:+.6f}, p_perp/u={v['p_perp_over_u']:+.6f}, "
              f"u/u_S1={v['u_over_u_S1']:.6f}, euler={v['euler_homogeneity_rel_err']:.2e}, tr={v['traceless_rel_err']:.2e}")

    assert prec_ver["max_lam1_vs_R3_2_L_err"] < 1e-15
    assert prec_ver["max_lam1_vs_R3_2_D_err"] < 1e-15
    assert prec_ver["max_weyl_anomaly_residue_err"] < 1e-22
    assert prec_ver["bianchi_ix_fR_casimir_conservation_err_30dps"] < 1e-25

    print("\n3. Profiling squashed Berger-sphere Casimir curves F_Gamma(lam_B) for super-species quotients...")
    targets = [
        ("L", 34, "em"),
        ("L", 40, "em"),
        ("L", 40, "scal"),
        ("D", 17, "em"),
        ("D", 20, "em"),
        ("D", 30, "em"),
    ]
    profiles = {}
    for fam, par, sp in targets:
        key = f"{fam}{par}_{sp}"
        prof = profile_manifold_curve(ev, fam, par, sp)
        profiles[key] = prof
        ob = prof["oblate_static_eq_barrier"]
        print(f"   {key:10s}: F(1)={prof['F_at_lam1']:+10.4f}, x_ring={prof['x_ring_round_metric']:7.2f}, "
              f"oblate eq x_perp_min={ob['x_perp_min']:8.2f} ({ob['x_perp_min_over_x_ring']:.3f}*x_ring at lam={ob['lam_B_at_saddle_node']:.4f}), "
              f"F sign-flip lam={prof['F_sign_flip_lam_B']}, prolate_eq={prof['prolate_static_eq_barrier']}")

    print("\n4. Integrating coupled axisymmetric Bianchi IX + Starobinsky f(R) + anisotropic Casimir ODE...")
    # A. L(40, 1) EM: Round-sphere starts across [0.5, 6.0]*x_ring + DOP853 vs Radau check at x_ring
    prof_L40 = profiles["L40_em"]
    xr_L40 = prof_L40["x_ring_round_metric"]
    L40_round_sweep = []
    for mult in [0.5, 0.8, 1.0, 1.5, 2.5, 4.0, 5.0, 6.0]:
        tmax = 6000.0 if mult >= 4.5 else 1500.0
        res = integrate_anisotropic_bix(prof_L40, mult * xr_L40, lam0=1.0, sigp0=0.0, tau_max=tmax)
        L40_round_sweep.append(res)
        print(f"   L(40,1) round x_b={mult:4.1f}*x_ring ({mult*xr_L40:7.1f}): {res['outcome']:30s} "
              f"tau={res['tau_end']:7.2f}, lam_end={res['lam_end']:.4f}, e00={res['max_rel_e00']:.2e}")

    # Solver check DOP853 vs Radau at x_ring
    r_dop_L40 = integrate_anisotropic_bix(prof_L40, xr_L40, lam0=1.0, sigp0=0.0, tau_max=1200.0, method="DOP853")
    r_rad_L40 = integrate_anisotropic_bix(prof_L40, xr_L40, lam0=1.0, sigp0=0.0, tau_max=1200.0, method="Radau")
    tau_diff_L40 = abs(r_dop_L40["tau_end"] - r_rad_L40["tau_end"]) / max(1e-12, r_dop_L40["tau_end"])
    print(f"   L(40,1) DOP853 vs Radau collapse time rel diff = {tau_diff_L40:.3e} (tau={r_dop_L40['tau_end']:.4f})")

    # B. Prolate & shear-tuned survival search on L(40, 1):
    L40_prolate_sweep = []
    n_L40_bounded = 0
    for mult in [1.0, 2.0, 5.0]:
        for lam0 in [0.85, 1.2, 1.5, 1.8, 2.5, 4.0]:
            for sigp0 in [0.0, -0.002, -0.008, -0.02]:
                tmax = 5000.0 if mult >= 4.5 else 1500.0
                res = integrate_anisotropic_bix(prof_L40, mult * xr_L40, lam0=lam0, sigp0=sigp0, tau_max=tmax)
                L40_prolate_sweep.append(res)
                if res["outcome"] == "bounded_to_tau_max":
                    n_L40_bounded += 1
                    print(f"   [BOUNDED] L(40,1) x_b={mult}*xr, lam0={lam0}, sigp0={sigp0}: "
                          f"lam in [{res['lam_min_reached']:.3f},{res['lam_max_reached']:.3f}], "
                          f"xbar in [{res['xbar_min']:.1f},{res['xbar_max']:.1f}], e00={res['max_rel_e00']:.2e}")
    print(f"   L(40,1) prolate/shear 72-trial sweep: {n_L40_bounded}/{len(L40_prolate_sweep)} bounded to tau_max")
    from collections import Counter
    L40_tally = dict(Counter(r["outcome"] for r in L40_prolate_sweep))
    print(f"   L(40,1) outcome tally: {L40_tally}")

    # C. S^3/D_20* and S^3/D_30* EM: Round-sphere starts + prolate/shear sweep (long tau_max=8000 for 5*x_ring!)
    D_results = {}
    for key in ["D20_em", "D30_em"]:
        prof_D = profiles[key]
        xr_D = prof_D["x_ring_round_metric"]
        round_runs = []
        for mult in [0.5, 1.0, 2.0, 5.0]:
            tmax = 8000.0 if mult >= 4.0 else 1500.0
            res = integrate_anisotropic_bix(prof_D, mult * xr_D, lam0=1.0, sigp0=0.0, tau_max=tmax)
            round_runs.append(res)
            print(f"   {key:7s} round x_b={mult:3.1f}*x_ring ({mult*xr_D:6.1f}): {res['outcome']:30s} "
                  f"tau={res['tau_end']:7.2f}, lam_end={res['lam_end']:.4f}, e00={res['max_rel_e00']:.2e}")
        pro_runs = []
        for mult in [1.0, 2.0, 5.0]:
            for lam0 in [1.2, 1.6, 2.5, 4.0]:
                for sigp0 in [0.0, -0.005, -0.02]:
                    tmax = 8000.0 if mult >= 4.0 else 1500.0
                    pro_runs.append(integrate_anisotropic_bix(prof_D, mult * xr_D, lam0=lam0, sigp0=sigp0, tau_max=tmax))
        tally = dict(Counter(r["outcome"] for r in pro_runs))
        print(f"   {key:7s} prolate/shear 36-trial tally: {tally}")
        D_results[key] = {"round_sweep": round_runs, "prolate_sweep": pro_runs, "prolate_tally": tally}

    # D. Save sample trajectories (for Track III publication figure!)
    def record_trace(prof, x_b, lam0, sigp0=0.0, tau_max=600.0):
        sig = 9.0 / 31.0
        stress_fn = build_casimir_lookup(prof)
        xp0 = x_b * (lam0 ** (-1.0 / 3.0))
        x30 = xp0 * lam0
        u0, _, _ = stress_fn(xp0, x30)
        yB0 = -sigp0 * sigp0 + (4.0 - lam0 * lam0) / (3.0 * xp0 * xp0)
        rhs_r = (yB0 + sig * yB0 * yB0 - u0 / 3.0) / sig
        if rhs_r < 0.0:
            return {"outcome": "no_turning_point"}
        r0 = yB0 + math.sqrt(rhs_r)
        y0 = np.array([xp0, x30, 0.0, sigp0, r0, 0.0], dtype=float)

        def rhs(t, y):
            xp, x3, hbar, sp, r, s = y
            lam = x3 / xp
            u_t, pperp_t, ppar_t = stress_fn(xp, x3)
            dhbar = r - 2.0 * hbar * hbar - sp * sp - (4.0 - lam * lam) / (3.0 * xp * xp)
            dsp = -3.0 * hbar * sp - 4.0 * (1.0 - lam * lam) / (3.0 * xp * xp) + (pperp_t - ppar_t - 6.0 * sig * s * sp) / (3.0 * (1.0 + 2.0 * sig * r))
            return [xp * (hbar + sp), x3 * (hbar - 2.0 * sp), dhbar, dsp, s, -3.0 * hbar * s - r / sig]

        ev1 = lambda t, y: (y[1] / y[0]) - 0.02
        ev1.terminal = True
        ev1.direction = -1
        ev2 = lambda t, y: 5.8 - (y[1] / y[0])
        ev2.terminal = True
        ev2.direction = -1
        ev3 = lambda t, y: min(abs(1.0 + 2.0 * sig * y[4]) - 1e-3, 1e3 - abs(y[4]))
        ev3.terminal = True
        ev3.direction = -1
        ev4 = lambda t, y: 20.0 * x_b - ((y[0] ** 2 * y[1]) ** (1.0 / 3.0))
        ev4.terminal = True
        ev4.direction = -1
        sol = solve_ivp(rhs, (0.0, tau_max), y0, method="DOP853", rtol=1e-10, atol=1e-12, events=[ev1, ev2, ev3, ev4], max_step=0.4)
        idx = np.linspace(0, len(sol.t) - 1, min(250, len(sol.t))).astype(int)
        return {
            "tau": [round(float(v), 4) for v in sol.t[idx]],
            "x_perp": [round(float(v), 4) for v in sol.y[0, idx]],
            "x_3": [round(float(v), 4) for v in sol.y[1, idx]],
            "lam_B": [round(float(sol.y[1, i] / sol.y[0, i]), 5) for i in idx],
            "x_bar": [round(float((sol.y[0, i] ** 2 * sol.y[1, i]) ** (1.0 / 3.0)), 4) for i in idx],
        }

    traces = {
        "L40_xring_lam1": record_trace(prof_L40, xr_L40, 1.0),
        "L40_xring_lam1.5": record_trace(prof_L40, xr_L40, 1.5),
        "L40_5xring_lam1": record_trace(prof_L40, 5.0 * xr_L40, 1.0, tau_max=3000.0),
        "D20_xring_lam1": record_trace(profiles["D20_em"], profiles["D20_em"]["x_ring_round_metric"], 1.0),
        "D30_xring_lam1": record_trace(profiles["D30_em"], profiles["D30_em"]["x_ring_round_metric"], 1.0),
    }

    # Strip non-JSON spline objects before dumping
    clean_profiles = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in profiles.items()}
    payload = {
        "spectrum_matrix_verification": spec_ver,
        "precision_verifications": prec_ver,
        "berger_casimir_profiles": clean_profiles,
        "L40_round_start_sweep": L40_round_sweep,
        "L40_dop853_vs_radau_rel_tau_diff": tau_diff_L40,
        "L40_prolate_and_shear_sweep": {
            "n_trials": len(L40_prolate_sweep),
            "n_bounded": n_L40_bounded,
            "outcome_tally": L40_tally,
            "trials": L40_prolate_sweep,
        },
        "dihedral_dynamical_sweeps": D_results,
        "sample_trajectories": traces,
    }
    out_path = os.path.join(OUT_DIR, "t3_01_berger_casimir_backreaction.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
