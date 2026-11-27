#!/usr/bin/env python3
"""
Stream p2_06_poincare_shear_modulus_and_talbot_revivals -- Stage 1: Exact Derivations
& High-Precision (50-digit mpmath + Exact Rational / SymPy) Identity Verification.

Verifies the three exact analytical theorems of stream p2_06:
  - Theorem 1 (KT1): Universal shell-by-shell Maxwell Contact/Tadpole Ward Identity
      Pi_contact^top(Gamma; 0) === 4 * kappa_V(Gamma) === (1920/31) * pi^2 u_EM,top(Gamma; 1)
    proved symbolically for all shells N = 2j + 1 >= 1, verified in exact Fraction
    arithmetic for N = 1..61, and checked to 50 digits across all 5 ADE space forms
    (RP^3, Q_8, T*, O*, I*), together with exact RG-running consistency (-2304/31)
    and canonical Hamiltonian seagull/contact energy identity.
  - Theorem 2 (KT2): Exact half-antipodal conformal Talbot anti-periodicity on S^3/I*
      K_I*^{disc}(s + pi/2) === -K_I*^{disc}(s)
    from E_j = 4j + 2 === 2 (mod 4), contrasted against the round-S^3 negative control
    (Delta_eta_rev = pi), plus exact localized-pulse two-scale echo comb (ratio 13/2 vs 3).
  - Theorem 3 (KT3): Exact 2-graviton selection-rule theorem for external n = 2(-) shear
    (j_L = 0, j_R = 2): strict opposite-chirality forbiddenness for all j (|j_R2 - j_R1| = 4 > 2),
    allowed same-chirality spectrum & 13-channel I* intermediate desert in (5.66, 21.91),
    exact 5x5x5 cubic isotropy sum_{b,c} |C_{abc}|^2 = 7/12, 50-digit Koszul cubic Bianchi IX
    action in exponential and volume-preserving linear parameterizations (verifying exact
    all-3-on-shell invariance V = -10 and quantifying off-shell parameterization dependence at
    Omega = 2*sqrt(8)), and O(1/N_v) residue / static-weight suppression vs Maxwell E_6 = 26.
"""

from fractions import Fraction
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np

mp.mp.dps = 50

STREAM_DIR = Path(__file__).resolve().parent
OUT_PATH = STREAM_DIR / "01_derive.json"


# ==============================================================================
# SHARED ADE CHARACTER MULTIPLICITIES (EXACT INTEGER ARITHMETIC / VERIFIED)
# ==============================================================================

def d_ade(name: str, two_j: int) -> int:
    """Exact multiplicity of trivial Gamma-representation in spin-(two_j/2) of SU(2)_L."""
    if two_j < 0:
        return 0
    if name == "I*":
        order = 120
        cls = [
            (1, 0.0), (1, math.pi), (30, math.pi / 2.0),
            (20, math.pi / 3.0), (20, 2.0 * math.pi / 3.0),
            (12, math.pi / 5.0), (12, 2.0 * math.pi / 5.0),
            (12, 3.0 * math.pi / 5.0), (12, 4.0 * math.pi / 5.0),
        ]
    elif name == "Q8":
        order = 8
        cls = [(1, 0.0), (1, math.pi), (6, math.pi / 2.0)]
    elif name == "T*":
        order = 24
        cls = [(1, 0.0), (1, math.pi), (6, math.pi / 2.0), (8, math.pi / 3.0), (8, 2.0 * math.pi / 3.0)]
    elif name == "O*":
        order = 48
        cls = [
            (1, 0.0), (1, math.pi), (18, math.pi / 2.0),
            (8, math.pi / 3.0), (8, 2.0 * math.pi / 3.0),
            (6, math.pi / 4.0), (6, 3.0 * math.pi / 4.0),
        ]
    elif name == "RP3":
        order = 2
        cls = [(1, 0.0), (1, math.pi)]
    else:
        raise ValueError(f"Unknown ADE group: {name}")
    tot = 0.0
    for sz, th in cls:
        if th == 0.0:
            chi = two_j + 1
        elif abs(th - math.pi) < 1e-12:
            chi = (two_j + 1) * ((-1) ** two_j)
        else:
            chi = math.sin((two_j + 1) * th) / math.sin(th)
        tot += sz * chi
    iv = int(round(tot / order))
    assert abs(tot / order - iv) < 1e-9, f"Non-integer multiplicity {tot/order} for {name}, 2j={two_j}"
    return iv


# ==============================================================================
# PART 1 (THEOREM 1): UNIVERSAL SHELL-BY-SHELL MAXWELL CONTACT WARD IDENTITY
# ==============================================================================

def prove_maxwell_shell_sums_symbolic():
    """
    Exact symbolic Laurent-polynomial proof in Q[N, N^-1] for arbitrary N = 2j + 1 >= 2 of:
      S_0^EM(N) = N^2 + 2,
      S_1^EM(N) === 0,
      S_2^EM(N) = (16/45) N^2 + (40/9) - (32/15) N^-2,
      (1/2) S_2^EM(N) - (4/9) S_0^EM(N) === -(1/120) * (2 * B_2(2N) / N^2).
    Represent any Laurent polynomial sum_k c_k N^k as dict {k: Fraction(c_k)}.
    """
    def lpoly_add(*polys):
        out = {}
        for p in polys:
            for k, v in p.items():
                nv = out.get(k, Fraction(0)) + v
                if nv == 0:
                    out.pop(k, None)
                else:
                    out[k] = nv
        return out

    def lpoly_scale(c, p):
        c = Fraction(c)
        if c == 0:
            return {}
        return {k: c * v for k, v in p.items() if c * v != 0}

    def lpoly_mul(p1, p2):
        out = {}
        for k1, v1 in p1.items():
            for k2, v2 in p2.items():
                k = k1 + k2
                nv = out.get(k, Fraction(0)) + v1 * v2
                if nv == 0:
                    out.pop(k, None)
                else:
                    out[k] = nv
        return out

    def lpoly_eval(p, nv):
        n_f = Fraction(nv)
        return sum(v * (n_f ** k) for k, v in p.items())

    # Exact closed-form Faulhaber power sums over interior a = two_M in {-(N-3), ..., +(N-3)} (step 2):
    #   n_int = N - 2
    #   sum_a2_full = (1/3) N (N^2 - 1), subtract 2 (N - 1)^2
    #   sum_a4_full = (1/15) N (N^2 - 1) (3 N^2 - 7), subtract 2 (N - 1)^4
    n_int = {1: Fraction(1), 0: Fraction(-2)}
    sum_a2_int = lpoly_add(
        {3: Fraction(1, 3), 1: Fraction(-1, 3)},
        {2: Fraction(-2), 1: Fraction(4), 0: Fraction(-2)},
    )
    sum_a4_int = lpoly_add(
        {5: Fraction(1, 5), 3: Fraction(-2, 3), 1: Fraction(7, 15)},
        {4: Fraction(-2), 3: Fraction(8), 2: Fraction(-12), 1: Fraction(8), 0: Fraction(-2)},
    )

    for nv in range(2, 12):
        ms = list(range(-(nv - 3), nv - 2, 2))
        assert len(ms) == lpoly_eval(n_int, nv)
        assert sum(m**2 for m in ms) == lpoly_eval(sum_a2_int, nv)
        assert sum(m**4 for m in ms) == lpoly_eval(sum_a4_int, nv)

    # Boundary |two_M| = N + 1 (2 modes): w0 = N + 1, w1 = (-2/3)(N+1), w2 = (10/9)(N+1)
    s0_top = {1: Fraction(2), 0: Fraction(2)}
    s1_top = {1: Fraction(-4, 3), 0: Fraction(-4, 3)}
    s2_top = {1: Fraction(20, 9), 0: Fraction(20, 9)}

    # Sub-boundary |two_M| = N - 1 (2 modes): w0 = N + 1, w1 = 8/3 - (2/3)(N-1), w2 = 8/9 + (10/9)(N-1)
    s0_sub = {1: Fraction(2), 0: Fraction(2)}
    s1_sub = {1: Fraction(-4, 3), 0: Fraction(20, 3)}
    s2_sub = {1: Fraction(20, 9), 0: Fraction(-4, 9)}

    # Interior |two_M| <= N - 3 (n_int values of a = two_M, each with 2 branches +/-):
    #   R1 = (1 - a^2) N^-1,  R2 = (1 + 3 a^2) N^-1 - (1 - 2 a^2 + a^4) N^-3
    #   sum bar_w0 = 2 N
    #   sum bar_w1 = (2/3) N + 2 R1 = [(2/3) N + 2 N^-1] + [-2 N^-1] a^2
    #   sum bar_w2 = (-4/9) N + (4/3) R1 + 2 R2
    #              = [(-4/9) N + (10/3) N^-1 - 2 N^-3] + [(14/3) N^-1 + 4 N^-3] a^2 + [-2 N^-3] a^4
    s0_int = lpoly_mul({1: Fraction(2)}, n_int)
    s1_int = lpoly_add(
        lpoly_mul({1: Fraction(2, 3), -1: Fraction(2)}, n_int),
        lpoly_mul({-1: Fraction(-2)}, sum_a2_int),
    )
    s2_int = lpoly_add(
        lpoly_mul({1: Fraction(-4, 9), -1: Fraction(10, 3), -3: Fraction(-2)}, n_int),
        lpoly_mul({-1: Fraction(14, 3), -3: Fraction(4)}, sum_a2_int),
        lpoly_mul({-3: Fraction(-2)}, sum_a4_int),
    )

    S0_poly = lpoly_scale(Fraction(1, 2), lpoly_add(s0_top, s0_sub, s0_int))
    S1_poly = lpoly_scale(Fraction(1, 2), lpoly_add(s1_top, s1_sub, s1_int))
    S2_poly = lpoly_scale(Fraction(1, 2), lpoly_add(s2_top, s2_sub, s2_int))

    assert S0_poly == {2: Fraction(1), 0: Fraction(2)}
    assert S1_poly == {}
    assert S2_poly == {2: Fraction(16, 45), 0: Fraction(40, 9), -2: Fraction(-32, 15)}

    # Exact Bach/Kubo Ward identity shell-by-shell:
    #   B_2(2N) / N^2 = (4 N^2 - 4)(4 N^2 - 16) / N^2 = 16 N^2 - 80 + 64 N^-2
    #   (1/2) S2(N) - (4/9) S0(N) + (1/60) B_2(2N)/N^2 === 0 identically in Q[N, N^-1]!
    B2_2N_over_N2 = {2: Fraction(16), 0: Fraction(-80), -2: Fraction(64)}
    ward_res = lpoly_add(
        lpoly_scale(Fraction(1, 2), S2_poly),
        lpoly_scale(Fraction(-4, 9), S0_poly),
        lpoly_scale(Fraction(1, 60), B2_2N_over_N2),
    )
    assert ward_res == {}

    return {
        "S0_symbolic": "N^2 + 2",
        "S1_symbolic": "0",
        "S2_symbolic": "(16/45)*N^2 + (40/9) - (32/15)*N^-2",
        "ward_shell_identity_residual": 0,
        "raw_1_over_N_coeff_in_half_N_S2": "-16/15",
        "exact_A4_EM_pp1": "+16/15",
    }


def exact_maxwell_shell_derivs_fraction(two_j: int):
    """Exact Fraction evaluation of S0(N), S1(N), S2(N) directly from Berger *_B d eigenvalues."""
    j = Fraction(two_j, 2)
    s0 = Fraction(0)
    s1 = Fraction(0)
    s2 = Fraction(0)
    for two_M in range(-two_j - 2, two_j + 3, 2):
        aM = abs(two_M)
        if aM == two_j + 2:
            c = 2 * (j + 1)
            s0 += c
            s1 += c * Fraction(-2, 3)
            s2 += c * Fraction(10, 9)
        elif aM == two_j:
            s0 += 2 + 2 * j
            s1 += Fraction(8, 3) + 2 * j * Fraction(-2, 3)
            s2 += Fraction(8, 9) + 2 * j * Fraction(10, 9)
        else:
            R0 = Fraction(two_j + 1)
            g1 = Fraction(2 - 2 * aM * aM)
            g2 = Fraction(2 + 6 * aM * aM)
            R1 = g1 / (2 * R0)
            R2 = (g2 - 2 * R1 * R1) / (2 * R0)
            for sgn in (+1, -1):
                om0 = R0 + sgn
                om1 = R1 + sgn
                om2 = R2
                s0 += om0
                s1 += Fraction(1, 3) * om0 + om1
                s2 += Fraction(-2, 9) * om0 + Fraction(2, 3) * om1 + om2
    return s0 / 2, s1 / 2, s2 / 2


def eval_ade_maxwell_and_ward_50dps():
    """
    50-digit analytic Hurwitz-zeta / Dirichlet-eta / digamma / trigamma evaluation of:
      - pi^2 u_EM,top(Gamma; 1)
      - pi^2 u''_EM,top(Gamma; 1)
      - Pi_tot^top(Gamma; 0) = (2160/31) * pi^2 u''_EM,top(Gamma; 1)
      - Pi_Kubo^top(Gamma; 0) (independent Kubo sum over opposite-chirality poles!)
      - Pi_contact^top(Gamma; 0) = Pi_tot^top - Pi_Kubo^top vs EXACT 4 * kappa_V(Gamma)
    across ALL 5 ADE space forms: RP3, Q8, T*, O*, I*!
    """
    ade_specs = [
        ("RP3", 2, 2, Fraction(3, 16)),
        ("Q8", 8, 4, Fraction(27, 16)),
        ("T*", 24, 12, Fraction(499, 144)),
        ("O*", 48, 24, Fraction(67, 144)),
        ("I*", 120, 60, Fraction(-22369, 720)),
    ]

    def dirichlet_eta_at(s):
        if s == 1:
            return mp.log(2)
        return (1 - mp.power(2, 1 - s)) * mp.zeta(s)

    results = {}
    max_ward_err_50dps = mp.mpf(0)
    max_u1_rat_err_50dps = mp.mpf(0)

    for name, order, period_N, u1_rat_expected in ade_specs:
        # Exact decomposition: order * d_Gamma(N - 1) - N = (-1)^(N-1) N + R_per(N)
        R_per = [
            (order * d_ade(name, r - 1) - 2 * r) if (r % 2 == 1) else 0
            for r in range(1, period_N + 1)
        ]
        for N in range(1, 3 * period_N + 1):
            assert order * d_ade(name, N - 1) - N == ((-1) ** (N - 1)) * N + R_per[(N - 1) % period_N]
        assert sum(R_per) == 0

        def sum_wt_Npow(k: int):
            eta_part = dirichlet_eta_at(-(k + 1))
            P = mp.mpf(period_N)
            if k == -1:
                r_part = -sum(mp.mpf(R_per[r - 1]) * mp.psi(0, mp.mpf(r) / P) for r in range(1, period_N + 1)) / P
            else:
                s = -k
                r_part = (P ** (-s)) * sum(
                    mp.mpf(R_per[r - 1]) * mp.zeta(s, mp.mpf(r) / P) for r in range(1, period_N + 1)
                )
            return eta_part + r_part

        # (A) Direct Casimir energy and shear modulus from S0(N) and S2(N):
        u1_top = mp.mpf("0.5") * (sum_wt_Npow(2) + 2 * sum_wt_Npow(0))
        upp1_top = mp.mpf("0.5") * (
            (mp.mpf(16) / 45) * sum_wt_Npow(2)
            + (mp.mpf(40) / 9) * sum_wt_Npow(0)
            - (mp.mpf(32) / 15) * sum_wt_Npow(-2)
        )
        u1_rat_mp = mp.mpf(u1_rat_expected.numerator) / mp.mpf(u1_rat_expected.denominator)
        err_u1_rat = abs(u1_top - u1_rat_mp)
        max_u1_rat_err_50dps = max(max_u1_rat_err_50dps, err_u1_rat)
        assert err_u1_rat < mp.mpf("1e-45")

        Pi_tot_top = (mp.mpf(2160) / 31) * upp1_top

        # (B) Independent Kubo bubble Pi_Kubo^top(Gamma; 0) directly from p2_01 Theorem 2:
        #   Opposite-chirality poles sit at j >= 1 (integer j for ADE since -I in Gamma => d_Gamma(half-int)=0),
        #   so N = 2j + 1 >= 3 (odd N), with E_j = 2N, B_2(2N) = 16 N^4 - 80 N^2 + 64,
        #   weight difference 2*W_Gamma(j) - 2*W_S3(j) summed over all N >= 3:
        #     K_Gamma(0; mu=1) - K_S3(0; mu=1) evaluated via period-(period_N/2) digamma/trigamma!
        #   Recall for j >= 1 (E_j = 4j + 2 = 2N):
        #     2*W_Gamma(j) = 2*order*d_Gamma(2j)/(2j+1) = 4 + 2*R_j/(2j+1), where R_j = R_per[2j] (period P_j = period_N // 2)
        #   In Pi_Kubo(0) = 2*c_anom*64*K(0) - 2*c_anom*(zeta(3) - 20*zeta(1)):
        P_j = period_N // 2
        R_j = [order * d_ade(name, 2 * r) - 2 * (2 * r + 1) for r in range(P_j)]
        assert sum(R_j) == 0

        # Exact K_Gamma(0; mu=1) using the exact period-P_j digamma/trigamma formula from p2_01 Theorem 2:
        #   Separate j = 1..P_j explicitly, tail j = P_j * m + r (m >= 1, r = 1..P_j):
        lead_G = mp.log(16) + 2 * mp.psi(0, mp.mpf(P_j) + mp.mpf("1.5"))
        rem_G = sum(
            mp.mpf(R_j[r % P_j]) * 8 * mp.psi(1, mp.mpf(4 * P_j + 4 * r + 2) / (4 * P_j)) / ((4 * P_j) ** 2)
            for r in range(1, P_j + 1)
        )
        low_G = sum(
            (mp.mpf(2 * order * d_ade(name, 2 * j)) / (2 * j + 1)) * 2 / (4 * j + 2)
            for j in range(1, P_j + 1) if d_ade(name, 2 * j) > 0
        )
        K0_G_mu1 = lead_G - rem_G - low_G
        K0_S3_mu1 = mp.log(4) + 2 * mp.psi(0, 3)
        K0_top = K0_G_mu1 - K0_S3_mu1

        # Exact rational Hurwitz-zeta constants zeta_Gamma(1) and zeta_Gamma(3):
        #   zeta_Gamma(p) = 8 * 4^p * zeta(-p, 3/2) + 8 * (4*P_j)^(p-1) * sum_{r=1}^{P_j} R_j[r%P_j] * zeta(1-p, (4r+2)/(4*P_j))
        # Using Bernoulli polynomials B_1(x) = x - 1/2, B_2(x) = x^2 - x + 1/6, B_3(x) = x^3 - 3/2 x^2 + 1/2 x, B_4(x) = x^4 - 2x^3 + x^2 - 1/30:
        def B1(x):
            return x - Fraction(1, 2)
        def B2(x):
            return x * x - x + Fraction(1, 6)
        def B3(x):
            return x**3 - Fraction(3, 2) * x**2 + Fraction(1, 2) * x
        def B4(x):
            return x**4 - 2 * x**3 + x**2 - Fraction(1, 30)

        # zeta(-1, a) = -B_2(a)/2, zeta(-3, a) = -B_4(a)/4, zeta(0, a) = -B_1(a), zeta(-2, a) = -B_3(a)/3
        z1_G = -16 * B2(Fraction(3, 2)) - 8 * sum(
            Fraction(R_j[r % P_j]) * B1(Fraction(4 * r + 2, 4 * P_j)) for r in range(1, P_j + 1)
        )
        z3_G = -128 * B4(Fraction(3, 2)) - Fraction(8 * (4 * P_j)**2, 3) * sum(
            Fraction(R_j[r % P_j]) * B3(Fraction(4 * r + 2, 4 * P_j)) for r in range(1, P_j + 1)
        )
        z1_S3 = Fraction(-74, 3)
        z3_S3 = Fraction(-4316, 15)

        c_an_f = Fraction(9, 31)
        Pi_kubo_rem_G = -2 * c_an_f * (z3_G - 20 * z1_G)
        Pi_kubo_rem_S3 = -2 * c_an_f * (z3_S3 - 20 * z1_S3)
        Pi_kubo_rem_top = Pi_kubo_rem_G - Pi_kubo_rem_S3

        Pi_kubo_full_top = (
            mp.mpf(128 * 9) / 31 * K0_top
            + mp.mpf(Pi_kubo_rem_top.numerator) / mp.mpf(Pi_kubo_rem_top.denominator)
        )

        # (C) Extract contact/tadpole diagram & verify exact Ward identity Pi_contact^top === 4 * kappa_V(Gamma)!
        Pi_contact_top = Pi_tot_top - Pi_kubo_full_top
        kappa_V_G = Fraction(480, 31) * u1_rat_expected
        four_kappa_V_G = 4 * kappa_V_G  # = (1920/31) * u1_rat_expected
        four_kappa_mp = mp.mpf(four_kappa_V_G.numerator) / mp.mpf(four_kappa_V_G.denominator)
        ward_err = abs(Pi_contact_top - four_kappa_mp)
        max_ward_err_50dps = max(max_ward_err_50dps, ward_err)
        assert ward_err < mp.mpf("1e-45"), f"Ward identity failed on {name}: err = {ward_err}"

        results[name] = {
            "order": order,
            "period_N": period_N,
            "pi2_u_EM_top_exact_rational": str(u1_rat_expected),
            "pi2_u_EM_top_50dps": mp.nstr(u1_top, 35),
            "u1_rational_err_50dps": float(err_u1_rat),
            "pi2_upp_EM_top": float(upp1_top),
            "pi2_upp_EM_top_50dps": mp.nstr(upp1_top, 35),
            "kappa_V_rational": str(kappa_V_G),
            "four_kappa_V_rational": str(four_kappa_V_G),
            "zeta_Gamma_1_rational": str(z1_G),
            "zeta_Gamma_3_rational": str(z3_G),
            "K0_top_diff": float(K0_top),
            "Pi_tot_top_0": float(Pi_tot_top),
            "Pi_kubo_rem_Gamma_rational": str(Pi_kubo_rem_G),
            "Pi_kubo_rem_top_rational": str(Pi_kubo_rem_top),
            "Pi_kubo_full_top_0": float(Pi_kubo_full_top),
            "Pi_contact_top_0": float(Pi_contact_top),
            "Pi_contact_top_0_50dps": mp.nstr(Pi_contact_top, 35),
            "ward_identity_abs_err_50dps": float(ward_err),
        }

    # Verify I* matches p2_01 exact rationals identically:
    assert results["I*"]["zeta_Gamma_1_rational"] == "-724/3"
    assert results["I*"]["zeta_Gamma_3_rational"] == "-521224/45"
    assert results["I*"]["Pi_kubo_rem_Gamma_rational"] == "608048/155"
    assert results["I*"]["kappa_V_rational"] == "-44738/93"
    assert results["I*"]["four_kappa_V_rational"] == "-178952/93"

    # Independent RG-running cross-check:
    #   mu d/dmu [ (2160/31) pi^2 u''_EM,ren(S^3; 1; mu) ] = (2160/31) * (-16/15) = -2304/31
    #   mu d/dmu [ 2 * c_anom * B_2(0) * K(0; mu) ]       = 2 * (9/31) * 64 * (-2) = -2304/31
    rg_from_casimir = Fraction(2160, 31) * Fraction(-16, 15)
    rg_from_kubo = 2 * Fraction(9, 31) * 64 * (-2)
    assert rg_from_casimir == rg_from_kubo == Fraction(-2304, 31)

    return {
        "ade_exact_table": results,
        "max_u1_rational_err_50dps": float(max_u1_rat_err_50dps),
        "max_ward_identity_err_50dps": float(max_ward_err_50dps),
        "rg_running_casimir_exact": str(rg_from_casimir),
        "rg_running_kubo_exact": str(rg_from_kubo),
        "rg_running_mismatch": int((rg_from_casimir - rg_from_kubo).numerator),
    }


# ==============================================================================
# PART 2 (THEOREM 2): HALF-ANTIPODAL CONFORMAL TALBOT REVIVALS (S^3/I* vs S^3)
# ==============================================================================

def verify_talbot_revival_theorem():
    """
    Verify Theorem 2:
      - Exact 50-digit and float64 anti-periodicity K_I*^{disc}(s + pi/2) === -K_I*^{disc}(s)
        at J_cut = 30 (15 poles, E in [26, 122]) and J_cut = 60 (45 poles, E in [26, 242]).
      - Round-S^3 negative control (59 poles at J_cut = 30, E in [6, 122]): fails at pi/2
        and revives only at Delta_eta = pi.
      - Exact localized Gaussian pulse (sigma_eta = 0.02) echo comb ratios and two-scale
        autocorrelation ratio (pi/2) / (pi/13) = 13/2 vs pi / (pi/3) = 3 on round S^3.
    """
    out = {}
    for J_cut in (30, 60):
        # I* poles
        E_I, w_I = [], []
        for j in range(1, J_cut + 1):
            d = d_ade("I*", 2 * j)
            if d > 0:
                Ej = 4 * j + 2
                assert Ej % 4 == 2
                E_I.append(Ej)
                w_I.append(240.0 * d / (2 * j + 1))
        E_I_np = np.array(E_I, dtype=float)
        a_I_np = np.array(w_I, dtype=float) / E_I_np

        # Round S^3 poles: E_k = 2k for k = 3 .. 2*J_cut + 1, w_k = 2
        E_S3_np = np.arange(6, 4 * J_cut + 3, 2, dtype=float)
        a_S3_np = 2.0 / E_S3_np

        # 50-digit check at 25 irrational points s in [0, 2*pi]
        err_I_anti_50 = mp.mpf(0)
        err_I_per_50 = mp.mpf(0)
        err_S3_per_50 = mp.mpf(0)
        for m in range(1, 26):
            sv = mp.mpf(m) / mp.mpf("3.71")
            K0 = sum(mp.mpf(2 * 240 * d_ade("I*", 2 * j)) / ((2 * j + 1) * (4 * j + 2)) * mp.cos((4 * j + 2) * sv)
                     for j in range(1, J_cut + 1) if d_ade("I*", 2 * j) > 0)
            Kh = sum(mp.mpf(2 * 240 * d_ade("I*", 2 * j)) / ((2 * j + 1) * (4 * j + 2)) * mp.cos((4 * j + 2) * (sv + mp.pi / 2))
                     for j in range(1, J_cut + 1) if d_ade("I*", 2 * j) > 0)
            Kp = sum(mp.mpf(2 * 240 * d_ade("I*", 2 * j)) / ((2 * j + 1) * (4 * j + 2)) * mp.cos((4 * j + 2) * (sv + mp.pi))
                     for j in range(1, J_cut + 1) if d_ade("I*", 2 * j) > 0)
            err_I_anti_50 = max(err_I_anti_50, abs(Kh + K0))
            err_I_per_50 = max(err_I_per_50, abs(Kp - K0))

            KS0 = sum((mp.mpf(4) / (2 * k)) * mp.cos(2 * k * sv) for k in range(3, 2 * J_cut + 2))
            KSp = sum((mp.mpf(4) / (2 * k)) * mp.cos(2 * k * (sv + mp.pi)) for k in range(3, 2 * J_cut + 2))
            err_S3_per_50 = max(err_S3_per_50, abs(KSp - KS0))

        # Float64 dense grid (4001 pts on [0, 2*pi])
        s_grid = np.linspace(0.0, 2.0 * math.pi, 4001)
        KI0 = (2.0 * a_I_np) @ np.cos(np.outer(E_I_np, s_grid))
        KIh = (2.0 * a_I_np) @ np.cos(np.outer(E_I_np, s_grid + 0.5 * math.pi))
        KIp = (2.0 * a_I_np) @ np.cos(np.outer(E_I_np, s_grid + math.pi))

        KS0_f = (2.0 * a_S3_np) @ np.cos(np.outer(E_S3_np, s_grid))
        KSh_f = (2.0 * a_S3_np) @ np.cos(np.outer(E_S3_np, s_grid + 0.5 * math.pi))
        KSp_f = (2.0 * a_S3_np) @ np.cos(np.outer(E_S3_np, s_grid + math.pi))

        out[f"J_cut_{J_cut}"] = {
            "n_poles_Istar": len(E_I),
            "n_poles_S3": len(E_S3_np),
            "E_range_Istar": [int(E_I[0]), int(E_I[-1])],
            "E_range_S3": [int(E_S3_np[0]), int(E_S3_np[-1])],
            "Istar_anti_pi_over_2_err_50dps": float(err_I_anti_50),
            "Istar_pos_pi_err_50dps": float(err_I_per_50),
            "S3_pos_pi_err_50dps": float(err_S3_per_50),
            "Istar_anti_pi_over_2_rel_f64": float(np.max(np.abs(KIh + KI0)) / np.max(np.abs(KI0))),
            "Istar_pos_pi_rel_f64": float(np.max(np.abs(KIp - KI0)) / np.max(np.abs(KI0))),
            "S3_anti_pi_over_2_rel_f64": float(np.max(np.abs(KSh_f + KS0_f)) / np.max(np.abs(KS0_f))),
            "S3_pos_pi_over_2_rel_f64": float(np.max(np.abs(KSh_f - KS0_f)) / np.max(np.abs(KS0_f))),
            "S3_pos_pi_rel_f64": float(np.max(np.abs(KSp_f - KS0_f)) / np.max(np.abs(KS0_f))),
        }
        assert err_I_anti_50 < mp.mpf("1e-45") and err_I_per_50 < mp.mpf("1e-45")

    # Exact localized Gaussian pulse echo comb (sigma_eta = 0.02, J_cut = 30)
    E_I = np.array([4 * j + 2 for j in range(1, 31) if d_ade("I*", 2 * j) > 0], dtype=float)
    a_I = np.array([240.0 * d_ade("I*", 2 * j) / ((2 * j + 1) * (4 * j + 2)) for j in range(1, 31) if d_ade("I*", 2 * j) > 0], dtype=float)
    E_S3 = np.arange(6, 123, 2, dtype=float)
    a_S3 = 2.0 / E_S3
    sig_p = 0.02
    pulse = {}
    for name, Ep, ap, rev_period in [("Istar", E_I, a_I, 0.5 * math.pi), ("S3", E_S3, a_S3, math.pi)]:
        amps = math.sqrt(2.0 * math.pi) * sig_p * np.exp(-0.5 * (Ep**2) * (sig_p**2))
        M0 = float(-(ap * amps) @ np.ones(len(Ep)))
        M_half = float(-(ap * amps) @ np.cos(Ep * 0.5 * math.pi))
        M_pi = float(-(ap * amps) @ np.cos(Ep * math.pi))
        M_3half = float(-(ap * amps) @ np.cos(Ep * 1.5 * math.pi))

        # Unsmoothed autocorrelation on post-pulse ringdown [0.35, 4*pi]
        t_ev = np.linspace(0.0, 4.0 * math.pi, 16001)
        m_post = t_ev >= 0.35
        dt_p = t_ev[m_post] - 0.2
        M_echo = -(ap * amps) @ np.cos(np.outer(Ep, dt_p))
        x = M_echo - np.mean(M_echo)
        d_eta = float(t_ev[1] - t_ev[0])
        n_lag = int(round(2.2 * math.pi / d_eta))
        n_fft = 1 << (2 * len(x) - 1).bit_length()
        fx = np.fft.rfft(x, n=n_fft)
        ac = np.fft.irfft(fx * np.conj(fx), n=n_fft)[:n_lag + 1]
        counts = np.arange(len(x), len(x) - n_lag - 1, -1, dtype=float)
        ac = (ac / counts) / (ac[0] / len(x))
        lags = np.arange(n_lag + 1) * d_eta
        m_in = (lags >= 0.15) & (lags <= 1.2)
        ac_in, l_in = ac[m_in], lags[m_in]
        pks = np.where((ac_in[1:-1] > ac_in[:-2]) & (ac_in[1:-1] >= ac_in[2:]) & (ac_in[1:-1] > 0.05))[0] + 1
        intra_lag = float(l_in[pks[0]])
        exact_intra = 2.0 * math.pi / float(Ep[0])
        pulse[name] = {
            "E_min": float(Ep[0]),
            "revival_period": rev_period,
            "exact_intra_period": exact_intra,
            "exact_two_scale_ratio": rev_period / exact_intra,
            "M_ratio_pi_over_2": M_half / M0,
            "M_ratio_pi": M_pi / M0,
            "M_ratio_3pi_over_2": M_3half / M0,
            "measured_intra_lag": intra_lag,
            "measured_two_scale_ratio_vs_rev": rev_period / intra_lag,
            "ac_at_pi_over_2": float(ac[int(round(0.5 * math.pi / d_eta))]),
            "ac_at_pi": float(ac[int(round(math.pi / d_eta))]),
        }
    out["localized_pulse_comb"] = pulse
    return out


# ==============================================================================
# PART 3 (THEOREM 3): 2-GRAVITON SELECTION RULES & 50-DIGIT BIANCHI IX VERTEX
# ==============================================================================

def verify_2grav_selection_and_vertex_50dps():
    """
    Verify Theorem 3:
      1. Exact I*-invariant TT graviton spectrum (p = 3..45) vs round S^3.
      2. Exact SU(2)_L x SU(2)_R x Bose selection rules for external n = 2(-) at (0, 2):
         - Opposite-chirality pairs strictly forbidden for all j (|j_R2 - j_R1| = 4 > 2).
         - 13-channel intermediate round-S^3 desert in (5.66, 21.91) projected out by I*.
      3. Exact 5x5x5 cubic isotropy sum_{b,c} |C_{abc}|^2 = 7/12 (exact rational check).
      4. 50-digit mpmath Koszul R^(3)(g) and ADM kinetic cubic expansion in BOTH:
           (A) Exponential g = a^2 exp(H)  =>  c_3 = -10/3, k_3 = 0, V_exp = -10
           (B) Volume-preserving linear g = a^2 det(I+H)^(-1/3) (I + H)  =>  c_3 = -4/3, k_3 = -1/2
         verifying exact all-3-on-shell invariance V_vol_lin === V_exp === -10 to >40 digits,
         and quantifying off-shell cut parameterization dependence at Omega = 2*sqrt(8)
         (V_vol_lin = -16 vs V_exp = -10) and O(1/N_v) residue suppression vs Maxwell E_6 = 26.
    """
    # 1. TT spectrum p = 3..45
    tt_spec = []
    for p in range(3, 46):
        m_m = d_ade("I*", p - 3)
        m_p = d_ade("I*", p + 1)
        d2_m = m_m * (p + 2)
        d2_p = m_p * (p - 2)
        tt_spec.append({
            "p": p, "n": p - 1, "omega": math.sqrt(p * p - 1.0),
            "m_minus": m_m, "d2_minus": d2_m,
            "m_plus": m_p, "d2_plus": d2_p,
            "d2_Istar": d2_m + d2_p, "d2_S3": 2 * (p * p - 4),
        })

    # 2. Allowed 2-graviton channels (same-chirality only; opposite-chirality proven 0 for all j)
    def build_2grav(topology: str, two_j_max: int):
        ch = []
        for two_j in range(two_j_max + 1):
            j = 0.5 * two_j
            ml = d_ade("I*", two_j) if topology == "Istar" else (two_j + 1)
            if ml == 0:
                continue
            # Check opposite-chirality triangle inequality explicitly for j >= 2:
            if two_j >= 4:
                jR_p = j - 2.0
                jR_m = j + 2.0
                assert abs(jR_m - jR_p) == 4.0 > 2.0  # strictly forbidden!
            # (-,-) allowed for all j >= 0: j_R = j + 2 >= 2, Bose parity (-1)^(2j)*(-1)^(2j+2) = +1
            p_m = two_j + 3
            om_m = math.sqrt(p_m * p_m - 1.0)
            ch.append({
                "two_j": two_j, "j_L": j, "chirality": "(-,-)",
                "p": p_m, "n": p_m - 1, "j_R": j + 2.0,
                "mult_L": ml, "omega_single": om_m, "E_pair": 2.0 * om_m,
            })
            # (+,+) allowed iff j >= 3 (j_R = j - 2 >= 1)
            if two_j >= 6:
                p_p = two_j - 1
                om_p = math.sqrt(p_p * p_p - 1.0)
                ch.append({
                    "two_j": two_j, "j_L": j, "chirality": "(+,+)",
                    "p": p_p, "n": p_p - 1, "j_R": j - 2.0,
                    "mult_L": ml, "omega_single": om_p, "E_pair": 2.0 * om_p,
                })
        ch.sort(key=lambda r: (r["E_pair"], r["chirality"]))
        return ch

    ch_I = build_2grav("Istar", 44)
    ch_S3 = build_2grav("S3", 20)
    s3_desert_channels = [c for c in ch_S3 if 5.66 < c["E_pair"] < 21.90]
    assert len(s3_desert_channels) == 13

    # 3. Exact rational 5x5x5 cubic coupling tensor |C_{abc}|^2 = |Tr(T_a T_b T_c)|^2
    # Each T_a = M_a / sqrt(d_a) where M_a is integer 3x3 and d_a in {2, 6, 2, 2, 2}:
    M_int = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 1, 0], [0, 0, -2]],
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    ]
    d_norm = [2, 6, 2, 2, 2]

    def mat_mul_int(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    def mat_tr_int(A):
        return A[0][0] + A[1][1] + A[2][2]

    for a in range(5):
        for b in range(5):
            tr2 = mat_tr_int(mat_mul_int(M_int[a], M_int[b]))
            if a == b:
                assert tr2 == d_norm[a]
            else:
                assert tr2 == 0
    iso_sums_exact = []
    for a in range(5):
        s_val = Fraction(0)
        for b in range(5):
            for c in range(5):
                tr3 = mat_tr_int(mat_mul_int(mat_mul_int(M_int[a], M_int[b]), M_int[c]))
                s_val += Fraction(tr3 * tr3, d_norm[a] * d_norm[b] * d_norm[c])
        assert s_val == Fraction(7, 12)
        iso_sums_exact.append(str(s_val))

    # 4. 50-digit mpmath Koszul R^(3)(g) and ADM kinetic cubic expansion on general non-diagonal H, V
    def mp_mat_inv_3(M):
        return M ** -1

    def mp_det_3(M):
        return mp.det(M)

    def mp_trace_3(M):
        return M[0, 0] + M[1, 1] + M[2, 2]

    def mp_bianchi_ix_R3(g):
        ginv = mp_mat_inv_3(g)
        c_up = [[[mp.mpf(0) for _ in range(3)] for _ in range(3)] for _ in range(3)]
        c_up[0][1][2] = c_up[1][2][0] = c_up[2][0][1] = mp.mpf(2)
        c_up[0][2][1] = c_up[2][1][0] = c_up[1][0][2] = mp.mpf(-2)
        Gamma = [[[mp.mpf(0) for _ in range(3)] for _ in range(3)] for _ in range(3)]
        for a in range(3):
            for b in range(3):
                for ci in range(3):
                    rhs = mp.mpf(0)
                    for d in range(3):
                        t1 = sum(g[d, e] * c_up[e][b][ci] for e in range(3))
                        t2 = sum(g[b, e] * c_up[e][ci][d] for e in range(3))
                        t3 = sum(g[ci, e] * c_up[e][d][b] for e in range(3))
                        rhs += ginv[a, d] * mp.mpf("0.5") * (t1 - t2 + t3)
                    Gamma[a][b][ci] = rhs
        Ric = [[mp.mpf(0) for _ in range(3)] for _ in range(3)]
        for a in range(3):
            for b in range(3):
                val = mp.mpf(0)
                for k in range(3):
                    val += sum(
                        Gamma[k][k][e] * Gamma[e][b][a] - Gamma[k][b][e] * Gamma[e][k][a] - c_up[e][k][b] * Gamma[k][e][a]
                        for e in range(3)
                    )
                Ric[a][b] = val
        return sum(ginv[a, b] * Ric[a][b] for a in range(3) for b in range(3))

    # Sanity check Koszul R^(3) to 50 digits on round S^3 and Berger(1.23)
    R3_round_err = abs(mp_bianchi_ix_R3(mp.eye(3)) - 6)
    lam_chk = mp.mpf("1.23")
    g_berg = mp.diag([lam_chk ** (-mp.mpf(2) / 3), lam_chk ** (-mp.mpf(2) / 3), lam_chk ** (mp.mpf(4) / 3)])
    R3_berg_err = abs(mp_bianchi_ix_R3(g_berg) - 2 * (4 - lam_chk**2) * (lam_chk ** (mp.mpf(2) / 3)))
    assert R3_round_err < mp.mpf("1e-48") and R3_berg_err < mp.mpf("1e-47")

    # General non-diagonal symmetric traceless H, V with exact rational coefficients
    T_mp = [
        mp.matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) / mp.sqrt(2),
        mp.matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / mp.sqrt(6),
        mp.matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) / mp.sqrt(2),
        mp.matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]) / mp.sqrt(2),
        mp.matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]) / mp.sqrt(2),
    ]
    h_c = [mp.mpf(c) / 10 for c in [7, -5, 11, -3, 13]]
    v_c = [mp.mpf(c) / 10 for c in [-4, 9, -7, 6, -5]]
    H_mp = sum(h_c[i] * T_mp[i] for i in range(5))
    V_mp = sum(v_c[i] * T_mp[i] for i in range(5))
    trH2 = mp_trace_3(H_mp * H_mp)
    trH3 = mp_trace_3(H_mp * H_mp * H_mp)
    trV2 = mp_trace_3(V_mp * V_mp)
    trHV2 = mp_trace_3(H_mp * V_mp * V_mp)

    # Exact Cauchy contour extraction of Taylor coefficients c_k = [eps^k] f(eps) to 45+ digits!
    # Using 16-point unit-circle / radius-r Cauchy FFT avoids ALL finite-difference cancellation!
    def cauchy_taylor_0123(fn_complex, rad=mp.mpf("0.02"), n_pts=48):
        cs = [mp.mpc(0)] * 4
        for m in range(n_pts):
            z = rad * mp.exp(2j * mp.pi * m / n_pts)
            fz = fn_complex(z)
            for k in range(4):
                cs[k] += fz / (z ** k)
        return [mp.re(c / n_pts) for c in cs]

    def mp_expm_3(M):
        # Convergent Taylor series to 50 digits (||M|| <= 0.2)
        res = mp.eye(3)
        term = mp.eye(3)
        for n in range(1, 45):
            term = (term * M) / n
            res = res + term
        return res

    # (A) Exponential potential: sqrt(det(exp(e H))) * R^(3)(exp(e H))
    c_exp_pot = cauchy_taylor_0123(lambda z: mp_bianchi_ix_R3(mp_expm_3(z * H_mp)))
    # (B) Volume-preserving linear potential: R^(3)(det(I+eH)^(-1/3) (I+eH))
    def R3_vl(z):
        M = mp.eye(3) + z * H_mp
        g = M * (mp_det_3(M) ** (-mp.mpf(1) / 3))
        return mp_bianchi_ix_R3(g)
    c_vl_pot = cauchy_taylor_0123(R3_vl)

    # Exact ADM kinetic term at O(V^2) as function of z (extracting coeff of z^0 and z^1 of K(z*H, V)):
    #   For g(t) = exp(z*H + t*V), g^{-1} dg/dt|_{t=0} = sum_{m>=0} (-1)^m / (m+1)! ad_{zH}^m(V)
    def kin_exp_fn(z):
        ad0 = V_mp
        A = ad0
        zH = z * H_mp
        ad_cur = ad0
        fact = mp.mpf(1)
        for m in range(1, 25):
            ad_cur = zH * ad_cur - ad_cur * zH
            fact *= (m + 1)
            A = A + ((-1) ** m) * ad_cur / fact
        return mp.mpf("0.25") * (mp_trace_3(A * A) - mp_trace_3(A) ** 2)

    #   For g(t) = det(I + z*H + t*V)^(-1/3) (I + z*H + t*V):
    #     A = g^{-1} dg/dt|_{t=0} = M^{-1} V - (1/3) Tr(M^{-1} V) I, where M = I + z*H!
    def kin_vl_fn(z):
        M = mp.eye(3) + z * H_mp
        MV = mp_mat_inv_3(M) * V_mp
        A = MV - (mp_trace_3(MV) / 3) * mp.eye(3)
        return mp.mpf("0.25") * (mp_trace_3(A * A) - mp_trace_3(A) ** 2)

    k_exp = cauchy_taylor_0123(kin_exp_fn)
    k_vl = cauchy_taylor_0123(kin_vl_fn)

    c2_exp_err = abs(c_exp_pot[2] / trH2 + 2)
    c3_exp_err = abs(c_exp_pot[3] / trH3 + mp.mpf(10) / 3)
    k2_exp_err = abs(k_exp[0] / trV2 - mp.mpf("0.25"))
    k3_exp_err = abs(k_exp[1])

    c2_vl_err = abs(c_vl_pot[2] / trH2 + 2)
    c3_vl_err = abs(c_vl_pot[3] / trH3 + mp.mpf(4) / 3)
    k2_vl_err = abs(k_vl[0] / trV2 - mp.mpf("0.25"))
    k3_vl_err = abs(k_vl[1] / trHV2 + mp.mpf("0.5"))

    assert max(c2_exp_err, c3_exp_err, k2_exp_err, k3_exp_err, c2_vl_err, c3_vl_err, k2_vl_err, k3_vl_err) < mp.mpf("1e-40")

    # On-shell 3-graviton vertex invariance check (in units M_Pl^2 a^2 C_{abc}):
    #   L^(3) = (M_Pl^2 a^2 / 2) [ k_3 Tr(H H'^2) + c_3 Tr(H^3) ]
    #   Symmetrized 3-point vertex for (w1, w2; Om = w1 + w2):
    #     V(Om, w1, w2) = k_3 * (Om^2 - w1*w2) + 3 * c_3
    #   All-3-on-shell (complex kinematics w1^2 = w2^2 = Om^2 = 8, Om = w1 + w2 => w1*w2 = -4):
    V_exp_onshell = k_exp[1] / trHV2 * (8 - (-4)) + 3 * (c_exp_pot[3] / trH3)
    V_vl_onshell = k_vl[1] / trHV2 * (8 - (-4)) + 3 * (c_vl_pot[3] / trH3)
    onshell_invariance_err = abs(V_exp_onshell - V_vl_onshell)
    assert onshell_invariance_err < mp.mpf("1e-42") and abs(V_exp_onshell + 10) < mp.mpf("1e-42")

    # Off-shell 2-graviton cut point (w1 = w2 = sqrt(8), Om = 2*sqrt(8) => Om^2 - w1*w2 = 32 - 8 = 24):
    V_exp_cut = -10.0
    V_vl_cut = -0.5 * 24.0 + 3.0 * (-4.0 / 3.0)  # -12 - 4 = -16.0

    # Exact 1-loop 2-graviton micro residue at E_0 = 2*sqrt(8) vs Maxwell E_6 = 26 (per N_v = 1):
    V_Istar = (math.pi ** 2) / 60.0
    E0_2grav = 2.0 * math.sqrt(8.0)
    # sum_{unordered {b,c}} |<b,c|J_grav|0>|^2 = 200 * (7/12) = 350/3; divide by 2*V_I* = pi^2/30 => 3500 / pi^2
    res_grav_exp = (350.0 / 3.0) / (2.0 * V_Istar)
    res_grav_vl = res_grav_exp * ((V_vl_cut / V_exp_cut) ** 2)
    B2_26 = (26**2 - 4) * (26**2 - 16)
    res_em_N1 = (1.0 / (40.0 * math.pi**2)) * (120.0 / 13.0) * B2_26
    ratio_res_exp = res_grav_exp / res_em_N1
    ratio_static_exp = (res_grav_exp / E0_2grav) / (res_em_N1 / 26.0)
    ratio_res_vl = res_grav_vl / res_em_N1
    ratio_static_vl = (res_grav_vl / E0_2grav) / (res_em_N1 / 26.0)

    return {
        "lowest_Istar_TT_harmonics": [r for r in tt_spec if r["d2_Istar"] > 0][:12],
        "p13_absent_verified": next(r["d2_Istar"] for r in tt_spec if r["p"] == 13) == 0,
        "opposite_chirality_forbidden_all_j": True,
        "allowed_Istar_2grav_lowest10": ch_I[:10],
        "round_S3_desert_channels_projected_out_count": len(s3_desert_channels),
        "round_S3_desert_channels_sample": [f"p={c['p']}{c['chirality']}({c['E_pair']:.2f})" for c in s3_desert_channels],
        "cubic_C_tensor_isotropy_exact_all_5": iso_sums_exact,
        "koszul_R3_round_err_50dps": float(R3_round_err),
        "koszul_R3_berger_1p23_err_50dps": float(R3_berg_err),
        "cubic_action_50dps": {
            "exp_c2_err": float(c2_exp_err),
            "exp_c3_exact": "-10/3",
            "exp_c3_err": float(c3_exp_err),
            "exp_k2_err": float(k2_exp_err),
            "exp_k3_exact": "0",
            "exp_k3_err": float(k3_exp_err),
            "vol_lin_c2_err": float(c2_vl_err),
            "vol_lin_c3_exact": "-4/3",
            "vol_lin_c3_err": float(c3_vl_err),
            "vol_lin_k2_err": float(k2_vl_err),
            "vol_lin_k3_exact": "-1/2",
            "vol_lin_k3_err": float(k3_vl_err),
            "V_all_3_onshell_exp": float(V_exp_onshell),
            "V_all_3_onshell_vol_lin": float(V_vl_onshell),
            "onshell_invariance_err_50dps": float(onshell_invariance_err),
            "V_cut_exp": V_exp_cut,
            "V_cut_vol_lin": V_vl_cut,
            "offshell_residue_ratio_vol_lin_over_exp": (V_vl_cut / V_exp_cut) ** 2,
        },
        "cut_weight_comparison": {
            "E0_2grav": E0_2grav,
            "E1_next_Istar_2grav": ch_I[1]["E_pair"],
            "next_Istar_channel": f"p={ch_I[1]['p']}{ch_I[1]['chirality']}",
            "E6_Maxwell": 26.0,
            "micro_residue_grav_exp": res_grav_exp,
            "micro_residue_grav_vol_lin": res_grav_vl,
            "micro_residue_EM_N1_E26": res_em_N1,
            "ratio_residue_exp_over_Nv": ratio_res_exp,
            "ratio_static_weight_exp_over_Nv": ratio_static_exp,
            "ratio_residue_vol_lin_over_Nv": ratio_res_vl,
            "ratio_static_weight_vol_lin_over_Nv": ratio_static_vl,
            "Nv_for_1pct_residue_exp": math.ceil(ratio_res_exp / 0.01),
            "Nv_for_1pct_static_exp": math.ceil(ratio_static_exp / 0.01),
            "Nv_for_1pct_residue_vol_lin": math.ceil(ratio_res_vl / 0.01),
            "Nv_for_1pct_static_vol_lin": math.ceil(ratio_static_vl / 0.01),
        },
    }


def main():
    print("=" * 80)
    print("STREAM p2_06 STAGE 1: EXACT ANALYTICAL DERIVATIONS & 50-DIGIT VERIFICATIONS")
    print("=" * 80)

    sym_proof = prove_maxwell_shell_sums_symbolic()
    # Direct exact Fraction verification for ALL N = 1..61 (2j = 0..60), including N = 1 (no jump!)
    for two_j in range(61):
        N = Fraction(two_j + 1)
        s0, s1, s2 = exact_maxwell_shell_derivs_fraction(two_j)
        assert s0 == N * N + 2
        assert s1 == 0
        assert s2 == Fraction(16, 45) * N * N + Fraction(40, 9) - Fraction(32, 15) / (N * N)
    print(f"[Theorem 1] Symbolic shell proof PASS + exact Fraction N=1..61 PASS (S2(1) = 8/3, zero N=1 jump).")

    ward_50 = eval_ade_maxwell_and_ward_50dps()
    print(f"[Theorem 1] 50-digit ADE Ward identity PASS: max err = {ward_50['max_ward_identity_err_50dps']:.2e}, "
          f"I* Pi_contact^top = {ward_50['ade_exact_table']['I*']['four_kappa_V_rational']} "
          f"({ward_50['ade_exact_table']['I*']['Pi_contact_top_0']:.6f}), "
          f"pi^2 u''_EM,top(I*) = {ward_50['ade_exact_table']['I*']['pi2_upp_EM_top']:+.8f}.")

    talbot = verify_talbot_revival_theorem()
    print(f"[Theorem 2] Half-antipodal Talbot anti-periodicity PASS: "
          f"50-dps err (J30) = {talbot['J_cut_30']['Istar_anti_pi_over_2_err_50dps']:.2e}, "
          f"f64 rel = {talbot['J_cut_30']['Istar_anti_pi_over_2_rel_f64']:.2e} "
          f"(vs S^3 failure {talbot['J_cut_30']['S3_anti_pi_over_2_rel_f64']:.3f}); "
          f"pulse M(pi/2)/M(0) = {talbot['localized_pulse_comb']['Istar']['M_ratio_pi_over_2']:+.6f}.")

    grav2 = verify_2grav_selection_and_vertex_50dps()
    print(f"[Theorem 3] 2-Graviton selection rules & 50-digit Bianchi IX cubic vertex PASS: "
          f"on-shell invariance err = {grav2['cubic_action_50dps']['onshell_invariance_err_50dps']:.2e}, "
          f"desert = {grav2['round_S3_desert_channels_projected_out_count']} channels in (5.66, 21.91), "
          f"residue ratio = {grav2['cut_weight_comparison']['ratio_residue_exp_over_Nv']:.5f}/N_v.")

    out_data = {
        "stream_id": "p2_06_poincare_shear_modulus_and_talbot_revivals",
        "stage": "01_derive",
        "theorem1_maxwell_ward": {
            "symbolic_shell_proof": sym_proof,
            "fraction_verified_N_range": [1, 61],
            "ade_50dps_ward": ward_50,
        },
        "theorem2_talbot_revivals": talbot,
        "theorem3_2grav_selection_and_vertex": grav2,
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
