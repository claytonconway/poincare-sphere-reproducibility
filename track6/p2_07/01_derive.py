#!/usr/bin/env python3
"""
Stream p2_07_complete_model_b_kam_dynamics (CLM-P2-07)
Stage 1 (Round 2 Post-Audit): Exact Analytical Derivations, High-Precision (35-digit & 50-digit
         mpmath + Exact Fraction) Identity Verification, & Reproducible Complete Model B Parseval/MR
         Rate Tables.

Round-2 Cold-Audit Remediations Incorporated:
  - [P0-1] Explicit multiplicity factor g_2 included in Delta_L_{Cas,2} and its Weyl check.
  - [P0-3] Canonical kinetic-normalized Misner exponential map lam_B = exp(sqrt(3)*b_2) adopted as
           primary in Theorem 2 (with exact c_2^exp verified to 50 digits AND compared against
           linear lam_B = 1 + sqrt(3)*b_2 to quantify the exact 2.20x O(b_2^2) parameterization
           sensitivity, plus exact axisymmetric cubic curvature error (10/(3*sqrt(3)))*|b_2|).
  - [P1-2] Consistent win**2 Hanning weighting on E_avg in Parseval calculator and explicit
           parametric-instability flagging on shrinking-ring rows approaching 2*Omega_2 ~ Omega_scal.
"""

from fractions import Fraction
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

STREAM_DIR = Path(__file__).resolve().parent
OUT_PATH = STREAM_DIR / "01_derive.json"

KAPPA_V_EXACT = Fraction(-44738, 93)
SIGMA_0_EXACT = Fraction(9, 31)
C_ANOM_EXACT = Fraction(9, 31)
PI_KIN_REL_EXACT = Fraction(3900, 31)
PI_REM_REL_EXACT = Fraction(984896, 465)
PI_KIN_ABS_EXACT = Fraction(4344, 31)
PI_REM_ABS_FLOAT = 2066.126432446887

KAPPA_V_F = float(KAPPA_V_EXACT)
SIGMA_0_F = float(SIGMA_0_EXACT)
C_ANOM_F = float(C_ANOM_EXACT)
X_RING_0_F = math.sqrt(44672.0 / 93.0)
PI_KIN_REL_F = float(PI_KIN_REL_EXACT)
PI_REM_REL_F = float(PI_REM_REL_EXACT)
PI_KIN_ABS_F = float(PI_KIN_ABS_EXACT)


def d_istar(two_j: int) -> int:
    if two_j < 0:
        return 0
    cls = [
        (1, 0.0), (1, math.pi), (30, math.pi / 2.0),
        (20, math.pi / 3.0), (20, 2.0 * math.pi / 3.0),
        (12, math.pi / 5.0), (12, 2.0 * math.pi / 5.0),
        (12, 3.0 * math.pi / 5.0), (12, 4.0 * math.pi / 5.0),
    ]
    tot = 0.0
    for sz, th in cls:
        if th == 0.0:
            chi = two_j + 1
        elif abs(th - math.pi) < 1e-12:
            chi = (two_j + 1) * ((-1) ** two_j)
        else:
            chi = math.sin((two_j + 1) * th) / math.sin(th)
        tot += sz * chi
    iv = int(round(tot / 120.0))
    assert abs(tot / 120.0 - iv) < 1e-9
    return iv


# ==============================================================================
# PART 1 (THEOREM 1): 35-DIGIT COMPLETE MODEL B + DISCRETE BATH NOETHER CLOSURE
# ==============================================================================

def verify_theorem1_action_and_noether_35dps():
    with mp.workdps(35):
        sig = mp.mpf(9) / 31
        lam = mp.mpf("0.013")
        kap_V = mp.mpf(-44738) / 93
        Gamma_scal = mp.mpf("0.073")
        gamma_deph = mp.mpf("0.63661977")
        g_cpl = mp.mpf("314.0")

        modes = [
            {"g": mp.mpf(5),  "kn2": mp.mpf(8),   "gam_b": mp.mpf("0.012"), "pi_kin": mp.mpf(3900) / 31, "pi_rem": mp.mpf(984896) / 465},
            {"g": mp.mpf(9),  "kn2": mp.mpf(120), "gam_b": mp.mpf("0.015"), "pi_kin": mp.mpf(0), "pi_rem": mp.mpf(0)},
            {"g": mp.mpf(17), "kn2": mp.mpf(224), "gam_b": mp.mpf("-0.004"), "pi_kin": mp.mpf(0), "pi_rem": mp.mpf(0)},
        ]
        poles = []
        for j in range(1, 31):
            d = d_istar(2 * j)
            if d > 0:
                Ep = mp.mpf(4 * j + 2)
                wp = mp.mpf(240 * d) / (2 * j + 1)
                cp = g_cpl * mp.sqrt(wp)
                poles.append((Ep, wp, cp))
        assert len(poles) == 15

        xs = mp.mpf("18.7")
        hs = mp.mpf("0.043")
        phis = mp.mpf("0.17")
        ss = mp.mpf("-0.29")
        bvs = [
            (mp.mpf("0.08"), mp.mpf("-0.012")),
            (mp.mpf("0.02"), mp.mpf("0.011")),
            (mp.mpf("0.004"), mp.mpf("0.007")),
        ]
        qps = [
            (mp.mpf(str(0.001 * (i + 1) * ((-1) ** i))), mp.mpf(str(0.002 * (i + 2) * ((-1) ** (i + 1)))))
            for i in range(len(poles))
        ]
        krad = mp.mpf("12.5")

        # P0-1 fix: include explicit multiplicity factor g_2 in Delta_L_Cas2
        g2_m, pk2, pr2 = modes[0]["g"], modes[0]["pi_kin"], modes[0]["pi_rem"]
        b2_0, v2_0 = bvs[0]

        def dL_cas2(N, x):
            return g2_m * (-pk2 * x * (v2_0 ** 2) / N - pr2 * N * (b2_0 ** 2) / x)

        def L_bath_fn(N, x):
            S_val = phis * b2_0
            tot = mp.mpf(0)
            for (Ep, _, cp), (qp, qdp) in zip(poles, qps):
                tot += mp.mpf("0.5") * ((x / N) * (qdp ** 2) - (N / x) * (Ep ** 2) * ((qp - cp * S_val / (Ep ** 2)) ** 2))
            return tot

        weyl_cas2_err = abs(
            mp.diff(lambda N: dL_cas2(N, xs), mp.mpf(1)) + xs * mp.diff(lambda x: dL_cas2(mp.mpf(1), x), xs)
        )
        weyl_bath_err = abs(
            mp.diff(lambda N: L_bath_fn(N, xs), mp.mpf(1)) + xs * mp.diff(lambda x: L_bath_fn(mp.mpf(1), x), xs)
        )

        def eval_E00_and_bath(x, h, phi, s, bv, qp_list, kap_rad):
            inv_x2 = 1 / (x * x)
            y = h * h + inv_x2
            e00 = y - lam - y * y + sig * (phi * (2 * y - phi) + 2 * h * s) - (kap_V + kap_rad) * inv_x2 * inv_x2
            for m, (b, v) in zip(modes, bv):
                g, kn2, pk, pr = m["g"], m["kn2"], m["pi_kin"], m["pi_rem"]
                e00 -= (g / 3) * (
                    (1 + 2 * sig * phi - 6 * h * h + mp.mpf(2) / 3 * inv_x2 - pk * inv_x2) * v * v
                    + inv_x2 * (kn2 * (1 + 2 * sig * phi - 2 * h * h + mp.mpf(2) / 3 * inv_x2) + pr * inv_x2) * b * b
                    - 8 * kn2 * h * inv_x2 * b * v
                )
            S_val = phi * bv[0][0]
            H_bath = mp.mpf(0)
            for (Ep, _, cp), (qp, qdp) in zip(poles, qp_list):
                H_bath += mp.mpf("0.5") * (x * x * qdp * qdp + (Ep ** 2) * ((qp - cp * S_val / (Ep ** 2)) ** 2))
            kap_bath = H_bath / 3
            e00 -= kap_bath * inv_x2 * inv_x2
            return e00, kap_bath

        def eval_rhs(x, h, phi, s, bv, qp_list, kap_rad):
            inv_x2 = 1 / (x * x)
            x4 = x ** 4
            y = h * h + inv_x2
            b2, v2 = bv[0]
            S_val = phi * b2

            F_bath = mp.mpf(0)
            for (Ep, _, cp), (qp, _) in zip(poles, qp_list):
                F_bath += cp * (qp - cp * S_val / (Ep ** 2))

            tt_r_shift = sum(m["g"] * (v * v - m["kn2"] * inv_x2 * b * b) for m, (b, v) in zip(modes, bv))
            delta_dh = -tt_r_shift / 3 + sig * Gamma_scal * s - (F_bath * b2) / (6 * sig * x4)
            dh = phi - 2 * h * h - inv_x2 + delta_dh

            A_R = 1 + 2 * sig * phi - 2 * h * h + mp.mpf(2) / 3 * inv_x2
            K_R = 1 + 2 * sig * phi + 2 * h * h + mp.mpf(2) / 3 * inv_x2 - 4 * (dh + h * h)
            dA_R = 2 * sig * s - 4 * h * dh - mp.mpf(4) / 3 * h * inv_x2

            dv_list = []
            tr_tt = mp.mpf(0)
            dkap_tt = mp.mpf(0)
            for idx, (m, (b, v)) in enumerate(zip(modes, bv)):
                g, kn2, gam_b, pk, pr = m["g"], m["kn2"], m["gam_b"], m["pi_kin"], m["pi_rem"]
                A_eff = A_R - pk * inv_x2
                dA_eff = dA_R + 2 * pk * h * inv_x2
                bath_acc = (F_bath * phi) / (2 * g * x4 * A_eff) if idx == 0 else mp.mpf(0)
                a_n = -(3 * h + dA_eff / A_eff + 2 * gam_b) * v - ((kn2 * K_R + pr * inv_x2) * inv_x2 / A_eff) * b + bath_acc
                dv_list.append(a_n)
                tr_tt += g * (
                    (-2 * (1 + 2 * sig * phi) - 12 * h**2 - 4 * dh - 8 * kn2 * inv_x2) * v**2
                    + (kn2 * inv_x2) * (2 * (1 + 2 * sig * phi) - 4 * h**2 - 4 * dh) * b**2
                    - 16 * kn2 * h * inv_x2 * b * v
                    - 8 * h * v * a_n
                    - 8 * kn2 * inv_x2 * b * a_n
                )
                dkap_tt += mp.mpf(4) / 3 * g * gam_b * x4 * A_eff * v**2

            ds = -3 * h * s + (2 * lam - phi + 2 * (phi - y) * y - (1 + 2 * sig * phi - 2 * y) * delta_dh + tr_tt / 6) / sig

            dqp_list = []
            dkap_bath_rad = mp.mpf(0)
            for (Ep, _, cp), (qp, qdp) in zip(poles, qp_list):
                qddp = -(h + 2 * gamma_deph / x) * qdp - (Ep ** 2) * inv_x2 * (qp - cp * S_val / (Ep ** 2))
                dqp_list.append((qdp, qddp))
                dkap_bath_rad += (mp.mpf(2) / 3) * gamma_deph * x * qdp * qdp

            dkap = 2 * sig**2 * x4 * Gamma_scal * s**2 + dkap_tt + dkap_bath_rad
            return x * h, dh, s, ds, dv_list, dqp_list, dkap, F_bath, dkap_bath_rad

        dx, dh, dphi, ds, dv_list, dqp_list, dkap, F_bath_0, dkap_bath_rad_0 = eval_rhs(xs, hs, phis, ss, bvs, qps, krad)

        def x4E00_flow(step):
            xp, hp, pp, sp = xs + step * dx, hs + step * dh, phis + step * dphi, ss + step * ds
            bvp = [(b + step * v, v + step * a) for (b, v), a in zip(bvs, dv_list)]
            qpp = [(q + step * qd, qd + step * qdd) for (q, qd), (_, qdd) in zip(qps, dqp_list)]
            kp = krad + step * dkap
            e00_p, _ = eval_E00_and_bath(xp, hp, pp, sp, bvp, qpp, kp)
            return (xp ** 4) * e00_p

        def bath_energy_flow(step):
            xp, hp, pp, sp = xs + step * dx, hs + step * dh, phis + step * dphi, ss + step * ds
            bvp = [(b + step * v, v + step * a) for (b, v), a in zip(bvs, dv_list)]
            qpp = [(q + step * qd, qd + step * qdd) for (q, qd), (_, qdd) in zip(qps, dqp_list)]
            kp = krad + step * dkap
            _, kb_p = eval_E00_and_bath(xp, hp, pp, sp, bvp, qpp, kp)
            return kb_p

        err_noether = float(abs(mp.diff(x4E00_flow, mp.mpf(0))))
        dkb_dtau = mp.diff(bath_energy_flow, mp.mpf(0))
        dS_dtau = ss * bvs[0][0] + phis * bvs[0][1]
        err_bath_balance = float(abs(dkb_dtau + dkap_bath_rad_0 + (F_bath_0 * dS_dtau) / 3))

        # Analytical calibration of g_cpl to bare p2_02 continuum-log plateau on x_ring (sigma_t0 = 9/31)
        xr_m = mp.sqrt(mp.mpf(44672) / 93)
        phi_amp = mp.sqrt(2 / (xr_m * xr_m * sig))
        gamma_rad_p202_xring = mp.mpf("3.294e-3")
        rate_p202_per_b02 = 2 * gamma_rad_p202_xring * (mp.mpf(8) / 3) * xr_m * xr_m
        g_cpl_sq_cal = rate_p202_per_b02 / (mp.pi * phi_amp * phi_amp / (24 * xr_m))
        g_cpl_cal = mp.sqrt(g_cpl_sq_cal)
        sum_w_over_E2_30 = sum(wp / (Ep ** 2) for Ep, wp, _ in poles)
        cl_counterterm_frac = float(mp.mpf("0.5") * g_cpl_sq_cal * sum_w_over_E2_30 * phi_amp * phi_amp / (8 * xr_m * xr_m))

        assert float(weyl_cas2_err) < 1e-32 and float(weyl_bath_err) < 1e-32
        assert err_noether < 1e-30 and err_bath_balance < 1e-30

        return {
            "weyl_invariance_DeltaL_Cas2_err_35dps": float(weyl_cas2_err),
            "weyl_invariance_L_bath_err_35dps": float(weyl_bath_err),
            "dissipative_noether_d_x4E00_dtau_err_35dps": err_noether,
            "bath_energy_balance_err_35dps": err_bath_balance,
            "x_kin_relS3": float(mp.sqrt(mp.mpf(3900) / 31)),
            "x_kin_absMS": float(mp.sqrt(mp.mpf(4344) / 31)),
            "g_cpl_calibrated_to_p202_xring": float(g_cpl_cal),
            "bath_counterterm_frac_of_8xring2": cl_counterterm_frac,
            "n_poles_J30": len(poles),
            "sum_w_over_E2_J30": float(sum_w_over_E2_30),
            "all_verified": True,
        }


# ==============================================================================
# PART 2 (THEOREM 2): 50-DIGIT & EXACT-FRACTION MISNER EXP vs LINEAR M_R-SPLITTING
# ==============================================================================

def verify_theorem2_mr_splitting_50dps():
    with mp.workdps(50):
        # 1. Exact Fraction CG normalization & closed-form shell moments across j = 1..30
        cg_shell_stats = []
        max_cg_norm_err = Fraction(0)
        for j in range(1, 31):
            N = 2 * j + 1
            den = 8 * (N - 2) * (N - 1) * N * (N + 1) * (N + 2)
            s_p = Fraction(0)
            s_a2 = Fraction(0)
            s_a4 = Fraction(0)
            for M_R in range(-(j - 1), j):
                a = 2 * M_R
                p = Fraction(15 * ((N - 1) ** 2 - a ** 2) * ((N + 1) ** 2 - a ** 2), den)
                s_p += p
                s_a2 += p * (a ** 2)
                s_a4 += p * (a ** 4)
            assert s_p == Fraction(1)
            max_cg_norm_err = max(max_cg_norm_err, abs(s_p - 1))
            a2_closed = Fraction(N * N - 9, 7)
            assert s_a2 == a2_closed
            mean_c1_over_sqrt3 = Fraction(2, 3 * N) * (N * N + 3 - 3 * s_a2)
            mean_c1_closed = Fraction(8 * (N * N + 12), 21 * N)
            assert mean_c1_over_sqrt3 == mean_c1_closed
            var_c1_exact = Fraction(12, N * N) * (s_a4 - s_a2 * s_a2)

            if d_istar(2 * j) > 0 or j in (1, 2, 3):
                cg_shell_stats.append({
                    "j": j, "N": N, "E_round": 2 * N, "d_Istar": d_istar(2 * j),
                    "n_MR_states": 2 * j - 1,
                    "mean_a2_exact": str(s_a2),
                    "mean_c1": float(mean_c1_closed) * math.sqrt(3.0),
                    "std_c1": math.sqrt(float(var_c1_exact)),
                    "std_c1_over_E": math.sqrt(float(var_c1_exact)) / (2.0 * N),
                })

        # 2. 50-digit Cauchy extraction for BOTH canonical Misner exp(sqrt(3)*b_2) AND linear 1 + sqrt(3)*b_2
        def cauchy_012(fn_c, rad=mp.mpf("0.01"), n_pts=48):
            cs = [mp.mpc(0)] * 3
            for m in range(n_pts):
                z = rad * mp.exp(2j * mp.pi * m / n_pts)
                fz = fn_c(z)
                for k in range(3):
                    cs[k] += fz / (z ** k)
            return [mp.re(c / n_pts) for c in cs]

        s3 = mp.sqrt(3)
        max_err_exp = [mp.mpf(0), mp.mpf(0), mp.mpf(0)]
        max_err_lin = [mp.mpf(0), mp.mpf(0), mp.mpf(0)]
        test_pairs = [(1, 0), (6, 0), (6, 3), (6, 5), (10, 4), (12, 9), (15, 7), (20, 14), (30, 21)]
        for j, M_R in test_pairs:
            N = mp.mpf(2 * j + 1)
            a = mp.mpf(2 * M_R)
            R1 = (1 - a * a) / N
            R2 = (1 + 3 * a * a) / N - ((1 - a * a) ** 2) / (N ** 3)
            c0_ex = 2 * N
            c1_ex = 2 * (N * N + 3 - 3 * a * a) / (s3 * N)
            c2_lin_ex = 6 * (-N / 9 + R1 / 3 + R2 / 2)
            # Misner lam_B = exp(sqrt(3)*b_2) = 1 + sqrt(3)*b_2 + (3/2)*b_2^2 + ...
            # shifts c_2 by +(3/2)*(dE/dlam_B|_1) = +(sqrt(3)/2)*c_1 = (N^2 + 3 - 3*a^2)/N:
            c2_exp_ex = c2_lin_ex + (N * N + 3 - 3 * a * a) / N
            # Verify simplified closed form of c2_exp:
            #   c2_exp = (1/3)*N + (8 + 4*a^2)/N - 3*(1 - a^2)^2 / N^3
            c2_exp_simp = N / 3 + (8 + 4 * a * a) / N - 3 * ((1 - a * a) ** 2) / (N ** 3)
            assert abs(c2_exp_ex - c2_exp_simp) < mp.mpf("1e-48")

            def E_exp(z):
                lam = mp.exp(s3 * z)
                return 2 * (lam ** (mp.mpf(1) / 3)) * mp.sqrt(lam * lam + N * N - 1 + (1 / (lam * lam) - 1) * a * a)

            def E_lin(z):
                lam = 1 + s3 * z
                return 2 * (lam ** (mp.mpf(1) / 3)) * mp.sqrt(lam * lam + N * N - 1 + (1 / (lam * lam) - 1) * a * a)

            ce = cauchy_012(E_exp)
            cl = cauchy_012(E_lin)
            for k, ex_v in enumerate((c0_ex, c1_ex, c2_exp_ex)):
                max_err_exp[k] = max(max_err_exp[k], abs(ce[k] - ex_v))
            for k, ex_v in enumerate((c0_ex, c1_ex, c2_lin_ex)):
                max_err_lin[k] = max(max_err_lin[k], abs(cl[k] - ex_v))

        assert max(max_err_exp + max_err_lin) < mp.mpf("1e-44")

        # 3. Exact large-N continuum beta-integral CG moments of c_2^exp/N = 1/3 + 4 u^2 - 3 u^4 vs c_2^lin/N = -2/3 + 7 u^2 - 3 u^4
        #    On weight (15/16)*(1 - u^2)^2 on [-1, 1]: <1>=1, <u^2>=1/7, <u^4>=1/21, <u^6>=1/42, <u^8>=1/77?
        #    Check exact beta moments: int_0^1 u^(2k) (1-u^2)^2 du / int_0^1 (1-u^2)^2 du
        #      = (1/(2k+1) - 2/(2k+3) + 1/(2k+5)) / (8/15) = 15 / ((2k+1)(2k+3)(2k+5))!
        def u_mom(two_k: int) -> Fraction:
            return Fraction(15, (two_k + 1) * (two_k + 3) * (two_k + 5))

        m2, m4, m6, m8 = u_mom(2), u_mom(4), u_mom(6), u_mom(8)
        assert m2 == Fraction(1, 7) and m4 == Fraction(1, 21)

        def poly_mean_var(c0, c2, c4):
            mu = c0 + c2 * m2 + c4 * m4
            sq = c0*c0 + 2*c0*c2*m2 + (c2*c2 + 2*c0*c4)*m4 + 2*c2*c4*m6 + c4*c4*m8
            return mu, sq - mu * mu

        mu_exp, var_exp = poly_mean_var(Fraction(1, 3), Fraction(4), Fraction(-3))
        mu_lin, var_lin = poly_mean_var(Fraction(-2, 3), Fraction(7), Fraction(-3))
        std_exp = math.sqrt(float(var_exp))
        std_lin = math.sqrt(float(var_lin))

        # 4. Exact axisymmetric Bianchi IX Misner curvature expansion:
        #    R^(3)(lam_B) = 2*(4 - lam_B^2)*lam_B^(2/3) = 8*exp(2*b/sqrt(3)) - 2*exp(8*b/sqrt(3))
        #                 = 6 - 16*b^2 - (160/(3*sqrt(3)))*b^3 + ...
        def R3_axi(z):
            lam = mp.exp(s3 * z)
            return 2 * (4 - lam * lam) * (lam ** (mp.mpf(2) / 3))

        # Extract cubic term via 4-term Cauchy contour
        rad_r = mp.mpf("0.01")
        c3_axi = mp.re(sum(R3_axi(rad_r * mp.exp(2j * mp.pi * m / 48)) / ((rad_r * mp.exp(2j * mp.pi * m / 48)) ** 3) for m in range(48)) / 48)
        c3_axi_ex = -mp.mpf(160) / (3 * s3)
        axi_cubic_err = float(abs(c3_axi - c3_axi_ex))
        assert axi_cubic_err < 1e-44
        axi_cubic_ratio_coeff = float(mp.mpf(10) / (3 * s3))  # 1.924500897...

        return {
            "cg_norm_exact_err_j1_to_30": int(max_cg_norm_err.numerator),
            "exact_closed_form_mean_a2": "(N^2 - 9) / 7",
            "exact_closed_form_mean_c1": "(8 / (7 * sqrt(3))) * (N + 12/N)",
            "large_N_std_c1_over_E": float(Fraction(2, 7)),
            "misner_exp_cauchy_50dps_errs": [float(e) for e in max_err_exp],
            "linear_lam_cauchy_50dps_errs": [float(e) for e in max_err_lin],
            "large_N_c2_exp_mean_over_N": str(mu_exp),
            "large_N_c2_exp_std_over_N": std_exp,
            "large_N_c2_lin_mean_over_N": str(mu_lin),
            "large_N_c2_lin_std_over_N": std_lin,
            "c2_lin_over_exp_std_ratio": std_lin / std_exp,
            "axisymmetric_R3_cubic_coeff_exact": "-160 / (3 * sqrt(3))",
            "axisymmetric_R3_cubic_err_50dps": axi_cubic_err,
            "axisymmetric_cubic_rel_error_coeff_10_over_3sqrt3": axi_cubic_ratio_coeff,
            "axisymmetric_cubic_rel_err_at_b0p10": axi_cubic_ratio_coeff * 0.10,
            "axisymmetric_cubic_rel_err_at_b0p15": axi_cubic_ratio_coeff * 0.15,
            "shell_stats_Istar": [s for s in cg_shell_stats if s["d_Istar"] > 0],
            "all_verified": True,
        }


# ==============================================================================
# PART 3: REPRODUCIBLE COMPLETE MODEL B SHRINKING-RING PARSEVAL/MR RATE TABLES
# ==============================================================================

def compute_modelB_parseval_mr_rate(sigma_t, kap_rad, pi_kin, pi_rem, M_thresh=26.0, eta_end=40.0, N_grid=32768):
    """
    Self-contained Parseval 2-photon flux & Manley-Rowe graviton damping calculator for n=2
    under Complete Model B (Pi_kin, Pi_rem) on shrinking ring (sigma_t, kap_rad).
    P1-2 fix: uses consistent win**2 Hanning weighting on E_avg and explicitly detects
    parametric growth (amp_ratio > 2.0) when 2*Omega_2 approaches Omega_scal!
    """
    kap_eff = 1.0 - sigma_t + KAPPA_V_F + kap_rad
    xr = math.sqrt(abs(kap_eff))
    y0_bg = 1.0 / (xr * xr)
    hdot_b = math.sqrt((y0_bg - kap_eff * y0_bg * y0_bg) / sigma_t)
    y0 = [xr, 0.0, y0_bg + hdot_b, 0.0, 0.0]

    def rhs_bg(eta, y):
        x, h, r, s, _ = y
        inv_x2 = 1.0 / (x * x)
        yy = h * h + inv_x2
        return [x * x * h, x * (r - 2.0 * h * h - inv_x2), x * s, x * (-3.0 * h * s + (-r + 2.0 * (r - yy) * yy) / sigma_t), x]

    bg = solve_ivp(rhs_bg, (0.0, eta_end), y0, method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True, max_step=0.02)
    kn2 = 8.0

    def rhs_0(eta, y):
        st = bg.sol(eta)
        x, h, r = float(st[0]), float(st[1]), float(st[2])
        inv_x2 = 1.0 / (x * x)
        yy = h * h + inv_x2
        A_R = 1.0 + 2.0 * sigma_t * r - 2.0 * h * h + (2.0 / 3.0) * inv_x2
        K_R = 1.0 + 2.0 * sigma_t * r + 2.0 * h * h + (2.0 / 3.0) * inv_x2 - 4.0 * (r - yy)
        K2_eff = x * x * A_R - pi_kin
        S2_eff = kn2 * x * x * K_R + pi_rem
        return [y[1] / K2_eff, -S2_eff * y[0]]

    etas = np.linspace(0.0, eta_end, N_grid)
    d_eta = etas[1] - etas[0]
    st = bg.sol(etas)
    xs, hs, rs, taus = st[0], st[1], st[2], st[4]
    inv_x2 = 1.0 / (xs * xs)
    yy = hs * hs + inv_x2
    A_R = 1.0 + 2.0 * sigma_t * rs - 2.0 * hs * hs + (2.0 / 3.0) * inv_x2
    K_R = 1.0 + 2.0 * sigma_t * rs + 2.0 * hs * hs + (2.0 / 3.0) * inv_x2 - 4.0 * (rs - yy)
    K2_eff = xs * xs * A_R - pi_kin
    S2_eff = kn2 * xs * xs * K_R + pi_rem
    if float(np.min(K2_eff)) <= 0.5:
        return None

    x_mean = float(taus[-1] / eta_end)
    Om_s = x_mean / math.sqrt(sigma_t)
    wn = float(np.mean(np.sqrt(S2_eff / K2_eff)))

    win = np.hanning(N_grid)
    win2 = win ** 2
    win_norm2 = float(np.sum(win2)) * d_eta
    win2_sum = float(np.sum(win2))

    gt_l, gb_l, g1_l, amp_ratios = [], [], [], []
    for y0_2 in ([1.0, 0.0], [0.0, float(K2_eff[0]) * wn]):
        sol0 = solve_ivp(rhs_0, (0.0, eta_end), y0_2, method="DOP853", rtol=1e-11, atol=1e-13, t_eval=etas, max_step=0.02)
        b_arr = sol0.y[0]
        bp_arr = sol0.y[1] / K2_eff
        E_inst = K2_eff * bp_arr**2 + S2_eff * b_arr**2
        # P1-2 fix: consistent win**2 weighting on E_avg matching Hanning-windowed Parseval spectrum!
        E_avg = float(np.sum(win2 * E_inst) / win2_sum)
        amp_ratios.append(float(np.max(np.abs(b_arr[-N_grid // 4:])) / max(1e-30, float(np.max(np.abs(b_arr[:N_grid // 4]))))))

        bf = np.fft.rfft(b_arr * win) * d_eta
        freqs = 2.0 * np.pi * np.fft.rfftfreq(N_grid, d=d_eta)
        dOm = freqs[1] - freqs[0]
        Bn = (freqs**2 - 4.0) * (freqs**2 - 16.0)
        spec = (4.0 * C_ANOM_F / win_norm2) * freqs * Bn * (np.abs(bf) ** 2) * dOm
        m_tot = freqs > M_thresh
        gt_l.append(float(np.sum(spec[m_tot])) / (2.0 * E_avg * x_mean))

        P_grav = 0.0
        P_1st = 0.0
        claimed = np.zeros(len(freqs), dtype=bool)
        hw = min(0.45 * wn, 2.5)
        for m_h in [1, 2, 3]:
            c_up = m_h * Om_s + wn
            mu = (~claimed) & (freqs > M_thresh) & (np.abs(freqs - c_up) < hw)
            ml = np.zeros(len(freqs), dtype=bool)
            if m_h * Om_s > wn:
                c_lo = m_h * Om_s - wn
                ml = (~claimed) & (freqs > M_thresh) & (np.abs(freqs - c_lo) < hw)
            assert not np.any(mu & ml)
            if np.any(mu):
                P_grav += float(np.sum((wn / freqs[mu]) * spec[mu]))
            if np.any(ml):
                P_grav -= float(np.sum((wn / freqs[ml]) * spec[ml]))
            if m_h == 1:
                P_1st = float(np.sum(spec[mu | ml]))
            claimed |= (mu | ml)
        gb_l.append(P_grav / (2.0 * E_avg * x_mean))
        g1_l.append(P_1st / (2.0 * E_avg * x_mean))

    max_amp_ratio = max(amp_ratios)
    return {
        "kappa_rad": float(kap_rad),
        "x_ring_eff": float(xr),
        "x_mean": x_mean,
        "Omega_scal": Om_s,
        "Omega_2_eff": wn,
        "min_K2_eff": float(np.min(K2_eff)),
        "max_basis_amp_ratio": max_amp_ratio,
        "parametric_unstable_flagged": bool(max_amp_ratio > 2.0),
        "gamma_rad_tot": float(np.mean(gt_l)),
        "gamma_rad_1st_harm": float(np.mean(g1_l)),
        "gamma_b_MR": float(np.mean(gb_l)),
        "delta_lam_1_MR": -float(np.mean(gb_l)),
    }


def compute_modelB_shrinking_tables():
    tables = {}
    for label, pk, pr in [
        ("ModelB_relS3", PI_KIN_REL_F, PI_REM_REL_F),
        ("ModelB_absMS", PI_KIN_ABS_F, PI_REM_ABS_FLOAT),
    ]:
        print(f"\nComputing Complete Model B shrinking-ring Parseval/MR table ({label}) at sigma_t0 = 9/31...", flush=True)
        rows = []
        for kr in [0.0, 60.0, 120.0, 160.0, 190.0, 210.0, 225.0, 240.0, 260.0, 280.0, 300.0, 315.0]:
            r = compute_modelB_parseval_mr_rate(SIGMA_0_F, kr, pk, pr, M_thresh=26.0, eta_end=40.0, N_grid=32768)
            if r is None:
                print(f"  kr={kr:5.1f}: stopped (K2_eff approaches ghost pole)", flush=True)
                break
            rows.append(r)
            fl = " [PARAMETRIC_UNSTABLE]" if r["parametric_unstable_flagged"] else ""
            print(
                f"  x={r['x_ring_eff']:6.2f} (kr={kr:5.1f}, Om_s={r['Omega_scal']:5.2f}, Om_2={r['Omega_2_eff']:5.2f}, amp={r['max_basis_amp_ratio']:5.2f}): "
                f"g_tot={r['gamma_rad_tot']:.4e} | gam_b_MR={r['gamma_b_MR']:+.4e}{fl}",
                flush=True,
            )
        tables[label] = rows
    return tables


def main():
    print("=== STREAM p2_07 (CLM-P2-07) STAGE 1 (ROUND 2): EXACT DERIVATIONS & MODEL B PARSEVAL TABLES ===", flush=True)
    th1 = verify_theorem1_action_and_noether_35dps()
    print(f"  [Theorem 1] 35-digit Noether d/dtau(x^4 E_00) = {th1['dissipative_noether_d_x4E00_dtau_err_35dps']:.2e}, "
          f"bath balance = {th1['bath_energy_balance_err_35dps']:.2e}, g_cpl_cal = {th1['g_cpl_calibrated_to_p202_xring']:.2f}", flush=True)

    th2 = verify_theorem2_mr_splitting_50dps()
    print(f"  [Theorem 2] CG norm err = {th2['cg_norm_exact_err_j1_to_30']}, "
          f"Misner exp 50-dps errs = {[f'{e:.1e}' for e in th2['misner_exp_cauchy_50dps_errs']]}, "
          f"c2_lin/c2_exp std ratio = {th2['c2_lin_over_exp_std_ratio']:.4f}, "
          f"axi cubic err @ 0.10 = {100*th2['axisymmetric_cubic_rel_err_at_b0p10']:.2f}%", flush=True)

    sr_tables = compute_modelB_shrinking_tables()

    payload = {
        "stream_id": "p2_07_complete_model_b_kam_dynamics",
        "stage": "01_derive",
        "round": 2,
        "conventions": {
            "signature": "(-,+,+,+)",
            "units": "8*pi*G = M_Pl^-2 = 1, c = 1, H_S = 1",
            "kappa_V": str(KAPPA_V_EXACT),
            "sigma_0": str(SIGMA_0_EXACT),
            "c_anom": str(C_ANOM_EXACT),
            "Pi_kin_relS3": str(PI_KIN_REL_EXACT),
            "Pi_rem_relS3": str(PI_REM_REL_EXACT),
            "Pi_kin_absMS": str(PI_KIN_ABS_EXACT),
            "Pi_rem_absMS": PI_REM_ABS_FLOAT,
            "misner_berger_map": "lam_B = exp(sqrt(3) * b_2)",
        },
        "theorem1_modelB_bath_noether_35dps": th1,
        "theorem2_berger_mr_splitting_50dps": th2,
        "modelB_shrinking_ring_parseval_tables": sr_tables,
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
