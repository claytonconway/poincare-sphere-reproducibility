#!/usr/bin/env python3
"""
Phase 2 Stream p2_04_graviton_invariance_and_berger_stability:
Stage 1 Exact Derivations & Symbolic/50-Digit Identities (01_derive.py, Round-2 Audited)

Covers:
  KT1A: Exact 1-parameter metric field-redefinition invariance of 1-loop graviton
        backreaction across all 6 values alpha in [-1, 1.5] verified against exact
        nonlinear 4D Bianchi IX (K=+1, k^2=8) and 1+1D Gowdy (K=0, k in {1,3,7})
        Einstein tensors (50-digit mpmath).
  KT1B: Exact 2nd-order Sachs optical-scalar & null-geodesic verification:
        (i) exact Sachs-Einstein identity -R_uv^(2)[g] k^u k^v === sigma^2 + d(delta_theta^(2)_alpha)/dlambda
            for conformal null tangent k = d/d(eta) + d/dz (flat space & flat dS_4, arbitrary alpha),
            proving zero double-counting of gravitational-wave shear focusing;
        (ii) exact general-angle (theta, phi) nonlinear 4D Gowdy fixed-endpoint null-geodesic
            integration (including phi = pi/4 Schur-complement shear and oblique theta),
            verifying <Delta_chi / (L <h^2>)>_{S^2} === (1 - alpha)/6 = (-alpha/6 + 1/20 + 1/10 + 1/40 - 1/120);
        (iii) exact nonlinear curved-S^3 = SU(2) triaxial Bianchi IX (p = 3) null-geodesic
            integration across alpha in {0, 0.5, 1.0}, verifying exact -alpha/6 cancellation
            and Schur-complement baseline (-alpha/6 + 3/20) to 4.7e-4.
  KT1C: 50-digit global dS_4/Gamma parameterization-invariant totals across all 5 ADE space forms
        for both the exact Schur-complement geodesic baseline (Delta_eta_BR^FP + (1/15) int <h^2>)
        and the full 2nd-order WKB geodesic total (Delta_eta_BR^FP + (1/12) int <h^2>),
        contrasted with the straight-coordinate-line value (-1/30) and settling N_V^crit(I*) in {5, 6},
        plus the local 2D screen-area determinant identity at fixed (eta_w, chi_w).
  KT2:  Exact symbolic (SymPy) + matrix proof of Berger-sphere spin-2 TT Lichnerowicz stability:
        (i) exact stretched |M_L|=j+2 dispersion (all 2j >= 0, all lam_B > 0),
        (ii) exact oblate (j=0, M_L=0) dispersion (lam_oblate = 1/2),
        (iii) exact all-lam_B > 0 positivity of M_L = 0 for all integer j >= 1 + exact (-1)^j D_m* parity,
        (iv) exact sub-stretched |M_L|=j+1 positivity for all j >= 1, lam_B >= 1 + unique isolated
             tachyon at (j=1/2, |M_L|=3/2) crossing at lam_B = 3.243808,
        (v) finite-cutoff non-stretched scan across 2j <= 40 on BOTH oblate lam_B in [0.51, 0.95]
            AND prolate lam_B in [1.2, 6.0], full-matrix vs block check, and 16-quotient table.
"""

import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.linalg import expm
from scipy.optimize import brentq

mp.mp.dps = 50

STREAM_DIR = Path(__file__).resolve().parent
STREAM_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PART 1A: EXACT NONLINEAR 4D EINSTEIN TENSORS FOR GENERAL alpha IN R
# =============================================================================
print("=" * 78)
print("PART 1A: EXACT 1-PARAMETER METRIC REDEFINITION g_ij^(alpha) = a^2(gamma + h + alpha h^2)")
print("=" * 78)

c_axis_mp = [1 / mp.sqrt(6), 1 / mp.sqrt(6), -2 / mp.sqrt(6)]


def bianchi9_G_alpha(eta_val, h_val, hp_val, hpp_val, eps_val, alpha_val):
    a = 1 / mp.cos(eta_val)
    H = mp.tan(eta_val)
    Hp = H * H + 1
    ais, Kis, Kidots = [], [], []
    for ci in c_axis_mp:
        u = eps_val * ci * h_val
        up = eps_val * ci * hp_val
        upp = eps_val * ci * hpp_val
        f = 1 + u + alpha_val * u * u
        fp = (1 + 2 * alpha_val * u) * up
        fpp = (1 + 2 * alpha_val * u) * upp + 2 * alpha_val * up * up
        ai = a * mp.sqrt(f)
        ln_p = H + mp.mpf("0.5") * fp / f
        ln_pp = Hp + mp.mpf("0.5") * (fpp * f - fp * fp) / (f * f)
        Ki = ln_p / a
        Ki_dot = (ln_pp - H * ln_p) / (a * a)
        ais.append(ai)
        Kis.append(Ki)
        Kidots.append(Ki_dot)

    a1, a2, a3 = ais
    K1, K2, K3 = Kis
    K1d, K2d, K3d = Kidots
    R3 = 2 * (2 * (a1**2 * a2**2 + a2**2 * a3**2 + a3**2 * a1**2) - (a1**4 + a2**4 + a3**4)) / (a1**2 * a2**2 * a3**2)
    neg_G00 = (K1 * K2 + K2 * K3 + K3 * K1) + mp.mpf("0.5") * R3
    rho_eff = -neg_G00 + 3
    Gii_sum = -2 * (K1d + K2d + K3d) - 2 * (K1**2 + K2**2 + K3**2) - (K1 * K2 + K2 * K3 + K3 * K1) - mp.mpf("0.5") * R3
    P_eff = -(Gii_sum + 9) / 3
    return rho_eff, P_eff


def gowdy_G_alpha(eta_val, z_val, k_val, h_val, hp_val, hpp_val, eps_val, alpha_val, dS=True):
    if dS:
        a = -1 / eta_val
        H = a
        Hp = H * H
        Lam = mp.mpf(3)
    else:
        a = mp.mpf(1)
        H = mp.mpf(0)
        Hp = mp.mpf(0)
        Lam = mp.mpf(0)
    a2 = a * a
    cz, sz = mp.cos(k_val * z_val), mp.sin(k_val * z_val)
    W, Wt, Wtt = eps_val * h_val * cz, eps_val * hp_val * cz, eps_val * hpp_val * cz
    Wz, Wzz = -eps_val * k_val * h_val * sz, -eps_val * (k_val**2) * h_val * cz

    def u_derivs(sign):
        w, wt, wtt, wz, wzz = sign * W, sign * Wt, sign * Wtt, sign * Wz, sign * Wzz
        f = 1 + w + alpha_val * w * w
        ft = (1 + 2 * alpha_val * w) * wt
        ftt = (1 + 2 * alpha_val * w) * wtt + 2 * alpha_val * wt * wt
        fz = (1 + 2 * alpha_val * w) * wz
        fzz = (1 + 2 * alpha_val * w) * wzz + 2 * alpha_val * wz * wz
        return (
            mp.mpf("0.5") * ft / f,
            mp.mpf("0.5") * (ftt * f - ft * ft) / (f * f),
            mp.mpf("0.5") * fz / f,
            mp.mpf("0.5") * (fzz * f - fz * fz) / (f * f),
        )

    u1t, u1tt, u1z, u1zz = u_derivs(+1)
    u2t, u2tt, u2z, u2zz = u_derivs(-1)
    f0t, f0tt = H, Hp
    f1t, f1tt, f1z, f1zz = H + u1t, Hp + u1tt, u1z, u1zz
    f2t, f2tt, f2z, f2zz = H + u2t, Hp + u2tt, u2z, u2zz
    neg_G00 = ((f1t * f2t + f0t * (f1t + f2t)) - (f1zz + f2zz + f1z**2 + f2z**2 + f1z * f2z)) / a2
    G33 = (-(f1tt + f2tt + f1t**2 + f2t**2 + f1t * f2t - f0t * (f1t + f2t)) + f1z * f2z) / a2
    G11 = (-(f0tt + f2tt + f2t**2) + (f2zz + f2z**2)) / a2
    G22 = (-(f0tt + f1tt + f1t**2) + (f1zz + f1z**2)) / a2
    return -neg_G00 + Lam, -(G11 + G22 + G33 + 3 * Lam) / 3


def verify_alpha_functional_identity():
    eps = mp.mpf("1e-12")
    alphas = [mp.mpf("-1"), mp.mpf("0"), mp.mpf("0.25"), mp.mpf("0.5"), mp.mpf("1"), mp.mpf("1.5")]
    max_b9_err = mp.mpf(0)
    max_gowdy_err = mp.mpf(0)
    max_bianchi_res = mp.mpf(0)

    for al in alphas:
        da = al - mp.mpf("0.5")
        for es, hs, hps in [("0.4", "1.1", "-0.7"), ("-0.6", "0.8", "1.3")]:
            ev, hv, hpv = mp.mpf(es), mp.mpf(hs), mp.mpf(hps)
            a, H = 1 / mp.cos(ev), mp.tan(ev)
            hppv = -2 * H * hpv - 8 * hv
            rp, Pp = bianchi9_G_alpha(ev, hv, hpv, hppv, eps, al)
            rm, Pm = bianchi9_G_alpha(ev, hv, hpv, hppv, -eps, al)
            rho_num = 4 * a * a * (rp + rm) / (2 * eps * eps)
            P_num = 4 * a * a * (Pp + Pm) / (2 * eps * eps)
            rho_th = mp.mpf("0.5") * hpv**2 - 8 * da * H * hv * hpv + (4 + 4 * da) * hv**2
            P_th = (mp.mpf("0.5") + mp.mpf(8) / 3 * da) * hpv**2 + (-mp.mpf(4) / 3 - mp.mpf(68) / 3 * da) * hv**2
            max_b9_err = max(max_b9_err, abs(rho_num - rho_th), abs(P_num - P_th))

            Hp = H * H + 1
            d_a2rho = hpv * hppv - 8 * da * (Hp * hv * hpv + H * hpv**2 + H * hv * hppv) + 2 * (4 + 4 * da) * hv * hpv
            max_bianchi_res = max(max_bianchi_res, abs(d_a2rho + H * rho_th + 3 * H * P_th) / (a * a))

        for kval in [1, 3, 7]:
            km = mp.mpf(kval)
            ev, hv, hpv = mp.mpf("-0.7"), mp.mpf("1.2"), mp.mpf("-0.6")
            a = -1 / ev
            H = a
            hppv = -2 * H * hpv - (km**2) * hv
            r_avg, P_avg = mp.mpf(0), mp.mpf(0)
            n_z = 8
            for j in range(n_z):
                zv = (2 * mp.pi / km) * j / n_z
                rp, Pp = gowdy_G_alpha(ev, zv, km, hv, hpv, hppv, eps, al, dS=True)
                rm, Pm = gowdy_G_alpha(ev, zv, km, hv, hpv, hppv, -eps, al, dS=True)
                r_avg += (rp + rm) / (2 * eps * eps)
                P_avg += (Pp + Pm) / (2 * eps * eps)
            rho_num = 4 * a * a * r_avg / n_z
            P_num = 4 * a * a * P_avg / n_z
            rho_th = mp.mpf("0.5") * hpv**2 - 8 * da * H * hv * hpv + mp.mpf("0.5") * (km**2) * hv**2
            P_th = (mp.mpf("0.5") + mp.mpf(8) / 3 * da) * hpv**2 + (-mp.mpf(1) / 6 - mp.mpf(8) / 3 * da) * (km**2) * hv**2
            max_gowdy_err = max(max_gowdy_err, abs(rho_num - rho_th), abs(P_num - P_th))

    print(f"  Verified across 6 values of alpha in [-1, 1.5]:")
    print(f"    Bianchi IX (K=+1, k^2=8) max abs err: {float(max_b9_err):.3e}")
    print(f"    Gowdy (K=0, k=1,3,7) max abs err:     {float(max_gowdy_err):.3e}")
    print(f"    Exact Bianchi conservation residual:  {float(max_bianchi_res):.3e}")
    assert max_b9_err < mp.mpf("1e-22")
    assert max_gowdy_err < mp.mpf("1e-21")
    assert max_bianchi_res < mp.mpf("1e-45")
    return {
        "alphas_tested": [float(a) for a in alphas],
        "bianchi9_max_err": float(max_b9_err),
        "gowdy_max_err": float(max_gowdy_err),
        "bianchi_conservation_residual_50dps": float(max_bianchi_res),
    }


kt1a_nonlin = verify_alpha_functional_identity()


# =============================================================================
# PART 1B: EXACT 2ND-ORDER SACHS SHEAR + FULL GENERAL-ANGLE GEODESICS (K=0 & K=+1)
# =============================================================================
print("\n" + "=" * 78)
print("PART 1B: SACHS-EINSTEIN IDENTITY, GENERAL-ANGLE GOWDY & CURVED-S^3 BIANCHI IX GEODESICS")
print("=" * 78)


def verify_sachs_and_geodesics():
    """
    1. Exact Sachs-Einstein identity on 1+1D Gowdy metric g = a(eta)^2 g_tilde along
       conformal null tangent k = d/d(eta) + d/dz (conformal affine parameter lambda = eta;
       for physical tangent k_phys = a^{-1} k, multiply LHS by a^2):
         -R_uv^(2)[g] k^u k^v === sigma^2 + d(delta_theta^(2)_alpha)/dlambda
       where sigma^2 = (1/2) [(d_eta + d_z) W]^2 is strictly alpha-independent, and
       delta_theta^(2)_alpha = (alpha - 1/2) (d_eta + d_z)(W^2).
       Verified across flat space (a=1) and flat dS_4 (a=-1/eta) for alpha in [-0.5, 1.25].
    2. Exact general-angle (theta, phi) nonlinear 4D Gowdy fixed-endpoint null-geodesic
       integration (explicitly testing phi = 0 and phi = pi/4 Schur-complement shear, and
       oblique cos(theta) in {0, 0.25, 0.5, 0.75, 1.0}):
       Verifies exact isotropic solid-angle average
         <Delta_chi / (L <h^2>)>_{S^2} === (1 - alpha)/6
         = (-alpha/6 + 1/20) [straight line] + (+1/10) [Schur-complement shear (1/2) h_{nA}^2]
           + (+1/40) [transverse Fermat] + (-1/120) [longitudinal-redshift cross-term].
    3. Exact nonlinear curved-S^3 = SU(2) triaxial Bianchi IX (p = 3) null-geodesic
       integration from eta = -pi/2 to +pi/2 across alpha in {0, 0.5, 1.0}:
       Verifies exact -alpha/6 cancellation and Schur-complement geodesic shift (-alpha/6 + 3/20).
    """
    eps = mp.mpf("1e-12")
    k_val = mp.mpf("3")
    max_sachs_id_err = mp.mpf(0)

    for dS in [False, True]:
        ev = mp.mpf("-0.8") if dS else mp.mpf("0.5")
        a = (-1 / ev) if dS else mp.mpf(1)
        H = a if dS else mp.mpf(0)
        hv, hpv = mp.mpf("1.1"), mp.mpf("-0.7")
        hppv = -2 * H * hpv - (k_val**2) * hv
        zv = mp.mpf("0.41")

        cz, sz = mp.cos(k_val * zv), mp.sin(k_val * zv)
        W0, Wt0, Wtt0 = hv * cz, hpv * cz, hppv * cz
        Wz0, Wzz0, Wtz0 = -k_val * hv * sz, -(k_val**2) * hv * cz, -k_val * hpv * sz
        DkW = Wt0 + Wz0
        D2kW = Wtt0 + 2 * Wtz0 + Wzz0
        sigma2_exact = mp.mpf("0.5") * (DkW**2)

        for al in [mp.mpf("-0.5"), mp.mpf("0"), mp.mpf("0.5"), mp.mpf("1.25")]:
            # For conformal k = d/d(eta) + d/dz on flat dS_4 (a''/a - 2 H^2 = 0) and flat space (H = 0),
            # R_uv[g] k^u k^v = R_uv[g_tilde] k^u k^v = -(d_eta + d_z)^2 (u1 + u2) - [(d_eta+d_z)u1]^2 - [(d_eta+d_z)u2]^2.
            def Rkk_exact(e):
                vals = []
                for sgn in (+1, -1):
                    w, Dk_w, D2k_w = sgn * e * W0, sgn * e * DkW, sgn * e * D2kW
                    f = 1 + w + al * w * w
                    Dk_f = (1 + 2 * al * w) * Dk_w
                    D2k_f = (1 + 2 * al * w) * D2k_w + 2 * al * (Dk_w**2)
                    Dk_u = mp.mpf("0.5") * Dk_f / f
                    D2k_u = mp.mpf("0.5") * (D2k_f * f - Dk_f**2) / (f * f)
                    vals.append((Dk_u, D2k_u))
                (d1, dd1), (d2, dd2) = vals
                return -(dd1 + dd2) - (d1**2 + d2**2)

            neg_Rkk_o2 = -(Rkk_exact(eps) + Rkk_exact(-eps)) / (2 * eps * eps)
            d_dtheta_o2 = (al - mp.mpf("0.5")) * 2 * (DkW**2 + W0 * D2kW)
            max_sachs_id_err = max(max_sachs_id_err, abs(neg_Rkk_o2 - (sigma2_exact + d_dtheta_o2)))

    # (2) General-angle fixed-endpoint nonlinear 4D Gowdy geodesic integration
    def gowdy_fixed_endpoint_angle(alpha_val, theta, phi, k=1.0, e_step=1e-4, N=320):
        nx = math.sin(theta) * math.cos(phi)
        ny = math.sin(theta) * math.sin(phi)
        nz = math.cos(theta)
        M = 1 if abs(nz) < 1e-12 else (2 if abs(nz - 0.5) < 1e-12 else 4)
        L = 2.0 * math.pi * M / k

        def shoot_one(z0, e):
            cx, cy, pz0 = nx, ny, nz
            w0 = e * math.cos(k * z0)
            f1_0 = 1.0 + w0 + alpha_val * w0 * w0
            f2_0 = 1.0 - w0 + alpha_val * w0 * w0
            E0 = math.sqrt(pz0**2 + cx**2 / f1_0 + cy**2 / f2_0)

            def rhs(eta, y):
                x, yy, z, pz, E = y
                ce, se = math.cos(k * eta), math.sin(k * eta)
                cz, sz = math.cos(k * z), math.sin(k * z)
                w = e * ce * cz
                wt = -e * k * se * cz
                wz = -e * k * ce * sz
                f1 = 1.0 + w + alpha_val * w * w
                f2 = 1.0 - w + alpha_val * w * w
                f1t = (1.0 + 2.0 * alpha_val * w) * wt
                f2t = (-1.0 + 2.0 * alpha_val * w) * wt
                f1z = (1.0 + 2.0 * alpha_val * w) * wz
                f2z = (-1.0 + 2.0 * alpha_val * w) * wz
                return [
                    cx / (f1 * E),
                    cy / (f2 * E),
                    pz / E,
                    0.5 * (cx**2 * f1z / f1**2 + cy**2 * f2z / f2**2) / E,
                    -0.5 * (cx**2 * f1t / f1**2 + cy**2 * f2t / f2**2) / E,
                ]

            dt = L / N
            y = [0.0, 0.0, z0, pz0, E0]
            for step in range(N):
                t = step * dt
                k1 = rhs(t, y)
                k2 = rhs(t + 0.5 * dt, [y[i] + 0.5 * dt * k1[i] for i in range(5)])
                k3 = rhs(t + 0.5 * dt, [y[i] + 0.5 * dt * k2[i] for i in range(5)])
                k4 = rhs(t + dt, [y[i] + dt * k3[i] for i in range(5)])
                y = [y[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(5)]
            # Null constraint check at endpoint
            w_end = e * math.cos(k * L) * math.cos(k * y[2])
            f1_e = 1.0 + w_end + alpha_val * w_end**2
            f2_e = 1.0 - w_end + alpha_val * w_end**2
            nres = abs(y[4]**2 - y[3]**2 - cx**2 / f1_e - cy**2 / f2_e)
            dx, dy, dz = y[0] - L * nx, y[1] - L * ny, (y[2] - z0) - L * nz
            d_par = dx * nx + dy * ny + dz * nz
            d_perp2 = dx * dx + dy * dy + dz * dz - d_par * d_par
            return d_par - 0.5 * d_perp2 / L, nres

        nz0 = 8
        acc = 0.0
        max_nres = 0.0
        for j in range(nz0):
            z0 = (2.0 * math.pi / k) * j / nz0
            vp, rp = shoot_one(z0, e_step)
            vm, rm = shoot_one(z0, -e_step)
            acc += 0.5 * (vp + vm) / (e_step * e_step * L)
            max_nres = max(max_nres, rp, rm)
        return 2.0 * acc / nz0, max_nres  # normalized by <h^2> = 2<W^2> = 1/2

    gowdy_geo_checks = {}
    max_gowdy_s2_err = 0.0
    max_gowdy_null_res = 0.0
    for al_name, al_v in [("lin_0.0", 0.0), ("exp_0.5", 0.5), ("al_1.0", 1.0)]:
        f_nz = []
        phi_pi4_nz0 = None
        for nz_v in [0.0, 0.25, 0.5, 0.75]:
            th = math.acos(nz_v)
            v0, r0 = gowdy_fixed_endpoint_angle(al_v, th, 0.0)
            v4, r4 = gowdy_fixed_endpoint_angle(al_v, th, math.pi / 4.0)
            max_gowdy_null_res = max(max_gowdy_null_res, r0, r4)
            if abs(nz_v) < 1e-12:
                phi_pi4_nz0 = v4
            f_nz.append(0.5 * (v0 + v4))
        # Exact Simpson quadrature over cos(theta) in [0, 1] (polynomial of degree 4 -> Simpson is exact!)
        s2_avg = (f_nz[0] + 4.0 * f_nz[1] + 2.0 * f_nz[2] + 4.0 * f_nz[3] + 0.0) / 12.0
        pred = (1.0 - al_v) / 6.0
        err = abs(s2_avg - pred)
        max_gowdy_s2_err = max(max_gowdy_s2_err, err)
        gowdy_geo_checks[al_name] = {
            "alpha": al_v,
            "phi_pi4_theta_pi2_schur_check": phi_pi4_nz0,
            "phi_pi4_exact_pred": (1.0 - al_v) / 4.0,
            "s2_avg_fixed_endpoint": s2_avg,
            "exact_pred_1_minus_alpha_over_6": pred,
            "abs_err": err,
        }
        assert err < 1e-6, f"Gowdy S^2 avg mismatch at {al_name}: {s2_avg} vs {pred}"

    # (3) Exact nonlinear curved-S^3 = SU(2) triaxial Bianchi IX (p = 3) geodesic from eta = -pi/2 to +pi/2!
    c_axis_np = np.array([1.0 / math.sqrt(6.0), 1.0 / math.sqrt(6.0), -2.0 / math.sqrt(6.0)])
    cs_gl, ws_gl = np.polynomial.legendre.leggauss(12)
    int_h2_p3 = 5.0 * math.pi  # int_{-pi/2}^{pi/2} (8 cos^2 + 1) d(eta) = 5*pi

    def bianchi9_p3_geodesic_avg(alpha_val, e_step=1e-4, N=512):
        def h_fn(eta, mode):
            z = (3.0 * math.cos(eta) + 1j * math.sin(eta)) * (math.cos(3.0 * eta) - 1j * math.sin(3.0 * eta))
            return z.real if mode == "cos" else z.imag

        def shoot_su2(mode, n_vec, e):
            def rhs(eta, y):
                q0, q1, q2, q3, p1, p2, p3 = y
                hv = e * h_fn(eta, mode)
                f = 1.0 + c_axis_np * hv + alpha_val * (c_axis_np * hv) ** 2
                p = np.array([p1, p2, p3])
                E = math.sqrt(float(np.sum(p * p / f)))
                v = p / (f * E)
                dq0 = -(q1 * v[0] + q2 * v[1] + q3 * v[2])
                dq1 = q0 * v[0] + (q2 * v[2] - q3 * v[1])
                dq2 = q0 * v[1] + (q3 * v[0] - q1 * v[2])
                dq3 = q0 * v[2] + (q1 * v[1] - q2 * v[0])
                dp = 2.0 * np.cross(p, v)
                return np.array([dq0, dq1, dq2, dq3, dp[0], dp[1], dp[2]])

            dt = math.pi / N
            y = np.array([1.0, 0.0, 0.0, 0.0, n_vec[0], n_vec[1], n_vec[2]], dtype=float)
            for step in range(N):
                t = -0.5 * math.pi + step * dt
                k1 = rhs(t, y)
                k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
                k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
                k4 = rhs(t + dt, y + dt * k3)
                y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            q_vec = y[1:4]
            dchi = -float(np.dot(n_vec, q_vec))
            return dchi

        avg_dchi = 0.0
        for c_th, w_th in zip(cs_gl, ws_gl):
            s_th = math.sqrt(1.0 - c_th * c_th)
            nv = np.array([s_th, 0.0, c_th])
            for mode in ("cos", "sin"):
                dp = shoot_su2(mode, nv, +e_step)
                dm = shoot_su2(mode, nv, -e_step)
                avg_dchi += 0.5 * w_th * 0.5 * (dp + dm) / (e_step * e_step)
        return avg_dchi / int_h2_p3

    b9_geo_checks = {}
    for al_name, al_v in [("lin_0.0", 0.0), ("exp_0.5", 0.5), ("al_1.0", 1.0)]:
        coeff = bianchi9_p3_geodesic_avg(al_v)
        b9_geo_checks[al_name] = {
            "alpha": al_v,
            "curved_S3_p3_dchi_over_int_h2": coeff,
            "schur_baseline_minus_al6_plus_3_20": -al_v / 6.0 + 3.0 / 20.0,
            "diff_from_schur": coeff - (-al_v / 6.0 + 3.0 / 20.0),
        }
    # Verify exact -alpha/6 slope on curved S^3 Bianchi IX!
    b9_alpha_invariance_err = max(
        abs((b9_geo_checks["lin_0.0"]["curved_S3_p3_dchi_over_int_h2"] + 0.0 / 6.0) - (b9_geo_checks["exp_0.5"]["curved_S3_p3_dchi_over_int_h2"] + 0.5 / 6.0)),
        abs((b9_geo_checks["al_1.0"]["curved_S3_p3_dchi_over_int_h2"] + 1.0 / 6.0) - (b9_geo_checks["exp_0.5"]["curved_S3_p3_dchi_over_int_h2"] + 0.5 / 6.0)),
    )
    assert b9_alpha_invariance_err < 1e-7

    print(f"  Exact Sachs-Einstein identity (-R_uv^(2) k^u k^v == sigma^2 + d(delta_theta^(2))/dlam)")
    print(f"    max abs err across flat & dS_4, alpha in [-0.5, 1.25]: {float(max_sachs_id_err):.3e}")
    for k, v in gowdy_geo_checks.items():
        print(
            f"  General-angle Gowdy fixed-endpoint [{k}]: S^2 avg = {v['s2_avg_fixed_endpoint']:+.8f} "
            f"vs (1-alpha)/6 = {v['exact_pred_1_minus_alpha_over_6']:+.8f} (err {v['abs_err']:.2e}; phi=pi/4 check {v['phi_pi4_theta_pi2_schur_check']:+.6f})"
        )
    for k, v in b9_geo_checks.items():
        print(
            f"  Curved-S^3 Bianchi IX p=3 geodesic [{k}]: coeff = {v['curved_S3_p3_dchi_over_int_h2']:+.8f} "
            f"vs Schur (-al/6+3/20) = {v['schur_baseline_minus_al6_plus_3_20']:+.8f} (diff {v['diff_from_schur']:+.6f})"
        )
    print(f"  Curved-S^3 Bianchi IX p=3 alpha-slope invariance err: {b9_alpha_invariance_err:.3e}")
    # (D) Direct S^2-averaged 2x2 screen-metric determinant check of an explicit non-diagonal 3D TT tensor h_ij!
    # Verifies BOTH:
    #   <sqrt(det q_AB) - 1>_{S^2} / <h^2>         = alpha/3 - 1/10   (total with bg -1/3*(al-1/2): +1/15)
    #   <ln sqrt(det q_AB)>_{S^2} / <h^2>          = alpha/3 - 7/60   (total with bg -1/3*(al-1/2): +1/20)
    H_raw = np.array([[1.2, 0.7, -0.4], [0.7, -0.5, 0.9], [-0.4, 0.9, -0.7]], dtype=float)
    H_tt = H_raw / math.sqrt(float(np.sum(H_raw * H_raw)))  # Tr(H_tt) == 0, Tr(H_tt^2) == 1
    H_tt2 = H_tt @ H_tt
    mu_gl, w_gl = np.polynomial.legendre.leggauss(16)
    phis_gl = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    eps_q = 1e-4
    screen_det_checks = {}
    max_screen_area_ratio_err = 0.0
    max_screen_log_det_err = 0.0
    for al_v in [-1.0, 0.0, 0.5, 1.0, 1.5]:
        avg_area = 0.0
        avg_log = 0.0
        w_tot = 0.0
        for mui, wi in zip(mu_gl, w_gl):
            sth = math.sqrt(max(0.0, 1.0 - mui * mui))
            for phi in phis_gl:
                e1 = np.array([mui * math.cos(phi), mui * math.sin(phi), -sth])
                e2 = np.array([-math.sin(phi), math.cos(phi), 0.0])
                E = np.column_stack([e1, e2])  # 3x2 orthonormal screen basis perp to n
                Gp = np.eye(3) + eps_q * H_tt + al_v * (eps_q ** 2) * H_tt2
                Gm = np.eye(3) - eps_q * H_tt + al_v * (eps_q ** 2) * H_tt2
                qp = E.T @ Gp @ E
                qm = E.T @ Gm @ E
                sp_det = math.sqrt(qp[0, 0] * qp[1, 1] - qp[0, 1] * qp[1, 0])
                sm_det = math.sqrt(qm[0, 0] * qm[1, 1] - qm[0, 1] * qm[1, 0])
                c2_area = (sp_det + sm_det - 2.0) / (2.0 * eps_q * eps_q)
                c2_log = (math.log(sp_det) + math.log(sm_det)) / (2.0 * eps_q * eps_q)
                avg_area += wi * c2_area
                avg_log += wi * c2_log
                w_tot += wi
        avg_area /= w_tot
        avg_log /= w_tot
        pred_area = al_v / 3.0 - 0.1
        pred_log = al_v / 3.0 - 7.0 / 60.0
        ea = abs(avg_area - pred_area)
        el = abs(avg_log - pred_log)
        max_screen_area_ratio_err = max(max_screen_area_ratio_err, ea)
        max_screen_log_det_err = max(max_screen_log_det_err, el)
        screen_det_checks[str(al_v)] = {
            "s2_avg_area_ratio_coeff": avg_area,
            "pred_al_over_3_minus_1_over_10": pred_area,
            "total_area_ratio_with_bg": avg_area - (al_v - 0.5) / 3.0,
            "s2_avg_log_det_coeff": avg_log,
            "pred_al_over_3_minus_7_over_60": pred_log,
            "total_log_det_with_bg": avg_log - (al_v - 0.5) / 3.0,
        }
    print(
        f"  Direct 3D TT S^2-averaged 2x2 screen metric check: "
        f"area-ratio (+1/15) max err = {max_screen_area_ratio_err:.3e} | "
        f"log-det (+1/20) max err = {max_screen_log_det_err:.3e}"
    )
    assert max_screen_area_ratio_err < 1e-8 and max_screen_log_det_err < 1e-8
    assert max_sachs_id_err < mp.mpf("1e-21")
    return {
        "sachs_einstein_identity_max_err": float(max_sachs_id_err),
        "gowdy_general_angle_fixed_endpoint": gowdy_geo_checks,
        "gowdy_s2_max_err": max_gowdy_s2_err,
        "gowdy_null_constraint_max_res": max_gowdy_null_res,
        "curved_S3_bianchi9_p3_geodesic": b9_geo_checks,
        "curved_S3_bianchi9_alpha_invariance_err": b9_alpha_invariance_err,
        "direct_2x2_screen_determinant_S2_check": {
            "max_area_ratio_err": max_screen_area_ratio_err,
            "max_log_det_err": max_screen_log_det_err,
            "per_alpha": screen_det_checks,
        },
        "exact_decomposition_of_1_minus_alpha_over_6": {
            "straight_line_g_nn_inv_sqrt": "-alpha/6 + 1/20",
            "schur_complement_metric_shear": "+1/10",
            "transverse_fermat_deflection": "+1/40",
            "longitudinal_redshift_cross_term": "-1/120",
            "total_geodesic_LC": "-alpha/6 + 1/6",
            "total_with_BR_FP": "Delta_eta_BR^FP + (1/12) * int <h^2> d(eta)",
            "schur_baseline_with_BR_FP": "Delta_eta_BR^FP + (1/15) * int <h^2> d(eta)",
        },
        "local_2d_screen_area_ratio_alpha_invariant_coeff": 1.0 / 15.0,
        "local_2d_screen_log_det_alpha_invariant_coeff": 1.0 / 20.0,
    }


kt1b_sachs = verify_sachs_and_geodesics()


# =============================================================================
# PART 1C: 50-DIGIT GLOBAL dS_4/Gamma EVALUATION ACROSS ALL 5 ADE SPACE FORMS
# =============================================================================
print("\n" + "=" * 78)
print("PART 1C: 50-DIGIT GLOBAL dS_4/Gamma PARAMETERIZATION-INVARIANT OBSERVABLES (ALL 5 ADE)")
print("=" * 78)


def su2_ade_elements(name: str):
    if name == "RP3":
        return [(mp.mpf(1), 1), (mp.mpf(-1), 1)]
    if name == "Q8":
        return [(mp.mpf(1), 1), (mp.mpf(-1), 1), (mp.mpf(0), 6)]
    if name == "T*":
        return [(mp.mpf(1), 1), (mp.mpf(-1), 1), (mp.mpf(0), 6), (mp.mpf("0.5"), 8), (mp.mpf("-0.5"), 8)]
    if name == "O*":
        s2 = 1 / mp.sqrt(2)
        return [
            (mp.mpf(1), 1), (mp.mpf(-1), 1), (mp.mpf(0), 18),
            (mp.mpf("0.5"), 8), (mp.mpf("-0.5"), 8), (s2, 6), (-s2, 6),
        ]
    if name == "I*":
        phi = (1 + mp.sqrt(5)) / 2
        return [
            (mp.mpf(1), 1), (mp.mpf(-1), 1), (mp.mpf(0), 30),
            (mp.mpf("0.5"), 20), (mp.mpf("-0.5"), 20),
            (phi / 2, 12), (-phi / 2, 12), (1 / (2 * phi), 12), (-1 / (2 * phi), 12),
        ]
    raise ValueError(name)


def chi_j(j_twice: int, c_alpha):
    n = j_twice + 1
    if abs(c_alpha - 1) < mp.mpf("1e-30"):
        return mp.mpf(n)
    if abs(c_alpha + 1) < mp.mpf("1e-30"):
        return mp.mpf(n if j_twice % 2 == 0 else -n)
    alpha = mp.acos(c_alpha)
    return mp.sin(n * alpha) / mp.sin(alpha)


def ade_tt_multiplicity(name: str, p: int) -> int:
    elems = su2_ade_elements(name)
    order = sum(cnt for _, cnt in elems)
    inv_minus = sum(cnt * chi_j(p - 3, ca) for ca, cnt in elems) / order
    inv_plus = sum(cnt * chi_j(p + 1, ca) for ca, cnt in elems) / order
    val = (p + 2) * inv_minus + (p - 2) * inv_plus
    iv = int(mp.nint(val))
    assert abs(val - iv) < mp.mpf("1e-30")
    return iv


ADE_PERIODS = {"RP3": 2, "Q8": 4, "T*": 12, "O*": 24, "I*": 60}
ADE_ORDERS = {"RP3": 2, "Q8": 8, "T*": 24, "O*": 48, "I*": 120}


def compute_topological_invariants_ade(name: str):
    L = ADE_PERIODS[name]
    G = ADE_ORDERS[name]
    c_r, a_r, b_r = [], [], []
    for j in range(L):
        p0 = 3 + j
        ds = [G * ade_tt_multiplicity(name, p0 + n * L) - 2 * ((p0 + n * L) ** 2 - 4) for n in range(4)]
        assert ds[3] - 3 * ds[2] + 3 * ds[1] - ds[0] == 0
        C = mp.mpf(ds[2] - 2 * ds[1] + ds[0]) / (2 * L * L)
        A = (mp.mpf(ds[1] - ds[0]) - C * (2 * p0 * L + L * L)) / L
        B = mp.mpf(ds[0]) - C * p0 * p0 - A * p0
        c_r.append(C)
        a_r.append(A)
        b_r.append(B)
    assert abs(sum(c_r)) < mp.mpf("1e-45") and abs(sum(a_r)) < mp.mpf("1e-45") and abs(sum(b_r)) < mp.mpf("1e-45")

    S_p1, S_inv, S_sup = mp.mpf(0), mp.mpf(0), mp.mpf(0)
    for j in range(L):
        p0 = mp.mpf(3 + j)
        q, qm, qp = p0 / L, (p0 - 1) / L, (p0 + 1) / L
        C, A, B = c_r[j], a_r[j], b_r[j]
        S_p1 += C * (L**3) * mp.zeta(-3, q) + A * (L**2) * mp.zeta(-2, q) + B * L * mp.zeta(-1, q)
        S_inv += C * L * mp.zeta(-1, q) + A * mp.zeta(0, q) - (B / L) * mp.psi(0, q)
        S_sup += (
            C * (-mp.mpf("0.5") / L * (mp.psi(0, qm) + mp.psi(0, qp)))
            + A * (-mp.mpf("0.5") / L * (mp.psi(0, qm) - mp.psi(0, qp)))
            + B * (-mp.mpf("0.5") / L * (mp.psi(0, qm) + mp.psi(0, qp)) + (1 / L) * mp.psi(0, q))
        )
    return mp.mpf("0.5") * S_p1, mp.mpf("0.5") * S_inv, mp.mpf("0.5") * S_sup


def verify_ade_50digit_invariance():
    pi2_C_EM_Istar = -mp.mpf(22369) / 720
    alphas_test = [mp.mpf("-1"), mp.mpf("0"), mp.mpf("0.25"), mp.mpf("0.5"), mp.mpf("1"), mp.mpf("1.5")]
    chi_inj_map = {"RP3": mp.pi / 2, "Q8": mp.pi / 4, "T*": mp.pi / 6, "O*": mp.pi / 8, "I*": mp.pi / 10}
    ade_results = {}
    max_invariance_err_schur = mp.mpf(0)
    max_invariance_err_wkb = mp.mpf(0)
    max_screen_area_invariance_err = mp.mpf(0)

    for name in ["RP3", "Q8", "T*", "O*", "I*"]:
        S_p1, S_inv, S_sup = compute_topological_invariants_ade(name)
        pi2_C_rad = mp.mpf("0.5") * (S_p1 - S_inv)
        pi2_C_curv = mp.mpf("0.25") * S_inv
        pi2_int_h2 = mp.pi * (S_inv + 2 * S_sup)
        pi2_deta_BR_FP = (mp.pi / 4) * pi2_C_rad + (mp.pi / 6) * pi2_C_curv

        # Three alpha-invariant totals:
        #   1. Straight coordinate line (omitting Schur complement): BR_FP - (1/30) int <h^2> (diagnostic only)
        #   2. Exact Schur-complement geodesic baseline (all p>=3):  BR_FP + (1/15) int <h^2>
        #   3. Full 2nd-order WKB fixed-endpoint geodesic total:     BR_FP + (1/12) int <h^2>
        pi2_tot_straight = pi2_deta_BR_FP - (mp.mpf(1) / 30) * pi2_int_h2
        pi2_tot_schur = pi2_deta_BR_FP + (mp.mpf(1) / 15) * pi2_int_h2
        pi2_tot_wkb = pi2_deta_BR_FP + (mp.mpf(1) / 12) * pi2_int_h2

        per_alpha = {}
        for al in alphas_test:
            da = al - mp.mpf("0.5")
            I_t1 = (mp.mpf(8) / 3) * (3 * mp.pi / 8) * mp.mpf("0.5") * (S_p1 - S_inv)
            I_t2 = (-8) * (-mp.pi / 8) * mp.mpf("0.5") * S_inv
            I_t3a = -(mp.mpf(8) / 3) * ((3 * mp.pi / 8) * mp.mpf("0.5") * (S_p1 - S_inv) + (mp.pi / 2) * mp.mpf("0.5") * S_inv)
            I_t3b = (mp.mpf(8) / 3) * ((3 * mp.pi / 8) * mp.mpf("0.5") * S_inv + (mp.pi / 2) * mp.mpf("0.5") * S_sup)
            br_al = pi2_deta_BR_FP + mp.mpf("0.5") * da * (I_t1 + I_t2 + I_t3a + I_t3b)
            lc_schur_al = (-al / 6 + mp.mpf(3) / 20) * pi2_int_h2
            lc_wkb_al = (-al / 6 + mp.mpf(1) / 6) * pi2_int_h2
            tot_schur_al = br_al + lc_schur_al
            tot_wkb_al = br_al + lc_wkb_al
            max_invariance_err_schur = max(max_invariance_err_schur, abs(tot_schur_al - pi2_tot_schur))
            max_invariance_err_wkb = max(max_invariance_err_wkb, abs(tot_wkb_al - pi2_tot_wkb))
            per_alpha[str(float(al))] = {
                "pi2_deta_BR": float(br_al),
                "pi2_deta_LC_schur": float(lc_schur_al),
                "pi2_deta_LC_wkb": float(lc_wkb_al),
                "pi2_deta_tot_schur": float(tot_schur_al),
                "pi2_deta_tot_wkb": float(tot_wkb_al),
            }

        # Local fractional 2D screen-area shift <delta sqrt(det g_perp) / sqrt(det g_bar_perp)>_tot = +1/15 <h^2>
        # (and log-determinant shift <delta ln sqrt(det g_perp)>_tot = +1/20 <h^2>, verified directly in kt1b):
        chi = chi_inj_map[name]
        pi2_h2_waist = 2 * (mp.cos(chi) ** 2) * S_inv + 2 * S_sup
        ref_screen_sum = mp.mpf(1) / 15 * pi2_h2_waist
        for al in alphas_test:
            bg_a2_shift = -mp.mpf(1) / 3 * (al - mp.mpf("0.5")) * pi2_h2_waist
            screen_q_shift = (al / 3 - mp.mpf(1) / 10) * pi2_h2_waist
            max_screen_area_invariance_err = max(max_screen_area_invariance_err, abs((bg_a2_shift + screen_q_shift) - ref_screen_sum))

        ade_results[name] = {
            "order": ADE_ORDERS[name],
            "period": ADE_PERIODS[name],
            "S_p1": float(S_p1),
            "S_inv": float(S_inv),
            "S_sup": float(S_sup),
            "pi2_C_rad_FP": float(pi2_C_rad),
            "pi2_C_curv_FP": float(pi2_C_curv),
            "pi2_int_h2": float(pi2_int_h2),
            "pi2_deta_BR_FP": float(pi2_deta_BR_FP),
            "pi2_deta_BR_LIN": per_alpha["0.0"]["pi2_deta_BR"],
            "pi2_deta_tot_straight_line_only": float(pi2_tot_straight),
            "pi2_deta_tot_schur_baseline": float(pi2_tot_schur),
            "pi2_deta_tot_wkb_geodesic": float(pi2_tot_wkb),
            "per_alpha": per_alpha,
        }
        print(
            f"  {name:4s}: BR_FP={float(pi2_deta_BR_FP):+10.6f} | BR_LIN={per_alpha['0.0']['pi2_deta_BR']:+10.6f} | "
            f"TOT_SCHUR={float(pi2_tot_schur):+10.6f} | TOT_WKB={float(pi2_tot_wkb):+10.6f}"
        )

    pi2_deta_EM_Istar = (mp.pi / 4) * pi2_C_EM_Istar
    Istar = ade_results["I*"]
    x_FP_BR = -Istar["pi2_deta_BR_FP"] / float(pi2_deta_EM_Istar)
    x_LIN_BR = -Istar["pi2_deta_BR_LIN"] / float(pi2_deta_EM_Istar)
    x_STRAIGHT = -Istar["pi2_deta_tot_straight_line_only"] / float(pi2_deta_EM_Istar)
    x_SCHUR = -Istar["pi2_deta_tot_schur_baseline"] / float(pi2_deta_EM_Istar)
    x_WKB = -Istar["pi2_deta_tot_wkb_geodesic"] / float(pi2_deta_EM_Istar)

    print(f"\n  Max |TOT_WKB(alpha) - TOT_WKB(FP)| across 5 ADE & 6 alphas:   {float(max_invariance_err_wkb):.3e}")
    print(f"  Max |TOT_SCHUR(alpha) - TOT_SCHUR(FP)| across 5 ADE & 6 alphas: {float(max_invariance_err_schur):.3e}")
    print(f"  Max local 2D screen-area determinant identity err:             {float(max_screen_area_invariance_err):.3e}")
    print(f"  S^3/I* dark-photon crossover (pi^2 Delta_eta_EM = {float(pi2_deta_EM_Istar):+.6f}):")
    print(f"    FP backreaction alone (BR only):             x = {x_FP_BR:.6f} -> N_V^crit = {math.ceil(x_FP_BR)}")
    print(f"    Naive linear-metric alone (alpha-dep):       x = {x_LIN_BR:.6f} -> N_V^crit = {math.ceil(x_LIN_BR)}")
    print(f"    Straight-line only (-1/30, omits Schur):     x = {x_STRAIGHT:.6f} -> N_V^crit = {math.ceil(x_STRAIGHT)}  [INCOMPLETE]")
    print(f"    Schur-complement geodesic baseline (+1/15):  x = {x_SCHUR:.6f} -> N_V^crit = {math.ceil(x_SCHUR)}")
    print(f"    Full 2nd-order WKB geodesic total (+1/12):   x = {x_WKB:.6f} -> N_V^crit = {math.ceil(x_WKB)}")

    assert max_invariance_err_wkb < mp.mpf("1e-47")
    assert max_invariance_err_schur < mp.mpf("1e-47")
    assert max_screen_area_invariance_err < mp.mpf("1e-47")
    assert math.ceil(x_SCHUR) == 5 and math.ceil(x_WKB) == 6

    return {
        "ade_space_forms": ade_results,
        "max_global_deta_wkb_invariance_err_50dps": float(max_invariance_err_wkb),
        "max_global_deta_schur_invariance_err_50dps": float(max_invariance_err_schur),
        "max_local_2d_screen_area_identity_err_50dps": float(max_screen_area_invariance_err),
        "Istar_dark_photon_crossover": {
            "pi2_C_EM_exact": float(pi2_C_EM_Istar),
            "pi2_deta_EM": float(pi2_deta_EM_Istar),
            "x_FP_BR": x_FP_BR,
            "x_LIN_BR": x_LIN_BR,
            "x_straight_line_incomplete": x_STRAIGHT,
            "x_schur_geodesic_baseline": x_SCHUR,
            "x_wkb_geodesic_total": x_WKB,
            "N_V_crit_FP_BR": math.ceil(x_FP_BR),
            "N_V_crit_LIN_BR": math.ceil(x_LIN_BR),
            "N_V_crit_schur_baseline": math.ceil(x_SCHUR),
            "N_V_crit_wkb_total": math.ceil(x_WKB),
        },
    }


kt1c_ade = verify_ade_50digit_invariance()


# =============================================================================
# PART 2: EXACT SYMBOLIC (SYMPY) + MATRIX PROOF OF BERGER-SPHERE STABILITY (KT2)
# =============================================================================
print("\n" + "=" * 78)
print("PART 2 (KT2): EXACT SYMBOLIC & MATRIX PROOF OF TOPOLOGICAL PROLATE STABILIZATION")
print("=" * 78)


def symbolic_proof_berger_prolate_theorem():
    lam = sp.Symbol("lam", positive=True)
    j = sp.Symbol("j", positive=True)
    I = sp.I
    s2 = sp.sqrt(2)
    s6 = sp.sqrt(6)

    T_sym = [
        sp.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]]) / s2,
        sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / s6,
        sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]) / s2,
        sp.Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]) / s2,
        sp.Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]) / s2,
    ]
    L3 = sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    S3 = sp.Matrix(5, 5, lambda a, b: sum((T_sym[a][p, q] * (I * (L3 * T_sym[b] - T_sym[b] * L3))[p, q] for p in range(3) for q in range(3))))
    U_S3 = sp.Matrix([
        [1 / s2, 0, 0, 0, 1 / s2],
        [0, 0, 1, 0, 0],
        [-I / s2, 0, 0, 0, I / s2],
        [0, 1 / s2, 0, 1 / s2, 0],
        [0, -I / s2, 0, I / s2, 0],
    ])
    assert sp.simplify(U_S3.H * S3 * U_S3) == sp.diag(-2, -1, 0, 1, 2)
    U_V3 = sp.Matrix([[1 / s2, 0, 1 / s2], [-I / s2, 0, I / s2], [0, 1, 0]])
    assert sp.simplify(U_V3.H * (I * L3) * U_V3) == sp.diag(-1, 0, 1)

    c = [[[sp.Integer(0) for _ in range(3)] for _ in range(3)] for _ in range(3)]
    c[2][0][1], c[2][1][0] = 2 * lam, -2 * lam
    c[0][1][2], c[0][2][1] = 2 / lam, -2 / lam
    c[1][2][0], c[1][0][2] = 2 / lam, -2 / lam

    Gamma = [[[sp.simplify(sp.Rational(1, 2) * (c[a][k][b] - c[k][b][a] + c[b][a][k])) for k in range(3)] for b in range(3)] for a in range(3)]
    Riem = [[[[
        sp.simplify(sum(Gamma[a][e][k] * Gamma[e][b][l] - Gamma[a][e][l] * Gamma[e][b][k] - c[e][k][l] * Gamma[a][b][e] for e in range(3)))
        for l in range(3)] for k in range(3)] for b in range(3)] for a in range(3)]

    M_raw = []
    for k in range(3):
        Gk = sp.Matrix(3, 3, lambda a, b: Gamma[a][b][k])
        M_raw.append(sp.Matrix(5, 5, lambda a, b: sp.simplify(sum(T_sym[a][p, q] * (Gk * T_sym[b] - T_sym[b] * Gk)[p, q] for p in range(3) for q in range(3)))))

    K_riem_raw = sp.zeros(5, 5)
    for b in range(5):
        Mr = sp.Matrix(3, 3, lambda p, q: sp.simplify(-2 * sum(Riem[p][k][q][l] * T_sym[b][k, l] for k in range(3) for l in range(3))))
        for a in range(5):
            K_riem_raw[a, b] = sp.simplify(sum(T_sym[a][p, q] * Mr[p, q] for p in range(3) for q in range(3)))

    M_sph = [sp.simplify(U_S3.H * Mk * U_S3) for Mk in M_raw]
    Op_const = sp.simplify(-(M_sph[0] ** 2 + M_sph[1] ** 2 + M_sph[2] ** 2) + U_S3.H * K_riem_raw * U_S3)
    Op_Jp = sp.simplify(2 * I * (M_sph[0] - I * M_sph[1]))
    Op_Jm = sp.simplify(2 * I * (M_sph[0] + I * M_sph[1]))
    Op_J3 = sp.simplify((4 * I / lam) * M_sph[2])

    C_raw = [sp.Matrix(3, 5, lambda a, al: T_sym[al][k, a]) for k in range(3)]
    C_sph = [sp.simplify(U_V3.H * Ck * U_S3) for Ck in C_raw]
    Div_const = sp.simplify(C_sph[0] * M_sph[0] + C_sph[1] * M_sph[1] + C_sph[2] * M_sph[2])
    Div_Jp = sp.simplify(-I * (C_sph[0] - I * C_sph[1]))
    Div_Jm = sp.simplify(-I * (C_sph[0] + I * C_sph[1]))
    Div_J3 = sp.simplify((-2 * I / lam) * C_sph[2])

    c_j2 = 4 * j * (j + 1)
    c_lam = 4 * (1 / lam**2 - 1)
    omega2_stretched = sp.simplify(Op_const[4, 4] + j * Op_J3[4, 4] + c_j2 + c_lam * j**2)
    assert sp.simplify(omega2_stretched - (4 * (j + 2) ** 2 / lam**2 - 4 * (j + 2))) == 0
    print(f"  [PROVED 1] Stretched |M_L|=j+2 exact: omega^2 = {sp.factor(omega2_stretched)}")

    omega2_j0_M0 = sp.simplify(Op_const[2, 2])
    assert sp.simplify(omega2_j0_M0 - (sp.Rational(32, 3) * lam**2 - sp.Rational(8, 3))) == 0
    print(f"  [PROVED 2] Oblate j=0, M_L=0 exact:   omega^2 = {omega2_j0_M0}")

    jp_jm1 = sp.sqrt(j * (j + 1) - (j - 1) * j)
    Op_sub = sp.Matrix(2, 2, [
        Op_const[3, 3] + j * Op_J3[3, 3] + c_j2 + c_lam * j**2,
        Op_const[3, 4] + jp_jm1 * Op_Jp[3, 4],
        Op_const[4, 3] + jp_jm1 * Op_Jm[4, 3],
        Op_const[4, 4] + (j - 1) * Op_J3[4, 4] + c_j2 + c_lam * (j - 1) ** 2,
    ])
    Div_sub = sp.Matrix(1, 2, [Div_const[2, 3] + j * Div_J3[2, 3], Div_const[2, 4] + jp_jm1 * Div_Jp[2, 4]])
    v_sub = sp.Matrix([-Div_sub[0, 1], Div_sub[0, 0]])
    omega2_sub = sp.factor(sp.simplify((v_sub.H * Op_sub * v_sub)[0, 0] / (v_sub.H * v_sub)[0, 0]))
    lim_sub_inf = sp.limit(omega2_sub, lam, sp.oo)
    print(f"  [PROVED 3] Sub-stretched |M_L|=j+1 exact: lim_(lam->oo) omega^2 = {lim_sub_inf}")

    def build_M0_blocks(j_sym, include_pm2=True):
        s_list = [0, 1, 2, 3, 4] if include_pm2 else [1, 2, 3]
        v_list = [0, 1, 2]
        ns, nv = len(s_list), len(v_list)
        Op_m, Div_m = sp.zeros(ns, ns), sp.zeros(nv, ns)
        cj = 4 * j_sym * (j_sym + 1)

        def ladd(m_from, step):
            return sp.sqrt(j_sym * (j_sym + 1) - m_from * (m_from + step))

        for r, ir in enumerate(s_list):
            mr = -(ir - 2)
            for col, ic in enumerate(s_list):
                mc = -(ic - 2)
                val = sp.Integer(0)
                if mr == mc:
                    val += Op_const[ir, ic] + mc * Op_J3[ir, ic] + (cj + c_lam * mc**2 if ir == ic else 0)
                elif mr == mc + 1:
                    val += ladd(mc, +1) * Op_Jp[ir, ic]
                elif mr == mc - 1:
                    val += ladd(mc, -1) * Op_Jm[ir, ic]
                Op_m[r, col] = sp.simplify(val)
        for r, vr in enumerate(v_list):
            mr = -(vr - 1)
            for col, ic in enumerate(s_list):
                mc = -(ic - 2)
                val = sp.Integer(0)
                if mr == mc:
                    val += Div_const[vr, ic] + mc * Div_J3[vr, ic]
                elif mr == mc + 1:
                    val += ladd(mc, +1) * Div_Jp[vr, ic]
                elif mr == mc - 1:
                    val += ladd(mc, -1) * Div_Jm[vr, ic]
                Div_m[r, col] = sp.simplify(val)
        return Op_m, Div_m

    Op_j1, Div_j1 = build_M0_blocks(sp.Integer(1), include_pm2=False)
    ns_j1 = Div_j1.nullspace()
    assert len(ns_j1) == 1
    v_j1 = ns_j1[0]
    omega2_j1_M0 = sp.factor(sp.simplify((v_j1.H * Op_j1 * v_j1)[0, 0] / (v_j1.H * v_j1)[0, 0]))
    print(f"  [PROVED 4a] j=1, M_L=0 exact (1 TT mode): omega^2 = {omega2_j1_M0}")

    Op_M0, Div_M0 = build_M0_blocks(j, include_pm2=True)
    P_T2 = sp.Matrix([
        [1 / s2, 0, 0, 1 / s2, 0],
        [0, -1 / s2, 0, 0, 1 / s2],
        [0, 0, 1, 0, 0],
        [0, 1 / s2, 0, 0, 1 / s2],
        [1 / s2, 0, 0, -1 / s2, 0],
    ])
    P_V2 = sp.Matrix([[1 / s2, -1 / s2, 0], [0, 0, 1], [1 / s2, 1 / s2, 0]])
    Op_p2 = sp.simplify(P_T2.H * Op_M0 * P_T2)
    Div_p2 = sp.simplify(P_V2.H * Div_M0 * P_T2)
    assert sp.simplify(Op_p2[:3, 3:]) == sp.zeros(3, 2)
    assert sp.simplify(Div_p2[0:1, 3:5]) == sp.zeros(1, 2)
    assert sp.simplify(Div_p2[1:3, 0:3]) == sp.zeros(2, 3)

    det_Div_odd = sp.factor(sp.simplify(Div_p2[1:3, 3:5].det()))
    print(f"  [PROVED 4b] M_L=0 ODD-parity 2x2 Div det: {det_Div_odd} (invertible => 0 ODD TT modes)")

    Div_even, Op_even = Div_p2[0:1, 0:3], Op_p2[0:3, 0:3]
    d0, d1, d2 = Div_even[0, 0], Div_even[0, 1], Div_even[0, 2]
    B_tt = sp.Matrix([[-d1, 0], [d0, -d2], [0, d1]])
    assert sp.simplify(Div_even * B_tt) == sp.zeros(1, 2)

    u_pos = sp.Symbol("u_pos", positive=True)
    B_u, Op_u = B_tt.subs(j, 2 + u_pos), Op_even.subs(j, 2 + u_pos)
    simp_u = lambda e: sp.cancel(sp.expand(sp.powsimp(e, force=True)))
    H_2u = (B_u.H * Op_u * B_u).applyfunc(simp_u)
    G_2u = (B_u.H * B_u).applyfunc(simp_u)
    h00_M0 = sp.factor(simp_u(H_2u[0, 0]).subs(u_pos, j - 2))
    detH_M0 = sp.factor(simp_u(H_2u[0, 0] * H_2u[1, 1] - H_2u[0, 1] * H_2u[1, 0]).subs(u_pos, j - 2))
    detG_M0 = sp.factor(simp_u(G_2u[0, 0] * G_2u[1, 1] - G_2u[0, 1] * G_2u[1, 0]).subs(u_pos, j - 2))
    tr_num_u = simp_u(G_2u[1, 1] * H_2u[0, 0] + G_2u[0, 0] * H_2u[1, 1] - G_2u[0, 1] * H_2u[1, 0] - G_2u[1, 0] * H_2u[0, 1])
    tr_M0 = sp.factor(sp.cancel(tr_num_u.subs(u_pos, j - 2) / detG_M0))
    det_M0 = sp.factor(sp.cancel(detH_M0 / detG_M0))

    u_sym = sp.Symbol("u", nonnegative=True)
    x_sym = sp.Symbol("x", nonnegative=True)
    y_sym = sp.Symbol("y", nonnegative=True)
    pos_proofs = {}
    for label, expr in [("Sylvester_H00", h00_M0), ("Sylvester_detH", detH_M0), ("Det_omega2", det_M0), ("Trace_omega2", tr_M0)]:
        num, den = sp.fraction(sp.together(expr))
        poly_num = sp.Poly(sp.expand(num.subs({lam: sp.sqrt(1 + x_sym), j: 2 + u_sym})), u_sym, x_sym)
        poly_den = sp.Poly(sp.expand(den.subs({lam: sp.sqrt(1 + x_sym), j: 2 + u_sym})), u_sym, x_sym)
        if poly_den.coeffs()[0] < 0:
            poly_num, poly_den = -poly_num, -poly_den
        poly_num_y = sp.Poly(sp.expand(num.subs({lam: sp.sqrt(y_sym), j: 2 + u_sym})), u_sym, y_sym)
        poly_den_y = sp.Poly(sp.expand(den.subs({lam: sp.sqrt(y_sym), j: 2 + u_sym})), u_sym, y_sym)
        if poly_den_y.coeffs()[0] < 0:
            poly_num_y, poly_den_y = -poly_num_y, -poly_den_y
        mn_p, md_p = min(float(c) for c in poly_num.coeffs()), min(float(c) for c in poly_den.coeffs())
        mn_a, md_a = min(float(c) for c in poly_num_y.coeffs()), min(float(c) for c in poly_den_y.coeffs())
        assert mn_p >= 0 and md_p >= 0 and mn_a >= 0 and md_a >= 0
        pos_proofs[label] = {
            "exact_formula": str(expr),
            "prolate_min_num_coeff": mn_p,
            "prolate_min_den_coeff": md_p,
            "all_lam_min_num_coeff": mn_a,
            "all_lam_min_den_coeff": md_a,
        }
    print(f"  [PROVED 4c] M_L=0 EVEN 2x2 TT block: strictly positive for ALL j>=2 & ALL lam_B > 0!")

    num_s, den_s = sp.fraction(sp.together(omega2_sub))
    num_sy_half = sp.expand(num_s.subs({lam: sp.sqrt(y_sym), j: sp.Rational(1, 2)}))
    lam_jhalf = [math.sqrt(float(r)) for r in sp.nroots(num_sy_half) if abs(sp.im(r)) < 1e-12 and sp.re(r) > 0]
    u1 = sp.Symbol("u1", nonnegative=True)
    p_ns = sp.Poly(sp.expand(num_s.subs({lam: sp.sqrt(1 + x_sym), j: 1 + u1})), u1, x_sym)
    p_ds = sp.Poly(sp.expand(den_s.subs({lam: sp.sqrt(1 + x_sym), j: 1 + u1})), u1, x_sym)
    if p_ds.coeffs()[0] < 0:
        p_ns, p_ds = -p_ns, -p_ds
    min_sub_num, min_sub_den = min(float(c) for c in p_ns.coeffs()), min(float(c) for c in p_ds.coeffs())
    assert min_sub_num >= 0 and min_sub_den >= 0
    print(f"  [PROVED 3b] Sub-stretched: j=1/2 tachyon at lam={lam_jhalf[0]:.6f}; j>=1, lam>=1 proved positive (min_num={min_sub_num})")

    return {
        "stretched_formula": str(sp.factor(omega2_stretched)),
        "oblate_j0_M0_formula": str(omega2_j0_M0),
        "sub_stretched_formula": str(omega2_sub),
        "sub_stretched_j_half_tachyon_lam": lam_jhalf,
        "sub_stretched_j_ge_1_prolate_min_num_coeff": min_sub_num,
        "sub_stretched_j_ge_1_prolate_min_den_coeff": min_sub_den,
        "sub_stretched_large_lam_limit": str(lim_sub_inf),
        "M0_j1_dim_tt": 1,
        "M0_j1_exact_formula": str(omega2_j1_M0),
        "M0_odd_div_det": str(det_Div_odd),
        "M0_positivity_certificates": pos_proofs,
    }


kt2_sym = symbolic_proof_berger_prolate_theorem()

# =============================================================================
# PART 2B: NUMERICAL BLOCK SOLVER, OBLATE + PROLATE NON-STRETCHED SCANS & TABLE
# =============================================================================
print("\n" + "=" * 78)
print("PART 2B (KT2): OBLATE & PROLATE NON-STRETCHED SCANS (2j<=40) & FULL-MATRIX CHECK")
print("=" * 78)

T_basis_np = [
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=float) / math.sqrt(2.0),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=float) / math.sqrt(6.0),
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float) / math.sqrt(2.0),
    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=float) / math.sqrt(2.0),
    np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float) / math.sqrt(2.0),
]
L_gens_np = [
    np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float),
    np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float),
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float),
]
S_gens_np = []
for L in L_gens_np:
    Sm = np.zeros((5, 5), dtype=complex)
    for b in range(5):
        comm = 1j * (L @ T_basis_np[b] - T_basis_np[b] @ L)
        for a in range(5):
            Sm[a, b] = np.sum(T_basis_np[a] * comm)
    S_gens_np.append(Sm)

ev_S3, U_S3_np = np.linalg.eigh(S_gens_np[2])
U_S3_np = U_S3_np[:, np.argsort(ev_S3)]
ev_V3, U_V3_np = np.linalg.eigh(1j * L_gens_np[2])
U_V3_np = U_V3_np[:, np.argsort(ev_V3)]
C_raw_np = [np.array([[T_basis_np[al][k, a] for al in range(5)] for a in range(3)], dtype=complex) for k in range(3)]
C_sph_np = [U_V3_np.conj().T @ C_raw_np[k] @ U_S3_np for k in range(3)]
C_p_np, C_m_np, C_0_np = C_sph_np[0] - 1j * C_sph_np[1], C_sph_np[0] + 1j * C_sph_np[1], C_sph_np[2]


def berger_precompute_np(lam: float):
    c = np.zeros((3, 3, 3))
    c[2, 0, 1], c[2, 1, 0] = 2.0 * lam, -2.0 * lam
    c[0, 1, 2], c[0, 2, 1] = 2.0 / lam, -2.0 / lam
    c[1, 2, 0], c[1, 0, 2] = 2.0 / lam, -2.0 / lam
    Gamma = np.array([[[0.5 * (c[a, k, b] - c[k, b, a] + c[b, a, k]) for k in range(3)] for b in range(3)] for a in range(3)])
    Riem = np.array([[[[sum(Gamma[a, e, k] * Gamma[e, b, l] - Gamma[a, e, l] * Gamma[e, b, k] - c[e, k, l] * Gamma[a, b, e] for e in range(3)) for l in range(3)] for k in range(3)] for b in range(3)] for a in range(3)])
    M_raw = [np.array([[np.sum(T_basis_np[a] * (Gamma[:, :, k] @ T_basis_np[b] - T_basis_np[b] @ Gamma[:, :, k])) for b in range(5)] for a in range(5)], dtype=complex) for k in range(3)]
    K_riem = np.array([[np.sum(T_basis_np[a] * (-2.0 * np.einsum("ikjl,kl->ij", Riem, T_basis_np[b]))) for b in range(5)] for a in range(5)], dtype=complex)
    M_sph = [U_S3_np.conj().T @ Mk @ U_S3_np for Mk in M_raw]
    return {
        "lam": lam,
        "inv_lam2_m1": 4.0 * (1.0 / (lam * lam) - 1.0),
        "Op_const": -(M_sph[0] @ M_sph[0] + M_sph[1] @ M_sph[1] + M_sph[2] @ M_sph[2]) + U_S3_np.conj().T @ K_riem @ U_S3_np,
        "Op_Jp": 2j * (M_sph[0] - 1j * M_sph[1]),
        "Op_Jm": 2j * (M_sph[0] + 1j * M_sph[1]),
        "Op_J3": (4j / lam) * M_sph[2],
        "Div_const": C_sph_np[0] @ M_sph[0] + C_sph_np[1] @ M_sph[1] + C_sph_np[2] @ M_sph[2],
        "Div_Jp": -1j * C_p_np,
        "Div_Jm": -1j * C_m_np,
        "Div_J3": (-2j / lam) * C_0_np,
    }


def block_tt_eigs_np(two_j: int, two_M: int, pre: dict):
    j = 0.5 * two_j
    if (two_M - two_j) % 2 != 0:
        return np.empty(0, dtype=float)
    s_idx, s_mJ = [], []
    for i in range(5):
        two_mJ = two_M - 2 * (i - 2)
        if abs(two_mJ) <= two_j:
            s_idx.append(i)
            s_mJ.append(0.5 * two_mJ)
    ns = len(s_idx)
    if ns == 0:
        return np.empty(0, dtype=float)
    v_idx, v_mJ = [], []
    for i in range(3):
        two_mJ = two_M - 2 * (i - 1)
        if abs(two_mJ) <= two_j:
            v_idx.append(i)
            v_mJ.append(0.5 * two_mJ)
    nv = len(v_idx)
    Op_b = np.zeros((ns, ns), dtype=complex)
    c_j2, c_lam = 4.0 * j * (j + 1.0), pre["inv_lam2_m1"]
    for r in range(ns):
        ir, mr = s_idx[r], s_mJ[r]
        for col in range(ns):
            ic, mc = s_idx[col], s_mJ[col]
            val = 0.0j
            if abs(mr - mc) < 1e-9:
                val += pre["Op_const"][ir, ic] + mc * pre["Op_J3"][ir, ic] + (c_j2 + c_lam * (mc * mc) if ir == ic else 0.0)
            elif abs(mr - (mc + 1.0)) < 1e-9:
                val += math.sqrt(j * (j + 1.0) - mc * (mc + 1.0)) * pre["Op_Jp"][ir, ic]
            elif abs(mr - (mc - 1.0)) < 1e-9:
                val += math.sqrt(j * (j + 1.0) - mc * (mc - 1.0)) * pre["Op_Jm"][ir, ic]
            Op_b[r, col] = val
    if nv == 0:
        return np.array([Op_b[0, 0].real], dtype=float)
    Div_b = np.zeros((nv, ns), dtype=complex)
    for r in range(nv):
        vr, mr = v_idx[r], v_mJ[r]
        for col in range(ns):
            ic, mc = s_idx[col], s_mJ[col]
            val = 0.0j
            if abs(mr - mc) < 1e-9:
                val += pre["Div_const"][vr, ic] + mc * pre["Div_J3"][vr, ic]
            elif abs(mr - (mc + 1.0)) < 1e-9:
                val += math.sqrt(j * (j + 1.0) - mc * (mc + 1.0)) * pre["Div_Jp"][vr, ic]
            elif abs(mr - (mc - 1.0)) < 1e-9:
                val += math.sqrt(j * (j + 1.0) - mc * (mc - 1.0)) * pre["Div_Jm"][vr, ic]
            Div_b[r, col] = val
    _, sv, Vh = np.linalg.svd(Div_b, full_matrices=True)
    tol = max(1e-9, 1e-10 * (sv[0] if len(sv) > 0 else 1.0))
    null_mask = np.ones(ns, dtype=bool)
    null_mask[:len(sv)] = (sv < tol)
    if not np.any(null_mask):
        return np.empty(0, dtype=float)
    Q = Vh[null_mask].conj().T
    H_tt = Q.conj().T @ Op_b @ Q
    return np.linalg.eigvalsh(0.5 * (H_tt + H_tt.conj().T)).real


# 1. Prolate non-stretched scan (|M_L| < j+2, 2j <= 40, lam in [1.2, 6.0])
prolate_ns_tachyons = []
prolate_ns_min = 1e99
for lam_val in [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
    pre = berger_precompute_np(lam_val)
    for two_j in range(41):
        start_2M = 0 if (two_j % 2 == 0) else 1
        for two_M in range(start_2M, two_j + 3, 2):
            evs = block_tt_eigs_np(two_j, two_M, pre)
            if len(evs) > 0:
                prolate_ns_min = min(prolate_ns_min, float(evs[0]))
                if evs[0] < 0:
                    prolate_ns_tachyons.append((lam_val, two_j, 0.5 * two_M, float(evs[0])))

# 2. Oblate scan across ALL modes (except j=0, M_L=0 which vanishes at lam=0.50!) for lam in [0.51, 0.90], 2j <= 40!
oblate_other_min = 1e99
oblate_other_min_loc = None
for lam_val in [0.51, 0.55, 0.65, 0.75, 0.85, 0.95]:
    pre = berger_precompute_np(lam_val)
    for two_j in range(41):
        start_2M = 0 if (two_j % 2 == 0) else 1
        for two_M in range(start_2M, two_j + 5, 2):
            if two_j == 0 and two_M == 0:
                continue
            evs = block_tt_eigs_np(two_j, two_M, pre)
            if len(evs) > 0 and evs[0] < oblate_other_min:
                oblate_other_min = float(evs[0])
                oblate_other_min_loc = (lam_val, two_j, 0.5 * two_M)

print(f"  Prolate non-stretched (2j<=40, lam<=6): {len(prolate_ns_tachyons)} tachyons (all (2j=1, M_L=1.5) above lam=3.2438)")
print(f"  Oblate ALL other modes (2j<=40, lam in [0.51, 0.95]): min eig = {oblate_other_min:+.6f} at {oblate_other_min_loc} (ZERO oblate undercuts!)")
assert oblate_other_min > 5.0
assert all(t[1] == 1 and abs(t[2] - 1.5) < 1e-9 for t in prolate_ns_tachyons)

# 3. Symbolic vs numerical + full-matrix vs block + 16-quotient prolate table
max_sym_vs_num_err = 0.0
j_sym, lam_sym = sp.Symbol("j", positive=True), sp.Symbol("lam", positive=True)
f_tr = sp.lambdify((j_sym, lam_sym), sp.sympify(kt2_sym["M0_positivity_certificates"]["Trace_omega2"]["exact_formula"]))
f_det = sp.lambdify((j_sym, lam_sym), sp.sympify(kt2_sym["M0_positivity_certificates"]["Det_omega2"]["exact_formula"]))
for lam_val in [0.55, 1.0, 1.75, 3.2]:
    pre = berger_precompute_np(lam_val)
    for two_j in range(17):
        j_v = 0.5 * two_j
        ev_num = block_tt_eigs_np(two_j, two_j + 4, pre)[0]
        max_sym_vs_num_err = max(max_sym_vs_num_err, abs(ev_num - (4.0 * (j_v + 2.0) ** 2 / lam_val**2 - 4.0 * (j_v + 2.0))))
    for j_int in range(2, 11):
        evs_num = np.sort(block_tt_eigs_np(2 * j_int, 0, pre))
        tr_s, det_s = float(f_tr(j_int, lam_val)), float(f_det(j_int, lam_val))
        max_sym_vs_num_err = max(max_sym_vs_num_err, abs(evs_num[0] + evs_num[1] - tr_s) / abs(tr_s), abs(evs_num[0] * evs_num[1] - det_s) / abs(det_s))
assert max_sym_vs_num_err < 1e-11


def full_matrix_tt_eigs_check(two_j: int, lam: float):
    j = 0.5 * two_j
    c = np.zeros((3, 3, 3))
    c[2, 0, 1], c[2, 1, 0] = 2.0 * lam, -2.0 * lam
    c[0, 1, 2], c[0, 2, 1] = 2.0 / lam, -2.0 / lam
    c[1, 2, 0], c[1, 0, 2] = 2.0 / lam, -2.0 / lam
    Gamma = np.array([[[0.5 * (c[a, k, b] - c[k, b, a] + c[b, a, k]) for k in range(3)] for b in range(3)] for a in range(3)])
    Riem = np.array([[[[sum(Gamma[a, e, k] * Gamma[e, b, l] - Gamma[a, e, l] * Gamma[e, b, k] - c[e, k, l] * Gamma[a, b, e] for e in range(3)) for l in range(3)] for k in range(3)] for b in range(3)] for a in range(3)])
    dim = two_j + 1
    ms = [j - k for k in range(dim)]
    J3 = np.diag(ms).astype(complex)
    Jp, Jm = np.zeros((dim, dim), dtype=complex), np.zeros((dim, dim), dtype=complex)
    for k, m in enumerate(ms):
        if k > 0:
            Jp[k - 1, k] = math.sqrt(j * (j + 1) - m * (m + 1))
        if k < dim - 1:
            Jm[k + 1, k] = math.sqrt(j * (j + 1) - m * (m - 1))
    J1, J2, I_dim = 0.5 * (Jp + Jm), -0.5j * (Jp - Jm), np.eye(dim, dtype=complex)
    E_ops = [-2j * J1, -2j * J2, (-2j / lam) * J3]
    M_conn = [np.array([[np.sum(T_basis_np[a] * (Gamma[:, :, k] @ T_basis_np[b] - T_basis_np[b] @ Gamma[:, :, k])) for b in range(5)] for a in range(5)]) for k in range(3)]
    D_ops = [np.kron(np.eye(5), E_ops[k]) + np.kron(M_conn[k], I_dim) for k in range(3)]
    K_riem = np.array([[np.sum(T_basis_np[a] * (-2.0 * np.einsum("ikjl,kl->ij", Riem, T_basis_np[b]))) for b in range(5)] for a in range(5)])
    Op1 = -(D_ops[0] @ D_ops[0] + D_ops[1] @ D_ops[1] + D_ops[2] @ D_ops[2]) + np.kron(K_riem, I_dim)
    Div = np.zeros((3 * dim, 5 * dim), dtype=complex)
    for a in range(3):
        for k in range(3):
            for al in range(5):
                cka = T_basis_np[al][k, a]
                if abs(cka) > 1e-15:
                    Div[a * dim:(a + 1) * dim, :] += cka * D_ops[k][al * dim:(al + 1) * dim, :]
    _, sv, Vh = np.linalg.svd(Div, full_matrices=True)
    nm = np.ones(5 * dim, dtype=bool)
    nm[:len(sv)] = (sv < 1e-9)
    Q = Vh[nm].conj().T
    return np.sort(np.linalg.eigvalsh(0.5 * (Q.conj().T @ Op1 @ Q + (Q.conj().T @ Op1 @ Q).conj().T)).real)


max_full_vs_blk_err = 0.0
for lam_t in [0.65, 1.0, 1.35]:
    pre = berger_precompute_np(lam_t)
    for two_j in range(9):
        fev = full_matrix_tt_eigs_check(two_j, lam_t)
        bev = np.sort(np.array([x for tM in range(-two_j - 4, two_j + 5, 2) for x in block_tt_eigs_np(two_j, tM, pre)], dtype=float))
        max_full_vs_blk_err = max(max_full_vs_blk_err, float(np.max(np.abs(fev - bev))))
assert max_full_vs_blk_err < 1e-11


def min_eig_quotient(lam: float, kind: str, param: int, max_2j: int = 36):
    pre = berger_precompute_np(lam)
    best, best_mode = 1e99, None
    for two_j in range(max_2j + 1):
        start_2M = 0 if (two_j % 2 == 0) else 1
        for two_M in range(start_2M, two_j + 5, 2):
            if kind == "S3":
                allowed = True
            elif kind == "L":
                allowed = (two_M % param == 0)
            else:
                allowed = (two_M % (2 * param) == 0) and (two_M > 0 or two_j % 4 == 0)
            if not allowed:
                continue
            evs = block_tt_eigs_np(two_j, two_M, pre)
            if len(evs) > 0 and evs[0] < best:
                best, best_mode = float(evs[0]), (two_j, 0.5 * two_M)
    return best, best_mode


def exact_prolate_formula(kind: str, param: int) -> float:
    if kind == "S3" or (kind == "L" and param in (2, 4)):
        return math.sqrt(2.0)
    if kind == "L" and param == 3:
        return math.sqrt(3.0)
    if kind == "L" and param >= 5:
        return math.sqrt(0.5 * param)
    return math.sqrt(float(param))


quotient_prolate_table = {}
max_prolate_formula_err = 0.0
for label, kind, param in [
    ("S^3", "S3", 1), ("RP^3", "L", 2), ("L(3,1)", "L", 3), ("L(4,1)", "L", 4),
    ("L(5,1)", "L", 5), ("L(6,1)", "L", 6), ("L(8,1)", "L", 8), ("L(11,1)", "L", 11),
    ("L(14,1)", "L", 14), ("L(20,1)", "L", 20), ("Q_8", "Dm", 2), ("D_3*", "Dm", 3),
    ("D_4*", "Dm", 4), ("D_6*", "Dm", 6), ("D_8*", "Dm", 8), ("D_10*", "Dm", 10),
]:
    pred = exact_prolate_formula(kind, param)
    root = brentq(lambda x: min_eig_quotient(x, kind, param, max_2j=40)[0], pred - 0.05, pred + 0.05, xtol=1e-10)
    _, m_hi = min_eig_quotient(pred + 0.05, kind, param, max_2j=40)
    err = abs(root - pred)
    max_prolate_formula_err = max(max_prolate_formula_err, err)
    quotient_prolate_table[label] = {
        "lam_oblate": 0.5,
        "lam_prolate_num": float(root),
        "lam_prolate_exact": float(pred),
        "abs_err": float(err),
        "first_tachyon_2j": m_hi[0],
        "first_tachyon_ML": m_hi[1],
    }

print(f"  Symbolic vs num err: {max_sym_vs_num_err:.3e} | Full vs block err: {max_full_vs_blk_err:.3e} | 16-quotient table err: {max_prolate_formula_err:.3e}")
assert max_prolate_formula_err < 1e-9

out_derive = {
    "stream_id": "p2_04_graviton_invariance_and_berger_stability",
    "kt1a_nonlinear_alpha_identity": kt1a_nonlin,
    "kt1b_sachs_optical_scalar_and_geodesics": kt1b_sachs,
    "kt1c_ade_50digit_invariance": kt1c_ade,
    "kt2_symbolic_prolate_proof": kt2_sym,
    "kt2_numerical_verification": {
        "max_2j": 40,
        "prolate_lam_grid": [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
        "prolate_non_stretched_min": prolate_ns_min,
        "prolate_non_stretched_tachyons": prolate_ns_tachyons,
        "oblate_lam_grid": [0.51, 0.55, 0.65, 0.75, 0.85, 0.95],
        "oblate_all_other_modes_min": oblate_other_min,
        "oblate_all_other_modes_min_loc": oblate_other_min_loc,
        "symbolic_vs_numerical_max_err": max_sym_vs_num_err,
        "full_matrix_vs_block_max_err": max_full_vs_blk_err,
        "max_prolate_formula_err": max_prolate_formula_err,
        "quotient_prolate_table": quotient_prolate_table,
    },
}

out_path = STREAM_DIR / "01_derive.json"
with open(out_path, "w") as fh:
    json.dump(out_derive, fh, indent=2)
print(f"\nSaved {out_path}")
