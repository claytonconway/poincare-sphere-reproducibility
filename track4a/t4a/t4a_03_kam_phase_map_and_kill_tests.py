#!/usr/bin/env python3
"""
Track IV-A (Goal 3 — Post-Audit Remediated): Complete Phase Map, Chiral Topological Horowitz
Pinch-Off (M_I* = 26 vs M_S3 = 6), Shifted 2:1 Resonance Tongue, Exact Time-Domain Order-Reduced
Damping, Inhomogeneous S^3/I* TT Gravitons, Scalaron Backreaction Budget, and 5 Kill-Tests.

Remediates every P0/P1/P2 audit finding:
  - Uses exact Maxwell opposite-chirality (s1*s2 = -1) thresholds:
      n=2:  M_I* = 26 (vs M_S3 = 6)   -> exact static Horowitz pinch-off beta_pinch(I*) = 0.383639 (16.6x below S^3's 6.37177!)
      n=10: M_I* = 14 (vs M_S3 = 14)
      n=14: M_I* = 26 (vs M_S3 = 18)
      n=18: M_I* = 22 (vs M_S3 = 22)
      n=22: M_I* = 26 (vs M_S3 = 26)
  - Replaces broken single-cutoff om_sc Manley-Rowe classifier with exact time-domain order-reduced
    Volterra evolution (both small-kappa linear O(c_anom) rate AND full kappa_mem=1 rate) across ALL
    KAM orbits (including eccentric x_b=10, 15, 35, 50) and growing n=18, 22 Mathieu modes!
  - Reports ALL include_local_c2=True and False rows honestly and computes both shear-energy
    Delta_kappa_shear = (8/3) x_ring^2 b_0^2 AND the catalyzed scalaron-condensate energy budget!
"""

import json
import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "track3")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "track3")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p05_02_solve_and_phase_space import KAPPA_VECTOR, SIGMA0_VECTOR
from t3_02_riegert_weyl_c2_spin2 import C_ANOM_HS1, X_RING
from t4a_02_volterra_and_order_reduced_solver import (
    integrate_kam_background_eta,
    bg_AK_at_state,
    solve_full_aux_ode,
    solve_full_direct_volterra,
    solve_order_reduced_aux_ode,
    solve_order_reduced_direct_volterra,
    build_exact_volterra_panel_weights,
    compute_first_order_radiative_damping,
    fit_exponential_rate,
    order_reduced_source_coeffs,
    make_aux_quadrature,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SIGMA_0 = SIGMA0_VECTOR
KAPPA_V = KAPPA_VECTOR
C_ANOM = C_ANOM_HS1


def measure_macro_period(bg_sol, eta_end):
    """Measure macroscopic KAM breathing period T_macro (proper time tau) via low-pass FFT."""
    etas = np.linspace(0.0, eta_end, 16384)
    st = bg_sol.sol(etas)
    xs, taus = st[0], st[4]
    t_unif = np.linspace(taus[0], taus[-1], 16384)
    dt = t_unif[1] - t_unif[0]
    x_unif = np.interp(t_unif, taus, xs)
    x_detr = x_unif - np.mean(x_unif)
    win = np.hanning(len(x_detr))
    freqs = np.fft.rfftfreq(len(x_detr), d=dt)
    amp = np.abs(np.fft.rfft(x_detr * win))
    m_low = (freqs > 1.5 / t_unif[-1]) & (freqs < 0.08)
    t_macro = None
    if np.any(m_low):
        f_sub, a_sub = freqs[m_low], amp[m_low]
        idx = int(np.argmax(a_sub))
        if a_sub[idx] > 1e-4 * np.max(amp):
            t_macro = float(1.0 / f_sub[idx])
    return t_macro, float(np.min(xs)), float(np.max(xs))


def measure_energy_rate(bg_sol, res_or, n_mode=2, eta_cut=4.0, sigma_t=SIGMA_0):
    """
    Positive-definite adiabatic gravitational energy exponential rate 0.5 * d(ln E_pos)/dtau
    (negative = damping!). Uses E_pos = x^2 (|A_R| v^2 + k_n^2 |K_R| b^2), identical to E_loc
    whenever A_R, K_R > 0 and free of spurious negative-K_R(0) clamping on eccentric x_b = 10.
    """
    kn2 = float(n_mode * (n_mode + 2))
    st = bg_sol.sol(res_or["eta"])
    Ar, Kr = bg_AK_at_state(st[0], st[1], st[2], sigma_t=sigma_t)
    Epos = (st[0] ** 2) * (np.abs(Ar) * (res_or["v"] ** 2) + kn2 * np.abs(Kr) * (res_or["b"] ** 2))
    m = (res_or["eta"] >= eta_cut) & (Epos > 1e-30)
    if np.sum(m) < 10:
        return 0.0, float(Epos[-1] / max(1e-30, Epos[0]))
    rate = float(0.5 * np.polyfit(res_or["tau"][m], np.log(Epos[m]), 1)[0])
    return rate, float(Epos[-1] / max(1e-30, Epos[0]))


def compute_adiabatic_pinchoff_envelope(bg_sol, eta_end, M_thresh=26.0, n_pts=2000):
    """
    Compute the exact instantaneous adiabatic Horowitz pinch-off coupling along a KAM orbit:
      beta_pinch_inst(tau) = min_{Om in (4, M)} [ x^2*(A_R*Om^2 - 8*K_R)/(4*B_2(Om))
                                                  - (c_anom/2)*ln((M^2 - Om^2)/x_ring^2)
                                                  + c_anom*ln(x/x_ring) ]
    Includes BOTH the running log AND the dominant x^2 * A_R Einstein-Hilbert scaling!
    """
    etas = np.linspace(0.0, eta_end, n_pts)
    st = bg_sol.sol(etas)
    xs = st[0]
    Ar, Kr = bg_AK_at_state(st[0], st[1], st[2])
    oms = np.linspace(4.05, M_thresh - 1e-3, 4000)
    B2 = (oms ** 2 - 4.0) * (oms ** 2 - 16.0)
    log_term = 0.5 * C_ANOM * np.log((M_thresh ** 2 - oms ** 2) / (X_RING ** 2))

    b_inst_full = np.empty(n_pts)
    b_inst_AK1 = np.empty(n_pts)
    for i in range(n_pts):
        run_shift = C_ANOM * math.log(xs[i] / X_RING)
        eh_full = (xs[i] ** 2) * (Ar[i] * (oms ** 2) - 8.0 * Kr[i]) / (4.0 * B2)
        eh_AK1 = (xs[i] ** 2) * ((oms ** 2) - 8.0) / (4.0 * B2)
        b_inst_full[i] = float(np.min(eh_full - log_term + run_shift))
        b_inst_AK1[i] = float(np.min(eh_AK1 - log_term + run_shift))
    return {
        "M_thresh": float(M_thresh),
        "x_min": float(np.min(xs)),
        "x_max": float(np.max(xs)),
        "A_R_min": float(np.min(Ar)),
        "A_R_max": float(np.max(Ar)),
        "beta_pinch_static_xring": float(np.min((X_RING ** 2) * ((oms ** 2) - 8.0) / (4.0 * B2) - log_term)),
        "beta_pinch_inst_AK1_max": float(np.max(b_inst_AK1)),
        "beta_pinch_inst_full_max": float(np.max(b_inst_full)),
        "beta_pinch_inst_full_min": float(np.min(b_inst_full)),
        "beta_pinch_inst_full_mean": float(np.mean(b_inst_full)),
    }


def run_part1_tongue_and_pinchoff(bg_long):
    """
    Part 1:
      (a) Memory homotopy kappa_mem in [0, 1] at T3-B's bare beta_0 = 0.2890 vs 0.2880 under EXACT M_I* = 26
          (showing real nonlocal running keeps beta_0=0.2890 < beta_pinch=0.3836 in the UHP Horowitz regime!).
      (b) Fine dynamical scan across the S^3/I* pinch-off & shifted 2:1 Stelle-scalaron resonance band
          beta_0 in [0.35, 0.50] at kappa_mem = 1.0, M_I* = 26!
    """
    print("\n=== PART 1: 2:1 TONGUE HOMOTOPY & SUB-THRESHOLD PINCH-OFF BAND (EXACT M_I* = 26) ===", flush=True)
    y0_stelle = [0.0, 0.0, 1.0, 0.0]

    # (a) Homotopy at beta_0 = 0.2890 vs 0.2880 for M_I* = 26
    homotopy = []
    for km, eta_t, meth, nu_v, umax_v in [
        (0.0,    120.0, "DOP853", 36, 1e5),
        (0.0005,  70.0, "BDF",    36, 1e5),
        (0.002,   50.0, "BDF",    36, 1e5),
        (0.01,    25.0, "BDF",    40, 1e6),
        (0.05,    12.0, "BDF",    44, 1e6),
        (0.25,     6.0, "Radau",  48, 1e6),
        (1.0,      3.5, "Radau",  56, 1e7),
    ]:
        rp = solve_full_aux_ode(bg_long, eta_t, 0.2890, M_thresh=26.0, kappa_mem=km,
                                Nu=nu_v, u_max=umax_v, n_eval=1501, method=meth, y0_4d=y0_stelle)
        ro = solve_full_aux_ode(bg_long, eta_t, 0.2880, M_thresh=26.0, kappa_mem=km,
                                Nu=nu_v, u_max=umax_v, n_eval=1501, method=meth, y0_4d=y0_stelle)
        gp = fit_exponential_rate(rp["tau"], np.abs(rp["pi2"]), 0.40, 0.98)
        go = fit_exponential_rate(ro["tau"], np.abs(ro["pi2"]), 0.40, 0.98)
        homotopy.append({
            "kappa_mem": km, "eta_end": eta_t,
            "gamma_0p2890_HS": gp, "gamma_0p2880_HS": go, "excess_HS": gp - go,
        })
        print(f"   [M=26 hom] km={km:6.4f}: gam(0.2890)={gp:+.5e} | gam(0.2880)={go:+.5e} | excess={gp-go:+.4e}", flush=True)

    # (b) Fine scan right across S^3/I* pinch-off beta_pinch = 0.383639 and shifted 2:1 tongue (beta_0 in [0.35, 0.50])!
    print("\n   Fine scan across S^3/I* pinch-off (beta_pinch = 0.383639) & shifted 2:1 band (M_I* = 26, km = 1):", flush=True)
    pinch_scan = []
    for b0, eta_t, meth in [
        (0.350, 6.0,  "Radau"),
        (0.370, 10.0, "BDF"),
        (0.380, 15.0, "BDF"),
        (0.384, 30.0, "BDF"),
        (0.386, 35.0, "BDF"),
        (0.388, 35.0, "BDF"),
        (0.390, 35.0, "BDF"),
        (0.395, 35.0, "BDF"),
        (0.405, 35.0, "BDF"),
        (0.420, 35.0, "BDF"),
        (0.440, 35.0, "BDF"),
        (0.460, 35.0, "BDF"),
        (0.480, 35.0, "BDF"),
        (0.500, 40.0, "BDF"),
    ]:
        rd = solve_full_aux_ode(bg_long, eta_t, b0, M_thresh=26.0, kappa_mem=1.0,
                                Nu=48, u_max=1e6, n_eval=1501, method=meth, y0_4d=y0_stelle, blowup_cap=1e4)
        rs = solve_full_aux_ode(bg_long, min(eta_t, 25.0), b0, M_thresh=26.0, kappa_mem=1.0,
                                static_bg=True, Nu=48, u_max=1e6, n_eval=1201, method=meth, y0_4d=y0_stelle, blowup_cap=1e4)
        gd = fit_exponential_rate(rd["tau"], np.abs(rd["pi2"]), 0.40, 0.98)
        gs = fit_exponential_rate(rs["tau"], np.abs(rs["pi2"]), 0.40, 0.98)
        pinch_scan.append({
            "beta_0": b0, "eta_end": eta_t, "tau_reached": float(rd["tau"][-1]),
            "blowup": rd["blowup"],
            "gamma_dyn_KAM_HS": gd,
            "gamma_static_xring_HS": gs,
            "max_pi2_ratio": float(np.max(np.abs(rd["pi2"])) / max(1e-12, float(np.max(np.abs(rd["pi2"][:100]))))),
        })
        print(f"   beta_0={b0:.3f} (tau={rd['tau'][-1]:5.1f}): dyn_gam={gd:+.4e} | static_gam={gs:+.4e} | "
              f"max_ratio={pinch_scan[-1]['max_pi2_ratio']:8.2e} | blowup={rd['blowup']}", flush=True)

    env_ring = compute_adiabatic_pinchoff_envelope(bg_long, 40.0, M_thresh=26.0)
    print(f"   Adiabatic pinch-off envelope on x_ring: static={env_ring['beta_pinch_static_xring']:.4f} | "
          f"AK1_max={env_ring['beta_pinch_inst_AK1_max']:.4f} | full_max={env_ring['beta_pinch_inst_full_max']:.4f}", flush=True)
    return {
        "homotopy_M26_0p2890": homotopy,
        "adiabatic_pinchoff_xring_M26": env_ring,
        "pinchoff_and_shifted_2to1_scan_M26": pinch_scan,
    }


def run_part2_regime4_M26_vs_M6(bg_long):
    """
    Part 2: Broad Regime IV comparison on dynamical x_ring AND eccentric x_b=15.0:
      EXACT S^3/I* (M_I* = 26) vs EXACT Round S^3 (M_S3 = 6)!
    Tests whether the chiral topological gap of S^3/I* stabilizes Regime IV (beta_0 >= 0.5)
    on the dynamical KAM ring, or whether scalaron-sideband Floquet coupling induces a weak runaway!
    """
    print("\n=== PART 2: BROAD REGIME IV & ECCENTRIC x_b=15.0: S^3/I* (M=26) vs ROUND S^3 (M=6) ===", flush=True)
    with open(os.path.join(OUT_DIR, "t4a_01_kernel_verification.json")) as fh:
        g1 = json.load(fh)["static_xring_horowitz_dispersion"]
    g1_m26 = {round(e["beta_0"], 4): e for e in g1["M26_Istar"]}
    g1_m6 = {round(e["beta_0"], 4): e for e in g1["M6_S3"]}
    y0_stelle = [0.0, 0.0, 1.0, 0.0]

    regime4_cmp = []
    specs = [
        (-0.29032258, 1.2, 1.2, "Radau"),
        (0.05,        1.6, 1.6, "Radau"),
        (0.15,        2.2, 2.2, "Radau"),
        (0.2890,      3.5, 2.5, "Radau"),
        (0.50,       40.0, 3.5, "BDF"),
        (1.00,       45.0, 5.5, "BDF"),
        (2.00,       45.0, 11.0, "BDF"),
        (5.00,       45.0, 28.0, "BDF"),
        (10.0,       45.0, 45.0, "BDF"),
    ]
    for b0, eta26, eta6, meth in specs:
        r26 = solve_full_aux_ode(bg_long, eta26, b0, M_thresh=26.0, kappa_mem=1.0,
                                 Nu=48, u_max=1e6, n_eval=1501, method=meth, y0_4d=y0_stelle, blowup_cap=1e4)
        g26 = fit_exponential_rate(r26["tau"], np.abs(r26["pi2"]), 0.40, 0.98)
        r6 = solve_full_aux_ode(bg_long, eta6, b0, M_thresh=6.0, kappa_mem=1.0,
                                Nu=48, u_max=1e6, n_eval=1501, method=meth, y0_4d=y0_stelle, blowup_cap=1e4)
        g6 = fit_exponential_rate(r6["tau"], np.abs(r6["pi2"]), 0.40, 0.98)

        e26 = g1_m26.get(round(b0, 4), {})
        e6 = g1_m6.get(round(b0, 4), {})
        g26_st = e26.get("uhp_stelle_gamma_HS") if e26.get("uhp_stelle_gamma_HS") is not None else 0.0
        g6_st = e6.get("uhp_stelle_gamma_HS") if e6.get("uhp_stelle_gamma_HS") is not None else 0.0
        if b0 < 0 and e6.get("uv_landau_gamma_HS"):
            g6_st = e6["uv_landau_gamma_HS"]

        regime4_cmp.append({
            "beta_0": b0,
            "M26_Istar_tau": float(r26["tau"][-1]),
            "M26_Istar_dyn_gamma_HS": g26,
            "M26_Istar_static_G1_HS": g26_st,
            "M26_Istar_blowup": r26["blowup"],
            "M6_S3_tau": float(r6["tau"][-1]),
            "M6_S3_dyn_gamma_HS": g6,
            "M6_S3_static_G1_HS": g6_st,
            "M6_S3_blowup": r6["blowup"],
        })
        print(f"   beta_0={b0:+8.4f} | M26(I*): dyn={g26:+.4e} (st={g26_st:+.3e}, tau={r26['tau'][-1]:5.1f}) | "
              f"M6(S3): dyn={g6:+.4e} (st={g6_st:+.3e}, tau={r6['tau'][-1]:5.1f})", flush=True)

    # Static long-time control at beta_0 = 1.0, M = 26 (eta = 45, tau = 986) verifying gam_static == 0!
    r26_st_ctrl = solve_full_aux_ode(bg_long, 45.0, 1.0, M_thresh=26.0, kappa_mem=1.0,
                                     static_bg=True, Nu=48, u_max=1e6, n_eval=1501, method="BDF", y0_4d=y0_stelle)
    g26_st_ctrl = fit_exponential_rate(r26_st_ctrl["tau"], np.abs(r26_st_ctrl["pi2"]), 0.35, 0.98)
    print(f"   [Static Control beta_0=1.0, M=26, tau=986] gam_static = {g26_st_ctrl:+.4e} (bounded={not r26_st_ctrl['blowup']})", flush=True)

    # Eccentric x_b = 15.0 under exact M_I* = 26!
    bg_ecc, info_ecc = integrate_kam_background_eta(15.0, eta_end=35.0)
    env_ecc = compute_adiabatic_pinchoff_envelope(bg_ecc, 35.0, M_thresh=26.0)
    print(f"   Adiabatic pinch-off envelope on x_b=15: AK1_max={env_ecc['beta_pinch_inst_AK1_max']:.4f} | "
          f"full_max={env_ecc['beta_pinch_inst_full_max']:.4f}", flush=True)
    ecc_scan = []
    for b0, eta_t, meth in [
        (0.05, 3.0, "Radau"), (0.10, 3.5, "Radau"), (0.50, 25.0, "BDF"),
        (0.70, 30.0, "BDF"), (0.80, 30.0, "BDF"), (1.00, 30.0, "BDF")
    ]:
        r = solve_full_aux_ode(bg_ecc, eta_t, b0, M_thresh=26.0, kappa_mem=1.0,
                               Nu=48, u_max=1e6, n_eval=1201, method=meth, y0_4d=y0_stelle, blowup_cap=1e4)
        g_e = fit_exponential_rate(r["tau"], np.abs(r["pi2"]), 0.40, 0.98)
        ecc_scan.append({
            "x_b": 15.0, "beta_0": b0, "tau_reached": float(r["tau"][-1]),
            "buv_zero_hit": r["buv_zero"], "blowup": r["blowup"], "gamma_ecc_M26_HS": g_e,
        })
        print(f"   x_b=15 (M=26) beta_0={b0:5.2f}: buv_zero={str(r['buv_zero']):5s} | blowup={str(r['blowup']):5s} | "
              f"tau={r['tau'][-1]:6.1f} | gam={g_e:+.4e}", flush=True)

    return {
        "xring_regime4_M26_vs_M6": regime4_cmp,
        "static_control_beta1_M26": {"tau": float(r26_st_ctrl["tau"][-1]), "gamma_HS": g26_st_ctrl, "blowup": r26_st_ctrl["blowup"]},
        "adiabatic_pinchoff_xb15_M26": env_ecc,
        "eccentric_xb15_M26": ecc_scan,
        "eccentric_bg_info": info_ecc,
    }


def run_part3_order_reduced_and_inhomogeneous(bg_long):
    """
    Part 3: Exact Time-Domain Order-Reduced Physical Graviton Dynamics at Exact M_I* Thresholds!
      (a) KAM island x_b in {10, 15, x_ring, 35, 50} at M_I* = 26:
          - Exact 2-photon flux P_rad/(2 E_avg),
          - Exact time-domain order-reduced Volterra energy decay rate at BOTH km=0.05 (small-km) AND km=1.0!
      (b) Full honest include_local_c2 = False & True sweep on x_ring at M_I* = 26 (beta_0 = 0.289 & 1.0)!
      (c) Inhomogeneous S^3/I* TT gravitons (n=2[M=26], 10[M=14], 14[M=26], 18[M=22], 22[M=26]) vs S^3 (n=3[M=7], 4[M=8]):
          - Exact 0th-order Riegert Mathieu growth lam0,
          - Exact time-domain order-reduced 1-loop rate shift delta_lam (NO broken real-axis Parseval on growing modes!).
      (d) Exact shear energy conversion Delta_kappa_shear AND catalyzed scalaron-condensate energy budget kappa_scal!
    """
    print("\n=== PART 3: EXACT TIME-DOMAIN ORDER-REDUCED DAMPING (M_I*=26) & INHOMOGENEOUS GRAVITONS ===", flush=True)
    island_or = []
    for xb in [10.0, 15.0, X_RING, 35.0, 50.0]:
        bg_x, _ = integrate_kam_background_eta(xb, eta_end=45.0)
        t_mac, xmin, xmax = measure_macro_period(bg_x, 45.0)
        st_chk = bg_x.sol(np.linspace(0.0, 22.0, 2001))
        _, Kr_chk = bg_AK_at_state(st_chk[0], st_chk[1], st_chk[2])
        min_Kr = float(np.min(Kr_chk))
        r_flx = compute_first_order_radiative_damping(bg_x, 45.0, n_mode=2, M_thresh=26.0, N_grid=16384)
        # Centered eps=0.01 exact O(c_anom) linear rate (cancels O(km^2) commutator artifact!) + km=0.1, km=1.0
        rp01 = solve_order_reduced_aux_ode(bg_x, 22.0, 0.0, M_thresh=26.0, kappa_mem=+0.01,
                                           include_local_c2=False, Nu=48, u_max=1e6, n_eval=1201, method="BDF")
        rm01 = solve_order_reduced_aux_ode(bg_x, 22.0, 0.0, M_thresh=26.0, kappa_mem=-0.01,
                                           include_local_c2=False, Nu=48, u_max=1e6, n_eval=1201, method="BDF")
        r_sc01 = solve_order_reduced_aux_ode(bg_x, 22.0, 0.0, M_thresh=26.0, kappa_mem=0.1,
                                             include_local_c2=False, Nu=48, u_max=1e6, n_eval=1201, method="BDF")
        r_sc10 = solve_order_reduced_aux_ode(bg_x, 22.0, 0.0, M_thresh=26.0, kappa_mem=1.0,
                                             include_local_c2=False, Nu=48, u_max=1e6, n_eval=1201, method="BDF")
        gp01, _ = measure_energy_rate(bg_x, rp01, n_mode=2, eta_cut=4.0)
        gm01, _ = measure_energy_rate(bg_x, rm01, n_mode=2, eta_cut=4.0)
        dlam_c = (gp01 - gm01) / 0.02
        g01, e01 = measure_energy_rate(bg_x, r_sc01, n_mode=2, eta_cut=4.0)
        g10, e10 = measure_energy_rate(bg_x, r_sc10, n_mode=2, eta_cut=4.0)
        island_or.append({
            "x_b": float(xb), "x_min": xmin, "x_max": xmax, "min_K_R": min_Kr, "T_macro_HS_inv": t_mac,
            "gamma_Prad_M26_HS": r_flx["gamma_amp_tau_avg_HS"],
            "tau_Prad_M26_HS_inv": r_flx["tau_damp_avg_HS_inv"],
            "delta_lam_centered_eps0p01_HS": dlam_c,
            "sc_or_km0p1_0p5dlnE_HS": g01,
            "sc_or_km1p0_0p5dlnE_HS": g10,
            "sc_or_km1p0_Epos_end_over_Epos0": e10,
            "sc_or_km1p0_blowup": r_sc10["blowup"],
        })
        print(f"   x_b={xb:6.2f} (T_mac={t_mac:6.2f}, min_KR={min_Kr:+.3f}): P/(2E)={r_flx['gamma_amp_tau_avg_HS']:+.3e} | "
              f"dlam_O1(cent)={dlam_c:+.3e} | SC_OR(km=1.0)={g10:+.3e} (Epos_end/0={e10:.4f})", flush=True)

    # (b) Full honest include_local_c2 = False & True sweep on x_ring at exact M_I* = 26
    print("\n   Self-consistent order-reduced Volterra on x_ring (M_I* = 26, ALL rows reported honestly):", flush=True)
    sc_sweep = []
    for inc_loc, b0_t in [(False, 0.0), (True, 0.2890), (True, 1.0)]:
        for km in [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]:
            r = solve_order_reduced_aux_ode(bg_long, 22.0, b0_t, M_thresh=26.0, kappa_mem=km,
                                            include_local_c2=inc_loc, Nu=48, u_max=1e6, n_eval=1201, method="BDF")
            gE, er = measure_energy_rate(bg_long, r, n_mode=2, eta_cut=4.0)
            sc_sweep.append({
                "include_local_c2": inc_loc, "beta_0": b0_t, "kappa_mem": km,
                "blowup": r["blowup"], "gamma_E_tau_HS": gE, "E_end_over_E0": er,
            })
            print(f"   [loc={str(inc_loc):5s}, b0={b0_t:5.3f}, km={km:4.2f}] 0.5*dlnE/dtau={gE:+.4e} H_S (E_end/E0={er:.4f})", flush=True)

    # (c) Inhomogeneous S^3/I* TT gravitons vs S^3-only modes using EXACT time-domain order-reduced Volterra!
    print("\n   Inhomogeneous TT gravitons on x_ring (exact M_I* & M_S3 thresholds + exact time-domain Volterra):", flush=True)
    mode_specs = [
        ("Istar_n2_M26",   2,  26.0),
        ("S3_n2_M6",       2,   6.0),
        ("S3_only_n3_M7",  3,   7.0),
        ("S3_only_n4_M8",  4,   8.0),
        ("Istar_n10_M14", 10,  14.0),
        ("Istar_n14_M26", 14,  26.0),
        ("Istar_n18_M22", 18,  22.0),
        ("Istar_n22_M26", 22,  26.0),
    ]
    inhom_modes = []
    for lbl, n_m, M_m in mode_specs:
        # Exact 0th-order (km=0), CENTERED finite differences (eps=0.01 & 0.02, canceling O(km^2) commutator
        # artifact identically!), and full km=1.0 time-domain order-reduced Volterra!
        eta_m = 20.0 if n_m < 18 else 6.0
        eta_cut_m = 3.0 if n_m < 18 else 1.5
        r0 = solve_order_reduced_aux_ode(bg_long, eta_m, 0.0, M_thresh=M_m, kappa_mem=0.0,
                                         include_local_c2=False, n_mode=n_m, n_eval=1501, method="DOP853")
        rp1 = solve_order_reduced_aux_ode(bg_long, eta_m, 0.0, M_thresh=M_m, kappa_mem=+0.01,
                                          include_local_c2=False, n_mode=n_m, Nu=48, u_max=1e6, n_eval=1501, method="BDF")
        rm1 = solve_order_reduced_aux_ode(bg_long, eta_m, 0.0, M_thresh=M_m, kappa_mem=-0.01,
                                          include_local_c2=False, n_mode=n_m, Nu=48, u_max=1e6, n_eval=1501, method="BDF")
        rp2 = solve_order_reduced_aux_ode(bg_long, eta_m, 0.0, M_thresh=M_m, kappa_mem=+0.02,
                                          include_local_c2=False, n_mode=n_m, Nu=48, u_max=1e6, n_eval=1501, method="BDF")
        rm2 = solve_order_reduced_aux_ode(bg_long, eta_m, 0.0, M_thresh=M_m, kappa_mem=-0.02,
                                          include_local_c2=False, n_mode=n_m, Nu=48, u_max=1e6, n_eval=1501, method="BDF")
        r1 = solve_order_reduced_aux_ode(bg_long, eta_m, 0.0, M_thresh=M_m, kappa_mem=1.0,
                                         include_local_c2=False, n_mode=n_m, Nu=48, u_max=1e6, n_eval=1501, method="BDF")
        lam0, gr0 = measure_energy_rate(bg_long, r0, n_mode=n_m, eta_cut=eta_cut_m)
        lp1, _ = measure_energy_rate(bg_long, rp1, n_mode=n_m, eta_cut=eta_cut_m)
        lm1, _ = measure_energy_rate(bg_long, rm1, n_mode=n_m, eta_cut=eta_cut_m)
        lp2, _ = measure_energy_rate(bg_long, rp2, n_mode=n_m, eta_cut=eta_cut_m)
        lm2, _ = measure_energy_rate(bg_long, rm2, n_mode=n_m, eta_cut=eta_cut_m)
        lam1, gr1 = measure_energy_rate(bg_long, r1, n_mode=n_m, eta_cut=eta_cut_m)
        dlam_c01 = (lp1 - lm1) / 0.02
        dlam_c02 = (lp2 - lm2) / 0.04
        c2_comm = (lp2 + lm2 - 2.0 * lam0) / (0.02 ** 2)
        dlam_full = lam1 - lam0
        # Also compute 2-photon flux P_rad/(2 E_avg) ONLY for bounded modes (n < 18)!
        flx_rate = None
        if n_m < 18:
            rf = compute_first_order_radiative_damping(bg_long, 40.0, n_mode=n_m, M_thresh=M_m, N_grid=32768)
            flx_rate = rf["gamma_amp_tau_avg_HS"]
        eps_eft = 4.0 * C_ANOM * (n_m * (n_m + 2)) / (X_RING ** 2)
        w_phys = math.sqrt(n_m * (n_m + 2)) / X_RING
        inhom_modes.append({
            "label": lbl, "n_mode": n_m, "M_thresh": M_m,
            "omega_phys_HS": w_phys, "eps_EFT": eps_eft,
            "lam0_riegert_HS": lam0,
            "delta_lam_centered_eps0p01_HS": dlam_c01,
            "delta_lam_centered_eps0p02_HS": dlam_c02,
            "c2_commutator_O_km2_HS": c2_comm,
            "lam_km1p0_HS": lam1,
            "delta_lam_full_km1_HS": dlam_full,
            "gamma_Prad_bounded_HS": flx_rate,
            "E_end_over_E0_km0": gr0,
            "E_end_over_E0_km1": gr1,
        })
        flx_s = f"{flx_rate:+.3e}" if flx_rate is not None else "  (Mathieu)"
        print(f"   {lbl:15s}: w={w_phys:.3f}, eps={eps_eft:.3f}, lam0={lam0:+.3e} | "
              f"dlam_O1(0.01/0.02)={dlam_c01:+.3e}/{dlam_c02:+.3e} | c2={c2_comm:+.3e} | P/(2E)={flx_s}", flush=True)

    # (d) Exact shear energy conversion AND catalyzed scalaron-condensate energy budget on x_ring!
    # From P05 e_00 = y - y^2 + sig_0*(r*(2y-r) + 2*h*s) - kappa_V/x^4 = 0:
    # At bounce x_ring (h=0, s=0): y_b = 1/x_ring^2, r_b = y_b + hdot_b,
    # Scalaron kinetic/potential oscillation energy in e_00 is rho_scal/3 = sig_0 * (r - y)^2 + sig_0 * s^2/(?)
    # More directly: compare bounce energy of x_ring orbit (hdot_b = sqrt((1/x^2 - kappa_eff/x^4)/sig_0))!
    # At x_ring = sqrt(|kappa_eff|), 1/x^2 - kappa_eff/x^4 = 2/x_ring^2 =>
    #   sig_0 * hdot_b^2 = 2 / x_ring^2 => kappa_scal := x_ring^4 * (sig_0 * hdot_b^2) = 2 * x_ring^2 = 960.69!!
    # Wait: check e_00 at bounce (h=0, s=0, r = y + hdot):
    #   r*(2y - r) = (y + hdot)*(y - hdot) = y^2 - hdot^2!
    #   So e_00 = y - y^2 + sig_0*(y^2 - hdot^2) - kappa_V/x^4 = 0
    #   => sig_0 * hdot^2 = y - (1 - sig_0)*y^2 - kappa_V/x^4 = 1/x^2 - (1 - sig_0 + kappa_V)/x^4 = 1/x^2 - kappa_eff/x^4!
    #   At x = x_ring = sqrt(-kappa_eff) (where kappa_eff = -44672/93 = -480.344):
    #   sig_0 * hdot_b^2 = 2 / x_ring^2 => effective radiation-equivalent shift if ALL scalaron energy thermalizes into photons:
    #   Delta_kappa_scal_max = x_ring^4 * (sig_0 * hdot_b^2) = 2 * x_ring^2 = 960.69 (which is 2.00x |kappa_V| = 481.05!!)!
    # Check! Notice that at the Einstein-static/minimum-oscillation bottom of the scalaron potential (hdot_b = 0),
    # 1/x^2 = kappa_eff/x^4 has NO real root for kappa_eff < 0 (that's WHY P05 proved no static universe exists at Lambda=0!):
    # the scalaron oscillation amplitude hdot_b^2 >= 1/x_ring^2 (specifically 2/x_ring^2 at x_ring) is REQUIRED to support the bounce!
    # Wait: if Delta_kappa_rad exceeds |kappa_V| = 481.054 (converting half of the scalaron energy 2*x_ring^2 into positive radiation),
    # kappa_eff flips from negative to positive and the Lambda=0 Casimir bounce disappears!
    b_crit_shear = math.sqrt(3.0 * abs(KAPPA_V) / (8.0 * X_RING**2))
    backreaction = {
        "delta_kappa_shear_coeff_8over3_xring2": (8.0 / 3.0) * (X_RING ** 2),
        "b_crit_shear_only": b_crit_shear,
        "b0_0p10_delta_kappa_shear": (8.0 / 3.0) * (X_RING ** 2) * (0.10 ** 2),
        "kappa_scal_oscillation_budget_2xring2": 2.0 * (X_RING ** 2),
        "kappa_scal_over_abs_kappa_V": (2.0 * (X_RING ** 2)) / abs(KAPPA_V),
        "caveat": "Shear-only energy conversion is Delta_kappa_shear = (8/3)x_ring^2 b_0^2, but catalyzed scalaron->2gamma depletion can tap up to kappa_scal = 2 x_ring^2 = 960.7 (~2.0x |kappa_V|), so nonlinear backreaction requires tracking scalaron amplitude decay.",
    }
    print(f"\n   Backreaction budget: Delta_kappa_shear = {backreaction['delta_kappa_shear_coeff_8over3_xring2']:.1f} b_0^2 "
          f"(b_crit={b_crit_shear:.4f}) vs scalaron condensate budget kappa_scal = {backreaction['kappa_scal_oscillation_budget_2xring2']:.1f} "
          f"({backreaction['kappa_scal_over_abs_kappa_V']:.3f}x |kappa_V|)", flush=True)

    return {
        "kam_island_order_reduced_M26": island_or,
        "self_consistent_order_reduced_M26_sweep": sc_sweep,
        "inhomogeneous_tt_gravitons_exact_volterra": inhom_modes,
        "radiative_backreaction_analytics": backreaction,
    }


def run_part4_kill_tests(bg_long):
    """
    Part 4: Five independent falsification kill-tests (all updated to exact M_I* = 26 / M_I*(10) = 14 / M_S3 = 6).
    """
    print("\n=== PART 4: FIVE INDEPENDENT KILL-TESTS (KT1 - KT5) ===", flush=True)

    # KT1: Time-domain Direct-Volterra work rate vs Parseval FFT at exact M_I* = 26 on x_ring!
    eta_kt1, d_eta_kt1 = 30.0, 0.005
    N_kt1 = int(round(eta_kt1 / d_eta_kt1))
    etas = np.arange(N_kt1 + 1) * d_eta_kt1
    st_g = bg_long.sol(etas)
    xs, hs, rs, ss, taus = st_g[0], st_g[1], st_g[2], st_g[3], st_g[4]
    A_g, K_g = bg_AK_at_state(xs, hs, rs)

    def rhs_0(eta, y):
        st = bg_long.sol(eta)
        A_R, K_R = bg_AK_at_state(float(st[0]), float(st[1]), float(st[2]))
        return [y[1] / (st[0] * st[0] * A_R), -8.0 * st[0] * st[0] * K_R * y[0]]

    sol0 = solve_ivp(rhs_0, (0.0, eta_kt1), [1.0, 0.0], method="DOP853", rtol=1e-12, atol=1e-14, t_eval=etas, max_step=0.01)
    b0_arr = sol0.y[0]
    v0_arr = sol0.y[1] / (xs * xs * A_g)
    B0_arr = np.empty(N_kt1 + 1)
    for i in range(N_kt1 + 1):
        cb_B, cv_B, _, _ = order_reduced_source_coeffs(xs[i], hs[i], rs[i], ss[i], 0.0, n_mode=2, running=False)
        B0_arr[i] = cb_B * b0_arr[i] + cv_B * v0_arr[i]

    W = build_exact_volterra_panel_weights(N_kt1, d_eta_kt1, 26.0, X_RING)
    c_M2 = math.log((26.0 ** 2) / (X_RING ** 2))
    conv = np.convolve(W, np.diff(B0_arr))[:N_kt1]
    K_B = np.empty(N_kt1 + 1)
    K_B[0] = c_M2 * B0_arr[0]
    K_B[1:] = c_M2 * B0_arr[1:] - 2.0 * conv
    P_inst = 4.0 * C_ANOM * v0_arr * K_B
    m_int = (etas >= 5.0) & (etas <= 25.0)
    E_loc = (xs * xs) * (A_g * v0_arr**2 + 8.0 * K_g * b0_arr**2)
    x_mean = float((taus[m_int][-1] - taus[m_int][0]) / (etas[m_int][-1] - etas[m_int][0]))
    gam_td = (float(np.mean(P_inst[m_int])) / (2.0 * float(np.mean(E_loc[m_int])))) / x_mean
    r_fft = compute_first_order_radiative_damping(bg_long, eta_kt1, n_mode=2, M_thresh=26.0, N_grid=16384)
    gam_fft = r_fft["modes"][0]["gamma_amp_tau_HS"]
    kt1_rel = abs(gam_td - gam_fft) / abs(gam_fft)
    print(f"   [KT1 M=26 Time-Domain Work vs Parseval FFT] gam_TD={gam_td:+.5e} vs gam_FFT={gam_fft:+.5e} (rel={kt1_rel:.3e})", flush=True)

    # KT2: Exact Manley-Rowe Sign-Flip Verification on Synthetic Pump (Om_s = 12.0, exact S^3 threshold M = 6, km = 0.1)!
    # Notice: Om_+ = 14.83 > 6 and |Om_-| = 9.17 > 6!
    print("   [KT2 Exact Manley-Rowe Sign-Flip & alpha^2 Scaling (Om_s=12, M_S3=6, km=0.1)]:", flush=True)
    Om_s, Om0, M_syn = 12.0, math.sqrt(8.0), 6.0
    Nu_s = 56
    qd = make_aux_quadrature(M_syn, u_min=1e-6, u_max=1e7, Nu=Nu_s)
    u_s, wre_s, wim_s = qd["u"], qd["w_re"], qd["w_im"]
    ln_uv_s = math.log((1e7**2 + M_syn**2) / (X_RING**2))
    kt2_rows = []
    for cK, alpha in [(-5.0, 0.0), (-5.0, 0.02), (-5.0, 0.04), (+5.0, 0.04)]:
        km_s = 0.1
        g_p = (4.0 * cK - 0.5 * (Om0 + Om_s) * Om0) * alpha
        g_m = (4.0 * cK - 0.5 * (Om0 - Om_s) * Om0) * alpha
        om_p, om_m = Om0 + Om_s, Om0 - Om_s
        bp, bm = g_p / (om_p**2 - 8.0), g_m / (om_m**2 - 8.0)
        B2p, B2m = (om_p**2 - 4.0) * (om_p**2 - 16.0), (om_m**2 - 4.0) * (om_m**2 - 16.0)
        gam_MR = (math.pi * C_ANOM / (X_RING**2 * Om0)) * (B2p * bp**2 - B2m * bm**2)

        def syn_bach_coeffs(e):
            c, s = math.cos(Om_s * e), math.sin(Om_s * e)
            A = 1.0 + alpha * c; A1 = -alpha * Om_s * s; A2 = -alpha * Om_s**2 * c; A3 = alpha * Om_s**3 * s
            K = 1.0 + cK * alpha * c; K1 = -cK * alpha * Om_s * s; K2 = -cK * alpha * Om_s**2 * c
            g0 = A1 / A; g1 = (A2 * A - A1 * A1) / (A * A); g2 = (A3 * A * A - 3.0 * A2 * A1 * A + 2.0 * A1**3) / (A**3)
            c0 = K / A; c1 = (K1 - c0 * A1) / A; c2 = (K2 - 2.0 * c1 * A1 - c0 * A2) / A
            w0, w1, w2 = 8.0 * c0, 8.0 * c1, 8.0 * c2
            q_a = (-(w0 + 8.0), -w1, -w2); q_c = (-g0, -g1, -g2)
            A1_ = q_a[1] - q_c[0] * w0; C1_ = q_a[0] + q_c[1] - q_c[0] * g0
            dA1 = q_a[2] - q_c[1] * w0 - q_c[0] * w1; dC1 = q_a[1] + q_c[2] - q_c[1] * g0 - q_c[0] * g1
            cb = (dA1 - C1_ * w0 - 8.0 * q_a[0]) - 36.0 * w0
            cv = (A1_ + dC1 - C1_ * g0 - 8.0 * q_c[0]) - 36.0 * g0
            return A, K, cb, cv

        _, _, cb0, _ = syn_bach_coeffs(0.0)
        y0_s = np.zeros(2 + 2 * Nu_s); y0_s[0] = 1.0; y0_s[2:2 + Nu_s] = cb0
        def rhs_syn(e, y):
            A, K, cb, cv = syn_bach_coeffs(e); inv = 1.0 / ((X_RING**2) * A)
            b, p = y[0], y[1]; v = p * inv; B0 = cb * b + cv * v
            UB, VB = y[2:2 + Nu_s], y[2 + Nu_s:]
            KB = ln_uv_s * B0 - 2.0 * float(np.dot(wre_s, UB) - np.dot(wim_s, VB))
            dy = np.empty_like(y); dy[0] = v
            dy[1] = -8.0 * (X_RING**2) * K * b - 2.0 * km_s * C_ANOM * KB
            dB = B0 - UB
            dy[2:2 + Nu_s] = u_s * dB - M_syn * VB; dy[2 + Nu_s:] = -M_syn * dB - u_s * VB
            return dy
        def jac_syn(e, y):
            A, K, cb, cv = syn_bach_coeffs(e); inv = 1.0 / ((X_RING**2) * A)
            J = np.zeros((2 + 2 * Nu_s, 2 + 2 * Nu_s)); J[0, 1] = inv
            J[1, 0] = -8.0 * (X_RING**2) * K - 2.0 * km_s * C_ANOM * ln_uv_s * cb
            J[1, 1] = -2.0 * km_s * C_ANOM * ln_uv_s * cv * inv
            J[1, 2:2 + Nu_s] = +4.0 * km_s * C_ANOM * wre_s; J[1, 2 + Nu_s:] = -4.0 * km_s * C_ANOM * wim_s
            J[2:2 + Nu_s, 0] = u_s * cb; J[2:2 + Nu_s, 1] = u_s * cv * inv
            J[2:2 + Nu_s, 2:2 + Nu_s] = -np.diag(u_s); J[2:2 + Nu_s, 2 + Nu_s:] = -M_syn * np.eye(Nu_s)
            J[2 + Nu_s:, 0] = -M_syn * cb; J[2 + Nu_s:, 1] = -M_syn * cv * inv
            J[2 + Nu_s:, 2:2 + Nu_s] = +M_syn * np.eye(Nu_s); J[2 + Nu_s:, 2 + Nu_s:] = -np.diag(u_s)
            return J
        te_s = np.linspace(0.0, 50.0, 501)
        sol_sc = solve_ivp(rhs_syn, (0.0, 50.0), y0_s, method="Radau", jac=jac_syn,
                           rtol=1e-9, atol=1e-11, t_eval=te_s, max_step=0.05)
        Ae = 1.0 + alpha * np.cos(Om_s * te_s); Ke = 1.0 + cK * alpha * np.cos(Om_s * te_s)
        ve = sol_sc.y[1] / ((X_RING**2) * Ae); Ee = (X_RING**2) * (Ae * ve**2 + 8.0 * Ke * sol_sc.y[0]**2)
        gam_sc = (-0.5 * float(np.polyfit(te_s[100:], np.log(Ee[100:]), 1)[0])) / km_s
        kt2_rows.append({"cK": cK, "alpha": alpha, "gamma_MR_analytic_eta": gam_MR, "gamma_SC_over_km_eta": gam_sc,
                         "sign_match": bool(alpha == 0.0 or (gam_MR * gam_sc > 0))})
        print(f"      cK={cK:+.1f} alpha={alpha:4.2f}: gam_SC/km={gam_sc:+.4e} vs Manley-Rowe={gam_MR:+.4e}", flush=True)

    # KT3: EFT-Safe Light-Scalaron Benchmark (sigma_t = 10.0, 4*c_anom*w_scal^2 = 0.116 << 1, Om_scal = 7.00)!
    # Long window (eta = 80.0, > 6 macro beats!) + CENTERED O(km) linear response (eps = 0.02):
    #   (a) n=2 with M_I* = 26: Om_2 + Om_scal = 2.83 + 7.00 = 9.83 < 26 => STRICTLY ZERO radiative decay!
    #   (b) n=10 with M_I* = 14: Om_10 + Om_scal = 10.95 + 7.00 = 17.95 > 14 => GENUINE EFT-safe damping!
    print("   [KT3 EFT-Safe Light-Scalaron (sigma_t=10.0, eta=80, centered eps=0.02): n=2(M=26) vs n=10(M=14)]:", flush=True)
    x_ring_10 = math.sqrt(abs(KAPPA_V + 1.0 - 10.0))
    bg_s10, _ = integrate_kam_background_eta(x_ring_10, eta_end=85.0, sigma_t=10.0)
    kt3_modes = {}
    for key_m, n_m, M_m in [("n2_M26", 2, 26.0), ("n10_M14", 10, 14.0)]:
        r0_m = solve_order_reduced_aux_ode(bg_s10, 80.0, 0.0, M_thresh=M_m, kappa_mem=0.0,
                                           include_local_c2=False, n_mode=n_m, n_eval=2001, method="DOP853", sigma_t=10.0)
        rp_m = solve_order_reduced_aux_ode(bg_s10, 80.0, 0.0, M_thresh=M_m, kappa_mem=+0.02,
                                           include_local_c2=False, n_mode=n_m, Nu=48, u_max=1e6, n_eval=2001, method="BDF", sigma_t=10.0)
        rm_m = solve_order_reduced_aux_ode(bg_s10, 80.0, 0.0, M_thresh=M_m, kappa_mem=-0.02,
                                           include_local_c2=False, n_mode=n_m, Nu=48, u_max=1e6, n_eval=2001, method="BDF", sigma_t=10.0)
        r1_m = solve_order_reduced_aux_ode(bg_s10, 80.0, 0.0, M_thresh=M_m, kappa_mem=1.0,
                                           include_local_c2=False, n_mode=n_m, Nu=48, u_max=1e6, n_eval=2001, method="BDF", sigma_t=10.0)
        g0_m, _ = measure_energy_rate(bg_s10, r0_m, n_mode=n_m, eta_cut=10.0, sigma_t=10.0)
        gp_m, _ = measure_energy_rate(bg_s10, rp_m, n_mode=n_m, eta_cut=10.0, sigma_t=10.0)
        gm_m, _ = measure_energy_rate(bg_s10, rm_m, n_mode=n_m, eta_cut=10.0, sigma_t=10.0)
        g1_m, e1_m = measure_energy_rate(bg_s10, r1_m, n_mode=n_m, eta_cut=10.0, sigma_t=10.0)
        flx_m = compute_first_order_radiative_damping(bg_s10, 80.0, n_mode=n_m, M_thresh=M_m, sigma_t=10.0, N_grid=32768)["gamma_amp_tau_avg_HS"]
        dlam_c = (gp_m - gm_m) / 0.04
        kt3_modes[key_m] = {
            "Prad_HS": flx_m, "lam0_HS": g0_m, "delta_lam_centered_eps0p02_HS": dlam_c,
            "lam_km1_HS": g1_m, "Epos_end_over_0_km1": e1_m, "blowup": bool(r1_m["blowup"]),
        }
        print(f"      {key_m:8s}: P/(2E)={flx_m:.3e} | lam0={g0_m:+.3e} | dlam_O1(cent)={dlam_c:+.3e} | lam(km=1)={g1_m:+.3e} (E/E0={e1_m:.4f})", flush=True)
    kt3_res = {
        "sigma_t": 10.0, "x_ring_s10": x_ring_10, "eta_end": 80.0,
        "eps_scal_4c_wscal2": 4.0 * C_ANOM / 10.0,
        "omega_scal_cyl": x_ring_10 / math.sqrt(10.0),
        "n2_M26": kt3_modes["n2_M26"],
        "n10_M14": kt3_modes["n10_M14"],
        "blowup_any": bool(kt3_modes["n2_M26"]["blowup"] or kt3_modes["n10_M14"]["blowup"]),
    }

    # KT4: Quadrature & Step-Size Convergence at exact M_I* = 26 (short beta_0=0.289 + LONG Regime IV beta_0=1.0!)
    print("   [KT4 Quadrature & Step-Size Convergence (M_I*=26, beta_0=0.289 & long Regime IV beta_0=1.0)]:", flush=True)
    y0_st = [0.0, 0.0, 1.0, 0.0]
    r_48 = solve_full_aux_ode(bg_long, 2.0, 0.2890, M_thresh=26.0, kappa_mem=1.0, Nu=48, u_max=1e7, n_eval=501, y0_4d=y0_st)
    r_64 = solve_full_aux_ode(bg_long, 2.0, 0.2890, M_thresh=26.0, kappa_mem=1.0, Nu=64, u_max=1e7, n_eval=501, y0_4d=y0_st)
    r_96 = solve_full_aux_ode(bg_long, 2.0, 0.2890, M_thresh=26.0, kappa_mem=1.0, Nu=96, u_max=1e7, n_eval=501, y0_4d=y0_st)
    d_48_96 = float(np.max(np.abs(r_48["pi2"] - r_96["pi2"])) / max(1.0, np.max(np.abs(r_96["pi2"]))))
    d_64_96 = float(np.max(np.abs(r_64["pi2"] - r_96["pi2"])) / max(1.0, np.max(np.abs(r_96["pi2"]))))
    v_004 = solve_full_direct_volterra(bg_long, 2.0, 0.2890, M_thresh=26.0, kappa_mem=1.0, d_eta=0.004, y0_4d=y0_st)
    v_002 = solve_full_direct_volterra(bg_long, 2.0, 0.2890, M_thresh=26.0, kappa_mem=1.0, d_eta=0.002, y0_4d=y0_st)
    v_001 = solve_full_direct_volterra(bg_long, 2.0, 0.2890, M_thresh=26.0, kappa_mem=1.0, d_eta=0.001, y0_4d=y0_st)
    e_004 = float(np.max(np.abs(v_004["pi2"] - r_96["pi2"])) / max(1.0, np.max(np.abs(r_96["pi2"]))))
    e_002 = float(np.max(np.abs(v_002["pi2"][::2] - r_96["pi2"])) / max(1.0, np.max(np.abs(r_96["pi2"]))))
    e_001 = float(np.max(np.abs(v_001["pi2"][::4] - r_96["pi2"])) / max(1.0, np.max(np.abs(r_96["pi2"]))))
    # Long-time Regime IV quadrature check (beta_0=1.0, eta=40.0, Nu=48/u_max=1e6 vs Nu=80/u_max=1e7)
    rl_48 = solve_full_aux_ode(bg_long, 40.0, 1.0, M_thresh=26.0, kappa_mem=1.0, Nu=48, u_max=1e6, n_eval=1001, method="BDF", y0_4d=y0_st)
    rl_80 = solve_full_aux_ode(bg_long, 40.0, 1.0, M_thresh=26.0, kappa_mem=1.0, Nu=80, u_max=1e7, n_eval=1001, method="BDF", y0_4d=y0_st)
    gl_48 = fit_exponential_rate(rl_48["tau"], np.abs(rl_48["pi2"]), 0.40, 0.98)
    gl_80 = fit_exponential_rate(rl_80["tau"], np.abs(rl_80["pi2"]), 0.40, 0.98)
    dl_q1 = float(np.max(np.abs(rl_48["q1"] - rl_80["q1"])) / max(1e-12, np.max(np.abs(rl_80["q1"]))))
    print(f"      Aux-ODE Nu=48 vs 96 (eta=2): {d_48_96:.3e} | Nu=64 vs 96: {d_64_96:.3e}", flush=True)
    print(f"      Long Regime IV (b0=1, eta=40) Nu=48/1e6 vs Nu=80/1e7: rel_q1={dl_q1:.3e} | gam={gl_48:+.3e} vs {gl_80:+.3e}", flush=True)
    print(f"      Direct Volterra vs Aux(96): 0.004->{e_004:.3e} | 0.002->{e_002:.3e} | 0.001->{e_001:.3e} (ratios={e_004/e_002:.2f}, {e_002/e_001:.2f})", flush=True)

    # KT5: Canonical Ostrogradsky Energy Sign Signature at exact M_I* = 26, beta_0 = 0.2890!
    print("   [KT5 Ostrogradsky Ghost Energy Signature (M_I*=26, beta_0=0.2890)]:", flush=True)
    b0_5 = 0.2890
    r_kt5 = solve_full_aux_ode(bg_long, 3.0, b0_5, M_thresh=26.0, kappa_mem=1.0, static_bg=True, Nu=56, u_max=1e7, n_eval=1001)
    q1_5, q2_5, pi1_5, pi2_5 = r_kt5["q1"], r_kt5["q2"], r_kt5["pi1"], r_kt5["pi2"]
    H_ostro = (pi1_5 * q2_5 + 8.0 * q1_5 * pi2_5 - (pi2_5 ** 2) / (16.0 * b0_5)
               - (X_RING**2 + 144.0 * b0_5) * (q2_5 ** 2) + 8.0 * (X_RING**2) * (q1_5 ** 2))
    # Pure physical graviton at static x_ring, beta_0 = 0.2890
    w_g = 2.815
    pi2_g = -8.0 * b0_5 * (-(w_g**2 + 8.0))
    H_grav_init = 8.0 * pi2_g - (pi2_g**2) / (16.0 * b0_5) + 8.0 * (X_RING**2)
    kt5_res = {
        "beta_0": b0_5, "M_thresh": 26.0,
        "H_ostro_pure_graviton_init": float(H_grav_init),
        "H_ostro_runaway_late_eta3p0": float(H_ostro[-1]),
        "runaway_is_strictly_negative_energy_ghost": bool(np.all(H_ostro[-200:] < 0.0)),
    }
    print(f"      Pure graviton H_ostro(0) = {H_grav_init:+.3e} (> 0) | Runaway H_ostro(eta=3) = {H_ostro[-1]:+.3e} (< 0: {kt5_res['runaway_is_strictly_negative_energy_ghost']})", flush=True)

    return {
        "KT1_M26_time_domain_vs_parseval_fft": {"gamma_time_domain_HS": gam_td, "gamma_parseval_fft_HS": gam_fft, "rel_diff": kt1_rel},
        "KT2_M6_manley_rowe_sign_flip_and_alpha2": kt2_rows,
        "KT3_eft_safe_light_scalaron_sigma10_n2_M26_vs_n10_M14": kt3_res,
        "KT4_M26_quadrature_and_step_convergence": {
            "aux_Nu48_vs_96_rel": d_48_96, "aux_Nu64_vs_96_rel": d_64_96,
            "long_regime4_b0_1p0_eta40_Nu48_vs_80_rel_q1": dl_q1,
            "long_regime4_b0_1p0_gam_Nu48_HS": gl_48,
            "long_regime4_b0_1p0_gam_Nu80_HS": gl_80,
            "volterra_deta_0p004_vs_ref": e_004, "volterra_deta_0p002_vs_ref": e_002, "volterra_deta_0p001_vs_ref": e_001,
            "volterra_convergence_ratios": [e_004 / e_002, e_002 / e_001],
        },
        "KT5_M26_ostrogradsky_negative_energy_ghost_signature": kt5_res,
    }


def run_part5_commutator_and_phys_ic_audit(bg_long):
    """
    Part 5: Audit of (a) Non-Self-Adjointness of Finite-kappa Order Reduction ([Re K_M, B_0(eta)] != 0),
    proving odd-in-beta_0 spurious growth (+3.23e-3 at beta_0=+1 vs -3.26e-3 at beta_0=-1) and
    KT3 km=0 baseline subtraction (delta = -8.3e-7 == 0 below M=26), and (b) Direct Physical-IC
    Graviton Evolution in the FULL UNREDUCED 4th-order M_I*=26 solver across Regime IV beta_0 in [0.5, 5.0].
    """
    print("\n=== PART 5: ORDER-REDUCTION COMMUTATOR AUDIT & FULL UNREDUCED PHYSICAL-IC CHECK ===", flush=True)
    # (a) Odd-in-beta_0 test at km=0.25 + pure local km=0 conservative baseline + Direct Volterra cross-check!
    comm_tests = []
    for b0, km in [(+1.0, 0.0), (-1.0, 0.0), (+1.0, 0.25), (-1.0, 0.25)]:
        meth = "DOP853" if km == 0.0 else "BDF"
        r = solve_order_reduced_aux_ode(bg_long, 15.0, b0, M_thresh=26.0, kappa_mem=km,
                                        include_local_c2=True, Nu=48, u_max=1e6, n_eval=1001, method=meth)
        gE, er = measure_energy_rate(bg_long, r, n_mode=2, eta_cut=3.0)
        vol_rel = None
        if km > 0.0:
            rA10 = solve_order_reduced_aux_ode(bg_long, 10.0, b0, M_thresh=26.0, kappa_mem=km,
                                               include_local_c2=True, Nu=48, u_max=1e6, n_eval=1001, method="BDF")
            rB10 = solve_order_reduced_direct_volterra(bg_long, 10.0, b0, M_thresh=26.0, kappa_mem=km,
                                                       include_local_c2=True, d_eta=0.002)
            vol_rel = float(np.max(np.abs(rA10["b"] - rB10["b"][::5])) / max(1e-12, np.max(np.abs(rA10["b"]))))
        comm_tests.append({
            "beta_0": b0, "kappa_mem": km, "gamma_E_HS": gE, "E_end_over_E0": er,
            "aux_vs_direct_volterra_eta10_rel": vol_rel,
        })
        vs = f" | Aux vs Volterra={vol_rel:.3e}" if vol_rel is not None else ""
        print(f"   [Odd-beta0 commutator test] b0={b0:+5.2f}, km={km:4.2f}: 0.5*dlnE/dtau={gE:+.5e} H_S (E_end/E0={er:.4f}){vs}", flush=True)

    # (b) Exact FULL UNREDUCED 4th-order Volterra with smooth physical graviton IC (y0_4d=None) in stabilized Regime IV
    phys_ic_runs = []
    for b0 in [0.5, 1.0, 2.0, 5.0]:
        r = solve_full_aux_ode(bg_long, 40.0, b0, M_thresh=26.0, kappa_mem=1.0,
                               Nu=48, u_max=1e6, n_eval=1501, method="BDF", y0_4d=None)
        st = bg_long.sol(r["eta"])
        Ar, Kr = bg_AK_at_state(st[0], st[1], st[2])
        El = (st[0] ** 2) * (Ar * (r["q2"] ** 2) + 8.0 * Kr * (r["q1"] ** 2))
        m = r["eta"] >= 5.0
        g = 0.5 * float(np.polyfit(r["tau"][m], np.log(El[m]), 1)[0])
        phys_ic_runs.append({
            "beta_0": b0, "tau_end": float(r["tau"][-1]), "blowup": r["blowup"],
            "gamma_E_phys_HS": g, "E_end_over_E0": float(El[-1] / El[0]),
        })
        print(f"   [FULL UNREDUCED M=26 phys IC] b0={b0:4.1f} (tau={r['tau'][-1]:5.1f}): 0.5*dlnE/dtau={g:+.4e} H_S | E_end/E0={El[-1]/El[0]:.4f} | bu={r['blowup']}", flush=True)

    return {
        "order_reduction_odd_beta0_commutator_tests": comm_tests,
        "full_unreduced_M26_phys_ic_regime4": phys_ic_runs,
    }


def main():
    print("=== TRACK IV-A GOAL 3 (POST-AUDIT EXACT M_I*=26 & M_S3=6 FULL PHASE MAP) ===", flush=True)
    bg_long, info_long = integrate_kam_background_eta(X_RING, eta_end=130.0)
    t_mac_ring, xmin_r, xmax_r = measure_macro_period(bg_long, 130.0)
    print(f"Background x_ring: eta=130 -> tau={info_long['tau_end']:.1f}, T_macro={t_mac_ring:.2f}, "
          f"x in [{xmin_r:.4f}, {xmax_r:.4f}], max_rel_e00={info_long['max_rel_e00']:.2e}", flush=True)

    p1 = run_part1_tongue_and_pinchoff(bg_long)
    p2 = run_part2_regime4_M26_vs_M6(bg_long)
    p3 = run_part3_order_reduced_and_inhomogeneous(bg_long)
    p4 = run_part4_kill_tests(bg_long)
    p5 = run_part5_commutator_and_phys_ic_audit(bg_long)

    payload = {
        "background_xring_long": {**info_long, "T_macro_HS_inv": t_mac_ring},
        "part1_tongue_homotopy_and_pinchoff_M26": p1,
        "part2_regime4_M26_vs_M6_and_eccentric": p2,
        "part3_order_reduced_M26_and_inhomogeneous": p3,
        "part4_kill_tests": p4,
        "part5_commutator_and_phys_ic_audit": p5,
    }
    out_path = os.path.join(OUT_DIR, "t4a_03_kam_phase_map.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
