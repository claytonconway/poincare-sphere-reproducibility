#!/usr/bin/env python3
"""
Stream p2_02_nonlinear_kam_backreaction (CLM-OPEN-02)
Stage 4a (Round 2): Publication-Quality 4-Panel Diagnostic Figure.

Generates p2_02_nonlinear_kam_backreaction_diagnostic.png:
  (a) Trans-unitarity sigma_t0 = 9/31: Nonlinear Mathieu n=18,22 backreaction (full MR scalaron share)
  (b) P0-1 Chiral Self-Quenching Bifurcation: Radius-dependent n=2 catalyzed depletion
      (b0=0.05 g2=1 & b0=0.02 g2=5 self-quenching in [12.49, 15.53] vs b0=0.10 punch-through crash)
  (c) Sub-Unitarity EFT Window [4, 12] & I* Spectral Desert vs Matched S^3-Forbidden Modes
      (exact S^3 degeneracies {24,42,64,90}, equal 0.002/mode energy, plus n=10 1:1 tongue at 4.20)
  (d) Unified 5-Mode True Un-Floored |E_00|_rel(tau) Conservation (P1-1 fix)
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT_DIR)
from importlib import import_module

s2 = import_module("02_solve")


def make_figure():
    rate_tables = s2.load_rate_tables()
    n2_interp = s2.build_n2_x_dependent_interpolator(rate_tables)
    with open(os.path.join(OUT_DIR, "p2_02_02_dynamics.json")) as f:
        d2 = json.load(f)

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.2,
        "axes.labelsize": 9.5,
        "legend.fontsize": 7.8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), dpi=180)
    (axA, axB), (axC, axD) = axes

    # ------------------------------------------------------------------
    # Panel (a): Exp A Mathieu n=18,22 backreaction at sigma_t0 = 9/31
    # ------------------------------------------------------------------
    tbl0 = rate_tables["five_mode_tables_by_sigma"]["sigma0_9over31"]
    modes_m = [
        {"n": 18, "g": 17.0, "kn2": 360.0,
         "gamma_rad_tot": tbl0["18"]["gamma_rad_tot"], "delta_lam_1": tbl0["18"]["delta_lam_1"]},
        {"n": 22, "g": 46.0, "kn2": 528.0,
         "gamma_rad_tot": tbl0["22"]["gamma_rad_tot"], "delta_lam_1": tbl0["22"]["delta_lam_1"]},
    ]
    rA_diss = s2.integrate_coupled(s2.X_RING_0, s2.SIGMA_0, lam=0.0, modes=modes_m, b0_list=[1e-4, 1e-4],
                                   tau_end=150.0, max_step=0.10, n_eval=2001)
    modes_m0 = [
        {"n": 18, "g": 17.0, "kn2": 360.0, "gamma_rad_tot": 0.0, "delta_lam_1": 0.0},
        {"n": 22, "g": 46.0, "kn2": 528.0, "gamma_rad_tot": 0.0, "delta_lam_1": 0.0},
    ]
    rA_nod = s2.integrate_coupled(s2.X_RING_0, s2.SIGMA_0, lam=0.0, modes=modes_m0, b0_list=[1e-4, 1e-4],
                                  tau_end=150.0, max_step=0.10, n_eval=2001)

    tA = rA_diss["_ts"]
    mA = tA <= rA_diss["tau_safe"]
    axA.plot(tA[mA], rA_diss["_kap_scal"][mA], color="#1f4e79", lw=1.6, label=r"Scalaron $\kappa_{\mathrm{scal}}(\tau)$")
    axA.plot(tA[mA], rA_diss["_kap_tt"][mA], color="#c0392b", lw=1.8, label=r"Mathieu TT $\kappa_{\mathrm{TT}}(\tau)$ ($n=18,22$)")
    axA.plot(tA[mA], rA_diss["_kap_rad"][mA], color="#d97706", lw=1.6, ls="--", label=r"2-photon $\kappa_{\mathrm{rad}}(\tau)$ (MR share)")
    axA.plot(tA[mA], rA_diss["_kap_eff"][mA], color="#111827", lw=1.8, label=r"Barrier $\kappa_{\mathrm{eff,tot}}(\tau)$")
    axA.axhline(0.0, color="#6b7280", lw=0.9, ls=":")
    axA.axvline(rA_nod["tau_end"], color="#7f1d1d", lw=1.2, ls="-.", label=fr"No-diss crash ($\tau={rA_nod['tau_end']:.1f}$)")
    axA.axvline(rA_diss["tau_end"], color="#dc2626", lw=1.4, ls=":", label=fr"+1-loop MR crash ($\tau={rA_diss['tau_end']:.1f}$)")
    axA.set_ylim(-600, 1250)
    axA.set_xlim(0, 115)
    axA.set_xlabel(r"Dimensionless Starobinsky time $\tau = H_S t$")
    axA.set_ylabel(r"Conformal energy budget $\kappa = x^4 \rho / 3$")
    axA.set_title(r"(a) Trans-Unitarity $\tilde\sigma_0=9/31$: $n=18,22$ Mathieu Backreaction")
    axA.legend(loc="upper left", frameon=True, framealpha=0.92)
    axA.grid(alpha=0.22)

    # ------------------------------------------------------------------
    # Panel (b): P0-1 Chiral Self-Quenching vs 2nd-Harmonic Punch-Through
    # ------------------------------------------------------------------
    m2_g1 = [{"n": 2, "g": 1.0, "kn2": 8.0, "x_interp": n2_interp}]
    m2_g5 = [{"n": 2, "g": 5.0, "kn2": 8.0, "x_interp": n2_interp}]
    runs_B = [
        (m2_g1, 0.05, "#0f766e", r"$b_0=0.05\,(g_2=1)$: self-quenched ($|b|\to 0.005$)"),
        (m2_g5, 0.02, "#1d4ed8", r"$b_0=0.02\,(g_2=5)$: self-quenched ($|b|\to 0.001$)"),
        (m2_g1, 0.10, "#b91c1c", r"$b_0=0.10\,(g_2=1)$: punch-through crash ($\tau=3475$)"),
    ]
    axB.axhspan(12.485, 15.533, color="#d1fae5", alpha=0.55, label=r"Chiral damping window $x\in[12.49, 15.53]$")
    axB.axhspan(0.0, 7.0, color="#fecaca", alpha=0.38, label=r"P05 stochastic / crash zone ($x \lesssim 7$)")
    for modes_b, b0, col, lbl in runs_B:
        rB = s2.integrate_coupled(s2.X_RING_0, s2.SIGMA_0, lam=0.0, modes=modes_b, b0_list=[b0],
                                  tau_end=10000.0, max_step=0.25, n_eval=2501)
        ts_b, xs_b, kr_b = rB["_ts"], rB["_ys"][0], rB["_kap_rad"]
        dt_b = ts_b[1] - ts_b[0]
        w_b = max(3, int(round(40.0 / dt_b)))
        x_sm = np.convolve(xs_b, np.ones(w_b) / w_b, mode="valid")
        t_sm = ts_b[w_b // 2 : w_b // 2 + len(x_sm)]
        axB.plot(t_sm, x_sm, color=col, lw=1.7, label=lbl)
        if b0 == 0.05:
            x_pred = np.sqrt(np.maximum(0.0, abs(1.0 - s2.SIGMA_0 + s2.KAPPA_V) - kr_b))
            axB.plot(ts_b[::10], x_pred[::10], color="#111827", lw=1.2, ls="--",
                     label=r"Analytic $\sqrt{|\kappa_{\mathrm{eff}}(0)| - \kappa_{\mathrm{rad}}(\tau)}$ ($b_0=0.05$)")
    axB.set_xlim(0, 10000)
    axB.set_ylim(0, 26)
    axB.set_xlabel(r"Dimensionless Starobinsky time $\tau = H_S t$")
    axB.set_ylabel(r"Smoothed scale factor $\bar{x}(\tau)$")
    axB.set_title(r"(b) $n=2$ Radius-Dependent Catalyzed Depletion: Chiral Self-Quenching ($\Lambda=0$)")
    axB.legend(loc="lower right", frameon=True, framealpha=0.92)
    axB.grid(alpha=0.22)

    # ------------------------------------------------------------------
    # Panel (c): I* Spectral Desert vs Matched S^3-Forbidden Modes across sigma_t
    # ------------------------------------------------------------------
    scan = d2["exp_C_sub_unitarity_eft_and_spectral_desert"]["sigma_t_spectral_desert_scan"]
    sigs = [r["sigma_t"] for r in scan]
    gI = [r["Istar_dom_amp_gain"] for r in scan]
    gF = [r["S3_forb_dom_amp_gain"] for r in scan]
    axC.axvspan(4.0, 12.0, color="#d1fae5", alpha=0.55, label=r"Sub-unitarity EFT window $\tilde\sigma\in[4,12]$")
    axC.plot(sigs, gI, "o-", color="#1d4ed8", lw=1.8, ms=5.0, label=r"$S^3/I^*$ allowed $\{2,10,14,18,22\}$")
    axC.plot(sigs, gF, "s--", color="#dc2626", lw=1.5, ms=5.0, label=r"Matched $S^3$-forbidden $\{3,4,5,6\}$ (exact $g_n$)")
    axC.set_xscale("log")
    axC.set_yscale("log")
    axC.set_xlim(0.24, 21.0)
    axC.set_ylim(0.7, 3500.0)
    axC.annotate(r"$n=18$", xy=(0.29, 1196), xytext=(0.33, 1700), fontsize=8, color="#1d4ed8")
    axC.annotate(r"$n=14$", xy=(0.50, 973), xytext=(0.54, 1300), fontsize=8, color="#1d4ed8")
    axC.annotate(r"$n=10$", xy=(1.00, 772), xytext=(1.08, 950), fontsize=8, color="#1d4ed8")
    axC.annotate(r"$n=10$ (1:1)", xy=(4.20, 70.7), xytext=(1.8, 120), fontsize=7.6, color="#1d4ed8",
                 arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=0.8))
    axC.annotate(r"$n=4$ ($S^3$)", xy=(5.05, 1005), xytext=(2.8, 1400), fontsize=8, color="#dc2626",
                 arrowprops=dict(arrowstyle="->", color="#dc2626", lw=0.8))
    axC.annotate(r"$n=3$ ($S^3$)", xy=(8.60, 269), xytext=(6.8, 600), fontsize=8, color="#dc2626",
                 arrowprops=dict(arrowstyle="->", color="#dc2626", lw=0.8))
    axC.annotate(r"$n=2$ wall", xy=(16.5, 635), xytext=(10.5, 1500), fontsize=8, color="#1d4ed8",
                 arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=0.8))
    axC.set_xlabel(r"Renormalized scalaron coupling $\tilde\sigma = H_S^2 / M^2$")
    axC.set_ylabel(r"Peak graviton amplification $\max|b_n|/b_n(0)$")
    axC.set_title(r"(c) $I^*$ Spectral Desert vs. Matched $S^3$-Forbidden Controls ($0.002$/mode)")
    axC.legend(loc="lower center", frameon=True, framealpha=0.92)
    axC.grid(alpha=0.22, which="both")

    # ------------------------------------------------------------------
    # Panel (d): Unified 5-Mode True Un-Floored |E_00|_rel Conservation
    # ------------------------------------------------------------------
    tbls = rate_tables["five_mode_tables_by_sigma"]
    rD4 = s2.integrate_coupled(s2.x_ring_for_sigma(4.0), 4.0, lam=0.0,
                               modes=s2.build_five_modes(tbls["sigma_4p0"], g2=1.0),
                               b0_list=[0.05, 1e-3, 1e-3, 1e-3, 1e-3], tau_end=10000.0, max_step=0.25, n_eval=2001)
    rD10_02 = s2.integrate_coupled(s2.x_ring_for_sigma(10.0), 10.0, lam=0.0,
                                   modes=s2.build_five_modes(tbls["sigma_10p0"], g2=1.0),
                                   b0_list=[0.02, 1e-3, 1e-3, 1e-3, 1e-3], tau_end=10000.0, max_step=0.25, n_eval=2001)
    rD0 = s2.integrate_coupled(s2.X_RING_0, s2.SIGMA_0, lam=0.0,
                               modes=s2.build_five_modes(tbls["sigma0_9over31"], g2=1.0, n2_x_interp=n2_interp),
                               b0_list=[0.05, 1e-4, 1e-4, 1e-4, 1e-4], tau_end=1000.0, max_step=0.15, n_eval=1001)

    axD.semilogy(rD4["_ts"], np.maximum(1e-16, rD4["_rel_e00"]), color="#0f766e", lw=1.4,
                 label=fr"$\tilde\sigma=4.0$ 5-mode EFT (survived, true $\max={rD4['max_rel_E00']:.1e}$)")
    axD.semilogy(rD10_02["_ts"], np.maximum(1e-16, rD10_02["_rel_e00"]), color="#1d4ed8", lw=1.4, alpha=0.85,
                 label=fr"$\tilde\sigma=10.0$ 5-mode ($b_2=0.02$, true $\max={rD10_02['max_rel_E00']:.1e}$)")
    axD.semilogy(rD0["_ts"], np.maximum(1e-16, rD0["_rel_e00"]), color="#dc2626", lw=1.7,
                 label=fr"$\tilde\sigma_0=9/31$ 5-mode (blowup $\tau={rD0['tau_end']:.1f}$, $\max={rD0['max_rel_E00']:.1e}$)")
    axD.axhline(1e-6, color="#6b7280", lw=1.0, ls="--", label="KT1 true-relative tolerance ($10^{-6}$)")
    axD.set_xlim(0, 10000)
    axD.set_ylim(1e-16, 1e-5)
    axD.set_xlabel(r"Dimensionless Starobinsky time $\tau = H_S t$")
    axD.set_ylabel(r"True un-floored Friedmann residual $|E_{00}|_{\mathrm{rel}}(\tau)$")
    axD.set_title(r"(d) Unified 5-Mode Un-Clamped, Un-Floored Constraint Conservation ($\Lambda=0$)")
    axD.legend(loc="lower right", frameon=True, framealpha=0.92)
    axD.grid(alpha=0.22, which="both")

    plt.tight_layout(pad=1.6)
    fig_path = os.path.join(OUT_DIR, "p2_02_nonlinear_kam_backreaction_diagnostic.png")
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)
    print(f"Saved {fig_path}", flush=True)


if __name__ == "__main__":
    make_figure()
