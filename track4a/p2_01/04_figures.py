#!/usr/bin/env python3
"""
Stream p2_01_discrete_photon_pole_sum — Stage 4: Publication-Grade 4-Panel Diagnostic Figure.
Generates p2_01_discrete_pole_sum_diagnostic.png directly from 01_derive.json, 02_solve.json,
and 03_kill_tests.json (zero hardcoded data arrays).
"""

import json
from pathlib import Path
from importlib.machinery import SourceFileLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STREAM_DIR = Path(__file__).resolve().parent
FIG_PATH = STREAM_DIR / "p2_01_discrete_pole_sum_diagnostic.png"

m2 = SourceFileLoader("m2_fig", str(STREAM_DIR / "02_solve.py")).load_module()


def eval_K_istar_vec(oms, J_cut=60):
    spec = m2.build_discrete_pole_spec("Istar", J_cut=J_cut, include_tail=True, match_exact_K0=True)
    Mt = spec["M_tail"]
    K = np.log(np.abs(Mt * Mt - oms * oms) / (m2.X_RING ** 2)) + spec["delta_C_tail"]
    for Ep, wp in zip(spec["E_p"], spec["w_p"]):
        K = K - wp * (2.0 * Ep) / (Ep * Ep - oms * oms)
    return K


def main():
    with open(STREAM_DIR / "01_derive.json") as fh:
        d1 = json.load(fh)
    with open(STREAM_DIR / "02_solve.json") as fh:
        d2 = json.load(fh)
    with open(STREAM_DIR / "03_kill_tests.json") as fh:
        d3 = json.load(fh)

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 7.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.9), dpi=220)
    fig.suptitle(
        r"Exact Discrete $S^3/I^*$ Two-Photon Spectral Pole Sum & Chiral Horowitz Pinch-Off Shift",
        fontsize=12.2, fontweight="bold", y=0.98,
    )

    # ------------------------------------------------------------------
    # Panel A: Discrete S^3/I* Kernel vs Continuum-Log M=26 (0 <= Omega <= 52)
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    om_sub = np.linspace(0.0, 25.65, 600)
    om_mid = np.linspace(26.35, 41.65, 400)
    om_hi = np.linspace(42.35, 49.65, 300)
    for seg_i, seg in enumerate([om_sub, om_mid, om_hi]):
        lbl = r"Exact discrete $\mathcal{K}_{I^*}^{\mathrm{disc}}(\Omega)$" if seg_i == 0 else None
        ax.plot(seg, eval_K_istar_vec(seg, J_cut=60), color="#0B5394", lw=1.9, label=lbl)
    om_all = np.linspace(0.0, 52.0, 900)
    k_cont = np.log(np.maximum(1e-6, np.abs(676.0 - om_all ** 2)) / (m2.X_RING ** 2))
    ax.plot(om_all, k_cont, color="#C27BA0", lw=1.6, ls="--",
            label=r"Continuum-log $\ln(|26^2-\Omega^2|/\mu_{\mathrm{cyl}}^2)$")
    for Ep, txt, xoff, yval in [
        (26, r"$E_6=26$" + "\n" + r"($2W=18.46$)", -6.6, 8.2),
        (42, r"$E_{10}=42$" + "\n" + r"($2W=11.43$)", -7.5, 8.2),
        (50, r"$E_{12}=50$", -4.6, -12.8),
    ]:
        ax.axvline(Ep, color="#CC0000", lw=0.95, ls=":", alpha=0.85)
        ax.text(Ep + xoff, yval, txt, color="#990000", fontsize=7.3,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.82))
    ax.axvspan(0.0, 26.0, color="#CFE2F3", alpha=0.22, label=r"Exact chiral gap $|\Omega|<26$ ($\mathrm{Im}\,\Pi\equiv 0$)")
    ax.axvspan(38.60, 43.47, color="#FCE5CD", alpha=0.45, label=r"Modulated $\Omega_{\mathrm{scal}}(\eta)\in[38.6,43.5]$")
    ax.set_xlim(0, 52)
    ax.set_ylim(-14, 12)
    ax.set_xlabel(r"Conformal frequency $\Omega = a\omega$")
    ax.set_ylabel(r"Real kernel $\mathrm{Re}\,\mathcal{K}(\Omega;\,\mu_{\mathrm{cyl}})$")
    ax.set_title("A. Discrete Pole Repulsion vs. Continuum-Log & Chiral Gap")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92)

    # ------------------------------------------------------------------
    # Panel B: Static Pinch-Off Curves beta_0(Omega) & KT3 Cutoff Ladder
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    oms_b = np.linspace(4.3, 25.5, 600)
    B2 = (oms_b ** 2 - 4.0) * (oms_b ** 2 - 16.0)
    eh = (m2.X_RING ** 2) * (oms_b ** 2 - 8.0) / (4.0 * B2)
    b_cont_curve = eh - 0.5 * m2.C_ANOM * np.log((676.0 - oms_b ** 2) / (m2.X_RING ** 2))
    b_disc_curve = eh - 0.5 * m2.C_ANOM * eval_K_istar_vec(oms_b, J_cut=60)
    ps = d1["kt3_pinch_off_and_convergence"]["pinch_off_summary"]
    b_modB_curve = (
        eh - (2.0 * m2.C_ANOM * (724.0 / 3.0) * (oms_b ** 2)) / (4.0 * B2)
        - 0.5 * m2.C_ANOM * eval_K_istar_vec(oms_b, J_cut=60)
    )

    ax.plot(oms_b, b_disc_curve, color="#0B5394", lw=2.0,
            label=rf"Conformal Weyl Model A ($\beta_{{\mathrm{{pinch}}}}={ps['Istar_discrete_exact_ModelA']['beta_pinch']:.4f}$)")
    ax.plot(oms_b, b_modB_curve, color="#E69138", lw=1.6, ls="-.",
            label=rf"Partial Kubo Model B ($\Omega^2$ only, $\beta_{{\mathrm{{pinch}}}}={ps['Istar_discrete_ModelB_abs']['beta_pinch']:.4f}$)")
    ax.plot(oms_b, b_cont_curve, color="#C27BA0", lw=1.6, ls="--",
            label=rf"Continuum $M=26$ ($\beta_{{\mathrm{{pinch}}}}={ps['Istar_continuum_M26']['beta_pinch']:.4f}$)")
    ax.scatter([ps["Istar_discrete_exact_ModelA"]["Omega_pinch"]], [ps["Istar_discrete_exact_ModelA"]["beta_pinch"]],
               color="#0B5394", s=40, zorder=5)
    ax.scatter([ps["Istar_continuum_M26"]["Omega_pinch"]], [ps["Istar_continuum_M26"]["beta_pinch"]],
               color="#C27BA0", s=40, zorder=5)
    ax.set_xlim(5.0, 25.5)
    ax.set_ylim(0.0, 3.0)
    ax.set_xlabel(r"Real sub-threshold frequency $\Omega \in (4, 26)$")
    ax.set_ylabel(r"Pinch-off coupling $\beta_0(\Omega)$")
    ax.set_title(r"B. Static Pinch-Off Shift ($+78.2\%$ Model A) & Cutoff Convergence")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)

    # Inset: period-30 O(1/J^2) truncated vs O(1/J^4) K0-matched hybrid convergence
    ins = ax.inset_axes([0.33, 0.48, 0.54, 0.29])
    ladder = [r for r in d1["kt3_pinch_off_and_convergence"]["cutoff_convergence_ladder"] if "per" in r["label"]]
    js = [r["J_cut"] for r in ladder]
    errs_t = [r["rel_err_trunc_vs_exact"] for r in ladder]
    errs_hm = [r["rel_err_hybrid_K0matched_vs_exact"] for r in ladder]
    ins.loglog(js, errs_t, "o-", color="#0B5394", ms=4, lw=1.3, label=r"Truncated $O(J^{-2})$")
    ins.loglog(js, errs_hm, "s--", color="#38761D", ms=4, lw=1.2, label=r"$K(0)$-matched $O(J^{-4})$")
    ins.set_ylim(5e-9, 5e-3)
    ins.set_xticks([30, 60, 120, 240])
    ins.set_xticklabels(["30", "60", "120", "240"])
    ins.xaxis.set_minor_formatter(plt.NullFormatter())
    ins.set_xlabel(r"$J_{\mathrm{cut}}$ (period-30)", fontsize=7.4)
    ins.set_ylabel(r"$|\Delta\beta_{\mathrm{pinch}}|/\beta$", fontsize=7.4)
    ins.tick_params(labelsize=6.8)
    ins.legend(fontsize=6.5, loc="lower left")

    # ------------------------------------------------------------------
    # Panel C: KT4 Dynamical Unreduced Growth Rate Across Regime IV
    #          (Filled markers: gamma > 0; Open markers: gamma <= 0; zero silent clamping!)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    xr_disc = d3["xring_discrete_Istar_scan"]
    xr_cont = d3["xring_control_continuum_M26"]
    xr_s3 = d3["xring_control_discrete_S3"]
    ec_disc = d3["eccentric_xb15_discrete_Istar_scan"]

    def plot_gam_signed(rows, color, marker, ls, label):
        bs = np.array([r["beta_0"] for r in rows])
        gs_raw = np.array([r["gamma_HS"] for r in rows])
        gs_abs = np.abs(gs_raw)
        ax.semilogy(bs, gs_abs, color=color, ls=ls, lw=1.3, alpha=0.85, label=label)
        m_pos = gs_raw > 0.0
        m_neg = ~m_pos
        if np.any(m_pos):
            ax.semilogy(bs[m_pos], gs_abs[m_pos], marker=marker, color=color, ls="none", ms=4.6)
        if np.any(m_neg):
            ax.semilogy(bs[m_neg], gs_abs[m_neg], marker=marker, color=color, mfc="white", mew=1.2, ls="none", ms=4.8)

    plot_gam_signed(xr_disc, "#0B5394", "o", "-", r"$x_{\mathrm{ring}}$ Discrete $S^3/I^*$ ($\beta_{\mathrm{env}}=0.727$)")
    plot_gam_signed(xr_cont, "#C27BA0", "s", "--", r"$x_{\mathrm{ring}}$ Continuum $M=26$ ($\beta_{\mathrm{env}}=0.421$)")
    plot_gam_signed(ec_disc, "#E69138", "^", "-.", r"$x_b=15$ Discrete $S^3/I^*$ ($\beta_{\mathrm{env}}=1.127$)")
    plot_gam_signed(xr_s3, "#CC0000", "d", ":", r"$x_{\mathrm{ring}}$ Discrete Round $S^3$ ($\beta_{\mathrm{pinch}}=7.72$)")

    ax.axvline(d3["adiabatic_pinchoff_envelopes"]["x_ring_discrete_Istar"]["beta_pinch_inst_full_max"],
               color="#0B5394", lw=1.0, ls=":")
    ax.axvline(d3["adiabatic_pinchoff_envelopes"]["x_b15_discrete_Istar"]["beta_pinch_inst_full_max"],
               color="#E69138", lw=1.0, ls=":")
    ax.axhspan(1e-7, 6e-4, color="#D9EAD3", alpha=0.35, label="Quiescent bounded band (filled: +, open: $-$)")
    ax.set_xlim(0.35, 8.6)
    ax.set_ylim(1e-7, 0.8)
    ax.set_xlabel(r"Weyl coupling $\beta_0$")
    ax.set_ylabel(r"Dynamical envelope rate $|\gamma_{\mathrm{dyn}}|\;[H_S]$")
    ax.set_title(r"C. Dynamical Regime IV Phase Map ($x_{\mathrm{ring}}$ & $x_b=15$)")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)

    # ------------------------------------------------------------------
    # Panel D: Weyl-Anomaly UV Recovery & Un-Clamped Verification Summary
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    wins = d1["kt1_and_kt2_verification"]["kt2_sliding_window_weyl_anomaly"]
    emids = [w["E_mid"] for w in wins]
    w_errs = [w["rel_err_vs_9_over_31"] for w in wins]
    ax.loglog(emids, w_errs, "o-", color="#0B5394", lw=1.9, ms=5.5,
              label=r"Dissipative avg $|\bar{\rho}/(B_2/40\pi) - 1|$ ($\rho = 2\,\mathrm{Im}\,\chi_R$)")
    e_ref = np.array([emids[0], emids[-1]])
    ax.loglog(e_ref, w_errs[0] * (emids[0] / e_ref) ** 2, "--", color="#666666", lw=1.2, label=r"$O(E^{-2})$ reference slope")

    cc = d2["dual_solver_cross_checks"]
    lh = d2["long_horizon_eta35_bdf_vs_radau_audit"]
    max_lh_dg = max(v["abs_gamma_diff_HS"] for v in lh.values())
    summary_txt = (
        "Verification Summary (Un-Clamped)\n"
        f"• C_R(j) CG / Matrix err: {d1['triple_overlap_and_kt1_selection']['max_matrix_err']:.2e}\n"
        f"• Friedmann E_00 max rel (eta=40/45): {d2['background_info']['max_rel_e00']:.2e} / {d3['backgrounds']['x_ring']['max_rel_e00']:.2e}\n"
        f"• H_tot drift (DOP853 / BDF35): {d2['exact_hamiltonian_conservation']['H_tot_max_rel_drift']:.1e} / {d2['exact_hamiltonian_conservation']['H_tot_kt4_prod_bdf_eta35_max_rel_drift']:.1e}\n"
        f"• RG invariance (A / B): {d2['rg_scale_invariance']['solver_A_aux_ode_rel_diff']:.1e} / {d2['rg_scale_invariance']['solver_B_direct_volterra_rel_diff']:.1e}\n"
        f"• Long eta=35 BDF vs Radau |dg|: {max_lh_dg:.2e} H_S\n"
        f"• Aux-ODE vs Volterra (Rich): {cc['Istar_disc_beta0_1.0']['dyn_KAM_rel_diff_AuxODE_vs_Volterra_Richardson']:.2e}\n"
        f"• Volterra O(h^2) ratio: {cc['Istar_disc_beta0_1.0']['dyn_volterra_order2_ratio']:.2f}\n"
        f"• J_cut=30 vs 60 (b0=1.0): {d2['time_domain_Jcut_convergence']['beta0_1.0']['rel_diff_pi2_J30_vs_J60']:.2e}"
    )
    ax.text(
        0.04, 0.05, summary_txt, transform=ax.transAxes, fontsize=7.2,
        family="monospace", verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F8F9FA", edgecolor="#CCCCCC", alpha=0.95),
    )
    ax.set_xticks([200, 500, 1000, 2000, 5000])
    ax.set_xticklabels(["200", "500", "1000", "2000", "5000"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"Sliding-window center energy $E_{\mathrm{mid}}$ (period-30 width $\Delta E = 120$)")
    ax.set_ylabel(r"Relative deviation from $c_{\mathrm{anom}} = 9/31$")
    ax.set_title("D. Weyl-Anomaly UV Recovery & Verification Benchmarks")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)

    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    fig.savefig(FIG_PATH, dpi=220)
    plt.close(fig)
    print(f"Saved {FIG_PATH}", flush=True)


if __name__ == "__main__":
    main()
