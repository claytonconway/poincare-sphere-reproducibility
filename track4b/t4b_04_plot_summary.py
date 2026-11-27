"""T4B-04: Generate Audited Publication-Clean 6-Panel Diagnostic Summary Figure for Track IV-B
(Dynamical SM Hosotani Restoration, Real-Time rho_2 -> S_02 -> rho_0 Avalanche Cosmology,
and Nonlinear n=12 Saturation on S^3/I*).

All cold-audit figure refinements applied:
  - Panel A: Visual dashed separator distinguishing fixed M_W=2.5 sweep (E=50..1000) from high-E M_W=4.0 point (E=2500).
  - Panel C: Legend positioned cleanly away from the red Higgs curve; coherent rebound N_CS = +0.492 explicitly annotated.
  - Panel D: Explicit shaded UV-cascade zone (eta > 0.90, a <= 1.09 at N_max=36) marking where local Gauss leakage > 10%
             begins after the converged pre-bounce avalanche & gamma_c=1/4 bounce/crunch split.
  - Panel E: Plots strictly positive, non-clamped kinetic inhomogeneous share f_{kin, n>=12}(eta) (mean 74.1% post-sat),
             eliminating the max(0, 1 - E_hom/E_tot) clamping dip artifact.
  - Panel F: Dense 400-point N_max=20..36 convergence showing monotonic 912x drop of G_n20(a<=8) and G_rel(full)=6.84e-4 at N=36.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    d1 = json.load(open(os.path.join(OUT_DIR, "t4b_01_goldstone_covariant.json")))
    d3 = json.load(open(os.path.join(OUT_DIR, "t4b_03_dynamical_phenomena.json")))
    d3b = json.load(open(os.path.join(OUT_DIR, "t4b_03b_avalanche_bifurcations.json")))
    d5 = json.load(open(os.path.join(OUT_DIR, "t4b_05_audit_remediation.json")))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.94,
        "legend.fontsize": 7.8,
    })

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6), dpi=200)
    fig.patch.set_facecolor("white")

    # ------------------------------------------------------------------
    # Panel A: Goal 1 — SM Hosotani vs. Stiff-Higgs & Pure-YM Linear Amplification
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    picks = [(50.0, 2.5), (200.0, 2.5), (500.0, 2.5), (1000.0, 2.5), (2500.0, 4.0)]
    labels_a = ["E=50\n$M_W$=2.5", "E=200\n$M_W$=2.5", "E=500\n$M_W$=2.5", "E=1000\n$M_W$=2.5", "E=2500\n$M_W$=4.0"]
    x_idx = np.arange(len(picks))
    ym_n12, st_n12, sm_n12, sm_n4, sm_n20 = [], [], [], [], []
    for E_p, MW_p in picks:
        row = next(r for r in d1["dynamical_hosotani_survey"] if r["E_YM_init"] == E_p and r["M_W"] == MW_p)
        ym_n12.append(row["regimes"]["pure_YM"]["amps"]["n=12"]["max_all"])
        st_n12.append(row["regimes"]["stiff_20.0"]["amps"]["n=12"]["max_all"])
        sm_n12.append(row["regimes"]["SM_1.556"]["amps"]["n=12"]["max_all"])
        sm_n4.append(row["regimes"]["SM_1.556"]["amps"]["n=4"]["max_all"])
        sm_n20.append(row["regimes"]["SM_1.556"]["amps"]["n=20"]["max_all"])

    # Connect fixed M_W=2.5 points (0..3) with lines and plot high-E M_W=4.0 (4) separately across vertical divider!
    ax.axvline(3.5, color="#94A3B8", ls="--", lw=1.0, alpha=0.8)
    ax.plot(x_idx[:4], sm_n4[:4], "s--", color="#C0392B", lw=1.8, ms=6, label=r"$S^3$ $n=4$ (SM $\beta_H=1.556$)")
    ax.plot(x_idx[4], sm_n4[4], "s", color="#C0392B", ms=6.5)
    ax.plot(x_idx[:4], ym_n12[:4], "o:", color="#7F8C8D", lw=1.8, ms=5.5, label=r"$S^3/I^*$ $n=12$ (Pure YM)")
    ax.plot(x_idx[4], ym_n12[4], "o", color="#7F8C8D", ms=6.0)
    ax.plot(x_idx[:4], st_n12[:4], "^-", color="#D97706", lw=1.8, ms=6, label=r"$S^3/I^*$ $n=12$ (Stiff $\beta_H=20$)")
    ax.plot(x_idx[4], st_n12[4], "^", color="#D97706", ms=6.5)
    ax.plot(x_idx[:4], sm_n12[:4], "D-", color="#1D4ED8", lw=2.2, ms=6.5, label=r"$S^3/I^*$ $n=12$ (SM $\beta_H=1.556$)")
    ax.plot(x_idx[4], sm_n12[4], "D", color="#1D4ED8", ms=7.0)
    ax.plot(x_idx[:4], sm_n20[:4], "v-", color="#0F766E", lw=1.5, ms=5, label=r"$S^3/I^*$ $n=20$ (SM $\beta_H=1.556$)")
    ax.plot(x_idx[4], sm_n20[4], "v", color="#0F766E", ms=5.5)

    ax.set_yscale("log")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels_a, fontsize=8.0)
    ax.set_ylabel(r"Bogoliubov Bounce Amplification $\mathcal{A}_{\max}$")
    ax.set_title(r"A. Goldstone-Covariant Bounce Stability ($S^3$ vs. $S^3/I^*$)")
    ax.set_ylim(0.8, 3500)
    ax.grid(True, which="major", alpha=0.25)
    ax.annotate("436× quench\n(1789× → 4.1×)", xy=(3.96, sm_n12[-1]), xytext=(2.15, 22),
                arrowprops=dict(arrowstyle="->", color="#1D4ED8", lw=1.2),
                fontsize=8.0, color="#1D4ED8", fontweight="bold")
    ax.legend(loc="upper left")

    # ------------------------------------------------------------------
    # Panel B: Goal 3A — Nonlinear n=12 Energy Fraction (eps_12 = 0.1)
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    runs_3a = [r for r in d3["part_3a_nonlinear_hosotani_n12"]["runs"] if abs(r["eps_12"] - 0.1) < 1e-9]
    vals_e500 = [next(r["peak_inhom_E_frac"] * 100.0 for r in runs_3a if r["label"] == k)
                 for k in ["E500_pureYM", "E500_Stiff_MW2.5", "E500_SM_MW2.5"]]
    vals_e2500 = [next(r["peak_inhom_E_frac"] * 100.0 for r in runs_3a if r["label"] == k)
                  for k in ["E2500_pureYM", "E2500_Stiff_MW4.0", "E2500_SM_MW4.0"]]
    gx = np.arange(2)
    w_bar = 0.24
    b1 = ax.bar(gx - w_bar, [vals_e500[0], vals_e2500[0]], width=w_bar, color="#64748B", label="Pure YM ($M_W=0$)")
    b2 = ax.bar(gx, [vals_e500[1], vals_e2500[1]], width=w_bar, color="#D97706", label=r"Stiff Higgs ($\beta_H=20$)")
    b3 = ax.bar(gx + w_bar, [vals_e500[2], vals_e2500[2]], width=w_bar, color="#1D4ED8", label=r"SM Hosotani ($\beta_H=1.556$)")
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.45, f"{h:.2f}%", ha="center", va="bottom", fontsize=8.0)
    ax.set_xticks(gx)
    ax.set_xticklabels([r"$E_{\rm init}=500, M_W=2.5$", r"$E_{\rm init}=2500, M_W=4.0$"])
    ax.set_ylabel(r"Peak Inhomogeneous Energy Fraction $f_{E,n\geq 12}$ (%)")
    ax.set_ylim(0, 23.5)
    ax.set_title(r"B. Nonlinear $k=1$ Bounce Saturation ($\epsilon_{12}=0.1$)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left")

    # ------------------------------------------------------------------
    # Panel C: Goal 3B — Complete Expanding rho_2 -> S_02 -> rho_0 Avalanche
    # ------------------------------------------------------------------
    ax = axes[0, 2]
    tr_exp = d3b["complete_expanding_avalanche"]["trace_downsampled"]
    etas = np.array([r["eta"] for r in tr_exp])
    a_tr = np.array([r["a"] for r in tr_exp])
    ncs_tr = np.array([r["N_CS"] for r in tr_exp])
    phi_tr = np.array([r["phi_rms"] for r in tr_exp])
    a_rest = d3b["complete_expanding_avalanche"]["a_restore"]
    idx_rest = int(np.argmin(np.abs(a_tr - a_rest)))
    eta_rest = etas[idx_rest]

    l1, = ax.plot(etas, ncs_tr, color="#1D4ED8", lw=2.2, label=r"$N_{\rm CS}(\eta)$ ($\frac{49}{120} \to$ osc.)")
    ax.axhline(49.0 / 120.0, color="#1D4ED8", ls=":", lw=1.0, alpha=0.6)
    ax.axhline(0.24063, color="#64748B", ls="--", lw=1.0, alpha=0.7)
    ax.text(0.03, 0.255, r"$S_{02}$ ($0.2406$)", fontsize=8.0, color="#475569")
    ax.axvline(eta_rest, color="#0F766E", ls="-.", lw=1.3)
    ax.text(eta_rest + 0.018, 0.34, f"$a=a_{{\\rm rest}}$\n({a_rest:.2f})", fontsize=8.0, color="#0F766E")
    # Annotate coherent rebound peak +0.492
    idx_reb = int(np.argmax(ncs_tr))
    ax.annotate("Coherent rebound\n(+0.492)", xy=(etas[idx_reb], ncs_tr[idx_reb]), xytext=(0.48, 0.485),
                arrowprops=dict(arrowstyle="->", color="#1D4ED8", lw=1.1),
                fontsize=7.6, color="#1D4ED8")
    ax.set_xlabel(r"Conformal time $\eta$")
    ax.set_ylabel(r"Chern–Simons Number $N_{\rm CS}(\eta)$", color="#1D4ED8")
    ax.set_ylim(-0.03, 0.56)
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    l2, = ax2.plot(etas, phi_tr, color="#DC2626", lw=2.0, ls="--", label=r"Higgs $\phi_{\rm rms}(\eta)/v$")
    ax2.set_ylabel(r"Higgs Amplitude $\phi_{\rm rms}/v$", color="#DC2626")
    ax2.set_ylim(-0.05, 1.44)
    ax.set_title(r"C. Real-Time $\rho_2 \to S_{02} \to \rho_0$ Avalanche ($k=7$, SM)")
    ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="lower left", bbox_to_anchor=(0.01, 0.16))

    # ------------------------------------------------------------------
    # Panel D: Goal 3B — Pre-Bounce Contracting Avalanche gamma_c = 1/4 Bifurcation
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    tr_bnc = d3b["prebounce_contracting_gamma_c_bifurcation"]["trace_bounce"]
    tr_crn = d3b["prebounce_contracting_gamma_c_bifurcation"]["trace_crunch"]
    # Mark UV-cascade local-Gauss breakdown zone (eta > 0.90 at N_max=36; eta > 0.82 at N_max=24)
    eta_g_onset = d5["p0_2_contracting_N_max_gauss_audit"]["kap_5e6_bounce"][-1]["eta_onset_gauss_rel_0p10"]
    ax.axvspan(eta_g_onset, 4.6, color="#FEF3C7", alpha=0.35, zorder=0)
    ax.axvline(eta_g_onset, color="#B45309", ls=":", lw=1.1)
    ax.text(eta_g_onset + 0.06, 3.15, "UV cascade\n($\\mathcal{G}_{\\rm rel}>0.1$;\nmacro $\\gamma$ conv.)",
            fontsize=7.3, color="#92400E")

    ax.plot([r["eta"] for r in tr_bnc], [r["a"] for r in tr_bnc], color="#0F766E", lw=2.2,
            label=r"$\kappa_{\rm YM}=5\times 10^{-6}$ ($\gamma_{\rm tot}=0.1315 < \frac{1}{4}$: Bounce)")
    ax.plot([r["eta"] for r in tr_crn], [r["a"] for r in tr_crn], color="#C0392B", lw=2.2, ls="--",
            label=r"$\kappa_{\rm YM}=10^{-5}$ ($\gamma_{\rm tot}=0.2622 > \frac{1}{4}$: Crunch)")
    ax.axhline(1.0 / np.sqrt(2.0), color="#64748B", ls=":", lw=1.2)
    ax.text(2.85, 0.77, r"Barrier $a = 1/\sqrt{2}$", fontsize=8.0, color="#475569")
    ax.axhline(a_rest, color="#1D4ED8", ls="-.", lw=1.0, alpha=0.65)
    ax.text(2.98, 1.54, r"$a_{\rm restore}(\rho_2)=1.44$", fontsize=8.0, color="#1D4ED8")
    ax.set_xlabel(r"Conformal time $\eta$")
    ax.set_ylabel(r"Scale factor $a(\eta)$")
    ax.set_ylim(0.0, 4.7)
    ax.set_xlim(0.0, 4.6)
    ax.set_title(r"D. Contracting $\rho_2$ Avalanche Across $\gamma_c = 1/4$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    # ------------------------------------------------------------------
    # Panel E: Goal 3C — Non-Clamped Kinetic Inhomogeneous Share & Molien Cascade
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    lt = d3["part_3c_nonlinear_n12_saturation"]["longtime_fixed_a_E2500"]["trace"]
    et_lt = np.array([r["eta"] for r in lt])
    k0 = np.array([r["kin_frac_by_n"]["0"] * 100.0 for r in lt])
    k12 = np.array([r["kin_frac_by_n"]["12"] * 100.0 for r in lt])
    k20 = np.array([r["kin_frac_by_n"]["20"] * 100.0 for r in lt])
    k24 = np.array([r["kin_frac_by_n"]["24"] * 100.0 for r in lt])
    fkin_inhom = 100.0 - k0  # Strictly positive, ZERO clamping!
    mean_fkin = d5["p1_6_nonclamped_fkin_longtime"]["mean_fkin_inhom_post_sat"] * 100.0

    ax.plot(et_lt, fkin_inhom, color="#111827", lw=2.2, label=f"Total $f_{{\\rm kin, n\\geq 12}}$ (mean {mean_fkin:.1f}%)")
    ax.plot(et_lt, k12, color="#1D4ED8", lw=1.4, alpha=0.85, label=r"Kinetic $n=12$")
    ax.plot(et_lt, k20, color="#0F766E", lw=1.25, alpha=0.80, label=r"Kinetic $n=20$")
    ax.plot(et_lt, k24, color="#D97706", lw=1.25, alpha=0.80, label=r"Kinetic $n=24$")
    ax.set_xlabel(r"Conformal time $\eta$ (fixed $a$, $E_{\rm YM}=2500$, $\epsilon_{12}=10^{-3}$)")
    ax.set_ylabel("Non-Clamped Kinetic Share (%)")
    ax.set_ylim(-2, 118)
    ax.set_title(r"E. Pure-YM Nonlinear $n=12$ Saturation & Cascade ($N_{\max}=24$)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=2, fontsize=7.5)

    # ------------------------------------------------------------------
    # Panel F: Goal 3D — Dense N_max=20..36 Convergence of k=7 Gauss & Flatness
    # ------------------------------------------------------------------
    ax = axes[1, 2]
    conv_dense = d5["p1_4_p1_8_expanding_dense_N_max"]
    ns_c = [r["N_max"] for r in conv_dense]
    g20_pl = [r["max_gauss_n20_a_le_8"] for r in conv_dense]
    grel_f = [r["max_gauss_l2_rel_full"] for r in conv_dense]
    efl_c = [r["E_flat_init"] for r in conv_dense]
    hres_c = [r["max_H_res_rel"] for r in conv_dense]
    gn0_c = [r["max_gauss_n0"] for r in conv_dense]

    ax.semilogy(ns_c, g20_pl, "o-", color="#C0392B", lw=2.0, ms=6.5, label=r"$\max \|\mathcal{G}_{n\leq 20}\|_2$ ($a\leq 8$, 912× drop)")
    ax.semilogy(ns_c, grel_f, "v--", color="#D97706", lw=1.7, ms=5.5, label=r"$\max \mathcal{G}_{\rm rel}$ (full $a\leq 25$)")
    ax.semilogy(ns_c, efl_c, "s--", color="#1D4ED8", lw=1.7, ms=5.5, label=r"Initial $E_{\rm YM}(c_{\rho_2,\rm flat}^{(N_{\max})})$")
    ax.semilogy(ns_c, gn0_c, "^-", color="#0F766E", lw=1.6, ms=5.5, label=r"Global Gauss $|\mathcal{G}_{n=0}|$")
    ax.semilogy(ns_c, hres_c, "D:", color="#475569", lw=1.6, ms=5.0, label=r"Friedmann $|\mathcal{H}_{\rm res}|$")
    ax.set_xticks(ns_c)
    ax.set_xlabel(r"Spectral Harmonic Cutoff $N_{\max}$ ($k=7$)")
    ax.set_ylabel("Residual / Cutoff Energy (log scale)")
    ax.set_ylim(1e-16, 15)
    ax.set_title(r"F. Dense $k=7$ Convergence & Exact Conservation ($N_{\max}=20\text{–}36$)")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="center left", fontsize=7.2)

    fig.suptitle(
        r"Track IV-B: Dynamical Standard-Model Hosotani Restoration, Real-Time $\rho_2 \to S_{02} \to \rho_0$ Avalanche Cosmology, & Nonlinear $n=12$ Saturation on $S^3/I^*$",
        fontsize=12.0,
        fontweight="bold",
        y=0.985,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    out_png = os.path.join(OUT_DIR, "t4b_dynamical_hosotani_rho2_summary.png")
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
