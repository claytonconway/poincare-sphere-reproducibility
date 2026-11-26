"""Generate publication-grade 6-panel diagnostic figure for Open Problem 3 (Galois flat connection rho_2 on S^3/I*)."""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)


def load_json(name: str) -> dict:
    with open(os.path.join(OUT_DIR, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    g1 = load_json("goal1_rho2_verification.json")
    g2_t1 = load_json("goal2_tier1_k7.json")
    g2_t2 = load_json("goal2_tier2_full3d.json")
    g2_conn = load_json("goal2_connectivity_and_anatomy.json")
    g3 = load_json("goal3_higgs_lifting.json")
    g3_pd = load_json("goal3_coupled_phase_diagram.json")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "figure.dpi": 180,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.8), constrained_layout=True)
    fig.suptitle(
        r"Open Problem 3: Galois-Conjugate Flat Connection $\rho_2$ ($\mathrm{CS}=49/120$), Sphaleron Ridge $S_{02}$, & Coupled $SU(2)_L$–Higgs Phase Diagram on $S^3/I^*$",
        fontsize=12.5,
        fontweight="bold",
    )

    # -------------------------------------------------------------------------
    # Panel A: Exponential Spectral Convergence of A_{rho_2} = u_7^{-1} du_7
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    conv = g1["A_rho2_truncation_convergence"]
    ns = [r["N_max"] for r in conv]
    eyms = [r["E_YM_S3_Istar"] for r in conv]
    rmsb = [r["rms_B"] for r in conv]
    ax.semilogy(ns, eyms, "o-", color="#1b4f72", lw=2.0, ms=6, label=r"$E_{\mathrm{YM}}^{S^3/I^*}(A_{\rho_2}^{(N_{\max})})$")
    ax.semilogy(ns, rmsb, "s--", color="#c0392b", lw=1.7, ms=5.5, label=r"$\mathrm{rms}\,|B_A|$")
    ax.axhline(g1["flatness_and_cs_2D_hopf"]["max_B"], color="#27ae60", ls=":", lw=1.5, label=r"Exact $u_7^{-1}du_7$ ($\max|B|=1.15\times 10^{-13}$)")
    ax.set_title(r"(A) Spectral Convergence of $A_{\rho_2} = u_7^{-1}du_7$")
    ax.set_xlabel(r"Molien Harmonic Cutoff $N_{\max}$")
    ax.set_ylabel("Flatness Residual (log scale)")
    ax.set_xticks(ns)
    ax.legend(fontsize=8.0, loc="lower left")
    ax.text(
        0.97, 0.94,
        r"$N_{\mathrm{CS}} = \frac{49}{120}$" + "\n" + r"err $= 6.1\times 10^{-16}$" + "\n9/9 $I^*$ classes verified",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#bdc3c7"),
    )

    # -------------------------------------------------------------------------
    # Panel B: Minimum-Energy Path (MEP) & Three-Vacuum Morse Topology
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    t1_36 = g2_t1["tier1_k7_convergence"][-1]
    ncs_mep = t1_36["neb_N_CS"]
    e_mep = t1_36["neb_energies"]
    # Also plot naive straight-line path s A_{rho_2}: E(s) = 16 s^2 (1-s)^2 E_mid, N_CS(s) = (3s^2 - 2s^3)*49/120
    s_lin = np.linspace(0.0, 1.0, 101)
    ncs_lin = (3.0 * s_lin ** 2 - 2.0 * s_lin ** 3) * (49.0 / 120.0)
    e_lin = 16.0 * (s_lin ** 2) * ((1.0 - s_lin) ** 2) * g1["naive_linear_barriers"]["line_02_midpoint_E_S3_Istar"]

    ax.plot(ncs_lin, e_lin, ":", color="#7f8c8d", lw=1.6, label=r"Naive linear $s A_{\rho_2}$ (peak $61.56$)")
    ax.plot(ncs_mep, e_mep, "o-", color="#1f618d", lw=2.2, ms=4.5, label=r"CI-NEB MEP ($N_{\max}=36$, $S_{02}=31.4934$)")
    # Mark rho_0, rho_1, rho_2, S_01, S_02
    ax.scatter([0.0, 1.0 / 120.0, 49.0 / 120.0], [0.0, 0.0, 0.0], color="#27ae60", s=55, zorder=5, label=r"Flat vacua $\rho_0, \rho_1, \rho_2$")
    ax.scatter([t1_36["polished_N_CS"]], [t1_36["polished_E_YM_S3_Istar"]], color="#c0392b", marker="*", s=140, zorder=6, label=r"Morse-1 Saddle $S_{02}$ ($127.64\times S_{01}$)")
    ax.annotate(r"$\rho_0, \rho_1$ ($S_{01}=0.247$)", xy=(1.0 / 120.0, 0.25), xytext=(0.01, 27.0),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.0), fontsize=8.0)
    ax.annotate(r"$\rho_2$ ($49/120$)", xy=(49.0 / 120.0, 0.0), xytext=(0.32, 11.0),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.0), fontsize=8.0)
    ax.set_title(r"(B) Pure-YM Sphaleron Ridge $\rho_1 \leftrightarrow \rho_0 \leftrightarrow \rho_2$")
    ax.set_xlabel(r"Chern–Simons Functional $N_{\mathrm{CS}}^{S^3/I^*}[A]$")
    ax.set_ylabel(r"$E_{\mathrm{YM}}^{S^3/I^*}$  [$ (g_{\mathrm{YM}}^2 a)^{-1} $]")
    ax.set_xlim(-0.02, 0.44)
    ax.set_ylim(-2.0, 66.0)
    ax.legend(fontsize=7.8, loc="upper left")

    # -------------------------------------------------------------------------
    # Panel C: Convergence of S_02 Barrier & Stiff-Higgs Saddle-Node mu_{W,sn}
    # -------------------------------------------------------------------------
    ax = axes[0, 2]
    n_t1 = [r["N_max"] for r in g2_t1["tier1_k7_convergence"]]
    e_s02_err = [abs(r["polished_E_YM_S3_Istar"] - g2_t1["tier1_k7_convergence"][-1]["polished_E_YM_S3_Istar"]) + 1e-12 for r in g2_t1["tier1_k7_convergence"][:-1]]
    n_b1 = [r["N_max"] for r in g3["part_b1_stiff_k7_convergence"]]
    mu_sn_ref = g3["part_b1_stiff_k7_convergence"][-1]["mu_W_sn_mid"]
    mu_sn_err = [abs(r["mu_W_sn_mid"] - mu_sn_ref) + 1e-12 for r in g3["part_b1_stiff_k7_convergence"][:-1]]

    ax.semilogy(n_t1[:-1], e_s02_err, "o-", color="#1b4f72", lw=2.0, ms=6, label=r"$|E(S_{02}^{(N)}) - E(S_{02}^{(36)})|$")
    ax.semilogy(n_b1[:-1], mu_sn_err, "s--", color="#d35400", lw=2.0, ms=6, label=r"$|\mu_{W,\mathrm{sn}}^{(N)} - \mu_{W,\mathrm{sn}}^{(36)}|$")
    ax.set_title(r"(C) Truncation Convergence ($N_{\max} = 12 \to 36$)")
    ax.set_xlabel(r"Harmonic Cutoff $N_{\max}$")
    ax.set_ylabel("Absolute Deviation from $N_{\max}=36$")
    ax.set_xticks(n_t1[:-1])
    ax.legend(fontsize=8.2, loc="upper right")
    ax.text(
        0.04, 0.08,
        r"$E(S_{02}) = 31.493447$" + "\n"
        + r"$\mu_{W,\mathrm{sn}}(\rho_2) = 1.996199$" + "\n"
        + r"Full-3D $N_{12}$: $1.97341$ (exact match)",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#bdc3c7"),
    )

    # -------------------------------------------------------------------------
    # Panel D: Stiff-Higgs Energy Lifting & Full-3D Lowest Physical Eigenvalue
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    sw36 = g3["part_b1_stiff_k7_convergence"][-1]["sweep"]
    mus_36 = [0.0] + [r["mu_W"] for r in sw36 if r["ok"]]
    etot_36 = [0.0] + [r["E_tot"] for r in sw36 if r["ok"]]
    # Append exact critical point right below sn
    sn_b = g3["part_b1_stiff_k7_convergence"][-1]["at_sn_below"]
    mus_36.append(g3["part_b1_stiff_k7_convergence"][-1]["mu_W_sn_lower"])
    etot_36.append(sn_b["E_tot"])

    # Exact rho_1 stiff energy up to mu = 1/sqrt(2): E_1 = (pi^2/20) * [0.5*(f^2-1)^2 + 0.5*mu^2*(1+f)^2]
    mu_r1 = np.linspace(0.0, 1.0 / math.sqrt(2.0), 100)
    f_r1 = 0.5 * (1.0 + np.sqrt(np.maximum(0.0, 1.0 - 2.0 * mu_r1 ** 2)))
    e_r1 = (math.pi ** 2 / 20.0) * (0.5 * (f_r1 ** 2 - 1.0) ** 2 + 0.5 * (mu_r1 ** 2) * ((1.0 + f_r1) ** 2))

    ax.plot(mus_36, etot_36, "-", color="#1f618d", lw=2.3, label=r"$\rho_2(\mu_W)$ stiff branch ($N_{\max}=36$)")
    ax.plot(mu_r1, e_r1, "-", color="#27ae60", lw=2.2, label=r"$\rho_1(\mu_W)$ stiff branch (exact)")
    ax.scatter([1.0 / math.sqrt(2.0)], [e_r1[-1]], color="#27ae60", s=60, zorder=5)
    ax.scatter([mus_36[-1]], [etot_36[-1]], color="#c0392b", marker="X", s=85, zorder=6, label=r"$\rho_2$ saddle-node ($\mu_W=1.9962$)")
    ax.axvline(1.0 / math.sqrt(2.0), color="#27ae60", ls=":", lw=1.2)
    ax.axvline(mus_36[-1], color="#c0392b", ls=":", lw=1.2)
    ax.set_title(r"(D) Stiff-Higgs Lifting & Saddle-Node ($\rho_2$ vs. $\rho_1$)")
    ax.set_xlabel(r"Dimensionless Gauge Mass $\mu_W = a M_W$")
    ax.set_ylabel(r"Metastable Vacuum Energy $\mathcal{E}$ [$ (g_{\mathrm{YM}}^2 a)^{-1} $]")
    ax.set_xlim(0.0, 2.2)
    ax.legend(fontsize=8.0, loc="upper left")
    ax.text(
        0.58, 0.14,
        r"$\mu_W \to 0^+$ slope:" + "\n" + r"$\rho_2 / \rho_1 = 21.085\times$" + "\n" + r"$\mu_{W,\mathrm{sn}}(\rho_2)/\mu_{W,\mathrm{sn}}(\rho_1) = 2.823\times$",
        transform=ax.transAxes, fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#bdc3c7"),
    )

    # -------------------------------------------------------------------------
    # Panel E: Exact Coupled (A, f) 2D Phase Diagram in (beta_H, mu_W) Plane!
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    betas_c = np.linspace(1.3, 18.0, 300)
    mu_rest2_curve = 3.0 * math.sqrt(14.0) / betas_c
    mu_rest1_curve = math.sqrt(6.0) / betas_c

    # Extract numerical coupled mu_sn(beta_H) directly from unified phase_table
    pt = g3_pd["phase_table"]
    tri = g3_pd["tricritical_analytical"]
    b_tri2 = tri["rho2_exact_beta_tri_k7_N36"]

    b2_pts, mu2_sn_pts = [b_tri2], [3.0 * math.sqrt(14.0) / b_tri2]
    for r in pt:
        if r["rho2_has_broken_window"] and r["beta_H"] <= 20.0:
            b2_pts.append(r["beta_H"])
            mu2_sn_pts.append(r["rho2_mu_W_sn_coupled"])
    order = np.argsort(b2_pts)
    b2_pts = np.array(b2_pts)[order]
    mu2_sn_pts = np.array(mu2_sn_pts)[order]

    # Plot rho_2 phase boundaries
    ax.plot(betas_c, mu_rest2_curve, "-", color="#1f618d", lw=2.2, label=r"$\rho_2$ Hosotani $\mu_W = 3\sqrt{14}/\beta_H$")
    ax.plot(b2_pts, mu2_sn_pts, "o-", color="#c0392b", lw=2.2, ms=4.5, label=r"$\rho_2$ coupled saddle-node $\mu_{W,\mathrm{sn}}(\beta_H)$")
    ax.fill_between(
        b2_pts, 3.0 * math.sqrt(14.0) / b2_pts, mu2_sn_pts,
        color="#f5b041", alpha=0.28, label=r"$\rho_2$ broken-symmetry metastable window",
    )
    # Also show rho_1 Hosotani curve for contrast
    ax.plot(betas_c, mu_rest1_curve, "--", color="#27ae60", lw=1.5, label=r"$\rho_1$ Hosotani $\mu_W = \sqrt{6}/\beta_H$")
    ax.axhline(1.996199, color="#c0392b", ls=":", lw=1.2)
    ax.axvline(125.1 / 80.4, color="#8e44ad", ls="-.", lw=1.6, label=r"SM $\beta_H = 1.556$ (avalanche)")
    ax.scatter([b_tri2], [3.0 * math.sqrt(14.0) / b_tri2], color="#000000", s=65, zorder=7)
    ax.annotate(r"Tricritical $\beta_{\mathrm{tri}}(\rho_2)=2.477$", xy=(b_tri2, 3.0 * math.sqrt(14.0) / b_tri2),
                xytext=(4.3, 5.7), arrowprops=dict(arrowstyle="->", color="black", lw=1.0), fontsize=8.0)

    ax.set_title(r"(E) Coupled $(A, f)$ Phase Diagram of $\rho_2$ vs. $\rho_1$")
    ax.set_xlabel(r"Higgs-to-$W$ Mass Ratio $\beta_H = m_H / M_W$")
    ax.set_ylabel(r"Dimensionless Gauge Mass $\mu_W = a M_W$")
    ax.set_xlim(1.2, 16.0)
    ax.set_ylim(0.0, 8.2)
    ax.legend(fontsize=7.3, loc="upper right")

    # -------------------------------------------------------------------------
    # Panel F: Frozen-Flat-A_{rho_2} Condensate vs. Coupled Termination
    # -------------------------------------------------------------------------
    ax = axes[1, 2]
    colors_b = {
        "beta_1.556": ("#8e44ad", r"SM $\beta_H = 1.556$ (coupled: avalanche)", 7.2141, None),
        "beta_3.464": ("#2980b9", r"$\beta_H = 2\sqrt{3} \approx 3.464$", 3.40, None),
        "beta_6.000": ("#27ae60", r"$\beta_H = 6.0$ (coupled SN $\mu_W=2.54$)", 2.5354, 6.0),
        "beta_10.000": ("#d35400", r"$\beta_H = 10.0$ (coupled SN $\mu_W=2.20$)", 2.2031, 10.0),
        "beta_20.000": ("#c0392b", r"$\beta_H = 20.0$ (coupled SN $\approx 2.05$)", 2.05, None),
    }
    for key, (col, lbl, mu_cut, b_lookup) in colors_b.items():
        blk = g3["part_c_finite_beta_higgs"]["by_beta"][key]
        sw = blk["sweep"]
        ms = np.array([r["mu_W"] for r in sw])
        frms = np.array([r["f_rms_rho2"] for r in sw])
        fmin = np.array([r["f_min_rho2"] for r in sw])
        fmax = np.array([r["f_max_rho2"] for r in sw])
        # Solid line inside coupled metastable window, dashed line beyond coupled annihilation (frozen-A only)
        m_live = ms <= mu_cut + 0.05
        ax.plot(ms, frms, "--", color=col, lw=1.3, alpha=0.55)
        ax.plot(ms[m_live], frms[m_live], "-", color=col, lw=2.3, label=lbl)
        ax.fill_between(ms[m_live], fmin[m_live], fmax[m_live], color=col, alpha=0.18)
        if b_lookup is not None:
            row_c = [r for r in pt if abs(r["beta_H"] - b_lookup) < 1e-6][0]
            ax.scatter(
                [row_c["rho2_mu_W_sn_coupled"]],
                [row_c["rho2_at_sn_below"]["f_rms"]],
                color=col, marker="X", s=65, zorder=6,
            )

    ax.set_title(r"(F) Frozen-Flat-$A_{\rho_2}$ vs. Coupled Higgs Condensate")
    ax.set_xlabel(r"Dimensionless Gauge Mass $\mu_W = a M_W$")
    ax.set_ylabel(r"Higgs Amplitude $|\hat{\Phi}| = \sqrt{2}|\Phi|/v$ (band: min–max)")
    ax.set_xlim(0.0, 9.5)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.0, loc="lower right")
    ax.text(
        0.04, 0.95,
        r"Solid: coupled metastable ($\times$: SN)" + "\n" + r"Dashed: frozen flat $A_{\rho_2}$ only",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#bdc3c7"),
    )

    out_png = os.path.join(OUT_DIR, "p03_galois_rho2_summary.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
