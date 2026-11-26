"""Generate publication-grade 6-panel diagnostic figure for Problem 1 (Tilted Bianchi IX on S^3/I*)."""

import json
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(BASE_DIR, exist_ok=True)
U0_ALG = 8.0 * math.sqrt(3.0) / 27.0


def main():
    with open(os.path.join(BASE_DIR, "deep_phase_and_killtests.json"), "r") as f:
        deep = json.load(f)
    with open(os.path.join(BASE_DIR, "synchronous_and_bounds.json"), "r") as f:
        sync_data = json.load(f)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), dpi=220)
    fig.patch.set_facecolor("white")

    # Palette (Okabe-Ito inspired)
    C_BLUE = "#005AB5"
    C_RED = "#DC3220"
    C_TEAL = "#009E73"
    C_AMBER = "#E69F00"
    C_PURPLE = "#785EF0"
    C_GRAY = "#4D4D4D"

    # ------------------------------------------------------------------
    # Panel A: Survival Ceilings vs Initial Triaxiality r0 = a1/a2
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    curve = deep["fine_r0_curve_a2_s1"]
    r0s = [x["r0"] for x in curve]
    m2_u = [x["m2_u0_max"] for x in curve]
    m1_u = [x["m1_u0_max"] for x in curve]

    ax.axhline(0.997, color=C_GRAY, ls=":", lw=1.3, label=r"Oblate Standing Berger ($0.997$)")
    ax.axhline(64.0 / 81.0, color=C_AMBER, ls=(0, (5, 2)), lw=1.4, label=r"Prolate Transverse Berger ($64/81 \approx 0.790$)")
    ax.axhline(0.750, color=C_PURPLE, ls="-.", lw=1.3, label=r"Isotropic EM-$\Lambda$ ($3/4 = 0.750$)")
    ax.axhline(U0_ALG, color=C_RED, ls="--", lw=1.6, label=r"Null-Dust Theorem ($8\sqrt{3}/27 \approx 0.513$)")
    ax.plot(r0s, m2_u, "o-", color=C_BLUE, lw=2.0, ms=4.5, label="Model II: Pure Null Dust (dyn.)")
    ax.plot(r0s, m1_u, "s-", color=C_TEAL, lw=2.0, ms=4.5, label="Model I: Exact Page 1 Maxwell")
    ax.fill_between(r0s, 0, m1_u, color=C_TEAL, alpha=0.10)
    ax.set_xlim(1.00, 1.38)
    ax.set_ylim(0.0, 1.12)
    ax.set_xlabel(r"Initial Triaxiality $r_0 = a_1(0)/a_2(0)$ ($\bar{a}_0=2,\,s_0=1$)")
    ax.set_ylabel(r"Max Surviving Radiation $u_{0,\max}$")
    ax.set_title("A. Bounce Survival Ceilings vs. Triaxiality")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.3)

    # ------------------------------------------------------------------
    # Panel B: Representative Bounce vs Collapse Scale Factors
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    tr_m1 = deep["traces"]["m1_bounce_a2_r1p15_u0p40"]["trace"]
    tr_col = deep["traces"]["m2_collapse_a2_r1p15_u0p46"]["trace"]
    t1 = np.array(tr_m1["t"])
    ax.plot(t1, tr_m1["a1"], color=C_BLUE, lw=1.9, label=r"M-I Bounce ($u_0=0.40$): $a_1$")
    ax.plot(t1, tr_m1["a2"], color=C_TEAL, lw=1.9, label=r"M-I Bounce ($u_0=0.40$): $a_2$")
    ax.plot(t1, tr_m1["a3"], color=C_AMBER, lw=2.0, label=r"M-I Bounce ($u_0=0.40$): $a_3 = \sqrt{g_{33}}$")
    tc = np.array(tr_col["t"])
    ax.plot(tc, tr_col["a2"], color=C_RED, ls="--", lw=1.7, label=r"M-II Collapse ($u_0=0.46$): $a_2 \to 0$")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlim(0.0, 4.8)
    ax.set_ylim(0.0, 5.5)
    ax.set_xlabel(r"Cosmic Proper Time $t$ ($H_\Lambda = 1$)")
    ax.set_ylabel(r"Principal Scale Factors $a_i(t)$")
    ax.set_title(r"B. Nonsingular Bounce ($g_{33} > 0$) vs. Collapse")
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    # ------------------------------------------------------------------
    # Panel C: Gravitational Birefringence & Poynting Reversal (Model I)
    # ------------------------------------------------------------------
    ax = axes[0, 2]
    ax.plot(t1, tr_m1["S3_norm"], color=C_BLUE, lw=2.0, label=r"Poynting $\hat{S}_3 = -S_3/u$")
    ax.plot(t1, tr_m1["I1_norm"], color=C_RED, lw=1.3, alpha=0.85, label=r"$I_1 = (E^2 - B^2)/(2u)$")
    ax.plot(t1, tr_m1["I2_norm"], color=C_AMBER, lw=1.3, alpha=0.85, label=r"$I_2 = (\mathbf{E}\cdot\mathbf{B})/u$")
    ax.axhline(0.0, color=C_GRAY, ls=":", lw=1.0)
    ax.set_xlim(0.0, 4.8)
    ax.set_ylim(-1.12, 1.12)
    ax.set_xlabel(r"Cosmic Proper Time $t$")
    ax.set_ylabel("Normalized Maxwell Invariants")
    ax.set_title("C. Gravitational Birefringence & Null Breaking")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92)

    # ------------------------------------------------------------------
    # Panel D: Avoided-Crossing Coordinate Ergosurface vs Smooth Sync Metric
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    sw = sync_data["synchronous_avoided_crossing"]["sync_trace_window"]
    ts = np.array(sw["t"])
    ax.plot(ts, sw["gam11"], color=C_BLUE, lw=2.0, label=r"Sync $\gamma_{11}(t)$")
    ax.plot(ts, sw["gam22"], color=C_TEAL, lw=2.0, label=r"Sync $\gamma_{22}(t)$")
    ax.plot(ts, sw["gam12"], color=C_PURPLE, lw=1.8, label=r"Sync $\gamma_{12}(t)$")
    ax.set_xlabel(r"Proper Time $t$ (Zoom Around Avoided Crossing)")
    ax.set_ylabel(r"Synchronous Metric $\gamma_{ab}(t)$ ($g_{00}^{\mathrm{sync}} \equiv -1$)")
    ax.set_title(r"D. Smooth Synchronous Metric Through $g_{00}^{\mathrm{corot}}$ Spike")

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    g00_log = np.log10(np.maximum(np.array(sw["g00_corot"]) + 2.0, 1.0))
    ax2.plot(ts, g00_log, color=C_RED, ls="--", lw=1.6, label=r"$\log_{10}(g_{00}^{\mathrm{corot}}+2)$")
    ax2.set_ylabel(r"Co-Rotating $\log_{10}(g_{00}^{\mathrm{corot}} + 2)$", color=C_RED)
    ax2.tick_params(axis="y", labelcolor=C_RED)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, framealpha=0.92)

    # ------------------------------------------------------------------
    # Panel E: Sharpness of 8*sqrt(3)/27 & Asymptotic Scaling vs a_mean
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    asymp = deep["asymptotic_scaling"]
    ams = [x["a_mean"] for x in asymp]
    m2_as = [x["best_model_II"]["u0_max"] for x in asymp]
    m1_as = [x["best_model_I"]["u0_max"] for x in asymp]

    ax.axhline(64.0 / 81.0, color=C_AMBER, ls=(0, (5, 2)), lw=1.4, label=r"Prolate $S_3\!=\!0$ Ceiling ($64/81 \approx 0.790$)")
    ax.axhline(U0_ALG, color=C_RED, ls="--", lw=1.5, label=r"Null-Dust Ceiling ($8\sqrt{3}/27 \approx 0.513$)")
    ax.plot([0.724], [0.78389], "^", color=C_PURPLE, ms=8.0, zorder=5, label=r"M-I Drained-Bounce ($\mathcal{I}_{\mathrm{EM}}\!=\!0.784$)")
    ax.plot([0.68], [0.513199], "*", color=C_RED, ms=10.5, zorder=5, label=r"M-II Complete Bounce ($99.9998\%$)")
    ax.plot([0.70], [0.510], "D", color=C_AMBER, ms=6.0, zorder=5, label=r"M-I Null-at-Bounce ($99.38\%$)")
    ax.plot(ams, m2_as, "o-", color=C_BLUE, lw=2.0, ms=5.0, label=r"M-II Untuned $H_1\!=\!H_2\!=\!H_3$")
    ax.plot(ams, m1_as, "s-", color=C_TEAL, lw=2.0, ms=5.0, label=r"M-I Untuned Page 1 Null")
    ax.set_xscale("log")
    ax.set_xlim(0.55, 25.0)
    ax.set_ylim(0.0, 0.88)
    ax.set_xlabel(r"Initial Mean Scale Factor $\bar{a}_0 = (a_1 a_2 a_3)^{1/3}$")
    ax.set_ylabel(r"Max Surviving $u_0$ / $\mathcal{I}_{\mathrm{EM}}$")
    ax.set_title("E. Ceiling Hierarchy & Asymptotic Scaling")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=6.9)

    # ------------------------------------------------------------------
    # Panel F: Exact Unclamped Constraint Residuals |H(t)| & |M_3(t)|
    # ------------------------------------------------------------------
    ax = axes[1, 2]
    tr_m2 = deep["traces"]["m2_bounce_a2_r1p15_u0p40"]["trace"]
    tr_fp = deep["traces"]["m2_near_fp_a1p05_u0p510"]["trace"]
    ax.semilogy(tr_m1["t"], np.maximum(tr_m1["H_res"], 1e-17), color=C_BLUE, lw=1.8, label=r"M-I ($u_0=0.40$): $|\mathcal{H}|$")
    ax.semilogy(tr_m1["t"], np.maximum(tr_m1["M_res"], 1e-17), color=C_BLUE, ls="--", lw=1.5, label=r"M-I ($u_0=0.40$): $|\mathcal{M}_3|$")
    ax.semilogy(tr_m2["t"], np.maximum(tr_m2["H_res"], 1e-17), color=C_TEAL, lw=1.7, label=r"M-II ($u_0=0.40$): $|\mathcal{H}|$")
    ax.semilogy(tr_fp["t"], np.maximum(tr_fp["H_res"], 1e-17), color=C_AMBER, lw=1.7, label=r"M-II Near-ES ($u_0=0.510$): $|\mathcal{H}|$")
    ax.axhline(1e-11, color=C_RED, ls=":", lw=1.4, label=r"Target Gate ($10^{-11}$)")
    ax.set_xlim(0.0, 4.8)
    ax.set_ylim(1e-17, 1e-9)
    ax.set_xlabel(r"Cosmic Proper Time $t$")
    ax.set_ylabel("Absolute Constraint Residual")
    ax.set_title("F. Unclamped Hamiltonian & Momentum Residuals")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.5)

    fig.suptitle(
        r"Problem 1: Tilted Rotating Triaxial Bianchi IX Einstein–Maxwell–$\Lambda$ Cosmology on $S^3/I^*$",
        fontsize=13.0, fontweight="bold", y=0.985,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(BASE_DIR, "p01_tilted_null_bianchi9_summary.png")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print("Saved 6-panel summary figure ->", out_png)


if __name__ == "__main__":
    main()
