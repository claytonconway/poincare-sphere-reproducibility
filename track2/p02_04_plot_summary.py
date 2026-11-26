"""P02-04: Publication-Grade 4-Panel Summary Figure for Problem 2 (Inhomogeneous SU(2)_L-Higgs on S^3/I*).

Panels:
  (A) Exact Pure-YM Tachyonic Threshold E_tach(n) on S^3 vs. S^3/I* (highlighting the 280.1x I* spectral-gap jump
      from n = 4 [E_tach = 128/81 = 1.580] to n = 12 [E_tach = 6480000/14641 = 442.593]).
  (B) Exact Floquet Growth Rate sigma_Fl = ln(|mu_max|)/T vs. E_YM in [0.6, 5000] for n = 0 (homogeneous),
      S^3-only modes (n = 1, 2, 4), and surviving S^3/I* modes (n = 12, 20), showing both tachyonic bands
      and sub-tachyonic parametric/Lamé tongues.
  (C) Higgs Stabilization Asymmetry at E_tot = 1000 vs. conformal W mass mu_W: S^3/I* (n = 0, 12, 20)
      completely stabilized by mu_W = 5.0 while S^3-only n = 4 remains unstable (sigma = 0.903).
  (D) Dynamical Gravitating Bounce Amplification (gamma_YM = 0.10) vs. E_YM across S^3 vs. S^3/I*
      with inset showing exact O(gamma_YM) gauge-graviton backreaction shift across N = 3..19.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    d1 = json.load(open(os.path.join(OUT_DIR, "p02_01_operator_verification.json")))
    d2 = json.load(open(os.path.join(OUT_DIR, "p02_02_floquet_instability.json")))
    d3 = json.load(open(os.path.join(OUT_DIR, "p02_03_bounce_and_gravity.json")))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
    })

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6))
    fig.patch.set_facecolor("#FAF9F6")
    for ax in axes.flat:
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle=":", alpha=0.45, color="#9AA5B1")

    # Colors: crisp, colorblind-safe technical palette
    c_hom = "#1F2937"   # charcoal (n=0)
    c_s3a = "#DC2626"   # crimson (n=1)
    c_s3b = "#EA580C"   # orange (n=2)
    c_s3c = "#9333EA"   # purple (n=4, parity twin)
    c_is1 = "#0D9488"   # teal (n=12, S^3/I*)
    c_is2 = "#2563EB"   # royal blue (n=20, S^3/I*)

    # ------------------------------------------------------------------
    # Panel A: Exact Tachyonic Threshold E_tach(n) on S^3 vs. S^3/I*
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    ns_all = list(range(0, 31))
    molien = d1["molien_first_32"]["m_n_trivial"]
    Et_all = []
    for n in ns_all:
        m = (n + 3) if n <= 3 else (n - 1)
        Et_all.append(((m * m - 1.0) ** 4) / (32.0 * (m ** 4)))

    ns_s3_only = [n for n in ns_all if molien[n] == 0]
    Et_s3_only = [Et_all[n] for n in ns_s3_only]
    ns_istar = [n for n in ns_all if molien[n] > 0]
    Et_istar = [Et_all[n] for n in ns_istar]

    ax.axvspan(0.5, 11.5, color="#FEE2E2", alpha=0.55, label=r"$I^*$ Forbidden Gap ($1 \leq n \leq 11$, $m_n=0$)")
    ax.semilogy(ns_s3_only, Et_s3_only, "o", color=c_s3a, markersize=6, alpha=0.75, label=r"$S^3$-only modes (eliminated by $I^*$)")
    ax.semilogy(ns_istar, Et_istar, "D", color=c_is1, markersize=8.5, markeredgecolor="#111827", markeredgewidth=0.9,
                label=r"Surviving $S^3/I^*$ modes ($m_n \geq 1$)")

    ax.annotate(r"$n=0, K=2$" + "\n" + r"$E_{\rm tach}=\frac{128}{81}=1.58$",
                xy=(0, Et_all[0]), xytext=(1.8, 0.55),
                arrowprops=dict(arrowstyle="->", color=c_hom, lw=1.0), fontsize=8.5, color=c_hom)
    ax.annotate(r"$n=4, K=0$ ($S^3$ twin)" + "\n" + r"$E_{\rm tach}=1.58$ (FORBIDDEN)",
                xy=(4, Et_all[4]), xytext=(5.2, 2.2),
                arrowprops=dict(arrowstyle="->", color=c_s3c, lw=1.1), fontsize=8.5, color=c_s3c, fontweight="bold")
    ax.annotate(r"First $S^3/I^*$ inhom." + "\n" + r"$n=12, K=4$: $E_{\rm tach}=442.6$" + "\n" + r"($280.1\times$ higher!)",
                xy=(12, Et_all[12]), xytext=(13.5, 65.0),
                arrowprops=dict(arrowstyle="->", color=c_is1, lw=1.2), fontsize=8.5, color=c_is1, fontweight="bold")

    ax.set_xlabel(r"Harmonic degree $n = 2j$")
    ax.set_ylabel(r"Pure-YM Tachyonic Threshold $E_{\rm tach}(n)$")
    ax.set_title(r"(A) Exact Tachyonic Thresholds: $S^3$ vs. $S^3/I^*$ ($280.1\times$ Gap)")
    ax.set_xlim(-1.0, 31.2)
    ax.set_ylim(0.35, 5e4)
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.92)

    # ------------------------------------------------------------------
    # Panel B: Pure-YM Floquet Growth Rate sigma_Fl vs. E_YM
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    scan = d2["pure_ym_floquet_scan"]
    Es = np.array([r["E_YM"] for r in scan])
    s0 = np.array([r["by_n"]["0"]["sigma_fl"] for r in scan])
    s1 = np.array([r["by_n"]["1"]["sigma_fl"] for r in scan])
    s4 = np.array([r["by_n"]["4"]["sigma_fl"] for r in scan])
    s12 = np.array([r["by_n"]["12"]["sigma_fl"] for r in scan])
    s20 = np.array([r["by_n"]["20"]["sigma_fl"] for r in scan])

    ax.semilogx(Es, s0, "-", color=c_hom, lw=1.6, alpha=0.75, label=r"$n=0$ (homogeneous)")
    ax.semilogx(Es, s1, "-", color=c_s3a, lw=1.8, label=r"$n=1$ ($S^3$-only)")
    ax.semilogx(Es, s4, "--", color=c_s3c, lw=1.6, label=r"$n=4$ ($S^3$-only, twin)")
    ax.semilogx(Es, s12, "-", color=c_is1, lw=2.3, label=r"$n=12$ (first $S^3/I^*$)")
    ax.semilogx(Es, s20, "-", color=c_is2, lw=1.8, label=r"$n=20$ ($S^3/I^*$)")

    ax.axvline(128.0 / 81.0, color=c_hom, ls=":", lw=1.0, alpha=0.7)
    ax.axvline(442.5927, color=c_is1, ls=":", lw=1.2, alpha=0.85)
    ax.text(442.6 * 1.08, 3.35, r"$E_{\rm tach}(12)=442.6$", color=c_is1, fontsize=8.2, fontweight="bold")
    ax.annotate(r"Sub-tach. tongue" + "\n" + r"($E\approx 195$, $\sigma=0.33$)",
                xy=(195.0, 0.33), xytext=(15.0, 1.35),
                arrowprops=dict(arrowstyle="->", color=c_is1, lw=1.0), fontsize=8, color=c_is1)
    ax.axvspan(0.6, 147.0, color="#CCFBF1", alpha=0.32, label=r"100% $S^3/I^*$ inhom. stable ($E<147$)")

    ax.set_xlabel(r"Conformal Yang–Mills Energy $E_{\rm YM}$")
    ax.set_ylabel(r"Floquet Growth Rate $\sigma_{\rm Fl} = \ln|\mu_{\max}| / T$")
    ax.set_title(r"(B) Pure-YM Floquet Growth Rate $\sigma_{\rm Fl}(E_{\rm YM})$")
    ax.set_xlim(0.6, 5000.0)
    ax.set_ylim(-0.1, 4.0)
    ax.legend(loc="upper left", fontsize=7.8, frameon=True, framealpha=0.92)

    # ------------------------------------------------------------------
    # Panel C: Higgs Proca Stabilization at E_tot = 500 (n=12 2K=8 peak) & E_tot = 1000
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    row500 = [r for r in d2["higgs_proca_floquet_map"]["map"] if abs(r["E_tot"] - 500.0) < 1e-6][0]
    row1000 = [r for r in d2["higgs_proca_floquet_map"]["map"] if abs(r["E_tot"] - 1000.0) < 1e-6][0]
    mus = np.array([e["mu_W"] for e in row500["by_mu_W"]])

    # Plot E_tot = 500 (where n=12 2K=8 starts at peak sigma = 0.898!) and n=4,12 at E=1000
    for n_str, col, ls, mk, lbl in [
        ("0", c_hom, "-", "s", r"$n=0$ ($E=500$, hom.)"),
        ("1", c_s3a, "-", "o", r"$n=1$ ($E=500$, $S^3$)"),
        ("4", c_s3c, "-", "^", r"$n=4$ ($E=500$, $S^3$)"),
        ("12", c_is1, "-", "D", r"$n=12$ ($E=500$, $2K=8$ peak)"),
        ("4", c_s3c, "--", "^", r"$n=4$ ($E=1000$, $2K=8$ survives at $\mu_W=5$)"),
        ("12", c_is2, "--", "v", r"$n=12$ ($E=1000$, $2K=10/14$ Lamé)"),
    ]:
        src = row1000 if "1000" in lbl else row500
        ys = np.array([e["by_n"][n_str]["sigma_fl"] for e in src["by_mu_W"]])
        ax.plot(mus, ys, linestyle=ls, marker=mk, color=col, lw=1.9, markersize=5.5, label=lbl)

    ax.annotate(r"$E=1000$: $n=0,12,20$ all $\sigma=0$" + "\n" + r"while $S^3$ $n=4$ ($2K=8$) stays $\sigma=0.903$",
                xy=(5.0, 0.903), xytext=(0.08, 1.22),
                bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec="#D8B4FE", alpha=0.92),
                arrowprops=dict(arrowstyle="->", color=c_s3c, lw=1.1, connectionstyle="arc3,rad=0.14"),
                fontsize=7.6, color=c_s3c, fontweight="bold")
    ax.set_xlabel(r"Conformal W-Boson Mass $\mu_W = a M_W$")
    ax.set_ylabel(r"Floquet Growth Rate $\sigma_{\rm Fl}$")
    ax.set_title(r"(C) Higgs Stabilization ($E_{\rm tot}=500$ Peak & $E_{\rm tot}=1000$)")
    ax.set_xlim(-0.15, 5.2)
    ax.set_ylim(-0.1, 2.55)
    ax.legend(loc="upper right", fontsize=7.2, frameon=True, framealpha=0.94)

    # ------------------------------------------------------------------
    # Panel D: Dynamical Gravitating Bounce Amplification (gamma = 0.10)
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    b_rows = d3["dynamical_bounce_pure_ym_gamma010"]
    Eb = np.array([r["E_YM"] for r in b_rows])
    for n_str, col, ls, mk, lbl in [
        ("0", c_hom, "-", "s", r"$n=0$ (hom.)"),
        ("1", c_s3a, "-", "o", r"$n=1$ ($S^3$)"),
        ("4", c_s3c, "--", "^", r"$n=4$ ($S^3$)"),
        ("12", c_is1, "-", "D", r"$n=12$ ($S^3/I^*$)"),
        ("20", c_is2, ":", "v", r"$n=20$ ($S^3/I^*$)"),
    ]:
        amps = np.array([r["by_n"][n_str]["max_A_energy"] for r in b_rows])
        ax.loglog(Eb, amps, linestyle=ls, marker=mk, color=col, lw=2.0, markersize=6.5, label=lbl)

    ax.set_xlabel(r"Bounce Yang–Mills Energy $E_{\rm YM}$ ($\gamma_{\rm YM}=\kappa_{\rm YM}E_{\rm YM}=0.10$)")
    ax.set_ylabel(r"Max Energy-Normalized Amplification $\mathcal{A}_{\rm bounce}$")
    ax.set_title(r"(D) Gravitating Bounce Amplification ($a: 2 \to a_{\min} \to 6$)")
    ax.set_ylim(0.75, 4500.0)
    ax.legend(loc="upper left", ncol=2, fontsize=7.6, frameon=True, framealpha=0.95)

    # Inset in empty middle-left space (no overlap with data curves!)
    inset = ax.inset_axes([0.11, 0.40, 0.42, 0.32])
    inset.set_facecolor("#F8FAFC")
    g_rows = [r for r in d3["gravity_backreaction_audit"] if r["gamma_YM"] == 0.10]
    shifts_in = [abs(100.0 * r["rel_shift_qq"]) for r in g_rows]
    labels_in = [f"n={r['n']}\nN={r['N']}" for r in g_rows]
    inset.bar(range(len(g_rows)), shifts_in, color=[c_hom, c_s3a, c_s3c, c_is1, c_is2], alpha=0.88, width=0.58)
    inset.set_xticks(range(len(g_rows)))
    inset.set_xticklabels(labels_in, fontsize=6.6)
    inset.set_ylabel(r"$|\Delta\mathcal{A}/\mathcal{A}|$ (%)", fontsize=7.2)
    inset.set_title(r"$O(\gamma)$ GW Backreaction ($\gamma=0.10, E=200$)", fontsize=7.2, fontweight="bold", pad=3)
    inset.tick_params(axis="y", labelsize=6.8)
    inset.set_ylim(0, 5.8)
    inset.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        r"Problem 2: Inhomogeneous $SU(2)_L$–Higgs Perturbations Around Cosmological Bounces on $S^3$ vs. $S^3/I^*$",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.962])
    out_png = os.path.join(OUT_DIR, "p02_summary_figure.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
