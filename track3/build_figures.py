#!/usr/bin/env python3
"""Generate all 4 publication-clean figures for Track III (Semiclassical Starobinsky-Casimir, Berger Collapse, Riegert-Weyl C^2).

Outputs into --out-dir (default: ../generated_figures/track3):
  - fig1_chiral_ade_casimir_eta.png
  - fig2_starobinsky_casimir_kam.png
  - fig3_berger_anisotropic_collapse.png
  - fig4_riegert_weyl_c2_spin2.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

TRACK_DIR = Path(__file__).resolve().parent
DATA_DIR = TRACK_DIR / "data"

if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from p05_02_solve_and_phase_space import (
    KAPPA_VECTOR,
    SIGMA0_VECTOR,
    ode_rhs,
    bounce_initial_state,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.3,
    "axes.titlesize": 10.3,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.3,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "legend.frameon": True,
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#cccccc",
    "legend.fontsize": 7.9,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
})


def make_fig1(out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=220)
    fig.patch.set_facecolor("#ffffff")

    ax = axes[0]
    neg_l = [2, 14, 22, 26, 32]
    neg_d = [3, 15, 23, 27, 33]
    pos_l = [12, 20, 24, 30, 32]
    pos_d = [11, 19, 23, 29, 31]

    ax.bar([x - 0.35 for x in neg_l], [-d for d in neg_d], width=0.85, color="#c0392b", alpha=0.9,
           label=r"Negative helicity $\lambda = -(n+2)$, mult $m_n(n+3)$")
    ax.bar([x + 0.35 for x in pos_l], pos_d, width=0.85, color="#1f618d", alpha=0.9,
           label=r"Positive helicity $\lambda = +n$, mult $m_n(n-1)$")
    ax.axhline(0, color="#222222", lw=0.9)
    ax.axvline(32, color="#7f8c8d", ls="--", lw=1.0)
    ax.annotate(
        "First chiral overlap\n" + r"at $|\lambda| = 32$ ($-32$: 33, $+32$: 31)",
        xy=(31.6, 25.5), xytext=(11.5, 25.0),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9),
        fontsize=7.6, bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9f9", ec="#7f8c8d", lw=0.8),
    )
    ax.text(
        1.0, -35.5,
        r"ADE round-$S^3/\Gamma$ Maxwell $E_{\mathrm{Cas}}^{\mathrm{EM}}$ ($\eta_{\mathrm{curl}}(0)$):" + "\n"
        + r"$S^3$: $+\frac{11}{120}$ ($0$)   |   $S^3/T^*$: $+\frac{79}{270}$ ($-\frac{49}{36}$)" + "\n"
        + r"$S^3/O^*$: $+\frac{23}{1080}$ ($-\frac{121}{72}$)   |   $S^3/I^*$: $-\frac{349}{675}$ ($-\frac{361}{180}$)" + "\n"
        + r"Palindromic identity: $\Delta E_{\mathrm{Cas}} \equiv 0$ identically",
        fontsize=7.5, bbox=dict(boxstyle="round,pad=0.35", fc="#fdfefe", ec="#1f618d", lw=0.85),
    )
    ax.set_xlabel(r"Curl frequency $|\lambda|$ (units of $1/a$)")
    ax.set_ylabel(r"Signed degeneracy ($\pm d_\pm$)")
    ax.set_title(r"A. Chiral Maxwell Curl Spectrum on $S^3/I^*$")
    ax.set_xlim(-0.5, 35.5)
    ax.set_ylim(-39, 44)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[1]
    vac_labels = [
        r"$\rho_0$ ($\mathrm{CS}=0$)" + "\n" + r"$h^0=3,\ \mathrm{SF}=0$",
        r"$\rho_1$ ($\mathrm{CS}=\frac{1}{120}$)" + "\n" + r"$h^0=0,\ \mathrm{SF}=1$",
        r"$\rho_2$ ($\mathrm{CS}=\frac{49}{120}$)" + "\n" + r"$h^0=0,\ \mathrm{SF}=5$",
    ]
    x_pos = np.arange(3)
    E_ym = [-349.0 / 225.0, -221.0 / 150.0, +79.0 / 150.0]
    E_sc = [-43553.0 / 14400.0, -4891.0 / 4800.0, +1829.0 / 4800.0]
    w = 0.32
    ax.bar(x_pos - w / 2, E_ym, width=w, color=["#1f618d", "#2980b9", "#c0392b"],
           edgecolor="#222222", lw=0.7, label=r"Pure $SU(2)$ YM $E_{\mathrm{Cas}}^{\mathrm{YM}}(\mathrm{Ad}\,\rho_i)$ $[\hbar c/a]$")
    ax.bar(x_pos + w / 2, E_sc, width=w, color=["#7fb3d5", "#a9cce3", "#f1948a"],
           edgecolor="#555555", lw=0.7, hatch="//", label=r"Adjoint conformal scalar $E_{\mathrm{Cas}}^{\mathrm{scal}}(\mathrm{Ad}\,\rho_i)$")
    ax.axhline(0, color="#222222", lw=1.0)
    ann_ym = [r"$-\frac{349}{225}$", r"$-\frac{221}{150}$", r"$+\frac{79}{150}$"]
    for i, (val, txt) in enumerate(zip(E_ym, ann_ym)):
        dy = -0.26 if val < 0 else 0.14
        ax.text(x_pos[i] - w / 2, val + dy, txt, ha="center", va="center", fontsize=8.3, fontweight="bold")
    ax.annotate("", xy=(2 - w / 2, E_ym[2] - 0.02), xytext=(1 - w / 2, E_ym[1] + 0.02),
                arrowprops=dict(arrowstyle="<->", color="#8e44ad", lw=1.4))
    ax.text(
        1.26, 0.36, r"$\Delta E^{\mathrm{YM}}(\rho_2 - \rho_1) = +2\,\frac{\hbar c}{a}$" + "\n(exact pure integer)",
        fontsize=7.5, color="#6c3483", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="#f5eef8", ec="#8e44ad", lw=0.8),
    )
    ax.text(
        0.03, 0.96,
        r"1-loop Hosotani ordering ($g_{\mathrm{YM}}\ll 1$):" + "\n"
        + r"$\rho_0 < \rho_1 < 0 < \rho_2$" + "\n"
        + r"$\rho_0$: KAM $x_{\mathrm{ring}}=21.92$" + "\n"
        + r"$\rho_1$: KAM $x_{\mathrm{ring}}=21.36$" + "\n"
        + r"$\rho_2$: $E>0 \Rightarrow$ KAM destroyed",
        transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fef9e7", ec="#d4ac0d", lw=0.8),
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(vac_labels, fontsize=8.3)
    ax.set_ylabel(r"Exact 1-loop Casimir energy $[\hbar c/a]$")
    ax.set_title(r"B. $\mathrm{Ad}(\rho_i)$-Twisted Pure-$SU(2)$ YM Vacuum Lifting on $S^3/I^*$")
    ax.set_ylim(-3.45, 1.55)
    ax.grid(True, axis="y")
    ax.legend(loc="lower right", fontsize=7.4)

    fig.suptitle(
        r"Chiral Maxwell Spectral Asymmetry, ADE Casimir Energies, and Twisted $SU(2)$ Lifting on $S^3/I^*$",
        fontsize=11.0, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = out_dir / "fig1_chiral_ade_casimir_eta.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"OK: {out}")
    return out


def make_fig2(out_dir: Path) -> Path:
    d_p05 = json.loads((DATA_DIR / "p05_04_critic_fixes.json").read_text(encoding="utf-8"))
    d_r32 = json.loads((DATA_DIR / "r3_02_ade_lens_species_escape.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6), dpi=220)
    fig.patch.set_facecolor("#ffffff")

    ax = axes[0, 0]
    x_ring = math.sqrt(44672.0 / 93.0)
    for xb, col, lbl in [
        (x_ring, "#1f618d", rf"Core ring $x_b = x_{{\mathrm{{ring}}}} = 21.92$"),
        (10.0, "#16a085", r"Eccentric KAM $x_b = 10.0$ ($x_{\max}\approx 44.5$)"),
        (5.5, "#c0392b", r"Plateau $\to$ recollapse crash ($x_b = 5.5$)"),
    ]:
        y0 = bounce_initial_state(xb, SIGMA0_VECTOR, 0.0, KAPPA_VECTOR, s_b=0.0, sign_hdot=+1)
        tmax = 600.0 if xb >= 10.0 else 80.0
        ev_bl = lambda t, y: 1e3 - max(abs(y[1]), abs(y[2]))
        ev_bl.terminal = True
        ev_bl.direction = -1
        sol = solve_ivp(
            lambda t, y: ode_rhs(t, y, SIGMA0_VECTOR, 0.0), (0.0, tmax), y0,
            method="DOP853", rtol=1e-11, atol=1e-13, events=[ev_bl], max_step=0.25,
        )
        ax.plot(sol.t, sol.y[0], color=col, lw=1.4 if xb >= 10 else 1.7, label=lbl)
    ax.axhline(x_ring, color="#1f618d", ls=":", lw=0.9, alpha=0.7)
    ax.set_xlabel(r"Dimensionless Starobinsky time $\tau = H_S t$")
    ax.set_ylabel(r"Dimensionless scale factor $x(\tau) = H_S a$")
    ax.set_title(r"A. $S^3/I^*$ Adiabatic KAM Breathing Island ($\Lambda = 0, \tilde{\sigma}_0 = 9/31$)")
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 56)
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.text(
        0.03, 0.06,
        r"$\omega_{\mathrm{scal}}/\Omega_{\mathrm{macro}} = 40.68$" + "\n"
        + r"$\tau=10^4$ envelope drift: $5.30\times 10^{-6}$" + "\n"
        + r"Pure Euler ($\tilde{\sigma}=0$): bounce destroyed",
        transform=ax.transAxes, fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#eef6fb", ec="#1f618d", lw=0.8),
    )

    ax = axes[0, 1]
    pmap = d_p05["critical_1_2d_phase_map_and_lambda_KAM"]
    lam_kam = pmap["lambda_KAM_pinchoff_at_xring"]
    kap_eff = -44672.0 / 93.0
    lam_line = np.geomspace(1e-5, 0.25, 300)
    xb_max_line = np.sqrt((1.0 + np.sqrt(1.0 - 4.0 * lam_line * kap_eff)) / (2.0 * lam_line))
    xcrit_line = 3.94068 + (4.58393 - 3.94068) * (lam_line / 0.24) ** 0.85

    ax.fill_betweenx(lam_line, 2.0, xcrit_line, color="#f5b7b1", alpha=0.85, label=r"UV blowup ($x_b < x_{\mathrm{crit}}(\lambda)$)")
    ax.fill_betweenx(lam_line, xcrit_line, xb_max_line, color="#abebc6", alpha=0.75, label=r"Eternal IR-dS $\leftrightarrow$ IR-dS bounce")
    ax.fill_betweenx(lam_line, xb_max_line, 350.0, color="#ebedef", alpha=0.95, label=r"No turning point ($e_{00} = 0$ forbidden)")
    ax.plot(xcrit_line, lam_line, color="#c0392b", lw=1.6)
    ax.plot(xb_max_line, lam_line, color="#566573", lw=1.5)

    pc = [r for r in pmap["pocket_curve"] if r.get("x_lo") is not None and r.get("x_hi") is not None]
    pc_lam = np.array([r["lam"] for r in pc])
    pc_lo = np.array([r["x_lo"] for r in pc])
    pc_hi = np.array([r["x_hi"] for r in pc])
    ax.fill_betweenx(pc_lam, pc_lo, pc_hi, color="#1f618d", alpha=0.85, label=r"Trapped KAM breathing pocket ($\lambda < \lambda_{\mathrm{KAM}}$)")
    ax.plot([97.97, 124.29], [2e-5, 2e-5], color="#8e44ad", lw=4.5, solid_capstyle="butt",
            label=r"Recollapse blowup tongue ($\lambda = 2\times 10^{-5}$)")
    ax.axhline(lam_kam, color="#154360", ls="--", lw=1.2)
    ax.text(30.0, lam_kam * 1.25, r"$\lambda_{\mathrm{KAM}}(x_{\mathrm{ring}}) = 1.545\times 10^{-4}$",
            fontsize=7.6, color="#154360", fontweight="bold")
    ax.axvline(x_ring, color="#154360", ls=":", lw=0.9, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Bounce scale factor $x_b = H_S a_b$")
    ax.set_ylabel(r"Cosmological constant $\lambda = \Lambda / (3 H_S^2)$")
    ax.set_title(r"B. Global $(\lambda, x_b)$ Bounce Phase Diagram on $S^3/I^*$")
    ax.set_xlim(2.5, 280)
    ax.set_ylim(1e-5, 0.28)
    ax.grid(True, which="both", alpha=0.45)
    ax.legend(loc="upper right", fontsize=6.9)

    ax = axes[1, 0]
    lens_rows = d_r32["lens_tower"]
    dih_rows = d_r32["dihedral_tower"]
    ax.plot([r["order_G"] for r in lens_rows], [r["L_min_aCas_V_over_l_sp_red"] for r in lens_rows],
            color="#d35400", lw=1.8, ls="--", label=r"$L(p,1)$ static round metric ($p \geq 34$ exceeds 1)")
    ax.plot([r["order_G"] for r in dih_rows], [r["L_min_aCas_V_over_l_sp_red"] for r in dih_rows],
            color="#8e44ad", lw=1.8, ls="-.", label=r"$S^3/D_m^*$ static round metric ($m \geq 17$ exceeds 1)")
    iso_pts = [
        (2, 0.1768, r"$\mathbb{R}P^3$"),
        (8, 0.2652, r"$S^3/Q_8$"),
        (24, 0.2533, r"$S^3/T^*$"),
        (48, 0.0696, r"$S^3/O^*$"),
        (120, 0.4551, r"$S^3/I^*$"),
    ]
    ax.scatter([x for x, _, _ in iso_pts], [y for _, y, _ in iso_pts],
               color="#1f618d", s=62, zorder=6, edgecolor="#ffffff", lw=0.8,
               label=r"All 5 isotropy-irreducible $S^3/\Gamma$ (all $<1$, $\forall N_V\geq 1$)")
    offsets = [(0, 0.44), (4, 0.56), (12, -0.10), (7, 0.14), (-18, 0.14)]
    for (gx, gy, name), (dx, dy) in zip(iso_pts, offsets):
        ax.annotate(name, xy=(gx, gy), xytext=(gx + dx, gy + dy), fontsize=8.0, fontweight="bold",
                    color="#154360", arrowprops=dict(arrowstyle="-", color="#7f8c8d", lw=0.7))
    ax.axhline(1.0, color="#c0392b", lw=1.3, ls="-", label=r"Dvali species threshold $L_{\min}(a_{\mathrm{Cas}})/\ell_{\mathrm{sp}}^{\mathrm{red}} = 1$")
    ax.axhspan(0, 1.0, color="#fdedec", alpha=0.35)
    ax.set_xlabel(r"Quotient group order $|\Gamma|$ (fixed $N_V = 1$; exact $N$-invariant)")
    ax.set_ylabel(r"$L_{\min}(a_{\mathrm{Cas}}) / \ell_{\mathrm{sp}}^{\mathrm{red}}$")
    ax.set_title(r"C. Isotropy vs. Species-Scale Squeeze Across $S^3/\Gamma$")
    ax.set_xlim(0, 145)
    ax.set_ylim(0, 2.25)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7.2)

    ax = axes[1, 1]
    frac_V = np.linspace(0.0, 1.0, 250)
    pi2_dC = (1.0 - frac_V) * (87109.0 / 1440.0) + frac_V * (22369.0 / 720.0)
    N_sp = (1.0 - frac_V) + 2.0 * frac_V
    ratio_mix = (1.0 / 5.0) * np.sqrt(pi2_dC / (3.0 * N_sp))
    c_num = (1.0 - frac_V) * 1.0 + frac_V * 12.0
    a_num = (1.0 - frac_V) * 1.0 + frac_V * 62.0
    unit_ratio = 6.0 * math.pi * c_num / a_num

    ax.plot(frac_V, ratio_mix, color="#1f618d", lw=2.2, label=r"$L_{\min}(a_{\mathrm{Cas}})/\ell_{\mathrm{sp}}^{\mathrm{red}} \leq 0.8981 < 1$")
    ax.axhline(1.0, color="#c0392b", ls="--", lw=1.2, label=r"Semiclassical threshold $= 1$")
    ax.fill_between(frac_V, ratio_mix, 1.0, color="#ebf5fb", alpha=0.5)
    ax.set_xlabel(r"Vector fraction $f_V = N_V / (N_S + N_V)$ (all $N_{\mathrm{tot}} \geq 1$)")
    ax.set_ylabel(r"Casimir / species ratio $L_{\min}(a_{\mathrm{Cas}})/\ell_{\mathrm{sp}}^{\mathrm{red}}$", color="#1f618d")
    ax.set_ylim(0.0, 1.18)
    ax.set_xlim(0.0, 1.0)
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.plot(frac_V, unit_ratio, color="#d35400", lw=1.9, ls="-.", label=r"UV plateau $H_S^2 / (M_{\mathrm{unit}}^{\mathrm{HW}})^2 \in [3.65, 18.85]$")
    ax2.set_ylabel(r"UV unitarity ratio $H_S^2 / (M_{\mathrm{unit}}^{\mathrm{HW}})^2$", color="#d35400")
    ax2.set_ylim(0.0, 21.5)
    ax.set_title(r"D. Exact $N$-Invariant Species Squeeze & Spin-2 Unitarity on $S^3/I^*$")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=7.3)

    fig.suptitle(
        r"Self-Consistent $O(\hbar)$ Starobinsky–Casimir Cosmology and the Species-Scale Squeeze on $S^3/\Gamma$",
        fontsize=11.3, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = out_dir / "fig2_starobinsky_casimir_kam.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"OK: {out}")
    return out


def make_fig3(out_dir: Path) -> Path:
    d1 = json.loads((DATA_DIR / "t3_01_berger_casimir_backreaction.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), dpi=220)
    fig.patch.set_facecolor("#ffffff")

    ax = axes[0]
    pL40 = d1["berger_casimir_profiles"]["L40_em"]
    pD20 = d1["berger_casimir_profiles"]["D20_em"]
    lam_L = np.array(pL40["lam_grid"])
    ppar_L = np.array(pL40["p_par_over_u"])
    pperp_L = np.array(pL40["p_perp_over_u"])
    m_neg = lam_L <= 3.2
    ax.plot(lam_L[m_neg], ppar_L[m_neg], color="#c0392b", lw=2.0, label=r"$L(40,1)$ EM: $p_\parallel / u$ (fiber tension, $p_\parallel < 0$)")
    ax.plot(lam_L[m_neg], pperp_L[m_neg], color="#1f618d", lw=2.0, label=r"$L(40,1)$ EM: $p_\perp / u$ (transverse pressure, $p_\perp > 0$)")

    lam_D = np.array(pD20["lam_grid"])
    mD = lam_D <= 3.2
    ax.plot(lam_D[mD], np.array(pD20["p_par_over_u"])[mD], color="#e67e22", lw=1.4, ls="--", label=r"$S^3/D_{20}^*$ EM: $p_\parallel / u$")
    ax.plot(lam_D[mD], np.array(pD20["p_perp_over_u"])[mD], color="#2980b9", lw=1.4, ls="--", label=r"$S^3/D_{20}^*$ EM: $p_\perp / u$")

    ax.axhline(3.0, color="#c0392b", ls=":", lw=0.9, alpha=0.7)
    ax.axhline(-1.0, color="#1f618d", ls=":", lw=0.9, alpha=0.7)
    ax.axhline(1.0 / 3.0, color="#7f8c8d", ls="-.", lw=0.9, label=r"Isotropic radiation ($p/u = 1/3$)")
    ax.axvline(1.0, color="#333333", ls=":", lw=1.0)
    ax.text(1.06, 1.35, r"Round $\lambda_B = 1$:" + "\n" + r"$p_\parallel/u = +3.08$" + "\n" + r"$p_\perp/u = -1.04$",
            fontsize=7.6, bbox=dict(boxstyle="round,pad=0.25", fc="#fdfefe", ec="#555555", lw=0.8))
    ax.text(0.08, 1.45, r"Pinched $S^1\times S^2$ ($\lambda_B\to 0$):" + "\n" + r"$p_\parallel/u \to +3,\ p_\perp/u \to -1$",
            fontsize=7.4, color="#922b21", bbox=dict(boxstyle="round,pad=0.25", fc="#fdedec", ec="#c0392b", lw=0.75))
    ax.set_xlabel(r"Berger-sphere squashing ratio $\lambda_B = a_3 / a_\perp$")
    ax.set_ylabel(r"Anisotropic equation-of-state ratio $p_i / u_{\mathrm{Cas}}$")
    ax.set_title(r"A. Exact Berger-Sphere Topological Casimir Pressures ($\Delta T^\mu{}_\mu \equiv 0$)")
    ax.set_xlim(0.0, 3.25)
    ax.set_ylim(-2.5, 6.2)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7.0)

    ax = axes[1]
    tr = d1["sample_trajectories"]
    curves = [
        ("L40_xring_lam1", "#c0392b", "-", r"$L(40,1)$ round $x_b = x_{\mathrm{ring}} = 230.9$ ($\tau_c = 87.7$)"),
        ("D20_xring_lam1", "#e67e22", "--", r"$S^3/D_{20}^*$ round $x_b = x_{\mathrm{ring}} = 230.6$ ($\tau_c = 87.5$)"),
        ("D30_xring_lam1", "#8e44ad", "-.", r"$S^3/D_{30}^*$ round $x_b = x_{\mathrm{ring}} = 524.0$ ($\tau_c = 199.9$)"),
        ("L40_5xring_lam1", "#1f618d", "-", r"$L(40,1)$ round $x_b = 5 x_{\mathrm{ring}} = 1155$ ($\tau_c = 1603.5$)"),
    ]
    for key, col, ls, lbl in curves:
        t_arr = np.array(tr[key]["tau"])
        l_arr = np.array(tr[key]["lam_B"])
        ax.plot(t_arr, l_arr, color=col, ls=ls, lw=1.9, label=lbl)
    ax.axhline(
        math.sqrt(2.0 / 3.0), color="#16a085", ls=":", lw=1.2,
        label=r"Static saddle-node $\lambda_B^* = \sqrt{2/3} \approx 0.8165$ ($x_{\perp,\min} \approx \frac{9}{2}x_{\mathrm{ring}}$)",
    )
    ax.set_xscale("symlog", linthresh=30.0)
    ax.set_xlabel(r"Dimensionless Starobinsky time $\tau = H_S t$")
    ax.set_ylabel(r"Hopf-fiber squashing ratio $\lambda_B(\tau) = a_3(\tau) / a_\perp(\tau)$")
    ax.set_title(r"B. Universal Runaway Hopf-Fiber Collapse ($a_3 \to 0$) in Bianchi IX + $f(R)$")
    ax.set_xlim(0, 2200)
    ax.set_ylim(0.0, 1.15)
    ax.grid(True, which="both", alpha=0.45)
    ax.legend(loc="lower left", fontsize=7.0)
    ax.text(
        0.03, 0.44,
        "All 16 round-sphere starts collapse\n"
        + "All 144 prolate/shear trials: 0/144 bounded\n"
        + r"(31/31 valid bounces $\to a_3 \to 0$)" + "\n"
        + r"DOP853 vs. Radau: $2.11\times 10^{-11}$",
        transform=ax.transAxes, fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.30", fc="#fdedec", ec="#c0392b", lw=0.85),
    )

    fig.suptitle(
        r"Anisotropic Berger-Sphere Casimir Backreaction Destroys Super-Species KAM Islands on $L(p,1)$ and $S^3/D_m^*$",
        fontsize=10.8, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = out_dir / "fig3_berger_anisotropic_collapse.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"OK: {out}")
    return out


def make_fig4(out_dir: Path) -> Path:
    d2 = json.loads((DATA_DIR / "t3_02_riegert_weyl_c2_spin2.json").read_text(encoding="utf-8"))
    d_r33 = json.loads((DATA_DIR / "r3_03_kam_bianchi9_shear_stability.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), dpi=220)
    fig.patch.set_facecolor("#ffffff")

    ax = axes[0]
    rs = [r for r in d2["pure_variational_riegert_island_sweep"] if r["sign_hdot"] == 1]
    xbs_r = [r["x_b"] for r in rs]
    min_AR = [r["min_A_Rieg"] for r in rs]
    min_cs2 = [max(r["min_cs2"], -2.2) for r in rs]

    r33_s = [r for r in d_r33["kam_island_shear_sweep"] if r["sign_hdot"] == 1]
    xbs_33 = [r["x_b"] for r in r33_s]
    min_A33 = [r["min_A_full"] for r in r33_s]

    ax.plot(xbs_r, min_AR, "o-", color="#1f618d", lw=2.0, ms=5.0,
            label=r"Variational Riegert $\min A_{\mathrm{Rieg}}(\tau) > 0$ on ALL $[7, 60]$")
    ax.plot(xbs_r, min_cs2, "s--", color="#16a085", lw=1.7, ms=4.5,
            label=r"Variational Riegert $\min c_s^2 = \min(K_{\mathrm{Rieg}}/A_{\mathrm{Rieg}})$")
    ax.plot(xbs_33, min_A33, "^:", color="#c0392b", lw=1.6, ms=5.0,
            label=r"Non-variational local ${}^{(3)}H$ $\min A_{\mathrm{eff}}$ (singular at $7.79$)")
    ax.axhline(0.0, color="#222222", lw=1.0)
    ax.axvspan(15.0, 35.0, color="#e8f8f5", alpha=0.65, label=r"Core $[15, 35]$: $A_{\mathrm{Rieg}}>0$ & $c_s^2>0$ always")
    ax.axvline(21.9168, color="#1f618d", ls=":", lw=1.0)
    ax.text(22.6, 1.07, r"$x_{\mathrm{ring}}=21.92$", fontsize=7.6, color="#154360", fontweight="bold")
    ax.text(
        0.55, 0.45,
        "Symplectic Wronskian drift:\n"
        + r"Local ${}^{(3)}H$: $61.2\%$ (non-var.)" + "\n"
        + r"Riegert: $5.67\times 10^{-13}$ ($\tau=6000$)",
        transform=ax.transAxes, fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.28", fc="#fdfefe", ec="#1f618d", lw=0.85),
    )
    ax.set_xlabel(r"Bounce scale factor $x_b = H_S a_b$ across $S^3/I^*$ KAM island")
    ax.set_ylabel(r"Minimum kinetic / sound-speed coefficient")
    ax.set_title(r"A. Variational Covariant Riegert ($\beta_{\mathrm{eff}}=0$) vs. Local ${}^{(3)}H$")
    ax.set_xlim(5.5, 66.5)
    ax.set_ylim(-2.15, 1.25)
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=6.8)

    ax = axes[1]
    scan = d2["weyl_c2_xring_beta0_scan"]
    tongue = d2["weyl_c2_2to1_tongue_tau6000"]
    long3k = d2["weyl_c2_long_tau3000"]
    override_lyap = {
        0.5: abs(long3k["beta0_0.5"]["effective_lyapunov_rate"]),
        1.0: abs(long3k["beta0_1.0"]["effective_lyapunov_rate"]),
        5.0: abs(long3k["beta0_5.0"]["effective_lyapunov_rate"]),
    }
    t_931 = next(r for r in tongue if abs(r["beta_0"] - 9.0 / 31.0) < 1e-4)
    override_lyap[round(9.0 / 31.0, 5)] = abs(t_931["effective_lyapunov_rate"])

    pos_b, pos_l = [], []
    for r in scan:
        if r["beta_0"] > 0:
            b = r["beta_0"]
            lv = override_lyap.get(b, override_lyap.get(round(b, 5), abs(r["effective_lyapunov_rate"])))
            pos_b.append(b)
            pos_l.append(max(lv, 2e-6))
    neg_b = [r["beta_0"] for r in scan if r["beta_0"] < 0]
    neg_l = [r["effective_lyapunov_rate"] for r in scan if r["beta_0"] < 0]

    tb = [r["beta_0"] for r in tongue]

    ax.axvspan(-1.3, 0.0, color="#fdedec", alpha=0.65, label=r"Regime I ($\beta_0 < 0$): tachyonic spin-2 blowup")
    ax.axvspan(0.0, 0.106, color="#fef9e7", alpha=0.75, label=r"Regime II ($x_b=15$ RG zero-crossing $\beta_{\mathrm{eff}}\to 0$)")
    ax.plot(neg_b, neg_l, "D-", color="#c0392b", lw=1.8, ms=6.0, label=r"Tachyonic rate $\gamma \approx 1/(2\sqrt{|\beta_0|})$")
    ax.plot(pos_b, pos_l, "o-", color="#1f618d", lw=1.7, ms=4.8, label=r"Stelle window ($\beta_0 > 0$, $\tau=600$–$6000$): $|\lambda_{\mathrm{eff}}|$")
    ax.plot([0.2890], [1.593e-3], "s", color="#8e44ad", ms=6.5, zorder=7,
            label=r"2:1 Stelle–scalaron tongue peak ($\beta_0 \approx 0.2890$, $1995\times$)")

    ax.set_xscale("symlog", linthresh=0.08)
    ax.set_xticks([-1.0, -0.3, 0.0, 0.1, 0.3, 1.0, 5.0, 20.0])
    ax.set_xticklabels(["$-1$", "$-0.3$", "$0$", "$0.1$", "$0.3$", "$1$", "$5$", "$20$"])
    ax.set_yscale("log")
    ax.set_xlabel(r"Renormalized Weyl coupling $\beta_0 = \beta_{\mathrm{eff}}(x_{\mathrm{ring}})$ ($c_{\mathrm{anom}} = 9/31$)")
    ax.set_ylabel(r"Effective spin-2 growth rate $|\lambda_{\mathrm{eff}}|$ (units of $H_S$)")
    ax.set_title(r"B. 4th-Order Ostrogradsky Weyl $C^2$ Spin-2 Stability on $x_{\mathrm{ring}}$")
    ax.set_xlim(-1.25, 25.0)
    ax.set_ylim(1.5e-6, 4.8)
    ax.grid(True, which="both", alpha=0.45)
    ax.legend(loc="upper right", fontsize=6.5)

    ax_in = ax.inset_axes([0.07, 0.13, 0.39, 0.36])
    ax_in.set_facecolor("#faf5fc")
    ax_in.plot(tb, [r["effective_lyapunov_rate"] * 1e3 for r in tongue], "s-", color="#8e44ad", lw=1.5, ms=3.8)
    ax_in.axhline(0.0, color="#555555", lw=0.7, ls="--")
    ax_in.set_title(r"2:1 Tongue ($\tau=6000$)", fontsize=7.5, fontweight="bold", color="#6c3483", pad=2)
    ax_in.set_xlabel(r"$\beta_0$", fontsize=7.0, labelpad=1)
    ax_in.set_ylabel(r"$10^3 \lambda_{\mathrm{eff}}$", fontsize=7.0, labelpad=1)
    ax_in.tick_params(labelsize=6.4)
    ax_in.set_xlim(0.2878, 0.2913)
    ax_in.set_ylim(-0.15, 1.85)
    ax_in.grid(True, alpha=0.5)
    ax.annotate("", xy=(0.2890, 1.593e-3), xytext=(0.015, 1.5e-3),
                arrowprops=dict(arrowstyle="->", color="#6c3483", lw=1.1, ls="--"))

    fig.suptitle(
        r"Covariant Riegert Anomaly Action and $O(1)$ Weyl-Squared ($c_{\mathrm{anom}}/a_{\mathrm{anom}} = 18/31$) Spin-2 Stability on $S^3/I^*$",
        fontsize=10.8, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = out_dir / "fig4_riegert_weyl_c2_spin2.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"OK: {out}")
    return out


def build_track3_figures(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        make_fig1(out_dir),
        make_fig2(out_dir),
        make_fig3(out_dir),
        make_fig4(out_dir),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Track III publication figures.")
    ap.add_argument("--out-dir", type=Path, default=TRACK_DIR.parent / "generated_figures" / "track3")
    args = ap.parse_args()
    build_track3_figures(args.out_dir)


if __name__ == "__main__":
    main()
