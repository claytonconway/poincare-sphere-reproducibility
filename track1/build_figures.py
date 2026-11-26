#!/usr/bin/env python3
"""Generate all 5 publication-clean figures for Track I (Classical GR, Bianchi IX, Thin Shells).

Outputs into --out-dir (default: ../generated_figures/track1):
  - fig1_einstein_maxwell_lambda.png
  - fig2_tilted_bianchi9.png
  - fig3_su2_tilted_bianchi9.png
  - fig4_israel_darmois.png
  - fig5_faceted_thin_shells.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRACK_DIR = Path(__file__).resolve().parent
DATA_DIR = TRACK_DIR / "data"


def _run_patched_script(src_path: Path, replacements: list[tuple[str, str]], out_png: Path) -> None:
    code = src_path.read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in code:
            raise RuntimeError(f"Expected target snippet not found in {src_path.name}: {old!r}")
        code = code.replace(old, new)
    ns = {"__file__": str(src_path), "__name__": "__main__"}
    old_cwd = os.getcwd()
    old_sys_path = list(sys.path)
    try:
        os.chdir(TRACK_DIR)
        if str(TRACK_DIR) not in sys.path:
            sys.path.insert(0, str(TRACK_DIR))
        exec(compile(code, str(src_path), "exec"), ns)
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path
    if not (out_png.exists() and out_png.stat().st_size > 10000):
        raise RuntimeError(f"Failed to write figure {out_png}")
    print(f"OK: {out_png}")


def build_track1_figures(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    # Fig 1: Exact Closed-Form Einstein-Maxwell-Lambda on S^3/I*
    fig1 = out_dir / "fig1_einstein_maxwell_lambda.png"
    _run_patched_script(
        TRACK_DIR / "explore_einstein_maxwell_lambda.py",
        [
            ('"Page 3 parabolic bounce (w = -2/3)"', 'r"$w = -2/3$ parabolic ansatz"'),
            ('"Einstein static ceiling γ_c = 1/4"', 'r"Einstein-static ceiling $\\gamma_c = 1/4$"'),
            ('"γ_c = 1/4 (Einstein static y = 1/√2)"', 'r"$\\gamma_c = 1/4$ (Einstein static $y = 1/\\sqrt{2}$)"'),
            ('"γ_c = 1/4 (barrier vanishes at y = 1/√2)"', 'r"$\\gamma_c = 1/4$ (barrier vanishes at $y = 1/\\sqrt{2}$)"'),
            ('"Dimensionless Cosmic Time H_Λ t"', 'r"Dimensionless Cosmic Time $H_\\Lambda t$"'),
            ('"Dimensionless Scale Factor y = a / a_dS"', 'r"Dimensionless Scale Factor $y = a / a_{\\mathrm{dS}}$"'),
            ('"3. Exact Bounce Barrier & Radiation Ceiling γ_c = 1/4"', 'r"3. Exact Bounce Barrier & Radiation Ceiling $\\gamma_c = 1/4$"'),
            ('"Bounce Radiation-to-Λ Ratio  u_rad / u_Λ"', 'r"Bounce Radiation-to-$\\Lambda$ Ratio $u_{\\mathrm{rad}} / u_\\Lambda$"'),
            ('"u_rad / u_Λ at bounce"', 'r"$u_{\\mathrm{rad}} / u_\\Lambda$ at bounce"'),
            ('"Classically forbidden (V_eff > 0)"', 'r"Classically forbidden ($V_{\\mathrm{eff}} > 0$)"'),
            ('"4. Exponential Convergence of Resonances γ_k → 1/4"', 'r"4. Exponential Convergence of Resonances $\\gamma_k \\to 1/4$"'),
            ('"Margin Below Einstein-Static Ceiling  (1/4 − γ_k)"', 'r"Margin Below Einstein-Static Ceiling ($1/4 - \\gamma_k$)"'),
            ('"Ceiling margin  1/4 − γ_k"', 'r"Ceiling margin $1/4 - \\gamma_k$"'),
            ('out_png = os.path.join(RESULTS_DIR, "einstein_maxwell_lambda.png")', f'out_png = r"{fig1}"'),
            ('out_json = save_result("einstein_maxwell_lambda", payload)', 'out_json = "(skipped)"'),
        ],
        fig1,
    )
    outputs.append(fig1)

    # Fig 2: Tilted Rotating Triaxial Bianchi IX
    fig2 = out_dir / "fig2_tilted_bianchi9.png"
    _run_patched_script(
        TRACK_DIR / "plot_p01_summary.py",
        [
            ('label="Model I: Exact Page 1 Maxwell"', 'label="Model I: Source-Free Maxwell"'),
            ('label=r"M-I Untuned Page 1 Null"', 'label=r"Model I: Untuned Null Beltrami"'),
            (
                'r"Problem 1: Tilted Rotating Triaxial Bianchi IX Einstein–Maxwell–$\\Lambda$ Cosmology on $S^3/I^*$"',
                'r"Tilted Rotating Triaxial Bianchi IX Einstein–Maxwell–$\\Lambda$ Cosmology on $S^3/I^*$"',
            ),
            ('out_png = os.path.join(BASE_DIR, "p01_tilted_null_bianchi9_summary.png")', f'out_png = r"{fig2}"'),
        ],
        fig2,
    )
    outputs.append(fig2)

    # Fig 4: Spherical Israel-Darmois Thin Shells
    fig4 = out_dir / "fig4_israel_darmois.png"
    p04_code = (TRACK_DIR / "killtests_and_benchmarks.py").read_text(encoding="utf-8")
    for old, new in [
        (
            'label=r"Page 7 full $\\mathcal{E}_{\\min}=Mc^2$: $\\pi/80 = 0.03927$ ($1.331\\times$ above)"',
            'label=r"Full-volume ansatz $\\mathcal{E}_{\\min}=Mc^2$: $\\pi/80 = 0.03927$ ($1.331\\times$ above)"',
        ),
        (
            'label=r"Page 7 kept-vol: $\\frac{\\pi}{80}(1-f_{\\rm exc})$ (crosses at $0.866\\,\\chi_{\\rm inj}$)"',
            'label=r"Kept-volume ansatz: $\\frac{\\pi}{80}(1-f_{\\rm exc})$ (crosses at $0.866\\,\\chi_{\\rm inj}$)"',
        ),
        (
            'ax.set_title(r"(a) Bounce-Epoch WEC Ceiling vs. Page 7 ($\\mathcal{E}_{\\min}=Mc^2$)")',
            'ax.set_title(r"(a) Bounce-Epoch WEC Ceiling vs. Full-Volume Ansatz ($\\pi/80$)")',
        ),
        ('fig_path = OUT_DIR / "israel_darmois_poincare.png"', f'fig_path = Path(r"{fig4}")'),
    ]:
        if old not in p04_code:
            raise RuntimeError(f"Expected snippet missing in killtests_and_benchmarks.py: {old!r}")
        p04_code = p04_code.replace(old, new)
    old_sys_path = list(sys.path)
    try:
        if str(TRACK_DIR) not in sys.path:
            sys.path.insert(0, str(TRACK_DIR))
        ns_p04 = {"__file__": str(TRACK_DIR / "killtests_and_benchmarks.py"), "__name__": "p04_pub"}
        exec(compile(p04_code, "p04_pub", "exec"), ns_p04)
        bgg_res = ns_p04["bgg_benchmark"]()
        ns_p04["make_diagnostic_figure"](bgg_res, None)
    finally:
        sys.path[:] = old_sys_path
    if not (fig4.exists() and fig4.stat().st_size > 10000):
        raise RuntimeError(f"Failed to write {fig4}")
    print(f"OK: {fig4}")
    outputs.append(fig4)

    # Shared plotting style for Figs 3 & 5
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "legend.frameon": True,
        "legend.facecolor": "#ffffff",
        "legend.edgecolor": "#cccccc",
        "legend.fontsize": 8.0,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
    })

    d4 = json.loads((DATA_DIR / "r3_04_faceted_shell.json").read_text(encoding="utf-8"))
    d5 = json.loads((DATA_DIR / "r3_05_tilted_su2_bianchi9.json").read_text(encoding="utf-8"))

    # Fig 3: Tilted Homogeneous SU(2)_L Yang-Mills in Rotating Triaxial Bianchi IX
    fig3 = out_dir / "fig3_su2_tilted_bianchi9.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), dpi=220)
    fig.patch.set_facecolor("#ffffff")
    ax = axes[0]
    phi_grid = np.linspace(0.0, 0.5 * math.pi, 300)
    for rv, col in zip([1.0, 1.3, 1.7], ["#1f618d", "#d35400", "#8e44ad"]):
        q1_e = 2.0 * math.sqrt(1.0 + rv * rv) * np.cos(phi_grid)
        q2_e = 2.0 * math.sqrt(1.0 + 1.0 / (rv * rv)) * np.sin(phi_grid)
        ax.plot(q1_e, q2_e, color=col, lw=1.9, label=rf"Gauss ellipsoid $a_1/a_2 = {rv:.1f}$")
    ax.scatter([2.0], [2.0], color="#c0392b", s=65, zorder=5,
               label=r"Irreducible flat vacuum $\rho_1$ ($\|\mathbf{A}_1\|=\|\mathbf{A}_2\|=2$)")
    ax.scatter([0.0], [0.0], color="#2c3e50", marker="x", s=60, zorder=5,
               label=r"Trivial $\rho_0$: small-$|A| \Rightarrow \det A_\perp=0$ (Abelian)")
    ax.set_xlabel(r"$\|\mathbf{A}_1\|$ (color-1/2 transverse norm)")
    ax.set_ylabel(r"$\|\mathbf{A}_2\|$ (color-1/2 transverse norm)")
    ax.set_title(r"A. Compact Pure-Null Gauss Ellipsoid ($\vec{\mathcal{G}}=0$)")
    ax.set_xlim(-0.2, 4.3)
    ax.set_ylim(-0.2, 3.35)
    ax.grid(True)
    ax.text(
        0.15, 0.42,
        "Full-ellipsoid paired pure-null\n($|S_3/u|_0=1$, $u_0\\in[0.514,0.75]$):\n"
        "Abelian: 0 / 400 bounced\nSU(2)_L: 0 / 400 bounced\n(400/400 verified collapses)",
        fontsize=7.9, bbox=dict(boxstyle="round,pad=0.35", fc="#fdf2e9", ec="#d35400", lw=0.9),
    )
    ax.legend(loc="upper right", fontsize=7.5, borderpad=0.35)

    ax = axes[1]
    pt = d5["paired_partially_tilted_coupling_sweep"]
    g_labels = [f"g={pt[k]['g_YM']:g}\n(u3/u={pt[k]['median_init_u3_frac']:.2f})" for k in pt]
    x_idx = np.arange(len(g_labels))
    w = 0.26
    y_su2 = [pt[k]["n_bounced_su2"] for k in pt]
    y_u1s3 = [pt[k]["n_bounced_match_S3_only_U1"] for k in pt]
    y_u1x2 = [pt[k]["n_bounced_match_S3_u3_T33_U1xU1"] for k in pt]
    ax.bar(x_idx - w, y_su2, width=w, color="#1f618d", label=r"Non-Abelian $SU(2)_L$")
    ax.bar(x_idx, y_u1x2, width=w, color="#16a085", label=r"$T_{33}$-matched $U(1)_A\times U(1)_B$ (matched $u, S_3, T_{33}$)")
    ax.bar(x_idx + w, y_u1s3, width=w, color="#bdc3c7", edgecolor="#555555", lw=0.7, label=r"$S_3$-only $U(1)$ ($u_3=0$, mismatched $T_{33}$)")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(g_labels, fontsize=7.9)
    ax.set_ylabel(r"Contracting bounces / 150 ($u_0 \in [0.514, 0.75]$)")
    ax.set_ylim(0, 12.2)
    ax.set_title(r"B. Partially-Tilted $SU(2)_L$ vs. Matched Abelian Controls")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=7.5, borderpad=0.35)
    ax.text(
        -0.22, 5.2,
        r"Full turning-point $\mathcal{I}^-\leftrightarrow\mathcal{I}^+$:" + "\n"
        + r"50 bounces $> 0.5132$ ($\max\,u_0 = 0.7684 > 0.750$)",
        fontsize=7.8, bbox=dict(boxstyle="round,pad=0.35", fc="#eafaf1", ec="#16a085", lw=0.9),
    )
    fig.suptitle(
        r"Tilted Homogeneous $SU(2)_L$ Yang–Mills in Rotating Triaxial Bianchi IX on $S^3/I^*$",
        fontsize=11.2, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig3, dpi=220)
    plt.close(fig)
    print(f"OK: {fig3}")
    outputs.append(fig3)

    # Fig 5: I*-Equivariant l=6 Faceted Thin Shells
    fig5 = out_dir / "fig5_faceted_thin_shells.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), dpi=220)
    fig.patch.set_facecolor("#ffffff")
    ax = axes[0]
    sw = d4["linearized_O_eps_mass_ceilings"]["sweep"]
    eps_lin = [r["eps_rad"] * 1e3 for r in sw]
    wec_opt = [r["mu_over_yplus_WEC_opt_multipole"] for r in sw]
    dec_rw = [r["mu_over_yplus_DEC_RW_P62"] for r in sw]
    wec_c0 = [r["mu_over_yplus_WEC_pure_Schw"] for r in sw]
    nec_es = [r["mu_over_yplus_StaticNEC_gamma_quarter_RW_P62"] for r in sw]
    sw_nl = d4["nonlinear_finite_amplitude"]["sweep"]
    eps_nl = [r["eps_rad"] * 1e3 for r in sw_nl]
    nl_c0 = [r["nonlinear_pure_Schw_WEC_mu_over_yplus"] for r in sw_nl]

    ax.axvspan(0.0, 3.6, color="#e8f4f0", alpha=0.7, label=r"Controlled $O(\epsilon)$ ($\epsilon_{\mathrm{eff}} \leq 0.5$)")
    ax.axhline(math.pi / 80.0, color="#c0392b", ls="--", lw=1.6, label=r"Full-volume ansatz $\pi/80 = 0.03927$")
    ax.axhline(
        d4["geometry_and_Y6"]["dirichlet_dodecahedron"]["sin3_chi_eq"], color="#7f8c8d", ls=":", lw=1.3,
        label=r"100% $S^3/I^*$ vol. $\sin^3\chi_{\mathrm{eq}} = 0.03790$",
    )
    ax.plot(eps_lin, wec_opt, "-o", ms=3.5, color="#1f618d", lw=1.7, label=r"Lin. WEC (opt. $l=6$ multipole)")
    ax.plot(eps_lin, dec_rw, "-s", ms=3.5, color="#2e86c1", lw=1.5, label=r"Lin. WEC+DEC (full-$S^2$, RW $P_6^2$)")
    ax.plot(eps_lin, wec_c0, "--^", ms=3.5, color="#d35400", lw=1.3, label=r"Lin. WEC (pure Schw. $c_+=0$)")
    ax.plot(eps_nl, nl_c0, "-D", ms=3.8, color="#8e44ad", lw=1.8, label=r"Exact NL WEC (pure Schw. $c_+=0$)")
    ax.plot(eps_lin, nec_es, "-.", color="#16a085", lw=1.4, label=r"Static NEC ($\gamma=1/4$, full-$S^2$)")
    ax.set_xlabel(r"Faceting amplitude $\epsilon$ [$10^{-3}$ rad]")
    ax.set_ylabel(r"Bounce mass ceiling $\mu / y_+$")
    ax.set_title(r"A. $l=6$ Faceted Shell Ceilings vs. Full-Volume Ansatz $\pi/80$")
    ax.set_xlim(-0.4, 19.0)
    ax.set_ylim(-0.002, 0.044)
    ax.grid(True)
    ax.legend(loc="lower left", ncol=2, fontsize=6.6, borderpad=0.35, columnspacing=0.65, handlelength=1.6)

    ax = axes[1]
    mH_nl = [r["two_mH_minus_over_a0"] for r in sw_nl]
    max_Hm = [r["max_H_minus"] for r in sw_nl]
    ax.axhline(0.0, color="#333333", lw=0.9)
    ax.axhline(math.sin(math.pi / 10.0) ** 3, color="#2e86c1", ls="--", lw=1.2, label=r"Spherical $\sin^3(\pi/10) = 0.02951$")
    l1 = ax.plot(eps_nl, mH_nl, "-o", ms=4.5, color="#1f618d", lw=2.0, label=r"Interior Hawking mass $2m_H^- / a_0$")
    ax.axvline(
        d4["nonlinear_finite_amplitude"]["hawking_mass_peak"]["eps_zero_crossing_rad"] * 1e3,
        color="#c0392b", ls=":", lw=1.3, label=r"$m_H^- = 0$ at $\epsilon = 6.67\times 10^{-3}$ rad",
    )
    ax.set_xlabel(r"Faceting amplitude $\epsilon$ [$10^{-3}$ rad]")
    ax.set_ylabel(r"Interior Hawking mass $2 m_H^-(\mathcal{S}) / a_0$", color="#1f618d")
    ax.tick_params(axis="y", labelcolor="#1f618d")
    ax.set_ylim(-0.22, 0.055)
    ax.grid(True)
    ax2 = ax.twinx()
    l2 = ax2.plot(eps_nl, max_Hm, "--s", ms=3.8, color="#c0392b", lw=1.6, label=r"$\max_{S^2} H_-$ (face convexity flip)")
    ax2.set_ylabel(r"Max interior mean curvature $\max H_-$ [$a_0^{-1}$]", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_ylim(-8.0, 11.0)
    lines_b = l1 + l2 + [ax.lines[1], ax.lines[3]]
    ax.legend(lines_b, [ln.get_label() for ln in lines_b], loc="center left", bbox_to_anchor=(0.01, 0.57), fontsize=7.4, borderpad=0.35)
    ax.set_title(r"B. Nonlinear $m_H^-$ Collapse & Face Convexity Flip")
    fig.suptitle(
        r"$I^*$-Equivariant $l=6$ ($n=12$) Faceted Thin-Shell Matching on the $t=0$ Bounce Slice of $S^3/I^*$",
        fontsize=11.2, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig5, dpi=220)
    plt.close(fig)
    print(f"OK: {fig5}")
    outputs.append(fig5)

    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Track I publication figures.")
    ap.add_argument("--out-dir", type=Path, default=TRACK_DIR.parent / "generated_figures" / "track1")
    args = ap.parse_args()
    build_track1_figures(args.out_dir)


if __name__ == "__main__":
    main()
