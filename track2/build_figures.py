#!/usr/bin/env python3
"""Generate all 3 publication-clean figures for Track II (SU(2) Yang-Mills-Higgs & Galois rho_2).

Outputs into --out-dir (default: ../generated_figures/track2):
  - fig1_p02_inhomogeneous_spectrum.png
  - fig2_p03_galois_rho2_sphaleron.png
  - fig3_r3_01_twisted_casimir_eta.png
"""

from __future__ import annotations

import argparse
import json
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


def build_track2_figures(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    # Fig 1: Inhomogeneous SU(2)_L-Higgs Spectrum on S^3 vs S^3/I*
    fig1 = out_dir / "fig1_p02_inhomogeneous_spectrum.png"
    _run_patched_script(
        TRACK_DIR / "p02_04_plot_summary.py",
        [
            (
                'r"Problem 2: Inhomogeneous $SU(2)_L$–Higgs Perturbations Around Cosmological Bounces on $S^3$ vs. $S^3/I^*$"',
                'r"Inhomogeneous $SU(2)_L$–Higgs Perturbations Around Cosmological Bounces on $S^3$ vs. $S^3/I^*$"',
            ),
            ('out_png = os.path.join(OUT_DIR, "p02_summary_figure.png")', f'out_png = r"{fig1}"'),
        ],
        fig1,
    )
    outputs.append(fig1)

    # Fig 2: Galois Flat Connection rho_2, Sphaleron S_02, & Coupled SU(2)_L-Higgs Phase Diagram
    fig2 = out_dir / "fig2_p03_galois_rho2_sphaleron.png"
    _run_patched_script(
        TRACK_DIR / "generate_figures.py",
        [
            (
                'r"Open Problem 3: Galois-Conjugate Flat Connection $\\rho_2$ ($\\mathrm{CS}=49/120$), Sphaleron Ridge $S_{02}$, & Coupled $SU(2)_L$–Higgs Phase Diagram on $S^3/I^*$"',
                'r"Galois-Conjugate Flat Connection $\\rho_2$ ($\\mathrm{CS}=49/120$), Sphaleron Ridge $S_{02}$, & Coupled $SU(2)_L$–Higgs Phase Diagram on $S^3/I^*$"',
            ),
            (r'Full-3D $N_{12}$: $1.97341$', r'Full-3D $N_{\max}=12$: $1.97341$'),
            ('out_png = os.path.join(OUT_DIR, "p03_galois_rho2_summary.png")', f'out_png = r"{fig2}"'),
        ],
        fig2,
    )
    outputs.append(fig2)

    # Fig 3: Ad(rho_0, rho_1, rho_2)-Twisted Chiral Curl Spectrum, APS Spectral Flow & 1-Loop YM Casimir
    fig3 = out_dir / "fig3_r3_01_twisted_casimir_eta.png"
    data = json.loads((DATA_DIR / "r3_01_twisted_casimir_eta.json").read_text(encoding="utf-8"))

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

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), dpi=220)
    fig.patch.set_facecolor("#ffffff")

    ax = axes[0]
    vac_specs = [
        ("rho_0", r"$\operatorname{Ad}(\rho_0)$ ($\mathrm{CS}=0$)", "#1f618d", -0.35),
        ("rho_1", r"$\operatorname{Ad}(\rho_1)$ ($\mathrm{CS}=1/120$)", "#16a085", 0.0),
        ("rho_2", r"$\operatorname{Ad}(\rho_2)$ ($\mathrm{CS}=49/120$)", "#c0392b", +0.35),
    ]
    for key, label, col, dx in vac_specs:
        modes = [m for m in data["vacua"][key]["lowest_12_curl_modes"] if abs(m["lambda"]) <= 22]
        xs = [m["lambda"] + dx for m in modes]
        ys = [m["mult"] for m in modes]
        ax.bar(xs, ys, width=0.55, color=col, alpha=0.88, label=label)

    ax.axvline(0.0, color="#333333", lw=1.0, ls="--")
    ax.set_xlabel(r"Signed chiral curl eigenvalue $\lambda$ ($*d\,\omega = \lambda\,\omega$)")
    ax.set_ylabel(r"Mode multiplicity $d_{\pm}(\lambda)$")
    ax.set_title(r"A. Lowest $\operatorname{Ad}(\rho_i)$-Twisted Chiral Curl Modes ($|\lambda| \leq 22$)")
    ax.set_xlim(-24.5, 24.5)
    ax.set_ylim(0, 82)
    ax.grid(True, axis="y")
    ax.legend(loc="upper center", borderpad=0.35)
    ax.text(
        -10.2, 41,
        "APS spectral flow\n"
        r"($h^0(\rho_0)=3$, $h^0(\rho_{1,2})=0$):" + "\n"
        r"$\mathrm{SF}(\rho_0\!\to\!\rho_1) = 1$" + "\n"
        r"$\mathrm{SF}(\rho_0\!\to\!\rho_2) = 5$",
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.35", fc="#f8f9f9", ec="#7f8c8d", lw=0.85),
    )

    ax = axes[1]
    x = np.arange(3)
    w = 0.25
    e_ym = [data["vacua"][k]["E_Cas_YM_float"] for k in ("rho_0", "rho_1", "rho_2")]
    e_sc = [data["vacua"][k]["E_Cas_scal_Ad_float"] for k in ("rho_0", "rho_1", "rho_2")]
    eta_v = [data["vacua"][k]["eta_curl_0_float"] for k in ("rho_0", "rho_1", "rho_2")]

    b1 = ax.bar(x - w, e_ym, width=w, color="#1f618d", label=r"Pure YM $E_{\mathrm{Cas}}^{\mathrm{YM}}$ [$\hbar c/a$]")
    ax.bar(x, e_sc, width=w, color="#16a085", label=r"Adjoint scalar $E_{\mathrm{Cas}}^{\mathrm{scal}}$ [$\hbar c/a$]")
    b3 = ax.bar(x + w, eta_v, width=w, color="#d35400", label=r"APS $\eta_{\mathrm{curl}}(0;\operatorname{Ad}\rho_i)$")

    ax.axhline(0.0, color="#333333", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([
        r"$\rho_0$ ($\mathrm{CS}=0$)" + "\n" + r"$x_{\mathrm{ring}}=21.92$",
        r"$\rho_1$ ($\mathrm{CS}=1/120$)" + "\n" + r"$x_{\mathrm{ring}}=21.36$",
        r"$\rho_2$ ($\mathrm{CS}=49/120$)" + "\nNo KAM ring",
    ], fontsize=8.5)
    ax.set_ylabel("Exact rational spectral invariant")
    ax.set_title(r"B. 1-Loop Pure-YM Hosotani Ordering & Sign Flip at $\rho_2$")
    ax.set_ylim(-6.9, 1.8)
    ax.grid(True, axis="y")

    ym_lbls = [r"$-\frac{349}{225}$", r"$-\frac{221}{150}$", r"$+\frac{79}{150}$"]
    eta_lbls = [r"$-\frac{361}{60}$", r"$-\frac{23}{20}$", r"$+\frac{9}{20}$"]
    for rect, lbl in zip(b1, ym_lbls):
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        dy = 0.08 if h >= 0 else -0.10
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + dy, lbl, ha="center", va=va, fontsize=8.0, color="#1f618d", fontweight="bold")
    for rect, lbl in zip(b3, eta_lbls):
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        dy = 0.08 if h >= 0 else -0.10
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + dy, lbl, ha="center", va=va, fontsize=8.0, color="#a04000", fontweight="bold")

    ax.legend(loc="lower right", borderpad=0.35)
    ax.text(
        1.02, -3.85,
        r"Exact jump ($g_{\mathrm{YM}}(1/a)\ll 1$):" + "\n"
        r"$E_{\mathrm{Cas}}^{\mathrm{YM}}(\rho_2)-E_{\mathrm{Cas}}^{\mathrm{YM}}(\rho_1) = +2$" + "\n"
        r"$E_{\mathrm{Cas}}^{\mathrm{YM}}(\rho_1)+E_{\mathrm{Cas}}^{\mathrm{YM}}(\rho_2) = -\frac{71}{75}$" + "\n"
        r"$\Delta E_{\mathrm{Cas}}(\operatorname{Ad}\rho_i) \equiv 0$ (all 3)",
        fontsize=7.9,
        bbox=dict(boxstyle="round,pad=0.35", fc="#fef9e7", ec="#d35400", lw=0.85),
    )

    fig.suptitle(
        r"$\operatorname{Ad}(\rho_0, \rho_1, \rho_2)$-Twisted Chiral Curl Spectrum, APS Spectral Flow, & 1-Loop Pure-YM Casimir Lifting on $S^3/I^*$",
        fontsize=11.0, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig3, dpi=220)
    plt.close(fig)
    print(f"OK: {fig3}")
    outputs.append(fig3)

    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Track II publication figures.")
    ap.add_argument("--out-dir", type=Path, default=TRACK_DIR.parent / "generated_figures" / "track2")
    args = ap.parse_args()
    build_track2_figures(args.out_dir)


if __name__ == "__main__":
    main()
