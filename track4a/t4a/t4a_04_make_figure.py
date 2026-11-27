#!/usr/bin/env python3
"""
Track IV-A (Goal 4 — Second-Audit Remediated, Exact M_I* = 26 vs M_S3 = 6):
Publication-Grade 4-Panel Diagnostic Figure for Causal Retarded 1-Loop Nonlocal
Weyl-Squared C ln(-Box/mu^2) C Spin-2 Dynamics on the S^3/I* KAM Breathing Island.

Panels:
  (A) Chiral Topological Pinch-Off Cliff & Full Adiabatic Envelope on S^3/I* (M_I* = 26):
      Static pinch-off beta_pinch(I*) = 0.3836 vs full instantaneous adiabatic envelope
      max_tau beta_pinch_inst(x, A_R, K_R) = 0.4206 (AK1 max = 0.4417), plotting ALL scan points
      beta_0 in [0.289, 0.500] (showing quiescence right above 0.4206), plus clean kappa_mem
      homotopy inset distinguishing positive vs negative growth points.
  (B) Chiral Topological Stabilization of Regime IV: S^3/I* (M_I* = 26) vs Round S^3 (M_S3 = 6):
      30-digit static UHP poles + dynamical KAM Volterra rates across beta_0 in [0.05, 10],
      highlighting the 16.6x lower pinch-off and complete stabilization across beta_0 in [0.44, 6.37]
      with completely unobstructed beta_0 = 0.5 markers for both ghost IC and physical graviton IC.
  (C) Dissipation Positivity Theorem & Maxwell s1*s2 = -1 Chiral Photon Gap:
      Conformal Bach symbol B_2(Omega) = (Omega^2 - 4)(Omega^2 - 16) showing magnetic-Weyl
      C^2 < 0 band (2, 4), round S^3 threshold M_S3 = 6 (+2,-4), S^3/I* chiral gap [6, 26),
      exact S^3/I* threshold M_I* = 26 (+12,-14), and KAM scalaron sidebands {38.2, 43.9}.
  (D) Inhomogeneous S^3/I* TT Graviton Stability Map (Centered O(c_anom) Rates, n = 2..22):
      Exact time-domain 0th-order Mathieu rates |lam_n^(0)|, centered O(c_anom) 1-loop shifts
      Delta_lam_n^(1) (eps=0.01, canceling O(km^2) commutator bias identically; distinct markers
      for damping vs weak anti-damping, zero lines bridging sign flips), bounded 2-photon flux
      P_rad/(2 E_avg), and exact coincident line omega_n = omega_scal/2 == (eps_EFT = 1) at n=19.36.
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
C_ANOM = 9.0 / 31.0
X_RING = math.sqrt(44672.0 / 93.0)


def main():
    with open(os.path.join(OUT_DIR, "t4a_01_kernel_verification.json")) as fh:
        g1 = json.load(fh)
    with open(os.path.join(OUT_DIR, "t4a_03_kam_phase_map.json")) as fh:
        g3 = json.load(fh)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.3,
        "axes.titlesize": 10.1,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.3,
        "legend.fontsize": 7.4,
        "xtick.labelsize": 8.3,
        "ytick.labelsize": 8.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 220,
    })

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.8), constrained_layout=True)
    axA, axB, axC, axD = axes.ravel()

    # =========================================================================
    # PANEL A: S^3/I* Pinch-Off Cliff & Full Adiabatic Envelope (beta_0 in [0.28, 0.515])
    # =========================================================================
    p1 = g3["part1_tongue_homotopy_and_pinchoff_M26"]
    pscan = p1["pinchoff_and_shifted_2to1_scan_M26"]
    env_r = p1["adiabatic_pinchoff_xring_M26"]
    b_pinch = g1["static_xring_horowitz_dispersion"]["pinch_off_thresholds"]["M26_Istar"]["beta_pinch"]
    b_full_max = env_r["beta_pinch_inst_full_max"]
    b_ak1_max = env_r["beta_pinch_inst_AK1_max"]

    r2_m26 = {round(e["beta_0"], 4): e for e in g3["part2_regime4_M26_vs_M6_and_eccentric"]["xring_regime4_M26_vs_M6"]}
    b_dyn = np.array([0.2890] + [e["beta_0"] for e in pscan])
    g_dyn_signed = np.array([r2_m26[0.2890]["M26_Istar_dyn_gamma_HS"]] + [e["gamma_dyn_KAM_HS"] for e in pscan])
    g_dyn_abs = np.maximum(1e-5, np.abs(g_dyn_signed))
    m_pos_dyn = g_dyn_signed > 0.0

    # Static curve from Goal 1 + pinch-off
    st26 = [e for e in g1["static_xring_horowitz_dispersion"]["M26_Istar"]
            if e["beta_0"] >= 0.25 and e["uhp_stelle_gamma_HS"] is not None]
    p_map = {round(e["beta_0"], 4): e for e in pscan}
    b_st = np.array([e["beta_0"] for e in st26] + [0.370, 0.380, b_pinch])
    g_st = np.array([e["uhp_stelle_gamma_HS"] for e in st26] + [p_map[0.37]["gamma_static_xring_HS"], p_map[0.38]["gamma_static_xring_HS"], 1e-5])

    axA.axvspan(b_pinch, b_full_max, color="#F39C12", alpha=0.25,
                label=rf"Adiabatic pinch-off band $({b_pinch:.4f},\,{b_full_max:.4f})$")
    axA.semilogy(b_st, g_st, "--", color="#1F4E79", lw=1.8, label=r"Static $x_{\mathrm{ring}}$ UHP pole ($M_{I^*}=26$)")
    axA.semilogy(b_dyn, g_dyn_abs, "-", color="#C0392B", lw=1.6, alpha=0.75)
    axA.semilogy(b_dyn[m_pos_dyn], g_dyn_abs[m_pos_dyn], "o", color="#C0392B", ms=5.3, zorder=5,
                 label=r"Dynamical KAM Volterra ($\gamma > 0$)")
    axA.semilogy(b_dyn[~m_pos_dyn], g_dyn_abs[~m_pos_dyn], "o", color="#1E8449", mfc="#D5F5E3", mew=1.6, ms=5.6, zorder=6,
                 label=r"Dynamical KAM bounded ($\gamma < 0$, ratio $= 1.00$)")

    axA.axvline(b_pinch, color="#1F4E79", ls=":", lw=1.4)
    axA.axvline(b_full_max, color="#B76E00", ls="-.", lw=1.4)
    axA.axvline(b_ak1_max, color="#B76E00", ls=":", lw=1.0, alpha=0.65)
    axA.axhline(1.0 / 136.88, color="#666666", ls=":", lw=1.1, label=r"$1/T_{\mathrm{macro}} = 7.31\times 10^{-3}\,H_S$")

    axA.annotate(rf"Static pinch-off" + "\n" + rf"$\beta_{{\mathrm{{pinch}}}}(I^*)={b_pinch:.4f}$",
                 xy=(b_pinch, 0.0022), xytext=(0.295, 0.00055),
                 arrowprops=dict(arrowstyle="->", color="#1F4E79", lw=1.0),
                 fontsize=7.5, color="#1F4E79")
    axA.annotate(rf"Full envelope $\max\beta_{{\mathrm{{inst}}}}={b_full_max:.4f}$" + "\n(ratio $=1.00$ for $\\beta_0\\geq 0.44$)",
                 xy=(0.440, 1.47e-3), xytext=(0.450, 0.00090),
                 arrowprops=dict(arrowstyle="->", color="#1E8449", lw=1.0),
                 fontsize=7.2, color="#145A32", fontweight="bold")
    axA.annotate("Quiescent\n" + r"$|\gamma|\leq 5.3\times 10^{-5}$",
                 xy=(0.500, 5.30e-5), xytext=(0.442, 4.2e-5),
                 arrowprops=dict(arrowstyle="->", color="#1E8449", lw=1.0),
                 fontsize=7.5, color="#1E8449", fontweight="bold")

    axA.set_xlabel(r"Renormalized Weyl Coupling $\beta_0 = \beta_{\mathrm{eff}}(x_{\mathrm{ring}})$ ($\mu = H_S$)")
    axA.set_ylabel(r"Growth Rate $|\gamma| / H_S$")
    axA.set_title(r"(A) $S^3/I^*$ Chiral Pinch-Off & Adiabatic Envelope ($M_{I^*}=26$)")
    axA.set_xlim(0.28, 0.515)
    axA.set_ylim(2e-5, 0.65)
    axA.grid(True, which="major", alpha=0.25)
    axA.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=6.9)

    # Clean inset in upper right (ZERO overlap with annotations or vertical lines!)
    axA_in = axA.inset_axes([0.63, 0.63, 0.35, 0.33])
    hom = p1["homotopy_M26_0p2890"]
    kms = np.array([h["kappa_mem"] for h in hom if h["kappa_mem"] > 0])
    gp = np.array([h["gamma_0p2890_HS"] for h in hom if h["kappa_mem"] > 0])
    go = np.array([h["gamma_0p2880_HS"] for h in hom if h["kappa_mem"] > 0])
    axA_in.loglog(kms[gp > 0], gp[gp > 0], "o-", color="#C0392B", lw=1.3, ms=3.5, label=r"$0.289\,(\gamma>0)$")
    if np.any(gp < 0):
        axA_in.loglog(kms[gp < 0], np.abs(gp[gp < 0]), "o", color="#C0392B", mfc="white", mew=1.3, ms=4.0, label=r"$0.289\,(\gamma<0)$")
    axA_in.loglog(kms[go > 0], go[go > 0], "s--", color="#2B5B84", lw=1.2, ms=3.2, label=r"$0.288\,(\gamma>0)$")
    axA_in.axhline(hom[0]["gamma_0p2890_HS"], color="#555555", ls=":", lw=1.0)
    axA_in.set_xlabel(r"$\kappa_{\mathrm{mem}}$ (at $\beta_0=0.289$)", fontsize=7.0)
    axA_in.set_ylabel(r"$|\gamma|/H_S$", fontsize=7.0)
    axA_in.set_title("T3-B tongue homotopy", fontsize=7.5, pad=2)
    axA_in.tick_params(labelsize=6.6)
    axA_in.set_ylim(8e-5, 0.5)
    axA_in.grid(True, which="major", alpha=0.2)
    axA_in.legend(loc="lower right", fontsize=5.8, frameon=True, framealpha=0.92)

    # =========================================================================
    # PANEL B: Regime IV Comparison: S^3/I* (M=26) vs Round S^3 (M=6)
    # (Legend kept strictly left of x=0.36 so beta_0=0.5 markers are 100% unobstructed!)
    # =========================================================================
    st6_all = g1["static_xring_horowitz_dispersion"]["M6_S3"]
    st26_all = g1["static_xring_horowitz_dispersion"]["M26_Istar"]
    b_pinch_s3 = g1["static_xring_horowitz_dispersion"]["pinch_off_thresholds"]["M6_S3"]["beta_pinch"]

    b6_st = np.array([e["beta_0"] for e in st6_all if e["beta_0"] > 0 and e["uhp_stelle_gamma_HS"] is not None])
    g6_st = np.array([e["uhp_stelle_gamma_HS"] for e in st6_all if e["beta_0"] > 0 and e["uhp_stelle_gamma_HS"] is not None])
    b26_st = np.array([e["beta_0"] for e in st26_all if e["beta_0"] > 0 and e["uhp_stelle_gamma_HS"] is not None])
    g26_st = np.array([e["uhp_stelle_gamma_HS"] for e in st26_all if e["beta_0"] > 0 and e["uhp_stelle_gamma_HS"] is not None])

    r2 = [e for e in g3["part2_regime4_M26_vs_M6_and_eccentric"]["xring_regime4_M26_vs_M6"] if e["beta_0"] > 0]
    b_r2 = np.array([e["beta_0"] for e in r2])
    g6_dyn = np.array([abs(e["M6_S3_dyn_gamma_HS"]) for e in r2])
    g26_dyn = np.array([abs(e["M26_Istar_dyn_gamma_HS"]) for e in r2])
    m6_run = np.array([e["M6_S3_dyn_gamma_HS"] > 1e-3 for e in r2])
    m26_run = np.array([e["M26_Istar_dyn_gamma_HS"] > 1e-3 for e in r2])

    p5_phys = g3["part5_commutator_and_phys_ic_audit"]["full_unreduced_M26_phys_ic_regime4"]
    b_phys = np.array([e["beta_0"] for e in p5_phys])
    g_phys = np.array([abs(e["gamma_E_phys_HS"]) for e in p5_phys])

    axB.axvspan(b_full_max, b_pinch_s3, color="#27AE60", alpha=0.14,
                label=r"Chiral-protected ($I^*$ safe, $S^3$ runs away)")
    axB.loglog(b6_st, g6_st, "-", color="#7F8C8D", lw=1.8, label=r"$S^3$ ($M=6$) static UHP pole")
    axB.loglog(b_r2[m6_run], g6_dyn[m6_run], "s", color="#566573", ms=5.8, zorder=5, label=r"$S^3$ ($M=6$) dyn runaway")
    axB.loglog(b_r2[~m6_run], g6_dyn[~m6_run], "s", color="#566573", mfc="white", mew=1.5, ms=5.8, zorder=5)

    axB.loglog(b26_st, g26_st, "-", color="#C0392B", lw=2.0, label=r"$S^3/I^*$ ($M=26$) static UHP")
    axB.loglog(b_r2[m26_run], g26_dyn[m26_run], "o", color="#C0392B", ms=6.2, zorder=6, label=r"$S^3/I^*$ dyn runaway")
    axB.loglog(b_r2[~m26_run], g26_dyn[~m26_run], "o", color="#1E8449", mfc="#D5F5E3", mew=1.7, ms=6.8, zorder=7,
               label=r"$S^3/I^*$ bounded ghost IC")
    axB.loglog(b_phys, g_phys, "^", color="#145A32", ms=6.2, zorder=8,
               label=r"$S^3/I^*$ bounded phys IC")

    axB.axvline(b_pinch, color="#C0392B", ls="--", lw=1.2)
    axB.axvline(b_full_max, color="#1E8449", ls=":", lw=1.1)
    axB.axvline(b_pinch_s3, color="#566573", ls="--", lw=1.2)
    axB.axhline(1.0 / 136.88, color="#666666", ls=":", lw=1.1)

    axB.text(b_pinch * 0.90, 0.0020, rf"$\beta_{{\mathrm{{pinch}}}}(I^*)={b_pinch:.3f}$",
             rotation=90, ha="right", va="bottom", fontsize=7.3, color="#C0392B", fontweight="bold")
    axB.text(b_pinch_s3 * 1.12, 0.00025, rf"$\beta_{{\mathrm{{pinch}}}}(S^3)={b_pinch_s3:.2f}$",
             rotation=90, ha="left", va="bottom", fontsize=7.4, color="#566573", fontweight="bold")
    axB.text(1.55, 0.28, "16.6$\\times$ lower pinch-off:\n$S^3/I^*$ stabilizes Regime IV\nby $>4500\\times$ at $\\beta_0=0.5$",
             fontsize=7.7, color="#145A32", ha="center",
             bbox=dict(boxstyle="round,pad=0.30", fc="#EAFAF1", ec="#1E8449", lw=0.8))

    axB.set_xlabel(r"Renormalized Weyl Coupling $\beta_0 = \beta_{\mathrm{eff}}(x_{\mathrm{ring}})$ ($\mu = H_S$)")
    axB.set_ylabel(r"Exponential Rate $|\gamma| / H_S$")
    axB.set_title(r"(B) Chiral Topological Stabilization of Regime IV: $S^3/I^*$ ($M=26$) vs Round $S^3$ ($M=6$)")
    axB.set_xlim(0.032, 15.0)
    axB.set_ylim(8e-7, 1.4)
    axB.grid(True, which="major", alpha=0.25)
    # Place legend cleanly in the empty lower-left quadrant (strictly left of beta_0 = 0.38!)
    axB.legend(loc="lower left", bbox_to_anchor=(0.01, 0.01), frameon=True, framealpha=0.95, fontsize=6.5, handlelength=1.4)

    # =========================================================================
    # PANEL C: Dissipation Positivity & Maxwell s1*s2=-1 Chiral Photon Gap
    # =========================================================================
    om_c = np.linspace(0.0, 48.0, 1200)
    B2_c = (om_c**2 - 4.0) * (om_c**2 - 16.0)
    axC.plot(om_c, B2_c, color="#1B365D", lw=2.0, label=r"$B_2(\Omega) = (\Omega^2 - 4)(\Omega^2 - 16)$")
    axC.axhline(0.0, color="k", lw=0.8)

    m_neg = (om_c >= 2.0) & (om_c <= 4.0)
    axC.fill_between(om_c[m_neg], B2_c[m_neg], 0.0, color="#E74C3C", alpha=0.35,
                     label=r"Magnetic-Weyl $B_2 < 0$ band $(2,4)$ ($\bar{B}^2 > \bar{E}^2$)")
    axC.axvspan(6.0, 26.0, color="#F5B041", alpha=0.20,
                label=r"$S^3/I^*$ chiral photon gap $[6, 26)$ ($+2..+10$ absent)")
    axC.axvspan(26.0, 48.0, color="#27AE60", alpha=0.14,
                label=r"Exact $S^3/I^*$ 2-photon continuum $\Omega \geq M_{I^*}=26$")

    axC.axvline(math.sqrt(8.0), color="#8E44AD", ls="--", lw=1.3)
    axC.axvline(6.0, color="#566573", ls="-.", lw=1.4)
    axC.axvline(26.0, color="#1E8449", ls="-.", lw=1.7)
    axC.axvline(38.22, color="#D35400", ls=":", lw=1.4)
    axC.axvline(43.88, color="#D35400", ls=":", lw=1.4)
    axC.scatter([math.sqrt(8.0)], [-32.0], color="#8E44AD", s=50, zorder=6)

    axC.set_yscale("symlog", linthresh=50.0, linscale=0.8)
    axC.set_xlim(0.0, 48.0)
    axC.set_ylim(-120.0, 4.0e7)

    axC.annotate(r"$\Omega_0=\sqrt{8}$ ($B_2=-32$)" + "\nNaive $M=0$: $+9.6\\times 10^{-4}H_S$",
                 xy=(math.sqrt(8.0), -32.0), xytext=(5.5, -82.0),
                 arrowprops=dict(arrowstyle="->", color="#8E44AD", lw=1.0),
                 fontsize=7.4, color="#6C3483")
    axC.text(6.5, 1.8e4, r"$M_{S^3}=6$" + "\n$(+2,-4)$", fontsize=7.5, color="#566573", fontweight="bold")
    axC.text(16.0, 3.5e2, "Chiral protection:\nStelle pole $\\Omega_{\\mathrm{Stelle}}$\npinches off below $26$",
             ha="center", fontsize=7.5, color="#935116",
             bbox=dict(boxstyle="round,pad=0.25", fc="#FEF9E7", ec="#D68910", lw=0.7))
    axC.text(26.6, 3.5e4, r"$M_{I^*}=26$" + "\n$(+12,-14)$", fontsize=7.6, color="#145A32", fontweight="bold")
    axC.text(41.0, 2.5e3, r"Scalaron" + "\nsidebands\n" + r"$\Omega_\pm\!\approx\!38.2, 43.9$",
             ha="center", fontsize=7.3, color="#A04000",
             bbox=dict(boxstyle="round,pad=0.22", fc="#FDEDEC", ec="#D35400", lw=0.6))

    axC.set_xlabel(r"Conformal-Time Frequency $\Omega$ (unit $S^3/I^*$)")
    axC.set_ylabel(r"Conformal Bach Symbol $B_2(\Omega)$ (symlog)")
    axC.set_title(r"(C) Dissipation Positivity & Opposite-Chirality ($s_1 s_2 = -1$) Gap $M_{I^*}=26$")
    axC.grid(True, alpha=0.25)
    axC.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=7.0)

    # =========================================================================
    # PANEL D: Inhomogeneous S^3/I* TT Gravitons (Centered O(c_anom) Rates!)
    # =========================================================================
    im = g3["part3_order_reduced_M26_and_inhomogeneous"]["inhomogeneous_tt_gravitons_exact_volterra"]
    im_map = {e["label"]: e for e in im}
    istar_keys = ["Istar_n2_M26", "Istar_n10_M14", "Istar_n14_M26", "Istar_n18_M22", "Istar_n22_M26"]
    s3_keys = ["S3_n2_M6", "S3_only_n3_M7", "S3_only_n4_M8"]

    n_is = np.array([im_map[k]["n_mode"] for k in istar_keys])
    lam0_is = np.array([abs(im_map[k]["lam0_riegert_HS"]) for k in istar_keys])
    dl1_is = np.array([im_map[k]["delta_lam_centered_eps0p01_HS"] for k in istar_keys])
    prad_n = np.array([im_map[k]["n_mode"] for k in istar_keys if im_map[k]["gamma_Prad_bounded_HS"] is not None])
    prad_v = np.array([im_map[k]["gamma_Prad_bounded_HS"] for k in istar_keys if im_map[k]["gamma_Prad_bounded_HS"] is not None])

    n_s3 = np.array([im_map[k]["n_mode"] for k in s3_keys])
    dl1_s3 = np.array([abs(im_map[k]["delta_lam_centered_eps0p01_HS"]) for k in s3_keys])
    prad_s3 = np.array([im_map[k]["gamma_Prad_bounded_HS"] for k in s3_keys])

    axD.axvspan(2.5, 9.5, color="#D5DBDB", alpha=0.35)
    axD.text(6.0, 2.2e-4, "$I^*$ topological\ngap\n($n=3..9$ absent;\n$n=12$ absent)",
             ha="center", va="center", fontsize=7.5, color="#555555")

    axD.semilogy(prad_n, prad_v, "^--", color="#2B5B84", lw=1.6, ms=6.2,
                 label=r"Bounded 2-photon flux $P_{\mathrm{rad}}/(2E_{\mathrm{avg}})$ ($S^3/I^*$)")
    axD.semilogy(n_s3, prad_s3, "^", color="#7F8C8D", mfc="none", mew=1.4, ms=5.8,
                 label=r"Round $S^3$ ($n=2,3,4$): flux ($\triangle$) & $|\Delta\lambda^{(1)}|$ ($\diamond$)")

    # Centered O(c_anom) linear shifts: ZERO line bridging sign flip between n=10 (+) and n=14 (-)!
    m_damp = dl1_is < 0.0
    m_anti = dl1_is > 0.0
    axD.semilogy(n_is[m_damp], np.abs(dl1_is[m_damp]), "o-", color="#1E8449", lw=1.8, ms=6.4,
                 label=r"Centered $O(c_{\mathrm{anom}})$ damping ($n=14,18,22$)")
    axD.semilogy(n_is[m_anti], np.abs(dl1_is[m_anti]), "o", color="#1E8449", mfc="white", mew=1.9, ms=6.8,
                 label=r"Centered $O(c_{\mathrm{anom}})$ weak anti-damping ($n=2,10$)")
    axD.semilogy(n_s3, dl1_s3, "d", color="#7F8C8D", mfc="white", mew=1.3, ms=5.6)

    axD.semilogy(n_is, lam0_is, "s-", color="#C0392B", lw=1.9, ms=6.2,
                 label=r"0th-order Riegert rate $|\lambda_n^{(0)}|$ ($n=18,22$ Mathieu)")

    # Exact algebraic coincidence: 4*c_anom*w_scal^2 = 4 => omega_n = omega_scal/2 IDENTICAL to eps_EFT = 1 at n = 19.3625!
    n_coinc = math.sqrt(44672.0 / 108.0 + 1.0) - 1.0
    axD.axvline(n_coinc, color="#7D3C98", ls="-.", lw=1.3, alpha=0.9)
    axD.text(n_coinc + 0.35, 3.0e-5, r"$\omega_n=\omega_{\mathrm{scal}}/2 \equiv (\epsilon_{\mathrm{EFT}}=1)$" + f" ($n={n_coinc:.2f}$)",
             fontsize=6.6, color="#7D3C98", rotation=90, va="bottom")
    axD.text(17.4, 0.145, "2:1 Mathieu reheating\n($n=18, 22$)", ha="center", fontsize=7.4, color="#C0392B", fontweight="bold")

    axD.set_xlabel(r"TT Tensor Harmonic Index $n$ on $S^3/I^*$")
    axD.set_ylabel(r"Rate ($H_S$ units)")
    axD.set_title(r"(D) $S^3/I^*$ TT Gravitons: Exact Centered $O(c_{\mathrm{anom}})$ & Mathieu Rates")
    axD.set_xticks([2, 6, 10, 14, 18, 22])
    axD.set_xlim(1.0, 23.5)
    axD.set_ylim(5e-7, 0.35)
    axD.grid(True, which="major", alpha=0.25)
    axD.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=6.7)

    fig.suptitle(
        "Track IV-A: Causal 1-Loop Nonlocal $C\\ln(-\\mathrm{\u25a1}/\\mu^2)C$ Spin-2 Dynamics & Chiral Stabilization on $S^3/I^*$",
        fontsize=11.6, fontweight="bold"
    )

    out_png = os.path.join(OUT_DIR, "t4a_nonlocal_weyl_kam_diagnostic.png")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
