"""Exact Einstein-Maxwell-Lambda Exploration on the Poincaré Homology Sphere S^3 / I*.

Replaces Page 3's exotic w = -2/3 fluid with a standard positive cosmological
constant Lambda > 0 (w = -1) paired with conformal Beltrami-Maxwell radiation:

1. Stress-Energy Tensor Audit & Rank-2 No-Go Verification:
   - Computes exact 4x4 T_{mu nu} of Page 1's single null Beltrami wave,
     demonstrating T = u k_mu k_nu (rank 1, nonzero Poynting flux T_{03} = u,
     anisotropic pressures (0, 0, u)) -> CANNOT exactly source isotropic FLRW.
   - Verifies exact isotropic radiation tensor T = diag(u, u/3, u/3, u/3) for
     an orthogonal 3-axis Beltrami standing-mode triad (U(1)^3 / incoherent bath /
     classical homogeneous SU(2) Yang-Mills on S^3/I*).
2. Exact Closed-Form Einstein-Maxwell-Lambda Cosmology on S^3/I*:
   - Dimensionless Friedmann eq: (dy/dt)^2 = y^2 - 1 + gamma / y^2, gamma = Lambda*B/(3 c^2).
   - Verifies exact closed-form bounce y^2 = 1/2 + sqrt(1/4 - gamma) cosh(2t) for 0 <= gamma < 1/4,
     exact static Einstein universe at gamma_c = 1/4 (y_ES = 1/sqrt(2)), and
     exact singular Big-Bang -> de Sitter branch y^2 = 1/2 + sqrt(gamma - 1/4) sinh(2t) for gamma > 1/4.
3. Exact Conformal Closure & Phase-Locked Poincaré Resonance Spectrum:
   - Proves & verifies Delta_eta(gamma) = (2 / y_+) K(y_-^2 / y_+^2) via scipy.special.ellipk
     against direct quadrature (< 1e-13 error).
   - Pure de Sitter (gamma = 0) gives Delta_eta = pi: EXACTLY ONE closed geodesic circuit
     and ONE full Beltrami-Maxwell oscillation cycle (cos 2*eta) across all of cosmic time!
   - Solves to 15 digits for the discrete phase-locked spectrum gamma_k where Delta_eta(gamma_k) = k*pi
     (k = 1..8), pinpointing the exact gamma_2 reproducing Page 4's 2*pi circle with ZERO exotic matter.
4. Energy Conditions & Side-by-Side Comparison vs. Manuscript's w = -2/3 Model.
5. Renders 4-panel figure data/einstein_maxwell_lambda.png.
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize, special

from common import RESULTS_DIR, save_result


def maxwell_stress_energy(E: np.ndarray, cB: np.ndarray) -> np.ndarray:
    """Compute 4x4 orthonormal-tetrad EM stress-energy tensor T_{mu nu} (units eps_0 = 1)."""
    u = 0.5 * (float(np.dot(E, E)) + float(np.dot(cB, cB)))
    S = np.cross(E, cB)  # Poynting vector / c
    # Maxwell spatial stress sigma_ab = -(E_a E_b + cB_a cB_b) + u * delta_ab
    sigma = -(np.outer(E, E) + np.outer(cB, cB)) + u * np.eye(3)
    T = np.zeros((4, 4), dtype=np.float64)
    T[0, 0] = u
    T[0, 1:] = -S
    T[1:, 0] = -S
    T[1:, 1:] = sigma
    return T


def audit_stress_energy_isotropy() -> dict:
    """Compare single Page 1 null wave vs. 3-axis standing Beltrami triad."""
    etas = np.linspace(0.0, 2.0 * math.pi, 65)

    # 1. Single Page 1 null wave: E = (cos 2eta, sin 2eta, 0), cB = (sin 2eta, -cos 2eta, 0)
    T_null_samples = []
    for eta in etas:
        E = np.array([math.cos(2 * eta), math.sin(2 * eta), 0.0])
        cB = np.array([math.sin(2 * eta), -math.cos(2 * eta), 0.0])
        T_null_samples.append(maxwell_stress_energy(E, cB))
    T_null = T_null_samples[0]
    null_time_variation = max(float(np.max(np.abs(T - T_null))) for T in T_null_samples)
    # Check T_null == u * k k^T with k_mu = (1, 0, 0, +1) (since S_3 = -u -> T_{03} = +u)
    k_cov = np.array([1.0, 0.0, 0.0, 1.0])
    null_rank1_err = float(np.max(np.abs(T_null - np.outer(k_cov, k_cov))))
    # Minkowskian trace: -T_00 + T_11 + T_22 + T_33
    null_trace = float(-T_null[0, 0] + np.trace(T_null[1:, 1:]))

    # 2. Single standing Beltrami mode along X_a: E = cos(2eta) e_a, cB = sin(2eta) e_a
    #    Sum stress-energy of 3 independent (incoherent or U(1)^3 / SU(2)) modes along X_1, X_2, X_3
    T_triad_samples = []
    for eta in etas:
        T_sum = np.zeros((4, 4), dtype=np.float64)
        for a in range(3):
            e_a = np.zeros(3)
            e_a[a] = 1.0
            E = math.cos(2 * eta + a * 0.7) * e_a
            cB = math.sin(2 * eta + a * 0.7) * e_a
            T_sum += maxwell_stress_energy(E, cB)
        T_triad_samples.append(T_sum)
    T_triad = T_triad_samples[0]
    u_triad = T_triad[0, 0]
    T_iso_target = np.diag([u_triad, u_triad / 3.0, u_triad / 3.0, u_triad / 3.0])
    triad_iso_err = max(float(np.max(np.abs(T - T_iso_target))) for T in T_triad_samples)

    # 3. Verify Rank-2 No-Go numerically over 10,000 random single U(1) (E, cB) pairs
    rng = np.random.default_rng(20261123)
    Es = rng.normal(size=(10000, 3))
    Bs = rng.normal(size=(10000, 3))
    min_anisotropy = 1e9
    for E, B in zip(Es, Bs):
        u = 0.5 * (float(np.dot(E, E)) + float(np.dot(B, B)))
        M = np.outer(E, E) + np.outer(B, B)
        # Spatial stress sigma = u I - M; isotropic requires M = (2u/3) I
        # Since rank(M) <= 2, smallest eigenvalue of M is 0, so ||M - (2u/3)I||_op >= 2u/3!
        aniso = float(np.linalg.norm(M - (2.0 * u / 3.0) * np.eye(3), ord=2)) / u
        if aniso < min_anisotropy:
            min_anisotropy = aniso

    return {
        "single_null_T_matrix": T_null.tolist(),
        "single_null_time_variation": null_time_variation,
        "single_null_rank1_error": null_rank1_err,
        "single_null_trace": null_trace,
        "single_null_principal_pressures_over_u": [float(T_null[1, 1]), float(T_null[2, 2]), float(T_null[3, 3])],
        "single_null_poynting_T03_over_u": float(T_null[0, 3]),
        "rank2_nogo_min_anisotropy_10k_random": min_anisotropy,
        "rank2_nogo_exact_lower_bound": 2.0 / 3.0,
        "triad_3axis_exact_isotropy_error": triad_iso_err,
        "triad_w_eff": float(T_triad[1, 1] / u_triad),
    }


def verify_closed_form_solutions() -> dict:
    """Verify exact closed-form bounce and singular solutions against (dy/dt)^2 = y^2 - 1 + gamma/y^2."""
    ts = np.linspace(-3.0, 3.0, 2001)
    dt = 1e-6

    # Bounce branch: gamma = 0.15 < 1/4
    g_b = 0.15
    rad_b = math.sqrt(0.25 - g_b)
    y_b = np.sqrt(0.5 + rad_b * np.cosh(2.0 * ts))
    dy_dt_b = (rad_b * np.sinh(2.0 * ts)) / y_b
    res_b = float(np.max(np.abs(dy_dt_b ** 2 - (y_b ** 2 - 1.0 + g_b / (y_b ** 2)))))

    # Singular branch: gamma = 0.40 > 1/4, t > t_bang where 0.5 + sqrt(g - 1/4) sinh(2t) > 0
    g_s = 0.40
    rad_s = math.sqrt(g_s - 0.25)
    t_bang = -0.5 * math.asinh(0.5 / rad_s)
    ts_s = np.linspace(t_bang + 0.05, 3.0, 2001)
    y_s = np.sqrt(0.5 + rad_s * np.sinh(2.0 * ts_s))
    dy_dt_s = (rad_s * np.cosh(2.0 * ts_s)) / y_s
    res_s = float(np.max(np.abs(dy_dt_s ** 2 - (y_s ** 2 - 1.0 + g_s / (y_s ** 2)))))

    # Verify exact elliptic formula Delta_eta(gamma) = (2/y_+) K(y_-^2 / y_+^2) vs scipy.integrate.quad
    ellip_checks = []
    for g in [0.0, 0.05, 0.15, 0.22, 0.245, 0.2499]:
        yp2 = 0.5 + math.sqrt(0.25 - g)
        ym2 = 0.5 - math.sqrt(0.25 - g)
        yp = math.sqrt(yp2)
        eta_ellip = (2.0 / yp) * float(special.ellipk(ym2 / yp2))

        # Numerical check via y = yp + s^2 -> dy / sqrt((y^2-yp^2)(y^2-ym^2))
        def integrand(s):
            y = yp + s * s
            # (y^2 - yp^2) = s^2 * (y + yp)
            return 4.0 / math.sqrt((y + yp) * (y * y - ym2))

        eta_quad, _ = integrate.quad(integrand, 0.0, np.inf, epsabs=1e-13, epsrel=1e-13)
        ellip_checks.append(
            {
                "gamma": g,
                "y_min": yp,
                "delta_eta_ellip": eta_ellip,
                "delta_eta_quad": eta_quad,
                "abs_err": abs(eta_ellip - eta_quad),
            }
        )

    return {
        "bounce_ode_max_residual": res_b,
        "singular_ode_max_residual": res_s,
        "elliptic_formula_max_err": max(c["abs_err"] for c in ellip_checks),
        "elliptic_checks": ellip_checks,
    }


def delta_eta_bounce_ellip(gamma: float) -> float:
    yp2 = 0.5 + math.sqrt(max(0.0, 0.25 - gamma))
    ym2 = 0.5 - math.sqrt(max(0.0, 0.25 - gamma))
    return (2.0 / math.sqrt(yp2)) * float(special.ellipk(ym2 / yp2))


def delta_eta_singular(gamma: float) -> float:
    """Total conformal time from Big Bang (y=0) to future de Sitter infinity (y=inf) for gamma > 1/4."""
    def integrand(y):
        return 1.0 / math.sqrt(y ** 4 - y ** 2 + gamma)

    val, _ = integrate.quad(integrand, 0.0, np.inf, epsabs=1e-12, epsrel=1e-12)
    return float(val)


def solve_phase_locked_resonance_spectrum(max_k: int = 8) -> list:
    """Find exact gamma_k in [0, 1/4) where Delta_eta(gamma_k) = k * pi.

    At gamma_k, light completes EXACTLY k closed geodesic circuits around S^3/I*
    and Page 1's Beltrami-Maxwell field completes EXACTLY k full oscillation cycles
    (cos 2*eta) across the entire infinite bounce history t in (-inf, +inf).
    """
    resonances = []
    for k in range(1, max_k + 1):
        if k == 1:
            g_k = 0.0
        else:
            target = k * math.pi
            # Root bracket in (0, 0.25 - 1e-15)
            g_k = float(optimize.brentq(lambda g: delta_eta_bounce_ellip(g) - target, 1e-12, 0.25 - 1e-15, xtol=1e-15, rtol=1e-15))
        yp = math.sqrt(0.5 + math.sqrt(max(0.0, 0.25 - g_k)))
        # Ratio of radiation density to Lambda density at the bounce:
        # u_rad / u_Lambda = (gamma / y_min^4) / 1 = gamma / y_+^4
        rad_to_lambda_at_bounce = g_k / (yp ** 4)
        # Check residual
        res = abs(delta_eta_bounce_ellip(g_k) - k * math.pi)
        resonances.append(
            {
                "k_circuits_and_wave_cycles": k,
                "gamma_k": g_k,
                "distance_to_ceiling_0p25_minus_gamma": 0.25 - g_k,
                "y_min": yp,
                "u_rad_over_u_Lambda_at_bounce": rad_to_lambda_at_bounce,
                "total_conformal_time_over_pi": delta_eta_bounce_ellip(g_k) / math.pi,
                "closure_err": res,
            }
        )
    return resonances


def check_energy_conditions() -> dict:
    """Check NEC, WEC, DEC, SEC for Einstein-Maxwell-Lambda vs. Page 3's w = -2/3 fluid."""
    # For Lambda > 0 (u_L > 0, p_L = -u_L) + radiation (u_r >= 0, p_r = +u_r/3):
    # u_tot = u_L + u_r > 0
    # p_tot = -u_L + u_r/3
    # NEC: u + p = (4/3) u_r >= 0 -> SATISFIED for all u_r >= 0!
    # WEC: u >= 0 and u + p >= 0 -> SATISFIED!
    # DEC: u >= |p| -> u_L + u_r >= |-u_L + u_r/3| -> SATISFIED for all u_L, u_r >= 0!
    # SEC: u + 3p = -2 u_L + 2 u_r >= 0 -> violated when u_L > u_r (standard accelerating/bouncing condition!)
    # Note at the bounce for 0 <= gamma < 1/4: u_r/u_L = gamma/y_+^4 < 1 (since at gamma->1/4, y_+^4->1/4 => ratio -> 1!)
    # So SEC is violated at the bounce (as required by Penrose-Hawking singularity theorem to avoid singularity!),
    # while NEC, WEC, DEC are ALL strictly satisfied everywhere!
    # For Page 3's w = -2/3 fluid (u > 0, p = -2/3 u):
    # NEC: u + p = +u/3 >= 0 (satisfied)
    # WEC: u >= 0, u+p >= 0 (satisfied)
    # DEC: u >= |p| = 2u/3 (satisfied)
    # SEC: u + 3p = -u < 0 (violated, enabling bounce)
    return {
        "Einstein_Maxwell_Lambda": {
            "NEC": "Satisfied (u + p = 4/3 u_rad >= 0)",
            "WEC": "Satisfied (u > 0, u + p >= 0)",
            "DEC": "Satisfied (|p| <= u everywhere)",
            "SEC": "Violated for u_rad < u_Lambda (standard de Sitter bounce condition; u_rad/u_Lambda -> 1 at gamma_c = 1/4)",
            "Standard_Physics_Matter": True,
        },
        "Manuscript_w_minus23": {
            "NEC": "Satisfied (u + p = u/3 > 0)",
            "WEC": "Satisfied",
            "DEC": "Satisfied",
            "SEC": "Violated (u + 3p = -u < 0)",
            "Standard_Physics_Matter": False,
            "Note": "Requires frustrated cosmic domain-wall network (not EM radiation).",
        },
    }


def render_figure(resonances: list) -> str:
    """Render 4-panel comparison figure of Einstein-Maxwell-Lambda on S^3/I*."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.8,
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#f8fafc",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), dpi=160)
    fig.subplots_adjust(hspace=0.33, wspace=0.28, left=0.065, right=0.925, top=0.90, bottom=0.08)
    fig.suptitle(
        "Standard Einstein–Maxwell–Λ Cosmology on the Poincaré Homology Sphere S³/I*",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
        y=0.96,
    )

    # Panel 1: Exact Conformal Closure Delta_eta(gamma)/pi & Phase-Locked Spectrum
    ax = axes[0, 0]
    g_b = np.concatenate([np.linspace(0.0, 0.24, 150), np.linspace(0.2401, 0.24995, 120)])
    eta_b = [delta_eta_bounce_ellip(g) / math.pi for g in g_b]
    g_s = np.concatenate([np.linspace(0.2505, 0.27, 40), np.linspace(0.271, 0.60, 80)])
    eta_s = [delta_eta_singular(g) / math.pi for g in g_s]

    ax.plot(g_b, eta_b, "-", color="#0f766e", lw=2.2, label="Bounce branch (0 ≤ γ < 1/4)")
    ax.plot(g_s, eta_s, "--", color="#be123c", lw=1.8, label="Singular Big-Bang branch (γ > 1/4)")
    ax.axvline(0.25, color="#b45309", ls=":", lw=1.8, label="Einstein static ceiling γ_c = 1/4")

    # Mark phase-locked integer resonances k = 1, 2, 3, 4
    for r in resonances[:4]:
        k = r["k_circuits_and_wave_cycles"]
        gk = r["gamma_k"]
        ax.plot(gk, k, "o", color="#1d4ed8", ms=6.5, zorder=5)
        offset_x = 0.015 if k == 1 else -0.095
        offset_y = 0.12 if k == 1 else 0.08
        lbl = f"k={k} (γ={gk:.4f})" if k != 2 else f"k=2: Δ η=2π!\n(γ₂={gk:.5f})"
        ax.annotate(lbl, (gk, k), xytext=(gk + offset_x, k + offset_y), fontsize=7.6, color="#1e3a8a",
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=0.7) if k >= 2 else None)

    ax.set_xlim(-0.01, 0.55)
    ax.set_ylim(0.5, 4.8)
    ax.set_xlabel("Dimensionless Radiation Weight γ = ΛB / (3c²)")
    ax.set_ylabel("Total Conformal Time Δ η / π  (= Poincaré Circuits)")
    ax.set_title("1. Exact Conformal Closure & Phase-Locked Spectrum")
    ax.legend(fontsize=7.8, loc="upper right", framealpha=0.93)
    ax.grid(True, alpha=0.25)

    # Panel 2: Bounce & Singular Trajectories y(t)
    ax = axes[0, 1]
    t_arr = np.linspace(-2.2, 2.2, 400)
    g2 = resonances[1]["gamma_k"]
    ax.plot(t_arr, np.cosh(t_arr), "-", color="#0f766e", lw=2.2, label="γ₁ = 0 (Pure dS: Δ η = π)")
    ax.plot(t_arr, np.sqrt(0.5 + math.sqrt(0.25 - g2) * np.cosh(2 * t_arr)), "-", color="#1d4ed8", lw=2.2,
            label=f"γ₂ = {g2:.4f} (Phase-locked Δ η = 2π)")
    ax.plot(t_arr, np.full_like(t_arr, 1.0 / math.sqrt(2.0)), ":", color="#b45309", lw=1.8,
            label="γ_c = 1/4 (Einstein static y = 1/√2)")
    # Singular trajectory gamma = 0.35
    rad_s = math.sqrt(0.35 - 0.25)
    tb = -0.5 * math.asinh(0.5 / rad_s)
    t_sing = np.linspace(tb + 1e-4, 2.2, 300)
    ax.plot(t_sing, np.sqrt(0.5 + rad_s * np.sinh(2 * t_sing)), "--", color="#be123c", lw=1.9,
            label="γ = 0.35 > 1/4 (Big Bang singularity)")
    # Manuscript w=-2/3 parabolic bounce for comparison
    ax.plot(t_arr, 1.0 + 0.25 * (t_arr ** 2), "-.", color="#64748b", lw=1.5, label="Page 3 parabolic bounce (w = -2/3)")

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(0.0, 4.2)
    ax.set_xlabel("Dimensionless Cosmic Time H_Λ t")
    ax.set_ylabel("Dimensionless Scale Factor y = a / a_dS")
    ax.set_title("2. Scale-Factor Dynamics Across γ = 1/4 Bifurcation")
    ax.legend(fontsize=7.6, loc="upper center", framealpha=0.93)
    ax.grid(True, alpha=0.25)

    # Panel 3: Effective Potential V_eff(y) = -(y^2 - 1 + gamma/y^2)
    ax = axes[1, 0]
    ys = np.linspace(0.25, 1.5, 400)
    for g, col, lbl in [
        (0.0, "#0f766e", "γ = 0 (barrier height 1.0)"),
        (0.15, "#2563eb", "γ = 0.15 (bounce + trapped singular pocket)"),
        (0.25, "#b45309", "γ_c = 1/4 (barrier vanishes at y = 1/√2)"),
        (0.35, "#be123c", "γ = 0.35 (no barrier → singularity)"),
    ]:
        V_eff = -(ys ** 2 - 1.0 + g / (ys ** 2))
        ax.plot(ys, V_eff, lw=2.0, color=col, label=lbl)
    ax.axhline(0.0, color="#0f172a", lw=1.0)
    ax.fill_between(ys, 0, 1.2, color="#e2e8f0", alpha=0.35, label="Classically forbidden (V_eff > 0)")
    ax.set_xlim(0.25, 1.45)
    ax.set_ylim(-1.0, 1.1)
    ax.set_xlabel("Dimensionless Scale Factor y = a / a_dS")
    ax.set_ylabel("Effective Friedmann Potential  −(dy/dt)²")
    ax.set_title("3. Exact Bounce Barrier & Radiation Ceiling γ_c = 1/4")
    ax.legend(fontsize=7.4, loc="lower left", framealpha=0.93)
    ax.grid(True, alpha=0.25)

    # Panel 4: Comparison of Radiation Ceiling & Radiation-to-Dark-Energy Ratio at Bounce
    ax = axes[1, 1]
    ks = [r["k_circuits_and_wave_cycles"] for r in resonances]
    ratios = [r["u_rad_over_u_Lambda_at_bounce"] for r in resonances]
    dists = [r["distance_to_ceiling_0p25_minus_gamma"] for r in resonances]
    ax.semilogy(ks, dists, "o-", color="#b45309", lw=2.0, ms=6, label="Ceiling margin  1/4 − γ_k")
    ax.set_xlabel("Phase-Locked Circuit / Beltrami Wave Count  k  (Δ η = k π)")
    ax.set_ylabel("Margin Below Einstein-Static Ceiling  (1/4 − γ_k)", color="#b45309")
    ax.tick_params(axis="y", labelcolor="#b45309")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(ks, ratios, "s--", color="#0f766e", lw=1.8, ms=5.5, label="u_rad / u_Λ at bounce")
    ax2.set_ylabel("Bounce Radiation-to-Λ Ratio  u_rad / u_Λ", color="#0f766e")
    ax2.tick_params(axis="y", labelcolor="#0f766e")
    ax2.set_ylim(-0.05, 1.08)
    ax.set_title("4. Exponential Convergence of Resonances γ_k → 1/4")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.6, loc="center right", framealpha=0.93)

    out_png = os.path.join(RESULTS_DIR, "einstein_maxwell_lambda.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def main():
    iso = audit_stress_energy_isotropy()
    sols = verify_closed_form_solutions()
    res_spec = solve_phase_locked_resonance_spectrum(max_k=8)
    ecs = check_energy_conditions()
    fig_path = render_figure(res_spec)

    # Geometric volume comparison on S^3/I* vs Euclidean Hubble ball:
    # Vol(S^3/I*, a_dS) = (pi^2 / 60) a_dS^3; Euclidean ball = (4*pi/3) a_dS^3 -> ratio = pi / 80
    vol_ratio_ds = math.pi / 80.0

    payload = {
        "model": "Standard Einstein-Maxwell-Lambda on Poincaré Homology Sphere S^3/I*",
        "stress_energy_isotropy_audit": iso,
        "closed_form_verification": sols,
        "exact_critical_ceiling_gamma_c": 0.25,
        "einstein_static_radius_y_ES": 1.0 / math.sqrt(2.0),
        "poincare_vol_over_euclidean_dS_ball": vol_ratio_ds,
        "phase_locked_resonance_spectrum": res_spec,
        "energy_conditions": ecs,
        "figure_path": fig_path,
    }
    out_json = save_result("einstein_maxwell_lambda", payload)
    print(f"Saved JSON -> {out_json}")
    print(f"Saved PNG  -> {fig_path}")
    print(f"\n[Rank-2 No-Go] min anisotropy over 10k random single U(1) fields = {iso['rank2_nogo_min_anisotropy_10k_random']:.4f} (exact lower bound = 2/3 = {2/3:.4f})")
    print(f"[3-Axis Triad] isotropy error = {iso['triad_3axis_exact_isotropy_error']:.2e}, w_eff = {iso['triad_w_eff']:.6f}")
    print(f"[Closed Form]  bounce ODE res = {sols['bounce_ode_max_residual']:.2e}, singular ODE res = {sols['singular_ode_max_residual']:.2e}, ellip err = {sols['elliptic_formula_max_err']:.2e}")
    print("\n[Phase-Locked Poincaré-Beltrami Resonance Spectrum (Delta_eta = k * pi)]:")
    for r in res_spec[:6]:
        print(f"  k={r['k_circuits_and_wave_cycles']}: gamma_k = {r['gamma_k']:.12f}  "
              f"(1/4 - gamma_k = {r['distance_to_ceiling_0p25_minus_gamma']:.3e}, "
              f"y_min = {r['y_min']:.6f}, u_rad/u_Lambda = {r['u_rad_over_u_Lambda_at_bounce']:.6f})")


if __name__ == "__main__":
    main()
