#!/usr/bin/env python3
"""
p05_02_solve_and_phase_space.py
===============================
Goal 2 analytical + high-precision numerical solver for stream p05_starobinsky_casimir:
- Exact 4-variable nonsingular trace ODE (x, h, r, s) with exact Hamiltonian conservation monitor x^4 * e_00
- Starobinsky (1980) S^3 benchmarks (exact cosh(tau) de Sitter bounce + graceful-exit vs runaway perturbations)
- Asymptotic de Sitter fixed points & linearized stability eigenvalues
- Einstein-static fixed points & exact 4th-order biquadratic characteristic polynomial check
- Comprehensive S^3/I* bounce phase-space mapping across (sector, sigma_tilde, lambda, x_b, s_b)
"""

import json
import math
import os
import numpy as np
from scipy.integrate import solve_ivp

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)

# Exact dimensionless parameters from Goal 1 (p05_01_identities.json)
KAPPA_VECTOR = -44738.0 / 93.0          # -481.0537634408602 (any N_V >= 1 pure vector)
SIGMA0_VECTOR = 9.0 / 31.0              # +0.2903225806451613
KAPPA_SCALAR = -174218.0 / 3.0          # -58072.666666666664 (any N_S >= 1 pure scalar)
SIGMA0_SCALAR = -1.0                    # tachyonic at alpha_ct = 0
KAPPA_SM = 21991472.0 / 5973.0          # +3681.813494056588
SIGMA0_SM = 14.0 / 181.0                # +0.07734806629834254


def e00_constraint(x, h, r, s, sigma_t, lam, kappa):
    """Exact dimensionless (00) Friedmann constraint e_00(x, h, r, s). Should equal 0."""
    y = h * h + 1.0 / (x * x)
    return y - lam - y * y + sigma_t * (r * (2.0 * y - r) + 2.0 * h * s) - kappa / (x**4)


def ode_rhs(tau, state, sigma_t, lam):
    """Exact 4-variable nonsingular trace ODE for (x, h, r, s)."""
    x, h, r, s = state
    inv_x2 = 1.0 / (x * x)
    y = h * h + inv_x2
    dx = x * h
    dh = r - 2.0 * h * h - inv_x2
    dr = s
    ds = -3.0 * h * s + (2.0 * lam - r + 2.0 * (r - y) * y) / sigma_t
    return [dx, dh, dr, ds]


def bounce_initial_state(x_b, sigma_t, lam, kappa, s_b=0.0, sign_hdot=+1):
    """
    Construct exact initial state (x_b, h=0, r_b, s_b) on e_00 = 0 at a turning point.
    Returns None if no real turning point exists at x_b.
    """
    kappa_eff = 1.0 - sigma_t + kappa
    rhs = (1.0 / (x_b * x_b) - lam - kappa_eff / (x_b**4)) / sigma_t
    if rhs < 0.0:
        return None
    hdot_b = sign_hdot * math.sqrt(rhs)
    r_b = 1.0 / (x_b * x_b) + hdot_b
    return np.array([x_b, 0.0, r_b, s_b], dtype=float)


def classify_trajectory(x_b, sigma_t, lam, kappa, s_b=0.0, tau_max=60.0, sign_hdot=+1):
    """
    Integrate forward from tau=0 to tau_max with strict tolerances and event detection.
    Classifies outcome:
      - 'no_turning_point': bounce IC forbidden by e_00=0
      - 'singularity_blowup': |h| or |r| exceeds blowup ceiling (curvature runaway R -> +-inf)
      - 'singularity_collapse': x drops below collapse floor (Big Crunch x -> 0)
      - 'ir_de_sitter_bounce': converges to IR de Sitter fixed point h_dS,- (when lam > 0)
      - 'uv_de_sitter_stay': stays near UV Starobinsky de Sitter h_dS,+
      - 'minkowski_oscillatory_exit': lam==0 graceful exit (x growing, h -> 0+, oscillating scalaron)
      - 'recollapse_bounce_cycle': turned around (h crossed 0 downward) without collapsing yet
      - 'expanding_unresolved': still expanding at tau_max without clear asymptote
    """
    y0 = bounce_initial_state(x_b, sigma_t, lam, kappa, s_b=s_b, sign_hdot=sign_hdot)
    if y0 is None:
        return {"outcome": "no_turning_point"}

    # Event: blowup
    def ev_blowup(t, y, sigma_t, lam):
        return 1e4 - max(abs(y[1]), abs(y[2]))
    ev_blowup.terminal = True
    ev_blowup.direction = -1

    # Event: collapse
    def ev_collapse(t, y, sigma_t, lam):
        return y[0] - 1e-4
    ev_collapse.terminal = True
    ev_collapse.direction = -1

    # Event: huge scale factor (exponential inflation well underway, x > 1e8 * x_b)
    x_cap = min(1e12, max(1e8, 1e6 * x_b))
    def ev_inflated(t, y, sigma_t, lam):
        return x_cap - y[0]
    ev_inflated.terminal = True
    ev_inflated.direction = -1

    # Event: turnaround h crosses 0 downward (after initial tau > 0)
    def ev_turnaround(t, y, sigma_t, lam):
        return y[1] if t > 1e-5 else 1.0
    ev_turnaround.terminal = False
    ev_turnaround.direction = -1

    sol = solve_ivp(
        ode_rhs,
        (0.0, tau_max),
        y0,
        args=(sigma_t, lam),
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        events=[ev_blowup, ev_collapse, ev_inflated, ev_turnaround],
        max_step=0.25,
    )

    xs, hs, rs, ss = sol.y
    # Compute exact Hamiltonian constraint residual: |e_00| and |x^4 e_00|
    e00_vals = np.array([e00_constraint(xs[i], hs[i], rs[i], ss[i], sigma_t, lam, kappa) for i in range(len(sol.t))])
    x4_e00_vals = (xs**4) * e00_vals
    # Relative residual normalized by max term magnitude in e_00
    rel_res = []
    for i in range(len(sol.t)):
        y_i = hs[i]**2 + 1.0 / (xs[i]**2)
        scale = max(1.0, abs(y_i), abs(y_i**2), abs(sigma_t * rs[i] * (2 * y_i - rs[i])), abs(sigma_t * 2 * hs[i] * ss[i]), abs(kappa / xs[i]**4))
        rel_res.append(abs(e00_vals[i]) / scale)
    max_rel_e00 = float(np.max(rel_res))
    max_abs_x4_e00 = float(np.max(np.abs(x4_e00_vals)))

    h_end = float(hs[-1])
    x_end = float(xs[-1])
    r_end = float(rs[-1])
    t_end = float(sol.t[-1])

    # Determine outcome
    if len(sol.t_events[0]) > 0:
        outcome = "singularity_blowup_pos" if r_end > 0 else "singularity_blowup_neg"
    elif len(sol.t_events[1]) > 0:
        outcome = "singularity_collapse"
    else:
        # Check asymptotic de Sitter fixed points if lam <= 0.25
        if lam > 0.0 and lam <= 0.25:
            disc = math.sqrt(max(0.0, 1.0 - 4.0 * lam))
            h_ir = math.sqrt(0.5 * (1.0 - disc))
            h_uv = math.sqrt(0.5 * (1.0 + disc))
            if abs(h_end - h_ir) < 1e-3 * max(1.0, h_ir) and x_end > 10.0 * x_b:
                outcome = "ir_de_sitter_bounce"
            elif abs(h_end - h_uv) < 1e-2 * h_uv and x_end > 10.0 * x_b:
                outcome = "uv_de_sitter_stay"
            elif len(sol.t_events[3]) > 0:
                outcome = "recollapse_or_oscillating"
            else:
                outcome = "expanding_to_ir_dS" if (h_end > 0 and x_end > 100 * x_b) else "expanding_unresolved"
        elif lam == 0.0:
            if h_end > 0 and h_end < 0.2 and x_end > 2.0 * x_b:
                outcome = "minkowski_oscillatory_exit"
            elif abs(h_end - 1.0) < 1e-2 and x_end > 10.0 * x_b:
                outcome = "uv_de_sitter_stay"
            elif len(sol.t_events[3]) > 0:
                outcome = "recollapse_or_oscillating"
            else:
                outcome = "expanding_unresolved"
        else:
            outcome = "expanding_unresolved"

    return {
        "outcome": outcome,
        "x_b": float(x_b),
        "r_b": float(y0[2]),
        "hdot_b": float(y0[2] - 1.0 / (x_b * x_b)),
        "t_end": t_end,
        "x_end": x_end,
        "h_end": h_end,
        "r_end": r_end,
        "n_turnarounds": int(len(sol.t_events[3])),
        "max_rel_e00": max_rel_e00,
        "max_abs_x4_e00": max_abs_x4_e00,
        "sol_t": sol.t,
        "sol_y": sol.y,
    }


def strip_sol(d):
    return {k: v for k, v in d.items() if k not in ("sol_t", "sol_y")}


def run_starobinsky_1980_benchmarks():
    """Verify exact Starobinsky (1980) S^3 benchmarks (kappa=0, lam=0)."""
    out = {}
    # 1. Exact closed de Sitter bounce x(tau) = cosh(tau) at sigma_t = 9/31
    res_exact = classify_trajectory(1.0, SIGMA0_VECTOR, 0.0, 0.0, s_b=0.0, tau_max=10.0)
    t_arr = res_exact["sol_t"]
    x_exact = np.cosh(t_arr)
    err_cosh = float(np.max(np.abs(res_exact["sol_y"][0] - x_exact) / x_exact))
    out["S3_exact_deSitter_cosh"] = {
        **strip_sol(res_exact),
        "max_rel_err_vs_cosh": err_cosh,
    }

    # 2. Perturbed S^3 bounce: x_b = 1 + 1e-4 (graceful exit to Minkowski scalaron oscillations!)
    res_plus = classify_trajectory(1.0 + 1e-4, SIGMA0_VECTOR, 0.0, 0.0, s_b=0.0, tau_max=80.0)
    out["S3_perturbed_plus_1e4_graceful_exit"] = strip_sol(res_plus)

    # 3. Perturbed S^3 bounce: x_b = 1 - 1e-4 (blowup runaway!)
    res_minus = classify_trajectory(1.0 - 1e-4, SIGMA0_VECTOR, 0.0, 0.0, s_b=0.0, tau_max=40.0)
    out["S3_perturbed_minus_1e4_runaway"] = strip_sol(res_minus)

    # 4. Light scalaron Starobinsky inflation benchmark: sigma_t = 100 (M = 0.1 H_S), x_b = 1 + 1e-4
    res_light = classify_trajectory(1.0 + 1e-4, 100.0, 0.0, 0.0, s_b=0.0, tau_max=120.0)
    out["S3_light_scalaron_sigma100_plus1e4"] = strip_sol(res_light)

    # 5. Tachyonic scalar benchmark: sigma_t = -1, x_b = 1 + 1e-3 (check de Sitter attractor!)
    # Wait: at sigma_t = -1, kappa=0, lam=0: kappa_eff = 1 - (-1) = 2.
    # For x_b = 1 + 1e-3: rhs = (1/x_b^2 - 2/x_b^4)/(-1) = (2/x_b^4 - 1/x_b^2) > 0!
    res_tach = classify_trajectory(1.0 + 1e-3, -1.0, 0.0, 0.0, s_b=0.0, tau_max=12.0)
    out["S3_tachyonic_scalar_sigma_m1_dS_attractor"] = strip_sol(res_tach)
    return out


def verify_einstein_static_stability():
    """Verify exact biquadratic characteristic polynomial for Einstein-static fixed points."""
    tests = []
    for lam, sigma_t in [(0.05, 9.0 / 31.0), (0.15, 1.0), (0.20, 10.0), (0.10, -5.0)]:
        # Exact ES requires kappa_eff = 1/(4*lam) => kappa = 1/(4*lam) - 1 + sigma_t
        kappa_es = 1.0 / (4.0 * lam) - 1.0 + sigma_t
        x_es = 1.0 / math.sqrt(2.0 * lam)
        r_es = 2.0 * lam
        # Check exact residual at ES
        e00_es = e00_constraint(x_es, 0.0, r_es, 0.0, sigma_t, lam, kappa_es)
        rhs_es = ode_rhs(0.0, [x_es, 0.0, r_es, 0.0], sigma_t, lam)

        # Numerical Jacobian of ode_rhs at ES
        eps = 1e-7
        base = np.array([x_es, 0.0, r_es, 0.0])
        J = np.zeros((4, 4))
        for j in range(4):
            d = np.zeros(4)
            d[j] = eps
            fp = np.array(ode_rhs(0.0, base + d, sigma_t, lam))
            fm = np.array(ode_rhs(0.0, base - d, sigma_t, lam))
            J[:, j] = (fp - fm) / (2.0 * eps)
        eig_num = np.sort_complex(np.linalg.eigvals(J))

        # Analytical roots of sigma_t * w^4 + (1 - 4*lam*(1+sigma_t))*w^2 - 4*lam = 0
        coeffs = [sigma_t, 0.0, 1.0 - 4.0 * lam * (1.0 + sigma_t), 0.0, -4.0 * lam]
        eig_ana = np.sort_complex(np.roots(coeffs))
        max_eig_err = float(np.max(np.abs(eig_num - eig_ana)))

        tests.append({
            "lam": lam,
            "sigma_t": sigma_t,
            "kappa_es": kappa_es,
            "x_es": x_es,
            "e00_at_ES": float(e00_es),
            "max_rhs_at_ES": float(np.max(np.abs(rhs_es))),
            "max_eig_err_vs_biquadratic": max_eig_err,
            "eigenvalues": [f"{z.real:+.6f}{z.imag:+.6f}j" for z in eig_ana],
            "linearly_stable": bool(np.all(np.real(eig_ana) <= 1e-12)),
        })
    return tests


def scan_poincare_bounces_lam0():
    """
    Scan S^3/I* bounces at Lambda = 0 across pure vector (kappa = -481.054) and
    pure conformal scalar (kappa = -58072.67) for multiple sigma_t > 0 values.
    """
    scans = {}

    # 1. Pure vector at zeta zero-counterterm sigma_t = 9/31, lam = 0
    xb_grid = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 21.933, 25.0, 30.0, 50.0, 100.0, 500.0]
    vec_sig0 = []
    for xb in xb_grid:
        res = classify_trajectory(xb, SIGMA0_VECTOR, 0.0, KAPPA_VECTOR, s_b=0.0, tau_max=60.0)
        vec_sig0.append(strip_sol(res))
    scans["vector_sigma0_lam0"] = vec_sig0

    # 2. Pure vector across renormalized sigma_t values at lam = 0
    # Check sigma_t = 1, 10, 100, 481.054 (matching |kappa|!), 1000, 10000
    vec_sigma_scan = {}
    for sig in [0.05, 1.0, 10.0, 100.0, 481.0537634408602, 1000.0, 10000.0]:
        sub = []
        for xb in [1.0, 5.0, 15.0, 22.0, 30.0, 50.0, 100.0, 300.0]:
            res = classify_trajectory(xb, sig, 0.0, KAPPA_VECTOR, s_b=0.0, tau_max=100.0)
            sub.append(strip_sol(res))
        vec_sigma_scan[f"sigma_{sig:.4g}"] = sub
    scans["vector_sigma_sweep_lam0"] = vec_sigma_scan

    # 3. Check time-asymmetric bounces s_b != 0 at vector sigma_0, lam = 0
    asym_vec = []
    for xb in [5.0, 22.0, 50.0]:
        for sb in [-50.0, -5.0, -1.0, -0.1, 0.1, 1.0, 5.0, 50.0]:
            res = classify_trajectory(xb, SIGMA0_VECTOR, 0.0, KAPPA_VECTOR, s_b=sb, tau_max=60.0)
            asym_vec.append({"s_b": sb, **strip_sol(res)})
    scans["vector_asymmetric_sb_lam0"] = asym_vec

    # 4. Pure conformal scalar at lam = 0 for sigma_t in {1.0, 100.0, 58072.67, 1e5}
    scal_scan = {}
    for sig in [1.0, 100.0, 58072.6666667, 2e5]:
        sub = []
        for xb in [10.0, 100.0, 241.0, 400.0, 1000.0]:
            res = classify_trajectory(xb, sig, 0.0, KAPPA_SCALAR, s_b=0.0, tau_max=120.0)
            sub.append(strip_sol(res))
        scal_scan[f"sigma_{sig:.4g}"] = sub
    scans["scalar_sigma_sweep_lam0"] = scal_scan

    return scans


def scan_poincare_bounces_lam_pos():
    """
    Scan S^3/I* bounces at Lambda > 0 (0 < lam < 0.25) to locate exact nonsingular
    eternal IR de Sitter bounces vs runaway singularities and compute exact critical boundary x_{b,crit}(lam)!
    """
    results = {}

    # Pure vector, sigma_t = 9/31, across lam in {1e-4, 1e-3, 1e-2, 0.05, 0.10, 0.20}
    vec_lam_list = []
    for lam in [1e-4, 1e-3, 1e-2, 0.05, 0.10, 0.20]:
        kappa_eff = 1.0 - SIGMA0_VECTOR + KAPPA_VECTOR
        # Exact upper bound x_{b,max} where hdot_b = 0: lam*x^4 - x^2 + kappa_eff = 0
        x_b_max = math.sqrt((1.0 + math.sqrt(1.0 - 4.0 * lam * kappa_eff)) / (2.0 * lam))
        # Exact sigma=0 GR-branch bounce radius x_GR_bounce:
        # Y_-(x) = 1/x^2 => x^{-2} - x^{-4} = lam + kappa/x^4 => lam*x^4 - x^2 + (1 + kappa) = 0!
        x_gr_bounce = math.sqrt((1.0 + math.sqrt(1.0 - 4.0 * lam * (1.0 + KAPPA_VECTOR))) / (2.0 * lam))

        # Sample across (0.05*x_b_max, 0.999*x_b_max)
        fracs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.995, 0.999]
        pts = []
        for f in fracs:
            xb = f * x_b_max
            res = classify_trajectory(xb, SIGMA0_VECTOR, lam, KAPPA_VECTOR, s_b=0.0, tau_max=max(80.0, 25.0 / math.sqrt(lam)))
            pts.append({"frac": f, **strip_sol(res)})

        # Bisect critical boundary between blowup and ir_de_sitter_bounce if transition exists
        good = [p for p in pts if p["outcome"] in ("ir_de_sitter_bounce", "expanding_to_ir_dS")]
        bad = [p for p in pts if "blowup" in p["outcome"]]
        x_crit = None
        if good and bad:
            lo = max(p["x_b"] for p in bad if p["x_b"] < min(g["x_b"] for g in good)) if any(p["x_b"] < min(g["x_b"] for g in good) for p in bad) else None
            hi = min(g["x_b"] for g in good)
            if lo is not None:
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    r_mid = classify_trajectory(mid, SIGMA0_VECTOR, lam, KAPPA_VECTOR, s_b=0.0, tau_max=max(80.0, 25.0 / math.sqrt(lam)))
                    if r_mid["outcome"] in ("ir_de_sitter_bounce", "expanding_to_ir_dS"):
                        hi = mid
                    else:
                        lo = mid
                x_crit = 0.5 * (lo + hi)

        vec_lam_list.append({
            "lam": lam,
            "x_b_max": x_b_max,
            "x_gr_bounce_sigma0": x_gr_bounce,
            "x_b_crit_bisected": x_crit,
            "crit_over_max": (x_crit / x_b_max) if x_crit is not None else None,
            "n_eternal_bounces": len(good),
            "n_blowups": len(bad),
            "sample_points": pts,
        })
    results["vector_sigma0_lam_pos"] = vec_lam_list

    # Also test light scalaron sigma_t = 100, lam = 0.01 (and lam = 0.001) on S^3/I* vs S^3!
    light_comp = {}
    for label, kap in [("S3_Istar_vector", KAPPA_VECTOR), ("S3_round", 0.0)]:
        lam = 0.01
        sig = 100.0
        kappa_eff = 1.0 - sig + kap
        x_b_max = math.sqrt((1.0 + math.sqrt(1.0 - 4.0 * lam * kappa_eff)) / (2.0 * lam))
        pts = []
        for f in [0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 0.999]:
            xb = f * x_b_max
            res = classify_trajectory(xb, sig, lam, kap, s_b=0.0, tau_max=200.0)
            pts.append({"frac": f, **strip_sol(res)})
        light_comp[label] = {"lam": lam, "sigma_t": sig, "x_b_max": x_b_max, "points": pts}
    results["light_scalaron_sigma100_lam0p01"] = light_comp

    # Standard Model comparison (kappa_SM = +3681.81, sigma0_SM = 14/181) at lam = 1e-4 (check Einstein static & bounces!)
    # Note: for SM, kappa_eff = 1 - 14/181 + 3681.81 = +3682.74 > 0!
    # So turning points exist iff 4*lam*kappa_eff <= 1 => lam <= 1/(4*3682.74) = 6.788e-5!
    lam_sm_crit = 1.0 / (4.0 * (1.0 - SIGMA0_SM + KAPPA_SM))
    sm_pts = []
    lam_sm = 0.5 * lam_sm_crit  # subcritical: two roots x_- < x_+
    kappa_eff_sm = 1.0 - SIGMA0_SM + KAPPA_SM
    disc_sm = math.sqrt(1.0 - 4.0 * lam_sm * kappa_eff_sm)
    x_sm_minus = math.sqrt((1.0 - disc_sm) / (2.0 * lam_sm))
    x_sm_plus = math.sqrt((1.0 + disc_sm) / (2.0 * lam_sm))
    for f in [0.1, 0.5, 0.9, 0.99]:
        xb = x_sm_minus + f * (x_sm_plus - x_sm_minus)
        res = classify_trajectory(xb, SIGMA0_SM, lam_sm, KAPPA_SM, s_b=0.0, tau_max=600.0)
        sm_pts.append({"frac_in_pocket": f, **strip_sol(res)})
    results["standard_model_subcritical_lam"] = {
        "lam_ES_critical": lam_sm_crit,
        "lam_tested": lam_sm,
        "x_sm_minus": x_sm_minus,
        "x_sm_plus": x_sm_plus,
        "points": sm_pts,
    }

    return results


def main():
    print("Running Starobinsky (1980) S^3 benchmarks...")
    bench = run_starobinsky_1980_benchmarks()
    print("Verifying Einstein-static stability biquadratic...")
    es = verify_einstein_static_stability()
    print("Scanning S^3/I* bounces at Lambda = 0...")
    lam0 = scan_poincare_bounces_lam0()
    print("Scanning S^3/I* bounces at Lambda > 0...")
    lam_pos = scan_poincare_bounces_lam_pos()

    out = {
        "starobinsky_1980_S3_benchmarks": bench,
        "einstein_static_stability": es,
        "poincare_bounces_lam0": lam0,
        "poincare_bounces_lam_pos": lam_pos,
    }
    out_path = os.path.join(OUT_DIR, "p05_02_phase_space.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)

    # Print concise summary
    print("\n=== STAROBINSKY S^3 BENCHMARKS ===")
    for k, v in bench.items():
        print(f"  {k}: outcome={v['outcome']}, max_rel_e00={v['max_rel_e00']:.2e}, x_end={v['x_end']:.3e}, h_end={v['h_end']:.4f}")
    print("\n=== EINSTEIN-STATIC STABILITY ===")
    for t in es:
        print(f"  lam={t['lam']}, sig={t['sigma_t']:.3f}: max_eig_err={t['max_eig_err_vs_biquadratic']:.2e}, stable={t['linearly_stable']}, eigs={t['eigenvalues']}")
    print("\n=== S^3/I* VECTOR (sigma_0=9/31, lam=0) ===")
    for p in lam0["vector_sigma0_lam0"]:
        print(f"  x_b={p['x_b']:6.2f}: outcome={p['outcome']:25s}, t_end={p['t_end']:6.2f}, x_end={p['x_end']:.3e}, h_end={p['h_end']:+.3e}, max_rel_e00={p['max_rel_e00']:.2e}")
    print("\n=== S^3/I* VECTOR (sigma_sweep, lam=0) ===")
    for k, arr in lam0["vector_sigma_sweep_lam0"].items():
        outcomes = [f"{p['x_b']:.0f}:{p['outcome'][:8]}" for p in arr]
        print(f"  {k:15s}: " + ", ".join(outcomes))
    print("\n=== S^3/I* VECTOR (sigma_0=9/31, lam>0) ===")
    for blk in lam_pos["vector_sigma0_lam_pos"]:
        print(f"  lam={blk['lam']:7.4f}: x_gr={blk['x_gr_bounce_sigma0']:7.3f}, x_max={blk['x_b_max']:7.3f}, x_crit={str(round(blk['x_b_crit_bisected'],4)) if blk['x_b_crit_bisected'] else 'None':>8s}, eternal={blk['n_eternal_bounces']}/{len(blk['sample_points'])}")
    print("\n=== LIGHT SCALARON sigma=100, lam=0.01 ===")
    for lbl, blk in lam_pos["light_scalaron_sigma100_lam0p01"].items():
        outcomes = [f"{p['frac']:.2f}:{p['outcome'][:8]}" for p in blk["points"]]
        print(f"  {lbl:16s} (x_max={blk['x_b_max']:.2f}): " + ", ".join(outcomes))
    print("\n=== STANDARD MODEL SUBCRITICAL lam ===")
    sm = lam_pos["standard_model_subcritical_lam"]
    print(f"  lam_ES_crit={sm['lam_ES_critical']:.3e}, x_in=[{sm['x_sm_minus']:.2f}, {sm['x_sm_plus']:.2f}]")
    for p in sm["points"]:
        print(f"    f={p['frac_in_pocket']:.2f} (x_b={p['x_b']:.2f}): outcome={p['outcome']}, max_rel_e00={p['max_rel_e00']:.2e}")


if __name__ == "__main__":
    main()
