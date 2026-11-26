#!/usr/bin/env python3
"""
p05_02b_deep_dynamics.py
========================
Deep dynamical characterization for Goal 2 (p05_starobinsky_casimir):
1. Exact bisection of Starobinsky separatrix x_crit(sigma_t, lam) on S^3/I* (vectors & scalars) and S^3
2. Exact e-folding plateau & scalaron reheating frequency verification right above x_crit
3. Long-time fate of recollapsing trajectories at Lambda = 0 (x_b > x_crit)
4. Small-sigma_t -> 0+ limit vs exact algebraic pure-Euler (sigma_t = 0) solution
5. Full two-sided (-tau, +tau) asymmetric bounce scan (s_b != 0)
"""

import json
import math
import os
import numpy as np
from scipy.integrate import solve_ivp

from p05_02_solve_and_phase_space import (
    KAPPA_VECTOR,
    SIGMA0_VECTOR,
    KAPPA_SCALAR,
    ode_rhs,
    e00_constraint,
    bounce_initial_state,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)


def integrate_bounce(x_b, sigma_t, lam, kappa, s_b=0.0, tau_max=200.0, direction=+1, x_cap=1e12, blowup_cap=1e5):
    """Integrate forward (+1) or backward (-1) from bounce IC."""
    y0 = bounce_initial_state(x_b, sigma_t, lam, kappa, s_b=s_b, sign_hdot=+1)
    if y0 is None:
        return None

    def ev_blowup(t, y, sig, l):
        return blowup_cap - max(abs(y[1]), abs(y[2]))
    ev_blowup.terminal = True
    ev_blowup.direction = -1

    def ev_collapse(t, y, sig, l):
        return y[0] - 1e-4
    ev_collapse.terminal = True
    ev_collapse.direction = -1

    def ev_xcap(t, y, sig, l):
        return x_cap - y[0]
    ev_xcap.terminal = True
    ev_xcap.direction = -1

    def ev_h0(t, y, sig, l):
        return y[1] if abs(t) > 1e-5 else (1.0 if direction > 0 else -1.0)
    ev_h0.terminal = False
    ev_h0.direction = 0

    sol = solve_ivp(
        ode_rhs,
        (0.0, direction * tau_max),
        y0,
        args=(sigma_t, lam),
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        events=[ev_blowup, ev_collapse, ev_xcap, ev_h0],
        max_step=0.25,
    )
    xs, hs, rs, ss = sol.y
    e00s = np.array([e00_constraint(xs[i], hs[i], rs[i], ss[i], sigma_t, lam, kappa) for i in range(len(sol.t))])
    rel = []
    for i in range(len(sol.t)):
        yi = hs[i]**2 + 1.0 / (xs[i]**2)
        sc = max(1.0, abs(yi), abs(yi**2), abs(sigma_t * rs[i] * (2 * yi - rs[i])), abs(sigma_t * 2 * hs[i] * ss[i]), abs(kappa / xs[i]**4))
        rel.append(abs(e00s[i]) / sc)

    if len(sol.t_events[0]) > 0:
        fate = "blowup_pos" if rs[-1] > 0 else "blowup_neg"
    elif len(sol.t_events[1]) > 0:
        fate = "collapse"
    elif len(sol.t_events[2]) > 0:
        fate = "inflated_cap"
    else:
        fate = "bounded_to_taumax"

    return {
        "fate": fate,
        "t_end": float(sol.t[-1]),
        "x_end": float(xs[-1]),
        "h_end": float(hs[-1]),
        "r_end": float(rs[-1]),
        "h_max": float(np.max(np.abs(hs))),
        "x_max": float(np.max(xs)),
        "x_min": float(np.min(xs)),
        "n_h0_crossings": int(len(sol.t_events[3])),
        "max_rel_e00": float(np.max(rel)),
        "t": sol.t,
        "y": sol.y,
    }


def is_uv_runaway(x_b, sigma_t, lam, kappa, tau_max=60.0):
    """
    Returns True if the bounce at x_b shoots OVER the Starobinsky de Sitter separatrix
    (either h exceeds 1.5 with r > 3, or positive curvature blowup).
    """
    res = integrate_bounce(x_b, sigma_t, lam, kappa, tau_max=tau_max, x_cap=1e15, blowup_cap=1e4)
    if res is None:
        return None
    if res["fate"] == "blowup_pos":
        return True
    if res["h_end"] > 1.5 and res["r_end"] > 3.0:
        return True
    return False


def bisect_separatrix(sigma_t, lam, kappa, x_lo, x_hi, n_iter=45, tau_max=80.0):
    """Bisect the exact Starobinsky UV separatrix x_crit between runaway (x_lo) and graceful exit (x_hi)."""
    f_lo = is_uv_runaway(x_lo, sigma_t, lam, kappa, tau_max=tau_max)
    f_hi = is_uv_runaway(x_hi, sigma_t, lam, kappa, tau_max=tau_max)
    if not (f_lo is True and f_hi is False):
        return {"bracket_failed": True, "f_lo": f_lo, "f_hi": f_hi}
    for _ in range(n_iter):
        mid = 0.5 * (x_lo + x_hi)
        if is_uv_runaway(mid, sigma_t, lam, kappa, tau_max=tau_max):
            x_lo = mid
        else:
            x_hi = mid
    x_crit = 0.5 * (x_lo + x_hi)
    # Check bounce curvature invariants right at x_crit
    y0 = bounce_initial_state(x_crit, sigma_t, lam, kappa)
    hdot_b = float(y0[2] - 1.0 / x_crit**2)
    return {
        "x_crit": x_crit,
        "bracket_width": x_hi - x_lo,
        "hdot_b_at_crit": hdot_b,
        "r_b_at_crit": float(y0[2]),
        "x_crit_over_kappa_quarter": x_crit / (abs(kappa)**0.25) if kappa != 0 else None,
    }


def main():
    out = {}

    # 1. Exact separatrix x_crit(sigma_t, lam=0) for pure vector S^3/I* across sigma_t
    print("1. Bisecting Starobinsky separatrix x_crit(sigma_t, lam=0) for S^3/I* vector...")
    sep_vec_lam0 = []
    # For sigma_t = 0.01 -> 1000, bracket runaway vs non-runaway
    for sig in [0.01, 0.05, 0.1, SIGMA0_VECTOR, 1.0, 3.0, 10.0, 30.0, 100.0, 481.0537634408602, 1000.0]:
        # Find bracket automatically
        grid = np.geomspace(0.3, 50.0, 40)
        lo, hi = None, None
        for xv in grid:
            flag = is_uv_runaway(float(xv), sig, 0.0, KAPPA_VECTOR, tau_max=60.0)
            if flag is True:
                lo = float(xv)
            elif flag is False and lo is not None:
                hi = float(xv)
                break
        if lo is not None and hi is not None:
            b = bisect_separatrix(sig, 0.0, KAPPA_VECTOR, lo, hi, n_iter=42, tau_max=80.0)
        else:
            b = {"bracket_failed": True, "lo": lo, "hi": hi}
        sep_vec_lam0.append({"sigma_t": sig, **b})
        print(f"  sigma={sig:8.4f}: x_crit={b.get('x_crit')}, r_b={b.get('r_b_at_crit')}, hdot_b={b.get('hdot_b_at_crit')}")
    out["separatrix_vector_lam0"] = sep_vec_lam0

    # 2. Pure vector separatrix vs Lambda at sigma_0 = 9/31 (including lam = 0, 1e-4, ..., 0.24)
    print("\n2. Bisecting x_crit(lam) at sigma_0=9/31 for S^3/I* vector...")
    sep_vec_lam = []
    for lam in [0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.10, 0.15, 0.20, 0.24]:
        b = bisect_separatrix(SIGMA0_VECTOR, lam, KAPPA_VECTOR, 2.0, 5.5, n_iter=42, tau_max=80.0)
        kappa_eff = 1.0 - SIGMA0_VECTOR + KAPPA_VECTOR
        x_max = math.sqrt((1.0 + math.sqrt(1.0 - 4.0 * lam * kappa_eff)) / (2.0 * lam)) if lam > 0 else None
        sep_vec_lam.append({"lam": lam, "x_b_max": x_max, **b})
        print(f"  lam={lam:6.4f}: x_crit={b.get('x_crit'):.8f}, x_max={x_max}")
    out["separatrix_vector_sigma0_vs_lam"] = sep_vec_lam

    # 3. Pure conformal scalar separatrix at lam = 0 for sigma_t in {0.1, 1.0, 10.0, 100.0, 1000.0}
    print("\n3. Bisecting x_crit(sigma_t, lam=0) for S^3/I* conformal scalar...")
    sep_scal = []
    for sig in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        grid = np.geomspace(1.0, 200.0, 40)
        lo, hi = None, None
        for xv in grid:
            flag = is_uv_runaway(float(xv), sig, 0.0, KAPPA_SCALAR, tau_max=60.0)
            if flag is True:
                lo = float(xv)
            elif flag is False and lo is not None:
                hi = float(xv)
                break
        if lo is not None and hi is not None:
            b = bisect_separatrix(sig, 0.0, KAPPA_SCALAR, lo, hi, n_iter=42, tau_max=80.0)
        else:
            b = {"bracket_failed": True, "lo": lo, "hi": hi}
        sep_scal.append({"sigma_t": sig, **b})
        print(f"  sigma={sig:8.2f}: x_crit={b.get('x_crit')}, x_crit/|kappa|^(1/4)={b.get('x_crit_over_kappa_quarter')}")
    out["separatrix_scalar_lam0"] = sep_scal

    # 4. Exact Starobinsky inflationary plateau right above x_crit (vector, sigma_0=9/31, lam=0 and lam=1e-4)
    print("\n4. Testing Starobinsky inflationary plateau & scalaron reheating near x_crit...")
    xc0 = sep_vec_lam[0]["x_crit"]
    plateau_runs = []
    for eps in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
        res = integrate_bounce(xc0 + eps, SIGMA0_VECTOR, 0.0, KAPPA_VECTOR, tau_max=100.0, x_cap=1e20)
        t, xs, hs, rs = res["t"], res["y"][0], res["y"][1], res["y"][2]
        # Count e-folds while h > 0.5 (Starobinsky inflationary plateau)
        mask = hs > 0.5
        if np.any(mask):
            idx = np.where(mask)[0]
            efolds_plateau = float(np.log(xs[idx[-1]] / xs[idx[0]]))
            tau_plateau = float(t[idx[-1]] - t[idx[0]])
        else:
            efolds_plateau = 0.0
            tau_plateau = 0.0
        # Measure scalaron oscillation period after exit (t > t_exit + 5, r zero crossings)
        t_exit = float(t[idx[-1]]) if np.any(mask) else 0.0
        post = t > t_exit + 5.0
        osc_freq_measured = None
        if np.sum(post) > 20:
            tp, rp = t[post], rs[post]
            zc = np.where(np.diff(np.sign(rp)) != 0)[0]
            if len(zc) >= 4:
                half_periods = np.diff(tp[zc])
                mean_T = 2.0 * float(np.mean(half_periods))
                osc_freq_measured = 2.0 * math.pi / mean_T
        plateau_runs.append({
            "eps": eps,
            "x_b": xc0 + eps,
            "efolds_plateau": efolds_plateau,
            "tau_plateau": tau_plateau,
            "x_end": res["x_end"],
            "h_end": res["h_end"],
            "max_rel_e00": res["max_rel_e00"],
            "osc_freq_measured": osc_freq_measured,
            "osc_freq_theory_M_over_HS": 1.0 / math.sqrt(SIGMA0_VECTOR),
        })
        print(f"  eps={eps:.0e}: N_e={efolds_plateau:.3f}, tau_plat={tau_plateau:.3f}, omega_osc={osc_freq_measured} (theory {1/math.sqrt(SIGMA0_VECTOR):.4f}), e00={res['max_rel_e00']:.2e}")
    out["starobinsky_plateau_near_xcrit_lam0"] = plateau_runs

    # 5. Long-time fate of recollapsing Lambda=0 bounces (x_b = 5, 10, 22, 50)
    print("\n5. Tracking long-time fate of Lambda=0 recollapsing bounces...")
    recollapse_fates = []
    for xb in [4.5, 5.0, 10.0, 22.0, 50.0]:
        res = integrate_bounce(xb, SIGMA0_VECTOR, 0.0, KAPPA_VECTOR, tau_max=600.0, x_cap=1e12, blowup_cap=1e5)
        t, xs, hs, rs = res["t"], res["y"][0], res["y"][1], res["y"][2]
        # Find first turnaround time and x_max
        recollapse_fates.append({
            "x_b": xb,
            "fate": res["fate"],
            "t_end": res["t_end"],
            "x_max": res["x_max"],
            "x_min": res["x_min"],
            "x_end": res["x_end"],
            "h_end": res["h_end"],
            "r_end": res["r_end"],
            "n_h0_crossings": res["n_h0_crossings"],
            "max_rel_e00": res["max_rel_e00"],
        })
        print(f"  x_b={xb:5.1f}: fate={res['fate']:18s}, t_end={res['t_end']:7.2f}, x_max={res['x_max']:8.2f}, x_end={res['x_end']:.3e}, h_end={res['h_end']:+.3e}, r_end={res['r_end']:+.3e}, n_h0={res['n_h0_crossings']}")
    out["long_time_recollapse_lam0"] = recollapse_fates

    # 6. Double-sided asymmetric bounces (s_b != 0) at lam = 0.01 (eternal IR de Sitter) and lam = 0
    print("\n6. Testing two-sided (-tau, +tau) asymmetric bounces (s_b != 0)...")
    asym_results = []
    for lam, xb in [(0.01, 10.0), (0.01, 5.0), (0.0, 5.0)]:
        for sb in [0.0, 0.1, 1.0, 5.0]:
            fwd = integrate_bounce(xb, SIGMA0_VECTOR, lam, KAPPA_VECTOR, s_b=sb, tau_max=150.0, direction=+1)
            bwd = integrate_bounce(xb, SIGMA0_VECTOR, lam, KAPPA_VECTOR, s_b=sb, tau_max=150.0, direction=-1)
            asym_results.append({
                "lam": lam, "x_b": xb, "s_b": sb,
                "fwd_fate": fwd["fate"], "fwd_t_end": fwd["t_end"], "fwd_h_end": fwd["h_end"], "fwd_e00": fwd["max_rel_e00"],
                "bwd_fate": bwd["fate"], "bwd_t_end": bwd["t_end"], "bwd_h_end": bwd["h_end"], "bwd_e00": bwd["max_rel_e00"],
            })
            print(f"  lam={lam}, x_b={xb}, s_b={sb:4.1f}: fwd={fwd['fate']} (h={fwd['h_end']:+.3f}), bwd={bwd['fate']} (h={bwd['h_end']:+.3f})")
    out["two_sided_asymmetric_bounces"] = asym_results

    out_path = os.path.join(OUT_DIR, "p05_02b_deep_dynamics.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
