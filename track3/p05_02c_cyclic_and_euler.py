#!/usr/bin/env python3
"""
p05_02c_cyclic_and_euler.py
===========================
Test the discovered Lambda = 0 scalaron-dust + negative-Casimir cyclic breathing
universe on S^3/I* vs S^3 (kappa=0), and verify its long-time stability / secular drift
and exact effective-potential law x_min * x_max ~ |kappa_eff|.
"""

import json
import math
import os
import numpy as np

from p05_02_solve_and_phase_space import KAPPA_VECTOR, SIGMA0_VECTOR, KAPPA_SCALAR
from p05_02b_deep_dynamics import integrate_bounce

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)


def analyze_breathing(x_b, sigma_t, kappa, tau_max=3000.0):
    res = integrate_bounce(x_b, sigma_t, 0.0, kappa, s_b=0.0, tau_max=tau_max, x_cap=1e12, blowup_cap=1e5)
    t, xs, hs, rs = res["t"], res["y"][0], res["y"][1], res["y"][2]

    # Smooth macro-envelope by taking local extrema of x(t) separated by > 20 in tau
    # Better: compute moving average over 1 scalaron period T_osc = 2*pi*sqrt(sigma_t)
    T_osc = 2.0 * math.pi * math.sqrt(sigma_t)
    # Sample onto uniform grid for clean envelope extrema
    t_unif = np.linspace(0.0, t[-1], min(60000, max(2000, int(t[-1] * 20))))
    x_unif = np.interp(t_unif, t, xs)
    dt = t_unif[1] - t_unif[0]
    win = max(3, int(round(T_osc / dt)))
    kernel = np.ones(win) / win
    x_bar = np.convolve(x_unif, kernel, mode="valid")
    t_bar = t_unif[win // 2 : win // 2 + len(x_bar)]

    # Find macro turning points of x_bar
    dx = np.diff(x_bar)
    macro_max = []
    macro_min = []
    for i in range(1, len(dx)):
        if dx[i - 1] > 0 and dx[i] <= 0:
            macro_max.append((float(t_bar[i]), float(x_bar[i])))
        elif dx[i - 1] < 0 and dx[i] >= 0:
            macro_min.append((float(t_bar[i]), float(x_bar[i])))

    kappa_eff = abs(1.0 - sigma_t + kappa)
    predicted_partner = kappa_eff / x_b

    return {
        "x_b": x_b,
        "kappa": kappa,
        "sigma_t": sigma_t,
        "fate": res["fate"],
        "t_end": res["t_end"],
        "x_min_global": res["x_min"],
        "x_max_global": res["x_max"],
        "x_end": res["x_end"],
        "max_rel_e00": res["max_rel_e00"],
        "predicted_partner_kappa_eff_over_xb": predicted_partner,
        "n_macro_max": len(macro_max),
        "n_macro_min": len(macro_min),
        "first_5_macro_max": macro_max[:5],
        "last_3_macro_max": macro_max[-3:],
        "first_5_macro_min": macro_min[:5],
        "last_3_macro_min": macro_min[-3:],
    }


def main():
    results = {"S3_Istar_vector_cyclic_test": [], "S3_round_control_lam0": []}

    print("=== S^3/I* Vector (sigma_0=9/31, lam=0, tau_max=3000) ===")
    for xb in [4.5, 5.0, 6.0, 7.0, 10.0, 15.0, 21.926, 30.0, 50.0]:
        r = analyze_breathing(xb, SIGMA0_VECTOR, KAPPA_VECTOR, tau_max=3000.0)
        results["S3_Istar_vector_cyclic_test"].append(r)
        print(
            f"  x_b={xb:6.2f} (partner={r['predicted_partner_kappa_eff_over_xb']:6.2f}): "
            f"fate={r['fate']:17s} t_end={r['t_end']:7.1f} x_range=[{r['x_min_global']:6.2f}, {r['x_max_global']:7.2f}] "
            f"n_cycles={r['n_macro_max']:2d} e00={r['max_rel_e00']:.2e}"
        )
        if r["first_5_macro_max"]:
            print(f"     maxima: {[(round(a,1), round(b,2)) for a,b in r['first_5_macro_max'][:3]]} ... {[(round(a,1), round(b,2)) for a,b in r['last_3_macro_max'][-2:]]}")
        if r["first_5_macro_min"]:
            print(f"     minima: {[(round(a,1), round(b,2)) for a,b in r['first_5_macro_min'][:3]]} ... {[(round(a,1), round(b,2)) for a,b in r['last_3_macro_min'][-2:]]}")

    print("\n=== S^3 Round Control (kappa=0, sigma_0=9/31, lam=0, tau_max=1000) ===")
    for xb in [1.0 + 1e-4, 1.01, 1.1, 2.0, 5.0, 10.0, 22.0]:
        r = analyze_breathing(xb, SIGMA0_VECTOR, 0.0, tau_max=1000.0)
        results["S3_round_control_lam0"].append(r)
        print(
            f"  x_b={xb:7.4f}: fate={r['fate']:17s} t_end={r['t_end']:7.1f} "
            f"x_range=[{r['x_min_global']:6.3f}, {r['x_max_global']:9.2f}] n_cycles={r['n_macro_max']:2d} e00={r['max_rel_e00']:.2e}"
        )

    # Also test conformal scalar at sigma_t = 10.0, x_b = sqrt(|kappa_eff|) = 240.96
    print("\n=== S^3/I* Conformal Scalar (sigma=10, lam=0, tau_max=1500) ===")
    r_sc = analyze_breathing(240.96, 10.0, KAPPA_SCALAR, tau_max=1500.0)
    results["S3_Istar_scalar_cyclic_test"] = r_sc
    print(
        f"  x_b=240.96: fate={r_sc['fate']} t_end={r_sc['t_end']:.1f} "
        f"x_range=[{r_sc['x_min_global']:.2f}, {r_sc['x_max_global']:.2f}] e00={r_sc['max_rel_e00']:.2e}"
    )

    out_path = os.path.join(OUT_DIR, "p05_02c_cyclic_breathing.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
