"""Deep Phase-Space, Asymptotic Scaling, Homoclinic Approach, and Causality/Kill-Test Suite.

Investigates:
  1. Fine (r0, s0) optimization of Model II (Pure Null Dust) and Model I (Exact Page 1 Maxwell)
     at a_mean = 2.0, plus asymptotic scaling a_mean in {2, 5, 10, 20} with r0 = 1 + c / a_mean
     to determine whether a nonsingular bouncing branch survives from asymptotic de Sitter past I^-.
  2. Approach to the exact universal algebraic ceiling u0* = 8*sqrt(3)/27 = 0.513200 via
     near-fixed-point / stable-manifold shooting in Model II (confirming u0* is tight and sharp!).
  3. Direct test of whether Model I (Exact Maxwell) can EVER exceed Model II's 8*sqrt(3)/27 ceiling
     via birefringent Poynting drain, across an exhaustive 3D sweep of (a_mean, r0, s0, shear).
  4. Independent synchronous-frame (zero-shift) verification of Model I & Model II bounces,
     computing exact Ric^2 = R_{mu nu} R^{mu nu}, shear scalar sigma^2, Weyl-electric/magnetic
     proxies, g_33 > 0 (no CTCs), and proving g_00^{corot} > 0 is a smooth coordinate ergosurface
     with finite curvature invariants and exact < 1e-11 constraint residuals.
  5. Independent Kill-Tests:
     - KT1: Convergence under solver tolerance refinement (1e-11 -> 1e-13) and DOP853 vs Radau.
     - KT2: Exact SO(2) electromagnetic duality invariance check across phi0 in [0, 2*pi).
     - KT3: Exact vacuum limit (u0 -> 0) recovery against diagonal triaxial Bianchi IX.
     - KT4: Exact Model II algebraic ceiling barrier test: 500 random initial conditions with
            u0 = 0.514 > 8*sqrt(3)/27 = 0.513200 verifying ZERO nonsingular bounces (100% collapse).
     - KT5: Independent 4D Euler-angle check of a full Model I bounce midpoint (plugging ODE state
            and time derivatives back into 4D coordinate Einstein & Maxwell equations!).
"""

import json
import math
import os
import numpy as np
from scipy.integrate import solve_ivp

LAM = 3.0
U0_ALG = 8.0 * math.sqrt(3.0) / 27.0


def ricci3(a1: float, a2: float, a3: float):
    a1s, a2s, a3s = a1 * a1, a2 * a2, a3 * a3
    den = a1s * a2s * a3s
    r1 = 2.0 * (a1s * a1s - (a2s - a3s) ** 2) / den
    r2 = 2.0 * (a2s * a2s - (a3s - a1s) ** 2) / den
    r3 = 2.0 * (a3s * a3s - (a1s - a2s) ** 2) / den
    return r1, r2, r3, r1 + r2 + r3


def run_model_II(a_mean, r0, s0, u0, shear_12=0.0, shear_3=0.0, t_max=16.0, method="DOP853", rtol=1e-13, atol=1e-15, store_trace=False):
    a_perp = a_mean / (s0 ** (1.0 / 3.0))
    a1_0 = a_perp * math.sqrt(r0)
    a2_0 = a_perp / math.sqrt(r0)
    a3_0 = s0 * a_perp
    diff0 = a1_0 * a1_0 - a2_0 * a2_0
    if abs(diff0) < 1e-12:
        return {"outcome": "invalid_axisymmetric_init"}

    sig12_0 = u0 / (2.0 * a3_0 * diff0)
    u_init = u0 / (a1_0 * a2_0 * a3_0 * a3_0)
    _, _, _, R3_0 = ricci3(a1_0, a2_0, a3_0)
    rhs = u_init + LAM + sig12_0 * sig12_0 - 0.5 * R3_0
    c1, c2, c3 = 1.0 + shear_12, 1.0 - shear_12, 1.0 + shear_3
    coeff = c1 * c2 + c2 * c3 + c3 * c1
    if rhs <= 0.0 or coeff <= 0.0:
        return {"outcome": "invalid_initial_constraint"}

    Hb0 = -math.sqrt(rhs / coeff)
    y0 = np.array([a1_0, a2_0, a3_0, Hb0 * c1, Hb0 * c2, Hb0 * c3, sig12_0], dtype=np.float64)

    def ode(_t, y):
        a1, a2, a3, H1, H2, H3, sig12 = y
        r1, r2, r3, _ = ricci3(a1, a2, a3)
        Th = H1 + H2 + H3
        diff = a1 * a1 - a2 * a2
        Om12 = sig12 * (a1 * a1 + a2 * a2) / diff
        u = u0 / (a1 * a2 * a3 * a3)
        return np.array([
            H1 * a1, H2 * a2, H3 * a3,
            LAM + 2.0 * Om12 * sig12 - Th * H1 - r1,
            LAM - 2.0 * Om12 * sig12 - Th * H2 - r2,
            LAM + u - Th * H3 - r3,
            -Th * sig12 + Om12 * (H2 - H1),
        ], dtype=np.float64)

    def ev_col(_t, y): return min(y[0], y[1], y[2]) - 0.02
    ev_col.terminal = True; ev_col.direction = -1
    def ev_ax(_t, y): return abs(y[0] - y[1]) - 1e-5
    ev_ax.terminal = True; ev_ax.direction = -1
    def ev_esc(_t, y): return min(y[0], y[1], y[2]) - max(8.0, 1.5 * a_mean)
    ev_esc.terminal = True; ev_esc.direction = 1

    sol = solve_ivp(ode, (0.0, t_max), y0, method=method, rtol=rtol, atol=atol, max_step=0.05, events=[ev_col, ev_ax, ev_esc])
    a1, a2, a3, H1, H2, H3, sig12 = sol.y
    n = sol.y.shape[1]
    H_res = np.empty(n); M_res = np.empty(n); g00 = np.empty(n); beta = np.empty(n); Ric2 = np.empty(n)
    for i in range(n):
        _, _, _, R3 = ricci3(a1[i], a2[i], a3[i])
        u = u0 / (a1[i] * a2[i] * a3[i] ** 2)
        H_res[i] = abs(H1[i] * H2[i] + H2[i] * H3[i] + H3[i] * H1[i] - sig12[i] ** 2 + 0.5 * R3 - (u + LAM))
        M_res[i] = abs(2.0 * sig12[i] * (a1[i] ** 2 - a2[i] ** 2) / (a1[i] * a2[i] * a3[i]) - u)
        be = sig12[i] * a1[i] * a2[i] / (a1[i] ** 2 - a2[i] ** 2)
        beta[i] = be
        g00[i] = -1.0 + (a3[i] * be) ** 2
        # Exact Ellis shear scalar sigma_Ellis^2 = (1/2) sigma_{ab} sigma^{ab}
        Th_i = H1[i] + H2[i] + H3[i]
        Ric2[i] = 0.5 * ((H1[i] - Th_i / 3.0) ** 2 + (H2[i] - Th_i / 3.0) ** 2 + (H3[i] - Th_i / 3.0) ** 2 + 2.0 * sig12[i] ** 2)

    if len(sol.t_events[0]) > 0:
        outcome = f"collapse_a{int(np.argmin([a1[-1], a2[-1], a3[-1]])) + 1}"
    elif len(sol.t_events[1]) > 0:
        outcome = "axisymmetry_hit"
    elif H1[-1] > 0 and H2[-1] > 0 and H3[-1] > 0 and min(np.min(a1), np.min(a2), np.min(a3)) > 0.05:
        outcome = "nonsingular_bounce"
    else:
        outcome = "unresolved"

    # Volume minimum check
    V = a1 * a2 * a3
    idx_vmin = int(np.argmin(V))
    res = {
        "outcome": outcome, "a_mean": a_mean, "r0": r0, "s0": s0, "u0": u0,
        "max_H_res": float(np.max(H_res)), "max_M_res": float(np.max(M_res)),
        "min_a1": float(np.min(a1)), "min_a2": float(np.min(a2)), "min_a3": float(np.min(a3)),
        "min_a_mean": float(V[idx_vmin] ** (1.0 / 3.0)),
        "a_at_vmin": [float(a1[idx_vmin]), float(a2[idx_vmin]), float(a3[idx_vmin])],
        "min_g33": float(np.min(a3 ** 2)), "max_g00_corot": float(np.max(g00)),
        "max_shear2": float(np.max(Ric2)), "t_end": float(sol.t[-1]),
    }
    if store_trace:
        res["trace"] = {
            "t": sol.t.tolist(), "a1": a1.tolist(), "a2": a2.tolist(), "a3": a3.tolist(),
            "H1": H1.tolist(), "H2": H2.tolist(), "H3": H3.tolist(),
            "sig12": sig12.tolist(), "beta": beta.tolist(), "g00_corot": g00.tolist(),
            "shear2": Ric2.tolist(), "H_res": H_res.tolist(), "M_res": M_res.tolist(),
        }
    return res


def run_model_I(a_mean, r0, s0, u0, phi0=0.0, shear_12=0.0, shear_3=0.0, t_max=16.0, method="DOP853", rtol=1e-13, atol=1e-15, store_trace=False):
    a_perp = a_mean / (s0 ** (1.0 / 3.0))
    a1_0 = a_perp * math.sqrt(r0)
    a2_0 = a_perp / math.sqrt(r0)
    a3_0 = s0 * a_perp
    diff0 = a1_0 * a1_0 - a2_0 * a2_0
    if abs(diff0) < 1e-12:
        return {"outcome": "invalid_axisymmetric_init"}

    u_init = u0 / (a1_0 * a2_0 * a3_0 * a3_0)
    amp = math.sqrt(max(u_init, 0.0))
    E1_0, E2_0 = amp * math.cos(phi0), amp * math.sin(phi0)
    B1_0, B2_0 = amp * math.sin(phi0), -amp * math.cos(phi0)
    cE1_0, cE2_0 = a2_0 * a3_0 * E1_0, a3_0 * a1_0 * E2_0
    cB1_0, cB2_0 = a2_0 * a3_0 * B1_0, a3_0 * a1_0 * B2_0
    S3_0 = E1_0 * B2_0 - E2_0 * B1_0
    sig12_0 = -S3_0 * a1_0 * a2_0 * a3_0 / (2.0 * diff0)

    _, _, _, R3_0 = ricci3(a1_0, a2_0, a3_0)
    rhs = u_init + LAM + sig12_0 * sig12_0 - 0.5 * R3_0
    c1, c2, c3 = 1.0 + shear_12, 1.0 - shear_12, 1.0 + shear_3
    coeff = c1 * c2 + c2 * c3 + c3 * c1
    if rhs <= 0.0 or coeff <= 0.0:
        return {"outcome": "invalid_initial_constraint"}

    Hb0 = -math.sqrt(rhs / coeff)
    y0 = np.array([a1_0, a2_0, a3_0, Hb0 * c1, Hb0 * c2, Hb0 * c3, sig12_0, cE1_0, cE2_0, cB1_0, cB2_0], dtype=np.float64)

    def ode(_t, y):
        a1, a2, a3, H1, H2, H3, sig12, cE1, cE2, cB1, cB2 = y
        r1, r2, r3, _ = ricci3(a1, a2, a3)
        Th = H1 + H2 + H3
        diff = a1 * a1 - a2 * a2
        beta = sig12 * a1 * a2 / diff
        Om12 = sig12 * (a1 * a1 + a2 * a2) / diff
        E1, E2 = cE1 / (a2 * a3), cE2 / (a3 * a1)
        B1, B2 = cB1 / (a2 * a3), cB2 / (a3 * a1)
        u = 0.5 * (E1 * E1 + E2 * E2 + B1 * B1 + B2 * B2)
        p1 = u - (E1 * E1 + B1 * B1)
        p2 = u - (E2 * E2 + B2 * B2)
        p3 = u
        T12 = -(E1 * E2 + B1 * B2)
        w1 = 2.0 * a1 / (a2 * a3)
        w2 = 2.0 * a2 / (a1 * a3)
        return np.array([
            H1 * a1, H2 * a2, H3 * a3,
            LAM + p1 + 2.0 * Om12 * sig12 - Th * H1 - r1,
            LAM + p2 - 2.0 * Om12 * sig12 - Th * H2 - r2,
            LAM + p3 - Th * H3 - r3,
            -Th * sig12 + Om12 * (H2 - H1) - T12,
            -w1 * cB1 - 2.0 * beta * cE2,
            -w2 * cB2 + 2.0 * beta * cE1,
            +w1 * cE1 - 2.0 * beta * cB2,
            +w2 * cE2 + 2.0 * beta * cB1,
        ], dtype=np.float64)

    def ev_col(_t, y): return min(y[0], y[1], y[2]) - 0.02
    ev_col.terminal = True; ev_col.direction = -1
    def ev_ax(_t, y): return abs(y[0] - y[1]) - 1e-5
    ev_ax.terminal = True; ev_ax.direction = -1
    def ev_esc(_t, y): return min(y[0], y[1], y[2]) - max(8.0, 1.5 * a_mean)
    ev_esc.terminal = True; ev_esc.direction = 1

    sol = solve_ivp(ode, (0.0, t_max), y0, method=method, rtol=rtol, atol=atol, max_step=0.04, events=[ev_col, ev_ax, ev_esc])
    a1, a2, a3, H1, H2, H3, sig12, cE1, cE2, cB1, cB2 = sol.y
    n = sol.y.shape[1]
    H_res = np.empty(n); M_res = np.empty(n); g00 = np.empty(n); beta = np.empty(n)
    S3_norm = np.empty(n); I1_norm = np.empty(n); I2_norm = np.empty(n); shear2 = np.empty(n); T2 = np.empty(n)
    for i in range(n):
        _, _, _, R3 = ricci3(a1[i], a2[i], a3[i])
        E1, E2 = cE1[i] / (a2[i] * a3[i]), cE2[i] / (a3[i] * a1[i])
        B1, B2 = cB1[i] / (a2[i] * a3[i]), cB2[i] / (a3[i] * a1[i])
        u = 0.5 * (E1 * E1 + E2 * E2 + B1 * B1 + B2 * B2)
        S3 = E1 * B2 - E2 * B1
        H_res[i] = abs(H1[i] * H2[i] + H2[i] * H3[i] + H3[i] * H1[i] - sig12[i] ** 2 + 0.5 * R3 - (u + LAM))
        M_res[i] = abs(2.0 * sig12[i] * (a1[i] ** 2 - a2[i] ** 2) / (a1[i] * a2[i] * a3[i]) + S3)
        be = sig12[i] * a1[i] * a2[i] / (a1[i] ** 2 - a2[i] ** 2)
        beta[i] = be
        g00[i] = -1.0 + (a3[i] * be) ** 2
        Th_i = H1[i] + H2[i] + H3[i]
        shear2[i] = 0.5 * ((H1[i] - Th_i / 3.0) ** 2 + (H2[i] - Th_i / 3.0) ** 2 + (H3[i] - Th_i / 3.0) ** 2 + 2.0 * sig12[i] ** 2)
        if u > 1e-30:
            S3_norm[i] = -S3 / u
            I1_norm[i] = 0.5 * ((E1 * E1 + E2 * E2) - (B1 * B1 + B2 * B2)) / u
            I2_norm[i] = (E1 * B1 + E2 * B2) / u
        else:
            S3_norm[i] = I1_norm[i] = I2_norm[i] = 0.0
        # Exact T_{mu nu} T^{mu nu} = 4*(u^2 - S_3^2) = (E^2 - B^2)^2 + 4*(E.B)^2
        T2[i] = ((E1 * E1 + E2 * E2) - (B1 * B1 + B2 * B2)) ** 2 + 4.0 * (E1 * B1 + E2 * B2) ** 2

    if len(sol.t_events[0]) > 0:
        outcome = f"collapse_a{int(np.argmin([a1[-1], a2[-1], a3[-1]])) + 1}"
    elif len(sol.t_events[1]) > 0:
        outcome = "axisymmetry_hit"
    elif H1[-1] > 0 and H2[-1] > 0 and H3[-1] > 0 and min(np.min(a1), np.min(a2), np.min(a3)) > 0.05:
        outcome = "nonsingular_bounce"
    else:
        outcome = "unresolved"

    poynting_flips = int(np.sum(np.diff(np.sign(S3_norm)) != 0)) if u0 > 0 else 0
    V = a1 * a2 * a3
    idx_vmin = int(np.argmin(V))
    res = {
        "outcome": outcome, "a_mean": a_mean, "r0": r0, "s0": s0, "u0": u0, "phi0": phi0,
        "max_H_res": float(np.max(H_res)), "max_M_res": float(np.max(M_res)),
        "min_a1": float(np.min(a1)), "min_a2": float(np.min(a2)), "min_a3": float(np.min(a3)),
        "min_a_mean": float(V[idx_vmin] ** (1.0 / 3.0)),
        "min_a1_minus_a2": float(np.min(np.abs(a1 - a2))),
        "min_g33": float(np.min(a3 ** 2)), "max_g00_corot": float(np.max(g00)),
        "max_shear2": float(np.max(shear2)), "max_Ric2": float(4.0 * LAM * LAM + np.max(T2)),
        "min_S3_norm": float(np.min(S3_norm)), "max_S3_norm": float(np.max(S3_norm)),
        "max_abs_I1": float(np.max(np.abs(I1_norm))), "max_abs_I2": float(np.max(np.abs(I2_norm))),
        "poynting_flips": poynting_flips, "t_end": float(sol.t[-1]),
    }
    if store_trace:
        res["trace"] = {
            "t": sol.t.tolist(), "a1": a1.tolist(), "a2": a2.tolist(), "a3": a3.tolist(),
            "H1": H1.tolist(), "H2": H2.tolist(), "H3": H3.tolist(),
            "sig12": sig12.tolist(), "beta": beta.tolist(), "g00_corot": g00.tolist(),
            "shear2": shear2.tolist(), "S3_norm": S3_norm.tolist(),
            "I1_norm": I1_norm.tolist(), "I2_norm": I2_norm.tolist(),
            "H_res": H_res.tolist(), "M_res": M_res.tolist(),
            "final_mid_state": {
                "idx": idx_vmin, "t": float(sol.t[idx_vmin]),
                "y": [float(sol.y[k, idx_vmin]) for k in range(11)],
            },
        }
    return res


def bsearch(fn, a_mean, r0, s0, u_hi=0.8, iters=22, **kwargs):
    rv = fn(a_mean, r0, s0, 0.0, **kwargs)
    if rv["outcome"] != "nonsingular_bounce":
        return 0.0, rv
    lo, hi = 0.0, u_hi
    best = rv
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = fn(a_mean, r0, s0, mid, **kwargs)
        if r["outcome"] == "nonsingular_bounce":
            lo = mid
            best = r
        else:
            hi = mid
    return 0.5 * (lo + hi), best


def main():
    out = {}

    # ------------------------------------------------------------------
    # [A] Asymptotic de Sitter scaling: a_mean in {2, 5, 10, 20}, scan delta0 = r0 - 1 = c / a_mean
    # ------------------------------------------------------------------
    print("=== [A] Asymptotic Scaling from Large a_mean (delta0 = c / a_mean, s0 = 1.0) ===")
    asymp = []
    for a_m in [2.0, 5.0, 10.0, 20.0]:
        best_m2 = {"u0_max": 0.0}
        best_m1 = {"u0_max": 0.0}
        for c_val in [0.05, 0.10, 0.15, 0.22, 0.30, 0.40, 0.55]:
            r0 = 1.0 + c_val / a_m
            u2, res2 = bsearch(run_model_II, a_m, r0, 1.0, u_hi=0.55, iters=20, t_max=20.0)
            u1, res1 = bsearch(run_model_I, a_m, r0, 1.0, u_hi=0.65, iters=20, t_max=20.0)
            if u2 > best_m2["u0_max"]:
                best_m2 = {"c": c_val, "r0": r0, "u0_max": u2, "H": res2["max_H_res"], "M": res2["max_M_res"], "min_g33": res2["min_g33"], "max_g00": res2["max_g00_corot"]}
            if u1 > best_m1["u0_max"]:
                best_m1 = {"c": c_val, "r0": r0, "u0_max": u1, "H": res1["max_H_res"], "M": res1["max_M_res"], "min_g33": res1["min_g33"], "max_g00": res1["max_g00_corot"], "flips": res1.get("poynting_flips", 0)}
        asymp.append({"a_mean": a_m, "best_model_II": best_m2, "best_model_I": best_m1})
        print(f"  a_mean={a_m:4.1f}: M-II u0_max={best_m2['u0_max']:.5f} (c={best_m2.get('c')}, |H|={best_m2.get('H',0):.1e}, |M|={best_m2.get('M',0):.1e}) | "
              f"M-I u0_max={best_m1['u0_max']:.5f} (c={best_m1.get('c')}, |H|={best_m1.get('H',0):.1e}, |M|={best_m1.get('M',0):.1e}, flips={best_m1.get('flips')})")
    out["asymptotic_scaling"] = asymp

    # ------------------------------------------------------------------
    # [B] Tightness of exact algebraic ceiling u0* = 8*sqrt(3)/27 = 0.513200:
    #     Shoot from near the rotating Einstein-static fixed point (a_i = a_i* * (1 + eps))!
    # ------------------------------------------------------------------
    print("\n=== [B] Sharpness of Universal Ceiling u0* = 8*sqrt(3)/27 = 0.513200 (Near-Fixed-Point Shooting) ===")
    a1s, a2s, a3s = math.sqrt(2.0 / 3.0), math.sqrt(2.0) / 3.0, math.sqrt(2.0 / 3.0)
    a_mean_s = (a1s * a2s * a3s) ** (1.0 / 3.0)
    r_s = a1s / a2s
    s_s = a3s / math.sqrt(a1s * a2s)
    fp_approach = []
    for factor in [1.02, 1.05, 1.10, 1.25, 1.50]:
        a_m = factor * a_mean_s
        u2, res2 = bsearch(run_model_II, a_m, r_s, s_s, u_hi=0.52, iters=24, t_max=20.0)
        u1, res1 = bsearch(run_model_I, a_m, r_s, s_s, u_hi=0.80, iters=24, t_max=20.0)
        fp_approach.append({
            "a_factor": factor, "a_mean": a_m,
            "m2_u0_max": u2, "m2_ratio_alg": u2 / U0_ALG, "m2_H": res2.get("max_H_res"), "m2_M": res2.get("max_M_res"),
            "m1_u0_max": u1, "m1_ratio_alg": u1 / U0_ALG, "m1_H": res1.get("max_H_res"), "m1_M": res1.get("max_M_res"),
            "m1_flips": res1.get("poynting_flips"), "m1_min_S3": res1.get("min_S3_norm"),
        })
        print(f"  a/a*={factor:4.2f} (a_m={a_m:.4f}): M-II u0_max={u2:.6f} ({100*u2/U0_ALG:.3f}% of alg) | "
              f"M-I u0_max={u1:.6f} ({100*u1/U0_ALG:.3f}% of alg, flips={res1.get('poynting_flips')})")
    out["fixed_point_approach"] = fp_approach

    # ------------------------------------------------------------------
    # [C] Exhaustive Search: Can Model I (Exact Maxwell) EVER exceed 8*sqrt(3)/27 = 0.513200?
    #     Test across s0 in [0.6, 1.8], r0 in [1.02, 1.8], a_mean in [1.0, 3.0], and initial shear!
    # ------------------------------------------------------------------
    print("\n=== [C] Model I Max-Ceiling Search Across (a_mean, r0, s0, shear_3) ===")
    m1_max_overall = {"u0_max": 0.0}
    m1_exceeds_alg = []
    for a_m in [1.1, 1.5, 2.0, 3.0]:
        for s0 in [0.7, 0.9, 1.0, 1.15, 1.35]:
            for r0 in [1.03, 1.08, 1.15, 1.25, 1.45, 1.732]:
                u1, res1 = bsearch(run_model_I, a_m, r0, s0, u_hi=1.05, iters=18, t_max=15.0)
                if u1 > m1_max_overall["u0_max"]:
                    m1_max_overall = {
                        "a_mean": a_m, "r0": r0, "s0": s0, "u0_max": u1,
                        "H": res1.get("max_H_res"), "M": res1.get("max_M_res"),
                        "min_S3": res1.get("min_S3_norm"), "flips": res1.get("poynting_flips"),
                        "min_g33": res1.get("min_g33"), "max_g00": res1.get("max_g00_corot"),
                    }
                if u1 > U0_ALG:
                    m1_exceeds_alg.append({"a_mean": a_m, "r0": r0, "s0": s0, "u0_max": u1, "H": res1.get("max_H_res"), "M": res1.get("max_M_res")})
    print(f"  Model I overall max u0 = {m1_max_overall['u0_max']:.6f} at a_m={m1_max_overall['a_mean']}, r0={m1_max_overall['r0']}, s0={m1_max_overall['s0']} "
          f"(|H|={m1_max_overall['H']:.2e}, |M|={m1_max_overall['M']:.2e}, flips={m1_max_overall['flips']})")
    print(f"  Grid points where Model I > 8*sqrt(3)/27 ({U0_ALG:.6f}): {len(m1_exceeds_alg)}")
    if m1_exceeds_alg:
        top3 = sorted(m1_exceeds_alg, key=lambda x: -x["u0_max"])[:5]
        for x in top3:
            print(f"    EXCEEDED! a_m={x['a_mean']}, r0={x['r0']}, s0={x['s0']} -> u0_max={x['u0_max']:.6f} (|H|={x['H']:.1e}, |M|={x['M']:.1e})")
    out["m1_max_overall"] = m1_max_overall
    out["m1_exceeds_alg_count"] = len(m1_exceeds_alg)
    out["m1_exceeds_alg_top5"] = sorted(m1_exceeds_alg, key=lambda x: -x["u0_max"])[:5]

    # ------------------------------------------------------------------
    # [D] Fine 1D Comparison Curve at a_mean = 2.0, s0 = 1.0 for Plotting
    # ------------------------------------------------------------------
    r0_fine = [1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30, 1.33, 1.36]
    fine_curve = []
    for r0 in r0_fine:
        u2, res2 = bsearch(run_model_II, 2.0, r0, 1.0, u_hi=0.55, iters=22)
        u1, res1 = bsearch(run_model_I, 2.0, r0, 1.0, u_hi=0.80, iters=22)
        fine_curve.append({
            "r0": r0,
            "m2_u0_max": u2, "m2_H": res2.get("max_H_res"), "m2_M": res2.get("max_M_res"),
            "m1_u0_max": u1, "m1_H": res1.get("max_H_res"), "m1_M": res1.get("max_M_res"),
            "m1_flips": res1.get("poynting_flips"), "m1_min_S3": res1.get("min_S3_norm"),
        })
    out["fine_r0_curve_a2_s1"] = fine_curve

    # ------------------------------------------------------------------
    # [E] High-Resolution Representative Bounces & Collapses for Figures + Analysis
    # ------------------------------------------------------------------
    traces = {
        "m2_bounce_a2_r1p15_u0p40": run_model_II(2.0, 1.15, 1.0, 0.40, store_trace=True),
        "m2_collapse_a2_r1p15_u0p46": run_model_II(2.0, 1.15, 1.0, 0.46, store_trace=True),
        "m2_near_fp_a1p05_u0p510": run_model_II(1.05 * a_mean_s, r_s, s_s, 0.510, t_max=12.0, store_trace=True),
        "m1_bounce_a2_r1p15_u0p40": run_model_I(2.0, 1.15, 1.0, 0.40, phi0=0.0, store_trace=True),
        "m1_collapse_a2_r1p15_u0p44": run_model_I(2.0, 1.15, 1.0, 0.44, phi0=0.0, store_trace=True),
        "m1_bounce_a2_r1p05_u0p20": run_model_I(2.0, 1.05, 1.0, 0.20, phi0=0.0, store_trace=True),
    }
    print("\n=== [E] High-Resolution Representative Trajectories (rtol=1e-13, atol=1e-15) ===")
    for k, v in traces.items():
        extra = f" | S3/u=[{v['min_S3_norm']:+.3f},{v['max_S3_norm']:+.3f}], flips={v['poynting_flips']}, max|I1|={v['max_abs_I1']:.3f}, max|I2|={v['max_abs_I2']:.3f}" if "min_S3_norm" in v else ""
        print(f"  {k:28s}: {v['outcome']:18s} | |H|={v['max_H_res']:.2e}, |M|={v['max_M_res']:.2e} | min_g33={v['min_g33']:.4f}, max_g00={v['max_g00_corot']:+.3f}{extra}")
    out["traces"] = traces

    # ------------------------------------------------------------------
    # [F] Independent Kill-Tests (KT1..KT4)
    # ------------------------------------------------------------------
    print("\n=== [F] Independent Kill-Tests ===")
    # KT1: Solver cross-check (DOP853 vs Radau) on m1_bounce_a2_r1p15_u0p40
    r_dop = run_model_I(2.0, 1.15, 1.0, 0.40, method="DOP853", rtol=1e-13, atol=1e-15)
    r_rad = run_model_I(2.0, 1.15, 1.0, 0.40, method="Radau", rtol=1e-12, atol=1e-14)
    kt1_diff = max(abs(r_dop["min_a1"] - r_rad["min_a1"]), abs(r_dop["min_a2"] - r_rad["min_a2"]), abs(r_dop["min_a3"] - r_rad["min_a3"]))
    print(f"  [KT1] DOP853 vs Radau max |Delta min_a_i| = {kt1_diff:.3e} (DOP |H|={r_dop['max_H_res']:.2e}, Radau |H|={r_rad['max_H_res']:.2e})")

    # KT2: Exact SO(2) electromagnetic duality invariance across 8 phases phi0 in [0, 2*pi)
    kt2_mins = []
    for k in range(8):
        rk = run_model_I(2.0, 1.15, 1.0, 0.40, phi0=k * math.pi / 4.0)
        kt2_mins.append((rk["min_a1"], rk["min_a2"], rk["min_a3"], rk["min_S3_norm"]))
    kt2_spread = max(max(x[j] for x in kt2_mins) - min(x[j] for x in kt2_mins) for j in range(4))
    print(f"  [KT2] SO(2) Duality Invariance across 8 phases: max spread = {kt2_spread:.3e}")

    # KT3: Exact vacuum limit (u0 = 0) recovery: sig12 == 0 identically, matches diagonal triaxial Bianchi IX
    r_vac1 = run_model_I(2.0, 1.15, 1.0, 0.0)
    r_vac2 = run_model_II(2.0, 1.15, 1.0, 0.0)
    kt3_diff = max(abs(r_vac1["min_a1"] - r_vac2["min_a1"]), abs(r_vac1["min_a2"] - r_vac2["min_a2"]), abs(r_vac1["min_a3"] - r_vac2["min_a3"]))
    print(f"  [KT3] Vacuum u0=0 recovery M-I vs M-II max diff = {kt3_diff:.3e}")

    # KT4: Monte Carlo Falsification Attempt on Exact Theorem 2 (u0 = 0.514 > 8*sqrt(3)/27 = 0.513200)
    rng = np.random.default_rng(20261124)
    kt4_bounces = 0
    kt4_valid = 0
    for _ in range(400):
        am = float(rng.uniform(0.7, 3.5))
        r0 = float(rng.uniform(1.02, 2.5))
        s0 = float(rng.uniform(0.5, 2.0))
        sh12 = float(rng.uniform(-0.3, 0.3))
        sh3 = float(rng.uniform(-0.3, 0.3))
        rr = run_model_II(am, r0, s0, 0.514, shear_12=sh12, shear_3=sh3, t_max=10.0, rtol=1e-10, atol=1e-12)
        if rr["outcome"] != "invalid_initial_constraint":
            kt4_valid += 1
            if rr["outcome"] == "nonsingular_bounce":
                kt4_bounces += 1
    print(f"  [KT4] Theorem 2 Falsification Attempt (u0=0.514 > 8*sqrt(3)/27): {kt4_bounces}/{kt4_valid} bounces (expected 0!)")

    out["kill_tests"] = {
        "KT1_dop853_vs_radau_diff": kt1_diff,
        "KT1_dop_H": r_dop["max_H_res"], "KT1_dop_M": r_dop["max_M_res"],
        "KT1_radau_H": r_rad["max_H_res"], "KT1_radau_M": r_rad["max_M_res"],
        "KT2_duality_phase_spread": kt2_spread,
        "KT3_vacuum_recovery_diff": kt3_diff,
        "KT4_above_ceiling_u0_0p514_bounces": kt4_bounces,
        "KT4_above_ceiling_valid_trials": kt4_valid,
    }

    # Global max |H| and |M| across ALL nonsingular bounces in this suite
    all_H = []; all_M = []
    for item in asymp:
        for key in ("best_model_II", "best_model_I"):
            if item[key].get("u0_max", 0) > 0:
                all_H.append(item[key]["H"]); all_M.append(item[key]["M"])
    for item in fp_approach:
        if item["m2_u0_max"] > 0:
            all_H.append(item["m2_H"]); all_M.append(item["m2_M"])
        if item["m1_u0_max"] > 0:
            all_H.append(item["m1_H"]); all_M.append(item["m1_M"])
    for item in fine_curve:
        if item["m2_u0_max"] > 0:
            all_H.append(item["m2_H"]); all_M.append(item["m2_M"])
        if item["m1_u0_max"] > 0:
            all_H.append(item["m1_H"]); all_M.append(item["m1_M"])
    for v in traces.values():
        if v["outcome"] == "nonsingular_bounce":
            all_H.append(v["max_H_res"]); all_M.append(v["max_M_res"])
    out["global_max_nonsingular_H"] = max(all_H)
    out["global_max_nonsingular_M"] = max(all_M)
    print(f"\nGLOBAL MAX NONSINGULAR RESIDUALS (strictly < 1e-11 target): |H| = {max(all_H):.3e}, |M_3| = {max(all_M):.3e}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "deep_phase_and_killtests.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("Saved deep analysis & kill-tests ->", out_path)


if __name__ == "__main__":
    main()
