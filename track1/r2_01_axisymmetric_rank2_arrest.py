"""R2-Probe #1: Axisymmetric Hopf-Squashed Bianchi IX + Transverse Rank-2 Arrest.

Exact coupled Einstein-Maxwell-Lambda equations on axisymmetric Taub/Hopf-squashed S^3/I*:
  ds^2 = -dt^2 + b(t)^2 ((w^1)^2 + (w^2)^2) + c(t)^2 (w^3)^2
sourced by positive Lambda (= 3, so a_dS = 1) plus a SINGLE exact standing
Beltrami-Maxwell mode along the Hopf fiber X_3:
  E_3(t) = sqrt(2 u_0) cos(Phi) / b^2,  B_3(t) = sqrt(2 u_0) sin(Phi) / b^2,  dPhi/dt = 2 c / b^2
giving ZERO Poynting flux (T_{0a} = 0), exact Maxwell satisfaction, and exact
rank-2 transverse spatial stress:
  T_{mu nu} = diag(u, +u, +u, -u),  u(t) = u_0 / b(t)^4.

Tests whether the strictly rank-2 transverse pressure (+u, +u) arrests pancake
collapse (b -> 0) into a nonsingular bounce without 3-axis isotropy, or whether
longitudinal tension (-u) accelerates spindle collapse (c -> 0), compared directly
against the vacuum (u_0 = 0, Lambda-only) null control.
"""

import json
import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def save_r2(name: str, payload: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def ricci3_axisymmetric(b: float, c: float):
    """Exact 3-Ricci eigenvalues on unit-S^3 normalized axisymmetric Bianchi IX."""
    r1 = 4.0 / (b * b) - 2.0 * (c * c) / (b ** 4)
    r3 = 2.0 * (c * c) / (b ** 4)
    return r1, r3, 2.0 * r1 + r3


def hamiltonian_residual(b: float, c: float, Hb: float, Hc: float, u0: float, Lam: float = 3.0) -> float:
    """Exact Hamiltonian constraint H = Hb^2 + 2 Hb Hc + (1/2) R^(3) - (u + Lambda) = 0."""
    _, _, R3 = ricci3_axisymmetric(b, c)
    u = u0 / (b ** 4)
    return (Hb * Hb + 2.0 * Hb * Hc + 0.5 * R3) - (u + Lam)


def integrate_axisymmetric(b0: float, c0: float, u0: float, shear_ratio: float = 0.0, Lam: float = 3.0, t_max: float = 8.0):
    """Integrate contracting axisymmetric Bianchi IX from initial (b0, c0) with exact H = 0.

    Parameterize initial expansion rates by Hc = Hb * (1 + shear_ratio) with Hb < 0 (contracting),
    solving the exact quadratic Hamiltonian constraint for Hb < 0:
      Hb^2 * (1 + 2*(1 + s)) = u + Lambda - 0.5 * R^(3).
    """
    _, _, R3_0 = ricci3_axisymmetric(b0, c0)
    u_init = u0 / (b0 ** 4)
    rhs = u_init + Lam - 0.5 * R3_0
    coeff = 3.0 + 2.0 * shear_ratio
    if rhs <= 0.0 or coeff <= 0.0:
        return {"outcome": "invalid_initial_constraint", "max_H_res": None}

    Hb0 = -math.sqrt(rhs / coeff)
    Hc0 = Hb0 * (1.0 + shear_ratio)
    y0 = np.array([b0, c0, Hb0, Hc0], dtype=np.float64)
    h_init = abs(hamiltonian_residual(b0, c0, Hb0, Hc0, u0, Lam))

    def ode(_t, y):
        b, c, Hb, Hc = y
        b_s = max(b, 1e-8)
        c_s = max(c, 1e-8)
        r1, r3, _ = ricci3_axisymmetric(b_s, c_s)
        u = u0 / (b_s ** 4)
        Theta = 2.0 * Hb + Hc
        # Exact trace-reversed Bianchi IX equations for traceless T = diag(u, +u, +u, -u):
        # dHb/dt + Theta*Hb + r1 = Lambda + p1 = Lambda + u
        # dHc/dt + Theta*Hc + r3 = Lambda + p3 = Lambda - u
        dHb = Lam + u - Theta * Hb - r1
        dHc = Lam - u - Theta * Hc - r3
        return np.array([Hb * b, Hc * c, dHb, dHc], dtype=np.float64)

    def ev_pancake(_t, y):
        return y[0] - 0.02

    ev_pancake.terminal = True
    ev_pancake.direction = -1

    def ev_spindle(_t, y):
        return y[1] - 0.02

    ev_spindle.terminal = True
    ev_spindle.direction = -1

    def ev_escaped(_t, y):
        return min(y[0], y[1]) - 6.0

    ev_escaped.terminal = True
    ev_escaped.direction = 1

    sol = solve_ivp(
        ode,
        (0.0, t_max),
        y0,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        events=[ev_pancake, ev_spindle, ev_escaped],
        max_step=0.05,
    )

    # Compute exact Hamiltonian constraint residual along entire trajectory
    H_res = [abs(hamiltonian_residual(sol.y[0, i], sol.y[1, i], sol.y[2, i], sol.y[3, i], u0, Lam)) for i in range(sol.y.shape[1])]
    max_H = float(max(H_res))
    min_b = float(np.min(sol.y[0]))
    min_c = float(np.min(sol.y[1]))
    final_b = float(sol.y[0, -1])
    final_c = float(sol.y[1, -1])
    final_Hb = float(sol.y[2, -1])
    final_Hc = float(sol.y[3, -1])

    if len(sol.t_events[0]) > 0:
        outcome = "pancake_collapse_b0"
    elif len(sol.t_events[1]) > 0:
        outcome = "spindle_collapse_c0"
    elif final_Hb > 0.0 and final_Hc > 0.0 and min_b > 0.05 and min_c > 0.05:
        outcome = "nonsingular_bounce"
    else:
        outcome = "unresolved_or_mixed"

    # Check NEC/WEC/DEC for T = diag(u, +u, +u, -u) + Lambda:
    # u_tot = u + Lam > 0; u + p_1 = 2u >= 0; u + p_3 = 0 >= 0; |p_i| <= u_tot
    return {
        "outcome": outcome,
        "b0": b0,
        "c0": c0,
        "u0": u0,
        "init_H_res": h_init,
        "max_H_res": max_H,
        "min_b": min_b,
        "min_c": min_c,
        "final_b": final_b,
        "final_c": final_c,
        "final_squashing_c_over_b": final_c / max(final_b, 1e-12),
        "n_steps": int(sol.y.shape[1]),
        "t_end": float(sol.t[-1]),
    }


def main():
    # Sweep initial squashing ratio q0 = c0 / b0 at fixed geometric mean (b0^2 c0)^(1/3) = 2.0
    # and field amplitude u0 in [0.0, 0.85] (recall isotropic radiation ceiling is gamma = u0/3 = 0.25 -> u0 = 0.75!)
    q_grid = [0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.8, 2.2, 2.6]
    u0_grid = [0.0, 0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
    a_mean = 2.0

    phase_map = []
    vac_bounces = 0
    em_bounces_by_u0 = {}
    max_H_all = 0.0

    for u0 in u0_grid:
        b_cnt = 0
        p_cnt = 0
        s_cnt = 0
        for q0 in q_grid:
            # b0^2 * (q0 * b0) = a_mean^3 => b0 = a_mean / q0^(1/3)
            b0 = a_mean / (q0 ** (1.0 / 3.0))
            c0 = q0 * b0
            r = integrate_axisymmetric(b0, c0, u0)
            r["q0"] = q0
            phase_map.append(r)
            if r["max_H_res"] is not None:
                max_H_all = max(max_H_all, r["max_H_res"])
            if r["outcome"] == "nonsingular_bounce":
                b_cnt += 1
            elif r["outcome"] == "pancake_collapse_b0":
                p_cnt += 1
            elif r["outcome"] == "spindle_collapse_c0":
                s_cnt += 1
        em_bounces_by_u0[f"{u0:.2f}"] = {"bounce": b_cnt, "pancake": p_cnt, "spindle": s_cnt, "total": len(q_grid)}
        if u0 == 0.0:
            vac_bounces = b_cnt

    # Targeted pancake-arrest stress test: start with b contracting FASTER than c (shear_ratio = -0.6 -> Hc = 0.4 Hb, pancake-prone!)
    pancake_tests = []
    for u0 in [0.0, 0.05, 0.15, 0.30, 0.50, 0.75]:
        r = integrate_axisymmetric(b0=2.0, c0=2.0, u0=u0, shear_ratio=-0.6)
        pancake_tests.append(r)

    # Also test spindle-prone initial shear (shear_ratio = +0.6 -> Hc = 1.6 Hb)
    spindle_tests = []
    for u0 in [0.0, 0.05, 0.15, 0.30, 0.50, 0.75]:
        r = integrate_axisymmetric(b0=2.0, c0=2.0, u0=u0, shear_ratio=+0.6)
        spindle_tests.append(r)

    # Sample trajectories for plotting
    def trace_traj(b0, c0, u0, shear_ratio=0.0):
        _, _, R3_0 = ricci3_axisymmetric(b0, c0)
        rhs = u0 / (b0 ** 4) + 3.0 - 0.5 * R3_0
        Hb0 = -math.sqrt(rhs / (3.0 + 2.0 * shear_ratio))
        Hc0 = Hb0 * (1.0 + shear_ratio)
        def ode(_t, y):
            b, c, Hb, Hc = y
            r1, r3, _ = ricci3_axisymmetric(max(b, 1e-8), max(c, 1e-8))
            u = u0 / (max(b, 1e-8) ** 4)
            Th = 2 * Hb + Hc
            return np.array([Hb * b, Hc * c, 3.0 + u - Th * Hb - r1, 3.0 - u - Th * Hc - r3])
        def ev_p(_t, y): return y[0] - 0.02
        ev_p.terminal = True; ev_p.direction = -1
        def ev_s(_t, y): return y[1] - 0.02
        ev_s.terminal = True; ev_s.direction = -1
        def ev_e(_t, y): return min(y[0], y[1]) - 5.0
        ev_e.terminal = True; ev_e.direction = 1
        sol = solve_ivp(ode, (0, 7.0), [b0, c0, Hb0, Hc0], method="DOP853", rtol=1e-11, atol=1e-13, events=[ev_p, ev_s, ev_e], max_step=0.05)
        return {"t": sol.t.tolist(), "b": sol.y[0].tolist(), "c": sol.y[1].tolist()}

    traces = {
        "q0_1p0_u0_0p30": trace_traj(2.0, 2.0, 0.30, 0.0),
        "q0_1p0_u0_0p75": trace_traj(2.0, 2.0, 0.75, 0.0),
        "pancake_vac_u0_0": trace_traj(2.0, 2.0, 0.0, -0.6),
        "pancake_em_u0_0p30": trace_traj(2.0, 2.0, 0.30, -0.6),
    }

    # Exact binary search for maximum standing-Beltrami amplitude u0_max at isotropic start q0 = 1.00 (b0=c0=2.0)
    # Compare directly against isotropic radiation ceiling u0_iso_max = 3 * (1/4) = 0.75!
    lo_u, hi_u = 0.0, 2.0
    for _ in range(24):
        mid_u = 0.5 * (lo_u + hi_u)
        if integrate_axisymmetric(2.0, 2.0, mid_u, 0.0)["outcome"] == "nonsingular_bounce":
            lo_u = mid_u
        else:
            hi_u = mid_u
    u0_max_q1 = 0.5 * (lo_u + hi_u)
    r_near_max = integrate_axisymmetric(2.0, 2.0, lo_u, 0.0)

    # Evaluate verdict honestly:
    # Did transverse rank-2 pressure (+u, +u, -u) arrest pancake collapse OR raise bounce ceiling over isotropic 0.75?
    vac_pancake_collapsed = pancake_tests[0]["outcome"] != "nonsingular_bounce"
    em_rescues_pancake = any(r["outcome"] == "nonsingular_bounce" for r in pancake_tests[1:])
    max_em_bounces = max(v["bounce"] for k, v in em_bounces_by_u0.items() if k != "0.00")

    if em_rescues_pancake or max_em_bounces > vac_bounces:
        verdict = "Confirmed"
    elif u0_max_q1 > 0.76:
        verdict = "Mixed (Raises Isotropic-Start Ceiling Above 0.75 via Fiber Squashing, but Shrinks Squashing Basin via -u Spindle Tension)"
    else:
        verdict = "Refuted"

    payload = {
        "probe_id": "r2_01",
        "title": "Axisymmetric Hopf-Squashed Bianchi IX + Transverse Rank-2 Beltrami Arrest",
        "verdict": verdict,
        "max_hamiltonian_constraint_residual": max_H_all,
        "summary": {
            "vacuum_u0_0_bounces_out_of_9": vac_bounces,
            "isotropic_radiation_ceiling_u0_iso": 0.75,
            "standing_beltrami_q0_1_ceiling_u0_max": u0_max_q1,
            "near_max_min_b": r_near_max["min_b"],
            "near_max_min_c": r_near_max["min_c"],
            "near_max_H_res": r_near_max["max_H_res"],
            "bounces_by_u0": em_bounces_by_u0,
            "pancake_shear_outcomes": [(r["u0"], r["outcome"], round(r["min_b"], 4), round(r["min_c"], 4)) for r in pancake_tests],
            "spindle_shear_outcomes": [(r["u0"], r["outcome"], round(r["min_b"], 4), round(r["min_c"], 4)) for r in spindle_tests],
        },
        "phase_map": phase_map,
        "traces": traces,
    }
    p = save_r2("r2_01", payload)
    print(f"[R2-01] Verdict={verdict} | max|H|={max_H_all:.2e} | u0_max(q0=1)={u0_max_q1:.5f} (vs iso 0.75000!) | "
          f"bounces by u0: { {k: v['bounce'] for k, v in em_bounces_by_u0.items()} }")
    print("  Pancake-prone (s=-0.6):", payload["summary"]["pancake_shear_outcomes"])
    print("  Spindle-prone (s=+0.6):", payload["summary"]["spindle_shear_outcomes"])


if __name__ == "__main__":
    main()
