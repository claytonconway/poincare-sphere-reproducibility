"""R2-Probe #3: Warm Dynamical-Scalar Receding-Ceiling Bounce vs. a^-6 Kination.

Replaces rigid Lambda on closed FLRW S^3/I* with a canonical scalar field phi(t)
(V(phi) = V0 + 0.5 * m^2 * phi^2, obeying NEC, WEC, DEC everywhere) coupled to
isotropic radiation u_rad(t) via standard warm-cosmology friction Upsilon * (dphi/dt)^2 >= 0
(guaranteed 2nd law of thermodynamics dS_rad/dt >= 0!).

Tests two distinct mechanisms:
1. Conservative dynamical scalar (Upsilon = 0) rolling UP V(phi) during contraction
   (Hubble anti-friction -3 H dphi/dt > 0 when H < 0 pumps kinetic energy into climbing V(phi)!):
   Does this lift V(phi) ahead of incoming radiation and rescue gamma_0 > 1/4, OR does
   scalar kinetic energy (0.5 dphi/dt^2 ~ a^-6) trigger kination collapse?
2. Warm dissipative coupling (Upsilon > 0, draining scalar kinetic energy into radiation
   while obeying 2nd law T dS_rad/dt = Upsilon dphi/dt^2 >= 0): Does warm dissipation
   suppress a^-6 kination and widen the bounce basin beyond gamma_c = 1/4, OR does the
   extra dumped radiation push gamma_eff even further above 1/4?
"""

import json
import math
import os

import numpy as np
from scipy.integrate import solve_ivp

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def save_r2(name: str, payload: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def integrate_warm_scalar_bounce(
    gamma0: float,
    m_phi: float,
    phi0: float,
    dphi0: float,
    upsilon: float,
    a0: float = 2.0,
    V0: float = 3.0,
    t_max: float = 10.0,
):
    """Integrate closed FLRW scalar + radiation on S^3/I* (8 pi G = c = 1; V0 = 3 => a_dS = 1).

    Equations:
      H^2 = (u_phi + u_rad)/3 - 1/a^2,  u_phi = 0.5*dphi^2 + V(phi),  V(phi) = V0 + 0.5*m^2*phi^2
      dH/dt = -0.5*(dphi^2 + (4/3)*u_rad) + 1/a^2
      ddphi + (3 H + upsilon) dphi + V'(phi) = 0
      du_rad/dt + 4 H u_rad = +upsilon * dphi^2   (guaranteed >= 0: 2nd Law obeyed!)
    Note: at initial a0, u_rad0 = 3 * gamma0 / a0^4 (matches static Lambda=3 convention where gamma = B/1).
    """
    u_rad0 = 3.0 * gamma0 / (a0 ** 4)
    V_init = V0 + 0.5 * (m_phi ** 2) * (phi0 ** 2)
    u_phi0 = 0.5 * (dphi0 ** 2) + V_init
    rhs = (u_phi0 + u_rad0) / 3.0 - 1.0 / (a0 ** 2)
    if rhs <= 0.0:
        return {"outcome": "invalid_initial", "max_H_res": None}
    H0 = -math.sqrt(rhs)
    y0 = np.array([a0, H0, phi0, dphi0, u_rad0], dtype=np.float64)

    def ode(_t, y):
        a, H, phi, dphi, u_rad = y
        a_s = max(a, 1e-8)
        dH = -0.5 * (dphi ** 2 + (4.0 / 3.0) * u_rad) + 1.0 / (a_s ** 2)
        ddphi = -(3.0 * H + upsilon) * dphi - (m_phi ** 2) * phi
        du_rad = -4.0 * H * u_rad + upsilon * (dphi ** 2)
        return np.array([H * a, dH, dphi, ddphi, du_rad], dtype=np.float64)

    def ev_sing(_t, y):
        return y[0] - 0.02

    ev_sing.terminal = True
    ev_sing.direction = -1

    def ev_esc(_t, y):
        return y[0] - 8.0

    ev_esc.terminal = True
    ev_esc.direction = 1

    sol = solve_ivp(ode, (0.0, t_max), y0, method="DOP853", rtol=1e-11, atol=1e-13, events=[ev_sing, ev_esc], max_step=0.04)

    # Check Hamiltonian constraint & NEC/WEC/DEC at every step
    a_arr, H_arr, phi_arr, dphi_arr, ur_arr = sol.y
    V_arr = V0 + 0.5 * (m_phi ** 2) * (phi_arr ** 2)
    up_arr = 0.5 * (dphi_arr ** 2) + V_arr
    pp_arr = 0.5 * (dphi_arr ** 2) - V_arr
    u_tot = up_arr + ur_arr
    p_tot = pp_arr + ur_arr / 3.0

    H_res = np.abs(H_arr ** 2 - ((u_tot / 3.0) - 1.0 / (a_arr ** 2)))
    nec_min = float(np.min(u_tot + p_tot))
    wec_min = float(np.min(u_tot))
    dec_min = float(np.min(u_tot - np.abs(p_tot)))
    max_V_ratio = float(np.max(V_arr) / V0)
    max_kin_ratio = float(np.max(0.5 * dphi_arr ** 2 / np.maximum(V_arr, 1e-12)))

    if len(sol.t_events[0]) > 0:
        outcome = "singular_collapse"
    elif H_arr[-1] > 0.0 and np.min(a_arr) > 0.05:
        outcome = "nonsingular_bounce"
    else:
        outcome = "unresolved"

    # Compute effective gamma at bounce/minimum: gamma_eff = (u_rad * a^4) / 3 vs effective V/V0
    idx_min = int(np.argmin(a_arr))
    return {
        "outcome": outcome,
        "gamma0": gamma0,
        "m_phi": m_phi,
        "phi0": phi0,
        "dphi0": dphi0,
        "upsilon": upsilon,
        "max_H_res": float(np.max(H_res)),
        "min_a": float(a_arr[idx_min]),
        "V_at_min_over_V0": float(V_arr[idx_min] / V0),
        "max_V_over_V0": max_V_ratio,
        "max_kin_over_V": max_kin_ratio,
        "nec_min": nec_min,
        "wec_min": wec_min,
        "dec_min": dec_min,
        "all_ECs_satisfied": bool(nec_min >= -1e-10 and wec_min >= -1e-10 and dec_min >= -1e-10),
        "t": sol.t[:: max(1, len(sol.t) // 100)].tolist(),
        "a": a_arr[:: max(1, len(a_arr) // 100)].tolist(),
        "V_over_V0": (V_arr / V0)[:: max(1, len(V_arr) // 100)].tolist(),
    }


def find_max_bouncing_gamma0(m_phi: float, phi0: float, dphi0: float, upsilon: float):
    """Binary search maximum initial gamma0 that still executes a nonsingular bounce."""
    lo, hi = 0.0, 2.5
    if integrate_warm_scalar_bounce(lo, m_phi, phi0, dphi0, upsilon)["outcome"] != "nonsingular_bounce":
        return 0.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if integrate_warm_scalar_bounce(mid, m_phi, phi0, dphi0, upsilon)["outcome"] == "nonsingular_bounce":
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    # Baseline static Lambda (m_phi = 0, phi0 = 0, dphi0 = 0, upsilon = 0) -> MUST recover gamma_c = 0.2500!
    g_static = find_max_bouncing_gamma0(0.0, 0.0, 0.0, 0.0)

    # Sweep 1: Conservative climbing scalar (upsilon = 0, phi0 = 0.5, dphi0 = +0.1..+1.0 climbing UP V(phi)!)
    # Check whether Hubble anti-friction during contraction lifts V(phi) enough to rescue gamma0 > 0.25!
    climb_sweep = []
    max_H_all = 0.0
    ec_all_ok = True
    for m_phi in [0.5, 1.0, 2.0, 4.0]:
        for dphi0 in [0.0, 0.2, 0.5, 1.0]:
            gc = find_max_bouncing_gamma0(m_phi, phi0=0.5, dphi0=dphi0, upsilon=0.0)
            sample = integrate_warm_scalar_bounce(max(0.0, gc - 0.01), m_phi, 0.5, dphi0, 0.0)
            if sample["max_H_res"] is not None:
                max_H_all = max(max_H_all, sample["max_H_res"])
                ec_all_ok = ec_all_ok and sample["all_ECs_satisfied"]
            # Compare against effective static ceiling using INITIAL V_init = 3 + 0.5*m^2*phi0^2!
            # Recall: static ceiling scales linearly with V_init / V0!
            # Wait: check exact scaling! H^2 = (V_init/3) - 1/a^2 + gamma0/a^4
            # Setting y = a * sqrt(V_init/3): y'^2 = y^2 - 1 + gamma0*(V_init/3) / y^2
            # So static ceiling for V_init is gamma_c_static(V_init) = 0.25 * (3 / V_init)!!
            # WAIT A SECOND! CHECK THAT SCALING!
            # Higher Lambda REDUCES gamma_c = 0.25 * (V0 / V_init) at fixed B,
            # OR in physical units B_max = 3 c^2 / (4 Lambda) -> HIGHER Lambda means SMALLER a_dS and SMALLER max B!
            # Wait: check that!! At bounce, u_rad / u_Lambda <= 1 AND a_min = a_ES = sqrt(3/(2 Lambda)).
            # So u_rad_max = u_Lambda => B_max = u_Lambda * a_ES^4 / 3 = (Lambda/3) * (9 / (4 Lambda^2)) = 3 / (4 Lambda)!!
            # WHOA!! HIGHER V(phi) LOWERS THE MAXIMUM ALLOWED COMOVING RADIATION B_max ~ 1/V(phi)!!
            V_init = 3.0 + 0.5 * (m_phi ** 2) * (0.5 ** 2)
            static_Vinit_ceiling = 0.25 * (3.0 / V_init)
            climb_sweep.append(
                {
                    "m_phi": m_phi,
                    "dphi0": dphi0,
                    "V_init_over_V0": V_init / 3.0,
                    "static_Vinit_ceiling_gamma0": static_Vinit_ceiling,
                    "observed_gamma0_max": gc,
                    "ratio_to_static_V0": gc / 0.25,
                    "ratio_to_static_Vinit": gc / static_Vinit_ceiling,
                    "V_at_bounce_over_V0": sample.get("V_at_min_over_V0"),
                    "max_kin_over_V": sample.get("max_kin_over_V"),
                }
            )

    # Sweep 2: Rolling DOWN V(phi) (decreasing V(phi) -> LOWER Lambda_eff -> HIGHER B_max = 3/(4 Lambda_eff)!)
    # WAIT A SECOND! What if phi rolls DOWN a gentle hill during contraction (or warm friction damps dphi so kination stays small)?
    downhill_sweep = []
    for upsilon in [0.0, 0.5, 2.0, 5.0]:
        for phi0 in [1.0, 2.0]:
            gc = find_max_bouncing_gamma0(m_phi=1.0, phi0=phi0, dphi0=-0.2, upsilon=upsilon)
            V_init = 3.0 + 0.5 * (1.0 ** 2) * (phi0 ** 2)
            static_Vinit = 0.25 * (3.0 / V_init)
            sample = integrate_warm_scalar_bounce(max(0.0, gc - 0.01), 1.0, phi0, -0.2, upsilon)
            if sample["max_H_res"] is not None:
                max_H_all = max(max_H_all, sample["max_H_res"])
                ec_all_ok = ec_all_ok and sample["all_ECs_satisfied"]
            downhill_sweep.append(
                {
                    "upsilon": upsilon,
                    "phi0": phi0,
                    "V_init_over_V0": V_init / 3.0,
                    "static_Vinit_ceiling_gamma0": static_Vinit,
                    "observed_gamma0_max": gc,
                    "ratio_to_static_V0": gc / 0.25,
                    "ratio_to_static_Vinit": gc / static_Vinit,
                    "V_at_bounce_over_V0": sample.get("V_at_min_over_V0"),
                }
            )

    best_climb = max(r["observed_gamma0_max"] for r in climb_sweep)
    best_down = max(r["observed_gamma0_max"] for r in downhill_sweep)

    # Check if ANY configuration beats static V0 ceiling 0.2500!
    if best_climb > 0.26:
        verdict = "Confirmed (Uphill Lift Rescues gamma0 > 1/4)"
    elif best_down > 0.26:
        verdict = "Inverted Discovery (Downhill V Reduction Rescues gamma0 > 1/4)"
    else:
        verdict = "Refuted (Sign Inversion: Raising V(phi) Lowers B_max ~ 1/V, and Kination/Warm Dumping Shrink Ceiling)"

    payload = {
        "probe_id": "r2_03",
        "title": "Warm Dynamical-Scalar Receding-Ceiling Bounce vs. a^-6 Kination",
        "verdict": verdict,
        "max_hamiltonian_constraint_residual": max_H_all,
        "all_energy_conditions_satisfied": ec_all_ok,
        "summary": {
            "baseline_static_gamma_c": g_static,
            "best_uphill_gamma0_max": best_climb,
            "best_downhill_warm_gamma0_max": best_down,
            "exact_analytic_sign_inversion": "B_max = 3 c^2 / (4 Lambda_eff) ~ 1 / V(phi): lifting V(phi) SHRINKS the maximum comoving radiation B_max rather than raising it!",
        },
        "climb_sweep": climb_sweep,
        "downhill_warm_sweep": downhill_sweep,
        "sample_traces": {
            "static_bounce_g0p24": integrate_warm_scalar_bounce(0.24, 0.0, 0.0, 0.0, 0.0),
            "uphill_collapse_g0p24": integrate_warm_scalar_bounce(0.24, 2.0, 0.5, 0.5, 0.0),
            "downhill_warm_g0p20": integrate_warm_scalar_bounce(0.20, 1.0, 1.0, -0.2, 2.0),
        },
    }
    save_r2("r2_03", payload)
    print(f"[R2-03] Verdict={verdict} | max|H|={max_H_all:.2e} | ECs={ec_all_ok} | "
          f"static={g_static:.5f}, best_climb={best_climb:.5f}, best_down={best_down:.5f}")


if __name__ == "__main__":
    main()
