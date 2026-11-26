#!/usr/bin/env python3
"""
p05_01_derivation_and_identities.py
===================================
Goal 1 verification for stream p05_starobinsky_casimir:
- Exact 4D FLRW curvature & anomaly tensor identities on closed S^3/Gamma (C_munurhosigma == 0)
- Exact Bianchi / covariant conservation identity d/dt(a^4 E_00) + a^4 H E_tr == 0 (50-digit mpmath)
- Exact S^3 Casimir recovery from (a_anom, b_anom) across spin 0, 1/2, 1
- Exact scheme-independent topological Casimir shifts Delta C_top on S^3/I* vs S^3
- Exact dimensionless hierarchy ratios H_S^2 |K_top|, sigma_0 H_S^2, and Einstein-static criteria
"""

import json
import os
from fractions import Fraction
import mpmath as mp

mp.mp.dps = 50

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)


def verify_flrw_identities():
    """
    Verify on an arbitrary non-trivial smooth test scale factor a(t) = 1.3 + 0.4*sin(0.7*t) + 0.15*exp(-0.3*t^2):
    1. R = 6*(Hdot + 2*H^2 + 1/a^2)
    2. G_euler = 24*(Hdot + H^2)*(H^2 + 1/a^2) == (2/3)*R^2 - 2*R_munu^2 (since Weyl C == 0)
    3. (3)H_00 == -3*(H^2 + 1/a^2)^2 and trace (3)H == (1/2)*G_euler
    4. (1)H_00 == 18*(2*H*Hddot - Hdot^2 + 6*H^2*Hdot - 2*H^2/a^2 + 1/a^4) and trace (1)H == 6*Box(R)
    5. Exact conservation identity: d/dt(a^4 * E_00) + a^4 * H * E_tr == 0 for arbitrary (H_S, sigma, Lambda, K_top)
    """
    def a_fn(t):
        return mp.mpf("1.3") + mp.mpf("0.4") * mp.sin(mp.mpf("0.7") * t) + mp.mpf("0.15") * mp.exp(mp.mpf("-0.3") * t * t)

    def H_fn(t):
        return mp.diff(a_fn, t) / a_fn(t)

    def R_fn(t):
        a = a_fn(t)
        H = H_fn(t)
        Hdot = mp.diff(H_fn, t)
        return 6 * (Hdot + 2 * H**2 + 1 / a**2)

    def BoxR_fn(t):
        H = H_fn(t)
        Rdot = mp.diff(R_fn, t)
        Rddot = mp.diff(R_fn, t, 2)
        return -Rddot - 3 * H * Rdot

    # Arbitrary non-trivial parameters
    H_S_sq = mp.mpf("2.718281828459")
    sigma = mp.mpf("0.314159265359")
    Lam = mp.mpf("0.85")
    K_top = mp.mpf("-1.23456789")

    def E00_fn(t):
        a = a_fn(t)
        H = H_fn(t)
        Hdot = mp.diff(H_fn, t)
        Hddot = mp.diff(H_fn, t, 2)
        Y = H**2 + 1 / a**2
        H1_00_over_18 = 2 * H * Hddot - Hdot**2 + 6 * H**2 * Hdot - 2 * H**2 / a**2 + 1 / a**4
        # E_00 = 3*(H^2 + 1/a^2) - Lam - 3*[ Y^2/H_S^2 - sigma * H1_00_over_18 + K_top/a^4 ]
        return 3 * (Y - Lam / 3 - Y**2 / H_S_sq + sigma * H1_00_over_18 - K_top / a**4)

    def Etr_fn(t):
        a = a_fn(t)
        H = H_fn(t)
        Hdot = mp.diff(H_fn, t)
        R = R_fn(t)
        BoxR = BoxR_fn(t)
        G_euler = 24 * (Hdot + H**2) * (H**2 + 1 / a**2)
        # E_tr = -R + 4*Lam - (-G_euler/(2*H_S^2) - sigma*BoxR)
        # Check: a_anom/M_Pl^2 = 1/(2*H_S^2), 12*alpha_eff/M_Pl^2 = sigma
        return -R + 4 * Lam + G_euler / (2 * H_S_sq) + sigma * BoxR

    test_times = [mp.mpf("0.0"), mp.mpf("0.37"), mp.mpf("1.15"), mp.mpf("-0.83")]
    max_euler_err = mp.mpf("0")
    max_H3_err = mp.mpf("0")
    max_H1_err = mp.mpf("0")
    max_bianchi_err = mp.mpf("0")

    for t0 in test_times:
        a = a_fn(t0)
        H = H_fn(t0)
        Hdot = mp.diff(H_fn, t0)
        Hddot = mp.diff(H_fn, t0, 2)
        X = Hdot + H**2
        Y = H**2 + 1 / a**2

        # Direct tensor components in orthonormal frame (-,+,+,+)
        R00 = -3 * X
        R11 = X + 2 * Y
        R = -R00 + 3 * R11
        R_sq_tensor = R00**2 + 3 * R11**2

        # Euler density when C = 0
        G_from_ricci = mp.mpf(2) / 3 * R**2 - 2 * R_sq_tensor
        G_formula = 24 * X * Y
        max_euler_err = max(max_euler_err, abs(G_from_ricci - G_formula))

        # (3)H_00 = R_0^rho R_{rho 0} - (2/3) R R_00 - (1/2) R_munu^2 g_00 + (1/4) R^2 g_00
        H3_00_direct = -(R00**2) - (mp.mpf(2) / 3) * R * R00 + mp.mpf("0.5") * R_sq_tensor - mp.mpf("0.25") * R**2
        H3_00_formula = -3 * Y**2
        max_H3_err = max(max_H3_err, abs(H3_00_direct - H3_00_formula))

        # (1)H_00 = 2 R R_00 - (1/2) R^2 g_00 - 2 nabla_0 nabla_0 R + 2 g_00 Box R
        Rdot = mp.diff(R_fn, t0)
        Rddot = mp.diff(R_fn, t0, 2)
        BoxR = -Rddot - 3 * H * Rdot
        H1_00_direct = 2 * R * R00 + mp.mpf("0.5") * R**2 - 2 * Rddot - 2 * BoxR
        H1_00_formula = 18 * (2 * H * Hddot - Hdot**2 + 6 * H**2 * Hdot - 2 * H**2 / a**2 + 1 / a**4)
        max_H1_err = max(max_H1_err, abs(H1_00_direct - H1_00_formula))

        # Bianchi / covariant conservation identity: d/dt(a^4 E_00) + a^4 H E_tr == 0
        lhs = mp.diff(lambda s: a_fn(s)**4 * E00_fn(s), t0)
        rhs = -a**4 * H * Etr_fn(t0)
        max_bianchi_err = max(max_bianchi_err, abs(lhs - rhs))

    return {
        "max_euler_identity_err": float(max_euler_err),
        "max_H3_00_identity_err": float(max_H3_err),
        "max_H1_00_identity_err": float(max_H1_err),
        "max_bianchi_conservation_err": float(max_bianchi_err),
    }


def verify_exact_spin_coefficients():
    """
    Exact rational verification of:
    - Anomaly coefficients a_anom * 5760*pi^2 and b_anom * 2880*pi^2
    - S^3 Casimir energy recovery: E_Cas(S^3) = 12*pi^2*(a_anom + 0.5*b_anom)
    - S^3/I* Casimir energy, density, and scheme-independent topological shift Delta C_top * pi^2
    - Exact dimensionless ratios H_S^2 * |K_top| and sigma_0 * H_S^2
    - Exact physical scales in unreduced l_P = sqrt(G*hbar/c^3) and reduced l_Pl = sqrt(8*pi)*l_P
    """
    # Exact 40-digit-verified Casimir energies in units of hbar*c/a (from Monograph Section 5)
    E_S3 = {
        "scalar": Fraction(1, 240),
        "weyl": Fraction(17, 960),
        "vector": Fraction(11, 120),
    }
    E_Istar = {
        "scalar": Fraction(-43553, 43200),
        "weyl": Fraction(567443, 345600),
        "vector": Fraction(-349, 675),
    }
    # Universal Euler anomaly a_anom = a_num / (5760 * pi^2)
    a_num = {
        "scalar": Fraction(1, 1),
        "weyl": Fraction(11, 2),
        "vector": Fraction(62, 1),
    }
    # Zeta / point-splitting Box R anomaly b_anom = b_num / (2880 * pi^2)
    b_num = {
        "scalar": Fraction(1, 1),
        "weyl": Fraction(3, 1),
        "vector": Fraction(-18, 1),
    }
    # Propagating degrees of freedom per species
    ndof = {"scalar": 1, "weyl": 2, "vector": 2}

    results = {}
    pi_val = mp.pi
    for sp in ["scalar", "weyl", "vector"]:
        # Check S^3 Casimir recovery: 12*pi^2*(a_num/(5760*pi^2) + 0.5*b_num/(2880*pi^2)) = (a_num + b_num)/480
        E_S3_from_anom = (a_num[sp] + b_num[sp]) / 480
        assert E_S3_from_anom == E_S3[sp], f"Mismatch for {sp}: {E_S3_from_anom} vs {E_S3[sp]}"

        # Densities * pi^2 (in units hbar c / a^4):
        # Vol(S^3) = 2 pi^2 a^3 => u(S^3) * pi^2 = E_S3 / 2
        # Vol(S^3/I*) = pi^2 a^3 / 60 => u(S^3/I*) * pi^2 = 60 * E_Istar
        u_S3_pi2 = E_S3[sp] / 2
        u_Istar_pi2 = 60 * E_Istar[sp]
        delta_C_pi2 = u_Istar_pi2 - u_S3_pi2

        # In Friedmann H^2 + 1/a^2 = ... + K_top/a^4 + Y^2/H_S^2 - sigma*(...)
        # 1/H_S^2 = (a_num / (360*pi)) * l_P^2
        # sigma_0 = (-b_num / (360*pi)) * l_P^2
        # K_top   = (8*pi/3) * (delta_C_pi2 / pi^2) * l_P^2 = (8 * delta_C_pi2 / (3*pi)) * l_P^2
        # K_eff(sigma_0) = 1/H_S^2 - sigma_0 + K_top = (8 * u_Istar_pi2 / (3*pi)) * l_P^2
        inv_HS2_times_pi = a_num[sp] / 360
        sigma0_times_pi = -b_num[sp] / 360
        K_top_times_pi = Fraction(8, 3) * delta_C_pi2
        K_eff0_times_pi = inv_HS2_times_pi - sigma0_times_pi + K_top_times_pi
        K_eff0_direct_times_pi = Fraction(8, 3) * u_Istar_pi2
        assert K_eff0_times_pi == K_eff0_direct_times_pi, f"K_eff mismatch for {sp}"

        # Exact dimensionless ratios (pi and l_P cancel identically!)
        HS2_K_top = K_top_times_pi / inv_HS2_times_pi
        sigma0_HS2 = sigma0_times_pi / inv_HS2_times_pi
        HS2_K_eff0 = K_eff0_times_pi / inv_HS2_times_pi

        # Numerical scales in unreduced l_P and reduced l_Pl = sqrt(8*pi)*l_P
        a_S_lP = float(mp.sqrt(mp.mpf(inv_HS2_times_pi.numerator) / (inv_HS2_times_pi.denominator * pi_val)))
        a_Cas_top_lP = float(mp.sqrt(abs(mp.mpf(K_top_times_pi.numerator) / (K_top_times_pi.denominator * pi_val))))
        a_Cas_full_lP = float(mp.sqrt(abs(mp.mpf(K_eff0_times_pi.numerator) / (K_eff0_times_pi.denominator * pi_val))))
        sqrt8pi = float(mp.sqrt(8 * pi_val))
        Lmin_factor = float(pi_val / 5)

        results[sp] = {
            "ndof": ndof[sp],
            "a_num_over_5760pi2": str(a_num[sp]),
            "b_num_over_2880pi2": str(b_num[sp]),
            "E_S3_exact": str(E_S3[sp]),
            "E_S3_from_anomaly_match": True,
            "E_Istar_exact": str(E_Istar[sp]),
            "u_Istar_times_pi2": str(u_Istar_pi2),
            "delta_C_top_times_pi2": str(delta_C_pi2),
            "delta_C_top_float": float(mp.mpf(delta_C_pi2.numerator) / (delta_C_pi2.denominator * pi_val**2)),
            "inv_HS2_times_pi_over_lP2": str(inv_HS2_times_pi),
            "sigma0_times_pi_over_lP2": str(sigma0_times_pi),
            "K_top_times_pi_over_lP2": str(K_top_times_pi),
            "K_eff0_times_pi_over_lP2": str(K_eff0_times_pi),
            "HS2_times_K_top_exact": str(HS2_K_top),
            "HS2_times_K_top_float": float(HS2_K_top),
            "sigma0_times_HS2_exact": str(sigma0_HS2),
            "sigma0_times_HS2_float": float(sigma0_HS2),
            "HS2_times_K_eff0_exact": str(HS2_K_eff0),
            "a_S_over_lP": a_S_lP,
            "a_S_over_lPl": a_S_lP / sqrt8pi,
            "a_Cas_top_over_lP": a_Cas_top_lP,
            "a_Cas_full_over_lP": a_Cas_full_lP,
            "Lmin_at_a_S_over_lP": Lmin_factor * a_S_lP,
            "Lmin_at_a_Cas_top_over_lP": Lmin_factor * a_Cas_top_lP,
            "Lmin_at_a_Cas_top_over_l_sp_unred": (Lmin_factor * a_Cas_top_lP) / float(mp.sqrt(ndof[sp])),
            "Lmin_at_a_Cas_top_over_l_sp_red": (Lmin_factor * a_Cas_top_lP) / (sqrt8pi * float(mp.sqrt(ndof[sp]))),
            "Lmin_at_a_S_over_l_sp_unred": (Lmin_factor * a_S_lP) / float(mp.sqrt(ndof[sp])),
            "Lmin_at_a_S_over_l_sp_red": (Lmin_factor * a_S_lP) / (sqrt8pi * float(mp.sqrt(ndof[sp]))),
        }

    # Standard Model massless conformal check (N_S=4, N_W=45, N_V=12)
    NS, NW, NV = 4, 45, 12
    a_SM = NS * a_num["scalar"] + NW * a_num["weyl"] + NV * a_num["vector"]
    b_SM = NS * b_num["scalar"] + NW * b_num["weyl"] + NV * b_num["vector"]
    dC_SM_pi2 = (
        NS * (60 * E_Istar["scalar"] - E_S3["scalar"] / 2)
        + NW * (60 * E_Istar["weyl"] - E_S3["weyl"] / 2)
        + NV * (60 * E_Istar["vector"] - E_S3["vector"] / 2)
    )
    E_SM_Istar = NS * E_Istar["scalar"] + NW * E_Istar["weyl"] + NV * E_Istar["vector"]
    results["standard_model_conformal"] = {
        "N_S": NS, "N_W": NW, "N_V": NV, "N_dof": NS + 2 * NW + 2 * NV,
        "a_num_SM": str(a_SM),
        "b_num_SM": str(b_SM),
        "sigma0_times_HS2_SM": str(-b_SM / a_SM),
        "sigma0_times_HS2_SM_float": float(-b_SM / a_SM),
        "E_SM_Istar_exact": str(E_SM_Istar),
        "E_SM_Istar_float": float(E_SM_Istar),
        "delta_C_top_SM_times_pi2": str(dC_SM_pi2),
        "HS2_times_K_top_SM_exact": str((Fraction(8, 3) * dC_SM_pi2) / (a_SM / 360)),
        "HS2_times_K_top_SM_float": float((Fraction(8, 3) * dC_SM_pi2) / (a_SM / 360)),
    }
    return results


def main():
    identities = verify_flrw_identities()
    coeffs = verify_exact_spin_coefficients()
    out = {
        "goal": "Goal 1: Exact O(hbar) Semiclassical Friedmann + Trace Anomaly + Topological Casimir Identities on S^3/I*",
        "flrw_identities_50dps": identities,
        "exact_spin_coefficients": coeffs,
    }
    out_path = os.path.join(OUT_DIR, "p05_01_identities.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
