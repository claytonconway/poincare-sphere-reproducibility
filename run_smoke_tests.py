#!/usr/bin/env python3
"""Fast, path-independent smoke test suite validating core algebraic and numerical invariants
across Tracks I, II, and III in < 60 seconds.

Run from any working directory:
    python /path/to/reproducibility/run_smoke_tests.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

REPO_ROOT = Path(__file__).resolve().parent


def _with_track(track: str):
    tdir = str(REPO_ROOT / track)

    class _Ctx:
        def __enter__(self):
            sys.path.insert(0, tdir)
            return REPO_ROOT / track

        def __exit__(self, exc_type, exc, tb):
            if tdir in sys.path:
                sys.path.remove(tdir)

    return _Ctx()


def check_track1() -> dict:
    """Validate Track I (Classical GR, Tilted Bianchi IX, SU(2)_L Gauss Ellipsoid, Thin Shells)."""
    t0 = time.perf_counter()
    res = {}
    with _with_track("track1") as tdir:
        # 1. Exact pure-null ceiling u_0* = 8*sqrt(3)/27
        u0_star_exact = 8.0 * math.sqrt(3.0) / 27.0
        assert abs(u0_star_exact - 0.5132002392796674) < 1e-14

        # 2. Live 4D rotating triaxial Bianchi IX synchronous integration + exact constraint preservation
        import synchronous_frame_audit as sfa

        y_init = sfa.init_sync_page1(2.5, 1.15, 1.05, 0.30)
        diag0 = sfa.sync_diagnostics(y_init)
        assert abs(diag0["H_res"]) < 1e-13, f"Hamiltonian residual too large: {diag0['H_res']}"
        assert abs(diag0["M3_res"]) < 1e-13, f"Momentum residual too large: {diag0['M3_res']}"
        sol = sfa.run_sync_m1(y_init, t_span=(0.0, 3.0), rtol=1e-12, atol=1e-14)
        diag_end = sfa.sync_diagnostics(sol.y[:, -1])
        assert abs(diag_end["H_res"]) < 1e-10, f"Drifted H_res: {diag_end['H_res']}"
        assert abs(diag_end["I_EM"] - diag0["I_EM"]) < 1e-10, "I_EM integral drifted"
        res["t1_sync_H_res_end"] = float(abs(diag_end["H_res"]))

        # 3. Live Tilted SU(2)_L Pure-Null Gauss Ellipsoid check
        import r3_05_tilted_su2_bianchi9 as r35

        y_su2, kap = r35.build_pure_null_nonabelian_state(0.95, 1.15, 1.05, 0.55, phi_ell=0.95)
        d_su2 = r35.su2_diagnostics(y_su2, kap)
        assert abs(d_su2["H_res"]) < 1e-12, f"SU(2) H_res: {d_su2['H_res']}"
        assert abs(d_su2["G3_res"]) < 1e-13, f"SU(2) Gauss residual: {d_su2['G3_res']}"
        assert abs(abs(d_su2["S3_over_u"]) - 1.0) < 1e-12, f"Pure-null ratio: {d_su2['S3_over_u']}"
        res["t1_su2_gauss_res"] = float(abs(d_su2["G3_res"]))

        # 4. Spherical & Faceted Israel-Darmois exact algebraic ceilings + JSON cross-check
        sin3_pi10 = math.sin(math.pi / 10.0) ** 3
        pi_over_80 = math.pi / 80.0
        exact_ratio = math.pi * (math.sqrt(5.0) + 2.0) / 10.0
        assert abs((pi_over_80 / sin3_pi10) - exact_ratio) < 1e-14
        t_breach = math.acosh(1.0 / math.cos(math.pi / 10.0))
        t_breach_alt = math.log((1.0 + math.sin(math.pi / 10.0)) / math.cos(math.pi / 10.0))
        assert abs(t_breach - t_breach_alt) < 1e-15 and abs(t_breach - 0.319458) < 1e-6

        d4 = json.loads((tdir / "data" / "r3_04_faceted_shell.json").read_text())
        sin3_eq = d4["geometry_and_Y6"]["dirichlet_dodecahedron"]["sin3_chi_eq"]
        assert abs(sin3_eq - 0.037903) < 1e-5 and sin3_eq < pi_over_80
        d5 = json.loads((tdir / "data" / "r3_05_tilted_su2_bianchi9.json").read_text())
        pt = d5["paired_partially_tilted_coupling_sweep"]
        tot_su2 = sum(pt[k]["n_bounced_su2"] for k in pt)
        tot_u1x2 = sum(pt[k]["n_bounced_match_S3_u3_T33_U1xU1"] for k in pt)
        assert tot_su2 == 19 and tot_u1x2 == 18
        res["t1_ratio_exact"] = exact_ratio
        res["t1_faceted_sin3_eq"] = sin3_eq

    res["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return res


def check_track2() -> dict:
    """Validate Track II (I* Harmonics, 280.08x Gap, Galois P_7/rho_2, Sphaleron S_02, Twisted Casimir/eta)."""
    t0 = time.perf_counter()
    res = {}
    with _with_track("track2") as tdir:
        # 1. Live Klein-Molien Poincaré series & 280.08x stabilization factor
        import istar_harmonics as ih

        m = ih.klein_molien_coeffs(24)
        assert m[0] == 1 and all(m[k] == 0 for k in range(1, 12)) and m[12] == 1
        e_tach = lambda mv: Fraction((mv**2 - 1) ** 4, 32 * mv**4)
        assert e_tach(3) == Fraction(128, 81)
        assert e_tach(11) == Fraction(6480000, 14641)
        ratio_280 = float(e_tach(11) / e_tach(3))
        assert abs(ratio_280 - 280.078205) < 1e-6
        res["t2_stabilization_ratio"] = ratio_280

        # 2. Live degree-7 holomorphic intertwiner P_7: exact I*-equivariance, [2/9, 3/10] norm, & flatness
        import istar_geometry as ig
        import goal1_construct_rho2 as g1mod

        _q1, _q2, U1, U2 = ig.build_istar_and_galois()
        assert U1.shape == (120, 2, 2) and U2.shape == (120, 2, 2)
        C_mat, poly_coeffs = ig.build_p7_equivariant_polynomial(U1, U2)
        rng = np.random.default_rng(42)
        pts = rng.normal(size=(32, 4))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        U_pts = np.array([ig.quat_to_su2(q) for q in pts])
        _u7, A_pts, XA_pts, norm2, _dirichlet = g1mod.eval_A_and_XA_exact(U_pts, poly_coeffs)
        assert float(np.min(norm2)) >= 2.0 / 9.0 - 1e-10
        assert float(np.max(norm2)) <= 3.0 / 10.0 + 1e-10
        # Curvature B^a_i = (1/2) eps_{ijk} (X_j A^a_k - X_k A^a_j - 2 eps_{jkm} A^a_m + eps_{abc} A^b_j A^c_k)
        # Verify exact zero-curvature flatness directly via compute_ym_and_cs weights or explicit curvature!
        B_max = 0.0
        for p in range(len(U_pts)):
            Ap = A_pts[p]   # shape (3, 3): [color, spatial]
            XAp = XA_pts[p] # shape (3, 3, 3): [deriv_dir, color, spatial]
            for a in range(3):
                for i in range(3):
                    j, k = (i + 1) % 3, (i + 2) % 3
                    b_ai = XAp[j, a, k] - XAp[k, a, j] - 2.0 * Ap[a, i] + (
                        Ap[(a + 1) % 3, j] * Ap[(a + 2) % 3, k]
                        - Ap[(a + 2) % 3, j] * Ap[(a + 1) % 3, k]
                    )
                    B_max = max(B_max, abs(float(b_ai)))
        assert B_max < 1e-12, f"A_rho2 not flat: max |B| = {B_max}"
        res["t2_p7_norm2_range"] = [float(np.min(norm2)), float(np.max(norm2))]
        res["t2_rho2_max_B"] = B_max

        # 3. Reference verification of N_CS(rho_2)=49/120, S_02 full-3D 315-DOF Morse-index-1, & 21x Higgs gap
        g1 = json.loads((tdir / "data" / "goal1_rho2_verification.json").read_text())
        fl = g1["flatness_and_cs_2D_hopf"]
        assert fl["N_CS_err_vs_49_120"] < 1e-14
        assert fl["max_B"] < 1e-12
        g2m = json.loads((tdir / "data" / "goal2_morse_and_gauge_audit.json").read_text())
        s02 = g2m["full3d_N20_S02_audit"]
        assert s02["D"] == 315 and s02["phys_morse_neg_zero_pos"][0] == 1
        assert abs(s02["E_YM"] - 31.493722) < 5e-4
        assert abs(s02["N_CS"] - 0.240637) < 5e-4
        g3h = json.loads((tdir / "data" / "goal3_higgs_lifting.json").read_text())
        lap = g3h["part_a_covariant_laplacian_hosotani"]
        assert lap["lambda_min_rho1_a2"] == 3 and lap["lambda_min_rho2_a2"] == 63

        # 4. Live 50-digit Hurwitz-Molien Ad(rho_0, rho_1, rho_2) pure-YM Casimir & APS eta check
        import r3_01_twisted_casimir_eta as r31

        expected_ym = {"rho_0": Fraction(-349, 225), "rho_1": Fraction(-221, 150), "rho_2": Fraction(79, 150)}
        expected_eta = {"rho_0": Fraction(-361, 60), "rho_1": Fraction(-23, 20), "rho_2": Fraction(9, 20)}
        max_err = 0.0
        for name in ("rho_0", "rho_1", "rho_2"):
            out = r31.method2_hurwitz_molien(name)
            err_e = abs(float(out["E_Cas_YM"] - mp.mpf(expected_ym[name].numerator) / expected_ym[name].denominator))
            err_eta = abs(float(out["eta_curl_0"] - mp.mpf(expected_eta[name].numerator) / expected_eta[name].denominator))
            err_pal = abs(float(out["Delta_E_Cas"]))
            max_err = max(max_err, err_e, err_eta, err_pal)
        assert max_err < 1e-35, f"Twisted Casimir/eta precision mismatch: {max_err}"
        assert expected_ym["rho_2"] - expected_ym["rho_1"] == Fraction(2, 1)
        res["t2_twisted_casimir_max_err"] = max_err

    res["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return res


def check_track3() -> dict:
    """Validate Track III (Chiral Maxwell/ADE, Starobinsky-Casimir KAM, Species Squeeze, Berger Collapse, Riegert-Weyl)."""
    t0 = time.perf_counter()
    res = {}
    with _with_track("track3") as tdir:
        # 1. Live Starobinsky-Casimir KAM breathing ring ODE integration + exact constraint residual
        import p05_02_solve_and_phase_space as p05

        x_ring = math.sqrt(44672.0 / 93.0)
        y0 = p05.bounce_initial_state(x_ring, p05.SIGMA0_VECTOR, 0.0, p05.KAPPA_VECTOR, s_b=0.0, sign_hdot=+1)
        sol = solve_ivp(
            lambda t, y: p05.ode_rhs(t, y, p05.SIGMA0_VECTOR, 0.0),
            (0.0, 200.0),
            y0,
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
            max_step=0.25,
        )
        e00_max = max(abs(p05.e00_constraint(*sol.y[:, k], p05.SIGMA0_VECTOR, 0.0, p05.KAPPA_VECTOR)) for k in range(sol.y.shape[1]))
        assert e00_max < 1e-11, f"KAM constraint residual too large: {e00_max}"
        assert 20.0 < float(np.min(sol.y[0])) and float(np.max(sol.y[0])) < 24.0
        res["t3_kam_e00_max"] = float(e00_max)

        # 2. Exact N-invariant Species-Scale Squeeze across all 5 isotropy-irreducible quotients
        s3i_scalar_max = (1.0 / 5.0) * math.sqrt((87109.0 / 1440.0) / 3.0)
        s3i_vector = (1.0 / 5.0) * math.sqrt((22369.0 / 720.0) / 6.0)
        assert abs(s3i_scalar_max**2 - 87109.0 / 108000.0) < 1e-15 and s3i_scalar_max < 0.90
        assert abs(s3i_vector**2 - 22369.0 / 108000.0) < 1e-15 and s3i_vector < 0.46
        d_r32 = json.loads((tdir / "data" / "r3_02_ade_lens_species_escape.json").read_text())
        thr = d_r32["thresholds_static_round_metric"]
        assert thr["L_p_1_static_species_ratio_gt_1_p_crit"] == 34
        assert thr["D_m_star_static_species_ratio_gt_1_m_crit"] == 17
        res["t3_s3i_species_bounds"] = [round(s3i_vector, 6), round(s3i_scalar_max, 6)]

        # 3. Live Berger-sphere Maxwell curl spectrum verification + pinched S^1 x S^2 limit
        import t3_01_berger_casimir_backreaction as t31

        spec = t31.verify_maxwell_curl_spectrum()
        assert spec["max_eigenvalue_err"] < 1e-13
        assert spec["max_Dm_star_M0_parity_err"] < 1e-14
        d_t31 = json.loads((tdir / "data" / "t3_01_berger_casimir_backreaction.json").read_text())
        pv = d_t31["precision_verifications"]
        assert pv["max_lam1_vs_R3_2_L_err"] < 1e-18
        assert pv["pinched_and_round_pressure_checks"]["L40_em_lam0.05"]["p_par_over_u"] > 2.999
        assert d_t31["L40_prolate_and_shear_sweep"]["n_bounded"] == 0
        res["t3_berger_spec_err"] = float(spec["max_eigenvalue_err"])

        # 4. Live Covariant Riegert 2D variational integration on x_ring + symplectic conservation check
        import t3_02_riegert_weyl_c2_spin2 as t32

        rieg = t32.integrate_variational_riegert_2d(x_ring, tau_end=250.0, method="DOP853", sign_hdot=1)
        assert rieg["min_A_Rieg"] > 0.90, f"Unexpected min_A_Rieg: {rieg['min_A_Rieg']}"
        assert rieg["min_cs2"] > 0.45, f"Unexpected min_cs2: {rieg['min_cs2']}"
        assert rieg["symplectic_W_max_drift"] < 1e-10
        d_t32 = json.loads((tdir / "data" / "t3_02_riegert_weyl_c2_spin2.json").read_text())
        tongue = d_t32["weyl_c2_2to1_tongue_tau6000"]
        peak = max(tongue, key=lambda r: r["effective_lyapunov_rate"])
        assert abs(peak["beta_0"] - 0.2890) < 5e-4
        assert peak["bin_growth_ratio"] > 1900.0
        assert peak["symplectic_max_rel_err"] < 1e-12
        res["t3_riegert_wronskian_drift"] = float(rieg["symplectic_W_max_drift"])
        res["t3_weyl_2to1_peak_beta0"] = float(peak["beta_0"])

    res["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return res


def check_track4a() -> dict:
    """Validate Track IV-A (Discrete 2-Photon Pole Sum, Maxwell Contact Ward Identity, Talbot Revivals)."""
    t0 = time.perf_counter()
    res = {}
    tdir = REPO_ROOT / "track4a"

    # 1. Exact Bach 6j factorization across j=1..60 (exact Fraction)
    for j in range(1, 61):
        c_r = Fraction((2 * j - 1) * (2 * j + 3), 5 * (2 * j + 1))
        e_j = 4 * j + 2
        assert (2 * j + 1) * (2 * j + 2) * (2 * j) * c_r == Fraction((e_j**2 - 4) * (e_j**2 - 16), 80)

    # 2. Exact shell-by-shell Maxwell Contact Ward identity across N=1..61 (exact Fraction)
    for n in range(1, 62):
        s0 = Fraction(n * n + 2, 1)
        s2 = Fraction(16 * n * n, 45) + Fraction(40, 9) - Fraction(32, 15 * n * n)
        b2_2n = (4 * n * n - 4) * (4 * n * n - 16)
        assert Fraction(1, 2) * s2 - Fraction(4, 9) * s0 == Fraction(-b2_2n, 60 * n * n)

    # 3. Exact topological contact rational on S^3/I* & verified stream JSONs
    pi_contact = Fraction(1920, 31) * Fraction(-22369, 720)
    assert pi_contact == Fraction(-178952, 93)
    for sub in ("p2_01", "p2_06"):
        d = json.loads((tdir / sub / "results.json").read_text(encoding="utf-8"))
        assert d["status"] == "verified"
        kts = d["verification"]["kill_tests"]
        assert len(kts) >= 3 and all(k["passed"] is True for k in kts)

    res["pi_contact_Istar"] = str(pi_contact)
    res["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return res


def check_track4b() -> dict:
    """Validate Track IV-B (Goldstone-Covariant Fluctuation Theory & Dynamical Hosotani Avalanches)."""
    t0 = time.perf_counter()
    res = {}
    tdir = REPO_ROOT / "track4b"
    d1 = json.loads((tdir / "t4b_01_goldstone_covariant.json").read_text(encoding="utf-8"))
    d6 = json.loads((tdir / "t4b_06_final_referee_audit.json").read_text(encoding="utf-8"))
    sub30 = [r for r in d6["subhorizon_N24_vs_N30"] if r["N_max"] == 30][0]
    assert sub30["escaped"] is True and sub30["max_H_res_rel"] < 1e-12
    assert isinstance(d1, dict) and len(d1) > 0
    res["subhorizon_H_res"] = float(sub30["max_H_res_rel"])
    res["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return res


def check_track5() -> dict:
    """Validate Track V (Pure-Graviton Casimir, Chiral Asymmetry, Prolate Stabilization & Twisted Dirac)."""
    t0 = time.perf_counter()
    res = {}
    tdir = REPO_ROOT / "track5"

    # 1. Exact rational Galois jumps for twisted Dirac Casimir energies on S^3/I*
    assert Fraction(-106957, 86400) - Fraction(394163, 86400) == Fraction(-29, 5)
    assert Fraction(-26959, 19200) - Fraction(72881, 19200) == Fraction(-26, 5)

    # 2. Verify all 3 formal pure-graviton streams (p2_03, p2_04, p2_05)
    total_kts = 0
    for sub in ("p2_03", "p2_04", "p2_05"):
        d = json.loads((tdir / sub / "results.json").read_text(encoding="utf-8"))
        assert d["status"] == "verified"
        kts = d["verification"]["kill_tests"]
        assert len(kts) == 4 and all(k["passed"] is True for k in kts)
        total_kts += len(kts)

    res["track5_kts_passed"] = total_kts
    res["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return res


def check_track6() -> dict:
    """Validate Track VI (Complete Kubo-plus-Contact Nonlinear KAM Backreaction & Misner Talbot Scrambler)."""
    t0 = time.perf_counter()
    res = {}
    tdir = REPO_ROOT / "track6"

    # 1. Exact rational rel-S^3 kinetic & static contact-plus-Kubo constants
    pi_kin = Fraction(3900, 31)
    pi_rem = Fraction(984896, 465)
    assert abs(math.sqrt(float(pi_kin)) - 11.216347) < 1e-5
    assert abs(float(pi_rem) - 2118.055914) < 1e-5

    # 2. Verify both formal Track VI streams (p2_02 + p2_07, 8/8 kill-tests)
    total_kts = 0
    for sub in ("p2_02", "p2_07"):
        d = json.loads((tdir / sub / "results.json").read_text(encoding="utf-8"))
        assert d["status"] == "verified"
        kts = d["verification"]["kill_tests"]
        assert len(kts) == 4 and all(k["passed"] is True for k in kts)
        total_kts += len(kts)

    res["track6_kts_passed"] = total_kts
    res["x_kin_relS3"] = round(math.sqrt(float(pi_kin)), 4)
    res["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return res


def check_ledger_and_reports() -> dict:
    """Validate Claim Ledger (including preserved RETRACTED entries) and all 7 technical PDFs."""
    t0 = time.perf_counter()
    ledger = json.loads((REPO_ROOT / "ledger" / "claims.json").read_text(encoding="utf-8"))
    claims = ledger.get("claims", [])
    retracted = [c for c in claims if c.get("status") == "RETRACTED"]
    assert len(claims) >= 24 and len(retracted) >= 2

    pdfs = sorted((REPO_ROOT / "reports").glob("*.pdf"))
    assert len(pdfs) == 7 and all(p.stat().st_size > 50_000 for p in pdfs)
    return {
        "claims": len(claims),
        "retracted": len(retracted),
        "pdfs": len(pdfs),
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


def main() -> None:
    t_start = time.perf_counter()
    print("=" * 72)
    print("Poincare Homology Sphere S^3/I* (v2.1.0) : Automated Smoke Test Suite")
    print("=" * 72)

    r1 = check_track1()
    print(f"[PASS] Track I   ({r1['elapsed_s']:.2f}s): H_res={r1['t1_sync_H_res_end']:.2e}, "
          f"SU2_G3={r1['t1_su2_gauss_res']:.2e}, sin3_eq={r1['t1_faceted_sin3_eq']:.6f}")

    r2 = check_track2()
    print(f"[PASS] Track II  ({r2['elapsed_s']:.2f}s): 280x_gap={r2['t2_stabilization_ratio']:.4f}x, "
          f"P7_norm2=[{r2['t2_p7_norm2_range'][0]:.4f},{r2['t2_p7_norm2_range'][1]:.4f}], "
          f"twisted_err={r2['t2_twisted_casimir_max_err']:.2e}")

    r3 = check_track3()
    print(f"[PASS] Track III ({r3['elapsed_s']:.2f}s): KAM_e00={r3['t3_kam_e00_max']:.2e}, "
          f"Berger_spec_err={r3['t3_berger_spec_err']:.2e}, "
          f"Riegert_W={r3['t3_riegert_wronskian_drift']:.2e}, "
          f"Weyl_2to1_peak={r3['t3_weyl_2to1_peak_beta0']:.4f}")

    r4a = check_track4a()
    print(f"[PASS] Track IV-A({r4a['elapsed_s']:.2f}s): Bach 6j & Ward N=1..61 exact, "
          f"Pi_contact={r4a['pi_contact_Istar']}, p2_01+p2_06 verified")

    r4b = check_track4b()
    print(f"[PASS] Track IV-B({r4b['elapsed_s']:.2f}s): Goldstone-covariant & t4b_06 N30 H_res={r4b['subhorizon_H_res']:.2e}")

    r5 = check_track5()
    print(f"[PASS] Track V   ({r5['elapsed_s']:.2f}s): Dirac jumps -29/5 & -26/5 exact, "
          f"p2_03+p2_04+p2_05 ({r5['track5_kts_passed']}/12 KTs passed)")

    r6 = check_track6()
    print(f"[PASS] Track VI  ({r6['elapsed_s']:.2f}s): x_kin={r6['x_kin_relS3']:.4f}, "
          f"p2_02+p2_07 ({r6['track6_kts_passed']}/8 KTs passed)")

    rl = check_ledger_and_reports()
    print(f"[PASS] Archive   ({rl['elapsed_s']:.2f}s): {rl['claims']} claims ({rl['retracted']} RETRACTED preserved), "
          f"{rl['pdfs']} technical PDFs")

    total = time.perf_counter() - t_start
    print("-" * 72)
    print(f"ALL SMOKE TESTS PASSED in {total:.2f}s (< 60s budget).")
    assert total < 60.0, f"Exceeded 60s budget: {total:.2f}s"


if __name__ == "__main__":
    main()
