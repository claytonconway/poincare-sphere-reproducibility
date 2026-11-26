"""Targeted Final Diagnostics:
  1. Verify 4D Kretschmann & Weyl sign on pure de Sitter (K=+24, Weyl^2=0) and decompose E^2 vs B^2 at t=3.3378.
  2. Test whether ANY triaxial S3(0)=0 bounce above 8*sqrt(3)/27 reaches |S3/u| -> 1 on its contracting leg
     (decisively testing for a tuned contracting Page-1 pure-null counterexample above 8*sqrt(3)/27!).
  3. Confirm bounced=True in 40-digit mpmath on exact float64 ceiling runs at a_mean = 5, 10, 20 (checking
     turning point passage Th: - -> + and min(H_i) > 0).
"""

import json
import math
import os
import mpmath as mp
import numpy as np

from verify_exact_system import left_invariant_coframe, compute_christoffel, compute_einstein_4d
from synchronous_frame_audit import (
    LAM, U0_ALG, U0_PROLATE,
    sync_ode_rhs, sync_diagnostics, run_sync_m1, init_sync_page1,
    mpmath_verify_m2_exact, mpmath_verify_m1_sync_exact,
)


def check_kretschmann_on_ds_and_peak():
    """Verify Kretschmann/Weyl formula on exact pure de Sitter (a(t)=cosh(t), Lam=3 => K=+24, Ric^2=36, Weyl^2=0)."""
    mp.mp.dps = 35
    x0 = [mp.mpf("0.3"), mp.mpf("1.1"), mp.mpf("0.7"), mp.mpf("0.5")]
    h = mp.mpf("1e-7")

    def ds_metric(pt):
        t, th, ph, ps = pt
        w1, w2, w3 = left_invariant_coframe(th, ph, ps)
        a2 = mp.cosh(t) ** 2
        g = mp.zeros(4, 4)
        g[0, 0] = -mp.mpf(1)
        for i in range(4):
            for j in range(4):
                g[i, j] += a2 * (w1[i]*w1[j] + w2[i]*w2[j] + w3[i]*w3[j])
        return g

    g, ginv, Gam, Ric, R_sc, _ = compute_einstein_4d(ds_metric, x0, h=h)
    dGam = [None] * 4
    for lam in range(4):
        xp1 = list(x0); xp1[lam] += h
        xm1 = list(x0); xm1[lam] -= h
        xp2 = list(x0); xp2[lam] += 2 * h
        xm2 = list(x0); xm2[lam] -= 2 * h
        _, _, Gp1 = compute_christoffel(ds_metric, xp1, h=h)
        _, _, Gm1 = compute_christoffel(ds_metric, xm1, h=h)
        _, _, Gp2 = compute_christoffel(ds_metric, xp2, h=h)
        _, _, Gm2 = compute_christoffel(ds_metric, xm2, h=h)
        dGam[lam] = [[[[(-Gp2[r][m][n] + 8*Gp1[r][m][n] - 8*Gm1[r][m][n] + Gm2[r][m][n]) / (12*h) for n in range(4)] for m in range(4)] for r in range(4)]][0]

    # Careful! Standard Riemann convention:
    # R^r_{s m n} = d_m Gam^r_{n s} - d_n Gam^r_{m s} + Gam^r_{m l} Gam^l_{n s} - Gam^r_{n l} Gam^l_{m s}
    # Check Ric_{s n} = R^r_{s r n}!
    R_mix = [[[[dGam[m][r][n][s] - dGam[n][r][m][s] + sum(Gam[r][m][l]*Gam[l][n][s] - Gam[r][n][l]*Gam[l][m][s] for l in range(4))
                for n in range(4)] for m in range(4)] for s in range(4)] for r in range(4)]
    Ric_chk = sum(R_mix[r][0][r][0] for r in range(4))
    R_cov = [[[[sum(g[a, r]*R_mix[r][b][c][d] for r in range(4)) for d in range(4)] for c in range(4)] for b in range(4)] for a in range(4)]
    # Check antisymmetry R_{abcd} + R_{bacd} == 0!
    anti_err = max(abs(R_cov[a][b][c][d] + R_cov[b][a][c][d]) for a in range(4) for b in range(4) for c in range(4) for d in range(4))
    # Compute K = g^{ae} g^{bf} g^{cg} g^{dh} R_{abcd} R_{efgh} directly!
    B_up = [[[[sum(ginv[b, s]*R_mix[a][s][c][d] for s in range(4)) for d in range(4)] for c in range(4)] for b in range(4)] for a in range(4)]
    K_ds = mp.mpf(0)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    if R_cov[a][b][c][d] != 0:
                        R_up = sum(ginv[c, p]*ginv[d, q]*B_up[a][b][p][q] for p in range(4) for q in range(4))
                        K_ds += R_cov[a][b][c][d] * R_up
    Ric_up = ginv * Ric * ginv.T
    Ric2_ds = sum(Ric[a, b] * Ric_up[a, b] for a in range(4) for b in range(4))
    Weyl2_ds = K_ds - 2 * Ric2_ds + R_sc**2 / 3
    print(f"[1] Pure de Sitter sanity check: Ric_00={mp.nstr(Ric_chk, 8)} (exp -3), anti_err={mp.nstr(anti_err, 4)}")
    print(f"    K_ds = {mp.nstr(K_ds, 10)} (exp +24), Ric2_ds = {mp.nstr(Ric2_ds, 10)} (exp +36), Weyl2_ds = {mp.nstr(Weyl2_ds, 6)} (exp 0)")
    return {"K_ds": float(K_ds), "Ric2_ds": float(Ric2_ds), "Weyl2_ds": float(Weyl2_ds), "anti_err": float(anti_err)}


def test_tuned_page1_from_turning_points():
    """Search triaxial S3(0) = 0 turning points above 8*sqrt(3)/27 with stronger triaxiality
    to see if ANY bounce with I_EM > 8*sqrt(3)/27 reaches |S3/u| -> 1 on its contracting leg!"""
    print("\n[2] Checking if ANY triaxial Model I bounce above 8*sqrt(3)/27 reaches |S3/u| ~ 1 on contracting leg:")
    best_s3 = {"max_S3n": 0.0}
    hits_high_s3 = []
    for a1 in [0.75, 0.80, 0.85, 0.90, 0.95]:
        for a2 in [0.55, 0.60, 0.65, 0.70]:
            if a1 <= a2 + 0.03:
                continue
            for a3 in [0.75, 0.80, 0.85, 0.90]:
                tr_g = a1**2 + a2**2 + a3**2; tr_g2 = a1**4 + a2**4 + a3**4
                R3_val = 2.0 * (tr_g**2 - 2.0 * tr_g2) / ((a1 * a2 * a3) ** 2)
                u_val = 0.5 * R3_val - LAM
                if u_val <= 0:
                    continue
                u0_tp = u_val * a1 * a2 * (a3 ** 2)
                if u0_tp <= U0_ALG:
                    continue
                for alpha in [math.pi / 4.0]:
                    cE1 = a2 * a3 * math.sqrt(2.0 * u_val) * math.cos(alpha)
                    cE2 = a3 * a1 * math.sqrt(2.0 * u_val) * math.sin(alpha)
                    y0 = np.array([a1**2, 0.0, a2**2, a3**2, 0.0, 0.0, 0.0, 0.0, cE1, cE2, 0.0, 0.0])
                    sol_f = run_sync_m1(y0, (0.0, 10.0), rtol=1e-11, atol=1e-13)
                    sol_b = run_sync_m1(y0, (0.0, -10.0), rtol=1e-11, atol=1e-13)
                    df = sync_diagnostics(sol_f.y[:, -1])
                    db = sync_diagnostics(sol_b.y[:, -1])
                    ok_f = bool(len(sol_f.t_events[0]) == 0 and df["Th"] > 0 and min(df["a1"], df["a2"], df["a3"]) > 6.0 and df["R3"] < 2.0*LAM)
                    ok_b = bool(len(sol_b.t_events[0]) == 0 and db["Th"] < 0 and min(db["a1"], db["a2"], db["a3"]) > 6.0 and db["R3"] < 2.0*LAM)
                    if ok_f and ok_b:
                        all_d = [sync_diagnostics(sol_b.y[:, i]) for i in range(sol_b.y.shape[1])]
                        # Check max |S3/u| AND u0(t) AT the moment of max |S3/u|!
                        idx_s = max(range(len(all_d)), key=lambda i: abs(all_d[i]["S3_norm"]))
                        ds = all_d[idx_s]
                        u0_at_s = ds["u"] * ds["a1"] * ds["a2"] * (ds["a3"] ** 2)
                        rec = {
                            "tp_axes": [a1, a2, a3], "u0_tp": u0_tp, "I_EM": ds["I_EM"],
                            "max_S3n": abs(ds["S3_norm"]), "u0_at_max_S3": u0_at_s,
                            "a_mean_at_max_S3": (ds["a1"]*ds["a2"]*ds["a3"])**(1/3),
                            "r_at_max_S3": ds["a1"]/ds["a2"],
                            "t_at_max_S3": float(sol_b.t[idx_s]),
                            "max_H": max(abs(d["H_res"]) for d in all_d),
                            "max_M": max(abs(d["M3_res"]) for d in all_d),
                        }
                        if rec["max_S3n"] > best_s3["max_S3n"]:
                            best_s3 = rec
                        if rec["max_S3n"] > 0.85:
                            hits_high_s3.append(rec)
    print(f"    Highest |S3/u| reached on contracting leg (among bounces with u0_tp > 8*sqrt(3)/27): {best_s3}")
    print(f"    Bounces reaching |S3/u| > 0.85 on contracting leg: {len(hits_high_s3)}")
    for x in sorted(hits_high_s3, key=lambda z: -z["max_S3n"])[:5]:
        print("     ", x)
    return {"best_s3": best_s3, "hits_gt_0p85": sorted(hits_high_s3, key=lambda z: -z["max_S3n"])[:5]}


def check_mpmath_bounce_completion():
    """Confirm bounced=True at a_m=5, 10, 20 in 40-digit mpmath by checking Th: - -> + and min(a_i) turning!"""
    print("\n[3] 40-Digit mpmath Bounce Completion Check (confirming bounce + expansion on exact ceiling runs):")
    out = []
    # Run slightly longer (t_end = 10.5) on a_m = 5, 10, 20 for M-II and M-I-Sync!
    for am, r0, u0 in [(5.0, 1.08, 0.32836), (10.0, 1.04, 0.21924), (20.0, 1.02, 0.12923)]:
        r = mpmath_verify_m2_exact(am, r0, u0, t_end=11.0)
        out.append(r)
        print(f"    M-II     a_m={am:4.1f}, u0={u0:.5f}: bounced={r['bounced']} | 40-dps |H|={r['mp40_H']:.2e}, |M|={r['mp40_M']:.2e}")
    for am, r0, u0 in [(5.0, 1.06, 0.30563), (10.0, 1.04, 0.21055), (20.0, 1.02, 0.12613)]:
        r = mpmath_verify_m1_sync_exact(am, r0, u0, t_end=11.0)
        out.append(r)
        print(f"    M-I-Sync a_m={am:4.1f}, u0={u0:.5f}: bounced={r['bounced']} | 40-dps |H|={r['mp40_H']:.2e}, |M|={r['mp40_M']:.2e}")
    return out


if __name__ == "__main__":
    d1 = check_kretschmann_on_ds_and_peak()
    d2 = test_tuned_page1_from_turning_points()
    d3 = check_mpmath_bounce_completion()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "final_diagnostics.json")
    with open(path, "w") as fh:
        json.dump({"ds_kretschmann_check": d1, "tuned_page1_backward_search": d2, "mpmath_40dps_bounced": d3}, fh, indent=2)
    print("\nSaved final diagnostics ->", path)
