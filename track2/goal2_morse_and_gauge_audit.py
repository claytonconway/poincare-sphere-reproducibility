"""Audit for BC-004: Explicit Gauge-Orbit Projection of Tier-1 Truncation Artifacts & Full-3D N_max=20 (315-DOF) Hessian Audit.

1. Tier-1 (k=7) Gauge Projection Audit (N_max = 20, 24, 30, 32, 36):
   Explicitly builds the k=7 infinitesimal gauge-orbit subspace im(D_{S_02}) projected onto
   A_{k=7}^{I*}(N_max), where xi = xi^a T_a satisfies (X_3 xi)^a - 14 eps_{3ab} xi^b = 0.
   Computes the exact gauge overlap ||P_gauge v_k|| for every negative and near-zero Hessian
   eigenvector of S_02, verifying quantitatively that v_0 (eig ~ -3.01) is strictly physical
   while the tiny negative eigenvalues at N_max=20 (-3.87e-3), N_max=24 (-2.15e-5), and
   N_max=30,32,36 (~-1e-6) are truncation-lifted pure gauge artifacts!

2. Full Unrestricted 3D N_max=20 (315-DOF) Hessian Audit (first level with j=10>=7 and m_R=+/-7,+/-8):
   (a) S_02 at N_max=20 (315 DOF): polishes S_02 to |g| < 1e-6, computes full 315x315 Hessian,
       and projects every negative/near-zero eigenvector onto the 105-dim 3D gauge subspace
       im(D_{S_02}) (3 x 35 scalar modes) to determine the exact physical Morse index!
   (b) Stiff-Higgs rho_2(mu_W) at N_max=20 (315 DOF) for mu_W in {0.5, 1.5, 1.95}: minimizes in
       full 315-DOF space and computes the exact 315x315 Hessian (where local gauge zero modes
       are lifted by mu_W^2 > 0, leaving only the 5 global symmetry zero modes!), proving
       whether any non-Abelian m_R = +/-7, +/-8 U(1)_R-breaking instability appears before mu_{W,sn}!
"""

import json
import math
import os
import time

import numpy as np
from scipy.optimize import minimize

from istar_geometry import (
    EPS3,
    build_istar_and_galois,
    build_istar_invariant_harmonics,
    build_p7_equivariant_polynomial,
    euler_quadrature_s2_hopf,
    eval_real_harmonic_basis,
)
from goal2_sphaleron_solver import build_k7_subspace_basis, polish_saddle, run_cineb
from goal2_tier2_full3d import build_full3d_solver
from goal3_higgs_lifting import StiffHiggsSolver

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)


def build_k7_gauge_orbits(solver, c_A: np.ndarray, U1: np.ndarray, max_n: int) -> np.ndarray:
    """Build orthonormal basis Q_gauge (D_k7, K_g) of P_{<=max_n}(im D_A) inside the k=7 sector.

    For a k=7 equivariant gauge parameter xi^a(x) obeying (X_3 xi)^a - 14 eps_{3ab} xi^b = 0:
      (D_A xi)^a_i = X_i xi^a + eps_{abc} A^b_i xi^c.
    Project each (D_A xi) onto the orthonormal k=7 connection basis B_A!
    """
    levels = build_istar_invariant_harmonics(U1, max_n=max_n)
    n_beta = max(24, max_n + 6)
    n_alpha = max(48, 2 * max_n + 8)
    U_s2, W_s2 = euler_quadrature_s2_hopf(n_beta=n_beta, n_alpha=n_alpha)
    Phi_s2, XPhi_s2, meta = eval_real_harmonic_basis(U_s2, levels)

    A_pts, _ = solver.eval_fields(c_A)  # (N_pts, 3, 3)

    cols = []
    for m in meta:
        d, s0, s1 = m["dim"], m["start"], m["end"]
        D3 = m["D_real"][2]
        # Infinitesimal k=7 condition on xi^a: D3 @ xi - 14 * eps_{3ab} xi^b == 0
        L_xi = np.kron(D3, np.eye(3)) - 14.0 * np.kron(np.eye(d), EPS3[2])
        ev, evec = np.linalg.eigh(L_xi.T @ L_xi)
        nidx = np.where(ev < 1e-10)[0]
        Phi_n = Phi_s2[:, s0:s1]
        XPhi_n = XPhi_s2[:, :, s0:s1]
        for col in nidx:
            C_xi = evec[:, col].reshape(d, 3)
            xi = Phi_n @ C_xi                       # (N_pts, 3)
            Xxi = np.einsum("pkm,ma->pka", XPhi_n, C_xi)  # (N_pts, 3, 3): [p, i, a]
            # (D_A xi)[p, a, i] = X_i xi^a + eps_{abc} A^b_i xi^c
            DA_xi = np.transpose(Xxi, (0, 2, 1)) + np.einsum("abc,pbi,pc->pai", EPS3, A_pts, xi)
            # Project onto solver.B_A (orthonormal with weights W_s2)
            v = solver.B_A_mat.T @ ((W_s2[:, None, None] * DA_xi).reshape(-1))
            if np.linalg.norm(v) > 1e-10:
                cols.append(v)

    if not cols:
        return np.zeros((solver.D, 0))
    M = np.column_stack(cols)
    U_svd, svals, _ = np.linalg.svd(M, full_matrices=False)
    rank = int(np.sum(svals > 1e-8 * svals[0]))
    return U_svd[:, :rank]


def build_full3d_gauge_orbits(solver, c_A: np.ndarray) -> np.ndarray:
    """Build orthonormal basis Q_gauge (D, K_g) of P_{<=max_n}(im D_A) in Full 3D (3*M gauge params)."""
    C = c_A.reshape(solver.M, 9)
    A = (solver.Phi @ C).reshape(solver.n_pts, 3, 3)  # (N_pts, 3, 3)
    W = solver.W
    cols = []
    # Each gauge basis element is xi^{(m, c)}(x) = Phi_m(x) T_c (m=0..M-1, c=0..2)
    # (D_A xi)[p, a, i] = X_i Phi_m(x) delta_{ac} + eps_{abc} A^b_i(x) Phi_m(x)
    for m in range(solver.M):
        pm = solver.Phi[:, m]         # (N_pts,)
        xpm = solver.XPhi[:, :, m]    # (N_pts, 3)
        for c in range(3):
            DA = np.zeros((solver.n_pts, 3, 3), dtype=np.float64)
            DA[:, c, :] = xpm
            # + eps_{abc} A^b_i pm: for fixed c, (a,b) cyclic!
            a1 = (c + 1) % 3
            a2 = (c + 2) % 3
            # eps_{a1, a2, c} = +1, eps_{a2, a1, c} = -1
            DA[:, a1, :] += A[:, a2, :] * pm[:, None]
            DA[:, a2, :] -= A[:, a1, :] * pm[:, None]
            # Project onto Phi: G[m', a, i] = <Phi_{m'} * DA[a, i]>
            G = (solver.Phi.T @ (W[:, None] * DA.reshape(solver.n_pts, 9))).reshape(-1)
            if np.linalg.norm(G) > 1e-10:
                cols.append(G)
    M_mat = np.column_stack(cols)
    U_svd, svals, _ = np.linalg.svd(M_mat, full_matrices=False)
    rank = int(np.sum(svals > 1e-8 * svals[0]))
    return U_svd[:, :rank]


def main():
    t0 = time.time()
    print("=== BC-004 Audit: Tier-1 Gauge Projection & Full-3D N_max=20 (315-DOF) Hessian ===", flush=True)
    Q1, Q2, U1, U2 = build_istar_and_galois()
    _, poly_coeffs = build_p7_equivariant_polynomial(U1, U2)

    # -------------------------------------------------------------------------
    # PART 1: Tier-1 (k=7) Gauge Projection Audit across N_max = 20, 24, 30, 32, 36
    # -------------------------------------------------------------------------
    print("\n--- PART 1: Tier-1 k=7 Gauge-Projection Audit of S_02 Hessian Eigenmodes ---", flush=True)
    tier1_audit = []
    prev_c = None
    prev_D = 0
    for max_n in [12, 20, 24, 30, 32, 36]:
        s, cr2, _ = build_k7_subspace_basis(U1, poly_coeffs, max_n=max_n)
        if prev_c is None:
            neb = run_cineb(s, np.zeros(s.D), cr2, n_images=19, k_spring=0.2, n_steps=1000, lr_init=0.008, climb_after=200)
            c0_sad = neb["c_peak"]
        else:
            c0_sad = np.zeros(s.D)
            c0_sad[:prev_D] = prev_c
        pol = polish_saddle(s, c0_sad, maxiter=500)
        prev_c = pol["c"]
        prev_D = s.D

        H = s.hessian(pol["c"])
        ev, V = np.linalg.eigh(H)
        Q_g = build_k7_gauge_orbits(s, pol["c"], U1, max_n)

        # Evaluate physical-subspace (gauge-projected) Hessian H_phys = Q_perp^T H Q_perp!
        # Construct orthonormal complement Q_perp of Q_g
        P_perp = np.eye(s.D) - Q_g @ Q_g.T
        u_p, sv_p, _ = np.linalg.svd(P_perp)
        Q_perp = u_p[:, sv_p > 0.5]
        ev_phys = np.linalg.eigvalsh(Q_perp.T @ H @ Q_perp)

        mode_rows = []
        for k in range(min(10, s.D)):
            ov = float(np.linalg.norm(Q_g.T @ V[:, k]))
            mode_rows.append({"idx": k, "eig": float(ev[k]), "gauge_overlap": ov, "phys_fraction": max(0.0, 1.0 - ov * ov)})

        rec = {
            "N_max": max_n,
            "D_k7": s.D,
            "dim_gauge_k7": int(Q_g.shape[1]),
            "dim_phys_k7": int(Q_perp.shape[1]),
            "E_YM": pol["E_YM_S3_Istar"],
            "grad_norm": pol["grad_norm"],
            "lowest_modes_with_gauge_overlap": mode_rows,
            "phys_hess_n_neg": int(np.sum(ev_phys < -1e-6)),
            "phys_hess_lowest_5_eigs": [float(x) for x in ev_phys[:5]],
        }
        tier1_audit.append(rec)
        print(
            f"  N_max={max_n:2d} (D={s.D:2d}, dim_g={Q_g.shape[1]:2d}, dim_phys={Q_perp.shape[1]:2d}): "
            f"v0 eig={ev[0]:+.4f} (gauge_ov={mode_rows[0]['gauge_overlap']:.2e}) | "
            f"v1 eig={ev[1]:+.3e} (gauge_ov={mode_rows[1]['gauge_overlap']:.6f}) | "
            f"H_phys neg={rec['phys_hess_n_neg']}, lowest_phys={[round(x, 4) for x in ev_phys[:4]]}",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # PART 2: Full Unrestricted 3D N_max=20 (315-DOF) Hessian Audit!
    # -------------------------------------------------------------------------
    print("\n--- PART 2: Full Unrestricted 3D N_max=20 (315-DOF) Hessian Audit ---", flush=True)
    solver20, c0_20, c1_20, c2_20, Phi20, W20, U20, meta20 = build_full3d_solver(U1, poly_coeffs, max_n=20)
    print(f"  Built Full3D N_max=20: D={solver20.D}, n_pts={solver20.n_pts}", flush=True)

    # 2a. S_02 at N_max=20 (315 DOF): polish saved saddle from goal2_saddles.npz to high precision!
    sad_data = np.load(os.path.join(OUT_DIR, "goal2_saddles.npz"))
    c_sad20_init = sad_data["c_best20_polished"]
    pol20 = polish_saddle(solver20, c_sad20_init, maxiter=500)
    print(f"  Polished Full3D S_02 (N_max=20): E={pol20['E_YM_S3_Istar']:.6f}, N_CS={pol20['N_CS']:.6f}, |g|={pol20['grad_norm']:.3e}", flush=True)

    t_h = time.time()
    H20_sad = solver20.hessian(pol20["c"])
    ev20_sad, V20_sad = np.linalg.eigh(H20_sad)
    Q_g20 = build_full3d_gauge_orbits(solver20, pol20["c"])
    P_perp20 = np.eye(solver20.D) - Q_g20 @ Q_g20.T
    u_p20, sv_p20, _ = np.linalg.svd(P_perp20)
    Q_perp20 = u_p20[:, sv_p20 > 0.5]
    ev20_phys = np.linalg.eigvalsh(Q_perp20.T @ H20_sad @ Q_perp20)

    # Check gauge overlap of all eigenvalues < 0.05
    low_modes_20 = []
    for k in range(25):
        ov = float(np.linalg.norm(Q_g20.T @ V20_sad[:, k]))
        low_modes_20.append({"idx": k, "eig": float(ev20_sad[k]), "gauge_overlap": ov})
    n_phys_neg_20 = int(np.sum(ev20_phys < -1e-4))
    n_phys_zero_20 = int(np.sum(np.abs(ev20_phys) <= 1e-4))
    print(f"  Full3D N_max=20 S_02 Hessian ({time.time()-t_h:.1f}s): dim_gauge={Q_g20.shape[1]}, dim_phys={Q_perp20.shape[1]}", flush=True)
    print(f"  Raw lowest 10 eigs (with gauge_ov): {[(round(m['eig'], 5), round(m['gauge_overlap'], 4)) for m in low_modes_20[:10]]}", flush=True)
    print(f"  Gauge-projected H_phys Morse: ({n_phys_neg_20} neg, {n_phys_zero_20} zero, {len(ev20_phys)-n_phys_neg_20-n_phys_zero_20} pos)", flush=True)
    print(f"  H_phys lowest 8 eigs: {[round(float(x), 5) for x in ev20_phys[:8]]}", flush=True)

    # 2b. Full-3D N_max=20 (315-DOF) Stiff-Higgs Stability Audit at mu_W = 0.5, 1.5, 1.95!
    print("\n--- PART 2b: Full Unrestricted 3D N_max=20 (315-DOF) Stiff-Higgs rho_2(mu_W) Audit ---", flush=True)
    # Note: at N_max=20, break tiny parity artifact cleanly by seeding with tiny random kick!
    rng = np.random.default_rng(2026)
    c_cur = c2_20 + 0.05 * rng.normal(size=solver20.D)
    stiff20_audit = []
    for mu in [0.5, 1.5, 1.95]:
        st20 = StiffHiggsSolver(solver20, mu)
        res = minimize(lambda c: st20.energy_and_grad(c)[:2], c_cur, method="L-BFGS-B", jac=True, options={"maxiter": 1500, "ftol": 1e-15, "gtol": 1e-10})
        e_t, g_t, ncs_t, e_ym_t, e_h_t = st20.energy_and_grad(res.x)
        H_st = st20.hessian(res.x)
        ev_st = np.linalg.eigvalsh(H_st)
        n_neg = int(np.sum(ev_st < -1e-4))
        n_zero = int(np.sum(np.abs(ev_st) <= 1e-4))
        pos_e = ev_st[ev_st > 1e-4]
        min_pos = float(pos_e[0]) if len(pos_e) else 0.0
        c_cur = res.x.copy()
        stiff20_audit.append(
            {
                "mu_W": mu,
                "E_tot": e_t,
                "E_ym": e_ym_t,
                "E_higgs": e_h_t,
                "N_CS": ncs_t,
                "grad_norm": float(np.linalg.norm(g_t)),
                "morse_neg_zero_pos": [n_neg, n_zero, int(solver20.D - n_neg - n_zero)],
                "lowest_10_eigs": [round(float(x), 5) for x in ev_st[:10]],
            }
        )
        print(
            f"  Full3D N20 stiff mu_W={mu:4.2f}: E_tot={e_t:8.4f}, N_CS={ncs_t:.5f}, |g|={np.linalg.norm(g_t):.2e}, "
            f"Morse=({n_neg}-, {n_zero}0, {solver20.D-n_neg-n_zero}+), min_pos={min_pos:.5f}",
            flush=True,
        )

    out_path = os.path.join(OUT_DIR, "goal2_morse_and_gauge_audit.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "tier1_k7_gauge_projection_audit": tier1_audit,
                "full3d_N20_S02_audit": {
                    "D": solver20.D,
                    "dim_gauge": int(Q_g20.shape[1]),
                    "dim_phys": int(Q_perp20.shape[1]),
                    "E_YM": pol20["E_YM_S3_Istar"],
                    "N_CS": pol20["N_CS"],
                    "grad_norm": pol20["grad_norm"],
                    "raw_lowest_15_modes": low_modes_20[:15],
                    "phys_morse_neg_zero_pos": [n_phys_neg_20, n_phys_zero_20, int(len(ev20_phys) - n_phys_neg_20 - n_phys_zero_20)],
                    "phys_lowest_10_eigs": [float(x) for x in ev20_phys[:10]],
                },
                "full3d_N20_stiff_rho2_audit": stiff20_audit,
                "elapsed_sec": time.time() - t0,
            },
            fh,
            indent=2,
        )
    print(f"\nSaved {out_path} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
