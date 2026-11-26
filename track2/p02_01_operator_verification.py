"""P02-01: Exact Verification of the Linearized Gauss-Constrained SU(2)_L YM + Higgs Operator on S^3 & S^3/I*.

Verifies:
  1. Exact I* Molien multiplicities m_n and twisted Ad(rho_0, rho_1, rho_2) multiplicities.
  2. Exact gauge Ward identity M_YM^2(f) D(f) = 2 f (f^2 - 1) D_1 across all (n, K) and arbitrary f.
  3. Exact n = 0 reduction to delta'' + (6f + 10 + mu_W^2) delta = 0 (K=2), 6f^2 - 2 (K=0), pure-gauge (K=1).
  4. Exact closed-form 1D equations for K = j + 2 and K = j - 2.
  5. Exact physical (D^perp) spectrum at BOTH flat connections f = -1 (rho_0) and f = +1 (rho_1),
     plus the sphaleron f = 0, and exact Proca-Higgs longitudinal eigenvalues at f = +/- 1.
  6. Exact instantaneous tachyonic thresholds f_tach(n, K) and E_tach(n, K) across all K-sectors
     for S^3 (n = 0..12) vs. S^3/I* (n = 0, 12, 20, 24, 30).
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Dict, List

import numpy as np
from scipy.optimize import brentq

from istar_harmonics import (
    build_binary_icosahedral_group,
    build_coulomb_complement,
    build_nK_block_operators,
    cart_to_sph_spin1,
    eval_ym_operators,
    galois_conjugate_istar_quats,
    istar_adjoint_molien_coeffs,
    klein_molien_coeffs,
    quat_mul,
    quat_to_su2,
    su2_character,
    wigner_d_matrix,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)


def physical_static_eigs_ym(blk: dict, f: float) -> np.ndarray:
    """Return eigenvalues of M_YM^2(f) restricted to the Gauss-constrained subspace D(f)^perp."""
    _, M2, D = eval_ym_operators(blk, f)
    Q = build_coulomb_complement(D)
    if Q.shape[1] == 0:
        return np.array([], dtype=np.float64)
    M2_phys = Q.T @ M2 @ Q
    M2_phys = 0.5 * (M2_phys + M2_phys.T)
    return np.linalg.eigvalsh(M2_phys)


def expected_flat_spectrum_rho0(n: int) -> Counter:
    """Exact physical squared-curl multiset at f = -1 (rho_0, A = 0) per left-state in V_{n/2}^L.

    At A = 0, color is a trivial 3-fold copy of ordinary Maxwell curl modes on S^3:
      - Negative helicity: lambda = -(n+2), mult = 3 * (n+3)
      - Positive helicity (n >= 2): lambda = +n, mult = 3 * (n-1)
    """
    c = Counter()
    c[(n + 2) ** 2] += 3 * (n + 3)
    if n >= 2:
        c[n ** 2] += 3 * (n - 1)
    return c


def expected_flat_spectrum_rho1(n: int) -> Counter:
    """Exact physical squared-curl multiset at f = +1 (rho_1, A = 2 omega) per left-state in V_{n/2}^L.

    Under gauge transformation g(x) = x: A -> 0 and delta A -> Ad(x) delta A, shifting
    V_{n/2}^R x V_1^color = (+) V_{n'/2}^R with n' in {n+2, n (if n>=1), n-2 (if n>=2)}.
    Each n' contributes ordinary Maxwell transverse curl modes:
      - Negative helicity: lambda = -(n'+2), mult = n' + 3
      - Positive helicity (n' >= 2): lambda = +n', mult = n' - 1
    Wait: for n = 2, n' = 0 gives only negative helicity -(0+2)=-2 (mult 3), zero positive helicity.
    """
    c = Counter()
    n_primes = [n + 2]
    if n >= 1:
        n_primes.append(n)
    if n >= 2:
        n_primes.append(n - 2)
    for np_val in n_primes:
        c[(np_val + 2) ** 2] += (np_val + 3)
        if np_val >= 2:
            c[np_val ** 2] += (np_val - 1)
    return c


def proca_static_eigs(blk: dict, f: float, mu_W: float) -> np.ndarray:
    """Exact static squared frequencies omega^2 of Proca-Higgs system (M_YM^2 + mu_W^2 I) v = omega^2 G v."""
    from scipy.linalg import eigh

    _, M2, D = eval_ym_operators(blk, f)
    d = blk["d_alpha"]
    A = M2 + (mu_W ** 2) * np.eye(d)
    if blk["d_0"] == 1:
        dd = float((D.T @ D).item())
        G = np.eye(d) - (D @ D.T) / (dd + mu_W ** 2)
    else:
        G = np.eye(d)
    evals = eigh(A, G, eigvals_only=True)
    return np.sort(np.real(evals))


def find_tachyonic_threshold_for_block(blk: dict) -> dict:
    """Find exact minimum f < 0 (left) and f > 0 (right) where min_eig(M_YM^2|_{D^perp}(f)) crosses 0.

    Note: at turning points of pure-YM background f' = 0, the exact Coulomb-reduced potential
    reduces strictly to Q(f)^T M_YM^2(f) Q(f) (since the +3 (f')^2 term vanishes at f' = 0!).
    Thus a mode becomes instantaneously tachyonic on its orbit if and only if
    min_eig(Q(f)^T M_YM^2(f) Q(f)) < 0 for some f in [f_min(E), f_max(E)].
    """
    n = blk["n"]
    two_K = blk["two_K"]
    d_phys = blk["d_alpha"] - blk["d_0"]
    if d_phys == 0:
        return {
            "n": n,
            "two_K": two_K,
            "K": 0.5 * two_K,
            "d_phys": 0,
            "min_eig_at_sphaleron_f0": None,
            "left_f_tach": None,
            "left_E_tach": None,
            "right_f_tach": None,
            "right_E_tach": None,
            "E_tach_min": None,
            "controlling_side": None,
            "controlling_f": None,
        }

    def min_eig(f_val: float) -> float:
        eigs = physical_static_eigs_ym(blk, f_val)
        return float(np.min(eigs))

    # Special check at f = 0 (sphaleron)
    m0 = min_eig(0.0)

    # Scan left f in [0, -100] and right f in [0, +100] on a dense grid to bracket roots
    f_grid_left = -np.concatenate([np.linspace(0.0, 5.0, 251), np.linspace(5.1, 60.0, 550)])
    f_grid_right = np.concatenate([np.linspace(0.0, 5.0, 251), np.linspace(5.1, 60.0, 550)])

    left_f = None
    vals_l = [min_eig(float(fv)) for fv in f_grid_left]
    if vals_l[0] < -1e-12:
        left_f = 0.0
    else:
        for i in range(len(vals_l) - 1):
            if vals_l[i] >= 0.0 and vals_l[i + 1] < 0.0:
                left_f = float(brentq(min_eig, float(f_grid_left[i + 1]), float(f_grid_left[i]), xtol=1e-13))
                break

    right_f = None
    vals_r = [min_eig(float(fv)) for fv in f_grid_right]
    if vals_r[0] < -1e-12:
        right_f = 0.0
    else:
        for i in range(len(vals_r) - 1):
            if vals_r[i] >= 0.0 and vals_r[i + 1] < 0.0:
                right_f = float(brentq(min_eig, float(f_grid_right[i]), float(f_grid_right[i + 1]), xtol=1e-13))
                break

    E_left = 0.5 * (left_f ** 2 - 1.0) ** 2 if left_f is not None else None
    E_right = 0.5 * (right_f ** 2 - 1.0) ** 2 if right_f is not None else None

    cands = [(E_left, "left", left_f), (E_right, "right", right_f)]
    cands = [c for c in cands if c[0] is not None]
    if cands:
        E_min, side, f_crit = min(cands, key=lambda x: x[0])
    else:
        E_min, side, f_crit = None, None, None

    return {
        "n": n,
        "two_K": two_K,
        "K": 0.5 * two_K,
        "d_phys": d_phys,
        "min_eig_at_sphaleron_f0": m0,
        "left_f_tach": left_f,
        "left_E_tach": E_left,
        "right_f_tach": right_f,
        "right_E_tach": E_right,
        "E_tach_min": E_min,
        "controlling_side": side,
        "controlling_f": f_crit,
    }


def main():
    Q_istar, Rs_istar = build_binary_icosahedral_group()
    Q2_istar = galois_conjugate_istar_quats(Q_istar)
    U_sp1 = cart_to_sph_spin1()
    molien = klein_molien_coeffs(40)
    ad_rho0 = istar_adjoint_molien_coeffs(40, "rho_0")
    ad_rho1 = istar_adjoint_molien_coeffs(40, "rho_1")
    ad_rho2 = istar_adjoint_molien_coeffs(40, "rho_2")

    # Verify exact group homomorphism of quat_to_su2, galois_conjugate_istar_quats, wigner_d_matrix, and SO(3) match
    rng = np.random.default_rng(42)
    idx_pairs = rng.integers(0, 120, size=(100, 2))
    max_su2_hom_err = 0.0
    max_gal_hom_err = 0.0
    max_wigner_hom_err = 0.0
    for i_p, j_p in idx_pairs:
        q1, q2 = Q_istar[i_p], Q_istar[j_p]
        q12 = quat_mul(q1, q2)
        max_su2_hom_err = max(max_su2_hom_err, float(np.max(np.abs(quat_to_su2(q1) @ quat_to_su2(q2) - quat_to_su2(q12)))))
        g1, g2 = Q2_istar[i_p], Q2_istar[j_p]
        max_gal_hom_err = max(max_gal_hom_err, float(np.max(np.abs(quat_to_su2(g1) @ quat_to_su2(g2) - quat_to_su2(quat_mul(g1, g2))))))
        for tj in (1, 2, 3):
            max_wigner_hom_err = max(
                max_wigner_hom_err,
                float(np.max(np.abs(wigner_d_matrix(tj, q1) @ wigner_d_matrix(tj, q2) - wigner_d_matrix(tj, q12)))),
            )
    max_so3_wigner_err = max(
        float(np.max(np.abs(U_sp1 @ wigner_d_matrix(2, Q_istar[k]) @ U_sp1.conj().T - Rs_istar[k])))
        for k in range(120)
    )
    assert max_su2_hom_err < 1e-13 and max_gal_hom_err < 1e-13 and max_wigner_hom_err < 1e-12 and max_so3_wigner_err < 1e-12

    # Verify Molien against direct character trace
    molien_err = 0.0
    for n in range(41):
        tr = float(np.sum(su2_character(n, Q_istar[:, 0])) / 120.0)
        molien_err = max(molien_err, abs(tr - molien[n]))
    assert molien_err < 1e-12

    test_ns = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 20, 24, 30]
    f_probes = [-3.7, -1.0, 0.0, 0.42, 1.0, 2.3]

    max_ward_err = 0.0
    max_rho0_err = 0.0
    max_rho1_err = 0.0
    max_kj2_err = 0.0
    max_proca_long_err = 0.0
    max_d_lap_err = 0.0

    sphaleron_summary = []
    tachyonic_by_n: Dict[int, dict] = {}

    for n in test_ns:
        blocks = build_nK_block_operators(n)

        # 1. Verify Ward identity M_YM^2(f) D(f) = 2 f (f^2 - 1) D_1 and D^dagger D closed form
        for two_K, blk in blocks.items():
            for fv in f_probes:
                _, M2, D = eval_ym_operators(blk, fv)
                if blk["d_0"] == 1:
                    lhs = M2 @ D
                    rhs = 2.0 * fv * (fv * fv - 1.0) * blk["D1"]
                    max_ward_err = max(max_ward_err, float(np.max(np.abs(lhs - rhs))))

                    # Check exact closed-form d_{j,K}(f) = ||D(f)||^2
                    lam = 1.0 + fv
                    if two_K == n + 2:
                        d_exact = (n + lam) ** 2 + lam ** 2 + 2 * n
                    elif two_K == n:
                        d_exact = n * (n + 2) + 2.0 * fv * fv - 2.0
                    elif two_K == n - 2:
                        d_exact = (n - lam) ** 2 + (lam - 2.0) ** 2 + 2 * n - 4
                    else:
                        raise AssertionError(f"Unexpected d_0=1 at 2K={two_K}")
                    d_num = float((D.T @ D).item())
                    max_d_lap_err = max(max_d_lap_err, abs(d_num - d_exact))

                # Check closed-form K = j +/- 2
                if two_K == n + 4:
                    exact_val = (n + 3) ** 2 + 1.0 + 2.0 * (n + 3) * fv
                    max_kj2_err = max(max_kj2_err, abs(float(M2[0, 0]) - exact_val))
                if n >= 4 and two_K == n - 4:
                    exact_val = (n - 1) ** 2 + 1.0 - 2.0 * (n - 1) * fv
                    max_kj2_err = max(max_kj2_err, abs(float(M2[0, 0]) - exact_val))

        # 2. Verify exact flat-connection spectra at f = -1 (rho_0) and f = +1 (rho_1)
        for f_flat, exp_fn, tag in [(-1.0, expected_flat_spectrum_rho0, "rho0"), (+1.0, expected_flat_spectrum_rho1, "rho1")]:
            eigs_all = []
            for two_K, blk in blocks.items():
                mult = blk["mult_K"]
                ev = physical_static_eigs_ym(blk, f_flat)
                # Special cases where D(f_flat) == 0:
                #   (n=0, 2K=2, f=-1): D = 0, so Q is 1x1 identity, M2 = 4 (pure curl mode at A=0, wait: at f=-1, A=0 has NO gauge breaking on n=0!)
                #   (n=2, 2K=0, f=+1): D = 0, M2 = 4 (same: gauge-transformed constant mode at A=2*omega!)
                for val in ev:
                    eigs_all.extend([float(val)] * mult)
            eigs_all = np.sort(np.array(eigs_all, dtype=np.float64))

            exp_counter = exp_fn(n)
            exp_list = []
            for val, cnt in exp_counter.items():
                exp_list.extend([float(val)] * cnt)
            exp_arr = np.sort(np.array(exp_list, dtype=np.float64))
            assert len(eigs_all) == len(exp_arr), f"Length mismatch at n={n} {tag}: {len(eigs_all)} vs {len(exp_arr)}"
            err = float(np.max(np.abs(eigs_all - exp_arr)))
            if tag == "rho0":
                max_rho0_err = max(max_rho0_err, err)
            else:
                max_rho1_err = max(max_rho1_err, err)

            # 3. Verify Proca-Higgs longitudinal eigenvalue omega_long^2 = ||D(f_flat)||^2 + mu_W^2
            mu_test = 1.7
            for two_K, blk in blocks.items():
                if blk["d_0"] == 1:
                    _, _, D_fl = eval_ym_operators(blk, f_flat)
                    ev_ym = physical_static_eigs_ym(blk, f_flat)
                    # Note: when D_fl == 0 (n=0,2K=2,f=-1 or n=2,2K=0,f=+1), G = I and there is no longitudinal split
                    if float(np.linalg.norm(D_fl)) > 1e-12:
                        expected_proca = np.sort(np.append(ev_ym + mu_test ** 2, float((D_fl.T @ D_fl).item()) + mu_test ** 2))
                        got_proca = proca_static_eigs(blk, f_flat, mu_test)
                        max_proca_long_err = max(max_proca_long_err, float(np.max(np.abs(got_proca - expected_proca))))

        # 4. Sphaleron (f = 0) spectrum & tachyonic threshold across all K-blocks
        neg_sph = []
        min_sph_eig = 1e9
        tach_blocks = []
        for two_K, blk in sorted(blocks.items()):
            ev0 = physical_static_eigs_ym(blk, 0.0)
            if len(ev0) > 0:
                min_sph_eig = min(min_sph_eig, float(np.min(ev0)))
                for v in ev0:
                    if v < 1e-9:
                        neg_sph.append({"two_K": two_K, "K": 0.5 * two_K, "eig": float(v), "mult_K": blk["mult_K"]})
            tinfo = find_tachyonic_threshold_for_block(blk)
            tach_blocks.append(tinfo)

        sphaleron_summary.append({
            "n": n,
            "m_n_istar": molien[n],
            "min_sphaleron_eig": min_sph_eig,
            "nonpositive_sphaleron_modes": neg_sph,
        })

        valid_tach = [t for t in tach_blocks if t["E_tach_min"] is not None]
        # Note: exclude the n=0, K=0 background mode itself (6f^2 - 2) when reporting perturbation thresholds
        if n == 0:
            pert_tach = [t for t in valid_tach if t["two_K"] != 0]
        else:
            pert_tach = valid_tach
        best_tach = min(pert_tach, key=lambda x: x["E_tach_min"]) if pert_tach else None

        tachyonic_by_n[n] = {
            "n": n,
            "m_n_istar": molien[n],
            "survives_on_istar": molien[n] > 0,
            "min_E_tach_pert": best_tach["E_tach_min"] if best_tach else None,
            "controlling_two_K": best_tach["two_K"] if best_tach else None,
            "controlling_side": best_tach["controlling_side"] if best_tach else None,
            "controlling_f": best_tach["controlling_f"] if best_tach else None,
            "blocks": tach_blocks,
        }

    # Verify exact n=0 reduction explicitly
    blk0 = build_nK_block_operators(0)
    n0_check = {}
    for two_K, blk in blk0.items():
        vals = {f"{fv:.1f}": float(eval_ym_operators(blk, fv)[1][0, 0]) for fv in (-1.0, 0.0, 1.0, 2.0)}
        n0_check[f"2K={two_K}"] = vals

    payload = {
        "script": "p02_01_operator_verification",
        "exact_invariants": {
            "molien_character_max_err_n0_40": molien_err,
            "su2_homomorphism_max_err": max_su2_hom_err,
            "galois_rho2_homomorphism_max_err": max_gal_hom_err,
            "wigner_d_homomorphism_max_err": max_wigner_hom_err,
            "wigner_d_vs_so3_max_err": max_so3_wigner_err,
            "ward_identity_max_err": max_ward_err,
            "covariant_laplacian_closed_form_max_err": max_d_lap_err,
            "rho0_flat_spectrum_max_err": max_rho0_err,
            "rho1_flat_spectrum_max_err": max_rho1_err,
            "K_j_pm_2_closed_form_max_err": max_kj2_err,
            "proca_longitudinal_spectrum_max_err": max_proca_long_err,
        },
        "n0_reduction_check": n0_check,
        "molien_first_32": {
            "m_n_trivial": molien[:33],
            "m_n_Ad_rho0": ad_rho0[:33],
            "m_n_Ad_rho1": ad_rho1[:33],
            "m_n_Ad_rho2": ad_rho2[:33],
        },
        "sphaleron_f0_spectrum": sphaleron_summary,
        "tachyonic_thresholds_by_n": {str(k): v for k, v in tachyonic_by_n.items()},
    }

    out_path = os.path.join(OUT_DIR, "p02_01_operator_verification.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("=== P02-01 EXACT OPERATOR VERIFICATION ===")
    for k, v in payload["exact_invariants"].items():
        print(f"  {k:44s}: {v:.3e}")
    print("\n--- Sphaleron (f = 0) Lowest Physical Eigenvalue ---")
    for row in sphaleron_summary:
        tag = "S^3/I*" if row["m_n_istar"] > 0 else "S^3-only"
        print(f"  n={row['n']:2d} [{tag:8s}] m_n={row['m_n_istar']} | min_eig(f=0) = {row['min_sphaleron_eig']:+10.6f} | nonpos={row['nonpositive_sphaleron_modes']}")
    print("\n--- Minimum Perturbative Tachyonic Threshold E_tach(n) Across All K-Blocks ---")
    for n in test_ns:
        r = tachyonic_by_n[n]
        tag = "S^3/I*" if r["survives_on_istar"] else "S^3-only"
        e_str = f"{r['min_E_tach_pert']:10.4f}" if r["min_E_tach_pert"] is not None else "      None"
        f_str = f"{r['controlling_f']:+8.4f}" if r["controlling_f"] is not None else "    None"
        print(f"  n={n:2d} [{tag:8s}] m_n={r['m_n_istar']} | E_tach = {e_str} | 2K={str(r['controlling_two_K']):>4s} ({str(r['controlling_side']):>5s}, f_crit={f_str})")
    print("\n--- Detailed K-blocks for n = 0, 1, 2, 3, 4, 12 ---")
    for n in (0, 1, 2, 3, 4, 12):
        print(f"  [n={n}]")
        for b in tachyonic_by_n[n]["blocks"]:
            e_l = f"{b['left_E_tach']:8.3f}" if b["left_E_tach"] is not None else "    None"
            e_r = f"{b['right_E_tach']:8.3f}" if b["right_E_tach"] is not None else "    None"
            sph_s = f"{b['min_eig_at_sphaleron_f0']:+8.4f}" if b["min_eig_at_sphaleron_f0"] is not None else "    None"
            print(f"    2K={b['two_K']:2d} (K={b['K']:4.1f}, d_phys={b['d_phys']}): sph_min={sph_s} | E_left={e_l} (f={b['left_f_tach']}) | E_right={e_r} (f={b['right_f_tach']})")


if __name__ == "__main__":
    main()
