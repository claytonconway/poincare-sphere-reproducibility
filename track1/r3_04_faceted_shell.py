#!/usr/bin/env python3
"""
Round-3 Micro-Scout R3-4:
Non-Spherical I*-Equivariant Faceted Thin-Shell Israel-Darmois Matching
on the t = 0 Bounce Slice of the Poincare Homology Sphere S^3/I*.

Tests whether deforming P04's spherical thin shell (inscribed inside chi_inj = pi/10 = 18 deg,
mu/y_+ <= sin^3(pi/10) = 0.0295085 < Page 7's pi/80 = 0.0392699) into an I*-equivariant
l = 6 (S^3 harmonic degree n = 12) faceted dodecahedral shell bulging into the 20 vertex
pockets (chi_circ = 22.2388 deg) can lift the WEC / WEC+DEC / WEC+NEC bounce mass ceiling
above pi/80 = 0.0392699.

Independent Verification Pillars:
  1. Exact 60-element icosahedral rotation group I = I*/{+-1} invariance of Y_6^I (< 1e-14),
     exact eigenvalue Delta_{S^2} Y_6^I = -42 Y_6^I, exact rational extrema (+1, +9/16, -9/5),
     exact max traceless Hessian ||T_6||/sqrt(2) = 189*sqrt(5)/16 at the 30 edge midpoints,
     and exact Dirichlet dodecahedron geometry (120 spherical Schwarz right triangles).
  2. Independent 35-digit mpmath 3D/4D differential-geometry oracle verifying:
     - Exact nonlinear S^3 extrinsic curvature K^A_B^- against 3D Christoffel finite diffs,
     - Exact nonlinear Brill-Lindquist vacuum R^(3)+ = 0 and K^A_B^+ against 3D Christoffel diffs,
     - Exact 4D static Regge-Wheeler l = 6 vacuum metric (Q_6^2 and P_6^2): all 10 linearized
       4D vacuum Einstein equations delta R_{mu nu} = 0 and 4D finite-diff K^A_B^+, K^tau_tau^+,
     - Exact universal identity R_0*(8*pi*G*delta_sigma)/(eps*Y_l) = +3*sin(chi_0) at
       x = x_iso, u_0 = cos(chi_0) independent of (l, k).
  3. Complete O(eps) mass-ceiling optimization across all four exterior vacuum branches
     (Brill-Lindquist wormhole k=+6, Brill-Lindquist exterior k=-7, 4D static Regge-Wheeler
     P_6^2 and Q_6^2) + pure Schwarzschild (c=0), under WEC, ddot{chi}-independent WEC+DEC,
     and static/comoving WEC+NEC (gamma=0 and gamma=1/4).
  4. Exact nonlinear finite-amplitude evaluation on S^2 (exact nonlinear interior Hawking mass
     2 m_H^-/a_0, nonlinear Brill-Lindquist & pure Schwarzschild stress eigenvalues, and
     explicit O(eps^2) first-junction residual ||q^+ - q^-|| / ||q^-||).
"""

import json
import math
from pathlib import Path
import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import eval_legendre

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHI_INJ = math.pi / 10.0
PAGE7_CEILING = math.pi / 80.0
PHI = 0.5 * (1.0 + math.sqrt(5.0))


# =============================================================================
# 1. EXACT ICOSAHEDRAL GROUP I = I*/{+-1}, DIRICHLET DODECAHEDRON & Y_6^I
# =============================================================================

def build_dodecahedron_and_group():
    """Build exact 120 elements of I*, 60 SO(3) rotations of I, 12 face centers,
    20 dodecahedron vertices, and 30 edge midpoints."""
    # Exact 120 unit quaternions of I* in standard golden coordinates
    even_perms = [
        (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
        (1, 0, 3, 2), (1, 2, 0, 3), (1, 3, 2, 0),
        (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
        (3, 0, 2, 1), (3, 1, 0, 2), (3, 2, 1, 0),
    ]
    pts = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = [0.0, 0.0, 0.0, 0.0]
            v[i] = s
            pts.append(tuple(v))
    for s0 in (-0.5, 0.5):
        for s1 in (-0.5, 0.5):
            for s2 in (-0.5, 0.5):
                for s3 in (-0.5, 0.5):
                    pts.append((s0, s1, s2, s3))
    base = [0.0, 0.5, 0.5 * PHI, 0.5 / PHI]
    for p in even_perms:
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                for s3 in (-1.0, 1.0):
                    raw = [0.0, s1 * base[1], s2 * base[2], s3 * base[3]]
                    v = [0.0, 0.0, 0.0, 0.0]
                    for idx, pos in enumerate(p):
                        v[pos] = raw[idx]
                    pts.append(tuple(v))
    uniq = {tuple(np.round(r, 10)): r for r in np.array(pts, dtype=np.float64)}
    Q_istar = np.array(list(uniq.values()), dtype=np.float64)
    assert Q_istar.shape == (120, 4)

    # Convert each quaternion q = (w, x, y, z) to its SO(3) adjoint rotation R(q)
    # (quotienting by +-1 yields exactly 60 rotations of the icosahedral group I)
    rot_map = {}
    for q in Q_istar:
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ], dtype=np.float64)
        rot_map[tuple(np.round(R.ravel(), 8))] = R
    R_ico = np.array(list(rot_map.values()), dtype=np.float64)
    assert R_ico.shape == (60, 3, 3)

    # 12 nearest neighbors of e=(1,0,0,0) in I* give the 12 face-center directions on S^2
    # (exact match with I* quaternion coordinates!)
    dots_e = Q_istar[:, 0]
    mask_nn = np.abs(dots_e - math.cos(math.pi / 5.0)) < 1e-10
    nn = Q_istar[mask_nn]
    assert nn.shape == (12, 4)
    faces = nn[:, 1:] / np.linalg.norm(nn[:, 1:], axis=1, keepdims=True)

    # 20 dodecahedron vertices (triples of mutually adjacent faces, dot = 1/sqrt(5))
    v_map = {}
    e_map = {}
    cos_adj = 1.0 / math.sqrt(5.0)
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(np.dot(faces[i], faces[j]) - cos_adj) < 1e-9:
                em = faces[i] + faces[j]
                em /= np.linalg.norm(em)
                e_map[tuple(np.round(em, 8))] = em
                for k in range(j + 1, 12):
                    if (abs(np.dot(faces[i], faces[k]) - cos_adj) < 1e-9
                            and abs(np.dot(faces[j], faces[k]) - cos_adj) < 1e-9):
                        vm = faces[i] + faces[j] + faces[k]
                        vm /= np.linalg.norm(vm)
                        v_map[tuple(np.round(vm, 8))] = vm
    verts = np.array(list(v_map.values()), dtype=np.float64)
    edges = np.array(list(e_map.values()), dtype=np.float64)
    assert verts.shape == (20, 3)
    assert edges.shape == (30, 3)

    return Q_istar, R_ico, faces, verts, edges


def eval_Y6_jet(n, faces, yv_norm):
    """Exact Y_6^I(n), tangential gradient D_A Y_6, covariant Hessian D_A D_B Y_6,
    and traceless tensor eigenvalue ||T_6||/sqrt(2) on unit S^2 (in ambient R^3 tangent coords)."""
    n = np.atleast_2d(np.asarray(n, dtype=np.float64))
    x = n @ faces.T  # (N, 12)
    p0 = (231.0 * x**6 - 315.0 * x**4 + 105.0 * x**2 - 5.0) / 16.0
    p1 = (1386.0 * x**5 - 1260.0 * x**3 + 210.0 * x) / 16.0
    p2 = (6930.0 * x**4 - 3780.0 * x**2 + 210.0) / 16.0

    Y = np.sum(p0, axis=-1) / yv_norm
    F = (p1 @ faces) / yv_norm  # (N, 3)
    nF = np.sum(n * F, axis=-1)
    grad_t = F - nF[:, None] * n

    Hamb = np.einsum("nk,ki,kj->nij", p2, faces, faces) / yv_norm
    P = np.eye(3)[None, :, :] - n[:, :, None] * n[:, None, :]
    Hess_t = np.einsum("nia,nab,njb->nij", P, Hamb, P) - nF[:, None, None] * P
    T_mat = Hess_t + 21.0 * Y[:, None, None] * P  # traceless symmetric 2-tensor on T_n S^2
    ev_T = np.sqrt(np.maximum(0.0, 0.5 * np.sum(T_mat * T_mat, axis=(1, 2))))
    lap_err = float(np.max(np.abs(np.trace(Hess_t, axis1=1, axis2=2) + 42.0 * Y)))
    return Y, grad_t, Hess_t, T_mat, ev_T, lap_err


def chi_dirichlet(n, faces):
    """Exact Dirichlet dodecahedron boundary radius chi_D(n) on unit S^3/I*."""
    n = np.atleast_2d(np.asarray(n, dtype=np.float64))
    max_dot = np.max(n @ faces.T, axis=-1)
    return np.arctan(math.tan(CHI_INJ) / max_dot)


def verify_geometry_and_Y6(R_ico, faces, verts, edges):
    """Verify exact I-invariance of Y_6^I, Dirichlet dodecahedron area/volume, and max inscribed l=6 shell."""
    yv_raw = float(np.sum(eval_legendre(6, verts[0] @ faces.T)))
    yf_raw = float(np.sum(eval_legendre(6, faces[0] @ faces.T)))
    ye_raw = float(np.sum(eval_legendre(6, edges[0] @ faces.T)))

    # 1. Exact I-invariance across all 60 rotations on 50 random unit vectors
    rng = np.random.default_rng(20261124)
    pts = rng.normal(size=(50, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    y_base = eval_Y6_jet(pts, faces, yv_raw)[0]
    max_ico_err = 0.0
    for R in R_ico:
        y_rot = eval_Y6_jet(pts @ R.T, faces, yv_raw)[0]
        max_ico_err = max(max_ico_err, float(np.max(np.abs(y_rot - y_base))))
    assert max_ico_err < 1e-13, f"I-invariance failed: {max_ico_err}"

    # 2. Evaluate on 64x128 Gauss-Legendre x Fourier grid + all 62 special symmetry points
    n_th, n_ph = 64, 128
    mu_gl, w_gl = leggauss(n_th)
    ph_gl = np.linspace(0.0, 2.0 * math.pi, n_ph, endpoint=False)
    MU, PH = np.meshgrid(mu_gl, ph_gl, indexing="ij")
    ST = np.sqrt(1.0 - MU * MU)
    N_grid = np.stack([ST * np.cos(PH), ST * np.sin(PH), MU], axis=-1).reshape(-1, 3)
    W_dOm = np.broadcast_to(w_gl[:, None] * (2.0 * math.pi / n_ph), MU.shape).reshape(-1)
    N_all = np.vstack([N_grid, faces, verts, edges])

    Y_all, gY_all, _, _, evT_all, lap_err = eval_Y6_jet(N_all, faces, yv_raw)
    Y_grid = Y_all[: len(N_grid)]
    mean_Y6 = float(np.sum(W_dOm * Y_grid) / (4.0 * math.pi))
    rms_Y6 = math.sqrt(float(np.sum(W_dOm * Y_grid * Y_grid) / (4.0 * math.pi)))
    assert abs(mean_Y6) < 1e-14

    T_edge_exact = 189.0 * math.sqrt(5.0) / 16.0
    evT_face = float(np.max(evT_all[len(N_grid) : len(N_grid) + 12]))
    evT_vert = float(np.max(evT_all[len(N_grid) + 12 : len(N_grid) + 32]))
    evT_edge = float(np.max(evT_all[len(N_grid) + 32 :]))
    evT_global = float(np.max(evT_all))
    assert abs(evT_edge - T_edge_exact) < 1e-12
    assert abs(evT_global - T_edge_exact) < 1e-12

    # 3. Exact Dirichlet dodecahedron geometry via 120 congruent Schwarz right triangles
    F0 = faces[0]
    cos_adj = 1.0 / math.sqrt(5.0)
    adj = [f for f in faces if abs(np.dot(f, F0) - cos_adj) < 1e-9]
    F1 = adj[0]
    F2 = [f for f in adj if abs(np.dot(f, F1) - cos_adj) < 1e-9][0]
    E0 = F0 + F1; E0 /= np.linalg.norm(E0)
    V0 = F0 + F1 + F2; V0 /= np.linalg.norm(V0)

    def q_on_face(nv):
        c = math.atan(math.tan(CHI_INJ) / float(np.dot(nv, F0)))
        return np.array([math.cos(c), *(math.sin(c) * nv)])

    qF, qE, qV = q_on_face(F0), q_on_face(E0), q_on_face(V0)
    sFE = math.acos(float(np.dot(qF, qE)))
    sEV = math.acos(float(np.dot(qE, qV)))
    sFV = math.acos(float(np.dot(qF, qV)))
    ang_E = math.acos((math.cos(sFV) - math.cos(sFE) * math.cos(sEV)) / (math.sin(sFE) * math.sin(sEV)))
    ang_F = math.acos((math.cos(sEV) - math.cos(sFE) * math.cos(sFV)) / (math.sin(sFE) * math.sin(sFV)))
    ang_V = math.acos((math.cos(sFE) - math.cos(sEV) * math.cos(sFV)) / (math.sin(sEV) * math.sin(sFV)))
    area_dodeca = 120.0 * (ang_E + ang_F + ang_V - math.pi)
    R_area_dodeca = math.sqrt(area_dodeca / (4.0 * math.pi))

    chi_eq = float(brentq(lambda c: 2.0 * c - math.sin(2.0 * c) - math.pi / 60.0, 0.1, 0.6))
    chi_page7 = math.asin(PAGE7_CEILING ** (1.0 / 3.0))

    # 4. Exact continuous 2D optimization of chi0_max(eps) on one smooth Schwarz right triangle (F0, E0, V0)
    #    (closes P2-3: eliminates 64x128 coarse-grid bias AND kink quadrature error!)
    #    Wait: inside triangle (F0, E0, V0), nearest face is F0, so chi_D(n) = arctan(tan(pi/10)/(n . F0)) is C^inf smooth!
    def exact_chi0_max_continuous(eps):
        # Coarse warm start on a dense barycentric grid inside the triangle (F0, E0, V0)
        u_g = np.linspace(0.0, 1.0, 61)
        pts_tri = []
        uv_tri = []
        for ui in u_g:
            for vi in u_g:
                if ui + vi <= 1.0 + 1e-12:
                    w = (1.0 - ui - vi) * F0 + ui * E0 + vi * V0
                    pts_tri.append(w / np.linalg.norm(w))
                    uv_tri.append((ui, vi))
        pts_tri = np.array(pts_tri)
        vals = np.arctan(math.tan(CHI_INJ) / (pts_tri @ F0)) - eps * eval_Y6_jet(pts_tri, faces, yv_raw)[0]
        i_min = int(np.argmin(vals))
        u0, v0 = uv_tri[i_min]

        def obj(uv):
            u, v = uv[0], uv[1]
            if u < 0.0 or v < 0.0 or u + v > 1.0:
                return 1e6 + 1e4 * (max(0.0, -u) + max(0.0, -v) + max(0.0, u + v - 1.0))
            w = (1.0 - u - v) * F0 + u * E0 + v * V0
            nv = w / np.linalg.norm(w)
            cD_v = math.atan(math.tan(CHI_INJ) / float(np.dot(nv, F0)))
            y6_v = float(np.sum(eval_legendre(6, nv @ faces.T)) / yv_raw)
            return cD_v - eps * y6_v

        res = minimize(obj, [u0, v0], method="Nelder-Mead", options={"xatol": 1e-13, "fatol": 1e-14, "maxiter": 2000})
        return float(min(vals[i_min], res.fun))

    # Exact smooth Gauss-Legendre volume integral on the single Schwarz right triangle (120 copies = full S^2!)
    # Parameterize triangle (F0, E0, V0) in spherical coords centered at F0:
    #   azimuth phi_f in [0, pi/5] (between edge midpoint E0 at phi_f=0 and vertex V0 at phi_f=pi/5),
    #   polar angle theta_f in [0, theta_max(phi_f)] where tan(theta_max) * cos(phi_f) = tan(theta_FE)!
    theta_FE = math.acos(float(np.dot(F0, E0)))
    gl_u, gl_wu = leggauss(32)
    gl_v, gl_wv = leggauss(32)
    vol_tri = 0.0
    mean_chi_tri = 0.0
    area_s2_tri = 0.0
    for i in range(32):
        phi_f = 0.5 * (math.pi / 5.0) * (gl_u[i] + 1.0)
        w_phi = 0.5 * (math.pi / 5.0) * gl_wu[i]
        th_max = math.atan(math.tan(theta_FE) / math.cos(phi_f))
        for j in range(32):
            th_f = 0.5 * th_max * (gl_v[j] + 1.0)
            w_th = 0.5 * th_max * gl_wv[j]
            # On this face, n . F0 = cos(th_f) => chi_D = arctan(tan(pi/10) / cos(th_f))!
            c_loc = math.atan(math.tan(CHI_INJ) / math.cos(th_f))
            dOm = math.sin(th_f) * w_th * w_phi
            area_s2_tri += dOm
            mean_chi_tri += c_loc * dOm
            vol_tri += (2.0 * c_loc - math.sin(2.0 * c_loc)) * dOm
    # 120 triangles cover S^2 (area = 4*pi => each triangle has solid angle pi/30)!
    mean_chi_D = float(mean_chi_tri / area_s2_tri)
    vol_check = float((60.0 / math.pi) * (vol_tri / area_s2_tri))
    assert abs(120.0 * area_s2_tri - 4.0 * math.pi) < 1e-13
    assert abs(vol_check - 1.0) < 1e-13, f"Smooth Schwarz triangle volume check failed: {vol_check}"

    res_opt = minimize_scalar(
        lambda eps: -exact_chi0_max_continuous(eps),
        bounds=(0.0, 0.04),
        method="bounded",
        options={"xatol": 1e-10},
    )
    eps_chi0_max = float(res_opt.x)
    chi0_max_l6 = float(-res_opt.fun)

    return {
        "order_I_rotations": 60,
        "max_I_invariance_err_Y6": max_ico_err,
        "max_laplacian_err_Y6": lap_err,
        "mean_Y6": mean_Y6,
        "rms_Y6": rms_Y6,
        "Y6_raw_values": {"face": yf_raw, "vertex": yv_raw, "edge": ye_raw},
        "Y6_norm_values": {"vertex": 1.0, "edge": ye_raw / yv_raw, "face": yf_raw / yv_raw},
        "T6_norm_over_sqrt2": {
            "face": evT_face, "vertex": evT_vert, "edge": evT_edge,
            "global_max": evT_global, "exact_189sqrt5_over_16": T_edge_exact,
        },
        "max_grad_Y6": float(np.max(np.linalg.norm(gY_all, axis=-1))),
        "dirichlet_dodecahedron": {
            "chi_inj_deg": math.degrees(CHI_INJ),
            "chi_edge_deg": math.degrees(float(chi_dirichlet(edges[0], faces)[0])),
            "chi_circ_deg": math.degrees(float(chi_dirichlet(verts[0], faces)[0])),
            "mean_chi_D_deg": math.degrees(mean_chi_D),
            "sin3_mean_chi_D": math.sin(mean_chi_D) ** 3,
            "chi_eq_100pct_vol_deg": math.degrees(chi_eq),
            "sin3_chi_eq": math.sin(chi_eq) ** 3,
            "chi_page7_deg": math.degrees(chi_page7),
            "page7_pi_over_80": PAGE7_CEILING,
            "exact_boundary_area": area_dodeca,
            "R_area_dodeca": R_area_dodeca,
            "R_area_dodeca_cubed": R_area_dodeca ** 3,
            "vol_quadrature_check": vol_check,
        },
        "max_inscribed_l6_shell": {
            "eps_opt_rad": eps_chi0_max,
            "eps_opt_deg": math.degrees(eps_chi0_max),
            "chi0_max_rad": chi0_max_l6,
            "chi0_max_deg": math.degrees(chi0_max_l6),
            "sin3_chi0_max": math.sin(chi0_max_l6) ** 3,
            "deficit_below_page7_pct": 100.0 * (1.0 - (math.sin(chi0_max_l6) ** 3) / PAGE7_CEILING),
        },
    }, yv_raw, N_grid, W_dOm, N_all, exact_chi0_max_continuous


# =============================================================================
# 2. INDEPENDENT 35-DIGIT MPMATH 3D & 4D DIFFERENTIAL-GEOMETRY ORACLE
# =============================================================================

def verify_diffgeom_oracle_mpmath():
    """Verify ALL differential-geometry formulas directly from 3D/4D Christoffel symbols
    in 35-digit mpmath:
      (a) Exact nonlinear S^3 extrinsic curvature K^A_B^- for finite eps,
      (b) Exact nonlinear Brill-Lindquist R^(3)+ = 0 and K^A_B^+ for finite eps,
      (c) Exact 4D static Regge-Wheeler l=6 vacuum metric (Q_6^2 and P_6^2):
          all 10 linearized 4D Einstein equations delta R_{mu nu} = 0 AND
          exact O(eps) formulas for delta(K^A_B^+) and delta(K^tau_tau^+),
      (d) Exact universal identity R_0*(8*pi*G*delta_sigma)/(eps*Y_l) = +3*sin(chi_0)
          at x = x_iso, u_0 = cos(chi_0) across l in {2, 6, 10, 12}.
    """
    mp.mp.dps = 35
    l = 6
    h = mp.mpf("1e-7")

    def P_l(z):
        return mp.legendre(l, z)

    def Q_l(z):
        q0 = mp.mpf("0.5") * mp.log((z + 1) / (z - 1))
        q1 = z * q0 - 1
        qa, qb = q0, q1
        for n in range(1, l):
            qa, qb = qb, ((2 * n + 1) * z * qb - n * qa) / (n + 1)
        return qb

    def d4_central(fn, pt, dim=3, step=h):
        out = []
        for a in range(dim):
            p1, m1, p2, m2 = list(pt), list(pt), list(pt), list(pt)
            p1[a] += step; m1[a] -= step; p2[a] += 2 * step; m2[a] -= 2 * step
            out.append((-fn(p2) + 8 * fn(p1) - 8 * fn(m1) + fn(m2)) / (12 * step))
        return out

    def christoffel_nd(g_fn, pt, dim=3, step=h):
        g = g_fn(pt)
        ginv = g ** (-1)
        dg = d4_central(g_fn, pt, dim=dim, step=step)
        Gam = [[[mp.mpf(0) for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
        for r in range(dim):
            for m in range(dim):
                for n in range(dim):
                    Gam[r][m][n] = mp.mpf("0.5") * sum(
                        ginv[r, s] * (dg[m][n, s] + dg[n][m, s] - dg[s][m, n]) for s in range(dim)
                    )
        return g, ginv, Gam

    def ricci_nd(g_fn, pt, dim=3, step=h):
        g, ginv, Gam = christoffel_nd(g_fn, pt, dim=dim, step=step)
        dGam = []
        for a in range(dim):
            p1, m1, p2, m2 = list(pt), list(pt), list(pt), list(pt)
            p1[a] += step; m1[a] -= step; p2[a] += 2 * step; m2[a] -= 2 * step
            _, _, Gp1 = christoffel_nd(g_fn, p1, dim=dim, step=step)
            _, _, Gm1 = christoffel_nd(g_fn, m1, dim=dim, step=step)
            _, _, Gp2 = christoffel_nd(g_fn, p2, dim=dim, step=step)
            _, _, Gm2 = christoffel_nd(g_fn, m2, dim=dim, step=step)
            dGam.append([
                [[(-Gp2[r][m][n] + 8 * Gp1[r][m][n] - 8 * Gm1[r][m][n] + Gm2[r][m][n]) / (12 * step)
                  for n in range(dim)] for m in range(dim)]
                for r in range(dim)
            ])
        Ric = mp.zeros(dim, dim)
        for m in range(dim):
            for n in range(dim):
                s = mp.mpf(0)
                for r in range(dim):
                    s += dGam[r][r][m][n] - dGam[n][r][m][r]
                    for k in range(dim):
                        s += Gam[r][r][k] * Gam[k][m][n] - Gam[r][n][k] * Gam[k][m][r]
                Ric[m, n] = s
        R_sc = sum(ginv[m, n] * Ric[m, n] for m in range(dim) for n in range(dim))
        return g, ginv, Gam, Ric, R_sc

    # --- (a) Exact nonlinear S^3 K^A_B^- check (axisymmetric Y = P_6(cos th), finite eps = 0.015) ---
    a0 = mp.mpf("1.0")
    chi0 = mp.mpf("0.33")
    eps_nl = mp.mpf("0.015")
    th0 = mp.mpf("0.85")
    ph0 = mp.mpf("0.4")

    def chi_s(th):
        return chi0 + eps_nl * P_l(mp.cos(th))

    def g_s3(pt):
        c, th, _ph = pt
        return mp.diag([a0**2, a0**2 * mp.sin(c)**2, a0**2 * mp.sin(c)**2 * mp.sin(th)**2])

    c_at = chi_s(th0)
    dc = mp.diff(chi_s, th0)
    d2c = mp.diff(chi_s, th0, 2)
    pt3 = [c_at, th0, ph0]
    g3, g3inv, Gam3 = christoffel_nd(g_s3, pt3, dim=3)
    # Unit normal n_i = s_keep * a0 / W * (1, -dc, 0) with s_keep = -1
    W = mp.sqrt(1 + dc**2 / mp.sin(c_at)**2)
    n_cov = [-a0 / W, a0 * dc / W, mp.mpf(0)]
    # Tangent vectors e_1 = (dc, 1, 0), e_2 = (0, 0, 1)
    e_vec = [[dc, mp.mpf(1), mp.mpf(0)], [mp.mpf(0), mp.mpf(0), mp.mpf(1)]]
    # Induced 2-metric q_AB = e_A^i e_B^j g_{ij}
    q_mat = mp.matrix([[sum(e_vec[A][i] * e_vec[B][j] * g3[i, j] for i in range(3) for j in range(3))
                        for B in range(2)] for A in range(2)])
    # Direct K_AB = - n_i * (d_A e_B^i + Gam^i_{jk} e_A^j e_B^k)
    de_dth = [[d2c, mp.mpf(0), mp.mpf(0)], [mp.mpf(0), mp.mpf(0), mp.mpf(0)]]
    K_dir = mp.zeros(2, 2)
    for A in range(2):
        for B in range(2):
            acc = [mp.mpf(0)] * 3
            for i in range(3):
                if A == 0:
                    acc[i] += de_dth[B][i]
                for j in range(3):
                    for k in range(3):
                        acc[i] += Gam3[i][j][k] * e_vec[A][j] * e_vec[B][k]
            K_dir[A, B] = -sum(n_cov[i] * acc[i] for i in range(3))
    K_mix_dir = (q_mat ** (-1)) * K_dir

    # Analytical formula: K_AB^- = (-a0/W) * [sin c cos c gamma_AB + 2 cot c D_A c D_B c - D_A D_B c]
    # On S^2 (th, ph): D_th D_th c = d2c, D_ph D_ph c = sin(th)cos(th) dc!
    K_ana = mp.diag([
        (-a0 / W) * (mp.sin(c_at) * mp.cos(c_at) + 2 * (mp.cos(c_at) / mp.sin(c_at)) * dc**2 - d2c),
        (-a0 / W) * (mp.sin(c_at) * mp.cos(c_at) * mp.sin(th0)**2 - mp.sin(th0) * mp.cos(th0) * dc),
    ])
    K_mix_ana = (q_mat ** (-1)) * K_ana
    err_s3_K = max(abs(K_mix_dir[A, B] - K_mix_ana[A, B]) for A in range(2) for B in range(2))

    # --- (b) Exact nonlinear Brill-Lindquist R^(3)+ = 0 and K^A_B^+ ---
    M_bl = mp.mpf("0.012")
    rho0 = mp.mpf("0.30")
    c_bl = mp.mpf("0.02")

    def Psi_bl(rho, th):
        return 1 + M_bl / (2 * rho) + c_bl * ((rho / rho0) ** l) * P_l(mp.cos(th))

    def g_bl(pt):
        rho, th, _ph = pt
        p4 = Psi_bl(rho, th) ** 4
        return mp.diag([p4, p4 * rho**2, p4 * rho**2 * mp.sin(th)**2])

    _, _, Gam_bl, _, R3_bl = ricci_nd(g_bl, [rho0, th0, ph0], dim=3)
    # Check exact conformal formula K^A_B^+ = Psi^-2 (bar_K^A_B + (2/Psi)(bar_n^i d_i Psi) delta^A_B)
    # on a finite-amplitude surface rho(th) = rho0 + 0.01 * P_6(cos th)
    def rho_s(th):
        return rho0 + mp.mpf("0.01") * P_l(mp.cos(th))

    r_at = rho_s(th0)
    dr = mp.diff(rho_s, th0)
    d2r = mp.diff(rho_s, th0, 2)
    pt_bl = [r_at, th0, ph0]
    g_b, _, Gam_b = christoffel_nd(g_bl, pt_bl, dim=3)
    Psi_at = Psi_bl(r_at, th0)
    dPsi_r = mp.diff(lambda rr: Psi_bl(rr, th0), r_at)
    dPsi_t = mp.diff(lambda tt: Psi_bl(r_at, tt), th0)
    W_b = mp.sqrt(1 + (dr / r_at) ** 2)
    # Outward normal s_+ = -1: n_i = (-Psi^2 / W_b) * (1, -dr, 0)
    n_bl_cov = [-(Psi_at**2) / W_b, (Psi_at**2) * dr / W_b, mp.mpf(0)]
    eb = [[dr, mp.mpf(1), mp.mpf(0)], [mp.mpf(0), mp.mpf(0), mp.mpf(1)]]
    qb = mp.matrix([[sum(eb[A][i] * eb[B][j] * g_b[i, j] for i in range(3) for j in range(3))
                     for B in range(2)] for A in range(2)])
    deb = [[d2r, mp.mpf(0), mp.mpf(0)], [mp.mpf(0), mp.mpf(0), mp.mpf(0)]]
    Kb_dir = mp.zeros(2, 2)
    for A in range(2):
        for B in range(2):
            acc = [mp.mpf(0)] * 3
            for i in range(3):
                if A == 0:
                    acc[i] += deb[B][i]
                for j in range(3):
                    for k in range(3):
                        acc[i] += Gam_b[i][j][k] * eb[A][j] * eb[B][k]
            Kb_dir[A, B] = -sum(n_bl_cov[i] * acc[i] for i in range(3))
    Kb_mix_dir = (qb ** (-1)) * Kb_dir

    # Conformal analytical formula:
    # bar_q_AB = diag(r^2 + dr^2, r^2 sin^2 th)
    # bar_K_AB = (-1/W_b) * [r gamma_AB + (2/r) d_A r d_B r - D_A D_B r]
    bar_q = mp.diag([r_at**2 + dr**2, r_at**2 * mp.sin(th0)**2])
    bar_K = mp.diag([
        (-1 / W_b) * (r_at + (2 / r_at) * dr**2 - d2r),
        (-1 / W_b) * (r_at * mp.sin(th0)**2 - mp.sin(th0) * mp.cos(th0) * dr),
    ])
    bar_K_mix = (bar_q ** (-1)) * bar_K
    bar_n_dPsi = (-1 / W_b) * (dPsi_r - (dr / (r_at**2)) * dPsi_t)
    Kb_mix_ana = (Psi_at ** (-2)) * (bar_K_mix + (2 / Psi_at) * bar_n_dPsi * mp.eye(2))
    err_bl_K = max(abs(Kb_mix_dir[A, B] - Kb_mix_ana[A, B]) for A in range(2) for B in range(2))

    # --- (c) Exact 4D Static Regge-Wheeler l=6 Vacuum Metric (Q_6^2 and P_6^2) ---
    rw_checks = {}
    R0 = mp.mpf("0.32")
    mu_rw = 2 * M_bl
    eps_lin = mp.mpf("1e-10")  # O(eps^2) ~ 1e-20 << O(eps) ~ 1e-10

    for mode in ("Q", "P"):
        fn = Q_l if mode == "Q" else P_l
        # Normalize H(R0) = 1 for clean conditioning!
        x_R0 = R0 / M_bl - 1
        h_raw_fn = lambda xx: (xx**2 - 1) * mp.diff(fn, xx, 2)
        norm_H = h_raw_fn(x_R0)

        def HK_mode(r):
            x = r / M_bl - 1
            H = h_raw_fn(x) / norm_H
            dH_dr = mp.diff(h_raw_fn, x) / (M_bl * norm_H)
            K = (2 * M_bl * dH_dr + (l * (l + 1) - 2 + 4 * M_bl * (r - M_bl) / (r * (r - 2 * M_bl))) * H) / ((l - 1) * (l + 2))
            dK_dr = dH_dr + (2 * M_bl / (r * (r - 2 * M_bl))) * H
            return H, dH_dr, K, dK_dr

        def g_rw(pt, eps_val):
            _t, r, th, _ph = pt
            f = 1 - mu_rw / r
            H, _, K, _ = HK_mode(r)
            Y = P_l(mp.cos(th))
            # Exact Regge-Wheeler static even-parity sign: g_tt = -f*(1 - eps*H*Y), g_rr = f^-1*(1 + eps*H*Y)!
            return mp.diag([
                -f * (1 - eps_val * H * Y),
                (1 / f) * (1 + eps_val * H * Y),
                (r**2) * (1 + eps_val * K * Y),
                (r**2) * (mp.sin(th) ** 2) * (1 + eps_val * K * Y),
            ])

        pt4 = [mp.mpf(0), R0, th0, ph0]
        # Linearized 4D Ricci tensor: (Ric(+eps) - Ric(-eps)) / (2*eps) must vanish!
        _, _, _, Ric_p, _ = ricci_nd(lambda p: g_rw(p, +eps_lin), pt4, dim=4, step=mp.mpf("1e-5"))
        _, _, _, Ric_m, _ = ricci_nd(lambda p: g_rw(p, -eps_lin), pt4, dim=4, step=mp.mpf("1e-5"))
        dRic = (Ric_p - Ric_m) / (2 * eps_lin)
        max_dRic = max(abs(dRic[a, b]) for a in range(4) for b in range(4))

        # Also verify O(eps) formulas for delta(K^A_A^+) and delta(K^tau_tau^+) at r(th) = R0 + eps * r1 * P_6(cos th)!
        r1 = mp.mpf("0.17")

        def eval_K_4d(eps_val):
            r_fn = lambda tt: R0 + eps_val * r1 * P_l(mp.cos(tt))
            r_v = r_fn(th0)
            dr_v = mp.diff(r_fn, th0)
            d2r_v = mp.diff(r_fn, th0, 2)
            p_s = [mp.mpf(0), r_v, th0, ph0]
            g4, g4inv, Gam4 = christoffel_nd(lambda p: g_rw(p, eps_val), p_s, dim=4, step=mp.mpf("1e-5"))
            # Normal covector proportional to (0, -1, +dr_v, 0) (s_+ = -1)
            raw_n = [mp.mpf(0), mp.mpf(-1), dr_v, mp.mpf(0)]
            norm_n = mp.sqrt(sum(g4inv[i, j] * raw_n[i] * raw_n[j] for i in range(4) for j in range(4)))
            n4 = [v / norm_n for v in raw_n]
            # Tangents: e_tau = (1/sqrt(-g_00), 0, 0, 0), e_th = (0, dr_v, 1, 0), e_ph = (0, 0, 0, 1)
            u0 = 1 / mp.sqrt(-g4[0, 0])
            # K^tau_tau = -K_{tau tau} = +n_i * (Gam^i_{00} * u0^2)
            Ktt = sum(n4[i] * Gam4[i][0][0] * (u0**2) for i in range(4))
            # Spatial K^A_A:
            et = [[mp.mpf(0), dr_v, mp.mpf(1), mp.mpf(0)], [mp.mpf(0), mp.mpf(0), mp.mpf(0), mp.mpf(1)]]
            det_t = [[mp.mpf(0), d2r_v, mp.mpf(0), mp.mpf(0)], [mp.mpf(0)] * 4]
            q2 = mp.matrix([[sum(et[A][i] * et[B][j] * g4[i, j] for i in range(4) for j in range(4))
                             for B in range(2)] for A in range(2)])
            K2 = mp.zeros(2, 2)
            for A in range(2):
                for B in range(2):
                    acc = [mp.mpf(0)] * 4
                    for i in range(4):
                        if A == 0:
                            acc[i] += det_t[B][i]
                        for j in range(4):
                            for k in range(4):
                                acc[i] += Gam4[i][j][k] * et[A][j] * et[B][k]
                    K2[A, B] = -sum(n4[i] * acc[i] for i in range(4))
            K2m = (q2 ** (-1)) * K2
            return K2m[0, 0] + K2m[1, 1], K2m[0, 0] - K2m[1, 1], Ktt

        Kp_tr, Kp_sh, Kp_tt = eval_K_4d(+eps_lin)
        Km_tr, Km_sh, Km_tt = eval_K_4d(-eps_lin)
        dKtr_num = (Kp_tr - Km_tr) / (2 * eps_lin)
        dKsh_num = (Kp_sh - Km_sh) / (2 * eps_lin)
        dKtt_num = (Kp_tt - Km_tt) / (2 * eps_lin)

        # Analytical formulas (s_+ = -1):
        H0, dH0, K0, dK0 = HK_mode(R0)
        f0 = 1 - mu_rw / R0
        u0_rw = mp.sqrt(f0)
        Y0 = P_l(mp.cos(th0))
        dY_th = mp.diff(lambda tt: P_l(mp.cos(tt)), th0)
        d2Y_th = mp.diff(lambda tt: P_l(mp.cos(tt)), th0, 2)
        T_diff = d2Y_th - (mp.cos(th0) / mp.sin(th0)) * dY_th

        dKtr_ana = (-1 / R0) * ((r1 / (R0 * u0_rw)) * (l * (l + 1) - 2 + 3 * mu_rw / R0) + u0_rw * (R0 * dK0 - H0)) * Y0
        dKsh_ana = (r1 / (R0**2 * u0_rw)) * T_diff
        df_dr_sqrtf = (mu_rw / (R0**3 * u0_rw)) * (-2 - mu_rw / (2 * R0 * f0))
        fp0 = mu_rw / (R0**2)
        # With g_tt = -f*(1 - eps*H*Y): d/dr ln(-g_tt) = f'/f - eps*H'*Y, so -u0_rw*dH0!
        dKtt_ana = mp.mpf("-0.5") * (df_dr_sqrtf * r1 - u0_rw * dH0 - (fp0 / (2 * u0_rw)) * H0) * Y0

        rw_checks[mode] = {
            "max_linearized_4D_Ricci_err": float(max_dRic),
            "dK_trace_4D_err": float(abs(dKtr_num - dKtr_ana)),
            "dK_shear_4D_err": float(abs(dKsh_num - dKsh_ana)),
            "dK_tautau_4D_err": float(abs(dKtt_num - dKtt_ana)),
            "H0_norm": float(H0),
            "K0_over_H0": float(K0 / H0),
            "R0_dK_over_H0": float(R0 * dK0 / H0),
        }

    # --- (c2) Exact nonlinear pure-Schwarzschild H_+ 3D Christoffel oracle check ---
    def g_schw3(pt):
        r, th, _ph = pt
        f = 1 - mu_rw / r
        return mp.diag([1 / f, r**2, r**2 * mp.sin(th)**2])

    r_s_fn = lambda tt: mp.sin(chi_s(tt))
    rs_at = r_s_fn(th0)
    drs = mp.diff(r_s_fn, th0)
    d2rs = mp.diff(r_s_fn, th0, 2)
    gs, gsinv, Gams = christoffel_nd(g_schw3, [rs_at, th0, ph0], dim=3)
    fs = 1 - mu_rw / rs_at
    Ws = mp.sqrt(1 + (drs**2) / (fs * rs_at**2))
    ns_cov = [-1 / (mp.sqrt(fs) * Ws), drs / (mp.sqrt(fs) * Ws), mp.mpf(0)]
    es = [[drs, mp.mpf(1), mp.mpf(0)], [mp.mpf(0), mp.mpf(0), mp.mpf(1)]]
    des = [[d2rs, mp.mpf(0), mp.mpf(0)], [mp.mpf(0)] * 3]
    qs = mp.matrix([[sum(es[A][i] * es[B][j] * gs[i, j] for i in range(3) for j in range(3)) for B in range(2)] for A in range(2)])
    Ks = mp.zeros(2, 2)
    for A in range(2):
        for B in range(2):
            acc = [mp.mpf(0)] * 3
            for i in range(3):
                if A == 0:
                    acc[i] += des[B][i]
                for j in range(3):
                    for k in range(3):
                        acc[i] += Gams[i][j][k] * es[A][j] * es[B][k]
            Ks[A, B] = -sum(ns_cov[i] * acc[i] for i in range(3))
    Ks_mix = (qs ** (-1)) * Ks
    Hp_dir = Ks_mix[0, 0] + Ks_mix[1, 1]
    # Formula check:
    fps = mu_rw / (rs_at**2)
    c_dr = 2 / rs_at + mp.mpf("0.5") * fps / fs
    Lap_r = d2rs + (mp.cos(th0) / mp.sin(th0)) * drs
    dr_Dr_dr = (drs**2) * d2rs
    tr1 = Lap_r - c_dr * (drs**2) - 2 * rs_at * fs
    tr2 = (dr_Dr_dr - c_dr * (drs**4) - rs_at * fs * (drs**2)) / (fs * rs_at**2 * Ws**2)
    Hp_ana = (1 / (rs_at**2 * mp.sqrt(fs) * Ws)) * (tr1 - tr2)
    err_schw_Hp = abs(Hp_dir - Hp_ana)

    # --- (d) Exact universal identity check across l in {2, 6, 10, 12} ---
    univ_max_err = mp.mpf(0)
    for l_t in (2, 6, 10, 12):
        for k_t in (l_t, -(l_t + 1)):
            c_t = mp.pi / 10
            sc, cc = mp.sin(c_t), mp.cos(c_t)
            u_t = cc
            Ax = l_t * (l_t + 1) + 1 - (2 * k_t + 1) * u_t
            B0 = -(l_t * (l_t + 1) - 2) / sc + (2 * k_t + 1 - 3 * u_t) * (cc / sc)
            val = Ax * (1 / sc) + B0
            univ_max_err = max(univ_max_err, abs(val - 3 * sc))

    return {
        "nonlinear_S3_K_mix_3D_oracle_err": float(err_s3_K),
        "nonlinear_BrillLindquist_R3_vacuum_err": float(abs(R3_bl)),
        "nonlinear_BrillLindquist_K_mix_3D_oracle_err": float(err_bl_K),
        "nonlinear_pure_Schw_Hp_3D_oracle_err": float(err_schw_Hp),
        "static_4D_ReggeWheeler_l6": rw_checks,
        "universal_3sin_chi0_identity_err": float(univ_max_err),
    }


# =============================================================================
# 3. COMPLETE O(EPS) MASS CEILING ANALYSIS (WEC, WEC+DEC, STATIC WEC+NEC)
# =============================================================================

def eval_rw_ratios_mp(l_deg, mu, R0, mode="P"):
    """Compute exact static 4D Regge-Wheeler dimensionless ratios at r = R0 (in 35-digit mpmath)."""
    mp.mp.dps = 35
    M = mp.mpf(repr(0.5 * mu))
    r0 = mp.mpf(repr(R0))
    x = r0 / M - 1

    def P_l(z):
        return mp.legendre(l_deg, z)

    def Q_l(z):
        q0 = mp.mpf("0.5") * mp.log((z + 1) / (z - 1))
        q1 = z * q0 - 1
        qa, qb = q0, q1
        for n in range(1, l_deg):
            qa, qb = qb, ((2 * n + 1) * z * qb - n * qa) / (n + 1)
        return qb

    fn = Q_l if mode == "Q" else P_l
    h_fn = lambda xx: (xx**2 - 1) * mp.diff(fn, xx, 2)
    H = h_fn(x)
    dH_dr = mp.diff(h_fn, x) / M
    K = (2 * M * dH_dr + (l_deg * (l_deg + 1) - 2 + 4 * M * (r0 - M) / (r0 * (r0 - 2 * M))) * H) / ((l_deg - 1) * (l_deg + 2))
    dK_dr = dH_dr + (2 * M / (r0 * (r0 - 2 * M))) * H
    return {
        "H_over_K": float(H / K),
        "R0_dH_over_K": float(r0 * dH_dr / K),
        "R0_dK_over_K": float(r0 * dK_dr / K),
    }


def linear_exterior_coeffs(chi0, u0, model="BL_worm", l_deg=6):
    """Return exact O(eps) coefficients (A_x, B_0, C_Ktt_x, C_Ktt_0) for a given exterior model,
    parameterized by the dimensionless radial-displacement variable x = r_1 / (R_0 * u_0) (= rho_1/rho_0):
      - 8*pi*G * delta_sigma * R_0 / eps = (A_x * x + B_0) * Y_6(n)
      - ||[hat K]||/sqrt(2) * R_0 / eps = |x - 1/sin(chi0)| * ||T_6(n)||/sqrt(2)
      - delta(K^tau_tau^+) * R_0 / eps = (C_Ktt_x * x + C_Ktt_0) * Y_6(n)  (static RW models only)
    """
    sc, cc = math.sin(chi0), math.cos(chi0)
    cotc = cc / sc
    R0 = sc
    mu = R0 * (1.0 - u0 * u0)

    if model in ("BL_worm", "BL_ext"):
        k = l_deg if model == "BL_worm" else -(l_deg + 1)
        A_x = l_deg * (l_deg + 1) + 1.0 - (2 * k + 1) * u0
        B_0 = -(l_deg * (l_deg + 1) - 2.0) / sc + (2 * k + 1 - 3.0 * u0) * cotc
        return A_x, B_0, None, None

    # 4D Static Regge-Wheeler (P_6^2 regular at horizon, or Q_6^2 asymptotically flat)
    rw_mode = "P" if model == "RW_P" else "Q"
    ratios = eval_rw_ratios_mp(l_deg, max(mu, 1e-9), R0, mode=rw_mode)
    # First junction condition: r_1/R_0 + 0.5*K = cot(chi0) => K = 2*(cot(chi0) - u0*x)
    # From delta(K^A_A^+) = (-1/R_0)*[ x*(l(l+1)-2+3mu/R0) + u0*(R0*K' - H) ]
    # Write u0*(R0*K' - H) = u0 * (R0_dK_over_K - H_over_K) * K = 2*u0*(R0_dK_over_K - H_over_K)*(cotc - u0*x)
    alpha_K = 2.0 * u0 * (ratios["R0_dK_over_K"] - ratios["H_over_K"])
    A_x = (l_deg * (l_deg + 1) - 2.0 + 3.0 * mu / R0) - u0 * alpha_K
    B_0 = -(l_deg * (l_deg + 1) - 2.0) / sc + alpha_K * cotc

    # Static time-time jump (with g_tt = -f*(1 - eps*H*Y)): R0 * delta(K^tau_tau^+) / eps
    # = -0.5 * [ R0 * df_dr_sqrtf * (R0 * u0 * x) - u0 * (R0*dH/K) * K - (mu/(2 R0 u0)) * (H/K) * K ]
    # Note: R0^2 * u0 * df_dr_sqrtf = (mu/R0) * (-2 - mu/(2 R0 u0^2))
    mu_R = mu / R0
    c_r1 = mu_R * (-2.0 - 0.5 * mu_R / (u0 * u0))
    c_K = -u0 * ratios["R0_dH_over_K"] - (0.5 * mu_R / u0) * ratios["H_over_K"]
    # Multiply by -0.5 and substitute K = 2*(cotc - u0*x):
    C_Ktt_x = -0.5 * (c_r1 - 2.0 * u0 * c_K)
    C_Ktt_0 = -c_K * cotc
    return A_x, B_0, C_Ktt_x, C_Ktt_0


def analyze_linearized_ceilings(N_all, faces, yv_raw, exact_chi0_fn, eps_opt_rad):
    """Optimize O(eps) WEC, WEC+DEC (ddot{chi}-independent), and Static WEC+NEC ceilings
    across ALL 8,254 points of S^2 (closes P1-4, P1-3, P1-5, P2-3!)."""
    Y_all, _, _, _, evT_all, _ = eval_Y6_jet(N_all, faces, yv_raw)
    Y_face, Y_vert = -1.8, 1.0

    def intersect_linear_bounds(a_arr, b_arr):
        """Given vectors a_i, b_i, determine whether there exists dx in R such that a_i*dx + b_i >= 0 for ALL i."""
        lo = -1e18
        hi = 1e18
        pos = a_arr > 1e-15
        neg = a_arr < -1e-15
        zero = ~(pos | neg)
        if np.any(b_arr[zero] < -1e-13):
            return False
        if np.any(pos):
            lo = max(lo, float(np.max(-b_arr[pos] / a_arr[pos])))
        if np.any(neg):
            hi = min(hi, float(np.min(-b_arr[neg] / a_arr[neg])))
        return lo <= hi + 1e-13

    def check_feasible_dec(chi0, eps, u0, model):
        """Full-S^2 WEC + DEC check: 8piG*sigma(n) >= ||[hat K](n)||/sqrt(2) across ALL 8,254 points of N_all!"""
        sc, cc = math.sin(chi0), math.cos(chi0)
        sig0 = 2.0 * (u0 - cc)
        if eps == 0.0:
            return sig0 >= 0.0
        A_x, B_0, _, _ = linear_exterior_coeffs(chi0, u0, model=model)
        x_iso = 1.0 / sc
        S_iso = A_x * x_iso + B_0
        # Enforce sig0 + eps*(S_iso + A_x*dx)*Y_i >= +eps*dx*T_i  AND  >= -eps*dx*T_i  for ALL i in N_all!
        # (Note: since T_i >= 0, this automatically implies sig0 + eps*(S_iso + A_x*dx)*Y_i >= 0 everywhere too!)
        a_pos = eps * (A_x * Y_all - evT_all)
        a_neg = eps * (A_x * Y_all + evT_all)
        b_vec = sig0 + eps * S_iso * Y_all
        return intersect_linear_bounds(np.concatenate([a_pos, a_neg]), np.concatenate([b_vec, b_vec]))

    def check_feasible_static_nec_es(chi0, eps, u0, model):
        """Full-S^2 WEC + Static/Comoving NEC check at Einstein-static gamma = 1/4 (ddot{a}_0 = 0, so static == comoving EXACTLY; closes P1-4 & P1-5!)."""
        sc, cc = math.sin(chi0), math.cos(chi0)
        sig0 = 2.0 * (u0 - cc)
        Ktt0 = -0.5 * (1.0 - u0 * u0) / u0
        nec0 = Ktt0 + 0.5 * sig0
        if eps == 0.0:
            return (sig0 >= 0.0) and (nec0 >= 0.0)
        A_x, B_0, C_x, C_0 = linear_exterior_coeffs(chi0, u0, model=model)
        x_iso = 1.0 / sc
        S_iso = A_x * x_iso + B_0
        N_slope = C_x + 0.5 * A_x
        N_iso = (C_x * x_iso + C_0) + 0.5 * S_iso
        # WEC everywhere on S^2:
        a_wec = eps * A_x * Y_all
        b_wec = sig0 + eps * S_iso * Y_all
        # NEC everywhere on S^2 (both signs of |dx|*T_i):
        a_np = eps * (N_slope * Y_all - evT_all)
        a_nn = eps * (N_slope * Y_all + evT_all)
        b_nec = nec0 + eps * N_iso * Y_all
        return intersect_linear_bounds(
            np.concatenate([a_wec, a_np, a_nn]),
            np.concatenate([b_wec, b_nec, b_nec]),
        )

    def solve_max_mu(chi0, eps, feas_fn):
        sc, cc = math.sin(chi0), math.cos(chi0)
        if not feas_fn(1.0 - 1e-10):
            return 0.0
        if feas_fn(cc):
            u_min = cc
        else:
            u_min = brentq(lambda u: 1.0 if feas_fn(u) else -1.0, cc, 1.0 - 1e-10, xtol=1e-11)
        return sc * (1.0 - u_min * u_min)

    eps_grid = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, eps_opt_rad]
    rows = []
    for eps in eps_grid:
        chi0 = exact_chi0_fn(eps)
        sc, cc = math.sin(chi0), math.cos(chi0)
        eps_eff = 42.0 * eps / (sc * cc)

        # Pure Schwarzschild (c=0): x = cot(chi0)/u0; must satisfy sigma >= 0 everywhere on S^2!
        def min_sig_c0(u0):
            A_x, B_0, _, _ = linear_exterior_coeffs(chi0, u0, model="BL_worm")
            x = (cc / sc) / u0
            S = A_x * x + B_0
            return 2.0 * (u0 - cc) + eps * min(S * Y_face, S * Y_vert)

        if eps == 0.0:
            mu_c0_wec = sc ** 3
        else:
            u_scan = np.linspace(cc, 1.0 - 1e-10, 4000)
            vals = [min_sig_c0(float(u)) for u in u_scan]
            pos_idx = [i for i, v in enumerate(vals) if v >= 0.0]
            if not pos_idx:
                mu_c0_wec = 0.0
            else:
                i0 = pos_idx[0]
                u_c0 = cc if i0 == 0 else float(brentq(min_sig_c0, float(u_scan[i0 - 1]), float(u_scan[i0]), xtol=1e-12))
                mu_c0_wec = sc * (1.0 - u_c0 * u_c0)

        mu_wec_opt = sc ** 3
        mu_dec_bl_worm = solve_max_mu(chi0, eps, lambda u: check_feasible_dec(chi0, eps, u, "BL_worm"))
        mu_dec_bl_ext = solve_max_mu(chi0, eps, lambda u: check_feasible_dec(chi0, eps, u, "BL_ext"))
        mu_dec_rw_P = solve_max_mu(chi0, eps, lambda u: check_feasible_dec(chi0, eps, u, "RW_P"))
        mu_dec_rw_Q = solve_max_mu(chi0, eps, lambda u: check_feasible_dec(chi0, eps, u, "RW_Q"))
        mu_nec_es_P = solve_max_mu(chi0, eps, lambda u: check_feasible_static_nec_es(chi0, eps, u, "RW_P"))
        mu_nec_es_Q = solve_max_mu(chi0, eps, lambda u: check_feasible_static_nec_es(chi0, eps, u, "RW_Q"))
        # Pure dS (gamma=0) comoving NEC: exact 1-line zero-mean theorem!
        # Background nec0 = -(sc^2 + 0.5*(1-u0^2))/u0 + (u0-cc) < 0 for ALL u0 in [cc, 1] (max at u0=1 is 1-cc-sc^2 = (1-cc)*(1-1-cc) < 0!).
        # Since int_{S^2} Y_6 dOmega = 0 and ||T_6|| >= 0, min_{S^2}(nec0 + eps*c*Y_6 - eps*|dx|*||T_6||/sqrt(2)) <= nec0 < 0 ALWAYS!
        mu_nec_dS_exact = 0.0

        rows.append({
            "eps_rad": eps,
            "eps_eff_curvature": eps_eff,
            "in_controlled_perturbative_regime": bool(eps_eff <= 0.5),
            "chi0_deg": math.degrees(chi0),
            "mu_over_yplus_WEC_pure_Schw": mu_c0_wec,
            "mu_over_yplus_WEC_opt_multipole": mu_wec_opt,
            "mu_over_yplus_DEC_BL_wormhole": mu_dec_bl_worm,
            "mu_over_yplus_DEC_BL_exterior": mu_dec_bl_ext,
            "mu_over_yplus_DEC_RW_P62": mu_dec_rw_P,
            "mu_over_yplus_DEC_RW_Q62": mu_dec_rw_Q,
            "mu_over_yplus_StaticNEC_gamma_quarter_RW_P62": mu_nec_es_P,
            "mu_over_yplus_StaticNEC_gamma_quarter_RW_Q62": mu_nec_es_Q,
            "mu_over_yplus_ComovingNEC_pure_dS_exact_zeromean": mu_nec_dS_exact,
        })

    ctrl_rows = [r for r in rows if r["in_controlled_perturbative_regime"]]
    summary = {
        "controlled_perturbative_max_eps_eff_le_0p5": {
            "max_WEC_opt_multipole": max(r["mu_over_yplus_WEC_opt_multipole"] for r in ctrl_rows),
            "max_DEC_RW_P62": max(r["mu_over_yplus_DEC_RW_P62"] for r in ctrl_rows),
            "max_StaticNEC_ES_RW_P62": max(r["mu_over_yplus_StaticNEC_gamma_quarter_RW_P62"] for r in ctrl_rows),
        },
        "extrapolated_envelope_all_eps": {
            "max_WEC_pure_Schw": max(r["mu_over_yplus_WEC_pure_Schw"] for r in rows),
            "max_WEC_opt_multipole": max(r["mu_over_yplus_WEC_opt_multipole"] for r in rows),
            "max_DEC_BL_wormhole": max(r["mu_over_yplus_DEC_BL_wormhole"] for r in rows),
            "max_DEC_BL_exterior": max(r["mu_over_yplus_DEC_BL_exterior"] for r in rows),
            "max_DEC_RW_P62": max(r["mu_over_yplus_DEC_RW_P62"] for r in rows),
            "max_DEC_RW_Q62": max(r["mu_over_yplus_DEC_RW_Q62"] for r in rows),
            "max_StaticNEC_ES_RW_P62": max(r["mu_over_yplus_StaticNEC_gamma_quarter_RW_P62"] for r in rows),
            "max_StaticNEC_ES_RW_Q62": max(r["mu_over_yplus_StaticNEC_gamma_quarter_RW_Q62"] for r in rows),
            "max_ComovingNEC_pure_dS": 0.0,
        },
        "all_strictly_below_pi_over_80": bool(
            max(r["mu_over_yplus_WEC_opt_multipole"] for r in rows) < PAGE7_CEILING
        ),
        "full_S2_grid_points_checked": int(len(N_all)),
    }
    return {"summary": summary, "sweep": rows}


# =============================================================================
# 4. EXACT NONLINEAR FINITE-AMPLITUDE INTERIOR HAWKING MASS & JUNCTION CHECK
# =============================================================================

def analyze_nonlinear_finite_amplitude(N_grid, W_dOm, N_all, faces, yv_raw, exact_chi0_fn):
    """Compute exact nonlinear S^3 interior Hawking mass 2*m_H^-(S)/a_0 and exact nonlinear
    surface stress eigenvalues in pure Schwarzschild (c=0), honestly tracking the O(eps^2)
    first-junction residual ||q^+ - q^-|| / ||q^-||."""
    Y_all, gY_all, HY_all, _, _, _ = eval_Y6_jet(N_all, faces, yv_raw)
    nG = len(N_grid)
    Y_g, gY_g, HY_g = Y_all[:nG], gY_all[:nG], HY_all[:nG]

    def nonlinear_interior_hawking(eps):
        chi0 = exact_chi0_fn(eps)
        chi = chi0 + eps * Y_g
        sc, cc = np.sin(chi), np.cos(chi)
        gchi = eps * gY_g
        g2 = np.sum(gchi * gchi, axis=-1)
        W = np.sqrt(1.0 + g2 / (sc * sc))
        dA = (sc * sc) * W * W_dOm
        Area = float(np.sum(dA))
        gHg = eps * np.einsum("ni,nij,nj->n", gchi, HY_g, gchi)
        t1 = 2.0 * sc * cc + 2.0 * (cc / sc) * g2 + 42.0 * eps * Y_g
        t2 = (sc * cc * g2 + 2.0 * (cc / sc) * (g2 * g2) - gHg) / (sc * sc * W * W)
        H_m = -(1.0 / (sc * sc * W)) * (t1 - t2)
        int_H2 = float(np.sum(H_m * H_m * dA)) / (16.0 * math.pi)
        two_mH = 2.0 * math.sqrt(Area / (16.0 * math.pi)) * (1.0 - int_H2)
        max_slope = float(np.max(np.sqrt(g2) / sc))
        max_W = float(np.max(W))
        # Exact O(eps^2) non-conformal induced metric distortion |grad chi|^2 / sin^2(chi)
        return {
            "eps_rad": eps,
            "chi0_deg": math.degrees(chi0),
            "Area_over_4pi": Area / (4.0 * math.pi),
            "R_area": math.sqrt(Area / (4.0 * math.pi)),
            "Willmore_H2_over_16pi": int_H2,
            "two_mH_minus_over_a0": two_mH,
            "max_slope_grad_over_sin": max_slope,
            "max_W": max_W,
            "max_O_eps2_nonconformal_ratio": max_slope ** 2,
            "min_H_minus": float(np.min(H_m)),
            "max_H_minus": float(np.max(H_m)),
        }

    # Find exact peak of 2*m_H^-/a_0 and exact zero crossing!
    res_pk = minimize_scalar(
        lambda e: -nonlinear_interior_hawking(e)["two_mH_minus_over_a0"],
        bounds=(0.0, 0.004),
        method="bounded",
        options={"xatol": 1e-10},
    )
    eps_mH_peak = float(res_pk.x)
    mH_peak_val = float(-res_pk.fun)
    eps_mH_zero = float(brentq(lambda e: nonlinear_interior_hawking(e)["two_mH_minus_over_a0"], 0.002, 0.012))

    # Nonlinear pure-Schwarzschild exact embedding check: r(Omega) = sin(chi(Omega)) (a_0 = 1)
    # In Schwarzschild, q_AB^+ = sin^2(chi) gamma_AB + (cos^2(chi)/(1 - mu/sin(chi))) D_A chi D_B chi.
    # Exact nonlinear trace H_+ and eigenvalues of [K^A_B] at finite eps!
    def nonlinear_pure_schw_stress(eps, mu):
        chi0 = exact_chi0_fn(eps)
        chi = chi0 + eps * Y_all
        sc, cc = np.sin(chi), np.cos(chi)
        gchi = eps * gY_all
        g2 = np.sum(gchi * gchi, axis=-1)
        f = 1.0 - mu / sc
        if np.any(f <= 1e-12):
            return None
        Wm = np.sqrt(1.0 + g2 / (sc * sc))
        # In Schwarzschild with r = sc: dr = cc * dchi, so W_+ = sqrt(1 + (f^-1 * dr^2)/r^2) = sqrt(1 + cc^2 * g2 / (f * sc^2))
        Wp = np.sqrt(1.0 + (cc * cc * g2) / (f * sc * sc))
        # Exact metric mismatch between q_AB^- and q_AB^+ along gradient direction:
        # |q_++ - q_--| / q_-- = |(cc^2/f - 1) * g2 / (sc^2 + g2)|
        q_mismatch = float(np.max(np.abs((cc * cc / f - 1.0) * g2 / (sc * sc + g2))))

        # Exact nonlinear H_- (s_keep = -1):
        gHg = eps * np.einsum("ni,nij,nj->n", gchi, HY_all, gchi)
        t1m = 2.0 * sc * cc + 2.0 * (cc / sc) * g2 + 42.0 * eps * Y_all
        t2m = (sc * cc * g2 + 2.0 * (cc / sc) * (g2 * g2) - gHg) / (sc * sc * Wm * Wm)
        H_m = -(1.0 / (sc * sc * Wm)) * (t1m - t2m)

        # Exact nonlinear H_+ in Schwarzschild ds^2 = f^-1 dr^2 + r^2 gamma_AB for r = sc(Omega), s_+ = -1:
        # Exact formula for K_AB^+ = (-1/(sqrt(f)*Wp)) * [ r*f*gamma_AB + (2 - r*f'/(2f))*dr_A dr_B - D_A D_B r ]
        # Wait: let's compute via dr_A = cc*gchi_A, D_A D_B r = cc*D_A D_B chi - sc*gchi_A gchi_B,
        # and verify against exact isotropic/Schwarzschild 3D Christoffel symbols!
        # In Schwarzschild 3-metric g_rr = 1/f, g_AB = r^2 gamma_AB:
        #   Gam^r_rr = -f'/(2f), Gam^r_AB = -r*f*gamma_AB, Gam^C_rB = (1/r) delta^C_B
        #   n_i = (-1/(sqrt(f)*Wp)) * (1, -dr_A)
        #   e_A^j nabla_j e_B^r = d_A d_B r + Gam^r_rr dr_A dr_B + Gam^r_AB = D_A D_B r - (f'/(2f)) dr_A dr_B - r*f*gamma_AB
        #   e_A^j nabla_j e_B^C = (2/r) delta^C_(A dr_B) + {2}Gam^C_AB
        #   K_AB^+ = -n_i (e_A^j nabla_j e_B^i)
        #          = (1/(sqrt(f)*Wp)) * [ D_A D_B r - (f'/(2f)) dr_A dr_B - r*f*gamma_AB - (2/r) dr_A dr_B ]
        # Wait: check sign! n_r = -1/(sqrt(f)*Wp), so -n_r = +1/(sqrt(f)*Wp) -> spherical (dr=0) gives -r*f / (sqrt(f)) = -r*sqrt(f) -> H_+ = -2*sqrt(f)/r! EXACT!
        fp = mu / (sc * sc)
        # D^A D_A r = cc * (-42 eps Y) - sc * g2
        # g^A g^B (D_A D_B r) = cc * gHg - sc * g2^2
        # dr^2 = cc^2 * g2
        dr2 = cc * cc * g2
        c_dr = (2.0 / sc + 0.5 * fp / f)
        Lap_r = -42.0 * eps * cc * Y_all - sc * g2
        g_Dr_g = cc * gHg - sc * g2 * g2  # = (1/cc^2) * dr^A dr^B D_A D_B r
        # q_+^{AB} = (1/sc^2) * [ P^{AB} - (dr^A dr^B) / (f * sc^2 * Wp^2) ]
        # Trace P^{AB} K_{AB}^+ = (1/(sqrt(f)*Wp)) * [ Lap_r - c_dr * dr2 - 2*sc*f ]
        # Contraction dr^A dr^B K_{AB}^+ = (1/(sqrt(f)*Wp)) * [ cc^2*g_Dr_g - c_dr * dr2^2 - sc*f*dr2 ]
        s_f = np.sqrt(f)
        tr1 = Lap_r - c_dr * dr2 - 2.0 * sc * f
        tr2 = (cc * cc * g_Dr_g - c_dr * dr2 * dr2 - sc * f * dr2) / (f * sc * sc * Wp * Wp)
        H_p = (1.0 / (sc * sc * s_f * Wp)) * (tr1 - tr2)
        sig_8piG = H_m - H_p
        return {
            "min_8piG_sigma": float(np.min(sig_8piG)),
            "sigma_at_face": float(sig_8piG[nG]),
            "sigma_at_vertex": float(sig_8piG[nG + 12]),
            "sigma_at_edge": float(sig_8piG[nG + 32]),
            "max_q_mismatch_O_eps2": q_mismatch,
        }

    nl_rows = []
    for eps in [0.0, 0.00033, 0.002, 0.005, 0.008, 0.010, 0.014, 0.018]:
        hk = nonlinear_interior_hawking(eps)
        # Find exact nonlinear WEC ceiling in pure Schwarzschild (c=0):
        chi0 = math.radians(hk["chi0_deg"])
        mu_hi = min(0.99 * math.sin(chi0 - 1.8 * eps), math.sin(chi0) ** 3 * 1.2)
        if eps == 0.0:
            mu_nl_c0 = math.sin(CHI_INJ) ** 3
            st_c0 = nonlinear_pure_schw_stress(0.0, mu_nl_c0 * (1.0 - 1e-9))
        else:
            # Find root where min_8piG_sigma == 0
            if nonlinear_pure_schw_stress(eps, 1e-6)["min_8piG_sigma"] > 0:
                mu_nl_c0 = float(brentq(
                    lambda m: nonlinear_pure_schw_stress(eps, m)["min_8piG_sigma"],
                    1e-6, mu_hi, xtol=1e-10
                ))
                st_c0 = nonlinear_pure_schw_stress(eps, mu_nl_c0)
            else:
                mu_nl_c0 = 0.0
                st_c0 = nonlinear_pure_schw_stress(eps, 1e-6)
        hk["nonlinear_pure_Schw_WEC_mu_over_yplus"] = mu_nl_c0
        hk["nonlinear_pure_Schw_q_mismatch_at_ceiling"] = st_c0["max_q_mismatch_O_eps2"]
        hk["nonlinear_pure_Schw_sigma_face_at_ceiling"] = st_c0["sigma_at_face"]
        hk["nonlinear_pure_Schw_sigma_vert_at_ceiling"] = st_c0["sigma_at_vertex"]
        nl_rows.append(hk)

    return {
        "hawking_mass_peak": {
            "eps_peak_rad": eps_mH_peak,
            "eps_peak_deg": math.degrees(eps_mH_peak),
            "two_mH_minus_peak_over_a0": mH_peak_val,
            "gain_over_sphere_pct": 100.0 * (mH_peak_val / (math.sin(CHI_INJ) ** 3) - 1.0),
            "eps_zero_crossing_rad": eps_mH_zero,
            "eps_zero_crossing_deg": math.degrees(eps_mH_zero),
        },
        "max_nonlinear_pure_Schw_WEC_mu_over_yplus": max(r["nonlinear_pure_Schw_WEC_mu_over_yplus"] for r in nl_rows),
        "sweep": nl_rows,
    }


def main():
    print("=" * 78)
    print("SCOUT R3-4: NON-SPHERICAL I*-EQUIVARIANT FACETED THIN SHELL (l=6 / n=12)")
    print("=" * 78)

    _, R_ico, faces, verts, edges = build_dodecahedron_and_group()
    geom_res, yv_raw, N_grid, W_dOm, N_all, exact_chi0_fn = verify_geometry_and_Y6(R_ico, faces, verts, edges)
    print(f"[1] I-invariance err = {geom_res['max_I_invariance_err_Y6']:.2e}, Laplacian err = {geom_res['max_laplacian_err_Y6']:.2e}")
    print(f"    Y6_norm: vert=+1, edge={geom_res['Y6_norm_values']['edge']:+.4f}, face={geom_res['Y6_norm_values']['face']:+.4f}")
    print(f"    ||T_6||/sqrt(2): face={geom_res['T6_norm_over_sqrt2']['face']:.2e}, vert={geom_res['T6_norm_over_sqrt2']['vertex']:.2e}, "
          f"edge={geom_res['T6_norm_over_sqrt2']['edge']:.6f} (exact 189*sqrt(5)/16 = {geom_res['T6_norm_over_sqrt2']['exact_189sqrt5_over_16']:.6f})")
    dd = geom_res["dirichlet_dodecahedron"]
    print(f"    Dirichlet dodecahedron (smooth Schwarz tri vol check = {dd['vol_quadrature_check']:.15f}):")
    print(f"      chi_inj={dd['chi_inj_deg']:.3f} deg, chi_edge={dd['chi_edge_deg']:.3f} deg, chi_circ={dd['chi_circ_deg']:.4f} deg")
    print(f"      Volume-equiv sphere (100% of S^3/I*!): chi_eq={dd['chi_eq_100pct_vol_deg']:.4f} deg -> sin^3(chi_eq)={dd['sin3_chi_eq']:.6f} < pi/80={PAGE7_CEILING:.6f}!")
    print(f"      Mean boundary <chi_D>={dd['mean_chi_D_deg']:.4f} deg -> sin^3(<chi_D>)={dd['sin3_mean_chi_D']:.6f} < pi/80={PAGE7_CEILING:.6f}!")
    ml6 = geom_res["max_inscribed_l6_shell"]
    print(f"    Max inscribed l=6 shell (continuous 2D Schwarz tri): chi0={ml6['chi0_max_deg']:.5f} deg (eps={ml6['eps_opt_rad']:.6f} rad) -> sin^3(chi0)={ml6['sin3_chi0_max']:.6f} ({ml6['deficit_below_page7_pct']:.2f}% below pi/80!)")

    print("\n[2] Running 35-digit mpmath 3D/4D differential-geometry oracle...")
    oracle = verify_diffgeom_oracle_mpmath()
    print(f"    Nonlinear S^3 K^A_B^- oracle err:             {oracle['nonlinear_S3_K_mix_3D_oracle_err']:.2e}")
    print(f"    Nonlinear Brill-Lindquist R^(3)+ err:         {oracle['nonlinear_BrillLindquist_R3_vacuum_err']:.2e}")
    print(f"    Nonlinear Brill-Lindquist K^A_B^+ oracle err: {oracle['nonlinear_BrillLindquist_K_mix_3D_oracle_err']:.2e}")
    print(f"    Nonlinear Pure-Schw H_+ 3D oracle err:        {oracle['nonlinear_pure_Schw_Hp_3D_oracle_err']:.2e}")
    for m, d in oracle["static_4D_ReggeWheeler_l6"].items():
        print(f"    4D Static RW ({m}_6^2): max|dRic|={d['max_linearized_4D_Ricci_err']:.2e}, "
              f"dK_tr={d['dK_trace_4D_err']:.2e}, dK_sh={d['dK_shear_4D_err']:.2e}, dK_tt={d['dK_tautau_4D_err']:.2e}")
    print(f"    Universal +3*sin(chi0) identity err:          {oracle['universal_3sin_chi0_identity_err']:.2e}")

    print("\n[3] Running FULL-S^2 (8,254 pts) O(eps) mass-ceiling optimization across all exterior vacuum branches...")
    lin_res = analyze_linearized_ceilings(N_all, faces, yv_raw, exact_chi0_fn, ml6["eps_opt_rad"])
    print("    eps      eps_eff chi0(deg)  WEC(c=0)  WEC(opt)  DEC(BL+)  DEC(RW_P)  DEC(RW_Q)  StatNEC(ES,P)  ComovNEC(dS)")
    for r in lin_res["sweep"]:
        tag = "*" if r["in_controlled_perturbative_regime"] else " "
        print(f"   {tag}{r['eps_rad']:7.5f} {r['eps_eff_curvature']:5.2f}   {r['chi0_deg']:8.4f}   {r['mu_over_yplus_WEC_pure_Schw']:.6f}  "
              f"{r['mu_over_yplus_WEC_opt_multipole']:.6f}  {r['mu_over_yplus_DEC_BL_wormhole']:.6f}   "
              f"{r['mu_over_yplus_DEC_RW_P62']:.6f}   {r['mu_over_yplus_DEC_RW_Q62']:.6f}   "
              f"{r['mu_over_yplus_StaticNEC_gamma_quarter_RW_P62']:.6f}      "
              f"{r['mu_over_yplus_ComovingNEC_pure_dS_exact_zeromean']:.6f}")
    ls = lin_res["summary"]["extrapolated_envelope_all_eps"]
    lc = lin_res["summary"]["controlled_perturbative_max_eps_eff_le_0p5"]
    print(f"    -> Controlled perturbative (eps_eff<=0.5): WEC(opt)={lc['max_WEC_opt_multipole']:.6f}, DEC(RW_P)={lc['max_DEC_RW_P62']:.6f}, StatNEC(ES)={lc['max_StaticNEC_ES_RW_P62']:.6f}")
    print(f"    -> Extrapolated envelope (all eps): WEC(c=0)={ls['max_WEC_pure_Schw']:.6f}, WEC(opt)={ls['max_WEC_opt_multipole']:.6f}, "
          f"DEC(BL+/RW_P/RW_Q)={ls['max_DEC_BL_wormhole']:.6f}/{ls['max_DEC_RW_P62']:.6f}/{ls['max_DEC_RW_Q62']:.6f}, StatNEC(ES,P)={ls['max_StaticNEC_ES_RW_P62']:.6f}")

    print("\n[4] Running exact nonlinear finite-amplitude interior Hawking mass & pure-Schw junction...")
    nl_res = analyze_nonlinear_finite_amplitude(N_grid, W_dOm, N_all, faces, yv_raw, exact_chi0_fn)
    pk = nl_res["hawking_mass_peak"]
    print(f"    Interior Hawking mass 2*m_H^-/a0 peak = {pk['two_mH_minus_peak_over_a0']:.6f} (+{pk['gain_over_sphere_pct']:.3f}%) at eps={pk['eps_peak_rad']:.5f} rad ({pk['eps_peak_deg']:.4f} deg)")
    print(f"    Interior Hawking mass crosses ZERO at eps={pk['eps_zero_crossing_rad']:.5f} rad ({pk['eps_zero_crossing_deg']:.3f} deg), drops to {nl_res['sweep'][-1]['two_mH_minus_over_a0']:+.6f} at eps=0.018!")
    print("    eps     chi0(deg)   2*m_H^-/a0   NL_WEC(c=0)  max_slope  q_mismatch(O(eps^2))")
    for r in nl_res["sweep"]:
        print(f"    {r['eps_rad']:6.5f} {r['chi0_deg']:8.4f}   {r['two_mH_minus_over_a0']:+9.6f}   "
              f"{r['nonlinear_pure_Schw_WEC_mu_over_yplus']:.6f}    {r['max_slope_grad_over_sin']:.4f}     "
              f"{r['nonlinear_pure_Schw_q_mismatch_at_ceiling']:.4e}")

    payload = {
        "scout": "R3-4",
        "title": "Non-Spherical I*-Equivariant Faceted Thin-Shell Israel-Darmois Matching (l=6 / n=12)",
        "geometry_and_Y6": geom_res,
        "mpmath_35dps_diffgeom_oracle": oracle,
        "linearized_O_eps_mass_ceilings": lin_res,
        "nonlinear_finite_amplitude": nl_res,
        "verdict": {
            "can_l6_faceted_shell_reach_page7_pi_over_80": bool(ls["max_WEC_opt_multipole"] >= PAGE7_CEILING),
            "page7_pi_over_80": PAGE7_CEILING,
            "spherical_inscribed_ceiling": math.sin(CHI_INJ) ** 3,
            "controlled_perturbative_WEC_opt_ceiling": lc["max_WEC_opt_multipole"],
            "controlled_perturbative_DEC_RW_P62_ceiling": lc["max_DEC_RW_P62"],
            "extrapolated_l6_WEC_only_ceiling": ls["max_WEC_opt_multipole"],
            "extrapolated_l6_WEC_DEC_wormhole_ceiling": ls["max_DEC_BL_wormhole"],
            "extrapolated_l6_WEC_DEC_RW_P62_ceiling": ls["max_DEC_RW_P62"],
            "extrapolated_l6_WEC_DEC_RW_Q62_ceiling": ls["max_DEC_RW_Q62"],
            "extrapolated_l6_WEC_pure_Schw_linear": ls["max_WEC_pure_Schw"],
            "best_l6_WEC_pure_Schw_nonlinear": nl_res["max_nonlinear_pure_Schw_WEC_mu_over_yplus"],
            "volume_equivalent_100pct_S3Istar_ceiling": dd["sin3_chi_eq"],
        },
    }
    out_path = OUT_DIR / "r3_04_faceted_shell.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
