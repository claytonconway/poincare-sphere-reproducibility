"""Shared mathematical primitives for the 5 Scout Step experiments.

Implements exact formulas from Pages 13–20 of the compiled manuscript:
- Setspace fibers F_n, Mersenne horizons, phase coordinates mu_x, mu_tilde_x
- Page 17–18 Number Field channels P_n, F_n, Gram matrix G_n
- Page 13 & 20 centered divisor polynomials Phi_n(z)
- Standardized JSON artifact writer for cross-scout comparison
"""

import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def prime_factorization(n: int) -> List[Tuple[int, int]]:
    """Return sorted list of (prime, exponent) pairs for integer n >= 1."""
    if n <= 1:
        return []
    factors: List[Tuple[int, int]] = []
    d = 2
    rem = n
    while d * d <= rem:
        if rem % d == 0:
            exp = 0
            while rem % d == 0:
                rem //= d
                exp += 1
            factors.append((d, exp))
        d += 1 if d == 2 else 2
    if rem > 1:
        factors.append((rem, 1))
    return factors


def is_mersenne(n: int) -> bool:
    """Return True iff n = 2^d - 1 for some integer d >= 1."""
    return n >= 1 and ((n + 1) & n) == 0


def setspace_fiber(n: int) -> List[Tuple[int, int, int, float]]:
    """Return list of (n, d, mu_x, mu_tilde_x) for x in F_n (Page 17)."""
    d_n = int(math.ceil(math.log2(n + 1)))
    if is_mersenne(n):
        ds = [d_n, d_n + 1]
    else:
        ds = [d_n]
    mus = [n + d - 1 for d in ds]
    mean_mu = sum(mus) / len(mus)
    return [(n, d, mu, mu - mean_mu) for d, mu in zip(ds, mus)]


def make_domain_grid(grid_size: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build midpoint quadrature grid on D* = [-1, 1]^2 \\ {(0,0)}.

    Using cell centers avoids (0, 0) identically for any even grid_size and
    gives exact 2D midpoint integration weights (du*dv / 4 = 1 / grid_size^2).
    """
    if grid_size % 2 != 0:
        raise ValueError("grid_size must be even so (0, 0) is avoided cleanly")
    step = 2.0 / grid_size
    coords = -1.0 + step * (np.arange(grid_size) + 0.5)
    u, v = np.meshgrid(coords, coords, indexing="xy")
    rho = np.hypot(u, v)
    theta = np.arctan2(v, u)
    return u, v, rho, theta


def number_field_channels(
    n: int,
    u: np.ndarray,
    v: np.ndarray,
    rho: np.ndarray,
    theta: np.ndarray,
    phase_shift: float = 0.0,
    factors_override: List[Tuple[int, int]] = None,
) -> Tuple[List[str], np.ndarray]:
    """Compute Page 17–18 primitive channel stack P_n on grid (u, v).

    Parameters
    ----------
    n : int
        Positive integer >= 1.
    u, v, rho, theta : np.ndarray
        Grid coordinates on D*.
    phase_shift : float
        Additive shift delta applied to mu_x (used by Scout #2 Moiré probe).
    factors_override : list of (p, a), optional
        Optional surrogate prime factorization for null-control experiments.

    Returns
    -------
    labels : list[str]
        Channel identifiers matching Page 18 A_n ordering.
    channels : np.ndarray
        Array of shape (nu_n, H, W).
    """
    fiber = setspace_fiber(n)
    m_n = len(fiber)
    factors = prime_factorization(n) if factors_override is None else factors_override
    omega = len(factors)
    big_omega = sum(a for _, a in factors)

    labels: List[str] = []
    chans: List[np.ndarray] = []

    exp_neg_rho2 = np.exp(-(rho ** 2))
    exp_neg_rho_m1_2 = np.exp(-((rho - 1.0) ** 2))

    # 1. Parity channels: ({par} x F_n)
    for _, d, mu_x, mu_tilde in fiber:
        mu_eff = mu_x + phase_shift
        rho_x = np.hypot(u, v - mu_tilde)
        if n % 2 == 0:
            g_par = np.exp(-(rho_x ** 2)) + np.exp(-((rho_x - 1.0) ** 2))
        else:
            g_par = u * np.exp(-(rho_x ** 2)) + np.sin(theta + mu_eff) * exp_neg_rho_m1_2
        labels.append(f"par:d={d}")
        chans.append(g_par)

    # 2. Harmonic channels: ({harm} x F_n x J_n)
    for _, d, mu_x, _ in fiber:
        mu_eff = mu_x + phase_shift
        for k, (p_k, a_k) in enumerate(factors):
            rho_circ = (k + 1.0) / (omega + 1.0)
            g_harm = (a_k / (big_omega + 1.0)) * np.cos(p_k * theta - mu_eff) * np.exp(-((rho - rho_circ) ** 2))
            labels.append(f"harm:d={d}:k={k}")
            chans.append(g_harm)

    # 3. Radial channels: ({rad} x J_n), weighted by m_n per Page 18
    for k, (p_k, a_k) in enumerate(factors):
        g_rad = (exp_neg_rho2 / (omega + 1.0)) * np.cos((a_k + 1.0) * math.pi * rho + p_k)
        labels.append(f"rad:k={k}")
        chans.append(m_n * g_rad)

    # 4. Global phase channels: ({ph} x F_n)
    log_n = math.log(n)
    for _, d, mu_x, _ in fiber:
        mu_eff = mu_x + phase_shift
        g_ph = math.cos(log_n + mu_eff) * exp_neg_rho2
        labels.append(f"ph:d={d}")
        chans.append(g_ph)

    return labels, np.stack(chans, axis=0)


def gram_matrix(channels: np.ndarray) -> np.ndarray:
    """Compute Page 18 L2 Gram matrix G_n with <f, g>_2 = (1/4) int_{[-1,1]^2} f g du dv."""
    nu_n = channels.shape[0]
    flat = channels.reshape(nu_n, -1)
    n_pts = flat.shape[1]
    return (flat @ flat.T) / float(n_pts)


def all_divisors(n: int) -> List[int]:
    """Return sorted list of all positive divisors of n."""
    divs = [1]
    for p, a in prime_factorization(n):
        cur = []
        p_pow = 1
        for _ in range(a + 1):
            for d in divs:
                cur.append(d * p_pow)
            p_pow *= p
        divs = cur
    return sorted(divs)


def phi_n_array(n: int, z: np.ndarray) -> np.ndarray:
    """Evaluate centered divisor polynomial Phi_n(z) = n^(-z/2) sum_{d|n} d^z."""
    divs = np.array(all_divisors(n), dtype=np.float64)
    log_divs = np.log(divs)
    # Broadcast: z has arbitrary shape; compute sum_d exp(z * (log d - 0.5 log n))
    shifts = log_divs - 0.5 * math.log(n)
    return np.sum(np.exp(np.multiply.outer(z, shifts)), axis=-1)


def save_result(scout_id: str, payload: Dict) -> str:
    """Persist standardized scout result JSON under data/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{scout_id}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    return out_path
