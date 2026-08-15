"""
Finite-size scaling (FSS) pipeline for WPSPL bond percolation, following
Sec. VI.A of the paper: spanning probability W(p,L) computed for several
system sizes L, critical threshold p_c from curve intersection, and the
correlation-length exponent 1/nu from the finite-size shift
    p_c - p(L) ~ L^{-1/nu}.

System size: the paper's own yardstick relation delta(t) ~ t^{-1/2}
(Eq. 11) means the natural linear system size is L(t) = sqrt(t), which is
consistent with N(t) ~ L^{2(sqrt(3+q)-1)} matching the fractal dimension
formula validated in Stage 2.
"""
import numpy as np

from lattice import WPSPL
from adjacency import build_adjacency
from percolation import edges_from_adjacency
from percolation_spanning import (
    newman_ziff_spanning_run, canonical_average, build_node_arrays,
)


def estimate_p_window(q, n_steps, half_width=0.20, n_points=141,
                       coarse_n_points=61, seed=1):
    """Quick single-lattice, sparse-grid scan to locate roughly where the
    spanning transition sits for this q, then build a p_values grid
    centered on it. This replaces a hardcoded p range (which only works
    for whichever q it happened to be tuned on) with one that adapts
    automatically -- p_c shifts substantially with q, so a fixed window
    silently fails (curves can saturate before the window even starts,
    or never cross within it) for q far from the one it was tuned for."""
    coarse_p = np.linspace(0.05, 0.95, coarse_n_points)
    W_p, M, _ = spanning_curve_for_size(q, n_steps, n_lattices=1,
                                         n_bond_realizations=5,
                                         p_values=coarse_p, seed=seed)
    p_center = find_W_crossing_p(coarse_p, W_p, 0.5)
    if p_center is None:
        p_center = coarse_p[np.argmin(np.abs(W_p - 0.5))]
    lo = max(0.0, p_center - half_width)
    hi = min(1.0, p_center + half_width)
    return np.linspace(lo, hi, n_points), p_center


def system_size_L(n_steps):
    return np.sqrt(n_steps)


def _one_lattice_n_c_samples(args):
    """Worker: generate one WPSPL lattice and run n_bond_realizations
    independent Newman-Ziff spanning sweeps on it. Must be a module-level
    function (not a closure/lambda) so it can be pickled for
    multiprocessing."""
    q, n_steps, n_bond_realizations, lat_seed, bond_seed = args
    lat = WPSPL(q=q, n_steps=n_steps, seed=lat_seed).generate()
    occ = lat.occupied_ids()
    adj = build_adjacency(lat, occ)
    n_nodes, edges, _ = edges_from_adjacency(occ, adj)
    area, flags = build_node_arrays(lat, occ)
    M = len(edges)

    rng = np.random.default_rng(bond_seed)
    n_c_list = []
    for _ in range(n_bond_realizations):
        _, _, n_c = newman_ziff_spanning_run(area, flags, edges, rng)
        n_c_list.append(n_c if n_c is not None else M + 1)
    return n_c_list, M


def spanning_curve_for_size(q, n_steps, n_lattices, n_bond_realizations,
                             p_values, seed, pool=None):
    """Run n_lattices independent WPSPL realizations at fixed n_steps,
    each with n_bond_realizations independent bond orderings, pool ALL
    the resulting spanning thresholds n_c together, and return W(p) on
    the given p_values grid (canonical, via binomial reweighting of the
    pooled empirical CDF).

    If `pool` (a multiprocessing.Pool) is given, the n_lattices
    realizations are distributed across worker processes -- this is the
    expensive, embarrassingly-parallel part of the computation, so it
    benefits directly from however many cores are available."""
    args = [(q, n_steps, n_bond_realizations,
              seed + lat_idx, seed + 1000000 + lat_idx)
            for lat_idx in range(n_lattices)]

    if pool is not None:
        results = pool.map(_one_lattice_n_c_samples, args)
    else:
        results = [_one_lattice_n_c_samples(a) for a in args]

    all_n_c = []
    M_ref = None
    for n_c_list, M in results:
        all_n_c.extend(n_c_list)
        if M_ref is None:
            M_ref = M

    M_common = M_ref
    n_c_arr = np.array(all_n_c)
    ns = np.arange(M_common + 1)
    W_n = np.array([(n_c_arr <= n).mean() for n in ns])
    W_p = canonical_average(W_n, p_values)
    return W_p, M_common, len(all_n_c)


def find_pairwise_crossing(p_values, W_a, W_b, w_lo=0.02, w_hi=0.98):
    """Find the p where two W(p) curves cross, restricted to the region
    where BOTH curves are away from their saturated tails (0 or 1). Far
    out in the tails, W is numerically indistinguishable from 0/1 and
    floating-point noise creates spurious sign changes that have nothing
    to do with the real transition; restricting to the non-saturated
    region avoids picking one of those up instead of the genuine
    crossing."""
    diff = W_a - W_b
    active = (W_a > w_lo) & (W_a < w_hi) & (W_b > w_lo) & (W_b < w_hi)
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    sign_changes = [i for i in sign_changes if active[i] and active[i + 1]]
    if len(sign_changes) == 0:
        return None
    # if multiple genuine crossings remain, use the one closest to where
    # both curves are nearest 0.5 (most central / best-sampled point)
    i = min(sign_changes, key=lambda k: abs(W_a[k] - 0.5) + abs(W_b[k] - 0.5))
    p0, p1 = p_values[i], p_values[i + 1]
    d0, d1 = diff[i], diff[i + 1]
    if d1 == d0:
        return 0.5 * (p0 + p1)
    frac = -d0 / (d1 - d0)
    return p0 + frac * (p1 - p0)


def find_W_crossing_p(p_values, W_p, W_target=0.5):
    """p at which W(p) crosses W_target, via linear interpolation."""
    idx = np.searchsorted(W_p, W_target)
    if idx <= 0 or idx >= len(p_values):
        return None
    p0, p1 = p_values[idx - 1], p_values[idx]
    w0, w1 = W_p[idx - 1], W_p[idx]
    if w1 == w0:
        return p0
    frac = (W_target - w0) / (w1 - w0)
    return p0 + frac * (p1 - p0)


def robust_p_c_estimate(p_values, W_curves, sizes, tol=0.05):
    """Aggregate pairwise curve crossings into a single p_c estimate,
    discarding likely-spurious crossings.

    Crossings between CLOSELY-SPACED sizes are unreliable: their W(p)
    curves are similar enough in shape that sampling noise can create a
    spurious crossing far from the true transition (seen in practice for
    adjacent small sizes with modest sample counts). The per-curve
    W=0.5 points (p_half) are individually noisier but don't suffer from
    this specific failure mode, so they make a reasonable reference:
    any pairwise crossing far from the p_half cluster is dropped."""
    p_half = np.array([find_W_crossing_p(p_values, W_curves[n], 0.5) for n in sizes],
                       dtype=float)
    valid_half = p_half[~np.isnan(p_half)]
    reference = np.median(valid_half) if len(valid_half) else None

    crossings = []
    for i in range(len(sizes) - 1):
        c = find_pairwise_crossing(p_values, W_curves[sizes[i]], W_curves[sizes[i + 1]])
        if c is None:
            continue
        if reference is not None and abs(c - reference) > tol:
            print(f"  discarding outlier crossing t={sizes[i]}&{sizes[i+1]}: "
                  f"p={c:.4f} (>{tol} from reference {reference:.4f})")
            continue
        crossings.append(c)

    if crossings:
        return float(np.mean(crossings)), crossings, p_half
    elif reference is not None:
        # fall back to the p_half median if every pairwise crossing was
        # rejected (e.g. all sizes too closely spaced)
        return float(reference), crossings, p_half
    return None, crossings, p_half
    """p at which W(p) crosses W_target, via linear interpolation."""
    idx = np.searchsorted(W_p, W_target)
    if idx <= 0 or idx >= len(p_values):
        return None
    p0, p1 = p_values[idx - 1], p_values[idx]
    w0, w1 = W_p[idx - 1], W_p[idx]
    if w1 == w0:
        return p0
    frac = (W_target - w0) / (w1 - w0)
    return p0 + frac * (p1 - p0)
