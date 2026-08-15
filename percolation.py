"""
Newman-Ziff incremental bond percolation.

Core idea: instead of running percolation separately for each occupation
probability p (which would mean rebuilding clusters from scratch each
time), add bonds ONE AT A TIME in random order and track cluster sizes
incrementally with a union-find structure. After n bonds have been added,
the state is exactly a bond-percolation configuration with n occupied
bonds out of M total ("microcanonical" ensemble). A single sweep through
all M bonds therefore yields the full curve of any cluster observable as
a function of n = 0..M, in O(M alpha(M)) time.

To get results at fixed occupation PROBABILITY p (the physically usual
"canonical" ensemble), the microcanonical curve Q(n) is reweighted by the
binomial distribution:

    <Q>(p) = sum_{n=0}^{M} B(n; M, p) * Q(n),   B(n;M,p) = C(M,n) p^n (1-p)^(M-n)

Reference: M.E.J. Newman & R.M. Ziff, Phys. Rev. E 64, 016706 (2001).
"""
import numpy as np
from scipy.stats import binom


class UnionFind:
    """Union-find (disjoint set) with union-by-size and path compression."""

    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        """Union the sets containing a and b. Returns the root of the
        merged set, or None if a and b were already in the same set."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra


def newman_ziff_run(n_nodes, edges, rng):
    """Single Newman-Ziff sweep.

    Parameters
    ----------
    n_nodes : int, number of nodes (must be labeled 0..n_nodes-1)
    edges   : (M,2) int array of bonds
    rng     : numpy Generator

    Returns
    -------
    largest   : (M+1,) array, largest cluster size after n bonds added
    sum_sq    : (M+1,) array, sum over all clusters of (cluster size)^2
                after n bonds added
    """
    M = len(edges)
    order = rng.permutation(M)

    uf = UnionFind(n_nodes)
    largest = np.empty(M + 1, dtype=np.int64)
    sum_sq = np.empty(M + 1, dtype=np.float64)

    cur_largest = 1 if n_nodes > 0 else 0
    cur_sum_sq = float(n_nodes)  # N clusters of size 1 -> sum of squares = N

    largest[0] = cur_largest
    sum_sq[0] = cur_sum_sq

    for step in range(M):
        a, b = edges[order[step]]
        ra, rb = uf.find(a), uf.find(b)
        if ra != rb:
            sa, sb = uf.size[ra], uf.size[rb]
            cur_sum_sq += 2.0 * sa * sb  # (sa+sb)^2 - sa^2 - sb^2
            new_root = uf.union(ra, rb)
            new_size = uf.size[new_root]
            if new_size > cur_largest:
                cur_largest = new_size
        largest[step + 1] = cur_largest
        sum_sq[step + 1] = cur_sum_sq

    return largest, sum_sq


def average_realizations(n_nodes, edges, n_realizations, seed=None):
    """Run several independent Newman-Ziff sweeps (independent random bond
    orderings) on the SAME graph and average the microcanonical curves.
    Averaging microcanonical data before canonical reweighting is
    equivalent to (and cheaper than) averaging canonical curves, since
    canonical reweighting is linear."""
    rng = np.random.default_rng(seed)
    M = len(edges)
    largest_sum = np.zeros(M + 1, dtype=np.float64)
    sum_sq_sum = np.zeros(M + 1, dtype=np.float64)
    for _ in range(n_realizations):
        largest, sum_sq = newman_ziff_run(n_nodes, edges, rng)
        largest_sum += largest
        sum_sq_sum += sum_sq
    return largest_sum / n_realizations, sum_sq_sum / n_realizations


def canonical_average(microcanonical_values, p_values):
    """Reweight a microcanonical curve (indexed n=0..M) to canonical
    occupation-probability values p via the binomial distribution.
    O(M) per p value."""
    M = len(microcanonical_values) - 1
    ns = np.arange(M + 1)
    out = np.empty(len(p_values), dtype=np.float64)
    for i, p in enumerate(p_values):
        if p <= 0.0:
            out[i] = microcanonical_values[0]
        elif p >= 1.0:
            out[i] = microcanonical_values[-1]
        else:
            weights = binom.pmf(ns, M, p)
            out[i] = np.dot(weights, microcanonical_values)
    return out


def order_parameter_and_susceptibility(n_nodes, edges, p_values,
                                        n_realizations=1, seed=None):
    """Convenience wrapper: returns P(p) = <largest cluster>/N and the
    finite-size susceptibility analog chi(p) = <sum_sq - largest^2>/N
    (second moment excluding the largest/spanning cluster)."""
    largest_avg, sum_sq_avg = average_realizations(
        n_nodes, edges, n_realizations, seed=seed)
    P_micro = largest_avg / n_nodes
    chi_micro = (sum_sq_avg - largest_avg ** 2) / n_nodes

    P_p = canonical_average(P_micro, p_values)
    chi_p = canonical_average(chi_micro, p_values)
    return P_p, chi_p


def edges_from_adjacency(occ_ids, adj):
    """Convert an adjacency dict (block_id -> set(neighbor block_ids), as
    produced by adjacency.build_adjacency) into a contiguous 0..N-1 node
    labeling and an (M,2) edge array, suitable for newman_ziff_run.

    Returns (n_nodes, edges, id_to_node) where id_to_node maps the
    original lattice block id -> contiguous node index.
    """
    occ_ids = list(occ_ids)
    id_to_node = {bid: k for k, bid in enumerate(occ_ids)}
    edge_set = set()
    for bid in occ_ids:
        a = id_to_node[bid]
        for nb in adj.get(bid, ()):
            b = id_to_node.get(nb)
            if b is None:
                continue  # neighbor not in occ_ids (shouldn't happen)
            e = (a, b) if a < b else (b, a)
            edge_set.add(e)
    edges = np.array(sorted(edge_set), dtype=np.int64)
    return len(occ_ids), edges, id_to_node
