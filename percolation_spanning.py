"""
Area-weighted Newman-Ziff bond percolation with spanning detection, per
Sec. VI of the paper.

Key differences from the simple node-counting version in percolation.py:
  - Cluster "size" = SUM OF AREAS of the blocks (sites) in the cluster,
    not the number of sites (paper, Sec. VI: "clusters formed by occupied
    bonds are characterized not by the number of sites but by their total
    area").
  - The percolation threshold is obtained from the spanning probability
    W(p, L): the fraction of realizations in which a cluster connects one
    side of the lattice to the opposite side, as a function of p, for
    several system sizes L. Curves for different L intersect near p_c
    (Sec. VI.A, eq. 29-32).
"""
import numpy as np
from scipy.stats import binom


class AreaUnionFind:
    """Union-find where each site carries a real-valued weight (area)
    instead of unit weight, and each cluster tracks whether it touches
    each of the 4 domain boundaries (for spanning detection)."""

    def __init__(self, area, touches_left, touches_right,
                 touches_bottom, touches_top):
        n = len(area)
        self.parent = np.arange(n, dtype=np.int64)
        self.area = np.array(area, dtype=np.float64)  # cluster total area, indexed by root
        self.touches = np.zeros((n, 4), dtype=bool)    # [L, R, B, T] per root
        self.touches[:, 0] = touches_left
        self.touches[:, 1] = touches_right
        self.touches[:, 2] = touches_bottom
        self.touches[:, 3] = touches_top

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        if self.area[ra] < self.area[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.area[ra] += self.area[rb]
        self.touches[ra] |= self.touches[rb]
        return ra

    def is_spanning(self, root):
        L, R, B, T = self.touches[root]
        return (L and R) or (B and T)


def newman_ziff_spanning_run(node_area, boundary_flags, edges, rng):
    """One Newman-Ziff sweep tracking area-weighted largest cluster,
    area^2 sum (for susceptibility), and the bond count at which a
    spanning cluster first appears.

    Parameters
    ----------
    node_area : (N,) array of per-site areas
    boundary_flags : (N,4) bool array, columns = [touches_left,
                     touches_right, touches_bottom, touches_top]
    edges : (M,2) int array
    rng : numpy Generator

    Returns
    -------
    largest_area : (M+1,) array, largest cluster's total area after n bonds
    sum_area_sq   : (M+1,) array, sum over clusters of (cluster area)^2
    n_c           : int or None, bond count at which spanning first occurs
                     (None if never within this sweep)
    """
    M = len(edges)
    order = rng.permutation(M)
    uf = AreaUnionFind(node_area, boundary_flags[:, 0], boundary_flags[:, 1],
                        boundary_flags[:, 2], boundary_flags[:, 3])

    largest_area = np.empty(M + 1, dtype=np.float64)
    sum_area_sq = np.empty(M + 1, dtype=np.float64)

    cur_largest = float(node_area.max()) if len(node_area) else 0.0
    cur_sum_sq = float(np.sum(node_area ** 2))
    largest_area[0] = cur_largest
    sum_area_sq[0] = cur_sum_sq

    n_c = None
    # check if any single isolated site already spans (degenerate case)
    for i in range(len(node_area)):
        if uf.is_spanning(i):
            n_c = 0
            break

    for step in range(M):
        a, b = edges[order[step]]
        ra, rb = uf.find(a), uf.find(b)
        if ra != rb:
            aa, ab = uf.area[ra], uf.area[rb]
            cur_sum_sq += 2.0 * aa * ab
            new_root = uf.union(ra, rb)
            new_area = uf.area[new_root]
            if new_area > cur_largest:
                cur_largest = new_area
            if n_c is None and uf.is_spanning(new_root):
                n_c = step + 1
        largest_area[step + 1] = cur_largest
        sum_area_sq[step + 1] = cur_sum_sq

    return largest_area, sum_area_sq, n_c


def build_node_arrays(lattice, occ_ids, tol=1e-9):
    """Per-occupied-block area and boundary-touching flags, in the same
    contiguous 0..N-1 node order used by edges_from_adjacency."""
    occ_ids = list(occ_ids)
    rects = lattice.rects
    area = np.array([lattice.area[i] for i in occ_ids])
    flags = np.zeros((len(occ_ids), 4), dtype=bool)
    for k, i in enumerate(occ_ids):
        x0, y0, x1, y1 = rects[i]
        flags[k, 0] = x0 < tol            # touches left
        flags[k, 1] = x1 > 1.0 - tol      # touches right
        flags[k, 2] = y0 < tol            # touches bottom
        flags[k, 3] = y1 > 1.0 - tol      # touches top
    return area, flags


def canonical_average(microcanonical_values, p_values):
    """Same binomial reweighting as in percolation.py (kept local here to
    avoid a cross-module import cycle)."""
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
