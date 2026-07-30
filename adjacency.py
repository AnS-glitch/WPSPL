from collections import defaultdict
import numpy as np


def _pairs_from_groups(left_groups, right_groups):
    edges = []
    for c, right_list in right_groups.items():
        left_list = left_groups.get(c)
        if not left_list:
            continue
        A = sorted(right_list)  # (lo, hi, id)
        B = sorted(left_list)
        j_start = 0
        for lo_a, hi_a, id_a in A:
            for lo_b, hi_b, id_b in B:
                if lo_b >= hi_a:
                    break
                lo = max(lo_a, lo_b)
                hi = min(hi_a, hi_b)
                if hi - lo > 1e-12:
                    edges.append((id_a, id_b))
    return edges


def build_adjacency(lattice, ids=None, coord_round=9):
    if ids is None:
        ids = lattice.occupied_ids()
    ids = list(ids)

    x0_groups = defaultdict(list)  # coord -> [(y0,y1,id)]  blocks whose LEFT edge is at coord
    x1_groups = defaultdict(list)  # blocks whose RIGHT edge is at coord
    y0_groups = defaultdict(list)  # blocks whose BOTTOM edge is at coord
    y1_groups = defaultdict(list)  # blocks whose TOP edge is at coord

    rects = lattice.rects
    for i in ids:
        x0, y0, x1, y1 = rects[i]
        x0r, x1r = round(x0, coord_round), round(x1, coord_round)
        y0r, y1r = round(y0, coord_round), round(y1, coord_round)
        x0_groups[x0r].append((y0, y1, i))
        x1_groups[x1r].append((y0, y1, i))
        y0_groups[y0r].append((x0, x1, i))
        y1_groups[y1r].append((x0, x1, i))

    # vertical shared edges: right-edge owner (x1_groups[c]) touches left-edge owner (x0_groups[c])
    v_edges = _pairs_from_groups(x0_groups, x1_groups)
    # horizontal shared edges: top-edge owner (y1_groups[c]) touches bottom-edge owner (y0_groups[c])
    h_edges = _pairs_from_groups(y0_groups, y1_groups)

    adj = defaultdict(set)
    for a, b in v_edges + h_edges:
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
    return adj


def brute_force_adjacency(lattice, ids=None, tol=1e-9):
    if ids is None:
        ids = lattice.occupied_ids()
    ids = list(ids)
    rects = lattice.rects
    adj = defaultdict(set)
    for a in range(len(ids)):
        ia = ids[a]
        x0a, y0a, x1a, y1a = rects[ia]
        for b in range(a + 1, len(ids)):
            ib = ids[b]
            x0b, y0b, x1b, y1b = rects[ib]
            # vertical shared edge (side by side)
            if abs(x1a - x0b) < tol or abs(x1b - x0a) < tol:
                lo = max(y0a, y0b)
                hi = min(y1a, y1b)
                if hi - lo > tol:
                    adj[ia].add(ib)
                    adj[ib].add(ia)
                    continue
            # horizontal shared edge (stacked)
            if abs(y1a - y0b) < tol or abs(y1b - y0a) < tol:
                lo = max(x0a, x0b)
                hi = min(x1a, x1b)
                if hi - lo > tol:
                    adj[ia].add(ib)
                    adj[ib].add(ia)
    return adj
