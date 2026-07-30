
import numpy as np


class Fenwick:

    def __init__(self, capacity):
        self.n = capacity
        self.tree = np.zeros(capacity + 1, dtype=np.float64)
        # highest power of two <= n, used by find()
        p = 1
        while p * 2 <= self.n:
            p *= 2
        self._top_pow = p

    def update(self, i, delta):
        """Add `delta` to slot i (0-indexed)."""
        i += 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def prefix_sum(self, i):
        """Sum of slots [0..i] inclusive (0-indexed)."""
        i += 1
        s = 0.0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def total(self):
        s = 0.0
        i = self.n
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def find(self, target):
        """Smallest 0-indexed slot i such that prefix_sum(0..i) >= target.
        Assumes 0 < target <= total(). Standard Fenwick binary search."""
        idx = 0
        remaining = target
        bitmask = self._top_pow
        while bitmask != 0:
            next_idx = idx + bitmask
            if next_idx <= self.n and self.tree[next_idx] < remaining:
                idx = next_idx
                remaining -= self.tree[next_idx]
            bitmask //= 2
        return idx  # 0-indexed slot


class WPSPL:
    def __init__(self, q, n_steps, seed=None):		#q = Probability of retained daughter block
        assert 0.0 <= q <= 1.0
        self.q = q
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)

        max_blocks = 1 + 4 * n_steps + 8
        self.rects = np.zeros((max_blocks, 4), dtype=np.float64)  # x0,y0,x1,y1
        self.area = np.zeros(max_blocks, dtype=np.float64)
        self.alive = np.zeros(max_blocks, dtype=bool)   # currently occupied leaf
        self.is_void = np.zeros(max_blocks, dtype=bool)  # permanently removed leaf
        self.was_processed = np.zeros(max_blocks, dtype=bool)  # subdivided already

        self.fen = Fenwick(max_blocks)
        self.next_id = 0
        self.t = 0
        self.n_noop = 0
        self.n_splits = 0

        # root block
        self._add_block(0.0, 0.0, 1.0, 1.0, occupied=True)

    def _add_block(self, x0, y0, x1, y1, occupied):
        i = self.next_id
        self.next_id += 1
        a = (x1 - x0) * (y1 - y0)
        self.rects[i] = (x0, y0, x1, y1)
        self.area[i] = a
        if occupied:
            self.alive[i] = True
            self.fen.update(i, a)
        else:
            self.is_void[i] = True
        return i

    def step(self):
        M = self.fen.total()
        R1 = self.rng.uniform(0.0, 1.0)
        if R1 > M:
            self.n_noop += 1
            self.t += 1
            return
        idx = self.fen.find(R1)
        x0, y0, x1, y1 = self.rects[idx]

        # remove parent from occupied set
        self.fen.update(idx, -self.area[idx])
        self.alive[idx] = False
        self.was_processed[idx] = True

        u = self.rng.uniform(x0, x1)
        v = self.rng.uniform(y0, y1)
        sub_rects = [
            (x0, y0, u, v),   # 0: bottom-left  (subject to retention test)
            (u, y0, x1, v),   # 1: bottom-right
            (x0, v, u, y1),   # 2: top-left
            (u, v, x1, y1),   # 3: top-right
        ]
        for k, r in enumerate(sub_rects):
            if k == 0:
                R2 = self.rng.uniform(0.0, 1.0)
                keep = R2 <= self.q
                self._add_block(*r, occupied=keep)
            else:
                self._add_block(*r, occupied=True)

        self.n_splits += 1
        self.t += 1

    def generate(self):
        for _ in range(self.n_steps):
            self.step()
        return self

    def occupied_ids(self):
        return np.nonzero(self.alive[: self.next_id])[0]

    def void_ids(self):
        return np.nonzero(self.is_void[: self.next_id])[0]

    def leaf_ids(self):
        mask = self.alive[: self.next_id] | self.is_void[: self.next_id]
        return np.nonzero(mask)[0]
