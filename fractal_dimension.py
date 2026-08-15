"""
Estimate the Hausdorff (box-counting) dimension of the occupied region of
a WPSPL lattice, and compare against the paper's analytic prediction
(Sec. III, eq. 15 maximum of f(alpha)):

    d_f(q) = 2 * (sqrt(3 + q) - 1)

Method: rasterize the occupied blocks onto a fine boolean grid, then count
how many boxes of size eps (eps = box_size / grid_size) contain at least
one occupied pixel, for a range of box sizes. A power law
    N(eps) ~ eps^-d_f
appears as a straight line on a log(N) vs log(1/eps) plot; its slope is
the fractal dimension.
"""
import numpy as np


def rasterize_occupied(lattice, occ_ids, grid_size):
    """Boolean grid_size x grid_size raster: True where any occupied
    block covers that pixel."""
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    rects = lattice.rects
    for i in occ_ids:
        x0, y0, x1, y1 = rects[i]
        ix0 = int(np.floor(x0 * grid_size))
        ix1 = max(ix0 + 1, int(np.ceil(x1 * grid_size)))
        iy0 = int(np.floor(y0 * grid_size))
        iy1 = max(iy0 + 1, int(np.ceil(y1 * grid_size)))
        ix1 = min(ix1, grid_size)
        iy1 = min(iy1, grid_size)
        grid[iy0:iy1, ix0:ix1] = True
    return grid


def box_count(grid, box_size):
    """Count boxes of side `box_size` (must divide grid.shape[0]) that
    contain at least one True pixel."""
    n = grid.shape[0]
    assert n % box_size == 0, f"{box_size} does not divide grid size {n}"
    m = n // box_size
    reshaped = grid.reshape(m, box_size, m, box_size)
    occupied_boxes = reshaped.any(axis=(1, 3))
    return int(occupied_boxes.sum())


def box_counting_dimension(lattice, occ_ids, grid_size=4096,
                            fit_range=None):
    """Returns (box_sizes, eps, N_eps, d_f_fit, intercept).
    fit_range: (min_box, max_box) inclusive powers-of-two range to use in
    the linear fit (to exclude trivial extremes: box_size=1 pixel noise,
    box_size=grid_size trivial single box)."""
    grid = rasterize_occupied(lattice, occ_ids, grid_size)

    # box sizes = all power-of-two divisors of grid_size, excluding the
    # trivial single-box case (box_size == grid_size)
    box_sizes = []
    b = 1
    while b < grid_size:
        box_sizes.append(b)
        b *= 2
    box_sizes = np.array(box_sizes)

    N_eps = np.array([box_count(grid, b) for b in box_sizes])
    eps = box_sizes / grid_size

    if fit_range is None:
        # default: drop the finest 1-2 scales (pixelation noise) and the
        # coarsest 1-2 scales (too few boxes for reliable statistics)
        lo, hi = box_sizes[1], box_sizes[-3]
    else:
        lo, hi = fit_range
    mask = (box_sizes >= lo) & (box_sizes <= hi)

    log_inv_eps = np.log(1.0 / eps[mask])
    log_N = np.log(N_eps[mask])
    slope, intercept = np.polyfit(log_inv_eps, log_N, 1)
    d_f = slope  # N(eps) ~ eps^-d_f = (1/eps)^d_f

    return box_sizes, eps, N_eps, d_f, intercept, mask


def analytic_df(q):
    return 2.0 * (np.sqrt(3.0 + q) - 1.0)
