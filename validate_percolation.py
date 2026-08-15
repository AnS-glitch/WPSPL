import os
import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from lattice import WPSPL
from adjacency import build_adjacency
from percolation import edges_from_adjacency, order_parameter_and_susceptibility

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

Q_VALUES = [1.0 , 0.95, 0.9, 0.85, 0.80, 0.75, 0.70]
N_STEPS = 100000
N_LATTICES = 20            # independent WPSPL realizations, averaged
N_BOND_REALIZATIONS = 40   # independent bond orderings per lattice, averaged
SEED = 123

# stage 1 (coarse) pre-scan: 1 lattice, few bond realizations, sparse full-range grid
COARSE_N_POINTS = 101
COARSE_BOND_REALIZATIONS = 3

# stage 2 (medium) refinement: a few lattices, moderate grid, around the coarse guess
MEDIUM_HALF_WIDTH = 0.20
MEDIUM_N_POINTS = 250
MEDIUM_LATTICES = 3
MEDIUM_BOND_REALIZATIONS = 5

# stage 3 (final) refined grid: built around the medium-stage peak estimate
N_FINE_POINTS = 1500
FINE_HALF_WIDTH = 0.08      # dense window = [p_c_guess - w, p_c_guess + w]
N_TAIL_POINTS = 40          # sparse coverage outside the dense window


def build_adaptive_grid(p_c_guess, half_width, n_fine, n_tail=N_TAIL_POINTS):
    lo = max(0.0, p_c_guess - half_width)
    hi = min(1.0, p_c_guess + half_width)
    fine = np.linspace(lo, hi, n_fine)
    left = np.linspace(0.0, lo, n_tail, endpoint=False) if lo > 0 else np.array([])
    right = np.linspace(hi, 1.0, n_tail) if hi < 1.0 else np.array([])
    grid = np.unique(np.concatenate([left, fine, right]))
    return grid


def coarse_scan(q):
    """Stage 1: cheap single-lattice, sparse full-range grid scan to get a
    rough first guess of the transition location."""
    lat = WPSPL(q=q, n_steps=N_STEPS, seed=SEED).generate()
    occ = lat.occupied_ids()
    adj = build_adjacency(lat, occ)
    n_nodes, edges, _ = edges_from_adjacency(occ, adj)
    coarse_p = np.linspace(0.0, 1.0, COARSE_N_POINTS)
    _, chi_coarse = order_parameter_and_susceptibility(
        n_nodes, edges, coarse_p, n_realizations=COARSE_BOND_REALIZATIONS,
        seed=SEED + 999)
    p_c_guess = coarse_p[np.argmax(chi_coarse)]
    return p_c_guess


def medium_scan(q, p_c_guess, pool):
    """Stage 2: a few lattices, moderate-density grid within a wide window
    around the coarse guess, to pin down the peak location reliably before
    committing to the expensive final dense pass. Needed because a single
    coarse scan is noisy enough (few realizations, sparse grid) that the
    tight final window could otherwise miss the true peak, especially
    since p_c shifts substantially with q."""
    grid = build_adaptive_grid(p_c_guess, MEDIUM_HALF_WIDTH, MEDIUM_N_POINTS,
                                n_tail=20)
    args = [(q, r, grid, MEDIUM_BOND_REALIZATIONS, SEED + 500000 + r)
            for r in range(MEDIUM_LATTICES)]
    results = pool.map(_simulate_worker, args)
    chi_sum = np.zeros(len(grid))
    for _, chi in results:
        chi_sum += chi
    chi_avg = chi_sum / MEDIUM_LATTICES
    return grid[np.argmax(chi_avg)]


def _simulate_worker(args):
    q, r, p_values, n_bond_realizations, bond_seed = args
    lat = WPSPL(q=q, n_steps=N_STEPS, seed=SEED + r).generate()
    occ = lat.occupied_ids()
    adj = build_adjacency(lat, occ)
    n_nodes, edges, _ = edges_from_adjacency(occ, adj)

    P, chi = order_parameter_and_susceptibility(
        n_nodes, edges, p_values,
        n_realizations=n_bond_realizations, seed=bond_seed)
    return P, chi


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.plasma(np.linspace(0.15, 0.75, len(Q_VALUES)))

    with Pool() as pool:
        for q, c in zip(Q_VALUES, colors):
            t0 = time.time()

            # stage 1: cheap coarse scan, full range
            p_c_coarse = coarse_scan(q)
            print(f"q={q}: stage1 (coarse) p_c guess={p_c_coarse:.3f}")

            # stage 2: moderate refinement around the coarse guess
            p_c_guess = medium_scan(q, p_c_coarse, pool)
            print(f"q={q}: stage2 (medium) p_c guess={p_c_guess:.3f}")

            # stage 3: final dense window around the refined guess
            p_values = build_adaptive_grid(p_c_guess, FINE_HALF_WIDTH, N_FINE_POINTS)
            print(f"q={q}: stage3 (final) grid has {len(p_values)} points "
                  f"(dense in [{max(0,p_c_guess-FINE_HALF_WIDTH):.3f}, "
                  f"{min(1,p_c_guess+FINE_HALF_WIDTH):.3f}])")

            args = [(q, r, p_values, N_BOND_REALIZATIONS, SEED + 100000 + r)
                    for r in range(N_LATTICES)]
            results = pool.map(_simulate_worker, args)

            P_sum = np.zeros(len(p_values))
            chi_sum = np.zeros(len(p_values))
            for P, chi in results:
                P_sum += P
                chi_sum += chi
            P_p = P_sum / N_LATTICES
            chi_p = chi_sum / N_LATTICES

            dt = time.time() - t0
            peak_idx = np.argmax(chi_p)
            p_c_est = p_values[peak_idx]
            print(f"q={q}: time={dt:.1f}s, refined chi-peak p_c~={p_c_est:.4f}")

            ax1.plot(p_values, P_p, "-", color=c, label=f"q={q}", lw=1.2)
            ax2.plot(p_values, chi_p, "-", color=c,
                      label=f"q={q} (peak~{p_c_est:.3f})", lw=1.2)

    ax1.set_xlabel("bond occupation probability $p$")
    ax1.set_ylabel("$P(p)$")
    ax1.set_title("Order parameter")
    ax1.legend()
    ax2.set_xlabel("bond occupation probability $p$")
    ax2.set_ylabel(r"$\chi(p)$")
    ax2.set_title("Susceptibility analog")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_percolation_Pp_chip.png"),
                dpi=1200, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_percolation_Pp_chip.pdf"),
                bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
    print("Done")
