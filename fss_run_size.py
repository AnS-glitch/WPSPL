import sys
import os
import time
import numpy as np
from multiprocessing import Pool, cpu_count

from fss import spanning_curve_for_size, estimate_p_window

OUTPUT_DIR = "outputs/fss_checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIZES = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
Q_VALUES = [1.0, 0.95, 0.9, 0.85]  # List of q values to iterate through
N_LATTICES = 20
N_BOND_REALIZATIONS = 40
SEED = 321
REPRESENTATIVE_N_STEPS = 6000  # used only to auto-center the p-window for this q


def run_size(q, n_steps, pool=None):
    t0 = time.time()
    # Compute auto-centered p-window for the specific 'q'
    p_values, p_center = estimate_p_window(q, REPRESENTATIVE_N_STEPS, seed=SEED)
    print(f"[q={q}] auto-centered p-window: center~{p_center:.3f}, "
          f"range=[{p_values[0]:.3f}, {p_values[-1]:.3f}]")
    
    W_p, M, n_samples = spanning_curve_for_size(
        q, n_steps, N_LATTICES, N_BOND_REALIZATIONS, p_values, seed=SEED,
        pool=pool)
    dt = time.time() - t0
    
    # ✅ Unique output path per size and q value to avoid overwriting checkpoint files
    out_path = os.path.join(OUTPUT_DIR, f"size_{n_steps}_q{q}.npz")
    np.savez(out_path, p_values=p_values, W_p=W_p, M=M,
             n_samples=n_samples, n_steps=n_steps, q=q, time_taken=dt)
    
    print(f"[q={q}] t={n_steps}: M~{M}, {n_samples} samples, time={dt:.1f}s -> saved {out_path}")


if __name__ == "__main__":
    n_workers = min(N_LATTICES, cpu_count())
    print(f"Using a pool of {n_workers} worker processes "
          f"(cpu_count={cpu_count()}, N_LATTICES={N_LATTICES})")
    
    # ✅ Open Pool once outside both loops to reuse worker processes
    with Pool(n_workers) as pool:
        for q in Q_VALUES:
            print(f"\n=========================================")
            print(f" STARTING SIMULATION FOR q = {q}")
            print(f"=========================================")
            
            for size in SIZES:
                print(f"\n--- Running simulation for size: {size} (q={q}) ---")
                run_size(q, size, pool=pool)

    print("\nDone: All Q values and sizes processed.")
