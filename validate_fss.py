import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt

from fss import spanning_curve_for_size, system_size_L, find_W_crossing_p, robust_p_c_estimate, estimate_p_window

OUTPUT_DIR = "outputs"

# ✅ Changed to a list of Q values
Q_VALUES = [1.0, 0.95, 0.9, 0.85] 
SIZES = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]   # n_steps values (t)
N_LATTICES = 20
N_BOND_REALIZATIONS = 40
SEED = 321
REPRESENTATIVE_N_STEPS = 6000  # used only to auto-center the p-window for this q


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_workers = min(N_LATTICES, cpu_count())
    print(f"Using a pool of {n_workers} worker processes "
          f"(cpu_count={cpu_count()}, N_LATTICES={N_LATTICES})")
    print(f"{N_LATTICES} lattices x {N_BOND_REALIZATIONS} bond realizations "
          f"= {N_LATTICES*N_BOND_REALIZATIONS} samples per size.")

    # ✅ Open Pool once outside to reuse workers for all Q values
    with Pool(n_workers) as pool:
        for q in Q_VALUES:
            print(f"\n=========================================")
            print(f" STARTING SIMULATION FOR q = {q}")
            print(f"=========================================")

            # Auto-center window dynamically for the current q value
            p_values, p_center = estimate_p_window(q, REPRESENTATIVE_N_STEPS, seed=SEED)
            print(f"auto-centered p-window for q={q}: center~{p_center:.3f}, "
                  f"range=[{p_values[0]:.3f}, {p_values[-1]:.3f}]")

            W_curves = {}
            L_values = []
            
            for n_steps in SIZES:
                t0 = time.time()
                # Pass current 'q' and the shared 'pool'
                W_p, M, n_samples = spanning_curve_for_size(
                    q, n_steps, N_LATTICES, N_BOND_REALIZATIONS, p_values,
                    seed=SEED, pool=pool)
                L = system_size_L(n_steps)
                L_values.append(L)
                W_curves[n_steps] = W_p
                print(f"t={n_steps} (L={L:.1f}): M~{M}, {n_samples} samples, "
                      f"time={time.time()-t0:.1f}s")

            L_values = np.array(L_values)

            # ---------------------------------------------------------------
            # Figure: W(p,L) for each system size
            # ---------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(SIZES)))
            for n_steps, c in zip(SIZES, colors):
                ax.plot(p_values, W_curves[n_steps], "-", color=c,
                        label=f"t={n_steps} (L={system_size_L(n_steps):.0f})")
            ax.set_xlabel("bond occupation probability $p$")
            ax.set_ylabel("$W(p,L)$ (spanning probability)")
            ax.set_title(f"Spanning probability vs $p$, q={q}")
            ax.legend(fontsize=8)
            plt.tight_layout()
            # ✅ Appended _q{q} to filenames to prevent overwriting
            plt.savefig(os.path.join(OUTPUT_DIR, f"fig6_spanning_probability_q{q}.png"), dpi=1200)
            plt.savefig(os.path.join(OUTPUT_DIR, f"fig6_spanning_probability_q{q}.pdf"))
            plt.close()

            # ---------------------------------------------------------------
            # p_c estimate: robust aggregation of pairwise curve crossings
            # ---------------------------------------------------------------
            p_c_est, crossings, p_half = robust_p_c_estimate(p_values, W_curves, SIZES)
            print(f"q={q} pairwise crossings used: {crossings}")
            print(f"q={q} p_c estimate = {p_c_est}")

            # ---------------------------------------------------------------
            # 1/nu: p(L) at W=0.5 for each L, fit ln(p_c - p) vs ln(L)
            # ---------------------------------------------------------------
            print(f"q={q} p(L) at W=0.5:", dict(zip(SIZES, p_half)))
            valid = ~np.isnan(p_half)

            if p_c_est is not None and valid.sum() >= 2:
                diffs = np.abs(p_c_est - p_half)
                ok = valid & (diffs > 1e-6)
                if ok.sum() >= 2:
                    x = np.log(L_values[ok])
                    y = np.log(diffs[ok])
                    slope, intercept = np.polyfit(x, y, 1)
                    inv_nu = -slope
                    print(f"Fitted 1/nu = {inv_nu:.4f}  (slope of ln(p_c-p) vs ln(L))")

                    fig, ax = plt.subplots(figsize=(5.5, 5))
                    ax.plot(x, y, "o", color="crimson")
                    ax.plot(x, slope * x + intercept, "-", color="black",
                            label=f"fit: 1/nu={inv_nu:.3f}")
                    ax.set_xlabel("ln(L)")
                    ax.set_ylabel("ln|p_c - p(L)|")
                    ax.set_title(f"Correlation-length exponent fit, q={q}")
                    ax.legend()
                    plt.tight_layout()
                    # ✅ Appended _q{q} to filenames
                    plt.savefig(os.path.join(OUTPUT_DIR, f"fig7_nu_fit_q{q}.png"), dpi=1200)
                    plt.savefig(os.path.join(OUTPUT_DIR, f"fig7_nu_fit_q{q}.pdf"))
                    plt.close()

                    # ---------------------------------------------------------------
                    # Collapse: W vs (p - p_c) * L^(1/nu)
                    # ---------------------------------------------------------------
                    fig, ax = plt.subplots(figsize=(6, 5))
                    for n_steps, c in zip(SIZES, colors):
                        L = system_size_L(n_steps)
                        x_scaled = (p_values - p_c_est) * L ** inv_nu
                        ax.plot(x_scaled, W_curves[n_steps], "-", color=c,
                                label=f"t={n_steps}")
                    ax.set_xlabel(r"$(p-p_c)\,L^{1/\nu}$")
                    ax.set_ylabel("$W(p,L)$")
                    ax.set_title(f"Scaling collapse, q={q}, p_c~{p_c_est:.3f}, 1/nu~{inv_nu:.3f}")
                    ax.legend(fontsize=8)
                    plt.tight_layout()
                    # ✅ Appended _q{q} to filenames
                    plt.savefig(os.path.join(OUTPUT_DIR, f"fig8_collapse_q{q}.png"), dpi=1200)
                    plt.savefig(os.path.join(OUTPUT_DIR, f"fig8_collapse_q{q}.pdf"))
                    plt.close()
                else:
                    print(f"q={q}: Not enough valid points with positive (p_c - p) for a fit.")
            else:
                print(f"q={q}: Could not estimate p_c or insufficient valid W=0.5 crossings.")

    print("done All Q values processed successfully.")


if __name__ == "__main__":
    main()
