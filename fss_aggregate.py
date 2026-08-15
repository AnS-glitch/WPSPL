import os
import numpy as np
import matplotlib.pyplot as plt

from fss import system_size_L, robust_p_c_estimate

OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "outputs/fss_checkpoints"

# Configuration
Q_VALUES = [1.0, 0.95, 0.9, 0.85]
SIZES = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Outer loop across all Q values
for q in Q_VALUES:
    print(f"\n=========================================")
    print(f" LOADING CHECKPOINTS & ANALYZING q = {q}")
    print(f"=========================================")

    data = {}
    p_values = None
    
    # Load .npz checkpoints for the current Q value
    for n_steps in SIZES:
        file_path = os.path.join(CHECKPOINT_DIR, f"size_{n_steps}_q{q}.npz")
        
        if not os.path.exists(file_path):
            print(f"Warning: Checkpoint missing: {file_path}. Skipping.")
            continue
            
        d = np.load(file_path)
        if p_values is None:
            p_values = d["p_values"]
            
        data[n_steps] = d["W_p"]
        print(f"t={n_steps}: M~{int(d['M'])}, samples={int(d['n_samples'])}, "
              f"time={float(d['time_taken']):.1f}s")

    # Skip plotting if data is missing or incomplete for this Q
    if p_values is None or len(data) < len(SIZES):
        print(f"Skipping Q={q} analysis due to incomplete checkpoint files.")
        continue

    L_values = np.array([system_size_L(n) for n in SIZES])

    # ---------------------------------------------------------------
    # Figure: W(p,L)
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(SIZES)))
    for n_steps, c in zip(SIZES, colors):
        ax.plot(p_values, data[n_steps], "-", color=c,
                label=f"t={n_steps} (L={system_size_L(n_steps):.0f})")
    ax.set_xlabel("bond occupation probability $p$")
    ax.set_ylabel("$W(p,L)$ (spanning probability)")
    ax.set_title(f"Spanning probability vs $p$, q={q} (300 samples/size)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"fig6_spanning_probability_q{q}.png"), dpi=1200)
    plt.savefig(os.path.join(OUTPUT_DIR, f"fig6_spanning_probability_q{q}.pdf"))
    plt.close()

    # ---------------------------------------------------------------
    # p_c via robust aggregation of pairwise crossings
    # ---------------------------------------------------------------
    p_c_est, crossings, p_half = robust_p_c_estimate(p_values, data, SIZES)
    print(f"q={q} pairwise crossings used: {crossings}")
    print(f"q={q} p_c estimate = {p_c_est}")

    # ---------------------------------------------------------------
    # 1/nu fit
    # ---------------------------------------------------------------
    print(f"q={q} p(L) at W=0.5:", dict(zip(SIZES, p_half)))

    if p_c_est is not None:
        diffs = np.abs(p_c_est - p_half)
        ok = diffs > 1e-6
        print(f"q={q} |p_c - p(L)|:", diffs)
        if ok.sum() >= 2:
            x = np.log(L_values[ok])
            y = np.log(diffs[ok])
            slope, intercept = np.polyfit(x, y, 1)
            inv_nu = -slope
            print(f"Fitted 1/nu = {inv_nu:.4f}")

            # Figure 7: Nu Fit
            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.plot(x, y, "o", color="crimson")
            ax.plot(x, slope * x + intercept, "-", color="black",
                    label=f"fit: 1/nu={inv_nu:.3f}")
            ax.set_xlabel("ln(L)")
            ax.set_ylabel("ln|p_c - p(L)|")
            ax.set_title(f"Correlation-length exponent fit, q={q}")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"fig7_nu_fit_q{q}.png"), dpi=1200)
            plt.savefig(os.path.join(OUTPUT_DIR, f"fig7_nu_fit_q{q}.pdf"))
            plt.close()

            # Figure 8: Scaling Collapse
            fig, ax = plt.subplots(figsize=(6, 5))
            for n_steps, c in zip(SIZES, colors):
                L = system_size_L(n_steps)
                x_scaled = (p_values - p_c_est) * L ** inv_nu
                ax.plot(x_scaled, data[n_steps], "-", color=c, label=f"t={n_steps}")
            ax.set_xlabel(r"$(p-p_c)\,L^{1/\nu}$")
            ax.set_ylabel("$W(p,L)$")
            ax.set_title(f"Scaling collapse, q={q}, p_c~{p_c_est:.3f}, 1/nu~{inv_nu:.3f}")
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"fig8_collapse_q{q}.png"), dpi=1200)
            plt.savefig(os.path.join(OUTPUT_DIR, f"fig8_collapse_q{q}.pdf"))
            plt.close()
        else:
            print(f"q={q}: Not enough points with p_c > p(L) for a fit.")

print("\nDone: Generated plots for all Q values.")
