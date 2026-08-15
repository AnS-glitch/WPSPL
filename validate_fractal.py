import os
import numpy as np
import matplotlib.pyplot as plt

from lattice import WPSPL
from fractal_dimension import box_counting_dimension, analytic_df

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

Q_VALUES = [1.00, 0.95, 0.9, 0.85, 0.80, 0.75, 0.70]
N_STEPS = 250000
GRID_SIZE = 4096
SEED = 11

results = []
box_curves = {}

for q in Q_VALUES:
    lat = WPSPL(q=q, n_steps=N_STEPS, seed=SEED).generate()
    occ = lat.occupied_ids()
    bs, eps, N_eps, df, intercept, mask = box_counting_dimension(
        lat, occ, grid_size=GRID_SIZE)
    porosity = 1.0 - lat.area[occ].sum()
    results.append((q, df, analytic_df(q), porosity, len(occ)))
    box_curves[q] = (eps, N_eps, df, intercept, mask)
    print(f"q={q}: N_occ={len(occ)}, porosity={porosity:.3f}, "
          f"measured d_f={df:.3f}, analytic d_f={analytic_df(q):.3f}")

# ---------------------------------------------------------------
# Figure 1: box-counting log-log curves with fitted slopes
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5.5))
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(Q_VALUES)))
for (q, c) in zip(Q_VALUES, colors):
    eps, N_eps, df, intercept, mask = box_curves[q]
    ax.loglog(1.0 / eps, N_eps, "o", color=c, ms=4,
              label=f"q={q} (fit d_f={df:.3f})")
    x_fit = 1.0 / eps[mask]
    ax.loglog(x_fit, np.exp(intercept) * x_fit ** df, "--", color=c, lw=1.2)

ax.set_xlabel(r"$1/\epsilon$ (inverse box size)")
ax.set_ylabel(r"$N(\epsilon)$ (occupied box count)")
ax.set_title("Box-counting dimension of occupied WPSPL region")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_box_counting.svg"))
plt.close()

# ---------------------------------------------------------------
# Figure 2: measured d_f(q) vs analytic formula d_f(q) = 2(sqrt(3+q)-1)
# ---------------------------------------------------------------
q_fine = np.linspace(0.5, 1.0, 200)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(q_fine, analytic_df(q_fine), "-", color="black",
        label=r"analytic: $d_f(q)=2(\sqrt{3+q}-1)$")
qs = [r[0] for r in results]
dfs_meas = [r[1] for r in results]
ax.plot(qs, dfs_meas, "o", color="crimson", ms=7, label="measured (box counting)")
ax.set_xlabel("porosity parameter $q$")
ax.set_ylabel(r"fractal dimension $d_f$")
ax.set_title(f"Fractal dimension vs $q$  (t={N_STEPS} steps)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_df_vs_q.svg"))
plt.close()

print("\nSummary:")
print(f"{'q':>6} {'N_occ':>10} {'porosity':>10} {'d_f measured':>14} {'d_f analytic':>14} {'diff':>8}")
for q, dfm, dfa, por, nocc in results:
    print(f"{q:6.2f} {nocc:10d} {por:10.3f} {dfm:14.3f} {dfa:14.3f} {dfm-dfa:8.3f}")
