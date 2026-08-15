#Run this file to create a Lattice
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from lattice import WPSPL
from adjacency import build_adjacency

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

Q = 0.85
N_STEPS = 20000
SEED = 42

lat = WPSPL(q=Q, n_steps=N_STEPS, seed=SEED).generate()
occ = lat.occupied_ids()
void = lat.void_ids()

print(f"q={Q}, t={N_STEPS}: splits={lat.n_splits}, noop={lat.n_noop}, "
      f"occupied={len(occ)}, void={len(void)}")
print(f"total leaf area (should be 1.0): "
      f"{lat.area[occ].sum() + lat.area[void].sum():.8f}")
print(f"occupied area fraction M = {lat.area[occ].sum():.4f} "
      f"(porosity = {1 - lat.area[occ].sum():.4f})")

# Figure 1: lattice tiling (analog of paper's Fig. 1)

fig, ax = plt.subplots(figsize=(7, 7))

occ_patches = []
for i in occ:
    x0, y0, x1, y1 = lat.rects[i]
    occ_patches.append(Rectangle((x0, y0), x1 - x0, y1 - y0))
occ_coll = PatchCollection(occ_patches, facecolor="white",
                            edgecolor="steelblue", linewidth=0.4)
ax.add_collection(occ_coll)

void_patches = []
for i in void:
    x0, y0, x1, y1 = lat.rects[i]
    void_patches.append(Rectangle((x0, y0), x1 - x0, y1 - y0))
void_coll = PatchCollection(void_patches, facecolor="0.55",
                             edgecolor="steelblue", linewidth=0.4)
ax.add_collection(void_coll)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_title(f"WPSPL snapshot: q={Q}, t={N_STEPS}\n"
             f"(gray = pores, {len(void)} voids / {len(occ)} occupied blocks)")
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_lattice_tiling.png"), dpi=200)
plt.close()


# Figure 2: coordination-number (degree) distribution of the dual graph

adj = build_adjacency(lat, occ)
degrees = np.array([len(adj.get(i, ())) for i in occ])

k_vals, counts = np.unique(degrees, return_counts=True)
P_k = counts / counts.sum()

# log-log power-law fit P(k) ~ k^-gamma over the well-sampled range
mask = (k_vals >= 2) & (counts >= 5)
logk = np.log(k_vals[mask])
logP = np.log(P_k[mask])
slope, intercept = np.polyfit(logk, logP, 1)
gamma = -slope

fig, ax = plt.subplots(figsize=(6, 5))
ax.loglog(k_vals, P_k, "o", color="steelblue", ms=5, label="data")
kk = np.linspace(k_vals[mask].min(), k_vals[mask].max(), 50)
ax.loglog(kk, np.exp(intercept) * kk ** slope, "r--",
          label=fr"fit: $P(k)\sim k^{{-{gamma:.2f}}}$")
ax.set_xlabel("coordination number $k$")
ax.set_ylabel("$P(k)$")
ax.set_title(f"Coordination-number distribution (q={Q}, t={N_STEPS}, N={len(occ)})")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_coordination_number_distribution.png"), dpi=200)
plt.close()

print(f"Fitted power-law exponent gamma = {gamma:.3f}  "
      f"(paper reports gamma=5.66 for the non-porous q=1 WPSL2 case)")
print(f"mean degree = {degrees.mean():.3f}, min={degrees.min()}, max={degrees.max()}")
