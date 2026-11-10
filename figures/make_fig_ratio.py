# -*- coding: utf-8 -*-
"""
Convergence figure for the analytic layer: the dimensionless combinations
  F * delta^2  ->  1                       (universal, parameter-free)
  Pi * delta   ->  C_Pi = gamma^{3/2} sqrt(kappa) (n+1)^2 / ((1+r)(2n+1))
  F / Pi^2     ->  (1+r)^2 (2n+1)^2 / (kappa gamma^3 (n+1)^4)
plotted against G/G_th from the sweep data (sweep_data.csv).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.path.dirname(os.path.abspath(__file__))
kappa, gamma, nth = 0.1, 1e-4, 100.0
G_th = np.sqrt(kappa * gamma) / 2.0
r = gamma / kappa

C_Pi = gamma**1.5 * np.sqrt(kappa) * (nth + 1)**2 / ((1 + r) * (2 * nth + 1))
C_ratio = (1 + r)**2 * (2 * nth + 1)**2 / (kappa * gamma**3 * (nth + 1)**4)
print(f"C_Pi    = {C_Pi:.6e}")
print(f"C_ratio = {C_ratio:.6e}")

d = np.genfromtxt(os.path.join(OUT, "sweep_data.csv"), delimiter=",",
                  names=True, dtype=None, encoding="utf-8")
b = d[d["side"] == "blue"]
G, F, Pi = b["G"], b["QFI"], b["Pi_entropy_production"]
delta = G_th - G
x = G / G_th

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6, "axes.grid": False,
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "figure.dpi": 300,
})
TW = 5.46                      # = \textwidth of the MDPI Entropy layout (inch)
fig, axes = plt.subplots(2, 1, figsize=(TW, 5.5))

# (a) F*delta^2 -> 1 and Pi*delta -> C_Pi
ax = axes[0]
ax.text(0.0, 1.03, "(a)", transform=ax.transAxes, va="bottom", ha="left")
ax.loglog(x[1:], (F * delta**2)[1:], "o", ms=3.2, mfc="none",
          color="#0000FF", label=r"$F_G\,(G_{\mathrm{th}}-G)^2$")
ax.axhline(1.0, color="#000000", ls="--", lw=1.0,
           label="asymptotic value 1")
ax.loglog(x[1:], (Pi * delta)[1:] / C_Pi, "s", ms=3.2, mfc="none",
          color="#FF0000",
          label=r"$\Pi\,(G_{\mathrm{th}}-G)/C_\Pi$")
ax.set_xlabel(r"$G/G_{\mathrm{th}}$")
ax.set_ylabel("rescaled combination")
ax.legend(loc="lower right", framealpha=1.0, frameon=True)

# (b) F/Pi^2 -> closed form
ax = axes[1]
ax.text(0.0, 1.03, "(b)", transform=ax.transAxes, va="bottom", ha="left")
ax.loglog(x[1:], (F / Pi**2)[1:], "o", ms=3.2, mfc="none",
          color="#8000FF", label=r"$F_G/\Pi^2$ (blue sideband)")
ax.axhline(C_ratio, color="#000000", ls="--", lw=1.2,
           label="analytic constant")
ax.set_xlabel(r"$G/G_{\mathrm{th}}$")
ax.set_ylabel(r"$F_G/\Pi^2$")
ax.legend(loc="upper right", framealpha=1.0, frameon=True)

fig.tight_layout()
figpath = os.path.join(OUT, "fig_ratio.png")
fig.savefig(figpath, dpi=300)
fig.savefig(os.path.join(OUT, "fig_ratio.pdf"))   # vector version for the paper
print("figure saved:", figpath)

# report the C_F correction coefficient from the near-threshold window
# (the universal limit F*delta^2 -> 1 is approached only as G -> G_th)
m = slice(-12, None)
cf = (F * delta**2)[m]
print("C_F range near threshold:", cf.min(), "..", cf.max())
# linear fit of 1 - C_F vs eps, restricted to the near-threshold window
eps = delta / G_th
c = np.polyfit(eps[m], 1 - cf, 1)
print(f"1 - C_F ~ {c[0]:.4f} * eps  (linear-in-eps correction, no logs)")
