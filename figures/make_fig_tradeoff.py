"""Figure 1: F_G, Pi, n_b versus G/G_th, and the (Pi, F_G) plane with the
analytic asymptote F = Pi^2 / C_Pi^2 for the blue sideband (no fit).  The red
sideband sweep is extended to 20 G_th to show the saturation of Pi_red."""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov

OUT = os.path.dirname(os.path.abspath(__file__))
kappa, gamma, nth = 0.1, 1e-4, 100.0
G_th = np.sqrt(kappa * gamma) / 2.0
r = gamma / kappa
C_Pi = gamma**1.5 * np.sqrt(kappa) * (nth + 1)**2 / ((1 + r) * (2 * nth + 1))
C_ratio = 1.0 / C_Pi**2
print(f"C_Pi    = {C_Pi:.6e}")
print(f"1/C_Pi^2 = {C_ratio:.6e}")

OMEGA = np.zeros((4, 4))
OMEGA[0, 1] = 1.0; OMEGA[1, 0] = -1.0
OMEGA[2, 3] = 1.0; OMEGA[3, 2] = -1.0

def A_full(G, side):
    s = 1.0 if side == "red" else -1.0
    A = np.zeros((4, 4))
    A[0, 3] = s * G; A[1, 2] = -G; A[2, 1] = s * G; A[3, 0] = -G
    A[0, 0] -= kappa / 2; A[1, 1] -= kappa / 2
    A[2, 2] -= gamma / 2; A[3, 3] -= gamma / 2
    return A

D = np.diag([kappa / 2, kappa / 2, gamma * (nth + 0.5), gamma * (nth + 0.5)])
A_IRR = np.diag([-kappa / 2, -kappa / 2, -gamma / 2, -gamma / 2])

def qfi_gaussian(sigma, dsigma):
    M = np.kron(sigma, sigma) - np.kron(OMEGA, OMEGA) / 4.0
    v = dsigma.flatten(order="F")
    return 0.5 * v @ (np.linalg.pinv(M, rcond=1e-13, hermitian=True) @ v)

def entropy_production(sigma):
    Dinv = np.linalg.inv(D)
    return (2.0 * np.trace(A_IRR)
            + 2.0 * np.trace(A_IRR.T @ Dinv @ A_IRR @ sigma)
            + 0.5 * np.trace(np.linalg.inv(sigma) @ D))

# load the existing sweep
rows = []
with open(os.path.join(OUT, "sweep_data.csv"), newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append((row["side"], float(row["G"]), float(row["QFI"]),
                     float(row["Pi_entropy_production"])))

data = {"blue": [], "red": []}
for side, G, F, Pi in rows:
    data[side].append((G, F, Pi))
for side in data:
    data[side].sort()

# extend the red sideband to 20 G_th
G_red_extra = G_th * np.geomspace(0.995, 20.0, 30)
for G in G_red_extra:
    A = A_full(G, "red")
    sig = 0.5 * (solve_continuous_lyapunov(A, -D)
                 + solve_continuous_lyapunov(A, -D).T)
    h = 1e-6 * max(G, 1e-3) * G_th
    sp = solve_continuous_lyapunov(A_full(G + h, "red"), -D)
    sm = solve_continuous_lyapunov(A_full(G - h, "red"), -D)
    dsig = 0.5 * ((sp - sm) / (2 * h) + ((sp - sm) / (2 * h)).T)
    data["red"].append((G, qfi_gaussian(sig, dsig), entropy_production(sig)))
data["red"].sort()

Gb = np.array([p[0] for p in data["blue"]])
Fb = np.array([p[1] for p in data["blue"]])
Pb = np.array([p[2] for p in data["blue"]])
Gr = np.array([p[0] for p in data["red"]])
Fr = np.array([p[1] for p in data["red"]])
Pr = np.array([p[2] for p in data["red"]])
print(f"red extended: G/G_th {Gr.min()/G_th:.3f}..{Gr.max()/G_th:.2f}, "
      f"Pi {Pr.min():.2e}..{Pr.max():.2e}, F {Fr.min():.2e}..{Fr.max():.2e}")

# figure
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6, "axes.grid": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "figure.dpi": 300,
})
TW = 5.46
fig, axes = plt.subplots(2, 1, figsize=(TW, 5.5))

# (a)
ax = axes[0]
ax.text(0.0, 1.13, "(a)", transform=ax.transAxes, va="bottom", ha="left")
xb = Gb / G_th
l1, = ax.semilogy(xb, Fb, "-", color="#0000FF", label=r"$F_G$ (QFI)")
l2, = ax.semilogy(xb, Pb, "--", color="#FF0000",
                  label=r"$\Pi$ (entropy production)")
ax.axvline(1.0, color="k", ls="-.", lw=1.0)
ax.text(0.965, 0.06, r"$G_{\mathrm{th}}$", transform=ax.transAxes,
        ha="right", va="bottom")
ax.set_xlabel(r"$G/G_{\mathrm{th}}$")
ax.set_ylabel("value")
# n_b recomputed from the CSV columns (kept identical to the reviewed version)
n_b_vals = []
for side, G, F, Pi in rows:
    if side != "blue":
        continue
    A = A_full(G, "blue")
    sig = solve_continuous_lyapunov(A, -D)
    n_b_vals.append(0.5 * (sig[2, 2] + sig[3, 3]) - 0.5)
ax.semilogy(xb, n_b_vals, ":", color="#00FF00",
            label=r"$n_b$ (mechanical occupation)")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.08),
          ncol=3, frameon=False, borderaxespad=0)

# (b)
ax = axes[1]
ax.text(0.0, 1.05, "(b)", transform=ax.transAxes, va="bottom", ha="left")
ax.loglog(Pb, Fb, "o", ms=3.6, mfc="none", color="#0000FF",
          label="blue sideband (su(1,1))")
ax.loglog(Pr, Fr, "s", ms=3.6, mfc="none", color="#FF8000",
          label="red sideband (beam splitter)")
Piw = np.array([1.7e-2, 0.75])
ax.loglog(Piw, C_ratio * Piw**2, "--", color="k", lw=1.2,
          label=r"analytic asymptote $F_G=\Pi^{2}/C_{\Pi}^{2}$")
ax.set_xlabel(r"entropy production rate $\Pi$")
ax.set_ylabel(r"quantum Fisher information $F_G$")
ax.legend(loc="upper left", framealpha=1.0, frameon=True, handlelength=1.6)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_tradeoff.png"), dpi=300)
fig.savefig(os.path.join(OUT, "fig_tradeoff.pdf"))
print("saved fig_tradeoff.png / .pdf")
