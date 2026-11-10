"""
Generate sweep_data.csv: steady-state QFI, entropy production and occupations
over G in [0.02, 0.985] G_th for both sidebands (kappa=0.1, gamma=1e-4,
n_th=100).  The figures are drawn from this CSV by make_fig_tradeoff.py and
make_fig_ratio.py.
"""

import numpy as np
from scipy.linalg import solve_continuous_lyapunov, expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# model building blocks
# quadrature ordering: u = (x_d, p_d, x_b, p_b)
OMEGA = np.zeros((4, 4))
OMEGA[0, 1] = 1.0; OMEGA[1, 0] = -1.0   # mode d
OMEGA[2, 3] = 1.0; OMEGA[3, 2] = -1.0   # mode b

def A_H(G, side):
    """Hamiltonian (symplectic) part of the drift matrix."""
    A = np.zeros((4, 4))
    s = 1.0 if side == "red" else -1.0
    # blue: xdot_d = -G p_b ; red: xdot_d = +G p_b
    A[0, 3] = s * G
    # both: pdot_d = -G x_b
    A[1, 2] = -G
    # blue: xdot_b = -G p_d ; red: +G p_d
    A[2, 1] = s * G
    # both: pdot_b = -G x_d
    A[3, 0] = -G
    return A

def A_full(G, side, kappa, gamma):
    A = A_H(G, side).copy()
    A[0, 0] -= kappa / 2
    A[1, 1] -= kappa / 2
    A[2, 2] -= gamma / 2
    A[3, 3] -= gamma / 2
    return A

def D_mat(kappa, gamma, nth):
    return np.diag([kappa / 2, kappa / 2,
                    gamma * (nth + 0.5), gamma * (nth + 0.5)])

A_IRR = None  # set per (kappa, gamma)

def entropy_production(sigma, D):
    """Unconditional entropy production rate (arXiv:1908.09382, Supp. Mat.)."""
    Airr = A_IRR
    Dinv = np.linalg.inv(D)
    t1 = 2.0 * np.trace(Airr)
    t2 = 2.0 * np.trace(Airr.T @ Dinv @ Airr @ sigma)
    t3 = 0.5 * np.trace(np.linalg.inv(sigma) @ D)
    return t1 + t2 + t3

def qfi_gaussian(sigma, dsigma):
    """Banchi-Braunstein-Pirandola Gaussian QFI (zero mean)."""
    M = np.kron(sigma, sigma) - np.kron(OMEGA, OMEGA) / 4.0
    v = dsigma.flatten(order="F")
    Minv_v = np.linalg.pinv(M, rcond=1e-13, hermitian=True) @ v
    return 0.5 * v @ Minv_v

def sigma_of_t(sigma0, A, D, t):
    """Exact sigma(t) for the linear Lyapunov dynamics via matrix exponential."""
    K = np.kron(np.eye(4), A) + np.kron(A, np.eye(4))  # column-major vec
    d = D.flatten(order="F")
    aug = np.zeros((17, 17))
    aug[:16, :16] = K
    aug[:16, 16] = d
    E = expm(aug * t)
    v = E[:16, :16] @ sigma0.flatten(order="F") + E[:16, 16]
    return v.reshape((4, 4), order="F")

def symplectic_evals(sigma):
    ev = np.linalg.eigvals(1j * OMEGA @ sigma)
    return np.sort(ev.real)

def von_neumann_entropy(sigma):
    """S = sum_i g(nu_i), nu_i = symplectic eigenvalues (hbar=1)."""
    def g(nu):
        if nu < 0.5 - 1e-12:
            nu = 0.5  # numerical floor
        return (nu + 0.5) * np.log(nu + 0.5) - (nu - 0.5) * np.log(nu - 0.5)
    return sum(g(nu) for nu in symplectic_evals(sigma))

def var_X(sigma):
    """<X^2> for X = q_d q_b - p_d p_b in the zero-mean Gaussian state sigma.

    Quantum Wick theorem with the OPERATOR ordering respected:
      <x1 x2 x3 x4> = A12 A34 + A13 A24 + A14 A23,   A_ij = sigma_ij + (i/2) Omega_ij.
    (Using sigma alone drops the c-number commutator terms and is WRONG here.)
    """
    s = sigma  # indices: 0=x_d,1=p_d,2=x_b,3=p_b
    Amat = s + 0.5j * OMEGA

    def W(i1, i2, i3, i4):
        return (Amat[i1, i2] * Amat[i3, i4]
                + Amat[i1, i3] * Amat[i2, i4]
                + Amat[i1, i4] * Amat[i2, i3])

    # X^2 = q_d q_b q_d q_b - q_d q_b p_d p_b - p_d p_b q_d q_b + p_d p_b p_d p_b
    val = (W(0, 2, 0, 2) - W(0, 2, 1, 3) - W(1, 3, 0, 2) + W(1, 3, 1, 3))
    return val.real  # <X> = 0 identically (X generates the evolution itself)

# validation 1: pure-state unitary family vs the QFI formula
# kappa, gamma -> 0 : state = TMS squeezed vacuum, F = 4 t^2 Var(X(t))
print("Validation 1: pure-state unitary family  F =? 4 t^2 Var(X)")
kappa_v, gamma_v = 1e-8, 1e-8
G_v, t_v = 0.3, 1.0
A_pure = A_H(G_v, "blue")
D_pure = np.zeros((4, 4))
sig0 = np.eye(4) / 2.0
sig_t = sigma_of_t(sig0, A_pure, D_pure, t_v)
F_target = 4 * t_v ** 2 * var_X(sig_t)
# numerical d(sigma)/dG
h = 1e-6 * G_v
sp = sigma_of_t(sig0, A_H(G_v + h, "blue"), D_pure, t_v)
sm = sigma_of_t(sig0, A_H(G_v - h, "blue"), D_pure, t_v)
F_num = qfi_gaussian(sig_t, (sp - sm) / (2 * h))
print(f"  4 t^2 Var(X)  = {F_target:.6f}")
print(f"  Banchi QFI    = {F_num:.6f}")
print(f"  ratio         = {F_num / F_target:.4f}")

# validation 2: entropy production >= 0 during thermal relaxation
print("Validation 2: Pi(t) >= 0 during relaxation to equilibrium (G = 0)")
kappa_0, gamma_0, nth_0 = 0.5, 0.05, 20.0
A_IRR = np.diag([-kappa_0 / 2, -kappa_0 / 2, -gamma_0 / 2, -gamma_0 / 2])
A_eq = A_full(0.0, "blue", kappa_0, gamma_0)
D_eq = D_mat(kappa_0, gamma_0, nth_0)
# single-mode squeezed vacuum on the optical mode, r = 0.6
r = 0.6
S1 = np.array([[np.cos(r), np.sin(r)], [np.sin(r), np.cos(r)]])  # x-p rotation is NOT squeeze; use true squeeze
S1 = np.array([[np.exp(-r), 0], [0, np.exp(r)]])                 # squeezing along x
sig0 = np.eye(4) / 2.0
sig0[:2, :2] = S1 @ (np.eye(2) / 2) @ S1.T
Pi_vals = []
tgrid = np.linspace(0, 12 / gamma_0, 14)  # long enough for the slow mechanical relaxation
for t in tgrid:
    sig_t = sigma_of_t(sig0, A_eq, D_eq, t)
    Pi_vals.append(entropy_production(sig_t, D_eq))
Pi_vals = np.array(Pi_vals)
print("  t/kappa^-1:", np.array2string(tgrid * kappa_0, precision=2))
print("  Pi(t)     :", np.array2string(Pi_vals, precision=5))
print(f"  min Pi    = {Pi_vals.min():.3e}  (must be >= 0)")
print(f"  final Pi  = {Pi_vals[-1]:.3e}  (must be ~ 0)")

# main sweep: steady state, blue vs red sideband
kappa, gamma, nth = 0.1, 1e-4, 100.0
G_th = np.sqrt(kappa * gamma) / 2.0
print(f"  kappa = {kappa}, gamma = {gamma}, n_th = {nth}")
print(f"  G_th  = {G_th:.6e}  (blue-sideband instability threshold)")

A_IRR = np.diag([-kappa / 2, -kappa / 2, -gamma / 2, -gamma / 2])
D = D_mat(kappa, gamma, nth)
Gs = G_th * np.linspace(0.02, 0.985, 60)

rows = []
for side in ("blue", "red"):
    for G in Gs:
        A = A_full(G, side, kappa, gamma)
        sig = solve_continuous_lyapunov(A, -D)  # scipy solves A X + X A^H = Q; we need A sigma + sigma A^T = -D
        sig = 0.5 * (sig + sig.T)
        h = 1e-6 * max(G, 1e-3) * G_th
        sp = solve_continuous_lyapunov(A_full(G + h, side, kappa, gamma), -D)
        sm = solve_continuous_lyapunov(A_full(G - h, side, kappa, gamma), -D)
        dsig = 0.5 * ((sp - sm) / (2 * h) + ((sp - sm) / (2 * h)).T)
        F = qfi_gaussian(sig, dsig)
        Pi = entropy_production(sig, D)
        n_b = 0.5 * (sig[2, 2] + sig[3, 3]) - 0.5
        n_d = 0.5 * (sig[0, 0] + sig[1, 1]) - 0.5
        rows.append((side, G, F, Pi, n_b, n_d))

import csv
with open(os.path.join(OUT, "sweep_data.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["side", "G", "QFI", "Pi_entropy_production", "n_mech", "n_opt"])
    w.writerows(rows)

data = {"blue": {}, "red": {}}
for side in ("blue", "red"):
    sel = [r for r in rows if r[0] == side]
    data[side] = {k: np.array([r[i] for r in sel])
                  for i, k in enumerate(["G", "F", "Pi", "n_b", "n_d"], start=1)}

for side in ("blue", "red"):
    d = data[side]
    print(f"  [{side}] F range: {d['F'].min():.3e} .. {d['F'].max():.3e}, "
          f"Pi range: {d['Pi'].min():.3e} .. {d['Pi'].max():.3e}")

print("data saved:   ", os.path.join(OUT, "sweep_data.csv"))
