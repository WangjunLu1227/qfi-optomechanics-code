# -*- coding: utf-8 -*-
"""
Reduction of the universal coefficient: F*delta^2 -> 1 near threshold.

Two ingredients:
  Sigma = lim_{delta->0} delta * sigma_ss(G_th - delta)      (rank-deficient)
  V     = lim_{delta->0} delta^2 * d(sigma_ss)/dG|_{G_th-delta}
Then, since sigma ~ Sigma/delta and d(sigma)/dG ~ V/delta^2, and
M = sigma (x) sigma - Omega (x) Omega/4 ~ (Sigma (x) Sigma)/delta^2,

    F * delta^2  ->  1/2 vec(V)^T (Sigma (x) Sigma)^{-1P} vec(V).

We evaluate this exactly (rational arithmetic where possible) for several
(r, n_th) and check it equals 1.
"""

import sympy as sp
import mpmath as mp

k, g, G, nth, d = sp.symbols('kappa gamma G n_th delta', positive=True)
Gth = sp.sqrt(k*g)/2

s00, s01, s02, s03 = sp.symbols('s00 s01 s02 s03')
s11, s12, s13 = sp.symbols('s11 s12 s13')
s22, s23 = sp.symbols('s22 s23')
s33 = sp.symbols('s33')
unk = [s00, s01, s02, s03, s11, s12, s13, s22, s23, s33]
S = sp.Matrix([[s00, s01, s02, s03], [s01, s11, s12, s13],
               [s02, s12, s22, s23], [s03, s13, s23, s33]])
A = sp.Matrix([[-k/2, 0, 0, -G], [0, -k/2, -G, 0],
               [0, -G, -g/2, 0], [-G, 0, 0, -g/2]])
D = sp.diag(k/2, k/2, g*(nth + sp.Rational(1, 2)), g*(nth + sp.Rational(1, 2)))
E = A*S + S*A.T + D
sol = sp.solve([sp.expand(E[i, j]) for i in range(4) for j in range(i, 4)],
               unk, dict=True)[0]
SS = S.subs(sol)

# G = Gth - delta ; expand around delta = 0
SS_d = SS.subs(G, Gth - d)
dSS_d = sp.diff(SS, G).subs(G, Gth - d)

Sigma = sp.zeros(4, 4)
V = sp.zeros(4, 4)
print("=== Sigma = lim delta*sigma ===")
for i in range(4):
    for j in range(4):
        Sigma[i, j] = sp.simplify(sp.limit(d*SS_d[i, j], d, 0))
        V[i, j] = sp.simplify(sp.limit(d**2*dSS_d[i, j], d, 0))
        if Sigma[i, j] != 0:
            print(f"  Sigma[{i},{j}] =", Sigma[i, j])
print("=== V = lim delta^2*d(sigma)/dG ===")
for i in range(4):
    for j in range(4):
        if V[i, j] != 0:
            print(f"  V[{i},{j}] =", sp.simplify(V[i, j]))

# --- 1) exact rational check of the algebraic identity ---
def coeff_matrix(rv, nv):
    sub = {k: 1, g: rv, nth: nv}
    Sig = sp.Matrix(4, 4, lambda i, j: sp.nsimplify(Sigma[i, j].subs(sub)))
    Vm = sp.Matrix(4, 4, lambda i, j: sp.nsimplify(V[i, j].subs(sub)))
    return Sig, Vm

def algebraic_value(Sig, Vm):
    """1/2 vec(V)^T (Sig (x) Sig)^-1P vec(V) with exact rationals."""
    M = sp.kronecker_product(Sig, Sig)
    v = sp.Matrix(16, 1, lambda i, j: Vm[i % 4, i // 4])   # column-major vec
    # restricted pseudo-inverse: solve on the range via least squares (exact)
    # M is rational -> use sympy's pinv if tractable, else solve normal eqs
    Mp = M.pinv()
    val = sp.Rational(1, 2) * (v.T * Mp * v)[0]
    return sp.nsimplify(sp.simplify(val))

print("\n=== exact (rational) evaluation of 1/2 vec(V)^T (Sigma(x)Sigma)^-1P vec(V) ===")
for (rv, nv) in [(sp.Rational(1, 10 ** 3), 100),
                 (sp.Rational(1, 10 ** 2), 10),
                 (sp.Rational(1, 10 ** 4), 1000)]:
    Sig, Vm = coeff_matrix(rv, nv)
    val = algebraic_value(Sig, Vm)
    print(f"  r = {rv}, n_th = {nv}:  F*delta^2 (limit) = {val}")

# --- 2) closed forms of Sigma and V in terms of (k, g, n) for the appendix ---
print("\n=== compact closed forms (k=1 gauge kept symbolic) ===")
for (i, j) in [(0, 0), (2, 2), (0, 3)]:
    print(f"  Sigma[{i},{j}] =", sp.simplify(Sigma[i, j]))
for (i, j) in [(0, 0), (2, 2), (0, 3)]:
    print(f"  V[{i},{j}]     =", sp.simplify(V[i, j]))
