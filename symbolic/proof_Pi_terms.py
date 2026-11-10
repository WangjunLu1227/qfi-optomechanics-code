# -*- coding: utf-8 -*-
"""
Term-by-term decomposition of the divergent part of the entropy production Pi.

Question addressed (paper, Section 5 and Appendix A):
    Pi = 2 Tr[A_irr] + 2 Tr[A_irr^T D^{-1} A_irr sigma] + (1/2) Tr[sigma^{-1} D]
Which of the three terms carries the 1/delta divergence, and what is the exact
coefficient C_Pi compared with the closed form

    C_Pi = gamma^{3/2} sqrt(kappa) (n+1)^2 / ((1+gamma/kappa)(2n+1)) ?

Method
------
Blue-sideband (two-mode squeezing) model in quadrature order u = (q_d, p_d, q_b, p_b):

    A = [[-k/2, 0, 0, -G], [0, -k/2, -G, 0],
         [0, -G, -g/2, 0], [-G, 0, 0, -g/2]]
    D = diag(k/2, k/2, g(n+1/2), g(n+1/2)),      k = kappa, g = gamma.

The U(1) symmetry of A fixes the steady-state covariance to the three-parameter
block form (same structure as the closed form used in the paper)

    sigma = [[a, 0, 0, c], [0, a, c, 0], [0, c, b, 0], [c, 0, 0, b]],

so the Lyapunov equation is a 3x3 linear system and the 4x4 inverse is obtained
analytically from 2x2 blocks, 1/(ab-c^2) [[b, -c], [-c, a]]. This keeps the
whole calculation exact and fast (a few seconds), while evaluating exactly the
same three terms as the original 10-unknown brute-force version.

Run:  python proof_Pi_terms.py
"""

import sympy as sp

k, g, G, nth, dl = sp.symbols('kappa gamma G n_th delta', positive=True)
Gth = sp.sqrt(k * g) / 2

a, b, c = sp.symbols('a b c')
S = sp.Matrix([[a, 0, 0, c], [0, a, c, 0], [0, c, b, 0], [c, 0, 0, b]])
A = sp.Matrix([[-k / 2, 0, 0, -G], [0, -k / 2, -G, 0],
               [0, -G, -g / 2, 0], [-G, 0, 0, -g / 2]])
D = sp.diag(k / 2, k / 2, g * (nth + sp.Rational(1, 2)), g * (nth + sp.Rational(1, 2)))
AIRR = sp.diag(-k / 2, -k / 2, -g / 2, -g / 2)

# ---- steady state: three independent components of the Lyapunov equation ----
E = A * S + S * A.T + D
eqs = [sp.expand(E[0, 0]), sp.expand(E[0, 3]), sp.expand(E[2, 2])]
sol = sp.solve(eqs, [a, b, c], dict=True)[0]
aa, bb, cc = sol[a], sol[b], sol[c]

# ---- the three terms of Pi, in closed form ----
t1 = 2 * sp.trace(AIRR)                       # finite: -(2k+g)
t2 = 2 * k * aa + 2 * g * bb / (2 * nth + 1)  # 2 Tr[A_irr^T D^-1 A_irr sigma]
t3 = (k * bb + g * (2 * nth + 1) * aa) / (aa * bb - cc ** 2)   # (1/2) Tr[sigma^-1 D]

near = {G: Gth - dl}   # approach threshold from below, delta = G_th - G

print("=== individual terms:  delta * term  as delta -> 0 ===")
for name, t in (("t1 = 2Tr[A_irr]", t1),
                ("t2 = 2Tr[A_irr^T D^-1 A_irr sigma]", t2),
                ("t3 = (1/2)Tr[sigma^-1 D]", t3)):
    expr = sp.cancel(dl * t.subs(near))
    lim = sp.limit(expr, dl, 0)
    print(f"  {name:38s} * delta ->  {sp.simplify(lim)}")

Pi = t1 + t2 + t3
expr = sp.cancel(dl * Pi.subs(near))
C = sp.simplify(sp.limit(expr, dl, 0))
claimed = g ** sp.Rational(3, 2) * sp.sqrt(k) * (nth + 1) ** 2 / ((1 + g / k) * (2 * nth + 1))

print("\n  exact   C_Pi = lim delta*Pi =", sp.factor(C))
print("  claimed        C_Pi =", sp.factor(claimed))
print("  difference     =", sp.simplify(C - claimed))

assert sp.simplify(C - claimed) == 0, "C_Pi does not match the closed form"
print("\nOK: the divergent part of Pi reproduces the closed-form C_Pi exactly.")
