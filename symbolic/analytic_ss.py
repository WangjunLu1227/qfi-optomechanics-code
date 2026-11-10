"""
Closed-form steady state of the linearized blue-sideband (two-mode squeezing)
model, and near-threshold expansions of the occupations n_d, n_b, the entropy
production rate Pi, the symplectic eigenvalues, d(sigma)/dG and the Gaussian
QFI (mpmath).

Conventions:  quadratures u = (x_d, p_d, x_b, p_b), [q,p] = i, vacuum
sigma = I/2, hbar = 1, dsigma/dt = A sigma + sigma A^T + D, threshold
G_th = sqrt(k*g)/2.
"""

import sympy as sp
import mpmath as mp

sp.mp = mp
x0, x1, x2, x3 = sp.symbols('s00 s01 s02 s03')
y1, y2, y3 = sp.symbols('s11 s12 s13')
z2, z3 = sp.symbols('s22 s23')
w3 = sp.symbols('s33')
unk = [x0, x1, x2, x3, y1, y2, y3, z2, z3, w3]

k, g, G, nth = sp.symbols('kappa gamma G n_th', positive=True)

S = sp.Matrix([[x0, x1, x2, x3],
               [x1, y1, y2, y3],
               [x2, y2, z2, z3],
               [x3, y3, z3, w3]])

A_BLUE = sp.Matrix([[-k/2, 0,   0,   -G ],
                    [0,   -k/2, -G,  0  ],
                    [0,   -G,   -g/2, 0  ],
                    [-G,  0,    0,   -g/2]])

D = sp.diag(k/2, k/2, g*(nth + sp.Rational(1, 2)), g*(nth + sp.Rational(1, 2)))

AIRR = sp.diag(-k/2, -k/2, -g/2, -g/2)

# steady state: solve A S + S A^T + D = 0
E = A_BLUE * S + S * A_BLUE.T + D
eqs = [sp.expand(E[i, j]) for i in range(4) for j in range(i, 4)]
sol = sp.solve(eqs, unk, dict=True)[0]
SS = S.subs(sol)
for e in unk:
    SS = sp.simplify(SS)
print("Steady-state covariance (blue sideband):")
for (i, j) in [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3),
               (2, 2), (2, 3), (3, 3)]:
    print(f"  sigma[{i},{j}] = {sp.simplify(SS[i, j])}")

# residual check
res = sp.simplify(A_BLUE * SS + SS * A_BLUE.T + D)
print("Lyapunov residual (max abs entry):", res)

# critical eigenvalue of A
lam = list(A_BLUE.eigenvals().keys())
print("\nEigenvalues of A (raw):", lam)
lam_c = sp.simplify(-sp.Rational(1, 4)*(k + g) + sp.sqrt(G**2 + (k - g)**2/16))
print("critical eigenvalue candidate:", sp.simplify(lam_c))
check = sp.simplify((A_BLUE - lam_c*sp.eye(4)).det())
print("det(A - lam_c) = 0 ?", check == 0, check)

# nondimensionalize: k -> 1, r = g/k, eps = (G_th - G)/G_th
r, eps = sp.symbols('r eps', positive=True)
subs_dim = {k: 1, g: r, G: sp.sqrt(r)/2*(1 - eps)}
SS_d = sp.simplify(SS.subs(subs_dim))

def series(expr, order=3):
    return sp.simplify(sp.series(sp.expand(expr), eps, 0, order).removeO())

n_d = sp.simplify((SS_d[0, 0] + SS_d[1, 1])/2 - sp.Rational(1, 2))
n_b = sp.simplify((SS_d[2, 2] + SS_d[3, 3])/2 - sp.Rational(1, 2))

print("\nNear-threshold expansions (dimensionless, eps = (G_th-G)/G_th):")
print("  n_d   ~", series(n_d))
print("  n_b   ~", series(n_b))

# entropy production, dimensionless
def Pi_expr(Sm, dim=False):
    Air = AIRR if dim else AIRR.subs({k: 1, g: r})
    Dd = D if dim else D.subs({k: 1, g: r})
    Sinv = sp.simplify(Sm.inv())
    t1 = 2*sp.trace(Air)
    t2 = 2*sp.trace(Air.T * Dd.inv() * Air * Sm)
    t3 = sp.Rational(1, 2)*sp.trace(Sinv * Dd)
    return sp.simplify(t1 + t2 + t3)

Pi_d = Pi_expr(SS_d)
print("  Pi    ~", series(Pi_d))
print("  Pi (next order) ~", series(Pi_d, 4))

# symplectic eigenvalues: eigenvalues of i*Omega*S
Om = sp.zeros(4)
Om[0, 1] = 1; Om[1, 0] = -1; Om[2, 3] = 1; Om[3, 2] = -1
nu = sp.Symbol('nu')
cp = sp.factor(sp.expand(sp.simplify((sp.I*Om*SS_d).charpoly(nu).as_expr())))
roots = []
numer, _denom = sp.together(cp).as_numer_denom()
for fac in sp.Mul.make_args(numer):
    if fac.has(nu):
        roots += sp.solve(sp.expand(fac), nu)
print("  symplectic eigenvalues, eps-expansion (r < 1):")
SQ = sp.sqrt(r**2 - 2*r + 1)
for s in roots:
    s2 = sp.simplify(sp.simplify(s).subs(SQ, 1 - r))
    ser = sp.series(s2, eps, 0, 2).removeO()
    a = sp.simplify(sp.limit(ser*eps, eps, 0))
    b = sp.simplify(sp.limit(ser - a/eps, eps, 0))
    print("   nu =", sp.nsimplify(sp.simplify(a)), "/eps  +", sp.simplify(b), "+ O(eps)")

# d(sigma)/dG: differentiate the closed form before substitution
dSS_dG = sp.simplify(SS.diff(G).subs(subs_dim))
print("\n  d sigma/dG leading entries:")
for (i, j) in [(0, 0), (0, 1), (0, 3), (1, 2), (2, 2), (3, 3)]:
    print(f"   d sigma[{i},{j}]/dG ~", series(dSS_dG[i, j]))

# QFI coefficient via mpmath at tiny eps (analytic sigma and dsigma)
print("\nQFI asymptotic coefficient (mpmath, closed forms):")

# SS_d(r, eps): sigma as a function of the dimensionless ratio r = g/k
# and eps = (G_th - G)/G_th, with G_th = sqrt(r)/2 in units of kappa.
# dSS_dG_d(r, eps) is d(sigma)/dG expressed in the same variables:
# it is already the VALUE of the derivative (chain rule applied
# symbolically before substitution) -- do NOT rescale again here.
SS_fun = sp.lambdify((r, eps, nth), SS_d, 'mpmath')
# differentiate wrt G symbolically, then substitute
dSS_dG_d = sp.simplify(SS.diff(G)).subs(subs_dim)
dSS_fun = sp.lambdify((r, eps, nth), dSS_dG_d, 'mpmath')

mp.mp.dps = 60

OM_m = mp.matrix([[0, 1, 0, 0], [-1, 0, 0, 0],
                  [0, 0, 0, 1], [0, 0, -1, 0]])

def qfi_mp(sig, dsig):
    """Banchi QFI in high precision: F = 1/2 vec(dsig)^T M^-1 vec(dsig)."""
    n = 4
    M = mp.matrix(n*n, n*n)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    M[a*n + c, b*n + d] = sig[a, b]*sig[c, d] \
                        - (OM_m[a, b]*OM_m[c, d])/4
    v = mp.matrix(n*n, 1)
    for a in range(n):
        for b in range(n):
            v[a*n + b] = dsig[a, b]
    u, s, vt = mp.svd(M)          # M = U diag(S) Vt  (Vt = V^H)
    utv = u.H * v
    x = mp.matrix(n*n, 1)
    for i in range(n*n):
        if abs(s[i]) > mp.mpf('1e-30'):
            # add (utv[i]/s[i]) * V[:,i], V = vt^H  -> V[i,j] = conj(vt[j,i])
            for j in range(n*n):
                x[j] += (utv[i]/s[i]) * mp.conj(vt[i, j])
    return mp.mpf('0.5') * (v.H * x)[0].real

def Pi_num(sig, rval, nv):
    Airr = mp.diag([-0.5, -0.5, -rval/2, -rval/2])
    Dd = mp.diag([0.5, 0.5, rval*(nv + 0.5), rval*(nv + 0.5)])
    Di = Dd**-1
    t1 = 2*mp.trace(Airr)
    t2 = 2*mp.trace(Airr.T * Di * Airr * sig)
    t3 = 0.5*mp.trace(Di * sig**-1)
    return t1 + t2 + t3

# cross-check of qfi_mp against a numpy implementation of the same formula
import numpy as np

OMEGA_NP = np.zeros((4, 4))
OMEGA_NP[0, 1] = 1.0; OMEGA_NP[1, 0] = -1.0   # mode d
OMEGA_NP[2, 3] = 1.0; OMEGA_NP[3, 2] = -1.0   # mode b

def qfi_gaussian(sigma, dsigma):
    """Banchi-Braunstein-Pirandola Gaussian QFI (zero mean)."""
    M = np.kron(sigma, sigma) - np.kron(OMEGA_NP, OMEGA_NP) / 4.0
    v = dsigma.flatten(order="F")
    Minv_v = np.linalg.pinv(M, rcond=1e-13, hermitian=True) @ v
    return 0.5 * v @ Minv_v

r_ck, eps_ck = 1e-3, mp.mpf('1e-3')
sig_ck = mp.matrix(SS_fun(r_ck, eps_ck, 100))
dsig_ck = mp.matrix(dSS_fun(r_ck, eps_ck, 100))
F_mp = qfi_mp(sig_ck, dsig_ck)
sig_np = np.array([[float(sig_ck[i, j]) for j in range(4)] for i in range(4)])
dsig_np = np.array([[float(dsig_ck[i, j]) for j in range(4)] for i in range(4)])
F_np = qfi_gaussian(sig_np, dsig_np)
print(f"  cross-check @eps=1e-3: F_mpmath = {mp.nstr(F_mp, 10)}")
print(f"                         F_numpy  = {F_np:.6e}")
print(f"                         ratio    = {mp.nstr(F_mp/F_np, 8)}")

def C_F(rval, nthv, epsv=mp.mpf('1e-6')):
    sig = mp.matrix(SS_fun(rval, epsv, nthv))
    dsig = mp.matrix(dSS_fun(rval, epsv, nthv))
    return qfi_mp(sig, dsig) * (mp.sqrt(rval)*epsv/2)**2   # F * delta_dim^2

def C_Pi(rval, nthv):
    # analytic from the series: Pi_dim * delta_dim -> r^{3/2}(n+1)^2 / ((1+r)(2n+1))
    return rval**mp.mpf('1.5') * (nthv+1)**2 / ((1+rval)*(2*nthv+1))

print(f"\n  C_Pi (analytic) = {mp.nstr(C_Pi(1e-3, 100), 8)}   [dimensionless]")

# convergence of C_F = F * delta^2 toward its eps->0 value
print(f"\n  convergence of C_F(r=1e-3, n=100):")
for epsv in [mp.mpf('1e-4'), mp.mpf('1e-5'), mp.mpf('1e-6')]:
    print(f"    eps={mp.nstr(epsv, 3)}:  C_F = {mp.nstr(C_F(1e-3, 100, epsv), 8)}")

# parameter scan of the asymptotic ratio F/Pi^2 -> C_F/C_Pi^2
print(f"\n  asymptotic ratio  F/Pi^2 = C_F/C_Pi^2  (eps=1e-6):")
print(f"  {'r':>8s} {'n_th':>6s} {'C_F':>14s} {'C_Pi':>14s} {'ratio':>14s} {'closed form':>14s}")
scan = []
for rv in ('1e-2', '1e-3', '1e-4'):
    for nv in (10, 100, 1000):
        rv_m = mp.mpf(rv)
        cf = C_F(rv_m, nv)
        cp = C_Pi(rv_m, nv)
        closed = (1+rv_m)**2*(2*nv+1)**2/(rv_m**3*(nv+1)**4)
        scan.append((rv, nv, cf, cp))
        print(f"  {rv:>8s} {nv:>6d} {mp.nstr(cf, 10):>14s} "
              f"{mp.nstr(cp, 10):>14s} {mp.nstr(cf/cp**2, 10):>14s} "
              f"{mp.nstr(closed, 10):>14s}")

# try to recognize the closed form of the ratio in (r, n)
def ratio_float(rv, nv):
    for (rvs, nvs, cf, cp) in scan:
        if rvs == rv and nvs == nv:
            return float(cf/cp**2)
rr = [ratio_float('1e-3', nv) for nv in (10, 100, 1000)]
print("\n  ratio vs n_th (r=1e-3):", [f"{v:.6e}" for v in rr])
print("  ratios of ratios (n=10->100, 100->1000):",
      f"{rr[1]/rr[0]:.6f}, {rr[2]/rr[1]:.6f}")
print("  predicted (2n+1)^2/(n+1)^4 ratios:",
      f"{((201)**2/(101)**4)/((21)**2/(11)**4):.6f}, "
      f"{((2001)**2/(1001)**4)/((201)**2/(101)**4):.6f}")
rn = [ratio_float(rv, 100) for rv in ('1e-2', '1e-3', '1e-4')]
print("  ratio vs r (n=100):", [f"{v:.6e}" for v in rn])

# red sideband: saturation statements
print("\nRed sideband (beam splitter):")
A_RED = sp.Matrix([[-k/2, 0,   0,   G],
                   [0,   -k/2, -G,  0],
                   [0,    G,   -g/2, 0],
                   [-G,   0,   0,  -g/2]])
E_r = A_RED * S + S * A_RED.T + D
sol_r = sp.solve([sp.expand(E_r[i, j]) for i in range(4) for j in range(i, 4)],
                 unk, dict=True)[0]
SS_r = S.subs(sol_r)
SS_r = sp.simplify(SS_r)
res_r = sp.simplify(A_RED * SS_r + SS_r * A_RED.T + D)
print("  Lyapunov residual (max abs entry):", res_r)
SSr_d = sp.simplify(SS_r.subs(subs_dim))
n_d_r = sp.simplify((SSr_d[0, 0] + SSr_d[1, 1])/2 - sp.Rational(1, 2))
Pi_r = Pi_expr(SSr_d)
print("  n_d(red) as G->infinity (G->oo):",
      sp.simplify(sp.limit(n_d_r.subs(eps, sp.Symbol('ee')*0 + 0), eps, 0))
      if False else sp.simplify(n_d_r))
# limit G -> inf: substitute G -> oo directly on the dimensional form
n_d_r_inf = sp.limit(sp.simplify((SS_r[0, 0] + SS_r[1, 1])/2 - sp.Rational(1, 2)), G, sp.oo)
Pi_r_inf = sp.limit(sp.simplify(Pi_expr(SS_r, dim=True)), G, sp.oo)
print("  lim_{G->oo} n_d(red) =", sp.simplify(n_d_r_inf))
print("  lim_{G->oo} Pi(red)  =", sp.simplify(Pi_r_inf))
dSSr = sp.simplify(SS_r.diff(G))
print("  lim_{G->oo} d(sigma)/dG (red), entrywise:")
for (i, j) in [(0, 0), (0, 3), (2, 2), (3, 0)]:
    print(f"    [{i},{j}] ->", sp.simplify(sp.limit(dSSr[i, j], G, sp.oo)))

# numeric symplectic eigenvalues near threshold
print("\nNumeric symplectic eigenvalues (r=1e-3, n=100):")
import numpy.linalg as npl
Om_np = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=complex)
for epsv in (mp.mpf('1e-3'), mp.mpf('1e-6')):
    sig = mp.matrix(SS_fun(mp.mpf('1e-3'), epsv, 100))
    sig_np = np.array([[complex(sig[i, j]) for j in range(4)] for i in range(4)])
    ev = np.sort(npl.eigvals(1j*Om_np @ sig_np).real)
    print(f"  eps={mp.nstr(epsv, 3)}:  nu = {np.round(ev, 6)}")
