"""
Rank-two structure of Sigma = lim_{delta->0} delta*sigma_ss.

Part (A): symbolic proof that a*b - c^2 = 0 for the RWA blue-sideband steady
state with an arbitrary diagonal diffusion D = diag(d1,d1,d2,d2), i.e. the
rank-two structure does not depend on the bath occupations.

Part (B): numerics for the counter-rotating (non-RWA) terms: linear coefficient
of the critical eigenvalue, eigenvalue scaling of delta*sigma, F*delta^2.
"""

import mpmath as mp
import sympy as sp

mp.mp.dps = 60

# (A) symbolic: rank-two independence of the bath occupations

G, k, g, d1, d2 = sp.symbols('G kappa gamma d1 d2', positive=True)
d = sp.symbols('delta', positive=True)
Gth = sp.sqrt(k * g) / 2

AH = sp.Matrix([[0, 0, 0, -G],
                [0, 0, -G, 0],
                [0, -G, 0, 0],
                [-G, 0, 0, 0]])
Airr = sp.diag(-k / 2, -k / 2, -g / 2, -g / 2)
A = AH + Airr
D = sp.diag(d1, d1, d2, d2)

# solve A sigma + sigma A^T = -D  (ten independent entries; use the known block ansatz)
a, b, c = sp.symbols('a b c')
S = sp.Matrix([[a, 0, 0, c],
               [0, a, c, 0],
               [0, c, b, 0],
               [c, 0, 0, b]])
res = sp.simplify(A * S + S * A.T + D)
eqs = []
for i in range(4):
    for j in range(i, 4):
        e = sp.simplify(res[i, j])
        if e != 0:
            eqs.append(e)
sol = sp.solve(eqs, [a, b, c], dict=True)[0]
a_s, b_s, c_s = sp.simplify(sol[a]), sp.simplify(sol[b]), sp.simplify(sol[c])
print("  sigma_qdqd =", sp.factor(a_s))
print("  sigma_qbqb =", sp.factor(b_s))
print("  sigma_qdpb =", sp.factor(c_s))

Delta4 = 4 * G ** 2 - k * g
# substitute G = Gth - delta and expand to leading order in delta
sub = {G: Gth - d}
a_lead = sp.simplify(sp.limit(sp.simplify(a_s.subs(sub)) * d, d, 0))
b_lead = sp.simplify(sp.limit(sp.simplify(b_s.subs(sub)) * d, d, 0))
c_lead = sp.simplify(sp.limit(sp.simplify(c_s.subs(sub)) * d, d, 0))
print("\n  Sigma entries (leading order in delta):")
print("    a =", sp.simplify(a_lead))
print("    b =", sp.simplify(b_lead))
print("    c =", sp.simplify(c_lead))
print("\n  a*b - c^2 =", sp.factor(sp.simplify(a_lead * b_lead - c_lead ** 2)))
# sanity: with d1 = k/2, d2 = g(n+1/2) this reproduces the closed forms
n = sp.symbols('n', positive=True)
chk_a = sp.simplify(a_lead.subs({d1: k / 2, d2: g * (n + sp.Rational(1, 2))}))
chk_b = sp.simplify(b_lead.subs({d1: k / 2, d2: g * (n + sp.Rational(1, 2))}))
chk_c = sp.simplify(c_lead.subs({d1: k / 2, d2: g * (n + sp.Rational(1, 2))}))
print("\n  with d1 = kappa/2, d2 = gamma_m(n_th+1/2):")
print("    a =", sp.simplify(chk_a))
print("    b =", sp.simplify(chk_b))
print("    c =", sp.simplify(chk_c))

# (B) non-RWA numerics

N = 4
OM = mp.matrix(N, N)
for i in (0, 2):
    OM[i, i + 1] = 1
    OM[i + 1, i] = -1


def vec(X):
    v = mp.matrix(N * N, 1)
    for i in range(N):
        for j in range(N):
            v[i + N * j] = X[i, j]
    return v


def lya(A, R):
    K = mp.matrix(N * N, N * N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    K[i + N * j, k + N * l] = \
                        (A[j, l] if i == k else 0) + (A[i, k] if j == l else 0)
    x = mp.lu_solve(K, vec(R))
    X = mp.matrix(N, N)
    for i in range(N):
        for j in range(N):
            X[i, j] = x[i + N * j]
    return X


def qfi(sig, dsig, rel_tol=mp.mpf('1e-28')):
    M = mp.matrix(N * N, N * N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    M[i + N * j, k + N * l] = \
                        sig[i, k] * sig[j, l] - OM[i, k] * OM[j, l] / 4
    E, Q = mp.eigsy(M)
    mx = max(abs(e) for e in E)
    tol = rel_tol * mx
    qv = Q.T * vec(dsig)
    s = mp.mpf(0)
    for i in range(N * N):
        if abs(E[i]) > tol:
            s += qv[i] ** 2 / E[i]
    return s / 2


def A_irr(k, g):
    A = mp.matrix(N, N)
    A[0, 0] = -k / 2
    A[1, 1] = -k / 2
    A[2, 2] = -g / 2
    A[3, 3] = -g / 2
    return A


def AH_nonrwa(G, wm):
    Mm = mp.matrix(N, N)
    Mm[0, 0] = -wm
    Mm[1, 1] = -wm
    Mm[2, 2] = wm
    Mm[3, 3] = wm
    Mm[0, 2] = 2 * G
    Mm[2, 0] = 2 * G
    return OM * Mm


k, g, nth = mp.mpf('0.1'), mp.mpf('1e-4'), mp.mpf(100)
Gth = mp.sqrt(k * g) / 2
D = mp.diag([k / 2, k / 2, g * (nth + mp.mpf('0.5')), g * (nth + mp.mpf('0.5'))])
Airr = A_irr(k, g)

print()
print("non-RWA: critical eigenvalue and scaling of delta*sigma")
for ratio in ('0.4', '0.2', '0.1', '0.05', '0.02'):
    r = mp.mpf(ratio)
    w = k / r
    Aof = lambda x: AH_nonrwa(x, w) + Airr

    def maxre(x):
        E, _ = mp.eig(Aof(x))
        return max(e.real for e in E)

    Gs = mp.findroot(maxre, Gth * (1 + mp.mpf('0.01')),
                     tol=mp.mpf('1e-40'), maxsteps=300)
    # linear coefficient of the critical eigenvalue
    h = Gs * mp.mpf('1e-8')
    slope = (maxre(Gs + h) - maxre(Gs - h)) / (2 * h)
    print(f"\n  kappa/w_m = {ratio:>5s}:  G*/G_th-1 = {mp.nstr(Gs/Gth-1, 10)}"
          f"   d(Re lam_c)/dG = {mp.nstr(slope, 10)}")
    print(f"    {'eps*':>8s} {'F*delta^2':>18s}    eigenvalues of delta*sigma (normalised)")
    for eps in ('1e-3', '1e-4', '1e-5', '1e-6', '1e-8'):
        e = mp.mpf(eps)
        G = Gs * (1 - e)
        A = Aof(G)
        sig = lya(A, -D)
        hh = Gs * mp.mpf('1e-9')
        Ap = (Aof(G + hh) - Aof(G - hh)) / (2 * hh)
        dsig = lya(A, -(Ap * sig + sig * Ap.T))
        F = qfi(sig, dsig)
        delta = Gs - G
        E = mp.eigsy(sig * delta)[0]
        mx = max(abs(x) for x in E)
        ev = [mp.nstr(x / mx, 4) for x in E]
        print(f"    {eps:>8s} {mp.nstr(F*delta**2, 14):>18s}    {ev}")
