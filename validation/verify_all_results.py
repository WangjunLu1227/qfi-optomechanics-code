"""
Checks of the paper results in three arithmetics:
A1-A10  SymPy (closed forms, charpoly, asymptotics, C_Pi)
B1-B6   60-digit mpmath (mechanism eq, eq:C, c1, n_b, trade-off, red sideband)
C1-C5   double precision, same sweep as the figures
"""
import numpy as np
import sympy as sp
import mpmath as mp

mp.mp.dps = 60

# A. symbolic (sympy)

k, g, G = sp.symbols("kappa gamma G", positive=True)
lam = sp.Symbol("lambda")  # plain symbol: charpoly strips assumptions
n = sp.symbols("n_th", positive=True)
d, db, dd = sp.symbols("delta d_b d_d", positive=True)

def A_sym(side="blue", Gx=G):
    s = -1 if side == "blue" else 1
    A = sp.zeros(4, 4)
    A[0, 3] = s * Gx
    A[1, 2] = -Gx
    A[2, 1] = s * Gx
    A[3, 0] = -Gx
    A[0, 0] = A[1, 1] = -k / 2
    A[2, 2] = A[3, 3] = -g / 2
    return A

print("\nA1. closed-form steady state solves the Lyapunov equation exactly")
Gth = sp.sqrt(k * g) / 2
D4 = 4 * G**2 - k * g
s00 = (4 * G**2 * (k - g * (2 * n + 1)) - k * g * (k + g)) / (2 * (k + g) * D4)
s22 = (2 * G**2 * (g * (2 * n + 1) - k) - k * g * (k + g) * (n + sp.Rational(1, 2))) / ((k + g) * D4)
s03 = 2 * G * g * k * (n + 1) / ((k + g) * D4)
Sig = sp.Matrix([[s00, 0, 0, s03], [0, s00, s03, 0], [0, s03, s22, 0], [s03, 0, 0, s22]])
D = sp.diag(k / 2, k / 2, g * (n + sp.Rational(1, 2)), g * (n + sp.Rational(1, 2)))
A = A_sym("blue")
R = sp.simplify(A * Sig + Sig * A.T + D)
print("    residual max |A sigma + sigma A^T + D| =", max(abs(x) for x in R))

print("\nA2. characteristic polynomial factorisation")
cp = sp.factor(A_sym("blue").charpoly(lam).as_expr())
paper_cp = ((lam + (k + g) / 4) ** 2 - (16 * G**2 + (k - g) ** 2) / 16) ** 2
print("    det(A - lambda I) - expected form =", sp.simplify(sp.factor(sp.expand(cp) - sp.expand(paper_cp))))
diffnum = (cp - paper_cp).subs({k: 0.37, g: 0.11, G: 0.21, lam: 0.53})
print("    numerical spot check:", float(diffnum))

print("\nA3. threshold: lambda_+ vanishes at Gth = sqrt(kappa*gamma)/2")
lp = -(k + g) / 4 + sp.sqrt(16 * G**2 + (k - g) ** 2) / 4
print("    lambda_+(Gth) =", sp.simplify(lp.subs(G, Gth).subs(sp.sqrt(4 * k * g + (k - g) ** 2), k + g)))

print("\nA4. critical-eigenvalue slope: lambda_+ = -2 sqrt(kg)/(k+g) delta + O(delta^2)")
dlt = sp.symbols("delta", positive=True)
slope = sp.diff(lp, G).subs(G, Gth).subs(sp.sqrt(4 * k * g + (k - g) ** 2), k + g)
coeff = sp.simplify(slope)
print("    d(lambda_+)/dG at Gth =", coeff, "  (expected 2 sqrt(kg)/(k+g))")
print("    difference:", sp.simplify(coeff - 2 * sp.sqrt(k * g) / (k + g)))

print("\nA5. asymptotic sigma = Sigma/delta + O(1);  a, b, c coefficients")
def lim(expr):
    return sp.simplify(sp.limit(dlt * expr.subs(G, Gth - dlt), dlt, 0))
a_num = lim(s00)
b_num = lim(s22)
c_num = lim(s03)
a_paper = g**sp.Rational(3, 2) * sp.sqrt(k) * (n + 1) / (4 * (k + g))
b_paper = sp.sqrt(g) * k**sp.Rational(3, 2) * (n + 1) / (4 * (k + g))
c_paper = -k * g * (n + 1) / (4 * (k + g))
print("    a - a_paper =", sp.simplify(a_num - a_paper))
print("    b - b_paper =", sp.simplify(b_num - b_paper))
print("    c - c_paper =", sp.simplify(c_num - c_paper))

print("\nA6. rank two: a*b - c^2 = 0")
print("    a*b - c^2 =", sp.simplify(a_paper * b_paper - c_paper**2))

print("\nA7. entropy-production coefficient")
CPi_paper = g**sp.Rational(3, 2) * sp.sqrt(k) * (n + 1) ** 2 / ((1 + g / k) * (2 * n + 1))
lhs = 2 * k * a_paper + g * b_paper / (n + sp.Rational(1, 2))
print("    2*kappa*a + gamma*b/(n+1/2) - C_Pi =", sp.simplify(lhs - CPi_paper))

print("\nA8. general diffusion D = diag(d_d,d_d,d_b,d_b): closed forms + C_Pi-general")
Dg = sp.diag(dd, dd, db, db)
Ag = A_sym("blue")
# solve Lyapunov symbolically with the block-symmetric ansatz
x1, x2, x3 = sp.symbols("x1 x2 x3")
Sg = sp.Matrix([[x1, 0, 0, x3], [0, x1, x3, 0], [0, x3, x2, 0], [x3, 0, 0, x2]])
eqs = list(Ag * Sg + Sg * Ag.T + Dg)
sol = sp.solve(eqs, [x1, x2, x3], simplify=True)
a_gen = sp.simplify(sp.limit(dlt * sol[x1].subs(G, Gth - dlt), dlt, 0))
b_gen = sp.simplify(sp.limit(dlt * sol[x2].subs(G, Gth - dlt), dlt, 0))
c_gen = sp.simplify(sp.limit(dlt * sol[x3].subs(G, Gth - dlt), dlt, 0))
a_gp = sp.sqrt(g) * (dd * g + db * k) / (4 * sp.sqrt(k) * (k + g))
b_gp = sp.sqrt(k) * (dd * g + db * k) / (4 * sp.sqrt(g) * (k + g))
c_gp = -(dd * g + db * k) / (4 * (k + g))
print("    a_gen - paper =", sp.simplify(a_gen - a_gp))
print("    b_gen - paper =", sp.simplify(b_gen - b_gp))
print("    c_gen - paper =", sp.simplify(c_gen - c_gp))
print("    a_gen*b_gen - c_gen^2 =", sp.simplify(a_gen * b_gen - c_gen**2))
CPg_paper = sp.sqrt(k * g) * (dd * g + db * k) / (4 * (k + g)) * (k / dd + g / db)
CPg_lhs = sp.simplify(k**2 * a_gp / dd + g**2 * b_gp / db)
print("    kappa^2 a/dd + gamma^2 b/db - C_Pi-general =", sp.simplify(CPg_lhs - CPg_paper))
red = sp.simplify(CPg_paper.subs([(dd, k / 2), (db, g * (n + sp.Rational(1, 2)))]) - CPi_paper)
print("    C_Pi-general reduced to standard case:", red)

print("\nA9. red sideband: purely imaginary eigenvalue pairs")
Ar = A_sym("red")
ev = sp.factor(Ar.charpoly(lam).as_expr())
print("    char.poly (red) =", ev)
print("    -> purely complex eigenvalues -(k+g)/4 +- i*sqrt(G^2-(k-g)^2/16), doubly degenerate")
print("    (n_d and Pi_red limits verified numerically in B6 below)")

print("\nA10. Lemma 1 check on an explicit rank-2 matrix")
# build explicit rank-2 matrix
P = sp.Matrix([[1, 0, 2], [0, 1, 3], [0, 0, 0]])
M2 = P * P.T
vecS = sp.Matrix([M2[i, j] for i in range(3) for j in range(3)])
MK = sp.kronecker_product(M2, M2)
pinv = MK.pinv()
Cq = sp.simplify(vecS.T * pinv * vecS)
print("    1/2 vec(S)^T pinv(S kron S) vec(S) =", Cq, " (Lemma 1 predicts r/2 =", M2.rank() / 2, ")")

# B. 60-digit mpmath

def kron(a, b):
    m, n_ = a.rows, a.cols
    p, q = b.rows, b.cols
    out = mp.matrix(m * p, n_ * q)
    for i in range(m):
        for j in range(n_):
            for u in range(p):
                for v in range(q):
                    out[i * p + u, j * q + v] = a[i, j] * b[u, v]
    return out

def sigma_blue_mp(G, k_, g_, nth):
    D4 = 4 * G**2 - k_ * g_
    s00 = (4 * G**2 * (k_ - g_ * (2 * nth + 1)) - k_ * g_ * (k_ + g_)) / (2 * (k_ + g_) * D4)
    s22 = (2 * G**2 * (g_ * (2 * nth + 1) - k_) - k_ * g_ * (k_ + g_) * (nth + mp.mpf("0.5"))) / ((k_ + g_) * D4)
    s03 = 2 * G * g_ * k_ * (nth + 1) / ((k_ + g_) * D4)
    S = mp.matrix(4, 4)
    S[0, 0] = S[1, 1] = s00
    S[2, 2] = S[3, 3] = s22
    S[0, 3] = S[3, 0] = S[1, 2] = S[2, 1] = s03
    return S

OM = mp.matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])

def pinv_sym(M, tol=None):
    # symmetric pseudoinverse via eigendecomposition
    lam, V = mp.eigsy(M)
    mx = max(abs(x) for x in lam)
    if tol is None:
        tol = mx * mp.mpf(10) ** (-mp.mp.dps + 10)
    out = mp.matrix(M.rows, M.cols)
    for i in range(M.rows):
        if abs(lam[i]) > tol:
            out += lam[i] ** (-1) * mp.matrix(V[:, i]) * mp.matrix(V[:, i]).T
    return out

def qfi_mp(S, dS):
    M = kron(S, S) - kron(OM, OM) / 4
    v = mp.matrix(16, 1)
    idx = 0
    for i in range(4):
        for j in range(4):
            v[idx, 0] = dS[i, j]
            idx += 1
    return mp.mpf("0.5") * (v.T * pinv_sym(M) * v)[0]

def diff_sigma(G, k_, g_, nth):
    # analytic derivative via mp.diff on each entry
    f00 = lambda x: sigma_blue_mp(x, k_, g_, nth)[0, 0]
    f22 = lambda x: sigma_blue_mp(x, k_, g_, nth)[2, 2]
    f03 = lambda x: sigma_blue_mp(x, k_, g_, nth)[0, 3]
    S = mp.matrix(4, 4)
    d00, d22, d03 = mp.diff(f00, G), mp.diff(f22, G), mp.diff(f03, G)
    S[0, 0] = S[1, 1] = d00
    S[2, 2] = S[3, 3] = d22
    S[0, 3] = S[3, 0] = S[1, 2] = S[2, 1] = d03
    return S

def trace_mp(M):
    return sum(M[i, i] for i in range(M.rows))

def Pi_mp(S, k_, g_, nth):
    Airr = mp.diag([-k_ / 2, -k_ / 2, -g_ / 2, -g_ / 2])
    Dm = mp.diag([k_ / 2, k_ / 2, g_ * (nth + mp.mpf("0.5")), g_ * (nth + mp.mpf("0.5"))])
    Dinv = mp.matrix(4, 4)
    for i_ in range(4):
        Dinv[i_, i_] = 1 / Dm[i_, i_]
    t1 = 2 * trace_mp(Airr)
    t2 = 2 * trace_mp(Airr.T * Dinv * Airr * S)
    Sinv = mp.inverse(S)
    t3 = mp.mpf("0.5") * trace_mp(Sinv * Dm)
    return t1 + t2 + t3

print("\nB1. mechanism equation Sigma = (tau_c/2) P_c D P_c^T at delta = 1e-15*Gth")
k0, g0, n0 = mp.mpf("0.1"), mp.mpf("1e-4"), mp.mpf(100)
Gth0 = mp.sqrt(k0 * g0) / 2
delta = mp.mpf(10) ** (-15) * Gth0
G0 = Gth0 - delta
A_mp = mp.matrix(4, 4)
A_mp[0, 0] = A_mp[1, 1] = -k0 / 2
A_mp[2, 2] = A_mp[3, 3] = -g0 / 2
A_mp[0, 3] = -G0
A_mp[1, 2] = -G0
A_mp[2, 1] = -G0
A_mp[3, 0] = -G0
# spectral projector onto the two critical eigenvalues
lamA, V = mp.eig(A_mp)
order = sorted(range(4), key=lambda i: lamA[i].real, reverse=True)
crit = order[:2]
Vc = mp.matrix(4, 2)
for a_, i in enumerate(crit):
    for r_ in range(4):
        Vc[r_, a_] = V[r_, i]
# oblique projector P_c = Vc (Wc^T Vc)^{-1} Wc^T with left eigenvectors
# left eigenvectors of A = right eigenvectors of A^T
lamT, VT = mp.eig(A_mp.T)
Wc = mp.matrix(4, 2)
for a_, i in enumerate(crit):
    for r_ in range(4):
        Wc[r_, a_] = VT[r_, i]
Mnorm = Wc.T * Vc
Pc = Vc * mp.inverse(Mnorm) * Wc.T
Dm = mp.diag([k0 / 2, k0 / 2, g0 * (n0 + mp.mpf("0.5")), g0 * (n0 + mp.mpf("0.5"))])
Sig_pred = (mp.mpf("0.5") * ((k0 + g0) / (2 * mp.sqrt(k0 * g0)))) * (Pc * Dm * Pc.T)
Sig_exact = delta * sigma_blue_mp(G0, k0, g0, n0)
err = max(abs(Sig_pred[i, j] - Sig_exact[i, j]) for i in range(4) for j in range(4))
scale = max(abs(Sig_exact[i, j]) for i in range(4) for j in range(4))
print(f"    max |pred-exact| = {mp.nstr(err, 3)}   scale = {mp.nstr(scale, 3)}   rel = {mp.nstr(err/scale, 3)}")

print("\nB2. eq:C : 1/2 vec(Sigma)^T pinv(Sigma kron Sigma) vec(Sigma) = 1 (exact Sigma)")
S = sigma_blue_mp(Gth0 * mp.mpf("0.99"), k0, g0, n0)
# exact Sigma from a,b,c
a_ = g0**1.5 * mp.sqrt(k0) * (n0 + 1) / (4 * (k0 + g0))
b_ = mp.sqrt(g0) * k0**1.5 * (n0 + 1) / (4 * (k0 + g0))
c_ = -k0 * g0 * (n0 + 1) / (4 * (k0 + g0))
Sx = mp.matrix(4, 4)
Sx[0, 0] = Sx[1, 1] = a_
Sx[2, 2] = Sx[3, 3] = b_
Sx[0, 3] = Sx[3, 0] = Sx[1, 2] = Sx[2, 1] = c_
MK = kron(Sx, Sx)
v = mp.matrix(16, 1)
idx = 0
for i in range(4):
    for j in range(4):
        v[idx, 0] = Sx[i, j]
        idx += 1
Cq = mp.mpf("0.5") * (v.T * pinv_sym(MK) * v)[0]
print(f"    C = {mp.nstr(Cq, 12)}   (Lemma 1: r/2 = 1)")

print("\nB3. F_G delta^2 -> 1 and subleading c1 (expect c1 ~= 0.79 for kappa=0.1, g=1e-4, n=100)")
for (k_, g_, n_, note) in [(mp.mpf("0.1"), mp.mpf("1e-4"), mp.mpf(100), "main params"),
                           (mp.mpf(1), mp.mpf(1), mp.mpf(5), "symmetric params"),
                           (mp.mpf("0.3"), mp.mpf("3e-3"), mp.mpf(50), "(0.3, 3e-3, 50)")]:
    Gth_ = mp.sqrt(k_ * g_) / 2
    for eta in [mp.mpf("1e-6"), mp.mpf("1e-8")]:
        dlt_ = eta * Gth_
        G_ = Gth_ - dlt_
        F = qfi_mp(sigma_blue_mp(G_, k_, g_, n_), diff_sigma(G_, k_, g_, n_))
        c1 = (1 - F * dlt_**2) / eta
        print(f"    {note}: eta={mp.nstr(eta,3)}:  F*delta^2-1 = {mp.nstr(F*dlt_**2-1, 8)},  c1 = {mp.nstr(c1, 8)}")

print("\nB4. discussion n_b = kappa(n+1)sqrt(kg) / [4(k+g) delta]")
for eta in [mp.mpf("1e-5"), mp.mpf("1e-8")]:
    dlt_ = eta * Gth0
    G_ = Gth0 - dlt_
    S_ = sigma_blue_mp(G_, k0, g0, n0)
    nb = (S_[2, 2] + S_[3, 3]) / 2 - mp.mpf("0.5")
    claimed = k0 * (n0 + 1) * mp.sqrt(k0 * g0) / (4 * (k0 + g0) * dlt_)
    print(f"    eta={mp.nstr(eta,3)}:  n_b exact = {mp.nstr(nb, 10)},  paper formula = {mp.nstr(claimed, 10)},  rel.diff = {mp.nstr(abs(nb-claimed)/nb, 2)}")

print("\nB5. trade-off constant F/Pi^2")
ref = (1 + g0 / k0) ** 2 * (2 * n0 + 1) ** 2 / (k0 * g0**3 * (n0 + 1) ** 4)
for eta in [mp.mpf("1e-5"), mp.mpf("1e-7")]:
    dlt_ = eta * Gth0
    G_ = Gth0 - dlt_
    S_ = sigma_blue_mp(G_, k0, g0, n0)
    F = qfi_mp(S_, diff_sigma(G_, k0, g0, n0))
    P_ = Pi_mp(S_, k0, g0, n0)
    print(f"    eta={mp.nstr(eta,3)}:  F/Pi^2 = {mp.nstr(F/P_**2, 10)},  closed form = {mp.nstr(ref, 10)},  rel.diff = {mp.nstr(abs(F/P_**2-ref)/ref, 2)}")

print("\nB6. red sideband Pi -> closed form at large G")
Gth_r = Gth0
for Gr in [mp.mpf(50), mp.mpf(200)]:
    # red drift
    Ar = mp.matrix(4, 4)
    Ar[0, 0] = Ar[1, 1] = -k0 / 2
    Ar[2, 2] = Ar[3, 3] = -g0 / 2
    Ar[0, 3] = Gr
    Ar[1, 2] = -Gr
    Ar[2, 1] = Gr
    Ar[3, 0] = -Gr
    # solve Lyapunov in 60 digits: (I kron A + A kron I) vec(S) = -vec(D)
    K = kron(mp.eye(4), Ar) + kron(Ar, mp.eye(4))
    Dv = mp.matrix(16, 1)
    idx = 0
    for i in range(4):
        for j in range(4):
            Dv[idx, 0] = Dm[i, j]
            idx += 1
    sv = -mp.inverse(K) * Dv
    Sr = mp.matrix(4, 4)
    idx = 0
    for i in range(4):
        for j in range(4):
            Sr[i, j] = sv[idx, 0]
            idx += 1
    Sr = (Sr + Sr.T) / 2
    nd = (Sr[0, 0] + Sr[1, 1]) / 2 - mp.mpf("0.5")
    CP_red = 4 * g0 * k0 * n0**2 / (2 * g0 * n0 + g0 + 2 * k0 * n0 + k0)
    P_ = Pi_mp(Sr, k0, g0, n0)
    print(f"    G={Gr}:  n_d = {mp.nstr(nd, 8)} (limit {mp.nstr(g0*n0/(k0+g0), 8)}),  Pi = {mp.nstr(P_, 8)},  closed form = {mp.nstr(CP_red, 8)}")

# C. dataset claims (double precision, same sweep as the figures)
from scipy.linalg import solve_continuous_lyapunov

def A_full(side, G_, k_, g_):
    A = np.zeros((4, 4))
    s = -1.0 if side == "blue" else 1.0
    A[0, 3] = s * G_
    A[1, 2] = -G_
    A[2, 1] = s * G_
    A[3, 0] = -G_
    A[0, 0] = A[1, 1] = -k_ / 2
    A[2, 2] = A[3, 3] = -g_ / 2
    return A

def ss_np(side, G_, k_, g_, n_):
    D = np.diag([k_ / 2, k_ / 2, g_ * (n_ + 0.5), g_ * (n_ + 0.5)])
    S = solve_continuous_lyapunov(A_full(side, G_, k_, g_), -D)
    return 0.5 * (S + S.T)

def qfi_np(S, dS):
    o = np.array([[0, 1], [-1, 0]])
    Om = np.kron(np.eye(2), o)
    M = np.kron(S, S) - np.kron(Om, Om) / 4
    Minv = np.linalg.pinv(M, rcond=1e-13)
    return 0.5 * float(dS.reshape(-1) @ Minv @ dS.reshape(-1))

kD, gD, nD = 0.1, 1e-4, 100.0
GthD = np.sqrt(kD * gD) / 2
Gs = GthD * np.linspace(0.02, 0.985, 60)
h = 1e-6 * GthD
res = {"blue": {"F": [], "Pi": []}, "red": {"F": [], "Pi": []}}
for side in ("blue", "red"):
    for G_ in Gs:
        S = ss_np(side, G_, kD, gD, nD)
        Sp = ss_np(side, G_ + h, kD, gD, nD)
        Sm = ss_np(side, G_ - h, kD, gD, nD)
        F = qfi_np(S, (Sp - Sm) / (2 * h))
        Airr = np.diag([-kD / 2, -kD / 2, -gD / 2, -gD / 2])
        D = np.diag([kD / 2, kD / 2, gD * (nD + 0.5), gD * (nD + 0.5)])
        t1 = 2 * np.trace(Airr)
        t2 = 2 * np.trace(Airr.T @ np.linalg.inv(D) @ Airr @ S)
        t3 = 0.5 * np.trace(np.linalg.inv(S) @ D)
        res[side]["F"].append(F)
        res[side]["Pi"].append(t1 + t2 + t3)
Fb = np.array(res["blue"]["F"]); Pb = np.array(res["blue"]["Pi"])
Fr = np.array(res["red"]["F"]); Pr = np.array(res["red"]["Pi"])

print("\nC1. red-sideband claims: max F = 4.7e5, max Pi = 9.8e-3, 3 orders below blue")
print(f"    max F_red = {Fr.max():.3e}   max Pi_red = {Pr.max():.3e}")
print(f"    F_blue(0.985 Gth) = {Fb[-1]:.3e}   ratio F_blue/F_red_max = {Fb[-1]/Fr.max():.2e}")

print("\nC2. fitted exponents F ~ Pi^alpha, windows top 1/3, 1/10, 1/20 (expect 1.75, 1.91, 1.94)")
for frac, ref in [(3, 1.75), (10, 1.91), (20, 1.94)]:
    m = len(Fb) // frac
    x = Pb[-m:]; y = Fb[-m:]
    c = np.polyfit(np.log(x), np.log(y), 1)
    print(f"    top 1/{frac}: alpha = {c[0]:.3f}")

print("\nC3. F vs (Gth - G) exponent in narrowest windows (table: -2.10)")
for npts in [20, 10, 6, 3]:
    dl = GthD - Gs[-npts:]
    c = np.polyfit(np.log(dl), np.log(Fb[-npts:]), 1)
    print(f"    last {npts} points: exponent = {c[0]:.3f}")

print("\nC4. Pi delta / C_Pi -> 1 and F delta^2 -> 1 along the sweep (double precision)")
CPi = gD**1.5 * np.sqrt(kD) * (nD + 1) ** 2 / ((1 + gD / kD) * (2 * nD + 1))
dl = GthD - Gs
print(f"    eta = 0.015 (last point): F*d^2-1 = {Fb[-1]*dl[-1]**2-1:+.4e},  Pi*d/C_Pi-1 = {Pb[-1]*dl[-1]/CPi-1:+.4e}")
print(f"    eta = 0.05 : F*d^2-1 = {Fb[-46]*dl[-46]**2-1:+.4e},  Pi*d/C_Pi-1 = {Pb[-46]*dl[-46]/CPi-1:+.4e}")

print("\nC5. n_b at delta/Gth = 1.5e-2 for Verhagen-like scaling claim (~1e3 with n+1=101)")
eta = 1.5e-2
nb_claim = kD * (nD + 1) * np.sqrt(kD * gD) / (4 * (kD + gD) * eta * GthD) * GthD  # careful: formula uses delta
nb_claim = kD * (nD + 1) * np.sqrt(kD * gD) / (4 * (kD + gD) * (eta * GthD))
print(f"    n_b = {nb_claim:.3e}  (order 1e3 expected)")
# same with the Verhagen rates
kv, gv = 0.6 / 3.7, 0.1 / 3.7e6
nbv = kv * 101 * np.sqrt(kv * gv) / (4 * (kv + gv) * (eta * np.sqrt(kv * gv) / 2))
print(f"    Verhagen rates, n_b = {nbv:.3e}")
