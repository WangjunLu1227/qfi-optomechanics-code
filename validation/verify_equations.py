"""Numerical checks of the displayed equations (double precision, mpmath where
noted)."""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
import mpmath as mp

mp.mp.dps = 40

# ------------------------------------------------------------------ model
def A_H(side, G, kappa, gam):
    A = np.zeros((4, 4))
    s = 1.0 if side == "red" else -1.0
    A[0, 3] = s * G    # qd <- pb  (blue -G, red +G)
    A[1, 2] = -G       # pd <- qb  (both -G)
    A[2, 1] = s * G    # qb <- pd  (blue -G, red +G)
    A[3, 0] = -G       # pb <- qd  (both -G)
    return A

def A_full(side, G, kappa, gam):
    return A_H(side, G, kappa, gam) - np.diag([kappa/2, kappa/2, gam/2, gam/2])

def D_mat(kappa, gam, nth):
    return np.diag([kappa/2, kappa/2, gam*(nth+0.5), gam*(nth+0.5)])

def ss(side, G, kappa, gam, nth):
    A = A_full(side, G, kappa, gam)
    S = solve_continuous_lyapunov(A, -D_mat(kappa, gam, nth))
    return 0.5*(S+S.T)

def Pi_rate(sig, D, kappa, gam):
    Airr = np.diag([-kappa/2, -kappa/2, -gam/2, -gam/2])
    t1 = 2*np.trace(Airr)
    t2 = 2*np.trace(Airr.T @ np.linalg.inv(D) @ Airr @ sig)
    t3 = 0.5*np.trace(np.linalg.inv(sig) @ D)
    return t1, t2, t3

def qfi_gaussian(sig, dsig, Om=None):
    n = sig.shape[0]
    if Om is None:
        o = np.array([[0, 1], [-1, 0]])
        Om = np.kron(np.eye(2), o) if n == 4 else o
    M = np.kron(sig, sig) - np.kron(Om, Om)/4.0
    w = np.linalg.svd(M, compute_uv=False)
    tol = max(M.shape) * np.finfo(float).eps * w.max()
    Minv = np.linalg.pinv(M, rcond=1e-13)
    return 0.5 * float(np.kron(np.eye(1), dsig).reshape(-1) @ Minv @ dsig.reshape(-1))

def symplectic_eigs(sig):
    o = np.array([[0, 1], [-1, 0]])
    Om = np.kron(np.eye(2), o)
    M = 1j * (Om @ sig)
    ev = np.linalg.eigvals(M)
    ev = np.real_if_close(ev).real
    pos = np.sort(ev[ev > 0])
    return pos

def ok(name, val, ref, tol):
    flag = "ok" if abs(val - ref) <= tol else "MISMATCH"
    print(f"  {name}: got {val:.6e}, expect {ref:.6e}  [{flag}]")

kappa, gam, nth = 0.1, 1e-4, 100.0
Gth = np.sqrt(kappa*gam)/2

print("-- spectrum formula")
for G in [0.3*Gth, 0.9*Gth]:
    ev = np.linalg.eigvals(A_full("blue", G, kappa, gam))
    lp = -((kappa+gam)/4) + np.sqrt(16*G**2+(kappa-gam)**2)/4
    lm = -((kappa+gam)/4) - np.sqrt(16*G**2+(kappa-gam)**2)/4
    ok("lambda_+", max(ev.real), lp, 1e-12)
    cnt_p = np.sum(np.abs(ev - lp) < 1e-12); cnt_m = np.sum(np.abs(ev - lm) < 1e-12)
    print(f"     degeneracy at G={G/Gth:.2f}Gth:  n(lambda+)={cnt_p}, n(lambda-)={cnt_m}")

print("\n-- critical-eigenvalue slope")
for (k2, g2) in [(kappa, gam), (1.0, 1.0)]:
    gth2 = np.sqrt(k2*g2)/2
    for d in [1e-6*gth2, 1e-7*gth2]:
        ev = np.linalg.eigvals(A_full("blue", gth2-d, k2, g2))
        lp = max(ev.real)
        s_exact = 2*np.sqrt(k2*g2)/(k2+g2)        # paper:  lambda+ = -2 sqrt(kg)/(k+g) * delta
        ok(f"kappa={k2},gamma={g2},d={d:.1e}  -lambda+/d", s_exact, -lp/d, 1e-3*np.abs(lp/d))

print("\n-- closed-form steady state")
def closed_forms(G, k2, g2, n2):
    D4 = 4*G**2 - k2*g2
    sqd = (4*G**2*(k2 - g2*(2*n2+1)) - k2*g2*(k2+g2)) / (2*(k2+g2)*D4)
    sqb = (2*G**2*(g2*(2*n2+1)-k2) - k2*g2*(k2+g2)*(n2+0.5)) / ((k2+g2)*D4)
    sx  = 2*G*g2*k2*(n2+1) / ((k2+g2)*D4)
    return sqd, sqb, sx
for (k2, g2, n2, fG) in [(0.1, 1e-4, 100.0, 0.9), (1.0, 1.0, 0.0, 0.8), (2.0, 0.5, 3.0, 0.6)]:
    gth2 = np.sqrt(k2*g2)/2; G = fG*gth2
    S = ss("blue", G, k2, g2, n2)
    sqd, sqb, sx = closed_forms(G, k2, g2, n2)
    ok(f"s_qdqd k={k2},g={g2},n={n2}", S[0, 0], sqd, 1e-9*abs(sqd))
    ok(f"s_qbqb k={k2},g={g2},n={n2}", S[2, 2], sqb, 1e-9*abs(sqb))
    ok(f"s_qdpb k={k2},g={g2},n={n2}", S[0, 3], sx, 1e-9*abs(sx))
    print(f"     sigma_pdqb sign check: S[1,2]={S[1,2]:+.4e}, S[0,3]={S[0,3]:+.4e}, equal: {np.isclose(S[1,2],S[0,3])}")
    print(f"     also pdpd={S[1,1]:.6e} (={S[0,0]:.6e}? {np.isclose(S[1,1],S[0,0])}), "
          f"p_b={S[3,3]:.6e} vs q_b {S[2,2]:.6e} equal: {np.isclose(S[3,3],S[2,2])}, "
          f"s_pdqb={S[1,2]:.6e} vs s_qdpb {S[0,3]:.6e} equal: {np.isclose(S[1,2],S[0,3])}")

print("\n-- limiting block entries a, b, c")
a = gam**1.5*np.sqrt(kappa)*(nth+1)/(4*(kappa+gam))
b = np.sqrt(gam)*kappa**1.5*(nth+1)/(4*(kappa+gam))
c = -kappa*gam*(nth+1)/(4*(kappa+gam))
for drel in [1e-6, 1e-7, 1e-8, 1e-9]:
    d = drel*Gth
    S = ss("blue", Gth-d, kappa, gam, nth)
    ra = abs(S[0,0]*d/a-1); rb = abs(S[2,2]*d/b-1); rc = abs(S[0,3]*d/c-1)
    print(f"     d={drel:.0e}Gth: rel.err a={ra:.2e} b={rb:.2e} c={rc:.2e}")
d = 1e-6*Gth          # double precision loses ~6 digits at 1e-9 Gth; 1e-6 is safe
S = ss("blue", Gth-d, kappa, gam, nth)
ok("delta*s_qdqd -> a", S[0, 0]*d, a, 1e-5*abs(a))
ok("delta*s_qbqb -> b", S[2, 2]*d, b, 1e-5*abs(b))
ok("delta*s_qdpb -> c", S[0, 3]*d, c, 1e-5*abs(c))
print(f"     ab - c^2 = {a*b-c*c:.3e}  (must be 0)")

print("\n-- Sigma = (tau_c/2) P_c D P_c^T")
A = A_full("blue", Gth-d, kappa, gam)
w, V = np.linalg.eig(A)
idx = np.argsort(w.real)[-2:]          # two critical eigenvalues
Pc = V[:, idx] @ np.linalg.inv(V)[idx, :]   # oblique spectral projector
D = D_mat(kappa, gam, nth)
Sig_mech = Pc @ D @ Pc.T
tau_c = (kappa+gam)/(2*np.sqrt(kappa*gam))
Sdig = S*d   # = Sigma numeric
pred = (tau_c/2)*Sig_mech
ok("  [qd,qd]", pred[0, 0], Sdig[0, 0], 1e-4*abs(Sdig[0, 0]))
ok("  [qb,qb]", pred[2, 2], Sdig[2, 2], 1e-4*abs(Sdig[2, 2]))
ok("  [qd,pb]", pred[0, 3], Sdig[0, 3], 1e-4*abs(Sdig[0, 3]))

print("\n-- entropy-production asymptotics")
for d in [1e-6*Gth, 1e-7*Gth, 1e-8*Gth]:
    S = ss("blue", Gth-d, kappa, gam, nth)
    D = D_mat(kappa, gam, nth)
    t1, t2, t3 = Pi_rate(S, D, kappa, gam)
    Pi = t1+t2+t3
    CPi = gam**1.5*np.sqrt(kappa)*(nth+1)**2/((1+gam/kappa)*(2*nth+1))
    ok(f"d*Pi (d={d/Gth:.0e}Gth)", d*Pi, CPi, 1e-5*CPi)
    print(f"     terms: t1={t1:.4f} (O(1)), t2*d={t2*d:.6e}, t3={t3:.4f} (O(1))  -> divergence in t2: {abs(t2*d-CPi)<1e-3*CPi}")
CPi = gam**1.5*np.sqrt(kappa)*(nth+1)**2/((1+gam/kappa)*(2*nth+1))
print(f"     closed form C_Pi = {CPi:.6e};  2*kappa*a + gamma*b/(n+1/2) = {2*kappa*a + gam*b/(nth+0.5):.6e}")

print("\n-- arbitrary diffusion")
rng = np.random.default_rng(7)
for i in range(3):
    dd, db = rng.uniform(0.01, 2), rng.uniform(0.01, 2)
    Dg = np.diag([dd, dd, db, db])
    A = A_full("blue", Gth-d, kappa, gam)
    Sg = 0.5*(solve_continuous_lyapunov(A, -Dg) + solve_continuous_lyapunov(A, -Dg).T)
    t1, t2, t3 = Pi_rate(Sg, Dg, kappa, gam)
    CPg = np.sqrt(kappa*gam)*(dd*gam+db*kappa)/(4*(kappa+gam))*(kappa/dd+gam/db)
    ok(f"random d=({dd:.3f},{db:.3f})  d*Pi", d*(t1+t2+t3), CPg, 1e-5*CPg)

print("\n-- F/Pi^2")
for d in [1e-6*Gth, 1e-7*Gth]:
    S = ss("blue", Gth-d, kappa, gam, nth)
    D = D_mat(kappa, gam, nth)
    Pi = sum(Pi_rate(S, D, kappa, gam))
    h = 1e-9*Gth
    Sp = ss("blue", Gth-d+h, kappa, gam, nth); Sm = ss("blue", Gth-d-h, kappa, gam, nth)
    F = qfi_gaussian(S, (Sp-Sm)/(2*h))
    ref = (1+gam/kappa)**2*(2*nth+1)**2/(kappa*gam**3*(nth+1)**4)
    ok(f"F/Pi^2 (d={d/Gth:.0e}Gth)", F/Pi**2, ref, 1e-3*ref)

print("\n-- subleading c1")
for (k2, g2, n2) in [(0.1, 1e-4, 100.0), (1.0, 1.0, 5.0), (0.3, 3e-3, 50.0)]:
    gth2 = np.sqrt(k2*g2)/2
    h = 1e-9*gth2; d = 1e-6*gth2
    S = ss("blue", gth2-d, k2, g2, n2)
    Sp = ss("blue", gth2-d+h, k2, g2, n2); Sm = ss("blue", gth2-d-h, k2, g2, n2)
    F = qfi_gaussian(S, (Sp-Sm)/(2*h))
    c1 = (1-F*d**2)/(d/gth2)
    print(f"     kappa={k2}, gamma={g2}, n={n2}:  c1 = {c1:.4f}")

print("\n-- red sideband")
for G in [0.5*np.abs(kappa-gam)/4*1.1, 3*np.abs(kappa-gam)/4]:
    ev = np.linalg.eigvals(A_full("red", G, kappa, gam))
    if G > abs(kappa-gam)/4:
        ref = -(kappa+gam)/4 + 1j*np.sqrt(G**2-(kappa-gam)**2/16)
        ok(f"red Re(lambda) G={G:.2e}", ev.real.max(), ref.real, 1e-10)
        ok(f"red Im(lambda) G={G:.2e}", np.abs(ev.imag).max(), abs(ref.imag), 1e-10)
G = 50.0
Sr = ss("red", G, kappa, gam, nth)
nd = (Sr[0, 0]+Sr[1, 1])/2 - 0.5
ok("red n_d -> gamma*nth/(kappa+gamma)", nd, gam*nth/(kappa+gam), 1e-4*gam*nth/(kappa+gam))
D = D_mat(kappa, gam, nth)
t1, t2, t3 = Pi_rate(Sr, D, kappa, gam)
CPred = 4*gam*kappa*nth**2/(2*gam*nth+gam+2*kappa*nth+kappa)
ok("red Pi(G=50) -> closed form", t1+t2+t3, CPred, 1e-3*CPred)

print("\n-- symplectic eigenvalue near threshold")
for d in [1e-7*Gth, 1e-8*Gth]:
    S = ss("blue", Gth-d, kappa, gam, nth)
    nu = symplectic_eigs(S)
    eta = d/Gth
    ref = (1-gam/kappa)*(nth+1)/(2*(1+gam/kappa))/eta
    ok(f"nu_+ (d={d/Gth:.0e}Gth)", nu[1], ref, 1e-4*ref)
    print(f"     nu_- = {nu[0]:.6f} (should -> 0.5)")

print("\n-- n_b near threshold")
for d in [1e-5*Gth, 1e-6*Gth]:
    S = ss("blue", Gth-d, kappa, gam, nth)
    nb = (S[2, 2]+S[3, 3])/2 - 0.5
    bcoef = np.sqrt(gam)*kappa**1.5*(nth+1)/(4*(kappa+gam))
    print(f"     d={d/Gth:.0e}Gth:  d*n_b={nb*d:.6e}  vs  b = {bcoef:.6e}")

