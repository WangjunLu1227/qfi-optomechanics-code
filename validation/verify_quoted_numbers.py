# -*- coding: utf-8 -*-
"""Cross-check every quantitative value quoted in the text."""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
import mpmath as mp
mp.mp.dps = 60

print("=== (1) nine-combination check: F*delta^2 at eta=1e-6 ===")
def sigma_blue(G, k, g, nth):
    D4 = 4*G**2 - k*g
    s00 = (4*G**2*(k - g*(2*nth+1)) - k*g*(k+g))/(2*(k+g)*D4)
    s22 = (2*G**2*(2*g*nth+g-k) - k*g*(k+g)*(nth+mp.mpf('0.5')))/((k+g)*D4)
    s03 = 2*G*g*k*(nth+1)/((k+g)*D4)
    return mp.matrix([[s00,0,0,s03],[0,s00,s03,0],[0,s03,s22,0],[s03,0,0,s22]])
def kron(a, b):
    m, n = a.rows, a.cols
    p, q = b.rows, b.cols
    out = mp.matrix(m*p, n*q)
    for i in range(m):
        for j in range(n):
            for u in range(p):
                for v in range(q):
                    out[i*p+u, j*q+v] = a[i, j]*b[u, v]
    return out
OM = mp.matrix([[0,1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]])
def pinv_sym(M):
    lam, V = mp.eigsy(M)
    mx = max(abs(x) for x in lam)
    tol = mx * mp.mpf(10) ** (-mp.mp.dps + 10)
    out = mp.matrix(M.rows, M.cols)
    for i in range(M.rows):
        if abs(lam[i]) > tol:
            col = mp.matrix(V[:, i])
            out += lam[i] ** (-1) * col * col.T
    return out

def qfi(sig, dsig):
    M = kron(sig,sig) - kron(OM,OM)/4
    v = mp.matrix([[dsig[i,j]] for i in range(4) for j in range(4)])
    return mp.mpf('0.5')*(v.T*pinv_sym(M)*v)[0]
worst = 0
for lr in ('1e-2','1e-3','1e-4'):
    for nth in (10, 100, 1000):
        k = mp.mpf('0.1'); g = k*mp.mpf(lr); nth = mp.mpf(nth)
        Gth = mp.sqrt(k*g)/2
        eta = mp.mpf('1e-6')
        G = Gth*(1-eta); d = Gth-G
        s = sigma_blue(G,k,g,nth)
        h = mp.mpf('1e-9')*Gth
        ds = (sigma_blue(G+h,k,g,nth)-sigma_blue(G-h,k,g,nth))/(2*h)
        F = qfi(s,ds)
        val = F*d**2
        worst = max(worst, abs(val-1))
        print(f"  r={lr:>6s} n={mp.nstr(nth,5):>5s}:  F*delta^2 = {mp.nstr(val,8)}")
print(f"  worst |dev| = {mp.nstr(worst,3)}")

print("\n=== (2) pure-state check ratio ===")
print("  (already printed by demo_entropy_qfi.py; see Validation 1)")

print("\n=== (3) trade-off constant for Verhagen-2012 rates ===")
om = 1.0
k = 0.6/3.7*om          # kappa/2pi=0.6MHz, w_m/2pi=3.7MHz
g = 0.1/3.7e6*om        # gamma/2pi=0.1Hz
for nth in (100.0,):
    C = (1+g/k)**2*(2*nth+1)**2/(k*g**3*(nth+1)**4)
    print(f"  kappa/w={k:.4f} gamma/w={g:.3e} n={nth}:  F/Pi^2 = {C:.4e}  (units 1/w_m)")
print("  kappa/gamma =", k/g)
