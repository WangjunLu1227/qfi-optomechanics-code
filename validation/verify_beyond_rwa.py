"""
Instability threshold of the blue-sideband model without the RWA.

Drive-frame Hamiltonian H = -Delta d'd + w b'b + G (d+d')(b+b'), Delta = +w.
In quadratures u = (q_d,p_d,q_b,p_b):  A = Omega h - diag(k/2,k/2,g/2,g/2)
with h = [[-D,0,2G,0],[0,-D,0,0],[2G,0,w,0],[0,0,0,w]], so the threshold
follows from the eigenvalues of A directly (no Floquet).  The RWA gives
G_th = sqrt(k g)/2; the correction is quantified below.
"""

import numpy as np
from scipy.optimize import brentq

OM = np.zeros((4, 4))
OM[0, 1] = 1; OM[1, 0] = -1; OM[2, 3] = 1; OM[3, 2] = -1

def drift_nonRWA(G, Delta, w, k, g):
    h = np.zeros((4, 4))
    h[0, 0] = -Delta; h[1, 1] = -Delta
    h[2, 2] = w;      h[3, 3] = w
    h[0, 2] = h[2, 0] = 2*G          # G (d+d^dag)(b+b^dag) = 2G q_d q_b
    return OM @ h - np.diag([k/2, k/2, g/2, g/2])

def max_real(G, Delta, w, k, g):
    return np.linalg.eigvals(drift_nonRWA(G, Delta, w, k, g)).real.max()

def threshold(Delta, w, k, g):
    Gmax = 4*np.sqrt(k*g) + 10*k
    lo, hi = 0.0, Gmax
    if max_real(hi, Delta, w, k, g) < 0:
        return None
    for _ in range(80):
        mid = (lo + hi)/2
        if max_real(mid, Delta, w, k, g) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi)/2

kappa, gamma = 0.1, 1e-4
Gth_rwa = np.sqrt(kappa*gamma)/2
print(f"RWA threshold:  G_th = sqrt(k g)/2 = {Gth_rwa:.6e}")
print(f"{'w_m':>7s} {'k/w_m':>7s} {'G* (non-RWA)':>13s} {'G*/G_th':>9s} "
      f"{'shift/G_th':>11s}")
rows = []
for w in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05]:
    Gs = threshold(+w, w, kappa, gamma)
    shift = (Gs - Gth_rwa)/Gth_rwa
    rows.append((w, kappa/w, Gs, shift))
    print(f"{w:>7.2f} {kappa/w:>7.2f} {Gs:>13.6e} {Gs/Gth_rwa:>9.5f} "
          f"{shift:>+11.3e}")

# check the scaling of the shift against (kappa/w)^2
print("\nshift / (k/w_m)^2:")
for (w, _, _, shift) in rows:
    print(f"  w={w:>5.2f}:  shift/(k/w)^2 = {shift/(kappa/w)**2:+.4f}")

# red sideband: no instability at all for Delta = -w
print("\nred sideband (Delta=-w): max Re eig at G=10*sqrt(kg):",
      f"{max_real(10*np.sqrt(kappa*gamma), -1.0, 1.0, kappa, gamma):+.3e} (should be < 0)")
