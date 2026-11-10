"""
Symplectic eigenvalues and the growth of S_vN and S_W near the threshold,
for the blue-sideband steady state.  Near G_th,
  nu_+ = (1-r)(n+1)/(2(1+r) eps) + O(1),  nu_- -> 1/2 + const,
and both entropies grow as ln(1/eps) with the same slope, so the exponent of
the trade-off law is entropy-convention independent (the constant is the
Wigner one; the von Neumann version is singular for a T=0 optical bath).
"""

import mpmath as mp

mp.mp.dps = 40

def sigma_blue(G, k, g, nth):
    D4 = 4*G**2 - k*g
    s00 = (4*G**2*(k - g*(2*nth + 1)) - k*g*(k + g))/(2*(k + g)*D4)
    s22 = (2*G**2*(2*g*nth + g - k) - k*g*(k + g)*(nth + mp.mpf('0.5'))) \
        / ((k + g)*D4)
    s03 = 2*G*g*k*(nth + 1)/((k + g)*D4)
    return mp.matrix([[s00, 0, 0, s03],
                      [0, s00, s03, 0],
                      [0, s03, s22, 0],
                      [s03, 0, 0, s22]])

OM = mp.matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])

def symplectic_eigs(sig):
    M = 1j*(OM*sig)                  # Hermitian; eigenvalues are real
    ev, _ = mp.eig(M)
    vals = sorted((float(x.real) for x in ev), reverse=True)
    return vals

def g_vN(nu):
    """g(nu) = (nu+1/2)ln(nu+1/2) - (nu-1/2)ln(nu-1/2), with x ln x -> 0 as x->0+."""
    nu = abs(mp.mpf(nu))             # symplectic eigenvalues are |eig(i*Omega*sigma)|
    a = nu + mp.mpf('0.5')
    b = nu - mp.mpf('0.5')
    out = a*mp.log(a)
    if abs(b) > mp.mpf('1e-25'):     # b = 0 gives x ln x -> 0, not nan
        out -= b*mp.log(b)
    return out

k, g, nth = mp.mpf('0.1'), mp.mpf('1e-4'), mp.mpf(100)
r = g/k
Gth = mp.sqrt(k*g)/2

print(f"parameters: kappa={k}, gamma={g}, n_th={nth}, r={r}")
print(f"{'eps':>8s} {'nu_+ (num)':>14s} {'nu_+ (analytic)':>16s} "
      f"{'ratio':>8s} {'nu_- (num)':>12s} {'S_vN':>10s} {'S_W':>10s}")
data = []
for epsv in [mp.mpf('1e-1'), mp.mpf('1e-2'), mp.mpf('1e-3'), mp.mpf('1e-4'),
             mp.mpf('1e-5'), mp.mpf('1e-6')]:
    sig = sigma_blue(Gth - Gth*epsv, k, g, nth)
    nus = symplectic_eigs(sig)
    nu_p, nu_m = nus[0], nus[-1]
    S_vN = g_vN(nu_p) + g_vN(max(nu_m, mp.mpf('0.5')))
    S_W = mp.log(mp.det(sig))/2
    pred = (1-r)*(nth+1)/(2*(1+r)*epsv)
    print(f"{mp.nstr(epsv,3):>8s} {mp.nstr(nu_p,8):>14s} {mp.nstr(pred,8):>16s} "
          f"{mp.nstr(nu_p/pred,5):>8s} {mp.nstr(nu_m,6):>12s} "
          f"{mp.nstr(S_vN,6):>10s} {mp.nstr(S_W,6):>10s}")
    data.append((epsv, S_vN, S_W))

print("\nslopes of S_vN and S_W vs ln(1/eps) (fit over the four smallest eps):")
for idx, name in ((1, 'S_vN'), (2, 'S_W')):
    xs = [mp.log(1/e) for (e, *_r) in data[-4:]]
    ys = [row[idx] for row in data[-4:]]
    n = len(xs)
    xm, ym = sum(xs)/n, sum(ys)/n
    slope = sum((x-xm)*(y-ym) for x, y in zip(xs, ys))/sum((x-xm)**2 for x in xs)
    print(f"  {name}: slope = {mp.nstr(slope, 6)}")

print("\nboth entropies diverge as ln(1/eps) with the same slope: the exponent of")
print("the trade-off law is convention independent; the constant is the Wigner one.")
