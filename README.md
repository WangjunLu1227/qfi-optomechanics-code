# qfi-optomechanics-code

Code for the paper *Universal Quadratic Trade-off between Entropy Production
and Quantum Fisher Information near an Optomechanical Parametric Instability*.

There is no experimental data set: every number in the paper follows from the
closed-form expressions derived there, evaluated with the parameters stated in
the paper. The scripts below reproduce those numbers and both figures.

## Model

Linearized two-mode Gaussian dynamics (cavity mode `d`, mechanical mode `b`),
rotating frame, RWA. Quadratures `u = (q_d, p_d, q_b, p_b)`, `[q,p] = i`,
vacuum `sigma = I/2`, `hbar = 1`:

    dsigma/dt = A sigma + sigma A^T + D
    D = diag(kappa/2, kappa/2, gamma(n_th+1/2), gamma(n_th+1/2))
    G_th = sqrt(kappa*gamma)/2   (blue sideband, two-mode squeezing)

Entropy production rate after Belenchia, Mancino, Landi & Paternostro,
npj Quantum Inf. 6, 97 (2020); Gaussian QFI after Banchi, Braunstein &
Pirandola, PRL 115, 260501 (2015).
Default parameters: `kappa = 0.1`, `gamma = 1e-4`, `n_th = 100`.

## Layout

- `symbolic/analytic_ss.py` - closed-form steady state; near-threshold
  expansions of `n_d`, `n_b`, `Pi`, symplectic eigenvalues, `d sigma/dG`,
  and the QFI coefficient (mpmath). Slowest script, ~50 s.
- `symbolic/proof_Pi_terms.py` - the `1/delta` divergence of `Pi` is carried
  entirely by the second term `2 Tr[A_irr^T D^-1 A_irr sigma]`; exact `C_Pi`.
- `symbolic/proof_universality.py` - `F delta^2 -> 1`, evaluated exactly for
  several `(r, n_th)`.
- `symbolic/rank2_structure.py` - `a*b - c^2 = 0` for an arbitrary diagonal
  `D`; threshold and scaling beyond the RWA.
- `validation/verify_equations.py` - the displayed equations, double precision.
- `validation/verify_all_results.py` - the same quantities in 60-digit mpmath,
  plus the numbers quoted in the text (`c_1`, max `F_red`, fitted exponents).
- `validation/verify_beyond_rwa.py` - threshold shift without the RWA.
- `validation/verify_von_neumann_entropy.py` - `S_vN` and `S_W` versus
  `ln(1/eps)` near the threshold.
- `figures/compute_sweep_data.py` - generates `sweep_data.csv`.
- `figures/make_fig_tradeoff.py`, `figures/make_fig_ratio.py` - Figures 1 and 2.

Every script runs without arguments and prints what it tests. `sweep_data.csv`
is included, so the figures can be redrawn without rerunning the sweep.

## Usage

    pip install -r requirements.txt

    python symbolic/proof_Pi_terms.py
    python validation/verify_all_results.py
    python figures/make_fig_tradeoff.py

Python 3.10 or newer. The figure scripts write `fig_tradeoff.*` and
`fig_ratio.*` next to themselves.

## License

CC BY 4.0, see [LICENSE](LICENSE). If you use the code, please cite the paper.
