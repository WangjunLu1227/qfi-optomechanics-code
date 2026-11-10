# Numerical and symbolic code for

**Universal Quadratic Trade-off between Entropy Production and Quantum Fisher Information
near an Optomechanical Parametric Instability**

This repository contains the symbolic derivations, the high-precision numerical checks and the
plotting scripts behind every quantitative statement of the paper. There is no experimental data
set: all reported numbers follow from the closed-form expressions of the paper, evaluated with the
parameters stated there, and every figure can be regenerated from the scripts below.

## Model and conventions

Linearized two-mode Gaussian dynamics (cavity mode `d`, mechanical mode `b`), rotating frame, RWA:

* blue sideband (two-mode squeezing): threshold `G_th = sqrt(kappa*gamma)/2`
* red sideband (beam splitter), included for contrast
* quadrature order `u = (q_d, p_d, q_b, p_b)`, `[q,p] = i`, vacuum `sigma = I/2`, `hbar = 1`
* `dsigma/dt = A sigma + sigma A^T + D`, `D = diag(kappa/2, kappa/2, gamma*(n_th+1/2), gamma*(n_th+1/2))`
* entropy production rate: Belenchia, Mancino, Landi & Paternostro, npj Quantum Inf. **6**, 97 (2020)
* Gaussian QFI: Banchi, Braunstein & Pirandola, PRL **115**, 260501 (2015)

Default parameters unless stated otherwise: `kappa = 0.1`, `gamma = 1e-4`, `n_th = 100`.

## Layout

```
.
├── symbolic/                        exact computer algebra (SymPy)
│   ├── analytic_ss.py               closed-form steady state, near-threshold expansions
│   ├── proof_Pi_terms.py            term-by-term divergence of Pi and the coefficient C_Pi
│   ├── proof_universality.py        reduction of F*delta^2 -> 1 (universal coefficient)
│   └── rank2_structure.py           rank-two structure of Sigma, survival beyond the RWA
├── validation/                      numerical verification (mpmath / SciPy)
│   ├── verify_equations.py          every displayed equation tested as a hypothesis
│   ├── verify_all_results.py        three independent methods: SymPy, 60-digit mpmath, double
│   ├── verify_quoted_numbers.py     every number quoted in the text
│   ├── verify_beyond_rwa.py         threshold and critical eigenvalues without the RWA
│   └── verify_von_neumann_entropy.py von Neumann vs Wigner entropy near threshold
└── figures/
    ├── compute_sweep_data.py        generates sweep_data.csv (60 points x 2 sidebands)
    ├── sweep_data.csv               raw sweep data used by both figures
    ├── make_fig_tradeoff.py         Figure 1 (F_G, Pi, n_b; parametric collapse)
    └── make_fig_ratio.py            Figure 2 (convergence to the universal asymptotes)
```

## What each script checks

| script | content | where in the paper | runtime |
|---|---|---|---|
| `symbolic/analytic_ss.py` | exact steady state, expansions of `n_d`, `n_b`, `Pi`, symplectic eigenvalues, `d sigma/dG`; mpmath QFI coefficient with a NumPy cross-check | Eqs. (11)–(13), (18), (19), Appendix A | ~50 s |
| `symbolic/proof_Pi_terms.py` | the `1/delta` divergence of `Pi` is carried entirely by the second term; exact `C_Pi` | Eqs. (18), (27)–(29), Sec. 5 | ~3 s |
| `symbolic/proof_universality.py` | `F_G delta^2 -> 1` for several `(r, n_th)`; the coefficient is parameter-free | Eq. (23), Appendix A.3 | ~2 s |
| `symbolic/rank2_structure.py` | `a*b - c^2 = 0` for an arbitrary diagonal `D`; non-RWA counter-rotating corrections | Eq. (24), Appendix A.3 | ~3 s |
| `validation/verify_equations.py` | 35 checks, one per displayed equation (spectrum, closed forms, asymptotics, `C_Pi`, trade-off, `c_1`) | whole paper, Appendix A | ~1 s |
| `validation/verify_all_results.py` | A1–A10 SymPy, B1–B6 60-digit mpmath, C1–C5 data claims: `c_1 = 0.79`, `max F_red = 4.737e5`, fitted exponents | Table 1, Appendix B, Sec. 6 | ~2 s |
| `validation/verify_quoted_numbers.py` | nine parameter combinations with `F_G delta^2 -> 1` (worst deviation 1.8e-5); trade-off constant for the Verhagen-2012 parameters | Eq. (23), Sec. 6 | ~1 s |
| `validation/verify_beyond_rwa.py` | drift-matrix eigenvalues without the RWA; correction to `G_th` | Sec. 6, Appendix B | ~1 s |
| `validation/verify_von_neumann_entropy.py` | `S_vN` and `S_W` both grow as `ln(1/delta)` with unit slope | Appendix C | <1 s |
| `figures/compute_sweep_data.py` | regenerates `sweep_data.csv` | data for Figures 1–2 | ~3 s |
| `figures/make_fig_tradeoff.py` | Figure 1, no fitted power laws, analytic asymptote only | Figure 1 | ~2 s |
| `figures/make_fig_ratio.py` | Figure 2, rescaled quantities approaching unity | Figure 2 | ~2 s |

Every script prints what it tests and the numerical difference from the closed form; none of them
requires arguments or input files other than `figures/sweep_data.csv`.

## Requirements and usage

Python 3.10 or newer:

```
pip install -r requirements.txt
```

Run any script from the repository root, for example

```
python symbolic/proof_Pi_terms.py
python validation/verify_all_results.py
python figures/compute_sweep_data.py
python figures/make_fig_tradeoff.py
```

The two figure scripts write `fig_tradeoff.pdf/png` and `fig_ratio.pdf/png` next to themselves.
`compute_sweep_data.py` overwrites `sweep_data.csv`; the file shipped here is reproduced
byte-for-byte, so the figures can be regenerated without rerunning the sweep.

## Reproducibility notes

* All comparisons are done in 40–60 digit arithmetic (`mpmath`) or exact rational arithmetic
  (SymPy); double precision is used only where explicitly stated.
* `symbolic/analytic_ss.py` is the slowest script (~50 s) because it carries out the symbolic
  expansions; the remaining scripts finish in a few seconds.
* `verify_all_results.py` includes the parameter scan of Table 1 and repeats the checks of
  `verify_equations.py` by independent routes, so the two are expected to agree to the last digit
  reported.

## License and citation

The symbolic scripts, the numerical verification scripts and the data in this repository are
released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**;
see [`LICENSE`](LICENSE) or <https://creativecommons.org/licenses/by/4.0/>. You are free to
share and adapt the material for any purpose, including commercially, provided that you give
appropriate credit, link to the license, and indicate whether changes were made.

If you use this code, please cite the paper above.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
