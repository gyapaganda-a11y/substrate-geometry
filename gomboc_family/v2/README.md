# gomboc_family/v2: verification suite for arXiv:2604.17120 (v3)

Scripts and results behind *Sloan's Analytical Gömböc: Strict Convexity Narrows the Admissible β Range* (arXiv:2604.17120, v3, the amendment of record). The paper source is `main.tex`. Licence: see the repository root.

## What is here

| Path | Contents |
|---|---|
| `scripts/verify_K_analytical.py` | Gaussian curvature of the radial surface, symbolic derivatives, dense (θ, φ) grid with a resolution sweep |
| `scripts/verify_R_critpoints_analytical.py` | critical points of the radial function (root-finding from a grid of starts, Hessian classification, Morse-Euler check) |
| `scripts/two_maximizer_test.py` | the two-maximizer behavioural check at β = 0.05 |
| `scripts/crosscheck_implicit_K.py` | independent second implementation (implicit-surface curvature, no shared code) used for the v3 cross-check; covers Sloan's Eq. 12 and Eq. 16 shapes |
| `scripts/crosscheck_poles_and_mathworld.py` | compares the least-radius-point curvature with the exact values in the Wolfram MathWorld entry for the Gömböc, and shows the pole behaviour of the curvature (Gömböc 1 is not C² at the poles) |
| `scripts/identity_based_classifier.py`, `re_classify_catalog.py` | classification of the v1 catalog members (see the caveat below) |
| `results/` | JSON outputs of the scripts |
| `claims_to_scripts.md` | which script and result file back each quantitative claim in `main.tex`, plus the reproduction status |
| `preconditions_check.md` | preconditions of the theorems the paper applies, and the script that tests each |
| `../results/catalog.json` | the v1 catalog parameters; the classifier reads it. Twelve of the thirteen members are retired by the paper; the file is kept because the classifier needs it |

## Quick reproduction

Python 3.11+, numpy, scipy, sympy. From this directory (scripts import their siblings, so run them as files):

```bash
python scripts/verify_K_analytical.py --family sloan --beta 0.05 --out results/my_K_b005.json
# expect K_min = -0.569048 at 640x1280, K < 0 over 4.01% of the surface
python scripts/verify_K_analytical.py --family sloan-phase --beta 0.023149 --coef 0.234433 --k 1 --out results/my_K_phase1.json
# expect K_min = +0.090691
python scripts/verify_R_critpoints_analytical.py --family sloan-phase --beta 0.023149 --coef 0.234433 --k 1
# expect 2 critical points (1 maximum, 1 minimum)
python scripts/two_maximizer_test.py --family sloan --beta 0.05
python scripts/crosscheck_implicit_K.py --out results/my_crosscheck.json
```

Use new `--out` names: `identity_based_classifier.py` overwrites its own stored output if run in place, and the other scripts overwrite whatever you point `--out` at.

## What has and has not been re-run

The five commands above were re-run from a clean copy on 2026-11-02 and matched the stored results (details in `claims_to_scripts.md`). The one-shot regeneration of the full catalog table with `identity_based_classifier.py` (about 45 minutes, and it imports the earlier numerical gate) has not been re-run against the final scripts; the per-body values in the table come from the analytical scripts above.

## Related data

Meshes and analysis results: [10.5281/zenodo.20674394](https://doi.org/10.5281/zenodo.20674394).

Sloan's source paper: arXiv:2306.14914. This work's companion papers: arXiv:2604.12238, arXiv:2604.17095, arXiv:2608.11213.
