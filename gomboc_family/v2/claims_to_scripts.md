# Claims-to-Scripts Map — Catalog Paper v2 (FINAL)

Every quantitative claim in v2 `main.tex` maps to a runnable script in `scripts/` producing the claimed number from a fresh clone.

**Consistency-pass discipline:** grep v2 main.tex for "we comput", "we verif", "we ran", "analytical", "multi-start", "convergence", "validates", "at N directions". Each phrase must map to a row in this table or be a quoted v1 statement being corrected.

---

## §2 The Várkonyi-Gauss Identity

| Claim in main.tex | Script | Output file |
|---|---|---|
| Statement of Gauss-map identity, equation (3) | n/a (cited from Schneider 2014 §1.7) | — |
| Theorem 1 (ECS via R-critical-point count) | n/a (proof in main text) | — |
| §2.1 K_min on Sloan β=0.02 = +0.494, β=0.03 = +0.188, β=0.05 = −0.569, β=0.08 = −2.224 (Table 1) | `verify_K_analytical.py --family sloan --beta {0.02,0.03,0.05,0.08}` | `results/V_sloan_b00{2,3,5,8}_K.json` |
| §2.1 grid 640×1280, frac K<0 values | same | same |
| §2.2 two distinct maximizers at β=0.05, n=(−1,0,0) | `two_maximizer_test.py --family sloan --beta 0.05` | `results/two_maximizer_sloan_b005.json` |

## §3 Sloan's Parameterization Does Not Produce Mono-Monostatic Bodies

| Claim | Script | Output |
|---|---|---|
| Identity-based ECS classification (Table 2) | `identity_based_classifier.py` (already run) | `results/catalog_v2_identity_based.json` |
| Sloan β=0.02 has ECS=1 via identity | from above; also in V_sloan_b002_K.json + R-critpoint count via the sloan-family analytical derivation | (R-critpoints for sloan family are 2 by structure of Sloan's parameterization) |

## §4 The Analytical Verification Suite

| Claim | Script | Output |
|---|---|---|
| §4.1 sympy-symbolic K pipeline | `verify_K_analytical.py` | all V_* and V3*_K_*.json |
| §4.1 640×1280 grid, convergence to 4 dec places | resolution sweep inside verify_K_analytical | each output's `resolution_sweep` field |
| §4.2 scipy.optimize.root from 24×48 grid of starts | `verify_R_critpoints_analytical.py` | all V4*_R_critpoints_*.json |
| §4.2 Morse-Euler check M−S+m=2 | same script (reports `euler_min_minus_saddle_plus_max`) | same |

## §5 Phase-1 Verified

| Claim | Script | Output |
|---|---|---|
| §5 Table 4: K_min = +0.0907, R has 2 critpts | `verify_K_analytical.py --family sloan-phase --beta 0.023149 --coef 0.234433 --k 1` AND `verify_R_critpoints_analytical.py` same args | `results/V2_analytical_K_phase1.json` AND R-critpoint output (re-run TODO; result confirmed by count_R_critpoints.py earlier in session) |

## §6 v1 Catalog Reclassification

| Claim | Script | Output |
|---|---|---|
| Table 5 (full reclassification) | `identity_based_classifier.py` (uses verify_K_analytical + verify_R_critpoints_analytical via subprocess) — currently uses the older numerical pipeline; v2 final-form should re-run with analytical-only stack | `results/catalog_v2_identity_based.json` (numerical stack, used for diagnosis); analytical re-run recommended for the polished v2 numbers |
| Phase-2 K_min = −0.431 | `verify_K_analytical.py --family sloan-phase --beta 0.0321 --coef 0.137606 --k 2` | `results/V3a_analytical_K_phase2.json` |
| Phase-3 K_min = −0.889 | `verify_K_analytical.py --family sloan-phase --beta 0.0517 --coef -0.055157 --k 3` | `results/V3b_analytical_K_phase3.json` |
| Min-asym β=0.005 has 8 critpts (3 max, 2 min, 3 saddle) | `verify_R_critpoints_analytical.py --family sloan-radial --beta 0.005 --coef 0.05102 --f-type f3` | `results/V4_R_critpoints_minasym_b0005.json` |
| Radial-f4 primary has 6 critpts (2 max, 2 min, 2 saddle) | `verify_R_critpoints_analytical.py --family sloan-radial --beta 0.023149 --coef 0.211721 --f-type f4` | `results/V4_R_critpoints_radialf4_primary.json` |

## §9 Amendment-of-Record

The amendment-of-record section §9 of main.tex enumerates the v1→v2 changes. Each numbered amendment item references either:
- a script in v2/scripts/ that replaces v1's approach, OR
- a v1 claim being retracted (no v2 script needed; the v1 claim itself is the artifact)

| Amendment item | Script |
|---|---|
| 3 (mesh-vertex ECS oracle) | replaced by `identity_based_classifier.py` (v2 §4.3) |
| 4 (v1 §3.2 "analytical confirmation" phantom paragraph) | replaced by `verify_K_analytical.py` (v2 §4.1) |
| 5 (v1 §3.3 "global surface information" framing) | replaced by §2 (V-G identity statement) + §2.1 (precondition failure data) |
| 6 (convexity metric: mesh_vol/hull_vol → K_min) | `verify_K_analytical.py` |
| 7 (catalog reclassification) | `identity_based_classifier.py` outputs |
| 8 (r=0.9993 retraction) | n/a (no replacement; deferred for future work on polystatic subset) |
| 10 (scripts added) | this very table |

## Scripts in v2/scripts/ (final manifest)

- `verify_K_analytical.py` — analytical Gaussian curvature via sympy (V1, V2, V3, Sloan β=0.02/0.03/0.08)
- `verify_R_critpoints_analytical.py` — analytical R-critical-point counting (V4, V4b)
- `identity_based_classifier.py` — full pipeline (numerical, used for diagnosis; analytical re-run recommended)
- `h_smooth_analytical.py` — illustrative h(d) computation (kept for sanity-check; not load-bearing for v2 claims)
- `strict_convexity_gate.py` — original finite-difference K gate (kept for reference; superseded by `verify_K_analytical.py`)
- `count_R_critpoints.py` — original Nelder-Mead R-critpoint counter (kept for reference; superseded by `verify_R_critpoints_analytical.py`)
- `re_classify_catalog.py` — original numerical re-classification driver (kept for reference)
- `diagnostic_phase1_sensitivity.py` — diagnostic showing h(d) optimizer noise (referenced in §3 discussion of why mesh-vertex oracle fails)
- `two_maximizer_test.py` — supports the "two distinct maximizers at β=0.05" claim in §2.2
- `verify_identity_2d.py` — 2D verification of the Várkonyi identity (referenced in §2)

## Reproduction status (2026-11-02)

Re-run from a clean copy of this folder (Python 3.11.9); outputs matched the stored results and the paper:

- `verify_K_analytical.py --family sloan --beta 0.05`: K_min = -0.569048, K < 0 over 4.01% of the surface.
- `verify_K_analytical.py --family sloan-phase --beta 0.023149 --coef 0.234433 --k 1`: K_min = +0.090691.
- `verify_R_critpoints_analytical.py` (Phase-1): 2 critical points (1 maximum, 1 minimum).
- `verify_R_critpoints_analytical.py` (radial f4 primary): 6 critical points (2 maximum, 2 minimum, 2 saddle).
- `two_maximizer_test.py --family sloan --beta 0.05`: two distinct maximizers in direction (-1, 0, 0), matching `results/two_maximizer_sloan_b005.json`.

Not re-run: the one-shot regeneration of the full catalog table, `identity_based_classifier.py` (about 45 minutes). It reads `../../results/catalog.json` (included at `gomboc_family/results/catalog.json`), writes `results/catalog_v2_identity_based.json`, and imports the earlier numerical gate (`strict_convexity_gate.py`, `h_smooth_analytical.py`) rather than the analytical scripts above. The per-body values in the table are backed by the analytical scripts and result files listed in the section 6 rows; regenerating the whole table in one command has not been verified.

## v3 additions (2026-11-02): independent curvature cross-check

| Claim in main.tex (v3 cross-check paragraph, section "Sloan's Admissible beta Ranges...") | Script | Output file |
|---|---|---|
| Implicit-surface curvature reproduces every tabulated Gomboc-2 K_min to the three decimals shown (beta 0.01 to 0.08) | `scripts/crosscheck_implicit_K.py` | `results/CROSSCHECK_implicit_K.json` (`shapes.g2.rows`) |
| Refined minima lie at most 8e-4 below the grid minima (Gomboc 2) | same | same (`K_min_grid` minus `K_min_refined`; max 7.4e-4) |
| Every Gomboc-2 minimum at (theta, phi) = (pi/2, pi); Gomboc-1 minimum at (pi/2, 3pi/2) | same | same (`location_theta_phi`) |
| K_min changes sign at beta ~ 0.0355 (Gomboc 2), factor 0.17/0.0355 ~ 4.8 | same | same (`shapes.g2.beta_star` = 0.0355048, `overshoot_factor` = 4.788) |
| Gomboc 1: K_min = +0.086 at beta 0.03, negative from ~0.032; factor ~4.6 against 0.15 | same | same (`shapes.g1`; beta_star = 0.0322967, factor 4.644) |
| s has two critical points (one max, one min), Morse-Euler M - S + m = 2 | same | same (`g2_critical_points_of_s`) |
| Sloan states only 0.15 and 0.17 (as "appears to suffice" / "should suffice"); 0.05 appears nowhere | n/a: read Sloan arXiv:2306.14914 text layer + Eq. 12, 16 page images | (pdftotext of the PDF: decimals found are .2, 0.15, 0.17) |

| Claim in main.tex (v3 cross-check paragraph, MathWorld comparison and poles) | Script | Output file |
|---|---|---|
| Least-radius-point curvature equals the exact values in the MathWorld entry: Gomboc 1 at beta 0.15, K = -(575/64) sqrt(5/2) = -14.2055; Gomboc 2 at beta 0.17, K = -45(136 pi^2 - 25)/(2048 sqrt 2) = -20.4664 (agreement to about 1e-14) | `scripts/crosscheck_poles_and_mathworld.py` | `results/CROSSCHECK_poles_mathworld.json` (`mathworld_comparison`) |
| Gomboc 1 is not C^2 at the poles (K limit depends on phi: 0.597 to 1.498 at beta 0.03); Gomboc 2 shows no such dependence | same | same (`pole_limits_beta_0.03`) |
| MathWorld's sufficient bound 0 < beta < 1/(5 + 6 pi + 9 pi^2/4) ~ 0.0217 for Gomboc 2 | n/a: quoted from the MathWorld entry, not re-derived here | n/a |
