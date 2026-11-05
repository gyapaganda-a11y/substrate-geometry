# Part III: construction and analysis scripts

Scripts behind *Computational Construction and Engineering Evaluation of a Verified Mono-Monostatic Body*
(arXiv:2604.17095, v2). Meshes (STL) and the stored result files are archived on Zenodo:
https://doi.org/10.5281/zenodo.20674391. This directory holds the code only.

## What this directory lets you do

1. Rebuild the published bodies from their parameters (`build_published_instances.py`).
2. Regenerate the reported equilibrium-count (ECS), dynamics, ballast, application, cross-layer and
   resolution results from those meshes.
3. Check the strict convexity of the three published bodies from their analytic surfaces
   (`check_strict_convexity.py`).

## Read this first: what the paper claims and what it does not

* Only the **primary** instance (beta = 0.023149, a1 = 0.234433, phase `eta + a1 sin(eta)`) is strictly
  convex. The second and third instances reach ECS = 1 under the drainage-basin oracle but have negative
  Gaussian curvature on about 1.6% and 5% of their surface, so they are reported as oracle-level results,
  not as convex mono-monostatic bodies. Run `python check_strict_convexity.py` to see the numbers.
* The ECS oracle is a drainage-basin count on a sampled sphere. Its counts depend on the mesh resolution,
  the number of sample directions, the neighbour graph and the merge threshold (see the paper's
  validation table and resolution section), and it is not a proof.
* The verification applies to the **published parameters**. The differential-evolution script
  for the primary instance (`phase2c_optimize_ecs1.py`) is included as an exploratory tool, but it does
  **not** regenerate the stored second-stage result of the paper's optimization table, and we have not
  identified why. (The searches for the second and third instances in `phase5c_additions.py` do reproduce.) Run the
  downstream scripts on meshes built by `build_published_instances.py`, not on optimizer output.

## Requirements

Python 3.11 with `numpy`, `scipy`, `trimesh`, `matplotlib`, and `sympy` (for the convexity check).
Tested with numpy 2.4.6, scipy 1.17.1, trimesh 4.12.2. All scripts use paths relative to their own
location (`../meshes`, `../results`, `../preprint`) and create those folders.

## Run order

```
python phase1_mesh_generation.py          # Sloan Gomboc 1 and 2 meshes and the baseline bodies
python build_published_instances.py       # the published bodies, from their parameters
python phase3_dynamics_v2.py              # self-righting metrics
python phase4_density_perturbation.py     # ballast experiment
python phase5_applications_v2.py          # needs the dynamics and ballast results above
python phase6_cross_layer.py              # contact and thermal scores (simplified re-implementations)
python phase7_resolution_sensitivity.py   # ECS across mesh resolutions
python phase2_ecs_oracle.py               # ECS table for every mesh in ../meshes
python phase2b_beta_sweep_and_analytical.py   # Sloan Gomboc beta sweep
python check_strict_convexity.py
python generate_figures.py
```

| script | writes |
|---|---|
| `phase1_mesh_generation.py` | `../meshes/*.stl`, `../results/mesh_generation.json` |
| `build_published_instances.py` | the four published meshes in `../meshes` |
| `phase2_ecs_oracle.py` | `../results/ecs_scores.json` (10,000 directions, 12-neighbour graph, 2% merge) |
| `phase2b_beta_sweep_and_analytical.py` | `../results/phase2b_investigation.json` |
| `phase3_dynamics_v2.py` | `../results/dynamics.json` |
| `phase4_density_perturbation.py` | `../results/density_perturbation.json` (5,000 directions, 1% merge) |
| `phase5_applications_v2.py` | `../results/applications.json` |
| `phase5c_additions.py` | `../results/midaudit_additions.json` |
| `phase6_cross_layer.py` | `../results/cross_layer.json` |
| `phase7_resolution_sensitivity.py` | `../results/resolution_sensitivity.json` |

`superseded/` holds two earlier versions (`phase3_dynamics.py`, `phase5_applications.py`) that were replaced by
the `_v2` scripts; they are kept for the record and are not used for any reported number.

## Reproduction status

Checked on 2026-11-05 by running the scripts in an empty folder and comparing with the stored results
(numerical tolerance 1e-6 relative, timing fields ignored):

| step | result |
|---|---|
| `phase1_mesh_generation.py` | `mesh_generation.json` identical; all six STL files byte-identical to the archived ones |
| `build_published_instances.py` | primary meshes identical to the archived ones; the two other instances agree to float32 rounding (about 2e-6) |
| `phase2_ecs_oracle.py` | every ECS count identical to the stored scores (new meshes add extra rows) |
| `phase2b_beta_sweep_and_analytical.py` | the printed beta-sweep table reproduces the paper's table. The stored JSON from the original run was truncated by a serialization bug (a numpy boolean), now fixed here, so the file is regenerated rather than archived |
| `phase3_dynamics_v2.py`, `phase4_density_perturbation.py`, `phase6_cross_layer.py`, `phase7_resolution_sensitivity.py` | identical to the stored results |
| `phase5_applications_v2.py` | identical except one free-text `note` field that was reworded |
| `phase5c_additions.py` | identical to the stored results, including its two differential-evolution searches, which return the second and third instances exactly (about 15 minutes) |
| `check_strict_convexity.py` | grid and finite-difference curvatures agree: primary K_min = +0.091; second -0.431; third -0.889 |
| `phase2c_optimize_ecs1.py` | **does not reproduce** the stored second-stage result (see above) |

## Conventions that differ between scripts

* `phase2c_optimize_ecs1.py` and `phase5c_additions.py` index Fourier coefficients so that `coeffs[k]`
  multiplies `sin((k+1) eta)`. The primary body is built in `phase7_resolution_sensitivity.py` as
  `eta + a1 sin(eta)`, which is the body the stored results and meshes correspond to.
* The merge threshold is 2% of the height range in `phase2_ecs_oracle.py` and 1% in `phase3_dynamics_v2.py`,
  `phase4_density_perturbation.py`, `phase5c_additions.py` and `phase7_resolution_sensitivity.py`.

## Citation

Couey, V. W., *Computational construction and engineering evaluation of a verified mono-monostatic body*,
arXiv:2604.17095, 2026.
