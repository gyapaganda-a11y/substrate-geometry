# Substrate Geometry Research Program

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20674508.svg)](https://doi.org/10.5281/zenodo.20674508)

**Computational Validation of the Oloid as a Local Optimum in the Developable Roller Family**

A formal framework for evaluating geometric primitives as engineering substrates, classified by their operational invariants rather than symmetry groups.

## Research Program

This repository ships the seven frozen oracle artifacts behind **Paper I** (oloid validation, arXiv:2604.12238). The broader program covers four published or in-submission papers across rolling primitives, thermal substrates, methodology hardening, and mono-monostatic / constant-width primitives.

- **Paper I:** *Computational Validation of the Oloid as a Local Optimum in the Developable Roller Family.* arXiv:2604.12238. Backed by the seven oracles in this repo.
- **Paper II (in build):** *Formal Validation of the Zero-Mean-Curvature Invariant for Heat Exchanger Surface Design.* Gyroid TPMS application.
- **Paper III (methodology hardening):** Oracle audit framework, addressing discrete-K bias and surface-class generalization.
- **Paper IV (in submission):** Mono-monostatic catalog work extending M. L. Sloan's 2023 analytical Gömböc parameterization, with 13 members reported.

Full author page on arXiv: <https://arxiv.org/a/couey_v_1.html>

Program hub and broader research portfolio: <https://deepsynthesis.org/work/physics>

Threads on individual papers and replication observations: <https://x.com/VincentCouey>

Replication of any catalog member, independent extension to a new geometry, or addition of a new invariant is welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for scope, threshold, and coauthorship policy.

## Key Results

### Oloid Invariant Vector (5 independent physics measurements)

| Dimension | Oloid | Cylinder | Ratio | Tier |
|-----------|-------|----------|-------|------|
| CDS (contact time) | 8.20 × 10⁻⁷ | 4.75 × 10⁻⁵ | **58×** | Tier 1 |
| SDS (Hertz stress) | 8.07 × 10⁻⁷ | 4.68 × 10⁻⁵ | **58×** | Tier 1 |
| TDS (thermal) | 7.77 × 10⁻⁷ | 5.28 × 10⁻⁵ | **68×** | Tier 1 |
| WDS (wear volume) | 7.77 × 10⁻⁷ | 5.28 × 10⁻⁵ | **68×** | Tier 1 |
| FDS (fatigue) | 2.42 × 10⁻⁶ | ∞ | **∞** | Tier 2 |

**Two-tier structure:** First-order metrics (CDS, SDS, TDS, WDS) cluster at ~8×10⁻⁷ — the contact time invariant transfers losslessly to stress, thermal, and wear distributions. The nonlinear fatigue metric diverges due to Basquin S-N amplification but the oloid still dominates.

### Second Validated Primitive: Meissner Body

| Dimension | Oloid | Meissner T2 | Comparison |
|-----------|-------|-------------|------------|
| CDS | 8.20 × 10⁻⁷ | 2.76 × 10⁻⁶ | Oloid 3.4× better |
| CWS (constant width) | N/A | 1 × 10⁻⁸ | Invariant satisfied |

Contact distribution and constant width are **independent invariants** — the framework correctly separates these primitives.

## Repository Structure

```
# Core Oracles (frozen — cited in preprint)
contact_oracle.py              # Artifact 01 — Approximate CDS oracle
parametric_search.py           # Artifact 02 — Parametric search (1,430 genomes)
rigidbody_oracle.py            # Artifact 03 — Rigid-body Euler-equation CDS oracle
hertz_oracle.py                # Artifact 04 — Hertz contact pressure (SDS)
fatigue_oracle.py              # Artifact 05 — Basquin S-N fatigue (FDS)
thermal_oracle.py              # Artifact 06 — Frictional thermal (TDS)
wear_oracle.py                 # Artifact 07 — Archard wear (WDS)
parametric_search_results.json # Full 1,430-genome search results

# Generalized Oracle Runner
oracle_runner.py               # Standardized primitive scoring engine
invariants/                    # Invariant definitions + scoring functions
  registry.py                  # 5 registered invariants
  scorers.py                   # Contact distribution, constant width, H=0, etc.
modes/                         # Physics simulation modes
  registry.py                  # 6 modes (3 implemented, 3 stubs)
  simulators.py                # Rolling plane, constrained rolling, equilibrium

# Results
results/                       # Standardized JSON profiles per primitive

# Preprint
preprint/main.tex              # LaTeX source (arXiv:2604.12238)
preprint/fig_comparison.pdf    # Vector figure
preprint/fig_comparison.py     # Figure generation

# Meshes
meshes/                        # STL files for non-parametric primitives
```

## Reproducing Results

```bash
pip install trimesh numpy scipy

# Core oloid validation (7 oracles)
python contact_oracle.py
python parametric_search.py
python rigidbody_oracle.py
python hertz_oracle.py
python fatigue_oracle.py
python thermal_oracle.py
python wear_oracle.py

# Generalized oracle runner
python oracle_runner.py
```

FEniCS (optional, for Hertz FEM validation):
```bash
# WSL2/Ubuntu
sudo add-apt-repository -y ppa:fenics-packages/fenics
sudo apt install -y fenics
pip install trimesh scipy
```

## Framework

The framework classifies geometric forms by what they **guarantee under physical operation**, not by their symmetry group.

**Formal vocabulary:** Substrate Geometry, Invariant Primitive, Geometric Failure Mode, Contact Distribution Score, Pass-Gate Validation, Search Family, Invariant Vector.

**Oracle Runner:** Any watertight mesh plus invariant definition plus operating context yields a standardized primitive profile with scores across all shared dimensions and a unique invariant-specific score.

See `FRAMEWORK.md` for the complete specification.

## Dependencies

- Python 3.10+
- NumPy, SciPy, trimesh
- FEniCS (optional, for FEM validation)

## License

MIT

## Citation

> Couey, V. (2026). *Computational Validation of the Oloid as a Local Optimum in the Developable Roller Family.* arXiv:2604.12238.

Full author page and companion papers: <https://arxiv.org/a/couey_v_1.html>

## Acknowledgments

The oloid was discovered by Paul Schatz in 1929. Its developable surface properties were formally proven by Dirnböck and Stachel (1997). This work extends their lineage by providing the formal metric and computational infrastructure that confirms Schatz's finding rigorously. The mono-monostatic catalog work in paper IV extends the analytical parameterization established by M. L. Sloan (2023), itself building on the original Gömböc construction of Domokos and Várkonyi (2006) and the conjecture of V. I. Arnold (Hamburg, 1990s).

## Installation

Requires Python 3.9+ and the scientific stack:

```bash
git clone https://github.com/gyapaganda-a11y/substrate-geometry.git
cd substrate-geometry
pip install -r requirements.txt
```

Runtime dependencies: `numpy`, `scipy`, `trimesh`. (Mesh generation for some
modes additionally uses `gmsh`/FEniCS; the core oracles and the test suite do
not require them.)

## Quick start

Score any watertight mesh through the generalized runner. The example uses
`trimesh` primitives so it runs with no external mesh files:

```python
import trimesh
from oracle_runner import run_primitive

mesh     = trimesh.creation.box(extents=(1.0, 1.2, 1.5))
baseline = trimesh.creation.icosphere(subdivisions=3)

profile = run_primitive(
    mesh=mesh, invariant="contact_distribution",
    baseline_mesh=baseline, name="Box", baseline_name="Sphere",
)
print(profile.invariant_score)        # the contact-distribution score (CDS)
print(profile.vector["CDS"].value)    # the shared physics vector
```

Registered invariants are listed in `invariants/registry.py`; physics modes in
`modes/registry.py`. Adding a new invariant is writing a scoring function and a
registry entry, after which the runner discovers it by name.

## Running the tests

```bash
pip install -r requirements.txt pytest
pytest -v
```

The suite exercises the public API on analytically-known cases (discrete
Gauss-Bonnet, the equilibrium oracle, and the contact-distribution vector) and
runs in continuous integration on every push.

## Community guidelines

Contributions, bug reports, and replication observations are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for scope, thresholds, and coauthorship
policy, and open an issue for questions or problems.
