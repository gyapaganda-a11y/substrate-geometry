# Substrate Geometry Research Program

**Computational Validation of the Oloid as a Local Optimum in the Developable Roller Family**

A formal framework for evaluating geometric primitives as engineering substrates, classified by their operational invariants rather than symmetry groups.

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
preprint/main.tex              # LaTeX source (arXiv submission pending)
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

**Oracle Runner:** Any watertight mesh + invariant definition + operating context → standardized primitive profile with scores across all shared dimensions + unique invariant-specific score.

See `FRAMEWORK.md` for the complete specification.

## Dependencies

- Python 3.10+
- NumPy, SciPy, trimesh
- FEniCS (optional, for FEM validation)

## License

MIT

## Citation

> Couey, V. (2026). Computational Validation of the Oloid as a Local Optimum in the Developable Roller Family. *Substrate Geometry Research Program*. arXiv preprint [pending endorsement].

## Acknowledgments

The oloid was discovered by Paul Schatz in 1929. Its developable surface properties were formally proven by Dirnböck and Stachel (1997). This work extends their lineage by providing the formal metric and computational infrastructure that confirms Schatz's finding rigorously.
