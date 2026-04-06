# Substrate Geometry Research Program

**Computational Validation of the Oloid as a Local Optimum in the Developable Roller Family**

This repository contains the source code, data, and preprint for the Substrate Geometry research program — a formal framework for evaluating geometric primitives as engineering substrates classified by their operational invariants.

## Key Result

The oloid (Schatz, 1929) occupies a local minimum in Contact Distribution Score (CDS) within the developable roller family, confirmed under rigid-body rolling dynamics:

| Geometry | CDS | vs. Oloid |
|----------|-----|-----------|
| Oloid | 8.2 x 10^-7 | 1.00x |
| Cylinder (conventional) | 4.75 x 10^-5 | 58x worse |

## Repository Structure

```
contact_oracle.py              # Pipeline Artifact 01 — Approximate CDS oracle
rigidbody_oracle.py            # Pipeline Artifact 03 — Rigid-body Euler-equation oracle
parametric_search.py           # Pipeline Artifact 02 — Parametric search (45 genomes)
parametric_search_results.json # Full search results (supplementary data)
preprint/main.tex              # LaTeX source for the preprint
index.html                     # Synthesis Engine dashboard (interactive)
candidate_viz.html             # Candidate visualization
```

## Reproducing Results

```bash
pip install trimesh numpy scipy
python contact_oracle.py        # Approximate oracle validation
python parametric_search.py     # Parametric search over developable roller family
python rigidbody_oracle.py      # Rigid-body oracle validation (confirms oloid optimality)
```

## Dependencies

- Python 3.10+
- NumPy, SciPy, trimesh

## License

MIT

## Citation

If you use this work, please cite:

> Contact Distribution Score: Computational Validation of the Oloid as a Local Optimum in the Developable Roller Family. Substrate Geometry Research Program, 2026. arXiv preprint [forthcoming].
