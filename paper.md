---
title: 'Substrate Geometry: a computational framework for evaluating geometric primitives by operational invariant'
tags:
  - Python
  - computational geometry
  - discrete differential geometry
  - contact mechanics
  - rigid-body dynamics
  - mesh oracles
authors:
  - name: Vincent Wesley Couey
    orcid: 0009-0005-6869-308X
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 13 June 2026
bibliography: paper.bib
---

# Summary

`substrate-geometry` is a Python framework for evaluating three-dimensional
geometric primitives as engineering substrates, classifying them by their
*operational invariants* (how a shape behaves under contact, rolling, stress,
thermal loading, fatigue, wear, and resting equilibrium) rather than by their
symmetry group. The framework provides seven validated scoring "oracles", a
generalized runner that scores any watertight triangulated mesh against a
registry of invariants and physics modes, and a standardized output profile
that makes cross-geometry comparison reproducible. It is the computational
engine behind a series of independent research papers on rolling primitives,
minimal-surface heat-exchanger substrates, and mono-monostatic bodies
[@couey2026oloid; @couey2026gomboc; @couey2026catalog], and it ships with the
hardening test suite that documents each oracle's trust boundaries.

# Statement of need

Computational claims about geometric primitives are usually reported as single
numbers (a curvature, a contact uniformity, an equilibrium count) computed by a
discrete mesh operator whose limits are rarely stated. When the operator is
resolution-dependent or otherwise mis-specified, the resulting number conflates
a genuine geometric property with a meshing artifact, and the literature cannot
tell the two apart. `substrate-geometry` addresses this by (i) expressing each
operational property as a named invariant with an explicit scoring function and
a falsification threshold, (ii) scoring every primitive through one standardized
runner so that results are directly comparable, and (iii) shipping a permanent
test suite that encodes both analytical reference cases and the *characterized
limitations* of each oracle, so a regression that silently changes a trust
boundary is caught.

This matters for any program that ranks or selects shapes by computed
performance. The framework was built to make the substrate-geometry papers'
quantitative claims reproducible and auditable, and it generalizes:
a researcher can register a new invariant (a scoring function plus a threshold)
or a new physics mode (a simulator) without modifying the existing oracles, and
immediately score an arbitrary mesh against it.

# Functionality

The package is organized around a thin, extensible architecture:

- **Seven frozen oracles** implementing the invariant vector used across the
  papers: contact distribution (CDS), rigid-body Euler-equation contact, Hertz
  contact pressure (SDS), Basquin S-N fatigue (FDS), frictional thermal (TDS),
  and Archard wear (WDS), plus a parametric search over a developable-roller
  family.
- **A generalized runner** (`oracle_runner.run_primitive`) that loads a mesh,
  selects the simulator for an invariant's physics mode, computes the invariant
  score and the shared physics vector, and returns a standardized
  `PrimitiveProfile` with a baseline comparison.
- **Registries** for invariants (`invariants/`) and physics modes (`modes/`):
  adding an invariant is writing a scoring function and registering a
  definition; the runner discovers it by name.
- **A hardening test suite** encoding analytical reference cases (for example,
  a sphere's Hertz pressure against the closed form) and documented oracle
  limitations as calibrated expectations.

The framework's methodology contribution, the systematic characterization of
when these discrete mesh oracles can and cannot be trusted, is described
separately [@couey2026methodology]; that work documents resolution-scaling
behavior, a discrete-curvature failure mode on topology-mismatched meshes, and
a convergent orientation-domain replacement for the equilibrium estimator. This
software is the implementation those analyses run on.

# State of the field

Discrete differential geometry operators on triangulated surfaces have
well-studied convergence behavior [@borrelli2003; @hildebrandt2006], and the
classification of homogeneous convex bodies by their equilibria is established
in the mono-monostatic literature [@domokos2006]. `substrate-geometry` does not
replace those results; it provides an open, reproducible harness that applies
such operators uniformly across an invariant vector and preserves each
operator's trust boundary as executable infrastructure rather than prose.

# Availability and reuse

The software is released under the MIT License. Code, data, and the validation
test suite are openly available, and each tagged release is archived on Zenodo
with a citable DOI. Replication of any reported result, extension to a new
geometry, and registration of new invariants are welcomed; see `CONTRIBUTING.md`.

# Acknowledgements

This framework was developed as the computational infrastructure for the
Substrate Geometry research program.

# References
