# Contributing to Substrate Geometry

This is an independent computational research program. Most academic codebases are inert reference artifacts; this one is not. The framework is small enough to extend, and the oracle architecture is designed for additive work.

## What you can do here

1. **Extend an existing oracle to a new geometry.** The seven oracle artifacts (contact, parametric search, rigid-body, Hertz, fatigue, thermal, wear) are frozen for paper I reproducibility. New geometries can be scored under any of them by writing a single mesh-loading wrapper.
2. **Add a new invariant.** `invariants/registry.py` lists current invariant definitions. New invariants need a scoring function and a registry entry. Any invariant grounded in a published physical model is welcome.
3. **Add a new primitive.** Mesh-bearing primitives can be evaluated by `oracle_runner.py` with no code change. Submit STL via `meshes/` plus a one-paragraph `results/{primitive}.json` template entry.
4. **Replicate a constant-width / mono-monostatic claim.** The companion paper on mono-monostatic primitives (in submission) reports a 13-member catalog extending M. L. Sloan's 2023 analytical Gömböc parameterization. Independent computational replication of any catalog member is a genuine contribution.

## Coauthorship policy

If your independent extension produces (a) a member #14 of the mono-monostatic catalog, (b) the analog for a related body family (e.g., constant-width Reuleaux solids or non-developable rollers), or (c) a substantive new invariant validated against an existing primitive at the precision documented in the methodology paper, you are invited as coauthor on the next paper covering that result. The threshold is genuine technical contribution, not commentary. Please reach out before deep work so we can coordinate scope and avoid duplicated effort.

## Provenance and citation

The seven core oracles in this repo back arXiv:2604.12238 (paper I, oloid validation). For the framework, cite the paper I arXiv ID. Each paper in the program has its own arXiv submission and DOI; the author page is the canonical listing: <https://arxiv.org/a/couey_v_1.html>

## Reaching the author

For coauthorship coordination, technical questions, or independent replications: file a GitHub issue with a clear summary. For private correspondence: see the contact information on the arXiv author page.

## What we will not accept

- LLM-generated commentary or speculative extensions without computational evidence.
- Rewrites of frozen oracle artifacts. They are pinned for reproducibility.
- Edits to `FROZEN-CODEBASE.md` listings.
- Submissions that change paper claims without supporting numerical work.

## Style

- No em-dashes in any contributed text.
- Cite the physical model behind each invariant. The framework distinguishes itself by grounding scoring in published physics, not in heuristic optimization.
- Match the existing oracle output schema. Standardized JSON profiles per primitive live under `results/`.
