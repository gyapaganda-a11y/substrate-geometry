# Substrate Geometry — Framework Specification

**Last updated: 2026-04-06**
**Status: Active research program. Oloid fully validated. Framework generalizing.**

---

## 1. What Substrate Geometry Is

> "The study of geometric forms as engineering substrates, classified by operational invariants rather than symmetry groups. Where conventional geometry asks 'what shape is this?', substrate geometry asks 'what does this shape guarantee under physical operation?'"

The thesis in one line:

> "Changing the material delays the failure; changing the geometry eliminates it."

The long-term vision, in the user's own words:

> "I dont want to replace one cylinder of an engine with a cool shape and just revolutionize the engine. What I want? To discover if its possible to build on the whole of Archimedean architecture, to have someone look at a traditional current working system of parts and then my version of it, an array of newly designed geometric primitives re-distributing the foundational logic of the methodology a design intention was employed in physical space at the granular level up to the interlocking macro systems involved."

> "Its not even that new primitives can be discovered and used. Its that mapping this field, populating it with real working applications and new geometry that works with engineering principles, will reveal a new layer later on which is: How do all of these new applications of substrate geometry interact with EACH OTHER to produce even more vastly efficient methods of engineering."

---

## 2. The Invariant Vector

Every validated primitive receives a **complete invariant vector** — a set of independently measured scores across physics domains. This is the primitive's engineering identity card.

### The Oloid's Invariant Vector (first completed, 2026-04-06):

| Dimension | Score | Cylinder Ratio | Tier | Physics |
|---|---|---|---|---|
| CDS (contact time) | 8.20e-7 | 58x | 1 | Rolling dynamics |
| SDS (stress) | 8.07e-7 | 58x | 1 | Hertz contact theory |
| TDS (thermal) | 7.77e-7 | 68x | 1 | Frictional heat (p × v) |
| WDS_vol (wear volume) | 7.77e-7 | 68x | 1 | Archard's wear law |
| WDS_depth (wear depth) | 1.15e-6 | 46x | 2 | Area-normalized wear |
| FDS (fatigue) | 2.42e-6 | ∞x | 2 | Basquin S-N + Miner's rule |

### Two-Tier Structure

**Tier 1 (linear/multiplicative):** CDS, SDS, TDS, WDS_vol all cluster at 7.8-8.2e-7. Four independent physics measurements landing within 5% of each other. The contact time invariant propagates losslessly through stress, thermal, and wear volume physics. This is the geometric guarantee working as claimed.

**Tier 2 (nonlinear):** FDS and WDS_depth diverge. FDS because Basquin's S-N law is exponential — a 15% stress difference becomes a 23x fatigue life difference. WDS_depth because area normalization amplifies small face size variations. The oloid still dominates on both, but the invariant transfer is lossy at the nonlinear layer.

**Key finding:** TDS (7.77e-7) is the LOWEST score in the vector — lower than CDS itself. The oloid's rolling kinematics produce an inverse correlation between contact frequency and sliding velocity: faces that contact most often slide slower. The geometry is self-compensating thermally. This was not designed in. It emerged from the rolling kinematics of the developable surface.

**Key finding:** The cylinder's FDS = infinity. At 5000N operating load, all cylinder contact stresses fall below the endurance limit because contact is so localized. The cylinder doesn't fail by distributed fatigue — it fails by pitting and spalling at a single locus. The oracle correctly identified a qualitatively different failure mode, not just a quantitatively worse score.

### Shared Dimensions vs Unique Dimensions

Every primitive gets scored on the **shared dimensions** (CDS, SDS, TDS, FDS, WDS) — these are the common language that lets you compare across primitives.

Every primitive ALSO gets its own **unique invariant-specific score** — the measurement that captures what makes that specific primitive irreplaceable:
- Oloid: CDS is its unique dimension (contact distribution)
- Meissner body: CWS (constant-width score) — force variance between parallel constraining plates
- Gyroid: HMS (mean curvature score) — deviation from H=0

The shared dimensions let you query "show me all primitives with TDS below threshold X." The unique dimensions let you query "show me all primitives with a constant-width invariant."

---

## 3. Oracle Runner Architecture

The oracle runner is not "run mesh through 5 oracles." It's the engine that takes a shape and an engineering context and outputs a prediction an engineer can act on.

### Inputs

**Required:**
- **Mesh** — any watertight mesh (STL, OBJ). Can be parametrically generated, downloaded (Thingiverse), or modeled (Blender). The oracles don't care how it was made.
- **Invariant definition** — formal computable predicate. What does this shape claim to guarantee?
- **Baseline geometry** — you always need "better than what?" Every result is a delta, not an absolute.

**Operating context:**
- **Constraint geometry** — flat plane, parallel plates, cylindrical bore, pipe interior, plasma field. The constraint defines the simulation. An oloid on a flat plane is different from a Meissner body between plates.
- **Load envelope** — not one load but a range. The fatigue oracle showed the oloid and cylinder swap advantage at different loads. The runner should sweep and report crossovers.
- **Operating frequency** — RPM, cycling rate. A bearing at 3000 RPM accumulates thermal damage differently than at 300 RPM.
- **Material properties** — E, ν, σ_f', b, σ_e, k, H, thermal conductivity, etc. Not hardcoded. Passed as parameters.
- **Environment** — temperature, corrosion, lubrication regime. These modify material properties.
- **Failure criterion** — not just "score the distribution" but "at what cycle count does the first face exceed the material's failure threshold?" Turns scores into predicted service life.

**Metadata:**
- **Manufacturability flag** — can this shape be made? Machinable, 3D-printable, requires additive manufacturing, theoretical only.
- **Composition slots** — adjacent geometries in a system (for future invariant composition studies).

### Output: The Primitive Profile

```json
{
  "primitive": "Meissner body",
  "invariant": "constant width",
  "predicate": "∀θ: d(θ) = d₀",
  "mesh": "meissner_v1.stl",
  "context": {
    "constraint": "parallel plates",
    "load_envelope": [100, 500, 1000, 5000],
    "material": "bearing steel, E=200GPa, ν=0.3",
    "frequency": "variable",
    "environment": "lubricated, ambient temperature",
    "failure_criterion": "first face exceeding 1500K or D=1.0"
  },
  "baseline": "sphere (same width)",
  "vector": {
    "CDS": { "value": 8.2e-7, "confidence": "±0.3e-7", "mesh_resolution": "1198 faces" },
    "SDS": { "value": 8.1e-7, "confidence": "±0.3e-7" },
    "TDS": { "value": 7.8e-7, "confidence": "±0.2e-7" },
    "FDS": { "value": 2.4e-6, "confidence": "±0.5e-6" },
    "WDS": { "value": 7.8e-7, "confidence": "±0.2e-7" },
    "CWS": { "value": 1.2e-8, "note": "unique to this primitive" }
  },
  "baseline_ratios": {
    "CDS": "58x", "SDS": "58x", "TDS": "68x"
  },
  "crossover_loads": {
    "FDS_cylinder_crossover": "cylinder below endurance at 5000N"
  },
  "predicted_service_life": {
    "first_failure_cycles": 4.55e5,
    "mean_failure_cycles": 1.14e6,
    "life_vs_baseline": "cylinder fails at fixed locus before distributed fatigue initiates"
  },
  "manufacturability": "3D-printable (SLM additive)",
  "recommended_validation": "thermocouple array on 3D-printed specimen under controlled load",
  "composition_compatibility": []
}
```

### The Target Output

When the framework is pointed at a real engineering problem, the output should look like:

> "Gyroid electrode surface (H=0 invariant) under 50 kW/m² arc heat flux at 1200K predicts TDS = X vs flat plate TDS = Y, a Z% improvement in thermal uniformity. Predicted thermal cycling life: N cycles to first hotspot exceeding 1500K. Confidence: ±W% at current mesh resolution. Manufacturable via SLM additive. Recommended experimental validation: thermocouple array on 3D-printed specimen under controlled arc."

That's what "theoretical real experiments that map onto direct application" means. The computation defines the experiment. The experiment confirms the prediction.

---

## 4. Design Principles

### Use existing infrastructure before building from scratch

> "We just had an issue in scope, laid out what we needed as some grandiose workflow that wouldve taken me into building 3d models and away from the core work, realized architecture already existed that we could use for free instead, and shortened production time by around 40-60% conservatively. Lets keep applying that logic going forward."

- Free meshes on Thingiverse before modeling in Blender
- Import from existing oracles before writing new physics
- Published material properties before running custom characterization
- Existing simulation frameworks (FEniCS, OpenFOAM, DEAP) before custom solvers

### Every primitive on its own terms

> "I wont measure the Meissner body on the oloid's terms, that doesnt even make sense. I'd make an entirely new set of oracles for it and that becomes part of the library."

Each primitive gets:
- Its own folder in the project root (`reuleaux/`, `gomboc/`, `gyroid/`)
- Its own invariant-specific oracle
- Its own unique dimension in the invariant vector
- PLUS the shared dimensions for cross-primitive comparison

### Additive architecture only

Each new primitive's work is purely additive. New folders, new files, imports from the existing frozen codebase. See Section 7 for the frozen codebase rule.

### Granularity serves cross-referencing

> "Once we have a rich enough dataset and computational simulation library it wouldn't be that difficult to just point you in the direction of an industry and cross reference that with 'Synthesis Engine already knows x weird shape in substrate geometry maps cleanly onto a more efficient engine cylinder or heat exchanger or x or y component.'"

The library's value is proportional to:
- Number of validated primitives (more shapes = more candidates per query)
- Number of scored dimensions per primitive (more physics = more precise matching)
- Number of operating profiles tested (more contexts = more applicable results)

A three-shape comparison is basic. A 10-primitive library with 5+ dimensions each, scored across multiple operating regimes, is a real search space.

---

## 5. Path from Computation to Application

### The gap and how to close it

Computation produces predictions. Physical validation confirms them. Both are necessary. The framework's job is to make the computational predictions precise enough that the experimental design is obvious.

The bridge between "invariant vector score" and "demonstrated engineering improvement" is the **operating regime mapping**:

1. **Define the failure mode precisely.** Not "heat is bad" but "thermal concentration at the electrode surface exceeds material limits at X operating condition, causing failure at Y cycles."

2. **Define what geometric property would address it.** For thermal concentration: surface area distribution, curvature uniformity, or flow path geometry.

3. **Build/adapt the oracle that measures that property.**

4. **Run the oracle on candidate geometries including conventional baselines.** Get a number. Show the improvement computationally.

5. **Output a recommended experimental validation** — what physical test would confirm the computational prediction, what instrumentation is needed, what success criterion applies.

### Confidence bounds matter

Computational predictions have uncertainty. The runner should report not just "TDS = 7.77e-7" but "TDS = 7.77e-7 ± 0.3e-7 at this mesh resolution." That's what makes the prediction mappable to a real experiment — you know the error bars.

---

## 6. What "Done" Looks Like for a Primitive

A primitive is a validated substrate library entry when it has:

- [ ] Formal invariant stated as a computable predicate
- [ ] Watertight mesh (generated, downloaded, or modeled)
- [ ] Unique invariant-specific oracle scoring its claimed property
- [ ] Full shared invariant vector (CDS, SDS, TDS, FDS, WDS minimum)
- [ ] Baseline comparison with conventional geometry
- [ ] Results across a load envelope (not just one load)
- [ ] Material properties documented
- [ ] Constraint geometry specified
- [ ] Confidence bounds reported
- [ ] Cross-domain transfer evidence (invariant holds in >1 physics regime)
- [ ] Manufacturability assessment
- [ ] Recommended experimental validation defined
- [ ] Results committed to git in its own folder
- [ ] Primitive profile JSON in standardized schema

---

## 7. Frozen Codebase Rule

**THE OLOID CODEBASE IS FROZEN. DO NOT MODIFY EXISTING FILES.**

As of 2026-04-06, the following files have been:
- Pushed to GitHub (github.com/gyapaganda-a11y/substrate-geometry)
- Referenced in a preprint sent to three endorsement targets (Bäsel, Brecher, Stachel)
- Cited with specific numerical results in endorsement emails

### Frozen files (never edit):
```
contact_oracle.py           — Artifact 01
parametric_search.py        — Artifact 02
rigidbody_oracle.py         — Artifact 03
hertz_oracle.py             — Artifact 04
fatigue_oracle.py           — Artifact 05
thermal_oracle.py           — Artifact 06
wear_oracle.py              — Artifact 07
parametric_search_results.json
preprint/main.tex
```

### Why:
The numbers in these files (CDS = 8.2e-7, SDS = 8.07e-7, etc.) are cited in emails to professors requesting arXiv endorsement. If the code changes and produces different numbers, the paper doesn't match the code and the endorsement request is undermined.

### How to add new work:
Each new primitive gets its own folder:
```
D:/Downloads/Physics Project/reuleaux/
D:/Downloads/Physics Project/gomboc/
D:/Downloads/Physics Project/gyroid/
```

New oracle files IMPORT from the frozen codebase — use it as a library:
```python
from contact_oracle import generate_oloid, score_distribution
from hertz_oracle import compute_vertex_curvatures, hertz_peak_pressure
from thermal_oracle import simulate_rolling_with_velocity
```

### When can frozen files be edited?
Only after the arXiv paper is accepted/posted AND the user explicitly authorizes changes.

### What IS NOT frozen:
- `index.html` (dashboard) — update freely with new results
- `FRAMEWORK.md` (this document) — living document, update as architecture evolves
- Any new folders/files for new primitives

---

## 8. Current Pipeline Status

### Completed (2026-04-06):
- 7 oracle artifacts for oloid (contact, rigidbody, hertz, fatigue, thermal, wear + parametric search)
- Complete 5-dimension invariant vector with two-tier structure
- 1,430-genome parametric search confirming oloid as local optimum
- Preprint submitted for arXiv endorsement (3 targets: Bäsel, Brecher, Stachel)
- Dashboard deployed at synthesis-engine-8rm.pages.dev
- GitHub repo live at github.com/gyapaganda-a11y/substrate-geometry

### Next build:
1. Generalized oracle runner (`oracle_runner.py`) — standardized primitive onboarding
2. Meissner body — download STL, build constant-width oracle, full vector
3. Plasma electrode case study — adapt thermal oracle for arc heat flux

### Substrate library (10 entries, 1 validated):
- **Validated:** Oloid (full invariant vector, all 7 oracles)
- **Cataloged:** Reuleaux/Meissner, Gomboc, Schatz linkage, Gyroid, Auxetic lattice, Schwartz P, Helicoidal channel, Log spiral, Murray branching

### Future:
- DEAP evolutionary search over continuous parameter spaces
- Invariant composition (do invariants compose when primitives combine?)
- MHD channel segment proof-of-concept (3 composed primitives)
- Physical specimen validation

---

## 9. Operating Principles

> "If im bringing something up to you then it has tie ins to an overarching plan."

> "I just want to be shooting at the right targets as early as possible."

> "We should be granular enough to have enough of a library and contextual understanding of substrate geometry to begin to formalize the instinct to be able to point to the libraries and tools we're developing to come to the conclusion that 'an electrode heat exchanger's ability to dissipate heat can be improved by x substrate geometry shape at y efficiency' with pretty robust confidence."

The framework succeeds when it can take an engineering failure mode as input and return a geometric solution as output, with enough rigor that the recommended experiment is obvious and the predicted improvement has confidence bounds.

Everything built serves that goal. If a piece of work doesn't move toward that capability, it's not the right next step.
