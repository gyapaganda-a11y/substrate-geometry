"""
oracle_runner.py
---------------------------------------------------------------------------
Substrate Geometry — Generalized Oracle Runner

The engine that takes any primitive and produces a standardized
primitive profile with invariant vector scores, operating context,
baseline comparison, and recommended validation.

USAGE:
  from oracle_runner import run_primitive, load_mesh

  # Score a new primitive
  profile = run_primitive(
      mesh=load_mesh("reuleaux/meissner_v1.stl"),
      invariant="constant_width",
      baseline_mesh=load_mesh("sphere_r1.stl"),
      baseline_name="Sphere (same width)",
      operating_context={
          "load": 100.0,
          "material": MILD_STEEL,
          "plate_gap": 2.0,  # for constrained rolling
      },
      name="Meissner body",
  )

  # Score across a load envelope
  profiles = run_load_sweep(
      mesh=..., invariant="contact_distribution",
      baseline_mesh=..., baseline_name="Cylinder",
      loads=[100, 500, 1000, 5000],
      material=MILD_STEEL,
  )

ARCHITECTURE:
  1. Look up invariant definition in invariants/registry.py
  2. Determine physics mode from invariant definition
  3. Look up simulation engine in modes/registry.py
  4. Run simulation → get sim_data
  5. Score the invariant using the registered scorer
  6. Compute full invariant vector (shared dimensions)
  7. Compare against baseline
  8. Write results to results/<primitive_name>.json
  9. Return PrimitiveProfile

---------------------------------------------------------------------------
"""

import json
import os
import sys
import time
import importlib
import numpy as np
import trimesh
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from invariants import get_invariant, INVARIANT_REGISTRY
from modes import get_mode, MODE_REGISTRY
from contact_oracle import score_distribution
from hertz_oracle import score_stress_distribution


# ─────────────────────────────────────────────────────────────────
# MATERIAL PRESETS
# ─────────────────────────────────────────────────────────────────

MILD_STEEL = {
    "name": "Mild steel",
    "E": 200e9,
    "nu": 0.3,
    "sigma_f_prime": 700e6,
    "basquin_b": -0.085,
    "sigma_e": 250e6,
    "archard_k": 1e-4,
    "hardness": 2.0e9,
    "friction_mu": 0.15,
    "thermal_conductivity": 50.0,
}

BEARING_STEEL = {
    "name": "AISI 52100 bearing steel",
    "E": 210e9,
    "nu": 0.3,
    "sigma_f_prime": 900e6,
    "basquin_b": -0.075,
    "sigma_e": 350e6,
    "archard_k": 5e-5,
    "hardness": 7.5e9,
    "friction_mu": 0.10,
    "thermal_conductivity": 46.0,
}

COPPER = {
    "name": "Copper (electrode grade)",
    "E": 117e9,
    "nu": 0.34,
    "sigma_f_prime": 300e6,
    "basquin_b": -0.12,
    "sigma_e": 100e6,
    "archard_k": 3e-4,
    "hardness": 0.8e9,
    "friction_mu": 0.30,
    "thermal_conductivity": 385.0,
}


# ─────────────────────────────────────────────────────────────────
# PRIMITIVE PROFILE
# ─────────────────────────────────────────────────────────────────

@dataclass
class VectorEntry:
    """One dimension of the invariant vector."""
    name: str
    value: float
    confidence: float = 0.0
    mesh_resolution: int = 0
    note: str = ""


@dataclass
class PrimitiveProfile:
    """Complete scored profile for a geometric primitive."""
    # Identity
    name: str
    invariant_name: str
    invariant_description: str
    predicate: str
    mesh_source: str
    n_faces: int
    surface_area: float

    # Operating context
    physics_mode: str
    constraint: str
    material: str
    load: float
    environment: str

    # Invariant score
    invariant_score: float
    invariant_satisfied: bool
    invariant_threshold: float

    # Full invariant vector (shared dimensions)
    vector: Dict[str, VectorEntry]

    # Unique invariant-specific score
    unique_score: VectorEntry

    # Baseline comparison
    baseline_name: str
    baseline_invariant_score: float
    baseline_ratios: Dict[str, float]

    # Life prediction (if applicable)
    first_failure_cycles: float = float('inf')
    mean_failure_cycles: float = float('inf')

    # Metadata
    manufacturability: str = "unknown"
    recommended_validation: str = ""
    elapsed_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self):
        """Convert to JSON-serializable dict."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, VectorEntry):
                d[k] = {"name": v.name, "value": v.value,
                         "confidence": v.confidence, "note": v.note}
            elif isinstance(v, dict):
                inner = {}
                for dk, dv in v.items():
                    if isinstance(dv, VectorEntry):
                        inner[dk] = {"name": dv.name, "value": dv.value,
                                     "confidence": dv.confidence, "note": dv.note}
                    elif isinstance(dv, float) and (dv == float('inf') or dv != dv):
                        inner[dk] = str(dv)
                    else:
                        inner[dk] = dv
                d[k] = inner
            elif isinstance(v, float) and (v == float('inf') or v != v):
                d[k] = str(v)
            else:
                d[k] = v
        return d


# ─────────────────────────────────────────────────────────────────
# MESH LOADING
# ─────────────────────────────────────────────────────────────────

def load_mesh(path: str) -> trimesh.Trimesh:
    """Load a mesh from file (STL, OBJ, PLY, etc.)."""
    full_path = os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path
    mesh = trimesh.load(full_path)
    if not mesh.is_watertight:
        print(f"  WARNING: Mesh at {path} is not watertight. "
              f"Results may be unreliable.")
    return mesh


# ─────────────────────────────────────────────────────────────────
# SCORING HELPERS
# ─────────────────────────────────────────────────────────────────

def _resolve_function(dotted_path: str):
    """Import and return a function from a dotted path like 'module.func'."""
    parts = dotted_path.rsplit(".", 1)
    module_path, func_name = parts[0], parts[1]
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _compute_shared_vector(sim_data, n_faces):
    """
    Compute the shared invariant vector dimensions from sim_data.
    Only computes dimensions for which data exists in sim_data.
    """
    vector = {}
    face_areas = sim_data.get("face_areas")
    if face_areas is None:
        return vector

    n = len(face_areas)

    # CDS — contact distribution
    cc = sim_data.get("contact_counts")
    if cc is not None and cc.sum() > 0:
        cds = score_distribution(cc, face_areas)
        vector["CDS"] = VectorEntry("CDS", cds, mesh_resolution=n,
                                     note="contact time distribution")

    # SDS — stress distribution
    sa = sim_data.get("stress_accumulator")
    if sa is not None and sa.sum() > 0:
        sds = score_stress_distribution(sa, face_areas)
        vector["SDS"] = VectorEntry("SDS", sds, mesh_resolution=n,
                                     note="Hertz stress distribution")

    # TDS — thermal distribution
    ha = sim_data.get("heat_accumulator")
    if ha is not None and ha.sum() > 0:
        tds = score_stress_distribution(ha, face_areas)
        vector["TDS"] = VectorEntry("TDS", tds, mesh_resolution=n,
                                     note="frictional thermal distribution")

    # FDS — fatigue damage distribution
    da = sim_data.get("damage_accumulator")
    if da is not None and da.sum() > 0:
        fds = score_stress_distribution(da, face_areas)
        vector["FDS"] = VectorEntry("FDS", fds, mesh_resolution=n,
                                     note="Basquin fatigue damage distribution")

    # WDS_vol — wear volume distribution
    wv = sim_data.get("wear_vol_accumulator")
    if wv is not None and wv.sum() > 0:
        wds_vol = score_stress_distribution(wv, face_areas)
        vector["WDS_vol"] = VectorEntry("WDS_vol", wds_vol, mesh_resolution=n,
                                         note="Archard wear volume distribution")

    # WDS_depth — wear depth distribution
    wd = sim_data.get("wear_depth_accumulator")
    if wd is not None and wd.sum() > 0:
        wds_depth = score_stress_distribution(wd, face_areas)
        vector["WDS_depth"] = VectorEntry("WDS_depth", wds_depth, mesh_resolution=n,
                                           note="wear depth (area-normalized)")

    return vector


def _compute_fatigue_life(sim_data):
    """Extract fatigue life predictions from sim_data if available."""
    da = sim_data.get("damage_accumulator")
    n_orient = sim_data.get("n_orientations", 1)
    if da is None or da.max() == 0:
        return float('inf'), float('inf')

    contacted = da[da > 0]
    max_dmg = contacted.max()
    mean_dmg = contacted.mean()

    dmg_per_cycle_worst = max_dmg / n_orient
    first_failure = 1.0 / dmg_per_cycle_worst if dmg_per_cycle_worst > 0 else float('inf')

    dmg_per_cycle_mean = mean_dmg / n_orient
    mean_failure = 1.0 / dmg_per_cycle_mean if dmg_per_cycle_mean > 0 else float('inf')

    return first_failure, mean_failure


# ─────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────

def run_primitive(
    mesh: trimesh.Trimesh,
    invariant: str,
    baseline_mesh: trimesh.Trimesh,
    baseline_name: str = "Baseline",
    operating_context: Optional[Dict] = None,
    name: str = "unnamed",
    mesh_source: str = "generated",
    manufacturability: str = "unknown",
    recommended_validation: str = "",
    verbose: bool = True,
) -> PrimitiveProfile:
    """
    Run the full oracle pipeline on a primitive.

    Parameters
    ----------
    mesh : the primitive's watertight mesh
    invariant : name of the invariant to test (from registry)
    baseline_mesh : conventional geometry for comparison
    baseline_name : label for the baseline
    operating_context : dict of operating parameters (load, material, etc.)
    name : primitive name
    mesh_source : where the mesh came from
    manufacturability : "machinable", "3d-printable", "SLM additive", etc.
    recommended_validation : suggested physical experiment
    verbose : print progress

    Returns
    -------
    PrimitiveProfile with all scores and comparisons
    """
    t0 = time.time()
    ctx = operating_context or {}

    if verbose:
        print(f"\n{'='*72}")
        print(f"ORACLE RUNNER — {name}")
        print(f"{'='*72}")
        print(f"  Invariant: {invariant}")
        print(f"  Mesh: {len(mesh.faces)} faces, area={mesh.area:.4f}")
        print(f"  Baseline: {baseline_name}")

    # 1. Look up invariant
    inv_def = get_invariant(invariant)
    if verbose:
        print(f"  Predicate: {inv_def['predicate']}")
        print(f"  Physics mode: {inv_def['physics_mode']}")

    # 2. Look up physics mode
    mode_def = get_mode(inv_def["physics_mode"])
    simulator = _resolve_function(mode_def["simulator"])

    # 3. Run simulation on primary mesh
    if verbose:
        print(f"\n  Running simulation: {mode_def['description'][:60]}...")
    sim_data = simulator(mesh, ctx)

    # 4. Score the specific invariant
    scorer = _resolve_function(inv_def["scorer"])
    inv_score = scorer(mesh, sim_data)
    inv_satisfied = inv_score < inv_def["threshold"]

    if verbose:
        status = "SATISFIED" if inv_satisfied else "NOT SATISFIED"
        print(f"  Invariant score: {inv_score:.8f} — {status}")

    # 5. Compute shared vector
    if verbose:
        print(f"  Computing invariant vector...")
    vector = _compute_shared_vector(sim_data, len(mesh.faces))

    if verbose:
        for k, v in vector.items():
            print(f"    {k}: {v.value:.2e}")

    # 6. Run simulation on baseline
    if verbose:
        print(f"\n  Running baseline: {baseline_name}...")
    baseline_sim = simulator(baseline_mesh, ctx)
    baseline_inv_score = scorer(baseline_mesh, baseline_sim)
    baseline_vector = _compute_shared_vector(baseline_sim, len(baseline_mesh.faces))

    # 7. Compute ratios
    baseline_ratios = {}
    for k in vector:
        if k in baseline_vector and vector[k].value > 0:
            ratio = baseline_vector[k].value / vector[k].value
            baseline_ratios[k] = round(ratio, 1)
    if inv_score > 0:
        baseline_ratios["invariant"] = round(baseline_inv_score / inv_score, 1)

    if verbose:
        print(f"  Baseline invariant score: {baseline_inv_score:.8f}")
        for k, r in baseline_ratios.items():
            print(f"    {k}: baseline is {r}x worse")

    # 8. Fatigue life
    first_fail, mean_fail = _compute_fatigue_life(sim_data)
    base_first, base_mean = _compute_fatigue_life(baseline_sim)

    # 9. Build unique score entry
    unique = VectorEntry(
        name=inv_def["name"],
        value=inv_score,
        mesh_resolution=len(mesh.faces),
        note=f"threshold={inv_def['threshold']}"
    )

    # 10. Build profile
    mat = ctx.get("material", {})
    elapsed = time.time() - t0

    profile = PrimitiveProfile(
        name=name,
        invariant_name=inv_def["name"],
        invariant_description=inv_def["description"],
        predicate=inv_def["predicate"],
        mesh_source=mesh_source,
        n_faces=len(mesh.faces),
        surface_area=float(mesh.area),
        physics_mode=inv_def["physics_mode"],
        constraint=mode_def["description"],
        material=mat.get("name", "default"),
        load=ctx.get("load", 100.0),
        environment=ctx.get("environment", "standard"),
        invariant_score=inv_score,
        invariant_satisfied=inv_satisfied,
        invariant_threshold=inv_def["threshold"],
        vector=vector,
        unique_score=unique,
        baseline_name=baseline_name,
        baseline_invariant_score=baseline_inv_score,
        baseline_ratios=baseline_ratios,
        first_failure_cycles=first_fail,
        mean_failure_cycles=mean_fail,
        manufacturability=manufacturability,
        recommended_validation=recommended_validation,
        elapsed_seconds=elapsed,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    # 11. Save results
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{name.lower().replace(' ', '_')}.json")
    with open(out_path, 'w') as f:
        json.dump(profile.to_dict(), f, indent=2, default=str)

    if verbose:
        print(f"\n  Results saved: {out_path}")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"{'='*72}\n")

    return profile


# ─────────────────────────────────────────────────────────────────
# LOAD SWEEP
# ─────────────────────────────────────────────────────────────────

def run_load_sweep(
    mesh: trimesh.Trimesh,
    invariant: str,
    baseline_mesh: trimesh.Trimesh,
    baseline_name: str,
    loads: List[float],
    material: Dict = None,
    name: str = "unnamed",
    **kwargs
) -> List[PrimitiveProfile]:
    """Run the oracle at multiple load levels and return profiles for each."""
    material = material or MILD_STEEL
    profiles = []
    for load in loads:
        ctx = {"load": load, "material": material}
        ctx.update(kwargs)
        p = run_primitive(
            mesh=mesh, invariant=invariant,
            baseline_mesh=baseline_mesh, baseline_name=baseline_name,
            operating_context=ctx, name=f"{name} @ {load}N",
            verbose=True,
        )
        profiles.append(p)
    return profiles


# ─────────────────────────────────────────────────────────────────
# CLI DEMO
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from contact_oracle import generate_oloid, generate_cylinder

    print("=" * 72)
    print("SUBSTRATE GEOMETRY — ORACLE RUNNER DEMO")
    print("=" * 72)
    print()
    print("Available invariants:")
    for name, inv in INVARIANT_REGISTRY.items():
        print(f"  {name}: {inv['description'][:70]}")
    print()
    print("Available physics modes:")
    for name, mode in MODE_REGISTRY.items():
        status = "implemented" if "stub" not in mode["description"].lower() else "stub"
        print(f"  {name}: {mode['description'][:60]} [{status}]")
    print()

    # Demo: run oloid through the generalized runner
    oloid = generate_oloid(r=1.0, n_circle_pts=400)
    cylinder = generate_cylinder(r=1.0, h=2.0)

    profile = run_primitive(
        mesh=oloid,
        invariant="contact_distribution",
        baseline_mesh=cylinder,
        baseline_name="Cylinder (conventional)",
        operating_context={
            "load": 100.0,
            "material": MILD_STEEL,
        },
        name="Oloid",
        mesh_source="generated (convex hull of two circles, r=1.0, n=400)",
        manufacturability="machinable, 3D-printable",
        recommended_validation="Rolling contact fatigue test on machined "
                               "specimens, thermocouple array for thermal "
                               "distribution measurement",
    )

    # Print summary
    print("PRIMITIVE PROFILE SUMMARY")
    print("-" * 50)
    print(f"  Name: {profile.name}")
    print(f"  Invariant: {profile.invariant_name} — "
          f"{'SATISFIED' if profile.invariant_satisfied else 'NOT SATISFIED'}")
    print(f"  Score: {profile.invariant_score:.2e}")
    print(f"  Baseline ({profile.baseline_name}): "
          f"{profile.baseline_invariant_score:.2e}")
    print(f"\n  Invariant vector:")
    for k, v in profile.vector.items():
        ratio = profile.baseline_ratios.get(k, "?")
        print(f"    {k}: {v.value:.2e}  (baseline {ratio}x worse)")
    print(f"\n  Fatigue: first failure at {profile.first_failure_cycles:.2e} cycles")
    print(f"  Manufacturability: {profile.manufacturability}")
    print(f"  Saved to: results/oloid.json")
