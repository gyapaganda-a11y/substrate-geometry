"""
invariants/registry.py
---------------------------------------------------------------------------
Central registry of all known invariant definitions.

Each invariant is a dict with:
  - name: unique string identifier
  - description: plain English explanation
  - predicate: formal mathematical statement
  - scorer: dotted path to the scoring function (module.function)
  - threshold: score below this means invariant is satisfied
  - physics_mode: which simulation mode the oracle runner should use

To add a new invariant:
  1. Write the scoring function in invariants/scorers.py (or a new file)
  2. Add the definition dict to INVARIANT_REGISTRY below
  3. That's it — the oracle runner will find it by name

---------------------------------------------------------------------------
"""

INVARIANT_REGISTRY = {}


def register_invariant(definition: dict):
    """Register an invariant definition in the global registry."""
    name = definition["name"]
    required = ["name", "description", "predicate", "scorer", "threshold", "physics_mode"]
    missing = [k for k in required if k not in definition]
    if missing:
        raise ValueError(f"Invariant '{name}' missing required fields: {missing}")
    INVARIANT_REGISTRY[name] = definition
    return definition


def get_invariant(name: str) -> dict:
    """Retrieve an invariant definition by name."""
    if name not in INVARIANT_REGISTRY:
        available = list(INVARIANT_REGISTRY.keys())
        raise KeyError(f"Invariant '{name}' not found. Available: {available}")
    return INVARIANT_REGISTRY[name]


# ─────────────────────────────────────────────────────────────────
# REGISTERED INVARIANTS
# ─────────────────────────────────────────────────────────────────

register_invariant({
    "name": "contact_distribution",
    "description": "Every point on the surface accumulates contact time "
                   "proportional to its area during rolling.",
    "predicate": "∀p ∈ ∂S: lim_{T→∞} (1/T) ∫₀ᵀ 𝟙[p ∈ C(t)] dt = 1/|∂S|",
    "scorer": "invariants.scorers.score_contact_distribution",
    "threshold": 1e-4,
    "physics_mode": "rolling_plane",
})

register_invariant({
    "name": "constant_width",
    "description": "Distance between any two parallel supporting planes "
                   "is constant regardless of orientation.",
    "predicate": "∀θ ∈ SO(3): d(θ) = d₀  (width between parallel support planes)",
    "scorer": "invariants.scorers.score_constant_width",
    "threshold": 1e-4,
    "physics_mode": "rolling_constrained",
})

register_invariant({
    "name": "zero_mean_curvature",
    "description": "Mean curvature H = 0 everywhere on the surface "
                   "(minimal surface / TPMS).",
    "predicate": "∀p ∈ ∂S: H(p) = 0",
    "scorer": "invariants.scorers.score_zero_mean_curvature",
    "threshold": 1e-3,
    "physics_mode": "thermal_field",
})

register_invariant({
    "name": "mono_monostatic",
    "description": "Exactly one stable and one unstable equilibrium point "
                   "(Gomboc property).",
    "predicate": "#{stable equilibria} = 1 ∧ #{unstable equilibria} = 1",
    "scorer": "invariants.scorers.score_mono_monostatic",
    "threshold": 0.5,  # binary: 0 = satisfied, 1 = not
    "physics_mode": "static_equilibrium",
})

register_invariant({
    "name": "negative_poisson",
    "description": "Material expands laterally under tension "
                   "(auxetic property, ν < 0).",
    "predicate": "ν_eff < 0 under uniaxial tension",
    "scorer": "invariants.scorers.score_negative_poisson",
    "threshold": 0.0,  # Poisson ratio must be negative
    "physics_mode": "structural_load",
})
