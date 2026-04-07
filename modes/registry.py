"""
modes/registry.py
---------------------------------------------------------------------------
Central registry of physics simulation modes.

Each mode is a dict with:
  - name: unique identifier matching invariant physics_mode field
  - description: what this mode simulates
  - simulator: dotted path to the simulation function
  - required_params: list of operating context fields the mode needs

The simulator function signature is always:
  def simulate(mesh, operating_context) -> sim_data dict

The sim_data dict is passed to the invariant's scoring function.

---------------------------------------------------------------------------
"""

MODE_REGISTRY = {}


def register_mode(definition: dict):
    """Register a physics mode in the global registry."""
    name = definition["name"]
    required = ["name", "description", "simulator", "required_params"]
    missing = [k for k in required if k not in definition]
    if missing:
        raise ValueError(f"Mode '{name}' missing required fields: {missing}")
    MODE_REGISTRY[name] = definition
    return definition


def get_mode(name: str) -> dict:
    """Retrieve a physics mode by name."""
    if name not in MODE_REGISTRY:
        available = list(MODE_REGISTRY.keys())
        raise KeyError(f"Mode '{name}' not found. Available: {available}")
    return MODE_REGISTRY[name]


# ─────────────────────────────────────────────────────────────────
# REGISTERED MODES
# ─────────────────────────────────────────────────────────────────

register_mode({
    "name": "rolling_plane",
    "description": "Rigid body rolling on a flat horizontal plane under gravity. "
                   "Euler-equation dynamics, quaternion integration, no-slip constraint.",
    "simulator": "modes.simulators.simulate_rolling_plane",
    "required_params": ["load", "material"],
})

register_mode({
    "name": "rolling_constrained",
    "description": "Rigid body rolling between two parallel horizontal plates. "
                   "Measures force variance on constraining plates (constant-width test).",
    "simulator": "modes.simulators.simulate_rolling_constrained",
    "required_params": ["load", "material", "plate_gap"],
})

register_mode({
    "name": "thermal_field",
    "description": "Static surface exposed to a heat flux field. "
                   "Measures thermal distribution across the surface.",
    "simulator": "modes.simulators.simulate_thermal_field",
    "required_params": ["heat_flux", "material"],
})

register_mode({
    "name": "static_equilibrium",
    "description": "Analyze stable and unstable equilibrium points of a body "
                   "resting under gravity.",
    "simulator": "modes.simulators.simulate_static_equilibrium",
    "required_params": [],
})

register_mode({
    "name": "structural_load",
    "description": "FEM analysis under applied uniaxial load. "
                   "Measures effective Poisson ratio and stress distribution.",
    "simulator": "modes.simulators.simulate_structural_load",
    "required_params": ["load", "material"],
})

register_mode({
    "name": "fluid_flow",
    "description": "Fluid flow through a channel defined by the surface geometry. "
                   "Measures pressure drop, flow uniformity, mixing efficiency.",
    "simulator": "modes.simulators.simulate_fluid_flow",
    "required_params": ["flow_rate", "fluid"],
})
