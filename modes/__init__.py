"""
modes/
---------------------------------------------------------------------------
Physics simulation modes for the oracle runner.

Each mode defines how to simulate a primitive under specific physical
conditions. The oracle runner dispatches to the appropriate mode based
on the invariant's physics_mode field.

Implemented modes:
  - rolling_plane: rigid body rolling on a flat plane (oloid, etc.)
  - rolling_constrained: rolling between parallel plates (Meissner body)

Planned modes:
  - thermal_field: heat flux on a static surface (gyroid electrode)
  - fluid_flow: fluid through a channel geometry (helicoidal, Murray)
  - static_equilibrium: equilibrium point analysis (Gomboc)
  - structural_load: FEM under applied loads (auxetic)

---------------------------------------------------------------------------
"""

from modes.registry import MODE_REGISTRY, get_mode, register_mode
