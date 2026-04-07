"""
invariants/
---------------------------------------------------------------------------
Named invariant definitions for the oracle runner.

Each invariant is a config dict referencing a scoring function.
New primitives register their invariant here. The oracle runner
dispatches based on the invariant name.

SCHEMA:
  {
    "name": str,              # unique identifier
    "description": str,       # plain English
    "predicate": str,         # formal mathematical statement
    "scorer": str,            # dotted path to scoring function
    "threshold": float,       # score below this = invariant satisfied
    "physics_mode": str,      # which simulation mode to use
  }
---------------------------------------------------------------------------
"""

from invariants.registry import INVARIANT_REGISTRY, get_invariant, register_invariant
