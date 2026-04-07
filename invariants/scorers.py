"""
invariants/scorers.py
---------------------------------------------------------------------------
Scoring functions for each registered invariant.

Each function takes standardized inputs from the oracle runner and
returns a score. Lower = better. Score < threshold = invariant satisfied.

Scoring functions IMPORT from the frozen oloid codebase — they use it
as a library, never modify it.

---------------------------------------------------------------------------
"""

import numpy as np
import trimesh
import sys
import os

# Add project root to path so we can import from frozen codebase
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contact_oracle import score_distribution
from hertz_oracle import (
    compute_vertex_curvatures, compute_face_curvatures,
    hertz_peak_pressure, score_stress_distribution,
    E_STAR, REFERENCE_LOAD
)


# ─────────────────────────────────────────────────────────────────
# CONTACT DISTRIBUTION (oloid invariant)
# ─────────────────────────────────────────────────────────────────

def score_contact_distribution(mesh, sim_data):
    """
    Score: area-weighted variance of contact time distribution.
    CDS → 0 means every face contacts proportional to its area.

    sim_data must contain:
      - contact_counts: ndarray (n_faces,)
      - face_areas: ndarray (n_faces,)
    """
    return score_distribution(sim_data["contact_counts"], sim_data["face_areas"])


# ─────────────────────────────────────────────────────────────────
# CONSTANT WIDTH (Meissner body invariant)
# ─────────────────────────────────────────────────────────────────

def score_constant_width(mesh, sim_data):
    """
    Score: variance of width measurements across orientations.
    Width = distance between parallel supporting planes.

    For a perfect constant-width body, this is zero.
    Score = var(widths) / mean(widths)^2  (normalized variance)

    sim_data must contain:
      - widths: ndarray of width measurements at each sampled orientation
    """
    widths = sim_data["widths"]
    if len(widths) == 0:
        return float('inf')
    mean_w = np.mean(widths)
    if mean_w < 1e-15:
        return float('inf')
    return float(np.var(widths) / mean_w**2)


# ─────────────────────────────────────────────────────────────────
# ZERO MEAN CURVATURE (gyroid / TPMS invariant)
# ─────────────────────────────────────────────────────────────────

def score_zero_mean_curvature(mesh, sim_data):
    """
    Score: RMS of mean curvature across the surface.
    H = 0 everywhere → score = 0.

    Uses discrete curvature from hertz_oracle infrastructure.
    sim_data is unused (curvature is computed from mesh geometry).
    """
    H_vert, _, _ = compute_vertex_curvatures(mesh)
    # Area-weighted RMS of mean curvature
    face_areas = mesh.area_faces
    H_face = np.zeros(len(mesh.faces))
    for fi, face in enumerate(mesh.faces):
        H_face[fi] = np.mean(H_vert[face])

    total_area = face_areas.sum()
    rms_H = np.sqrt(np.sum(face_areas * H_face**2) / total_area)
    return float(rms_H)


# ─────────────────────────────────────────────────────────────────
# MONO-MONOSTATIC (Gomboc invariant)
# ─────────────────────────────────────────────────────────────────

def score_mono_monostatic(mesh, sim_data):
    """
    Score: |n_stable - 1| + |n_unstable - 1|
    Perfect Gomboc has exactly 1 stable + 1 unstable = score 0.

    Equilibria are found by checking which faces the body can rest
    on stably (center of mass above support polygon) and which
    vertices are unstable equilibrium points.

    sim_data must contain:
      - n_stable: int
      - n_unstable: int
    """
    n_s = sim_data.get("n_stable", 0)
    n_u = sim_data.get("n_unstable", 0)
    return float(abs(n_s - 1) + abs(n_u - 1))


# ─────────────────────────────────────────────────────────────────
# NEGATIVE POISSON RATIO (auxetic invariant)
# ─────────────────────────────────────────────────────────────────

def score_negative_poisson(mesh, sim_data):
    """
    Score: the effective Poisson's ratio itself.
    ν < 0 = invariant satisfied. More negative = stronger auxetic.

    sim_data must contain:
      - poisson_ratio: float (effective ν from FEM simulation)
    """
    return float(sim_data.get("poisson_ratio", 1.0))
