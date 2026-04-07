"""
modes/simulators.py
---------------------------------------------------------------------------
Physics simulation implementations for each mode.

Each simulator function takes (mesh, operating_context) and returns a
sim_data dict that the invariant scorer can consume.

IMPORTS from the frozen oloid codebase — uses it as a library.

Implemented:
  - simulate_rolling_plane (complete — wraps existing oracle infrastructure)
  - simulate_rolling_constrained (complete — parallel plate width measurement)
  - simulate_static_equilibrium (complete — equilibrium point counting)

Stubs (architecture defined, implementation pending):
  - simulate_thermal_field
  - simulate_structural_load
  - simulate_fluid_flow

---------------------------------------------------------------------------
"""

import numpy as np
import trimesh
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contact_oracle import score_distribution
from hertz_oracle import (
    compute_vertex_curvatures, compute_face_curvatures,
    hertz_peak_pressure, score_stress_distribution,
    E_STAR
)
from thermal_oracle import simulate_rolling_with_velocity


# ─────────────────────────────────────────────────────────────────
# ROLLING ON FLAT PLANE
# ─────────────────────────────────────────────────────────────────

def simulate_rolling_plane(mesh, ctx):
    """
    Full rolling simulation on a flat plane with all 5 invariant
    vector dimensions computed simultaneously.

    ctx fields used:
      load: float (N) — normal force for Hertz calculations
      material: dict with E, nu, sigma_f_prime, basquin_b, sigma_e,
                archard_k, hardness, friction_mu
      t_total, dt, sample_every, contact_threshold, damping, n_runs:
                simulation parameters (optional, have defaults)

    Returns sim_data dict with all accumulated arrays.
    """
    load = ctx.get("load", 100.0)
    mat = ctx.get("material", {})
    E = mat.get("E", 200e9)
    nu = mat.get("nu", 0.3)
    E_star = E / (2 * (1 - nu**2))
    mu_friction = mat.get("friction_mu", 0.15)
    archard_k = mat.get("archard_k", 1e-4)
    hardness = mat.get("hardness", 2.0e9)
    sigma_f_prime = mat.get("sigma_f_prime", 700e6)
    basquin_b = mat.get("basquin_b", -0.085)
    sigma_e = mat.get("sigma_e", 250e6)

    t_total = ctx.get("t_total", 10.0)
    dt = ctx.get("dt", 0.002)
    sample_every = ctx.get("sample_every", 25)
    contact_threshold = ctx.get("contact_threshold", 0.04)
    damping = ctx.get("damping", 0.02)
    n_runs = ctx.get("n_runs", 3)
    contact_dt = ctx.get("contact_dt", 0.05)

    # Curvatures
    H_vert, K_vert, _ = compute_vertex_curvatures(mesh)
    _, _, _, _, R_eff = compute_face_curvatures(mesh, H_vert, K_vert)

    n_faces = len(mesh.faces)
    centroids_body = mesh.triangles_center - mesh.centroid
    face_areas = mesh.area_faces.copy()

    contact_counts = np.zeros(n_faces)
    stress_acc = np.zeros(n_faces)
    heat_acc = np.zeros(n_faces)
    damage_acc = np.zeros(n_faces)
    wear_vol_acc = np.zeros(n_faces)
    wear_depth_acc = np.zeros(n_faces)

    # Load scaling for fatigue (reference vs operating load)
    load_scale = (load / 100.0)**(1.0 / 3.0)

    initial_pushes = [
        np.array([0.3, 1.5, 0.2]),
        np.array([-0.5, 1.0, 0.8]),
        np.array([0.7, -1.2, 0.4]),
        np.array([0.1, 0.8, -1.0]),
        np.array([-0.3, -0.6, 1.5]),
    ][:n_runs]

    total_samples = 0

    for push in initial_pushes:
        rotations, omegas = simulate_rolling_with_velocity(
            mesh, t_total=t_total, dt=dt,
            damping=damping, initial_push=push,
        )

        for i, (R, omega) in enumerate(zip(rotations, omegas)):
            if i % sample_every != 0:
                continue

            rotated = (R @ centroids_body.T).T
            rotated[:, 2] -= rotated[:, 2].min()

            in_contact = rotated[:, 2] < contact_threshold
            contact_counts[in_contact] += 1

            contact_faces = np.where(in_contact)[0]
            for fi in contact_faces:
                # Hertz pressure (reference load)
                p = hertz_peak_pressure(R_eff[fi], E_star, 100.0)
                stress_acc[fi] += p

                # Sliding velocity
                r_vec = rotated[fi]
                v_contact = np.cross(omega, r_vec)
                v_slide = np.sqrt(v_contact[0]**2 + v_contact[1]**2)

                # Thermal
                heat_acc[fi] += mu_friction * p * v_slide

                # Wear (Archard)
                w_vol = archard_k * p * v_slide * contact_dt / hardness
                wear_vol_acc[fi] += w_vol
                wear_depth_acc[fi] += w_vol / face_areas[fi]

                # Fatigue (scaled to operating load)
                p_scaled = p * load_scale
                if p_scaled >= sigma_e:
                    nf = (p_scaled / sigma_f_prime)**(1.0 / basquin_b)
                    if nf > 0 and nf != float('inf'):
                        damage_acc[fi] += 1.0 / nf

            total_samples += 1

    return {
        "contact_counts": contact_counts,
        "stress_accumulator": stress_acc,
        "heat_accumulator": heat_acc,
        "damage_accumulator": damage_acc,
        "wear_vol_accumulator": wear_vol_acc,
        "wear_depth_accumulator": wear_depth_acc,
        "face_areas": face_areas,
        "n_orientations": total_samples,
        "R_eff": R_eff,
        "H_vertex": H_vert,
        "K_vertex": K_vert,
    }


# ─────────────────────────────────────────────────────────────────
# ROLLING BETWEEN PARALLEL PLATES (CONSTANT WIDTH)
# ─────────────────────────────────────────────────────────────────

def simulate_rolling_constrained(mesh, ctx):
    """
    Roll a body and measure width (distance between parallel supporting
    planes) at each orientation. For a constant-width body, all widths
    should be equal.

    Also computes standard contact distribution on the lower plane.

    ctx fields used:
      plate_gap: float — nominal distance between plates (if known)
      t_total, dt, sample_every, damping, n_runs: simulation params

    Returns sim_data dict with widths array and contact data.
    """
    from rigidbody_oracle import (
        simulate_rolling, quat_to_matrix
    )

    t_total = ctx.get("t_total", 10.0)
    dt = ctx.get("dt", 0.002)
    sample_every = ctx.get("sample_every", 25)
    damping = ctx.get("damping", 0.02)
    n_runs = ctx.get("n_runs", 3)
    contact_threshold = ctx.get("contact_threshold", 0.04)

    n_faces = len(mesh.faces)
    verts_body = mesh.vertices - mesh.centroid
    centroids_body = mesh.triangles_center - mesh.centroid
    face_areas = mesh.area_faces.copy()
    contact_counts = np.zeros(n_faces)

    widths = []

    initial_pushes = [
        np.array([0.3, 1.5, 0.2]),
        np.array([-0.5, 1.0, 0.8]),
        np.array([0.7, -1.2, 0.4]),
    ][:n_runs]

    for push in initial_pushes:
        rotations = simulate_rolling(
            mesh, t_total=t_total, dt=dt,
            damping=damping, initial_push=push,
            verbose=False
        )

        for i, R in enumerate(rotations):
            if i % sample_every != 0:
                continue

            # Rotate vertices to world frame
            verts_world = (R @ verts_body.T).T

            # Width = max_z - min_z (distance between parallel horizontal planes)
            z_vals = verts_world[:, 2]
            width = z_vals.max() - z_vals.min()
            widths.append(width)

            # Also do contact distribution on lower plane
            rotated_c = (R @ centroids_body.T).T
            rotated_c[:, 2] -= rotated_c[:, 2].min()
            in_contact = rotated_c[:, 2] < contact_threshold
            contact_counts[in_contact] += 1

    return {
        "widths": np.array(widths),
        "contact_counts": contact_counts,
        "face_areas": face_areas,
        "n_orientations": len(widths),
    }


# ─────────────────────────────────────────────────────────────────
# STATIC EQUILIBRIUM (GOMBOC)
# ─────────────────────────────────────────────────────────────────

def simulate_static_equilibrium(mesh, ctx):
    """
    Count stable and unstable equilibrium points of a convex body.

    Stable equilibrium: face the body can rest on with center of mass
    directly above the support polygon.

    Unstable equilibrium: vertex the body can balance on (center of
    mass directly above the vertex).

    Returns sim_data with n_stable and n_unstable counts.
    """
    centroid = mesh.centroid
    normals = mesh.face_normals

    # Stable equilibria: faces where the centroid projects onto the face
    # and the normal points upward (face is on the bottom)
    n_stable = 0
    for fi, face in enumerate(mesh.faces):
        normal = normals[fi]
        # Face must point downward for the body to rest on it
        if normal[2] > -0.1:
            continue

        # Check if centroid projects into the face's support polygon
        face_verts = mesh.vertices[face]
        face_center = face_verts.mean(axis=0)

        # Project centroid onto face plane
        d = np.dot(centroid - face_center, normal)
        proj = centroid - d * normal

        # Check if projection is inside triangle (barycentric coords)
        v0 = face_verts[2] - face_verts[0]
        v1 = face_verts[1] - face_verts[0]
        v2 = proj - face_verts[0]

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-15:
            continue

        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom

        if u >= 0 and v >= 0 and u + v <= 1:
            n_stable += 1

    # Unstable equilibria: vertices where centroid is directly above
    n_unstable = 0
    for vi, vert in enumerate(mesh.vertices):
        # Check if centroid is directly above this vertex
        # (i.e., the lateral offset is very small compared to height)
        lateral = np.sqrt((centroid[0] - vert[0])**2 + (centroid[1] - vert[1])**2)
        height = centroid[2] - vert[2]
        if height > 0 and lateral < 0.01 * height:
            n_unstable += 1

    # This is simplified — real Gomboc analysis requires checking ALL
    # orientations, not just face-down. But it demonstrates the mode
    # interface. Full implementation would sample SO(3).

    return {
        "n_stable": n_stable,
        "n_unstable": n_unstable,
    }


# ─────────────────────────────────────────────────────────────────
# STUBS — architecture defined, implementation pending
# ─────────────────────────────────────────────────────────────────

def simulate_thermal_field(mesh, ctx):
    """
    Stub: static surface exposed to directional heat flux.
    Implementation requires FEniCS heat equation solver.

    When implemented, this will:
    1. Generate a volume mesh from the surface (gmsh)
    2. Apply heat flux boundary condition on exposed surface
    3. Solve steady-state heat equation
    4. Extract surface temperature distribution
    5. Return temperature variance as the thermal score
    """
    raise NotImplementedError(
        "thermal_field mode requires FEniCS heat equation solver. "
        "Use rolling_plane mode with friction-based thermal model "
        "for rolling primitives, or implement FEniCS solver for "
        "static surface applications."
    )


def simulate_structural_load(mesh, ctx):
    """
    Stub: FEM analysis under applied load.
    Implementation requires FEniCS linear elasticity solver.

    When implemented, this will:
    1. Generate volume mesh
    2. Apply uniaxial tension/compression
    3. Solve for displacement field
    4. Compute effective Poisson ratio from lateral/axial strain
    5. Return Poisson ratio and stress distribution
    """
    raise NotImplementedError(
        "structural_load mode requires FEniCS elasticity solver. "
        "Implement for auxetic lattice validation."
    )


def simulate_fluid_flow(mesh, ctx):
    """
    Stub: fluid flow through channel geometry.
    Implementation requires OpenFOAM or FEniCS Stokes solver.

    When implemented, this will:
    1. Define channel geometry from surface mesh
    2. Apply inlet/outlet boundary conditions
    3. Solve Navier-Stokes or Stokes equations
    4. Extract pressure drop, velocity uniformity, mixing metric
    5. Return flow distribution scores
    """
    raise NotImplementedError(
        "fluid_flow mode requires OpenFOAM or FEniCS Stokes solver. "
        "Implement for helicoidal channel and Murray branching validation."
    )
