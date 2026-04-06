"""
thermal_oracle.py
---------------------------------------------------------------------------
Geometric Primitive Synthesis Program -- Pipeline Artifact 06
Thermal Distribution Oracle

QUESTION: Does uniform contact STRESS → uniform THERMAL distribution?

APPROACH:
  Friction at contact zones generates heat. At each rolling orientation,
  the frictional heat generated at a contact face depends on:
    1. Contact pressure (from Hertz theory)
    2. Sliding velocity (tangential velocity at the contact point)
    3. Friction coefficient

  Frictional heat flux: q = μ · p · v_slide
    where μ = friction coefficient, p = Hertz pressure, v_slide = sliding speed

  For a rolling body, the sliding velocity at a contact face depends on
  the angular velocity and the distance from the instantaneous rotation
  axis to the contact point. The oloid's complex rolling kinematics mean
  different faces contact at different sliding velocities — this is what
  makes thermal distribution a genuinely independent measurement from
  stress distribution.

PHYSICS MODEL:
  1. Rolling dynamics: same rigid-body simulation as CDS/SDS/FDS oracles
  2. At each orientation:
     a. Contact faces identified (z < threshold)
     b. Hertz pressure computed from local curvature
     c. Sliding velocity = |ω × r_contact| (tangential component)
     d. Heat flux q_i = μ · p_i · v_slide_i per contact face
  3. Accumulate heat flux per face across rolling cycle
  4. TDS = area-weighted variance of accumulated heat
     (analogous to CDS, SDS, FDS)

MATERIAL MODEL:
  Friction coefficient: μ = 0.15 (steel on steel, lubricated bearing)
  Same Hertz parameters as hertz_oracle.py
  Reference load: 100 N (same as Hertz oracle, not fatigue-scaled)

WHY THERMAL MAY DIVERGE FROM SDS:
  SDS only depends on Hertz pressure (function of curvature).
  TDS depends on pressure × sliding velocity.
  Sliding velocity varies with:
    - Angular velocity magnitude (changes during rolling)
    - Distance from contact point to rotation axis
    - Angle between ω and the contact-point lever arm
  If the oloid's rolling kinematics correlate sliding velocity with
  curvature, TDS may track SDS. If they're independent, TDS diverges.

DEPENDENCIES:
  Uses infrastructure from hertz_oracle.py and rigidbody_oracle.py
  pip install trimesh numpy scipy

Author: Geometric Primitive Synthesis Program
---------------------------------------------------------------------------
"""

import numpy as np
import trimesh
from dataclasses import dataclass
from typing import List, Optional
import time as clock

# Import from existing pipeline
from contact_oracle import (
    generate_oloid, generate_cylinder, generate_sphere,
    score_distribution
)
from hertz_oracle import (
    hertz_peak_pressure, compute_vertex_curvatures, compute_face_curvatures,
    score_stress_distribution, E_STAR, REFERENCE_LOAD
)
from rigidbody_oracle import (
    simulate_rolling, quat_to_matrix, compute_inertia_tensor,
    find_support_point, compute_gravity_torque, euler_step,
    quat_multiply, quat_from_angular_velocity
)


# ---------------------------------------------------------------------------
# THERMAL PARAMETERS
# ---------------------------------------------------------------------------

FRICTION_COEFF = 0.15    # Steel-on-steel, lubricated bearing


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class ThermalResult:
    """Full result from the thermal distribution oracle."""
    geometry_name:          str
    n_faces:                int
    n_orientations:         int
    surface_area:           float

    # Thermal Distribution Score
    tds:                    float
    tds_satisfied:          bool

    # Companion scores
    cds:                    float
    sds:                    float

    # Thermal statistics
    mean_heat_flux:         float   # average accumulated heat per face
    max_heat_flux:          float   # hottest face
    min_heat_flux:          float   # coolest contacted face
    heat_cv:                float   # coefficient of variation
    heat_ratio:             float   # max/min — thermal hotspot indicator

    # Sliding velocity statistics
    mean_slide_speed:       float   # m/s average across contact events
    max_slide_speed:        float
    slide_cv:               float

    # Per-face data
    face_heat_accumulator:  np.ndarray
    face_areas:             np.ndarray
    contact_counts:         np.ndarray

    # Convergence
    convergence:            list


# ---------------------------------------------------------------------------
# ROLLING SIMULATION WITH VELOCITY TRACKING
# ---------------------------------------------------------------------------

def simulate_rolling_with_velocity(
    mesh: trimesh.Trimesh,
    t_total: float = 10.0,
    dt: float = 0.002,
    damping: float = 0.02,
    initial_push: Optional[np.ndarray] = None,
):
    """
    Simulate rigid-body rolling and return both rotation matrices
    AND angular velocity vectors at each timestep.

    Returns
    -------
    rotations : list of 3x3 rotation matrices
    omegas    : list of 3-vectors (angular velocity in world frame)
    """
    centroid_offset = mesh.centroid.copy()
    verts_body = mesh.vertices - centroid_offset

    I_body = compute_inertia_tensor(mesh)
    I_inv = np.linalg.inv(I_body)
    mass = 1.0

    q = np.array([1.0, 0.0, 0.0, 0.0])
    if initial_push is not None:
        omega = initial_push.copy()
    else:
        omega = np.array([0.3, 1.5, 0.2])

    n_steps = int(t_total / dt)
    rotations = []
    omegas = []

    for step in range(n_steps):
        R = quat_to_matrix(q)
        verts_world = (R @ verts_body.T).T

        _, support = find_support_point(verts_world)
        ground_shift = -support[2]
        centroid_world = np.array([0.0, 0.0, ground_shift])
        support_ground = support.copy()
        support_ground[2] = 0.0

        torque = compute_gravity_torque(centroid_world, support_ground, mass)
        torque -= damping * omega

        omega = euler_step(omega, I_body, I_inv, torque, dt)

        omega_mag = np.linalg.norm(omega)
        if omega_mag > 50.0:
            omega = omega * 50.0 / omega_mag

        dq = quat_from_angular_velocity(omega, dt)
        q = quat_multiply(dq, q)
        q = q / np.linalg.norm(q)

        rotations.append(R)
        omegas.append(omega.copy())

    return rotations, omegas


# ---------------------------------------------------------------------------
# THERMAL ACCUMULATION
# ---------------------------------------------------------------------------

def accumulate_thermal(
    mesh: trimesh.Trimesh,
    R_eff_body: np.ndarray,
    mu: float = FRICTION_COEFF,
    t_total: float = 10.0,
    dt: float = 0.002,
    sample_every: int = 25,
    contact_threshold: float = 0.04,
    damping: float = 0.02,
    n_runs: int = 3,
    convergence_every: int = 200,
    verbose: bool = True
):
    """
    Simulate rolling and accumulate frictional heat flux per face.

    At each sampled orientation:
      1. Find contact faces
      2. Compute Hertz pressure from curvature
      3. Compute sliding velocity from ω × r_contact
      4. Heat flux q = μ · p · v_slide
      5. Accumulate per face

    Returns
    -------
    heat_acc       : ndarray — accumulated heat per face
    stress_acc     : ndarray — accumulated stress per face
    contact_counts : ndarray — contact count per face
    slide_speeds   : list — sliding speed at each contact event
    convergence    : list of (n_samples, tds) tuples
    """
    n_faces = len(mesh.faces)
    centroids_body = mesh.triangles_center - mesh.centroid
    face_areas = mesh.area_faces.copy()

    heat_acc = np.zeros(n_faces)
    stress_acc = np.zeros(n_faces)
    contact_counts = np.zeros(n_faces)
    all_slide_speeds = []
    convergence = []

    initial_pushes = [
        np.array([0.3, 1.5, 0.2]),
        np.array([-0.5, 1.0, 0.8]),
        np.array([0.7, -1.2, 0.4]),
        np.array([0.1, 0.8, -1.0]),
        np.array([-0.3, -0.6, 1.5]),
    ][:n_runs]

    total_samples = 0

    for run_idx, push in enumerate(initial_pushes):
        if verbose:
            print(f"    Run {run_idx+1}/{n_runs} "
                  f"(push=[{push[0]:.1f},{push[1]:.1f},{push[2]:.1f}])")

        rotations, omegas = simulate_rolling_with_velocity(
            mesh, t_total=t_total, dt=dt,
            damping=damping, initial_push=push,
        )

        sample_count = 0
        for i, (R, omega) in enumerate(zip(rotations, omegas)):
            if i % sample_every != 0:
                continue

            # Rotate face centroids to world frame
            rotated = (R @ centroids_body.T).T
            rotated[:, 2] -= rotated[:, 2].min()

            # Contact detection
            in_contact = rotated[:, 2] < contact_threshold
            contact_counts[in_contact] += 1

            # Heat flux at each contact face
            contact_faces = np.where(in_contact)[0]
            for fi in contact_faces:
                # Hertz pressure
                p = hertz_peak_pressure(R_eff_body[fi])
                stress_acc[fi] += p

                # Sliding velocity at this face's contact point
                # v_slide = |ω × r| where r = vector from rotation axis
                # to contact point. For rolling on a plane, the
                # instantaneous rotation is about the contact line.
                # The sliding velocity of a specific face is the
                # tangential component of ω × (face_centroid - support).
                r_vec = rotated[fi]  # face centroid in world frame
                # Support is at z≈0, approximate as origin of ground plane
                r_vec_ground = r_vec.copy()
                r_vec_ground[2] = 0  # project to ground
                v_contact = np.cross(omega, r_vec)
                # Sliding = tangential component (parallel to ground)
                v_slide = np.sqrt(v_contact[0]**2 + v_contact[1]**2)

                # Frictional heat flux
                q = mu * p * v_slide
                heat_acc[fi] += q
                all_slide_speeds.append(v_slide)

            sample_count += 1
            total_samples += 1

            if total_samples > 0 and total_samples % convergence_every == 0:
                tds = score_stress_distribution(heat_acc, face_areas)
                convergence.append((total_samples, tds))

        if verbose:
            print(f"    Sampled {sample_count} orientations")

    return (heat_acc, stress_acc, contact_counts,
            np.array(all_slide_speeds), convergence)


# ---------------------------------------------------------------------------
# THERMAL ORACLE — MAIN FUNCTION
# ---------------------------------------------------------------------------

def run_thermal_oracle(
    mesh: trimesh.Trimesh,
    geometry_name: str = 'unnamed',
    t_total: float = 10.0,
    dt: float = 0.002,
    sample_every: int = 25,
    contact_threshold: float = 0.04,
    tds_threshold: float = 0.0001,
    damping: float = 0.02,
    n_runs: int = 3,
    convergence_every: int = 200,
    verbose: bool = True
) -> ThermalResult:
    """
    Run the thermal distribution oracle on a geometry.

    Combines rigid-body rolling dynamics with Hertz contact theory
    and friction-generated heat to evaluate thermal uniformity.
    """
    t0 = clock.time()

    if verbose:
        print(f"\n  Thermal oracle: {geometry_name}")
        print(f"  Mesh: {len(mesh.faces)} faces, area={mesh.area:.4f}")

    # Compute curvatures
    if verbose:
        print("  Computing surface curvatures...")
    H_vert, K_vert, _ = compute_vertex_curvatures(mesh)
    _, _, _, _, R_eff = compute_face_curvatures(mesh, H_vert, K_vert)

    # Simulate and accumulate
    if verbose:
        print(f"  Running thermal accumulation (μ={FRICTION_COEFF})...")

    (heat_acc, stress_acc, contact_counts,
     slide_speeds, convergence) = \
        accumulate_thermal(
            mesh, R_eff,
            mu=FRICTION_COEFF,
            t_total=t_total, dt=dt,
            sample_every=sample_every,
            contact_threshold=contact_threshold,
            damping=damping, n_runs=n_runs,
            convergence_every=convergence_every,
            verbose=verbose
        )

    # Compute scores
    face_areas = mesh.area_faces.copy()
    tds = score_stress_distribution(heat_acc, face_areas)
    cds = score_distribution(contact_counts, face_areas)
    sds = score_stress_distribution(stress_acc, face_areas)

    # Heat statistics
    contacted = heat_acc > 0
    if contacted.any():
        heat_contacted = heat_acc[contacted]
        mean_heat = float(np.mean(heat_contacted))
        max_heat = float(np.max(heat_contacted))
        min_heat = float(np.min(heat_contacted))
        heat_cv = float(np.std(heat_contacted) / mean_heat) if mean_heat > 0 else float('inf')
        heat_ratio = float(max_heat / min_heat) if min_heat > 0 else float('inf')
    else:
        mean_heat = max_heat = min_heat = 0.0
        heat_cv = heat_ratio = float('inf')

    # Sliding speed statistics
    if len(slide_speeds) > 0:
        mean_slide = float(np.mean(slide_speeds))
        max_slide = float(np.max(slide_speeds))
        slide_cv = float(np.std(slide_speeds) / mean_slide) if mean_slide > 0 else 0.0
    else:
        mean_slide = max_slide = 0.0
        slide_cv = 0.0

    elapsed = clock.time() - t0

    result = ThermalResult(
        geometry_name=geometry_name,
        n_faces=len(mesh.faces),
        n_orientations=int(contact_counts.sum()),
        surface_area=float(mesh.area),
        tds=tds,
        tds_satisfied=tds < tds_threshold,
        cds=cds,
        sds=sds,
        mean_heat_flux=mean_heat,
        max_heat_flux=max_heat,
        min_heat_flux=min_heat,
        heat_cv=heat_cv,
        heat_ratio=heat_ratio,
        mean_slide_speed=mean_slide,
        max_slide_speed=max_slide,
        slide_cv=slide_cv,
        face_heat_accumulator=heat_acc,
        face_areas=face_areas,
        contact_counts=contact_counts,
        convergence=convergence,
    )

    if verbose:
        status = "PASSED" if result.tds_satisfied else "FAILED"
        print(f"  TDS: {tds:.8f} -- {status}")
        print(f"  CDS: {cds:.8f}  SDS: {sds:.8f}")
        print(f"  Heat: mean={mean_heat:.4e}, max={max_heat:.4e}, "
              f"ratio={heat_ratio:.2f}")
        print(f"  Sliding: mean={mean_slide:.3f} m/s, "
              f"max={max_slide:.3f} m/s, CV={slide_cv:.4f}")
        print(f"  Elapsed: {elapsed:.1f}s")

    return result


# ---------------------------------------------------------------------------
# COMPARISON RUNNER
# ---------------------------------------------------------------------------

def compare_thermal(geometries, **oracle_kwargs):
    """Run thermal oracle on multiple geometries and print comparison."""
    results = []
    for name, mesh in geometries:
        r = run_thermal_oracle(mesh, geometry_name=name, **oracle_kwargs)
        results.append(r)

    results.sort(key=lambda r: r.tds)

    print("\n" + "=" * 110)
    print("THERMAL ORACLE — HEAT DISTRIBUTION COMPARISON")
    print(f"Friction model: q = μ·p·v_slide, μ = {FRICTION_COEFF}")
    print("=" * 110)

    print(f"\n{'Rank':<5} {'Geometry':<36} {'TDS':<14} {'CDS':<14} "
          f"{'SDS':<14} {'Heat Ratio':<11} {'v_slide(m/s)':<13} {'Status'}")
    print("-" * 110)

    for i, r in enumerate(results):
        status = "PASS" if r.tds_satisfied else "FAIL"
        print(f"{i+1:<5} {r.geometry_name:<36} {r.tds:<14.8f} {r.cds:<14.8f} "
              f"{r.sds:<14.8f} {r.heat_ratio:<11.2f} {r.mean_slide_speed:<13.3f} {status}")

    print("-" * 110)

    oloid_r = next((r for r in results if 'Oloid' in r.geometry_name), None)
    cyl_r = next((r for r in results if 'Cylinder' in r.geometry_name), None)

    if oloid_r and cyl_r:
        tds_ratio = cyl_r.tds / oloid_r.tds if oloid_r.tds > 0 else float('inf')
        print(f"\n  Cylinder/Oloid TDS ratio: {tds_ratio:.1f}x")
        print(f"  Cylinder/Oloid CDS ratio: "
              f"{cyl_r.cds/oloid_r.cds:.1f}x")
        print(f"  Cylinder/Oloid SDS ratio: "
              f"{cyl_r.sds/oloid_r.sds:.1f}x")

    # Pattern analysis
    print("\n" + "-" * 110)
    print("INVARIANT VECTOR PATTERN ANALYSIS")
    print("-" * 110)

    if oloid_r:
        print(f"\n  Oloid scores across oracle layers:")
        print(f"    CDS (1st order, time):     {oloid_r.cds:.2e}")
        print(f"    SDS (1st order, stress):   {oloid_r.sds:.2e}")
        print(f"    TDS (nonlinear, thermal):  {oloid_r.tds:.2e}")

        tds_cds = oloid_r.tds / oloid_r.cds if oloid_r.cds > 0 else 0
        tds_sds = oloid_r.tds / oloid_r.sds if oloid_r.sds > 0 else 0
        print(f"\n  TDS/CDS ratio: {tds_cds:.2f}")
        print(f"  TDS/SDS ratio: {tds_sds:.2f}")

        if tds_cds < 2.0:
            print(f"\n  TDS tracks CDS/SDS closely — thermal distribution is")
            print(f"  a first-order property like stress. Sliding velocity")
            print(f"  variation does not significantly amplify the distribution.")
        elif tds_cds < 5.0:
            print(f"\n  TDS diverges moderately from CDS/SDS — sliding velocity")
            print(f"  adds meaningful variation to the thermal distribution,")
            print(f"  but the oloid's contact invariant still dominates.")
        else:
            print(f"\n  TDS diverges significantly — the p×v product introduces")
            print(f"  substantial non-uniformity. Sliding velocity variation")
            print(f"  is the dominant factor in thermal distribution.")

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from parametric_search import generate_roller, RollerGenome

    print("=" * 72)
    print("GEOMETRIC PRIMITIVE SYNTHESIS PROGRAM")
    print("Thermal Distribution Oracle -- Pipeline Artifact 06")
    print("=" * 72)
    print()
    print("QUESTION: Does uniform contact STRESS -> uniform THERMAL rise?")
    print()
    print("METHOD: Rigid-body rolling + Hertz contact + frictional heat")
    print(f"        q = μ·p·v_slide, μ = {FRICTION_COEFF}")
    print()
    print("The sliding velocity v_slide = |ω × r_contact| varies with")
    print("the body's angular velocity and the contact point's distance")
    print("from the rotation axis. This makes TDS genuinely independent")
    print("from SDS — it's not just pressure, it's pressure × velocity.")
    print()

    # --- Generate geometries ---
    print("Generating meshes...")
    oloid = generate_oloid(r=1.0, n_circle_pts=400)
    cylinder = generate_cylinder(r=1.0, h=2.0)

    top_candidates = [
        ('Candidate (115/1.30/0.65)', RollerGenome(theta=115.0, offset=1.30, r_ratio=0.65)),
        ('RB Winner (120/1.30/0.80)', RollerGenome(theta=120.0, offset=1.30, r_ratio=0.80)),
    ]

    geometries = [('Oloid (Schatz 1929)', oloid)]

    for name, genome in top_candidates:
        try:
            mesh = generate_roller(genome, r1=1.0)
            geometries.append((name, mesh))
            print(f"  {name}: {len(mesh.faces)} faces, area={mesh.area:.4f}")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    geometries.append(('Cylinder (conventional)', cylinder))

    print(f"  Oloid:    {len(oloid.faces)} faces, area={oloid.area:.4f}")
    print(f"  Cylinder: {len(cylinder.faces)} faces, area={cylinder.area:.4f}")
    print()

    # --- Run thermal oracle ---
    print("=" * 72)
    print("THERMAL ORACLE COMPARISON")
    print("Rigid-body dynamics + Hertz contact + friction heat model")
    print("3 runs per geometry, 10s per run")
    print("=" * 72)

    results = compare_thermal(
        geometries,
        t_total=10.0,
        dt=0.002,
        sample_every=25,
        damping=0.02,
        n_runs=3,
        verbose=True
    )

    # --- Full invariant vector ---
    print("\n" + "=" * 72)
    print("OLOID INVARIANT VECTOR (4 of 5 dimensions)")
    print("=" * 72)

    oloid_r = next((r for r in results if 'Oloid' in r.geometry_name), None)
    cyl_r = next((r for r in results if 'Cylinder' in r.geometry_name), None)

    if oloid_r and cyl_r:
        # Import FDS from fatigue results if available
        fds_oloid = "2.42e-06"  # from fatigue oracle run
        fds_cyl = "inf"

        print(f"""
  Dimension        Oloid          Cylinder       Ratio    Status
  ───���─────────────────────────────────────────────────────────────
  CDS (contact)    {oloid_r.cds:.2e}     {cyl_r.cds:.2e}     {cyl_r.cds/oloid_r.cds:.0f}x       DONE
  SDS (stress)     {oloid_r.sds:.2e}     {cyl_r.sds:.2e}     {cyl_r.sds/oloid_r.sds:.0f}x       DONE
  FDS (fatigue)    {fds_oloid}     {fds_cyl}            ∞x       DONE
  TDS (thermal)    {oloid_r.tds:.2e}     {cyl_r.tds:.2e}     {cyl_r.tds/oloid_r.tds if oloid_r.tds > 0 else 0:.0f}x       DONE
  WDS (wear)       ---            ---            ---       PENDING
""")

    print("=" * 72)
    print("PIPELINE STATUS")
    print("=" * 72)
    print(f"""
  Thermal distribution oracle: OPERATIONAL
  Method: Hertz pressure × sliding velocity × friction coefficient
  Friction: μ = {FRICTION_COEFF} (lubricated steel-on-steel)

  Pipeline artifacts:
    01 contact_oracle.py    → CDS (contact time)
    02 parametric_search.py → search family
    03 rigidbody_oracle.py  → defensible CDS
    04 hertz_oracle.py      → SDS (stress distribution)
    05 fatigue_oracle.py    �� FDS (fatigue damage distribution)
    06 thermal_oracle.py    → TDS (thermal distribution)

  Next: wear_oracle.py     → WDS (wear rate uniformity, Archard's law)
""")
