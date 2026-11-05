"""
Phase 3: Self-Righting Dynamics — SRT, SRE, BOA
=================================================
Substrate Geometry Part III (arXiv:2604.17095): Gömböc

For a convex body on a flat surface, we simulate the rolling dynamics
by tracking the gravity direction's path on the COM height landscape.
The body rolls "downhill" on h(d) until it reaches the stable equilibrium.

Metrics:
  SRT  — Self-Righting Time: mean angular distance traveled to reach
         the stable equilibrium from random starting orientations.
         (Proxy for time — actual time depends on scale, mass, friction.)
  SRT_max — Maximum angular distance from any starting orientation.
  SRE  — Self-Righting Energy: mean potential energy drop (h_start - h_final)
         normalized by body weight. Proportional to kinetic energy dissipated.
  BOA  — Basin of Attraction width: for ECS=1 bodies this is 360° (the
         entire sphere). For ECS>1, it's the angular width of the largest
         basin. Measured as the solid angle fraction of the dominant basin.
  DPR  — Density Perturbation Robustness: computed in Phase 4.

Method:
  For each starting direction d_0, simulate gradient descent on h(d)
  along the sphere surface until convergence. Record path length (angular)
  and height drop.

Outputs:
  - ../results/dynamics.json
"""

import numpy as np
import trimesh
import json
import os
import time
from scipy.spatial import cKDTree

MESH_DIR = os.path.join(os.path.dirname(__file__), '..', 'meshes')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULT_DIR, exist_ok=True)


def fibonacci_sphere(n):
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def com_height(mesh, d):
    """COM height above support plane for gravity direction d."""
    return mesh.center_mass @ d - np.min(mesh.vertices @ d)


def gradient_on_sphere(mesh, d, delta=1e-4):
    """Compute gradient of h(d) in the tangent plane of S² at d."""
    # Build orthonormal tangent basis
    if abs(d[2]) < 0.9:
        t1 = np.cross(d, [0, 0, 1])
    else:
        t1 = np.cross(d, [1, 0, 0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(d, t1)

    h0 = com_height(mesh, d)
    d1 = d + delta * t1; d1 /= np.linalg.norm(d1)
    d2 = d + delta * t2; d2 /= np.linalg.norm(d2)

    g1 = (com_height(mesh, d1) - h0) / delta
    g2 = (com_height(mesh, d2) - h0) / delta

    return g1 * t1 + g2 * t2, h0


def simulate_rolling(mesh, d_start, step_size=0.02, max_steps=2000, tol=1e-8):
    """Simulate rolling from d_start by gradient descent on h(d).

    Returns:
        total_angle: total angular distance traveled (radians)
        h_drop: height drop from start to finish
        n_steps: number of steps taken
        d_final: final direction
    """
    d = d_start.copy()
    h_start = com_height(mesh, d)
    total_angle = 0.0

    for step in range(max_steps):
        grad, h_current = gradient_on_sphere(mesh, d)
        grad_mag = np.linalg.norm(grad)

        if grad_mag < tol:
            break

        # Step in negative gradient direction
        d_new = d - step_size * grad / grad_mag
        d_new /= np.linalg.norm(d_new)

        # Angular distance of this step
        cos_angle = np.clip(d @ d_new, -1, 1)
        angle = np.arccos(cos_angle)
        total_angle += angle

        d = d_new

    h_final = com_height(mesh, d)
    h_drop = h_start - h_final

    return total_angle, h_drop, step + 1, d


def compute_dynamics(mesh, label, n_orientations=500):
    """Compute SRT, SRE, BOA for a mesh."""
    t0 = time.time()

    directions = fibonacci_sphere(n_orientations)

    angular_distances = []
    height_drops = []
    final_directions = []

    for i, d_start in enumerate(directions):
        angle, h_drop, steps, d_final = simulate_rolling(mesh, d_start)
        angular_distances.append(angle)
        height_drops.append(h_drop)
        final_directions.append(d_final)

    angular_distances = np.array(angular_distances)
    height_drops = np.array(height_drops)
    final_dirs = np.array(final_directions)

    # SRT: mean angular distance (in degrees)
    srt_mean = float(np.degrees(np.mean(angular_distances)))
    srt_max = float(np.degrees(np.max(angular_distances)))
    srt_std = float(np.degrees(np.std(angular_distances)))

    # SRE: mean height drop (normalized — this is proportional to energy)
    sre_mean = float(np.mean(height_drops))
    sre_max = float(np.max(height_drops))

    # BOA: cluster final directions, find dominant basin fraction
    tree = cKDTree(final_dirs)
    _, neighbors = tree.query(final_dirs, k=min(13, n_orientations))
    heights_at_finals = np.array([com_height(mesh, d) for d in final_dirs])

    # Simple clustering: all finals within 10° of each other are same basin
    visited = np.zeros(n_orientations, dtype=bool)
    basins = []
    for i in range(n_orientations):
        if visited[i]:
            continue
        # Find all points within 10° angular distance
        cos_thresh = np.cos(np.radians(10))
        similarities = final_dirs @ final_dirs[i]
        cluster = np.where((similarities > cos_thresh) & ~visited)[0]
        visited[cluster] = True
        basins.append(len(cluster))

    basins.sort(reverse=True)
    boa = float(basins[0] / n_orientations)  # fraction of S² in dominant basin
    n_basins = len(basins)

    elapsed = time.time() - t0

    result = {
        'label': label,
        'n_orientations': n_orientations,
        'ECS_dynamic': n_basins,
        'SRT_mean_deg': round(srt_mean, 2),
        'SRT_max_deg': round(srt_max, 2),
        'SRT_std_deg': round(srt_std, 2),
        'SRE_mean': round(sre_mean, 6),
        'SRE_max': round(sre_max, 6),
        'BOA_fraction': round(boa, 4),
        'elapsed_seconds': round(elapsed, 1),
    }

    print(f"  {label:35s}  ECS={n_basins:2d}  SRT={srt_mean:6.1f}+/-{srt_std:4.1f} deg  "
          f"SRE={sre_mean:.5f}  BOA={boa:.3f}  {elapsed:.0f}s")

    return result


def main():
    print("\n" + "=" * 70)
    print("  PHASE 3: SELF-RIGHTING DYNAMICS")
    print("  Metrics: SRT (angular distance), SRE (energy drop), BOA (basin width)")
    print("=" * 70)

    results = {
        'phase': 'Phase 3: Self-Righting Dynamics',
        'method': 'Gradient descent on COM height landscape over S2',
        'scores': [],
    }

    # Meshes to test
    meshes_to_test = [
        ('gomboc_verified_ecs1_medres.stl', 'Gomboc (verified ECS=1)'),
    ]

    baselines = [
        ('baselines/sphere.stl', 'Sphere'),
        ('baselines/cylinder_LD2.stl', 'Cylinder (L/D=2)'),
        ('baselines/hemisphere.stl', 'Hemisphere'),
        ('baselines/ellipsoid.stl', 'Ellipsoid'),
        ('baselines/cube.stl', 'Cube'),
        ('baselines/capsule.stl', 'Capsule'),
    ]

    print(f"\n  {'Geometry':35s}  {'ECS':>4s}  {'SRT mean':>12s}  "
          f"{'SRE mean':>9s}  {'BOA':>5s}")
    print(f"  {'-'*35}  {'-'*4}  {'-'*12}  {'-'*9}  {'-'*5}")

    for filename, label in meshes_to_test + baselines:
        path = os.path.join(MESH_DIR, filename)
        if not os.path.exists(path):
            print(f"  SKIP: {path} not found")
            continue
        mesh = trimesh.load(path)
        result = compute_dynamics(mesh, label, n_orientations=200)
        results['scores'].append(result)

    # === Comparison table ===
    print("\n" + "=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    print(f"  {'Geometry':35s}  {'ECS':>3s}  {'SRT_mean':>8s}  {'SRT_max':>8s}  "
          f"{'SRE':>8s}  {'BOA':>5s}")
    print(f"  {'-'*35}  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*5}")

    for s in results['scores']:
        print(f"  {s['label']:35s}  {s['ECS_dynamic']:3d}  "
              f"{s['SRT_mean_deg']:7.1f}d  {s['SRT_max_deg']:7.1f}d  "
              f"{s['SRE_mean']:8.5f}  {s['BOA_fraction']:.3f}")

    # Save
    result_path = os.path.join(RESULT_DIR, 'dynamics.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {result_path}")


if __name__ == '__main__':
    main()
