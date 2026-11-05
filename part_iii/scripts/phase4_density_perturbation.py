"""
Phase 4: Density Perturbation Experiment
==========================================
Substrate Geometry Part III (arXiv:2604.17095): Gömböc

Central experiment:
  How much ballast does a conventional geometry need to match the
  Gömböc's ECS=1? If the Gömböc at uniform density outperforms a
  cylinder at 10% ballast, that's a meaningful engineering result.

Method:
  For non-uniform density, the COM shifts from the geometric centroid.
  We model this as: COM = (1-w)*centroid + w*bottom_point
  where w is the ballast weight fraction (0 = uniform, 1 = all weight at bottom).

  The COM height function becomes:
    h(d) = COM_shifted · d - min(vertices · d)

  We sweep w from 0 to 0.30 for each geometry and measure ECS at each level.

Experiment matrix:
  Geometries: Gömböc (verified), Sphere, Cylinder, Ellipsoid, Capsule, Cube
  Ballast levels: 0%, 1%, 2%, 3%, 5%, 7%, 10%, 15%, 20%, 30%
  At each (geometry, ballast): compute ECS, h_range, BOA
"""

import numpy as np
import trimesh
import json
import os
import time
from scipy.spatial import cKDTree

MESH_DIR = os.path.join(os.path.dirname(__file__), '..', 'meshes')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def fibonacci_sphere(n):
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    return np.stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)], axis=1)


def compute_ecs_with_shifted_com(mesh, com_shifted, n_samples=5000):
    """Compute ECS with an arbitrary COM position."""
    directions = fibonacci_sphere(n_samples)
    verts = mesh.vertices

    projections = directions @ verts.T
    support_proj = np.min(projections, axis=1)
    com_proj = directions @ com_shifted
    heights = com_proj - support_proj

    # Drainage basins
    tree = cKDTree(directions)
    _, neighbors = tree.query(directions, k=13)

    sink_of = np.full(n_samples, -1, dtype=int)
    def find_sink(i):
        visited = []
        current = i
        while sink_of[current] == -1:
            visited.append(current)
            nbr = neighbors[current]
            lowest = nbr[np.argmin(heights[nbr])]
            if lowest == current:
                for v in visited:
                    sink_of[v] = current
                return current
            current = lowest
        for v in visited:
            sink_of[v] = sink_of[current]
        return sink_of[current]

    for i in range(n_samples):
        if sink_of[i] == -1:
            find_sink(i)

    unique_sinks = np.unique(sink_of)
    sink_heights = heights[unique_sinks]
    h_range = float(np.max(heights) - np.min(heights))
    merge_threshold = max(h_range * 0.01, 1e-7)

    order = np.argsort(sink_heights)
    sorted_sinks = unique_sinks[order]
    sorted_heights = sink_heights[order]

    merged = []
    current_group = [sorted_sinks[0]]
    current_h = sorted_heights[0]
    for i in range(1, len(sorted_sinks)):
        if sorted_heights[i] - current_h <= merge_threshold:
            current_group.append(sorted_sinks[i])
        else:
            merged.append(current_group)
            current_group = [sorted_sinks[i]]
            current_h = sorted_heights[i]
    merged.append(current_group)

    ecs = len(merged)

    basin_sizes = []
    for group in merged:
        total = sum(int(np.sum(sink_of == s)) for s in group)
        basin_sizes.append(total / n_samples)

    boa = max(basin_sizes)

    return ecs, h_range, boa, basin_sizes[:5]


def run_experiment(mesh, label, ballast_levels):
    """Sweep ballast levels for a single geometry."""
    centroid = mesh.center_mass
    # "Bottom" = the point with minimum z coordinate (for the canonical orientation)
    bottom_point = mesh.vertices[np.argmin(mesh.vertices[:, 2])]

    results = []
    for w in ballast_levels:
        com_shifted = (1 - w) * centroid + w * bottom_point
        ecs, h_range, boa, basins = compute_ecs_with_shifted_com(mesh, com_shifted)
        results.append({
            'ballast_fraction': round(w, 3),
            'ECS': ecs,
            'h_range': round(h_range, 6),
            'BOA': round(boa, 4),
            'basins': [round(b, 4) for b in basins],
            'com_shift': round(float(np.linalg.norm(com_shifted - centroid)), 6),
        })
    return results


def main():
    print("\n" + "=" * 80)
    print("  PHASE 4: DENSITY PERTURBATION EXPERIMENT")
    print("  Question: How much ballast to match ECS=1?")
    print("=" * 80)

    ballast_levels = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]

    meshes = [
        ('gomboc_verified_ecs1_medres.stl', 'Gomboc'),
        ('baselines/sphere.stl', 'Sphere'),
        ('baselines/cylinder_LD2.stl', 'Cylinder'),
        ('baselines/ellipsoid.stl', 'Ellipsoid'),
        ('baselines/capsule.stl', 'Capsule'),
        ('baselines/cube.stl', 'Cube'),
    ]

    all_results = {}

    for filename, label in meshes:
        path = os.path.join(MESH_DIR, filename)
        if not os.path.exists(path):
            print(f"  SKIP: {path}")
            continue

        mesh = trimesh.load(path)
        print(f"\n  --- {label} ---")
        print(f"  {'Ballast':>8s}  {'ECS':>4s}  {'h_range':>8s}  {'BOA':>6s}  Basins")
        print(f"  {'-'*8}  {'-'*4}  {'-'*8}  {'-'*6}  {'-'*25}")

        t0 = time.time()
        sweep = run_experiment(mesh, label, ballast_levels)
        elapsed = time.time() - t0

        for r in sweep:
            marker = " <-- ECS=1!" if r['ECS'] == 1 else ""
            print(f"  {r['ballast_fraction']:8.1%}  {r['ECS']:4d}  "
                  f"{r['h_range']:8.5f}  {r['BOA']:6.3f}  {r['basins']}{marker}")

        # Find the threshold: minimum ballast for ECS=1
        ecs1_threshold = None
        for r in sweep:
            if r['ECS'] == 1:
                ecs1_threshold = r['ballast_fraction']
                break

        if ecs1_threshold is not None:
            print(f"  --> ECS=1 achieved at {ecs1_threshold:.0%} ballast")
        else:
            print(f"  --> ECS=1 NOT achieved even at {ballast_levels[-1]:.0%} ballast")

        all_results[label] = {
            'sweep': sweep,
            'ecs1_threshold': ecs1_threshold,
            'elapsed': round(elapsed, 1),
        }

    # === Summary Table ===
    print("\n" + "=" * 80)
    print("  SUMMARY: MINIMUM BALLAST FOR ECS=1")
    print("=" * 80)
    print(f"  {'Geometry':20s}  {'Ballast for ECS=1':>18s}  {'ECS at 0%':>10s}")
    print(f"  {'-'*20}  {'-'*18}  {'-'*10}")

    for label, data in all_results.items():
        uniform_ecs = data['sweep'][0]['ECS']
        thresh = data['ecs1_threshold']
        thresh_str = f"{thresh:.0%}" if thresh is not None else ">30%"
        if thresh == 0.0:
            thresh_str = "0% (inherent)"
        print(f"  {label:20s}  {thresh_str:>18s}  {uniform_ecs:>10d}")

    # Key finding
    print("\n  --- KEY FINDING ---")
    gomboc_data = all_results.get('Gomboc', {})
    cylinder_data = all_results.get('Cylinder', {})

    if gomboc_data.get('ecs1_threshold') == 0.0 and cylinder_data.get('ecs1_threshold') is not None:
        cyl_thresh = cylinder_data['ecs1_threshold']
        print(f"  The Gomboc achieves ECS=1 at 0% ballast (pure geometry).")
        print(f"  The Cylinder requires {cyl_thresh:.0%} ballast to match.")
        print(f"  The Gomboc's geometric invariant eliminates the need for")
        print(f"  {cyl_thresh*100:.0f}% of the ballast mass a cylinder requires.")
    elif gomboc_data.get('ecs1_threshold') == 0.0 and cylinder_data.get('ecs1_threshold') is None:
        print(f"  The Gomboc achieves ECS=1 at 0% ballast (pure geometry).")
        print(f"  The Cylinder does NOT achieve ECS=1 even at 30% ballast.")
        print(f"  The geometric invariant provides what ballast cannot.")

    # Save
    result_path = os.path.join(RESULT_DIR, 'density_perturbation.json')
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {result_path}")


if __name__ == '__main__':
    main()
