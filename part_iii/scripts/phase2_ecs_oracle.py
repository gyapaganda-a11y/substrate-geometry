"""
Phase 2: Equilibrium Count Score (ECS) Oracle — v3
====================================================
Substrate Geometry Part III (arXiv:2604.17095): Gömböc

For a convex body on a flat surface under gravity, stable equilibria
are directions d where the COM is directly above the support point.

Equivalently: the support point for direction d is the vertex with
minimum projection onto d. The equilibrium condition is that d is
parallel to (COM - support_point). Stable equilibria are where COM
height h(d) = (COM - support_point) · d_hat is at a local minimum.

Approach: Evaluate h(d) densely on S², then count distinct basins
by watershed-style analysis. A basin is a connected region of S²
that flows to the same local minimum under steepest descent of h(d).
The number of basins = ECS.

Simpler robust approach used here:
  1. Evaluate h(d) on a dense Fibonacci sphere
  2. Sort all h values
  3. The histogram of h values reveals distinct "valleys" separated by "ridges"
  4. Count valleys = ECS

Actually simplest correct approach:
  For each sampled direction d, assign it to its nearest local minimum
  via greedy descent on the neighbor graph. Count distinct sinks.
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
    """Generate n approximately uniformly distributed points on S²."""
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def com_height_function(mesh, directions):
    """Compute COM height above support plane for each gravity direction.

    For direction d (gravity pointing in direction d), the body rests on
    the vertex with maximum projection onto -d (= minimum projection onto d).
    COM height = COM·d - min(vertices·d).
    """
    com = mesh.center_mass
    vertices = mesh.vertices
    projections = directions @ vertices.T  # (N_dirs, N_verts)
    support_projections = np.min(projections, axis=1)  # (N,)
    com_projections = directions @ com  # (N,)
    heights = com_projections - support_projections  # (N,)
    return heights


def count_basins(directions, heights, k_neighbors=12):
    """Count distinct drainage basins on S² by greedy descent.

    For each sampled point, follow the steepest descent through the
    neighbor graph until reaching a local minimum (sink). Count
    distinct sinks = number of basins = ECS.

    Args:
        directions: (N, 3) unit vectors on S²
        heights: (N,) COM height at each direction
        k_neighbors: number of neighbors in the graph

    Returns:
        n_basins: number of distinct equilibria (ECS)
        sink_indices: index of the sink each point drains to
        basin_sizes: number of points in each basin
    """
    N = len(directions)
    tree = cKDTree(directions)
    _, neighbors = tree.query(directions, k=k_neighbors + 1)  # includes self at [0]

    # For each point, find which neighbor (including self) has lowest height
    # Follow the chain until we reach a point that is its own lowest neighbor (a sink)
    sink_of = np.full(N, -1, dtype=int)

    def find_sink(i):
        """Follow steepest descent from point i to its sink."""
        visited = []
        current = i
        while sink_of[current] == -1:
            visited.append(current)
            # Find lowest neighbor
            nbr = neighbors[current]  # includes self
            nbr_heights = heights[nbr]
            lowest = nbr[np.argmin(nbr_heights)]

            if lowest == current:
                # This is a sink
                for v in visited:
                    sink_of[v] = current
                return current
            current = lowest

        # current already has a sink assigned
        for v in visited:
            sink_of[v] = sink_of[current]
        return sink_of[current]

    for i in range(N):
        if sink_of[i] == -1:
            find_sink(i)

    # Count distinct sinks
    unique_sinks = np.unique(sink_of)

    # Basin sizes
    basin_sizes = {}
    for s in unique_sinks:
        basin_sizes[int(s)] = int(np.sum(sink_of == s))

    # === Merge sinks at similar heights ===
    # Two sinks at nearly the same COM height are likely the same equilibrium
    # fragmented by discretization (e.g., cylinder on-side ring).
    # Merge threshold: sinks within 2% of total h_range are treated as the same equilibrium
    # (the later experiments use 1%).
    sink_heights = heights[unique_sinks]
    h_range = np.max(heights) - np.min(heights)
    merge_threshold = max(h_range * 0.02, 1e-6)  # 2% of total range

    # Sort sinks by height
    order = np.argsort(sink_heights)
    sorted_sinks = unique_sinks[order]
    sorted_heights = sink_heights[order]

    # Greedy merge: group consecutive sinks within threshold
    merged_groups = []
    current_group = [sorted_sinks[0]]
    current_h = sorted_heights[0]

    for i in range(1, len(sorted_sinks)):
        if sorted_heights[i] - current_h <= merge_threshold:
            current_group.append(sorted_sinks[i])
        else:
            merged_groups.append(current_group)
            current_group = [sorted_sinks[i]]
            current_h = sorted_heights[i]
    merged_groups.append(current_group)

    n_basins = len(merged_groups)

    # Rebuild basin_sizes for merged groups
    merged_basin_sizes = {}
    merged_representative = []
    for group in merged_groups:
        # Representative = lowest height member
        rep = min(group, key=lambda s: heights[s])
        total_size = sum(basin_sizes[int(s)] for s in group)
        merged_basin_sizes[int(rep)] = total_size
        merged_representative.append(rep)

    return n_basins, sink_of, merged_basin_sizes, np.array(merged_representative)


def score_mesh(mesh_path, label, n_samples=10000, k_neighbors=12):
    """Compute ECS for a single mesh."""
    t0 = time.time()

    mesh = trimesh.load(mesh_path)

    # Sample directions
    directions = fibonacci_sphere(n_samples)

    # Compute COM height landscape
    heights = com_height_function(mesh, directions)

    # Count drainage basins
    n_basins, sink_of, basin_sizes, unique_sinks = count_basins(
        directions, heights, k_neighbors=k_neighbors)

    # Get sink details
    sink_details = []
    for s in unique_sinks:
        sink_details.append({
            'direction': directions[s].tolist(),
            'height': float(heights[s]),
            'basin_size': int(basin_sizes[s]),
            'basin_fraction': round(basin_sizes[s] / n_samples, 4),
        })
    sink_details.sort(key=lambda x: x['height'])

    # Statistics
    h_min = float(np.min(heights))
    h_max = float(np.max(heights))
    h_range = h_max - h_min

    elapsed = time.time() - t0

    result = {
        'label': label,
        'mesh_file': os.path.basename(mesh_path),
        'n_samples': n_samples,
        'k_neighbors': k_neighbors,
        'n_faces': len(mesh.faces),
        'ECS': n_basins,
        'equilibria': sink_details,
        'com_height_min': h_min,
        'com_height_max': h_max,
        'com_height_range': h_range,
        'elapsed_seconds': round(elapsed, 2),
    }

    print(f"  {label:40s} ECS={n_basins:3d}  h_range={h_range:.6f}  "
          f"basins: {[s['basin_fraction'] for s in sink_details[:5]]}  {elapsed:.1f}s")

    return result


def main():
    print("\n" + "=" * 70)
    print("  PHASE 2: EQUILIBRIUM COUNT SCORE (ECS) ORACLE — v3")
    print("  Method: Drainage basin counting on S² COM height landscape")
    print("=" * 70)

    results = {
        'phase': 'Phase 2: ECS Oracle v3',
        'method': 'Drainage basin counting via greedy descent on S² neighbor graph',
        'n_samples': 10000,
        'k_neighbors': 12,
        'scores': [],
    }

    # === Gömböc meshes ===
    print("\n  --- Gömböc Meshes ---")
    gomboc_files = sorted([f for f in os.listdir(MESH_DIR)
                           if f.startswith('gomboc') and f.endswith('.stl')])
    for f in gomboc_files:
        path = os.path.join(MESH_DIR, f)
        label = f.replace('.stl', '')
        result = score_mesh(path, label)
        results['scores'].append(result)

    # === Baseline meshes ===
    print("\n  --- Baseline Meshes ---")
    baseline_dir = os.path.join(MESH_DIR, 'baselines')
    baseline_files = sorted([f for f in os.listdir(baseline_dir)
                             if f.endswith('.stl')])
    for f in baseline_files:
        path = os.path.join(baseline_dir, f)
        label = f.replace('.stl', '')
        result = score_mesh(path, label)
        results['scores'].append(result)

    # === Summary ===
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Geometry':40s} {'ECS':>5s} {'h_range':>10s}  Basin fractions")
    print(f"  {'-'*40} {'-'*5} {'-'*10}  {'-'*30}")
    for s in results['scores']:
        fracs = [e['basin_fraction'] for e in s['equilibria'][:5]]
        print(f"  {s['label']:40s} {s['ECS']:5d} {s['com_height_range']:10.6f}  {fracs}")

    # === Validation ===
    print("\n  --- Validation ---")

    sphere = next((s for s in results['scores'] if s['label'] == 'sphere'), None)
    if sphere:
        print(f"  Sphere: ECS={sphere['ECS']}, h_range={sphere['com_height_range']:.6f} "
              f"(h_range~0 = all orientations equivalent -> ECS should be 1)")

    cyl = next((s for s in results['scores'] if s['label'] == 'cylinder_LD2'), None)
    if cyl:
        print(f"  Cylinder (L/D=2): ECS={cyl['ECS']} (expected: 2 — on side + on end)")

    cube = next((s for s in results['scores'] if s['label'] == 'cube'), None)
    if cube:
        print(f"  Cube: ECS={cube['ECS']} (expected: 3 — one per pair of opposite faces)")

    ellipsoid = next((s for s in results['scores'] if s['label'] == 'ellipsoid'), None)
    if ellipsoid:
        print(f"  Ellipsoid: ECS={ellipsoid['ECS']} (expected: 3 — one per axis)")

    gomboc_results = [s for s in results['scores'] if s['label'].startswith('gomboc')]
    all_ecs1 = all(s['ECS'] == 1 for s in gomboc_results)
    print(f"\n  Gömböc ECS=1 across all resolutions: {'YES ✓' if all_ecs1 else 'NO — INVESTIGATE'}")

    if not all_ecs1:
        for s in gomboc_results:
            print(f"    {s['label']}: ECS={s['ECS']}")
            if s['ECS'] <= 5:
                for eq in s['equilibria']:
                    print(f"      basin_frac={eq['basin_fraction']:.4f}  h={eq['height']:.6f}  "
                          f"d={[round(x,3) for x in eq['direction']]}")

    # Save
    result_path = os.path.join(RESULT_DIR, 'ecs_scores.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {result_path}")


if __name__ == '__main__':
    main()
