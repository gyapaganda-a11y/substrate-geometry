"""
Phase 3 v2: Self-Righting Dynamics — Direct Computation
=========================================================
Instead of simulating gradient descent (slow, convergence issues),
compute metrics directly from the COM height landscape:

SRT  — Approximate angular distance from each starting direction to the
       global minimum, measured as great-circle distance on S².
SRE  — Height drop: h(d_start) - h(d_min) for each starting direction.
BOA  — Basin of attraction: fraction of S² draining to dominant minimum
       (already computed by ECS oracle — reuse those results).

This avoids the simulation entirely and produces exact metrics.
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
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def compute_metrics(mesh, label, n_samples=5000):
    """Compute self-righting metrics from COM height landscape."""
    t0 = time.time()

    directions = fibonacci_sphere(n_samples)
    com = mesh.center_mass
    verts = mesh.vertices

    # COM height at each direction
    projections = directions @ verts.T
    support_proj = np.min(projections, axis=1)
    com_proj = directions @ com
    heights = com_proj - support_proj

    # Find global minimum (the stable equilibrium for ECS=1 bodies)
    min_idx = np.argmin(heights)
    d_min = directions[min_idx]
    h_min = heights[min_idx]

    # Find global maximum (the unstable equilibrium)
    max_idx = np.argmax(heights)
    d_max = directions[max_idx]
    h_max = heights[max_idx]

    # Angular distance from each direction to the global minimum
    cos_angles = np.clip(directions @ d_min, -1, 1)
    angular_distances_rad = np.arccos(cos_angles)
    angular_distances_deg = np.degrees(angular_distances_rad)

    # Height drops from each direction to the minimum
    height_drops = heights - h_min

    # SRT: mean/max angular distance to the stable equilibrium
    srt_mean = float(np.mean(angular_distances_deg))
    srt_max = float(np.max(angular_distances_deg))
    srt_std = float(np.std(angular_distances_deg))

    # For a uniform distribution on the sphere, mean angular distance
    # to any point is 90 degrees. Values significantly less than 90
    # mean the minimum is "accessible" from most orientations.

    # SRE: mean/max height drop (proportional to energy dissipated)
    sre_mean = float(np.mean(height_drops))
    sre_max = float(np.max(height_drops))

    # h_range: total variation in COM height
    h_range = float(h_max - h_min)

    # Steepness: average gradient magnitude (how quickly it self-rights)
    # Computed as height drop per unit angular distance
    with np.errstate(divide='ignore', invalid='ignore'):
        steepness = np.where(angular_distances_rad > 0.01,
                             height_drops / angular_distances_rad, 0)
    steepness_mean = float(np.mean(steepness[angular_distances_rad > 0.01]))

    # Basin analysis via drainage
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
    h_range_total = np.max(heights) - np.min(heights)
    merge_threshold = max(h_range_total * 0.01, 1e-7)

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
    basin_fractions = []
    for group in merged:
        total = sum(int(np.sum(sink_of == s)) for s in group)
        basin_fractions.append(round(total / n_samples, 4))

    boa = max(basin_fractions)

    # Angle between stable and unstable equilibrium
    stable_unstable_angle = float(np.degrees(np.arccos(np.clip(d_min @ d_max, -1, 1))))

    elapsed = time.time() - t0

    result = {
        'label': label,
        'n_samples': n_samples,
        'ECS': ecs,
        'SRT_mean_deg': round(srt_mean, 2),
        'SRT_max_deg': round(srt_max, 2),
        'SRT_std_deg': round(srt_std, 2),
        'SRE_mean': round(sre_mean, 6),
        'SRE_max': round(sre_max, 6),
        'h_range': round(h_range, 6),
        'steepness_mean': round(steepness_mean, 6),
        'BOA_fraction': round(boa, 4),
        'basin_fractions': basin_fractions[:5],
        'stable_direction': d_min.tolist(),
        'unstable_direction': d_max.tolist(),
        'stable_unstable_angle_deg': round(stable_unstable_angle, 2),
        'stable_height': round(float(h_min), 6),
        'unstable_height': round(float(h_max), 6),
        'elapsed_seconds': round(elapsed, 1),
    }

    return result


def main():
    print("\n" + "=" * 80)
    print("  PHASE 3 v2: SELF-RIGHTING DYNAMICS (direct computation)")
    print("=" * 80)

    results = {'phase': 'Phase 3: Dynamics v2', 'scores': []}

    meshes = [
        ('gomboc_verified_ecs1_medres.stl', 'Gomboc (verified ECS=1)'),
        ('baselines/sphere.stl', 'Sphere'),
        ('baselines/cylinder_LD2.stl', 'Cylinder (L/D=2)'),
        ('baselines/hemisphere.stl', 'Hemisphere'),
        ('baselines/ellipsoid.stl', 'Ellipsoid'),
        ('baselines/cube.stl', 'Cube'),
        ('baselines/capsule.stl', 'Capsule'),
    ]

    all_results = []
    for filename, label in meshes:
        path = os.path.join(MESH_DIR, filename)
        if not os.path.exists(path):
            print(f"  SKIP: {path}")
            continue
        mesh = trimesh.load(path)
        r = compute_metrics(mesh, label)
        all_results.append(r)
        results['scores'].append(r)

    # Print comparison table
    print("\n" + "=" * 80)
    print("  COMPARISON TABLE")
    print("=" * 80)
    print(f"  {'Geometry':30s}  {'ECS':>3s}  {'SRT_mean':>8s}  {'SRT_max':>8s}  "
          f"{'SRE_mean':>8s}  {'h_range':>8s}  {'steep':>7s}  {'BOA':>5s}  {'S-U angle':>9s}")
    print(f"  {'-'*30}  {'-'*3}  {'-'*8}  {'-'*8}  "
          f"{'-'*8}  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*9}")

    for r in all_results:
        print(f"  {r['label']:30s}  {r['ECS']:3d}  "
              f"{r['SRT_mean_deg']:7.1f}d  {r['SRT_max_deg']:7.1f}d  "
              f"{r['SRE_mean']:8.5f}  {r['h_range']:8.5f}  "
              f"{r['steepness_mean']:7.5f}  {r['BOA_fraction']:.3f}  "
              f"{r['stable_unstable_angle_deg']:8.1f}d")

    # Key comparisons
    print("\n  --- Key Comparisons ---")
    gomboc = next((r for r in all_results if 'verified' in r['label'].lower() or 'ECS=1' in r['label']), None)
    cylinder = next((r for r in all_results if 'Cylinder' in r['label']), None)
    capsule = next((r for r in all_results if 'Capsule' in r['label']), None)

    if gomboc and cylinder:
        sre_ratio = cylinder['SRE_mean'] / max(gomboc['SRE_mean'], 1e-10) if gomboc['SRE_mean'] > 0 else float('inf')
        print(f"  SRE cylinder/gomboc: {sre_ratio:.1f}x (cylinder dissipates {sre_ratio:.1f}x more energy)")
        print(f"  Gomboc ECS={gomboc['ECS']}, BOA={gomboc['BOA_fraction']:.3f} vs Cylinder ECS={cylinder['ECS']}, BOA={cylinder['BOA_fraction']:.3f}")

    # Save
    result_path = os.path.join(RESULT_DIR, 'dynamics.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {result_path}")


if __name__ == '__main__':
    main()
