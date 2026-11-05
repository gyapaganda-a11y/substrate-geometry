"""
Phase 2b: Beta Sweep + Analytical COM Height
=============================================
Note: this is the exploratory test behind the claim of v1 of the paper that no tested Sloan
parameter gives ECS=1. That claim is withdrawn in v2 (the beta values swept here were not
Sloan's, and the oracle's count can err in either direction); the output is retained as a
record of what the oracle returned.

Two tests to determine why ECS != 1 on Gomboc meshes:

Test 1: Beta sweep — generate meshes at beta=0.01, 0.02, 0.05, 0.10, 0.15
         and compute ECS at each. If ECS=1 appears at small beta, we found
         the operating range.

Test 2: Analytical COM height — compute h(d) directly from Sloan equations
         without mesh discretization. If the analytical landscape has 1 minimum,
         the multiple basins are mesh artifacts.
"""

import numpy as np
import trimesh
import json
import os
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

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


# =====================================================================
# TEST 1: BETA SWEEP (mesh-based)
# =====================================================================

def gomboc_surface(n_theta, n_phi, beta, variant):
    theta = np.linspace(1e-6, np.pi - 1e-6, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing='ij')

    if variant == 1:
        phase = 5 * T
    else:
        phase = (3 * np.pi / 2) * (np.cos(T) - np.cos(T)**3 / 3)

    r4 = 1 + 4 * beta * np.sin(T) * np.cos(P - phase)
    if np.any(r4 <= 0):
        raise ValueError(f"r^4 <= 0 at beta={beta}")
    r = r4 ** 0.25

    x = r * np.sin(T) * np.cos(P)
    y = r * np.sin(T) * np.sin(P)
    z = r * np.cos(T)

    vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    north_pole = np.array([[0, 0, 1.0]])
    south_pole = np.array([[0, 0, -1.0]])
    north_idx = len(vertices)
    south_idx = north_idx + 1
    vertices = np.vstack([vertices, north_pole, south_pole])

    faces = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            j_next = (j + 1) % n_phi
            v00 = i * n_phi + j
            v01 = i * n_phi + j_next
            v10 = (i + 1) * n_phi + j
            v11 = (i + 1) * n_phi + j_next
            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])
    for j in range(n_phi):
        j_next = (j + 1) % n_phi
        faces.append([north_idx, j, j_next])
    last_ring = (n_theta - 1) * n_phi
    for j in range(n_phi):
        j_next = (j + 1) % n_phi
        faces.append([south_idx, last_ring + j_next, last_ring + j])

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))


def mesh_ecs(mesh, n_samples=10000, k_neighbors=12, merge_frac=0.02):
    """Compute ECS via drainage basins with height-based merging."""
    directions = fibonacci_sphere(n_samples)
    com = mesh.center_mass
    verts = mesh.vertices

    projections = directions @ verts.T
    support_proj = np.min(projections, axis=1)
    com_proj = directions @ com
    heights = com_proj - support_proj

    # Drainage basins
    tree = cKDTree(directions)
    _, neighbors = tree.query(directions, k=k_neighbors + 1)

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
    basin_sizes = {int(s): int(np.sum(sink_of == s)) for s in unique_sinks}

    # Merge by height
    sink_heights = heights[unique_sinks]
    h_range = np.max(heights) - np.min(heights)
    merge_threshold = max(h_range * merge_frac, 1e-6)

    order = np.argsort(sink_heights)
    sorted_sinks = unique_sinks[order]
    sorted_heights = sink_heights[order]

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

    ecs = len(merged_groups)

    # Basin info
    basins = []
    for group in merged_groups:
        rep = min(group, key=lambda s: heights[s])
        total = sum(basin_sizes[int(s)] for s in group)
        basins.append({
            'height': float(heights[rep]),
            'fraction': round(total / n_samples, 4),
            'direction': directions[rep].tolist(),
        })
    basins.sort(key=lambda b: b['height'])

    return ecs, basins, float(h_range)


def test_beta_sweep():
    """Test 1: Sweep beta values to find where ECS=1 appears."""
    print("\n" + "=" * 70)
    print("  TEST 1: BETA SWEEP")
    print("=" * 70)

    betas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15]
    n_theta, n_phi = 100, 200  # medium resolution

    results = []

    for variant in [1, 2]:
        print(f"\n  --- Gomboc {variant} ---")
        print(f"  {'beta':>8s}  {'ECS':>5s}  {'h_range':>10s}  {'convex':>6s}  Basin fractions")
        print(f"  {'-'*8}  {'-'*5}  {'-'*10}  {'-'*6}  {'-'*30}")

        for beta in betas:
            try:
                mesh = gomboc_surface(n_theta, n_phi, beta, variant)
                mesh.fix_normals()

                # Check convexity
                hull = mesh.convex_hull
                vol_ratio = abs(mesh.volume) / abs(hull.volume)
                is_convex = vol_ratio > 0.999

                ecs, basins, h_range = mesh_ecs(mesh)
                fracs = [b['fraction'] for b in basins[:5]]

                print(f"  {beta:8.3f}  {ecs:5d}  {h_range:10.6f}  {str(is_convex):>6s}  {fracs}")

                results.append({
                    'variant': variant,
                    'beta': beta,
                    'ECS': ecs,
                    'h_range': h_range,
                    'is_convex': bool(is_convex),
                    'volume_ratio': round(vol_ratio, 6),
                    'basins': basins[:5],
                })
            except Exception as e:
                print(f"  {beta:8.3f}  ERROR: {e}")
                results.append({'variant': variant, 'beta': beta, 'error': str(e)})

    return results


# =====================================================================
# TEST 2: ANALYTICAL COM HEIGHT (no mesh)
# =====================================================================

def analytical_com_height(d, beta, variant):
    """Compute COM height for gravity direction d using analytical Gomboc surface.

    For a convex body with surface r=F(theta,phi), the support point in
    direction d is the point on the surface with minimum projection onto d.
    The COM is at the origin (by construction). So COM height = -min(r*d_hat(theta,phi) . d).

    Actually: h(d) = 0 - min over surface of (point . d)
            = -min over (theta,phi) of [F(theta,phi) * (sin(theta)cos(phi)*dx + sin(theta)sin(phi)*dy + cos(theta)*dz)]

    Since COM is at origin, h(d) = -min_surface(point . d).
    """
    dx, dy, dz = d

    def neg_projection(params):
        """Negative of surface point projected onto d. Minimize this = find support point."""
        th, ph = params
        if variant == 1:
            phase = 5 * th
        else:
            phase = (3 * np.pi / 2) * (np.cos(th) - np.cos(th)**3 / 3)

        r4 = 1 + 4 * beta * np.sin(th) * np.cos(ph - phase)
        if r4 <= 0:
            return 1e10  # invalid
        r = r4 ** 0.25

        x = r * np.sin(th) * np.cos(ph)
        y = r * np.sin(th) * np.sin(ph)
        z = r * np.cos(th)

        return x * dx + y * dy + z * dz  # projection onto d

    # Multi-start optimization to find global minimum of projection
    best_val = np.inf
    n_starts = 20
    for _ in range(n_starts):
        th0 = np.random.uniform(0.1, np.pi - 0.1)
        ph0 = np.random.uniform(0, 2 * np.pi)
        result = minimize(neg_projection, [th0, ph0],
                         method='Nelder-Mead',
                         options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000})
        if result.fun < best_val:
            best_val = result.fun

    # COM height = 0 (COM at origin) - support_projection = -best_val
    # Actually h = COM.d - support.d = 0 - best_val = -best_val
    # Wait: best_val is the minimum projection, which is negative (support is below COM).
    # h = 0 - best_val. Since best_val is the most negative projection, h = -best_val > 0.
    return -best_val


def test_analytical():
    """Test 2: Compute COM height landscape analytically, find minima."""
    print("\n" + "=" * 70)
    print("  TEST 2: ANALYTICAL COM HEIGHT (no mesh)")
    print("=" * 70)

    results = []

    for variant in [1, 2]:
        beta = 0.15 if variant == 1 else 0.17

        print(f"\n  --- Gomboc {variant}, beta={beta} ---")

        # Sample directions on S2
        n_dirs = 2000
        directions = fibonacci_sphere(n_dirs)

        print(f"  Computing analytical h(d) for {n_dirs} directions...")
        t0 = time.time()

        heights = np.zeros(n_dirs)
        for i, d in enumerate(directions):
            heights[i] = analytical_com_height(d, beta, variant)
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{n_dirs} done...")

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        h_min = float(np.min(heights))
        h_max = float(np.max(heights))
        h_range = h_max - h_min

        # Find the minimum direction
        min_idx = np.argmin(heights)
        min_dir = directions[min_idx]

        # Find the maximum direction
        max_idx = np.argmax(heights)
        max_dir = directions[max_idx]

        print(f"  h_min = {h_min:.8f} at d = {[round(x,4) for x in min_dir]}")
        print(f"  h_max = {h_max:.8f} at d = {[round(x,4) for x in max_dir]}")
        print(f"  h_range = {h_range:.8f}")

        # Count basins via drainage on analytical heights
        tree = cKDTree(directions)
        _, neighbors = tree.query(directions, k=13)

        sink_of = np.full(n_dirs, -1, dtype=int)
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

        for i in range(n_dirs):
            if sink_of[i] == -1:
                find_sink(i)

        unique_sinks = np.unique(sink_of)
        raw_basins = len(unique_sinks)

        # Merge by height (same approach as mesh oracle)
        sink_heights = heights[unique_sinks]
        merge_threshold = max(h_range * 0.02, 1e-8)

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

        print(f"  Raw basins (before merge): {raw_basins}")
        print(f"  ECS (after merge at {merge_threshold:.6f}): {ecs}")

        # Print basin details
        for gi, group in enumerate(merged):
            sizes = [int(np.sum(sink_of == s)) for s in group]
            total = sum(sizes)
            rep = min(group, key=lambda s: heights[s])
            print(f"    Basin {gi}: h={heights[rep]:.8f}, "
                  f"fraction={total/n_dirs:.4f}, "
                  f"d={[round(x,3) for x in directions[rep]]}")

        results.append({
            'variant': variant,
            'beta': beta,
            'n_directions': n_dirs,
            'h_min': h_min,
            'h_max': h_max,
            'h_range': h_range,
            'raw_basins': raw_basins,
            'ECS_merged': ecs,
            'merge_threshold': merge_threshold,
            'elapsed': round(elapsed, 1),
        })

    return results


def main():
    all_results = {}

    # Test 1: Beta sweep
    all_results['beta_sweep'] = test_beta_sweep()

    # Test 2: Analytical (takes longer due to optimization at each direction)
    all_results['analytical'] = test_analytical()

    # Save
    result_path = os.path.join(RESULT_DIR, 'phase2b_investigation.json')
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {result_path}")


if __name__ == '__main__':
    main()
