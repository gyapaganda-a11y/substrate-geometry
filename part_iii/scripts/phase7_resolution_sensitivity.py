"""
Phase 7: Resolution Sensitivity
=================================
Verify ECS=1 holds across mesh resolutions for the verified Gomboc.
Resolution dependence was a confound in Part II, so this check is included.

Generate the verified Gomboc at 5 resolutions and compute ECS at each.
Also check: h_range, BOA, COM offset, convexity ratio.
"""

import numpy as np
import trimesh
import json
import os
import time
from scipy.spatial import cKDTree

MESH_DIR = os.path.join(os.path.dirname(__file__), '..', 'meshes')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

BETA = 0.023149
A1 = 0.234433

def eta(theta):
    return (3 * np.pi / 2) * (np.cos(theta) - np.cos(theta)**3 / 3)


def generate_verified_gomboc(n_theta, n_phi):
    """Generate verified ECS=1 Gomboc at given resolution."""
    theta = np.linspace(1e-6, np.pi - 1e-6, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    e = eta(T)
    phase = e + A1 * np.sin(e)
    r = (1 + 4 * BETA * np.sin(T) * np.cos(P - phase)) ** 0.25
    x, y, z = r*np.sin(T)*np.cos(P), r*np.sin(T)*np.sin(P), r*np.cos(T)

    verts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    verts = np.vstack([verts, [[0,0,1]], [[0,0,-1]]])
    ni, si = len(verts)-2, len(verts)-1

    faces = []
    for i in range(n_theta-1):
        for j in range(n_phi):
            jn = (j+1) % n_phi
            faces += [[i*n_phi+j, (i+1)*n_phi+j, i*n_phi+jn],
                       [i*n_phi+jn, (i+1)*n_phi+j, (i+1)*n_phi+jn]]
    for j in range(n_phi):
        jn = (j+1) % n_phi
        faces.append([ni, j, jn])
    last = (n_theta-1)*n_phi
    for j in range(n_phi):
        jn = (j+1) % n_phi
        faces.append([si, last+jn, last+j])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    mesh.fix_normals()
    return mesh


def fibonacci_sphere(n):
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    return np.stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)], axis=1)


def compute_ecs(mesh, n_samples=5000):
    """Compute ECS with drainage basins + height merge."""
    directions = fibonacci_sphere(n_samples)
    com = mesh.center_mass
    verts = mesh.vertices

    projections = directions @ verts.T
    support_proj = np.min(projections, axis=1)
    heights = directions @ com - support_proj

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
    merge_t = max(h_range * 0.01, 1e-7)

    order = np.argsort(sink_heights)
    ss, sh = unique_sinks[order], sink_heights[order]
    mg = []; cg = [ss[0]]; ch = sh[0]
    for i in range(1, len(ss)):
        if sh[i] - ch <= merge_t:
            cg.append(ss[i])
        else:
            mg.append(cg); cg = [ss[i]]; ch = sh[i]
    mg.append(cg)

    ecs = len(mg)
    dom = max(sum(int(np.sum(sink_of == s)) for s in g) for g in mg)
    boa = dom / n_samples

    return ecs, h_range, boa


def main():
    print("\n" + "=" * 70)
    print("  PHASE 7: RESOLUTION SENSITIVITY")
    print("  Does ECS=1 hold across mesh resolutions?")
    print("=" * 70)

    resolutions = [
        (20, 40, "Very low"),
        (40, 80, "Low"),
        (60, 120, "Medium-low"),
        (80, 160, "Medium"),
        (100, 200, "Medium-high"),
        (150, 300, "High"),
        (200, 400, "Very high"),
    ]

    results = {
        'phase': 'Phase 7: Resolution Sensitivity',
        'parameters': {'beta': BETA, 'a1': A1},
        'sweeps': [],
    }

    print(f"\n  {'Resolution':>12s}  {'Faces':>8s}  {'ECS':>4s}  {'h_range':>10s}  "
          f"{'BOA':>6s}  {'COM offset':>10s}  {'Convex ratio':>12s}  {'Time':>6s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*4}  {'-'*10}  "
          f"{'-'*6}  {'-'*10}  {'-'*12}  {'-'*6}")

    for nt, np_, label in resolutions:
        t0 = time.time()
        mesh = generate_verified_gomboc(nt, np_)

        com_offset = float(np.linalg.norm(mesh.center_mass))
        try:
            hull = mesh.convex_hull
            vol_ratio = abs(mesh.volume) / abs(hull.volume)
        except:
            vol_ratio = 0

        ecs, h_range, boa = compute_ecs(mesh)
        elapsed = time.time() - t0

        entry = {
            'label': label,
            'resolution': f'{nt}x{np_}',
            'n_faces': len(mesh.faces),
            'ECS': ecs,
            'h_range': round(h_range, 8),
            'BOA': round(boa, 4),
            'com_offset': round(com_offset, 8),
            'convexity_ratio': round(float(vol_ratio), 6),
            'elapsed': round(elapsed, 1),
        }
        results['sweeps'].append(entry)

        ecs_mark = "OK" if ecs == 1 else f"FAIL (ECS={ecs})"
        print(f"  {nt:3d}x{np_:3d}  {label:>5s}  {len(mesh.faces):8d}  {ecs:4d}  {h_range:10.6f}  "
              f"{boa:6.3f}  {com_offset:10.8f}  {vol_ratio:12.6f}  {elapsed:5.1f}s  {ecs_mark}")

    # Summary
    print("\n  --- SUMMARY ---")
    all_ecs1 = all(s['ECS'] == 1 for s in results['sweeps'])
    print(f"  ECS=1 across all resolutions: {'YES' if all_ecs1 else 'NO'}")

    if all_ecs1:
        h_ranges = [s['h_range'] for s in results['sweeps']]
        print(f"  h_range convergence: {min(h_ranges):.6f} to {max(h_ranges):.6f}")
        print(f"  h_range variation: {(max(h_ranges)-min(h_ranges))/np.mean(h_ranges)*100:.2f}%")
        print(f"  ECS=1 is RESOLUTION-INDEPENDENT for this geometry.")
    else:
        for s in results['sweeps']:
            if s['ECS'] != 1:
                print(f"    FAILED at {s['resolution']}: ECS={s['ECS']}")

    # COM offset convergence
    offsets = [s['com_offset'] for s in results['sweeps']]
    print(f"  COM offset range: {min(offsets):.8f} to {max(offsets):.8f}")
    print(f"  COM offset converges to zero: {'YES' if offsets[-1] < 0.005 else 'NO'}")

    # Save
    result_path = os.path.join(RESULT_DIR, 'resolution_sensitivity.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {result_path}")


if __name__ == '__main__':
    main()
