"""
Phase 5c: Additional Checks
=============================
Three additional checks:

1. Ellipsoid and Capsule through IMU precision comparison
   (both have ECS=1 — does the Gomboc's gentle self-righting matter?)

2. Generate 2-3 additional verified mono-monostatic instances
   using Phase 2c optimization with different parameterizations

3. Merge threshold sensitivity test (robustness check)
   Test ECS across threshold values 0.5%, 1%, 2%, 5%
"""

import numpy as np
import trimesh
import json
import os
import time
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution

MESH_DIR = os.path.join(os.path.dirname(__file__), '..', 'meshes')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

BETA_VERIFIED = 0.023149
A1_VERIFIED = 0.234433


def eta(theta):
    return (3 * np.pi / 2) * (np.cos(theta) - np.cos(theta)**3 / 3)

def fibonacci_sphere(n):
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    return np.stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)], axis=1)


def com_heights(mesh, directions):
    verts = mesh.vertices
    com = mesh.center_mass
    proj = directions @ verts.T
    return directions @ com - np.min(proj, axis=1)


def compute_ecs_at_threshold(mesh, n_samples=5000, merge_frac=0.01):
    """Compute ECS with specified merge threshold."""
    directions = fibonacci_sphere(n_samples)
    heights = com_heights(mesh, directions)

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
    merge_t = max(h_range * merge_frac, 1e-8)

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

    # Also get raw basin count (before merge)
    raw_basins = len(unique_sinks)

    return ecs, boa, h_range, raw_basins


def build_gomboc(beta, coeffs, n_theta=80, n_phi=160):
    """Build Gomboc mesh from extended parameters."""
    theta = np.linspace(1e-6, np.pi-1e-6, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing='ij')

    e = eta(T)
    phase = e.copy()
    for k, a in enumerate(coeffs):
        phase = phase + a * np.sin((k+1) * e)

    r4 = 1 + 4*beta*np.sin(T)*np.cos(P - phase)
    if np.any(r4 <= 0):
        return None
    r = r4**0.25
    x,y,z = r*np.sin(T)*np.cos(P), r*np.sin(T)*np.sin(P), r*np.cos(T)

    verts = np.stack([x.ravel(),y.ravel(),z.ravel()],axis=1)
    verts = np.vstack([verts, [[0,0,1]], [[0,0,-1]]])
    ni, si = len(verts)-2, len(verts)-1
    faces = []
    for i in range(n_theta-1):
        for j in range(n_phi):
            jn=(j+1)%n_phi
            faces+=[[i*n_phi+j,(i+1)*n_phi+j,i*n_phi+jn],[i*n_phi+jn,(i+1)*n_phi+j,(i+1)*n_phi+jn]]
    for j in range(n_phi):
        jn=(j+1)%n_phi; faces.append([ni,j,jn])
    last=(n_theta-1)*n_phi
    for j in range(n_phi):
        jn=(j+1)%n_phi; faces.append([si,last+jn,last+j])
    m = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    m.fix_normals()
    return m


def com_constraint_violation(beta, coeffs, n_quad=500):
    theta = np.linspace(1e-6, np.pi-1e-6, n_quad)
    e = eta(theta)
    P = e.copy()
    for k, a in enumerate(coeffs):
        P = P + a * np.sin((k+1) * e)
    integrand = np.sin(theta)**3 * np.exp(1j * P)
    return float(abs(np.sum(integrand) * (theta[1]-theta[0])))


# =====================================================================
# ADDITION 1: IMU precision comparison — ECS=1 bodies
# =====================================================================

def addition1_imu_comparison():
    """Compare all ECS=1 bodies on self-righting gentleness for IMU application."""
    print("\n" + "=" * 70)
    print("  ADDITION 1: IMU PRECISION — ECS=1 BODY COMPARISON")
    print("  Which ECS=1 body self-rights most gently?")
    print("=" * 70)

    meshes = [
        ('gomboc_verified_ecs1_medres.stl', 'Gomboc'),
        ('baselines/ellipsoid.stl', 'Ellipsoid'),
        ('baselines/capsule.stl', 'Capsule'),
    ]

    results = []
    for filename, label in meshes:
        path = os.path.join(MESH_DIR, filename)
        mesh = trimesh.load(path)
        directions = fibonacci_sphere(5000)
        heights = com_heights(mesh, directions)

        min_idx = np.argmin(heights)
        h_min = heights[min_idx]
        h_max = np.max(heights)
        h_range = h_max - h_min

        # SRE: mean height drop
        sre = float(np.mean(heights - h_min))

        # Steepness: mean gradient magnitude
        height_drops = heights - h_min
        cos_angles = np.clip(directions @ directions[min_idx], -1, 1)
        ang_dists = np.arccos(cos_angles)
        mask = ang_dists > 0.01
        steepness = float(np.mean(height_drops[mask] / ang_dists[mask]))

        # For IMU: lower SRE and lower steepness = gentler self-righting
        # = less mechanical shock to sensor during orientation
        entry = {
            'geometry': label,
            'h_range': round(h_range, 6),
            'SRE_mean': round(sre, 6),
            'steepness': round(steepness, 6),
        }
        results.append(entry)

        print(f"  {label:15s}  h_range={h_range:.6f}  SRE={sre:.6f}  steepness={steepness:.6f}")

    # Ranking
    print("\n  --- GENTLENESS RANKING (lower SRE = gentler) ---")
    ranked = sorted(results, key=lambda r: r['SRE_mean'])
    for i, r in enumerate(ranked):
        ratio_vs_worst = ranked[-1]['SRE_mean'] / max(r['SRE_mean'], 1e-10)
        print(f"  {i+1}. {r['geometry']:15s}  SRE={r['SRE_mean']:.6f}  ({ratio_vs_worst:.1f}x gentler than {ranked[-1]['geometry']})")

    return results


# =====================================================================
# ADDITION 2: Generate additional verified mono-monostatic instances
# =====================================================================

def addition2_additional_instances():
    """Generate 2 more verified ECS=1 bodies with different parameterizations."""
    print("\n" + "=" * 70)
    print("  ADDITION 2: ADDITIONAL MONO-MONOSTATIC INSTANCES")
    print("  Can we generate more verified ECS=1 bodies?")
    print("=" * 70)

    # Strategy: use the same extended Sloan ansatz but with different
    # Fourier term indices. The verified instance uses sin(1*eta).
    # Try sin(2*eta) and sin(3*eta) as the perturbation term.

    results = []

    for fourier_idx, seed in [(2, 123), (3, 456)]:
        print(f"\n  --- Trying P(theta) = eta + a*sin({fourier_idx}*eta) ---")

        def objective(params):
            beta, a = params
            if beta <= 0 or beta > 0.15:
                return 100.0
            coeffs = [0.0] * (fourier_idx - 1) + [a]  # place coefficient at right index

            cv = com_constraint_violation(beta, coeffs)
            if cv > 0.1:
                return 50.0 + cv * 10

            mesh = build_gomboc(beta, coeffs, n_theta=60, n_phi=120)
            if mesh is None:
                return 100.0

            try:
                hull = mesh.convex_hull
                vr = abs(mesh.volume) / abs(hull.volume)
            except:
                vr = 0
            if vr < 0.995:
                return 30.0

            ecs, boa, h_range, _ = compute_ecs_at_threshold(mesh, n_samples=2000, merge_frac=0.01)
            if ecs <= 1:
                return 0.0  # perfect
            # Compute basin gap
            directions = fibonacci_sphere(2000)
            heights = com_heights(mesh, directions)
            tree = cKDTree(directions)
            _, neighbors = tree.query(directions, k=13)
            sink_of = np.full(2000, -1, dtype=int)
            def find_sink(i):
                visited = []; current = i
                while sink_of[current] == -1:
                    visited.append(current)
                    nbr = neighbors[current]
                    lowest = nbr[np.argmin(heights[nbr])]
                    if lowest == current:
                        for v in visited: sink_of[v] = current
                        return current
                    current = lowest
                for v in visited: sink_of[v] = sink_of[current]
                return sink_of[current]
            for i in range(2000):
                if sink_of[i] == -1: find_sink(i)
            sinks = np.unique(sink_of)
            sink_h = sorted(heights[sinks])
            gap = sink_h[1] - sink_h[0] if len(sink_h) > 1 else 0
            return gap + cv * 5 + max(0, ecs - 1) * 0.01

        t0 = time.time()
        result = differential_evolution(
            objective,
            bounds=[(0.005, 0.12), (-0.8, 0.8)],
            seed=seed,
            maxiter=40,
            popsize=15,
            tol=1e-6,
        )
        elapsed = time.time() - t0

        beta_opt, a_opt = result.x
        coeffs = [0.0] * (fourier_idx - 1) + [a_opt]

        mesh = build_gomboc(beta_opt, coeffs)
        if mesh is not None:
            ecs, boa, h_range, _ = compute_ecs_at_threshold(mesh, n_samples=5000, merge_frac=0.01)
            cv = com_constraint_violation(beta_opt, coeffs)

            print(f"  beta={beta_opt:.6f}, a{fourier_idx}={a_opt:.6f}")
            print(f"  ECS={ecs}, BOA={boa:.3f}, h_range={h_range:.6f}, COM_viol={cv:.8f}")
            print(f"  Time: {elapsed:.0f}s")

            entry = {
                'fourier_index': fourier_idx,
                'beta': round(float(beta_opt), 6),
                'coefficient': round(float(a_opt), 6),
                'ECS': ecs,
                'BOA': round(boa, 4),
                'h_range': round(h_range, 6),
                'com_violation': round(float(cv), 8),
                'score': round(float(result.fun), 6),
            }

            if ecs == 1:
                print(f"  *** ECS=1 ACHIEVED — new verified instance! ***")
                fname = f'gomboc_instance_sin{fourier_idx}eta.stl'
                mesh_hires = build_gomboc(beta_opt, coeffs, n_theta=150, n_phi=300)
                if mesh_hires:
                    mesh_hires.export(os.path.join(MESH_DIR, fname))
                    entry['mesh_file'] = fname
                    print(f"  Saved to {fname}")

            results.append(entry)

    return results


# =====================================================================
# ADDITION 3: Merge threshold sensitivity
# =====================================================================

def addition3_threshold_sensitivity():
    """Test ECS across merge threshold values."""
    print("\n" + "=" * 70)
    print("  ADDITION 3: MERGE THRESHOLD SENSITIVITY")
    print("  Is ECS=1 an artifact of the 1% threshold?")
    print("=" * 70)

    mesh = trimesh.load(os.path.join(MESH_DIR, 'gomboc_verified_ecs1_medres.stl'))

    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    results = []

    print(f"\n  {'Threshold':>10s}  {'ECS':>4s}  {'BOA':>6s}  {'Raw basins':>10s}")
    print(f"  {'-'*10}  {'-'*4}  {'-'*6}  {'-'*10}")

    for t in thresholds:
        ecs, boa, h_range, raw = compute_ecs_at_threshold(mesh, n_samples=5000, merge_frac=t)
        entry = {
            'threshold_frac': t,
            'threshold_pct': f'{t*100:.1f}%',
            'ECS': ecs,
            'BOA': round(boa, 4),
            'raw_basins': raw,
        }
        results.append(entry)
        print(f"  {t*100:9.1f}%  {ecs:4d}  {boa:6.3f}  {raw:10d}")

    # Is ECS=1 robust?
    ecs_values = [r['ECS'] for r in results]
    if all(e == 1 for e in ecs_values):
        print(f"\n  ECS=1 is ROBUST across all thresholds (0.1% to 10%)")
    else:
        ecs1_range = [r['threshold_pct'] for r in results if r['ECS'] == 1]
        print(f"\n  ECS=1 holds at thresholds: {ecs1_range}")
        ecs_other = [(r['threshold_pct'], r['ECS']) for r in results if r['ECS'] != 1]
        print(f"  ECS != 1 at: {ecs_other}")

    return results


def main():
    all_results = {}

    all_results['imu_comparison'] = addition1_imu_comparison()
    all_results['additional_instances'] = addition2_additional_instances()
    all_results['threshold_sensitivity'] = addition3_threshold_sensitivity()

    result_path = os.path.join(RESULT_DIR, 'midaudit_additions.json')
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  All results saved to {result_path}")


if __name__ == '__main__':
    main()
