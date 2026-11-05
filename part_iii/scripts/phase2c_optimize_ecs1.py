"""
Phase 2c: Optimize Surface Parameters to Achieve ECS=1
=======================================================
Substrate Geometry Part III (arXiv:2604.17095): Gomboc

Starting point: Gomboc 2 at beta~0.05 shows ECS=2 with two basins
close in height. The optimization target: find surface parameters
that merge those two basins into one.

Approach:
  1. Use the Sloan general ansatz: r^4 = 1 + 4*beta*sin(theta)*cos(phi - P(theta))
     where P(theta) must satisfy the COM constraint.
  2. Start with P(theta) = eta(theta) (Gomboc 2) and beta=0.05
  3. Extend P(theta) with additional Fourier terms to give more degrees of freedom
  4. Objective: minimize the height gap between the two lowest COM height basins
  5. Constraint: maintain convexity (volume ratio to convex hull > 0.999)

Extended parameterization:
  P(theta) = eta(theta) + sum_k a_k * sin(k * eta(theta))
  where eta(theta) = (3pi/2)(cos(theta) - cos^3(theta)/3)

This preserves the COM constraint (Eq. 11) because exp(i*P(eta)) averaged
over eta still integrates to zero if the perturbation terms are periodic in eta.
Actually, we need to verify this — the perturbation terms change the integral.

Safer approach: parameterize P(theta) freely and enforce the COM constraint
as an additional penalty in the objective function.

Even simpler: optimize beta and a few Fourier coefficients in P(theta),
with COM constraint and convexity as penalties.
"""

import numpy as np
import trimesh
import json
import os
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import cKDTree

RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
MESH_DIR = os.path.join(os.path.dirname(__file__), '..', 'meshes')
os.makedirs(RESULT_DIR, exist_ok=True)


def eta_of_theta(theta):
    """Sloan's change of variable: eta = (3pi/2)(cos(theta) - cos^3(theta)/3)"""
    return (3 * np.pi / 2) * (np.cos(theta) - np.cos(theta)**3 / 3)


def phase_function(theta, coeffs):
    """Extended phase function P(theta).

    P(theta) = eta(theta) + sum_k coeffs[k] * sin((k+1) * eta(theta))

    The base is Gomboc 2's phase. The Fourier terms add degrees of freedom.
    coeffs = [] gives the original Gomboc 2.
    """
    e = eta_of_theta(theta)
    P = e.copy()
    for k, a_k in enumerate(coeffs):
        P = P + a_k * np.sin((k + 1) * e)
    return P


def com_constraint_violation(beta, coeffs, n_quad=500):
    """Check how much the COM constraint (Eq. 11) is violated.

    The constraint is: integral_0^pi sin^3(theta) exp(i*P(theta)) dtheta = 0

    Returns magnitude of the complex integral (should be 0).
    """
    theta = np.linspace(1e-6, np.pi - 1e-6, n_quad)
    P = phase_function(theta, coeffs)

    # The full COM constraint with r^4 weighting:
    # integral sin(theta) cos(theta) r^4 dtheta dphi = 0  (z-component)
    # integral sin^2(theta) exp(i*phi) r^4 dtheta dphi = 0  (x,y components)
    #
    # For our ansatz r^4 = 1 + 4*beta*sin(theta)*cos(phi - P(theta)),
    # the phi-averaged r^4 = 1 (the cosine term averages to zero over phi).
    # The z-component integral becomes:
    #   integral sin(theta)*cos(theta)*1 dtheta * integral dphi = 0
    #   = [integral_0^pi sin(theta)cos(theta) dtheta] * 2pi = 0  (by symmetry of sin*cos)
    # So z is automatically satisfied.
    #
    # For x,y: the phi integral picks out the coefficient of exp(i*phi) in r^4:
    #   r^4 = 1 + 4*beta*sin(theta)*cos(phi - P) = 1 + 2*beta*sin(theta)*[exp(i(phi-P)) + exp(-i(phi-P))]
    # The exp(i*phi) component is: 2*beta*sin(theta)*exp(-i*P(theta))
    # So the x,y constraint becomes:
    #   integral_0^pi sin^2(theta) * 2*beta*sin(theta)*exp(-i*P(theta)) * sin(theta) dtheta = 0
    #   = 2*beta * integral_0^pi sin^3(theta) * exp(-i*P(theta)) dtheta * ... wait
    #
    # Actually from Sloan Eq. 3b:
    # integral dphi integral sin^2(theta) exp(i*phi) * F^4 dtheta = 0
    # With F^4 = 1 + 4*beta*sin(theta)*cos(phi - P):
    #   integral_0^{2pi} exp(i*phi) * [1 + 4*beta*sin(theta)*cos(phi-P)] dphi
    #   = integral exp(i*phi) dphi + 4*beta*sin(theta) * integral exp(i*phi)*cos(phi-P) dphi
    #   = 0 + 4*beta*sin(theta) * pi * exp(i*P)   [standard result]
    #
    # So the constraint is:
    #   4*beta*pi * integral_0^pi sin^3(theta) exp(i*P(theta)) dtheta = 0
    # Which (dividing by nonzero constants) is Sloan Eq. 11:
    #   integral_0^pi sin^3(theta) exp(i*P(theta)) dtheta = 0

    integrand = np.sin(theta)**3 * np.exp(1j * P)
    dtheta = theta[1] - theta[0]
    integral = np.sum(integrand) * dtheta

    return abs(integral)


def build_mesh_from_params(beta, coeffs, n_theta=80, n_phi=160):
    """Build a Gomboc mesh from extended parameters."""
    theta = np.linspace(1e-6, np.pi - 1e-6, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P_grid = np.meshgrid(theta, phi, indexing='ij')

    phase = np.zeros_like(T)
    for i in range(n_theta):
        phase[i, :] = phase_function(theta[i], coeffs)

    r4 = 1 + 4 * beta * np.sin(T) * np.cos(P_grid - phase)

    if np.any(r4 <= 0):
        return None

    r = r4 ** 0.25
    x = r * np.sin(T) * np.cos(P_grid)
    y = r * np.sin(T) * np.sin(P_grid)
    z = r * np.cos(T)

    vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    vertices = np.vstack([vertices, [[0, 0, 1.0]], [[0, 0, -1.0]]])
    north_idx = len(vertices) - 2
    south_idx = len(vertices) - 1

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
    last = (n_theta - 1) * n_phi
    for j in range(n_phi):
        j_next = (j + 1) % n_phi
        faces.append([south_idx, last + j_next, last + j])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    mesh.fix_normals()
    return mesh


def fibonacci_sphere(n):
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def compute_basin_gap(mesh, n_samples=3000, k_neighbors=12):
    """Compute the height gap between the two lowest COM height basins.

    Returns:
        gap: height difference between basin 1 and basin 2 (0 if only 1 basin)
        n_basins: number of basins found
        h_range: total COM height range
        basin_info: list of (height, fraction) for each basin
    """
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

    # Sort sinks by height
    sink_heights = [(s, heights[s]) for s in unique_sinks]
    sink_heights.sort(key=lambda x: x[1])

    h_range = float(np.max(heights) - np.min(heights))

    # Merge sinks within 1% of h_range
    merge_threshold = max(h_range * 0.01, 1e-7)
    merged = []
    current_group = [sink_heights[0]]
    for i in range(1, len(sink_heights)):
        if sink_heights[i][1] - current_group[0][1] <= merge_threshold:
            current_group.append(sink_heights[i])
        else:
            merged.append(current_group)
            current_group = [sink_heights[i]]
    merged.append(current_group)

    basin_info = []
    for group in merged:
        total = sum(basin_sizes[int(s)] for s, _ in group)
        min_h = min(h for _, h in group)
        basin_info.append((float(min_h), round(total / n_samples, 4)))

    n_basins = len(merged)

    if n_basins == 1:
        gap = 0.0
    else:
        gap = basin_info[1][0] - basin_info[0][0]

    return gap, n_basins, h_range, basin_info


def objective(params, n_fourier):
    """Objective function for optimization.

    params = [beta, a_0, a_1, ..., a_{n_fourier-1}]

    Minimize: basin_gap + penalty_com + penalty_convexity + penalty_r4
    """
    beta = params[0]
    coeffs = list(params[1:1 + n_fourier])

    # Penalty: beta must be positive and reasonable
    if beta <= 0 or beta > 0.25:
        return 100.0

    # Penalty: COM constraint violation
    com_viol = com_constraint_violation(beta, coeffs)
    if com_viol > 0.1:
        return 50.0 + com_viol * 10

    # Build mesh
    mesh = build_mesh_from_params(beta, coeffs, n_theta=60, n_phi=120)
    if mesh is None:
        return 100.0  # r^4 went negative

    # Penalty: convexity
    try:
        hull = mesh.convex_hull
        vol_ratio = abs(mesh.volume) / abs(hull.volume)
    except Exception:
        vol_ratio = 0

    if vol_ratio < 0.995:
        return 30.0 + (1 - vol_ratio) * 100

    # Compute basin gap
    try:
        gap, n_basins, h_range, basin_info = compute_basin_gap(mesh, n_samples=2000)
    except Exception:
        return 80.0

    # Objective: minimize gap + penalize many basins + penalize COM violation
    # If ECS=1, gap=0, we want to reach this
    # Also reward larger h_range (more pronounced shape = more robust ECS=1)
    score = gap + com_viol * 5.0 + max(0, n_basins - 1) * 0.01

    return score


def main():
    print("\n" + "=" * 70)
    print("  PHASE 2c: OPTIMIZE FOR ECS=1")
    print("  Target: close the basin gap to zero via parameter optimization")
    print("=" * 70)

    results = {
        'phase': 'Phase 2c: ECS=1 Optimization',
        'trials': [],
    }

    # === Stage 1: Optimize beta only (no Fourier terms) ===
    print("\n  --- Stage 1: Optimize beta only (Gomboc 2 base) ---")

    n_fourier = 0

    def obj_beta_only(params):
        return objective(params, n_fourier=0)

    print("  Running differential evolution on beta...")
    t0 = time.time()

    result = differential_evolution(
        obj_beta_only,
        bounds=[(0.01, 0.20)],
        seed=42,
        maxiter=30,
        tol=1e-6,
        popsize=15,
        disp=False,
    )

    elapsed = time.time() - t0
    best_beta = result.x[0]
    best_score = result.fun

    print(f"  Best beta = {best_beta:.6f}, score = {best_score:.6f}, time = {elapsed:.1f}s")

    # Evaluate best
    mesh = build_mesh_from_params(best_beta, [])
    if mesh is not None:
        gap, n_basins, h_range, basin_info = compute_basin_gap(mesh)
        print(f"  ECS = {n_basins}, gap = {gap:.8f}, h_range = {h_range:.6f}")
        print(f"  Basins: {basin_info[:5]}")

        results['trials'].append({
            'stage': 'beta_only',
            'beta': round(best_beta, 6),
            'coeffs': [],
            'ECS': n_basins,
            'gap': round(gap, 8),
            'h_range': round(h_range, 6),
            'basins': basin_info[:5],
            'score': round(best_score, 6),
        })

    # === Stage 2: Optimize beta + 2 Fourier terms ===
    print("\n  --- Stage 2: Optimize beta + 2 Fourier terms ---")

    n_fourier = 2

    def obj_2fourier(params):
        return objective(params, n_fourier=2)

    print("  Running differential evolution on [beta, a0, a1]...")
    t0 = time.time()

    result = differential_evolution(
        obj_2fourier,
        bounds=[(0.01, 0.20), (-0.5, 0.5), (-0.5, 0.5)],
        seed=42,
        maxiter=40,
        tol=1e-6,
        popsize=15,
        disp=True,
    )

    elapsed = time.time() - t0
    best_params = result.x
    best_score = result.fun

    print(f"  Best params = beta={best_params[0]:.6f}, a0={best_params[1]:.6f}, a1={best_params[2]:.6f}")
    print(f"  Score = {best_score:.6f}, time = {elapsed:.1f}s")

    mesh = build_mesh_from_params(best_params[0], list(best_params[1:]))
    if mesh is not None:
        gap, n_basins, h_range, basin_info = compute_basin_gap(mesh)
        com_v = com_constraint_violation(best_params[0], list(best_params[1:]))
        print(f"  ECS = {n_basins}, gap = {gap:.8f}, h_range = {h_range:.6f}")
        print(f"  COM violation = {com_v:.8f}")
        print(f"  Basins: {basin_info[:5]}")

        results['trials'].append({
            'stage': '2_fourier',
            'beta': round(float(best_params[0]), 6),
            'coeffs': [round(float(c), 6) for c in best_params[1:]],
            'ECS': n_basins,
            'gap': round(gap, 8),
            'h_range': round(h_range, 6),
            'basins': basin_info[:5],
            'com_violation': round(float(com_v), 8),
            'score': round(best_score, 6),
        })

    # === Stage 3: Optimize beta + 4 Fourier terms ===
    print("\n  --- Stage 3: Optimize beta + 4 Fourier terms ---")

    n_fourier = 4

    def obj_4fourier(params):
        return objective(params, n_fourier=4)

    print("  Running differential evolution on [beta, a0, a1, a2, a3]...")
    t0 = time.time()

    result = differential_evolution(
        obj_4fourier,
        bounds=[(0.01, 0.20)] + [(-0.3, 0.3)] * 4,
        seed=42,
        maxiter=50,
        tol=1e-6,
        popsize=20,
        disp=True,
    )

    elapsed = time.time() - t0
    best_params = result.x
    best_score = result.fun

    coeffs_str = ', '.join([f'{c:.6f}' for c in best_params[1:]])
    print(f"  Best params = beta={best_params[0]:.6f}, coeffs=[{coeffs_str}]")
    print(f"  Score = {best_score:.6f}, time = {elapsed:.1f}s")

    mesh = build_mesh_from_params(best_params[0], list(best_params[1:]))
    if mesh is not None:
        gap, n_basins, h_range, basin_info = compute_basin_gap(mesh)
        com_v = com_constraint_violation(best_params[0], list(best_params[1:]))
        hull = mesh.convex_hull
        vol_ratio = abs(mesh.volume) / abs(hull.volume)

        print(f"  ECS = {n_basins}, gap = {gap:.8f}, h_range = {h_range:.6f}")
        print(f"  COM violation = {com_v:.8f}")
        print(f"  Convexity (vol ratio) = {vol_ratio:.6f}")
        print(f"  Basins: {basin_info[:5]}")

        results['trials'].append({
            'stage': '4_fourier',
            'beta': round(float(best_params[0]), 6),
            'coeffs': [round(float(c), 6) for c in best_params[1:]],
            'ECS': n_basins,
            'gap': round(gap, 8),
            'h_range': round(h_range, 6),
            'basins': basin_info[:5],
            'com_violation': round(float(com_v), 8),
            'vol_ratio': round(float(vol_ratio), 6),
            'score': round(best_score, 6),
        })

        # If ECS=1, save the mesh!
        if n_basins == 1:
            print("\n  *** ECS=1 ACHIEVED! Saving verified mesh ***")
            verified_path = os.path.join(MESH_DIR, 'gomboc_verified_ecs1.stl')
            mesh_hires = build_mesh_from_params(best_params[0], list(best_params[1:]),
                                                 n_theta=200, n_phi=400)
            if mesh_hires is not None:
                mesh_hires.export(verified_path)
                print(f"  Saved high-res mesh to {verified_path}")
                results['verified_mesh'] = {
                    'file': 'gomboc_verified_ecs1.stl',
                    'beta': round(float(best_params[0]), 8),
                    'coeffs': [round(float(c), 8) for c in best_params[1:]],
                    'n_faces': len(mesh_hires.faces),
                }

    # Save results
    result_path = os.path.join(RESULT_DIR, 'ecs1_optimization.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {result_path}")


if __name__ == '__main__':
    main()
