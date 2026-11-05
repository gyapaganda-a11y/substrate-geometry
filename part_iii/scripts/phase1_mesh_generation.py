"""
Phase 1: Gömböc Mesh Generation from Sloan 2023 Analytical Equations
=====================================================================
Source: M. L. Sloan, "An Analytical Gomboc," arXiv:2306.14914 (2023)

Generates Gömböc meshes at multiple resolutions from the analytical
parameterization. Validates watertight, convexity, and center of mass.

Outputs:
  - ../meshes/gomboc1_beta{b}_res{n}.stl
  - ../meshes/gomboc2_beta{b}_res{n}.stl
  - ../meshes/baselines/*.stl
  - ../results/mesh_generation.json
"""

import numpy as np
import trimesh
import json
import os
import sys

MESH_DIR = os.path.join(os.path.dirname(__file__), '..', 'meshes')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(MESH_DIR, exist_ok=True)
os.makedirs(os.path.join(MESH_DIR, 'baselines'), exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def gomboc_surface(n_theta, n_phi, beta, variant):
    """Generate Gömböc surface points from Sloan 2023 Eq. 12 or 16.

    Args:
        n_theta: number of latitude divisions
        n_phi: number of longitude divisions
        beta: perturbation parameter (<=0.15 for variant 1, <=0.17 for variant 2)
        variant: 1 (helical, Eq. 12) or 2 (single revolution, Eq. 16)

    Returns:
        vertices: (n_theta * n_phi, 3) array of Cartesian coordinates
        faces: (M, 3) array of triangle indices
    """
    theta = np.linspace(1e-6, np.pi - 1e-6, n_theta)  # avoid exact poles
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing='ij')

    if variant == 1:
        phase = 5 * T
    elif variant == 2:
        phase = (3 * np.pi / 2) * (np.cos(T) - np.cos(T)**3 / 3)
    else:
        raise ValueError("variant must be 1 or 2")

    r4 = 1 + 4 * beta * np.sin(T) * np.cos(P - phase)

    if np.any(r4 <= 0):
        raise ValueError(f"r^4 <= 0 detected. beta={beta} is too large for variant {variant}")

    r = r4 ** 0.25

    x = r * np.sin(T) * np.cos(P)
    y = r * np.sin(T) * np.sin(P)
    z = r * np.cos(T)

    # Build vertex array
    vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    # Add pole vertices
    north_pole = np.array([[0, 0, 1.0]])  # r=1 at theta=0
    south_pole = np.array([[0, 0, -1.0]])  # r=1 at theta=pi
    north_idx = len(vertices)
    south_idx = north_idx + 1
    vertices = np.vstack([vertices, north_pole, south_pole])

    # Build faces from quad grid
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

    # North pole cap (connect first ring to north pole)
    for j in range(n_phi):
        j_next = (j + 1) % n_phi
        faces.append([north_idx, j, j_next])

    # South pole cap (connect last ring to south pole)
    last_ring_start = (n_theta - 1) * n_phi
    for j in range(n_phi):
        j_next = (j + 1) % n_phi
        faces.append([south_idx, last_ring_start + j_next, last_ring_start + j])

    faces = np.array(faces)
    return vertices, faces


def generate_gomboc(n_theta, n_phi, beta, variant, label):
    """Generate and save a Gömböc mesh."""
    vertices, faces = gomboc_surface(n_theta, n_phi, beta, variant)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()

    filename = f"gomboc{variant}_beta{beta}_res{n_theta}x{n_phi}.stl"
    filepath = os.path.join(MESH_DIR, filename)
    mesh.export(filepath)

    info = {
        'label': label,
        'variant': variant,
        'beta': beta,
        'resolution': f'{n_theta}x{n_phi}',
        'n_faces': len(mesh.faces),
        'n_vertices': len(mesh.vertices),
        'watertight': mesh.is_watertight,
        'volume': float(mesh.volume),
        'center_of_mass': mesh.center_mass.tolist(),
        'com_offset_from_origin': float(np.linalg.norm(mesh.center_mass)),
        'bounding_box': mesh.bounding_box.extents.tolist(),
        'is_convex': mesh.is_convex,
        'filename': filename,
    }

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for k, v in info.items():
        if k == 'center_of_mass':
            print(f"  {k}: [{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}]")
        elif isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    return info


def generate_baseline(name, mesh, target_volume=None):
    """Save a baseline mesh and return its info dict."""
    if target_volume is not None:
        current_vol = abs(mesh.volume)
        if current_vol > 0:
            scale = (target_volume / current_vol) ** (1/3)
            mesh.apply_scale(scale)

    mesh.fix_normals()
    filename = f"{name}.stl"
    filepath = os.path.join(MESH_DIR, 'baselines', filename)
    mesh.export(filepath)

    info = {
        'label': name,
        'n_faces': len(mesh.faces),
        'n_vertices': len(mesh.vertices),
        'watertight': mesh.is_watertight,
        'volume': float(mesh.volume),
        'center_of_mass': mesh.center_mass.tolist(),
        'bounding_box': mesh.bounding_box.extents.tolist(),
        'is_convex': mesh.is_convex,
        'filename': f'baselines/{filename}',
    }

    print(f"\n  {name}: faces={info['n_faces']}, watertight={info['watertight']}, "
          f"volume={info['volume']:.4f}, convex={info['is_convex']}")

    return info


def main():
    results = {
        'phase': 'Phase 1: Mesh Generation',
        'source': 'Sloan 2023, arXiv:2306.14914',
        'gomboc_meshes': [],
        'baseline_meshes': [],
    }

    # === Gömböc meshes at multiple resolutions ===
    print("\n" + "="*60)
    print("  GÖMBÖC MESH GENERATION")
    print("="*60)

    configs = [
        # (n_theta, n_phi, beta, variant, label)
        (50, 100, 0.15, 1, "Gömböc 1 — Low res"),
        (100, 200, 0.15, 1, "Gömböc 1 — Medium res"),
        (200, 400, 0.15, 1, "Gömböc 1 — High res"),
        (50, 100, 0.17, 2, "Gömböc 2 — Low res"),
        (100, 200, 0.17, 2, "Gömböc 2 — Medium res"),
        (200, 400, 0.17, 2, "Gömböc 2 — High res"),
    ]

    for n_theta, n_phi, beta, variant, label in configs:
        info = generate_gomboc(n_theta, n_phi, beta, variant, label)
        results['gomboc_meshes'].append(info)

    # Get reference volume from high-res Gömböc 2 for baseline scaling
    ref_volume = results['gomboc_meshes'][-1]['volume']  # high-res Gömböc 2
    print(f"\n  Reference volume (Gömböc 2 high-res): {ref_volume:.6f}")

    # === Baseline geometries ===
    print("\n" + "="*60)
    print("  BASELINE MESH GENERATION")
    print("="*60)

    # Sphere
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    results['baseline_meshes'].append(
        generate_baseline('sphere', sphere, target_volume=ref_volume))

    # Cylinder (L/D = 2)
    cylinder = trimesh.creation.cylinder(radius=0.5, height=2.0, sections=64)
    results['baseline_meshes'].append(
        generate_baseline('cylinder_LD2', cylinder, target_volume=ref_volume))

    # Hemisphere (dome — keep only z>=0 vertices of sphere, add flat base)
    hemisphere_sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    # Keep faces where all vertices have z >= -0.01 (upper hemisphere)
    mask = np.all(hemisphere_sphere.vertices[hemisphere_sphere.faces][:, :, 2] >= -0.01, axis=1)
    hemisphere = hemisphere_sphere.submesh([mask], append=True)
    # Fill the base with a simple fan
    equator_verts = hemisphere.vertices[np.abs(hemisphere.vertices[:, 2]) < 0.05]
    # Use convex hull to get a watertight dome
    dome_verts = hemisphere.vertices[hemisphere.vertices[:, 2] >= -0.01]
    # Add center bottom vertex and build dome via convex hull
    dome_verts = np.vstack([dome_verts, [[0, 0, 0]]])
    hemisphere = trimesh.convex.convex_hull(dome_verts)
    results['baseline_meshes'].append(
        generate_baseline('hemisphere', hemisphere, target_volume=ref_volume))

    # Ellipsoid (3 distinct axes: a=1.5, b=1.0, c=0.7)
    ellipsoid = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    ellipsoid.apply_transform(np.diag([1.5, 1.0, 0.7, 1.0]))
    results['baseline_meshes'].append(
        generate_baseline('ellipsoid', ellipsoid, target_volume=ref_volume))

    # Cube
    cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    results['baseline_meshes'].append(
        generate_baseline('cube', cube, target_volume=ref_volume))

    # Capsule (pharmaceutical — cylinder with hemispherical caps, L/D ≈ 3)
    capsule = trimesh.creation.capsule(height=1.5, radius=0.3, count=[32, 32])
    results['baseline_meshes'].append(
        generate_baseline('capsule', capsule, target_volume=ref_volume))

    # === Summary ===
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    n_gomboc_watertight = sum(1 for m in results['gomboc_meshes'] if m['watertight'])
    n_gomboc_convex = sum(1 for m in results['gomboc_meshes'] if m['is_convex'])
    n_baseline_watertight = sum(1 for m in results['baseline_meshes'] if m['watertight'])

    print(f"  Gömböc meshes: {len(results['gomboc_meshes'])} generated, "
          f"{n_gomboc_watertight} watertight, {n_gomboc_convex} convex")
    print(f"  Baseline meshes: {len(results['baseline_meshes'])} generated, "
          f"{n_baseline_watertight} watertight")

    # Check COM offset (should be near zero for Gömböc)
    for m in results['gomboc_meshes']:
        offset = m['com_offset_from_origin']
        status = "OK" if offset < 0.01 else "WARNING"
        print(f"  {m['label']}: COM offset = {offset:.6f} [{status}]")

    # Save results
    result_path = os.path.join(RESULT_DIR, 'mesh_generation.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {result_path}")


if __name__ == '__main__':
    main()
