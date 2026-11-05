"""
Generate figures for the Part III paper (arXiv:2604.17095)
=========================================================
Figure 1: Verified Gomboc mesh visualization (3 views)
Figure 2: COM height landscape on S2 showing single basin
Figure 3: Density perturbation comparison
"""

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
import os

MESH_DIR = os.path.join(os.path.dirname(__file__), '..', 'meshes')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'preprint')
os.makedirs(FIG_DIR, exist_ok=True)

BETA = 0.023149
A1 = 0.234433


def eta(theta):
    return (3 * np.pi / 2) * (np.cos(theta) - np.cos(theta)**3 / 3)


def fibonacci_sphere(n):
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    return np.stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)], axis=1)


# =====================================================================
# FIGURE 1: Verified Gomboc Mesh (3 views + sphere overlay)
# =====================================================================

def figure1_mesh_visualization():
    print("  Generating Figure 1: Mesh visualization...")

    # Generate the verified Gomboc surface
    nt, np_ = 100, 200
    theta = np.linspace(1e-6, np.pi - 1e-6, nt)
    phi = np.linspace(0, 2 * np.pi, np_, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    e = eta(T)
    phase = e + A1 * np.sin(e)
    r = (1 + 4 * BETA * np.sin(T) * np.cos(P - phase)) ** 0.25
    x = r * np.sin(T) * np.cos(P)
    y = r * np.sin(T) * np.sin(P)
    z = r * np.cos(T)

    # Also generate a unit sphere for comparison
    r_sphere = np.ones_like(T)
    xs = r_sphere * np.sin(T) * np.cos(P)
    ys = r_sphere * np.sin(T) * np.sin(P)
    zs = r_sphere * np.cos(T)

    # Deviation from sphere (for coloring)
    deviation = r - 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                              subplot_kw={'projection': '3d'})

    views = [
        (25, 45, 'Perspective view'),
        (0, 0, 'Front view (stable equilibrium face)'),
        (90, 0, 'Top view'),
    ]

    for ax, (elev, azim, title) in zip(axes, views):
        # Plot surface colored by deviation from sphere
        surf = ax.plot_surface(x, y, z, facecolors=cm.RdBu_r((deviation - deviation.min()) /
                               (deviation.max() - deviation.min() + 1e-10)),
                               alpha=0.85, shade=True, lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=45))

        # Wireframe of unit sphere for reference
        stride = 10
        ax.plot_wireframe(xs[::stride, ::stride], ys[::stride, ::stride],
                          zs[::stride, ::stride], color='gray', alpha=0.15, linewidth=0.3)

        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_zlim(-1.15, 1.15)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([1, 1, 1])

    plt.suptitle(r'Verified Mono-Monostatic Body ($\beta=0.023$, $a_1=0.234$, ECS$=1$)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig_gomboc_mesh.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIG_DIR, 'fig_gomboc_mesh.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("    Saved fig_gomboc_mesh.pdf/.png")


# =====================================================================
# FIGURE 2: COM Height Landscape on S2
# =====================================================================

def figure2_height_landscape():
    print("  Generating Figure 2: COM height landscape...")

    mesh = trimesh.load(os.path.join(MESH_DIR, 'gomboc_verified_ecs1_medres.stl'))

    # Sample directions on a regular theta-phi grid for plotting
    n_t, n_p = 180, 360
    theta_grid = np.linspace(0.01, np.pi - 0.01, n_t)
    phi_grid = np.linspace(0, 2 * np.pi, n_p, endpoint=False)
    TG, PG = np.meshgrid(theta_grid, phi_grid, indexing='ij')

    directions = np.stack([
        np.sin(TG.ravel()) * np.cos(PG.ravel()),
        np.sin(TG.ravel()) * np.sin(PG.ravel()),
        np.cos(TG.ravel()),
    ], axis=1)

    com = mesh.center_mass
    verts = mesh.vertices
    proj = directions @ verts.T
    heights = directions @ com - np.min(proj, axis=1)
    H = heights.reshape(n_t, n_p)

    # Find minimum (stable) and maximum (unstable)
    min_idx = np.unravel_index(np.argmin(H), H.shape)
    max_idx = np.unravel_index(np.argmax(H), H.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Mollweide projection of h(d)
    # Convert to longitude/latitude for Mollweide
    lat = np.degrees(np.pi/2 - TG)  # 90 at pole, -90 at other pole
    lon = np.degrees(PG - np.pi)     # -180 to 180

    ax1_m = fig.add_subplot(121, projection='mollweide')
    c = ax1_m.pcolormesh(np.radians(lon), np.radians(lat), H,
                          cmap='viridis', shading='auto')
    ax1_m.plot(np.radians(lon[min_idx]), np.radians(lat[min_idx]),
               'w*', markersize=15, markeredgecolor='black', markeredgewidth=1,
               label=f'Stable (h={H[min_idx]:.4f})')
    ax1_m.plot(np.radians(lon[max_idx]), np.radians(lat[max_idx]),
               'r^', markersize=12, markeredgecolor='black', markeredgewidth=1,
               label=f'Unstable (h={H[max_idx]:.4f})')
    ax1_m.set_title('COM Height Landscape $h(\\mathbf{d})$ on $S^2$',
                     fontsize=12, fontweight='bold')
    ax1_m.legend(loc='lower left', fontsize=9)
    ax1_m.grid(True, alpha=0.3)
    plt.colorbar(c, ax=ax1_m, shrink=0.6, label='COM height $h$')

    # Right: Height distribution histogram
    ax1.remove()
    ax2_h = fig.add_subplot(122)
    ax2_h.hist(H.ravel(), bins=80, color='steelblue', edgecolor='navy', alpha=0.8)
    ax2_h.axvline(H[min_idx], color='green', linewidth=2, linestyle='--',
                   label=f'Stable eq. (h={H[min_idx]:.4f})')
    ax2_h.axvline(H[max_idx], color='red', linewidth=2, linestyle='--',
                   label=f'Unstable eq. (h={H[max_idx]:.4f})')
    ax2_h.set_xlabel('COM height $h$', fontsize=11)
    ax2_h.set_ylabel('Count (directions on $S^2$)', fontsize=11)
    ax2_h.set_title('Height Distribution (single basin)', fontsize=12, fontweight='bold')
    ax2_h.legend(fontsize=9)

    h_range = H[max_idx] - H[min_idx]
    ax2_h.text(0.95, 0.95, f'$h$-range = {h_range:.4f}\nECS = 1\nBOA = 1.000',
               transform=ax2_h.transAxes, fontsize=10, verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig_height_landscape.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIG_DIR, 'fig_height_landscape.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("    Saved fig_height_landscape.pdf/.png")


# =====================================================================
# FIGURE 3: Density Perturbation Comparison
# =====================================================================

def figure3_density_perturbation():
    print("  Generating Figure 3: Density perturbation comparison...")

    # Load Phase 4 results
    import json
    result_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'density_perturbation.json')
    with open(result_path) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left panel: ECS vs ballast fraction
    geometries = ['Gomboc', 'Sphere', 'Cylinder', 'Capsule', 'Cube']
    colors = {'Gomboc': '#2ecc71', 'Sphere': '#3498db', 'Cylinder': '#e74c3c',
              'Capsule': '#9b59b6', 'Cube': '#f39c12'}
    markers = {'Gomboc': 'o', 'Sphere': 's', 'Cylinder': '^',
               'Capsule': 'D', 'Cube': 'p'}

    for geom in geometries:
        if geom not in data:
            continue
        sweep = data[geom]['sweep']
        ballast = [s['ballast_fraction'] * 100 for s in sweep]
        ecs = [s['ECS'] for s in sweep]
        ax1.plot(ballast, ecs, f'-{markers[geom]}', color=colors[geom],
                 label=geom, linewidth=2, markersize=7)

    ax1.axhline(y=1, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax1.text(25, 1.3, 'ECS = 1 (mono-monostatic)', color='green', fontsize=9, alpha=0.7)
    ax1.set_xlabel('Ballast fraction (%)', fontsize=11)
    ax1.set_ylabel('ECS (Equilibrium Count Score)', fontsize=11)
    ax1.set_title('ECS vs. Ballast: Geometry vs. Mass Distribution',
                   fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_ylim(0, 45)
    ax1.set_xlim(-1, 32)
    ax1.grid(True, alpha=0.3)

    # Right panel: BOA vs ballast fraction
    for geom in geometries:
        if geom not in data:
            continue
        sweep = data[geom]['sweep']
        ballast = [s['ballast_fraction'] * 100 for s in sweep]
        boa = [s['BOA'] for s in sweep]
        ax2.plot(ballast, boa, f'-{markers[geom]}', color=colors[geom],
                 label=geom, linewidth=2, markersize=7)

    ax2.axhline(y=1.0, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax2.text(15, 0.95, 'BOA = 1.0 (all orientations reach stable)', color='green',
             fontsize=9, alpha=0.7)
    ax2.set_xlabel('Ballast fraction (%)', fontsize=11)
    ax2.set_ylabel('BOA (Basin of Attraction fraction)', fontsize=11)
    ax2.set_title('Basin Coverage vs. Ballast',
                   fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_xlim(-1, 32)
    ax2.grid(True, alpha=0.3)

    # Add annotation for key finding
    fig.text(0.5, -0.02,
             'The cylinder and cube cannot achieve ECS=1 through ballast alone, even at 30%.\n'
             'The mono-monostatic invariant provides what mass distribution cannot.',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig_density_perturbation.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIG_DIR, 'fig_density_perturbation.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("    Saved fig_density_perturbation.pdf/.png")


def main():
    print("\n  Generating figures...")
    figure1_mesh_visualization()
    figure2_height_landscape()
    figure3_density_perturbation()
    print("\n  All figures saved to ../preprint/")


if __name__ == '__main__':
    main()
