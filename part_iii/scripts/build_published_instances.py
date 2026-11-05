"""
Rebuild the published bodies from their parameters (no optimization).

Substrate Geometry Part III (arXiv:2604.17095).

The bodies are r^4 = 1 + 4 beta sin(theta) cos(phi - P(theta)) with the Sloan phase
function eta(theta) = (3 pi / 2)(cos theta - cos^3 theta / 3) extended by one Fourier term,
P = eta + a_k sin(k eta). The verification in the paper applies to these published
parameters. The primary body is built exactly as in phase7_resolution_sensitivity.py
(P = eta + a1 sin(eta)); the stored verified meshes match it to float32 precision.
The differential-evolution script (phase2c_optimize_ecs1.py) is exploratory: it does not
regenerate the stored optimization results (see the README), so the downstream scripts
should be run on meshes built here.

The second and third instances are built with phase5c_additions.py (its coeffs[k]
multiplies sin((k+1) * eta), so [0, a2] is a2 sin(2 eta)).

Instances (Table 4 of the paper; only the primary instance is strictly convex):
  primary : beta = 0.023149, a1 = 0.234433, a1 sin(eta)
  second  : beta = 0.0321,   a2 = 0.137606, a2 sin(2 eta)
  third   : beta = 0.051692, a3 = -0.055157, a3 sin(3 eta)

Writes into ../meshes:
  gomboc_verified_ecs1_medres.stl   primary, 80 x 160
  gomboc_verified_ecs1.stl          primary, 200 x 400
  gomboc_instance_sin2eta.stl       second, 150 x 300
  gomboc_instance_sin3eta.stl       third,  150 x 300
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MESH_DIR = os.path.join(HERE, '..', 'meshes')
os.makedirs(MESH_DIR, exist_ok=True)

import phase5c_additions as p5c
import phase7_resolution_sensitivity as p7


def main():
    jobs = [
        ('gomboc_verified_ecs1_medres.stl', p7.generate_verified_gomboc(80, 160)),
        ('gomboc_verified_ecs1.stl', p7.generate_verified_gomboc(200, 400)),
        ('gomboc_instance_sin2eta.stl', p5c.build_gomboc(0.0321, [0.0, 0.137606], 150, 300)),
        ('gomboc_instance_sin3eta.stl', p5c.build_gomboc(0.051692, [0.0, 0.0, -0.055157], 150, 300)),
    ]
    for name, mesh in jobs:
        if mesh is None:
            raise SystemExit(f"could not build {name}: radius became non-positive")
        mesh.export(os.path.join(MESH_DIR, name))
        print(f"  wrote {name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")


if __name__ == '__main__':
    main()
