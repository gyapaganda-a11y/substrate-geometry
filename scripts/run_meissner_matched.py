"""
Run Meissner bodies at matched mesh resolution (~1200 faces) for
fair comparison against oloid (1198 faces).
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oracle_runner import run_primitive, MILD_STEEL
from contact_oracle import generate_oloid, generate_cylinder
import trimesh

def normalize_mesh(mesh, target_width=2.0):
    bounds = mesh.bounds
    max_width = (bounds[1] - bounds[0]).max()
    scale = target_width / max_width
    scaled = mesh.copy()
    scaled.apply_transform(np.diag([scale, scale, scale, 1.0]))
    scaled.vertices -= scaled.centroid
    return scaled

# Load originals
m1 = trimesh.load("meshes/Meissner_Type_1_-_Meissner.stl")
m2 = trimesh.load("meshes/Meissner_Type_2_-_Meissner.stl")

# Downsample to ~1200 faces (target_reduction is a ratio 0-1)
target = 1200
print("Downsampling meshes...")

m1_simple = m1.simplify_quadric_decimation(face_count=target)
print(f"  Type 1: {len(m1.faces)} -> {len(m1_simple.faces)} faces, "
      f"watertight={m1_simple.is_watertight}")

m2_simple = m2.simplify_quadric_decimation(face_count=target)
print(f"  Type 2: {len(m2.faces)} -> {len(m2_simple.faces)} faces, "
      f"watertight={m2_simple.is_watertight}")

# If not watertight, try convex hull as fallback
if not m1_simple.is_watertight:
    print(f"  Type 1: repairing via convex hull...")
    m1_simple = trimesh.convex.convex_hull(m1_simple.vertices)
    print(f"  Type 1 repaired: {len(m1_simple.faces)} faces, "
          f"watertight={m1_simple.is_watertight}")

if not m2_simple.is_watertight:
    print(f"  Type 2: repairing via convex hull...")
    m2_simple = trimesh.convex.convex_hull(m2_simple.vertices)
    print(f"  Type 2 repaired: {len(m2_simple.faces)} faces, "
          f"watertight={m2_simple.is_watertight}")

# Normalize to unit scale
m1_norm = normalize_mesh(m1_simple)
m2_norm = normalize_mesh(m2_simple)

oloid = generate_oloid(r=1.0, n_circle_pts=400)
cylinder = generate_cylinder(r=1.0, h=2.0)

print(f"\n  Resolution-matched meshes:")
print(f"    Oloid:         {len(oloid.faces)} faces, area={oloid.area:.4f}")
print(f"    Meissner T1:   {len(m1_norm.faces)} faces, area={m1_norm.area:.4f}")
print(f"    Meissner T2:   {len(m2_norm.faces)} faces, area={m2_norm.area:.4f}")
print(f"    Cylinder:      {len(cylinder.faces)} faces, area={cylinder.area:.4f}")
print()

# Run all three through identical rolling_plane mode
print("=" * 72)
print("RESOLUTION-MATCHED COMPARISON")
print(f"All meshes ~{target} faces, rolling_plane mode, identical params")
print("=" * 72)

p1 = run_primitive(
    mesh=m1_norm,
    invariant="contact_distribution",
    baseline_mesh=cylinder,
    baseline_name="Cylinder (conventional)",
    operating_context={"load": 100.0, "material": MILD_STEEL},
    name="Meissner Type 1 (matched)",
    mesh_source=f"Thingiverse STL, decimated to {len(m1_norm.faces)} faces",
)

p2 = run_primitive(
    mesh=m2_norm,
    invariant="contact_distribution",
    baseline_mesh=cylinder,
    baseline_name="Cylinder (conventional)",
    operating_context={"load": 100.0, "material": MILD_STEEL},
    name="Meissner Type 2 (matched)",
    mesh_source=f"Thingiverse STL, decimated to {len(m2_norm.faces)} faces",
)

p_oloid = run_primitive(
    mesh=oloid,
    invariant="contact_distribution",
    baseline_mesh=cylinder,
    baseline_name="Cylinder (conventional)",
    operating_context={"load": 100.0, "material": MILD_STEEL},
    name="Oloid (reference)",
)

# Comparison table
print("\n" + "=" * 90)
print("RESOLUTION-MATCHED RESULTS")
print("=" * 90)

dims = ["CDS", "SDS", "TDS", "FDS", "WDS_vol", "WDS_depth"]
print(f"\n  {'Dimension':<14} {'Oloid':<16} {'Meissner T1':<16} {'Meissner T2':<16}")
print(f"  {'-'*60}")

for dim in dims:
    vo = p_oloid.vector.get(dim)
    v1 = p1.vector.get(dim)
    v2 = p2.vector.get(dim)
    o_str = f"{vo.value:.2e}" if vo else "N/A"
    t1_str = f"{v1.value:.2e}" if v1 else "N/A"
    t2_str = f"{v2.value:.2e}" if v2 else "N/A"
    print(f"  {dim:<14} {o_str:<16} {t1_str:<16} {t2_str}")

print(f"\n  Face counts: Oloid={len(oloid.faces)}, "
      f"T1={len(m1_norm.faces)}, T2={len(m2_norm.faces)}")

# Ratios
if p_oloid.invariant_score > 0:
    r1 = p1.invariant_score / p_oloid.invariant_score
    r2 = p2.invariant_score / p_oloid.invariant_score
    print(f"\n  CDS ratios (vs Oloid):")
    print(f"    Meissner T1: {r1:.2f}x {'(better)' if r1 < 1 else '(worse)'}")
    print(f"    Meissner T2: {r2:.2f}x {'(better)' if r2 < 1 else '(worse)'}")

# FDS comparison (Type 1 vs Type 2)
f1 = p1.vector.get("FDS")
f2 = p2.vector.get("FDS")
if f1 and f2 and f1.value > 0:
    print(f"\n  FDS (fatigue) T1 vs T2: {f2.value/f1.value:.1f}x difference")
    print(f"    Edge rounding pattern effect on fatigue damage")

print(f"\n  RESOLUTION SENSITIVITY:")
print(f"    If Meissner scores are similar to oloid at matched resolution,")
print(f"    the 40x improvement at 11K faces was a discretization artifact.")
print(f"    If they're still significantly better, the geometry genuinely")
print(f"    outperforms the oloid on contact distribution.")
