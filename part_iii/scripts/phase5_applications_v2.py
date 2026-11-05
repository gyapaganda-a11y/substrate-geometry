"""
Phase 5 v2: Application-Specific Experiments (Fixed)
=====================================================
Uses consistent 5000-sample resolution and 1% merge threshold
matching the verified Phase 2c/Phase 4 results.

For seed pod: uses Phase 4 ECS results directly (already computed).
For IMU: computes orientation sensitivity analytically.
For buoy: reuses Phase 3 dynamics results.
"""

import numpy as np
import trimesh
import json
import os

MESH_DIR = os.path.join(os.path.dirname(__file__), '..', 'meshes')
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def main():
    print("\n" + "=" * 80)
    print("  PHASE 5 v2: APPLICATION-SPECIFIC RESULTS")
    print("  (Using verified Phase 2-4 data)")
    print("=" * 80)

    # Load Phase 4 density perturbation results (authoritative ECS values)
    with open(os.path.join(RESULT_DIR, 'density_perturbation.json')) as f:
        phase4 = json.load(f)

    # Load Phase 3 dynamics results
    with open(os.path.join(RESULT_DIR, 'dynamics.json')) as f:
        phase3 = json.load(f)

    results = {}

    # =====================================================================
    # APPLICATION 1: IMU CALIBRATION HOUSING
    # =====================================================================
    print("\n  === APPLICATION 1: IMU CALIBRATION HOUSING ===")
    print()

    # The Gomboc's precision is determined by manufacturing tolerance.
    # At 10 cm scale, shape tolerance is 0.01 mm -> ~0.006 deg angular precision.
    # The Gomboc ALWAYS returns to the same single orientation (ECS=1).
    # The question is: how precisely can that orientation be known?

    # From Phase 3: stable-unstable angle = 153.8 deg
    # The single stable equilibrium is well-separated from the unstable one.

    gomboc_dynamics = next(s for s in phase3['scores'] if 'ECS=1' in s['label'] or 'verified' in s['label'].lower())

    # Gomboc shape tolerance: 0.01 mm at 10 cm = 0.01% = 10^-4
    # Angular precision ~ arctan(tolerance / radius) ~ tolerance / radius (small angle)
    scale_cm = 10.0
    tolerance_mm = 0.01
    angular_precision_deg = np.degrees(tolerance_mm / (scale_cm * 10))  # convert to same units

    print(f"  Gomboc ECS = 1 (verified, Phase 2c)")
    print(f"  Stable-unstable angle = {gomboc_dynamics['stable_unstable_angle_deg']} deg")
    print(f"  BOA = {gomboc_dynamics['BOA_fraction']} (100% of orientations reach stable)")
    print()
    print(f"  Manufacturing precision at {scale_cm} cm scale:")
    print(f"    Shape tolerance: {tolerance_mm} mm")
    print(f"    Angular precision: {angular_precision_deg:.4f} deg ({angular_precision_deg*3600:.1f} arcsec)")
    print()
    print(f"  Meets precision requirements:")
    print(f"    Navigation grade (0.001 deg): {'YES' if angular_precision_deg < 0.001 else 'NO'}")
    print(f"    Tactical grade (0.01 deg):    {'YES' if angular_precision_deg < 0.01 else 'NO'}")
    print(f"    Industrial grade (0.1 deg):   {'YES' if angular_precision_deg < 0.1 else 'NO'}")
    print(f"    Consumer MEMS (1.0 deg):      {'YES' if angular_precision_deg < 1.0 else 'NO'}")
    print()
    print(f"  Comparison: current field calibration error = 2-3 deg (automotive)")
    print(f"  Improvement: {2.0 / angular_precision_deg:.0f}x over current practice")
    print()
    print(f"  Key advantage: passive, no power, no operator skill, repeatable.")
    print(f"  A Gomboc housing placed on any flat surface always presents the same")
    print(f"  orientation to {angular_precision_deg:.4f} deg — sufficient for tactical-grade")
    print(f"  IMU calibration without a precision turntable.")

    results['imu'] = {
        'application': 'IMU Calibration Housing',
        'ECS': 1,
        'BOA': gomboc_dynamics['BOA_fraction'],
        'stable_unstable_angle_deg': gomboc_dynamics['stable_unstable_angle_deg'],
        'scale_cm': scale_cm,
        'tolerance_mm': tolerance_mm,
        'angular_precision_deg': round(angular_precision_deg, 6),
        'angular_precision_arcsec': round(angular_precision_deg * 3600, 1),
        'meets_navigation': angular_precision_deg < 0.001,
        'meets_tactical': angular_precision_deg < 0.01,
        'meets_industrial': angular_precision_deg < 0.1,
        'meets_consumer': angular_precision_deg < 1.0,
        'improvement_over_field_cal': round(2.0 / angular_precision_deg),
        'note': 'Passive self-orienting IMU calibration fixture; see the paper for scope.',
    }

    # =====================================================================
    # APPLICATION 2: SEED POD POST-LANDING ORIENTATION
    # =====================================================================
    print("\n  === APPLICATION 2: SEED POD ORIENTATION ===")
    print()

    # From Phase 4: ECS values at 0% ballast
    seed_data = []
    for label in ['Gomboc', 'Sphere', 'Cylinder', 'Capsule', 'Cube']:
        if label in phase4:
            ecs_uniform = phase4[label]['sweep'][0]['ECS']
            boa_uniform = phase4[label]['sweep'][0]['BOA']

            # For ECS=1 bodies, 100% correct orientation
            # For ECS>1, correct orientation = BOA of dominant basin
            correct_frac = boa_uniform if ecs_uniform > 1 else 1.0
            if ecs_uniform == 1:
                correct_frac = 1.0

            # Published germination reduction: 20-67% for wrong orientation
            wrong_frac = 1 - correct_frac
            germ_loss_30 = wrong_frac * 0.30
            germ_loss_67 = wrong_frac * 0.67

            seed_data.append({
                'geometry': label,
                'ECS': ecs_uniform,
                'correct_orientation': round(correct_frac, 4),
                'wrong_orientation': round(wrong_frac, 4),
                'germination_loss_conservative': round(germ_loss_30, 4),
                'germination_loss_aggressive': round(germ_loss_67, 4),
            })

    print(f"  {'Geometry':12s}  {'ECS':>3s}  {'Correct':>8s}  {'Wrong':>6s}  {'Germ Loss (30%)':>15s}  {'Germ Loss (67%)':>15s}")
    print(f"  {'-'*12}  {'-'*3}  {'-'*8}  {'-'*6}  {'-'*15}  {'-'*15}")
    for d in seed_data:
        print(f"  {d['geometry']:12s}  {d['ECS']:3d}  {d['correct_orientation']:7.1%}  "
              f"{d['wrong_orientation']:5.1%}  {d['germination_loss_conservative']:14.1%}  "
              f"{d['germination_loss_aggressive']:14.1%}")

    print()
    print(f"  The Gomboc and Capsule (both ECS=1) achieve 100% correct orientation.")
    print(f"  The Cylinder (ECS=3) achieves 83.5% — 16.5% of landings are wrong.")
    print(f"  Applied to aerial reforestation: using a Gomboc-shaped pod instead")
    print(f"  of a cylinder eliminates 5-11% germination loss from orientation failure.")
    print()
    print(f"  Published baseline: orientation causes 20-67% germination reduction")
    print(f"  (Vishnu et al., 2023, JANS). The Gomboc eliminates this entirely.")
    print()
    print(f"  NOTE: The Gomboc orients AFTER landing, not during descent.")
    print(f"  Aerodynamic orientation during flight depends on ballistic coefficient,")
    print(f"  not mono-monostatic geometry. Post-landing self-righting on soil is")
    print(f"  the mechanism. Sufficient pod mass for gravitational dominance over")
    print(f"  aerodynamic torques during descent is an engineering design parameter.")

    results['seed_pod'] = seed_data

    # =====================================================================
    # APPLICATION 3: MARINE BUOY (flat-surface baseline)
    # =====================================================================
    print("\n  === APPLICATION 3: MARINE BUOY (flat-surface baseline) ===")
    print()

    buoy_data = []
    for s in phase3['scores']:
        label = s['label']
        if any(k in label for k in ['Gomboc', 'Cylinder', 'Sphere', 'Capsule']):
            buoy_data.append({
                'geometry': label,
                'ECS': s['ECS'],
                'BOA': s['BOA_fraction'],
                'SRE_mean': s['SRE_mean'],
                'h_range': s['h_range'],
                'steepness': s['steepness_mean'],
            })

    print(f"  {'Geometry':30s}  {'ECS':>3s}  {'BOA':>5s}  {'SRE':>8s}  {'h_range':>8s}  {'Steep':>7s}")
    print(f"  {'-'*30}  {'-'*3}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*7}")
    for d in buoy_data:
        print(f"  {d['geometry']:30s}  {d['ECS']:3d}  {d['BOA']:.3f}  "
              f"{d['SRE_mean']:8.5f}  {d['h_range']:8.5f}  {d['steepness']:7.5f}")

    print()
    print(f"  The Gomboc (ECS=1, BOA=1.000) provides guaranteed single-orientation")
    print(f"  self-righting on a flat surface. The Cylinder (ECS=3, BOA=0.835)")
    print(f"  has a 16.5% probability of settling in the wrong orientation.")
    print()
    print(f"  SCOPE: These results are computed on a flat rigid surface")
    print(f"  under uniform gravity. Ocean waves introduce buoyancy forces,")
    print(f"  fluid damping, and periodic perturbation that this oracle does")
    print(f"  not model. The flat-surface ECS=1 result is a necessary but")
    print(f"  possibly not sufficient condition for marine self-righting.")
    print(f"  Wave-perturbed dynamics require fluid-structure interaction")
    print(f"  simulation and are scoped as future work.")
    print()
    print(f"  Baseline: NOAA discus buoys capsize in seas >10m Hs and are")
    print(f"  NOT designed to self-right. RNLI lifeboats achieve self-righting")
    print(f"  in ~6 seconds via weighted hull + buoyant superstructure.")

    results['buoy'] = buoy_data

    # Save
    result_path = os.path.join(RESULT_DIR, 'applications.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {result_path}")


if __name__ == '__main__':
    main()
