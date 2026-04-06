"""
parametric_search.py
-----------------------------------------------------------------------------
Geometric Primitive Synthesis Program — Pipeline Artifact 02
Parametric Search over the Developable Roller Family

This is the first automated discovery run. The pipeline transitions from
library (cataloging known primitives) to discovery instrument (generating
and scoring shapes no human has individually tested).

SEARCH FAMILY:
  Developable Rollers — closed convex surfaces formed by the convex hull
  of N circles in various orientations. The oloid (Schatz, 1929) is the
  known fixed point at:
    theta  = 90°   (angle between generating circle planes)
    offset = r     (distance between circle centers)
    r1/r2  = 1.0   (ratio of generating circle radii)
    n_circles = 2  (number of generating circles)

  The search perturbs each parameter and scores every variant with the
  contact distribution oracle from contact_oracle.py.

WHAT THIS DOES:
  1. Defines a 4-parameter genome for the developable roller family
  2. Generates a grid of ~100 mesh variants by perturbing oloid parameters
  3. Scores each variant with run_oracle() from contact_oracle.py
  4. Ranks results by Contact Distribution Score (CDS)
  5. Outputs results to parametric_search_results.json
  6. Prints a ranked table + analysis

OUTCOMES THAT MATTER:
  - If nothing beats the oloid: Schatz found a local optimum by intuition.
    Publishable negative result — the oracle confirms his insight.
  - If something beats the oloid: first invariant primitive discovered by
    computation. That shape enters the Candidate Zone.
  - Either way: the pipeline works end-to-end. That's the methodology.

LINEAGE:
  Paul Schatz (1929) -> discovered the oloid by physical intuition
  This script -> searches the family his intuition found, computationally

DEPENDENCIES:
  pip install trimesh numpy scipy

USAGE:
  python parametric_search.py
  python parametric_search.py --quick     # fast run (fewer samples)
  python parametric_search.py --deep      # thorough run (more samples, more steps)

Author: Geometric Primitive Synthesis Program
-----------------------------------------------------------------------------
"""

import numpy as np
import trimesh
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

# Import the oracle from the same directory
from contact_oracle import run_oracle, generate_oloid, OracleResult


# -----------------------------------------------------------------------------
# SEARCH GENOME
# -----------------------------------------------------------------------------

@dataclass
class RollerGenome:
    """
    Parameterization of a generalized developable roller.

    The oloid is the fixed point:
      theta=90, offset=1.0, r_ratio=1.0, n_circles=2

    Each parameter has a physical meaning:
      theta     — angle in degrees between generating circle planes
      offset    — distance between circle centers, as fraction of r1
      r_ratio   — ratio r2/r1 (1.0 = equal circles)
      n_circles — number of generating circles (2 = oloid, 3+ = novel)
    """
    theta:     float = 90.0    # degrees
    offset:    float = 1.0     # as fraction of r1
    r_ratio:   float = 1.0     # r2/r1
    n_circles: int   = 2

    def label(self):
        return f"th={self.theta:.0f}° off={self.offset:.2f} r2/r1={self.r_ratio:.2f} n={self.n_circles}"

    def is_oloid(self):
        return (abs(self.theta - 90) < 1e-6 and
                abs(self.offset - 1.0) < 1e-6 and
                abs(self.r_ratio - 1.0) < 1e-6 and
                self.n_circles == 2)


@dataclass
class SearchResult:
    """Result for a single genome point in the search."""
    genome:             dict
    label:              str
    cds:                float       # Contact Distribution Score
    invariant_passed:   bool
    surface_area:       float
    n_faces:            int
    contact_cv:         float
    max_contact_ratio:  float
    is_oloid:           bool
    is_valid:           bool        # mesh generation succeeded
    error:              Optional[str] = None


# -----------------------------------------------------------------------------
# MESH GENERATION — GENERALIZED ROLLER
# -----------------------------------------------------------------------------

def generate_roller(genome: RollerGenome, r1: float = 1.0,
                    n_pts: int = 300) -> Optional[trimesh.Trimesh]:
    """
    Generate a generalized developable roller from a genome.

    For n_circles=2:
      Circle 1 lies in the XZ plane, centered at origin, radius r1.
      Circle 2 lies in a plane rotated theta degrees from XZ,
      centered at (offset * r1, 0, 0), radius r2 = r1 * r_ratio.

    For n_circles=3:
      Three circles spaced evenly at 120° azimuthal separation,
      each tilted theta degrees from the base plane, with the
      same offset and radius rules.

    Returns None if the convex hull fails (degenerate geometry).
    """
    theta_rad = np.radians(genome.theta)
    r2 = r1 * genome.r_ratio
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    all_points = []

    if genome.n_circles == 2:
        # Circle 1: XZ plane, centered at origin
        c1 = np.column_stack([
            r1 * np.cos(t),
            np.zeros(n_pts),
            r1 * np.sin(t)
        ])
        all_points.append(c1)

        # Circle 2: tilted by theta from XZ plane, centered at (offset*r1, 0, 0)
        # Rotation around Z axis by theta
        c2_local = np.column_stack([
            r2 * np.cos(t),
            np.zeros(n_pts),
            r2 * np.sin(t)
        ])
        # Rotate around Z axis by theta
        Rz = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad), 0],
            [np.sin(theta_rad),  np.cos(theta_rad), 0],
            [0,                  0,                  1]
        ])
        c2_rotated = (Rz @ c2_local.T).T
        c2_rotated[:, 0] += genome.offset * r1
        all_points.append(c2_rotated)

    elif genome.n_circles >= 3:
        # N circles evenly spaced azimuthally
        for i in range(genome.n_circles):
            azimuth = 2 * np.pi * i / genome.n_circles
            ri = r1 if i == 0 else r2

            # Circle in XZ plane
            ci_local = np.column_stack([
                ri * np.cos(t),
                np.zeros(n_pts),
                ri * np.sin(t)
            ])

            # Tilt by theta around the X axis
            Rx = np.array([
                [1, 0,                  0],
                [0, np.cos(theta_rad), -np.sin(theta_rad)],
                [0, np.sin(theta_rad),  np.cos(theta_rad)]
            ])
            ci_tilted = (Rx @ ci_local.T).T

            # Rotate around Z axis by azimuth
            Rz = np.array([
                [np.cos(azimuth), -np.sin(azimuth), 0],
                [np.sin(azimuth),  np.cos(azimuth), 0],
                [0,                0,                1]
            ])
            ci_final = (Rz @ ci_tilted.T).T
            ci_final[:, 0] += genome.offset * r1 * np.cos(azimuth)
            ci_final[:, 1] += genome.offset * r1 * np.sin(azimuth)
            all_points.append(ci_final)

    points = np.vstack(all_points)

    try:
        mesh = trimesh.convex.convex_hull(points)
        # Validate: must be watertight and have reasonable face count
        if not mesh.is_watertight or len(mesh.faces) < 20:
            return None
        return mesh
    except Exception:
        return None


# -----------------------------------------------------------------------------
# SEARCH GRID
# -----------------------------------------------------------------------------

def build_search_grid(mode: str = 'standard') -> List[RollerGenome]:
    """
    Build the parameter grid to search.

    The oloid (90°, 1.0, 1.0, 2) is always included as the anchor.

    Modes:
      quick    — ~30 samples, fast screening
      standard — ~90 samples, balanced exploration
      deep     — ~200+ samples, thorough sweep
    """
    genomes = []

    if mode == 'quick':
        thetas   = [60, 75, 90, 105, 120]
        offsets  = [0.7, 1.0, 1.3]
        r_ratios = [0.8, 1.0, 1.2]
        n_circs  = [2]
    elif mode == 'deep':
        thetas   = [45, 55, 65, 75, 80, 85, 90, 95, 100, 105, 115, 125, 135]
        offsets  = [0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
        r_ratios = [0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 2.0]
        n_circs  = [2, 3]
    else:  # standard
        thetas   = [55, 65, 75, 80, 85, 90, 95, 100, 105, 115, 125]
        offsets  = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
        r_ratios = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
        n_circs  = [2, 3]

    # Build grid for n=2 (theta × offset × r_ratio)
    for theta in thetas:
        for offset in offsets:
            for r_ratio in r_ratios:
                genomes.append(RollerGenome(theta=theta, offset=offset,
                                            r_ratio=r_ratio, n_circles=2))

    # Build grid for n=3+ (theta × offset, fixed r_ratio=1.0)
    if 3 in n_circs:
        for theta in thetas:
            for offset in offsets:
                genomes.append(RollerGenome(theta=theta, offset=offset,
                                            r_ratio=1.0, n_circles=3))

    # Ensure the oloid anchor is in the list
    oloid_genome = RollerGenome(theta=90.0, offset=1.0, r_ratio=1.0, n_circles=2)
    if not any(g.is_oloid() for g in genomes):
        genomes.insert(0, oloid_genome)

    # Deduplicate
    seen = set()
    unique = []
    for g in genomes:
        key = (g.theta, g.offset, g.r_ratio, g.n_circles)
        if key not in seen:
            seen.add(key)
            unique.append(g)

    return unique


# -----------------------------------------------------------------------------
# SEARCH RUNNER
# -----------------------------------------------------------------------------

def run_search(genomes: List[RollerGenome],
               oracle_steps: int = 300,
               r1: float = 1.0) -> List[SearchResult]:
    """
    Run the oracle on every genome in the list.

    Uses reduced oracle steps (300 vs 600) for search speed.
    Top candidates should be re-validated at 600+ steps.
    """
    results = []
    total = len(genomes)
    start = time.time()

    print(f"\n{'='*72}")
    print(f"PARAMETRIC SEARCH — Developable Roller Family")
    print(f"{'='*72}")
    print(f"Grid size: {total} genomes")
    print(f"Oracle steps per geometry: {oracle_steps}")
    print(f"Anchor: Oloid (Schatz 1929) at th=90° off=1.0 r2/r1=1.0 n=2")
    print(f"{'='*72}\n")

    for i, genome in enumerate(genomes):
        elapsed = time.time() - start
        eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
        print(f"[{i+1:>3}/{total}] {genome.label()}"
              f"  (ETA: {eta:.0f}s)", end='')

        # Generate mesh
        mesh = generate_roller(genome, r1=r1)
        if mesh is None:
            print(f"  — SKIP (degenerate hull)")
            results.append(SearchResult(
                genome=asdict(genome), label=genome.label(),
                cds=float('inf'), invariant_passed=False,
                surface_area=0, n_faces=0, contact_cv=0,
                max_contact_ratio=0, is_oloid=genome.is_oloid(),
                is_valid=False, error='degenerate convex hull'
            ))
            continue

        # Score with oracle (quiet mode)
        try:
            oracle_result = run_oracle(
                mesh, geometry_name=genome.label(),
                n_steps=oracle_steps,
                score_threshold=0.0001,
                verbose=False
            )
            results.append(SearchResult(
                genome=asdict(genome), label=genome.label(),
                cds=oracle_result.invariant_score,
                invariant_passed=oracle_result.invariant_satisfied,
                surface_area=oracle_result.surface_area,
                n_faces=oracle_result.n_faces,
                contact_cv=oracle_result.contact_cv,
                max_contact_ratio=oracle_result.max_contact_ratio,
                is_oloid=genome.is_oloid(),
                is_valid=True
            ))
            marker = "*" if oracle_result.invariant_score < 2e-6 else " "
            print(f"  CDS={oracle_result.invariant_score:.8f} {marker}")
        except Exception as e:
            print(f"  — ERROR: {e}")
            results.append(SearchResult(
                genome=asdict(genome), label=genome.label(),
                cds=float('inf'), invariant_passed=False,
                surface_area=0, n_faces=0, contact_cv=0,
                max_contact_ratio=0, is_oloid=genome.is_oloid(),
                is_valid=False, error=str(e)
            ))

    return results


# -----------------------------------------------------------------------------
# ANALYSIS
# -----------------------------------------------------------------------------

def analyze_results(results: List[SearchResult]):
    """Print ranked results and analysis."""
    valid = [r for r in results if r.is_valid]
    invalid = [r for r in results if not r.is_valid]

    # Sort by CDS (lower = better)
    valid.sort(key=lambda r: r.cds)

    # Find the oloid result
    oloid_result = next((r for r in valid if r.is_oloid), None)
    oloid_cds = oloid_result.cds if oloid_result else float('inf')
    oloid_rank = valid.index(oloid_result) + 1 if oloid_result else -1

    print(f"\n{'='*72}")
    print(f"SEARCH RESULTS — Ranked by Contact Distribution Score")
    print(f"{'='*72}")
    print(f"Valid geometries: {len(valid)} / {len(results)}")
    print(f"Degenerate/failed: {len(invalid)}")
    if oloid_result:
        print(f"Oloid CDS (anchor): {oloid_cds:.8f}")
        print(f"Oloid rank: #{oloid_rank} of {len(valid)}")
    print(f"{'='*72}\n")

    # Top 20 table
    print(f"{'Rank':<6} {'CDS':<16} {'vs Oloid':<12} {'th':<7} {'off':<7} "
          f"{'r2/r1':<8} {'n':<4} {'Area':<8} {'Pass':<6}")
    print("-" * 76)

    for i, r in enumerate(valid[:20]):
        g = r.genome
        ratio = r.cds / oloid_cds if oloid_cds > 0 else float('inf')
        marker = " < OLOID" if r.is_oloid else ""
        better = "* BETTER" if r.cds < oloid_cds and not r.is_oloid else ""
        pass_str = "[OK]" if r.invariant_passed else "[FAIL]"
        print(f"{i+1:<6} {r.cds:<16.8f} {ratio:<12.2f}× "
              f"{g['theta']:<7.0f} {g['offset']:<7.2f} "
              f"{g['r_ratio']:<8.2f} {g['n_circles']:<4} "
              f"{r.surface_area:<8.2f} {pass_str:<6}{marker}{better}")

    # Count how many beat the oloid
    better_than_oloid = [r for r in valid if r.cds < oloid_cds and not r.is_oloid]
    worse_than_oloid = [r for r in valid if r.cds > oloid_cds and not r.is_oloid]

    print(f"\n{'-'*72}")
    print(f"ANALYSIS")
    print(f"{'-'*72}")
    print(f"Geometries scoring BETTER than oloid: {len(better_than_oloid)}")
    print(f"Geometries scoring WORSE than oloid:  {len(worse_than_oloid)}")

    if better_than_oloid:
        best = better_than_oloid[0]
        improvement = (1 - best.cds / oloid_cds) * 100
        print(f"\n* BEST DISCOVERY: {best.label}")
        print(f"  CDS = {best.cds:.8f} ({improvement:.1f}% improvement over oloid)")
        print(f"  Surface area = {best.surface_area:.4f}")
        print(f"  This geometry should be re-validated at 600+ oracle steps")
        print(f"  and formally entered into the Candidate Zone if confirmed.")
    else:
        print(f"\nThe oloid is a LOCAL OPTIMUM in the developable roller family.")
        print(f"Schatz (1929) found the best contact-distributing roller in this")
        print(f"search family by physical intuition alone. The oracle confirms it.")
        print(f"This is a publishable negative result — the invariant landscape")
        print(f"around the oloid is a basin with the oloid at its minimum.")

    # Parameter sensitivity analysis
    print(f"\n{'-'*72}")
    print(f"PARAMETER SENSITIVITY (at n_circles=2)")
    print(f"{'-'*72}")

    # Theta sensitivity (fix offset=1.0, r_ratio=1.0)
    theta_results = [r for r in valid
                     if r.genome['n_circles'] == 2
                     and abs(r.genome['offset'] - 1.0) < 0.01
                     and abs(r.genome['r_ratio'] - 1.0) < 0.01]
    theta_results.sort(key=lambda r: r.genome['theta'])
    if theta_results:
        print(f"\nTheta sweep (offset=1.0, r2/r1=1.0):")
        for r in theta_results:
            bar = "#" * int(min(50, max(1, 50 * r.cds / (oloid_cds * 10))))
            print(f"  th={r.genome['theta']:>6.0f}°  CDS={r.cds:.8f}  {bar}")

    # Offset sensitivity (fix theta=90, r_ratio=1.0)
    offset_results = [r for r in valid
                      if r.genome['n_circles'] == 2
                      and abs(r.genome['theta'] - 90) < 0.01
                      and abs(r.genome['r_ratio'] - 1.0) < 0.01]
    offset_results.sort(key=lambda r: r.genome['offset'])
    if offset_results:
        print(f"\nOffset sweep (th=90°, r2/r1=1.0):")
        for r in offset_results:
            bar = "#" * int(min(50, max(1, 50 * r.cds / (oloid_cds * 10))))
            print(f"  off={r.genome['offset']:>5.2f}  CDS={r.cds:.8f}  {bar}")

    # r_ratio sensitivity (fix theta=90, offset=1.0)
    ratio_results = [r for r in valid
                     if r.genome['n_circles'] == 2
                     and abs(r.genome['theta'] - 90) < 0.01
                     and abs(r.genome['offset'] - 1.0) < 0.01]
    ratio_results.sort(key=lambda r: r.genome['r_ratio'])
    if ratio_results:
        print(f"\nRadius ratio sweep (th=90°, offset=1.0):")
        for r in ratio_results:
            bar = "#" * int(min(50, max(1, 50 * r.cds / (oloid_cds * 10))))
            print(f"  r2/r1={r.genome['r_ratio']:>5.2f}  CDS={r.cds:.8f}  {bar}")

    # 3-circle results
    tri_results = [r for r in valid if r.genome['n_circles'] == 3]
    if tri_results:
        tri_results.sort(key=lambda r: r.cds)
        print(f"\nBest 3-circle rollers:")
        for r in tri_results[:5]:
            ratio = r.cds / oloid_cds if oloid_cds > 0 else 0
            print(f"  {r.label}  CDS={r.cds:.8f}  ({ratio:.2f}× oloid)")

    return valid, better_than_oloid


# -----------------------------------------------------------------------------
# REVALIDATION
# -----------------------------------------------------------------------------

def revalidate_top(results: List[SearchResult], top_n: int = 5,
                   oracle_steps: int = 600) -> List[SearchResult]:
    """
    Re-run the oracle at full resolution on the top N candidates.
    This confirms whether search-speed scores hold at validation depth.
    """
    valid = [r for r in results if r.is_valid]
    valid.sort(key=lambda r: r.cds)
    top = valid[:top_n]

    print(f"\n{'='*72}")
    print(f"REVALIDATION — Top {top_n} at {oracle_steps} steps")
    print(f"{'='*72}\n")

    revalidated = []
    for r in top:
        genome = RollerGenome(**r.genome)
        mesh = generate_roller(genome)
        if mesh is None:
            print(f"  {r.label} — mesh generation failed on revalidation")
            continue

        oracle_result = run_oracle(
            mesh, geometry_name=r.label,
            n_steps=oracle_steps,
            score_threshold=0.0001,
            verbose=True
        )
        revalidated.append(SearchResult(
            genome=r.genome, label=r.label,
            cds=oracle_result.invariant_score,
            invariant_passed=oracle_result.invariant_satisfied,
            surface_area=oracle_result.surface_area,
            n_faces=oracle_result.n_faces,
            contact_cv=oracle_result.contact_cv,
            max_contact_ratio=oracle_result.max_contact_ratio,
            is_oloid=genome.is_oloid(),
            is_valid=True
        ))
        print(f"  Search CDS: {r.cds:.8f} -> Validated CDS: {oracle_result.invariant_score:.8f}\n")

    return revalidated


# -----------------------------------------------------------------------------
# JSON OUTPUT
# -----------------------------------------------------------------------------

def save_results(results: List[SearchResult], revalidated: List[SearchResult],
                 output_path: str = 'parametric_search_results.json'):
    """Save results to JSON for dashboard consumption."""
    # Convert inf/nan to null for JSON
    def clean(val):
        if isinstance(val, float) and (np.isinf(val) or np.isnan(val)):
            return None
        return val

    data = {
        'metadata': {
            'program': 'Geometric Primitive Synthesis Program',
            'artifact': 'Parametric Search — Developable Roller Family',
            'search_family': 'developable_roller',
            'anchor': 'Oloid (Schatz, 1929)',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'total_genomes': len(results),
            'valid_genomes': len([r for r in results if r.is_valid]),
        },
        'search_results': [
            {
                'genome': r.genome,
                'label': r.label,
                'cds': clean(r.cds),
                'invariant_passed': r.invariant_passed,
                'surface_area': clean(r.surface_area),
                'n_faces': r.n_faces,
                'contact_cv': clean(r.contact_cv),
                'max_contact_ratio': clean(r.max_contact_ratio),
                'is_oloid': r.is_oloid,
                'is_valid': r.is_valid,
                'error': r.error,
            }
            for r in sorted(results, key=lambda r: r.cds if r.is_valid else float('inf'))
        ],
        'revalidated': [
            {
                'genome': r.genome,
                'label': r.label,
                'cds': clean(r.cds),
                'invariant_passed': r.invariant_passed,
                'surface_area': clean(r.surface_area),
                'contact_cv': clean(r.contact_cv),
                'max_contact_ratio': clean(r.max_contact_ratio),
                'is_oloid': r.is_oloid,
            }
            for r in revalidated
        ]
    }

    path = Path(output_path)
    path.write_text(json.dumps(data, indent=2))
    print(f"\nResults saved to {path.absolute()}")
    return data


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Parse mode from command line
    mode = 'standard'
    if '--quick' in sys.argv:
        mode = 'quick'
    elif '--deep' in sys.argv:
        mode = 'deep'

    oracle_steps_search = 200 if mode == 'quick' else 300
    oracle_steps_validate = 600

    print("=" * 72)
    print("GEOMETRIC PRIMITIVE SYNTHESIS PROGRAM")
    print("Parametric Search — Developable Roller Family")
    print("=" * 72)
    print(f"\nMode: {mode}")
    print(f"Search oracle steps: {oracle_steps_search}")
    print(f"Revalidation oracle steps: {oracle_steps_validate}")
    print()
    print("Lineage: Paul Schatz (1929) discovered the oloid by sawing a cube.")
    print("This search explores the family his intuition found, computationally.")
    print("If anything scores better, it is a novel invariant primitive.")
    print("If nothing does, Schatz found the optimum 97 years ago by hand.")
    print()

    # Build grid
    genomes = build_search_grid(mode=mode)
    print(f"Search grid: {len(genomes)} genomes")

    # Run search
    start = time.time()
    results = run_search(genomes, oracle_steps=oracle_steps_search)
    search_time = time.time() - start
    print(f"\nSearch completed in {search_time:.1f}s "
          f"({search_time/len(genomes):.1f}s per genome)")

    # Analyze
    valid, better = analyze_results(results)

    # Revalidate top 5 at full resolution
    revalidated = revalidate_top(results, top_n=5,
                                 oracle_steps=oracle_steps_validate)

    # Save
    output_path = str(Path(__file__).parent / 'parametric_search_results.json')
    save_results(results, revalidated, output_path=output_path)

    # Final summary
    print(f"\n{'='*72}")
    print(f"PIPELINE STATUS")
    print(f"{'='*72}")
    print(f"""
  [OK] Parametric search: COMPLETE
  [OK] Genomes evaluated: {len(genomes)}
  [OK] Valid geometries: {len(valid)}
  [OK] Better than oloid: {len(better)}
  [OK] Results saved: parametric_search_results.json
  [OK] Top 5 revalidated at {oracle_steps_validate} steps

  -> Next: load results into Oracle Console on Synthesis Engine dashboard
  -> Next: if novel primitives found, enter into Candidate Zone
  -> Next: FEniCS Hertz contact pressure oracle (structural validation)
  -> Next: DEAP evolutionary search (genome -> mesh -> oracle -> fitness)
""")
