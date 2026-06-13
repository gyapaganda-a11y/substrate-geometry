"""Automated tests for the substrate-geometry oracle framework.
Exercise the public API on analytically-known cases. Run: `pytest` from repo root."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import trimesh
import pytest


def test_imports():
    import oracle_runner
    from invariants import registry as inv_registry
    from modes import registry as mode_registry
    assert hasattr(oracle_runner, "run_primitive")


def test_invariant_registry_populated():
    from invariants.registry import INVARIANT_REGISTRY
    assert len(INVARIANT_REGISTRY) >= 5
    assert "contact_distribution" in INVARIANT_REGISTRY
    assert "mono_monostatic" in INVARIANT_REGISTRY


def test_gauss_bonnet_sphere():
    """Total angle defect of any closed genus-0 mesh equals 4*pi (discrete Gauss-Bonnet)."""
    m = trimesh.creation.icosphere(subdivisions=3)
    total_defect = float(np.sum(m.vertex_defects))
    assert math.isclose(total_defect, 4 * math.pi, rel_tol=1e-6)


def test_equilibrium_oracle_runs():
    """The equilibrium (mono_monostatic) oracle returns a finite, non-negative score on a box."""
    from oracle_runner import run_primitive
    box = trimesh.creation.box(extents=(1.0, 1.2, 1.5))
    base = trimesh.creation.icosphere(subdivisions=2)
    prof = run_primitive(mesh=box, invariant="mono_monostatic",
                         baseline_mesh=base, name="Box", baseline_name="Sphere")
    assert prof.invariant_score is not None
    assert float(prof.invariant_score) >= 0.0


def test_contact_distribution_vector():
    """The rolling oracle produces a positive contact-distribution score and a physics vector."""
    from oracle_runner import run_primitive
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    base = trimesh.creation.icosphere(subdivisions=2)
    prof = run_primitive(mesh=box, invariant="contact_distribution",
                         baseline_mesh=base, name="Cube", baseline_name="Sphere")
    cds = prof.vector.get("CDS")
    assert cds is not None and float(cds.value) > 0.0
